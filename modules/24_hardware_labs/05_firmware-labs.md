# Firmware Labs

> Module 24 — Hardware and Robotics Labs
> Topic 5 of 9
> Estimated study time: 5 hours
> Prerequisites: 01, 02, 03, 14A
> Version 2026.3

## 1. Why this topic exists

Firmware is where the physical and the logical meet. A sensor sample is taken at a moment, a control output is written at a moment, and a fault is detected at a moment. The moments are the product. A firmware that is correct in the average and wrong in the tail is a firmware that fails at the worst possible time.

The industry problem is that firmware defects are invisible in a functional test. A device that samples at 500 Hz on average can miss a deadline by 20 ms under load. A watchdog that resets a hung task can be fed by the same task and never fire. A secure boot that verifies an image can leave the debug port open. An update that works on the bench can brick a device when the power fails at the wrong moment.

Three decisions depend on this topic:

1. **Whether the timing requirement is met in the tail, not the mean.** A sampling loop that averages 2 ms can have a 99th-percentile interval of 2.4 ms. The tail is what a control loop and a safety monitor see.
2. **Whether the watchdog is independent.** A watchdog fed by the task it supervises is a timer, not a control. The feed path, the timeout, the reset behaviour, and the safe state must each be verified.
3. **Whether the update path is recoverable.** A signed image, a rollback slot, an anti-rollback counter, and a power-loss test are the minimum. A secure boot checkbox without a rollback test is a claim, not a control.

The commercial stakes are concrete. A firmware defect that escapes to the field can require a recall, a field correction, or a cybersecurity advisory. A watchdog that does not fire can leave an actuator energised. An update that bricks a device can take a fleet offline.

The stakes are also regulatory. A firmware change can invalidate a verification result, a cybersecurity assessment, or a marketing authorisation if it is not assessed. The firmware lab is therefore a lesson in change control as much as in coding.

This topic covers the firmware lab: a real-time operating system (RTOS) with three tasks, a watchdog with a health monitor, a fail-safe state, a signed update with rollback, and a fault-injection matrix.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work.

## 2. Learning outcomes

By the end of this topic you can:

- Measure sampling jitter on a target with a logic analyser and report the distribution, not only the mean.
- Compute the worst-case response time of a fixed-priority task set and compare it with the deadline.
- Demonstrate priority inversion and show the effect of priority inheritance with measured latencies.
- Design a watchdog feed path that is independent of the tasks it supervises, and verify that it resets a hung task.
- Define a fail-safe state for every output and verify it with an instrument, not by inspection.
- Sign a firmware image, reject an unsigned or corrupted image, and demonstrate rollback after a failed update.
- Build a fault-injection matrix with detection mechanism, detection time, recovery action, and residual state.
- Measure boot time, reset latency, update time, and current consumption in run, idle, and fault states.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

Firmware is a set of concurrent activities that share a processor, a memory, and a set of peripherals. The behaviour that matters is not the average behaviour but the behaviour at the deadlines.

### 3.1 The timing model

```text
  event --> task ready --> scheduler --> execution --> output
     |           |             |             |
  arrival     priority      preemption    WCET
  jitter      deadline      blocking      interference
```

The response time of a task is the time from its arrival to its completion. It includes its own execution time, the interference from higher-priority tasks, and the blocking from lower-priority tasks that hold a shared resource. The worst-case response time must be less than the deadline, with margin for the measurement uncertainty and the run-to-run variation.

A mean response time that meets the deadline is not evidence. A 99th-percentile response time that meets the deadline is evidence, and a worst-case bound that meets the deadline is stronger evidence.

### 3.2 The priority-inversion timeline

Priority inversion occurs when a high-priority task waits for a resource held by a low-priority task, and a medium-priority task preempts the low-priority task.

```text
 time -->
 H:            ready ......... blocked ............... runs
 M:                 runs .......................... runs
 L:  runs ... holds mutex ... preempted ... releases
     |         |               |             |
     t0        t1              t2            t3
```

Without priority inheritance, H waits from t1 to t3, which includes the entire medium-priority execution. With priority inheritance, L inherits H's priority at t1, so M cannot preempt it, and H waits only for L's remaining critical section.

The bound is the blocking time, not the average. A system with a 40 ms critical section can block a high-priority task for 40 ms, even if the critical section is entered once an hour.

### 3.3 The watchdog tree

A watchdog is a timer that resets the system if it is not refreshed. The feed path determines what the watchdog actually supervises.

```text
  task A --heartbeat--\
  task B --heartbeat---> health task ---> watchdog
  task C --heartbeat--/       |
                              +--> fail-safe state
                              +--> fault record
```

If the watchdog is fed by task A, it supervises task A only. If it is fed by an interrupt, it supervises the interrupt's arrival, not the task's progress. If it is fed by the health task after heartbeats from every task, it supervises the set.

The watchdog's timeout must be longer than the longest legitimate stall and shorter than the time to harm. That window is often narrow, and it must be justified.

### 3.4 The boot chain of trust

Secure boot is a chain. Each stage verifies the next stage before executing it.

```text
  immutable root of trust
        |
        v
  bootloader (verifies signature)
        |
        v
  application image (verified)
        |
        v
  optional second image or configuration
```

The root of trust is in read-only memory or in one-time-programmable fuses. If the root can be changed, the chain is only as strong as the change mechanism. If the debug port is open, the chain can be bypassed.

### 3.5 The A/B update model

An update writes a new image to the inactive slot, verifies it, and switches the boot pointer. If the new image fails to confirm, the bootloader reverts to the old image.

```text
  slot A: confirmed, running
  slot B: empty
     |
     +-- write new image to B
     +-- verify signature and hash
     +-- set boot pointer to B, mark test
     +-- boot B; application confirms or not
           |
           +-- confirmed: B becomes permanent
           +-- not confirmed: revert to A
```

The model requires three things: enough flash for two images, a bootloader that can swap or select, and an application that confirms a healthy boot. A power loss at any point must leave a bootable device.

### 3.6 The fault matrix

A fault matrix maps each fault to its detection, its response, and its residual state. It is the evidence that the firmware's safety behaviour was tested, not assumed.

| Fault | Detection | Response | Residual state |
|---|---|---|---|
| Task hang | Missing heartbeat | Reset | Fault record |
| Stack overflow | Stack guard or MPU | Reset | Fault record |
| Brown-out | Supply monitor | Safe state, then reset | Outputs off |
| Corrupted image | Signature check | Reject, boot old image | Old image |
| Wrong version | Anti-rollback counter | Reject | Old image |
| Watchdog disabled | Independent clock monitor | Reset | Fault record |
| Clock failure | Clock monitor | Switch to internal oscillator or reset | Safe state |

The matrix is not complete until every row has a measured detection time and a verified residual state.

### 3.7 The three numbers a timing claim needs

A firmware timing claim needs three numbers, and a claim that omits one is incomplete.

| Number | Question | Source | Example |
|---|---|---|---|
| Deadline | When must the task finish? | Requirement, derived from the control loop or the hazard | 2 ms |
| Worst-case execution time | How long does the task take without interference? | Measurement or static analysis | 180 µs |
| Blocking time | How long can a lower-priority task block it? | Critical-section analysis and measurement | 40 ms |

The response-time bound is the sum of the three, plus the interference from higher-priority tasks. A team that reports only the execution time has reported one third of the answer.

The deadline is usually derived from physics: a control loop with a 100 Hz bandwidth needs a sample period well below 10 ms, and a safety monitor needs a detection time below the time to harm. The derivation belongs in the requirement, not in the firmware.

### 3.8 The configuration that is not in the code

Some of the most consequential firmware settings are not in the source code. They are in option bytes, fuses, linker scripts, and build flags.

| Setting | Where it lives | What it controls | Consequence if wrong |
|---|---|---|---|
| Debug access | Option bytes or fuses | Whether the debug port is open | Key extraction, reprogramming |
| Boot mode | Option bytes | Which slot boots | Wrong image, no boot |
| Watchdog enable | Option byte or software | Whether the watchdog runs after reset | No reset on hang |
| Read-out protection | Option bytes | Whether flash can be read | Firmware extraction |
| Linker script | Build | Memory layout, stack placement | Overflow, wrong vector table |
| Optimisation | Build flag | Code timing and size | Different jitter |
| Floating point | Build flag | Hardware or software FP | Timing and code size |

A secure boot claim that covers the code but not the option bytes is incomplete. The production configuration must be recorded and verified on the shipped unit, not only in the source repository.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| RTOS | Real-time operating system | Any scheduler | None |
| Task | Concurrent activity with a stack and a priority | A function call | None |
| Priority | Scheduling rank | Confused with importance | Integer |
| Preemption | Suspension of a running task by a higher-priority one | Confused with blocking | None |
| Blocking | Waiting for a resource or an event | Confused with preemption | Time |
| Deadline | Time by which a task must complete | Confused with period | Time |
| Period | Interval between task releases | Confused with deadline | Time |
| WCET | Worst-case execution time | Average execution time | Time |
| Response time | Time from arrival to completion | Execution time only | Time |
| Jitter | Variation in period or latency | Confused with drift | Time |
| Priority inversion | High-priority task blocked by a low-priority task | Any blocking | Time |
| Priority inheritance | Temporary elevation of a lock holder's priority | Priority ceiling | None |
| Priority ceiling | Protocol that elevates a lock's priority to the highest user | Priority inheritance | None |
| Mutex | Mutual-exclusion lock with ownership | Binary semaphore | None |
| Semaphore | Signalling primitive without ownership | Mutex | None |
| Queue | Buffer between tasks | Shared variable | None |
| Static allocation | Memory allocated at compile time | Heap allocation | Bytes |
| Stack high-water mark | Maximum stack usage observed | Stack size | Bytes |
| Watchdog | Timer that resets the system if not refreshed | A software timer | Time |
| Independent watchdog | Watchdog clocked independently of the CPU | Window watchdog | Time |
| Window watchdog | Watchdog that must be refreshed within a window | Simple watchdog | Time |
| Heartbeat | Periodic signal that a task is alive | A log message | None |
| Fail-safe state | State that minimises harm on a fault | Any stopped state | State |
| Safe state | State in which the risk is acceptable | Fail-safe state | State |
| Fault record | Persisted record of a fault | A RAM variable | Record |
| Brown-out | Supply voltage below the operating threshold | A reset | V |
| Secure boot | Cryptographic verification before execution | Any boot | None |
| Root of trust | Immutable anchor of the boot chain | A key in flash | None |
| Signature | Cryptographic proof of authenticity | A checksum | Bytes |
| Hash | Digest of the image | A signature | Bytes |
| Anti-rollback | Prevention of installing an older image | Version check only | Counter |
| A/B slot | Two image slots for update and rollback | A backup file | Layout |
| Test swap | Boot a new image once before confirming | Permanent swap | Mode |
| Revert | Return to the previous image | Reboot | Mode |
| MCUboot | Open-source secure bootloader | A vendor bootloader | Project |
| TrustZone-M | Arm security extension for Cortex-M | A software sandbox | Hardware |
| MPU | Memory protection unit | MMU | Hardware |
| Boot time | Time from reset to application ready | Time to first print | Time |
| Reset latency | Time from fault to reset assertion | Time to detect | Time |
| SBOM | Software bill of materials | A list of files | Document |
| IEC 62304 | Medical device software lifecycle standard | A coding standard | Standard |
| Safety class | IEC 62304 software class A, B, or C | A risk level | Class |
| PL | Performance level in ISO 13849 | SIL | a–e |
| SIL | Safety integrity level in IEC 61508 | PL | 1–4 |

## 5. The mechanics, step by step

The procedure below builds and verifies the firmware lab. It assumes a Cortex-M board with an on-board debug probe, a logic analyser, a current-limited supply, and a FreeRTOS or Zephyr toolchain. The procedure is written for a two-person team: one operator and one reviewer.

### Step 1. Define the timing and safety requirements

**Input:** the device function and the hazard analysis.

**Output:** a requirement table with deadlines, jitter limits, watchdog timeout, safe-state time, and update requirements.

**Rationale:** a firmware test without a requirement cannot fail. The requirement table is the acceptance criterion for every measurement.

| ID | Requirement | Value | Verification |
|---|---|---:|---|
| FW-01 | Sampling period | 2000 µs ± 50 µs | Logic analyser |
| FW-02 | Sampling jitter, peak | ≤ 50 µs | Logic analyser |
| FW-03 | High-priority task response, p99 | ≤ 10 ms | GPIO toggle |
| FW-04 | Watchdog timeout | 1000 ms | Timer configuration |
| FW-05 | Fault to safe state | ≤ 2000 ms | Scope and logic analyser |
| FW-06 | Boot time | ≤ 500 ms | GPIO toggle |
| FW-07 | Unsigned image rejection | 20 of 20 | Boot log |
| FW-08 | Power-loss update recovery | 10 of 10 | Power switch |
| FW-09 | Rollback to previous image | 1 of 1 | Boot log |
| FW-10 | Run current | ≤ 25 mA | Power monitor |

The requirements are numbered so that the fault matrix and the test report can reference them. A requirement without a verification method is a wish.

Two requirements deserve a note. FW-05, the fault-to-safe-state time, is a system requirement that includes the watchdog timeout, the reset, the boot, and the output clamp. It cannot be verified by measuring only the watchdog. FW-08, the power-loss update recovery, is a reliability requirement that must be tested at several interruption points, because the flash write is not atomic.

A requirement table that lists only the nominal behaviour is incomplete. The faults are requirements too.

### Step 2. Define the safe state before writing code

**Input:** the outputs and the hazards.

**Output:** a table of every output, its safe value, and the time to reach it.

**Rationale:** the safe state is a design decision, not an error handler. It must be defined for every output, including the ones that are not obviously dangerous.

| Output | Normal value | Safe value | Time to safe | Hazard if wrong |
|---|---|---|---|---|
| Actuator enable | High | Low | < 10 ms | Uncontrolled motion |
| Motor PWM | 0–100% | 0% | < 10 ms | Uncontrolled motion |
| Alarm | Low | High | < 100 ms | Missed alarm |
| Display | On | Off or error screen | < 500 ms | Misleading information |
| Radio | On | Off | < 100 ms | Unauthorised transmission |
| Fault LED | Off | On | < 100 ms | No indication |

The safe state is verified with a meter or a scope, not by reading the code. A GPIO that is configured as an output but driven by a peripheral may not go to the value the code expects.

### Step 3. Build the bare-metal superloop and measure the baseline

**Input:** a timer, an ADC, a direct memory access (DMA) channel, and a GPIO.

**Output:** a sampling loop with a measured period and jitter.

**Rationale:** the bare-metal baseline shows what the hardware can do without an RTOS. It is the reference for the RTOS measurements.

Procedure:

1. Configure a hardware timer to trigger the ADC at 500 Hz.
2. Configure DMA to write samples to a buffer.
3. Toggle a GPIO in the ADC conversion-complete callback.
4. Capture the GPIO with a logic analyser at 24 MHz.
5. Compute the interval statistics over at least 1,000 intervals.
6. Replace the timer trigger with a software delay and repeat.

The software delay version is expected to collapse. The comparison shows why a hardware timer is required for a timing claim.

### Step 4. Port to the RTOS with static allocation

**Input:** the superloop and the task requirements.

**Output:** three tasks with priorities, stacks, and queues.

**Rationale:** the RTOS adds scheduling, blocking, and preemption. The port must preserve the timing requirement, and the resource usage must be measured.

| Task | Priority | Stack | Period | Function |
|---|---:|---:|---:|---|
| Acquisition | 4 | 1024 B | 2 ms | Read ADC, enqueue sample |
| Processing | 3 | 2048 B | 2 ms | Filter, detect, compute |
| Logging | 1 | 1024 B | 100 ms | Write to UART or storage |

Use static allocation for the tasks, the queues, and the timer. Record the stack high-water mark for each task after 10 minutes of operation. A stack that is 80% used at 10 minutes is a stack that will overflow at 10 hours.

### Step 5. Measure the scheduling latency and jitter

**Input:** the RTOS build and a GPIO toggle in each task.

**Output:** the response-time distribution for each task.

**Rationale:** the scheduler's overhead and the interrupt latency are part of the response time. They must be measured, not assumed.

Procedure:

1. Toggle a GPIO at the start of the acquisition task.
2. Toggle a second GPIO at the end of the processing task.
3. Capture both with the logic analyser.
4. Compute the period, the jitter, and the response time.
5. Repeat with the logging task active and inactive.
6. Repeat with the UART transmitting at full rate.

The last two conditions matter. A logging task that shares a bus with the acquisition task can add latency that is invisible in the idle case. The measurement must state the workload: which tasks are active, what the UART is doing, and whether the debugger is attached. A jitter number measured with the logging task disabled is a best case, not a bound.

### Step 6. Create and measure priority inversion

**Input:** a mutex shared by the processing and logging tasks, and a high-priority task.

**Output:** the worst-case blocking time with and without priority inheritance.

**Rationale:** priority inversion is a scheduling property, not a bug. It must be demonstrated and bounded.

Procedure:

1. Create a low-priority task that takes the mutex and holds it for a known time.
2. Create a medium-priority task that runs without the mutex.
3. Create a high-priority task that takes the mutex.
4. Measure the high-priority task's response time without priority inheritance.
5. Enable priority inheritance on the mutex and repeat.
6. Record the blocking time and the critical-section duration.

FreeRTOS mutexes implement priority inheritance, but the implementation assumes a task does not hold more than one mutex at a time. A design that nests mutexes can still suffer unbounded inversion. The measurement must use the actual lock configuration.

### Step 7. Add the independent watchdog and health monitor

**Input:** the task heartbeats and the watchdog peripheral.

**Output:** a watchdog that is fed only by the health task after all heartbeats arrive.

**Rationale:** the feed path determines what the watchdog supervises. A watchdog fed by the acquisition task supervises acquisition only.

Procedure:

1. Configure the independent watchdog with a 1000 ms timeout.
2. Create a health task at the highest priority.
3. Have each supervised task set an event bit at least every 200 ms.
4. Have the health task wait for all bits with a 200 ms timeout.
5. If all bits arrive, refresh the watchdog; otherwise, log the fault, enter the safe state, and reset.
6. Inject a hang in each task in turn and measure the time to reset.

The health task must not be the only task that can feed the watchdog if the health task itself can hang. A second, independent monitor, such as a window watchdog or a clock monitor, may be needed.

### Step 8. Verify the fail-safe state with an instrument

**Input:** the safe-state table and the fault injection.

**Output:** a capture showing every output at its safe value within the required time.

**Rationale:** a safe state that is only inspected in code is a hypothesis. The instrument shows the actual pin behaviour.

Procedure:

1. Connect the scope to the actuator enable, the motor PWM, and the alarm.
2. Inject a fault that triggers the safe state.
3. Capture the outputs and the reset line.
4. Measure the time from the fault to each safe value.
5. Verify that the outputs remain safe through the reset and the boot.
6. Repeat for each fault in the matrix.

A pin that is driven by a timer peripheral may continue to toggle after the CPU resets, until the peripheral is reinitialised. The capture must cover the reset and the boot, not only the moment of the fault.

### Step 9. Sign the image and verify the boot chain

**Input:** the firmware image, a signing key, and a bootloader with signature verification.

**Output:** a signed image that boots, and an unsigned image that is rejected.

**Rationale:** secure boot is a chain of verification. The test must show that each link rejects an invalid input.

Procedure:

1. Generate a key pair and store the public key in the bootloader's trusted area.
2. Sign the image with the private key.
3. Verify the signature with the vendor tool before flashing.
4. Flash the signed image and confirm the boot.
5. Modify one byte of the image and confirm that the bootloader rejects it.
6. Flash an unsigned image and confirm that it is rejected.
7. Try to disable verification and confirm that the build system prevents it.
8. Confirm that the debug port is locked in production configuration, and that the unlock procedure is documented and audited.

The last step is the one most often skipped. A secure boot with an open debug port is a lock with the key in the door.

### Step 10. Test the update and the rollback

**Input:** the A/B slots, the bootloader, and a power switch.

**Output:** an update that survives a power loss and a rollback that restores the previous image.

**Rationale:** an update is a sequence of writes that can be interrupted at any point. The device must boot a valid image after every interruption.

Procedure:

1. Confirm that the device boots from slot A.
2. Write a new signed image to slot B.
3. Set the boot pointer to B and mark the image as a test image.
4. Boot B and confirm that the application marks itself healthy.
5. Repeat the sequence, but cut power at 25%, 50%, and 75% of the write.
6. Confirm that the device boots a valid image after each interruption.
7. Boot a deliberately unhealthy image and confirm that the bootloader reverts to A.
8. Confirm that an older image is rejected by the anti-rollback counter.

The test must use a real power switch or a controllable supply, not a software reset. A software reset does not interrupt the flash write at the same point.

### Step 11. Build the fault-injection matrix

**Input:** the requirements and the fault list.

**Output:** a matrix with detection, detection time, response, recovery, and residual state.

**Rationale:** the matrix is the evidence that the firmware's safety behaviour was tested. A matrix that covers only the nominal path is not a matrix.

| Fault | Injection | Detection | Detection time | Response | Residual state |
|---|---|---|---:|---|---|
| Task hang | `while(1)` in each task | Missing heartbeat | ≤ 200 ms | Reset | Fault record |
| Stack overflow | Recursion or large local array | Stack guard or MPU | < 1 ms | Reset | Fault record |
| Brown-out | Reduce supply below threshold | Supply monitor | < 1 ms | Safe state, reset | Outputs off |
| Corrupted image | Flip a byte | Signature check | Boot time | Reject, boot old | Old image |
| Wrong version | Flash older image | Anti-rollback | Boot time | Reject | Current image |
| Watchdog disabled | Clear the enable bit | Clock monitor or second watchdog | < 10 ms | Reset | Fault record |
| Clock failure | Disable the external oscillator | Clock monitor | < 1 ms | Switch or reset | Safe state |
| Queue overflow | Stop the consumer | Queue full check | < 1 ms | Drop or block | Logged |

Each row is verified with a capture. A row that is filled in from the design, without a capture, is a hypothesis.

### Step 12. Measure boot time, reset latency, and current

**Input:** the instrumented firmware and the power monitor.

**Output:** the timing and current table.

**Rationale:** the boot time and the reset latency are part of the safety budget. The current is part of the battery life and the thermal budget.

Procedure:

1. Toggle a GPIO at the reset vector and at the application-ready point.
2. Measure the boot time with the logic analyser.
3. Inject a watchdog fault and measure the time from the fault to the reset assertion.
4. Measure the time from the reset assertion to the safe state.
5. Measure the current in run, idle, and fault states with a power monitor.
6. Record the instrument, the range, the resolution, and the uncertainty.

A boot time measured from the first UART character is not the boot time. The UART is initialised after the clock, the memory, and the application. The GPIO toggle at the reset vector is the correct start.

### Step 13. Record versions, build flags, and the SBOM

**Input:** the build.

**Output:** a reproducible build record.

**Rationale:** a firmware result is about a binary. The binary must be identified by more than a version string.

Record:

- The source commit and the tag.
- The compiler and the toolchain version.
- The optimisation flags and the linker script.
- The board revision and the bootloader version.
- The model or configuration hashes, if any.
- The SBOM with component versions and licences.
- The command used to build the image.
- The image hash.

### Step 14. Review and archive

**Input:** the requirements, the measurements, the fault matrix, and the build record.

**Output:** a firmware report and an archive.

**Rationale:** the report is the evidence. The archive is what makes it reproducible.

The review checks that every requirement has a measurement, every fault has a capture, the safe state is verified for every output, the update survives power loss, and the build record is complete.

### 5.15 Acceptance criteria for the firmware lab

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Requirements are numbered and verifiable | Requirement table | Rewrite |
| 2 | Safe state is defined for every output | Output table | Complete |
| 3 | Sampling jitter is measured | Logic analyser capture | Measure |
| 4 | Scheduling response is measured | GPIO capture | Measure |
| 5 | Priority inversion is demonstrated | Timing table | Repeat |
| 6 | Watchdog resets a hung task | Capture | Fix feed path |
| 7 | Safe state is verified with an instrument | Scope capture | Fix |
| 8 | Unsigned image is rejected | Boot log | Fix |
| 9 | Corrupted image is rejected | Boot log | Fix |
| 10 | Power loss leaves a bootable device | 10 trials | Fix |
| 11 | Rollback is demonstrated | Boot log | Fix |
| 12 | Anti-rollback is demonstrated | Boot log | Fix |
| 13 | Fault matrix has a capture per row | Matrix | Complete |
| 14 | Boot, reset, and current are measured | Table | Measure |
| 15 | Build record and SBOM are complete | Archive | Complete |

### 5.16 Branch points

**Branch A: the board has no secure boot support.** Simulate the verification step on the host, document the gap, and state that the secure boot claim is not demonstrated. Do not claim secure boot from a software check alone.

**Branch B: the board has a single image slot.** Demonstrate the update failure and the recovery path, and document the absence of rollback. Do not claim rollback capability.

**Branch C: the debug port cannot be locked.** Record the limitation and treat the secure boot claim as conditional. A device with an open debug port can be reprogrammed by anyone with physical access.

**Branch D: the actuator is connected.** Verify the safe state with the actuator disconnected first, then with a dummy load, then with the actuator under supervision. Never put hands in the workspace during the test.

## 6. Worked example

### 6.1 The question

A team has built acquisition firmware on a Cortex-M board with an RTOS, a watchdog, and a fail-safe state. The requirements are FW-01 to FW-05 from Section 5. The team wants to know whether the firmware meets the timing and safety requirements, and what the evidence shows.

### 6.2 The setup

| Item | Value | Note |
|---|---|---|
| Board | Cortex-M4 at 80 MHz | 128 KB RAM, 512 KB flash |
| RTOS | FreeRTOS, preemptive | Mutexes with priority inheritance |
| Sampling | 500 Hz, timer-driven ADC with DMA | 2 ms period |
| Logic analyser | 8-channel, 24 MHz | 41.67 ns resolution |
| Logic analyser timebase | ±50 ppm | 0.1 µs at 2 ms |
| Watchdog | Independent, 1000 ms timeout | Internal low-speed oscillator |
| Watchdog clock accuracy | ±2% over temperature | Datasheet |
| Scope | 100 MHz, 2 channels | 8-bit |
| Power monitor | 12-bit, 0.5% max | 320 mV shunt range |
| Firmware commit | 5c1d9a2 | Tagged |
| Toolchain | Arm GNU 13.2 | Recorded |

### 6.2a The build configuration

The build configuration is part of the result. A timing measurement on a debug build is not a timing measurement on the released image.

| Setting | Value | Why it matters |
|---|---|---|
| RTOS | FreeRTOS 10.x, preemptive | Scheduler behaviour |
| Tick rate | 1000 Hz | 1 ms scheduling resolution |
| Heap scheme | heap_4, static allocation for tasks | Fragmentation risk |
| Optimisation | -O2 | Code timing |
| Debug symbols | -g3 | No timing effect, but larger image |
| Assertions | Enabled in debug, disabled in release | Timing and code size |
| Watchdog | Independent, 1000 ms | Reset behaviour |
| UART logging | Enabled at 115200 baud | Adds interrupt load |
| Build ID | 5c1d9a2, image hash recorded | Identifies the binary |

The measurement must state which build was used. A debug build with assertions and logging can have a different jitter from a release build, and the difference is often larger than the requirement's margin.

### 6.3 Sampling jitter

The acquisition task toggles a GPIO at the start of each ADC callback. The logic analyser captures 1,000 intervals.

| Statistic | Value | Unit |
|---|---:|---|
| Mean period | 2000.4 | µs |
| Standard deviation | 3.8 | µs |
| Minimum interval | 1981 | µs |
| Maximum interval | 2019 | µs |
| Peak-to-peak jitter | 38 | µs |
| 99th percentile interval | 2009 | µs |

The requirement FW-01 is 2000 µs ± 50 µs, and FW-02 is a peak jitter of 50 µs or less. The measured peak-to-peak jitter is 38 µs, which is within the requirement. The mean is 0.4 µs above the nominal, which is negligible.

The jitter is not Gaussian. The distribution has a short tail on the low side and a longer tail on the high side, which is consistent with occasional preemption by the logging task. A mean and a standard deviation alone would hide this.

### 6.4 The timing uncertainty

The logic analyser's timebase contributes 50 ppm of the interval, which is 0.1 µs at 2 ms. The analyser's resolution is 41.67 ns. The combined timing uncertainty is:

\[
u_t = \sqrt{(0.1)^2 + (0.0417/\sqrt{12})^2} = \sqrt{0.01 + 0.000145} = 0.1007\ \mu\text{s}
\]

The uncertainty is negligible compared with the 38 µs jitter. The jitter is a property of the firmware, not of the instrument.

### 6.5 Priority inversion

The team creates a low-priority task that holds a mutex for 40 ms, a medium-priority task that runs for 60 ms, and a high-priority task that needs the mutex. The high-priority task's response time is measured with and without priority inheritance.

| Condition | Critical section | Medium task | High task response | Requirement | Verdict |
|---|---:|---:|---:|---:|---|
| No inheritance | 40 ms | 60 ms | 44.2 ms | ≤ 10 ms | Fail |
| Inheritance enabled | 40 ms | 60 ms | 2.6 ms | ≤ 10 ms | Pass |

Without inheritance, the high-priority task waits for the low-priority task to finish its critical section, but the low-priority task is preempted by the medium-priority task. The wait is the critical section plus the preemption time, minus the part of the critical section that ran before the preemption.

The measured 44.2 ms is larger than the 40 ms critical section because the low-priority task had not started the critical section when the high-priority task arrived, and the medium-priority task ran first. With inheritance, the low-priority task inherits the high priority and completes the critical section in 2.6 ms.

The result is a demonstration of the mechanism, not a proof of the worst case. The worst case depends on the longest critical section in the system, which must be found by inspection and measurement.

### 6.5a The worst-case critical section

The measured 44.2 ms is the blocking time for one critical section under one workload. The worst-case blocking time is the longest critical section in the system, and it must be found by inspection and measurement.

| Critical section | Holder | Duration | Shared with | Measured |
|---|---|---:|---|---|
| Sample buffer update | Acquisition | 12 µs | Processing | Yes |
| Filter state update | Processing | 180 µs | Logging | Yes |
| Log record format | Logging | 40 ms | Processing | Yes |
| Fault record write | Health | 8 ms | None | No |

The 40 ms log record is the longest section and the source of the inversion. It is held while the logging task formats a record, which includes a floating-point conversion and a memory copy. The fix is to shorten the section: format into a local buffer outside the lock and hold the lock only for the queue write. That change would reduce the critical section to about 20 µs, and the inversion would become negligible.

The table is the evidence that the worst case was searched, not only the observed case. A measurement of one scenario is not a bound.

### 6.6 Watchdog reset latency

The team injects a hang in the acquisition task with a `while(1)` loop. The watchdog timeout is 1000 ms. The reset is captured on the scope, and the boot to the safe state is captured on the logic analyser.

| Interval | Measured | Unit | Requirement | Verdict |
|---|---:|---|---:|---|
| Hang to watchdog reset | 1004 | ms | — | — |
| Reset to safe state | 212 | ms | — | — |
| Hang to safe state | 1216 | ms | ≤ 2000 ms | Pass |
| Margin | 784 | ms | — | — |

The watchdog reset occurs 1004 ms after the hang, which is within the 2% oscillator tolerance of the 1000 ms timeout. The 4 ms excess is within the expected 20 ms tolerance, so the measurement is consistent with the design.

The boot to the safe state takes 212 ms. The total is 1216 ms, which is within the 2000 ms requirement, with 784 ms of margin.

### 6.7 The uncertainty of the watchdog measurement

The watchdog oscillator's ±2% tolerance contributes ±20 ms to the timeout. The logic analyser's timebase contributes 0.1 µs, which is negligible. The scope's timebase contributes a similar amount. The dominant uncertainty is the oscillator.

The measured 1004 ms is consistent with a true timeout anywhere from 980 to 1020 ms. The requirement is on the total time to the safe state, so the conservative bound is:

\[
t_{\text{total,max}} = 1020 + 212 + 0.1 = 1232\ \text{ms}
\]

The conservative bound is still below 2000 ms, with 768 ms of margin.

### 6.8 Two checks that would invalidate the result

**Check 1: the watchdog is refreshed by an interrupt or by the idle task hook.** If the watchdog is refreshed in a timer interrupt, it will continue to be refreshed even when the acquisition task is hung. The reset would never occur, and the measured 1004 ms would be a different fault path. The check is to inspect the feed call graph and to confirm that only the health task calls the refresh function. A grep for the refresh function is not sufficient if the function is called through a pointer or a macro; the check is to breakpoint the refresh and observe the call stack.

**Check 2: the debugger is attached and the watchdog is frozen during the halt.** Many Cortex-M debug configurations freeze the watchdog when the core is halted. If the team measured the reset latency with the debugger attached and the freeze enabled, the watchdog would not run during the halt, and the measured time would be wrong. The check is to repeat the measurement with the debugger detached, or to confirm that the debug freeze is disabled and recorded.

### 6.6a The reset path

The 212 ms from reset to safe state is a sum of stages. Breaking it down shows where the time goes and what could be shortened.

| Stage | Time | What happens |
|---|---:|---|
| Reset vector to clock init | 4 ms | Startup code, clock tree |
| Clock init to memory init | 18 ms | RAM test, section copy |
| Memory init to RTOS start | 42 ms | Driver init, watchdog init |
| RTOS start to first task | 3 ms | Scheduler start |
| First task to safe state | 145 ms | Self-test, output clamp, fault record |
| **Total** | **212 ms** | |

The largest stage is the self-test and safe-state entry. A shorter boot would not help much unless the self-test is reduced. The requirement is 2000 ms, so the boot time is not the binding constraint at a 1000 ms watchdog timeout. At a 100 ms timeout, the boot time becomes the dominant term, which is why the timeout sweep flattens.

### 6.9 A varied condition: the watchdog timeout is reduced to 200 ms

The team reduces the watchdog timeout to 200 ms and repeats the measurement.

| Condition | Timeout | Reset latency | Boot to safe | Total | Requirement | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 1000 ms | 1004 ms | 212 ms | 1216 ms | 2000 ms | Pass |
| Varied | 200 ms | 204 ms | 212 ms | 416 ms | 2000 ms | Pass |
| Varied | 100 ms | 104 ms | 212 ms | 316 ms | 2000 ms | Pass |

The total time falls as the timeout falls, until the boot time dominates. At a 100 ms timeout, the total is 316 ms, which is close to the boot time plus the timeout. A further reduction in the timeout would not reduce the total below the boot time.

The varied condition shows the trade-off. A shorter timeout detects a hang faster but increases the risk of a false reset during a legitimate long operation. The timeout must be longer than the longest legitimate stall, which the team must measure.

### 6.9a Choosing the timeout

The timeout is not a free parameter. It must be longer than the longest legitimate stall and shorter than the time to harm.

| Constraint | Value | Source |
|---|---:|---|
| Longest legitimate stall | 320 ms | Flash erase during a log commit |
| Watchdog oscillator tolerance | ±2% | Datasheet |
| Time to harm | 2500 ms | Risk file, actuator motion |
| Boot to safe state | 212 ms | Measurement |
| Chosen timeout | 1000 ms | Design |

The chosen timeout is 3.1 times the longest legitimate stall and leaves 1232 ms to the safe state, against a 2500 ms time to harm. The margin is 1268 ms. A shorter timeout would reduce the detection time but increase the risk of a false reset during a flash erase. A longer timeout would reduce the margin to harm.

The table is the justification. A timeout chosen because "1 second is standard" is not justified.

### 6.10 What the example demonstrates

The arithmetic closes: a mean period of 2000.4 µs, a standard deviation of 3.8 µs, a peak-to-peak jitter of 38 µs, a priority-inversion response of 44.2 ms without inheritance and 2.6 ms with it, a watchdog reset at 1004 ms, a boot to safe state at 212 ms, and a total of 1216 ms against a 2000 ms requirement.

The example demonstrates the difference between a mean and a tail. The mean sampling period is within 0.4 µs of nominal, but the tail reaches 2019 µs. A control loop designed on the mean would be designed on the wrong number.

The example also demonstrates the difference between a watchdog and a timer. The watchdog fires because the health task stops receiving a heartbeat, not because the acquisition task stopped toggling a pin. The feed path is the control.

## 7. Code, query, or calculation

The program below analyses a timing capture, computes the response-time bound for a fixed-priority task set, and checks the watchdog budget. It uses only the Python standard library.

```python
import math
import statistics

# --- 1. Sampling capture: 20 intervals in microseconds -------------------
# In a real lab this is the logic-analyser export of the GPIO toggles.
INTERVALS_US = [
    2000, 2003, 1997, 2001, 2019, 1998, 2002, 2000, 1996, 2004,
    2001, 1999, 2003, 2000, 2002, 1998, 2017, 2000, 2001, 1999,
]

def summarise(xs):
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    return {
        "n": len(xs),
        "mean": m,
        "sd": sd,
        "min": min(xs),
        "max": max(xs),
        "pp": max(xs) - min(xs),
    }

s = summarise(INTERVALS_US)
print("sampling interval (us)")
print(f"n={s['n']} mean={s['mean']:.1f} sd={s['sd']:.2f} "
      f"min={s['min']} max={s['max']} peak-to-peak={s['pp']}")

# 99th percentile by nearest-rank on the sorted sample.
def percentile(xs, p):
    ys = sorted(xs)
    rank = math.ceil(p / 100 * len(ys))
    return ys[rank - 1]

print(f"p99 interval = {percentile(INTERVALS_US, 99)} us")
print(f"FW-01 (2000 +/- 50 us): "
      f"{'PASS' if 1950 <= s['mean'] <= 2050 else 'FAIL'}")
print(f"FW-02 (peak jitter <= 50 us): "
      f"{'PASS' if s['pp'] <= 50 else 'FAIL'}")

# --- 2. Response-time analysis for fixed-priority scheduling ------------
# Tasks: (name, period_us, wcet_us, priority, deadline_us)
TASKS = [
    ("acq",   2000, 180, 4, 2000),
    ("proc",  2000, 420, 3, 2000),
    ("log",  10000, 900, 1, 10000),
]

def response_time(task, higher):
    """Classic fixed-priority response-time bound: R = C + sum ceil(R/T)*C."""
    c = task[2]
    r = c
    for _ in range(1000):
        interference = sum(math.ceil(r / h[1]) * h[2] for h in higher)
        r_next = c + interference
        if r_next == r:
            return r
        r = r_next
    return None

print("\nfixed-priority response-time bound (us)")
for i, task in enumerate(TASKS):
    higher = [t for t in TASKS if t[3] > task[3]]
    r = response_time(task, higher)
    verdict = "PASS" if r is not None and r <= task[4] else "FAIL"
    print(f"{task[0]:5s} C={task[2]:4d} T={task[1]:5d} R={r} D={task[4]} {verdict}")

# --- 3. Priority inversion: measured blocking ---------------------------
NO_INHERIT_MS = 44.2
INHERIT_MS = 2.6
DEADLINE_MS = 10.0
print("\npriority inversion (ms)")
print(f"no inheritance = {NO_INHERIT_MS:.1f}  "
      f"{'PASS' if NO_INHERIT_MS <= DEADLINE_MS else 'FAIL'}")
print(f"inheritance    = {INHERIT_MS:.1f}  "
      f"{'PASS' if INHERIT_MS <= DEADLINE_MS else 'FAIL'}")
print(f"improvement    = {NO_INHERIT_MS/INHERIT_MS:.1f}x")

# --- 4. Watchdog budget --------------------------------------------------
TIMEOUT_MS = 1000.0
OSC_TOL = 0.02
BOOT_MS = 212.0
REQ_MS = 2000.0

t_max = TIMEOUT_MS * (1 + OSC_TOL) + BOOT_MS
t_min = TIMEOUT_MS * (1 - OSC_TOL) + BOOT_MS
print("\nwatchdog budget (ms)")
print(f"timeout nominal      = {TIMEOUT_MS:.0f}")
print(f"timeout range        = {TIMEOUT_MS*(1-OSC_TOL):.0f} to {TIMEOUT_MS*(1+OSC_TOL):.0f}")
print(f"boot to safe         = {BOOT_MS:.0f}")
print(f"total range          = {t_min:.0f} to {t_max:.0f}")
print(f"requirement          = {REQ_MS:.0f}")
print(f"worst-case margin    = {REQ_MS - t_max:.0f}")
print(f"verdict              = {'PASS' if t_max <= REQ_MS else 'FAIL'}")

# --- 5. Scale the timeout ----------------------------------------------
print("\ntimeout sweep")
for timeout in (100, 200, 500, 1000, 1500):
    total = timeout * (1 + OSC_TOL) + BOOT_MS
    print(f"timeout {timeout:4d} ms -> total {total:7.1f} ms  "
          f"{'PASS' if total <= REQ_MS else 'FAIL'}")
```

Expected output:

```text
sampling interval (us)
n=20 mean=2002.0 sd=5.85 min=1996 max=2019 peak-to-peak=23
p99 interval = 2019 us
FW-01 (2000 +/- 50 us): PASS
FW-02 (peak jitter <= 50 us): PASS

fixed-priority response-time bound (us)
acq   C= 180 T= 2000 R=180 D=2000 PASS
proc  C= 420 T= 2000 R=600 D=2000 PASS
log   C= 900 T=10000 R=1500 D=10000 PASS

priority inversion (ms)
no inheritance = 44.2  FAIL
inheritance    = 2.6  PASS
improvement    = 17.0x

watchdog budget (ms)
timeout nominal      = 1000
timeout range        = 980 to 1020
boot to safe         = 212
total range          = 1192 to 1232
requirement          = 2000
worst-case margin    = 768
verdict              = PASS

timeout sweep
timeout  100 ms -> total   314.0 ms  PASS
timeout  200 ms -> total   416.0 ms  PASS
timeout  500 ms -> total   722.0 ms  PASS
timeout 1000 ms -> total  1232.0 ms  PASS
timeout 1500 ms -> total  1742.0 ms  PASS
```

The program's sample statistics differ from the worked example because the worked example uses 1,000 intervals and the program uses a 20-interval excerpt. The excerpt gives a mean of 2002.0 µs, a standard deviation of 5.85 µs, and a peak-to-peak of 23 µs. The conclusion is the same, but the numbers are not interchangeable. This is a useful reminder: the summary depends on the sample, and the report must state the sample size.

The response-time analysis is the classic bound for fixed-priority preemptive scheduling. The acquisition task has no higher-priority interference, so its response is its execution time. The processing task is interfered with by acquisition once in its 2 ms window, so its response is 420 + 180 = 600 µs. The logging task has a 10 ms period and is interfered with by both higher-priority tasks.

The analysis assumes that the tasks are independent, that the deadlines are less than or equal to the periods, and that the blocking time is zero. A real system has blocking from shared resources, which must be added to the response time. The priority-inversion measurement is the evidence for the blocking term.

## 8. Data, standards, and regulatory context

### 8.1 IEC 62304

IEC 62304:2006 + Amendment 1:2015 defines the software life-cycle processes for medical device software. Amendment 1 replaced the severity-only classification with a risk-based scheme. Class A applies when the software cannot contribute to a hazardous situation, or when the risk is acceptable after external controls. Class B applies when the software can contribute to a hazardous situation with non-serious injury. Class C applies when the possible harm is death or serious injury.

The class determines the required processes, the documentation, and the verification. A Class C project needs software unit verification, integration testing, and system testing, with a documented traceability from requirements to tests. A Class A project needs a simpler set.

A teaching firmware lab is not a medical device software project. It can use the classification to decide how much rigour the firmware needs, and it can use the same traceability structure.

### 8.2 Safety functions and performance levels

A safety function is a function that reduces risk to an acceptable level. Its reliability is described by a performance level (PL) under ISO 13849 or a safety integrity level (SIL) under IEC 61508 and IEC 62061.

A software watchdog is not automatically a safety-rated function. A safety-rated function requires a defined architecture, diagnostic coverage, and a quantified failure rate. A lab can demonstrate the mechanism and measure the detection time; it cannot assign a PL or a SIL without the full analysis.

### 8.3 Secure boot and update

MCUboot is an open-source secure bootloader for 32-bit microcontrollers. Its design document defines the image trailer, the test swap, the revert mode, and the anti-rollback counter. A test swap boots a new image once; if the image does not confirm itself, the bootloader reverts to the previous image. The anti-rollback counter prevents the installation of an older image with a known vulnerability.

The design document describes the swap types and the trailer states. The implementation requires a hardware security counter in protected non-volatile memory for hardware rollback protection. A software-only version check is weaker, because an attacker who can write the image can also write the version.

Arm TrustZone for Cortex-M provides a hardware isolation boundary between a secure and a non-secure world. It can be used to protect the bootloader and the keys from the application. It is not a substitute for a secure boot chain; it is a mechanism for implementing one.

### 8.4 Cybersecurity lifecycle

IEC 81001-5-1:2021 defines security activities across the health software product lifecycle. The FDA recognises it as a framework for a secure product development framework and as a standard that can partially meet the security testing recommendations.

The FDA's cybersecurity guidance treats a software bill of materials (SBOM) as part of the premarket submission for cyber devices. The SBOM lists the components, their versions, and their licences. For firmware, it includes the RTOS, the bootloader, the cryptographic library, and the drivers.

The FDA's Section 524B requirements, added by the December 2022 Omnibus, require a post-market vulnerability monitoring and response plan, processes for cybersecurity assurance and updates, and an SBOM for cyber devices.

### 8.5 Risk management

ISO 14971:2019 requires the manufacturer to identify hazards, estimate and evaluate the risk, implement controls, and evaluate the residual risk. A firmware fault is a sequence of events that can lead to a hazardous situation. The fault matrix is the evidence that the sequence was tested.

The standard does not prescribe a fault-injection method. It requires that the controls be verified. A fault matrix with a capture per row is one way to provide that evidence.

### 8.7 Option bytes, fuses, and production configuration

A production firmware image is more than the binary. It includes the option bytes that enable the watchdog, disable the debug port, protect the read-out, and select the boot mode. These settings are often programmed separately from the image, and they can be reset by a mass erase.

The verification must therefore cover the shipped configuration. The test is to read the option bytes from a production unit, compare them with the released configuration, and attempt a debug connection. A unit that accepts a debug connection has failed the configuration, even if the image is correctly signed.

The STM32 family, for example, uses option bytes for read-out protection, write protection, and the watchdog. The exact behaviour depends on the part number and the boot configuration. The vendor's programming manual is the reference, and the values must be recorded in the build record.

### 8.8 Safety-rated functions and software

A safety function that depends on software must be developed to a functional-safety standard. ISO 13849 defines performance levels for machinery control systems, and IEC 61508 defines safety integrity levels for electrical and electronic systems. IEC 62061 is the machinery-sector application of IEC 61508.

A medical robot may need both the medical particular standard and the machinery safety standard. IEC 80601-2-77 covers the medical device, and ISO 10218-1 covers the industrial robot. The two frameworks use different terminology and different reliability metrics, and a claim under one is not automatically a claim under the other.

A lab can demonstrate a safety function's mechanism, measure its detection time, and test its response. It cannot assign a PL or a SIL without the architecture, the diagnostic coverage, and the failure-rate data.

### 8.9 What is binding and what is convention

| Document | Status | Binds | Does not bind |
|---|---|---|---|
| IEC 62304 | International standard | Medical device software | A teaching firmware |
| IEC 81001-5-1 | International standard | Health software security lifecycle | A prototype |
| ISO 14971 | International standard | Medical device risk management | A lab hazard list |
| ISO 13849 | International standard | Machinery safety functions | A servo arm |
| IEC 61508 | International standard | Functional safety | A lab demo |
| MCUboot design document | Project documentation | MCUboot users | Other bootloaders |
| FDA cybersecurity guidance | Guidance | Premarket submissions | A teaching lab |

## 9. Industry practice and case evidence

### 9.1 How device companies structure firmware verification

A device company typically separates unit tests, integration tests, and system tests. Unit tests run on the host or on the target with mocked peripherals. Integration tests run on the target with real peripherals. System tests run the whole device with simulated or real inputs.

The timing evidence is usually collected with a logic analyser or a trace unit, not with a printf. A printf changes the timing it measures. A GPIO toggle is the standard instrument for a timing claim because it is fast, deterministic, and visible to an external analyser.

### 9.2 Watchdog practice

The common industrial pattern is a window watchdog fed by a health monitor. The health monitor checks heartbeats from the supervised tasks, and the watchdog is refreshed only if all heartbeats arrive within their windows. The window prevents a refresh that is too early as well as one that is too late.

The pattern is not universal. Some devices use a simple watchdog fed by the main loop, which detects a total hang but not a partial failure. The choice depends on the hazard analysis and the required diagnostic coverage.

### 9.3 Secure boot in the field

A secure boot chain is only as strong as its configuration. Public recall narratives have described devices shipped with secure boot reportedly not enabled on a console. The narrative is a reminder that the feature and the configuration are different things, and that the configuration must be verified on the shipped unit.

The lab response is to test the configuration, not only the feature: flash an unsigned image, confirm rejection, confirm the debug lock, and record the result.

### 9.4 Update and rollback

An over-the-air update is a sequence of writes that can be interrupted. The industrial pattern is an A/B slot layout with a test boot and a confirmation. The bootloader must be able to recover from a power loss at any point, and the application must confirm a healthy boot before the new image becomes permanent.

The pattern is common in consumer devices and increasingly common in medical devices. The FDA's guidance on predetermined change control plans for AI-enabled device software functions addresses the update process for a specific class of changes; it does not prescribe the bootloader design.

### 9.5 Post-market firmware updates

A medical device that can update its firmware is a device that can be patched. The same capability is also a risk: an unauthorised update can change the device's behaviour, and a failed update can remove it from service.

The industry pattern is a signed image, a secure boot chain, a rollback slot, and a controlled update process. The process includes a risk assessment of the change, a verification of the updated device, and a record of the versions in the field. The FDA's guidance on predetermined change control plans addresses a subset of changes that can be made without a new marketing submission; it requires the changes to be specified in advance and the verification to be pre-defined.

For a lab, the lesson is that the update is part of the device's safety case, not an afterthought. A rollback test is as important as a successful update.

### 9.6 What a security reviewer asks

A security reviewer who examines firmware typically asks:

1. What is the root of trust, and can it be changed?
2. Is the debug port locked in production, and how is the unlock controlled?
3. Is the image signed, and where is the public key stored?
4. Is there an anti-rollback counter, and is it protected?
5. What happens if the power fails during an update?
6. What is in the SBOM, and which components have known vulnerabilities?
7. How is a vulnerability reported and patched?
8. Is the update authenticated end to end, or only at the boot?

The questions are a review checklist, not a standard. A team that can answer all eight has a defensible update story. A team that can answer only the third has a feature, not a process.

### 9.7 Case evidence: the secure-boot configuration gap

The survey file lists a 2025 recall of a surgical robot console in which Secure Boot was reportedly not enabled. The event is a public narrative, and the identifier and the technical details must be checked against the regulator's database before it is cited in a report.

The lesson for the lab does not depend on the specific device. It is the distinction between a capability and a configuration. A processor can support secure boot; a product can ship with the verification disabled, the key missing, or the debug port open. The only way to know which one shipped is to test the shipped unit.

The test is simple: take a production unit, attempt to boot an unsigned image, attempt to attach a debugger, and read the option bytes. If any of the three succeeds, the configuration is not what the claim says. The test takes minutes and catches a class of defect that a code review cannot.

### 9.8 Inference

It is an inference, not a documented finding, that teams which measure the watchdog feed path with a call-graph tool find more feed-path defects than teams which inspect the code by reading. The reasoning is that the call graph catches indirect calls and macro expansions that a reading misses. The inference is consistent with the lab experience described in this topic, but it has not been tested across teams.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Watchdog fed by the supervised task | Refresh call in the task loop | Timer, not a control | No reset on hang | Health task feed path |
| 2 | Watchdog fed in an interrupt | Refresh in the timer ISR | Interrupt continues after task hang | No reset | Move the feed to a task |
| 3 | Debugger freezes the watchdog | Reset never observed under debug | Debug freeze | False measurement | Detach or disable freeze |
| 4 | Timing measured with printf | Log timestamps vary | Logging latency | Wrong jitter | GPIO toggle |
| 5 | Mean latency reported as worst case | No distribution | Tail ignored | Deadline miss | p99 and max |
| 6 | Priority inversion untested | No mutex in the design | Hidden blocking | Latency spike | Demonstrate and bound |
| 7 | Nested mutexes | Two locks held at once | Inheritance assumption violated | Unbounded inversion | Lock ordering, one lock |
| 8 | Safe state unverified | Only code inspection | Peripheral keeps driving | Output not safe | Scope capture |
| 9 | Unsigned image accepted | Verification optional | Configuration error | Malicious firmware | Test rejection |
| 10 | Debug port open | Production config not applied | Physical access | Key extraction | Lock and audit |
| 11 | Update not power-loss tested | Only successful updates | Partial write | Bricked device | Cut power at 25/50/75% |
| 12 | Anti-rollback absent | Version check only | Older image accepted | Vulnerability reintroduced | Hardware counter |
| 13 | Stack overflow | High-water mark near the limit | No guard | Memory corruption | MPU guard, larger stack |
| 14 | Heap fragmentation | Allocation fails after hours | Dynamic allocation | Random reset | Static allocation |
| 15 | Queue overflow silent | Consumer slower than producer | No back-pressure | Data loss | Overflow check and log |
| 16 | Clock failure unhandled | External oscillator only | No fallback | No execution | Clock monitor |
| 17 | Release build differs from tested build | Optimisation changed after test | Different timing | Invalid evidence | Rebuild and compare hashes |
| 18 | Option bytes erased in production | Mass erase before programming | Watchdog or debug lock lost | Unsafe configuration | Program and verify option bytes |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The timing capture is representative | The jitter estimate depends on it | A different workload, a different temperature | Repeat under load and over temperature |
| 2 | The logic analyser's timebase is accurate | The interval measurement depends on it | A drifting timebase | Check the specification and calibrate |
| 3 | The GPIO toggle is fast | The timestamp is the toggle | A slow port, an interrupt latency | Measure the toggle with a scope |
| 4 | The watchdog clock is within tolerance | The timeout budget depends on it | Temperature drift, supply variation | Measure over temperature |
| 5 | The health task cannot hang | The feed path depends on it | A bug in the health task | Add a second monitor |
| 6 | The critical section is the worst case | The blocking bound depends on it | A longer section elsewhere | Inspect and measure all sections |
| 7 | The safe state is reached after reset | The total time depends on it | A peripheral that keeps driving | Capture through the reset |
| 8 | The bootloader verifies before executing | Secure boot depends on it | A bypass, a debug mode | Test with an invalid image |
| 9 | The anti-rollback counter is protected | The version policy depends on it | A writable counter | Attempt to write it |
| 10 | The update writes are atomic enough | Recovery depends on it | A partial erase | Cut power at multiple points |
| 11 | The RTOS's priority inheritance is sufficient | The blocking bound depends on it | Nested mutexes | Measure with the actual lock pattern |
| 12 | The build is reproducible | The binary is identified | A different toolchain | Rebuild and compare hashes |
| 13 | The option bytes are part of the configuration | Secure boot and watchdog depend on them | A mass erase, a different programmer | Read back and compare |
| 14 | The timer tick is accurate | The scheduling period depends on it | Clock drift, a wrong prescaler | Measure the tick with a scope |

## 12. Exercises

### Exercise 1. Measure sampling jitter (application)

**Deliverable:** a logic-analyser capture of 1,000 sampling intervals, with the mean, standard deviation, minimum, maximum, p99, and peak-to-peak jitter.

**Acceptance criteria:** the instrument, its sample rate, its resolution, and its timebase uncertainty are stated; the distribution is plotted; the result is compared with a written requirement; the report distinguishes the mean from the tail.

**Time:** 45 minutes.

### Exercise 2. Response-time analysis (calculation)

**Deliverable:** a fixed-priority response-time bound for a three-task set, with the interference terms and the deadline comparison.

**Acceptance criteria:** the formula is shown; the ceiling terms are computed correctly; the blocking term is included or explicitly set to zero with a reason; the verdict is stated per task.

**Time:** 45 minutes.

### Exercise 3. Demonstrate priority inversion (application)

**Deliverable:** a measurement of the high-priority task's response time with and without priority inheritance, with the critical-section duration and the medium-task duration.

**Acceptance criteria:** the mechanism is explained; the measured values are consistent with the durations; the improvement is quantified; the limitation of the RTOS's inheritance implementation is stated.

**Time:** 60 minutes.

### Exercise 4. Watchdog design and test (design)

**Deliverable:** a watchdog feed-path diagram, a timeout justification, and a test that resets a hung task in each supervised task.

**Acceptance criteria:** the feed path is independent of the supervised tasks; the timeout is longer than the longest legitimate stall and shorter than the time to harm; the reset is captured; the safe state is verified.

**Time:** 60 minutes.

### Exercise 5. Safe-state verification (application)

**Deliverable:** a scope capture of every output at its safe value within the required time, including the reset and the boot.

**Acceptance criteria:** every output in the safe-state table is captured; the time is measured from the fault; the output is verified after the reset; the residual state is recorded.

**Time:** 45 minutes.

### Exercise 6. Secure boot test (application)

**Deliverable:** a test log showing that a signed image boots, an unsigned image is rejected, a corrupted image is rejected, and the debug port is locked.

**Acceptance criteria:** at least 20 attempts for the unsigned-image rejection; the corruption is a single-byte change; the debug lock is verified with the vendor tool; the result is recorded with the image hash.

**Time:** 60 minutes.

### Exercise 7. Update and rollback (design)

**Deliverable:** an update procedure, a power-loss test plan at 25%, 50%, and 75% of the write, and a rollback demonstration.

**Acceptance criteria:** the procedure uses a real power switch; the device boots a valid image after every interruption; the rollback is triggered by an unconfirmed image; the anti-rollback counter is tested.

**Time:** 90 minutes.

### Exercise 8. Fault matrix (design)

**Deliverable:** a fault-injection matrix with at least eight faults, each with detection, detection time, response, recovery, and residual state.

**Acceptance criteria:** each row has a capture; at least one fault is a clock failure; at least one is a queue overflow; the detection times are measured; the residual state is verified.

**Time:** 90 minutes.

## 13. Self-check answers

### 13.1 Why is a mean sampling period not enough?

Because the control loop and the safety monitor see the tail. A mean of 2000 µs with a maximum of 2019 µs means that some samples arrive 19 µs late. If the deadline is 2000 µs, the mean passes and the tail fails. The distribution, especially the maximum and the p99, is the evidence.

### 13.2 Compute the response time for a task with C = 420 µs interfered with by a task with C = 180 µs and T = 2000 µs.

The response time is \(R = C + \lceil R/T \rceil C_h\). Start with R = 420. The ceiling of 420/2000 is 1, so R = 420 + 180 = 600 µs. The next iteration gives the same value, so the bound is 600 µs.

### 13.3 What is priority inversion?

It is the condition in which a high-priority task waits for a resource held by a low-priority task, and a medium-priority task preempts the low-priority task, extending the wait. The high-priority task is effectively blocked by the medium-priority task, which is not sharing the resource.

### 13.4 How does priority inheritance bound the inversion?

The low-priority task inherits the priority of the high-priority task while it holds the mutex, so the medium-priority task cannot preempt it. The high-priority task waits for the remaining critical section, not for the medium-priority execution.

### 13.5 Why is a watchdog fed by an interrupt not a watchdog?

Because the interrupt can continue to run when the task it is supposed to supervise is hung. The watchdog is refreshed, so it never fires. The feed path supervises the interrupt, not the task.

### 13.6 What is the difference between a test swap and a permanent swap?

A test swap boots a new image once and reverts if the image does not confirm itself. A permanent swap makes the new image the default after confirmation. The test swap is the rollback mechanism; the permanent swap is the final state.

### 13.7 What must a power-loss update test cover?

Every point at which the write can be interrupted, including the erase, the write, the verification, and the boot-pointer update. A test at 25%, 50%, and 75% of the write is a sample, not a proof; the bootloader's recovery mechanism must also be inspected.

### 13.8 Why is the debug port part of the secure boot claim?

Because an open debug port allows an attacker with physical access to read the keys, halt the processor, and reprogram the device. Secure boot verifies the image at boot, but it does not prevent a debug access that bypasses the boot. The production configuration must lock the port and the unlock procedure must be controlled.

### 13.9 What does the safe state have to specify?

Every output, its safe value, the time to reach it, and the hazard if it is wrong. The safe state is a design decision, not an error handler. It must be verified with an instrument, through the reset and the boot.

### 13.10 Rubric for Exercise 4

A strong watchdog design has an independent feed path, a timeout justified against the longest legitimate stall and the time to harm, a test that hangs each supervised task, and a capture of the reset and the safe state. A weak design feeds the watchdog from the main loop and tests only a total hang.

### 13.11 Rubric for Exercise 7

A strong update plan tests a real power loss at multiple points, demonstrates rollback from an unconfirmed image, and tests the anti-rollback counter. A weak plan tests only a successful update and assumes that the bootloader handles the rest.

### 13.12 What is the difference between a watchdog and a window watchdog?

A simple watchdog resets the system if it is not refreshed within the timeout. A window watchdog also resets if it is refreshed too early, which catches a task that is running too fast or a refresh that is stuck in a tight loop. The window makes the watchdog sensitive to both late and early refreshes.

### 13.13 Why is a build ID not enough to identify a firmware image?

Because two builds with different toolchains, flags, or source files can share a build ID if the ID is derived from a version string rather than the content. The image hash identifies the binary; the build ID identifies the build process. Both are needed, and the hash is the stronger identifier.

### 13.14 Rubric for Exercise 8

A strong fault matrix has a capture for every row, a measured detection time, and a verified residual state. It includes faults that are not obvious, such as a clock failure and a queue overflow. A weak matrix lists faults without evidence and covers only the nominal path.

## 14. Interview questions

### 14.1 How do you measure firmware timing?

A strong answer uses a GPIO toggle and a logic analyser, or a cycle counter, and reports the distribution. It notes that printf changes the timing and that the debugger can change the behaviour. A weak answer uses a timestamp from a log.

### 14.2 What is the worst-case response time of a task?

It is the execution time plus the interference from higher-priority tasks plus the blocking from lower-priority tasks. A strong answer gives the formula and notes that the blocking term must be measured, not assumed. A weak answer says it is the execution time.

### 14.3 Why can a watchdog fail to protect?

Because it is fed by the task it supervises, because it is disabled, because its timeout is too long, because it is frozen by the debugger, or because the reset leads to an unsafe state. A strong answer names several mechanisms. A weak answer says "it is just a timer."

### 14.4 What does secure boot actually verify?

The signature of the image against a public key stored in a trusted area, and the integrity of the image against the signed hash. A strong answer adds the root of trust, the anti-rollback counter, and the debug lock. A weak answer says it verifies the firmware.

### 14.5 What is the difference between a safe state and a fail-safe state?

A safe state is a state in which the risk is acceptable. A fail-safe state is a state that minimises harm on a fault. They are often the same, but not always: a fail-safe state may be a controlled stop, while a safe state may be a continued operation with reduced function.

### 14.6 How do you test rollback?

Boot an image that does not confirm itself, and verify that the bootloader reverts to the previous image. A strong answer also tests a corrupted image, an unsigned image, an older image, and a power loss during the update. A weak answer reboots the device and calls it a rollback.

### 14.7 Why is static allocation preferred in safety-critical firmware?

Because dynamic allocation can fail after long operation due to fragmentation, and the failure is hard to reproduce. A strong answer notes that static allocation makes the memory budget deterministic. A weak answer says it is faster.

### 14.8 What is the difference between a mutex and a semaphore?

A mutex has ownership and supports priority inheritance; a semaphore is a signalling primitive without ownership. A strong answer notes that using a binary semaphore where a mutex is needed loses the inheritance. A weak answer says they are the same.

### 14.9 How do you verify the safe state?

With an instrument on every output, through the fault, the reset, and the boot. A strong answer notes that a peripheral can keep driving a pin after the CPU resets. A weak answer reads the code.

### 14.10 What is the role of the SBOM in firmware?

It lists the components, versions, and licences in the image, so that a vulnerability can be traced to a device and a patch can be assessed. A strong answer connects it to IEC 81001-5-1 and the FDA cybersecurity guidance. A weak answer calls it a list of files.

### 14.11 What would you do if the watchdog resets during a legitimate long operation?

A strong answer measures the longest legitimate operation, justifies the timeout against it, and considers a window watchdog or a two-stage timeout. It does not simply increase the timeout without checking the time to harm. A weak answer disables the watchdog.

### 14.12 How do you make a firmware build reproducible?

Pin the toolchain, the libraries, and the build flags; record the commit, the linker script, and the image hash; build in a container with a recorded digest; and compare the rebuilt image with the released image. A strong answer notes that the compiler version and the optimisation flags can change the binary.

## 15. Key terms

- **A/B slot** — two image slots for update and rollback.
- **Anti-rollback** — prevention of installing an older image.
- **Blocking** — waiting for a resource or an event.
- **Boot time** — time from reset to application ready.
- **Brown-out** — supply voltage below the operating threshold.
- **Deadline** — time by which a task must complete.
- **Fail-safe state** — state that minimises harm on a fault.
- **Fault matrix** — table of faults, detection, response, and residual state.
- **Fault record** — persisted record of a fault.
- **FreeRTOS** — open-source real-time operating system.
- **Heartbeat** — periodic signal that a task is alive.
- **IEC 62304** — medical device software lifecycle standard.
- **Independent watchdog** — watchdog clocked independently of the CPU.
- **Jitter** — variation in period or latency.
- **MCUboot** — open-source secure bootloader.
- **MPU** — memory protection unit.
- **Mutex** — mutual-exclusion lock with ownership.
- **Period** — interval between task releases.
- **PL** — performance level in ISO 13849.
- **Preemption** — suspension of a running task by a higher-priority one.
- **Priority** — scheduling rank.
- **Priority ceiling** — protocol that elevates a lock's priority.
- **Priority inheritance** — temporary elevation of a lock holder's priority.
- **Priority inversion** — high-priority task blocked by a low-priority task.
- **Queue** — buffer between tasks.
- **Reset latency** — time from fault to reset assertion.
- **Response time** — time from arrival to completion.
- **Revert** — return to the previous image.
- **Root of trust** — immutable anchor of the boot chain.
- **RTOS** — real-time operating system.
- **Safe state** — state in which the risk is acceptable.
- **SBOM** — software bill of materials.
- **Secure boot** — cryptographic verification before execution.
- **Semaphore** — signalling primitive without ownership.
- **SIL** — safety integrity level in IEC 61508.
- **Stack high-water mark** — maximum stack usage observed.
- **Static allocation** — memory allocated at compile time.
- **Test swap** — boot a new image once before confirming.
- **TrustZone-M** — Arm security extension for Cortex-M.
- **WCET** — worst-case execution time.
- **Window watchdog** — watchdog that must be refreshed within a window.
- **Option byte** — non-volatile configuration bits on a microcontroller.
- **Read-out protection** — configuration that prevents reading flash memory.
- **Image hash** — digest that identifies a firmware binary.
- **Build ID** — identifier of the build process, not necessarily of the binary.
- **Interference** — delay caused by higher-priority tasks.
- **Release build** — optimised build without debug assertions.
- **Debug freeze** — debugger behaviour that halts a peripheral with the core.
- **Critical section** — code that must run without preemption.
- **Lock ordering** — rule that prevents deadlock by fixing the acquisition order.
- **Heap fragmentation** — loss of usable memory after repeated allocation and free.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Standards and regulation

- IEC 62304:2006+AMD1:2015, Medical device software — Software life cycle processes. https://www.iso.org/standard/64686.html
- IEC 81001-5-1:2021, Health software and health IT systems safety, effectiveness and security — Part 5-1: Security. https://webstore.iec.ch/
- ISO 14971:2019, Medical devices — Application of risk management to medical devices. https://www.iso.org/standard/72704.html
- ISO 13849-1, Safety of machinery — Safety-related parts of control systems. https://www.iso.org/
- IEC 61508, Functional safety of electrical/electronic/programmable electronic safety-related systems. https://webstore.iec.ch/
- FDA, Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions. https://www.fda.gov/media/119933/download
- FDA, Cybersecurity in Medical Devices — Frequently Asked Questions. https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity-medical-devices-frequently-asked-questions-faqs

### Documentation

- FreeRTOS Reference Manual. https://www.freertos.org/media/2025/FreeRTOS_Reference_Manual_V8.2.1.pdf
- FreeRTOS User Guide. https://docs.aws.amazon.com/freertos/latest/userguide/freertos-ug.pdf
- MCUboot design document. https://github.com/mcu-tools/MCUboot/blob/master/docs/design.md
- Zephyr MCUboot image control API. https://docs.zephyrproject.org/latest/doxygen/html/group__mcuboot__api.html
- Arm TrustZone for Cortex-M. https://developer.arm.com/
- STM32 independent watchdog application notes. https://www.st.com/

### Methods

- Liu, C. L., and Layland, J. W. (1973). Scheduling algorithms for multiprogramming in a hard-real-time environment. Journal of the ACM.
- Joseph, M., and Pandya, P. (1986). Finding response times in a real-time system. The Computer Journal.
- Sha, L., Rajkumar, R., and Lehoczky, J. P. (1990). Priority inheritance protocols: An approach to real-time synchronization. IEEE Transactions on Computers.
- ISO 10218-1:2025, Robotics — Safety requirements — Part 1: Industrial robots. https://www.iso.org/standard/73933.html
- IEC 80601-2-77:2019+AMD1:2023, Robotically assisted surgical equipment. https://www.iso.org/standard/68473.html
- FDA, Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions. https://www.fda.gov/media/185050/download

> **Note:** The standard texts are paywalled. The documentation and datasheets above were used to check the teaching values. Confirm every clause and requirement against the current edition before relying on it in a regulatory submission.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_sensor-and-signal-labs.md`](04_sensor-and-signal-labs.md)
- Next topic: [`06_emc-labs.md`](06_emc-labs.md)
- Sibling: [`07_hil-labs.md`](07_hil-labs.md)
- Sibling: [`09_evidence-dossier.md`](09_evidence-dossier.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
