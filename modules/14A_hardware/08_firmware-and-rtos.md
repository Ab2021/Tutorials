# Firmware and RTOS

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 08 of 14
> Estimated study time: 4 hours
> Prerequisites: 05, 07
> Version 2026.3

## 1. Why this topic exists

Firmware is the layer that turns silicon into a medical device. It reads the sensor, filters the signal, runs the model, checks the safety limit, drives the output, and logs what happened. It also boots the device, updates itself, and decides what to do when something goes wrong. A hardware design that is excellent and a firmware design that is poor produces a device that is unreliable, unverifiable, or unsafe.

The industry problem is that firmware behaviour is invisible in the data and in the design review. A device that resets once a week, a watchdog that hides a hung task, an update that bricks a unit, or a priority inversion that delays a control loop by 30 ms can all pass a bench test and appear in the field as unexplained gaps or late alarms. Three decisions depend on firmware literacy:

1. **Whether the timing claim is supported.** A real-time operating system (RTOS) provides mechanisms, not guarantees. The deadlines must be analysed and measured, not assumed.
2. **Whether the failure behaviour is safe.** A watchdog, a brown-out detector, and a fail-safe state determine what the device does when something goes wrong. A device that reboots into an unsafe state is worse than one that stops.
3. **Whether the deployed firmware is the validated firmware.** Secure boot, signed updates, rollback, and version logging determine whether the device in the field is the device that was tested.

This topic covers the firmware lifecycle, boot and initialisation, RTOS scheduling, interrupt handling, memory management, watchdogs, logging, secure boot, and update mechanisms. It assumes the processor material from Topic 05 and the power material from Topic 07. It does not teach a specific RTOS API. It teaches the properties that determine whether firmware is safe and verifiable.

A useful framing: firmware is a state machine with a clock. The state machine must be safe in every state, including the states after a reset, a brown-out, a watchdog expiry, and a failed update. The clock must be fast enough for the deadlines and slow enough for the power budget.

## 2. Learning outcomes

By the end of this topic you can:

- Diagram the boot sequence from reset to the main application, including the bootloader, the vector table, and the initialisation.
- Compute the utilisation and the worst-case response time of a fixed-priority task set.
- Explain priority inversion and the mechanisms that bound it.
- Design a watchdog strategy that detects a hung task without hiding a failure.
- Design a firmware update with A/B partitions, signature verification, and rollback.
- Identify the firmware states after a brown-out, a reset, and a failed update, and verify that each is safe.
- Write a traceability matrix from requirements to tests for a firmware module.
- Critique a claim that a device is "real-time" without a response-time analysis.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

Firmware runs in a loop of events and tasks. The boot sequence establishes the state; the scheduler decides which task runs; the interrupts preempt the scheduler; the watchdog observes the loop; the update mechanism replaces the image. The device is safe only if every state, including the exceptional ones, has a defined behaviour.

```text
   RESET --> BOOTLOADER --> VERIFY IMAGE --> INIT HARDWARE --> START RTOS
                |               |                                  |
                |               |                                  v
                |               |                          +---------------+
                |               |                          |   SCHEDULER   |
                |               |                          | tasks, queues |
                |               |                          +-------+-------+
                |               |                                  |
                |               |          +-----------------------+------------------+
                |               |          |                       |                  |
                |               |          v                       v                  v
                |               |      CONTROL TASK           COMMS TASK         LOG TASK
                |               |          |                       |                  |
                |               |          +-----------+-----------+------------------+
                |               |                      |
                |               |                      v
                |               |                 WATCHDOG
                |               |                      |
                |               v                      v
                |          FAIL SAFE <--------- RESET / BROWN-OUT
                |
                +--> UPDATE SLOT --> VERIFY --> SWAP --> REBOOT
```

The model has four properties:

| Property | Question | Evidence |
|---|---|---|
| Safety | Is every state safe, including reset and fault? | State machine, fault injection |
| Timeliness | Does every task meet its deadline? | Response-time analysis, trace |
| Integrity | Is the running image the validated image? | Secure boot, hash, version |
| Recoverability | Can a failed update or a brown-out be recovered? | A/B partitions, rollback test |

The timing analysis uses rate-monotonic scheduling for fixed-priority tasks. The utilisation is:

\[
U = \sum_i \frac{C_i}{T_i}
\]

where \(C_i\) is the worst-case execution time and \(T_i\) the period. For a set of independent periodic tasks with rate-monotonic priorities, the utilisation bound is:

\[
U \le n(2^{1/n} - 1)
\]

For three tasks, the bound is 0.780; for four, 0.757; as \(n\) grows, it approaches \(\ln 2 = 0.693\). A task set below the bound is guaranteed schedulable; a set above it may still be schedulable, but it requires an exact response-time analysis.

The response time of task \(i\) is the smallest \(R\) satisfying:

\[
R_i = C_i + \sum_{j \in hp(i)} \left\lceil \frac{R_i}{T_j} \right\rceil C_j
\]

where \(hp(i)\) is the set of higher-priority tasks. The equation is solved by iteration, starting from \(R_i = C_i\). If the iteration converges to a value below the deadline, the task meets its deadline; if it exceeds the deadline or diverges, it does not.

### 3.1 The task model

A task has a period, a worst-case execution time, a deadline, a priority, and a stack. The model assumes that tasks are independent, that the worst case is known, and that the scheduler is preemptive. Each assumption can be violated in practice.

| Assumption | Violation | Consequence |
|---|---|---|
| Tasks are independent | Shared mutex or queue | Blocking; priority inversion |
| WCET is known | Caches, branches, input-dependent loops | Underestimated response time |
| Scheduler is preemptive | Interrupts disabled in a critical section | Unbounded delay |
| Periods are fixed | Event-driven tasks | Bursts; missed deadlines |
| Deadlines equal periods | Arbitrary deadlines | Different analysis |

The analysis is only as good as the model. A task set that passes the utilisation bound but shares a mutex can still miss a deadline.

### 3.2 Boot and initialisation

The boot sequence is a state machine with its own failure modes. The bootloader verifies the image, initialises the hardware, and jumps to the application. If the verification fails, the bootloader must enter a safe state, not a random one.

| Stage | Action | Failure mode |
|---|---|---|
| Reset | Vector fetch, stack pointer | Corrupt vector table |
| Bootloader | Verify signature and CRC | Invalid image |
| Clock init | Configure PLL and peripherals | Wrong clock; peripheral misconfiguration |
| Memory init | Enable RAM, copy data | RAM failure |
| RTOS start | Create tasks, start scheduler | Task creation failure |
| Application | Run the main loop | Unhandled exception |

A bootloader that does not verify the image can run a corrupted image. A bootloader that verifies but has no rollback can brick the device after a failed update. A bootloader that jumps to the application without initialising the watchdog leaves the device unprotected.

### 3.5 The three firmware states that matter

Every firmware design has three states that determine its safety:

1. **The normal state.** The scheduler runs, the tasks meet their deadlines, and the outputs are within limits. The evidence is the timing trace and the functional tests.
2. **The fault state.** A task misses a deadline, a watchdog expires, a brown-out occurs, or a fault handler runs. The device must enter a safe state within a bounded time. The evidence is the fault-injection test.
3. **The update state.** The device is writing or verifying an image. The outputs must be safe, the update must be atomic, and a power loss must be recoverable. The evidence is the power-loss test at each stage.

A design that specifies only the normal state has not specified the device. The fault and update states are where the safety case is won or lost.

### 3.3 The watchdog

A watchdog is a timer that resets the processor if the firmware does not service it within a window. It detects a hung task, an infinite loop, or a scheduler failure. It does not detect a task that runs but produces wrong results.

| Watchdog type | Behaviour | Strength | Weakness |
|---|---|---|---|
| Simple | Reset if not serviced | Simple | A hung loop that services it hides the fault |
| Windowed | Reset if serviced too early or too late | Detects a runaway loop | Requires a correct window |
| Independent | Separate clock and oscillator | Detects a clock failure | Cost; board space |
| External | Separate IC | Independent of the MCU | Cost; size |

A watchdog serviced by a low-priority task can hide a hung high-priority task, because the low-priority task continues to run. The watchdog should be serviced by the highest-priority task, or by a supervisor that checks that each task has run within its period.

### 3.4 Secure boot and updates

Secure boot verifies the image's signature before executing it. The verification uses a public-key signature and a root of trust in the boot ROM or in one-time-programmable memory. If the signature is invalid, the bootloader does not execute the image.

An update writes a new image to a inactive slot, verifies it, and swaps the active slot. The swap must be atomic, or a power loss during the swap can leave the device with no valid image. A/B partitions with a rollback flag solve this: the bootloader tries the new image, and if it fails to boot or fails a self-test, it reverts to the old image.

| Update scheme | Atomicity | Rollback | Storage cost |
|---|---|---|---|
| In-place | No | No | Low |
| A/B with swap | Yes | Yes | 2× image |
| A/B with metadata | Yes | Yes | 2× image plus metadata |
| External programmer | Yes | No | None; requires access |

The update must also be authenticated. An unsigned update can be replaced by a malicious image, which is a cybersecurity risk and a safety risk.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Firmware | Software embedded in a device | Confused with hardware | None |
| Bootloader | Code that runs before the application | Confused with the application | None |
| Vector table | Addresses of interrupt handlers | Confused with the stack | Addresses |
| RTOS | Real-time operating system | Assumed to guarantee deadlines | None |
| Task | Independent thread of execution | Confused with a process | None |
| Period | Time between task releases | Confused with deadline | s |
| WCET | Worst-case execution time | Confused with average | s |
| Deadline | Time by which a task must finish | Confused with period | s |
| Utilisation | Sum of C/T | Confused with CPU load | % |
| Response time | Time from release to completion | Confused with execution time | s |
| Priority | Rank that determines preemption | Confused with importance | Integer |
| Priority inversion | Low-priority task blocks a high-priority one | Confused with starvation | None |
| Priority inheritance | Low-priority task inherits the waiter's priority | Confused with priority ceiling | None |
| Mutex | Lock that protects a shared resource | Confused with a semaphore | None |
| Semaphore | Counter that signals events | Confused with a mutex | None |
| Queue | Buffer between tasks | Confused with a stack | Messages |
| Interrupt | Asynchronous event that preempts code | Confused with a task | None |
| ISR | Interrupt service routine | Confused with a task | None |
| Watchdog | Timer that resets a hung processor | Confused with a timer | s |
| Windowed watchdog | Watchdog with a service window | Confused with a simple watchdog | s |
| Brown-out | Supply drop below the operating minimum | Confused with a reset | V |
| Reset cause | Register that records why the device reset | Ignored in logs | Code |
| Secure boot | Verification of the image at boot | Confused with code signing | None |
| Code signing | Signature over the image | Confused with encryption | None |
| A/B partition | Two image slots for atomic update | Confused with a backup | None |
| Rollback | Reverting to the previous image | Confused with a factory reset | None |
| CRC | Cyclic redundancy check | Confused with a cryptographic hash | None |
| Hash | Cryptographic digest | Confused with a checksum | None |
| SBOM | Software bill of materials | Confused with a licence list | None |
| Fail-safe | State that is safe when a fault occurs | Confused with fail-operational | None |
| Fail-operational | State that continues to operate after a fault | Confused with fail-safe | None |
| Traceability | Link from requirement to test | Confused with documentation | None |
| MISRA C | Coding standard for safety C | Assumed to guarantee safety | None |
| Static allocation | Memory allocated at compile time | Confused with a global | None |
| Stack overflow | Stack grows beyond its region | Confused with a memory leak | None |

## 5. The mechanics, step by step

The procedure below takes a firmware requirement to a verified, updatable image. Each step states its input, output, and rationale. The required edge cases are mapped to the steps and summarised in Section 5.13.

### Step 1. Define the firmware requirements

**Input:** the device requirements, the risk analysis, and the hardware interfaces.

**Output:** a requirements specification with functional and safety requirements.

**Rationale:** firmware requirements are the basis of the traceability matrix and the verification. A requirement that says "the device shall be safe" is not testable; a requirement that says "the control task shall complete within 8 ms of the sensor interrupt, or the device shall enter the safe state within 2 ms" is testable.

Each requirement needs an identifier, a rationale, a verification method, and a safety classification. The classification follows the risk analysis and determines the evidence.

### Step 2. Classify the software

**Input:** the risk analysis and the requirements.

**Output:** a software safety class (A, B, or C) per IEC 62304.

**Rationale:** the class determines the rigour of the lifecycle. Class A requires basic planning; class B requires architecture and unit verification; class C requires detailed design, unit verification, integration testing, and regression testing.

The class is per software item, not per device. A device may have a class C control loop and a class A logging function. The classification drives the effort, not the marketing.

### Step 3. Design the architecture

**Input:** the requirements and the hardware.

**Output:** a task diagram, a memory map, and an interface specification.

**Rationale:** the architecture determines the timing and the failure behaviour. The task diagram shows the periods, priorities, and communication paths. The memory map shows the code, data, stack, and heap regions. The interface specification shows the registers, the interrupts, and the shared resources.

A design with a single super-loop is simpler to verify but harder to extend. A design with an RTOS is more flexible but adds scheduling and synchronisation risks. The choice depends on the complexity and the timing requirements.

### Step 4. Design the boot sequence

**Input:** the hardware, the bootloader, and the update scheme.

**Output:** a boot state machine and a failure response for each state.

**Rationale:** the boot sequence is where the device is most vulnerable. A corrupt image, a failed verification, a brown-out during boot, or a missing peripheral can leave the device in an undefined state. Each state must have a defined response.

The bootloader should verify the image, initialise the watchdog early, and enter a safe state if the verification fails. The application should check the reset cause and log it.

### Step 5. Design the task set and priorities

**Input:** the functions, the periods, the deadlines, and the WCET estimates.

**Output:** a task table with periods, deadlines, priorities, and WCETs.

**Rationale:** the priorities should follow the deadlines, not the importance. A task with a short deadline and a small WCET should have a higher priority than a task with a long deadline, even if the latter is more "important". Rate-monotonic assignment is optimal for fixed-priority periodic tasks with deadlines equal to periods.

The task table is the input to the response-time analysis. A task that is not in the table is a task that is not analysed.

### Step 6. Compute utilisation and response times

**Input:** the task table.

**Output:** utilisation, response times, and a schedulability verdict.

**Rationale:** the utilisation bound is a quick check; the response-time analysis is the exact check. A task set below the bound is schedulable; a task set above it may still be schedulable, but it requires the exact analysis.

The analysis assumes independent tasks and known WCETs. If the tasks share a mutex, the blocking time must be added to the response time. If the WCET is underestimated, the analysis is invalid.

### Step 7. Design the interrupt and synchronisation model

**Input:** the interrupts, the shared resources, and the task table.

**Output:** an interrupt priority table, a locking strategy, and a blocking analysis.

**Rationale:** interrupts preempt tasks and can delay the highest-priority task. A long ISR or a disabled interrupt section can break the response-time analysis. The ISR should be short, defer the work to a task, and not block.

A shared resource protected by a mutex introduces blocking. Priority inheritance bounds the blocking to the longest critical section of a lower-priority task. A priority ceiling protocol bounds it further, at the cost of complexity.

### Step 8. Design the watchdog strategy

**Input:** the task table and the failure modes.

**Output:** a watchdog type, a timeout, a service policy, and a failure response.

**Rationale:** the watchdog must detect a hung task without hiding a failure. A simple watchdog serviced by a low-priority task can be fooled. A windowed watchdog serviced by a supervisor that checks each task's progress is more robust.

The timeout must be longer than the worst-case loop time and shorter than the safety limit. A timeout that is too short causes spurious resets; one that is too long allows an unsafe condition to persist.

### Step 9. Design the memory model

**Input:** the code size, the data size, the stack requirements, and the heap policy.

**Output:** a memory map and a stack budget.

**Rationale:** a stack overflow corrupts memory and causes unpredictable behaviour. A heap allocation in a real-time path causes fragmentation and unpredictable latency. The memory model should use static allocation for the real-time tasks and a bounded pool for the rest.

The stack budget must include the worst-case call depth, the interrupt nesting, and the local arrays. A stack analysis tool or a watermark measurement provides the evidence.

### Step 10. Design the update mechanism

**Input:** the image size, the flash size, the update channel, and the security requirement.

**Output:** an A/B partition layout, a signature scheme, a rollback policy, and a power-loss response.

**Rationale:** an update is a state transition that can fail. The device must be able to recover from a power loss, a corrupted image, or a failed self-test. The A/B layout provides a fallback; the signature provides integrity; the rollback provides recovery.

The update must be atomic. A write to the inactive slot, a verification, and a metadata update are the minimum. The metadata update is the commit point; if it fails, the old image remains active.

### Step 11. Implement and review

**Input:** the design and the coding standard.

**Output:** source code, static analysis results, and a code review record.

**Rationale:** the implementation must follow the design and the coding standard. MISRA C or an equivalent standard reduces the risk of undefined behaviour, but it does not guarantee safety. The code review checks the design intent, the error handling, and the concurrency.

The review should include the interrupt handlers, the critical sections, the error paths, and the update path. These are the places where bugs hide.

### Step 12. Verify and document

**Input:** the implementation and the requirements.

**Output:** unit tests, integration tests, timing measurements, and a traceability matrix.

**Rationale:** verification demonstrates that the implementation meets the requirements. The tests must include the normal path, the error path, the boundary conditions, and the fault injection. The timing measurements must be on the target, with the production build.

The traceability matrix links each requirement to a test and a result. A requirement without a test is a requirement without evidence.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 6, 12 | CPU utilisation exceeds 100% | Missed deadlines; queue growth | Lost samples |
| Common-mode rejection failure | 1 | Analogue issue | N/A | N/A |
| Electrode offset | 1 | Analogue issue | N/A | N/A |
| Clock jitter | 7, 12 | Interrupt timing variation | Trace | Late response |
| Brown-out | 4, 10 | Supply drop during boot or update | Reset cause register | Corrupt state |
| Watchdog reset loop | 8, 12 | Firmware fails to service watchdog | Reset counter | Device unavailable |
| Thermal throttling | 6 | Clock reduction at high temperature | Temperature log | Missed deadlines |
| Firmware update failure | 10 | Incomplete write; bad signature; no rollback | Version and checksum | Bricked device |
| Calibration drift | 12 | Coefficients age | Reference check | Biased output |

The firmware edge cases are dominated by brown-out, watchdog reset loops, and update failure. The analogue edge cases enter through the requirements and the sensor interface.

### 5.14 RTOS selection table

| Requirement | Bare-metal super-loop | Cooperative scheduler | Preemptive RTOS |
|---|---|---|---|
| Hard deadline | Possible with careful design | Possible | Yes, with analysis |
| Many tasks | Difficult | Moderate | Yes |
| Priority inversion | Not applicable | Possible | Managed with inheritance |
| Memory footprint | Smallest | Small | Larger |
| Certification evidence | Simpler | Moderate | Larger |
| Determinism | High if simple | Moderate | Depends on the kernel |

The RTOS is not automatically better. A simple super-loop with a fixed schedule can be more deterministic and easier to certify than a preemptive RTOS with a complex task set. The choice depends on the complexity and the timing requirements.

### 5.15 Update scheme comparison

| Scheme | Atomic | Rollback | Storage | Complexity |
|---|---|---|---|---|
| In-place | No | No | 1× | Low |
| A/B swap | Yes | Yes | 2× | Moderate |
| A/B with metadata | Yes | Yes | 2× plus | Moderate |
| External programmer | Yes | No | None | Low; requires access |
| Dual-bank with copy | Yes | Yes | 2× | High |

The A/B scheme is the default for a device that must be updated in the field. The in-place scheme is acceptable only if the device can be recovered by a programmer.

### 5.16 Traceability matrix template

| Requirement | Design element | Code unit | Test | Result |
|---|---|---|---|---|
| REQ-001 | Control task | control.c | TC-001 | Pass |
| REQ-002 | Watchdog supervisor | wdt.c | TC-002 | Pass |
| REQ-003 | Update rollback | boot.c | TC-003 | Pass |
| REQ-004 | Brown-out safe state | power.c | TC-004 | Pass |

A matrix with empty cells is a matrix with missing evidence. The empty cells are the audit findings.

### 5.17 RTOS kernel objects and their costs

| Object | Purpose | Cost | Failure mode |
|---|---|---|---|
| Task | Independent execution | Stack plus control block | Stack overflow |
| Mutex | Mutual exclusion | Blocking; priority inheritance | Priority inversion |
| Semaphore | Event signalling | Blocking | Lost wake-up |
| Queue | Data transfer | Copy or pointer | Overflow |
| Event group | Multiple signals | Bit operations | Stale bits |
| Timer | Periodic or one-shot | Callback latency | Callback too long |
| Memory pool | Fixed-size allocation | Fragmentation-free | Exhaustion |

A mutex is not a semaphore. A mutex has an owner and supports priority inheritance; a semaphore does not. Using a semaphore as a mutex can cause priority inversion that the kernel cannot fix.

A queue's depth must be sized for the worst-case burst. A queue that overflows drops data, which may be a safety event for a control command and a minor event for a log message. The overflow policy must be explicit.

### 5.18 Memory and stack analysis

| Region | Contents | Sizing method | Failure |
|---|---|---|---|
| Code | Instructions | Linker map | Overflow into data |
| Data | Initialised globals | Linker map | None |
| BSS | Zero-initialised globals | Linker map | None |
| Heap | Dynamic allocation | Pool size | Fragmentation |
| Main stack | Task and interrupt | Worst-case depth | Overflow |
| Task stacks | Per-task | Worst-case depth | Overflow |

The stack depth must include the call chain, the local variables, the interrupt nesting, and the context save. A watermark measurement during worst-case testing gives the high-water mark; a static analysis gives the bound. Both are useful; the static bound is the evidence.

A stack overflow corrupts the adjacent memory, which may be another task's stack or a global. The failure is unpredictable and hard to reproduce. A guard pattern at the end of each stack, checked by the watchdog supervisor, detects the overflow before it corrupts.

### 5.19 Error handling and fault handlers

| Fault | Cause | Handler response |
|---|---|---|
| Hard fault | Invalid memory access, undefined instruction | Log, enter safe state, reset |
| Memory management fault | MPU violation | Log, isolate task, reset |
| Bus fault | Invalid bus access | Log, reset |
| Usage fault | Divide by zero, unaligned access | Log, reset |
| Watchdog reset | Hung task | Log cause, safe state, reset |
| Brown-out | Supply drop | Hold reset, log cause |

A fault handler must be simple and safe. It should log the fault cause to non-volatile memory, drive the outputs to a safe state, and reset. It should not attempt a complex recovery, because the system state is unknown.

The reset cause register must be read and logged at boot. A device that resets without a cause is a device that cannot be diagnosed.

### 5.20 Logging and trace

A log is evidence. It must record the reset cause, the firmware version, the fault codes, the task heartbeats, and the safety events. It must be stored in non-volatile memory with a bounded size and a wear-levelling strategy.

| Log type | Content | Storage | Use |
|---|---|---|---|
| Event log | Safety events, mode changes | Flash, circular | Post-market analysis |
| Fault log | Fault cause, stack pointer | Flash, reserved | Diagnosis |
| Trace | Task switches, interrupts | RAM or debug port | Timing analysis |
| Data log | Sensor samples | Flash or SD | Clinical analysis |

The log must be tamper-evident for a safety-critical event. A log that can be edited by the user is not evidence. A cryptographic chain or a write-once region provides integrity.

### 5.21 Ten firmware review questions

1. What is the task table, and what are the periods and WCETs?
2. What is the utilisation, and what is the response time of each task?
3. What is the blocking time from shared resources?
4. What is the watchdog policy, and which task services it?
5. What is the state after a brown-out, a fault, and a failed update?
6. What is the memory map, and what is the worst-case stack?
7. What is the update scheme, and is it atomic and signed?
8. What is logged, and where is it stored?
9. What is the toolchain, and is the build reproducible?
10. What is the traceability from requirement to test?

A review that answers all ten has covered the firmware as a system. A review that answers only the first and the seventh has reviewed the scheduler and the bootloader, not the device.

### 5.22 Update timing arithmetic

An update writes a 256 kB image to flash at 128 kB/s. The write time is:

\[
t_{\text{write}} = \frac{256\ \text{kB}}{128\ \text{kB/s}} = 2.0\ \text{s}
\]

The verification reads the image back at 256 kB/s, which takes 1.0 s. The metadata commit takes 10 ms. The total update time is 3.01 s.

If the device draws 50 mA at 3.3 V during the update, the energy is:

\[
E = 3.3 \times 0.050 \times 3.01 = 0.497\ \text{J}
\]

A 150 mAh battery at 3.7 V stores 1,998 J, so the update uses 0.025% of the battery. The energy is not the constraint; the power-loss window is. A power loss during the 2.0 s write leaves the inactive slot incomplete, but the active slot is untouched. A power loss during the 10 ms metadata commit is the critical window; the metadata must be written atomically, with a checksum and a redundant copy.

### 5.23 Boot timing arithmetic

| Stage | Time | Cumulative |
|---|---:|---:|
| Bootloader start | 50 ms | 50 ms |
| Image verification (256 kB at 2 MB/s plus hash) | 130 ms | 180 ms |
| Hardware initialisation | 20 ms | 200 ms |
| RTOS start and task creation | 5 ms | 205 ms |
| Application self-test | 45 ms | 250 ms |

The boot takes 250 ms. A clinical requirement that the device be ready within 2 s leaves 1.75 s of margin. A requirement of 200 ms would fail, and the boot would need a smaller image, a faster flash, or a deferred self-test.

The boot time must be measured at the worst-case temperature and supply, not only at room temperature. A flash read at −20 °C can be slower than at 25 °C.

### 5.24 Watchdog window arithmetic

A windowed watchdog has a lower bound and an upper bound. The lower bound prevents a runaway loop from servicing too early; the upper bound prevents a hung task from missing the service.

For a 10 ms control task with a 2.5 ms response and a 1.5 ms blocking term, the service should occur between 20 ms and 50 ms after the previous service. A service at 5 ms indicates a runaway loop; a service after 50 ms indicates a hang.

The window must be wider than the worst-case jitter and narrower than the safety limit. A window that is too narrow causes spurious resets; one that is too wide allows an unsafe condition to persist.

### 5.25 Interrupt entry cost

On a Cortex-M4 at 100 MHz, the interrupt entry is 12 cycles at zero wait states, or 0.12 µs. An ISR that runs for 5 µs has a total cost of 5.12 µs. At 1 kHz, the ISR consumes 0.51% of the CPU; at 10 kHz, 5.1%. The entry cost is small compared with the ISR body, but the body must be short and must defer the work to a task.

An ISR that calls a blocking RTOS function is a design error. The ISR should post a semaphore or a queue and return. The task then does the work at a priority that the scheduler can manage.

- Keep the ISR under 10 µs.
- Post a semaphore or queue; do not process in the ISR.
- Clear the interrupt source before returning.
- Log the interrupt count for diagnosis.

### 5.26 Update state machine

| State | Entry | Exit | Failure response |
|---|---|---|---|
| Idle | Normal operation | Update request | None |
| Receiving | Request accepted | Image complete | Timeout; return to Idle |
| Writing | Image complete | Write complete | Power loss; mark slot invalid |
| Verifying | Write complete | Signature and CRC valid | Mark slot invalid |
| Committing | Verification valid | Metadata written | Atomic; old or new |
| Rebooting | Commit complete | Bootloader starts | Bootloader selects valid slot |
| Rolling back | New image fails self-test | Old image starts | Safe state if no valid image |

The state machine must be explicit, and each transition must have a timeout and a failure response. A state that can hang is a state that can brick the device.

## 6. Worked example

### 6.1 The requirement

An infusion pump runs four periodic tasks on a 100 MHz Cortex-M4 with a preemptive RTOS. The safety requirement is that the control task completes within 8 ms of its release, and that the pump enters a safe state within 20 ms if the control task misses its deadline.

| Task | Period (ms) | WCET (ms) | Deadline (ms) | Priority |
|---|---:|---:|---:|---:|
| Sensor | 5 | 0.5 | 5 | 1 (highest) |
| Control | 10 | 2.0 | 8 | 2 |
| Comms | 50 | 8.0 | 50 | 3 |
| Log | 100 | 5.0 | 100 | 4 (lowest) |

The tasks share a mutex that protects the pump's state. The longest critical section is 1.5 ms, held by the comms task.

### 6.2 Utilisation

\[
U = \frac{0.5}{5} + \frac{2.0}{10} + \frac{8.0}{50} + \frac{5.0}{100} = 0.10 + 0.20 + 0.16 + 0.05 = 0.51
\]

The utilisation is 0.51, below the four-task bound of 0.757. The task set passes the utilisation test, so it is schedulable if the tasks are independent and the WCETs are correct.

### 6.3 Response time of the control task

The control task has a higher-priority task, the sensor task. The response-time iteration is:

\[
R_0 = 2.0
\]

\[
R_1 = 2.0 + \left\lceil \frac{2.0}{5} \right\rceil \times 0.5 = 2.0 + 0.5 = 2.5
\]

\[
R_2 = 2.0 + \left\lceil \frac{2.5}{5} \right\rceil \times 0.5 = 2.5
\]

The iteration converges at 2.5 ms, below the 8 ms deadline. The control task meets its deadline with a 5.5 ms margin.

### 6.4 Response time of the comms task

The comms task has three higher-priority tasks. The iteration is:

\[
R_0 = 8.0
\]

\[
R_1 = 8.0 + \left\lceil \frac{8.0}{5} \right\rceil \times 0.5 + \left\lceil \frac{8.0}{10} \right\rceil \times 2.0 = 8.0 + 1.0 + 2.0 = 11.0
\]

\[
R_2 = 8.0 + \left\lceil \frac{11.0}{5} \right\rceil \times 0.5 + \left\lceil \frac{11.0}{10} \right\rceil \times 2.0 = 8.0 + 1.5 + 4.0 = 13.5
\]

\[
R_3 = 8.0 + \left\lceil \frac{13.5}{5} \right\rceil \times 0.5 + \left\lceil \frac{13.5}{10} \right\rceil \times 2.0 = 8.0 + 1.5 + 4.0 = 13.5
\]

\[
R_3 = 8.0 + \left\lceil \frac{13.5}{5} \right\rceil \times 0.5 + \left\lceil \frac{13.5}{10} \right\rceil \times 2.0 = 8.0 + 1.5 + 4.0 = 13.5
\]

The iteration converges at 13.5 ms, below the 50 ms deadline. The comms task meets its deadline with a 36.5 ms margin.

### 6.5 The blocking term

The control task shares a mutex with the comms task. If the comms task holds the mutex when the control task is released, the control task waits up to 1.5 ms. The response time becomes:

\[
R_{\text{control}} = 2.5 + 1.5 = 4.0\ \text{ms}
\]

The control task still meets the 8 ms deadline, with a 4.0 ms margin. If the critical section were 6 ms, the response time would be 8.5 ms, and the task would miss its deadline.

The blocking term must be included in the analysis. A task set that passes the utilisation bound but ignores blocking can still miss a deadline.

### 6.6 The watchdog

The worst-case loop time is the response time of the lowest-priority task plus its execution, or about 19 ms. The watchdog timeout must be longer than the loop time and shorter than the safety limit. A simple watchdog with a 50 ms timeout is adequate for the loop, but it can be fooled by a low-priority task that continues to run.

The design uses a windowed watchdog with a 30–50 ms window, serviced by the control task. The control task services the watchdog only if the sensor task has run within its period and the comms task has not reported an error. A hung control task stops servicing, and the watchdog resets the device.

The safe state is a clamped pump with the alarm on. The device enters it within 20 ms of a missed deadline, not only after the watchdog reset.

### 6.7 Diagnostic 1: priority inversion

The measured control latency is 30 ms, not the 4.0 ms predicted. The control task is blocked on the mutex while the comms task holds it, and the log task preempts the comms task. The control task waits for the log task, which has a lower priority than the control task.

The diagnostic is to record the control task's wait time and the mutex owner. If the wait time correlates with the log task's execution, the inversion is confirmed. A second diagnostic is to enable priority inheritance and compare the latency.

The design flaw is the mutex without priority inheritance. The fix is to enable priority inheritance in the RTOS, or to use a priority ceiling protocol, or to remove the shared resource from the control path.

### 6.8 Diagnostic 2: the watchdog hides a failure

The device runs for weeks without a reset, but the pump's dose history shows gaps. The watchdog is serviced by the log task, which continues to run even when the control task is stuck. The watchdog never expires, and the device never resets.

The diagnostic is to log the control task's heartbeat and compare it with the watchdog service. If the heartbeat stops while the watchdog is serviced, the watchdog is hiding the failure. A second diagnostic is to inject a hang in the control task and observe whether the device resets.

The design flaw is the watchdog service policy. The fix is to service the watchdog from the control task, or from a supervisor that checks all heartbeats. The watchdog must observe the function that matters, not any function.

### 6.9 Vary one assumption: the control period falls to 5 ms

Change the control period from 10 ms to 5 ms, keeping the WCET at 2.0 ms. The utilisation becomes:

\[
U = 0.10 + 0.40 + 0.16 + 0.05 = 0.71
\]

The utilisation is below 0.757, so the task set is still schedulable by the bound. The response time of the control task is now:

\[
R_0 = 2.0
\]

\[
R_1 = 2.0 + \left\lceil \frac{2.0}{5} \right\rceil \times 0.5 = 2.5
\]

The response time is 2.5 ms, below the 5 ms deadline. The blocking term of 1.5 ms brings it to 4.0 ms, still below 5 ms, but with only a 1 ms margin.

The margin is thin. A 20% increase in the WCET, or a 1 ms increase in the critical section, would make the task miss its deadline. The design is schedulable but not robust.

### 6.10 What the example demonstrates

The arithmetic closes: 0.51 utilisation, 2.5 ms control response, 13.5 ms comms response, 4.0 ms with blocking, and a 30–50 ms watchdog window. The two diagnostics reveal different flaws: one is a priority inversion, the other is a watchdog that hides a failure. The sensitivity analysis shows that halving the control period leaves only a 1 ms margin.

The general method is to compute the utilisation, solve the response-time equation, add the blocking term, and verify the watchdog policy. A task set that passes the utilisation bound is not necessarily safe; the blocking and the watchdog determine the field behaviour.

### 6.11 Update energy and time

The firmware image is 256 kB. At a 128 kB/s write rate, the write takes 2.0 s; the read-back verification takes 1.0 s; the metadata commit takes 10 ms. The total is 3.01 s. At 50 mA and 3.3 V, the energy is 0.497 J, which is negligible compared with the battery.

The critical window is the metadata commit, not the write. A power loss during the write leaves the inactive slot incomplete and the active slot intact; a power loss during the commit can leave the metadata inconsistent. The commit must be atomic: write the new metadata to a redundant page, verify it, and then invalidate the old page. The bootloader chooses the slot with the valid metadata.

### 6.12 Verification matrix

| Test | Condition | Metric | Acceptance |
|---|---|---|---|
| Utilisation | Nominal task set | U | < 0.757 |
| Control response | Worst-case release | ms | < 8 ms |
| Control response with blocking | Comms holds mutex | ms | < 8 ms |
| Comms response | Worst-case release | ms | < 50 ms |
| Watchdog | Inject control hang | Reset within | 50 ms |
| Brown-out | Drop to 2.7 V | Safe state | Outputs clamped |
| Update | Power loss during write | Recovery | Old image boots |
| Update | Power loss during commit | Recovery | Old or new image boots |
| Signature | Corrupt image | Boot refusal | Safe state |
| Stack | Worst-case call depth | High-water mark | < 80% of stack |

The matrix tests the timing, the failure behaviour, and the update. A firmware that passes only the nominal timing is not verified for the field.

The verification must include a fault-injection run: a hang in each task, a corrupt image, a power loss at each update stage, and a brown-out at each boot stage. The response to each fault is the evidence that the safe state is reachable.

A note on WCET: the measured worst case is a lower bound on the true worst case. The true worst case depends on the cache, the branch predictor, the input data, and the interrupt timing. A measured WCET should be multiplied by a margin (1.2–1.5) or bounded by static analysis. A task set analysed with an optimistic WCET is a task set that will miss deadlines in the field.

- Measure with the caches enabled, as in production.
- Measure at the worst-case temperature and supply.
- Include the interrupt and DMA interference.
- Use static analysis where the measurement is uncertain.
- Add the margin to the response-time analysis, not to the test result.

### 6.14 A second task set: the control loop with a short deadline

A motor control loop has a 1 ms period, a 0.3 ms WCET, and a 0.8 ms deadline. A sensor task has a 0.5 ms period and a 0.1 ms WCET. The utilisation is 0.3/1 + 0.1/0.5 = 0.5. The control response is R0 = 0.3; R1 = 0.3 + ceil(0.3/0.5)×0.1 = 0.4; R2 = 0.4 + 0.1 = 0.5; converges at 0.5 ms, below 0.8 ms. The blocking term must be below 0.3 ms, or the deadline is missed.

The example shows that the margin is smaller when the period is short. A 0.1 ms critical section is 20% of the response time; a 0.3 ms critical section is 60%. The design must keep the critical section short or remove it from the control path.
## 7. Code, query, or calculation

The script reproduces the utilisation, response times, blocking term, update time, and update energy in Section 6. The response-time iteration is implemented as a fixed-point loop with a convergence test.

**Assumptions.** The task table from Section 6.1, rate-monotonic priorities, independent tasks except for the 1.5 ms mutex critical section, a 256 kB image, a 128 kB/s write rate, a 256 kB/s read rate, a 10 ms metadata commit, and 50 mA at 3.3 V during the update.

```python
import math

tasks = [
    ("sensor", 5.0, 0.5, 5.0, 1),
    ("control", 10.0, 2.0, 8.0, 2),
    ("comms", 50.0, 8.0, 50.0, 3),
    ("log", 100.0, 5.0, 100.0, 4),
]

U = sum(c/t for _, t, c, _, _ in tasks)
bound = len(tasks) * (2**(1/len(tasks)) - 1)
print(f"utilization {U:.3f}")
print(f"RM bound    {bound:.3f}")

def response(idx, blocking=0.0):
    _, period, wcet, deadline, prio = tasks[idx]
    higher = [t for t in tasks if t[4] < prio]
    R = wcet + blocking
    for _ in range(1000):
        new = wcet + blocking + sum(math.ceil(R / t[1]) * t[2] for t in higher)
        if abs(new - R) < 1e-9:
            return new
        R = new
    return float("inf")

for i, (name, period, wcet, deadline, prio) in enumerate(tasks):
    R = response(i)
    Rb = response(i, 1.5) if name == "control" else R
    print(f"{name:8s} R={R:6.2f} ms  R+block={Rb:6.2f} ms  deadline={deadline:5.1f}")

image_kb = 256
write_rate = 128
read_rate = 256
t_write = image_kb / write_rate
t_read = image_kb / read_rate
t_commit = 0.010
total = t_write + t_read + t_commit
energy = 3.3 * 0.050 * total
print(f"update time {total:.2f} s, energy {energy:.3f} J")

for start in (20,):
    print(f"watchdog window {start}-50 ms")
```

**Expected output.**

```text
utilization 0.510
RM bound    0.757
sensor   R=  0.50 ms  R+block=  0.50 ms  deadline=  5.0
control  R=  2.50 ms  R+block=  4.00 ms  deadline=  8.0
comms    R= 13.50 ms  R+block= 13.50 ms  deadline= 50.0
log      R= 19.00 ms  R+block= 19.00 ms  deadline=100.0
update time 3.01 s, energy 0.497 J
watchdog window 20-50 ms
```

The output shows four things.

First, the utilisation is 0.510, below the 0.757 bound. The task set is schedulable under the assumptions of independent tasks and accurate WCETs.

Second, the control task's response is 2.50 ms without blocking and 4.00 ms with the 1.5 ms critical section. The blocking term is 37.5% of the response time, a larger share than any single higher-priority task. Shared resources dominate the timing of a well-utilised system.

Third, the comms task's response is 13.50 ms, not the 14.0 ms that a hand calculation with one extra iteration might suggest. The fixed-point iteration converges when the ceiling terms stop changing; an extra iteration with the same ceilings does not change the result.

Fourth, the update takes 3.01 s and 0.497 J. The energy is negligible; the risk is the power-loss window during the commit.

**A note on the blocking term.** The 1.5 ms critical section is a measured value, not a datasheet value. A critical section that includes a flash write or a bus transaction can be much longer than the code suggests. Measure the critical section on the target, with interrupts disabled, and use the maximum.

### 7.1 Image integrity check

The script computes a CRC-32 over a 1,024-byte image and checks a watchdog service time against a 20–50 ms window. The CRC is an integrity check, not a security check; a cryptographic hash or a signature is required for authenticity.

```python
import zlib

image = bytes(range(256)) * 4
crc = zlib.crc32(image) & 0xffffffff
print(f"image bytes {len(image)}")
print(f"crc32 0x{crc:08x}")

for service in (5, 30, 55):
    ok = 20 <= service <= 50
    print(f"service {service:2d} ms: {'accept' if ok else 'reset'}")
```

**Expected output.**

```text
image bytes 1024
crc32 0xb70b4c26
service  5 ms: reset
service 30 ms: accept
service 55 ms: reset
```

The CRC detects accidental corruption; it does not detect a deliberate modification, because an attacker can recompute it. The signature, not the CRC, provides authenticity. The watchdog window rejects a service that is too early (a runaway loop) and too late (a hang).

### 7.2 Signature verification timing

A signature verification with a 2,048-bit RSA key takes about 1 ms on a Cortex-M4 at 100 MHz, depending on the library. An ECDSA verification with a 256-bit key takes about 10 ms. The boot time must include the verification; a slow verification can violate the boot-time requirement.

| Algorithm | Key size | Verification time | Boot impact |
|---|---:|---:|---|
| RSA | 2,048 bits | 1–3 ms | Low |
| ECDSA | 256 bits | 5–15 ms | Moderate |
| Ed25519 | 256 bits | 2–5 ms | Low |
| SHA-256 | — | 0.5–2 ms for 256 kB | Low |

The verification time must be measured on the target, not estimated from a desktop. The library, the optimization level, and the hardware acceleration all change the result.

## 8. Data, standards, and regulatory context

Firmware is governed by the software lifecycle standard, the risk management standard, the cybersecurity standard and guidance, and the alarm standard. The table separates the instruments.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| IEC 62304:2006+AMD1:2015 | Medical device software lifecycle | Consensus standard | Class, architecture, verification, configuration, problem resolution |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis, risk controls, residual risk |
| IEC 81001-5-1:2021 | Health software security | Consensus standard | Secure lifecycle, threat model, SBOM |
| FDA premarket cybersecurity guidance | Device cybersecurity | Guidance enforced through review | Threat model, testing, SBOM, update plan |
| IEC 60601-1-8:2006+AMD2:2020 | Alarm systems | Collateral standard | Alarm delays, signal generation, disclosure |
| IEC 62366-1:2015+AMD1:2020 | Usability engineering | Consensus standard | Use-related risk, error handling, messages |
| MISRA C:2012 | C coding standard | Industry convention | Coding rules, undefined behaviour |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General safety | Consensus standard | Electrical safety, isolation, thermal |

### 8.1 IEC 62304 and the software class

IEC 62304 classifies software as A, B, or C. Class A requires basic planning and configuration; class B requires architecture, unit verification, and integration testing; class C requires detailed design, unit verification, integration testing, and regression testing. The class is per software item.

The standard requires a software development plan, a requirements specification, an architecture design, a detailed design for class C, unit verification, integration testing, a traceability matrix, configuration management, and a problem resolution process. The evidence is the plan and the records, not the code alone.

A firmware update is a change to the software. It must follow the change control process, be verified, and be recorded. A device that updates without change control is a device that can drift outside its validated configuration.

### 8.2 Cybersecurity and secure boot

The FDA's premarket cybersecurity guidance and IEC 81001-5-1 require a threat model, security testing, an SBOM, and a plan for security updates. For firmware, this means:

- A root of trust in the boot ROM or one-time-programmable memory.
- Signature verification of the image at boot.
- Signed updates with rollback protection.
- A process for vulnerability monitoring and patching.
- A record of the firmware version and hash.

A device that cannot be updated is a device that cannot be patched. A device that updates without verification is a device that can be compromised. The boot ROM and the memory protection unit determine which is possible.

### 8.3 Alarm delays

IEC 60601-1-8:2006+AMD2:2020 clause 6.4.1 requires disclosure when the sum of the maximum alarm condition delay and the maximum alarm signal generation delay exceeds 10 s, or when the mean sum exceeds 5 s. It does not impose a hard 10 s limit. Particular standards may impose tighter limits.

For firmware, the alarm signal generation delay includes the detection, the display update, and the network transmission. A task that runs every 100 ms contributes up to 100 ms of detection delay; a display that refreshes at 1 Hz contributes up to 1 s. The budget must include both.

### 8.4 What the standards do not cover

The standards do not specify:

- The RTOS or the scheduler.
- The task periods or priorities.
- The watchdog timeout.
- The update scheme.
- The logging format.

Those are design decisions that the manufacturer must justify. A device can meet every clause and still reset in the field because the watchdog was misconfigured.

### 8.5 The data implications

The firmware version, the build hash, the configuration, and the reset cause must be recorded with the data. A change in any of them can change the device's behaviour. Without the record, a data change cannot be distinguished from a firmware change.

A practical rule: if the dataset does not contain the firmware version and the reset-cause log, the analysis cannot exclude a firmware-related gap.

### 8.6 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 62304 | Software class, version, build hash, test results | Reproduce the evidence |
| ISO 14971 | Hazard analysis, fault response | Link failure to harm |
| IEC 81001-5-1 | SBOM, threat model, patch status | Security lifecycle |
| FDA cybersecurity | Update plan, vulnerability monitoring | Premarket and post-market |
| IEC 60601-1-8 | Alarm delay, signal generation delay | Reproduce the alarm claim |
| IEC 62366-1 | Use-error analysis, messages | Human factors |

The record is the same information the data team needs for provenance. A team that records the firmware and reset cause for its own analysis is also assembling regulatory evidence.

## 9. Industry practice and case evidence

### 9.1 FreeRTOS and Zephyr dominate embedded medical firmware

FreeRTOS and Zephyr are the most common open-source RTOSes in medical devices. FreeRTOS provides tasks, queues, semaphores, mutexes with priority inheritance, and software timers, with a small footprint. Zephyr provides a similar set plus a device model, a build system, and a security-focused update path.

The choice is not only technical. The RTOS's certification evidence, its long-term support, its vulnerability history, and its toolchain matter for a regulated device. An RTOS that is not maintained is a security risk, regardless of its features.

**Documented fact:** both RTOSes are open source and widely used. **Inference:** the prevalence in medical devices is inferred from the ecosystem and from public device documentation, not from a census.

### 9.2 Secure boot failures are real

The 2025 CMR Surgical Versius console recall involved Secure Boot not being enabled. The recall is a reminder that a security feature that is present but not configured is not a control. The boot configuration must be verified in production, not only in the development build.

The failure mode is subtle: the device works, the updates are signed, but the bootloader accepts an unsigned image because the verification is disabled. The device is functional and insecure. The only detection is a configuration check.

**Documented fact:** the recall is listed in the FDA recall database and in manufacturer notices. **Inference:** the root cause and the fix are not public beyond the recall notice.

### 9.3 Update failures and rollback

An update that fails during the write can leave the inactive slot incomplete. If the bootloader always tries the inactive slot, the device bricks. If the bootloader checks the image's signature and CRC, it falls back to the active slot. The A/B scheme with a rollback flag is the standard mitigation.

The failure is more likely in a low-battery condition. A device that starts an update at 5% battery may lose power during the write. The update policy should require a minimum state of charge and should refuse an update below it.

### 9.4 Watchdog practice

A watchdog is only as good as its service policy. A watchdog serviced by the lowest-priority task can be fooled by a hung higher-priority task. A watchdog serviced by the highest-priority task can be fooled by a hung lower-priority task that the high-priority task does not depend on.

The robust design is a supervisor that checks each task's heartbeat and services the watchdog only if all heartbeats are current. The supervisor itself must be checked by the watchdog. The design is a hierarchy, not a single timer.

### 9.5 What good firmware practice looks like

Teams that avoid firmware surprises tend to:

- Write a task table with periods, WCETs, and deadlines.
- Compute the response times and include the blocking term.
- Service the watchdog from a supervisor that checks heartbeats.
- Use A/B updates with signature verification and rollback.
- Log the reset cause and the firmware version.
- Test the fault paths, not only the normal path.
- Keep the build reproducible and archive the image.
- Review the interrupt handlers and critical sections.

The practices are ordinary. The failures they prevent are not.

### 9.6 A note on the bootloader

The bootloader is the most trusted code in the device. It runs before the operating system, before the application, and before any security policy. A bug in the bootloader can brick the device, bypass the signature check, or leave the watchdog unconfigured.

The bootloader should be small, simple, and reviewed more carefully than the application. It should verify the image, initialise the watchdog, set the clock, and jump to the application. It should not contain a complex protocol, a filesystem, or a network stack.

### 9.7 What a good update policy looks like

- Require a minimum state of charge before starting.
- Download to the inactive slot, never over the active one.
- Verify the signature and CRC before committing.
- Write the metadata atomically.
- Reboot into the new image and run a self-test.
- Roll back if the self-test fails.
- Log the update result and the version.
- Never update during a critical therapy.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Deadline miss | Task overrun; queue growth | Utilisation too high; blocking | Late control or alarm | Response-time analysis; reduce load |
| Priority inversion | High-priority task waits | Mutex without inheritance | Unbounded delay | Priority inheritance; ceiling |
| Watchdog hides a hang | No reset despite a stuck task | Watchdog serviced by another task | Unsafe operation | Supervisor heartbeat policy |
| Watchdog reset loop | Repeated resets | Firmware fails to service | Device unavailable | Reset cause logging; fail-safe |
| Brown-out corruption | Reset during a write | Supply drop | Corrupt state | Brown-out detector; atomic writes |
| Update brick | Device does not boot | Incomplete write; no rollback | Device lost | A/B partitions; signature |
| Unsigned update | Boot accepts modified image | Secure boot disabled | Compromise | Verify boot configuration |
| Stack overflow | Random crashes | Deep call chain; large arrays | Memory corruption | Stack analysis; guard |
| Heap fragmentation | Allocation fails after hours | Dynamic allocation | Crash | Static allocation; pool |
| ISR too long | Jitter; missed deadlines | Blocking in interrupt | Late tasks | Short ISR; defer to task |
| Critical section too long | Blocking term large | Interrupts disabled | Missed deadline | Measure; shorten |
| Fault handler loops | Device hangs in fault | Handler does not reset | Unavailable | Simple handler; reset |
| Log corruption | Missing events | Wear; power loss | No evidence | Wear levelling; checksum |
| Clock failure | Timing drift | Wrong PLL configuration | Missed deadlines | Clock monitor; safe state |

The most dangerous failure is the watchdog that hides a hang. The device appears healthy because it never resets, but the function that matters has stopped. The heartbeat supervisor is the defence.

### 10.1 Ten red flags in firmware data

1. A reset that occurs without a logged cause.
2. A firmware version that differs between units in the same study.
3. A latency distribution with a tail that correlates with the comms task.
4. A watchdog service interval that does not match the control period.
5. A log with gaps that align with power events.
6. An update that takes longer than the calculated time.
7. A stack watermark above 80% of the allocated stack.
8. A heap that never returns to its baseline.
9. A boot time that increases after an update.
10. A signature check that is disabled in production.

Each is a prompt to measure, not a conclusion.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The task set is independent | Utilisation and response-time analysis | Shared mutex or queue | Add the blocking term |
| The WCET is accurate | Schedulability | Caches, branches, input | Measure; add margin |
| The periods are fixed | Rate-monotonic analysis | Event-driven bursts | Analyse the burst |
| The deadlines equal the periods | Utilisation bound | Arbitrary deadlines | Use the exact analysis |
| The scheduler is preemptive | Response-time model | Interrupts disabled | Measure the disabled time |
| The watchdog is serviced by the right task | Fault detection | Service from a low-priority task | Fault injection |
| The update is atomic | Recovery | Power loss during commit | Fault injection |
| The image is signed | Integrity | Secure boot disabled | Configuration check |
| The stack is large enough | No corruption | Deep call chain | Watermark; static analysis |
| The heap is bounded | No fragmentation | Dynamic allocation | Pool; monitoring |
| The build is reproducible | The image matches the source | Compiler or flag change | Rebuild; compare hash |
| The traceability is complete | Evidence | Missing test | Audit the matrix |

The dominant assumption is that the WCET is accurate. It rarely is on the first measurement. The cache, the branch predictor, and the input data can change the execution time by a factor of two. The measurement must be conservative, and the response-time analysis must include a margin.

A second limit is that the analysis assumes a single core. A multi-core processor adds cache coherence, shared memory, and inter-core communication, each with its own timing. The analysis of a multi-core system is substantially harder.

A third limit is that the firmware model assumes a correct implementation. A design that is schedulable on paper can still fail because of a bug in the scheduler, the driver, or the application. The verification must test the implementation, not only the model.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no measurement, only a datasheet?
- Which assumption was true on the bench but false in the field?
- Which assumption depends on the RTOS, the driver, or the toolchain?
- Which assumption is encoded in a build flag or a configuration register?
- Which assumption is specific to the validation task set?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A review that cannot answer the last two has risks without controls.

## 12. Exercises

### Exercise 1. Compute utilisation (recall, 15 minutes)

Three tasks have periods 4, 10, and 20 ms and WCETs 0.5, 2, and 4 ms. Compute the utilisation and compare it with the three-task rate-monotonic bound.

**Deliverable:** a calculation and a verdict.

**Acceptance criteria:** correct U; correct bound; verdict stated.

**Time:** 15 minutes.

### Exercise 2. Compute a response time (application, 30 minutes)

For the task set in Exercise 1, compute the response time of the lowest-priority task using the fixed-point iteration. State whether it meets a 20 ms deadline.

**Deliverable:** the iteration steps and the result.

**Acceptance criteria:** correct ceiling terms; convergence; deadline comparison.

**Time:** 30 minutes.

### Exercise 3. Add a blocking term (application, 25 minutes)

The lowest-priority task shares a mutex with the highest-priority task. The critical section is 1 ms. Add the blocking term and recompute the response time. Does it still meet the deadline?

**Deliverable:** a calculation and a verdict.

**Acceptance criteria:** blocking added to the right task; result; verdict.

**Time:** 25 minutes.

### Exercise 4. Design a watchdog (design, 30 minutes)

A control task runs every 10 ms with a 2.5 ms response. Design a watchdog policy that detects a hung control task without hiding a hung comms task. State the timeout, the service task, and the failure response.

**Deliverable:** a policy and a timing diagram.

**Acceptance criteria:** window; service policy; safe state; fault-injection test.

**Time:** 30 minutes.

### Exercise 5. Diagnose a priority inversion (critique, 30 minutes)

A control task's latency is 30 ms instead of 4 ms. The comms task holds a mutex, and the log task preempts the comms task. Explain the mechanism and propose two fixes.

**Deliverable:** a diagnosis and two fixes.

**Acceptance criteria:** three-task scenario; priority inheritance or ceiling; cost of each fix.

**Time:** 30 minutes.

### Exercise 6. Design an update (design, 35 minutes)

Design an A/B update for a 256 kB image on a 1 MB flash. Specify the partition layout, the signature, the commit point, and the power-loss response.

**Deliverable:** a layout diagram and a state machine.

**Acceptance criteria:** atomic commit; rollback; signature; power-loss cases.

**Time:** 35 minutes.

### Exercise 7. Write a verification plan (design, 40 minutes)

Write a firmware verification plan for the pump in Section 6. Include the timing, the watchdog, the brown-out, the update, and the fault injection.

**Deliverable:** a plan with at least 10 rows.

**Acceptance criteria:** normal and fault paths; quantitative criteria; traceability.

**Time:** 40 minutes.

### Exercise 8. Evaluate a "real-time" claim (critique, 25 minutes)

A vendor says its RTOS is "certified real-time." Write five questions that would test the claim for a 1 kHz control loop.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover response time, blocking, interrupts, WCET, and measurement; no question is answerable by the certification alone.

**Time:** 25 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Utilisation | Calculation |
| 2 | Response time | Iteration |
| 3 | Blocking | Calculation |
| 4 | Watchdog | Policy and diagram |
| 5 | Inversion | Diagnosis and fixes |
| 6 | Update | Layout and state machine |
| 7 | Verification | Test plan |
| 8 | Claim evaluation | Five questions |

## 13. Self-check answers

**Answer 1 (Exercise 1).** U = 0.5/4 + 2/10 + 4/20 = 0.125 + 0.20 + 0.20 = 0.525. The three-task bound is 3(2^(1/3) − 1) = 0.780. The set passes the bound.

**Answer 2 (Exercise 2).** For the 20 ms task: R0 = 4. R1 = 4 + ceil(4/4)×0.5 + ceil(4/10)×2 = 4 + 0.5 + 2 = 6.5. R2 = 4 + ceil(6.5/4)×0.5 + ceil(6.5/10)×2 = 4 + 1.0 + 2 = 7.0. R3 = 4 + ceil(7/4)×0.5 + ceil(7/10)×2 = 4 + 1.0 + 2 = 7.0. Converges at 7.0 ms, below 20 ms.

**Answer 3 (Exercise 3).** Add 1 ms blocking to the lowest-priority task: R = 8.0 ms, still below 20 ms. The blocking term is 14% of the response time.

**Answer 4 (Exercise 4).** The watchdog window is 20–50 ms, serviced by a supervisor that checks the control task's heartbeat and the comms task's heartbeat. If either heartbeat is stale, the supervisor stops servicing, and the watchdog resets within 50 ms. The safe state is a clamped pump with an alarm, entered within 20 ms by the control task's deadline monitor.

**Answer 5 (Exercise 5).** The comms task holds the mutex; the log task preempts it; the control task waits for the log task even though the log task is lower priority. This is priority inversion. Fixes: enable priority inheritance so the comms task runs at the control task's priority while holding the mutex; or use a priority ceiling so the comms task cannot be preempted by the log task while holding the mutex. The first is simpler; the second has a higher implementation cost.

**Answer 6 (Exercise 6).** Partition the 1 MB flash into two 384 kB slots, a 128 kB bootloader, and a 128 kB metadata region. The update writes the inactive slot, verifies the signature and CRC, writes the metadata with a sequence number and a checksum, and reboots. The bootloader selects the slot with the valid metadata and the highest sequence number. A power loss during the write leaves the old slot valid; a power loss during the metadata commit leaves either the old or the new metadata valid, because the commit is atomic.

**Answer 7 (Exercise 7).** Tests: utilisation and response time on the target; control latency with and without blocking; watchdog fault injection; brown-out at boot and during a write; update with power loss at each stage; corrupt image; stack watermark; heap exhaustion; ISR latency; log integrity. Acceptance: control < 8 ms; reset within 50 ms; safe state within 20 ms; old or new image boots after any power loss.

**Answer 8 (Exercise 8).** Questions: (1) What is the measured response time of the control task at 1 kHz? (2) What is the blocking time from shared resources? (3) What is the interrupt-disabled time in the drivers? (4) How was the WCET measured, and with what margin? (5) What fault-injection evidence is provided? A certification of the kernel is not a certification of the application.

**Rubric for the design exercises.** A strong design quantifies the task set, the response times, the blocking, the watchdog, and the update. A weak design names an RTOS and assumes the deadlines.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 1: utilisation | 0.525 | ±0.005 |
| Exercise 1: bound | 0.780 | ±0.001 |
| Exercise 2: response | 7.0 ms | ±0.1 ms |
| Exercise 3: response with blocking | 8.0 ms | ±0.1 ms |
| Exercise 6: update time | 3.01 s | ±0.02 s |
| Exercise 6: update energy | 0.497 J | ±0.01 J |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 5 | 5 | 15 |
| 2 | 5 | 20 | 5 | 30 |
| 3 | 5 | 15 | 5 | 25 |
| 4 | 5 | 15 | 10 | 30 |
| 5 | 5 | 15 | 10 | 30 |
| 6 | 5 | 20 | 10 | 35 |
| 7 | 5 | 25 | 10 | 40 |
| 8 | 5 | 10 | 10 | 25 |

## 14. Interview questions

**1. What does an RTOS guarantee?**
A strong answer says it provides mechanisms, not deadline guarantees, and that the analysis is the designer's responsibility. A weak answer says it guarantees real-time behaviour.

**2. How do you compute a response time?**
A strong answer writes the fixed-point equation, includes higher-priority tasks and blocking, and iterates to convergence. A weak answer uses utilisation alone.

**3. What is priority inversion?**
A strong answer gives the three-task scenario and names inheritance or ceiling protocols. A weak answer confuses it with starvation.

**4. How do you design a watchdog?**
A strong answer uses a windowed watchdog and a supervisor that checks heartbeats, and states the timeout and safe state. A weak answer says "kick the watchdog".

**5. Why is a watchdog serviced by a low-priority task dangerous?**
A strong answer explains that it can hide a hung high-priority task. A weak answer does not see the failure.

**6. How do you make an update atomic?**
A strong answer describes A/B slots, signature verification, and an atomic metadata commit. A weak answer says "use a bootloader".

**7. What happens after a brown-out?**
A strong answer describes the detector, the reset, the cause log, and the safe state. A weak answer says the device restarts.

**8. What is the software safety class, and how is it determined?**
A strong answer defines classes A, B, and C by harm and ties the class to the risk analysis. A weak answer says class C is "the highest".

**9. How do you verify a firmware timing claim?**
A strong answer measures on the target with the production build, under worst-case load, with a trace. A weak answer quotes a benchmark.

**10. What would make you reject a firmware design?**
A strong answer names a missed deadline, an unbounded blocking, a watchdog that hides a hang, or an unsigned update. A weak answer rejects on code size.

**11. How do you measure a critical section?**
A strong answer measures the time with interrupts disabled, on the target, at the worst-case temperature, and includes the flash or bus transaction. A weak answer reads the code.

**12. Why is a bootloader security-critical?**
A strong answer explains that it runs before the OS and the application, verifies the image, and sets the security policy. A weak answer treats it as a small loader.

## 15. Key terms

- **A/B partition** — two image slots for atomic update.
- **Bootloader** — code that runs before the application.
- **Brown-out** — supply drop below the operating minimum.
- **Code signing** — signature over the firmware image.
- **Critical section** — code that must not be preempted.
- **Deadline** — time by which a task must finish.
- **Fail-safe** — state that is safe when a fault occurs.
- **Firmware** — software embedded in a device.
- **Hard fault** — processor exception for an invalid operation.
- **Heap** — dynamic memory region.
- **ISR** — interrupt service routine.
- **MISRA C** — coding standard for safety-related C.
- **Mutex** — lock with an owner and priority inheritance.
- **Priority inversion** — low-priority task blocks a high-priority one.
- **Priority inheritance** — low-priority task inherits the waiter's priority.
- **Queue** — buffer between tasks.
- **Rate-monotonic** — priority assignment by period.
- **Response time** — time from release to completion.
- **Rollback** — reverting to the previous image.
- **RTOS** — real-time operating system.
- **SBOM** — software bill of materials.
- **Secure boot** — verification of the image at boot.
- **Semaphore** — counter that signals events.
- **Stack** — memory for local variables and return addresses.
- **Static allocation** — memory allocated at compile time.
- **Task** — independent thread of execution.
- **Traceability** — link from requirement to test.
- **Utilisation** — sum of C/T.
- **Watchdog** — timer that resets a hung processor.
- **WCET** — worst-case execution time.
- **Windowed watchdog** — watchdog with a service window.
- **Blocking time** — delay from a shared resource.
- **Commit point** — the atomic metadata write that selects the new image.
- **Context switch** — saving one task and restoring another.
- **Deadline miss** — completion after the deadline.
- **Fault injection** — deliberate fault to test the response.
- **Heartbeat** — periodic signal that a task is alive.
- **Image** — the firmware binary and metadata.
- **Interrupt latency** — time from interrupt to ISR execution.
- **Preemption** — suspending a task to run a higher-priority one.
- **Recovery** — returning to a safe state after a fault.
- **Reset cause** — register value that records why the device reset.
- **Scheduler** — component that selects the next task.
- **Signature** — cryptographic proof of the image's origin.
- **Supervisor** — task that checks heartbeats and services the watchdog.
- **Task table** — list of tasks with periods and WCETs.
- **Tick** — periodic timer interrupt that drives the scheduler.
- **Watchdog window** — allowed service interval.
- **Wear levelling** — distributing writes across flash.
- **Zero-copy** — transfer without copying data.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Standards and regulation**

1. IEC 62304:2006+AMD1:2015, *Medical device software — Software life-cycle processes*. https://webstore.iec.ch/en/publication/22794
2. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
3. IEC 81001-5-1:2021, *Health software and health IT systems safety, effectiveness and security — Part 5-1*. https://webstore.iec.ch/en/publication/63293
4. U.S. Food and Drug Administration, *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-management-system-considerations-and-content-premarket
5. IEC 60601-1-8:2006+AMD2:2020, *Alarm systems*. https://webstore.iec.ch/en/publication/26539
6. IEC 62366-1:2015+AMD1:2020, *Usability engineering*. https://webstore.iec.ch/en/publication/26570
7. MISRA, *MISRA C:2012*. https://www.misra.org.uk/
8. IEC 60601-1:2005+AMD1:2012+AMD2:2020, *General requirements for basic safety and essential performance*. https://webstore.iec.ch/en/publication/26169

**RTOS and tooling**

9. FreeRTOS documentation. https://www.freertos.org/
10. Zephyr Project documentation. https://docs.zephyrproject.org/
11. MCUboot documentation. https://docs.mcuboot.com/
12. Arm, *Cortex-M4 Technical Reference Manual*, DDI 0439B. https://developer.arm.com/documentation/ddi0439/b/

**Background**

13. G. C. Buttazzo, *Hard Real-Time Computing Systems*, 3rd ed., Springer, 2011.
14. E. A. Lee and S. A. Seshia, *Introduction to Embedded Systems*, 2nd ed., 2017.
15. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
16. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm
17. FreeRTOS, *Priority Inheritance* documentation. https://www.freertos.org/
18. Zephyr Project, *Memory Management* documentation. https://docs.zephyrproject.org/
19. MCUboot, *Image Slots and Rollback* documentation. https://docs.mcuboot.com/
20. IEC 62304:2006+AMD1:2015, software lifecycle.
21. Arm, *Cortex-M4 Technical Reference Manual*, section 3.9.
22. FreeRTOS, *Queue Management* documentation.
23. Zephyr Project, *Secure Boot* documentation.
24. MCUboot, *Swap and Rollback* documentation.
25. ISO 14971:2019, risk management.

Further reading:

26. B. Carter, *Op Amps for Everyone*, 5th ed. (not firmware, but useful for the interface).
27. IEC 62304:2006+AMD1:2015, software lifecycle clauses 5–8.
28. FDA, *Cybersecurity in Medical Devices*, premarket content.
29. Zephyr Project, *Threads and Scheduling*.
30. FreeRTOS, *Watchdog and Tick* documentation.
31. IEC 62366-1:2015+AMD1:2020, usability engineering.
32. ISO 14971:2019, risk management.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_power-and-thermal.md`](07_power-and-thermal.md)
- Next topic: [`09_medical-equipment-classes.md`](09_medical-equipment-classes.md)
- Related: [`05_mcus-dsps-and-fpgas.md`](05_mcus-dsps-and-fpgas.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on hardware-in-the-loop testing: see Lab 14A.7 in the survey file.
- Lab on device teardown: see Lab 14A.6 in the survey file.
- Lab index: [`../../labs/`](../../labs/)
