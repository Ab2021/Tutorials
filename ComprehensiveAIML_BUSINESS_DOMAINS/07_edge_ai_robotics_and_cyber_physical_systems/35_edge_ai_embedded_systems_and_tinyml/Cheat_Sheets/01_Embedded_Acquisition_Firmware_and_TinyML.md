---
collection: "module35-edge-ai-embedded-tinyml"
file_id: "01"
topic: "Embedded acquisition, firmware contracts and TinyML integration"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change to target MCU, scheduler, model artifact, UART protocol, or power profile"
notation_style: "plain-text"
prerequisites: ["Basic C memory ownership", "rates and SI units"]
pattern_tags: ["bounded-buffer", "deadline-contract", "safe-handoff"]
source_status: "SOURCED, DERIVED, ILLUSTRATIVE, OPEN"
owner_role: "embedded systems reviewer"
source_index: "INDEX.md"
---

# Embedded Acquisition, Firmware Contracts and TinyML Integration

## 1. File Header

**Purpose:** Design an ownership and timing contract from ADC sample acquisition through a bounded TinyML decision and UART handoff.

**Prerequisites:** C object lifetime, unsigned arithmetic, rates, bytes, and the [module index](../INDEX.md).

**Key outcome:** Given sample rate, block size, serial format, task budget and memory map, calculate feasibility and identify the first violated firmware contract.

**This file does NOT cover:** medical interpretation, certification, a particular board configuration, or measured latency on any device.

**Level:** 1 Foundations with production boundaries.

**Evidence:** Derived protocol and scheduling reasoning; illustrative arithmetic; sourced API/device references only where cited.

**Retrieval cue:** If a model is accurate offline but the device drops or repeats windows, start at Sections 4, 5 and 8.

The directory inventory names fifteen firmware topics; it does not provide their original bodies. This sheet teaches visible subjects without certifying STM32F401RE configuration, CMSIS version behavior, FreeRTOS port details, or a deployed device.

**Takeaway:** A TinyML result is usable only when the sample, buffer, clock and decision that produced it remain identifiable.

## 2. Concept Map / Mental Model

```text
ADC conversion -> DMA write half-buffer -> completion event -> ready queue
sample clock -> timestamped samples -> quality gate -> DSP/inference -> freshness gate -> UART frame
watchdog/safe behavior <- deadline counters <- acknowledgement/telemetry <- host parser
```

The DMA-to-task arrow transfers ownership, not merely a pointer. The action boundary checks age against the end timestamp. A UART acknowledgement is transport evidence, not proof that a remote actor took the intended action.

**Takeaway:** Publish immutable blocks and validate age where an output will be used.

## 3. Foundational Definitions

| Term | Meaning | Units/type | Boundary |
|---|---|---|---|
| `f_s` | sample rate per channel | samples/s | positive; not aggregate rate |
| `N` | samples in one block | samples | positive integer |
| `T_w` | block duration | s | `N/f_s` only for uniform timing |
| `D` | decision deadline | s | needs a named clock origin |
| `C` | execution budget | s | workload and target specific |
| `I` | interference bound | s | includes only stated work |
| `q` | publication sequence | unsigned integer | wrap needs a bounded comparison |
| tensor arena | inference scratch allocation | bytes | excludes stack unless stated |
| 8N1 | serial frame format | ten symbols/byte | no parity or flow control |
| freshness | age at use | s | must name timestamp origin |

**Takeaway:** Define rate, ownership, byte count, timestamp origin and deadline before optimizing code.

## 4. Core Formulas with Full Derivation

### 4.1 Acquisition and UART throughput

Validity before use: samples are uniform at `f_s`; a sample consumes fixed `b_s` bytes; header `h` occurs once per `N` samples; UART is 8N1 without stalls. These are workload assumptions, not a target guarantee.

Derivation: samples/s times bytes/sample gives payload bytes/s. Frames/s is samples/s divided by samples/frame. Eight data bits plus start and stop use ten symbols per payload byte.

```text
FORMULA [FW-01]: R = f_s*b_s + (f_s/N)*h
WHERE: R [bytes/s]; f_s [samples/s]; b_s [bytes/sample]; N [samples/frame]; h [bytes/frame].
CONSTRAINT: fixed records; excludes CRC, escaping, retries and software overhead.
NUMERICAL EXAMPLE: f_s=360, b_s=2, N=100, h=20; R=360*2+(360/100)*20=792 bytes/s.
INTERPRETATION: constructed raw content is 792 bytes/s before protocol additions.

FORMULA [FW-02]: R_8N1=B/10
WHERE: R_8N1 [payload bytes/s]; B [symbols/s, baud].
CONSTRAINT: exact 8N1 framing; no parity or flow control.
NUMERICAL EXAMPLE: B=115200; R_8N1=11520 bytes/s; 792/11520=0.06875=6.875%.
INTERPRETATION: line-rate headroom does not establish bounded queue delay.
```

Independent check: 3.6 frames/s times 20 header bytes is 72 bytes/s; sample payload is 720 bytes/s; total 792. A hexadecimal text encoding would consume at least four characters per 16-bit sample, so using two bytes would understate its load.

### 4.2 Deadline and buffer safety

Validity before use: producer releases a block every `T_w`; consumer execution is bounded by `C`; interference is bounded by `I`; output path cannot block forever.

```text
FORMULA [FW-03]: T_w=N/f_s
WHERE: T_w [s]; N [samples]; f_s [samples/s].
CONSTRAINT: no dropped or inserted samples.
NUMERICAL EXAMPLE: N=36, f_s=360; T_w=0.1 s=100 ms.
INTERPRETATION: ten completed blocks arrive each second.

FORMULA [FW-04]: R=C+I <= D < (K-1)*T_w
WHERE: R response [s]; C execution [s]; I interference [s]; D deadline [s]; K concurrently allocated DMA blocks, including the block just completed.
CONSTRAINT: the producer alternates through K blocks and reuses the just-completed block after the other K-1 block periods; full response-time analysis needs all releases and blocking.
NUMERICAL EXAMPLE: C=18 ms, I=12 ms, D=80 ms, K=2, T_w=100 ms; R=30 ms <=80 ms <100 ms.
INTERPRETATION: 50 ms deadline slack and 70 ms before the next reuse of a two-block ping-pong buffer.
```

Sensitivity: with `I=70 ms`, `R=88 ms>D` but remains below the 100 ms reuse time; the bytes may survive while the result is already too late. With `I=90 ms`, `R=108 ms` exceeds both deadline and reuse time, so the consumer can also race the next DMA write. Expire the output or redesign; this model cannot justify continued actuation from late data.

### 4.3 Average energy

Validity before use: current is approximately constant in each named state and voltage is known. Include regulator, sensor and transition energy before using this for a product estimate.

```text
FORMULA [FW-05]: E=V*sum(I_i*t_i); P_avg=E/sum(t_i)
WHERE: E [J]; V [V]; I [A]; t [s]; P_avg [W].
CONSTRAINT: state currents and duty cycle are representative.
NUMERICAL EXAMPLE: 3.3 V; 18 mA for 20 ms; 0.2 mA for 80 ms.
E=3.3*(0.018*0.020+0.0002*0.080)=0.0012408 J; P_avg=12.408 mW.
INTERPRETATION: active time dominates this constructed period.
```

Dimension check: V*A*s is J. Charge is 0.000376 C; multiplying by 3.3 V returns 0.0012408 J.

### 4.4 Publish-after-complete invariant

Precondition: DMA alone writes a designated producer block. State transition: `FILLING(q)` becomes `READY(q,t_end)` after the target-defined completion boundary; a task claims it to `PROCESSING`; release returns it to `FREE`. Invariant: a consumer reads only a completed `PROCESSING` block, and a producer writes only `FREE` storage. Sequence and timestamp travel with bytes.

Trace: DMA fills A as q=41. Completion publishes A. Task claims A while DMA fills B. Task releases A before DMA needs A again. If it cannot, record overrun and use declared degradation; never overwrite a claimed block.

Failure boundary: volatile flags alone may not establish ordering or cache visibility. Use the target's documented synchronization mechanism.

**Takeaway:** Every calculation is conditional and every buffer is a single-writer state machine.

## 5. Worked Numerical Example

**Evidence: ILLUSTRATIVE and hand-VERIFIED arithmetic, not a device benchmark.** One channel produces 360 signed 16-bit samples/s. Blocks contain 36 samples. DSP plus INT8 inference uses 18 ms; quantified interference is 12 ms. UART uses 115200 baud 8N1 and groups 100 samples with a 20-byte header.

1. `36/360=0.100 s`; ten blocks/s.
2. `18+12=30 ms`, below the constructed 80 ms deadline and below the 100 ms two-buffer reuse interval.
3. `360*2+(360/100)*20=792 bytes/s` raw content.
4. `115200/10=11520 bytes/s` ideal payload capacity.
5. `792/11520=6.875%` ideal line utilization.
6. The energy calculation in FW-05 is `12.408 mW` under its stated state currents.

| Stress | Recalculation | Decision consequence |
|---|---|---|
| interference reaches 70 ms | `18+70=88 ms>80 ms` | deadline violation; contain or redesign |
| four channels | `4*720+72=2952 bytes/s` | recheck ADC sequencing, RAM and protocol |
| execution reaches 110 ms | `110+12=122 ms>100 ms` | producer can reuse a still-claimed ping-pong block; change model/window/schedule |
| binary becomes text | payload is at least 1440 bytes/s for 4 chars/sample | account for parser and CPU cost |

**Takeaway:** Passing mean bandwidth cannot repair a failed ownership or response-time condition.

## 6. Visual / Geometric Intuition

```text
time --->
DMA A: [fill q41........] [free]
Task :                   [claim A][DSP+NN][release]
DMA B:                   [fill q42........]
Deadline q41:            |----80 ms----|
Reuse boundary A:        |----------200 ms----------|
```

The deadline bounds usefulness; the reuse boundary bounds integrity. A system can preserve bytes yet make a stale decision, or meet an algorithm deadline while corrupting unreleased storage.

**Takeaway:** Put ownership and freshness on the same timeline.

## 7. Comparison / Selection Table

| Choice | Use when | Disqualifier | Bounded fallback |
|---|---|---|---|
| interrupt per sample | very low rate, trivial handler | handler can exceed interarrival | DMA blocks |
| circular DMA plus task | continuous streams and DSP | consumer cannot keep up | explicit block drop/degradation |
| bare metal loop | one bounded control path | multiple independent timing needs | static-priority RTOS |
| RTOS tasks | measured separable workloads | unbounded lock or I/O | redesign critical path |
| binary UART | compact controlled peer | manual text is required | binary data plus diagnostic channel |
| MCU decision | hard local deadline | model exceeds measured budget | send advisory output |
| MPU/Pi decision | larger model requires OS | OS delay treated deterministic | retain MCU freshness gate |

**Takeaway:** Use the strongest disqualifier, not the most attractive feature, to choose a boundary.

## 8. Failure Modes & Edge Cases

### DMA race
**Symptom:** mixed or repeated windows.
**Competing causes:** consumer reads filling storage; wrong boundary; visibility error.
**Discriminating diagnostic:** sequence plus split-versus-unsplit oracle.
**Bounded mitigation:** claim only completed blocks.
**Mitigation risk:** A mitigation can reduce visibility, increase latency or reject useful data; measure that cost.
**Verification:** wrap stress with no mixed window.
**Provenance:** HYPOTHETICAL teaching scenario; not an incident report.

### priority inversion
**Symptom:** deadline spikes.
**Competing causes:** long lock; low-priority owner; interrupt load.
**Discriminating diagnostic:** blocking trace and lock owner.
**Bounded mitigation:** shorten lock and bound critical section.
**Mitigation risk:** A mitigation can reduce visibility, increase latency or reject useful data; measure that cost.
**Verification:** full-load response trace.
**Provenance:** HYPOTHETICAL teaching scenario; not an incident report.

### UART frame loss
**Symptom:** parser desynchronizes.
**Competing causes:** noise; ring overflow; length bug.
**Discriminating diagnostic:** CRC, sequence gaps and high-water.
**Bounded mitigation:** length+CRC+resynchronization.
**Mitigation risk:** A mitigation can reduce visibility, increase latency or reject useful data; measure that cost.
**Verification:** byte-loss injection.
**Provenance:** HYPOTHETICAL teaching scenario; not an incident report.

### stack arena collision
**Symptom:** load-dependent crash.
**Competing causes:** stack growth; arena size; buffer overwrite.
**Discriminating diagnostic:** map file and stack watermark.
**Bounded mitigation:** named allocations with margin.
**Mitigation risk:** A mitigation can reduce visibility, increase latency or reject useful data; measure that cost.
**Verification:** worst-input watermark.
**Provenance:** HYPOTHETICAL teaching scenario; not an incident report.

### stale decision
**Symptom:** correct output arrives late.
**Competing causes:** queue delay; old timestamp; host pause.
**Discriminating diagnostic:** action time minus window end.
**Bounded mitigation:** freshness expiry.
**Mitigation risk:** A mitigation can reduce visibility, increase latency or reject useful data; measure that cost.
**Verification:** delayed-valid transport test.
**Provenance:** HYPOTHETICAL teaching scenario; not an incident report.

### tensor mismatch
**Symptom:** target differs from reference.
**Competing causes:** scale/layout/signedness mismatch.
**Discriminating diagnostic:** per-layer integer vectors.
**Bounded mitigation:** pin conversion manifest.
**Mitigation risk:** A mitigation can reduce visibility, increase latency or reject useful data; measure that cost.
**Verification:** edge-shape golden vectors.
**Provenance:** HYPOTHETICAL teaching scenario; not an incident report.

### filter reset
**Symptom:** boundary transient.
**Competing causes:** lost DSP state; reordered blocks.
**Discriminating diagnostic:** long versus partitioned oracle.
**Bounded mitigation:** retain state per stream.
**Mitigation risk:** A mitigation can reduce visibility, increase latency or reject useful data; measure that cost.
**Verification:** discontinuity reset test.
**Provenance:** HYPOTHETICAL teaching scenario; not an incident report.

### wake gap
**Symptom:** invalid post-wake samples.
**Competing causes:** sensor settling; clock start; stale calibration.
**Discriminating diagnostic:** wake epoch trace.
**Bounded mitigation:** qualify settling period.
**Mitigation risk:** A mitigation can reduce visibility, increase latency or reject useful data; measure that cost.
**Verification:** repeated wake trace.
**Provenance:** HYPOTHETICAL teaching scenario; not an incident report.

**Takeaway:** Identify an observable broken contract before changing model parameters.

## 9. Complexity & Cost Analysis

| Operation | Model | Excludes | Target evidence needed |
|---|---|---|---|
| DMA publication | O(1) per block, O(KN) storage | bus/cache effects | completion-to-release trace |
| FIR | O(NL) MACs, O(L) state | kernel effects | cycles and state placement |
| inference | sum of layer operations; liveness-driven arena | memory bandwidth | arena, stack and cycles |
| UART | O(bytes), lower line cost `10*bytes/B` | escaping and peer stalls | queue high-water |
| trace | O(events), finite ring | observer perturbation | timestamp overhead |

**Takeaway:** Growth notation identifies scaling risk; only a target trace establishes acceptance.

## 10. Key Theorems / Results Reference

| Result | Statement and limit | Reference |
|---|---|---|
| Rate-period identity | `T=1/f` for positive uniform rate; says nothing about jitter. | DERIVED |
| Byte conservation | queue change equals arrivals minus completed service for named boundaries. | DERIVED |
| Task priority | priority scheduling requires bounded workloads; it is not a deadline proof. | [FreeRTOS task priorities](https://www.freertos.org/Documentation/02-Kernel/02-Kernel-features/01-Tasks-and-co-routines/03-Task-priorities), accessed 2026-11-09 |
| CMSIS-NN | kernel behavior depends on specified tensor contracts and version. | [CMSIS-NN](https://arm-software.github.io/CMSIS-NN/latest/), accessed 2026-11-09 |
| Exact device facts | part behavior must come from the exact device documentation. | [ST STM32F401RE](https://www.st.com/en/microcontrollers-microprocessors/stm32f401re.html), accessed 2026-11-09 |

**Takeaway:** Pin changing device and software facts to primary documentation.

## 11. Interview / Exam Questions

### Q01. Why is DMA completion not a valid ML input?
**Checked answer:** It identifies a transfer boundary; input validity also needs buffer identity, units, sequence and memory visibility.
**Reasoning:** It identifies a transfer boundary; input validity also needs buffer identity, units, sequence and memory visibility.
**Follow-up:** Cacheable memory requires documented cache synchronization.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** DMA removes all races.
**Scoring:** 2 points: ownership plus visibility.

### Q02. Compute 230400 baud 8N1 payload capacity.
**Checked answer:** `230400/10=23040 bytes/s` before overhead.
**Reasoning:** `230400/10=23040 bytes/s` before overhead.
**Follow-up:** Parity changes symbols per byte.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** baud equals bytes/s.
**Scoring:** 2 points: framing arithmetic.

### Q03. What does double buffering guarantee?
**Checked answer:** Only alternate storage when consumer release precedes reuse; it proves neither freshness nor deadline.
**Reasoning:** Only alternate storage when consumer release precedes reuse; it proves neither freshness nor deadline.
**Follow-up:** Use an overrun counter.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** It prevents all loss.
**Scoring:** 2 points: conditional guarantee.

### Q04. When does priority inheritance help?
**Checked answer:** For bounded mutex inversion; not for long locks, interrupt load or I/O.
**Reasoning:** For bounded mutex inversion; not for long locks, interrupt load or I/O.
**Follow-up:** Keep state updates short.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** Raise every priority.
**Scoring:** 2 points: mechanism boundary.

### Q05. Why retain FIR state?
**Checked answer:** The filter depends on prior samples; reset changes each block boundary.
**Reasoning:** The filter depends on prior samples; reset changes each block boundary.
**Follow-up:** Reset only at declared discontinuity.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** Each block is independent.
**Scoring:** 2 points: state reasoning.

### Q06. How validate INT8 target equivalence?
**Checked answer:** Compare golden inputs and intermediate integer tensors, metadata and saturation.
**Reasoning:** Compare golden inputs and intermediate integer tensors, metadata and saturation.
**Follow-up:** Final labels can hide divergence.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** Compare only float outputs.
**Scoring:** 2 points: target path.

### Q07. Mean latency passes but deadlines miss. First check?
**Checked answer:** Inspect tail response, blocking, interference and input-dependent execution.
**Reasoning:** Inspect tail response, blocking, interference and input-dependent execution.
**Follow-up:** Expire late decisions safely.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** Optimize mean latency.
**Scoring:** 2 points: deadline evidence.

### Q08. Why is UART ACK not a safety result?
**Checked answer:** It scopes to protocol delivery, not receiver interpretation or physical action.
**Reasoning:** It scopes to protocol delivery, not receiver interpretation or physical action.
**Follow-up:** Use command ID and local gate.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** ACK means done.
**Scoring:** 2 points: evidence scope.

### Q09. How estimate energy?
**Checked answer:** Sum `V*I*t` across named measured states.
**Reasoning:** Sum `V*I*t` across named measured states.
**Follow-up:** Include sensor and regulator where relevant.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** Use active current alone.
**Scoring:** 2 points: units and scope.

### Q10. What belongs in a firmware bug report?
**Checked answer:** Exact part, configuration, build, priorities, input, trace, map and safe effect.
**Reasoning:** Exact part, configuration, build, priorities, input, trace, map and safe effect.
**Follow-up:** Device variants matter.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** It crashed sometimes.
**Scoring:** 2 points: reproducibility.

### Q11. Why can duplicates appear at host?
**Checked answer:** Producer retry and parser replay are distinct; compare sequence and transport events.
**Reasoning:** Producer retry and parser replay are distinct; compare sequence and transport events.
**Follow-up:** Deduplicate with wrap-aware retention.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** Drop blindly.
**Scoring:** 2 points: competing causes.

### Q12. When reject a remote command?
**Checked answer:** When authority, version, frame, safety preconditions or freshness fails at use.
**Reasoning:** When authority, version, frame, safety preconditions or freshness fails at use.
**Follow-up:** Log reason and choose hazard-specific safe behavior.
**Acceptable alternative:** A materially equivalent answer that states a measurable condition and a bounded check.
**Red flag:** Only on network failure.
**Scoring:** 2 points: action boundary.

**Takeaway:** Defensible answers name the condition under which a familiar technique is insufficient.

## 12. Quick Reference Card

| Need | Rule | Check | Critical assumption | Avoid when |
|---|---|---|---|---|
| Acquire | publish completed immutable DMA blocks | sequence/wrap oracle | completion means visible bytes | consumer may read filling storage |
| Budget | state `R=C+I` as simplified model | deadline and interference trace | interference bounded | mean latency is deadline proof |
| UART | 8N1 payload upper bound is `B/10` | include headers/CRC/escaping | no stalls | baud is bytes/s |
| DSP | retain state per stream | partitioned signal oracle | contiguous ordering | blocks are independent |
| INT8 | pin metadata and golden vectors | layer outputs | kernel contract matches | float output substitutes |
| Energy | sum `V*I*t` per state | measure duty cycle | state model applies | data-sheet current is battery life |
| Act | validate authority and freshness at use | late command test | safe action is hazard-specific | remote ACK proves action |

**Source inventory:** [directory.txt lines 1094-1110](../../../../Templates/directory.txt). **Source syllabus:** [edge-arm-firmware](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-arm-firmware). **Evidence boundary:** numbers and traces are illustrative hand calculations; no target build, power measurement, clinical evaluation or safety validation was executed.
