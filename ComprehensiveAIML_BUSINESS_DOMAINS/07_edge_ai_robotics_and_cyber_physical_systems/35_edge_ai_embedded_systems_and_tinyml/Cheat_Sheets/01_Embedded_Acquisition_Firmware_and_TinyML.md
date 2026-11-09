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


### 3.1 Clock, conversion, and frame vocabulary

| Term | Meaning | Units/type | Required distinction |
|---|---|---|---|
| `t_adc` | instant an ADC conversion represents | timer-tick timestamp | not the DMA-completion time |
| `t_pub` | instant firmware publishes a completed block | monotonic tick | may follow `t_adc` |
| `t_use` | instant an output is consumed or acted upon | monotonic tick | used to calculate freshness |
| `J_s` | sample-period jitter | s or ticks | a bound, distribution, or unknown; never an average called a bound |
| `b_adc` | converter resolution | bits/sample | differs from storage width |
| `b_store` | stored record width | bytes/sample | includes sign extension or packing if used |
| `G` | ADC group | ordered channel sequence | channel order is part of the wire contract |
| `O` | overrun count | monotonic count | report separately from sequence gaps |
| `CRC` | error-detection remainder | integer field | detection is not correction or authentication |
| `epoch` | reset/configuration generation | unsigned integer | separates valid sequence wrap from restart |

An ADC register is not a time series merely because DMA copies it. A conversion group needs an explicit order, per-channel scaling rule, trigger source, and timestamp convention. If one timer trigger launches a scan of three channels, the samples have distinct conversion instants unless the device documentation establishes simultaneous sampling. Treating them as simultaneous can create an apparent phase error in a downstream feature.

The source inventory names STM32F401RE register access and HAL abstraction, but its body is unavailable. Therefore this sheet does not specify register bits, channel limits, DMA stream mapping, cache behavior, or interrupt names for any STM32 variant. The portable contract is: configuration establishes the declared trigger and transfer unit; the completion boundary establishes ownership; the firmware records enough metadata to detect a violated declaration.

### 3.2 Execution and synchronization vocabulary

| Term | Meaning | Units/type | Nonexample |
|---|---|---|---|
| ISR | interrupt service routine entered by an interrupt source | execution context | a deferred task callback |
| task | scheduled execution entity | RTOS object | an interrupt handler |
| critical section | bounded region preventing a stated interference class | code region | a general-purpose performance lock |
| mutex | ownership lock with a release owner | synchronization object | an ISR-to-task notification |
| semaphore | token/event synchronization primitive | count or binary state | proof that data fields are coherent |
| priority inversion | higher-priority work waits on lower-priority ownership | scheduling phenomenon | any delay of a high-priority task |
| priority inheritance | temporary priority boost for a mutex owner | policy | a cure for interrupt load |
| WCET | justified upper bound on execution time | s | a sample mean |
| response time | release-to-completion delay | s | CPU time alone |
| jitter | variation in release, start, or completion timing | s | a missing timestamp |

For a DMA completion event, the ISR should do the least work needed to preserve the handoff: identify the completed storage, stamp or associate its end time, update a sequence, and notify a bounded consumer. Filtering, formatting, logging, dynamic allocation, and blocking transport inside that path enlarge interference for every other deadline. This is a design argument, not an instruction to use a particular RTOS API.

Priority inheritance addresses the case where a low-priority task owns a mutex needed by a higher-priority task and a medium-priority task would otherwise run first. It does not bound a low-priority task that performs slow I/O while owning the mutex, prevent an interrupt from preempting all tasks, or make a lock safe for an ISR. An ISR-to-task handoff should use the synchronization mechanism documented for that context.

### 3.3 Model-artifact vocabulary

| Term | Meaning | Units/type | Audit question |
|---|---|---|---|
| model blob | immutable serialized parameters and graph | flash bytes | which hash/version produced it? |
| resolver/operator set | permitted executable operations | finite set | are all graph operations registered? |
| tensor arena | reusable activation/scratch storage | SRAM bytes | what is peak liveness? |
| persistent tensor | state/parameter retained across calls | bytes | who initializes and resets it? |
| quantization metadata | scale, zero-point, dtype and axis | manifest fields | does it match the kernel contract? |
| golden vector | fixed input plus expected intermediate/output values | test artifact | which preprocessing revision made it? |
| model window | samples presented to inference | shape and time span | does it end at the declared timestamp? |
| confidence | model output quantity | dimensionless value | is it calibrated for this population? |

CMSIS-NN is named in the visible inventory as an INT8 kernel topic, while TFLite Micro and compiler choices belong to the next module. A kernel optimization cannot repair a malformed tensor contract. Before calling any inference engine, bind the input shape, layout, signedness, scale, zero point, window ordering, and model artifact identity. Before accepting an output, bind its class order, threshold revision, time span, and stale-output policy.

### 3.4 Memory-map vocabulary

| Region | Typical purpose | Ownership rule | Evidence required |
|---|---|---|---|
| flash/ROM | program and immutable model | write only through controlled update path | linker/map report and artifact hash |
| initialized data | explicit global initial values | copied before application logic | startup/linker declaration |
| zero-initialized data | static zero state | initialized once at reset | startup/linker declaration |
| stack | call frames and automatic storage | private to execution context or explicitly sized | watermark under representative target test |
| heap | dynamic allocation if permitted | lifecycle must be bounded | allocation policy and fragmentation evidence |
| DMA buffers | peripheral-owned then consumer-owned records | no concurrent writer/reader | placement and completion semantics |
| tensor arena | inference scratch | one invocation owner unless proven reentrant | liveness/allocation report |

The linker script topic in the source inventory matters because an array declaration does not prove usable memory. A map file can show that static buffers, model blob, initialized data, and runtime sections fit their addressed regions, but it cannot by itself establish stack peak, DMA reachability, cache coherency, or concurrent ownership. Treat it as one necessary artifact in a memory evidence chain.


**Takeaway:** Timestamp semantics, synchronization context, tensor metadata, and placement are contracts that must be visible before a timing number means anything.
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
### 4.5 A concrete DMA reuse timeline

Validity before use: `K` is the total number of equal block stores in the producer's circular schedule, including the store currently being filled. The producer fills exactly one block during each `T_w`; it does not stall; the consumer claims a block only after that block's completion. This is a storage-integrity bound, not a real-time scheduling proof.

Derivation: suppose block `A` completes at time zero and DMA then fills the next `K-1` distinct stores. Each fill takes `T_w`. DMA begins writing `A` again after those other stores have each been filled. Hence the consumer's available reuse interval is `(K-1) * T_w`, not `K*T_w`. Counting `K*T_w` incorrectly includes the fill interval that ended when `A` became ready.

```text
FORMULA [FW-06]: T_reuse=(K-1)*T_w
WHERE: T_reuse [s] is completion of a block to start of its next DMA write;
       K [blocks] is total circular stores, including the just-completed block;
       T_w [s/block] is producer fill duration.
CONSTRAINT: equal block schedule, no producer stalls, completion at block end;
            a different DMA mode or descriptor schedule requires its own trace.
NUMERICAL EXAMPLE: K=2, T_w=100 ms; T_reuse=(2-1)*100=100 ms.
                   K=3, T_w=100 ms; T_reuse=200 ms.
INTERPRETATION: ping-pong buffering gives one block period after completion,
                not two; a third store adds one additional period.
```

Worked state trace, with `K=3` and `T_w=100 ms`:

```text
t=  0 ms: A completes q=41; A=READY; DMA begins filling B.
t=  0 ms: consumer claims A; A=PROCESSING.
t=100 ms: B completes q=42; DMA begins filling C.
t=200 ms: C completes q=43; DMA begins filling A.
```

The safe consumer-release condition is therefore `t_release(A) < 200 ms` in this trace. A consumer finishing at 199 ms preserves `A` for this schedule but has only 1 ms margin; a consumer finishing at 201 ms races a new producer write. A deadline can be shorter than this integrity interval: with `D=80 ms`, a 90 ms result preserves bytes but is not useful for the stated decision. Conversely, a 50 ms deadline does not make a 150 ms consumer safe merely because its output is expired; it still owns storage that DMA will reuse.

Counterexample: If `K=1`, the formula gives zero. A single block may be read only if producer and consumer have a different, explicitly synchronized transfer protocol; it is not a circular-DMA handoff buffer. If completion occurs at half-buffer boundaries, define each half as a block and recompute `T_w` from the half size. If descriptors have unequal lengths, enumerate the actual future descriptor durations rather than multiplying by an average.

### 4.6 Freshness budget and decision age

Validity before use: timestamps are monotonic in one clock domain or have a bounded, documented conversion error. The decision uses a window ending at `t_end`, and an action occurs at `t_use`. It deliberately measures age of information, not algorithm CPU time.

Derivation: elapsed age is the action time less the timestamp of the newest physical sample represented by the model window. If a system sets a maximum permissible age `A_max`, accept only nonnegative ages no larger than that bound.

```text
FORMULA [FW-07]: A=t_use-t_end; accept iff 0 <= A <= A_max
WHERE: A [s] age at action; t_use [tick or s] action timestamp;
       t_end [same unit] last sample's representation time; A_max [s] permitted age.
CONSTRAINT: common clock semantics; t_end is not task wake time or UART ACK time.
NUMERICAL EXAMPLE: t_end=1.000 s, t_use=1.062 s, A_max=80 ms;
                   A=62 ms, so the constructed decision is admissible.
INTERPRETATION: a correct classification may be rejected solely because it is stale.
```

If the ADC block spans 100 ms and timestamps only its completion, then the oldest sample is already nearly 100 ms older than `t_end`. Whether that matters depends on the feature and decision contract. A feature requiring a complete 100 ms history can legitimately use the end timestamp for freshness; a control loop that assumes a current instantaneous sensor value cannot silently substitute the block end for the measurement time.

Counterexample: A host sends a new command promptly but its payload refers to a sensor epoch from before a device reset. Low transport delay does not make it fresh. Carry `epoch`, sequence, and source end timestamp; reject an epoch mismatch before evaluating numeric age.

### 4.7 Ring occupancy and loss accounting

Validity before use: a queue has capacity `M` records; arrivals and completions are counted at the same queue boundary. This is an accounting identity, not a bound on latency or a license to drop data.

```text
FORMULA [FW-08]: Q(t_1)=min(M,max(0,Q(t_0)+A-S-L))
WHERE: Q [records] queue occupancy; A [records] accepted arrivals;
       S [records] successful removals; L [records] declared drops in interval.
CONSTRAINT: all terms use one interval and one ownership boundary.
NUMERICAL EXAMPLE: M=8, Q(t_0)=5, A=6, S=2;
                   raw occupancy is 9, so at least one arrival needs a stated policy.
INTERPRETATION: a full queue is an observable contract violation, not an excuse
                to overwrite an unaccounted record.
```

Choose the loss policy before overload: reject the newest command, retain only the newest sensor window, decimate with a recorded ratio, or transition to a safe state. Each changes semantics. For a diagnostic recorder, dropping the newest event can preserve causal context; for a real-time display, retaining only the newest state may be better. Neither is universally safe.

Counterexample: A queue high-water mark of two does not prove no loss when an ISR overwrites the oldest element without incrementing `L`. Couple occupancy telemetry with producer sequence gaps and explicit drop reasons.

### 4.8 Fixed-point conversion and saturation

Validity before use: a physical quantity `x` has an affine calibration `x=a*r+c` from raw code `r`; coefficient units are known; integer arithmetic has enough width for intermediate products. Calibration is illustrative unless sourced for the exact sensor and configuration.

```text
FORMULA [FW-09]: x=a*r+c
WHERE: x [physical unit]; r [ADC code]; a [unit/code]; c [unit].
CONSTRAINT: calibration range and reference conditions apply; intermediate width cannot overflow.
NUMERICAL EXAMPLE: a=0.001 V/code, r=1234 codes, c=-1.000 V;
                   x=0.001*1234-1.000=0.234 V.
INTERPRETATION: the model should receive a declared quantity, not an ambiguous code.
```

For signed INT8 quantization, an illustrative dequantization is `x_hat=s*(q-z)` with scale `s` and zero point `z`. The scale's unit is that of `x` per integer step. A saturated `q=127` is not merely a rounded value: it is evidence that the representable range was exceeded. Preserve a saturation counter per tensor or preprocessing stage when feasible; a stable final class can conceal systematic clipping.

Counterexample: applying an offset after narrowing to INT16 may overflow even when the final calibrated physical value would be small. Perform the product and offset in a documented wider type, then apply a defined rounding and saturation rule.

**Takeaway:** A buffer becomes available after the other `K-1` stores, while a decision remains admissible only until its separately declared age limit.

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

### 5.1 Hardware-independent firmware oracle

The following constructed trace is an oracle input, not a target execution result. It is deliberately expressed as events and declared state transitions so a host unit test, simulator, or board trace can evaluate the same contracts. A board/HIL/safety test has **not** been executed.

```text
CONFIG:
  f_s=360 samples/s/channel
  N=36 samples/block
  K=3 DMA blocks: A,B,C
  D=80 ms
  A_max=80 ms
  C=18 ms
  I=12 ms
  epoch=7
  model_id="m-17"
  input_shape=[36,1]
  uart=115200 baud, 8N1

EVENTS:
  0 ms: COMPLETE(A,q=41,t_end=1.000 s,epoch=7)
  0 ms: CLAIM(A,q=41)
 18 ms: DSP_DONE(A,q=41)
 30 ms: INFER_DONE(A,q=41,class=2,score=91/127)
 62 ms: ACTION(A,q=41,t_use=1.062 s)
 62 ms: RELEASE(A,q=41)
100 ms: COMPLETE(B,q=42,t_end=1.100 s,epoch=7)
200 ms: COMPLETE(C,q=43,t_end=1.200 s,epoch=7)
```

Expected results:

| Oracle property | Derivation | Expected result |
|---|---|---|
| block period | `36/360 s` | `100 ms` |
| A reuse start | `(3-1)*100 ms` after A completion | `200 ms` |
| response | `18+12 ms` | `30 ms <= 80 ms` |
| action age | `1.062-1.000 s` | `62 ms <= 80 ms` |
| ownership | A released at 62 ms before 200 ms reuse | PASS |
| sequence | 41, 42, 43 within epoch 7 | contiguous |
| output binding | action has `q=41`, epoch 7, model m-17 | PASS |

The score `91/127` is intentionally not interpreted as a probability or a clinical result. The oracle checks association and timing only. If output selection requires a confidence threshold, that threshold must be versioned and validated separately from the timing test.

### 5.2 Negative oracle: late but storage-safe

```text
CONFIG change: C=18 ms, I=70 ms; all other values unchanged.
EVENTS:
  0 ms: COMPLETE(A,q=41,t_end=1.000 s)
  0 ms: CLAIM(A,q=41)
 88 ms: INFER_DONE(A,q=41)
 88 ms: ACTION_ATTEMPT(A,q=41,t_use=1.088 s)
 88 ms: RELEASE(A,q=41)
200 ms: DMA begins filling A
```

Expected result: `R=88 ms`, and age is `88 ms`; both exceed the constructed 80 ms decision bound. Release still occurs before the 200 ms reuse boundary. The correct oracle outcome is `STALE_REJECT`, not `SUCCESS` and not `BUFFER_OVERRUN`. This separation prevents a test suite from treating data integrity as a substitute for timely use.

### 5.3 Negative oracle: ownership violation

```text
CONFIG change: K=2, C=95 ms, I=12 ms; D=80 ms.
EVENTS:
  0 ms: COMPLETE(A,q=41)
  0 ms: CLAIM(A,q=41)
100 ms: DMA attempts to begin filling A
107 ms: consumer attempts RELEASE(A,q=41)
```

Expected result: `T_reuse=(2-1)*100=100 ms`; the producer reaches A before release. A correct implementation must not silently allow both accesses. Depending on its declared policy, it may drop the new block with `DROP_DMA_OVERRUN`, abort the consumer before it reads, or enter a safe degraded state. It must increment a reasoned counter and preserve the event ordering. It must not relabel the result as an ordinary deadline miss.

### 5.4 Negative oracle: epoch and sequence behavior

```text
EVENTS:
  epoch=7, COMPLETE(A,q=65535,t_end=9.900 s)
  epoch=7, COMPLETE(B,q=0,t_end=10.000 s)
  reset
  epoch=8, COMPLETE(A,q=0,t_end=0.100 s)
```

Expected result: `65535 -> 0` within epoch 7 is a declared unsigned wrap and can be contiguous under the chosen width. `epoch=7,q=0 -> epoch=8,q=0` is a reset boundary, not a duplicate. The receiver must not calculate a negative age across the reset or accept an epoch-7 action for epoch-8 data. The precise modulo comparison and retained history length are implementation choices; document them and test the chosen one.

### 5.5 Input and preprocessing oracle

Use the following dimensionless signed sample window to test ordering and state without claiming sensor realism:

```text
x41 = [0, 1, 2, 3, 4, 5, 6, 7]
gain = 2
bias = -3
state_0 = 0
rule: y[i] = gain*x[i] + bias + state[i]
      state[i+1] = x[i]
```

The expected output is `[-3, -1, 1, 3, 5, 7, 9, 11]`; the final state is `7`. Partitioning the input after four values and carrying the final first-block state gives the same eight outputs. Resetting state before the second partition produces `[-3,-1,1,3,-3,7,9,11]`, exposing the block-boundary error. This small oracle checks state carry, ordering, and integer arithmetic without pretending to validate an ECG filter.

### 5.6 UART frame oracle

Define an illustrative binary frame:

```text
SOF(1) | version(1) | epoch(2) | sequence(2) | t_end_ticks(4) |
model_id(2) | payload_length(1) | payload(n) | CRC(2)
```

For `n=8`, frame length is `1+1+2+2+4+2+1+8+2=23 bytes`. Under 8N1, ideal serialization time is `23*10/115200 s`, or approximately `1.997 ms`. This number excludes transmitter queueing, interrupt service, escaping, flow control, radio/USB bridge behavior, receiver parsing, and acknowledgement. A test oracle can corrupt one payload byte and expect a CRC failure plus resynchronization at the next `SOF`; it cannot claim a particular CRC's detection strength without specifying the polynomial and threat model.

### 5.7 Memory-budget oracle

The following is a bookkeeping input, not a statement about STM32F401RE capacity:

| Allocation class | Bytes | Ownership/lifetime |
|---|---:|---|
| DMA blocks, `K=3`, `N=36`, 2 bytes/sample | 216 | circular acquisition stores |
| DSP state | 128 | persistent per stream |
| tensor arena | 12,288 | one inference invocation |
| task stacks | 2,048 | declared static allocation |
| UART rings | 1,024 | producer/consumer queues |
| static control records | 512 | persistent metadata |
| reserved headroom | 2,048 | not allocated by this design |

Sum: `216+128+12288+2048+1024+512+2048=18264 bytes`. The useful review question is not “does 18,264 fit a board?” but “does the linker/map plus runtime watermark identify each term, its region, and its peak overlap?” A replacement model needing an additional 4,096-byte arena changes the sum to 22,360 bytes and must be evaluated alongside stack peak and DMA-placement constraints rather than compared only by parameter count.

### 5.8 Schedule-ledger input

| Work item | Release | Declared upper execution | Can block? | Required output |
|---|---|---:|---|---|
| DMA completion handoff | every 100 ms | 0.2 ms illustrative budget | no | block ID, sequence, notification |
| DSP/inference task | block notification | 18 ms | no I/O in critical path | result or explicit reject |
| UART formatting task | accepted result | 2 ms | bounded ring enqueue only | immutable frame |
| transport drain | asynchronous | unknown until measured | peer-dependent | high-water and error counters |
| health monitor | 1 s | 1 ms | no | counter snapshot |

This ledger is intentionally incomplete as a response-time proof: it lacks task priorities, all interrupt sources, execution distributions, interrupt masking, lock blocking, and target cycle measurements. It is still useful because it makes the missing proof inputs explicit. Do not replace those unknowns with a scheduler name.

**Takeaway:** A useful hardware-independent test names events, timestamps, ownership, and expected verdicts; it does not turn illustrative arithmetic into board evidence.

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

### ADC trigger drift or channel-order mismatch

**Symptom:** features change when the nominally same physical input is replayed, or one channel appears delayed by one conversion.

**Competing causes:** timer trigger is not the declared source; scan order differs from the model manifest; an interleaved DMA record is decoded as planar data; a timer-clock change changed `f_s`.

**Discriminating diagnostic:** inject a distinct constant code per logical channel and inspect the first three DMA records with sequence and trigger timestamps. The expected order must match the acquisition manifest, not an assumption based on variable names.

**Root-cause evidence:** a configuration snapshot, raw record dump, and timer counter trace locate whether the error is before DSP or in decode. A final ML label is not sufficient evidence because a model can tolerate or conceal a channel permutation.

**Immediate mitigation:** inhibit inference/action on manifest mismatch and export raw records through a bounded diagnostic path.

**Correction:** make channel order, stored width, endianness, trigger source, and effective sample rate versioned fields consumed by both preprocessing and test oracle.

**Prevention:** require a calibration-vector test after code generation, board configuration, clock-tree, or sensor wiring changes.

**Verification:** replay the constant-code vector and an ordered ramp; assert every decoded sample position and reported rate.

**Mitigation risk:** diagnostic export can consume UART budget; cap it and preserve acquisition priority.

**Provenance:** HYPOTHETICAL teaching scenario; exact STM32 configuration is not verified.

### Cache or memory-visibility mismatch

**Symptom:** a completion event is observed but the consumer sees an old or partially updated block on a memory system where this is possible.

**Competing causes:** wrong completion boundary; cache maintenance missing or applied to wrong range; compiler reordering; consumer is reading an alias; hardware fault is not assumed without evidence.

**Discriminating diagnostic:** compare a known incrementing DMA pattern with the consumer's copied snapshot at each completion and record address, length, sequence, and synchronization operation. A mismatch tied to cacheable placement differs from a repeatable indexing bug.

**Immediate mitigation:** move the tested buffer to a target-documented coherent/DMA-safe region or use the documented synchronization primitive around the exact ownership transition.

**Correction:** document which actor writes, which actor invalidates/cleans if required, and which barrier establishes publication. `volatile` qualifies neither cache nor ownership by itself.

**Prevention:** make buffer placement and cacheability part of the linker/configuration review and test at block edges, not only at steady state.

**Verification:** run the incrementing-pattern oracle through enough wraps to exercise every block and report zero stale snapshots under the declared configuration.

**Mitigation risk:** cache maintenance may add bounded work and must enter the response-time ledger.

**Provenance:** HYPOTHETICAL; exact target memory semantics must be obtained from primary documentation.

### Cortex-M versus Cortex-A placement error

**Symptom:** a design assumes the same timing, protection, or process behavior on a microcontroller and a Linux-capable application processor.

**Competing causes:** an MCU path accumulated operating-system services; an MPU path assumes an OS scheduler is deterministic; a Pi-side bridge sends an unbounded request back into the local deadline path.

**Discriminating diagnostic:** draw the decision boundary and annotate every clock domain, process/task boundary, network/serial queue, and fallback. A local freshness gate missing from that diagram is a design defect even if the host model is faster on average.

**Immediate mitigation:** retain a bounded MCU-side validity/action gate and demote remote/MPU output to advisory until a complete timing and safety argument exists.

**Correction:** assign each responsibility—sampling, preprocessing, inference, transport, authorization, actuation—to one owner with a timeout and safe behavior.

**Prevention:** review Cortex-M4F and Cortex-A76/NEON work as different programming models, not as a scalar speed upgrade.

**Verification:** inject a delayed host response while local acquisition continues; expected result is a freshness reject or declared safe behavior, never reuse of an old host result.

**Mitigation risk:** local gating can reduce apparent availability; report its reject reason rather than bypass it.

**Provenance:** HYPOTHETICAL architecture analysis; no Pi5 or Cortex-A76 measurement was run.

### Toolchain or linker-layout regression

**Symptom:** a build fits before a compiler, linker-script, code-generation, or model update and then crashes or loses a peripheral path.

**Competing causes:** model blob growth; section placement change; stack use changed after optimization; a generated startup/peripheral file changed; debugger settings mask timing.

**Discriminating diagnostic:** compare map files by section, symbol, and region; compare startup/configuration sources; record exact compiler/linker version and flags. A total binary-size change alone cannot locate a stack collision.

**Immediate mitigation:** prevent release when named region budget or artifact hash differs without review; choose a known-good artifact only after compatibility checks.

**Correction:** make flash, static SRAM, stack budget, tensor arena, DMA buffers, and reserved headroom independent ledger rows.

**Prevention:** preserve generated-code revision and manual modifications separately; regenerate into a reviewable diff rather than editing generated output as the only source of truth.

**Verification:** inspect map assertions and execute the existing software oracle; target stack/DMA validation remains required and is not claimed here.

**Mitigation risk:** pinning a toolchain can defer security or correctness upgrades; review pins periodically.

**Provenance:** HYPOTHETICAL; CubeIDE and linker behavior are version-specific and not tested.

### Debug instrumentation changes the schedule

**Symptom:** a deadline failure disappears under SWD/JTAG debugging, ITM logging, semihosting, or verbose UART prints.

**Competing causes:** breakpoint halts alter time; logging fills a transport ring; optimization changes code generation; a real race is timing-sensitive.

**Discriminating diagnostic:** compare minimally intrusive counter traces against instrumented traces, with the same input sequence and build identity. Semihosting or breakpoint observations cannot be treated as production timing evidence.

**Immediate mitigation:** use bounded in-memory trace records with drop accounting; remove blocking diagnostics from the critical path.

**Correction:** define trace event size, ring capacity, timestamp source, overflow policy, and whether instrumentation is enabled in the evaluated build.

**Prevention:** test both observability-on and observability-off configurations; treat a difference as a finding rather than discarding the slower run.

**Verification:** replay the scheduling oracle and show whether its event order changes under instrumentation. Board timing evidence remains unexecuted.

**Mitigation risk:** a finite trace ring can lose the event preceding a fault; include a loss counter and capture trigger policy.

**Provenance:** HYPOTHETICAL teaching scenario; no debugger trace was captured.

### Power-state continuity error

**Symptom:** post-wake samples, timestamps, UART frames, or model state are inconsistent even though the device resumes execution.

**Competing causes:** sensor has not settled; timer clock changed; DMA was not restarted as assumed; retained RAM or model state is stale; wake cause is misclassified.

**Discriminating diagnostic:** attach a new wake epoch to configuration, first valid sample, and first accepted action; compare retained state checksum and timestamp monotonicity.

**Immediate mitigation:** mark the stream invalid until the declared settling/initialization conditions pass; reject actions from pre-wake epochs.

**Correction:** specify per-state retention, restart order, settling duration source, and which calculations reset. Sleep, Stop, and Standby are not interchangeable labels for an unknown board configuration.

**Prevention:** maintain a wake-state transition table and test each advertised transition with the same data/sequence oracle.

**Verification:** repeated software state traces must show no accepted action before qualification. Current and wake timing require target measurement.

**Mitigation risk:** qualification delay increases stale rejections and energy use; assess against the actual decision requirement.

**Provenance:** HYPOTHETICAL; no power profile or wake behavior has been measured.

### CMSIS-DSP state or coefficient contract error

**Symptom:** filtering differs between continuous and blockwise paths, or output amplitude changes after a coefficient update.

**Competing causes:** state reset; wrong block length; coefficient order mismatch; fixed-point scaling/saturation; input units changed upstream.

**Discriminating diagnostic:** run a fixed input through one long call and declared partitions while carrying state; compare outputs plus final state. Then repeat with a deliberately reset state to verify that the oracle is sensitive to the expected defect.

**Immediate mitigation:** stop using the affected feature for decisions and emit a diagnostic status rather than silently applying a new filter.

**Correction:** version coefficient arrays, state length, data type, scaling, and block contract alongside the model manifest.

**Prevention:** include partition equivalence, impulse/step behavior, and saturation counters in software regression tests.

**Verification:** the Section 5 preprocessing oracle provides a minimal state-carry test; a signal-specific filter oracle and target kernel validation are still required.

**Mitigation risk:** retaining more state raises SRAM use and reset complexity; include it in the memory ledger.

**Provenance:** HYPOTHETICAL; no CMSIS-DSP kernel run is claimed.

### CMSIS-NN integration mismatch

**Symptom:** the target's integer output differs from the conversion reference or only fails on boundary-shaped inputs.

**Competing causes:** input layout mismatch; quantization axis/scale mismatch; unsupported operator substitution; signedness mismatch; scratch allocation aliasing.

**Discriminating diagnostic:** compare model input bytes, selected intermediate integer tensors, output bytes, and operator-resolution report for golden vectors including minima, maxima, zeros, and non-square shapes where relevant.

**Immediate mitigation:** reject the artifact for deployment and retain the previous compatible artifact only if its manifest, authority, and rollback conditions pass.

**Correction:** generate an immutable deployment manifest containing artifact hash, runtime/kernel version, operator list, tensor specs, preprocessing revision, and golden-vector expected values.

**Prevention:** prohibit “successful conversion” from being the only acceptance signal; require numerical equivalence bounds and explicit unsupported-operation handling.

**Verification:** run the manifest vectors through the target integration when hardware is available; this document supplies inputs but reports no target execution.

**Mitigation risk:** intermediate tracing can expose sensitive data or add memory pressure; use synthetic vectors and bounded capture.

**Provenance:** HYPOTHETICAL; no CMSIS-NN integration result is asserted.

**Takeaway:** Identify an observable broken contract before changing model parameters.

## 9. Complexity & Cost Analysis

| Operation | Model | Excludes | Target evidence needed |
|---|---|---|---|
| DMA publication | O(1) per block, O(KN) storage | bus/cache effects | completion-to-release trace |
| FIR | O(NL) MACs, O(L) state | kernel effects | cycles and state placement |
| inference | sum of layer operations; liveness-driven arena | memory bandwidth | arena, stack and cycles |
| UART | O(bytes), lower line cost `10*bytes/B` | escaping and peer stalls | queue high-water |
| trace | O(events), finite ring | observer perturbation | timestamp overhead |

### 9.1 Work and storage sizing from declared inputs

For an `L`-tap direct FIR applied to `N` samples, a straightforward implementation performs approximately `N*L` multiply-accumulate operations per block after startup. This is an operation count, not a cycle estimate: DSP instructions, memory layout, compiler choice, precision, vectorization, and instruction-cache behavior affect target time. With illustrative `N=36` and `L=16`, the count is `576` MACs/block, or `5,760` MACs/s at ten blocks/s. It helps compare two stated algorithms, but it does not demonstrate that either meets 18 ms.

For a dense layer mapping `n_in` values to `n_out` outputs, the usual count is `n_in*n_out` multiply-accumulates plus `n_out` bias additions. An illustrative `36 -> 8` layer has `288` MACs and eight bias additions. If a model has multiple layers, sum its declared layer operations and add preprocessing, memory moves, activation, formatting, and synchronization to the actual timing ledger. Parameter count is not the same as execution work for every architecture, and neither is peak activation storage.

| Quantity | Formula/model | Illustrative input | Result | Decision boundary |
|---|---|---|---|---|
| raw block bytes | `N*b_store` | `36*2` | 72 bytes | excludes metadata/alignment |
| three DMA blocks | `K*N*b_store` | `3*36*2` | 216 bytes | assumes equal stores |
| FIR work/block | `N*L` MACs | `36*16` | 576 MACs | not cycles |
| frame serialization lower bound | `10*bytes/B` | `10*23/115200` | 1.997 ms | excludes queueing |
| telemetry/day | `R*86400` | `792*86400` | 68,428,800 bytes | excludes file/container overhead |

The telemetry estimate is 68.4 MB in decimal bytes, or approximately 65.26 MiB using `1 MiB=1,048,576 bytes`. It is not storage provisioning because retries, indexes, compression, encryption, retention copies, and payload evolution are excluded. The purpose is to expose the rate-to-retention calculation that a design must later refine.

### 9.2 Cost of a bounded diagnostic path

Each trace record competes with application memory and transport. If a record is 16 bytes and a ring contains 128 records, the ring reserves `2,048` bytes. At ten block completions per second it holds 12.8 seconds of those events if no other event types share it. If an ISR emits multiple records per block or a burst occurs, this retention time changes; a circular ring overwrites history according to a declared policy.

Logging can also violate serial capacity. A 16-byte binary trace record at 100 records/s is 1,600 payload bytes/s before frame overhead, consuming 13.9% of the ideal 11,520 bytes/s 115200-8N1 payload capacity. This does not mean tracing is safe: priority, queue bursts, formatting, and peer stalls remain unmeasured. It means the budget must include instrumentation rather than treating it as free.

### 9.3 Architecture selection under resource evidence

| Boundary | Prefer MCU-side path when | Prefer MPU/host-side path when | Must remain local |
|---|---|---|---|
| acquisition | trigger, DMA and timestamps need bounded ownership | never delegate raw ownership solely for convenience | completion and overrun accounting |
| DSP | state and work fit a measured local budget | richer analysis is advisory or has a bounded handoff | data-quality gate |
| inference | artifact/operators/arena fit and local deadline is binding | larger model is useful but remote timing is not trusted | freshness decision |
| transport | compact diagnostics/control acknowledgment needed | archival, UI, or noncritical retrieval needs OS services | frame validation and authority gate |
| update | compatible signed artifact and rollback evidence exists | build/download orchestration needs host tooling | compatibility and safe activation check |

The named Cortex-M4F/CMSIS and Cortex-A76/NEON variants in the inventory are source scope, not a performance table. A selection is defensible only when it ties a requirement to evidence: model artifact size, supported operations, memory regions, tail response, energy, recovery behavior, and safe fallback. “The faster processor” does not state what happens after a queue stalls or a model revision is incompatible.

**Takeaway:** Operation counts and byte totals bound what to measure; target-specific cycles, stack peaks, power, and recovery behavior remain evidence obligations.

**Takeaway:** Growth notation identifies scaling risk; only a target trace establishes acceptance.

## 10. Key Theorems / Results Reference

| Result | Statement and limit | Reference |
|---|---|---|
| Rate-period identity | `T=1/f` for positive uniform rate; says nothing about jitter. | DERIVED |
| Byte conservation | queue change equals arrivals minus completed service for named boundaries. | DERIVED |
| Task priority | priority scheduling requires bounded workloads; it is not a deadline proof. | [FreeRTOS task priorities](https://www.freertos.org/Documentation/02-Kernel/02-Kernel-features/01-Tasks-and-co-routines/03-Task-priorities), accessed 2026-11-09 |
| CMSIS-NN | kernel behavior depends on specified tensor contracts and version. | [CMSIS-NN](https://arm-software.github.io/CMSIS-NN/latest/), accessed 2026-11-09 |
| Exact device facts | part behavior must come from the exact device documentation. | [ST STM32F401RE](https://www.st.com/en/microcontrollers-microprocessors/stm32f401re.html), accessed 2026-11-09 |

### 10.1 Result boundaries used in this manuscript

**Conservation of records.** When every accepted arrival, completed removal, and declared drop is counted at one queue boundary, the occupancy identity in FW-08 follows by accounting. It does not tell an implementer which record should be dropped, and it becomes false if a hidden overwrite is omitted from `L`. Use it to audit counters, not to infer no loss from a low average queue depth.

**Periodic-release spacing.** For uniform acquisition, `T_w=N/f_s` follows from rate as reciprocal period multiplied by samples per block. It fails under dropped triggers, variable-rate sensing, timebase changes, and ambiguous multi-channel scan semantics. A measured period trace may validate a target configuration; the numerical substitution here validates only arithmetic.

**Single-writer ownership invariant.** If DMA writes only `FILLING` storage, publication happens only after its documented completion boundary, and a consumer reads only a claimed completed block, then producer writes and consumer reads do not concurrently target the same declared block. This is a conditional correctness argument. It says nothing about cache visibility, a wrong DMA address, an ISR that changes state twice, or a consumer that fails to release.

**Response-time decomposition.** `R=C+I` is a deliberately simplified upper-model only when the stated `C` and `I` already bound all relevant execution, blocking, and interference. A real fixed-priority analysis may need release jitter, higher-priority interference, shared-resource blocking, and iteration to a fixed point. Do not attach the simple equation to a FreeRTOS task and call it a proof.

**Framing lower bound.** Under 8N1, one payload byte consumes one start bit, eight data bits, and one stop bit, giving `B/10` bytes/s in the absence of all other protocol work. Parity, extra stop bits, escaping, packet gaps, retransmission, and flow control alter the contract. This is why a UART link can be under its nominal line rate but still overflow a producer ring.

**Fixed-point range rule.** An integer representation is valid only while calibrated/scaled values lie in its declared range and each intermediate operation has a defined width, rounding, and saturation behavior. A result within final range does not rescue an overflowing intermediate. The exact overflow behavior is language, compiler, and target dependent; do not leave it implicit in a safety-relevant path.

### 10.2 Evidence register for source-named variants

| Visible source topic | What this sheet preserves | What remains OPEN |
|---|---|---|
| Cortex-M4F pipeline, Thumb-2, DSP, FPU | instruction selection changes cycle and numeric behavior; count work before claiming time | exact instruction timing and configuration |
| STM32F401RE map/register/HAL | configuration, map, and abstraction must be reviewable | exact register/clock/DMA setup |
| ADC/DMA ECG circular buffers | trigger, block ownership, reuse deadline, and timestamp contract | sensor/clinical validity and board trace |
| FreeRTOS scheduling/mutex/semaphore | context-safe handoff and bounded inversion distinction | port configuration and measured response |
| CMSIS-DSP FIR/IIR/FFT | persistent state, coefficients, units, and partition oracle | kernel output/cycle validation |
| UART DMA Pi bridge/frame | frame identity, CRC scope, sequencing, and queue accounting | physical link behavior and parser test |
| sleep/Stop/Standby power | wake epoch, qualification, and state accounting | actual current and retention behavior |
| Cortex-A76/NEON/cache | different processor/OS boundary and local freshness gate | platform performance and coherency details |
| MCU versus MPU models | ownership and deadline boundary cannot be delegated by assertion | actual partition design |
| IDE/code generation/debug | revisions and observability alter evidence | toolchain-specific execution |
| SWD/JTAG/ITM/semihosting | debugging can perturb timing | measured observer effect |
| linker/sections/tensor arena | map is necessary but not sufficient memory evidence | target stack and DMA reachability |
| HAL GPIO/ADC/UART/timer | abstraction does not remove peripheral contract | actual driver semantics |
| CMSIS-NN INT8 kernels | manifest and integer golden vectors precede acceptance | target equivalence/latency |
| integration hero project | end-to-end path requires every boundary above | HIL, power, safety, and board test |

**Takeaway:** The referenced results constrain a design only under their stated premises; the source inventory supplies scope, not unexecuted hardware evidence.

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

### Q13. A three-buffer DMA stream has 25 ms blocks. A block completes at 0 ms. When may DMA start overwriting it, and what separate condition makes its decision usable?

**Checked answer:** With `K=3`, overwrite can begin after `(3-1)*25=50 ms`, assuming equal uninterrupted block fills. The decision is usable only if its action age is within the separately declared freshness bound.

**Reasoning:** The completed block is one of the three total stores. DMA fills the two other stores from 0–25 ms and 25–50 ms, then returns to the completed storage. Counting 75 ms would wrongly include the interval in which the completed block was being filled.

**Worked check:** If `t_end=2.000 s`, `t_use=2.040 s`, and `A_max=30 ms`, storage release before 50 ms is safe but the action is stale at 40 ms. Integrity and usefulness have different verdicts.

**Follow-up:** If descriptors have 20, 25, and 30 ms lengths, enumerate the next descriptors instead of applying the equal-block formula.

**Acceptable alternative:** An answer that derives 50 ms from the explicit schedule and states its assumptions.

**Red flag:** “Three buffers gives 75 ms” or “release before deadline is enough.”

**Scoring:** 3 points: 1 for `(K-1)`, 1 for unit-correct result, 1 for freshness distinction.

### Q14. An integer target output disagrees with a float reference only for an input near the maximum sensor code. What evidence sequence distinguishes saturation from a layout mismatch?

**Checked answer:** First verify raw input bytes, declared dtype/signedness, and calibrated range; then inspect preprocessing saturation counters and intermediate integer tensors at the first divergence. A layout mismatch usually changes positions broadly, while saturation should correlate with range boundary and clipping evidence.

**Reasoning:** Final class disagreement is too late to locate the contract break. A maximum-valued input exercises affine conversion, intermediate width, quantization range, and kernel semantics simultaneously. Golden vectors must include boundary values, not only ordinary samples.

**Worked check:** For INT8 `q`, any attempted value beyond 127 that is clamped to 127 is saturation evidence if the implementation records the clamp. A shape `[1,36]` interpreted as `[36,1]` can preserve many values but route them to different feature positions; its symptom need not correlate with magnitude.

**Follow-up:** Pin the model manifest and operator resolver; an unsupported-operation substitution is a third competing cause.

**Acceptable alternative:** A layered comparison beginning with bytes/metadata and ending at the first diverging tensor.

**Red flag:** Re-quantize repeatedly until labels match, or compare only aggregate accuracy.

**Scoring:** 3 points: boundary vector, intermediate diagnosis, and a discriminating expected signal.

### Q15. A debug build meets an 80 ms response limit but the optimized release build misses it. What may and may not be concluded?

**Checked answer:** Neither build proves the other. Compare input trace, artifact/configuration identity, compiler flags, priorities, trace instrumentation, and timing origin. The release miss is actionable evidence for that release configuration; the debug pass does not waive it.

**Reasoning:** Breakpoints, semihosting, logging, optimization, inlining, stack layout, and scheduling perturbation can change execution and races. A meaningful comparison requires the same oracle events and explicit record of what differs.

**Worked check:** If debug emits 100 trace records/s at 1,600 payload bytes/s while release emits none, transport and ISR interference differ even before code-generation differences are considered. The correct next test is a bounded, minimally intrusive trace, not removal of the deadline.

**Follow-up:** If the trace itself changes the event order, state that as an observation limitation and use a smaller capture or hardware-supported mechanism where available.

**Acceptable alternative:** Any answer that refuses to generalize across builds and proposes a controlled comparison.

**Red flag:** “Optimized code is always faster” or “debug timing is the production result.”

**Scoring:** 3 points: configuration control, observer effect, and a bounded verification plan.

### Interview answer-evidence rubric

| Criterion | Strong answer observable | Weak but repairable answer | Reject/coach signal |
|---|---|---|---|
| units and origins | names block end, action time, bytes versus symbols | correct arithmetic without origin | treats baud as bytes/s |
| ownership | names writer, claimant, release and reuse | says “double buffer” without timing | permits reader/writer overlap |
| diagnosis | offers competing causes plus a signal separating them | names one plausible cause | changes a model threshold first |
| evidence scope | distinguishes oracle, target test, and field/safety evidence | says “test it” without artifact | calls illustrative values measured |
| safe behavior | rejects/contains a violated contract with a reason | says “retry” without bounds | uses an ACK as proof of action |

For a scored interview, accept a different architecture or API if it preserves the required reasoning. Do not award correctness for reciting a framework function without its context restrictions. Ask the candidate to state exactly which timestamp feeds the freshness check; this exposes a common but consequential ambiguity.

If a candidate chooses an MCU/MPU split, request a failure trace in which the remote side pauses. A defensible answer specifies local behavior while the remote result is late, not merely a faster remote model. If a candidate proposes an RTOS, request the lock owner, bounded critical section, and interrupt interference assumptions rather than an unqualified priority claim.

**Source inventory:** [directory.txt lines 1094-1110](../../../../Templates/directory.txt). **Source syllabus:** [edge-arm-firmware](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-arm-firmware). **Evidence boundary:** all arithmetic and oracle traces are illustrative/hand-derived; no target build, board/HIL, power, clinical, or safety validation was executed.

The oracle inputs are deliberately hardware-independent: sample identifiers, integer windows, declared timing, and frame fields can be tested in a host harness before any board is available. They do not model ADC analog error, DMA controller semantics, cache configuration, task-port behavior, UART electrical behavior, sensor settling, power draw, clinical interpretation, physical actuation, or fault containment.

For a board test, retain the same event schema and add the exact part revision, clock configuration, compiler/linker versions, model and manifest hashes, DMA configuration, task priorities, input source, timestamp source, and capture method. Report a passed or failed assertion with its input trace and environment; do not collapse it to “firmware works.”

For a hardware-in-the-loop or safety claim, additionally define the controlled plant/actuator boundary, hazard-specific safe state, fault injection, independent observation, stop conditions, recovery authorization, and review owner. None of those conditions is satisfied by the software traces above.

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

