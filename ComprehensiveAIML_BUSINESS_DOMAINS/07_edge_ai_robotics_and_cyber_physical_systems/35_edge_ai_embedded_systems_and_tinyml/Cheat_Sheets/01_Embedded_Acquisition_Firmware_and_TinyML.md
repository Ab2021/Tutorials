---
title: "Embedded Acquisition, Firmware Contracts and TinyML Integration"
topic: "Embedded acquisition, firmware contracts and TinyML integration"
collection: "module35-edge-ai-embedded-tinyml"
file_id: "01"
level: "foundation"
evidence: "DERIVED"
created: "2026-11-09"
last_reviewed: null
content_status: "draft"
status: "draft"
version: "2.1"
review_trigger: "Change to target MCU, scheduler, model artifact, UART protocol, or power profile"
notation_style: "plain-text"
prerequisites: ["Basic C memory ownership", "rates and SI units"]
pattern_tags: ["bounded-buffer", "deadline-contract", "safe-handoff"]
owner_role: "embedded systems reviewer"
source_index: "INDEX.md"
---

# Embedded Acquisition, Firmware Contracts and TinyML Integration

## 1. File Header

**Purpose:** Build one ownership and timing contract that runs from ADC conversion, through a bounded TinyML decision, out to a UART handoff, and can be falsified on a host before any board exists.

**Prerequisite skills:** C object lifetime and pointer aliasing, unsigned modular arithmetic, SI unit conversion, and the ability to read a linker map. If you cannot say who writes a buffer and when a reader may touch it, sections 4 and 8 will read as folklore rather than mechanism.

**Key outcome:** Given sample rate, channel count, block size, serial framing, decision budget and memory map, compute feasibility, name the first violated contract, and state the observation that would distinguish it from its nearest look-alike.

**Exclusions:** This sheet does not cover medical interpretation, functional-safety certification, a specific board bring-up, or measured latency on any device. No number here is a measurement of a shipped product.

**Level:** Foundation with production boundaries. The arithmetic is elementary; the difficulty is in the bookkeeping.

**Evidence:** Protocol and scheduling reasoning is derived. Every numeric example is ILLUSTRATIVE and hand-computed. Device parameters are nominal data-sheet style values used as inputs, not observations of a target.

**Retrieval cue:** Model accuracy looks good offline yet the device drops, repeats, or mislabels windows — start at sections 4.4, 5, and 8, then check the epoch counter before touching the model.

**Scope note:** The unit inventory names fifteen firmware topics; it supplies no original bodies. This sheet teaches the visible subjects and does not certify any particular STM32F401RE configuration, CMSIS version, FreeRTOS port, or deployed device behaviour.

**Reading order:** Definition table first, then the four formula families, then the feasibility walkthrough. The failure catalogue in section 8 is written to be usable as a checklist without reading anything else.

**Takeaway:** A TinyML output is defensible only while the sample, buffer, clock and decision that produced it remain individually identifiable.

## 2. Concept Map / Mental Model

```text
analog front end -> ADC conversion -> DMA write into half-buffer -> completion IRQ
completion IRQ -> publish pointer + sequence -> ready queue -> consumer claim
consumer claim -> timestamp check -> feature window -> DSP -> quantize -> inference
inference -> class + score + epoch -> freshness gate -> UART frame -> host parser
watchdog <- deadline counters <- overrun counters <- ack/timeout accounting
```

Read the diagram as four separate contracts stacked end to end. The first contract is electrical: the sensor output must settle inside the ADC sampling window, otherwise every later stage inherits an unmodelled error. The second is ownership: the DMA engine and the consumer task must never touch the same storage at the same time. The third is identity: a block that arrives must carry enough metadata to say which sample interval produced it. The fourth is usefulness: a correct decision delivered after its use window has closed is a defect, not a late success.

Each arrow in the diagram is a place where a guarantee silently weakens. The DMA-to-task arrow transfers a pointer, but what actually matters is the pairing of pointer, length, sequence and visibility. If any of those four is implicit, a reader downstream has to guess, and a guess is indistinguishable from a bug until the day it is not.

The decision path crosses three independent clocks. The sample clock determines when measurements exist. The CPU tick determines when code runs. The wall clock or host clock determines when an operator sees a result. Any latency statement that mixes these origins is unfalsifiable; a 30 ms figure can mean 30 ms of sampling, 30 ms of queueing, or 30 ms of transport depending on where the timestamps were taken.

**Takeaway:** Publish immutable blocks, and validate age at the point of use rather than at the point of production.

### 2.1 The four contracts in one sentence each

Acquisition contract: every published block covers exactly `N` consecutive samples per channel with no gap and no overlap, and its bytes are visible to the consumer before the sequence counter increments.

Scheduling contract: the worst-case response time of the inference task, including blocking and interference, is below the block period, not merely below it on average.

Artifact contract: the model that runs, the preprocessing constants that feed it, and the class-label order that decodes it are identified by one hash and cannot drift independently.

Action contract: a consumer rejects a decision whose age exceeds the declared bound, and the rejection is a defined, observable state rather than a silent discard.

### 2.2 Where the model actually sits

The neural network is a small pure function from a fixed-shape integer tensor to a fixed-shape integer tensor. Everything that makes embedded deployment hard sits outside that function: obtaining the tensor on time, obtaining it in the declared numeric format, and preventing downstream code from treating a low-confidence output as an instruction.

Learners routinely over-invest in model compression and under-invest in the window ledger. A 40 KB model in a 512 KiB flash part leaves room; a window that is silently 16 ms late every 4.3 s leaves none.

A useful mental test before any code is written: if the model were replaced by a stub that always returns class 0, which counters would still have to be correct? The answer — sequence, epoch, block count, timestamp origin, frame CRC — is the part of the system that must be built and tested first, because it survives every model change.

### 2.3 Reading a defect report as a contract audit

When a field report says "sometimes it says the wrong thing", the productive next step is not to open the model. It is to ask which of the four contracts has no counter attached. A team that cannot answer "how many blocks were published and how many were consumed in that hour" has no way to distinguish a lost sample from a misclassified one, and every subsequent discussion is guesswork dressed as diagnosis.

The audit sequence is fixed and cheap. Publish count versus consume count. Sequence gaps versus resets. Frame CRC failures versus parse successes. Rejection count at the freshness gate. If all four ledgers are complete and reconcile, the defect is inside the function and the model deserves attention. If any ledger is missing, the defect may be anywhere and the model is the least likely culprit.

This ordering is why the sheet spends its length on counters rather than on compression ratios. Compression changes a number in a table; a missing ledger changes what questions are askable.

## 3. Foundational Definitions

| Term | Meaning | Units/type | Boundary |
|---|---|---|---|
| `f_s` | sample rate per channel | samples/s | positive; not the aggregate rate across channels |
| `M` | number of simultaneously sampled channels | count | integer, at least 1 |
| `N` | samples per block, per channel | samples | positive integer |
| `T_w` | block duration | s | equals `N/f_s` only when timing is uniform |
| `b` | ADC result width used on the wire | bit | 8, 12, 16 or 24 in practice |
| `B_b` | bytes per block after packing | byte | `N*M*ceil(b/8)` before padding |
| `D` | decision deadline | s | meaningless until a clock origin is named |
| `A` | action age at use | s | `t_use - t_sample_end` |
| `A_max` | maximum acceptable action age | s | declared per use, not per model |
| `C_i` | worst-case execution time of task `i` | s | measured or bounded, never nominal |
| `R_i` | worst-case response time of task `i` | s | includes blocking and interference |
| `U` | processor utilisation | fraction | sum of `C_i/T_i`; ignores blocking |
| `K` | number of DMA buffer stores in rotation | count | 2 for double, 3 for triple |
| `J` | completion jitter, peak deviation | s | distribution tail, not standard deviation alone |
| `seq` | monotonically increasing block counter | uint32 | wraps; comparison must be wrap-safe |
| `epoch` | boot or configuration generation | uint16 | changes on reset, reconfiguration, model swap |
| `q` | stored integer code after quantization | int8 or uint8 | saturating, not wrapping |
| `s` | quantization scale | real units per LSB | strictly positive |
| `Z` | quantization zero point | integer code | lands inside the code range |
| `x̂` | dequantized real value | real units | `s*(q-Z)` |
| `T_fill` | time to fill one DMA store | s | equals `T_w` for uniform fills |
| `T_reuse` | earliest instant DMA may overwrite a store | s | `(K-1)*T_fill` for equal fills |
| `W_c` | worst-case consumer service time | s | compute plus transport, excluding wait |
| `L_q` | mean queue occupancy | blocks | by Little's law, `lambda*W` |
| `lambda` | publish rate | blocks/s | equals `1/T_w` when no loss |
| `mu` | sustainable consume rate | blocks/s | measured under worst-case input |
| `W` | mean time in system | s | queue wait plus service |
| `rho` | offered load on a service | fraction | `lambda/mu`; must stay below 1 |
| `E_blk` | energy per block | J | sum over named states |
| `I_act` | active current | A | measured at a stated voltage |
| `I_idle` | idle current | A | excludes regulator quiescent draw unless stated |
| `f_cpu` | core clock | Hz | the clock the cycle counts were taken at |
| `MAC` | multiply-accumulate operation | count | one per weight per activation |
| `arena` | scratch memory for tensor lifetimes | byte | not the same as model size |
| `WCET` | worst-case execution time | s | a bound with a method attached |
| `ODR` | sensor output data rate | Hz | must match or exceed `f_s` before decimation |
| `alias band` | frequencies folded by decimation | Hz | `[f_s/2, ODR/2)` after decimation by `ODR/f_s` |
| `e_esc` | fraction of payload bytes that need escaping | fraction | 0 to 1; a property of the framing, not the data |
| `B_tok` | burst depth absorbed by an admission budget | blocks | converted to time by `B_tok*T_w` |

Three of these definitions do the most work in practice. `T_w` is a property of the sampling schedule, not of the buffer length: if a descriptor is 30 ms but the nominal block is 25 ms, the schedule has already diverged from the arithmetic that sized the ring. `A` requires an end-of-sample timestamp, so a timestamp taken at interrupt entry is already wrong by the interrupt latency plus the tail of the transfer. `R_i` is not `C_i`; the gap between them is blocking time plus interference, and that gap is what usually breaks the deadline.

A fourth definition earns its own warning: `U` is computed from per-task utilisation and therefore cannot see a high-priority interrupt that arrives every 2 ms and runs for 300 microseconds. Add interrupt load explicitly as a utilisation term, or the acceptance test will pass on paper and fail on the bench.

`J` deserves a note of its own because it is the parameter most often replaced by its own mean. A completions stream with a 20 ms nominal period and a 3 ms peak deviation has a short interval of 17 ms somewhere in every long run; a ring sized from 20 ms and then validated over a short capture will never see that interval.

`mu` is the term teams substitute their nominal throughput for. A consumer that sustains 50 blocks per second on average with a 30 ms worst-case block does not sustain 50 blocks per second; its sustainable rate under worst-case input is 33 per second, and an admission design built on the mean loses data only in the conditions that matter.

### 3.1 Clock, conversion and frame vocabulary

Conversion time is the interval between the ADC sampling switch opening and the result register becoming valid. It is not the same as the interval between two published blocks, and conflating them makes a 12-bit conversion look like a 20 ms operation.

Frame is the unit the UART carries: a start bit, the data bits, an optional parity bit and one or more stop bits. An eight-data-bit, no-parity, one-stop-bit frame is ten symbols, which is why payload capacity is `baud/10` and not `baud/8`.

Symbol rate equals bit rate only for binary modulation with no encoding expansion. Any escaping scheme, byte stuffing, or base64 wrapping spends symbols on representation rather than payload, and a utilisation budget computed without that expansion is optimistic by the expansion factor.

Sequence is a count of published blocks. It answers "how many" but not "which configuration", so it must travel with `epoch` wherever a consumer might survive a restart. A consumer that deduplicates on sequence alone will silently merge two different boot sessions.

A fifth term worth separating is settle time: the interval a sensor or analog front end needs after being enabled or after a channel switch before its output is inside its specified error band. It is a datasheet number with a test condition attached, and using it without the condition is how a wake cycle silently loses its first three samples.

### 3.2 Execution and synchronisation vocabulary

Interrupt latency is the delay from an event to the first instruction of its handler. It is a distribution with a long tail, not a constant, because higher-priority handlers and bus contention both extend it.

Priority inversion occurs when a high-priority task waits on a resource held by a low-priority task which is itself preempted by a medium-priority task. Priority inheritance bounds the inversion; it does not remove blocking, and it does nothing for inversion caused by a shared peripheral rather than a mutex.

Jitter is the variation in the interval between events that are nominally periodic. A scheduler can meet every mean period while producing a 3 ms tail on a 20 ms block, which is enough to overrun a two-store ring sized for zero jitter.

Critical section length is the longest interval during which interrupts that matter to a deadline are disabled. It is bounded by code, not by intent, and it must be measured with interrupts enabled/disabled toggling observable on a pin.

Blocking time is the interval a task spends ready-but-not-running because it waits on a resource. Interference is the interval it spends not running because something of higher priority runs. Both extend response time; only interference appears in the utilisation sum, which is why the sum is optimistic.

### 3.3 Model-artifact vocabulary

Quantization is the mapping from a real tensor to integer codes with an affine rule. Calibration is the process that chooses `s` and `Z` from representative data. Both are preprocessing decisions that must be versioned with the model, because the same weight bytes under a different scale produce a different function.

Operator resolution is the mapping from a serialized operation to a kernel implementation. A resolver that falls back from an optimized kernel to a reference kernel preserves approximate numerics but changes timing, so a latency budget validated on one resolver is not automatically valid on another.

Metadata is the manifest describing shapes, dtypes, quantization parameters, and often the intended input layout. It is executable information in the sense that a mismatch silently changes results, which is why comparing final labels is too coarse a test.

Arena is the scratch region whose lifetime is managed by the interpreter. Its size depends on the operator schedule, so it can grow when the graph is rewritten even though the weight bytes are unchanged.

Calibration set coverage is a term rarely written down and often decisive: the scales are computed from a sample of inputs, and the range of that sample becomes the representable range. If the deployment sees amplitudes 20 percent above the calibration maximum, every such window clamps, and the clamped windows are exactly the interesting ones.

### 3.4 Memory-map vocabulary

Flash residency is where the weights live; execution residency is where they are read. On a part with no cache, reading weights from flash every MAC costs flash access time; on a part with cache, the first pass pays a miss and subsequent passes depend on whether the working set fits.

Static allocation is memory reserved at link time. Its size is visible in the map file and its failure mode is a link error, which is the cheapest possible failure. Heap allocation moves the same failure to run time, where it appears as a fragment-induced allocation failure after hours of operation.

The stack is the region where automatic variables and interrupt frames live. Its high-water mark is discovered by painting it with a pattern and measuring the deepest overwritten byte, not by reading the linker script.

Alignment is the requirement that a multi-byte object start at an address that is a multiple of its size or of a bus-line size. Misaligned DMA on some parts faults, on others silently corrupts adjacent bytes, and on others is merely slow — three different symptoms from one configuration error.

### 3.5 Unit conventions that cause real defects

Memory is quoted in kilobytes by part vendors (10^3) and in kibibytes by linker maps (2^10), and a 96 KiB SRAM is 98,304 bytes, not 96,000. A percentage computed against the wrong base is wrong by 2.4 percent relative, which is enough to hide a two-kilobyte overrun.

Storage endurance is quoted in erase cycles per page, not in bytes. A write-rate budget that never mentions page size cannot predict wear, and a log that writes 800 bytes per second into 2 KiB pages performs one erase per page every 2.56 s.

Current is quoted at a stated voltage and with stated peripherals enabled. A 12 mA figure at 3.3 V with the radio disabled is not the same number as 12 mA at 1.8 V with the radio enabled, and averaging them produces a battery life that no measurement will reproduce.

Rounding conventions matter at the boundaries. A block duration computed as `32/1600 = 0.020 s` is exact; a value printed as 20 ms after `int(1000*0.019999)` is one sample short, and a sample count ledger computed from the printed value drifts by 0.005 percent per block, which is 2.6 s per day.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**FORMULA**

```text
T_w   = N / f_s                 [s]
B_b   = N * M * (b/8)           [byte]
R_b   = M * f_s * (b/8)         [byte/s]
U_bus = R_b / C_link            [fraction]
C_link = baud / 10              [byte/s, 8N1]
```

**WHERE:** `N` is samples per channel per block (samples), `f_s` per-channel rate (samples/s), `M` channels, `b` bits per sample as packed on the wire, `baud` the configured line rate (symbols/s), and `C_link` the 8N1 payload ceiling (byte/s). All rates are per second; no averaging window is implied.

**CONSTRAINT:** The equalities hold only when sampling is uniform, packing has no padding, and the frame uses one start bit, eight data bits, no parity, one stop bit. If the ADC over-samples and decimates internally, `f_s` is the post-decimation rate and the internal conversion rate must still satisfy the sensor settling requirement.

**NUMERICAL EXAMPLE:** Take `f_s=1600`, `M=3`, `N=32`, `b=16`, link `921600` baud.

```text
T_w   = 32/1600 = 0.020 s = 20 ms
B_b   = 32*3*2 = 192 byte
R_b   = 3*1600*2 = 9600 byte/s
C_link = 921600/10 = 92160 byte/s
U_bus = 9600/92160 = 0.1042
```

**INTERPRETATION:** Raw acquisition consumes about 10.4 percent of a 921600-baud link, so the link is not the binding constraint for streaming raw blocks. That headroom is not free capacity for features and logs: the decision frames, the acknowledgement traffic and the diagnostic stream all share the same wire, and the honest budget assigns each of them a slice before declaring headroom. A 192-byte block at 50 blocks per second is 9.6 kB/s, which is a comfortable figure that becomes uncomfortable the moment a debug log is added at one line per sample.

A second reading of the same numbers is worth stating, because it is the one that decides architecture. The payload rate equals `M*f_s*b/8` regardless of how the data is blocked, so changing `N` from 32 to 64 changes interrupt frequency and latency but not bandwidth. Teams frequently double the block size hoping to relieve a link that is already saturated; the computation says the link is indifferent and only the CPU and the reaction time change.

### 4.2 General case

**FORMULA**

```text
C_link  = (baud / (1 + n_d + p + n_s)) * (1 - e_esc)
T_reuse = sum over the next (K-1) descriptors of T_fill[j]
A       = t_use - t_sample_end
R_i     = C_i + B_i + sum over j in hp(i) of ceil(R_i/T_j) * C_j
feasible if max over i of R_i <= D_i
```

**WHERE:** `n_d` data bits, `p` parity bits, `n_s` stop bits, `e_esc` the fraction of symbols spent on escaping; `T_fill[j]` the j-th following descriptor duration (s); `B_i` worst-case blocking time for task `i` (s); `hp(i)` the set of higher-priority tasks; `D_i` the deadline of task `i`.

**CONSTRAINT:** The response-time recurrence is solved by fixed-point iteration starting from `R_i = C_i + B_i`; it is valid for fixed-priority preemptive scheduling with independent periodic tasks and bounded blocking. It does not model self-suspension, and it does not model a task that blocks on an I/O device whose driver re-enables interrupts late.

**NUMERICAL EXAMPLE:** Descriptors of 20, 25 and 30 ms with `K=3`; the completed store may be reused only after the two following stores complete.

```text
T_reuse = 25 + 30 = 55 ms   (not (3-1)*20 = 40 ms)
```

Then, with `C=0.50 ms`, `B=2.00 ms`, one higher-priority task with `C_hp=0.03 ms` and `T_hp=20 ms`:

```text
R0 = 0.50 + 2.00 = 2.50 ms
R1 = 2.50 + ceil(2.50/20)*0.03 = 2.50 + 0.03 = 2.53 ms
R2 = 2.53 + 0.03 = 2.56 ms  -> converged
2.56 ms <= 20 ms deadline, margin 17.44 ms
```

**INTERPRETATION:** The general case shows why "three buffers gives you two block periods" is a slogan rather than a result. With unequal descriptors the reuse instant is the sum of the specific descriptors that follow, and using the nominal block period understates protection when later descriptors are longer. The response-time recurrence converges quickly here because interference is small; on a loaded system the same recurrence can jump by a full period per iteration, which is exactly the behaviour that makes a mean-latency measurement useless.

Note also what the recurrence does not contain: the time the task spends waiting for a DMA block that has not arrived. That wait is a self-suspension, and if a task both waits for input and computes, the correct model is a jitter term on its period rather than a blocking term on its response. Mixing the two is how a scheduler analysis reports 2.56 ms while the observed response is 20 ms.

### 4.3 Variants

**FORMULA**

```text
E_blk  = V * (I_act*t_act + I_idle*(T_w - t_act))        [J per block]
P_avg  = E_blk / T_w                                     [W]
t_batt = (Q_mAh/1000 * 3600 * V_batt) / E_blk * T_w      [s]
SNR_q  = 6.02*b_q + 1.76                                 [dB, full-scale sine]
e_max  = s/2 ,  var(e) = s^2/12 ,  sd(e) = s/sqrt(12)
```

**WHERE:** `V` supply voltage (V), `I_act` current while computing (A), `t_act` active interval (s), `I_idle` current otherwise (A), `Q_mAh` cell capacity (mAh), `V_batt` nominal cell voltage (V), `b_q` quantizer bits, `s` quantization step in real units.

**CONSTRAINT:** The energy expression is a two-state model; it is valid only if the states are exhaustive for the interval and if the regulator efficiency is folded into the currents. The SNR expression assumes a full-scale sinusoid and a uniform quantizer with negligible overload; it does not describe a signal that occupies a small part of the range, where effective SNR is much worse.

**NUMERICAL EXAMPLE:** `V=3.3`, `I_act=12 mA` for `t_act=6 ms`, `I_idle=1.2 mA` for the remaining 14 ms of a 20 ms block.

```text
E_blk = 3.3*(0.012*0.006 + 0.0012*0.014)
      = 3.3*(7.2e-5 + 1.68e-5) = 3.3*8.88e-5 = 2.9304e-4 J
P_avg = 2.9304e-4/0.020 = 0.014652 W = 14.65 mW
t_batt= (0.220*3600*3.7)/0.014652 = 2930.4/0.014652 = 199,999 s = 55.6 h
```

Then with `s=0.05` real units per LSB:

```text
e_max = 0.025 units, var = 0.0025/12 = 2.0833e-4, sd = 0.01443 units
```

**INTERPRETATION:** The two-state model predicts 55.6 hours from a 220 mAh cell, and the honest reading is that this is the best case for the states listed. Regulator quiescent current, sensor draw, and the sleep-transition cost are absent, and each of them can dominate an idle current of 1.2 mA. The quantization figures say a step of 0.05 costs at most 0.025 units of amplitude error; if the physical signal of interest has an amplitude of 0.05 units, the quantizer has destroyed the measurement regardless of model quality.

**Second variant — integer requantization between two layers.**

```text
M     = (s_in * s_w) / s_out         [dimensionless]
M0    = round(M * 2^30)              [Q30 integer multiplier]
q_out = clamp((acc * M0) >> 30, q_min, q_max) + Z_out
```

**WHERE:** `s_in` input activation scale (real units per LSB), `s_w` weight scale, `s_out` required output scale, `acc` the integer accumulator for one output element, `M0` a 31-bit multiplier, and the shift is fixed at 30 by the Q30 representation. `Z_out` is the output zero point.

**CONSTRAINT:** The product `acc*M0` must fit the chosen integer width. A 31-bit multiplier times a 32-bit accumulator needs up to 63 bits, so a 32-bit part must either use two multiply-and-adds or accept a narrower multiplier. Truncation rather than rounding introduces a bias of half an LSB per layer in one direction; that bias accumulates linearly with depth while the random part accumulates as a square root.

**NUMERICAL EXAMPLE:** `s_in=0.02`, `s_w=0.05`, `s_out=0.4`, `acc=12345`.

```text
M   = (0.02*0.05)/0.4 = 0.001/0.4 = 0.0025
M0  = round(0.0025*2^30) = round(0.0025*1,073,741,824) = 2,684,354.56 -> 2,684,355
acc*M0 = 12345*2,684,355 = 33,138,362,475  (needs 35 bits: exceeds a 32-bit register)
>> 30  = 33,138,362,475 / 1,073,741,824 = 30.86 -> truncated to 30
exact real value = 12345*0.02*0.05 = 12.345 units ; 12.345/0.4 = 30.8625 codes
```

**INTERPRETATION:** The multiplier arithmetic is exact to within one LSB of the output, and the example shows the error in both directions: 0.8625 codes, or 0.345 real units at `s_out=0.4`. That single-layer error is tolerable; three such layers with truncation in the same direction give a systematic offset near 1.2 units, which is enough to move a decision boundary when class margins are a few units. The 35-bit intermediate is the second lesson: the arithmetic cannot run in the accumulator width of a 32-bit core without care, and a silent 32-bit wrap produces outputs that look like a model error rather than a numeric one.

**Third variant — jitter consumed from the reuse window.**

```text
T_excl(K) = sum of the (K-1) fills after the completion instant
T_excl_min = (K-1)*T_w - (K-1)*J            [worst-case early arrivals]
depth_needed = ceil((W_c + J)/T_w) + 1      [stores to absorb a slow consumer]
```

**WHERE:** `T_excl` is the reader's guaranteed exclusivity interval (s), `J` the peak completion jitter (s), `W_c` the consumer's worst-case service time (s), and `depth_needed` an integer count of stores.

**CONSTRAINT:** `T_excl_min` assumes every fill can arrive as much as `J` early; that is the pessimistic case and the correct one for sizing. `depth_needed` assumes the consumer starts immediately on claim and that service time does not itself jitter; if it does, add its jitter to `W_c` first.

**NUMERICAL EXAMPLE:** `K=3`, `T_w=20 ms`, `J=3 ms`, `W_c=6 ms`.

```text
T_excl     = 2*20 = 40 ms nominal
T_excl_min = 2*20 - 2*3 = 34 ms worst case
depth_needed = ceil((6+3)/20) + 1 = ceil(0.45) + 1 = 1 + 1 = 2
```

**INTERPRETATION:** A two-store rotation is sufficient for a 6 ms consumer with 3 ms jitter, because the consumer releases long before reuse; the third store buys margin against a consumer that occasionally takes 30 ms, and 34 ms of worst-case exclusivity is exactly why. The number to remember is that jitter is subtracted from the reader's budget, not added to the block period: a ring sized at 40 ms of nominal protection actually offers 34 ms when the timer is allowed to arrive early.

### 4.4 Limits and counterexamples

**FORMULA**

```text
A = t_use - t_sample_end
stale if A > A_max
q = clamp(round(x/s) + Z, q_min, q_max)
x̂ = s*(q - Z)
loss_rate = max(0, lambda - mu)          [blocks/s]
t_wrap = ring_bytes / write_rate          [s]
```

**WHERE:** `lambda` publish rate, `mu` sustainable consume rate, `ring_bytes` circular log capacity, `write_rate` bytes written per second. `q_min`,`q_max` are the code-range ends, `-128..127` for int8 and `0..255` for uint8.

**CONSTRAINT:** `A` is only computable if the timestamp marks the end of the last sample in the block, not the start of the transfer and not the interrupt entry. The loss expression is a fluid approximation valid over intervals long compared with `T_w`; over a single block, loss depends on phase and can be zero even when `lambda` exceeds `mu` transiently.

**NUMERICAL EXAMPLE:** A 512-sample feature window at 1600 Hz ends at `t_sample_end = 2.000 s`; the consumer acts at `t_use = 2.040 s` with `A_max = 30 ms`.

```text
A = 2.040 - 2.000 = 0.040 s = 40 ms > 30 ms -> stale
block is storage-safe (T_reuse = 55 ms) yet the decision must be rejected
```

**INTERPRETATION:** This is the counterexample the whole sheet is built around. Storage integrity and decision usefulness are independent verdicts: the buffer contract was never violated, and the decision is still unusable. Any test that asserts only "no overrun was reported" is answering a different question from "the decision was fresh enough to act on", and the two answers can diverge while every counter in the system reads zero.

A second counterexample attacks the quantization bound. With `s=0.05` and `Z=-3`, the representable range is computed as `[(q_min-Z)*s, (q_max-Z)*s]`, giving `[(-128+3)*0.05, (127+3)*0.05] = [-6.25, 6.5]` real units. An input of 7.0 units does not produce a small error; it produces a clamped code that is 0.5 units away and indistinguishable from a genuinely moderate signal. Saturation is a failure of the input contract, not a numeric rounding detail.

A third counterexample attacks the jitter arithmetic directly. Suppose the same three-store rotation is validated at 40 ms of nominal exclusivity and the consumer is measured at 35 ms worst case. Under `J=3 ms` the guaranteed interval is 34 ms, so the consumer overruns by 1 ms and the first symptom is a block whose tail belongs to the next fill — a single-sample corruption that survives every checksum computed downstream because the bytes are individually valid. The design fails not because the average was wrong but because the worst case was never in the model.

A fourth counterexample attacks decimation. A sensor produces data at `ODR=3200` Hz, firmware decimates by two to obtain `f_s=1600` Hz, and the anti-alias requirement applies to the band `[800, 1600)` Hz. A mechanical resonance at 1400 Hz appears after decimation at `1600-1400 = 200` Hz with undiminished amplitude if no filter preceded the decimation. A model trained on that same pipeline learns the artefact as a feature, and the artefact moves whenever the sample rate changes, which is exactly what a later firmware revision does.

A fifth counterexample attacks the link budget. At 115200 baud the 8N1 ceiling is `11520` bytes/s, and a 9600 byte/s raw stream uses 83.3 percent of it. With a framing scheme that escapes the bytes `0x7E` and `0x7D`, random binary payload expands by `2/256 = 0.78` percent, taking the rate to `9600*1.0078 = 9675` bytes/s and utilisation to 84.0 percent — still feasible. A payload that is structured rather than random, for example a header byte repeated once per record, can escape far more often, and a worst case in which every payload byte needs escaping doubles the stream to 19200 bytes/s, which exceeds the link and makes the design infeasible on that baud rate.

**Validity condition for the whole section:** every formula above assumes a single writer per buffer at any instant, a timestamp origin that is stated, parameters constant over the interval of interest, and a worst case rather than a mean wherever a deadline is involved. Where any of those fails, the formula is a model of a system that was not built.

### 4.5 Admission control under sustained overload

**FORMULA**

```text
rho        = lambda / mu                          [fraction]
backlog(t) = B_tok + (lambda - mu)*t              [blocks]
t_saturate = B_tok / (lambda - mu)                [s, when lambda > mu]
absorbed_burst = B_tok * T_w                      [s of input]
shed_policy: drop_newest | drop_oldest | decimate_decision_rate
```

**WHERE:** `B_tok` the admission budget in blocks (the ring depth minus the block in service), `lambda` the publish rate (blocks/s), `mu` the sustainable consume rate measured under worst-case input (blocks/s), `t` elapsed time (s). `drop_newest` preserves contiguity of the history at the cost of losing the newest evidence; `drop_oldest` preserves recency at the cost of a gap.

**CONSTRAINT:** The fluid model assumes `lambda` and `mu` are stationary over the interval. It also assumes the consumer never slows further as its queue grows, which is false for a consumer that logs its queue depth or that touches more memory per iteration when backlogged.

**NUMERICAL EXAMPLE:** `B_tok=8`, `T_w=20 ms`, `lambda=50`, `mu=42`.

```text
absorbed_burst = 8*0.020 = 0.160 s of input
rho            = 50/42 = 1.190
backlog rate   = 8 block/s
t_saturate     = 8/8 = 1.0 s
```

**INTERPRETATION:** The ring absorbs a 160 ms burst and then saturates exactly one second into a sustained 19 percent deficit. This is the arithmetic that separates two very different designs: a system sized for bursts needs 8 blocks of depth, while a system whose consumer is genuinely 16 percent slower than its producer needs either a faster consumer or an intentional shedding policy, and no amount of buffer depth substitutes for the missing 8 blocks per second.

The choice between shedding policies is a statement about what the downstream consumer can tolerate. A classifier that averages over windows tolerates a gap and prefers recency, so `drop_oldest` is defensible. A sequence-integrity check that rejects any gap must instead `drop_newest` and report the loss, because a silent gap would be read as a corrupt stream. Declaring the policy is the requirement; there is no universally correct one.

The third policy is the one most teams miss: decimate the decision rate rather than the data. If the consumer sustains 42 blocks per second and the model runs every fourth block, reducing the decision cadence to every fifth block reduces the work without losing any raw sample, and the loss budget then applies to decisions rather than to measurements.

### 4.6 Sequence wrap and session identity

**FORMULA**

```text
t_wrap  = 2^32 / lambda                     [s]
valid   = (uint32)(seq_new - seq_old) < 2^31
same_session = (epoch_new == epoch_old)
gap     = (uint32)(seq_new - seq_old) - 1   [blocks lost]
```

**WHERE:** `seq` is a 32-bit unsigned block counter, `epoch` a 16-bit generation counter incremented at every boot and at every model or configuration swap, `lambda` the publish rate in blocks/s. The comparison form is modular subtraction, not magnitude comparison.

**CONSTRAINT:** The half-range rule is valid only if genuinely older records are never compared across more than `2^31` increments, which at 50 blocks/s means 497 days. Any retention policy longer than that must store the epoch as well and compare sessions first.

**NUMERICAL EXAMPLE:** `lambda=50` blocks/s.

```text
t_wrap = 4,294,967,296/50 = 85,899,345.92 s
       = 994.2 days = 2.72 years
```

**INTERPRETATION:** A wrap is not a practical concern at this rate, which is exactly why a duplicate sequence number observed during a one-week test is evidence of a reset rather than of arithmetic reaching its end. That observation is the fastest way to distinguish "the device restarted" from "the host duplicated a record", and it costs one counter to make.

The epoch comparison is what makes the ledger usable across restarts. A host that deduplicates on sequence alone will discard a legitimate first record of a new session because its sequence value matches one from the previous session, and the discarded record is the one that would have shown the restart. Deduplicating on the pair `(epoch, seq)` keeps both facts.

The gap arithmetic is where off-by-one errors live. A jump from sequence 100 to sequence 105 means four blocks were lost, not five, because the difference is five and one of the five is the record that arrived. Reporting five overstates loss by 20 percent at that magnitude, and a 20 percent error in a loss figure is enough to send an investigation to the wrong subsystem.

## 5. Worked Numerical Example

The scenario is ILLUSTRATIVE: a three-axis accelerometer stream, a 512-sample sliding feature window, a 36-feature INT8 vector, a small fully connected classifier, and a UART decision frame. Target class: a Cortex-M4F with 512 KiB flash, 96 KiB SRAM, 64 MHz core, no data cache.

### 5.1 Step 1 — acquisition budget

```text
f_s = 1600 Hz per axis, M = 3, N = 32
T_w = 32/1600 = 20 ms
B_b = 32*3*2 = 192 byte per block
publish rate = 1/0.020 = 50 block/s
raw rate = 192*50 = 9600 byte/s
```

The block period is 20 ms, so the consumer has 20 ms of slack per block before the next publish arrives. That slack is the entire margin, and it is consumed by queueing, inference, and transport in that order.

### 5.2 Step 2 — DMA rotation and reuse

With `K=3` equal stores of 20 ms each:

```text
T_reuse = (3-1)*20 = 40 ms from the completion instant
```

Interpreting 40 ms as "the reader has 40 ms of guaranteed exclusivity" is correct only if DMA fills the other two stores back to back. If a descriptor is lengthened to 30 ms by a configuration change, the interval becomes 50 ms for the store that completes just before the long descriptor and 40 ms for the one that completes just after it. The rotation is a schedule, not a constant.

### 5.3 Step 3 — feature window and decision latency

```text
window = 512 samples
window span = 512/1600 = 0.320 s = 320 ms
window advance per block = 32 samples = 20 ms
features per window = 36 (mean, sd, min, max per axis, plus 12 derived)
inference: 36->16->8->3
MACs = 36*16 + 16*8 + 8*3 = 576 + 128 + 24 = 728
cycles at 1 MAC/cycle = 728
t_infer = 728/64e6 = 11.4 us ideal
with 2.5x kernel and framework overhead = 28.4 us
```

End-to-end for a decision stamped at the window end:

```text
t_total = 0 (sampling complete) + 0.028 ms + 0.499 ms (UART) + 1.5 ms (queue) = 2.03 ms
```

### 5.4 Step 4 — transport frame

```text
header 8 byte + payload 36 byte + CRC 2 byte = 46 byte
t_frame = 46/92160 = 0.499 ms
at 5 decision/s (every tenth block): 5*46 = 230 byte/s
U_bus = 230/92160 = 0.0025
```

Five decisions per second is a design choice, not a physical limit: the model could run at 50 Hz and the wire would carry `50*46 = 2300` bytes per second, still 2.5 percent of the link. The reason to decimate the decision rate is downstream consumption, not bandwidth.

### 5.5 Step 5 — memory accounting

```text
weights: 728 int8  = 728 byte
biases : 27 int32  = 108 byte
metadata + header  = 200 byte (illustrative)
model image        = 1036 byte
arena (63 int8 activations + interpreter overhead) = 2048 byte
DMA stores: 3 * 192 = 576 byte
feature window: 512 * 3 * 2 = 3072 byte (or 1536 byte as int16 pairs)
ring for logging: 8 * 192 = 1536 byte
stack (painted high-water, illustrative): 4096 byte
```

Total static demand is 12,364 bytes against 98,304 bytes of SRAM, so memory is not the binding constraint. The interesting columns are the ones that grow without a link error: the arena can grow with a graph rewrite, and the stack high-water mark grows with a deeper call chain under interrupt.

### 5.6 Step 6 — energy and lifetime

```text
E_blk = 2.9304e-4 J (from 4.3), P_avg = 14.65 mW
energy per inference = 3.3*0.012*28.4e-6 = 1.125e-6 J
share of block energy spent inferring = 1.125e-6/2.9304e-4 = 0.38 percent
```

The inference is 0.38 percent of block energy while the acquisition and DSP states dominate: the active interval is `7.2e-5` of `8.88e-5` J, or 81.1 percent, and the idle term is the remaining 18.9 percent. Optimising the model further is therefore an energy non-story at this duty cycle; the actionable lever is the active interval `t_act`, which is set mostly by the DSP and logging around the model rather than by the model itself.

### 5.7 Independent arithmetic re-check

Re-derive the two numbers that everything else depends on, by a different route.

```text
Check A (block rate from duration):
  20 ms per block -> 1000/20 = 50 blocks per second. matches 1/0.020.
Check B (raw rate from block size):
  192 byte * 50 = 9600 byte/s. cross-check with M*f_s*b/8 = 3*1600*2 = 9600. matches.
Check C (energy per second):
  E_blk*50 = 2.9304e-4*50 = 1.4652e-2 J/s = 14.65 mW. matches P_avg.
Check D (battery hours from Joules):
  2930.4 J / 0.014652 J/s = 199,999 s; 199999/3600 = 55.56 h. matches 55.6 h.
Check E (frame occupancy):
  5*46/92160 = 230/92160 = 0.002496. earlier 0.0025. matches to 4 decimals.
Check F (requantization cross-check):
  728 MAC * 5 decision/s = 3640 MAC/s; at 64 MHz that is 5.7e-5 s of cycles per
  second = 0.057 ms/s, which matches the 28.4 us per decision times 5.
```

Check D is the one worth repeating in an exam: a battery figure is a division of stored energy by average power, so its sensitivity to any single input is that input's share of the total. A 10 percent change in the idle current moves life by `0.10*0.189 = 1.89` percent, or about 1.05 hours, while the same relative change in the active current moves it by 8.11 percent, or 4.5 hours. Anyone quoting 55.6 hours must also quote the two currents and the cell capacity, or the number is unfalsifiable.

### 5.8 Sensitivity

| Parameter | Base | Change | Effect recomputed | Reading |
|---|---|---|---|---|
| block size `N` | 32 | 64 | `T_w` 40 ms, rate unchanged at 9600 B/s, ring latency doubles | fewer interrupts, coarser reaction |
| channel count `M` | 3 | 6 | raw 19,200 B/s, `U_bus` 0.208 | link still fine, DMA bandwidth doubles |
| sample rate `f_s` | 1600 | 3200 | window span 160 ms, `T_w` 10 ms, rate 19,200 B/s | latency halves, CPU and link double |
| active current | 12 mA | 18 mA | `E_blk` 4.1184e-4 J, lifetime 39.5 h | lifetime scales inversely with current |
| idle current | 1.2 mA | 0.6 mA | `E_blk` 2.6532e-4 J, lifetime 61.4 h | idle term matters less than active here |
| quantizer step | 0.05 | 0.02 | `sd(e)` 0.00577 units, range narrows from 12.75 to 5.10 units | finer resolution, earlier saturation |
| decision rate | 5/s | 50/s | `U_bus` 0.025, 2300 B/s | wire is not the constraint |

The idle-current row corrects an intuition rather than confirming one. Halving `I_idle` from 1.2 to 0.6 mA moves life from 55.6 to 61.4 hours, about 10 percent, because the idle term is `1.68e-5` of the `8.88e-5` J per block — 18.9 percent of the total. The active term, at `7.2e-5` J, is 81.1 percent, so the active current is the lever in this particular duty cycle. The general lesson survives: which term dominates is an arithmetic question that must be answered per design, and the answer here is the opposite of the usual low-power folklore.

The quantizer-step row carries a warning that the table cannot show: halving the step halves the representable range for a fixed code width, because the range is `(q_max-q_min)*s`. Finer resolution and wider range trade against each other at constant bits, and a design that picks the step from resolution alone will discover clipping at the first unusual window.

### 5.9 Window ledger reconciliation

Take a one-minute capture and reconcile counts rather than reading a rate counter.

```text
expected blocks = 60 s / 0.020 s = 3000
observed blocks = 2983
deficit         = 17 block = 17*20 ms = 340 ms
relative deficit= 340/60000 = 0.005667 = 5667 ppm = 0.567 percent
```

An internal RC oscillator with a ±1 percent tolerance explains a 5667 ppm deficit on its own, so no loss mechanism needs to be invented. The discriminating experiment is duration: over one hour the same defect predicts `17*60 = 1020` missing blocks, or 20.4 s of missing samples, and the deficit should scale linearly. A deficit that appears in lumps of a few blocks at irregular intervals is a different mechanism — a missed wake event or an aborted transfer — and the two are separable with a log of timestamps rather than a count.

### 5.10 Static memory ledger

| Region | Size (byte) | Growth driver | Detection before field |
|---|---|---|---|
| model image | 1,036 | weight precision, layer width | link error, artifact hash |
| runtime arena | 2,048 | operator schedule, graph rewrite | interpreter report at init |
| DMA stores | 576 | `K`, `N`, `M`, `b` | link error |
| feature window | 3,072 | window length, dtype | link error |
| log ring | 1,536 | diagnostic depth | link error |
| stack | 4,096 | call depth, interrupt nesting | painted high-water mark |
| total | 12,364 | all of the above | 12.6 percent of 98,304 byte |

The headroom is 85,940 bytes, or 87.4 percent of SRAM, which sounds generous until the growth drivers are named. The arena and the stack are the two regions whose size is not fixed by the design at freeze, and they are the two that produce a field failure rather than a link error. A project that ends design freeze at 96 percent utilisation has spent its entire margin on features that were not yet written.

Two unit conventions make this table easy to misread. The total of 12,364 bytes is 12.3 kB in vendor decimal units and 12.07 KiB in linker units, and the SRAM figure of 96 KiB is 98,304 bytes rather than 96,000. Percentages computed with mixed conventions differ by 2.4 percent relative, which is enough to hide a 2 KiB overrun.

### 5.11 Admission budget for the decision path

```text
publish 50 block/s, sustainable decision path 42 block/s
ring depth 8 block, one block in service -> B_tok = 8
absorbed burst = 8*0.020 = 0.160 s
t_saturate at rho = 1.190 -> 1.0 s
```

If the decision rate is decimated from every block to every second block, the consumer's load falls to `25` block-equivalents per second against a capacity of `42`, so `rho` becomes `25/42 = 0.595` and the queue is stable with the same memory. This is the cheapest fix available in the design: no new hardware, no model change, one counter in the scheduler, and the cost is a decision cadence of 25 Hz instead of 50 Hz.

### 5.12 Framing overhead ledger

| Framing | Bytes per record | Overhead | Worst-case escaping | Utilisation at 5/s |
|---|---|---|---|---|
| raw payload only | 36 | 0 percent | none | 0.20 percent |
| length + CRC + header | 46 | 27.8 percent | none | 0.25 percent |
| escaped binary, random data | 46 plus 0.78 percent | 28.8 percent | 100 percent on delimiter-heavy data | 0.25 percent |
| base64 text | 62 | 72.2 percent | none | 0.34 percent |
| JSON text with keys | 118 | 227.8 percent | none | 0.64 percent |

The table makes the usual trade explicit: readability costs payload. A 118-byte JSON record at five per second is 590 bytes per second, which is nothing on a 92160 byte/s link and would be 5.1 percent of a 115200-baud link. The decision is therefore about the host's parser and the field engineer's ability to read a log, not about bandwidth, until the decision rate rises into the hundreds per second.

## 6. Visual / Geometric Intuition

### 6.1 The rotation as a timeline

```text
store A | fill 0-20  | fill 60-80 | ...        reader may hold A from 20 to 60
store B | fill 20-40 | fill 80-100| ...        reader may hold B from 40 to 80
store C | fill 40-60 | fill 100-120           reader may hold C from 60 to 100
         20   40   60   80   100  120 ms
```

Each store is exclusive to the reader for exactly `(K-1)` fills after its own completion. The geometric reading is that the reader owns a window whose length is the sum of the next `K-1` fill durations, and the window slides forward in steps equal to one fill. Shrinking `K` to 2 halves the window and removes all tolerance for a late consumer; growing it to 4 buys 60 ms at the cost of 384 more bytes and one more interrupt of latency in the pointer handoff.

### 6.2 Ring occupancy over time

```text
occupancy
 8 |            *                        ring full, drop begins
 7 |         *  *  *
 6 |      *        *
 5 |   *              *
 4 |*                    *        *      steady drain at 42 block/s
 3 |                        *   *
 2 |                          *
   +------------------------------------> t (s)
      0   1   2   3   4   5   6   7
```

The sawtooth is the whole story of a producer faster than its consumer. Between 0 and 1 s the ring fills at 8 blocks per second net; at 1 s it is full and every further publish either overwrites the oldest block or is rejected, depending on the policy. Overwrite makes the consumer see a gap; rejection makes it see stale data. Both are correct behaviours if the policy is declared, and both are defects if it is not.

### 6.3 Quantization grid against a signal

```text
real value   -6.25 ....... 0 ....... 6.5
codes        -128   ...   -3   ...   127
step 0.05    |  |  |  |  |  |  |  |  |
signal that swings +-8 units      ^ clipped at both ends
signal that swings +-0.05 units   ^ one or two codes only
```

The same grid fails two different signals for opposite reasons. A large-amplitude signal loses its peaks to clamping; a small-amplitude signal loses its entire shape to the step size. Choosing `s` from the observed range without checking the range of the signal at inference time is the single most common quantization error.

### 6.4 Cache lines versus block boundaries

```text
cache line 64 B:  |0..63|64..127|128..191|
block 192 B:      |----------- block ----------|  (3 whole lines)
block 100 B:      |------ block ------|          (1 whole + 1 partial)
```

A 192-byte block aligns with three cache lines only when the buffer base is 64-byte aligned; otherwise it straddles four lines and each handoff requires a maintenance operation on a partially shared line. The geometric rule is that block size and alignment should be chosen together on a cached core, and neither should be chosen from the arithmetic alone.

### 6.5 Deadline versus latency distribution

```text
count
  |        latency histogram
  |      __
  |     /  \_
  |    /     \__
  |___/         \______ tail
  +--------|------------> ms
         mean   deadline
```

The mean sits to the left of the deadline and the tail to the right of it. Every plot of this shape refutes a mean-latency acceptance test, and the discriminating question is whether the tail comes from queueing (input-dependent arrival), blocking (a lock held by someone else) or interference (a higher-priority handler that grows with input). These three produce different tail shapes and different fixes.

### 6.6 Jitter erodes the reuse window from the inside

```text
nominal:   |--17--|--20--|--20--|--23--|     fills of 17..23 ms, mean 20
exclusivity after completion at 0:
   nominal assumption 2*20 = 40 ms ......... reader released by 40
   worst case        17+20 = 37 ms ......... still safe for W_c = 6 ms
   worst case        17+17 = 34 ms ......... the bound quoted in 4.3
```

The lesson is geometric rather than arithmetic: the reader's window is bounded by the *shortest* pair of following fills, so a plot of fill durations has its minimum at the relevant statistic, not its mean or its maximum. Jitter that is symmetrical in time is asymmetrical in consequence, because early arrivals shorten the window while late ones merely delay the consumer's next claim.

### 6.7 Where a 320 ms window sits inside a 20 ms cadence

```text
samples 0 ......................... 512          window W0, ends t=0.320 s
samples 32 ........................ 544          window W1, ends t=0.340 s
overlap between consecutive windows = 480/512 = 93.75 percent
distinct windows per second = 50
window end times are multiples of 20 ms only if the advance is exact
```

A 93.75 percent overlap means consecutive feature vectors are strongly correlated, so a five-point accuracy difference measured on consecutive windows is not five independent observations. Evaluation that treats overlapping windows as independent overstates confidence, and the arithmetic for the effective sample count is `n_independent ≈ sample_span/window_span` rather than the number of windows.

### 6.8 The freshness gate as a sliding band

```text
t_sample_end                     t_use
     |-------- A --------|
     |<--- accept --->|                 A <= A_max
     |                |---- reject ----| A >  A_max
window end moves 20 ms per block, so the accept band advances in steps
```

The gate is not a property of the model but a band that slides with the sample clock. Because the band advances in 20 ms steps while the consumer's service time varies continuously, a decision that passes on one block can fail on the next with identical compute time. That step behaviour, not model variance, is why rejection counts cluster at particular input phases.

### 6.9 Backlog growth against sustainable rate

```text
backlog
 8 |                        /  lambda 50, mu 42
 6 |                    /
 4 |                /
 2 |            /
 0 +--------|---------|--------------> t (s)
            0.5       1.0   (ring full, shedding begins)
```

A straight line is the signature of a stationary deficit. Its slope is the loss rate, and its intercept with the buffer depth is the time at which shedding begins. A backlog curve that curves upward indicates a consumer that slows as it queues, which is the one case in which adding buffer makes the failure later but worse.

## 7. Comparison / Selection Table

### 7.1 Where to run the inference

| Option | Strength | Binding constraint | Fails when | Choose when |
|---|---|---|---|---|
| Cortex-M4F, on-device | deterministic, no link in the loop, microjoule energy | flash, SRAM, no cache | model exceeds arena, or features need float dynamic range | decisions must be made with milliseconds of latency and no network |
| Cortex-A class, Linux | large models, easier tooling, filesystem | power, boot time, nondeterministic scheduling | a deadline must be met every 20 ms | the decision is supervisory and tolerates tens of milliseconds |
| DSP core | deep pipelining, MAC throughput | separate toolchain, tight scratch | the workload is control-dominated rather than filter-dominated | the dominant cost is a long FIR/FFT chain |
| FPGA / fabric | true parallelism, hard timing | development cost, fixed function | the algorithm changes weekly | the same kernel runs for years at high rate |
| Host / edge server | unlimited memory, fleet updates | network, privacy, availability | connectivity is intermittent or data cannot leave | decisions are advisory and batch-tolerant |

### 7.2 Buffer strategy

| Strategy | Guarantees | Byte cost at `N=32,M=3` | Hazard |
|---|---|---|---|
| single buffer with copy | reader gets a private copy | 384 plus copy time | copy itself may be preempted past the refill |
| double buffer | alternation, no overlap if released | 384 | no tolerance for a late consumer |
| triple buffer | two-period tolerance | 576 | still no freshness guarantee |
| ring of blocks | FIFO with drop policy | `N_r*192` | occupancy must be bounded and counted |

### 7.3 Timing measurement

| Method | What it observes | Distortion | Valid claim |
|---|---|---|---|
| GPIO toggle plus scope | interval at the pin | small, if toggles are cheap | hardware-visible interval, with jitter |
| cycle counter read | CPU cycles inside one task | small, but misses preemption | execution time only |
| UART trace | event order and payload | large, transport competes for the link | ordering, not timing |
| host-side timestamp | arrival at host | includes transport and buffering | user-visible latency, not device latency |

Selection rule: choose the observation method whose distortion is smaller than the effect you claim to measure. If the effect is a 300 microsecond regression and the trace costs 400 microseconds per event, the observation has inverted the conclusion.

### 7.4 Quantization scheme selection

| Scheme | Metadata cost | Strength | Fails when |
|---|---|---|---|
| per-tensor symmetric | 2 byte per tensor | simplest kernels, predictable range | one outlier channel widens the range for all |
| per-axis on a 3-axis sensor | 2 byte per axis | preserves range per physical axis | the kernel does not support it, forcing a fallback |
| per-channel on a layer | 2 byte per output channel | best accuracy per bit on convolutions | the runtime ignores the array and uses element zero |
| float fallback on the first layer | 4 byte per value | absorbs wide input dynamic range | flash and cycle budget cannot afford it |

### 7.5 Diagnostic medium selection

| Medium | Capacity | Timing impact | Endurance | Use for |
|---|---|---|---|---|
| UART trace | link-limited, 9.6 kB/s at 921600 | high if per sample | none | event ordering during bring-up |
| RAM ring, dumped on fault | tens of kB | low, one copy per block | none | the window preceding a failure |
| flash circular log | hundreds of kB | medium, page erase stalls | limited erase cycles | post-mortem after a field return |
| counters only | a few dozen byte | negligible | none | long-run rate and loss accounting |
| no instrumentation | zero | zero | none | production, with counters retained |

### 7.6 Splitting acquisition from inference across two parts

| Split | Benefit | Added failure mode | Test that must exist before release |
|---|---|---|---|
| MCU acquires, MCU infers | one clock domain, one artifact | none new; flash and SRAM bound | golden vectors on the MCU |
| MCU acquires, MPU infers | large model, filesystem, logging | link latency and remote pause become part of the deadline | remote-pause trace with local fallback |
| MCU acquires, host infers | unlimited compute, fleet retraining | availability and privacy of the raw stream | offline-degradation mode with declared behaviour |
| MCU acquires and gates, host refines | local safety plus remote quality | two label maps and two thresholds to keep in sync | reconciliation test on a shared labelled set |

The row that fails most often in practice is the last one, because two decision paths mean two label maps, and a disagreement between them is usually diagnosed as a model problem when it is a taxonomy problem.

### 7.7 Frame integrity selection

| Integrity | Detects | Misses | Byte cost |
|---|---|---|---|
| none, fixed length | nothing | transposition, truncation | 0 |
| length field only | truncation and resynchronisation | any bit flip in payload | 1 to 2 |
| 16-bit CRC | all single-bit and most burst errors | errors that collide by construction | 2 |
| 32-bit CRC | the same, with a much smaller collision set | nothing practical | 4 |
| CRC plus sequence | the above plus reordering and loss | duplication only if sequence repeats | 2 to 6 |

For 46-byte records at five per second, the 16-bit CRC costs 2 bytes and 0.25 percent of the link, and there is no defensible reason to omit it: the alternative is silent corruption that presents as a model error with no exception raised anywhere.

## 8. Failure Modes & Edge Cases

### 8.1 DMA completion treated as input validity

Symptom: the consumer occasionally reads a block whose first few samples belong to the previous fill; class outputs flicker on an otherwise stable input.

Competing causes: (a) the consumer claimed the store still being filled, (b) the completion interrupt fired before the final bytes became visible on a cached core, (c) the pointer was published before the sequence counter incremented.

Discriminating check: record the sequence at publication and again at claim, and separately verify memory visibility by comparing a checksum computed in the consumer against one computed in the handler from the same bytes. Equal checksums with a wrong-pointer symptom isolates (a); differing checksums isolates (b).

Correction: publish a small descriptor (pointer, length, sequence, epoch) through a single-word atomic handoff after the visibility operation, and never expose the raw DMA pointer to the consumer.

### 8.2 Priority inversion under a mutex

Symptom: the inference task's worst-case response jumps from 2.5 ms to 40 ms only when a low-priority logging task is active.

Competing causes: (a) classic mutex inversion with a medium-priority task preempting the holder, (b) a critical section that disables interrupts long enough to delay the block ISR, (c) the logging task itself flushing flash and blocking on the bus.

Discriminating check: measure the low-priority holder's time on CPU versus its time blocked, and measure interrupt-off duration on a pin. If interrupt-off is short but response is long, the mechanism is the mutex, not interrupt masking.

Correction: bound the critical section, enable priority inheritance on the shared mutex, and move flash writes to a separate lower-bandwidth path with a bounded queue.

### 8.3 UART frame loss that looks like model error

Symptom: the host observes a plausible but wrong class on roughly one decision in fifty, with no transport counter incremented.

Competing causes: (a) a frame was dropped by the host parser due to a partial read, (b) the device reused a sequence number after wrap, (c) the model genuinely misclassified on those windows.

Discriminating check: log both the device-side decision and the host-side parse with the same sequence and epoch; a device record with no host counterpart proves transport, while a matching record proves the model. Comparing only host-side labels cannot separate them.

Correction: frame with a length field plus CRC, parse with an explicit state machine, and reconcile device and host ledgers on sequence gaps rather than on rate counters.

### 8.4 Stack overflow discovered as a wrong class

Symptom: intermittent corruption of a static feature buffer, appearing only when a deeper code path runs under a rare combination of inputs.

Competing causes: (a) stack overflow into the adjacent static region, (b) a DMA overwrite of the region, (c) a pointer arithmetic error in feature extraction.

Discriminating check: paint the stack with a pattern and read the high-water mark after the failure; also compare the corrupted region's address against the stack bounds from the map file. A high-water mark within 64 bytes of the limit plus corruption above the stack top is conclusive for (a).

Correction: reduce the stack frame, move large buffers out of automatic storage, and add a link-time or run-time guard region whose corruption is detected rather than tolerated.

### 8.5 Stale decision accepted downstream

Symptom: the actuator responds correctly but late, and the late response is worse than no response because it acts on a state that has already changed.

Competing causes: (a) the freshness gate compares against the wrong timestamp, (b) the gate exists but the consumer bypasses it on the fast path, (c) the age budget was set for the model rather than for the action.

Discriminating check: inject a delay before the consumer and confirm the rejection count increments. If the gate never rejects under an injected delay, the gate is not on the path.

Correction: put the freshness decision at the action boundary, make rejection a counted state with a defined fallback, and derive `A_max` from the physics of the action rather than from the model's input window.

### 8.6 Tensor shape mismatch that survives to production

Symptom: accuracy drops by a few points with no error and no exception; the same model scores well in the float harness.

Competing causes: (a) the input tensor layout is transposed relative to training, (b) the dtype differs (int8 versus uint8) with the same byte width, (c) the class index order differs between the label file and the softmax output.

Discriminating check: run a single golden input through both paths and compare intermediate tensors, not labels. A permutation that preserves values but reorders positions shows up immediately at the first layer; a label-order error shows up only at the output, after all layers agree.

Correction: fix the layout in one place, assert shapes and dtypes at load, and derive the label map from the same artifact as the model.

### 8.7 Filter state reset at every block

Symptom: a periodic transient at each block boundary, visible as a spectral line at the block rate and as a small accuracy loss.

Competing causes: (a) FIR/IIR state reset per block, (b) a windowing function applied per block where a continuous filter was intended, (c) an overlap that is off by one sample.

Discriminating check: feed a pure sinusoid at a frequency far from the block rate and inspect the spectrum; a line at `1/T_w` and its harmonics indicates boundary discontinuity, and its amplitude relative to the tone separates reset from off-by-one overlap.

Correction: retain filter state per stream across blocks, reset only on a declared discontinuity, and test continuity by concatenating block outputs and comparing to a single-shot run.

### 8.8 Wake gap between low-power intervals

Symptom: sample count per minute is short by a small percentage, and the shortfall is invisible in any single block.

Competing causes: (a) the part is not sampling during wake-up and settling, (b) the sensor needs a settling interval that is shorter than the configured delay, (c) the DMA is re-armed after the first conversion rather than before.

Discriminating check: count published blocks over a long window and compare with `elapsed/T_w`; then measure the interval from wake to first conversion with a pin. A block-count deficit that scales with the number of wake cycles isolates the re-arm or settling mechanism.

Correction: start the sensor and settle before arming conversions, account for the wake interval in the schedule, and treat the sample-count ledger as a first-class counter.

### 8.9 ADC trigger drift and channel-order mismatch

Symptom: axes appear swapped in some builds, or the phase between channels shifts by one sample, changing derived features.

Competing causes: (a) the scan sequence in the ADC configuration does not match the buffer interpretation order, (b) the DMA packs channels in the order they complete rather than the order configured, (c) the feature code indexes with `[sample][channel]` while the producer wrote `[channel][sample]`.

Discriminating check: drive one channel with a distinctive waveform (a square wave at a low rate) and leave the others near a constant; the channel that shows the square wave identifies the mapping unambiguously. Then read one sample triple and print all three codes before any averaging.

Correction: define the layout in one header shared by producer and consumer, assert it with a static layout test at boot, and never let the feature code compute an index it did not define.

### 8.10 Cache and memory-visibility mismatch on a cached core

Symptom: works on the uncached microcontroller, fails on the Cortex-A class part with the same source; failures cluster after a long idle period.

Competing causes: (a) the DMA wrote to memory but dirty cache lines were not invalidated before reading, (b) the consumer read a stale line because the producer's clean missed a partially shared line, (c) a compiler reordered the flag store ahead of the data store without a barrier.

Discriminating check: disable the cache (or mark the region non-cacheable) and re-run. If the symptom disappears, the mechanism is coherence; if it persists, the mechanism is ordering and the fix is a barrier rather than maintenance.

Correction: declare DMA regions non-cacheable, or perform clean/invalidate by exact address range with sizes rounded to the line, and place an explicit barrier between data publication and pointer publication.

### 8.11 Cortex-M versus Cortex-A placement error

Symptom: latency numbers quoted from the Linux side are an order of magnitude better than the device behaves, or the reverse.

Competing causes: (a) the measurement excluded the wake or scheduling delay that exists only on the larger part, (b) the small part pays flash-read cost per weight on every inference, (c) the larger part's measurement included a warm cache that is never warm in operation.

Discriminating check: measure the same model, same input, same quantizer on both, with cold and warm state declared. If the large part is faster warm and slower cold, the comparison was a cache-state comparison rather than an architecture comparison.

Correction: state the placement decision with its worst case, not its best; treat any cross-architecture latency claim as invalid without matched input and declared cache state.

### 8.12 Toolchain and linker-layout regression

Symptom: a firmware rebuild with no source change moves accuracy or breaks timing; the diff shows only a compiler version.

Competing causes: (a) different optimisation changed floating-point contraction, (b) a section grew and pushed a buffer across an alignment boundary, (c) an inlined kernel changed the stack high-water mark into a previously safe margin.

Discriminating check: compare the map file section sizes and addresses between builds, and run the golden-vector harness on both binaries. Identical vectors with different timing isolates layout; different vectors isolates floating-point contraction.

Correction: pin the toolchain, publish the map delta with each release, and keep the golden-vector harness as a build gate rather than a manual step.

### 8.13 Debug instrumentation changes the schedule

Symptom: the debug build meets an 80 ms limit and the release build misses it, or the traced build never reproduces the race.

Competing causes: (a) semihosting or logging perturbs timing such that the race window closes, (b) optimisation changed memory ordering and exposed a missing barrier, (c) the release build runs a different artifact or configuration than assumed.

Discriminating check: compare artifact hashes, compiler flags and configuration between the two builds, then use a minimally intrusive trace (a single pin toggle or a small in-RAM buffer dumped after the fact). If the race reappears only when instrumentation is reduced, the observer was part of the mechanism.

Correction: keep one build configuration for the acceptance evidence, record exactly what instrumentation it contains, and never generalise a traced result to an untraced build.

### 8.14 Power-state continuity error

Symptom: after a low-power transition, the first few decisions after wake are wrong and then recover.

Competing causes: (a) the model arena or filter state was in a power domain that lost retention, (b) the sensor needed a settling interval that the wake path skipped, (c) the clock changed and a timing-dependent kernel produced different intermediate values.

Discriminating check: mark the retained regions and verify them byte-for-byte after wake; a zeroed or pattern-filled arena proves retention loss before any model question is asked. If retention is intact, compare the first three sample triples against the expected settling curve.

Correction: either retain and verify state at wake, or re-initialise it deterministically and discard the first window after wake. Both are defensible; an implicit partial reset is not.

### 8.15 CMSIS-DSP state or coefficient contract error

Symptom: the filter output is plausible but has the wrong gain or a phase shift relative to the reference implementation.

Competing causes: (a) coefficients generated for a different sample rate, (b) fixed-point coefficient scaling mismatched to the block floating-point shift, (c) the state array sized for a shorter filter than the one configured.

Discriminating check: run an impulse through the block and compare the first `n_taps` outputs against the design's impulse response. A gain error scales the whole response; a state-size error truncates it after a fixed number of samples.

Correction: generate coefficients and state sizes from one parameter set, assert the derived sizes at compile time, and keep an impulse-response test in the harness.

### 8.16 CMSIS-NN integration mismatch

Symptom: the integer output is systematically offset by a small amount that grows with input magnitude.

Competing causes: (a) the requantisation multiplier or shift was taken from a different tensor's parameters, (b) the accumulation width is narrower than the kernel expects for the declared dimensions, (c) the activation range clamp differs between the reference and the optimized kernel.

Discriminating check: place a golden input whose activations stay well inside the clamp range; if the offset persists, the mechanism is the multiplier or the accumulator width. Then raise the input until the clamp binds and observe whether the divergence changes character.

Correction: derive multipliers from the same manifest the model was exported with, assert accumulator width against the largest possible dot product, and include boundary vectors that exercise the clamp.

### 8.17 Decimation factor misconfigured after a sensor change

Symptom: features look smoothed, the class distribution shifts toward the dominant class, and every timing number in the firmware still agrees with its own arithmetic.

Competing causes: (a) the sensor now runs at half the previous `ODR` so the effective `f_s` halved and the window span doubled, (b) the decimation filter length changed and altered group delay, (c) the accelerometer range setting changed, lowering resolution without changing timing.

Discriminating check: measure the actual completion period with a pin and compare with `N/f_s`; a period twice the expected value proves (a) before any spectral question is asked. Then inject a known-frequency tone at the front end and confirm where it appears after feature extraction.

Correction: derive `f_s` from the sensor configuration rather than from a constant, assert the measured block period at start-up against the configured value, and refuse to run inference if they disagree by more than one sample period.

### 8.18 Firmware update ships a new model but not its constants

Symptom: accuracy falls after a remote update, and the golden vectors from the previous release still pass because the harness pins the old artifact.

Competing causes: (a) the preprocessing normalisation constants were compiled into code and did not change with the weights, (b) the label map is stored separately and still reflects the old class order, (c) the runtime arena grew and the allocator silently truncated a tensor.

Discriminating check: hash the weights, the constants file and the label map at boot and publish all three; a mismatch against the release manifest localises the defect to one artifact. If all three match, compare intermediate activations against the released golden vectors.

Correction: bind weights, constants and labels into one signed artifact with one hash, and make the boot path refuse to run if the artifact's declared hash is not present in the manifest it was shipped with.

### 8.19 Watchdog reset loop hides a slow path

Symptom: the device appears alive because the boot counter increments, but decisions are delivered in bursts and the host sees sequence gaps that always start near the same value.

Competing causes: (a) a rare input path exceeds the watchdog window and resets the part, (b) a blocking write to a full transport queue stalls the task that feeds the watchdog, (c) the watchdog is refreshed by a timer interrupt rather than by the task that must make progress.

Discriminating check: record the reset cause register and the last sequence number before reset; a repeating block of consecutive sequence values at the gap proves the reset happens at a consistent point in the input, which rules out the transport stall.

Correction: refresh the watchdog only from the task that must make progress, declare a maximum service time shorter than the window, and make reset cause and last sequence part of the persistent record.

### 8.20 Transport backpressure blocks the consumer

Symptom: overrun counts rise exactly when decision traffic increases, and the inference task's own execution time looks unchanged.

Competing causes: (a) the UART write is blocking and waits for the shift register when the host is slow, (b) the transmit ring overflow policy discards silently and the discard is counted as a model fallback, (c) the host parser sends flow control that the device ignores.

Discriminating check: measure time spent inside the write call separately from compute; if the write dominates during the overrun episodes, the mechanism is backpressure rather than compute. Then slow the host deliberately and confirm the overrun rate scales.

Correction: make the decision path non-blocking with a bounded ring and a declared overflow policy, and count drops at the ring rather than at the host.

### 8.21 Device and host time bases drift apart

Symptom: latency attribution changes depending on which log is read, and deduplication removes records that are genuinely distinct.

Competing causes: (a) the device timestamp counts sample periods while the host timestamp counts milliseconds, (b) both count the same units but from different origins, so an offset masquerades as drift, (c) a genuine clock tolerance difference, on the order of hundreds of ppm, accumulates over hours.

Discriminating check: plot the difference between device and host timestamps over an hour. A constant offset with no slope is (b); a linear slope in parts per million is (c); a slope that changes sign at resets is (a) combined with origin resets.

Correction: publish timestamps in one declared unit with an explicit origin, re-synchronise the offset at each session, and never subtract timestamps from different origins without stating the offset used.

### 8.22 Interrupt storm from an unconfigured input

Symptom: jitter spikes that appear at random times, a raised idle current, and inference deadlines missed only when a cable is attached.

Competing causes: (a) a floating input pin with an enabled interrupt oscillates and generates a burst of edges, (b) an edge-triggered interrupt is cleared late so the handler re-enters, (c) a shared interrupt line is serviced by two handlers that both clear the wrong flag.

Discriminating check: count interrupts per second for each source and compare with the expected rate; a source whose count is orders of magnitude above expectation with no corresponding data identifies the pin or the latch. Then disable that source and confirm the jitter disappears.

Correction: configure unused pins with a defined level and no interrupt, clear flags at the correct point, and add a per-source interrupt count to the diagnostic record so a storm is visible rather than inferred.

### 8.23 Saturation counters that count frames rather than samples

Symptom: the preprocessing saturation counter reads zero while the model's accuracy on high-amplitude windows collapses, and every boundary test passes.

Competing causes: (a) the counter increments once per block if any sample saturated, so a block with 500 of 512 samples clamped reports one event, (b) the counter is incremented after clamping but reset at each window, losing the history that would show a trend, (c) the check tests a threshold on the raw code rather than the post-calibration value, so a calibration gain error is invisible.

Discriminating check: log the maximum absolute raw code per block alongside the counter; a block whose maximum is at the code limit with a counter value of one and a 512-sample window is conclusive for (a). Then compare against the calibration applied in the host to confirm (c).

Correction: count samples, not blocks, publish the per-block maximum, and compare both against the calibration range so a gain error surfaces as a distribution shift rather than as a spike.

### 8.24 Block size changed without re-deriving the window

Symptom: the feature vector is the right shape and the model runs without error, but the physical span it represents has doubled and the model's behaviour changes on every dynamic event.

Competing causes: (a) the window length is declared in blocks in some modules and in samples in others, (b) the feature extraction indexes a ring of fixed byte capacity, so a larger block silently wraps earlier, (c) the window advance was not scaled with the block size, changing the overlap ratio.

Discriminating check: compute the window span in seconds from the produced configuration and compare with the documented value; then feed a step input and measure the time from the step to the first feature change, which is the empirical span.

Correction: declare window length in seconds and derive both the sample count and the block count from `f_s` and `N`, assert the derived capacity at compile time, and test the step response as part of the harness.

### 8.25 Timestamp tick source changed by a configuration merge

Symptom: ages are computed correctly on one branch and consistently small on another, and the freshness gate stops rejecting on the branch that ships.

Competing causes: (a) the timestamp is taken from a 1 kHz tick on one branch and a millisecond counter with a different origin on the other, (b) the counter is read in the handler before the last sample lands, (c) the comparison subtracts unsigned values across a wrap.

Discriminating check: print both raw counter values for a single event and the difference; a value that is ten times the expected interval proves a tick-unit mismatch before any gate logic is examined.

Correction: define one timestamp type with its unit and origin in a single header, convert at the boundary, and include a unit test that asserts `t_use - t_sample_end` equals a known injected delay.

### 8.26 Evaluation leakage through overlapping windows

Symptom: offline accuracy is high, field accuracy is lower by a wide margin, and the confusion matrix shows errors concentrated on transitions between activities or states.

Competing causes: (a) train and test splits were made on windows rather than on recording sessions, so adjacent overlapping windows sit on both sides of the split, (b) the normalisation statistics were computed over the full capture before splitting, (c) the label for a window was assigned from the state at its end rather than its majority state, so transitions are mislabelled.

Discriminating check: re-split by session or by subject and re-evaluate without retraining; a large drop that appears only under a session-grouped split is conclusive for (a). Then recompute the statistics per split to test (b).

Correction: split by session, fit all preprocessing inside the training fold, and label transitions by majority or exclude them explicitly with a declared rule.

## 9. Complexity & Cost Analysis

### 9.1 Work and storage sizing from declared inputs

```text
sample work       = f_s*M samples/s
feature work      = n_features * window_advance_rate
filter work       = f_s*M*n_taps MAC/s
inference work    = MACs * decision_rate
transport work    = frame_bytes * decision_rate
```

With `f_s*M = 4800` samples/s and a 32-tap filter, the filter costs `4800*32 = 153,600` MAC/s. At 64 MHz that is 2.4 ms of cycles per second, or 0.24 percent of the core. The inference costs `728*5 = 3640` MAC/s, which is 0.057 ms per second. The DSP chain is therefore two orders of magnitude more expensive than the model, and a project that spends its optimisation budget on the model has optimised the wrong term.

Storage sizing follows the same discipline: the largest single allocation is the feature window (`512*3*2 = 3072` bytes), followed by the DMA rotation (576 bytes) and the model arena (2048 bytes). None of these is large, which is exactly why a 96 KiB part is comfortable and a 32 KiB part is not: the margin lives in the aggregate, not in one item.

### 9.2 Cost of a bounded diagnostic path

A ring of 8 blocks at 192 bytes is 1536 bytes and holds 160 ms of history. That is enough to answer "what did the window look like just before the failure" for a decision whose window is 320 ms only if the ring covers the window, which it does not. Covering a full window costs 512 samples per channel, so 3072 bytes for the raw history plus the decision record.

The realistic diagnostic budget is one extra copy of the current window (3072 bytes) plus a 200-byte decision record, and the honest statement of its cost is 3.3 KiB of SRAM and no timing impact if the copy is done in the consumer. Any design that logs per sample over UART at 1600 Hz needs 9600 bytes per second of link, which is the same as the raw acquisition rate and therefore doubles the streaming cost.

### 9.3 Architecture selection under resource evidence

```text
flash needed   = model image + code + constants + slack
SRAM needed    = windows + arena + stacks + DMA + ring + slack
slack rule     = at least 20 percent of each, and at least one full feature window spare
```

The 20 percent rule is a heuristic, not a theorem, and it exists because the two things that grow after the design is frozen are the arena (graph rewrites) and the stack (deeper call chains). A part whose map shows 96 percent SRAM utilisation at design freeze has no room for either, and its first field bug will be an allocation failure rather than a wrong answer.

### 9.4 Engineering cost of the contract

The pieces that cost schedule are not the model: they are the golden-vector harness, the timing instrumentation, the frame parser, and the failure-injection path. A reasonable split for a first deployment is one unit of effort on the model, one on acquisition and framing, one on the harness, and one on the power and wake path. Teams that spend all four units on the model ship something that classifies well in a notebook and loses blocks in the field.

### 9.5 Cost of getting the timestamp wrong

If the timestamp marks interrupt entry rather than the end of the last sample, every age is understated by the interrupt latency plus the trailing transfer. With a 200 microsecond latency and a 30 ms budget the error is 0.67 percent, which is tolerable. With a 2 ms latency on a busy system it is 6.7 percent, and it becomes decisive whenever the budget is tight. The cost of fixing it is one counter read at the right place; the cost of not fixing it is a freshness gate that passes stale data.

### 9.6 Cost of a false-rejection policy

A freshness gate with `A_max=30 ms` on a decision path whose age is 12 ms mean with a 6 ms standard deviation rejects the far tail. Under a normal approximation, `z=(30-12)/6=3`, so the one-sided rejection probability is 0.00135, or 0.135 percent. At five decisions per second that is `5*0.00135 = 0.00675` rejections per second, or 24.3 per hour.

That rate is cheap to absorb, and the arithmetic matters because the alternative policies have very different costs. A gate at `A_max=24 ms` gives `z=2`, a rejection probability of 0.0228, and 410 rejections per hour, which is a visible operational burden. A gate with no bound at all converts every one of those tail events into a late action, whose cost is a safety question rather than a counter. Choosing the bound is therefore a trade between rejection frequency and staleness tolerance, and the trade can be computed in one line once the distribution is known.

### 9.7 Cost of provisioning memory for the worst case

The design's static demand is 12,364 bytes against 98,304, so the chosen part leaves 85,940 bytes spare, or 87.4 percent. The interesting comparison is with the smallest part that satisfies the stated slack rule: `12364/0.8 = 15,455` bytes, so a 16 KiB (16,384 byte) part would sit at `12364/16384 = 75.5` percent utilisation with 4,020 bytes spare — above the 3,072-byte one-window rule but below the 4,096-byte stack allowance that the arena and stack are expected to grow into.

The economics therefore turn on a risk judgement rather than on a hard constraint, and the arithmetic bounds the judgement. At an illustrative 0.35 currency units per unit for the larger part across 10,000 units, the delta is `0.35*10000 = 3,500` currency units. Against it stand an illustrative six engineer-weeks at 2,000 currency units per week, or `6*2000 = 12,000` currency units, to re-qualify a design that has run out of arena and stack. The two are equal at `12000/0.35 = 34,286` units, so the larger part is the cheaper response at any volume below that figure — which is the entire range in which this class of device is usually built.

### 9.8 Cost of evaluation on overlapping windows

A 512-sample window advancing 32 samples per block produces 50 windows per second with 93.75 percent overlap. A test that treats 3,000 windows from a 60 s capture as independent has an effective independent count closer to `60000/320 = 187`. The confidence interval on an accuracy estimate is therefore about `sqrt(3000/187) = 4.0` times wider than the naive computation suggests.

That factor is the difference between "accuracy improved by 1.2 points, significant" and "accuracy moved by 1.2 points, indistinguishable from sampling noise" on the same data. The cost of correcting it is one line in the evaluation script; the cost of not correcting it is a model selection decision made on noise.

### 9.9 Cost of a retry policy

A decision frame that is retried on missing acknowledgement doubles or triples its link cost and its latency:

```text
attempts    = 1 + r + r^2 + ... bounded by n_max
mean_cost   = frame_time * (1 - r^(n_max+1))/(1 - r)   [s, when r < 1]
```

With a per-attempt loss probability `r=0.02` and `n_max=3`, the mean attempt count is `1/(1-0.02) = 1.0204` truncated at three attempts, so the expected cost is close to one frame and the tail cost is three frames, or `3*0.499 = 1.497 ms`. The design consequence is that retries inflate the worst-case latency by a factor of three while barely moving the mean, which is the same asymmetry that makes mean-latency budgets fail.

### 9.10 Cost of two model copies for A/B comparison

Running a candidate model alongside a champion on the same device doubles the compute and can be affordable precisely because compute is 0.38 percent of block energy. The real cost is the second arena (2048 bytes) and the second artifact identity, which must be published separately so that a regression can be attributed. A comparison that runs both but reports only the champion's decisions produces no evidence, because the candidate's outputs were never recorded with their own artifact hash.

### 9.11 Cost of the golden-vector harness

The harness is a table of inputs and expected integer outputs, executed on the host and on the target. Its cost is one engineer-week for a first version and its value is that it converts every later question from "does it work" to "which layer differs". At 40 vectors of 36 bytes each it occupies 1,440 bytes of test data, which fits in any part, and it executes in milliseconds, so it can run at every boot in a debug build without affecting the acceptance evidence.

## 10. Key Theorems / Results Reference

### 10.1 Sampling and aliasing

The Nyquist–Shannon condition states that a signal band-limited to `f_max` is reconstructible from uniform samples taken faster than `2*f_max`. In engineering practice the operative statement is stronger: any energy above `f_s/2` that reaches the converter folds into the baseband and cannot be removed afterwards. With `ODR=3200` and decimation to `f_s=1600`, the band `[800, 1600)` Hz folds onto `[0, 800)` Hz, so the anti-alias filter must attenuate that band before decimation.

The bound is exact for ideal band-limited signals and misleading for real ones. A vibration signal with a resonance at 1.4 kHz sampled at 1600 Hz produces a fold at 200 Hz that looks like a genuine low-frequency component and will be learned by any model trained on the same pipeline.

### 10.2 Queueing: Little's law and stability

Little's law gives `L = lambda*W` for a stable system: mean occupancy equals arrival rate times mean time in system. For `lambda=50` blocks/s and `W=40 ms`, `L=2` blocks on average. The law is distribution-free and holds for any work-conserving system, which is why it is the right first check on a ring: if the measured mean occupancy exceeds `lambda*W`, the system is not in the steady state the model assumes.

Stability requires the offered load `rho = lambda/mu` to stay below one. At `lambda=50` and a consumer that sustains 42 blocks/s, `rho=1.19` and the queue grows without bound; the ring does not fail, it converts overflow into loss, and the loss rate is `50-42 = 8` blocks/s.

### 10.3 Fixed-priority schedulability

For `n` periodic tasks under rate-monotonic priority assignment, the utilisation bound is `n*(2^(1/n)-1)`, giving 0.8284 for `n=2`, 0.7798 for `n=3`, and approaching `ln 2 = 0.693` as `n` grows. Exceeding the bound does not prove infeasibility; it only withdraws the guarantee, and the exact response-time test must then be applied.

For the three tasks in this sheet (`U` about 0.089 including interrupt load), the sufficient test passes with a wide margin. That margin is not permission to add a fourth heavy task without redoing the test, and it is not evidence that any individual deadline is met — it is a statement about the set.

### 10.4 Quantization error bounds

A uniform quantizer with step `s` has maximum error `s/2` and error variance `s^2/12` when the input is much larger than the step and does not overload. For a full-scale sinusoid the signal-to-quantization-noise ratio is `6.02*b_q + 1.76` dB, which gives 49.9 dB at 8 bits and 98 dB at 24 bits.

The theorem's precondition is the part that fails in practice: it assumes no overload and a signal occupying the full range. A signal at one tenth of full scale loses 20 dB, and a signal that clamps loses the theorem entirely. Empirically, the useful statement is that the effective number of bits is set by the smallest signal you must resolve, not by the converter's nominal width.

### 10.5 Error propagation through a quantized network

The error in a quantized layer output is bounded by the accumulation of input error through the weight matrix plus the requantization error at each stage. For a layer with `n` inputs, worst-case output error grows with the sum of absolute weights times the per-input error, and requantization at each layer adds up to half a step.

The practical consequence is that error grows with depth and with the magnitude of the weights, and that a network whose final layer has large weights can amplify a small first-layer error into a label flip. This is why intermediate tensor comparison, not final-label comparison, is the diagnostic of record.

### 10.6 WCET bounds and their cost

A sound WCET bound requires either exhaustive measurement on the target with all cache and branch states exercised, or static analysis with a model of the memory hierarchy. Measurement-based bounds are optimistic whenever the worst path was not observed; static bounds are pessimistic by a factor that grows with the sophistication of the core.

The engineering rule is to state the method with the number: "4.1 ms by measurement over 10^6 blocks, worst observed, on part revision X at 64 MHz" is a defensible claim; "about 4 ms" is not, and neither is a static bound quoted without its assumptions.

### 10.7 Energy-delay trade-off

For a workload with a fixed amount of work, raising the clock reduces the active interval but raises dynamic power, which scales roughly with frequency times the square of voltage. The product of the two is not monotone, which is why dynamic voltage and frequency scaling can reduce energy for a latency-tolerant workload and increase it for a latency-critical one.

In the two-state model of this sheet, the actionable statement is narrower and quantitative: the idle term exceeds the active term only when `I_idle*(T_w - t_act) > I_act*t_act`, which at `t_act=6 ms` of a 20 ms block requires `I_idle > 5.14 mA` against an active current of 12 mA. At the illustrative 1.2 mA the active term wins by a factor of 4.3, so frequency scaling that shortens `t_act` reduces energy while scaling that lengthens it increases energy.

### 10.8 Coherence and the memory model

On a core with a data cache, DMA writes bypass the cache and the core may read a stale line. The necessary conditions for correctness are either a non-cacheable mapping for the shared region, or maintenance by exact range before the consumer reads and after the producer writes, plus a barrier ordering the maintenance with the flag update.

The subtle part is that a partially shared line breaks the "invalidate by range" approach: invalidating a line that also contains live data written by the core discards it. This is why block size and alignment interact with coherence and why the buffer should be line-aligned on a cached core.

### 10.9 Amdahl's argument for the embedded pipeline

Speeding up a fraction `p` of the pipeline by a factor `k` changes total time to `(1-p) + p/k` of the original. In this pipeline the inference is about `28.4 us` of a `2.03 ms` decision path, so `p = 0.014`. Making the model infinitely fast removes 1.4 percent of the path; the remaining 98.6 percent is sampling completion, queueing and transport.

The result is why model optimisation is the last lever rather than the first. The same argument applies to sampling: halving the block period halves the sampling contribution but doubles interrupt frequency, so the benefit is bounded by the fraction that sampling actually occupies.

### 10.10 Information rate and what decimation discards

The information rate of the stream is `f_s*M*b = 1600*3*16 = 76,800` bit/s, or 9600 byte/s, and it is invariant to blocking. Decimation from 3200 Hz to 1600 Hz halves that rate, which is only lossless if the band above 800 Hz carried nothing of interest.

The useful corollary is that a 320 ms window at 1600 Hz contains the same sample count as a 160 ms window at 3200 Hz, but not the same information: the faster stream resolves transients that the slower one averages away. Choosing a rate is therefore a statement about the fastest physical event that must survive, not about storage.

### 10.11 End-to-end error budget

The pipeline's own numeric error, before any model error, can be assembled from the terms derived above:

```text
ADC quantization per sample      sd = 0.0144 units  (s = 0.05)
requantization per layer         sd = 0.4/sqrt(12) = 0.1155 units
three layers combined            sd = sqrt(3)*0.1155 = 0.2000 units
total pipeline sd                sd = sqrt(0.0144^2 + 0.2000^2) = 0.2005 units
```

The total is dominated by the requantization term, at 0.2000 of 0.2005. A class margin of 0.3 units is therefore comparable with the pipeline's own noise, which means a decision near the boundary can flip for reasons that have nothing to do with the physical signal. The engineering response is either wider margins in the model's training objective or a narrower output step, and the arithmetic says the output step is where the leverage is.

### 10.12 Sufficient statistics and their overflow

Window mean and standard deviation can be updated incrementally from `sum` and `sum_sq`, which avoids a second pass over 512 samples:

```text
mean = sum/n
var  = sum_sq/n - mean^2
range(sum)    = 512 * 32767            = 16,776,704        < 2^31
range(sum_sq) = 512 * 32767^2          = 5.5e11            > 2^31
```

The second line is the trap: a 32-bit accumulator holds the sum comfortably and overflows on the sum of squares by a factor of 256. The failure is silent on most cores, produces a negative variance, and a square root of a negative number is either a NaN in floating point or a garbage value in integer code — in both cases a feature that looks like a sensor fault.

The numerically safer form computes the variance in the two-pass manner or shifts the samples by their mean first, which is what a 64-bit accumulator or a scaled integer path achieves. The cost is one extra pass over 512 samples, which at 64 MHz is microseconds.

### 10.13 Result boundaries used in this manuscript

| Claim type | What it can support | What it cannot support |
|---|---|---|
| hand-computed arithmetic | feasibility of a design at stated inputs | performance on any device |
| host harness oracle | contract behaviour, frame parsing, wrap logic | interrupt latency, DMA semantics |
| target timing measurement | that build's response distribution | another build or another part |
| field telemetry | observed behaviour of the deployed population | behaviour outside its operating range |

## 11. Interview / Exam Questions

### Q01. A block completes at 0 ms in a three-store DMA rotation with 20 ms fills. When may the engine begin overwriting it, and what independent condition makes its decision usable?

**Answer:** Overwrite may begin at `(3-1)*20 = 40 ms`, assuming the two following fills are uninterrupted. Usability is a separate verdict: the decision must satisfy `t_use - t_sample_end <= A_max`, which the rotation says nothing about.

**Reasoning:** The completed store is one of three; the engine fills the other two before returning, so the interval is two fills, not three. Counting the fill that produced the block would give 60 ms and overstate protection by a whole period.

**Follow-up:** With descriptors of 20, 25 and 30 ms, enumerate the next two rather than multiplying: `T_reuse` is 25+30 for the store that completes just before the long descriptor.

**Red flag:** "Three buffers gives 60 ms" or "it completed, so it is fresh."

**Scoring:** 3 points: one for `(K-1)`, one for the 40 ms with units, one for separating freshness from integrity.

### Q02. Compute the payload ceiling of a 921600-baud 8N1 link and the time to send a 46-byte decision frame.

**Answer:** `921600/10 = 92160` bytes/s; a 46-byte frame takes `46/92160 = 0.499 ms`.

**Reasoning:** An 8N1 frame is ten symbols, so bytes per second is one tenth of the symbol rate. The frame time follows by dividing bytes by bytes per second, and the result is independent of frame content.

**Follow-up:** At five frames per second the link carries 230 bytes/s, which is 0.25 percent utilisation; add escaping and the payload shrinks by the escape fraction.

**Red flag:** Dividing the baud rate by eight, which silently assumes no framing overhead.

**Scoring:** 2 points: correct divisor and a unit-correct frame time.

### Q03. Mean decision latency is 12 ms against a 20 ms budget, yet deadlines are missed. What is the first measurement, and what would change your answer?

**Answer:** Measure the latency distribution and attribute the tail: queue wait, blocking time, and interrupt interference separately. If the tail comes from queueing, the arrival process is bursty; if from blocking, a lock is held too long; if from interference, a higher-priority handler scales with input.

**Reasoning:** A mean below the deadline says nothing about the fraction above it. The three tail mechanisms have different signatures and different fixes, and choosing a fix before attributing the tail is guesswork.

**Follow-up:** If the tail is input-dependent, the next test is a swept input amplitude to see whether execution time grows with magnitude, which points at a data-dependent loop or a denormal or saturation path.

**Red flag:** Optimising the mean, or raising the clock before attributing the tail.

**Scoring:** 3 points: distribution, attribution, and one discriminating follow-up.

### Q04. Why must an FIR filter retain state across blocks, and what evidence shows that it did not?

**Answer:** The filter output at sample `n` depends on the previous `n_taps-1` samples, so resetting state at each block boundary computes a different function. Evidence is a spectral line at the block rate and its harmonics when a pure tone is filtered.

**Reasoning:** Block boundaries are an artefact of transport, not of the signal. A reset injects a transient whose period equals the block period, which is precisely a line at `1/T_w` in the spectrum.

**Follow-up:** If the line amplitude is small and the phase error is one sample, the mechanism is an off-by-one overlap rather than a reset; the fix differs.

**Red flag:** "Each block is independent" — true of the transport, false of the signal.

**Scoring:** 2 points: state dependence and a spectral check.

### Q05. An INT8 target disagrees with the float reference only near the maximum sensor code. Give the evidence sequence that separates saturation from a layout mismatch.

**Answer:** Verify raw input bytes and declared dtype/signedness, then check preprocessing saturation counters, then compare intermediate tensors at the first divergence. Saturation correlates with magnitude and produces clamps at the code limit; a layout mismatch moves values between positions regardless of magnitude.

**Reasoning:** A boundary input exercises affine conversion, intermediate width, quantizer range and kernel semantics at once, so the final label is the least informative output available.

**Follow-up:** Pin the operator resolver as well; a reference-kernel fallback changes numerics without changing shapes and is a third competing cause.

**Red flag:** Re-quantizing repeatedly until labels happen to agree.

**Scoring:** 3 points: boundary vector, intermediate comparison, and a stated discriminator.

### Q06. How do you estimate battery life for a duty-cycled node, and which term do you expect to dominate?

**Answer:** Sum `V*I*t` over an exhaustive set of named states to get energy per period, divide by the period for average power, then divide stored energy by average power. Which term dominates is not a general rule; it follows from the comparison `I_idle*(T_w - t_act)` against `I_act*t_act`, and must be evaluated per design.

**Reasoning:** In this sheet's example the active term is 81.1 percent and the idle term 18.9 percent, so halving the idle current moves life from 55.6 to 61.4 hours while shortening the active interval moves it more. Halving model compute alone changes nothing measurable, because inference is 0.38 percent of block energy.

**Follow-up:** Include regulator quiescent current and sensor draw; a 1.2 mA idle assumption with a 0.9 mA regulator floor is not a measurement, and the equality `I_idle = 5.14 mA` is the threshold at which the idle term would overtake the active one in this configuration.

**Red flag:** Quoting data-sheet active current as battery life, or leaving idle current out of the sum.

**Scoring:** 2 points: exhaustive state inventory and unit-correct division.

### Q07. What makes a firmware bug report actionable for a TinyML defect?

**Answer:** Part and revision, clock and power configuration, build identity and map delta, task priorities, exact input trace or seed, observed event trace with timestamp origin, and the safe effect. Anything less cannot be reproduced on a second unit.

**Reasoning:** The defects that matter are configuration-dependent, so a report that omits configuration is a story rather than a reproduction. The map delta is included because a layout change can move a buffer across an alignment boundary without any source change.

**Follow-up:** Attach the golden-vector result for the same build; it separates "the binary computes differently" from "the binary is scheduled differently".

**Red flag:** "It crashed sometimes" or a screenshot of a wrong class with no input.

**Scoring:** 2 points: reproduction fields and the configuration/map requirement.

### Q08. Why is a UART acknowledgement not evidence that an action occurred?

**Answer:** An acknowledgement is produced by the receiver's protocol layer and scopes to frame receipt and parsing. It says nothing about whether the receiver's own safety gate accepted the command, whether its actuator moved, or whether the movement had the intended physical effect.

**Reasoning:** Each layer can acknowledge and then refuse. A protocol ACK followed by a local rejection is a normal, correct behaviour that would be misread as success by anyone treating the ACK as completion.

**Follow-up:** Require a command identifier echoed with the receiver's decision reason, and treat the absence of that decision record as unknown rather than as failure.

**Red flag:** "The ACK came back, so the valve closed."

**Scoring:** 2 points: evidence scope per layer.

### Q09. A host sees duplicate decision records. Enumerate the competing causes and the test that separates them.

**Answer:** Causes are producer retry after a missing ACK, parser replay after a partial frame, and a genuine sequence-wrap collision. Compare sequence and epoch at the device ledger against the host ledger; a device-side single record with two host records is a parser or transport issue, while two device records is a producer issue.

**Reasoning:** Deduplicating on sequence alone merges two boot sessions and hides the difference between a retry and a wrap. Epoch makes the sessions distinguishable.

**Follow-up:** With a 32-bit sequence at 50 blocks/s, wrap takes about 2.72 years, so a wrap collision on a short test is evidence of a reset rather than of arithmetic.

**Red flag:** Dropping duplicates without asking which layer produced them.

**Scoring:** 3 points: three causes, the ledger comparison, and epoch.

### Q10. When should a device reject a remote command outright?

**Answer:** When authority, artifact version, frame integrity, safety preconditions, or freshness fails at the point of use. Each rejection needs a reason code and a hazard-specific safe behaviour, not a generic retry.

**Reasoning:** Acceptance is a conjunction, and any false conjunct makes the action unjustified. A stale command that passes integrity is still unsafe if the plant state it assumed has changed.

**Follow-up:** The test is an injected delay plus a replayed command; if the device still acts, the freshness conjunct is not implemented.

**Red flag:** Rejecting only on transport failure.

**Scoring:** 2 points: the conjunction and a safe behaviour per hazard.

### Q11. Compute the ring retention window and the wear implication for a 512 KiB logging region at 800 bytes/s with a 69 MB/day write volume.

**Answer:** `512 KiB = 524,288` bytes; at 800 bytes/s the region wraps in `524288/800 = 655 s`, about 10.9 minutes. At 69.12 MB/day the daily volume is 69,120,000 bytes, and `524288/69,120,000` of a day is 0.00758 day, which is 655 s — the two routes agree.

**Reasoning:** Both computations are the same division expressed in different units, and agreement between them is the check on the byte-rate conversion. The retention window, not the write rate, is what determines whether a failure can be diagnosed after the fact.

**Follow-up:** With 2 KiB pages and 10,000 erase cycles per page, the total bytes per page over its life is `2048*10000 = 20.5 MB`, so a 69 MB/day write volume consumes roughly 3.4 page-lives per day; that is the wear figure the design must survive.

**Red flag:** Reporting the write rate as the answer, or ignoring erase endurance.

**Scoring:** 3 points: wrap time, unit agreement, and wear.

### Q12. What does the utilisation bound `n*(2^(1/n)-1)` actually promise?

**Answer:** For `n` independent periodic tasks under rate-monotonic assignment, total utilisation at or below the bound guarantees every deadline is met. Above it, the guarantee is withdrawn but the task set may still be feasible, and the exact test is the response-time recurrence.

**Reasoning:** The bound is sufficient, not necessary, and it ignores blocking and interrupt load. A passing bound is a statement about a task model, not about a device.

**Follow-up:** At `n=3` the bound is `3*(2^(1/3)-1) = 0.7798`. If measured utilisation is 0.089 but deadlines are missed, the cause is outside the model: blocking, interrupt jitter, or a task whose period is shorter than declared.

**Red flag:** Treating the bound as feasibility in both directions.

**Scoring:** 3 points: sufficient-not-necessary, the numeric bound, and the missing terms.

### Q13. A two-state power model predicts 55.6 hours. What would you check first if the unit dies in 30?

**Answer:** Check whether a third state exists: wake-up and settling intervals, radio retries, or a regulator that never enters its low-current mode. Then verify that the idle current was measured at the actual supply voltage with the actual peripherals enabled.

**Reasoning:** The model's arithmetic was checked and is right for its inputs, so a large error means the state set is incomplete rather than the division being wrong. Wake transitions are the usual omission because they are short and easy to discount.

**Follow-up:** If the deficit scales with the number of wake cycles, compute the per-wake energy as `(2930.4 - P*108000)/n_wake` and compare against the sensor's settling requirement.

**Red flag:** Halving the predicted idle current until the number matches.

**Scoring:** 3 points: missing state, measurement conditions, and a quantitative attribution.

### Q14. Why does a 192-byte block behave differently on a Cortex-M4F and a Cortex-A core even with identical arithmetic?

**Answer:** On the cached core the block spans three whole cache lines only if line-aligned, so an unaligned base straddles four; handoff then requires maintenance on a partially shared line, and the DMA write bypasses the cache entirely. On the uncached part there is no coherence question, only bus arbitration.

**Reasoning:** The arithmetic is identical; the visibility contract is not. This is why a design that works on the microcontroller can fail on the larger part after a long idle period, when a stale line is most likely to be read.

**Follow-up:** Mark the region non-cacheable and re-run; if the failure disappears, the mechanism is coherence, and the fix is mapping or maintenance rather than more barriers everywhere.

**Red flag:** Copying the same DMA code to both parts with no mapping change.

**Scoring:** 3 points: line alignment, bypass semantics, and the discriminating experiment.

### Q15. What is the difference between "the model is accurate" and "the deployment is correct", and which would you test first?

**Answer:** Accuracy is a property of the function on a distribution; correctness is a property of the pipeline that produces the tensor and consumes the label. Test the pipeline first with golden vectors, because a pipeline defect invalidates every accuracy number produced through it.

**Reasoning:** An offline accuracy figure computed on data that was assembled differently from the device's pipeline measures a function that is not deployed. The pipeline is also cheaper to test: integer inputs, fixed shapes, no board required.

**Follow-up:** If golden vectors pass and field accuracy is still low, the remaining candidates are distribution shift, label-order drift and window misalignment, in that order of likelihood.

**Red flag:** Retraining before the golden-vector harness exists.

**Scoring:** 3 points: definitional separation, test order, and a ranked residual list.

### Q16. A consumer needs 30 ms per block and the rotation has three 20 ms stores with 3 ms peak jitter. Is the design safe, and what changes if jitter doubles?

**Answer:** Worst-case exclusivity is `2*20 - 2*3 = 34 ms`, so a 30 ms consumer fits with 4 ms of margin. With `J=5 ms` the bound becomes `40-10 = 30 ms`, equal to the service time and therefore unsafe, because equality leaves no margin for any additional delay.

**Reasoning:** Jitter is subtracted from the reader's budget rather than added to the block period, and the reader's window is bounded by the shortest following fills. Equality is a failure condition in a design that also has queueing and transport after the read.

**Follow-up:** The quantitative fix is either another store, giving `3*20 - 3*3 = 51 ms` for `K=4`, or a shorter service time; the choice depends on whether SRAM or CPU is the scarcer resource.

**Red flag:** Using the mean block period, or claiming safety from equality.

**Scoring:** 3 points: worst-case subtraction, equality is unsafe, one quantified fix.

### Q17. Compute the requantization multiplier for `s_in=0.02`, `s_w=0.05`, `s_out=0.4` and predict the bias a truncating pipeline accumulates over three layers.

**Answer:** `M = 0.001/0.4 = 0.0025`, so `M0 = round(0.0025*2^30) = 2,684,355`. Truncation loses on average half an output LSB per layer, so three layers give about `1.5` LSB, or `1.5*0.4 = 0.6` real units of systematic offset in one direction.

**Reasoning:** The random part of the per-layer error adds in quadrature, giving `sqrt(3)*0.1155 = 0.20` units, while the truncation bias adds linearly. A pipeline that only measures variance will therefore understate a defect that is visible as a constant shift.

**Follow-up:** Switching to round-to-nearest halves the bias and leaves the variance unchanged, which is why the change shows up as a mean-accuracy improvement with no change in spread.

**Red flag:** Assuming error accumulates as a square root when the mechanism is a one-directional truncation.

**Scoring:** 3 points: the multiplier value, the linear-versus-quadrature distinction, and the unit conversion.

### Q18. A one-minute capture contains 2983 blocks where 3000 are expected. Give two mechanisms consistent with the deficit and the experiment that separates them.

**Answer:** The deficit is 340 ms in 60 s, or 5667 ppm. A clock-tolerance mechanism (an internal RC oscillator with a 1 percent specification) is consistent, as is a mechanism that occasionally loses a whole block. The separating experiment is duration and shape: tolerance scales linearly and smoothly over an hour, predicting 1020 missing blocks, while lost blocks appear as lumps at irregular intervals.

**Reasoning:** A per-block loss of even one sample in three thousand would be 33 ppm, far too small to explain 5667 ppm, so a per-sample leak is arithmetically excluded and only clock scale or whole-block loss remain.

**Follow-up:** Compare the device's block counter against a host timestamp at the start and end of the hour; a slope in ppm identifies tolerance, while a step identifies a reset or an aborted transfer.

**Red flag:** Attributing the deficit to "packet loss" without checking whether the arithmetic can produce that magnitude.

**Scoring:** 3 points: the ppm conversion, two named mechanisms, and a duration-based discriminator.

### Q19. A ring of eight blocks holds 160 ms of input, the producer runs at 50 blocks/s and the consumer at 42. When does the system start shedding, and what are the three policy options?

**Answer:** The deficit is 8 blocks/s against 8 blocks of budget, so shedding begins at 1.0 s. The policies are drop-newest (preserve contiguous history), drop-oldest (preserve recency), and decimate the decision rate so the consumer's load falls below its capacity.

**Reasoning:** `rho=50/42=1.19`, so the backlog grows linearly at 8 blocks per second and the depth is consumed in exactly one second. The policy choice is a statement about what downstream tolerates, not about arithmetic.

**Follow-up:** Decimating decisions from every block to every second block halves the load to 21 blocks/s and makes `rho=0.5` with no extra memory, which is the cheapest of the three.

**Red flag:** Adding ring depth and calling the problem solved, which only delays saturation.

**Scoring:** 3 points: the time, three policies, and one load calculation.

### Q20. A window's standard deviation is computed from `sum` and `sum_sq` in 32-bit integers over 512 samples of 16-bit data. What breaks, and what is the least invasive fix?

**Answer:** The sum of squares reaches `5.5e11`, which overflows 32 bits by a factor of 256, while the sum reaches `1.68e7` and fits. The computed variance becomes negative and the square root yields a NaN or garbage feature; the least invasive fix is a 64-bit accumulator for `sum_sq`.

**Reasoning:** The bound is arithmetic, not statistical: `512*32767^2 = 5.497e11` against a `2^31-1` limit of `2.147e9`. Any input near full scale on any sample triggers it, so the failure appears on exactly the loud windows that matter.

**Follow-up:** An alternative that keeps 32-bit arithmetic is to subtract a fixed offset from each sample before accumulating, which reduces the magnitude of the squares; it works only if the offset is bounded and the residual range is small.

**Red flag:** Blaming the sensor for a negative variance, or casting to float only at the square root.

**Scoring:** 3 points: the overflow arithmetic, the negative-variance symptom, and a concrete fix.

### 11.1 Answer-evidence rubric

| Criterion | Strong answer observable | Weak but repairable | Reject or coach signal |
|---|---|---|---|
| units and origins | names block end, action time, bytes versus symbols | correct arithmetic, no origin | treats baud as bytes per second |
| ownership | names writer, claimant, release instant, reuse instant | says "double buffer" with no timing | permits overlap by implication |
| diagnosis | two or three competing causes plus a separating signal | one plausible cause | changes a threshold first |
| evidence scope | separates harness, target, and field evidence | says "test it" without artifact | calls illustrative values measured |
| safe behaviour | rejects a violated contract with a reason code | says "retry" | treats an ACK as proof of action |

A scored interview should accept a different architecture or API when it preserves the reasoning. Do not award credit for reciting a framework call without its preconditions; ask instead which timestamp feeds the freshness check, which is where ambiguity usually hides. If a candidate proposes a split between a small part and a larger one, request a trace in which the remote side pauses, and a defensible answer will state local behaviour during the pause rather than a faster remote model.

**Source inventory:** [directory.txt lines 1094-1110](../../../../Templates/directory.txt). **Source syllabus:** [edge-arm-firmware](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-arm-firmware). **Evidence boundary:** all arithmetic is illustrative and hand-derived; no target build, board test, power measurement, clinical or safety validation was executed.

The oracle inputs are deliberately hardware-independent — sample identifiers, integer windows, declared timing, frame fields — so they can be exercised in a host harness before a board exists. They do not model ADC analog error, DMA controller semantics, cache configuration, port behaviour, line electrical behaviour, sensor settling, power draw, or fault containment.

## 12. Quick Reference Card

| Need | Rule | Check | Assumption | Avoid when |
|---|---|---|---|---|
| Acquire | publish immutable completed blocks | sequence and epoch ledger | completion implies visible bytes | reader may touch filling storage |
| Rotation | `T_reuse` is the sum of the next `K-1` fills | descriptor-length trace | fills are as configured | descriptors vary per cycle |
| Budget | `R = C + B + interference` | tail histogram, not mean | blocking is bounded | mean latency is the acceptance test |
| Link | 8N1 payload is `baud/10` minus escaping | frame utilisation ledger | no flow stalls | baud quoted as bytes per second |
| DSP | retain filter state per stream | impulse and boundary test | contiguous sample order | blocks treated as independent |
| Quantize | pin `s`, `Z` and layout with the artifact | intermediate tensor golden vectors | kernel contract matches manifest | float output stands in for integer |
| Overload | `t_saturate = B_tok/(lambda-mu)` | backlog slope against depth | rates stationary | buffer depth offered as the fix |
| Energy | sum `V*I*t` over an exhaustive state list | measured idle at real voltage | states are complete | data-sheet current quoted as life |
| Act | check age and authority at the point of use | injected-delay rejection test | safe state is hazard-specific | transport ACK treated as action |
