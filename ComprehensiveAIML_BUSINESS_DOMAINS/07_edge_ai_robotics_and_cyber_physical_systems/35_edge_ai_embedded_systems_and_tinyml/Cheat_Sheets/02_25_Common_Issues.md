---
title: "25 Common Issues — Edge AI, Embedded Systems and TinyML"
topic: "Diagnostic catalogue for embedded acquisition, firmware contracts and TinyML integration"
collection: "module35-edge-ai-embedded-tinyml"
file_id: "02"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
status: "draft"
version: "1.0"
review_trigger: "New target MCU, new sensor, new model artifact, or a defect class not covered by the catalogue"
notation_style: "plain-text"
prerequisites: ["Cheat sheet 01 sections 3-5", "unsigned arithmetic", "linker map reading"]
pattern_tags: ["fault-catalogue", "discriminating-check", "counter-ledger"]
owner_role: "embedded systems reviewer"
source_index: "INDEX.md"
---

# 25 Common Issues — Edge AI, Embedded Systems and TinyML

## 1. File Header

**Purpose:** Provide a unit-specific fault catalogue for TinyML deployments on microcontrollers: thirty issues, each with the observation that separates it from its nearest look-alike, so that a defect is diagnosed by evidence rather than by editing a threshold.

**Prerequisites:** Cheat sheet 01 sections 3 through 5 for the vocabulary, comfort with unsigned modular arithmetic, and the ability to read a linker map. A reader who cannot state who writes a buffer and when a reader may touch it will find the catalogue a list of superstitions rather than mechanisms.

**Key outcome:** Given a symptom — dropped windows, wrong labels near range limits, latency tails, memory growth, resets — select the two or three candidate mechanisms, name the counter or experiment that distinguishes them, and state what the fix changes in the contract.

**Exclusions:** No medical interpretation, no functional-safety certification, no board-specific register configuration, and no measured result for any commercial product. Every number is ILLUSTRATIVE and hand-computed.

**Level:** Core. The individual facts are elementary; the value is in the discrimination between mechanisms that produce the same symptom.

**Evidence:** Reasoning is derived from protocol arithmetic, scheduling theory, fixed-point numerics and memory-hierarchy behaviour. Device parameters are nominal inputs, not observations.

**Retrieval cue:** The catalog begins at section 8. Section 7 is a symptom-to-candidate lookup for use before reading anything else.

**How to use it:** Find the symptom in section 7, open the two or three candidate issues, run the discriminating check listed for each, and only then apply a correction. Applying a correction before running a check is how a second defect is introduced on top of the first.

**What makes this unit-specific:** Every issue below is tied to acquisition timing, bounded buffers, integer model artifacts, or the transport that carries a decision. Generic advice such as "monitor the model" or "add logging" appears nowhere, because neither discriminates between any two mechanisms in the list.

**Takeaway:** A symptom is not a diagnosis; the catalogue exists to convert one into the other with a check that costs minutes.

## 2. Concept Map / Mental Model

```text
symptom observed at host
   -> which ledger is incomplete?  (publish, consume, frame, decision, action)
   -> which contract is unverified? (acquisition, scheduling, artifact, action)
   -> which mechanism is consistent with the ledger arithmetic?
   -> discriminating experiment -> correction -> counter that would have caught it
```

The catalogue is organised around five ledgers, and the first diagnostic move is always to ask which of them is missing. A publish ledger counts completed blocks handed to the consumer. A consume ledger counts blocks that reached feature extraction. A frame ledger counts frames emitted, parsed and rejected. A decision ledger counts classes with scores, epochs and artifact hashes. An action ledger counts accepted, rejected and executed commands.

When all five ledgers exist and reconcile, the defect is inside one of them and the arithmetic narrows the candidates quickly. When one is missing, the defect may be anywhere, and the missing ledger is itself the first finding — not a preliminary to the investigation but a result worth reporting.

The second move is to ask which contract the symptom violates. A dropped window is an acquisition or scheduling question. A wrong label with correct timing is an artifact question. A correct label delivered too late to act on is an action question. Confusing the three produces the most expensive error in this domain: retraining a model to fix a transport defect.

**Takeaway:** Ledger first, contract second, mechanism third, fix last.

### 2.1 What each symptom class points at

```text
counts drift, no gap seen at host        -> acquisition or clock scale
gap seen, count matches device           -> transport or parser
counts match, labels wrong               -> artifact or preprocessing
labels right, action late                -> scheduling or action gate
everything right, unit resets            -> watchdog, power, or fault path
```

Each row narrows the search to a subsystem before any code is read. The rows are ordered by cost of investigation: a counter read is cheaper than a trace, which is cheaper than a board-level measurement.

### 2.2 Why the catalogue is ordered by mechanism, not by severity

Severity depends on the deployment, and a catalogue ordered by severity would be rewritten for every product. The catalogue is ordered so that adjacent entries share a symptom, which puts the competing causes next to each other where they are actually needed. A reader who arrives with "accuracy dropped" should find CI-06, CI-08, CI-14, CI-17 and CI-25 within a page of each other, because those five produce indistinguishable reports.

### 2.3 The three questions that eliminate most candidates

**Is the sample count right?** If published blocks match `elapsed/T_w` within the clock tolerance, the acquisition path is intact and every candidate that assumes lost samples is eliminated at once.

**Is the integer tensor right?** If golden vectors produce identical intermediate tensors on host and target, the artifact path is intact and every candidate that assumes a numeric or layout defect is eliminated.

**Is the age right?** If injected delay produces a rejection at the action boundary, the freshness gate is on the path and every candidate that assumes a missing gate is eliminated.

Three questions, each answerable with an existing counter or a five-minute test, eliminate roughly two thirds of the catalogue.

## 3. Foundational Definitions

| Term | Meaning | Units | Boundary used in this catalogue |
|---|---|---|---|
| `f_s` | per-channel sample rate | samples/s | post-decimation rate, not the sensor ODR |
| `M` | simultaneously sampled channels | count | at least 1 |
| `N` | samples per block per channel | samples | positive integer |
| `T_w` | nominal block duration | s | `N/f_s`, valid only for uniform timing |
| `K` | DMA stores in rotation | count | 2 or 3 in most designs |
| `lambda` | publish rate | blocks/s | `1/T_w` when no loss |
| `mu` | sustainable consume rate | blocks/s | measured under worst-case input |
| `B_tok` | admission budget | blocks | ring depth minus the block in service |
| `J` | peak completion jitter | s | tail deviation, not standard deviation |
| `A` | action age at use | s | `t_use - t_sample_end` |
| `A_max` | maximum acceptable age | s | declared per action, not per model |
| `W_c` | worst-case consumer service time | s | excludes queue wait |
| `s` | quantization scale | real units per LSB | strictly positive |
| `Z` | quantization zero point | integer code | must lie inside the code range |
| `q_min`,`q_max` | code range ends | integer | -128..127 int8, 0..255 uint8 |
| `M0` | requantization multiplier | Q30 integer | fits in 31 bits plus sign |
| `acc` | integer accumulator | integer | width must bound the dot product |
| `sum_sq` | sum of squared window samples | integer | `N*max^2`; overflow candidate |
| `epoch` | boot or configuration generation | uint16 | increments on reset or swap |
| `seq` | block counter | uint32 | compared by modular subtraction |
| `ppm` | parts per million | 1e-6 | used for clock-scale deficits |
| `rho` | offered load | fraction | `lambda/mu` |
| `T_reuse` | earliest DMA overwrite instant | s | sum of the next `K-1` fills |
| `T_excl` | guaranteed reader exclusivity | s | `T_reuse` reduced by jitter |
| `d_eff` | effective independent samples | count | span divided by window length |
| `A_hw` | flash access time per byte | s/byte | from the part's interface spec |
| `I_act`,`I_idle` | active and idle current | A | measured at a stated voltage |
| `E_blk` | energy per block | J | sum over named states |

Several of these terms carry a trap that the catalogue exploits. `T_w` is nominal; the interval that exists on the device is the descriptor duration, and a defect appears when the two diverge. `mu` is a worst-case figure; substituting a mean throughput makes every overload computation optimistic by exactly the amount that matters. `Z` is an integer that must land inside the code range, and a `Z` outside the range silently shifts every dequantized value by a constant.

**Why `d_eff` is in a fault catalogue:** it is the term that converts a valid evaluation into an invalid one. Overlapping windows make consecutive samples dependent, and an accuracy difference computed as if they were independent is a defect in the evidence rather than in the device.

### 3.1 Categories used for every issue

**VIOLATED_ASSUMPTION** — the design relied on a property that does not hold: a buffer nobody else writes, a timer that never arrives early, a sensor already settled, a clock whose tolerance is zero.

**NUMERICAL_EDGE** — the arithmetic is correct in the middle of the range and wrong at a boundary: overflow, saturation, sign extension, truncation bias, a zero point outside the code range.

**SCALE_FAILURE** — the mechanism is right and its magnitude is wrong: a queue that saturates, a log that wears out, a stack that grows with call depth, a link that carries more than it was budgeted for.

**CONCEPT_CONFUSION** — two quantities that look alike are treated as one: baud and bytes per second, sample rate and sensor output rate, an acknowledgement and an action, a mean and a deadline.

The four categories have different diagnostic signatures. Violated assumptions appear only under specific conditions and vanish in the lab. Numerical edges appear at range extremes and vanish at nominal input. Scale failures appear after an interval proportional to the margin, so short tests miss them entirely. Concept confusions appear everywhere and are stable, which makes them the easiest to find if anyone asks the definitional question.

## 4. Core Formulas with Full Derivation

### 4.1 Base case — counter reconciliation

**FORMULA**

```text
expected_blocks = T_elapsed / T_w
deficit         = expected_blocks - observed_blocks
deficit_ppm     = deficit / expected_blocks * 1e6
loss_rate       = max(0, lambda - mu)
t_saturate      = B_tok / loss_rate
```

**WHERE:** `T_elapsed` the observation window in seconds, `T_w` the nominal block period in seconds, `observed_blocks` the publish count over that window, `lambda` and `mu` in blocks/s, `B_tok` in blocks.

**CONSTRAINT:** The deficit is meaningful only if `T_elapsed` is measured on the same clock as `T_w`. If the block period comes from one oscillator and the elapsed time from another, the deficit includes the difference between them, and at 1 percent tolerance that difference is 10,000 ppm — larger than most real defects. The loss expression is a fluid approximation, valid over intervals long compared with `T_w`.

**NUMERICAL EXAMPLE:** A one-hour capture reports 179,000 blocks where 180,000 are expected.

```text
expected = 3600/0.020 = 180,000
deficit  = 180,000 - 179,000 = 1,000
ppm      = 1,000/180,000 * 1e6 = 5555.6 ppm
```

Then with `B_tok=8`, `lambda=50`, `mu=42`:

```text
loss_rate  = 8 blocks/s
t_saturate = 8/8 = 1.0 s
```

**INTERPRETATION:** A 5556 ppm deficit is inside the tolerance of an uncalibrated internal oscillator, so no loss mechanism is required to explain it and the next investigation step is to measure the same interval against a reference clock rather than to look for dropped blocks. The same deficit produced by a loss mechanism would have to lose one block every 3.6 s, which is a detectable pattern rather than a smooth rate.

The saturation time is the other half of the diagnostic. One second of margin means a two-minute bench test cannot distinguish a stable system from one in permanent overload, because the shedding begins after the test ends.

### 4.2 General case — detection thresholds

**FORMULA**

```text
p_min_detect = k / n                     [smallest rate distinguishable from zero]
n_needed     = k / p_expected            [samples for a target rate]
ovf_margin   = 2^(w-1) - 1 / max_term     [headroom before overflow]
bias_total   = n_layers * s_out/2        [truncation bias, worst case]
```

**WHERE:** `k` the number of events that must be observed for a rate estimate to be credible (use 3 to 10 in practice), `n` the number of trials, `w` the accumulator width in bits, `max_term` the largest single term in a sum, `n_layers` the count of truncating stages, `s_out` the output quantization step in real units.

**CONSTRAINT:** The detection bound assumes independent trials; consecutive overlapping windows are not independent, so `n` must be the effective count `d_eff`, not the record count. The bias expression is a worst case in which every truncation goes in the same direction, which is the correct assumption for a signed truncation implemented by shifting.

**NUMERICAL EXAMPLE:** A defect rate of 1 in 50,000 blocks is suspected. With `k=3` the trial count needed is:

```text
n_needed = 3/0.00002 = 150,000 blocks
at 50 block/s that is 3,000 s = 50 minutes
```

And for overflow, with a 32-bit signed accumulator and a maximum term of 32767^2:

```text
limit       = 2^31 - 1 = 2,147,483,647
max_term    = 32767^2  = 1,073,676,289
capacity    = 2,147,483,647/1,073,676,289 = 2.0000 samples
n_at_512    = 512 * 1,073,676,289 = 549,722,259,968  (about 5.497e11)
overflow_ratio = 5.497e11/2.147e9 = 256.0
```

**INTERPRETATION:** The first result says a rare-defect claim needs fifty minutes of continuous capture, which is why one-minute tests "pass" systems that fail in the field. The second says a 32-bit sum of squares holds exactly two full-scale 16-bit samples, so any window longer than two samples overflows — not marginally, but by a factor of 256 at `N=512`.

A third reading is worth stating explicitly, because it is the one that changes behaviour: the correction is not "use a bigger type everywhere" but "know which accumulator must be wider". The plain sum of 512 samples reaches `1.68e7` and fits comfortably; only the square sum needs 64 bits. Widening both wastes cycles and hides which one was at risk.

### 4.3 Variants

**FORMULA**

```text
T_excl     = (K-1)*T_w - (K-1)*J
safe       = T_excl >= W_c + margin
C_link     = baud/(1+n_d+p+n_s) * (1-e_esc)
U_bus      = frame_bytes*decision_rate / C_link
alias_f    = |f_s - f_signal|            [fold for f_signal in (f_s/2, f_s)]
SNR_q      = 6.02*b_q + 1.76
```

**WHERE:** `n_d` data bits, `p` parity bits, `n_s` stop bits, `e_esc` escape fraction, `f_signal` a component above `f_s/2` that reaches the converter, `b_q` quantizer bits.

**CONSTRAINT:** The fold formula gives the observed frequency for a component between `f_s/2` and `f_s` only; components above `f_s` fold repeatedly by the same rule, and a component exactly at a multiple of `f_s` appears at zero frequency, which is indistinguishable from a DC offset.

**NUMERICAL EXAMPLE:** `K=3`, `T_w=20 ms`, `J=3 ms`, `W_c=30 ms`:

```text
T_excl = 2*20 - 2*3 = 34 ms
34 >= 30 + 4 -> passes with exactly 4 ms of margin
if J becomes 5 ms:  40 - 10 = 30 ms -> equality -> fails
```

Then at 460800 baud with `e_esc=0`:

```text
C_link = 460800/10 = 46,080 byte/s
U_bus  = 46*5/46,080 = 230/46,080 = 0.00499
```

And for aliasing, `f_s=1600`, component at 1400 Hz:

```text
alias_f = |1600 - 1400| = 200 Hz
```

**INTERPRETATION:** The jitter variant shows the failure mode that no mean-based analysis catches: a design passes with 4 ms of margin and fails when jitter doubles, with no change to any code path. Equality is a failure, not a pass, because the margin must also cover the queue wait that follows the read.

The link variant shows the opposite: at 460800 baud the decision traffic is 0.5 percent of capacity, so transport bandwidth is essentially never the constraint at five decisions per second. A team that suspects the link in this configuration is looking in the wrong place, and the arithmetic says so before any measurement.

The aliasing variant is the one that corrupts datasets rather than devices. A 1400 Hz resonance appears at 200 Hz, which is inside the band any model will happily learn, so the artefact becomes a feature and survives every validation that reuses the same pipeline.

### 4.4 Limits and counterexamples

**FORMULA**

```text
q = clamp(round(x/s) + Z, q_min, q_max)
range = [(q_min - Z)*s , (q_max - Z)*s]
x_eff = s*(q - Z)                        [after clamping]
saturation_excess = |x| - max(|range ends|)   [when positive]
gap = (uint32)(seq_new - seq_old) - 1
```

**WHERE:** `x` the real input, `s` and `Z` the quantization parameters, `q` the stored code, `range` the representable interval in real units, `seq` a 32-bit counter compared by modular subtraction.

**CONSTRAINT:** The range formula assumes `Z` lies inside `[q_min, q_max]`. If `Z` is outside, one end of the range becomes unreachable and the effective representable interval shrinks without any error raised. The gap formula excludes the arriving record itself; omitting the `-1` overstates loss by one block per gap.

**NUMERICAL EXAMPLE:** `s=0.05`, `Z=-3`, int8 codes `-128..127`:

```text
lower = (-128 - (-3))*0.05 = (-125)*0.05 = -6.25
upper = ( 127 - (-3))*0.05 = (130)*0.05 =  6.50
input x = 7.00 -> clipped to 6.50, error 0.50 units = 10 LSB
```

Then a sequence jump from 100 to 105:

```text
difference = 105 - 100 = 5
lost       = 5 - 1 = 4 blocks
reporting 5 overstates loss by 25 percent
```

**INTERPRETATION:** The clamping example is why a boundary test is not optional: at `x=7.00` the quantizer produces a code that dequantizes to 6.50, and no downstream stage can distinguish that from a genuine 6.50. The error is 10 LSB, twenty times the half-step bound that the quantization theorem promises, because the theorem's no-overload precondition has been violated.

The gap example is small arithmetic with a large consequence: a 25 percent error in a loss figure is enough to send an investigation after a transport defect that does not exist, while the real defect is a producer that skipped a block.

**Validity condition for the catalogue:** every check below assumes that the firmware publishes counters rather than only logs, that timestamps have a stated origin and unit, and that a comparison is made against a reference clock or a golden vector rather than against another build of the same code. Where a check depends on a behaviour that cannot be observed, the issue says so and names the observation that would be needed.

## 5. Worked Numerical Example

The scenario is ILLUSTRATIVE: a wrist-worn three-axis device, 1600 Hz per axis, 32-sample blocks, a 512-sample window, a 36-feature int8 vector, a three-class model, decisions every fourth block, UART at 921600 baud, 96 KiB SRAM, no data cache. A field report says accuracy on two of the classes fell by 0.4 points after a firmware update and that occasional decisions look "one class late".

### 5.1 Step 1 — establish the ledgers

```text
publish rate      lambda = 1/0.020 = 50 block/s
consume rate      mu     = 42 block/s measured under worst-case input
offered load      rho    = 50/42 = 1.190
admission budget  B_tok  = 8 block
t_saturate              = 8/(50-42) = 1.0 s
```

The consume rate is below the publish rate, so the device is in permanent overload and shedding from the second second of every session. That single number explains "one class late": the model is classifying a window that is older than the ledger claims, because the queue holds blocks that were never counted as late.

### 5.2 Step 2 — quantify the age at use

```text
queue wait at occupancy L = lambda*W  ->  W = L/lambda
at L = 8 block full:  W = 8/50 = 0.160 s
window span = 512/1600 = 0.320 s
age at use = W + (0.320 - 0.020) + service
           = 0.160 + 0.300 + 0.006 = 0.466 s
A_max declared = 0.120 s
```

An age of 466 ms against a 120 ms bound means every decision taken while the ring is full should have been rejected. The fact that none was rejected localises a second defect: the freshness gate is comparing timestamps that both move with the queue, so their difference stays small while the absolute age grows.

### 5.3 Step 3 — recompute the accuracy claim

```text
evaluation windows in a 60 s capture = 60/0.020 = 3,000
window span = 0.320 s, so d_eff = 60/0.320 = 187.5 -> 187
inflation factor = sqrt(3000/187) = sqrt(16.04) = 4.006
reported 95 percent CI half-width, naive = 1.96*sqrt(0.9*0.1/3000) = 0.0107
corrected half-width = 4.006*0.0107 = 0.0429
```

The reported 0.4 point change is one tenth of the corrected confidence half-width, so the accuracy claim does not survive its own arithmetic. The correct conclusion from that capture is "no detectable change", and the search should move to the timing defects already found.

### 5.4 Step 4 — independent re-check of the central numbers

```text
Check A (block rate): 1000/20 = 50 block/s; 1/0.020 = 50. agree.
Check B (deficit at rho): 50-42 = 8 block/s; 8 block budget / 8 = 1.0 s. agree.
Check C (age): 0.160 + 0.300 + 0.006 = 0.466 s. Recompute as
         160 ms + 300 ms + 6 ms = 466 ms. agree.
Check D (effective samples): 60 s/0.320 s = 187.5; sqrt(3000/187.5) = 4.0. agree.
Check E (CI): sqrt(0.09/3000) = sqrt(3e-5) = 0.005477; *1.96 = 0.01074; *4 = 0.0429. agree.
Check F (fraction of a CI): 0.004/0.0429 = 0.0932, about 9.3 percent. agree.
```

The cross-check that matters is Check F: the claimed change is 9.3 percent of the corrected half-width, which is a quantitative statement that the observation is noise. Stating it that way ends the discussion, whereas "it might be noise" invites another capture and another round of speculation.

### 5.5 Step 5 — cost the fixes and rank them

| Fix | Effect on the arithmetic | Memory cost | Schedule cost | Risk |
|---|---|---|---|---|
| decimate decisions to every second block | `rho` 0.5, age falls to 0.186 s | none | one scheduler change | decision cadence halves |
| raise `mu` by reducing per-block DSP | `rho` below 1 if `mu>50` | none | DSP work | accuracy may change |
| add ring depth | delays saturation, does not remove it | 192 byte per block | none | hides the deficit |
| fix freshness gate to use absolute time | rejects stale decisions correctly | 8 byte | small | operational rejections appear |
| retrain the model | no effect on any number above | flash | largest | misattributes the defect |

The ranking is by evidence, not by effort: the first two address the arithmetic that was actually violated, the third conceals it, the fourth exposes it, and the fifth addresses a question the data does not support.

### 5.6 Step 6 — what the counters must publish afterwards

| Counter | Unit | Why it discriminates |
|---|---|---|
| publish count | block | converts any rate claim into an integer count |
| consume count | block | separates acquisition loss from consumer backlog |
| drop count by policy | block | makes the shedding policy observable |
| ring high-water occupancy | block | reveals permanent overload in one number |
| absolute age at use | ms | catches a gate that compares two queue-relative clocks |
| rejection count and reason | count | proves the gate is on the path |
| artifact hash at boot | hex | attributes any post-update change to a specific artifact |

Six of these seven cost fewer than forty bytes in total. The seventh, the artifact hash, is the only one that requires a design decision about where the manifest lives.

## 6. Visual / Geometric Intuition

### 6.1 Counter reconciliation as a bar comparison

```text
expected |############################################| 180,000
observed |########################################### | 179,000
deficit                              |#|               1,000
         the visible bar is 99.44 percent of the expected one
```

The eye cannot see a 0.56 percent shortfall, which is why rate counters and "looks fine" judgments fail on this class of defect. The arithmetic is the only sense organ available, and it needs a reference clock.

### 6.2 Backlog growth and saturation

```text
backlog
 8 |                    *  shedding begins
 7 |                 *
 6 |              *
 5 |           *
 4 |        *
 3 |     *
 2 |  *
 1 |*
   +--|----|----|----|----|----> t (s)
      0.2 0.4 0.6 0.8 1.0 1.2
```

The slope is `lambda-mu`, the intercept with the depth is `t_saturate`, and the shape is a straight line because the deficit is stationary. A curve that bends downward means the consumer speeds up as it queues, which happens when the queue is in cache; a curve that bends upward means it slows, and no amount of depth fixes that.

### 6.3 Saturation against the calibration range

```text
calibration range   |--------- 5.00 ---------|
deployment range          |----------- 7.00 -----------|
quantizer range     |-- -6.25 ....... 6.50 --|
clipped region                          |##|  above 6.50
```

The clipping region is where the model receives a constant and reports a confident class. Nothing in the device says "clipped" unless a counter says it, and the counter must count samples rather than blocks, or 500 clipped samples in one block report as a single event.

### 6.4 Jitter erodes exclusivity

```text
fills      |--17--|--20--|--20--|--23--|
exclusivity after completion at 0:
   nominal 2*20 = 40 ms  ................ safe for W_c=6
   worst   17+17 = 34 ms  ............... safe for W_c=6 with 28 ms spare
   design  2*20 - 2*3 = 34 ms ........... the number to compare
   failure 34 ms vs W_c=35 ms ........... overrun by 1 ms
```

The last line is the whole point of the diagram: a one-millisecond overrun produces a block whose tail belongs to the next fill, and every checksum downstream still passes because each byte is individually valid.

### 6.5 Sequence space as a circle

```text
            0
       2^32-1   1
       ...       ...
     2^31+1   2^31
valid comparison: any difference less than half the range (2,147,483,648)
```

Modular arithmetic means "newer" is a direction rather than a magnitude. A consumer that compares magnitudes treats every wrapped value as ancient and drops it, and the symptom is a total data outage exactly once every 994 days at 50 blocks per second.

### 6.6 Stack growth under interrupt nesting

```text
SRAM top
  |  main stack frame            2,048 byte
  |  + task A frame                512 byte
  |  + task B frame                640 byte
  |  + nested ISR frames           896 byte
  v  high-water mark              4,096 byte measured
  -------------------------------------------
     guard region                  256 byte
  -------------------------------------------
  |  static feature buffer       3,072 byte
```

The diagram shows why the failure presents as corrupted features rather than as a crash: the stack grows downward into a static region that is not protected by any exception. The guard region is what converts a silent corruption into a detectable one.

## 7. Comparison / Selection Table

### 7.1 Symptom to candidate issues

| Observed symptom | First candidates | Check that eliminates half of them |
|---|---|---|
| accuracy fell after an update | CI-06, CI-08, CI-18, CI-23 | compare artifact hashes and golden vectors |
| decisions arrive in bursts | CI-09, CI-20, CI-26 | publish versus consume counts |
| one class systematically late | CI-11, CI-27, CI-10 | absolute age at use, not relative |
| wrong labels near range limits | CI-06, CI-07, CI-12, CI-24 | boundary input and saturation count |
| device resets under load | CI-21, CI-22, CI-28 | reset cause and last sequence |
| filters ring at the block rate | CI-14, CI-16 | single tone and spectrum |
| axes swapped in some builds | CI-16, CI-23 | one-channel square wave |
| timing passes in debug, fails in release | CI-05, CI-29 | artifact and map delta |
| link utilisation suspiciously high | CI-03, CI-21 | bytes per second, not baud |
| energy life far below prediction | CI-28, CI-30 | state inventory and wake count |

### 7.2 Category to typical detection cost

| Category | Appears after | Detected by | Often missed because |
|---|---|---|---|
| VIOLATED_ASSUMPTION | a specific condition | targeted injection | the lab never produces the condition |
| NUMERICAL_EDGE | a boundary input | boundary vectors | averages hide extremes |
| SCALE_FAILURE | a proportional interval | long-run counters | tests are shorter than the margin |
| CONCEPT_CONFUSION | always | definitional question | nobody asks what a unit means |

### 7.3 Instrumentation coverage against the catalogue

| Instrument | Issues it can confirm | Issues it cannot touch |
|---|---|---|
| publish and consume counters | CI-01, CI-09, CI-20, CI-26 | CI-06, CI-08, CI-12 |
| per-layer golden vectors | CI-06, CI-07, CI-12, CI-18, CI-23, CI-24 | CI-09, CI-10, CI-11 |
| absolute age at use | CI-10, CI-11, CI-27 | CI-06, CI-16, CI-23 |
| stack high-water paint | CI-17 | everything outside the stack |
| reset cause register | CI-21, CI-22, CI-28, CI-30 | CI-06, CI-14 |
| saturation sample count | CI-06, CI-08, CI-24 | CI-09, CI-11 |

The coverage table is the cheapest possible audit of a defect backlog: if the instruments available cannot confirm a hypothesis, the hypothesis cannot be tested on that device and the investigation needs a different instrument before it needs a different theory.

### 7.4 Choosing a discriminating experiment

| Candidate pair | Shared symptom | Experiment | Expected separation |
|---|---|---|---|
| CI-01 versus CI-02 | one block with a wrong tail | checksum inside versus outside the handler | equal checksums isolate the pointer |
| CI-09 versus CI-20 | host-side gap | device publish count over the same interval | no device gap isolates transport |
| CI-06 versus CI-23 | wrong labels at extremes | boundary vector plus intermediate tensors | clamp evidence isolates saturation |
| CI-14 versus CI-16 | block-rate artefacts | single tone at a non-harmonic frequency | line at `1/T_w` isolates reset |
| CI-17 versus CI-01 | corrupted static buffer | paint pattern and address comparison | high-water near the limit isolates stack |
| CI-29 versus CI-05 | timing differs between builds | artifact hash and map delta | identical binary isolates instrumentation |

## 8. Failure Modes & Edge Cases

The catalogue below is the required issue register. Each entry states the category, the symptom as it presents, the competing causes that produce that symptom, the check that separates them, the correction, the prevention that would have stopped it, and the verification that proves the correction took effect.

| ID | Category | One-line symptom |
|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | a block's tail belongs to the next fill |
| CI-02 | VIOLATED_ASSUMPTION | reader and writer touch the same store |
| CI-03 | CONCEPT_CONFUSION | link reported as bytes per second from baud |
| CI-04 | SCALE_FAILURE | ring sized from mean jitter overruns at the tail |
| CI-05 | NUMERICAL_EDGE | variance comes out negative on loud windows |
| CI-06 | NUMERICAL_EDGE | int8 output clamps at the calibration edge |
| CI-07 | CONCEPT_CONFUSION | scale and zero point exchanged between layers |
| CI-08 | VIOLATED_ASSUMPTION | deployment amplitude exceeds calibration range |
| CI-09 | SCALE_FAILURE | consumer slower than producer, permanent backlog |
| CI-10 | CONCEPT_CONFUSION | mean latency offered as deadline evidence |
| CI-11 | VIOLATED_ASSUMPTION | freshness gate absent from the action path |
| CI-12 | NUMERICAL_EDGE | requantization product exceeds accumulator width |
| CI-13 | SCALE_FAILURE | flash read time dominates per-MAC cost |
| CI-14 | VIOLATED_ASSUMPTION | filter state reset at every block boundary |
| CI-15 | CONCEPT_CONFUSION | sensor ODR used as the sample rate |
| CI-16 | NUMERICAL_EDGE | 12-bit code not sign-extended into 16 bits |
| CI-17 | SCALE_FAILURE | stack grows into a static buffer |
| CI-18 | VIOLATED_ASSUMPTION | weight bytes updated without their constants |
| CI-19 | CONCEPT_CONFUSION | transport acknowledgement treated as an action |
| CI-20 | NUMERICAL_EDGE | sequence gap counted one too high |
| CI-21 | SCALE_FAILURE | per-sample logging saturates the link |
| CI-22 | VIOLATED_ASSUMPTION | watchdog refreshed by an interrupt, not a task |
| CI-23 | CONCEPT_CONFUSION | tensor layout transposed between producer and model |
| CI-24 | NUMERICAL_EDGE | zero point outside the code range |
| CI-25 | SCALE_FAILURE | overlapping windows treated as independent samples |
| CI-26 | VIOLATED_ASSUMPTION | cache coherence assumed for a DMA region |
| CI-27 | CONCEPT_CONFUSION | timestamp taken at interrupt entry, not sample end |
| CI-28 | NUMERICAL_EDGE | energy per block divided by the wrong interval |
| CI-29 | VIOLATED_ASSUMPTION | debug-build timing generalised to release |
| CI-30 | SCALE_FAILURE | wake settle time omitted from the sample ledger |

### CI-01 — DMA completion treated as input validity

- **Category:** VIOLATED_ASSUMPTION. The design assumes that a completion event implies a usable buffer, which is false on a cached core and false whenever the publish order is wrong.
- **Symptom:** the consumer occasionally reads a block whose final samples belong to the following fill; class outputs flicker on otherwise stationary input, and the flicker is not reproducible in a quiet lab because it needs load.
- **Competing causes:** (a) the consumer claimed a store still being filled because the pointer was published before the sequence increment; (b) the completion interrupt fired but the last bytes were not visible to the consumer's core; (c) the consumer released the store early and the engine reused it while a later stage still held a reference.
- **Discriminating check:** compute a checksum over the block inside the handler and again inside the consumer from the same pointer. Divergent checksums prove a visibility problem; identical checksums with a corrupted tail prove a pointer or ordering problem; identical checksums with no corruption but a wrong class prove neither and point at the artifact.
- **Correction:** publish a four-field descriptor — pointer, length, sequence, epoch — with a single-word atomic store after the visibility operation, and give the consumer no access to the raw descriptor that the engine writes.
- **Prevention:** one producer and one consumer with a declared release point, plus a boot-time self-test that publishes a known pattern and verifies it through the consumer path.
- **Verification:** inject a deliberate delay of 60 ms at the consumer for 100 blocks and confirm the overrun counter increments while the class output remains defined, which shows the release contract is enforced rather than assumed.

### CI-02 — Double buffering without a release protocol

- **Category:** VIOLATED_ASSUMPTION. Alternating storage is treated as a guarantee of exclusivity, which it is only if the consumer releases before the engine returns to that store.
- **Symptom:** corruption that appears only when a downstream stage is slow, for example when logging is enabled, and that vanishes when the log is removed.
- **Competing causes:** (a) no release call at all, so the engine assumes the store is free after one period; (b) a release call present on the fast path and absent on an error path; (c) the release implemented as a flag write that the engine never reads.
- **Discriminating check:** instrument the release path and count releases against claims. A mismatch that grows only while logging is enabled is conclusive for (a) or (b); a permanent match with intermittent corruption is conclusive for (c).
- **Correction:** make the release unconditional in the consumer's epilogue, including error returns, and have the engine refuse to write a store whose release flag is clear, reporting the refusal rather than overwriting.
- **Prevention:** encode the store state in the descriptor so a claim and a release are both observable, and make the state machine small enough to review exhaustively.
- **Verification:** run the fault-injection build with an error injected in the consumer, and confirm the engine reports refusals instead of corrupting; then confirm normal operation produces zero refusals.

### CI-03 — Baud rate used as a byte rate

- **Category:** CONCEPT_CONFUSION. A symbol rate is compared with a payload rate, and an eight-data-bit frame is assumed to carry eight bits per ten symbols.
- **Symptom:** the link is believed to have 25 percent more capacity than it has, so a design passes on paper and overruns when escaping, headers and acknowledgements are added.
- **Competing causes:** (a) genuine confusion about framing; (b) a calculation that used `baud/8` and was copied into a budget document; (c) a parity or stop-bit configuration that differs from the one assumed in the arithmetic.
- **Discriminating check:** compute `baud/(1+n_d+p+n_s)` for the actual configuration and compare with the measured payload over a timed interval. A 25 percent discrepancy confirms (a); a smaller discrepancy indicates (c) or escaping.
- **Correction:** express the budget in bytes per second, derived from the configured frame format, and multiply by `(1+e_esc)` for the framing scheme in use.
- **Prevention:** put the frame format in one constant that the budget calculation and the driver both read, so the two cannot disagree.
- **Verification:** transmit one kilobyte of known payload and measure the elapsed interval against the budget figure; agreement within the escaping allowance proves the arithmetic matches the wire.

### CI-04 — Ring depth sized from mean jitter

- **Category:** SCALE_FAILURE. The design uses the mean block period to size the buffer, and the guaranteed exclusivity is the sum of the shortest following fills, which is shorter.
- **Symptom:** rare one-sample corruptions at the end of a block, appearing only under interrupt load, and never during a short quiet test.
- **Competing causes:** (a) jitter not measured at all; (b) jitter measured as a standard deviation rather than a peak; (c) the consumer's service time measured on average instead of at its worst.
- **Discriminating check:** capture the interval between consecutive completions for at least ten thousand blocks and read the minimum pair sum, then compare with `W_c`. A minimum pair sum below `W_c` proves the mechanism and eliminates every model-side explanation.
- **Correction:** size the depth from `(K-1)*(T_w - J)` and require that bound to exceed `W_c` by a stated margin, or increase `K`.
- **Prevention:** make the sizing expression part of the boot-time self-test, evaluated from the configured descriptor lengths and the measured jitter counter.
- **Verification:** replay a capture with injected jitter at the measured peak and confirm zero overruns, then at twice the peak and confirm the refusal path engages rather than corrupting.

### CI-05 — Sum of squares overflow

- **Category:** NUMERICAL_EDGE. A 32-bit accumulator is used for a quantity whose maximum is `N*max^2`, which exceeds the type for any realistic window.
- **Symptom:** the standard deviation feature is negative, zero, or wildly large on loud windows only, and the model's output changes character between quiet and loud recordings.
- **Competing causes:** (a) genuine overflow of the square sum; (b) an overflow of an intermediate product before accumulation; (c) a sign error in converting an unsigned sum to a signed type.
- **Discriminating check:** recompute the window's variance in 64-bit arithmetic on the host from the same raw samples. If the device value differs and the host value is stable, the mechanism is overflow; then compare an intermediate product to distinguish (a) from (b).
- **Correction:** use a 64-bit accumulator for the square sum, or subtract a fixed centring offset before squaring and bound the residual range at compile time.
- **Prevention:** an assertion at build time that `N*max^2` fits the chosen accumulator, written as a static comparison rather than a comment.
- **Verification:** run the harness with a full-scale square wave and confirm the variance is positive, finite, and equal to the host's 64-bit computation within one LSB.

### CI-06 — Int8 output clamps at the calibration edge

- **Category:** NUMERICAL_EDGE. The quantizer's representable range is narrower than the deployment's amplitude range, so large inputs saturate and dequantize to a plausible moderate value.
- **Symptom:** accuracy is good on the calibration corpus and poor on the energetic recordings, and the confusion is concentrated in the class that correlates with amplitude.
- **Competing causes:** (a) calibration range narrower than deployment; (b) correct range but a preprocessing gain that differs from the one used at calibration; (c) a clamp implemented with wraparound rather than saturation.
- **Discriminating check:** count clipped samples per window and publish the per-block maximum raw code. A window whose maximum sits at the code limit with a nonzero clip count isolates (a) or (b); a value that jumps from maximum to minimum isolates (c).
- **Correction:** widen the calibration range to cover the deployment's percentile range, or narrow the input with an explicit gain that is versioned with the artifact.
- **Prevention:** calibration on data that includes the deployment's loudest decile, with the resulting range published in the manifest and asserted at load.
- **Verification:** feed a swept-amplitude signal through both paths and confirm the clip count is zero up to the declared maximum, and that beyond it the count increases monotonically rather than the output wrapping.

### CI-07 — Scale and zero point exchanged

- **Category:** CONCEPT_CONFUSION. Two per-tensor parameters are applied in the wrong roles, or one layer's parameters are applied to another layer's tensor.
- **Symptom:** outputs are shifted and scaled by a constant that varies by layer, so the final class is often right and the confidence is wrong.
- **Competing causes:** (a) arguments transposed at the call site; (b) parameters read from an array indexed by the wrong layer; (c) a symmetric quantizer's zero point assumed to be zero when the artifact declares otherwise.
- **Discriminating check:** dequantize a single intermediate tensor with the manifest's parameters on the host, and compare with the device's intermediate values. A constant multiplicative offset isolates (a); an offset that grows with depth isolates (b).
- **Correction:** pass a single struct that carries scale and zero point together, so the two cannot be separated, and read it from the manifest rather than from positional arguments.
- **Prevention:** a static assertion on the parameter structure plus a unit test that dequantizes a known code and compares with a hand-computed value.
- **Verification:** for a code of 100 with `s=0.05` and `Z=-3`, the dequantized value must be `0.05*(100+3) = 5.15`; confirm the device reports exactly that.

### CI-08 — Deployment amplitude outside the calibration range

- **Category:** VIOLATED_ASSUMPTION. The calibration corpus is assumed to be representative of the deployment distribution, which fails for a new mounting, a new subject population, or a new operating mode.
- **Symptom:** the failure is silent for weeks and then appears as a class collapse in a specific condition, with no error and no counter movement.
- **Competing causes:** (a) calibration corpus not representative; (b) a sensor range setting changed at the same time, altering the raw scale; (c) a firmware gain applied before quantization but not after.
- **Discriminating check:** compare the deployment's raw amplitude percentiles against the calibration set's. A deployment 99th percentile above the calibration maximum by more than the quantization step isolates (a); a step change in raw codes at a firmware boundary isolates (b) or (c).
- **Correction:** recalibrate on data that spans the deployment range, and publish the declared input range so the device can refuse or flag out-of-range windows.
- **Prevention:** an out-of-range counter with a declared behaviour, plus a monitoring threshold on the counter as part of the release criteria.
- **Verification:** inject a window 20 percent above the declared maximum and confirm the flag is raised and the decision carries the out-of-range marker rather than a confident class.

### CI-09 — Consumer permanently slower than the producer

- **Category:** SCALE_FAILURE. `rho` exceeds one, so the backlog grows without bound and the buffer converts overload into loss or staleness rather than absorbing it.
- **Symptom:** decisions arrive in bursts, the oldest block in the ring is always the same age, and the drop count rises linearly with time after start-up.
- **Competing causes:** (a) `mu` measured on nominal input and the worst case slower; (b) `mu` correct but `lambda` higher than designed because the descriptor length shrank; (c) an intermittent stall, such as a flash erase, that lowers the effective `mu` without changing the nominal figure.
- **Discriminating check:** plot ring occupancy against time. A straight line to the depth and then a plateau proves sustained overload; a sawtooth with occasional spikes proves (c); a plateau at a higher `lambda` than designed proves (b).
- **Correction:** either raise `mu` by removing work from the decision path, or lower `lambda` by decimating the decision rate, and declare the shedding policy explicitly.
- **Prevention:** measure `mu` under the worst-case input in the acceptance test, and require `rho <= 0.8` rather than `rho < 1`.
- **Verification:** run at the design rate for one hour and confirm ring high-water stays below the declared budget with zero drops.

### CI-10 — Mean latency offered as deadline evidence

- **Category:** CONCEPT_CONFUSION. A mean is compared with a bound that applies to every instance, and the comparison is presented as a pass.
- **Symptom:** a timing report says 12 ms against a 20 ms limit while operators observe late actions at a low rate.
- **Competing causes:** (a) genuinely a mean-versus-bound confusion; (b) a worst-case figure computed from a capture too short to contain the tail; (c) a deadline that is not the same quantity as the latency measured, for example device time versus host arrival time.
- **Discriminating check:** compute the fraction above the bound from the same capture and compare with the operator-reported rate. A nonzero fraction with a passing mean proves (a); agreement between the fraction and the operator rate but a short capture proves (b).
- **Correction:** replace the acceptance criterion with a fraction-above-bound limit and a maximum, both computed from a capture long enough to contain at least a hundred deadline events.
- **Prevention:** the acceptance script prints the distribution, not a single number, and fails on the fraction rather than on the mean.
- **Verification:** inject a 40 ms stall in one block out of a thousand and confirm the script fails, which proves the criterion is sensitive to the tail.

### CI-11 — Freshness checked at the producer

- **Category:** VIOLATED_ASSUMPTION. Age is validated where the decision is made rather than where it is used, so the gate passes anything that was fresh at production time.
- **Symptom:** actions occur correctly but late, and the rejection counter never moves even when the transport is deliberately slowed.
- **Competing causes:** (a) the gate is genuinely at the producer; (b) the gate exists at the action boundary but compares two clocks that move together; (c) the gate rejects but the caller ignores the return value and acts anyway.
- **Discriminating check:** inject a delay between decision and action and watch the rejection count. No movement proves (a) or (c); movement without a behavioural change proves (c) specifically, since the rejection is computed and discarded.
- **Correction:** move the age comparison to the action boundary, using an end-of-sample timestamp and a monotonic local clock, and make rejection a state that downstream code must handle explicitly.
- **Prevention:** a fault-injection test in the acceptance suite that delays the action path and asserts the rejection count increases.
- **Verification:** with a 200 ms injected delay and a 120 ms bound, every action must be rejected and the count must equal the number of injected events.

### CI-12 — Requantization product exceeds the accumulator width

- **Category:** NUMERICAL_EDGE. A 31-bit multiplier applied to a 32-bit accumulator needs up to 63 bits, and a 32-bit implementation wraps silently.
- **Symptom:** outputs are correct for small activations and wrong, often dramatically, for large ones, so the error correlates with input magnitude in a way that looks like saturation.
- **Competing causes:** (a) genuine wrap of the product; (b) a shift amount that is wrong by one, producing a factor-of-two error everywhere; (c) a rounding constant added at the wrong point, producing a bias rather than an outlier.
- **Discriminating check:** compare with a 64-bit host implementation at three magnitudes. Errors that appear only above a threshold prove (a); a uniform factor of two proves (b); a small constant offset proves (c).
- **Correction:** implement the multiply as two 32-bit operations with explicit carry handling, or restrict the multiplier to 16 bits with a compensating shift, and assert the accumulator bound at compile time.
- **Prevention:** compute the maximum possible dot product for the declared shapes and compare it with the accumulator range in a static assertion.
- **Verification:** run the golden vectors at the maximum activation value and confirm equality with the 64-bit reference; then run at twice the maximum to confirm the kernel saturates rather than wraps.

### CI-13 — Flash read time dominates per-MAC cost

- **Category:** SCALE_FAILURE. The cost model assumes one MAC per cycle, which holds only when weights are in a cache or in tightly coupled memory.
- **Symptom:** measured inference time is two to four times the predicted figure, and the ratio grows with model size while the MAC count is unchanged.
- **Competing causes:** (a) weights fetched from external flash on every pass; (b) a cache that is large enough for the first layer and too small for the whole model, so later layers miss; (c) a bus contention effect from DMA traffic that competes with instruction fetch.
- **Discriminating check:** compare inference time with DMA disabled and with the model entirely in RAM. A large improvement in the RAM case proves (a); an improvement that depends on layer order proves (b); sensitivity to acquisition rate proves (c).
- **Correction:** place the hot portion of the model in tightly coupled memory, or restructure the schedule so that a layer's weights are streamed once and used fully.
- **Prevention:** the latency budget states the memory placement explicitly and the acceptance test runs with the production placement, not a debug one.
- **Verification:** measure with the production map and confirm the predicted and observed times agree within the stated tolerance; a mismatch reopens the budget rather than the model.

### CI-14 — Filter state reset at each block

- **Category:** VIOLATED_ASSUMPTION. Blocks are treated as independent signals, which is true of the transport and false of the sampled waveform.
- **Symptom:** a spectral line at the block rate and its harmonics, a small accuracy loss, and a visible transient at every boundary in a plotted waveform.
- **Competing causes:** (a) state genuinely reset per block; (b) an overlap that is off by one sample, producing a small phase discontinuity; (c) a windowing function applied where a continuous filter was intended.
- **Discriminating check:** filter a pure tone far from any harmonic of the block rate and inspect the spectrum. A line at exactly `1/T_w` isolates (a); a small phase error with no block-rate line isolates (b); a broadened tone isolates (c).
- **Correction:** retain state per stream across blocks and reset only on a declared discontinuity such as a sensor range change.
- **Prevention:** concatenate block outputs and compare with a single-shot run over the same samples, as a harness test rather than a manual exercise.
- **Verification:** confirm the concatenated output matches the single-shot output to within one LSB over ten thousand samples.

### CI-15 — Sensor output rate used as the sample rate

- **Category:** CONCEPT_CONFUSION. The device's internal conversion rate is treated as the rate delivered to the buffer, ignoring decimation and averaging inside the sensor.
- **Symptom:** the window's physical span is a multiple of the intended one, features are smoother than expected, and every firmware timing calculation agrees with itself.
- **Competing causes:** (a) genuine rate confusion after a sensor configuration change; (b) correct rate but a changed filter setting inside the sensor that adds group delay; (c) a decimation stage in firmware whose factor was not propagated to the window length.
- **Discriminating check:** count completions over ten seconds and compare with `1/T_w`. A period that is exactly twice the expected value proves (a) or (c); a correct period with smoothed content proves (b).
- **Correction:** derive the window's sample count from a measured or read-back rate rather than from a constant, and assert the measured block period against the configured value at start-up.
- **Prevention:** read the sensor's actual configuration back at boot and publish it alongside the artifact hash.
- **Verification:** a step input must change the first feature within one window span of the declared value, measured empirically.

### CI-16 — Twelve-bit code not sign-extended

- **Category:** NUMERICAL_EDGE. A signed converter result is stored in a wider field without sign extension, so negative codes appear as large positive values.
- **Symptom:** an axis appears offset by a constant half-range, features on that axis have a doubled mean, and the class confusion looks like a mounting error.
- **Competing causes:** (a) missing sign extension; (b) correct extension but a byte-order mismatch in the DMA packing; (c) a channel-order mismatch that swaps a signed axis with an unsigned one.
- **Discriminating check:** print one triple of raw codes with the sensor held still and level. A negative value appearing as a large positive one proves (a); a byte-swapped pattern proves (b); a plausible triple on the wrong channel proves (c).
- **Correction:** shift the code into the high bits of the wider type and then perform an arithmetic shift right, which extends the sign by construction rather than by a case analysis.
- **Prevention:** a unit test on the packing function using `-1` and the most negative code, both of which expose missing extension immediately.
- **Verification:** the two extreme codes must dequantize to their expected real values with the correct sign; a sign error at `-1` fails immediately.

### CI-17 — Stack grows into a static buffer

- **Category:** SCALE_FAILURE. The stack high-water mark is not bounded by design, and the region below it is occupied by static data rather than by a guard.
- **Symptom:** intermittent corruption of a feature buffer or a counter, appearing only under rare input combinations that deepen the call chain, and vanishing when the build changes.
- **Competing causes:** (a) genuine stack overflow; (b) a large automatic array in a rarely taken path; (c) a DMA target overlapping the stack region because of a linker script error.
- **Discriminating check:** paint the stack and read the high-water mark after an occurrence; compare the corrupted region's address with the stack bounds from the map. A high-water mark at the limit proves (a) or (b); a corrupted region outside the stack proves (c).
- **Correction:** move large buffers to static storage, bound recursion, and insert a guard region whose corruption raises a defined fault.
- **Prevention:** the map file is reviewed for adjacency between the stack and static regions at every release, and the high-water mark is published in telemetry.
- **Verification:** run the harness with the deepest input path and confirm the high-water mark stays below the declared limit with the guard untouched.

### CI-18 — Weights updated without their constants

- **Category:** VIOLATED_ASSUMPTION. The artifact is assumed to be the weights alone, while the normalisation constants and label map are assumed to be stable.
- **Symptom:** accuracy falls immediately after an update, and the previous release's golden vectors still pass because the harness pins the old artifact.
- **Competing causes:** (a) constants compiled into code and not updated; (b) label map stored separately and still in the old order; (c) an arena that grew and caused a tensor to be truncated silently.
- **Discriminating check:** hash weights, constants and labels at boot and compare each against the release manifest. A mismatch in one hash localises the defect to one artifact; three matching hashes eliminate the whole class.
- **Correction:** bind the three into one signed artifact with one hash, and refuse to run when the boot hash does not match the manifest.
- **Prevention:** the harness executes the shipped artifact rather than a reference copy, so an inconsistency fails before release.
- **Verification:** publish a deliberately mismatched constants file and confirm the device refuses to start inference and reports the mismatch.

### CI-19 — Transport acknowledgement treated as an action

- **Category:** CONCEPT_CONFUSION. A protocol-level acknowledgement is read as evidence that a physical action occurred.
- **Symptom:** a control loop believes an action succeeded, while the receiver's own gate rejected it; the discrepancy appears only when the plant is in a state that triggers rejection.
- **Competing causes:** (a) the acknowledgement's scope is misunderstood; (b) the receiver acknowledges on receipt and separately decides, and the second result is not transmitted; (c) the sender treats a timeout as a failure and a late acknowledgement as a success, so retries duplicate actions.
- **Discriminating check:** log the receiver's decision reason alongside the acknowledgement. An acknowledgement with a rejection reason proves (a) or (b); two actions for one command proves (c).
- **Correction:** require the receiver to return a decision record carrying the command identifier and its own reason code, and treat its absence as unknown rather than as success.
- **Prevention:** the protocol specification names the layer each message belongs to, and the sender's state machine has no transition from "acknowledged" to "done".
- **Verification:** force a rejection at the receiver and confirm the sender's state records "unknown" and raises an operator-visible event rather than reporting success.

### CI-20 — Sequence gap counted one too high

- **Category:** NUMERICAL_EDGE. The difference between consecutive sequence numbers is reported as the number of lost blocks, ignoring that the arriving record is one of them.
- **Symptom:** loss reports are systematically high by one per gap, which inflates a rare-event rate enough to change a release decision.
- **Competing causes:** (a) the off-by-one itself; (b) modular comparison implemented as a magnitude comparison, so a wrap reports a huge gap; (c) gaps counted at the host across a device reset, where the sequence legitimately restarts.
- **Discriminating check:** for a known injected loss of exactly three blocks, the report must read three. If it reads four, (a) is confirmed; if a single event reports a gap near the counter's range, (b) is confirmed; if gaps appear only at resets, (c) is confirmed.
- **Correction:** compute the gap as modular difference minus one, and compare epoch before comparing sequence, discarding cross-session gaps as resets.
- **Prevention:** a unit test with injected gaps of 1, 2 and 3 blocks, plus one across a wrap.
- **Verification:** the three injected gaps must report 1, 2 and 3 exactly, and a wrap test must report zero loss.

### CI-21 — Per-sample logging saturates the link

- **Category:** SCALE_FAILURE. Diagnostic output is added to the same link that carries the data, and its volume is not included in the utilisation budget.
- **Symptom:** overruns appear as soon as logging is enabled, the drop count correlates with log verbosity, and the defect disappears when the log is removed for a demo.
- **Competing causes:** (a) log volume exceeding the remaining link budget; (b) a blocking write that stalls the producer; (c) a non-blocking ring whose overflow policy discards data silently while counting it as a model fallback.
- **Discriminating check:** measure utilisation with and without logging, and compare the drop count against the log's byte rate. A drop rate proportional to log volume proves (a) or (b); drops that continue after logging is muted prove (c).
- **Correction:** move diagnostic output to a separate channel or a RAM ring dumped on demand, and keep the decision path's transport budget independent of the diagnostic budget.
- **Prevention:** the link budget table has a row per producer, and the logging row has a hard cap enforced by the logger.
- **Verification:** enable maximum verbosity and confirm the decision path's drop count remains zero for an hour.

### CI-22 — Watchdog refreshed by an interrupt

- **Category:** VIOLATED_ASSUMPTION. The refresh is assumed to prove that the work is progressing, which it does not when a higher-priority handler performs it.
- **Symptom:** the device resets under load with no evidence in the application log, and the reset cause reads as a watchdog event with no associated fault.
- **Competing causes:** (a) refresh from a timer interrupt; (b) refresh from a task that is not the one that must make progress; (c) a genuine application hang, in which case the watchdog is doing its job.
- **Discriminating check:** log the last sequence number before reset. A consistent sequence value proves (c) or (b) depending on which task last ran; a sequence value that continues to advance in the log while the reset occurs proves (a), because the log producer was alive.
- **Correction:** refresh only from the task that must make progress, and declare a maximum service time shorter than the window.
- **Prevention:** the refresh call is private to one module and cannot be called from an interrupt context, enforced by a context check.
- **Verification:** block the progress task deliberately and confirm a reset occurs within the window plus one service time.

### CI-23 — Tensor layout transposed

- **Category:** CONCEPT_CONFUSION. The producer writes `[sample][channel]` while the model expects `[channel][sample]`, and both shapes contain the same number of elements.
- **Symptom:** accuracy is a few points low with no error, and the confusion is concentrated where channels differ strongly, such as a single-axis impact.
- **Competing causes:** (a) genuine layout transposition; (b) the layout is right and the feature order differs, so means are assigned to standard deviations; (c) the model expects normalised inputs and receives raw.
- **Discriminating check:** drive one channel with a distinctive waveform and leave the others constant. If the feature vector shows the waveform in the wrong position, (a) or (b) is confirmed; then compare a mean computed by hand with the vector's first element to separate them.
- **Correction:** declare the layout in one header, produce and consume it through named accessors, and assert the element count and a layout tag at load.
- **Prevention:** a golden input whose channels are set to distinct constants, so a transposition changes the vector visibly.
- **Verification:** with channels set to 1, 2 and 3, the feature vector's first three elements must be 1, 2 and 3 in the declared order.

### CI-24 — Zero point outside the code range

- **Category:** NUMERICAL_EDGE. A zero point that does not lie inside `[q_min, q_max]` makes part of the representable interval unreachable and shifts every dequantized value.
- **Symptom:** outputs are offset by a constant, the offset is stable across inputs, and an accuracy ceiling appears that no amount of training removes.
- **Competing causes:** (a) zero point from a different tensor; (b) correct zero point but a `s` computed for a different code width; (c) a signed-to-unsigned conversion applied twice.
- **Discriminating check:** compute the representable range from `s` and `Z` and compare with the observed input distribution. A range that does not contain the data's mean isolates (a) or (b); a doubled conversion shows as a range exactly twice as wide as expected.
- **Correction:** validate at load that `q_min <= Z <= q_max` and that the range covers the declared input bounds, and refuse to run otherwise.
- **Prevention:** the manifest carries both range ends, and the loader checks them rather than trusting the exporter.
- **Verification:** feed a code equal to `Z` and confirm the dequantized output is exactly zero; a nonzero result proves the parameters are inconsistent.

### CI-25 — Overlapping windows treated as independent

- **Category:** SCALE_FAILURE. Evaluation confidence is computed from the record count while consecutive records share most of their samples.
- **Symptom:** small accuracy differences appear significant, model selection is unstable across reruns, and the chosen model does not reproduce its advantage in the field.
- **Competing causes:** (a) independence assumed across overlapping windows; (b) train and test split by window rather than by session, so near-duplicates appear on both sides; (c) normalisation statistics fitted on the full corpus before splitting.
- **Discriminating check:** re-split by session and re-evaluate without retraining. A large drop proves (b); a stable accuracy with a widened confidence interval proves (a); a drop only after recomputing statistics per fold proves (c).
- **Correction:** compute confidence from `d_eff = span/window`, split by session, and fit all preprocessing inside the training fold.
- **Prevention:** the evaluation script refuses to report a confidence interval when the split is by window without a stated correction.
- **Verification:** the reported half-width must grow by the computed inflation factor when the correction is enabled, and the significance of a null difference must disappear.

### CI-26 — Cache coherence assumed for a DMA region

- **Category:** VIOLATED_ASSUMPTION. A buffer written by DMA is treated as visible to the core without maintenance, which is false on any core with a data cache.
- **Symptom:** the design works on the microcontroller and fails on the larger part, with failures clustering after an idle period when stale lines are most likely to be read.
- **Competing causes:** (a) missing invalidation before reading; (b) missing clean after writing, so the DMA sends stale bytes; (c) maintenance performed by range on a partially shared line, discarding live core data.
- **Discriminating check:** mark the region non-cacheable and re-run. Disappearance of the symptom proves (a) or (b); a new corruption that appears only with maintenance enabled proves (c).
- **Correction:** declare DMA regions non-cacheable where the core supports it, or align buffers to line size and maintain by exact range with a barrier around the publication.
- **Prevention:** the linker script places DMA regions in a section with the correct attributes, and a boot self-test verifies a known pattern round-trips through the region.
- **Verification:** after every maintenance operation, a checksum computed by the core must equal one computed from the raw region; equality over a thousand blocks proves visibility.

### CI-27 — Timestamp taken at interrupt entry

- **Category:** CONCEPT_CONFUSION. The event time and the processing time are conflated, so every age is understated by the latency plus the trailing transfer.
- **Symptom:** the freshness gate passes decisions that are visibly stale, and the error grows with interrupt load rather than with the model.
- **Competing causes:** (a) the age origin is wrong; (b) the age is right but measured on a clock that also scales with load, such as a tick counter incremented by the same handler; (c) the comparison is made against a deadline that was itself derived from the model's window.
- **Discriminating check:** inject a known delay of 40 ms before the action and compute the reported age. A report near zero proves (a) or (b); a report that grows with the injected delay but is offset by a constant proves (a) specifically.
- **Correction:** timestamp the end of the last sample inside the handler using a free-running counter read as late as possible, and convert to seconds only at the point of comparison.
- **Prevention:** one timestamp type with unit and origin in its name, and a unit test asserting that an injected delay appears in the age with the correct magnitude.
- **Verification:** the reported age for a 40 ms delay must be between 40 and 41 ms; anything below 40 ms means the origin is wrong.

### CI-28 — Energy per block divided by the wrong interval

- **Category:** NUMERICAL_EDGE. An energy figure is divided by the active interval rather than by the period, or a per-second power is multiplied by a period that is not the one the energy was computed for.
- **Symptom:** predicted battery life is optimistic by the duty-cycle ratio, often by a factor between two and fifty.
- **Competing causes:** (a) interval confusion in the division; (b) energy computed for the active state only and divided by the period, which understates by the idle term; (c) cell capacity in mAh multiplied by voltage without the 3.6 factor for joules.
- **Discriminating check:** recompute the average power from an independent route — total charge over an hour, multiplied by voltage — and compare. Agreement eliminates (a); a discrepancy equal to the idle fraction proves (b); a factor of 3.6 proves (c).
- **Correction:** state energy per period explicitly in joules, divide once by the period to obtain watts, and convert capacity to joules before dividing.
- **Prevention:** a dimensional check in the spreadsheet or script: joules per second must appear as watts, and any result in joules per second multiplied by hours must be converted back to joules before the final division.
- **Verification:** the predicted life for a known measured current profile must agree with a bench discharge within ten percent.

### CI-29 — Debug-build timing generalised to release

- **Category:** VIOLATED_ASSUMPTION. A measurement from one build is treated as a property of the source rather than of the binary.
- **Symptom:** an acceptance run on a debug build passes and the shipped release misses the same limit, or a race reproduces only in one of the two.
- **Competing causes:** (a) instrumentation perturbs timing enough to hide the defect; (b) optimisation changed memory ordering and exposed a missing barrier; (c) the builds differ in configuration, not only in flags.
- **Discriminating check:** compare artifact hashes and configuration first, then run a minimally intrusive trace on both. A defect that disappears with heavier instrumentation proves (a); one that appears only at higher optimisation proves (b).
- **Correction:** designate one build configuration for acceptance evidence, record its instrumentation, and never generalise across builds.
- **Prevention:** the acceptance script prints the artifact hash and build flags in its report, so a claim is bound to a binary.
- **Verification:** the report must contain the hash of the binary that produced the timing, and a rebuild must invalidate the evidence until re-run.

### CI-30 — Wake settle time omitted from the ledger

- **Category:** SCALE_FAILURE. The sample count is computed from the active interval only, and the settle interval after each wake is not accounted for.
- **Symptom:** the sample count per hour is short by the wake fraction, the deficit scales with the number of wake cycles rather than with time, and each individual block looks correct.
- **Competing causes:** (a) settle time omitted from the schedule; (b) the sensor's settle time longer than the datasheet figure under the actual supply and temperature; (c) the DMA re-armed after the first conversion, losing the first block of each wake.
- **Discriminating check:** plot the deficit against wake-cycle count. Proportionality proves (a) or (c); a deficit that grows with temperature or with supply droop proves (b).
- **Correction:** include the settle interval in the schedule, arm conversions before the sensor settles, and count the deficit per wake cycle as a published counter.
- **Prevention:** the sample ledger is modelled as active plus settle plus sleep and the model is checked against a measured hour.
- **Verification:** the measured block count over an hour must agree with the model within the clock tolerance, and the per-wake deficit counter must read zero.

## 9. Complexity & Cost Analysis

### 9.1 Cost of each check in engineering time

| Check | Instrument needed | Time to run | Time to build |
|---|---|---|---|
| counter reconciliation | publish and consume counters | minutes | one day |
| golden vectors per layer | harness with intermediate taps | minutes | one week |
| injected delay for freshness | test hook on the action path | minutes | two days |
| jitter capture over ten thousand blocks | completion timestamp ring | minutes | two days |
| stack high-water paint | paint at boot, read at failure | minutes | one day |
| saturation sample count | per-sample clip counter | minutes | half a day |
| artifact hash publication | boot-time hash of weights, constants, labels | seconds | two days |

The asymmetry is the point: every check costs minutes to run and days to build once, and the catalogue is only usable when the instruments exist before the defect does. A team that builds instruments after a field failure spends the same days with a customer waiting.

### 9.2 Cost of a misdiagnosis

| Misdiagnosis | Typical action | Cost | Actual defect often |
|---|---|---|---|
| "the model is weak" | retrain, re-quantize | weeks of ML time | timing or range defect |
| "the link is saturated" | raise baud, add compression | days, new failure modes | parser defect or duplication |
| "the sensor is noisy" | add filtering | days, new latency | sign extension or layout |
| "the device is unstable" | add resets and retries | days, masks evidence | watchdog refreshed by an ISR |
| "the sample rate is fine" | none | nothing | decimation or ODR change |

The table's pattern is that every misdiagnosis adds a mechanism rather than removing a defect. Each added mechanism then becomes a source of new issues, which is why the catalogue insists on a discriminating check before a correction.

### 9.3 Memory cost of the instrumentation itself

```text
publish counter            4 byte
consume counter            4 byte
drop counters by policy   12 byte
ring high-water            4 byte
jitter min/max pair        8 byte
saturation sample count    4 byte
last sequence before reset 4 byte
reset cause                2 byte
artifact hash             16 byte
total                     58 byte
```

Fifty-eight bytes of counters would have made eight of the thirty issues in this catalogue diagnosable in the field rather than reproducible only on a bench. Against a 96 KiB SRAM budget that is 0.06 percent, and against the cost of a single misdiagnosis it is free.

### 9.4 Cost of running long enough

```text
test duration for a 1-in-50,000 defect at 95 percent confidence, k=3  = 3/0.00002
                                                                    = 150,000 blocks
at 50 block/s                                                       = 3,000 s = 50 min
for a 1-in-1,000,000 defect, k=3                                     = 3,000,000 blocks
at 50 block/s                                                       = 60,000 s = 16.7 h
```

Two conclusions follow from the arithmetic. First, a one-minute smoke test cannot detect anything rarer than about 1 in 3,000 blocks, so a "clean" one-minute run is evidence of nothing. Second, rare-defect claims need continuous capture with a ring that can be dumped after the fact, because sixteen hours of runtime will not be repeated on demand.

### 9.5 Cost of an incomplete state inventory

```text
two-state model, 55.6 h predicted from 220 mAh at 14.65 mW
observed 30 h -> implied average power = 2930.4/108000 = 0.02713 W = 27.1 mW
missing power = 27.1 - 14.65 = 12.5 mW
```

The 12.5 mW gap is the cost of an incomplete inventory: a regulator quiescent draw, a sensor that never sleeps, or a wake transition whose energy was attributed to nothing. The arithmetic does not say which, but it bounds the search to components whose average draw sums to 12.5 mW, which is a much shorter list than the bill of materials.

### 9.6 Cost of duplicate decision paths

Two decision paths, local and remote, mean two label maps, two thresholds and two artifact hashes. The reconciliation cost is a shared labelled set evaluated through both paths:

```text
set size for 95 percent confidence on a 2-point difference in a 3-class task
  n per class ~ 16 * p*(1-p)/(delta^2) with p=0.33, delta=0.02
            ~ 16 * 0.2211/0.0004 = 8,844 per class -> 26,532 windows
at 50 window/s of continuous data that is 531 s of labelled recording
```

A five-hundred-second recording per reconciliation is cheap compared with the alternative, which is discovering the disagreement in the field as a class that one path never emits.

## 10. Key Theorems / Results Reference

### 10.1 Little's law as a ledger check

For a stable system, mean occupancy equals arrival rate times mean time in system. It is distribution-free and holds for any work-conserving service, which makes it the correct first check on a ring: if measured mean occupancy differs from `lambda*W`, the system is not in the steady state that the design assumed, and every downstream latency claim is void.

The law says nothing about the tail. A system can satisfy Little's law exactly and still miss a deadline on one percent of blocks, which is why occupancy and distribution are separate instruments.

### 10.2 Stability of a single-server queue

With `rho >= 1` the queue is unstable, and no finite buffer changes that. Buffer depth converts instability into a delay in observing it: with depth `B_tok` and deficit `lambda-mu`, saturation occurs at `B_tok/(lambda-mu)`, so doubling the depth doubles the time before the symptom appears without changing the outcome.

This is the theorem that refutes the most common fix in embedded practice, which is to add buffer.

### 10.3 Nyquist–Shannon and its practical inversion

Sampling above twice the highest frequency of interest is sufficient for reconstruction of a band-limited signal. In a fault catalogue the useful form is the inverse: any energy above half the sample rate folds into the band and cannot be removed, so a component at `f_s - delta` appears at `delta` and is indistinguishable from a genuine low-frequency signal.

### 10.4 Rate-monotonic utilisation bound

For `n` independent periodic tasks under rate-monotonic assignment, utilisation at or below `n*(2^(1/n)-1)` guarantees all deadlines. The bound is sufficient, not necessary, and it ignores blocking and interrupt load. Its value in this catalogue is negative: a passing bound is not evidence that a device meets a deadline, and CI-10 exists because that inference is made constantly.

### 10.5 Quantization error and its preconditions

A uniform quantizer with step `s` has maximum error `s/2` and variance `s^2/12`, provided the input never overloads and is large compared with the step. For a full-scale sinusoid the signal-to-noise ratio is `6.02*b_q + 1.76` dB, giving 49.9 dB at 8 bits, 74.0 dB at 12 bits and 98.1 dB at 16 bits.

Every one of those figures assumes no overload. Once a value clamps, the error is bounded by the distance to the code limit, which can be twenty times the half-step, and that is the arithmetic behind CI-06 and CI-08.

### 10.6 Modular comparison and the half-range rule

For an `n`-bit counter, `(uint)(a-b) < 2^(n-1)` is the correct test for "a is newer than b", and it is valid for any interval shorter than half the counter's range. At 50 blocks per second a 32-bit counter wraps in 2.72 years, so the rule is never the binding constraint — but a magnitude comparison instead of a modular one fails completely at the first wrap.

### 10.7 Amdahl's argument applied to triage

If a fraction `p` of the total path is improved by a factor `k`, the path becomes `(1-p) + p/k` of its original duration. In a 2.03 ms decision path where inference is 28.4 us, `p = 0.014`, so an infinitely fast model removes 1.4 percent. The corollary for a fault catalogue is that a defect occupying 98.6 percent of the path will be misattributed to the 1.4 percent that is easiest to change.

### 10.8 WCET and the meaning of a bound

A worst-case execution time is a claim with a method attached. Measurement-based bounds are optimistic when the worst path was not observed; static bounds are pessimistic by a factor that grows with core complexity. The catalogue's rule is that a bound quoted without its method is not evidence and cannot be used to close an issue.

### 10.9 Overflow bounds as a design constant

For a sum of `N` terms each bounded by `T_max`, the accumulator must hold `N*T_max`. For a sum of squares it must hold `N*T_max^2`. With `N=512` and `T_max=32767`, the first is `1.68e7` and the second is `5.50e11`, a ratio of `T_max` — which is why the two accumulators have different widths and why widening "the accumulator" without deciding which one is a guess rather than a fix.

### 10.10 Energy as a sum over states

Average power is the sum of `V*I*t` over an exhaustive state set divided by the period. The theorem's content is the word exhaustive: an omitted state contributes a positive error that no amount of measurement precision on the included states can detect, which is why CI-28's discriminating check recomputes the total from an independent route.

### 10.11 Detection limits of a counter

A rate estimated from `k` observed events in `n` trials has a relative uncertainty near `1/sqrt(k)`, so three events give roughly 58 percent relative uncertainty. This is why a release criterion phrased as "no drops observed" is weaker than it looks: observing zero events in `n` trials bounds the rate only at about `3/n`, and the bound must be stated with the trial count to mean anything.

### 10.12 Evidence boundaries for this catalogue

| Claim type | Supports | Does not support |
|---|---|---|
| hand computation | arithmetic consistency of a check | behaviour of any device |
| counter design | observability of a mechanism | that the mechanism is absent |
| injected-fault test | that a gate is on the path | that the gate's bound is correct |
| long-run capture | a rate with a stated uncertainty | rarity beyond the capture's resolution |

## 11. Interview / Exam Questions

### Q01. A host reports that accuracy fell 0.4 points after a firmware update, from a 60-second capture of 512-sample windows advancing 32 samples at a time. What is the first quantitative objection?

**Answer:** The windows overlap by 93.75 percent, so 3,000 records correspond to 187 effective independent samples, and the confidence half-width is about 4.0 times the naive value, or 4.3 points. A 0.4-point change is about 9 percent of that half-width and carries no information.

**Reasoning:** `d_eff = 60/0.320 = 187.5`, and `sqrt(3000/187.5) = 4.0`. The naive half-width at `p=0.9` is `1.96*sqrt(0.09/3000) = 1.07` points, so the corrected figure is 4.3 points, and the observation sits inside it.

**Discriminating check:** re-split by recording session rather than by window and re-evaluate. A real regression survives a session split; a noise-level difference does not.

**Red flag:** accepting the 0.4-point change and beginning a retraining effort.

**Scoring:** 3 points: the effective count, the inflation factor with arithmetic, and the conclusion about the claim.

### Q02. A device publishes 179,000 blocks in an hour where 180,000 are expected. Give the ppm figure and two mechanisms, with the experiment that separates them.

**Answer:** The deficit is 5,556 ppm. Clock tolerance on an uncalibrated oscillator is one mechanism; whole-block loss is the other. Duration and shape separate them: tolerance scales smoothly and linearly, while block loss appears as irregular lumps.

**Reasoning:** `1000/180000 = 0.005556`. A per-sample leak cannot produce that magnitude — losing one sample in three thousand would be 333 ppm — so only scale error or whole-block loss remain.

**Discriminating check:** measure the interval against a reference clock at the start and end of the hour, and log sequence gaps with timestamps rather than a count.

**Red flag:** attributing the deficit to transport without checking whether the arithmetic can produce it.

**Scoring:** 3 points: the ppm conversion, two named mechanisms, and a duration-based discriminator.

### Q03. Standard deviation is computed from 32-bit `sum` and `sum_sq` over 512 samples of 16-bit data. What fails, and what is the minimum fix?

**Answer:** `sum_sq` overflows: the maximum is `512*32767^2 = 5.497e11` against a signed 32-bit limit of `2.147e9`, a factor of 256. The variance becomes negative and the square root produces NaN or garbage. The minimum fix is a 64-bit accumulator for the square sum only.

**Reasoning:** The plain sum reaches `512*32767 = 1.68e7` and fits, so widening both wastes cycles and obscures which accumulator was at risk.

**Discriminating check:** recompute the variance on the host in 64-bit arithmetic from the same raw samples; a stable host value with a negative device value proves overflow.

**Red flag:** casting to float only at the square root, which cannot recover the already-corrupted sum.

**Scoring:** 3 points: the overflow arithmetic, the negative-variance symptom, and the narrow fix.

### Q04. A three-store rotation with 20 ms fills and 3 ms peak jitter runs a consumer that needs 30 ms. Is it safe, and what happens at 5 ms jitter?

**Answer:** Exclusivity is `2*20 - 2*3 = 34 ms`, so it is safe with 4 ms of margin. At 5 ms jitter the bound is `40 - 10 = 30 ms`, equal to the service time, which is a failure because no margin remains for queueing.

**Reasoning:** Jitter is subtracted from the reader's window because early arrivals shorten it; late arrivals merely delay the next claim and do not help.

**Discriminating check:** capture ten thousand completion intervals and read the minimum pair sum; compare it against the worst-case service time, not the mean.

**Red flag:** using `(K-1)*T_w` without the jitter term, or treating equality as a pass.

**Scoring:** 3 points: the worst-case subtraction, the equality-is-failure point, and a quantified remedy.

### Q05. A design has 8 blocks of admission budget, publishes at 50 blocks per second and consumes at 42. When does shedding begin, and which policy would you choose for a windowed classifier?

**Answer:** Shedding begins at `8/(50-42) = 1.0 s`. For a windowed classifier that averages, drop-oldest is defensible because recency matters more than gap-free history; a consumer that rejects gaps needs drop-newest and an explicit loss report.

**Reasoning:** `rho = 50/42 = 1.19`, so the backlog grows at 8 blocks per second. The third option, decimating the decision rate, halves the load to 21 blocks per second and removes the overload without extra memory.

**Discriminating check:** plot ring occupancy against time; a straight line to the depth followed by a plateau proves sustained overload rather than a burst.

**Red flag:** adding depth, which merely moves the failure later.

**Scoring:** 3 points: the saturation time, a policy justified by the consumer's tolerance, and the decimation option.

### Q06. Reported action age stays near zero even when the transport is delayed by 200 ms. What are the two most likely mechanisms, and how do you separate them?

**Answer:** Either the age is computed from two clocks that move together, such as two queue-relative timestamps, or the gate sits on the producer path and never sees the delay. Injecting the delay between decision and action and observing no rejection proves the second; observing rejection counts that are computed and ignored proves an implementation variant of the first.

**Reasoning:** A correct gate compares an absolute end-of-sample timestamp with an absolute local clock at the point of use, so a 200 ms transport delay must appear in full.

**Discriminating check:** log both raw timestamps for one event and subtract by hand; if the difference is correct in isolation but the gate passes, the gate is not on the path.

**Red flag:** concluding that the transport is not delayed because the reported age is small.

**Scoring:** 3 points: two mechanisms, the injection test, and the hand subtraction.

### Q07. What is wrong with an acceptance report that says "mean decision latency 12 ms against a 20 ms limit"?

**Answer:** The limit applies to every instance and the statistic applies to the middle of the distribution. The report needs the fraction above the bound and the maximum, from a capture long enough to contain at least a hundred deadline events.

**Reasoning:** A mean of 12 ms with a 6 ms standard deviation implies roughly 9 percent above 20 ms under a normal approximation, which is a large operational defect hidden by a passing number.

**Discriminating check:** recompute the fraction above the bound from the same capture; if it is nonzero, the acceptance criterion is wrong rather than the system.

**Red flag:** raising the clock before computing the fraction, which changes the mean without addressing the tail's cause.

**Scoring:** 3 points: statistic mismatch, the recomputed fraction, and a corrected criterion.

### Q08. A filter shows a spectral line at exactly the block rate. Distinguish a state reset from an off-by-one overlap.

**Answer:** A reset injects a transient whose period is the block period, so the line appears at `1/T_w` with harmonics and its amplitude is set by the filter's step response. An off-by-one overlap produces a small phase discontinuity with a much weaker line and a measurable phase error between concatenated and single-shot outputs.

**Reasoning:** Both mechanisms are boundary artefacts, so the frequency alone does not separate them; the amplitude relative to the tone, and the phase comparison, do.

**Discriminating check:** concatenate block outputs and compare with a single-shot run over the same samples. An exact boundary error of one sample is visible as a fixed phase offset; a reset is visible as a decaying transient at every boundary.

**Red flag:** adding a windowing function, which changes both mechanisms and hides the distinction.

**Scoring:** 3 points: mechanism-specific signatures, the concatenation test, and a stated discriminator.

### Q09. An int8 quantizer has `s = 0.05` and `Z = -3`. Give the representable range and the error for an input of 7.0.

**Answer:** The range is `[-6.25, 6.50]`. An input of 7.0 clamps to 6.50, an error of 0.5 real units, or 10 LSB — twenty times the half-step bound.

**Reasoning:** `(-128+3)*0.05 = -6.25` and `(127+3)*0.05 = 6.50`. The half-step bound of 0.025 units applies only while no value overloads.

**Discriminating check:** publish the per-block maximum raw code and a per-sample clip count; a maximum at the code limit with clips proves overload rather than rounding.

**Red flag:** quoting 0.025 units as the error without checking the input range.

**Scoring:** 3 points: both range ends, the clamp error, and the violated precondition.

### Q10. Requantization uses `s_in = 0.02`, `s_w = 0.05`, `s_out = 0.4` with a Q30 multiplier. Compute the multiplier and identify the overflow risk.

**Answer:** `M = 0.001/0.4 = 0.0025`, so `M0 = round(0.0025*2^30) = 2,684,355`. The product with a 32-bit accumulator needs up to 63 bits, so a 32-bit implementation wraps.

**Reasoning:** For `acc = 12345`, the product is `33,138,362,475`, which needs 35 bits, and the shifted result is 30 codes against an exact 30.8625 — a sub-LSB error on this input but a full wrap on larger ones.

**Discriminating check:** compare the device's intermediate tensor with a 64-bit host reference at three magnitudes; errors that appear only above a threshold prove the wrap.

**Red flag:** increasing the shift to avoid the overflow, which changes the scale by a power of two.

**Scoring:** 3 points: the multiplier, the bit-width statement, and the magnitude-dependent signature.

### Q11. Why does a 192-byte DMA block behave differently on a cached core, and what is the cheapest experiment?

**Answer:** The block spans three 64-byte lines only if line-aligned; otherwise it straddles four, and the handoff requires maintenance on a partially shared line while DMA bypasses the cache entirely. The cheapest experiment is to map the region non-cacheable and re-run.

**Reasoning:** If the symptom disappears, the mechanism is coherence; if it persists, the mechanism is ordering and the fix is a barrier rather than wider maintenance.

**Discriminating check:** compare a core-computed checksum with one computed from the raw region after maintenance; repeated agreement over a thousand blocks proves visibility.

**Red flag:** adding a full cache flush around every handoff, which is correct and far too slow to keep the deadline.

**Scoring:** 3 points: line alignment, bypass semantics, and the cheaper experiment first.

### Q12. A sequence jump runs from 100 to 105. How many blocks were lost, and what does the wrong answer imply?

**Answer:** Four blocks. Reporting five overstates loss by 25 percent at that magnitude, which is enough to send an investigation after a transport defect that does not exist.

**Reasoning:** The difference is five and one of the five is the record that arrived, so lost equals difference minus one. Modular arithmetic must be used, and epoch compared first so a reset is not read as a gap.

**Discriminating check:** inject a known loss of three blocks and confirm the report reads three, not four.

**Red flag:** comparing sequence magnitudes rather than differences, which reports a near-full-range gap at every wrap.

**Scoring:** 2 points: the count and the consequence for a loss-rate claim.

### Q13. An application log shows no error, but the device resets under load. Give the check that identifies a watchdog fed by an interrupt.

**Answer:** Log the last sequence number before reset. If the sequence continues to advance in the log while resets occur, the log producer was alive and the refresh came from a context that does not prove progress; if the last sequence is identical every time, the stall is in the progress path.

**Reasoning:** A watchdog refreshed by a timer interrupt monitors the timer, not the work. The distinguishing evidence is whether unrelated activity continued during the stall.

**Discriminating check:** block the progress task deliberately and confirm a reset occurs within the window plus one service time; then confirm the same test does not reset when the refresh is moved to the task.

**Red flag:** increasing the watchdog window, which converts a reset into a longer hang.

**Scoring:** 3 points: the evidence, the controlled test, and the reason the window is not the fix.

### Q14. Battery life is predicted at 55.6 hours and observed at 30. Compute the implied missing power and name the likely components.

**Answer:** Stored energy is `0.220*3600*3.7 = 2,930.4 J`. Observed life implies `2930.4/108000 = 27.1 mW` against a predicted 14.65 mW, so about 12.5 mW is unaccounted for.

**Reasoning:** The prediction's arithmetic is internally consistent, so a factor-of-two error indicates an omitted state rather than a division error. Candidates are regulator quiescent draw, a sensor that never sleeps, and wake-transition energy.

**Discriminating check:** measure the idle current with the sensor powered down and with it enabled; the difference isolates the sensor, and the residual isolates the regulator.

**Red flag:** halving the assumed idle current until the prediction matches.

**Scoring:** 3 points: the stored-energy arithmetic, the implied power, and a bounded component list.

### Q15. Why is "no drops observed" weak evidence, and how should the claim be stated?

**Answer:** Observing zero events in `n` trials bounds the rate only at roughly `3/n`, so the claim must carry the trial count. Zero drops in 3,000 blocks bounds the rate at about one in a thousand, which is three orders of magnitude weaker than a field requirement of one in a million.

**Reasoning:** The bound comes from the Poisson upper limit at 95 percent confidence for zero observed events, which is near three. Without the trial count the sentence has no content.

**Discriminating check:** compute `3/n` for the actual capture and compare with the requirement; if the bound exceeds the requirement, the test cannot support the release.

**Red flag:** repeating the same short test on more units and adding the counts as if the trials were independent of the conditions.

**Scoring:** 3 points: the bound, the trial-count requirement, and the comparison with the requirement.

### Q16. A saturation counter reads one event while accuracy collapses on loud windows. What is wrong, and what should be published instead?

**Answer:** The counter increments per block rather than per sample, so a block with 500 of 512 samples clamped reports a single event. It should publish a per-sample clip count and the per-block maximum raw code.

**Reasoning:** A per-block boolean hides magnitude, which is the variable that correlates with the accuracy loss. The maximum code is the cheapest magnitude proxy and costs four bytes.

**Discriminating check:** log the maximum absolute raw code per block; a maximum at the code limit with a counter value of one and a 512-sample window proves the aggregation error.

**Red flag:** treating the counter as an alarm threshold without asking what it counts.

**Scoring:** 3 points: the aggregation error, the two replacement quantities, and the reason magnitude matters.

### 11.1 Answer standard

| Criterion | Strong answer | Weak but repairable | Reject signal |
|---|---|---|---|
| arithmetic | shows the division and the units | correct result, no working | quotes a figure with no derivation |
| discrimination | names the observation that separates causes | lists causes without a test | picks the most familiar cause |
| scope | states what the evidence supports | says "tested and works" | generalises across builds |
| correction | changes the contract that was violated | adds a buffer or a retry | raises a threshold |
| residual | names what remains unverified | silent | claims closure |

An examiner should accept a different architecture when it preserves the reasoning chain. The question to ask a candidate who proposes a fix is which counter would have caught the defect before the field report, because that answer distinguishes a repair from a prevention.

**Source basis:** companion cheat sheet 01 sections 3 through 5 for the vocabulary and derivations, and the unit index for the topic map. **Evidence boundary:** all arithmetic is hand-computed and ILLUSTRATIVE; no target build, board test, power measurement, or field data was executed or observed for this catalogue.

## 12. Quick Reference Card

| Symptom | First check | Likely issue | Fix that changes the contract |
|---|---|---|---|
| counts drift smoothly | reference-clock comparison | CI-15, CI-30 | derive rates from read-back configuration |
| counts gap in lumps | device versus host ledger | CI-09, CI-20, CI-26 | declare a shedding policy and count it |
| labels wrong at extremes | clip count and maximum code | CI-06, CI-08, CI-24 | widen calibration range and publish it |
| labels wrong everywhere | golden vectors per layer | CI-07, CI-23 | bind layout and parameters to one struct |
| actions late, no rejections | injected delay at the action | CI-11, CI-27 | move the gate to the point of use |
| timing passes in debug only | artifact hash and map delta | CI-05, CI-29 | one build configuration for evidence |
| memory corrupts under load | stack high-water paint | CI-01, CI-17, CI-26 | guard region plus non-cacheable mapping |
| link overruns with logging | utilisation with and without | CI-03, CI-21 | separate diagnostic budget with a cap |
| device resets under load | reset cause and last sequence | CI-22, CI-28 | refresh the watchdog from the task |
| life below prediction | independent power route | CI-28, CI-30 | exhaustive state inventory in joules |
