---
case_id: "CASE-M35-01"
case_title: "Wearable activity classifier: choose a local classification and alert boundary"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Embedded TinyML — wearable inertial activity classification"
primary_pattern: "bounded acquisition"
secondary_patterns: ["freshness gate", "degraded local behavior"]
evidence_level: "illustrative"
source_basis: "illustrative"
source_status: "OPEN — source inventory names firmware topics; no participant recordings, device datasheet, trained artifact, benchmark, or clinical evidence was supplied"
as_of_date: "2026-11-10"
owner_role: "wearable product manager (decision owner); embedded lead (implementation owner)"
source_index: "INDEX.md"
planned_from: "PLAN-M35-01"
review_trigger: "Change to sensor rate/range, device memory, model or quantization artifact, label protocol, radio policy, battery, alert purpose, or participant population"
---

# Case study: Wearable activity classifier

## 1. The One-Line Framing

Given **one timestamped tri-axial accelerometer window from a consenting participant**, decide **whether to display an on-device activity label or `UNAVAILABLE`** for the **wearable product manager and wearer**, to provide immediate descriptive feedback while preserving a bounded battery, RAM, flash, and freshness budget, compared with **fixed-rate raw upload for later manual review or a simple magnitude threshold**.

This is a predictive classification decision at one result every `T_h=1.0 s`; it is not diagnosis, fall detection, a safety alarm, or a claim about health, impairment, calories, adherence, or clinical state.

**Evidence declaration.** This is an illustrative design exercise. Participant counts, model size, latency, currents, battery capacity, and all numerical outputs are assumptions selected for auditable arithmetic, not measured device results, target benchmarks, or field outcomes.

The primary pattern is **bounded acquisition**: immutable sensor blocks carry sequence and end-time metadata from DMA completion to inference. The secondary patterns are a **freshness gate**, which expires late work, and **degraded local behavior**, which shows no activity label when integrity cannot be established.

The simpler framing, “classify the latest samples,” is unsafe as an engineering statement. A window can be statistically plausible yet belong to the wrong person, overlap a training window, have been overwritten by DMA, be quantized with the wrong scale, or be too old to represent the displayed moment.

## 2. Context & Background

A hypothetical product team is considering a wrist-worn activity-feedback feature for adults who explicitly opt in to local inertial sensing. The wearer sees one of `STILL`, `WALK`, `STAIRS`, or `OTHER`; the device may retain a bounded diagnostic counter but does not upload raw motion by default.

The existing baseline is fixed-rate upload of raw accelerometer data when a companion phone is available, followed by manual or batch review. It can preserve data for later analysis, but it has radio-energy cost, connection delay, privacy exposure, and no immediate local label.

A simpler alternative is a magnitude threshold: report `WALK` when the window’s acceleration magnitude variance exceeds a fixed threshold, otherwise report `STILL`. That alternative is cheap and inspectable, but it cannot reliably separate stairs, wrist gestures, vehicle motion, or sensor placement changes without an explicitly validated rule set.

The product manager owns whether to fund an on-device classifier. The embedded lead owns firmware integrity. A data lead owns label protocol and split design. A human-factors reviewer owns whether the label wording implies more than an activity description.

The workflow is deliberately narrow. The sensor samples motion; firmware constructs a window; a local model emits scores; a threshold and freshness gate choose a label or `UNAVAILABLE`; the UI renders that result with a capture interval. No result controls a therapy, emergency workflow, access decision, insurance decision, or staff performance process.

A participant can start, stop, climb steps, ride in a vehicle, remove the device, or wear it loosely. These ordinary changes are sources of distribution shift, not mistakes by the participant.

The initial illustrative scope is 24 participants, two sessions per participant, and four scripted activities plus a free-living `OTHER` collection. It is a planning population, not evidence that a model generalizes to age, gait, disability, placement, culture, occupation, or any clinical population.

The source inventory for Module 35 names ADC/DMA, RTOS, CMSIS-DSP/NN, UART, power modes, linker layout, and debugging topics. Its bodies were unavailable. This case therefore specifies portable contracts rather than MCU register settings, device-specific currents, or a vendor benchmark.

## 3. The Business Question

The stakeholder question is illustrative and paraphrased: “Should this feature classify activity locally every second, or should we keep raw upload/manual review or a transparent threshold until we have evidence that the local path is feasible and honest?”

| Decision contract | Bounded definition |
|---|---|
| Question | Select the local classification boundary, or retain a baseline, for one wrist IMU stream. |
| Decision owner | Wearable product manager; embedded lead can reject an infeasible memory/timing design. |
| Trigger and deadline | Feature planning gate; classification result must be fresh when rendered. |
| Available actions | Keep upload/manual review; use magnitude threshold; prototype INT8 local classifier; defer feature. |
| Value | Immediate descriptive feedback and less raw-radio duty if local results are valid. |
| Downside | Misleading labels, stale output, battery loss, privacy expansion, and firmware data corruption. |
| Objective | Maximize valid, fresh descriptive labels per eligible window, not apparent accuracy alone. |
| Hard constraints | No diagnosis claim; no output after integrity/freshness failure; no buffer overwrite; static memory budget. |
| Scope | One accelerometer, four display classes, opt-in adult research protocol, local display. |
| Out of scope | Clinical detection, fall response, personalization, cloud training, and commercial deployment. |

The decision metric cannot be plain accuracy because windows cluster within a participant and session. The primary proposed metric is macro recall among eligible, fresh holdout windows, reported by participant and session; an ineligible window is a correct abstention only if the UI shows `UNAVAILABLE`, not the previous label.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence / check | Status | Impact if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Sensor delivers 3 axes at 50 samples/s/axis | workload | timestamp trace | illustrative | all buffer math changes | recompute sizes | embedded lead |
| A-02 | Samples are signed 16-bit values | representation | sensor contract | illustrative | byte budget changes | version schema | embedded lead |
| A-03 | A window is 2.56 s and hop is 1.00 s | design | protocol review | proposed | overlap/leakage and age change | change split and budget | data lead |
| A-04 | DMA has three blocks | memory | ownership trace | proposed | reuse margin changes | add block or reduce work | embedded lead |
| A-05 | Local model has 18 KiB parameters | resource | build/map file | unmeasured | flash headroom invalid | choose smaller model | embedded lead |
| A-06 | Arena peak is 22 KiB | resource | target allocator report | unmeasured | SRAM collision | resize or redesign | embedded lead |
| A-07 | Inference plus preprocessing takes 24 ms | timing | target trace | assumption | deadline may fail | expire result | embedded lead |
| A-08 | Radio summary costs 18 mA for 40 ms per minute | energy | current trace | assumption | battery estimate invalid | lower radio duty | product manager |
| A-09 | Battery usable capacity is 120 mAh | energy | discharge test | assumption | runtime estimate invalid | state capacity range | product manager |
| A-10 | Labels are prompted and adjudicated | data | label audit | proposed | target is ambiguous | collect protocol evidence | data lead |
| C-01 | Display age may not exceed 1.35 s after window end | freshness | monotonic clock trace | proposed | stale UI | show unavailable | embedded lead |
| C-02 | No sample block is read while DMA writes it | integrity | sequence/state trace | required | corrupted model input | count overrun and abstain | embedded lead |
| C-03 | No training, validation, and test split shares a participant | leakage | split manifest | required | optimistic performance | rebuild split | data lead |
| C-04 | Overlapping windows may not cross split boundaries | leakage | interval audit | required | label leakage | group by session | data lead |
| C-05 | Activity wording cannot be diagnostic | product | copy review | required | unsafe interpretation | remove label | product manager |

The reverse-design assumptions are A-03, A-06, and A-07. A larger hop changes both energy and freshness; a larger arena can consume all SRAM; a slower target path can preserve bytes while producing an expired answer.

## 5. Data, Tools & Method

### 5a. Data Specification

The training grain is one labeled time interval, not one window. Windows are derived records carrying `participant_id`, `session_id`, `activity_interval_id`, start/end sample sequence, start/end monotonic time, sensor-placement note, and preprocessing/model revision.

The runtime grain is one candidate window of `N=128` tri-axial sample frames. Each frame has three signed 16-bit readings and one implicit position in the block; block metadata supplies the sequence and end time. The model sees a tensor shaped `[128,3]`, ordered oldest to newest and X/Y/Z within each frame.

| Field | Definition |
|---|---|
| Population | Consent-based illustrative research participants; no clinical inference. |
| Session | One contiguous recording under a recorded placement and protocol condition. |
| Activity label | Prompted instructed activity, then participant confirmation and reviewer adjudication. |
| Label interval | Start/end timestamps, transition exclusion zones, adjudicator ID, and rationale. |
| Eligible window | Entire interval is one final label; no gap, duplicate, overrun, or quality violation. |
| Excluded window | Transition, device removal, missing label, timestamp gap, overflow, or ambiguous label. |
| Features | Raw scaled accelerometer axes; learned normalization fitted only on training participants. |
| Output | Four score values and an abstention reason; scores are not probabilities unless calibrated. |

**Label protocol.** A facilitator gives a scripted activity instruction, a visible session clock marks instruction changes, and the participant confirms the performed activity after each segment. Two reviewers inspect ambiguous boundaries against a session log. A label begins only after a declared settling interval and ends before the next transition interval. `OTHER` is not a waste basket for unknown quality; it is a deliberate non-target activity class with its own collection rule.

**Participant/session/time split.** Split first by participant: 16 participants for training, 4 for validation, and 4 for final test. Keep both sessions for a participant in the same partition. Within each participant session, preserve time order for diagnostics but do not claim that a random window split estimates performance on a new wearer. Reserve the final participant group before hyperparameter choice.

**Overlap leakage.** With a 2.56 s window and 1.00 s hop, adjacent candidate windows overlap by 1.56 s, or 60.94% of samples. A random split can put nearly identical signal samples into train and test. The rule is stronger than “do not duplicate rows”: all windows originating from a participant stay in that participant partition, and any time-based within-participant experiment must purge a guard interval of at least `T_w` on both sides of a boundary.

### 5b. Tools & Stack

| Component | Proposed role | Required evidence |
|---|---|---|
| Accelerometer + timer | Produces 50 Hz tri-axial frames | rate/timestamp trace and range configuration |
| DMA circular buffers | Moves samples to immutable completed blocks | ownership and sequence trace |
| Fixed-point preprocessing | Converts sensor counts to INT8 tensor values | scale, zero point, saturation count |
| INT8 runtime | Executes supported local model | model hash, operator list, arena peak |
| Static linker map | Accounts for flash/SRAM sections | map file and stack watermark |
| Monotonic clock | Calculates window end and age | wrap/epoch policy |
| UI state machine | Displays label or `UNAVAILABLE` | stale-output test trace |
| Optional radio | Sends coarse summary, never raw by default | duty policy and payload definition |

No named runtime, MCU, or device configuration is asserted as used. The stack is a proposed interface contract. A target build must pin compiler, runtime, model converter, quantization metadata, and model hash before any feasibility claim.

### 5c. Method Summary

The proposed path is `DMA-completed frames -> sequence/quality check -> fixed-point scale -> INT8 model -> score margin/quality gate -> freshness gate -> display or UNAVAILABLE`.

| Candidate | Evidence considered | Strength | Limitation | Status |
|---|---|---|---|---|
| Upload/manual review | conceptual baseline | preserves raw context | radio/privacy/latency; no immediate result | not measured |
| Magnitude threshold | specified comparator | tiny compute, explainable | weak class separation and placement sensitivity | not measured |
| INT8 local classifier | resource model | can represent multi-axis temporal pattern | needs split, quantization, and target evidence | proposed |
| No feature | honest option | no misleading result | no feedback | retained fallback |

The threshold baseline uses the mean squared acceleration magnitude deviation in the window. It outputs `WALK` only when that statistic exceeds a fixed training-derived threshold and otherwise `STILL`. It deliberately cannot emit `STAIRS`; its point is to make the local model earn its complexity on the same held-out people and the same abstention policy.

The model choice is intentionally unspecified beyond INT8-compatible input/output and static arena. Architecture selection must compare parameter bytes, peak activations, class-wise holdout performance, quantization drift, and target timing. Selecting a fashionable architecture before these checks is not a method.

### 5d. Data provenance, quality, exclusions and leakage controls

`SRC-01` is the unavailable Module 35 source inventory; it supports topic planning only. `DATA-01` is proposed consented research collection, unavailable at authoring. `MODEL-01` is a future model artifact, unavailable at authoring. No observed recordings, model result, or device trace exists in this case.

Quality gates reject timestamp gaps greater than one sample period plus declared jitter, sequence discontinuity, sensor-range clipping above a threshold chosen before test inspection, device-removal flag, incomplete block, and a label interval that crosses a transition exclusion.

The split manifest is generated before window construction. It maps participant to partition, session to participant, and time intervals to labels. Feature normalization statistics, quantization calibration range, class weights, threshold, and model selection are fitted only from training participants. Validation participants choose a configuration once; test participants are used once for the final proposed evaluation.

A common wrong approach is to window everything, randomly shuffle windows, and then split. That leaks 60.94% shared raw samples between neighboring windows and lets participant-specific gait, strap fit, and session context appear in both sides. A second wrong approach is to split sessions but keep the same participant in train and test; it removes literal overlap but not person identity.

### 5e. Reproducibility and method-selection ledger

| ID | Location / version | Purpose | Accessibility | Limitation |
|---|---|---|---|---|
| SRC-01 | `00_SOURCE_COVERAGE_EDGE_INDUSTRY.md`, reviewed 2026-11-10 | visible topic provenance | local planning text | source bodies OPEN |
| DATA-01 | proposed consented recording set | labels and split | unavailable | no data claim possible |
| SPLIT-01 | participant/session manifest | leakage audit | proposed | not executed |
| CFG-01 | `fs=50,N=128,H=50,K=3` | illustrative sizing | in this case | not a device config |
| MODEL-01 | future hash + quantization manifest | reproducible inference | unavailable | no trained artifact |
| RUN-01 | arithmetic smoke in Section 11 | checks stated boundary | hand-derived | not target measurement |

A reproducible target run must record source checksum, participant split manifest checksum, preprocessing revision, calibration subset, random seed, converter/runtime/compiler versions, model hash, linker map, stack watermark, clock source, and raw trace checksum. A seed cannot recreate an unavailable data collection or a different compiler’s fixed-point behavior.

## 6. Analysis Walkthrough

The decisive result is conditional: the stated workload fits the illustrative byte and average-energy arithmetic only if every completed DMA block is released before reuse and every result reaches the UI before its freshness deadline. This is a feasibility screen, not performance evidence.

### 6.1 Window, hop, and overlap

Validity: uniform 50 Hz sampling, one tri-axial frame per tick, no missing samples, and `N=128` frames. Derivation: duration is frame count divided by frames/s; overlap is window duration minus hop duration.

```text
FORMULA [W-01]: T_w = N / f_s = 128 / 50 = 2.56 s
WHERE: T_w [s]; N [frames]; f_s [frames/s].
CONSTRAINT: uniform sample clock; one frame contains all three axes.
NUMERICAL EXAMPLE: H=50 frames gives T_h=H/f_s=1.00 s.
INTERPRETATION: candidate windows start every second and cover 2.56 seconds.

FORMULA [W-02]: O = (N-H)/N = (128-50)/128 = 0.609375 = 60.9375%
WHERE: O [fraction]; H [frames/hop].
CONSTRAINT: adjacent windows from one uninterrupted stream.
NUMERICAL EXAMPLE: 78 of 128 frames recur in the next window.
INTERPRETATION: random window splitting is leakage, not independent testing.
```

Independent check: `2.56 s - 1.00 s = 1.56 s`; `1.56/2.56 = 0.609375`. Sensitivity: changing to `H=128` removes overlap but emits only every 2.56 s; changing to `H=25` emits every 0.5 s but raises inference opportunities to two/s and overlap to 80.47%.

### 6.2 Sample and buffer budget

A frame has three axes times two bytes per axis, so `b_f=6 bytes/frame`. A DMA block holds 50 frames, exactly one hop interval, plus 16 bytes metadata stored outside the DMA payload.

```text
FORMULA [B-01]: S_block = 50*6 + 16 = 316 bytes
WHERE: S_block [bytes]; 50 [frames]; 6 [bytes/frame]; 16 [metadata bytes].
CONSTRAINT: metadata includes sequence, end time, state, and checksum field; alignment excluded.
NUMERICAL EXAMPLE: K=3 blocks gives S_DMA=3*316=948 bytes.
INTERPRETATION: raw DMA storage is under 1 KiB before alignment and queue objects.

FORMULA [B-02]: S_window = 128*3*2 = 768 bytes
WHERE: S_window [bytes] is an optional contiguous INT16 staging window.
CONSTRAINT: it is separate from DMA blocks; a streaming preprocessor may avoid it.
NUMERICAL EXAMPLE: 768 bytes plus a 384-byte INT8 input tensor is 1152 bytes.
INTERPRETATION: do not count an input tensor as if it aliases raw samples unless code proves it.
```

The proposed static SRAM ledger is 948 bytes DMA blocks, 768 bytes staging, 384 bytes INT8 input, 22 KiB arena, 6 KiB task stacks, 2 KiB queues/metadata, and 4 KiB reserve: `948+768+384+22528+6144+2048+4096=36916 bytes`, or 36.05 KiB. Against an illustrative 64 KiB usable SRAM boundary, headroom is `65536-36916=28620 bytes` (27.95 KiB). This is not a claim about a particular MCU.

Flash ledger: 18 KiB model plus 46 KiB program/runtime plus 4 KiB constants equals 68 KiB. Against an illustrative 128 KiB usable flash budget, remaining flash is 60 KiB. The model’s INT8 parameter bytes do not include operator code, metadata, or alignment.

### 6.3 Fixed-point quantization

Validity: each input channel uses declared affine quantization `q=round(x/s)+z`, with saturation to the INT8 range; `x` is the scaled physical quantity, `s>0` its scale, and `z` its zero point. This needs calibration values selected without test participants.

```text
FORMULA [Q-01]: q = clip(-128,127, round(x/s)+z)
WHERE: q [int8]; x [scaled acceleration]; s [scaled acceleration/count]; z [int].
CONSTRAINT: same axis order, units, scale, zero point, rounding, and saturation rule in training and firmware.
NUMERICAL EXAMPLE: x=0.80 g, s=0.02 g/count, z=0; q=round(40)=40.
INTERPRETATION: dequantized value is 40*0.02=0.80 g before rounding error.

FORMULA [Q-02]: epsilon = x - s*(q-z)
WHERE: epsilon [g] is reconstruction error.
CONSTRAINT: unsaturated input; nearest rounding.
NUMERICAL EXAMPLE: x=0.81 g gives q=41 and epsilon=0.81-0.82=-0.01 g.
INTERPRETATION: error magnitude is at most s/2=0.01 g without saturation.
```

Counterexample: `x=3.00 g` with the same scale maps to 150 before clipping and becomes 127; dequantized value is 2.54 g, so the error is 0.46 g, not bounded by half a step. Track saturation per axis and reject a calibration/model claim if clipping differs materially across participant partitions.

### 6.3.1 Bounded signal conditioning before quantization

The proposed conditioning path is deliberately short: validate frame count and timestamp continuity; convert signed sensor counts using the manifest's count-to-`g` factor; optionally subtract the training-derived channel center; apply the declared affine INT8 transform; then count saturation. It has no adaptive filter state, per-session normalization, interpolation, or dynamic allocation. Those omissions are boundaries: an adaptive value learned from a held-out session would change the input distribution, and interpolation would invent a frame after a gap.

For a channel with raw count `r`, count-to-`g` factor `c`, and training-only center `mu`, the stated pre-quantization value is `x=c*r-mu`. Both `c` and `mu` are manifest fields. If gravity is retained during training, it remains in `x`; if it is removed during training, the removal operation and its state initialization must be reproducible on target. This case does not select either convention.

The implementation must use a wide intermediate before adding the zero point: compute `round(x/s)` in a representation that cannot wrap for the declared calibration range, then add `z`, then clamp once to `[-128,127]`. Clamping the raw sensor count before unit conversion or allowing signed overflow can create a tensor that appears valid while representing another value.

For each window, record saturation counts by axis and total input values (`128` values per axis; `384` total). The count is a quality diagnostic, not a substitute label. A future acceptance rule may define a maximum count only before final-test inspection; until then, observed clipping requires reporting rather than a claim of harmlessness.

The conditioning revision belongs in the same identity bundle as the model. A change from nearest-even to away-from-zero rounding, a different integer division rule for negative values, or a reordered center subtraction changes model input and therefore requires new golden vectors and target parity evidence.

### 6.4 Ownership, sequence integrity, and freshness

Precondition: DMA is the only writer of a `FILLING` block. At completion, ISR code stamps `q` and `t_end`, changes the block to `READY`, and posts its index. A single inference task changes `READY` to `PROCESSING`; only it changes `PROCESSING` to `FREE`. DMA selects only `FREE` blocks. The invariant is that a model input is constructed only from completed, consecutive sequences and never from storage currently owned by DMA.

Worked trace: at 1000 ms, block A completes `q=120`; DMA fills B. The consumer claims A and copies/streams it into its rolling window. At 2000 ms B completes `q=121`; DMA fills C. At 3000 ms C completes `q=122`; DMA begins A. Thus A must be released before 3000 ms. With three hop blocks, the completion-to-reuse interval is `(K-1)*T_h=2.0 s`, not 3.0 s.

A sequence gap is not repaired by repeating the prior frame. Repetition changes a motion waveform without a physical observation. The correct recovery is to invalidate the current rolling window until 128 contiguous new frames exist, increment a loss counter, and display `UNAVAILABLE`.

Freshness is age at UI use, not inference CPU time. If the window ends at `t_end`, score finishes at `t_done`, and UI uses it at `t_use`, then `A=t_use-t_end`. With 24 ms processing and at most 30 ms UI scheduling, `A=54 ms` after window end under the assumption that the window is already complete. The 1.35 s display deadline leaves 1.296 s margin, but a queued 1.4 s radio operation must not delay the local UI task past it.

### 6.5 Radio and battery duty budget

The local classifier runs once/s. The proposed radio sends one 20-byte summary once/minute, not raw samples. At 8N1, 20 bytes require 200 symbols; at 115200 symbols/s, wire time is 1.736 ms before protocol wake, but this case uses an illustrative 40 ms radio-active period to account for a larger transaction assumption.

```text
FORMULA [E-01]: I_avg = sum(I_i*t_i) / T
WHERE: I_avg [mA]; I_i [mA]; t_i [s]; T [s].
CONSTRAINT: constant current within stated states; regulator loss, aging, temperature, and peaks excluded.
NUMERICAL EXAMPLE per 60 s: sensing=0.50 mA*60 s; inference=6.0 mA*1.44 s; radio=18 mA*0.040 s; sleep=0.05 mA*(60-1.44-0.040) s.
INTERPRETATION: I_avg=(30+8.64+0.72+2.926)/60=0.70477 mA.
FORMULA [E-02]: days = C / (24*I_avg)
WHERE: C [mAh]; I_avg [mA]; days [d].
CONSTRAINT: usable capacity is C and average current is representative.
NUMERICAL EXAMPLE: C=120 mAh; days=120/(24*0.70477)=7.0945 days.
```

Sensitivity: if radio active time rises from 40 ms to 400 ms per minute, radio charge rises from 0.72 to 7.2 mA·s/min. Average current becomes `(30+8.64+7.2+2.908)/60=0.81247 mA`, yielding `6.1541 days`. If inference time doubles to 48 ms/s, its charge is 17.28 mA·s/min and runtime becomes `120/(24*((30+17.28+0.72+2.854)/60))=5.8992 days`. These models omit real discharge behavior and so cannot select a battery.

## 7. The Finding

The recommended decision is to keep the upload/manual and threshold baselines available while authorizing only a proposed, opt-in INT8 on-device prototype with strict participant/session splitting, immutable DMA ownership, and an `UNAVAILABLE` outcome for any integrity or freshness failure. The illustrative SRAM and energy arithmetic does not reject the stated configuration, but it does not measure model quality, target latency, current, or battery life. The on-device path is justified only if future target traces and held-out-participant evaluation beat the threshold baseline on the predeclared eligible-window metric without hiding abstentions. No activity label is a diagnosis or a safety action.

## 8. System/Solution Design

### 8.1 Input-to-output architecture

```text
accelerometer timer -> DMA FILLING block -> completion ISR(q,t_end) -> READY queue
  -> rolling-window assembler -> quality/sequence gate -> INT16-to-INT8 quantizer
  -> INT8 inference -> margin gate -> freshness gate -> UI(label | UNAVAILABLE)
  -> optional 1/min summary queue -> radio task

failure path: gap/overrun/arena error/late result -> invalidate window -> counter -> UNAVAILABLE
```

The radio task owns its transmission buffer and cannot hold an acquisition block. The UI owns a copied result record `{epoch, q_end, t_end, model_hash_prefix, label, reason}`. It never reads an inference tensor or DMA payload after publication.

### 8.2 State and interface contracts

| Interface | Producer | Consumer | Must carry | Reject when |
|---|---|---|---|---|
| DMA block | DMA/ISR | assembler | epoch, q, t_end, count, state | count/sequence/state invalid |
| rolling window | assembler | quantizer | q_start, q_end, contiguous flag | any missing frame |
| input tensor | quantizer | runtime | shape, axis order, scale ID | scale/model mismatch |
| scores | runtime | policy | model hash, class order, q_end | runtime error or NaN-equivalent contract |
| result | policy | UI | end time, age basis, reason | age exceeds 1.35 s |
| summary | UI/policy | radio | aggregate only, no raw | queue full; drop summary |

### 8.3 Deadline margins

The hop arrival period is 1.0 s. Let assumed processing be 24 ms, interfering work 16 ms, and local UI scheduling 30 ms. Response to render is 70 ms, leaving 930 ms before the next hop and 1,280 ms before the 1.35 s freshness limit. These are model margins; a real response-time proof must include ISR time, priority blocking, cache/flash stalls, logging, and every higher-priority release.

If processing plus interference becomes 1.02 s, the next result competes with a newly completed hop. If age becomes 1.36 s, bytes may still be intact, but the UI must display `UNAVAILABLE`; freshness and storage integrity are separate constraints.

### 8.4 Degradation and recovery

A sensor fault, sequence gap, queue overrun, runtime error, stale result, or quantization manifest mismatch maps to `UNAVAILABLE`, a reason code, and a monotonic counter. It does not replay the last label because a still-looking UI is easily mistaken for a current observation.

Recovery requires `N=128` consecutive valid new frames after a discontinuity. A reset increments epoch; sequence numbers are compared only within epoch. A radio failure drops a summary after bounded retries and cannot block classification. A model update is accepted only after signature/authorization policy outside this case and a manifest check that binds shape, class order, scale, zero point, and model hash.

## 9. Evaluation & Validation

No validation below has been executed. The table defines the evidence required before the design can advance.

| Test ID | Population/workload | Numerator | Denominator | Pass interpretation | Failure response |
|---|---|---|---|---|---|
| VAL-01 | held-out participants | correct eligible windows | eligible labeled windows | report macro recall and class recall | inspect label/split/model |
| VAL-02 | held-out participants | eligible fresh outputs | candidate labeled windows | report availability separately | inspect integrity and timing |
| VAL-03 | threshold baseline | same metric as model | same windows | comparable decision boundary | retain simpler baseline if equal |
| VAL-04 | participant split audit | participants in >1 split | all participants | must be zero | rebuild manifest |
| VAL-05 | overlap audit | cross-split shared samples | all samples | must be zero | group/purge windows |
| VAL-06 | quantization parity | tolerable output differences | golden vectors | define tolerance before run | inspect scales/layout |
| VAL-07 | saturation audit | clipped values | all per-axis values | report by participant/class | recalibrate or abstain |
| VAL-08 | DMA wrap trace | blocks read while filling | all claims | must be zero | fix ownership state |
| VAL-09 | sequence trace | gaps detected | injected gaps | must be all injected gaps | fix counter/epoch logic |
| VAL-10 | overflow trace | stale labels after overload | overload trials | must be zero | prioritize/expire output |
| VAL-11 | freshness trace | rendered age <=1.35 s | rendered results | must be 100% | UI gate correction |
| VAL-12 | target timing | response <= declared budget | representative windows | report distribution and max | redesign workload |
| VAL-13 | memory | peak SRAM/flash <= budgets | target build | map plus stack evidence | reduce/repartition |
| VAL-14 | power | measured charge by state | timed states | replace illustrative estimate | revise product claim |
| VAL-15 | radio duty | transactions within policy | minute intervals | radio cannot block local path | isolate queue/priority |
| VAL-16 | removal/loose strap | correct abstentions | induced quality events | no confident invented state | improve quality gate |
| VAL-17 | transition labels | windows excluded correctly | boundary windows | no mixed label training rows | adjust protocol |
| VAL-18 | class wording | unsafe interpretation reports | usability participants | no diagnosis implication | revise copy |

The primary evaluation population is held-out people, not randomly held-out windows. Report per-participant distributions because an aggregate denominator dominated by long sessions can conceal a participant with systematic failure. Also report the abstention reason distribution; a model that looks accurate by rejecting every difficult window is not useful.

A quantitative comparison must use the same eligible windows for threshold and model. If the model runs locally but the threshold uses cloud-smoothed data, the comparison answers a different question. Calibration and threshold tuning are confined to validation participants.

## 10. Action Taken

In this illustrative case, no device feature was deployed and no model was trained. The action is a proposed decision gate: collect consented, protocol-labeled recordings; freeze a participant-level split; implement the DMA state machine and golden-vector quantization check; then compare the threshold and INT8 alternatives on held-out participants.

The product manager does not authorize diagnostic wording, automatic action, raw-data upload by default, or a battery-life claim from the arithmetic. The embedded lead must reject any implementation that dynamically allocates in the hot path, permits DMA to overwrite a claimed block, or displays a late result.

## 11. Outcome & Results

**Outcome status: not-measured.** The only results are illustrative arithmetic checks: a 2.56 s window with a 1.00 s hop has 60.9375% overlap; three 316-byte hop blocks consume 948 bytes; the stated SRAM ledger is 36,916 bytes; and the stated current model yields 0.70477 mA average and 7.0945 days from 120 mAh.

These results support a narrow conclusion: the written assumptions are mutually arithmetically consistent. They do not establish classifier accuracy, generalization, latency, current, battery life, radio reliability, user benefit, clinical validity, or suitability for any named hardware.

### Arithmetic/state/decision smoke

**Inputs:** `f_s=50 frames/s`, `N=128`, `H=50`, `b_f=6 bytes`, `metadata=16 bytes`, `K=3`, `C=24 ms`, `I=16 ms`, `UI=30 ms`, freshness limit `1.35 s`, and current-state inputs from [E-01].

**Results:** `T_w=2.56 s`; overlap `78/128=60.9375%`; `S_block=316 bytes`; `S_DMA=948 bytes`; `T_reuse=(3-1)*1.00=2.00 s`; response `24+16+30=70 ms`; freshness margin `1350-70=1280 ms`; SRAM headroom `65536-36916=28620 bytes`; and `I_avg=0.70477 mA`.

**Boundary decisions:** a `2010 ms` consumer release after a `q=120` completion with three 1 s blocks races DMA reuse and is invalid; a `1360 ms` rendered age exceeds the `1350 ms` freshness limit and must be `UNAVAILABLE`; a fourth ready block when only three block stores exist is an overflow event, not a queueing success. These are hand-checked illustrative state boundaries, not a permanent test or target measurement.

## 12. Failure Modes & What Went Wrong

| Failure | Competing causes | Discriminating signal | Bounded recovery | Prevention / verification |
|---|---|---|---|---|
| Repeated windows | consumer reused prior pointer; sensor clock stopped | duplicate q or end time | invalidate rolling window | sequence trace across wrap |
| Torn samples | DMA overwrote processing block; cache/order error | q changes during read | stop claim, count overrun | ownership state/injected delay |
| Stale UI label | blocked UI; radio priority inversion | age > limit with normal inference | show unavailable | timestamp at render |
| Good random accuracy | overlap; same participant leakage | shared raw samples/IDs across split | rebuild split | manifest and interval audit |
| Bad holdout person | placement change; true model weakness | per-person slice failure | abstain or collect evidence | placement protocol |
| Quantization drift | wrong scale; axis order swapped | golden vector mismatch | reject artifact | manifest/hash parity |
| Saturation | range too narrow; impact event | high clip count by axis | quality abstention | calibration held out |
| Arena collision | underestimated activation; stack growth | map/watermark failure | refuse feature | static allocation report |
| Queue overflow | inference interference; logging block | ready count exceeds capacity | drop result, reset window | bounded queue stress |
| Radio drains battery | retries; larger transaction | active-time counter high | drop summaries | separate radio budget |
| Manual label noise | transition mixed; instruction mismatch | reviewer disagreement | mark ambiguous | double review protocol |
| Threshold looks better | different eligibility set; leakage | denominator mismatch | align comparator windows | fixed evaluation manifest |
| Misread as diagnosis | unsafe copy; user expectation | comprehension interview | remove/clarify display | copy review |
| Device removal classified | strap loose; gesture resembles walk | quality flags/placement log | unavailable | removal test |
| Clock reset corrupts age | epoch omitted; wrap mishandled | negative/huge age | reset state | epoch trace |
| Model update mismatch | stale scale/class order | model/manifest hashes differ | reject update | atomic artifact bundle |

A particularly deceptive failure is a late result with a correct label. The model may have classified the intended window perfectly, yet the product violates its own meaning because the wearer sees it after the state changed. Correctness requires data identity and decision time, not only a score.

## 13. Transferable Lesson

An embedded classifier is a chain of contracts: sampling cadence, sample ownership, interval labels, split group, preprocessing representation, model artifact, response age, UI semantics, and energy policy. Improving one link does not compensate for violating another.

Participant-level splitting is a product-requirement analogue of memory ownership: each keeps evidence from silently borrowing identity from a region that is supposed to be independent. Window overlap is not inherently bad for streaming inference; it is bad only when an evaluation pretends overlapped records are independent test evidence.

Fixed-point deployment is not “convert to INT8 and hope.” The scale, zero point, axis order, rounding mode, saturation behavior, and calibration population are all part of the model. A byte-saving claim without these bindings is not reproducible.

The safe degraded behavior for a descriptive wearable feature is abstention. That does not solve a safety application, which needs its own hazard analysis and validated control path. Do not transfer this case’s `UNAVAILABLE` behavior to a safety-critical function without a separate decision and evidence basis.

## 14. Connections & Cross-References

Related local material: [Embedded Acquisition, Firmware Contracts and TinyML Integration](../Cheat_Sheets/01_Embedded_Acquisition_Firmware_and_TinyML.md) covers the reusable DMA, timing, UART, energy, and memory contracts. The collection [INDEX](INDEX.md) retains the distinct industrial-vibration case; activity labels must not be reused as maintenance evidence.

### Interview pack

#### Q1. Why split by participant rather than window?

**Checked answer:** Windows from one person share gait, device fit, and overlapping samples, so random windows estimate recognition of known people rather than generalization to a new wearer.

**Reasoning:** Participant grouping prevents both shared identities and the 78 reused adjacent frames from crossing the final-test boundary.

**Follow-up:** Ask for the split manifest before model code.

**Acceptable alternatives:** Session split may supplement participant split but cannot replace it when new-person generalization matters.

**Red flags:** Saying shuffle is enough.

**Scoring:** 2: names identity leakage; 1: mentions overfit; 0: random split only.

**Answer support:** Section 5a.

#### Q2. Compute the overlap for N=128 and H=50.

**Checked answer:** Overlap is (128-50)/128=78/128=60.9375%; 1.56 seconds of each 2.56-second window recurs.

**Reasoning:** The numerator is repeated frames, while the denominator is the complete 128-frame window; seconds provide an independent check.

**Follow-up:** What changes if H=128?

**Acceptable alternatives:** Non-overlap reduces repeated evidence but lowers decision cadence.

**Red flags:** Calling hop the overlap.

**Scoring:** 2: correct formula and units; 1: correct number only; 0: wrong denominator.

**Answer support:** Section 6.1.

#### Q3. What makes an activity label non-diagnostic?

**Checked answer:** It describes a motion class only and must not infer disease, fitness, adherence, impairment, or risk.

**Reasoning:** The permitted output vocabulary names observed motion categories, whereas the excluded claims require health or risk evidence absent here.

**Follow-up:** Who reviews wording?

**Acceptable alternatives:** A human-factors/product review may reject misleading wording.

**Red flags:** Use cautious wording plus explicit limitations. Equating a motion class with health status.

**Scoring:** 2: distinguishes purpose; 1: says disclaimer; 0: claims medical meaning.

**Answer support:** Sections 1 and 3.

#### Q4. Why is a DMA completion interrupt not permission to retain a pointer forever?

**Checked answer:** Completion transfers a bounded block; circular DMA will reuse it after two future 1-second blocks with K=3.

**Reasoning:** With three blocks, two subsequent one-second completions return the original store to DMA eligibility, making the reuse interval 2.0 seconds.

**Follow-up:** What happens at 2010 ms?

**Acceptable alternatives:** The block may be being written again, so invalidate rather than read.

**Red flags:** Copy into owned storage if its cost is budgeted. Counting K periods instead of K-1.

**Scoring:** 2: explains state/reuse; 1: says race; 0: relies on volatile pointer.

**Answer support:** Section 6.4.

#### Q5. How do you calculate result freshness?

**Checked answer:** Subtract window end time from UI-use time; CPU time alone omits queue and UI delay.

**Reasoning:** The decision is consumed at rendering, so a correct score can still be stale after queueing or UI scheduling.

**Follow-up:** What is the response at 1360 ms?

**Acceptable alternatives:** It is expired against 1350 ms and must display UNAVAILABLE.

**Red flags:** Use a monotonic shared clock or bounded conversion. Checking age only at inference finish.

**Scoring:** 2: correct origin and action; 1: time mention; 0: CPU-only.

**Answer support:** Sections 4 and 8.

#### Q6. What could make INT8 accuracy fail although float validation passes?

**Checked answer:** A wrong scale, zero point, axis order, rounding rule, calibration range, or saturation pattern changes inputs or kernels.

**Reasoning:** Quantization changes both numeric values and discrete implementation behavior; matching float accuracy does not prove tensor parity.

**Follow-up:** What artifact detects it?

**Acceptable alternatives:** Golden vectors and a quantization manifest bind expected values.

**Red flags:** QAT or a smaller model may be considered after diagnosis. Assuming INT8 is lossless.

**Scoring:** 2: names contract plus test; 1: says quantization error; 0: retrain blindly.

**Answer support:** Section 6.3.

#### Q7. What is the stated SRAM headroom?

**Checked answer:** The ledger totals 36,916 bytes; 65,536 minus that equals 28,620 bytes, before proving target stack peak.

**Reasoning:** The subtraction is a static ledger calculation, not an allocation trace; stack peaks and alignment can consume remaining capacity.

**Follow-up:** Does that prove fit?

**Acceptable alternatives:** No; alignment, runtime allocations, DMA reachability, and measured stack peak remain open.

**Red flags:** Use static allocation and map plus watermark evidence. Calling nominal remaining bytes a guarantee.

**Scoring:** 2: arithmetic and limitation; 1: arithmetic; 0: unsupported guarantee.

**Answer support:** Section 6.2.

#### Q8. Why compare the threshold baseline on the identical eligible windows?

**Checked answer:** Different exclusions create different denominators, so an apparent model gain may be selection rather than classification improvement.

**Reasoning:** Holding interval identity constant isolates the classifier decision from eligibility and coverage differences.

**Follow-up:** What if the baseline cannot emit STAIRS?

**Acceptable alternatives:** Report its defined task honestly or retain it as a two-class comparator.

**Red flags:** A manual-review comparator can answer a different latency question. Comparing headline accuracy from different datasets.

**Scoring:** 2: identifies denominator; 1: says fairness; 0: accepts different windows.

**Answer support:** Sections 5c and 9.

#### Q9. How is a label interval created?

**Checked answer:** Instruction, visible time marker, participant confirmation, reviewer adjudication, and transition exclusion define the interval.

**Reasoning:** The label is an interval-level claim, so a transition can contain more than one instructed motion even when each frame has a timestamp.

**Follow-up:** Why exclude transitions?

**Acceptable alternatives:** A mixed transition lacks one activity target.

**Red flags:** Use an ambiguity label only if it has a collection rule. Assigning the facilitator's guess to every frame.

**Scoring:** 2: gives protocol and boundary; 1: mentions annotation; 0: ignores uncertainty.

**Answer support:** Section 5a.

#### Q10. What changes if radio active time is ten times larger?

**Checked answer:** From 40 to 400 ms/minute, the stated model gives 0.81247 mA and 6.1541 illustrative days instead of 0.70477 mA and 7.0945 days.

**Reasoning:** Radio active charge grows by 6.48 mA·s/minute, while sleep shrinks by 0.36 mA·s/minute, producing the stated net increase.

**Follow-up:** Does this establish battery life?

**Acceptable alternatives:** No; it omits discharge and measured state current.

**Red flags:** Reduce summary rate or transaction cost after measurement. Claiming the arithmetic is a benchmark.

**Scoring:** 2: computes and limits; 1: direction only; 0: ignores duty.

**Answer support:** Section 6.5.

#### Q11. Diagnose a stream with duplicated end times but no sequence gaps.

**Checked answer:** Possible causes include timestamp assignment in the ISR, clock resolution, or consumer metadata reuse; inspect raw q/time pairs and ownership transitions.

**Reasoning:** Sequence establishes order, not timestamp provenance; comparing payload, metadata, and ownership state separates these fault families.

**Follow-up:** What would distinguish sensor replay?

**Acceptable alternatives:** A sensor replay may repeat payload values with advancing q; metadata reuse repeats headers inconsistently.

**Red flags:** Fail closed to UNAVAILABLE while tracing. Immediately blame the model.

**Scoring:** 2: competing causes and check; 1: one cause; 0: retrain.

**Answer support:** Section 12.

#### Q12. A manager asks for fall alerts from this classifier. What changes?

**Checked answer:** That is a separate safety/clinical decision with different labels, false-negative consequences, escalation, validation, and likely regulatory analysis; this case cannot authorize it.

**Reasoning:** Changing the output purpose changes consequences and required evidence, not merely the number of classes in the existing tensor.

**Follow-up:** Can the current label be reused?

**Acceptable alternatives:** Only as an unvalidated research input, not as evidence of fall performance.

**Red flags:** Defer or create a new bounded case. Adding a FALL class to the current UI.

**Scoring:** 2: identifies changed contract; 1: says out of scope; 0: treats as class addition.

**Answer support:** Sections 1 and 13.

### Operational evidence checklist

These are proposed acceptance records, not claims that a device run occurred. Each record has a named artifact, an explicit population or injected-fault denominator, and a disposition that prevents uncertain input from becoming a label.

### Evidence control 01: Acquisition trace

**Artifact.** For every DMA completion, retain timer tick, block index, epoch, `q`, frame count, and state transition.

**Question answered.** Did the advertised sample cadence and one-frame-per-tick contract actually reach the firmware boundary?

**Denominator.** All DMA completions in a timed capture, with the capture duration and timer source stated.

**Failure signature.** A repeated `q`, skipped `q`, count other than 50, or completion before the expected tick is a trace failure.

**Disposition.** Mark the affected interval invalid, clear the rolling window, and require 128 newly contiguous frames.

**Change trigger.** Recollect after changing the sensor ODR, bus, ISR, DMA configuration, or clock source.

### Evidence control 02: Capture-time convention

**Artifact.** Document whether `t_end` is latched at the 128th sample capture, at DMA transfer completion, or at ISR entry.

**Question answered.** Is display age measured from the physical observation rather than a later software event?

**Denominator.** Every rendered result in a clock-alignment trace, including ones rejected for lateness.

**Failure signature.** A result whose age shrinks when ISR delay is injected uses an invalid timestamp origin.

**Disposition.** Reject the result; do not substitute the interrupt time for the frame time.

**Change trigger.** Recheck after timer wrap handling, sensor timestamp mode, or scheduler changes.

### Evidence control 03: Axis and units contract

**Artifact.** Bind axis order, handedness, count-to-`g` conversion, gravity treatment, and tensor channel order in one manifest.

**Question answered.** Does firmware feed the representation that training and calibration expected?

**Denominator.** Every model artifact revision and its golden-vector set, not merely one successful inference.

**Failure signature.** An X/Y/Z permutation or units mismatch changes the expected INT8 tensor.

**Disposition.** Disable inference for that artifact revision and retain the raw diagnostic code.

**Change trigger.** Rebuild evidence after sensor placement convention, preprocessing, or model conversion changes.

### Evidence control 04: Full-scale and clipping contract

**Artifact.** Record configured accelerometer range, conversion factor, per-axis clipped-value count, and calibration range.

**Question answered.** Is an apparent activity pattern produced by bounded physical input rather than silently clipped counts?

**Denominator.** All channel values in each declared participant/class partition; report counts and total values.

**Failure signature.** A held-out partition has clipping outside the predeclared acceptance rule or a range differs from its manifest.

**Disposition.** Classify the window as quality-ineligible and investigate range/calibration before changing the model.

**Change trigger.** Repeat for sensor range, placement, collection-protocol, or quantization-scale changes.

### Evidence control 05: Consent and withdrawal lineage

**Artifact.** Store consent revision, participant pseudonym, session inclusion state, and withdrawal processing event in the data manifest.

**Question answered.** Which sessions are permitted to contribute to the stated research population?

**Denominator.** All scheduled sessions, including incomplete, withdrawn, and rejected sessions.

**Failure signature.** A derived window refers to a missing consent record or a withdrawn session remains in an export.

**Disposition.** Remove all derived records for that session, regenerate split/window manifests, and report the changed denominator.

**Change trigger.** Reconcile before each data export or protocol revision.

### Evidence control 06: Session boundary trace

**Artifact.** Log donning, removal, charging, clock reset, firmware restart, and declared session start/end events.

**Question answered.** Are contiguous sample sequences also physically continuous observations?

**Denominator.** All sessions and all reset/restart events during collection or fault injection.

**Failure signature.** A rolling window spans a declared physical or clock discontinuity.

**Disposition.** Increment epoch, discard partial history, and begin eligibility only after 128 new frames.

**Change trigger.** Repeat when session tooling or reset recovery changes.

### Evidence control 07: Transition exclusion policy

**Artifact.** Preserve instruction-change markers, settling durations, start/stop/turn markers, and the rule that excluded each interval.

**Question answered.** Does one retained training window have one defensible activity target?

**Denominator.** All candidate windows touching an instruction or adjudication boundary.

**Failure signature.** A window crosses labels, a transition zone, or an unadjudicated boundary.

**Disposition.** Exclude it from training and performance denominators rather than assigning its majority label.

**Change trigger.** Freeze the rule before model-error review; revise only with a new protocol version.

### Evidence control 08: Class exposure accounting

**Artifact.** Report prompted seconds, retained seconds, eligible windows, and participant count for each display class.

**Question answered.** Is a high count caused by more people or simply more overlapping windows from a long session?

**Denominator.** Declared participants and labeled seconds, alongside derived windows; never windows alone.

**Failure signature.** A class appears supported only by one participant or by transition-heavy retained intervals.

**Disposition.** State the limitation and collect according to the protocol; do not fabricate balance by duplicating windows.

**Change trigger.** Recompute after any label, exclusion, hop, or split-manifest change.

### Evidence control 09: Missing-frame handling

**Artifact.** Retain expected tick, observed tick, missing-frame count, and first missing sequence for each discontinuity.

**Question answered.** Did the pipeline distinguish absent motion measurements from a low-motion signal?

**Denominator.** All expected sample ticks in a capture plus every deliberate dropped-frame injection.

**Failure signature.** A gap is followed by a retained window before 128 consecutive replacement frames arrive.

**Disposition.** Never interpolate for this case; invalidate the window and surface `UNAVAILABLE`.

**Change trigger.** Repeat after bus retry, DMA, or sequence-counter changes.

### Evidence control 10: Threshold-selection isolation

**Artifact.** Version the score threshold, margin rule, validation participant IDs, and selection date with the model configuration.

**Question answered.** Was abstention policy selected without observing final test participants?

**Denominator.** Every candidate configuration considered before the single final-test run.

**Failure signature.** A threshold revision cites test errors, test participants, or pooled test score distributions.

**Disposition.** Invalidate that comparison and repeat selection on training/validation partitions only.

**Change trigger.** Reopen after class definition, model, calibration, or decision-cost changes.

### Evidence control 11: UI time semantics

**Artifact.** Capture rendered text, capture interval/end time, reason category, and UI-use timestamp for result-state transitions.

**Question answered.** Can a wearer distinguish a current descriptive label from an unavailable or expired observation?

**Denominator.** All render attempts, including suppressed labels and repeated UI refreshes.

**Failure signature.** The UI retains a prior label after a rejected result or omits the unavailable state.

**Disposition.** Replace prior content with `UNAVAILABLE`; preserve the reason without presenting it as activity evidence.

**Change trigger.** Recheck after copy, localization, UI scheduler, or freshness-limit changes.

### Evidence control 12: Telemetry isolation

**Artifact.** Record summary payload schema, enqueue time, retry count, queue depth, drop count, and radio active intervals.

**Question answered.** Can optional communication consume storage or scheduler time owned by acquisition and local display?

**Denominator.** Every summary enqueue attempt and every radio transaction in the stated observation interval.

**Failure signature.** A full radio queue blocks an assembler, inference, UI task, or DMA-block release.

**Disposition.** Drop the summary according to policy; never retain a DMA block or delay the local outcome.

**Change trigger.** Repeat for payload, retry, transport, or priority changes.

### Evidence control 13: Power-state attribution

**Artifact.** Measure current with synchronized markers for sensing, preprocessing, inference, radio, and sleep.

**Question answered.** Which design state explains a charge increase rather than merely whether average current changed?

**Denominator.** Timed state occupancy over a stated interval, with instrument bandwidth and sampling method recorded.

**Failure signature.** An average-current claim has no state marker, duration, or repeatable configuration identity.

**Disposition.** Keep the arithmetic illustrative and decline a battery-life claim until a trace can be attributed.

**Change trigger.** Re-measure after firmware, radio duty, model, battery, temperature, or regulator changes.

### Evidence control 14: Arena peak and failure path

**Artifact.** Retain target-build allocator peak, requested arena size, runtime/model hash, and behavior when the arena is insufficient.

**Question answered.** Does static inference memory fit with a known refusal path rather than a desktop assumption?

**Denominator.** Each supported model/runtime build and all inference initialization attempts in its target trace.

**Failure signature.** Peak exceeds the 22 KiB assumption, allocation fails ambiguously, or an error leaves a prior result visible.

**Disposition.** Reject the artifact, report `UNAVAILABLE`, and resize or redesign before enabling inference.

**Change trigger.** Repeat for converter, operator set, compiler, model, or runtime updates.

### Evidence control 15: Stack and interrupt margin

**Artifact.** Capture task stack high-water marks and ISR nesting/latency under normal, logging, radio, and injected-overload paths.

**Question answered.** Is the 6 KiB stack ledger credible when error handling and preemption occur?

**Denominator.** Every named execution scenario, with a separate measurement for each task and interrupt context.

**Failure signature.** Watermark crosses reserve, stack growth is unmeasured, or a high-priority path delays block release.

**Disposition.** Reduce stack use or allocate a bounded stack; do not borrow the 4 KiB reserve silently.

**Change trigger.** Recollect after scheduler priority, logging, compiler, or error-path changes.

### Evidence control 16: Golden-vector parity

**Artifact.** Store immutable raw frames, expected scaled values, INT8 tensor, scores, class order, and tolerances.

**Question answered.** Where does float-to-firmware divergence first appear: conditioning, quantization, or runtime?

**Denominator.** A preselected vector set spanning zero, rounding ties, negative values, near-range values, and saturation.

**Failure signature.** Tensor bytes differ where equality is required, or scores exceed the declared tolerance.

**Disposition.** Block the artifact and compare intermediate arrays before retraining or changing thresholds.

**Change trigger.** Regenerate only from training/calibration provenance after any manifest or arithmetic change.

### Evidence control 17: Artifact identity

**Artifact.** Atomically bind model hash, input shape, class order, scales, zero points, rounding policy, converter, and runtime version.

**Question answered.** Does a byte blob have one unambiguous semantic interpretation?

**Denominator.** Every boot-time artifact load and every model package accepted by the update path.

**Failure signature.** A required field is absent, hashes disagree, or metadata is paired with another model blob.

**Disposition.** Refuse inference and record `ARTIFACT_MISMATCH`; a fallback label is not permitted.

**Change trigger.** Recheck at build, packaging, update, and rollback boundaries.

### Evidence control 18: Atomic update boundary

**Artifact.** Record package version, integrity result, accepted/rejected state, prior active version, and restart behavior.

**Question answered.** Can weights, quantization metadata, and class mapping change as one transaction?

**Denominator.** All update and rollback attempts, including power-interruption injection where applicable.

**Failure signature.** A reboot selects partially written contents or combines a new model with old scale metadata.

**Disposition.** Retain the last complete accepted package or disable inference; do not use a mixed package.

**Change trigger.** Repeat after packaging, storage layout, authorization, or bootloader changes.

### Evidence control 19: Ready-queue overload

**Artifact.** Trace block state, ready-queue depth, claim/release times, and reason code while delaying the consumer.

**Question answered.** Does the system identify saturation before DMA is offered a non-FREE block?

**Denominator.** Each injected consumer delay and every completion during that delay.

**Failure signature.** Queue depth reaches capacity, a READY block is overwritten, or a label survives an overflow epoch.

**Disposition.** Declare overflow, invalidate rolling history, drain stale work, increment epoch or loss generation, and rebuild 128 frames.

**Change trigger.** Repeat after queue capacity, priorities, block count, processing time, or logging changes.

### Evidence control 20: Clock reset and wrap

**Artifact.** Record clock width, wrap period, epoch increment, reset marker, and age calculation around induced reset/wrap.

**Question answered.** Can age remain nonnegative and meaningful when a monotonic counter restarts or wraps?

**Denominator.** All injected resets and wrap-boundary traces, plus every result immediately after them.

**Failure signature.** Age becomes negative, implausibly huge, or is computed across epochs.

**Disposition.** Invalidate partial windows and labels from the old epoch, then restart accumulation from the new epoch.

**Change trigger.** Revalidate after clock driver, timebase width, sleep mode, or timestamp representation changes.

### Evidence control 21: Participant-level reporting

**Artifact.** Publish per-held-out-participant eligible-window count, class recall, availability, and abstention reasons beside pooled values.

**Question answered.** Does aggregate performance hide a wearer whose recordings consistently fail the declared contract?

**Denominator.** Each final-test participant and their eligible labeled windows; disclose participants with zero eligible windows.

**Failure signature.** Only a pooled metric is reported or a participant is silently omitted after split assignment.

**Disposition.** Retain the participant row, state the limitation, and investigate protocol or model boundaries without relabeling outcomes.

**Change trigger.** Recompute after any final-test inclusion, eligibility, or class-definition change.

### Evidence control 22: Placement-shift boundary

**Artifact.** Record intended placement, declared alternative placement, strap condition, and whether data belong to the primary or shift slice.

**Question answered.** Is a placement change distinguished from normal variation in the intended use condition?

**Denominator.** Separate windows and participants for intended placement and each declared shift condition.

**Failure signature.** Shift recordings are pooled into the primary metric or a placement flag is missing.

**Disposition.** Report shift behavior as a limitation; do not reinterpret it as evidence of general population performance.

**Change trigger.** Repeat when hardware form, wear instructions, sensor orientation, or target population changes.

### Evidence control 23: Free-living `OTHER` boundary

**Artifact.** Separate collection purpose, labeling rule, reviewer basis, and retention decision for free-living recordings.

**Question answered.** Is `OTHER` a defined non-target category rather than a container for unknown quality or unreviewed activity?

**Denominator.** All free-living intervals offered for `OTHER`, with accepted, rejected, and ambiguous intervals counted separately.

**Failure signature.** Missing labels, removal events, or transition segments are silently assigned `OTHER`.

**Disposition.** Route quality failures to `UNAVAILABLE`; retain only intervals meeting the independent `OTHER` rule.

**Change trigger.** Revisit after changing the display taxonomy or free-living protocol.

### Evidence control 24: Baseline comparator timing

**Artifact.** Record capture-to-review elapsed time, reviewer availability, data coverage, and eligible interval IDs for upload/manual review.

**Question answered.** Does the stated baseline answer the same immediate-descriptive-label question as the local path?

**Denominator.** The same predeclared eligible windows used by the local model and threshold comparator.

**Failure signature.** Manual review uses additional context, different intervals, or an unstated delay while being called real-time.

**Disposition.** Describe it as an offline comparator and prevent headline accuracy or latency comparisons across mismatched sets.

**Change trigger.** Repeat when review protocol, upload policy, or decision deadline changes.

### Evidence control 25: Data-egress boundary

**Artifact.** List every field leaving the device, destination, retention event, raw-frame prohibition, and summary drop rule.

**Question answered.** Does local inference correspond to the stated data-flow boundary rather than an unsupported privacy claim?

**Denominator.** Every egress message type and every attempted transaction in a reviewed firmware configuration.

**Failure signature.** Raw frames, persistent identifiers, or an undeclared diagnostic payload appear in the radio queue.

**Disposition.** Block the message type, update the data-flow record, and require review before re-enabling it.

**Change trigger.** Re-audit after telemetry schema, debug build, transport, or retention-policy changes.

### Cross-control decision boundaries

**Window identity.** A displayed result must trace to `{epoch,q_start,q_end,t_end}` and an immutable assembly history; an unlabeled pointer is not evidence.

**Population identity.** Each participant belongs to exactly one partition and each session inherits that assignment before windows exist.

**Interval identity.** A retained label requires one final activity interval and no transition, gap, removal, or exclusion flag.

**Tensor identity.** Axis order, units, scale, zero point, rounding, saturation, shape, class order, and model hash must agree as one contract.

**Memory identity.** DMA payload, metadata, staging, tensor, arena, stacks, queues, alignment, and reserve must reconcile to the target linker/map evidence.

**Reuse identity.** The producer may select only FREE storage; a consumer must release its block before the 2,000 ms reuse boundary.

**Queue identity.** A queue's capacity is an admission limit, not permission to preserve old work after overflow.

**Freshness identity.** The UI calculates `t_use-t_end` using one epoch-aware monotonic timebase and rejects age above 1,350 ms.

**Energy identity.** Sensing, inference, radio, and sleep charge remain separately attributable; an average is not a battery result.

**Comparator identity.** Threshold, local model, and upload/manual review disclose which identical eligible intervals they do and do not cover.

**Abstention identity.** Quality, sequence, overflow, artifact, runtime, and freshness rejection stay distinguishable in records and UI semantics.

**Scope identity.** A request for fall response, diagnosis, emergency action, adherence, or productivity inference creates a new decision contract.

**Recovery identity.** After a discontinuity, 128 consecutive new frames—not old buffer contents or a prior label—are the earliest path back to eligibility.

**Evidence identity.** Illustrative arithmetic remains illustrative until a named target configuration, method, trace, and denominator support a narrower statement.

**Decision identity.** The product owner may retain a baseline or defer the feature when any required evidence record is absent or contradictory.

### Design assurance scenarios

**Scenario A01 — Sensor-rate drift.**
**Stimulus.** The timer trace reports 48 completed tri-axial frames per second rather than the manifest rate of 50 frames/s.
**Calculation.** A 128-frame window then spans `128 / 48 = 2.6667 s`, not the declared 2.56 s.
**Diagnostic.** Compare successive capture timestamps against the declared 20 ms period before assembling a tensor.
**Decision.** Mark that configuration ineligible because the trained receptive field and runtime meaning have both changed.
**Recovery.** Restore the rate or create a separately versioned collection, preprocessing, and model evaluation path.
**Evidence.** Retain the timer configuration, capture trace, and model-manifest rate in one review record.
**Boundary.** A stable 48 Hz stream is not repaired by relabeling it as 50 Hz; the duration mismatch persists.
**Owner.** The embedded lead owns the clock trace, while the data lead decides whether a new model study is warranted.
**Review.** Repeat this check after oscillator, power-mode, sensor ODR, or timer-driver changes.

**Scenario A02 — Single-frame discontinuity.**
**Stimulus.** Sequence number 802 is followed by 804 after an otherwise normal DMA completion.
**Calculation.** The missing sequence represents one absent 20 ms frame at the declared 50 frames/s cadence.
**Diagnostic.** Compare expected and observed sequence increments before copying either block into rolling history.
**Decision.** Reject all windows containing the gap rather than duplicating frame 802 or compressing time.
**Recovery.** Resume eligibility only after 128 newly consecutive frames, which takes 2.56 s at the stated rate.
**Evidence.** Record the first absent sequence, epoch, source block, and recovery-complete sequence.
**Boundary.** A low acceleration value is an observation; an absent acceleration value is an integrity failure.
**Owner.** Firmware owns gap detection, and product copy must expose `UNAVAILABLE` instead of an old label.
**Review.** Inject one lost frame at each block boundary and confirm no early recovery result is rendered.

**Scenario A03 — Timestamp-jitter interpretation.**
**Stimulus.** A capture interval of 29 ms appears where the nominal interval is 20 ms.
**Calculation.** The interval exceeds the nominal cadence by `29 - 20 = 9 ms`, or 45% of one sample period.
**Diagnostic.** Evaluate timestamps and sequence count together, because a delayed interrupt can differ from delayed sampling.
**Decision.** Do not assume a sensor fault until the timestamp-origin convention identifies the delayed event.
**Recovery.** If capture-time gaps exceed the predeclared jitter allowance, invalidate the rolling window and restart accumulation.
**Evidence.** Preserve sensor time, DMA-completion time, ISR-entry time, and the conversion rule used by the age calculation.
**Boundary.** Using ISR time as capture time can hide bus delay, making an old observation appear fresh.
**Owner.** The embedded lead documents the timebase; the evaluator verifies that all traces use the same convention.
**Review.** Re-run the fault trace after changing bus speed, interrupt priority, or timestamp width.

**Scenario A04 — Full-scale clipping.**
**Stimulus.** One axis reports 127 INT8 values after preprocessing for 14 of the 128 samples in a candidate window.
**Calculation.** The axis clipping fraction is `14 / 128 = 0.109375`, or 10.9375% of that axis's values.
**Diagnostic.** Compare raw-count range, configured sensor range, scale manifest, and saturation counters by axis.
**Decision.** Treat the window as quality-ineligible until the predeclared clipping policy says otherwise.
**Recovery.** Investigate range selection or placement; never reinterpret the clipped plateau as a distinctive activity feature.
**Evidence.** Retain the raw-range metadata and saturation count without claiming that a particular class caused it.
**Boundary.** Increasing a model score threshold does not restore information already lost to saturation.
**Owner.** The embedded lead owns range configuration, and the data lead owns calibration-provenance review.
**Review.** Slice clipping by participant, class, placement, and collection session before selecting any revised range.

**Scenario A05 — Orientation contract break.**
**Stimulus.** A board revision swaps the physical X and Y accelerometer routes while the tensor manifest still states X/Y/Z.
**Calculation.** The tensor remains shaped `[128,3]`, so shape validation alone cannot detect the semantic permutation.
**Diagnostic.** Run a golden vector whose expected channels differ, then compare every INT8 tensor position to the reference.
**Decision.** Reject the artifact-device pairing when channel identity, handedness, or gravity orientation differs.
**Recovery.** Update the declared acquisition transform and regenerate training-to-target parity evidence before enabling inference.
**Evidence.** Bind board revision, placement drawing, channel mapping, model hash, and golden-vector checksum.
**Boundary.** A model that returns a plausible label after an axis swap has not demonstrated semantic correctness.
**Owner.** Hardware and firmware jointly own the mapping; a model owner cannot infer it from aggregate accuracy.
**Review.** Recheck on every sensor SKU, board spin, orientation convention, and preprocessing release.

**Scenario A06 — Feature extraction drift.**
**Stimulus.** Offline training subtracts a training-set channel center, but firmware accidentally subtracts a center recomputed from the current window.
**Calculation.** The deployed feature becomes input-dependent even when the raw 128-frame sequence equals the training sequence.
**Diagnostic.** Compare intermediate scaled arrays before and after centering for a fixed golden vector.
**Decision.** Block release because runtime normalization now uses information unavailable to the declared training transformation.
**Recovery.** Restore the immutable training-derived center or train and validate a separately specified adaptive algorithm.
**Evidence.** Store center values, units, source partition, arithmetic order, and rounding semantics in the manifest.
**Boundary.** Per-window centering may be useful in another design, but it is not interchangeable with this model contract.
**Owner.** The data lead specifies the transform; firmware proves byte-level implementation parity.
**Review.** Add negative-valued and near-zero inputs to the parity suite to expose integer-division differences.

**Scenario A07 — Rounding-tie mismatch.**
**Stimulus.** A scaled channel value is `x=0.35 g` with `s=0.02 g/count` and zero point `z=0`.
**Calculation.** The unrounded quotient is `0.35 / 0.02 = 17.5`, so nearest-even and away-from-zero rules need explicit comparison.
**Diagnostic.** Nearest-even yields 18; dequantization gives `18 * 0.02 = 0.36 g`, an error of `-0.01 g`.
**Decision.** Do not accept a firmware result merely because its final class matches the float model on this one vector.
**Recovery.** Pin a rounding rule and validate both tensor bytes and score tolerances across tie values of both signs.
**Evidence.** Include the language/compiler conversion path and all tie-case expected bytes in the test artifact.
**Boundary.** The half-step error bound applies before saturation; it says nothing about clipped values.
**Owner.** The converter and firmware owners jointly approve the quantization arithmetic definition.
**Review.** Repeat after compiler upgrades, DSP-library changes, or replacement of integer arithmetic helpers.

**Scenario A08 — Input-order mismatch.**
**Stimulus.** The model expects oldest-to-newest frames, but a rolling buffer is copied from newest-to-oldest after a wrap.
**Calculation.** The input still contains 384 INT8 values, yet temporal order is reversed without a length or allocation error.
**Diagnostic.** Use a monotonic ramp vector where the expected first and last tensor rows are observably different.
**Decision.** Refuse the release when sequence order in the assembled tensor does not match the training specification.
**Recovery.** Correct the wrap-copy indices and rebuild the vector suite rather than reversing output labels post hoc.
**Evidence.** Retain q-start, q-end, buffer offsets, tensor checksum, and the row-order declaration.
**Boundary.** A stationary vector may pass this test while concealing reversal, so it is insufficient as the only parity vector.
**Owner.** Firmware owns ring traversal; the model manifest states temporal and axis ordering.
**Review.** Test windows that cross every ring boundary and every DMA block ownership transition.

**Scenario A09 — Flash-budget reconciliation.**
**Stimulus.** A candidate build reports 18 KiB model parameters, 46 KiB runtime/program code, and 4 KiB immutable constants.
**Calculation.** The stated total is `18 + 46 + 4 = 68 KiB`; against 128 KiB, nominal remaining flash is 60 KiB.
**Diagnostic.** Reconcile linker sections, alignment, duplicate model blobs, debug options, and bootloader reservation against this ledger.
**Decision.** Treat 60 KiB as planning headroom, not permission to add unreviewed logging or a second artifact.
**Recovery.** Remove unnecessary code or select a smaller supported model if the actual map exceeds the approved boundary.
**Evidence.** Archive the map file, build flags, artifact hash, and flash-region definition with the reviewed build.
**Boundary.** KiB accounting is invalid if one input is decimal kB or if the usable partition excludes bootloader space.
**Owner.** The embedded lead owns map interpretation; product owns any feature trade-off imposed by the result.
**Review.** Repeat for compiler flags, runtime operators, model conversion, localization, and update-storage changes.

**Scenario A10 — Arena peak overrun.**
**Stimulus.** A target allocator trace reports a 23 KiB inference peak while the design assumption reserved 22 KiB.
**Calculation.** The overrun is `23 KiB - 22 KiB = 1 KiB`, or 1024 bytes when KiB denotes 1024 bytes.
**Diagnostic.** Confirm that the trace exercised the selected operator path and did not omit error handling or initialization allocations.
**Decision.** Reject the configuration; a one-KiB miss is a hard static-memory constraint, not a rounding nuisance.
**Recovery.** Resize the arena within the complete SRAM ledger or change the model/runtime design and measure again.
**Evidence.** Retain allocator peak, tensor-arena address range, model hash, runtime version, and workload vector identity.
**Boundary.** Borrowing the general reserve without updating stack and DMA analysis hides rather than resolves the collision.
**Owner.** The embedded lead approves arena size only after target evidence, not desktop allocation behavior.
**Review.** Measure peak under every supported model package and each runtime/compiler configuration.

**Scenario A11 — Stack-margin erosion.**
**Stimulus.** The inference task's measured high-water use is 5,900 bytes against a 6,144-byte allocated stack.
**Calculation.** The remaining local margin is `6,144 - 5,900 = 244 bytes`, before any new error-path depth is introduced.
**Diagnostic.** Separate task stack usage from ISR stack use and collect maxima during normal, radio, logging, and fault paths.
**Decision.** Do not describe 244 bytes as sufficient without a stated required margin and representative stress evidence.
**Recovery.** Reduce call depth, allocate a bounded larger stack, or remove nonessential work from the task.
**Evidence.** Record watermark method, compiler settings, optimization level, interrupt nesting, and scenario duration.
**Boundary.** A map file reserves stack storage but cannot prove that a rare recursive or formatting path never reaches it.
**Owner.** Firmware owns stack measurement; reviewers challenge unmeasured paths rather than accepting nominal allocation.
**Review.** Repeat after logging, diagnostics, RTOS priority, compiler, or error-reporting changes.

**Scenario A12 — Freshness despite correct inference.**
**Stimulus.** The nominal local response is 70 ms, but a blocked UI adds 1,300 ms before the result is rendered.
**Calculation.** Age becomes `70 + 1,300 = 1,370 ms`, exceeding the 1,350 ms limit by 20 ms.
**Diagnostic.** Timestamp at UI use and compare `t_use - t_end`; do not substitute inference-completion time for UI-use time.
**Decision.** Render `UNAVAILABLE` even if the score was computed correctly and the source buffer remained intact.
**Recovery.** Expire the result, record the late reason, and remove the source of UI blocking before another eligibility attempt.
**Evidence.** Retain render-attempt traces including expired attempts so availability cannot be inflated by omission.
**Boundary.** Meeting the 1-second hop period does not guarantee freshness when a lower-priority UI waits longer.
**Owner.** UI scheduling and firmware timing owners share responsibility for this externally visible contract.
**Review.** Inject radio, logging, and display contention while preserving the actual capture-time timestamp.

**Scenario A13 — Sensing-charge accounting.**
**Stimulus.** The illustrative sensing state draws 0.50 mA continuously through one 60-second accounting interval.
**Calculation.** Its charge term is `0.50 mA * 60 s = 30 mA·s` during that interval.
**Diagnostic.** Confirm that sensing current is measured or explicitly assumed separately from the processor sleep state.
**Decision.** Keep the 30 mA·s term distinct so a change in sensor ODR can be attributed rather than hidden in an average.
**Recovery.** Replace this constructed term with synchronized measured current and occupancy before making a battery statement.
**Evidence.** Preserve instrument bandwidth, regulator path, temperature, battery condition, and state-marker timestamps.
**Boundary.** A 0.50 mA steady-state assumption excludes startup spikes and does not predict aged-cell capacity.
**Owner.** Power measurement ownership belongs to the embedded team, while product controls any external runtime claim.
**Review.** Recompute after rate, range, bus traffic, power mode, or sensor hardware changes.

**Scenario A14 — Inference-duty accounting.**
**Stimulus.** The local model runs for 24 ms once per second at an illustrative 6.0 mA active current.
**Calculation.** Over 60 seconds it is active for `24 ms * 60 = 1,440 ms = 1.44 s`, contributing `6.0 * 1.44 = 8.64 mA·s`.
**Diagnostic.** Separate measured execution occupancy from wake overhead, cache effects, and other work sharing the same power state.
**Decision.** Use the term only as an illustrative component of the stated average-current model.
**Recovery.** If target timing grows, remeasure both duration and current instead of scaling a desktop benchmark.
**Evidence.** Bind current trace segments to model hash, clock frequency, compiler build, and representative input workload.
**Boundary.** The charge calculation assumes constant active current; dynamic voltage or burst behavior invalidates that simplification.
**Owner.** The embedded lead maintains the state definitions and their instrumentation markers.
**Review.** Repeat for quantization changes, operator selection, clock policy, and runtime-library updates.

**Scenario A15 — Radio back-pressure isolation.**
**Stimulus.** Four summary records are queued and a fifth arrives while the optional-radio queue capacity is four.
**Calculation.** The fifth record has no free queue slot; capacity `4 - 4 = 0` is an admission result, not a latency estimate.
**Diagnostic.** Trace whether the enqueue path touches DMA blocks, rolling-window storage, inference tensors, or UI publication.
**Decision.** Drop the summary under its bounded policy and keep local classification work schedulable.
**Recovery.** Increment a summary-drop counter and transmit no raw frames as a substitute for the omitted summary.
**Evidence.** Record queue depth, enqueue time, retry count, transaction duration, and the field schema for every retained summary.
**Boundary.** Enlarging the radio queue can consume SRAM and still cannot justify delaying a freshness-critical local result.
**Owner.** Telemetry owns its queue contract; it cannot borrow acquisition ownership to improve delivery statistics.
**Review.** Induce network loss and repeated retries while checking that display-age and block-release invariants remain true.

**Scenario A16 — Availability denominator.**
**Stimulus.** An illustrative evaluation offers 300 labeled candidate windows, of which 240 are eligible and 216 produce fresh outputs.
**Calculation.** Fresh availability among eligible windows is `216 / 240 = 0.90`, while fresh coverage over all candidates is `216 / 300 = 0.72`.
**Diagnostic.** Report both denominators with the exclusion reasons instead of calling either figure unqualified accuracy.
**Decision.** Do not accept a model comparison unless threshold and model use the same candidate and eligibility definitions.
**Recovery.** Investigate whether exclusions arise from labels, quality, timing, or implementation before changing a classifier threshold.
**Evidence.** Retain candidate IDs, eligibility flags, freshness status, reason codes, and split assignment for each denominator.
**Boundary.** A high conditional availability can coexist with poor overall coverage when many candidate windows are rejected.
**Owner.** The evaluation owner defines denominators before final-test access; firmware supplies auditable reason codes.
**Review.** Slice availability by participant, session, class, placement, and induced-fault condition.

**Scenario A17 — Macro-recall interpretation.**
**Stimulus.** Illustrative class recalls on identical eligible held-out windows are 0.90 for STILL, 0.80 for WALK, 0.60 for STAIRS, and 0.70 for OTHER.
**Calculation.** Macro recall is `(0.90 + 0.80 + 0.60 + 0.70) / 4 = 0.75`.
**Diagnostic.** Inspect each class denominator because a macro mean gives equal class weight regardless of recording duration.
**Decision.** Report the 0.75 as illustrative only and expose the 0.60 STAIRS slice rather than hiding it in a pooled score.
**Recovery.** Review labels, placement, training coverage, and error examples before proposing any class-specific remediation.
**Evidence.** Store per-class confusion counts, participant identifiers, eligibility flags, and the frozen final-test manifest.
**Boundary.** Macro recall is undefined for a missing held-out class unless the predeclared handling rule addresses that denominator.
**Owner.** The data lead owns metric specification; product decides whether the weakest class meets the descriptive feature purpose.
**Review.** Use participant-level intervals or resampling only with an explicitly stated uncertainty method and scope.

**Scenario A18 — Margin is not probability.**
**Stimulus.** A model emits scores `[0.40, 0.36, 0.14, 0.10]` after an INT8 runtime invocation.
**Calculation.** The top-two score margin is `0.40 - 0.36 = 0.04`; against an illustrative margin rule of 0.08, it is insufficient.
**Diagnostic.** Verify class order and score representation before interpreting a margin, because raw logits and calibrated probabilities differ.
**Decision.** Produce `UNAVAILABLE` with a low-margin reason if that rule was selected on validation participants before final test.
**Recovery.** Preserve the rejected result for aggregate reason accounting but never display the previous accepted class in its place.
**Evidence.** Version the margin threshold, validation source, score type, class mapping, and artifact hash together.
**Boundary.** A large margin can still be wrong under distribution shift, so it is not a safety guarantee.
**Owner.** The model policy owner chooses the gate; UI and firmware owners ensure its abstention semantics are honored.
**Review.** Compare margin distributions across classes and placement conditions without using final-test results to retune the gate.

**Scenario A19 — Epoch reset recovery.**
**Stimulus.** A watchdog reset occurs after 73 valid frames have accumulated for the current rolling window.
**Calculation.** The prior 73 frames are fewer than the required 128, and even a full prior window would belong to the old epoch.
**Diagnostic.** Confirm that sequence comparisons, timestamps, and UI result records include an epoch identifier.
**Decision.** Discard partial history and suppress prior labels immediately when the epoch changes.
**Recovery.** Require 128 consecutive valid frames in the new epoch before a new candidate window can become eligible.
**Evidence.** Retain reset cause, old and new epoch values, first new sequence, and first post-reset accepted result.
**Boundary.** Continuing a modulo sequence counter across reset can make unrelated samples look contiguous.
**Owner.** Firmware owns reset behavior; product must accept that a short unavailable interval is safer than false continuity.
**Review.** Induce watchdog, power-cycle, clock-reset, and counter-wrap events in the target trace suite.

**Scenario A20 — Safety-purpose escalation.**
**Stimulus.** A stakeholder asks the descriptive activity feature to trigger a fall alert when unusual motion is observed.
**Calculation.** No arithmetic conversion turns the four display labels into a validated detector for an event with false-negative consequences.
**Diagnostic.** Compare intended action, escalation recipient, response deadline, hazard severity, and required evidence against the current contract.
**Decision.** Reject the request as out of scope and create a separate safety decision definition before collecting or reusing evidence.
**Recovery.** Retain local activity feedback only; do not silently add alerts, emergency copy, or automated communications.
**Evidence.** Document the request, hazard-analysis owner, validation population, and decision to defer within the relevant governance record.
**Boundary.** Showing `UNAVAILABLE` is a safe degraded behavior for this descriptive feature, not a complete safety-control strategy.
**Owner.** The product decision owner escalates to safety, clinical, and regulatory review where applicable.
**Review.** Reopen the case only when purpose, action path, labels, evidence plan, and operating constraints are all explicitly redefined.

## 15. Quick Reference Card

| Field | Card |
|---|---|
| Decision | Prototype local INT8 classification only behind integrity, freshness, and held-out-participant gates. |
| Owner | Product manager decides feature; embedded lead owns firmware feasibility; data lead owns labels/splits. |
| Input / output | `[128,3]` timestamped IMU window -> label or `UNAVAILABLE`. |
| Baselines | Fixed-rate upload/manual review; magnitude threshold. |
| Critical numbers | 50 Hz; 128 frames; 2.56 s window; 1.00 s hop; 60.9375% overlap; 1.35 s freshness limit. |
| Resource model | 948-byte DMA blocks total; 36,916-byte SRAM ledger; 28,620-byte illustrative headroom. |
| Integrity invariant | DMA writes only FREE/FILLING storage; consumer reads only completed, consecutive sequences. |
| Failure rule | Gap, overrun, manifest mismatch, or stale result -> invalidate and show `UNAVAILABLE`. |
| Evidence | Illustrative and not-measured; source bodies, data, artifact, target traces, and battery measurement remain OPEN. |
| Stage / review | Proposed intervention; draft document; review on any sensor, model, protocol, radio, battery, or population change. |
