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

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** Ask for the split manifest before model code.

**Acceptable alternatives:** Session split may supplement participant split but cannot replace it when new-person generalization matters.

**Red flags:** Saying shuffle is enough.

**Scoring:** 2: names identity leakage; 1: mentions overfit; 0: random split only.

**Answer support:** Section 5a.

#### Q2. Compute the overlap for N=128 and H=50.

**Checked answer:** Overlap is (128-50)/128=78/128=60.9375%; 1.56 seconds of each 2.56-second window recurs.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** What changes if H=128?

**Acceptable alternatives:** Non-overlap reduces repeated evidence but lowers decision cadence.

**Red flags:** Calling hop the overlap.

**Scoring:** 2: correct formula and units; 1: correct number only; 0: wrong denominator.

**Answer support:** Section 6.1.

#### Q3. What makes an activity label non-diagnostic?

**Checked answer:** It describes a motion class only and must not infer disease, fitness, adherence, impairment, or risk.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** Who reviews wording?

**Acceptable alternatives:** A human-factors/product review may reject misleading wording.

**Red flags:** Use cautious wording plus explicit limitations. Equating a motion class with health status.

**Scoring:** 2: distinguishes purpose; 1: says disclaimer; 0: claims medical meaning.

**Answer support:** Sections 1 and 3.

#### Q4. Why is a DMA completion interrupt not permission to retain a pointer forever?

**Checked answer:** Completion transfers a bounded block; circular DMA will reuse it after two future 1-second blocks with K=3.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** What happens at 2010 ms?

**Acceptable alternatives:** The block may be being written again, so invalidate rather than read.

**Red flags:** Copy into owned storage if its cost is budgeted. Counting K periods instead of K-1.

**Scoring:** 2: explains state/reuse; 1: says race; 0: relies on volatile pointer.

**Answer support:** Section 6.4.

#### Q5. How do you calculate result freshness?

**Checked answer:** Subtract window end time from UI-use time; CPU time alone omits queue and UI delay.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** What is the response at 1360 ms?

**Acceptable alternatives:** It is expired against 1350 ms and must display UNAVAILABLE.

**Red flags:** Use a monotonic shared clock or bounded conversion. Checking age only at inference finish.

**Scoring:** 2: correct origin and action; 1: time mention; 0: CPU-only.

**Answer support:** Sections 4 and 8.

#### Q6. What could make INT8 accuracy fail although float validation passes?

**Checked answer:** A wrong scale, zero point, axis order, rounding rule, calibration range, or saturation pattern changes inputs or kernels.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** What artifact detects it?

**Acceptable alternatives:** Golden vectors and a quantization manifest bind expected values.

**Red flags:** QAT or a smaller model may be considered after diagnosis. Assuming INT8 is lossless.

**Scoring:** 2: names contract plus test; 1: says quantization error; 0: retrain blindly.

**Answer support:** Section 6.3.

#### Q7. What is the stated SRAM headroom?

**Checked answer:** The ledger totals 36,916 bytes; 65,536 minus that equals 28,620 bytes, before proving target stack peak.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** Does that prove fit?

**Acceptable alternatives:** No; alignment, runtime allocations, DMA reachability, and measured stack peak remain open.

**Red flags:** Use static allocation and map plus watermark evidence. Calling nominal remaining bytes a guarantee.

**Scoring:** 2: arithmetic and limitation; 1: arithmetic; 0: unsupported guarantee.

**Answer support:** Section 6.2.

#### Q8. Why compare the threshold baseline on the identical eligible windows?

**Checked answer:** Different exclusions create different denominators, so an apparent model gain may be selection rather than classification improvement.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** What if the baseline cannot emit STAIRS?

**Acceptable alternatives:** Report its defined task honestly or retain it as a two-class comparator.

**Red flags:** A manual-review comparator can answer a different latency question. Comparing headline accuracy from different datasets.

**Scoring:** 2: identifies denominator; 1: says fairness; 0: accepts different windows.

**Answer support:** Sections 5c and 9.

#### Q9. How is a label interval created?

**Checked answer:** Instruction, visible time marker, participant confirmation, reviewer adjudication, and transition exclusion define the interval.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** Why exclude transitions?

**Acceptable alternatives:** A mixed transition lacks one activity target.

**Red flags:** Use an ambiguity label only if it has a collection rule. Assigning the facilitator's guess to every frame.

**Scoring:** 2: gives protocol and boundary; 1: mentions annotation; 0: ignores uncertainty.

**Answer support:** Section 5a.

#### Q10. What changes if radio active time is ten times larger?

**Checked answer:** From 40 to 400 ms/minute, the stated model gives 0.81247 mA and 6.1541 illustrative days instead of 0.70477 mA and 7.0945 days.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** Does this establish battery life?

**Acceptable alternatives:** No; it omits discharge and measured state current.

**Red flags:** Reduce summary rate or transaction cost after measurement. Claiming the arithmetic is a benchmark.

**Scoring:** 2: computes and limits; 1: direction only; 0: ignores duty.

**Answer support:** Section 6.5.

#### Q11. Diagnose a stream with duplicated end times but no sequence gaps.

**Checked answer:** Possible causes include timestamp assignment in the ISR, clock resolution, or consumer metadata reuse; inspect raw q/time pairs and ownership transitions.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** What would distinguish sensor replay?

**Acceptable alternatives:** A sensor replay may repeat payload values with advancing q; metadata reuse repeats headers inconsistently.

**Red flags:** Fail closed to UNAVAILABLE while tracing. Immediately blame the model.

**Scoring:** 2: competing causes and check; 1: one cause; 0: retrain.

**Answer support:** Section 12.

#### Q12. A manager asks for fall alerts from this classifier. What changes?

**Checked answer:** That is a separate safety/clinical decision with different labels, false-negative consequences, escalation, validation, and likely regulatory analysis; this case cannot authorize it.

**Reasoning:** The answer follows from the stated contract rather than a device benchmark.

**Follow-up:** Can the current label be reused?

**Acceptable alternatives:** Only as an unvalidated research input, not as evidence of fall performance.

**Red flags:** Defer or create a new bounded case. Adding a FALL class to the current UI.

**Scoring:** 2: identifies changed contract; 1: says out of scope; 0: treats as class addition.

**Answer support:** Sections 1 and 13.

### Operational evidence checklist

These controls are proposed evidence obligations, not claims that they ran.

### Evidence control 01: Acquisition Trace

**Record.** Record timer tick, DMA completion tick, block index, epoch, and sequence for every completion.

**Decision relevance.** A monotonic sequence makes a dropped block observable even when payload values happen to repeat.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 02: Timestamp Convention

**Record.** Define window end as the capture time of frame 128, not DMA interrupt arrival.

**Decision relevance.** Interrupt latency is processing delay; it cannot move the physical observation forward in time.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 03: Axis Contract

**Record.** Record sensor orientation convention and whether gravity is retained.

**Decision relevance.** Changing orientation preprocessing at inference makes a new model input space.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 04: Range Contract

**Record.** Record configured full scale and count-to-g conversion.

**Decision relevance.** A range change without scale revision silently rescales every feature.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 05: Label Consent

**Record.** Record consent version and withdrawal handling before collection.

**Decision relevance.** A withdrawn session must be excluded through the manifest rather than silently retained.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 06: Session Boundary

**Record.** Record donning, removal, charging, and clock-reset events.

**Decision relevance.** A session boundary prevents physical discontinuity from masquerading as continuous behavior.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 07: Transition Policy

**Record.** Mark starts, stops, turns, and instruction changes as excluded intervals.

**Decision relevance.** The exclusion rule must be set before reviewing model errors.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 08: Class Balance

**Record.** Report labeled seconds and eligible windows by class and participant.

**Decision relevance.** Window counts alone overweight long recordings and repeated overlap.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 09: Missingness

**Record.** Report missing frames, not just missing windows.

**Decision relevance.** A filled-in frame is fabricated motion unless a separate imputation method is validated.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 10: Threshold Policy

**Record.** Bind score threshold and margin to a validation-only revision.

**Decision relevance.** Selecting it after reading test errors is test-set tuning.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 11: Ui Policy

**Record.** Render the capture interval and unavailable reason category where appropriate.

**Decision relevance.** A label without time semantics invites an assumption of real-time certainty.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 12: Radio Policy

**Record.** Bound payload, attempts, and queue depth independently from acquisition.

**Decision relevance.** Telemetry is expendable; acquisition integrity is not.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 13: Power Trace

**Record.** Measure current with state markers that identify sensing, inference, radio, and sleep.

**Decision relevance.** An average current without state attribution cannot guide a design change.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 14: Arena Evidence

**Record.** Capture allocator peak and failure behavior on target build.

**Decision relevance.** A desktop allocation report is not SRAM evidence.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 15: Stack Evidence

**Record.** Measure high-water marks during overload and error paths.

**Decision relevance.** Normal-path stack use does not bound exception/logging paths.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 16: Golden Vectors

**Record.** Store representative raw frames, expected quantized tensor, and expected scores.

**Decision relevance.** Golden vectors locate preprocessing drift before aggregate accuracy hides it.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 17: Model Identity

**Record.** Bundle model hash, class order, input shape, and quantization manifest atomically.

**Decision relevance.** A model blob without metadata has no stable interpretation.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 18: Update Boundary

**Record.** Reject partial updates and record accepted artifact version.

**Decision relevance.** Mixing old scales with new weights is a silent semantic failure.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 19: Overrun Injection

**Record.** Delay the consumer until a ready queue fills, then assert no stale label renders.

**Decision relevance.** A performance test that never reaches overflow does not exercise recovery.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 20: Clock Reset

**Record.** Inject reset during a window and require epoch change plus invalidation.

**Decision relevance.** Sequence wrap and reset must not be confused.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 21: Participant Report

**Record.** Report median and range across test participants, not only pooled score.

**Decision relevance.** A pooled metric can hide a consistently failing wearer subgroup.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 22: Placement Shift

**Record.** Test declared placement changes separately from intended placement.

**Decision relevance.** A shift result is a limitation, not evidence of normal performance.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 23: Free-Living Boundary

**Record.** Keep free-living OTHER separate from scripted activity labels.

**Decision relevance.** Unreviewed free-living recordings cannot become clean labels by default.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 24: Manual Comparator

**Record.** Time manual review from capture to review decision and state its coverage.

**Decision relevance.** Manual review is not automatically a real-time baseline.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Evidence control 25: Privacy Boundary

**Record.** Document whether raw samples, summaries, and identifiers leave the device.

**Decision relevance.** Local inference does not itself prove minimal data collection.

**Test.** Capture both a conforming trace and a deliberately violated trace; identify the first field or state transition that distinguishes them.

**Denominator.** State whether the evidence covers all attempted windows, all participant sessions, calibration-only inputs, or one injected fault. A success count without this denominator is not a rate.

**Containment.** A missing, contradictory, or out-of-scope record makes the affected result ineligible; preserve its reason code and show `UNAVAILABLE` rather than inheriting a prior label.

**Recheck.** Reopen this evidence when the producing sensor, data protocol, model artifact, compiler/runtime, population, or energy/radio policy changes.

### Cross-control decision boundaries

A successful model score is eligible only when acquisition, sequence, label provenance, tensor manifest, and display age all pass; eligibility is an AND contract, not an average.

A single failed boundary must name the earliest failed condition so that firmware repair does not become a data-science guessing exercise.

When a condition is unknown rather than failed, record `UNKNOWN` separately from `PASS` and `FAIL`; an unknown denominator must not be silently counted as success.

A recovery trace begins at the discontinuity, clears the rolling window, obtains 128 contiguous frames, and only then permits a new label.

The local UI path has priority over optional telemetry because a delayed summary is a data-loss cost, while a delayed label changes the meaning of the feature.

### Boundary ledger for release review

**Window identity:** The release reviewer traces a displayed label backward through `q_end`, epoch, and end time to the immutable input interval; a label without that chain is ineligible.

**Split identity:** The data reviewer checks that every participant is assigned once, every session inherits that assignment, and every derived window carries the same partition.

**Label identity:** The protocol reviewer checks that every retained label interval has a prompt, confirmation or adjudication source, and an exclusion decision at both boundaries.

**Tensor identity:** The embedded reviewer compares axis order, units, scale, zero point, tensor shape, class order, and model hash against the approved manifest before inference is enabled.

**Memory identity:** The build reviewer reconciles DMA storage, staging, arena, stacks, queue objects, alignment, and reserve against the linker map rather than relying on source declarations.

**Timing identity:** The scheduler trace records completion-to-render age and block-release time separately; passing the reuse bound does not pass the freshness bound.

**Energy identity:** The power trace ties each measured current region to sensing, preprocessing, inference, radio, or sleep, so a change in average current has an actionable cause.

**Abstention identity:** The UI records whether `UNAVAILABLE` arose from quality, sequence, overflow, artifact, or freshness; these reasons must not be pooled into an unexplained accuracy exclusion.

**Comparator identity:** The threshold, local model, and upload/manual alternative use the same participant-held-out intervals when their descriptive-label quality is compared.

**Scope identity:** Any request to infer health, emergency, adherence, productivity, or fall state triggers a new decision contract rather than expanding the meaning of `WALK`.

**Recovery identity:** After an epoch change, the first permitted decision follows a newly assembled 128-frame window; partial pre-reset history has no owner.

**Radio identity:** A retry counter belongs to the summary transaction, not to the local label, so communication failure cannot create duplicate activity decisions.

**Evidence identity:** Illustrative calculations remain labeled as assumptions until a named target trace, measurement method, and population make a narrower observed claim possible.

**Decision identity:** The owner may choose the baseline when any ledger entry is unresolved; deferral is a valid result, not a hidden implementation failure.
This ledger makes the rejection reason inspectable at the same granularity as the claimed label.

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
