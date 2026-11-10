---
collection: "healthcare-life-sciences-biomedical-ai"
file_id: "02"
topic: "Biomedical signal pipeline and validation"
level: 1
status: "draft — 12-section substantive manuscript; course line threshold met; acquisition, clinical-validity, and independent-review evidence remain OPEN"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Sensor, sampling protocol, acquisition population, label definition, or preprocessing version changes"
notation_style: "plain-text"
prerequisites: ["01_Clinical_Decision_Support_Evidence_and_Handoff.md", "sampling basics"]
pattern_tags: ["biomedical-signals", "signal-quality", "point-in-time-provenance"]
source_status: "OPEN — original biomedical-signal source body unavailable"
owner_role: "Clinical signal-science lead"
source_index: "INDEX.md"
---

# Biomedical Signal Pipeline and Validation

> **Purpose:** Turn a sampled physiological waveform into a traceable analysis input while preserving units, timing, sensor quality, and clinical review boundaries.
>
> **Prerequisites:** [clinical decision-support evidence](01_Clinical_Decision_Support_Evidence_and_Handoff.md), sampling terminology, and local acquisition protocol.
>
> **Key outcome:** Given sampling rate, duration, channel metadata, and an intended decision, you can calculate sample count, specify a quality gate, and reject a preprocessing path that changes the target meaning.
>
> **This file does NOT cover:** diagnosis from any waveform, medical-device validation, or a claim that a source organization used a particular signal pipeline.
>
> **Level:** 1 — Foundations.
>
> **Evidence:** DERIVED sampling relations; VERIFIED illustrative arithmetic; OPEN acquisition and target-specific evidence.
>
> **Retrieval cue:** If a model is trained on “ECG data” with no timing, lead, or quality record, start at Sections 3, 4, and 8.
>
> **Technical handoff:** The detailed ECG sampling/aliasing, anatomy, filter, QRS/HRV, PPG/SpO2, FFT/wavelet, IMU, dataset, and edge-feature commitments belong to [Module 37's biomedical source brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-signals), not this cross-domain validation sheet. The existing [ECG sampling and aliasing draft](../../../07_edge_ai_robotics_and_cyber_physical_systems/37_sensor_fusion_iot_and_realtime_perception/Cheat_Sheets/01_ecg_sampling_and_aliasing.md) covers only part of that planned source scope.

## 1. File Header

A biomedical signal is not just an array. Its meaning depends on sensing modality, channel identity, physical units, calibration, body placement, acquisition conditions, timestamps, filtering, segment selection, and label linkage. A pipeline should preserve enough metadata to reconstruct whether a value is physiological variation, sensor artifact, or an irreversible processing choice.

**Takeaway:** A waveform without acquisition context is an ambiguous measurement, not a reproducible model input.

### 1.1 Decision unit and observation unit

Choose two units before writing the schema. The observation unit is the measured stream, such as one named channel from one sensor. The decision unit is the entity receiving an output, such as a ten-second encounter segment. Many observation units can contribute to one decision unit, but the aggregation rule must be declared.

| Contract item | Required statement | Example for this teaching pipeline | Invalid shortcut |
|---|---|---|---|
| subject scope | de-identified research subject or governed encounter reference | encounter key held by an authorized service | joining by name or room alone |
| record | acquisition instance with a start and end | `R01`, with source timestamps | file name without acquisition time |
| channel | named physical measurement and placement | `ECG-II`, microvolts after calibration | column 2 means lead II |
| segment | half-open time interval plus selection rule | `[09:00:00,09:00:10)` in the record clock | first 2,500 values are always ten seconds |
| feature | versioned function of permitted inputs | quality count over the segment | function of the whole retrospective record |
| output | bounded support or research result | quality pass with reasons | clinical-normal assertion |
| target | independently defined evaluation reference | adjudication available at 16:00 | quality flag treated as disease label |

The decision unit determines what information may be used. A later segment may inform a retrospective label, but it cannot silently enter a feature intended to be available at the earlier decision time. Conversely, excluding all future information from an offline training label would confuse a legitimate outcome definition with feature leakage. Store the label availability time and its permitted role.

### 1.2 Minimum acquisition envelope

A usable raw reference includes more than numeric samples. Keep the following fields together or connect them through durable identifiers:

- acquisition protocol and protocol version;
- sensor modality, model-independent device identifier, and firmware/schema version;
- channel names, placement protocol, order, physical units, and polarity convention;
- calibration status, calibration version, and any conversion from digital counts;
- nominal sampling rate and actual timestamp representation;
- clock source, time zone where applicable, and synchronization status;
- valid value range, saturation encoding, missing-value encoding, and disconnected-channel behavior;
- record start/end, segment boundaries, and whether endpoints are inclusive;
- raw-artifact checksum or governed storage reference;
- transformation code/configuration version and execution identity;
- access scope, retention class, and deletion linkage;
- quality result, reason codes, reviewer/adjudicator role where applicable, and disposition;
- target definition and target availability time, stored separately from features.

Do not invent a value for an unavailable field. Represent it as unknown and evaluate whether the intended use permits unknown provenance. A missing unit cannot safely be repaired by guessing from amplitude because amplitude can change with placement, gain, calibration, and physiology.

### 1.3 Stage ownership and rejection boundary

| Stage | Accountable role | May decide | Must not silently decide |
|---|---|---|---|
| acquisition design | signal-science and protocol owners | rate, band, channel map, calibration procedure | clinical usefulness of an unvalidated target |
| ingestion | data/platform owner | schema acceptance, timestamp parsing, immutable raw reference | that schema acceptance implies physiological validity |
| signal quality | protocol owner with domain review | analyzability under explicit quality rules | diagnosis or outcome status |
| feature generation | pipeline/model owner | transform and feature versions within approved protocol | target redefinition after observing errors |
| model evaluation | evaluation owner | split, metric, uncertainty, baseline, exclusions | deployment authorization from a single aggregate score |
| operational handoff | accountable workflow owner | review route, timeout, fallback, escalation | automatic consequential action without authorization |

An unknown required field should stop the affected route rather than be converted to zero. Zero is a measurement; unknown is a provenance state. The same distinction applies to missing samples and saturated samples: neither is interchangeable with a physiological zero.

### 1.4 Evidence ledger for this sheet

The relations for sample counts, payload sizes, intervals, and missing fractions are DERIVED and their illustrative arithmetic is VERIFIED. Device bandwidth, filter performance, calibration error, sensor reliability, clinical validity, and population transportability remain OPEN until supported by primary studies and the local protocol. This sheet does not infer those facts from the filename of an unavailable source.

A review record should name what was inspected, what was executed, and what remains unobserved. For example, recomputing `250*10=2,500` checks arithmetic only. It does not show that a device actually sampled uniformly, that twelve sensors were attached, or that a downstream score is clinically meaningful.

## 2. Concept Map / Mental Model

```text
[subject + protocol + sensor placement]
             |
             v
[analog physiological signal] -> [sensor/calibration] -> [sampled channels]
             |                                              |
             |                                      quality gate fails
             |                                              v
             |                                     [repeat / exclude / human review]
             v
[versioned preprocessing] -> [segmentation] -> [feature/model input]
                                                   |
                                                   v
                                [bounded output + clinician/research review]
                                                   |
                                                   v
                          [label/outcome stored separately with time provenance]
```

The pipeline is a chain of transformations. Each transformation needs an input contract and an output contract: sample rate, unit, channel order, missing-data encoding, clock, and version. Do not overwrite raw material with filtered material; link derived artifacts to a governed raw reference when policy permits.

**Takeaway:** The quality gate belongs before interpretation, and the raw-to-derived lineage belongs beside every model result.

### 2.1 Pipeline mechanism and invariants

**Preconditions:** an authorized acquisition reference, declared clock/channel/unit contracts, and a versioned transformation are available. **Input:** immutable raw channels and acquisition metadata. **Output:** derived samples, segment assignments, quality decisions, and a lineage record. **Termination:** every segment reaches pass, fail, abstain, or an explicit pending state; a timeout cannot become pass.

| Invariant | Implementation consequence | Test that can falsify it |
|---|---|---|
| raw immutability | a new transform writes a new derived version | raw checksum differs after preprocessing |
| channel identity | names and mapping version accompany every matrix | reordering columns changes output without a mapping error |
| temporal identity | every sample/segment maps to the declared clock | a dropped interval changes duration but timestamps are unchanged |
| unit consistency | calibration conversion declares input/output units | integer counts are passed to a microvolt-calibrated feature |
| transform reproducibility | code, parameters, dependency version, and input IDs are recorded | the same inputs/config produce a different unrecorded result |
| quality visibility | rejected and imputed intervals retain masks/reasons | the exported derived array has no indication of interpolation |
| label separation | targets carry definition and availability time | target fields appear in a feature table before their availability |
| bounded fallback | fail/abstain has a governed next route | an invalid segment receives a default negative model result |

A transform version must change when its behavior changes. Reordering columns, changing a gap threshold, replacing interpolation, or fitting a different scaler are behavioral changes even if the source code filename stays the same. A version label alone is insufficient if it does not identify the effective configuration.

### 2.2 Half-open segmentation

For fixed-duration, nonoverlapping windows of length `w` seconds, define segment `k` as `[t_0+k*w,t_0+(k+1)*w)`. The start is included and the end excluded, so adjacent windows do not duplicate a boundary sample. Record whether windows that extend beyond available data are dropped, padded, or marked partial.

For an online decision, a trailing window ending at `t_d` must use only observations available by the permitted processing deadline. If the device buffers two seconds before transmission, a waveform whose last physical sample precedes `t_d` may still be unavailable at `t_d`. Store both the observation interval and ingestion/availability timestamps.

Overlapping windows can improve temporal coverage but create correlated training examples. Split by subject, encounter, acquisition session, or another independent unit before windowing when overlap or shared context can cross a boundary. Randomly splitting windows first can put near-duplicates in training and evaluation.

### 2.3 Quality routing as a state machine

```text
RECEIVED
  | schema/identity accepted             | required field unknown/invalid
  v                                      v
QUALITY_EVALUATION ------------------> REJECTED_OR_ABSTAINED
  | all required checks pass                  |
  v                                           | governed recovery
ANALYZABLE                                    v
  |                                  reacquire / review / baseline route
  v
FEATURES_AND_BOUNDED_OUTPUT
  |
  v
REVIEW_AND_DISPOSITION
```

Every transition records a reason and policy version. Reacquisition creates a new record linked to the failed attempt; it does not overwrite the failure. Human review can resolve metadata ambiguity within the reviewer's authority, but it cannot make an unmeasured waveform observed. A baseline route must have its own input and safety contract rather than being an untracked manual workaround.

## 3. Foundational Definitions

| Term | Meaning | Units/type | Boundary |
|---|---|---|---|
| sampling frequency `f_s` | samples acquired per second per channel | hertz, samples/s | not the highest reliably interpretable physiology frequency by itself |
| sampling interval `Delta t` | elapsed time between ideal consecutive samples | seconds/sample | may vary with jitter or dropped samples |
| segment | a bounded time interval with inclusion rule | array plus start/end timestamps | not “one row” without time semantics |
| channel/lead | explicitly identified sensing stream | named vector | channel order alone is unsafe |
| artifact | non-target signal caused by sensor, motion, electrical/environmental or processing effects | event/flag | not automatically noise that may be discarded |
| reference label | independently specified target/annotation and adjudication process | label plus provenance | a future outcome silently joined to a segment |
| aliasing | indistinguishable lower-frequency representation caused by inadequate sampling/filtering | mechanism | repaired after sampling by a model |

A quality flag is different from a target label. It reports whether a segment meets a predefined analysis contract; it must not be learned as a disguised clinical outcome.

### 3.1 Units, gain, and calibration

A digital sample may initially be an integer ADC count rather than a physical quantity. If an approved linear conversion is `x=a*q+b`, then `q` is a raw count, `a` has physical-units per count, `b` has the output physical unit, and `x` is calibrated output. This is a local calibration model, not a universal device formula. Offset, gain, nonlinearity, clipping, and valid range need verification against the device/protocol.

An identity test is useful but incomplete: applying known reference inputs can expose a swapped scale or offset, while agreement at one reference level does not establish linearity across the valid range. Preserve calibration values and their applicable time/version. Applying two conversions because a downstream stage assumed raw counts can produce plausible but wrong values.

### 3.2 Missing, flat, saturated, and disconnected states

| State | Observable signature | Competing explanations | Required treatment |
|---|---|---|---|
| missing | absent timestamp/sample or explicit missing marker | packet loss, acquisition gap, export defect | preserve gap and physical duration; apply gap policy |
| flatline | repeated constant values | disconnected lead, true low variation, stuck converter | check device state, calibration, and protocol; never auto-diagnose |
| saturation | values at declared extrema | excessive gain, out-of-range physiology, invalid calibration | flag clipped intervals; do not rescale into apparent normality |
| disconnected | device/status evidence plus invalid signal | lead off, sensor fault, schema misread | block affected channel and route under protocol |
| interpolated | derived values replacing a gap | visualization smoothing, feature requirement | preserve mask, method, version, and original extent |
| duplicate | repeated timestamp/value block | retry, replay, export join, frozen stream | compare acquisition ID and timestamp sequence before deduplicating |

A flatline detector is a quality heuristic, not proof of disconnection. A short constant interval may be legitimate at the precision of the instrument; a varying signal can still come from an improperly placed sensor. Combine independent status and protocol evidence rather than treating one threshold as a diagnosis.

### 3.3 Sampling clocks and time bases

Distinguish device time, gateway receive time, server ingest time, and target availability time. A nominal rate does not guarantee exact intervals. When measured timestamps exist, calculate actual gaps and inspect ordering, duplicates, discontinuities, and synchronization quality. When they do not, label the grid as nominal and restrict timing claims accordingly.

Resampling onto a common grid can simplify downstream computation but requires an interpolation model and a declared time origin. It cannot recover missing physical information. Resampling separate channels without accounting for clock offsets may create artificial alignment, and synchronizing to a future timestamp can violate an online feature boundary.

### 3.4 Reference labels and annotation windows

A reference label needs a target population, annotation unit, window, adjudication process, and availability timestamp. “Positive for the recording” may not mean “positive for every ten-second segment.” Propagating an encounter label to all windows assumes constant target status and can manufacture false labels.

Multiple annotators require a reconciliation protocol; majority vote and expert adjudication are different label-generating processes. Inter-rater agreement can describe annotation consistency but does not by itself establish truth or downstream utility. Preserve disagreement where it affects eligibility, weighting, or uncertainty analysis.

**Takeaway:** Define the clock, unit, channel, segment, and label lineage before extracting features.

## 4. Core Formulas with Full Derivation

### 4.1 Base: sample count

**Validity before use:** The relation below assumes a constant nominal sampling frequency, one channel, a duration measured in the same time unit, and no drops. For irregularly sampled data, use timestamps rather than fabricating a uniform grid.

A constant rate of `f_s` samples/s accumulated for `T` seconds gives rate multiplied by time:

```text
FORMULA [SIG-01]: n = f_s * T
WHERE: n = samples per channel [count]; f_s = sampling frequency [samples/s];
       T = duration [s].
CONSTRAINT: fixed-rate acquisition; count excludes metadata and assumes no missing samples.
NUMERICAL EXAMPLE: f_s=250 samples/s; T=10 s; n=250*10=2,500 samples/channel.
INTERPRETATION: a 10-second, 12-channel record contains 30,000 sample values before metadata.
```

**Independent check:** `2,500 / 250 = 10 s`. For 12 channels, `12*2,500=30,000` values. This does not determine byte storage until representation and overhead are specified.

### 4.2 General: sampling interval

```text
FORMULA [SIG-02]: Delta t = 1 / f_s
WHERE: Delta t = nominal interval [s/sample]; f_s [samples/s].
CONSTRAINT: regular clock; a device-reported nominal rate is not evidence that timestamps have no jitter.
NUMERICAL EXAMPLE: f_s=250 Hz; Delta t=1/250=0.004 s=4 ms.
INTERPRETATION: an event between samples is represented only through the sampled sequence; timing claims should not exceed this resolution without a stated method.
```

### 4.3 Variant: Nyquist condition

For an ideal band-limited signal with highest retained frequency `f_max`, a necessary sampling condition is `f_s > 2f_max`. This is not sufficient for real acquisition: anti-alias filtering, sensor bandwidth, noise, clock behavior, and feature requirements still matter.

```text
FORMULA [SIG-03]: f_s > 2 f_max
WHERE: f_max = highest frequency retained by the acquisition/analysis contract [Hz].
CONSTRAINT: ideal band-limited model and effective anti-alias filter; strict inequality leaves transition-band design unresolved.
NUMERICAL EXAMPLE: retain up to 80 Hz; minimum theoretical rate is greater than 160 Hz; 250 Hz clears only this necessary condition.
INTERPRETATION: 250 Hz is not a universal “good ECG rate”; appropriateness depends on intended use and verified acquisition protocol.
```

### 4.4 Missing fraction and longest-gap contract

For a required channel, let `m` be the count of nominal samples marked missing in a segment of expected count `n`. The missing fraction is `r=m/n`, with `0<=m<=n` and `n>0`. It measures amount, not arrangement. One hundred isolated missing samples and one contiguous block of one hundred can have the same `r` while having different effects on a feature.

```text
FORMULA [SIG-04]: r = m / n
WHERE: r = nominal-sample missing fraction [dimensionless];
       m = missing nominal samples [count]; n = expected samples [count].
CONSTRAINT: one channel, one segment, n>0; missingness is identified independently
            of the target label; timestamps supersede nominal counts when irregular.
NUMERICAL EXAMPLE: n=2,500; m=12; r=12/2500=0.0048=0.48%.
INTERPRETATION: a low aggregate missing fraction can coexist with an unacceptable
                contiguous gap; apply both amount and longest-gap rules.
```

For regular sampling, `g` consecutive missing nominal intervals span at least a nominal duration on the order of `g/f_s`; exact endpoint semantics and measured timestamps determine the precise observed gap. Do not claim a precise physical gap duration from an assumed rate alone after a transport interruption.

### 4.5 Payload and daily retention estimate

For `C` channels, `n` values per channel, and `b` bytes per stored value, uncompressed sample payload is `S=C*n*b`. This excludes metadata, quality masks, timestamps, indexes, redundancy, and derived versions.

```text
FORMULA [SIG-05]: S_day = R * C * f_s * T * b
WHERE: S_day = raw sample payload per day [bytes/day];
       R = records/day [count/day]; C [channels/record];
       f_s [samples/(channel*s)]; T [s/record]; b [bytes/sample].
CONSTRAINT: fixed shape/rate/representation across R records; no compression;
            excludes all non-sample storage and replication.
NUMERICAL EXAMPLE: R=10,000; C=12; f_s=250; T=10; b=2;
                   S_day=10000*12*250*10*2=600,000,000 bytes/day.
INTERPRETATION: the number is payload-only sizing, not a storage quote or
                retention design; add measured overhead and policy requirements.
```

If each record also retains a 4,096-byte acquisition/lineage envelope once, that adds `10,000*4,096=40,960,000 bytes/day` before masks and derived data. The sum is `640,960,000 bytes/day`; this remains an illustrative lower-bound calculation rather than an observed footprint.

### 4.6 Limit / counterexample

Upsampling a 100 Hz recording to 250 Hz adds interpolated points, not new physical information. Conversely, a high-rate file can still be unusable if one channel is disconnected, saturated, mislabeled, or time-shifted.

A second counterexample concerns calibration. Multiplying an already calibrated microvolt value by a microvolts-per-count scale is dimensionally wrong: the output would carry squared or otherwise inconsistent units depending on the mistaken convention. Check the input unit of every conversion rather than looking only for a finite numeric output.

**Takeaway:** Sampling equations bound representation; they do not validate sensor placement, calibration, labeling, or downstream use.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE and arithmetic VERIFIED; not a measured clinical dataset.

A ten-second record has 12 named channels sampled at 250 Hz with a stated fixed-rate protocol. Per channel, `n=250*10=2,500`; total values `12*2,500=30,000`. If each sample is stored as a signed 16-bit value, payload is `30,000*2=60,000 bytes`, approximately `58.6 KiB` using 1 KiB = 1,024 bytes. This excludes headers, timestamps, encryption, compression, replication, and annotations.

Suppose the quality gate requires at least 9 of 12 channels present and no contiguous missing interval longer than 40 ms. At 250 Hz, 40 ms corresponds to `0.040/0.004=10` nominal sample intervals. A segment with 11 working channels but a 12-sample gap in one required channel fails the stated gate even if its total missing fraction looks small.

**Sensitivity:** At 500 Hz, the sample count and payload double to 60,000 values and 120,000 bytes under the same representation. The 40 ms threshold becomes 20 intervals. Changing rate requires changing sample-count checks and filter/feature validation; it is not a harmless rescale.

**Takeaway:** Make quality criteria physical-time and channel-contract statements, then map them to samples for a specific acquisition rate.

### 5.1 Worked acquisition-and-routing trace

**Inputs:** twelve channels named by `CM-1`; `f_s=250 Hz`; nominal ten-second segment; int16 sample storage; all channels present except `ECG-II`, which has twelve missing nominal intervals. The illustrative policy requires at least nine of twelve channels present, less than 1% aggregate missing samples per required channel, and no required-channel contiguous gap longer than 40 ms. A failed required-channel rule yields abstention, not a negative clinical result.

| Check | Substitution | Result | Decision effect |
|---|---|---|---|
| expected channel count | `250*10=2,500` | 2,500 values/channel | establishes nominal shape only |
| present channel count | 11 of 12 | 11>=9 | channel-count check passes |
| aggregate missing fraction | `12/2,500=0.0048` | 0.48%<1% | aggregate check passes |
| nominal gap duration | `12*(1/250)s=0.048s` | 48 ms>40 ms | longest-gap check fails |
| raw sample payload | `12*2,500*2=60,000` | 60,000 bytes | storage estimate, not quality proof |
| quality disposition | required longest-gap rule failed | ABSTAIN: `GAP_EXCEEDS_LIMIT` | route to governed recovery |

This trace demonstrates why conjunctive quality criteria matter. Selecting only the channel-count and missing-fraction checks would pass the segment, but that would violate the stated policy. The failure is not repaired by reporting the passing checks more prominently.

### 5.2 Split-first preprocessing trace

Suppose there are 100 encounters, each contributing ten overlapping segments. A random segment split assigns 500 segments to training and 500 to evaluation. Because every encounter's windows can appear in both sets, the evaluation does not test generalization to a new encounter. The corrected protocol assigns all of an encounter's segments to one side before normalization fitting and window-specific feature generation.

Under an 80/20 encounter split, 80 encounters contribute 800 training segments and 20 contribute 200 evaluation segments. Fit any learned imputation or scaler using the 800 training segments only; freeze its parameters; apply them unchanged to evaluation. This removes one leakage route, not all dependence: repeated encounters from the same subject may require a subject-level split, and acquisition-site effects may require a separate transportability evaluation.

### 5.3 Sensitivity to storage representation

Changing int16 to float32 doubles bytes per value from 2 to 4. For the same 10,000-record day, sample payload becomes `1,200,000,000 bytes/day`. If a lossless codec achieved a 2:1 reduction relative to that float32 payload, the result would be 600,000,000 bytes/day, but no such compression ratio is assumed or measured here. Do not use an illustrative ratio to provision production storage.

A separate sensitivity is channel count: at six channels, int16 payload halves to 300,000,000 bytes/day. Reduced storage does not imply equivalent model quality because discarded channels may carry necessary measurements. Resource and analytical decisions require separate acceptance evidence.

## 6. Visual / Geometric Intuition

Visualize the raw waveform as points on a time axis, each tagged with channel and unit. A filter changes the path between points; segmentation chooses a window; normalization changes coordinate scale. These are not neutral operations: a per-segment normalization can erase amplitude differences, while a global normalization fitted on all data can leak future/test distribution information.

Quality flags form a parallel track. Do not hide rejected samples by smoothing them into a plausible-looking curve. A reviewer should be able to see raw, derived, and masked intervals with the transformation version.

**Takeaway:** Preserve the ability to distinguish physiology, artifact, and preprocessing effect in the same time coordinate system.

### 6.1 Causal and noncausal filtering on a timeline

A causal output at time `t` depends on current/past inputs under the filter definition. A centered offline filter can depend on samples both before and after `t`. If a feature is meant to be computed at `t_d`, centered filtering that reads samples after `t_d` changes the task from online inference to retrospective analysis.

```text
online time:       past ---------------- t_d ---------------- future
permitted input:   [available history ---- t_d]
invalid shortcut:  [past ------------ t_d ------------ future input]
offline analysis:  may use a declared retrospective window, with a different contract
```

Zero-phase forward/backward filtering is inherently an offline operation over a chosen signal extent. It can reduce phase distortion under its mathematical assumptions, but that does not make it valid for an online decision or valid across missing boundaries. Padding and endpoint handling also affect the first and last samples; evaluate those regions rather than cropping them silently after scoring.

### 6.2 Artifact overlay and quality masks

A diagnostic plot should distinguish at least raw values, derived values, missing/imputed masks, saturation/disconnection flags, and segment boundaries. Use the same time axis; otherwise a filter delay or clock correction can look like a physiological change. Plotting only the cleaned waveform hides the acquisition defects that determine whether the cleaned result is legitimate.

For example, a linearly interpolated gap appears smooth and may improve a visual impression. The mask still identifies that no measurement exists there. If the analysis requires observed peaks or intervals inside that gap, interpolation cannot create the needed evidence. A quality flag must therefore influence feature eligibility, not merely chart color.

### 6.3 Normalization coordinate systems

Per-segment centering subtracts a segment-derived location; it can remove absolute amplitude information. Per-channel scaling can make channel amplitudes comparable but may also erase clinically or scientifically relevant scale. Cohort-level scaling estimated on training data preserves a common transform but may be sensitive to acquisition shifts.

The selection question is not “which normalization is best?” It is “which coordinate information does the target and deployment protocol require?” Test the chosen transform on raw/derived reference traces and by acquisition subgroup, and record whether each parameter is fixed, training-fitted, or computed from the current segment.

## 7. Comparison / Selection Table

| Approach | Use when | Strength | Avoid when | Required check |
|---|---|---|---|---|
| raw waveform with provenance | audit/reprocessing is needed | preserves evidence | retention/access prohibits it | governed raw reference and access control |
| fixed filter and fixed windows | protocol is stable | reproducible comparison | signal timing/rate varies unaccounted | versioned parameters and time alignment |
| adaptive/artifact-aware processing | artifacts vary and review exists | can preserve usable segments | it masks failures or lacks trace | compare raw/derived and rejection rates |
| aggregate features | decision is defined on validated features | compact and interpretable | feature extraction uses future windows | point-in-time extraction test |

**Takeaway:** Choose a transformation because its assumptions match the acquisition and decision contract, not because it improves a retrospective score.

### 7.1 Choosing a quality policy

Write the quality contract as eligibility rules before fitting a model. For each required channel, state presence, valid unit, calibration status, allowable missing fraction, longest allowed physical gap, saturation/artifact handling, and timestamp tolerance. Declare whether a rule applies to acquisition, segment inclusion, individual features, or the final output; these are not interchangeable gates.

| Candidate policy | Benefit | Selection cost | Verification |
|---|---|---|---|
| reject whole record | simple and conservative | loses unaffected channels/segments | count exclusions and downstream coverage by subgroup |
| reject affected segment | preserves other time intervals | can select easier time periods | compare accepted/target distributions and gap rules |
| retain with mask | preserves partial information | model must interpret mask correctly | masked ablation and raw/derived audit |
| impute within a bounded gap | supports fixed-shape features | fabricates intermediate measurements | validate on simulated held-out intervals and preserve mask |
| route to human/reacquisition | avoids unsupported model inference | delay and workload | measure route completion, timeout, and fallback |

Exclusions are part of evaluation. A model evaluated only on high-quality accepted records can perform well while the overall workflow fails frequently at acquisition. Report both conditional performance among accepted segments and end-to-end coverage/disposition among eligible records. Neither substitutes for the other.

### 7.2 Choosing filters and resampling

Start from the target-relevant frequency band, allowable phase/group delay, online/offline timing, edge behavior, and acquisition filter. Then choose the simplest transform satisfying those constraints. A higher-order filter is not automatically preferable: ringing, instability, delay, and sensitivity to parameters can offset apparent attenuation.

For a rate change, distinguish decimation/downsampling from interpolation/upsampling. Lowering the rate without appropriate band limiting can introduce aliasing; raising it adds no new measured band. The filter and resampling design must be checked against the actual signal and protocol, not just a generic Nyquist slogan. Preserve the pre-resampling reference and version the target grid.

### 7.3 Validation matrix

| Validation layer | Minimum comparison | Population/slice | Failure action |
|---|---|---|---|
| schema/unit | known channel map and calibration fixtures | every acquisition schema | reject unknown mapping/unit |
| timing | measured timestamp gaps and availability trace | device/gateway/time period | quarantine incompatible timing |
| signal quality | raw trace plus independent status evidence | site/device/channel/artifact | abstain or governed recovery |
| transform | reference signals and split-safe refit | filter/version/rate | block affected derived version |
| model | approved baseline with uncertainty | independent subjects and relevant subgroups | no promotion from aggregate alone |
| workflow | end-to-end coverage, review load, fallback | eligible population and failure routes | adjust authorized operating protocol |

A new acquisition version is a change to the input contract, even if the model file is unchanged. Re-run relevant schema, calibration, timing, quality, and subgroup checks. An unchanged model score on old retained data does not validate the new device route.

## 8. Failure Modes & Edge Cases

| ID | Competing causes | Discriminating diagnostic | Correction / verification |
|---|---|---|---|
| SIG-F01 aliasing | inadequate rate; missing anti-alias filter; wrong frequency assumption | inspect acquisition filter/rate and controlled frequency response | reacquire or constrain analysis; verify with known test signal |
| SIG-F02 channel swap/mislabelling | connector error; export order change; metadata mapping bug | compare channel identity against calibration trace and schema version | reject mapping; pin schema and replay test |
| SIG-F03 hidden missingness | dropped packets; flatline; interpolation | gap/flatline metrics plus raw trace | preserve mask, route/reacquire; verify missing interval rule |
| SIG-F04 leakage through preprocessing | fit scaler/filter selection on full cohort; future window in segment | split-aware pipeline audit and timestamp trace | refit training-only transforms; rerun evaluation |
| SIG-F05 label misalignment | clock offset; wrong encounter join; delayed adjudication | inspect segment-label offset and sampled joins | correct linkage; adjudicate a held-out audit sample |

Mitigations have risks: rejecting too many records can make a selected population look easier, while aggressive interpolation can conceal hardware defects. Report exclusion counts and reasons by site/device/time when applicable.

**Takeaway:** Signal failures require a physical or temporal diagnostic, not merely a lower model score.

### 8.1 Additional discriminating failures

| ID | Competing causes | Expected diagnostic signal | Bounded correction and mitigation risk | Verification |
|---|---|---|---|---|
| SIG-F06 saturation hidden by scaling | excessive gain; clipping; wrong unit conversion | repeated extrema in raw values and device range evidence | reject affected intervals; rescaling can conceal rather than repair | replay known saturated raw fixture and confirm mask survives |
| SIG-F07 clock drift | unsynchronized devices; clock reset; timestamp parsing | changing offsets or nonmonotonic/duplicate timestamps | quarantine sync status; forced realignment can invent event order | compare with a controlled common-time fixture |
| SIG-F08 phase/timing shift | causal filter delay; offline zero-phase transform; padding | feature/event offset differs by transform | choose transform consistent with target timing; compensating delay may not restore edge validity | measure known-event alignment before/after transform |
| SIG-F09 window duplication across split | random window split; repeated encounters; retries | same subject/session/raw lineage across train/test | split independent units before windowing; reduces nominal evaluation size | assert disjoint subject/session IDs and raw lineage |
| SIG-F10 selection through quality gate | artifacts correlate with site/device/target | accepted-population mix changes after filtering | report coverage/exclusions; relaxing gate can increase invalid inputs | compare target and subgroup rates before/after acceptance |
| SIG-F11 stale calibration | wrong version; expired/invalid procedure | known reference produces biased converted values | bind calibration to record/time; retroactive recalibration needs audit | reference-level and range tests under approved procedure |
| SIG-F12 label granularity mismatch | record-level label applied to windows; wrong annotation interval | implausible all-positive or all-negative window propagation | use interval-specific labels or broader decision unit | inspect annotation mapping and adjudicated counterexamples |
| SIG-F13 deduplication destroys repeats | same values at different times; legitimate repeated segments | raw record counts change after value-only dedup | deduplicate by acquisition identity and replay lineage | replay retry plus legitimate repeated-value fixture |
| SIG-F14 fallback silently fails | unavailable reviewer; reacquisition impossible; timeout | pending route has no terminal disposition | explicit timeout and authorized baseline route; default pass is forbidden | simulate unavailable route and inspect terminal audit state |

### 8.2 Diagnostic sequence

When performance changes, do not start by tuning the model. First compare the accepted-record count, channel map, units, sampling/timestamp distribution, missing-gap profile, calibration version, transform version, label protocol, and independent-unit split against the baseline. Then isolate a single changed stage with raw/derived fixtures.

A useful regression report includes:

1. eligible, received, accepted, abstained, and rejected counts with denominators;
2. reasons by channel, site/device/schema, and time period where authorized;
3. actual versus nominal sample intervals and availability delays;
4. calibration and transformation versions;
5. split disjointness checks for subject, encounter, session, and raw lineage as applicable;
6. model metrics on the accepted population plus end-to-end workflow coverage;
7. route completion, timeout, and manual/baseline fallback outcomes;
8. known limitations, unreviewed evidence, and rollback/forward-recovery disposition.

This sequence narrows mechanisms; it does not prove causality. If multiple changes shipped together, use controlled comparisons or restore the last compatible acquisition/transform version under the change protocol. Do not rewrite historical raw inputs to make the regression disappear.

## 9. Complexity & Cost Analysis

For `C` channels and `n` samples per channel, basic scanning/filtering is commonly proportional to `O(Cn)` under fixed filter order; memory is `O(Cn)` if the entire record is retained, or lower for a streaming stateful filter. These are algorithmic statements, not device benchmarks. At the illustrative 30,000 values/record and 10,000 records/day, raw 16-bit payload alone is 600,000,000 bytes/day before metadata and retention overhead.

The operational cost is often quality review and lineage retention rather than arithmetic. Price, retention duration, and compliance requirements are not asserted here and require local sources.

**Takeaway:** Size storage from signal shape and representation, then separately account for the audit artifacts that make the analysis defensible.

### 9.1 Windowing and streaming cost

For nonoverlapping windows of length `n_w` samples, processing a length-`n` channel examines approximately `n/n_w` windows, ignoring partial-window policy. For hop `h<n_w`, windows overlap and the approximate count is `1+(n-n_w)/h` when the record exactly accommodates the declared final full window. This count does not establish arithmetic cost per window; a feature can scan all samples, sort them, or run a more expensive model.

Streaming retains filter state and a bounded window rather than all historical samples. If the only retained state is one length-`n_w` window across `C` channels, its sample buffer is `O(C*n_w)`, excluding model and metadata state. Overlapping windows may re-read samples unless results or intermediate states are reused. Benchmark actual latency and memory for the selected implementation; asymptotic notation does not provide a device deadline guarantee.

### 9.2 Illustrative 30-day payload ledger

Using the earlier 10,000-record day and 60,000-byte raw sample payload per record:

| Artifact assumption | Daily payload | 30-day payload |
|---|---:|---:|
| one raw int16 sample array | 600,000,000 bytes | 18,000,000,000 bytes |
| one acquisition envelope of 4,096 bytes/record | 40,960,000 bytes | 1,228,800,000 bytes |
| one equally sized derived float16 array, illustrative only | 600,000,000 bytes | 18,000,000,000 bytes |
| sum of those listed components | 1,240,960,000 bytes | 37,228,800,000 bytes |

The table is not a storage architecture. It excludes timestamps, masks, annotations, database indexes, temporary processing, backups, replication, encryption overhead, additional transform versions, deletion, and access controls. Compression is not assumed. Actual sizing requires measured distributions and retention/replication policy.

### 9.3 Review-work boundary

If 10,000 records/day produce a 2% abstention rate, the review/reacquisition queue receives 200 records/day before retries. At three minutes of review per record, that is 600 minutes = 10 hours/day. These are constructed workload assumptions, not observed abstention or review rates. If review capacity is below incoming work, the queue grows unless arrival or service changes; raising model acceptance solely to clear the queue would change the quality contract and require independent validation.

## 10. Key Theorems / Results Reference

- **Sampling theorem (DERIVED scope):** under ideal band-limited assumptions, sampling above twice the maximum retained frequency prevents alias ambiguity in that model; real anti-alias design remains required.
- **Dimensional check:** `samples/s * s = samples`; `s/sample` is the reciprocal unit of sampling frequency.
- **Point-in-time invariance:** transformations that access future samples or labels change the intended real-time decision problem.
- **No semantic guarantee:** a valid sample rate does not establish valid calibration, label, clinical interpretation, or intervention benefit.

**Takeaway:** The theorem reference constrains signal representation, while the acquisition protocol constrains whether the represented signal means what the analysis assumes.

### 10.1 Reusable validation results

| Result | Preconditions | Consequence | Non-implication |
|---|---|---|---|
| finite-schema rejection | required identity/unit/version absent | no model input is formed | not proof of poor physiology |
| raw lineage preservation | immutable raw reference and versioned derivation | transformations can be audited/replayed | not proof the transform is clinically appropriate |
| split disjointness | independent unit and raw lineage do not cross splits | removes an identified leakage route | not independence of all sites/devices/populations |
| mask preservation | missing/imputed intervals remain identified | downstream eligibility can respect unmeasured data | not recovery of the original samples |
| dual reporting | accepted-only and end-to-end denominators are defined | exposes selection and conditional performance separately | not clinical utility or deployment approval |
| deterministic disposition | each segment has explicit terminal/pending semantics | failed inputs cannot disappear silently | not reviewer availability or safe operation |

### 10.2 Acceptance gate before publication

A production-specific signal sheet requires approved protocol/source citations, verified device and calibration behavior, actual preprocessing references, an independent-unit split protocol, target-specific validation, uncertainty, operational review/fallback capacity, and evidence review. This file remains a general teaching draft because those artifacts are not available here. Passing the arithmetic fixture in this repository cannot close those gates.

For reuse, copy the contract pattern rather than the illustrative thresholds. The 40 ms gap, nine-channel minimum, 1% missing fraction, 2% abstention, and three-minute review time are constructed examples. They must not become a device, clinical, or procurement recommendation.

## 11. Interview / Exam Questions

The fifteen questions in this section are deliberately different checks of the
contract. Questions 1–10 are rapid oral checks; Questions 11–15 require a
calculation or a design decision. A response earns credit only when it names
the applicable boundary rather than treating a plausible array as evidence of
clinical validity.

1. **Calculate samples for 250 Hz over 10 seconds.** **Answer:** `250*10=2,500`
   samples per channel, conditional on fixed-rate acquisition without dropped
   samples. **Reasoning:** hertz is samples per second, so multiply rate by
   elapsed seconds. **Follow-up:** state what changes under timestamp jitter.
   **Red flag:** treating the count as a clinical validation. **Scoring:** full
   credit includes both the unit and the fixed-rate condition.
2. **Why is 250 Hz not enough information to judge suitability?** **Answer:**
   modality, retained band, anti-alias behavior, sensor/placement, calibration,
   timestamps, target, and intended use are unspecified. **Follow-up:** ask
   which of those a schema can check. **Red flag:** “the model accepts 250 Hz.”
   **Scoring:** full credit separates rate from measurement meaning.
3. **What does a 40 ms gap equal at 250 Hz?** **Answer:** ten nominal intervals
   because `0.040 s/(1/250 s)=10`; use observed timestamps if the clock is not
   demonstrably regular. **Follow-up:** decide a 12-interval gap under a
   40 ms ceiling. **Red flag:** using a sample index when a clock reset exists.
   **Scoring:** full credit names the timestamp boundary.
4. **Why retain raw and derived lineage?** **Answer:** it permits an auditor to
   reconstruct whether an output followed the declared acquisition and
   transform contract. **Follow-up:** list raw ID, transform version, and
   configuration hash. **Red flag:** retaining only feature values. **Scoring:**
   full credit distinguishes replayability from clinical appropriateness.
5. **Can interpolation repair aliasing?** **Answer:** no; interpolation creates
   values between samples but cannot identify information lost through
   inadequate sampling or undocumented anti-aliasing. **Follow-up:** name the
   correction: approved reacquisition design. **Red flag:** smoother plotting.
   **Scoring:** full credit distinguishes representation from measurement.
6. **How can normalization leak?** **Answer:** fitting its statistics on
   evaluation or future data lets those distributions influence training.
   **Follow-up:** say where the fitted statistics are stored. **Red flag:**
   “it is only preprocessing.” **Scoring:** full credit requires a training-only
   fit and an unchanged later application.
7. **What distinguishes a flatline from stable physiology?** **Answer:** a
   protocol-specific quality check using channel behavior, calibration,
   acquisition state, and cross-channel/context evidence; no single amplitude
   heuristic is sufficient. **Follow-up:** give its disposition when the
   calibration is unknown. **Red flag:** declaring a diagnosis. **Scoring:**
   full credit returns abstain/review rather than a biological conclusion.
8. **A device changes channel order. What fails?** **Answer:** the schema and
   channel-identity contract; pin the named mapping version and reject an
   unknown layout. **Follow-up:** why not infer order from waveform shape?
   **Red flag:** “there are still twelve columns.” **Scoring:** full credit
   protects identity rather than dimensionality.
9. **What is a defensible quality-gate output?** **Answer:** pass, fail, or
   abstain with reason codes, measured signals, record/segment identity,
   configuration version, and an accountable fallback. **Follow-up:** state
   whether `PASS` means a disease target is negative. **Red flag:** silently
   dropping failures. **Scoring:** full credit says no and preserves auditability.
10. **What evidence is OPEN here?** **Answer:** local acquisition protocol,
    target labels, device behavior, clinical validity, and original source
    bodies. **Follow-up:** distinguish a checked arithmetic fixture from a
    device test. **Red flag:** converting an illustrative input into a
    measurement. **Scoring:** full credit identifies the missing primary
    evidence and the appropriate validation owner.

**Takeaway:** A concise answer is strong only if it keeps measurement,
decision-time information, and evidence maturity separate.

### 11.1 Illustrative pipeline oracle inputs

These paper-oracle inputs test acquisition and lineage invariants. They are
ILLUSTRATIVE teaching traces, not device runs, clinical validation, or evidence
about any named dataset or manufacturer.

#### SIG-O01 — regular 12-channel record

- **Input:** `record=R01`, `f_s=250 samples/s`, `T=10 s`, `C=12 channels`,
  `dtype=int16`, no drops, named channel map version `CM-1`.
- **Expected arithmetic:** `n=250*10=2,500` samples/channel and
  `C*n=30,000` samples total.
- **Expected payload:** `30,000*2=60,000 bytes`, excluding all metadata.
- **Quality result:** PASS only if all 12 named channels satisfy the declared
  presence and calibration contract.
- **Audit output:** record ID, time zone, start/end, channel map, units,
  acquisition version, raw-reference ID, transform version, and quality result.
- **Failure boundary:** a correct count does not validate placement or physiology.

#### SIG-O02 — physical-time gap at 250 Hz

- **Input:** `record=R02`, `f_s=250 Hz`, maximum contiguous gap `40 ms`,
  required channel has 12 missing nominal intervals.
- **Expected conversion:** `Delta t=1/250 s=4 ms`; 12 intervals correspond to
  `48 ms`, greater than the 40 ms gate.
- **Expected result:** FAIL or ABSTAIN with reason `GAP_EXCEEDS_LIMIT`.
- **Discriminating check:** compare timestamps when available; do not assume
  exactly 4 ms spacing after a transport interruption.
- **Recovery:** reacquire or use the governed non-signal fallback; interpolation
  cannot convert an unobserved interval into measured evidence.
- **Failure boundary:** missing fraction alone can hide one clinically material gap.

#### SIG-O03 — rate change without pipeline update

- **Input:** `record=R03`, device changes from `250 Hz` to `500 Hz`, while
  segmentation still assumes 2,500 samples means ten seconds.
- **Expected arithmetic:** at 500 Hz, 2,500 samples span `5 s`, not `10 s`.
- **Expected result:** REJECT schema/pipeline compatibility before feature work.
- **Discriminating check:** compare device sampling metadata and segment duration
  computed from timestamps.
- **Recovery:** pin a new acquisition version; reassess filter, window, quality,
  and feature contracts for the new rate.
- **Failure boundary:** resampling the array does not prove equivalence.

#### SIG-O04 — channel-order export change

- **Input:** `record=R04`, values have 12 columns, export schema changes order,
  while a downstream model maps columns by position rather than channel name.
- **Expected result:** REJECT unknown mapping; no model output is permitted.
- **Discriminating check:** verify a signed channel-map version and calibration
  trace instead of visually plausible waveform shape.
- **Recovery:** install the approved mapping or retain baseline processing.
- **Audit output:** source schema, observed mapping, expected mapping, rejection
  reason, and operator response.
- **Failure boundary:** channel count equality does not establish channel identity.

#### SIG-O05 — anti-aliasing counterexample

- **Input:** `record=R05`, retained frequency claim `f_max=80 Hz`,
  `f_s=100 Hz`, no documented anti-alias filter.
- **Expected condition check:** `100 > 2*80` is false; the stated necessary
  sampling condition fails.
- **Expected result:** the representation is unsuitable for that retained band;
  downstream classification cannot repair the ambiguity.
- **Recovery:** reacquire with an appropriate approved acquisition design.
- **Audit output:** declared band, rate, anti-alias evidence status, and decision.
- **Failure boundary:** a smooth plotted waveform is not evidence against aliasing.

#### SIG-O06 — scaler leakage

- **Input:** training segments end 30 June; final evaluation begins 1 July;
  normalization mean and variance were fit on both periods together.
- **Expected result:** REJECT evaluation because future/test distribution informed
  the training transform.
- **Discriminating check:** inspect fit artifact input IDs and cutoff timestamps.
- **Recovery:** fit transform on training data only, apply unchanged to later data,
  and repeat the evaluation protocol.
- **Failure boundary:** a deterministic scaler can still leak if fit on all data.

#### SIG-O07 — label-time mismatch

- **Input:** segment begins `09:00`, target adjudication becomes available
  `16:00`, join uses patient ID and calendar day without a decision-time rule.
- **Expected result:** label is evaluation-only; it cannot enter features or
  contemporaneous eligibility logic.
- **Discriminating check:** inspect join predicate and label availability timestamp.
- **Recovery:** declare the target maturity protocol and preserve an unresolved
  label state where required.
- **Failure boundary:** same-day linkage is not point-in-time validity.

#### SIG-O08 — quality flag is not diagnosis

- **Input:** a segment passes channel/gap checks but has no approved target label.
- **Expected result:** `QUALITY_PASS` means analyzable under the signal contract,
  not clinically normal, safe, or diagnostically negative.
- **Discriminating check:** review UI wording and downstream label semantics.
- **Recovery:** separate quality result from any research or clinical target output.
- **Failure boundary:** a clean waveform can still be irrelevant to a decision.

### 11.2 Expanded interview answers

11. **Why does a fixed 250 Hz rate not certify a signal pipeline?** **Checked
answer:** rate defines nominal temporal sampling only; channel identity, physical
units, sensor/calibration, anti-aliasing, timestamps, quality, target protocol,
and intended use still determine validity. **Follow-up:** show the device
contract. **Red flag:** “the model accepts 250 Hz.” **Scoring:** full credit
names measurement and decision boundaries.

12. **A 10-second, 12-channel, 250 Hz record has how many values?** **Checked
answer:** `12*250*10=30,000`. **Reasoning:** count per channel first, then
channels. **Follow-up:** what raw int16 payload follows? 60,000 bytes before
metadata. **Red flag:** call values bytes. **Scoring:** full credit keeps units.

13. **Why can interpolation not repair aliasing?** **Checked answer:** it
constructs additional sampled points but cannot recover frequency information
lost before or during inadequate sampling. **Follow-up:** what must change?
Acquisition and anti-alias design. **Red flag:** “a smoother trace is repaired.”
**Scoring:** full credit distinguishes representation from measurement.

14. **What should a quality-gate failure produce?** **Checked answer:** a
versioned fail/abstain reason, source and segment identity, and governed
reacquisition, human-review, or baseline route. **Follow-up:** should it become
a negative target? No. **Red flag:** silently dropping it. **Scoring:** full
credit names a traceable fallback.

15. **How does preprocessing leak?** **Checked answer:** future samples, labels,
or test-distribution statistics influence an earlier training/evaluation input.
**Follow-up:** give a correction: fit transformations on training data only.
**Red flag:** “filters are always safe.” **Scoring:** full credit names the
point-in-time boundary.

**Takeaway:** The oral pack contains exactly fifteen answered questions; its
purpose is to test contract reasoning, not to authorize signal interpretation.

### 11.4 Acquisition contract worksheet

Use this worksheet before implementation. It deliberately leaves local facts unpopulated rather than supplying universal defaults.

| Field | Required answer | Evidence to attach | Rejection state if absent |
|---|---|---|---|
| intended decision | output, user, timing, and prohibited use | approved decision/protocol brief | no target-specific validation |
| independent unit | subject/encounter/session and split rule | cohort and lineage specification | evaluation cannot be interpreted as independent |
| observation interval | segment start/end and endpoint convention | timestamped raw reference | no temporal feature contract |
| availability | acquisition/transmission/ingestion deadline | timestamp trace under normal/failure conditions | online validity unknown |
| channel map | names, placement, order, and schema version | approved mapping/calibration fixture | reject positional interpretation |
| physical units | raw and calibrated units, gain/offset | calibration procedure/version | reject unit-dependent features |
| clock | source, precision, sync, gap representation | device/gateway timing evidence | nominal-only limitation or rejection |
| band/rate | target band, rate, anti-alias design | acquisition/filter specification | cannot establish representation suitability |
| valid range | saturation/missing/disconnected encoding | device/schema evidence | preserve unknown; no zero default |
| quality rules | channel, amount, longest-gap, artifact rules | policy and validation examples | pending quality decision |
| transforms | code/config/version, causal/offline boundary | reproducible transform record | block online use if future-aware |
| target | definition, window, adjudication, availability | annotation/maturity protocol | no supervised metric claim |
| fallback | reacquire/review/baseline, owner, timeout | local workflow authorization | monitored exception; never default pass |
| retention | raw/derived/mask/audit retention and deletion | approved data lifecycle | deployment storage/lifecycle unknown |

### 11.5 Release review record template

| Review item | Evidence actually required | Result vocabulary |
|---|---|---|
| arithmetic | independent recomputation of counts, units, gaps, and payload | VERIFIED for stated arithmetic only |
| raw/derived replay | same raw reference/configuration produces expected derived artifact | observed trace, not assumed reproducibility |
| negative acquisition tests | swapped channel, unknown unit, saturation, gap, duplicate, clock reset | observed rejection and reason per case |
| leakage tests | disjoint independent units and training-only fitted transforms | executed lineage assertion, not pipeline diagram |
| model evaluation | approved comparator, independent target, slices, uncertainty | measured/estimated with denominator and limits |
| workflow rehearsal | unavailable recipient, reacquisition failure, timeout, fallback | observed terminal/exception disposition |
| clinical/local review | accountable domain and operations reviewers | approved, rejected, or open with named evidence |
| change response | new sensor/schema/rate/filter/label triggers defined | owner and rollback/forward-recovery path |

A completed worksheet is still not clinical validation. It makes the intended validation auditable and exposes missing prerequisites before they are obscured by model metrics.

### 11.6 Compact reasoning examples

**Case A — correct payload, unusable measurement.** A file contains the expected 60,000 sample bytes, but calibration is unknown and two channel names are missing. **Decision:** reject the affected interpretation; expected size checks only transport shape. **Recovery:** restore schema/calibration lineage and rerun known-reference fixtures.

**Case B — valid retrospective analysis, invalid online deployment.** A centered transform is validated for offline research using the complete record. **Decision:** retain it only for the declared retrospective task. **Boundary:** the same feature code cannot be promoted to an earlier online decision without redesign and re-evaluation.

**Case C — passing model, failing service.** Accepted-record discrimination is unchanged, but abstentions double after a firmware change. **Decision:** investigate acquisition and quality dispositions before model tuning. **Reason:** the conditional score does not capture lost coverage or added review work.

**Case D — late but legitimate label.** An outcome is adjudicated after the feature timestamp. **Decision:** it can be a target under the declared evaluation protocol, provided it never enters earlier features/eligibility. **Boundary:** records with immature labels require a declared exclusion/censoring policy rather than negative imputation.

### 11.7 Executed teaching-fixture record

On 2026-11-09, an independent Python arithmetic fixture checked the sheet's sample counts, 60,000-byte record payload, 48 ms gap, 0.48% missing fraction, 600,000,000-byte daily sample payload, 40,960,000-byte metadata addition, 37,228,800,000-byte 30-day component sum, ten-hour review example, and 500 Hz sensitivity. All 13 stated arithmetic assertions passed.

A separate in-memory paper oracle checked four dispositions: eleven present channels with a 48 ms gap abstained for `GAP_EXCEEDS_LIMIT`; eight present channels abstained for `CHANNEL_COUNT`; 25 missing of 2,500 abstained for `MISSING_FRACTION`; and twelve isolated missing samples with a ten-interval longest gap passed under the constructed policy. A split fixture generated 800 training and 200 evaluation windows from disjoint sets of 80 and 20 encounters and confirmed no encounter overlap.

**Scope:** these checks execute constructed formulas and control flow only. They do not test a sensor, signal filter, clinical target, human workflow, storage system, or production implementation. Primary evidence and substantive independent manuscript review remain OPEN.

### 11.8 Decision-time audit traces

The following traces are compact enough to execute during a design review and
are more useful than a generic “data quality passed” flag. They are
ILLUSTRATIVE; neither trace establishes a clinical workflow.

| Trace field | `R06` accepted research segment | `R07` rejected segment | Why it changes the decision |
|---|---|---|---|
| acquisition ID | immutable `R06` | immutable `R07` | prevents value-only de-duplication |
| observed interval | `[10:00:00,10:00:10)` | `[10:05:00,10:05:10)` | defines the physical observation window |
| availability time | `10:00:12` | `10:05:12` | must be no later than a proposed decision |
| channel map | `CM-1`, complete | `CM-unknown` | unknown identity blocks interpretation |
| unit/calibration | recorded | recorded, but mapping absent | units alone do not repair identity |
| gap result | longest 8 ms | longest 8 ms | the quality metric is not the reason for rejection |
| transform | `TR-4`, causal | not executed | a transform cannot normalize an invalid schema |
| target | absent from feature path | absent from feature path | neither route may infer a diagnosis |
| disposition | research-eligible pending target | abstain and recover mapping | action is bounded by evidence |

**Trace A — availability check.** Suppose `R06` is observed through 10:00:10,
arrives at 10:00:12, and a proposed operational decision is due at 10:00:11.
Although the signal's event interval is before the deadline, the artifact is
not available by it. The correct output is `NOT_AVAILABLE_AT_DECISION`, not a
backfilled feature. If the decision instead occurs at 10:00:15 and the
authorized access check succeeds, the timing predicate is true:

```text
usable_at(t_d) = (t_available <= t_d) AND schema_ok AND quality_ok AND authorized
usable_at(10:00:11) = false
usable_at(10:00:15) = true only if the remaining predicates are true
```

This is a mechanism/invariant block rather than a diagnostic formula. Its
preconditions are comparable timestamps, an explicit time zone/clock source,
and a declared decision deadline. A later event does not become earlier merely
because it is stored beside an earlier encounter key. The counterexample is a
retrospective research table: it may legitimately contain the later artifact,
but it cannot be used to claim an earlier online decision was feasible.

**Trace B — split lineage check.** Let `window_id -> acquisition_id -> encounter_id`
be an immutable lineage map. A split is admissible only when the sets of
independent encounter identifiers are disjoint:

```text
E_train ∩ E_eval = ∅
```

The executed teaching fixture used 80 training and 20 evaluation encounter
identifiers and obtained an empty intersection. This checks the constructed
mapping, not independence across people, sites, devices, or time. If two
encounters belong to one person and person-level correlation matters, the
independent unit must be changed before creating windows. Randomly assigning
windows after overlap is a counterexample: the set of window IDs can be
disjoint while the source physiology is shared.

**Takeaway:** Audit records must expose why an input was usable at a given time,
not merely that a later database query returned it.

### 11.9 Four diagnostic failure dossiers

#### Dossier SIG-D01 — apparent drop in accepted-case error

**Trigger/environment:** a firmware or placement change increases quality
abstentions while the accepted subset's model metric improves. **Symptom:** a
dashboard shows lower accepted-case error but reviewers report more missing
work. **Competing causes:** genuine model improvement; changed case mix due to
quality exclusions; a denominator/reporting defect; target-label delay.
**Discriminating diagnostic:** reconcile `eligible -> acquired -> quality pass
-> output -> target-mature` counts by acquisition version and subgroup. The
accepted-only denominator must be displayed beside end-to-end coverage.
**Root-cause evidence:** a version transition aligned with a rise in
`QUALITY_ABSTAIN` supports selection change, not a model claim.

**Immediate mitigation:** freeze automated promotion and route abstentions to
the governed fallback. **Correction:** restore/validate the acquisition
contract, then re-evaluate a predeclared cohort with fixed denominators.
**Prevention:** release gates require eligibility and disposition counts, not
only conditional metrics. **Verification:** compare the same cohort and target
maturity rule before/after a controlled acquisition version change. **Risk:**
blocking promotion may delay a legitimate improvement; do not waive the
denominator check to recover speed. **Escalation:** signal-science owner and
workflow owner jointly decide whether to suspend intake.

#### Dossier SIG-D02 — unexpectedly high score after a pipeline refactor

**Trigger/environment:** a new normalizer, segmenter, or feature job is fit
while all historical records are queryable. **Symptom:** evaluation score rises
without a correspondingly plausible change in raw acquisition. **Competing
causes:** genuine feature improvement; test/future-statistic leakage; duplicate
acquisition replay; overlapping lineage; an altered target definition.
**Discriminating diagnostic:** inspect every fitted artifact's input IDs and
maximum availability timestamp; then compute intersections at the declared
independent unit. A nonempty intersection or post-cutoff fit input invalidates
the comparison.

**Immediate mitigation:** mark the score non-comparable and stop downstream
selection based on it. **Correction:** rebuild transforms from training-only
inputs, reconstruct splits before windowing, and use the unchanged final
evaluation holdout. **Prevention:** persist fit manifests and reject artifacts
whose input set crosses the split boundary. **Verification:** an independent
replay produces an empty lineage intersection and the same transform digest.
**Risk:** a stricter split may reduce apparent sample size; that is an honest
loss of unsupported precision. **Escalation:** evaluation owner resolves target
or lineage ambiguity before reporting results.

#### Dossier SIG-D03 — plausible waveform, wrong channel semantics

**Trigger/environment:** an export retains the same number of columns after a
vendor/gateway schema change. **Symptom:** array shape and basic statistics
look ordinary, yet downstream feature distribution shifts. **Competing causes:**
column permutation; polarity change; scaling/gain change; genuine population
shift; corruption in one channel. **Discriminating diagnostic:** compare the
signed channel-map and calibration versions to the approved acquisition
manifest; visual inspection is insufficient because several channels can look
plausible.

**Immediate mitigation:** reject the mapping and use the non-signal baseline or
reacquisition route. **Correction:** obtain the governed mapping, run a
known-reference/calibration fixture, and version the compatibility decision.
**Prevention:** schemas bind names, placement, polarity, units, and calibration
to column positions, with fail-closed ingestion. **Verification:** a deliberately
permuted fixture is rejected and the approved fixture preserves named-channel
features. **Risk:** fail-closed handling increases review load; capacity must be
owned explicitly. **Escalation:** acquisition-protocol owner, not the model
owner, approves a new channel map.

#### Dossier SIG-D04 — a clean retrospective feature fails online

**Trigger/environment:** an offline pipeline uses a centered window or an
outcome-adjacent field when evaluating a decision meant to occur earlier.
**Symptom:** replayed retrospective results are strong, while a live-like
simulation cannot produce the feature by the deadline. **Competing causes:**
future sample use; delayed ingestion; unavailable enrichment; clock-zone
conversion; confusion between label and feature.
**Discriminating diagnostic:** for each feature, list observation end,
transmission/availability time, transform look-ahead, and decision deadline.
The feature is permitted only when all input availability times satisfy the
contract; a causal-looking function can still use a late source.

**Immediate mitigation:** remove the feature from the online route and issue an
abstain/fallback disposition. **Correction:** redesign it as history-only,
change the decision timing, or retain the method as retrospective research.
**Prevention:** a feature registry records availability semantics separately
from event timestamp and label maturity. **Verification:** a deadline-aware
replay rejects every injected late feature and accepts the same record only
after the declared deadline. **Risk:** removing information can reduce
discrimination; that trade-off is preferable to claiming an impossible
decision-time system. **Escalation:** workflow owner decides whether a later
decision is operationally acceptable.

**Takeaway:** Each diagnostic begins with a competing explanation and ends with
an observable verification; a model retune alone is not a correction.

### 11.10 Resource and review-capacity worksheet

Capacity planning must keep measurement size separate from operational claims.
For an illustrative batch of `R=1,000` ten-second records, `C=12` channels,
`f_s=250 samples/s`, and signed 16-bit samples:

```text
values per record = C * f_s * T = 12 * 250 * 10 = 30,000 values
raw sample bytes per record = 30,000 * 2 = 60,000 bytes
raw sample bytes per batch = 1,000 * 60,000 = 60,000,000 bytes
```

The calculation assumes packed 2-byte samples and excludes headers, encryption,
indexes, masks, retries, compression, and storage replication. It is therefore
a lower-bound payload calculation, not a procurement estimate. Its independent
unit check is `60,000 / (12*250*10) = 2 bytes/value`. If channel count doubles,
the sample payload doubles; if only duration doubles, it also doubles. Neither
sensitivity establishes a device's actual throughput.

| Operation | Input-size notation | Time / space relation | Illustration | Boundary |
|---|---|---|---|---|
| schema check | `C` named channels | `O(C)` time, `O(C)` manifest lookup | 12 names | does not inspect physiological correctness |
| sample/gap scan | `C*n` values | `O(C*n)` time; streaming state `O(C)` | 30,000 values | timestamp parsing may dominate if external |
| feature window | `C*w` values/window | `O(C*w)` per simple aggregate | 12×2,500 | transform-specific cost varies |
| lineage join | `R` records | expected `O(R)` with indexed immutable keys | 1,000 IDs | collision/duplicate policy is semantic |
| split audit | `E_train+E_eval` IDs | `O(E)` set construction | 80+20 encounters | does not prove biological independence |
| human review | `A` abstentions | `A*t_r` labor time | 20×3 min = 60 min | availability/queueing not modeled |

For the last row, the validity condition is one reviewer doing sequential
reviews with a stable three-minute service assumption. The arithmetic is
`20*3=60` reviewer-minutes. With two equally available reviewers and no
coordination cost, the idealized lower bound is 30 elapsed minutes, not a
guarantee. A counterexample is a queue with specialty routing: matching volume
does not imply that either reviewer is authorized or available. Record both
abstention count and routing-time distribution before making a capacity claim.

**Takeaway:** Linear byte or review arithmetic can size an illustrative
workload, but it cannot establish device performance or operational readiness.

### 11.11 Source and claim boundary register

| ID | Statement supported | Evidence label | Support available here | Does not establish | Review trigger |
|---|---|---|---|---|---|
| SIG-CLM-01 | `n=f_s*T` under fixed-rate acquisition | DERIVED; arithmetic VERIFIED | dimensional derivation and fixture | actual device sampling regularity | changed rate/clock |
| SIG-CLM-02 | 12×250×10 values imply 60,000 packed int16 bytes | ILLUSTRATIVE; VERIFIED | stated constructed inputs | storage/network cost | dtype/framing change |
| SIG-CLM-03 | training-only transform fitting prevents one identified leakage route | DERIVED | explicit time/split contract | generalization or clinical validity | split/feature change |
| SIG-CLM-04 | named mapping is required for positional-channel interpretation | HEURISTIC/contract | this sheet's invariant | vendor schema behavior | export/protocol change |
| SIG-CLM-05 | quality pass is not a diagnosis | DERIVED from output definition | declared semantic separation | target validity | target/workflow change |
| SIG-CLM-06 | particular device, filter, or patient outcome behavior | OPEN | no primary/local evidence in repository | any performance or safety claim | primary evidence supplied |

The course's source inventory supplies topic labels, not source-body proof. The
canonical detailed ECG/HRV/filter teaching commitment remains in Module 37; it
must be consulted or independently sourced before a target-specific signal
claim is made. This sheet deliberately does not turn that handoff into an
assertion about a dataset, device, or clinical workflow.

**Takeaway:** A claim register limits what a checked example means and tells the
next reviewer exactly which evidence could widen the conclusion.

### 11.12 Final pre-use gate

Before reusing this pattern, the accountable owner should answer each question
with an artifact rather than a verbal assurance:

| Gate | Required artifact | Fail-closed disposition | Why a substitute is insufficient |
|---|---|---|---|
| observation identity | immutable acquisition and segment IDs | no feature/output | an array hash cannot identify a repeated acquisition |
| channel semantics | approved versioned map, placement and polarity | schema rejection | equal column count does not show equivalent channels |
| measurement meaning | units and calibration evidence | no unit-dependent feature | finite numeric values are not calibrated measurements |
| timing | clock source plus availability trace | unavailable-at-decision | event time does not show arrival time |
| quality | declared rules and observed reason codes | abstain/reacquire/review | a pass rate hides individual failures |
| transformation | versioned causal/offline declaration | restrict to declared use | deterministic code can still look ahead |
| evaluation | independent-unit split and target maturity protocol | no comparative score | a random row split can share source lineage |
| action | owner, authorized route, timeout and fallback | baseline/manual route | a score does not allocate authority |

The gate is intentionally conservative. It does not assert that a record
failing one route is unusable for every retrospective research question; it
asserts that the route must be declared and its information boundary preserved.
An owner who cannot produce the artifact must preserve that uncertainty in the
disposition rather than converting it to a default pass.

**Takeaway:** The final decision is constrained by the weakest unproven
contract component, not by the most convincing waveform or model score.
## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Unit | named channel samples plus start/end timestamps and physical units |
| Sample count | `n=f_s*T`; only for fixed-rate, no-drop assumption |
| Time resolution | `Delta t=1/f_s`; do not invent sub-sample certainty |
| Quality gate | explicit channel, gap, saturation/artifact, and fallback rules |
| Data lineage | raw reference → versioned transform → segment/features → output |
| Leakage guard | fit transforms within training/evaluation split; respect decision window |
| Audit output | quality flags, exclusions, schema/version, and label linkage |
| Avoid | treating a waveform array, sample rate, or retrospective score as clinical validity |
