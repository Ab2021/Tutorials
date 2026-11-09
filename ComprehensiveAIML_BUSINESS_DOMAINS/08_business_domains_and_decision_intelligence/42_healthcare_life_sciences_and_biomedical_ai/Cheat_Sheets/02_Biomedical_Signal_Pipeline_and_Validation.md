---
collection: "healthcare-life-sciences-biomedical-ai"
file_id: "02"
topic: "Biomedical signal pipeline and validation"
level: 1
status: "draft — substantive expansion required before publication"
version: "0.1"
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

## 1. File Header

A biomedical signal is not just an array. Its meaning depends on sensing modality, channel identity, physical units, calibration, body placement, acquisition conditions, timestamps, filtering, segment selection, and label linkage. A pipeline should preserve enough metadata to reconstruct whether a value is physiological variation, sensor artifact, or an irreversible processing choice.

**Takeaway:** A waveform without acquisition context is an ambiguous measurement, not a reproducible model input.

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

### 4.4 Limit / counterexample

Upsampling a 100 Hz recording to 250 Hz adds interpolated points, not new physical information. Conversely, a high-rate file can still be unusable if one channel is disconnected, saturated, mislabeled, or time-shifted.

**Takeaway:** Sampling equations bound representation; they do not validate sensor placement, calibration, labeling, or downstream use.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE and arithmetic VERIFIED; not a measured clinical dataset.

A ten-second record has 12 named channels sampled at 250 Hz with a stated fixed-rate protocol. Per channel, `n=250*10=2,500`; total values `12*2,500=30,000`. If each sample is stored as a signed 16-bit value, payload is `30,000*2=60,000 bytes`, approximately `58.6 KiB` using 1 KiB = 1,024 bytes. This excludes headers, timestamps, encryption, compression, replication, and annotations.

Suppose the quality gate requires at least 9 of 12 channels present and no contiguous missing interval longer than 40 ms. At 250 Hz, 40 ms corresponds to `0.040/0.004=10` nominal sample intervals. A segment with 11 working channels but a 12-sample gap in one required channel fails the stated gate even if its total missing fraction looks small.

**Sensitivity:** At 500 Hz, the sample count and payload double to 60,000 values and 120,000 bytes under the same representation. The 40 ms threshold becomes 20 intervals. Changing rate requires changing sample-count checks and filter/feature validation; it is not a harmless rescale.

**Takeaway:** Make quality criteria physical-time and channel-contract statements, then map them to samples for a specific acquisition rate.

## 6. Visual / Geometric Intuition

Visualize the raw waveform as points on a time axis, each tagged with channel and unit. A filter changes the path between points; segmentation chooses a window; normalization changes coordinate scale. These are not neutral operations: a per-segment normalization can erase amplitude differences, while a global normalization fitted on all data can leak future/test distribution information.

Quality flags form a parallel track. Do not hide rejected samples by smoothing them into a plausible-looking curve. A reviewer should be able to see raw, derived, and masked intervals with the transformation version.

**Takeaway:** Preserve the ability to distinguish physiology, artifact, and preprocessing effect in the same time coordinate system.

## 7. Comparison / Selection Table

| Approach | Use when | Strength | Avoid when | Required check |
|---|---|---|---|---|
| raw waveform with provenance | audit/reprocessing is needed | preserves evidence | retention/access prohibits it | governed raw reference and access control |
| fixed filter and fixed windows | protocol is stable | reproducible comparison | signal timing/rate varies unaccounted | versioned parameters and time alignment |
| adaptive/artifact-aware processing | artifacts vary and review exists | can preserve usable segments | it masks failures or lacks trace | compare raw/derived and rejection rates |
| aggregate features | decision is defined on validated features | compact and interpretable | feature extraction uses future windows | point-in-time extraction test |

**Takeaway:** Choose a transformation because its assumptions match the acquisition and decision contract, not because it improves a retrospective score.

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

## 9. Complexity & Cost Analysis

For `C` channels and `n` samples per channel, basic scanning/filtering is commonly proportional to `O(Cn)` under fixed filter order; memory is `O(Cn)` if the entire record is retained, or lower for a streaming stateful filter. These are algorithmic statements, not device benchmarks. At the illustrative 30,000 values/record and 10,000 records/day, raw 16-bit payload alone is 600,000,000 bytes/day before metadata and retention overhead.

The operational cost is often quality review and lineage retention rather than arithmetic. Price, retention duration, and compliance requirements are not asserted here and require local sources.

**Takeaway:** Size storage from signal shape and representation, then separately account for the audit artifacts that make the analysis defensible.

## 10. Key Theorems / Results Reference

- **Sampling theorem (DERIVED scope):** under ideal band-limited assumptions, sampling above twice the maximum retained frequency prevents alias ambiguity in that model; real anti-alias design remains required.
- **Dimensional check:** `samples/s * s = samples`; `s/sample` is the reciprocal unit of sampling frequency.
- **Point-in-time invariance:** transformations that access future samples or labels change the intended real-time decision problem.
- **No semantic guarantee:** a valid sample rate does not establish valid calibration, label, clinical interpretation, or intervention benefit.

**Takeaway:** The theorem reference constrains signal representation, while the acquisition protocol constrains whether the represented signal means what the analysis assumes.

## 11. Interview / Exam Questions

1. **Calculate samples for 250 Hz over 10 seconds.** `250*10=2,500` per channel; state that drops violate the simple assumption.
2. **Why is 250 Hz not enough information to judge suitability?** Modality, intended frequency content, anti-alias filter, sensor fidelity, and target use are missing.
3. **What does a 40 ms gap equal at 250 Hz?** Ten nominal intervals; use timestamps if clock irregularity is material.
4. **Why retain raw and derived lineage?** To audit whether a feature/model result follows from the acquisition and transformation contract.
5. **Can interpolation repair aliasing?** No; it cannot recover frequency information lost at acquisition.
6. **How can normalization leak?** Fitting normalization statistics on test/future data transmits distribution information into training.
7. **What distinguishes a flatline from a stable physiology signal?** Device/channel quality checks, calibration, cross-channel context, and protocol—not a single heuristic.
8. **A device changes channel order. What fails?** Schema/identity contract; pin a mapping version and reject unknown layouts.
9. **What is a defensible quality-gate output?** pass/fail/abstain plus reason codes, measured signals, version, and fallback.
10. **What evidence is OPEN here?** Local acquisition protocol, target labels, device behavior, clinical validity, and original source body.

**Takeaway:** Defend every signal result by its acquisition contract, temporal alignment, and evidence boundary.

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
