# Sensor Fusion, IoT and Real-Time Perception — Cheat Sheets

**Status:** Two research-education drafts; other topics are planned. Follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master. This material is not clinical guidance, device approval or validation evidence.

## 1. Overview

This collection is for engineers reasoning about sampled physical signals and edge systems. Its current draft develops electrocardiogram (ECG) sampling, aliasing and spectral interpretation. Readiness means explaining the analog-to-digital boundary, computing simple sampling quantities and stating when a real signal or device violates the model.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Practitioner | [ECG sampling and aliasing](01_ecg_sampling_and_aliasing.md) | Calculate sample spacing, frequency folding and DFT bin spacing; requires basic signals and units | Draft; unreviewed |
| Planned | Sensor fusion patterns | Combine sensors with stated timing, uncertainty and failure assumptions | Not written |
| Practitioner | [Sensor time synchronization and calibration](02_Sensor_Time_Sync_and_Calibration.md) | State clock, frame, uncertainty and recalibration contracts | Draft; unreviewed |
| Planned | Telemetry schema | Define timestamps, provenance, units and quality flags | Not written |
| Planned | Common Issues | Distinct collection-wide sensor diagnostics | Not written |

## 3. Quick topic lookup

| Question | Reference | Condition and warning |
|---|---|---|
| Can a high frequency appear as a low frequency? | [Worked example](01_ecg_sampling_and_aliasing.md#5-worked-numerical-example) | Folding is irreversible without additional information; realistic signals and filters need separate validation. |
| What does a DFT bin spacing mean? | [Formula section](01_ecg_sampling_and_aliasing.md#4-core-formulas-with-full-derivation) | Bin spacing is not automatically the ability to resolve two nearby components. |
| Is digital filtering sufficient before sampling? | [Failure modes](01_ecg_sampling_and_aliasing.md#8-failure-modes--edge-cases) | Digital processing after conversion cannot generally undo aliasing introduced at the ADC. |

## 4. Reading paths

- **Beginner:** Review time, frequency, analog-to-digital conversion and basic trigonometry; read the ECG sheet's mental model and definitions. Sensor-fusion fundamentals remain planned.
- **Practitioner:** Read the complete ECG sheet, calculate the illustrative sampling and aliasing examples, and identify the analog filtering and resampling assumptions.
- **Advanced:** Assess clock drift, finite-window spectra, artifacts, acquisition provenance and resampling; validate on appropriate equipment and data. Planned calibration and telemetry sheets extend this path.

## 5. Prerequisite skills

Use seconds, hertz and amplitudes consistently; understand sampling periods, discrete Fourier transform (DFT) windows, anti-alias filtering and analog-to-digital converters (ADCs). Understand that public research recordings are not clinical device certifications.

## 6. What this collection does not cover

The current draft is not a diagnosis, treatment recommendation, validated ECG algorithm, electrical safety design, or medical device verification plan. It does not cover the full fusion, calibration and telemetry curriculum, which remains planned.

## 7. Master symbol glossary

| Symbol | Meaning and units | Scope / warning | Reference |
|---|---|---|---|
| f_s | Sampling frequency, hertz | Actual clock accuracy and jitter may differ from a nominal rate | [Definitions](01_ecg_sampling_and_aliasing.md#3-foundational-definitions) |
| T_s | Sampling interval, seconds | T_s = 1/f_s for a uniform positive sampling rate | [Formula section](01_ecg_sampling_and_aliasing.md#4-core-formulas-with-full-derivation) |
| N | Number of acquired samples | Observation duration and window choice affect spectral interpretation | [Worked example](01_ecg_sampling_and_aliasing.md#5-worked-numerical-example) |
| f_s/2 | Nyquist frequency, hertz | An analog low-pass anti-alias filter needs a realizable transition band | [ECG sheet](01_ecg_sampling_and_aliasing.md) |

## 8. Quick Reference Card directory

| Card | Core rule | Use and critical condition |
|---|---|---|
| [ECG sampling and aliasing](01_ecg_sampling_and_aliasing.md#12-quick-reference-card) | Sampling spacing is the reciprocal of sample rate. | Uniform sampling and a suitable analog anti-alias filter are required for the simplified model. |

## 9. Common Issues locator

A standalone Common Issues artifact is **planned, not written**; no collection-wide issue total is claimed. Consult the draft's [failure modes and edge cases](01_ecg_sampling_and_aliasing.md#8-failure-modes--edge-cases) for clock, filtering and spectral-interpretation diagnostics.

## 10. Sources, versions and maintenance

The ECG sheet uses illustrative arithmetic and cites the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) for its documented dataset characteristics. The original inventoried course source was unavailable; do not infer its unseen content. Review on dataset, acquisition hardware, sample-rate, filter or intended-use changes.

## 11. Collection acceptance checklist

- [ ] Planned sensor topics and a distinct Common Issues artifact are authored and reviewed.
- [ ] Each sheet retains all 12 sections, checked calculations or traces, failure diagnostics, answered interviews and a final card.
- [ ] Dataset, hardware, intended-use, safety and clinical limitations are explicit.
- [ ] Links, glossary, reading paths, issue locations and editorial statuses match the actual files.
