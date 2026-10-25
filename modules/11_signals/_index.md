# Module 11 — Physiological Signals and Wearables

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../11_physiological_signals_wearables.md`](../../11_physiological_signals_wearables.md)
> **Folder:** `modules/11_signals/`
> **Version:** 2026.3

---

## 1. Module purpose

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01A, 06 |
| Unlocks | 14A, 14B, 18 |
| Assessment | Digital biomarker validation plan (Week 19) |
| Survey file | `11_physiological_signals_wearables.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_signal-fundamentals.md`](01_signal-fundamentals.md) | Signal Fundamentals | Sampling, aliasing, filtering, noise, and the signal chain. | 1,200-1,800 | complete |
| 2 | [`02_ecg.md`](02_ecg.md) | ECG | Leads, intervals, rhythms, quality indices, and automated interpretation. | 1,200-1,800 | planned |
| 3 | [`03_eeg.md`](03_eeg.md) | EEG | Montages, frequency bands, artifacts, seizures, and sleep staging. | 1,200-1,800 | planned |
| 4 | [`04_ppg-and-wearables.md`](04_ppg-and-wearables.md) | PPG and Wearables | Optical sensing, SpO2, heart rate, and motion artifact. | 1,200-1,800 | planned |
| 5 | [`05_respiration-and-other-signals.md`](05_respiration-and-other-signals.md) | Respiration and Other Signals | Respiratory effort, airflow, EMG, EDA, and continuous glucose. | 1,200-1,800 | planned |
| 6 | [`06_feature-engineering.md`](06_feature-engineering.md) | Feature Engineering | Time-domain, frequency-domain, and morphology features with clinical meaning. | 1,200-1,800 | planned |
| 7 | [`07_deep-learning-for-signals.md`](07_deep-learning-for-signals.md) | Deep Learning for Signals | 1D CNNs, recurrent models, transformers, and augmentation. | 1,200-1,800 | planned |
| 8 | [`08_digital-biomarkers.md`](08_digital-biomarkers.md) | Digital Biomarkers | Definition, qualification, and the evidence a payer or regulator accepts. | 1,200-1,800 | planned |
| 9 | [`09_validation.md`](09_validation.md) | Validation | Reference standards, co-primary endpoints, and confounding by activity. | 1,200-1,800 | planned |
| 10 | [`10_deployment-on-device.md`](10_deployment-on-device.md) | Deployment on Device | Latency, power, memory, and firmware constraints for signal models. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Signal Fundamentals ->
02 ECG ->
03 EEG ->
04 PPG and Wearables ->
05 Respiration and Other Signals ->
06 Feature Engineering ->
07 Deep Learning for Signals ->
08 Digital Biomarkers ->
09 Validation ->
10 Deployment on Device
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 9 |
| draft | 0 |
| complete | 1 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

