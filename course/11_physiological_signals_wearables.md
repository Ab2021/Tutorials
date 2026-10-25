# Module 11 — Physiological Signals and Wearables

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 32 hours (12 lecture, 4 case clinic, 16 lab)
> **Track fit:** MedTech & Sensors core; elective for Health Systems
> **Prerequisites:** Modules 01, 06, 07
> **Tools:** Python, NumPy, SciPy, WFDB, NeuroKit2, PyTorch, a wearable dataset
> **Industry question:** "Does this signal-derived measurement mean the same thing in the lab, on the wrist, and in the home — and does it change care?"

---

## 1. Why industry cares

Physiological signals are the closest data we have to continuous biology. They power:

- Cardiac monitoring and arrhythmia detection.
- Sleep assessment.
- Remote patient monitoring.
- Deterioration detection.
- Rehabilitation and mobility.
- Mental health and stress research.
- Clinical trials with digital endpoints.
- Consumer wellness products.

They also generate enormous artifact, privacy, and validation problems. A wrist-worn photoplethysmography (PPG) signal is not a hospital ECG. A single-lead wearable is not a 12-lead diagnostic ECG. A sleep-stage algorithm trained on one device may fail on another.

This module teaches signal processing, machine learning, validation, and productization for physiological data. For sensor physics, analog front ends, chips, firmware, and device hardware, see Module 14A.

## 2. Learning outcomes

By the end of this module, you can:

- Explain sampling, aliasing, filtering, and time-frequency analysis for biomedical signals.
- Assess signal quality and artifact for ECG, PPG, EEG, and respiratory data.
- Build feature-based and deep-learning pipelines for signal classification.
- Design patient-level and device-held-out validation.
- Validate a digital biomarker analytically and clinically.
- Evaluate wearable algorithms for skin-tone, body-composition, and activity bias.
- Design an edge or cloud deployment with latency and privacy constraints.
- Interpret regulatory requirements for consumer vs clinical claims.
- Write a signal validation report and monitoring plan.

## 3. Mental model: signal → measurement → biomarker → decision

```
Sensor → Analog front end → Sampling → Quality control → Segmentation → Feature/representation
   → Measurement → Digital biomarker → Clinical validation → Decision → Monitoring
```

A signal is not a measurement. A measurement is not a biomarker. A biomarker is not a clinical decision.

## 4. Signal fundamentals

### 4.1 Sampling and aliasing

- **Nyquist theorem:** sample at more than twice the highest frequency of interest.
- **Aliasing:** high-frequency content folds into lower frequencies if undersampled.
- **Resolution:** ADC bits determine amplitude resolution.
- **Timing:** clock drift and jitter affect interval measurements.

| Signal | Typical sampling rate | Bandwidth of interest |
|---|---:|---:|
| ECG | 250–1,000 Hz | 0.05–150 Hz |
| PPG | 25–1,000 Hz | 0.5–10 Hz |
| EEG | 200–1,000 Hz | 0.5–100 Hz |
| Respiratory | 10–100 Hz | 0.1–3 Hz |
| Accelerometry | 25–100 Hz | 0–20 Hz |
| EMG | 1,000–10,000 Hz | 20–500 Hz |

### 4.2 Filtering

- **Low-pass:** remove high-frequency noise.
- **High-pass:** remove baseline wander.
- **Band-pass:** isolate signal band.
- **Notch:** remove powerline interference (50/60 Hz).
- **Median filter:** remove spikes.

**Rule:** filter cutoffs must be justified by the physiology and the task, and applied consistently at training and inference.

### 4.3 Time-frequency analysis

- **Short-time Fourier transform:** frequency content over time.
- **Wavelet transform:** multi-resolution analysis.
- **Spectrogram:** visual time-frequency representation.
- **Empirical mode decomposition:** adaptive decomposition.

Useful for EEG, ECG, and PPG, where frequency content changes over time.

### 4.4 Quality indices

| Signal | Quality indicators |
|---|---|
| ECG | Baseline wander, muscle noise, lead off, clipping, QRS detectability |
| PPG | Perfusion index, motion, ambient light, signal-to-noise ratio |
| EEG | Electrode impedance, muscle artifact, eye blinks, line noise |
| Respiratory | Motion, sensor displacement, breath detectability |
| Accelerometry | Non-wear, saturation, orientation changes |

**Rule:** reject or flag low-quality segments. Do not let artifact become a label.

## 5. Modalities and tasks

### 5.1 ECG

Tasks:

- Rhythm classification.
- Arrhythmia detection (AF, PVC, PAC).
- Ischemia detection.
- QT measurement.
- Heart-rate variability.
- Deterioration prediction.

Challenges:

- Lead placement.
- Device differences.
- Paroxysmal events.
- Patient-level leakage.
- Label adjudication.

### 5.2 PPG

Tasks:

- Heart rate.
- Oxygen saturation.
- Atrial fibrillation screening.
- Blood pressure estimation (research).
- Sleep staging.
- Stress and hydration (emerging).

Challenges:

- Motion artifact.
- Skin pigmentation.
- Perfusion.
- Ambient light.
- Wrist vs finger vs ear.

### 5.3 EEG

Tasks:

- Seizure detection.
- Sleep staging.
- Anesthesia depth.
- Brain-computer interfaces.
- Neurologic prognosis.

Challenges:

- Montage and electrode placement.
- Artifact.
- Inter-reader variability.
- Long recordings.

### 5.4 Respiratory signals

Tasks:

- Apnea detection.
- Respiratory rate.
- Respiratory effort.
- Weaning.
- Deterioration.

Challenges:

- Sensor displacement.
- Motion.
- Speech and swallowing.

### 5.5 Accelerometry

Tasks:

- Activity recognition.
- Fall detection.
- Gait analysis.
- Sleep-wake.
- Adherence.

Challenges:

- Device placement.
- Non-wear.
- Activity confounders.

## 6. Feature engineering

| Feature family | Examples |
|---|---|
| Time domain | Mean, variance, skewness, kurtosis, zero-crossings |
| Frequency domain | Power spectral density, band power, dominant frequency |
| Time-frequency | Wavelet coefficients, spectrogram patches |
| Morphology | QRS width, R-peak amplitude, PPG pulse shape |
| Variability | RMSSD, SDNN, pNN50, entropy |
| Cross-signal | Pulse transit time, respiratory sinus arrhythmia |
| Context | Activity, posture, time of day, temperature |

**Rule:** features must be robust to device, sampling rate, and artifact. Document the preprocessing chain.

## 7. Deep learning for signals

### 7.1 Representations

- Raw waveform (1D CNN).
- Spectrogram or scalogram (2D CNN).
- Patch/token sequences (transformer).
- Learned embeddings (foundation models).

### 7.2 Architectures

- 1D CNN for local morphology.
- ResNet for deeper features.
- LSTM/GRU for temporal dependence.
- Transformer for long-range context.
- Hybrid CNN + transformer.

### 7.3 Training

- Patient-level splits.
- Segment-level labels with patient aggregation.
- Class imbalance and label noise.
- Augmentation: time shift, amplitude scaling, noise injection, lead dropout.
- Calibration.

## 8. Digital biomarkers

### 8.1 Definition

A digital biomarker is a characteristic derived from digital data that indicates a biological or behavioral process, disease state, or response to intervention.

### 8.2 Validation stages

| Stage | Question | Evidence |
|---|---|---|
| Analytical validation | Does the sensor measure the signal accurately? | Bench testing, reference device |
| Technical validation | Does the algorithm produce the intended measurement? | Agreement with reference |
| Clinical validation | Is the measurement associated with the clinical outcome? | Prospective cohort |
| Clinical utility | Does using it improve decisions or outcomes? | Impact study |
| Usability | Can people use it correctly? | Human factors testing |

### 8.3 Agreement metrics

- Bland-Altman plot.
- Mean absolute error.
- Limits of agreement.
- Intraclass correlation.
- Sensitivity/specificity for threshold-based biomarkers.

## 9. Validation and bias

### 9.1 Reference standard

- ECG for arrhythmia.
- Polysomnography for sleep.
- Arterial line for blood pressure.
- Lab test for glucose.
- Clinical adjudication for events.

The reference standard has its own error. Report it.

### 9.2 Free-living vs lab

Lab validation may not generalize to home use. Free-living data include motion, poor placement, and variable adherence. Validate in the intended setting.

### 9.3 Skin tone and body composition

PPG and pulse oximetry can be affected by skin pigmentation, perfusion, and body composition. Report performance across Fitzpatrick skin types or other appropriate measures, and avoid claiming fairness without evidence.

### 9.4 Activity and behavior

Heart rate and sleep estimates are confounded by activity, stress, caffeine, alcohol, and illness. A model may learn behavior rather than physiology.

### 9.5 Adherence and missingness

Wearable data are missing when the device is off, charging, or not worn. Missingness is informative. Do not impute long gaps blindly.

## 10. Deployment

| Consideration | Edge | Cloud |
|---|---|---|
| Latency | Low | Network-dependent |
| Privacy | Data stay on device | Data transfer and storage |
| Compute | Constrained | Scalable |
| Model size | Quantized/pruned | Larger |
| Updates | App/OTA | Server-side |
| Battery | Critical | Less critical |
| Monitoring | Device telemetry | Centralized |

**Rule:** the architecture must match the clinical latency and privacy requirements.

## 11. Industry case studies

### 11.1 Apple Heart Study

The Apple Heart Study used a smartwatch irregular-rhythm notification to identify atrial fibrillation. It demonstrated feasibility at scale but also highlighted false positives and the need for confirmatory ECG.

**Lesson:** consumer screening generates downstream clinical workload; the care pathway is part of the product.

Source: Perez, M. V. et al., *New England Journal of Medicine*, 2019. https://doi.org/10.1056/NEJMoa1901183

### 11.2 PhysioNet 2025 Chagas challenge

The challenge used ECGs to prioritize patients for scarce serological testing, with a top-5% ranking metric.

**Lesson:** capacity-constrained metrics reflect real screening programs better than threshold-free AUROC.

Source: https://moody-challenge.physionet.org/2025/

### 11.3 PhysioNet 2026 sleep and cognition

The challenge predicted later cognitive impairment from polysomnography and used age-conditioned AUROC to remove the age shortcut.

**Lesson:** metric design can prevent a model from exploiting an obvious confounder.

Source: https://moody-challenge.physionet.org/2026/

### 11.4 Wearable sleep staging

Consumer sleep staging has been validated against polysomnography with variable agreement. Performance depends on device, population, and sleep disorder.

**Lesson:** sleep stages are not equally detectable; report per-stage agreement and clinical limitations.

Source: Chinoy, E. D. et al., "Performance of seven consumer sleep-tracking devices compared with polysomnography," *Sleep*, 2021. https://doi.org/10.1093/sleep/zsaa291

### 11.5 Remote patient monitoring for heart failure

Remote monitoring can reduce readmissions in selected populations, but benefit depends on adherence, alert management, and integration with care.

**Lesson:** the intervention and workflow determine success, not the sensor alone.

Source: Inglis, S. C. et al., "Structured telephone support or non-invasive telemonitoring for patients with heart failure," *Cochrane Database of Systematic Reviews*, 2015. https://doi.org/10.1002/14651858.CD007228.pub3

### 11.6 Apple hypertension notification (2025)

Apple received FDA clearance in September 2025 for a passive hypertension notification feature using PPG over 30-day windows. It does not measure blood pressure. In the validation study of 2,229 participants, sensitivity was 41.2% overall (53.7% for Stage 2 hypertension) and specificity was 92.3%.

**Lesson:** a narrow claim, a clear limitation ("not a blood pressure measurement"), and a large validation set can support a consumer-facing notification. High specificity reduces false alarms, but modest sensitivity means many hypertensive users will not be notified. The product must communicate that limitation honestly.

Source: Apple validation paper, September 2025. https://www.apple.com/health/pdf/Hypertension_Notifications_Validation_Paper_September_2025.pdf

## 12. Labs

### Lab 11.1 — ECG quality and R-peak detection

**Objective:** Assess ECG quality and detect R peaks.

**Steps:**

1. Load a 12-lead or single-lead ECG.
2. Apply baseline removal and band-pass filtering.
3. Detect R peaks with a standard algorithm.
4. Compute heart rate and RR intervals.
5. Flag noisy segments.
6. Compare with annotations.

**Deliverable:** signal-quality report.

**Acceptance criteria:** correct units; artifact flagged; agreement quantified.

### Lab 11.2 — AF classification

**Objective:** Classify AF vs normal rhythm.

**Steps:**

1. Split by patient.
2. Extract RR-interval and morphology features.
3. Train logistic regression and a 1D CNN.
4. Evaluate AUROC, sensitivity, specificity, and calibration.
5. Analyze false positives.

**Deliverable:** model and evaluation.

**Acceptance criteria:** patient-level split; calibration; error analysis.

### Lab 11.3 — PPG artifact and heart rate

**Objective:** Estimate heart rate from PPG and quantify artifact.

**Steps:**

1. Filter and segment the signal.
2. Detect systolic peaks.
3. Compute heart rate.
4. Compare with ECG reference.
5. Stratify by motion and skin tone if available.

**Deliverable:** validation report.

**Acceptance criteria:** Bland-Altman; subgroup analysis; limitations.

### Lab 11.4 — Sleep staging

**Objective:** Train a sleep-stage classifier on PSG or wearable data.

**Steps:**

1. Segment into 30-second epochs.
2. Extract spectral and time-domain features.
3. Train a model.
4. Evaluate per-stage kappa and confusion matrix.
5. Analyze errors by sleep disorder or device.

**Deliverable:** sleep model and evaluation.

**Acceptance criteria:** per-stage metrics; patient-level split; clinical limitations.

### Lab 11.5 — Edge deployment

**Objective:** Quantize and benchmark a signal model for edge inference. For the full distillation/quantization workflow, see Module 14B.

**Steps:**

1. Train a small CNN.
2. Quantize to int8.
3. Measure latency and memory.
4. Compare accuracy.
5. Estimate battery impact.

**Deliverable:** edge benchmark.

**Acceptance criteria:** latency and memory measured; accuracy trade-off reported.

### Lab 11.6 — Missingness and adherence

**Objective:** Analyze non-wear and its impact on a digital biomarker.

**Steps:**

1. Identify gaps.
2. Classify missingness mechanisms.
3. Compare complete-case and imputed estimates.
4. Assess bias by subgroup.
5. Recommend a handling strategy.

**Deliverable:** missingness report.

**Acceptance criteria:** mechanisms distinguished; bias quantified.

## 13. Assignment: digital biomarker validation plan

Write a 2,000-word plan for a wearable-derived biomarker:

1. Intended use and decision.
2. Sensor and signal.
3. Analytical validation.
4. Technical validation.
5. Clinical validation.
6. Usability and adherence.
7. Bias and equity.
8. Deployment architecture.
9. Regulatory pathway.
10. Monitoring plan.

## 14. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Signal understanding | 20% | Applies generic preprocessing | Correct filtering and quality control | Links processing to physiology |
| Model | 20% | Single architecture | Baselines and validation | Robustness and calibration |
| Validation | 25% | Random split | Patient-level and external | Free-living and subgroup |
| Bias | 15% | Ignores | Basic subgroup analysis | Mechanisms and mitigation |
| Deployment | 10% | Cloud only | Edge/cloud trade-off | Latency, battery, privacy |
| Communication | 10% | Technical | Clear | Clinical and regulatory ready |

## 15. Regulatory, ethical, and commercial lens

- **FDA:** a wearable algorithm that diagnoses or treats disease may be a device. General wellness claims have limits.
- **EU MDR:** clinical evaluation and post-market surveillance.
- **Privacy:** continuous physiological data are sensitive. Minimize collection and retention.
- **Consent:** consumers may not understand what is inferred from their data.
- **Equity:** device access and sensor performance vary by skin tone, body composition, and income.
- **Commercial:** reimbursement for remote monitoring varies; workflow and adherence drive value.

## 16. Failure modes and red flags

- Random segment split with patient leakage.
- Filtering after segmentation, causing leakage.
- Ignoring device and sampling-rate differences.
- Treating PPG as ECG.
- No artifact rejection.
- No free-living validation.
- No skin-tone or body-composition analysis.
- Imputing long non-wear gaps.
- Claiming clinical validity from consumer data.
- No plan for battery or connectivity.

## 17. Interview and discussion questions

1. **What is aliasing?**
   Hint: undersampling high-frequency content into lower frequencies.
2. **How do you assess ECG quality?**
   Hint: baseline, noise, clipping, QRS detectability, lead-off.
3. **Why is patient-level splitting essential?**
   Hint: repeated segments from the same patient leak identity and morphology.
4. **How do you validate a PPG heart-rate algorithm?**
   Hint: reference ECG, Bland-Altman, subgroups, motion.
5. **What is a digital biomarker?**
   Hint: digital data-derived indicator of biology, behavior, or treatment response.
6. **What are the stages of biomarker validation?**
   Hint: analytical, technical, clinical, utility, usability.
7. **How do you handle non-wear?**
   Hint: distinguish mechanisms, avoid blind imputation, sensitivity analysis.
8. **What are the risks of consumer AF screening?**
   Hint: false positives, anxiety, downstream workload, confirmation.
9. **How would you deploy a model on a watch?**
   Hint: quantization, latency, battery, privacy, over-the-air updates.
10. **How do you test for skin-tone bias?**
    Hint: stratified performance, appropriate measurement, diverse validation.
11. **What is the difference between lab and free-living validation?**
    Hint: artifact, adherence, context, generalizability.
12. **How do you monitor a wearable model?**
    Hint: device mix, firmware, missingness, calibration, subgroup performance.

## 18. Key terms

| Term | Definition |
|---|---|
| Sampling rate | Samples per second |
| Nyquist limit | Half the sampling rate |
| Aliasing | High-frequency folding |
| Baseline wander | Low-frequency drift |
| PPG | Photoplethysmography |
| HRV | Heart-rate variability |
| Digital biomarker | Digital data-derived health indicator |
| Analytical validation | Sensor/measurement accuracy |
| Clinical validation | Association with clinical outcome |
| Free-living | Data collected outside the lab |
| Non-wear | Time device is not worn |
| Edge inference | Computation on the device |
| Quantization | Reducing numerical precision |

## 19. Sources and further reading

### Datasets and tools
- PhysioNet: https://physionet.org/
- WFDB: https://physionet.org/content/wfdb/
- NeuroKit2: https://neurokit2.readthedocs.io/
- Sleep-EDF: https://physionet.org/content/sleep-edfx/
- TUH EEG: https://www.isip.piconepress.com/projects/tuh_eeg/
- PTB-XL: https://physionet.org/content/ptb-xl/
- Apple Heart Study: https://med.stanford.edu/appleheartstudy.html

### Methods and evidence
- Perez, M. V. et al., "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation," *New England Journal of Medicine*, 2019. https://doi.org/10.1056/NEJMoa1901183
- Chinoy, E. D. et al., "Performance of seven consumer sleep-tracking devices compared with polysomnography," *Sleep*, 2021. https://doi.org/10.1093/sleep/zsaa291
- Inglis, S. C. et al., "Structured telephone support or non-invasive telemonitoring for patients with heart failure," *Cochrane Database of Systematic Reviews*, 2015. https://doi.org/10.1002/14651858.CD007228.pub3
- Goldberger, A. L. et al., *Goldberger's Clinical Electrocardiography*.
- Thaler, M. S., *The Only EKG Book You'll Ever Need*.

### Guidance
- FDA, *General Wellness: Policy for Low Risk Devices*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-wellness-policy-low-risk-devices
- FDA, *Digital Health Center of Excellence*. https://www.fda.gov/medical-devices/digital-health-center-excellence
- EU MDR: https://eur-lex.europa.eu/eli/reg/2017/745/oj
- Digital Medicine Society: https://www.dimesociety.org/

> **Next:** [Module 12](12_omics_precision_medicine.md) moves from physiology to molecules, cells, and populations.
