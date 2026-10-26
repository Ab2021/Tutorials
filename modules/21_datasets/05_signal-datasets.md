# Signal Datasets

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 05 of 12
> Estimated study time: 4 hours
> Prerequisites: Topic 01, Topic 02
> Version 2026.3

## 1. Why this topic exists

A physiological signal dataset is a device dataset. The sampling rate, the analogue front end, the filter, the electrode placement, the firmware, and the subject's motion are all part of the distribution. Two datasets of the same modality can look identical in a table and be incompatible in a model. A team that treats "ECG" as one data type will train on one device and deploy on another.

The failures are specific and expensive:

- A model is trained on 12-lead clinical ECGs and deployed on a single-lead wrist device. The lead count, the axis, and the noise floor differ, and the model's learned morphology does not transfer.
- Windows from the same recording are split across training and test. Adjacent windows share the patient's rhythm and noise, so the reported performance reflects memorisation.
- A dataset is labelled from a cardiologist's statement rather than from an adjudicated episode. The label is a summary of an interpretation, not a measurement of the signal.
- A challenge metric is optimised without reading it. A metric based on precision at the top 5 percent of a ranked cohort implies a capacity constraint that the model does not otherwise respect.
- A dataset of 15 healthy volunteers under a laboratory stress protocol is used to claim clinical stress detection.
- A polysomnography dataset scored under one version of the AASM manual is pooled with another scored under a different version.

Signal data are cheap to collect and expensive to label. A wearable can produce days of waveform per subject, and the annotated events are a tiny fraction of the record. The annotation depth, not the sample count, is the binding constraint. A dataset with 10,000 hours and 50 annotated events is a small dataset for the event.

This topic covers the major signal collections: PTB-XL, the PhysioNet Challenge 2025 Chagas data, the 2026 polysomnography challenge, Sleep-EDF, TUH EEG, MIMIC-IV waveform, Icentia11K, WESAD, and NSRR. It treats signal quality, device shift, window leakage, and metric-to-decision mapping as the core mechanics.

The asymmetry between data volume and annotation volume is the defining constraint. A wearable produces millions of windows per subject-day and a handful of adjudicated events. A team that counts windows will believe it has a large dataset; a team that counts events will know it has a small one. The event count is the denominator that matters for the claim.

A second framing point: the signal is not the physiology. It is a device's rendering of the physiology under a protocol. The model learns the rendering, and the evaluation must test the rendering it will see in deployment.

> **Note:** Counts and versions were checked on 26 October 2026. PTB-XL v1.0.3 contains 21,799 records from 18,869 patients; the earlier v1.0.0 counts of 21,837 and 18,885 are still widely cited. Use the version and DOI that match the file you downloaded.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** clinical, ambulatory, wearable, and ICU signal datasets by device, placement, and annotation depth. Assessed by Exercise 1 and self-check item 1.
2. **Audit** window-level splits for patient and record leakage, and redesign the split at the record or patient level. Assessed by Exercise 2.
3. **Evaluate** label provenance for a signal dataset, including statement-derived labels and episode adjudication. Assessed by Exercise 3.
4. **Map** a challenge metric to the operational constraint it encodes, and compute precision at capacity under a prevalence shift. Assessed by Exercise 4.
5. **Check** sampling rate, filtering, and electrode placement against the deployment device. Assessed by Exercise 5.
6. **Design** a preprocessing pipeline that fits filters and normalisation on training records only. Assessed by Exercise 6.
7. **Detect** device shift and site shift between development and evaluation data. Assessed by Exercise 7.
8. **Write** a signal data card that states the device, the annotation depth, and the claims the data cannot support. Assessed by Exercise 8.

## 3. The core mental model

A signal dataset is a chain from physiology to device to annotation to metric. Each link transforms the data, and each transformation is a potential source of shift.

```text
   PHYSIOLOGY
   electrical | mechanical | optical | acoustic
        |
        v
   SENSOR AND DEVICE
   electrode | photodiode | accelerometer | firmware | sampling rate
        |
        v
   ACQUISITION
   placement | contact | motion | ambient noise | filter
        |
        v
   RECORDING
   single lead | multi-lead | channel set | duration
        |
        v
   ANNOTATION
   adjudicated episode | statement | algorithm | none
        |
        v
   WINDOWING AND MODEL
   window length | overlap | sampling | split
        |
        v
   METRIC
   AUROC | sensitivity at capacity | event detection | time-to-alarm
```

**The device rule.** The device defines the distribution. A 12-lead clinical ECG at 500 Hz, a single-lead ambulatory patch at 200 Hz, and a wrist photoplethysmogram at 25 Hz are different measurement systems. A model trained on one is not a model for another without recalibration and validation.

**The annotation rule.** Annotation depth matters more than record count. A dataset with 10,000 records and statement-level labels can support a classification benchmark; a dataset with 500 records and adjudicated episode onset can support an event-detection claim. The claim determines which dataset is adequate.

**The window rule.** A signal is continuous. Windowing creates samples, and overlapping windows create near-duplicates. Split by record or patient, not by window. A random window split is the signal equivalent of an image-level split.

**The metric rule.** Challenge metrics encode operational constraints. Precision at the top 5 percent encodes a fixed review capacity. Age-conditioned AUROC removes a trivial age shortcut. A metric that does not match the decision produces a model that ranks well and operates badly.

**The quality rule.** Signal quality is not missingness. A noisy segment is present and uninterpretable. Quality indices, lead-off detection, and motion flags are part of the data and should be features or exclusion criteria, not silent filters.

### Signal chain failures

| Stage | Failure | Observable symptom | Consequence |
|---|---|---|---|
| Sensor | Poor contact or dry electrode | High impedance, flat segments | Lost signal |
| Placement | Electrode misplacement | Wrong axis or amplitude | Morphology shift |
| Front end | Saturation or clipping | Flat-topped peaks | Unreadable events |
| Filter | Wrong cut-off | Distorted ST segment or missing high frequency | Measurement bias |
| Sampling | Aliasing | High-frequency noise in the band | Spurious features |
| Transmission | Dropped packets | Gaps in the record | False asystole |
| Storage | Compression | Loss of fine detail | Subtle events missed |
| Annotation | Missing events | Low event count | Overstated sensitivity |
| Windowing | Overlap | Near-duplicate samples | Leakage |
| Splitting | Window-level split | High test performance | Memorisation |

| Layer | Question | Artefact | Failure if skipped |
|---|---|---|---|
| Device | What sensor, placement, and sampling rate? | Device specification | Domain shift at deployment |
| Acquisition | What filters and quality flags? | Acquisition report | Noise learned as signal |
| Annotation | What is the label and who made it? | Provenance record | Metric measures interpretation |
| Windowing | What is a sample and how is it split? | Split specification | Window leakage |
| Metric | Does it encode the decision? | Metric memo | Wrong operating point |

## 4. Definitions and vocabulary

### 4.1 Signal and device terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Sampling rate | Samples per second. | Assumed sufficient without a bandwidth check. | Hz |
| Nyquist frequency | Half the sampling rate. | Ignored when downsampling. | Hz |
| Bandwidth | The frequency range preserved by the system. | Confused with sampling rate. | Hz |
| Filter | A transformation that removes frequency content. | Applied differently across sites. | Filter type and cut-off |
| Notch filter | A narrow filter at a specific frequency, often line noise. | Removes signal at the same frequency. | Hz |
| Baseline wander | Low-frequency drift from respiration or movement. | Removed with a filter that distorts ST segments. | Hz |
| Electrode | A sensor that transduces electrical activity. | Assumed identical across devices. | Type and placement |
| Lead | A potential difference between electrodes. | Confused with an electrode. | Channel |
| Montage | The arrangement of electrodes and channels. | Different across EEG systems. | Configuration |
| Photoplethysmography | Optical measurement of blood volume change. | Treated as equivalent across skin tones. | Signal |
| Accelerometer | A sensor measuring acceleration. | Ignored as metadata. | m/s² |
| Firmware | Device software that shapes the signal. | Not recorded. | Version |
| Dynamic range | The range of amplitudes a device can record. | Clipping treated as artefact. | Decibels or bits |

### 4.2 Annotation and label terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Episode | A period of abnormal activity. | Confused with a record-level label. | Start and end time |
| Onset | The start of an episode. | Not annotated. | Timestamp |
| Offset | The end of an episode. | Not annotated. | Timestamp |
| Statement | A diagnostic statement attached to a record. | Treated as an episode label. | Code |
| SCP-ECG statement | A standard ECG statement code. | Treated as a diagnosis. | Code |
| Adjudication | Resolution of disagreement. | Assumed when one reader annotated. | Process |
| Inter-rater agreement | Agreement between annotators. | Reported without the scale. | Kappa |
| Weak label | A noisy or indirect label. | Treated as clean. | Label type |
| Event detection | Identifying the presence and time of an event. | Reduced to record classification. | Task |
| Alarm burden | The number of alarms per unit time. | Ignored in metric choice. | Alarms per day |
| Annotation depth | The proportion of events that are annotated. | Assumed complete. | Proportion |

### 4.3 Evaluation terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Window | A fixed-length segment of signal. | Treated as independent. | Seconds |
| Overlap | The share of signal shared by adjacent windows. | Ignored in splitting. | Proportion |
| Record-level split | Keeping all windows of a record on one side. | Replaced by a window split. | Split key |
| Patient-level split | Keeping all records of a patient on one side. | Absent when patient IDs are missing. | Split key |
| Precision at capacity | True positives among the top k ranked. | Replaced by AUROC. | Proportion |
| Sensitivity at capacity | True positives detected at a fixed review rate. | Reported without the rate. | Proportion |
| Time-to-alarm | Delay between event onset and alarm. | Ignored in window classification. | Seconds |
| False alarm rate | False alarms per unit time. | Reported per window. | Alarms per hour |
| Calibration | Agreement between predicted risk and observed frequency. | Omitted for event detection. | Slope |
| Event-level metric | Performance on episodes rather than windows. | Replaced by window accuracy. | Proportion |

### 4.4 Quality and processing terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Impedance | Resistance at the electrode-tissue interface. | Ignored until the signal fails | Ohms |
| Saturation | Clipping at the device's dynamic range. | Treated as a valid extreme value | Flag |
| Flatline | A segment with no variation. | Treated as asystole | Flag |
| Motion artefact | Noise from movement. | Removed without recording the removal | Flag |
| Quality index | A score summarising signal usability. | Treated as a diagnosis | Score |
| Epoch | A fixed analysis interval, often 30 seconds in sleep. | Confused with a window in other tasks | Seconds |
| Resampling | Changing the sampling rate. | Done without anti-aliasing | Hz |
| Anti-aliasing filter | A low-pass filter before downsampling. | Omitted | Filter specification |
| Normalisation | Scaling the signal amplitude. | Fitted on all records | Method |
| Phase distortion | Change in waveform shape from a filter. | Ignored in morphology tasks | Filter property |

### 4.5 Resource terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| PTB-XL | Public 12-lead ECG dataset from PTB, Germany. | Treated as a wearable dataset. | Dataset |
| CODE-15% | Large Brazilian ECG dataset with weak labels. | Treated as adjudicated truth. | Dataset |
| SaMi-Trop | Chagas cohort with validated serology. | Treated as a general ECG dataset. | Cohort |
| Sleep-EDF | Public polysomnography dataset. | Treated as a clinical sleep cohort. | Dataset |
| TUH EEG | Temple University EEG corpus. | Treated as a uniform montage. | Corpus |
| MIMIC-IV waveform | ICU waveforms linked to MIMIC-IV. | Treated as ambulatory data. | Dataset |
| Icentia11K | Single-lead ambulatory ECG dataset. | Treated as a wrist wearable. | Dataset |
| WESAD | Wearable stress and affect dataset. | Treated as clinical stress. | Dataset |
| NSRR | Sleep and cardiometabolic cohort repository. | Treated as one protocol. | Repository |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the task and device, audit signal quality and labels, design windows and splits, and evaluate against the operational metric.

### Phase A — Define the task and device

**Step 1. Write the decision and the alarm action.**
Input: the clinical or consumer workflow.
Output: a decision statement with the actor, the action, the timing, the population, and the cost of false and missed alarms.
Rationale: an alarm that no one acts on is not a decision. The action determines the metric and the acceptable alarm burden.
Branch: if the action is a review of a ranked list, the metric is precision at capacity; if it is an immediate alarm, the metric includes time-to-alarm and false alarms per hour.
Edge case: a consumer notification with no clinical action is a different claim from a clinical alarm, even if the signal is the same.

**Step 2. Specify the deployment device.**
Input: the sensor, placement, sampling rate, filter, firmware, and form factor.
Output: a device specification with the parameters that affect the signal.
Rationale: the device defines the distribution. A model trained on a different device is a different model.
Branch: if the deployment device is not represented in any dataset, the candidate set fails device fit, and the plan needs a prospective collection.
Edge case: the same manufacturer can change the firmware or the sensor between hardware revisions without changing the product name.

**Step 3. Define the target population and the clinical context.**
Input: the intended users and the setting.
Output: inclusion criteria for age, comorbidity, activity, and setting.
Rationale: signal characteristics change with age, body composition, perfusion, and motion.
Branch: if the dataset is healthy volunteers and the target is older patients with comorbidity, population fit is weak.
Edge case: skin tone and perfusion affect optical signals; a dataset without diversity cannot support an optical claim across populations.

**Step 4. Map the label to the event and the action.**
Input: the decision statement and the available annotations.
Output: a label definition with onset, offset, and adjudication.
Rationale: a record-level statement is not an episode label, and an episode label is not an alarm label.
Branch: if the dataset has only record-level labels, the claim is record classification, not event detection.
Edge case: a paroxysmal event can occur outside the monitored window, so a negative record is not proof of absence.

### Collection profiles

| Collection | Custodian | Version inspected | Device and channels | Scale | Access and licence | What it cannot answer |
|---|---|---|---|---|---|---|
| PTB-XL | PTB and PhysioNet | v1.0.3, November 2022 | 12-lead clinical ECG, 500 Hz and 100 Hz, 10 s | 21,799 records; 18,869 patients | CC BY 4.0 | Wearable performance; episode timing |
| CODE-15% | Brazilian collaboration | Current release | 12-lead ECG, 400 Hz, 7.3 to 10.2 s | More than 300,000 ECGs | Public release; verify terms | Adjudicated truth; general population |
| SaMi-Trop | Brazilian Chagas cohort | Challenge release | 12-lead ECG | 1,631 validated-positive Chagas ECGs | Challenge data terms | Non-Chagas diversity |
| Challenge 2025 negatives | PhysioNet | 2025 release | 12-lead ECG | 21,799 PTB-XL records assumed negative | Per challenge terms | True absence of Chagas |
| Sleep-EDF | PhysioNet | Current release | PSG: EEG, EOG, EMG, temperature, respiration | 197 recordings | Open, ODC Attribution | Clinical sleep-disorder prevalence |
| TUH EEG | Temple University | Current release | Clinical EEG with varied montages | Tens of thousands of sessions | Registration and DUA | Uniform montage; ambulatory EEG |
| MIMIC-IV waveform | PhysioNet and MIT | Current release | ICU waveforms linked to MIMIC-IV | A subset of ICU stays | Credentialed | Ward or ambulatory signals |
| Icentia11K | Icentia and PhysioNet | Current release | Single-lead ambulatory ECG | About 11,000 patients | Open; verify terms | Wrist device performance |
| WESAD | University of Siegen | Current release | Chest and wrist signals for stress | 15 subjects | Open for research | Clinical stress or anxiety |
| NSRR | National Sleep Research Resource | Current release | Sleep and cardiometabolic cohorts | Multiple cohorts | Application and DUA | A single uniform protocol |
| Challenge 2026 | PhysioNet | 2026 release | Polysomnography, EDF | Five US institutions; about 214 GiB small and 1.2 TiB large | Challenge registration; verify licence | Causal link from sleep to later impairment |

The scale column mixes records, patients, sessions, and subjects. A record count is not a patient count, and a subject count is not an event count. The last column is the one that decides fitness.

### Signal quality indices

| Index | What it measures | Use | Limitation |
|---|---|---|---|
| Signal-to-noise ratio | Signal power relative to noise | Segment selection | Requires a signal estimate |
| Lead-off flag | Electrode contact | Exclusion or feature | Device-specific |
| Flatline detection | Absence of variation | Artefact detection | Can resemble asystole |
| Saturation | Clipping at the dynamic range | Quality flag | Device-specific |
| Baseline wander | Low-frequency drift | Filtering decision | Can overlap with slow waves |
| Motion index | Accelerometer variance | Activity context | Not all motion is artefact |
| Perfusion index | PPG pulsatile amplitude | Optical quality | Varies with temperature and site |
| Skin tone | Melanin concentration | Optical signal calibration | Often unrecorded |
| Impedance | Electrode-tissue contact | EEG and ECG quality | Changes over time |
| Artefact annotation | Human or algorithmic flag | Supervised quality model | Annotation cost |

### Annotation depth by task

| Task | Minimum annotation | What it supports | What it does not |
|---|---|---|---|
| Record classification | One label per record | Ranking and triage | Event timing |
| Episode detection | Onset and offset | Time-to-alarm and burden | Causal attribution |
| Sleep staging | 30-second epoch labels | Staging accuracy | Long-term outcomes |
| Seizure detection | Onset, offset, type | Detection latency | Clinical consequence |
| Arrhythmia classification | Beat or episode labels | Rhythm classification | Stroke risk |
| Stress detection | Self-report or task condition | Laboratory classification | Clinical diagnosis |
| Deterioration | Vital-sign trajectory and event | Early warning | Treatment effect |
| Apnea | Respiratory event scoring | Severity classification | Daytime function |

### Filter trade-offs

| Filter | Purpose | Risk | Alternative |
|---|---|---|---|
| High-pass at 0.5 Hz | Remove baseline wander | Distorts ST segment | 0.05 Hz or none for ST analysis |
| Low-pass at 40 Hz | Remove muscle noise | Removes high-frequency QRS detail | 100 Hz for morphology |
| Notch at 50 or 60 Hz | Remove line noise | Removes signal at that frequency | Adaptive filtering |
| Median filter | Remove spikes | Can remove sharp events | Morphological filtering |
| Band-pass 0.5 to 40 Hz | General ECG conditioning | Hides device differences | Report the exact filter |
| Wavelet denoising | Non-stationary noise | Can create artefacts | Validate on annotated events |
| Per-record normalisation | Amplitude comparability | Removes absolute amplitude information | Report both |
| Causal filter | Real-time use | Phase distortion | Match the deployment filter |

### Sampling rate adequacy

| Signal | Common rate | Bandwidth of interest | Adequate for | Not adequate for |
|---|---|---|---|---|
| ECG | 500 Hz | 0.05 to 150 Hz | Morphology and ST analysis | Very high-frequency notching |
| ECG | 100 Hz | 0.05 to 40 Hz | Rhythm and QRS timing | Fine morphology |
| EEG | 256 Hz | 0.5 to 70 Hz | Clinical EEG | High-frequency oscillations |
| PPG | 25 to 125 Hz | 0.5 to 10 Hz | Heart rate and rhythm | Fine waveform morphology |
| Respiration | 10 to 25 Hz | 0.1 to 1 Hz | Rate and effort | Detailed airflow shape |
| Accelerometer | 25 to 100 Hz | 0 to 20 Hz | Activity and motion | High-impact events |

Downsampling without an anti-aliasing filter folds high-frequency noise into the band of interest. If the deployment device samples lower than the training device, resample the training data with a proper filter and test the performance loss.

### Windowing policy decisions

| Decision | Options | Effect on the claim | What to record |
|---|---|---|---|
| Window length | 1 s, 10 s, 30 s, 60 s | Detection latency and label dilution | Seconds |
| Overlap | None, 25%, 50%, 90% | Effective sample size and leakage | Proportion |
| Alignment | Fixed grid or event-centred | Event-centred leaks label timing | Alignment rule |
| Padding | Zero, reflect, repeat | Boundary artefacts | Method |
| Sampling | Every window or stratified | Class balance | Policy |
| Label assignment | Any positive, majority, event overlap | Prevalence and ceiling | Rule |
| Quality filter | None, index threshold, manual | Selection bias | Threshold |
| Per-record normalisation | Yes or no | Amplitude information | Method |

An event-centred window uses the label to choose the segment, which is a form of leakage if the same rule is not available at deployment. Fixed-grid windows are the safe default for a real-time claim.

### Windowing arithmetic

| Record length | Window | Overlap | Windows | Effective independent samples |
|---:|---:|---:|---:|---:|
| 10 s | 10 s | 0% | 1 | 1 |
| 60 s | 10 s | 0% | 6 | 6, if the windows are independent |
| 60 s | 10 s | 50% | 11 | Far fewer than 11 |
| 600 s | 10 s | 0% | 60 | 60 at most, often fewer |
| 600 s | 10 s | 90% | 591 | Close to 1 per record |
| 8 h | 30 s | 0% | 960 | Bounded by annotated events |
| 8 h | 30 s | 50% | 1,919 | Bounded by annotated events |
| 24 h | 60 s | 0% | 1,440 | Bounded by annotated events |

The "effective independent samples" column is the point. Overlapping windows are not independent observations, and a split that treats them as independent overstates the sample size and leaks the record. A window count is not a sample size.

### Challenge metric examples

| Metric | What it encodes | Operational implication | Failure if ignored |
|---|---|---|---|
| Precision at top 5% | Fixed review capacity | Only 5% can be reviewed | Model optimises ranking without capacity |
| Sensitivity at fixed recall | Follow-up capacity | A fixed number of recalls | Alarm burden explodes |
| Age-conditioned AUROC | Removes an age shortcut | Performance within age strata | A trivial age model looks strong |
| Event-level F1 | Episode detection | Alarms tied to events | Window accuracy hides misses |
| Time-to-alarm penalty | Latency matters | Earlier detection is better | A late alarm is counted as correct |
| False alarms per hour | Alarm burden | Staff tolerance | A sensitive model is unusable |
| Weighted kappa | Ordinal agreement | Grade distance matters | Adjacent grades treated as equal |

Reading the metric before the leaderboard is not a formality. The metric states the operational constraint, and a model that wins on a different constraint is not the winner for the deployment.

### Quality exclusion decision

| Observation | Decision | Risk |
|---|---|---|
| Lead-off for most of the record | Exclude the record | Informative missingness |
| Short noise burst | Keep and flag | Noise may correlate with outcome |
| Flatline for a segment | Flag, do not treat as asystole | False event |
| Saturation during a event | Keep and flag | Event may be unreadable |
| Unknown skin tone in an optical dataset | Report the gap | Equity risk |
| Montage differs from deployment | Restrict or harmonise | Channel mismatch |

### Quality standards for a signal dataset

| Standard | What it covers | Why it matters | Limit |
|---|---|---|---|
| WFDB annotation format | Event annotations | Portable event labels | Does not define the event |
| EDF+ annotation channel | Sleep and EEG events | Clinical interchange | Scoring version varies |
| SCP-ECG | ECG statements | Structured interpretation | Not an episode label |
| AASM manual | Sleep scoring | Comparable staging | Version changes |
| IEC 60601 | Device safety | Electrical and EMC safety | Not model validity |
| ISO 14971 | Risk management | Device risk file | Not performance |
| HL7 FHIR Observation | Exchange of observations | Interoperability | Not waveform storage |
| DICOM waveform | Clinical waveform | Integrated with imaging | Complex and less used |

The standard fixes the container and leaves the meaning to the protocol. A WFDB file with an annotation channel is portable; the annotation rule that produced it is the evidence.

### Phase B — Audit signal quality and labels

**Step 5. Inventory records, channels, and durations.**
Input: the dataset manifest and headers.
Output: counts by record, patient, channel, duration, and device.
Rationale: the inventory exposes channel mismatch, short records, and duplicate patients.
Branch: if the channel set differs across records, the model needs a channel-agnostic design or a restricted subset.
Edge case: a record can contain multiple leads but only one is usable; the manifest may not say.

**Step 6. Audit sampling rate and bandwidth.**
Input: the header and the signal.
Output: the effective sampling rate, the anti-aliasing filter, and the bandwidth.
Rationale: downsampling without an anti-aliasing filter aliases high-frequency noise into the signal.
Branch: if the deployment device samples lower than the training device, resample and test the loss.
Edge case: a 100 Hz ECG preserves QRS timing and loses high-frequency morphology; the task determines whether that matters.

**Step 7. Audit filters and preprocessing.**
Input: the acquisition settings and any published preprocessing.
Output: a filter specification with type, cut-off, and order.
Rationale: a high-pass filter can distort the ST segment; a notch filter can remove signal at the notch frequency.
Branch: if the dataset was filtered differently from the deployment device, replicate the deployment filter and test.
Edge case: a non-causal filter used offline cannot be used in a real-time device; use a causal equivalent.

**Step 8. Audit signal quality and missing segments.**
Input: quality indices, lead-off flags, and motion annotations.
Output: the proportion of usable signal by record and channel.
Rationale: noisy segments are present and uninterpretable; excluding them silently biases the evaluation.
Branch: if quality is associated with the outcome, exclusion is informative.
Edge case: a lead-off period can look like a flat line and be mistaken for asystole.

**Step 9. Audit label provenance and annotation depth.**
Input: the annotation protocol, the readers, and the adjudication rule.
Output: a provenance statement, agreement, and the fraction of events likely annotated.
Rationale: the label is a measurement, and the annotation depth bounds the event-level metric.
Branch: if the labels are statement-derived, the claim is about statements; if they are adjudicated episodes, the claim can be about events.
Edge case: a dataset can label the record but not the event time, which prevents time-to-alarm evaluation.

### Phase C — Design windows and splits

**Step 10. Choose the window length and overlap.**
Input: the event duration and the alarm latency requirement.
Output: a window length, an overlap, and a sampling policy.
Rationale: the window must contain enough signal to detect the event and be short enough to meet the latency.
Branch: if the event is shorter than the window, the label is diluted; if the window is longer than the latency, the alarm is late.
Edge case: overlapping windows create near-duplicates and inflate the effective sample size.

**Step 11. Split by record or patient.**
Input: the record and patient identifiers.
Output: a split with group keys and a hash.
Rationale: windows from the same record share rhythm, noise, and patient identity.
Branch: if patient identifiers are absent, split by record and state the limitation.
Edge case: a patient can have multiple records across devices or sites; keep them on one side where possible.

**Step 12. Fit preprocessing on training records only.**
Input: the training split and the preprocessing plan.
Output: stored parameters applied identically to all splits.
Rationale: normalisation statistics and filter design choices can leak test information.
Branch: if the normalisation is per-record, fit within each record and document it.
Edge case: a global scaler fitted on all records is a common and quiet leak.

**Step 13. Check class balance at the event and window level.**
Input: the labels and the windows.
Output: prevalence at both levels and the implied alarm burden.
Rationale: a window-level prevalence can be very different from an event-level prevalence.
Branch: if the window prevalence is low, accuracy is misleading; use sensitivity at capacity.
Edge case: a record with one short event can produce many positive windows if the window is long.

### Phase D — Evaluate against the operational metric

**Step 14. Compute the metric at the deployment prevalence.**
Input: sensitivity, specificity, and target prevalence.
Output: precision, false alarm rate, and the expected workload.
Rationale: a model evaluated at 20 percent prevalence can be unusable at 1 percent.
Branch: if the target prevalence is much lower, recalibrate and recompute.
Edge case: prevalence varies by site and season; report a range.

**Step 15. Evaluate at the event level and the alarm level.**
Input: the predictions and the event annotations.
Output: event sensitivity, time-to-alarm, and false alarms per hour.
Rationale: window accuracy does not equal event detection.
Branch: if the event annotations are absent, the event-level claim cannot be made.
Edge case: a single event can trigger many alarms; count alarms, not windows.

**Step 16. Evaluate by device, site, and subgroup.**
Input: the metadata and the test set.
Output: metrics with confidence intervals by device, site, age, sex, and skin tone where available.
Rationale: device and population shift are the main transport risks.
Branch: if a subgroup has too few events, report descriptively.
Edge case: a subgroup with high performance can reflect access or detection bias.

**Step 17. Score the gates and record the evidence.**
Input: the outputs of steps 1 to 16.
Output: a gate table with evidence and residual uncertainty.
Rationale: the decision must be reviewable.
Branch: a zero on any gate is a veto for the claim as written.
Edge case: a dataset can be admissible for a methods claim and inadmissible for a product claim.

**Step 18. Re-run on device, firmware, or label change.**
Input: the release notes, the device revisions, and the annotation protocol.
Output: an updated gate record with a new date.
Rationale: devices and datasets change, and a model can degrade without a code change.
Branch: re-run when the device, firmware, sampling rate, filter, or label protocol changes.
Edge case: a manufacturer can change a component without changing the product name.

### Device-fit decision table

| Deployment device | Dataset device | Fit | Action |
|---|---|---|---|
| 12-lead clinical ECG | 12-lead clinical ECG | Strong | Validate on the target population |
| 12-lead clinical ECG | Single-lead ambulatory | Weak | Add leads or change the claim |
| Single-lead patch | Single-lead patch, different manufacturer | Moderate | Recalibrate and test |
| Wrist PPG | Clinical PPG | Weak | Collect wrist data |
| Wrist PPG | Wrist PPG, different sensor | Moderate | Test skin tone and motion |
| ICU monitor | ICU monitor, same unit | Strong | Validate on the site |
| ICU monitor | Ambulatory patch | Weak | Do not transfer |
| EEG clinical montage | Different montage | Weak | Harmonise channels or restrict |

### Split decision table

| Split | What it estimates | Requirement | Limitation |
|---|---|---|---|
| Random window | Memorisation resistance | None | Leaks record and patient |
| Record-level | Generalisation to new records | Record ID | Same patient may recur |
| Patient-level | Generalisation to new patients | Patient ID | Same device and site |
| Device-level | Generalisation to a new device | Device ID | Few devices means high variance |
| Site-level | Generalisation to a new site | Site ID | Confounded with population |
| Temporal | Generalisation to a later period | Dates | Firmware change confounded with time |

### Metric decision table

| Decision | Primary metric | Secondary metric | Why |
|---|---|---|---|
| Fixed review capacity | Precision at k | Sensitivity at k | Capacity is the constraint |
| Immediate alarm | Sensitivity at fixed false-alarm rate | Time-to-alarm | Alarm burden is the constraint |
| Screening | Sensitivity at fixed recall | Number needed to review | Follow-up capacity |
| Episode detection | Event sensitivity | False alarms per hour | Events, not windows |
| Ranking | AUROC | Calibration | Discrimination across thresholds |
| Consumer notification | Positive predictive value | Alert rate | User burden |

### Edge cases that change the procedure

**Licence forbids commercial use.** PTB-XL is released under CC BY 4.0, which permits commercial use with attribution. Other datasets have research-only terms. Check each dataset's licence rather than assuming that "open" means commercial.

**Dataset withdrawn.** Access can be withdrawn for governance or consent reasons. Stop new work, confirm the deletion requirement, and re-run the source selection.

**Credentialing takes months.** Open signal datasets are usually fast; credentialed ICU waveforms and clinical corpora can take weeks. A timeline shorter than the access path fails Gate 3.

**Version drift.** PTB-XL v1.0.3 has 21,799 records and 18,869 patients, while v1.0.0 had 21,837 and 18,885. A pipeline built on the older counts can fail a row-count check. Pin the version and DOI.

**Label errors discovered.** If the statement labels are wrong or incomplete, the event-level ceiling falls. Re-adjudicate a sample and report the error.

**Benchmark contamination.** A public dataset can appear in a pretrained model's training data. Check the model's training data and dates, and use a private holdout where possible.

## 6. Worked example

### The case

A consumer health company wants to add an atrial fibrillation (AF) detection feature to a wrist wearable. The user records a 30-second single-lead ECG when the device prompts them, and the app advises a clinical review if the rhythm is irregular. The company has 12 months to a regulated launch in the United States.

**Decision statement.** "From a 30-second wrist single-lead ECG, flag probable atrial fibrillation so that the user is advised to seek clinical confirmation within a week; a false alert causes anxiety and an unnecessary visit, and a missed AF delays anticoagulation in a patient at stroke risk."

**Device.** A wrist device with a single dry electrode, 25 to 250 Hz sampling depending on mode, a motion sensor, and firmware that applies a band-pass filter. The exact filter and firmware version are proprietary but documented for the company.

**Claim.** The consumer claim is a notification, not a diagnosis. The clinical claim is that the notification identifies people who should be evaluated. The two claims need different evidence.

### Candidate long list

| Candidate | Custodian | Version inspected | Device and channels | Scale | Access and licence | Why it is on the list |
|---|---|---|---|---|---|---|
| Icentia11K | Icentia and PhysioNet | Current release | Single-lead ambulatory ECG | About 11,000 patients | Open; verify terms | Closest device class and rhythm labels |
| PTB-XL | PTB and PhysioNet | v1.0.3 | 12-lead clinical ECG, 10 s | 21,799 records; 18,869 patients | CC BY 4.0 | Strong labels and open commercial terms |
| CODE-15% | Brazilian collaboration | Current release | 12-lead ECG, 400 Hz | More than 300,000 ECGs | Public release; verify terms | Scale and rhythm diversity |
| MIMIC-IV waveform | PhysioNet and MIT | Current release | ICU waveforms | Subset of ICU stays | Credentialed | High-frequency ICU data |
| WESAD | University of Siegen | Current release | Chest and wrist signals | 15 subjects | Open for research | Wrist sensor data |
| Partner wearable | Device manufacturer | Not collected | Exact wrist device | Site-dependent | Contract | Exact device and population |

### Gate scores for the consumer claim

| Gate | Icentia11K | PTB-XL | CODE-15% | MIMIC-IV waveform | WESAD | Partner |
|---|---:|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 2 | 2 | 1 | 0 | 2 |
| 2 Population fit | 1 | 1 | 1 | 0 | 0 | 2 |
| 3 Access fit | 2 | 2 | 2 | 2 | 2 | 0 |
| 4 Licence fit | 1 | 2 | 1 | 0 | 1 | 2 |
| 5 Benchmark fit | 1 | 1 | 1 | 1 | 0 | 2 |
| Veto | none | none | none | population, licence | decision, population | access |

**Icentia11K, Gate 1 and Gate 2.** The dataset is a single-lead ambulatory ECG with rhythm labels, which is closer to the wrist device than a 12-lead clinical ECG. It is still a patch, not a wrist device, and the population is a clinical cohort rather than consumers. Scores 2 and 1.

**Icentia11K, Gate 4.** The release terms must be verified. If the dataset is open for research only, commercial training is not permitted. Score 1 until confirmed.

**PTB-XL, Gate 4.** Released under CC BY 4.0, which permits commercial use with attribution. The device is a 12-lead clinical ECG, so the data are useful for pretraining and morphology, not for the wrist claim. Score 2 on licence and 1 on device fit.

**CODE-15%, Gate 4.** The public release terms must be verified. The labels are weak and self-reported, and the device is a 12-lead ECG. Score 1.

**MIMIC-IV waveform, Gate 2 and Gate 4.** ICU patients with continuous monitoring are a different population and setting, and the PhysioNet DUA is research-only. Veto on both.

**WESAD, Gate 1.** The dataset is a laboratory stress protocol with 15 subjects and no AF labels. Veto on decision fit.

**Partner, Gate 3.** The device manufacturer has the exact device and population, but no agreement, no IRB approval, and no data-sharing contract exists. Twelve months is not credible. Veto for the timeline.

### The two diagnostics

**Diagnostic 1: prevalence shift.** The AF prevalence in a clinical cohort used for development is much higher than in the consumer population that will use the feature. At a sensitivity of 0.95 and a specificity of 0.97, the positive predictive value is about 78 percent at 10 percent prevalence and about 14 percent at 0.5 percent prevalence. The same model, with the same sensitivity and specificity, produces a very different user experience and a very different clinical workload. The dataset cannot answer the consumer claim unless the evaluation is reweighted to the target prevalence.

**Diagnostic 2: device and window mismatch.** Icentia11K records are multi-day ambulatory patch recordings, and the label is a rhythm annotation over that period. The product records 30 seconds from a wrist. A paroxysmal AF episode that is captured in a 14-day patch may not appear in a 30-second wrist recording, and the wrist signal has different motion and contact noise. The dataset can support a rhythm-classification benchmark and cannot support the 30-second wrist claim without a device-specific collection.

A third diagnostic is label quality: the AF labels in an ambulatory dataset may be derived from an algorithm or a single reader, and the annotation depth for short episodes is unknown. The team must estimate the event-level sensitivity of the reference before using it as truth.

### The selected route

The admissible set is Icentia11K and PTB-XL. The plan uses PTB-XL for pretraining and morphology, Icentia11K for rhythm classification and a device-shift test, and the partner wearable for the final evaluation. The consumer claim is not made until the partner data exist.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Licence verification | Confirm Icentia and CODE-15% terms | Legal | 3 weeks | Staff time |
| Pretraining | PTB-XL with attribution | ML team | 4 weeks | Compute, about $3,000 |
| Rhythm model | Icentia11K training and record-level test | ML team | 10 weeks | Compute, about $8,000 |
| Device-shift test | Evaluate on a wrist-like subset or partner data | ML team | 6 weeks | Staff time |
| Regulatory pathway | Determine device classification and evidence needs | Regulatory | 8 weeks | Consultant time |
| Partner validation | Contract, IRB, prospective collection | Clinical lead | 6 to 12 months | Six figures |

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Licence review | Terms for each dataset | 3 weeks | Legal, 30 hours |
| Pretraining | PTB-XL | 4 weeks | About $3,000 |
| Model development | Icentia11K and baselines | 10 weeks | About $8,000 |
| Prevalence reweighting | Evaluation at target prevalence | 2 weeks | Staff time |
| Regulatory review | Classification and evidence plan | 8 weeks | Consultant |
| Partner validation | Prospective wrist data | 6 to 12 months | Six figures |

The prevalence reweighting is cheap and decisive. It converts a benchmark number into an estimate of the user experience, and it often changes the threshold and the claim.

### Substitute plan if the partner route fails

If the wearable manufacturer cannot provide data, the plan changes in three ways:

1. The claim narrows to a clinical 12-lead or patch-based rhythm classification, which is not the consumer feature.
2. The model is developed and published as a research contribution on open data, with no regulated launch.
3. The company collects its own wrist data through a research study, which adds 9 to 12 months and an IRB.

The fallback preserves the science and loses the product timeline. The decision record should say that the open datasets cannot substitute for the device-specific evidence.

### Vary one assumption: Icentia11K is research-only

Suppose the licence review finds that Icentia11K permits research use and prohibits commercial training. Gate 4 falls to 0 for the product claim. The plan changes:

- PTB-XL remains usable for commercial pretraining under CC BY 4.0.
- Icentia11K is used for a methods paper and for architecture selection, not for the product model.
- The product model must be trained on partner data or another commercial-permissive single-lead dataset.
- The timeline extends by the partner data collection, and the budget shifts from open-data compute to a prospective study.

If instead the variation is size, a smaller single-lead dataset would weaken the rhythm benchmark but not change the device conclusion. If the variation is label quality, a lower event-level sensitivity in the reference would lower the ceiling and force a larger evaluation set.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | Icentia11K | Single-lead records and rhythm labels | 2 | Event timing and annotation depth |
| 1 | PTB-XL | 12-lead records and SCP statements | 2 | Statement is not an episode label |
| 2 | Icentia11K | Clinical ambulatory cohort | 1 | Not consumers; device differs |
| 2 | PTB-XL | German clinical cohort | 1 | Not a wrist device |
| 3 | Icentia11K | Open release route | 2 | Terms verification |
| 3 | PTB-XL | Open download | 2 | None |
| 4 | Icentia11K | Release terms | 1 | Commercial training unresolved |
| 4 | PTB-XL | CC BY 4.0 | 2 | Attribution required |
| 5 | Icentia11K | No frozen wrist split | 1 | Split is ours |
| 5 | PTB-XL | `strat_fold` available | 1 | Folds 9 and 10 are the recommended test |
| 5 | Partner | Prospective wrist collection possible | 2 | Contract not started |

### Decision timeline

| Month | Licence track | Data track | Clinical track | Deliverable |
|---:|---|---|---|---|
| 1 | Verify dataset terms | Download PTB-XL and Icentia | Define target prevalence | Licence memo |
| 2 | Confirm attribution | Pretrain on PTB-XL | Protocol for partner study | Pretrained model |
| 3-4 | Partner contract draft | Train rhythm model | Site identification | Internal model v1 |
| 5 | Partner signature | Record-level test | IRB submission | Test report |
| 6-7 | Regulatory classification | Device-shift test | Data collection | Regulatory plan |
| 8-10 | Launch review | Prevalence reweighting | Prospective evaluation | Evidence package |
| 11-12 | Final approval | Monitoring setup | Launch readiness | Launch decision |

### What the case demonstrates

- The closest device class was not the deployment device. A single-lead patch is closer than a 12-lead clinical ECG and still not a wrist wearable.
- The prevalence shift changed the user experience by a factor that no model improvement could offset. Sensitivity and specificity were held constant and the PPV moved from 78 percent to 14 percent.
- The window-level split leaked half the test windows through shared records, which is the signal equivalent of an image-level split.
- The open licence on PTB-XL made it usable for commercial pretraining, while the other datasets needed verification. "Open" was not a single legal category.
- The partner route was the only path to the device-specific claim, and it was also the longest.
- The regulatory claim and the consumer claim needed different evidence, even though the same model produced both.

### Conditions under which the case would fail

- **The partner device data do not arrive.** The consumer claim cannot be made, and the deliverable is a research model on open data.
- **The prevalence in the target population is below 0.5 percent.** The PPV falls below 15 percent, and the product must change the notification or the target population.
- **The Icentia terms prohibit commercial training.** The product model must be trained on partner data, and the timeline extends.
- **The annotation depth is low.** Event-level sensitivity is overstated, and the evaluation needs a larger adjudicated sample.
- **The firmware filter changes.** The device-shift test must be repeated, and the model may need re-training.
- **The regulatory pathway requires prospective evidence.** The partner study becomes mandatory, not optional.

### Sensitivity of the decision

| Change | Gate affected | Does the source change? | Plan response |
|---|---|---|---|
| Target prevalence rises to 5% | 5 | No | PPV improves; the threshold can rise |
| Icentia terms permit commercial use | 4 | No | Use for product training |
| Partner data available in month three | 3 and 5 | Yes, becomes primary | Shift training and evaluation |
| Device firmware changes the filter | 1 and 5 | No | Re-run the device-shift test |
| A new single-lead wrist dataset is released | 1 and 2 | Possibly | Audit the device and licence |
| Regulatory pathway requires prospective data | 5 | No | The partner route becomes mandatory |

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement a window leakage audit, a prevalence reweighting calculation, and a sampling-rate check.

### 7.1 Window leakage audit

```python
WINDOWS = [
    ("r01", "p01", "train"), ("r01", "p01", "train"), ("r01", "p01", "test"),
    ("r02", "p02", "train"), ("r02", "p02", "test"),
    ("r03", "p03", "test"), ("r03", "p03", "test"),
    ("r04", "p04", "train"), ("r05", "p05", "test"),
    ("r06", "p06", "train"), ("r06", "p06", "test"),
]
train_records = {r for r, p, s in WINDOWS if s == "train"}
test_records = {r for r, p, s in WINDOWS if s == "test"}
train_patients = {p for r, p, s in WINDOWS if s == "train"}
test_patients = {p for r, p, s in WINDOWS if s == "test"}

print(f"Windows: {len(WINDOWS)}")
print(f"Train windows: {sum(1 for _, _, s in WINDOWS if s == 'train')}")
print(f"Test windows:  {sum(1 for _, _, s in WINDOWS if s == 'test')}")
print(f"Record overlap:  {sorted(train_records & test_records)}")
print(f"Patient overlap: {sorted(train_patients & test_patients)}")
print()
print(f"Share of test windows from a leaked record: "
      f"{sum(1 for r, p, s in WINDOWS if s == 'test' and r in train_records) / sum(1 for _, _, s in WINDOWS if s == 'test'):.0%}")
```

Expected output:

```text
Windows: 11
Train windows: 5
Test windows:  6
Record overlap:  ['r01', 'r02', 'r06']
Patient overlap: ['p01', 'p02', 'p06']

Share of test windows from a leaked record: 50%
```

The audit is small on purpose. The same logic scales to millions of windows: build the record and patient sets, intersect them, and report the share of test windows that come from a leaked group.

### 7.2 Prevalence reweighting

```python
def ppv(sensitivity, specificity, prevalence):
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return tp / (tp + fp)

print(f"{'Prevalence':>12}{'PPV':>10}{'False alarms per 1,000':>24}")
for prev in (0.001, 0.005, 0.01, 0.05, 0.10, 0.20):
    value = ppv(0.95, 0.97, prev)
    fp_per_1000 = round((1 - 0.97) * (1 - prev) * 1000)
    print(f"{prev:>12.1%}{value:>10.1%}{fp_per_1000:>24}")
print()
print("Assumptions: sensitivity 0.95, specificity 0.97.")
print("At 0.5% prevalence, about 14% of alerts are true.")
```

Expected output:

```text
  Prevalence       PPV  False alarms per 1,000
        0.1%      3.1%                      30
        0.5%     13.7%                      30
        1.0%     24.2%                      30
        5.0%     62.5%                      29
       10.0%     77.9%                      27
       20.0%     88.8%                      24

Assumptions: sensitivity 0.95, specificity 0.97.
At 0.5% prevalence, about 14% of alerts are true.
```

The false-alarm count per 1,000 is roughly constant because specificity dominates. The positive predictive value changes by a factor of 25 across the prevalence range. A consumer launch at 0.5 percent prevalence should expect most alerts to be false, and the product and the claim must be designed for that.

### 7.3 Sampling rate and Nyquist

```python
print(f"{'Sampling rate':>14}{'Nyquist':>10}  Note")
for rate in (25, 100, 250, 500, 1000):
    print(f"{rate:>11} Hz{rate/2:>7.0f} Hz  filter content above {rate/2:.0f} Hz")
print()
print("A 100 Hz ECG preserves 0 to 50 Hz. QRS timing survives; fine morphology")
print("does not. Without an anti-aliasing filter, higher content folds into the band.")
```

Expected output:

```text
 Sampling rate   Nyquist  Note
         25 Hz     12 Hz  filter content above 12 Hz
        100 Hz     50 Hz  filter content above 50 Hz
        250 Hz    125 Hz  filter content above 125 Hz
        500 Hz    250 Hz  filter content above 250 Hz
       1000 Hz    500 Hz  filter content above 500 Hz

A 100 Hz ECG preserves 0 to 50 Hz. QRS timing survives; fine morphology
does not. Without an anti-aliasing filter, higher content folds into the band.
```

The check is a first-pass device audit. It does not replace a spectral analysis of the actual recordings, and it catches the common error of downsampling without a filter.

### 7.4 Event-level versus window-level sensitivity

```python
events = [
    ("e1", 10, 10),
    ("e2", 8, 1),
    ("e3", 12, 11),
    ("e4", 6, 0),
    ("e5", 15, 14),
]
total_windows = sum(w for _, w, d in events)
detected_windows = sum(d for _, w, d in events)
event_detected = sum(1 for _, w, d in events if d > 0)

print(f"{'Event':<6}{'Windows':>9}{'Detected':>10}{'Window sensitivity':>20}")
for name, w, d in events:
    print(f"{name:<6}{w:>9}{d:>10}{d/w:>20.0%}")
print()
print(f"Window-level sensitivity: {detected_windows/total_windows:.1%}")
print(f"Event-level sensitivity:  {event_detected/len(events):.1%}")
print()
print("Event e2 was detected in only one of eight windows and still counts as")
print("a detected event. Report both levels, the missed events, and the latency.")
```

Expected output:

```text
Event   Windows  Detected  Window sensitivity
e1           10        10                100%
e2            8         1                 12%
e3           12        11                 92%
e4            6         0                  0%
e5           15        14                 93%

Window-level sensitivity: 70.6%
Event-level sensitivity:  80.0%

Event e2 was detected in only one of eight windows and still counts as
a detected event. Report both levels, the missed events, and the latency.
```

The two levels answer different questions. Window sensitivity describes how much of the signal is classified correctly; event sensitivity describes whether the event was found at all. A single detected window can make an event count as detected while the alarm quality is poor. Report both, plus the number of missed events and the time to the first alarm.

## 8. Data, standards, and regulatory context

Signal data are governed by device standards, file formats, scoring manuals, and medical-device regulation. The table separates the binding from the conventional.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| WFDB | De facto standard, open | Waveform and annotation file format | Guarantee signal quality or label truth |
| EDF and EDF+ | De facto standard in sleep and EEG | Recording format and annotations | Enforce a scoring manual |
| SCP-ECG | Standard for ECG statements | Statement codes and structure | Guarantee adjudicated labels |
| AASM scoring manual | Professional standard, often contractual | Sleep stage and event scoring | Cover all sleep-related claims |
| IEC 60601 | Binding for medical electrical equipment | Device safety and electromagnetic compatibility | Validate a model |
| IEC 62304 | Voluntary, often contractual | Medical device software lifecycle | Validate clinical performance |
| ISO 14971 | Voluntary, often contractual | Risk management | Prescribe model metrics |
| FDA device regulation | Binding when applicable | Clearance or approval and labeling | Certify fairness |
| EU MDR and IVDR | Binding in the EU | Device classification and clinical evidence | Address dataset licensing |
| EU AI Act | Binding, phased | Risk management and documentation | Replace device regulation |
| HIPAA | Binding in the US | PHI handling and de-identification | Grant commercial rights |
| GDPR | Binding in the EU | Personal data processing and transfer | Define signal quality |

**The format is not the standard.** WFDB and EDF store the signal and the annotations. They do not define what counts as an event, how the signal was filtered, or who adjudicated the label. A dataset in a standard format can still be unusable for a claim.

**The scoring manual matters.** Sleep staging and apnea scoring follow the AASM manual, and the manual changes across versions. A dataset scored under one version and a model evaluated under another are not comparable without a mapping. The version should be recorded in the dataset card.

**The device standard is not the model standard.** IEC 60601 covers electrical safety; IEC 62304 covers software lifecycle; neither validates a model's clinical performance. A device can be safe and a model useless. The regulatory file needs both device evidence and clinical evidence.

**Consumer and clinical claims differ.** A consumer feature that encourages a user to seek care is not a diagnostic device in the same sense as a clinical alarm, and the regulatory pathway and evidence standard differ. The claim must be written before the dataset is chosen, because the claim determines the evidence.

### Signal format comparison

| Format | Domain | Strength | Limit |
|---|---|---|---|
| WFDB | ECG, EEG, waveforms | Mature tooling and annotations | Less common outside research |
| EDF and EDF+ | Sleep and EEG | Widely used in clinical sleep | Annotation conventions vary |
| HDF5 | Machine learning | Efficient arrays and metadata | Not a clinical interchange format |
| CSV | Any | Simple | Large files and no metadata standard |
| Parquet | Analytics | Columnar and compressed | Not a waveform standard |
| MAT | Research | MATLAB compatibility | Version and schema drift |
| DICOM waveform | Clinical | Integrated with imaging | Complex and less used in research |

### Device standards and what they cover

| Standard | Covers | Does not cover |
|---|---|---|
| IEC 60601-1 | General electrical safety | Algorithm performance |
| IEC 60601-2-25 | Electrocardiograph safety | Diagnostic accuracy |
| IEC 60601-2-26 | Electroencephalograph safety | Seizure detection accuracy |
| IEC 62304 | Software lifecycle | Model validation |
| ISO 14971 | Risk management | Statistical performance |
| ISO 13485 | Quality management system | Clinical evidence |
| IEC 82304-1 | Health software safety | Dataset quality |
| FDA guidance | Submission content | A guarantee of clearance |

The device standards establish that the product is safe and the process is controlled. They do not establish that the model detects the event. The clinical evidence is separate.

### Regulatory artefacts for a signal product

- **Intended use** names the signal, the device, the population, and the action.
- **Reference standard** describes the adjudication and the readers.
- **Algorithm description** includes filtering, windowing, and thresholds.
- **Validation** includes the device, the prevalence, the event-level metric, and the subgroups.
- **Human factors** describes the notification and the user response.
- **Cybersecurity** covers the device, the app, and the data transfer.
- **Monitoring** covers firmware changes and drift.

A signal product that reports window accuracy and no event-level or device-specific evidence has an incomplete file.

## 9. Industry practice and case evidence

### What organisations actually do

- **Treat the device as part of the dataset.** Device, firmware, and filter are recorded with every model.
- **Split by record and patient.** Window splits are treated as a defect.
- **Report at the event level.** Alarm burden and time-to-alarm are reported with discrimination.
- **Reweight to the deployment prevalence.** A model evaluated at cohort prevalence is reweighted before a launch decision.
- **Run a device-shift test.** A holdout device or a prospective wrist collection is part of the evidence.
- **Version the annotation manual.** Scoring rules change and the version is recorded.
- **Monitor alarm burden after launch.** A model that is accurate and generates too many alarms is not deployable.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| PTB-XL | Version 1.0.3 contains 21,799 records from 18,869 patients, is released under CC BY 4.0, and provides a recommended `strat_fold` split with folds 9 and 10 for validation and test. | A well-documented open dataset with a licence that permits commercial use is rare and valuable. The split is a recommendation, not a guarantee of a clean external test. |
| PhysioNet Challenge 2025 | The challenge used CODE-15% (more than 300,000 ECGs with weak self-reported labels), SaMi-Trop (1,631 validated positives), and PTB-XL (21,799 assumed negatives), and scored the fraction of Chagas positives in the top 5 percent of a ranked cohort. | The metric encodes a fixed serology capacity, and the negatives are assumed rather than confirmed. A high score can reflect the metric and the label assumptions. |
| Sleep-EDF | The dataset is a small public polysomnography collection with whole-night recordings and sleep stage annotations. | It supports sleep staging methods and not clinical prevalence or device performance. |
| Icentia11K | The dataset is a large single-lead ambulatory ECG collection with rhythm labels. | It is closer to a wearable than a 12-lead dataset and still not a wrist device. |
| WESAD | The dataset contains chest and wrist signals from 15 subjects in a laboratory stress protocol. | It supports laboratory stress classification and not clinical anxiety or ambulatory stress. |

The first two rows are stable and verified. The challenge details are edition-specific and should be checked against the current challenge page before reuse.

### Device classes and their evidence

| Device class | Typical signal | Claim it supports | Evidence needed | Common failure |
|---|---|---|---|---|
| Clinical 12-lead ECG | 10 s, resting | Diagnostic classification | Reader-adjudicated test set | Wearable transfer |
| Ambulatory patch | Days, single or dual lead | Rhythm monitoring | Episode annotations | Device mismatch |
| Wrist ECG | 30 s, single lead | Consumer notification | Prospective wrist data | Prevalence shift |
| Wrist PPG | Continuous optical | Heart rate and rhythm hint | Optical validation across skin tones | Motion and perfusion |
| ICU monitor | Continuous multi-channel | Deterioration and alarms | Site-specific validation | Ward transfer |
| Holter | 24 to 48 h | Arrhythmia burden | Episode annotations | Annotation depth |
| Sleep PSG | Overnight multi-channel | Sleep staging and apnea | Scored epochs | Manual version |
| EEG clinical | Minutes to hours | Seizure and encephalopathy | Montage-specific labels | Montage heterogeneity |

The device class is a stronger determinant of transfer than the modality label. "ECG" spans a resting 12-lead recording and a 30-second wrist trace, and the two are not interchangeable.

### What good looks like

- Device, firmware, sampling rate, and filter are recorded with every model.
- Splits are by patient and record, never by window.
- The label is an episode with onset and offset where the claim is event detection.
- The evaluation is reweighted to the deployment prevalence.
- Event-level sensitivity, false alarms per hour, and time-to-alarm are reported.
- A device-shift test is part of the evidence.
- Subgroups include device, age, sex, and skin tone where relevant.
- The data card states what the dataset cannot support.

### Documented fact versus inference

- **Documented fact:** a dataset paper, a licence, a challenge page, or a device specification.
- **Inference:** a planning consequence for a different device or population.
- **Unverified summary:** a secondary account of a count or a metric.

A case table should carry a source and a status. The distinction is especially important for device claims, where a firmware change can invalidate an inference without changing the dataset.

### Procurement questions for a signal product

- Which device and firmware were used for training and validation?
- What is the sensitivity and specificity at the operating point, and at what prevalence?
- What is the event-level sensitivity and the false alarms per hour?
- What is the time-to-alarm distribution?
- How does performance change on a different device or firmware?
- What are the subgroup results by age, sex, and skin tone?
- What happens when the signal quality is poor?
- What is the monitoring plan for firmware and population drift?

A vendor that cannot answer these has a benchmark, not a product.

### Why challenge metrics are operational statements

A challenge that scores precision at the top 5 percent is telling competitors that the deployment can review only 5 percent of the cohort. A challenge that conditions on age is telling them that age is a shortcut the organisers want removed. A challenge that scores event-level F1 is telling them that misses and false alarms both matter. The metric is a compressed description of the operational constraint, and reading it is part of reading the benchmark.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Window leakage | Test performance is much higher than clinical intuition | Windows from one record on both sides | Memorisation reported as generalization | Split by record and patient |
| Device mismatch | The dataset is 12-lead, the product is single-lead | Different sensor and placement | Model fails at deployment | Device-specific data or a narrower claim |
| Prevalence neglect | PPV is not reported | Cohort prevalence differs from target | Most alerts are false | Reweight and recompute |
| Statement label treated as event | The label is an SCP statement | Statement summarises an interpretation | No event timing | Use episode annotations or change the claim |
| Annotation depth unknown | No estimate of missed events | Only some events annotated | Event sensitivity is overstated | Audit a sample |
| Filter mismatch | Training and deployment filters differ | Filter shapes the signal | Morphology shift | Match the deployment filter |
| Non-causal filtering | A zero-phase filter is used offline | The filter uses future samples | Real-time performance differs | Use a causal filter |
| Downsampling without anti-aliasing | High-frequency noise appears in the band | Aliasing | Spurious features | Filter before resampling |
| Quality exclusion bias | Noisy records are dropped silently | Quality correlates with outcome | Biased evaluation | Report exclusion and test sensitivity |
| Alarm burden ignored | Only sensitivity is reported | False alarms per hour unknown | Unusable product | Report alarms per hour |
| Subgroup silence | No skin tone, age, or device breakdown | Optical and electrical differences | Equity and safety risk | Report subgroups |
| Version drift | Record count changes after an update | Dataset revised | Pipeline breaks or cohort shifts | Pin version and DOI |
| Test-set reuse | The same split across many papers | Leaderboard tuning | Inflated and saturated results | Private holdout |
| Synthetic validation | Synthetic signals used for performance | Controlled distribution | Unsupported claim | Synthetic only for pipelines |
| Lead-off false event | A flat segment is classified as an event | Quality not checked | False alarm | Lead-off flagging |
| Epoch boundary error | Events split across epochs are mislabelled | Fixed epoch boundaries | Under-counted events | Boundary-aware scoring |
| Montage mismatch | Training and deployment montages differ | Different channel sets | Channel-specific failure | Harmonise or restrict |
| Optical calibration | PPG model ignores skin tone and perfusion | Dataset lacks diversity | Equity and safety risk | Stratified validation |
| Transmission gap | Missing packets look like bradycardia | No gap flag | False event | Gap detection |

Red flags that should stop a signal project:

- The split unit is a window.
- The label is a statement with no event time, and the claim is event detection.
- The deployment device is not represented in any dataset.
- The metric is AUROC and the decision has a fixed capacity.
- The evaluation prevalence is not reported.
- The filter is different between training and deployment.
- The annotation manual and version are unknown.
- The dataset has 15 subjects and the claim is clinical.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The record key is correct | Splitting depends on it | Records relabeled across releases | Reconcile counts with the release |
| The patient key is correct | Patient-level splitting depends on it | Missing or reused identifiers | Linkage audit |
| The label measures the event | The metric depends on it | Statement labels or missed events | Sample adjudication |
| The device matches deployment | Transport depends on it | Different sensor or placement | Device-shift test |
| The filter matches deployment | Morphology depends on it | Different band-pass | Re-filter and compare |
| The sampling rate is adequate | The task depends on it | Aliasing or lost morphology | Spectral check |
| The prevalence is known | PPV depends on it | Cohort selected for disease | Reweight to target |
| Quality exclusion is unbiased | Evaluation depends on it | Quality correlates with outcome | Sensitivity analysis |
| The annotation is complete | Event sensitivity depends on it | Partial annotation | Audit a sample |
| The metric matches the decision | Utility depends on it | AUROC for a capacity constraint | Metric memo |
| The subgroup has enough events | Equity analysis depends on it | Small counts | Confidence intervals |
| The licence covers the use | Product depends on it | Research-only terms | Clause review |
| The lead-off flag is reliable | Quality exclusion depends on it | Device-specific behaviour | Compare with visual review |
| The epoch boundaries preserve events | Scoring depends on it | Events split across epochs | Boundary-aware rescoring |
| The montage matches deployment | EEG transfer depends on it | Different channel sets | Channel harmonisation |
| The optical signal is calibrated across skin tones | Equity depends on it | Dataset lacks diversity | Stratified test |
| Transmission gaps are flagged | Event detection depends on it | Missing packets | Gap detection |

The deepest limit is that signal classification is not clinical benefit. A model that detects AF does not by itself reduce stroke, and a model that detects deterioration does not by itself reduce mortality. Those are workflow and outcome claims that require a prospective evaluation.

## 12. Exercises

**Exercise 1. Classify signal datasets by device and annotation.**
Deliverable: a table of six datasets with device, channels, sampling rate, annotation depth, and claim supported.
Acceptance criteria: six datasets; every row states what it cannot answer; one device mismatch identified.
Time: 40 minutes.

**Exercise 2. Audit a window split.**
Deliverable: a script or spreadsheet that computes record and patient overlap and the share of leaked test windows.
Acceptance criteria: reproduces the section 7 output; proposes a record-level split; names the affected records.
Time: 50 minutes.

**Exercise 3. Evaluate label provenance.**
Deliverable: a provenance record for one dataset, with label type, reader count, and agreement or its absence.
Acceptance criteria: cites the dataset paper or card; distinguishes statement from episode; states the event-level implication.
Time: 45 minutes.

**Exercise 4. Compute precision at capacity.**
Deliverable: a table of PPV across four prevalences at fixed sensitivity and specificity, with an interpretation.
Acceptance criteria: arithmetic correct; target prevalence justified; workload implication stated.
Time: 35 minutes.

**Exercise 5. Check device fit.**
Deliverable: a comparison of the training device and the deployment device on sensor, placement, sampling, filter, and firmware.
Acceptance criteria: five dimensions; one mismatch quantified; a test proposed.
Time: 40 minutes.

**Exercise 6. Design preprocessing.**
Deliverable: a filter and normalisation specification with fit scope and storage.
Acceptance criteria: causal where real-time; fit on training only; parameters versioned.
Time: 45 minutes.

**Exercise 7. Detect device shift.**
Deliverable: an evaluation plan that holds out a device or site and reports event-level metrics.
Acceptance criteria: holdout named; event-level metric; confidence intervals.
Time: 45 minutes.

**Assessment note.** Exercises 2, 3, 4, and 5 produce checkable artefacts: a leakage report, a provenance record, a PPV table, and a device comparison. Exercises 1, 6, 7, and 8 are design tasks and are marked against the rubric. Exercise 8 is the integration task.

**Exercise 8. Write a signal data card.**
Deliverable: a two-page card with device, acquisition, annotation, preprocessing, uses, and prohibited claims.
Acceptance criteria: every field populated or marked unknown; one prohibited claim; no marketing language.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. Why is a signal dataset a device dataset?**
The sensor, placement, sampling rate, filter, and firmware shape the signal. Two datasets of the same modality can have different distributions. A model trained on one device is not automatically a model for another.

**Item 2. What is the difference between a record and a window?**
A record is a continuous recording from one device session; a window is a segment of it. Windows from the same record are correlated. Splitting by window leaks the record.

**Item 3. Why is a record-level split stronger than a window-level split?**
It keeps all windows from a record on one side, preventing the model from seeing the same rhythm, noise, and patient in training and test.

**Item 4. What does precision at the top 5 percent encode?**
A fixed review capacity: only 5 percent of the cohort can be reviewed. The metric rewards ranking true positives at the top and reflects the operational constraint.

**Item 5. Why does PPV fall when prevalence falls?**
Because false positives from the non-diseased majority accumulate faster than true positives from the smaller diseased group. Sensitivity and specificity can be unchanged while PPV changes by an order of magnitude.

**Item 6. What is the difference between a statement label and an episode label?**
A statement label describes an interpretation attached to a record; an episode label marks the onset and offset of an event. Only the latter supports time-to-alarm and event-level detection.

**Item 7. Why can a non-causal filter be a problem?**
It uses future samples, which a real-time device cannot. The offline performance can differ from the deployed performance. Use a causal filter for real-time claims.

**Item 8. What is aliasing?**
The folding of content above the Nyquist frequency into the lower band when a signal is sampled or downsampled without an anti-aliasing filter. It creates spurious features.

**Item 9. Why is a 15-subject laboratory dataset insufficient for a clinical claim?**
It is small, selected, and collected under a protocol that does not match ambulatory or clinical use. It can support a laboratory method and not a clinical or consumer claim.

**Item 10. When is a synthetic signal acceptable?**
For pipeline development, augmentation research, and privacy-safe demos. It is not acceptable for clinical performance or fairness claims.

**Item 11. Why is a flatline not asystole until the lead is checked?**
A flatline can result from lead-off, saturation, or transmission loss. Asystole is a physiological diagnosis that requires a valid signal and a clinical context. Treating a technical flatline as an event creates false alarms and can mislead a review.

**Item 12. What is the difference between sensitivity at capacity and AUROC?**
Sensitivity at capacity is measured at a fixed review rate and reflects the workload. AUROC integrates over all thresholds and is insensitive to the chosen capacity. A model with a high AUROC can have poor sensitivity at the capacity the programme can staff.

**Item 13. Why does a firmware change matter if the sensor is the same?**
Firmware can change filtering, sampling, gain, and artefact handling. The signal distribution changes even though the hardware name does not. The device-shift test must be repeated after a firmware change.

**Item 14. Why is a challenge leaderboard not a device validation?**
The leaderboard evaluates a model on a hidden test set from the challenge distribution. It does not evaluate the deployment device, the target prevalence, the workflow, or the alarms per hour. A strong answer separates benchmark performance from device evidence. A weak answer treats the rank as validation.

**Item 15. What is the difference between a record-level and a patient-level split, and when can you not do the latter?**
A record-level split keeps all windows of a record together but allows another record from the same patient on the other side. A patient-level split keeps all records of a patient together. Without a patient identifier, the latter is impossible, and the limitation must be stated.

**Rubric for the design exercises (2, 5, 6, 7).** Full marks require a named device, a record-level split, a prevalence reweighting, and a prohibited-claim statement. Partial marks for a complete structure with no device test. No marks for a window-level evaluation presented as generalization.

## 14. Interview questions

**1. How would you split an ECG dataset?**
By patient, then by record, and by device or site where possible. A strong answer explains what each split estimates. A weak answer says random 80/20.

**2. A model reports 0.95 AUROC for AF detection. What do you ask?**
The split unit, the label source, the device, the prevalence, the PPV at target prevalence, the event-level sensitivity, and the alarm burden. A strong answer asks for the data card. A weak answer asks only for the sample size.

**3. Why is 12-lead data not a substitute for wrist data?**
The lead count, axis, placement, contact, and noise differ. A strong answer proposes a device-shift test. A weak answer says the model will generalise.

**4. What does precision at the top 5 percent mean operationally?**
Only 5 percent of the cohort can be reviewed. A strong answer connects the metric to staffing. A weak answer treats it as a generic ranking metric.

**5. How do you estimate the annotation depth of a dataset?**
Compare the annotated events with an independent review of a sample, and estimate the missed-event rate. A strong answer reports the uncertainty. A weak answer assumes the annotations are complete.

**6. What is the difference between a statement and an episode label?**
A statement is an interpretation; an episode has a start and an end. A strong answer connects the label to the claim. A weak answer uses them interchangeably.

**7. How do you handle a firmware change?**
Re-run the device-shift test and compare the signal and the performance. A strong answer treats firmware as part of the device. A weak answer assumes the product name implies the same signal.

**8. Why can a high-specificity model still generate too many false alarms?**
At low prevalence, even a small false-positive rate produces many false alarms relative to true positives. A strong answer computes alarms per 1,000. A weak answer focuses on specificity alone.

**9. How do you evaluate a sleep staging model?**
By epoch-level agreement and by subject-level agreement, with the scoring manual version and the cohort described. A strong answer reports both. A weak answer reports accuracy only.

**10. When would you refuse a signal dataset?**
When the device is wrong, the labels cannot support the event claim, the licence blocks the use, or the population is irrelevant. A strong answer treats refusal as a result. A weak answer uses it because it is large.

**11. What is the difference between event sensitivity and window sensitivity?**
Event sensitivity counts detected events; window sensitivity counts detected windows. A single event can produce many windows, so window sensitivity can be high while event sensitivity is low.

**12. How do you report subgroup performance for an optical signal?**
By skin tone, age, perfusion, and device, with confidence intervals and a description of the measurement conditions. A strong answer notes that skin tone is often unrecorded. A weak answer reports only age and sex.

**13. How do you evaluate a model on a device it has never seen?**
Hold out the device, report event-level and window-level metrics with intervals, and compare with the in-device result. A strong answer treats the drop as the finding. A weak answer averages across devices and hides the failure.

**14. What is the first check on a new signal dataset?**
The device and channel manifest, then the record and patient counts, then the label type. A strong answer verifies the sampling rate and filter before modeling. A weak answer starts with a class balance table.

**15. How do you report a signal result to a regulator?**
Device, firmware, sampling, filter, population, prevalence, split, event-level metrics, subgroup results, alarm burden, and monitoring. A strong answer separates the consumer claim from the clinical claim. A weak answer reports window accuracy.

**16. How do you handle a dataset with no episode annotations?**
Use it for record classification and pretraining, not for event detection. A strong answer states the claim that remains and proposes a collection for the event claim. A weak answer treats record labels as event labels.

## 15. Key terms

- **Accelerometer** — a sensor measuring acceleration.
- **Aliasing** — folding of high-frequency content into a lower band.
- **Annotation depth** — the proportion of events that are annotated.
- **Alarm burden** — the number of alarms per unit time.
- **Bandwidth** — the frequency range preserved by a system.
- **Baseline wander** — low-frequency drift in a signal.
- **Causal filter** — a filter that uses only past samples.
- **CODE-15%** — large Brazilian ECG dataset with weak labels.
- **Electrode** — a sensor that transduces electrical activity.
- **Episode** — a period of abnormal activity with onset and offset.
- **Event detection** — identifying the presence and time of an event.
- **Filter** — a transformation that removes frequency content.
- **Firmware** — device software that shapes the signal.
- **Icentia11K** — single-lead ambulatory ECG dataset.
- **Lead** — a potential difference between electrodes.
- **Montage** — the arrangement of electrodes and channels.
- **Notch filter** — a narrow filter at a specific frequency.
- **Nyquist frequency** — half the sampling rate.
- **Photoplethysmography** — optical measurement of blood volume change.
- **Precision at capacity** — true positives among the top k ranked.
- **PTB-XL** — public 12-lead ECG dataset.
- **Record-level split** — keeping all windows of a record on one side.
- **Sampling rate** — samples per second.
- **SCP-ECG** — standard for ECG statements.
- **Signal-to-noise ratio** — signal power relative to noise.
- **Sleep-EDF** — public polysomnography dataset.
- **Statement** — a diagnostic statement attached to a record.
- **Time-to-alarm** — delay between event onset and alarm.
- **TUH EEG** — Temple University EEG corpus.
- **Waveform** — a sampled physiological signal.
- **Weak label** — a noisy or indirect label.
- **WESAD** — wearable stress and affect dataset.
- **Window** — a fixed-length segment of signal.
- **Anti-aliasing filter** — a low-pass filter applied before downsampling.
- **Epoch** — a fixed analysis interval, often 30 seconds.
- **Event sensitivity** — the share of events detected.
- **False alarm rate** — false alarms per unit time.
- **Flatline** — a segment with no variation.
- **Impedance** — resistance at the electrode-tissue interface.
- **Motion artefact** — noise from movement.
- **Normalisation** — scaling signal amplitude.
- **Phase distortion** — waveform change from a filter.
- **Quality index** — a score summarising signal usability.
- **Resampling** — changing the sampling rate.
- **Saturation** — clipping at the device's dynamic range.
- **Signal chain** — the path from physiology to metric.
- **Skin tone** — melanin concentration affecting optical signals.
- **Window sensitivity** — the share of windows classified correctly.
- **Ambulatory** — outside a clinical facility.
- **Holter** — a 24 to 48 hour ECG recording.
- **Photoplethysmogram** — optical blood-volume signal.
- **Polysomnography** — overnight sleep recording.
- **Apnea** — cessation of airflow during sleep.
- **Epoch-level agreement** — agreement per scoring interval.
- **Subject-level agreement** — agreement per participant.
- **Lead-off** — loss of electrode contact.
- **Dynamic range** — the amplitude range a device can record.
- **Causal** — using only past samples.
- **Zero-phase** — filtering without phase distortion, usually offline.
- **Annotation file** — a file storing event times and labels.
- **Beat** — a single cardiac cycle.
- **Bradycardia** — a slow heart rate.
- **Tachycardia** — a fast heart rate.
- **Heart rate variability** — variation in beat-to-beat intervals.
- **Respiratory rate** — breaths per minute.
- **Oxygen saturation** — estimated arterial oxygen from PPG.
- **Perfusion** — blood flow at the sensor site.
- **Motion flag** — a marker for movement artefact.
- **Scoring version** — the manual edition used for annotations.
- **Channel** — a single recorded signal stream.
- **Artifact** — an unwanted signal component.
- **Beat-to-beat interval** — time between consecutive beats.
- **Calibration** — agreement between predicted risk and observed frequency.
- **Detection latency** — delay from event onset to detection.
- **Event annotation** — a labelled event with timing.
- **Feature** — a derived signal characteristic.
- **Record** — a continuous recording from one session.
- **Segmentation** — dividing a signal into intervals.
- **Signal quality** — usability of a recording for the task.
- **Window label** — the label assigned to a window.
- **Band-pass filter** — a filter retaining a frequency range.
- **Contact quality** — adequacy of sensor-skin contact.
- **Digital twin** — a simulated physiological model.
- **Event onset** — the start time of an event.
- **Event offset** — the end time of an event.
- **Nominal sampling rate** — the rate specified by the device.
- **Quantisation** — amplitude resolution of a sampled signal.
- **Signal clipping** — truncation at the device limit.
- **Wandering baseline** — slow drift in the signal.

## 16. Sources

All URLs accessed 26 October 2026. Versions and access terms change; verify at the custodian page.

1. Wagner P, Strodthoff N, Bousseljot RD, et al. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* 2020;7:154. https://doi.org/10.1038/s41597-020-0495-6
2. PhysioNet. PTB-XL v1.0.3. https://physionet.org/content/ptb-xl/1.0.3/
3. PhysioNet. PTB-XL license. https://www.physionet.org/content/ptb-xl/view-license/1.0.0/
4. PhysioNet. George B. Moody PhysioNet Challenge 2025. https://moody-challenge.physionet.org/2025/
5. PhysioNet. George B. Moody PhysioNet Challenge 2026. https://moody-challenge.physionet.org/2026/
6. Ribeiro AH, Ribeiro MH, Paixao GMM, et al. Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications* 2020;11:1760. https://doi.org/10.1038/s41467-020-15432-4
7. PhysioNet. Sleep-EDF. https://physionet.org/content/sleep-edfx/
8. Temple University. TUH EEG Corpus. https://www.isip.piconepress.com/projects/tuh_eeg/
9. PhysioNet. MIMIC-IV waveform. https://physionet.org/content/mimic4wdb/
10. PhysioNet. Icentia11K continuous ECG. https://physionet.org/content/icentia11k-continuous-ecg/
11. Schmidt P, Reiss A, Duerichen R, et al. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI* 2018. https://doi.org/10.1145/3242969.3242985
12. National Sleep Research Resource. https://sleepdata.org/
13. PhysioNet. WFDB software package. https://physionet.org/content/wfdb-python/
14. American Academy of Sleep Medicine. Scoring manual. https://aasm.org/clinical-resources/scoring-manual/
15. International Electrotechnical Commission. IEC 60601 series. https://www.iec.ch/
16. International Electrotechnical Commission. IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/publication/22794
17. US Food and Drug Administration. AI-enabled device software functions guidance, draft January 2025. https://www.fda.gov/media/184856/download
18. European Parliament and Council. Regulation (EU) 2017/745 (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
19. PhysioNet. eICU and MIMIC waveform documentation. https://physionet.org/content/mimic4wdb/
20. National Sleep Research Resource. Data access. https://sleepdata.org/
21. PhysioNet. Icentia11K dataset page. https://physionet.org/content/icentia11k-continuous-ecg/
22. American Academy of Sleep Medicine. Scoring manual resources. https://aasm.org/clinical-resources/scoring-manual/
23. Module 21 survey file, section 7. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, imaging datasets: [`04_imaging-datasets.md`](04_imaging-datasets.md)
- EHR, claims, and ICU data: [`03_ehr-claims-and-icu.md`](03_ehr-claims-and-icu.md)
- Access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the baseline reproducibility lab: [`../../labs/`](../../labs/)

