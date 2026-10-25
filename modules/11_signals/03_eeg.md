# Electroencephalography: Montages, Frequency Bands, Artifacts, Seizures, and Sleep

> Module 11 — Physiological Signals and Wearables
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (signal fundamentals)
> Version 2026.3

## 1. Why this topic exists

Electroencephalography (EEG) measures the summed electrical activity of cortical neurons at the scalp. It is the reference signal for epilepsy, sleep, anesthesia depth, and brain-computer interfaces, and it is the most artifact-prone signal in this module. The amplitude of a cortical signal at the scalp is tens of microvolts. The amplitude of an eye blink is hundreds of microvolts. The amplitude of muscle activity is comparable to the cortical signal but at a much higher frequency. The result is a signal in which the largest deflections are usually not brain activity.

Three industry problems follow.

**Artifact that looks like a finding.** A frontal slow wave from an eye movement is indistinguishable, to a power-based classifier, from the delta activity of deep sleep. A muscle burst from jaw clenching looks like high-frequency seizure activity. An electrode pop looks like a spike. A pipeline that computes band power without artifact control will report sleep stages and seizure likelihoods that reflect the face and the jaw, not the brain.

**Montage-dependent measurements.** EEG is recorded as a set of channels, each of which is a difference between two electrodes. The same cortical event appears with different amplitude and polarity depending on the montage. A referential montage measures each electrode against a common reference; a bipolar montage measures adjacent electrodes against each other. A model trained on one montage and applied to another sees a different signal, even when the underlying brain activity is identical.

**Sleep staging without the rules.** Sleep staging is a decision procedure defined by the American Academy of Sleep Medicine (AASM). It uses 30 s epochs, specific waveform definitions (K-complexes, sleep spindles, slow waves), amplitude thresholds, and rules for continuation and transition. A model that predicts five stages from a wearable without implementing those definitions is not measuring sleep stages; it is predicting a label that was itself produced by a human applying the rules. The distinction matters for validation, for regulatory claims, and for clinical interpretation.

This topic covers the EEG chain: the 10-20 electrode system, montages, frequency bands, artifact recognition, transient detection, seizure annotation, and AASM sleep staging. It builds on Topic 01 for sampling and filtering and on Topic 02 for the idea of a spatial projection. It sets up Topic 06 for feature engineering and Topic 07 for representation learning.

### 1.1 Three composite failures

**The sleep study that scored the face.** A research group builds a sleep stager on a single frontal EEG channel. The training data are polysomnography recordings with a full montage, but the model uses only F4-M1. The model learns that large frontal slow waves indicate N3. In deployment, participants sleep with their eyes closed and occasionally blink or roll their eyes under closed lids; the frontal channel records those movements as slow waves. The model reports more N3 than the polysomnography reference. The error is not in the classifier; it is in the feature, which cannot distinguish a slow wave from an eye movement in one channel.

**The seizure detector that alarmed on chewing.** A wearable with a temporal electrode detects high-frequency bursts and flags them as electrographic seizures. The bursts are temporalis muscle activity during meals. The model was trained on in-hospital EEG with a full montage and an EMG channel; in deployment, the EMG channel is absent and the muscle artifact looks like the high-frequency onset of a seizure. The false alarms lead users to stop wearing the device.

**The montage mismatch.** A model is trained on referential EEG from one laboratory and deployed on a bipolar montage from another. The amplitude of a generalized slow wave is smaller in the bipolar montage because the widespread activity is subtracted between adjacent electrodes. The N3 amplitude threshold is never met, and the model under-reports deep sleep. The spectral content is similar; the amplitude is not.

The three failures share a structure: a feature that was valid in the training montage or sensor set is applied outside it. The check that catches them is the same: state the montage, the channel, and the derivation for every feature, and test the feature across montages before deployment.

A fourth failure is a claim that exceeds the recording. A 20 min routine EEG that is normal does not exclude epilepsy, and a single night of sleep staging does not characterise a patient's sleep. The negative result is bounded by the observation window, and the report should say so.

### 1.2 Why EEG is the hardest modality in this module

EEG has four properties that make it harder than ECG or PPG.

- **Low signal-to-artifact ratio.** Cortical activity at the scalp is tens of microvolts; eye and muscle artifacts are hundreds.
- **Montage dependence.** The same event has different amplitude and polarity depending on the channel derivation.
- **State dependence.** The signal changes with sleep, wakefulness, medication, and age.
- **Reference dependence.** A referential recording is only as good as its reference.

Each property is manageable with a documented chain. None is manageable with a model alone.

The cost of getting the chain wrong is different from ECG because the decisions are often softer. A false sleep stage changes a wellness report or a research endpoint; it rarely harms a patient directly. A false seizure alarm, by contrast, can lead to medication changes or to a night in an emergency department. The same technical error has different consequences depending on the decision it feeds, which is why the decision must be specified before the chain is designed.

The industry question for EEG is not "can we classify this signal" but "what can this montage and this duration observe." A team that starts with the classifier and works backward to the sensor will overclaim. A team that starts with the observation and builds the smallest classifier that supports it will ship a narrower, defensible product.

A second asymmetry is that EEG is often used to answer a question that the recording cannot answer. A routine 20 min EEG in a patient with paroxysmal events has a low yield; a 72 h ambulatory EEG has a higher yield. The recording duration is a property of the measurement, not of the algorithm, and a negative short recording is not evidence of absence.

## 2. Learning outcomes

By the end of this topic you can:

1. **Describe** the 10-20 electrode system and construct a longitudinal bipolar montage from channel labels.
2. **Explain** why the same event has different amplitude and polarity in referential and bipolar montages.
3. **Compute** band power in delta, theta, alpha, beta, and gamma bands for a 30 s epoch, with stated window and normalization.
4. **Identify** eye blink, lateral eye movement, muscle, line noise, electrode pop, and sweat artifacts from their morphology and topography.
5. **Apply** the AASM V3 rules for wake, N1, N2, N3, and REM to a 30 s epoch, including the slow-wave duration criterion.
6. **Distinguish** a spike, a sharp wave, and a seizure pattern from artifact, and state the localizing evidence.
7. **Evaluate** a seizure detector against an annotated dataset with patient-level splitting and event-level scoring.
8. **Critique** a consumer sleep-staging claim for sensor set, reference standard, per-stage agreement, and population.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

The outcomes are ordered so that the spatial and spectral layers come before the decision layers. This is the same order as the chain. A stage label produced without a montage, a band definition, and an artifact check is a label without a measurement behind it.

Two outcomes are assessed by critique rather than calculation. Outcome 7 asks you to evaluate a seizure detector, which requires event-level thinking and an understanding of annotation. Outcome 8 asks you to critique a consumer sleep claim, which requires knowing what the AASM rules require and what a reduced sensor set can observe. Both are the outcomes most often skipped in technical training and most often tested in product review.

## 3. The core mental model

EEG is a spatial and spectral measurement. The spatial part is the montage; the spectral part is the band power; the temporal part is the transient. A finding is a combination of all three.

```text
    cortex (pyramidal cells, synchronous activity)
                 |
        volume conduction through skull and scalp
                 |
        electrodes at 10-20 positions
                 |
     +-----------+-----------+
     |                       |
 referential             bipolar
 (electrode - ref)     (electrode - neighbour)
     |                       |
     +-----------+-----------+
                 |
         channels (montage)
                 |
    +------------+------------+------------+
    |            |            |            |
 spectrum     transients    topography    continuity
 (bands)      (spikes,      (field,       (state,
              spindles)     phase)        transitions)
    |            |            |            |
    +------------+------------+------------+
                 |
        decision: stage, event, state
```

Redraw it with the electrode names on the left and the decision on the right. The key idea is that no single channel and no single feature carries the decision; the decision is a pattern across channels, frequencies, and time.

### 3.1 The 10-20 system

The international 10-20 system places electrodes at fixed fractional distances across the skull. The letters indicate the region: F (frontal), C (central), P (parietal), O (occipital), T (temporal). The numbers indicate laterality: odd numbers on the left, even on the right, and z on the midline. Fp1 and Fp2 are frontopolar; F7 and F8 are anterior temporal; T3 and T4 are mid-temporal (also called T7 and T8 in the modified combinatorial nomenclature); T5 and T6 are posterior temporal (P7 and P8); O1 and O2 are occipital.

A full montage uses 19 to 21 electrodes. A reduced montage uses fewer, and a wearable may use 1 to 4. The reduction changes what can be localized. A single frontal electrode can detect a blink and a slow wave but cannot distinguish a frontal event from a generalized one.

The naming is not arbitrary. The number indicates the distance from the midline in 10% or 20% steps, and the letter indicates the anterior-posterior position. A technologist can reconstruct the position from the name, which is why the names are used in reports and in datasets. A pipeline that renames channels to "ch1, ch2" removes that information and makes the montage unreadable.

### 3.2 Referential and bipolar montages

A **referential** channel is the voltage at one electrode minus the voltage at a reference. The reference may be a linked-ear reference, an average reference, or a single electrode such as Cz. Referential montages preserve the absolute field of an event, but they are sensitive to reference contamination: if the reference electrode is active, every channel shows the reference activity.

A **bipolar** channel is the voltage difference between two adjacent electrodes. The longitudinal bipolar montage chains electrodes from front to back:

```text
 Left temporal:   Fp1 - F7 - T3 - T5 - O1
 Left parasagittal: Fp1 - F3 - C3 - P3 - O1
 Midline:         Fz - Cz - Pz
 Right parasagittal: Fp2 - F4 - C4 - P4 - O2
 Right temporal:  Fp2 - F8 - T4 - T6 - O2
```

Each chain is displayed as a sequence of differences: Fp1-F7, F7-T3, T3-T5, T5-O1. The pattern of two chains over each hemisphere is why the montage is called "double banana."

The bipolar montage has a property that referential montages lack: **phase reversal**. When a focal negative event occurs at an electrode shared by two adjacent channels, the two channels deflect in opposite directions. The shared electrode is the maximum. This is the main localizing tool in clinical EEG. It also means that the amplitude of an event depends on the inter-electrode distance and on the orientation of the dipole relative to the chain.

### 3.3 Frequency bands

EEG is conventionally divided into bands. The boundaries are conventions, not physical laws, and they differ slightly across sources.

| Band | Frequency | Typical state | Common confusion |
|---|---|---|---|
| Delta | 0.5-4 Hz | Deep sleep, encephalopathy | Eye movements, sweat |
| Theta | 4-8 Hz | Drowsiness, N1 | Slow eye movements |
| Alpha | 8-13 Hz | Relaxed wakefulness, eyes closed | Mu rhythm (same band, different topography) |
| Beta | 13-30 Hz | Active wakefulness, drugs | Muscle artifact |
| Gamma | 30-100 Hz | Cognitive processing, research | Muscle, line noise harmonics |

The alpha rhythm is the most useful single feature in adult EEG. It is maximal over the occipital regions, appears when the eyes are closed, and attenuates with eye opening. The AASM wake rule uses alpha in the occipital region: an epoch is wake if more than 50% contains posterior dominant alpha rhythm.

### 3.4 The transient

A transient is a brief waveform that stands out from the background. The clinically important transients are:

- **Spike**: a negative deflection of 20 ms to 70 ms, often followed by a slow wave.
- **Sharp wave**: a negative deflection of 70 ms to 200 ms.
- **K-complex**: a negative sharp wave followed by a positive component, at least 0.5 s, maximal frontally.
- **Sleep spindle**: a train of 11 Hz to 16 Hz activity, at least 0.5 s, maximal centrally.
- **Slow wave**: 0.5 Hz to 2 Hz, peak-to-peak above 75 µV, maximal frontally.
- **Seizure pattern**: rhythmic activity that evolves in frequency, amplitude, and topography, usually lasting more than 10 s.

The first two are epileptiform; the next three are sleep features; the last is a seizure. Each has a definition with an amplitude, a duration, a frequency, and a topography. A detector that omits one of these dimensions is not detecting the same thing.

The duration thresholds are not interchangeable. A spike is defined up to 70 ms, a sharp wave from 70 ms to 200 ms, a K-complex from 0.5 s. At 250 samples/s, 70 ms is 17.5 samples and 0.5 s is 125 samples. A detector with a 100 ms smoothing window cannot resolve a spike; a detector with a 10 ms window sees noise. The window must match the transient.

### 3.5 Measurement, feature, biomarker, decision

The distinction from Topic 01 applies with particular force in EEG.

- **Measurement**: the voltage difference between two electrodes, in microvolts, at a stated sampling rate and filter setting.
- **Feature**: band power, spectral edge, spindle count, slow-wave fraction.
- **Biomarker**: a feature with a validated relationship to a state or outcome, such as alpha attenuation as a marker of drowsiness, or sleep spindle density as a marker of learning.
- **Decision**: score the epoch as N2, flag a seizure, adjust anesthesia.

A band-power feature is not a sleep stage. A sleep stage is a decision made by applying rules to features and context. The rules include the epoch length, the amplitude threshold, the topography, and the continuation logic. A model that predicts the stage directly from the raw signal has learned the rules implicitly, and its errors are the rules it did not learn.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| EEG | Electroencephalography; scalp recording of cortical activity | Calling any brain signal EEG | µV |
| 10-20 system | Standard electrode placement | Assuming all montages use it | electrode names |
| Electrode | Conductive contact on the scalp | Confusing with a channel | position |
| Montage | Ordered set of channels | Calling a single channel a montage | named montage |
| Referential | Channel = electrode minus reference | Ignoring reference contamination | µV |
| Bipolar | Channel = electrode minus neighbour | Comparing amplitudes across montages | µV |
| Double banana | Longitudinal bipolar montage | Treating chains as independent | named montage |
| Phase reversal | Opposite deflections at a shared electrode | Calling it artifact | polarity |
| Common average reference | Average of all electrodes used as reference | Including the eyes in the average | µV |
| Linked ears | Ears joined as reference | Assuming the ears are silent | µV |
| Delta | 0.5-4 Hz | Counting eye movements as delta | Hz |
| Theta | 4-8 Hz | Confusing with alpha at the boundary | Hz |
| Alpha | 8-13 Hz | Calling mu rhythm alpha | Hz |
| Beta | 13-30 Hz | Counting muscle as beta | Hz |
| Gamma | 30-100 Hz | Counting line noise as gamma | Hz |
| Mu rhythm | 8-13 Hz over sensorimotor cortex | Confusing with occipital alpha | Hz |
| Posterior dominant rhythm | Alpha over occipital regions | Assuming it exists in every adult | Hz, µV |
| Spike | Negative transient, 20-70 ms | Calling any sharp transient a spike | ms, µV |
| Sharp wave | Negative transient, 70-200 ms | Confusing with spike | ms, µV |
| K-complex | Frontal negative-positive wave, ≥0.5 s | Counting a blink as a K-complex | s, µV |
| Sleep spindle | 11-16 Hz train, ≥0.5 s, central | Counting alpha as a spindle | Hz, s |
| Slow wave | 0.5-2 Hz, >75 µV, frontal | Counting sweat artifact | Hz, µV |
| Seizure | Evolving rhythmic activity, usually >10 s | Calling a single spike a seizure | categorical |
| Electrographic seizure | Seizure visible on EEG | Requiring clinical signs | categorical |
| Arousal | Abrupt shift in EEG frequency, ≥3 s | Confusing with a movement | s |
| Epoch | Fixed scoring window, usually 30 s | Using a different length silently | s |
| Stage W | Wake | Calling any alpha epoch wake without the 50% rule | categorical |
| Stage N1 | Light sleep | Calling all low-amplitude epochs N1 | categorical |
| Stage N2 | Sleep with spindles or K-complexes | Calling any spindle epoch N2 without continuation rules | categorical |
| Stage N3 | Slow-wave sleep | Using power instead of duration | categorical |
| Stage R | REM sleep | Scoring from EEG alone | categorical |
| REM | Rapid eye movement | Confusing with slow eye movement | categorical |
| EOG | Electrooculography | Treating it as EEG | µV |
| EMG | Electromyography | Treating it as EEG | µV |
| Impedance | Electrode-skin resistance | Ignoring high impedance | kΩ |
| Electrode pop | Abrupt impedance change | Calling it a spike | µV |
| Sweat artifact | Slow baseline swing | Calling it delta | Hz |
| Line noise | 50/60 Hz interference | Calling it gamma | Hz |
| Artifact | Non-cerebral signal | Treating all artifact as noise to delete | varies |
| Annotator | Human who marks events | Treating annotations as ground truth | categorical |
| Inter-rater agreement | Concordance between annotators | Reporting only one annotator | kappa |
| Vertex sharp wave | Sharp negative transient at Cz | Confusing with a spike | ms, µV |
| Sleep efficiency | Sleep time divided by time in bed | Using recorded time as denominator | % |
| Hypnogram | Stage sequence over the night | Reporting only stage percentages | categorical |
| Slow-wave fraction | Proportion of epoch above amplitude threshold | Using band power instead | % |
| Spectral edge | Frequency below which a fraction of power lies | Reporting without the fraction | Hz |
| Relative power | Band power divided by total power | Comparing across different band sets | dimensionless |
| Absolute power | Band power in physical units | Comparing across window lengths | µV² |
| Spectrogram | Time-frequency representation | Treating it as a diagnosis | dB/Hz |
| Cortical dipole | Source of scalp field | Treating scalp maximum as source | model |
| Volume conduction | Spread of field through tissue | Assuming zero spatial mixing | physics |
| Common mode | Signal shared by both inputs | Ignoring reference contamination | µV |
| Impedance mismatch | Difference in electrode impedance | Underestimating mains coupling | kΩ |
| 10-20 | Electrode placement convention | Assuming universal coverage | named positions |
| MCN | Modified combinatorial nomenclature | Confusing T3/T7 names | naming |
| Anesthesia index | Processed EEG depth measure | Treating as raw EEG | dimensionless |
| Electrode gel | Conductive medium | Ignoring drying over long records | — |
| Event-level scoring | Scoring whole events, not epochs | Fragmenting a seizure | count |
| False alarm rate | Alerts per unit time | Reporting only sensitivity | per 24 h |

## 5. The mechanics, step by step

The procedure is written for a clinical or research EEG with a known montage, sampled at 250 Hz or higher. It scales to a wearable with fewer channels by reducing the localizing steps and the confidence of the conclusions.

### Step 1. Define the question and the montage

**Input:** the clinical or product question, the available channels, and the recording duration.

**Output:** a montage specification and a statement of what the montage can and cannot localize.

**Rationale:** the montage determines the spatial resolution. A 19-channel 10-20 montage can localize a focal event to a lobe or a region. A 4-channel wearable can detect a generalized event or a state change but cannot localize. A question about focal epilepsy requires a montage that samples the relevant region; a question about sleep stage requires the channels that the AASM rules use (EEG, EOG, EMG).

**Branch point:** if the question is sleep staging, use at least one frontal EEG, one central EEG, EOG, and chin EMG. If the question is seizure detection, use a montage that covers both hemispheres and the temporal chains. If the question is anesthesia depth, a frontal derivation is often sufficient because the processed index is designed for it.

**Edge case:** a montage can be nominally 19-channel but have a failed electrode. The channel count is not the coverage; the usable channels are.

**Channel count and spatial resolution.** The number of electrodes determines the smallest distinguishable field, not the number of independent sources. With 19 electrodes spaced about 6 cm apart, a focal cortical source produces a field spanning several electrodes, and the localization uncertainty is on the order of centimetres. With 64 or 128 electrodes, the spacing is smaller and the uncertainty improves, but the volume conduction still blurs the source. A 4-channel wearable has an uncertainty on the order of lobes, not centimetres.

| Montage size | Typical spacing | Localization | Suitable questions |
|---:|---|---|---|
| 1-4 channels | 6-10 cm | lobe or global | sleep-wake, generalized events |
| 19-21 channels | 5-7 cm | region | focal epilepsy, sleep staging |
| 32-64 channels | 3-5 cm | sub-region | research, source modeling |
| 128+ channels | 1-3 cm | improved but blurred | high-density research |

The table is a guide to what a claim can say, not a hardware recommendation. A 4-channel device that claims focal localization is overreaching regardless of its algorithm.

### Step 2. Verify electrode placement and impedance

**Input:** the channel labels, the impedance measurements, and the recording.

**Output:** a channel validity mask and a placement confidence flag.

**Rationale:** an electrode with high impedance picks up line noise and movement. A misplaced electrode changes the topography and can make a generalized event look focal. Impedance is measured before recording in most clinical systems; a value above 5 kΩ to 10 kΩ is usually flagged, though the acceptable range depends on the amplifier.

**Arithmetic:** at 50 Hz or 60 Hz, a high-impedance electrode forms a voltage divider with the amplifier input impedance. If the electrode impedance is 20 kΩ and the input impedance is 10 MΩ, the attenuation is 20/10,000 = 0.2%, which is negligible for the signal but the common-mode rejection degrades because the two inputs are mismatched. A 20 kΩ mismatch can convert a 1 V common-mode mains signal into a 2 mV differential signal, which is larger than the EEG.

**Branch point:** if impedance is high on one electrode, that channel is unreliable, and any bipolar channel that includes it is also unreliable. Exclude the channel rather than interpolating it, unless the montage can be rebuilt without it.

**Edge case:** impedance can be acceptable at the start and degrade during the recording as the gel dries. A channel that becomes noisy halfway through is a candidate for a time-varying validity mask, not a global rejection.

**Impedance and noise arithmetic.** The thermal noise of an electrode-skin interface is not the main problem at EEG frequencies; the main problem is the mismatch. Two inputs with impedances Z1 and Z2 form a bridge. A common-mode voltage Vc appears as a differential voltage of approximately Vc × (Z1 - Z2)/(Zin), where Zin is the amplifier input impedance. With Vc = 1 V, Zin = 10 MΩ, and a mismatch of 20 kΩ, the differential error is 2 mV. That is two orders of magnitude larger than the EEG. Matching the impedances to within 1 kΩ reduces the error to 100 µV, which is still large relative to a 20 µV signal, which is why high input impedance and active electrodes matter.

The practical rule is to record impedance at the start and at the end, and to flag any channel whose impedance changes by more than a factor of two. The flag is a quality statement, not a rejection.

### Step 3. Check amplifier settings and filters

**Input:** the hardware filter settings, the sampling rate, and the display settings.

**Output:** a documented filter chain.

**Rationale:** clinical EEG displays use a low-frequency filter around 0.5 Hz to 1 Hz and a high-frequency filter around 70 Hz, with a notch at the mains frequency. These settings are for display; the stored signal may be wider. A model trained on display-filtered data and applied to raw data sees a different signal, particularly for slow waves and for muscle artifact.

**Arithmetic:** a 0.5 Hz high-pass attenuates a 0.25 Hz sweat artifact by 55% but also attenuates a 0.5 Hz slow wave by 29%. A 70 Hz low-pass preserves most EEG content but removes muscle artifact above 70 Hz, which is useful for display and potentially harmful for a muscle-detection algorithm.

**Branch point:** if the question is N3 sleep, the high-pass must be low enough to preserve 0.5 Hz content. A 1 Hz high-pass, common in clinical display, attenuates the slow waves that define N3.

**Edge case:** the notch filter can ring around electrode pops and can create the appearance of a slow wave after a transient. Check the raw signal before trusting a notched one.

**Filter settings and the constructs they change.** The display filter is not neutral.

- A 1 Hz high-pass removes sweat and movement but attenuates the 0.5 Hz slow waves that define N3. A sleep study should use 0.3 Hz or lower for the EEG channels.
- A 70 Hz low-pass removes muscle but also removes high-frequency seizure onset patterns. A seizure detector should use a wider band or a separate high-frequency channel.
- A 50 Hz or 60 Hz notch removes line noise but can remove a genuine 50 Hz oscillation, which is rare but reported in some research contexts.

The filter chain should be chosen per construct: sleep, seizure, anesthesia, and research each have different requirements. A single chain for all four is a compromise that is wrong for at least one.

### Step 4. Assess artifacts

**Input:** the multi-channel signal and the auxiliary channels (EOG, EMG, accelerometer).

**Output:** an artifact mask per channel and per time segment, with artifact type.

**Rationale:** artifact identification is the core skill of EEG analysis. Each artifact has a signature that combines morphology, topography, and timing. Removing artifact without identifying it risks removing the finding.

**Signatures:**

| Artifact | Morphology | Topography | Frequency | Distinguishing feature |
|---|---|---|---|---|
| Eye blink | Large positive frontal deflection | Fp1, Fp2, F7, F8 | 0.5-2 Hz | No posterior field; synchronous with EOG |
| Lateral eye movement | Opposite polarity at F7/F8 | Anterior temporal | 0.5-2 Hz | Horizontal EOG correlation |
| Muscle | High-frequency spiky activity | Frontal, temporal | 20-100 Hz | Bursts with jaw clench |
| Line noise | Sinusoidal | All channels | 50/60 Hz | Exact frequency; impedance correlation |
| Electrode pop | Abrupt step with exponential decay | One electrode | broadband | Isolated; impedance change |
| Sweat | Slow baseline swing | Variable | < 0.5 Hz | Gradual; no field |
| Movement | Broadband with baseline shift | Multiple channels | variable | Accelerometer correlation |
| ECG | Periodic QRS-like spikes | Variable | 1-2 Hz | Heart-rate periodicity |

**Branch point:** if the artifact is isolated to one channel, exclude the channel for that segment. If it is global, exclude the segment or use an artifact regression method. If it overlaps the event of interest, the event is uninterpretable.

**Edge case:** an eye blink and a frontal spike can look similar in a single frontal channel. The blink has no posterior field and is time-locked to the EOG; the spike often has a phase reversal and a following slow wave. A single-channel detector cannot distinguish them reliably.

**Artifact regression.** An alternative to rejection is to model the artifact and subtract it. For eye blinks, a common approach is to regress the EOG channels onto the EEG channels and subtract the fitted component. The method preserves the EEG during the blink but assumes that the EOG is a clean measure of the artifact and that the propagation is linear. It fails when the blink is asymmetric, when the EOG electrode is misplaced, or when the EEG and EOG share a source.

For muscle artifact, regression is less effective because the muscle signal is broadband and spatially broad. Independent component analysis (ICA) can separate some muscle components, but it requires many channels and a stationary mixture. A 4-channel wearable cannot use ICA reliably.

The choice between rejection and regression is a trade-off between data loss and residual artifact. Rejection is simpler and unbiased if the rejected segments are missing at random; regression preserves data but can remove neural signal correlated with the artifact.

### Step 5. Segment into epochs

**Input:** the continuous record and the scoring rules.

**Output:** epochs with start and end times and a stage or event label.

**Rationale:** the epoch length is defined by the decision. Sleep staging uses 30 s. Spike detection may use 1 s to 4 s windows. Seizure detection uses event-level annotation, not fixed epochs. Using the wrong window changes the feature and the label.

**Arithmetic:** a 30 s epoch at 250 Hz is 7,500 samples. A 1 s window is 250 samples. A 4 s window is 1,000 samples. The frequency resolution of a 30 s rectangular window is 0.033 Hz; of a 1 s window, 1 Hz. A 1 s window cannot resolve the delta band boundaries.

**Branch point:** if the stage rules require the previous epoch (N2 continuation), the segmentation must preserve order and context. An epoch cannot be scored independently of its neighbours.

**Edge case:** the recording start time may not align with the epoch grid. AASM specifies epochs starting at the study start; an arbitrary offset changes the boundaries and the stage assignments.

**Windowing the epoch.** An epoch can be analysed as a single 30 s window or as a sequence of shorter windows. A single window gives one spectrum with 0.033 Hz resolution. A sequence of 1 s windows gives 30 spectra with 1 Hz resolution and a time course. For staging, the single window is closer to the AASM rule, which scores the epoch as a unit. For event detection, the sequence is better because it localizes the event in time. The two analyses answer different questions and should not be substituted for each other. A staging model trained on 30 s features cannot localize a 3 s arousal, and an event detector trained on 1 s windows cannot apply the 20% epoch rule. The window is part of the construct.

### Step 6. Compute the spectrum and band power

**Input:** the epoch, the sampling rate, and the band definitions.

**Output:** absolute and relative power per band, with the window and normalization stated.

**Rationale:** band power is the most common EEG feature. It is also the most easily miscomputed. The choices that matter are the window function, the detrending, the frequency resolution, and whether the power is absolute or relative.

**Arithmetic:** for a sine of amplitude A, the power is A²/2. A 50 µV alpha sine has power 1,250 µV². A 10 µV alpha sine has power 50 µV². A relative power is the band power divided by the total power in a defined range; changing the range changes the denominator and the value.

**Branch point:** if the total power is dominated by artifact, relative band power is distorted. A 50 Hz line noise at 20 µV has power 200 µV², which can exceed the alpha power and drive the relative alpha toward zero even though the alpha is unchanged.

**Edge case:** the FFT length must cover the epoch. Truncating a 30 s epoch to a 4,096-sample segment at 250 Hz uses only 16.4 s and discards 45% of the data. Zero-padding to 8,192 preserves the full epoch and gives 0.0305 Hz bins.

**Absolute versus relative power.** Both are useful, and they answer different questions.

- Absolute power in µV² describes the amplitude of the activity. It is comparable across epochs and channels if the derivation and window are the same.
- Relative power describes the share of the total. It is robust to amplifier gain and to overall amplitude differences, but it is sensitive to the denominator and to artifact.

A report that gives only relative alpha cannot distinguish a low-amplitude alpha from a high-amplitude alpha with the same share. A report that gives only absolute power cannot compare subjects with different skull thickness and electrode impedance. Report both, and state the band set used for the denominator.

**Window choice.** A Hann window reduces leakage but broadens peaks. A rectangular window has the best nominal resolution but leaks. For band power over wide bands, the choice matters less than for peak detection. For alpha peak frequency, a Hann or Hamming window is preferable. The window must be stated because it changes the power by a few percent and the peak by a fraction of a bin.

### Step 7. Detect transients

**Input:** the filtered signal and the transient definitions.

**Output:** candidate events with time, channel, amplitude, duration, and topography.

**Rationale:** transients are defined by duration, amplitude, frequency, and topography. A detector that uses only amplitude will fire on artifact. A detector that uses only frequency will fire on continuous rhythms.

**Methods:** a spindle detector band-passes 11 Hz to 16 Hz, computes the envelope, and requires a duration above threshold. A K-complex detector looks for a frontal negative deflection of at least 0.5 s with a positive component. A spike detector looks for a negative deflection of 20 ms to 70 ms with a field across adjacent channels.

**Branch point:** if the transient is present in one channel only, it may be an electrode artifact. If it is present across a chain with a phase reversal, it is likely cerebral. If it is present globally with no field, it may be a reference artifact.

**Edge case:** a sleep spindle and a burst of alpha can overlap in frequency. The topography (central versus occipital) and the context (sleep versus wake) distinguish them.

**Transient detection arithmetic.** A spindle detector that band-passes 11 Hz to 16 Hz and thresholds the envelope must choose an envelope window. A 0.2 s RMS window smooths the envelope enough to avoid gaps but shortens a 0.5 s spindle at the edges. A 0.1 s window preserves the duration but produces a noisier envelope. The detector's reported duration depends on this choice, and the AASM 0.5 s minimum is applied to the visual waveform, not to the envelope.

A K-complex detector that requires a negative deflection of at least 0.5 s must decide whether to measure from the negative peak or from the onset. The AASM definition is based on the visual waveform, which includes the negative and positive components. A peak-based detector will report a shorter duration than a scorer.

### Step 8. Localize the event

**Input:** the channel set and the montage.

**Output:** a topographic description and, when possible, a localization.

**Rationale:** localization is what distinguishes EEG from a single-channel state monitor. A focal spike with a phase reversal at C3 points to the left central region. A generalized spike-and-wave discharge points to a bilateral process. A frontal slow wave with no posterior field points to the eyes.

**Method:** in a bipolar montage, find the shared electrode where adjacent channels deflect in opposite directions. In a referential montage, find the electrode with the maximum amplitude and the field distribution. Compare with the contralateral channels.

**Branch point:** if the event is generalized, localization is not meaningful and the report should say so. If the event is focal but the montage does not cover the region, the localization is incomplete.

**Edge case:** a dipole can produce a phase reversal at an electrode that is not the source, because the field depends on the orientation of the dipole. Localization is a hypothesis, not a measurement.

**Localization arithmetic.** In a bipolar chain Fp1-F3, F3-C3, C3-P3, a negative event at C3 produces:

- Fp1-F3: positive or small, depending on the field at Fp1.
- F3-C3: negative-going (F3 minus a more negative C3 gives a positive? The convention is input 1 minus input 2; if C3 is negative, F3-C3 is positive).
- C3-P3: positive (C3 negative minus P3 less negative gives a negative? Sign conventions vary by display).

The sign convention is easy to get wrong. The practical rule is that the two channels sharing the focal electrode deflect in opposite directions, and the shared electrode is the maximum. The direction of the deflection depends on whether the electrode is input 1 or input 2 in each channel.

A referential montage localizes by the maximum amplitude and the field gradient. The electrode with the largest negative deflection is the closest to the source, but volume conduction can make a distant electrode appear large if the source is deep or if the skull has a low-conductivity defect.

### Step 9. Apply the decision rules

**Input:** the features, the transients, the topography, and the context.

**Output:** a stage, an event label, or a state estimate.

**Rationale:** the rules are the decision layer. For sleep, AASM V3 defines the stages. For seizures, the definition is an evolving rhythmic pattern, usually with clinical correlation. For anesthesia, the index is a proprietary transformation of the spectrum.

**AASM V3 rules in brief:**

- Score 30 s epochs in sequence, one stage per epoch; if stages coexist, use the one occupying the greatest portion.
- **W**: more than 50% of the epoch has posterior dominant alpha, eye blinks, reading eye movements, or REMs with normal or high chin tone.
- **N1**: alpha attenuated and replaced by low-amplitude mixed-frequency activity for more than 50% of the epoch, or, in subjects without alpha, the earliest of background slowing, vertex sharp waves, or slow eye movements.
- **N2**: one or more K-complexes or sleep spindles in the first half of the epoch or the last half of the previous epoch, with continuation rules.
- **N3**: at least 20% of the epoch contains slow-wave activity of 0.5 Hz to 2 Hz and more than 75 µV peak-to-peak in frontal derivations.
- **R**: low-amplitude mixed-frequency EEG, low chin EMG for the majority of the epoch, and rapid eye movements.

**Branch point:** if the epoch is transitional, the continuation rules determine the stage. A spindle in the last half of the previous epoch can make the current epoch N2 even without a spindle in the current epoch.

**Edge case:** a subject without a posterior dominant rhythm cannot meet the alpha criterion for W. The rules provide an alternative based on eye blinks and background. An algorithm that requires alpha will misclassify these subjects.

**AASM rules as a decision tree.** The stage assignment is sequential, and the order matters.

```text
if >50% alpha (posterior dominant)            -> W
elif eye blinks or reading movements          -> W
elif REMs with normal/high chin tone          -> W
elif K-complex or spindle (current or prior)  -> N2
elif >=20% slow waves >75 uV frontal          -> N3
elif low chin EMG + REMs + LAMF               -> R
elif alpha attenuated + LAMF                  -> N1
else                                           -> N1 (default in ambiguous epochs)
```

The tree is a simplification. The actual manual has continuation rules, arousal rules, and special cases for body movement, major body movement, and transitions. The simplification is useful for understanding the order of the decisions, not for implementing the manual.

**The greatest-portion rule.** When two stages coexist in an epoch, the stage occupying the greatest portion is assigned, except where a specific rule overrides. A 30 s epoch with 14 s of N2 and 16 s of N3 is N3. A 30 s epoch with 10 s of wake and 20 s of N1 is N1. The rule sounds simple, but it requires segmenting the epoch into sub-epochs, which introduces its own boundary decisions.

### Step 10. Aggregate across epochs and nights

**Input:** epoch-level labels.

**Output:** a hypnogram, stage percentages, and event counts.

**Rationale:** sleep is a sequence, not a bag of epochs. The transition structure, the sleep efficiency, and the arousal index are derived from the sequence. Aggregating epoch probabilities by averaging loses the sequence; aggregating by majority vote loses the transitions.

**Arithmetic:** sleep efficiency is total sleep time divided by time in bed. If a night has 480 min in bed and 420 min asleep, the efficiency is 87.5%. A per-epoch accuracy of 85% can still produce a badly wrong hypnogram if the errors are concentrated at transitions.

**Branch point:** if the decision is a nightly summary, report the summary and its uncertainty. If the decision is an event alert, report the event and its duration. The two aggregations are different.

**Edge case:** a night with a long non-wear period has a denominator problem. Time in bed is not the same as time recorded. Report the coverage.

**Aggregation arithmetic.** A hypnogram with 960 epochs (8 h at 30 s) has 960 stage labels. The stage percentages are counts divided by 960. If 48 epochs are unscorable, the denominator should be 912, not 960. Reporting 5% REM over 960 epochs when 48 epochs are missing understates the REM share if the missing epochs were REM.

The arousal index is the number of arousals per hour of sleep, not per hour in bed. The distinction matters for a night with long wake periods: the index rises if the denominator is sleep time and falls if it is time in bed. State the denominator.

### Step 11. Report with uncertainty and limits

**Input:** the measurements, features, and decisions.

**Output:** a report with values, units, methods, confidence, and limitations.

**Rationale:** EEG reports are used for clinical decisions and for research endpoints. A stage label without the rules, the channels, and the agreement is not reproducible. A seizure count without the annotation protocol is not comparable across studies.

**Contents:** montage and sampling; filter settings; artifact summary; per-stage percentages with the scoring rules; event list with times and channels; agreement with a reference reader if available; and the limits of the channel set.

**Branch point:** if the report is for a consumer product, the claim should be limited to sleep-wake or to a stage subset that the sensor can support. A five-stage claim from a two-channel wearable is not supported by the sensor set.

**Edge case:** an automated report that disagrees with a scorer should be reported as a disagreement, not silently overridden. The disagreement rate is a validation metric.

**A minimal EEG report.** A reproducible report has the following structure:

```text
Record: 19-channel 10-20, referential to linked ears, 250 samples/s
Filters: 0.3 Hz high-pass, 70 Hz low-pass, 50 Hz notch
Quality: 18 of 19 channels usable; T5 high impedance after 02:14
Background: 9 Hz posterior dominant alpha, 25 µV, attenuates with eye opening
Transients: none epileptiform; two vertex sharp waves in N1
Sleep: W 12%, N1 18%, N2 44%, N3 11%, R 15%; 6 arousals
Limits: single night; no prior comparison; T5 excluded after 02:14
```

The structure forces each statement to name its evidence. A report that says "normal EEG" without the montage, the filters, and the limits is not reproducible and is not auditable.

**The report is not the label.** A clinical EEG report describes the background, the transients, the artifacts, and the clinical context. A sleep report describes the hypnogram, the stage percentages, the arousals, and the respiratory events. An automated pipeline that outputs only a stage label has removed the context that makes the label interpretable. If the product is a triage tool, the output should say what was measured, what was uncertain, and what a reader should check.

### Step 12. Validate against the reference

**Input:** the algorithm output and the reference annotations.

**Output:** per-stage sensitivity, specificity, agreement, and confusion matrix.

**Rationale:** sleep staging and seizure detection are classification problems with a reference standard that has its own variability. Inter-rater agreement between sleep scorers is about 80% to 90% for some stage pairs and lower for others. A model cannot be expected to exceed the reference's own reliability by much.

**Metrics:** per-stage sensitivity and positive predictive value; Cohen's kappa or intraclass correlation for stage agreement; event-level sensitivity and false alarms per 24 h for seizure detection; and a confusion matrix that shows which stages are confused.

**Branch point:** if the model is evaluated on epoch-level accuracy, a majority-class baseline can look strong. Report the baseline and the per-stage metrics.

**Edge case:** patient-level splitting is essential. Adjacent epochs from the same night are highly correlated, and a random epoch split leaks the night into the test set.

**The reference is not ground truth.** Two experienced scorers applying AASM V3 to the same night disagree on a meaningful fraction of epochs, especially at transitions and for N1. The disagreement is not a failure of either scorer; it is a property of the rules. A model's agreement with one scorer is an upper bound on its agreement with the construct, and the bound should be reported.

**A validation example.** Suppose a model is evaluated on 100 nights with 960 epochs each, and the reference has 20% N3. A majority-class baseline that predicts N2 for every epoch achieves 44% accuracy if N2 is 44% of epochs, and 0% sensitivity for N3. A model with 85% overall accuracy can still have 40% sensitivity for N3 if the errors concentrate in N3 and in transitions. The overall accuracy hides the clinically important error, which is why per-stage sensitivity and the confusion matrix are required.

### Decision table for artifact handling

| Artifact | Channel scope | Overlaps event | Action |
|---|---|---|---|
| Eye blink | Frontal, bilateral | Yes, if frontal event | Mark; use EOG; exclude from frontal features |
| Lateral eye movement | Anterior temporal, opposite polarity | Yes, if temporal event | Mark; use horizontal EOG |
| Muscle | Frontal/temporal, high frequency | Yes, if beta/gamma feature | Mark; exclude band or segment |
| Line noise | Global, narrowband | No, usually | Notch; re-check spectrum |
| Electrode pop | One electrode | Yes, if focal event | Exclude channel and dependent bipolar channels |
| Sweat | Slow baseline | Yes, if delta feature | High-pass; flag |
| Movement | Multiple channels, broadband | Yes, if event | Exclude segment; use accelerometer |
| ECG | Periodic, variable | Rarely | Mark; remove with template subtraction |
| Reference contamination | Global, same morphology | Yes, if generalized event | Re-reference; exclude reference electrode |

The table is ordered by channel scope, not by severity. A global artifact is easier to detect than a focal one because it appears in every channel, but it is harder to remove without re-referencing. A focal artifact is easy to remove by excluding one channel, but it is easy to mistake for a focal cerebral event. The order of the rows is a reminder that the channel scope determines both the detection method and the correction method. It also determines the cost: excluding one channel is cheap, re-referencing a record is not, and neither is free if the artifact overlaps the event.

A final note on the decision table: it is a guide to the first action, not a substitute for inspecting the signal. Every row can be confirmed or refuted by looking at the raw trace, and the trace is the ultimate arbiter.

## 6. Worked example

### 6.1 The case

A 30 s sleep epoch is recorded at 250 samples/s from a single frontal EEG channel. The subject is a healthy adult. The epoch contains a posterior dominant rhythm at 10 Hz with an amplitude of 15 µV, a 1 Hz delta component of 10 µV, a 6 Hz theta component of 5 µV, a 20 Hz beta component of 5 µV, and broadband noise of 2 µV RMS. The analysis goal is to compute band powers, classify the epoch, and test what happens when a 250 µV blink is added.

### 6.2 Acquisition arithmetic

- Samples: 250 samples/s × 30 s = 7,500.
- Frequency resolution with a 30 s rectangular window: 1/30 = 0.0333 Hz.
- FFT length: the next power of two above 7,500 is 8,192, so the zero-padded transform has bins of 250/8,192 = 0.0305 Hz.
- The zero-padding does not add information; it interpolates the spectrum. The true resolution is still about 0.033 Hz.
- Amplitude resolution: at 16 bits over a ±1 mV range, the LSB is 2,000/65,536 = 0.0305 µV. A 15 µV alpha wave is about 492 codes.

### 6.3 Window and leakage arithmetic

A 30 s epoch at 250 Hz has 7,500 samples. The FFT is zero-padded to 8,192. The Hann window is applied to the 7,500 real samples, and the sum of the window is used to normalize the power.

For a Hann window of length N, the sum is (N-1)/2 = 3,749.5. The window's coherent gain is 0.5, and its equivalent noise bandwidth is 1.5 bins. The normalization 2|X|²/sw² recovers A²/2 for a sine that falls exactly on a bin. For a sine between bins, the Hann window reduces the peak by up to 1.4 dB but the band power is preserved to within a few percent when the band is wider than the window's main lobe.

The alpha band is 5 Hz wide. At 0.0305 Hz per bin, that is 164 bins. The main lobe of the Hann window is about 4 bins wide, so the band contains the full peak and the leakage is small. The delta band is 3.5 Hz wide, about 115 bins, also wide relative to the lobe. This is why band power is more robust than peak frequency: the band integrates over the leakage.

The cumulative power check is a useful diagnostic. Sum the power in the four bands and compare it with the total power from 0.5 Hz to 30 Hz. If the four bands account for less than 80% of the total, there is substantial power outside the bands, which may be line noise, high-frequency artifact, or a band boundary problem.

| Range | Power (µV²) | Share of 0.5-40 Hz |
|---|---:|---:|
| 0.5-4 Hz | 82 | 27% |
| 4-8 Hz | 14 | 5% |
| 8-13 Hz | 182 | 59% |
| 13-30 Hz | 18 | 6% |
| 30-40 Hz | 4 | 1% |
| 49-51 Hz | 0 | 0% |

The table is illustrative and sums to about 98% of the in-band power. The exact values depend on the window and the noise realisation; the point is that a cumulative check catches a missing band or an unexpected peak.

The zero-padding does not change the underlying resolution. It interpolates the spectrum and makes the peak location finer, but the ability to separate two close frequencies is still about 1/T = 0.033 Hz. A 10 Hz alpha and a 10.2 Hz mu rhythm are not separable in a 30 s window; they are separable in a 5 min window.

### 6.4 Band power arithmetic

For a sine of amplitude A, power is A²/2.

| Component | Amplitude (µV) | Power (µV²) | Band |
|---|---:|---:|---|
| Alpha 10 Hz | 15 | 112.5 | 8-13 Hz |
| Delta 1 Hz | 10 | 50.0 | 0.5-4 Hz |
| Theta 6 Hz | 5 | 12.5 | 4-8 Hz |
| Beta 20 Hz | 5 | 12.5 | 13-30 Hz |
| Noise | 2 RMS | 4.0 | broadband |

Total power in the four bands is 112.5 + 50.0 + 12.5 + 12.5 = 187.5 µV², plus the noise contribution within the bands. The measured total in the code is 306 µV², which is higher because the noise is broadband and the window and finite-length effects spread power.

Relative alpha = 112.5/187.5 = 0.60 before noise. The measured value is 0.595, which is consistent.

Relative delta = 50/187.5 = 0.267 before noise; measured 0.269.

The alpha relative power exceeds 0.5, so the naive rule classifies the epoch as wake. The AASM rule also requires the alpha to be posterior dominant; in a single frontal channel, the topography cannot be confirmed. This is a limitation of the one-channel montage, not of the arithmetic.

### 6.5 Transient arithmetic

The N2 epoch contains a 14 Hz spindle of 25 µV for 1.2 s. Its power is 25²/2 = 312.5 µV² during the burst, and the burst occupies 1.2/30 = 4% of the epoch. The spindle band power in the code is 33.9 µV², which is the burst power averaged over the epoch with window leakage.

The N3 epoch contains a 1 Hz delta wave of 100 µV. Its power is 100²/2 = 5,000 µV², and the measured total is 8,268 µV². The slow-wave fraction, computed as the proportion of 0.5 s windows with peak-to-peak above 75 µV, is 1.000.

The N2 epoch has a slow-wave fraction of 0.000, even though its delta relative power is 0.476. The reason is that the delta power comes from low-amplitude background, not from waves above 75 µV. This is the difference between a power ratio and the AASM amplitude criterion.

The arithmetic of the N2 delta power is worth checking. The epoch has a 4 µV, 1 Hz component, whose power is 4²/2 = 8 µV². That is small in absolute terms, but the total power of the N2 epoch is only 167 µV², so the delta share is 0.476. A quiet epoch makes a small absolute delta look dominant. This is a general property of relative power: it amplifies small components when the total is small. A stage rule that uses relative delta without an absolute floor will call a quiet N2 epoch N3. The code's absolute threshold of 500 µV² prevents that, but the threshold is a design choice, not an AASM rule.

### 6.5.1 Comparing the three epochs

| Epoch | Delta rel | Alpha rel | Spindle power (µV²) | Slow fraction | Naive stage | AASM-consistent stage |
|---|---:|---:|---:|---:|---|---|
| Wake | 0.269 | 0.595 | 0.3 | 0.000 | W | W, if alpha is posterior dominant |
| N2 | 0.476 | 0.010 | 33.9 | 0.000 | N2 | N2, if spindle is central and ≥0.5 s |
| N3 | 0.990 | 0.000 | 0.7 | 1.000 | N3 | N3, if slow waves are frontal |
| Wake + blink | 0.448 | 0.449 | 0.3 | 0.025 | N1 | W, if no sleep features |

The table shows that the naive stage and the AASM-consistent stage agree for the three clean epochs and disagree for the blink epoch. The disagreement is the teaching point: the features are the same, and the rule is what changes the conclusion.

The N3 row deserves a second look. The delta relative power is 0.990, which is almost all of the in-band power. The total power is 8,268 µV², which is 27 times the wake total. The absolute amplitude and the relative share agree, which is why N3 is the easiest stage to detect when the signal is clean. The stage that is hardest is N1, because it is defined by the absence of features (no alpha, no spindles, no slow waves) and by the presence of low-amplitude mixed-frequency activity, which is close to noise.

### 6.6 Diagnostic 1: a blink scored as sleep

A 250 µV blink lasting 0.4 s is added to the wake epoch. The blink has most of its energy at about 1.25 Hz, inside the delta band. The measured delta relative power rises from 0.269 to 0.448, and the alpha relative power falls from 0.595 to 0.449. The naive classifier, which uses alpha > 0.5 for wake and delta > 0.2 for N3, no longer sees wake. It scores the epoch as N1 because the delta power is below the absolute threshold used for N3.

The error is a wake epoch scored as light sleep. The consequence in a sleep study is a shortened sleep latency and a distorted hypnogram. In a consumer product, it is a false report of sleep during a period when the user was awake with eyes open.

The diagnostic is the slow-wave duration, not the power ratio. The blink produces a peak-to-peak above 75 µV in only 2.5% of the epoch. The AASM N3 rule requires at least 20%. The duration criterion correctly rejects N3. The frontal-occipital ratio is a second diagnostic: the blink's delta power is 181.9 µV² in the frontal channel and 82.5 µV² in the occipital channel, a ratio of 2.2, whereas a true slow wave is more widespread and the ratio is closer to 1.

### 6.7 Diagnostic 2: line noise hidden by relative power

A 20 µV, 50 Hz tone is added to the wake epoch. The absolute power in the 49 Hz to 51 Hz band is 328.6 µV², which is larger than the alpha power of 112.5 µV². The relative alpha power is unchanged at 0.595 because the band definitions exclude 50 Hz.

A pipeline that reports only relative band power will not see the line noise. A pipeline that reports total power will see it but may attribute it to gamma. The diagnostic is to inspect the full spectrum and to check the power at the mains frequency and its harmonics. At 250 samples/s, Nyquist is 125 Hz, so 50 Hz and 100 Hz are both in band; a 150 Hz harmonic would alias to 100 Hz.

The arithmetic of the alias is worth checking. A 150 Hz harmonic sampled at 250 Hz folds to 250 - 150 = 100 Hz. A 200 Hz harmonic folds to 50 Hz, where it is indistinguishable from the fundamental mains. A 300 Hz harmonic folds to 50 Hz as well. A device with a nonlinear front end can therefore put energy at the mains frequency from harmonics that were never at the mains frequency.

The line-noise check should therefore include a test of whether the 50 Hz peak changes when the subject moves or when the electrode impedance changes. A true mains tone is stable in frequency and amplitude; an alias from a harmonic changes with the source. A notch filter at 50 Hz will remove both, which is acceptable for the decision but hides the distinction in the record.

### 6.8 Varying one assumption

Suppose the epoch is 10 s instead of 30 s. The frequency resolution becomes 0.1 Hz, which is still adequate for band power but too coarse to separate the alpha peak from a nearby mu rhythm. The slow-wave fraction becomes noisier because there are fewer 0.5 s windows. The N3 rule, which requires 20% of the epoch, now requires 2 s of slow waves instead of 6 s; the same 2 s of slow activity is N3 in a 10 s epoch and not N3 in a 30 s epoch.

Suppose the montage is bipolar instead of referential. A frontal slow wave measured between Fp1 and F3 has a different amplitude from the same wave measured at Fp1 against a linked-ear reference, because the bipolar channel subtracts the activity at F3. A slow wave that is broad and synchronous across both electrodes is attenuated; a focal wave at Fp1 is preserved. The amplitude criterion of 75 µV is defined for a specific derivation, and applying it to another derivation changes the stage.

The subtraction arithmetic is simple. If the slow wave has amplitude 100 µV at Fp1 and 80 µV at F3, with the same phase, the referential Fp1 amplitude is 100 µV and the bipolar Fp1-F3 amplitude is 20 µV. The bipolar channel attenuates a widespread wave by 80%. A focal wave with 100 µV at Fp1 and 20 µV at F3 gives a bipolar amplitude of 80 µV, which is close to the referential value. This is why bipolar montages are good for focal events and poor for generalized slow waves.

Suppose the high-pass filter is 1 Hz instead of 0.3 Hz. A 0.5 Hz slow wave is attenuated by the filter. A first-order 1 Hz high-pass has a gain of about 0.45 at 0.5 Hz, so a 100 µV wave becomes 45 µV, below the 75 µV threshold. The N3 epoch is scored as N2 or N1. The filter, not the brain, changed the stage.

Suppose the alpha frequency is 8 Hz instead of 10 Hz. The alpha band starts at 8 Hz, so the peak sits at the boundary. A window with leakage spreads power into theta, and the relative alpha falls. The stage may change from W to N1 without any change in the subject's state. The band boundary is a convention, and the feature depends on it.

The boundary arithmetic is concrete. A 8 Hz sine with a Hann window has its main lobe from about 7.94 Hz to 8.06 Hz, plus side lobes. The alpha band [8, 13) includes the upper half of the main lobe and excludes the lower half. Roughly 50% of the peak power falls below 8 Hz and is counted as theta. A 10 Hz sine is centred well inside the band and loses almost nothing. Two subjects with the same alpha amplitude but different peak frequencies get different relative alpha values, and the difference is an artifact of the band edge.

Suppose the reference is linked ears and the ears are active. A 10 µV alpha rhythm at the ears appears in every channel with the same polarity. A bipolar montage would subtract it; a referential montage would show it as a generalized alpha. The stage rule would see alpha in the frontal channels and might call the epoch wake. The reference, not the cortex, created the pattern. The diagnostic is to compare the referential recording with a bipolar derivation or with an average reference that excludes the ears.

### 6.9 Conclusion of the case

The wake epoch has relative alpha 0.595 and is classified W by the naive rule, consistent with the AASM criterion if the alpha is posterior dominant. The N2 epoch has a spindle and is classified N2. The N3 epoch has a slow-wave fraction of 1.000 and is classified N3.

The two diagnostics show the limits. A blink can push a wake epoch into N1 if the classifier uses band power alone; the slow-wave duration and the frontal-occipital ratio correct it. Line noise can be invisible in relative band power; the full spectrum and the mains check reveal it.

The assumption that matters most is the epoch length, because it changes the denominator of the 20% rule and the frequency resolution. The second most important is the derivation, because the 75 µV threshold is derivation-specific. The third is the high-pass filter, because a 1 Hz cutoff can remove the slow waves that define N3.

A fourth is the reference. The case has no reference because it is single-channel, but every real recording does. A linked-ear reference and an average reference produce different amplitudes and different topographies for the same event. The reference belongs in the checklist alongside the montage.

### 6.10 What the case does not show

The case is a single channel with synthetic signals. It does not show:

- **Multi-channel topography.** A real blink is identified by its frontal field and its EOG correlation. The case approximates the field with a second simulated channel, but it does not model volume conduction.
- **Inter-scorer variability.** The AASM rules are applied by humans with disagreement, especially at transitions and for N1.
- **Pediatric or infant rules.** The case uses adult rules.
- **Medication effects.** Many drugs change the spectrum, especially beta and alpha, and the case has no drug state.
- **Respiratory and arousal events.** A full sleep study includes apneas, hypopneas, and arousals, which change the stage sequence.

Each omission is a reason to validate on real polysomnography with a full montage, not a reason to distrust the arithmetic. The arithmetic is the part that transfers; the thresholds and the rules do not transfer without validation.

### 6.11 Feature definitions used in the case

| Feature | Definition | Window | Unit |
|---|---|---|---|
| Delta power | Sum of power in 0.5-4 Hz | 30 s | µV² |
| Theta power | Sum of power in 4-8 Hz | 30 s | µV² |
| Alpha power | Sum of power in 8-13 Hz | 30 s | µV² |
| Beta power | Sum of power in 13-30 Hz | 30 s | µV² |
| Spindle power | Sum of power in 11-16 Hz | 30 s | µV² |
| Relative alpha | Alpha power / four-band total | 30 s | dimensionless |
| Relative delta | Delta power / four-band total | 30 s | dimensionless |
| Alpha centroid | Power-weighted mean in 8-13 Hz | 30 s | Hz |
| Spectral edge 95 | Frequency below which 95% of 0.5-40 Hz power lies | 30 s | Hz |
| Slow-wave fraction | Fraction of 0.5 s windows with peak-to-peak >75 µV | 30 s | fraction |

The definitions are deliberately explicit about the window and the band set. A feature table without these columns is not comparable across studies, because a relative power computed with a different denominator is a different feature with the same name.

### 6.12 Reproducibility checklist

| Item | Value in this case |
|---|---|
| Sampling rate | 250 samples/s |
| Epoch length | 30 s |
| Window | Hann over 7,500 samples, zero-padded to 8,192 |
| Bands | delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30 Hz |
| Relative power denominator | sum of the four bands |
| Spindle band | 11-16 Hz |
| Slow-wave criterion | 0.5 s windows, peak-to-peak >75 µV |
| Stage rule | naive alpha/delta thresholds, not AASM |
| Channel | single frontal, not a polysomnography montage |

The checklist is the minimum for a reader to reproduce the numbers. It is also the minimum for a reviewer to see that the stage rule is not the AASM rule, which is the most important limitation of the case.

The checklist is also a teaching device: it shows that every number in the case is conditional on a parameter that could have been chosen differently. A reader who reproduces the numbers with a different window or a different band set will get different values and may conclude that the code is wrong. The checklist is the defence: the values are conditional on these choices, and the choices are stated. This is the difference between a reproducible example and an anecdote with numbers.

## 7. Code, query, or calculation

The script computes band power with a Hann window, spectral centroid and edge, a naive stage classifier, a line-noise check, and a slow-wave duration criterion. It uses only the standard library.

```python
import math, cmath, random

FS, DUR = 250.0, 30.0
N = int(FS * DUR)

def fft(x):
    n = len(x)
    if n == 1: return [complex(x[0])]
    if n & (n-1): raise ValueError("power of two required")
    ev, od = fft(x[0::2]), fft(x[1::2]); out = [0j]*n
    for k in range(n//2):
        t = cmath.exp(-2j*math.pi*k/n)*od[k]
        out[k] = ev[k]+t; out[k+n//2] = ev[k]-t
    return out

def hann(n):
    return [0.5 - 0.5*math.cos(2*math.pi*i/(n-1)) for i in range(n)]

def spectrum(x, fs):
    n = 1
    while n < len(x): n *= 2
    w = hann(len(x)); sw = sum(w)
    seg = [x[i]*w[i] for i in range(len(x))] + [0.0]*(n-len(x))
    X = fft(seg); df = fs/n
    p = [0.0]*(n//2)
    for k in range(1, n//2):
        p[k] = 2*abs(X[k])**2/(sw**2)
    return p, df

def band(p, df, lo, hi):
    return sum(p[k] for k in range(1, len(p)) if lo <= k*df < hi)

def band_centroid(p, df, lo, hi):
    num = sum(k*df*p[k] for k in range(1,len(p)) if lo <= k*df < hi)
    den = band(p, df, lo, hi)
    return num/den if den > 0 else 0.0

def spectral_edge(p, df, lo, hi, frac=0.95):
    total = band(p, df, lo, hi); acc = 0.0
    for k in range(1, len(p)):
        f = k*df
        if lo <= f < hi:
            acc += p[k]
            if acc >= frac*total: return f
    return hi

def synth_wake(fs, dur, seed=1):
    rnd = random.Random(seed); n=int(fs*dur); x=[]
    for i in range(n):
        t=i/fs
        x.append(15*math.sin(2*math.pi*10*t) + 10*math.sin(2*math.pi*1*t)
                 + 5*math.sin(2*math.pi*6*t) + 5*math.sin(2*math.pi*20*t)
                 + rnd.gauss(0,2))
    return x

def synth_n2(fs, dur, seed=2):
    rnd = random.Random(seed); n=int(fs*dur); x=[]
    for i in range(n):
        t=i/fs; v = 8*math.sin(2*math.pi*5*t) + 4*math.sin(2*math.pi*1*t) + rnd.gauss(0,2)
        if 10.0 <= t < 11.2:
            v += 25*math.sin(2*math.pi*14*(t-10.0))
        if 20.0 <= t < 20.6:
            v += -80*math.sin(math.pi*(t-20.0)/0.6)
        x.append(v)
    return x

def synth_n3(fs, dur, seed=3):
    rnd = random.Random(seed); n=int(fs*dur); x=[]
    for i in range(n):
        t=i/fs
        x.append(100*math.sin(2*math.pi*1*t) + 10*math.sin(2*math.pi*5*t) + rnd.gauss(0,3))
    return x

def add_blink(x, fs, at=5.0, dur=0.4, amp=250):
    y = list(x)
    for i in range(int(at*fs), min(len(y), int((at+dur)*fs))):
        y[i] += amp*math.sin(math.pi*(i/fs-at)/dur)
    return y

def classify(p, df):
    d = band(p,df,0.5,4); th = band(p,df,4,8); al = band(p,df,8,13); be = band(p,df,13,30)
    tot = d+th+al+be
    rd, ra = d/tot, al/tot
    sp = band(p,df,11,16)
    if ra > 0.5: stage = "W"
    elif rd > 0.2 and d > 500: stage = "N3"
    elif sp > 20: stage = "N2"
    else: stage = "N1"
    return stage, rd, ra, sp, tot

def slow_fraction(x, fs, amp=75.0, win=0.5):
    w = int(win*fs); hits = 0; total = 0
    for i in range(0, len(x)-w, w//2):
        seg = x[i:i+w]
        if max(seg)-min(seg) > amp: hits += 1
        total += 1
    return hits/total

epochs = {"wake": synth_wake(FS,DUR), "n2": synth_n2(FS,DUR), "n3": synth_n3(FS,DUR)}
for name, x in epochs.items():
    p, df = spectrum(x, FS)
    st, rd, ra, sp, tot = classify(p, df)
    print("%s: stage=%s delta_rel=%.3f alpha_rel=%.3f spindle=%.1f total=%.0f"
          % (name, st, rd, ra, sp, tot))
    print("   centroid_alpha=%.1f edge95=%.1f df=%.4f"
          % (band_centroid(p,df,8,13), spectral_edge(p,df,0.5,40), df))

xw = epochs["wake"]
x50 = [v + 20*math.sin(2*math.pi*50*i/FS) for i,v in enumerate(xw)]
p50, df50 = spectrum(x50, FS)
rel50 = band(p50,df50,8,13)/(band(p50,df50,0.5,4)+band(p50,df50,4,8)+band(p50,df50,8,13)+band(p50,df50,13,30))
print("wake+50Hz: alpha_rel=%.3f power_49_51=%.1f" % (rel50, band(p50,df50,49,51)))

xb = add_blink(xw, FS)
pb, dfb = spectrum(xb, FS)
stb, rdb, rab, spb, totb = classify(pb, dfb)
print("wake+blink: stage=%s delta_rel=%.3f alpha_rel=%.3f" % (stb, rdb, rab))

occ = synth_wake(FS, DUR, seed=4)
po, dfo = spectrum(occ, FS)
print("occipital_wake: delta=%.1f alpha_rel=%.3f stage=%s"
      % (band(po,dfo,0.5,4), band(po,dfo,8,13)/(band(po,dfo,0.5,4)+band(po,dfo,4,8)+band(po,dfo,8,13)+band(po,dfo,13,30)),
         classify(po,dfo)[0]))
print("frontal_blink_delta=%.1f occipital_delta=%.1f ratio=%.1f"
      % (band(pb,dfb,0.5,4), band(po,dfo,0.5,4), band(pb,dfb,0.5,4)/band(po,dfo,0.5,4)))
print("slow_frac_wake=%.3f blink=%.3f n3=%.3f"
      % (slow_fraction(xw,FS), slow_fraction(xb,FS), slow_fraction(epochs["n3"],FS)))
```

Expected output:

```text
wake: stage=W delta_rel=0.269 alpha_rel=0.595 spindle=0.3 total=306
   centroid_alpha=10.0 edge95=20.0 df=0.0305
n2: stage=N2 delta_rel=0.476 alpha_rel=0.010 spindle=33.9 total=167
   centroid_alpha=11.7 edge95=14.3 df=0.0305
n3: stage=N3 delta_rel=0.990 alpha_rel=0.000 spindle=0.7 total=8268
   centroid_alpha=10.4 edge95=1.0 df=0.0305
wake+50Hz: alpha_rel=0.595 power_49_51=328.6
wake+blink: stage=N1 delta_rel=0.448 alpha_rel=0.449
occipital_wake: delta=82.5 alpha_rel=0.599 stage=W
frontal_blink_delta=181.9 occipital_delta=82.5 ratio=2.2
slow_frac_wake=0.000 blink=0.025 n3=1.000
```

Four implementation notes.

The FFT length is the next power of two at or above the epoch length, not the largest power of two below it. Truncating to 4,096 samples would use 16.4 s of a 30 s epoch and change every band power. The difference between 4,096 and 8,192 is not cosmetic; it is 45% of the data.

The Hann window reduces spectral leakage but broadens the peak. The alpha peak at 10 Hz spreads over about two bins. The centroid and the band power are computed on the windowed spectrum, so the spreading is already accounted for in the normalization by the window sum squared.

The classifier is deliberately naive. It uses relative power and an absolute delta threshold. It is not the AASM rule, and the output shows where it fails: the blink epoch is scored N1. The slow-wave fraction is the AASM-style correction.

The slow-wave fraction uses 0.5 s windows with 50% overlap and a peak-to-peak threshold of 75 µV. It is an approximation of the AASM rule, which requires 20% of the epoch to contain slow waves in a frontal derivation. The approximation is adequate to separate a 0.4 s blink from continuous slow-wave sleep.

## 8. Data, standards, and regulatory context

EEG standards are less prescriptive than ECG standards. The acquisition and display conventions are professional, the montage nomenclature is a naming standard, and the device standards cover safety and performance rather than diagnostic accuracy.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| International 10-20 system | Electrode placement and names | De facto standard | Comparability across labs |
| AASM Manual V3 (2023) | Sleep stage definitions | Professional; required for accredited sleep centres | 30 s epochs, stage rules, amplitude thresholds |
| ACNS guidelines | EEG terminology, montages, reporting | Professional | Standard montage names and report structure |
| IEC 60601-2-26 | EEG equipment safety and performance | Binding when claimed | Amplifier and safety requirements |
| IEC 60601-2-25 | Diagnostic ECG (adjacent, not EEG) | Binding when claimed | Not applicable to EEG, but often confused |
| TUH EEG Corpus | Clinical EEG data, 250 Hz, 16-bit | Research dataset | Channel labels must be used, not positions |
| CHB-MIT | Pediatric seizure EEG, 256 Hz | Research dataset | Event-level seizure annotations |
| Sleep-EDF | PSG and sleep staging, 100 Hz | Research dataset | Two EEG channels, EOG, EMG |
| HIPAA | De-identification of EEG | Binding in the US | Dates and identifiers must be removed |
| EU MDR | Clinical evaluation | Binding in the EU | Post-market surveillance |

### 8.1 AASM V3 and what it requires

The AASM Manual for the Scoring of Sleep and Associated Events, Version 3, was released in February 2023, and accredited facilities were required to implement it by 31 December 2023. The rules that matter most for a data pipeline are:

- 30 s sequential epochs, one stage per epoch, with the greatest-portion rule when stages coexist.
- Wake requires more than 50% posterior dominant alpha, or eye blinks, reading eye movements, or REMs with normal or high chin tone.
- N1 requires alpha attenuation with low-amplitude mixed-frequency activity for more than 50% of the epoch, or the earliest of background slowing, vertex sharp waves, or slow eye movements.
- N2 requires a K-complex or sleep spindle in the first half of the epoch or the last half of the previous epoch, with continuation rules.
- N3 requires at least 20% of the epoch with 0.5 Hz to 2 Hz activity above 75 µV peak-to-peak in frontal derivations.
- R requires low-amplitude mixed-frequency EEG, low chin EMG for the majority of the epoch, and rapid eye movements.

The rules are professional, not legal. A consumer device is not required to follow them, but if it claims to report sleep stages, the claim is implicitly a claim about the same constructs. The validation must show agreement with the rules as applied by human scorers.

### 8.2 What the standards do not cover

Three gaps matter.

First, the AASM rules assume a full polysomnography montage: EEG, EOG, EMG, and often respiratory channels. A wearable with one EEG channel and no EOG or EMG cannot apply the rules as written. It can approximate some stages, but the approximation must be validated against the full montage.

Second, the rules define stages for adults and have separate pediatric and infant rules. A model trained on adults and applied to children is out of scope unless validated.

Third, the standards do not specify how to handle artifact. A blink that contaminates a frontal channel is a scoring problem, and the scorer uses EOG and topography to resolve it. An automated pipeline must implement an equivalent logic or accept the error.

A fourth gap is the definition of a hypnogram summary. Two sleep studies can report the same stage percentages with different transition structures and different clinical meanings. The standards define the stage of an epoch, not the quality of a night. A product that reports "sleep quality" from stage percentages is making an inference that the standards do not license.

The standards are also silent on the choice of reference for a wearable. A behind-the-ear or mastoid reference is not the same as a linked-ear reference, and the amplitude and topography of every event change with the reference. A product that does not document its reference cannot compare its measurements with a clinical record.

A fifth gap is the treatment of arousals. An arousal is defined by an abrupt shift in EEG frequency lasting at least 3 s, with additional rules for REM. A 3 s arousal inside a 30 s epoch may not change the stage but may matter clinically. A stage-only pipeline loses the arousal, which is often the more actionable event.

### 8.3 Montage naming and interoperability

EEG channel names are not standardised in the way ECG lead names are. The same electrode can appear as T3 or T7, as Fp1-REF or Fp1-A1, and as an average-reference derivation or a linked-ear derivation. The TUH corpus documentation lists more than 40 configurations, and the channel order is not fixed.

A pipeline that reads channels by position will silently swap electrodes when the configuration changes. A pipeline that reads by label still needs a mapping from the label to a canonical position and reference. The mapping should be explicit, versioned, and tested on at least two configurations.

The modified combinatorial nomenclature (MCN) uses T7/T8 for T3/T4 and P7/P8 for T5/T6. A file that mixes the two conventions has duplicate positions under different names. A mapping that does not handle the synonyms will drop or duplicate channels.

### 8.4 Regulatory status of seizure detection

Seizure detection is a clinical claim. A device that detects seizures and alerts a caregiver is a medical device in the US and the EU. The validation must use event-level annotations, patient-level splitting, and a prespecified false-alarm rate. The CHB-MIT database is a common research benchmark, but it is pediatric, medicated-withdrawal, and small; a model that performs well on CHB-MIT has not demonstrated performance on an adult ambulatory population.

## 9. Industry practice and case evidence

### 9.1 Consumer sleep staging

Chinoy et al. compared seven consumer sleep-tracking devices with polysomnography. The devices differed in sensor set and in stage agreement. Devices with only accelerometry estimated sleep-wake and could not stage sleep; devices with PPG added autonomic proxies; devices with EEG could observe cortical features.

The documented finding is that agreement varies by device and by stage, and that some devices perform well for total sleep time and poorly for REM. The inference is that the sensor set bounds the construct: no algorithm recovers a stage that the sensors cannot observe. A product that reports five stages from a two-channel wearable is making a claim that requires validation against the stages, not only against sleep-wake.

### 9.2 The TUH EEG Corpus

The Temple University Hospital EEG Corpus contains clinical EEGs recorded at 250 Hz, 16 bits per sample, with 24 to 36 channels in EDF+ format. The corpus has more than 40 channel configurations because it draws from intensive care, epilepsy monitoring, emergency, and outpatient settings. Channels must be accessed by label, not by position, because the order is not fixed.

The corpus is valuable because it is large and clinical, and difficult because the montages are heterogeneous. A model trained on one configuration and applied to another must harmonise the channels or use a configuration-aware architecture. The documented lesson is that channel heterogeneity is a first-class problem, not a preprocessing detail.

The corpus also illustrates the difference between a clinical dataset and a benchmark. The recordings were made for clinical care, not for a machine-learning task. The annotations reflect clinical questions, not a balanced class design. A model trained on the corpus inherits the clinical referral pattern: the patients were referred for EEG because a clinician suspected something, and the prevalence of abnormalities is correspondingly high. A model deployed in a general population sees a different prior.

### 9.3 CHB-MIT and seizure detection

CHB-MIT contains continuous scalp EEG from 23 pediatric cases, sampled at 256 Hz, with 198 seizures annotated by onset and end. The recordings were made after withdrawal of anti-seizure medication, which increases seizure frequency and changes the signal relative to a treated outpatient.

The dataset is a benchmark, not a validation set. A detector evaluated on CHB-MIT has demonstrated performance on pediatric, medication-withdrawn, in-hospital EEG. The false-alarm rate in an ambulatory adult is a different measurement. The documented lesson is that the population and the setting are part of the claim.

### 9.4 Sleep-EDF and reproducible pipelines

Sleep-EDF provides PSG recordings at 100 Hz with two EEG channels (Fpz-Cz and Pz-Oz), EOG, EMG, and event markers. It is widely used for sleep staging benchmarks because it is small, clean, and has hypnogram labels.

The 100 Hz sampling rate is adequate for the delta, theta, alpha, and beta bands, but it aliases content above 50 Hz. Muscle artifact and line noise at 50 Hz or 60 Hz are at or above Nyquist. A model trained on Sleep-EDF and applied to a 250 Hz clinical record sees a different noise environment.

The dataset's two EEG channels, Fpz-Cz and Pz-Oz, are a frontal and a parietal derivation. They support stage rules that depend on frontal slow waves and occipital alpha, but they do not support a full 10-20 montage or a focal localization. A model trained on these channels learns a two-channel view of sleep, which is closer to a wearable than to a clinical polysomnogram.

The 100 Hz rate also changes the spindle band. A 14 Hz spindle is well below Nyquist, so it is preserved. A 30 Hz beta rhythm is at 0.6 of Nyquist and is preserved with two samples per cycle, which is marginal for amplitude but adequate for power. A 45 Hz gamma rhythm aliases to 55 Hz, outside the analysis band, but it can fold into beta if the sampling rate is lower than stated in a derivative file.

### 9.5 Clinical workflow and over-read

Clinical EEG is read by a neurologist or an epileptologist, often with a technologist's preliminary review. The over-read includes the montage, the artifacts, the background, the transients, and the clinical context. The report is a narrative, not a set of labels.

An automated pipeline that produces labels without the narrative loses the context that the clinician uses. The inference for product teams is that the output should support the reader, not replace the report. A seizure alert is a triage tool; a diagnosis is a clinical judgment.

A practical implication is that the product's output should include the evidence, not only the label. A stage label with the band powers and the slow-wave fraction lets a reader see why the algorithm decided. A label alone forces the reader to trust an opaque decision, which is the opposite of how clinical EEG is read.

### 9.6 Documented fact versus inference

| Statement | Status |
|---|---|
| AASM V3 was released in February 2023 | Documented by AASM |
| AASM V3 uses 30 s epochs | Documented in the manual |
| TUH EEG is sampled at 250 Hz, 16 bits | Documented by Temple University |
| CHB-MIT is sampled at 256 Hz and contains 198 seizures | Documented by PhysioNet |
| Sleep-EDF provides two EEG channels at 100 Hz | Documented by PhysioNet |
| Consumer sleep devices vary by stage agreement | Documented in Chinoy et al. |
| A single frontal channel cannot distinguish a blink from a slow wave | Inference from topography and field spread |
| Relative band power hides line noise | Inference from the definition of the denominator; demonstrated in Section 6 |
| A five-stage claim from a two-channel device is unsupported | Inference from the AASM channel requirements |
| Montage harmonisation improves cross-dataset transfer | Inference from signal theory; magnitude is study-specific |

The first six rows are checkable against the cited sources. The last four are reasoned inferences, and they are the ones a product team should test rather than assume.

The table also shows why the AASM manual version matters. A dataset scored with V2 rules and a model trained on V3 rules are not the same task, even if the stage names are the same. The version should be recorded in the dataset card and in the model card, alongside the sampling rate and the montage.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Blink scored as delta | Frontal delta with no posterior field | Eye movement energy in 0.5-2 Hz | False N3 or false slowing | EOG; topography; slow-wave duration |
| Muscle scored as beta | High-frequency bursts with jaw clench | EMG broadband | False arousal or seizure | EMG channel; spectral shape |
| Line noise scored as gamma | Peak at 50/60 Hz | Mains interference | False high-frequency power | Notch; spectrum check |
| Electrode pop scored as spike | Isolated sharp transient in one channel | Impedance step | False epileptiform event | Channel validity mask |
| Reference contamination | Same morphology in all channels | Active reference | False generalized event | Re-reference; inspect reference |
| Montage mismatch | Amplitudes differ from training | Referential versus bipolar | Model degradation | Harmonise montage; document |
| Epoch truncation | Frequency resolution too coarse | FFT shorter than epoch | Band power bias | Zero-pad to full length |
| Relative power hides artifact | Alpha falls without state change | Artifact dominates denominator | Misclassification | Report absolute and relative |
| Slow-wave power instead of duration | N3 from low-amplitude delta | Power ratio not amplitude criterion | False N3 | Duration above 75 µV |
| Sleep staging without EOG/EMG | REM indistinguishable from N1 | Missing channels | Stage confusion | Limit claim; validate |
| Random epoch split | Near-duplicate epochs in test | Temporal leakage | Inflated accuracy | Patient-level split |
| Stage accuracy without baseline | High accuracy, low kappa | Majority class | Misleading claim | Per-stage metrics; kappa |
| Annotation single-reader | No agreement reported | Reference variability | Unmeasured error | Multiple scorers |
| Seizure count without duration | Many short detections | Fragmentation | Alert fatigue | Event-level merging |
| Reference electrode active | Identical slow activity in all channels | Reference contamination | False generalized slowing | Re-reference; inspect ears |
| Montage synonyms | T3 and T7 both present | Naming mismatch | Duplicate or missing channels | Canonical mapping |
| Channel order assumed | Channels read by index | Heterogeneous configurations | Electrode swap | Read by label |
| Sleep stage from a single channel | REM scored without EOG/EMG | Missing criteria | High stage error | Limit claim; validate |
| Arousal counted from movement | Motion artifact in EEG | Movement mimics arousal | Inflated arousal index | EMG and accelerometer |
| N3 from relative power | Small total power inflates share | No absolute floor | False N3 | Absolute and duration criteria |
| Filter cutoff too high | Slow waves attenuated | 1 Hz high-pass | Under-reported N3 | Use 0.3 Hz or lower |
| Alpha peak at band edge | Relative alpha unstable | 8 Hz boundary | Stage flip without state change | Report peak frequency |
| Sleep summary without coverage | Percentages sum to 100 | Missing epochs excluded silently | Biased stage shares | Report recorded time |
| Arousal missed by stage-only model | Low arousal index | 3 s events not modelled | Under-reported fragmentation | Add arousal detection |
| Claim exceeds recording duration | Normal short EEG | Paroxysmal events missed | False reassurance | State duration and yield |
| Reference not documented | Amplitudes differ from clinical | Unknown derivation | Non-comparable measurements | Record reference |
| Report without evidence | Label-only output | Opaque decision | Reader cannot audit | Include features and rule |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Electrodes are correctly placed | Topography and localization | Misplacement, failed electrode | Impedance; inspection |
| The montage is known and stable | Amplitude and polarity | Montage change mid-record | Header; channel labels |
| The reference is inactive | Referential interpretation | Active reference | Re-reference; compare |
| Artifacts can be identified | Clean features | Unusual artifact, movement | EOG/EMG/accelerometer |
| The epoch length matches the rules | Stage definitions | Arbitrary segmentation | Use 30 s for AASM |
| Alpha is posterior dominant | Wake rule | Frontal alpha, no alpha | Multi-channel; alternative rule |
| The 75 µV threshold applies | N3 rule | Different derivation | State derivation |
| The subject is an adult | Reference ranges | Pediatric, infant | Use age-specific rules |
| The reference scorer is reliable | Validation | Single reader, fatigue | Inter-rater kappa |
| The population matches deployment | Generalisation | Different setting, comorbidity | External validation |
| The sensor set supports the claim | Construct validity | One channel for five stages | Limit claim; validate |
| The recording covers the night | Summary metrics | Non-wear, gaps | Report coverage |
| The alpha peak is inside the band | Relative alpha | 8 Hz boundary | Check peak frequency |
| The derivation is consistent | Amplitude thresholds | Montage change | Record derivation |
| The scorer follows the same manual version | Reference labels | V2 versus V3 rules | Record manual version |
| The stage sequence is modelled | Hypnogram metrics | Independent epoch scoring | Check transition structure |
| The artifact mask is independent of the outcome | Unbiased rejection | Artifact correlated with disease | Compare rejection by group |
| The channel mapping is complete | Cross-dataset training | Unmapped or duplicated labels | Assert every channel maps once |
| The recording is long enough for the question | Event detection | Paroxysmal events missed | Report duration and yield |
| The report includes the evidence | Clinical trust | Label-only output | Unsupported decision | Report features and rules |
| The scoring manual version is recorded | Comparability | V2 versus V3 | Mixed labels | Version the dataset |

## 12. Exercises

### Exercise 1. Montage construction (recall, 25 min)

**Deliverable:** write the eight channels of a left longitudinal bipolar chain and the eight of a right chain, using 10-20 names.

**Acceptance criteria:** correct order from front to back; each channel is a difference of adjacent electrodes; no electrode used twice within a chain.

**Time:** 25 minutes.

### Exercise 2. Band power by hand (application, 30 min)

**Deliverable:** compute the power of a 20 µV alpha sine, a 10 µV delta sine, and a 5 µV beta sine, then the relative alpha if these are the only components.

**Acceptance criteria:** A²/2 used correctly; relative power sums to 1; units in µV².

**Time:** 30 minutes.

### Exercise 3. Artifact identification (application, 40 min)

**Deliverable:** a table of six artifacts with morphology, topography, frequency, and a discriminating test for each.

**Acceptance criteria:** each artifact has a test that does not rely on a single channel; blink and frontal spike distinguished.

**Time:** 40 minutes.

### Exercise 4. AASM staging (application, 45 min)

**Deliverable:** stage four 30 s epochs from described features: alpha 60%, alpha 20% with slow eye movements, a spindle in the last half of the previous epoch, and 25% slow waves above 75 µV.

**Acceptance criteria:** each stage justified by a named rule; continuation rule applied; no stage assigned from a single feature without context.

**Time:** 45 minutes.

### Exercise 5. Blink contamination (critique, 35 min)

**Deliverable:** explain why a 250 µV blink can produce a delta-band power increase and how to distinguish it from N3. Include the arithmetic of the slow-wave fraction.

**Acceptance criteria:** frequency content of the blink estimated; duration criterion computed; frontal-occipital comparison proposed.

**Time:** 35 minutes.

### Exercise 6. Seizure validation design (design, 60 min)

**Deliverable:** a one-page design for a seizure detector validation on CHB-MIT, including patient-level splits, event-level scoring, false alarms per 24 h, and a prespecified alert threshold.

**Acceptance criteria:** no epoch-level leakage; event merging rule; confidence intervals; population limits stated.

**Time:** 60 minutes.

### Exercise 7. Consumer sleep claim (critique, 30 min)

**Deliverable:** a review of a claim that a two-channel wearable reports five sleep stages, naming the sensor limitations and the validation needed.

**Acceptance criteria:** per-stage agreement requested; REM and N1 limitations named; reference standard specified.

**Time:** 30 minutes.

### Exercise 8. Spectral resolution (application, 25 min)

**Deliverable:** compute the frequency resolution for 10 s, 30 s, and 5 min windows, and state which bands each can resolve.

**Acceptance criteria:** 1/T used correctly; delta/theta boundary resolution assessed; zero-padding distinguished from true resolution.

**Time:** 25 minutes.

### Submission standard

Calculations must show the arithmetic and the units. Artifact tables must give a discriminating test that uses more than one channel. Staging exercises must name the AASM rule that justifies each stage, including the continuation rule where it applies. Validation designs must state the split unit, the event definition, and the metric. A stage label without a rule is not a pass, even if the label is correct.

For the design exercise, the acceptance criteria are also the rubric: a design that omits the split unit or the false-alarm rate fails regardless of how sophisticated the model description is. The exercise is about the validation logic, not the architecture.

A common weak submission is a design that validates on CHB-MIT with a random epoch split and reports sensitivity only. The weaknesses are the leakage, the missing false-alarm rate, and the population mismatch. A strong submission addresses all three in one page, which is enough space if the design is clear.

## 13. Self-check answers

**1. What is the 10-20 position of F7?** Anterior temporal, left hemisphere, between Fp1 and T3.

**2. Why is a bipolar montage called double banana?** The two longitudinal chains over each hemisphere, drawn front to back, resemble bananas; there are two chains per hemisphere in the standard display.

**3. What is phase reversal?** Opposite deflections in two adjacent bipolar channels that share an electrode. The shared electrode is the maximum of the field.

**4. Compute the power of a 20 µV sine.** 20²/2 = 200 µV².

**5. What is the relative alpha if alpha is 200 µV², delta 100 µV², theta 50 µV², and beta 50 µV²?** Total 400 µV²; relative alpha = 0.50.

**6. Why is a 30 s epoch used for sleep staging?** It is the AASM scoring unit. It gives 0.033 Hz resolution and enough time for slow waves to occupy 20% of the epoch.

**7. What is the AASM N3 criterion?** At least 20% of the epoch contains 0.5-2 Hz activity above 75 µV peak-to-peak in frontal derivations.

**8. Why can a blink look like delta?** A 0.4 s blink has a fundamental near 1.25 Hz, inside the delta band. Its frontal topography and short duration distinguish it.

**9. What is the slow-wave fraction for a 0.4 s blink in a 30 s epoch?** About 0.4/30 = 1.3%, below the 20% N3 threshold. The measured value in the code is 2.5% because the 0.5 s windows overlap the blink.

**10. Why does relative band power hide line noise?** The denominator is the sum of the four bands, which excludes the mains frequency. The line noise is invisible in the ratio even when its absolute power is large.

**11. Why must the FFT length cover the epoch?** Truncating discards data and changes the band powers. Zero-padding to the next power of two preserves the full epoch and interpolates the spectrum.

**12. What is the difference between a spike and a sharp wave?** A spike is 20-70 ms; a sharp wave is 70-200 ms. Both are negative transients, but the duration distinguishes them.

**13. Why is CHB-MIT not a validation set for an adult ambulatory product?** It is pediatric, recorded after medication withdrawal, and in-hospital. The population and setting differ from the deployment.

**14. What is the difference between epoch accuracy and kappa?** Accuracy counts correct epochs; kappa corrects for agreement expected by chance. A high accuracy with a majority class can have a low kappa.

**15. Why does a single-channel sleep device struggle with REM?** REM requires rapid eye movements and low chin EMG in addition to EEG. A single EEG channel cannot observe the EOG or EMG criteria.

**16. What is the difference between a spike and a sharp wave, and why does it matter?** A spike is 20-70 ms; a sharp wave is 70-200 ms. The distinction affects the epileptiform classification and the detector's duration threshold. A detector tuned to spikes will miss sharp waves.

**17. Why does a 150 Hz harmonic at 250 samples/s appear at 100 Hz?** 150 modulo 250 is 150, which is above Nyquist (125), so the alias is 250 - 150 = 100 Hz. The same arithmetic applies to any harmonic above Nyquist.

**18. What is the difference between a referential and a bipolar montage for a generalized slow wave?** The bipolar channel subtracts the activity at the second electrode, so a widespread synchronous wave is attenuated. The referential channel preserves it. The 75 µV N3 threshold is therefore derivation-specific.

**19. Why does zero-padding not improve true resolution?** Zero-padding interpolates the spectrum but does not add information. The ability to separate two frequencies is set by the observation time, about 1/T, not by the number of bins.

**20. What is the greatest-portion rule?** When two stages coexist in an epoch, the stage occupying the greatest portion is assigned, unless a specific rule overrides. It requires segmenting the epoch, which introduces boundary decisions.

**21. Why is a referential montage vulnerable to reference contamination?** Every channel includes the reference voltage. If the reference electrode is active, its activity appears in all channels, often as a generalized rhythm or slow wave.

**Rubric for Exercise 6 (seizure validation).** A strong design uses patient-level splits, event-level annotations, a false-alarm rate per 24 h, and a prespecified alert rule. A weak design uses random epochs and reports only sensitivity.

**Rubric for Exercise 7 (consumer claim).** A strong review asks for per-stage agreement, a polysomnography reference, and a limitation to sleep-wake if the sensor set is insufficient. A weak review accepts total sleep time as evidence for five stages.

## 14. Interview questions

**1. Why is EEG harder than ECG?**
A strong answer cites the low cortical amplitude relative to artifact, the montage dependence, and the lack of a single universal fiducial. A weak answer says "it is noisier."

**2. Walk me through how you would stage a 30 s epoch.**
A strong answer applies the AASM rules in order: wake, N1, N2, N3, R, with the continuation rules and the slow-wave duration. A weak answer lists frequency bands and guesses.

**3. How do you distinguish a blink from a frontal slow wave?**
A strong answer uses EOG correlation, frontal topography, absence of a posterior field, and duration. A weak answer uses amplitude alone.

**4. What is the effect of a referential versus bipolar montage on amplitude?**
A strong answer explains the subtraction, the common-mode rejection of widespread activity, and the phase reversal. A weak answer says "bipolar is smaller."

**5. How would you validate a sleep staging algorithm?**
A strong answer uses polysomnography, patient-level splits, per-stage sensitivity and kappa, and a confusion matrix. A weak answer reports accuracy only.

**6. Why is relative band power dangerous?**
A strong answer explains the denominator, the artifact sensitivity, and the need to report absolute power and the full spectrum. A weak answer says "it normalizes."

**7. What is a seizure on EEG?**
A strong answer describes evolving rhythmic activity in frequency, amplitude, and topography, usually lasting more than 10 s, with clinical correlation. A weak answer says "spikes."

**8. How do you handle a channel with high impedance?**
A strong answer excludes the channel and any bipolar channel that depends on it, and reports the loss of coverage. A weak answer interpolates it.

**9. What does the AASM 20% rule mean in practice?**
A strong answer computes the duration of slow waves above 75 µV in a frontal derivation and compares it with 6 s in a 30 s epoch. A weak answer uses band power.

**10. How do you avoid leakage in an EEG dataset?**
A strong answer splits by patient or recording, not by epoch, and checks for near-duplicate segments. A weak answer uses a random split.

**11. What can a 4-channel wearable claim?**
A strong answer limits the claim to sleep-wake or to a validated subset, and states the sensor limitations. A weak answer claims five stages.

**12. Why is the alpha band boundary a problem?**
A strong answer explains that 8 Hz sits at the theta-alpha boundary and that leakage and individual variation move the peak across it. A weak answer treats bands as fixed.

**13. How would you harmonise EEG channels across two hospitals?**
A strong answer maps labels to canonical 10-20 positions, handles T3/T7 synonyms, records the reference, and tests the mapping on both configurations before training. A weak answer renames columns to a fixed order.

**14. What is the first diagnostic you would run on an unexpected band-power result?**
A strong answer checks the full spectrum, the mains frequency, and the epoch length. A weak answer adjusts the band boundaries.

**15. When would you reject a channel rather than interpolate it?**
A strong answer rejects when the channel has high impedance, a pop, or a reference problem, because interpolation fabricates a signal and can spread the artifact to neighbours. A weak answer interpolates to preserve the channel count.

**16. How do you compare two sleep studies scored with different manual versions?**
A strong answer maps both to the same version, re-scores a sample, and reports the disagreement attributable to the version. A weak answer compares percentages directly.

## 15. Key terms

- **10-20 system** — standard electrode placement.
- **Alpha rhythm** — 8-13 Hz posterior dominant activity.
- **Arousal** — abrupt EEG frequency shift lasting at least 3 s.
- **Bipolar montage** — channels between adjacent electrodes.
- **Beta** — 13-30 Hz activity.
- **Common average reference** — mean of all electrodes used as reference.
- **Delta** — 0.5-4 Hz activity.
- **Double banana** — longitudinal bipolar montage.
- **Electrode pop** — abrupt impedance change artifact.
- **Epoch** — fixed scoring window.
- **Gamma** — activity above 30 Hz.
- **K-complex** — frontal negative-positive wave of at least 0.5 s.
- **Montage** — ordered set of channels.
- **Mu rhythm** — 8-13 Hz sensorimotor activity.
- **N1** — light sleep stage.
- **N2** — sleep stage with spindles or K-complexes.
- **N3** — slow-wave sleep.
- **Phase reversal** — opposite deflections at a shared electrode.
- **Polysomnography** — multi-channel sleep recording.
- **Referential montage** — channels against a common reference.
- **REM** — rapid eye movement sleep.
- **Slow wave** — 0.5-2 Hz activity above 75 µV.
- **Sleep spindle** — 11-16 Hz train of at least 0.5 s.
- **Spike** — negative transient of 20-70 ms.
- **Sharp wave** — negative transient of 70-200 ms.
- **Theta** — 4-8 Hz activity.
- **TUH EEG Corpus** — clinical EEG dataset at 250 Hz.
- **Wake (W)** — wake stage.
- **Anterior temporal** — F7/F8 electrode region.
- **Coherent gain** — factor by which a window reduces a sine's amplitude.
- **Equivalent noise bandwidth** — window's effective spectral width.
- **Greatest-portion rule** — stage occupying most of an epoch wins.
- **Hypnogram** — stage sequence over a night.
- **MCN** — modified combinatorial nomenclature.
- **Slow-wave fraction** — share of epoch above the amplitude threshold.
- **Volume conduction** — spread of field through tissue.
- **Coherent gain** — amplitude factor of a window.
- **Spectral leakage** — power spreading into neighbouring bins.
- **Absolute power** — band power in µV².
- **Relative power** — band share of a defined total.
- **Reference contamination** — active reference appearing in all channels.
- **Event-level scoring** — scoring whole events rather than epochs.
- **Inverse problem** — inferring sources from scalp fields.
- **Linked ears** — ears joined as a reference.
- **Average reference** — mean of electrodes used as reference.
- **Bipolar chain** — sequence of adjacent electrode differences.
- **Greatest-portion rule** — majority stage within an epoch.
- **Sleep efficiency** — sleep time divided by time in bed.
- **Electrographic seizure** — seizure visible on EEG without clinical signs.
- **Spectrogram** — time-frequency power representation.
- **Spectral edge** — frequency below which a power fraction lies.
- **Slow-wave fraction** — share of epoch above 75 µV.
- **Stage N1** — light sleep with low-amplitude mixed frequency.
- **Stage N2** — sleep with spindles or K-complexes.
- **Stage N3** — slow-wave sleep.
- **Stage R** — REM sleep.
- **Stage W** — wake.

## 16. Sources

All URLs accessed 25 October 2026.

1. Troester MM, Quan SF, Berry RB, et al. The AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and Technical Specifications. Version 3. American Academy of Sleep Medicine; 2023. https://aasm.org/clinical-resources/scoring-manual/
2. American Academy of Sleep Medicine. Scoring manual resources. https://aasm.org/clinical-resources/scoring-manual/
3. Temple University. TUH EEG Corpus overview. https://isip.piconepress.com/projects/tuh_eeg/html/overview.shtml
4. Ferrell S, et al. The Temple University Hospital EEG Corpus: Electrode Location and Channel Labels. 2020. https://par.nsf.gov/biblio/10199699
5. PhysioNet. CHB-MIT Scalp EEG Database v1.0.0. https://physionet.org/content/chbmit/1.0.0/
6. Goldberger AL, Amaral LA, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*. 2000;101:e215-e220. https://doi.org/10.1161/01.CIR.101.23.e215
7. PhysioNet. Sleep-EDF Database Expanded. https://physionet.org/content/sleep-edfx/
8. Beniczky S, Schomer DL. Electroencephalography: basic biophysical and technological aspects important for clinical applications. *Epileptic Disorders*. 2020. https://doi.org/10.1684/epd.2020.1217
9. Britton JW, Frey LC, Hopp JL, et al. *Electroencephalography: An Introductory Text and Atlas*. American Epilepsy Society; 2016. https://www.ncbi.nlm.nih.gov/books/NBK390358/
10. Learning EEG. Montages and technical. https://www.learningeeg.com/montages
11. Learning EEG. Artifacts. https://www.learningeeg.com/artifacts
12. IEC 60601-2-26, Electroencephalographs. https://webstore.iec.ch/
13. Chinoy ED, Cuellar JA, Huwa KE, et al. Performance of seven consumer sleep-tracking devices compared with polysomnography. *Sleep*. 2021;44(5):zsaa291. https://doi.org/10.1093/sleep/zsaa291
14. American Clinical Neurophysiology Society. Guidelines for standard EEG. https://www.acns.org/
15. IEC 60601-2-26:2012, Electroencephalographs. https://webstore.iec.ch/
16. PhysioNet. Sleep-EDF Database Expanded documentation. https://physionet.org/content/sleep-edfx/
17. Berry RB, et al. AASM scoring rules and transition rules, V3 summary. https://aasm.org/clinical-resources/scoring-manual/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_ecg.md`](02_ecg.md)
- Next topic: [`04_ppg-and-wearables.md`](04_ppg-and-wearables.md)
- Related topic: [`06_feature-engineering.md`](06_feature-engineering.md)
- Module survey file: [`../../course/11_physiological_signals_wearables.md`](../../course/11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/)
- Related topic: [`07_deep-learning-for-signals.md`](07_deep-learning-for-signals.md)
- Dataset: [`https://physionet.org/content/chbmit/1.0.0/`](https://physionet.org/content/chbmit/1.0.0/)
