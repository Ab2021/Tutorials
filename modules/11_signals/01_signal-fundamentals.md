# Signal Fundamentals: Sampling, Aliasing, Filtering, and the Signal Chain

> Module 11 — Physiological Signals and Wearables
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: Module 01 (probability and data), Module 06 (machine learning basics); no earlier topic in this module
> Version 2026.3

## 1. Why this topic exists

A physiological signal arrives as a column of numbers. The numbers alone are not a measurement. They become a measurement only when the sampling rate, the amplitude scale, the filter settings, the sensor site, and the time base are known and correct. Most defects that later look like model failures are introduced here, in the first metre of the pipeline, and they are cheap to prevent and expensive to discover.

Three failures recur in industry work.

The first is **aliasing**. A wearable samples at 100 Hz to save power. Mains interference at 60 Hz has content above the 50 Hz Nyquist limit, so it folds to 40 Hz. A 40 Hz peak in the spectrum looks like muscle artifact, not like line noise, so the team designs a 40 Hz low-pass filter, removes real signal content, and leaves the interference in place. Nothing in the time-domain plot reveals the mistake. Only the sampling metadata and a frequency-domain check do.

The second is **filtering that changes the measurand**. A high-pass filter at 0.5 Hz removes baseline wander effectively. It also attenuates the low-frequency content of the ST segment and the T wave. A pipeline built for rhythm detection is then reused for ischemia analysis, and the ST deviation it reports is a property of the filter, not of the heart. The filter was correct for one decision and wrong for another.

The third is **quality that becomes a label**. A photoplethysmography (PPG) segment with motion artifact is labelled "irregular rhythm" because the peak detector fired on motion spikes. The model learns the artifact, the validation set contains the same artifact, and the reported performance is good until the device is used while walking.

Each of these is a signal-chain problem, not a modelling problem. The teams that avoid them share a practice: they treat acquisition parameters as part of the dataset, they test the chain on known inputs, and they keep the measurement, the feature, the biomarker, and the decision separate.

A fourth failure is less visible than the other three: **a chain that cannot be described**. The pipeline works, the model performs, and no one can say which filter produced the training data. The inability to describe the chain becomes a problem at the first audit, the first device change, or the first attempt to reproduce the result. It is a failure of documentation, but it behaves like a technical failure because it blocks every subsequent correction.

That separation is the spine of this module:

- A **measurement** is a quantity with a unit, an uncertainty, and a defined acquisition procedure. Heart rate in beats per minute is a measurement.
- A **feature** is a computed number or vector derived from the measurement or the waveform. The standard deviation of RR intervals is a feature.
- A **biomarker** is a feature or measurement with a demonstrated relationship to a biological process, disease state, or treatment response. A resting heart rate above a threshold associated with incident cardiovascular events can be a biomarker if the association is validated.
- A **decision** is an action taken under uncertainty: notify, refer, escalate, do nothing.

A sampling error corrupts the measurement. A feature computed on a corrupted measurement is corrupted. A biomarker built on that feature inherits the corruption. The decision inherits it last and pays for it most.

This topic gives you the vocabulary and the procedure to keep that chain honest for any modality. Later topics apply the same chain to electrocardiography (ECG), electroencephalography (EEG), PPG, respiration, and continuous glucose monitoring (CGM).

### 1.1 What a chain defect costs

A defect at the acquisition stage is not recovered by a better model. It is recovered by re-collection, which is the most expensive operation in a wearable study. Participants have to be re-recruited, devices re-shipped, consent re-confirmed, and the calendar re-planned. In a regulated submission, the cost is larger: a premarket review that finds the signal chain undocumented may result in a deficiency letter and a new review cycle.

The costs fall into four buckets.

- **Re-collection cost.** The raw data cannot support the intended measurement. No amount of processing creates information that was never sampled.
- **Analysis cost.** The team spends weeks building features on a corrupted signal, then discovers the corruption during a holdout evaluation. The model work is discarded with the data.
- **Decision cost.** A clinical or product decision is made on a measurement that is biased in a subgroup. The bias is discovered after deployment, when the cost is reputational and regulatory rather than analytical.
- **Reproducibility cost.** A published result cannot be reproduced because the filter chain, the sampling rate, or the epoch definition is missing. The result is not necessarily wrong, but it cannot be checked, and a check is what a regulator or a payer asks for.

A useful test for whether a chain is documented: could a competent engineer who has never spoken to you reproduce your heart-rate number from the raw file and your written methods? If the answer requires a question, the chain is under-specified.

### 1.2 The composite failure

Consider a composite example, assembled from common patterns rather than a single named product. A team builds an atrial fibrillation (AF) detector on a smartwatch. The training data are 30 s single-lead ECG strips recorded in a clinic at 500 Hz. The deployment data are 30 s strips recorded at 200 Hz on a device that low-pass filters at 40 Hz. The model uses RR-interval features and a measure of P-wave energy. At 200 Hz with a 40 Hz low-pass, P-wave energy is attenuated; at 500 Hz it is preserved. The model learned a feature that is partly a property of the acquisition chain. Performance drops at deployment, and the team's first hypothesis is "domain shift in the population." The actual mechanism is a sampling and filtering mismatch. The fix is to resample and re-filter the training data to the deployment chain, or to train on both chains and treat the chain as a domain variable.

The example is inference, not a published case, but the pattern is documented in cross-dataset ECG studies: models trained on one sampling rate and bandwidth do not automatically transfer to another.

### 1.3 What this topic is not

This topic is not a digital signal processing textbook. It does not derive the z-transform, and it does not cover adaptive filters, wavelets, or machine-learning representations; those belong to Topics 06 and 07. It covers the subset of signal processing that a data scientist must be able to specify, audit, and defend in a product or regulatory context. The mathematics is kept to what can be computed with a calculator or the standard-library script in Section 7.

The module's industry question is whether a signal-derived measurement means the same thing in the lab, on the wrist, and in the home. This topic supplies the first half of the answer: the conditions under which two measurements are comparable at all. If the sampling rates, bandwidths, and quality rules differ, the measurements are not the same quantity, and no statistical adjustment makes them the same. Comparability is established at acquisition, not at analysis.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** the Nyquist frequency for a given sampling rate and predict the alias frequency of a sinusoid above Nyquist, including the fold for multi-frequency records.
2. **Choose** an anti-alias filter order and cutoff for a stated signal bandwidth, and justify the margin between the cutoff and Nyquist.
3. **Calculate** the least significant bit (LSB), the number of codes occupied by a signal of interest, and the theoretical signal-to-quantization-noise ratio for an analog-to-digital converter (ADC).
4. **Design** a preprocessing chain of high-pass, notch, and low-pass filters for a named decision, stating each cutoff, order, and phase response.
5. **Distinguish** clipping, saturation, lead-off, baseline wander, powerline interference, motion artifact, dropped samples, clock drift, and non-wear from the waveform and from metadata.
6. **Explain** why a causal filter shifts fiducial points and why zero-phase filtering is preferred for interval measurement but not for real-time detection.
7. **Audit** a dataset for missing acquisition metadata and write the minimum metadata contract needed to reproduce a measurement.
8. **Critique** a published preprocessing description for reproducibility, naming the parameters that are absent.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

A note on the level of each outcome. Outcomes 1 through 4 are computational: you should be able to produce a number and show the arithmetic. Outcomes 5 and 6 are diagnostic: you should be able to look at a waveform or a methods paragraph and name the defect. Outcomes 7 and 8 are professional: you should be able to write a specification that another engineer can implement without asking you what you meant.

The mapping is deliberate. A course that only teaches computation produces analysts who can run a filter but cannot tell whether the filter is appropriate. A course that only teaches critique produces reviewers who cannot build. The exercises alternate between the two.

The outcomes are written so that each can be assessed with an artefact rather than a quiz. "Compute" is assessed by an arithmetic table. "Design" is assessed by a specification with justified parameters. "Critique" is assessed by a written review of a methods section. "Audit" is assessed by a checklist applied to a real dataset. This matters for the assignment at the end of the module, which asks for a validation plan rather than an examination answer.

## 3. The core mental model

The signal chain is a sequence of conversions, and each conversion destroys information that cannot be recovered later.

```text
 physical      analog front end        ADC                 conditioning           analysis
 quantity  ->  gain, filter, buffer -> sample + quantize -> digital filters    -> features
                                                          -> quality index      -> measurement
                                                                                -> biomarker
                                                                                -> decision

   Pa     ->      mV                 ->    codes           ->    mV (filtered)  ->    bpm, ms, %
   continuous                        ->    discrete time   ->    discrete time
   continuous amplitude              ->    discrete amplitude
```

Redraw it from memory with four labels on every arrow: what is the input, what is the output, what is the unit, and what information is lost.

### 3.1 The three budgets

Every acquisition makes three budgets, and every downstream analysis depends on them.

**Bandwidth budget.** The signal of interest occupies a band. ECG diagnostic content runs from about 0.05 Hz to 150 Hz in adults. PPG heart-rate content runs from about 0.5 Hz to 10 Hz. Respiration runs from about 0.1 Hz to 3 Hz. The sampling rate must exceed twice the highest frequency that can reach the converter. The analog anti-alias filter enforces that limit before sampling; the digital system cannot enforce it afterwards.

**Amplitude budget.** The ADC maps a voltage range onto a finite set of integer codes. A 16-bit converter over a 10 mV peak-to-peak range has 65,536 codes and an LSB of about 0.153 µV. A 1 mV QRS complex occupies about 6,554 codes. A 12-bit converter over the same range has an LSB of about 2.44 µV, and the same QRS occupies 410 codes. Both may be adequate; the question is the size of the smallest feature you must resolve, not the bit count in isolation.

**Time budget.** Interval measurements depend on the clock. A 50 parts-per-million (ppm) clock error accumulates 0.5 ms over 10 s and 4.32 s over 24 h. For heart-rate variability (HRV) on 10 s windows the error is negligible. For beat-to-beat intervals over a night, or for aligning two devices, it is not. Jitter, the short-term variation in sample timing, matters more than slow drift for spectral features.

### 3.2 The frequency axis and folding

Aliasing is easiest to see on a frequency axis. Sampling at fs creates copies of the spectrum centred at multiples of fs. Any content above fs/2 folds down.

```text
 true frequency:     0        40       50       60       80      100  Hz
                     |--------|--------|--------|--------|--------|
 Nyquist at fs=100:                    ^
 60 Hz component folds to 100-60 = 40 Hz
 80 Hz component folds to 100-80 = 20 Hz
 150 Hz at fs=250 folds to 250-150 = 100 Hz
```

The fold rule for a positive frequency f sampled at fs is: reduce f modulo fs, then if the result exceeds fs/2, subtract it from fs. A 60 Hz tone at 100 Hz sampling becomes 40 Hz. A 150 Hz tone at 250 Hz sampling becomes 100 Hz. The amplitude is preserved; only the apparent frequency changes. That is why aliasing is invisible in the time domain when the source is broadband noise and obvious only when the source is a pure tone.

### 3.3 Filter order and what it buys

A first-order low-pass attenuates at 6 dB per octave, a second-order at 12 dB per octave, a fourth-order at 24 dB per octave. The order determines how much of the stopband survives. It also determines phase behaviour and ringing.

For a diagnostic ECG, the AHA/ACCF/HRS 2007 statement calls for a low-frequency cutoff of 0.05 Hz and a high-frequency cutoff of at least 150 Hz for adults, with a sampling rate of at least 500 samples/s for storage. Those numbers are not arbitrary. The 0.05 Hz cutoff keeps the ST segment accurate; a 0.5 Hz cutoff is acceptable for rhythm monitoring but distorts the ST segment. The 150 Hz cutoff preserves QRS notching and pacemaker spikes that a 40 Hz monitor bandwidth removes.

The practical rule: choose the filter from the decision, not from the device default. A rhythm algorithm and an ST-segment algorithm need different chains, and a chain tuned for one must not be silently reused for the other.

### 3.4 Why order of operations matters

Four ordering rules cover most pipeline defects.

1. Anti-alias before sampling. A digital filter cannot remove content that has already folded.
2. Filter the continuous record before segmentation. Filtering each epoch independently creates edge transients at every boundary and makes the result depend on where you cut.
3. Detect quality before computing features. A feature computed on a clipped or lead-off segment is not a noisy version of the truth; it is a different number.
4. Keep raw data. Store the unfiltered signal alongside the filtered version so that a later decision can reprocess without re-collecting.

### 3.5 The uncertainty chain

Every stage adds uncertainty, and uncertainties do not cancel.

```text
 sensor noise        ->  voltage uncertainty
 quantization        ->  + LSB/sqrt(12) RMS
 anti-alias filter   ->  amplitude and phase distortion
 motion artifact     ->  non-stationary additive noise
 peak detection      ->  fiducial timing error
 interval calculation->  propagated timing error
```

A heart rate derived from 10 s of data has a resolution of about 6 bpm if you count beats, because one beat more or less changes the count by one. A heart rate derived from the mean RR interval of 11 beats has a much finer resolution but is sensitive to a single missed or extra peak. The uncertainty depends on the estimator, not only on the sensor.

## 4. Definitions and vocabulary

The table is the working vocabulary for the rest of the module. Where a term has a unit, the unit is given; where it is a label, the coding system or convention is given. Terms are listed roughly in the order they appear in the chain, from acquisition to decision.

Three conventions are used throughout.

- Frequencies are in hertz (Hz). A sampling rate is also in hertz but is written as samples per second where the distinction matters.
- Amplitudes are in millivolts (mV) or microvolts (µV) for electrical signals, and in arbitrary units or percentage for optical signals. An optical signal reported in millivolts is a front-end voltage, not an oxygen saturation.
- Time intervals are in seconds (s) or milliseconds (ms). A heart rate is in beats per minute (bpm), which is not an SI unit but is the clinical convention.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Signal | A time-varying physical quantity transduced into a measurable form | Calling any data column a signal | mV, µV, arbitrary units |
| Sampling rate (fs) | Number of samples per second | Confusing it with the signal bandwidth | Hz (samples/s) |
| Sampling interval | Time between consecutive samples, 1/fs | Reporting it in seconds when fs is in Hz without conversion | s or ms |
| Nyquist frequency | Half the sampling rate | Calling it the maximum measurable frequency without noting the anti-alias requirement | Hz |
| Nyquist theorem | A band-limited signal is reconstructible if sampled above twice its highest frequency | Applying it to a signal that was never band-limited | dimensionless |
| Aliasing | Folding of content above Nyquist into lower frequencies | Calling any noise "aliasing" | Hz (alias frequency) |
| Anti-alias filter | Analog low-pass filter before the ADC | Assuming a digital low-pass can substitute | cutoff in Hz, order |
| Quantization | Mapping a continuous amplitude to a finite code set | Confusing it with sampling in time | codes |
| LSB | Voltage step represented by one code | Reporting it without the full-scale range | µV/code |
| Full-scale range | Voltage span the ADC can encode | Confusing differential and single-ended range | mV peak-to-peak |
| Dynamic range | Ratio of largest to smallest representable signal | Equating it with bit count alone | dB |
| Signal-to-noise ratio (SNR) | Signal power divided by noise power | Reporting SNR without a bandwidth | dB |
| Effective number of bits (ENOB) | Resolution actually achieved after noise and distortion | Quoting nominal bits as ENOB | bits |
| Baseline wander | Low-frequency drift from respiration, perspiration, or electrode motion | Treating it as a physiological wave | Hz, mV |
| Powerline interference | Narrowband noise at 50 Hz or 60 Hz and harmonics | Calling it "60 Hz noise" when the local mains is 50 Hz | Hz |
| Notch filter | Narrow band-stop filter at a specified frequency | Using a notch wider than needed and deleting signal | centre Hz, Q |
| High-pass filter | Passes frequencies above a cutoff | Using 0.5 Hz for ST-segment analysis | cutoff Hz, order |
| Low-pass filter | Passes frequencies below a cutoff | Using 40 Hz for diagnostic ECG | cutoff Hz, order |
| Band-pass filter | Passes a band between two cutoffs | Describing only one cutoff | low/high Hz, order |
| Cutoff frequency | Frequency at which gain is conventionally -3 dB | Calling it the edge of the passband with no definition | Hz |
| Filter order | Number of poles or equivalent | Reporting order without filter family | integer |
| Roll-off | Stopband attenuation slope | Quoting dB/decade when the reader expects dB/octave | dB/octave or dB/decade |
| Phase response | Frequency-dependent delay introduced by a filter | Ignoring it when measuring intervals | degrees or ms |
| Group delay | Delay of the envelope at a frequency | Assuming it is constant across the band | ms |
| Zero-phase filtering | Forward-backward filtering that cancels phase distortion | Using it in real time | dimensionless |
| Settling time | Time for a filter to recover from a step or start transient | Analysing the first seconds of a filtered record | s |
| Clipping | Waveform touching the ADC or amplifier rail | Confusing it with a physiological plateau | codes or mV |
| Saturation | Amplifier output limited by its supply | Treating saturation as signal | mV |
| Lead-off | Electrode disconnected or high impedance | Reporting lead-off as asystole | boolean or impedance in kΩ |
| Motion artifact | Additive and multiplicative noise from movement | Labelling it as arrhythmia | varies |
| Non-wear | Device not on the body | Treating non-wear as zero activity | minutes or % of day |
| Dropped sample | Missing sample in an otherwise regular stream | Interpolating without recording the gap | count, index |
| Clock drift | Slow divergence of the device clock from reference time | Ignoring it across days | ppm, s/day |
| Jitter | Short-term sample-timing variation | Confusing it with drift | ms RMS |
| Epoch | Fixed analysis window | Choosing an epoch without stating its length | s |
| Window function | Taper applied before spectral analysis | Using a rectangular window and ignoring leakage | named function |
| Spectral leakage | Energy from one frequency appearing in neighbouring bins | Calling it a real spectral peak | dB |
| DFT bin | Frequency cell of a discrete Fourier transform | Reporting a frequency finer than the bin | Hz |
| Decimation | Reducing the sampling rate after low-pass filtering | Decimating without an anti-alias filter | integer factor |
| Interpolation | Estimating samples between known samples | Using it to fill long gaps | method |
| Resampling | Changing the sampling rate | Resampling without documenting the method | Hz |
| Quality index | A scalar summarising usability of a segment | Using it as a diagnosis | dimensionless |
| Artifact | Non-physiological content in the signal | Treating all artifact as noise to delete | varies |
| Measurement | Quantity with unit and uncertainty | Reporting a feature as a measurement | unit |
| Feature | Computed descriptor of the signal | Calling every feature a biomarker | unit or dimensionless |
| Biomarker | Feature with validated biological meaning | Claiming validation from correlation alone | unit |
| Decision | Action under uncertainty | Presenting a decision as a measurement | categorical |
| Fiducial point | A landmark on the waveform, such as the R-peak | Assuming it is a physical instant rather than a detector output | sample index or s |
| Refractory period | Minimum time between accepted detections | Setting it longer than the shortest physiological interval | ms |
| Guard band | Samples discarded at an epoch edge | Discarding signal instead of filtering continuously | s |
| Settling | Recovery of a filter from its initial state | Reporting the first epoch of a record as valid | s |
| Dynamic range | Ratio of maximum to minimum representable amplitude | Quoting bit depth as dynamic range | dB |
| Passband ripple | Variation of gain within the passband | Ignoring it when amplitudes are compared | dB |
| Stopband attenuation | Gain reduction beyond the transition band | Quoting the cutoff as if it were the stopband | dB |
| Time base | Reference against which sample times are defined | Using sample index as if it were wall-clock time | s, ISO 8601 |
| Epoch aggregation | Rule combining epoch-level results into a record-level result | Averaging probabilities without stating the rule | mean, median, max |
| Measurement uncertainty | Range within which the true value is expected | Reporting a standard deviation of features as uncertainty | unit, confidence level |

Two further distinctions are used repeatedly and are worth stating outside the table.

A **parameter** is a number that defines the acquisition or processing, such as a cutoff frequency. A **hyperparameter** is a number chosen by the model developer, such as a window length in a convolutional network. The two are often confused because both are "settings." The difference is that a parameter changes the measurement, while a hyperparameter changes the model's fit to the measurement. Changing a filter cutoff can invalidate a regulatory claim; changing a learning rate usually cannot.

A **label** is the reference value used for supervision, such as "AF present" from adjudication. A **prediction** is the model output. A **decision** is what the product does with the prediction. A model can have good discrimination and still produce a bad decision if the threshold is wrong for the population. This topic stops at the measurement; Topic 08 and Topic 09 carry the distinction into biomarkers and validation.

## 5. The mechanics, step by step

The procedure below is modality-agnostic. It is written for a single-channel physiological signal with a known sensor, and it scales to multi-channel records by repeating steps 3 through 10 per channel while sharing the time base.

### Step 1. Write the measurement specification

**Input:** the decision the data must support, the clinical or behavioural question, and the setting (clinic, home, ambulatory, sleep).

**Output:** a one-page specification containing the measurand, the bandwidth of interest, the amplitude range, the required time resolution, the required duration, the acceptable latency, and the reference standard.

**Rationale:** every later parameter is derived from this page. Without it, filter cutoffs are copied from a previous project and sampling rates are chosen by the chip vendor. The specification is also the document a regulator or a reviewer will ask for when the algorithm's inputs are questioned.

**Branch point:** if the decision is rhythm detection, the bandwidth of interest is roughly 0.5 Hz to 40 Hz and latency may be seconds. If the decision is ST-segment analysis, the bandwidth extends down to 0.05 Hz and the allowable distortion is much smaller. If the decision is sleep staging, the epoch is 30 s and the bandwidth depends on the channel. The same device can support several decisions only with several documented chains.

**Edge case:** a specification that says "heart rate" without saying whether it is instantaneous, averaged over 10 s, or averaged over a minute is incomplete. Three different measurements share the name.

### Step 2. Select the sensor and analog front end

**Input:** the specification from Step 1.

**Output:** a sensor type, a front-end gain, an input impedance, a common-mode rejection ratio (CMRR), and a noise floor.

**Rationale:** the front end sets the achievable SNR before any digital processing. For ECG, electrode-skin impedance imbalances convert common-mode mains voltage into differential noise; a high CMRR and high input impedance reduce that conversion. For PPG, the front end sets the LED current, photodiode gain, and ambient-light rejection, which determine whether the pulsatile component is above the noise floor at low perfusion.

**Branch point:** a diagnostic ECG uses a high-resolution front end with a wide bandwidth. A wrist PPG uses a highly integrated optical front end with aggressive power cycling. The same digital algorithm cannot assume the same noise characteristics.

**Edge case:** a front end with automatic gain control changes the amplitude scale mid-record. If the gain changes are not logged, the signal is non-stationary in a way that no filter can fix.

**Verification:** measure the input-referred noise with the inputs shorted and with a known resistor across the inputs. Compare the result with the datasheet and with the quantization noise from Step 4. If the front-end noise is much larger than one LSB, additional bits do not improve the measurement.

### Step 3. Set the anti-alias filter and sampling rate

**Input:** the highest frequency of interest f_max, the desired stopband attenuation, and the available power and silicon.

**Output:** an analog low-pass cutoff f_c, a filter order n, a sampling rate fs, and a documented margin.

**Rationale:** the sampling rate must exceed 2 × f_max. The margin between f_max and fs/2 is occupied by the transition band of the anti-alias filter and by any out-of-band content that would otherwise fold. A common choice is fs = 4 × f_max to 5 × f_max, with an analog filter that reaches the required attenuation by fs/2.

**Arithmetic:** for ECG with f_max = 150 Hz, fs = 500 Hz gives Nyquist = 250 Hz, a margin of 100 Hz. A fourth-order Butterworth low-pass at 200 Hz reaches about 24 dB of attenuation one octave above cutoff; at 250 Hz the attenuation is finite, so residual content between 200 Hz and 250 Hz folds into 250 Hz to 200 Hz. The filter must be specified against the actual out-of-band environment, not only against the signal.

**Branch point:** if the device must reject 50 Hz or 60 Hz mains, the anti-alias filter can be placed to attenuate it, or the notch can be digital. A digital notch cannot help if the mains frequency is above Nyquist. At fs = 100 Hz, a 60 Hz notch is impossible because 60 Hz has already folded to 40 Hz.

**Edge case:** decimation inside the device. A front end may sample at 1,000 Hz and decimate to 250 Hz. The decimation filter must be a proper anti-alias low-pass at the new Nyquist, not a simple sample drop.

**Transition-band arithmetic.** A fourth-order Butterworth low-pass with a -3 dB cutoff at 200 Hz has a magnitude response |H(f)| = 1/√(1 + (f/200)^8). At 250 Hz the gain is 1/√(1 + 1.25^8) = 1/√(1 + 5.96) = 0.379, or -8.4 dB. At 300 Hz it is 1/√(1 + 1.5^8) = 1/√(1 + 25.6) = 0.194, or -14.2 dB. If the out-of-band interferer at 260 Hz is 1 mV and the signal of interest is 50 µV, an -8.4 dB attenuation leaves 379 µV of alias, which is still seven times the signal. The filter order or the sampling rate must change. This kind of arithmetic is what "choose an anti-alias filter" means in practice; the cutoff alone is not a design.

**Oversampling as a strategy.** Sampling at 4 kHz and decimating to 500 Hz relaxes the analog filter: the transition band is now 250 Hz to 2,000 Hz, which a second-order filter can handle. The cost is power and data rate in the front end. This is why many biopotential front ends oversample and decimate rather than relying on a steep analog filter.

### Step 4. Set ADC resolution and range

**Input:** the expected signal amplitude, the largest expected artifact or offset, and the smallest feature to resolve.

**Output:** bit depth, full-scale range, gain, and the resulting LSB.

**Rationale:** the range must accommodate the largest signal without clipping, including electrode offset and motion. The resolution must place the smallest feature well above one LSB. A 1 mV ECG on a 10 mV range at 16 bits uses 6,554 codes; at 12 bits it uses 410 codes. Both can support R-peak detection, but only the 16-bit configuration supports fine ST-segment measurement with margin.

**Arithmetic:** LSB = full-scale range / 2^bits. For 10 mV and 16 bits, LSB = 0.1526 µV. The RMS quantization noise for a uniformly distributed error is LSB/√12 = 0.044 µV. For a full-scale sinusoid, the theoretical signal-to-quantization-noise ratio is 6.02 × bits + 1.76 dB = 98.1 dB.

**Branch point:** if the input offset is large and the signal is small, a DC-coupled front end wastes codes. An AC-coupled front end or a baseline restore circuit recovers the range but introduces its own high-pass cutoff.

**Edge case:** a 24-bit ADC with a noisy front end does not deliver 24 effective bits. Report ENOB from measured noise, not the datasheet bit count.

**Offset and range.** Electrode half-cell potentials can reach hundreds of millivolts, far above the 1 mV ECG. A DC-coupled front end with a ±5 mV range would saturate immediately. Clinical front ends use a baseline restore circuit or AC coupling with a cutoff around 0.05 Hz. The restore circuit is itself a high-pass filter, and its cutoff must be included in the chain description. A device that reports "0.05 Hz to 150 Hz" usually means the combination of the analog restore and the digital filter, not a single filter.

**Headroom.** Leave at least 3 dB to 6 dB of headroom above the largest expected signal. Motion and electrode artifacts can be ten times the physiological amplitude. A range chosen for the clean signal will clip on the first movement, and clipping is not recoverable.

### Step 5. Establish and verify the time base

**Input:** the device clock, the sample counter, and any external time reference.

**Output:** timestamps for each sample or frame, a measured clock offset, and a drift model.

**Rationale:** every interval measurement is a difference of timestamps. If the sample counter is the only time base, a dropped sample shifts every later sample earlier by one sampling interval. If the device clock drifts, long records accumulate error. A 50 ppm drift is 4.32 s per day.

**Arithmetic:** at fs = 500 Hz, one sample is 2 ms. A single dropped sample creates a 2 ms error in every subsequent interval if the gap is not recorded. A 50 ppm clock error over a 10 s window is 0.5 ms; over an 8 h night it is 1.44 s.

**Branch point:** if the record is short and the decision is rate-based, sample counting is adequate. If the record spans days or is compared across devices, an external time reference or periodic resynchronisation is required.

**Edge case:** timestamps may be recorded at the start of a frame, at the end, or at the middle. Mixing conventions between devices creates a constant offset that looks like a latency difference.

### Step 6. Ingest with metadata and validate the schema

**Input:** raw files or streams, a header, and a data dictionary.

**Output:** a validated record with units, sampling rate, channel labels, gain, filter settings, and a quality flag.

**Rationale:** most public datasets ship a header (WFDB, EDF, or vendor-specific). Most internal pipelines strip it. Once stripped, the signal cannot be interpreted without archaeology. The schema check catches unit errors, channel swaps, and sampling-rate mismatches before they propagate.

**Branch point:** if the sampling rate is absent, the record cannot be resampled safely. Either recover it from the file header or treat the record as unusable for interval measurement.

**Edge case:** a file may contain multiple sampling frequencies, with some channels oversampled by an integer factor. WFDB represents this with samples-per-frame; naive readers decimate by averaging and silently lose bandwidth.

**The metadata contract.** The following fields are the minimum for a reproducible single-channel record. If a field is absent, record the absence in the dataset card rather than assuming a value.

| Field | Example | Why it matters |
|---|---|---|
| Sampling rate | 500 Hz | Timing and frequency |
| Amplitude unit | mV | Thresholds and clinical interpretation |
| Amplitude scale | 1 mV = 1,000 counts | Conversion from codes |
| ADC bits | 16 | Resolution and LSB |
| Full-scale range | ±5 mV | Saturation detection |
| Channel label | Lead I | Montage and interpretation |
| Sensor site | Left wrist | Physiology and artifact |
| Hardware filter | 0.05-150 Hz, analog | Bandwidth and phase |
| Digital filter | 0.5-40 Hz, 4th order Butterworth | Reproducibility |
| Device and firmware | model, version | Known differences |
| Time base | device clock, UTC offset | Alignment and drift |
| Quality rule | version, thresholds | Rejection and missingness |

The contract is deliberately short. A dataset with these twelve fields can be reprocessed by a third party. A dataset without them can still be analysed, but the analysis is not reproducible, and the difference should be stated in the methods.

### Step 7. Detect and handle invalid samples

**Input:** the raw digital signal and the device status flags.

**Output:** a sample-level validity mask and a segment-level quality label.

**Rationale:** invalid samples are not random noise. Clipping produces a flat top that a peak detector may treat as a plateau; lead-off produces a rail or a large swing that a QRS detector may treat as a beat; dropped samples produce a discontinuity that a derivative-based detector treats as a spike. Each has a signature and each requires a different action.

**Branch point:** see the decision table at the end of this section. The choice between interpolation, flagging, and rejection depends on the fraction of invalid samples and the decision's tolerance.

**Edge case:** a signal can be saturated without touching the digital rail if the analog amplifier saturates first. The digital values look plausible but are clipped in the analog domain. Compare the observed amplitude distribution with the known front-end range.

**Combining flags.** Flags are not independent. A motion burst often produces baseline shift, which triggers a baseline-wander flag, and may also produce clipping. A lead-off event often produces a rail that looks like saturation. The combination rule should be explicit:

1. If the device reports lead-off, reject regardless of any other flag.
2. If saturation or clipping is present on a fiducial point, reject the beat, not the whole epoch, when the epoch has enough usable beats.
3. If motion is present but no clipping is present, flag the epoch and compute features with a robust estimator; do not reject automatically.
4. If baseline wander is present with no other flag, filter and keep.
5. If non-wear is present, mark missing and do not impute.

The order matters. A rule that rejects on motion before checking lead-off may reject a segment that is actually a disconnected electrode, which is a different downstream problem. A rule that treats saturation as motion will impute a signal that does not exist.

**Detection methods.** Each defect has a cheap detector that runs before any physiological feature is computed.

| Defect | Detector | Parameter | Typical threshold |
|---|---|---|---|
| Clipping | Count samples equal to the maximum code in a rolling window | 100 ms | ≥ 3 consecutive equal maxima |
| Lead-off | Device status bit, or signal variance near zero with a rail offset | 1 s | status bit set |
| Dropped sample | Sample counter difference greater than one | per sample | gap ≥ 1 |
| Saturation | Fraction of samples within 1% of the rail | 1 s | > 0.1% |
| Motion | Accelerometer magnitude variance, or broadband power ratio | 1 s | study-specific |
| Non-wear | Accelerometer stillness plus optical/electrical absence | 5 min | all axes < 0.01 g |

The thresholds are starting points, not universal constants. Each must be validated against adjudicated examples on the target device. The table is useful mainly because it forces the team to name a detector and a parameter for each defect instead of writing "artifacts were removed."

### Step 8. Remove baseline wander

**Input:** the valid signal.

**Output:** a detrended signal and a record of the filter used.

**Rationale:** baseline wander is low-frequency content from respiration, perspiration, electrode motion, or PPG venous pooling. It reduces the usable dynamic range and confuses threshold-based detectors. A high-pass filter removes it, but the cutoff must be below the lowest physiological content the decision needs.

**Arithmetic:** a first-order high-pass at 0.5 Hz has a gain of 0.706 at 0.5 Hz (-3.02 dB), 0.447 at 0.25 Hz (-7.00 dB), and 0.892 at 1 Hz (-0.99 dB). A first-order high-pass at 0.05 Hz has a gain of 0.980 at 0.25 Hz (-0.17 dB). The 0.5 Hz filter removes wander better and distorts the ST segment more.

**Branch point:** for rhythm analysis, 0.5 Hz is acceptable. For diagnostic ECG, 0.05 Hz is required. For PPG, a 0.5 Hz high-pass is common because the heart rate band starts around 0.5 Hz to 1 Hz, but it can distort the pulse waveform used for morphology.

**Edge case:** a causal high-pass introduces phase distortion and a start transient. Zero-phase forward-backward filtering removes the phase distortion but is non-causal; it cannot be used in a real-time stream without a buffer and a delay.

### Step 9. Remove powerline interference

**Input:** the detrended signal and the known mains frequency.

**Output:** a signal with the narrowband component attenuated, or a decision to leave it and rely on later averaging.

**Rationale:** mains interference is narrowband and stable in frequency but variable in amplitude. A notch filter at the fundamental removes most of it. A notch also removes any physiological content at the same frequency, which is usually negligible, and introduces ringing around transients.

**Arithmetic:** a second-order notch at 60 Hz with Q = 30 at fs = 500 Hz has a gain of 0.484 at 59.5 Hz (-6.31 dB) and a theoretical null at 60 Hz. At 40 Hz the gain is 0.9993 (-0.01 dB), so the notch is narrow enough to leave the QRS band intact. A wider notch (Q = 5) would attenuate neighbouring frequencies and distort the QRS.

**Branch point:** if the interference is not stable, or if the mains frequency is unknown (50 Hz versus 60 Hz), estimate the line frequency from the spectrum before designing the notch. A notch at the wrong frequency does nothing and may add ringing.

**Edge case:** harmonics. A 60 Hz notch does not remove 120 Hz or 180 Hz. If the front end is nonlinear, harmonics can be larger than the fundamental. Inspect the spectrum rather than assuming a single tone.

**Alternatives to a notch.** A notch is not the only option, and it is not always the best one.

- **Spectral interpolation.** Estimate the line-noise amplitude and phase from a quiet segment and subtract the fitted sinusoid. This avoids the notch's ringing and preserves signal at the same frequency. It requires a stable line frequency.
- **Comb filter.** A filter with notches at the fundamental and its harmonics. Useful when the front end generates strong harmonics, but it removes more signal bands.
- **Median filter.** A nonlinear filter that removes narrow spikes without the phase distortion of a linear filter. It is useful for impulse noise, not for a continuous sinusoid.
- **Averaging.** If the decision is a rate over many beats, the line noise averages down. This is not a substitute for a notch when the detector is threshold-based, because the threshold sees the instantaneous noise.

The choice is a design decision with a documented rationale. "We applied a 60 Hz notch" is not a rationale; "we applied a 60 Hz notch with Q = 30 because the line frequency was stable to ±0.1 Hz and the QRS band starts at 5 Hz" is.

### Step 10. Low-pass for noise and task bandwidth

**Input:** the notched signal and the task bandwidth.

**Output:** a signal band-limited to the task.

**Rationale:** high-frequency noise increases false detections and obscures morphology. A low-pass at the task's upper band improves SNR but removes content that another task needs. The filter must be applied consistently at training and inference; a model trained on 40 Hz low-passed data and deployed on 150 Hz data sees a different input distribution.

**Arithmetic:** a 5-sample moving average at fs = 500 Hz has a gain of 0.765 at 40 Hz (-2.33 dB), 0.517 at 60 Hz (-5.74 dB), and a null at 100 Hz. A moving average is a poor anti-alias filter because its stopband lobes re-open, but it is cheap and adequate for display smoothing.

**Branch point:** if the decision is QRS detection, 15 Hz to 40 Hz is often enough. If the decision is pacemaker spike detection, content above 100 Hz is essential and the low-pass must be much higher.

**Edge case:** filtering after segmentation creates edge effects. If each 10 s epoch is filtered independently, the first and last 0.5 s to 2 s are distorted by the filter's start transient. Filter the whole record, then segment, or discard a guard band at each edge.

**Filter families and what they imply.** The family matters as much as the cutoff.

| Family | Phase | Ripple | Typical use | Caution |
|---|---|---|---|---|
| Butterworth | Nonlinear near cutoff | Maximally flat passband | General band limiting | Phase distortion; use zero-phase offline |
| Chebyshev I | Nonlinear | Ripple in passband | Steep transition with low order | Ripple changes amplitudes |
| Chebyshev II | Nonlinear | Ripple in stopband | Good stopband rejection | Slower roll-off near cutoff |
| Elliptic | Nonlinear | Ripple in both | Very steep transition | Ringing; avoid for morphology |
| Bessel | Near-linear | Flat | Preserve waveform shape | Gentle roll-off |
| FIR linear-phase | Linear | Design-dependent | Offline measurement | Longer filter, more edge effect |
| Moving average | Linear within passband | Sinc-shaped | Cheap smoothing | Stopband lobes re-open |

For interval measurement, a Bessel or a linear-phase FIR preserves fiducial timing better than a Butterworth of the same order. For offline analysis where causality does not matter, forward-backward filtering with a Butterworth gives zero phase and a magnitude response equal to the square of the one-pass response. That squaring is easy to forget: a forward-backward second-order filter has the attenuation of a fourth-order filter but the cutoff definition still refers to the one-pass -3 dB point, so the effective -3 dB point moves.

### Step 11. Segment, window, and compute a quality index

**Input:** the conditioned signal and the decision's epoch length.

**Output:** epochs with a quality label and a quality score.

**Rationale:** most decisions are made on windows, not on individual samples. The window length determines the frequency resolution (Δf ≈ 1/T) and the number of independent observations. The quality index determines whether the window's features are interpretable.

**Arithmetic:** a 10 s window gives a frequency resolution of 0.1 Hz for a rectangular window. A 30 s window gives 0.033 Hz. A 1 s window gives 1 Hz and cannot resolve the boundary between the theta and alpha bands. For HRV, 5 min windows are conventional for frequency-domain measures; 10 s windows are too short for reliable low-frequency power.

**Branch point:** if the quality score is below threshold, the epoch can be rejected, flagged, or repaired. Rejection loses data and can bias the sample; flagging preserves data but requires a model that handles missingness; repair is only appropriate for short, isolated defects.

**Edge case:** quality indices are device-specific and population-specific. A threshold tuned on one device will not transfer. Report the index definition and the threshold, and validate the threshold on the target device.

**Choosing the epoch.** The epoch is the unit of analysis, and it is usually inherited from a convention rather than chosen. Three conventions dominate: 10 s for resting ECG rhythm, 30 s for sleep staging and clinical EEG, and 5 min for frequency-domain HRV. Each carries a resolution limit and a stationarity assumption. A 10 s epoch cannot resolve 0.05 Hz content; a 30 s epoch cannot resolve 0.02 Hz; a 5 min epoch assumes the heart rate is stationary over five minutes, which is false during exercise or stress.

When the decision does not dictate the epoch, choose the shortest epoch that supports the feature, because shorter epochs give more samples and better temporal localization. Then check that the feature is stable when the epoch is doubled. If it is not, the feature is measuring the window, not the physiology.

### Step 12. Extract features, measurements, and uncertainty

**Input:** quality-labelled epochs.

**Output:** a measurement table with units, a feature table, and an uncertainty estimate per row.

**Rationale:** a number without an uncertainty cannot be compared with a threshold or with another device. The uncertainty can be derived analytically (quantization, timing) or empirically (repeat measurements, Bland-Altman). Reporting it separates a measurement from a guess.

**Branch point:** if the uncertainty is larger than the decision margin, the decision cannot be supported by this signal. That is a valid and important output: the correct action is to collect a better signal or choose a different measurement, not to tighten the threshold.

**Edge case:** propagating uncertainty through a nonlinear feature (for example, the ratio of two noisy band powers) requires either a delta method or a bootstrap. A standard deviation computed on the feature values is not the same as the measurement uncertainty.

### Step 13. Write the chain specification

**Input:** the choices made in Steps 1 through 12.

**Output:** a versioned specification with every parameter, every unit, and the rationale for each cutoff.

**Rationale:** the specification is the artefact that survives staff turnover, device firmware updates, and regulatory review. It is also the artefact that makes a reproduction attempt possible. A chain that exists only as code is a chain that cannot be audited by someone who does not read code.

**Contents:** measurand and decision; sensor and site; sampling rate and anti-alias; ADC bits and range; time base; filter family, order, cutoff, and phase convention per stage; epoch length and window; quality rule and thresholds; feature definitions; aggregation rule; known limits.

**Branch point:** if the chain is used for a regulated claim, the specification must also state the device, firmware version, and the validation evidence for each parameter. A parameter without evidence is a hypothesis, not a validated setting.

**Edge case:** a chain that is correct for one firmware version may be wrong for the next. Firmware updates can change the internal sampling rate, the automatic gain control, or the filter coefficients. Version the chain against the firmware, and re-verify after an update.

**Edge cases the procedure does not solve.** Three situations remain outside the twelve steps, and each requires a product decision rather than a signal-processing one.

- **Non-wear that looks like a valid signal.** A PPG sensor on a table can return a clean DC level with no pulsatile component. The absence of a pulse is not the same as a pulse of zero amplitude. Only a wear detector, typically an accelerometer or a capacitive sensor, distinguishes the two.
- **Motion that is the behaviour of interest.** In activity recognition, motion is the signal, not the artifact. A chain that removes motion before feature extraction destroys the measurement. The same accelerometer stream can be artifact for heart rate and signal for gait.
- **A physiological event that resembles artifact.** A seizure can produce high-amplitude rhythmic EEG that looks like movement artifact; a ventricular fibrillation can look like noise on a single lead. Quality rules must not reject the events the product exists to detect. This is why quality thresholds are validated against adjudicated events, not only against clean recordings.
- **A device that changes its own chain.** Firmware may switch sampling rate when the battery is low or when the user starts an exercise mode. The switch is documented in the firmware release notes, not in the signal. A pipeline that assumes one rate produces a discontinuity that looks like a physiological event.
- **A record assembled from multiple sessions.** Concatenating two recordings without a gap marker creates an apparent interval between the last beat of one session and the first beat of the next. The interval can be hours long, and a naive rate calculation will include it. Insert an explicit gap and exclude the boundary intervals.

### A reference parameter set

The following table is a starting point for a rest ECG recorded on a patch at 500 Hz. It is not a universal recommendation. Each row is a choice that must be justified by the decision.

| Stage | Parameter | Value | Justification |
|---|---|---|---|
| Acquisition | Sampling rate | 500 Hz | 150 Hz diagnostic band with margin |
| Acquisition | Anti-alias | 4th-order Butterworth, 200 Hz | Attenuate above 250 Hz |
| Acquisition | ADC | 16 bit, ±5 mV | 0.153 µV LSB; 6,554 codes per mV |
| Time base | Clock | 50 ppm or better | 0.5 ms per 10 s |
| Baseline | High-pass | 0.5 Hz, 1st order, zero-phase | Rhythm only; not ST analysis |
| Mains | Notch | 60 Hz, Q = 30 | Narrow; preserves 40 Hz |
| Noise | Low-pass | 40 Hz, 4th order, zero-phase | Rhythm bandwidth |
| Quality | Clipping | ≥ 3 equal maxima in 100 ms | Detect rail |
| Quality | Lead-off | status bit or flat rail | Reject segment |
| Segmentation | Epoch | 10 s, no overlap | 0.1 Hz resolution |
| Features | HR | 60 / mean RR | Beat-based |
| Features | RMSSD | successive differences | Short-window HRV |
| Aggregation | Record HR | median of epoch HR | Robust to one bad epoch |

The reference set produces the numbers in Section 6. If you change the high-pass to 0.05 Hz, the rhythm numbers do not change, but the ST analysis becomes possible. If you change the sampling rate to 250 Hz, the rhythm numbers still do not change, but the pacemaker-spike analysis does. The table is a chain, not a recipe.

### Decision table for invalid or low-quality data

| Condition | Detect by | Fraction | Action | Rationale |
|---|---|---|---|---|
| Lead-off | Flat rail, impedance flag, or step to rail | Any | Reject segment; do not interpolate | No physiological signal exists |
| Clipping | Repeated values at ADC maximum or minimum | < 1% of samples | Flag; interpolate short runs; reject if on a fiducial point | Clipped peaks bias amplitude and timing |
| Saturation (analog) | Amplitude distribution truncated below digital rail | Any | Reject segment | Information lost before digitisation |
| Baseline wander | Low-frequency power above threshold | > 20% of epoch | High-pass at task-appropriate cutoff | Removable if cutoff is safe |
| 50/60 Hz interference | Spectral peak at mains frequency | Any | Notch or spectral removal; re-check | Narrowband and removable |
| Motion artifact | Burst of broadband power, often with baseline shift | < 10% of epoch | Flag; exclude from features; do not relabel | Artifact can mimic arrhythmia |
| Dropped samples | Counter discontinuity or timestamp gap | Any | Record gap; interpolate only if short and isolated | Timing error propagates |
| Clock drift | Offset versus reference grows linearly | Any | Correct timestamps; report residual | Interval error accumulates |
| Non-wear | Accelerometer stillness plus optical absence | > 5 min | Mark as missing, not as zero | Missingness is informative |
| Poor perfusion (PPG) | Low perfusion index, low SNR | Any | Flag; do not report SpO2 | Signal cannot support the measurement |

## 6. Worked example

### 6.1 The case

A single-lead ECG patch records 10 s at 500 samples/s. The ADC is 16-bit over a ±5 mV range. The subject is at rest. The raw record contains baseline wander at 0.25 Hz with an amplitude of 0.4 mV, mains interference at 60 Hz with an amplitude of 0.08 mV, and a 1 mV QRS complex. The true heart rate is approximately 72 beats per minute (bpm). The analysis goal is heart rate and a short-window HRV measure, not ST-segment analysis.

### 6.2 Acquisition arithmetic

- Number of samples: N = fs × T = 500 samples/s × 10 s = 5,000 samples.
- Nyquist frequency: fs/2 = 250 Hz.
- Storage at 16 bits: 5,000 samples × 2 bytes = 10 kB per lead, before compression. A 24 h record at the same rate and resolution is 500 × 86,400 × 2 = 86.4 MB per lead, which is why ambulatory devices use compression and often a lower rate.
- Beat count expected at 72 bpm over 10 s: 72/60 × 10 = 12 beats, or about 11 complete RR intervals after the first peak. The detector found 11 peaks, which is consistent with the expected count, not with a missing beat.
- Signal bandwidth of interest for rhythm: 0.5 Hz to 40 Hz. The sampling rate is 12.5 times the upper band edge, so the margin is large.
- ADC LSB: 10 mV / 65,536 = 0.1526 µV per code.
- QRS amplitude in codes: 1 mV / 0.1526 µV = 6,554 codes.
- Baseline wander amplitude in codes: 0.4 mV / 0.1526 µV = 2,621 codes. The wander alone occupies 40% of the QRS amplitude, which is why detrending is necessary before threshold detection.
- Mains amplitude in codes: 0.08 mV / 0.1526 µV = 524 codes. It is 8% of the QRS amplitude but its derivative is large because its frequency is high.

### 6.3 Noise budget

A signal chain is only as good as its worst noise source. The budget below is computed at the input of the detector, after the 60 Hz notch and the 5 Hz high-pass. Each entry is an RMS amplitude in microvolts.

| Source | Calculation | RMS (µV) | Share of power |
|---|---|---:|---:|
| Quantization | LSB/√12 = 0.1526/3.464 | 0.044 | < 0.01% |
| Front-end thermal and amplifier noise | datasheet, input-referred | 2.0 | 1.0% |
| Mains residual after 40 dB notch | 80 µV × 0.01 | 0.8 | 0.2% |
| Baseline residual after 5 Hz high-pass | 400 µV × 0.0499 | 19.97 | 98.8% |
| Motion | not present in this record | 0 | 0% |
| Total (root sum of squares) | √(0.044² + 2.0² + 0.8² + 19.97²) | 20.09 | 100% |

The total noise is 20.09 µV RMS against a 1 mV QRS, an SNR of 49.8, or 33.9 dB. The dominant term is not the ADC and not the mains; it is the residual baseline wander that survived the 5 Hz high-pass. This is the point of the budget: it tells you where to spend effort. Buying a 24-bit ADC would improve the quantization term from 0.044 µV to 0.003 µV and change the total from 20.09 µV to 20.09 µV. Raising the high-pass order or cutoff would change the total materially.

The budget also shows why "16-bit is enough" is context-dependent. For this record, 16 bits is more than enough because the baseline dominates. For a record with aggressive baseline removal and a low-noise front end, the quantization term could become visible, and 24 bits would matter.

### 6.4 Filter design arithmetic

The chain is: 60 Hz notch (second-order, Q = 30), then a 5 Hz first-order high-pass for detection, then a 40 Hz low-pass for display. A separate diagnostic chain would use 0.05 Hz and 150 Hz; it is not used here.

Notch gain at the frequencies of interest, computed from the biquad coefficients:

| Frequency | Gain | Attenuation |
|---:|---:|---:|
| 0.5 Hz | 1.0000 | 0.00 dB |
| 40 Hz | 0.9993 | -0.01 dB |
| 59.5 Hz | 0.4837 | -6.31 dB |
| 60 Hz | 0.0000 | theoretical null |

The notch is narrow. At 40 Hz it removes essentially nothing, so the QRS band is preserved. At 59.5 Hz it removes about half the amplitude; at 60 Hz it removes the tone. The residual after the notch depends on how stable the mains frequency is. If the mains wanders by ±0.5 Hz, the attenuation at the edge of the wander is only about 6 dB, and a residual tone remains.

High-pass gain at 0.25 Hz:

| Cutoff | Gain at 0.25 Hz | Attenuation |
|---:|---:|---:|
| 0.05 Hz | 0.9803 | -0.17 dB |
| 0.5 Hz | 0.4469 | -7.00 dB |

A 0.5 Hz high-pass removes 55% of the 0.25 Hz wander amplitude, which is enough for detection but not enough for a clean baseline. A 0.05 Hz high-pass removes only 2% of it. For rhythm detection, a higher cutoff is acceptable. For ST analysis, it is not.

Low-pass check with a 5-sample moving average at 500 Hz:

| Frequency | Gain | Attenuation |
|---:|---:|---:|
| 0.5 Hz | 1.0000 | 0.00 dB |
| 40 Hz | 0.7649 | -2.33 dB |
| 60 Hz | 0.5167 | -5.74 dB |
| 100 Hz | 0.0000 | null |

A 5-sample moving average is not a serious low-pass. It removes half the 60 Hz amplitude and nothing above 100 Hz in a predictable way, but its stopband lobes re-open. For a real chain, use a fourth-order Butterworth or an equiripple design and verify the stopband.

### 6.4.1 The filter response the decision needs

The chain in this case is for rhythm, not for morphology. The decision is "is the rate in a normal range," and the measurement is the mean RR over 10 s. The filter choices follow from that.

- A 5 Hz high-pass removes baseline wander and T-wave content. The T wave is not needed for rate.
- A 40 Hz low-pass removes mains harmonics and muscle noise above the QRS band. The QRS fundamental is around 10 Hz to 20 Hz, so 40 Hz preserves it.
- The notch is placed at the mains frequency. It is not a substitute for the high-pass, and the high-pass is not a substitute for the notch.

If the decision were "is there ST elevation," the chain would need 0.05 Hz and 150 Hz, and the 5 Hz high-pass would be a defect. The same record supports both decisions, but not with the same chain. The specification must name the decision, not only the signal.

### 6.5 Detection and rate arithmetic

The detection chain is: notch, 5 Hz high-pass, derivative, squaring, 120 ms moving-window integration, threshold at 50% of the maximum integrated value, and a 200 ms refractory skip. The detector finds 11 R peaks in the 10 s window.

- Mean RR interval: 0.8304 s.
- Heart rate: 60 / 0.8304 = 72.25 bpm.
- Successive differences: 10 differences from 11 intervals.
- RMSSD (root mean square of successive differences): 14.24 ms.

The true rate is about 72 bpm, and the estimate is 72.25 bpm. The difference is within one beat-count quantum over the window: 60 / 10 s = 6 bpm per beat, so a 0.25 bpm difference is far below the count resolution. This is an important point: the rate estimate is more precise than the beat count because it uses the mean interval, but it is not more accurate if a peak is missed.

**Precision of the mean-interval estimate.** The RR intervals in this record vary because the synthetic generator applies 2% jitter. The standard deviation of the observed intervals is about 16.6 ms. The standard error of the mean of 10 intervals is 16.6/√10 = 5.25 ms. The heart rate is HR = 60/RR, so the derivative is dHR/dRR = -60/RR² = -60/0.6896 = -87.0 bpm per second. Multiplying by the standard error gives 87.0 × 0.00525 = 0.46 bpm. The estimate is therefore 72.25 ± 0.46 bpm (standard error), before any timing or fiducial bias is considered.

The two numbers, 6 bpm per beat and 0.46 bpm standard error, answer different questions. The first is the resolution of a count. The second is the precision of an interval-based estimate. A report that says "heart rate 72 bpm" without saying which estimator was used is ambiguous by a factor of more than ten.

**Fiducial bias.** The detector's refine step searches for the maximum absolute amplitude within ±50 ms of the integrated peak. The causal 5 Hz high-pass shifts the waveform by about 26 ms, so the detected peak is at 0.376 s while the synthetic R wave is at 0.350 s. The shift is nearly constant across beats, so the mean RR is unbiased. If the shift varied with heart rate or with waveform morphology, the RR intervals would be biased in a rate-dependent way. That is the failure mode that zero-phase filtering prevents.

### 6.6 Diagnostic 1: a short interval in the RR series

The first run used a threshold at 35% of the maximum integrated value. It found 12 peaks, including one at 0.138 s and one at 0.376 s, an interval of 0.238 s, which corresponds to 252 bpm. No human at rest has a 0.238 s RR interval. The extra peak was the P wave of the first beat, which crossed the lower threshold because the P wave's derivative survived the 5 Hz high-pass.

The diagnostic is the interval histogram: a cluster of intervals near 0.83 s and one outlier at 0.24 s. The outlier is shorter than any physiological interval and shorter than the refractory period of the detector's own skip window. The fix is to raise the threshold or to add a refractory rule of 250 ms; the higher threshold removed the false peak and produced 11 peaks.

The consequence of missing this diagnostic: heart rate 77.27 bpm instead of 72.25 bpm, an error of 5.02 bpm, or 6.9%. In a screening product, that is the difference between "normal" and "elevated" for a subject near the threshold.

### 6.7 Diagnostic 2: a 40 Hz spectral peak that is not muscle noise

The raw spectrum shows a peak at 40 Hz, not at 60 Hz. The naive reading is that the subject has muscle artifact at 40 Hz. The correct reading is that the signal was sampled at 100 Hz somewhere in the chain before storage, so the 60 Hz mains tone aliased to 100 - 60 = 40 Hz. A 40 Hz low-pass would remove the peak but also remove the real 40 Hz content, and the mains tone would still be present in the original acquisition.

Two checks confirm the alias: (1) the peak frequency changes when the sampling rate changes, which a physiological source would not do; (2) the peak is absent when the same subject is recorded on a device that samples at 500 Hz. The fix is to correct the acquisition, not to filter the stored record.

A third check is arithmetic: 60 Hz at 100 Hz sampling must appear at 40 Hz. If the observed peak is at 40 Hz and the device documentation says the internal rate is 100 Hz, the alias is explained without any further data. If the observed peak is at 40 Hz and the device samples at 500 Hz, the peak is physiological or the true source is an 40 Hz interferer, not mains.

### 6.7.1 Comparing the two diagnostics

| Diagnostic | Signal | What a false negative looks like | What it protects |
|---|---|---|---|
| RR interval histogram | One interval at 0.238 s among ten near 0.83 s | A single short interval averaged away by the mean | Prevents a P wave or noise spike from inflating HR |
| Spectral alias check | Peak at 40 Hz with fs = 100 Hz | A 40 Hz low-pass that removes signal and leaves noise | Prevents a sampling error from being treated as physiology |

The two diagnostics are complementary. The first operates in the time domain and protects the rate estimate. The second operates in the frequency domain and protects the interpretation of noise. A pipeline that runs both before feature extraction catches the two most common chain errors in this case.

### 6.8 Varying one assumption

Suppose the sampling rate is 250 Hz instead of 500 Hz. Nyquist becomes 125 Hz, so the 150 Hz diagnostic content folds to 100 Hz. For rhythm detection with a 40 Hz band, the fold is harmless because the 100 Hz alias is outside the band and the 40 Hz low-pass removes it. For pacemaker spike detection, the fold is fatal: a 150 Hz spike appears at 100 Hz and may be mistaken for noise or removed by the low-pass.

Suppose instead that the high-pass cutoff is 0.5 Hz instead of 0.05 Hz. For rhythm detection, nothing changes: the QRS band is above 5 Hz and the detector's own 5 Hz high-pass dominates. For ST-segment analysis, the 0.5 Hz filter attenuates a 0.25 Hz ST change to 44.7% of its true amplitude, so a 0.1 mV ST elevation is reported as 0.045 mV. The conclusion changes from "ischemic" to "normal" without any change in the patient.

The sensitivity table makes the consequences explicit.

| Assumption varied | Baseline result | Variant result | Change | Interpretation |
|---|---|---:|---:|---|
| Detector threshold 0.50 → 0.35 | 11 peaks, 72.25 bpm | 12 peaks, 77.27 bpm | +5.02 bpm (+6.9%) | A false P-wave detection changes the screening conclusion |
| High-pass 5 Hz → 0.5 Hz | 72.25 bpm, RMSSD 14.24 ms | approximately unchanged | below the 0.5 bpm standard error | Rhythm is robust to this change |
| High-pass 0.05 Hz for ST | ST gain 0.98 at 0.25 Hz | ST gain 0.447 at 0.5 Hz cutoff | -55% amplitude | ST interpretation is not robust |
| Sampling 500 → 250 Hz | 150 Hz content preserved | 150 Hz folds to 100 Hz | Alias | Rhythm unaffected; spike detection broken |
| Window 10 s → 5 min | Short-window HRV only | LF power estimable | New feature | The measurement changes with the window |
| Notch omitted | 80 µV mains | 80 µV in band | Detector threshold unstable | The notch is load-bearing for this chain |

The first row is the one that changes a conclusion. A 6.9% heart-rate error is enough to move a subject across a 100 bpm threshold in a screening application, and it arises from a single threshold choice, not from the sensor or the patient.

### 6.9 Conclusion of the case

The chain produces a heart rate of 72.25 bpm and an RMSSD of 14.24 ms from a 10 s window. The estimate is adequate for rhythm screening and inadequate for frequency-domain HRV, because 10 s is shorter than the 5 min convention for low-frequency power. The two diagnostics (short RR outlier, 40 Hz alias peak) catch the two errors that would otherwise survive into the feature table. The assumption that matters most is the filter cutoff: it is safe for rhythm and unsafe for ST analysis.

### 6.10 What the RMSSD does and does not mean

RMSSD is the root mean square of successive differences between adjacent RR intervals. It is a time-domain HRV feature that reflects short-term vagally mediated variation. In this record it is 14.24 ms, computed from 10 differences. Three cautions apply.

First, the value is computed on 10 intervals. The confidence interval around an RMSSD from 10 intervals is wide, and the estimate is sensitive to a single ectopic beat or a single detection error. A clinical HRV report typically uses 5 min segments and edits ectopic beats before computing RMSSD. Neither step is present here.

Second, the synthetic generator's 2% jitter is not physiology. The 14.24 ms value is a property of the generator, not a normative value. Do not compare it with published HRV norms.

Third, RMSSD depends on the fiducial point. If the detector times the R peak on the upstroke in one beat and on the downstroke in another, the successive differences include a morphology component. Consistent fiducial definition matters more for RMSSD than for mean RR, because RMSSD differences adjacent beats rather than averaging over many.

The lesson generalizes: a feature is only as interpretable as its input and its estimator. The number 14.24 ms is arithmetically correct and clinically meaningless without the window, the editing rule, the population, and the fiducial definition.

A final arithmetic check ties the feature to the record. With a mean RR of 0.8304 s and 11 intervals, the total analysed time is 10 × 0.8304 = 8.304 s of RR data inside a 10 s record. The remaining 1.696 s is the time before the first detected peak and after the last. That is why the beat count and the window length do not correspond exactly, and why a 10 s window yields 10 differences rather than 11.

This also explains why a 10 s window is a poor choice for a rate trend. Each window contributes one estimate with a standard error near 0.5 bpm, but adjacent windows overlap in their beat sets and are not independent. A trend computed from overlapping 10 s windows has fewer effective degrees of freedom than the number of windows suggests. Use non-overlapping windows or a model that accounts for the overlap.

## 7. Code, query, or calculation

The following script uses only the Python standard library and reproduces every number in Section 6. It is deliberately explicit: the filters are implemented as difference equations, the FFT is radix-2, and the R-peak detector is a simplified Pan-Tompkins chain. Run it with Python 3.11 or later.

```python
import math, cmath, random

FS, DUR, BITS = 500.0, 10.0, 16
FULL_SCALE_MV = 10.0

# --- ADC resolution -------------------------------------------------
lsb_uv = FULL_SCALE_MV * 1000.0 / (2 ** BITS)
print("LSB_uV=%.4f  QRS_codes=%.0f  SQNR_dB=%.1f"
      % (lsb_uv, 1e-3 / (lsb_uv * 1e-6), 6.02 * BITS + 1.76))

# --- aliasing -------------------------------------------------------
def alias(f, fs):
    f %= fs
    return fs - f if f > fs / 2 else f
for f, fs in ((60, 100), (60, 500), (150, 250)):
    print("alias_%g_at_%g=%.1f" % (f, fs, alias(f, fs)))

# --- moving-average low-pass ---------------------------------------
def ma_gain(f, fs, m):
    return abs(math.sin(math.pi*f*m/fs) / (m*math.sin(math.pi*f/fs)))
for f in (0.5, 40.0, 60.0, 100.0):
    g = ma_gain(f, FS, 5)
    print("MA5_%g_Hz=%.4f (%.2f dB)" % (f, g, 20*math.log10(max(g, 1e-12))))

# --- one-pole high-pass --------------------------------------------
def hp_alpha(fc, fs):
    rc = 1.0/(2*math.pi*fc)
    return rc/(rc + 1.0/fs)

def hp_gain(f, fc, fs):
    a = hp_alpha(fc, fs); w = 2*math.pi*f/fs
    return abs(a*(1-cmath.exp(-1j*w))) / abs(1-a*cmath.exp(-1j*w))

def hp_apply(x, fc, fs):
    a = hp_alpha(fc, fs); y = [0.0]*len(x)
    for n in range(1, len(x)):
        y[n] = a*(y[n-1] + x[n] - x[n-1])
    return y
for fc in (0.05, 0.5):
    print("HP%g_alpha=%.5f  gain_at_0.25=%.4f (%.2f dB)"
          % (fc, hp_alpha(fc, FS), hp_gain(0.25, fc, FS),
             20*math.log10(hp_gain(0.25, fc, FS))))

# --- biquad notch ---------------------------------------------------
def notch_coeffs(f0, q, fs):
    w0 = 2*math.pi*f0/fs; al = math.sin(w0)/(2*q)
    b = [1/(1+al), -2*math.cos(w0)/(1+al), 1/(1+al)]
    a = [1.0, -2*math.cos(w0)/(1+al), (1-al)/(1+al)]
    return b, a

def biquad_gain(f, b, a, fs):
    z = cmath.exp(-1j*2*math.pi*f/fs)
    return abs((b[0]+b[1]*z+b[2]*z*z)/(a[0]+a[1]*z+a[2]*z*z))

def notch_apply(x, b, a):
    y = [0.0]*len(x); x1=x2=y1=y2=0.0
    for n, xn in enumerate(x):
        yn = b[0]*xn + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2
        x2, x1 = x1, xn; y2, y1 = y1, yn; y[n] = yn
    return y
b, a = notch_coeffs(60.0, 30.0, FS)
for f in (0.5, 40.0, 59.5, 60.0):
    g = biquad_gain(f, b, a, FS)
    print("NOTCH60_Q30_%g=%.5f (%.2f dB)" % (f, g, 20*math.log10(max(g,1e-12))))

# --- radix-2 FFT and band power ------------------------------------
def fft(x):
    n = len(x)
    if n == 1: return [complex(x[0])]
    if n & (n-1): raise ValueError("power of two required")
    ev, od = fft(x[0::2]), fft(x[1::2]); out=[0j]*n
    for k in range(n//2):
        t = cmath.exp(-2j*math.pi*k/n)*od[k]
        out[k] = ev[k]+t; out[k+n//2] = ev[k]-t
    return out

def band_power(x, fs, lo, hi):
    n = 1
    while n*2 <= len(x): n *= 2
    spec = fft(x[:n]); df = fs/n; p = 0.0
    for k in range(1, n//2):
        f = k*df
        if lo <= f < hi: p += 2*abs(spec[k])**2/n**2
    return p, df
fs2, n2 = 256.0, 1024
sig = [math.sin(2*math.pi*10*t/fs2)+0.5*math.sin(2*math.pi*60*t/fs2) for t in range(n2)]
for lo, hi in ((8,13),(55,65),(1,128)):
    p, df = band_power(sig, fs2, lo, hi)
    print("power_%d_%d=%.4f" % (lo, hi, p))
print("df_256_1024=%.4f" % df)

# --- synthetic ECG, detection, HRV ---------------------------------
def synth_ecg(fs, dur, hr, jitter, seed):
    rnd = random.Random(seed); x=[0.0]*int(fs*dur); t=0.35
    while t < dur-0.6:
        for c, amp, w in ((t-0.20,0.12,0.025),(t,1.00,0.010),(t+0.28,0.30,0.045)):
            for i in range(max(0,int((c-4*w)*fs)), min(len(x),int((c+4*w)*fs))):
                x[i] += amp*math.exp(-0.5*((i/fs-c)/w)**2)
        t += (60.0/hr)*(1 + rnd.uniform(-jitter, jitter))
    return x

def detect_r(x, fs):
    d=[0.0]*len(x)
    for i in range(1,len(x)): d[i]=x[i]-x[i-1]
    sq=[v*v for v in d]; m=int(0.12*fs); integ=[0.0]*len(x); acc=0.0
    for i,v in enumerate(sq):
        acc += v
        if i>=m: acc -= sq[i-m]
        integ[i]=acc/m
    thr=0.50*max(integ); peaks=[]; i=0
    while i < len(integ):
        if integ[i] > thr:
            j=i
            while j+1 < len(integ) and integ[j+1] > thr: j+=1
            pk=max(range(i,j+1), key=lambda k: integ[k])
            lo=max(0,pk-int(0.05*fs)); hi=min(len(x),pk+int(0.05*fs))
            peaks.append(max(range(lo,hi), key=lambda k: abs(x[k])))
            i=j+int(0.2*fs)
        else:
            i+=1
    return peaks

raw = synth_ecg(FS, DUR, 72.0, 0.02, 3)
raw = [v + 0.4*math.sin(2*math.pi*0.25*i/FS) + 0.08*math.sin(2*math.pi*60*i/FS)
       for i, v in enumerate(raw)]
clean = notch_apply(hp_apply(raw, 5.0, FS), b, a)
peaks = detect_r(clean, FS)
rr = [(peaks[i+1]-peaks[i])/FS for i in range(len(peaks)-1)]
mean_rr = sum(rr)/len(rr)
hr = 60.0/mean_rr
rmssd = math.sqrt(sum((rr[i+1]-rr[i])**2 for i in range(len(rr)-1))/(len(rr)-1))
print("peaks=%d mean_RR=%.4f HR=%.2f RMSSD=%.2f_ms"
      % (len(peaks), mean_rr, hr, rmssd*1000))
print("first_peaks_s=%s" % [round(p/FS,3) for p in peaks[:4]])
print("drift_50ppm_10s=%.3f_ms drift_50ppm_24h=%.2f_s"
      % (50e-6*10*1000, 50e-6*86400))
```

Expected output (Python 3.11, standard library only):

```text
LSB_uV=0.1526  QRS_codes=6554  SQNR_dB=98.1
alias_60_at_100=40.0
alias_60_at_500=60.0
alias_150_at_250=100.0
MA5_0.5_Hz=1.0000 (-0.00 dB)
MA5_40_Hz=0.7649 (-2.33 dB)
MA5_60_Hz=0.5167 (-5.74 dB)
MA5_100_Hz=0.0000 (-240.00 dB)
HP0.05_alpha=0.99937  gain_at_0.25=0.9803 (-0.17 dB)
HP0.5_alpha=0.99376  gain_at_0.25=0.4469 (-7.00 dB)
NOTCH60_Q30_0.5=1.00000 (-0.00 dB)
NOTCH60_Q30_40=0.99930 (-0.01 dB)
NOTCH60_Q30_59.5=0.48365 (-6.31 dB)
NOTCH60_Q30_60=0.00000 (-240.00 dB)
power_8_13=0.5000
power_55_65=0.1250
power_1_128=0.6250
df_256_1024=0.2500
peaks=11 mean_RR=0.8304 HR=72.25 RMSSD=14.24_ms
first_peaks_s=[0.376, 1.212, 2.048, 2.876]
drift_50ppm_10s=0.500_ms drift_50ppm_24h=4.32_s
```

Three details deserve comment.

First, the moving-average gain at 100 Hz prints as -240 dB because the analytic gain is exactly zero at the null. The number is a floating-point artifact, not a physical attenuation. Report "null at 100 Hz" instead.

Second, the notch gain at 60 Hz is a numerical zero. Real filters have finite attenuation, limited by coefficient quantization and by the stability of the mains frequency. A measured attenuation of 30 dB to 50 dB is more realistic than the ideal infinite null.

Third, the first detected peak is at 0.376 s while the synthetic R wave is at 0.350 s. The 26 ms difference is the group delay of the causal 5 Hz high-pass. It is nearly constant across beats, so the RR intervals are unaffected, but the absolute fiducial time is shifted. For interval measurement, either use zero-phase filtering or calibrate the delay.

The band-power block is a self-check on the FFT: a unit-amplitude 10 Hz sine has power 0.5, a 0.5-amplitude 60 Hz sine has power 0.125, and the sum is 0.625. If the numbers differ, the spectral scaling is wrong. This is the fastest way to catch a factor-of-two or factor-of-N error in a periodogram.

## 8. Data, standards, and regulatory context

Signal-chain parameters are constrained by standards, by dataset conventions, and by regulators. The table separates binding requirements from conventions that are widely followed but not legally mandatory.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| AHA/ACCF/HRS 2007, Part I (Kligfield et al.) | ECG acquisition, bandwidth, sampling for storage, display | Professional scientific statement; not law | 500 samples/s storage, 0.05 Hz to 150 Hz diagnostic bandwidth in adults |
| ANSI/AAMI EC11 | Diagnostic electrocardiograph performance | Consensus standard, often cited by FDA | Frequency response, accuracy, and test methods |
| IEC 60601-2-25:2011 | Basic safety and essential performance of diagnostic electrocardiographs | Binding when claimed or required by regulators | Accuracy tables for intervals and waveforms; test data sets |
| IEC 60601-2-27 | Cardiac monitors with alarms | Binding when claimed | Bandwidth and alarm performance differ from diagnostic ECG |
| IEC 60601-2-47 | Ambulatory electrocardiographic systems | Binding when claimed | Holter and patch performance expectations |
| ISO 80601-2-61 | Pulse oximeter equipment | Binding when claimed | SpO2 accuracy claims and test methods |
| IEEE 11073 | Point-of-care device communication | Voluntary; required by some integrations | Device and metric encoding |
| HL7 FHIR R5 Observation | Exchange of measurements and sampled data | Voluntary; required by interoperability programmes | `SampledData` carries origin, period, factor, dimensions |
| LOINC | Codes for measurements | Voluntary; required by many payers and registries | Heart rate, SpO2, respiratory rate codes |
| SNOMED CT | Clinical concepts and findings | Voluntary; licensed | Rhythm and finding semantics |
| FDA guidance, device software functions (2023) | Premarket documentation for software | Binding through the premarket review process | Software description, hazard analysis, verification |
| FDA AI-enabled device software guidance (2025) | Lifecycle management for AI/ML functions | Binding through review | Predetermined change control plans and monitoring |
| EU Medical Device Regulation 2017/745 | Clinical evaluation and post-market surveillance | Binding in the EU | Clinical evidence and PMS obligations |
| HIPAA Security Rule (US) | Protection of identifiable health data | Binding in the US | Encryption, access control, audit |

### 8.1 How the standards interact

The standards in the table do different jobs, and a compliance claim that mixes them is usually wrong.

- **Performance standards** (IEC 60601-2-25, 2-27, 2-47, ISO 80601-2-61) state what the device must achieve and how to test it. They are the basis of a safety and essential-performance claim.
- **Professional recommendations** (AHA/ACCF/HRS) state what a clinically adequate recording looks like. They are not law, but they are the reference a clinical reviewer uses.
- **Data standards** (FHIR, LOINC, IEEE 11073) state how the measurement is represented and exchanged. They do not state whether the measurement is accurate.
- **Regulatory guidance** (FDA, EU MDR) states what evidence a marketing application must contain. It does not specify filter cutoffs.

A device can be compliant with a performance standard and still produce a measurement that is unusable for a particular decision. A device can also be clinically adequate and not compliant with a standard it has not claimed. The two questions are separate: "does it meet the standard" and "does it support the decision."

### 8.2 FHIR SampledData in practice

The FHIR R5 `SampledData` datatype is the standard way to attach a waveform to an Observation. Its fields are easy to misread.

```text
SampledData
  origin    Quantity   the value that the first sample represents (baseline)
  period    decimal    time between samples, in milliseconds
  factor    decimal    multiplier applied to each sample
  lowerLimit decimal   lower bound of the sampled range
  upperLimit decimal   upper bound of the sampled range
  dimensions integer   number of samples per time point
  data      string     whitespace-separated values, with E for missing
```

The `period` is in milliseconds. A 500 Hz signal has a period of 2 ms, not 0.002. A pipeline that assumes seconds will interpret a 2 ms period as 2 s and resample the record by a factor of 1,000. This is a real integration failure, not a hypothetical one: the field is `decimal`, so no unit error is caught by a type system.

The `factor` and `origin` are also easy to drop. A waveform stored as integer codes with a factor of 0.1526 µV and an origin of 0 mV must be converted before any threshold is applied. A reader that ignores `factor` sees amplitudes in codes and compares them with millivolt thresholds.

### 8.3 Claims and the chain they require

| Claim | Measurement | Chain requirement | Evidence |
|---|---|---|---|
| General wellness heart rate | Mean rate over a window | Rhythm bandwidth, motion gating | Bench and human factors; no diagnostic claim |
| Heart rate for a clinical trial endpoint | Rate with defined aggregation | Validated against ECG; documented filters | Analytical and technical validation |
| Atrial fibrillation notification | Irregular rhythm plus confirmation | Single-lead ECG or PPG with quality control | Clinical validation in the intended population |
| ST-segment analysis | ST level at a defined fiducial | 0.05 Hz high-pass, 150 Hz low-pass | Diagnostic ECG performance standard |
| Sleep staging | Stage per 30 s epoch | EEG, EOG, EMG or validated surrogate | Agreement with polysomnography |

The table is a design aid, not a regulatory classification. The classification depends on the intended use, the population, and the claims language, which is a legal and regulatory question.

Three points about this table.

**The 500 samples/s figure is a storage and interchange recommendation, not a universal acquisition requirement.** A device may acquire at 1,000 samples/s or higher and decimate. The AHA statement discusses front-end oversampling to capture narrow pacemaker stimuli, and separately recommends at least 500 samples/s for the processed signal used in storage and exchange. Mixing the two numbers is a common error.

**IEC 60601-2-25 constrains accuracy, not a specific sampling rate.** The standard's measurement-accuracy tables and test data sets imply the bandwidth and resolution needed to meet them, but the standard does not simply say "sample at X Hz." A vendor can meet it with different internal architectures. When a vendor claims compliance, ask for the test report and the claimed performance, not just the certificate.

**FHIR `SampledData` is where many interoperability projects lose the units.** The `period` element is in milliseconds, `factor` scales the values, and `origin` gives the baseline. A pipeline that assumes period is in seconds will resample by a factor of 1,000. This is a concrete, checkable failure mode in real integrations.

The regulatory distinction that matters for this topic is between a **measurement claim** and a **wellness claim**. A product that reports "heart rate for general wellness" operates under enforcement discretion in the US, subject to the general wellness policy. A product that reports "detects atrial fibrillation" or "notifies users of hypertension" is making a medical claim and needs a device pathway. The signal chain is the same; the evidence around it is not. The acquisition parameters that support a wellness heart-rate display are not automatically adequate for a diagnostic claim.

## 9. Industry practice and case evidence

Public information about commercial signal chains is uneven. Device vendors publish little about filters; regulators publish summaries; academic groups publish reproducible pipelines. The following practices are documented or directly inferable from public sources, and the inference is labelled.

**PhysioNet datasets document their acquisition.** The MIT-BIH Arrhythmia Database is distributed at 360 samples/s with 11-bit resolution over a 10 mV range and a 0.1 Hz to 100 Hz analog bandwidth. PTB-XL is distributed at 500 samples/s, with a 100 samples/s version for some tasks. Sleep-EDF is distributed at 100 samples/s for the PSG channels. These parameters are in the dataset documentation and must be read before any cross-dataset model is trained. A model trained on 360 Hz MIT-BIH data and applied to 500 Hz PTB-XL data without resampling sees a different effective filter and a different timing grid. This is documented fact, not inference.

**The Apple Heart Study illustrates the difference between screening and diagnosis.** The study enrolled 419,297 participants. Among them, 2,161 (0.52%) received an irregular-rhythm notification. Among participants who returned ECG patches, 34% had atrial fibrillation confirmed. The study demonstrated feasibility at scale and also showed that a notification generates downstream confirmatory testing. The signal-chain lesson is that the tachogram quality and the artifact rejection determine the notification rate as much as the classifier does. The care-pathway lesson belongs to Topic 09. The figures are from the peer-reviewed publication.

**Consumer sleep devices differ in sensor set and staging logic.** Chinoy et al. compared seven consumer sleep-tracking devices with polysomnography and found variable agreement by device and by sleep stage. Devices that use only accelerometry cannot observe EEG features and therefore estimate sleep-wake rather than stage. Devices that add PPG can estimate autonomic proxies but still cannot measure cortical activity. The signal-chain lesson: the sensor set bounds the measurable construct. No amount of modelling recovers a stage that the sensor cannot observe. This is documented fact from the cited study.

**Remote monitoring programmes fail on workflow more than on signal quality.** The Cochrane review of structured telephone support and non-invasive telemonitoring for heart failure found modest and heterogeneous effects on readmission and mortality. The signal chain in these programmes is often simple (weight, blood pressure, symptoms), and the failure mode is alert fatigue and non-adherence rather than aliasing. The lesson is that a technically correct chain can still fail if the decision it feeds is not resourced.

**Wearable ECG clearance is narrow.** The Apple ECG app was cleared by the US Food and Drug Administration in 2018 for recording an ECG and indicating the presence of atrial fibrillation or normal sinus rhythm in symptomatic users. The clearance does not extend to asymptomatic screening or to other arrhythmias. Inference: the acquisition chain in a wrist device is adequate for a single-lead rhythm classification in the claimed population, and not adequate for 12-lead diagnostic interpretation. The distinction is regulatory, not merely technical.

**Sampling rates in consumer optical devices are often much lower than clinical standards.** A wrist PPG may sample at 25 Hz to 100 Hz to save power. At 25 Hz, Nyquist is 12.5 Hz, which is enough for heart rate at rest but marginal for pulse morphology and insufficient for the dicrotic notch. Inference: a PPG algorithm validated on 100 Hz data may not transfer to a 25 Hz device even if the sensor is otherwise identical.

**Data platforms increasingly preserve raw and processed streams.** Cloud pipelines for ambulatory monitoring commonly store the raw waveform, a processed waveform, and a feature table, with the filter parameters in metadata. This is the practice that makes reprocessing possible when a filter is found to be wrong. It is also the practice that most internal prototypes skip. Inference from vendor documentation and from published pipeline descriptions, not a universal rule.

### 9.1 What public challenges reveal about chain choices

The PhysioNet/Computing in Cardiology challenges are useful because the organisers publish the data, the scoring, and the rules, and the results show what chain choices survive contact with a held-out set.

- The 2020 challenge on 12-lead ECG classification distributed records at 500 Hz and required teams to handle variable record lengths and missing leads. A team that resampled to a single rate without checking the header produced a different signal from a team that preserved the native rate. The challenge scoring did not reward one choice directly, but the leaderboard separated them.
- The 2021 challenge on 12-lead, 6-lead, 3-lead, and 2-lead ECGs required the same model to work across reduced lead sets. A chain tuned to 12 leads and applied to 2 leads without adaptation lost information that no filter could restore. The lesson is that the sensor set bounds the measurable construct, which is the same lesson as the sleep-device example.
- The 2025 Chagas disease challenge used ECG to prioritise serological testing, with a top-5% ranking metric. The metric reflects a capacity constraint, not a chain choice, but the preprocessing still determined which records could be scored at all.

These are documented facts about the challenges, available on the challenge websites. The inference is that chain choices are evaluated implicitly by any benchmark, even when the benchmark does not name them.

### 9.2 Ambulatory and consumer sampling rates

Clinical ambulatory ECG (Holter) commonly records 2 or 3 leads at 200 Hz to 1,000 Hz, with a diagnostic bandwidth that depends on the device. Consumer ECG devices often record a single lead at a few hundred hertz for 30 s. Consumer PPG devices often sample at 25 Hz to 100 Hz. The differences are not arbitrary: they reflect the decision, the power budget, and the sensor.

A 25 Hz PPG has a Nyquist frequency of 12.5 Hz. At rest, the heart rate fundamental is around 1 Hz to 2 Hz and the first few harmonics carry the pulse shape. At 25 Hz, the second and third harmonics are preserved, but the dicrotic notch, which contains content above 10 Hz, is attenuated. A heart-rate algorithm can work; a pulse-morphology algorithm cannot. Inference: the sampling rate is a statement about which measurements the device can support, whether or not the vendor says so.

### 9.3 Documented fact versus inference

| Statement | Status |
|---|---|
| MIT-BIH Arrhythmia Database is distributed at 360 samples/s | Documented in the database header and PhysioNet documentation |
| PTB-XL is distributed at 500 samples/s | Documented in the PhysioNet dataset description |
| The Apple Heart Study enrolled 419,297 participants and 0.52% received a notification | Documented in the peer-reviewed publication |
| Among notified participants who returned patches, 34% had AF confirmed | Documented in the peer-reviewed publication |
| Consumer sleep devices differ by sensor set and stage agreement | Documented in the Chinoy et al. comparison |
| A 25 Hz PPG cannot support dicrotic-notch morphology | Inference from sampling theory and pulse-wave content |
| Most internal prototypes do not store filter metadata | Inference from common practice, not a measured statistic |
| A model trained at 360 Hz transfers poorly to 500 Hz without resampling | Inference from signal theory; specific transfer loss is study-dependent |

The table is deliberately short. The point is not to catalogue every claim but to make the labelling habit explicit. A report that mixes documented facts and inferences without labels is harder to audit than one that separates them.

### 9.4 Clinical monitoring vendors and the hidden chain

Bedside monitors from established vendors display ECG, PPG, respiration, and invasive pressures on a shared time base. Their signal chains are validated as part of the device clearance, and the filters are often selectable by mode: a monitoring mode with a 0.5 Hz to 40 Hz bandwidth, a diagnostic mode with a wider band, and a filter mode for pacemaker detection. The mode is part of the measurement, and a data export that does not record the mode cannot be reprocessed reliably.

This is inference from device manuals and from the structure of the relevant standards, not a claim about a specific product. The practical consequence is concrete: when a hospital exports monitor data for a research study, the export must include the mode and the filter settings, or the waveforms are not comparable across beds. Many retrospective studies discover this only after data collection.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Aliasing | Spectral peak at an unexpected frequency | Content above Nyquist folds down | Interference misdiagnosed; real content removed | Anti-alias filter; verify fs in header; spectral check |
| Sampling-rate mismatch | Heart rate off by a constant factor | File header says 500 Hz, pipeline assumes 250 Hz | Every interval and frequency scaled by 2 | Read header; assert units and rate at ingest |
| Unit error | Amplitudes 1,000 times too large or small | mV versus V versus µV | Thresholds invalid; saturation false alarms | Schema validation with units; range checks |
| Filter cutoff copied from another task | ST segment distorted in rhythm pipeline | 0.5 Hz high-pass removes low-frequency content | Measurement changes with filter, not physiology | Specify filter per decision; document |
| Causal filter delay | Fiducial times shifted by tens of ms | Nonlinear phase near cutoff | Interval bias if delay varies with rate | Zero-phase filtering for measurement; calibrate delay |
| Filter start transient | First 1 s of each epoch distorted | IIR filter initial conditions | Features at epoch edges biased | Filter continuous record; discard guard band |
| Clipping | Flat-topped peaks | ADC or amplifier rail | Amplitude and timing bias | Detect runs of identical codes; flag |
| Analog saturation | Plausible but truncated waveform | Front-end supply limit | Information lost before ADC | Compare amplitude distribution with front-end range |
| Lead-off | Rail or large step | Electrode disconnected | False asystole or false beat | Impedance flag; reject segment |
| Powerline interference | Narrowband peak at 50/60 Hz | Capacitive coupling | False detections; SNR loss | Notch; check harmonics; verify mains frequency |
| Motion artifact | Burst of broadband power with baseline shift | Electrode or sensor movement | Artifact labelled as arrhythmia | Accelerometer gate; quality index; reject |
| Dropped samples | Timestamp gap; derivative spike | Buffer overrun or radio loss | Timing error propagates | Sample counter check; record gap; interpolate short gaps |
| Clock drift | Growing offset versus reference | Oscillator tolerance | Interval error over long records | Drift model; resynchronise; report residual |
| Non-wear | Accelerometer stillness; optical absence | Device off body | Missingness treated as zero | Classify non-wear; do not impute as zero |
| Over-filtering | Loss of QRS notching or spike | Low-pass too low | Diagnostic content removed | Match bandwidth to task; keep raw |
| Under-filtering | Noisy baseline; false peaks | No high-pass or notch | Detector threshold unstable | Chain design; quality index |
| Resampling without anti-alias | New high-frequency content appears | Decimation by sample dropping | Alias in the resampled record | Low-pass before decimation |
| Interpolation of long gaps | Smooth but false waveform | Linear or spline fill over minutes | Fabricated physiology | Limit interpolation to short gaps; flag |

### 10.1 Audit questions for a signal pipeline

Before accepting a chain, ask six questions. Each maps to a row in the table.

1. What is the sampling rate, and where is it recorded? If the answer is "the default," the chain is not reproducible.
2. What is the anti-alias filter, and is it analog? If the answer is "a digital low-pass after sampling," aliasing is unaddressed.
3. What is the high-pass cutoff, and what decision does it protect? If the same cutoff serves rhythm and ST analysis, one of the two is wrong.
4. How are clipping, lead-off, and saturation detected? If the answer is "visual inspection," the chain does not scale.
5. How is the time base verified? If the answer is "the device timestamp," drift and dropped samples are unaddressed.
6. What happens to a rejected segment? If the answer is "it is dropped," the missingness mechanism must be reported.

Two red flags deserve emphasis because they are easy to miss in review.

A **filter described only by its cutoff** is not reproducible. The same 0.5 Hz cutoff with a first-order Butterworth, a fourth-order Butterworth, and a forward-backward filter produces three different ST segments. A methods section that says "band-pass filtered 0.5-40 Hz" without the family, order, and phase convention cannot be reproduced, and the reported measurement may not replicate.

A **dataset without a sampling-rate field** is not a signal dataset. It is a table of numbers. The absence of the field is a finding to report, not a gap to fill with an assumption.

A third red flag is **a quality rule with no false-negative analysis**. Every rejection rule has two error modes: keeping bad data and discarding good data. Teams usually measure the first and ignore the second. A rule that rejects 30% of a record may remove the very segments that contain the events of interest, particularly if artifact and disease co-occur. Report the rejection rate by outcome group, not only overall.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The signal is band-limited below Nyquist | Sampling and reconstruction theory | Sharp transients, pacemaker spikes, motion | Inspect spectrum; compare sampling rates |
| The anti-alias filter is analog and before the ADC | Digital filters cannot remove folded content | Digital-only anti-alias; decimation without filter | Review schematic; test with a tone sweep |
| The sampling rate in the header is correct | All timing and frequency calculations | Header error, firmware change, resampling | Count samples against a known-duration calibration signal |
| The ADC range and gain are constant | Amplitude thresholds and calibration | Automatic gain control; gain switching | Check gain log; amplitude histogram |
| Noise is stationary within an epoch | Filter design and quality thresholds | Motion bursts, electrode drying | Rolling SNR; quality index |
| The filter cutoff is below the physiological band of interest | Preserve the measurand | ST analysis with 0.5 Hz high-pass | Compare filters on the same record |
| Fiducial delay is constant or corrected | Interval measurement | Rate-dependent phase; different filter | Zero-phase comparison; calibration |
| Mains frequency is known and stable | Notch design | 50 Hz versus 60 Hz; wandering line frequency | Estimate from spectrum; check both |
| Missing samples are missing at random | Imputation and aggregation | Non-wear correlated with health state | Compare complete-case and imputed estimates |
| Quality thresholds transfer across devices | Automated rejection | Different sensors, firmware, populations | Re-validate threshold on target device |
| A feature computed on a filtered signal is comparable across devices | Pooled training and benchmarking | Different filter chains | Harmonise preprocessing; device-held-out test |
| Clock drift is negligible for the record length | Interval and spectral accuracy | Multi-day records; cross-device alignment | Compare timestamps with reference; measure ppm |
| The quality index is independent of the physiological state | Rejection is unbiased | Artifact correlated with illness or activity | Compare rejection rates by subgroup and outcome |
| The epoch is long enough for the feature | Spectral and variability estimates | Short windows for LF power | Compute the feature at two window lengths; check stability |
| The reference device measures the same construct | Agreement analysis | Different sensor site or bandwidth | Compare against two references; report disagreement |
| The epoch boundary is not a physiological event | Segmentation | A beat split across epochs | Use overlap or assign beats by R-peak time |
| The stored precision is sufficient for the feature | Feature stability | 8-bit export of a 16-bit record | Compare features from full and reduced precision |

The largest limit is scope. This topic describes a single-channel chain. Multi-channel records add inter-channel timing, montage, and reference issues that Topic 03 covers for EEG and Topic 02 covers for ECG leads. The second largest limit is that the procedure assumes the sensor is measuring what it claims. A PPG sensor measures light, not oxygen saturation; the conversion from light to SpO2 is a calibration model with its own assumptions. That belongs to Topic 04.

## 12. Exercises

### Exercise 1. Alias arithmetic (recall, 20 min)

**Deliverable:** a table of alias frequencies for 50 Hz, 60 Hz, 120 Hz, and 150 Hz at sampling rates of 100 Hz, 250 Hz, and 500 Hz.

**Acceptance criteria:** every entry computed with the fold rule; at least one entry identified as unaliased because it is below Nyquist; units in Hz.

**Time:** 20 minutes.

### Exercise 2. ADC budget (application, 30 min)

**Deliverable:** a one-page calculation for a 12-bit and a 16-bit ADC over a ±2.5 mV range, including LSB, codes for a 0.5 mV signal, and the theoretical SNR. State which configuration supports a 20 µV ST-segment change.

**Acceptance criteria:** correct arithmetic; a stated decision with justification; recognition that SNR assumes a full-scale sinusoid.

**Time:** 30 minutes.

### Exercise 3. Filter design for two decisions (application, 45 min)

**Deliverable:** two filter chains for the same ECG record, one for rhythm detection and one for ST analysis. For each, state high-pass cutoff and order, low-pass cutoff and order, notch, and phase convention.

**Acceptance criteria:** cutoffs justified by the measurement; no chain uses a cutoff that removes content the decision needs; the difference between the chains is explicit.

**Time:** 45 minutes.

### Exercise 4. Metadata audit (critique, 40 min)

**Deliverable:** a checklist of the ten metadata fields required to reproduce a heart-rate measurement, and an audit of one public dataset against the checklist.

**Acceptance criteria:** every field has a unit and a source; missing fields are named; the audit cites the dataset documentation.

**Time:** 40 minutes.

### Exercise 5. Quality rule (design, 45 min)

**Deliverable:** a decision rule that classifies a 10 s ECG epoch as accept, flag, or reject, using at least three measurable criteria.

**Acceptance criteria:** each criterion has a threshold and a rationale; the rule is deterministic; edge cases (one clipped beat, 2 s of motion) are handled.

**Time:** 45 minutes.

### Exercise 6. Reproduce a published chain (application, 60 min)

**Deliverable:** a short report attempting to reproduce the preprocessing of a paper that reports heart rate from a wearable, naming every parameter the paper omits.

**Acceptance criteria:** at least five omitted parameters identified; the effect of each on the measurement stated; no invented parameters.

**Time:** 60 minutes.

### Exercise 7. Clock-drift calculation (recall, 15 min)

**Deliverable:** the accumulated timing error for a 100 ppm clock over 1 s, 1 h, and 7 days, and the resulting error in a mean RR interval of 0.8 s.

**Acceptance criteria:** correct units; the distinction between absolute time error and interval error is stated.

**Time:** 15 minutes.

### Exercise 8. Chain design under constraint (design, 90 min)

**Deliverable:** a signal-chain specification for a 25 Hz wrist PPG that must report heart rate at rest and during walking, including sampling, filtering, quality, and the measurement's limits.

**Acceptance criteria:** Nyquist consequences stated; motion handling specified; the claim is limited to what 25 Hz can support; at least one alternative sampling rate considered.

**Time:** 90 minutes.

### Submission standard

For exercises that produce a calculation, submit a one-page PDF or Markdown file with the arithmetic shown, units on every number, and a short statement of the assumption that would change the answer. For design exercises, submit a two-page specification with a parameter table and a rationale column. For critique exercises, submit a numbered list of findings, each with a location in the source and a severity. Code is not required unless the exercise asks for it; the standard-library script in Section 7 is the reference implementation for the calculations.

A submission that reports a number without its unit, or a filter without its family and order, is returned without a grade. The standard is intentionally the same as the one used in Section 8 and Section 11.

## 13. Self-check answers

**1. What is the alias of 60 Hz at 100 Hz sampling?** 40 Hz. Reduce 60 modulo 100 to 60; 60 is above Nyquist (50), so the alias is 100 - 60 = 40 Hz. The amplitude is preserved.

**2. What is the alias of 150 Hz at 250 Hz sampling?** 100 Hz. 150 modulo 250 is 150; 150 is above 125, so the alias is 250 - 150 = 100 Hz.

**3. Why is a 500 samples/s storage rate recommended for diagnostic ECG?** The diagnostic bandwidth extends to 150 Hz. Nyquist requires more than 300 samples/s, and the recommendation adds margin for filter transition bands and measurement accuracy, yielding 500 samples/s. The number is a storage and interchange recommendation, not a universal acquisition minimum.

**4. What is the LSB of a 16-bit ADC over ±5 mV?** Full-scale span is 10 mV = 10,000 µV. 10,000 / 65,536 = 0.1526 µV per code.

**5. How many codes does a 1 mV QRS occupy at that resolution?** 1,000 µV / 0.1526 µV = 6,554 codes.

**6. What is the attenuation of a first-order high-pass at 0.5 Hz, evaluated at 0.25 Hz?** Gain 0.4469, which is -7.00 dB. The filter removes about 55% of the amplitude at half the cutoff.

**7. Why does a 0.5 Hz high-pass distort the ST segment?** The ST segment contains low-frequency content near and below 0.5 Hz. Attenuating that band changes the apparent ST level. The distortion depends on the filter order and phase, not only on the cutoff.

**8. What does a notch at 60 Hz with Q = 30 do at 59.5 Hz?** Gain 0.4837, or -6.31 dB. The notch is narrow; a small wander in mains frequency leaves a residual.

**9. Why is a causal filter a problem for interval measurement?** It introduces frequency-dependent delay. Near the cutoff, the delay changes with frequency, so fiducial points shift by an amount that depends on waveform shape and rate. Zero-phase filtering removes the phase distortion at the cost of non-causality.

**10. What is the frequency resolution of a 10 s window?** Approximately 1/T = 0.1 Hz for a rectangular window. A 30 s window gives about 0.033 Hz.

**11. What is the difference between a measurement, a feature, a biomarker, and a decision?** A measurement is a quantity with a unit and uncertainty. A feature is a computed descriptor. A biomarker is a feature with validated biological meaning. A decision is an action taken under uncertainty.

**12. How does a dropped sample affect timing?** If the gap is not recorded, every later sample is treated as if it occurred one sampling interval earlier. At 500 Hz, one dropped sample is a 2 ms shift. The effect on RR intervals is a constant bias unless the drop occurs between the two peaks being compared.

**13. Why is non-wear not the same as zero activity?** Non-wear means the sensor was not on the body, so no measurement exists. Zero activity is a measured value. Treating non-wear as zero biases activity and sleep estimates downward.

**14. What is the first check when a spectrum shows an unexpected peak?** Check the sampling rate and compute the alias of known sources such as mains frequency. A peak that moves when fs changes is an alias, not physiology.

**15. What is the minimum metadata contract for a heart-rate measurement?** Sampling rate, amplitude units and scale, sensor site, filter settings, epoch length, aggregation rule, device model and firmware, time base, and quality threshold.

**16. What is the difference between a notch filter and a band-stop filter?** A notch is a narrow band-stop filter, usually with a high Q, designed to remove a single frequency such as 50 Hz or 60 Hz. A band-stop filter can be wide and remove a band. The terms are often used interchangeably, but the width determines how much signal is lost.

**17. Why does a forward-backward filter change the effective cutoff?** Each pass applies the magnitude response once, so the combined response is the square of the one-pass response. A one-pass -3 dB point becomes -6 dB after two passes, and the effective -3 dB point of the combined filter is lower than the nominal cutoff. Report the convention.

**18. What is the difference between sampling jitter and clock drift?** Jitter is short-term variation in the interval between samples; drift is a slow, systematic divergence of the clock from a reference. Jitter degrades high-frequency spectral estimates; drift degrades long-record timing and cross-device alignment.

**Rubric for Exercise 5 (quality rule).** A strong answer defines three or more criteria with thresholds, states the order of evaluation, handles partial-epoch defects, and distinguishes flag from reject. A weak answer lists "noise" without a measurable definition.

**Rubric for Exercise 8 (chain design).** A strong answer states that 25 Hz supports heart rate at rest but not pulse morphology, specifies motion gating and a conservative claim, and considers 50 Hz as an alternative. A weak answer copies a 500 Hz ECG chain without adapting it.

## 14. Interview questions

**1. A colleague hands you a CSV of "heart rate" with no sampling rate. What do you do?**
A strong answer refuses to treat it as a signal, asks for the header or device export, and explains that without fs the intervals and frequencies are uninterpretable. A weak answer proceeds with descriptive statistics and assumes the values are in bpm.

**2. Explain aliasing to a product manager in two minutes.**
A strong answer uses the folding diagram, gives a concrete example (60 Hz at 100 Hz sampling becomes 40 Hz), and states the fix (analog anti-alias before sampling). A weak answer says "high frequencies get lost" without explaining folding.

**3. Why can a filter be correct for one task and wrong for another?**
A strong answer separates the measurand from the filter and gives the 0.5 Hz versus 0.05 Hz ECG example. A weak answer says "it depends on the data."

**4. How do you decide the sampling rate for a new sensor?**
A strong answer starts from the highest frequency the decision needs, adds margin for the anti-alias transition band, and checks power and storage. A weak answer picks the vendor default or the highest available rate.

**5. What is the difference between quantization noise and thermal noise?**
A strong answer describes quantization as deterministic given the signal and converter, thermal noise as random, and notes that both appear in the noise floor. A weak answer treats them as the same thing.

**6. When would you use zero-phase filtering, and when not?**
A strong answer uses it for offline interval and morphology measurement, avoids it for real-time detection, and notes the edge handling. A weak answer says "zero-phase is always better."

**7. How do you detect clipping without the raw ADC codes?**
A strong answer looks for repeated maxima, flat tops, and amplitude histograms with a spike at the rail, and notes that analog saturation can occur below the digital rail. A weak answer looks only at the maximum value.

**8. A model performs well in the lab and poorly at home. Where do you look first?**
A strong answer checks the signal chain: sampling, motion, placement, non-wear, and filter consistency, before blaming the model. A weak answer retrains the model on more data.

**9. What does a 50 ppm clock error mean for a 7-day record?**
A strong answer computes 50 × 10^-6 × 604,800 s = 30.24 s and explains the effect on cross-device alignment and interval trends. A weak answer says "small."

**10. Why is a 10 s window inadequate for frequency-domain HRV?**
A strong answer cites the 5 min convention for low-frequency power, the 1/T resolution limit, and the non-stationarity of heart rate. A weak answer says "it is too short" without a mechanism.

**11. What would you put in a dataset card for a physiological signal?**
A strong answer lists fs, units, sensor, filters, device, firmware, population, setting, quality rules, and known gaps. A weak answer lists only the number of records.

**12. How do you know your quality threshold is right?**
A strong answer validates it against adjudicated segments on the target device and reports sensitivity and specificity of the quality label. A weak answer uses a default from a library.

**13. A vendor says the device samples at 1,000 Hz and has a 150 Hz bandwidth. What else do you need to know?**
A strong answer asks for the anti-alias filter, the decimation ratio, the storage rate, and the digital filter settings. A 1,000 Hz acquisition rate does not imply a 1,000 Hz stored signal or a 150 Hz diagnostic bandwidth; the chain between the ADC and the file determines both. A weak answer accepts the two numbers as sufficient.

**14. When is a 10 s window preferable to a 5 min window?**
A strong answer uses 10 s when the decision is instantaneous rate or rhythm and the signal is non-stationary, and uses 5 min when the decision needs frequency-domain HRV or a stable average. A weak answer says "longer is better" without naming the feature and the stationarity assumption.

## 15. Key terms

- **Aliasing** — folding of content above Nyquist into lower frequencies.
- **Anti-alias filter** — analog low-pass before sampling that prevents folding.
- **Baseline wander** — low-frequency drift from respiration, motion, or perspiration.
- **Biomarker** — feature or measurement with validated biological meaning.
- **Clipping** — waveform limited by the ADC or amplifier rail.
- **Clock drift** — slow divergence of the device clock from reference time.
- **CMRR** — common-mode rejection ratio; ability to reject shared noise.
- **Cutoff frequency** — frequency at which a filter's gain is conventionally -3 dB.
- **Decision** — action taken under uncertainty.
- **Decimation** — reducing the sampling rate after low-pass filtering.
- **ENOB** — effective number of bits actually achieved.
- **Epoch** — fixed analysis window.
- **Feature** — computed descriptor of a signal.
- **Group delay** — frequency-dependent delay of a filter.
- **Jitter** — short-term variation in sample timing.
- **LSB** — voltage step represented by one ADC code.
- **Measurement** — quantity with unit and uncertainty.
- **Non-wear** — time the device is not on the body.
- **Notch filter** — narrow band-stop filter at a specified frequency.
- **Nyquist frequency** — half the sampling rate.
- **Powerline interference** — narrowband noise at 50 Hz or 60 Hz.
- **Quality index** — scalar summary of segment usability.
- **Quantization** — mapping continuous amplitude to finite codes.
- **Sampling rate** — samples per second.
- **Saturation** — amplifier output limited by its supply.
- **Spectral leakage** — energy spreading into neighbouring DFT bins.
- **Zero-phase filtering** — forward-backward filtering that cancels phase distortion.
- **Anti-alias margin** — distance between the signal band edge and Nyquist.
- **Baseline restore** — analog circuit that removes electrode offset.
- **Comb filter** — filter with notches at a fundamental and its harmonics.
- **Fiducial point** — waveform landmark used for timing.
- **Guard band** — signal discarded at an epoch boundary.
- **Headroom** — margin between the largest expected signal and the ADC rail.
- **Refractory period** — minimum interval between accepted detections.
- **Settling time** — time for a filter to recover from its initial state.
- **Transition band** — frequency range between passband and stopband.
- **Passband ripple** — gain variation within the passband.
- **Stopband attenuation** — gain reduction beyond the transition band.
- **ENOB** — effective number of bits after noise and distortion.
- **Wear detector** — sensor or logic that distinguishes on-body from off-body.
- **Alias frequency** — apparent frequency of folded content.
- **Sample counter** — device counter used to detect dropped samples.

## 16. Sources

All URLs accessed 25 October 2026.

1. Kligfield P, Gettes LS, Bailey JJ, et al. Recommendations for the standardization and interpretation of the electrocardiogram: Part I: The electrocardiogram and its technology. *Journal of the American College of Cardiology*. 2007;49(10):1109-1127. https://doi.org/10.1016/j.jacc.2007.01.024
2. American Heart Association. Circulation version of the same statement. https://doi.org/10.1161/CIRCULATIONAHA.106.180200
3. IEC 60601-2-25:2011, Medical electrical equipment — Part 2-25: Particular requirements for the basic safety and essential performance of electrocardiographs. https://webstore.iec.ch/en/publication/2636
4. ANSI/AAMI EC11, Diagnostic electrocardiographic devices. https://webstore.ansi.org/
5. PhysioNet. WFDB Programmer's Guide, multi-frequency records. https://physionet.org/physiotools/wpg/wpg_42.htm
6. PhysioNet. WFDB Applications Guide. https://physionet.org/physiotools/wag/wag.pdf
7. PhysioNet. MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/
8. PhysioNet. PTB-XL, a large publicly available electrocardiography dataset. https://physionet.org/content/ptb-xl/
9. PhysioNet. Sleep-EDF Database Expanded. https://physionet.org/content/sleep-edfx/
10. US Food and Drug Administration. Content of Premarket Submissions for Device Software Functions. 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
11. US Food and Drug Administration. Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations. 2025. https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
12. HL7. FHIR R5 Observation resource, SampledData datatype. https://hl7.org/fhir/R5/observation.html
13. Perez MV, Mahaffey KW, Hedlin H, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. *New England Journal of Medicine*. 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
14. Chinoy ED, Cuellar JA, Huwa KE, et al. Performance of seven consumer sleep-tracking devices compared with polysomnography. *Sleep*. 2021;44(5):zsaa291. https://doi.org/10.1093/sleep/zsaa291
15. Inglis SC, Clark RA, Dierckx R, et al. Structured telephone support or non-invasive telemonitoring for patients with heart failure. *Cochrane Database of Systematic Reviews*. 2015. https://doi.org/10.1002/14651858.CD007228.pub3
16. PhysioNet/Computing in Cardiology Challenge 2020. https://physionet.org/content/challenge-2020/
17. PhysioNet/Computing in Cardiology Challenge 2021. https://physionet.org/content/challenge-2021/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_ecg.md`](02_ecg.md)
- Related topic: [`06_feature-engineering.md`](06_feature-engineering.md)
- Related topic: [`10_deployment-on-device.md`](10_deployment-on-device.md)
- Module survey file: [`../../course/11_physiological_signals_wearables.md`](../../course/11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/) — see `11_ecg_quality.ipynb` for the ECG quality and R-peak lab.
- Related topic: [`08_digital-biomarkers.md`](08_digital-biomarkers.md)
- Dataset: [`https://physionet.org/content/mitdb/`](https://physionet.org/content/mitdb/)
