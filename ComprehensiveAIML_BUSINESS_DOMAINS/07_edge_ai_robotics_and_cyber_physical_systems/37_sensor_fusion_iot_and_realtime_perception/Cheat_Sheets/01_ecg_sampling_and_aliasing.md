---
collection: sensor-fusion-iot-realtime-perception
file_id: '01'
topic: ECG sampling and aliasing
level: 1
status: draft
version: '1.0'
created: '2026-11-07'
reviewed: null
review_trigger: Changes to dataset documentation, sampling assumptions, or intended use
notation_style: plain-text
prerequisites: Basic algebra, sinusoids, and SI unit conversion
source_status: OPEN
owner_role: Course maintainer
source_index: INDEX.md
---

# ECG Sampling and Aliasing

## 1. File Header

**Purpose:** Determine whether a sampling and resampling plan preserves a declared signal band, and diagnose plausible aliasing without treating a spectral peak as proof of its source.

**Audience:** Engineers and analysts beginning biomedical signal processing.

**Prerequisites:** Rates, reciprocals, basic sinusoids, decibels, and array indexing.

**Outcome:** Given a sample rate, target band, interfering frequency, and segment length, compute time spacing, folding, and DFT bin spacing, then identify the necessary filtering and evidence gaps.

**Exclusions:** Clinical diagnosis, device validation, clinical bandwidth recommendations, patient treatment, and the design certification of a medical device.

**Status:** Draft; not reviewed. No experimental or device validation has been performed.

**Provenance:** The HighEnd Module01 sampling title is a topic commitment in the [directory manifest](../../../../Templates/directory.txt); its listing alone is not evidence that the underlying source lesson was inspected. The course [crosswalk](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#edge-signals) maps it to Module 37. Dataset facts are sourced separately below. All constructed signals and calculations are illustrative, not measurements from a patient or device.

**Source:** Moody and Mark, [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/), PhysioNet, published 2005, consulted 2026-11-07. It reports 48 approximately half-hour, two-channel recordings from 47 subjects; original analog recordings were digitized at 360 samples/s per channel, 11-bit resolution across a 10 mV range. It does not establish the acquisition specifications of a new device.

**Retrieval cue:** An unexpected low-frequency component after a sample-rate change: start with Sections 4, 7, and 8.

**Takeaway:** A rate alone is not evidence that the acquired ECG band was preserved.

## 2. Concept Map / Mental Model

```text
Continuous ECG plus interference
  -> analog front end: gain, range, analog anti-alias filter
  -> ADC aperture and quantization at actual sample times
  -> timestamped discrete samples and acquisition metadata
  -> optional digital low-pass filter BEFORE reducing sample rate
  -> decimation / resampling with updated time base
  -> finite window and DFT for inspection
  -> diagnostic decision with uncertainty and evidence
```

A filter after the ADC cannot separate two analog sinusoids that already produced identical samples.
A filter before a decimator can suppress components that would otherwise fold at the lower output rate.
The physical front end and the later digital pipeline solve related but distinct problems.
If timestamps are irregular, a standard uniformly sampled DFT needs additional justification.
A spectrum is not a direct measurement of the original analog spectrum unless the chain is characterized.

**Takeaway:** Put each anti-alias filter before the sampling operation whose folding it must prevent.

## 3. Foundational Definitions

| Term | Meaning | Units and boundary |
|---|---|---|
| ECG | Electrical potential difference measured by a specified lead | Typically expressed in mV; lead configuration matters |
| Analog input | Continuous-time voltage entering the acquisition chain | V or mV; not already digitized |
| Sample | Voltage representation at one acquisition instant | A discrete value, not a time interval |
| Sampling rate `fs` | Samples per second per channel | Hz; positive |
| Sampling period `Ts` | Time between uniform adjacent samples | seconds/sample; `1/fs` |
| Nyquist frequency | Half of the uniform sample rate | Hz; not a filter cutoff |
| Signal bandwidth | Frequency range required by the declared task | Hz; not a universal ECG constant |
| Alias | Indistinguishable discrete-time representation of different continuous frequencies | Depends on sample rate |
| Analog anti-alias filter | Pre-ADC filter attenuating out-of-band analog energy | Finite transition and stopband |
| Decimation | Retaining samples at a lower rate after suitable filtering | Factor must be stated |
| DFT | Discrete Fourier transform of a finite sequence | Its bin separation depends on `fs` and `N` |
| Quantization | Mapping an input amplitude to an ADC code | Error is distinct from frequency aliasing |
| Clipping | Input or intermediate value exceeds available range | Nonlinear distortion; not fixed by raising `fs` |
| Jitter | Variation of sample instants from their ideal grid | Time units; effects depend on signal slope |
| Gain | Scale converting physical voltage to stored units | Use the record header, not a guessed constant |

For zero-based sample `n`, uniform time relative to the first sample is `n/fs` seconds.
A vector of `N` samples occupies `N/fs` seconds in the usual half-open analysis window, while the first-to-last timestamp span is `(N-1)/fs`.
These are different questions, not conflicting answers.
The rate of a two-channel stream is 360 samples/s **per channel**, not 180 per channel.
A total count across channels is not a sample rate for either individual waveform.

**Takeaway:** Every rate, count, amplitude, and time must name its unit and channel convention.

## 4. Core Formulas with Full Derivation

### 4.1 Uniform sampling and sample spacing

Validity: An ideal uniform grid is assumed; gaps and clock drift must be handled separately.
Derivation: `fs` samples in one second imply one sample every `1/fs` seconds.
**FORMULA:** `Ts = 1/fs`; `t[n] = t0 + n/fs`.
**WHERE:** `fs` is Hz, `Ts` is seconds, `n` is a zero-based integer, and `t0` is the first sample time.
**CONSTRAINT:** `fs > 0` and actual acquisition times match the grid sufficiently for the intended analysis.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** At 360 Hz, `Ts = 1/360 s = 2.777... ms`.
**INTERPRETATION:** A single index step corresponds to about 2.78 ms under this assumed time base.
Boundary: A missing frame breaks the mapping if the array is silently compacted.

### 4.2 Nyquist and the band-limit condition

Validity: Ideal reconstruction requires a uniformly sampled signal with known band limitation and ideal reconstruction conditions.
Derivation sketch: Sampling creates spectral replicas spaced by `fs`; a band limited to below `B` avoids overlap if `fs > 2B`.
**FORMULA:** `fN = fs/2`; choose `fs > 2B` for a strictly bounded baseband of maximum frequency `B`.
**WHERE:** `fN`, `fs`, and `B` are in Hz; `B` is a task-specific upper frequency.
**CONSTRAINT:** Strict separation, negligible out-of-band energy after the analog filter, and an appropriate uniform sampling model.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** `fs = 360 Hz` gives `fN = 180 Hz`; a hypothetical `B = 40 Hz` leaves 140 Hz between the band edge and Nyquist.
**INTERPRETATION:** The margin allows a transition band; it does not by itself specify sufficient stopband attenuation.
Boundary: An exact Nyquist-frequency sinusoid has phase ambiguity; equality at `fs = 2B` is not a robust design margin.
Boundary: Real filters do not create a brick-wall band limit and actual interference may be arbitrarily strong.

### 4.3 Folding a sinusoid into the first Nyquist zone

Validity: This scalar folding rule applies to a real sinusoid sampled uniformly at a known constant rate.
Derivation: Adding any integer multiple of `fs` changes sampled phase by an integer multiple of `2π`; cosine's positive and negative frequencies also coincide up to phase.
**FORMULA:** `fa = |f - k fs|`, with integer `k` chosen so `0 <= fa <= fs/2`.
**WHERE:** `f` is the original sinusoidal frequency in Hz, `fa` is its nonnegative alias frequency in Hz, and `k` is an integer.
**CONSTRAINT:** The rule predicts sampled frequency equivalence; it does not establish actual amplitude, phase, or the origin of an observed peak.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** At `fs = 360 Hz`, `f = 250 Hz` and `k = 1` give `fa = |250 - 360| = 110 Hz`.
**INTERPRETATION:** A 250 Hz sinusoid and a suitable 110 Hz sinusoid produce indistinguishable sampled frequencies.
Boundary: Folding may invert phase; two arbitrary sinusoids need appropriate phase and amplitude to yield identical samples.
Boundary: A nonstationary or broadband interferer folds as a spectrum, not necessarily into one peak.

### 4.4 DFT grid and segment length

Validity: A length-`N` uniformly sampled sequence uses the ordinary DFT grid.
Derivation: Its periodic extension has period `N/fs` seconds, so adjacent discrete frequencies differ by its reciprocal.
**FORMULA:** `Δf = fs/N = 1/T`, where `T = N/fs`.
**WHERE:** `N` is the number of samples per channel, `T` is the half-open window duration in seconds, and `Δf` is Hz.
**CONSTRAINT:** This is bin spacing, not a guarantee of resolving two tones; window shape, signal-to-noise ratio, and separation matter.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** `N = 3600` at `fs = 360 Hz` gives `T = 10 s` and `Δf = 0.1 Hz`.
**INTERPRETATION:** The DFT grid is spaced at 0.1 Hz for this window.
Boundary: Zero-padding interpolates the displayed grid but does not create the resolving power of a longer observation.

### 4.5 Decimation and output bandwidth

Validity: Integer decimation by `M` requires appropriate digital filtering before retaining every `M`th sample.
Derivation: One output sample per `M` input periods makes the output period `M/fs` and hence rate `fs/M`.
**FORMULA:** `fs_out = fs_in/M`, `fN_out = fs_in/(2M)`.
**WHERE:** `M` is a positive integer, rates are Hz, and input and output refer to each channel separately.
**CONSTRAINT:** Suppress frequencies that would fold into the desired output band; allocate finite transition width and specify allowable error.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** `360 Hz / 2 = 180 Hz`, with output Nyquist `90 Hz`.
**INTERPRETATION:** An unfiltered 110 Hz component at the input maps to 70 Hz after this rate reduction.
Boundary: Filtering after the decimation cannot uniquely recover the original 70 Hz component from its 110 Hz alias.

### 4.6 Attenuation and amplitude ratios

Validity: Decibels below are for a voltage/amplitude ratio, not a power ratio.
Derivation: Power is proportional to squared amplitude under a common impedance; `10 log10(A²) = 20 log10(A)`.
**FORMULA:** `L_dB = 20 log10(Aout/Ain)`; attenuation magnitude is `-L_dB` for a gain below one.
**WHERE:** `Ain` and `Aout` are positive amplitudes in the same units.
**CONSTRAINT:** The ratio is a linear, frequency-specific gain; characterize the complete actual filter.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** A gain of `0.01` gives `20 log10(0.01) = -40 dB`.
**INTERPRETATION:** A 1 mV sinusoid would emerge as 0.01 mV under that idealized gain at that frequency.
Boundary: A nominal cutoff does not tell you the attenuation at every aliasing frequency.

### 4.7 Sampling phase and reflected aliases

Validity: A single real sinusoid is observed on a uniform grid; no time-varying clock is assumed.
Derivation: Substitute `t=n/fs` into `cos(2πft+φ)` and write `f=k fs-fa`.
The integer term contributes `2πkn`, which leaves cosine unchanged.
Cosine is even, so the residual becomes `cos(2π fa n/fs-φ)`.
**FORMULA:** `cos(2π(k fs-fa)n/fs+φ) = cos(2π fa n/fs-φ)`.
**WHERE:** `k` and `n` are integers, `fa` and `fs` are Hz, and `φ` is radians.
**CONSTRAINT:** This reflected form assumes uniform sample times and an ideal scalar sinusoid.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** At 360 Hz, a 250 Hz cosine with phase 0.3 rad has the same samples as a 110 Hz cosine with phase -0.3 rad.
**INTERPRETATION:** Frequency reflection also changes phase; equal-frequency components can reinforce or cancel.
Boundary: A sine represented with an unchanged phase is not generally identical after reflection.

### 4.8 A finite-difference timing sensitivity

Validity: A differentiable analog waveform is sampled with a small timing perturbation relative to its local curvature timescale.
Derivation: The first-order Taylor expansion of `x(t+δt)` is `x(t)+x′(t)δt`.
**FORMULA:** `δx ≈ x′(t)δt`.
**WHERE:** `δx` is voltage, `x′` is volts per second, and `δt` is seconds.
**CONSTRAINT:** The neglected higher-order terms must be small; deterministic and random errors require different summaries.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** A slope of 0.5 V/s and timing error of 10 µs give about 5 µV error.
**INTERPRETATION:** The same clock error affects a steep transition more than a flat segment.
Boundary: A clipped waveform is not made accurate by calculating this local linear estimate.

### 4.9 Quantization spacing versus effective performance

Validity: An ideal uniform converter has `2^b` equal bins over a specified full-scale range.
Derivation: Divide the total input range by the number of quantization intervals.
**FORMULA:** `q = Vrange/2^b`.
**WHERE:** `q` is volts per code, `Vrange` is volts, and `b` is a positive integer bit count.
**CONSTRAINT:** This ideal bin-width calculation excludes noise, offset, nonlinearities, saturation and front-end gain.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** A 10 mV range divided into 2048 bins gives approximately 4.883 µV per bin.
**INTERPRETATION:** Nominal code spacing is a unit conversion, not a claim of 4.883 µV measurement accuracy.
Boundary: Actual code-to-voltage conversion must use the applicable record or device calibration.

### 4.10 Time-base drift and accumulated timing error

Validity: Compare two constant-rate clocks over the same sample count, with no gaps.
Derivation: Divide `N` by each sample rate and subtract their durations.
**FORMULA:** `ΔT = N/factual - N/fnominal`.
**WHERE:** `N` is samples, both rates are samples per second, and `ΔT` is seconds.
**CONSTRAINT:** Rates are constant; a drifting clock needs a time-dependent model or timestamp measurements.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** 360,000 samples at 360.036 Hz span a half-open duration of about 999.900 s, versus 1000 s at 360 Hz.
**INTERPRETATION:** A 100 ppm fast clock produces about 0.100 s discrepancy over this nominal interval.
Boundary: The first-to-last timestamp uses `N-1`, not `N` intervals.

### 4.11 Frequency-dependent FIR behavior

Validity: A fixed, causal, finite impulse response filter is applied to uniformly spaced input samples.
Derivation: Substitute a complex sinusoid into the convolution sum and factor out its time-dependent term.
**FORMULA:** `H(f) = Σ h[k] exp(-j 2π f k/fs)` for `k=0,...,L-1`.
**WHERE:** `h[k]` are dimensionless coefficients, `L` is tap count, and `H` is complex gain.
**CONSTRAINT:** The filter must be linear and time invariant over the analyzed interval.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** A two-tap average `[0.5, 0.5]` at 360 Hz has gain 1 at 0 Hz and 0 at 180 Hz.
**INTERPRETATION:** A zero at one frequency does not establish sufficient attenuation throughout a stopband.
Boundary: Coefficient quantization and implementation arithmetic can alter the actual response.

### 4.12 Two-tap average at an intermediate frequency

Validity: Use the ideal two-tap average above and a sinusoid at a fixed frequency.
Derivation: Factor the response into a phase term times `cos(π f/fs)`.
**FORMULA:** `|H(f)| = |cos(π f/fs)|`.
**WHERE:** `f` and `fs` are in Hz; the magnitude is a dimensionless amplitude ratio.
**CONSTRAINT:** Both coefficients must equal 0.5 and the filter must operate at the stated input rate.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** At 90 Hz with `fs=360 Hz`, gain is `cos(π/4) ≈ 0.707`, or approximately -3.01 dB.
**INTERPRETATION:** This simple smoother changes frequencies well inside the original Nyquist band.
Boundary: It is not automatically an adequate anti-alias filter for a 180 Hz output.

### 4.13 Allowable interference and rejection

Validity: Apply this budget to one sinusoidal interferer and a linear filter at its frequency.
Derivation: Require output amplitude no greater than the specified maximum, then take the amplitude-ratio logarithm.
**FORMULA:** `Arequired_dB = 20 log10(Ain/Amax)` when `Ain > Amax > 0`.
**WHERE:** `Ain` and `Amax` are amplitudes in identical units; the result is positive attenuation in dB.
**CONSTRAINT:** This budget excludes other interference, tolerances, quantization, and nonlinear saturation.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** A 2 mV interferer limited to 0.02 mV requires at least 40 dB attenuation at that frequency.
**INTERPRETATION:** Specify rejection from the input amplitude and allowable output error rather than using a cutoff alone.
Boundary: Multiple coherent interferers may add in amplitude; their combined contribution needs a separate budget.

### 4.14 Coherent addition after folding

Validity: Two sinusoids have identical discrete frequency and fixed phase difference during observation.
Derivation: Add their complex phasors and take the magnitude of the resulting vector.
**FORMULA:** `Aout = sqrt(A1²+A2²+2 A1 A2 cos(Δφ))`.
**WHERE:** `A1` and `A2` are nonnegative amplitudes in the same unit and `Δφ` is radians.
**CONSTRAINT:** The frequencies coincide after sampling; neither component is clipped.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** Two 0.1 mV components with equal phase produce 0.2 mV; opposite phases produce zero.
**INTERPRETATION:** A missing line does not establish that neither interfering component was present.
Boundary: Independent random-phase interference requires an appropriate statistical model rather than guaranteed coherent cancellation.

### 4.15 Sample-count conversion across rational rates

Validity: A finite duration is resampled between constant rates with a specified time origin and endpoint convention.
Derivation: Multiply the duration by each rate, then take their ratio before endpoint rounding.
**FORMULA:** `Nout ≈ Nin × fs_out/fs_in`.
**WHERE:** Counts are per channel and rates are samples per second per channel.
**CONSTRAINT:** Exact output count depends on the chosen inclusive or half-open endpoints and implementation.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** A 10 s half-open interval at 360 Hz has 3600 inputs; at 250 Hz it has 2500 outputs.
**INTERPRETATION:** The conversion is a time-grid mapping, not permission to discard arbitrary samples.
Boundary: Rate conversion still needs appropriate interpolation and anti-alias filtering.

### 4.16 Frequency normalization for filter design

Validity: Use this conversion only when the particular library defines normalized frequency relative to Nyquist.
Derivation: Divide the physical frequency by `fs/2`.
**FORMULA:** `w = f/(fs/2)`.
**WHERE:** `w` is dimensionless and `f`, `fs` are Hz.
**CONSTRAINT:** The library convention must be checked; some APIs accept Hz or cycles per sample instead.
**NUMERICAL EXAMPLE — ILLUSTRATIVE:** A 40 Hz edge at 360 Hz corresponds to `40/180 = 2/9 ≈ 0.2222` of Nyquist.
**INTERPRETATION:** Passing 40 as a normalized number to such an API is invalid, not a 40 Hz specification.
Boundary: A valid number alone does not imply a suitable filter order or stopband.

**Takeaway:** The equations require a uniform clock, a declared band, and known filtering; a Nyquist number alone is insufficient.

## 5. Worked Numerical Example

### 5.1 Record-independent 360 Hz segment

**ILLUSTRATIVE setup:** Two channels; each has 3600 consecutive samples acquired at exactly 360 Hz.
The choice of 360 Hz reflects the cited dataset rate, but these samples are hypothetical and no MIT-BIH waveform was loaded.
**Question:** What are the sample spacing, observation duration, time span, Nyquist frequency, and DFT spacing?
1. Input: `fs = 360 samples/s/channel`; `N = 3600 samples/channel`.
2. Sample interval: `1/360 s = 2.777... ms`.
3. Half-open analysis duration: `N/fs = 10 s`.
4. Last index: `N - 1 = 3599`.
5. Last timestamp relative to the first: `3599/360 = 9.997222... s`.
6. Nyquist frequency: `360/2 = 180 Hz`.
7. DFT grid spacing: `360/3600 = 0.1 Hz`.
8. Total stored sample values across both channels: `2 × 3600 = 7200`.
9. Independent check: `10 s × 360 samples/s = 3600 samples` per channel.
10. Independent check: `1/(10 s) = 0.1 Hz`.
11. Decision: Record the 10 s window and time origin; do not label 180 Hz a validated analog passband.
12. Limit: An actual waveform may contain missing data, tape artifacts, or other record-specific concerns.

### 5.2 Change one assumption: downsample without filtering

Start from a hypothetical 110 Hz sinusoid represented in the 360 Hz sequence.
Take every second sample without a digital low-pass filter.
The output rate is `180 Hz`, with output Nyquist `90 Hz`.
The output alias is `|110 - 180| = 70 Hz`.
A genuine 70 Hz tone may already be present and combines with the alias according to phase.
The lower-rate samples alone cannot uniquely separate the two contributions.
A suitable pre-decimation filter would attenuate the 110 Hz input before sample removal, subject to its specified response.
If the desired band instead included 110 Hz, 180 Hz output would not preserve that band; retain a higher rate or change the requirement.

### 5.3 Distinguish an analog and a digital fold

A hypothetical 250 Hz analog sinusoid sampled directly at 360 Hz folds to 110 Hz.
If that sampled sequence is then reduced to 180 Hz without filtering, the 110 Hz component folds again to 70 Hz.
A digital filter before decimation could attenuate the sampled 110 Hz component.
It cannot tell whether that component originated at 110 Hz or was an analog 250 Hz alias.
Preventing the first fold requires sufficient attenuation before the original ADC.

### 5.4 Additional folding exercises

Each case is a constructed single-tone example; no recording or device was tested.
Apply the Section 4.3 folding rule and distinguish the input sampling boundary from any later rate change.

#### Above the first Nyquist edge
- Given: sampling rate `360 Hz` per channel and input tone `190 Hz`.
- Compute the Nyquist frequency: `360/2 = 180 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 360 = 360 Hz`.
- Compute the distance: `|190 - 360| = 170 Hz`.
- Check: `170 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `170 Hz` under the ideal uniform-grid model.
- Context: A tone just above 180 Hz reflects just below the edge.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### One full sampling period above zero
- Given: sampling rate `360 Hz` per channel and input tone `360 Hz`.
- Compute the Nyquist frequency: `360/2 = 180 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 360 = 360 Hz`.
- Compute the distance: `|360 - 360| = 0 Hz`.
- Check: `0 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `0 Hz` under the ideal uniform-grid model.
- Context: A sinusoid at exactly the rate appears constant, with amplitude depending on phase.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Above one full sampling rate
- Given: sampling rate `360 Hz` per channel and input tone `400 Hz`.
- Compute the Nyquist frequency: `360/2 = 180 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 360 = 360 Hz`.
- Compute the distance: `|400 - 360| = 40 Hz`.
- Check: `40 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `40 Hz` under the ideal uniform-grid model.
- Context: An out-of-band tone can land directly in a 40 Hz analysis band.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Near the second multiple
- Given: sampling rate `360 Hz` per channel and input tone `710 Hz`.
- Compute the Nyquist frequency: `360/2 = 180 Hz`.
- Choose the nearest sampling-rate multiple: `2 × 360 = 720 Hz`.
- Compute the distance: `|710 - 720| = 10 Hz`.
- Check: `10 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `10 Hz` under the ideal uniform-grid model.
- Context: High-frequency energy need not remain near the edge after sampling.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Second multiple plus offset
- Given: sampling rate `360 Hz` per channel and input tone `740 Hz`.
- Compute the Nyquist frequency: `360/2 = 180 Hz`.
- Choose the nearest sampling-rate multiple: `2 × 360 = 720 Hz`.
- Compute the distance: `|740 - 720| = 20 Hz`.
- Check: `20 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `20 Hz` under the ideal uniform-grid model.
- Context: The alias is 20 Hz even though the input exceeds two clock cycles per sample.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Lower output rate
- Given: sampling rate `180 Hz` per channel and input tone `140 Hz`.
- Compute the Nyquist frequency: `180/2 = 90 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 180 = 180 Hz`.
- Compute the distance: `|140 - 180| = 40 Hz`.
- Check: `40 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `40 Hz` under the ideal uniform-grid model.
- Context: A 140 Hz component in an input sequence becomes indistinguishable from 40 Hz after unfiltered rate reduction.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Threefold decimation
- Given: sampling rate `120 Hz` per channel and input tone `80 Hz`.
- Compute the Nyquist frequency: `120/2 = 60 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 120 = 120 Hz`.
- Compute the distance: `|80 - 120| = 40 Hz`.
- Check: `40 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `40 Hz` under the ideal uniform-grid model.
- Context: A 360 Hz input reduced by three has a 60 Hz output Nyquist edge.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Quarter-rate output
- Given: sampling rate `90 Hz` per channel and input tone `70 Hz`.
- Compute the Nyquist frequency: `90/2 = 45 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 90 = 90 Hz`.
- Compute the distance: `|70 - 90| = 20 Hz`.
- Check: `20 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `20 Hz` under the ideal uniform-grid model.
- Context: A 70 Hz input component folds to 20 Hz at 90 Hz output.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Frequency below output Nyquist
- Given: sampling rate `180 Hz` per channel and input tone `40 Hz`.
- Compute the Nyquist frequency: `180/2 = 90 Hz`.
- Choose the nearest sampling-rate multiple: `0 × 180 = 0 Hz`.
- Compute the distance: `|40 - 0| = 40 Hz`.
- Check: `40 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `40 Hz` under the ideal uniform-grid model.
- Context: A 40 Hz component remains at 40 Hz, subject to filter gain and phase.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Frequency at output Nyquist
- Given: sampling rate `180 Hz` per channel and input tone `90 Hz`.
- Compute the Nyquist frequency: `180/2 = 90 Hz`.
- Choose the nearest sampling-rate multiple: `0 × 180 = 0 Hz`.
- Compute the distance: `|90 - 0| = 90 Hz`.
- Check: `90 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `90 Hz` under the ideal uniform-grid model.
- Context: An exact-edge sinusoid has phase ambiguity and is a poor design boundary.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Mains-like synthetic interference
- Given: sampling rate `100 Hz` per channel and input tone `60 Hz`.
- Compute the Nyquist frequency: `100/2 = 50 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 100 = 100 Hz`.
- Compute the distance: `|60 - 100| = 40 Hz`.
- Check: `40 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `40 Hz` under the ideal uniform-grid model.
- Context: This is a constructed 60 Hz test tone, not an attribution of a measured line.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Frequency above two output rates
- Given: sampling rate `100 Hz` per channel and input tone `230 Hz`.
- Compute the Nyquist frequency: `100/2 = 50 Hz`.
- Choose the nearest sampling-rate multiple: `2 × 100 = 200 Hz`.
- Compute the distance: `|230 - 200| = 30 Hz`.
- Check: `30 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `30 Hz` under the ideal uniform-grid model.
- Context: The nearest multiple is 200 Hz; folding is not restricted to one replica.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### A low-rate clock
- Given: sampling rate `50 Hz` per channel and input tone `48 Hz`.
- Compute the Nyquist frequency: `50/2 = 25 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 50 = 50 Hz`.
- Compute the distance: `|48 - 50| = 2 Hz`.
- Check: `2 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `2 Hz` under the ideal uniform-grid model.
- Context: A high input close to the sampling rate produces a slowly varying sequence.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### A half-rate tone
- Given: sampling rate `100 Hz` per channel and input tone `50 Hz`.
- Compute the Nyquist frequency: `100/2 = 50 Hz`.
- Choose the nearest sampling-rate multiple: `0 × 100 = 0 Hz`.
- Compute the distance: `|50 - 0| = 50 Hz`.
- Check: `50 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `50 Hz` under the ideal uniform-grid model.
- Context: Its sample values depend strongly on phase at the Nyquist boundary.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### A low-band interferer
- Given: sampling rate `250 Hz` per channel and input tone `260 Hz`.
- Compute the Nyquist frequency: `250/2 = 125 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 250 = 250 Hz`.
- Compute the distance: `|260 - 250| = 10 Hz`.
- Check: `10 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `10 Hz` under the ideal uniform-grid model.
- Context: An out-of-band component can alias to a plausible slow variation.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### An edge near a new rate
- Given: sampling rate `250 Hz` per channel and input tone `240 Hz`.
- Compute the Nyquist frequency: `250/2 = 125 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 250 = 250 Hz`.
- Compute the distance: `|240 - 250| = 10 Hz`.
- Check: `10 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `10 Hz` under the ideal uniform-grid model.
- Context: A decimation or conversion stage must reject such energy before rate change.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### Different original rates
- Given: sampling rate `500 Hz` per channel and input tone `460 Hz`.
- Compute the Nyquist frequency: `500/2 = 250 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 500 = 500 Hz`.
- Compute the distance: `|460 - 500| = 40 Hz`.
- Check: `40 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `40 Hz` under the ideal uniform-grid model.
- Context: The same observed 40 Hz could arise from an analog 460 Hz sinusoid.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### A high multiple at 500 Hz
- Given: sampling rate `500 Hz` per channel and input tone `1010 Hz`.
- Compute the Nyquist frequency: `500/2 = 250 Hz`.
- Choose the nearest sampling-rate multiple: `2 × 500 = 1000 Hz`.
- Compute the distance: `|1010 - 1000| = 10 Hz`.
- Check: `10 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `10 Hz` under the ideal uniform-grid model.
- Context: The second replica folds a 1010 Hz component to 10 Hz.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### A 200 Hz sampled sequence
- Given: sampling rate `200 Hz` per channel and input tone `130 Hz`.
- Compute the Nyquist frequency: `200/2 = 100 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 200 = 200 Hz`.
- Compute the distance: `|130 - 200| = 70 Hz`.
- Check: `70 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `70 Hz` under the ideal uniform-grid model.
- Context: A real 70 Hz input and reflected 130 Hz component overlap in the samples.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

#### A second reflected case
- Given: sampling rate `200 Hz` per channel and input tone `170 Hz`.
- Compute the Nyquist frequency: `200/2 = 100 Hz`.
- Choose the nearest sampling-rate multiple: `1 × 200 = 200 Hz`.
- Compute the distance: `|170 - 200| = 30 Hz`.
- Check: `30 Hz` is within the nonnegative Nyquist interval.
- Interpretation: the sampled frequency is `30 Hz` under the ideal uniform-grid model.
- Context: The phase of a reflected cosine changes sign relative to its positive-frequency representation.
- The observed magnitude also depends on the analog or digital filter at this boundary.
- The observed phase depends on input phase, folding direction, and time origin.
- A matching output peak is not proof that this particular input frequency existed.
- Preserve the higher-rate or pre-ADC evidence when the origin matters.

### 5.5 Worked digital-filter budgets

All amplitudes are illustrative sinusoidal peak amplitudes; these are requirements, not measured filter responses.

#### Budget 1
- Given input amplitude `1 mV` and permitted output amplitude `0.1 mV`.
- Divide input by allowed residual: `1/0.1 = 10`.
- Use the amplitude-ratio equation in Section 4.13: `20 log10(10) = 20 dB`.
- The required gain is at most `0.1` at the interference frequency.
- Multiplying the input amplitude by that gain recovers the stated residual bound.
- Interpretation: A modest 20 dB specification permits a tenth of the input amplitude.
- Include component tolerance and implementation effects in a practical margin.
- Measure the actual gain over the relevant stopband, not just at its edge.
- Verify that input circuitry remains linear before relying on this linear gain budget.
- This example does not define a clinical or medical-device acceptance limit.

#### Budget 2
- Given input amplitude `1 mV` and permitted output amplitude `0.01 mV`.
- Divide input by allowed residual: `1/0.01 = 100`.
- Use the amplitude-ratio equation in Section 4.13: `20 log10(100) = 40 dB`.
- The required gain is at most `0.01` at the interference frequency.
- Multiplying the input amplitude by that gain recovers the stated residual bound.
- Interpretation: A 100-fold amplitude reduction requires 40 dB, not 20 dB.
- Include component tolerance and implementation effects in a practical margin.
- Measure the actual gain over the relevant stopband, not just at its edge.
- Verify that input circuitry remains linear before relying on this linear gain budget.
- This example does not define a clinical or medical-device acceptance limit.

#### Budget 3
- Given input amplitude `2 mV` and permitted output amplitude `0.02 mV`.
- Divide input by allowed residual: `2/0.02 = 100`.
- Use the amplitude-ratio equation in Section 4.13: `20 log10(100) = 40 dB`.
- The required gain is at most `0.01` at the interference frequency.
- Multiplying the input amplitude by that gain recovers the stated residual bound.
- Interpretation: Changing both amplitude units together leaves the ratio unchanged.
- Include component tolerance and implementation effects in a practical margin.
- Measure the actual gain over the relevant stopband, not just at its edge.
- Verify that input circuitry remains linear before relying on this linear gain budget.
- This example does not define a clinical or medical-device acceptance limit.

#### Budget 4
- Given input amplitude `5 mV` and permitted output amplitude `0.005 mV`.
- Divide input by allowed residual: `5/0.005 = 1000`.
- Use the amplitude-ratio equation in Section 4.13: `20 log10(1000) = 60 dB`.
- The required gain is at most `0.001` at the interference frequency.
- Multiplying the input amplitude by that gain recovers the stated residual bound.
- Interpretation: A thousandfold amplitude reduction calls for 60 dB rejection.
- Include component tolerance and implementation effects in a practical margin.
- Measure the actual gain over the relevant stopband, not just at its edge.
- Verify that input circuitry remains linear before relying on this linear gain budget.
- This example does not define a clinical or medical-device acceptance limit.

#### Budget 5
- Given input amplitude `0.5 mV` and permitted output amplitude `0.005 mV`.
- Divide input by allowed residual: `0.5/0.005 = 100`.
- Use the amplitude-ratio equation in Section 4.13: `20 log10(100) = 40 dB`.
- The required gain is at most `0.01` at the interference frequency.
- Multiplying the input amplitude by that gain recovers the stated residual bound.
- Interpretation: A small input can still need strong rejection when the error budget is tighter.
- Include component tolerance and implementation effects in a practical margin.
- Measure the actual gain over the relevant stopband, not just at its edge.
- Verify that input circuitry remains linear before relying on this linear gain budget.
- This example does not define a clinical or medical-device acceptance limit.

#### Budget 6
- Given input amplitude `10 mV` and permitted output amplitude `0.1 mV`.
- Divide input by allowed residual: `10/0.1 = 100`.
- Use the amplitude-ratio equation in Section 4.13: `20 log10(100) = 40 dB`.
- The required gain is at most `0.01` at the interference frequency.
- Multiplying the input amplitude by that gain recovers the stated residual bound.
- Interpretation: A larger interferer needs a correspondingly larger absolute residual allowance for the same requirement.
- Include component tolerance and implementation effects in a practical margin.
- Measure the actual gain over the relevant stopband, not just at its edge.
- Verify that input circuitry remains linear before relying on this linear gain budget.
- This example does not define a clinical or medical-device acceptance limit.

#### Budget 7
- Given input amplitude `4 mV` and permitted output amplitude `0.04 mV`.
- Divide input by allowed residual: `4/0.04 = 100`.
- Use the amplitude-ratio equation in Section 4.13: `20 log10(100) = 40 dB`.
- The required gain is at most `0.01` at the interference frequency.
- Multiplying the input amplitude by that gain recovers the stated residual bound.
- Interpretation: The response must be evaluated where this interferer occurs.
- Include component tolerance and implementation effects in a practical margin.
- Measure the actual gain over the relevant stopband, not just at its edge.
- Verify that input circuitry remains linear before relying on this linear gain budget.
- This example does not define a clinical or medical-device acceptance limit.

#### Budget 8
- Given input amplitude `0.2 mV` and permitted output amplitude `0.02 mV`.
- Divide input by allowed residual: `0.2/0.02 = 10`.
- Use the amplitude-ratio equation in Section 4.13: `20 log10(10) = 20 dB`.
- The required gain is at most `0.1` at the interference frequency.
- Multiplying the input amplitude by that gain recovers the stated residual bound.
- Interpretation: This bound is for one component; multiple residuals require a combined budget.
- Include component tolerance and implementation effects in a practical margin.
- Measure the actual gain over the relevant stopband, not just at its edge.
- Verify that input circuitry remains linear before relying on this linear gain budget.
- This example does not define a clinical or medical-device acceptance limit.

**Takeaway:** The same observed 70 Hz line can have multiple histories, so record every rate and filter boundary.

## 6. Visual / Geometric Intuition

```text
Analog input frequency, sampled at 360 Hz:
0 ---------------- 180 ---------------- 360 Hz
0 ---------------- 180 ----------------   0 Hz  <- folded frequency
                   ^
               Nyquist edge
Example: 250 Hz lies 110 Hz from 360 Hz and folds to 110 Hz.
```

This picture is a frequency mapping, not a measured spectrum.
Actual filter gain, phase, and signal amplitude are not shown.
A real input above 360 Hz can fold again; choose the nearest integer multiple of the sample rate.

```text
Sampling before versus after filtering:
interference -> ADC -> digital filter : first fold may already be irreversible
interference -> analog filter -> ADC : out-of-band input can be attenuated first
samples -> digital LPF -> decimate : protects the new lower-rate Nyquist band
```

A sinusoid can look slow at sample instants even when its continuous oscillation is fast.
The sampled sequence is compatible with multiple continuous signals unless prior bandwidth information excludes the alternatives.
A finite observation can also spread energy across DFT bins when the tone does not align with a bin.
Leakage is not the same mechanism as sampling aliasing.

**Takeaway:** Folding is a many-to-one mapping; a plot cannot reconstruct the missing analog history by itself.

## 7. Comparison / Selection Table

| Choice | Use when | Benefit | Failure boundary / alternative |
|---|---|---|---|
| Keep original 360 Hz data | Storage and processing permit; no lower-rate need | Avoids an extra irreversible rate reduction | Does not undo original acquisition aliasing |
| Digital low-pass then integer decimation | Required band fits comfortably below new Nyquist | Lower storage and processing rates | Need transition, attenuation, phase, state and edge handling |
| Drop every other sample directly | Only for a deliberately constructed, already adequately band-limited teaching signal | Minimal computation | Unsafe as a general ECG pipeline; unfiltered content folds |
| Increase ADC rate | Analog chain and timing can support it | Wider separation between desired band and replicas | Does not replace an adequately characterized analog filter |
| Analog low-pass before ADC | Original continuous input may contain high-frequency interference | Reduces energy before first fold | Finite attenuation and component tolerance leave residual aliases |
| Digital notch after ADC | A known in-band narrow interferer is present | Can reduce that represented component | May remove desired content; cannot distinguish a pre-existing alias |
| Longer observation | Frequency discrimination needs more information and stationarity is plausible | Smaller natural DFT bin spacing | Changes latency and may violate stationarity |
| Zero-padding | Smoother visualization of a fixed observation is useful | Finer plotted frequency grid | No new observations or inherent resolution gain |

**Decision order:** Specify the task's required band and acceptable distortion; measure or bound the interference; select sample rate and analog filter; choose a digital resampling filter only if rate reduction is needed; inspect resulting output with suitable test signals.
**Disqualifier:** No evidence of analog filtering or clock behavior means that absence of aliasing is unproven.
**Fallback:** Retain original-rate samples and metadata; do not infer missing analog information from a cleaner-looking display.

### 7.1 Safe acquisition and test boundaries

This sheet teaches signal processing, not construction or operation of a human-connected acquisition system.
Do not connect a general-purpose oscilloscope, signal generator, microcontroller, or improvised circuit to a person.
An ordinary bench instrument can introduce an unintended conductive path through its power supply or protective earth.
Battery operation by itself does not establish isolation, leakage-current safety, or suitability for human connection.
Wireless communication does not establish that all conductive paths have been controlled.
A USB cable can create a new power or ground path even when a sensor originally runs on a battery.
A laptop power adapter can change the grounding arrangement compared with battery-only operation.
Test equipment, chargers, computers, shields, and accessories are part of the complete electrical system.
A block diagram without isolation and power boundaries is not evidence of safety.
An isolation component's rating alone does not establish system-level protection.
Compliance cannot be inferred from nominal ADC resolution, sample rate, or visually clean data.
For human-connected acquisition, use an appropriately reviewed system and institutional procedures applicable to its intended use.
Obtain qualified electrical-safety and regulatory review before any human connection or configuration change.
Follow the equipment manufacturer's authorized instructions and the relevant institutional approval process.
This text specifies no current limits, insulation distances, or patient-interface design values.
Those requirements depend on the applicable equipment classification, environment, accessories, and regulations.
Use a documented simulator, a suitable isolated test fixture, or a synthetic digital sequence for teaching exercises.
Confirm that a bench test arrangement is safe for its actual equipment before energizing it.
A digital synthetic test exercises downstream logic but does not test the analog front end.
A simulator exercises only the paths and operating conditions represented by that simulator.
A bench source connected at the ADC input does not characterize electrode-interface effects.
A test at one amplitude does not establish linearity across the entire available input range.
A test at one frequency does not characterize the passband and stopband.
A test with one cable arrangement does not establish behavior with every accessory.
Do not bypass protections to make a test waveform look cleaner.
Do not infer clinical suitability from this educational sample-rate example.
Record whether a test is synthetic, bench-based, or performed on an appropriately governed acquisition system.
Record source amplitude, source impedance where relevant, frequency, waveform and measurement uncertainty.
Record the exact injection point so that bypassed stages are not mistakenly considered tested.
Record connected instruments, isolation arrangement, power sources and configuration identifiers.
Record whether results are measured, calculated, assumed, or unavailable.
A safe setup remains a prerequisite even when all generated signals are nominally small.

### 7.2 Filter placement and saturation budget

An analog anti-alias filter attenuates interference only after the signal reaches that filter.
If an amplifier before the filter saturates, a later filter cannot undo the nonlinear distortion.
If an interference source is large enough to saturate the ADC, a digital low-pass filter is too late.
Conversely, an analog filter alone does not protect a later digital decimation boundary.
The correct test injection point depends on the boundary being evaluated.
A synthetic sequence inserted after the ADC can test decimator behavior.
It cannot establish the analog anti-alias response or input protection behavior.
A swept bench signal before the analog filter can test the transfer of the included stages.
The source and measurement setup must remain within the approved and documented bench limits.
Keep the source amplitude sufficiently controlled to distinguish linear filtering from overload.
Repeat at multiple safe amplitudes to detect amplitude-dependent distortion.
Distinguish a filter's attenuation at one frequency from its attenuation over a continuous stopband.
If interference is not bounded, a finite attenuation number cannot guarantee an absolute residual bound.
If the allowable in-band residual is unspecified, there is no quantitative acceptance decision.
Sum coherent worst-case residual amplitudes when their phase relationship is unconstrained and a worst-case bound is required.
Do not assume unrelated residuals are independent without a justified statistical model.
For a digital decimator, include coefficient quantization and actual numeric precision in the response evidence.
Store the filter version and sampling rate because the same coefficients have different physical-Hz responses at different rates.
Maintain separate acceptance criteria for alias suppression, passband distortion, timing and numerical overflow.
A passing spectral test does not establish the safety of human contact.
A safety review does not establish that the chosen signal-processing response preserves a particular task's information.
Both evidence categories remain open until the relevant qualified assessments are completed.

**Takeaway:** Choose a rate and filter together, based on the allowed error and measured interference, not a universal ECG frequency.

## 8. Failure Modes & Edge Cases

All entries below are **HYPOTHETICAL** diagnostics, not observed incidents.

### 8.1 A low-frequency spectral line appears

- Competing causes: an actual low-frequency input; a folded high-frequency interferer; spectral leakage; or a processing artifact.
- Discriminating check: inject a known out-of-band sinusoid into a test acquisition chain and sweep its frequency while recording the output line.
- Expected signal: an alias follows the predicted fold as the test frequency crosses the Nyquist boundary.
- Mitigation: preserve raw data and prevent use of the ambiguous feature until the chain is characterized.
- Correction: adjust the pre-ADC analog filter or sample rate to meet the declared rejection requirement.
- Prevention: retain filter response, rate, clock source, and test conditions with each acquisition configuration.
- Verification: repeat the controlled sweep and compare residual in-band amplitude against a predeclared error budget.
- Limitation: a moving peak alone cannot prove that an uncontrolled real-world component has the same source.

### 8.2 A peak moves after downsampling

- Competing causes: aliasing from unfiltered decimation; incorrect output-rate metadata; or a changed analysis window.
- Discriminating check: compare the frequency computed using both old and new rates and inspect the pre-decimation spectrum.
- Expected signal: a 110 Hz input to a 180 Hz output rate folds to 70 Hz if it is not sufficiently attenuated.
- Mitigation: stop using the decimated output for the affected frequency band and return to the original sequence.
- Correction: filter before decimation using a documented passband, transition, and stopband.
- Prevention: make filter and sample-rate metadata part of the resampling contract.
- Verification: inject known in-band and out-of-band tones and measure both passband error and aliased output.
- Limitation: the original sequence may already contain an acquisition alias.

### 8.3 A timing interval is unexpectedly short

- Competing causes: using aggregate channel rate as per-channel rate; a wrong header; or skipped samples.
- Discriminating check: compare per-channel count, duration, timestamp deltas, and the actual stream schema.
- Expected signal: interpreting two 360 Hz channels as one 720 Hz channel halves inferred intervals.
- Mitigation: quarantine the derived intervals and retain original timestamps and sample indices.
- Correction: use the actual per-channel sample rate and identify gaps rather than silently compacting the data.
- Prevention: store rate and channel layout alongside the record and test time-base conversions.
- Verification: compare known injected intervals and timestamp-derived durations.
- Limitation: device-clock drift still requires an independent reference to quantify.

### 8.4 A spectrum has broad shoulders

- Competing causes: non-bin-centered tone leakage; time-varying frequency; clipping; and broadband noise.
- Discriminating check: compare windows and segment lengths; inspect the waveform for flat tops and evaluate a stable injected tone.
- Expected signal: a window change alters leakage sidelobes, while clipping produces nonlinear harmonics.
- Mitigation: avoid assigning physiological meaning to the shoulders.
- Correction: select a suitable window and correct clipping at the front end when demonstrated.
- Prevention: record window, normalization, ADC range, and preprocessing choices.
- Verification: repeat the same analysis on a known-amplitude, known-frequency reference.
- Limitation: several causes can coexist and windowing alone does not identify them.

### 8.5 A waveform is distorted near a block boundary

- Competing causes: filter state reset; dropped block; padding or startup transient; and changes in gain.
- Discriminating check: compare continuous filtering with block filtering while preserving state and aligned indices.
- Expected signal: an avoidable discontinuity disappears when equivalent state is carried between blocks.
- Mitigation: mark the affected region and keep the unprocessed source.
- Correction: maintain filter state and define startup and gap handling explicitly.
- Prevention: include block boundaries and sample counters in stream tests.
- Verification: compare blockwise output with a continuous reference under the same filter and initial conditions.
- Limitation: zero-phase offline filtering is not equivalent to a causal real-time filter.

### 8.6 Two channels appear misaligned

- Competing causes: channel skew in the recording; buffering order; timestamp errors; and truly different lead morphology.
- Discriminating check: inspect record-specific documentation and use a shared injected timing event when acquisition hardware is available.
- Expected signal: a fixed delay in a shared test event supports channel timing skew.
- Mitigation: avoid precision cross-channel timing claims until alignment uncertainty is bounded.
- Correction: fix synchronization or apply a documented, justified correction.
- Prevention: retain channel IDs, shared clock information, and timing-calibration results.
- Verification: repeat the common-event test over time and operating conditions.
- Limitation: a different ECG morphology alone is not proof of skew; the dataset source notes possible interchannel skew.

### 8.7 Operational diagnostic playbook

Each entry is a hypothetical bench or data-pipeline investigation, not an observed incident.
Use a simulator or documented isolated test interface; do not connect improvised circuitry to a person.

#### D01. ADC saturation
- Symptom: repeated endpoint codes.
- Competing explanations: front-end overrange or a digital clamp.
- Discriminating observation: compare raw converter codes with downstream values.
- Expected evidence: endpoint occupancy appears in raw codes only for upstream saturation.
- Immediate containment: retain raw codes and mark the affected interval.
- Corrective direction: reduce test-source amplitude within documented input limits.
- Verification plan: check endpoint occupancy with a calibrated bench source.
- Boundary: Changing a digital filter cannot reconstruct clipped peaks.

#### D02. Input overload before the converter
- Symptom: flattening without ADC endpoint codes.
- Competing explanations: analog amplifier saturation or digital normalization.
- Discriminating observation: probe the isolated bench chain at accessible documented stages.
- Expected evidence: distortion before conversion supports an upstream overload.
- Immediate containment: stop interpreting the affected segment.
- Corrective direction: adjust the bench source or front-end range within its specification.
- Verification plan: repeat amplitude sweeps and compare waveform linearity.
- Boundary: Do not probe or modify an attached human-subject setup.

#### D03. Incorrect ADC scaling
- Symptom: amplitude differs by a constant factor.
- Competing explanations: wrong gain metadata or unit conversion.
- Discriminating observation: compare a known bench input with stored codes and the header.
- Expected evidence: a stable scale error tracks the applied input.
- Immediate containment: retain the raw integer samples.
- Corrective direction: correct the documented code-to-voltage conversion.
- Verification plan: check several levels and both polarities.
- Boundary: Nominal bit depth is not the same as effective accuracy.

#### D04. Channel swap
- Symptom: two traces appear exchanged.
- Competing explanations: interleaving order or wrong channel labels.
- Discriminating observation: inject distinct synthetic tones into separate bench channels.
- Expected evidence: the tones identify which sample stream belongs to each input.
- Immediate containment: suspend channel-specific interpretation.
- Corrective direction: fix the mapping in acquisition or decoding.
- Verification plan: repeat with channel labels and raw frames saved.
- Boundary: Different real lead morphology is not a reliable channel identifier.

#### D05. Byte order
- Symptom: codes alternate implausibly.
- Competing explanations: endianness mismatch or incorrect sample packing.
- Discriminating observation: decode a known byte pattern in both byte orders.
- Expected evidence: only the correct layout reproduces the expected codes.
- Immediate containment: quarantine decoded output.
- Corrective direction: correct the format definition and decoder.
- Verification plan: test zero, midscale and endpoint patterns.
- Boundary: Do not assume all ADC formats are signed integers.

#### D06. Signedness
- Symptom: baseline is displaced by half scale.
- Competing explanations: unsigned offset-binary data decoded as signed two-complement or vice versa.
- Discriminating observation: compare known zero-input code with format documentation.
- Expected evidence: the code mapping predicts the displacement.
- Immediate containment: keep source bytes unchanged.
- Corrective direction: implement the documented representation.
- Verification plan: test values on both sides of the representation boundary.
- Boundary: A DC offset can also originate in the analog chain.

#### D07. Packet loss
- Symptom: time series jumps at a block boundary.
- Competing explanations: lost frames or a real transient.
- Discriminating observation: inspect sequence numbers and timestamp gaps.
- Expected evidence: a counter discontinuity identifies missing transport records.
- Immediate containment: mark the time interval missing.
- Corrective direction: repair transport and preserve gaps rather than compacting samples.
- Verification plan: inject known frame loss in a bench replay.
- Boundary: A continuously increasing counter alone does not prove no analog samples were lost before framing.

#### D08. Duplicate block
- Symptom: a waveform segment repeats exactly.
- Competing explanations: transport retry without deduplication or buffer reuse.
- Discriminating observation: compare frame sequence IDs and raw payload hashes.
- Expected evidence: identical payload and repeated sequence support duplication.
- Immediate containment: exclude or flag duplicate blocks.
- Corrective direction: deduplicate using the stream contract.
- Verification plan: replay an intentional duplicate and verify sample positions.
- Boundary: A naturally repetitive waveform does not by itself prove duplicate transport.

#### D09. Counter wrap
- Symptom: sequence number suddenly decreases.
- Competing explanations: integer wrap or device reboot.
- Discriminating observation: compare the counter width with reset and boot identifiers.
- Expected evidence: a transition from maximum to zero can be normal wrap.
- Immediate containment: keep epoch context with the samples.
- Corrective direction: implement modular sequence comparison and explicit reboot handling.
- Verification plan: test both wrap and reset in replay.
- Boundary: Wrap and reboot are not always distinguishable without another identifier.

#### D10. Clock-rate metadata
- Symptom: all measured frequencies are proportionally wrong.
- Competing explanations: wrong declared sampling rate or clock configuration.
- Discriminating observation: compare a bench reference tone and independent clock observation.
- Expected evidence: a common scale factor across tones supports time-base error.
- Immediate containment: suspend quantitative frequency reports.
- Corrective direction: correct rate metadata only after establishing the clock source.
- Verification plan: repeat the test at multiple tones and durations.
- Boundary: A single line cannot separate clock error from an unknown source frequency.

#### D11. Clock drift
- Symptom: alignment error grows over a recording.
- Competing explanations: oscillator error or timestamp reconstruction.
- Discriminating observation: compare hardware time against an independent reference over time.
- Expected evidence: approximately linear growth suggests constant rate mismatch.
- Immediate containment: mark timing uncertainty.
- Corrective direction: calibrate or characterize the clock over relevant conditions.
- Verification plan: repeat long-duration timing comparison.
- Boundary: A clock can also vary with temperature or operating mode.

#### D12. Timestamp jitter
- Symptom: irregular timestamp differences.
- Competing explanations: real aperture jitter or delayed software timestamping.
- Discriminating observation: compare hardware-trigger timing with host arrival times.
- Expected evidence: stable hardware timing and variable arrivals implicate transport.
- Immediate containment: retain both hardware and receipt timestamps.
- Corrective direction: use acquisition-time semantics rather than arrival time.
- Verification plan: repeat under different host load.
- Boundary: Irregular arrival timestamps do not alone prove irregular ADC sampling.

#### D13. Acquisition gaps hidden by interpolation
- Symptom: smooth-looking output spans missing data.
- Competing explanations: automatic interpolation or packet loss.
- Discriminating observation: compare validity flags and raw indices with displayed samples.
- Expected evidence: a missing interval despite a smooth display reveals interpolation.
- Immediate containment: mark the interval and avoid treating invented values as observations.
- Corrective direction: preserve gaps and make interpolation explicit.
- Verification plan: replay a known gap and inspect downstream flags.
- Boundary: Interpolation cannot recover unknown high-frequency content.

#### D14. Filter startup transient
- Symptom: first samples differ from steady-state output.
- Competing explanations: initial state choice or padded data.
- Discriminating observation: compare cold-start and long-run synthetic input.
- Expected evidence: a decaying difference after startup supports an initialization transient.
- Immediate containment: label or exclude the affected initial interval.
- Corrective direction: define state initialization and warm-up policy.
- Verification plan: repeat after restart and after a discontinuity.
- Boundary: The appropriate transient duration depends on the filter and tolerance.

#### D15. Filter state contamination
- Symptom: new recording begins with a previous waveform.
- Competing explanations: state reuse across independent streams or residual analog settling.
- Discriminating observation: process the new synthetic stream with cleared and retained state.
- Expected evidence: a difference tied to prior input indicates carried state.
- Immediate containment: isolate streams and mark the boundary.
- Corrective direction: reset or initialize state according to the stream contract.
- Verification plan: run alternating independent records in both orders.
- Boundary: Continuous blocks of the same record generally need state preservation.

#### D16. Wrong filter coefficients
- Symptom: unexpected attenuation in a protected band.
- Competing explanations: incorrect coefficient order, scaling, or design rate.
- Discriminating observation: evaluate the implemented response using known synthetic tones.
- Expected evidence: a response mismatch follows the implemented coefficients.
- Immediate containment: retain the original data.
- Corrective direction: correct coefficient loading and rate convention.
- Verification plan: measure passband and stopband on the target implementation.
- Boundary: An offline design response is not a measurement of deployed arithmetic.

#### D17. Fixed-point overflow
- Symptom: large transient or wraparound after filtering.
- Competing explanations: accumulator overflow or coefficient scaling error.
- Discriminating observation: inspect intermediate range on controlled maximum-amplitude inputs.
- Expected evidence: wraparound or saturation follows arithmetic limits.
- Immediate containment: flag the affected output.
- Corrective direction: widen or rescale arithmetic with a documented error budget.
- Verification plan: exercise worst-case amplitudes and state transitions.
- Boundary: More bits do not repair an unstable or incorrectly designed filter.

#### D18. Downsampling phase
- Symptom: two decimated streams differ by one input sample.
- Competing explanations: different retained sample phase or time origin.
- Discriminating observation: trace original input indices retained in each output.
- Expected evidence: different index parity explains the apparent shift.
- Immediate containment: preserve index provenance.
- Corrective direction: set and document decimation phase.
- Verification plan: test an impulse with known input index.
- Boundary: Fractional alignment and filter delay still require separate treatment.

#### D19. Uncompensated group delay
- Symptom: processed peaks lag the source.
- Competing explanations: causal filter delay or transport buffering.
- Discriminating observation: inject an impulse and measure timestamped source and output.
- Expected evidence: a repeatable sample offset supports a fixed delay.
- Immediate containment: retain both original and processed time bases.
- Corrective direction: document delay or compensate only when justified.
- Verification plan: check delay over the useful frequency band.
- Boundary: IIR delay is generally frequency-dependent.

#### D20. Notch misidentification
- Symptom: a line is removed but the source remains uncertain.
- Competing explanations: true interference or an alias at the notch frequency.
- Discriminating observation: repeat using controlled injected tones and the unnotched raw sequence.
- Expected evidence: multiple analog frequencies may produce the same sampled line.
- Immediate containment: retain the unfiltered sequence.
- Corrective direction: address the demonstrated source at the appropriate boundary.
- Verification plan: compare passband distortion and residual interference.
- Boundary: A notch can remove desired signal content as well as interference.

#### D21. Power-line frequency drift
- Symptom: a narrow line moves around a nominal value.
- Competing explanations: time-varying interference or finite-window effects.
- Discriminating observation: inspect a time-frequency view and a known reference where available.
- Expected evidence: a smoothly moving line is consistent with drift but is not proof of origin.
- Immediate containment: avoid fixed-source attribution.
- Corrective direction: choose filtering only against a stated distortion budget.
- Verification plan: evaluate attenuation across the actual frequency range.
- Boundary: The frequency and strength of local interference are site-specific.

#### D22. Spectral leakage mistaken for noise
- Symptom: broad skirts surround a narrow component.
- Competing explanations: finite-window leakage or broadband contamination.
- Discriminating observation: compare a bin-centered test tone under multiple windows.
- Expected evidence: window-dependent sidelobes support leakage.
- Immediate containment: avoid interpreting skirts as a separate source.
- Corrective direction: choose a window appropriate to the measurement objective.
- Verification plan: check coherent gain and noise normalization.
- Boundary: A window change does not distinguish every combination of signals.

#### D23. Aliased noise floor
- Symptom: unexpected broadband in-band energy.
- Competing explanations: out-of-band analog noise or in-band front-end noise.
- Discriminating observation: compare controlled input filtering and noise spectra.
- Expected evidence: reduction with stronger pre-ADC rejection supports an alias contribution.
- Immediate containment: retain configuration and test conditions.
- Corrective direction: reduce out-of-band input or change the acquisition design.
- Verification plan: repeat across relevant bandwidth and amplitude conditions.
- Boundary: Noise contributions can coexist and may not be separable from one test.

#### D24. Improper zero padding
- Symptom: displayed peaks appear better resolved.
- Competing explanations: denser DFT display grid or genuinely longer observation.
- Discriminating observation: compare actual sample count and unpadded window duration.
- Expected evidence: padding changes displayed bins but not the observation duration.
- Immediate containment: report the original window length.
- Corrective direction: collect a longer stationary interval if the task permits.
- Verification plan: compare resolvability with separately acquired longer data.
- Boundary: Longer intervals can violate stationarity assumptions.

#### D25. Cross-channel timing skew
- Symptom: shared event appears shifted between channels.
- Competing explanations: multiplexed sampling or timestamp alignment error.
- Discriminating observation: apply a common isolated bench event to both inputs.
- Expected evidence: a consistent event offset bounds relative timing under test conditions.
- Immediate containment: avoid unqualified cross-channel timing conclusions.
- Corrective direction: characterize or correct timing if the application permits.
- Verification plan: repeat over temperature and acquisition modes.
- Boundary: A single bench event does not establish all operating-condition behavior.

#### D26. Dropped samples versus clock drift
- Symptom: record duration is inconsistent with sample count.
- Competing explanations: missing samples or a wrong time base.
- Discriminating observation: compare counters, hardware timestamps and an independent duration.
- Expected evidence: gaps indicate missing samples; gradual divergence suggests rate mismatch.
- Immediate containment: keep original timestamps and counters.
- Corrective direction: fix the proven cause without silently reshaping the signal.
- Verification plan: replay both a dropped-frame and a clock-error case.
- Boundary: Both mechanisms can occur in the same recording.

#### D27. Double filtering
- Symptom: signal is more attenuated than expected.
- Competing explanations: filter applied twice or incorrect response expectation.
- Discriminating observation: trace pipeline versions and compare against a single-stage reference.
- Expected evidence: two identical linear stages multiply their frequency responses.
- Immediate containment: retain the earliest available samples.
- Corrective direction: remove the unintended stage or revise the contract.
- Verification plan: test with a known frequency sweep.
- Boundary: Two intentional stages can be valid when their combined response is specified.

#### D28. Rate conversion rounding
- Symptom: output duration differs by a sample.
- Competing explanations: endpoint convention or conversion accumulator.
- Discriminating observation: compare time origin, half-open interval and output indices.
- Expected evidence: a one-sample difference can follow from endpoint choices.
- Immediate containment: record the convention and output length.
- Corrective direction: align implementation and downstream assumptions.
- Verification plan: test short and long records with known endpoints.
- Boundary: Do not silently add a sample just to match an expected length.

#### D29. Buffer overrun
- Symptom: periodic gaps appear under load.
- Competing explanations: producer-consumer imbalance or interrupt latency.
- Discriminating observation: inspect sequence continuity and buffer overrun counters.
- Expected evidence: gaps correlated with occupancy or overrun events support a capacity issue.
- Immediate containment: mark loss and preserve counters.
- Corrective direction: resize or schedule only after measuring the bottleneck.
- Verification plan: replay worst-case workload with loss instrumentation.
- Boundary: A larger buffer cannot solve sustained overload.

#### D30. Display autoscaling
- Symptom: waveform appears more prominent after filtering.
- Competing explanations: axis rescaling or real gain change.
- Discriminating observation: compare numerical amplitudes using fixed axis limits.
- Expected evidence: unchanged numerical amplitude with changed axes indicates display effect.
- Immediate containment: avoid visual-only acceptance decisions.
- Corrective direction: fix display scale during comparisons.
- Verification plan: compare calibrated numerical values and plots.
- Boundary: Display scaling does not cause the underlying samples to change.

**Takeaway:** Diagnose competing causes with discriminating observations; never equate a cleaner plot with proof of correctness.

## 9. Complexity & Cost Analysis

Let `N` be samples per channel, `C` the channel count, and `L` the number of taps in a direct-form FIR filter.
A straightforward FIR filter requires order `C × N × L` multiply-accumulate operations and `C × L` state values, up to implementation details.
A decimator can compute only retained outputs using an appropriate implementation; exact operation counts depend on the filter structure.
A radix-style FFT typically uses order `C × N log N` operations for a suitable transform size; this is not a device benchmark.
Raw sample storage grows with sample rate, channels, duration, and stored bytes per value.
**ILLUSTRATIVE:** `360 samples/s × 2 channels × 2 bytes = 1440 bytes/s`, excluding headers, compression, and transport overhead.
Over 10 s the same assumption gives `14,400 bytes`; this is not a claim about MIT-BIH on-disk encoding.
A higher rate increases processing and transfer load but can ease analog transition-band requirements.
A longer filter may improve a specified transition or attenuation but costs memory, compute, and potentially latency.
For a causal linear-phase FIR with `L` taps, group delay is `(L-1)/2` samples where its response is defined.
**ILLUSTRATIVE:** A 21-tap linear-phase FIR at 360 Hz has a 10-sample delay, approximately 27.78 ms.
IIR phase delay is generally frequency-dependent; offline forward-backward filtering has different causality and edge behavior.
Clock drift, jitter, power, analog tolerances, buffer overruns, and missing packets are deployment constraints, not consequences of the DFT equation.
No runtime, energy, numerical precision, or real-device performance was measured here.

**Takeaway:** Rate, filtering, latency, and storage are coupled, and theoretical operation counts are not benchmarks.

## 10. Key Theorems / Results Reference

**Sampling theorem — DERIVED sketch:** For an ideal continuous signal strictly band-limited to below half a uniform sample rate, the samples uniquely determine the signal under standard ideal-reconstruction assumptions.
A classical primary source is C. E. Shannon, [Communication in the Presence of Noise](https://doi.org/10.1109/JRPROC.1949.232969), 1949.
The theorem does not claim that arbitrary biological or interference signals are perfectly band-limited.
A physical acquisition chain has finite analog attenuation, noise, quantization, timing error, and finite precision.
The strict inequality avoids relying on the exceptional Nyquist boundary.
Real-world applicability requires an explicit tolerable error budget, not just a theorem citation.

**Frequency equivalence — DERIVED:** For integer sample index `n`, `exp(j 2π(f + k fs)n/fs) = exp(j 2πfn/fs)` for integer `k`.
For real signals, positive and negative frequency representations are also related by conjugate symmetry.
Consequently, sampled data alone cannot distinguish all analog frequencies that share the same discrete frequency.
Additional analog-bandwidth assumptions or independent observations are necessary.

**Finite observation — DERIVED:** Multiplying a signal by a finite window changes its spectrum by convolution with the window spectrum.
This explains leakage when a tone does not align with the DFT basis, without invoking aliasing.
Window choice trades main-lobe width against sidelobe behavior; no single window is optimal for every objective.

**Dataset scope — SOURCED:** The PhysioNet page reports two-channel analog recordings digitized at 360 samples/s per channel, 11-bit resolution, and a 10 mV range.
It identifies 48 records from 47 subjects and warns that tape artifacts may affect heart-rate-variability analysis and interchannel skew may affect combined-channel analysis.
Twenty-five records were deliberately selected for uncommon arrhythmias; the dataset cannot be treated as a general-population prevalence sample.
The page points to record headers and the database directory for record-specific details; inspect them before using a particular record.
These historical acquisition facts do not prove the bandwidth, accuracy, or clinical suitability of another system.

**Open evidence:** Actual analog filter response, clock error, input interference, device configuration, and the error tolerance for any proposed deployment are not measured in this sheet.
**Review trigger:** Revisit the source if the dataset version, record metadata, target acquisition chain, or task-specific bandwidth changes.

**Takeaway:** The theorem is a conditional uniqueness result, while dataset facts describe a historical collection, not a new device's performance.

## 11. Interview / Exam Questions

### Q1. What does 360 Hz mean for a two-channel ECG?
**Answer:** Each channel is sampled 360 times per second. Two channels yield 720 values per second in aggregate, but neither channel has a 720 Hz sampling rate. The PhysioNet dataset page states the per-channel rate.

### Q2. What is the time between adjacent ideal samples at 360 Hz?
**Answer:** `1/360 s`, or approximately `2.778 ms`. This assumes uniform sampling; missing samples and clock jitter must be represented separately.

### Q3. What is the Nyquist frequency at 360 Hz?
**Answer:** `360/2 = 180 Hz`. This is a property of the sample rate, not a measured analog passband or proof of clinical sufficiency.

### Q4. Why is sampling at exactly twice the highest frequency fragile?
**Answer:** At the boundary, phase ambiguity can make a Nyquist-frequency sinusoid inadequately represented, and a realizable filter needs nonzero transition width. Use strict separation and specify a practical filter and error budget.

### Q5. Where does a 250 Hz sinusoid appear if sampled at 360 Hz?
**Answer:** At `|250 - 360| = 110 Hz` in the first Nyquist zone, with phase dependent on the original signal. This is a frequency-equivalence calculation, not evidence that a particular observed 110 Hz peak came from 250 Hz.

### Q6. Can an after-ADC low-pass filter remove all analog aliasing?
**Answer:** No. Once two different analog components have produced indistinguishable discrete samples, digital processing cannot uniquely tell them apart. Suppress problematic analog energy before the ADC.

### Q7. What must happen before 360 Hz samples are decimated to 180 Hz?
**Answer:** Apply a suitable digital anti-alias filter before retaining every second sample. Its passband and stopband must meet the intended output-band and error requirements; the new Nyquist frequency is 90 Hz.

### Q8. What happens to an unfiltered 110 Hz component at 180 Hz output rate?
**Answer:** It folds to `|110 - 180| = 70 Hz`. It can overlap a real 70 Hz component and cannot be separated from it from output samples alone.

### Q9. How many samples and what DFT spacing does a 10 s window have at 360 Hz?
**Answer:** `3600` samples per channel, and `360/3600 = 0.1 Hz` bin spacing. This is an illustrative half-open uniform window; frequency resolution also depends on window and signal conditions.

### Q10. Why are the first-to-last timestamp span and window duration different?
**Answer:** With zero-based indices, 3600 samples at 360 Hz run from index 0 to 3599, so the span is `3599/360 ≈ 9.9972 s`. The conventional half-open window duration is `3600/360 = 10 s`.

### Q11. Does zero-padding resolve two tones that a short recording cannot resolve?
**Answer:** No. It samples the same finite-window spectrum more densely for display or interpolation; it does not add observations or the resolving power of a genuinely longer record.

### Q12. How would you distinguish leakage from aliasing?
**Answer:** Use a controlled sinusoid, vary the analysis window and length, and separately sweep its input frequency across a sampling boundary. Leakage changes with the window; alias frequency follows the folding relation. Both can coexist.

### Q13. Can you claim that MIT-BIH establishes a new device's clinical validity?
**Answer:** No. It is a historical research database with particular recording and selection properties. Its 360 Hz rate is not a validation of another analog front end, device, algorithm, population, or clinical use.

### Q14. What metadata should accompany a resampled segment?
**Answer:** Record source and channel identifiers, original and output rates, time origin, sample indices and gaps, filter coefficients or specification, state and edge handling, processing version, and any uncertainty. Those records make the transformation auditable but do not themselves prove correctness.

**Takeaway:** Strong answers state the numerical result, the sampling boundary, and the uncertainty that remains.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | ECG sampling, spectral aliasing and research data provenance |
| Core formula | Sampling rate `fs` gives Nyquist frequency `fs/2`; a uniformly sampled 250 Hz tone folds to 110 Hz at 360 Hz |
| Key assumption | Uniform timing and a suitable analog anti-alias filter precede digitization; actual filter behavior must be measured |
| Complexity | Sample and filter cost scale with channels, sampling rate and filter implementation; Section 9 gives the stated assumptions |
| Use when | Designing or evaluating a research signal acquisition or resampling pipeline |
| Avoid when | Treating a nominal sampling rate or a historical dataset as evidence of clinical validity for a device |
| Go deeper | [Sensor collection index](INDEX.md) and [MIT-BIH dataset v1.0.0](https://physionet.org/content/mitdb/1.0.0/) |
| Diagnostic | Preserve raw samples, timestamps, channel units, filter settings and resampling version |
| Evidence | Dataset properties are sourced; signal examples are illustrative; draft and unreviewed |
