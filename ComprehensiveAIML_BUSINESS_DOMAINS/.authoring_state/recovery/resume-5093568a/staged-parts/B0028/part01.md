---
title: "ECG Sampling and Aliasing"
topic: "ECG sampling and aliasing"
collection: "sensor-fusion-iot-and-realtime-perception"
file_id: "01"
level: "foundation"
evidence: "DERIVED"
created: "2026-11-07"
last_reviewed: null
content_status: "draft"
version: "2.0"
status: "draft; not independently reviewed"
notation_style: "plain-text"
prerequisites: ["rates and reciprocals", "basic sinusoids", "decibels", "array indexing"]
source_status: "DERIVED arithmetic; one SOURCED dataset description; OPEN device behaviour"
owner_role: "Perception and integration engineer"
review_trigger: "Change of sample rate, analog front end, filter specification, lead set, or intended action"
source_index: "INDEX.md"
---

# ECG Sampling and Aliasing

## 1. File Header

**Purpose.** Decide whether a proposed sampling, filtering, and decimation plan preserves a *declared* electrocardiogram band, and diagnose an unexpected spectral feature without mistaking a peak for proof of its physical cause. The card is about the arithmetic of the acquisition chain, not about rhythm interpretation.

**The decisions this card supports.**

1. Given a task band (for example 0.5–40 Hz for a monitoring-grade rhythm review), is a candidate rate and analog filter adequate, and with how much margin?
2. Given a suspicious low-frequency oscillation in a stored record, what set of analog frequencies could have produced it at this rate, and which checks discriminate between them?
3. Given a decimation step in the processing pipeline, where must the anti-alias filter sit, and how sharp must it be to reach the declared stopband attenuation?
4. Given a bit depth and input range, what amplitude is the smallest resolvable step, and does it sit below the smallest feature the task claims to measure?

**Audience and prerequisites.** Engineers and analysts beginning biomedical signal processing: firmware engineers sizing an acquisition front end, data engineers resampling archived waveforms, and analysts reading a spectrum for the first time. The card assumes comfort with rates and reciprocals, the shape of a sinusoid, decibels as a log ratio of amplitudes, and zero-based array indexing. It does not assume prior filter design.

**Outcome.** After working through Sections 4 and 5, a reader can compute the sample period, the Nyquist frequency, the alias of any single-frequency interferer, the discrete-Fourier bin spacing for a given record length, the least attenuation needed to push an interferer below one least-significant bit, and the minimum filter order implied by a chosen transition width. Each of those is a number with units, not a slogan.

**Exclusions.** Clinical diagnosis; recommendations about diagnostic bandwidth for a regulated device; treatment decisions; certification or validation of a medical device; the full theory of multirate filter banks. Where a claim would need bench measurement of a specific front end, this card says so instead of inventing it.

**Evidence status.** The arithmetic in this card is DERIVED: every formula is stated with its conditions and checked with a worked number. The dataset facts in the provenance paragraph are SOURCED from a public database description that was read at authoring time. Behaviour of any particular commercial front end is OPEN: no oscilloscope trace, no bench measurement, and no vendor datasheet was inspected for this card. Every constructed waveform, every assumed interference amplitude, and every filter specification is marked ILLUSTRATIVE, including cases where the arithmetic that follows from it has been recomputed.

**Provenance.** Moody and Mark, *MIT-BIH Arrhythmia Database* version 1.0.0, PhysioNet, published 2005, consulted 2026-11-10, <https://physionet.org/content/mitdb/1.0.0/>. The database description reports 48 approximately half-hour, two-channel ambulatory recordings from 47 subjects, collected at Boston's Beth Israel Hospital between 1975 and 1979, digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range, with approximately 110,000 beat annotations. Those numbers are used here only as a realistic anchor for worked arithmetic. They say nothing about the analog anti-alias filter that preceded the original digitizer, nothing about inter-channel timing, and nothing about any device built today.

**Why a sample rate is not a band.** A rate fixes the spacing of the time grid. It does not fix which analog frequencies reached that grid. Two acquisition chains can both write 360 samples per second, one with a 40 Hz analog low-pass and one with a 400 Hz analog low-pass; the second writes samples that contain energy the first never recorded, and no later processing can tell the two chains apart from the stored numbers alone. The rate is a property of the grid; the band is a property of the filter that precedes the grid. Confusing the two is the single most common error this card exists to prevent.

**Retrieval cue.** An unexpected low-frequency oscillation, a "notch" that works on one recording but not another, a beat detector that degrades after a resampling step, or a rhythm estimate that changes when the record is converted between rates: start with Sections 4.3, 7, and 8.

**Takeaway.** A rate alone is not evidence that the acquired band was preserved; the filter position and the declared task band are the evidence.

## 2. Concept Map / Mental Model

```text
continuous potential at the electrodes
  -> analog front end: electrode offset, gain, input range, differential rejection
  -> analog anti-alias low-pass (BEFORE the sampling operation it protects)
  -> sample-and-hold aperture, then ADC quantization at the actual instants
  -> timestamped integer codes plus acquisition metadata (rate, range, filter, lead)
  -> optional digital filter at the ORIGINAL rate
  -> decimation or rational resampling, with a filter before each rate reduction
  -> finite analysis window, window function, DFT or filter bank
  -> a number, an uncertainty, and a statement of what was not measured
```

Each arrow in that chain is a place where a claim can be lost. The chain is written in physical order because the order is the point: an operation can only protect the operations that come after it.

**Stage 1, the front end.** Electrode half-cell potentials can be tens of millivolts of offset, while the cardiac signal of interest is around one millivolt. The front end must remove that offset without removing the slow ST-segment content the task may need. A high-pass corner set too high turns a real ST depression into a smaller depression; a corner set at zero drifts into the amplifier rails. This stage sets the low end of the preserved band, and it is the stage that determines whether "0.5 Hz" is a real specification or a label.

**Stage 2, the analog anti-alias filter.** This filter must attenuate everything above the Nyquist frequency of the *sampling operation that follows it*. Its transition band is bounded below by the highest frequency the task requires and above by Nyquist. The wider that gap, the cheaper the filter; the narrower it is, the higher the order. Section 4.7 turns that trade into an order estimate.

**Stage 3, the sampling operation.** The sample-and-hold does not take an instantaneous value; it averages over an aperture. That averaging is itself a low-pass with a sinc response, and at frequencies near Nyquist the attenuation is not negligible (Section 4.5). The ADC then quantizes, which adds an amplitude error independent of any frequency error.

**Stage 4, stored codes and metadata.** An integer code becomes a voltage only through the recorded gain and range. A stored array without its range, its rate, and its filter description is an array of integers, not a signal. Metadata is part of the measurement, not documentation about it.

**Stage 5, digital filtering and decimation.** A digital filter at 360 Hz can remove anything below 180 Hz. It cannot remove energy that the ADC already folded into the band. Decimation then lowers the rate, and it requires its own filter ahead of the rate reduction, at the pre-decimation rate.

**Stage 6, the analysis window.** A finite window convolves the true spectrum with the window's transform. That operation smears energy, creates side lobes, and sets the smallest frequency difference the analysis can separate. It does not create energy that was not there, but it can hide or imitate a peak.

**Stage 7, the decision.** The output of the chain is not "the spectrum". It is a statement such as "a 10.0 Hz component of 0.4 mV appears in this 30-minute record, consistent with the seventh harmonic of 50 Hz folding at 360 Hz, and not distinguishable from a genuine 10 Hz oscillation by this channel alone".

**Four invariants worth memorising.**

- **Placement invariant.** Every anti-alias filter sits before the operation whose folding it prevents, and every decimation filter sits before the rate reduction it protects.
- **Non-recoverability invariant.** Once two analog frequencies have produced identical samples, no deterministic post-processing of those samples can separate them.
- **Unit invariant.** Every rate names a channel count; every amplitude names a gain reference; every window length names a unit of time.
- **Evidence invariant.** A spectral peak establishes that energy exists at a discrete-time frequency. It does not establish which analog frequency, which physical source, or which path produced it.

**Takeaway.** Read the chain left to right and ask at each arrow what could be lost; the answer locates the filter that is missing.

## 3. Foundational Definitions

| Term | Meaning | Units and boundary |
|---|---|---|
| Electrocardiogram | Potential difference between two or more electrode sites, over time | Millivolts; lead placement is part of the definition |
| Lead | The ordered pair of electrodes and the amplifier configuration that produces one channel | Named, not implied; a channel count is not a lead count |
| Analog input | Continuous-time voltage presented to the acquisition chain | Volts; not yet discretized |
| Sample | One stored representation of the input at one acquisition instant | Code or voltage; not a duration |
| Sampling rate `fs` | Sampling instants per second, per channel | Hz; strictly positive |
| Sampling period `Ts` | Time between adjacent uniform instants | Seconds per sample; equals `1/fs` |
| Nyquist frequency `fN` | Half the uniform sampling rate | Hz; a property of the grid, not a filter corner |
| Task band | Frequency range the stated decision actually requires | Hz; task-specific, never a universal constant |
| Bandwidth `B` | Highest frequency the task band requires | Hz; the number that enters design inequalities |
| Alias | A discrete-time frequency that two or more analog frequencies map onto | Hz; depends on the rate in force |
| Folding | The arithmetic that maps an analog frequency into the baseband | A map, not a physical process |
| Anti-alias filter | Analog low-pass preceding a sampling operation | Specified by passband edge, stopband edge, attenuation |
| Decimation | Reducing the rate by an integer factor after filtering | Factor `M`; the filter runs at the old rate |
| Rational resampling | Changing rate by `L/M` with interpolation and decimation | Integers `L`, `M`; order matters |
| Discrete Fourier transform | The exact spectrum of a finite sequence at `N` discrete frequencies | Bin spacing `fs/N` |
| Frequency resolution | Ability to separate two close components | Hz; set by window length and window shape |
| Bin | One output of a DFT | Index `k`; frequency `k·fs/N` |
| Spectral leakage | Energy from one component appearing across many bins | Caused by the finite window |
| Scalloping loss | Amplitude error when a component falls between bins | Decibels; window-dependent |
| Window function | A taper applied before the DFT to trade leakage against resolution | Dimensionless weights |
| Quantization | Mapping an amplitude to one of a finite set of codes | Volts per code; independent of frequency |
| Least significant bit | The voltage step represented by one code increment | Volts; equals range divided by code count |
| Clipping | Input beyond the converter's range | Nonlinear; not repairable by filtering |
| Sample-and-hold aperture | The finite interval over which the converter senses the input | Seconds; acts as a sinc low-pass |
| Jitter | Deviation of actual sampling instants from the ideal grid | Seconds; effect scales with signal slope |
| Clock drift | Slow change of the sampling clock relative to a reference | Parts per million, or seconds per day |
| Amplitude resolution | Smallest input change that reliably changes the code | Volts; bounded by noise, not only by the step |
| Effective number of bits | Bit depth implied by measured signal-to-noise ratio | Bits; always below the nominal depth |
| Coherence | Condition where a component completes an integer number of cycles in the window | Dimensionless; avoids leakage entirely |
| Folding frequency | The frequency at which the alias map reverses direction | Hz; equals `fN` |

**Conventions that the rest of the card relies on.**

*Time base.* Sample index `n` runs from zero. Under an ideal uniform grid, the time of sample `n` relative to the first sample is `n/fs`. A record of `N` samples spans `N/fs` seconds in the standard half-open window convention, and the first-to-last timestamp difference is `(N-1)/fs`. Both statements are correct; they answer different questions. A duration computed from a count uses the first, and a duration computed from two timestamps uses the second.

*Rates and channels.* The MIT-BIH description's "360 samples per second per channel" describes two interleaved streams at 360 Hz each. It does not describe one stream at 720 Hz, and it does not describe 180 Hz per channel. A rate is always stated with its channel scope.

*Amplitude.* A code becomes millivolts through the recorded gain and range. For 11 bits over a 10 mV range, one code step is 10 mV divided by 2048 codes, which is 4.8828 microvolts; the signed input range is roughly plus or minus 5 mV. That derivation is repeated in Section 4.8 because the number recurs throughout the card.

*Event time versus sample index.* A beat annotation, a device alarm, and a stored sample each have a time, and those times come from different clocks. This card treats the sampling clock as ideal; the companion sheet on time synchronization and calibration handles the case where it is not.

*What is not available at decision time.* When a stored record arrives without its analog filter description, the preserved band is unknown, and the honest answer is "unknown", not "assumed adequate". When the rate is known but the aperture is not, the attenuation near Nyquist is unknown. When the range is known but the gain calibration is not, the amplitude scale is unknown. Each of those gaps has to be stated rather than filled by a plausible default.

**Takeaway.** A definition in this table is usable only when its unit and its boundary travel with it.

## 4. Core Formulas with Full Derivation

The section follows one anatomy for every result: the formula, where each symbol lives, the conditions under which the formula holds, a numerical example, and the interpretation that follows. Each subsection closes with a boundary where the formula stops being trustworthy. Every number below is ILLUSTRATIVE unless it is arithmetic on a stated assumption.

### 4.1 Uniform sampling and sample spacing

**FORMULA.** `Ts = 1/fs` and `t[n] = t0 + n/fs`.

**WHERE.** `fs` is the sampling rate in hertz for one channel; `Ts` is the spacing in seconds per sample; `n` is a zero-based integer index; `t0` is the time of the first sample in seconds. `Ts` is a duration, `fs` is a rate, and their product is dimensionless and equal to one.

**CONSTRAINT.** The grid is uniform: every adjacent pair of instants differs by the same `Ts`, within whatever jitter the system tolerates. The index is contiguous: a dropped frame must be detected and recorded, never silently compacted, because compaction rewrites every subsequent time.

**NUMERICAL EXAMPLE — ILLUSTRATIVE.** At `fs = 360 Hz`, `Ts = 1/360 s = 0.002777… s`, which is 2.7778 ms per sample. A 30-minute record holds `1800 s × 360 samples/s = 648,000` samples per channel. A 100 ms QRS complex spans `0.1 s × 360 s⁻¹ = 36` samples. At 250 Hz the same complex spans 25 samples; at 500 Hz it spans 50.

**INTERPRETATION.** The sample spacing sets the smallest time interval the representation can express, which is not the same as the timing precision. Two beats 2 ms apart cannot be separated at 2.78 ms spacing, but a single beat's position can still be estimated to a fraction of a sample by interpolation when the waveform is smooth and the noise is low.

**Boundary.** A missing packet breaks the index-to-time map. If the acquisition firmware timestamps samples by index rather than by clock, one dropped frame shifts every later sample by 2.78 ms at this rate, and a 10-minute record with one dropped frame per minute accumulates 27.8 ms of error. Only a stored device clock or a sequence number can expose this; the waveform alone cannot.

### 4.2 Nyquist frequency and the band-limit condition

**FORMULA.** `fN = fs/2`, and a baseband signal whose highest component is `B` is representable without ambiguity when `fs > 2B`. In design practice the usable passband edge sits below `fN` by the filter's transition width, so the working inequality is `fpass + Δf ≤ fN` with `Δf` the transition width.

**WHERE.** `fN` is the Nyquist frequency in hertz; `fs` is the rate in hertz; `B` is the highest frequency the task requires, in hertz; `fpass` is the filter's passband edge in hertz; `Δf` is the width of the transition from passband edge to the stopband edge where the required attenuation is first met.

**CONSTRAINT.** The signal must be band-limited *before* the sampling operation, by an analog filter with finite stopband attenuation, and reconstruction must be appropriate. The theorem's ideal form assumes an exact band limit and ideal interpolation; no physical filter supplies either, which is why the inequality is written with slack rather than equality.

**NUMERICAL EXAMPLE — ILLUSTRATIVE.** At `fs = 360 Hz`, `fN = 180 Hz`. With a task band of 0.5–40 Hz, the distance from the passband edge to Nyquist is 140 Hz, which is an enormous transition region: an analog filter can be gentle and still meet a 60 dB stopband requirement (Section 4.7 computes the order). At `fs = 250 Hz`, `fN = 125 Hz` and the same 40 Hz task band leaves 85 Hz of transition. At `fs = 100 Hz`, `fN = 50 Hz`, and a 40 Hz task band leaves only 10 Hz, which forces a much steeper filter for the same attenuation.

**INTERPRETATION.** A high rate relative to the task band is not wasteful; it buys transition width, and transition width buys filter order, phase distortion, and cost. The rate should be chosen together with the filter, not before it.

**Boundary — the equality case.** A sinusoid exactly at `fN` is sampled at its zero crossings if its phase is zero, and at its peaks if its phase is 90 degrees. Both sequences are consistent with the same continuous sinusoid, so amplitude and phase are not recoverable at exactly `fN`; the sample sequence is `A·(−1)^n·sin(phase)` and carries one number, not two. Designing with `fs = 2B` exactly is therefore a defect, not a tight design. The practical margin is the transition width plus whatever interference headroom the front end can afford.

**Boundary — the strong-interferer case.** The inequality `fs > 2B` says nothing about how much out-of-band energy exists. A 50 mV interferer at 250 Hz sampled at 360 Hz folds to 110 Hz with 50 mV of amplitude; the fact that 110 Hz is above the 40 Hz task band does not help if the analysis later decimates or if the amplifier saturates on the interferer before the ADC. Amplitudes, not only frequencies, decide whether a chain survives.

### 4.3 Folding a single sinusoid into the baseband

**FORMULA.** For an analog frequency `f` sampled at rate `fs`, the discrete-time frequency is `fa = |f − k·fs|` for the integer `k` nearest to `f/fs`, equivalently `fa = |((f + fs/2) mod fs) − fs/2|` with the outer absolute value applied for real signals.

**WHERE.** `f` is the analog frequency in hertz, `fa` is the resulting baseband frequency in hertz with `0 ≤ fa ≤ fs/2`, `k` is an integer, and `mod` is the non-negative remainder. The map is many-to-one: every `f + k·fs` and every `−f + k·fs` lands on the same `fa`.

**CONSTRAINT.** The interferer is a single real sinusoid, the rate is constant over the window, and the sampling instants are uniform. The rule predicts frequency equivalence only; amplitude and phase of the observed component depend on the interferer's amplitude and phase, on the aperture, and on any analog filtering ahead of the ADC.

**DERIVATION.** Take `x(t) = cos(2π f t)` and sample at `t = n/fs`. Then `x[n] = cos(2π f n/fs)`. Write `f = k·fs + r` with integer `k` and `−fs/2 < r ≤ fs/2`. Substituting gives `x[n] = cos(2π k n + 2π r n/fs)`. Since `2π k n` is an integer multiple of `2π`, the cosine is unchanged: `x[n] = cos(2π r n/fs)`. Finally `cos` is even, so a negative `r` produces the same sequence as `|r|`, which is why the alias frequency is reported non-negative.

**NUMERICAL EXAMPLE — ILLUSTRATIVE, three cases at `fs = 360 Hz`.**

```text
f = 250 Hz : k = 1 (250/360 = 0.694) -> fa = |250 - 360| = 110 Hz
f = 350 Hz : k = 1                    -> fa = |350 - 360| =  10 Hz
f = 450 Hz : k = 1                    -> fa = |450 - 360| =  90 Hz
f = 650 Hz : k = 2 (650/360 = 1.806)  -> fa = |650 - 720| =  70 Hz
f = 500 Hz : k = 1                    -> fa = |500 - 360| = 140 Hz
```

**INTERPRETATION.** The 350 Hz case is the dangerous one, because 10 Hz lies inside the task band of a rhythm analysis and looks like a genuine slow oscillation of the baseline or a physiological rhythm. The 250 Hz and 450 Hz cases land above 40 Hz and would be removed by any digital low-pass at 40 Hz *if* the analysis keeps the original 360 Hz rate; if the pipeline decimates by four to 90 Hz, the 110 Hz alias folds again to `|110 − 90| = 20 Hz` and re-enters the band. Folding is not a one-time event; every rate reduction applies the same map again.

**Worked discrimination.** Given an observed 10.0 Hz component at 360 Hz, the candidate analog frequencies are 10, 350, 370, 710, 730 Hz and so on. Three checks separate them. First, look at the raw wideband spectrum before any 40 Hz digital filter: a genuine 10 Hz source and a folded 350 Hz source both appear at 10 Hz after sampling, so this check cannot separate them at all — the information is destroyed at the ADC, which is the point of the placement invariant. Second, change the rate: at 500 Hz, 350 Hz folds to 150 Hz while a true 10 Hz stays at 10 Hz; a re-acquisition at a different rate is the only clean discriminator. Third, inspect the analog domain with a spectrum analyzer ahead of the ADC, or compare against a channel with a different analog filter. A digital notch at 10 Hz removes both the interferer and the physiological signal, which is why "notch it out" is not a diagnosis.

**Boundary — several interferers at once.** If a 10 Hz physiological component and a folded 350 Hz interferer both exist, the samples contain their sum, and a single-channel spectrum shows one line whose amplitude and phase are the vector sum. No processing of those samples recovers the two amplitudes separately. The honest output is a bound, not a decomposition.

**Boundary — non-stationary interferers.** A broadband source such as muscle activity folds as a spectrum rather than a line: energy from every band above `fN` is mirrored into the baseband, raising the noise floor across the whole band. A single-peak model then understates the problem, and the diagnostic is a comparison of noise floor between records acquired at different rates.
