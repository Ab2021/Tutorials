## 7. Comparison / Selection Table

### 7.1 Rate and band choices against required analog order

All attenuation figures assume a Butterworth magnitude, which is a conservative lower bound: an
elliptic or Chebyshev design reaches the same stopband attenuation at a lower order, at the cost
of passband ripple or group-delay variation. Ratios are `(fs/2) / f_band`.

| Configuration | `fs` (Hz) | Declared band (Hz) | Ratio | Order for 60 dB | Order for 66.2 dB | 8th-order attenuation at edge |
|---|---|---|---|---|---|---|
| Bedside monitor, narrow band | 250 | 40 | 3.125 | 6.1 | 6.7 | 79.2 dB |
| Bedside monitor, wide band | 250 | 100 | 1.250 | 31.0 | 34.2 | 15.6 dB |
| Ambulatory recorder | 200 | 40 | 2.500 | 7.5 | 8.3 | 63.7 dB |
| Ward telemetry | 500 | 100 | 2.500 | 7.5 | 8.3 | 63.7 dB |
| Research-grade | 1000 | 150 | 3.333 | 5.7 | 6.3 | 83.7 dB |

Read the second row as the anti-pattern. A 250 Hz stream cannot protect a 100 Hz band with any
practical analog filter, because the folding edge is at 125 Hz and the transition band is only
25 Hz wide. The row to choose instead is the fourth: 500 Hz sampling protects a 100 Hz band with
8 poles, which reaches 63.7 dB at the folding edge. The rate is doing the work that filter order cannot.

### 7.2 Analog versus digital anti-alias placement

| Placement | Removes | Cannot remove | Cost | When appropriate |
|---|---|---|---|---|
| Analog low-pass before ADC | Energy above Nyquist that would fold | In-band interference, baseline wander | Board area, component tolerance, drift | Always; the only place folding can be prevented |
| Digital low-pass before decimation | Energy that would fold at the lower rate | Anything already folded at the ADC | Multiply-accumulate throughput, group delay | Always before any rate reduction |
| Digital notch after ADC | A narrow in-band tone (50/60 Hz) | Broadband artifacts, folded energy | Settling transient, ring, phase distortion | Mains interference that survives the analog stage |
| Digital low-pass after decimation | Out-of-band energy at the output rate | Nothing already folded; it cannot recover lost band | Cheap, but pointless for alias control | Shaping only; never an anti-alias substitute |
| Sample-rate increase | Nothing; it moves the folding edge | Artifacts already folded at the old rate | Power, data volume, storage | When the band requirement genuinely exceeds the old Nyquist |

The third row needs a warning that belongs here rather than in Section 8: a notch at 50 Hz with
`Q = 35` at a 250 Hz rate has a −3 dB half-width of `50/(2*35) = 0.714 Hz`, so its notch spans
49.29–50.71 Hz, and its impulse response decays with a time constant `1/(pi*BW) = 0.223 s`. That is
a settling transient of roughly 1.0 s at the 1% level, which is longer than a single P-QRS-T
complex. A notch is therefore suitable for continuous monitoring and unsuitable for analysis of
the first second after an epoch boundary.

### 7.3 Window selection for spectral inspection

| Window | First sidelobe | Mainlobe width | Scalloping loss (worst) | ENBW | Use when |
|---|---|---|---|---|---|
| Rectangular | −13.4 dB | `2*fs/N` | −3.92 dB | 1.00 bin | Tone exactly on a bin; transient capture |
| Hann | −31.5 dB | `4*fs/N` | −1.42 dB | 1.50 bin | General spectral inspection |
| Hamming | −44.2 dB | `4*fs/N` | −1.78 dB | 1.36 bin | Sidelobe suppression with modest widening |
| Blackman | −64.1 dB | `6*fs/N` | −1.10 dB | 1.73 bin | Weak components near strong ones |
| Kaiser (β = 8.6) | ≈ −90 dB | tunable | tunable | tunable | Design-point stopband with adjustable trade |

The scalloping column is the one people forget. With a rectangular window and a tone exactly half
a bin off centre the measured peak is 3.92 dB low; that is not a filter defect and not a patient
finding, it is the geometry of sampling a `sinc` at discrete points. Coherent sampling removes
scalloping entirely for a single tone, which is why a 30 s frame at 250 Hz with a 50 Hz tone
(1500 exact periods) reads the amplitude cleanly.

### 7.4 Word length selection

| Words | LSB over 10 mV | `sigma_q` (µV) | Theoretical SNR_fs | Practical verdict for surface ECG |
|---|---|---|---|---|
| 8 | 39.06 µV | 11.28 | 49.9 dB | Unusable for morphology; adequate for presence detection |
| 10 | 9.77 µV | 2.82 | 61.9 dB | Marginal; P waves lost |
| 12 | 2.44 µV | 0.705 | 74.0 dB | Adequate when the analog noise floor is above 1 µV RMS |
| 16 | 0.153 µV | 0.044 | 98.1 dB | Word length stops binding; analog and electrode noise dominate |
| 24 | 0.60 nV | 0.17 nV | 146.2 dB | Meaningless without a correspondingly quiet analog chain |

The last row is the useful negative example. Buying 24 bits does not produce 146 dB of usable
dynamic range on a surface ECG, because electrode offset, polarization drift and amplifier noise
occupy microvolts. A specification that quotes word length without quoting the input-referred
noise floor is not a specification of signal quality.

## 8. Failure Modes & Edge Cases

Each entry states the symptom, the competing causes that produce that same symptom, the check
that discriminates between them, and the correction. The competing-cause structure is the point:
the symptom alone never identifies the mechanism.

### 8.1 A clean 37 Hz oscillation appears after a firmware change

**Symptom.** A narrow spectral line at 37.0 Hz, stable in frequency, amplitude a few microvolts,
present on both channels but with different amplitudes.

**Competing causes.**
1. A 213 Hz artifact folding at 250 Hz (`|213 − 250| = 37`).
2. Genuine 37 Hz physiological content (unusual; the cardiac band of interest is lower).
3. A 37 Hz resonance in the electrode-lead mechanical assembly.

**Discriminating checks.** Change the sample rate to 500 Hz and recompute. A folded component
moves: at 500 Hz, 213 Hz stays at 213 Hz and the 37 Hz line disappears from the in-band region.
A mechanical resonance stays near 37 Hz regardless of rate. A physiological component stays at
37 Hz and scales with cardiac loading. Additionally, check whether the line is present when the
input is a shorted electrode: a folded artifact disappears, a self-generated resonance does not.

**Correction.** Increase analog attenuation at 213 Hz or raise the sample rate. Do not notch at
37 Hz: that would suppress genuine cardiac content between 30 and 40 Hz and would leave the
213 Hz source untouched.

**Verification.** With the input shorted, the 37 Hz bin should fall below the noise floor implied
by the measured `sigma_q`. With a known 213 Hz injected tone, the measured 37 Hz amplitude should
match `A * |H(213)|` within the filter tolerance.

### 8.2 A 50 Hz component survives every filter change

**Symptom.** The 50 Hz line amplitude is unchanged whether the analog corner is set to 45 or 55 Hz,
and unchanged whether a digital low-pass is enabled.

**Competing causes.**
1. The interference is entering after the analog filter — for example on the reference electrode
   or through a ground loop into the ADC input.
2. The interference is common-mode and the differential stage's CMRR is collapsing at 50 Hz.
3. The digital filter being toggled is applied after the source of the contamination in the
   signal chain.

**Discriminating checks.** Short both inputs together at the connector: if the line persists, the
contamination is downstream of the connector. Change the analog corner and measure the 50 Hz level
directly at the ADC pin with a probe: if the level does not move, the filter is not in the path
being blamed. Compare a driven-right-leg or reference-electrode variant: a common-mode problem
responds to the reference configuration, a ground loop does not.

**Correction.** Fix the reference and grounding topology; then re-check the analog corner. A
digital notch is a diagnostic instrument here, not a fix, because it hides the ingress without
removing it and its 1.0 s settling transient damages the first second of every segment.

**Verification.** The 50 Hz amplitude at the ADC pin should drop with the analog corner. If it does
not, the analog front-end schematic being assumed is not the one in the product.

### 8.3 Amplitude reads 3.9 dB low on a known calibration tone

**Symptom.** A 1 mV, 60 Hz calibration tone measures 0.64 mV in the spectrum, consistently across
segments, with no change in shape.

**Competing causes.**
1. Scalloping: the tone is not on a bin, so the windowed peak is attenuated.
2. The tone is on a bin but the window's coherent gain was not divided out.
3. The signal genuinely lost amplitude through a filter corner near 60 Hz.

**Discriminating checks.** Compute `f_tone * T_window`; if it is not an integer, scalloping is
present. Redo the estimate as the sum of squared magnitudes across the mainlobe and compare — a
scalloped peak recovers its energy when integrated, whereas a filtered signal does not. Then check
the filter response at 60 Hz: an 8th-order corner at 45 Hz attenuates 60 Hz by 20.0 dB, which is
far more than 3.9 dB, so a filter cannot be the cause of a 3.9 dB loss.

**Correction.** Use a coherent window length, or apply the window's amplitude correction, or
integrate the mainlobe. State in the report which estimator was used.

**Verification.** Inject a tone at an exact bin frequency and confirm the measured amplitude
matches the injected amplitude within 0.1 dB.

### 8.4 Beat-to-beat intervals jitter by exactly one sample

**Symptom.** RR intervals quantize to multiples of 4 ms with no intermediate values, and the
derived variability metric shows a spurious component at the frame rate.

**Competing causes.**
1. Peak detection on integer sample indices with no sub-sample interpolation.
2. Genuine sampling-phase variation (a real effect, but small).
3. Timestamp truncation to milliseconds in the storage layer.

**Discriminating checks.** Compute the autocorrelation of the RR series: a quantization artifact
has a comb at multiples of `Ts` in the interval histogram and no structure beyond that. Interpolate
the peak position with a parabolic fit on three samples and re-derive; a quantization artifact
disappears, a genuine phase variation does not. Inspect raw timestamps: if all timestamps are exact
multiples of 4 ms, option 3 is confirmed independently of the peak detector.

**Correction.** Interpolate peak location (parabolic or matched-filter), and store timestamps at a
resolution finer than the sample interval if the downstream metric needs it.

**Verification.** The interval histogram should fill intermediate values; the frame-rate component
of the variability spectrum should fall below the physiological band of interest.

### 8.5 Baseline wander appears only after decimation

**Symptom.** A slow oscillation below 1 Hz that is absent at the acquisition rate and present
after a 4:1 decimation.

**Competing causes.**
1. A near-DC component above the new Nyquist that folded to a low output frequency.
2. A high-pass filter whose transient was triggered by the decimation boundary.
3. A genuine respiratory component that was always there and became visible at lower resolution.

**Discriminating checks.** Compare the pre-decimation spectrum around `fs_out/2`: a folded
component has a discrete line there. Check the decimation filter's stopband attenuation at that
line's frequency: if it is under 40 dB, folding is plausible. Re-run without the high-pass: if the
wander persists, cause 2 is excluded.

**Correction.** Strengthen the decimation filter's stopband, or move the high-pass corner below the
respiratory band and accept a longer settling time.

**Verification.** The output spectrum below `fs_out/2` should contain no line whose pre-decimation
frequency lies in the fold region.

### 8.6 Quantization spurs at exactly `fs/4`

**Symptom.** Discrete spurs at `fs/4`, `fs/2` and combinations, amplitude proportional to signal
amplitude, disappearing when the input is noisy.

**Competing causes.**
1. Uncorrelated quantization error assumption violated: the input is periodic with the sample
   clock, so error is deterministic.
2. Converter DNL creating code-width errors at specific transitions.
3. A clock feedthrough path coupling into the analog input.

**Discriminating checks.** Add a small dither (half an LSB of noise) before conversion or
numerically before requantization: quantization spurs collapse, clock feedthrough and DNL do not.
Sweep the input frequency slowly: DNL spurs appear at code transitions and move with amplitude,
whereas `fs/4` spurs track the clock.

**Correction.** Dither, or accept the spurs when the downstream detector is amplitude-based rather
than spectral.

**Verification.** Spur level relative to the fundamental should drop by more than 20 dB with
dither applied.
