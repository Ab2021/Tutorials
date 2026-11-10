---
title: "ECG Sampling and Aliasing"
topic: "ECG sampling, anti-alias filtering, quantization and spectral interpretation"
collection: "sensor-fusion-iot-realtime-perception"
level: "foundation"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# ECG Sampling and Aliasing

This sheet is derived, not measured. Every formula is standard sampling theory or standard
filter theory; every number is recomputed here from stated inputs, and every constructed signal
or device parameter is labelled ILLUSTRATIVE. The one external fact set used is the published
description of the MIT-BIH Arrhythmia Database (48 recordings, 360 samples/s per channel, 11-bit
resolution over a 10 mV range), cited for scale only. Nothing in this file is a clinical
recommendation, a device validation, or a claim about any vendor's product.

## 1. File Header

**Purpose.** Decide whether a given sample rate, analog front end, digital filter chain and
spectral inspection method can preserve a declared ECG band, and diagnose a suspicious spectral
component without treating the presence of a peak as proof of its physical origin.

**Audience.** Engineers, data scientists and analysts who inherit an ECG stream and must state
what the samples do and do not represent.

**Prerequisites.** Rates and reciprocals; the sine and cosine; decibels; complex numbers at the
level of magnitude and phase; reading a small table of filter coefficients.

**Outcome.** Given a sample rate, a declared signal band, an interferer frequency and a segment
length, the reader can compute the sample interval, the Nyquist frequency, the folded frequency
of the interferer, the attenuation a filter must provide, the residual after a chosen filter
order, the quantization noise floor, the DFT bin spacing and the coherent-window length, then
name the next diagnostic step when the spectrum disagrees with the expectation.

**Exclusions.** Clinical interpretation of rhythm or ischemia; diagnostic bandwidth
recommendations for any specific device standard; certification of a medical device; electrode
chemistry; treatment decisions. Those belong to other sheets and to regulated processes.

**Provenance.** Topic commitment inherited from the unit index for sensor fusion, IoT and
real-time perception. The sampling mathematics is textbook material (Nyquist, Poisson
summation, Bennett's quantization model, bilinear transform). The ECG scale reference is Moody
and Mark, *MIT-BIH Arrhythmia Database v1.0.0*, PhysioNet, 2005: 48 approximately half-hour
two-channel records from 47 subjects, originally digitized at 360 samples per second per channel
with 11-bit resolution across a 10 mV range. That citation fixes an order of magnitude for rate,
word length and input span. It says nothing about the acquisition chain of any new device.

**Status.** Draft. Not independently reviewed. No acquisition hardware was characterized and no
patient data was processed while writing this sheet.

**Retrieval cue.** If a component appears at an unexpected low frequency after a rate change,
read Sections 4, 6 and 8 before touching filter coefficients. If a reported amplitude looks
wrong but the shape looks right, read Section 4.3 and Section 7 first.

**Takeaway.** A sample rate is a promise about the band that was removed before conversion, not
a promise about the band that survived.

## 2. Concept Map / Mental Model

```text
electrode potential (continuous, microvolts to millivolts)
  -> electrode-skin interface and lead network          (source impedance, polarization)
  -> instrumentation amplifier and gain stage           (range, saturation, CMRR)
  -> analog anti-alias low-pass                         (order, corner, tolerance)
  -> sample-and-hold aperture and settling              (Rs * Chold time constant)
  -> ADC quantization at uniform instants               (LSB, word length, DNL)
  -> timestamped integer samples plus header metadata    (gain, units, filter state)
  -> optional digital low-pass BEFORE any rate reduction (polyphase, FIR or IIR)
  -> decimation or rational resampling                   (new Nyquist, new bins)
  -> windowed DFT or model fit over a finite segment     (leakage, scalloping)
  -> claim about physiology, interference or artifact    (with uncertainty)
```

The single structural rule that this whole sheet elaborates: an anti-alias filter must sit
before the operation whose folding it is supposed to prevent. A digital filter placed after the
ADC cannot remove energy that already folded during conversion, because the folded component is
now numerically identical to an in-band component. A digital filter placed after a decimator
cannot remove energy that folded when the rate dropped. The chain therefore needs one guard per
rate reduction, and the guard must be evaluated at the *input* rate of that stage.

Three objects are routinely conflated and should be held apart:

- The **analog band limit** — what the front-end filter actually attenuated before conversion.
- The **digital band limit** — what the discrete-time filter removed at the working rate.
- The **inspection limit** — what the window and DFT can resolve and how much amplitude error the
  window admits.

A defect in any of the three produces a plausible-looking spectrum. Only the first two can
create components that do not exist in the analog signal; the third distorts the amplitude and
width of components that do exist.

```text
State the claim you need to defend
  -> Is the band limit of the analog path documented with order, corner and tolerance?
  -> Is every rate reduction preceded by a filter whose stopband covers the new folding band?
  -> Is the inspection window long enough for the resolution claimed?
  -> Only then interpret amplitude, frequency or phase.
```

**Takeaway.** Sampling preserves a band only when something upstream made everything else
small; the DFT then reports what survived, not what was promised.
## 3. Foundational Definitions

| Term | Meaning | Units, shape, boundary |
|---|---|---|
| ECG | Potential difference between two electrodes placed on the body surface | mV in a named lead; sign depends on lead polarity |
| Source impedance | Effective impedance of electrodes plus skin plus lead | kΩ to MΩ; drives settling and noise |
| Front-end gain | Scale from electrode voltage to ADC input voltage | V/V; must be paired with the ADC span |
| Full-scale span | Voltage range the ADC maps onto its code range | mV, bipolar; symmetric about mid-code |
| ADC code | Integer output of the converter at one conversion | integer, `0 .. 2^B - 1`, B = word length |
| LSB | Voltage step represented by one code increment | `span / 2^B`; in mV or µV |
| Sampling rate `fs` | Conversions per second on one channel | Hz (samples/s); positive, ideally from a crystal |
| Sampling interval `Ts` | Uniform spacing between successive conversion instants | s; `Ts = 1/fs` |
| Nyquist frequency | Half the sampling rate | Hz; `fs/2`; a number, not a filter setting |
| Passband | Band the design intends to preserve within a stated ripple | Hz plus a dB tolerance |
| Stopband | Band the design intends to suppress to a stated level | Hz plus a dB attenuation |
| Transition band | Region between passband edge and stopband edge | Hz; width drives filter order |
| Anti-alias filter | Analog low-pass ahead of the ADC | order, corner, tolerance, temperature drift |
| Decimation | Reducing rate by an integer factor after filtering | factor M; new rate `fs/M` |
| Folding frequency | Frequency to which an out-of-band tone maps after sampling | Hz; `|f - k*fs|` for integer k |
| Quantization noise | Error of representing a real value by a code | µV RMS under the uniform-error model |
| Coherent window | Segment length that is an integer number of signal periods | samples; avoids leakage of a pure tone |
| DFT bin spacing | Frequency increment of a discrete spectrum | Hz; `fs/N` for N samples |
| Leakage | Energy of one tone spread across bins by a non-integer window | dB relative to peak |
| Scalloping | Peak-amplitude loss when a tone falls between bins | dB; window dependent |
| Settling time | Time for the sample-and-hold to reach a stated fraction of final value | s; governed by `Rs*Chold` |
| Aperture jitter | Random variation of the conversion instant | s RMS; converts to phase noise |
| ENOB | Effective number of bits implied by measured SNR | bits; `(SNR - 1.76)/6.02` |

Two of these entries are the ones most often mis-stated in incident reviews. The Nyquist
frequency is a property of the converter clock and is *not* a filter corner: a 250 Hz stream has
a 125 Hz Nyquist frequency whether or not anything was filtered, and the useful band may be 40 Hz.
Second, the LSB is a property of the span and word length together. A 16-bit converter over a
10 mV span has an LSB of 0.153 µV, which is far below the electrode noise floor; a 12-bit
converter over the same span has an LSB of 2.44 µV, which is near the quantization limit for
cardiac signals. Word length alone is not a quality statement.

## 4. Core Formulas with Full Derivation

Every block below follows the same five-part anatomy: formula, where it comes from, the
constraint under which it holds, a numerical example with intermediate values, and the
interpretation. Section 4.4 closes the set with the cases where the formulas break.

### 4.1 Base case

**Formula.**

```text
Ts      = 1 / fs
f_Nyq   = fs / 2
f_alias = | f_in - k * fs |      k = round(f_in / fs)
```

**Where.** `Ts` is the interval between uniform conversion instants in seconds; `fs` is the
conversion rate in hertz; `f_Nyq` is the highest frequency the sampled sequence can represent
unambiguously; `f_in` is the frequency of a component present at the ADC input; `k` is the
integer number of sample-rate multiples that brings `f_in` into the interval `[0, fs/2]`.

**Constraint.** The folding relation assumes uniform sampling at exactly `fs` with no missing
conversions, and a single stationary sinusoid. It says which frequency the tone *appears* at; it
says nothing about its amplitude, which depends entirely on the analog path ahead of the ADC.

**Numerical example (ILLUSTRATIVE).** Take `fs = 250 Hz` and an interference tone from a
nearby switching supply at `f_in = 213 Hz`.

```text
Step 1  Ts = 1 / 250                = 0.004 s        = 4 ms
Step 2  f_Nyq = 250 / 2             = 125 Hz
Step 3  k = round(213 / 250)        = round(0.852)   = 1
Step 4  f_alias = |213 - 1*250|     = | -37 |        = 37 Hz
Step 5  check 37 < 125              -> inside the representable band
```

The tone at 213 Hz is indistinguishable, from the samples alone, from a 37 Hz tone. A clinician
looking at the trace sees a 37 Hz oscillation that is not in the patient. Note that `k` is odd
here; an odd `k` reverses the sign of the folded component, which matters for phase reasoning and
is why an inverted-looking artifact should not be dismissed as a plotting bug.

**Interpretation.** Folding is a statement about the *mapping of the sample index*, not about
energy. Nothing was destroyed; the 213 Hz tone's energy now sits at 37 Hz, superimposed on
whatever genuine 37 Hz content existed. The only ways to prevent this are to attenuate 213 Hz
before the ADC or to sample faster. Post-processing cannot separate the two contributions, because
they are numerically identical in the sample sequence.

**Second numerical example (ILLUSTRATIVE).** Same rate, five frequencies, to show that the map is
many-to-one:

```text
f_in = 37   -> k = 0 -> alias 37     (already in band)
f_in = 150  -> k = 1 -> alias 100
f_in = 213  -> k = 1 -> alias 37
f_in = 250  -> k = 1 -> alias 0      (DC, a baseline step every cycle)
f_in = 300  -> k = 1 -> alias 50
f_in = 400  -> k = 2 -> alias 100
```

At 37 Hz the map is the identity, at 213 Hz it folds to 37, at 400 Hz it folds through `k = 2`
and lands on 100. Three distinct physical tones (100, 150, 400 Hz) all appear at 100 Hz. No
spectral estimator can invert this; the information is gone.

### 4.2 General case

**Formula — folding for a rate reduction by an integer factor.**

```text
Stage input rate      fs_in
Stage output rate     fs_out = fs_in / M
New Nyquist           fs_out / 2
Fold for tone f       f' = | f - k * fs_out |,  k = round(f / fs_out)
Guard requirement     H(f) small for all f in [fs_out/2, fs_in/2]
```

**Where.** `M` is the integer decimation factor; `H(f)` is the magnitude response of the digital
filter applied *before* the downsampler; the guard band is the set of frequencies that the output
rate cannot represent. The requirement is a stopband specification, not a corner specification.

**Constraint.** The fold formula is valid for a rational or integer rate reduction only after the
anti-alias filter; it presumes the filter runs at `fs_in` so its normalized response is evaluated
against the input rate. Aliasing at 50 Hz in a 250 Hz stream (`k` odd) also flips sign; the
magnitude spectrum hides this.

**Numerical example (ILLUSTRATIVE).** A 1000 Hz acquisition must be reduced to 250 Hz for
storage.

```text
fs_in  = 1000 Hz
fs_out = 1000 / 4 = 250 Hz
new Nyquist = 125 Hz
band to protect = 0 .. 100 Hz (declared)
guard region = 125 .. 500 Hz
worst tone    = 500 Hz   -> folds to |500 - 2*250| = 0 Hz (DC)
tone at 375   -> folds to |375 - 1*250| = 125 Hz (band edge)
```

A tone at 500 Hz arrives as a baseline offset that alternates sign every output sample, which
after the decimator looks like a DC shift or a very low-frequency wander. A tone at 375 Hz lands
exactly on the new band edge. A 61-tap Kaiser-windowed low-pass at 1000 Hz with a 85 Hz corner and
β = 5.65 attenuates 125 Hz by about 77.5 dB and 100 Hz by about 18.8 dB — so it protects the fold
but eats into the declared band. Moving the corner to 80 Hz trades that: 100 Hz is attenuated by
26.1 dB while 125 Hz reaches 64.0 dB. Neither setting is right in the abstract; the design is a
choice about how much declared band to sacrifice for guard margin.

**Interpretation.** The transition band is the whole problem. With a declared 100 Hz band and a
125 Hz folding edge, the ratio `125/100 = 1.25` is so tight that the required order becomes
absurd: for 60 dB of attenuation at that ratio a Butterworth needs about 31 poles, and for
66.2 dB (the 12-bit SNR level derived in Section 5) it needs about 34. Practical designs move the
passband edge down, raise the input rate, or accept less attenuation. Section 7 tabulates the
trade.

### 4.3 Variants

**Variant A — Butterworth magnitude at a frequency.**

```text
|H(f)|^2 = 1 / (1 + (f/fc)^(2n))
A(f)     = 10 * log10(1 + (f/fc)^(2n))           [dB of attenuation]
n        = log10(10^(A/10) - 1) / (2 * log10(f/fc))
```

**Where.** `fc` is the −3 dB corner, `n` is the filter order (number of poles), `A` is attenuation
in dB at frequency `f`. The second line inverts the first to give the order needed to reach a
specified attenuation at a specified frequency.

**Constraint.** Butterworth is maximally flat and monotonic but has no stopband ripple, so its
roll-off is only `20n` dB per decade asymptotically. It is a *lower bound* on what a well-designed
elliptic or Chebyshev filter achieves, and the inversion assumes the analog prototype response,
before any bilinear warping. For a digital design the frequency axis is warped: a specified corner
`fc` is pre-warped by `Wc = (2/Ts) * tan(pi*fc*Ts)`, which at `fs = 250 Hz`, `fc = 45 Hz` gives
`Wc = 317.31 rad/s` against a desired analog `2*pi*45 = 282.74 rad/s`, a ratio of 1.122.

**Numerical example (ILLUSTRATIVE).** `fs = 250`, corner 45 Hz, interferer at 213 Hz,
ratio `213/45 = 4.733`.

```text
n = 2 : A = 10*log10(1 + 4.733^4)   = 10*log10(1 + 501.7)   = 27.0 dB
n = 4 : A = 10*log10(1 + 4.733^8)   = 10*log10(1 + 2.518e5) = 54.0 dB
n = 6 : A = 10*log10(1 + 4.733^12)  = 10*log10(1 + 1.264e8) = 81.0 dB
```

Cross-check with the inverse formula at 54.0 dB: `log10(10^5.4 - 1)/(2*log10(4.733))` gives
`5.4 / (2*0.6752) = 4.00`, which returns the `n = 4` used above. The consistency of the forward
and inverse forms is the arithmetic check.

Interpretation: a 4th-order analog section at 45 Hz leaves 5 mV * 10^(−54.0/20) = 10.0 µV of a
5 mV interferer, which is 4.08 LSB at 12 bits over 10 mV. That is visible on a resting trace.
Two more poles drop the residual to 0.445 µV, or 0.18 LSB, which is below the quantization step.

**Variant B — quantization noise.**

```text
LSB      = span / 2^B
sigma_q  = LSB / sqrt(12)
SNR_fs   = 6.02 * B + 1.76                    [dB, full-scale sine]
SNR_rel  = 20*log10(A_signal / sigma_q)       [dB, arbitrary signal]
```

**Where.** `span` is the bipolar full-scale voltage range (mV), `B` the word length in bits,
`sigma_q` the RMS quantization error under the uniform-error assumption, `A_signal` the RMS
amplitude of the signal of interest.

**Constraint.** The ±LSB/2 uniform model requires a busy, non-degenerate input that crosses many
code boundaries within the observation window. It fails for a slowly varying or DC-dominated
signal, where quantization error is correlated with the signal and behaves like distortion rather
than white noise. It also ignores DNL and INL, which dominate real converters.

**Numerical example (ILLUSTRATIVE).** 12 bits over a 10 mV span.

```text
LSB      = 10 mV / 4096        = 2.4414 µV
sigma_q  = 2.4414 / sqrt(12)   = 0.7048 µV
SNR_fs   = 6.02*12 + 1.76      = 74.0 dB
signal   = 1 mV peak = 0.707 mV RMS
SNR_rel  = 20*log10(707.1 / 0.7048) = 60.0 dB
```

The same converter at 16 bits over the same span gives an LSB of 0.1526 µV and `sigma_q` of
0.0440 µV, a theoretical 24.1 dB improvement in quantization-limited SNR. Whether that
improvement is realizable depends on the analog noise floor, which is usually the binding
constraint: electrode and amplifier noise of a few µV RMS swamps a 0.044 µV quantization step.

**Variant C — coherent window and bin spacing.**

```text
Delta_f   = fs / N
T_window  = N / fs = 1 / Delta_f
coherent  = f_tone * T_window is an integer
```

**Where.** `Delta_f` is DFT bin spacing in Hz, `N` the number of samples, `T_window` the
observation length in seconds.

**Constraint.** Coherence removes leakage for a single pure tone only. It does not remove
scalloping if the tone is not exactly on a bin, and it does not make the estimate of a
non-stationary signal meaningful. With a rectangular window and a tone exactly on a bin, the
neighbouring bins are numerically zero; with Hann, the same tone leaks a defined amount into
±1 bin.

**Numerical example (ILLUSTRATIVE).** `fs = 250`, `N = 2048`.

```text
Delta_f  = 250 / 2048      = 0.12207 Hz
T_window = 2048 / 250      = 8.192 s
50 Hz    = 409.6 bins      -> NOT an integer -> leakage with any window
50 Hz * 8.192 s            = 409.6 periods   -> non-coherent
N = 2000 -> Delta_f = 0.125 Hz, T = 8.0 s, 50 Hz = 400 bins exactly -> coherent
N = 7500 -> Delta_f = 0.0333 Hz, T = 30 s, 50 Hz = 1500 bins exactly -> coherent
```

The practical rule: choose the window length to be an integer number of periods of the
interference you intend to suppress, or accept a defined amount of leakage and use a window whose
sidelobes are below the level you care about.

### 4.4 Limits and counterexamples

**Limit 1 — the Nyquist criterion is necessary but not sufficient.** Sampling at `2*f_max`
preserves a band only if nothing above `f_max` reaches the converter. A 250 Hz rate can represent
a 100 Hz component, but a 150 Hz interferer at full amplitude folds to 100 Hz and is
indistinguishable from it. Conversely, sampling a band-limited signal at exactly `2*f_max` with
ideal reconstruction loses amplitude at the band edge: a 100 Hz tone sampled at 250 Hz is scaled
by `sinc(100/250) = sin(0.4π)/(0.4π) = 0.7568`, i.e. −2.42 dB, because the sample-and-hold and
the reconstruction kernel are not ideal impulses. The "safe" ratio is larger than 2.

**Limit 2 — the uniform-noise model fails for small signals.** At an amplitude of a few LSB, the
quantization error of a real converter is not uniform and not white; it produces harmonic spurs.
A 5 µV peak signal with a 2.44 µV LSB spans only about 4 codes and the "0.7048 µV RMS noise"
statement is meaningless. The correct statement is that the signal is unrepresentable at that
resolution.

**Limit 3 — the folding formula assumes uniform sampling.** With jittered or missing conversion
instants the sampled sequence is not a uniform sampling of the analog signal. A small timing
error `Delta_t` produces a phase error `2*pi*f*Delta_t`, which for `f = 40 Hz` and
`Delta_t = 1 µs` is `2*pi*40*1e-6 = 2.51e-4 rad`, a spurious level near −72 dB; at
`Delta_t = 100 ns` it improves to about −92 dB. Beyond roughly 1% timing deviation the spectrum
develops sidebands that no decimation filter removes, because the error is in the time base.

**Limit 4 — filter order cannot rescue a wrong corner.** With a declared 100 Hz band and a 125 Hz
folding edge, the ratio 1.25 needs 31 poles for 60 dB. That is not a filter, it is a specification
error: it means the declared band and the chosen rate are incompatible without a much faster
converter. The counterexample is instructive: doubling the rate to 500 Hz moves the folding edge
to 250 Hz, and the ratio to the 100 Hz band becomes 2.5, which needs about 7.5 poles for 60 dB —
still demanding but buildable in an analog section.
## 5. Worked Numerical Example

The scenario is constructed and ILLUSTRATIVE: a two-channel bedside ECG front end, 10 mV bipolar
span, 12-bit converter, nominal 250 Hz per channel, a declared signal band of 0.5–40 Hz, an
8th-order analog low-pass at 45 Hz, an interfering 50 Hz mains-like tone at 5 mV amplitude and a
213 Hz switching artifact at 5 mV. Every number below is computed from those assumptions; none of
it is a measurement of a real device.

### 5.1 Step-by-step computation

```text
Given
  span        S  = 10 mV bipolar
  word        B  = 12 bit
  rate        fs = 250 Hz
  declared band  = 0.5 .. 40 Hz
  analog filter  = 8th-order Butterworth, fc = 45 Hz
  interferers    = 50 Hz @ 5 mV, 213 Hz @ 5 mV

A. Resolution
  LSB       = 10 / 4096            = 2.4414 µV
  sigma_q   = 2.4414 / 3.4641      = 0.7048 µV
  SNR_fs    = 6.02*12 + 1.76       = 74.00 dB

B. Sampling
  Ts        = 1 / 250              = 0.004 s  = 4.000 ms
  f_Nyq     = 125 Hz
  k for 213 = round(213/250) = 1
  alias     = |213 - 250|          = 37.0 Hz
  k for 50  = round(50/250) = 0    -> 50 Hz stays at 50 Hz

C. Analog attenuation (8th order, fc = 45)
  (50/45)   = 1.111111
  (50/45)^16 = 5.3966
  A(50)  = 10*log10(1 + 5.3966)     = 8.06 dB
         -> residual 5 mV * 10^(-8.06/20)  = 1.977 mV
  (213/45)^16 = 1.264e8
  A(213) = 10*log10(1 + 1.264e8)    = 108.03 dB
         -> residual 5 mV * 10^(-108.03/20) = 0.0198 µV

D. Folded residual at 37 Hz
  0.0198 µV / 2.4414 µV per LSB        = 0.0081 LSB
  -> the 213 Hz artifact is below the quantization step after folding

E. In-band contamination at 50 Hz
  1.977 mV / 2.4414 µV per LSB         = 810 LSB  (~9.7 bits of the 12 available)
  relative to a 1 mV QRS peak          = 20*log10(1/1.977) = -5.9 dB
  -> the interferer is larger than the signal; no analog order fixes this economically

F. Observation window
  30 s at 250 Hz = 7500 samples
  Delta_f = 250/7500 = 0.03333 Hz
  50 Hz * 30 s = 1500 periods  -> coherent with N = 7500
  1 Hz * 30 s  = 30 periods    -> coherent
```

### 5.2 Independent re-check

The arithmetic is re-derived by a different route so that a single algebra slip cannot pass.

```text
Check 1 (attenuation from the squared magnitude, not the log identity)
  |H(50)|^2 = 1 / (1 + (50/45)^16)
  (50/45)^2   = 1.234568
  1.234568^8  = 5.39660           [recomputed as ((1.234568^2)^2)^2]
  |H(50)|^2   = 1 / 6.39660 = 0.156333
  |H(50)|     = 0.395390
  A(50)       = -20*log10(0.395390) = 8.059 dB  [matches the 8.06 dB above]

Check 2 (residual voltage against the dB route)
  route A: 5 mV * 0.39539         = 1.97695 mV
  route B: 5 mV * 10^(-8.059/20)  = 5 * 0.39536 = 1.9768 mV
  difference 0.0002 mV -> consistent to four significant figures

Check 3 (LSB and code count)
  10 mV / 4096 = 2.44141 µV
  2.44141 * 4096 = 10.0000 mV     [exact by construction]
  1.97695 mV / 2.44141 µV = 809.8 codes -> 810 integer codes

Check 4 (folding, and the absence of a second fold)
  213 - 250 = -37; magnitude 37
  round(37/250) = 0, so 37 Hz is already inside 0 .. 125 and does not fold again
```

All four checks agree. The figure most likely to be mis-stated in review is the 213 Hz residual
(0.0081 LSB), because it depends on both the 108 dB attenuation and the microvolt conversion; both
are carried through with three significant figures above.

### 5.3 Sensitivity

Each row changes one input and holds the rest fixed. All values are computed from the same
formulas, not measured.

| Change from base case | Effect on the binding number | Consequence |
|---|---|---|
| Word length 12 → 16 bits | `sigma_q` 0.7048 → 0.0440 µV; SNR_fs 74.0 → 98.1 dB | Quantization stops mattering; an assumed 3 µV analog noise floor binds instead |
| Rate 250 → 500 Hz | 213 Hz no longer folds (`k` becomes 0) | The artifact is out of band rather than in band, but a 213 Hz tone before a 250 Hz decimation must still be filtered |
| Analog order 8 → 4 at 45 Hz | A(50) 8.06 → 5.22 dB; residual 1.977 → 2.743 mV | Interferer grows 39%; a notch or a lower corner becomes mandatory |
| Corner 45 → 55 Hz | A(50) 8.06 → 0.86 dB; residual 1.977 → 4.53 mV; A(213) 108 → 94.1 dB | In-band droop improves, 213 Hz margin still ample, 50 Hz rejection collapses |
| Interferer 5 → 20 mV | 810 → 3240 codes, and 20 mV exceeds a 10 mV span | Front end clips; no filter recovers clipped samples |
| Segment 30 s → 8.192 s | Bin spacing 0.0333 → 0.1221 Hz; 50 Hz not coherent | Leakage into neighbouring bins unless a window is applied |

The last two rows are the ones that change decisions. Amplitude sensitivity means range selection
is a sampling decision as much as rate selection: an input that clips has lost information before
any filter or estimator sees it. Window-length sensitivity means a resolution claim is only valid
for the specific segment analysed, so a conclusion drawn from an 8.192 s frame does not transfer
to a 30 s frame without re-checking the bin alignment.

## 6. Visual / Geometric Intuition

### 6.1 The folding cone

Put sampled frequency on the horizontal axis from 0 to `fs`, and let the analog input frequency
run along a diagonal that wraps: every time it crosses a multiple of `fs/2` it reflects, because
the discrete-time frequency axis is a circle of circumference `fs` cut at `fs/2` and unfolded for
display.

```text
input f   0 ---- 125 ---- 250 ---- 375 ---- 500
sampled   0 ---- 125 ---- 0   ---- 125 ---- 0
                       (250 Hz -> DC)     (500 -> DC)
```

Points at `fs/2 + x` are indistinguishable from `fs/2 - x` for a real signal. The region between
`fs/2` and `fs` is a mirror, and above `fs` the pattern repeats. That is why 213 Hz and 37 Hz
coincide: `213 = 250 − 37`, so the two sit symmetrically about the 125 Hz mirror.

The practical use of the picture is reading off the danger bands. For a 40 Hz band at 250 Hz the
dangerous input regions are `210–290` (folding into 0–40 Hz from `k = 1`) and `460–540` (from
`k = 2`). A 213 Hz switcher sits inside the first; a 50 Hz mains tone sits nowhere near one.

### 6.2 The alias-free region as a step function

Draw axes: horizontal = analog input frequency, vertical = attenuation. The anti-alias response is
a curve that is flat in the passband and dives at the corner. The requirement is a step: below
`f_pass` the curve must be near 0 dB, above `fs − f_pass` it must be under the noise floor. The
distance between those two frequencies is the transition band, and its width relative to the
corner drives the order. Plot the required step and the achievable curve together: if the curve
cannot reach the floor before the step's vertical edge, no digital cleverness afterwards fixes it.
For the 250 Hz / 40 Hz case the step is from 40 to 210 Hz, a ratio of 5.25, which is comfortable.
For the 1000 Hz / 100 Hz case the step is from 100 to 400 Hz, a ratio of 4.0, which is also
comfortable. The pathological case is 250 Hz with a declared 100 Hz band: the step is 100 to
150 Hz, a ratio of 1.5, and no practical analog filter achieves 60 dB there.

### 6.3 What one LSB looks like on a cardiac trace

A 1 mV QRS amplitude against a 2.4414 µV LSB spans 410 codes. A 50 µV T wave spans 20 codes. A
10 µV P wave spans 4 codes. That last number is the practical reason 12-bit ECG is marginal for
P-wave morphology: the wave is represented by four discrete levels plus noise, so a downstream
detector sees quantization structure as much as physiology. At 16 bits over the same span the same
P wave spans 66 codes and the quantization structure disappears below the analog noise.

### 6.4 Leakage as a comb, not a spike

A finite-window DFT of a pure tone is not a spike. A rectangular window gives a mainlobe
`2*fs/N` wide with a first sidelobe at −13.4 dB; Hann gives a mainlobe `4*fs/N` wide with a first
sidelobe at −31.5 dB; Hamming reaches −44.2 dB; Blackman −64.1 dB. So a 50 Hz tone in a 30 s
segment rectangular-windowed leaks −13.4 dB into the adjacent region. If a genuine 0.033 Hz
neighbour were 20 dB below the 50 Hz tone it would be buried. The picture to hold: every peak
carries a skirt, and the skirt height is a property of the window, not of the patient.

### 6.5 Where the energy goes when the rate drops

Consider a decimation by 4 from 1000 Hz to 250 Hz. The bands `[125, 250]`, `[250, 375]`,
`[375, 500]` all fold onto `[0, 125]` in reverse order, and every higher band `[k*125, (k+1)*125]`
folds again. Draw the output band four times, mirrored, one mirror per zone. A tone at 375 Hz in
zone 3 lands at `|375 − 250| = 125 Hz`; a tone at 300 Hz lands at 50 Hz. The visual check: count
the non-empty zones in the input spectrum. Every non-empty zone above the first is an error the
decimation filter had to prevent, and the height of that zone's energy above the filter's
stopband floor is the size of the error.
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
6 poles. The rate is doing the work that filter order cannot.

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
## 9. Complexity & Cost Analysis

Sampling decisions have three separate costs: arithmetic, memory and bandwidth, and latency. Each
scales differently, which is why a design that is cheap on one axis can be expensive on another.

### 9.1 Arithmetic cost per stage

```text
FIR direct form, N taps, rate fs:      N multiplications per input sample
                                       N * fs multiplications per second
FIR group delay (linear phase):        (N - 1) / 2 samples = (N-1)/(2*fs) seconds
IIR biquad cascade, order n:           n/2 biquads, ~5 mults each per sample
                                       ~2.5*n*fs mults per second
Radix-2 FFT, length N:                 (N/2)*log2(N) complex butterflies
                                       per block of N/fs seconds
Goertzel single bin, length N:         N real multiplications per block
```

**Worked comparison (ILLUSTRATIVE).** Protect a 40 Hz band at 250 Hz with an 8th-order IIR against
a 91-tap linear-phase FIR.

```text
IIR  : 2.5 * 8 * 250              = 5,000 multiplications/s
       group delay at 5 Hz        = 16.3 ms (from the phase slope)
       group delay at 40 Hz       = 34.3 ms (non-constant; distorts waveform shape)

FIR  : 91 * 250                   = 22,750 multiplications/s
       group delay                = (91-1)/(2*250) = 180 ms, constant at all frequencies
       storage                    = 91 coefficients + 91 history samples
```

The FIR costs 4.55 times the arithmetic and 7.2 times the mid-band delay, and buys exact linear phase. For
a monitoring display that is a bad trade: 180 ms of latency and 22.75 kMAC/s to avoid a phase
distortion the eye cannot see. For an offline morphology study that compares waveform onset times,
it is the correct trade, because a frequency-dependent delay shifts the apparent onset of the QRS
relative to the T wave by tens of milliseconds.

### 9.2 Spectral estimation cost

```text
FFT of 2048 points at 250 Hz:
  butterflies   = (2048/2) * 11           = 11,264
  block period  = 2048 / 250              = 8.192 s
  rate          = 11,264 / 8.192          = 1,375 butterflies/s

Goertzel for the two mains bins only:
  cost          = 2 * 2048                = 4,096 multiplications per block
  rate          = 4,096 / 8.192           = 500 multiplications/s

Streaming PSD (Welch, 50% overlap, 512-point segments):
  segments/s    = 250 / (512/2)           = 0.977
  cost          = 0.977 * (256*9)         = 2,251 butterflies/s
```

The lesson is that spectral cost is dominated by how often the transform is recomputed, not by the
transform length. Recomputing a 2048-point FFT every 8.192 s costs 1.4 k butterflies/s; recomputing
it every 0.5 s costs 22.5 k butterflies/s with 75% redundant overlap. Overlap is worth the cost only
when the statistic being tracked changes on that timescale.

### 9.3 Storage and bandwidth

```text
Per channel, 12-bit packed, 250 Hz:     250 * 1.5      = 375 B/s   = 1.35 MB/h
Two channels, 12-bit packed:            2 * 375        = 750 B/s   = 2.70 MB/h
Two channels, 16-bit:                   2 * 500        = 1,000 B/s = 3.60 MB/h
Two channels, 24-bit at 1000 Hz:        2 * 3000       = 6,000 B/s = 21.6 MB/h
```

Against the published MIT-BIH format figures: 48 records, two channels each, nominally 1800 s per
record at 360 Hz, 11 bits per sample.

```text
samples per record      = 360 * 1800 * 2      = 1,296,000
11-bit packed bytes     = 1,296,000 * 11/8    = 1,782,000 B   = 1.78 MB per record
16-bit bytes            = 1,296,000 * 2       = 2,592,000 B   = 2.59 MB per record
ratio                   = 2.59 / 1.78         = 1.4545        (= 16/11, as expected)
```

The 1.45 factor is the entire argument for packed 11-bit storage in that archive: a 31% saving on
one and a half million samples per record. On a single 10 mV, 12-bit channel the equivalent saving
relative to 16-bit storage is 25%, which is why packing survives in constrained devices and
disappears in systems where disk is free.

### 9.4 Power

```text
ADC figure of merit FoM   = 100 fJ per conversion-step (a plausible illustrative value)
conversions/s at 250 Hz   = 250
resolution steps          = 2^12 = 4096
power                     = 100e-15 * 4096 * 250 = 1.024e-7 W = 0.102 µW
```

At 16 bits and 1000 Hz the same figure gives `100e-15 * 65536 * 1000 = 6.55 µW`, 64 times higher.
Converter power scales with the product of resolution, rate and an efficiency factor, so the
cheapest possible sampling plan is the one that meets the noise floor with the smallest word
length at the lowest rate that protects the band. Both halves of that sentence matter: cutting the
rate without cutting the analog filter converts a power saving into an aliasing defect.

### 9.5 Latency budget

```text
Analog anti-alias group delay     : 16-34 ms for an 8th-order 45 Hz corner (frequency dependent)
Sample-and-hold aperture          : microseconds; negligible in this budget
ADC conversion + serialization    : 10-100 µs for a slow converter
Decimation FIR (61 taps at 1000 Hz): 30 ms
Post-decimation smoothing (IIR)    : 5-25 ms
Display refresh                    : 16-40 ms
Peak detection look-ahead           : 1 beat (500-1200 ms) for a robust refractory rule
```

For a monitor, the last row dominates and the first four are irrelevant. For a closed-loop
application — pacing, stimulation, or a synchronised capture trigger — the sum of the first four
becomes the deadline, and a 180 ms linear-phase filter is impossible. This is the concrete reason
a minimum-phase IIR is chosen for the real-time path even when a linear-phase FIR would be
spectrally nicer.

## 10. Key Theorems / Results Reference

### 10.1 Sampling theorem, stated correctly

A signal whose spectrum is zero above `B` Hz is uniquely determined by its samples taken at a
uniform rate greater than `2B` Hz. The theorem is an existence statement about reconstruction from
*ideal* samples. It does not say that sampling at `2B` is adequate in practice: real reconstruction
is not an ideal low-pass, real samples are held for a finite aperture, and real anti-alias filters
have a transition band. The engineering form of the theorem is that the sample rate must exceed
`2 * (f_band + transition_width)`, not `2 * f_band`.

### 10.2 Poisson summation and the alias formula

The sampled signal's spectrum is the sum of shifted copies of the analog spectrum:
`X_s(f) = fs * sum_k X(f - k*fs)`. Each copy contributes energy at every frequency, so a component
at `f_in` contributes to the observed spectrum at `|f_in - k*fs|` for every integer `k`. The alias
formula in Section 4.1 is the single-term version of this sum, valid when only one term lands in
the observed band.

The practical corollary: if the analog spectrum has energy at many frequencies, the observed
spectrum at a low frequency is the sum of many contributions, not one. That is why a folded
broadband artifact (electrocautery, for example) appears as raised noise rather than a discrete
line.

### 10.3 Nyquist–Shannon for a bandpass signal

A signal confined to `[f_L, f_H]` can be sampled at a rate as low as `2*(f_H - f_L)` provided the
band is placed correctly, because the copies can be made to tile without overlap. For ECG this is
rarely useful — the band of interest starts near DC — but it matters for a modulated carrier path
and it is the mechanism behind undersampling receivers. The requirement is that `f_H/(f_H - f_L)`
be an integer, or that the placement be arranged with a guard band.

### 10.4 Bennett's quantization model

For a quantizer with uniform step `q`, an input that is busy across many steps, and no overload, the
error is approximately uniform on `[-q/2, q/2]`, independent of the input, with variance `q^2/12`.
The model's preconditions are exactly the ones listed in Section 4.3: busy input, no clipping, many
steps spanned. Its failure is the subject of failure mode 8.6.

A second result follows: adding `B` bits reduces the quantization noise power by `6.02` dB, which is
the origin of the `6.02*B + 1.76` full-scale-sine SNR formula. The `1.76` is the ratio between the
RMS of a full-scale sine (`A/sqrt(2)`) and the RMS of the quantization error (`q/sqrt(12)`) written
in decibels, for `A = q*2^(B-1)`.

```text
SNR = 20*log10( (q*2^(B-1)/sqrt(2)) / (q/sqrt(12)) )
    = 20*log10( 2^(B-1) * sqrt(6) )
    = 6.02*(B-1) + 7.78
    = 6.02*B + 1.76
```

### 10.5 Bilinear transform frequency warping

The bilinear transform maps the analog frequency axis onto the unit circle by
`Omega = (2/Ts) * tan(omega*Ts/2)`, compressing high frequencies. Consequences: a digital design
must pre-warp its corner (`Wc = (2/Ts)*tan(pi*fc*Ts)`) so that the specified digital corner lands
where intended; the response near Nyquist compresses so strongly that a digital filter appears to
have infinite attenuation exactly at `fs/2`; and phase is not preserved. At `fs = 250 Hz` and
`fc = 45 Hz`, the pre-warp factor is `tan(0.5655)/0.5655 = 1.122`, so an unprewarped design lands
about 12% high in corner frequency.

### 10.6 Cramér–Rao bound for frequency estimation

For a single tone in white noise of variance `sigma^2`, `N` samples, amplitude `A`, the variance of
an unbiased frequency estimate is bounded below by

```text
var(f_hat) >= 12 * sigma^2 / ( (2*pi)^2 * A^2 * Ts^2 * N * (N^2 - 1) )
```

The `N^3` dependence is the important part: frequency precision improves as the cube of the record
length, not linearly. Doubling the observation from 8.192 s to 16.384 s improves the frequency
estimate by a factor of about 8, while doubling the amplitude improves it by 4. This is why
estimating a heart-rate variability spectral peak from a 30 s frame is a different problem from
estimating it from a 5-minute frame, and why a claim of 0.001 Hz resolution from an 8 s segment is
not supportable regardless of how the spectrum is interpolated.

### 10.7 The uncertainty principle for windowed spectra

`Delta_f * Delta_t >= 1/(4*pi)` for Gaussian windows, and in practice the product of mainlobe width
and observation time is of order one for every practical window. Stated operationally: the smallest
frequency separation you can resolve is approximately `1/T_window`, and no zero-padding, spectral
interpolation, or model-fitting changes that. Zero-padding from 2048 to 8192 samples changes the bin
spacing from 0.1221 Hz to 0.0305 Hz while the true resolution stays `1/8.192 = 0.1221 Hz`; the
interpolated spectrum is a smooth curve through the same information, not new information.

### 10.8 Sampling of a non-stationary signal

None of the above theorems apply to a signal whose spectrum changes during the observation. The
heart is non-stationary on the scale of a beat, so a spectrum computed over 30 s is an average over
roughly 35 beats with different instantaneous rates. The correct statement about such a spectrum is
that it is an average, and the correct check is to compare spectra from two disjoint segments: if
they disagree by more than the Cramér–Rao bound predicts for a stationary tone, the stationarity
assumption is what failed, not the estimator.
## 11. Interview / Exam Questions

Fifteen questions. Ten require a computation, a selection or a diagnosis; the rest test whether a
mechanism can be stated precisely. Each answer is self-contained.

**Q1. A 250 Hz ECG stream shows a stable spectral line at 37.0 Hz that was not present at 500 Hz.
What is the most likely mechanism, and what number confirms it?**

Answer: aliasing of a 213 Hz component. At 250 Hz, `k = round(213/250) = 1` and
`|213 − 250| = 37 Hz`. At 500 Hz, `k = round(213/500) = 0`, so the component stays at 213 Hz and
nothing appears at 37 Hz. The confirming numbers are the fold relation and the disappearance of the
line when the rate doubles.

Reasoning: a real physiological line at 37 Hz would not respect the sample rate. A mechanical
resonance would not either. Only a component above the Nyquist frequency moves when the rate
changes, and the direction and magnitude of the move are predicted exactly by the fold formula.

Follow-up: what would you check before raising the analog filter order? Whether the 213 Hz source
is real and how much margin exists at that frequency — the required attenuation is set by the
amplitude you can tolerate at 37 Hz, not by a generic "high attenuation" goal.

Red flag: notching at 37 Hz, which removes genuine cardiac content and leaves the source.

Scoring: full credit for mechanism + fold arithmetic + a rate-change experiment.

**Q2. Compute the LSB and quantization noise for a 12-bit converter over a ±5 mV span, and state
whether a 10 µV peak P wave is representable.**

Answer: span is 10 mV total, so `LSB = 10 mV / 4096 = 2.4414 µV` and
`sigma_q = 2.4414/sqrt(12) = 0.7048 µV`. A 10 µV peak wave spans `10/2.4414 = 4.1` codes, i.e. it
crosses about 4 code boundaries at its peak. It is representable only as a crude staircase, and
Bennett's uniform-error model does not apply because the input does not span many codes. So: the
wave is not usable for morphology at this resolution.

Reasoning: resolution is a ratio of signal to LSB, not a statement about word length. The same
12 bits over a ±0.5 mV span would give a 0.244 µV LSB and a 41-code P wave.

Follow-up: what would you change first, word length or span? Span, if the signal range permits —
range selection buys resolution for free, whereas bits cost power and bandwidth.

Red flag: quoting "12-bit is fine for ECG" without a span.

Scoring: correct LSB, correct noise, plus an explicit statement about model validity.

**Q3. A filter must attenuate a 213 Hz interferer by at least 80 dB while passing 45 Hz. What
order Butterworth is needed, and is it buildable?**

Answer: ratio `213/45 = 4.733`. Solving `10*log10(1 + 4.733^(2n)) = 80` gives
`n = log10(10^8 − 1)/(2*log10(4.733)) = 8/1.3503 = 5.92`, so 6 poles give
`10*log10(1 + 4.733^12) = 81.0 dB`. Six poles at 45 Hz is a routine analog design: three cascaded
Sallen-Key or biquad sections. Residual: `5 mV * 10^(-81/20) = 0.445 µV`, which is 0.18 LSB at
12 bits over 10 mV.

Reasoning: the order follows from the ratio and the attenuation, and the ratio is set by the corner
choice. Moving the corner from 45 to 55 Hz drops the ratio to 3.87 and needs 6.8 poles, so the
corner has real leverage on the design.

Follow-up: what does the 6-pole filter cost in group delay? About 12 ms at low frequency rising to
25 ms at 40 Hz, which matters for a closed-loop application but not for a monitor display.

Red flag: choosing an order without stating the corner it is relative to.

Scoring: correct ratio, correct order, correct residual.

**Q4. A 1000 Hz acquisition is decimated to 250 Hz. A tone at 300 Hz is present at 5 mV before
decimation. Where does it appear, and what attenuation must the decimation filter provide?**

Answer: `k = round(300/250) = 1`, so the alias is `|300 − 250| = 50 Hz`. Because the declared band
is 40 Hz, the alias lands out of band but inside the new Nyquist (125 Hz), where it will still be
visible in a spectrum and will still contaminate any wideband analysis. The required attenuation is
set by the tolerable level at 50 Hz: to keep it at 0.5 LSB (1.22 µV) from a 5 mV input requires
`20*log10(5000/1.22) = 72.2 dB`.

Reasoning: the decimation filter must be evaluated at its input rate (1000 Hz) but specified
against the output Nyquist (125 Hz). Confusing the two rates is the classic error: a filter with
72 dB at 500 Hz but only 20 dB at 250 Hz would pass this test on the first number and fail on the
second.

Follow-up: is 72 dB realistic? Yes at a comfortable transition band. A 91-tap Kaiser design at
1000 Hz with a 85 Hz corner reaches 77.5 dB at 125 Hz, so the requirement is met with a filter that
is 91 taps long and costs 91 kMAC/s.

Red flag: computing the fold against 1000 Hz instead of 250 Hz.

Scoring: correct alias, correct stopband target, correct rate to evaluate at.

**Q5. Why can a digital notch at 50 Hz not be the primary remedy for 50 Hz interference in an ECG
front end?**

Answer: because the notch is placed after the ADC, so it removes an in-band tone rather than
preventing a folded one, and because it has a long transient. At `Q = 35` and `f0 = 50 Hz` in a
250 Hz stream the notch is only `50/35 = 1.43 Hz` wide and its impulse response decays with a time
constant of `1/(pi*1.43) = 0.223 s`, settling to 1% in about 1.0 s. Any analysis window that starts
within a second of a gain change, a lead connection, or a filter enable sees a transient larger
than the signal.

Reasoning: the notch is a valid tool when the interference is genuine narrowband in-band noise that
survived a competent analog stage. It is the wrong first tool when the interference is clipping
the front end or coupling after the filter, because in those cases the analog problem remains and
the notch merely hides the symptom while adding a transient.

Follow-up: what would you do instead? Fix the reference electrode and grounding, raise the analog
attenuation at 50 Hz if the passband allows, or average across many beats so that a stationary
interferer cancels in the ensemble.

Red flag: enabling a notch by default on every channel "to clean it up".

Scoring: transient arithmetic plus a statement of what the notch cannot fix.

**Q6. A 250 Hz stream has a declared band to 100 Hz. What analog order is needed for 60 dB at
Nyquist, and what does the answer tell you?**

Answer: ratio `125/100 = 1.25`, so
`n = log10(10^6 − 1)/(2*log10(1.25)) = 6/0.1938 = 30.96`, i.e. 31 poles. That is not an
implementable analog filter: 31 poles at 100 Hz would have enormous component sensitivity, terrible
group delay variation, and would need a matching digital section. The correct conclusion is that the
specification is inconsistent — either the band is narrower than 100 Hz, or the rate must rise.

Reasoning: order scales as `log(A)/log(ratio)`. When the ratio approaches 1 the required order
explodes, and no amount of design cleverness recovers it. This is why rates are chosen from the
band and a guard factor, not from a display refresh rate.

Follow-up: what guard factor is comfortable? A ratio of 3 or more puts the requirement in single
digits of poles; 2.5 is workable; below 2 is a warning sign.

Red flag: answering "use a 32nd-order filter".

Scoring: correct order plus recognition that the answer invalidates the specification.

**Q7. A calibration tone at 60 Hz reads 3.9 dB low in a rectangular-windowed 2048-point spectrum at
250 Hz. Diagnose.**

Answer: scalloping. `Delta_f = 250/2048 = 0.1221 Hz`, and 60 Hz corresponds to bin
`60/0.1221 = 491.52`, exactly half a bin off the grid. A rectangular window's worst-case scalloping
loss is the value of `sinc` at 0.5, `20*log10(sin(0.5π)/(0.5π)) = −3.92 dB`. The observed 3.9 dB is
that number.

Reasoning: the loss appears only for tones off the bin grid and is independent of amplitude and of
filter settings. A filter cannot produce exactly the geometric worst case, and a genuine attenuation
would vary with frequency rather than sit at the sinc value.

Follow-up: how do you recover the true amplitude? Use a coherent window length (2000 samples gives
`Delta_f = 0.125 Hz` and 60 Hz on bin 480 exactly), or sum the squared magnitudes across the
mainlobe, or divide by the window's coherent gain and correct for the offset.

Red flag: adjusting the gain calibration to compensate.

Scoring: identification of scalloping plus the arithmetic and a recovery method.

**Q8. Two acquisitions of the same patient at 200 Hz and 500 Hz show a component at 63 Hz and at
137 Hz respectively. Is either an alias, and what is the physical frequency?**

Answer: `63 Hz` is below 100 Hz (the 200 Hz Nyquist), and `137 Hz` is below 250 Hz, so neither is
folded in its own record. But the same physical tone cannot be at 63 and 137 Hz simultaneously, so
at least one record is aliased. Test the hypothesis of a 137 Hz tone at 200 Hz:
`k = round(137/200) = 1`, alias `|137 − 200| = 63 Hz`. Both observations are consistent with a
single 137 Hz physical component: 137 Hz measured directly at 500 Hz, and 63 Hz at 200 Hz by
folding.

Reasoning: the fold formula is invertible up to an unknown `k`. Two records at different rates
resolve the ambiguity because only one physical frequency produces both observations. That is the
general method for de-aliasing: change the rate and intersect the candidate sets.

Follow-up: what if both records showed the same frequency? Then either the tone is genuinely below
both Nyquist frequencies, or the folding at both rates happens to coincide — rare but possible, and
a third rate settles it.

Red flag: assuming the higher-rate record is necessarily correct without checking the analog path.

Scoring: the fold computation plus the two-record intersection argument.

**Q9. Why is a sample-and-hold settling requirement expressed as about 9 time constants, and what
happens at 12 bits with a 1 µs time constant?**

Answer: settling to half an LSB requires the error to fall below `2^-(B+1)` of full scale, so
`exp(-t/tau) < 2^-(B+1)` and `t > (B+1)*ln(2) = 13*0.693 = 9.01` time constants for `B = 12`. With
`tau = 1 µs` that is 9.01 µs of settling inside the sampling window, which is comfortable. With
`tau = 300 µs` (a high source impedance driving a small hold capacitor) it is 2.70 ms, longer than
the 4 ms sample interval at 250 Hz — the converter never settles and the error appears as
gain-dependent distortion.

Reasoning: the requirement couples the source impedance, the hold capacitance and the word length.
A high-impedance electrode with a long cable can easily push `tau` into the hundreds of microseconds.

Follow-up: what is the discriminating check? Drive the input with a fast step and look at the code
sequence for a settling tail; or compare readings at two sample rates — an unsettled input shows a
rate-dependent gain.

Red flag: assuming the ADC's specified aperture time is the settling budget.

Scoring: the 9 tau derivation plus the consequence at 250 Hz.

**Q10. State the exact condition under which the folded component in Q1 is below one LSB, given a
5 mV interferer at 213 Hz and a 12-bit converter over 10 mV.**

Answer: `1 LSB = 2.4414 µV`, so the required attenuation is
`20*log10(5000/2.4414) = 66.2 dB`. At a 45 Hz corner the ratio is 4.733, and
`10*log10(1 + 4.733^(2n)) = 66.2` gives `n = 4.90`, so 5 poles suffice (67.5 dB) and 4 poles do not
(54.0 dB). A 4-pole design leaves 10.0 µV, or 4.08 LSB, which is visible.

Reasoning: the target is set by the quantization step, not by a tradition like "60 dB". The same
computation at 16 bits would demand 101 dB, which is 8 poles at that corner — an instructive
demonstration that higher resolution raises the analog requirement rather than lowering it.

Follow-up: what if the interferer amplitude is unknown? Then the specification is not a filter
order but a measurement plan: determine the interferer amplitude at the electrode before choosing.

Red flag: choosing an order first and computing the residual afterwards.

Scoring: the 66.2 dB target, the order, and the 16-bit consequence.

**Q11. A decimation from 1000 to 250 Hz uses a filter with 72 dB at 500 Hz and 20 dB at 250 Hz. Is
that adequate?**

Answer: no. The folding edge at the output rate is 125 Hz, and everything between 125 and 500 Hz
folds into the output band. The relevant stopband is `[125, 500]`, and the quoted 72 dB applies at
500 Hz only. Energy at 250 Hz, attenuated by only 20 dB, folds to `|250 − 250| = 0` (DC) and
`|375 − 250| = 125` (band edge); energy at 300 Hz folds to 50 Hz with whatever attenuation the
filter provides there. The specification must be stated as a minimum attenuation across the whole
fold region.

Reasoning: a single-point stopband number is meaningless without its frequency. The correct
statement is "≥ 70 dB for all f in [125, 500] Hz", which is a transition-band specification.

Follow-up: what filter shape delivers that? A Kaiser-windowed FIR with a corner near 85 Hz and a
stopband from 125 Hz, about 91 taps at 1000 Hz, cost 91 kMAC/s.

Red flag: quoting a stopband attenuation without a band.

Scoring: identification of the fold region plus a corrected specification.

**Q12. Estimate the frequency precision achievable for a 0.5 mV, 40 Hz component from a 30 s
frame at 250 Hz with a 3 µV RMS noise floor, and say whether a claim of 0.001 Hz is supportable.**

Answer: two different limits must be separated. The transform's resolution is
`1/T = 1/30 = 0.0333 Hz`, so a peak-picking estimate cannot resolve features finer than that. The
Cramér–Rao bound for a parametric fit is far more permissive: with `A = 0.5 mV`,
`sigma = 3 µV`, `N = 7500`, `Ts = 0.004 s`,

```text
var(f) >= 12 * sigma^2 / ( (2*pi)^2 * A^2 * Ts^2 * N * (N^2 - 1) )
numerator   = 12 * (3e-6)^2                      = 1.08e-10
denominator = 39.478 * (5e-4)^2 * (0.004)^2 * 7500 * (7500^2 - 1)
            = 39.478 * 2.5e-7 * 1.6e-5 * 7500 * 5.625e7
            = 66.62
var(f)      = 1.08e-10 / 66.62                  = 1.62e-12
sd(f)       = 1.27e-6 Hz                        = 1.27 µHz
```

So 0.001 Hz is *not* excluded by the bound for an ideal stationary tone at that SNR — the bound is
about 785 times finer than the claim. The claim fails for a different reason: the bound assumes a
single stationary sinusoid with known noise, and a physiological 40 Hz component wanders during the
30 s frame as the instantaneous heart rate changes. Model error, not noise, sets the real precision.

Reasoning: the `N^1.5` scaling in the bound is real and large, which is why parametric estimators
beat the bin spacing. But a bound computed under a wrong model is not evidence about the system;
the discriminating check is to split the 30 s into two disjoint 15 s frames and compare estimates —
if they differ by more than the bound predicts, non-stationarity, not estimator variance, is binding.

Follow-up: what record length would make 0.001 Hz defensible for a stationary artifact such as a
mains harmonic? The bound scales as `N^-1.5`, so the current 1.27 µHz already suffices; a longer
record is needed only to average over the wander of a physiological source.

Red flag: quoting zero-padded bin spacing as the resolution, or quoting the CRB as achieved precision.

Scoring: both limits, the correct arithmetic, and the stationarity objection.

**Q13. What does an ENOB of 8.8 mean for a 12-bit converter in a specific system, and how is it
computed?**

Answer: ENOB is derived from a measured or assumed SNR: `ENOB = (SNR − 1.76)/6.02`. If the achievable
SNR against a 2.5 mV RMS signal is 54.5 dB (limited by a combined 3.33 µV RMS noise floor made of
3 µV amplifier noise, 1.26 µV jitter-induced noise at 40 Hz with 1 µs timing error, and 0.70 µV
quantization), then `ENOB = (54.5 − 1.76)/6.02 = 8.76` bits. The remaining 3.24 bits of the 12 are
consumed by analog noise, not by the converter.

```text
noise floor = sqrt(3.0^2 + 1.256^2 + 0.705^2) = sqrt(9 + 1.578 + 0.497)
            = sqrt(11.075) = 3.328 µV RMS
SNR         = 20*log10(2500 / sqrt(2) / 3.328) = 20*log10(531.1) = 54.5 dB
```

Reasoning: ENOB is a system property, not a datasheet property. Buying four more bits of converter
raises the theoretical floor by 24 dB but leaves the 3 µV analog floor unchanged, so the measured
ENOB barely moves.

Follow-up: what would raise the ENOB? Lower the amplifier noise, reduce the jitter, or narrow the
band — in that order of cost.

Red flag: reporting the datasheet ENOB as the system ENOB.

Scoring: formula, arithmetic, and the statement that ENOB is bounded by the analog chain.

**Q14. Why does zero-padding a spectrum not improve resolution, and what does it actually do?**

Answer: zero-padding interpolates the discrete-time Fourier transform between the existing bins. It
changes the bin spacing from `fs/N` to `fs/N_padded` but the information content is set by the
`N` real samples, so the true resolution remains approximately `1/T_window`. Concretely, padding
2048 samples to 8192 changes the displayed spacing from 0.1221 Hz to 0.0305 Hz while the underlying
resolution stays 0.1221 Hz. What it does buy: a smoother peak-shape estimate for interpolation
(parabolic or centroid fitting becomes more accurate), and a visually cleaner plot.

Reasoning: resolution is a property of the observation time and the window's mainlobe width. No
linear operation on a fixed set of samples creates new independent frequency information.

Follow-up: what does improve resolution? A longer record, or a parametric model with a correct
signal model — and the latter trades bias for variance.

Red flag: reporting a resolution equal to the padded bin spacing.

Scoring: distinction between interpolation and resolution, with the numbers.

**Q15. An engineer proposes sampling at exactly 80 Hz for a 40 Hz band because "Nyquist says 2×".
Give three independent reasons this is wrong.**

Answer: first, the sampling theorem requires the signal to be *strictly* band-limited below 40 Hz,
and a real ECG is not; any energy just above 40 Hz folds into the band. Second, at exactly `fs = 2B`
the reconstruction kernel is a sinc sampled at its zeros, so amplitude at the band edge is scaled by
`sinc(0.5) = 0.6366`, i.e. −3.92 dB, and the reconstruction is numerically fragile. Third, the
anti-alias filter would need an infinitely steep transition: with the band edge at 40 Hz and the fold
edge at 40 Hz the transition width is zero, so the required order is unbounded.

Reasoning: the theorem is an existence result about ideal sampling and ideal reconstruction. Every
practical use adds guard band for filter transition, aperture effects, and clock tolerance.

Follow-up: what rate would you choose? At 250 Hz the ratio is 3.125 and 8 poles give 79 dB; at 200 Hz
the ratio is 2.5 and 8 poles give 64 dB. Both are defensible; the choice depends on the band edge
tolerance you need.

Red flag: citing the theorem without the strict-band-limit precondition.

Scoring: three distinct reasons, at least one quantitative.

## 12. Quick Reference Card

| Item | Value or rule | Validity condition |
|---|---|---|
| Sample interval | `Ts = 1/fs`; 250 Hz → 4.000 ms | Uniform clock, no dropped conversions |
| Nyquist frequency | `fs/2`; 250 Hz → 125 Hz | A property of the clock, never a filter corner |
| Fold formula | `f' = \|f − k*fs\|`, `k = round(f/fs)` | Stationary tone, uniform sampling |
| Alias target | Attenuate to ≤ 1 LSB before the ADC | Set by word length and span, not by tradition |
| 1 LSB, 12 bit, 10 mV | 2.4414 µV (`span/2^B`) | Span is the total bipolar range |
| Quantization noise | `LSB/sqrt(12)` = 0.7048 µV at 12 bit/10 mV | Busy input spanning many codes |
| Full-scale sine SNR | `6.02*B + 1.76` = 74.0 dB at B = 12 | Ideal converter, no DNL/INL |
| Butterworth attenuation | `10*log10(1 + (f/fc)^(2n))` | Analog prototype, before bilinear warping |
| Order for a spec | `n = log10(10^(A/10) − 1)/(2*log10(f/fc))` | Monotonic response, no ripple |
| Guard ratio rule | `(fs/2)/f_band ≥ 3` for single-digit order; 1.25 needs 31 poles | Butterworth basis, 60 dB target |
| Bin spacing | `fs/N`; zero-padding interpolates but does not resolve | Resolution ≈ `1/T_window` |
| Coherent window | `f_tone * T_window` an integer; 50 Hz, 30 s → 1500 periods | Single tone; removes leakage only |
| Scalloping, rectangular | −3.92 dB worst case at half-bin offset | Tone off the bin grid |
| Window sidelobes | Rect −13.4, Hann −31.5, Hamming −44.2, Blackman −64.1 dB | First sidelobe, normalized |
| Decimation guard | Filter stopband must cover `[fs_out/2, fs_in/2]` before downsampling | Evaluated at the input rate |
| Notch transient | `Q = 35` at 50 Hz → 1.43 Hz wide, ~1.0 s to 1% | Second-order notch, 250 Hz rate |
| Settling budget | `(B+1)*ln(2) = 9.01` time constants | Exponential settling to half an LSB |

The card is a checklist, not a design. Every row names the condition under which the value holds;
using a row without its condition is how the failure modes in Section 8 arise. The two rows most
often applied out of context are the Nyquist frequency (treated as a filter corner) and the guard
ratio (applied to a band that was declared, not measured — an undeclared wideband artifact in the
"passband" defeats the whole calculation and is only visible in a spectrum taken at a rate high
enough to see it).
