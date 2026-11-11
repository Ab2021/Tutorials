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
