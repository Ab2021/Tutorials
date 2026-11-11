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

Follow-up: is 72 dB realistic? Yes at a comfortable transition band. A 61-tap Kaiser design at
1000 Hz with an 85 Hz corner reaches 77.5 dB at 125 Hz, so the requirement is met with 61 taps and
61 kMAC/s of arithmetic.

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
computation at 16 bits demands `20*log10(5000/0.1526) = 90.3 dB`, which at a ratio of 4.733 needs
6.7 poles, so 7 poles at 94.5 dB — an instructive demonstration that higher resolution raises the
analog requirement rather than lowering it.

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
