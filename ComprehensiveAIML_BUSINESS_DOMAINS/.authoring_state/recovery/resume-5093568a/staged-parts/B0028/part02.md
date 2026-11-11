
### 4.4 Aliasing of a two-tone input and the beat frequency

**FORMULA.** For `x(t) = A1·cos(2π f1 t) + A2·cos(2π f2 t)` sampled at `fs`, the sample sequence is the sum of the two folded components. When `f1` and `f2` are close, the envelope of the sum varies at the difference frequency `|f1 − f2|`, so the sampled record shows an amplitude modulation whose period is `1/|f1 − f2|`.

**WHERE.** `A1`, `A2` are amplitudes in millivolts; `f1`, `f2` are analog frequencies in hertz; `fs` is the rate in hertz. The envelope frequency is `|f1 − f2|` in hertz only when both tones survive the analog filter; after folding the envelope of the sampled sequence varies at `|fa1 − fa2|`.

**CONSTRAINT.** Both tones are stationary over the window, the rate is constant, and no clipping occurs. If the sum exceeds the converter's range, the observed envelope is a product of the true sum and a nonlinear clipping function, and the difference frequency is no longer the whole story.

**NUMERICAL EXAMPLE — ILLUSTRATIVE.** Take a 50 Hz interferer at 1.0 mV and a 60 Hz interferer at 0.6 mV, sampled at 360 Hz. Both are below Nyquist, so both pass at their own frequencies: `fa1 = 50`, `fa2 = 60`. The envelope period is `1/|60 − 50| = 0.1 s`, which is `0.1 × 360 = 36` samples. A naive peak detector sees a 10 Hz "rhythm" of maximum amplitude `1.0 + 0.6 = 1.6 mV` and minimum `|1.0 − 0.6| = 0.4 mV`.

Now sample the same two tones at 100 Hz. `fN = 50 Hz`, so 50 Hz sits exactly at Nyquist and 60 Hz folds to 40 Hz. The envelope period remains 0.1 s but now reflects one component at the Nyquist edge and one at 40 Hz. The observed amplitude modulation is unchanged in period, which shows that the *envelope* is preserved while the *carrier* structure is destroyed.

**INTERPRETATION.** A 10 Hz envelope on a 1-second-per-cycle trace is not necessarily a 10 Hz signal. Check whether the trace is the envelope of two nearby higher-frequency components before attributing the modulation to physiology. The discriminator is the high-frequency content of the trace: a genuine 10 Hz oscillation has slope consistent with 10 Hz; an envelope of two 50–60 Hz tones has slope consistent with 50–60 Hz on each half cycle.

**Boundary.** If one of the two tones is above Nyquist and the other below, the difference frequency after folding can land anywhere in the baseband, including at a frequency unrelated to either source's separation. At 360 Hz, 50 Hz and 410 Hz fold to 50 Hz and 50 Hz respectively (410 − 360 = 50), so the two tones become *coincident* and the envelope vanishes rather than beating. A measured absence of expected beating is therefore evidence about the rate, not about the sources.

### 4.5 Aperture effect: the sample-and-hold as a low-pass

**FORMULA.** A sample-and-hold that averages over an aperture `τ` multiplies the sampled spectrum by `sinc(f·τ) = sin(π f τ)/(π f τ)`, with magnitude attenuation `20·log10|sinc(f τ)|` decibels.

**WHERE.** `τ` is the aperture in seconds, `f` is the analog frequency in hertz, and the attenuation is a pure amplitude factor with no phase contribution beyond the linear ramp from the delay. When `τ = Ts` (a full-period hold), the response has nulls at every multiple of `1/τ = fs`.

**CONSTRAINT.** The hold is rectangular in time and the input is stationary over the aperture. Track-and-hold circuits with a short aperture approximate `τ → 0` and have negligible attenuation; integrating converters with a long aperture do not.

**NUMERICAL EXAMPLE — ILLUSTRATIVE, full-period hold at `fs = 360 Hz`.** With `τ = 1/360 s`:

```text
f =  40 Hz : f*tau = 0.1111 -> |sinc| = 0.9798 -> -0.18 dB
f = 100 Hz : f*tau = 0.2778 -> |sinc| = 0.8778 -> -1.13 dB
f = 150 Hz : f*tau = 0.4167 -> |sinc| = 0.7379 -> -2.64 dB
f = 180 Hz : f*tau = 0.5000 -> |sinc| = 0.6366 -> -3.92 dB
```

**INTERPRETATION.** A full-period hold attenuates a 40 Hz component by less than a fifth of a decibel — negligible for a rhythm task — but attenuates a component at Nyquist by 3.9 dB, which is a 36 percent amplitude error. Any task that measures amplitude near the top of the band must either use a short aperture or correct for the sinc factor. This is also why the sinc response is a partial anti-alias filter: it reaches a null at `fs` and beyond, but between `fN` and `fs` it attenuates only mildly, so it cannot substitute for an analog filter with a specified stopband.

**Boundary.** The formula assumes the hold is exactly rectangular. A converter whose sampling instant is a narrow strobe has a nearly flat response; an integrating sigma-delta modulator has a response determined by its digital filter's decimation, not by a simple sinc. Applying the full-period-hold formula to a sigma-delta converter understates its alias rejection by a large factor, so the converter architecture must be known before this formula is used.

### 4.6 DFT bin spacing and the analysis window

**FORMULA.** For `N` samples at rate `fs`, the DFT evaluates the spectrum at `N` frequencies `fk = k·fs/N` for `k = 0, 1, …, N−1`, so the spacing between adjacent bins is `Δf = fs/N`. The window duration is `T = N/fs` seconds, so `Δf = 1/T`.

**WHERE.** `N` is the number of samples in the analysis segment, `fs` is the rate in hertz, `k` is the bin index, `Δf` is the bin spacing in hertz, and `T` is the window duration in seconds.

**CONSTRAINT.** The segment is contiguous, uniformly sampled, and treated as periodic for the DFT. The signal need not be periodic, but the DFT implicitly repeats the segment, so a discontinuity at the wrap point produces leakage.

**NUMERICAL EXAMPLE — ILLUSTRATIVE.** A 30-minute MIT-BIH-style record at 360 Hz has `N = 1800 × 360 = 648,000` samples and window duration 1800 s, giving `Δf = 360/648,000 = 0.0005556 Hz`, which is 0.556 mHz per bin. A 10-second segment at the same rate has `N = 3600` and `Δf = 0.1 Hz`. A 5-second segment has `Δf = 0.2 Hz`.

**Coherence check.** A 50 Hz component completes `50 × 1800 = 90,000` cycles in the 1800 s window, an integer, so it lands exactly on bin 90,000 with no leakage. A component at 50.0005 Hz completes 90,000.9 cycles and leaks across many bins. In the 10-second window a 50 Hz component completes 500 cycles, also an integer; the same window at 50.25 Hz completes 502.5 cycles and leaks.

**INTERPRETATION.** Frequency resolution and record length are the same statement in different units. Claiming 0.001 Hz resolution requires a 1000-second window; claiming the ability to separate a 0.05 Hz component from a 0.10 Hz component requires at least `1/0.05 = 20 s` of window even with a perfect estimator, and more in practice because the estimator must resolve the window's own main lobe.

**Boundary.** `Δf = 1/T` is the bin spacing, not the achievable separation. With a rectangular window, a weak component two bins away from a strong one can be buried under the strong one's side lobes, whose first side lobe is only 13.26 dB below the main lobe. A Hann window lowers the side lobes to about 31 dB below the main lobe but widens the main lobe to roughly two bins each side, raising the minimum separation. A separation claim without a stated window is a claim about arithmetic, not about measurement.

### 4.7 Filter order implied by a transition width

**FORMULA.** For a Butterworth low-pass meeting a passband ripple of `Ap` decibels at `fp` and a stopband attenuation of `As` decibels at `fst`, the minimum order with the bilinear transform and prewarping is

```text
N = 0.5 * log10( (10^(As/10) - 1) / (10^(Ap/10) - 1) ) / log10( tan(pi*fst/fs) / tan(pi*fp/fs) )
```

rounded up to the next integer, where all frequencies are in hertz and `fs` is the rate at which the digital filter runs.

**WHERE.** `N` is the filter order (number of poles); `Ap` is allowed passband attenuation in decibels at `fp`; `As` is required stopband attenuation in decibels at `fst`; `fs` is the sample rate of the filter. The ratio inside the second logarithm is the prewarped frequency ratio, which accounts for the bilinear transform's compression of the frequency axis.

**CONSTRAINT.** The filter is Butterworth (maximally flat, monotonic), implemented as a cascade of biquads, at the stated rate. For an analog filter ahead of the ADC, the analog form of the same expression uses the plain frequency ratio `fst/fp` instead of the tangent ratio; for a low ratio the two agree closely, and they diverge as either frequency approaches `fs/2`.

**NUMERICAL EXAMPLE — ILLUSTRATIVE, digital filters at `fs = 360 Hz`, `Ap = 1 dB`, `As = 60 dB`.**

```text
fst =  45 Hz : N = 0.5*log10(999999/0.2589)/log10(tan(pi*45/360)/tan(pi*40/360))
             = 0.5*6.587/0.05615...  -> 58.65 -> 59 poles
fst =  50 Hz : N = 30.61 -> 31 poles
fst =  60 Hz : N = 16.44 -> 17 poles
fst =  90 Hz : N =  7.50 ->  8 poles
fst = 120 Hz : N =  4.86 ->  5 poles
fst = 180 Hz : N =  0.20 ->  1 pole
```

The last row is instructive: with the stopband at Nyquist there is effectively no transition region available to a digital filter, and the arithmetic degenerates. The analog filter ahead of the ADC is the one that must do that work, and its stopband edge is *not* the ADC's Nyquist but the frequency at which the ADC's own folding begins to matter for the task band.

**INTERPRETATION.** A 59-pole digital filter is not a practical design; it is a statement that a 5 Hz transition at 360 Hz with 60 dB of attenuation is unreasonable, and the design should either accept less attenuation, move the stopband edge, or raise the rate. The same table at 500 Hz or 1000 Hz relaxes the requirement only modestly, because the tangent ratio depends on the ratio of frequencies rather than on the rate alone: at 1000 Hz, a 5 Hz transition (40 to 45 Hz) needs 63.6 poles, slightly *more* than at 360 Hz, because the prewarping compresses the low-frequency region less.

**Boundary.** The Butterworth order is a lower bound for the stated `Ap` and `As` with a monotonic response. An elliptic design meets the same specification at a much lower order but with passband ripple and a highly nonlinear phase; a linear-phase FIR meets it at a much higher order but with constant group delay. Order alone does not specify a filter; the response shape, the phase behaviour, and the numerical precision of the implementation all matter. A 59-pole Butterworth in single precision will not behave like its continuous-time ideal.

### 4.8 Quantization, LSB size, and the smallest measurable feature

**FORMULA.** With `b` bits over a full-scale range `VFS` (span from the most negative to the most positive code), the step size is `q = VFS / 2^b`, the ideal quantization noise power is `q^2/12`, and the signal-to-quantization-noise ratio for a full-scale sinusoid is `SNR ≈ 6.02·b + 1.76` dB.

**WHERE.** `q` is volts per code (the least significant bit); `VFS` is the peak-to-peak range in volts; `b` is the number of bits; `SNR` is the ratio of a full-scale sinusoid's root-mean-square amplitude to the root-mean-square quantization error, in decibels. For a bipolar converter with `2^(b−1)` codes on each side of zero, the per-code step is still `VFS/2^b` when the range is expressed as total span.

**CONSTRAINT.** The input spans the full range without clipping; the quantization error is uniformly distributed and independent between samples; dither is absent. With an input occupying a small fraction of the range, the achieved signal-to-noise ratio falls by the ratio of the input amplitude to full scale, so the formula is a best case.

**NUMERICAL EXAMPLE — ILLUSTRATIVE, 11 bits over a 10 mV span.**

```text
q      = 10 mV / 2048 = 4.8828 uV per code
q_rms  = 4.8828 / sqrt(12) = 1.4095 uV
full-scale sine, A = 5 mV -> rms = 5/1.4142 = 3.5355 mV
SNR    = 20*log10(3.5355 mV / 1.4095 uV) = 67.99 dB
6.02*11 + 1.76 = 68.0 dB   (agrees to 0.02 dB)
```

A 1 mV peak-to-peak QRS spans `1000 uV / 4.8828 uV = 204.8` codes, about 10 percent of the 2048-code span. A 100 uV ST-segment deviation spans 20.5 codes. A 300 uV T wave spans 61.4 codes. A 1 mV baseline wander spans 204.8 codes and therefore consumes the same code budget as the entire QRS, which is why baseline removal changes what the converter can resolve.

**INTERPRETATION.** The resolution is a property of the range choice, not of the bit count alone. Doubling the range costs one bit; halving it gains one bit but risks clipping the largest expected QRS plus electrode offset artifact. The design question is "what is the largest input I must survive", and the answer is usually set by motion artifact, not by cardiac amplitude.

**Effective bits versus nominal bits.** If the front end's own noise is 3 uV root-mean-square, the smallest signal that can be distinguished from noise is 3 uV regardless of the 4.88 uV step; the converter's step is larger than the noise only when the range and bit count are chosen to match. Conversely, if the front end noise is 20 uV root-mean-square, the 11-bit step is finer than the noise, and the effective number of bits is `log2(1000 uV / 20 uV) ≈ 5.6` for a 1 mV full-scale sinusoid. A datasheet's bit count describes the converter, not the channel.

**Boundary.** Dither, idle tones, and correlated quantization error break the uniform-error assumption, and the measured signal-to-noise ratio can be several decibels worse than the formula. A converter whose code transitions are uneven (missing codes) produces distortion that no increase in rate can remove.

### 4.9 Decimation and the rate-reduction inequality

**FORMULA.** Decimating by an integer factor `M` produces rate `fs/M`; to do so without folding task-band energy, the signal must be low-pass filtered at the *original* rate with a stopband edge at or below `fs/(2M)`. In symbols: `fstop_required ≤ fs/(2M)`, and the attenuation at that edge must exceed the interferer amplitude divided by the tolerable residue.

**WHERE.** `M` is the decimation factor (a positive integer); `fs` is the original rate in hertz; `fs/M` is the new rate; `fstop_required` is the highest frequency that must be suppressed before the rate reduction, in hertz.

**CONSTRAINT.** The filter is applied before the samples are discarded, at the original rate. Applying it after discarding samples provides no protection: the information is already folded, and a filter cannot unfold it.

**NUMERICAL EXAMPLE — ILLUSTRATIVE, from 360 Hz to 90 Hz with `M = 4`.** The new Nyquist is `90/2 = 45 Hz`. A task band of 0.5–40 Hz leaves a 5 Hz transition from 40 Hz to 45 Hz, and Section 4.7's order expression gives `N = 58.65`, so 59 poles, for 60 dB at 45 Hz; demanding 80 dB at the same edge raises it to `N = 76.45`, so 77 poles. Relaxing the stopband edge to 50 Hz — which admits a small amount of energy between 45 and 50 Hz, acceptable only if the analog front end already attenuates that region — costs 31 poles at 60 dB and 40 poles at 80 dB. Every one of those numbers is an unreasonable filter, which is itself the finding.

**INTERPRETATION.** Decimation by four to reach a 40 Hz task band leaves almost no transition room at the original rate, which is why multirate chains decimate in stages: 360 → 180 → 90, with a filter before each stage. The first stage runs at 360 Hz and needs a stopband at `360/(2·2) = 90 Hz`, a 50 Hz transition, which costs 8 poles for 60 dB. The second runs at 180 Hz and needs a stopband at `180/(2·2) = 45 Hz`, a 5 Hz transition, which costs `N = 43.23`, so 44 poles. Staging does not reduce the pole count of the final sharp filter, but it lets that filter run at 180 Hz instead of 360 Hz, halving its arithmetic per output sample; the cheap first stage then removes the energy that would otherwise have forced the sharp filter to also attenuate the 90–180 Hz region.

**Boundary — non-integer factors.** Changing from 360 Hz to 250 Hz requires the ratio 25/36: interpolate by 25, filter, decimate by 36. The intermediate rate is 9000 Hz, and the anti-imaging and anti-alias filters operate there. Any resampling path that rounds the ratio (for example to 0.7) introduces a rate error of `0.6944/0.7 − 1 = −0.8` percent, which over a 30-minute record is a 14.4-second timing error — larger than any plausible beat-to-beat variability being studied.

### 4.10 Validity summary for Section 4

| Result | Holds when | Fails when |
|---|---|---|
| `Ts = 1/fs` | Grid is uniform, index is contiguous | Dropped frames, variable-rate recording |
| `fN = fs/2` | Real-valued uniform sampling | Complex or analytic sampling, non-uniform grids |
| `fa = |f − k·fs|` | Single stationary sinusoid, constant rate | Broadband or chirped interferers |
| Envelope `|f1 − f2|` | Two stationary tones, no clipping | Clipping, more than two tones |
| `sinc(f·τ)` | Rectangular full-period hold | Strobe sampling, sigma-delta decimation |
| `Δf = fs/N = 1/T` | Rectangular-window DFT of a contiguous segment | Windowed, zero-padded, or segmented analysis |
| Butterworth `N` | Monotonic response, stated `Ap`/`As` | Elliptic, Chebyshev, FIR designs |
| `q = VFS/2^b` | Range spans the input, no missing codes | Clipping, sub-range operation, dither |
| `fstop ≤ fs/(2M)` | Filter applied before decimation at the old rate | Filter after decimation, or absent |

**Takeaway for Section 4.** Every formula above is a relation between a *rate*, a *frequency*, and a *condition*. Dropping the condition is what turns a derivation into a superstition.
