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

The `N^3` dependence is the important part: the standard deviation of the estimate falls as
`N^-1.5`, so doubling the observation improves precision by `2^1.5 = 2.83`, and doubling the
amplitude improves it by 2 (variance goes as `1/A^2`). The bound is a statement about a stationary
single tone with known noise; it does not license a resolution claim based on peak-picking from a
windowed transform, because peak-picking is limited by `1/T_window` and model error, both of which
the bound excludes by assumption.

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
