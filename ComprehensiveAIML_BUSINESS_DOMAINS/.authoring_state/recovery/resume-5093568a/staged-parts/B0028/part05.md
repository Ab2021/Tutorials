
## 9. Complexity & Cost Analysis

Cost in this domain has four currencies: arithmetic, memory, latency, and wall-clock storage. They trade against each other, and a design that optimises one usually pays in another.

### 9.1 Arithmetic cost of the anti-alias filter

A biquad section costs about five multiplications and four additions per input sample in transposed direct form II; a filter of order `N` uses `N/2` sections. Per channel, per 30-minute record at 360 Hz:

```text
N = 648,000 samples per channel

 8 poles -> 4 biquads  -> 4*5*648000  = 12,960,000 mults   = 12.96 M
31 poles -> 15.5 bq   -> 15.5*5*648000 = 50,220,000 mults = 50.22 M
59 poles -> 29.5 bq   -> 29.5*5*648000 = 95,580,000 mults = 95.58 M
```

Those numbers are per channel, and they are recomputed from the sample count rather than quoted. Doubling the rate doubles them; doubling the pole count doubles them again. This is why the choice of a 5 Hz versus a 140 Hz transition is not an academic question: it is a factor of eight in the filter's arithmetic, and in an implanted or battery-powered device it is a factor of eight in a large share of the power budget.

### 9.2 Arithmetic cost of the analysis

The same 648,000-sample segment analysed two ways:

```text
Direct DFT:  N^2 = 648000^2 = 4.199e11 complex mults ~ 419.9 G
Radix-2 FFT: 0.5*N*log2(N) = 0.5*648000*19.3056 = 6.255e6 ~ 6.26 M
Ratio:       419.9e9 / 6.255e6 = 67,131x
```

A factor of 67,131 is the difference between "this analysis is impossible interactively" and "this analysis runs in a fraction of a second". The 648,000 length is not a power of two, so a real implementation uses either zero-padding to 1,048,576 (which changes the bin spacing to `360/1048576 = 0.000343 Hz` and adds 61.8 percent of padding, interpolating the spectrum without adding information) or a mixed-radix algorithm. Both are acceptable; silently pretending the length is a power of two is not.

### 9.3 Latency

Latency matters when the decision is real-time: an alarm, a closed-loop stimulator, or a display that a clinician reads while the patient is on the table.

```text
Linear-phase FIR, 131 taps at 360 Hz: delay = (131-1)/2 / 360 = 180.6 ms
Linear-phase FIR, 261 taps at 360 Hz: delay = (261-1)/2 / 360 = 361.1 ms
Butterworth, 8 poles, fc = 40 Hz:    group delay at DC ~ 20.4 ms
Butterworth, 31 poles, fc = 40 Hz:   group delay at DC ~ 78.6 ms
Butterworth, 59 poles, fc = 40 Hz:   group delay at DC ~ 149.5 ms
```

The IIR group delays are computed from `τ ≈ 1/(2π·fc·sin(π/(2N)))`, which is the low-frequency limit for a Butterworth of order `N`; the delay rises near the band edge and varies across the passband, which is the phase distortion that Section 8.7 describes.

**The trade in one sentence.** A 31-pole IIR meets a 60 dB stopband at 50 Hz with 78.6 ms of delay and 50.2 M multiplications; a linear-phase FIR meeting the same specification needs on the order of 261 taps, 361.1 ms of delay, and 261 × 648,000 = 169.1 M multiplications. The FIR buys constant delay with four times the latency and three times the arithmetic.

### 9.4 Storage

```text
Two channels, 30 minutes, 11-bit packed: 2*648000*11/8 = 1,782,000 B  = 1.78 MB
Two channels, 30 minutes, 16-bit:        2*648000*2      = 2,592,000 B  = 2.59 MB
All 48 records of the public database at 16 bits: 48*2.59 MB = 124.4 MB
Eight channels, 24 hours, 16-bit at 360 Hz: 8*360*86400*2 = 497.7 MB
Same eight channels decimated to 90 Hz:     8*90*86400*2  = 124.4 MB
250 beds, eight channels, one year at 360 Hz: 45.4 TB
```

The 45.4 TB figure is arithmetic on an assumed deployment (ILLUSTRATIVE), not a measurement of any hospital. It exists to show the scale at which the rate decision becomes an infrastructure decision: at 90 Hz the same fleet needs 11.4 TB, a fourfold reduction that comes entirely from the decimation, and the decimation is only legal because the task band is 40 Hz.

### 9.5 Memory

```text
Segment of 648,000 float32 samples: 648000*4 = 2,592,000 B = 2.59 MB per channel
Full 30-minute record, two channels, float32 in memory: 5.18 MB
Spectrum of the same: 648000 complex float32 = 5.18 MB
```

A 2.59 MB working set per channel fits comfortably in any modern machine and in many embedded systems, which is why full-record analysis is practical offline. The constraint in a real-time embedded device is different: the buffer must hold at least one analysis window plus the filter state, and a 59-pole filter needs 59 state variables per channel — negligible in arithmetic terms, but a real cost in a fixed-point implementation where each state must be scaled to avoid overflow.

### 9.6 Cost of getting the decision wrong

The cost of an inadequate analog filter is not the filter's cost; it is the cost of the false conclusions the record produces. A record with a folded 350 Hz interferer that is interpreted as a 10 Hz physiological oscillation may lead to a diagnosis, a treatment change, or a rejected record — each of which costs far more than the 6-pole analog filter that would have prevented it. That asymmetry is the reason this card treats the placement of the analog filter as a first-order design decision rather than a detail.

## 10. Key Theorems / Results Reference

### 10.1 The sampling theorem (Nyquist–Shannon), in the form used here

A signal whose spectrum is zero above `B` hertz is completely determined by its samples taken at a uniform rate `fs > 2B`, and can be reconstructed by ideal interpolation. The theorem's conclusion is about *representability*, and its hypotheses are exact band limitation and exact uniform sampling. Neither holds physically, so the practical reading is a design inequality with slack (Section 4.2), not a licence to sample at exactly `2B`.

**What the theorem does not say.** It does not say that sampling at `fs > 2B` removes out-of-band energy; it says that a signal with no such energy is representable. If out-of-band energy exists, it is aliased, and the theorem's conclusion does not apply to the contaminated sequence.

### 10.2 The aliasing identity

For any integer `k`, `cos(2π(f + k·fs)t)` sampled at `1/fs` yields the same sequence as `cos(2π f t)` sampled at `1/fs`. Equivalently, the discrete-time frequency is defined modulo `fs` up to the sign convention of real signals. Section 4.3 derives this in three lines; the identity is the entire mechanism of aliasing and of every fold computed in this card.

**Corollary used constantly.** Two analog frequencies `f` and `fs − f` are indistinguishable after sampling at `fs`. In a 360 Hz system, 350 and 10 Hz are the same sequence; so are 370 and 10, and 710 and 10.

### 10.3 The Poisson summation form

Sampling in time replicates the spectrum at multiples of `fs` in frequency: `X_s(f) = fs · Σ_k X(f − k·fs)`. This is the frequency-domain statement of the same fact, and it is what makes the "copies" picture in Section 6.3 exact rather than metaphorical. Overlap of adjacent copies is aliasing, and the condition for no overlap is that `X` be zero outside `[−fs/2, fs/2]`.

### 10.4 Parseval and window energy

For a finite sequence, `Σ_n |x[n]|^2 = (1/N) Σ_k |X[k]|^2`. The identity lets a band-energy calculation be done in either domain, and it is the check that a windowed spectrum has not lost or invented energy. A window that tapers the ends reduces the measured total energy by its mean-square value, so a band-energy comparison between two windows must correct for that factor (0.375 for a Hann window applied to a stationary signal, since the mean square of a Hann taper is 3/8).

### 10.5 The uncertainty principle for the short-time transform

A window of duration `T` cannot localise frequency better than about `1/T`; a window of bandwidth `W` cannot localise time better than about `1/W`. The product `T·W` is bounded below by a constant of order one, so improving time resolution degrades frequency resolution by the same factor. This is why the 10-second segments in Section 5.7 resolve 0.1 Hz while the 30-minute record resolves 0.556 mHz: the resolution is bought with duration, and no algorithm evades the trade.

### 10.6 The quantization noise result

For a quantizer with step `q` and a uniformly distributed error, the error power is `q^2/12`, and the signal-to-noise ratio for a full-scale sinusoid is `6.02b + 1.76` dB. The result assumes the error is uniform and independent between samples, which fails for a periodic input whose period is commensurate with the sampling period; in that case the error is deterministic and can appear as harmonic distortion rather than noise.

### 10.7 Oversampling gain

Oversampling by a factor `OSR` and then filtering to the signal band spreads the same quantization noise power over a wider band and removes most of it, improving the in-band signal-to-noise ratio by `10·log10(OSR)` dB, which is about 0.5 bits per doubling and 1.0 bit per factor of four:

```text
OSR = 2  -> 3.01 dB  -> 0.50 bits
OSR = 4  -> 6.02 dB  -> 1.00 bits
OSR = 8  -> 9.03 dB  -> 1.50 bits
OSR = 16 -> 12.04 dB -> 2.00 bits
```

This is the arithmetic behind sigma-delta converters: they trade rate for resolution at a fixed and predictable exchange rate, and the exchange is only valid because the decimation filter that follows is the anti-alias filter for the lower rate.

### 10.8 Why the theorems do not settle a design

Every result above is conditional. The sampling theorem needs band limitation; the aliasing identity needs a constant rate; the quantization result needs a uniform error; the oversampling gain needs a filter that realises the band limit. A design review that cites the theorem and omits the condition has cited a slogan. The condition is the engineering.

## 11. Interview / Exam Questions

The questions mix recall, computation, and diagnosis. Each has its own answer, and the answers are not interchangeable.

**Q1. A record is described as "360 samples per second, two channels". How many samples does a 30-minute recording contain in total, and what is the rate of each channel?**

Answer: 648,000 samples per channel, so 1,296,000 in total; each channel's rate is 360 Hz. Reasoning: `1800 × 360 = 648,000`; the channel count multiplies the total but does not change the per-channel rate. A common error is to divide 360 by two, which would be right only if the two channels shared one 360 Hz stream in an interleaved time-multiplexed sense; the database description says "per channel", so they do not.

**Q2. At `fs = 360 Hz`, a 250 Hz interferer appears at what baseband frequency? Show the arithmetic.**

Answer: 110 Hz. Reasoning: `k` nearest `250/360 = 0.694` is 1, so `fa = |250 − 360| = 110 Hz`. A reader who answers 250 Hz has forgotten that the rate, not the task band, defines the baseband.

**Q3. The same 250 Hz interferer, after decimation by four to 90 Hz, appears where? Does it enter a 40 Hz task band?**

Answer: it folds again to `|110 − 90| = 20 Hz`, which is inside a 40 Hz band. Reasoning: decimation is a second sampling operation at 90 Hz, and every rate reduction re-applies the fold map. This is the question that separates a candidate who understands multirate processing from one who has memorised a single fold formula.

**Q4. Why does a digital low-pass at 40 Hz, applied at 360 Hz, fail to remove a 350 Hz interferer?**

Answer: because by the time the samples exist, the 350 Hz interferer is indistinguishable from a genuine 10 Hz signal; the digital filter sees 10 Hz and cannot tell which analog source produced it. Reasoning: the alias identity. The only fixes are analog (filter before the ADC) or a change of rate at acquisition.

**Q5. Compute the least significant bit for 11 bits over a 10 mV range, and state what a 100 uV ST deviation occupies in codes.**

Answer: `q = 10 mV / 2048 = 4.8828 uV`; 100 uV is `100/4.8828 = 20.48` codes. Reasoning: the range divided by the code count gives the step; the feature divided by the step gives the code count. A candidate who divides by 1024 (forgetting the sign bit's contribution to the span) gets 9.77 uV and answers 10.2 codes, which is wrong by a factor of two.

**Q6. A 30-minute record at 360 Hz has what DFT bin spacing, and can it distinguish 10.00 Hz from 10.05 Hz?**

Answer: `Δf = 360/648000 = 0.556 mHz`; the two components are 90 bins apart, so yes, in principle — provided both are stationary over the full 1800 s, which is a stronger requirement than the bin arithmetic suggests. Reasoning: bin spacing equals `1/T`; the resolution claim needs the stationarity proviso.

**Q7. A team reports a 10 Hz oscillation and proposes a 10 Hz digital notch. Give two independent reasons this is a poor plan.**

Answer: first, a 10 Hz notch removes genuine physiological content in the same band (T-wave and baseline components), so it destroys the signal along with the artifact; second, if the 10 Hz is a folded 350 Hz interferer, the notch hides the evidence without fixing the acquisition, and the same interferer will fold somewhere else at any other rate. Reasoning: the notch is a baseband operation and cannot distinguish sources; the correct fix is at the front end.

**Q8. What filter order does a Butterworth need at 360 Hz to reach 60 dB at 50 Hz with 1 dB of passband ripple at 40 Hz, and why is that order impractical?**

Answer: 31 poles. It is impractical because 31 poles means about 16 biquad sections, each with component tolerances that shift the composite response, a group delay of roughly 79 ms at DC, and a phase response that varies strongly across the 0.5–40 Hz band, distorting waveform features whose timing matters.

**Q9. Your colleague says "we sample at 1000 Hz, so we do not need an analog anti-alias filter". Diagnose the error and describe the experiment that would reveal it.**

Answer: the error is treating the rate as a substitute for the filter. At 1000 Hz the Nyquist frequency is 500 Hz, so any energy above 500 Hz — including muscle activity and mains harmonics — folds into the baseband. The experiment is to inject a 700 Hz tone at the electrode and measure whether it appears at 300 Hz in the stored record; if it does, there is no effective analog filter. Reasoning: the rate sets where folding starts, not whether it happens.

**Q10. Distinguish quantization error from aliasing in one sentence each, and say which one a higher sample rate can fix.**

Answer: quantization maps a continuous amplitude to a finite code, producing an amplitude error bounded by half a step regardless of frequency; aliasing maps a continuous frequency to a baseband frequency, producing a frequency error that is independent of amplitude. A higher rate fixes aliasing (by moving the fold boundary out) and does nothing for quantization, except through oversampling's noise-spreading gain, which requires a filter to realise.

**Q11. A 30-minute record shows a stable 10.0 Hz line exactly on a DFT bin. A colleague says this proves the source is coherent with the sampling clock. Is that right?**

Answer: not quite. Landing exactly on a bin proves the frequency is an integer multiple of `1/1800 Hz`, which at 10 Hz is trivially true for any source within 0.556 mHz; it does not prove coherence with the clock. Coherence would be shown by the line's width being limited by the window alone and by its phase being stable across segments — which a mains-derived interferer usually is and a biological oscillator usually is not. The test is phase stability, not bin alignment.

**Q12. Two tones at 50 Hz and 60 Hz, each about 1 mV, produce what visual pattern at 360 Hz, and how would you distinguish that pattern from a genuine 10 Hz oscillation?**

Answer: an amplitude envelope with a 0.1 s period, i.e. 36 samples, whose maximum is the sum and minimum is the difference of the amplitudes. The discriminator is the internal structure: the genuine 10 Hz oscillation completes one cycle per 36 samples with 36 evenly spaced zero crossings per period, while the envelope contains 5 or 6 full 50/60 Hz cycles within each 0.1 s. Counting zero crossings within one envelope period separates them unambiguously.

**Q13. What is the effect of a full-period sample-and-hold aperture at 360 Hz on a 180 Hz component, and why does it not remove the need for an analog filter?**

Answer: it attenuates the 180 Hz component by `20·log10|sinc(0.5)| = −3.92 dB`, a 36 percent amplitude error. It does not remove the need for an analog filter because between 180 and 360 Hz its attenuation is gradual (it reaches its first null only at 360 Hz), so a strong interferer between those frequencies still folds into the baseband with most of its amplitude intact.

**Q14. Design a two-stage decimation from 360 Hz to 90 Hz for a 40 Hz task band. State the stopband edge and pole count for each stage, and the combined latency.**

Answer: stage one runs at 360 Hz with its stopband at 90 Hz, needing 8 poles for 60 dB; stage two runs at 180 Hz with its stopband at 45 Hz, needing 44 poles. The combined group delay at DC is roughly `20.4 ms` for the 8-pole stage plus `111.5 ms` for the 44-pole stage, about 132 ms (ILLUSTRATIVE, computed from `τ ≈ 1/(2π·fc·sin(π/(2N)))` at `fc = 40 Hz`). The staging is cheaper than a single 59-pole filter at 360 Hz because the expensive stage runs at half the rate, which halves its arithmetic per output sample even though the pole count is similar.

**Q15. A record's metadata does not state the analog filter. What can and cannot be concluded about its preserved band?**

Answer: the rate, the range, and the quantization step can be concluded from the header; the preserved band cannot. The honest statement is "band unknown, upper limit 180 Hz by the sampling theorem, actual passband unverified". A practitioner who assumes a 40 Hz band because that is the usual specification has invented a fact.

**Q16. Why does a 1 ppm clock error matter more for a 24-hour record than for a 30-minute one, and what is the accumulated error in samples at 360 Hz?**

Answer: the error accumulates linearly, so it is proportional to duration: `86400 s × 1e-6 = 0.0864 s` per day, which is 31.1 samples at 360 Hz, against `1800 × 1e-6 = 1.8 ms` or 0.648 samples for 30 minutes. The relevant question is not the clock's specification but the record's duration times that specification, compared against the timing precision the task needs.
