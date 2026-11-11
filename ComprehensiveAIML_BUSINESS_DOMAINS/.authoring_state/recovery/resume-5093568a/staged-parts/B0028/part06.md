
**Q17. A record shows a flat-topped excursion at exactly the maximum code on several beats. Give the diagnostic that separates clipping from a genuinely large signal, and the reason no filter can repair it.**

Answer: read the stored integer codes, not the rendered plot; clipping produces a run of identical extreme codes, which a physiological waveform does not sustain for more than a sample or two at 2.78 ms spacing. No filter can repair it because clipping is a memoryless nonlinearity that destroys the relationship between input amplitude and output code across the whole segment; the harmonics it creates are indistinguishable from real high-frequency content.

**Q18. Why is the correct answer to "our 11-bit converter cannot see a 30 uV feature" sometimes "keep the 11-bit converter"?**

Answer: when the front end's input-referred noise already exceeds 30 uV, the converter's 4.88 uV step is finer than the noise, and a 16-bit converter digitizes noise more finely without improving the smallest detectable feature. The check is to measure the noise with the electrodes shorted or connected to a quiet dummy; if it is 20–50 uV root-mean-square, the converter is not the limit and the front end is.

**Q19. State the placement invariant and give one consequence that a reviewer could check in a schematic.**

Answer: every anti-alias filter sits before the sampling operation whose folding it prevents. In a schematic, the check is that an analog low-pass appears between the last amplifier stage and the converter's input pin, with no path that bypasses it; a digital filter downstream of the converter, however sharp, does not satisfy the invariant.

**Q20. A 5-second segment at 360 Hz yields a bin spacing of 0.2 Hz. A 0.25 Hz baseline oscillation in that segment lands between bins. What is the observed amplitude error, and which window reduces it?**

Answer: with a rectangular window and the component exactly halfway between bins, the scalloping loss is `20·log10(2/π) = −3.92 dB`, so the observed amplitude is about 63.7 percent of the true amplitude. A flat-top window reduces the scalloping error to a few hundredths of a decibel at the cost of a much wider main lobe; a Hann window reduces it to −1.42 dB, which halves the error.

**Q21. Explain why an anti-alias filter that meets 60 dB at 90 Hz may still be inadequate in a 360 Hz system.**

Answer: because the folding map brings energy from every band above 180 Hz into the baseband, and the filter's 90 Hz specification says nothing about its attenuation at 250 or 350 Hz. A 6-pole Butterworth that is 60 dB down at 90 Hz is far more than 60 dB down at 900 Hz, so in that typical case it is adequate; the failure mode is a filter with a resonance or a stopband that rises again (as an elliptic filter's does between zeros), where the specification at 90 Hz can be met while 350 Hz passes almost unimpeded.

**Q22. Why does the envelope frequency of two tones survive a sample-rate change while the carrier frequencies do not?**

Answer: the envelope is the difference of the two baseband frequencies, and folding preserves differences modulo `fs` in a way that preserves the beat when both tones fold consistently; the carriers themselves map to different baseband frequencies because the fold is many-to-one. Section 4.4 works the case: at 360 Hz the 50/60 Hz pair produces a 0.1 s envelope, and at 100 Hz the same pair folds to 50 and 40 Hz with the same 0.1 s envelope.

**Q23. A pipeline decimates 360 Hz to 250 Hz by multiplying the sample index by 0.694444. What total timing error accumulates over a 30-minute record, and what is the correct approach?**

Answer: the exact factor is 25/36 = 0.694444…, so rounding it introduces an error; if the implementation used 0.6944, the relative error is `(0.6944/0.694444) − 1 = −6.4e-5`, which over 1800 s is 0.115 s, or 41 samples at 360 Hz. The correct approach is a polyphase resampler with integer up- and down-sampling factors and a filter designed at the intermediate rate, not a floating-point index multiplication.

**Q24. Why is the claim "the anti-alias filter is 40 Hz" incomplete without a stopband specification?**

Answer: because a 40 Hz corner says where attenuation begins, not how much attenuation is reached anywhere. A first-order filter at 40 Hz gives 16 dB at 250 Hz; a 59-pole filter at 40 Hz gives more than 60 dB at 45 Hz. Both could be described as "40 Hz", and they are different instruments. The specification needs a passband edge, a stopband edge, and an attenuation.

**Q25. In one paragraph, state the single check that most reliably separates an aliased artifact from a physiological oscillation, and why it works.**

Answer: re-acquire at a different sample rate and compare. If the component's frequency moves, it is aliased, because the folding map depends on the rate while the physical source does not. If the component's frequency stays at the same absolute value, it is baseband energy and therefore either genuine or introduced by the analog chain before the converter. The check works because it varies the one parameter — the sampling rate — that participates in the alias identity but not in the physical phenomenon, and it is the only non-destructive test that does so.

### 11.1 Question index by type

| Question | Type | Computed value required |
|---:|---|---|
| 1 | recall plus arithmetic | 648,000 and 1,296,000 |
| 2 | computation | 110 Hz |
| 3 | computation plus judgement | 20 Hz, inside the band |
| 4 | mechanism | none |
| 5 | computation | 4.88 uV, 20.5 codes |
| 6 | computation plus proviso | 0.556 mHz, 90 bins |
| 7 | judgement | none |
| 8 | computation plus judgement | 31 poles |
| 9 | diagnosis | none |
| 10 | conceptual distinction | none |
| 11 | diagnosis of a claim | none |
| 12 | mechanism plus diagnostic | 36 samples, 5–6 cycles |
| 13 | computation | −3.92 dB |
| 14 | design plus computation | 8 and 44 poles, ~132 ms |
| 15 | judgement about evidence | none |
| 16 | computation plus scaling | 0.0864 s, 31.1 samples |
| 17 | diagnosis | none |
| 18 | computation plus judgement | 4.88 uV versus 20–50 uV noise |
| 19 | recall plus verification | none |
| 20 | computation | −3.92 dB, 63.7 percent |
| 21 | design critique | none |
| 22 | mechanism | 0.1 s envelope |
| 23 | computation | 0.115 s, 41 samples |
| 24 | specification critique | 16 dB versus 60 dB |
| 25 | synthesis | none |

Sixteen of the twenty-five questions require a computed number, a design decision, or a diagnosis; nine require recall, mechanism, or judgement. That ratio is intentional: a candidate who can recite the sampling theorem but cannot compute a fold or a filter order has not learned this material.

## 12. Quick Reference Card

| Item | Value or rule | Condition |
|---|---|---|
| Sample period | `Ts = 1/fs`; 2.7778 ms at 360 Hz | Uniform grid, contiguous index |
| Nyquist frequency | `fN = fs/2`; 180 Hz at 360 Hz | Real uniform sampling |
| Representability | `fs > 2B` with slack for the transition band | Signal band-limited before sampling |
| Alias map | `fa = |f − k·fs|` with `k` nearest `f/fs` | Single stationary sinusoid, constant rate |
| Mirror pair | `f` and `fs − f` are indistinguishable after sampling | Real sinusoid |
| Decimation rule | Filter at the old rate with stopband at `fs/(2M)`, then keep every `M`-th sample | Filter precedes the rate reduction |
| Least significant bit | `q = VFS/2^b`; 4.8828 uV for 11 bits over 10 mV | No clipping, no missing codes |
| Quantization SNR | `6.02b + 1.76` dB; 68.0 dB for 11 bits | Full-scale sinusoid, uniform error |
| Oversampling gain | `10·log10(OSR)` dB, 1 bit per factor of four | Followed by a filter to the signal band |
| Bin spacing | `Δf = fs/N = 1/T`; 0.556 mHz for 1800 s at 360 Hz | Contiguous segment |
| Butterworth order | `N = 0.5·log10((10^(As/10)−1)/(10^(Ap/10)−1))/log10(Ωs/Ωp)` | Monotonic response, prewarped for digital |
| Aperture attenuation | `20·log10|sinc(f·τ)|`; −3.92 dB at `fN` for a full-period hold | Rectangular hold |
| Rectangular scalloping loss | −3.92 dB at the half-bin offset | Rectangular window |
| Diagnostic priority | Metadata filter record, then `fs`-relationship check, then second channel, then re-acquisition | Always in that order |
| Placement invariant | Analog filter before the ADC; decimation filter before the rate reduction | No exceptions |

**Three sentences to keep.** A rate fixes the grid and the filter fixes the band, and only the filter can prevent folding. A spectral peak proves baseband energy exists, not which analog source produced it. Every number in this card is conditional, and the condition is the engineering.
