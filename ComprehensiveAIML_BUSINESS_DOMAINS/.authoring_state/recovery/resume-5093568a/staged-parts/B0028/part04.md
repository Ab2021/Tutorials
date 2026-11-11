
## 7. Comparison / Selection Table

### 7.1 Rates against task bands

The numbers in this table are DERIVED from Sections 4.2 and 4.7; the "practical verdict" column is a judgement and is labelled as such.

| Rate `fs` | `fN` | Gap from 40 Hz to `fN` | Analog poles for 60 dB at `fN` | Practical verdict |
|---:|---:|---:|---:|---|
| 100 Hz | 50 Hz | 10 Hz | 33.98 → 34 | Not viable for a 40 Hz task band; an analog filter of that order is impractical |
| 125 Hz | 62.5 Hz | 22.5 Hz | 16.99 → 17 | Marginal; a steep analog design with tight component tolerances |
| 250 Hz | 125 Hz | 85 Hz | 6.66 → 7 | Workable for rhythm review; one op-amp stage of four biquads |
| 360 Hz | 180 Hz | 140 Hz | 5.04 → 6 | Comfortable; the historical database rate sits here |
| 500 Hz | 250 Hz | 210 Hz | 4.14 → 5 | Comfortable with margin for ST-segment work |
| 1000 Hz | 500 Hz | 460 Hz | 3.00 → 3 | Generous; the cost is data volume, not filter difficulty |

The analog column is computed with the analog form of the order expression, `N = 0.5·log10((10^(As/10)−1)/(10^(Ap/10)−1))/log10(fN/40)`, with `Ap = 1 dB` and `As = 60 dB`. The digital column would be meaningless here: a digital filter at rate `fs` with its stopband at its own Nyquist has no transition region at all, which the expression reports as `N = 0.2` for every rate in the table. That degenerate result is the arithmetic statement of the placement invariant — the analog filter, not the digital one, is what protects `fs`.

The table's third column is the honest design variable: what matters is not the rate alone but the gap between the passband edge and the frequency where the stopband requirement applies. A 125 Hz system and a 1000 Hz system both face a 40 Hz passband edge, and the 1000 Hz system has eight times the gap, which is why its analog filter is trivial.

### 7.2 Analog filter topologies

| Topology | Passband | Phase | Order needed at 40→50 Hz, 60 dB | Where it is the right choice |
|---|---|---|---|---|
| First-order RC | Monotonic, no ripple | Nearly linear over a narrow band | Never sufficient alone | Always insufficient; useful only as a pre-filter before a sharper stage |
| Second-order Sallen-Key | Monotonic (Butterworth) or peaked | Moderate nonlinearity | Never sufficient (2 poles vs 31 needed) | Anti-alias in a 1000 Hz system with a wide transition |
| Butterworth, 4th order | Maximally flat | Progressive, moderate group-delay variation | 4 of 31 | A 360 Hz system with a 140 Hz transition |
| Butterworth, 8th order | Maximally flat | More group-delay variation, needs tight tolerances | 8 of 31 | A 360 Hz system where the stopband edge is at 90 Hz |
| Chebyshev, 6th order | Ripple `Ap` | Worse than Butterworth of the same order | equivalent to 11.29 poles, so 6 is insufficient at 1 dB ripple | Systems where passband ripple is acceptable and order is at a premium |
| Elliptic, 4th order | Ripple plus zeros | Very nonlinear near the band edge | roughly 5 of 31 for 60 dB | Only when phase is corrected downstream; common in instrumentation |
| Switched-capacitor | Depends on clock | Depends on clock ratio | Variable | Digital-friendly systems, but the clock must be clean |

**Selection rule.** Choose the lowest order that meets the stopband requirement with margin, then check the group delay against the task. A rhythm task tolerates several milliseconds of group delay; an ST-segment task tolerates much less, because the segment's amplitude is read at a specific time after the QRS. An elliptic filter that meets the attenuation at 4 poles may distort the ST segment more than a Butterworth at 8 poles, and the order comparison alone would pick the wrong one.

### 7.3 Digital filter stages before decimation

| Stage | Runs at | Stopband edge required | Poles for 60 dB, 40 Hz passband (recomputed) | Arithmetic per input sample |
|---|---:|---:|---:|---|
| 360 → 180 | 360 Hz | 90 Hz | 7.50 → 8 | 8 multiply-accumulates |
| 360 → 120 | 360 Hz | 60 Hz | 16.44 → 17 | 17 |
| 360 → 90 | 360 Hz | 45 Hz | 58.65 → 59 | 59 |
| 180 → 90 | 180 Hz | 45 Hz | 43.23 → 44 | 44 |
| 90 → 45 | 90 Hz | 22.5 Hz | undefined: the 40 Hz passband exceeds the new Nyquist | not possible |
| 45 → 25 | 45 Hz | 12.5 Hz | undefined for the same reason | not possible |

The last two rows are the important ones. A 40 Hz task band cannot survive decimation below 80 Hz at all, because the new Nyquist would fall inside the passband; the row is not a hard design, it is an impossible one. Decimating to 90 Hz is the last legal step, and it is expensive: 59 poles at 360 Hz or, staged, 8 poles at 360 Hz followed by 44 poles at 180 Hz. That is why a 40 Hz task band and a 90 Hz storage rate is a design worth stating explicitly rather than arriving at by accident.

### 7.4 Windows

| Window | Main lobe width (bins) | Highest side lobe (dB) | Scalloping loss (dB) | Best for |
|---|---|---:|---:|---|
| Rectangular | 2 | −13.26 | −3.92 | Coherent components exactly on bins |
| Hann | 4 | −31.5 | −1.42 | General spectra; the usual default |
| Hamming | 4 | −42.7 | −1.78 | When the first side lobe dominates the error |
| Blackman | 6 | −58.1 | −1.10 | Weak components near strong ones |
| Flat-top | ~10 | −93 (nominal) | ≈ −0.02 | Amplitude accuracy over resolution |

**Selection rule.** Choose rectangular only when the component frequencies are known to be coherent with the window (integer cycles), because then the leakage is zero and the 3.92 dB scalloping case never arises. Otherwise choose Hann. Choose flat-top when amplitude is the deliverable and frequency separation is not, because its wide main lobe destroys separation but its amplitude error is negligible.

### 7.5 Diagnosis strategies for a suspicious component

| Strategy | What it establishes | Cost | Failure of the strategy |
|---|---|---|---|
| Read the record's metadata filter description | Whether the preserved band is known at all | Near zero | Metadata is often absent or wrong |
| Check the frequency's exact relationship to `fs` | Whether a fold from `fs − f` is numerically possible | Near zero | Any frequency has such a relationship |
| Compare against a second channel with a different analog filter | Whether the component is front-end-dependent | Low if a second channel exists | Both channels often share the analog filter |
| Re-acquire at a different rate | The folding map changes while the source does not | High, needs access | The patient or equipment may not be available again |
| Inspect the analog signal before the ADC | Direct evidence of the analog spectrum | High, needs equipment | Cannot be done retrospectively |
| Compute a high-resolution spectrum of the raw record | Whether the line is coherent and narrow | Low | The information was already destroyed if it was folded |

**Decision rule.** Start with the two zero-cost checks. If the frequency is an exact `fs − f` relationship to a known equipment source, and the line is narrow and stationary, the fold hypothesis is strong enough to justify the expensive re-acquisition test. If the line is broad or drifting, the fold hypothesis is weaker and a physiological explanation deserves attention first.

## 8. Failure Modes & Edge Cases

Each entry below gives the symptom, the competing causes that produce that symptom, the check that discriminates between them, and the correction. The discriminating checks are the substance; a failure catalogue without them is a list of worries.

### 8.1 A 50 Hz notch that works on one record and not another

**Symptom.** A digital notch at 50 Hz removes the visible interference from record A and does nothing on record B, although both were acquired on the same device model.

**Competing causes.** (a) Record B's interference is not at 50 Hz but at a different harmonic, or at 60 Hz. (b) Record B's interference is a 50 Hz source that folded, so its baseband frequency is not 50 Hz. (c) Record B's record was resampled before the notch, so the baseband frequency shifted. (d) Record B's interference is at 50 Hz but the notch is applied after decimation to a rate where 50 Hz no longer exists.

**Discriminating check.** Compute the spectrum of the *raw* record before any filtering, and compare the observed interference frequency `fa` against the nominal source `f`. If `fa = f` the notch is correct; if `fa ≠ f` compute `f + fa` and `|f − fa|` and see whether either is a multiple of the record's `fs`. For record B at 360 Hz with an observed 10 Hz line and a nominal 50 Hz source, neither 60 nor 40 is a multiple of 360, so the source is not a folded fundamental — it is a folded 350 Hz component, and no 50 Hz notch will touch it.

**Correction.** Notch at the *observed* frequency, not the nominal one, and record why. A 10 Hz notch is dangerous in a rhythm record because it removes genuine T-wave and baseline content; the correct action is to fix the acquisition, not to notch.

### 8.2 Beat detection degrades after resampling

**Symptom.** A detector tuned on 360 Hz records loses sensitivity when the same records are resampled to 250 Hz, and the loss is concentrated in records with high heart rates.

**Competing causes.** (a) The resampling path applied no anti-alias filter, so high-frequency QRS energy folded and distorted the waveform shape. (b) The detector's filter coefficients were designed for 360 Hz and were not redesigned for 250 Hz, so its passband moved. (c) The resampling ratio was rounded, so the time base is wrong and the detector's refractory window is expressed in the wrong number of samples. (d) The QRS complex genuinely has fewer samples at 250 Hz (25 instead of 36 for a 100 ms complex), and the detector's slope estimate became noisier.

**Discriminating check.** Resample one record by an exact rational factor (25/36 for 360 → 250) with a properly designed anti-alias filter, and compare the detector's output against the same record resampled by a deliberately unfiltered path. If the two differ, cause (a) is present. Then verify the time base by injecting a synthetic impulse train at a known rate and checking that the detected intervals scale exactly by 250/360.

**Correction.** Design the resampling filter for the new rate's Nyquist frequency, recompute every filter coefficient for 250 Hz, and use an exact rational ratio rather than a rounded decimal. The fourth cause is not a defect and should be accepted: a 25-sample complex is a coarser estimate, and the detector's performance must be measured at the rate it will actually see.

### 8.3 Baseline wander that survives a high-pass

**Symptom.** A high-pass at 0.5 Hz is applied, and the plotted trace still shows a slow oscillation with a period of about 10 seconds.

**Competing causes.** (a) The high-pass was applied only to the displayed trace, not to the data used for measurement. (b) The high-pass is a first-order design whose 0.5 Hz corner attenuates a 0.1 Hz wander by only 14 dB, leaving a visible residue. (c) The wander is not below 0.5 Hz — it is a 10 Hz component produced by a folded interferer, which a high-pass cannot touch. (d) The wander is a motion artifact that also drove the amplifier into a nonlinear region, producing harmonics above 0.5 Hz that the high-pass cannot remove.

**Discriminating check.** Measure the residual's frequency, not its amplitude. If it is at 0.1 Hz, cause (b) is present and the fix is a steeper filter. If it is at 10 Hz, cause (c) is present and the fix is in the acquisition chain. If it is at a harmonic of the motion frequency — for example 2 Hz from a 1 Hz motion — cause (d) is present and no linear filter will recover the signal.

**Correction.** For (b), use a second-order or fourth-order high-pass and verify the attenuation at the wander frequency, not just at the corner. For (c), re-acquire. For (d), reject the segment rather than filtering it, because the information destroyed by clipping is not recoverable.

### 8.4 The "resolution" complaint that is really a noise complaint

**Symptom.** A team reports that the 11-bit converter is insufficient because small features "disappear", and proposes a 16-bit converter.

**Competing causes.** (a) The converter step genuinely exceeds the feature size. (b) The front-end noise exceeds the feature size, so the step is irrelevant. (c) The feature is present but the display resolution hides it. (d) The feature is below the analog filter's stopband and was removed before digitization.

**Discriminating check.** Compute the step (`10 mV / 2048 = 4.88 uV`) and compare it against the measured input-referred noise of the front end with the electrodes shorted. If the noise is 20 uV root-mean-square, the step is four times smaller than the noise, and a 16-bit converter would digitize noise more finely without improving the smallest detectable feature. Then measure the feature on a bench with a known input to separate (a) from (c).

**Correction.** If the noise dominates, the fix is the front end, not the converter. If the step dominates, either narrow the range (accepting clipping risk) or increase the bit depth. If the analog filter removed the feature, no converter change helps.

### 8.5 Clipping mistaken for a large physiological signal

**Symptom.** A trace shows a flat-topped excursion at exactly the same amplitude on several beats, read as a very large QRS or a saturation of the physiological range.

**Competing causes.** (a) The input exceeded the ±5 mV range. (b) The amplifier saturated before the converter did. (c) The stored codes are correct but the display is clamped. (d) The signal genuinely reaches the rail, as a pacemaker spike or a lead-off transient can.

**Discriminating check.** Look at the code values, not the plot. A clipped signal occupies exactly the extreme code (or a short run of the same code), which is a signature no physiological signal produces for more than a sample or two at this rate. Then check the analog gain stage's input range against the electrode offset: an electrode offset of 30 mV with a gain that maps 10 mV to full scale will saturate the amplifier long before the converter.

**Correction.** Clipping is a nonlinearity, so the correct action is to prevent it: reduce the gain, widen the range, or add a DC-restoration stage with a fast recovery. Post-hoc filtering of a clipped segment produces plausible-looking but false waveforms, which is worse than rejecting the segment.

### 8.6 A "10 Hz rhythm" that is an envelope

**Symptom.** A slow oscillation of about 10 Hz appears in the amplitude of the trace but not in its instantaneous frequency.

**Competing causes.** (a) A genuine 10 Hz physiological component. (b) The beat frequency of two nearby higher-frequency interferers (Section 4.4). (c) A mechanical resonance of the lead cable excited by movement. (d) Amplitude modulation of a higher-frequency carrier by respiration.

**Discriminating check.** Examine the waveform within one envelope period. A genuine 10 Hz component completes one full cycle in 100 ms, which is 36 samples at 360 Hz, and its zero crossings are evenly spaced. An envelope of two 50 Hz and 60 Hz tones completes 5 or 6 full cycles within each 100 ms and its apparent "cycle" is a beat pattern with internal structure. Counting zero crossings per envelope period separates the two immediately.

**Correction.** If it is an envelope, the underlying tones must be identified and suppressed at their own frequencies; filtering at 10 Hz removes the visible pattern but leaves the tones, which may still saturate the amplifier.

### 8.7 Phase distortion that shifts the ST segment

**Symptom.** After adding a sharper anti-alias filter, the ST segment's measured level changes by more than the expected 100 uV, and the change depends on heart rate.

**Competing causes.** (a) The sharper filter has a nonlinear phase response, so different frequency components of the QRS and ST segment are delayed differently. (b) The filter's passband edge is too close to the ST segment's content, so the segment is attenuated. (c) The filter's group delay varies with heart rate because the beat interval changed relative to the filter's impulse response length. (d) The measurement window for the ST segment was defined in samples, not in milliseconds, so it moved when the rate changed.

**Discriminating check.** Compute the filter's group delay across 0.5–40 Hz. A Butterworth at 8 poles has a group delay that varies by several milliseconds across that band; a linear-phase FIR has constant delay. Then measure the ST level on a synthetic waveform with a known ST offset, passed through the same filter, to separate the filter's effect from the measurement's definition.

**Correction.** Use a linear-phase design when the timing of waveform features matters, or specify the measurement in milliseconds and correct for the known group delay. Never compare ST measurements made through different filters.

### 8.8 Timestamps that are sample indices after a resample

**Symptom.** Beat annotations supplied by the acquisition device no longer align with the waveform after the record was resampled, and the misalignment grows through the record.

**Competing causes.** (a) The annotations were stored as sample indices at the original rate and were not rescaled. (b) The annotations were stored as times but the resampling path changed the time base. (c) The resampling ratio was rounded, so the mapping from old index to new index drifts. (d) The record contains dropped frames, so the index-to-time map is not linear even at the original rate.

**Discriminating check.** Convert every annotation to seconds using the original rate, then to the new index using the new rate, and compare the resulting offset at the start and end of the record. A constant offset indicates (a); a growing offset indicates (c) or (d). Then check the total sample count: `N_new` should equal `N_old × 250/360` exactly for an exact rational resample, and any residue indicates a dropped or duplicated sample.

**Correction.** Store annotations in seconds with an explicit clock reference, and derive indices at use time. A sample index without its rate is not a time.

### 8.9 Anti-alias filter specified in the wrong domain

**Symptom.** A design review states "the anti-alias filter is a 40 Hz low-pass", and the implementation places it after the ADC as a digital filter.

**Competing causes.** (a) Genuine misunderstanding of where folding happens. (b) The analog filter exists but its corner is at a different frequency than the review states. (c) The digital filter is intended as a *second* stage after an adequate analog filter, and the review omitted the analog stage. (d) The converter is a sigma-delta type with substantial internal decimation, so the effective analog requirement is different from what a Nyquist converter would need.

**Discriminating check.** Inject a signal above the Nyquist frequency — for example 350 Hz in a 360 Hz system — and observe whether it appears in the baseband. If it does, there is no effective analog anti-alias filter, regardless of what the review says. For a sigma-delta converter, repeat at several frequencies above `fs` and compare against the converter's documented digital filter response.

**Correction.** Place a real analog low-pass before the converter, sized by Section 4.7, and document its corner, order, and stopband attenuation in the record metadata. The digital filter then handles only the decimation steps, and its specification is derived from the decimation factors rather than from the sampling rate.

### 8.10 Sampling-clock drift read as heart-rate variability

**Symptom.** A record shows a slow oscillation in the measured beat interval with a period of minutes, and the oscillation is suspected to be a physiological rhythm.

**Competing causes.** (a) A genuine respiratory sinus arrhythmia at 0.2–0.3 Hz. (b) Clock drift of the acquisition device relative to the reference used to timestamp beats. (c) A dropped-frame pattern that repeats at a fixed rate. (d) An annotation error that moves single beats.

**Discriminating check.** A 100 ppm clock error accumulates `1800 s × 100e-6 = 0.18 s` over a 30-minute record, which is 64.8 samples at 360 Hz. That drift is monotonic, not oscillatory, so it produces a trend rather than a cycle. A genuine respiratory oscillation is bounded and periodic. Plot the beat-interval sequence with the trend removed: if the residual oscillation disappears, cause (b) is present.

**Correction.** Timestamp with a disciplined clock, or record the device's own clock reference alongside the samples. For retrospective analysis, estimate the drift from a known periodic reference in the record (a mains component that is coherent with the sampling clock will not show drift; one from an independent source will) and correct the time base explicitly.

### 8.11 A filter that meets its specification and still fails the task

**Symptom.** The analog filter meets its stated 60 dB stopband at 90 Hz, the converter is 11 bits, and yet the record shows a 0.3 mV artifact at 20 Hz.

**Competing causes.** (a) The interferer is inside the passband and the filter was never meant to remove it. (b) The filter's stopband specification was met at 90 Hz but not at 350 Hz, where a real component exists, because the filter's response has a resonance or a component tolerance pushed the corner. (c) The artifact is generated after the filter, in the amplifier or in the cabling. (d) The converter's own sampling aperture and charge injection produce the artifact.

**Discriminating check.** Measure the filter's actual response at the suspect frequency with a network analyzer, or inject the suspect frequency at the electrode and observe the output. A specification on paper and a response on the bench are different claims, and component tolerances of 5 percent move a Butterworth's corner enough to change the attenuation at 350 Hz by several decibels.

**Correction.** Specify the stopband with margin at the *actual* interferer frequencies, verify with a measurement rather than a calculation, and add a second stage if the first cannot meet the requirement across tolerances.

### 8.12 Edge cases worth stating explicitly

- **Nyquist-exact components.** A 180 Hz component in a 360 Hz system is sampled at one phase; its amplitude and phase are not both recoverable. A component at exactly `fs/2` in a decimated system is the common case for a 45 Hz component decimated to 90 Hz.
- **Zero-frequency components.** A DC offset produces a bin at `k = 0` whose value depends on the window; subtracting the mean before the transform changes the low-frequency result and must be stated.
- **Records shorter than one cycle.** A 0.5 Hz component has a 2-second period, so a 1-second record cannot represent it; any spectrum of a 1-second segment has a bin spacing of 1 Hz and no meaningful content below that.
- **Non-power-of-two lengths.** A 30-minute record has 648,000 samples, which is not a power of two; an FFT requires zero-padding or a mixed-radix algorithm, and zero-padding changes the bin interpolation without adding information.
- **Integer overflow in accumulation.** A 59-tap filter with 11-bit inputs accumulates up to `59 × 2047 ≈ 120,773`, which fits in 17 bits; a 32-bit accumulator is safe, a 16-bit accumulator is not.
- **Coefficient quantization.** A 59-pole Butterworth in single precision has pole locations perturbed enough to change its response near the band edge; the design must be verified in the precision it will run in.

**Takeaway for Section 8.** Each of these failures produces a symptom that has at least two plausible causes, and the discriminating check is what converts a suspicion into a diagnosis.
