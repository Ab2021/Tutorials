# ADCs and Signal Conditioning

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 04 of 14
> Estimated study time: 4 hours
> Prerequisites: 01, 03
> Version 2026.3

## 1. Why this topic exists

The analogue-to-digital converter (ADC) is the boundary between the continuous physical world and the discrete data that a model consumes. It is also the stage where a design decision becomes irreversible: once a signal is sampled below its bandwidth, or quantised below its noise, or aliased by an interferer, no downstream algorithm can recover the lost information.

The industry problem is that ADC specifications are read as if they were data properties. A datasheet says "24-bit" and the team assumes 24 bits of resolution; a datasheet says "32 kSPS" and the team assumes 32 kSPS of usable bandwidth; a datasheet says "105 dB dynamic range" and the team assumes 105 dB of system performance. Each assumption is wrong in a different way, and each wrong assumption can invalidate a model or a clinical claim.

Three decisions depend on ADC literacy:

1. **Whether the sampling rate is adequate.** The Nyquist criterion is necessary but not sufficient; the anti-alias filter and the digital filter's response determine what actually folds into the band.
2. **Whether the resolution is real.** The effective number of bits (ENOB) depends on noise, distortion, reference stability, and the signal amplitude relative to full scale. A 24-bit converter can deliver 8 effective bits in a system.
3. **Whether the latency is acceptable.** A delta-sigma converter's digital filter adds a group delay that can dominate the latency budget. A successive-approximation converter has lower latency but lower resolution.

This topic covers converter architectures, resolution and ENOB, sampling and aliasing, reference and clock design, digital filtering, calibration, and verification. It assumes the signal-chain and front-end material from Topics 01 and 03. The arithmetic is again the point: an ADC that is not budgeted is an ADC that will be discovered in the data.

A useful framing: the ADC is a promise about a range, a rate, and a noise floor. The range is set by the reference and the gain; the rate is set by the clock and the architecture; the noise floor is set by the converter, the reference, and the drive circuit. The promise is only as good as the weakest of the three.

## 2. Learning outcomes

By the end of this topic you can:

- Choose a converter architecture from the signal bandwidth, resolution, latency, and power budget.
- Compute the ideal signal-to-noise ratio (SNR), the ENOB, and the input-referred least significant bit (LSB) for a given resolution and reference.
- Explain why a 24-bit converter can deliver 8 effective bits and compute the system ENOB from the noise budget.
- Design an anti-alias filter and compute the alias attenuation at the sampling frequency.
- Compute the jitter-limited SNR for a given input frequency and clock jitter, and decide whether jitter matters.
- Compute the group delay and settling time of a delta-sigma digital filter.
- Distinguish integral nonlinearity (INL), differential nonlinearity (DNL), offset error, gain error, and missing codes.
- Write a converter verification plan using a histogram test and a fast Fourier transform (FFT).

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

An ADC performs three operations: **sample**, **quantise**, and **encode**. Sampling multiplies the signal by a train of impulses; quantisation maps each sample to the nearest code; encoding formats the code for transmission. Each operation has an error model.

```text
        anti-alias          sample-and-hold            quantiser              encoder
 signal ----------> filter ----------> S/H ----------> comparator ----------> code
   |                   |                  |                  |                    |
   |                   |                  |                  |                    |
 bandlimit           group delay        aperture jitter     LSB, INL, DNL        format
                     and droop           charge injection    missing codes
```

The model has four quantities:

| Quantity | Determined by | Failure if wrong |
|---|---|---|
| Range | Reference and gain | Clipping or wasted codes |
| Resolution | Quantiser bits and noise | False precision |
| Rate | Clock and architecture | Aliasing or missed events |
| Latency | Digital filter and conversion | Late alarms |

The ideal quantisation noise for an \(N\)-bit converter with a full-scale range \(V_{\text{FS}}\) is:

\[
q = \frac{V_{\text{FS}}}{2^N}, \qquad e_{\text{q,rms}} = \frac{q}{\sqrt{12}}
\]

The ideal SNR for a full-scale sine is:

\[
\text{SNR}_{\text{ideal}} = 6.02N + 1.76\ \text{dB}
\]

This is the benchmark, not the measurement. The measured SNR is lower because of thermal noise, clock jitter, reference noise, distortion, and the signal not being at full scale. The ENOB is:

\[
\text{ENOB} = \frac{\text{SINAD} - 1.76}{6.02}
\]

where SINAD is the signal power divided by all noise and distortion. A converter with 24 bits and a SINAD of 50 dB has an ENOB of 8.0 bits, regardless of its marketed resolution.

### 3.1 The four converter architectures

| Architecture | Principle | Resolution | Speed | Latency | Best for |
|---|---|---|---|---|---|
| Successive approximation (SAR) | Binary search with a digital-to-analogue converter (DAC) | 8–20 bits | 10 kSPS–10 MSPS | One conversion | Multi-channel, moderate resolution, low latency |
| Delta-sigma | Oversampling, noise shaping, decimation | 16–32 bits | 1 SPS–1 MSPS | Many samples, filter delay | Biopotentials, precision sensors, low bandwidth |
| Pipeline | Multi-stage quantisation | 8–16 bits | 1–500 MSPS | Several cycles | Imaging, high-speed acquisition |
| Flash | Parallel comparators | 4–8 bits | > 1 GSPS | One cycle | Embedded in other architectures; very high speed |

The architecture determines the latency and the anti-alias burden. A delta-sigma converter oversamples and filters digitally, so the analogue anti-alias filter can be simple, but the digital filter adds group delay. A SAR converter samples at or near the signal bandwidth, so the analogue anti-alias filter must be sharp, but the latency is low.

### 3.2 The resolution–rate–power triangle

For a given technology, resolution, rate, and power trade against each other. A 24-bit delta-sigma at 500 SPS may draw 1 mW per channel; a 16-bit SAR at 1 MSPS may draw 10 mW; a 12-bit pipeline at 100 MSPS may draw 500 mW. The trade is not linear: doubling the resolution by oversampling requires a 4× increase in samples for every extra bit if the noise is white, and the power rises accordingly.

The practical rule is to choose the lowest resolution and rate that meet the requirement with margin. A 24-bit converter used to measure a 1 mV signal with a 2 µV noise floor is wasted; the extra bits are below the noise. A 12-bit converter used to measure a 1 µV signal is inadequate; the LSB is larger than the signal.

### 3.3 Sampling and aliasing

Sampling at \(f_s\) maps every frequency \(f\) to a frequency in the interval \([0, f_s/2]\) according to:

\[
f_{\text{alias}} = \left| f - f_s \times \text{round}\left(\frac{f}{f_s}\right) \right|
\]

A 550 Hz interferer sampled at 500 SPS appears at 50 Hz:

\[
f_{\text{alias}} = |550 - 500| = 50\ \text{Hz}
\]

The 50 Hz alias is indistinguishable from a real 50 Hz signal. No digital filter can separate them, because they occupy the same frequency. The only defence is an analogue anti-alias filter before the sampler, or a higher sampling rate that places the interferer above Nyquist and filters it digitally.

The anti-alias filter's required attenuation depends on the largest out-of-band signal and the acceptable in-band alias. If a 1 V interferer must be attenuated below 1 µV, the filter must provide 120 dB at the interferer frequency. A first-order filter provides 20 dB per decade; a fourth-order filter provides 80 dB per decade. The filter order follows from the arithmetic, not from a preference.

### 3.4 Clock jitter

Sampling at a slightly different instant than intended produces a voltage error proportional to the signal's slope:

\[
\Delta v = \frac{dv}{dt} \Delta t = 2\pi f V_0 \cos(2\pi f t) \Delta t
\]

For a full-scale sine and RMS jitter \(t_j\), the jitter-limited SNR is:

\[
\text{SNR}_{\text{jitter}} = 20 \log_{10}\left(\frac{1}{2\pi f t_j}\right)
\]

The limit depends on the input frequency, not the sampling rate. At 150 Hz and 200 ps RMS jitter, the limit is 134.5 dB; at 10 kHz, 98.0 dB; at 1 MHz, 58.0 dB. A converter with 100 dB SNR at DC is not jitter-limited at 150 Hz, but it may be at 1 MHz if the clock is poor.

### 3.5 The reference is part of the converter

The reference sets the full-scale range and therefore the LSB. Its noise adds directly to the measurement; its drift changes the gain; its settling after a load change can corrupt a conversion. A 2.4 V reference with 20 µV RMS noise and a gain of 2 contributes 10 µV at the input, which is small compared with a 2 µV amplifier noise but not negligible. A reference with 200 µV RMS noise would dominate.

The reference's temperature coefficient is a gain error, not an offset error. A 10 ppm/°C reference over a 20 °C change is 200 ppm, or 0.02%, which is 0.2 µV on a 1 mV signal. The error is systematic and does not average away.

### 3.6 The sampling theorem in practice

The Nyquist-Shannon theorem says a signal band-limited to \(B\) can be reconstructed from samples at \(f_s > 2B\). The theorem is true and almost never sufficient.

- Real signals are not strictly band-limited; they have noise and interference at all frequencies.
- The reconstruction requires an ideal interpolator, which a real device does not have.
- The samples are quantised, not exact.
- The sampling instants are not ideal; they have jitter and finite aperture.

The practical rule is to sample at 2.5 to 5 times the highest frequency of interest, and to use an anti-alias filter that attenuates everything above Nyquist. The factor depends on the filter order and the interference environment, not on a theorem.

A second practical rule: the sampling rate must be chosen with the digital filter and the latency in mind. A high rate reduces the anti-alias burden but increases the data volume and, for a delta-sigma converter, reduces the OSR and the resolution.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| ADC | Analogue-to-digital converter | Used as a synonym for resolution | None |
| Sampling rate | Number of samples per second | Confused with bandwidth | SPS or Hz |
| Nyquist frequency | Half the sampling rate | Confused with the sampling rate | Hz |
| Aliasing | Folding of frequencies above Nyquist into the band | Confused with filtering | None |
| Anti-alias filter | Analogue filter before sampling | Placed after the converter | Hz, order |
| Quantisation | Mapping a continuous value to discrete codes | Confused with noise | Codes |
| LSB | Smallest code step | Confused with accuracy | V |
| Full-scale range | Input range mapped to all codes | Assumed to equal the supply | V |
| Resolution | Number of bits or code steps | Confused with ENOB | bits |
| ENOB | Effective number of bits from SINAD | Confused with resolution | bits |
| SNR | Signal power divided by noise power | Quoted without bandwidth | dB |
| SINAD | Signal power divided by noise and distortion | Confused with SNR | dB |
| THD | Total harmonic distortion | Ignored in ENOB | dB or % |
| SFDR | Spurious-free dynamic range | Confused with SNR | dB |
| INL | Integral nonlinearity; deviation from a straight line | Confused with DNL | LSB or ppm |
| DNL | Differential nonlinearity; deviation of a code width | Confused with INL | LSB |
| Missing code | A code that never appears | Confused with noise | Count |
| Offset error | Output at zero input | Confused with gain error | LSB or V |
| Gain error | Deviation of the slope from ideal | Confused with offset | % or ppm |
| Monotonicity | Output never decreases as input increases | Assumed from resolution | Pass or fail |
| Settling time | Time for the input to settle before sampling | Confused with conversion time | s |
| Acquisition time | Time the sample-and-hold spends acquiring | Confused with conversion time | s |
| Aperture jitter | Variation in the sampling instant | Ignored at low frequency | s RMS |
| Aperture delay | Nominal delay between the clock edge and the sample | Confused with jitter | s |
| Oversampling ratio (OSR) | Ratio of modulator rate to output rate | Confused with decimation | Ratio |
| Decimation | Reducing the sample rate after filtering | Confused with averaging | Ratio |
| Sinc filter | Digital low-pass used in delta-sigma decimation | Assumed to be brick-wall | Order |
| Group delay | Delay of the signal envelope through a filter | Confused with conversion time | s |
| Reference | Voltage that defines full scale | Ignored in noise budget | V |
| PSRR | Power-supply rejection ratio | Confused with CMRR | dB |
| Pipeline | Multi-stage ADC architecture | Confused with pipelining in software | None |
| SAR | Successive-approximation register ADC | Confused with delta-sigma | None |

## 5. The mechanics, step by step

The procedure below selects and verifies a converter and its conditioning. Each step states its input, output, and rationale. The edge cases required by the assignment are mapped to the steps and summarised in Section 5.13.

### Step 1. Define the signal band and the required resolution

**Input:** the clinical signal, its bandwidth, its amplitude, and the required accuracy.

**Output:** a band, a resolution requirement in bits or microvolts, and a latency requirement.

**Rationale:** the band determines the sampling rate and the anti-alias filter; the resolution determines the converter and the gain; the latency determines the architecture. A requirement that omits any of the three is incomplete.

Write the requirement as: band 0.5–150 Hz, amplitude 0.5–5 mV, resolution 1 µV, latency < 100 ms, and a 10 s storage window. A requirement of "24-bit ECG" is a component choice, not a requirement.

### Step 2. Choose the architecture

**Input:** the band, the resolution, the latency, and the power.

**Output:** an architecture and a justification.

**Rationale:** a delta-sigma converter is the natural choice for a low-bandwidth, high-resolution biopotential signal. A SAR converter is the natural choice for a multi-channel, moderate-resolution signal with low latency. A pipeline converter is the natural choice for imaging or high-speed acquisition.

**Branch point:** if the latency requirement is below about 1 ms and the resolution requirement is above 16 bits, no single converter may meet both; consider a faster SAR with averaging or a delta-sigma with a faster output rate and a shorter filter.

### Step 3. Choose the resolution and the OSR

**Input:** the noise budget and the converter's noise table.

**Output:** a gain, an output rate, and an OSR that meet the resolution with margin.

**Rationale:** the converter's noise falls with the OSR and rises with the gain's effect on full scale. A 24-bit delta-sigma at 500 SPS and a gain of 2 may have 2.13 µV RMS noise and a ±600 mV range; at a gain of 8, the noise is 1.29 µV RMS but the range is ±150 mV. The gain that maximizes the effective resolution depends on the signal amplitude, not on the marketed bit count.

The effective resolution in bits is not the resolution of the converter. It is the ratio of full scale to noise, expressed in bits:

\[
\text{Effective resolution} = \log_2\left(\frac{V_{\text{FS}}}{e_{\text{noise}}}\right)
\]

For a ±600 mV range and 2.13 µV RMS noise, the ratio is 563,000, or 19.1 bits. The converter has 24 bits; four bits are below the noise.

### Step 4. Choose and condition the reference

**Input:** the required full scale, the reference options, and the noise budget.

**Output:** a reference, a decoupling network, and a reference-noise contribution.

**Rationale:** the reference must be stable, quiet, and capable of driving the converter's switched-capacitor load. A reference with a high output impedance and a large decoupling capacitor can ring or settle slowly after a conversion. A reference with low noise but poor drift is worse than a reference with moderate noise and good drift for a measurement that is calibrated once.

The reference noise is divided by the gain when referred to the input. A 20 µV RMS reference with a gain of 2 contributes 10 µV at the input; with a gain of 100, it contributes 0.2 µV. The gain helps the reference noise, but it also reduces the input range.

### Step 5. Design the input drive and settling

**Input:** the source impedance, the sample-and-hold capacitance, and the acquisition time.

**Output:** a drive amplifier, a series resistor, and a settling calculation.

**Rationale:** the sample-and-hold capacitor must charge to the input voltage within the acquisition time. The settling is exponential with a time constant \(R_{\text{source}} C_{\text{S/H}}\). For a 10 pF capacitor and a 10 kΩ source, the time constant is 100 ns; to settle to 1 LSB of a 24-bit converter, the required time is \(N \ln 2 \times \tau = 16.6 \times 100\ \text{ns} = 1.66\ \text{µs}\). If the acquisition time is shorter, the converter will produce a gain error that depends on the source impedance.

A buffer at the converter input isolates the source and provides the charge. The buffer's own noise and offset enter the budget. A switched-capacitor input without a buffer can also inject charge back into the source, which is a problem for high-impedance sensors.

### Step 6. Design the analogue anti-alias filter

**Input:** the sampling rate, the largest out-of-band interferer, and the acceptable alias level.

**Output:** a filter order, a corner, and the attenuation at the critical frequencies.

**Rationale:** the filter must attenuate every frequency that would alias into the band. The critical frequencies are \(f_s \pm f_{\text{band}}\), \(2f_s \pm f_{\text{band}}\), and so on. A 550 Hz interferer at a 500 SPS rate aliases to 50 Hz; a 950 Hz interferer aliases to 50 Hz as well. The filter must attenuate both.

The required attenuation is:

\[
A = 20 \log_{10}\left(\frac{V_{\text{interferer}}}{V_{\text{alias,max}}}\right)
\]

For a 1 V interferer and a 1 µV acceptable alias, \(A = 120\ \text{dB}\). A first-order filter at 150 Hz provides 20 dB per decade; the attenuation at 550 Hz is only 11.3 dB. A fourth-order filter provides 45 dB; an eighth-order filter provides 90 dB. The filter order follows from the arithmetic.

### Step 7. Design the digital filter and compute its latency

**Input:** the output rate, the OSR, and the filter type.

**Output:** a filter response, a group delay, and a settling time.

**Rationale:** a delta-sigma converter's digital filter removes the modulator's high-frequency noise and decimates to the output rate. A sinc\(^3\) filter with an OSR of \(N\) has a group delay of about \(1.5 N / f_{\text{mod}}\), or \(1.5 / f_{\text{data}}\) in output-rate terms. At 500 SPS, the group delay is about 3 ms. A higher-order filter has a longer delay but a sharper transition.

The filter's stopband attenuation is not infinite. A sinc\(^3\) filter has nulls at multiples of the output rate, but the attenuation between nulls is finite. An interferer near a multiple of the output rate can alias into the band if the analogue filter does not attenuate it first. The digital filter is not a substitute for the analogue filter; it is a complement.

### Step 8. Choose the clock and compute the jitter limit

**Input:** the input frequency, the required SNR, and the clock source.

**Output:** a jitter requirement and a clock selection.

**Rationale:** the jitter-limited SNR is \(20\log_{10}(1/(2\pi f t_j))\). For a 150 Hz signal and a 100 dB requirement, the jitter must be below \(1/(2\pi \times 150 \times 10^5) = 10.6\ \text{ns}\). For a 1 kHz signal, it must be below 1.6 ns. A crystal oscillator with 1 ps RMS jitter is far better than needed; a microcontroller's internal oscillator with 100 ps jitter is also adequate at these frequencies. Jitter becomes a first-order limit at 100 kHz and above, where a 100 ps clock gives an 84 dB limit and a 1 ns clock gives a 64 dB limit.

The clock's phase noise is the frequency-domain view of the same jitter. A clock with high phase noise at 1 kHz offset produces jitter that affects a 1 kHz signal. The datasheet's jitter number may be integrated over a different bandwidth; state the bandwidth.

### Step 9. Calibrate offset and gain

**Input:** the converter, the reference, and a reference source.

**Output:** offset and gain coefficients, a calibration interval, and an uncertainty.

**Rationale:** offset and gain errors are systematic and can be calibrated. INL and DNL are not calibrated by a two-point procedure; they require a lookup table or a better converter. A 24-bit converter with 7.5 ppm INL has a 4.5 µV deviation at a ±600 mV range, which is larger than a 2 µV noise floor; the INL, not the noise, limits the accuracy at mid-scale.

The calibration must cover the temperature range and the supply range. A calibration at 25 °C and 3.0 V may not hold at 40 °C and 2.7 V. The coefficients should be stored with the device and verified at startup.

### Step 10. Check throughput and multiplexing

**Input:** the channel count, the output rate, and the converter's aggregate rate.

**Output:** a schedule and a settling budget for each channel.

**Rationale:** a multiplexed converter shares one sampler among several channels. Each channel's acquisition time is reduced by the multiplexing, and the charge from the previous channel can remain on the sample capacitor. A 16-channel multiplexer at 1 MSPS gives 62.5 kSPS per channel, but the settling between channels may be the limiting factor, not the aggregate rate.

A simultaneous-sampling converter avoids the multiplexing delay but requires one converter per channel. The choice depends on whether the channels must be aligned in time.

### Step 11. Verify with a histogram and an FFT

**Input:** a converter and a test signal.

**Output:** offset, gain, noise, SINAD, THD, INL, DNL, and missing codes.

**Rationale:** the histogram test with a slowly varying input measures the code widths and the missing codes. The FFT test with a coherent sine measures the noise, distortion, and spurious components. The two tests answer different questions: the histogram measures static linearity, the FFT measures dynamic performance.

The FFT test must use a coherent sampling frequency and a window that matches the signal. A non-coherent sine leaks energy across bins and makes the noise floor look higher than it is. A windowed FFT corrects the leakage but changes the noise bandwidth; state the window and the equivalent noise bandwidth.

### Step 12. Document the configuration

**Input:** the converter, the filter, the reference, and the clock.

**Output:** a configuration record with the register values, the timing, and the calibration.

**Rationale:** the same converter can produce different data with different register settings. The output rate, the gain, the filter type, the reference, and the clock source must be recorded. Without the record, a change in the data cannot be distinguished from a change in the patient.

The record should include the converter's part number and revision, the register map, the output rate, the OSR, the filter type, the gain, the reference voltage, the clock frequency and source, the calibration coefficients, and the date.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 3, 5 | Input exceeds full scale | Code histogram at extremes | Clipping; lost signal |
| Common-mode rejection failure | 5, 6 | Drive or reference mismatch | Inject common mode | Interference |
| Electrode offset | 3, 5 | Offset exceeds range | Input range check | Saturation |
| Clock jitter | 8 | Aperture variation | SNR versus input frequency | Degraded high-frequency SNR |
| Brown-out | 12 | Reference or supply drops | Supply monitor | Invalid codes; reset |
| Watchdog reset loop | 12 | Firmware fails to service watchdog | Reset counter | Data gaps |
| Thermal throttling | 10, 12 | Clock or processor slows | Temperature log | Reduced throughput |
| Firmware update failure | 12 | Registers reset or coefficients lost | Readback and checksum | Wrong gain or filter |
| Calibration drift | 9, 11 | Reference or gain drift | Periodic calibration check | Systematic bias |
| Aliasing | 6, 7 | Interferer above Nyquist folds | Inject a tone above Nyquist | Irreversible corruption |
| Missing codes | 11 | DNL worse than −1 LSB | Histogram test | Nonlinearity |
| Reference noise | 4 | Reference feeds full scale | Spectrum with quiet input | Noise and gain error |

### 5.14 Architecture selection table

| Requirement | SAR | Delta-sigma | Pipeline | Flash |
|---|---|---|---|---|
| 24-bit resolution at < 1 kSPS | No | Yes | No | No |
| 16-bit at 1 MSPS | Yes | Marginal | Yes | No |
| Latency < 10 µs | Yes | No | Yes | Yes |
| Lowest power at 1 kSPS | Yes | Yes | No | No |
| Multi-channel simultaneous | With multiple ADCs | With multiple modulators | With multiple ADCs | With multiple ADCs |
| Anti-alias simplicity | Hard | Easy | Hard | Hard |
| Intrinsic latency | Low | High | Medium | Low |

The table is a starting point. The final choice depends on the noise, the reference, the drive, and the power, not only on the architecture name.

### 5.15 Error budget for a converter channel

| Error term | Typical value | Referred to input | Calibrated? |
|---|---:|---:|---|
| Quantisation | LSB/√12 | 20.6 nV for 24-bit, ±600 mV | No |
| Converter thermal noise | 2.13 µV RMS | 2.13 µV | No |
| Reference noise | 20 µV RMS / gain 2 | 10 µV | No |
| INL | 7.5 ppm FSR | 4.5 µV | Lookup table |
| Offset | 1 mV | 1 mV | Yes |
| Gain error | 0.1% | 1 µV on 1 mV | Yes |
| Clock jitter | 200 ps at 150 Hz | 0.16 µV equivalent | No |
| Aliased interference | Depends on filter | Depends | No |

The table shows that the converter's quantisation is negligible; the thermal noise, the reference, and the INL dominate. A design that quotes the quantisation noise as the resolution is off by two orders of magnitude.

### 5.16 Verification tests

| Test | Input | Metric | Acceptance |
|---|---|---|---|
| Histogram | Slow ramp or sine | Offset, gain, INL, DNL, missing codes | INL < 7.5 ppm; no missing codes |
| FFT | Coherent sine | SNR, THD, SINAD, SFDR | SINAD > requirement |
| Noise | Shorted input | RMS noise in band | Below budget |
| Alias | Tone above Nyquist | Alias amplitude | Below 1 LSB |
| Settling | Step at input | Settling to 1 LSB | Within acquisition time |
| Reference | Load step | Reference deviation | Within budget |
| Jitter | High-frequency sine | SNR versus frequency | Meets requirement |
| Temperature | Sweep | Offset and gain drift | Within budget |

A verification that tests only the FFT at mid-scale misses the static linearity and the alias. A verification that tests only the histogram misses the dynamic performance. Both are needed.

### 5.17 Delta-sigma noise shaping

A delta-sigma modulator does not eliminate quantisation noise; it moves it. The modulator's feedback loop shapes the quantisation noise so that it is low in the signal band and high near the modulator's sampling frequency. The digital decimation filter then removes the out-of-band noise.

For a second-order modulator, the in-band noise falls by 15 dB per doubling of the OSR, or about 2.5 bits per octave. For a first-order modulator, the fall is 9 dB per octave, or 1.5 bits. The exact slope depends on the order and the modulator's stability.

| Modulator order | Noise fall per OSR doubling | Bits gained per octave | Stability |
|---|---:|---:|---|
| First | 9 dB | 1.5 | Unconditionally stable |
| Second | 15 dB | 2.5 | Stable for moderate inputs |
| Third | 21 dB | 3.5 | Requires care |
| Fourth | 27 dB | 4.5 | Conditional |

The trade is between resolution and stability. A higher-order modulator gives more resolution at the same OSR, but it can become unstable at high input amplitudes. The datasheet's maximum input range already accounts for this; exceeding it produces not clipping but oscillation and a sudden loss of noise shaping.

The OSR is the ratio of the modulator clock to the output rate. At a 4.096 MHz modulator clock and a 500 SPS output rate, the OSR is 8,192. Halving the output rate to 250 SPS doubles the OSR and improves the in-band noise by about 15 dB for a second-order modulator, at the cost of doubling the latency and halving the channel throughput.

### 5.18 SAR timing and settling

A SAR converter performs a binary search over \(N\) clock cycles. Each cycle compares the input against a DAC output and decides one bit. The conversion time is \(N\) clock periods plus the acquisition time. For a 16-bit SAR at 1 MSPS, the conversion is 16 cycles at 1 MHz, or 16 µs, which is longer than the 1 µs sample period; the actual throughput is lower than the clock rate suggests.

The acquisition time is the time the sample-and-hold capacitor has to charge. It must be long enough for the source to settle to the required accuracy:

\[
t_{	ext{acq}} \ge R_{	ext{source}} C_{	ext{S/H}} \ln(2^N)
\]

For a 10 kΩ source and a 10 pF capacitor, the 16-bit settling time is \(10^4 	imes 10^{-11} 	imes 11.09 = 1.11\ 	ext{µs}\). A 1 MSPS converter with a 1 µs period cannot allocate 1.11 µs to acquisition and 16 cycles to conversion; the maximum rate is lower, or the source must be buffered.

The SAR's input capacitance also injects charge back into the source at the end of the acquisition. A high-impedance source cannot absorb the charge, so the input voltage changes and the conversion is wrong. A buffer or a large capacitor at the input solves the problem, at the cost of settling time and noise.

### 5.19 Reference drive and decoupling

A reference is an amplifier with a finite output impedance and a finite loop bandwidth. The converter's switched-capacitor input draws a charge packet at each conversion, which produces a transient on the reference. The reference must recover before the next conversion, or the full-scale range changes.

The decoupling capacitor supplies the instantaneous charge, and the reference's loop restores the average. A 1 µF capacitor with a 1 Ω output impedance has a time constant of 1 µs; for a 1 MSPS converter, the reference may not recover between conversions. A 10 µF capacitor reduces the transient but increases the startup time and the settling after a load change.

The reference's noise and the decoupling capacitor's dielectric absorption both affect the measurement. A ceramic capacitor with a high dielectric constant (X5R or X7R) has voltage coefficient and ageing; a film or C0G capacitor is more stable but larger. The choice is part of the accuracy budget.

### 5.20 Digital filter types

| Filter | Passband | Stopband | Group delay | Best for |
|---|---|---|---|---|
| Sinc\(^1\) | Wide | Slow roll-off | 0.5 output periods | Fast settling |
| Sinc\(^2\) | Moderate | Moderate | 1.0 output periods | General |
| Sinc\(^3\) | Narrow | Good | 1.5 output periods | Precision |
| Sinc\(^4\) | Narrower | Better | 2.0 output periods | High resolution |
| FIR (designed) | Sharp | Sharp | (N−1)/(2f_s) | Custom response |
| IIR (designed) | Sharp | Sharp | Frequency-dependent | Efficient sharp filtering |

A sinc filter's frequency response has nulls at multiples of the output rate. An interferer exactly at a null is removed; an interferer between nulls is attenuated but not eliminated. The worst-case attenuation is at the midpoint between nulls, and it improves with the filter order. A sinc\(^3\) filter's worst-case stopband attenuation is about 60 dB; a sinc\(^4\) is about 80 dB. Neither is infinite, which is why the analogue filter still matters.

The filter's group delay is proportional to the number of output periods. At 500 SPS, a sinc\(^3\) filter delays the signal by about 3 ms. At 4 kSPS, the same filter delays by 0.375 ms. The delay is part of the latency budget; a faster output rate reduces it, at the cost of more noise.

### 5.21 What a converter datasheet does not tell you

A datasheet describes the converter in a test circuit. It does not describe:

- The noise of the reference you will use.
- The alias rejection of your anti-alias filter.
- The settling of your source.
- The jitter of your clock.
- The crosstalk between your channels.
- The behaviour of your firmware when the converter's data-ready signal is missed.
- The crosstalk between your channels at the frequencies of interest.
- The settling of your reference after a load step.

Each of these is a system property. The datasheet is necessary but not sufficient; the verification must test the system, not the evaluation board.

### 5.22 Histogram and FFT details

A histogram test uses a input that visits every code many times. A slow ramp is ideal but difficult to generate with low distortion; a sine is easier but spends more time near the extremes. The code widths are estimated from the counts, and the INL and DNL are computed from the widths. A code with zero counts is a missing code; a code with a width of zero but a neighbour with a double width is a DNL of −1 LSB.

An FFT test uses a coherent sine: the input frequency is an integer number of cycles in the record, so the energy falls exactly on one bin. A non-coherent sine leaks energy into adjacent bins and raises the apparent noise floor. A window function reduces the leakage but spreads the signal over several bins, which changes the noise bandwidth. The test report should state the record length, the window, the number of averages, and the input amplitude relative to full scale.

The FFT's noise floor is not the converter's noise floor. The FFT processing gain is \(10\log_{10}(M/2)\) for an \(M\)-point transform, so a 4,096-point FFT on a 12-bit converter shows a floor about 33 dB below the converter's SNR. A reader who compares the FFT floor with the datasheet SNR without accounting for the processing gain will conclude that the converter is much better than it is.

### 5.23 Missing codes and monotonicity

A converter is monotonic if its output never decreases as the input increases. Monotonicity is required for a control loop and for a measurement that must not reverse direction near a threshold. A DNL of −1 LSB means a code is missing; a DNL worse than −1 LSB means the transfer function is non-monotonic.

A 24-bit converter with a 7.5 ppm INL is not necessarily monotonic at the 24-bit level. The INL is a smooth deviation, but the DNL can still produce missing codes at the low end. The datasheet's "no missing codes" specification is usually stated for a specific resolution, not for all 24 bits. Read the condition.

A missing code is not a random error; it is a systematic nonlinearity. A model that uses the raw code as a feature may learn the code pattern rather than the signal. A calibration or a dither can reduce the effect, but neither restores the missing information.

### 5.25 Noise shaping versus averaging

Averaging \(N\) samples reduces white noise by \(\sqrt{N}\), or \(10\log_{10}N\) dB, if the noise is uncorrelated. Noise shaping reduces the in-band noise by moving it out of band, without requiring \(N\) identical measurements. The two are different tools:

| Method | Gain | Cost | Limitation |
|---|---|---|---|
| Averaging | 3 dB per doubling | Time or samples | Correlated noise does not average |
| Noise shaping | 9–27 dB per OSR doubling | Modulator complexity; latency | Stability; out-of-band noise |
| Increasing gain | Reduces input-referred converter noise | Input range | Saturation |
| Reducing bandwidth | Reduces integrated noise | Signal content | Clinical information loss |
| Better reference | Reduces reference noise | Cost, power | Reference is only one term |

A design that averages 16 samples gains 12 dB, or two bits, but only if the noise is white. A delta-sigma modulator gains more per octave, but its noise is not white and its latency is higher. The choice depends on whether the limiting noise is white or shaped.


### 5.24 Ten converter review questions

1. What is the signal band, and what is the Nyquist frequency at the chosen output rate?
2. What is the anti-alias filter order, and what attenuation does it provide at every alias frequency?
3. Does the source settle to the converter's resolution within the acquisition time?
4. What is the reference noise and drift, referred to the input?
5. What is the clock jitter, and at what input frequency does it limit the SNR?
6. What is the digital filter's group delay and settling time?
7. What are the INL and DNL, and do they support the accuracy claim?
8. How are offset and gain calibrated, and over what temperature and supply range?
9. Has an alias test been performed with a tone above Nyquist?
10. What configuration is recorded with each data stream?

A review that answers all ten has covered the converter as a system. A review that answers only the first and the seventh has reviewed the datasheet, not the design.


## 6. Worked example

### 6.1 The requirement

A 3-channel ECG monitor uses a 24-bit, 8-channel, simultaneous-sampling delta-sigma converter. The relevant datasheet values at an output rate of 500 SPS and a gain of 2 are:

- Input-referred noise: 2.13 µV RMS.
- Full-scale range: ±600 mV differential.
- Dynamic range: 106 dB.
- Effective resolution: 19.1 bits.
- INL: 7.5 ppm of full scale.
- Reference: 1.2 V internal.
- Modulator clock: 4.096 MHz.
- Aggregate data rate: up to 32 kSPS.

The front end from Topic 03 has a 2.13 µV RMS noise floor. The signal is a 1.0 mV peak-to-peak ECG. The clock has 200 ps RMS jitter. The analogue anti-alias filter is a single RC with a 1.59 kHz corner.

### 6.2 The resolution budget

The LSB at the converter input is:

\[
\text{LSB} = \frac{2 \times 1.2}{2 \times 2^{24}} = 71.5\ \text{nV}
\]

The converter noise of 2.13 µV RMS is:

\[
\frac{2.13\ \text{µV}}{71.5\ \text{nV}} = 29.8\ \text{LSB RMS}
\]

The effective resolution is:

\[
\log_2\left(\frac{1.2\ \text{V}}{2.13\ \text{µV}}\right) = 19.1\ \text{bits}
\]

which matches the datasheet. The converter has 24 bits; the noise consumes 4.9 bits.

The INL of 7.5 ppm of full scale is:

\[
7.5 \times 10^{-6} \times 1.2\ \text{V} = 9.0\ \text{µV}
\]

at the converter input, or 4.5 µV at the input with a gain of 2. The INL is 2.1 times the noise. The accuracy at mid-scale is limited by the INL, not by the noise. A 24-bit converter with a 7.5 ppm INL cannot deliver 24-bit accuracy, even with perfect noise.

The system noise is the root sum square of the front-end noise and the converter noise:

\[
E_{\text{sys}} = \sqrt{2.13^2 + 2.13^2} = 3.01\ \text{µV RMS}
\]

The SNR for a 1.0 mV peak-to-peak signal is:

\[
\text{SNR} = 20 \log_{10}\left(\frac{354\ \text{µV}}{3.01\ \text{µV}}\right) = 41.4\ \text{dB}
\]

The system ENOB is:

\[
\text{ENOB} = \frac{41.4 - 1.76}{6.02} = 6.6\ \text{bits}
\]

The converter's 19.1 effective bits become 6.6 system bits because the front end and the converter contribute equally and the signal is small relative to full scale. This is the central lesson: the converter's resolution is not the system's resolution.

### 6.3 The alias budget

The output rate is 500 SPS, so the Nyquist frequency is 250 Hz. The ECG band is 0.5–150 Hz. A 550 Hz interferer aliases to:

\[
|550 - 500| = 50\ \text{Hz}
\]

The RC filter with a 1.59 kHz corner attenuates 550 Hz by:

\[
A = 20 \log_{10}\left(\sqrt{1 + \left(\frac{550}{1590}\right)^2}\right) = 0.49\ \text{dB}
\]

A 1 V interferer at 550 Hz becomes 0.945 V at the converter input, and the alias at 50 Hz is 0.945 V, which is 945,000 times the 1 µV signal. The converter saturates. Even a 10 mV interferer becomes 9.45 mV, which is 9.45 times the ECG signal.

The digital filter does not help at 550 Hz, because the alias occurs at the sampler, before the digital filter. The digital filter can remove the 50 Hz alias only by removing the real 50 Hz signal as well. The analogue filter is the only defence, and a single RC is inadequate.

The required attenuation for a 1 V interferer to produce a 1 µV alias is 120 dB. A fourth-order Butterworth at 150 Hz provides about 45 dB at 550 Hz; an eighth-order filter provides about 90 dB. To reach 120 dB, the filter must be steeper or the sampling rate must be higher. Raising the output rate to 2 kSPS moves the Nyquist to 1 kHz and the 550 Hz interferer into the band, where it can be removed digitally, but the interferer is then a real in-band signal, not an alias.

### 6.4 The jitter budget

The jitter-limited SNR at 150 Hz with 200 ps RMS jitter is:

\[
\text{SNR}_{\text{jitter}} = 20 \log_{10}\left(\frac{1}{2\pi \times 150 \times 200 \times 10^{-12}}\right) = 74.5\ \text{dB}
\]

The jitter limit is well above the 41.4 dB system SNR, so jitter is not the limiting factor at 150 Hz. At 1 kHz, the limit is 118.0 dB; at 10 kHz, it is 98.0 dB. A monitor that claims a 100 kHz bandwidth with the same clock would have a 78.0 dB jitter limit, which is still above the 41.4 dB system SNR; at 1 MHz, the limit falls to 58.0 dB. Jitter becomes the limiting factor only when the input frequency is high or the resolution is very high.

The clock's jitter must be specified over the bandwidth that matters. A 200 ps RMS jitter integrated from 100 Hz to 1 MHz is not the same as 200 ps integrated from 1 kHz to 10 MHz. The datasheet's number is only useful with its integration bandwidth.

### 6.5 Diagnostic 1: the 50 Hz interference is an alias

The monitor shows a strong 50 Hz tone in the ECG. The team assumes it is mains interference and adds a 50 Hz notch filter. The tone remains, because it is not mains; it is a 550 Hz switching supply aliased to 50 Hz.

The diagnostic is to change the output rate from 500 SPS to 1 kSPS. If the tone moves to 450 Hz (the new alias of 550 Hz at 1 kSPS), the source is a 550 Hz interferer, not mains. If the tone stays at 50 Hz, the source is real 50 Hz. The test is cheap and decisive.

The design flaw is the single RC anti-alias filter. The fix is a higher-order filter, a higher sampling rate, or a shielded supply. The notch filter is the wrong fix because it removes the real 50 Hz signal and leaves the alias.

### 6.6 Diagnostic 2: the reference noise is mistaken for amplifier noise

The noise floor is 10 µV RMS instead of the expected 3 µV. The team replaces the front end, with no change. The reference is a 2.4 V source with 200 µV RMS noise, divided by the gain of 2, contributing 100 µV at the input; the converter's reference input is not the same as the front end's reference, and the two are not decoupled.

The diagnostic is to measure the noise with the input shorted and the reference replaced by a battery. If the noise falls, the reference is the source. A second test is to measure the reference's noise directly with an oscilloscope or a spectrum analyzer.

The design flaw is the assumption that the reference is quiet because it is a reference. A reference is an active circuit with its own noise, drift, and output impedance. It belongs in the noise budget, not in the "ideal" column.

### 6.7 Vary one assumption: the output rate falls to 250 SPS

Change only the output rate, from 500 to 250 SPS, to save power. The Nyquist frequency falls to 125 Hz. The ECG band extends to 150 Hz, so the 125–150 Hz content aliases into the 100–125 Hz band. The QRS high-frequency content is distorted, and the diagnostic bandwidth claim is no longer supported.

The alias of a 150 Hz component at 250 SPS is:

\[
|150 - 250| = 100\ \text{Hz}
\]

A real 100 Hz component and the aliased 150 Hz component are indistinguishable. The monitor's display may look plausible, because the QRS energy is concentrated below 100 Hz, but the high-frequency notches used by some algorithms are corrupted.

The fix is to keep the 500 SPS rate for the diagnostic channel and use 250 SPS for a separate rhythm channel, or to lower the anti-alias corner to 100 Hz and accept the loss of 100–150 Hz content. The choice is clinical: the diagnostic claim requires the band; the rhythm claim does not.

### 6.8 What the example demonstrates

The arithmetic closes: 71.5 nV LSB, 29.8 LSB RMS noise, 19.1 effective bits, 9.0 µV INL, 3.01 µV system noise, 41.4 dB SNR, 6.6 system ENOB, 0.49 dB alias attenuation, and 134.5 dB jitter limit at 150 Hz. The two diagnostics reveal different flaws: one is an alias mistaken for mains, the other is a reference mistaken for an ideal source. The sensitivity analysis shows that halving the output rate breaks the diagnostic bandwidth claim.

The general method is to write the budget, identify the dominant term, test the assumption behind it, and state the consequence. For a converter, the dominant term is often the reference or the aliasing, not the quantisation.

### 6.11 A second alias: the 950 Hz interferer

The 550 Hz interferer is not the only one. A 950 Hz interferer sampled at 500 SPS aliases to:

\[
|950 - 2 	imes 500| = 50\ 	ext{Hz}
\]

The RC filter attenuates 950 Hz by 1.32 dB, so 85.9% of a 1 V interferer reaches the converter. Two different interferers produce the same 50 Hz alias, which is why a single notch at 50 Hz cannot distinguish them.

The general rule is that every frequency \(k f_s \pm f_{	ext{alias}}\) maps to the same alias. The anti-alias filter must attenuate all of them, not only the first. A filter design that considers only \(f_s - f_{	ext{band}}\) is incomplete.

| Residual risk | Likelihood | Consequence | Control |
|---|---|---|---|
| Alias from a 550 Hz source | High with the single RC | False 50 Hz tone | Higher-order filter; alias test |
| Reference noise above budget | Medium | Reduced SNR | Low-noise reference; decoupling |
| Clock jitter at high input frequency | Low at 150 Hz | SNR loss at 1 kHz | Low-jitter oscillator |
| INL above the accuracy claim | Medium | Nonlinearity | Lookup calibration; better converter |

### 6.9 Power and throughput

The converter draws 6 mW for eight channels at 3 V. The three active channels use 2.25 mW; the other five are powered but idle. The AFE draws 0.51 mW, the microcontroller 5 mW, and the radio 3 mW, for a total of 10.76 mW. The daily energy is 930 J, and a 200 mAh battery at 3.7 V (2,664 J) gives a 2.86-day runtime.

The aggregate data rate is 500 SPS × 3 channels = 1,500 samples per second. At 3 bytes per sample, that is 4.5 kB/s, or 389 MB per day before compression. A 512 MB flash device is insufficient for a 24-hour recording with metadata; the design needs compression, a larger memory, or an event-triggered storage policy.

The throughput is well below the converter's 32 kSPS aggregate limit. The limiting factor is the power and the storage, not the converter. A design that adds channels without recalculating the storage will fail at the first long recording.

### 6.10 Verification results

| Test | Measured | Acceptance | Margin |
|---|---:|---:|---:|
| Noise, input shorted | 3.0 µV RMS | < 3.5 µV | 0.5 µV |
| SNR, 1 mVpp, 40 Hz | 41.5 dB | > 40 dB | 1.5 dB |
| INL | 8.1 ppm | < 10 ppm | 1.9 ppm |
| Alias, 550 Hz, 1 V | 0.945 V | < 1 µV | Fails |
| Jitter limit at 150 Hz | 134.5 dB | > 50 dB | 84.5 dB |
| Settling, 300 mV step | 1.5 s | < 2 s | 0.5 s |
| Power | 10.8 mW | < 12 mW | 1.2 mW |

The alias test fails, as Section 6.3 predicted. The design cannot ship with the single RC filter. The other tests pass with modest margin. The verification table is more useful than a pass/fail summary because it shows which margin is thin and which test failed.
## 7. Code, query, or calculation

The script reproduces the resolution, alias, jitter, filter-delay, and power results in Section 6. The full-scale span is computed as \(2V_{\text{REF}}/G\), which is the differential span for a converter with a ±\(V_{\text{REF}}/G\) range.

**Assumptions.** A 1.2 V reference, a gain of 2, a 24-bit converter, a 2.13 µV RMS noise floor, a 1 mV peak-to-peak signal, a 550 Hz interferer sampled at 500 SPS, a single RC anti-alias filter with a 1.59 kHz corner, 200 ps RMS clock jitter, and a 200 mAh battery.

```python
import math

vref = 1.2
bits = 24
gain = 2.0
noise = 2.13e-6
signal_rms = 353.6e-6
inl_ppm = 7.5

span = 2 * vref / gain
lsb = span / (2**bits)
ideal_snr = 6.02 * bits + 1.76
eff_res = math.log2(span / noise)
system_noise = math.sqrt(noise**2 + noise**2)
snr = 20 * math.log10(signal_rms / system_noise)
enob = (snr - 1.76) / 6.02
inl = inl_ppm * 1e-6 * span

print("RESOLUTION")
print(f"  span                   {span:8.2f} V")
print(f"  LSB                    {lsb*1e9:8.2f} nV")
print(f"  ideal SNR              {ideal_snr:8.2f} dB")
print(f"  effective resolution   {eff_res:8.2f} bits")
print(f"  system noise           {system_noise*1e6:8.2f} uV RMS")
print(f"  system SNR             {snr:8.2f} dB")
print(f"  system ENOB            {enob:8.2f} bits")
print(f"  INL                    {inl*1e6:8.2f} uV")
print()
print("ALIAS")
fs = 500.0
f = 550.0
fc = 1 / (2 * math.pi * 1000 * 100e-9)
att = 20 * math.log10(math.sqrt(1 + (f / fc)**2))
print(f"  alias of 550 Hz        {abs(f-fs):8.2f} Hz")
print(f"  RC corner              {fc:8.2f} Hz")
print(f"  attenuation at 550 Hz  {att:8.2f} dB")
print(f"  transmitted amplitude  {10**(-att/20):8.4f} V per V")
print()
print("JITTER (200 ps RMS)")
for fq in (150, 1000, 10000, 100000, 1000000):
    print(f"  {fq:8.0f} Hz: {20*math.log10(1/(2*math.pi*fq*200e-12)):8.2f} dB")
print()
print("DIGITAL FILTER")
for order in (1, 2, 3, 4):
    ratio = 1.5 if order == 3 else 0.5 * order
    print(f"  sinc{order}: group delay {ratio/fs*1e3:6.2f} ms at 500 SPS")
print()
print("POWER")
p_adc = 6.0 * 3 / 8
p_total = p_adc + 0.51 + 5.0 + 3.0
j_day = p_total * 1e-3 * 86400
battery = 0.200 * 3.7 * 3600
print(f"  ADC active channels    {p_adc:8.3f} mW")
print(f"  total                  {p_total:8.3f} mW")
print(f"  energy per day         {j_day:8.1f} J")
print(f"  runtime (200 mAh)      {battery/j_day:8.2f} days")
```

**Expected output.**

```text
RESOLUTION
  span                       1.20 V
  LSB                       71.53 nV
  ideal SNR                146.24 dB
  effective resolution      19.10 bits
  system noise               3.01 uV RMS
  system SNR                41.39 dB
  system ENOB                6.58 bits
  INL                        9.00 uV

ALIAS
  alias of 550 Hz           50.00 Hz
  RC corner               1591.55 Hz
  attenuation at 550 Hz      0.49 dB
  transmitted amplitude    0.9452 V per V

JITTER (200 ps RMS)
       150 Hz:   134.49 dB
      1000 Hz:   118.02 dB
     10000 Hz:    98.02 dB
    100000 Hz:    78.02 dB
   1000000 Hz:    58.02 dB

DIGITAL FILTER
  sinc1: group delay   1.00 ms at 500 SPS
  sinc2: group delay   2.00 ms at 500 SPS
  sinc3: group delay   3.00 ms at 500 SPS
  sinc4: group delay   4.00 ms at 500 SPS

POWER
  ADC active channels       2.250 mW
  total                    10.760 mW
  energy per day            929.7 J
  runtime (200 mAh)          2.87 days
```

The output makes three points.

First, the ideal SNR of a 24-bit converter is 146.2 dB, but the system SNR is 41.4 dB. The difference is 104.8 dB, or 17.4 bits. The converter's resolution is not the measurement's resolution; the front end, the reference, and the signal amplitude account for almost all of the loss.

Second, the single RC filter attenuates the 550 Hz interferer by 0.49 dB, so 94.5% of it reaches the converter. The alias at 50 Hz is indistinguishable from a real 50 Hz signal. A digital notch cannot fix it, because the notch would also remove the real signal.

Third, jitter is not a problem at 150 Hz with a 200 ps clock: the limit is 134.5 dB, far above the 41.4 dB system SNR. At 1 MHz, the limit is 58.0 dB, still above the system SNR but close enough to matter for a higher-resolution design. Jitter is a high-frequency problem, not a low-frequency one.

**A note on the 24-bit claim.** The converter's effective resolution of 19.1 bits is a property of the converter at a gain of 2 and a 500 SPS rate. At a gain of 8, the noise falls to 1.29 µV but the span falls to 300 mV, and the effective resolution becomes \(\log_2(0.3/1.29\times10^{-6}) = 17.8\) bits. The effective resolution falls with gain, even though the noise falls, because the range shrinks faster. The datasheet's table shows the same pattern.

## 8. Data, standards, and regulatory context

The converter is constrained by the performance standards of the device and by the general standards for safety and software. The table separates the instruments and the clauses that matter.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| IEC 60601-2-25:2011 | Electrocardiographs | Particular standard | Diagnostic bandwidth, input range, offset tolerance, accuracy of derived values |
| IEC 60601-2-27:2011 | ECG monitoring equipment | Particular standard | Dynamic range, heart-rate accuracy, alarm performance |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General medical electrical equipment | Consensus standard; binding when referenced | Isolation, leakage, applied parts, protection |
| IEC 60601-1-2:2014+AMD1:2020 | Electromagnetic disturbances | Collateral standard | Immunity of the converter and its cables |
| IEC 60601-1-8:2006+AMD2:2020 | Alarm systems | Collateral standard | Alarm condition delay; signal generation delay |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis for aliasing, saturation, and data loss |
| IEC 62304:2006+AMD1:2015 | Software lifecycle | Consensus standard | Firmware that configures and reads the converter |
| IEEE 1057-2017 | Waveform digitizer testing | Consensus standard | Terminology and test methods for ADCs |
| IEEE 1241-2010 | ADC terminology and test methods | Consensus standard | INL, DNL, SINAD, ENOB definitions |

### 8.1 What the device standards require of the converter

IEC 60601-2-25:2011 requires the electrocardiograph to record a ±5 mV input while tolerating ±300 mV of differential and common-mode offset, with a frequency response from 0.05 Hz to at least 150 Hz. The converter must therefore sample at a rate that preserves 150 Hz and must have a range that accommodates the offset after the front end removes it. The standard does not prescribe the converter architecture, the resolution, or the sampling rate; it prescribes the input and output behaviour.

IEC 60601-2-27:2011 covers ECG monitoring equipment, which has different requirements from diagnostic electrocardiographs. The monitoring bandwidth is narrower, the heart-rate accuracy is specified, and the alarm performance is tested. A monitor that uses a 250 SPS converter and a 40 Hz filter can meet the monitoring standard but not the diagnostic standard. The distinction matters for the data: a monitor's data cannot be used to support a diagnostic claim.

The general standard IEC 60601-1 constrains the isolation and leakage, which affect the converter's reference and ground. A converter that is referenced to the patient side of an isolation barrier has a different noise environment from one referenced to the mains side. The isolation barrier's capacitance can couple switching noise into the measurement.

### 8.2 What IEEE 1057 and IEEE 1241 standardise

IEEE 1057-2017 defines the terminology and test methods for waveform digitizers, including the sine-fit test, the histogram test, and the FFT test. IEEE 1241-2010 defines the ADC terminology, including INL, DNL, SINAD, SNR, THD, and ENOB. The standards are not regulatory requirements, but they are the common language for specifying and verifying converter performance.

Using the standard terminology matters for procurement and for cross-team communication. A vendor's "ENOB" may be computed from SINAD at a different frequency, amplitude, or bandwidth than the customer's. The standard defines the measurement conditions; the datasheet should state them.

### 8.3 What the standards do not cover

The standards do not specify:

- The anti-alias filter order or corner.
- The reference noise or drift.
- The clock jitter.
- The digital filter's latency.
- The calibration procedure.
- The handling of missing codes or saturation.

Those are design decisions that the manufacturer must justify through risk analysis and verification. A device can meet every clause and still produce data that are unsuitable for a model, because the clauses test the device's output, not the model's input.

### 8.6 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 60601-2-25 | Sample rate, bandwidth, gain, filter | Reproduce the diagnostic claim |
| IEC 60601-2-27 | Rate, alarm delay, heart-rate algorithm version | Reproduce the monitoring claim |
| IEC 60601-1 | Isolation configuration, reference ground | Explain noise and safety |
| IEC 60601-1-2 | Immunity levels, cable configuration | Interpret interference |
| ISO 14971 | Hazard analysis for alias and saturation | Link failure to harm |
| IEC 62304 | Firmware version, register map | Detect configuration change |
| IEEE 1057/1241 | Test conditions, window, record length | Reproduce the performance number |

The record is the same information the data team needs for provenance. A team that records the rate, gain, filter, and firmware for its own analysis is also assembling regulatory evidence.

### 8.4 The data implications

The converter's configuration is part of the measurement. The output rate, the gain, the filter type, the reference, and the clock source must be recorded with the data. A change in any of them changes the noise, the bandwidth, and the latency, and can invalidate a model trained on the previous configuration.

A practical rule: if the data dictionary does not contain the converter's output rate, gain, and filter setting, the dataset cannot be reproduced. The rule is simple, and it is violated often.

### 8.5 The regulatory pathway

A converter is a component, not a device. The device's regulatory pathway determines the evidence required for the converter's performance. In the United States, a class II device typically uses the 510(k) pathway and a predicate device; in the European Union, a class IIa or IIb device uses the Medical Device Regulation and a notified body. The converter's datasheet is supporting evidence, not a substitute for the device-level verification.

## 9. Industry practice and case evidence

### 9.1 Delta-sigma converters dominate biopotential front ends

Integrated biopotential AFEs almost always use delta-sigma converters. A representative part, the Texas Instruments ADS131M08, integrates eight simultaneous-sampling 24-bit delta-sigma converters, a programmable gain amplifier, and a reference. Its datasheet (SBAS950B) reports, at 500 SPS and a gain of 1, an input-referred noise of 2.39 µV RMS, a dynamic range of 111 dB, and an effective resolution of 19.9 bits. At a gain of 8, the noise falls to 1.29 µV RMS and the effective resolution to 17.8 bits, because the full-scale range shrinks faster than the noise.

The integration changes the data scientist's problem. The output rate, gain, and filter are register settings, not visible components. The same silicon can produce different data in two products. The configuration must be recorded with the data, or the provenance is incomplete.

**Documented fact:** the datasheet values are from the manufacturer's SBAS950B. **Inference:** the prevalence of delta-sigma in biopotential AFEs is inferred from the product landscape, not from a census.

### 9.2 SAR converters remain important for multi-channel and low-latency

A SAR converter is preferred when the signal bandwidth is moderate, the channel count is high, and the latency must be low. A 16-bit SAR at 1 MSPS can digitise eight channels at 125 kSPS each with a latency of a few microseconds, which is impossible with a delta-sigma converter at the same resolution. The trade is the anti-alias filter: a SAR samples at or near the signal bandwidth, so the analogue filter must be sharp.

The choice is not ideological. A temperature channel with a 1 Hz bandwidth does not need a SAR; a delta-sigma at 10 SPS is cheaper and quieter. A motor-current channel with a 100 kHz bandwidth cannot use a delta-sigma at a reasonable power; a SAR or a pipeline is the right choice.

### 9.3 The "24-bit" claim in marketing

A 24-bit converter is often described as providing 24-bit accuracy. The datasheet's noise table shows that the effective resolution is lower, and the system's ENOB is lower still. The gap is not a defect; it is the difference between resolution and accuracy. A team that quotes 24 bits in a clinical claim without the noise and INL evidence is making a claim that the data do not support.

The practical response is to report the system ENOB at the specified signal amplitude and bandwidth, and to state the noise floor and the INL separately. A single "resolution" number hides the trade-offs that matter.

### 9.4 Aliasing in real devices

Aliasing is a common cause of unexplained tones in device data. A switching supply, a display backlight, or a nearby radio can produce an interferer above Nyquist that appears as a low-frequency tone. The tone is often attributed to the patient or to mains interference, and the wrong fix (a notch filter) is applied.

The diagnostic is to change the sampling rate and observe whether the tone moves. If it moves according to the alias equation, the source is above Nyquist. The fix is an analogue filter or a higher sampling rate, not a digital notch.

### 9.5 What good converter practice looks like

Teams that avoid converter surprises tend to:

- State the system ENOB, not the converter's resolution.
- Compute the alias attenuation at every critical frequency.
- Include the reference and the clock in the noise budget.
- Test with a tone above Nyquist.
- Record the output rate, gain, filter, and reference.
- Re-verify after any configuration change.
- Separate the diagnostic and monitoring bandwidths.

The practices are ordinary. The failures they prevent are not.

### 9.6 A note on evaluation boards and app notes

A converter evaluation board is a test vehicle, not a product. Its layout, reference, clock, and power supply are optimised for measurement, not for the product's size, cost, or safety. A performance number measured on the board is an upper bound, not a guarantee.

Application notes are similarly conditional. They describe a circuit that worked for the vendor's test conditions. The source impedance, the cable, and the interference environment in the product may be different. The note is a starting point, not evidence that the product meets its specification.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Aliasing | Tone that moves when the sample rate changes | Interferer above Nyquist folds into band | False low-frequency signal | Analogue filter; higher rate; alias test |
| Insufficient anti-alias filter | Tone amplitude follows the filter slope | Single RC; corner too high | Irreversible corruption | Higher-order filter; verify |
| Reference noise | Noise floor higher than the converter specification | Reference feeds full scale | Reduced SNR | Low-noise reference; decoupling |
| Reference drift | Gain changes with temperature | Reference temperature coefficient | Systematic bias | Calibrated reference; temperature compensation |
| Clock jitter | SNR falls at high input frequency | Aperture uncertainty | Degraded high-frequency content | Low-jitter clock; lower bandwidth |
| Missing codes | Histogram gaps | DNL worse than −1 LSB | Nonlinearity | Better converter; dither; calibration |
| INL above claim | Deviation at mid-scale | Process variation | Accuracy limit | Lookup calibration; specification |
| Settling error | Gain depends on source impedance | Sample capacitor not settled | Gain and offset error | Buffer; longer acquisition |
| Multiplexer crosstalk | Signal from one channel appears in another | Charge sharing; insufficient settling | Misattributed signal | Simultaneous sampling; buffer |
| Saturation | Codes at full scale | Input exceeds range | Clipping | Headroom; saturation flag |
| Data-ready missed | Periodic gaps | Firmware timing | Lost samples | Interrupt priority; FIFO |
| Brown-out | Reset during conversion | Supply drop | Invalid data | Supply monitor; reset flag |
| Configuration reset | Gain or rate changes after update | Registers reset to default | Data mismatch | Readback; checksum; record |
| Thermal drift | INL or offset changes after warm-up | Temperature coefficient | Bias | Temperature compensation; soak test |

The most dangerous converter failure is the one that produces a clean-looking but wrong signal. An alias at 50 Hz looks like mains; a missing code looks like noise; a reference drift looks like a physiological trend. The red flag is a change in the relationship between the converter output and an independent reference, not necessarily an error flag.

### 10.2 What to do when the alias test fails

1. Confirm the alias frequency by changing the output rate; if the tone moves according to the alias equation, the diagnosis is confirmed.
2. Identify the interferer with a spectrum analyzer or by shielding candidates one at a time.
3. Compute the required attenuation at every alias frequency, not only the first.
4. Choose a filter order and corner that meet the requirement with margin.
5. Re-test with the filter in place and with the interferer at its worst-case amplitude.
6. Record the test conditions and the result.

A failed alias test is not a reason to add a digital notch. It is a reason to fix the analogue filter or the sampling rate.
### 10.1 Ten red flags in converter data

1. A tone whose frequency changes when the output rate changes.
2. A noise floor that is lower than the converter's specified noise.
3. A histogram with gaps or with spikes at specific codes.
4. A gain that changes after a firmware update.
5. A reference voltage that differs from the configured value.
6. A sample rate that drifts with temperature.
7. A channel that is silent when its neighbour is driven.
8. A data-ready signal that is missed at high channel counts.
9. A signal that clips at a code other than full scale.
10. A calibration coefficient that does not match the device serial number.

Each is a prompt to test, not a conclusion. The tests are usually cheap.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The signal is band-limited to below Nyquist | Sampling is valid | Interferers above Nyquist | Inject a tone above Nyquist |
| The anti-alias filter attenuation is adequate | Alias is below the budget | Component tolerance, layout | Sweep the filter; measure |
| The reference is quiet and stable | Full scale is accurate | Reference noise, drift, loading | Measure with a quiet input; temperature sweep |
| The clock jitter is low enough | SNR is not jitter-limited | Poor clock, long traces | Measure SNR versus input frequency |
| The source settles in the acquisition time | Conversion is accurate | High source impedance, short acquisition | Vary source impedance; measure gain |
| The converter noise is stationary | RMS budget is valid | Flicker noise, interference | Measure over time and temperature |
| The INL is within the accuracy claim | Mid-scale accuracy | Process variation | Histogram test |
| The calibration covers the operating range | Offset and gain are correct | Temperature, supply, ageing | Calibration verification |
| The digital filter latency is acceptable | Alarms are timely | Filter order, output rate | Measure step response |
| The configuration is recorded | Provenance | Firmware update, service | Readback; checksum |
| The channel timing is aligned | Multi-channel analysis is valid | Multiplexing, simultaneous sampling | Inject a common signal |
| The data-ready signal is serviced | No samples are lost | Interrupt latency, FIFO overflow | Count samples; check sequence |

The dominant assumption is that the anti-alias filter is adequate. It is easy to verify with a tone above Nyquist, and it is often omitted because the converter's digital filter is assumed to do the job. The digital filter cannot remove an alias that has already occurred at the sampler.

A second limit is that the converter model is static. Real converters have settling, crosstalk, and memory effects that depend on the previous samples. A multiplexed converter's channel-to-channel crosstalk is not captured by a single-channel noise budget. The verification must test the multi-channel case, not only the single-channel case.

A third limit is that the ENOB is a single number for a single test condition. It depends on the input amplitude, the frequency, the reference, and the clock. An ENOB measured at full scale and 40 Hz is not the ENOB for a 1 mV signal at 150 Hz. State the conditions with the number.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no test, only a datasheet?
- Which assumption was true on the bench but is false in the home?
- Which assumption depends on the filter, the reference, or the clock?
- Which assumption is encoded in a register that can be reset?
- Which assumption is specific to the validation population?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A design review that cannot answer the last two has risks without controls.

## 12. Exercises

### Exercise 1. Compute ideal resolution (recall, 20 minutes)

For a 16-bit converter with a 2.5 V reference and a gain of 10, compute the differential span, the LSB at the converter input, the LSB at the source, the ideal SNR, and the ENOB. Repeat for a 24-bit converter with the same reference and gain.

**Deliverable:** a table with both converters.

**Acceptance criteria:** correct span and LSB; ideal SNR from 6.02N + 1.76; units on every value.

**Time:** 20 minutes.

### Exercise 2. Compute an alias (application, 30 minutes)

A 50 Hz signal is sampled at 120 SPS. A 70 Hz interferer is present. Compute the alias frequency and explain whether a digital low-pass filter at 50 Hz can remove it without removing the signal.

**Deliverable:** a calculation and a two-sentence explanation.

**Acceptance criteria:** alias equation correct; the explanation distinguishes filtering after sampling from filtering before.

**Time:** 30 minutes.

### Exercise 3. Design an anti-alias filter (application, 40 minutes)

A 1 V, 550 Hz interferer must be attenuated to below 1 µV at a 500 SPS sampling rate. Compute the required attenuation, choose a filter order and corner, and estimate the attenuation at 550 Hz. State the group delay.

**Deliverable:** a filter specification and a calculation.

**Acceptance criteria:** required attenuation correct; filter order justified; group delay estimated.

**Time:** 40 minutes.

### Exercise 4. Compute the jitter limit (application, 25 minutes)

A 16-bit converter must achieve an SNR of 90 dB at a 10 kHz input. What RMS jitter is required? Repeat for a 1 kHz input. Which case is harder, and why?

**Deliverable:** two calculations and a one-sentence answer.

**Acceptance criteria:** jitter formula correct; units in seconds; the frequency dependence explained.

**Time:** 25 minutes.

### Exercise 5. Diagnose a missing code (critique, 30 minutes)

A histogram of a 16-bit converter shows a missing code at mid-scale and a neighbouring code with twice the expected width. Explain the DNL, the likely cause, and two mitigations.

**Deliverable:** a diagnosis and two mitigations.

**Acceptance criteria:** DNL calculated; cause plausible; mitigations distinguish calibration from converter replacement.

**Time:** 30 minutes.

### Exercise 6. Write a verification plan (design, 40 minutes)

Write a verification plan for the converter in Section 6. Include histogram, FFT, noise, alias, settling, reference, jitter, and temperature tests. State the input, the metric, and the acceptance criterion for each.

**Deliverable:** a test matrix with at least 10 rows.

**Acceptance criteria:** covers static and dynamic tests; includes an alias test above Nyquist; each row has a quantitative criterion.

**Time:** 40 minutes.

### Exercise 7. Analyse a latency budget (application, 30 minutes)

A delta-sigma converter at 500 SPS uses a sinc\(^3\) filter with a 3 ms group delay. The algorithm uses a 2 s window, the network adds 100 ms, and the display adds 33 ms. Compute the total latency and identify the dominant term. Propose one change that reduces the total by at least 50%.

**Deliverable:** a table and a recommendation.

**Acceptance criteria:** all stages included; dominant term identified; the proposed change is feasible and its effect stated.

**Time:** 30 minutes.

### Exercise 8. Design a configuration record (design, 30 minutes)

Design a record for a converter channel that would allow a later analyst to reproduce the measurement. Include at least 15 fields and the diagnostic use of each.

**Deliverable:** a table with field, type, example, and use.

**Acceptance criteria:** covers part number, revision, output rate, OSR, gain, filter, reference, clock, calibration, and firmware; each field has a stated use.

**Time:** 30 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Resolution arithmetic | Two-converter table |
| 2 | Alias calculation | Frequency and explanation |
| 3 | Filter design | Specification and delay |
| 4 | Jitter limit | Two calculations |
| 5 | Static linearity | Diagnosis and mitigations |
| 6 | Verification | Test matrix |
| 7 | Latency | Budget and recommendation |
| 8 | Provenance | Configuration table |

## 13. Self-check answers

**Answer 1 (Exercise 1).** For 16 bits, a 2.5 V reference, and a gain of 10, the differential span is 2 × 2.5/10 = 0.5 V. The LSB at the converter is 0.5/65,536 = 7.63 µV; at the source, 0.763 µV. The ideal SNR is 6.02 × 16 + 1.76 = 98.1 dB; ENOB = 16 bits by definition. For 24 bits, the span is the same 0.5 V, the LSB is 0.5/16,777,216 = 29.8 nV at the converter and 2.98 nV at the source, and the ideal SNR is 146.2 dB. The 24-bit converter's LSB is below the noise of most front ends, so the extra bits are useful only for averaging.

**Answer 2 (Exercise 2).** The alias of 70 Hz at 120 SPS is |70 − 120| = 50 Hz. A digital low-pass at 50 Hz cannot remove the alias without removing the real 50 Hz signal, because the two are at the same frequency after sampling. The only defence is an analogue anti-alias filter before the sampler, or a higher sampling rate.

**Answer 3 (Exercise 3).** Required attenuation = 20 log10(1 V / 1 µV) = 120 dB. A fourth-order Butterworth at 150 Hz gives 80 dB/decade; at 550 Hz, the ratio is 3.67, and the attenuation is about 45 dB. An eighth-order filter gives about 90 dB. To reach 120 dB, use a higher order, a lower corner, or a higher sampling rate. The group delay of a fourth-order Butterworth at 150 Hz is about 2.6 ms; an eighth-order filter has roughly twice the delay.

**Answer 4 (Exercise 4).** For 90 dB at 10 kHz: \(t_j = 1/(2\pi \times 10^4 \times 10^{90/20}) = 1.59\ \text{ns}\). For 1 kHz: 15.9 ns. The 10 kHz case is harder because the signal slope is ten times larger for the same jitter. Jitter is a high-frequency problem.

**Answer 5 (Exercise 5).** A missing code means a code width of zero, which is a DNL of −1 LSB. A neighbouring code with twice the width has a DNL of +1 LSB. The likely cause is a converter with a DNL specification worse than ±1 LSB, or a reference or drive problem. Mitigations: use a converter with a guaranteed DNL better than ±1 LSB, or add dither and average, which spreads the code transitions. A two-point calibration does not fix DNL.

**Answer 6 (Exercise 6).** The matrix: histogram with a slow ramp for offset, gain, INL, DNL, and missing codes; FFT with a coherent sine at 40 Hz for SNR, THD, and SINAD; noise with a shorted input; alias with a 550 Hz tone at 1 V; settling with a 300 mV step; reference with a load step; jitter with a high-frequency sine; temperature at 15, 25, and 40 °C. Each row states the input, metric, and criterion; the alias row is expected to fail with the single RC.

**Answer 7 (Exercise 7).** Total = 3 ms + 2,000 ms + 100 ms + 33 ms = 2,136 ms. The algorithm window is 93.6% of the total. Reducing the window to 1 s cuts 1,000 ms, or 46.8%, which is just below the 50% target; reducing it to 0.9 s cuts 1,100 ms, or 51.5%, but reduces the number of beats available for rate estimation. The converter and network are not the problem.

**Answer 8 (Exercise 8).** Fields: converter part number; revision; channel count; output rate; OSR; gain; filter type and order; reference voltage and source; clock frequency and source; clock jitter; calibration coefficients and date; offset and gain trim; firmware version; register map checksum; board revision. Each field supports reproduction and change detection.

**Rubric for the design exercises.** A strong design states the band, the rate, the filter, the reference, the clock, and the verification, with numbers. A weak design names a converter and assumes the rest.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 1: 16-bit LSB at converter | 7.63 µV | ±0.05 µV |
| Exercise 1: 24-bit LSB at converter | 29.8 nV | ±0.5 nV |
| Exercise 2: alias of 70 Hz at 120 SPS | 50 Hz | exact |
| Exercise 3: required attenuation | 120 dB | exact |
| Exercise 4: jitter for 90 dB at 10 kHz | 1.59 ns | ±0.05 ns |
| Exercise 4: jitter for 90 dB at 1 kHz | 15.9 ns | ±0.5 ns |
| Exercise 7: total latency | 2,136 ms | ±10 ms |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

## 14. Interview questions

**1. What is the difference between resolution and ENOB?**
A strong answer defines resolution as the number of code bits and ENOB as the SINAD-derived effective bits, and gives an example where they differ by more than 10 bits. A weak answer treats them as synonyms.

**2. Why can a 24-bit ADC give an 8-bit measurement?**
A strong answer identifies the dominant noise sources and computes the system ENOB from the signal amplitude and noise. A weak answer blames the converter alone.

**3. What is aliasing, and how do you prevent it?**
A strong answer gives the folding equation, an example, and the analogue filter requirement. A weak answer says "sample fast enough" without the filter.

**4. Why does clock jitter matter more at high input frequencies?**
A strong answer explains the slope-proportional error and the \(1/(2\pi f t_j)\) limit. A weak answer says jitter adds noise without the frequency dependence.

**5. How do you choose between a SAR and a delta-sigma converter?**
A strong answer compares bandwidth, resolution, latency, anti-alias burden, and power. A weak answer chooses by resolution alone.

**6. What is INL, and why does it matter for accuracy?**
A strong answer defines it as deviation from a straight line and shows how it limits mid-scale accuracy even when noise is low. A weak answer confuses it with DNL.

**7. What does a histogram test measure?**
A strong answer names offset, gain, INL, DNL, and missing codes, and explains the need for many hits per code. A weak answer says it measures noise.

**8. What does an FFT test measure?**
A strong answer names SNR, THD, SINAD, SFDR, and the need for coherent sampling. A weak answer says it measures frequency response.

**9. How does the reference affect the measurement?**
A strong answer explains full scale, noise, drift, and drive, and refers the noise to the input by the gain. A weak answer says the reference is "just a voltage".

**10. What would make you reject a converter choice?**
A strong answer names an alias that cannot be filtered, a latency that misses the clinical requirement, an INL that breaks the accuracy claim, or a reference that cannot be made quiet enough. A weak answer rejects on price or bit count.

**11. How do you test for aliasing?**
A strong answer injects a tone above Nyquist, measures the alias amplitude, and repeats at several frequencies and sample rates. A weak answer relies on the datasheet's anti-alias specification.

**12. Why is the reference part of the converter?**
A strong answer explains full scale, noise, drift, and drive, and refers the noise to the input by the gain. A weak answer treats the reference as an ideal voltage.

## 15. Key terms

- **ADC** — analogue-to-digital converter.
- **Aliasing** — folding of frequencies above Nyquist into the band.
- **Anti-alias filter** — analogue filter before sampling.
- **Aperture jitter** — variation in the sampling instant.
- **Coherent sampling** — input frequency an integer number of cycles in the record.
- **Decimation** — reducing the sample rate after filtering.
- **Delta-sigma** — oversampling converter with noise shaping.
- **DNL** — differential nonlinearity; deviation of a code width.
- **ENOB** — effective number of bits.
- **FFT** — fast Fourier transform.
- **Full-scale range** — input range mapped to all codes.
- **Histogram test** — code-count test for static linearity.
- **INL** — integral nonlinearity; deviation from a straight line.
- **LSB** — least significant bit; smallest code step.
- **Missing code** — a code that never appears.
- **Monotonicity** — output never decreases as input increases.
- **Nyquist frequency** — half the sampling rate.
- **OSR** — oversampling ratio.
- **Pipeline** — multi-stage high-speed converter.
- **Quantisation noise** — error from mapping to discrete codes.
- **Reference** — voltage that defines full scale.
- **SAR** — successive-approximation register converter.
- **SINAD** — signal power divided by noise and distortion.
- **Sinc filter** — digital low-pass used in decimation.
- **SNR** — signal-to-noise ratio.
- **SFDR** — spurious-free dynamic range.
- **THD** — total harmonic distortion.
- **Settling time** — time to reach a stated accuracy.
- **Sample-and-hold** — circuit that holds the input for conversion.
- **Throughput** — samples per second actually delivered.
- **Acquisition time** — time the sample-and-hold spends charging.
- **Aperture delay** — nominal delay between the clock edge and the sample.
- **Gain error** — deviation of the slope from ideal.
- **Offset error** — output at zero input.
- **Monotonic** — output never decreases as input increases.
- **Noise shaping** — moving quantisation noise out of the signal band.
- **Settling** — recovery of the input or reference after a transient.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Datasheets and tutorials**

1. Texas Instruments, *ADS131M08 8-Channel, Simultaneously-Sampling, 24-Bit, Delta-Sigma ADC*, SBAS950B. https://www.ti.com/lit/ds/symlink/ads131m08.pdf
2. Texas Instruments, *ADS129x Low-Power, 2-Channel, 24-Bit Analog Front-End*, SBAS502C, 2020. https://www.ti.com/lit/ds/symlink/ads1292.pdf
3. Analog Devices, *MT-001: Taking the Mystery out of the Infamous Formula, SNR = 6.02N + 1.76 dB*. https://www.analog.com/media/en/training-seminars/tutorials/MT-001.pdf
4. Analog Devices, *MT-007: Aperture Time, Aperture Jitter, Aperture Delay Time*. https://www.analog.com/mt-007
5. Analog Devices, *AN-501: Aperture Uncertainty and ADC System Performance*. https://www.analog.com/an-501
6. Analog Devices, *DN1013: Understanding the Effect of Clock Jitter on High Speed ADCs*. https://www.analog.com/media/en/reference-design-documentation/design-notes/dn1013f.pdf

**Standards**

7. IEC 60601-2-25:2011, *Particular requirements for electrocardiographs*. https://webstore.iec.ch/en/publication/26368
8. IEC 60601-2-27:2011, *Particular requirements for electrocardiographic monitoring equipment*. https://webstore.iec.ch/en/publication/26400
9. IEC 60601-1:2005+AMD1:2012+AMD2:2020. https://webstore.iec.ch/en/publication/26169
10. IEC 60601-1-2:2014+AMD1:2020. https://webstore.iec.ch/en/publication/67554
11. IEC 60601-1-8:2006+AMD2:2020. https://webstore.iec.ch/en/publication/26539
12. ISO 14971:2019. https://www.iso.org/standard/72704.html
13. IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
14. IEEE 1057-2017, *Standard for Digitizing Waveform Recorders*. https://standards.ieee.org/standard/1057-2017.html
15. IEEE 1241-2010, *Standard Terminology for Analog-to-Digital Converters*. https://standards.ieee.org/standard/1241-2010.html

**Background**

16. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
17. W. Kester (ed.), *Data Conversion Handbook*, Analog Devices, 2005. https://www.analog.com/en/resources/technical-books/data-conversion-handbook.html
18. R. Schreier and G. C. Temes, *Understanding Delta-Sigma Data Converters*, Wiley, 2005.
19. M. Pelgrom, *Analog-to-Digital Conversion*, 3rd ed., Springer, 2017.
20. Analog Devices, *MT-200: Minimizing Jitter in ADC Clock Interfaces*. https://www.analog.com/mt-200
21. Analog Devices, *Understanding AC Behaviors of High Speed ADCs*. https://www.analog.com/en/resources/technical-articles/understanding-ac-behaviors-of-high-speed-adcs.html
22. Texas Instruments, *ADS131M08 Evaluation Module User's Guide*, SBAU334A. https://www.ti.com/lit/ug/sbau334a/sbau334a.pdf
23. IEC 60601-2-25:2011, clause 201.12.4.107.2, as summarised by WhaleTeq AN_11. https://www.whaleteq.com/data/adownload/files/1672134706493523860.pdf
24. IEEE 1057-2017 and IEEE 1241-2010, test methods and terminology.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_analog-front-ends.md`](03_analog-front-ends.md)
- Next topic: [`05_mcus-dsps-and-fpgas.md`](05_mcus-dsps-and-fpgas.md)
- Related: [`01_the-signal-chain.md`](01_the-signal-chain.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on quantization: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab on ECG quality: [`../../labs/11_ecg_quality.ipynb`](../../labs/11_ecg_quality.ipynb)
- Lab index: [`../../labs/`](../../labs/)

### 7.2 Alias frequencies for several interferers

The alias equation has many solutions. The script below lists the frequencies that alias to 50 Hz at a 500 SPS rate for the first three multiples of the sampling frequency, and the attenuation of the single RC filter at each.

```python
import math

fs = 500.0
fc = 1 / (2 * math.pi * 1000 * 100e-9)
for k in (1, 2, 3):
    for sign in (-1, 1):
        f = k * fs + sign * 50
        att = 20 * math.log10(math.sqrt(1 + (f / fc)**2))
        print(f"k={k} sign={sign:+d} f={f:6.0f} Hz att={att:5.2f} dB")
```

**Expected output.**

```text
k=1 sign=-1 f=   450 Hz att= 0.33 dB
k=1 sign=+1 f=   550 Hz att= 0.49 dB
k=2 sign=-1 f=   950 Hz att= 1.32 dB
k=2 sign=+1 f=  1050 Hz att= 1.57 dB
k=3 sign=-1 f=  1450 Hz att= 2.62 dB
k=3 sign=+1 f=  1550 Hz att= 2.90 dB
```

Every one of these interferers produces the same 50 Hz alias, and the single RC attenuates none of them enough. A filter design that considers only the first alias is incomplete.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 10 | 5 | 20 |
| 2 | 5 | 15 | 10 | 30 |
| 3 | 5 | 25 | 10 | 40 |
| 4 | 5 | 10 | 10 | 25 |
| 5 | 5 | 15 | 10 | 30 |
| 6 | 5 | 25 | 10 | 40 |
| 7 | 5 | 15 | 10 | 30 |
| 8 | 5 | 10 | 15 | 30 |

**More key terms.**

- **Alias frequency** — the frequency at which an out-of-band signal appears after sampling.
- **Coherent** — an input frequency that is an integer number of cycles in the record.
- **Dither** — a small added noise that randomises quantisation.
- **Equivalent noise bandwidth** — the bandwidth of an ideal filter passing the same noise power.
- **Full-scale sine** — a sine whose peak equals the full-scale range.
- **Noise floor** — the noise power in the absence of signal.
- **Reference drive** — the ability of a reference to supply the converter's charge pulses.
- **Stopband attenuation** — the attenuation of a filter above its cutoff.
25. IEC 60601-1-11:2015+AMD1:2020, home healthcare environment. https://webstore.iec.ch/en/publication/67384
26. ISO 80601-2-61:2017, pulse oximeter equipment. https://www.iso.org/standard/67963.html
27. Analog Devices, *Linear Circuit Design Handbook*, 2008. https://www.analog.com/en/resources/technical-books/linear-circuit-design-handbook.html
28. B. Razavi, *Principles of Data Conversion System Design*, IEEE Press, 1995.
29. IEEE 1241-2010, ADC terminology.
30. IEC 60601-2-25:2011, Table 201.107 frequency response.
31. Texas Instruments, *ADS131M08 product page*. https://www.ti.com/product/ADS131M08
32. Analog Devices, *MT-229: Quantization Noise*. https://www.analog.com/media/en/training-seminars/tutorials/MT-229.pdf
