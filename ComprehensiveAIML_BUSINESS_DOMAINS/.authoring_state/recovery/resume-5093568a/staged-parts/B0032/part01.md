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
