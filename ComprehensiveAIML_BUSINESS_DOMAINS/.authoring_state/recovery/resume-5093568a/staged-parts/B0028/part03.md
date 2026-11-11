
## 5. Worked Numerical Example

### 5.1 The situation

An analyst receives a 30-minute two-channel ambulatory record in the MIT-BIH style: 360 samples per second per channel, 11-bit resolution over a 10 mV range, stored as integer codes with a header that states the rate and the range but **not** the analog filter. The stated task is rhythm review in a 0.5–40 Hz band. A downstream pipeline decimates the record by four to 90 Hz before plotting, because the plots are cheaper to render and the vendor's viewer does that.

The analyst observes a steady 10.0 Hz oscillation at about 0.2 mV peak-to-peak on the first channel, present for the whole record, and asks whether it is physiological. Everything below is ILLUSTRATIVE: the record is constructed, the amplitudes are chosen for arithmetic, and no measurement of a patient or device is claimed.

Three candidate sources exist in the analyst's notes, each with a stated amplitude at the electrode:

```text
Source A: mains fundamental at  50 Hz, 2.0 mV peak
Source B: mains third harmonic at 150 Hz, 0.5 mV peak
Source C: equipment switching tone at 350 Hz, 0.2 mV peak
```

The analyst must determine which of those three can appear at 10.0 Hz, through which route, and what evidence would separate them.

### 5.2 Step 1 — the time base and the sample budget

```text
fs        = 360 Hz per channel
Ts        = 1/360 s = 2.7778 ms
duration  = 1800 s
N         = 1800 * 360 = 648,000 samples per channel
```

Two channels are stored, so the file holds 1,296,000 samples in total. That total is not a rate. At 16 bits per sample the raw payload is `1,296,000 × 2 bytes = 2,592,000 bytes = 2.59 MB`, which is consistent with the public database's stated 104.3 MB uncompressed total once annotation files, headers, and the extra records are included; that consistency is a sanity check on the arithmetic, not a measurement of this constructed record.

### 5.3 Step 2 — folding each candidate at 360 Hz

```text
Alias map at fs = 360 Hz, fN = 180 Hz:
  fa = |f - k*360|, k nearest to f/360

Source A, f =  50 Hz, k = 0 -> fa = 50 Hz        (below Nyquist, no fold)
Source B, f = 150 Hz, k = 0 -> fa = 150 Hz       (below Nyquist, no fold)
Source C, f = 350 Hz, k = 1 -> fa = |350-360| = 10 Hz

Check k choices:
  50/360  = 0.139 -> k = 0
  150/360 = 0.417 -> k = 0
  350/360 = 0.972 -> k = 1
```

So at the native 360 Hz rate, only Source C produces energy at 10 Hz. Sources A and B land at 50 and 150 Hz, both outside the 0.5–40 Hz task band, and both would be removed by any digital low-pass at 40 Hz applied at 360 Hz. Source C lands exactly inside the band at 10 Hz and survives that filter, because to a digital filter at 360 Hz the sequence *is* a 10 Hz signal.

An important negative result follows immediately: because Source C's 10 Hz is genuine baseband energy after sampling, the analyst's 40 Hz low-pass cannot remove it without also removing a physiological 10 Hz component. The chain has no way to distinguish them, which is the non-recoverability invariant in action.

### 5.4 Step 3 — the second fold introduced by decimation

The pipeline decimates by four. Whether that is safe depends on what is between 45 and 180 Hz at the native rate.

```text
Decimation by M = 4: 360 Hz -> 90 Hz, new fN = 45 Hz
Folding map at 90 Hz:
  Source A baseband  50 Hz, k = 1 -> |50 - 90| = 40 Hz
  Source B baseband 150 Hz, k = 2 -> |150 - 180| = 30 Hz
  Source C baseband  10 Hz, k = 0 -> 10 Hz

Also, the pre-decimation spectrum contains the FOLDED images:
  Source C at 110 Hz (the image of 350 Hz after k=1) -> |110 - 90| = 20 Hz
```

Three separate components now sit inside the 0–45 Hz band of the decimated record: 40 Hz from Source A, 30 Hz from Source B, 10 Hz from Source C, and 20 Hz from Source C's image. Without a filter before the decimation, every one of those is a spurious component that a 40 Hz plot cannot distinguish from physiology.

This is the decisive computation for the analyst's question. The observed 10 Hz survives decimation unchanged, but the decimation also creates a 20 Hz line and a 30 Hz line and a 40 Hz line from sources that were never in the task band. If the analyst's plot shows only 10 Hz and nothing else, that is evidence that a filter *was* applied before decimation — and the absence of the 20 Hz line is precisely the check that reveals whether the pipeline filtered at all.

### 5.5 Step 4 — how much attenuation is enough

The tolerance is set by the smallest feature the task must resolve, not by taste. Using the converter step derived in Section 4.8:

```text
q = 10 mV / 2048 = 4.8828 uV per code
```

Requiring every folded interferer to fall below one code gives a per-source attenuation requirement:

```text
Source A, 2.0 mV  = 2000 uV: 20*log10(2000/4.8828) = 52.2 dB
Source B, 0.5 mV  =  500 uV: 20*log10( 500/4.8828) = 40.2 dB
Source C, 0.2 mV  =  200 uV: 20*log10( 200/4.8828) = 32.2 dB
```

So the analog filter ahead of the ADC needs at least 52.2 dB at 50 Hz if Source A must be suppressed to one code, 40.2 dB at 150 Hz for Source B, and 32.2 dB at 350 Hz for Source C. Those are three different frequencies with three different requirements, and a single low-pass can meet all three with far less steepness than the worst single case suggests: a filter that is 52 dB down at 50 Hz will be far more than 52 dB down at 350 Hz if it has any reasonable order.

Now invert the question and ask what the *digital* filter before decimation must do. Its job is not to suppress sources that the analog filter already suppressed; its job is to suppress whatever survived to 360 Hz. If the analog filter delivered 52 dB at 50 Hz, then Source A arrives at the ADC with `2000 uV / 10^(52.2/20) = 4.9 uV`, which is already at one code, and no additional digital attenuation is needed for it. The digital filter's real job is the region between 45 and 180 Hz, where the analog filter's attenuation is unknown because its specification was not recorded — which is exactly the metadata gap that Section 1 warns about.

### 5.6 Step 5 — the filter order that the requirement forces

Using the order expression of Section 4.7 at `fs = 360 Hz` with `Ap = 1 dB` and `As = 60 dB` (a 7.8 dB margin over the 52.2 dB requirement, chosen so that the passband ripple and stopband margin do not eat the budget):

```text
fst =  50 Hz -> N = 30.61 -> 31 poles
fst =  60 Hz -> N = 16.44 -> 17 poles
fst =  90 Hz -> N =  7.50 ->  8 poles
fst = 140 Hz -> N =  3.75 ->  4 poles
fst = 180 Hz -> N =  0.20 ->  1 pole
```

Read this table as a trade curve. A 31-pole filter at 50 Hz is a difficult analog design with tight component tolerances and a group delay that varies strongly across the passband. An 8-pole filter at 90 Hz is a routine design. The reason the second is acceptable is that the *decimation* only folds energy above 45 Hz, and a filter that is 60 dB down at 90 Hz still attenuates the 45–90 Hz region substantially even though it does not meet 60 dB there. The design question becomes: what is the actual interferer amplitude in the 45–90 Hz region? If the answer is unknown, the honest design is the 31-pole one, or a re-acquisition at a higher rate.

### 5.7 Step 6 — the analysis window and what it can resolve

The analyst wants to separate the observed 10.0 Hz from a possible 10.5 Hz physiological rhythm, both inside a 30-minute record.

```text
N     = 648,000 samples
T     = 1800 s
df    = 360/648,000 = 0.0005556 Hz = 0.556 mHz per bin

10.0 Hz lands at bin 10.0/0.0005556 = 18,000 (exactly)
10.5 Hz lands at bin 18,900 (exactly)
Separation = 900 bins
```

A separation of 900 bins is enormous, so the two components are trivially separable in a full-record transform — provided the 10 Hz component is stationary for all 1800 s. That proviso is the real constraint, not the resolution: a component that drifts in frequency by more than 0.556 mHz over the record smears across bins and may look like a broad hump rather than a line.

Redo the same computation on the 10-second segments the analyst actually plots:

```text
N = 3600, T = 10 s, df = 0.1 Hz
10.0 Hz -> bin 100
10.5 Hz -> bin 105
Separation = 5 bins
```

Five bins is still separable with a rectangular window, but the rectangular window's first side lobe is 13.26 dB below the main lobe, so a 0.2 mV component at 10.0 Hz and a 0.02 mV component at 10.5 Hz would show the smaller one buried under the larger one's side lobe. A Hann window would separate them at the cost of widening each main lobe to about four bins; a 5-bin separation then sits at the edge of resolution. The choice of window is therefore part of the answer to "can these be separated", not a cosmetic detail.

### 5.8 Step 7 — an independent arithmetic re-check

Every number above is re-derived here by a different route, because a worked example that cannot be re-checked is a story.

```text
Check 1 (alias): 350 Hz at 360 Hz.
  Alternative route: the sequence is cos(2*pi*350*n/360) = cos(2*pi*n*(350/360)).
  350/360 = 35/36 = 1 - 1/36, and cos(2*pi*n*(1 - 1/36)) = cos(2*pi*n - 2*pi*n/36) = cos(2*pi*n/36).
  cos(2*pi*n/36) has period 36 samples, i.e. 360/36 = 10 Hz. Agrees with 10 Hz.

Check 2 (decimation fold): 110 Hz at 90 Hz.
  Alternative route: 110/90 = 11/9, so fa = 90*(11/9 - 1) = 90*(2/9) = 20 Hz. Agrees with 20 Hz.

Check 3 (LSB): 10 mV over 2048 codes.
  2048 = 2^11, so q = 10/2048 mV = 0.0048828 mV = 4.8828 uV.
  Cross-check: 1000 uV / 4.8828 uV = 204.8 codes per millivolt, and 2048/10 = 204.8 codes per millivolt. Agrees.

Check 4 (bin spacing): 360/648,000.
  360/648,000 = 1/1800 = 0.0005555... Hz, and 1/T = 1/1800. Agrees.

Check 5 (attenuation): 2000 uV to 4.8828 uV.
  Ratio = 409.6. 20*log10(409.6) = 20*(2.6124) = 52.25 dB. Agrees with 52.2 dB.

Check 6 (order at 50 Hz):
  numerator   = log10((10^6 - 1)/(10^0.1 - 1)) = log10(999999/0.25893) = log10(3.8621e6) = 6.5868
  denominator = log10(tan(pi*50/360)/tan(pi*40/360)) = log10(0.46631/0.36397) = log10(1.28116) = 0.10755
  N = 0.5 * 6.5868 / 0.10755 = 30.62. Agrees with 30.61 within rounding of the tangents.
```

Six checks, six agreements. The residual 0.01-pole disagreement in Check 6 comes from rounding the tangent ratio to five decimal places; a practitioner should not read significance into a filter order below about 0.1.

### 5.9 Step 8 — sensitivity: what changes the answer

| Parameter changed | From | To | Effect on the conclusion |
|---|---|---|---|
| Sample rate | 360 Hz | 500 Hz | 350 Hz folds to 150 Hz, not 10 Hz; the 10 Hz observation is no longer explained by Source C |
| Sample rate | 360 Hz | 250 Hz | 350 Hz folds to 100 Hz, then decimation by 2 to 125 Hz folds it to 25 Hz |
| Interferer amplitude | 0.2 mV | 2.0 mV | Required attenuation rises from 32.2 to 52.2 dB; the analog filter order rises by roughly 20 poles at a 50 Hz edge |
| Bit depth | 11 bits | 16 bits | `q` falls from 4.88 to 0.305 uV; all attenuation requirements rise by 24.1 dB |
| Task band | 0.5–40 Hz | 0.05–150 Hz | Decimation by four becomes impossible without a much sharper pre-filter; the whole design changes |
| Decimation factor | 4 | 2 | New Nyquist becomes 90 Hz, so the 45–90 Hz transition disappears and the pre-filter is cheap |

**Most sensitive parameter.** The sample rate dominates. Changing 360 Hz to 500 Hz moves Source C from inside the task band to outside it, which flips the diagnosis completely. This is why "re-acquire at a different rate" is the strongest available discriminator against an aliased interferer: it is the only test that changes the folding map while leaving the physical source unchanged.

**Least sensitive parameter.** Bit depth, within the range 11 to 16 bits, changes the attenuation requirements uniformly by about 1.5 dB per bit but does not change which frequencies fold where. A resolution complaint is therefore a different problem from a folding complaint, and the two should be diagnosed separately.

### 5.10 What the analyst can and cannot conclude

The observation of a 10.0 Hz component is *consistent* with a 350 Hz interferer folding at 360 Hz. It is also consistent with a genuine 10 Hz physiological oscillation, and with several other analog frequencies (370, 710, 730 Hz). The record contains no information that separates those hypotheses, because the separation was destroyed at the ADC. What the analyst can conclude is:

- the component is exactly at a frequency that a 350 Hz source would fold to, and exactly on a bin boundary of the full-record transform, which is what a coherent artifact does and what a biological oscillator usually does not;
- the component's amplitude is stable across the whole record, which is more typical of an equipment source than of a rhythm;
- the pipeline's decimation should have produced a 20 Hz companion line if no pre-filter existed, and the absence of that line is evidence about the pipeline rather than about the source.

What the analyst cannot conclude is that the source *is* 350 Hz. Stating the hypothesis, the check that would separate it, and the fact that the check has not been run is the difference between an analysis and a guess.

## 6. Visual / Geometric Intuition

### 6.1 The folding fan

Draw a horizontal axis of analog frequency from 0 to `4·fs`. Mark the baseband interval `[0, fs/2]`. Now draw the alias map for each integer `k`: the interval `[k·fs − fs/2, k·fs + fs/2]` folds back onto the baseband, reversed for odd `k` and preserved for even `k`. The result looks like a fan of intervals stacked side by side, each of width `fs`, each mapping onto the same baseband.

```text
analog frequency axis, fs = 360 Hz:

0        180        540        900       1260
|--base--|          |          |          |
     ^        ^          ^          ^
     |        |          |          |
  [0,180] [180,540]  [540,900]  [900,1260]
     |        |          |          |
     v        v          v          v
  0..180   180..0     0..180     180..0     <- direction of the fold
```

The fan makes three facts visible at once. First, every analog frequency has exactly one image in the baseband, so the map is a function from analog to baseband (many-to-one). Second, the direction reverses at each `fs/2` boundary, which is why a frequency just above `fs/2` appears just below `fs/2` rather than near zero. Third, the width of each interval equals `fs`, so raising the rate widens every interval and pushes the first fold boundary further out.

Marking 350 Hz on this diagram puts it in the second interval at 10 Hz from the left edge of that interval, folding to 10 Hz from the right edge of the baseband, which is 10 Hz. Marking 370 Hz puts it 10 Hz from the right edge of the second interval, folding to 10 Hz as well. The two analog frequencies are mirror images about 360 Hz and are indistinguishable after sampling — the diagram shows why without any algebra.

### 6.2 The sine that becomes a slower sine

Plot `cos(2π·350·t)` on a fine grid and put sample dots at `t = n/360`. The continuous curve completes 35 cycles in 36 samples' worth of time, which is nearly one cycle per sample; the dots trace a slow 10 Hz envelope because each successive sample lands 1/36 of a cycle earlier in phase. The picture is the geometric content of the derivation in Section 4.3: subtracting one full cycle per sample leaves a residual of `1/36` cycle per sample, which is 10 Hz at 360 samples per second.

The same picture drawn at 500 Hz puts the dots at a different phase increment, `350/500 = 0.7` cycle per sample, which is `0.7 − 1 = −0.3` cycle per sample, or 150 Hz. The continuous curve is unchanged; only the spacing of the dots changed. That is the entire mechanism of aliasing, and it is visible without any transform.

### 6.3 The spectrum as a set of copies

Sampling multiplies the spectrum by a train of impulses in frequency, which replicates the original spectrum at every multiple of `fs`. Draw the original spectrum as a triangle from 0 to 40 Hz. After sampling at 360 Hz the drawing contains copies centred at 0, ±360, ±720 Hz. If the original spectrum were wider than `fs/2`, adjacent copies would overlap, and the overlap region is the aliased part: energy from one copy landing on top of another.

```text
before sampling:      /\            (0..40 Hz)
after sampling:   /\      /\      /\
                --+-------+-------+----+--> f
                  0      360     720
```

Overlap appears when the triangle's base exceeds `fs/2`. The 40 Hz task band in a 360 Hz system has a base of 40 Hz against a half-period of 180 Hz, so the copies are widely separated and the fan has room. In a 100 Hz system the copies are 100 Hz apart and a 40 Hz band occupies 40 Hz of that, leaving 60 Hz of gap — and any energy between 50 and 60 Hz folds into the gap between copies, landing inside the 40 Hz band.

### 6.4 The aperture as a smearing window

Draw the sample-and-hold as a rectangle of width `τ` sliding along the time axis: each output is the area under the input weighted by that rectangle. A fast oscillation averages to nearly zero under a wide rectangle; a slow one passes almost unchanged. The sinc response in Section 4.5 is the frequency-domain picture of that rectangle, and its first null at `1/τ` is the frequency at which one full cycle fits exactly inside the aperture and averages to zero.

Plotting the sinc curve on a decibel axis shows the practical problem: between `fN` and `1/τ` it falls only gradually, reaching 3.9 dB at `fN` for a full-period hold. The curve is a gentle slope where a filter needs a cliff, which is why designers add a real analog filter rather than relying on the aperture.

### 6.5 The window's main lobe and side lobes

Draw the transform of a rectangular window: a tall narrow peak with side lobes at 13.26 dB below it, decaying slowly. Draw the Hann window's transform beside it: a wider peak (about two bins each side) with side lobes at 31 dB. Overlay a weak component five bins away from a strong one and observe that under the rectangular window it sits inside the strong component's skirt, while under the Hann window it sits clear.

The geometric lesson is that window choice moves the *threshold of visibility*, not the underlying spectrum. If the question is "is there energy at 10.5 Hz", the answer depends on the window; if the question is "what is the total energy in the band", a wider main lobe requires a wider band definition to compensate. Neither window is more correct; they answer different questions.

### 6.6 The decimation staircase

Draw the signal chain as a staircase descending in rate: 360 → 180 → 90 → 45 Hz. At each step, draw a horizontal line at the new Nyquist frequency and shade the region above it, which is the energy that must be removed *before* that step. The shading shows that each step only needs to protect its own band, and that a gentle filter at a high rate can remove a wide region cheaply, leaving a narrow region for a sharper filter at a lower rate.

```text
360 Hz  |############|  remove >90 Hz before first /2
180 Hz  |#####|         remove >45 Hz before second /2
 90 Hz  |##|            keep 0..45 Hz for the task
```

The staircase also shows the failure mode: if a step is missing, the shading from that step lands inside the band of the next, and the diagram acquires a spurious component that no later filter can remove.

### 6.7 The amplitude ruler

Draw the 10 mV converter range as a ruler from −5 mV to +5 mV, divided into 2048 equal ticks of 4.88 uV. Mark a 2 mV peak-to-peak QRS as 205 ticks (10 percent of the ruler), a 100 uV ST deviation as 20 ticks, a 300 uV T wave as 61 ticks, and a 1 mV baseline wander as 205 ticks. The picture shows immediately why baseline removal is not cosmetic: wander consumes the same ruler length as the QRS, so a wander of 6 mV pushes the QRS off the end of the ruler entirely, and clipping is a nonlinearity that no filter can undo.

### 6.8 Reading a suspicious spectrum

A practical reading order for the picture:

1. Locate the Nyquist frequency on the plot and mark it; anything above it in a native-rate plot is already an alias.
2. Check whether the suspicious frequency is an exact integer divisor relationship with the rate: 10 Hz at 360 Hz is `360/36`, which is the signature of a fold from `360 − 10`.
3. Check whether the suspicious frequency is a harmonic of the mains fundamental: 10 Hz is not, but 150 Hz and 250 Hz are harmonics of 50 Hz, and their folds are the usual suspects.
4. Check the width of the line: a coherent artifact is narrow (limited only by the window), while a biological oscillator drifts.
5. Check for companion lines: a folded source leaves images at predictable places, and their presence or absence is evidence about the filter chain.

**Takeaway for Section 6.** Every geometric picture in this section is a way of seeing the same map: the baseband holds one image of many analog frequencies, and the job is to know which one.
