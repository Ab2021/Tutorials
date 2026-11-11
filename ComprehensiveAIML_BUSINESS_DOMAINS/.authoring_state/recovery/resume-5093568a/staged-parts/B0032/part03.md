## 5. Worked Numerical Example

The scenario is constructed and ILLUSTRATIVE: a two-channel bedside ECG front end, 10 mV bipolar
span, 12-bit converter, nominal 250 Hz per channel, a declared signal band of 0.5–40 Hz, an
8th-order analog low-pass at 45 Hz, an interfering 50 Hz mains-like tone at 5 mV amplitude and a
213 Hz switching artifact at 5 mV. Every number below is computed from those assumptions; none of
it is a measurement of a real device.

### 5.1 Step-by-step computation

```text
Given
  span        S  = 10 mV bipolar
  word        B  = 12 bit
  rate        fs = 250 Hz
  declared band  = 0.5 .. 40 Hz
  analog filter  = 8th-order Butterworth, fc = 45 Hz
  interferers    = 50 Hz @ 5 mV, 213 Hz @ 5 mV

A. Resolution
  LSB       = 10 / 4096            = 2.4414 µV
  sigma_q   = 2.4414 / 3.4641      = 0.7048 µV
  SNR_fs    = 6.02*12 + 1.76       = 74.00 dB

B. Sampling
  Ts        = 1 / 250              = 0.004 s  = 4.000 ms
  f_Nyq     = 125 Hz
  k for 213 = round(213/250) = 1
  alias     = |213 - 250|          = 37.0 Hz
  k for 50  = round(50/250) = 0    -> 50 Hz stays at 50 Hz

C. Analog attenuation (8th order, fc = 45)
  (50/45)   = 1.111111
  (50/45)^16 = 5.3966
  A(50)  = 10*log10(1 + 5.3966)     = 8.06 dB
         -> residual 5 mV * 10^(-8.06/20)  = 1.977 mV
  (213/45)^16 = 1.264e8
  A(213) = 10*log10(1 + 1.264e8)    = 108.03 dB
         -> residual 5 mV * 10^(-108.03/20) = 0.0198 µV

D. Folded residual at 37 Hz
  0.0198 µV / 2.4414 µV per LSB        = 0.0081 LSB
  -> the 213 Hz artifact is below the quantization step after folding

E. In-band contamination at 50 Hz
  1.977 mV / 2.4414 µV per LSB         = 810 LSB  (~9.7 bits of the 12 available)
  relative to a 1 mV QRS peak          = 20*log10(1/1.977) = -5.9 dB
  -> the interferer is larger than the signal; no analog order fixes this economically

F. Observation window
  30 s at 250 Hz = 7500 samples
  Delta_f = 250/7500 = 0.03333 Hz
  50 Hz * 30 s = 1500 periods  -> coherent with N = 7500
  1 Hz * 30 s  = 30 periods    -> coherent
```

### 5.2 Independent re-check

The arithmetic is re-derived by a different route so that a single algebra slip cannot pass.

```text
Check 1 (attenuation from the squared magnitude, not the log identity)
  |H(50)|^2 = 1 / (1 + (50/45)^16)
  (50/45)^2   = 1.234568
  1.234568^8  = 5.39660           [recomputed as ((1.234568^2)^2)^2]
  |H(50)|^2   = 1 / 6.39660 = 0.156333
  |H(50)|     = 0.395390
  A(50)       = -20*log10(0.395390) = 8.059 dB  [matches the 8.06 dB above]

Check 2 (residual voltage against the dB route)
  route A: 5 mV * 0.39539         = 1.97695 mV
  route B: 5 mV * 10^(-8.059/20)  = 5 * 0.39536 = 1.9768 mV
  difference 0.0002 mV -> consistent to four significant figures

Check 3 (LSB and code count)
  10 mV / 4096 = 2.44141 µV
  2.44141 * 4096 = 10.0000 mV     [exact by construction]
  1.97695 mV / 2.44141 µV = 809.8 codes -> 810 integer codes

Check 4 (folding, and the absence of a second fold)
  213 - 250 = -37; magnitude 37
  round(37/250) = 0, so 37 Hz is already inside 0 .. 125 and does not fold again
```

All four checks agree. The figure most likely to be mis-stated in review is the 213 Hz residual
(0.0081 LSB), because it depends on both the 108 dB attenuation and the microvolt conversion; both
are carried through with three significant figures above.

### 5.3 Sensitivity

Each row changes one input and holds the rest fixed. All values are computed from the same
formulas, not measured.

| Change from base case | Effect on the binding number | Consequence |
|---|---|---|
| Word length 12 → 16 bits | `sigma_q` 0.7048 → 0.0440 µV; SNR_fs 74.0 → 98.1 dB | Quantization stops mattering; an assumed 3 µV analog noise floor binds instead |
| Rate 250 → 500 Hz | 213 Hz no longer folds (`k` becomes 0) | The artifact is out of band rather than in band, but a 213 Hz tone before a 250 Hz decimation must still be filtered |
| Analog order 8 → 4 at 45 Hz | A(50) 8.06 → 5.22 dB; residual 1.977 → 2.743 mV | Interferer grows 39%; a notch or a lower corner becomes mandatory |
| Corner 45 → 55 Hz | A(50) 8.06 → 0.86 dB; residual 1.977 → 4.53 mV; A(213) 108 → 94.1 dB | In-band droop improves, 213 Hz margin still ample, 50 Hz rejection collapses |
| Interferer 5 → 20 mV | 810 → 3240 codes, and 20 mV exceeds a 10 mV span | Front end clips; no filter recovers clipped samples |
| Segment 30 s → 8.192 s | Bin spacing 0.0333 → 0.1221 Hz; 50 Hz not coherent | Leakage into neighbouring bins unless a window is applied |

The last two rows are the ones that change decisions. Amplitude sensitivity means range selection
is a sampling decision as much as rate selection: an input that clips has lost information before
any filter or estimator sees it. Window-length sensitivity means a resolution claim is only valid
for the specific segment analysed, so a conclusion drawn from an 8.192 s frame does not transfer
to a 30 s frame without re-checking the bin alignment.

## 6. Visual / Geometric Intuition

### 6.1 The folding cone

Put sampled frequency on the horizontal axis from 0 to `fs`, and let the analog input frequency
run along a diagonal that wraps: every time it crosses a multiple of `fs/2` it reflects, because
the discrete-time frequency axis is a circle of circumference `fs` cut at `fs/2` and unfolded for
display.

```text
input f   0 ---- 125 ---- 250 ---- 375 ---- 500
sampled   0 ---- 125 ---- 0   ---- 125 ---- 0
                       (250 Hz -> DC)     (500 -> DC)
```

Points at `fs/2 + x` are indistinguishable from `fs/2 - x` for a real signal. The region between
`fs/2` and `fs` is a mirror, and above `fs` the pattern repeats. That is why 213 Hz and 37 Hz
coincide: `213 = 250 − 37`, so the two sit symmetrically about the 125 Hz mirror.

The practical use of the picture is reading off the danger bands. For a 40 Hz band at 250 Hz the
dangerous input regions are `210–290` (folding into 0–40 Hz from `k = 1`) and `460–540` (from
`k = 2`). A 213 Hz switcher sits inside the first; a 50 Hz mains tone sits nowhere near one.

### 6.2 The alias-free region as a step function

Draw axes: horizontal = analog input frequency, vertical = attenuation. The anti-alias response is
a curve that is flat in the passband and dives at the corner. The requirement is a step: below
`f_pass` the curve must be near 0 dB, above `fs − f_pass` it must be under the noise floor. The
distance between those two frequencies is the transition band, and its width relative to the
corner drives the order. Plot the required step and the achievable curve together: if the curve
cannot reach the floor before the step's vertical edge, no digital cleverness afterwards fixes it.
For the 250 Hz / 40 Hz case the step is from 40 to 210 Hz, a ratio of 5.25, which is comfortable.
For the 1000 Hz / 100 Hz case the step is from 100 to 400 Hz, a ratio of 4.0, which is also
comfortable. The pathological case is 250 Hz with a declared 100 Hz band: the step is 100 to
125 Hz, a ratio of 1.25, and no practical analog filter achieves 60 dB there.

### 6.3 What one LSB looks like on a cardiac trace

A 1 mV QRS amplitude against a 2.4414 µV LSB spans 410 codes. A 50 µV T wave spans 20 codes. A
10 µV P wave spans 4 codes. That last number is the practical reason 12-bit ECG is marginal for
P-wave morphology: the wave is represented by four discrete levels plus noise, so a downstream
detector sees quantization structure as much as physiology. At 16 bits over the same span the same
P wave spans 66 codes and the quantization structure disappears below the analog noise.

### 6.4 Leakage as a comb, not a spike

A finite-window DFT of a pure tone is not a spike. A rectangular window gives a mainlobe
`2*fs/N` wide with a first sidelobe at −13.4 dB; Hann gives a mainlobe `4*fs/N` wide with a first
sidelobe at −31.5 dB; Hamming reaches −44.2 dB; Blackman −64.1 dB. So a 50 Hz tone in a 30 s
segment rectangular-windowed leaks −13.4 dB into the adjacent region. If a genuine 0.033 Hz
neighbour were 20 dB below the 50 Hz tone it would be buried. The picture to hold: every peak
carries a skirt, and the skirt height is a property of the window, not of the patient.

### 6.5 Where the energy goes when the rate drops

Consider a decimation by 4 from 1000 Hz to 250 Hz. The bands `[125, 250]`, `[250, 375]`,
`[375, 500]` all fold onto `[0, 125]` in reverse order, and every higher band `[k*125, (k+1)*125]`
folds again. Draw the output band four times, mirrored, one mirror per zone. A tone at 375 Hz in
zone 3 lands at `|375 − 250| = 125 Hz`; a tone at 300 Hz lands at 50 Hz. The visual check: count
the non-empty zones in the input spectrum. Every non-empty zone above the first is an error the
decimation filter had to prevent, and the height of that zone's energy above the filter's
stopband floor is the size of the error.
