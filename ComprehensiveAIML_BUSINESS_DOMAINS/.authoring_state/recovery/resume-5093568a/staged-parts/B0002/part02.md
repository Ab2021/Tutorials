### 4.3 Variants: window aggregates, cyclical pairs, and hashed keys

Three encodings recur so often that they deserve their own derivations, and each one has a leak or a distortion that is invisible without algebra.

#### 4.3.1 Trailing-window aggregates and the label-window overlap

**Preconditions.** An event process with rate `λ` events per day; a feature window of width `W` days ending at `t_cut`; a label window of `H` days starting at `t_cut`; a bug in which the feature window was implemented as `[t_cut − W, t_cut + δ]` for some positive `δ`.

**State transition.** The feature's expected value rises from `λW` to `λ(W+δ)` while its nominal definition still says `λW`. The label is a function of events in `(t_cut, t_cut + H]`, so any `δ > 0` overlaps the label window by exactly `δ` days.

**Invariant.** A feature window must intersect the label window in an empty set. Written as a set identity: `[t_cut − W, t_cut] ∩ (t_cut, t_cut + H] = ∅`. This invariant is checkable in code by comparing the SQL window bounds against the label bounds, which is why it belongs in a test rather than in a document.

```text
WINDOW LEAK
FORMULA
  E[feature] = λ(W + δ) instead of λW
  excess in standard deviations = λδ / sqrt(λW)
WHERE
  λ is the event rate per day, W the window width in days
  δ is the unintended overlap in days, assumed small relative to W
CONSTRAINT
  events are Poisson with constant rate; features are counts, not rates
NUMERICAL EXAMPLE
  λ = 0.02 claims per policy-day, W = 180 days, δ = 14 days
  correct mean = 3.60   correct sd = sqrt(3.60) = 1.897
  leaked mean  = 3.88   excess = 0.28 events = 0.148 sd
  overlap fraction δ/W = 7.8%
INTERPRETATION
  a 14-day overlap is invisible in a histogram of a count feature with mean 3.6,
  yet it carries 0.28 events of label-window information per row
```

The reason this is dangerous rather than merely wrong: a 0.148-sd shift is well inside normal-looking variation, so no distribution check flags it, and the label-window events are precisely the ones the label is made of. If the label is "any claim in the next 90 days" and the overlapping 14 days contain a claim, the feature and the label are mechanically correlated, and the model will weight this feature far above its true predictive value.

**Boundary.** As `δ → W`, the excess reaches `λW` and the feature is a trailing-plus-forward count with no separation at all; the derived shift `λδ/sqrt(λW)` stays finite while the actual information about the label goes to its maximum. The linearised shift formula is therefore *not* a leakage detector — it only quantifies the distributional move. The detector is the set-intersection test, and the formula exists to explain why distribution checks miss the problem.

#### 4.3.2 Cyclical time pairs and their chord distortion

**Preconditions.** A coordinate `h` in hours with period `P = 24`; a model using either the raw integer or the pair `(sin(2πh/24), cos(2πh/24))`.

**State transition.** Raw encoding maps hour 23 and hour 0 to values that are 23 units apart in the feature space and 1 hour apart in time. The pair maps them to points on the unit circle whose Euclidean distance is `2 sin(π/24) = 0.2611`, matching the true circular separation.

**Invariant and justification.** The pair preserves circular distance in the sense that Euclidean distance between encoded points is a strictly increasing function of the circular arc length, specifically `d = 2 sin(a/2)` for arc length `a` in radians. Strict monotonicity is what a distance-based model needs; exact proportionality is not available, and it is not needed.

```text
CYCLICAL ENCODING DISTORTION
FORMULA
  chord(a) = 2 sin(a/2)      arc(a) = a
  compression ratio = chord(a)/arc(a) = sin(a/2)/(a/2)
WHERE
  a = 2πΔh/P is the arc length between two hours, Δh in hours, P = 24
CONSTRAINT
  a in (0, π]; the mapping is monotone but not isometric
NUMERICAL EXAMPLE
  Δh = 1  -> a = 0.2618, chord = 0.2611, ratio 0.997
  Δh = 3  -> a = 0.7854, chord = 0.7654, ratio 0.975
  Δh = 6  -> a = 1.5708, chord = 1.4142, ratio 0.900
  Δh = 12 -> a = 3.1416, chord = 2.0000, ratio 0.637
INTERPRETATION
  the pair compresses long separations by up to 36%; opposite hours are 2 units
  apart, not π, so any distance threshold must be set on chord units
```

The practical consequence is that "half a day apart" and "a quarter day apart" are not separated by a factor of 2 in the encoded space but by `2.0/1.414 = 1.414`. A nearest-neighbour or kernel model that treats the two columns as ordinary Euclidean coordinates therefore weights nearby hours more heavily than the geometry of a clock suggests. That is acceptable when the underlying risk is itself smooth in clock time, and wrong when the risk is a threshold (for example a cut-off at 06:00), in which case the correct feature is the indicator, not the pair.

**Two-column budget.** Replacing 24 one-hot columns with 2 cyclical columns reduces width by a factor of 12, and each column has variance 0.5 under a uniform hour distribution, so the pair carries the full circular coordinate. But the pair cannot represent a *non-periodic* interaction with a third variable cleanly: if risk differs by "hour >= 22 or hour < 6", the linear model on `sin` and `cos` can represent that only approximately, whereas an explicit night indicator reproduces it exactly. Use the pair for continuous periodic effects and indicators for named intervals, and do not expect the pair to substitute for the indicator.

#### 4.3.3 Hashed encodings and the collision budget

**Preconditions.** `k` distinct levels (for example, free-text reason codes after normalisation); a hash map into `M` buckets, `M` a power of two; collisions are tolerable if rare.

**State transition.** Encode with a width-`M` vector where bucket `h(s) mod M` is 1 for the row's level. Distinct levels that collide become indistinguishable, which merges their statistics exactly as if the two levels were one.

**Invariant.** For a well-mixed hash function and `k << sqrt(M)`, the number of colliding pairs is approximately `k(k−1)/(2M)` and the number of occupied buckets is `M(1 − e^{−k/M})`. Both come from the birthday-problem approximation and both are accurate to a few percent for the sizes below.

```text
HASH COLLISION BUDGET
FORMULA
  E[colliding pairs]  = k(k−1) / (2M)
  E[occupied buckets] = M (1 − e^{−k/M})
  share of levels sharing a bucket = (k − occupied) / k
WHERE
  k = number of distinct levels, M = number of buckets (power of two)
CONSTRAINT
  uniform hashing, k << M; at k on the order of sqrt(M) collisions are no
  longer rare events but the norm
NUMERICAL EXAMPLE
  k = 40,000 levels, M = 2^18 = 262,144
  pairs     = 40000·39999 / 524288 = 3,051.7
  occupied  = 262144 · (1 − e^{−0.15259}) = 37,097.7
  sharing   = (40000 − 37097.7)/40000 = 7.26%
INTERPRETATION
  about 2,900 levels lose their identity; if the levels are reason codes and the
  top 20 codes carry 60% of rows, those 20 mostly survive, so the loss is
  concentrated in the long tail that the encoding was meant to serve
```

The collision count is not the decision variable; the *mass-weighted* collision count is. A `k = 40,000` column in which the top 20 levels cover 60% of rows loses almost no rows to collisions even though 7.3% of levels collide. A `k = 400` column with a flat distribution loses 0.08% of levels but those levels are a large share of rows. Compute the share of rows in levels that collide with a more frequent level, and compare it against the tolerable error, before choosing `M`.

**Boundary.** At `k = M`, `E[occupied] = M(1 − e^{−1}) = 0.632M`, so 37% of levels collide and the encoding is closer to a random projection than a categorical encoding. At that point the hashing trick is still usable as a *dimensionality reduction* — that is what it was designed for — but it is no longer an identity-preserving encoding, and any per-level coefficient interpretation must be abandoned.

### 4.4 Limits and counterexamples

Four limits matter because each one is a case where the correct-looking pattern is the wrong one.

**Limit 1: smoothing cannot fix a level with zero positives.** If `c_ℓ = 0` and `n_ℓ = 5`,000, the smoothed encoding is `m·p̄/(5000 + m)`, which for `m = 38` and `p̄ = 0.086` is `0.00065` — visually zero and statistically indistinguishable from zero. The encoding is not wrong; the problem is that a zero-event level is the single strongest signal in most risk models and the encoding compresses it into the same region as a level with 1 event in 5,000. The fix is a transform of the encoding (log-odds with a floor) or a second column flagging zero-event levels, not a larger `m`.

**Limit 2: information value is not comparable across cardinalities.** The same underlying variable binned 5 ways scored `IV = 0.3629`; binned 20 ways on the identical population it scored `0.3715` in a 200-replicate simulation that split each level's rows into four bins without changing any true rate. The bias grows as bins get smaller relative to the event count. Comparing a 5-bin and a 20-bin encoding of the *same* variable by `IV` therefore favours the 20-bin version for no reason other than bin count. The fix is to hold bin count fixed when comparing variables, or to use a bin-count-invariant measure.

**Limit 3: IV and mutual information answer different questions, and the gap is large.** On the example table `IV = 0.3629` nats, which converts to `0.3629/ln 2 = 0.5236` bits. The mutual information between band and label is `0.01916` bits against a label entropy of `0.42298` bits — a ratio of 4.53%. So the correct statement is not that the two measures disagree in direction (both are positive, both are correct) but that `IV` overstates *usable* information by a factor of 27 in this population: `0.5236/0.01916 = 27.3`. The mechanism is the weighting. `IV` weights each level's divergence by the class-conditional mass `P(ℓ|y)`, so a rare level that is strongly skewed still contributes a large term; mutual information weights by the joint mass `P(ℓ,y)`, so a level carrying 1.2% of positives cannot contribute much regardless of how skewed it is. Per-level ratios in this table run from 16.5 (level A) to 22.7 (level E), meaning every level overstates its joint contribution, not just the rare ones.

**Limit 4: the exchangeability assumption behind the prior breaks for time-varying populations.** `p̄` is computed on the training rows, but if the population's base rate is trending — say 0.075 in the first month and 0.098 in the third — the pooled `p̄ = 0.086` is a prior that no individual month agrees with. The encodings then shrink toward a value that was never true at any single point in time, which biases the level encodings of the most recent month. The diagnostic is to refit `m` and `p̄` per month and check whether they move; if they do, the prior must be time-indexed.

## 5. Worked Numerical Example

### 5.1 The dataset, and why it is shaped this way

The example is a five-level risk band on a 48,000-row population with a 8.6% base rate. The levels are deliberately unbalanced (18,000 down to 2,400) and the rates are deliberately monotone (0.140 down to 0.020), because that shape exposes both the leak and the shrinkage in one table. All values are ILLUSTRATIVE; the arithmetic is exact as shown.

```text
level   n_ℓ      c_ℓ     r̂_ℓ      share of rows   share of positives
A      18,000   2,520    0.1400      37.5%             61.0%
B      12,000     900    0.0750      25.0%             21.8%
C       9,600     480    0.0500      20.0%             11.6%
D       6,000     180    0.0300      12.5%              4.4%
E       2,400      48    0.0200       5.0%              1.2%
total  48,000   4,128    0.0860     100.0%            100.0%
```

Check the totals independently: `18000·0.14 = 2520`; `12000·0.075 = 900`; `9600·0.05 = 480`; `6000·0.03 = 180`; `2400·0.02 = 48`. Sum `= 2520+900+480+180+48 = 4128`. `4128/48000 = 0.086`. The column shares sum to 1.000 by construction, and the positive shares `0.610 + 0.218 + 0.116 + 0.044 + 0.012 = 1.000`.

### 5.2 Derivation 1: smoothed encodings with the derived constant

Using `m* = 38.45` from Section 4.2 (rounded to 38.5 for display) and `p̄ = 0.086`:

```text
level  formula                                    value     vs raw    relative
A      (2520 + 38.45·0.086) / (18000 + 38.45)     0.139880  −0.000120  −0.09%
B      (900  + 38.45·0.086) / (12000 + 38.45)     0.075037  +0.000037  +0.05%
C      (480  + 38.45·0.086) / (9600  + 38.45)     0.050149  +0.000149  +0.30%
D      (180  + 38.45·0.086) / (6000  + 38.45)     0.030372  +0.000372  +1.24%
E      (48   + 38.45·0.086) / (2400  + 38.45)     0.021042  +0.001042  +5.21%
```

Independent re-check for level E, in two orders of operations. Direct: `38.45 · 0.086 = 3.3067`; `48 + 3.3067 = 51.3067`; `2400 + 38.45 = 2438.45`; `51.3067/2438.45 = 0.021042`. Via the alternative form `r̃ = p̄ + w(r̂ − p̄)` with `w = n/(n+m) = 2400/2438.45 = 0.984231`: `0.021042 = 0.086 + 0.984231·(0.020 − 0.086) = 0.086 − 0.064959 = 0.021041`. Both agree to 6 decimals, and they must, because the two forms are algebraically identical:

```text
  (c + m p̄)/(n + m) = (n r̂ + m p̄)/(n + m) = p̄ + [n/(n+m)](r̂ − p̄)
```

**Sensitivity.** The treatment moves the rarest level by 5.21% of its own value while moving the largest by 0.09% — a 58-fold difference. That ratio is exactly `(n_A + m)/(n_E + m) = 18038.45/2438.45 = 7.40` in shrinkage weight, and the rate difference amplifies it. This is why a single smoothing constant is safe for the head of the distribution and consequential only in the tail, and why the tail is where a hard-coded `m` causes trouble.

Now compare against two alternatives for level E:

```text
variant                          encoding   error vs raw   verdict
raw rate, no smoothing            0.020000    0.000000      leaks and is noisy
m = 10 (folklore)                 0.020274   +0.000274     3.6% less shrunk
m = 38.45 (derived)               0.021042   +0.001042      balanced
m = 500 (over-shrunk)             0.031379   +0.011379     level mostly erased
```

The last row is `(48 + 500·0.086)/(2400 + 500) = (48 + 43)/2900 = 91/2900 = 0.031379`. Over-shrinkage is not hypothetical. At `m = 500` the encoding has covered `(0.031379 − 0.020)/(0.086 − 0.020) = 0.1724` of the distance to the base rate — 17%, not most of it — but the collateral damage is larger than that number suggests: level D under the same constant is `(180 + 43)/6500 = 0.034308`, so the D-to-E separation falls from `0.030 − 0.020 = 0.010` to `0.034308 − 0.031379 = 0.002929`, a 71% loss. Shrinkage pulls levels toward a common prior in proportion to their counts, and levels with similar counts lose their mutual separation almost completely long before any single encoding looks obviously wrong.

### 5.3 Derivation 2: information value and its cardinality inflation

Compute the weight of evidence and information value for the 5-level encoding.

```text
level  P(ℓ|y=1)   P(ℓ|y=0)     WOE = ln(col1/col2)   IV contribution
A       0.61047     0.35284         +0.54819             0.141226
B       0.21802     0.25301         −0.14882             0.005207
C       0.11628     0.20788         −0.58096             0.053215
D       0.04360     0.13266         −1.11262             0.099083
E       0.01163     0.05361         −1.52834             0.064164
                                               total     0.362893
```

Re-check the denominators and one row by hand. `P(A|y=1) = 2520/4128 = 0.610465`; `P(A|y=0) = (18000−2520)/(48000−4128) = 15480/43872 = 0.352844`; `WOE_A = ln(0.610465/0.352844) = ln(1.730134) = 0.548194`; contribution `= (0.610465 − 0.352844)·0.548194 = 0.257621·0.548194 = 0.141226`. Level E: `48/4128 = 0.011628`; `2352/43872 = 0.053611`; `ln(0.011628/0.053611) = ln(0.216891) = −1.528344`; contribution `= (0.011628 − 0.053611)·(−1.528344) = (−0.041983)·(−1.528344) = 0.064162`. The 0.000002 difference from the table is display rounding.

The sum is dominated by level A (39% of the total) and levels C, D, E together supply another 60%. Level B — 25% of all rows — contributes 1.4% of the information value. Reading `IV` alone would therefore tell you this is a strong feature; reading it per level tells you the strength lives entirely in the extremes, which is exactly where the sample sizes are smallest and where Section 5.4's variance matters most.

Now the cardinality inflation check:

```text
bins   positives per bin (mean)   IV       note
2      2,064                     0.3317   A vs B..E merged
5        826                     0.3629   the encoding above
20       206                     0.3715   200-replicate simulation, same
                                           generating rates, noise only
```

The 20-bin version adds no information: the simulation split each level's rows into four bins without changing the level's true rate. `IV` rose by 0.0086, or 2.4%, purely from binomial noise. That is the bias term of `IV` under bin splitting, and it is why `IV` thresholds quoted as absolute numbers ("above 0.3 is strong") are meaningless without a bin count.

### 5.4 Derivation 3: the out-of-fold variance of the rarest level

Take `K = 5` folds, so each fold holds 480 of level E's 2,400 rows and the encoding for a held-out row is computed from `n^{(-k)} = 1,920` rows.

```text
FORMULA     Var[r̃] = n^{(-k)} p (1 − p) / (n^{(-k)} + m)²
WHERE       p = 0.020, n^{(-k)} = 1920, m = 38.45
CONSTRAINT  independence across rows within the level; m fixed
EXAMPLE     numerator   = 1920 · 0.02 · 0.98 = 37.632
            denominator = (1920 + 38.45)² = 1958.45² = 3,835,526
            Var = 9.8116e−6, sd = 0.0031325
            encoding = (38.4 + 3.3067)/1958.45 = 0.021300
            relative sd = 0.0031325 / 0.021300 = 14.7%
INTERPRETATION  the rare level's training encoding moves by about ±3% of the
            base rate across folds, i.e. its log-odds moves by ±0.15
```

Check the arithmetic independently. `1920 · 0.02 = 38.4`; `38.4 · 0.98 = 37.632`. `1920 + 38.45 = 1958.45`; `1958.45² = 3,835,527` (since `1958.45² = 3,835,525.4`). `37.632/3,835,525.4 = 9.8114e−6`; `sqrt = 0.0031323`. Encoding: `c^{(-k)}` for a level with `48·0.8 = 38.4` positives → `(38.4 + 3.3067)/1958.45 = 41.7067/1958.45 = 0.021296`. Relative `0.0031323/0.021296 = 14.71%`.

**Sensitivity to fold count.** At `K = 2`, `n^{(-k)} = 1,200` and the sd rises to `0.0040`; at `K = 10`, `n^{(-k)} = 2,160` and it falls to `0.00295`. The gain from more folds shrinks like `1/sqrt(1 − 1/K)`: going from 5 to 10 folds reduces the sd by 5.9%, while the number of fitted encodings doubles. That is the quantitative case for `K = 5` as a default.

### 5.5 Derivation 4: population stability index on a drifted band distribution

Reference month proportions `a = (0.375, 0.250, 0.200, 0.125, 0.050)` and current month `b = (0.330, 0.245, 0.205, 0.140, 0.080)`.

```text
level   a_i     b_i     a−b      ln(a/b)    contribution
A      0.375   0.330  +0.045   +0.127833     0.005753
B      0.250   0.245  +0.005   +0.020203     0.000101
C      0.200   0.205  −0.005   −0.024693     0.000123
D      0.125   0.140  −0.015   −0.113329     0.001700
E      0.050   0.080  −0.030   −0.470004     0.014100
                                        PSI  0.021777
```

Both distributions sum to 1.000, which is a precondition: `0.375+0.250+0.200+0.125+0.050 = 1.000` and `0.330+0.245+0.205+0.140+0.080 = 1.000`. Each term is non-negative because `(a−b)` and `ln(a/b)` always share sign, which is why `PSI` cannot be negative and why a `PSI` of 0.0218 means a genuinely small move.

**Sensitivity.** Suppose level E doubles to 0.100 and level A falls to 0.310, with B, C, D unchanged. The recomputed terms are `A: (0.375 − 0.310)·ln(0.375/0.310) = 0.065 · 0.190354 = 0.012373`, `B: 0.000101`, `C: 0.000123`, `D: 0.001700`, `E: (0.050 − 0.100)·ln(0.050/0.100) = (−0.050)(−0.693147) = 0.034657`. Total `PSI = 0.048955`. The rarest level doubled — a fact that would change a risk model's output for every row in that band — and `PSI` moved from 0.022 to 0.049, still well below the conventional 0.10 "stable" line. `PSI` is a row-mass-weighted statistic and the tail has little mass; a level-count monitor or a per-band share alarm is required alongside it.
