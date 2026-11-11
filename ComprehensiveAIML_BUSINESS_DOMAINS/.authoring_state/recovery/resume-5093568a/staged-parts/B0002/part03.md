## 6. Visual / Geometric Intuition

### 6.1 Four pictures that carry the whole sheet

The algebra in Section 4 has geometric content, and the four pictures below are what a reviewer should be able to sketch on a whiteboard without the formulas.

**Picture 1: the shrinkage seesaw.** Plot each level's raw rate against its row count on log-log axes. The raw rates of rare levels scatter widely around the population mean; as counts grow, the scatter narrows like `1/sqrt(n)`. Smoothing replaces each observed rate with a point on the straight line between the raw rate and `p̄`, at a fraction `m/(n+m)` of the distance. So the shrinkage is *negligible at the head and large at the tail*, and the correct shape of the transform is a curve that bends the tail toward `p̄` while leaving the head alone. If your encoding maps a 18,000-row level and a 2,400-row level with the same weight, the seesaw has no fulcrum and you are just adding noise to the head.

**Picture 2: the circle versus the line.** A raw integer hour is a straight line from 0 to 23. Two things sit at the ends of that line that are adjacent in reality: 23:00 and 00:00. The `(sin, cos)` pair rolls the line into a circle so that the ends meet. Draw the circle, mark 00:00 at angle 0 and 12:00 at angle π, and then draw the chord between 00:00 and 12:00 — its length is 2.0, while the arc between them is 3.1416. That gap (a factor of 0.637) is the distortion quantified in Section 4.3.2, and it is visible as "the far side of the clock looks closer than it is".

**Picture 3: the collision cloud.** With `k` levels falling into `M` buckets, draw `M` boxes and drop `k` balls. The number of boxes hit by two or more balls is `k − M(1 − e^{−k/M})`. At `k/M = 0.15` (the 40,000-into-262,144 case) that is 7.3% of levels; at `k/M = 1` it is 36.8%. The picture makes a point that the number alone hides: collisions concentrate in the *occupied* region, so doubling `M` does not halve the damage linearly — going from `2^18` to `2^20` cuts the sharing from 7.26% to 1.88%, a factor of 3.9, for a 4× memory cost.

**Picture 4: the window that grows a tail.** Draw the timeline: `t_cut` as a vertical line, the feature window as a shaded interval ending exactly on the line, the label window as a second shaded interval starting exactly at the line. A bug where the feature window extends past `t_cut` by `δ` produces a thin sliver of overlap. The sliver's *area* is δ/W of the feature window — 7.8% in the worked example — which looks small, but the sliver is inside the label window, so all of its signal is label signal. This picture is the fastest way to explain to a product owner why a 14-day off-by-one is a modelling defect rather than a cosmetic one.

### 6.2 Geometry of the encoding space

The choice of encoding fixes the geometry the downstream model sees, and different estimators are sensitive to different properties of that geometry.

| Encoding | Space | Distance between levels | Sensitivity |
|---|---|---|---|
| One-hot | vertices of a simplex in `R^k` | all pairs equal (`sqrt 2`) | any model; width `k` |
| Ordinal integer | points on a line in `R` | proportional to code difference | trees fine, linear wrong |
| Target encoding | points on a line in `R` | proportional to rate difference | linear fine, monotone |
| WOE | points on a line in `R`, natural-log scale | log-odds difference | logistic models, additive |
| Hashed | vertices of a simplex in `R^M`, shared | unknown, collisions collapse | high-capacity models |
| Cyclical pair | unit circle | chord of the arc | distance-based models |

The one-hot column set gives every level equal distance from every other level, which is the right assumption when levels are nominal and unrelated (region codes), and the wrong assumption when levels are ordered (risk bands, where A and B are more similar than A and E). An integer code imposes equal spacing on an ordered variable and a spurious ordering on a nominal one — which is why "ordinal encode everything" is a defect that a tree model hides and a linear model punishes.

The target-encoding space is a line where distance equals rate difference, and this interacts with the loss: a logistic model applied to a target-encoded column is fitting a linear-in-log-odds relationship to a quantity that is linear in probability, which is a mis-specification unless the column is converted to WOE or log-odds first. That is a real and frequently missed step.

### 6.3 Why the circle needs two columns and not one

A single column `sin(2πh/24)` maps hour `h` and hour `12 − h` to the same value, because `sin(θ) = sin(π − θ)`. So 01:00 and 11:00 are indistinguishable, and 02:00 and 10:00 likewise. Adding `cos` breaks the tie because `cos` is symmetric about 0 rather than about π/2: `cos(2π·1/24) = 0.9659` while `cos(2π·11/24) = −0.2588`. Together the pair is injective on the 24-hour circle (up to the 24 discrete values), which is the minimum number of real coordinates needed to embed a circle in Euclidean space: one coordinate is a line, and a circle is not a line.

This is an instance of a general rule worth stating: **a periodic coordinate needs two columns, and a periodic coordinate crossed with a second variable needs four.** If risk depends on hour *and* on weekday, and both are periodic, the product of two circles is a torus, which needs four coordinates to embed without distortion; two `sin`/`cos` pairs is the standard construction and it captures the interaction that separate additive features cannot.

### 6.4 The monotone constraint picture

A risk model with a monotone constraint on the target-encoded band column imposes `score(A) >= score(B) >= ... >= score(E)`. In the encoding space this is a constraint on the fitted coefficient sign, not on the encoding, so it works only if the encoding preserves order. Target encoding preserves order because it is a monotone function of the rate within the training data — but *smoothing can break the order* when two levels have similar rates and very different counts. Concretely, take two levels with raw rates 0.030 (n = 6,000) and 0.028 (n = 600). Raw order: 0.030 > 0.028. Smoothed with `m = 38.45`, `p̄ = 0.086`:

```text
level  n      raw     smoothed        formula
D      6000   0.030   0.030357        (180 + 3.307)/6038.45
D'      600   0.028   0.031493        (16.8 + 3.307)/638.45
```

The small level's smoothed value is *higher* than the large level's, so the encoded column now contradicts the raw ordering that the business rule asserts. Anyone who imposes a monotone constraint on the encoded column and then checks it against raw rates will see a violation that is not a bug in the constraint — it is the correct empirical-Bayes behaviour, because a 600-row level with 2.8% is too noisy to outrank a 6,000-row level with 3.0%. The resolution is to order levels by smoothed value at fit time, not by raw rate, and to record that choice.

The same effect appears with a level whose row count changes rather than its rate. Take the D' level at three counts, all with a raw 2.8%:

```text
n      c      smoothed    rank versus D (0.030357)
600    16.8   0.031493    above D
240     6.72   0.036009    well above D
120     3.36   0.042074    far above D
```

Every shrinkage of a below-average level moves it *toward* 0.086, and the smaller the count the further it moves. Consequently the monotone order of the smoothed column is a function of counts as much as of risk, and a segment that loses rows to an eligibility change can silently jump up the ranking. The guard is to freeze the level→ordinal mapping at fit time and assert that the mapping is unchanged at serving time; a mapping that changes between vintages means the constraint is being applied to a different ordering than the one that was validated.

### 6.5 Geometry of window aggregates

A count over a window `W` has mean `λW` and variance `λW`, so it lives on a square-root scale: plot `sqrt(count)` and the variance is roughly constant across levels of the mean. This is why `sqrt` or `log1p` transforms of count features stabilise tree splits, and why a raw count with mean 3.6 and maximum 40 forces a tree to spend most of its split budget on the handful of extreme values. The transform is not cosmetic: it changes which splits are available.

The window width itself is a bias-variance dial. A 7-day window has low variance and high noise; a 365-day window has low noise but responds slowly to a regime change and overlaps with the label window if the label horizon is long. The practical rule is that the window should be long enough to contain at least 10 expected events at the median row (`λW >= 10`), which for `λ = 0.02` per day means `W >= 500` days — a figure large enough to explain why claims-count features on rare-event populations are almost always too noisy to be useful at short windows.

## 7. Comparison / Selection Table

### 7.1 Encoding selection by column semantics and cardinality

Read the first column as the fact you know about the column, then take the row that matches. The "avoid when" column is the operative one; most encoding mistakes are a correct choice applied where its precondition fails.

| Column semantics | Cardinality | Recommended | Avoid when | Failure signature |
|---|---|---|---|---|
| Nominal, low (region, product) | 3–30 | one-hot | categories with < 200 rows each | rare levels get unstable coefficients |
| Nominal, medium | 30–1,000 | target encoding with OOF + `m*` | levels with < 10 events expected | encoding variance dominates |
| Nominal, high (ids, postcodes) | > 5,000 | hashed, `M >= 8k`, or frequency count | the level matters *and* rows/level < 5 | collision merges, unlearnable |
| Ordinal (risk band) | 3–20 | ordinal integer or target/ WOE | bands are not truly ordered | linear model imposes equal spacing |
| Binary | 2 | single 0/1 column | — | duplicated columns |
| Periodic (hour, month) | 12–24 | sin/cos pair | risk is a threshold at one time | chord distortion at 12h |
| Cyclical + interval | 24 + 3 | pair plus explicit indicators | interval is the only signal | pair approximates the step poorly |
| Count / event volume | unbounded | `log1p` or `sqrt`, plus zero flag | counts are genuinely rare (95% zeros) | splits wasted on the tail |
| Continuous, heavy tail | unbounded | winsorise at p99 + `log1p` | the tail *is* the signal | tail information erased |
| Text, short (reason codes) | 100–10,000 | normalise, then target or hashed | raw strings used as levels | near-duplicate levels split counts |
| Free text | open | TF-IDF then dense reduction | vocabulary < 100 terms | sparse noise, 40k columns |
| Time since event | unbounded | days, plus indicator for "never" | "never" coded as 0 or 999 | a single value dominates |

### 7.2 The five encodings against operational criteria

The table above chooses by semantics. This one chooses by the criterion that usually decides in practice: what breaks first when the data changes.

| Criterion | One-hot | Ordinal | Target (OOF) | WOE | Hashed |
|---|---|---|---|---|---|
| Width | `k` | 1 | 1 | 1 | `M` |
| Leak risk if fitted carelessly | none | none | severe | severe | moderate |
| Unseen level at serving | all zeros | undefined | needs prior fallback | needs prior fallback | hashes to a bucket |
| Interpretable coefficient | yes | yes | yes | yes | no |
| Handles `k = 40,000` | no (2.3 GB dense fp32) | yes | noise-limited | noise-limited | yes |
| Update frequency needed | none | none | every refit | every refit | none |
| Monotone-friendly | no | yes | yes (if ordered) | yes | no |
| Works with trees directly | yes | yes | yes (overfits) | yes | yes |

Three cells deserve comment because they are the ones that get argued about in reviews.

*Unseen level.* One-hot produces an all-zero row, which for a linear model means the level contributes exactly the intercept — harmless. Target/WOE encodings need an explicit fallback (the prior `p̄`, or a dedicated "unseen" bucket), and omitting it is a common serving-time crash: the lookup raises `KeyError` or returns `NaN`, and the score becomes `NaN` for exactly the rows the model has never seen, which is the worst possible failure population.

*Leak risk "moderate" for hashing.* Hashing does not itself leak, but hashing is often applied to a combination of columns (`user_id × day`), which is a near-unique key, and then target encoding is applied on top of the hashed buckets. The leak risk enters through the composition, not through the hash.

*Interpretable coefficient.* A hashed coefficient is a sum over all levels that land in that bucket, so attributing it to any single level is invalid. Reporting it as though it were a level effect is a specific, checkable error: if the coefficient's sign flips when the hash seed changes, it was never a level effect.

### 7.3 Selection procedure with a stopping rule

The procedure below is ordered so that each step can veto the next, and it terminates in a decision rather than a comparison table.

1. **Establish the null.** Fit the model with the column dropped. Record the metric and its uncertainty interval. Without this, every later comparison is a difference of two numbers with no scale.
2. **Compute the level table.** Rows and positives per level, with the minimum count and minimum expected events printed. If any level has fewer than `10/p̄` rows (1,163 rows at `p̄ = 0.086` for 10 expected events), record it as tail and expect to merge.
3. **Derive `m*`.** Use Section 4.2. If `m* > 0.25·N/k` — meaning the between-level spread is smaller than the noise — stop: the column has no level-level signal, and the correct encoding is a constant or the column is dropped.
4. **Fit with OOF at `K = 5`.** Compare against the null on the validation split, not the training split, and report the interval. If the interval includes zero, stop.
5. **Test the prequential version.** Refit the encoding on data up to `t` and score `(t, t+1]`. If the OOF gain vanishes, the gain was a fold artefact and the column is conditionally useless on future data.
6. **Price it.** Width in columns, memory, per-row compute, refresh cadence. If the cost exceeds the budget from Section 1.3, replace with a cheaper encoding that keeps most of the gain.
7. **Register the monitor.** Level shares, encoding distribution, unseen-level rate, and the fallback path. A column that reaches step 7 without a monitor is not finished.

The stopping rule matters as much as the steps: at step 3 and step 4 the honest answer is often "drop it", and teams that skip the null never learn that most candidate columns are worth nothing.

### 7.4 When not to engineer a feature at all

Three situations where the correct action is to stop rather than to encode better:

- **The column is a proxy for the label's process.** If "number of claim documents uploaded" is collected by the same workflow that closes claims, it will predict the label at whatever point in the workflow it is measured. No encoding fixes this; only a cut-off audit does, and the audit usually ends with dropping the column.
- **The column is an identifier.** A row identifier, a policy number, or a timestamp with second resolution has no stable relationship to the label, and any measured relationship is memorisation. Target-encoding an identifier is the canonical way to build a model that scores 0.95 in validation and 0.51 in production.
- **The column's population is not the scoring population.** A feature available only for customers who completed an onboarding form is missing-not-at-random for everyone else, and the model learns the form-completion pattern. This is a population problem, and Section 8.3 gives the discriminating check.

## 8. Failure Modes & Edge Cases

Each entry states the mechanism, at least two competing causes that produce a similar symptom, and the check that separates them. "Monitor the model" is not a check; each discriminating check below names a quantity, a comparison, and a threshold.

### 8.1 Off-by-one window includes the label window

**Mechanism.** The feature query bounds are `[t_cut − W, t_cut + δ]` with `δ > 0`, typically because a date filter was written as `<= date_add(t_cut, 14)` to capture "late-arriving" events, or because the label window start was set one day before `t_cut`. The feature now contains label-window events, and the model's discrimination jumps for that column alone.

**Competing causes.** (a) The column is genuinely predictive because it is a legitimate trailing aggregate. (b) The validation split shares the same bug, so the gain looks stable. (c) The label itself was computed with a window that starts before `t_cut`, moving the leak into the label instead of the feature.

**Discriminating check.** Recompute the feature with the window closed at `t_cut` and refit. If the column's coefficient drops by more than 30% or its permutation importance falls by more than half, the leak was real. To separate (a) from (b): the legitimately predictive column keeps its importance when the window is *shortened* by a week, while a leaking column loses a disproportionate share because the overlap is a fixed δ regardless of W. To test (c), recompute the label with an explicit `> t_cut` bound and check whether the positive count changes; if it does, both the label and the feature are wrong.

### 8.2 Target encoding fitted on the full table

**Mechanism.** The encoding statistics are computed once on all rows, then the table is split. Each row's encoding contains `1/n_ℓ` of its own label. The model detects a column that partially *is* the label and gives it high weight; validation shows a large gain because the validation rows were encoded the same way.

**Competing causes.** (a) The column is genuinely strong. (b) The gain is real but the encoding variance is what the model is exploiting, not the signal — a low-cardinality column encoded naively can act as a noisy copy of the label. (c) The validation split is not time-ordered, so the "gain" reflects the population being homogeneous across the split.

**Discriminating check.** Fit the naive encoding and the OOF encoding and compare their *validation* metrics, not training. Naive typically shows a validation gain that exceeds OOF by a margin proportional to `1/n_ℓ`; on the worked example, the rarest level carries 2.4% of its own label at `n = 2,400` and 100% at `n = 1`. A second check: sort the levels by `n_ℓ` and plot the train-validation gap per level. Leakage produces a gap that is largest for the smallest levels and near zero for the largest — a specific, falsifiable shape that a genuinely strong column does not produce.

### 8.3 Missing-not-at-random join

**Mechanism.** The feature comes from a table that only has rows for a subset of the population (customers who registered, claims that were digitised, devices that reported). The left join produces `NULL`, which is silently filled with 0 or the mean. The resulting indicator of "missing" is often more predictive than the imputed value, because missingness correlates with the label's process.

**Competing causes.** (a) Missingness is random and the imputation is fine. (b) The missingness indicator is the true signal and the imputed value is noise. (c) The population join itself is wrong — the key is not the scoring grain — so the join both loses rows and duplicates others.

**Discriminating check.** Report three numbers per joined column: join hit rate, target rate among rows with a hit, target rate among rows without. If the two rates differ by more than the width of their confidence intervals, missingness carries signal and an explicit indicator is required. To separate (b): fit with both the indicator and the imputed value, then drop the value and refit. If the metric holds, the value was noise. To test (c): compare `COUNT(*)` before and after the join; any increase means duplicate keys at the join grain and the aggregates are inflated.

### 8.4 Rare level with zero events treated as zero risk

**Mechanism.** A level with no positives gets an encoding at or near 0.00065 (Section 4.4). Downstream, a tree splits at 0.001 and treats the level as the lowest-risk category, when in fact the level's true rate is unknown and its confidence interval extends well above the population mean. The model's most confident-looking prediction sits on the least-supported level.

**Competing causes.** (a) The level really is near-zero risk, and a large sample confirms it. (b) The level is a data-quality artefact — a mistyped code, a legacy system, a test account. (c) The level's rows arrive only recently, so the label window has not closed for most of them.

**Discriminating check.** Compute the one-sided 95% upper bound on the level's rate; for 0 events in 2,400 rows with `m = 38.45` the smoothed value is 0.00065 and the exact upper bound is `1 − 0.05^{1/2400} = 0.00125`, which is 69× below the base rate — so the bound itself supports "low risk" only if the rows are trustworthy. Test (b) by listing the level's most recent 20 records and checking the source system; a legacy code with a creation date after the migration is an artefact. Test (c) by comparing the label-observation completeness per level: if the level's rows cluster in the last 90 days, its zero count is censoring, not risk.

### 8.5 Cyclical feature applied to a non-cyclic coordinate

**Mechanism.** A "month" column is encoded as `sin/cos` with period 12, which asserts that December and January are adjacent in risk. For many business quantities that is false — fiscal year boundaries, school terms, and promotion cycles make December and January very different — and the encoding forces the model to treat them as neighbours.

**Competing causes.** (a) The coordinate genuinely is cyclic and the encoding is right. (b) The coordinate is cyclic but has a discontinuity that the pair smooths over. (c) The model is linear, so the two columns can only represent a sinusoidal effect, which is too rigid regardless of cyclicity.

**Discriminating check.** Fit two models: one with the `sin/cos` pair, one with a 12-level one-hot. Compare validation metrics. If one-hot wins by more than the interval, the coordinate is not smooth-cyclic at the resolution that matters. A direct check on the data: compute the risk rate for the two boundary periods and for the two periods adjacent to them. If `|rate(Dec) − rate(Jan)|` is comparable to `|rate(Nov) − rate(Dec)|`, the cyclic assumption is harmless; if the boundary difference is several times larger, the cycle has a seam and one-hot (or a seam indicator) is correct.

### 8.6 Aggregate computed over a window that crosses a population change

**Mechanism.** A 365-day trailing count is computed for rows in a month when the product definition changed 200 days earlier. For rows after the change, the window mixes two populations; for rows before, it does not. The feature's distribution is therefore different for early and late rows, and a model trained on the mixed data learns an average relationship that holds for neither.

**Competing causes.** (a) The product change is irrelevant to the feature's meaning. (b) The feature is fine but the *label* definition changed, which is a worse problem. (c) The apparent difference is ordinary seasonality.

**Discriminating check.** Split the training rows at the change date and refit the feature-to-label relationship in each half; compare the slopes (or the binned rates). If the relationship differs by more than the intervals, the window must be truncated at the change or the feature must be segmented. To separate (c), repeat the same split at a date with no known change; if the difference is of similar magnitude, the effect is seasonal noise rather than a regime change.

### 8.7 Hash collision concentrated in a high-mass level

**Mechanism.** Two frequent levels collide into one bucket. The bucket's statistic is the pooled rate, and every row in either level is scored with the pooled value. The damage is proportional to the mass of the colliding levels, not to the collision count, so a pipeline that monitors "collision rate" (7.26% here) misses a case where 40% of rows lose their level identity.

**Competing causes.** (a) The collision is harmless because the two levels have similar rates. (b) The hash function is not uniform because the level strings share a prefix (many implementations hash the string, so `CODE_1000` through `CODE_1999` are well spread, but truncated or numeric-parsed keys can cluster). (c) The width `M` is adequate but the level set grew past the design assumption.

**Discriminating check.** Build the collision map keyed by bucket, and for each bucket compute the share of rows and the spread of true rates among the levels in it. Flag any bucket where the row share exceeds 1% and the rate spread exceeds the pooled standard error. For (b), test the hash with the *unhashed* value: compare the observed bucket occupancy against the birthday expectation; a chi-square test on occupancy with `M − 1` degrees of freedom detects non-uniformity, and a simple check is whether the maximum bucket load exceeds `k/M + 4·sqrt(k/M)`.

### 8.8 Encoding drift from a stale refit cadence

**Mechanism.** The estimator is refitted monthly, the encoding statistics quarterly. The model's coefficients are tuned to encodings from Q1 while the served encodings come from Q2, and the two vintages disagree on the levels that moved. The model is not wrong at any single point, but its calibration degrades in a way that no per-model monitor attributes to the encoding.

**Competing causes.** (a) Genuine population drift. (b) The label definition changed and the base rate moved. (c) The feature pipeline is computing a different thing after a code change.

**Discriminating check.** Store the encoding table's `fit_timestamp` and its `p̄` and `m` in the model artifact. At scoring time, log which encoding table was used. If the served table's `fit_timestamp` is older than the estimator's by more than one cycle, the vintages are misaligned by construction. To separate (a) from (c): recompute the encoding from the *original* code on current data. If the distributions match, the drift is population; if they differ, the code changed the feature.

### 8.9 Float32 and count overflow in aggregate features

**Mechanism.** A cumulative count stored in float32 has 24 bits of mantissa, so integers above `2^24 = 16,777,216` are not exactly representable; the increment `+1` is lost. A running total of lifetime events crosses that boundary after enough accumulation, and the feature silently saturates.

**Competing causes.** (a) The count is genuinely capped by the source system. (b) The aggregation query joins a table with duplicate keys and inflates the count. (c) The window boundary is wrong, so the count covers the wrong period.

**Discriminating check.** Compare the float32 value against the integer value from a direct `COUNT(*)` for the ten largest rows. Any mismatch is representability, not aggregation. For (b), compare `COUNT(DISTINCT event_id)` against `COUNT(*)`; a ratio above 1.0 proves duplication.

### 8.10 Edge case summary

| Edge case | Symptom | Guard |
|---|---|---|
| Level present in validation, absent in training | NaN encoding at serving | explicit unseen fallback = `p̄` |
| All rows of a level in one fold | encoding undefined | cap `K` so `n_ℓ/K >= 5` |
| Window `W < 1/λ` | count feature is 0 for most rows | require `λW >= 10` |
| Level with `c = n` | encoding = 1.0, log-odds infinite | clip encoding to `[ε, 1−ε]` |
| Ties in quantile binning | bins with unequal width | break ties by row id, record the rule |
| Negative or zero values before `log` | `−inf` or NaN | use `log1p`, assert `x >= 0` |
| Timezone mismatch in an hour feature | 5.5-hour shift on some regions | store UTC, convert once at a named step |
| Duplicate rows after a join | inflated counts, duplicated keys | assert unique grain post-join |
