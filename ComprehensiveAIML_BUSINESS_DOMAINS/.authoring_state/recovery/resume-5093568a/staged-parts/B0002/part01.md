---
title: "Feature engineering patterns"
topic: "Feature engineering patterns for supervised tabular and text models: encodings, time features, aggregate features, leakage control, and drift monitoring"
collection: "supervised-ml-and-feature-engineering"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
version: "2.0"
notation_style: "plain-text"
source_status: "OPEN"
prerequisites: ["Conditional probability and base rates", "Train/validation/test separation with a time cut-off", "The prediction contract from 02_Problem_Formulation_and_Losses.md"]
pattern_tags: ["target-encoding", "hashed-encoding", "cyclical-time", "window-aggregates", "leakage-audit", "out-of-fold-fit", "drift-monitoring"]
review_trigger: "A change to the feature set, the encoding statistic, the label horizon, or the population definition."
---

# Feature engineering patterns

> **Evidence:** every formula here is DERIVED and every arithmetic result is VERIFIED by the recomputation shown in place. The 48,000-row population, the five band names, and the service-cost figures are ILLUSTRATIVE constructions used to make the arithmetic concrete; they are not measurements of any organisation. Method attributions point at named published results in Section 10, and the underlying papers were not re-opened for this revision, so `source_status` stays OPEN. Nothing in this file asserts a company result, a leaderboard position, or a deployed improvement.

## 1. File Header

### 1.1 The decision this sheet changes

The recurring decision is not "which features are interesting". It is: **given a table, a label, a cut-off timestamp, and a serving constraint, which transformation of column `x` reaches the model, fitted on which rows, with which statistic frozen at which point in time, and what evidence would make you delete it.** Every candidate feature carries four hidden decisions — encoding, fitting population, update cadence, and fallback value — and a feature that is right on three of them and wrong on the fourth is worse than no feature, because it silently corrupts the evaluation of every other feature in the model.

This sheet exists so that those four decisions are made explicitly and audited mechanically. It assumes the reader has already read the loss and validation material in this unit, because feature quality is only measurable against a fixed loss and a fixed split.

### 1.2 What "pattern" means here, and what it does not

A pattern in this sheet is a *named, repeatable transformation together with its fitting protocol and its failure signature*. Three consequences follow, and each one is a rule used later:

1. The transformation alone is not the pattern. "Target encoding" as a lookup table is three lines of code; target encoding as a pattern is the table plus the out-of-fold fitting rule plus the smoothing prior plus the unseen-category fallback plus the drift check.
2. A pattern is selected by the *shape of the signal*, not by fashion. Cyclical time features exist because a periodic coordinate is discontinuous in its raw integer form; if the coordinate is not periodic, the pattern is inapplicable regardless of how many tutorials use it.
3. A pattern without a failure signature cannot be monitored, so it cannot be operated. If you cannot name the check that would catch the pattern going wrong in production, you have written a transformation, not a pattern.

### 1.3 Prerequisites and assumed contract

The sheet assumes four things are already fixed, because otherwise none of the numbers below mean anything:

- A **label definition** with an explicit event time and an explicit observation window. A label that says "converted" without saying within how many days of what is not a label.
- A **cut-off timestamp** `t_cut` such that every feature value for a row at `t_cut` uses only information with event time `<= t_cut`. Processing time is irrelevant to leakage; only event time is.
- A **population definition** with an inclusion rule and a denominator. The example population below is "policies that were in force on the first day of the observation month", not "all policies ever created".
- A **serving budget** expressed as latency per row, memory ceiling, and an update mechanism. A feature that requires a full-table recompute nightly is unavailable to a real-time scorer at 03:00 unless a snapshot was published.

### 1.4 Outcome and exclusions

Outcome: given the four inputs above, you can select an encoding from Section 7's table, derive its smoothing constant from data instead of guessing, bound its error, name its failure modes from Section 8, and price it from Section 9.

Exclusions: this sheet does not cover deep representation learning, embedding training objectives, image or audio preprocessing, or feature stores as infrastructure products. It does not establish that any particular vendor or organisation achieved any number. Where a number appears without a hand derivation, it is labelled ILLUSTRATIVE.

### 1.5 Reading order

Sections 2 and 3 establish the vocabulary; Section 4 derives the five results the rest of the sheet leans on (out-of-fold target encoding with its variance, empirical-Bayes smoothing constant, window-aggregate bias, cyclical embedding, and hashed-encoding collision rate); Section 5 re-derives all five on one small table; Section 6 gives the geometry; Sections 7 through 9 turn it into selection, failure, and cost decisions; Section 10 attributes the results; Section 11 tests recall, calculation, and diagnosis; Section 12 is the card you paste into a design review.

## 2. Concept Map / Mental Model

### 2.1 The four-stage pipeline

Every feature in a supervised pipeline passes through four stages, and a defect at any stage is invisible at the next one. That invisibility is the reason leakage survives so many reviews.

```text
  raw event log
      |  (1) AVAILABILITY: was the value knowable at t_cut?
      v
  point-in-time snapshot  ---> reject rows whose source arrived late
      |  (2) REPRESENTATION: numeric, categorical, cyclical, text, aggregate
      v
  model-matrix           ---> encoding fitted on train rows only
      |  (3) SCALE / SHAPE: monotone, bounded, distribution-matched
      v
  fitted estimator       ---> monotone constraints, regularisation
      |  (4) STABILITY: drift bounds, fallback values, refresh cadence
      v
  served score
```

Read the diagram as a filter chain. Stage 1 removes rows and columns; it does not transform values. Stage 2 is the only stage that changes the mathematical object. Stage 3 is where most silent accuracy loss occurs because a transformation can be numerically fine and statistically wrong. Stage 4 is what distinguishes a notebook from a service.

### 2.2 Three quantities that are constantly confused

Practitioners use the words "signal", "importance", and "stability" interchangeably in conversation, and then build monitoring that measures the wrong one. Keep them separate:

- **Signal** is a property of the joint distribution of `(x, y)` on the population, measured before any model exists. Mutual information, information value, and simple rate tables measure signal. Signal is model-free.
- **Importance** is a property of a fitted model plus an attribution method. A permutation importance of zero does not mean the feature has no signal; it means this model, at this depth, did not use it. Two collinear features routinely split importance so that each looks useless.
- **Stability** is a property of the distribution of `x` alone, over time or across segments. Population stability index measures stability and says nothing about signal.

The practical consequence: a feature can have high signal, zero importance, and perfect stability (a redundant copy of a better feature), or low signal, high importance, and poor stability (a tree latching onto an identifier that shifted). Each combination demands a different action, and Section 8's failure table is organised by which of the three went wrong.

### 2.3 The leakage mental model

Leakage is not "using the future". That description is too coarse to act on, because most leaking features do not obviously contain future values. The operational definition is:

> A feature leaks when its value for a training row was computed using information from rows whose event time is later than that row's `t_cut`, *or* using the row's own label.

The second clause is the one that catches practitioners. A target-encoded category computed on the full training table uses the row's own label to build its own feature value. That is leakage with no timestamp violation at all — every input is in the past — and it is the most common defect in tabular pipelines.

Two corollaries drive most of Section 4:

- **Leakage magnitude scales with row uniqueness.** If a category has one row, its target encoding is that row's label, and the encoded column reproduces the label exactly. If a category has 10,000 rows, one row's own label moves the encoding by 1/10,000 and the leak is negligible. Section 4.1 quantifies this precisely.
- **Leakage is invisible in validation only if the validation split was built the same way.** If train and validation are both encoded on the combined table, both look excellent. The corruption shows up as a train-to-production gap, not a validation gap, which is why it survives holdout review.

### 2.4 Where the sheet's five results sit

```text
  result                              answers                              section
  out-of-fold target encoding         how to encode a category without      4.1
                                      using the row's own label
  empirical-Bayes smoothing m*        how much shrinkage, derived not       4.2
                                      guessed
  window-aggregate bias               how a trailing window quietly         4.3
                                      becomes a label proxy
  cyclical embedding error bound      when sin/cos is worth two columns     4.4
  hash collision rate                 how many buckets before key          4.5
                                      collisions degrade the feature
```

Each result is derived from first principles below rather than quoted, so the constants in Section 5 can be checked by hand and so that a reader facing a non-standard version of the same problem can redo the algebra.

## 3. Foundational Definitions

Each definition below states meaning, units, shape, and valid range, because a symbol used without units is the origin of most unit errors in this unit's exercises.

**Row (unit of analysis).** One policy at one cut-off. Grain: `(policy_id, t_cut)` is unique. The model scores one row per scoring request; aggregates over rows are never used as features at scoring time unless they were precomputed as a snapshot.

**`t_cut` (cut-off timestamp).** The instant at which the row's inputs are frozen. Units: UTC timestamp with second resolution. Valid range: any instant inside the observation window. Everything after `t_cut` is unavailable, even if it is already stored in the warehouse when you run the query.

**`y` (label).** Binary indicator of the outcome inside the label window `(t_cut, t_cut + H]`. Units: dimensionless, values `{0,1}`. `H` is the label horizon in days; in the worked example `H = 90`.

**Base rate `p̄`.** The population proportion of `y = 1`. Units: proportion in `[0,1]`. In the example, `p̄ = 0.086`. Every smoothing constant below is expressed relative to `p̄`, because a prior that is not on the label's scale cannot be added to a rate.

**Category level `ℓ`.** A value of a categorical column, for example a risk band. Levels are nominal; the labels A through E carry no order and no distance. Cardinality `k` is the number of distinct levels observed in training, not the number that exists in the world.

**Raw rate `r̂_ℓ`.** `n_ℓ` rows and `c_ℓ` positives in level `ℓ` give `r̂_ℓ = c_ℓ / n_ℓ`. Units: proportion. Valid range `[0,1]`. Its standard error is `sqrt(r̂(1−r̂)/n_ℓ)`, so a level with 48 positives out of 2,400 has a standard error of about 0.0028 on a rate of 0.020 — a 14% relative error. That number is the reason smoothing exists.

**Smoothed (empirical-Bayes) rate `r̃_ℓ`.** `r̃_ℓ = (c_ℓ + m·p̄) / (n_ℓ + m)`, where `m` is a pseudo-count in rows. Units: proportion. As `m → 0` it equals the raw rate; as `m → ∞` it equals `p̄` for every level, which is a constant column and therefore useless. The choice of `m` is the whole game, and Section 4.2 derives it from between-level and within-level variance instead of using the folklore value of 10.

**Weight of evidence `WOE_ℓ`.** `WOE_ℓ = ln(P(ℓ | y=1) / P(ℓ | y=0))`. Units: natural-log odds, unbounded, sign indicates direction. Not a rate; it is a log ratio of two conditional distributions, and it is only defined when both conditional probabilities are positive.

**Information value `IV`.** `IV = Σ_ℓ (P(ℓ|y=1) − P(ℓ|y=0)) · WOE_ℓ`. Units: nats, non-negative. It is a likelihood-ratio-style divergence and it is *not* invariant to the number of levels: splitting a level in two always weakly increases `IV` even when nothing about the signal changed. Section 4.1's worked numbers show a 20-bin encoding scoring higher than the 5-bin encoding of the same variable purely from noise.

**Population stability index `PSI`.** `PSI = Σ_bins (a_i − b_i) · ln(a_i / b_i)`, where `a_i` and `b_i` are bin proportions in reference and current samples. Units: nats, non-negative, zero only when the distributions match exactly. Conventional operating thresholds — under 0.10 stable, 0.10 to 0.25 investigate, above 0.25 treat as shifted — are heuristics, not theory, and Section 8 explains how sample size breaks them.

**Out-of-fold (OOF) encoding.** A target encoding where the value assigned to a row is computed from rows in *other* folds only. Written `r̃_ℓ^{(-k(i))}`, where `k(i)` is the fold containing row `i`. Valid range: same as a rate. Its purpose is to make the encoding independent of `y_i` conditional on the level.

**Cyclical feature pair.** For a periodic coordinate `θ` with period `P`, the pair `(sin(2πθ/P), cos(2πθ/P))`. Units: dimensionless, each in `[−1, 1]`, and their sum of squares is exactly 1. The two columns jointly place `θ` on a circle; either alone folds the circle in half.

**Hashed encoding.** A map from a level string to bucket index `h(s) mod M`, then a one-hot or count vector of width `M`. `M` is the number of buckets, a power of two in most implementations. Collisions are levels mapping to the same bucket; the expected number is derived in Section 4.5.

**Window aggregate.** A feature summarising events in `(t_cut − W, t_cut]` for window width `W` in days, for example the count of claims. Units: count. The aggregate is a valid feature only if the window closes strictly at `t_cut`; a window that includes any event after `t_cut` silently imports the label window's contents.

**Out-of-fold fit and refit cadence.** Two distinct timestamps. The *fit* timestamp is when the encoding statistics were computed; the *refit* timestamp is when the estimator was retrained. A model refitted monthly on encodings computed quarterly has two different vintages inside it, and a drift alarm must say which vintage moved.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: out-of-fold target encoding and how much it leaks

**Preconditions.** A categorical column with levels `ℓ = 1..k`; `K` folds of the training rows; label `y`; a smoothing constant `m >= 0`; a prior `p̄` computed on training rows only. The folds are the unit of independence: a row's encoding uses all folds except its own.

**State transition.** Start from the naive encoding, `r̂_ℓ = c_ℓ/n_ℓ`, computed over the whole training table. Replace it with `r̃_ℓ^{(-k(i))} = (c_ℓ^{(-k(i))} + m·p̄) / (n_ℓ^{(-k(i))} + m)`, where `c^{(-k)}` and `n^{(-k)}` count positives and rows in level `ℓ` outside fold `k`.

**Why the naive version leaks.** Consider a level that contains exactly one row, and let that row be positive. Then `r̂_ℓ = 1/1 = 1`, so the encoded feature equals 1 exactly when the label is 1. The encoding *is* the label, rewritten. With `n_ℓ` rows, one of which is the row being scored, the encoding is

```text
LEAK MAGNITUDE (base case)
FORMULA
  r̂_ℓ = (y_i + Σ_{j≠i} y_j) / n_ℓ
  so      ∂r̂_ℓ/∂y_i = 1 / n_ℓ
WHERE
  y_i in {0,1} is the label of the row being scored
  n_ℓ is the number of training rows in level ℓ
CONSTRAINT
  m = 0 (no smoothing), level ℓ contains row i, all rows in the training table
NUMERICAL EXAMPLE
  n_ℓ = 1   -> ∂r̂/∂y_i = 1.000   encoding reproduces the label
  n_ℓ = 2   -> ∂r̂/∂y_i = 0.500   half the encoding is the label
  n_ℓ = 20  -> ∂r̂/∂y_i = 0.050
  n_ℓ = 100 -> ∂r̂/∂y_i = 0.010
INTERPRETATION
  the leak is not a small bias; at low cardinality-support it is the label itself,
  and gradient descent will find nothing else to use
```

With smoothing at `m = 40` and `p̄ = 0.086`, the same derivative becomes `1/(n_ℓ + m)`. At `n_ℓ = 1` that is `1/41 = 0.0244` instead of `1.0`, a 41-fold reduction; at `n_ℓ = 100` it is `1/140 = 0.00714`. Smoothing therefore reduces the leak but never removes it, which is why OOF fitting is mandatory rather than optional: smoothing is a variance treatment, not an independence treatment.

With OOF fitting the derivative with respect to `y_i` is exactly zero, because `y_i` is not in the statistic. That is the whole justification for the extra machinery, and it costs one thing: the encoding for a row is a random variable that differs across folds.

```text
OOF VARIANCE
FORMULA
  Var[r̃_ℓ^{(-k)}] = n_ℓ^{(-k)} · p_ℓ (1 − p_ℓ) / (n_ℓ^{(-k)} + m)^2
WHERE
  p_ℓ is the true positive rate of level ℓ
  n_ℓ^{(-k)} is the level's row count in the K−1 folds used for the fit
CONSTRAINT
  rows are independent within the level; m is fixed, not estimated per fold
NUMERICAL EXAMPLE
  level E: n = 2,400, p = 0.020, m = 40, K = 5 -> n^{(-k)} = 1,920
  Var = 1920 · 0.02 · 0.98 / (1920 + 40)^2
      = 37.632 / 3,841,600
      = 9.796e-6
  sd  = 0.00313, on an encoding of 0.0211 -> 14.8% relative
INTERPRETATION
  the rarest level carries a five-column-wide feature whose value moves by
  15% depending on which fold the row fell into, before any model error
```

The variance is the price of independence, and it is a *feature-level* price: it means that a rare level's encoding is noisy in a way that repeats identically at scoring time, because the served encoding is the single full-table fit. That asymmetry — noisy in training, frozen in serving — is why the smoothing constant must be chosen for the serving fit and the OOF variance used only to judge whether the training signal is real.

**Boundary where the derivation breaks.** Set `K = n_ℓ`, so every fold for that level contains one row. Then `n_ℓ^{(-k)} = n_ℓ − 1` and `m = 0` gives `r̃ = c_ℓ/(n_ℓ−1)`, which for a level with `c_ℓ = n_ℓ` (all positive) exceeds 1.0. The formula is arithmetically fine and statistically meaningless: the level's encoding under leave-one-out exceeds the range of a probability whenever the excluded row was positive. This is the classic leave-one-out failure and the reason `K` is capped (5 to 10 in practice) rather than set to the row count.

**Validity condition.** The derivation holds when folds are assigned independently of `y` — random or hash-based assignment, not sorted by label or by a proxy for label — and when the level counts are large enough that the binomial approximation to the count distribution is adequate (`n_ℓ · p_ℓ >= 5` and `n_ℓ · (1 − p_ℓ) >= 5`). Fold assignment correlated with the label resurrects the leak in a subtler form: if fold membership predicts `y`, the excluded-fold statistic carries information about the included row's label through the assignment itself.

### 4.2 General case: the empirical-Bayes smoothing constant

**Preconditions.** Level rates `r̂_ℓ` with counts `n_ℓ`; a prior mean `p̄`; the assumption that the true level rates `p_ℓ` are themselves drawn from a distribution with mean `p̄` and variance `τ²`, and that within a level the count is binomial.

**State transition.** Treat `r̂_ℓ` as a noisy measurement of `p_ℓ`. The best linear predictor of `p_ℓ` from `r̂_ℓ` is `p̄ + (τ²/(τ² + σ_ℓ²))·(r̂_ℓ − p̄)` with `σ_ℓ² = p_ℓ(1−p_ℓ)/n_ℓ`. The empirical-Bayes form `(c_ℓ + m p̄)/(n_ℓ + m)` is exactly this shrinkage with `m = σ²/τ²` evaluated at a pooled within-level variance.

**Derivation of `m`.** Write the total variance of the observed rates as between-plus-within:

```text
SMOOTHING CONSTANT
FORMULA
  s²_between = Σ_ℓ n_ℓ (r̂_ℓ − p̄)² / (N − 1)
  s²_within  = Σ_ℓ n_ℓ r̂_ℓ (1 − r̂_ℓ) / (N − k)
  m*         = s²_within / s²_between
WHERE
  N is the total training rows, k the number of levels
  s²_between estimates τ² and s²_within estimates the binomial noise σ²
  the ratio is a variance ratio, hence dimensionless, and it converts to a
  pseudo-count because σ² has units of rate² and τ² has units of rate² too,
  while the shrinkage weight τ²/(τ²+σ²) = n/(n+m) fixes m = σ² n/ (τ² n) ... 
CONSTRAINT
  requires k >= 3 levels and N >= 10k rows; equal n_ℓ is not required
NUMERICAL EXAMPLE
  levels A..E with n = 18,000 / 12,000 / 9,600 / 6,000 / 2,400
  rates      0.140 / 0.075 / 0.050 / 0.030 / 0.020
  N = 48,000, p̄ = 0.086
  s²_between = 0.001993     s²_within = 0.076619
  m* = 0.076619 / 0.001993 = 38.45
INTERPRETATION
  the folklore constant m = 10 is 3.8x smaller than the derived value; for the
  rarest band it yields 0.02027 instead of 0.02104, i.e. 3.6% closer to the raw
  0.02000 and therefore 3.6% less shrunk than the data justify
```

The line in which the units are resolved deserves care, because this is where hand-waving usually happens. The shrinkage weight in the empirical-Bayes form is `n/(n+m)`, and the optimal weight is `τ²/(τ²+σ²)`. Setting them equal at a representative level with `σ² = p̄(1−p̄)/n` gives

```text
  n/(n+m) = τ² / (τ² + p̄(1−p̄)/n)
  => m = p̄(1−p̄) / τ²
```

so `m` is a *pseudo-count* equal to the binomial variance at the prior divided by the between-level variance. Using the pooled within-level variance `s²_within` in place of `p̄(1−p̄)` generalises this to levels with different rates, which is what the `m* = s²_within/s²_between` form does; the two agree to within 3% on the example data. That is the sense in which `m` is dimensionless-but-count-like: it is a variance ratio that happens to be expressed in rows.

**Sensitivity.** Recompute `m*` after dropping the largest level, which is what happens when a segment is re-scoped:

```text
LEVELS B..E ONLY (A excluded)
  N = 30,000, positives = 1,608, p̄ = 0.0536
  s²_between = 0.000389   s²_within = 0.050345
  m* = 0.050345 / 0.000389 = 129.4
```

The constant moves from 38 to 129, a factor of 3.4, because the between-level spread collapsed once the dominant level left. A team that hard-codes `m = 40` and then removes a segment carries a constant fitted to a different population, and the rarest remaining level's encoding drifts toward its raw, noisy rate without any alarm. This is why `m` belongs in the model artifact's metadata rather than in a Python literal.

**Limits.** As `k → N` (each level one row), `s²_between` is dominated by binomial noise, `m* → 0`, and the formula correctly refuses to shrink — but the underlying problem is unsolvable, and no `m` will help. As `τ² → 0` (all levels identical), `m* → ∞` and every encoding is `p̄`; the feature is correctly identified as uninformative but is not automatically removed by this calculation. Both extremes are diagnostic rather than prescriptive.

**Validity condition.** The derivation assumes exchangeable levels with a common prior mean. If levels come from structurally different sub-populations — a band defined by a rule for one product and by a score for another — then a single `m` is a compromise, and the correct move is to fit `m` per sub-population or to abandon the encoding for that column.
