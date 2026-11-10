---
title: "Change-point detection"
topic: "Change-point detection: retrospective segmentation and online sequential detection"
collection: "unsupervised-ml-anomaly-detection-and-segmentation"
file_id: "03"
level: "core"
version: "2.1"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
review_trigger: "Change in sampling interval, penalty convention, autocorrelation structure, or online-vs-offline decision latency."
notation_style: "plain-text"
pattern_tags: ["changepoint", "pelt", "cusum", "bocpd", "penalised-segmentation"]
source_status: "OPEN"
owner_role: "Monitoring analytics reviewer"
source_index: "INDEX.md"
---

# Change-point detection

> **Evidence:** All formulas below are `DERIVED` from stated likelihood or stopping-time assumptions, and every arithmetic line is `VERIFIED` by an independent recomputation shown inline. Event counts, sampling cadences, reviewer costs and fleet sizes are `ILLUSTRATIVE` constructed values. Two `ARL0` figures for the EWMA design are explicitly `SIMULATED` from a seeded Monte Carlo and labelled as such, because no closed form is used. Bibliographic attributions are `SOURCED` by author and title only; no source body was opened. Any claim about a real deployment is `OPEN`.

## 1. File Header

**Purpose.** Decide whether a series actually changed regime, where the change sits, and when an alarm should fire — using an explicit false-alarm budget rather than an eyeball judgement about a chart.

**Prerequisites.** Expectation and variance of a sample mean; the normal and chi-square quantiles; the idea of a log-likelihood ratio; the idea of a stopping time as a random variable. You should be able to compute a pooled variance by hand and read a distribution function at a stated quantile without a library.

**Key outcome.** Given an ordered series, a noise scale, a sampling interval, and a false-alarm budget expressed per series per day, you can pick retrospective segmentation versus online monitoring, choose a penalty that keeps spurious segments bounded, compute the smallest shift visible at that length, compute expected detection delay and alarm load, and reject a "change point" that is really a sampling, schema, or autocorrelation artefact.

**This file does not cover.** Causal attribution of a regime change to a specific deploy or actor; forecasting after a change; the review workflow that consumes an alarm, which lives in `05_Human_Review_Loops.md`; and any measured production deployment, because none is sourced here.

**Level and status.** Core methods, with production diagnostics in Sections 8 and 9. Version 2.1, drafted 2026-11-10, not independently reviewed, review OPEN.

**Retrieval cue.** If a dashboard claims a change on Tuesday and nobody agrees it happened, read Section 8 first and then Section 4.1. If someone asks how small a change you can see, go to CPD-03 and answer with a number and a length.

| In scope here | Out of scope, handed elsewhere |
|---|---|
| Penalised segmentation of one ordered series | Multi-series causal attribution |
| Sequential detection with a stated alarm budget | Label-driven classification of windows |
| Penalty and threshold selection with stated units | Reviewer staffing policy |
| Locating and dating a boundary | Proving that a change was harmful |
| Diagnostics that separate artefact from regime | Regulatory or contractual interpretation |

| Symbol | Meaning | Units | Typical range |
|---|---|---|---|
| `n` | series length in observations | count | 20 to 10^6 |
| `sigma` | noise scale of the series | series units | positive |
| `delta` | size of a level shift | series units | any sign |
| `delta'` | shift in sigma units, `delta/sigma` | dimensionless | 0.2 to 4 |
| `s` | candidate split index | observation index | 1 to n-1 |
| `beta` | segmentation penalty | squared series units | 0 to n sigma^2 |
| `k`, `k'` | CUSUM reference, raw and standardised | series units, sigma units | 0.25 to 2 |
| `h`, `h'` | CUSUM threshold, raw and standardised | series units, sigma units | 3 to 8 |
| `ARL0` | expected observations to a false alarm | count | 10 to 10^7 |
| `H` | BOCPD prior mean regime length | count | 20 to 10^4 |
| `rho` | lag-1 autocorrelation | dimensionless | 0 to 0.99 |

Notation convention: one convention per file, plain text inside fenced blocks, `phi(x)` never used, and every symbol defined at first use in prose as well as in the table above. Quantities that are estimates carry a hat in formulas and lose it in prose once defined.

## 2. Concept Map / Mental Model

### 2.1 Two problems wearing one name

Retrospective segmentation and online detection look like one chapter in a textbook and behave like two different businesses in production. The first consumes the whole series and answers "how many regimes, and where". The second consumes one observation at a time and answers "alarm now or wait". Their objectives are formally different, so their tuning knobs are different, and no single number describes both.

```text
RETROSPECTIVE (offline)                     ONLINE (sequential)
---------------------------                 ---------------------------
whole series y_1..y_n is available          only y_1..y_t is available
choose 0 = t_0 < t_1 < ... < t_m = n        choose a stopping time T
objective: fit + beta * (number of          objective: E[T - tau | tau] small,
segments)                                   subject to ARL0 >= target
no notion of latency                        latency is the whole point
output: boundaries and segment statistics   output: an alarm time, or silence
failure: too many segments                  failure: too few alarms, or too late
budget: the penalty beta                    budget: the pair (k, h)
```

A team that tunes a penalty and then wires the same routine into a real-time loop has changed the problem without changing the code. That single mistake explains most "the detector is noisy" tickets, and Section 5 Step 12 puts a number on it.

### 2.2 Four things that can change, and why the choice follows

Precision pays here because the method follows from which quantity moved. A cost function is a set of assumptions about what is allowed to change.

| What changes | Formal statement | Typical cause | Cost that can see it |
|---|---|---|---|
| Level | `E[y_t]` steps by delta | deploy, pricing, policy | squared error on the mean |
| Scale | `Var[y_t]` steps | instrumentation, load, mixing | log-variance cost |
| Dependence or shape | autocovariance or family changes | batching, caching, retry storm | kernel or spectral cost |
| Trend | slope `b` changes | capacity exhaustion, gradual rollout | piecewise-linear cost |

A mean-only cost is blind to rows two and three by construction, because it standardises away exactly the information that changed. The symptom is a series that looks different to a human while the segmenter reports one regime, and worse, a variance jump inflates the estimated noise scale, which raises every threshold and hides the level shift as a side effect.

### 2.3 The information flow and the three leaks

```text
raw events --(bucket)--> series y_t --(detrend, scale)--> residual
    |                        |                              |
 event vs              cadence, missing                cost function
 processing time       buckets                          (mean, var, kernel)
    |                        |                              |
    +--> late arrivals       +--> bucket-edge artefact      v
         rewrite history          at every boundary     segmentation --> alarm --> review
                                                              |            |         |
                                                        penalty beta   threshold h   label, lag
```

All three leaks sit on the left of that diagram and none of them is fixed by a better detector. Event time versus processing time means the change appears when the system noticed events, not when they happened. Late arrivals rewrite buckets that a retrospective pass has already segmented. Bucket edges that straddle an event create a one-bucket dip that a mean-cost segmenter reads as two boundaries. Each leak produces a boundary at a suspiciously round clock time, which is the most useful single diagnostic in Section 8.

### 2.4 The invariant that ties the two problems together

Every method in this sheet is an instance of one comparison: the fit of a single model on a stretch of data against the fit of two models split at some point, measured in the same units as the noise, and compared with a threshold that grows with how many splits you were allowed to try.

```text
gain(s)  =  cost(one model on 1..n)  -  cost(two models split at s)
report a boundary at s  iff  max_s gain(s)  >  penalty(n, sigma, trials)
```

Everything that matters to an engineer is in the right-hand side. A constant penalty lets the spurious count grow without bound in `n`. A penalty proportional to `log n` keeps it bounded. A penalty chosen by looking at the data you then report on produces a segment count that cannot be validated on that same data, because the same observations were used twice.

### 2.5 Three separate budgets, usually conflated

Statistical budget: how often may the method fire when nothing changed. Latency budget: how long may it take to fire when something did. Capacity budget: how many alarms can a human process per hour. A design is feasible only when all three are satisfiable at once, and the usual failure is to optimise the first while ignoring the third.

```text
feasible iff   ARL0 >= (series_count * observations_per_day) / alarms_per_day_allowed
         and   delay_q90 <= detection_SLA
         and   alarms_per_day * review_minutes / 1440 <= reviewer_FTE
```

The three clauses are not independent: reducing delay requires reducing `h`, which reduces `ARL0`, which raises the alarm count, which breaks the third clause. Section 9 solves this trade in numbers for a 500-series fleet and finds it infeasible at a 30-minute SLA, which is a result, not a reason to try harder.

### 2.6 Aggregation is a modelling decision, not a data-cleaning step

Before any detector runs, someone chooses the bucket width: raw events, or 1-minute sums, or 5-minute means. That choice feels administrative and is actually the largest single determinant of what the detector can see, because it sets both `sigma` and `n`.

For white noise, aggregating `k` raw points into one bucket divides the noise scale by `sqrt(k)` and divides the number of points by `k`. Substituting into CPD-03, the smallest detectable shift becomes `sqrt(8 sigma^2 ln(n/k) / n)`, in which `k` appears only inside the logarithm. Aggregating therefore does not improve detectability at all in the main term — the signal averages down exactly as fast as the noise does — while it destroys time resolution, invites bucket-edge artefacts, and shortens the series enough to weaken the multiple-comparison penalty slightly.

| aggregation `k` | effective `n` | `delta_min` at `sigma = 6` raw | time resolution |
|---|---|---|---|
| 1 (raw) | 1440 | 1.206 | 5 minutes |
| 5 | 288 | 1.064 | 25 minutes |
| 15 | 96 | 0.955 | 75 minutes |

The 21% improvement from 15-fold aggregation is entirely the `ln(n/k)` term, and it is bought by a 15-fold loss of resolution. Anyone proposing to "reduce noise by aggregating" is proposing to trade localisation for a logarithmic crumb, and the honest framing is that aggregation is chosen for storage or for matching a decision cadence, never for statistical power.

The exception matters: aggregation *does* help when the noise is not white. If the high-frequency component is measurement jitter with variance `sigma_j^2` added to a slowly varying signal of variance `sigma_s^2`, then averaging `k` points attenuates the jitter by `k` and the signal not at all, so the effective noise falls faster than `1/sqrt(k)` until the signal's own variation dominates. That is a real gain, it is bounded, and it should be estimated from the autocorrelation function rather than assumed.

### 2.7 What "significant" means in each problem

In the retrospective problem, significance is a statement about a whole partition: the probability that the maximum gain over all splits exceeds the penalty under the null. In the online problem, significance is a statement about a stopping time: the expected number of observations until the first alarm under the null. They are not convertible. A segmentation whose per-pass spurious expectation is 0.05 is perfectly well calibrated, and re-running it 1440 times a day still yields about 72 spurious boundaries; that arithmetic is done in Section 4.4 and again in Step 12 of Section 5.

## 3. Foundational Definitions

**Change point.** An index `tau` in `1..n-1` such that the distribution of `y_1..y_tau` differs from that of `y_{tau+1}..y_n` in a stated respect. The definition is incomplete without the stated respect: level, scale, dependence, or full distribution. A change point is a property of the generating process and is never observed; only an estimate of it is.

**Regime.** The maximal contiguous block sharing one parameter value under the model. With `m` change points there are `m+1` regimes. "Regime" is the modelling word, "segment" is the output word, and they coincide only when the model is correctly specified.

**Segmentation.** A partition of `1..n` into contiguous non-empty blocks produced by minimising a penalised cost. Different from clustering, which ignores order, and from classification, which needs labels. A segmentation of a series with no change point still returns boundaries, which is why the null behaviour has to be characterised before any output is believed.

**Retrospective detection.** Estimation using the full series, where later observations can confirm or refute an earlier candidate. Latency is unbounded by construction. Used for audits, incident post-mortems, capacity history, and for producing the reference against which an online detector is calibrated.

**Online detection.** A stopping time `T` adapted to the history generated by `y_1..y_t`, so the decision at `t` uses only data up to `t`. The two error types are asymmetric in cost and are therefore parameterised separately, which is why online detection reports the pair `(ARL0, delay)` instead of one accuracy number.

**ARL0.** Expected number of observations until a false alarm under the null. It is a mean, not a rate, and it does not fix the distribution: two designs with the same mean can differ by an order of magnitude in the tail. Convert with `rate = 1/ARL0` alarms per observation, then multiply by the cadence to obtain alarms per hour.

**ARL1.** Expected delay from the true change point to the alarm, under a stated shift. It means nothing without the shift attached, because delay is a decreasing function of shift size and the function is steep.

**Detection delay.** The realised value `T - tau`, whose mean is `ARL1` under the design shift. Report a high quantile as well, because the operationally painful case is the 90th percentile rather than the mean.

**Reference value `k`.** The CUSUM allowance, `delta_target/2` in series units or `delta'_target/2` in sigma units. A design choice about which shift you care about, not something estimated from data. Setting it equal to the target shift rather than half of it makes the expected drift zero, which is the classic tuning error.

**Threshold `h`.** The alarm level for the accumulated statistic, in `k`-times-observations units. It is the only knob that should be used to meet a false-alarm budget, because it trades directly against `ARL0` and only indirectly against delay.

**CUSUM statistic.** One-sided upward form `S_t = max(0, S_{t-1} + (y_t - mu_0) - k)`, with a mirrored downward form. The reset to zero is what makes the statistic remember only the current excursion, and it is also what gives the procedure a finite `ARL0`. Removing the reset leaves a cumulative sum, which is not a test.

**Penalised cost.** The segmentation objective `sum over segments of cost(segment) plus beta times the number of segments`, with `beta` in the units of the cost. With squared error and variance `sigma^2`, `beta = 2 sigma^2 log n` is the conservative default and `beta = sigma^2 log n` is the one-parameter BIC variant. The factor of two is a convention difference, not an error, and it changes the reported segment count.

**Segment cost.** For a Gaussian mean model, `C(a..b) = (b-a+1) log(2 pi sigma^2) + SSE(a..b)/sigma^2`. The first term is linear in block length, so it telescopes across a partition and cancels when `sigma` is fixed. When `sigma` is estimated per segment it does not cancel, and the reported count changes.

**Optimal partitioning.** The dynamic program `F(t) = min over s<t of F(s) + C(s+1..t) + beta` with `F(0) = -beta`. Exact for additive costs, quadratic time, linear memory. This is the reference implementation against which every faster method is checked on short series.

**PELT.** The same recursion with a pruning rule. If a candidate `s` is already worse than the current optimum at time `t` by more than the pruning constant, and the cost is superadditive, then `s` can never be optimal at any later time and is discarded. Average linear, worst-case quadratic, exact only under the superadditivity condition.

**Binary segmentation.** Recursively apply a single-split test to the whole series and then to each half. Fast and easy to reason about, but greedy: a large change splits the series such that two smaller changes fall inside one half, where their combined contrast must clear a penalty calibrated for the full length.

**BOCPD.** Bayesian online change-point detection. Maintains a posterior over the run length `r_t`, the number of observations since the last change, updating by either growing the run or resetting it. The output is a distribution over change times rather than a point estimate, which is its main advantage and the reason its probability cannot be compared across different priors.

**Hazard rate.** The prior probability per observation that a change happens now, given none has happened yet, usually written `1/H`. `H` is the prior mean regime length in observations. It is the most consequential prior in the model and is usually set by folk wisdom rather than by measured regime lengths.

**Run length.** The number of observations since the most recent change point, as a random variable under the posterior. Truncating its support is the standard approximation that makes BOCPD practical, and the truncation costs detection delay for regimes longer than the truncation point.

**Penalty convention.** Whether `beta` includes the factor of two. Two analysts using "the BIC penalty" on the same data can report different segment counts and both be internally consistent, so any reported count must name the convention. This file uses `2 sigma^2 log n` and says so.

**Penalty scaling.** The growth of `beta` in `n`. Constant penalties let spurious counts grow with series length; `log n` penalties keep them bounded. The growth rate is what licenses a claim that the segment count is stable under a longer replay.

**Segment resolution.** The shortest segment the method can report, set by the per-segment parameter count and the penalty. Stating a result without its resolution is stating a claim that cannot be falsified, because a method that cannot report a 3-point segment has not tested for one.

**Minimum segment length.** An explicit floor on block length, used to protect against edge artefacts and to encode a physical prior such as "a regime lasts at least an hour". It changes the estimator's null behaviour and must be stated with the penalty.

**Localisation error.** The distance between the estimated and the true change index. Its scale is proportional to `sigma^2/delta^2`, so small shifts are located imprecisely even when detected, and reporting a boundary to the nearest observation is a precision claim that the data usually cannot support.

**Power.** The probability of reporting a change that exists. At the smallest detectable shift from CPD-03 the power is near 0.5 by construction, so a "detectable" shift is a coin flip and should not be described as reliably detected.

**Spurious change point.** A reported boundary at which no parameter changed, so the count is too high. Distinct from a missed change (power failure) and from a misplaced one (localisation error). The three have different diagnostics, and conflating them wastes most incident time.

**Event time and processing time.** Event time is when the observation occurred; processing time is when the pipeline observed it. A processing-time boundary that precedes the event-time boundary by a fixed offset is a pipeline artefact, and the offset is the diagnostic.

**Multiple comparisons in segmentation.** Testing every candidate split at level alpha gives at most `(n-1) alpha` family-wise error by the union bound. That is the whole reason a constant threshold fails and a growing penalty is needed; the bound is loose but adequate for choosing a growth rate.

**Offline-online consistency.** A property of a deployed system in which an online alarm is recoverable by retrospective analysis of the same history. When a month of retrospective analysis shows no boundary where the online detector fired twenty times, the online detector is measuring cadence, bucket edges, or late data rather than the process.

**In-control state.** The condition in which the series is generated by the reference model with no change point. Every false-alarm statement is conditional on this state being identifiable, which requires a stretch of history over which the model was plausibly true. A baseline estimated over a period that already contained a change is not an in-control baseline, and every threshold derived from it is wrong in the direction that hides the change.

**Drift.** The expected increment of a sequential statistic per observation. For the CUSUM with reference `k`, the drift is `-k` in control and `delta - k` after a shift of size `delta`, both in series units per observation. Drift is the quantity that determines how fast evidence accumulates, which is why `k = delta` makes the statistic a martingale and the detector useless for that shift.

**Reference window.** The stretch of history from which `mu_0`, `sigma`, and any seasonal terms are estimated, with an explicit start and end timestamp. It must precede the monitoring period and must be short enough to be stationary and long enough to estimate `sigma` to a usable precision. Reporting a threshold without its reference window is reporting a number with no provenance.

**Censoring at the series boundary.** The fact that a change occurring before the start of the observed window, or after its end, cannot be located and will be reported as a boundary at the edge or not at all. Censored boundaries must be labelled as censored, because a boundary at index 1 is almost always the estimator saying "the change is at or before here" rather than "the change is here".

**Detection floor.** The smallest shift that a stated method, series length, noise scale and false-alarm budget can support. It is the answer to "how small a change can we see", it is computed from CPD-03 rather than measured, and it should be quoted before any claim that a system detects anomalies in a series.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: one mean shift, known variance, fixed split

Take `y_1..y_n` independent `N(mu, sigma^2)` up to index `s` and `N(mu + delta, sigma^2)` after, with `sigma` known and `delta` free. This base case exists so that every later formula can be read as a perturbation of one number.

```text
Model: y_t = mu + delta * 1{t > s} + eps_t,  eps_t ~ N(0, sigma^2) iid
Log-likelihood ratio with delta profiled out:
    Lambda(s) = [ SSE_0 - SSE(s) ] / (2 sigma^2)
Identity: SSE_0 - SSE(s) = (n1 n2 / n) * (ybar_1 - ybar_2)^2

FORMULA [CPD-01]: G(s) = (n1*n2/n) * (ybar_1 - ybar_2)^2
WHERE: ybar_1, ybar_2 segment means [series units]; n1 = s and n2 = n - s
       [observations]; G in squared series units.
CONSTRAINT: independent observations; fixed split tested once; sigma known
       or estimated from within-segment residuals rather than from the whole.
NUMERICAL EXAMPLE: s = 12, n = 24 => n1 n2/n = 144/24 = 6.0
       ybar_1 = 100.5, ybar_2 = 110.0 => difference 9.5, squared 90.25
       G(12) = 6.0 * 90.25 = 541.5
INTERPRETATION: 541.5 squared units of fit improvement, bought at the price
       of one extra parameter. Compare it with the residual it must beat,
       never with zero.
```

Derive the identity once and it stops being mysterious. The single-model residual is the sum of squared deviations from the grand mean; the two-model residual is the sum of squared deviations from each segment mean. Their difference is the between-segment sum of squares, `n1 (ybar_1 - ybar)^2 + n2 (ybar_2 - ybar)^2`, and substituting the grand mean `(n1 ybar_1 + n2 ybar_2)/n` collapses it to the form in CPD-01. This is the one-way ANOVA decomposition with groups defined by a split index.

Three consequences worth carrying without re-deriving. The gain is invariant to adding a constant to the series, so de-meaning changes nothing. The gain is not invariant to scaling, which is why every threshold must be expressed through `sigma`. And the gain is symmetric under reversal, so `G(s)` for a reversed series equals `G(n-s)`, which is a cheap unit test for an implementation.

The coefficient `n1 n2/n` is maximised at the balanced split, where it equals `n/4`. Edge splits retain up to `(n-1)/n` of that coefficient, so a single extreme value at the edge can still dominate a modest balanced shift. Section 4.4 turns that observation into the counterexample that breaks naive segmenters.

**Threshold from the null maximum.** Under the null, `G(s)/sigma^2` is chi-square with one degree of freedom for a fixed `s`, because it is a squared projection onto one contrast direction. Maximising over `n-1` splits inflates that, and the inflation grows logarithmically.

```text
FORMULA [CPD-02]: beta(n)     = 2 * sigma^2 * ln(n)   (conservative default)
                  beta_BIC(n) =     sigma^2 * ln(n)   (one-parameter BIC)

WHERE: sigma^2 noise variance [squared series units]; n length [count];
       beta penalty in the same units as the segment cost.
CONSTRAINT: light-tailed independent noise; n large enough for the
       extreme-value approximation to be sane, roughly n >= 20.
NUMERICAL EXAMPLE: sigma = 4, n = 24, ln 24 = 3.1781
       beta     = 2 * 16 * 3.1781 = 101.6977
       beta_BIC =     16 * 3.1781 =  50.8489
INTERPRETATION: 50 to 102 squared units is the entry price for reporting one
       boundary at this length; a gain of 541.5 pays it five to ten times over.
```

**Smallest detectable shift.** Setting the balanced gain equal to the penalty and solving for the shift converts a statistical threshold into an engineering specification: how big a change this series length can support.

```text
Balanced gain: G = (n/4) delta^2 ; set equal to 2 sigma^2 ln n
    => delta_min = sqrt(8 sigma^2 ln n / n)

FORMULA [CPD-03]: delta_min = sqrt(8 * sigma^2 * ln(n) / n)
WHERE: delta_min smallest shift [series units]; sigma noise scale; n length.
CONSTRAINT: balanced split, iid Gaussian, one change, conservative penalty;
       power at delta_min is close to 0.5, so "detectable" means even odds.
NUMERICAL EXAMPLE: sigma = 4, n = 24
       delta_min = sqrt(8*16*3.1781/24) = sqrt(406.80/24) = sqrt(16.9496)
                 = 4.1171
       observed shift 9.5 => 9.5/4.1171 = 2.307 times the floor
INTERPRETATION: with 24 points and noise scale 4, the honest answer to "how
       small a change can you see" is about 4 units.
```

The floor scales in a way that surprises people. It falls only as `sqrt(log n / n)`, so quadrupling the series length from 360 to 1440 cuts it by a factor of `sqrt(4 * 7.27/5.89) = 2.22`, not by four. A useful table, all recomputed at `sigma = 6`:

| `n` observations | `ln n` | `delta_min` (sigma units) | at 5-min cadence, span covered |
|---|---|---|---|
| 24 | 3.1781 | 1.029 | 2 hours |
| 144 | 4.9698 | 0.526 | 12 hours |
| 1440 | 7.2724 | 0.201 | 5 days |
| 14400 | 9.5750 | 0.073 | 50 days |

Read the table as a budget statement: detecting a fifth of a sigma from a single 5-minute series requires five days of data, so a same-day claim about a 0.2-sigma shift on one series is arithmetically impossible regardless of method. The only escapes are pooling series, using a covariate to reduce `sigma`, or accepting a larger target shift.

**Boundary and counterexample.** Take the edge split `s = 1` on the same 24-point series. The coefficient is `1*23/24 = 0.9583`, the first value is 99, and the mean of the rest is `(2526 - 99)/23 = 2427/23 = 105.5217`, so the gain is `0.9583 * 6.5217^2 = 0.9583 * 42.5326 = 40.76`. Now suppose one field lost a decimal and the first value is 9.9 instead of 99: the difference becomes `-95.6217`, and the gain becomes `0.9583 * 9143.5 = 8762`. The formula is answering correctly and the inference is nonsense, because it has detected a data-entry error rather than a regime. Every edge-split pathology has this shape, and no penalty can suppress it, since the evidence really is in the data.

**A second limit of the base case.** The single-split gain says nothing about the shape of the change. A step that persists and a pulse that reverts after ten observations both raise `G(s)` at the same index, and the pulse raises it more when the pulse amplitude is larger. Distinguishing them requires fitting both a two-segment and a three-segment model and comparing penalised costs, which is why the general case exists.

### 4.2 General case: penalised segmentation

Given a block cost computable in constant time after linear precomputation, choose the number and location of breakpoints to minimise total cost plus a per-breakpoint penalty.

```text
FORMULA [CPD-04]: minimise over 0 = t_0 < t_1 < ... < t_m < t_{m+1} = n
                  sum_{j=0..m} C(t_j + 1 .. t_{j+1}) + beta * m
WHERE: C block cost [squared units or log-likelihood units]; m number of
       change points [count]; beta penalty in the same units as C.
CONSTRAINT: additive cost; contiguous non-empty blocks; m unknown a priori.
NUMERICAL EXAMPLE: sigma = 4 (sigma^2 = 16), n = 24, beta = 101.6977
       m = 0: C(1..24) = 730.5/16 = 45.656 ; total 45.656
       m = 1 at 12: 189.0/16 + 101.6977 = 11.8125 + 101.6977 = 113.510
       decision: 45.656 < 113.510, so report m = 0
INTERPRETATION: at the conservative penalty this series is NOT segmented,
       even though the raw gain is 541.5. The constant term cancels, which
       is why the comparison is 45.656 against 113.510.
```

That example is the most instructive arithmetic in the sheet because it contradicts intuition. With `sigma = 4` the within-segment residual per degree of freedom is `189.0/22 = 8.59`, which is well below the assumed `16`. The penalty is therefore calibrated to a noise level larger than the data's, and it refuses a change that a test at the empirical scale would accept. Re-estimate `sigma` at `sqrt(8.59) = 2.93` and the penalty falls to `2*8.59*3.1781 = 54.61`, at which point the split wins 76.61 against a null of 85.04. The conclusion flips on a variance assumption, which is why "the algorithm found two segments" is never a result on its own.

**Exact dynamic programming.** The recursion is three lines and it is the ground truth for every faster method.

```text
F(0) = -beta
F(t) = min over s in 0..t-1 of [ F(s) + C(s+1..t) + beta ]
backtrack from t = n through the argmin chain to recover the partition

FORMULA [CPD-05]: F(t) = min_{s<t} [ F(s) + C(s+1..t) + beta ]
WHERE: F(t) optimal penalised cost of the prefix 1..t [cost units];
       s candidate last breakpoint before t; the -beta offset at F(0) makes
       the m = 0 case cost exactly C(1..n).
CONSTRAINT: additive cost; ties broken consistently; memory O(n) for
       backtracking pointers.
NUMERICAL EXAMPLE (a four-point check that catches off-by-one errors):
       y = 1, 2, 10, 11 ; sigma^2 = 1 ; beta = 6
       C(1..4) = 82.0 ; C(1..2) + C(3..4) + 6 = 0.5 + 0.5 + 6 = 7.0
       F(0) = -6, F(1) = 0.0, F(2) = 0.5, F(3) = 6.5, F(4) = 7.0
INTERPRETATION: F(4) = 7.0 recovers the split at 2, and the intermediate
       values are the audit trail. Any implementation that returns a
       partition but not F(t) cannot be checked on this example.
```

**PELT pruning.** The same optimum, with candidates removed once they can no longer win.

```text
Prune candidate s at time t when  F(s) + C(s+1..t) > F(t) - K
Condition for correctness:  C(a..t) + C(t+1..u) + K <= C(a..u)  for all a<t<u

FORMULA [CPD-06]: keep only candidates s with F(s) + C(s+1..t) <= F(t)
WHERE: K pruning constant, 0 in the standard form; the inequality is
       checked at every t and pruned candidates are never revisited.
CONSTRAINT: superadditive cost. Squared error and fixed-parameter Gaussian
       negative log-likelihood satisfy it; some robust and rank-based costs
       do not, and pruning then silently loses the optimum.
NUMERICAL EXAMPLE: at t = 24 the optimum is F(24) = 113.51 and candidate
       s = 3 gives F(3) + C(4..24) = 128.40. Since 128.40 > 113.51, s = 3 is
       removed and will not be evaluated at t = 25 or later.
INTERPRETATION: pruning is data-dependent, so average cost is near linear on
       series with few changes and near quadratic on series that change
       constantly. Measure the candidate-set size in production.
```

**CUSUM: the false-alarm budget comes first.** In the online problem the primary object is the mean time between false alarms, not a fit criterion. In standardised units the Siegmund-type approximation below is accurate to a few percent for moderate thresholds and degrades at the extremes.

```text
Standardise y' = (y - mu_0)/sigma ; shift delta' = delta/sigma
Reference k' = delta'_target/2 ; statistic S_t = max(0, S_{t-1} + y'_t - k')
Alarm when S_t > h'

FORMULA [CPD-07]: ARL0 ~ [ exp(2 k' h') - 2 k' h' - 1 ] / (2 k'^2)
WHERE: k' reference in sigma units; h' threshold in sigma units of
       accumulated standardised deviation; ARL0 in observations.
CONSTRAINT: iid Gaussian noise; one-sided test; k' in the range 0.25 to 2.
NUMERICAL EXAMPLE: k' = 0.5 (tuned to a 1-sigma shift), h' = 5
       exp(2*0.5*5) = exp(5) = 148.4132
       numerator = 148.4132 - 5 - 1 = 142.4132 ; denominator 2*0.25 = 0.5
       ARL0 = 284.83 observations = 1424.1 min = 23.74 h at 5-min cadence
INTERPRETATION: roughly one false alarm per series per day. Across 500
       series that is 21 alarms per hour before any real change exists.
```

That last sentence is the whole design discussion in one number, so it is worth tabulating across thresholds. Every row below is recomputed from CPD-07 with `k' = 0.5`, a 5-minute cadence, 500 series and 20 minutes per review.

| `h'` | `ARL0` observations | `ARL0` in hours | alarms/hour, 500 series | reviewer-hours/hour |
|---|---|---|---|---|
| 3 | 32.17 | 2.68 | 186.5 | 62.2 |
| 4 | 99.20 | 8.27 | 60.49 | 20.16 |
| 5 | 284.83 | 23.74 | 21.07 | 7.02 |
| 6 | 792.86 | 66.07 | 7.57 | 2.52 |
| 7 | 2177.27 | 181.44 | 2.76 | 0.92 |

The reviewer column is `alarms * 20/60` and it is the one that kills designs. At `h' = 5` a 500-series fleet consumes seven full-time reviewers on false positives alone, before a single true change is investigated. Raising `h'` to 6 cuts that to 2.5 hours, which is the kind of trade a capacity plan can absorb.

**Detection delay from the same two constants.** The companion number, and the one the SLA is written against.

```text
FORMULA [CPD-08]: ARL1 ~ h' / (delta' - k')      for delta' > k'
                  delay_min = ARL1 * cadence_min

WHERE: delta' true shift in sigma units; k' reference; h' threshold;
       cadence in minutes.
CONSTRAINT: delta' > k' strictly; the approximation is first-order and
       optimistic as delta' - k' shrinks below about 0.5 sigma.
NUMERICAL EXAMPLE: k' = 0.5, h' = 5, delta' = 1.0
       ARL1 = 5/(1.0 - 0.5) = 10 observations = 50 minutes
       with sigma = 6 series units the shift is 6.0 units
INTERPRETATION: a 6-unit step is flagged about 50 minutes after onset on
       average; report the 90th percentile from simulation, typically 1.5x
       to 2x the mean at this margin.
```

**BOCPD: the run-length posterior.** The Bayesian alternative replaces a threshold with a prior on regime length and returns a distribution over the change time. The recursion is one line of conditional probability plus a hazard term.

```text
Growth: p(r_t = r+1 | y_1..y_t) ∝ p(r_{t-1} = r) * (1 - 1/H) * L_grow(r)
Reset:  p(r_t = 0     | y_1..y_t) ∝ [ sum_r p(r_{t-1}=r) * (1/H) ] * L_new

FORMULA [CPD-09]: w(r) ∝ (1 - 1/H) * L_grow(r)   for r >= 1
                  w(0) ∝ (1/H) * L_new
WHERE: H prior mean regime length [observations]; L_grow, L_new predictive
       likelihoods of y_t under the continued and the fresh regime.
CONSTRAINT: constant hazard, i.e. geometric regime length; conjugate or
       approximated predictive; O(t) work per step, O(t^2) in total.
NUMERICAL EXAMPLE: H = 144 so 1/H = 0.006944
       grow prior 0.993056, reset prior 0.006944
       with L_grow = 0.35 and L_new = 0.65 (relative densities)
       grow ∝ 0.993056*0.35 = 0.347569 ; reset ∝ 0.006944*0.65 = 0.004514
       normalised reset probability = 0.004514/0.352084 = 0.01282
INTERPRETATION: even with the new-regime likelihood nearly twice the old
       one, the posterior probability of a change is 1.3%. The prior, not
       the data, dominates at short run lengths.
```

The practical consequence is that a BOCPD change probability is not comparable across values of `H`. Reporting it without `H` is reporting a number with no interpretation, the same failure as reporting an unsupervised score without its model version. Thresholding it at 0.5 is a third error: at `H = 144` a 0.5 posterior requires a likelihood ratio near 140, so the detector would essentially never fire.

**Kernel cost for distributional change.** When the change is not a mean shift, the mean cost is the wrong instrument.

```text
MMD^2(P,Q) = E k(x,x') - 2 E k(x,y) + E k(y,y')
Block cost  C(a..b) = - MMD^2( y_a..y_c , y_{c+1}..y_b ), c the midpoint

FORMULA [CPD-10]: MMD_hat^2 = 1/m^2 sum k(x_i,x_j) - 2/(mn) sum k(x_i,y_j)
                              + 1/n^2 sum k(y_i,y_j)
WHERE: k kernel with bandwidth ell in series units; m, n block sizes.
CONSTRAINT: iid within block; bandwidth chosen on a held-out stretch;
       naive cost O((m+n)^2) per block, random Fourier features for long
       blocks; the unbiased estimator can return small negative values.
NUMERICAL EXAMPLE: ell = 5, m = n = 12, within-block kernel means 0.85 and
       0.80, cross mean 0.42: MMD^2 = 0.85 - 0.84 + 0.80 = 0.81. The
       threshold comes from a permutation null, not from this value.
INTERPRETATION: MMD responds to scale and shape simultaneously. For a pure
       level change it has lower power than a targeted mean test, because it
       spends variance on directions that did not move.
```

### 4.3 Variants

**EWMA.** A geometric moving average with a control limit, cheaper than CUSUM and better than a fixed threshold at catching slow drift. Two constants matter: the smoothing weight and the limit multiplier.

```text
z_t = lambda * y_t + (1 - lambda) * z_{t-1},  z_0 = mu_0
Steady-state sd:  sigma_z = sigma * sqrt( lambda / (2 - lambda) )
Control limit:    |z_t - mu_0| > L * sigma_z

FORMULA [CPD-11]: sigma_z = sigma * sqrt(lambda/(2-lambda)) ; limit = L sigma_z
WHERE: lambda smoothing weight in (0,1]; L limit multiplier, typically 2.7
       to 3.0; sigma_z sd of the smoothed statistic [series units].
CONSTRAINT: iid noise; z_0 initialised at the target mean, otherwise the
       first few hundred points carry start-up bias.
NUMERICAL EXAMPLE: sigma = 4, lambda = 0.1
       sigma_z = 4 * sqrt(0.1/1.9) = 4 * 0.22942 = 0.91767
       L = 3 gives a limit of 2.753 units on the smoothed statistic
       equivalent window length 2/lambda - 1 = 19 observations
INTERPRETATION: the smoothed statistic is 4.36x less variable than a single
       observation, so a 2.75-unit limit is a 3-sigma rule on the average of
       roughly 19 points. The EWMA is a one-sided low-pass filter with a
       tuned gain, not a change-point estimator.
```

The `ARL0` of an EWMA has no simple closed form. A seeded Monte Carlo with 8000 replications of `100000` maximum observations, at `sigma = 4`, gives `841 +/- 18` observations for `lambda = 0.1, L = 3.0`, and `559 +/- 12` for `lambda = 0.2, L = 3.0`. Those two figures are `SIMULATED`, not derived, and they should be quoted with their Monte Carlo error; a design that relies on a simulated `ARL0` must publish the replication count.

**Variance change.** Replace the mean cost with a Gaussian negative log-likelihood with free variance: `C(a..b) = (b-a+1) log(SSE(a..b)/(b-a+1))`. The penalty must then pay for two parameters per segment, and the null distribution is no longer a single chi-square, because the variance ratio has a point mass at one under the null. This is the most common source of "significant variance change" findings that fail to replicate.

**Slope change.** Fit a line per segment and use the residual sum of squares. For equal halves of length `L` with a common boundary and slopes differing by `Delta_b`, the gain is `Delta_b^2 * S / 2` where `S = L(L^2 - 1)/12` is the within-segment design sum of squares.

```text
FORMULA [CPD-12]: G_slope = (S1 S2 / (S1 + S2)) * (b1 - b2)^2
                  S_i = L_i (L_i^2 - 1) / 12

WHERE: b_i segment slopes [series units per observation]; S_i within-block
       design sums of squares [observation^2]; G_slope [squared units].
CONSTRAINT: two straight lines with a shared boundary and independent
       slopes; noise iid; blocks long enough that the slope is identified.
NUMERICAL EXAMPLE: L = 24, b1 = 1.0, b2 = 2.0 per observation
       S = 24*(576 - 1)/12 = 1150 ; S1 = S2 = 1150
       coefficient = 1150*1150/2300 = 575 ; gain = 575 * 1.0^2 = 575.0
       measured on a constructed series: 575.75, 0.13% higher because the
       two blocks are not identically centred around a shared origin
INTERPRETATION: gain scales with L^3 for slope changes against L for level
       changes, so long blocks are disproportionately powerful for slopes
       and a 6-point block cannot identify a slope at any noise level.
```

The `L^3` scaling is worth verifying numerically because it drives design decisions. On constructed series with slopes 1.0 and 2.0: `L = 4` gives gain 2.619, `L = 6` gives 8.934, `L = 12` gives 71.87 and `L = 24` gives 575.75. Dividing by `L^3` gives 0.0409, 0.0414, 0.0416 and 0.0416, a constant to within 2%, which is the empirical form of CPD-12.

**Multivariate cost.** For `d`-dimensional observations, `C(a..b) = (b-a+1) log det Sigma_hat(a..b)`. The penalty must scale with `d`, because each boundary introduces order `d^2` covariance parameters. Using a univariate penalty on a 50-dimensional series produces dozens of boundaries, every one of them located at a spike in a single coordinate.

**Robust cost.** Substitute a Huber or quantile loss for squared error. The advantage is bounded influence from single observations, which removes the edge pathology of Section 4.4; the cost is that the superadditivity condition behind PELT may fail, so pruning must be dropped or verified against exact DP on a subsample.

**Two-stage coarse then fine.** Run a cheap detector on a coarse aggregation to find candidate regions, then run exact segmentation on a narrow window around each candidate. This yields near-linear behaviour on very long series at the price of missing changes that vanish under aggregation, since a 20-minute spike disappears in a daily mean.

### 4.4 Limits and counterexamples

**Limit 1: the null maximum grows with the series, so a fixed threshold must eventually fail.** The null maximum gain grows like `2 sigma^2 log n`. At `sigma = 4` the value is `2*16*3.1781 = 101.7` for `n = 24` and `2*16*7.2724 = 232.7` for `n = 1440`, a rise of 131 squared units purely from lengthening the same series sixty-fold. A penalty copied from a pilot at 24 points will report boundaries in every production replay, and the boundaries will look plausible.

**Limit 2: autocorrelation invalidates every threshold in this sheet.** All `ARL0` and penalty expressions assume independent observations. Under AR(1) dependence with correlation `rho`, the variance of a segment mean inflates by `(1+rho)/(1-rho)`, so the effective length is `n_eff = n (1-rho)/(1+rho)`. At `rho = 0.9` and `n = 1440`, `n_eff = 1440 * 0.1/1.9 = 75.79`, and the standard error inflates by `sqrt(1440/75.79) = 4.36`. A threshold computed as if `n = 1440` is 4.36 times too permissive, and the smallest detectable shift rises from `1.206` to `4.055` sigma units.

**Limit 3: the confidently wrong answer.** A single dropped decimal at index 1 yields a gain of 8762 and a segmentation reporting "regime change at the start of the series". Every downstream statistic then compares a mean computed from one bad value against the rest. No penalty can fix this, because the evidence is real under the model; only a range check or a robust cost can.

**Limit 4: online and offline thresholds are not interchangeable.** A penalty controls the expected number of spurious boundaries in one pass. A threshold controls the expected time to a false alarm. Running a segmentation on a trailing window every minute performs 1440 passes per day, each entitled to its own spurious boundary, so the daily count is at least an order of magnitude above the per-pass expectation even after accounting for overlap.

**Limit 5: no method is uniformly good over all configurations.** Localisation error scales like `sigma^2/delta^2`, so arbitrarily small shifts are located arbitrarily imprecisely, and detection delay for small shifts is unbounded. There is no finite-sample procedure that avoids this, so any claim of detecting "any change" is a claim about the shifts that have been seen.

## 5. Worked Numerical Example

**Setup (ILLUSTRATIVE).** Twenty-four consecutive 5-minute observations of a latency index in milliseconds above a reference. Indices 1 to 12 come from a stable regime, 13 to 24 from a shifted one. The series is constructed for teaching; nothing here is measured on a real system.

| idx | y | idx | y | idx | y | idx | y |
|---|---|---|---|---|---|---|---|
| 1 | 99 | 7 | 102 | 13 | 111 | 19 | 115 |
| 2 | 104 | 8 | 100 | 14 | 108 | 20 | 107 |
| 3 | 101 | 9 | 97 | 15 | 114 | 21 | 110 |
| 4 | 96 | 10 | 105 | 16 | 106 | 22 | 113 |
| 5 | 103 | 11 | 99 | 17 | 112 | 23 | 106 |
| 6 | 98 | 12 | 102 | 18 | 109 | 24 | 109 |

**Step 1: moments.** `sum y = 2526` and `mean = 105.25`. The first twelve sum to 1206 with mean 100.5; the last twelve sum to 1320 with mean 110.0; the difference is 9.5.

**Step 2: two routes to the same gain.** `RSS_0 = sum (y - 105.25)^2 = 730.5`. Splitting at 12 gives `RSS_1 = 87.0` and `RSS_2 = 102.0`, within total 189.0. Then `730.5 - 189.0 = 541.5`, and CPD-01 gives `6.0 * 90.25 = 541.5`. Agreement to the last decimal is the check that the coefficient and the difference are both correct; a mismatch of exactly a factor of `n/4` usually means the coefficient was omitted.

**Step 3: the gain profile.** Scanning all 23 splits, with values recomputed from CPD-01:

| split | gain | split | gain | split | gain |
|---|---|---|---|---|---|
| 4 | 132.30 | 10 | 386.79 | 14 | 403.24 |
| 6 | 206.72 | 11 | 484.88 | 15 | 280.90 |
| 8 | 285.19 | 12 | 541.50 | 16 | 285.19 |
| 9 | 396.90 | 13 | 440.82 | 20 | 86.70 |

Note `G(8) = G(16) = 285.1875` exactly. That equality is the reversal symmetry predicted in Section 4.1 and it is the cheapest available test that an implementation indexes splits correctly.

**Step 4: significance at the estimated split.** Treating `s = 12` as fixed, `G/sigma^2 = 541.5/16 = 33.844`, and the upper tail of a one-degree-of-freedom chi-square at 33.844 is `5.98e-9`. The union bound over 23 splits gives at most `23 * 5.98e-9 = 1.38e-7`. A distribution-free check agrees: the two halves are completely separated, since `max(y_1..y_12) = 105` and `min(y_13..y_24) = 106`, so the two-sample Kolmogorov-Smirnov statistic is `D = 1.0` against a 5% critical value of `1.36 sqrt(2/12) = 0.5552`. Under the exact permutation distribution, complete separation occurs in exactly 2 of the `C(24,12) = 2704156` index assignments, so `p = 7.40e-7`.

```text
F-form check that does not assume a known sigma:
    F = G / (RSS_w / (n - 2)) = 541.5 / (189.0/22) = 541.5/8.5909 = 63.03
    F(1,22) at alpha 0.05 is 4.301 and at alpha 0.001 is 14.38
    p-value of F = 63.03 on (1,22) degrees of freedom is 6.7e-8
    implied within-segment sigma = sqrt(8.5909) = 2.931
INTERPRETATION: the split clears the 0.1% quantile by a factor of 4.4, and
    the series is 27% quieter than the assumed sigma of 4.0.
```

**Step 5: sensitivity to the noise assumption.** The verdict is not robust to `sigma`, and saying so is part of the result.

| assumed `sigma` | penalty `2 sigma^2 ln 24` | split total | null total | verdict |
|---|---|---|---|---|
| 4.00 (stated) | 101.70 | 11.81 + 101.70 = 113.51 | 45.66 | one regime |
| 2.93 (empirical) | 54.61 | 22.00 + 54.61 = 76.61 | 85.04 | two regimes |
| 6.00 (conservative) | 228.82 | 5.25 + 228.82 = 234.07 | 20.29 | one regime |

Solve for the switch point rather than picking a row: the split wins when `541.5/sigma^2 > 2 sigma^2 ln 24`, i.e. `sigma^4 < 541.5/(2*3.1781) = 85.20`, so `sigma < 3.037`. That inequality is the result. "We detect it provided the noise scale is below 3.04 units" is falsifiable; "we detect it" is not.

**Step 6: an exact DP sweep that shows the whole verdict surface.** With `sigma^2 = 8.5909` fixed at the empirical value and the penalty varied, exact optimal partitioning gives:

| `beta` | optimum total | segments | partition |
|---|---|---|---|
| 10 | 32.00 | 2 | 1–12, 13–24 |
| 30 | 52.00 | 2 | 1–12, 13–24 |
| 54.61 | 76.61 | 2 | 1–12, 13–24 |
| 60 | 82.00 | 2 | 1–12, 13–24 |
| 63.03 | 85.03 | tie | either |
| 80 | 85.03 | 1 | 1–24 |
| 101.70 | 85.03 | 1 | 1–24 |

The switch sits at `beta = 63.03`, which equals `G/sigma^2 = 541.5/8.5909`. That identity is not a coincidence: the penalty at which the two partitions tie is exactly the gain expressed in cost units, which is why `delta_min` and `beta` are the same statement written two ways.

**Step 7: a variance-change test on the same data, to show what a mean cost cannot see.** The within-half variances are `s_1^2 = 7.25` and `s_2^2 = 8.50`, so the mean of the two is `7.875`. The Gaussian profile log-likelihood gain for a variance change at that split is `12 ln(7.875/7.25) + 12 ln(7.875/8.50)`. Evaluate term by term: `ln(1.08621) = 0.08262`, times 12 is `0.99146`; `ln(0.92647) = -0.07638`, times 12 is `-0.91655`; the sum is `0.07491`, which is the likelihood-ratio statistic on one degree of freedom. Since `sqrt(0.07491) = 0.2737`, the upper tail is `2 * (1 - Phi(0.2737)) = 0.784`, so there is no evidence of a variance change at all against overwhelming evidence of a level change.

That contrast is the point. A detector built for level shifts should not be read as evidence about scales, and a report that says "the regime changed in every respect" is claiming a test that was never run. If scale change is plausible, run the variance cost in CPD-11's spirit and report the two statistics separately.

**Step 8: a slope-change example on a clean construction.** Twelve observations increasing by 2.0 per step until index 6, then by 1.0 per step: values 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26.

| model | residual sum of squares | parameters |
|---|---|---|
| one mean | 304.25 | 1 |
| one line | 8.9336 | 2 |
| two lines, break at 6 | 0.0000 | 4 |

The slope change is worth `8.9336` squared units, computed as `one-line RSS minus the sum of the two within-block line residuals`, whereas letting the level drift at all is worth `304.25 - 8.9336 = 295.32`. With noise of scale 1 and a penalty of `2 ln 12 = 4.97`, both are accepted, but a penalty chosen for a level model at a larger noise scale would accept only the trend and miss the slope change entirely.

**Step 9: the online design on the same numbers.** A monitoring deployment cannot use a retrospective `sigma`, so take `sigma = 6`, `mu_0 = 100`, target shift 6 units, hence `k' = 0.5`, `k = 3`, `h' = 5`, `h = 30`. The trace of `S_t = max(0, S_{t-1} + y_t - 103)`:

| t | y | `S_t` | t | y | `S_t` |
|---|---|---|---|---|---|
| 1 | 99 | 0.0 | 13 | 111 | 8.0 |
| 2 | 104 | 1.0 | 14 | 108 | 13.0 |
| 3 | 101 | 0.0 | 15 | 114 | 24.0 |
| 10 | 105 | 2.0 | 16 | 106 | 27.0 |
| 11 | 99 | 0.0 | 17 | 112 | 36.0 |
| 12 | 102 | 0.0 | 18 | 109 | 42.0 |

The first alarm above the threshold of 30 falls at `t = 17`, five observations after the true change at 12, i.e. 25 minutes at the 5-minute cadence. CPD-08 with the design shift predicts `5/0.5 = 10` observations; the realised delay is shorter because the actual shift is 9.5 units rather than 6, and CPD-08 with `delta' = 1.5833` predicts `5/1.0833 = 4.62`, which brackets the observed 5. That agreement is the reason to trust the formula as a design tool rather than a decoration.

**Step 10: the residual after the first regime.** Nothing in the trace resets after index 13, because every subsequent increment is positive: the increments are `8, 5, 11, 3, 9, 6, 12, 4, 7, 10, 3, 6`. A CUSUM that never resets is telling you the new regime is above the design centre, which is exactly what a level shift looks like. The diagnostic value is in the resets before index 13 — four of them — which show the old regime was centred at the assumed `mu_0`.

**Step 11: the alarm budget for a fleet.** At `h' = 5` the design yields `ARL0 = 284.83` observations per series, so 500 series at a 5-minute cadence produce `(500/284.83)*12 = 21.07` false alarms per hour, or 506 per day. At 20 minutes of review each that is `7.02` reviewer-hours per clock hour, and a two-reviewer desk cannot absorb it. Raising `h'` to 6 gives `7.57` alarms per hour and `2.52` reviewer-hours, which is feasible with three reviewers and a margin.

**Step 12: the cost of re-running a segmentation online.** A per-pass spurious expectation of 0.05 boundaries, re-run every minute for a day, gives `1440 * 0.05 = 72` boundaries per day before any deduplication. Deduplicating by boundary index reduces the reported count but not the instability, because the underlying statistic is being recomputed on almost the same data. The correct architecture is a sequential test with an `ARL0` budget, and the number 72 is what a batch segmentation in a loop will produce.

**Step 13: the same question on count data, where the cost function changes.** Nothing above transfers to counts without changing the cost, and this is the most common wrong transfer in practice. Take 24 five-minute buckets of an error count, with the first twelve summing to 137 and the last twelve to 219, so the block means are `11.4167` and `18.25` and the pooled mean is `356/24 = 14.8333`.

```text
Poisson deviance for a split at s (profile over both rates):
    D(s) = 2 * [ sum_{1..s} y log(ybar_1/ybar) + sum_{s+1..n} y log(ybar_2/ybar) ]
    with the convention 0 log 0 = 0.

FORMULA [CPD-13]: D(s) = 2 * [ S_1 ln(S_1/(n_1 ln)) + S_2 ln(S_2/(n_2 ln)) ]
                          where ln = S/n is the pooled rate, S_i block sums
WHERE: y_t counts per bucket [count]; S_i block totals [count]; D in
       deviance units, asymptotically chi-square with one degree of freedom.
CONSTRAINT: Poisson variance equal to the mean; independent counts; no
       overdispersion, which is checked by the dispersion ratio.
NUMERICAL EXAMPLE: S_1 = 137, S_2 = 219, pooled rate 14.8333
       term 1: 137 * ln(11.4167/14.8333) = 137 * (-0.26180) = -35.8670
       term 2: 219 * ln(18.25/14.8333) = 219 * (0.20729) =  45.3961
       D = 2 * (45.3961 - 35.8670) = 2 * 9.5292 = 19.0583
       p-value at one degree of freedom: 6.34e-6
INTERPRETATION: the change is real under a Poisson model, but the penalty
      for count data is 2 ln n = 6.356 deviance units, not 2 sigma^2 ln n,
      because the deviance is already scaled. D = 19.06 clears it threefold.
```

The overdispersion check is not optional here, because the same arithmetic on overdispersed counts produces false alarms at several times the nominal rate. The within-half dispersion ratios for this series are `4.0833/11.4167 = 0.3577` and `4.3864/18.25 = 0.2403`, both well below one, which means the counts are *under*dispersed relative to Poisson and the deviance test is conservative rather than anti-conservative. A ratio of 3 would require dividing `D` by 3 before comparison, which would move `19.06` to `6.35` and put it exactly at the penalty.

The smallest detectable rate change follows the same substitution as CPD-03 with the variance equal to the mean: `delta_min = sqrt(8 * lambda * ln n / n) = sqrt(8*14.8333*3.1781/24) = sqrt(15.712) = 3.964` counts per bucket. So a rise from 11.4 to 15.4 is at the detection floor at this length, which is a 35% relative change — a considerably worse relative sensitivity than the Gaussian example achieved, because count noise scales with the square root of the level.

**Step 14: conclusion with its conditions.** With `sigma` estimated at 2.93 from within segments, one change point at index 12 is supported at `p < 1e-6` by both a parametric and a parameter-free test, with segment means 100.5 and 110.0. The result is conditional on independent noise, which cannot be estimated from 24 points; on the absence of a single-observation artefact, which the KS complete separation supports because a bad value would not sit on the correct side of all twelve others; and on `sigma < 3.037` for the conservative penalty. Reporting all three conditions is the difference between a segmentation and a guess. The count-data companion in Step 13 adds a fourth condition of its own: a dispersion ratio near one, without which the deviance has no calibrated null.

## 6. Visual / Geometric Intuition

### 6.1 A split is a projection

Every split test here is a projection of the centred data onto a two-dimensional subspace spanned by the constant vector and the step function `1{t > s}`. Least squares on that subspace removes two degrees of freedom, so the gain is the squared length of the projection onto the step direction after the constant is removed. That is why `G(s)/sigma^2` has one degree of freedom rather than two: the constant appears in both models and cancels.

```text
y (an n-dimensional vector)
   |
   +-- projection onto span{1, step_s}
   |        = grand mean * 1  +  ((ybar_1 - ybar_2)/2) * step_s
   |                                  |
   |                                  +-- squared length = G(s)
   +-- residual, orthogonal to the subspace

cos^2(theta_s) = G(s) / RSS_0   (fraction of variation explained by the split)
```

The geometric picture makes the multiple-comparison problem obvious. You are choosing the best of `n-1` directions, and adjacent directions differ by one observation, so they are nearly parallel. The effective number of independent directions is closer to `log n` than to `n`, which is precisely why the penalty grows like `log n`.

### 6.2 The run-length matrix

BOCPD's state is a distribution over run lengths, and its evolution is clearest as an upper-triangular array whose row is time and whose column is run length. Each step shifts every cell one to the right and seeds a new column zero on the diagonal.

```text
time ->
      r=0    r=1   r=2   r=3   r=4
t=1   w10
t=2   w20    w11
t=3   w30    w21   w12
t=4   w40    w31   w22   w13
t=5   w50    w41   w32   w23   w14

mass marching right along a row = no change supported
mass re-seeding at column 0     = change supported at that time
two ridges alive at once        = ambiguous change time
```

Reading the matrix as a picture is faster than reading the probability series. A clean regime shows a single ridge advancing; a change shows the ridge dying and a new one starting. Two simultaneous ridges are the case a point estimate hides, and they are the honest answer when the data cannot separate two candidate boundary times.

### 6.3 CUSUM is a ratchet

The `max(0, .)` is a ratchet: positive deviations accumulate and negative runs reset the statistic to zero. Geometrically the statistic is the height of the largest drawup in the random walk `sum (y_t - mu_0 - k)`.

```text
S
^                                    /|  alarm at h
|                              /----/ |
|                        /----/       |
|            /----------/             |
|      /----/                         |
|  ---/                               |
+-----------------------------------------> t
       \      /\     /\                  reset on a negative run
        \____/  \___/  \___
```

Two design instincts follow. A larger `k` flattens the drift, so only large steps reach `h` quickly; a larger `h` raises the bar, so small steps take longer. Every "the detector is too slow" complaint is a request for a smaller `h`, which is a request for more false alarms, and that trade must be made in the open.

### 6.4 Why the estimated boundary is not where the change was

The gain coefficient `n1 n2/n` peaks at the middle of the series, so under the null the estimate is pulled centrally, and under a true change the estimate has a spread of order `sigma^2/delta^2`. Two analysts with different penalties often report boundaries two or three observations apart on identical data and then argue about the truth, when both are inside the estimator's own uncertainty. Reporting a boundary to the nearest observation is a precision claim that the data usually does not support at a 5-minute cadence.

### 6.5 How to read a segment plot without being fooled

Plot the series with the fitted segment means, and then plot the residuals of the fitted model on the same time axis. A correct segmentation leaves residuals with no visible structure. Three patterns are diagnostic: residuals that ramp within a segment mean a slope was fitted as a level; residuals that change scale mid-segment mean the variance changed; and residuals that spike at the boundary mean the boundary was placed to absorb an outlier rather than a regime. None of these is visible in a plot that shows only the series and the means.

A second reading habit: mark the penalty in the same units as the residual sum of squares on the vertical axis. A gain of 541.5 looks enormous on a plot whose vertical range is 20 units, and looks modest when the residual sum of squares is 730.5. The comparison that matters is between the gain and the noise, never between the gain and zero.

### 6.6 Localisation is an interval, not a point

The gain profile around the maximum carries the uncertainty of the boundary, and reading it off is cheaper than a bootstrap. On the worked series with `sigma^2 = 8.5909`, the profile is `G(11) = 484.88`, `G(12) = 541.50`, `G(13) = 440.82`, `G(10) = 386.79`, `G(14) = 403.24`. The drops from the maximum are 56.62, 100.68, 154.71 and 138.26 respectively.

A likelihood-ratio style confidence set keeps every `s` whose loss is within a chi-square quantile of the maximum. At one degree of freedom and 68% coverage the cut is 1.0, at 95% it is 3.84, and at 99% it is 6.63. Since every neighbouring loss exceeds 56.62, which is 6.6 times the local noise variance, the 99% set contains index 12 alone. That is a strong localisation, and it is strong precisely because the shift is 9.5 units against a noise scale of 2.93.

Now shrink the shift and watch the interval open. With a shift of 2.0 units and the same noise, the neighbouring loss would scale as `(delta^2)` times the coefficient difference, falling to about `56.62 * (2/9.5)^2 = 2.51`, which is below the 3.84 cut, so the 95% set would span indices 11 through 13. Reporting "the change was at 12" would then be a 1-observation precision claim on 3-observation evidence. The rule to carry away: the confidence width scales as `sigma^2/delta^2`, so a boundary quoted to the nearest observation is only meaningful when the shift is at least a few sigma.

### 6.7 Seasonality is the most common spurious structure

A periodic component produces a large gain at any split that puts high-phase and low-phase points in different segments, even when no regime changed. Take a pure sinusoid of amplitude `A` and period `T = 288` observations (one day at a 5-minute cadence) and split it into equal halves of 144. The mean of a full half-period of a sinusoid is `2A/pi = 0.6366 A`, so the two halves have means `+0.6366A` and `-0.6366A` and their difference is `1.2732 A`.

```text
FORMULA [CPD-14]: G_cycle = (n/4) * (2 * 2A/pi)^2 = 4 n A^2 / pi^2
                  for a split into two equal halves of a complete period
WHERE: A sinusoid amplitude [series units]; n series length; the split
       coefficient n1 n2/n equals n/4 at a balanced split; the mean of a
       half period of A sin(2 pi t/T) is 2A/pi.
CONSTRAINT: exactly half a period per segment and a split at a zero crossing;
       any other split or window gives a smaller realised difference.
NUMERICAL EXAMPLE: A = 3, T = 288, n = 288, split at 144
       half-mean 2*3/3.141593 = 1.90986 ; difference 3.81972 ; squared 14.590
       coefficient 144*144/288 = 72 ; G = 72 * 14.590 = 1050.5
       penalty 2 * 36 * ln 288 = 72 * 5.66296 = 407.73
INTERPRETATION: the sinusoid clears the penalty 2.6 times over with no regime
       change at all. Solving 4*288*A^2/pi^2 = 407.73 gives A = 1.869, so any
       daily cycle above 1.87 units in amplitude is reported as a change point
       by an unadjusted mean-cost segmenter at this noise scale.
```

That is the arithmetic behind the most common false finding in monitoring: a daily cycle, a weekly cycle, or a rolling-deploy cadence presented as a regime change. Three repairs exist and they differ in what they cost. Remove the periodic component by regression on harmonic terms, which is right when the period is known and stationary; segment the residual, which then describes deviations from the seasonal norm rather than the series itself; or use a cost that compares full-period blocks, which requires the regime to be at least a period long and therefore rules out detecting anything faster than a day.

## 7. Comparison / Selection Table

| Method | Problem | Cost per pass | Key assumption | Use when | Avoid when |
|---|---|---|---|---|---|
| Fixed-threshold | online | O(1) | iid, large shift | shift above 3 sigma | persistent small shifts |
| Shewhart with runs rules | online | O(1) | iid | need early warning on drift | multiple testing inflates alarms |
| EWMA | online | O(1) | iid, tuned smoothing | gradual drift | abrupt single-sample step |
| CUSUM | online | O(1) | iid, known baseline | 0.5 to 2 sigma persistent shift | baseline unknown or drifting |
| BOCPD | online | O(t) per step | conjugate predictive, geometric prior | need a change-time distribution | long series without truncation |
| Optimal partitioning | retrospective | O(n^2) | additive cost | exact answer, n in low thousands | n above 10^4 |
| PELT | retrospective | O(n) average | additive and superadditive | long series, exact answer | non-superadditive robust cost |
| Binary segmentation | retrospective | O(n log n) | one dominant change per level | fast first pass, few changes | clustered or small changes |
| Sliding-window test | both | O(nL) | local stationarity | localising a known change | unknown change count |
| Kernel MMD segmentation | retrospective | O(nL^2) naive | iid within block, kernel chosen | distributional change | pure level change |

**Selection rules that survive production.** If the decision must be made within one sampling interval, only the first five rows are available. If the answer must be exact and `n` is a few thousand, use PELT and verify it against DP on a random subset of series. If the change may be distributional, no mean-cost method will find it, so choose between a variance cost and a kernel cost and say which. If the series is contaminated, robustness beats exactness, because an exact answer to the wrong question is still wrong.

**What choosing wrong costs.** The table below prices the four common mis-selections, using the arithmetic established earlier.

| Mis-selection | Failure mode | Quantitative consequence |
|---|---|---|
| Mean cost for a variance change | blind, reports one regime | on the worked series, variance gain 0.0378 gives p = 0.85 |
| Fixed threshold for long series | spurious boundaries | null maximum rises from 101.7 at n = 24 to 232.7 at n = 1440 |
| Binary segmentation for clustered changes | misses the small ones | second-level gain falls below the penalty of 4.97 at L = 10 |
| Batch segmentation in a loop | alarm flood | 1440 passes times 0.05 equals 72 boundaries per day |

The second row is the one that generates the most rework, because a penalty calibrated at pilot length is not obviously wrong: it produces plausible boundaries that a reviewer will investigate and eventually discredit, after spending the review hours.

## 8. Failure Modes & Edge Cases

**F1: spurious boundaries grow with the replay length.**
Symptom: a 30-day replay reports 40 boundaries, each individually plausible, and the count rises when the replay is extended.
Competing causes: a constant penalty copied from a short pilot; a noise scale underestimated because it was computed over the changes themselves; genuinely frequent changes.
Discriminating check: recompute the penalty as `2 sigma^2 ln n` with `sigma` from a known-quiet stretch, then recount. If the count collapses, the penalty was constant. If it collapses only when `sigma` comes from a robust estimator such as the median absolute deviation, the noise was underestimated. If it does not move, inspect boundary alignment with clock times.
Correction: penalty proportional to `log n`, and a stated convention for the factor of two.
Prevention: every reported count carries `n`, `sigma`, and the penalty formula.
Verification: segment a synthetic null series of the same length and confirm the boundary count is near zero.

**F2: autocorrelation read as change.**
Symptom: boundaries at every seasonal trough and at the start of every slow ramp, on series with no step.
Competing causes: the threshold assumes independence but the series is AR(1) with `rho` near 0.9, inflating the standard error 4.36-fold; the aggregation window exceeds the correlation length, creating pseudo-observations; a real slow drift is being modelled as a step.
Discriminating check: compute lag-1 autocorrelation of the residuals and the effective length `n(1-rho)/(1+rho)`. If the boundaries survive only when the effective length is ignored, dependence is the cause. If they align with window multiples, aggregation is the cause.
Correction: block bootstrap or permutation thresholds that preserve dependence, or thin the series to near-independence.
Prevention: report `rho` and `n_eff` beside every threshold, and recompute `delta_min` with `n_eff`.
Verification: a null replay of an AR(1) surrogate with matched `rho` should produce the designed alarm rate.

**F3: a pipeline change read as a business change.**
Symptom: every series in the fleet reports a boundary inside the same 24-hour window, or boundaries land exactly at a UTC day rollover.
Competing causes: a deploy changed the event schema so a field is now null and imputed differently; a cadence change altered bucket widths; a genuine fleet-wide event.
Discriminating check: compute the fraction of series whose best boundary falls inside a 2-hour window. Under schema or cadence causes that fraction approaches one. Then test magnitude homogeneity: a genuine global event usually produces heterogeneous effect sizes, while a schema change produces a similar shift everywhere.
Correction: segment the residual after removing a common fleet factor, so per-series boundaries describe business variation.
Prevention: log schema and cadence versions as covariates and annotate the series with them.
Verification: re-run the segmentation with the annotated windows excluded and confirm the boundaries move.

**F4: a mean cost missing a scale change.**
Symptom: operators insist a segment is different, the segmenter reports one regime, and the means really are close.
Competing causes: the change is in variance only; the change is in shape with an unchanged mean; the change is real but confined to a subpopulation diluted by aggregation.
Discriminating check: refit with a log-variance cost and with a kernel cost at the same penalty. If the variance cost finds a boundary at the operator's time, scale is the cause. If a multivariate `log det Sigma` cost finds it, shape is the cause. If only a per-cohort fit finds it, dilution is the cause.
Correction: choose a cost matched to the changed quantity and report which cost produced the boundary.
Prevention: state the cost function in the same sentence as the segment count.
Verification: synthetic series with a known variance step should be detected by the chosen cost and missed by the mean cost, confirming the contrast.

**F5: detecting the announcement rather than the event.**
Symptom: the boundary lands at the deploy timestamp minus the propagation lag, and retrospective and online views disagree by exactly that lag.
Competing causes: the series is indexed by processing time, so late events are attributed to when they were seen; the estimator biases boundaries towards the series centre; the feature genuinely lags the event.
Discriminating check: re-key by event time and re-run. If the boundary moves by a fixed offset equal to the median ingestion latency, indexing is the cause. If it is stable but interior-biased, the estimator is the cause.
Correction: segment on event time, and keep processing time only as a separate diagnostic series.
Prevention: store both timestamps on every aggregated row.
Verification: compare the median ingestion latency with the observed offset; equality confirms the mechanism.

**F6: the boundary at the very edge.**
Symptom: the reported change is at index 1 or at index `n`, and segment statistics come from a handful of points.
Competing causes: a single contaminating observation, as in the 8762-gain counterexample; a genuine change just before the window, so the first segment is truncated; a windowing scheme that reuses overlapping windows.
Discriminating check: winsorise the first and last three observations at the 1st and 99th percentiles and re-run. If the boundary disappears, contamination is the cause. If it persists with the first segment much shorter than the rest, truncation is the cause, and the honest report is "change at or before the start of the window".
Correction: enforce a minimum segment length and treat edge boundaries as censored rather than located.
Prevention: never report a segment shorter than three observations without saying so.
Verification: inject a single outlier at the edge of a null series and confirm the guard rejects it.

**F7: an alarm flood from batch segmentation in a loop.**
Symptom: offline segmentation of a month finds two boundaries, while the online monitor on the same month fired thirty times.
Competing causes: the monitor re-runs a segmentation on a trailing window each minute, multiplying the per-pass expectation by the number of passes; the monitor's threshold came from a segmentation penalty, which controls a different quantity; the monitor sees partial buckets the offline pass does not.
Discriminating check: replay a clean month with notifications suppressed and count. Compare against `passes * per_pass_expectation`, which for 1440 passes and 0.05 is 72. Then check whether alarm times cluster at bucket edges, which indicates partial buckets.
Correction: replace the batch-in-a-loop with a sequential test whose `ARL0` is set from the budget in CPD-07.
Prevention: publish the design pair `(ARL0, delay)` with the detector, not just a threshold.
Verification: a null replay must reproduce the designed alarm count within sampling error.

**F8: the ARL0 formula silent on heavy tails.**
Symptom: the measured false-alarm rate is three to five times the formula's prediction on data that look stationary.
Competing causes: occasional extreme values push the statistic faster than Gaussian theory allows; the pre-change mean is estimated and drifts, so the nominal `k` is wrong; the series mixes populations so no single baseline exists.
Discriminating check: compute kurtosis. Above about 6 the exponential approximation is unsupported. Then re-centre the baseline monthly: if the excess disappears, drift is the cause. If the histogram is bimodal, mixing is the cause.
Correction: bootstrap the statistic under observed noise, or use a rank-based statistic whose null does not depend on tails.
Prevention: never quote a theoretical `ARL0` without a null-replay confirmation.
Verification: measure the empirical inter-alarm distribution and compare its mean and upper quantiles with the design.

## 9. Complexity & Cost Analysis

**Computational cost.** The table counts cost-function evaluations, which is the implementation-independent quantity; the constant factor is dominated by the inner loop of the cost.

| Method | Time | Memory | Evaluations at n = 1440 | At 1 microsecond each |
|---|---|---|---|---|
| CUSUM, EWMA | O(1) per observation | O(1) | 1440 | 1.4 ms |
| Binary segmentation | O(n log n) | O(n) | about 15100 | 15 ms |
| PELT average | O(n) | O(n) | about 4300 | 4.3 ms |
| PELT worst case | O(n^2) | O(n) | 2073600 | 2.07 s |
| Optimal partitioning | O(n^2) | O(n) | 1036800 | 1.04 s |
| BOCPD, all steps | O(n^2) total | O(n) | 1036800 | 1.04 s |
| MMD, naive per block | O(L^2) per block | O(L) | depends on L | grows quadratically |

The PELT average assumes a handful of true changes; with a change every 20 observations the candidate set stops shrinking and runtime approaches the worst case, which is the assumption to check before deploying on a noisy series. The BOCPD figure is a sum over steps, since each step costs `O(t)`; practical implementations truncate the run-length support at a few hundred states, which is an approximation whose cost is delay for regimes longer than the truncation.

**Prefix sums make block costs constant time.** For squared error, precomputing `sum y` and `sum y^2` lets any block cost be evaluated in constant time, so the quadratic term in the complexity is the number of candidate splits, not the cost evaluation. For `n = 1440` that precomputation is 2880 additions, which is negligible against the 1.04 million evaluations of exact DP.

**Fleet compute.** At 10^4 series with 1440 five-minute points per day and three cost evaluations per observation, PELT performs `10^4 * 1440 * 3 = 4.32e7` evaluations per day, under a minute of single-core time at a microsecond each. Compute is almost never the binding constraint, which is why an infrastructure budget without a review budget is an incomplete plan.

**Review cost is the real budget.** From CPD-07 with `k' = 0.5`, 500 series, 5-minute cadence and 20 minutes per review:

```text
ARL0 = 284.83 observations
alarms per hour = (500 / 284.83) * (60 / 5) = 1.7555 * 12 = 21.07
reviewer-hours per hour = 21.07 * (20/60) = 7.02
```

Those 7.02 hours are spent on false alarms alone; true changes consume additional time proportional to their arrival rate. A two-reviewer desk has a 3.5-fold deficit before any true positive arrives, so the design must change: pool series, relax the cadence, or raise `h'`.

**Latency budget versus capacity budget.** A 30-minute response SLA at a 5-minute cadence allows 6 observations, which at `delta' = 1.0` requires `h' <= 3` by CPD-08. But `h' = 3` gives `ARL0 = 32.17`, i.e. 2.68 hours between false alarms per series, or 186.5 alarms per hour across 500 series. The two budgets cannot both be met for a 1-sigma shift on a single 5-minute series. The honest options are to pool series so each alarm covers many entities, to lengthen the cadence, or to accept a larger target shift and say so in the SLA.

**Value arithmetic.** If a missed day of a 1-day outage costs 40,000 currency units and a review costs 15 units, a review is worth doing whenever the probability of a true change exceeds `15/40000 = 3.75e-4`. That very low bar is the quantitative form of "in outage detection, alarm aggressively" — and it is exactly why the capacity constraint, not the value calculation, ends up binding.

**Amortisation of the retrospective pass.** A monthly exact segmentation of 10^4 series costs about `10^4 * 10^-3` seconds per series at a millisecond each, roughly three hours of single-core work, and it produces the reference against which the online detector is audited. That audit is what makes offline-online consistency measurable rather than aspirational, so the retrospective pass should be budgeted as a control, not as a one-off analysis.

## 10. Key Theorems / Results Reference

**Page (1954), the CUSUM procedure.** Introduces the cumulative-sum scheme and reads the statistic as accumulated evidence. What it licenses: the recursive `max(0, .)` form is equivalent to restarting a sequential test after each reset, which makes false-alarm behaviour analysable. What it does not license: a closed-form `ARL0` for arbitrary noise, which is why CPD-07 is an approximation.

**Lorden (1971), minimax optimality.** Among stopping rules meeting an `ARL0` constraint, CUSUM minimises the worst-case expected delay over change times and pre-change histories. What it licenses: a principled default when the shift size is uncertain. What it does not license: optimality at a specific shift, where a likelihood-ratio rule tuned to that shift can be better on average.

**Moustakides (1986), exact optimality.** Sharpens Lorden to exact optimality for a simple alternative. What it licenses: if the shift and the noise family are known, the tuned CUSUM is not merely convenient but minimax optimal. What it does not license: transferring that guarantee to a tuned-on-the-data threshold, which is a different procedure.

**Yao (1988), consistency of penalised segmentation.** With a penalty growing like `log n` and a fixed number of change points, estimated locations converge and the estimated count converges. What it licenses: the `2 sigma^2 log n` penalty and the claim that more data improves localisation. What it does not license: a fixed penalty, or consistency when the number of changes grows with `n`.

**Killick, Fearnhead and Eckley (2012), PELT.** Proves that under superadditivity, pruning discards only candidates that can never be optimal, so PELT returns the exact penalised optimum in linear average time. What it licenses: exactness at scale for squared error and fixed-parameter likelihoods. What it does not license: exactness for arbitrary robust costs, where the condition can fail.

**Venkatraman and Olshen (2007), inconsistency of binary segmentation.** Constructs configurations where a fixed-size change exists yet the probability of detecting it tends to zero. What it licenses: the warning against binary segmentation for clustered changes. What it does not license: a general claim that binary segmentation is useless, since it remains a good first pass on well-separated changes.

**Barry and Hartigan (1993), product partition models.** A Bayesian formulation with a prior over partitions, returning a posterior rather than a single optimum. What it licenses: reporting uncertainty in the number of segments. What it does not license: interpreting a posterior segment count as a frequentist error rate.

**Adams and MacKay (2007), BOCPD.** Derives the run-length recursion with constant hazard. What it licenses: exact online posteriors for conjugate models in `O(t)` per step. What it does not license: comparing probabilities across different `H`, or reading the posterior as a calibrated alarm probability.

**Gretton et al. (2012), kernel two-sample test.** Establishes the MMD statistic and consistency against any alternative for a characteristic kernel. What it licenses: distributional change detection without a parametric family. What it does not license: uniform power, since an ill-chosen bandwidth has low power against precisely the change of interest and the test gives no warning.

**Truong, Oudre and Vayatis (2020), review of change-point detection.** Frames the field as one penalised-comparison template with different costs, separating offline from online objectives. What it licenses: the taxonomy in Section 2.1. What it does not license: any specific penalty or threshold, which remain application decisions.

**Inclan and Tiao (1994), variance change detection.** Provides a cumulative-sum-of-squares statistic for detecting variance changes, with a distribution that differs from the mean-shift case. What it licenses: the warning that a variance test needs its own null distribution. What it does not license: reusing mean-shift thresholds for scale changes.

**The negative result worth memorising.** No procedure estimates both the count and the locations at a rate that is uniform over all configurations, because localisation error scales like `sigma^2/delta^2` and delay for small shifts is unbounded. Every "we detect any change" claim is a claim about the shifts that have been sampled, and a specification should say which.

## 11. Interview / Exam Questions

**Q1. A 24-point series with `sigma = 4` gives a maximum split gain of 541.5. Is that a change point?**
Answer: under the stated noise model yes, with a large margin, but the answer is a conditional one. The conservative penalty is `2*16*ln 24 = 101.70`, so the gain clears it 5.3 times; the empirical penalty using `sigma = 2.93` from within segments is 54.61, so the margin improves.
Reasoning: the decision compares `RSS_w/sigma^2 + beta` with `RSS_0/sigma^2`, which for `sigma = 4` is 113.51 against 45.66, and that comparison favours the null. The gain clears the penalty but the scaled residual does not, which is the subtlety: the verdict flips at `sigma = 3.037`.
Follow-up: at what noise scale does the decision reverse, and which way? Reverse above 3.037, where the penalty outgrows the gain.
Red flag: answering "yes, 541 is large" without naming a penalty or a variance.

**Q2. Compute the smallest shift detectable at `n = 1440` with `sigma = 6`.**
Answer: `delta_min = sqrt(8*36*ln 1440/1440) = sqrt(2094.5/1440) = sqrt(1.4545) = 1.206` series units, which is 0.201 sigma.
Reasoning: balanced gain is `n delta^2/4 = 360 delta^2`; the penalty is `2*36*7.2724 = 523.6`; setting them equal gives `delta^2 = 1.4544`. The check is that `360 * 1.4544 = 523.6`.
Follow-up: with `rho = 0.9` and `n_eff = 75.79`, the floor rises to `sqrt(8*36*4.328/75.79) = 4.055`, a 3.4-fold penalty for dependence.
Red flag: using `ln n` with `n` counted in days rather than observations.

**Q3. Your CUSUM at `k' = 0.5`, `h' = 5` produces 300 alarms a day across 500 series at a 5-minute cadence. Is the code broken?**
Answer: no, it is behaving as designed. `ARL0 = 284.83` observations per series, so the fleet produces `500/284.83 = 1.7555` alarms per sampling interval, and with 288 intervals per day that is `505.5` alarms per day. Observing 300 is below the design rate.
Reasoning: convert `ARL0` to a per-series per-observation rate, then scale by series count and intervals per day. The unit error to avoid is treating a per-series `ARL0` as a fleet-level daily figure.
Follow-up: to get under 20 alarms per day at 500 series you need `ARL0 >= 500*288/20 = 7200`, which by CPD-07 means `h'` near 6.9.
Red flag: concluding "the threshold is too low" without computing the specified rate.

**Q4. Why is `ARL0` insufficient to describe false-alarm behaviour?**
Answer: it is a mean of a positive random variable, so it fixes neither the spread nor the tail. Two designs with `ARL0 = 500` can differ by an order of magnitude in the 99th percentile of inter-alarm time, and operations cares about clusters rather than long-run averages.
Reasoning: false alarms arrive in bursts when the statistic is correlated, so the Poisson reading of "alarms per hour" is itself unverified. A design that is fine on average can still flood a shift.
Follow-up: measure the empirical inter-alarm distribution on a null replay and check for burstiness before publishing a rate.
Red flag: treating `1/ARL0` as an independent per-observation probability.

**Q5. Distinguish a misplaced boundary from a spurious one, with one diagnostic each.**
Answer: a misplaced boundary has the right number of regimes but the wrong index; a spurious boundary has no regime change, so the count is too high.
Reasoning: the causes differ. Misplacement is usually event-time versus processing-time indexing or aggregation, so the diagnostic is a fixed offset from a known deploy equal to the median ingestion latency. Spuriousness is usually a penalty or dependence problem, so the diagnostic is recounting after replacing a constant penalty with `2 sigma^2 ln n` and re-estimating `sigma` robustly.
Follow-up: what if a boundary is both misplaced and spurious? Then the segment statistics describe nothing and the run should be discarded rather than adjusted.
Red flag: calling both "false positives" and applying one fix.

**Q6. Compute reviewer load at `h' = 6` for 500 series, 5-minute cadence, 20 minutes per review.**
Answer: `exp(2*0.5*6) = exp(6) = 403.4288`; numerator `403.4288 - 6 - 1 = 396.4288`; divide by `2*0.25 = 0.5` to get `ARL0 = 792.86`. Then `(500/792.86)*12 = 7.567` alarms per hour, and `7.567*20/60 = 2.52` reviewer-hours per hour.
Reasoning: two conversions, `ARL0` to rate and rate to hours, and the common error is expanding `2k'h'` as `12` when it is `6`, which inflates `ARL0` by `exp(6)`.
Follow-up: halving review time to 10 minutes halves the load to 1.26 hours, which is cheaper than raising `h'` to 7, since that costs 3.6 times the delay.
Red flag: computing `exp(h')` instead of `exp(2k'h')`.

**Q7. A team reports 40 boundaries in 1440 points and concludes the system is unstable. What is wrong?**
Answer: the implicit regime length is `1440/41 = 35.1` observations, or 2.9 hours at a 5-minute cadence. That is almost certainly below the physical regime length of the process, so the prior is contradicted and the claim is about the penalty, not the system.
Reasoning: a segment count implies a regime length, and that length can be compared with domain knowledge before any statistics are computed. A count that implies a 3-hour regime in a system whose regimes last days is a specification error.
Follow-up: recompute with `2 sigma^2 ln n` using a robust `sigma`. If the count collapses, the original `sigma` was inflated by the very changes it was meant to detect.
Red flag: reading a large segment count as evidence of instability.

**Q8. Why can binary segmentation miss two small changes that follow one large change?**
Answer: the first split is chosen at the largest contrast, which partitions the series so both small changes fall in the same half. Within that half, the sample length available to test them has shrunk, but the penalty has not shrunk with it.
Reasoning: with a half of length 10 and two 0.5-sigma changes, the combined gain is about `10*(1)^2/4 = 2.5` squared sigma units, while the penalty `2 ln 10 = 4.61` exceeds it. With a half of length 100 the same pair gives `100/4 = 25`, which clears `2 ln 100 = 9.21`, so the failure is length-specific and not universal.
Follow-up: the fix is exact global search, or a penalty scaled to the current segment length, and either way the exactness claim should be tested against DP.
Red flag: asserting binary segmentation is always consistent.

**Q9. A null replay shows four times the formula's false-alarm rate. Give three causes and a check for each.**
Answer: heavy tails, checked by kurtosis above about 6; an estimated and drifting baseline `mu_0`, checked by re-centring monthly; and population mixing, checked by a bimodal histogram that no single baseline can represent.
Reasoning: the formula is an iid Gaussian result, and the three causes violate tail shape, parameter stationarity and homogeneity respectively, which are the three assumptions independently.
Follow-up: the robust repair is a bootstrap of the statistic under observed noise, which preserves tails and dependence at once, at the cost of simulation per release.
Red flag: defaulting to "implementation bug" as the only hypothesis.

**Q10. A series is segmented at index 1 with a gain of 8762. Most likely explanation and correct action?**
Answer: one contaminating observation, a lost decimal turning 99 into 9.9. The edge coefficient is `23/24 = 0.958`, so a deviation of 95.6 units gives `0.958*95.6^2 = 8762`; the arithmetic is right and the data are wrong.
Reasoning: the edge coefficient is only 6.3 times smaller than the maximum coefficient of 6.0 at the balanced split, so a deviation 2.5 times larger at the edge beats a balanced shift. That is why edge splits dominate contaminated series.
Follow-up: the structural fix is a range check plus a minimum segment length plus a robust cost; a minimum length alone does not help, because a 3-point edge segment can still be built from one bad value.
Red flag: raising the penalty until the boundary disappears.

**Q11. State the condition under which PELT is not exact, and how to test for it.**
Answer: the pruning argument requires `C(a..t) + C(t+1..u) + K <= C(a..u)` for all `a < t < u`. Squared error and fixed-parameter Gaussian negative log-likelihood satisfy it; some robust and rank-based costs do not, and pruning can then discard an optimum.
Reasoning: the inequality is what guarantees that a pruned candidate cannot become optimal later; without it, discarding is irreversible within a pass and the returned optimum is approximate.
Follow-up: run DP and PELT on 200 random short series and compare optima; any disagreement proves the condition fails for the chosen cost, and pruning must be disabled.
Red flag: "PELT is exact" without naming the cost.

**Q12. An analyst sets `k` equal to the shift they want to detect. What is wrong?**
Answer: the drift under a shift is `delta - k`, so `k = delta` gives zero drift and infinite expected delay. The standard choice is `k = delta/2`, which halves the drift but keeps it positive and balances detection against false alarms.
Reasoning: substituting into the walk shows the expected increment is exactly zero at `k = delta`, so the statistic is a martingale and the threshold is not reached in expectation. At `k = delta/2` the drift is `delta/2`, so the threshold is crossed in about `2h/delta` observations.
Follow-up: for a range of shifts, run two CUSUMs at half the small and half the large target and accept the doubled false-alarm rate, or combine alarms with a rule and recompute `ARL0` by simulation.
Red flag: setting `k` from the observed shift after looking at the data.

**Q13. Compute the daily spurious count for a segmentation re-run every minute on a trailing window, given a per-pass expectation of 0.05.**
Answer: `1440 * 0.05 = 72` boundaries per day before any deduplication. Consecutive passes share 1439 of 1440 points, so the true count is below 72, but still of that order, because overlap concentrates repeated detections of the same boundary rather than suppressing them.
Reasoning: the per-pass expectation applies to each pass, and a batch estimator has no notion of having already reported. Deduplication reduces the reported number without reducing the instability.
Follow-up: the correct architecture is a sequential test with an `ARL0` budget from CPD-07; the batch-in-a-loop is the defect, not the threshold value.
Red flag: dividing by two "for overlap" without a measurement.

**Q14. What does a BOCPD change probability of 1.3% mean, and why is 0.5 the wrong threshold?**
Answer: it means that, under a geometric prior with `H = 144`, the posterior mass on a reset at this observation is 0.0128. The prior alone puts 0.00694 on a reset, so a likelihood ratio of 1.85 lifts it to 0.0128, which is weak evidence.
Reasoning: `0.004514/0.352084 = 0.01282`, and reaching 0.5 would need a likelihood ratio of roughly 140 at this hazard, so a 0.5 threshold essentially never fires. The threshold should come from a decision cost, `p * value > cost`.
Follow-up: changing `H` to 24 multiplies the reset prior by six and rescales every reported probability, which is why the number is meaningless without `H`.
Red flag: quoting a change probability with no `H` and no decision cost.

**Q15. When is kernel MMD strictly worse than a mean-shift test?**
Answer: for a pure location shift with a bandwidth not matched to the shift size. MMD spreads power over all directions in feature space, so at the same false-alarm rate it has lower power against a one-dimensional alternative than a targeted mean test.
Reasoning: with a squared-exponential kernel, the MMD signal for a shift is second order in `delta/ell` when `ell` is much larger than `delta`, so it vanishes relative to sampling noise while a mean test reads the shift at first order.
Follow-up: choose the bandwidth by the median heuristic on a held-out stretch, then verify power by permutation against a synthetic shift of the size you care about.
Red flag: choosing MMD because it is "more general" without a power comparison.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Split gain | `G(s) = (n1 n2/n)(ybar_1 - ybar_2)^2`, the between-segment sum of squares |
| Penalty | `beta = 2 sigma^2 ln n`, BIC variant `sigma^2 ln n`; name the convention |
| Smallest shift | `delta_min = sqrt(8 sigma^2 ln n / n)` at even odds, balanced split |
| CUSUM reference | `k = delta_target/2`; drift is `delta - k`, never zero |
| CUSUM false alarms | `ARL0 ~ [exp(2k'h') - 2k'h' - 1]/(2k'^2)`, then rate `= 1/ARL0` |
| CUSUM delay | `ARL1 ~ h'/(delta' - k')` observations; add the 90th percentile |
| Fleet load | `alarms/h = (series/ARL0)(60/cadence)`; review hours usually bind |
| Dependence | `n_eff = n(1-rho)/(1+rho)`; 4.36-fold error inflation at `rho = 0.9` |
| Verdict honesty | state the critical `sigma` at which the segmentation decision flips |
| Never report | a boundary count without `n`, `sigma`, penalty convention and cadence |
