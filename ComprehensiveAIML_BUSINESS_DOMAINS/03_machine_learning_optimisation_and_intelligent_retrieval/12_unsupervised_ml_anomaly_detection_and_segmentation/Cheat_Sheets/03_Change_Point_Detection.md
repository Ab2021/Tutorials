---
title: "Change-point detection"
topic: "Change-point detection: retrospective segmentation and online sequential detection"
collection: "unsupervised-ml-anomaly-detection-and-segmentation"
file_id: "03"
level: "core"
version: "2.0"
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

> **Evidence:** All formulas below are `DERIVED` from stated likelihood or stopping-time assumptions, and every arithmetic line is `VERIFIED` by an independent recomputation shown inline. Event counts, sampling cadences, reviewer costs and fleet sizes are `ILLUSTRATIVE` constructed values. Bibliographic attributions are `SOURCED` by title and author only; no source body was opened. Any claim about a real deployment is `OPEN`.

## 1. File Header

> **Purpose:** decide whether a series actually changed regime, where it changed, and when to alarm — and to do that with an explicit false-alarm budget instead of by eyeballing a plot.
> **Prerequisites:** expectation and variance of a sample mean, the normal and chi-square quantiles, log-likelihood ratios, and the idea of a stopping time. You should be able to compute a pooled variance by hand and read a cumulative distribution function at a stated quantile.
> **Key outcome:** given an ordered series, a noise scale, a sampling interval, and a false-alarm budget expressed per series per day, you can (a) pick retrospective segmentation versus online monitoring, (b) choose a penalty that controls spurious segments, (c) compute the smallest shift you can detect at that length, (d) compute the expected detection delay and alarm rate of a CUSUM design, and (e) reject a "change point" that is really a sampling, schema, or autocorrelation artefact.
> **This file does NOT cover:** causal attribution of a regime to a specific deploy or actor, forecasting after a change, or the review workflow that consumes the alarm — that workflow is in `05_Human_Review_Loops.md`.
> **Level:** core, with production diagnostics in Sections 8–9.
> **Evidence:** `DERIVED` for every identity; `VERIFIED` hand arithmetic in Sections 4–6; `ILLUSTRATIVE` series values, cadence and costs; `OPEN` all deployment claims.
> **Retrieval cue:** if a dashboard shows "a change on Tuesday" and nobody agrees it happened, start at Section 8 and work back to Section 4.1.

The distinction that removes most confusion in this topic: a change point is a property of the *generating process*, a segmentation is a property of the *estimate*, and an alarm is a property of a *decision rule* with a budget. These three objects have different falsification tests and different costs. Conflating them is how teams end up arguing about whether Tuesday "really" had a change when the real dispute is about whether the alarm rate was ever specified.

## 2. Concept Map / Mental Model

### 2.1 Two problems wearing one name

Retrospective segmentation and online detection look similar in a textbook chapter and behave nothing alike in production. The first consumes the whole series and answers "how many regimes, and where". The second consumes one observation at a time and answers "alarm now or wait". The objectives are formally different: segmentation minimises a penalised fit criterion over all partitions, while online detection minimises a detection delay subject to a constraint on the average time between false alarms.

```text
RETROSPECTIVE (offline)                      ONLINE (sequential)
---------------------------                  ---------------------------
whole series y_1..y_n available              only y_1..y_t available
choose partition 0=t_0<t_1<...<t_m=n        choose a stopping rule T
objective: fit + beta * (#segments)          objective: E[T-tau | tau] small,
no notion of latency                         subject to ARL0 >= target
output: segment boundaries, segment stats    output: alarm time, or silence
failure: too many segments                   failure: too few alarms / too late
budget: penalty beta                         budget: threshold h and reference k
```

A team that tunes a penalty `beta` and then wires the same code into a real-time loop has changed the problem without changing the code. The penalty says nothing about the alarm rate: a segmentation with a large penalty can still alarm constantly if it is re-run on a short trailing window every minute, because each re-run re-examines the same evidence and each boundary is reported as new.

### 2.2 The four things that can "change"

Precision here pays for itself, because the method choice follows from which quantity moved.

| What changes | Formal statement | Typical cause | Cost that detects it |
|---|---|---|---|
| Level | `E[y_t]` steps by `delta` | deploy, price change, policy change | squared error on the mean |
| Scale | `Var[y_t]` steps | instrumentation change, load, mixing | log-variance cost |
| Shape / dependence | autocorrelation or distribution family | batching, caching, retry storm | kernel or spectral cost |
| Trend / slope | drift rate `b` changes | capacity exhaustion, gradual rollout | piecewise-linear cost |

A mean-only cost function is blind to the second and third rows by construction: it standardises away exactly the information that changed. The symptom is a series that obviously looks different to a human, with a segmenter reporting zero change points — and a variance jump that inflates the estimated noise scale, which in turn raises every threshold and hides the level shift too.

### 2.3 The information flow, and where it leaks

```text
raw events --(aggregation window)--> series y_t --(detrend/scale)--> residual
    |                                    |                             |
 event time vs                     sampling cadence,               cost function
 processing time                   missing buckets                 (mean/var/kernel)
    |                                    |                             |
    +--> late arrivals reorder history    +--> bucket boundary artefact |
                                                                        v
                                             segmentation  -->  alarm  -->  review
                                                  |              |            |
                                            penalty beta    threshold h   label, lag
```

Three leaks sit on the left of that diagram and none of them is fixed by a better detector. Event-time versus processing-time mismatch means the change appears at the time the system *noticed* the events, not when they occurred; late arrivals rewrite buckets that a retrospective pass has already segmented; and bucket boundaries that straddle an event create a one-bucket dip that a mean-cost segmenter reads as two changes. Each of these produces a boundary at a suspiciously round clock time, which is the single most useful diagnostic in Section 8.

### 2.4 The invariant that ties both problems together

Every method in this sheet is an instance of one comparison: the fit of a single model on a stretch of data versus the fit of two models split at some point, measured in the same units as the noise, and compared against a threshold that grows with how many splits you were allowed to try.

```text
gain(s)  =  cost(one model on [1..n])  -  cost(two models split at s)
alarm    iff  max_s gain(s)  >  penalty(n, sigma, alpha)
```

The right-hand side is where all the engineering lives. If `penalty` is a constant, the number of spurious segments grows with `n`. If `penalty` grows like `log n`, the number of spurious segments stays bounded. If `penalty` is chosen by looking at the data you then report on, the reported segment count is optimistic and cannot be validated on the same series.

## 3. Foundational Definitions

**Change point (change-point index).** An index `tau` in `{1,...,n-1}` such that the distribution of `y_1..y_tau` differs from that of `y_{tau+1}..y_n` in some stated respect. The definition is incomplete without the stated respect: level, scale, dependence, or full distribution. A change point is not observable; only an estimate of it is.

**Regime (segment).** The maximal contiguous block of observations sharing one parameter value under the model. With `m` change points there are `m+1` regimes. "Regime" is the modelling word; "segment" is the output word; they coincide only when the model is correctly specified.

**Segmentation.** A partition of `{1,...,n}` into contiguous blocks, output by minimising a penalised cost. Distinct from clustering, which ignores order, and from classification, which needs labels. A segmentation of a series with no real change point still returns boundaries — the estimator always returns something, which is why the null behaviour must be characterised before the output is believed.

**Retrospective (offline) detection.** Estimation using the full series, where future observations are available to confirm or refute a candidate boundary. Latency is unbounded by construction. Useful for audits, incident post-mortems, capacity history, and for producing the ground truth against which an online detector is calibrated.

**Online (sequential) detection.** A stopping rule `T` adapted to the filtration generated by `y_1..y_t`. The decision at time `t` may use only `y_1..y_t`. The two error types are asymmetric in cost and are therefore parameterised separately, which is why online detection reports a pair `(ARL0, delay)` rather than a single accuracy number.

**False-alarm rate and ARL0.** `ARL0` is the expected number of observations until a false alarm when no change has occurred. It is a mean, not a rate, and it does not determine the distribution: two designs with the same `ARL0 = 500` can differ by an order of magnitude in the tail. Convert to a rate with `rate = 1/ARL0` alarms per observation, then multiply by the sampling cadence to obtain alarms per hour.

**Average run length under change, ARL1.** The expected delay from the true change point `tau` to the alarm, given a shift of size `delta`. Together with `ARL0` it forms the design pair. Neither number is meaningful alone: a detector with `ARL0 = 10^9` and `ARL1 = 2000` observations is useless, and one with `ARL0 = 50` and `ARL1 = 2` is a siren.

**Detection delay.** The realised quantity `T - tau`, of which `ARL1` is the mean under a stated shift. Report both the mean and a high quantile, because the operationally painful case is the 90th percentile, not the mean.

**Reference value (allowance) `k`.** In CUSUM, the half-shift against which the test is tuned: `k = delta_target / 2` in the units of the series, or `k' = delta_target/(2 sigma)` in standardised units. `k` is a design choice about which shift you care about, not a quantity estimated from data. Setting `k` too large blinds the detector to small shifts; too small makes it slow to accumulate evidence for the shift you actually care about.

**Threshold `h`.** The alarm level for the accumulated statistic, in the same units as `k` times observations. `h` controls `ARL0` and is the only knob that should be used to meet a false-alarm budget.

**CUSUM statistic.** One-sided upward: `S_t = max(0, S_{t-1} + (y_t - mu_0) - k)`. One-sided downward is the mirror. The `max(0, ...)` is the "reset" that makes the statistic forget evidence older than the current excursion, and it is what gives the procedure its sequential optimality. A version without the reset is a cumulative sum, not a CUSUM, and has no finite `ARL0`.

**Page's test / two-sided CUSUM.** The pair `(S^+ , S^-)` with alarm when either exceeds `h`. The two-sided `ARL0` is approximately half the one-sided `ARL0` for symmetric noise, because either of two independent-ish excursions can fire — an approximation, not an identity, and one that degrades when the two statistics are correlated.

**Penalised cost (segmentation objective).** `sum over segments of cost(segment) + beta * m`. The penalty `beta` is in the units of the cost. For squared error with variance `sigma^2`, `beta = 2 sigma^2 log n` is the conservative default and `beta = sigma^2 log n` the one-parameter BIC version. The factor of two is a convention difference, not an error, and it changes the reported segment count.

**Segment cost.** The negative twice log-likelihood contribution of a block: for a Gaussian mean model, `C(a..b) = (b-a+1) * log(2 pi sigma^2) + SSE(a..b)/sigma^2`. Because the first term is linear in the block length, it telescopes across a partition and drops out of the comparison when `sigma` is fixed. With `sigma` estimated per segment it does not drop out, and the segment count changes.

**Optimal partitioning (exact DP).** The dynamic program `F(t) = min over s<t of F(s) + C(s+1..t) + beta`, with `F(0) = -beta`. Exact for additive costs, `O(n^2)` time, `O(n)` memory. This is the reference implementation against which every faster method is checked on short series.

**PELT (pruned exact linear time).** The same recursion with a pruning rule: if `F(s) + C(s+1..t) + beta > F(t)` for some `s`, and the cost satisfies `C(a..t) + C(t+1..u) <= C(a..u)` (a superadditivity condition satisfied by squared error and by negative log-likelihood with fixed parameters), then `s` can never be optimal for any later `t` and is discarded. Average `O(n)`, worst case `O(n^2)`; exact when the condition holds.

**Binary segmentation.** Recursively apply a single-change-point test to the whole series, then to each half. `O(n log n)` and easy to reason about, but it can miss changes that are only visible after a larger change is removed, because the first split is chosen greedily and the residual contrast between two small segments inside the same half may never dominate.

**BOCPD (Bayesian online change-point detection).** Maintains a posterior over the current run length `r_t`: how many observations have elapsed since the last change point. Each step either grows the run (`r_t = r_{t-1} + 1`) or resets it (`r_t = 0`), weighted by a hazard rate and the predictive likelihood. Output is a distribution, not a point, so the "change time" is reported with an uncertainty that a point estimator hides.

**Hazard rate `1/H`.** The prior probability per observation that a change occurs now, given none has occurred yet. `H` is the prior mean regime length in observations. `H` is the single most consequential prior in the model and is almost always set by folk wisdom rather than by measured regime lengths.

**MMD / kernel two-sample cost.** For a segment, the maximum mean discrepancy between the first and second halves under a kernel `k`. Detects changes in distribution rather than only in mean, at the cost of an `O(L^2)` or approximated `O(L)` computation per block and a threshold that depends on the kernel bandwidth.

**Spurious change point.** A reported boundary at which no generating-parameter change occurred. Distinct from a *missed* change point (power failure) and from a *misplaced* one (localisation error). The three have different diagnostics and are often confused in bug reports: a misplaced boundary is usually an aggregation artefact, a spurious one is usually a penalty or autocorrelation problem, and a missed one is usually a cost-function mismatch.

**Segment resolution.** The shortest segment length the method can report, which depends on the cost's effective degrees of freedom and on the penalty. With a per-segment parameter penalty, segments shorter than a few observations are never reported; stating the resolution is part of stating the result, because a segmentation that cannot report a 3-observation segment has not tested for one.

**Event time versus processing time.** Event time is when the observation occurred; processing time is when the pipeline saw it. A change point in processing time that precedes the change point in event time by a fixed offset is a pipeline artefact; the offset is the diagnostic.

**Multiple comparisons in segmentation.** Testing every candidate split at level `alpha` gives `n-1` tests, so the family-wise error rate is at most `(n-1) alpha` by the union bound. This is the entire reason a constant threshold fails and a growing penalty is required; the union bound is loose but adequate for choosing the growth rate.

**Offline-online consistency.** A property of a deployed system: the online detector's alarms should be recoverable by a retrospective segmentation of the same history. If retrospective analysis of a month shows no boundary where the online detector fired twenty times, the online detector is measuring something else — usually cadence, bucket boundaries, or late data.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: one mean shift, known variance, fixed split

The base case exists to make every later formula interpretable as a perturbation of one number. Take `y_1..y_n` independent `N(mu, sigma^2)` up to index `s` and `N(mu + delta, sigma^2)` after, with `sigma` known and `delta` free.

```text
Model:  y_t = mu + delta * 1{t > s} + eps_t,   eps_t ~ N(0, sigma^2) iid
Log-likelihood ratio for split s against no change, delta profiled out:
    Lambda(s) = [ SSE_0 - SSE(s) ] / (2 sigma^2)
Identity (the one that matters):
    SSE_0 - SSE(s) = (n1 * n2 / n) * (ybar_1 - ybar_2)^2,   n1 = s, n2 = n - s

FORMULA [CPD-01]: G(s) = (n1*n2/n) * (ybar_1 - ybar_2)^2
WHERE:  ybar_1 mean of y_1..y_s [series units]; ybar_2 mean of y_{s+1}..y_n;
        n1, n2 counts [observations]; G(s) gain in squared-error units.
CONSTRAINT: sigma known or estimated from within-segment residuals;
        the two segments are independent; the split is fixed before the test.
NUMERICAL EXAMPLE: s = 12, n = 24  ->  n1*n2/n = 12*12/24 = 6.0
        ybar_1 = 100.5, ybar_2 = 110.0  ->  difference 9.5, squared 90.25
        G(12) = 6.0 * 90.25 = 541.5
INTERPRETATION: 541.5 is the squared-error reduction, in squared series units,
        available for buying one extra parameter (the split). Compare it with
        the residual it must beat, not with zero.
```

The identity is worth deriving once because it explains why the gain is quadratic in the shift and linear in the harmonic-mean-ish count. Write `SSE_0 = sum (y_t - ybar)^2` where `ybar` is the grand mean, and `SSE(s) = SSE_1 + SSE_2` with each within-segment sum of squares about its own mean. The difference is the between-segment sum of squares, `n1 (ybar_1 - ybar)^2 + n2 (ybar_2 - ybar)^2`, and substituting `ybar = (n1 ybar_1 + n2 ybar_2)/n` collapses it to `n1 n2/n (ybar_1 - ybar_2)^2`. This is the same one-way ANOVA decomposition, applied to two groups defined by a split point.

Consequences that a practitioner should be able to state without re-deriving:

- The gain is invariant to adding a constant to the series, so de-meaning changes nothing. It is *not* invariant to scaling, which is why the threshold must be expressed in units of `sigma`.
- The gain is symmetric under reversal of the series: `G(s)` reversed is `G(n-s)`. Any online statistic that treats direction asymmetrically is doing something beyond a split test.
- The factor `n1 n2 / n` is maximised at the balanced split, where it equals `n/4`. Edge splits therefore get at most `(n-1)/n * 1` times the balanced coefficient, so a single extreme observation at the edge can still produce a large gain if it is extreme enough. That is the counterexample developed below.

**The threshold.** Under the null, `G(s)/sigma^2` is `chi^2` with one degree of freedom for a fixed `s`, because it is the squared projection of standard normal noise onto a single contrast vector. Taking the maximum over all `s` inflates this: the maximum of `n-1` dependent one-degree-of-freedom statistics grows like `2 log n` in the mean for the penalised formulation.

```text
FORMULA [CPD-02]: beta(n) = 2 * sigma^2 * ln(n)      (conservative default)
                  beta_BIC(n) = sigma^2 * ln(n)      (one-parameter BIC)
WHERE: sigma^2 noise variance [squared series units]; n series length
       [observations]; beta penalty [squared series units].
CONSTRAINT: Gaussian or light-tailed noise; independent observations;
       n large enough that the log-n approximation to the extreme-value
       regime is not nonsense (n >= 20 or so).
NUMERICAL EXAMPLE: sigma = 4, n = 24  ->  ln 24 = 3.1781
       beta     = 2 * 16 * 3.1781 = 101.6977
       beta_BIC =     16 * 3.1781 =  50.8489
INTERPRETATION: a gain of 50 to 100 squared units is the price of admission
       for reporting one boundary at n = 24; 541.5 clears both by 5x and 10x.
```

**Smallest detectable shift.** Setting the balanced-case gain equal to the penalty and solving for the shift converts a statistical threshold into an engineering specification: how big a change will this series length let me see.

```text
Balanced gain: G = (n/4) delta^2 ;  set G = beta = 2 sigma^2 ln n
    =>  delta_min = sqrt( 8 sigma^2 ln n / n ) = sqrt(4 beta / n)

FORMULA [CPD-03]: delta_min = sqrt(8 * sigma^2 * ln(n) / n)
WHERE: delta_min smallest shift detected with even odds [series units];
       sigma noise scale [series units]; n length [observations].
CONSTRAINT: balanced split, iid Gaussian noise, single change, penalty
       at the conservative default; "even odds" is a shorthand, not a power
       calculation — the true power at exactly delta_min is near 0.5.
NUMERICAL EXAMPLE: sigma = 4, n = 24
       delta_min = sqrt(8*16*3.1781/24) = sqrt(406.80/24) = sqrt(16.9496)
                 = 4.1171
       observed delta = 9.5  ->  9.5 / 4.1171 = 2.307 times the floor
INTERPRETATION: with 24 points and noise scale 4 the honest answer to
       "how small a change can you see" is about 4 units; claiming to see
       a 1.5-unit shift from 24 points is claiming a factor of 7.5 lower
       than the floor, which is only possible with borrowed structure.
```

**Boundary and counterexample.** Take the same series and inspect the edge split `s = 1`. There `n1 n2/n = 1*23/24 = 0.9583`; with `ybar_1 = 99` and `ybar_2 = (2526 - 99)/23 = 2427/23 = 105.5217`, the difference is `-6.5217` and `G(1) = 0.9583 * 42.5328 = 40.76`. So a single observation 6.5 units below the rest produces a gain of 40.8 — comparable to the BIC penalty at this length. Now suppose one sensor reports a single dropped decimal, giving `y_1 = 9.9` instead of `99`: the difference becomes `-95.6`, and `G(1) = 0.9583 * 9139.4 = 8758`. The formula is behaving correctly and the inference is nonsense: it is detecting a data-entry error, not a regime. The lesson generalises — at short series the edge gain is dominated by single observations, so either a minimum segment length or a robust cost is mandatory, and neither is in the base-case formula.

Another boundary: **what the formula cannot tell you.** `G(s)` says nothing about the *direction of time*. A level shift and a level shift that reverts after ten observations produce very different gains once you penalise two segments instead of one, and the single-split formula ranks the transient above the durable shift whenever the transient is larger. Reporting a change point from a single split test therefore cannot distinguish a step from a spike, and that ambiguity is resolved only by fitting the two-model alternatives — step (two segments) versus pulse (three segments, with the middle one short) — and comparing penalised costs.

### 4.2 General case: penalised segmentation, PELT, CUSUM, and BOCPD

**The penalised objective.** Given a segment cost `C(a..b)` additively computable in `O(1)` after `O(n)` precomputation, choose the number and location of breakpoints to minimise total cost plus a per-breakpoint penalty.

```text
FORMULA [CPD-04]: minimise over 0 = t_0 < t_1 < ... < t_m < t_{m+1} = n
                  sum_{j=0..m} C(t_j + 1 .. t_{j+1})  +  beta * m
WHERE: C segment cost [log-likelihood units or squared units];
       m number of change points [count]; beta penalty [same units as C].
CONSTRAINT: additive cost; the number of change points is unknown;
       segments must be contiguous and non-empty.
NUMERICAL EXAMPLE: sigma = 4, n = 24, beta = 101.6977
       null partition (m = 0): C(1..24) = RSS_0 / sigma^2 = 730.5/16 = 45.656
       split partition (m = 1 at 12): (189.0/16) + 101.6977 = 11.8125 + 101.6977
                                      = 113.510
       null total = 45.656 ; split total = 113.510  ->  choose m = 0
INTERPRETATION: at the conservative penalty this 24-point series is NOT
       significantly segmented, even though the raw gain is 541.5. The
       constant term (n log 2 pi sigma^2) cancels between partitions, which
       is why the comparison is on 45.656 vs 113.510 and not on raw SSE.
```

That last numerical example is the most instructive line in this sheet, because it looks wrong to anyone who has eyeballed the data. With `sigma = 4` the within-segment residual is 189.0, i.e. `189/22 = 8.59` per degree of freedom, which is *less* than `sigma^2 = 16` — the series is quieter than the assumed noise. The penalty of `2 sigma^2 ln n = 101.7` is therefore calibrated to a noise level larger than the data's, and it refuses a change that a test at the nominal scale would accept. If instead `sigma` is estimated as `sqrt(8.59) = 2.93`, the penalty becomes `2 * 8.59 * 3.1781 = 54.61` and the split total is `11.81 + 54.61 = 66.42` against a null of `730.5/8.59 = 85.04`; now the split wins. The conclusion flips on a variance assumption, which is why "the algorithm found two segments" is never a result without the noise estimate attached.

**PELT recursion and the pruning condition.** Exact optimal partitioning is `O(n^2)`; PELT keeps the same optimum while discarding candidates.

```text
F(0) = -beta
F(t) = min over s in {0..t-1} of [ F(s) + C(s+1..t) + beta ]
pruning: let R_t = { s : F(s) + C(s+1..t) <= F(t) }.
   If C(a..t) + C(t+1..u) + K <= C(a..u) for some constant K >= 0 and all a<t<u,
   and s satisfies F(s) + C(s+1..t) > F(t), then s is never optimal for any u>t.

FORMULA [CPD-05]: prune candidate s at time t when
                  F(s) + C(s+1..t) > F(t) - K
WHERE: F optimal penalised cost of prefix [cost units]; K the pruning
       constant (0 for the standard form); C block cost.
CONSTRAINT: superadditivity of the cost (true for squared error and for
       fixed-parameter Gaussian negative log-likelihood); exactness holds
       only when that condition is met — a robust cost that is not
       superadditive breaks the pruning guarantee.
NUMERICAL EXAMPLE: suppose at t = 24 the optimal cost F(24) = 113.510 and
       candidate s = 3 gives F(3) + C(4..24) = 128.4. Since 128.4 > 113.510,
       candidate 3 is removed from the candidate set for all later t.
INTERPRETATION: pruning is data-dependent, so the average cost is near
       linear on series with few changes and approaches quadratic on series
       that change constantly. Measure the candidate-set size in production;
       it is the only honest predictor of runtime.
```

**CUSUM design: false-alarm budget first.** In the online problem the primary object is the mean time between false alarms. The standard approximation for the one-sided Gaussian CUSUM, in standardised units, is the Siegmund-type expression below; it is accurate to a few percent for moderate `h'` and degrades for very large or very small values.

```text
Standardise: y' = (y - mu_0)/sigma ; shift delta' = delta/sigma
Reference:   k' = delta'_target / 2
Statistic:   S_t = max(0, S_{t-1} + y'_t - k')
Alarm:       S_t > h'

FORMULA [CPD-06]: ARL0 ~ [ exp(2 k' h') - 2 k' h' - 1 ] / (2 k'^2)
WHERE: k' reference in sigma units [dimensionless]; h' threshold in
       sigma units of accumulated standardised deviation; ARL0 in
       observations [count].
CONSTRAINT: iid Gaussian noise; one-sided; k' in (0, ~2); the exponential
       form is an approximation, not an identity.
NUMERICAL EXAMPLE: k' = 0.5 (tuned to a 1 sigma shift), h' = 5
       exp(2*0.5*5) = exp(5) = 148.4132
       numerator = 148.4132 - 5 - 1 = 142.4132 ; denominator = 2*0.25 = 0.5
       ARL0 = 142.4132 / 0.5 = 284.83 observations
       at a 5-minute cadence: 284.83 * 5 = 1424.1 min = 23.74 hours
INTERPRETATION: about one false alarm per series per day. For 500 series
       that is 21.1 alarms per hour before any real change exists, which is
       the number that decides whether the design is viable at all.
```

**The same design at other thresholds**, showing why `h'` is the only honest capacity knob. These five rows are the whole argument for treating the alarm budget as a design input rather than a post-hoc complaint.

| `h'` | `ARL0` (observations) | `ARL0` at 5-min cadence | alarms/hour, 500 series | reviewer-hours/hour at 20 min each |
|---|---|---|---|---|
| 3 | 32.17 | 2.68 h | 186.5 | 62.2 |
| 4 | 97.20 | 8.10 h | 61.7 | 20.6 |
| 5 | 284.83 | 23.74 h | 21.06 | 7.02 |
| 6 | 792.86 | 66.07 h | 7.57 | 2.52 |
| 7 | 2177.26 | 181.44 h | 2.76 | 0.92 |

Every entry above is recomputed from CPD-06, and the reviewer column is `alarms/hour * 20/60`. The last column is the one that kills designs: at `h' = 5` a 500-series fleet consumes seven full-time reviewers on false alarms alone, before a single true change is investigated.

**Detection delay.** The companion number, computed from the same two design constants.

```text
FORMULA [CPD-07]: ARL1 ~ h' / (delta' - k')      for delta' > k'
                  delay_minutes = ARL1 * cadence_minutes
WHERE: delta' true shift in sigma units; k' reference; h' threshold;
       cadence sampling interval [minutes].
CONSTRAINT: delta' > k' strictly; the approximation is first-order and
       optimistic at small delta' - k'; use simulation when the margin
       is under 0.5 sigma.
NUMERICAL EXAMPLE: k' = 0.5, h' = 5, delta' = 1.0
       ARL1 = 5 / (1.0 - 0.5) = 10 observations
       delay = 10 * 5 = 50 minutes
       with sigma = 6 series units, delta = 6.0 units
INTERPRETATION: a 6-unit step is flagged about 50 minutes after it starts,
       on average. The tail matters more than the mean: report the 90th
       percentile from simulation, typically 1.5x to 2x the mean here.
```

**BOCPD: the run-length posterior.** The Bayesian alternative replaces the threshold with a prior on regime length and returns a distribution over the change time. The recursion is one line of conditional probability plus a hazard term.

```text
r_t = run length at time t (observations since last change)
Growth:   p(r_t = r+1 | y_1..y_t) ∝ p(r_{t-1} = r | y_1..y_{t-1}) * (1 - 1/H) * p(y_t | r, params)
Reset:    p(r_t = 0     | y_1..y_t) ∝ [ sum_r p(r_{t-1}=r) * (1/H) ] * p(y_t | prior)

FORMULA [CPD-08]: w(r) ∝ (1 - 1/H) * L_grow(r)  for r >= 1
                  w(0) ∝ (1/H) * L_new
WHERE: H prior mean regime length [observations]; L predictive likelihood
       of y_t under the regime associated with run length r;
       w posterior weight after normalisation.
CONSTRAINT: the hazard is constant in run length (geometric regime length,
       memoryless); the parametric predictive must be conjugate or
       approximated; O(t) work per step, O(t^2) total.
NUMERICAL EXAMPLE: H = 144, so 1/H = 0.006944
       continuing weight (ignoring likelihood): (1 - 0.006944) = 0.993056
       change weight:                            0.006944
       with L_grow = 0.35 and L_new = 0.65 (relative predictive densities)
       w_grow  ∝ 0.993056 * 0.35 = 0.34757
       w_reset ∝ 0.006944 * 0.65 = 0.004514
       normalised w_reset = 0.004514 / 0.352084 = 0.01282
INTERPRETATION: even with the new-regime likelihood nearly twice the old
       one, the posterior probability of a change now is 1.3%. That is not a
       weakness; it is the geometric prior doing its job, and it explains why
       BOCPD outputs must be thresholded on a decision cost, not on 0.5.
```

The practical consequence of CPD-08 is that BOCPD's "change probability" is not comparable across different `H`. Reporting it without stating `H` is reporting a number that cannot be interpreted — the same failure mode as reporting a raw score without its model version.

**MMD cost for distribution change.** When the change is not a mean shift, the mean-cost function is the wrong instrument. The maximum mean discrepancy between two halves of a block gives a segment cost that responds to any distributional difference the kernel can see.

```text
MMD^2(P,Q) = E[k(x,x')] - 2 E[k(x,y)] + E[k(y,y')]
Segment cost  C_MMD(a..b) = - MMD^2( y_a..y_c , y_{c+1}..y_b )  at midpoint c

FORMULA [CPD-09]: MMD_hat^2 = 1/m^2 sum_{i,j} k(x_i,x_j)
                              - 2/(mn) sum_{i,j} k(x_i,y_j)
                              + 1/n^2 sum_{i,j} k(y_i,y_j)
WHERE: k kernel with bandwidth ell [same units as y]; m, n block sizes
       [counts]; the statistic is >= 0 in expectation, and can be slightly
       negative for a finite sample with an unbiased estimator.
CONSTRAINT: iid within block; bandwidth chosen on a held-out stretch;
       the naive estimator is O((m+n)^2) — use random Fourier features for
       long blocks.
NUMERICAL EXAMPLE: ell = 5 units, m = n = 12, a squared-exponential kernel.
       If all cross-kernel terms are 0.42 and all within terms are 0.85 and
       0.80, then MMD^2 = 0.85 - 2*0.42 + 0.80 = 0.81. With the biased
       estimator this is the value; the threshold is set by permutation on
       a null stretch, not by the formula.
INTERPRETATION: MMD responds to scale and shape changes at the same time,
       which is either the point or a bug depending on the question. If the
       question is "did the level move", MMD has strictly worse power than
       a mean test because it spends variance on directions that did not move.
```

### 4.3 Variants

**Variance change.** Replace the mean cost with the negative log-likelihood of a Gaussian with free variance, `C(a..b) = (b-a+1) * log(SSE(a..b)/(b-a+1))`. The penalty must then account for two parameters per segment, and the null distribution is no longer chi-square with one degree of freedom — it is a mixture, because the variance ratio under the null has a point mass at one. This is the single most common source of "significant variance change" findings that do not replicate.

**Piecewise linear / slope change.** Fit `a + b t` per segment and use the residual sum of squares. The gain for a slope change is quadratic in the slope difference but scaled by the *cube* of segment length, so long segments are disproportionately powerful for slope changes and powerless for short ones. A 6-point segment cannot estimate a slope usefully at any noise level.

**Multivariate cost.** For `d`-dimensional observations, `C(a..b) = (b-a+1) * log det Sigma_hat(a..b)`. The penalty must scale with `d`, because each boundary introduces `O(d^2)` parameters in the covariance case. Using a univariate penalty on a 50-dimensional series produces a segmentation with dozens of spurious boundaries, every one of them located at a spike in a single coordinate.

**Robust cost.** Substitute a Huber or quantile loss for squared error. The advantage is bounded influence from single observations, which removes the edge-split pathology in Section 4.4; the disadvantage is that the superadditivity condition required by PELT pruning may fail, so either drop the pruning (losing the speed) or verify exactness against the `O(n^2)` reference on short series.

**Two-stage: coarse then fine.** Run a cheap detector at a coarse aggregation to find candidate regions, then run exact segmentation on a narrow window around each candidate. This is the standard way to get near-`O(n)` behaviour on very long series while keeping exact localisation, at the price of missing changes that vanish under aggregation — a 20-minute spike disappears entirely in a daily mean.

### 4.4 Limits and counterexamples

**Limit 1: the null maximum grows with `n`, so a fixed threshold must eventually fail.** The expected maximum gain under the null grows like `2 sigma^2 log n`; with a fixed threshold `beta_0`, the spurious segment count grows without bound in `n`. Concretely, at `sigma = 4` the null maximum at `n = 24` is about `2*16*3.1781 = 101.7`, and at `n = 1440` it is `2*16*7.2724 = 232.7` — the same series extended 60-fold raises the noise floor by 129 squared units. A penalty copied from a pilot at `n = 24` will report boundaries in every production replay, and the boundaries will look plausible.

**Limit 2: autocorrelation invalidates every threshold in this sheet.** All the `ARL0` and penalty formula above assume independent observations. With AR(1) correlation `rho`, the variance of a segment mean inflates by `(1+rho)/(1-rho)`, so the effective sample size is `n_eff = n (1-rho)/(1+rho)`. At `rho = 0.9` and `n = 1440`, `n_eff = 1440 * 0.1/1.9 = 75.79`, a factor of `sqrt(1440/75.79) = 4.36` inflation in the standard error. Every threshold computed as if `n = 1440` is 4.36 times too permissive. The symptom is a detector that fires at every seasonal trough and every slow drift, and the fix is either thinning, differencing, or a block bootstrap threshold.

**Limit 3: the counterexample where the method is confidently wrong.** A series of 24 observations with a single dropped-decimal error at index 1 gives a gain of 8758 at `s = 1`, far above any penalty, and a segmentation that reports "regime change at the start of the series". Every downstream statistic then compares a mean computed from one bad value against the rest. No penalty can fix this, because the evidence really is in the data; only a robust cost or a range check can. The general form of this failure: the detector is answering the question it was asked, and the question was wrong.

**Limit 4: online and offline thresholds are not interchangeable.** A segmentation penalty controls the expected number of spurious boundaries over one pass. An online threshold controls the expected time to a false alarm. Running a segmentation on a trailing window every minute performs 1440 passes per day, each of which is permitted its own spurious boundary; the daily spurious count is therefore at least an order of magnitude above the per-pass expectation, in addition to the overlap between passes. If the per-pass expectation is 0.05 boundaries and the window is short enough that consecutive passes are nearly independent, 1440 passes yield roughly 72 spurious boundaries per day. That arithmetic alone explains most "the detector is noisy" complaints.

## 5. Worked Numerical Example

**Setup (ILLUSTRATIVE).** Twenty-four consecutive 5-minute observations of a service latency index, in milliseconds above a reference. Values 1–12 come from a stable regime; 13–24 from a shifted one. This is a constructed teaching series, not a measured system.

| idx | y | idx | y | idx | y | idx | y |
|---|---|---|---|---|---|---|---|
| 1 | 99 | 7 | 102 | 13 | 111 | 19 | 115 |
| 2 | 104 | 8 | 100 | 14 | 108 | 20 | 107 |
| 3 | 101 | 9 | 97 | 15 | 114 | 21 | 110 |
| 4 | 96 | 10 | 105 | 16 | 106 | 22 | 113 |
| 5 | 103 | 11 | 99 | 17 | 112 | 23 | 106 |
| 6 | 98 | 12 | 102 | 18 | 109 | 24 | 109 |

**Step 1 — moments.** `sum(y) = 2526`, `mean = 105.25`. `sum(y_1..y_12) = 1206`, `mean = 100.5`. `sum(y_13..y_24) = 1320`, `mean = 110.0`. Difference `= 9.5`.

**Step 2 — total and within sums of squares.** `RSS_0 = sum (y - 105.25)^2 = 730.5`. Splitting at 12: `RSS_1 = 87.0`, `RSS_2 = 102.0`, within total `189.0`. Then `RSS_0 - RSS_within = 730.5 - 189.0 = 541.5`, matching `G(12) = 6.0 * 90.25 = 541.5` from CPD-01. Two routes, one number.

**Step 3 — the gain profile.** Scanning every split, the largest values are the ones that matter; the shape near the true boundary is what localisation depends on.

| split `s` | gain `G(s)` | split `s` | gain `G(s)` |
|---|---|---|---|
| 4 | 132.30 | 14 | 403.24 |
| 6 | 206.72 | 15 | 280.90 |
| 8 | 285.19 | 16 | 285.19 |
| 9 | 396.90 | 11 | 484.88 |
| 10 | 386.79 | **12** | **541.50** |
| 12 | 541.50 | 13 | 440.82 |

The profile is symmetric about 12 only in the sense that the gain falls away on both sides; note `G(8) = G(16) = 285.1875` exactly, which is the reversal symmetry predicted in Section 4.1 and is a useful check that the implementation indexes correctly.

**Step 4 — significance at the estimated split.** Treating `s = 12` as fixed, `G/sigma^2` with `sigma = 4` gives `541.5/16 = 33.844`, and the upper tail of `chi^2_1` there is `5.98e-9`. Correcting for the 23 splits examined by the union bound gives at most `23 * 5.98e-9 = 1.38e-7`. Both are far below any sensible level.

```text
Independent re-check of the F form (does not assume known sigma):
    F = G / (RSS_within / (n - 2)) = 541.5 / (189.0 / 22)
      = 541.5 / 8.5909 = 63.03
    Reference F(1, 22) at 0.001 is about 14.4, so 63.0 clears it 4.4x.
    Equivalently: the implied sigma within segments is sqrt(8.5909) = 2.931,
    versus the assumed 4.0 — the series is 27% quieter than assumed.
```

That discrepancy between assumed `sigma = 4` and empirical `2.93` is the crux of the sensitivity analysis below, and it is the kind of check that separates a reported segmentation from a defensible one.

**Step 5 — sensitivity to the noise assumption.** Recompute the penalty and the decision under three assumptions.

| assumed `sigma` | `beta = 2 sigma^2 ln 24` | split total `RSS_w/sigma^2 + beta` | null total `RSS_0/sigma^2` | verdict |
|---|---|---|---|---|
| 4.0 (stated) | 101.70 | 11.81 + 101.70 = 113.51 | 45.66 | no segmentation |
| 2.93 (empirical) | 54.61 | 22.00 + 54.61 = 76.61 | 85.04 | one segment |
| 6.0 (conservative) | 228.82 | 5.25 + 228.82 = 234.07 | 20.29 | no segmentation |

The verdict flips on an assumption that is usually left implicit. The defensible move is not to pick the assumption that produces the desired answer but to report the decision boundary: the critical `sigma` at which the verdict changes solves `189/sigma^2 + 2 sigma^2 ln 24 = 730.5/sigma^2`, i.e. `541.5 = 2 sigma^4 ln 24`, so `sigma^4 = 541.5/(2*3.1781) = 85.20` and `sigma = 3.037`. Above a noise scale of about 3.04 the conservative penalty refuses this boundary; below it, it accepts. Stating "we detect it for `sigma < 3.04`" is a result; stating "we detect it" is not.

**Step 6 — a distributional check that does not assume a mean model.** Two-sample Kolmogorov–Smirnov on the two halves: the empirical CDFs are completely separated, since `max(y_1..y_12) = 105` and `min(y_13..y_24) = 106`. The KS statistic is therefore `D = 1.0`, against a 5% critical value of `1.36 sqrt(2/12) = 0.555`. For the exact permutation distribution, complete separation can arise in exactly 2 of the `C(24,12) = 2,704,156` equally likely assignments of 12 indices, so `p = 2/2704156 = 7.40e-7`. A test with no free parameters and no Gaussian assumption agrees with Step 4 at a similar order of magnitude.

**Step 7 — online version on the same numbers.** Design a CUSUM for a 6-unit shift with `sigma = 6` (a deliberately conservative noise figure, because the online monitor must use a pre-change estimate and must absorb regime heterogeneity the retrospective pass does not face). Then `delta' = 1.0`, `k' = 0.5`, `h' = 5`.

```text
Standardised increments y'_t - k' with mu_0 = 100, sigma = 6, k = 3:
    t   y    y'      incr    S_t
    1   99    -0.167  -3.167  0.0
    2   104    0.667  -2.333  0.0
    5   103    0.500  -2.500  0.0
    10  105    0.833  -2.167  1.0
    12  102    0.333  -2.667  0.0
    13  111    1.833  -1.167  7.0
    14  108    1.333  -1.667  5.33 -> 5.33
    15  114    2.333  -0.667  4.67
```

The trace above uses the incremental form on raw units, `S_t = max(0, S_{t-1} + y_t - mu_0 - k)` with `mu_0 = 100`, `k = 3`: the increments are `y_t - 103`, giving `0.0` throughout the first regime (every value is at or below 103 except index 10, which pushes `S` to 1.0 and is then reset), and from index 13 the increments are `+8, +5, +11, +3, +9, ...`, reaching `72.0` by index 24. With `h' = 5` in standardised units the raw threshold is `h = h' * sigma = 30`, so the alarm fires at index 17, where `S = 31`, which is 5 observations after the true change at 12. That is consistent with the CPD-07 prediction of `h'/(delta' - k') = 5/0.5 = 10` observations being a rough upper bound and the realised delay being shorter because the actual shift (9.5 units, `delta' = 1.58`) exceeded the design shift of 1.0 sigma.

**Step 8 — the alarm budget consequence.** At `h' = 5`, `ARL0 = 284.83` observations, i.e. 23.74 hours at a 5-minute cadence. A fleet of 500 series generates `500/23.74 = 21.06` false alarms per hour; at 20 minutes per review that is `21.06 * 20/60 = 7.02` reviewer-hours per clock hour. If the desk has 2 reviewers, the design is infeasible before any true positive arrives, and the correct response is to raise `h'` to 6 (2.52 reviewer-hours per hour, still tight) or to pool series before monitoring.

**Step 9 — the conclusion, stated with its conditions.** With `sigma` estimated at 2.93 from within segments, a single change point at index 12 is supported at `p < 1e-6`, with the segment means 100.5 and 110.0. The result is conditional on iid noise (the series is short enough that autocorrelation cannot be estimated), on the absence of a single-observation artefact (checked by the KS complete separation, which requires 12 values above every one of the other 12 — an accident that a dropped decimal would not survive), and on the variance assumption `sigma < 3.04`.

## 6. Visual / Geometric Intuition

### 6.1 The split as a projection

Every split test in this sheet is a projection of the centred data onto a two-dimensional subspace: the constant vector and the step function `1{t > s}`. Least squares on that subspace removes two degrees of freedom; the gain `G(s)` is the length of the projection onto the step direction after the constant is removed. This is why `G(s)/sigma^2` is chi-square with one degree of freedom and not two: the constant is in both models and cancels.

```text
             y (n-dim)
                |
                |   projection onto span{1, step_s}
                v
      ybar*1  +  ((ybar_1 - ybar_2)/2) * (-1,+1 step)
                |                     |
                |                     +-- length^2 = G(s) = n1 n2/n (d)^2
                +-- removed by centring

cos^2(theta_s) = G(s) / RSS_0        (fraction of variation explained)
```

The geometric statement makes the multiple-comparison problem obvious: you are choosing the *best* of `n-1` directions, so the maximum is over a fan of directions that are highly correlated (adjacent splits differ in one observation). The effective number of independent directions is roughly `log n` rather than `n`, which is exactly why the penalty grows like `log n` and not like `n`.

### 6.2 The run-length matrix

BOCPD's state is a distribution over run lengths, and its evolution is best seen as an upper-triangular matrix whose row is time and whose column is run length. At each step, every cell shifts one to the right and a new column zero is born on the diagonal.

```text
time t ->
     r=0   1    2    3    4    5
t=1  w10
t=2  w20  w11
t=3  w30  w21  w12
t=4  w40  w31  w22  w13
t=5  w50  w41  w32  w23  w14
t=6  w60  w51  w42  w33  w24  w15

shift right + down each step; diagonal = "changed at t"
mass concentrating along a column = no change supported
mass piling on the diagonal = change support
```

The mass in column 0 at time `t` is the posterior probability that the last change was at `t`; the mass in column `r` is the posterior probability that the run is `r` long. A clean regime shows the mass marching rightwards as a ridge; a change shows it re-seeding at column 0 and the old ridge dying. Reading the matrix as a picture is faster than reading the probability time series, and it exposes the case where two ridges coexist — an ambiguous change time — which a point estimate hides.

### 6.3 The CUSUM as a ratchet

CUSUM's `max(0, .)` is a ratchet: it accumulates positive deviations and forgets negative ones. Geometrically, the statistic is the height of the maximum drawup in the random walk `sum (y_t - mu_0 - k)`. A run of small positive increments lifts the walk slowly; a step lifts it by the step size; a run of negatives resets it to zero.

```text
S
^                                    /|  alarm h
|                              /----/ |
|                        /----/       |
|            /----------/             |
|      /----/                         |
|  ---/                               |
+---------------------------------------> t
     \        /\    /\                 reset to 0 on a run of negatives
      \______/  \__/  \___
```

The area under the walk between resets is the accumulated evidence, and `h` is the height that must be earned. Two design instincts follow: a larger `k` flattens the drift so only large steps can reach `h` quickly, and a larger `h` raises the bar so smaller steps take longer. Every "the detector is too slow" complaint is a request for a smaller `h`, which is a request for more false alarms, and that trade must be made explicitly.

### 6.4 Where the true boundary sits relative to the estimated one

The estimated boundary is biased towards the middle of the series under the null, because the gain coefficient `n1 n2/n` is maximised there. Under a true change, the estimate is biased towards the change but with a spread proportional to `sigma^2/delta^2`. This means a reported boundary is a point estimate whose uncertainty is asymmetric and depends on the shift size — and it explains why two analysts running different penalties on the same series often report boundaries two or three observations apart and then argue about which is "the" change.

## 7. Comparison / Selection Table

| Method | Problem | Cost per pass | Assumption | Use when | Avoid when |
|---|---|---|---|---|---|
| Shewhart / fixed-threshold | online | `O(1)` per obs | iid, large shift | shift > 3 sigma expected | small persistent shifts |
| CUSUM | online | `O(1)` per obs | iid, known pre-change mean | 0.5–2 sigma persistent shifts | slow drift, unknown baseline |
| EWMA | online | `O(1)` per obs | iid, exponential weighting | gradual drift, smooth signal | abrupt single-sample spikes |
| BOCPD | online | `O(t)` per step | parametric predictive, geometric prior | need change-time distribution | very long series without approximation |
| Optimal partitioning (DP) | retrospective | `O(n^2)` time, `O(n)` memory | additive cost | `n` in the low thousands, exact answer needed | `n > 10^4` |
| PELT | retrospective | `O(n)` average, `O(n^2)` worst | additive + superadditive cost | long series, exact answer needed | non-superadditive robust cost |
| Binary segmentation | retrospective | `O(n log n)` | single dominant change per level | quick first pass, few changes | clustered or small changes |
| Window / sliding test | both | `O(n L)` | local stationarity within window | localising a known change | unknown change count |
| Kernel MMD segmentation | retrospective | `O(n L^2)` naive | iid within block, kernel choice | distributional (not level) change | level-only change, large `n` |
| Robust / quantile cost | retrospective | `O(n)` average with pruning caveat | superadditivity to prune | contamination expected | need exact PELT speed |

**Selection rules that survive contact with production.** If the decision must be made within one sampling interval, only the first three rows are available. If the answer must be exact and `n` is a few thousand, use PELT and verify it against DP on a subset. If the change is suspected to be distributional rather than a level shift, no mean-cost method will find it, and the choice is between a variance cost and a kernel cost. If the series is contaminated, robustness is worth more than exactness, because an exact answer to the wrong question is still wrong.

**A note on the two `O(n)` claims.** PELT's linear average holds when the number of true change points is bounded; with `O(n)` changes the candidate set stops shrinking and the runtime degrades towards quadratic. Binary segmentation's `O(n log n)` counts the cost evaluations but ignores that each level needs a threshold decision, and it makes no exactness claim at all. Comparing the two on speed alone is comparing an exact method with an approximate one, which is a category error unless the approximate method's miss rate is measured.

## 8. Failure Modes & Edge Cases

**F1 — Spurious boundaries from a fixed threshold on a longer series.** Symptom: the segmentation of a 30-day replay reports 40 boundaries, all of them plausible-looking, and the count rises with the replay length. Competing causes: (a) the penalty is a constant copied from a short pilot, so it fails to grow with `log n`; (b) the noise scale is underestimated because it was computed including the changes; (c) the series genuinely has many changes. Discriminating check: recompute the penalty as `2 sigma^2 ln n` with `sigma` estimated from a known-quiet stretch, and count boundaries again. If the count collapses, cause (a). If it collapses only when `sigma` is re-estimated robustly (median absolute deviation), cause (b). If it does not move, inspect the boundary locations for round clock times, which indicates (c) is really an aggregation artefact. Correction: penalty proportional to `log n`; prevention: state the replay length with every segment count; verification: segment a synthetic null series of the same length and confirm the expected boundary count.

**F2 — Autocorrelation masquerading as change.** Symptom: boundaries appear at every seasonal trough and at the start of every slow ramp; the detector fires on smooth series with no step. Competing causes: (a) the threshold assumes iid but the series is AR(1) with `rho` near 0.9, inflating the standard error 4.4x; (b) the aggregation window is longer than the correlation length and creates pseudo-observations; (c) a genuine slow drift is being modelled as a step. Discriminating check: compute the lag-1 autocorrelation of the residuals from the fitted segmentation, and recompute the effective sample size `n(1-rho)/(1+rho)`. If the boundaries survive only when `n_eff` is ignored, cause (a). If they align with the window length, cause (b). If they move earlier as the penalty falls, cause (c). Correction: block bootstrap or permutation thresholds that preserve the dependence structure. Prevention: always report `rho` and `n_eff` beside the threshold.

**F3 — Pipeline change read as a business change.** Symptom: every series in the fleet reports a boundary within the same 24-hour window, or the boundary sits exactly at a UTC day rollover. Competing causes: (a) a deploy changed the event schema, so a field is now null and imputed differently; (b) a sampling cadence change altered the buckets; (c) a genuine fleet-wide event. Discriminating check: the coincidence test — compute the fraction of series whose best boundary falls within a 2-hour window. Under (a) or (b) that fraction approaches one; under (c) it is concentrated only if the event was truly global, and then the magnitude distribution should be homogeneous too. Second check: a boundary at a round clock time is essentially never a business change, because business changes do not respect bucket edges. Correction: segment the residual after removing a common fleet factor; then per-series boundaries are business and the common factor is the pipeline.

**F4 — Mean-only cost missing a scale change.** Symptom: operators insist a segment is different, the segmenter reports one regime, and the mean difference is indeed small. Competing causes: (a) the change is in variance only; (b) the change is in shape with an unchanged mean; (c) the change is real but concentrated in a subpopulation diluted by aggregation. Discriminating check: fit the same segmentation with a log-variance cost and with an MMD cost at the same penalty, and compare the reported boundaries. If variance cost finds a boundary at the operator's time and the mean cost does not, cause (a). If `log det Sigma` in a multivariate version finds it, cause (b). If a per-cohort segmentation finds it while the aggregate does not, cause (c). Correction: use a cost matched to the changed quantity; report which cost was used.

**F5 — Detecting the announcement, not the event.** Symptom: the change point lands at the deploy timestamp minus the propagation lag, and the retrospective and online views disagree by exactly that lag. Competing causes: (a) the series is indexed by processing time, so late-arriving events are attributed to when they were seen; (b) the change point estimator is biased towards the middle of the window; (c) the feature has a genuine delay. Discriminating check: re-key the series by event time and re-run. If the boundary moves by a fixed offset equal to the median ingestion latency, cause (a). If the boundary is stable but interior-biased, cause (b). If neither, cause (c) and the lag is business-real. Correction: always segment on event time, and keep processing time only as a separate diagnostic series.

**F6 — Boundary at the very edge of the series.** Symptom: the reported change is at index 1 or at index `n`, and the segment statistics are computed from a handful of points. Competing causes: (a) a single contaminating observation, as in the `y_1 = 9.9` counterexample, where the gain reaches 8758; (b) a genuine change that occurred just before the series started, so the first segment is a truncated regime; (c) a truncation artefact from a windowing scheme that reuses half-overlapping windows. Discriminating check: winsorise the first and last three observations at the 1st/99th percentile and re-run. If the boundary disappears, cause (a). If it persists and the first segment is much shorter than the others, cause (b) — and the correct report is "change at or before the start of the window". If boundaries are always at fixed offsets from the window ends, cause (c). Correction: minimum segment length; and treat edge boundaries as censored rather than located.

**F7 — Too many segments from double counting in an online replay.** Symptom: an offline segmentation of a month finds two boundaries; the online monitor on the same month fired thirty times. Competing causes: (a) the monitor re-runs a segmentation on a trailing window each minute, multiplying the per-pass expectation by the number of passes; (b) the monitor's threshold was set from the offline penalty, which controls a different quantity; (c) the monitor sees data the offline pass does not, such as partial buckets. Discriminating check: hold out a clean month, run the monitor with alarms suppressed, and count. Compare with `passes * per_pass_expectation`. If the counts match, cause (a) or (b). Then check whether alarm times cluster at bucket boundaries, which indicates (c). Correction: convert the online design to an explicit `ARL0` budget via CPD-06, and verify with a null replay.

**F8 — The ARL0 formula silently invalid on heavy tails.** Symptom: measured false-alarm rate is three to five times the formula's prediction even though the data look stationary. Competing causes: (a) occasional extreme values inflate the statistic faster than Gaussian theory predicts; (b) the pre-change mean `mu_0` is itself estimated and drifts, so the nominal `k` is wrong; (c) the series is a mix of populations, so no single `mu_0` exists. Discriminating check: compute the empirical kurtosis. If it exceeds about 6, cause (a) — the exponential formula assumes Gaussian light tails. If re-centring `mu_0` monthly removes the excess, cause (b). If the histogram is bimodal, cause (c). Correction: simulate `ARL0` under a bootstrap of the observed noise, or use a rank-based / sign-based statistic whose null does not depend on tails.

## 9. Complexity & Cost Analysis

**Computational cost per method.** These are operation counts on cost-function evaluations; a real implementation's constant factor is dominated by the cost's inner loop, so the ratios matter more than the absolute numbers.

| Method | Time | Memory | Evaluations at `n = 1440` | At 1 us each |
|---|---|---|---|---|
| CUSUM / EWMA | `O(1)` per obs | `O(1)` | 1,440 | 1.4 ms |
| Binary segmentation | `O(n log n)` | `O(n)` | ~15,100 | 15 ms |
| PELT | `O(n)` avg | `O(n)` | ~4,300 avg | 4.3 ms |
| PELT worst | `O(n^2)` | `O(n)` | 2,073,600 | 2.07 s |
| Optimal DP | `O(n^2)` | `O(n)` | 1,036,800 | 1.04 s |
| BOCPD | `O(n)` per step | `O(n)` | 1,036,800 total | 1.04 s |
| MMD naive | `O(n L^2)` | `O(L)` | depends on `L` | grows fast |

The PELT average figure assumes a handful of true changes; with a change every 20 observations the candidate set never shrinks and the runtime approaches the worst case. The BOCPD figure is the total over all steps, since each step does `O(t)` work — this is the reason practical BOCPD implementations truncate the run-length distribution at a few hundred states, which is an approximation with a measurable cost in detection delay for long regimes.

**Fleet-level cost.** At `10^4` series, 1440 five-minute points per day, a PELT pass averaging three cost evaluations per observation is `10^4 * 1440 * 3 = 4.32e7` evaluations per day, which is under a minute of single-core time at 1 us per evaluation. Compute is therefore almost never the binding constraint; the binding constraint is the false-alarm review load. Presenting an infrastructure cost estimate without the reviewer cost is the standard way these projects get approved and then fail.

**Reviewer-hour cost is the real budget.** From CPD-06, at `k' = 0.5`:

```text
alarms_per_hour = (series_count / ARL0) * (60 / cadence_minutes)
reviewer_hours_per_hour = alarms_per_hour * review_minutes / 60

For 500 series, cadence 5 min, review 20 min, ARL0 = 284.83:
    alarms/hour = (500 / 284.83) * (60/5) = 1.7555 * 12 = 21.07
    reviewer-hours/hour = 21.07 * 20/60 = 7.02
```

Those 7.02 hours are spent on *false* alarms; true changes consume additional time proportional to their rate. If the desk has 2 reviewer-hours available per clock hour, the design has a 3.5x deficit on noise alone, and no amount of model improvement fixes it — the series must be pooled, the cadence relaxed, or `h'` raised. This is the arithmetic that converts a statistical design into a staffing decision, and it belongs in the same document as the threshold.

**Latency budget.** CPD-07 gives the mean delay; the operational budget must be set on a high quantile. At `k' = 0.5` and `delta' = 1.0`, the mean is `h'/(1 - 0.5)` = 2h' observations; the 90th percentile from simulation is typically 1.5x to 2x that. For a 30-minute response SLA at a 5-minute cadence, the budget is 6 observations, which forces `h' <= 3` at `delta' = 1.0` — and `h' = 3` gives `ARL0 = 32.17`, i.e. 2.68 hours between false alarms per series. A 30-minute detection SLA and a low false-alarm rate are not jointly achievable for a 1-sigma shift on a single 5-minute series; the honest options are to pool series, to lengthen the aggregation window, or to accept a larger shift as the detection target.

**Cost of a missed change versus a false alarm.** If a missed 1-day outage costs `V = 40,000` currency units and a false alarm costs `c = 15` units of reviewer time, then a review is worth doing whenever the probability of a true change exceeds `c/V = 15/40000 = 3.75e-4`. That is a very low bar, which is the quantitative version of the observation that in outage detection the asymmetry between costs makes aggressive alarming rational — provided the reviewer queue has the capacity to absorb it. When capacity binds, the correct objective is not "minimise delay subject to `ARL0`" but "maximise expected value saved per reviewer-hour", which is a ranking problem over series rather than a threshold problem within one series.

## 10. Key Theorems / Results Reference

**Page (1954) — the CUSUM procedure.** Introduces the cumulative-sum control scheme and the interpretation of the statistic as accumulated evidence. Result used here: the recursive form with the `max(0, .)` reset is equivalent to restarting a sequential test after each reset, which is what makes the false-alarm behaviour tractable.

**Lorden (1971) — minimax optimality of CUSUM.** Among all stopping rules satisfying an `ARL0` constraint, CUSUM minimises the worst-case expected delay, where the worst case is taken over the change time and over the pre-change history. This is the formal justification for the CUSUM's privileged position, and it is a *worst-case* statement: a likelihood-ratio rule tuned to a specific shift can beat CUSUM on average at that shift while losing in the worst case.

**Moustakides (1986) — exact optimality.** Sharpens Lorden's result to exact optimality for the simple alternative, removing the slack in the earlier bound. Practical reading: if you know the shift size and the noise family, the CUSUM tuned to that shift is not merely convenient, it is optimal in the minimax delay sense.

**Yao (1988) — consistency of penalised segmentation.** For a penalised least-squares criterion with a penalty that grows like `log n` and a fixed number of change points, the estimated change-point locations converge to the true ones and the estimated count converges to the true count. The rate is `O_p(1)` in localisation under fixed jump sizes. This is the result that licenses the `2 sigma^2 log n` penalty.

**Killick, Fearnhead & Eckley (2012) — PELT exactness and pruning.** Proves that under the superadditivity condition, the pruning step discards only candidates that can never be optimal, so PELT returns the exact optimum of the penalised objective while achieving linear average cost. The condition fails for some robust costs, so the theorem is not transferable to arbitrary cost functions.

**Venkatraman & Olshen (2007) — binary segmentation can be inconsistent.** Constructs configurations in which a change point of a fixed size exists but binary segmentation's probability of detecting it tends to zero, because a larger change in the same region dominates every greedy split. This is the specific result behind the warning not to use binary segmentation for clustered or small changes.

**Barry & Hartigan (1993) — product partition model.** A Bayesian formulation of the segmentation problem with a prior on the number and location of change points, giving a posterior over partitions rather than a single penalised optimum. The practical consequence is that uncertainty in the segment count is available, not just a point estimate.

**Adams & MacKay (2007) — BOCPD recursion.** Derives the run-length posterior recursion used as CPD-08, with the constant hazard giving geometric regime lengths and an `O(t)` per-step update for conjugate models. The approximation cost of truncating the run-length distribution is not in the paper and must be measured per application.

**Gretton et al. (2012) — kernel two-sample test.** Establishes the MMD statistic, its unbiased estimator, and the consistency of the test against any alternative where the kernel is characteristic. The dependence on the kernel choice is the practical weakness: an ill-chosen bandwidth has low power against exactly the change you care about, and the test itself gives no warning.

**Truong, Oudre & Vayatis (2020) — selective review of change-point detection.** Frames the field as a single penalised-comparison template with different costs, and separates offline from online objectives. Useful as the map that makes the rest of this sheet legible; the taxonomy in Section 2.1 follows it.

**The negative result worth memorising.** No method estimates both the number and the locations of change points at a rate that is uniformly good over all configurations: localisation error scales like `1/delta^2`, so arbitrarily small shifts are detected arbitrarily late, and there is no finite-sample procedure that avoids this. Every claimed "we detect any change" system is a claim about the shifts it has seen, and the specification should say so.

## 11. Interview / Exam Questions

**Q1. A 24-point series with `sigma = 4` gives a maximum split gain of 541.5. Is that a change point?** Answer: it depends on the penalty and on whether `sigma = 4` is credible. Compute the penalty `2*16*ln 24 = 101.70`; the gain clears it 5.3x, so under the stated noise model yes. Recompute with the empirical `sigma = 2.93` from within segments and the penalty falls to 54.6, so the verdict strengthens. But if the true noise is 6, the penalty is 228.8 and the split total (234.07) exceeds the null total (20.29) only in the sense of being worse, so the answer is no. Reasoning: the decision is `RSS_w/sigma^2 + beta` versus `RSS_0/sigma^2`, and the sign of that comparison is controlled by `sigma`. Follow-up: at what `sigma` does the verdict flip? Red flag: answering "yes, 541 is large" without naming the penalty or the variance.

**Q2. Derive the smallest shift detectable at `n = 1440` with `sigma = 6`.** Answer: `delta_min = sqrt(8 sigma^2 ln n / n) = sqrt(8*36*7.2724/1440) = sqrt(2094.4/1440) = sqrt(1.4544) = 1.206`. Reasoning: balanced split gives gain `n delta^2/4`; set equal to `2 sigma^2 ln n` and solve. Check: `(1440/4)*1.4544 = 360*1.4544 = 523.6`, and `2*36*7.2724 = 523.6`. Follow-up: what happens with `rho = 0.9`, `n_eff = 75.79`? `delta_min` rises to `sqrt(8*36*ln(75.79)/75.79) = sqrt(8*36*4.328/75.79) = sqrt(16.45) = 4.06`, a 3.4x penalty for dependence. Red flag: using `ln n` with `n` equal to the number of *days* rather than observations.

**Q3. Your CUSUM at `k' = 0.5`, `h' = 5` produces 300 alarms a day across 500 series at a 5-minute cadence. Is the implementation broken?** Answer: not necessarily. `ARL0 = 284.83` observations per series, so 500 series produce `500/284.83 = 1.755` alarms per sampling interval, and there are 288 intervals per day, giving `1.755*288 = 505.5` alarms per day. Observing 300 is *below* the theoretical rate, so the design is behaving as specified. Reasoning: convert `ARL0` to a rate per series per observation, scale by series count and observations per day. Follow-up: to get under 20 alarms per day with 500 series you need `ARL0 >= 500*288/20 = 7200` observations, which means `h'` around 6.9. Red flag: comparing a per-series `ARL0` directly with a fleet-level daily count.

**Q4. Why is `ARL0` not enough to describe a detector's false-alarm behaviour?** Answer: it is a mean, and two designs with the same mean can differ in the tail by an order of magnitude. A design with `ARL0 = 500` and near-geometric run-length distribution has a materially different 99th percentile of inter-alarm time than one with heavier-tailed behaviour, and operations cares about clusters of alarms rather than the long-run average. Reasoning: the mean of a positive random variable does not determine its distribution; false alarms arrive in bursts when the statistic is correlated (as it is under autocorrelation), so the Poisson assumption underlying "alarms per hour" is itself unverified. Follow-up: measure the empirical inter-alarm time distribution on a null replay and check for burstiness. Red flag: treating `1/ARL0` as an independent per-observation probability.

**Q5. Distinguish a misplaced change point from a spurious one, and give one diagnostic for each.** Answer: a misplaced boundary is at the wrong index with the regime count correct; a spurious boundary is at an index where nothing changed, so the count is too high. Diagnostic for misplacement: check whether the boundary is offset from a known deploy by the median ingestion latency — a fixed offset means event-versus-processing-time indexing. Diagnostic for spuriousness: check the boundary's alignment with round clock times and with fleet-wide coincidence, and separately re-run with a penalty that grows as `log n`. Reasoning: the two errors have different causes — localisation error is usually an indexing or aggregation-window problem, spurious counts are usually a penalty or dependence problem. Follow-up: what if the boundary is both misplaced and spurious? Then the segment statistics are meaningless and the run must be discarded rather than adjusted. Red flag: calling both "false positives".

**Q6. Compute the reviewer cost of a design, given `h' = 6`, 500 series, 5-minute cadence, 20 minutes per review.** Answer: `exp(12) = 162754.79`; `2k'h' = 6`; numerator `162754.79 - 6 - 1 = 162747.79`; denominator `0.5`; `ARL0 = 325495.6`? — no, check the exponent: `2 k' h' = 2*0.5*6 = 6`, so `exp(6) = 403.4288`, numerator `403.4288 - 6 - 1 = 396.4288`, `ARL0 = 792.86`. Alarms per hour `= (500/792.86)*12 = 0.6306*12 = 7.567`; reviewer-hours per hour `= 7.567*20/60 = 2.522`. Reasoning: two steps, `ARL0` then rate, then multiply by review time; the trap is miscomputing `exp(2k'h')` as `exp(2 k' h')` with `h'` not multiplied by `k'`. Follow-up: how much capacity would halving the review time free? It halves the reviewer-hours to 1.26, which is a cheaper intervention than raising `h'` further, since raising to `h' = 7` costs 3.6x in detection delay. Red flag: computing the exponential as `exp(6)/0.25` by using `2k'^2 = 0.5` incorrectly as `0.25`.

**Q7. A team segments a series of 1440 points, gets 40 boundaries, and reports "the system is unstable". What is wrong with the inference?** Answer: 40 boundaries in 1440 points means a mean regime length of 36 observations; at a 5-minute cadence that is 3 hours, which is almost certainly below the physical regime length of whatever is being measured. The prior on regime length is being contradicted, so either the penalty is too low or the noise scale is underestimated or the series is dependent. Reasoning: the segment count implies a regime length, and that length can be compared with domain knowledge before any statistics are computed. Follow-up: recompute with `2 sigma^2 ln n` using a robust `sigma`; if the count collapses, the original `sigma` was inflated by the changes themselves. Red flag: treating a large segment count as evidence of instability rather than as evidence of a mis-specified penalty.

**Q8. Explain why binary segmentation fails on two adjacent small changes preceded by one large change.** Answer: the first split is chosen at the largest contrast, which is the large change. That split partitions the series such that the two small changes fall in the same half, and within that half their combined contrast must now exceed the penalty on its own. If each is a 1-sigma shift, their combined gain within the half is at most `n_half * (2 sigma)^2/4 = n_half sigma^2`, which for `n_half = 100` is `100 sigma^2` — larger than the penalty `2 sigma^2 ln 100 = 9.2 sigma^2`, so this particular case actually succeeds. The failure needs the small changes to be closer together in a shorter half: with `n_half = 10` and each a 0.5-sigma shift, the combined gain is `10*(1 sigma)^2/4 = 2.5 sigma^2`, below `2 sigma^2 ln 10 = 4.6 sigma^2`, so they are missed. Reasoning: the greedy split changes the sample size available for the remaining tests, and the penalty does not shrink accordingly. Follow-up: what fixes it? PELT/DP on the full series, or binary segmentation with a penalty scaled to the current segment length. Red flag: saying binary segmentation "is always consistent".

**Q9. Your null replay shows the measured false-alarm rate is 4x the formula. Name three causes and the check for each.** Answer: (a) heavy tails — check the empirical kurtosis, above about 6 invalidates the exponential approximation; (b) `mu_0` estimated and drifting — check whether monthly re-centring removes the excess; (c) population mixing — check the histogram for bimodality, which means no single `mu_0` exists. Reasoning: the formula is an iid-Gaussian result and each of the three violates a different assumption: tail shape, parameter stationarity, and homogeneity. Follow-up: the robust fix is a bootstrap of the statistic under the observed noise, which preserves tails and dependence simultaneously. Red flag: "the implementation has a bug" as the only hypothesis.

**Q10. A series is segmented at index 1 with a gain of 8758. What is the most likely explanation and what is the correct action?** Answer: a single contaminating observation at index 1. The edge coefficient is `n1 n2/n = 23/24 = 0.958`, so a single value 95 units from the rest yields `0.958*9139 = 8758`. The correct action is not to raise the penalty — the evidence is genuine under the model — but to range-check the observation, winsorise or drop it, and re-run. Reasoning: the gain formula weights the edge by 0.958, which is only 6.5x below the maximum coefficient of 6.0, so a 2.5x larger deviation at the edge beats a balanced split of a modest shift. Follow-up: what structural fix prevents recurrence? A minimum segment length of 3 plus a robust cost, because a minimum length alone does not stop a 3-point edge segment built from one bad value. Red flag: declaring the data "anomalous" and keeping the segmentation.

**Q11. Give the condition under which PELT is not exact.** Answer: when the segment cost is not superadditive in the sense `C(a..t) + C(t+1..u) + K <= C(a..u)`, the pruning rule can discard a candidate that would have been optimal later. Squared-error and fixed-parameter Gaussian log-likelihood costs satisfy it; some robust and rank-based costs do not. Reasoning: the pruning theorem is conditional on that inequality, and discarding a candidate is irreversible within a pass. Follow-up: how do you detect the problem in practice? Run the `O(n^2)` DP and PELT on 200 random short series and compare the optima; any disagreement proves the condition fails for your cost. Red flag: "PELT is exact" without naming the cost.

**Q12. An analyst sets `k` to the shift they want to detect. What is wrong?** Answer: `k` should be half the target shift. The CUSUM drift under a shift `delta` is `delta - k`, so `k = delta` gives zero drift and infinite delay — the detector is tuned to the boundary of its own detection region. `k = delta/2` maximises the drift-slack trade-off and is the standard choice. Reasoning: substitute into the drift expression; at `k = delta` the expected increment is exactly zero, so the walk is a martingale and `h` is never reached on average. Follow-up: what if you care about a range of shifts? Use two CUSUMs in parallel, one at `k = delta_small/2` and one at `k = delta_large/2`, and combine the alarms, accepting the doubled false-alarm rate. Red flag: setting `k` from the data's observed shift after the fact.

**Q13. Compute the daily spurious boundary count for a segmentation re-run every minute on a 1440-point trailing window, given a per-pass expectation of 0.05.** Answer: 1440 passes per day times 0.05 = 72 boundaries per day, minus overlap effects. Consecutive passes share 1439 of 1440 points, so the passes are not independent and the true count is lower than 72 but still of that order — the overlap concentrates repeated detections of the same boundary rather than suppressing them, and a deduplication step by boundary index reduces the *reported* count without reducing the underlying instability. Reasoning: the per-pass expectation applies to each pass; re-running the same test many times multiplies the opportunity. Follow-up: the correct architecture is a sequential test with an `ARL0` budget, not a repeated batch segmentation. Red flag: multiplying by passes and then dividing by two "because of overlap" with no measurement.

**Q14. What does BOCPD's change probability of 1.3% mean, and why is 0.5 the wrong threshold?** Answer: it means that, under the geometric prior with `H = 144`, the posterior mass on "the run length reset at this observation" is 0.013. The prior puts only `1/144 = 0.69%` mass on a reset before any evidence, so a likelihood ratio of 1.86 lifts it to 1.3% — a factor of 1.86, which is weak evidence. Thresholding at 0.5 would require a likelihood ratio around 140, so it would essentially never fire. The threshold should come from a decision cost: alarm when `p * value_of_detection > cost_of_false_alarm`. Reasoning: the posterior is prior-dominated at short run lengths, and the hazard is the prior. Follow-up: changing `H` to 24 multiplies the reset prior by 6 and rescales every reported probability; the number is not comparable across `H`. Red flag: reporting a BOCPD change probability without `H`.

**Q15. When is a kernel MMD segmentation strictly worse than a mean-shift segmentation?** Answer: when the change is a pure location shift and the kernel bandwidth is not matched to it. MMD spends its power over all directions in the feature space, so for a mean shift it has strictly lower power than the targeted mean test at the same false-alarm rate, in the same way a two-sided omnibus test is less powerful than a one-sided targeted test. Concretely, with a squared-exponential kernel of bandwidth much larger than the shift, the MMD signal is second-order in `delta/ell` and vanishes relative to noise. Reasoning: the MMD is a smoothed function of the characteristic-function difference, and its sensitivity depends on the kernel's Fourier decay. Follow-up: how do you choose the bandwidth? Median-heuristic on a held-out stretch, then verify power by permutation on a synthetic shift of the size you care about. Red flag: using MMD because it is "more general" without a power comparison.

## 12. Quick Reference Card

| Item | Value / rule |
|---|---|
| Gain for a split | `G(s) = (n1 n2/n)(ybar_1 - ybar_2)^2`; equals the between-segment sum of squares |
| Conservative penalty | `beta = 2 sigma^2 ln n`; BIC variant is `sigma^2 ln n`; report which convention |
| Smallest shift at length `n` | `delta_min = sqrt(8 sigma^2 ln n / n)`, balanced split, ~50% power |
| CUSUM reference | `k = delta_target/2`; never `k = delta_target` |
| CUSUM false alarms | `ARL0 ~ [exp(2k'h') - 2k'h' - 1]/(2k'^2)`; convert to rate with `1/ARL0` |
| CUSUM delay | `ARL1 ~ h'/(delta' - k')` observations; report the 90th percentile too |
| Fleet alarm load | `alarms/h = (series/ARL0)*(60/cadence)`; reviewer cost is the binding constraint |
| Dependence correction | `n_eff = n(1-rho)/(1+rho)`; 4.36x standard-error inflation at `rho = 0.9` |
| Verdict flip condition | State the critical `sigma` at which the segmentation decision changes |
| Never report | a boundary count without `sigma`, penalty convention, `n`, and cadence |
