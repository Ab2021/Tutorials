> Module 04 — Statistics and Causal Inference
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.01, 01A.11

# Probability, Sampling, Bootstrap, and Intervals

---

## 1. Why this topic exists

Every number a supply chain team reports is an estimate wearing the clothes of a measurement. The
mean lead time of 24.2 days is computed from 36 orders. The fill rate of 94.6% is computed from 12
weeks. Each number has a procedure behind it and a spread around it, and the spread usually decides
whether anyone should act.

Three failures recur, and all three are failures of uncertainty rather than of arithmetic.

**One. The promise date built on a mean.** A lane's mean lead time is 24.2 days over 36 weeks. The
commercial team promises 26 days and treats the gap as a buffer. The 95% confidence interval for
the mean is about [23.1, 25.4] days, which looks comfortable. The 95% prediction interval for the
*next order* is about [17.6, 33.3] days, which does not. The team promised against the second while
quoting the first, roughly one order in five arrives outside the promise, and the post-mortem
blames the supplier.

**Two. The supplier ranking that is mostly noise.** A team ranks 40 suppliers by mean lead time.
The top averages 21.4 days on 7 orders, the bottom 27.9 on 9. Within-supplier standard deviations
are around 6 days, so each mean has a standard error near 2.1 days. The 6.5-day gap is about one and
a half standard errors of the difference: a plausible ordering of the true means and a poor
ordering of the observed ones, on which the team proposes to move volume.

**Three. The improvement that is inside the noise.** A distribution centre reports fill rate
rising from 94.1% to 94.6% after a process change, on 12 weeks either side. The interval around
each estimate is about plus or minus 1.3 points at 95%. The change is indistinguishable from zero,
the programme is declared a success, and the next two quarters give the gain back.

None of these is a modelling failure. In each case the estimate is correct and the uncertainty is
missing, mis-specified, or attached to the wrong object.

**Why the standard toolkit does not settle it.** The textbook interval is
`estimate ± 1.96 × standard error`, with standard error `s / sqrt(n)`. That rests on independent
observations and a sampling distribution close enough to normal for the multiplier 1.96 to be
right. Supply chain data violates both, in the same direction.

- Lead times are right-skewed and floored near zero; the normal interval runs below zero.
- Daily demand has a point mass at zero, so a mean and a standard deviation describe it badly.
- Observations are ordered in time; reshuffling them destroys the structure that makes the mean
  uncertain.
- Observations are nested; ten orders from one supplier are one supplier observed ten times.

The consequence is directional and therefore dangerous: the naive interval is too narrow, the
reported confidence is too high, and the error is invisible because nothing in the output flags it.

**What this topic does about it.** It gives you the bootstrap, which replaces a distributional
assumption with computation, and a taxonomy: a **confidence interval**, a **prediction interval**,
and a **tolerance interval** answer three questions, and reporting uses whichever one the software
produced.

**The claim.** An interval is a property of a procedure, not of a dataset, and it is only as
trustworthy as the resampling scheme's fidelity to how the data were generated. The number of
resamples is the least consequential choice you make. The resampling unit is the most
consequential, and the one most often left implicit.

**Where this sits.** Topic 01 fixed the estimand. This topic attaches uncertainty to the estimate
of that estimand. Topic 03 turns intervals into evidence, Topic 05 repairs regression standard
errors under autocorrelation, and Topic 07 narrows intervals by design rather than by arithmetic.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** what a 95% interval means operationally, in terms of repeated procedures, and
   **name** three things it does not mean.
2. **Distinguish** confidence, prediction, and tolerance intervals, and **select** the correct one
   for a stated supply chain decision.
3. **Explain** the plug-in principle, and **describe** what the bootstrap substitutes for the
   unknown distribution.
4. **Test** the three conditions a bootstrap needs — exchangeable resampling units, enough
   distinct values, and a statistic that is not tail-extreme — and **predict** the direction of
   failure when each is violated.
5. **Implement** an iid bootstrap, a moving block bootstrap, and a cluster bootstrap, and
   **justify** the block length and the clustering level.
6. **Compute** percentile and bias-corrected and accelerated (BCa) intervals, and **explain** when
   the percentile interval undercovers.
7. **Quantify** the finite population correction and the design effect, and **state** when each
   materially changes a reported interval.
8. **Diagnose** a reported interval that is too narrow, too wide, or answering the wrong question,
   and **rewrite** the claim so that it is defensible.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold four pictures: a procedure that produces intervals, a finite population of item-weeks, a
resampling device that mimics the data-generating process, and three interval types that look
alike in a report and answer different questions.

### 3.1 An interval is a procedure, not a statement about your data

The frequentist definition is a statement about repetition. Build the interval the same way on
many independent samples. About 95% of those intervals contain the true value. Your interval is
one draw from that collection. It either contains the value or it does not; the 95% describes the
procedure, not this instance.

This sounds like pedantry until it changes a decision. "There is a 95% chance the true lead time is
between 23 and 25 days" invites a planner to treat 25 as a bound. "The procedure that produced
[23, 25] captures the true mean in 95% of repeated studies, and this lane's true mean may sit
outside it" invites the question the planner should ask: what happens if it does.

> **Note:** A Bayesian credible interval does support the sentence "there is a 95% probability the
> parameter lies in this interval", given a prior and a model. The two intervals coincide
> numerically under a flat prior and a normal model, and diverge otherwise. If a report says
> "95% probability" and the method was frequentist, the sentence is wrong even when the number is
> right.

### 3.2 The picture to redraw by hand

```text
  THE SAMPLING DISTRIBUTION, AND THE INTERVAL IT PRODUCES

     density
        |                    .-'''-.          <- distribution of the ESTIMATOR
        |                  .'        '.         over hypothetical repeated
        |                 /            \        samples of the same size
        |                /              \
        |               /                \
        |              /                  \
        +-----[========|==========|========]----> estimate
               a        ^          ^       b
                      2.5%      97.5%
                      <---- 95% ---->

     A 95% interval is the middle 95% of THIS distribution.
     It is a property of the PROCEDURE. Your data gave one draw.

  WHAT IT DOES NOT SAY
    1. "There is a 95% chance the true value is inside this interval."
       The value is fixed; the interval is what moves between studies.
    2. "95% of lead times fall inside this interval."
       That is a prediction interval, and it is much wider.
    3. "If we recompute next month we will get an interval like this."
       Only if next month is exchangeable with this sample. A supplier
       change, a port closure, or a demand regime shift breaks that.

  AND THE PART THAT DECIDES THE WIDTH
       width  ~  spread of the data / sqrt(n)        for independent data
       width  ~  spread x sqrt((1+rho)/(1-rho)) / sqrt(n)   for AR(1) data
       width  ~  unchanged by n when the unit is the cluster, not the row
```

The last three lines are the whole topic in miniature. The width is set by the spread, the sample
size, and the dependence structure. The bootstrap estimates the width without a formula, but it
estimates the width of whatever structure you resampled. Resample the wrong structure and you get
a confident, precise, wrong number.

### 3.3 The population is often finite, small, and already enumerated

The textbook sampling picture assumes a large or infinite population and a small sample. Supply
chain analytics usually works the other way round: you have all 104 weeks of a lane's history, or
38 of the 42 distribution centres. The "population" is a list you can count.

That changes the question. There are two different targets, and they need different intervals.

- **A descriptive target.** The mean lead time *of these 36 weeks*. If you have every week, there
  is no sampling error, only measurement error; resampling answers a different question.
- **A predictive target.** The mean lead time *next quarter*. The 36 weeks are now evidence about a
  process, and the uncertainty has two parts: the process is not fully known, and next quarter may
  differ from its average.

When the sample is a substantial fraction of a finite frame, the sampling variance shrinks by the
**finite population correction**:

```text
  Var(mean) = (s^2 / n) x (N - n) / (N - 1)

  N = frame size (weeks in the frame, lanes in the category)
  n = units sampled
  the factor is 1 when n is small relative to N, and 0 when n = N
```

For 40 lanes drawn from a frame of 60, the factor is `(60 - 40) / 59 = 0.339`, and the standard
error is 58% of the uncorrected value. Reporting uncorrected is conservative, but it can make a
decision look less certain than the evidence warrants when 40 of 60 lanes is most of the frame.

> **Note:** The correction assumes simple random sampling from the frame. If the 40 lanes were
> chosen because they are the high-volume ones, the frame is not the 60 lanes and the correction
> does not apply. State the frame before applying the factor.

### 3.4 The bootstrap replaces the unknown distribution with the observed one

The plug-in principle is one sentence: if you do not know the distribution `F` that generated the
data, use the empirical distribution `F_hat`, which puts probability `1/n` on each observation.

```text
  TARGET        the sampling distribution of theta_hat under F
  SUBSTITUTE    the sampling distribution of theta_hat under F_hat
  MECHANISM     draw n values with replacement from the observed n,
                recompute theta_hat, repeat B times
  RESULT        B values whose spread estimates the spread of theta_hat
```

**The bootstrap can only reproduce values it has seen.** Every resample is built from the observed
values, so it cannot generate a lead time longer than the longest observed one. For the mean this
is harmless; for a 99th percentile it is fatal, as section 6.7 shows numerically.

**The bootstrap inherits the dependence structure of the resampling unit.** Resample single rows
from an autocorrelated series and each resample is a shuffled series with no serial dependence. The
bootstrap distribution is then the one you would see if the data were independent: narrower than
the truth.

### 3.5 Three conditions, and what each one buys

| Condition | What it means | What breaks it | Direction of failure |
|---|---|---|---|
| Exchangeable resampling units | Any unit could have been any other unit; the resample is a plausible redraw of the data-generating process | Time order, clustering, nesting, selection | Interval too narrow; confidence overstated |
| Enough distinct values | The observed values span the region the statistic depends on; the empirical distribution is not degenerate | Heavy ties, few observations, coarsely binned data | Interval endpoints jump between a few values; coverage erratic |
| A statistic that is not tail-extreme | The statistic depends on the body of the distribution, not on the largest or smallest observations | Maximum, minimum, 99.9th percentile, tail index, value at risk | Bootstrap distribution is defective; no amount of `B` repairs it |

The three conditions are not equally familiar. The first is discussed in every tutorial, the second
is a small-sample caveat. The third is usually omitted, and it is the one that silently produces
nonsense in supply chain work, because the quantity planners care about — the lead time you have to
protect against — is exactly a tail quantity.

### 3.6 Three intervals that look alike in a report

| Interval | Target | Narrows as n grows? | Typical supply chain use |
|---|---|---|---|
| Confidence interval (CI) | A parameter of the process, such as the mean lead time | Yes, proportional to `1/sqrt(n)` | "How well do we know the average?" |
| Prediction interval (PI) | A future observation, such as the next order's lead time | Barely; it converges to the spread of the process | "What range should I plan a single order against?" |
| Tolerance interval | A stated proportion of the process, with stated confidence | Slowly; it needs large n to be useful distribution-free | "What range contains 90% of lead times, with 95% confidence?" |

The commonest reporting error in supply chain analytics is a confidence interval presented where a
prediction interval is needed. It is not small: in section 6.6 the confidence interval is 22.7 days
wide and the prediction interval 68.1, for the same eight observations.

## 4. Definitions and vocabulary

The fourth column names the unit or the coding convention. A definition without one is the defect
this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Estimand | The precise quantity to be estimated, fixed before analysis: population, unit, outcome, window, comparison | Used as a synonym for "metric" | Statement, not a number |
| Estimator | The procedure that maps data to an estimate, such as the sample mean | Confused with the estimate it produces | Function of the data |
| Standard error (SE) | The standard deviation of the estimator's sampling distribution | Called "standard deviation", which is a property of the data | Units of the estimand |
| Sampling distribution | The distribution of an estimator over repeated samples of the same size and design | Confused with the distribution of the data | Units of the estimand |
| Coverage | The long-run fraction of intervals, built by a stated procedure, that contain the true value | Reported as a property of one interval | Proportion in [0, 1] |
| Nominal coverage | The coverage the procedure claims, such as 95% | Assumed to be achieved | Proportion |
| Actual coverage | The coverage the procedure delivers under the true data-generating process | Never checked | Proportion |
| Confidence interval (CI) | An interval for a parameter with stated nominal coverage under repeated sampling | Interpreted as a probability statement about the parameter | Units of the parameter |
| Prediction interval (PI) | An interval for a future observation, or a function of future observations | Used interchangeably with a CI | Units of the observation |
| Tolerance interval | An interval intended to contain at least a stated proportion of the population, with stated confidence | Called a prediction interval, or a CI | Units of the observation |
| Percentile interval | A bootstrap interval taking the `alpha/2` and `1 - alpha/2` quantiles of the bootstrap distribution | Used at small `n` without a coverage check | Units of the estimand |
| BCa interval | Bias-corrected and accelerated bootstrap interval, adjusting for median bias and skewness | Applied without enough resamples for the adjusted tail | Units of the estimand |
| Bootstrap distribution | The distribution of the statistic over `B` resamples | Treated as the posterior or as the sampling distribution itself | Units of the estimand |
| Resampling unit | The object drawn with replacement: row, block, or cluster | Left implicit | Identifier, such as `order_id` |
| Exchangeability | The property that the joint distribution of the units is unchanged by permutation | Assumed rather than argued | None; a property of the design |
| Independent and identically distributed (IID) | Units drawn independently from one common distribution | Asserted for time-ordered data | None |
| Block bootstrap | Resampling contiguous blocks to preserve within-block dependence | Applied with a block length chosen by habit | Block length in periods |
| Moving block bootstrap | Blocks of fixed length `l` drawn from all `n - l + 1` start positions | Blocks allowed to wrap without saying so | `l` in periods |
| Stationary bootstrap | Blocks of random length, geometric with mean `l`, wrapping at the end | Preferred without justification; it is not always better | Mean block length in periods |
| Block length | The number of consecutive periods resampled together | Chosen as `n/10` with no rationale | Periods |
| Cluster bootstrap | Resampling whole clusters, such as items or suppliers, rather than rows | Resampling rows and calling it clustered | Cluster identifier |
| Finite population correction (FPC) | The factor `sqrt((N - n)/(N - 1))` applied when sampling a large share of a finite frame | Applied to a convenience sample | Dimensionless factor |
| Design effect (DEFF) | The ratio of the variance under a complex design to the variance under simple random sampling | Ignored when rows are clustered | Dimensionless, at least 1 |
| Effective sample size | `n / DEFF`, the sample size with equivalent independent information | Reported as the raw row count | Count |
| Intra-cluster correlation (ICC) | The correlation between two observations in the same cluster | Assumed zero by default | Dimensionless in [-1, 1] |
| Plug-in principle | Substituting the empirical distribution for the unknown population distribution | Treated as assumption-free | None |
| Smooth functional | A statistic whose distribution changes gradually with the data, such as a mean or a ratio | Assumed for maxima and extreme quantiles | None |
| Extreme quantile | A quantile in the region where the empirical distribution has few or no observations | Estimated by the bootstrap | Units of the variable |
| Subsampling | Drawing subsets of size `m < n` without replacement, valid under weaker conditions than the bootstrap | Unknown in practice | Subset size `m` |
| Monte Carlo error | The error in a bootstrap result caused by finite `B` | Reported to three decimals | Standard error of the bootstrap quantity |
| Jackknife | Leave-one-out recomputation, used for bias and for the BCa acceleration constant | Confused with the bootstrap | Units of the estimand |

Five of these carry most of the practical damage.

**Coverage versus confidence.** Coverage is a frequency: the fraction of intervals that work.
Confidence is the label attached to one interval. A procedure with 95% nominal coverage can have
85% actual coverage on your data, and the label will still say 95%.

**Prediction versus tolerance.** A prediction interval covers one future observation with stated
probability. A tolerance interval covers a stated *proportion* of the population with stated
confidence. A 95% prediction interval and a 90%/95% tolerance interval are different objects with
different widths, and they coincide only in special cases. A `beta`-expectation tolerance interval
is mathematically the same as a prediction interval for one observation; a `beta`-content,
`gamma`-confidence tolerance interval is wider and is what most standards mean by "tolerance
interval". Say which.

**Block versus cluster.** Blocks handle dependence *along an index*, usually time. Clusters handle
dependence *within a group*, usually a hierarchy. Data that is both — weekly orders nested in
suppliers — needs the cluster level to be resampled and, if the within-cluster series is itself
autocorrelated, blocks within clusters. Resampling blocks of rows across suppliers is a common and
wrong hybrid.

**Percentile versus BCa.** The percentile interval is the simplest and the least accurate at small
`n` and at skew. BCa corrects for median bias and for the change of variance with the parameter.
BCa needs more resamples to estimate its adjusted tail probabilities and can behave badly when the
acceleration is large. If the two disagree materially, that disagreement is information about
skew, not a bug.

**Exchangeability versus independence.** Independence implies exchangeability; the converse is
false. Exchangeability is the weaker and more honest assumption: it says the order of the units
carries no information. For a shuffled iid sample that is true. For a lead-time series in which
this week resembles last week, it is false, and no amount of resampling repairs it.

## 5. The mechanics, step by step

The procedure has four phases and thirteen steps. Each step names what you do, what you check, and
what fails if you skip the check.

### Phase A — Fix the target and the resampling unit

**Step 1. Write the estimand in one sentence, with its unit.** "The mean lead time, in days, for
lane Rotterdam–Chicago, over orders placed in 2025." Check for population, unit, outcome, and
window. Failure: a precise interval for the 95th percentile when the decision needs the mean.

**Step 2. Name the data-generating unit and the independence unit.** The row might be an order
line, an order, a shipment, or an item-week; the independence unit is what you would treat as
exchangeable, and two orders from one supplier in one month are not. Failure: 4,800 order lines
treated as 4,800 facts.

**Step 3. Draw the dependence map.** Is the series autocorrelated? Is there a grouping, and at what
level was treatment assigned? Are groups crossed, as when an order belongs to a supplier and a
customer? Failure: choosing a scheme before knowing the structure.

**Step 4. Choose the resampling scheme that preserves that structure.**

| Structure | Resample | Do not resample |
|---|---|---|
| IID rows | Rows with replacement | — |
| Autocorrelated series | Contiguous blocks of length `l` | Single rows |
| Nested hierarchy, one level | Whole clusters at the level of assignment | Rows |
| Nested hierarchy, two levels | Whole top-level clusters, then inner units within them | Rows, or inner units alone |
| Crossed groups | One grouping factor at a time, or a multiway scheme | Either factor alone |
| Panel of items over weeks | Items, if the item is the unit; item-weeks if within-item variation is the target | Rows across items |

### Phase B — Check the three conditions before trusting any output

**Step 5. Argue exchangeability, do not assert it.** Ask what process produced this sample and
whether a redraw would look like it. Check selection (only surviving suppliers), censoring
(stock-outs truncating demand), and regime change (a new carrier). Failure: an interval describing
a world that no longer exists.

**Step 6. Count distinct values and the smallest relevant cell.** Report distinct observations,
ties, and the count in the tail region the statistic uses: twenty observations with five distinct
values are effectively a sample of five for any quantile. Failure: endpoints that are two of five
possible values, reported to one decimal.

**Step 7. Classify the statistic as smooth or tail-extreme.** Means, totals, ratios, regression
coefficients, and correlations are smooth. Maxima, minima, extreme quantiles, and tail indices are
not; for those the bootstrap is the wrong tool, as section 6.7 shows. Failure: a 99th-percentile
"confidence interval" whose upper endpoint is the largest observation ever seen.

### Phase C — Run, diagnose, and choose the interval

**Step 8. Choose `B` from the precision you need, not from the default.** The Monte Carlo standard
error of an estimated tail probability `p` is `sqrt(p(1-p)/B)`: 0.0049 at `B = 1,000`, 0.0016 at
`B = 10,000`, 0.0010 at `B = 25,000`. Failure: `B = 200` endpoints reported to three decimals,
where the last two digits are noise.

**Step 9. Compute at least two interval types and compare them.** Percentile and BCa. If they
agree, report the simpler. If they differ by more than the Monte Carlo error, the bootstrap
distribution is skewed or biased, and the wider interval is usually the more honest. Failure: one
method, because it is the library default.

**Step 10. Check coverage where you can.** In descending order of cost: simulation from a plausible
generative model; split-sample comparison; a literature benchmark. At small `n` this is not
optional: in section 6.5 a nominal 95% percentile interval at `n = 8` covered 85%.

### Phase D — Report as a decision input

**Step 11. Find the decision threshold and say which side of it the interval falls on.** "The
reorder point for 95% service is 31 units; the interval for the true 95th percentile of lead-time
demand is [26, 44], so 31 units works only if the low end is right." A number without a threshold
is decoration.

**Step 12. Report the method, the unit, the `n`, and one alternative.** Estimate, interval, method,
resampling unit, `B`, and what the interval would be under one defensible alternative. Failure: a
range with no provenance, which cannot be audited or improved.

**Step 13. State what the interval does not cover.** Model misspecification, selection, a future
regime change, a structural break. Those are separate risks with separate mitigations, and naming
them stops the interval being read as a complete statement of uncertainty.

### 5.1 Choosing a block length

The theory gives conditions, not a number: the block length `l` must grow with `n` and vanish
relative to it. In practice you choose among three approaches.

| Approach | Rule | When it works | When it fails |
|---|---|---|---|
| Rule of thumb | `l` near `n^(1/3)` | Weakly dependent series, variance targets | Strong or long-memory dependence; quantile targets |
| Data-driven | Choose `l` minimising an estimate of MSE of the block bootstrap variance | Stationary series with enough length | Short series, where the estimate is noisy |
| Sensitivity | Compute the interval at `l`, `2l`, and `3l`; report the range | Always, as a check | Never as a substitute for thinking |

For a quarterly series of 12 observations, no block length is defensible: `n^(1/3)` is about 2.3,
so the block covers a fifth of the data, and there are only ten possible block starts. The honest
answer is that the data cannot support a block bootstrap, and the interval should come from a
parametric model with stated assumptions or be reported as a policy range.

### 5.2 Which interval to report

| The decision is about | Report | Why not the others |
|---|---|---|
| The level of a process, such as average lead time | CI for the mean | A PI describes one order, not the level |
| A single future order or shipment | PI | A CI is far too narrow and shrinks with `n` |
| A planning range covering most future demand | PI at the service level, or a tolerance interval | A CI says nothing about individual outcomes |
| A contractually promised band | Tolerance interval with `beta` and `gamma` stated | A PI covers one observation, not a proportion |
| Whether an improvement is real | CI for the difference, on the same resampling unit | Two separate CIs are not a comparison |

### 5.3 Why block joins matter

The block bootstrap keeps dependence inside blocks and destroys it at the joins. That is why the
block length matters: too short and the joins dominate; too long and there are too few distinct
blocks to resample from.

## 6. Worked example

Calder and Voss Industrial distributes hydraulic components. It buys a coupling, part CV-118, from
a single European supplier and ships it by ocean to a Chicago distribution centre. Over 14 months
the planner has recorded eight complete replenishment cycles. All figures are constructed to be
internally consistent and checkable by hand.

### 6.1 The data and the question

**Table 6.1 — Eight observed lead times, days**

| Order | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lead time (days) | 21 | 22 | 22 | 23 | 24 | 26 | 35 | 61 |

```text
  n        = 8 orders
  sum      = 21 + 22 + 22 + 23 + 24 + 26 + 35 + 61 = 234 days
  mean     = 234 / 8 = 29.250 days
  median   = (23 + 24) / 2 = 23.5 days

  deviations from the mean:
    -8.250, -7.250, -7.250, -6.250, -5.250, -3.250, 5.750, 31.750
  squared deviations:
     68.0625, 52.5625, 52.5625, 39.0625, 27.5625, 10.5625, 33.0625, 1008.0625
  sum of squares = 1291.5000 days squared
  s^2 = 1291.5000 / 7 = 184.5000 days squared
  s   = 13.583 days
  SE  = s / sqrt(8) = 13.583 / 2.8284 = 4.802 days
```

Two estimands are in play, and they are not the same question. The **mean** lead time is what you
report to a supplier review; the **next order's** lead time is what the customer promise depends
on; the planning team also wants the 95th percentile, which sets safety stock. All three are
computed below, and they disagree about how much is known.

The data are visibly not normal: seven values sit between 21 and 35 days, and one at 61.

### 6.2 The normal shortcut, and where it breaks

With `t(7) = 2.3646` for a two-sided 95% interval:

```text
  CI for the mean = 29.250 +/- 2.3646 x 4.802
                  = 29.250 +/- 11.356
                  = [17.894, 40.606] days

  PI for the next order = 29.250 +/- 2.3646 x 13.583 x sqrt(1 + 1/8)
                        = 29.250 +/- 2.3646 x 13.583 x 1.06066
                        = 29.250 +/- 34.083
                        = [-4.817, 63.317] days
```

The prediction interval runs below zero. Lead time cannot be negative, so the model has assigned
probability to an impossible event. Truncating the interval afterwards does not fix it: that
concentrates the omitted probability on the upper tail and pulls the upper bound too low. The fix
is a model with the right support. A lognormal fit to the same eight values gives a predictive 95%
interval of approximately [12.9, 58.1] days: asymmetric, positive, 45 days wide rather than 68.

> **Note:** The lognormal predictive interval uses the same eight observations and is therefore not
> better supported. It is better *shaped*. With `n = 8` the parameter estimates are themselves
> uncertain, and neither interval should be read as a settled answer.

### 6.3 Ten resamples, by hand

A bootstrap resamples eight values with replacement from the eight observed values, computes the
mean, and repeats. Below are ten resamples with the indices drawn, so each row can be checked
against Table 6.1. Indices refer to the sorted data: 1 = 21, 2 = 22, 3 = 22, 4 = 23, 5 = 24, 6 = 26,
7 = 35, 8 = 61.

**Table 6.2 — Ten bootstrap resamples of the mean**

| `b` | Indices drawn | Sum (days) | Mean (days) |
|---:|---|---:|---:|
| 1 | 1 2 3 4 5 6 7 8 | 234 | 29.250 |
| 2 | 1 1 2 2 3 3 4 4 | 176 | 22.000 |
| 3 | 5 6 7 8 8 8 8 8 | 390 | 48.750 |
| 4 | 1 2 3 5 6 7 8 8 | 272 | 34.000 |
| 5 | 2 2 4 4 6 6 8 8 | 264 | 33.000 |
| 6 | 1 1 1 1 2 2 2 2 | 172 | 21.500 |
| 7 | 3 4 5 6 7 8 8 8 | 313 | 39.125 |
| 8 | 1 5 6 7 8 8 8 8 | 350 | 43.750 |
| 9 | 1 2 3 4 5 6 7 1 | 194 | 24.250 |
| 10 | 1 2 3 4 5 6 5 4 | 185 | 23.125 |

```text
  sum of the ten resample means = 318.750 days
  bootstrap mean (average of the ten) = 31.875 days
  sample mean                        = 29.250 days
  Monte Carlo error at B = 10        = +2.625 days
```

That gap is the first lesson. With `B = 10` the average resample mean should estimate the sample
mean; it misses by 2.6 days because these draws over-represented the 61-day observation. At
`B = 10` the Monte Carlo error is larger than the effect most studies try to detect.

**The percentile interval from these ten resamples.** Sort the means: 21.50, 22.00, 23.125, 24.25,
29.25, 33.00, 34.00, 39.125, 43.75, 48.75. The 2.5th percentile falls between the first and second
values, at about 21.63; the 97.5th falls between the ninth and tenth, at about 47.63. The interval
is [21.63, 47.63] days.

It is wider than the converged answer below, not narrower: with ten resamples the 97.5th
percentile is nearly the largest resample mean, and that value is an extreme draw, not a quantile.
This is why `B` is not a cosmetic setting.

### 6.4 The exact bootstrap distribution

Because the sample is small, the bootstrap distribution can be enumerated exactly rather than
sampled. There are `C(15, 8) = 6,435` distinct resamples of eight values, which collapse to 250
distinct values of the mean because of ties and duplicate observations. Weighting each by its
multinomial probability gives the exact bootstrap distribution.

**Table 6.3 — Exact bootstrap distribution of the mean, selected quantiles**

| Quantity | Value (days) |
|---|---:|
| Bootstrap mean | 29.250 |
| Bootstrap standard deviation | 4.492 |
| 2.5th percentile | 22.625 |
| 97.5th percentile | 39.000 |

The exact percentile interval is **[22.63, 39.00] days**, 16.4 wide, against 22.7 for the `t`
interval [17.89, 40.61]. The bootstrap interval is narrower on both sides, partly because the
bootstrap standard deviation is 4.492 against `s / sqrt(n) = 4.802`; the ratio is
`sqrt(7/8) = 0.935`, the difference between the `1/n` and `1/(n-1)` divisors. The deeper reason is
that the sample may have missed the tail, and the bootstrap cannot know it did.

### 6.5 BCa, and a coverage check

The percentile interval assumes the bootstrap distribution is centred and its spread does not
depend on the parameter. Neither holds here. The BCa interval corrects for both, using two
constants computed from the data:

```text
  bias correction z0 = Phi^-1( fraction of bootstrap means below theta_hat )
                     = Phi^-1(0.5471)
                     = 0.118

  acceleration a     from the jackknife values theta_(i):
    30.429, 30.286, 30.286, 30.143, 30.000, 29.714, 28.429, 24.714
    a = sum (mean_jack - theta_(i))^3
        -----------------------------------------------
        6 x [ sum (mean_jack - theta_(i))^2 ]^(3/2)
      = 0.109
```

The acceleration is large because dropping the 61-day observation moves the mean from 29.25 to
24.71. The BCa adjustment gives **[23.38, 44.13] days**, 20.8 wide: the upper end moves from 39.0
to 44.1, a 13% increase produced entirely by correcting for skew.

The coverage simulation answers which interval is right, and is not flattering to any of them.

**Table 6.4 — Simulated coverage, nominal 95%, lognormal population, 3,000 replications**

| `n` | `t` interval coverage | Percentile bootstrap coverage |
|---:|---:|---:|
| 8 | 0.911 | 0.850 |
| 15 | 0.925 | 0.903 |
| 30 | 0.935 | 0.921 |

The population is lognormal with mean 10 and coefficient of variation 0.6; resampling is iid with
`B = 300` per replication. Simulations run for this topic, illustrative not benchmarked.

Three readings. At `n = 8` the percentile interval covers 85% of the time while claiming 95%. The
`t` interval is also imperfect at 91%, because the population is skewed and the interval
symmetric. Coverage improves slowly: at `n = 30` the percentile interval is still three short.

**The practical rule that follows.** Below about 20 observations, do not report a percentile
interval as a 95% interval: report BCa, or a wider interval with a stated caveat, or the estimate
beside the range of methods. The rule is not a law; it is the direction the evidence points.

> **Note:** The BCa correction makes the interval wider here, but it is not guaranteed to. When
> the bootstrap distribution is left-skewed, BCa can produce a narrower interval than the
> percentile method. The correction follows the skew; it does not always add width.

### 6.6 Three intervals for three questions

**Table 6.5 — The same eight observations, five intervals**

| Interval | Target | Formula or method | 95% interval (days) | Width (days) |
|---|---|---|---:|---:|
| `t` CI | Mean lead time | `xbar +/- t(7) s / sqrt(n)` | [17.89, 40.61] | 22.72 |
| Percentile bootstrap | Mean lead time | 2.5th and 97.5th percentiles | [22.63, 39.00] | 16.37 |
| BCa bootstrap | Mean lead time | bias- and skew-adjusted | [23.38, 44.13] | 20.75 |
| `t` PI | Next order | `xbar +/- t(7) s sqrt(1 + 1/n)` | [-4.82, 63.32] | 68.14 |
| Lognormal PI | Next order | fitted predictive quantiles | [12.9, 58.1] | 45.2 |
| Distribution-free tolerance | 90% of orders, 95% confidence | min to max | [21, 61] at 52% confidence | 40 |

The last row deserves explanation, because it is the row most often faked. With eight
observations, the smallest-to-largest interval contains at least 90% of the population with
probability `1 - 0.9^7 = 0.522`: a 90% content interval at 52% confidence, not 95%. Reaching 95%
confidence needs `n = 30`, where `1 - 0.9^29 = 0.953`.

```text
  P( min to max covers at least proportion beta ) = 1 - beta^(n-1)

  n = 8:   1 - 0.9^7  = 0.522
  n = 20:  1 - 0.9^19 = 0.865
  n = 30:  1 - 0.9^29 = 0.953
  n = 40:  1 - 0.9^39 = 0.984
```

This is why "the tolerance interval is just the range" holds only when the confidence statement is
dropped. A planner who needs a band containing 90% of lead times with 95% confidence needs thirty
observations, not eight, and no statistical method creates them.

### 6.7 The statistic the decision needs is not smooth

The planner does not want the mean. Safety stock is set from an upper quantile of lead-time
demand, so bootstrap the 95th percentile of the same eight values and see what happens.

With `n = 8`, the 95th percentile of a resample is always one of the largest two or three observed
values. The bootstrap distribution of the maximum is worse: it has a point mass at 61, the largest
observed value, of size

```text
  P( resample contains the maximum ) = 1 - (1 - 1/n)^n
                                     = 1 - (7/8)^8
                                     = 1 - 0.3436
                                     = 0.6564
```

Two thirds of all resamples have a maximum of exactly 61 days. The bootstrap distribution is
therefore not an approximation to the sampling distribution of the maximum; it cannot exceed the
observed maximum, and it piles most of its mass on it. As `n` grows the mass tends to
`1 - e^-1 = 0.632`, so this is asymptotic behaviour, not a small-sample quirk.

**The mechanism.** The bootstrap redraws from `F_hat`, whose support ends at the largest observed
value. The true sampling distribution depends on how far beyond that value the distribution
extends, which `F_hat` cannot represent: the bootstrap is inconsistent for extreme order statistics,
and the same failure appears for the mean when the variance is infinite.

**What to do instead.** Three options, in order of preference: fit a parametric tail (lognormal,
gamma, or generalised Pareto above a threshold) and propagate parameter uncertainty, stating the
assumption; subsample rather than bootstrap, drawing subsets of size `m < n` without replacement,
which stays valid for extremes; or report a policy buffer, declared as a management decision
rather than dressed as an estimated quantile.

**The general lesson** is that the bootstrap is a tool for smooth functionals: the further into the
tail the decision sits, the less the data supports the estimate, and the bootstrap cannot supply
what the data lack.

### 6.8 The same lane, resampled as a series

Now take a different and more realistic dataset: 36 consecutive weeks of lead time on the same
lane, ordered in time. The values, in days, are:

```text
  22.9 25.3 23.9 22.8 20.3 21.2 26.5 27.0 29.9 28.1 27.8 26.8
  19.8 24.6 26.2 27.2 19.9 16.7 17.2 18.6 21.8 22.6 25.1 22.4
  24.2 25.6 22.5 30.0 29.5 32.2 25.7 22.3 21.9 22.5 25.4 25.7
```

```text
  mean            = 24.225 days
  standard dev    = 3.617 days
  iid SE          = 3.617 / sqrt(36) = 0.603 days
  lag-1 autocorr  = 0.598
  AR(1) inflation = sqrt((1 + 0.598) / (1 - 0.598)) = 1.994
  implied SE      = 1.994 x 0.603 = 1.202 days
```

Resampling single weeks destroys the serial dependence: consecutive values are related, and after
shuffling they are not. Blocks retain it within blocks. The block bootstrap standard errors, from
4,000 resamples each:

**Table 6.6 — Bootstrap standard error of the mean by scheme**

| Scheme | Block length `l` (weeks) | SE (days) | Ratio to iid |
|---|---:|---:|---:|
| IID rows | 1 | 0.602 | 1.00 |
| Moving block | 2 | 0.772 | 1.28 |
| Moving block | 3 | 0.856 | 1.42 |
| Moving block | 4 | 0.892 | 1.48 |
| Moving block | 6 | 0.939 | 1.56 |
| Moving block | 9 | 0.934 | 1.55 |

The iid interval for the mean is `24.225 +/- 1.96 x 0.602 = [23.05, 25.41]`. The block interval at
`l = 4` is `24.225 +/- 1.96 x 0.892 = [22.48, 25.97]`. The second is 48% wider. Same data, same
statistic, same nominal confidence, and the only difference is whether the resampling respected
the time order.

Neither matches the AR(1) implication of 1.20 days exactly, and that is honest. The AR(1) formula
is asymptotic and assumes a model; the block bootstrap is nonparametric but truncates dependence
at block boundaries. At `n = 36` and `rho` near 0.6 both are approximations, and convergence with
`l` shows as the flat region around 6 to 9 weeks.

### 6.9 What the analyst reports

The decision is a customer promise for a single order, plus a safety-stock review. The reporting
line should look like this.

```text
  ESTIMATE     Mean lead time, CV-118, Rotterdam-Chicago, 2025: 29.3 days.
  INTERVAL     BCa bootstrap 95% interval for the mean: [23.4, 44.1] days,
               n = 8 orders, B = 10,000, resampling unit = order.
  WARNING      Percentile interval [22.6, 39.0] is narrower and undercovers at
               this n; simulation gives 85% actual coverage for a nominal 95%.
  DECISION     For a single-order promise, the relevant interval is predictive,
               not the interval for the mean: lognormal predictive 95% range
               [12.9, 58.1] days. A 28-day promise will be missed on a
               material share of orders.
  LIMIT        Eight observations cannot support a 95th-percentile estimate.
               The 95% interval for the mean is not the safety-stock input.
```

The three sentences that make this useful are the warning, the decision line, and the limit; the
estimate alone would support any of three contradictory actions.

## 7. Code, query, or calculation

One block, deliberately small. It reproduces the percentile interval of section 6.4 by sampling
rather than enumeration, so the two can be compared.

```python
import random
L = [21, 22, 22, 23, 24, 26, 35, 61]          # lead times, days
rng = random.Random(2026)

def pctl(x, q):                               # linear-interpolated quantile
    s = sorted(x); k = (len(s) - 1) * q; f = int(k)
    return s[f] + (k - f) * (s[min(f + 1, len(s) - 1)] - s[f])

boot = sorted(sum(rng.choices(L, k=len(L))) / len(L) for _ in range(10_000))
print(round(pctl(boot, 0.025), 2), round(pctl(boot, 0.975), 2),
      round(sum(boot) / len(boot), 3))
```

Expected output: `22.62 39.12 29.218`.

Four checks before trusting any such block.

- **The bootstrap mean must equal the sample mean, up to Monte Carlo error.** The output gives
  29.218 against 29.250, a gap of 0.032 days. With `B = 10,000` that is the expected order of
  magnitude. If the gap is large, the resampling is biased — most often because the sample size
  passed to the resample is not `len(L)`.
- **Compare with the exact answer where one exists.** Enumeration gives 22.625 and 39.000; the
  sampling version gives 22.62 and 39.12. The upper endpoint is 0.12 days high, consistent with
  the Monte Carlo error of a 97.5% quantile at `B = 10,000`.
- **Change the seed and rerun.** Endpoints should move by less than about 0.2 days at `B = 10,000`.
  If they move by a day, `B` is too small for the precision being reported.
- **Report no more decimals than the Monte Carlo error supports.** The standard error of an
  estimated 2.5% tail probability at `B = 10,000` is `sqrt(0.025 x 0.975 / 10,000) = 0.0016`, which
  translates to roughly 0.2 days near this endpoint. Two decimals is the honest maximum.

> **Note:** `rng.choices(L, k=len(L))` samples with replacement and equal weight, which is the
> iid bootstrap. In production, the resampling function is where the whole topic lives: pass a
> list of blocks, or a list of clusters, not a list of rows.

## 8. Data, standards, and regulatory context

### 8.1 What the data must contain

An interval is only as good as the record of how the sample was produced. Before resampling, the
dataset must answer six questions.

| Question | Why it matters | What to record |
|---|---|---|
| What is one row? | The resampling unit is derived from it | Table, grain, primary key |
| Is there a time index, and is it ordered? | Determines whether blocks are needed | Timestamp and sort rule |
| What is the grouping variable? | Determines the cluster level | `supplier_id`, `item_id`, `lane_id` |
| Is the outcome censored? | Stock-outs truncate demand; the tail is missing | Censoring flag and rule |
| Is the sample the whole frame? | Determines whether an FPC applies | Frame size `N`, drawn `n` |
| Was the sample selected on the outcome? | Breaks exchangeability | Selection rule and date |

The fourth is the one that ruins tails. A demand series measured as shipments is truncated at
available stock: when service is poor, the largest true demands are exactly the ones that never
appear, and the bootstrap reports a confident interval around an understated spread.

### 8.2 Standards that constrain the vocabulary, not the method

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 3534-1:2016 | Statistics vocabulary and symbols | The meanings of confidence interval, coverage, quantile, expected value | B |
| JCGM 100:2008 (GUM) | Expression of uncertainty in measurement | Coverage factor `k` and the definition of expanded uncertainty | B |
| ISO 8601-1:2019, ISO 8601-2:2019 | Date and time representation | What a lead-time interval or duration is measured on | B |

No standard mandates a bootstrap, a block length, or a clustering level. ISO 3534-1 settles what
the words mean, not which interval your decision needs. JCGM 100 defines a coverage factor for an
expanded uncertainty; it assumes a model for the measurand, and the choice of that model is yours.

The practical consequence is that an interval claim is a **model-risk claim**, not a compliance
claim. The documentation that survives review answers five questions: what estimand, what
resampling unit, what `B`, what interval type, and what coverage evidence.

### 8.3 What regulation does not supply

- No regulation requires a prediction interval where a confidence interval was computed, and none
  will catch the substitution.
- No accounting standard constrains the uncertainty statement attached to a forecast; IFRS and US
  GAAP constrain recognition and measurement, not the width of a planning band.
- No traceability regulation constrains how you resample event data, only that events exist.

Uncertainty reporting is therefore a professional judgement with an audit trail, not a checkbox.

### 8.4 Convention rather than requirement

Several things that feel like rules are habits.

- **95%** is a quality-control convention inherited from acceptance sampling; the right level is set
  by the asymmetry of the loss, and a hospital stock-out is not a seasonal-item stock-out.
- **`B = 1,000`** is a software default from an era of slow processors, adequate for a standard
  error and inadequate for a reported 2.5% endpoint.
- **"Plus or minus"** without a coverage statement is a formatting habit, not a statistical claim,
  and **`n = 30`** as a threshold for normality is a rule of thumb about the mean of a symmetric
  population that does not transfer to quantiles, ratios, or clustered data.

## 9. Industry practice and case evidence

### 9.1 Provenance

| Contribution | Source | What it established | Grade |
|---|---|---|---|
| The bootstrap | Efron (1979), *Annals of Statistics* 7(1), 1–26 | Resampling the empirical distribution estimates the sampling distribution; introduced in the 1977 Rietz Lecture | A |
| BCa intervals | Efron (1987), *JASA* 82(397), 171–185 | Bias-corrected and accelerated intervals; second-order accuracy; reduces to percentile and BC as special cases | A |
| Block bootstrap | Künsch (1989), *Annals of Statistics* 17(3), 1217–1241 | Resampling blocks of `l` consecutive observations is consistent for stationary series when `l → ∞` and `l/n → 0` | A |
| Stationary bootstrap | Politis and Romano (1994), *JASA* 89(428), 1303–1313 | Random geometric block lengths produce a resample that is itself stationary | A |
| Cluster bootstrap | Field and Welsh (2007), *JRSS-B* 69(3), 369–390 | The cluster bootstrap is consistent under both transformation and random-effects models, where the residual bootstrap is consistent only under the former | A |
| Tolerance limits | Wilks (1941), *Annals of Mathematical Statistics* 12, 91–96 | Sample sizes for distribution-free tolerance limits from order statistics | A |

The dates matter. The block bootstrap is not a recent correction; it is thirty-five years old, and
it is still omitted from dashboards because the theory is settled and the software default is not.

### 9.2 What the methodological literature establishes

| Finding | Source | Implication for practice | Grade |
|---|---|---|---|
| The bootstrap is inconsistent for the mean when the variance is infinite; the limit is random | Athreya (1987), *Annals of Statistics* 15, 724–731 | Heavy-tailed cost or loss data may not be bootstrappable at all | A |
| The bootstrap for extremes does not converge to an extreme-value distribution; subsampling remains valid | Qi (2008), *Extremes* 11, 81–97, reviewing the extreme-value bootstrap literature | Do not bootstrap a 99.9th percentile and call it an interval | A |
| Cluster-robust inference over-rejects with few clusters; wild cluster bootstrap and bias-corrected variance estimators help | Cameron and Miller (2015), *Journal of Human Resources* 50(2), 317–372 | With 10–40 clusters, the naive cluster bootstrap is not enough | A |
| Confidence intervals are always narrower than prediction or tolerance intervals; the comparison between PI and tolerance depends on future sample size against content | Hahn, Meeker and Escobar (2017), *Statistical Intervals*, 2nd ed., ch. 2 | The three intervals are not interchangeable and the difference is large | A |
| A confidence interval used as a prediction interval can contain a small fraction of the intended probability | Hyndman (2013), "The difference between prediction intervals and confidence intervals" | The substitution is common and materially misleading | C |

The last row is graded C because it is expert commentary, not a peer-reviewed study. The
substitution it warns about is arithmetic: at `n = 8` the CI is 22.7 days wide and the PI 68.1.

### 9.3 What practice looks like

Planning dashboards overwhelmingly report point estimates with, at most, a symmetric band. Where
a band exists, the documented method is usually `mean ± k × standard deviation` applied to a
metric that is bounded, skewed, or clustered. Vendor documentation reviewed for this course in
2026 describes "confidence ranges" without stating the interval type, the resampling unit, or the
coverage. Grade D, dated 2026: product documentation, not evidence of accuracy.

Two conventions are worth naming, because they are ubiquitous and unexamined. **The supplier
scorecard band** often shows a mean with a standard deviation rather than a standard error: the
first describes order-to-order variability, the second how well the mean is known, and plotting the
first where the second is needed makes every supplier look indistinguishable. **The "±10%" planning
buffer** with no coverage statement is a policy, not an interval. It may be a good policy; it is not
evidence about the process.

### 9.4 What is missing from the public record

Searches for this topic returned no peer-reviewed audit of interval reporting practice in
commercial supply chain analytics, and almost none on which interval type organisations compute.
Claims about "most companies" should therefore be read as impressions.

What the record supports is narrower and still useful: the methods are old, the failure modes are
documented, and small-sample coverage is measurable, so a team that checks its own coverage is
ahead of the published baseline, whatever that baseline is.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| IID resampling of ordered data | Interval width is identical before and after shuffling the rows | Resampling destroys serial dependence | Interval too narrow; confidence overstated | Moving block or stationary bootstrap |
| Row-level resampling of clustered data | 4,800 rows from 40 suppliers treated as 4,800 facts | Within-cluster correlation ignored | Standard error too small by `sqrt(DEFF)` | Cluster bootstrap at the assignment level |
| Block length shorter than the dependence horizon | SE rises steeply as `l` increases, then plateaus above your choice | Blocks cut the dependence before it decays | Undercoverage, sometimes severely | Sensitivity table across `l`; choose in the plateau |
| Percentile interval at small `n` | `n` below about 20; percentile narrower than the `t` interval | Bootstrap distribution too narrow; it cannot see missed tails | Nominal 95% delivers about 85% | BCa, or a stated wider interval, or more data |
| Bootstrap of an extreme quantile | Endpoint equals the largest observation ever recorded | Empirical support ends at the sample maximum | Interval is an artefact of one or two points | Parametric tail model or subsampling |
| Bootstrap of a maximum or minimum | Point mass at the observed extreme | `F_hat` has no mass beyond the maximum | No valid interval at any `B` | Do not bootstrap; model the tail |
| CI reported where a PI is needed | Promise dates built from the mean's interval | CI narrows with `n`; individual outcomes do not | Systematic service failure | Compute the predictive interval |
| CI called a tolerance interval | "95% of lead times fall between 17.9 and 40.6" | Coverage of a parameter is not content of a population | Band is far too narrow for planning | State `beta` and `gamma`; use order statistics or a model |
| Normal `± 1.96 SE` on a bounded statistic | Interval extends below 0 or above 100% | Symmetric support, wrong shape | Impossible values; tail probability misplaced | Transform, or use a bootstrap on the right scale |
| `B` too small, decimals overstated | Endpoints move in the second decimal across seeds | Monte Carlo error larger than the reported precision | False precision in a contract or a slide | `B ≥ 10,000` for endpoints; report 2 decimals |

Two rows deserve expansion. **The dependence failure is the most consequential** because it is
invisible: the number looks like an interval, carries the right nominal coverage, and comes from a
reputable library. In the worked example the iid interval was 48% narrower on the same 36 weeks.
**The wrong interval type is the most common**, surviving because both are called "the range" in
conversation; one question — "the average, or the next one?" — resolves it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Exchangeability within the resampling unit | The resample must be a plausible redraw of the data | Time order, clustering, selection on the outcome | Compare resampled and original dependence statistics |
| Weak stationarity across the window | Blocks from different parts of the series must be comparable | Trend, regime change, supplier switch | Split the window; compare block bootstrap SEs |
| `l → ∞` and `l/n → 0` | Consistency of the block bootstrap | `l` fixed, or `l` a large fraction of `n` | Sensitivity table across `l` |
| The statistic is a smooth functional | The bootstrap distribution must converge to the sampling distribution | Maxima, extreme quantiles, tail indices | Ask whether the statistic depends on the top few observations |
| Finite variance, for the mean | The CLT and the bootstrap both need it | Heavy-tailed cost or claim data | Plot the tail; check stability of the mean under trimming |
| The sample represents the process | The interval is about the process, not the file | Censoring, survivorship, convenience samples | Compare frame and sample on observables |
| Clustering is at the level of assignment | The cluster bootstrap must resample the unit that was randomised or exposed | Treatment assigned at supplier, resampling at order | Ask how the treatment was assigned |
| Enough distinct values | Endpoints must be estimable | Ties, coarse bins, `n` under 20 | Count distinct values and tail cells |

**What no bootstrap fixes.** Model misspecification, confounding, selection on the outcome, a
future structural break. The bootstrap quantifies uncertainty for an estimator applied to a fixed
model and sample; if the model is wrong it gives a precise interval around the wrong number.

**Non-stationarity is the hard limit.** A block bootstrap on a trending series resamples blocks
from a process that no longer exists. If a split-window comparison gives block standard errors
that differ beyond Monte Carlo error, the series needs a structural model.

**The bootstrap cannot manufacture tail information.** With no observation beyond 61 days, no
resampling scheme will tell you how often 80 days occurs. It can describe a region it has visited
and nothing else.

## 12. Exercises

### Exercise 1 — Bootstrap six values by hand (calculation, 40 minutes)

Lead times in days: 12, 15, 15, 18, 22, 40. Compute the mean, the standard deviation, the `t` 95%
confidence interval, and the 95% prediction interval. Then take ten resamples of your own
choosing, compute their means, and form a percentile interval.

**Deliverable:** the summary statistics, both parametric intervals, the ten resample means in a
table, and the percentile interval.
**Acceptance criteria:** every quantity carries days; the resample indices are listed so the
arithmetic can be checked; the percentile interval is compared with the `t` interval and the
direction of the difference is explained.
**Time:** 40 minutes.

### Exercise 2 — Three intervals on one dataset (calculation, 45 minutes)

Using the eight lead times in section 6.1, compute the percentile interval, the BCa interval, and
the `t` interval. Compute the bias-correction constant `z0` and the acceleration `a` from the
jackknife values.

**Deliverable:** a table of the three intervals with widths, and the two BCa constants with their
intermediate quantities.
**Acceptance criteria:** jackknife values listed; `z0` computed from the proportion of bootstrap
means below the point estimate; the direction of the BCa adjustment explained in one sentence.
**Time:** 45 minutes.

### Exercise 3 — Coverage by simulation (computation, 60 minutes)

Simulate 2,000 samples of size 8, 15, and 30 from a lognormal population with mean 10 and
coefficient of variation 0.6. For each, compute the `t` interval and the percentile bootstrap
interval and record the fraction containing the true mean.

**Deliverable:** a coverage table by `n` and method, with the simulation settings stated.
**Acceptance criteria:** nominal coverage stated; Monte Carlo error of the coverage estimate
reported; the conclusion distinguishes bias from variance.
**Time:** 60 minutes.

### Exercise 4 — Block length sensitivity (calculation, 40 minutes)

Using the 36-week series in section 6.8, compute the moving block bootstrap standard error of the
mean at `l = 1, 2, 3, 4, 6, 9`. Plot or tabulate the result against `l`.

**Deliverable:** the table, the plot or its description, and a recommended `l` with justification.
**Acceptance criteria:** at least five block lengths; the plateau identified; the recommendation
states what would change it.
**Time:** 40 minutes.

### Exercise 5 — Design effect and effective sample size (calculation, 30 minutes)

A category has 40 suppliers and 12 monthly observations each, 480 rows in total. The estimated
intra-cluster correlation is 0.25. Compute the design effect, the effective sample size, and the
factor by which the standard error is understated by row-level resampling.

**Deliverable:** the three quantities with formulas and units.
**Acceptance criteria:** `DEFF = 1 + (m − 1) rho` written explicitly with `m` and `rho` defined; the
standard error factor computed as a square root; the interpretation stated for a supplier
comparison.
**Time:** 30 minutes.

### Exercise 6 — Choose the interval (design, 35 minutes)

For each decision, name the interval type and justify it: (a) a customer promise for one order;
(b) a safety-stock review for one item; (c) a contractual band covering 90% of deliveries;
(d) whether a new carrier reduced mean transit time; (e) the average lead time to report in a
supplier review.

**Deliverable:** a five-row table with decision, interval, target, and one-sentence justification.
**Acceptance criteria:** no decision assigned a CI where a PI is required; the tolerance interval
states `beta` and `gamma`; the carrier comparison uses an interval for a difference.
**Time:** 35 minutes.

### Exercise 7 — Show that the bootstrap fails at the maximum (calculation, 25 minutes)

For `n = 8`, compute the probability that a bootstrap resample contains the sample maximum. Repeat
for `n = 20` and `n = 100`, and compare with `1 − e^-1`.

**Deliverable:** the three probabilities with the formula, and a one-paragraph consequence for
estimating a 99th-percentile lead time.
**Acceptance criteria:** `1 − (1 − 1/n)^n` shown explicitly; the limit identified; the consequence
names a substitute method.
**Time:** 25 minutes.

### Exercise 8 — Finite population correction (calculation, 25 minutes)

A lane population contains 60 lanes; you have data on 40. Compute the FPC and the corrected
standard error for a sample standard deviation of 6 days and `n = 40`. State the assumption that
makes the correction valid.

**Deliverable:** the factor, the corrected standard error, and the assumption.
**Acceptance criteria:** factor computed as a square root of a ratio; both standard errors shown;
the simple-random-sampling assumption stated.
**Time:** 25 minutes.

### Exercise 9 — Write the reporting standard (design, 50 minutes)

Write a one-page standard that any analyst in a planning team could follow when reporting an
interval.

**Deliverable:** the standard, with a required reporting line and two worked examples.
**Acceptance criteria:** estimand, unit, interval type, resampling unit, `B`, and coverage evidence
all required; one example shows a CI and one a PI; the standard states what the interval does not
cover.
**Time:** 50 minutes.

## 13. Self-check answers

### Exercise 1 arithmetic

```text
  values 12, 15, 15, 18, 22, 40          n = 6,  sum = 122
  mean = 122 / 6 = 20.3333 days
  deviations: -8.3333, -5.3333, -5.3333, -2.3333, 1.6667, 19.6667
  sum of squares = 521.3333 days squared
  s^2 = 521.3333 / 5 = 104.2667;   s = 10.2111 days
  SE  = 10.2111 / sqrt(6) = 4.1687 days
  t(5) = 2.5706

  CI = 20.3333 +/- 2.5706 x 4.1687
     = 20.3333 +/- 10.7157 = [9.62, 31.05] days

  PI = 20.3333 +/- 2.5706 x 10.2111 x sqrt(1 + 1/6)
     = 20.3333 +/- 26.2487 x 1.08012
     = 20.3333 +/- 28.3523 = [-8.02, 48.69] days
```

The exact bootstrap distribution has 108 distinct values of the mean. Its 2.5th percentile is
14.50 days and its 97.5th is 28.67 days, so the percentile interval is **[14.50, 28.67]**, only
14.2 days wide against 21.4 for the `t` interval. A student who reports this as a 95% interval
without comment has reproduced the small-sample failure: with six observations the bootstrap
cannot see the tail the sample missed, and the interval is too narrow in a way that looks
reassuring. The prediction interval still runs below zero, for the same reason as in section 6.2.

A correct answer notes that the bootstrap mean equals the sample mean (20.33), that the bootstrap
standard deviation is 3.81 rather than 4.17 (`sqrt(5/6)` of `s/sqrt(n)`, the biased divisor again),
and that neither fact repairs the missing tail.

### Exercise 2 arithmetic

```text
  jackknife means (drop one observation):
    20.429, 20.286, 20.286, 20.143, 20.000, 29.714   -> mean 20.333
  z0 = Phi^-1(0.5471) = 0.118
  a  = 0.109        (dominated by dropping the 61-day order)

  t interval          [17.89, 40.61]    width 22.72 days
  percentile interval [22.63, 39.00]    width 16.37 days
  BCa interval        [23.38, 44.13]    width 20.75 days
```

The BCa upper end is 5.1 days above the percentile upper end. That is the correction doing its
job: the bootstrap distribution of the mean is right-skewed because one observation is far above
the rest, and the percentile method reads the skew in the wrong direction at the upper tail.

### Exercise 3 and Exercise 4 answers

Simulated coverage at nominal 95%, lognormal mean 10, CV 0.6, 3,000 replications: `n = 8` gives
0.911 (`t`) and 0.850 (percentile); `n = 15` gives 0.925 and 0.903; `n = 30` gives 0.935 and 0.921.
The Monte Carlo standard error of a coverage estimate at 3,000 replications near 0.90 is
`sqrt(0.9 x 0.1 / 3000) = 0.0055`, so the differences between `n` levels are real.

Block bootstrap standard errors for the 36-week series: `l = 1` 0.602, `l = 2` 0.772, `l = 3`
0.856, `l = 4` 0.892, `l = 6` 0.939, `l = 9` 0.934 days. The series is stationary by construction,
so the curve flattens from about `l = 6`. Choose `l = 6`; if the flat region had not appeared
before `n/10 = 3.6`, the series would be too short for this method.

### Exercise 7 and Exercise 8 answers

```text
  P(resample contains the maximum) = 1 - (1 - 1/n)^n
  n = 8:    1 - (7/8)^8   = 1 - 0.3436 = 0.6564
  n = 20:   1 - (19/20)^20 = 1 - 0.3585 = 0.6415
  n = 100:  1 - (99/100)^100 = 1 - 0.3660 = 0.6340
  limit:    1 - e^-1 = 0.6321
```

The mass barely moves with `n`, which is the point: this is not a small-sample problem that more
data fixes. A bootstrap interval for the maximum is therefore a statement about the sample, not
about the process, and a 99th-percentile lead time requires a tail model or subsampling.

```text
  FPC = sqrt((60 - 40) / (60 - 1)) = sqrt(20/59) = sqrt(0.33898) = 0.5822
  uncorrected SE = 6 / sqrt(40) = 0.9487 days
  corrected SE   = 0.5822 x 0.9487 = 0.5523 days
```

Valid only if the 40 lanes are a simple random sample of the 60. If they are the high-volume
lanes, the frame is not the 60 and the factor does not apply.

### Rubrics for Exercises 5, 6, and 9

**Exercise 5.** Strong: `DEFF = 1 + (12 − 1)(0.25) = 3.75`; effective `n = 480 / 3.75 = 128`;
standard error understated by `sqrt(3.75) = 1.94`, so a row-level interval is about half as wide as
it should be. Weak: divides by 12 or by 40 without stating which, or reports the design effect as
a variance ratio without converting to a standard error.

**Exercise 6.** Strong: (a) PI, because the decision is one order; (b) PI at the service level,
because safety stock is a quantile of lead-time demand; (c) tolerance interval with `beta = 0.90`
and `gamma = 0.95`; (d) CI for the difference in means, clustered by lane; (e) CI for the mean,
because the review is about the level. Weak: assigns a CI to (a) or (b), or reports two separate
CIs for (d) and calls it a comparison.

**Exercise 9.** Strong: mandates estimand, unit, interval type, resampling unit, `B`, and coverage
evidence; requires the predictive interval alongside the mean interval whenever a single-order
decision is involved; states that the interval does not cover regime change or model error. Weak:
specifies a formatting template without specifying the resampling unit, which is the one field
that makes the interval reproducible.

## 14. Interview questions

**1. "What does a 95% confidence interval mean?"**
Strong: it is a procedure-level statement — about 95% of intervals built this way contain the true
value — and it says nothing about the probability that this particular interval contains it.
Weak: "there is a 95% chance the true value is in the interval," which is the Bayesian statement
attached to a frequentist object.

**2. "Why not just use estimate ± 1.96 standard errors?"**
Strong: it assumes independence and approximate normality; supply chain data is skewed, bounded,
autocorrelated, and clustered, and every one of those makes the interval too narrow. Weak: says it
is "the standard formula" and stops.

**3. "When does the bootstrap fail?"**
Strong: names three conditions — exchangeable resampling units, enough distinct values, and a
statistic that does not depend on extreme order statistics — and gives an example of each.
Weak: says it fails when the sample is small, which is one of the three and the least specific.

**4. "Why block bootstrap a lead-time series?"**
Strong: iid resampling destroys serial dependence, so the resample is a shuffled series with the
same values and the wrong variance; blocks preserve dependence within the block. Weak: says
"because the data are time series" without saying what is preserved.

**5. "How would you choose the block length?"**
Strong: start near `n^(1/3)`, compute the standard error across a range of lengths, and choose in
the plateau; report the range. Weak: picks `n/10` or the software default.

**6. "You have 40 suppliers with 12 monthly observations each. What do you resample?"**
Strong: suppliers, because the comparison is between suppliers and the observations within a
supplier are not independent; notes the design effect and that 480 rows carry roughly 128 rows of
independent information. Weak: resamples the 480 rows.

**7. "Confidence interval or prediction interval for a customer promise?"**
Strong: prediction interval, because the promise is about one future order; the confidence
interval for the mean narrows with `n` and is irrelevant to a single outcome. Weak: uses the
confidence interval because it is what the software printed.

**8. "What is a tolerance interval for?"**
Strong: to contain a stated proportion of the population with stated confidence, which is what a
contractual band or a safety-stock range needs; distinguishes `beta`-content from
`beta`-expectation. Weak: describes it as a wider confidence interval.

**9. "Your bootstrap interval is narrower than the t interval. Is that good?"**
Strong: not necessarily — at small `n` the percentile interval undercovers, and the bootstrap
standard deviation uses the biased divisor; check coverage before celebrating precision.
Weak: assumes narrower means more precise and therefore better.

**10. "How many resamples do you need?"**
Strong: derives it from the Monte Carlo standard error `sqrt(p(1-p)/B)` for the tail probability
being reported, and notes that 1,000 is fine for a standard error and 10,000 for a reported
endpoint. Weak: says 1,000 because it is the default.

**11. "A dashboard says lead time improved from 24.0 to 22.5 days. What do you ask?"**
Strong: the interval type, the sample size, the resampling unit, whether the periods are
independent, and whether the change exceeds the interval; then what decision the claim supports.
Weak: congratulates the team or asks only for the sample size.

## 15. Key terms

- **Standard error (SE)** — the standard deviation of an estimator's sampling distribution.
- **Sampling distribution** — the distribution of an estimator over repeated samples.
- **Coverage** — the long-run fraction of intervals that contain the true value.
- **Nominal coverage** — the coverage a procedure claims.
- **Actual coverage** — the coverage it delivers under the true process.
- **Confidence interval (CI)** — an interval for a parameter with stated repeated-sampling coverage.
- **Prediction interval (PI)** — an interval for a future observation.
- **Tolerance interval** — an interval containing a stated proportion with stated confidence.
- **Credible interval** — a Bayesian interval with stated posterior probability.
- **Percentile interval** — the `alpha/2` and `1 − alpha/2` quantiles of the bootstrap distribution.
- **Basic interval** — the bootstrap interval reflected about the point estimate.
- **BCa interval** — bias-corrected and accelerated bootstrap interval.
- **Bootstrap distribution** — the distribution of a statistic over `B` resamples.
- **Resampling unit** — the object drawn with replacement: row, block, or cluster.
- **Exchangeability** — the joint distribution is unchanged by permuting the units.
- **IID** — independent and identically distributed.
- **Block bootstrap** — resampling contiguous blocks to preserve serial dependence.
- **Stationary bootstrap** — block bootstrap with geometric random block lengths.
- **Block length** — the number of consecutive periods resampled together.
- **Cluster bootstrap** — resampling whole clusters rather than rows.
- **Wild cluster bootstrap** — residual perturbation for settings with few clusters.
- **Design effect (DEFF)** — variance under a complex design divided by variance under simple random sampling.
- **Effective sample size** — `n / DEFF`.
- **Intra-cluster correlation (ICC)** — correlation between two observations in the same cluster.
- **Finite population correction (FPC)** — `sqrt((N − n)/(N − 1))` for a large sampling fraction.
- **Plug-in principle** — substituting the empirical distribution for the unknown one.
- **Subsampling** — drawing subsets of size `m < n`, valid under weaker conditions.
- **Monte Carlo error** — error from finite `B`.
- **Jackknife** — leave-one-out recomputation.

## 16. Sources

**Peer-reviewed and academic**

- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." *The Annals of Statistics*,
  7(1), 1–26. DOI: 10.1214/aos/1176344552. Grade A.
- Efron, B. (1987). "Better Bootstrap Confidence Intervals." *Journal of the American Statistical
  Association*, 82(397), 171–185. DOI: 10.1080/01621459.1987.10478410. Grade A.
- Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations."
  *The Annals of Statistics*, 17(3), 1217–1241. DOI: 10.1214/aos/1176347265. Grade A.
- Politis, D. N. and Romano, J. P. (1994). "The Stationary Bootstrap." *Journal of the American
  Statistical Association*, 89(428), 1303–1313. DOI: 10.1080/01621459.1994.10476870. Grade A.
- Athreya, K. B. (1987). "Bootstrap of the Mean in the Infinite Variance Case." *The Annals of
  Statistics*, 15, 724–731. DOI: 10.1214/aos/1176350371. Grade A.
- Hall, P. (1990). "Asymptotic Properties of the Bootstrap for Heavy-Tailed Distributions." *The
  Annals of Statistics*, 18, 1342–1360. Grade A.
- Arcones, M. A. and Giné, E. (1989). "The bootstrap of the mean with arbitrary bootstrap sample
  size." *Annales de l'Institut Henri Poincaré*, 25, 457–481. Grade A.
- Deheuvels, P., Mason, D. M. and Shorack, G. R. (1993). "Some results on the influence of extremes
  on the bootstrap." *Annales de l'Institut Henri Poincaré*, 29, 83–103. Grade A.
- Qi, Y. (2008). "Bootstrap and empirical likelihood methods in extremes." *Extremes*, 11, 81–97.
  Grade A.
- Field, C. A. and Welsh, A. H. (2007). "Bootstrapping Clustered Data." *Journal of the Royal
  Statistical Society: Series B*, 69(3), 369–390. DOI: 10.1111/j.1467-9868.2007.00593.x. Grade A.
- Cameron, A. C. and Miller, D. L. (2015). "A Practitioner's Guide to Cluster-Robust Inference."
  *Journal of Human Resources*, 50(2), 317–372. DOI: 10.3368/jhr.50.2.317. Grade A.
- Wilks, S. S. (1941). "Determination of sample sizes for setting tolerance limits." *Annals of
  Mathematical Statistics*, 12, 91–96. Grade A.

**Books and reference works**

- Hahn, G. J., Meeker, W. Q. and Escobar, L. A. (2017). *Statistical Intervals: A Guide for
  Practitioners and Researchers*, 2nd edition. Wiley. Chapter 2 distinguishes confidence,
  prediction, and tolerance intervals. Grade A.

**Reviews and tutorials**

- Tian, Y., Nordman, D. J. and Meeker, W. Q. "Methods to Compute Prediction Intervals: A Review
  and New Results." arXiv:2011.03065. Preprint; useful history of prediction and tolerance
  intervals. Grade B.
- Francq, B. G., Berger, M. and Boachie, C. (2020). "To tolerate or to agree: A tutorial on
  tolerance intervals in method comparison studies." *Statistics in Medicine*. Open access via
  PMC7756677. Grade A.
- Hyndman, R. J. (2013). "The difference between prediction intervals and confidence intervals."
  Blog post, 13 March 2013. Expert commentary on a common substitution. Grade C.
- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*, sections 1.7 and
  3.5 on the statistical forecasting perspective and prediction intervals. Grade B.

**Standards**

- ISO 3534-1:2016, *Statistics — Vocabulary and symbols — Part 1*. Grade B.
- ISO 8601-1:2019 and ISO 8601-2:2019, *Date and time — Representations for information
  interchange*. Grade B.
- JCGM 100:2008, *Evaluation of measurement data — Guide to the expression of uncertainty in
  measurement*. Grade B.

**Simulations and practitioner material**

- Coverage and block-bootstrap simulations reported in sections 6.5 and 6.8 were run for this
  topic in pure Python with seeds 2026, 7, and 11; 3,000 replications with `B = 300` for coverage
  and 4,000 resamples for the block standard errors. Illustrative, not a benchmark. Grade E.
- Planning and advanced planning system documentation reviewed in 2026 describes "confidence
  ranges" without stating interval type, resampling unit, or coverage. Grade D, dated 2026.

**Note on citations.** Findings are reported as the cited works state them and were not replicated
except where the text says a simulation was run for this topic. Citation counts mentioned in
research for this topic are not reproduced here because they were not independently verified. No
citation is invented; where no company-level evidence exists, the text says so.

## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- Previous topic, which fixes the estimand this topic quantifies: [`01_estimands-question-types.md`](01_estimands-question-types.md)
- Next topic, which turns intervals into evidence about effects: `03_hypothesis-tests.md` (planned)
- Autocorrelation and robust errors, the regression counterpart of section 6.8: `05_autocorrelation-robust-errors.md` (planned)
- Experiments and quasi-experiments, where the interval is narrowed by design: `07_experiments-quasi.md` (planned)
- The distribution background this topic assumes: [`../01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md`](../01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Module survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Return to the part map: [`../_index.md`](../_index.md)
