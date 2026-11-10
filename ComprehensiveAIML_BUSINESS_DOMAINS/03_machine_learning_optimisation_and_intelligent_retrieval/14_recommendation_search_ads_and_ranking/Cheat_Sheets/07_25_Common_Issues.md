---
title: "Common issues: recommendation and retrieval"
topic: "Defect catalogue for recommendation, search, ads and ranking systems"
collection: "Part 03"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Common issues: recommendation and retrieval

> **Evidence:** every number below is DERIVED from stated inputs or labelled ILLUSTRATIVE. No company deployment, benchmark, revenue figure or incident is claimed. Formulas are standard published results (inverse-propensity scoring, discounted cumulative gain, Pearson chi-square with one degree of freedom, BM25 term weighting); the arithmetic is recomputed here and can be checked line by line.

## 1. File Header

**Purpose.** Give a reviewer a single place to look up a *specific, named* defect in a recommendation, search, ads or ranking system, distinguish it from the two or three other defects that produce the same symptom, and choose a cheap discriminating check before spending engineering time. This sheet is a triage instrument, not a tutorial in any one method.

**Audience.** Engineers and analysts who already own or review a retrieval/ranking pipeline: someone who can read a metric definition, run a SQL or log query, and argue about whether a difference between two numbers is real.

**Prerequisites.** You should be able to (a) state a metric's numerator, denominator, population and window without looking it up, (b) read a histogram of request latency, (c) explain what a randomised experiment controls for, and (d) say what "event time" versus "processing time" means in your own stack.

**In scope.** Ranking-metric computation, exposure and position effects, candidate-generation recall, index and feature freshness, impression and click logging, experiment assignment and analysis, shard/topology asymmetries, and the feedback loop between logged data and the next model. Roughly forty systems defects are compressed into the twenty-eight catalogue entries, each of which names the category, the observable symptom, the competing causes, one discriminating check, the correction, the prevention measure and the verification that closes it.

**Out of scope.** Model architecture choice (covered by the architecture sheet), offline metric selection (metrics sheet), the online experiment design checklist, and the mechanics of phrase retrieval or BM25 statistics, which have their own sheets in this unit. Where a catalogue entry depends on one of those, the entry links by filename rather than restating the derivation.

**How to use it under time pressure.** Start from the symptom you actually observed — one of: a metric looks too good, a metric looks too bad, two systems disagree, traffic split is uneven, results are stale, results are duplicated, or a change shipped and nothing moved. Read the four-category axis in section 2 to pick the class, then jump to the catalogue entry whose symptom matches. Run only the discriminating check, not the full list; the point of the "competing causes" column is to stop you from rebuilding an index when the real bug is a dedupe key.

**Severity policy used in this file.** *Sev-1* means the defect silently corrupts a reported decision metric (someone will ship or revert on a wrong number). *Sev-2* means a user-visible ranking or relevance regression that is bounded and reversible. *Sev-3* means cost or operability degradation without correctness loss. Every catalogue entry names its usual severity, because severity, not category, decides whether you stop the release.

**Repair loop.** Symptom → category → catalogue entry → discriminating check → correction → prevention → verification. The verification column is not optional politeness: an issue is only closed when a check exists that would fail if the defect returned. Defects that get "fixed" without a verification line reappear at the next re-index, and the second occurrence is always more expensive because confidence in the metric is already damaged.

**Notation and units.** Everything is plain text. Ratios are dimensionless; counts carry an explicit unit (impressions, sessions, queries, milliseconds, bytes); rates are named by their denominator, never written as a bare percentage. Windows are given in seconds and always as `[start, end)` in event time unless the word "processing" appears. Money does not appear at all in this sheet, which removes a whole class of unit confusion.

**ID scheme.** Catalogue identifiers are `CI-nn` and are stable: they are assigned once and never renumbered, so a ticket or a post-mortem can cite `CI-17` and mean the same thing a year later. New issues take the next free integer. An entry is retired by prefixing its symptom with `RETIRED —` plus the date and the reason, never by deletion, because a retired issue often returns under a new technology stack.

**What "distinct" means here.** Two entries are distinct only if their discriminating check differs. If two rows share a check, one of them is a symptom of the other and belongs in the "competing causes" cell rather than in its own row. That rule is why the catalogue is 28 entries and not 90: many plausible-sounding issues collapse into the same observable.

**Evidence boundary.** Every numeric example in sections 4, 5 and 9 uses constructed inputs. The construction is stated so you can substitute your own traffic. If a number here disagrees with your production dashboard, your dashboard wins; the arithmetic pattern is what transfers.

## 2. Concept Map / Mental Model

The pipeline view is the fastest way to localise a defect, because each stage owns a different class of bug and produces a different signature.

```text
  content/events ──► ingestion ──► feature store ──► candidate indexes ──► ranker ──► policy ──► response
                        │               │                  │                 │          │
                    watermark        freshness          recall@k          score       diversity
                    lateness         skew               idf/avgdl         calibration  constraints
                        │               │                  │                 │          │
                        └───────────────┴──── logging ─────┴─────────────────┴──────────┘
                                                     │
                                        assignment ──┴──► analysis ──► decision
                                                     │
                                              feedback loop (next training set)
```

Four failure categories cut across those stages. Naming the category first prevents the classic mistake of debugging a *measurement* problem with *infrastructure* work.

**VIOLATED_ASSUMPTION.** The system is computing what it was told to compute, but one of the premises that made the computation meaningful is false: independent identically distributed exposure, a complete log, a stable population, an unbiased sample, a stationary click model, a correct key. These issues are dangerous because every dashboard stays green — the code is doing exactly what it says.

**NUMERICAL_EDGE.** The computation is arithmetically wrong at a boundary: division by an empty denominator, a tie broken arbitrarily, a duplicate counted twice, an identifier truncated, a weight normalised over the wrong set, a bucket boundary counted in both windows or neither. These issues are cheap to fix and expensive to notice, because they only show up on small populations where nobody looks.

**SCALE_FAILURE.** The behaviour is correct at the tested size and wrong at production size: shard-local statistics, sampling that drops rare items, top-k truncation before a global merge, timeouts that return partial results as if complete, memory growth with vocabulary. The signature is size-dependence — the defect vanishes on a 1% replay.

**CONCEPT_CONFUSION.** Two teams use one word for two quantities, so the metric that is optimised is not the metric that is reported: "CTR" meaning clicks per impression versus clicks per session; "recall" meaning candidate-set recall versus recall of the final list; "freshness" meaning index lag versus document age; "revenue" meaning attributed versus billed. No amount of code reading finds these; only writing both definitions side by side does.

**Triage order that saves the most time.** Check CONCEPT_CONFUSION first, because it costs a conversation and invalidates everything downstream. Then NUMERICAL_EDGE, because a five-line query settles it. Then VIOLATED_ASSUMPTION, which needs a controlled comparison. Then SCALE_FAILURE, which needs a replay at production size and is therefore the most expensive — which is exactly why it should be last, not first.

A second mental model is the **two clocks**. Every count in this domain is either event-time (when the user acted) or processing-time (when the system learned about it). Rolling windows, freshness SLAs and experiment exposure counts all behave differently depending on which clock the implementation uses, and mixing them produces a family of issues that look like data loss but are pure ordering semantics.

A third mental model is the **exposure cascade**. An item's measured quality is a product of its intrinsic quality and the attention the layout gave it. Position, module height, thumbnail, and autoplay all multiply that attention. Any metric computed on logged data therefore measures the *product*, and every correction in this sheet is an attempt to divide out the attention factor — either by randomising it (experiments), by modelling it (propensity), or by holding it fixed (counterfactual replay with a frozen layout).

## 3. Foundational Definitions

Each definition states the quantity, its unit, and the range in which it is meaningful. A definition without a unit is where concept confusion starts.

**Impression.** One rendering of one item in one response to one request. Unit: count. Valid range: non-negative integer; a response with ten slots contributes up to ten impressions, which is why impression counts exceed request counts and why "CTR per request" and "CTR per impression" differ by roughly the slot count.

**Click.** A user activation on an impressed item that the logging definition accepts (tap, dwell-qualified tap, purchase). Unit: count. The definition, not the event name, is the contract: changing the dwell threshold silently rescales every historical CTR series.

**Position propensity.** The probability that the logging policy placed an item in a given slot, conditional on the request. Unit: probability in (0, 1]. It is a property of the *policy*, not the item, and it must be recorded at serving time; reconstructing it later from logs is inference, not measurement.

**Exposure elasticity.** The multiplier relating attention at a slot to attention at the reference slot. Unit: dimensionless. Reference slot has elasticity 1.0 by construction. Used to explain why the same item earns different CTR in different slots, and never to be conflated with propensity, which is about placement probability rather than attention.

**Inverse propensity score (IPS).** The weight `1/π` applied to an observed outcome so that a policy-evaluation average is unbiased for the target policy. Unit: dimensionless weight. Valid only where the target policy puts support on actions the logging policy could have taken.

**Effective sample size (ESS).** `(Σw)² / Σw²` for weights `w`. Unit: an equivalent count of unweighted observations. Range: at most the number of observations, and it collapses quickly when a few weights dominate, which is the standard warning that an IPS estimate is numerically unstable rather than merely noisy.

**Relevance label.** A graded judgement of an item for a query or context, usually 0–4 or 0–3. Unit: ordinal grade, not a cardinal quantity, which is why averaging labels is a modelling assumption rather than arithmetic.

**Discounted cumulative gain (DCG).** A rank-weighted sum of graded relevance. Unit: gain units. Uses a logarithmic discount so a gain at rank 1 is worth more than the same gain at rank 10.

**Ideal DCG (IDCG).** The DCG of the same multiset of gains sorted in descending grade order. Unit: gain units. It is the denominator of NDCG and is defined on the *label multiset*, never on the system output — the single most common source of NDCG greater than one.

**Recall@k.** Fraction of the relevant items for a request that appear in the first `k` of a stage's output. Unit: ratio in [0,1]. Ambiguous unless the stage is named: candidate-generation recall and final-ranking recall are different quantities with different ceilings.

**Coverage.** Fraction of the eligible catalogue that received at least one impression in a window. Unit: ratio in [0,1]. Its complement is the tail that never gets a chance to earn a click, which is why coverage is a fairness guardrail rather than a relevance metric.

**Novelty and serendipity.** Novelty is the fraction of recommended items the user has not previously interacted with; serendipity is the fraction that is both novel and relevant. Unit: ratio. Serendipity requires a relevance judgement, so it can never be computed from click logs alone.

**Event time.** The timestamp attached to the user action at the source. Unit: epoch milliseconds. It can arrive out of order and can be arbitrarily late.

**Processing time.** The wall-clock instant at which the system observed or committed the event. Unit: epoch milliseconds. It is monotone per partition and therefore tempting to use for windows, but it silently rewrites history whenever a backfill or replay occurs.

**Watermark.** A declaration that all events with event time below a bound have been observed. Unit: epoch milliseconds. It is an assumption with a probability of being wrong, not a guarantee, and the wrongness probability is the only thing that makes a window's error budget computable.

**Lateness.** Event time minus arrival time for one record, or the distribution of that difference over a stream. Unit: seconds, reported as a distribution with a high quantile. A single mean lateness figure is nearly useless for window design.

**Tumbling window.** A fixed `[t, t+r)` partition of the timeline into non-overlapping buckets. Unit: seconds. Boundaries belong to the left-closed, right-open convention, and every consumer must agree on it.

**Sliding window.** A window of length `W` evaluated every `r` seconds. Unit: seconds. Its cost is the number of buckets read per evaluation, `ceil(W/r)`, which is the dominant design trade-off against boundary precision.

**Hopping window.** A variant of sliding where the emit interval differs from the bucket size; useful when you need per-minute curves but only two-minute refresh.

**Counter snapshot.** A periodic dump of accumulated counts keyed by a dimension tuple. Unit: count, plus its as-of timestamp. Snapshots introduce a rounding error of at most one refresh interval in every derived rate, which must be stated whenever a rate is quoted to four decimal places.

**Tombstone.** A delete marker retained so that merges and replicas agree that a document is gone. Unit: record. Its retention period bounds how long a resurrect-after-replay bug can occur.

**Index lag.** Processing time of the newest committed document minus its event time, at a percentile. Unit: seconds. Reported per shard and per language, because a single aggregate hides the shard that is failing.

**Shard-local statistic.** A corpus statistic (document frequency, average length, vocabulary size, popularity) computed inside one partition. Unit: varies. Its use in a scoring function that assumes global statistics is a SCALE_FAILURE by construction, and the error grows as partitions get smaller.

**Score calibration.** Agreement between a model score and an observed frequency. Unit: probability versus score. A ranker only needs monotone scores; an auction, a threshold, or a blended objective needs calibrated ones, and confusing the two is the classic auction-overpay defect.

**Auction reserve and blending.** The rule that converts multiple objectives into one ranked list, usually a weighted sum of bids and relevance with a floor. Unit: money per impression. Its weights are policy parameters, so a ranking-metric regression and a revenue drop can coexist without contradiction.

**Session.** A bounded interaction episode, defined by an inactivity timeout. Unit: episode. Session-scoped metrics change discontinuously when the timeout changes, which is why timeout values belong in the metric definition and not in a config file nobody reads.

**Guardrail metric.** A quantity that must not degrade beyond a stated tolerance even if the primary metric improves. Unit: whatever the guardrail measures. It must have a pre-declared tolerance, otherwise it becomes an argument after the fact.

**Sample ratio mismatch (SRM).** A statistically significant deviation of observed assignment counts from the designed ratio. Unit: dimensionless test statistic. It invalidates the comparison before any effect size is discussed, because the assignment mechanism, not the treatment, is what produced the imbalance.

**Selection bias.** A distortion caused by conditioning the analysis on a post-treatment variable. Unit: not measurable directly. In ranking it usually appears as analysing only the impressions that received a click, which conditions on the outcome.

**Feedback loop.** The process by which logged exposure determines the next training distribution. Unit: dimensionless amplification factor per cycle. It converts an exploration deficit into a compounding relevance deficit.

**Dedupe key.** The tuple used to collapse repeats into one logical item. Unit: identifier. Choosing item id versus canonical group id changes both the metric denominator and the user experience; the choice must be documented next to the metric, not buried in code.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

The base case is a randomised exposure with uniform propensity and a complete log. Every correction in this sheet is a departure from it, so its arithmetic must be exact.

```text
FORMULA      theta_IPS = (1/n) * sum over i of  (o_i * pi_target(a_i | x_i) / pi_log(a_i | x_i))
WHERE        n         = number of logged requests in the evaluation window
             o_i       = observed outcome for request i, in {0,1} for a click
             a_i       = action (slot and item set) actually shown
             pi_target = probability the candidate policy assigns to that action
             pi_log    = probability the logging policy assigned to it, recorded at serve time
CONSTRAINT   pi_log(a_i|x_i) > 0 for every action with pi_target > 0  (support condition)
             o_i independent of assignment conditional on x_i          (no hidden confounder)
NUMERICAL    ILLUSTRATIVE: 2,000,000 impressions, three slots with logging shares
             w = [0.50, 0.30, 0.20] and elasticities e = [1.00, 0.62, 0.41]
             intrinsic click rate c = 0.04
             observed clicks = 2,000,000 * c * sum(w_i * e_i)
                             = 2,000,000 * 0.04 * 0.768
                             = 61,440   -> naive CTR = 61,440 / 2,000,000 = 0.030720
             clicks attributable to slot 1 = 2,000,000 * 0.50 * 1.00 * 0.04 = 40,000
             target policy puts every impression in slot 1, so pi_target = 1 there and 0 elsewhere
             IPS = 40,000 * (1 / 0.50) / 2,000,000 = 80,000 / 2,000,000 = 0.040000
             weights are 2.0 on 1,000,000 slot-1 impressions and 0 elsewhere
             ESS = (1,000,000 * 2.0)^2 / (1,000,000 * 4.0) = 1,000,000
             relative standard error = sqrt(0.04 * 0.96 / 1,000,000) = 0.196%
INTERPRETATION  The naive figure 3.072% is 23.2% below the true 4.000% purely because two thirds of
             impressions sat in slots with less attention. IPS recovers 4.000% exactly here because
             the example has no randomness in the click outcomes; in a real log the same estimator
             carries the 0.196% sampling error and, more importantly, only the slot-1 impressions
             contribute information, so the estimate is much noisier than its denominator suggests.
VALIDITY     Holds while propensities are logged, the target policy's actions are all supported by the
             logging policy, and the window's traffic mix is the mix you intend to generalise to.
```

The second base quantity is the rank-weighted metric, because most relevance regressions are argued in NDCG units.

```text
FORMULA      DCG@k  = sum from i=1 to k of  g_i / log2(i + 1)
             IDCG@k = same sum over the top-k gains of the label multiset sorted descending
             NDCG@k = DCG@k / IDCG@k
WHERE        g_i    = graded relevance of the item at rank i, in {0,1,2,3} for a 4-grade scale
             i      = 1-based rank after deduplication
             k      = cut-off, typically 10 for a feed page and 20 for a search page
CONSTRAINT   the denominator uses the same multiset of labels the system was allowed to rank
             duplicates are collapsed before both the numerator and the denominator are built
NUMERICAL    ILLUSTRATIVE grades g = [3,2,3,1,2,3,1,0,2,1] at k = 10
             discounts 1/log2(i+1) = 1.000, 0.631, 0.500, 0.431, 0.387, 0.356, 0.333, 0.315, 0.302, 0.290
             DCG    = 3.000 + 1.261 + 1.500 + 0.431 + 0.774 + 1.069 + 0.333 + 0 + 0.604 + 0.290
                    = 9.2593   (recomputed term by term)
             ideal order [3,3,3,2,2,1,1,0] gives IDCG = 9.6901
             NDCG@10 = 9.2593 / 9.6901 = 0.9555
INTERPRETATION  A single swap of grades 2 and 3 at ranks 1 and 3 costs about 0.06 NDCG in this
             example, so a reported 0.02 movement is inside the noise of label disagreement rather
             than evidence of a better ranker.
VALIDITY     Holds for fixed k, fixed grade scale and a deduplicated candidate list; NDCG is not
             comparable across different k, different grade scales, or different label sets.
```

### 4.2 General case

The general case allows non-uniform propensities, multiple objectives and partial logs, which is where most real systems live.

```text
FORMULA      CTR_blend = sum over j of  beta_j * (clicks_j / impressions_j)
             with sum(beta_j) = 1 and each ratio computed on the same population and window
WHERE        j        = objective index (click, dwell-qualified click, purchase, ad revenue proxy)
             beta_j   = policy weight for objective j, dimensionless
             denominators are per-objective impression counts, not a shared one
CONSTRAINT   every ratio must share population, window and dedupe key, otherwise the sum mixes
             quantities and the beta weights stop meaning what the policy intended
NUMERICAL    ILLUSTRATIVE: clicks_j = 61,440 over 2,000,000 impressions gives 0.030720
             qualified clicks = 18,432 over the same impressions gives 0.009216
             purchases = 1,229 over the same impressions gives 0.0006145
             beta = [0.6, 0.3, 0.1] -> blend = 0.6*0.030720 + 0.3*0.009216 + 0.1*0.0006145
                                            = 0.018432 + 0.0027648 + 0.00006145
                                            = 0.02125825
INTERPRETATION  The blend is dominated by the click term because its rate is the largest, so a 1%
             relative move in clicks shifts the blend by 0.000184 while a 10% move in purchases
             shifts it by 0.0000061. Stated plainly: a weighted sum of rates with very different
             magnitudes is a click objective with decoration, and if that is not the intent the
             weights must be rescaled by a reference rate.
VALIDITY     Holds only when all three ratios are computed on one population; the moment one term
             uses a session denominator and another an impression denominator the blend is a unit error.
```

```text
FORMULA      SRM statistic  chi2 = sum over arms a of (O_a - E_a)^2 / E_a,  df = k - 1
WHERE        O_a = observed assignments to arm a in the analysis window
             E_a = total assignments * designed share of arm a
             k   = number of arms
CONSTRAINT   the test concerns counts of assignments, not outcomes; it must be run before any
             outcome metric is inspected, and on the raw assignment table rather than a joined one
NUMERICAL    ILLUSTRATIVE: O = [1,000,000 ; 1,020,000], designed 50/50, total 2,020,000
             E = [1,010,000 ; 1,010,000]
             chi2 = (10,000)^2 / 1,010,000 + (10,000)^2 / 1,010,000
                  = 99.0099 + 99.0099 = 198.0198
             critical value at alpha = 0.05 with df = 1 is 3.8415
INTERPRETATION  A 2% traffic imbalance that most reviewers would wave through is 51 times the
             critical value. The cause is a hash, a filter or a redirect, not the treatment, so the
             outcome comparison is void until the assignment path is fixed.
VALIDITY     Valid when assignment is intended to be independent and identically distributed across
             requests; if assignment is deliberately clustered (by user, by device), the test must be
             run at the randomisation unit or it will fire on expected within-cluster correlation.
```

### 4.3 Variants

**Self-normalised IPS.** Dividing by the sum of weights instead of `n` bounds the estimate inside the observed outcome range and is the default when weights are heavy-tailed. It trades a small bias for a large variance reduction, and it must be reported as a different estimator, because mixing self-normalised and plain IPS across reports makes two numbers incomparable.

**Doubly robust estimation.** Combine a direct model prediction with the IPS residual. The estimator is consistent if *either* the propensity model or the outcome model is right, which is why it is preferred when propensities were reconstructed rather than logged. The price is a second model whose own errors are invisible in the final number.

**Capped IPS.** Clip weights at a quantile, for example the 99th percentile. Capping reduces variance and introduces bias whose direction depends on where the clipped mass sits; report the cap value and the fraction of observations clipped, because a cap that clips 8% of rows is not the same estimator as one that clips 0.4%.

**Window variants for freshness.** A point-in-time counter snapshot gives an exact answer at one instant with a bounded staleness; a rolling sum over buckets approximates a moving window with boundary error; a streaming sketch gives an approximate distinct count with a known relative error. The three are not interchangeable, and quoting them to the same number of decimals is a defect in itself.

**Metric variants.** NDCG with binary gains degenerates to a discounted precision, which is why binary-gain NDCG and graded NDCG cannot be compared. Recall@k with `k` set to the candidate budget measures the generator; with `k` set to the page size it measures the whole system. Both are legitimate, and naming only "recall" is a CONCEPT_CONFUSION by construction.

### 4.4 Limits and counterexamples

```text
COUNTEREXAMPLE 1 — duplicates inflate NDCG above 1
             Same label multiset and same IDCG = 9.6901 as 4.1, but the served list repeats a
             grade-3 item at rank 7 where the honest list had a grade-1 item:
             served grades [3,2,3,1,2,3,3,0,2,1]
             DCG  = 3.000 + 1.262 + 1.500 + 0.431 + 0.774 + 1.069 + 1.000 + 0 + 0.602 + 0.289
                  = 9.9260
             NDCG = 9.9260 / 9.6901 = 1.0243
             The single repeated item adds 0.667 of DCG (rank-7 term rises from 0.333 to 1.000),
             which is larger than the entire gap between a good and a mediocre ranker.
             A metric that can exceed 1 is not a metric; the defect is duplicate counting, not the
             discount function. The discriminating check is a single assertion: max(NDCG) <= 1.
COUNTEREXAMPLE 2 — propensity support fails
             Suppose the logging policy never placed ads in slot 5 and the target policy does.
             Then pi_log(slot 5) = 0 for every logged request, the weight is infinite, and the
             estimate is undefined rather than merely biased. No amount of data fixes this; only a
             randomised slot-5 holdout does. The check is to compute, per target action, the number
             of logged requests with positive propensity — a zero count is a blocker.
COUNTEREXAMPLE 3 — window boundary double counting
             A tumbling window [0,300) and the next [300,600) both include an event stamped exactly
             300.0 s if the implementation uses <= on both sides. With 30 s buckets and 9,000 events
             per window, a boundary population of even 1% of the traffic injects 90 duplicate
             events per boundary, which at 15 boundaries per hour is 1,350 phantom events.
             The check is the equality test sum(window_i) == total for a closed interval.
VALIDITY CONDITION FOR SECTION 4 AS A WHOLE
             Every formula here assumes a log whose population, window and dedupe key are stated.
             Where any of the three is unstated, the number is not wrong, it is uninterpretable,
             and the correct action is to state the missing item rather than to compute harder.
```

## 5. Worked Numerical Example

ILLUSTRATIVE audit. A weekly relevance review receives four numbers that disagree with each other. This walkthrough reconciles them and then runs three independent arithmetic checks. Constructed inputs throughout; the arithmetic is exact to the digits shown.

**The four reported numbers.** (1) The dashboard says CTR = 2.3550%. (2) A product manager says CTR = 18.84% "per request". (3) A ranker report says NDCG@10 = 0.9555 with a note that a duplicate-tolerance patch raised it to 1.0243. (4) A shard report says the ranking score for a head term is 2.53 times the fleet-wide value. Four numbers, four different populations, one pipeline.

**Step 1 — reconcile the two CTR values.** The population is 7,000,000 requests over seven days in UTC, each filling 8 slots, giving 56,000,000 impressions. Logged clicks are 1,318,800.

```text
CHECK 1  clicks / impressions = 1,318,800 / 56,000,000 = 0.0235500
         clicks / requests    = 1,318,800 / 7,000,000  = 0.188400
         ratio of the two     = 8.0000  = impressions per request
         so the two figures are the same quantity twice; "18.84% per request" is 0.1884 clicks per
         request, a count-like rate with a ceiling of 8, not a probability with a ceiling of 1
INTERPRETATION  Both numbers are arithmetically right and neither is a defect. The defect would be
         to compare the 18.84% against a benchmark quoted per impression, which understates the
         system by a factor of 8.
```

**Step 2 — decide whether 2.3550% is a good number.** With 8 slots whose attention elasticities are `[1.00, 0.86, 0.72, 0.60, 0.50, 0.41, 0.34, 0.28]`, the mean elasticity depends entirely on the slot-mix distribution, and the slot mix is a policy choice, not a property of the ranker.

```text
CHECK 2  uniform mix:  mean e = (1.00+0.86+0.72+0.60+0.50+0.41+0.34+0.28)/8 = 5.71/8 = 0.71375
         actual mix:   shares [0.30,0.20,0.15,0.12,0.10,0.06,0.05,0.02], sum = 1.00
                       mean e = 0.30*1.00 + 0.20*0.86 + 0.15*0.72 + 0.12*0.60
                              + 0.10*0.50 + 0.06*0.41 + 0.05*0.34 + 0.02*0.28
                              = 0.3000 + 0.1720 + 0.1080 + 0.0720 + 0.0500 + 0.0246 + 0.0170 + 0.0056
                              = 0.7492
         implied intrinsic rate at the observed 0.023550 = 0.023550 / 0.7492 = 0.031434
         the same intrinsic rate under a uniform mix would report 0.031434 * 0.71375 = 0.022437
INTERPRETATION  The observed 2.3550% sits below the intrinsic 3.1434% by 25.08% because the observed
         slot mix is top-heavy relative to a uniform mix in its high-attention slots but has mass in
         low-elasticity slots too. Two dashboards that disagree by 25% on CTR can both be correct if
         their slot mixes differ; comparing them without the mix is the actual error.
```

**Step 3 — check whether the rolling window explains a slow Monday.** The window is `W = 300 s` evaluated every `r = 30 s`, so it comprises 10 buckets and holds `300 * 30 = 9,000` event-seconds of traffic at 30 events/s. Lateness has a 45-second tail: 0.361% of events arrive more than 45 s after their event time (exponential lateness with mean 8 s, computed in section 4's constraint block).

```text
CHECK 3  coverage of an event-time window closed at processing time L seconds late:
         covered = (W - L) * r / (W * r) = (300 - 45)/300 = 255/300 = 0.8500
         missing events per evaluation = L * r = 45 * 30 = 1,350 of 9,000
         a 15% shortfall is far larger than any realistic week-over-week relevance move
         the exponential tail check: exp(-45/8) = 0.00361, so 0.361% of events are genuinely later
         than 45 s, and the window's shortfall is 15%, not 0.361%
INTERPRETATION  The 15% is not lateness, it is the closing rule. The window is being closed at a
         watermark only 255 s into a 300 s window, which discards 45 s of every window regardless of
         how timely the stream is. Lateness and window-closing are different mechanisms and the
         numbers separate them cleanly: 15% versus 0.361%.
```

**Step 4 — check whether the shard score is a bug or a definition gap.** A single shard holds 100,000 documents and reports document frequency 50 for a term; the fleet holds 1,000,000 documents with document frequency 50,000 for the same term.

```text
CHECK 4  idf_shard = ln(1 + (100,000 - 50 + 0.5)/(50 + 0.5))   = ln(1 + 1979.71) = 7.5910
         idf_fleet = ln(1 + (1,000,000 - 50,000 + 0.5)/(50,000 + 0.5)) = ln(1 + 19.0000) = 2.9957
         ratio = 7.5910 / 2.9957 = 2.5340
INTERPRETATION  Exactly the reported 2.53. Nothing is broken; a shard that is one tenth of the corpus
         computes document frequency on one tenth of the corpus, so a term that is common fleet-wide
         looks ten times rarer locally and its inverse document frequency inflates by 2.5x. The
         correction is a two-level statistic (local frequency for scoring, fleet frequency for
         weighting), and the check that catches it is a comparison of the same term's idf on two
         differently sized shards.
```

**Step 5 — re-derive the 1.0243 to confirm it is duplicate inflation, not a better ranker.** Served grades `[3,2,3,1,2,3,3,0,2,1]`, IDCG from the label multiset `[3,3,3,2,2,2,1,1,1,0]`.

```text
CHECK 5  DCG  = 3.000 + 1.262 + 1.500 + 0.431 + 0.774 + 1.069 + 1.000 + 0.000 + 0.602 + 0.289
              = 9.9260
         IDCG = 3.000 + 1.893 + 1.500 + 1.292 + 0.774 + 0.712 + 0.667 + 0.315 + 0.301 + 0.000
              = 9.6901
         NDCG = 9.9260 / 9.6901 = 1.0243
         the only differing term between the honest 9.2593 and the duplicated 9.9260 is rank 7,
         where 1.000 replaces 0.333; difference = 0.6667 = 9.9260 - 9.2593 exactly
INTERPRETATION  The patch did not make ranking better; it let one item occupy two rank positions.
         A 0.667 DCG gain from a single rank is larger than any plausible whole-system improvement,
         which is itself the diagnostic: implausibly large metric moves are duplicate or leak
         signals before they are ranking signals.
```

**Sensitivity note.** Each check has one input that dominates. For CHECK 1 the denominator is everything: swapping impressions for requests changes the answer by 8x, so a CTR report must name its denominator before it names its value. For CHECK 2 elasticity ratios dominate; shifting the mean elasticity by 0.01 moves the implied intrinsic rate by about 0.0004 absolute, which is the same order as a week-over-week move, so a 1.4% relative change in reported CTR can be pure mix drift. For CHECK 3 the closing lag dominates linearly: reducing the lag from 45 s to 5 s raises coverage from 85.00% to 98.33%, and no change to the stream's punctuality achieves that. For CHECK 4 partition size dominates: the idf ratio is approximately `ln(N/k)/ln(N/(10k))` for small `k/N`, so halving the shard size moves the ratio by roughly 1.2x, not by a few percent. For CHECK 5 the metric's own structure dominates: because rank 7's discount is 0.333, a duplicate at that rank is worth three times a genuine improvement at the same rank, so short lists with low discounts are where duplicate bugs hide longest.

**What the audit concluded.** One real defect (CHECK 5, duplicate counting), one definition gap (CHECK 1, denominator unstated), one policy-dependent number that was not comparable across teams (CHECK 2), one window-closing bug masquerading as lateness (CHECK 3), and one genuine scale defect (CHECK 4). The order of investigation matters: CHECK 1 cost one division, CHECK 5 cost one assertion, and CHECK 4 cost a cross-shard comparison. None of them required a model retrain, which is the general shape of this domain — most reported relevance problems are measurement problems first.

## 6. Visual / Geometric Intuition

**The discount curve is a picture of how fast attention decays with rank.** Plotting `1/log2(i+1)` against `i` gives 1.000 at rank 1, 0.631 at 2, 0.500 at 3, 0.431 at 4, 0.387 at 5, 0.356 at 6, 0.333 at 7, 0.315 at 8. In log-log space it is a straight line of slope `-1`, which is why DCG improvements are almost always front-loaded: 63% of the first rank's weight is gone by rank 2, and the curve is flat enough by rank 8 that a duplicate there is worth only a third of a unit. Geometrically, the useful mental picture is a triangular weight profile whose area is `ln(k)` — so doubling `k` adds a constant, not a factor, and comparing NDCG@10 to NDCG@20 is comparing two different normalisations.

**Position elasticity is a staircase, not a line.** Attention does not decay smoothly with vertical offset; it drops at fold boundaries, at module transitions, and at the point where a thumbnail becomes a text row. That is why the eight-elasticity vector in this sheet looks like `[1.00, 0.86, 0.72, 0.60, 0.50, 0.41, 0.34, 0.28]` rather than a clean geometric sequence: the ratio between adjacent slots varies from 0.86 to 0.76 to 0.83 to 0.82 to 0.83 to 0.82, and the deviations are the layout boundaries. Practically, interpolation between slots is safe within a module and unsafe across one.

**The propensity-weighted view is a reweighting of a histogram.** Picture the impression histogram over slots. The naive CTR is the mean of per-slot CTRs weighted by raw impression counts. IPS replaces those counts with `count/π`, which is the histogram as it *would* look under the target policy. Visually the operation stretches the histogram to the right and shrinks it where the logging policy over-sampled. When a slot has zero mass under the logging policy, the stretch factor is infinite and the picture has a hole — that hole is the support failure in 4.4, and it is visible before any number is computed.

**Two clocks make a scatter plot, not a line.** Plot arrival time against event time. Perfectly ordered data lies on the diagonal. Real data is a cloud above the diagonal; its upper envelope is the lateness tail. A window closed at processing time `T` is a horizontal line; everything above the line at that moment is missing, and the missing area is a triangle whose height is the lateness and whose width is the bucket duration. The 15% shortfall in CHECK 3 is the area of that triangle relative to the window rectangle: `(L / W)`.

**Shard-local statistics are a zoom-in problem.** If the fleet-wide document frequency is `f` in `N` documents and a shard holds a fraction `p` of the corpus uniformly, the shard's frequency is `pf` in `pN`, so the ratio `df/N` is preserved and idf *should* be identical. The observed inflation happens because a shard is not a uniform sample: it is hash-partitioned on document id, so a term's local frequency fluctuates around `pf` with standard deviation `sqrt(pf)`. For `pf = 50`, that is 7.07, a 14% relative swing, which through the log in idf becomes the multi-x error. Zooming in changes the statistic's variance, not just its scale.

**Coverage is an area, not a count.** A catalogue of 1,000,000 items with 8 slots per request and 7,000,000 requests has 56,000,000 impressions available. If the head 20,000 items absorb 90% of them, the remaining 980,000 items share 5,600,000 impressions, giving each an expected 5.7 impressions per week. At 5.7 impressions and a 0.5% CTR, most of that tail earns zero clicks, so a click-based model cannot learn about it. The area view makes the feedback loop obvious: the region of the catalogue with no data is exactly the region the model cannot improve, and it stays dark because the model does not rank it.

**Dedupe changes the geometry of the metric, not just its value.** Two rows for the same item occupy two positions and therefore consume two discounts. Collapsing them returns the second discount to the pool, which raises the DCG of every later item by roughly `g/(log2(i+1) log2(i+2))` — a small per-item gain that compounds over a page. That is why dedupe defects look like "the ranker got better" rather than "the list got shorter".

```text
  rank:      1      2      3      4      5      6      7      8
  weight: 1.000  0.631  0.500  0.431  0.387  0.356  0.333  0.315
  cumulative share of total weight lost by this rank:
           0.0%  40.6%  52.9%  59.3%  63.9%  68.1%  71.7%  75.0%
  (computed as 1 - w_i / w_1 against an 8-rank total weight of 3.953)
```

**The two-dimensional triage plane.** Put "population stated?" on one axis and "arithmetic checked?" on the other. The four quadrants map onto the four issue categories: unstated population with correct arithmetic is CONCEPT_CONFUSION; stated population with unchecked arithmetic is NUMERICAL_EDGE; neither is VIOLATED_ASSUMPTION; and both stated and checked yet still wrong at scale is SCALE_FAILURE. Walking the plane from the origin outward is the cheapest possible debug order, and it is why section 2 puts concept checks first.

**Freshness as a queueing picture.** Index lag is the age of the oldest unprocessed update. In a system with arrival rate `λ` and service rate `μ`, the queue is stable only for `λ < μ`, and near saturation the lag explodes non-linearly. The useful intuition is that a shard at 85% utilisation has a mean lag about 5.7 times its service time, while the same shard at 50% utilisation has about 1.0 times — so a small traffic increase can triple perceived staleness without any code change. That non-linearity is why lag alarms must be on the tail percentile and on utilisation, not on mean lag alone.

**The exposure cascade as a product tree.** Request → slot → module height → thumbnail → text snippet → autoplay. Each node multiplies attention by a factor below 1. The product over six levels with factors `0.9, 0.8, 0.85, 0.9, 0.95, 0.7` is `0.9*0.8*0.85*0.9*0.95*0.7 = 0.3661`, so a single layout experiment that changes two levels can move CTR by tens of percent without touching relevance. Any A/B result on a ranking change must therefore be confirmed against a layout-frozen replay, otherwise the treatment effect is a mixture of ranking and layout.

**Why duplicate ratios are reported, not just counts.** The duplicate long-line ratio used by this course's own lint is a picture in miniature: a file whose 1,000 long lines contain 300 repeats has a ratio of 0.30, which is not "30% wasted effort" but "30% of the evidence is the same evidence counted twice". The same geometry applies to impression logs: a repeated impression is a repeated measurement, and averaging repeated measurements as if independent shrinks the reported confidence interval by `sqrt(n_repeats)`.

## 7. Comparison / Selection Table

The first table selects a *diagnostic*, given a symptom. Selecting the wrong diagnostic is the most expensive mistake in this domain because it burns a week of engineering on a healthy component.

| Observed symptom | Cheapest first diagnostic | Cost of the check | What a positive result proves | What a negative result rules out |
|---|---|---|---|---|
| Metric looks implausibly good | Assert `max(NDCG) <= 1` and `min(NDCG) >= 0` on the eval set | Minutes: one assertion | Duplicate or label-leak defect present | Metric formula errors, not the ranker |
| CTR fell 4% week over week | Compare slot-mix distribution between the two weeks | One histogram query | Mix shift, not relevance, explains the drop | Composition effects; leaves ranker and logging |
| Two teams report different CTR | Write both definitions with denominator and window | One conversation | Concept confusion is the whole disagreement | Arithmetic; forces a single shared definition |
| Traffic split is uneven | Pearson chi-square on assignment counts, df = 1 | One query, critical value 3.8415 | Assignment bug; outcomes are void | Assignment path; treatment effect analysis may proceed |
| Results are stale | Compare `max(processing_time - event_time)` per shard percentile | One log query | One shard's indexer is behind, not the fleet | Fleet-wide freshness; localises the shard |
| Same item appears twice | Group impressions by request and count distinct item ids | One query on a request sample | Dedupe key or merge race at serve time | Metric duplication; the defect is in the response |
| Recall@k dropped after a model change | Compare candidate-set recall and final-list recall separately | Two eval runs | The generator shrank, not the ranker's ordering | Ranker quality; narrows to candidate generation |
| Revenue fell but relevance rose | Check score calibration against observed rates | Reliability plot on a holdout | Auction threshold now mis-set relative to scores | Ranker quality; points at the policy layer |
| Tail latency p99 tripled | Read index lag and merge rate for the same window | Two dashboard panels | Merge backlog is the cause, not query cost | Query planning; localises to indexing |
| Guardrail breached with no primary-metric change | Recompute both metrics on the identical population | One join | The two metrics use different populations | A real guardrail regression; it is a unit mismatch |
| Click rate higher for a new module | Freeze layout and replay with a fixed slot policy | Half a day of replay | Position effect, not item quality, drove the lift | Item quality claims; the layout must be held fixed |
| Model performance fell after a retrain | Diff training-window exposure distribution against serving | One distribution comparison | Feedback-loop amplification, not label drift | Training pipeline bugs; points at exploration |

The second table selects a *statistic* for a rolling-window requirement, because most freshness disputes are really disputes about which counting mechanism was promised.

| Requirement | Mechanism | Error character | Cost per query | When it is the right choice |
|---|---|---|---|---|
| Exact count at a moment, tolerated staleness | Point-in-time counter snapshot | At most one refresh interval of rounding | One key read | Admission quotas where a 60 s window is acceptable |
| Approximate moving window over event time | Rolling sum over tumbling buckets | Boundary error of up to one bucket at each edge | `ceil(W/r)` key reads | Dashboards needing per-minute curves |
| Distinct-user counting | Hyperloglog sketch, 4,096 registers | 1.625% relative standard error, unbiased | One sketch merge per partition | Reach and unique-visitor metrics |
| Strict event-time window with late data | Watermark plus allowed lateness and retraction | Bounded by lateness distribution, exact after the watermark passes | Streaming state, no re-reads | Billing or quota decrements that must not double count |
| Near-exact moving window at low latency | Two adjacent buckets with linear interpolation | Linear between refresh points; worst case half a bucket | Two key reads | High-QPS counters where 0.5% boundary error is acceptable |
| Exact total with no window | Monotone cumulative counter | None, but no window semantics at all | One key read | Lifetime counts; never for rate limits |

**How to read the second table.** The snapshot row and the interpolation row differ by a factor of two in read cost and by an order of magnitude in boundary precision. Teams that pick the interpolated variant for a *billing* counter are trading a refund dispute for a saved read; teams that pick the snapshot for a *rate limit* are trading a 60-second burst window for a saved read. Both trades are legitimate, but only when stated — and the statement belongs in the metric definition, not in the code comment.

**Selection rule that survives review.** Choose the mechanism whose *error character* matches the consequence of being wrong, not the one with the lowest latency. If being wrong means a wrong number in a weekly report, pick the cheapest mechanism with a stated error bound. If being wrong means an over-delivery of a paid impression, pick the exact mechanism and accept the state cost.

## 8. Failure Modes & Edge Cases

Six mechanisms generate almost every defect in the catalogue below. They are described first as *mechanisms* because a mechanism tells you what to look at; the catalogue then gives the per-issue discriminating check.

**Mechanism A — the measured quantity is a product, not the quantity you meant.** Any logged metric equals intrinsic quality multiplied by attention multiplied by logging completeness. Because multiplication is commutative, the same 8% move can be produced by a genuine relevance change, a slot-mix change, or a 4% drop in impression logging. Competing causes therefore always include at least one *logging* cause and one *policy* cause. The discriminating check is to hold one factor fixed: a layout-frozen replay isolates ranking, and a slot-mix histogram isolates policy. Edge case that breaks the reasoning: if the layout change altered the number of slots, the replay is no longer comparable because the candidate budget changed too — a nine-slot replay against an eight-slot log changes both attention and supply.

**Mechanism B — a window boundary belongs to two windows or neither.** Left-closed right-open is the only convention that partitions a timeline without gaps or overlaps, and every consumer must implement it identically. With 30-second buckets and 9,000 events per 300-second window, a 1% boundary population is 90 events per boundary and 1,350 per hour at 15 boundaries. Competing causes: a genuine traffic spike at the top of each minute (cron-driven traffic does this) versus double counting. The discriminating check is the conservation identity `sum(all windows) == total distinct events` computed on one day; a double count makes the sum exceed the total by exactly the boundary mass, while a real spike leaves the identity intact. Edge case: a stream that retracts and re-emits on late data will break the identity even when boundaries are correct, so run the check on a window where no retraction occurred.

**Mechanism C — a corpus statistic computed on a partition is used as if global.** Document frequency, average document length, vocabulary size, popularity, and distinct-user counts are all corpus statistics. When computed per shard, their sampling variance is `sqrt(pf)` for a frequency `pf`, and the log in inverse document frequency amplifies the swing. In this sheet's example, a term with fleet frequency 5% has shard frequency 5% of 100,000 = 5,000... but the observed case was 50 documents in 100,000 against 50,000 in 1,000,000, which is a 10x scale difference with the *same* relative frequency — and the idf still differed by 2.534x, which proves the defect is not variance but scale: idf is not scale-invariant because of the `+0.5` smoothing and the `1 +` term. Correcting this requires fleet-level statistics refreshed on a schedule, with a staleness bound stated as part of the score definition. Edge case: a shard that genuinely holds a language subset cannot use fleet statistics without hurting that language's ranking, so the correct design is per-language statistics, not one global number.

**Mechanism D — time is recorded in two clocks and compared as if one.** Event time is when the user acted; processing time is when the pipeline saw it. Windows, freshness SLAs, and experiment exposure all read one or the other. Competing causes for a stale-looking feed: genuinely old documents (event time old, processing time recent), a slow indexer (processing time old), or a clock-skewed producer whose timestamps are 350 ms behind. The discriminating check is the joint distribution of `(event_time, processing_time)` for a 1% sample: a parallel band offset from the diagonal is skew; a horizontal band far below the diagonal is index lag; a cloud at the far left is genuinely old content. Edge case: NTP corrections can step a producer clock backwards, producing events whose lateness is negative; any implementation that clamps negative lateness to zero hides the step, and any implementation that trusts it will produce a watermark that never advances past the stepped-back point.

**Mechanism E — the feedback loop converts exploration debt into relevance debt.** The training set is the set of items the policy chose to show. If a segment receives 5.7 impressions per week, no click-based model can learn about it, so it will not be ranked, so it will not be shown. Competing causes for a segment's poor metrics: genuinely lower quality, cold-start absence of features, or the loop itself. The discriminating check is a randomised exposure holdout on that segment for a fixed budget: if the segment's CTR under random exposure is close to the head's, the loop is the cause; if it is far lower, quality is. Edge case: a holdout that shows items outside the model's supported score range can produce nonsense engagement (for example, items whose metadata is missing entirely), so the holdout must be filtered to items with complete features or it measures broken rendering rather than quality.

**Mechanism F — an identifier is used as a key in two places with different normalisation.** Case folding, Unicode normalisation, trailing whitespace, URL query ordering, and truncation all change whether two records are "the same". Competing causes for duplicate impressions: serve-time merge race, a dedupe key that ignores variant ids, and a client retry. The discriminating check is a request-level sample: group impressions by request id and count distinct canonical ids; duplicates inside one request are serve-time, duplicates across requests within one second are retries, and duplicates across minutes are key-definition. Edge case: two legitimate items can share a canonical group (a size variant family), so collapsing by canonical id changes the metric denominator and can hide genuine inventory; the dedupe key must therefore be declared separately for *display* and for *metric* purposes.

**Catalogue, part one (CI-01 … CI-18).** Columns follow the contract: id, category, symptom, competing causes, discriminating check, correction, prevention, verification. Every check is runnable with a log query or a unit test on an eval set; none requires a model retrain to *diagnose*.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-01 | NUMERICAL_EDGE | NDCG above 1.0 on an eval slice | Duplicate items in the served list; label multiset smaller than the served list; IDCG built from the served grades instead of the label multiset | Assert `0 <= NDCG <= 1` for all 100,000 eval requests and print the max with its request id | Collapse duplicates before scoring; build IDCG from the label multiset | Unit test that synthesises a duplicated list and expects rejection, not a score | Max NDCG over the eval set equals or falls below 1.0000 in CI |
| CI-02 | CONCEPT_CONFUSION | Two dashboards disagree on CTR by 8x | Denominator is impressions versus requests; window differs by timezone; one excludes bot traffic | Divide the two reported rates; a quotient of exactly the slot count identifies the denominator bug | Publish one metric contract naming numerator, denominator, population, window | Metric dictionary reviewed at definition time, with the denominator in the metric name | Contract test asserts each dashboard emits the same rate for a fixed fixture |
| CI-03 | VIOLATED_ASSUMPTION | Offline evaluation improves but online experiment is flat | Propensity support absent for the new actions; offline labels from a different population; experiment under-powered | Compute the fraction of target actions with zero logged propensity; any nonzero count invalidates the offline claim | Add exploration or restrict the target policy to supported actions | Log propensities at serve time and keep an exploration floor | Support coverage report shows non-zero propensity for every action the target can take |
| CI-04 | NUMERICAL_EDGE | Rollout metric moves 15% when nothing shipped | Window closed before the watermark; boundary double count; timezone-aligned backfill overlapping | Conservation identity `sum(window counts) == total distinct events` for one closed day | Close windows at the watermark plus allowed lateness; enforce left-closed right-open | Shared window utility module used by every consumer | Identity check runs hourly and alerts on any nonzero residual |
| CI-05 | SCALE_FAILURE | Score for a head term is 2.5x the fleet value | Shard-local idf; shard-local average length; per-shard popularity normalisation | Compare the same term's idf on a small and a large shard | Two-level statistics: local term frequency, fleet-level idf | Statistics service with a versioned snapshot id stamped on every score | Scores recomputed with a pinned statistic version are reproducible within 1e-6 |
| CI-06 | NUMERICAL_EDGE | Impression count exceeds request count by more than the slot count | Duplicate logging on retry; phantom impressions for unfilled slots; client prefetch counted as render | Group by request id: `distinct slots` versus `impression rows` per request | Drop phantom rows for empty slots; dedupe retries on a request key | Impression emitted once at render confirmation, keyed by request and slot | Daily assertion that impressions per request never exceeds configured slots |
| CI-07 | VIOLATED_ASSUMPTION | Guardrail metric degrades while primary metric improves and reverts on the third day | Novelty decay from over-exploitation; composition shift from a new segment; genuine guardrail harm | Recompute the guardrail on the identical population as the primary metric; then split by segment | Rebalance the objective or cap the offending objective's weight | Pre-declared guardrail tolerance and a segment-level readout in the experiment template | Experiment sign-off requires segment breakdowns, not just the aggregate |
| CI-08 | CONCEPT_CONFUSION | Candidate recall quoted as 0.95 but users still report missing items | Recall@k measured at candidate budget versus at page size; relevant set built from clicks only | Recompute recall with the page-size cut-off and a label-based relevant set | Report both stage recall and end-to-end recall, labelled by stage | Metric naming convention that includes the stage and the cut-off | Eval harness refuses a recall metric without a stated stage and k |
| CI-09 | SCALE_FAILURE | Tail queries return noticeably worse results than head queries | Vocabulary growth exceeded per-shard capacity; rare-term postings absent from the merged index; score truncation before merge | Compare recall for the tail against a brute-force oracle on 500 sampled tail queries | Merge per-shard top-k with a global heap; guarantee every shard contributes | Load test at production vocabulary size, not the sample | Oracle comparison on tail queries stays within the declared tolerance |
| CI-10 | VIOLATED_ASSUMPTION | Click model over-predicts engagement for low slots | Position bias unmodelled; click definition changed mid-window; bot filtering introduced | Fit observed CTR per slot against a model with and without a slot term; the slot term's coefficient reveals the bias | Add position as a feature or apply inverse propensity weighting | Log slot at serve time and keep the click definition versioned | Calibration plot per slot within a stated band over a holdout day |
| CI-11 | NUMERICAL_EDGE | Session count jumps 50% after a config change | Session timeout changed; timezone rollover; late events re-opening a closed session | Histogram of inter-event gaps before and after the change; the mass above the old timeout is the delta | Version session definitions and recompute the comparison window under one definition | Session timeout stored in the metric definition and change-reviewed | Backfill under both definitions reproduces the reported delta exactly |
| CI-12 | SCALE_FAILURE | Results are stale for one language only | One shard's indexer is behind; merge backlog; per-language pipeline starved by a queue | Per-shard p99 of `processing_time - event_time` for that language | Drain the backlog and rebalance shards by language volume | Lag alarm on p99 per shard and on queue depth per language | Lag dashboard shows every shard under the freshness budget |
| CI-13 | VIOLATED_ASSUMPTION | A deleted item keeps appearing for hours | Tombstone retention shorter than the merge horizon; replica not yet merged; cache not invalidated | Replay the delete against a replica built at the boundary of the retention window | Extend tombstone retention beyond the maximum merge and replay horizon | Retention derived from the merge SLA, documented with the SLA | Deleted-item probe suite shows zero impressions 2 hours after delete |
| CI-14 | CONCEPT_CONFUSION | "Revenue" reported by ranking and by billing differ by 4% | Attributed versus billed revenue; refunds applied in one pipeline only; currency conversion date | Reconcile one day at the transaction id level and count mismatched ids | Publish both definitions with the reconciliation delta as an accepted tolerance | Revenue metric carries its attribution window in the name | Daily reconciliation residual stays inside the tolerance |
| CI-15 | NUMERICAL_EDGE | Average document length drifted but the field says unchanged | Stored average computed on the pre-filter corpus; recomputation excluded stop words; incremental update did not refresh the average | Recompute the average over the served documents and compare to the stored value | Recompute statistics on the same filtered corpus used for indexing | Statistics snapshot validated against a sample recomputation on every build | Recomputed and stored averages agree within the stated rounding |
| CI-16 | SCALE_FAILURE | p99 latency tripled during a content spike | Merge backlog; query fan-out unchanged but postings longer; cache hit rate collapsed | Read merge rate and cache hit ratio for the same minute as the latency spike | Throttle ingestion or add index capacity; do not change query planning first | Capacity model relates documents per second to merge throughput | Load test replaying the spike reproduces latency within the budget |
| CI-17 | VIOLATED_ASSUMPTION | Experiment shows a lift only in the first three days | Novelty effect; logging pipeline backfill landed mid-experiment; assignment changed at day three | Plot the daily effect and the assignment counts by day; a step in assignment counts indicates a pipeline change | Analyse the post-change window only, or rerun with a fixed assignment version | Assignment version stamped on every exposure row | Analysis output refuses rows whose assignment version is not constant |
| CI-18 | NUMERICAL_EDGE | Top-k results contain fewer than `k` items on some requests | Filter applied after truncation; ACL filtering removed candidates; dedupe collapsed variants below `k` | Count returned items per request and correlate with filter pass rate | Fetch a larger candidate budget than the filtered output needs | Contract test asserting `len(results) == min(k, eligible)` | Distribution of returned counts has no mass below `k` for eligible requests |

**Catalogue, part two (CI-19 … CI-28).** These are the entries that survive first-pass triage and are usually discovered only after a metric has already been trusted once.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-19 | VIOLATED_ASSUMPTION | Sample ratio mismatch of 2% that nobody investigated | Hash salt changed; eligibility filter differs by arm; redirect dropping one arm's traffic | Chi-square on raw assignment counts with df = 1; 198.02 against a 3.8415 critical value is decisive | Fix the assignment or filter path, then discard the analysis window | SRM check automated before any metric readout | Readout tool refuses to print outcomes when the SRM test rejects |
| CI-20 | NUMERICAL_EDGE | Unique-visitor count is off by roughly 1.6% and never converges | Hyperloglog sketch precision; sketch merge across partitions; dedupe window overlapping midnight | Compare the sketch against an exact distinct count on one hour of traffic | Increase register count or accept and document the 1.625% standard error | Publish the sketch's error budget alongside the metric | Sketch versus exact comparison stays inside two standard errors |
| CI-21 | VIOLATED_ASSUMPTION | A new slot is evaluated as better but the auction overpays | Score not calibrated to probability; reserve price set against the old score distribution; blended objective changed the ranking order | Reliability plot of predicted versus observed rate on a holdout day; a 0.08 prediction against 0.05 observation is a 1.60x overprice | Calibrate scores or set the threshold from the observed rate distribution | Calibration gate before any auction-facing model is promoted | Auction settlement reconciliation shows no systematic over-delivery |
| CI-22 | SCALE_FAILURE | Phrase queries work in staging and fail at production vocabulary | Positions not stored for all fields; analyzer mismatch between index and query time; phrase candidates truncated before verification | Compare a brute-force phrase oracle against served results on 500 frequent-term phrases | Store positions for phrase-enabled fields; align analyzers by version | Analyzer version pinned in the index metadata and asserted at query time | Oracle comparison passes for a suite including repeated-term phrases |
| CI-23 | NUMERICAL_EDGE | Relevance score changes by 3% with no model change | Stored average document length stale; term frequency recomputed from a different tokenizer; boost weight edited in config | Recompute one document's score by hand from the stored statistics and diff against the served score | Rebuild statistics on the current corpus, or pin the statistics snapshot | Score regression fixtures with frozen statistics and frozen tokenizer version | Fixture scores reproducible to 1e-6 across builds |
| CI-24 | VIOLATED_ASSUMPTION | Popularity is measured by impressions and the tail never recovers | Exposure determines the measured popularity; coverage measured on eligible catalogue but computed on the served set; cold items with no impressions treated as zero quality | Compare the impression-weighted popularity rank against a random-exposure popularity estimate | Use a random-exposure or exploration-corrected popularity signal | Coverage reported alongside popularity, with the unserved mass stated | Popularity estimate is stable when exposure is randomised |
| CI-25 | NUMERICAL_EDGE | Deduplicated list is shorter than the page and CTR falls | Dedupe key too aggressive; canonical group collapsed size variants; pagination offset computed pre-dedupe | Group served ids by canonical id and count group sizes on a request sample | Dedupe on display key but keep variants as separate metric units, and page after dedupe | Pagination contract defined post-dedupe and tested | Page fill rate equals the designed slot count after dedupe |
| CI-26 | SCALE_FAILURE | Search times out for 0.4% of requests and those requests look worse | Timeout returns partial results labelled complete; shard fan-out without hedged requests; one slow shard dominates the tail | Compare result counts and recall for timed-out requests against the same queries on a fast replay | Mark partial responses explicitly and exclude them from quality metrics | Budget per shard with a documented degradation contract | Partial responses are counted separately in every quality readout |
| CI-27 | CONCEPT_CONFUSION | Freshness SLA is met while users see week-old items | SLA measures index lag but users see documents that are genuinely old; cache TTL longer than index lag; CDN version pinned | Joint distribution of document age and cache age for served items on one day | Define freshness as the age of the served document, not the lag of the pipeline | Freshness dashboard reports served-document age at p50 and p99 | p99 served-document age is inside the stated budget |
| CI-28 | VIOLATED_ASSUMPTION | Model retrain improves offline AUC but online engagement drops | Feedback loop amplified head bias; training window contains the previous policy's exposure only; feature computed with a different definition at serve time | Compare training-window exposure distribution against serving exposure; a head share gap of more than 20 points explains the drop | Add exploration, reweight training data by exposure, or shadow-score before promotion | Pre-promotion check that compares exposure distributions | Candidate model's exposure profile is within tolerance of the serving profile |

**Edge cases that break the catalogue itself.** The table is a triage tool, not a proof system. Three failure modes apply to it. First, two entries can co-occur and their symptoms can cancel: a duplicate-count inflation of +7% can mask a genuine -7% relevance regression, which is why CI-01's assertion should run *before* any relevance comparison. Second, an entry's discriminating check can be unavailable — no propensity logs means CI-03 cannot be diagnosed, only avoided going forward, and the honest action is to record the missing instrumentation as the defect rather than to guess. Third, a correction can create a new entry: extending tombstone retention to fix CI-13 increases merge cost, which is CI-16's territory, so corrections must be reviewed against the cost section.

**Severity assignment, and why it comes last.** Each entry carries an implicit severity: CI-01, CI-02, CI-03, CI-07, CI-19 and CI-21 are Sev-1 because they corrupt the number a decision is made on; CI-06, CI-08, CI-10, CI-12, CI-15, CI-17, CI-20, CI-23, CI-24, CI-27 and CI-28 are Sev-2 because they mislead a review without necessarily inverting a decision; the remaining entries are usually Sev-3 cost or operability issues unless the response contract explicitly promises completeness. Assign severity after the discriminating check, never before: a symptom that looks Sev-1 often resolves to a Sev-3 cache artefact, and shouting early destroys the credibility you need for the real one.

**Diagnostic recipe 1 — "the number is too good".** Run the bounds assertion first (CI-01), then check the denominator (CI-02), then check whether the evaluation population excludes the hard cases. The third step is the one people forget: an eval set sampled from requests that previously received a click has already removed the queries the system failed, so any metric on it is inflated by construction. The tell is a suspiciously small evaluation set relative to traffic — a 5,000-request eval set drawn from 7,000,000 weekly requests is 0.07% and almost certainly filtered.

**Diagnostic recipe 2 — "the number is too bad, and it changed on a day nothing shipped".** Compute the window's coverage under the closing rule (CI-04), then diff the slot-mix histogram across the boundary day, then diff the population filter. A step change with no deployment is nearly always one of: a backfill, a timezone-aligned job, a traffic mix shift from an unrelated surface, or a logging version bump. The version bump is the highest-yield suspect because it is deployed by a different team and does not appear in the ranking service's deploy log.

**Diagnostic recipe 3 — "relevance improved but engagement did not".** This is the most common false-positive in the domain and it is a measurement question before it is a modelling question. Freeze the layout and replay (Mechanism A); if the lift survives, check the ranker's score distribution for scale changes that altered the *threshold* used by a downstream filter (CI-21); if it still survives, check whether the metric that "did not move" is powered to see an effect of the size claimed (T13 says a 1% relative lift needs 6.5 million requests per arm at this baseline). Failing all three, the improvement is real but was not where users were looking.

**Diagnostic recipe 4 — "the shard-local hypothesis".** Compare three numbers for one term: local document frequency, fleet document frequency, and the served score divided by the score computed with fleet statistics. If the third equals the idf ratio, the score is a pure function of shard statistics and the fix is structural (CI-05). If it does not, something else contributes and the hypothesis is wrong — usually average length rather than frequency, which the same three-number comparison can separate by swapping `avgdl` instead of `idf`.

**Diagnostic recipe 5 — "freshness dispute".** Three clocks, three dashboards, one complaint. Ask for the age of the served document, not the lag of the pipeline (CI-27): the consumer's experience is `now - document.event_time` at render, and this includes cache TTL, CDN version pinning, and any client-side caching. A pipeline meeting a 60-second index-lag SLA with a 3,600-second cache TTL has a one-hour freshness floor and the SLA is technically true and practically false.

**Diagnostic recipe 6 — "the experiment that will not replicate".** If a lift failed to replicate, the candidates in order of base rate are: assignment changed (CI-19, SRM), the metric definition changed between runs (CI-02), the segment mix changed (Mechanism A), the effect was a novelty effect decaying within days (CI-17), and finally a genuinely false positive. The base rates matter: in most organisations the first four explain more non-replications than the fifth, which is why a replication protocol should diff the assignment version and metric version before rerunning.

**Diagnostic recipe 7 — "the delete that will not stay deleted".** Probe with three checks in order: (a) does the item appear in a fresh query against the primary, (b) does it appear against a replica built after the delete, (c) does it appear in the cached response path. (a) positive means the tombstone never applied; (b) positive with (a) negative means retention shorter than the merge horizon (CI-13); (c) positive with both others negative means the cache is the whole story. Each answer has exactly one owner, which is the point of ordering the probes this way.

## 9. Complexity & Cost Analysis

Cost is what decides whether a correct design is affordable, and in this domain the dominant costs are not the ones candidates expect. The four that matter are index storage, index build throughput, per-request scoring work, and analysis sample size.

**Positional index storage.** A corpus of 1,000,000 documents at 120 tokens each holds 120,000,000 postings. Storing a 1-byte position per posting costs 114.4 MiB; adding a 4-byte field id multiplies that by five. The arithmetic is why phrase support is a product decision, not a free flag: `1e6 * 120 * 1 B = 120 MB` is affordable, while storing positions for every field at 4 bytes is 572 MB and pushes merge time up proportionally.

**Delta encoding changes the constant, not the asymptote.** Posting lists store document ids as gaps. At a document frequency of 1,000,000 in a corpus of 1,000,000 (a term in every document), the gap is always 1 and a varint spends 1 byte; at a document frequency of 1,000 against 1,000,000, the mean gap is 1,000 and a varint spends 2 bytes. For 1,000,000 postings that is 1.91 MiB versus 7.63 MiB for fixed 8-byte ids — a 4x difference that only appears at realistic list lengths. The decision rule is to measure list-length distribution, not average list length, because the tail of long lists dominates bytes.

**Rolling-window read amplification.** A window of `W = 300 s` refreshed every `r` seconds reads `ceil(W/r)` counters per evaluation. At `r = 10` that is 30 reads per query; at 1,000 queries per second across all dashboards and admission checks it is 30,000 key reads per second, which is a real capacity line. Coarsening to `r = 30` cuts reads to 10,000 per second but quantises the window edge by 30 seconds and therefore admits bursts of up to 30 s of traffic above the limit. The trade is explicit: reads versus burst tolerance.

**Ranker scoring work.** A request that scores 1,000 candidates with an 800-feature model performs about `7e6 * 1000 * 800 = 5.6e12` feature evaluations per week, which is 5.6 TFLOP-equivalent hours before any tree traversal overhead. Doubling the candidate budget doubles that; halving it costs recall. The honest optimisation is staged scoring — a cheap model on 1,000 candidates, an expensive one on 100 — which changes the exponent rather than the constant.

**Embedding reindex cost.** 20,000,000 items at 768 float32 dimensions is `20e6 * 768 * 4 B = 57.2 GiB` of vectors. At a 200 MB/s write path, rewriting them takes 293 seconds of pure bandwidth, before the model forward pass. That number sets the floor for a full re-embed and is the reason incremental embedding with a version field beats a nightly rebuild.

**Experiment sample size is a cost too.** With a baseline CTR of 0.02355 and 7,000,000 requests per arm, the minimum detectable relative effect at 80% power is 0.96%; reducing traffic to 2,000,000 requests per arm raises it to 1.80%, and 500,000 raises it to 3.61%. Because the detectable effect scales as `1/sqrt(n)`, halving traffic costs `sqrt(2) = 1.414x` in detectable effect, not 2x — which is why the marginal experiment is usually worth running at half traffic rather than not running at all, provided the expected effect exceeds the new floor.

**The cost of a mis-set guardrail.** A guardrail breach discovered after a full ramp costs a ramp-down plus the analysis of a contaminated window. If the ramp delivered 1.9x traffic for 14 days, that is 3.8x the baseline click volume analysed under a possibly-broken model. The corrective action is cheap; the *lost confidence* is not, which is why the pre-declared tolerance (CI-07) is a cost control, not bureaucracy.

```text
COMPLEXITY SUMMARY (dimensions, not constants)
  index storage            O(total tokens * bytes per posting + positions)
  merge time per build     O(total postings * log(shards)) plus a per-byte I/O term
  query candidate retrieval O(sum of postings scanned) truncated by top-k per shard
  global merge             O(shards * k * log(shards * k)) with a heap
  rolling-window sum       O(W / r) reads per evaluation, independent of traffic volume
  IPS evaluation           O(n) time, effective sample size <= n / max weight inflation
  experiment duration      O(1 / effect^2) samples per arm, so 1/sqrt(n) in detectable effect
```

## 10. Key Theorems / Results Reference

Each result below is stated as *what it guarantees*, *what it costs*, and *what it does not cover*. The "does not cover" clause is the part practitioners skip and then get burned by.

**T1 — Inverse propensity scoring is unbiased under support.** For any target policy whose actions have positive probability under the logging policy, the IPS estimator of expected outcome is unbiased. The proof is a one-line tower property: condition on the context, take the expectation of `o * π_t/π_l`, and the ratio cancels. What it does not cover: the variance. Unbiasedness with a heavy-tailed weight distribution can produce an estimate with a confidence interval wide enough to conclude nothing, which is why the effective sample size must be reported alongside the point estimate.

**T2 — Self-normalised IPS is biased but bounded.** Dividing by the realised weight sum instead of `n` keeps the estimate inside the range of observed outcomes, so it can never report a click rate above 1. The bias is `O(1/n)` in the weight dispersion. What it does not cover: comparisons across reports that mix normalisations. Two numbers computed by different normalisations are not comparable even when both are individually sound.

**T3 — Doubly robust estimation is consistent if either nuisance model is correct.** The estimator adds a direct-model prediction to the IPS residual. If the propensity model is right, the residual has zero mean; if the outcome model is right, the correction term vanishes. What it does not cover: the case where both are wrong in the same direction, which is common when the outcome model is trained on the same biased log that produced the propensities.

**T4 — NDCG is bounded by construction only if the denominator is the ideal ordering of the same multiset.** `DCG <= IDCG` holds because the descending-gain ordering maximises a sum of positive weights over a fixed multiset. Break the multiset — add a duplicate, or build IDCG from the served list — and the bound fails, as CI-01 and 4.4 demonstrate with 1.0243. What it does not cover: comparability across `k`. NDCG@10 and NDCG@20 have different denominators and therefore different scales.

**T5 — The DCG discount `1/log2(i+1)` makes the metric front-loaded and comparison-stable.** The discount is convex and decreasing, so improvements at high ranks dominate. This is what makes NDCG robust to tail noise, and equally what makes it insensitive to tail regressions. What it does not cover: the tail. If the business cares about rank 20 to 50, NDCG@10 will not see it, and a separate tail metric is required rather than a larger `k` on the same report.

**T6 — Watermarks bound completeness with a stated probability, not a guarantee.** A watermark that advances past time `t` after observing events with lateness below `L` leaves a fraction `P(lateness > L)` of events unaccounted for until retraction. With exponential lateness of mean 8 s, `P(>45 s) = exp(-45/8) = 0.00361`. What it does not cover: clock skew. A producer whose clock is behind emits events that are late by construction, and no watermark setting fixes a clock.

**T7 — Tumbling windows partition the timeline if and only if boundaries are half-open and consumers agree.** Left-closed right-open gives `sum over all windows == total`, which is the only identity that makes per-window counts auditable. What it does not cover: replays. A replay that re-emits the same event into the same window breaks the identity while every boundary remains correct, so the identity test must be run on non-replaying windows.

**T8 — Rolling-sum boundary error is at most one bucket at each edge.** A window of length `W` built from buckets of size `r` can only represent boundaries at multiples of `r`, so the worst-case error at each edge is `r/2` if events are uniform, and `r` if they are bursty. What it does not cover: bursty traffic. For cron-shaped traffic, a 30-second bucket can hold all of a minute's load, making the boundary error equal to a full bucket rather than half.

**T9 — Hyperloglog's relative standard error is approximately `1.04/sqrt(m)`.** For `m = 4,096` registers that is 1.625%, so an estimate of 900,000 distinct users carries about ±14,625 at one standard deviation. What it does not cover: set difference. Subtracting two sketches does not preserve the error bound, so "new users this week" computed as `A - B` can be much noisier than either input.

**T10 — Inverse document frequency is not scale-invariant under the standard smoothing.** `ln(1 + (N - df + 0.5)/(df + 0.5))` with `df/N` fixed still changes with `N`, because of the `+0.5` terms and the `1 +`. The 100,000-document shard and 1,000,000-document fleet in this sheet differ by 2.534x for the same relative frequency. What it does not cover: the intended benefit of sharding. Sharding is a throughput decision; global statistics must be served from a separate path, at a stated staleness.

**T11 — BM25's saturation is monotone but not linear in term frequency.** The factor `tf(k1+1)/(tf + k1(1-b+b*dl/avgdl))` rises with `tf` and approaches `k1+1` asymptotically, so the third occurrence of a term in a document is worth much less than the first. What it does not cover: staleness in `avgdl`. A stored average that is 10% high under-weights long documents by about 3.3% in score, which is enough to reorder near-ties and therefore enough to look like a model change (CI-23).

**T12 — Chi-square with one degree of freedom at `α = 0.05` has a critical value of 3.8415.** A sample ratio mismatch statistic of 198.02 is 51.5 times that value, and the associated p-value is far below any practical threshold, so the conclusion does not depend on the exact alpha. What it does not cover: clustered assignment. When randomisation is by user but the test counts requests, the effective sample size is smaller and the test over-rejects; the test must run at the randomisation unit.

**T13 — Detectable effect scales as `1/sqrt(n)` per arm.** Going from 7,000,000 to 2,000,000 requests per arm moves the floor from 0.96% to 1.80% relative, a factor of 1.87 against the predicted `sqrt(3.5) = 1.87`. What it does not cover: multiple comparisons. Testing six metrics at `α = 0.05` without correction gives roughly a 26% chance of at least one false positive, so the pre-declared primary metric is not a stylistic choice.

**T14 — Deduplication changes both numerator and denominator unless the key is declared for each use.** Collapsing two rows for one item removes a duplicate discount (numerator) and removes a candidate slot (denominator). What it does not cover: which key is correct. The display key, the metric key, and the inventory key can legitimately differ, and a single shared key will be wrong for at least one of them.

**T15 — Calibration and ordering are independent properties.** A model can rank perfectly and be miscalibrated by a constant factor; the 0.08-prediction-versus-0.05-observation case overprices by 1.60x while ordering correctly. What it does not cover: threshold setting. Any reserve price or eligibility cut is a function of the score's *scale*, so it must be re-derived after every recalibration, not inherited from the previous model.

```text
RESULT SELECTION LADDER — which theorem answers which question
  "is my offline estimate trustworthy?"        T1, T2, T3 (support and variance first)
  "is my relevance metric sane?"               T4, T5, T14 (bounds, cut-off, dedupe key)
  "is my window complete?"                     T6, T7, T8 (watermark, partition, granularity)
  "is my count accurate?"                      T9 (sketch error), T10 (statistic scale)
  "is my change real?"                         T12, T13 (assignment, power)
  "is my auction safe?"                        T15 (calibration before threshold)
  "why does the score move with no model change?" T10, T11, T14 (statistics, saturation, keys)
```

## 11. Interview / Exam Questions

Fourteen questions, mixing recall, computation, diagnosis and judgement. Each has its own answer; none shares boilerplate. Score generously on method and strictly on the stated population.

**Q1 (recall).** What four categories does this sheet use to classify defects, and what does each category predict about where the bug lives?

**Answer.** VIOLATED_ASSUMPTION, NUMERICAL_EDGE, SCALE_FAILURE, CONCEPT_CONFUSION. An assumption violation predicts that the code is doing exactly what it says and the premise is false, so the fix is a design change (add exploration, log propensities, extend retention). A numerical edge predicts a boundary defect — empty denominator, tie, duplicate, truncation — that shows up only on small populations. A scale failure predicts size-dependence: correct on a replay, wrong at production volume. Concept confusion predicts that two teams are computing different quantities with one name, so the fix is a written definition, not code.

**Reasoning.** The categories are chosen by *what evidence would exonerate the component*. If an assumption is violated, no amount of code review finds it; you need a controlled comparison. If it is an edge case, a single assertion finds it. If it is scale, you need a production-size replay. If it is a concept, you need a second person in the room.

**Follow-up.** Which category would you check last and why? Scale, because a production-size replay is the most expensive diagnostic, and its base rate is lowest — most incidents resolve before it.

**Red flag.** Answering "bias, variance, leakage, overfitting", which are model-quality categories rather than system-defect categories and lead to the wrong investigation order.

**Scoring.** Full credit requires all four names plus the "what evidence exonerates" logic; partial credit for names only.

**Q2 (computation).** A team reports CTR of 18.84% per request and 2.355% per impression over 7,000,000 requests. Are these consistent, and what does the ratio tell you?

**Answer.** Yes: `1,318,800 / 7,000,000 = 0.188400` and `1,318,800 / 56,000,000 = 0.0235500`, and the ratio `0.1884 / 0.02355 = 8.000`, exactly the impression count per request. The ratio being an integer that matches the slot count is the signature of a denominator difference, not a data difference. The consequence is that the 18.84% has a ceiling of 8 and can never be compared to a benchmark quoted per impression.

**Reasoning.** Any two rates over the same events differ by the ratio of their denominators. When that ratio equals a structural constant (slots, sessions per user, hours per day), the discrepancy is definitional and the two reports are both correct.

**Follow-up.** If the ratio had been 7.94 instead of 8.000, what would that indicate? That some requests filled fewer than eight slots — a fill-rate defect worth its own investigation, not a rounding artefact.

**Red flag.** Averaging the two rates or "taking the higher one as the truth".

**Scoring.** Correct arithmetic plus the recognition that an integer ratio implies a denominator bug.

**Q3 (computation).** NDCG@10 comes back as 1.0243. Give the most likely cause, the arithmetic that confirms it, and the one-line assertion that would have caught it.

**Answer.** Duplicate counting. With served grades `[3,2,3,1,2,3,3,0,2,1]` and IDCG 9.6901 from the label multiset, DCG is 9.9260 and the ratio is 1.0243; the honest list scores 9.2593. The difference is exactly 0.6667, which is the rank-7 term rising from `1/0.3333` to `3/0.3333`... precisely, from 0.333 to 1.000 because a grade-3 item occupies rank 7 instead of a grade-1 item. The assertion is `assert 0.0 <= ndcg <= 1.0` on every eval request.

**Reasoning.** NDCG is a ratio of two sums over the same multiset, and the descending ordering maximises the sum, so any value above 1 proves the numerator used a multiset the denominator did not.

**Follow-up.** Why is a duplicate at rank 7 worth 0.667 when a duplicate at rank 2 is worth 1.893? Because the discount at rank 7 is 0.333 while at rank 2 it is 0.631; the increment is the *difference* between the duplicate's grade contribution and the displaced item's.

**Red flag.** Blaming the IDCG calculation alone without noticing that the numerator changed, or "clipping to 1.0" as a fix.

**Scoring.** Requires the numeric confirmation and the assertion; clipping is an automatic downgrade because it hides the defect.

**Q4 (diagnosis).** A feed's CTR is 25.08% lower than the intrinsic rate implied by a uniform slot mix. Nothing was deployed. What is your first check?

**Answer.** The slot-mix distribution across the two comparison windows. With elasticities `[1.00,0.86,0.72,0.60,0.50,0.41,0.34,0.28]`, the uniform-mix mean is 0.71375 and the observed mix `[0.30,0.20,0.15,0.12,0.10,0.06,0.05,0.02]` gives 0.7492; the ratio 0.71375/0.7492 explains a 4.7% difference, so a 25% gap needs a much larger mix change or a second mechanism. That arithmetic immediately says: mix alone does not explain it, keep looking.

**Reasoning.** Doing the arithmetic *before* the investigation is what makes this a diagnostic rather than a guess. If the candidate mechanism's maximum plausible effect is smaller than the observed effect, the mechanism is not the cause and the time is not worth spending.

**Follow-up.** What is the second mechanism to test? Impression logging completeness — a 25% shortfall in impressions inflates CTR rather than deflating it, so check whether the click log lost events instead.

**Red flag.** "Check the model" as a first move, or accepting "traffic mix" as an explanation without quantifying it.

**Scoring.** Full credit for computing the mix's maximum effect before investigating.

**Q5 (computation).** An experiment has 7,000,000 requests per arm at a 2.355% baseline CTR. What relative effect can you detect at 80% power, and what happens at 2,000,000 per arm?

**Answer.** Standard error per arm is `sqrt(2 * 0.02355 * 0.97645 / 7,000,000) = 8.106e-5`; the minimum detectable effect at 80% power is `2.802 * 8.106e-5 = 2.271e-4` absolute, or 0.964% relative. At 2,000,000 per arm the standard error is `sqrt(3.5) = 1.871x` larger, giving 1.80% relative.

**Reasoning.** The whole calculation is one line: `MDE ∝ sqrt(2p(1-p)/n)` with a multiplier of about 2.8 for 80% power at a two-sided 5% level. The multiplier is `1.96 + 0.842 = 2.802`.

**Follow-up.** A product owner asks for a 0.5% relative effect. How many requests per arm? `n = 2 * 2.802^2 * p(1-p) / (0.005p)^2 ≈ 26,000,000`, which is 3.7x current weekly traffic per arm.

**Red flag.** Quoting an absolute MDE without the baseline, which makes the number meaningless — 2.27e-4 is 0.96% relative here but 0.02% relative at a 100% baseline.

**Scoring.** Correct multiplier, correct conversion between absolute and relative, and the recognition that the floor scales as `1/sqrt(n)`.

**Q6 (diagnosis).** Two arms have 7,000,000 and 7,070,000 assignments against a 50/50 design. Is this a problem?

**Answer.** Yes. Expected is 7,035,000 per arm; `chi2 = 2 * (35,000^2 / 7,035,000) = 348.26` against a critical value of 3.8415 at `α = 0.05`, so the imbalance is decisive (about 91 times the critical value). A 1% imbalance at this volume is not noise; the smallest imbalance that reaches significance at 14,000,000 total assignments is about 3,667 assignments, which is 0.052% of an arm. The comparison is void until assignment is fixed.

**Reasoning.** Chi-square grows linearly in `n` for a fixed relative imbalance, so at large `n` even a trivial imbalance is significant. That is not a flaw in the test: a 1% assignment imbalance at this scale indicates a systematic filter, and any systematic filter can be correlated with the outcome.

**Follow-up.** What if the design intentionally assigns by user and each user has many requests? Then run the test on users, not requests; the request-level test will over-reject because requests within a user are correlated.

**Red flag.** "It's only 1%, that's within tolerance", or running the SRM test after seeing the outcome metric.

**Scoring.** Requires the chi-square value, the comparison to 3.8415, and the conclusion that outcomes cannot be read.

**Q7 (computation).** A 300-second rolling window refreshes every 30 seconds and closes at a watermark 45 seconds behind. What fraction of the window is missing, and how does that compare to a 0.361% lateness tail?

**Answer.** `(300 - 45)/300 = 0.8500`, so 15% of the window's event-time coverage is discarded — 1,350 of 9,000 event-seconds at 30 events/s. The lateness tail is `exp(-45/8) = 0.00361`, i.e. 0.361%. The two differ by a factor of 41, which proves the shortfall is the closing rule, not late data.

**Reasoning.** Closing at processing-time `T` means the window contains only events with event time below `T - L`, so it spans `W - L` seconds rather than `W`. The lateness distribution governs how many events *arrive* after the close; the closing rule governs how much *time* the window covers. Confusing them is the standard error.

**Follow-up.** If you close at `L = 5 s` instead, coverage rises to `295/300 = 98.33%`. What is the cost? A watermark that advances only 5 seconds behind real time will stall on any producer more than 5 seconds late, so windows emit late and the downstream join must tolerate retraction.

**Red flag.** Adjusting the watermark without computing coverage, or attributing the gap to "late events".

**Scoring.** Both numbers and the factor-of-41 comparison.

**Q8 (diagnosis).** A shard of 100,000 documents scores a term 2.534x higher than the fleet does for the same relative frequency. Explain, and say what the correct architecture is.

**Answer.** The shard computes `ln(1 + (100,000 - 50 + 0.5)/(50 + 0.5)) = 7.5910`; the fleet computes `ln(1 + (1,000,000 - 50,000 + 0.5)/(50,000 + 0.5)) = 2.9957`. The ratio is 2.534. Inverse document frequency is not scale-invariant under the standard smoothing because of the `+0.5` and `1 +` terms, so the same relative frequency yields a different absolute value at a different corpus size. The correct architecture separates the two uses: term frequency from the local postings (which the shard legitimately has), and idf from a fleet-level statistics snapshot with a version id, refreshed on a schedule and stamped onto every score.

**Reasoning.** The defect is not variance — averaging shard idfs would not fix it, because each shard's value is biased the same direction. It is a deterministic scale dependence, which is why the fix must be a global statistic rather than a bigger sample.

**Follow-up.** What breaks if you serve a per-language fleet statistic instead of one global number? Nothing, and it is usually better, because a term's rarity genuinely differs by language; the requirement is that the statistic be computed on the corpus that actually competes for the same queries.

**Red flag.** "Increase shard size" — the bias shrinks but the architecture remains wrong, and shard size is bounded by merge throughput.

**Scoring.** Correct two idf values, the ratio, and the two-level statistics fix.

**Q9 (judgement).** You are asked to choose between a snapshot counter, a rolling bucket sum, and a HyperLoglog sketch for a per-user daily impression cap. Which, and why?

**Answer.** The rolling bucket sum over the user's local day, or a snapshot if a bounded overshoot is acceptable — not the sketch. A cap is a *per-user* exact count; Hyperloglog estimates *distinct* counts and carries a 1.625% relative standard error, so at a 100,000-user scale about 1,625 users would breach the cap in either direction. The snapshot introduces at most one refresh interval of overshoot, which is a bounded, explainable error; the sketch introduces an unbounded per-user sign error that cannot be explained to an advertiser.

**Reasoning.** Choose the mechanism whose error character matches the consequence. For a cap, the consequence is a contractual over-delivery, so the acceptable error is "bounded and one-directional"; a sketch's error is neither.

**Follow-up.** Where would the sketch be right? Reach, unique-visitor, and distinct-query counts, where the total is large enough that 1.6% is a small absolute error and exact counting is prohibitively expensive.

**Red flag.** Picking the sketch for its memory advantage without computing the error at the decision's scale.

**Scoring.** Requires naming the error magnitude and matching error character to consequence.

**Q10 (computation).** A ranker scores 1,000 candidates per request with an 800-feature model at 7,000,000 requests per week. What is the weekly scoring work, and what does a staged model buy you?

**Answer.** `7e6 * 1000 * 800 = 5.6e12` feature evaluations per week. Staging to 1,000 candidates with a cheap model and 100 with the expensive one gives `7e6 * (1000 * 100 + 100 * 800) = 7e6 * 180,000 = 1.26e12`, a 4.4x reduction. If the cheap model uses 100 features instead of 100, the arithmetic is `1000*100 = 100,000` plus `100*800 = 80,000`, total 180,000 per request versus 800,000 unstaged.

**Reasoning.** The cost is the product of candidates, features and requests. Staging changes the *structure* of the product rather than any constant, which is why it beats micro-optimisation: a 10% faster kernel saves 10%, while staging saves 56%.

**Follow-up.** What does staging cost in quality? Recall is preserved if the cheap model's top-100 contains the expensive model's top-100 most of the time; measure that overlap on a holdout, because it is the quantity that decides whether staging is free.

**Red flag.** Quoting a per-request latency improvement without the candidate budget, since latency and candidate count are not proportional in a staged system.

**Scoring.** Correct product, correct staged product, and the observation that structure beats constants.

**Q11 (diagnosis).** Users see week-old content while the freshness dashboard is green at p99 = 42 seconds. Give three mechanisms and the check that separates them.

**Answer.** (a) The SLA measures index lag, and the served document is genuinely old — a crawl that stopped, a feed that stopped publishing. (b) A cache or CDN layer with a TTL of 3,600 seconds, which is 60x the index lag and never appears in an indexer metric. (c) A pinned CDN or API version serving an older index snapshot to a subset of traffic. The separating check is the joint distribution of document event age and cache age for served items on one day: a horizontal band at high document age is (a), a band at cache age near the TTL is (b), and a per-version split is (c).

**Reasoning.** Three different clocks feed one user-visible number, and each has a different owner. The dashboard can only be green because it measures the clock nobody complained about.

**Follow-up.** What should the SLA be redefined as? p99 of `render_time - document.event_time` over served impressions, split by cache tier, because that is the quantity the user experiences.

**Red flag.** Adding a cache-bypass flag and declaring the problem solved, which treats (b) while leaving (a) and (c) undetected.

**Scoring.** All three mechanisms plus a check that can distinguish them.

**Q12 (computation).** A dedupe change collapses 25% of rows on an eight-slot page. What happens to page fill and to any metric computed with a fixed slot denominator?

**Answer.** Page fill drops from 8 to `8 * 0.75 = 6` unless the candidate budget is increased to compensate — fetching `8/0.75 = 10.67`, so 11 candidates, restores fill. A metric computed as `clicks / (requests * 8)` now divides by 8 slots of exposure that no longer exists, so it understates CTR by 25%; recomputed against the actual 6 impressions per request, the rate is unchanged. The defect is the denominator, and the fix is either more candidates or a denominator that counts real impressions.

**Reasoning.** Dedupe removes both a candidate and an exposure. Any metric whose denominator is a *nominal* slot count rather than a *logged* impression count will misreport the change.

**Follow-up.** Which is the correct denominator for a business review? Logged impressions, because they are the exposures that could have produced a click; nominal slots are a capacity statement, not an exposure statement.

**Red flag.** Reporting the CTR drop as a relevance regression.

**Scoring.** Both the fill arithmetic and the denominator diagnosis.

**Q13 (judgement).** An offline evaluation shows +6% NDCG but the online experiment is flat at 1.80% detectable effect. What do you conclude, and what do you do next?

**Answer.** Conclude that the experiment cannot see a 6% effect only if 6% were below the floor — it is not, so a flat result is informative and suggests the offline gain did not transfer. The candidates, in order: propensity support (the offline evaluation rewarded actions the logging policy under-sampled), population mismatch (offline labels from one segment, online traffic from another), the position-effect product (the offline metric ignores layout while the online measurement multiplies by attention), and finally the possibility that NDCG gains landed at ranks the user never sees. Next step is a support-coverage report and a layout-frozen replay, both cheap, before any model work.

**Reasoning.** A powered null is evidence, not absence of evidence. The correct response is to test the *transfer assumption*, not to rerun with more traffic.

**Follow-up.** If the support-coverage report shows 12% of target actions have zero logged propensity, what does that do to the offline claim? It invalidates it for those actions, which is enough to explain the gap if they are the ones the model improved.

**Red flag.** "The experiment was underpowered" without checking the floor against the claimed effect.

**Scoring.** The power arithmetic plus at least two transfer mechanisms.

**Q14 (diagnosis).** A model retrains, offline AUC improves, online engagement drops. Give the mechanism, the check, and the guardrail you would have wanted.

**Answer.** Feedback-loop amplification: the training window contains only exposures the previous policy chose, so head items dominate the training distribution. The retrained model sharpens that bias, ranks head items more confidently, and reduces the diversity that produced engagement. The check is a comparison of exposure share by item decile between the training window and current serving traffic; a gap of more than 20 points in head share is enough to explain the drop. The guardrail is coverage or distinct-item impressions per session, with a pre-declared tolerance, plus a shadow-scoring period in which the candidate ranks but does not serve.

**Reasoning.** AUC measures ordering on the logged distribution, which is the distribution the model will reinforce. Improving fit to a biased distribution is not the same as improving the served outcome, and the two diverge fastest when exploration is absent.

**Follow-up.** What is the cheapest structural fix? An exploration floor — for example, reserving 2% of impressions for uniform random candidates — which both widens the training distribution and gives you an unbiased support check for IPS later.

**Red flag.** "Retrain with more data", which amplifies the same bias at greater cost.

**Scoring.** Requires the exposure-distribution check and a concrete guardrail with a tolerance.

**Q15 (computation, cross-check).** A team proposes storing positions for every token in every field. Estimate the cost for 1,000,000 documents at 120 tokens each, and say when the cost is justified.

**Answer.** `1,000,000 * 120 = 120,000,000` positions. At 1 byte each that is 114.4 MiB; adding a 4-byte field identifier to support per-field phrase matching makes it 572 MiB. The cost is justified when the product requires phrase matching or proximity ranking on those fields, because the alternative — verifying phrases by re-scanning stored text — costs much more per query at high query rates. It is not justified for fields that only need Boolean containment, which is the majority of fields in most schemas.

**Reasoning.** Position storage is a per-token multiplicative cost, so it scales with corpus size and token count rather than with vocabulary. Estimating it as "a bit more index" is how a 114 MiB decision becomes a 572 MiB one.

**Follow-up.** What is the cheapest middle ground? Store positions only for the fields that appear in phrase clauses, and verify remaining phrases against a stored-text forward index at query time with a cap on the number of candidates.

**Red flag.** Approving position storage without naming which query classes need it.

**Scoring.** Correct byte arithmetic in both variants and a condition under which the cost is justified.

