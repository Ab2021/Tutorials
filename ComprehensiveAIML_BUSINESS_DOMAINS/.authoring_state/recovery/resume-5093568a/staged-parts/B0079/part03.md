## 5. Worked Numerical Example

**Status: ILLUSTRATIVE.** Every quantity below is constructed to exercise the formulas, and every ratio has been recomputed independently. The arithmetic is verified; the scenario is not a measurement of any product.

**Setting.** A subscription video surface in one territory. The global catalog holds 12,000,000 items with 128-dimensional float32 embeddings. A request is one home-screen render for one viewer at one timestamp, and the candidate budget is k = 20 because the ranker and the slate both take 20. Relevance means "watched at least 60 seconds within 7 days of an impression", so the label horizon is 7 days and the cohort below is drawn from requests whose windows have matured.

### 5.1 The eligible set

Hard predicates applied for this request: territory rights, subscription entitlement, device capability, availability of a playable asset, and safety policy.

```text
catalog with embeddings            I        = 12,000,000
passes territory + entitlement + device + asset + safety
                                   E(q,t)   =    240,000
eligibility density                p = 240,000 / 12,000,000 = 0.0200
labelled relevant inside E         R(q,H)   =          7
```

The density of 0.02 is the single most consequential input in the whole calculation, and it is the one teams most often replace with a global average. Territory-level density here is 2%; a global average across all territories would be higher because large catalogs serve large territories, so planning from the global number under-provisions depth for small ones.

### 5.2 Depth requirement before filtering

Using [D1] with k = 20, p = 0.02 and tolerance alpha = 0.05, the smallest depth whose binomial tail reaches 0.95 is m = 1390. Independent re-check by a different route: the distribution has mean m·p = 27.8 and standard deviation sqrt(m·p·(1−p)) = sqrt(1390 × 0.02 × 0.98) = sqrt(27.244) = 5.220. Twenty is (27.8 − 20)/5.220 = 1.494 standard deviations below the mean. For a normal approximation the lower tail at 1.494 is about 0.068; the exact binomial tail is 0.0498 because the distribution is slightly right-skewed. Both routes say "just under a 5% shortfall", which is the intended condition; the exact value governs.

```text
mean kept at m=1390      = 1390 * 0.02            = 27.8
sd                       = sqrt(1390*0.02*0.98)   = 5.220
normalised margin        = (27.8 - 20)/5.220      = 1.494
exact tail P(X < 20)     = 1 - 0.9502             = 0.0498
```

### 5.3 Three pipelines on the same request

| Pipeline | Mechanism | Candidates kept | Relevant in candidates | Candidate recall | Underfill |
|---|---|---:|---:|---:|---|
| A — popularity + co-visit | Two pre-computed lists intersected with E, capped at 20 | 20 | 2 | 2/7 = 0.2857 | 0 |
| B — filter-aware ANN | Depth 1390, eligibility filter during traversal, cap 20 | 20 | 4 | 4/7 = 0.5714 | 0 |
| C — global ANN then filter | Depth 20 on the full index, filter afterwards | 3 | 1 | 1/7 = 0.1429 | 17 of 20 |

Pipeline C is the interesting one. Its configured budget is 20 and its delivered candidate set is 3. The ranker receives three items and the slate has twenty slots, so seventeen slots must either be filled by a fallback or left empty. Recall of 0.1429 is not a ranking failure; it is arithmetic. With p = 0.02 and depth 20, the expected number of eligible survivors is 20 × 0.02 = 0.4, and P(at least one survivor) = 1 − 0.98²⁰ = 1 − 0.6676 = 0.3324. That is, in two of every three requests, the global-then-filter pipeline returns *nothing* eligible at all.

```text
P(no eligible survivor of 20) = 0.98^20 = 0.6676
P(>=1 eligible survivor)      = 0.3324
expected survivors            = 20 * 0.02 = 0.4
```

Pipeline B's realised keep of 24 items (before capping at 20) is a draw of the m = 1390, p = 0.02 binomial, mean 27.8, and lies within one standard deviation of the mean.

### 5.4 Cohort comparison

The same request is not enough to choose a pipeline, so the evaluation runs over 400 sampled requests with matured labels.

```text
requests in cohort                         400
total eligible labelled pairs             3,120      (mean 7.8 per request)
pipeline A recovered pairs                1,200      pooled recall 1200/3120 = 0.3846
pipeline B recovered pairs                1,500      pooled recall 1500/3120 = 0.4808
absolute difference                                        0.0962 = 9.62 pp
relative improvement                    0.0962/0.3846  = 0.2501 = 25.0%
```

Wilson intervals at 95% on the per-request "recovered at least one relevant item" indicator, computed on 400 requests:

```text
A: 120/400 = 0.3000   interval [0.2572, 0.3466]
B: 180/400 = 0.4500   interval [0.4020, 0.4990]
```

The intervals do not overlap, which is consistent with a real difference on that indicator. For the pooled pair-level difference the standard error is sqrt(0.4808×0.5192/3120 + 0.3846×0.6154/3120) = 0.0125, giving z = 0.0962/0.0125 = 7.70. That z is optimistic because pairs within a request are correlated; with a design effect of 2 the effective sample halves and z falls to about 5.4, still large. The honest statement is "the difference is large relative to sampling noise under an independence assumption", not "p < 0.001 in production".

### 5.5 Independent re-check of the headline comparison

Recompute both pooled ratios from the raw counts rather than accepting the percentages.

```text
A: 1,200 / 3,120
   3120 * 0.38 = 1185.6   remainder 14.4 / 3120 = 0.00462
   => 0.38462, matches 0.3846

B: 1,500 / 3,120
   3120 * 0.48 = 1497.6   remainder 2.4 / 3120 = 0.00077
   => 0.48077, matches 0.4808

difference: 0.48077 - 0.38462 = 0.09615
```

A second, independent check reads the table the other way: if B recovers 300 more pairs out of 3,120, that is 300/3120 = 0.09615, identical. Both routes agree, which is the minimum bar for trusting a comparison table in a review.

### 5.6 The costs that paid for the gain

```text
latency budget, p99 target                            250 ms
authorization / entitlement lookup                      6 ms
ANN traversal at depth 1390                            38 ms
eligibility filter, union and dedup                     9 ms
deduplication hash pass                                 3 ms
ranker over 20 candidates                              52 ms
policy and diversity adjustment                        11 ms
serialization                                           7 ms
network to edge                                        24 ms
--------------------------------------------------------------
total                                                 150 ms
headroom                                             100 ms  (40% of budget)
```

Embedding memory, which is the recurring cost rather than the per-request cost:

```text
float32: 12,000,000 * 128 * 4 bytes = 6,144,000,000 B = 6.144 GB per replica
two replicas: 12.288 GB
graph links at 64 neighbours, 4 B each: 12e6*64*4 = 3,072,000,000 B = 3.072 GB per replica
per-replica total: 6.144 + 3.072 = 9.216 GB;  two replicas: 18.432 GB
int8 quantized vectors: 12e6*128*1 = 1,536,000,000 B = 1.536 GB per replica
```

Quantising to int8 cuts the vector payload by 4× (6.144 GB to 1.536 GB) but changes the score ordering, so it is a change of system, not a storage optimisation: it must be re-measured with [A1] fidelity against the float32 exact list before it is treated as equivalent.

### 5.7 Sensitivity

Four inputs dominate the conclusion, and each is perturbed independently.

```text
1. eligibility density falls to 0.01 (smaller territory):
   required depth 20 -> 2784 (ratio 139.2); ANN time 38 ms -> approx 76 ms
   new total 188 ms; headroom 62 ms (25% of budget)

2. k rises from 20 to 50 at p = 0.02:
   required depth 3102 (from the k=50 p=0.02 row: ratio 62.0 at k=50 p=0.02)
   ranker cost grows with candidates, not with k alone only if capped

3. label horizon cut from 7 days to 1 day:
   R shrinks; per-request relevant counts fall; recall rises mechanically
   because the denominator fell, not because retrieval improved

4. k >= |E| (small test catalog):
   recall saturates at 1.0; the metric stops discriminating
```

The fourth input is not hypothetical: a fixture with a 30-item catalog and k = 20 is close to saturation, and any A/B run on such a fixture reports a difference of zero or one and calls it a tie.

**Takeaway.** The worked example's real content is the ratio 1390/20 = 69.5: to reliably deliver 20 eligible candidates at 2% density you must look at about seventy times more items than you keep, and that multiplier is set by eligibility, not by the model.

## 6. Visual / Geometric Intuition

**Set nesting.** The three sets nest strictly, and the nesting order is what makes the metrics well-defined.

```text
+-----------------------------------------------------------+
| catalog I(t)                                   12,000,000 |
|  +-------------------------------------------------------+|
|  | eligible E(q,t)                            240,000    ||
|  |   +---------------------------------------------------+|
|  |   | labelled relevant R(q,H)                     7     ||
|  |   +---------------------------------------------------+|
|  +-------------------------------------------------------+|
+-----------------------------------------------------------+

recall denominator lives here: |R ∩ E| = 7
recall numerator lives here:   |C_k ∩ R| <= 7
```

Caption: the outer ring is everything the platform knows about; the middle ring is everything a policy allows this viewer to see now; the inner ring is everything with a matured label. A metric that divides by the outer ring is unattainable by construction.

**The truncation geometry of filtering.** Score is roughly monotone in distance from the query, so the index returns a prefix. Eligibility is a separate predicate that is not monotone in score. Filtering the prefix therefore keeps an unpredictable subset.

```text
rank:      1    2    3    4    5    6    7    8 ...  1390
eligible:  E    .    E    .    .    E    .    E        .
                      |
depth 20:  [. . . . . . . . . . . .]  -> 0 or 1 survivor, usually 0
depth 1390:[. . . . . . . . . . . . . . . . . . . .] -> ~28 survivors

E = passes filter, . = fails filter
```

Caption: increasing depth does not improve the precision of the filter; it increases the number of draws. The expected survivor count is depth × density, and the requirement is a tail probability, not an expectation.

**The candidate-to-display funnel as a multiplicative chain.**

```text
eligible relevant in E       7        (of 240,000 eligible)
retrieved into candidates    4        candidate recall 4/7  = 0.5714
survives ranking to slate    4        ranker retention      = 1.0000 for this request
displayed and acted on       3        label rate            = 0.75
```

Caption: the product 0.5714 × 1.0 × 0.75 = 0.4286 is the end-to-end relevant-display rate for this request. Attenuation anywhere in the chain is bounded by the first factor; a ranker cannot lift the product above 0.5714.

**Position and score are correlated, so raw CTR is not preference.**

```text
slot:        1     2     3     4
observed CTR: 0.080 0.052 0.041 0.033
examination:  1.00  0.62  0.48  0.40
corrected:    0.080 0.084 0.085 0.083

observed spread across slots: 0.080 -> 0.033   (2.4x)
corrected spread:              0.080 -> 0.083   (1.04x)
```

Caption: dividing observed click-through by an examination propensity collapses a 2.4× apparent slot effect into roughly nothing. Constructed numbers; the point is the direction and magnitude of the correction, not the specific values. If the corrected series had *increased* with slot, that would suggest examination propensities were estimated too aggressively.

**Latency as a fixed-width container.** Each stage consumes a slice of the 250 ms p99 budget. Depth is the only slice that scales with a policy parameter, which is why it is the first place to look when latency regresses.

```text
|auth 6| ANN 38 |filter 9|dedup 3| rank 52 |policy 11|ser 7| net 24 | headroom 100 |
0                                                                            250 ms
```

Caption: at 150 ms of committed work, a 60% depth increase fits inside headroom; a 100% increase does not, which is why the density-falls-to-0.01 case in 5.7 consumes most of the remaining slack.

**Diminishing returns of additional sources.**

```text
sources:   1      2      3      4      5
coverage: 0.250  0.438  0.578  0.684  0.763     (p=0.25, independent)

marginal:   --    +0.188 +0.141 +0.105 +0.079
```

Caption: each added source contributes less than the previous one under independence, and much less in practice because popularity and co-visit sources overlap heavily on head items. The series is the geometric decay of (1 − p)^s, not a tuning artefact.

### 6.1 Reading the funnel as a table of losses

The geometric pictures above compress into a single loss accounting that survives a review meeting, because each row is attributable to one owner and one change.

| Stage | Entering | Leaving | Loss | Loss attributable to | Recoverable by |
|---|---:|---:|---:|---|---|
| Catalog at ingest | 12,000,000 | 12,000,000 | 0 | — | — |
| Passes rights | 12,000,000 | 10,560,000 | 12.0% | deal book and territory map | territory partitioning |
| Passes entitlement | 10,560,000 | 8,659,200 | 18.0% of remainder | subscription tier mix | tier-partitioned indexes |
| Passes availability | 8,659,200 | 8,226,240 | 5.0% of remainder | asset pipeline lag | earlier asset readiness |
| Passes device and safety | 8,226,240 | 8,226,240 | 0% in this cohort | device mix | nothing needed |
| **Eligible set** | 12,000,000 | **240,000** | **98.0%** | the five predicates | not recoverable, by design |
| Entered candidate set | 240,000 | 1,390 looked at | — | depth provisioning | more depth, more latency |
| Kept after cap | 1,390 | 20 | 98.6% of looked-at | the budget k | larger k, more ranker cost |
| Relevant in candidates | 7 labelled | 4 | 3 lost | retrieval quality | better retrieval |
| Reach slate | 4 | 4 | 0 | ranker retention near 1.0 | nothing needed |

The table makes an argument that prose obscures: from 12 million items to 20 candidates, the largest single reduction is eligibility, and it is *correct*. The reduction that is a defect is the third-from-bottom row, where three of seven relevant items never entered the candidate set. Reviewers who treat the 98% eligibility loss as the problem and the 3-of-7 retrieval loss as normal have the priorities exactly inverted.

**The arithmetic of that inversion, stated once.** Improving retrieval from 4 of 7 to 6 of 7 raises candidate recall from 0.5714 to 0.8571, a gain of 28.6 percentage points, at a cost of depth and latency. Relaxing an eligibility predicate to admit 1% more of the catalog would raise the eligible pool by 2,400 items and might raise recall by less than one item, while creating a compliance exposure. Ratio of value to risk is not close.

### 6.2 The same funnel for a marketplace, where availability dominates

The video case above is rights-dominated. A marketplace candidate pipeline inverts two rows, and the inversion changes which infrastructure investment pays.

| Stage | Video surface | Marketplace surface | Why the difference matters |
|---|---|---|---|
| Dominant rejection | rights (territory, window) | availability (stock, zone) | rights change slowly; stock changes by the second |
| Refresh cadence that suffices | hourly policy sync plus request-time check | near-real-time stock feed plus request-time check | a stale stock view sells what cannot ship |
| Partitionable dimension | territory | fulfilment zone | zones are numerous, so partitioning multiplies storage more |
| Typical eligible density | 2% of a 12M catalog | 8% of a 2M catalog in-zone | higher density means shallower depth is sufficient |
| Depth implied at k=20, alpha=0.05 | 1390 | 344 at p=0.08 | 4.0x cheaper traversal |
| Failure that reaches the user | unplayable title | order cancelled after payment | the marketplace failure is more expensive per event |

The depth figure for the marketplace case follows from the same formula, not from a different one: at p = 0.08, k = 20, the required depth is between the p = 0.05 value of 554 and the p = 0.10 value of 275; direct evaluation gives 344 for a 95% fill probability, and 1390/344 = 4.04. The lesson is not the number but the method: density is measured per surface, then substituted.

**Where the marketplace case breaks the model.** Stock is not independent of score. A model trained on purchase history ranks popular items highly, and popular items are precisely the ones that sell out, so their eligibility is negatively correlated with their rank. The binomial depth model assumes independence and therefore under-predicts underfill exactly when it matters most. The correction is empirical: measure the realised kept-count distribution at the chosen depth and compare it with the binomial prediction. If the realised fifth percentile is materially below k while the model says 95%, correlation is present and depth must be raised until the measured tail, not the modelled tail, satisfies the tolerance.

**Takeaway.** The pictures encode three facts worth memorising: metrics must divide by the middle ring, depth requirements grow like k/p, and every added stage both costs latency and attenuates the fraction of relevant items that survives to display.

## 7. Comparison / Selection Table

Selection is by pipeline, not by model, because the failure modes live in the composition. Each row states the precondition that must hold before the method is admissible.

| Pipeline | Preconditions | What it guarantees | Cost driver | Choose when | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Pre-filtered popularity | Point-in-time activity counts and a cheap eligibility filter over a small list | Nothing about personalisation; coverage of head items only | One small lookup plus filter | Cold start, degraded mode, guardrail baseline | Catalog is long-tailed or cohort sensitive | HEURISTIC |
| Co-visitation | Interaction matrix with stable item identity and enough density | Learns co-occurrence, not causation | Offline job plus adjacency lookup | Repeat-interest surfaces with dense behaviour | New items, sparse users, or item churn is high | HEURISTIC |
| Content similarity | Authoritative item metadata present at ingest | Can surface new items with no interactions | Feature store plus similarity scoring | New-item cold start and metadata-rich catalogs | Metadata is sparse, stale, or itself sensitive | HEURISTIC |
| Exact dense scan | Vectors and score fixed; eligible set small or compute ample | Exact top-k under the defined score | O(|E|·d) per request | Oracle construction and small eligible sets | |E|·d exceeds the latency budget at target QPS | DERIVED |
| Global ANN then filter | Index over the whole catalog | Approximate top-k of the *global* index | Fixed depth; cost independent of eligibility | Never for a narrow-segment surface | Eligibility density is low or correlated with score | DERIVED |
| Filter-aware ANN at depth m | Depth provisioned from k, p, and alpha; filter inside traversal | Approximate top-k of the *eligible* set with bounded underfill | Depth, hence roughly k/p | Density is low and underfill is unacceptable | Depth from k/p breaks the latency budget | DERIVED |
| Filtered brute force on a per-segment index | Segments align with the dominant eligibility predicate | Exact eligible top-k within the segment | Segment index build and storage multiplication | One predicate dominates and segments are few | Segments proliferate per viewer or per entitlement tier | DERIVED |
| Two-stage rank then re-filter | Ranker cheap enough to score more than k | Ranked eligible set, but only up to ranker recall | Ranker cost times candidate count | Eligibility changes between stage and display | Final filter runs after truncation with no fallback | HEURISTIC |

**Selection procedure, in order.**

1. **Fail closed on authorization first.** Whatever the retrieval method, the last check before display must consult current authoritative policy. Depth tuning cannot substitute, because a revocation can land between the index refresh and the request.
2. **Measure eligibility density per segment, not globally.** Density is the input that sets depth. A single global number will mis-provision every small territory and every narrow entitlement tier.
3. **Compute the depth requirement from [D1] and compare against the latency envelope.** If m from k/p does not fit, the honest options are to reduce k, pre-partition the index by the dominant predicate, or accept a stated underfill rate. The dishonest option is to keep depth 20 and hope.
4. **Choose the retrieval family for the cohort with the least data.** New items need content features; new viewers need non-personalised or contextual retrieval. A pipeline whose only source is co-visitation fails both.
5. **Fix the evaluation cohort before running the comparison.** Same requests, same labels, same horizon, same cutoff, same eligibility snapshot.
6. **Report recall, underfill, latency, and exposure concentration together.** Any one alone permits a harmful trade.

**Why a pre-partitioned exact index is often better than deeper ANN.** If one predicate explains most of the rejection — territory is the classic case — then holding one exact index per large territory removes that rejection entirely, and the remaining density within a territory partition is much higher. The cost is storage multiplication and index build time; the benefit is that exactness replaces a tail-probability argument. The decision is a multiplication factor on storage versus a depth multiplier on latency, and both are computable before you build anything.

**Why the global-ANN row says "never".** It is listed to be explicitly rejected, because it is the default in most tutorials and the most common cause of a silent quality ceiling. It is defensible only when eligibility density is high and uncorrelated with score, which in practice describes a catalog with no rights or inventory constraints at all.

**Benchmark status: NOT MEASURED.** Any comparison of these pipelines must fix catalog snapshot, request cohort, embedding version, index build parameters, hardware, concurrency, k, filter policy, and label horizon, and must report latency distributions (p50 and p99), underfill rate, candidate recall with intervals, and memory. No such measurement is reported in this card, and none should be inferred from the constructed numbers in Section 5.

**Takeaway.** The pipeline chooses the ceiling; the model only chooses how much of the ceiling is reached. Comparing two rankers without first comparing their candidate pipelines measures the ceiling, not the ranker.
