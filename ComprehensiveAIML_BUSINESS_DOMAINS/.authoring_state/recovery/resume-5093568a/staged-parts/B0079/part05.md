## 9. Complexity & Cost Analysis

Cost analysis for a candidate pipeline has three distinct parts that are routinely conflated: the algorithmic cost of the retrieval primitive, the serving cost of the whole request path, and the recurring cost of keeping the index and the evaluation consistent. Each is dominated by a different term, and optimising the wrong one produces no visible benefit.

### 9a. Algorithmic complexity

**Exact scan over the eligible set.**
```text
FORMULA:   work = |E| * d multiplications per request, plus |E| log k for top-k selection
WHERE:     |E| = eligible items, d = embedding dimensions, k = cutoff
CONSTRAINT: dense float32 vectors resident or streamed at line rate
NUMERICAL EXAMPLE:
  |E| = 240,000, d = 128
  multiplications = 240,000 * 128 = 30,720,000 MACs
  FLOPs (2 per MAC)                  = 61,440,000
  at 200 GFLOP/s  -> 0.307 ms
  at 1000 GFLOP/s -> 0.061 ms
INTERPRETATION: arithmetic is not the binding constraint; memory movement is (see below).
```

**Why the arithmetic figure is misleading.** A 128-dimension float32 vector is 512 bytes. Scanning 240,000 of them reads 122.88 MB per request. At a sustained 100 GB/s that is 1.23 ms; at 50 GB/s, 2.46 ms. Now scale to the full catalog of 12 million: 6.144 GB per request, which is 61.4 ms at 100 GB/s and 122.9 ms at 50 GB/s. The scan is bandwidth-bound, not FLOP-bound, which is exactly why approximate indexes exist: they trade a small amount of exactness for a thousand-fold reduction in bytes touched.

```text
bytes touched per request, float32, d=128
  eligible set   240,000 * 512 B =   122,880,000 B = 122.88 MB
  full catalog 12,000,000 * 512 B = 6,144,000,000 B =   6.144 GB

at 100 GB/s: 122.88 MB -> 1.23 ms     6.144 GB -> 61.4 ms
at  50 GB/s: 122.88 MB -> 2.46 ms     6.144 GB -> 122.9 ms
at 400 GB/s: 122.88 MB -> 0.31 ms     6.144 GB -> 15.4 ms
```

**ANN traversal.** A graph index visits roughly `m + hops × M` nodes, where m is the requested depth, `hops ≈ log_N` for a graph with branching factor N, and M is the number of neighbours followed per hop.

```text
FORMULA:   D ≈ m + hops * M        hops ≈ log_16(|I|)
WHERE:     D = distance computations, m = depth, |I| = index size, M = neighbour fan-out
CONSTRAINT: graph is connected and vectors are resident in RAM
NUMERICAL EXAMPLE:
  |I| = 12,000,000 -> log_16 = log(12e6)/log(16) = 16.19/2.77 = 5.88 hops
  M = 16, m = 1390:  D ≈ 1390 + 5.88*16 = 1390 + 94 = 1,484
  M = 16, m =   20:  D ≈   20 + 5.88*16 = 20 + 94   =   114
  ratio = 1484/114 = 13.0
INTERPRETATION: a 69.5x increase in requested depth costs only 13x more distance computations,
because the graph-walk overhead is amortised. This is the quantitative case for deepening.
```

Bytes touched: 1,484 × 512 B = 759,808 B ≈ 742 KiB, versus 122.88 MB for the exact scan on the eligible set — a factor of 162 fewer bytes, at the price of approximate rather than exact ordering within the traversal.

**Union, filter, and dedup.** With `n` total source entries and expected constant-time hash membership:
```text
FORMULA:   expected O(n) time, O(k) space for the retained set
WHERE:     n = sum of source list lengths before dedup
CONSTRAINT: a good hash function and a bounded load factor; no sorting inside the union
NUMERICAL EXAMPLE: four sources of 500 rows = 2,000 rows; pairwise overlap 20% removes
  approximately 6 * 500 * 0.20 = 600 duplicate pairs, so the union is about 2,000 - 600 = 1,400
  distinct rows, well above k = 20 and therefore capped.
INTERPRETATION: the union is cheap per row but it is the stage where the candidate count
  becomes unpredictable, which is why it must be measured rather than estimated.
```

**Ranking.** Ranker cost is `c × f` feature evaluations for `c` candidates. It is linear in candidates and therefore the stage that punishes a large `k` most directly: at 2.6 ms per candidate, 20 candidates cost 52 ms and 1,400 candidates cost 3,640 ms. A cheap pre-ranker at 0.05 ms per candidate would cost 12,000 ms over the whole 240,000-item eligible set — which is why exact ranking of the eligible universe is not an option even though exact retrieval of it is.

**End-to-end.** The request path is a sum, not a product, of latencies, with the exception of any parallel source fan-out, which is a maximum.

### 9b. Back-of-envelope sizing

**Serving fleet.** The constructed budget in Section 5 commits 150 ms of p99 against a 250 ms target, of which 126 ms is server-side work (150 minus 24 ms of network). At 40 requests per second and a 50% target utilisation:

```text
server work per request                 = 126 ms = 0.126 s
concurrency needed at 40 rps            = 40 * 0.126        = 5.04 core-equivalents
at 50% target utilisation               = 5.04 / 0.5        = 10.08 cores
on 8-vCPU instances                     = 10.08 / 8         = 1.26 -> 2 instances (with N+1)
monthly compute at USD 0.30/vCPU-hour   = 2 * 8 * 0.30 * 730 = 3,504 USD
```

The per-request figure is the useful one: 1,000,000 requests at 40 rps is 1e6/40/3600 = 6.94 hours, so the compute attributed to a million requests is 6.94 × 0.126 = 0.875 core-hours before utilisation headroom, or 1.75 core-hours at 50%.

**Memory.** The dominant recurring cost is vectors, and it is a constant rather than a per-request cost:
```text
float32 vectors        12e6 * 128 * 4 B = 6.144 GB per replica
graph links (M=64)     12e6 * 64  * 4 B = 3.072 GB per replica
per replica                              = 9.216 GB
two replicas                             = 18.432 GB
int8 vectors           12e6 * 128 * 1 B = 1.536 GB per replica
int8 + graph, 2 replicas                 = 9.216 GB
```

At 8 GB per node the float32 layout needs 2.30 nodes before the operating system and query working set are counted, so the practical minimum is three nodes per replica group, not two. Quantising to int8 halves the footprint but changes score ordering, so it is a re-measurement obligation, not a free saving.

**Index build.** At a constructed 2,000 vectors per second per thread, a full rebuild of 12 million items is 6,000 seconds = 100 minutes single-threaded, or 6.25 minutes across 16 threads. Embedding generation, if done fresh at 5 ms per item, costs 12e6 × 0.005 = 60,000 seconds = 16.7 core-hours. This is why rebuilds are incremental: a full re-embed per rebuild costs more than serving.

**Offline evaluation.** An exact oracle over 400 requests against 240,000 eligible items is 400 × 240,000 = 9.6e7 distances, touching 9.6e7 × 512 B = 49.2 GB. That is a batch job of a few minutes, and it is the cost that makes matched-population evaluation affordable: the expensive part is not the oracle, it is assembling the point-in-time inputs.

**Cost of the depth decision, in one line.** Going from depth 20 to depth 1390 multiplies distance computations by 13.0 and index traversal time roughly in proportion, while leaving memory untouched. That is why depth is the first lever to try and why it must be bounded by a measured underfill target rather than by intuition.

**Cache layers change the shape, not the total.** A 70% hit rate at 4 ms with a 30% miss path at 150 ms gives a mean of 0.7 × 4 + 0.3 × 150 = 47.8 ms, but p99 is still governed by the miss path. Sizing a fleet from the mean under-provisions by more than 3× relative to the p99 requirement.

**Takeaway.** Compare pipelines on bytes touched and candidates scored, not on FLOPs and not on model quality alone; the two former quantities determine whether the design fits the latency and cost envelope at all.

## 10. Key Theorems / Results Reference

Each result is stated with the condition under which it holds, because every one of them is false outside that condition, and the standard misapplication is to quote the conclusion and drop the premise.

**T1 — No-recovery bound.** For any slate D drawn from a candidate set C_k and any relevant set R, `|D ∩ R| ≤ |C_k ∩ R|`. Consequently display recall ≤ candidate recall for every request. *Holds:* always, given D ⊆ C_k. *Misapplication:* quoting it as "ranking does not matter" — the bound is an upper limit, and the ranker determines how close to that limit the system operates.

**T2 — Saturation of recall at k ≥ |E|.** If k ≥ |E(q,t)| then recall@k = 1 for any pipeline returning all eligible items. *Holds:* given complete return. *Misapplication:* comparing systems on a fixture where this is true, which produces a guaranteed tie and is often misinterpreted as "the systems are equivalent".

**T3 — Pooled equals relevant-count weighting.** `sum(hits)/sum(|R|)` and `sum(|R|·recall)/sum(|R|)` are algebraically identical. *Holds:* always, by expansion. *Misapplication:* presenting them as two independent confirmations; they are one estimator computed twice.

**T4 — Macro and pooled can disagree in sign.** There exist request sets where a change increases macro recall and decreases pooled recall. *Holds:* constructible whenever the per-request relevant counts vary. *Misapplication:* defending a chosen aggregate after seeing results; the weighting must be declared in advance.

**T5 — Depth requirement grows like k/p.** The smallest m with `P(Binomial(m,p) ≥ k) ≥ 1−α` scales approximately as k/p, with the exact constant growing slowly as α falls. *Holds:* eligibility independent of rank position. *Misapplication:* applying it when eligibility correlates with score — the correlation always makes the true requirement larger.

**T6 — Independence overestimates joint survival when filters co-reject.** For rejection rates r1, r2 with excess co-rejection ω ≥ 0, survival is `1−r1−r2+r1r2+ω`, which exceeds the independence prediction by exactly ω. *Holds:* by inclusion–exclusion. *Misapplication:* assuming ω = 0 for filters fed by the same upstream system, which is the common case for rights and platform availability.

**T7 — Union coverage is concave in source count.** `1−(1−p)^s` has a strictly decreasing marginal gain `p(1−p)^{s−1}`. *Holds:* source independence. *Misapplication:* planning a five-source ensemble expecting linear coverage growth; the fifth source adds 0.079 where the first added 0.250 at p = 0.25.

**T8 — Approximate top-k fidelity is not relevance.** Fidelity measures agreement with an exact list under a fixed score; it is invariant to whether that score is a good model of utility. *Holds:* always. *Misapplication:* reporting high fidelity as evidence of recommendation quality.

**T9 — Cosine and inner product agree only on the unit sphere.** For unnormalised vectors, `argmax cos(q,x) ≠ argmax q·x` in general. *Holds:* equality when all item vectors have equal norms. *Misapplication:* switching the metric or dropping normalisation during a rebuild and assuming identical ranking behaviour.

**T10 — DCG is cutoff-dependent and grade-scale-dependent.** The same ranking yields different nDCG at different cutoffs, and DCG differences are not comparable across grade scales. *Holds:* always. *Misapplication:* reporting "nDCG improved by 0.03" without the cutoff and grade definition; in the worked example the identical ordering scores 0.952 at cutoff 10 and 0.832 at cutoff 5.

**T11 — Position correction requires positivity.** Inverse-propensity correction of slot effects is unbiased only where the examination propensity is strictly positive for every slot in the analysis. *Holds:* under the positivity condition. *Misapplication:* correcting slot-20 outcomes with an estimated propensity of nearly zero, which produces enormous variance and unstable estimates rather than a correction.

**T12 — Staleness exposure scales linearly with refresh interval.** For state changes arriving uniformly, the expected number of requests served from a stale snapshot is `(T/2)·λ`. *Holds:* uniform arrival. *Misapplication:* using the formula for a rights feed that lands in a daily batch, where the window is deterministic and can be far longer than T/2.

**T13 — Hypergeometric and binomial converge when the sample is small relative to the population.** For sampling m from a population of size E with m/E → 0, the hypergeometric distribution converges to the binomial. *Holds:* approximately when m/E ≤ 0.05. *Misapplication:* using the binomial for a small territory where depth approaches the eligible pool size, which overstates the required depth.

**T14 — Recall is undefined, not zero, on an empty relevant set.** `0/0` is undefined; any reported numeric value is a convention. *Holds:* always. *Misapplication:* substituting zero, which makes a cohort with immature labels look like a retrieval failure proportional to its label lag.

**T15 — Exposure concentration is a fixed point of a feedback loop.** A policy that trains on exposure-selected labels and selects by predicted engagement can drive Gini toward 1 without any change in user preference. *Holds:* for policies whose training distribution is their own output distribution. *Misapplication:* interpreting rising concentration as evidence of genuine demand concentration; the two are observationally similar and must be separated by exploration.

**T16 — Coverage of the eligible catalog bounds recall against observed labels.** Items that were never exposed cannot be labelled, so a recall metric computed on observed labels cannot detect a retrieval regression that affects only unexposed items. *Holds:* always. *Misapplication:* treating a stable recall figure as evidence that the candidate pipeline is healthy while coverage falls.

**Takeaway.** Each theorem is a constraint on a claim, not a result to cite. The useful reading is the "misapplication" line: those are the sentences that appear in review documents just before a decision goes wrong.

## 11. Interview / Exam Questions

Fifteen questions with independent answers. Questions 4, 5, 6, 8, 9, 10, 12, and 14 require a computation or a diagnostic decision; the rest test definitions and judgement. Answers are written to be scored on content, not on confidence.

### Q01 — Why does a ranker's quality not determine end-to-end recommendation quality?

**Answer:** Because the ranker only reorders what retrieval supplies. For any slate D ⊆ C_k, |D ∩ R| ≤ |C_k ∩ R|, so display recall cannot exceed candidate recall. If candidate recall is 0.45 and the ranker perfectly surfaces every relevant candidate, the ceiling is still 0.45. A better ranker closes the gap between 0.45 and the ceiling; it cannot raise the ceiling.

**Scoring:** full credit states the bound and attributes the ceiling to retrieval; partial credit says "both matter" without the inequality.

### Q02 — Define eligibility so that it can be computed from logs.

**Answer:** Eligibility is a predicate over the tuple (item, viewer, surface, decision timestamp) that returns true when every hard constraint — rights for the viewer's territory, entitlement, asset availability, device capability, safety policy — is satisfied at that instant. It is computable from a versioned policy state plus the viewer's attributes at the same timestamp. It is not a per-item attribute, and it is not a model score.

**Scoring:** full credit names the tuple and the timestamp; no credit for "whether the user can see the item" without the timestamp.

### Q03 — Why is an item that was never exposed not a negative label?

**Answer:** Its outcome is unobserved, and the observation mechanism depends on the policy that chose not to expose it. Encoding it as negative teaches the model to reproduce the exposure policy rather than to estimate preference, and the effect compounds because exposure decisions are then made from those predictions. The correct handling is to exclude unexposed items from label-based training or to model exposure explicitly.

**Scoring:** full credit identifies the selection mechanism; partial credit mentions "missing data" without the policy dependence.

### Q04 — With E = 240,000, R = 7, and C_20 containing 4 relevant items, compute candidate recall and state what it does not tell you.

**Answer:** Recall = 4/7 = 0.5714. It does not tell you the ranker's quality, the slate's quality, whether the other three relevant items were reachable, the precision (4/20 = 0.20), or anything about the population — this is one request. It also does not tell you whether the four items were ranked highly or placed at the bottom of the slate.

**Scoring:** full credit gives 0.5714 and at least three distinct limitations; the common error is reporting 4/20 = 0.20, which is precision.

### Q05 — At 2% eligibility density, k = 20, and a 5% underfill tolerance, how deep must you retrieve? Show the check.

**Answer:** m = 1390. Check: the binomial tail with mean 1390 × 0.02 = 27.8 and sd sqrt(1390 × 0.02 × 0.98) = 5.22 gives P(X ≥ 20) = 0.9502, so P(underfill) = 0.0498 ≤ 0.05. The ratio m/k = 69.5. At density 1% the same requirement is m = 2784; at 10% it falls to 275.

**Scoring:** full credit gives a depth within about 5% of 1390 and shows the tail computation; partial credit gives the ratio intuition (roughly k/p) without the tolerance.

### Q06 — A pipeline requests depth 20 globally and filters afterwards. With p = 0.02, what fraction of requests gets any eligible candidate?

**Answer:** Each of the 20 returned items survives filtering with probability 0.02 independently, so P(no survivor) = 0.98²⁰ = 0.6676 and P(at least one) = 0.3324. The expected number of survivors is 20 × 0.02 = 0.4. So roughly two thirds of requests receive an empty candidate set from this path, and the average request receives 0.4 items against a budget of 20.

**Scoring:** full credit computes 0.3324 and the expectation 0.4; partial credit recognises the problem without the probability.

### Q07 — Two filters reject 12% and 18% of items. What survives, and what changes if they co-reject?

**Answer:** Under independence, survival is 1 − 0.12 − 0.18 + 0.12 × 0.18 = 0.7216, so 72.16% survive. If the excess co-rejection ω is 0.06, survival rises to 0.7816 because more items are rejected by both filters simultaneously. The direction is counter-intuitive but forced by inclusion–exclusion: correlation in rejection increases joint survival.

**Scoring:** full credit gives both numbers and explains the direction; partial credit gives 0.7216 only.

### Q08 — Recall rises from 0.38 to 0.48 and the online primary metric is flat. Give three causes and the check for each.

**Answer:** (a) Ranker saturation — retention was already near 1.0, so extra candidates were redundant; check the share of relevant candidates reaching the slate before and after. (b) Objective mismatch — the added candidates satisfy the offline label but not the online objective; check the online outcome rate conditional on the newly added items. (c) Position — the new items land below the last examined slot; check their exposure-weighted position distribution. A fourth: the split was not honoured; check arm assignment against configuration.

**Scoring:** full credit gives at least three causes each with an observable check; naming causes without checks is partial.

### Q09 — Design an eligibility check that cannot be defeated by a stale index.

**Answer:** Keep the index as a pure retrieval structure containing only IDs and embeddings, with no eligibility decisions. Evaluate eligibility against a versioned policy service at request time, after retrieval and again immediately before render, and fail closed on lookup errors. Log the policy version and evaluated timestamp with the response so the check is auditable. Index freshness then affects candidate *quality* only, never correctness, and the residual risk is bounded to the latency of the policy service rather than to the refresh interval.

**Scoring:** full credit separates retrieval freshness from eligibility correctness and specifies fail-closed plus logging; partial credit proposes frequent rebuilds only, which does not bound the window.

### Q10 — You must choose between depth 1400 filtered ANN and a per-territory exact index. What decides it?

**Answer:** Two quantities. First, eligibility density: if territory is the dominant rejection and per-territory density is high (say 25%), the exact index is cheap and exactness is free. Second, storage multiplication: n territories multiplies vector memory by n, so ten territories at 9.216 GB per replica becomes 92.16 GB. The comparison is depth-induced latency (about 38 ms at depth 1390 in the constructed budget) against storage and rebuild time. If territories are few and large, partition; if many and small, use filter-aware depth.

**Scoring:** full credit names both density and storage multiplication with numbers; partial credit names one.

### Q11 — What is the difference between ANN fidelity and candidate recall?

**Answer:** Fidelity is |A_k ∩ X_k|/|X_k|, the share of the exact top-k under a fixed score that the approximate index recovered. It measures the index. Candidate recall is |C_k ∩ R|/|R ∩ E|, the share of labelled relevant items the candidate set contains. It measures the pipeline against ground-truth relevance. A pipeline can have fidelity 1.0 and recall 0.1: it reproduces the exact neighbours perfectly, and the exact neighbours are mostly irrelevant.

**Scoring:** full credit gives both formulas and the 1.0/0.1 example; partial credit contrasts them only qualitatively.

### Q12 — Compute nDCG@3 for grades [3,0,2] and for [3,2,0]. What does the difference mean?

**Answer:** DCG([3,0,2]) = 7/1 + 0/1.585 + 3/2 = 8.5000. DCG([3,2,0]) = 7/1 + 3/1.585 + 0/2 = 8.8928. The ideal ordering of these grades is [3,2,0], so IDCG = 8.8928 and nDCG is 0.9558 versus 1.0000. The difference of 4.4 points is what one adjacent swap of a grade-2 item is worth at these positions — so any claimed gain smaller than that is within the granularity of a single move.

**Scoring:** full credit computes both and interprets the granularity; partial credit computes without interpreting.

### Q13 — Why can a higher candidate recall be a bad decision?

**Answer:** Because recall is bought with depth, and depth costs latency, memory bandwidth, and often quality through the inclusion of marginal items. If the ranker cannot use the extra candidates (retention already near 1.0, or the slate has 20 slots), the gain is invisible to users while the latency regression is visible to all of them. Recall must be reported with the depth, latency, and underfill that produced it.

**Scoring:** full credit names the cost side and the saturation mechanism; partial credit says "it depends on the cost" without mechanism.

### Q14 — A cohort reports recall of exactly 1.00 with no variance. Diagnose.

**Answer:** Almost certainly a degenerate denominator or a fixture artefact: k ≥ |E| so every pipeline returns everything (T2), or |R ∩ E| = 1 for every request so each recall is 0 or 1 and the mean happens to be 1, or a fallback guarantees the single labelled item. Check the distribution of |R ∩ E| and of k/|E|. A genuine 1.00 across a cohort with varied denominators is essentially impossible.

**Scoring:** full credit names saturation and the denominator distribution check; partial credit says "suspicious" without a check.

### Q15 — What single measurement would you require before shipping a retrieval change?

**Answer:** Candidate recall on a matched cohort with confidence intervals, computed against an eligible labelled population with a fixed horizon, reported together with underfill rate and p99 latency at the target QPS. The three together prevent the two failure modes that pass a recall-only gate: a recall gain bought with underfill, and a recall gain bought with latency. If forced to choose one, candidate recall on a matched cohort, because without it nothing else is attributable.

**Scoring:** full credit names matched cohort plus intervals and at least one companion metric; partial credit names recall alone.

**Takeaway.** The distinguishing feature of a strong answer here is that it states what the number does *not* license. Every question above has a plausible wrong answer that sounds more confident than the right one.
