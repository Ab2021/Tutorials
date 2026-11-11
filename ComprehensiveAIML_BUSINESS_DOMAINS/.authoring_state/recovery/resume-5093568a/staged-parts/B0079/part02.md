## 4. Core Formulas with Full Derivation

Every result below follows the same anatomy: the formula, the symbols and their domains, the constraints that make it true, a checked numerical example, and the interpretation that the number licenses. Each subsection closes with an explicit validity condition, because the most common error is not arithmetic — it is applying a formula outside the population that defines it.

Notation used throughout: `|X|` is the cardinality of set X; `∩` is intersection; `q` indexes requests; `k` is the candidate budget in items; `H` is the label horizon; `E(q,t)` is the eligible set at decision time; `R(q,H)` is the labelled relevant subset; `C_k(q)` is the first k candidates under the declared ordering.

### 4.1 Base Case — candidate recall for one request and one source

**FORMULA [R1]**
```text
recall_at_k(q) = |C_k(q) ∩ R(q,H)| / |R(q,H) ∩ E(q,t)|
```

**WHERE**
- `C_k(q)`: the first k distinct IDs emitted by the pipeline, ordered by the pipeline's own score with ties broken by the declared rule. Units: item IDs. Domain: subset of the retrieval index at time t.
- `R(q,H)`: IDs satisfying the label rule inside horizon H after exposure. Units: item IDs. Domain: subset of E(q,t) by construction, because unshowable items cannot have observed labels.
- `E(q,t)`: IDs passing every hard predicate for this viewer, surface, and timestamp.
- Result: a proportion in [0,1], dimensionless, defined only when |R ∩ E| ≥ 1.

**CONSTRAINT**
1. `|R(q,H) ∩ E(q,t)| > 0`. With a zero denominator the ratio is undefined; report the count of excluded requests separately instead of substituting 0 or 1.
2. The label horizon H is identical for every system being compared, and the outcome window for each request has matured.
3. k counts distinct eligible IDs after deduplication, not raw source rows.
4. Eligibility is evaluated at the decision timestamp t, not at label time. Using eligibility at label time lets a system that would have shown a revoked item appear correct.
5. The label rule is fixed before inspection. Choosing the rule after seeing both systems' outputs converts measurement into selection.

**NUMERICAL EXAMPLE**
Take one constructed request. The eligible set is E = {A, B, C, D, E} of five IDs. The labelled relevant set inside the horizon is R = {B, D}. System P returns C_3 = {A, B, C}. System Q returns C_3 = {B, A, D}.

```text
P: |{A,B,C} ∩ {B,D}| = |{B}| = 1        recall = 1/2 = 0.50
Q: |{B,A,D} ∩ {B,D}| = |{B,D}| = 2      recall = 2/2 = 1.00
gap = 0.50 absolute = 50 percentage points on this single request
```

**INTERPRETATION** System P gives the ranker one of the two relevant items; Q gives both. On this request only, no statement about population superiority, precision, revenue, or significance follows. The correct use is as a unit test of the harness: if a pipeline of known composition does not reproduce these two numbers, the harness is wrong and no aggregate is trustworthy.

**Related count that is not complementary.** Precision at k is `|C_k ∩ R| / k`. For P it is 1/3 = 0.333 and for Q it is 2/3 = 0.667. Recall and precision both move here, but they need not: if R contained ten items and both systems returned two hits, recall would be identical at 0.2 while precision stayed 0.333 and 0.667. Precision measures how much of the budget was wasted; recall measures what was lost. Only recall is bounded by the candidate stage.

**The no-recovery bound, proved in one line.** For any displayed slate D ⊆ C_k, we have D ∩ R ⊆ C_k ∩ R, so |D ∩ R| ≤ |C_k ∩ R|. Dividing by the positive denominator |R ∩ E| preserves the inequality. Therefore display recall ≤ candidate recall for every request, with equality only when the ranker surfaces every relevant candidate. This is why an end-to-end recall regression can be diagnosed by checking candidate recall first: if candidate recall already fell, no ranker change can be the explanation.

**Validity condition for 4.1.** Applies to a single request with a nonempty eligible labelled set, a fixed cutoff k, and a declared determination of what counts as a candidate. It does not extend to a population until Section 4.2's aggregation rules are applied.

### 4.2 General Case — aggregating across requests, and correcting for filter correlation

**FORMULA [R2] — three defensible aggregates**
```text
macro   = (1/|Q+|) * sum over q in Q+ of recall_at_k(q)
micro   = (sum over q of hits(q)) / (sum over q of |R(q) ∩ E(q)|)
weighted= (sum over q of |R(q)| * recall_at_k(q)) / (sum over q of |R(q)|)
```
where `Q+` is the set of requests with |R(q) ∩ E(q)| ≥ 1 and `hits(q) = |C_k(q) ∩ R(q)|`.

**WHERE:** each `recall_at_k(q)` is the [R1] value; all three are proportions in [0,1]; `Q+` is a set of requests, so |Q+| is a count, not a rate.

**CONSTRAINT:** all three must be computed over the identical request set `Q+`, cutoff k, eligibility snapshot rule, and horizon H. Substituting one for another between systems invalidates the comparison. Micro equals weighted algebraically — both pool hits against pooled relevant counts — so the real choice is macro versus pooled, not three options.

**NUMERICAL EXAMPLE** Three constructed requests with candidate sets of size k = 4:

```text
request   hits   |R ∩ E|   recall
q1          2        4      0.50
q2          1        1      1.00
q3          2        8      0.25
totals      5       13

macro    = (0.50 + 1.00 + 0.25) / 3 = 1.75 / 3 = 0.58333
micro    = 5 / 13                                    = 0.38462
weighted = (4*0.50 + 1*1.00 + 8*0.25) / 13
         = (2 + 1 + 2) / 13                          = 0.38462
```

**INTERPRETATION** The macro figure says the typical request recovers about 58% of its relevant items. The pooled figure says 38% of all relevant item-request pairs were recovered. The gap of 19.9 percentage points exists entirely because q3 has eight relevant items and drags the pair-level rate down. Reporting only one of them without naming the weighting is the single most common way two teams disagree about the same pipeline.

**Why the choice is a product decision, not a statistical one.** If the business question is "does each user get a reasonable slate", weight by request (macro). If it is "what fraction of the inventory a user might want actually reaches them", weight by relevant items (pooled). A pipeline can be improved on one and degraded on the other simultaneously: adding a targeted source that only helps heavy-interest requests raises pooled recall while leaving macro flat, and adding a fallback that gives every request one hit raises macro while barely moving pooled.

**FORMULA [R3] — uncertainty on a proportion**
```text
Wilson: ( p̂ + z²/(2n) ± z * sqrt( p̂(1-p̂)/n + z²/(4n²) ) ) / (1 + z²/n)
```
with `p̂ = x/n`, x the count of successes, n the number of independent units, z = 1.96 for 95%.

**WHERE:** p̂ ∈ [0,1]; n is a count of independent units; the interval is on the same scale as p̂. For candidate recall, the "success" must be defined as a binary event per unit — for example "this request recovered at least one relevant item" or "this labelled pair was recovered" — because a per-request proportion is not a Bernoulli count.

**CONSTRAINT:** the units must be approximately independent. Labelled items clustered inside a request are not independent, and requests clustered inside a user are not independent either. When clustering is present the Wilson width is too narrow; multiply the variance by a design effect estimated from between-cluster variance before quoting the interval.

**NUMERICAL EXAMPLE** A constructed evaluation of 400 requests in which 180 recovered at least one relevant candidate.

```text
p_hat   = 180/400 = 0.450
z²/n    = 3.8416/400       = 0.0096040
z²/(2n) = 3.8416/800       = 0.0048020
centre  = (0.450 + 0.0048020) / 1.0096040 = 0.4548020/1.0096040 = 0.450470
variance term = p(1-p)/n = 0.45*0.55/400 = 0.00061875
z²/(4n²)      = 3.8416/640000            = 0.0000060025
root          = sqrt(0.0006247525)        = 0.02499505
half          = 1.96 * 0.02499505 / 1.0096040 = 0.048524
interval      = 0.450470 ± 0.048524       = [0.40195, 0.49899]
```

Rounded to four decimals the interval is [0.4020, 0.4990], which matches a direct numerical evaluation of the Wilson formula. At n = 40 requests with 18 successes the same computation gives [0.3071, 0.6017] — a width of 29.5 points, which is why a 40-request offline sample cannot adjudicate a 3-point difference.

**INTERPRETATION** With 400 requests, a 6-point gap between two arms (0.45 versus 0.39) yields a standard error of sqrt(0.45·0.55/400 + 0.39·0.61/400) = 0.0348 and z = 1.72, which is not significant at 95%. Reaching 80% power for that gap needs roughly 1,057 requests per arm; detecting a 3-point gap needs about 4,228 per arm. The arithmetic is the argument for a large offline cohort or a paired design, not for lowering the threshold.

**FORMULA [R4] — eligibility attrition with correlated filters**
```text
surviving(r1, r2, omega) = 1 - r1 - r2 + (r1*r2 + omega)
```
where `r1`, `r2` are rejection rates of two filters, and `omega = P(reject A ∩ reject B) - r1*r2` is the excess co-rejection above independence.

**WHERE:** r1, r2, ω ∈ [0,1]; survival ∈ [r1,r2-dependent], and the formula is exact by inclusion–exclusion: `P(keep both) = 1 - P(A) - P(B) + P(A ∩ B)` and `P(A ∩ B) = r1·r2 + ω`.

**CONSTRAINT:** ω must be estimated from a sample where both filters are evaluated on the same items. If the filters are applied in sequence and the second is only evaluated on survivors of the first, ω is unobservable from production logs and must come from an offline join.

**NUMERICAL EXAMPLE** Two filters: territory rights rejects r1 = 0.12, platform availability rejects r2 = 0.18.

```text
base            = 1 - 0.12 - 0.18        = 0.70
independence    = 0.70 + 0.12*0.18       = 0.70 + 0.0216 = 0.7216
co-rejecting    = 0.70 + 0.0216 + 0.06   = 0.7816      (omega = 0.06)
```

The independence assumption survives 72.16% of items; measured co-rejection of 6 points means only 78.16% are rejected overall, i.e. 6 more points survive than independence predicted. A planner who used independence would under-provision the candidate budget by 8.3% of the surviving set.

**INTERPRETATION** Filters driven by the same underlying commercial fact co-reject. Rights and platform availability both derive from distribution deals, so the overlap is large in practice; the sign of the error from ignoring ω is knowable in advance if you check which upstream system feeds both predicates. For two filters with ω = 0 the correction is exact and small; the estimate should still be recomputed per territory, because the overlap is a property of the deal book, not of the platform.

**Validity condition for 4.2.** Applies to a fixed request set with matched eligibility snapshot, horizon, cutoff, and label rule, with clustering accounted for in any interval. It does not license comparison across different k, different H, or different eligibility vintages.

### 4.3 Variants

**FORMULA [A1] — approximate-search fidelity**
```text
ann_fidelity_at_k = |A_k(q) ∩ X_k(q)| / |X_k(q)|
```
with `X_k` the exact top-k under the same vectors, metric, eligible universe, and tie rule, and `A_k` the index's output.

**WHERE:** both lists contain distinct eligible IDs; the denominator is the length of the exact list, which equals k only when at least k eligible items exist.

**CONSTRAINT:** identical item vectors and query vector at the same snapshot; identical similarity function (inner product and cosine diverge unless vectors are normalised); identical tie-breaking, otherwise the comparison is dominated by arbitrary order among equal scores.

**NUMERICAL EXAMPLE** Exact eligible top-3 X_3 = {A, B, C}, approximate A_3 = {A, C, D}. Overlap = {A, C}, so fidelity = 2/3 = 0.667. If ties at the boundary are numerous (say 40 items share the third-highest score), fidelity becomes a random variable with expectation depending on the tie rule, and the correct statement is a distribution over tie resolutions, not a point value.

**INTERPRETATION** Two of three exact-score neighbours were returned. This is a property of the index, not of the recommendations: a fidelity of 0.667 is compatible with excellent relevance and with terrible relevance, because X_k itself may be irrelevant.

**FORMULA [D1] — filtered-search depth requirement**
```text
P(survive) = sum_{i=k}^{m} C(m,i) p^i (1-p)^(m-i)   must be >= 1 - alpha
```
Solve for the smallest m given k, p, and a tolerance `alpha` (commonly 0.05). The model treats the top-m returned by the index as an ordered sample whose eligibility is Bernoulli(p) — an assumption justified when eligibility is independent of score, and violated when it is not.

**WHERE:** m = retrieval depth requested from the index; k = eligible items wanted; p = eligibility density for this request; alpha = tolerated underfill probability.

**CONSTRAINT:** independence of eligibility from rank position. This fails in exactly the interesting cases: geography-keyed catalogs, language filters, and subscription tier filters all correlate with score, because the model was trained on a catalog skewed the same way. When p is correlated with score, the binomial model under-predicts underfill.

**NUMERICAL EXAMPLE** Checked by direct summation of the binomial tail:

```text
k=20  p=0.01  m=2784  P(X>=20)=0.9501   (depth/k = 139.2)
k=20  p=0.02  m=1390  P(X>=20)=0.9502   (depth/k =  69.5)
k=20  p=0.05  m= 554  P(X>=20)=0.9509   (depth/k =  27.7)
k=20  p=0.10  m= 275  P(X>=20)=0.9514   (depth/k =  13.8)
k=10  p=0.02  m= 782  P(X>=10)=0.9501   (depth/k =  78.2)
k=50  p=0.05  m=1237  P(X>=50)=0.9502   (depth/k =  24.7)
k=50  p=0.01  m=6211  P(X>=50)=0.9501   (depth/k = 124.2)

sensitivity at k=20, p=0.02:
m=1000 -> P=0.5306   m=1390 -> P=0.9502
m=1500 -> P=0.9791   m=2000 -> P=0.9998
```

Mean survivors at m = 1390 with p = 0.02 is 1390 × 0.02 = 27.8, with standard deviation sqrt(1390 × 0.02 × 0.98) = 5.22. The 5% lower quantile of that distribution is 20, which is exactly the condition being imposed by requiring P(X ≥ 20) ≥ 0.95.

**INTERPRETATION** At 2% eligibility density, a depth-1000 request underfills more than 46% of the time. This quantifies the difference between "we request 1000 and keep 20" and "we reliably obtain 20" — the ratio m/k is 69.5 at p = 0.02 and k = 20, and 15.4 at p = 0.10 and k = 10. Depth is not a tuning knob; it is a cost that scales inversely with eligibility density.

**FORMULA [U1] — union coverage across independent sources**
```text
coverage(s) = 1 - (1 - p)^s
```
for s independent sources each independently recovering a given relevant item with probability p.

**WHERE:** p ∈ [0,1] per-source recovery probability, s ∈ integers ≥ 1, coverage = probability at least one source recovers the item.

**CONSTRAINT:** source independence. Popularity and co-visitation sources on a head-heavy catalog are strongly dependent, so the formula overstates coverage; the correct generalisation replaces `(1-p)^s` with the probability that all sources miss, measured empirically per item segment.

**NUMERICAL EXAMPLE** With p = 0.25 per source, three sources give 1 − 0.75³ = 1 − 0.4219 = 0.5781, and five sources give 1 − 0.75⁵ = 1 − 0.2373 = 0.7627. Adding the fourth and fifth sources buys 18.5 points, roughly half of what the first three bought.

**INTERPRETATION** Diminishing returns are structural, not an implementation flaw. A fifth source is worth adding only if its marginal coverage exceeds its marginal latency and its marginal risk of injecting low-quality or ineligible items.

**FORMULA [E1] — graded ranking quality and exposure concentration**
```text
DCG@n = sum_{i=1}^{n} (2^{rel_i} - 1) / log2(i + 1)
nDCG@n = DCG@n / IDCG@n
Gini  = sum_{i} (2i - n - 1) x_(i) / (n * sum_i x_(i))     with x sorted ascending
```

**WHERE:** `rel_i` is an integer grade ≥ 0 for the item at position i; IDCG is the DCG of the grading sorted descending; `x_(i)` are exposure shares summing to 1.

**CONSTRAINT:** gains must be defined on a fixed grade scale; a single swap of adjacent items changes DCG by an amount that depends on both grades, so DCG differences are not comparable across different grade scales.

**NUMERICAL EXAMPLE — nDCG with a real recomputation.**
```text
position:      1      2      3      4      5      6      7      8      9     10
grade:         3      2      0      1      0      2      1      0      0      1
gain 2^g-1:    7      3      0      1      0      3      1      0      0      1
log2(i+1):  1.000  1.585  2.000  2.322  2.585  2.807  3.000  3.170  3.322  3.459
discounted: 7.000  1.893  0.000  0.431  0.000  1.069  0.333  0.000  0.000  0.289
DCG@10 = 7.000+1.893+0.431+1.069+0.333+0.289 = 11.0145

ideal order [3,2,2,1,1,1,0,0,0,0]:
discounted: 7.000  1.893  1.500  0.431  0.387  0.356  ... = 11.5665
nDCG@10 = 11.0145 / 11.5665 = 0.9523

truncated at 5:
DCG@5   = 7.000+1.893+0+0.431+0 = 9.3235
IDCG@5  = 7.000+1.893+1.500+0.431+0.387 = 11.2103
nDCG@5  = 9.3235 / 11.2103 = 0.8317
```

The same ranking scores 0.952 at cutoff 10 and 0.832 at cutoff 5, so quoting "nDCG" without a cutoff is meaningless. The cutoff-5 penalty appears because two of the three remaining graded items sit at positions 6 and 7, outside the measured window.

**A single adjacent swap, sized.** Consider [3,0,2] versus [3,2,0] over three slots:
```text
DCG([3,0,2]) = 7/1.000 + 0/1.585 + 3/2.000 = 7 + 0 + 1.500 = 8.5000
DCG([3,2,0]) = 7/1.000 + 3/1.585 + 0/2.000 = 7 + 1.893 + 0 = 8.8928 = IDCG
nDCG         = 8.5000/8.8928 = 0.9558   versus   8.8928/8.8928 = 1.0000
```
A single swap of a grade-2 item from slot 3 to slot 2 moves nDCG by 4.4 points. Any claimed nDCG gain smaller than the granularity of one realistic swap is not evidence of a better ranker.

**Exposure concentration, sized.** Twelve constructed exposure shares summing to 1.00:
```text
[0.20, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
Gini = 0.3233
```
For comparison, a five-item distribution [0.50, 0.20, 0.15, 0.10, 0.05] gives Gini = 0.4000, and [0.25, 0.24, 0.23, 0.22, 0.06] gives 0.1600. Gini is scale-free in n but not in the truncation of the tail: computing it over the top 12 items of a 12-million-item catalog measures slate concentration, not catalog coverage.

**INTERPRETATION** nDCG answers "was the ordering good within the shown slots"; Gini answers "was exposure spread across the eligible inventory". A pipeline can maximise nDCG while driving Gini toward 1, and that combination is the signature of a feedback loop that only ever exposes what it already exposed.

**FORMULA [S1] — staleness exposure window**
```text
expected_stale_requests = (refresh_interval / 2) * request_rate
```
for uniformly distributed revocation times relative to a periodic refresh.

**WHERE:** refresh_interval in seconds, request_rate in requests per second, result in requests.

**CONSTRAINT:** uniform arrival of state changes relative to refresh boundaries; a batch rights feed landing at a fixed hour violates this and produces a deterministic window instead of an average.

**NUMERICAL EXAMPLE** A 900-second rebuild with 40 requests per second:
```text
mean age = 900/2 = 450 s
expected requests served from the pre-change snapshot = 450 * 40 = 18,000
```
**INTERPRETATION** Rebinding the catalog cannot bound the window; only a check at request time can. The 18,000 figure is the arithmetic argument for a last-mile eligibility check: it is the number of decisions that a periodic refresh alone would have made on stale state.

**Validity condition for 4.3.** Each variant holds under its stated independence or comparability assumption. A and D assume eligibility independent of score; U assumes source independence; E assumes a fixed grade scale and a declared cutoff; S assumes uniform arrival. Every one of those assumptions is testable.

### 4.4 Limits and Counterexamples

**Limit 1 — the random-selection floor.** The D1 binomial model says nothing about relevance; it only says whether the returned items are eligible. The relevant question is how many relevant items arrive by chance. For uniform random selection of k items from E containing R relevant items:
```text
P(no relevant in k) = C(E-R, k) / C(E, k)
E = 240,000, R = 40, k = 20:  C(239,960,20)/C(240,000,20) = 0.99667181
P(>=1 relevant) = 0.003328   versus the approximation kR/E = 20*40/240000 = 0.003333
```
Random retrieval recovers at least one relevant item about 0.33% of the time. A measured recall of 0.45 at k = 20 is therefore about 5,400 times the random floor, which is the honest way to express the strength of a retrieval signal: report lift over the floor, not the raw proportion, because a raw proportion near 0.45 is uninterpretable without knowing how many relevant items exist.

**Limit 2 — hypergeometric versus binomial.** The D1 depth calculation assumes independent draws with replacement-like eligibility; the exact finite-population statement is hypergeometric. At p = 0.02 and m = 1390 out of a 240,000-item eligible pool, the binomial and hypergeometric tails differ in the fourth decimal, so the approximation is safe. It stops being safe when m approaches the pool size, which happens for narrow cohorts: a small territory with 3,000 eligible items and a depth of 1,390 makes the finite-population correction material, and the required depth becomes smaller than the binomial model predicts.

**Counterexample 1 — filter after global search.** A pipeline requests the global top-20, then filters. Suppose the eligible relevant items exist but every one of the 20 global neighbours is ineligible in this territory. Retrieval recall is 0 while eligible exact neighbours are abundant. The defect is architectural: filtering after truncation cannot recover items below the truncation boundary. The discriminating check is to compare measured recall against a filter-aware oracle, not against the global list.

**Counterexample 2 — exact is not correct.** An exact top-k under a single embedding is exact only with respect to that embedding. If the embedding was trained on click data collected under a click-maximising policy, its exact neighbours are the items the old policy was already inclined to show. Calling this list "ground truth" mistakes the estimand for the truth, and an ANN index that deviates from it may be better rather than worse.

**Counterexample 3 — recall 1.0 by construction.** If k ≥ |E|, recall@k is 1.0 for every system that returns the whole eligible set. The metric saturates and can no longer discriminate. This happens quietly when a test fixture uses a ten-item catalog with k = 100. Guard: assert k < |E| in any fixture used to compare systems.

**Counterexample 4 — the empty denominator.** If R ∩ E is empty for a request — perfectly possible when the label horizon has not matured, or when the user's interest lies entirely outside the rights envelope — recall is 0/0. Teams that silently set it to 0 will report a pipeline as worse in proportion to how many immature requests a cohort contains. Teams that silently set it to 1 will report the reverse. Both are fabricated numbers; the correct output is the count of undefined requests and the reason distribution.

**Counterexample 5 — weighting flips the winner.** Under the [R2] example, suppose System P improves q3 (the eight-relevant-item request) by 0.10 and System Q improves q1 (four relevant items) by 0.30. Macro moves +0.033 for P and +0.100 for Q; pooled moves +0.80/13 = +0.062 for P and +1.20/13 = +0.092 for Q. The two aggregates agree in direction here, but with eight-relevant-item dominance reversed they readily disagree. The lesson is procedural: the weighting must be declared before the results are seen, because after the fact both aggregates are defensible.

**Counterexample 6 — union coverage with dependent sources.** With p = 0.25 per source and perfect dependence, five sources give coverage 0.25, not 0.7627. Measured values between 0.25 and 0.7627 identify the effective number of independent sources, which is a useful diagnostic: it tells you how much of the ensemble is redundant.

**Limit 3 — attrition compounding.** Sequential filters multiply. Starting from 1,000 candidates with rejection rates 0.12, 0.18, 0.05, and 0.07 in sequence:
```text
1000.0 -> 880.0 -> 721.6 -> 685.5 -> 637.5
```
Four modest-looking filters remove 36.3% of the candidates. A team that planned for "about 10% loss per filter" expects 656 survivors and is close; a team that planned for one filter's loss only is off by a factor of 1.57 in candidate budget.

**Validity condition for 4.4.** Each limit names the assumption it breaks. The general rule: any metric quoted without its universe, cutoff, horizon, and weighting is not a measurement but an anecdote with a decimal point.
