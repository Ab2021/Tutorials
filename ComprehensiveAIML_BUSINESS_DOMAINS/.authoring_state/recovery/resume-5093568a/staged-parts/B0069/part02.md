## 4. Core Formulas with Full Derivation

Every block below uses the same five-part anatomy: formula, where the symbols come from, the constraint under which the formula is valid, a numerical example with the arithmetic shown, and an interpretation that states what the number does and does not license. Each block closes with its validity condition; the section ends with the cases where the family breaks.

### 4.1 Base case: candidate recall for one request

**[F1] FORMULA:** `cand_recall_K(q) = |C_K(q) ∩ R(q)| / |R(q)|`

**WHERE:** `C_K(q)` is the de-duplicated, eligibility-filtered candidate list for request q truncated to budget K; `R(q)` is the set of eligible items satisfying the declared relevance rule inside the matured horizon H; the vertical bars count distinct IDs. The numerator is a count, the denominator is a count, and the quotient is dimensionless on [0,1].

**CONSTRAINT:** `|R(q)| > 0`. Duplicate handling, tie-breaking at the K boundary, the label rule, and the horizon must all be fixed before any two systems are compared. If `C_K` was built against a different eligibility snapshot than `R` was defined on, the quotient is not candidate recall of anything.

**NUMERICAL EXAMPLE:** Suppose `R = {i2, i3, i5, i9}` and a retrieval source returns `C_4 = {i2, i7, i9, i11}`. The intersection is `{i2, i9}`, so the numerator is 2 and the denominator is 4. `cand_recall_4 = 2/4 = 0.50`. If the budget were cut to K = 3 and the list became `{i2, i7, i9}`, the numerator stays 2 and the ratio stays 0.50; the budget cut removed an irrelevant item, so recall was unaffected.

**INTERPRETATION:** Half the known-relevant eligible items were available to the ranker. The ranker can at best display two of four; the other two are lost before ranking begins and no amount of model improvement recovers them. This number says nothing about whether the two surviving items were placed well, and nothing about items that are relevant but unlabeled.

**Validity condition:** The set R is stable, nonempty, and defined on the same eligibility snapshot as `C_K`; K is recorded per request rather than assumed constant.

### 4.2 General case: aggregating over a request population

**[F2] FORMULA:** `macro = (1/|Q+|) * Σ_{q∈Q+} cand_recall_K(q)`  and  `micro = Σ_{q∈Q+} hits(q) / Σ_{q∈Q+} |R(q)|`

**WHERE:** `Q+` is the set of requests with at least one eligible relevant item; `hits(q) = |C_K(q) ∩ R(q)|`. Macro weights each request equally; micro weights each request in proportion to how many relevant items it has.

**CONSTRAINT:** The same `Q+`, the same K distribution, the same eligibility rule and the same label horizon must hold for both sides of any comparison. If one system serves a request with K = 400 and another with K = 40 because of a degraded path, the two are not comparable at "K", and the mixture must be reported by path.

**NUMERICAL EXAMPLE (ILLUSTRATIVE):** Six requests with per-request recalls 1.00, 0.50, 0.00, 0.75, 0.50, 1.00 and relevant counts 2, 4, 1, 4, 2, 5. Hits are the products: 2, 2, 0, 3, 1, 5, summing to 13. Macro is `(1.00+0.50+0.00+0.75+0.50+1.00)/6 = 3.75/6 = 0.625`. Micro is `13/18 = 0.7222`. The gap of 0.097 arises because the request with the largest relevant set (5 items) scored perfectly while the request with a single relevant item scored zero.

**INTERPRETATION:** Macro answers "how often does a user get good retrieval" and micro answers "how much of the labelled mass was recovered". A product decision usually cares about the first; a capacity or coverage decision usually cares about the second. Reporting both is cheap and stops the argument about which one is "the" recall.

**Validity condition:** `Q+` nonempty and constructed identically on both sides; requests excluded because `|R| = 0` are counted and reported, not silently dropped, since a system change that removes labels changes the estimand.

### 4.3 Variants

Three variants matter in practice: fidelity of an approximate index, fusion of several sources, and the over-retrieval multiplier needed to survive a filter. Each has its own formula and its own failure mode.

#### 4.3.1 Approximate-index fidelity

**[F3] FORMULA:** `ann_fidelity_K = |A_K ∩ X_K| / |X_K|`

**WHERE:** `X_K` is the exact top-K under a stated score (inner product, cosine, or Euclidean) over the *eligible* universe with a deterministic tie rule; `A_K` is the approximate index output over the same universe, the same query vector and the same metric. Both are ordered lists of distinct IDs.

**CONSTRAINT:** `|X_K| > 0` and both lists are computed on the identical eligibility mask. If the approximate path filters after search, its list can be shorter than K, and the comparison must use the actual lengths or a fixed-K oracle that also filtered after search.

**NUMERICAL EXAMPLE:** `X_5 = {i11, i04, i27, i09, i33}` and `A_5 = {i11, i27, i02, i09, i33}`. Overlap is `{i11, i27, i09, i33}`, so the numerator is 4 and the denominator 5, giving 0.80. Scored positionally, the approximate list moved i27 from position 3 to position 2 and inserted i02 at position 3, so a rank-sensitive variant would show a smaller number than 0.80.

**INTERPRETATION:** Four of the five exact-score neighbours were recovered. This says nothing about relevance: if those five neighbours were all irrelevant, perfect fidelity still yields a zero-relevance candidate list. Fidelity is a property of the index, not of the product.

**Validity condition:** Same vectors, same metric, same eligibility mask, same tie rule, and the same definition of "eligible universe" on both sides.

#### 4.3.2 Rank fusion across heterogeneous sources

**[F4] FORMULA:** `rrf(item) = Σ_{s∈S} w_s / (c + rank_s(item))`, with items missing from source s contributing a predefined constant rather than zero.

**WHERE:** `S` is the set of retrieval sources; `rank_s` is the 1-based position of the item in source s's filtered list; `c` is a smoothing constant (commonly 60 in published rank-fusion work, kept as a hyperparameter here); `w_s ≥ 0` are source weights that sum to 1 if you want the output on a comparable scale.

**CONSTRAINT:** Ranks, not raw scores, are the input, because source scores from a dot product, a co-occurrence count and a heuristic are not on a common scale. Every source list must already be filtered to `E(q,t)` before ranking, otherwise an ineligible top-ranked item consumes fusion weight and displaces an eligible one.

**NUMERICAL EXAMPLE:** Two sources with `w = 0.6, 0.4`, `c = 60`. Item i09 is rank 1 in the collaborative source and rank 8 in the content source: `0.6/61 + 0.4/68 = 0.009836 + 0.005882 = 0.015718`. Item i14 is rank 4 in collaborative and rank 1 in content: `0.6/64 + 0.4/61 = 0.009375 + 0.006557 = 0.015932`. So i14 edges out i09 by 0.000214 despite i09 holding the top slot in the heavier-weighted source, because i14 appears near the top of both.

**INTERPRETATION:** Fusion rewards agreement across sources. The near-tie is also a warning: differences at the fourth decimal place are not a basis for claiming one item should outrank another, and the smoothing constant moves these numbers more than the source weights do. At `c = 10` the same ranks give `0.6/11 + 0.4/18 = 0.05455 + 0.02222 = 0.07677` versus `0.6/14 + 0.4/11 = 0.04286 + 0.03636 = 0.07922`, preserving the order but widening the gap to 0.00245 — an eleven-fold amplification of the same comparison.

**Validity condition:** Ranks are computed after eligibility filtering, the source set is fixed, and the smoothing constant is frozen across the compared configurations.

#### 4.3.3 Over-retrieval multiplier for a selective filter

**[F5] FORMULA:** Let `f` be the per-item probability of surviving the filter, independent across items. The number of survivors in a requested list of size n is `X ~ Binomial(n, f)`. The shortfall probability is `P(X < k) = Σ_{x=0}^{k-1} C(n,x) f^x (1-f)^{n-x}`, and the required n for a target shortfall probability α is the smallest n with `P(X < k) ≤ α`.

**WHERE:** `k` is the number of post-filter candidates the ranker needs; `n` is the over-retrieved count requested from the index; `f` is measured from logs as the empirical pass rate for the same filter, segment and catalog snapshot.

**CONSTRAINT:** Independence is the load-bearing assumption. When the filter is correlated with the retrieval score — for example a rights filter that removes exactly the items the popularity source favours — the binomial model understates shortfall badly, and the multiplier must be calibrated from the observed joint distribution rather than from a marginal pass rate.

**NUMERICAL EXAMPLE (ILLUSTRATIVE):** `k = 100`, `f = 0.13`, `α = 0.05`. Solving numerically, `P(X < 100)` falls below 0.05 at `n = 1593`, a multiplier of 15.93. At the naive `n = k/f = 769`, the expected count is exactly 100, so the shortfall probability is about 0.5 by symmetry — half of all requests would underfill. The same computation at `f = 0.30` gives `n = 655` (multiplier 6.55), and at `f = 0.05` gives `n = 4234` (multiplier 42.34).

**INTERPRETATION:** The naive ratio `1/f` is the wrong answer whenever a shortfall is what you are trying to avoid. The correct multiplier grows as f falls and shrinks slowly as k grows: at `f = 0.13`, k = 20 needs a multiplier of 32.45 while k = 500 needs 10.84. Small budgets are the ones that need the most over-retrieval.

**Validity condition:** Filter decisions are approximately independent across items given the request, `f` is measured on the same segment, and the index can actually return n items without truncating internally.

#### 4.3.4 Slate metrics under eligibility masking

**[F6] FORMULA:** `nDCG_n = DCG_n / IDCG_n` with `DCG_n = Σ_{p=1}^{n} g(item_p) / log2(p+1)` and `IDCG_n` the same sum over the best achievable ordering. Under a slate constraint such as a per-category cap, the ideal must be computed under the same cap.

**WHERE:** `g` is a graded gain (0 for irrelevant, positive for relevant or for an interaction count); `p` is the 1-based display position; the discount `1/log2(p+1)` makes position 1 worth 1.0 and position 2 worth 0.6309.

**CONSTRAINT:** The ideal ordering must be drawn from the same eligible set, and a constrained system must be compared with a constrained ideal. Comparing a capped slate against an unconstrained ideal charges the system for a constraint the product imposed deliberately, which is a comparison error rather than a quality signal.

**NUMERICAL EXAMPLE:** Grades for the eligible relevant items are two at 3, two at 2, and one at 1. Unconstrained top-5 has grades `[3,3,2,2,1]`: `DCG = 3/1 + 3/1.585 + 2/2 + 2/2.322 + 1/2.585 = 3 + 1.8928 + 1 + 0.8614 + 0.3869 = 7.1411`, and `IDCG` equals the same value, so `nDCG_5 = 1.000`. Apply a two-per-category cap that forces the fifth slot to an irrelevant filler: the list becomes grades `[3,3,2,1,0]`, `DCG = 3 + 1.8928 + 1 + 0.4307 + 0 = 6.3235`, `nDCG_5 = 0.8855`. Scoring against the unconstrained ideal charges 0.1145 of loss to the cap; scoring against the capped ideal, whose best achievable order is `[3,3,2,1,0]` at 6.3235, returns 1.000 and correctly reports that the ranker did as well as the constraint allows.

**INTERPRETATION:** The two denominators answer different questions. Against the unconstrained ideal you learn the price of the constraint; against the capped ideal you learn whether the ranker exploited what the constraint left available. Publish the first when arguing about product rules and the second when arguing about model quality.

**Validity condition:** Gains are defined once and applied to both lists; the ideal respects the same eligibility set and the same constraints; ties at the budget boundary are resolved by a stated rule.

### 4.4 Limits and counterexamples

The formulas above share an assumption that fails in a specific and common way: they treat `R` as known. In production `R` is itself estimated from logs produced by the current policy, so every recall number measures agreement with a policy-conditioned label, not with user utility.

**Counterexample 1 — the post-filter trap.** An index configured for global top-K returns `{i01, i02, i03}` at K = 3; the eligibility filter then rejects all three. Candidate recall is zero even though the eligible exact top-3 is `{i44, i51, i60}`. The fix is not a larger K but a partitioned index searched within the eligible universe, or pre-filtering with an eligibility-constrained search structure. Increasing K to 30 while keeping post-filtering raises the chance of survival but does not guarantee it; at `f = 0.13` there is still a 5 percent chance of complete emptiness at n = 1593 for k = 100, and complete emptiness for k = 1 has probability `(1-f)^n`.

**Counterexample 2 — the fidelity trap.** An approximate index achieves 0.99 fidelity at K = 500. This is not evidence that the pipeline is healthy: if the relevance rate among exact top-500 is 3 percent, then 0.99 fidelity on a list whose top 100 has a much higher relevance density can still reshuffle the head. Fidelity must be reported per K-depth band (top 10, top 50, top 100, top 500) because the head is what reaches the ranker.

**Counterexample 3 — the ceiling trap.** A team reports display recall of 0.42 and concludes the ranker is bad. If candidate recall at the same K is 0.45, the ranker captured `0.42/0.45 = 0.933` of what it was given, and the dominant loss is at retrieval. The same 0.42 with candidate recall 0.90 implies rank capture of 0.467 and a genuinely weak ranker. One number cannot distinguish these; the pair can.

**Counterexample 4 — the fusion trap.** Fusing a high-precision source and a high-recall source with equal weights lowers the fused list's precision relative to the high-precision source alone. Fusion improves recall and agreement, not precision, and the weights are the control. Reporting only the fused metric hides a precision regression in the source that carried the head of the slate.

**Takeaway:** Each formula is valid inside a stated universe and a stated label rule; the failures above are all violations of those two conditions rather than arithmetic errors.
