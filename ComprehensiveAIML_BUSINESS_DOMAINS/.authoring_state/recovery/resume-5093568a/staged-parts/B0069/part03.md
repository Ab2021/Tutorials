## 5. Worked Numerical Example

**ILLUSTRATIVE and arithmetically checked.** The catalog, the request, the sources and every count below are constructed to exercise the formulas in Section 4 on one request. No production system was measured.

**Decision:** Does a proposed slate constraint of two items per genre cost more quality than a proposed increase in candidate budget recovers?
**Request:** one authenticated user, one surface, decision timestamp t. The entitlement table, the stock table and the index are all read as of t.
**Snapshot:** catalog `I(t)` has 20 items. Fourteen are eligible: three are excluded by territory rights (`i30`), one by stock (`i31`), two by a safety policy (`i32`, `i33`).
**Labels:** seven eligible items satisfy the declared relevance rule inside the matured 14-day horizon: `R = {i2, i9, i5, i3, i14, i6, i10}`. Items without a positive label are treated as unknown, not negative, when computing recall.
**Candidate budget:** K = 10 unique eligible IDs, with ties broken by ascending item ID.

### 5.1 Stage counts for the request

| Stage | Count | Derivation |
|---|---:|---|
| Rows returned by three sources | 28 | 10 popularity + 10 collaborative + 8 content |
| Duplicate rows within and across sources | 11 | 28 rows collapse to 17 unique IDs |
| Unique IDs before eligibility | 17 | measured on the union |
| Unique IDs rejected by eligibility | 3 | `i30` rights, `i31` stock, `i32` safety |
| Unique eligible IDs | 14 | 17 − 3 |
| Relevant IDs among eligible unique IDs | 6 | `i2, i9, i5, i3, i14, i6`; `i10` never retrieved |
| Relevant IDs retained after truncation to K = 10 | 5 | `i6` sits at position 11 and is cut |
| Relevant IDs in the capped slate of 5 | 4 | `i3` displaced by the two-per-genre cap |

### 5.2 The three recall numbers, computed in order

```text
FORMULA: source_recall = |unique_eligible ∩ R| / |R|
WHERE:   unique_eligible = 14 IDs; R = 7 eligible relevant IDs
CONSTRAINT: the union is not truncated, isolating admission loss from budget loss
NUMERICAL EXAMPLE: intersection = {i2,i9,i5,i3,i14,i6}, |∩| = 6
                   6 / 7 = 0.8571
INTERPRETATION: retrieval missed i10 entirely; that loss is unrecoverable downstream
VALIDITY: R is nonempty and defined on the same eligibility snapshot
```

```text
FORMULA: cand_recall_K = |C_K ∩ R| / |R|, K = 10
WHERE:   C_10 = [i2,i9,i5,i3,i14,i1,i8,i4,i7,i12] in fusion order
CONSTRAINT: truncation applies after eligibility filtering and deduplication
NUMERICAL EXAMPLE: intersection = {i2,i9,i5,i3,i14}, |∩| = 5
                   5 / 7 = 0.7143
INTERPRETATION: truncation cost 1 of 7 relevant items, i.e. 0.8571 − 0.7143 = 0.1429
VALIDITY: the same K is applied to every request in any aggregate this feeds
```

```text
FORMULA: display_recall = |S_n ∩ R| / |R| with n = 5
WHERE:   capped slate S_5 = [i2,i9,i5,i14,i1]
CONSTRAINT: the slate is drawn only from C_K and respects the two-per-genre cap
NUMERICAL EXAMPLE: intersection = {i2,i9,i5,i14}, |∩| = 4
                   4 / 7 = 0.5714; rank capture = (4/7)/(5/7) = 4/5 = 0.8000
INTERPRETATION: the ranker-plus-cap stage lost 0.1429 of the population, the same
                magnitude as the truncation loss, so both stages matter equally here
VALIDITY: the denominator stays |R|, not |C_K ∩ R|, so stage losses compose additively
```

The additivity is worth stating explicitly because it is the reason the three numbers are useful: `0.8571 − 0.7143 = 0.1429` (budget) and `0.7143 − 0.5714 = 0.1429` (rank plus cap) sum to `0.2857`, and `1 − 0.5714 = 0.4286` includes the 0.1429 never retrieved at all. Three mutually exclusive loss buckets, each owned by a different team.

### 5.3 Independent re-check of the same result

Recompute from the other direction. Of the seven labelled relevant items, three are absent from the slate: `i3` (retrieved, in `C_10`, displaced by the cap), `i6` (retrieved, cut by the budget), and `i10` (never retrieved). Four are present. `7 − 3 = 4`, so `4/7 = 0.5714`. The two paths agree.

Check the cap arithmetic separately. In fusion order the genres are: `i2` news, `i9` news, `i5` sport, `i3` news, `i14` music, `i1` sport, `i8` sport, `i4` music, `i7` sport, `i12` news. Walking the list with counters: `i2` → news 1, accept. `i9` → news 2, accept. `i5` → sport 1, accept. `i3` → news 3, reject. `i14` → music 1, accept. `i1` → sport 2, accept. Slate full at five. Rejected items: `i3` only, because the slate filled before `i8`, `i4`, `i7` and `i12` were considered.

### 5.4 Ranking quality under the constraint, both denominators

```text
FORMULA: DCG_n = Σ_{p=1..n} g_p / log2(p+1)
WHERE:   grades: i2→3, i9→3, i5→2, i3→2, i14→1, i6→1, i1→0
CONSTRAINT: gains assigned before the constraint is applied; discount fixed
NUMERICAL EXAMPLE: unconstrained [3,3,2,2,1]:
                   3/1 + 3/1.5850 + 2/2.0000 + 2/2.3219 + 1/2.5850
                 = 3.0000 + 1.8928 + 1.0000 + 0.8614 + 0.3869 = 7.1411
                   capped [3,3,2,1,0]:
                   3.0000 + 1.8928 + 1.0000 + 0.4307 + 0 = 6.3235
INTERPRETATION: the cap costs 0.8176 DCG points on this request
VALIDITY: both lists drawn from the same eligible set and the same grade function
```

Against the unconstrained ideal, `nDCG = 6.3235/7.1411 = 0.8855`. Against the capped ideal, whose best achievable order is exactly `[3,3,2,1,0]` because the cap forbids a fifth relevant item, `nDCG = 6.3235/6.3235 = 1.0000`. The difference between 0.8855 and 1.0000 is not a modelling improvement; it is a change of question.

### 5.5 Filling the budget: how many rows must be requested

The three sources returned 17 unique IDs to yield 14 eligible ones, a measured pass rate of `14/17 = 0.8235` for this request. If the service needs 10 eligible candidates with 95 percent reliability, the binomial over-retrieval rule gives:

```text
FORMULA: n* = min{ n : P(Binomial(n, f) ≥ k) ≥ 1 − α }
WHERE:   f = 0.8235 (measured), k = 10, α = 0.05
CONSTRAINT: filter outcomes independent given the request
NUMERICAL EXAMPLE: P(X ≥ 10 | n=10) = 0.1435; (n=12) = 0.6424; (n=13) = 0.8159
                   first n with P ≥ 0.95 is n = 46, multiplier 4.60
INTERPRETATION: a flat 2x over-retrieval fails 36 percent of the time at these settings
VALIDITY: independence; correlated rejection raises n* well above 46
```

The independence assumption is false for `i30`–`i33`: rights, stock and safety exclusions are concentrated in specific genres and publishers, so the rejection events cluster. A practical calibration is to compute the empirical 5th percentile of survivors per requested-n bucket from logs and choose the smallest bucket whose 5th percentile reaches k, which for clustered rejections typically lands well above the binomial answer.

### 5.6 Sensitivity: what moves the conclusion

| Lever | Change | Effect on the numbers | Effect on the decision |
|---|---|---|---|
| Budget K | 10 → 14 | Candidate recall 0.7143 → 0.8571; `i6` recovered | Removes half the total loss for 40 percent more ranking work |
| Cap | 2/genre → 3/genre | Slate would admit `i3`; display recall 0.5714 → 0.7143 | Cap relaxation and budget increase are substitutes here |
| Label rule | click → 30-second completion | `i14` likely drops from relevant; recall falls | Changes the denominator; invalidates cross-rule comparisons |
| Horizon H | 14 days → 3 days | Fewer mature labels, so `R` shrinks and estimates get noisier | Narrower horizon raises variance more than it changes the mean |
| Pass rate f | 0.8235 → 0.30 | n* rises from 46 to 655 for k = 100 at α = 0.05 | Over-retrieval, not ranking, becomes the dominant cost |

**Sensitivity conclusion:** on this request, relaxing the cap and enlarging the budget are interchangeable at the margin, so the choice should be made on product grounds (catalog exposure policy) rather than on the metric. Changing the label rule, by contrast, is not interchangeable with anything: it redefines the target.

### 5.7 What this example does not show

Three things are deliberately absent. It does not show statistical significance, because a single request has no variance estimate. It does not show business impact, because a relevance label is not a revenue or retention outcome, and the mapping between them requires an experiment. It does not show that the cap is harmful, because a cap exists to change the exposure distribution across genres, and that distribution is not visible in a per-request metric at all. A per-genre exposure report over a week of requests is the correct instrument for the cap question.
