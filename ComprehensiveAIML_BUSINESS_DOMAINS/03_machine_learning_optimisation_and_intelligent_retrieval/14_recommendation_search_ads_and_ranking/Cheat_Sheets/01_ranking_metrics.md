---
title: "Ranking metrics and candidate generation"
topic: "Ranking metrics, candidate generation, and offline-to-online validity"
collection: "14_recommendation_search_ads_and_ranking"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Ranking metrics and candidate generation

All numbers in this sheet are constructed teaching values. The arithmetic is recomputed
line by line below, but no production system was measured and no company result is claimed.
Every figure that depends on an assumed catalogue, query volume, or judgment set is marked
`ILLUSTRATIVE` at the point where it first enters an equation.

## 1. File Header

**Purpose.** Give a practitioner the arithmetic, the selection rules, and the failure
surface for the metrics used to accept or reject a ranking change, plus the candidate
generation step that silently caps every one of those metrics.

**Decision this sheet supports.** Given one query population, one candidate generator,
and one ranking change, decide (a) which offline metric is a fair proxy, (b) what the
achievable ceiling is, and (c) whether the observed difference is larger than judgment
and sampling noise.

**Audience.** Search and recommendation engineers, applied scientists, and analysts who
must defend a launch decision in a review where someone will ask "what is the denominator".

**Prerequisites.** Comfort with logarithms base 2, set notation for recall, the idea of
an expectation, and the discipline of writing units next to every quantity. No deep
learning is required; the hard parts here are bookkeeping and attribution.

**Outcomes.** After working through the twelve sections a reader should be able to:
derive nDCG from the value-of-position argument; compute DCG, IDCG, nDCG@k, AP, and MRR
by hand for a graded run; compute the ceiling imposed by a candidate set; size an
offline comparison experiment; and name the specific check that distinguishes a real
regression from a judgment-pool artefact.

**Notation convention.** All display mathematics appears in fenced `text` blocks with
plain ASCII operators. Symbols are defined at first use as meaning, unit, and range.
Probabilities are dimensionless in [0,1]; gains are unitless integer grades; positions
are 1-based integers; time is milliseconds unless a table states seconds.

**Evidence boundary.** Formulas in section 4 are standard results restated with their
derivations. Section 5 arithmetic is verified by a second, independent decomposition.
Cost figures in section 9 are operation counts plus one illustrative latency budget;
they describe orders of magnitude, not a benchmark of any named system.

**Exclusions.** This sheet does not cover learning-to-rank loss functions in detail, does
not cover auction mechanics or pricing, and does not attempt to settle whether a given
product should optimise engagement or satisfaction. Those belong to sibling sheets and to
the case studies that cite this one.

**How to use it.** Read section 2 once for the mental model, keep section 7 open during a
metric argument, and use section 8 as a checklist when offline and online disagree. The
worked example in section 5 is deliberately small enough to redo on paper.

## 2. Concept Map / Mental Model

A ranking system is a funnel of shrinking sets, and each stage has its own kind of error.
The metric you quote measures one stage, but the business outcome is produced by all of
them multiplied together. Most metric disputes are really disagreements about which
stage was measured.

```text
catalogue (N items)
   |  candidate generation: lexical, ANN, co-visitation, trending
   v
candidate set C_q  (|C_q| = 300..2000, ILLUSTRATIVE)
   |  ranking model: pointwise / pairwise / listwise score
   v
ordered list L_q  (top-k presented, k = 10..50)
   |  presentation: truncation, interleaving of ads, UI density
   v
observed clicks and conversions (biased by position and trust)
```

Three separate ceilings exist, and confusing them is the most common analytical error:

1. **Candidate ceiling.** If a relevant item is not in `C_q`, no ranker can rank it and
   no offline metric computed on `C_q` can see the loss. The metric is then blind, not low.
2. **Ranker ceiling.** Given `C_q`, the best achievable ordering is the ideal ordering of
   the items actually present. nDCG divided by that ideal is normalised by construction,
   which hides candidate-stage damage.
3. **Presentation ceiling.** Given a list, only the visible slots can be clicked. A metric
   computed over 10 slots says nothing about slot 11, where the tail of demand lives.

The mental model that keeps these separate is a two-number contract. Every offline
evaluation reports a *rank quality* number computed on the candidate set, and a
*candidate recall* number computed against the judged pool. A launch argument that quotes
only the first number is incomplete by construction.

```text
rank quality  : nDCG@k over C_q          -> "how well did we order what we had"
candidate recall: |relevant ∩ C_q| / |relevant| -> "how much did we fail to consider"
```

A second mental model is the difference between a metric and an estimator. nDCG is a
function of a labelled list; the number you compute is a *sample* of that function over a
query population. The estimator has a denominator (number of queries), a variance, and a
bias from the judgment pool. Two systems can differ by 0.004 nDCG while neither is
distinguishable from the other, because the standard error over a small query sample is
larger than the effect.

A third model is *what the metric rewards under ties*. Graded metrics consume a total
ordering; when the ranker emits nearly equal scores, the realised order is decided by an
arbitrary tie-break. If a metric moves when only the tie-break changes, the metric is
measuring noise, and the fix is either score calibration or reporting a tie-aware band.

A fourth model is the attribution chain from offline gain to business gain. Offline nDCG
moves when the ranker orders better; revenue moves when the *clicked* items change. The
link between them is the click model, which is position-biased. So an offline lift is a
claim about ordering, and a business lift is a claim about behaviour; only an experiment
tests the second.

Finally, treat candidate generation as a retrieval problem with its own metric. Recall at
a fixed budget, measured against a pool, is the honest number. Recall is monotone in the
budget, so quoting recall without the budget is not a claim at all.

## 3. Foundational Definitions

**Query (q).** One user request with a timestamp, a surface, and a locale. The unit of
evaluation is usually the query, not the session; mixing the two changes the denominator.

**Impression.** One rendered slot for one item for one query. A query with 10 slots
produces 10 impressions. Click-through rate divides clicks by impressions, not by queries.

**Judged pool.** The finite set of (query, item) pairs a human or a model has labelled.
Pooling is depth-limited in practice, so unjudged items are not necessarily irrelevant;
this single fact drives most offline/online disagreements.

**Relevance grade (rel).** An ordinal label on a (query, item) pair, typically 0 to 4.
Zero means not relevant, not "unjudged". Grades are ordinal, so averaging grades is a
modelling choice, not an arithmetic truth.

**Candidate set (C_q).** The items the ranker is allowed to score for query q, produced by
one or more generators. Its size is a budget in items, not a quality measure.

**Candidate recall.** The fraction of relevant items for q that appear in C_q. It is a
property of the generators, and it is the ceiling on every downstream metric computed
inside C_q.

**Position (p).** The 1-based slot index in the presented list. Position 1 is the first
slot the user sees; discounting is a function of p, not of the item.

**DCG@k.** Discounted cumulative gain over the first k positions: a sum of per-item gains
each divided by a position discount. Units are gain units, so DCG is comparable only
across runs on the same judgment scale.

**IDCG@k.** The DCG of the best possible ordering of the judged items available to that
query, truncated at k. It is the normaliser, and it must be computed with the same gain
function and the same k as the numerator.

**nDCG@k.** DCG@k divided by IDCG@k, therefore dimensionless in [0,1]. The value 1 means
"as good as the ideal ordering of the available items", not "perfect search".

**Precision@k.** Relevant items among the first k divided by k. Its denominator is fixed
by the cut, so it falls mechanically as k grows past the number of relevant items.

**Recall@k.** Relevant items among the first k divided by the total relevant for the
query. Its denominator is the judgment set, so it is the metric that exposes a shallow
candidate pool.

**Average precision (AP).** The mean of precision@k evaluated at each position where a
relevant item appears; MAP is AP averaged over queries. AP weights early hits heavily
because precision is measured on a growing prefix.

**Reciprocal rank (RR).** One over the position of the first relevant item, zero if none
appears. Mean reciprocal rank (MRR) averages RR over queries and is the right metric when
the user wants exactly one good answer.

**Expected reciprocal rank (ERR).** A cascade metric in which each position's contribution
is the probability the user stops there times the reciprocal position. It encodes
"users stop when satisfied", which nDCG does not.

**Score-to-rank tie band.** The set of items whose scores are close enough that the model
cannot reliably order them. Reporting the metric range produced by arbitrary tie-breaks
inside the band is more honest than a single decimal.

**ANN recall.** For an approximate nearest-neighbour index, the fraction of the exact
top-k neighbours that the approximate search actually returned. It is measured against a
brute-force reference, not against human judgments.

**Inverse propensity weight (IPS).** One over the probability that an item was examined at
its logged position, used to reweight logged clicks into an unbiased relevance estimate.

**Guardrail metric.** A metric that must not degrade even if the primary metric improves:
latency, zero-result rate, revenue per query, complaint rate. Guardrails are evaluated
with the same rigour as the primary.

**Offline-online gap.** The systematic difference between the sign or size of a change
measured offline and the same change measured in an experiment. It is a property of the
evaluation pipeline, and it is measurable when both numbers exist for the same launches.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

Start from the smallest defensible model: the value of showing an item in slot p decays
with p because attention and patience decay. Write the value of a list as the sum of
per-item values, where an item's value depends only on its own grade and its slot.

```text
FORMULA (base case, binary relevance)
  DCG@k = sum over p = 1..k of  rel_p / log2(p + 1)
WHERE
  rel_p in {0,1} is the relevance of the item placed at position p
  k is the cut, an integer >= 1
  DCG has units of "relevance-weighted slots"
CONSTRAINT
  p is 1-based; log2(p+1) >= 1 for all p, so no division by zero
  gains must use one fixed judgment scale across compared systems
NUMERICAL EXAMPLE (k = 5)
  ranks: 1, 0, 1, 0, 0
  discounts: 1.0000, 0.63093, 0.50000, 0.43068, 0.38685
  DCG@5 = 1.0000 + 0 + 0.5000 + 0 + 0 = 1.5000
INTERPRETATION
  two relevant items at slots 1 and 3 are worth 1.5 "ideal-slot" units;
  the same two items at slots 4 and 5 would be worth 0.43068 + 0.38685 = 0.81753
```

The discount `1/log2(p+1)` is chosen, not derived from behaviour. Its properties are what
matter: it equals 1 at p = 1, so the top slot keeps full value; it is strictly decreasing,
so position always matters; and it is convex-decreasing, so the penalty for moving from
slot 1 to slot 2 (0.369) is much larger than from slot 7 to slot 8 (0.018). Any decreasing
convex curve with these endpoints would preserve the ordering of most systems; the exact
base only becomes decisive when two runs differ mainly in the tail.

Normalising the base case gives the dimensionless version. If a query has `m` relevant
items and the cut is k, the ideal list puts them first, so IDCG is the DCG of that ideal.

```text
FORMULA (nDCG, binary)
  IDCG@k = sum over p = 1..min(k, m) of  1 / log2(p + 1)
  nDCG@k = DCG@k / IDCG@k
WHERE
  m = number of relevant judged items available to the query
  nDCG is dimensionless and lies in [0,1]
CONSTRAINT
  if m = 0 then IDCG@k = 0 and nDCG is undefined; exclude the query or define 0/0 := 1
NUMERICAL EXAMPLE (k = 5, m = 2, run above)
  IDCG@5 = 1.0000 + 0.63093 = 1.63093
  nDCG@5 = 1.5000 / 1.63093 = 0.91972
INTERPRETATION
  the run captured 92% of the ideal discounted value available at five slots
```

### 4.2 General case

Real judgments are graded, the cut is chosen by the product, and the candidate set is
finite. Each of these three facts changes the arithmetic, and they must be changed
together or the numbers stop being comparable.

Graded gain with an exponential transform is the common convention. It makes grade 3
worth seven times grade 1 rather than three times, encoding diminishing returns in the
label scale rather than assuming linearity.

```text
FORMULA (graded DCG, exponential gain)
  gain(rel) = 2^rel - 1
  DCG@k = sum over p = 1..k of  (2^rel_p - 1) / log2(p + 1)
  IDCG@k = sum over p = 1..min(k, |R_q|) of (2^rel_(p) - 1) / log2(p + 1)
  where rel_(p) is the p-th largest grade among judged items for q
CONSTRAINT
  rel >= 0 integer; with rel capped at 4 the maximum single gain is 15
  the same k and the same gain transform must be used in numerator and denominator
NUMERICAL EXAMPLE (grades 3, 0, 2, 0, 1 at slots 1..5)
  gains: 7, 0, 3, 0, 1
  DCG@5 = 7*1.0000 + 0 + 3*0.5000 + 0 + 1*0.38685 = 8.88685
  sorted grades for the query: 3, 2, 2, 1, 1 -> gains 7, 3, 3, 1, 1
  IDCG@5 = 7*1.0000 + 3*0.63093 + 3*0.5000 + 1*0.43068 + 1*0.38685 = 11.21032
  nDCG@5 = 8.88685 / 11.21032 = 0.79274
INTERPRETATION
  the run loses most of its value at slot 2, where a zero-gain item sits
```

Average precision is the graded-free alternative, and it answers a different question:
"how much precision did we have at the moments a relevant item appeared". It is the
natural metric when the label is binary and the user may want many items.

```text
FORMULA (AP and MAP)
  AP@K = (1/|R_q|) * sum over p = 1..K of  [ rel_p * P@(p) ]
  P@(p) = (number of relevant items in positions 1..p) / p
  MAP = (1/|Q|) * sum over q in Q of AP_q
WHERE
  |R_q| is the total relevant for q (including items beyond K if labelled)
  |Q| is the number of evaluated queries
CONSTRAINT
  relevance binary; AP@K with |R_q| counted beyond K is NOT bounded by 1
  if all relevant items are inside K then AP@K in [0,1]
NUMERICAL EXAMPLE (run A, B, C at slots 1, 3, 5 of five relevant)
  p=1: rel, P@1 = 1/1 = 1.0000 -> contribution 1.0000
  p=3: rel, P@3 = 2/3 = 0.6667 -> contribution 0.6667
  p=5: rel, P@5 = 3/5 = 0.6000 -> contribution 0.6000
  AP = (1.0000 + 0.6667 + 0.6000) / 5 = 2.26667 / 5 = 0.45333
INTERPRETATION
  AP punishes the empty slots 2 and 4 because precision is measured on the prefix
```

When the user wants one answer, reciprocal rank replaces the sum entirely. It is cheap,
robust to pool depth beyond the first hit, and extremely high-variance per query.

```text
FORMULA (RR and MRR)
  RR_q = 1 / p_first    if any relevant item is retrieved, else 0
  MRR = (1/|Q|) * sum over q of RR_q
WHERE
  p_first is the rank of the first relevant item, 1-based
CONSTRAINT
  p_first must be inside the evaluated depth, otherwise RR is 0 by convention
NUMERICAL EXAMPLE (first relevant at slot 4)
  RR = 1/4 = 0.2500
  if 30% of queries have no hit at all, MRR <= 0.70 regardless of ordering quality
INTERPRETATION
  MRR is a statement about the first hit only; it cannot see anything after it
```

Candidate generation gets a formula too, and it is the one that bounds the rest. Recall
at a budget is a set intersection over the judged relevant set, so it requires a pool that
is not itself produced by the generator being measured.

```text
FORMULA (candidate recall at budget B)
  CRecall@B = (1/|Q|) * sum over q of  |Rel_q ∩ C_q(B)| / |Rel_q|
  with |C_q(B)| <= B for every query
WHERE
  Rel_q is the judged relevant set for q, from an independent pool
  C_q(B) is the candidate set the generator produced within budget B
CONSTRAINT
  pooling must be independent of the generator under test, else recall is inflated
  queries with |Rel_q| = 0 are excluded and reported as a count
NUMERICAL EXAMPLE (ILLUSTRATIVE: 60 relevant judged, 40 retrieved in 500 slots)
  CRecall@500 = 40/60 = 0.6667
  if a second page raises retrieval to 52 of 60: CRecall@1000 = 52/60 = 0.8667
INTERPRETATION
  the generator's headroom at B = 500 is 33 percentage points of recall
```

Finally, logged clicks are not relevance. The inverse-propensity estimator converts an
observed click into a relevance estimate by dividing by the examination probability of
the slot where the click happened.

```text
FORMULA (IPS relevance estimate)
  rel_hat(i) = click_i / o_(p_i)
  E[rel_hat(i)] = rel(i)   provided o_(p) > 0 for every position that can be logged
WHERE
  click_i in {0,1}; o_(p) in (0,1] is the examination propensity of position p
CONSTRAINT
  positivity: positions with o_(p) = 0 contribute nothing and cannot be corrected
  variance grows as 1/o^2; a slot with o = 0.02 has 2500x the variance of a slot at o = 1
NUMERICAL EXAMPLE
  click at slot 1 with o = 1.00 -> weight 1.00
  click at slot 6 with o = 0.20 -> weight 5.00
  two clicks, one each -> mean estimate (1.00 + 5.00)/2 = 3.00 on an unbounded scale
INTERPRETATION
  the estimate is unbiased but heavy-tailed; clip only with a stated bias budget
```

### 4.3 Variants

Four variants cover most production arguments. Choose by the question, and record the
choice next to the number, because two variants of "nDCG" can differ by more than the
effect being measured.

**Linear gain.** Replace `2^rel - 1` with `rel`. Linear gain is appropriate when the label
is already a magnitude (revenue, dwell seconds) rather than an ordinal grade. It compresses
the top of the scale: with linear gain, one grade-4 item equals four grade-1 items, while
exponential gain makes it fifteen. A system that concentrates the best item at slot 1 will
look better under exponential gain, so switching transforms between comparison runs is a
way to manufacture a win.

**Truncated nDCG at the presentation depth.** If the UI shows 10 slots, report nDCG@10 and
nDCG@50 together. The first describes the user-visible experience; the second describes
how much good material the ranker pushed just below the fold, which is what a UI-density
experiment would unlock. Reporting only @50 flatters a ranker that buries quality at slot 12.

**ERR and cascade metrics.** ERR multiplies the reciprocal rank by the probability the user
stops at that position, with the stop probability derived from the grade. It behaves like
nDCG for the head and like MRR for the tail, and it is the better proxy when a single
satisfying answer ends the session.

```text
FORMULA (ERR)
  ERR = sum over p = 1..k of  (1/p) * R_(p) * product over j < p of (1 - R_(j))
  R_(p) = (2^rel_p - 1) / 2^rel_max
WHERE
  rel_max is the maximum grade on the scale (e.g. 4)
CONSTRAINT
  R in [0,1]; grades must be mapped consistently or cascade probabilities drift
NUMERICAL EXAMPLE (grades 2, 4, 0 with rel_max = 4)
  R_1 = 3/16 = 0.1875; R_2 = 15/16 = 0.9375; R_3 = 0
  ERR = 1*0.1875 + (1/2)*0.9375*(1-0.1875) + 0 = 0.1875 + 0.38086 = 0.56836
INTERPRETATION
  a perfect item at slot 2 is still worth less than half its face value
```

**Rank-biased overlap (RBO) for set-like surfaces.** When the output is a ranked set rather
than a list with judged labels (related searches, tag suggestions), overlap with a reference
ranking can be measured without relevance labels. RBO sums the overlap at each depth with a
geometric persistence parameter, so it tolerates incomparable tails and does not require a
pool.

**Alpha-nDCG for novelty.** When duplicate or near-duplicate items are common, subtract a
penalty for items already covered by earlier results. The penalty is a design choice, and
the only defensible way to set it is against the duplicate rate observed in the pool.

### 4.4 Limits and counterexamples

Each formula above fails in a specific, reproducible way. Knowing the failure is more
useful than knowing the formula, because the failure is what shows up in a review.

**Counterexample 1 — nDCG rewards a shorter relevant set.** Take k = 5 and two queries
with identical DCG@5 = 1.5000. Query A has two relevant items, so IDCG@5 = 1.63093 and
nDCG = 0.91972. Query B has five relevant items, so IDCG@5 = 1 + 0.63093 + 0.5 + 0.43068
+ 0.38685 = 2.94846 and nDCG = 0.50875. The same absolute quality produces a score 40
points apart because the normaliser differs. Consequence: nDCG is comparable across
systems on the same query population, and misleading across populations with different
label density.

**Counterexample 2 — an unjudged item is scored as wrong.** If the pool judged 8 items for
a query and the ranker surfaces a ninth that is excellent but unjudged, its gain enters the
numerator as 0. A system with better coverage of the long tail is therefore penalised by
pool construction. The discriminating check is the unjudged rate in the top k; above a
threshold, the comparison is a measurement of pool coverage.

**Counterexample 3 — cut-insensitive metrics hide the second page.** nDCG@10 of two systems
can be identical while one puts five grade-3 items at slots 11 to 15. If the product is
considering denser UI, only nDCG@50 or a per-depth curve reveals the difference. Any
argument about "the ranker is equally good" that cites one cut is under-specified.

**Counterexample 4 — MRR with no-hit queries.** Suppose 40% of queries have no relevant
item inside depth 10. Then MRR <= 0.60 no matter how good the ordering is, and the metric
is dominated by candidate recall rather than rank quality. A 0.02 MRR gain in that regime
is evidence about the candidate generator, not the ranker.

**Counterexample 5 — averaging ratios.** nDCG is a ratio per query, and the mean of ratios
is not the ratio of means. If you compute DCG and IDCG in aggregate and then divide, you
weight long lists more heavily than short ones and you can reverse the sign of a comparison.
Always average per-query values, and state the query-level denominator.

**Counterexample 6 — gain transform switched mid-experiment.** Doubling the gain exponent
changes both numerator and denominator, so nDCG stays in [0,1] and looks plausible while
shifting by several points. Version the judgment set and the transform alongside the metric
name; a number without that triple is not a measurement.

**Validity condition for the whole section.** Every metric above compares two runs only if
they share the query population, the judgment pool version, the label scale, the gain
transform, and the cut. Break any one and the difference is uninterpretable, regardless of
how many decimals are reported.

## 5. Worked Numerical Example

The example uses one query with eight candidate slots and a five-item judgment. Grades are
on a 0 to 3 scale. Items marked X, Y, Z, W are judged irrelevant (grade 0). The run is
`A, X, B, Y, C, Z, D, W` with grades `3, 0, 2, 0, 1, 0, 2, 0`. All inputs are ILLUSTRATIVE;
the arithmetic is checked twice below.

**Step 1 — discounts.** For p = 1 to 8, compute `1/log2(p+1)`.

| p | item | grade | gain = 2^g - 1 | discount | contribution |
|---|---|---|---|---|---|
| 1 | A | 3 | 7 | 1.00000 | 7.00000 |
| 2 | X | 0 | 0 | 0.63093 | 0.00000 |
| 3 | B | 2 | 3 | 0.50000 | 1.50000 |
| 4 | Y | 0 | 0 | 0.43068 | 0.00000 |
| 5 | C | 1 | 1 | 0.38685 | 0.38685 |
| 6 | Z | 0 | 0 | 0.35621 | 0.00000 |
| 7 | D | 2 | 3 | 0.33333 | 1.00000 |
| 8 | W | 0 | 0 | 0.31546 | 0.00000 |

**Step 2 — DCG at two cuts.** DCG@5 sums rows 1 to 5: 7.00000 + 1.50000 + 0.38685 =
8.88685. DCG@8 adds row 7: 8.88685 + 1.00000 = 9.88685.

**Step 3 — the ideal ordering.** The judged grades available are 3, 2, 2, 1, 1. Sorting
descending gives 3, 2, 2, 1, 1, then zeros fill slots 6 to 8. IDCG@5 = 7*1.00000 +
3*0.63093 + 3*0.50000 + 1*0.43068 + 1*0.38685 = 7.00000 + 1.89279 + 1.50000 + 0.43068 +
0.38685 = 11.21032. IDCG@8 equals IDCG@5 because all five relevant items already fit inside
five slots and the remaining three slots carry zero gain.

**Step 4 — normalise.** nDCG@5 = 8.88685 / 11.21032 = 0.79274. nDCG@8 = 9.88685 /
11.21032 = 0.88194. The eight-slot number is 0.08920 higher purely because a relevant item
at slot 7 is inside the cut.

**Step 5 — independent re-check.** Recompute DCG@8 without positional bookkeeping by
grouping contributions per item: A contributes 7.0 at slot 1; B contributes 3 * 0.5 = 1.5
at slot 3; C contributes 1 * 0.38685 at slot 5; D contributes 3 * 0.33333 = 0.99999 at
slot 7. Sum: 7 + 1.5 + 0.38685 + 0.99999 = 9.88684. The 1e-5 difference against 9.88685 is
rounding in the fourth decimal of the discount, not a modelling difference. Both routes
agree to five significant figures.

**Step 6 — precision and recall at the same cuts.** Relevant items found by slot 3: A, B
= 2. P@3 = 2/3 = 0.66667; R@3 = 2/5 = 0.40000. By slot 5: A, B, C = 3. P@5 = 3/5 =
0.60000; R@5 = 3/5 = 0.60000. By slot 8: A, B, C, D = 4. P@8 = 4/8 = 0.50000; R@8 =
4/5 = 0.80000. Precision falls while recall rises, which is the normal shape of the curve
and not evidence of a regression.

**Step 7 — average precision.** Precision at each relevant position, in order: slot 1 ->
1/1 = 1.00000; slot 3 -> 2/3 = 0.66667; slot 5 -> 3/5 = 0.60000; slot 7 -> 4/7 = 0.57143.
Sum = 2.83810. Divide by the relevant count 5: AP = 0.56762. Dividing by the number of
retrieved relevant items (4) instead would give 0.70952 — a different metric, and the
choice must be stated because the two disagree by 25%.

**Step 8 — sensitivity to one label.** Suppose D's grade was mis-set and should be 1, not
2. The numerator loses (3 - 1) * 0.33333 = 0.66667, giving DCG@8 = 9.22018. The ideal list
becomes 3, 2, 1, 1, 1 with gains 7, 3, 1, 1, 1, so IDCG@5 = 7 + 1.89279 + 0.5 + 0.43068 +
0.38685 = 10.21032. nDCG@8 becomes 9.22018 / 10.21032 = 0.90302, against 0.88194 before.
One label move shifts the score by 0.021 absolute, which is larger than many launch effects.
Therefore a judgment audit of the top grades is a prerequisite for a fine-grained comparison.

**Step 9 — the candidate ceiling in the same run.** Now suppose the pool contains 60
relevant items for this query family but candidate generation returned 500 items containing
only 40 of them, and that the presentation depth is 10 (ILLUSTRATIVE). Candidate recall is
40/60 = 0.66667. No ranker can recover the 20 missing items. If nine of the ten top slots
can be filled with relevant items, the graded ceiling is 4.25450 / 4.54356 = 0.93638 for
nDCG@10 over ten slots with one irrelevant item, so 6.4% of the scale is structurally
unreachable before any modelling question is asked.

**Step 10 — why the ceiling matters numerically.** The gap between the ceiling 0.93638 and
the observed 0.88194 in this toy run is 0.05444, while the gap between the observed value
and 1.0 is 0.11806. Roughly 46% of the apparent headroom (0.05444 / 0.11806 = 0.4611) is
recoverable by ordering alone; the rest requires candidate recall. That ratio is the single
most useful number to attach to an offline report, because it tells the team where to spend
the next sprint.

## 6. Visual / Geometric Intuition

The discount curve is the geometry of the whole topic. Plot `1/log2(p+1)` against p: it
falls steeply from 1.0 to 0.63 across one slot, then flattens, reaching 0.32 at slot 8. The
area under this curve up to k is the maximum possible DCG for a query whose every slot is
relevant, which is why IDCG@k is literally the area of the unit-gain trapezoid.

| depth p | discount | drop from previous slot | cumulative sum to p |
|---|---|---|---|
| 1 | 1.00000 | n/a | 1.00000 |
| 2 | 0.63093 | 0.36907 | 1.63093 |
| 3 | 0.50000 | 0.13093 | 2.13093 |
| 4 | 0.43068 | 0.06932 | 2.56160 |
| 5 | 0.38685 | 0.04383 | 2.94846 |
| 6 | 0.35621 | 0.03064 | 3.30467 |
| 7 | 0.33333 | 0.02288 | 3.63800 |
| 8 | 0.31546 | 0.01787 | 3.95346 |
| 9 | 0.30103 | 0.01443 | 4.25450 |
| 10 | 0.28906 | 0.01197 | 4.54356 |

Read the third column: 70% of the total penalty between slots 1 and 10 happens in the first
two moves. This is why nDCG is dominated by the top three slots and why a ranker that fixes
slot 2 gains more than one that fixes slots 6 through 9. It is also why nDCG is a poor
instrument for measuring a tail improvement; use a deeper cut or a per-depth curve.

The funnel picture makes candidate loss visible as area rather than as a number. Draw the
catalogue as a wide bar, the candidate set as a narrower bar inside it, and the presented
list as a thin bar inside that. Relevant items are dots distributed through the catalogue.

```text
catalogue        [ . . o . . . o . . . . o . . . o . . . . . o . . ]   60 relevant
candidates (500)      [ . o . . o . . o . . . o . . o . . . ]         40 captured
presented (10)             [ o . o . . o . . . . ]                    4 shown
nDCG computed here ------------------------------------^              sees only these
```

The dots that fall outside the second bracket are invisible to every metric computed on the
candidate set, by construction rather than by error. Any offline report that shows only the
bottom bracket is answering "how well did we order 500 items", not "how good is search".

The tie-band picture explains variance. Suppose two items score 0.7312 and 0.7309. The
ranker cannot meaningfully distinguish them, so the two possible orders are
`... 0.7312, 0.7309 ...` and `... 0.7309, 0.7312 ...`. If the first item has grade 3 and the
second grade 0, nDCG@1 is either 1.0 or 0.0 while nDCG@2 is 1.0 in both orders. The metric's
sensitivity to slot assignment is therefore a function of cut proximity, not only of score
quality, and a robustness report should give the min and max over admissible tie-breaks.

```text
score axis (descending)          cut between slot 1 and 2
  0.7312 |<-- item P (grade 3)
  0.7309 |<-- item Q (grade 0)
         ^ swap inside band changes nDCG@1 by 1.00 and nDCG@2 by 0.00
```

A fourth picture is the precision-recall plane. As the cut moves right, the operating point
moves down and right: precision declines, recall climbs. AP is the area-weighted average of
precision at the relevant points, so it is the area under a staircase whose steps are placed
at hits. Two systems with identical P@10 can have APs that differ by 0.1 if one of them
places its hits earlier within the same ten slots.

Finally, picture the estimator rather than the metric. Each query contributes one point;
the reported number is a mean over points with a standard error. With per-query nDCG
standard deviation around 0.30 (ILLUSTRATIVE but typical of graded web judgments), the
standard error at 2,000 queries is 0.30/sqrt(2000) = 0.00671. A difference of 0.004 between
two systems is inside one standard error of a single-arm estimate, so the comparison needs
paired per-query differences, not two independent means.

## 7. Comparison / Selection Table

Metric choice is a design decision with a written justification, not a matter of taste.
The table below maps the decision being made to the metric that actually answers it, and
names the mistake that each wrong choice produces.

| Decision being made | Primary metric | Why it fits | Wrong choice and its symptom |
|---|---|---|---|
| Does ranker v2 order better than v1 on a fixed pool? | nDCG@k, graded | single scalar, graded, cut-matched | AP alone: ignores grade, flattens 4 and 1 |
| Did we recover a navigational query? | MRR | one right answer ends the task | nDCG: dilutes a single hit over ten slots |
| Is the candidate generator starving the ranker? | recall@B vs pool | measures the ceiling directly | nDCG: normalises away the missing items |
| Should we show 10 or 20 slots? | nDCG curve over depth | reads value below the fold | nDCG@10 only: blind to slots 11-20 |
| Did novelty improve? | alpha-nDCG | penalises redundant items | plain nDCG: rewards near-duplicates |
| Did the session get shorter because it succeeded? | ERR, dwell, task success | models stopping behaviour | CTR: rises with clickbait |
| Is the ANN index accurate enough? | ANN recall@k vs brute force | isolates index approximation | end-to-end nDCG: confounds index and ranker |
| Did the launch help the business? | experiment on revenue/CTR | causal, behavioural | offline nDCG: an ordering claim, not a behaviour claim |

The second table separates the four candidate generators by the property that decides
whether they belong in a production blend. A generator is admitted for what it adds to the
union, not for its standalone quality.

| Generator | Signal used | Typical budget | Strength | Failure that shows up |
|---|---|---|---|---|
| Lexical BM25 | term overlap, IDF, length norm | 300-800 terms | exact match, rare tokens, IDs | zero results on paraphrase |
| Dense ANN | embedding proximity | 200-1000 vectors | semantic near-miss, typos | confident nonsense on rare entities |
| Co-visitation | past co-click graph | 50-200 items | behavioural relevance | popularity feedback loop |
| Trending / recency | aggregate engagement | 20-100 items | cold start, news | age bias, freshness churn |
| Personalised history | user's own past | 20-60 items | repeat intent | filter bubble on single-category users |

Read the third column as a budget, because it is one: recall rises with the union size and
latency rises with it too. The blend arithmetic is a set union, so overlapping generators
waste budget. With BM25 returning 300 and ANN returning 300 with an overlap of 120, the
union is 300 + 300 - 120 = 480 candidates from 600 requested. If the ranker can only score
500, the blend loses 20 slots of headroom to duplication, and the fix may be a larger ANN
budget rather than a better model.

```text
FORMULA (blend arithmetic)
  |C| = sum over g of B_g  -  sum over pairs (g,h) of |C_g ∩ C_h|  +  higher-order corrections
WHERE
  B_g is the requested budget of generator g, |C| is the realised union size
CONSTRAINT
  pairwise subtraction double-counts items present in three generators
  exact union requires one deduplication pass over the concatenation
NUMERICAL EXAMPLE (ILLUSTRATIVE budgets 300, 300, 100)
  raw concatenation 700; observed union 610; overlap cost 90 slots
  if two of the generators share a 40-item trending block, effective overlap is 90 + 40
INTERPRETATION
  budget is spent on duplicates unless the blend deduplicates before truncation
```

Two normalisers are worth comparing directly because they change the direction of a
comparison. IDCG computed over the judged set rewards systems that find *more* relevant
items; IDCG computed over the candidate set only rewards ordering inside the candidate set.
With a pool of five relevant items but a candidate set containing three, the candidate-only
normaliser makes nDCG look excellent while the judged-set normaliser exposes the loss.

| Normaliser | Denominator | Good for | Blind to |
|---|---|---|---|
| judged-set IDCG | all relevant in the judgment file | end-to-end quality | nothing inside its scope |
| candidate-set IDCG | relevant items inside C_q | ranker-only comparison | candidate recall loss |
| fixed constant k | sum of discounts to k | cross-query comparability at fixed depth | label density differences |

The candidate-set normaliser is the correct instrument for an A/B test of the ranker with a
frozen candidate generator, and the wrong instrument for a launch decision. Stating which
one produced a number is not pedantry: in the toy run of section 5 the judged-set nDCG@8 is
0.88194, while a candidate-set normaliser that ignored the two relevant items missing from
the pool would return a materially higher value for the same ordering.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom an on-call engineer actually sees, two competing causes that
produce that symptom, and the check that separates them. The check is the deliverable; the
symptom alone is not actionable.

**FM-1 — offline nDCG rises, online CTR is flat.**
Competing causes: (a) the judgment pool is stale, so the ranker learned the pool rather than
user intent; (b) the improvement is confined to slots 4-10 where examination probability is
low, so the gain never becomes clicks. Discriminating check: split the offline delta by
position band. If the delta is concentrated below slot 3 while the online effect is zero,
cause (b) is sufficient. If the delta is uniform across bands and the online effect is zero,
test the pool version: re-judge 200 queries sampled from the changed traffic and recompute;
a delta that shrinks below half indicates cause (a). Guardrail: require the offline report to
show the top-3 contribution as a separate line before any launch argument.

**FM-2 — nDCG@10 identical, but users complain.**
Competing causes: (a) the ranker is fine and the complaints concern latency or rendering;
(b) two runs trade off head and tail so the cut-aggregate hides a swap. Discriminating
check: compute per-depth deltas at 1, 3, 5, 10 and the Kendall tau between the two orders on
the top 10. Identical aggregate with a tau below 0.5 and opposite per-depth signs means cause
(b); identical aggregate with high tau points outside the ranking stack, so pull p99 latency
and the zero-result rate for the same window.

**FM-3 — candidate recall looks perfect and the system still misses obvious results.**
Competing causes: (a) the pool was generated by the same retriever, so recall is measured
against its own output; (b) a filter (safety, stock, locale) removes items after the union is
formed, so recall was measured before filtering. Discriminating check: rebuild the pool for
50 queries with a brute-force exact scorer over the full catalogue and recompute recall. If
recall collapses, cause (a). If recall holds but the presented list lacks the item, instrument
the post-union filter stage and count removals per query: cause (b), and the fix is a filter,
not a model.

**FM-4 — a query with no relevant items returns nDCG 0.0 and drags the mean.**
Competing causes: (a) the query genuinely has no relevant document and should be excluded;
(b) the pool failed to judge the items that are relevant, which is a data defect. Check:
compare the unjudged rate at depth 10 for those queries against the population. A rate above
the population's 95th percentile indicates (b). Treat undefined 0/0 explicitly: define it as
excluded and report the excluded count, never as 0.0, because 0.0 is the score of a fully
wrong ranking and conflating the two biases every comparison downward.

**FM-5 — the same model scores 0.79 offline and 0.72 on the next run with no code change.**
Competing causes: (a) nondeterministic tie-breaking in the sort changed which equal-score
item landed at which slot; (b) the judgment file version changed underneath. Check: rerun
with a deterministic tie-break on item ID. If the number becomes stable, cause (a). Otherwise
hash the judgment file and diff it against the run manifest; a changed hash is cause (b), and
the fix is pinning the pool version in the manifest.

**FM-6 — nDCG improves on every query but the aggregate falls.**
Competing causes: (a) arithmetic bug, most often averaging the ratio but normalising with a
different k; (b) the improvement is real per query but the query mix changed between runs, so
the aggregate moved by composition. Check: recompute the aggregate on the intersection of
query IDs present in both runs. Recovery of the expected sign on the intersection proves (b);
persistence of the anomaly proves (a). Composition shifts are the reason evaluation sets are
frozen, and the frozen-set check costs one join.

**FM-7 — recall@k is 0.98 in the dashboard and the offline nDCG pipeline says 0.62.**
Competing causes: (a) the dashboard measures recall of the *candidate* set while the pipeline
measures recall of the *presented* list; (b) the dashboard's denominator excludes queries
with no judged relevant items while the pipeline includes them as zero. Check: print both
numerators and denominators for the same date. If the numerators match and denominators
differ, cause (b); if the numerators differ by the presentation depth, cause (a). Two
different recall definitions under one word is the most frequent metric dispute in search.

**FM-8 — a tail query returns one result and the metric still looks acceptable.**
Competing causes: (a) the catalogue lacks the item, so the single result is correct; (b) the
lexical generator required all terms and the union had no fallback. Check: count the number of
terms in the query, the number of matched terms, and the size of the pre-filter candidate set
for that query. A pre-filter size of 1 with all terms required indicates (b); a pre-filter
size of 0 indicates (a), which is a catalogue or coverage question. Track the "queries with
fewer than 3 candidates" rate as a first-class metric, because no ranking metric detects it.

**FM-9 — graded judgments drift mid-experiment.**
Competing causes: (a) two assessor teams applied different grade scales; (b) the same team
re-calibrated after a guideline revision. Check: embed 50 fixed calibration pairs in the
judgment batch and compute agreement per team and per week. Systematic offset between teams
indicates (a); a step change after a date indicates (b). Either way, the two halves of the
experiment are not comparable, and the honest move is to rejudge the overlapping slice rather
than to adjust the score.

**FM-10 — every metric improves after a latency optimisation.**
Competing causes: (a) faster responses change user behaviour, so the logs differ in population
rather than in quality; (b) the timeout fallback (a simpler ranker) was firing on 3% of
requests before and fires on 0.4% now. Check: segment by whether the fallback path served the
request. If the improvement lives entirely in the previously-timed-out segment, cause (b), and
the metric moved because the population of served responses changed. This is the classic
denominator trap: quality did not improve, more of the good path ran.

**Validity condition for this section.** A discriminating check is only discriminating if it
can come out either way. Before running one, write down what result would indicate each
competing cause; a check whose two outcomes both support the same story has not been designed.

## 9. Complexity & Cost Analysis

Metric computation is cheap; everything around it is not. The table gives the operation
count for each step so that "evaluation is slow" can be located rather than guessed.

| Step | Complexity | Work for one query with m judged items and cut k | Notes |
|---|---|---|---|
| discount table | O(k) or lookup | k = 10 -> 10 divisions, cacheable | compute once per cut, reuse |
| DCG@k | O(k) | k multiply-adds | no sorting |
| IDCG@k | O(m log m) | sort m grades, then k multiply-adds | sort dominates for large m |
| nDCG@k | O(1) after the above | one division | guard IDCG = 0 |
| AP@K | O(K) | running hit count | one pass |
| MRR | O(K) worst case | scan until first hit | early exit helps |
| IPS | O(K) | one multiply per logged impression | variance, not time, is the cost |
| paired test | O(Q) | one difference per query | Q = evaluated queries |

For a query set of 100,000 with m = 20 judged items and k = 10, the IDCG sort costs
100,000 * 20 * log2(20) = 100,000 * 20 * 4.32 = 8.64e6 comparison units, while the DCG
itself costs 100,000 * 10 = 1.0e6 multiply-adds. Sorting is roughly nine times the arithmetic
of scoring, which is why evaluation pipelines that re-sort per run are slower than the models
they evaluate. Precompute the sorted grade vector per query once and store it beside the
judgment file.

The serving-side cost picture is inverted. Take an ILLUSTRATIVE workload of 2.0e6 queries
per day: the average rate is 2.0e6 / 86,400 = 23.1 queries per second, and with a 3x peak
factor the design point is 69.4 queries per second. Each query scores 500 candidates of 128
dimensions, so the scoring arithmetic per query is 500 * 128 = 64,000 multiply-adds, and at
the peak rate that is 64,000 * 69.4 = 4.44e6 multiply-adds per second. That is negligible for
any modern core; the work is memory-bound, because each scoring call must gather 128 floats
per candidate from feature stores.

```text
FORMULA (batch bytes per scoring call)
  bytes = n_candidates * n_features * 4 bytes/float32  (+ ids and masks)
WHERE
  n_candidates = 500, n_features = 100 for a compact feature set
CONSTRAINT
  dense float64 doubles the payload; sparse features need an offset array
NUMERICAL EXAMPLE
  500 * 100 * 4 = 200,000 bytes = 195.3 KiB per query
  a batch of 64 queries = 64 * 195.3 KiB = 12.5 MiB per forward pass
INTERPRETATION
  at 69.4 qps the payload rate is 69.4 * 195.3 KiB = 13.3 MiB/s, which fits one link
  but not a shared one; batching to 64 amortises the syscall overhead
```

Index memory is the constraint that decides index type. A flat float32 index of 1,000,000
items with 128 dimensions needs 1e6 * 128 * 4 bytes = 5.12e8 bytes = 512 MB, or 0.51 GB. An
HNSW graph with M = 32 neighbours per node adds 1e6 * 2 * 32 * 4 bytes = 2.56e8 bytes = 256
MB, so the resident set is about 0.75 GB before replicas. That number, multiplied by replica
count, is the real argument against a large embedding dimension; 256 dimensions would make
the vectors alone 1.02 GB and push a three-replica deployment past 3 GB of RAM for a single
shard.

| Resource | Flat index, 1e6 x 128 | HNSW, M = 32 | Consequence |
|---|---|---|---|
| vector bytes | 512 MB | 512 MB | dimension is the lever |
| graph bytes | 0 | 256 MB | M trades recall for RAM |
| query work | 1e6 distance computations | hundreds to thousands of node visits | 100x+ latency difference |
| exactness | exact top-k | approximate, recall@k reported | approximate needs the recall check |

Judgment cost is a human cost and it dominates evaluation cadence. A pool of 500 queries at
depth 8 requires 500 * 8 = 4,000 judgments; at 25 seconds per judged pair that is 100,000
seconds = 27.8 hours, or about 3.5 assessor-days. Doubling the pool doubles the cost, which
is why teams sample rather than judge exhaustively, and why the sampling frame must be
written down. If the pool needs 2,000 queries for a fine-grained comparison, the cost is
14 assessor-days and the schedule, not the statistics, becomes the binding constraint.

Latency budget for one request, ILLUSTRATIVE and deliberately generous:

```text
stage                     ms
ANN / lexical retrieval   12
feature fetch             35
ranking forward pass       8
business reranking        28   (filters, diversity, policy)
blend / truncate           4
network + serialisation   20
                         ----
sum                      107
remaining to 200 ms       93
```

The budget line that matters is the slack: 93 ms of headroom against a 200 ms target. A
candidate-budget increase from 500 to 1,500 costs roughly 3x the feature-fetch stage, which
is 35 -> 105 ms and consumes the entire slack. Any proposal to raise recall by widening the
candidate set MUST be priced against this line before the model work begins, because the
recall gain and the latency cost come from the same parameter.

## 10. Key Theorems / Results Reference

**T1 — the maximum of a discounted sum is attained by sorting.** For non-negative gains and a
discount sequence that is strictly decreasing in position, the sum over a permutation is
maximised when gains are placed in descending order. This is a direct application of the
rearrangement inequality: a sum of products is maximised when both sequences are similarly
ordered. It is what licenses IDCG as a normaliser, because the ideal is computable without
searching over permutations.

**T2 — nDCG lies in [0,1] when the numerator and denominator use the same set, transform,
and cut.** Numerator is a discounted sum over some permutation of the judged items; T1 gives
it an upper bound equal to the denominator. Non-negative gains make it a lower bound of 0.
Break any of the three sharing conditions and the bound fails: numerator over candidates with
denominator over the pool can exceed 1 in principle, and this is the arithmetic signature of
a normaliser bug.

**T3 — nDCG is not comparable across pools with different label density.** Two queries can
have identical DCG and different nDCG, as shown in section 4.4 with 0.91972 against 0.50875
for the same DCG@5 of 1.5. Consequence: a population whose label density changed mid-quarter
produces a metric shift with no behavioural content.

**T4 — AP@K is bounded by 1 only when every relevant item lies inside K.** If |R_q| counts
items beyond K, then AP@K <= (sum of precision values at hits) / |R_q|, and the maximum
attainable value is the fraction of relevant items that fit in K. With 10 relevant items and
K = 5, the best possible AP@5 is 5/10 = 0.5 even when all five top items are relevant —
whereas nDCG@5 would be exactly 1.0 for the same run. Two metrics answering different
questions, and the difference between 1.0 and 0.5 here is definition, not quality.

**T5 — candidate recall is monotone non-decreasing in budget.** If C_q(B1) is a subset of
C_q(B2) for B1 < B2, then recall at B2 is at least recall at B1. Consequence: recall without
a budget is not a claim. Reporting "recall 0.87" while omitting the budget permits any system
to look good by spending latency, and a comparison of two generators at different budgets is
not a comparison of generators.

**T6 — IPS is unbiased under positivity and ignorability.** If every position that can be
logged has a non-zero examination probability, and examination is independent of the item
given the logged policy, then the expectation of click/propensity equals relevance. Both
conditions are assumptions, and both fail in practice: a slot can never be examined by any
user (positivity) and clickbait correlates with examination (ignorability). The estimator is
defensible with clipping and a stated bias budget, not unconditionally.

**T7 — the mean of ratios is not the ratio of means.** With one query at DCG 1 / IDCG 1 and
another at DCG 9 / IDCG 10, the mean of per-query ratios is (1.0 + 0.9)/2 = 0.95, while the
ratio of sums is 10/11 = 0.9091. The gap is 0.041 and it grows with heterogeneity in list
length. Evaluation code MUST average per-query values; aggregating numerators first silently
weights long or label-dense queries more heavily.

**T8 — ERR is bounded by the probability of satisfaction anywhere in the list.** Since
1/p <= 1, ERR <= sum over p of R_p * product over j<p of (1 - R_j) = 1 - product over p of
(1 - R_p) <= 1. For a single grade with R = 7/16 = 0.4375 repeated at every position, the
partial sums are 0.43750, 0.56055, 0.60669, 0.62616, 0.63492 and the limit of the series is
(R/(1-R)) * (-ln R) = (0.4375/0.5625) * 0.82668 = 0.64297. Convergence is fast enough that a
ten-slot cut captures 99.96% of the infinite series' value (0.642710 of 0.642972), which is why
presentation depth beyond ten rarely changes an ERR comparison.

**T9 — the discount curve is convex decreasing, so penalties concentrate in the head.** The
drop from slot 1 to 2 is 0.36907, from 2 to 3 is 0.13093, from 9 to 10 is 0.01197. The ratio
between the first drop and the last is 30.8. Any metric built on this discount will be an
order of magnitude more sensitive to head swaps than to tail swaps, and a claim about tail
quality requires a tail-weighted metric.

**T10 — precision declines mechanically with k once k exceeds the relevant count.** The
maximum attainable P@k is min(1, |R_q|/k). At k = 8 with 5 relevant items the cap is 0.625,
so an observed P@8 of 0.5 is 80% of achievable and not a failure. Reading precision without
its cap is the most common beginner error in a metric review.

**T11 — the standard error of a mean over Q queries scales as 1/sqrt(Q).** With per-query
nDCG standard deviation 0.30 (ILLUSTRATIVE), the standard error is 0.00671 at 2,000 queries
and 0.00300 at 10,000 queries. Quadrupling the query set halves the standard error, so the cost
of resolving a 0.003 difference is four times the judgment budget. Paired comparisons help
more than volume: if the per-query difference has correlation 0.8, its standard deviation is
sqrt(2 - 2*0.8) * 0.30 = 0.1897, and 62,700 queries resolve a 0.003 delta at conventional
power.

## 11. Interview / Exam Questions

**Q1 (calculation).** Grades at slots 1 to 4 are 4, 0, 2, 0. Compute DCG@3 and nDCG@3 with
exponential gain.
*Answer:* gains 15, 0, 3. DCG@3 = 15*1.0 + 0 + 3*0.5 = 16.5. The ideal ordering of the same
items is 4, 2, 0, 0 with gains 15, 3, 0, so IDCG@3 = 15 + 3*0.63093 = 16.89279. nDCG@3 =
16.5/16.89279 = 0.97675.
*Reasoning:* the only loss is the zero at slot 2, which displaces the grade-2 item by one
position and costs 3*(1.0 - 0.63093) = 1.10721 in gain units.
*Follow-up:* what does nDCG@4 become, and why is it equal here? Because the ideal list is also
only three non-zero items, IDCG@4 is unchanged, and the fourth slot contributes zero to both
sides, so 0.97675 stands.
*Red flag:* computing IDCG by sorting the observed sequence including the slot order.

**Q2 (calculation).** A pool holds 60 relevant items for a query family. Candidate generation
returns 500 items containing 40 of them. State candidate recall and the headroom.
*Answer:* 40/60 = 0.6667, so headroom is 20 items or 33.3 percentage points of recall. If the
budget rises to 1,000 and retrieval reaches 52, recall is 52/60 = 0.8667.
*Reasoning:* recall is a set-intersection ratio against an independently built pool, so its
denominator is the pool's relevant count, not the candidate size.
*Follow-up:* why can recall exceed nDCG but never be inferred from it? Because nDCG normalises
by items present in the candidate set, so it divides out exactly the loss recall measures.
*Red flag:* quoting recall without the budget, or measuring it against a pool built by the
same generator.

**Q3 (calculation).** Twelve items are relevant for a query. The ranked list has hits at
positions 2, 5, and 9. Compute AP@10 and recall@10.
*Answer:* precision at the hits is 1/2 = 0.5, 2/5 = 0.4, 3/9 = 0.3333. Sum = 1.2333. Divide
by 12: AP = 0.10278. Recall@10 = 3/12 = 0.25.
*Reasoning:* the denominator is the total relevant count, not the number of hits, so AP does
not saturate at 1 when the pool is deeper than the cut.
*Follow-up:* what is the maximum AP@10 for this query? 10/12 = 0.8333, reached only if ten of
the twelve relevant items fill the ten slots.
*Red flag:* dividing by 3 and reporting 0.4111 as AP.

**Q4 (diagnosis).** Two queries both report DCG@5 = 1.5 (binary relevance) but nDCG@5 of
0.91972 and 0.50875. What happened, and which number should be quoted in a launch review?
*Answer:* the queries have different relevant counts. With two relevant items IDCG@5 =
1.63093; with five, IDCG@5 = 2.94846. The absolute quality is identical and the normaliser
differs. Quote per-query values averaged over a fixed population, and never mix populations
with different label density.
*Reasoning:* nDCG is a ratio whose denominator is a property of the query, so it measures
relative achievement within a query's own potential.
*Follow-up:* how would you compare two systems on a population whose density changed? Freeze
the population, recompute both runs on the intersection of query IDs.
*Red flag:* concluding that the second query's ranker is worse.

**Q5 (calculation).** Forty percent of queries have no relevant item within depth 10. What is
the maximum possible MRR, and what does that imply about interpreting a 0.02 MRR gain?
*Answer:* the 60% of queries with a hit can contribute at most 1 each, so MRR <= 0.60. A gain
of 0.02 is 3.3% of the attainable ceiling, and it can arise entirely from candidate coverage
rather than from ordering.
*Reasoning:* RR is zero for a miss regardless of the ordering of the irrelevant tail, so the
metric's attainable range is set by hit rate before any ranking question is asked.
*Follow-up:* which metric would expose the missing results directly? Recall@10 against the
pool, or the no-hit query rate tracked as its own series.
*Red flag:* treating MRR as a general-purpose quality score.

**Q6 (calculation).** Logged clicks at slots 1, 1, 3, and 8 with examination propensities
1.0, 1.0, 0.5, and 0.1. Compute the unweighted mean and the IPS estimate.
*Answer:* unweighted 4/4 = 1.0. Weights are 1.0, 1.0, 2.0, 10.0, so the IPS mean is
(1 + 1 + 2 + 10)/4 = 3.5.
*Reasoning:* each click is divided by the probability its slot was examined, so a click at a
low-propensity slot counts as evidence of many unobserved clicks.
*Follow-up:* what is the variance cost? Weights span a factor of 10, so the effective sample
size is far below 4; clip large weights and report the bias introduced.
*Red flag:* using the unweighted mean and calling the ranking unbiased.

**Q7 (calculation).** Per-query nDCG has standard deviation 0.30 and the paired difference
between two rankers has correlation 0.8. How many queries are needed to detect a 0.003
difference at 80% power and 5% two-sided?
*Answer:* the difference's standard deviation is sqrt(2 - 2*0.8)*0.30 = 0.1897. Using
n = 2*(1.96 + 0.84)^2 * 0.1897^2 / 0.003^2 = 2 * 7.84 * 0.035986 / 9e-6 = 62,700 queries per
arm.
*Reasoning:* the paired design removes the shared query difficulty, which is why correlation
helps so much: without it the standard deviation would be 0.30*sqrt(2) = 0.424 and the query
requirement would more than quadruple.
*Follow-up:* what if the judgment budget only allows 5,000 queries? Then the smallest
detectable difference is roughly 0.003 * sqrt(62,700/5,000) = 0.0106, and the review should
state that resolution rather than reporting a three-decimal win.
*Red flag:* comparing two independent means and declaring significance from a 0.004 gap.

**Q8 (calculation).** An experiment has 100,000 queries per arm and baseline CTR 3.0%. Compute
the minimum detectable absolute difference in CTR.
*Answer:* the standard error of the difference is sqrt(2 * 0.03 * 0.97 / 100,000) = 0.00076,
so the minimum detectable effect at 80% power is (1.96 + 0.84) * 0.00076 = 0.00214, or 0.21
percentage points absolute.
*Reasoning:* for proportions the variance is fixed by the baseline rate, so the only levers
are sample size and a variance-reduction technique such as cuped or stratification.
*Follow-up:* a 0.15 pp CTR lift is claimed from this experiment. Is it credible? No: it is
below the 0.21 pp floor computed for the design, so the interval includes zero and the claim
needs a larger sample or a pre-registered direction.
*Red flag:* reporting a p-value without stating the design's detection floor.

**Q9 (selection).** An autocomplete dropdown must be evaluated. Which metric, and why not
nDCG@10?
*Answer:* MRR or ERR. The user takes the first acceptable completion and leaves, so value is
concentrated in the top few slots and the reciprocal-rank form matches the behaviour. nDCG@10
spreads credit over ten slots the user never reads, and it does not model stopping.
*Reasoning:* the metric's shape should match the user's stopping rule; cascade metrics encode
stopping, discounted-gain metrics do not.
*Follow-up:* what guardrail belongs with it? Suggestion acceptance rate and the rate of
suggestions that lead to a zero-result page, because a fast accepted completion to nothing is
a metric win and a product loss.
*Red flag:* choosing nDCG because it is familiar.

**Q10 (diagnosis).** Offline nDCG improves by 0.9%. The top-10 unjudged rate rose from 3% to
12% for the same period. What is the most likely explanation and what do you do next?
*Answer:* pool coverage degraded, so the ranker is being rewarded for what the pool happens to
contain. Twelve percent unjudged in the top 10 is far above the population's normal rate, and
unjudged items enter the numerator as zero gain. Next: re-judge a stratified sample of the new
items, then recompute on the expanded pool.
*Reasoning:* a metric computed on a partially labelled set is a lower bound whose looseness
varies with how far the ranker explores beyond the pool.
*Follow-up:* how would you make the diagnostic permanent? Emit unjudged-at-depth-k as a
first-class evaluation metric next to nDCG, with an alert threshold.
*Red flag:* attributing the gain to the model without checking label coverage.

**Q11 (calculation).** With ten presentation slots and one slot necessarily irrelevant, what
is the ceiling on nDCG@10 for a query whose ten relevant items all have grade 2?
*Answer:* the discount sum to 10 is 4.54356; removing the smallest term at slot 10 gives
4.25450. The ratio is 4.25450/4.54356 = 0.93638.
*Reasoning:* the ceiling is a property of the slot count and the label pattern, not of the
model, so it must be computed before any headroom claim.
*Follow-up:* if nine slots are relevant and one is not, and the irrelevant item can be placed
last, why does the ceiling not become 1.0? Because IDCG@10 uses ten relevant items while the
run can only place nine, and the tenth slot's gain is missing from the numerator.
*Red flag:* reporting 0.93 against a ceiling of 0.936 as if 7% of headroom were available.

**Q12 (calculation).** BM25 returns 300 candidates, a dense index returns 300, and the overlap
is 120. The ranker can score 500. What is the union size and what should change?
*Answer:* 300 + 300 - 120 = 480 distinct candidates, so 20 scoring slots are unused. Either
raise one generator's budget by 20 to fill the ranker, or accept 480 and reduce the scoring
batch.
*Reasoning:* budget is spent on duplicates before truncation, so overlap is a cost paid twice.
*Follow-up:* how do you decide which generator to widen? Add each generator's exclusive
contribution — here 180 from each — and widen the one with higher marginal recall per
millisecond.
*Red flag:* summing the two budgets and claiming 600 candidates.

**Q13 (judgment).** Two rankers report nDCG@10 of 0.7927 and 0.7960 with a single-arm standard
error of 0.0067 on 2,000 queries. Ship the second?
*Answer:* no decision is supportable yet. The gap of 0.0033 is roughly half a standard error,
so the paired interval almost certainly spans zero. Report the bound: with the paired
difference standard deviation around 0.19, the detectable gap at this sample is about 0.012.
*Reasoning:* the correct comparison is the distribution of per-query differences, not two
marginal means, and the effect size must be judged against the design's resolution.
*Follow-up:* what would justify shipping? Either a much larger evaluation set, a pre-registered
non-inferiority frame on a guardrail, or an online experiment where the primary metric has
adequate power.
*Red flag:* ranking the two decimals and calling the larger one the winner.

**Q14 (diagnosis).** A latency optimisation makes every offline metric improve, but no model
changed. Where do you look?
*Answer:* at the population of served requests. The timeout fallback was firing on 3% of
requests before and 0.4% after, so the evaluated population now contains fewer fallback-served
queries, which are systematically worse. The metric moved because the mixture changed.
*Reasoning:* this is the denominator trap: quality is unchanged and the share of the good path
increased.
*Follow-up:* how do you keep the comparison honest? Pin the population by query ID, or
evaluate both arms on the subset where neither arm timed out.
*Red flag:* accepting "the code got faster so ranking got better" without a population check.

**Q15 (calculation).** One query scores DCG 1 / IDCG 1, another DCG 9 / IDCG 10. Compute the
mean of ratios and the ratio of means, and say which the pipeline must report.
*Answer:* mean of ratios = (1.0 + 0.9)/2 = 0.95; ratio of means = 10/11 = 0.9091. Report the
mean of per-query ratios, 0.95, because the metric is defined per query and only the mean of
ratios keeps queries equally weighted.
*Reasoning:* aggregating numerators first gives weight proportional to IDCG, so a long or
label-dense query dominates the score.
*Follow-up:* which direction does the discrepancy usually go? The ratio of means is biased
toward queries with large normalisers, so it can flatter or punish a system purely by query
mix.
*Red flag:* computing the aggregate DCG and IDCG and dividing once.

## 12. Quick Reference Card

| Item | Value or rule |
|---|---|
| DCG@k | sum over p = 1..k of (2^rel - 1) / log2(p + 1) |
| nDCG@k | DCG@k / IDCG@k, dimensionless, in [0,1] |
| IDCG@k | same formula on grades sorted descending, truncated at k |
| AP@K | (1/|R_q|) * sum of precision at each hit; unbounded by 1 if relevant items sit beyond K |
| MRR | mean of 1/p_first; zero for a miss, so bounded by hit rate |
| Candidate ceiling | recall@B = |Rel ∩ C(B)| / |Rel|, pool must be independent of the generator |
| Noise floor | SE = sd/sqrt(Q); sd 0.30 gives 0.0067 at 2,000 queries, 0.0030 at 10,000 |
| Never | quote a metric without population, pool version, gain transform, and cut |
