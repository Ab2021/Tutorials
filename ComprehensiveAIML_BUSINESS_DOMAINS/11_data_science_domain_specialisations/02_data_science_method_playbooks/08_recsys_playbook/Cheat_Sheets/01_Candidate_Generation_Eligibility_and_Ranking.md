---
title: "Candidate Generation, Eligibility and Ranking"
topic: "retrieval candidate generation, point-in-time eligibility, ranking handoff"
collection: "recsys-playbook"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
notation_style: "plain-text fenced blocks"
prerequisites: ["set algebra", "dot products and cosine similarity", "binomial and hypergeometric reasoning", "event-time vs processing-time joins"]
pattern_tags: ["candidate-generation", "eligibility", "filtered-ann", "recall-at-k", "exposure-bias"]
owner_role: "recommendation engineering and product analytics"
source_index: "INDEX.md"
review_trigger: "Eligibility policy change, catalog or index schema change, ranker handoff contract change, or evaluation population change"
---

# Candidate Generation, Eligibility and Ranking

## 1. File Header

**Purpose:** specify how a recommendation request moves from a catalog to a displayed slate, and how to measure that movement without confusing retrieval quality with ranking quality or with business outcome.
**Audience:** recommendation engineers, applied scientists, product analysts, and interview candidates who must defend a candidate pipeline under scrutiny.
**Prerequisites:** you should be comfortable with set intersection and difference, dot products, variance of a proportion, and reconstructing a row of features as of a stated timestamp. If joins between event-time and processing-time tables are unfamiliar, read a window-frame reference first.
**Outcome:** given a point-in-time eligible catalog, a relevance-label rule, and two retrieval systems, you will be able to compute matched candidate recall with a confidence interval, detect filtered-ANN underfill before it reaches users, and size the candidate budget against a latency and cost envelope.
**Excluded here:** learning-to-rank loss design, counterfactual policy optimisation, and statutory interpretation of content rights. Those are separate decisions with separate evidence standards.
**Status:** draft and unreviewed. No product, deployment, latency percentile, or revenue figure in this card is measured; every number is either derived from a stated formula or labelled ILLUSTRATIVE.
**Evidence posture:** DERIVED for the algebra and the bounds, ILLUSTRATIVE for constructed catalogs and request traces, OPEN for any claim about a specific vendor index's recall or a specific company's funnel.
**Source boundary:** general information-retrieval and vector-search documentation informs the vocabulary; no source file for this unit was available for inspection, so no claim here rests on an unread primary source. Faiss project documentation is cited only for the existence of approximate search, not for measured properties.
**Retrieval cue:** Section 4 for the formulas, Section 5 for one fully worked request set, Section 8 for diagnosis, Section 9 for sizing, Section 12 for the card.

**The one decision this card supports.** Should a candidate pipeline be trusted to feed a ranker, and on what evidence? That question decomposes into three sub-decisions that fail for different reasons:

1. Is the eligible set correct at the instant of the decision, including every rights, entitlement, availability, and safety constraint?
2. Does the candidate set contain enough of the eligible relevant items for the ranker to have a chance?
3. Does the evaluation that produced those two answers use the same population, cutoff, and label horizon as the decision it claims to inform?

A pipeline can pass (1) and (3) while failing (2), and the symptom is not an error — it is a quiet loss of quality that looks like a ranking problem.

**Why the stages are separated at all.** Scoring every eligible item with a heavy ranker is usually infeasible: a catalog of 12 million items against a 128-dimension embedding and a 60-feature ranker does not fit a 250 ms budget. The standard resolution is a cascade: cheap retrieval proposes a few hundred to a few thousand items, and an expensive model orders them. The cascade introduces a hard structural fact that governs everything else in this card: **a ranker cannot display what retrieval did not return.** Every relevant item lost at the candidate stage is lost permanently for that request, no matter how good the ranker is.

**What "eligibility" means operationally.** Eligibility is not a property of an item in the abstract. It is a predicate over (item, viewer, surface, timestamp). A title can be eligible in one territory and not another, eligible for a subscriber and not for an ad-supported viewer, purchasable in one fulfilment zone and not another, and eligible at 10:00 but not at 10:01 after a rights revocation. Any evaluation that treats eligibility as an item-level attribute will silently average across incompatible populations.

**Reading order for a practitioner under time pressure.** Start at Section 8, find the failure whose symptom you are seeing, follow its discriminating check, then return to Section 4 for the formula that check depends on. The formulas are ordered by increasing realism, so an early exit at 4.1 is legitimate when the catalog is small and the eligible set is trivially correct.

## 2. Concept Map / Mental Model

```text
event-time facts                 decision-time predicate              display
--------------                   -----------------------              -------
catalog ingest ----+
rights change -----+--> snapshot at t ---> E(q,t) eligible set --+
inventory move ----+          |                                   |
entitlement ------+           |                                   v
                              |                        retrieval sources
                              |                     (popularity, co-visit,
                              |                      content, embedding ANN)
                              |                                   |
                              |                                   v
                              +-------------------> filter -> dedup -> cap k
                                                                  |
                                                                  v
                                                          candidate set C(q)
                                                                  |
                                                                  v
                                                    ranker -> policy -> slate
                                                                  |
                                                                  v
                                              exposure log (item, position, arm)
                                                                  |
                                                                  v
                                             outcome log (mature after horizon H)
                                                                  |
                                                                  v
                                             labels -> next evaluation population
```

The diagram has two distinct loops and they must not be merged. The top loop is a **state loop**: rights, inventory, and entitlement change state and invalidate cached decisions. The bottom loop is an **evidence loop**: exposures and outcomes become labels that feed the next offline evaluation. Merging them produces the classic defect where a model is trained on labels that were generated by the policy being evaluated.

**Four invariants the diagram asserts.** These are properties to test, not aspirations:

- **I1 Containment:** every ID in the candidate set is a member of the eligible set computed at the same timestamp. Formally C(q) ⊆ E(q,t).
- **I2 Uniqueness:** no ID appears twice in the candidate set, regardless of how many sources proposed it.
- **I3 Boundedness:** |C(q)| ≤ k for the declared budget k, or the underfill is explicitly reported.
- **I4 Determinism:** given identical inputs and a declared tie rule, the candidate set is identical. Without I4, A/B comparisons cannot attribute a difference to the treatment.

**The eligibility funnel, stated as attrition.** Eligibility is a sequence of independent-looking filters that are in fact correlated. A construct like "territory rights" and "platform availability" both depend on distribution deals, so a title excluded for one is often excluded for the other. Treating them as independent overestimates the surviving set. Section 4.2 gives the correction term; Section 5 works it numerically.

**Where the model sits and where it does not.** Retrieval models and rankers estimate affinity. They do not estimate eligibility, and they must not be asked to. A common architectural error is to let a learned scorer produce a single fused score that mixes relevance with a soft eligibility penalty. The moment eligibility becomes a soft penalty, an ineligible item can outrank an eligible one, and the failure is a policy violation rather than a quality regression. Eligibility belongs in a hard filter that runs after scoring and again before display.

**The three boundaries.** Practitioners conflate three different cuts, and most evaluation disputes come from mixing them:

| Boundary | What it separates | What breaks if confused |
|---|---|---|
| Authorization boundary | permitted vs forbidden items | a rights or entitlement violation reaches a user |
| Candidate boundary | proposed vs not proposed | recall is measured on a set the ranker cannot reach |
| Display boundary | shown vs not shown | precision, position effects, and CTR are misread as preference |

Each boundary has its own metric family: policy violation rate for the first, candidate recall and underfill for the second, exposure-weighted outcome metrics for the third. A single "quality score" that spans all three cannot be defended, because a gain in one can hide a loss in another.

**The no-recovery lemma, as a mental shortcut.** For any request, the number of relevant items eventually displayed is at most the number of relevant items in the candidate set. This is trivially true and enormously useful: it converts an argument about end-to-end quality into an argument about a set, and sets can be counted. Every measurement in Section 4 is an instance of counting a set against a declared universe.

**Takeaway.** Protect eligibility with hard filters at the last responsible moment, measure retrieval as a set-overlap problem against a declared universe, and never let a recall improvement be reported without the underfill and latency that paid for it.

## 3. Foundational Definitions

Definitions here are written as operational tests: each row states how you would determine membership or value from logs, not how the concept is described in a paper.

| Term | Operational test | Units / type | Confusion to avoid |
|---|---|---|---|
| Request q | One recommendation decision: viewer, surface, placement, and decision timestamp | event row | A page view can contain several requests; a session is not a request |
| Catalog I(t) | Distinct item IDs with a valid ingest record as of t | set of IDs | The catalog is not the inventory; an item can be listed but unfulfillable |
| Eligible set E(q,t) | IDs passing every hard predicate for this viewer, surface, and t | subset of I(t) | Eligibility is a predicate over a tuple, not an item property |
| Relevance set R(q,H) | Eligible IDs satisfying the declared label rule within horizon H | subset of E | Unobserved is not irrelevant; R is defined only where labels exist |
| Candidate set C(q) | Deduplicated IDs admitted by retrieval, before the ranker | set, cardinality ≤ k | Candidate count is not the displayed count |
| Source list S_j(q) | Ordered output of retrieval source j, with scores | ordered list | Sources are not interchangeable; their score scales are not comparable |
| Exact top-k X_k(q) | First k eligible IDs under a fixed score with a deterministic tie rule | ordered list | Exact with respect to a vector score, not with respect to utility |
| Approximate top-k A_k(q) | ANN output under the same score and eligible universe | ordered list | Approximate-search fidelity is not relevance recall |
| Exposure | Item rendered in a slot with a recorded position and arm | event | Retrieved is not exposed; logging failures make exposure unknowable |
| Impression window | Declared interval after which an exposed item's outcome is counted | duration | A short window on a slow-converting item manufactures false negatives |
| Label horizon H | Time from exposure to label maturity, fixed before evaluation | duration | Changing H between systems changes the estimand |
| Underfill | Requests where the candidate set is smaller than k | count or rate | Underfill with a small catalog is legitimate; underfill from filtering is a defect |
| Candidate recall@k | Share of R recovered by the first k candidates | proportion, [0,1] | Denominator is the eligible labelled count R ∩ E, not all labelled items |
| ANN fidelity | Share of exact top-k recovered by an approximate index | proportion, [0,1] | Says nothing about whether those neighbours are relevant |
| Eligibility density p | Share of the catalog passing all hard predicates for a request | proportion | Varies by request; a single global p hides per-cohort variation |
| Staleness | Decision timestamp minus the timestamp of the snapshot used | seconds | A freshly built index can embed a stale rights feed |
| Position bias | Dependence of the observed action on rendered slot | propensity by slot | CTR per item is not comparable across slots |
| Slate diversity | Spread of a declared attribute across displayed slots | metric-specific | Diversity is a constraint on the slate, not a property of one item |
| Guardrail | Separately monitored outcome bounding acceptable policy change | series with threshold | A recall gain does not satisfy a guardrail |

**How the sets nest.** E(q,t) is the universe for retrieval. R(q,H) is defined only inside E, because a label cannot be observed for an item that could not be shown. C(q) is drawn from the retrieval index and then intersected with E. The displayed slate D(q) is a prefix of the ranked C(q) after policy adjustment. This nesting is the reason candidate recall uses |R ∩ E| in the denominator: dividing by a larger, partially unshowable R produces a number that no pipeline could improve.

**Why R requires a horizon.** A relevance label is an assertion about a user's future action. "Relevant" for a video may mean "watched at least 60 seconds within 7 days"; for a marketplace item, "purchased within 14 days"; for a news article, "engaged with within 24 hours". Without the horizon the label is undefined, and systems evaluated under different horizons cannot be compared. Two pipelines can differ by 8 points of recall while differing by zero in behaviour, simply because one was labelled at 24 hours and the other at 7 days.

**Why exposure is a definition, not a log detail.** If exposure logging drops impressions for one surface, then R for that surface is computed over a subset of what users actually saw. The resulting recall estimate is biased in a direction that depends on which items were dropped. An exposure audit — counting logged exposures against server-side render decisions — is therefore a prerequisite for trusting any retrieval metric, not an optional data-quality chore.

### 3.1 Constructing the eligible set from a policy log

The definition only becomes an instrument when you can write the query that produces E. The construction below is the one that survives audit, because every predicate carries its own effective time and its own failure mode.

```text
FORMULA:   E(q,t) = { i in I(t) : R(i, terr(q), t) AND S(i, tier(q), t)
                                AND A(i, surf(q), t) AND F(i, loc(q), t)
                                AND Y(i, t) }
WHERE:     R = rights predicate, S = entitlement, A = asset availability,
           F = fulfilment or licensing region, Y = safety policy
           each predicate evaluated at t, not at ingest, and not at label time
CONSTRAINT: every predicate is a lookup into a versioned store keyed by
            (subject, attribute, valid_from, valid_to) with no NULL as allow
NUMERICAL EXAMPLE: for one request the five predicates reject 12%, 18%, 5%,
            7% and 0% of the catalog respectively; sequential application
            leaves 637.5 of every 1,000 candidates (Section 4.2).
INTERPRETATION: the eligible set is a join, and the join is only as correct as
            the temporal validity of each side.
```

**Three query-level defects that produce a wrong E.**

*Open-interval predicates.* A rights row stored as `valid_to = NULL` meaning "current" silently becomes "valid forever" once a revocation is written as a new row that some consumer does not read. The fix is to materialise explicit intervals and to reject any predicate that cannot resolve a `valid_to`; a NULL must never default to allow.

*Attribute timing.* The viewer's territory must come from the same snapshot as the policy evaluation. If territory is read from a profile updated at session start and policy is read at request time, a viewer who crosses a border mid-session is evaluated against a mismatched pair, and the resulting E is neither the old one nor the new one. This is not hypothetical for travel and VPN traffic; it is the standard cause of "the app shows content I cannot play".

*Predicate order in the presence of cost.* Rights lookups are usually more expensive than device-capability checks, so a query optimiser reorders them. Reordering changes the *rejection attribution* recorded in logs even when it does not change E. Any metric built on rejection reason codes must therefore be computed from a labelled evaluation of every predicate against a sample, not from the first-failing-predicate log, which is order-dependent.

**What "no NULL as allow" costs.** Failing closed converts a policy outage into an empty slate, which is a visible product event. That is the correct trade, but it must be instrumented: the authorization-error metric in Section 8.2 exists precisely to distinguish "nothing eligible" from "the policy service is down". Without it, an outage appears as a catalog scarcity problem and responders investigate the wrong subsystem.

**A cheap audit that finds most errors.** Take 200 requests, sample 5 candidate items each, and for every item record the decision of each predicate from both the request path and an independent offline re-evaluation. The disagreement rate per predicate is the correction you need to apply to the funnel numbers. In practice the rights predicate dominates; if you can only audit one, audit that one.

**Why "candidate" must exclude the post-ranker set.** Some teams call the top 100 after ranking "candidates". That usage destroys the diagnostic value of the term, because the post-ranker set is already a function of the score you are trying to evaluate. In this card, candidate always means pre-ranker, and the term for the later set is the slate.

**Takeaway.** Before computing any percentage, fix four things on paper: the universe (E), the labelled subset (R with its horizon), the candidate budget (k), and the timestamp of the snapshot. A metric without those four is not comparable to anything.
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
## 8. Failure Modes & Edge Cases

### 8.1 How to use this section

Each failure below is written so that a reader with production access can move from a symptom to a test that discriminates between competing causes. The discipline matters because the obvious explanation is usually wrong: a drop in click-through is attributed to the ranker when candidate recall fell, and a spike in empty slates is attributed to a catalog problem when a filter was applied after truncation.

The structure of each entry: **symptom** (what the alert says), **competing causes** (at least two, because a single-cause failure needs no diagnosis), **discriminating check** (a query or comparison that separates them and whose outcome is observable today), **correction** (what to change), **prevention** (what makes the class of failure impossible). Failures that share a symptom are grouped so the checks can be read against each other.

### F-01 — Revoked rights reach a display

**Symptom:** a policy audit finds a title rendered after its rights window closed. No error was logged and latency was normal.
**Competing causes:** (a) the cached slate was reused across requests without re-evaluating eligibility; (b) the index refresh ran but the served replica was the pre-refresh one; (c) the revocation was ingested with a future effective timestamp and the request time comparison used processing time instead of effect time.
**Discriminating check:** for the specific request, print the policy version and effective timestamp observed by the final check, and compare with the revocation record's effective timestamp. If the final check observed the post-revocation version, cause is not (a). If the replica identity in the response header is stale, cause is (b).
**Correction:** move the authoritative eligibility check to the last responsible moment before render, and make the check return a versioned decision that is logged with the response.
**Prevention:** an invariant test in staging that revokes a title mid-request and asserts the item is absent from the rendered slate, run on every deployment.

### F-02 — Filtered ANN underfills while eligible items exist

**Symptom:** slates with fewer items than slots, concentrated in narrow territories; recall unchanged on the global cohort.
**Competing causes:** (a) depth too small for the local density; (b) the filter is applied after truncation instead of during traversal; (c) the eligible set genuinely is small for that segment.
**Discriminating check:** compare |C(q)| against the maximum eligible relevant count over the same segment. If a filter-aware exact oracle on the same query returns 20 items regularly, cause (c) is excluded and the defect is in depth or in filter placement. Then check whether increasing depth from 20 to 400 changes the delivered count: if it does, cause is (a) or (b); if the delivered count stays at zero across all depths, the filter predicate is wrong.
**Correction:** provision depth from [D1] using the segment's measured density, and push the predicate into traversal so rejection does not consume the depth budget.
**Prevention:** a per-segment underfill dashboard with the eligible-set size as the companion series, so a small catalog and a broken filter never look alike.

### F-03 — Offline recall spikes after a label change

**Symptom:** recall@20 rises 12 points overnight with no deployment.
**Competing causes:** (a) the label horizon was shortened, shrinking the denominator; (b) the label rule's threshold changed (for example a minimum watch time); (c) a logging change increased exposure capture, adding labelled items that were previously invisible; (d) a genuine retrieval improvement shipped silently in a config change.
**Discriminating check:** recompute recall on the frozen cohort with the old and new label definitions. If the number moves with the definition, the metric moved, not the system. Then compare the denominator count: a rise in recall at constant numerator means the denominator fell, which points at (a) or (b), never at (d).
**Correction:** version the label rule alongside the model, and store the rule identifier in the evaluation output.
**Prevention:** make the label rule a first-class artefact with a version that appears in every metric's metadata; refuse to compare metrics whose label versions differ.

### F-04 — New viewer receives an empty slate

**Symptom:** first-session viewers get zero recommendations on the primary surface.
**Competing causes:** (a) the personalisation source requires an interaction history and returns nothing, and there is no fallback; (b) the fallback exists but its own eligibility filter removed everything because it keyed on a viewer attribute that is null for new accounts; (c) the request is malformed at the edge and never reaches retrieval.
**Discriminating check:** count requests by stage reached and by source-attempted outcome code. A request that reached the popularity source and got zero items after filtering points at (b); a request with no source attempts points at (c); a request with attempts and an explicit "insufficient history" code points at (a).
**Correction:** implement a tiered fallback: contextual, then locally eligible popularity, then editorial, each with its own eligibility filter that does not depend on the missing attribute.
**Prevention:** a synthetic new-account fixture in staging that must produce a full slate; alert on the share of requests whose slate is shorter than the slot count, split by account age.

### F-05 — New items never receive exposure

**Symptom:** items ingested in the last 14 days account for under 1% of impressions despite being 4% of the catalog.
**Competing causes:** (a) no retrieval source can propose an item without interactions, so content similarity is missing or unused; (b) exploration quota is set to zero; (c) the item is eligible but its embedding is a zero vector because the feature pipeline had not populated it at ingest; (d) the eligibility predicate rejects it for a legitimate reason such as an unapproved asset.
**Discriminating check:** for a sample of unexposed new items, compute their presence in each source's output when queried with a synthetic probe vector. Present in content source but never in slates points at (b). Absent from every source with a zero-norm embedding points at (c). Absent from the candidate pool entirely with a policy rejection code points at (d).
**Correction:** add a content-based source with its own quota, backfill embeddings before the item becomes eligible, and instrument a per-item "proposed count" counter.
**Prevention:** make "eligible but never proposed" a tracked state with an alert, because it is invisible in every outcome metric.

### F-06 — Approximate index degrades after a parameter change

**Symptom:** recall@20 drops 3 points two days after an index rebuild. No model changed.
**Competing causes:** (a) the rebuild used different graph parameters, changing fidelity; (b) the rebuild used a different embedding version because the artifact pointer moved; (c) the rebuild dropped vectors for a shard that failed and served from a smaller index.
**Discriminating check:** compute [A1] fidelity of the new index against a float32 exact oracle on a frozen query set. Fidelity down with identical vectors points at (a). Fidelity normal but score distribution shifted points at (b). Vector count in the index versus the catalog's eligible count points at (c).
**Correction:** pin the embedding version and index parameters in the build manifest; fail the build if the vector count deviates from expectation beyond a tolerance.
**Prevention:** store a fidelity report as a build artefact and gate promotion on it, so an index cannot reach production unmeasured.

### F-07 — Recall is high and outcomes are flat

**Symptom:** candidate recall improves from 0.38 to 0.48, and the online primary metric does not move.
**Competing causes:** (a) the ranker had already been saturating on the relevant items that were present, so extra candidates are redundant; (b) the added candidates are relevant under the offline label but not under the online objective (a label/objective mismatch); (c) the added candidates arrive late in the slate and are never examined; (d) the traffic split was not honoured, so the comparison is between cohorts that differ in ways unrelated to the treatment.
**Discriminating check:** compute ranker retention — the share of relevant candidates that reach the slate — before and after. If retention was already near 1.0, cause is (a). Then check the exposure-weighted position distribution of newly added relevant items; if they land below the last examined slot, cause is (c). Sample-inspect arm assignment against the configured split to exclude (d).
**Correction:** if (a) holds, the constraint is the ranker or the slate, so improving retrieval further has no value; re-target the work at ranking or at slot count.
**Prevention:** declare the binding constraint before starting work: candidate recall below roughly 0.6 with high ranker retention means retrieval is binding; the reverse means ranking is.

### F-08 — Two services disagree about eligibility

**Symptom:** an item appears in retrieval output and is rejected at display, at a rate of 9% of candidate rows.
**Competing causes:** (a) the two services read different policy sources, one of which lags; (b) they evaluate the predicate at different timestamps within the same request; (c) one service applies a default-allow on a policy lookup failure while the other applies default-deny.
**Discriminating check:** join both services' decision records for the same request and diff the policy version, the evaluated timestamp, and the failure code. A version mismatch points at (a); identical versions with timestamps a second apart points at (b); a failure code on one side with an allow decision points at (c).
**Correction:** make one service own the decision and have the other consume its signed result within a bounded age; treat any lookup failure as deny at both.
**Prevention:** a contract test that asserts both services return identical decisions for a corpus of constructed (item, viewer, timestamp) triples, run on every deploy of either.

### F-09 — A minority cohort sees poor coverage

**Symptom:** one language cohort has 30% fewer distinct items per week than the platform average.
**Competing causes:** (a) the cohort's eligible catalog is genuinely smaller; (b) the embedding model was trained on the majority cohort's behaviour and places the cohort's items far from its users' vectors; (c) a fallback source has no items in that cohort's language and dominates the slate.
**Discriminating check:** compare eligible catalog size per cohort first. If comparable, compute coverage — distinct items exposed over distinct eligible items — for each source separately, which isolates (c) from (b).
**Correction:** for (b), evaluate a per-cohort content source, and report recall per cohort rather than pooled so the deficit is visible.
**Prevention:** make cohort-stratified recall a standard output of the evaluation job, with a minimum sample per cohort, so pooled averages cannot hide a cohort at zero.

### F-10 — Duplicate items crowd the candidates

**Symptom:** the ranker sees 20 candidates but only 13 distinct product concepts; the slate looks repetitive.
**Competing causes:** (a) the same item ID appears from several sources and dedup keys on the row, not the ID; (b) variants (colour, size, edition) are distinct IDs but one concept, and dedup has no concept key; (c) the sources themselves contain duplicates because the upstream feed did.
**Discriminating check:** count rows, distinct item IDs, and distinct concept keys for the candidate set. rows > IDs points at (a); IDs > concepts points at (b); duplicates present within a single source's output points at (c).
**Correction:** dedup by ID first, then apply a concept-level cap so one concept cannot occupy more than a declared number of candidate slots.
**Prevention:** a candidate-set integrity assertion in the request path's test harness that fails when rows ≠ distinct IDs.

### F-11 — Latency regresses during index refresh

**Symptom:** p99 latency doubles for about 10 minutes every rebuild, then recovers.
**Competing causes:** (a) the rebuild competes for the same CPU or memory bandwidth as serving; (b) serving swaps to a cold index whose pages are not resident; (c) the swap causes a burst of cache misses in a downstream eligibility cache; (d) the new index is physically larger and traversal cost rose permanently, with the "recovery" being traffic variation rather than a fix.
**Discriminating check:** compare p99 in the window after swap with p99 an hour later at equal traffic. No recovery means (d). Rising resident-set size after swap with page-fault counters elevated means (b). Rebuild CPU pinned to the serving hosts means (a).
**Correction:** rebuild off-cluster, warm the new index with a replay of recent queries before the swap, and keep the old index resident until the new one is warm.
**Prevention:** measure p99 during a synthetic swap in staging, and make the swap a distinct deployable step with its own acceptance criteria.

### F-12 — A popularity fallback becomes unsafe

**Symptom:** the fallback source, meant for cold start, serves 40% of traffic after a primary outage and starts surfacing policy-sensitive items.
**Competing causes:** (a) the fallback was built before a policy was introduced and never re-filtered; (b) the fallback's precomputed list was snapshotted at build time and is stale; (c) the fallback bypasses the final eligibility check because it is considered trusted.
**Discriminating check:** run the fallback's output through the same eligibility oracle used for the primary. Non-empty rejections point at (a) or (b); distinguishing them requires comparing the list's build timestamp with the policy's effective timestamp. A fallback whose output never reaches the final check is visible in the trace as a missing policy call.
**Correction:** route fallback output through the identical filter chain, and rebuild the list from current state rather than embedding eligibility decisions in it.
**Prevention:** treat every source as untrusted for eligibility; the final check is the only place that decides.

### F-13 — Recall calculation divides by zero for a cohort

**Symptom:** a cohort reports recall of 0.00 or 1.00 with perfect confidence; the pipeline looks either broken or perfect.
**Competing causes:** (a) the cohort's labels are immature so R is empty; (b) the cohort's label rule never fires because its surface does not emit the outcome event; (c) the cohort is genuinely outside the rights envelope so R ∩ E is empty.
**Discriminating check:** report the count of requests with |R ∩ E| = 0 and the reason. If the count is high and the horizon is recent, cause is (a). If the outcome event count is zero for the surface, cause is (b).
**Correction:** exclude undefined requests from the ratio, report the exclusion count and reason distribution, and never substitute a numeric placeholder.
**Prevention:** make the metric function return a discriminated union — `defined(value, n)` or `undefined(reason)` — so a silent zero is impossible in the type system.

### F-14 — Item update never reaches the index

**Symptom:** an edited title or price is correct on the detail page and wrong in recommendations for days.
**Competing causes:** (a) the index is rebuilt on a schedule and the update fell between rebuilds; (b) the update event is consumed by the feature store but not by the index pipeline; (c) the index stores a copy of mutable fields and serves them as truth.
**Discriminating check:** compare the update event's consumption offsets across the two pipelines. A gap in the index pipeline's offset points at (b). An index that stores mutable fields is visible in its schema, which excludes (c) as a separate cause only if the schema is immutable.
**Correction:** index only identifiers and embeddings; hydrate mutable attributes at request time from the authoritative store.
**Prevention:** design the index as a pure retrieval structure with immutable keys, and enforce it with a schema check in the build.

### F-15 — Position confounds a source comparison

**Symptom:** source X appears better than source Y in a slate where X is placed first.
**Competing causes:** (a) genuine quality difference; (b) position effect, since the first slot is examined far more; (c) allocation policy that places the higher-scoring source first by construction; (d) different candidate budgets per source so X simply had more chances.
**Discriminating check:** compare per-source outcomes at matched slots. If the advantage disappears at slot 3 and above, cause is (b) or (c). If X also had a larger candidate count, cause is (d).
**Correction:** randomise slot assignment for the comparison, or correct by measured examination propensities, and equalise the candidate budget per source.
**Prevention:** log rendered position and arm for every exposure; without position in the log, no such correction is possible after the fact.

### F-16 — Vector rebuild changes rankings unexpectedly

**Symptom:** after an embedding retrain with unchanged evaluation metrics, the top-10 slate overlap with the previous model is 40%.
**Competing causes:** (a) the new model is genuinely different and the offline metric is insensitive; (b) vector normalisation changed so cosine and inner product no longer agree; (c) the item encoder and the query encoder were updated in different deployments, producing mismatched spaces.
**Discriminating check:** compute [A1] fidelity between the new index and its own exact oracle. High fidelity with low cross-version overlap isolates the change to the embedding, not the index. Then compare a cosine-ranked list with an inner-product-ranked list on the same vectors: divergence points at (b). Check encoder version metadata on both sides for (c).
**Correction:** deploy encoder pairs atomically with a shared version identifier and refuse to serve mismatched versions.
**Prevention:** evaluate not only recall but also slate stability — the overlap of top-k between versions on a frozen query set — so a large behavioural change is visible before rollout.

### F-17 — Exposure logging is incomplete

**Symptom:** a surface shows 20 items but the exposure table records 14 on average.
**Competing causes:** (a) client-side logging fires only for items that enter the viewport, while the server renders all 20; (b) the logger samples at a rate that was not recorded; (c) the client drops events on backgrounding; (d) the server renders a fallback the logger does not know about.
**Discriminating check:** join server render decisions with client exposure events by request ID and count the deficit by slot index. A deficit concentrated in low slots points at (a). A uniform 30% deficit points at (b). A deficit correlated with session-end points at (c).
**Correction:** log the server render decision as the authoritative exposure record and use client events only for viewability.
**Prevention:** an exposure reconciliation job whose output is a required field in every evaluation report.

### F-18 — Long-tail items disappear from candidates

**Symptom:** the bottom 80% of the eligible catalog by historical engagement accounts for 4% of exposures.
**Competing causes:** (a) popularity and co-visitation sources structurally cannot propose tail items; (b) the ANN graph is disconnected for tail items because they had too few neighbours during construction; (c) the ranker demotes them because their features are sparse; (d) exploration is disabled.
**Discriminating check:** measure presence of tail items in the raw source output before ranking. Present in candidates but absent from slates points at (c). Absent from candidates while present in the exact oracle points at (b). Absent from every source's output points at (a).
**Correction:** add an explicit tail source with a quota, ensure the index build includes items ingested after the last full rebuild, and reserve exploration slots.
**Prevention:** track coverage of the eligible catalog as a first-class metric, since no recall metric computed against observed labels will reveal items that never had a chance to be observed.

### F-19 — A deleted user's data persists in a derived store

**Symptom:** a deletion request is honoured in the primary store and the user still receives personalised recommendations.
**Competing causes:** (a) the feature store's deletion is asynchronous and lagging; (b) the interaction matrix used by co-visitation is rebuilt nightly and contains the user's rows until the next build; (c) the recommendations are non-personalised but appear personalised because they are popular.
**Discriminating check:** trace the deletion identifier through each derived store by lookup. Presence in the nightly matrix points at (b). Presence in the feature store with a lagging deletion timestamp points at (a). Absence everywhere plus identical output for a control account points at (c).
**Correction:** make deletion a pipeline with per-store verification and a bound on propagation time, not a single primary-store operation.
**Prevention:** maintain a derived-store inventory with an owner per store, and a deletion rehearsal that runs on a schedule and reports per-store latency.

### F-20 — Candidate budget grows without a decision

**Symptom:** the candidate budget silently rises from 200 to 1,400 over a year through successive small increases, and p99 latency rises with it.
**Competing causes:** (a) each change was locally justified by a recall measurement at constant k; (b) the ranker's cost per candidate fell, making more candidates cheap; (c) k was raised for a different surface and inherited by all surfaces.
**Discriminating check:** plot k over time against the deploy log. Attribute each step to a change record. If steps coincide with ranker efficiency work, cause is (b) and the increase may be justified; if steps coincide with retrieval changes, cause is (a).
**Correction:** declare k as a policy constant per surface with a written justification, and require a latency measurement with every change.
**Prevention:** put k, depth, and the latency budget in one versioned configuration reviewed together, so a change to one cannot land without the others in view.

### F-21 — An apparently negative item was never displayed

**Symptom:** a model learns to suppress a class of items, and inspection shows those items have low label rates.
**Competing causes:** (a) genuine low preference; (b) the class was never exposed above slot 12 where examination is near zero, so its label rate reflects position, not preference; (c) the class is legitimately constrained to a surface whose outcome event is under-logged.
**Discriminating check:** compute the label rate conditioned on slot for the affected class. If the rate is comparable to other classes at the same slot, cause is (b) rather than (a).
**Correction:** train with exposure and position as features, or restrict training to exposures in examined slots, and never treat an unexposed item as a negative.
**Prevention:** include "was exposed in an examined slot" as a filter in the label pipeline, and report label-rate-by-slot as a standard diagnostic.

### F-22 — Underfill is masked by a fallback

**Symptom:** every request returns 20 items and no alert fires, yet the personalisation appears weak.
**Competing causes:** (a) the fallback is filling most slots because the primary retrieval is under-delivering; (b) the fallback is genuinely appropriate for the cohort mix; (c) both are true for different cohorts.
**Discriminating check:** report candidate-set composition by originating source per request. A fallback share above roughly half means the primary path is not delivering, which is cause (a) whether or not the resulting slate looks acceptable.
**Correction:** alert on primary-path contribution, not on slate fill; a filled slate is not evidence that retrieval worked.
**Prevention:** make "slots filled by source" a required breakdown in the on-call dashboard, so a healthy-looking fill rate cannot hide an outage.

### 8.2 Measurement catalog

Each row is a proposed measurement. None of the thresholds are universal; they are starting points to be calibrated against a measured baseline. Rates require nonzero denominators and a declared window.

| Metric | Numerator | Denominator | What it detects | Benign explanation |
|---|---|---|---|---|
| Eligible catalog size | distinct IDs passing all predicates | request | segment starvation and policy over-reach | a genuinely narrow catalog in a small territory |
| Candidate recall@k | relevant eligible IDs in C_k | relevant eligible IDs | retrieval ceiling | an immature label horizon inflating the denominator |
| Pooled recall | hits summed over requests | relevant pairs summed | inventory-level recovery | a few heavy-interest requests dominating |
| Macro recall | per-request recall summed | requests with non-empty R | typical-request quality | tiny denominators making each request near 0 or 1 |
| Underfill rate | requests with fewer than k candidates | requests | depth provisioning and filter placement | catalog smaller than k |
| Primary-source share | candidates from the primary path | all candidates | hidden outages masked by fallback | a cohort that legitimately has no history |
| Rights rejection rate | candidates rejected for rights | candidates checked | territory misconfiguration | expected geography |
| Availability rejection rate | candidates rejected for stock | candidates checked | inventory lag versus index lag | genuine stock volatility |
| Authorization error rate | checks returning an error | checks attempted | fail-closed behaviour under dependency failure | a known dependency blip with no exposure |
| ANN fidelity@k | exact top-k IDs recovered | k, or exact list length | index configuration drift | tie-heavy score distributions |
| Depth-to-keep ratio | requested depth | delivered candidates | depth mis-provisioning | a dense cohort where depth is generous |
| Slate stability | top-k overlap with previous version | k | behaviour change from a model swap | an intended exploration increase |
| Exposure concentration | Gini over per-item exposure | — | feedback-loop narrowing | a head-heavy catalog with genuine demand |
| Coverage of eligible catalog | distinct items exposed | distinct eligible items | tail starvation | short measurement window |
| Label rate by slot | labelled outcomes at slot s | exposures at slot s | position confounding | a genuinely better item at slot 1 |
| Exposure reconciliation | client exposure events | server render decisions | logging loss | viewport-based logging by design |
| Primary-path latency p99 | request duration at p99 | — | depth or index regression | a rebuild window |
| Fallback activation rate | requests served with any fallback | requests | dependency health | a new-cohort influx |

### 8.3 Point-in-time and serving acceptance traces

These are constructed test cases for a deployment gate, not records of executed tests. Each names the behaviour that must hold and the wrong behaviour the trace is designed to catch. A trace that cannot fail is not a test.

| Trace | Setup | Required behaviour | Wrong behaviour it catches |
|---|---|---|---|
| Rights revoked mid-request | revoke between retrieval and render | item omitted; decision logged with the new policy version | cached allow replayed |
| Territory change between sessions | viewer attribute changes at t | next request uses new territory; no stale slate | territory cached at profile level |
| Entitlement downgrade | subscription ends before request | premium items excluded in the same request | slate cached across tier change |
| Policy service timeout | inject a controlled timeout | deny; no unchecked item rendered | timeout treated as allow |
| Inventory reaches zero | stock hits zero after retrieval | item omitted or shown as unavailable per policy | stale positive inventory served |
| Eligibility predicate empty | all items fail by construction | honest empty or editorial response | ineligible substitute injected |
| Global ANN underfill | depth 20 with density 0.02 | underfill reported, fallback invoked | silent 3-item candidate set passed to ranker |
| Segment density halved | territory density 0.01 | depth recomputed; underfill rate stays bounded | fixed depth retained, underfill doubles |
| Tie-heavy score boundary | 40 items share the k-th score | deterministic tie rule reproduced exactly | order-dependent output across replicas |
| Embedding version skew | query and item encoder versions differ | request refused; version mismatch logged | silent quality loss |
| Duplicate across sources | same ID from three sources | one candidate row | three rows consuming budget |
| Concept cap | six variants of one product | at most the declared per-concept cap | one concept filling the slate |
| Deleted account | deletion propagated | no personalised candidates | derived store still serving |
| Index swap | new index promoted | p99 within budget; fidelity report present | cold index served unmeasured |
| Exposure logging loss | client drops 30% of events | reconciliation flags the deficit | analysis run on partial exposures |
| Label horizon not matured | request inside the horizon | excluded from recall with a reason code | request counted as zero recall |

**Takeaway.** Diagnosis is a comparison, not an inspection: every entry above ends in a check that produces a number separating at least two hypotheses. A failure mode without such a check is a suspicion, and a runbook built from suspicions sends responders to the wrong service.
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
## 12. Quick Reference Card

| Item | Value and condition |
|---|---|
| Core decision | Trust a candidate pipeline only when eligibility is checked at decision time, candidate recall is measured on a matched cohort, and underfill plus latency are reported alongside it |
| Candidate recall | `|C_k ∩ R| / |R ∩ E|`, defined only when the eligible labelled count is positive; the denominator uses the eligible universe, never the whole catalog |
| Aggregates | macro weights requests equally; pooled (identical to relevant-count weighting) weights by relevant-item count; declare the choice before seeing results |
| Depth requirement | smallest m with `P(Binomial(m, p) ≥ k) ≥ 1 − α`; p = 0.02, k = 20, α = 0.05 gives m = 1390 (ratio 69.5); p = 0.08 gives m = 344 (ratio 17.2) |
| Eligibility attrition | `1 − r1 − r2 + (r1·r2 + ω)`; positive co-rejection ω raises joint survival, so independence under-estimates the surviving set |
| Approval gate | fail closed on authorization, push eligibility into traversal or partition the index, and keep eligibility decisions out of the index so staleness affects quality only |
| Cost anchors | exact scan is bandwidth-bound (6.144 GB per request at 12M × 128 float32); ANN at depth 1390 touches ~742 KiB; ranker cost scales with candidate count at roughly 2.6 ms per candidate |
| Use when | a ranker or slate budget limits how many items can be scored and eligibility is a nontrivial share of the catalog |
| Avoid when | the eligible labelled population cannot be assembled point-in-time, the label horizon is immature, or the comparison differs in k, H, or eligibility vintage |
| Signature failure | a full slate with weak personalisation: fallback rows fill the slots because the primary path under-delivers, and slate fill hides it |
