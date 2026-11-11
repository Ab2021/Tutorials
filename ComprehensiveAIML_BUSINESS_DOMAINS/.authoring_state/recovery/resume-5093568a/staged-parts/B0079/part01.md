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
