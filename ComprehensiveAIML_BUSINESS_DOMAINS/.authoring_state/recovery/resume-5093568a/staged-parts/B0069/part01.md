---
title: "Candidate Generation, Eligibility and Ranking"
topic: "Two-stage retrieval and ranking under point-in-time eligibility constraints"
collection: "recsys-playbook"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Candidate Generation, Eligibility and Ranking

## 1. File Header

**Purpose:** Give an engineer or analyst a decision procedure for building a two-stage recommender in which every displayed item is authorized at the moment of display and every reported retrieval number has a defensible numerator and denominator.
**Audience:** Recommendation engineers, applied scientists, product analysts, and candidates preparing system-design or metrics interviews.
**Prerequisites:** Set algebra, dot products and cosine similarity, conditional probability, joining event tables on time, and controlled experiment design. The [Stanford IR book](https://nlp.stanford.edu/IR-book/) is the standard reference for the ranking-metric half of this card.
**Outcome:** Given a point-in-time catalog snapshot, an authorization rule, relevance labels, and the raw outputs of several retrieval sources, you can compute matched recall, bound the recall ceiling implied by a candidate budget, price an over-retrieval multiplier, and choose where in the pipeline each eligibility filter must live.
**Excluded:** Learning-to-rank loss functions, counterfactual policy optimization, and statutory or contractual interpretation of what may be shown in a jurisdiction.
**Status:** Draft and unreviewed. No latency, revenue, engagement or quality result reported here was measured on a production system.
**Evidence:** DERIVED for the algebra and the probability statements, ILLUSTRATIVE for every constructed dataset, SOURCED for the two external references, OPEN for all operational claims about specific vendors or index configurations.
**Source boundary:** The unit's topic crosswalk names candidate generation, eligibility and ranking, but the original source files behind that crosswalk were not available for inspection; nothing here is a quotation from them.

### 1.1 How to read this card

Start at Section 4 if you need the three quantities you will actually put in a metrics table: candidate recall, matched recall at display, and approximate-index fidelity.
Read Section 5 before quoting any percentage, because it shows the same toy request scored three ways and the three answers differ by more than the effect sizes teams usually argue about.
Go to Section 8 when an alert fires, Section 9 before promising a latency or infrastructure budget, and Section 12 when you need the whole card on one screen.

### 1.2 Notation conventions used throughout

| Symbol | Meaning | Units / type | Valid range |
|---|---|---|---|
| I(t) | Catalog item identifiers known at instant t | finite set of IDs | may be empty |
| E(q,t) | Items permitted for request q at instant t | subset of I(t) | may be empty |
| R(q,H) | Eligible items that satisfy the declared relevance rule within horizon H | subset of E | may be empty |
| C_K(q) | Unique items returned by candidate generation, budget K | ordered list, length ≤ K | K ≥ 1 |
| S_n(q) | Displayed slate of length n after constraints | ordered list | n ≤ K |
| A_K(q) | Approximate nearest-neighbour output under the same score and universe | ordered list | may underfill |
| X_K(q) | Exact top-K under that score when every eligible item is scored | ordered list | requires full scan |

Two conventions remove most ambiguity before any number is computed. First, a **counterfactual** quantity is written with a tilde, as in the recall the pipeline would have achieved if the ranker had seen the whole eligible set. Second, every rate is written as `numerator / denominator` in the same sentence as its value, because a bare percentage in a slide deck is the single most common source of a wrong comparison in this domain.

### 1.3 Takeaway of the whole card

The candidate stage decides the maximum quality the product can reach, and the eligibility stage decides whether that quality is allowed to be shown; ranking only redistributes quality among items that already survived both tests.

## 2. Concept Map / Mental Model

```text
                          request q, context, decision timestamp t
                                        |
                                        v
        +----------------- authorization / rights / safety snapshot E(q,t) -------------+
        |                                                                                |
        v                                                                                |
  popularity pool     collaborative pool     content pool     dense ANN pool             |
        |                    |                    |                  |                    |
        +--------------------+---------+----------+------------------+                    |
                                        |                                                |
                                        v                                                |
                     union -> dedupe by item id -> filter to E(q,t)                     |
                                        |                                                |
                                        v                                                |
                         bounded candidate list C_K  (budget K)  -----------------------+
                                        |
                                        v
                        ranker over C_K  ->  constrained slate S_n
                                        |
                                        v
                    final eligibility check at display time (rights, stock, safety)
                                        |
                                        v
             impression log (item, position, model version, policy version, request id)
                                        |
                                        v
             outcome window H -> labels -> evaluation -> back to retrieval tuning
```

The diagram is a data-flow sketch, not a deployment topology. The last edge is a feedback loop and therefore a leakage risk, not a causal claim: labels produced by the current policy differ in distribution from labels the next policy would produce.

### 2.1 The three-stage contract

| Stage | Consumes | Produces | Guarantee it can offer | Guarantee it cannot offer |
|---|---|---|---|---|
| Eligibility | I(t), entitlement rules, availability, safety policy | E(q,t) | Every retained ID is permitted at t | That any retained item is interesting |
| Candidate generation | E(q,t), user and item features, budgets | C_K(q) | Recall of R relative to a declared score family, at cost bounded by K | That the top of C_K is the best order |
| Ranking plus display | C_K(q), learned scores, slate constraints | S_n(q) | An order under the trained objective, subject to declared constraints | That items outside C_K were handled fairly |

Read the table as an ownership map. A recall regression belongs to the candidate owner, a relevance regression at fixed candidate set belongs to the ranking owner, and any appearance of a non-eligible item belongs to whoever owns the final check, regardless of which stage leaked it.

### 2.2 Why the funnel shape is forced

Scoring every eligible item with a cross-feature ranker costs one forward pass per item. At 40,000 eligible items, a 3 ms per-batch model that handles 256 items per batch needs 157 sequential batches, so even at 1 ms per batch the ranker alone would consume more than the request budget. Candidate generation exists to convert a set of unknown size into a sorted list of known size, so that the expensive model is applied a fixed number of times per request.

The consequence is a **quality ceiling** that no downstream model can lift. If candidate generation recovers only 55 percent of the relevant eligible items at K = 10, the best possible display recall is 0.55, and a ranker that improves ordering moves display recall inside that ceiling but never above it.

### 2.3 Where errors actually enter

Errors enter at four seams, and each seam has a distinct signature. At the **snapshot seam**, the retrieval index and the authorization table are read at different times, so a revoked item is retrieved and then rejected, or worse is retrieved and not rejected. At the **union seam**, the same item arrives from several sources with incomparable scores, so deduplication silently keeps the wrong score. At the **budget seam**, a selective filter removes most of a fixed-size list, so the display underfills. At the **logging seam**, the impression record omits the policy version, so no later audit can reconstruct why an item was shown.

### 2.4 Mental model to carry into review

Treat the eligible set as a legal playing field, the candidate list as a shortlist with a documented admission rule, and the slate as a constrained selection from that shortlist. Then three questions become answerable in a design review: what fraction of the field did the shortlist admit, what fraction of the shortlist did the slate use, and what did each stage discard and why.

## 3. Foundational Definitions

| Term | Meaning and operational test | Units / type | Confusion to avoid |
|---|---|---|---|
| Request | One opportunity to recommend, identified by request id, surface, user, and decision timestamp | event record | A page view may contain several independent request slots |
| Catalog snapshot | Item IDs and attributes readable at instant t | set plus attribute rows | A fresh build can still embed stale attributes |
| Entitlement | A rule mapping user and item to permitted or denied at a time | predicate | Entitlement is not the same as availability |
| Availability | Whether an item can actually be fulfilled or streamed now | boolean plus reason code | Availability changes faster than entitlement |
| Eligibility | Entitlement and availability and safety constraints all satisfied | subset of catalog | Retrieval does not grant eligibility |
| Relevance label | A rule-derived judgement that an eligible item would have satisfied the user in horizon H | boolean or graded | Unobserved is not negative |
| Candidate budget K | Maximum unique IDs admitted to ranking | positive integer | Not the displayed slate length |
| Slate length n | Number of items actually rendered | positive integer | n can be smaller than K after constraints |
| Retrieval source | An independent producer of scored candidates | component | Sources do not share a score scale |
| Deduplication key | The field used to collapse repeated candidate rows | identifier field | Deduplicating on title merges distinct items |
| Fusion score | The number used to order the union of sources | real number | Rank-based and score-based fusion are different estimands |
| Candidate recall | Share of eligible relevant items present in C_K | ratio in [0,1] | Not the same as display recall |
| Display recall | Share of eligible relevant items present in the slate | ratio in [0,1] | Bounded above by candidate recall |
| Rank capture | Share of the candidate list's relevant items that reached the slate | ratio in [0,1] | Isolates ranking loss from retrieval loss |
| ANN fidelity | Share of exact top-K IDs recovered by an approximate index | ratio in [0,1] | Independent of whether those IDs are relevant |
| Underfill | A slate shorter than requested after filtering | count or rate | Distinct from an empty catalog |
| Over-retrieval multiplier | Ratio of requested candidates to desired post-filter survivors | dimensionless | Depends on the filter selectivity distribution |
| Exposure | An item rendered with a recorded position and policy | event | A retrieved but unrendered item has no exposure |
| Position bias | The dependence of interaction probability on display position | multiplicative factor by slot | Confounded with item quality |
| Freshness lag | Delay from a source change to its visibility in retrieval | duration | Index age is only one component |
| Tombstone | A deletion marker propagated through index and caches | record | Absence from a build is not a tombstone |
| Coverage | Share of catalog items that can be retrieved at all | ratio | High coverage can coexist with low relevance |
| Diversity | Declared spread of the slate across an attribute | metric-dependent | Not automatically a user benefit |
| Guardrail | A monitored bound that a change must not violate | threshold plus window | A click lift does not establish compliance |
| Slate constraint | A rule applied during selection, such as a per-genre cap | predicate over list prefixes | Applied after ranking it can invert scores |
| Counterfactual ceiling | Best attainable metric given the candidate list and constraints | ratio or score | Useful for attributing loss between stages |
| Leakage | Information unavailable at decision time entering training or evaluation | defect class | Also occurs through label windows, not only features |
| Policy version | Identifier of the rules and models active for a request | string | Missing from many impression logs |

Three definitions carry disproportionate diagnostic weight. **Eligibility** must be evaluated from the same snapshot the retrieval stage used, otherwise a recall number mixes an admission decision with a permission decision. **Relevance** must name its observation mechanism, because a click-based label and a completion-based label on the same item are different targets. **Candidate budget** must be recorded per request, because a system that silently drops to K = 40 on a slow path and K = 400 on a fast path produces a metric mixture that no single average describes.

A fourth distinction prevents a common argument in review. Coverage and relevance trade off against each other through popularity: the most-covered item set is whatever the popularity counter favours, and popularity counters concentrate. A catalog with 96 percent coverage can still leave most items with fewer than one exposure per thousand requests, which means their labels are too sparse for any model that relies on interaction history.
