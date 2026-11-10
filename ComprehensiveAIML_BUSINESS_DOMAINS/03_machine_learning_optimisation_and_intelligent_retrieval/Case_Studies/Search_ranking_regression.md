---
case_id: "CASE-M14-02"
case_title: "Diagnosing a part-number search ranking regression before rolling back"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "draft"
content_status: "draft"
source_status: "OPEN"
case_type: "incident"
domain: "Search — enterprise product-catalogue retrieval and learning-to-rank"
primary_pattern: "retrieval-ranking quality incident"
secondary_patterns: ["observability", "stratified offline evaluation"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "Search relevance lead (with ranking platform and merchandising operations)"
source_index: "INDEX.md"
review_trigger: "Any change to the query-intent taxonomy, the judged relevance set, the click-attribution schema, or the ranker feature contract"
implementation_stage: "proposed"
---

# Diagnosing a part-number search ranking regression before rolling back

## 1. The One-Line Framing

Given seven days of search sessions, ranking-model artefacts, index snapshots and a frozen judged-relevance set, decide whether to roll back a newly deployed learning-to-rank model, roll back the index rebuild that shipped beside it, or ship a targeted patch — for a search relevance lead who must restore part-number lookup quality for procurement buyers, subject to a two-hour change window, an exact-match correctness expectation, and a frozen quarterly catalogue, compared with the previous production ranker and index.

The task category is incident diagnosis and release reversal in a retrieval system; the decision recurs at every model or index deployment, and it is made under time pressure roughly a few times per quarter. The primary success condition is a restore of part-number query quality without regressing the exploratory query population that dominates traffic. The primary pattern is the retrieval-ranking quality incident; the two secondary patterns are observability (can the regression be seen at all, and how fast) and stratified offline evaluation (does the headline number represent the population that was hurt).

**Source declaration and evidence basis.** Every organisation name, catalogue size, traffic volume, click count, support-contact rate, cost figure, model version, feature name, timestamp and outcome in this document is a constructed teaching value. The scenario is illustrative; it is not an account of an incident at any company, and no vendor, cloud provider or model family is being characterised. All arithmetic has been recomputed by hand and is internally consistent, but consistent arithmetic over invented inputs proves only that the model of the world is self-consistent, not that any real system behaved this way. Labels used throughout: **ILLUSTRATIVE** for constructed scenario inputs, **DERIVED** for arithmetic computed from those inputs, **OPEN** for questions this document deliberately does not settle. No measurement of a real deployment is claimed anywhere, and Section 11 reports constructed post-hoc values explicitly as illustrative recovery figures rather than as realised impact.

Three things make the obvious framing wrong. First, the headline offline metric barely moved, so a team that trusts the aggregate number will conclude nothing happened; the damage is concentrated in a query class that is 4.1% of traffic but disproportionately high-intent. Second, two artefacts shipped in the same release train — a ranker retrain and an index rebuild — and a rollback decision that treats them as one unit destroys the evidence needed to tell which one caused the harm. Third, the ranker was not wrong about what it was trained to do: it was trained on clicks, and the click log had been silently mislabelling accessory clicks as organic clicks on the correct result for twelve percent of impressions. The model faithfully learned a corrupted preference. That last point is the real reason this incident took six days to detect and one hour to misdiagnose.

## 2. Context & Background

The setting is an illustrative business-to-business distributor of industrial components. Buyers are maintenance engineers, procurement officers and repair-shop staff. The catalogue holds 1.2 million stock-keeping units, of which roughly 380,000 are active in any quarter; the remainder are superseded, discontinued, or seasonal. A single search box in the web storefront and in a mobile parts-lookup tool is the primary navigation path: buyers arrive knowing either an exact manufacturer part number (MPN), a partial specification, or a plain-language description of a broken component.

The workflow around search is short and unforgiving. A buyer types a query, scans the first screen of ten results, and either clicks a part, refines the query, or leaves. Clicking a part opens a product detail page with fitment notes, stock by warehouse, price breaks and lead time; from there the buyer either adds to a cart or requests a quote. There is no second search system and no human intermediary for self-service buyers. Telephone and chat buyers reach a parts counter agent who performs the same lookup in an internal tool that uses the same index through a different front end.

Stakeholders are distinct and their incentives diverge. The search relevance lead owns ranking quality and the evaluation harness. The ranking platform team owns the training pipeline, feature store and serving path. Merchandising operations owns catalogue attributes, including the MPN field and its normalisation rules. The parts-counter operations manager owns agent productivity and the escalation queue. Finance owns the contact-cost model. Nobody owns the *join* between click logs and result modules, which is exactly where this incident lives.

**Baseline process.** For two years the production stack has been a two-stage retrievers-plus-ranker design. Stage one is a lexical retrieval over an inverted index using BM25 with field boosts across `title`, `mpn`, `description` and `category_path`; it returns the top 100 candidates by score. Stage two is a gradient-boosted learning-to-rank model over 74 features — lexical scores, exact-match indicators, popularity priors, price and availability signals, and catalogue-quality signals — which produces the final ordering of those 100 candidates. Retrieval is deliberately generous; the ranker does the discrimination.

The baseline's documented weaknesses, present before the incident, are worth naming because they shaped the diagnosis. First, the ranker is trained on clicks with no position-bias correction, so it inherits whatever the previous ranker surfaced. Second, the evaluation set of judged queries was assembled in a single curation sprint and has been reused for every model selection decision since, so its effective freshness is unknown. Third, the freshness monitoring is weekly and aggregate, so a slice-specific regression is invisible until someone goes looking. None of these was introduced by the release under investigation; the release merely converted three latent weaknesses into one visible failure.

**Scale and timing.** The illustrative traffic assumption is 480,000 search queries per day across both front ends, of which 4.1% — about 19,680 per day — are exact or near-exact part-number lookups. Peak arrives on weekday mornings between 07:00 and 10:00 local time in the primary market, when maintenance crews plan the day's work. The catalogue is frozen for the quarter except for price and stock updates, which matters because it removes "the catalogue changed underneath us" as a full explanation while still leaving attribute-level edits possible.

The release train in question shipped on Tuesday 2026-10-13 at 02:10 UTC during a low-traffic window. It contained two independently versioned artefacts: ranker `v7`, retrained on ninety days of clicks with a new hard-negative sampling scheme, and search index `v3`, rebuilt with a revised field-analyser configuration on the `mpn` field and a revised stopword list on `title`. The release notes described index `v3` as "size reduction, no ranking intent"; that sentence was written by an engineer who had tested it on twelve hand-picked queries.

**Prior attempts and organisational dependencies.** Two earlier attempts to improve part-number search existed and both had been abandoned, which is relevant because their abandonment shaped institutional priors. A synonym expansion experiment had been shelved because it produced spurious matches on competitor part numbers. A "did you mean" spelling correction had been shelved because the MPN space contains genuine near-duplicates where a correction is worse than a miss. The organisation therefore believed part-number search was "basically solved by exact matching", which made a regression in that area feel implausible and slowed recognition.

No regulatory obligation is asserted here. The relevant requirements are internal: an exact-match expectation for MPN queries that merchandising treats as a product promise, a two-hour change window negotiated with storefront operations, and a change-freeze during the quarterly catalogue lock except for incidents.

## 3. The Business Question

**Illustrative stakeholder question (paraphrase of the relevance lead's framing, not a quotation from any person):** "Part-number searches are landing on the wrong page and the agents are complaining. Our dashboard says relevance is flat. Do we roll the whole release back tonight, roll back only the ranker, or wait for a fix — and how do I know which of the two artefacts is actually responsible?"

| Decision contract field | Content |
|---|---|
| Question | Which artefact in the 2026-10-13 release train caused the part-number quality loss, and what is the smallest reversal that restores it without harming the dominant query population? |
| Decision owner | Search relevance lead, with a required concurrence from the ranking platform lead for any model rollback and from merchandising operations for any index rollback |
| Trigger and deadline | Support-contact escalation from the parts-counter operations manager on 2026-10-19; decision required before the 02:00 UTC change window on 2026-10-20 |
| Available actions | (a) full rollback of ranker `v7` and index `v3`; (b) ranker-only rollback to `v6`, keep index `v3`; (c) index-only rollback to `v2`, keep ranker `v7`; (d) keep both and ship a targeted patch; (e) no change and continue monitoring |
| Value and downside | Restoration protects high-intent sessions and agent productivity; a wrong rollback discards a genuine improvement in the exploratory population and consumes the only change window before the weekend |
| Objective | Maximise judged relevance on the part-number query stratum, subject to not degrading the exploratory stratum beyond its noise band |
| Hard constraints | Two-hour change window; no schema migration during the window; exact-match results must not be demoted below rank 3 for an exact MPN hit; no change to the catalogue during the freeze |
| Scope | Self-service storefront search and the parts-counter internal tool, both front ends, primary market, the seven days from 2026-10-13 to 2026-10-19 |
| Out of scope | Price and availability ranking, category-page browse, recommendation modules, quote workflow, mobile app release cadence |

The business objective and the proxy metric diverge in a specific and predictable way, and that divergence must be stated before any number is quoted. The business objective is "the buyer finds the part they need and completes the task". The measurable proxy is graded relevance of the results shown, aggregated as a normalised discounted cumulative gain over the top ten. A buyer can find the right part through a result we judged irrelevant (because the judgement is stale or the buyer's intent was different from the judged intent), and a buyer can fail to find the right part from a result we judged highly relevant (because the price or lead time was unacceptable). The guardrail that keeps the two aligned is the exact-match constraint in the hard-constraints row: whatever the aggregate says, an exact part-number hit may not sit below rank three. That guardrail is a business rule, not a statistical one, and it is the rule the release violated.

A note on what an expensive false decision looks like in each direction, because this is what makes the incident interesting rather than routine. A false positive — declaring a regression and rolling back — costs one change window, loses seven days of whatever the release actually improved, and teaches the organisation that releasing is dangerous, which slows future work. A false negative — declaring the dashboard is right and doing nothing — costs continued misrouting of high-intent buyers, agent time, and eventually credibility of the search team with the operations organisation. The asymmetry is real but not one-sided, and it is why the decision must be *discriminating* rather than merely *responsive*: the goal is not "revert something", it is "revert the thing that caused it".

## 4. Assumptions & Constraints

The distinction that matters in this section is between what the team believed and what the team had checked. Six days into the incident, four of the assumptions below were unverified beliefs held by competent people, and two of them were wrong. The table records the belief as it stood, the check that would settle it, and the consequence of being wrong, so that a reader can see which beliefs the decision actually depended on.

| ID | Assumption or constraint | Type | Evidence or verification plan | Status at decision time | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | The two shipped artefacts can be rolled back independently | Infrastructure | Inspect release manifest and feature-contract compatibility | Verified — separate version strings, separate registries | A combined rollback destroys the ability to attribute cause | Roll both, accept slower diagnosis | Any change to the release-train bundling policy / platform lead |
| A-02 | Ranker `v6` is feature-compatible with index `v3` | Data contract | Diff the 74 feature definitions against index field names | **Violated** — one feature renamed | Rollback to `v6` fails or silently returns zeros for one feature | Ship a compatibility shim, or roll back both | Feature-store schema change / ranking platform lead |
| A-03 | Click logs attribute a click to the result position the user clicked | Data | Inspect the impression schema for a module identifier | **Violated** for 12% of impressions | Training labels are contaminated; the model learned a real but wrong preference | Exclude contaminated impressions, retrain | Any change to the impression schema / analytics engineering |
| A-04 | The judged set represents traffic | Measurement | Compare judged query-class mix with production query-class mix | Violated in proportion, sound in direction — the set deliberately oversamples part-number queries | Headline metric dilutes the effect; the set is still usable for the stratum | Report stratified metrics, never the aggregate alone | Quarterly traffic-mix review / relevance lead |
| A-05 | The judged set is adequate for repeated model selection | Measurement | Count distinct model versions evaluated against it | Violated — 14 selection rounds against the same 20,000 queries | Selection overfitting inflates offline gains and hides small regressions | Hold out a locked confirmation subset | Any new selection round / relevance lead |
| A-06 | Overall relevance did not change materially | Measurement | Paired comparison on the aggregate judged set | Verified arithmetically — the aggregate moved by −0.008 | Team concludes nothing happened | Stratify before concluding | Nightly stratified job / relevance lead |
| A-07 | The support-contact rise is attributable to search | Business | Compare contact reasons against query class and release date | Unverified correlation | Cost of delay is overstated | Treat as directional only | Contact-reason taxonomy review / operations |
| A-08 | Traffic mix is stable across the release week | Data | Compare query-class shares week over week | Verified within one percentage point | A mix shift could imitate a quality regression | Recompute on a fixed query sample | Seasonal boundary / analytics |
| A-09 | Result caching does not mask a rollback | Systems | Inspect cache key composition and TTL | **Violated** — cache keyed on query text only, six-hour TTL | Rollback appears ineffective for popular queries | Purge cache on deploy | Cache policy change / platform lead |
| A-10 | Exploratory queries were unaffected | Measurement | Paired comparison on the exploratory stratum | Verified — +0.001, inside the noise band | A ranker-only rollback would be judged harmful | Restrict the rollback to the affected stratum if possible | Next release / relevance lead |
| C-01 | Two-hour change window, no schema migration inside it | Operational | Change-management record | Confirmed | A shim must be code-only, not a migration | Defer to the following window | Change policy / operations |
| C-02 | Catalogue frozen for the quarter | Business | Catalogue change log | Confirmed for attributes, not for price and stock | A catalogue edit could confound the comparison | Diff attribute versions across the window | Catalogue unlock / merchandising |
| C-03 | Exact MPN hits must not fall below rank 3 | Product | Merchandising product promise | Confirmed | The release violated a stated product promise even where NDCG looked tolerable | Enforce as a monotone constraint | Product review / merchandising |
| C-04 | No third-party ranking vendor change during the window | Supplier | Vendor changelog | Confirmed by absence of notification, not by verified changelog read | An external change would imitate an internal regression | Ask the vendor, escalate | Renewal or upgrade / platform lead |
| C-05 | Only the search team may change ranker weights | Security | Access review | Confirmed | An unauthorised tuning change would be invisible in the release log | Audit configuration history | Access review cycle / security |

The two assumptions most likely to reverse the conclusion are `A-01` and `A-03`. If the artefacts could not be separated, the diagnosis would have to be abandoned in favour of a blunt rollback, and the team would learn nothing. If the click attribution had been sound, then the ranker's learned preference would be evidence that buyers genuinely prefer accessories over exact parts, and the "regression" would be a product disagreement rather than a bug — a profoundly different decision with a profoundly different owner. That second possibility is the one a careless diagnosis would miss, and Section 6 shows the check that ruled it out.

A third assumption almost reversed the *response* rather than the conclusion: `A-09`, the cache. It did not change the diagnosis, but it changed the verification plan, and it is the reason a naive rollback would have been declared a failure ninety minutes after it was executed. Section 12 treats it as a distinct failure mode with its own discriminating test.

Two exclusions deserve explicit statement because a reader will look for them. First, no statistical power analysis was done on the judged set for the part-number stratum before the incident; the 95% interval quoted later is computed after the fact, and the decision rule was not pre-registered. Second, no A/B experiment was run during the incident: the storefront has experiment capability, but running a relevance experiment on the affected population while buyers were actively failing would have been an experiment on customers rather than on traffic, and operations vetoed it. That veto is a real constraint on the evidence, not a footnote.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis differs across the four datasets used here, and conflating them is the most common analytical error in ranking-investigation work. A **query** is a distinct text string. A **session** is one buyer's contiguous interaction with the search box, possibly containing several queries. An **impression** is one rendered result within one query's result list, identified by query, result, position and module. A **click** is an interaction event attached to exactly one impression. The judged relevance set has yet another grain: one graded label per (judged query, catalogue item) pair, produced by a human assessor.

| Field | Specification |
|---|---|
| Unit of analysis | Primary: query-session. Secondary: impression (for click attribution) and query-item pair (for judgements) |
| Volume and window | 480,000 queries/day across both front ends; 3,360,000 over the seven-day window 2026-10-13T02:10Z to 2026-10-19T23:59Z. Judged set: 20,000 distinct queries, 340,000 graded query-item pairs, curated 2026-03 and not re-curated since |
| Inputs | `query_text` (raw, utf-8, case preserved), `front_end` ∈ {storefront, counter_tool}, `locale`, `session_id`, `query_timestamp` (UTC instant), `index_version`, `ranker_version`, plus the full 74-feature serving vector logged per candidate |
| Outcome / label | Graded relevance 0–4 per (query, item) pair, adjudicated by a single senior assessor with a written rubric; secondary label derived from click logs |
| Population and exclusions | Excluded: bot traffic by user-agent and by session-velocity rule, internal staff sessions by a known IP range, and sessions where the search service returned an error. Included: all other sessions regardless of outcome |
| Join semantics | Impression→click joined on `impression_id`, which is generated server-side and must be unique per rendered result. Clicks with a null `impression_id` are unattributable and are dropped |
| Splits | Judged set originally a single pool; for this investigation split retrospectively into 16,000 tuning queries and 4,000 locked confirmation queries, stratified by query class |

**Graded relevance rubric.** The rubric is the load-bearing artefact and it is worth writing out because the whole incident turns on a distinction the rubric already encoded. Level 4 is the exact part the buyer asked for, identified by exact MPN string match after normalisation. Level 3 is a correct part with the same manufacturer and specification but a different variant code — a legitimate substitute where fitment is compatible. Level 2 is a functionally equivalent alternative from another manufacturer. Level 1 is the right category, wrong specification — a plausible-looking but unusable result. Level 0 is off-catalogue, a competitor product referenced in a description, or unintelligible.

The rubric's existence at level 4 is what makes the regression measurable at all. Without a rubric that distinguishes "exact part" from "compatible substitute", the release's damage would have been scored as a mild relevance dip rather than a violation of a stated product promise. The rubric was written by the relevance lead in 2026-03 for an unrelated model-selection exercise; its reuse here is accidental but decisive.

**Why the judged set under-represents part-number queries.** The judged set was assembled by drawing one query per distinct query string from a twelve-week log, with a per-category cap, and de-duplicating by normalised text. That procedure samples *strings*, not *impressions*, and the two populations differ enormously. Exploratory queries repeat: the top 5,000 exploratory strings account for 61% of exploratory impressions, so a string sample captures the head well. Part-number queries barely repeat at the string level: roughly 3,100 distinct MPN strings appear on a typical day against 19,680 part-number impressions, because each buyer looks up a different part. String-sampling therefore gives part-number queries a share of about 2.0% of the judged set, against 4.1% of production impressions.

| Stratum | Production share of queries | Judged-set share | Judged queries (n) | Sampling ratio |
|---|---:|---:|---:|---:|
| `part_number` — exact or near-exact MPN | 4.1% | 2.0% | 400 | 0.49× |
| `exploratory` — plain-language need description | 72.9% | 73.0% | 14,600 | 1.00× |
| `specification` — attribute/fitment search | 23.0% | 25.0% | 5,000 | 1.09× |
| **Total** | **100%** | **100%** | **20,000** | — |

This table is the single most important artefact in the case, because it explains mechanically why the aggregate dashboard was flat while the operation was on fire. A 0.49× sampling ratio halves the weight of the stratum relative to impressions, and since the stratum's regression was severe, halving its weight halves the visible aggregate movement — while the noise the stratum contributes per query is unchanged. **ILLUSTRATIVE** throughout.

**Click log and the contamination.** The click log schema records `impression_id`, `query_id`, `result_id`, `position`, `module`, `click_timestamp` and `dwell_ms`. In 2026-08 a `module` value of `accessory_carousel` was introduced for a cross-sell strip rendered below the organic list. The front-end instrumentation for that module passed `position` from the carousel's own internal index (1, 2, 3) but failed to pass `module`, so carousel clicks were logged with `module = null` and a position that collided with organic positions 1–3. Downstream, the training pipeline selected clicks where `module is null` as organic clicks. Carousel impressions were 12.1% of all impressions but 34.6% of impressions on high-volume part-number queries, so the contamination was not uniformly distributed — it concentrated exactly on the queries the incident would later damage.

**Point-in-time discipline.** Every feature must be computable from state that existed before the query was served. Two features in the serving vector violate this in subtle ways and were caught in the audit: `item_price_rank` was computed from the current price table rather than the price at impression time, and `stock_availability` was computed from a nightly snapshot whose load time was not recorded. Neither caused this incident, but both were found during the leakage audit that the incident triggered, and both would have inflated any offline evaluation that used them.

### 5b. Tools & Stack

The stack is described as it would be operated, with the caveat that versions are scenario values and any real deployment must confirm its own.

| Component | Role in this investigation | Version / configuration note |
|---|---|---|
| Inverted-index search engine | Serves retrieval; holds `v2` and `v3` index snapshots side by side during replay | Snapshot-level versioning; v3 differs in the `mpn` field's analyser chain and the `title` stopword list |
| Query parser / builder | Converts raw query text to a Boolean-plus-phrase query with field boosts | Separate deployment unit from the index; the analyser config must agree with the index's |
| Learning-to-rank service | Scores 100 retrieved candidates with a gradient-boosted model over 74 features | Models `v6` and `v7` served from a versioned model registry; both loadable concurrently |
| Feature store | Provides popularity, price, availability and catalogue-quality features | Two-tier: offline batch for history, online low-latency for serving |
| Impression and click logger | Emits one record per rendered result and one per click | Schema documented; `module` field added 2026-08 without a corresponding contract update |
| Judged relevance store | Holds the 20,000 graded queries and 340,000 labels | Immutable snapshot; re-curation requires a new snapshot ID |
| Offline replay harness | Runs a judged query set through a specified (index, ranker) pair and computes ranking metrics | Deterministic given the same index snapshot, model version and query set; this is the oracle for all numbers below |
| Session analytics warehouse | Holds sessionised logs for traffic-mix and outcome analysis | Daily partitions; query-class classifier applied at load time |
| Monitoring | Weekly aggregate relevance report; per-service latency and error dashboards | No per-stratum relevance alert; no null-rate alert on ranker features |

Two stack facts carry the diagnosis. First, the replay harness can pin an index snapshot and a model version independently, which is what makes the 2×2 factorial decomposition possible at all; a stack that could only serve the current pair would have forced a sequential, confounded rollback. Second, the query builder and the index are separate deployment units with a shared analyser contract that is not enforced by a schema check — there is no build-time assertion that the query-side analyser matches the index-side analyser. That asymmetry is the structural cause of the incident and the first thing any permanent fix must address.

### 5c. Method Summary

The method is deliberately unglamorous: establish what actually changed, separate the changes, measure each combination, and only then choose a reversal. The chain runs from log inspection through stratified re-measurement to a factorial replay and finally to a decision.

```text
raw logs (query, impression, click)
  -> sessionisation and query-class assignment
  -> stratified metric computation on judged set (aggregate vs per-class)
  -> feature-distribution audit on the serving vector
  -> 2x2 factorial replay (index v2/v3) x (ranker v6/v7)
  -> attribution of the loss to artefacts
  -> smallest sufficient reversal -> decision
```

| Candidate approach | Evidence considered | Strength | Limitation / avoid when | Status |
|---|---|---|---|---|
| Trust the aggregate dashboard | Weekly report showing −0.008 | Free; already exists | Dilutes stratum effects; cannot localise | Rejected — arithmetic in Section 6 shows 51.9× dilution |
| Roll back both artefacts immediately | Release manifest | Fast; guaranteed to remove the cause | Destroys attribution; forfeits any genuine gain; cannot distinguish ranker from index | Rejected as first action, adopted as the final action once attribution was complete |
| Roll back the ranker only | Model registry; feature-contract diff | Smaller change surface | Fails: ranker-only rollback leaves NDCG@10 at 0.6512 versus 0.8935 baseline | Rejected on evidence |
| Roll back the index only | Snapshot restore | Smaller change surface | Fails: index-only rollback leaves NDCG@10 at 0.6987 | Rejected on evidence |
| 2×2 factorial replay, then targeted reversal | Versioned snapshots and models | Attributes the loss exactly; cheap | Requires independent pinning of index and model — not universal | **Chosen** |
| Online A/B experiment on the affected population | Experiment platform | Direct causal estimate in production | Tests rankers on buyers who are failing now; operations vetoed | Not tested — explicitly blocked |
| Retrain the ranker on corrected clicks without touching the index | Click-log schema audit | Fixes the label corruption at its source | Does not restore the `mpn_exact` feature; would still leave the signal dead | Deferred to the permanent fix, not the incident fix |

The chosen approach dominates on decision quality per unit of delay. The replay costs minutes — 400 queries, four cells, a harness that already exists — and it answers the question the operations manager actually asked. The alternatives that look faster either destroy evidence (roll back both) or fail to restore the metric (single-artefact rollback). The alternative that would be scientifically strongest, an online experiment, was ruled out by a constraint the team did not control.

**Hyperparameters and stopping conditions.** The replay has no learned parameters. Its stopping condition is coverage: all 400 `part_number` queries and a 1,200-query stratified sample of the other two strata. The only discretionary choice is the metric: NDCG@10 is primary because it is what the team already reports, and exact-hit-in-top-3 is carried as a secondary confirmatory metric because it maps onto the product constraint `C-03`. No threshold tuning on the confirmation split was permitted; thresholds were set on the 16,000 tuning queries and evaluated once on the 4,000 locked queries.

### 5d. Data provenance, quality, exclusions and leakage controls

| Dataset | Source ID | Owner | Extraction | Known quality issue | Handling |
|---|---|---|---|---|---|
| Query and session logs | `SRC-Q` | Analytics engineering | Daily partition export, 2026-10-19 | Bot filtering by user-agent misses some scripted traffic | Velocity rule applied as a second filter; residual estimated below 0.3% |
| Impression logs | `SRC-I` | Storefront engineering | Streamed to warehouse, 2026-10-19 | `module` null for `accessory_carousel` from 2026-08 | Not repairable retroactively; excluded from training in the permanent fix |
| Click logs | `SRC-C` | Analytics engineering | Same export as impressions | Position collides with organic positions for carousel clicks | 12.1% of impressions flagged contaminated and excluded |
| Judged relevance | `SRC-J` | Relevance lead | Snapshot `J-2026-03`, immutable | Stale; under-represents part-number strings | Used stratified; locked confirmation subset held back |
| Index snapshots | `SRC-IDX` | Search platform | `v2` (2026-06-30), `v3` (2026-10-13) | `v2` snapshot predates an unrelated price-field change | Effect confined to price features; reported in sensitivity |
| Model artefacts | `SRC-M` | Ranking platform | Registry entries `ranker.v6`, `ranker.v7` | `v7` trained on contaminated clicks | Contamination quantified; used as evidence, not as a good model |
| Feature vectors | `SRC-F` | Ranking platform | Serving-vector log, 1% sample | Null-coercion to 0.0 hides missing features | Null-rate audit added |

**Leakage controls.** Three distinct leakage risks were checked, and the distinction between them matters because only one was realised.

The first is *temporal leakage in evaluation*: the judged set was curated in 2026-03 and the models were selected in 2026-06 through 2026-09, so the set is not contemporaneous with the models. This is a validity threat to the *level* of the metrics, not to their *difference* — a stale set can still rank two systems correctly if the queries remain representative. It was accepted with the caveat recorded in Section 9.

The second is *selection leakage through repeated use*: fourteen model versions had been selected against the same 20,000 queries before this incident. With 400 part-number queries and fourteen selection rounds, the effective optimism in the part-number stratum is not negligible. The retrospective 16,000/4,000 split is a partial remedy and is honestly labelled as retrospective rather than pre-registered. Critically, selection optimism biases *gains* upward; it cannot manufacture a loss of 0.42 on a held-back subset, which is why the direction of the finding survived this concern.

The third is *label leakage through the click log*: the ranker's training labels are clicks, and clicks are produced by the ranker's own previous ordering. This is the position-bias problem, and it is a permanent property of click-trained ranking, not a bug in this release. What this release added was a second, novel corruption — the carousel misattribution — which is a plumbing defect rather than a statistical one. Separating "the model inherits position bias, as always" from "the model was fed clicks on the wrong module" is what let the team size the fix: position-bias correction is a project, module attribution is an afternoon.

**What was unavailable.** No pre-release A/B test of either artefact exists, because the release was gated on offline evaluation only. No traffic replay at production scale was possible during the incident, because the harness runs on a snapshot and a fixed query set rather than live traffic. No human re-judgement of the affected queries was commissioned before the decision, so the graded labels used are the stale 2026-03 labels; this is the largest evidence limitation and is recorded as such in Section 9 and Section 14b.

### 5e. Reproducibility and method-selection ledger

| Artifact / source ID | Location | Version or date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `SRC-J` | Judged relevance store | `J-2026-03`, immutable snapshot | Ground truth for all ranking metrics | Internal, read-only | Cannot be regenerated; curator's rubric is versioned but assessor disagreement was never measured |
| `SRC-IDX-v2` | Search platform snapshot store | 2026-06-30 | Pre-release index | Internal | Contains a price-field layout that no longer matches the current price feature extractor; price-feature effect separated in sensitivity |
| `SRC-IDX-v3` | Search platform snapshot store | 2026-10-13 | Post-release index | Internal | Rebuild is deterministic given analyser config, but the config file is not itself version-controlled — reconstructed from a deployment bundle |
| `SRC-M-v6` / `SRC-M-v7` | Model registry | Registry entries, immutable | Rankers under comparison | Internal | `v7`'s training data snapshot was retained; `v6`'s was not, so `v6` cannot be retrained from scratch |
| `RUN-01` | Replay harness job log | `replay-20261019-4cell` | 2×2 factorial NDCG@10 | Internal | Single machine, cold cache; absolute query latency not comparable to serving |
| `RUN-02` | Replay harness job log | `replay-20261019-strata` | Stratified metrics at (v7, v3) | Internal | Same harness; the 1,200-query non-part-number sample has wider intervals than the part-number census |
| `RUN-03` | Feature audit job | `feataudit-20261019` | Null-rate and distribution shift per feature | Internal | 1% serving-vector sample; rare features may be under-sampled |
| `SRC-REF-01` | Järvelin and Kekäläinen, "Cumulated gain-based evaluation of IR techniques", *ACM TOIS* 20(4), 2002 | Published 2002 | NDCG definition and discount convention | Public | Establishes the metric; says nothing about this scenario's numbers |
| `SRC-REF-02` | Joachims, Swaminathan and Schnabel, "Unbiased learning-to-rank with biased feedback", *WSDM* 2017 | Published 2017 | Position-bias framing for click-trained rankers | Public | Establishes the general problem; the carousel defect is a different mechanism |
| `SRC-REF-03` | Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, ch. 8 and 11 | Published 2008 | Evaluation and probabilistic retrieval background | Public | Textbook basis; not evidence for any figure here |

**Reproducibility statement.** The replay is deterministic given `SRC-J`, `SRC-IDX-v2`, `SRC-IDX-v3`, `SRC-M-v6` and `SRC-M-v7`. A reader with those five artefacts reproduces every metric in Section 6 exactly, because the harness has no sampling. The stratified production metrics in Section 6 are *not* reproducible from this document alone: they require the session logs and the query-class classifier, both of which are internal. The classifier itself is a source of irreducible uncertainty — it assigns each query to a stratum by a rule the team can inspect but not verify per query, and misclassification between `exploratory` and `specification` is common while `part_number` is nearly deterministic (an MPN-shaped token matches a catalogue key or it does not). That asymmetry is why the part-number stratum, the one that matters, is the one with the cleanest assignment.

All numerical claims in this document are **ILLUSTRATIVE** scenario values unless explicitly labelled otherwise. The two published references establish metric definitions and the general position-bias problem; they do not validate any figure, and no part of this case should be read as a reproduction of those papers' experiments.

## 6. Analysis Walkthrough

The decisive result, stated first: the release cost the part-number stratum **0.4228 NDCG@10** — from 0.8935 to 0.4707, a 47.32% relative loss — while moving the aggregate judged metric by only **−0.0081**. The part-number stratum is 4.1% of production queries but only 2.0% of the judged set, and that sampling gap plus the stratum's own weight together hide a catastrophic local failure inside a numerically flat global report. The cause is not the ranker. It is a two-part interaction: index `v3`'s analyser change destroyed the `mpn_exact` feature for hyphenated part numbers, and ranker `v7` had simultaneously learned to prefer accessory items on those same queries because the carousel click contamination concentrated there. Either change alone would have been survivable; together they produced a 0.2135 interaction term that neither artefact's own offline evaluation could have surfaced.

| Step | Hypothesis or decision | Input snapshot | Method | Intermediate output | Interpretation | Evidence |
|---|---|---|---|---|---|---|
| 1 | Something changed on 2026-10-13 | Weekly relevance report, 12 weeks | Read the aggregate series | Flat, −0.008 in the release week | Either nothing happened or the effect is hidden | ILLUSTRATIVE, `SRC-J` |
| 2 | The complaint names a stratum | Support contacts, 7 days | Classify contact text by query class | 1,617 contacts vs 1,240 expected | 377 excess contacts, all part-number | ILLUSTRATIVE, `SRC-A`/`SRC-Q` |
| 3 | The judged set mis-weights the stratum | `SRC-J`, query classes | Compare shares | 2.0% judged vs 4.1% production | Aggregate dilutes the stratum by 2.05× | DERIVED |
| 4 | Stratify the metric | `SRC-J` with class labels | Per-class NDCG@10, paired | part 0.8935→0.4707 | The regression is real and large | DERIVED, `SRC-J` |
| 5 | Localise to a feature | 1% serving vectors, both versions | Null-rate and value-distribution audit | `mpn_exact` = 0.0 for 96.8% of part-number candidates | A binary feature went dead | ILLUSTRATIVE, `RUN-03` |
| 6 | Explain why the feature died | Index `v3` analyser config | Tokenise sample MPNs under both configs | `6ES7-132-4BF00-0AB0` → 4 tokens in `v3` | Exact string never reconstructed at query time | ILLUSTRATIVE, `SRC-IDX` |
| 7 | Separate the artefacts | Four version pairs | 2×2 factorial replay on 400 queries | Index main −0.1948, ranker main −0.0145, interaction −0.2135 | Both are needed to explain the loss | DERIVED, `RUN-01` |
| 8 | Check the ranker's learned preference | Click log, 90 days | Recompute labels with carousel excluded | 34.6% of part-number impressions contaminated | The model learned a wrong preference, not noise | ILLUSTRATIVE, `SRC-C` |
| 9 | Test the rollback options | Replay at each single-rollback cell | Compare NDCG@10 and exact-hit@3 | Ranker-only 0.6512; index-only 0.6987 | Neither single rollback restores quality | DERIVED, `RUN-01` |
| 10 | Verify the fix survives stratification | Locked 4,000-query confirmation split | One-shot evaluation, no tuning | part 0.8941, expl 0.7191, spec 0.7705 | Restores the stratum without the exploration cost | DERIVED, `RUN-02` |

### Step 1 — The dashboard is flat, and that is the first finding

The weekly relevance report showed NDCG@10 at 0.7354 before the release and 0.7272 after, a change of −0.0081. The report's own noise band, computed as a bootstrap over the judged set from the preceding twelve weeks, was ±0.0100. By the report's own criterion the movement was inside noise, and the on-call engineer correctly did not page anyone. This is worth dwelling on because it is not a story about an inattentive engineer; it is a story about a metric whose aggregation structure guarantees this outcome for a 4.1% stratum no matter how badly that stratum degrades.

The arithmetic that produced the flat number is straightforward once the strata are named. Let $w_j$ be the judged-set share of stratum $j$ and $m_j$ the stratum's mean metric. The reported aggregate is their weighted sum:

```text
FORMULA [R-01]: M_agg = SUM_j ( w_j * m_j ),  SUM_j w_j = 1
WHERE: M_agg aggregate relevance metric [dimensionless, 0..1];
       w_j judged-set share of stratum j [fraction];
       m_j mean metric within stratum j [NDCG@10, 0..1].
CONSTRAINT: additive aggregation of a per-stratum mean; valid only when the
            metric is a mean over units and strata partition the population.
NUMERICAL EXAMPLE: w=(0.020,0.730,0.250), m_before=(0.8935,0.7188,0.7712);
                   M_before = 0.020*0.8935 + 0.730*0.7188 + 0.250*0.7712
                            = 0.01787 + 0.52472 + 0.19280 = 0.73539.
                   m_after=(0.4707,0.7197,0.7698);
                   M_after  = 0.00941 + 0.52538 + 0.19245 = 0.72724.
INTERPRETATION: a 0.4228 collapse in one stratum moves the aggregate by
                0.008, which is smaller than the report's own noise band.
```

The sensitivity of the aggregate to the stratum is the product of the stratum's weight and its movement. Here $0.020 \times (-0.4228) = -0.00846$, and the other two strata contribute $+0.00066$ and $-0.00035$, giving $-0.00815$ before rounding. That is the whole mechanism. A stratum carrying 2% of the weight can lose 47% of its quality and register as a rounding difference.

### Step 2 — The operating signal is louder than the analytical one

The parts-counter operations manager escalated on 2026-10-19 with a specific and checkable claim: agents were re-typing part numbers into a manufacturer's own catalogue site because the internal search was returning accessories first. Support contacts mentioning search over the seven-day window totalled 1,617. The pre-release baseline rate, measured over the preceding eight weeks, was 0.90% of part-number queries.

```text
FORMULA [R-02]: E = r0 * Q,  X = C_obs - E
WHERE: E expected contacts under the baseline rate [contacts];
       r0 baseline contact rate per part-number query [fraction];
       Q part-number queries in the window [queries];
       C_obs observed contacts [contacts]; X excess contacts [contacts].
CONSTRAINT: contact rate stable absent the incident; contacts attributable
            to search by the support taxonomy; no queue-driven suppression.
NUMERICAL EXAMPLE: Q = 480000 * 0.041 * 7 = 137760 queries (7 full days);
                   E = 0.009 * 137760 = 1239.84 -> 1240 contacts;
                   X = 1617 - 1240 = 377 excess contacts.
INTERPRETATION: the operating signal is 1.30x the expected volume, which is
                a z of roughly 10 on this denominator and detectable in
                days; the analytical signal is below its own noise band.
```

Two consequences follow. First, the escalation was justified and the dashboard was not wrong — it was answering a different question. A per-stratum contact-rate monitor with a 3-sigma threshold on a daily bucket would have paged on 2026-10-15, four days earlier; the construction of that monitor is Section 8's first design decision and its alerting arithmetic is in Section 9.

Second, the excess-contact count is not a clean causal quantity. It is a lower bound on affected sessions, because buyers who silently abandon do not file a contact, and it may overstate impact if some contacts would have occurred anyway during a busy week. Assumption `A-07` records this honestly as unverified correlation. The point of the calculation is not precision; it is that an operationally meaningful signal existed and was independent of the broken metric.

### Step 3 — Stratification exposes the loss

Recomputing the metric per query class on the same judged set, with the same labels and the same harness, produced the table below. The pairing is per query, so each row is a paired difference and the interval is computed on the paired differences rather than on two independent samples.

| Stratum | Judged n | NDCG@10 before | NDCG@10 after | Paired Δ | Δ as % of baseline | 95% CI on Δ |
|---|---:|---:|---:|---:|---:|---|
| `part_number` | 400 | 0.8935 | 0.4707 | −0.4228 | −47.32% | [−0.4622, −0.3834] |
| `exploratory` | 14,600 | 0.7188 | 0.7197 | +0.0009 | +0.13% | [−0.0031, +0.0049] |
| `specification` | 5,000 | 0.7712 | 0.7698 | −0.0014 | −0.18% | [−0.0064, +0.0036] |
| Aggregate (judged weights) | 20,000 | 0.7354 | 0.7272 | −0.0081 | −1.11% | [−0.0137, −0.0025] |

The part-number interval is computed as $\bar{d} \pm 1.96 \, s_d/\sqrt{n}$ with $\bar{d} = -0.4228$, $s_d = 0.4021$ and $n = 400$: the standard error is $0.4021/\sqrt{400} = 0.0201$, giving half-width $0.0394$ and the interval shown. The paired $t$-statistic is $-0.4228/0.0201 = -21.0$. Note what this interval does and does not mean: it quantifies sampling variability of the mean over 400 judged queries drawn from a stale pool. It does not cover the possibility that the judged pool misrepresents today's part-number queries, and it does not cover assessor error. Both are larger threats than sampling here, which is why Section 9 reports them separately rather than folding them into the interval.

The exploratory stratum moved by +0.0009 with a confidence interval that comfortably brackets zero. That is the crucial guardrail fact: the release did not harm the population that dominates revenue, and any rollback must not either. A team that rolled back both artefacts would have surrendered a small genuine exploratory improvement for no diagnostic benefit.

### Step 4 — The exact-match feature is dead, and that is checkable directly

Before any factorial replay, the fastest discriminating test was a direct inspection of the serving vectors. The ranker consumes 74 features; one of them, `mpn_exact`, is a binary indicator that the candidate's normalised MPN equals the query's normalised MPN. It exists precisely to enforce the product constraint that an exact part number must surface.

In the 1% serving-vector sample from the release week, `mpn_exact` was 0.0 for 96.8% of candidates on part-number queries, against 0.4% in the pre-release week. A binary feature that is always zero carries no information and therefore receives whatever weight minimises training loss — which, for a feature that never activates, is an arbitrary weight that does nothing at serving time. The ranker was not demoted by the model; the model was handed a constant.

The mechanism is worth tracing exactly, because "the analyser changed" is not an explanation until you can name the token stream. Index `v3` changed the `mpn` field from a keyword-style field (lowercased and normalised, single token, exact-match retrievable) to the engine's standard analyser, which lowercases, splits on hyphens and dots, and removes a stopword list. Under `v2`, the string `6ES7-132-4BF00-0AB0` is one token. Under `v3`, it is four tokens:

```text
v2 token stream for "6ES7-132-4BF00-0AB0":
  ["6es7-132-4bf00-0ab0"]                     -> 1 token, exact retrieval possible

v3 token stream for "6ES7-132-4BF00-0AB0":
  ["6es7", "132", "4bf00", "0ab0"]            -> 4 tokens, exact retrieval impossible

v3 token stream for "6ES7-132-4BF00-0AB0" after the query-side builder
  strips the hyphen as a non-indexed character and ORs the pieces:
  match on "6es7"  -> 41,200 candidates
  match on "132"   -> 210,000 candidates (also a common numeric suffix)
```

Two failures compound here, and distinguishing them matters for the fix. First, the *feature extractor* computes `mpn_exact` by comparing the query string to the stored `mpn` value, but it obtains the stored value by reading the analysed field, which now returns a token list rather than a string; the comparison therefore fails for every hyphenated MPN, which is 96.8% of them. Second, the *retrieval* path no longer has a single-token exact match, so the exact part may not even be in the top 100 candidates — and a ranker cannot promote a candidate that was never retrieved. The second failure is more severe and was invisible to the ranker-side evaluation, because the ranker's own offline test harness retrieved from a cached candidate set generated under `v2`.

This is the mechanism the release notes called "size reduction, no ranking intent". The rebuild's stated goal was a 19% overall index-size reduction, achieved mainly through a `doc_values` change on `description` (34.0 GiB → 27.5 GiB) and a stopword-list trim on `title`. The `mpn` analyser change contributed 21.6 MB of the 6.5 GiB saving — 0.33% of the savings and 0.06% of the index — and cost the stratum its entire exact-match capability. The ratio of harm to benefit is roughly 0.06% of a storage saving against 47% of a stratum's quality, which is the kind of comparison that should be made before bundling "incidental cleanup" into a release train.

### Step 5 — The 2×2 factorial replay attributes the loss

With independent pinning available, the replay evaluated all four combinations on the 400 judged part-number queries. Each cell is a full re-retrieval plus re-ranking, so the retrieval failure and the ranking failure are both represented; there is no candidate-cache contamination across cells.

```text
FORMULA [R-03]: NDCG@10 = DCG@10 / IDCG@10
                DCG@10  = SUM_{i=1..10} (2^{r_i} - 1) / log2(i + 1)
WHERE: r_i graded relevance of the result at rank i, integer 0..4;
       IDCG@10 is the DCG of the same result set sorted by descending
       relevance, so NDCG@10 = 1 iff the ordering is already ideal.
CONSTRAINT: gains are exponential in the grade, so a grade-4 result at rank 1
            contributes 15.000 while the same result at rank 10 contributes
            4.336; the metric therefore penalises demoting exact matches far
            more than promoting irrelevant ones.
NUMERICAL EXAMPLE: see the position-by-position trace in Step 6.
INTERPRETATION: the discount is steep at the head, which is why a regression
                that pushes exact matches from rank 1 to rank 3 registers as
                a catastrophic loss rather than a cosmetic reordering.
```

| Cell | Index | Ranker | NDCG@10 (400 queries) | Exact-hit-in-top-3 |
|---|---|---|---:|---:|
| Baseline | `v2` | `v6` | 0.8935 | 96.50% (386/400) |
| Ranker only | `v2` | `v7` | 0.8790 | 93.75% (375/400) |
| Index only | `v3` | `v6` | 0.6987 | 66.00% (264/400) |
| Both | `v3` | `v7` | 0.4707 | 31.25% (125/400) |

Reading the table row by row gives the attribution directly. Compare row 1 to row 2: changing only the ranker costs 0.0145. Compare row 1 to row 3: changing only the index costs 0.1948. The index change alone explains 46.1% of the total loss. But the two do not add up, and the shortfall is the interesting part.

```text
FORMULA [R-04]: M_ij = mu + a_i + b_j + e_ij
WHERE: M_ij metric at index level i and ranker level j;
       mu grand reference (baseline cell); a_i ranker main effect;
       b_j index main effect; e_ij interaction residual.
       With a 2x2 design and the baseline as reference:
       a_1 = M(v7,v2) - M(v6,v2),  b_1 = M(v6,v3) - M(v6,v2),
       predicted M(v7,v3) = M(v6,v2) + a_1 + b_1,
       interaction = observed M(v7,v3) - predicted.
CONSTRAINT: requires each factor to be independently settable; the
            decomposition is descriptive, not causal-identified, because
            there is one run per cell and no experimental randomisation.
NUMERICAL EXAMPLE: a_1 = 0.8790 - 0.8935 = -0.0145;
                   b_1 = 0.6987 - 0.8935 = -0.1948;
                   predicted = 0.8935 - 0.0145 - 0.1948 = 0.6842;
                   observed  = 0.4707;
                   interaction = 0.4707 - 0.6842 = -0.2135.
INTERPRETATION: 50.5% of the total 0.4228 loss is interaction, so neither
                artefact is "the cause" and neither single rollback restores
                the metric.
```

The interaction is not a statistical artefact; it has a mechanical explanation, and the explanation is what makes the case teachable. Under `v2`+`v6`, the exact part is usually at rank 1 because both retrieval and ranking agree on it. Under `v2`+`v7`, the exact part is still retrieved and still has a live `mpn_exact` feature, so the ranker mostly respects it — 93.75% exact-hit-in-top-3. Under `v3`+`v6`, retrieval no longer guarantees the exact part is present, but `v6`'s other lexical features (title match, manufacturer match, category) still push it up when present; and critically `v6` had not been trained to prefer accessories, so exact parts that survived retrieval tended to stay near the top — 66.00%. Under `v3`+`v7`, the exact part is frequently absent from the candidate set *and* the ranker has learned that accessory items are the preferred click target on these queries, so accessories rank first even when the exact part is present — 31.25%.

The counterfactual is worth stating precisely because it is the decision-relevant one: had the ranker never been retrained, the index change alone would have cost 0.1948 and 30.5 percentage points of exact-hit-in-top-3. Had the index never been rebuilt, the ranker change alone would have cost 0.0145 and 2.75 points. Deploying both cost 0.4228 and 65.25 points. The multiplicative structure of the failure — retrieval removes the item, ranking prefers something else — is why the joint effect exceeds the sum of the parts by a factor of two.

### Step 6 — A position-by-position trace

Aggregate deltas are easy to accept and hard to believe. One query, traced through both configurations, makes the mechanism concrete and lets a reader check the arithmetic independently. The query is `6ES7-132-4BF00-0AB0`, a Siemens programmable-logic-controller module. The judged labels for the ten items shown are, in rank order under `v2`+`v6`: 4, 3, 2, 1, 1, 0, 0, 1, 0, 0. Under `v3`+`v7` the ordering is: 0, 1, 4, 0, 0, 2, 0, 0, 0, 1.

| Rank | $r_i$ before | Gain $2^{r}-1$ | Discount $1/\log_2(i+1)$ | Contribution | $r_i$ after | Gain after | Contribution |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 15 | 1.000000 | 15.000000 | 0 | 0 | 0.000000 |
| 2 | 3 | 7 | 0.630930 | 4.416508 | 1 | 1 | 0.630930 |
| 3 | 2 | 3 | 0.500000 | 1.500000 | 4 | 15 | 7.500000 |
| 4 | 1 | 1 | 0.430677 | 0.430677 | 0 | 0 | 0.000000 |
| 5 | 1 | 1 | 0.386853 | 0.386853 | 0 | 0 | 0.000000 |
| 6 | 0 | 0 | 0.356207 | 0.000000 | 2 | 3 | 1.068622 |
| 7 | 0 | 0 | 0.333333 | 0.000000 | 0 | 0 | 0.000000 |
| 8 | 1 | 1 | 0.315465 | 0.315465 | 0 | 0 | 0.000000 |
| 9 | 0 | 0 | 0.301030 | 0.000000 | 0 | 0 | 0.000000 |
| 10 | 0 | 0 | 0.289065 | 0.000000 | 1 | 1 | 0.289065 |
| | | | **DCG@10** | **22.049503** | | | **9.488616** |

The ideal ordering for the before-result-set is 4, 3, 2, 1, 1, 1, 0, 0, 0, 0, whose DCG is 22.090245. So the baseline NDCG@10 for this query is $22.049503/22.090245 = 0.998156$ — near-perfect, as an exact-MPN query should be. Using the same idealiser for the after-ordering, NDCG@10 is $9.488616/22.090245 = 0.429539$. The drop is 0.568617 for this single query.

One methodological subtlety is visible here and must be handled consistently, because handling it differently across queries is a classic way to manufacture a number. The after-ordering has a different *set* of items than the before-ordering (because retrieval changed), so its own ideal DCG is 17.823466 rather than 22.090245. Normalising the after-ordering by its *own* ideal gives $9.488616/17.823466 = 0.532367$. The correct choice for a paired regression comparison is to hold the idealiser fixed at the union of judged items, so that a system that retrieves fewer relevant items is penalised for the omission. Both conventions were computed during the investigation; the fixed-idealiser convention (0.4295) was used for the headline comparison because it is the one that penalises lost recall, and the own-idealiser convention (0.5324) was retained as a sensitivity check. The gap between them, 0.1028, is the size of the recall penalty the fixed convention charges. A reviewer who sees only the own-ideal number would conclude the query degraded by 0.466 rather than 0.569, and would understate the harm by 18%.

The trace also shows why the exact-hit-in-top-3 metric behaves so differently from NDCG. In the after-ordering the grade-4 item is at rank 3, so exact-hit-in-top-3 is satisfied for this query — yet NDCG fell by 0.57 because the grade-4 item moved from rank 1 to rank 3 and three grade-0 items entered the head. Aggregate confusion between these two metrics is a recurring diagnostic error, addressed as a failure mode in Section 12.

### Step 7 — The ranker's learned preference is a data defect, not a taste disagreement

The 2×2 replay explains the loss mechanically but not why `v7` prefers accessories. That question has two possible answers, and they lead to opposite decisions. Either buyers genuinely click accessories more often on part-number queries (a legitimate shift in preference that the model correctly captured, in which case the "regression" is a product argument about whether exact parts should outrank accessories), or the click data misattributes accessory clicks to organic positions (a plumbing defect, in which case the model learned a fiction).

The discriminating check is a schema-level inspection of how a click is attributed to a result, and it resolved the question decisively. In 2026-08 a cross-sell carousel module was introduced below the organic results. Its instrumentation emitted `position` from the carousel's own index but failed to emit `module`, so a carousel click on the third cross-sell tile was logged as `module = null, position = 3` — indistinguishable from an organic click on the third organic result. Because the training pipeline's organic filter was `module is null`, every carousel click on a part-number query became a positive label on the organic result at positions 1–3, which for these queries is usually an accessory (belts, filters, mounting kits) rather than the exact module.

| Quantity | Accessories | Organic | Total |
|---|---:|---:|---:|
| Impressions, all queries | 12.1% | 87.9% | 100% |
| Impressions, part-number queries | 34.6% | 65.4% | 100% |
| Clicks attributed as organic, part-number queries | 29.8% | — | — |
| Click-weight share in `v7` training for part-number queries | 71.4% | 28.6% | 100% |

The last row is the one that explains the model's behaviour. Clicks are sparse and are the training signal; a click on a position-1 accessory is one training example, and the click-through rate on accessories is high because a buyer who needs a module often also needs the belt. So accessories received 71.4% of the click weight in the part-number slice. `v7` had been trained with a new hard-negative sampling scheme that up-weighted rare positives, which amplified the mislabelled ones further — that is the only contribution `v7`'s own design change made: it made an existing data defect worse rather than introducing a new one.

The distinction between "the model is wrong" and "the model is right about the wrong target" determines the fix. If the model were wrong, more data or a better architecture would help. Because the model was right about a corrupted target, the fix is to correct the label definition and add a guardrail that makes the exact-match constraint a hard constraint rather than a learned preference. Both are in Section 8; the second is the one that would have contained this incident to 0.1948 instead of 0.4228.

### Sensitivity and robustness analysis

A single headline number over 400 judged queries on a stale judged set deserves adversarial pressure. Six tests were run; two of them materially qualify the conclusion, and both qualifications are recorded rather than smoothed away.

| Stress test | Original condition | Alternative | Result | Conclusion change | Decision consequence |
|---|---|---|---|---|---|
| Judged weights → production weights | 2.0% part-number share | 4.1% share | Aggregate Δ becomes −0.0170 instead of −0.0081 | None — still within a wider ±0.0100 band for the aggregate, but the stratum is unchanged | Confirms the aggregate is the wrong instrument, not that the harm is small |
| Own-idealiser NDCG convention | Fixed idealiser | Own idealiser per ordering | Part-number Δ becomes −0.3755 instead of −0.4228 | Small in direction, large in magnitude (−11.2% relative) | Report both conventions; fixed is primary |
| Exclude the 3.2% of part-number queries whose MPN contains no hyphen | 400 queries | 387 queries | Δ = −0.4301 | None | Non-hyphenated MPNs were damaged slightly more, because they also lost their exact field |
| Restrict to queries with ≥3 judged items at grade ≥3 | 400 queries | 341 queries | Δ = −0.4402 | None; interval widens to ±0.0442 | Conclusion robust to sparse-judgement queries |
| Re-judge a 60-query sample with a second assessor | Single assessor, 2026-03 labels | Two assessors, blinded | Cohen's $\kappa = 0.71$; Δ = −0.4188 on the re-judged subset | None | Assessor disagreement is a real but not decision-changing uncertainty |
| Traffic-mix shift correction | Fixed class shares | Weekly shares recomputed | Shares vary ≤1.0 pt week to week | None | `A-08` holds; mix shift is not an alternative explanation |

The own-idealiser result is the honest weak point. It says that between 11% and 12% of the reported magnitude is a measurement-convention choice rather than a system change, and a critic could reasonably argue the headline should be −0.3755. The response is not to pick the flattering number but to state both and note that the decision — restore the exact-match path — is identical under either. A decision that flips when the metric convention flips would be a decision resting on the metric rather than on the mechanism; this one does not.

The re-judgement check deserves a note on why it was done at all, given it was not pre-registered. The judged set is four months stale and was written by one assessor for a different purpose. If the labels themselves had drifted — if today's buyers would grade a compatible substitute as a 4 rather than a 3 — then the "regression" could be a label artefact. The 60-query blinded re-judgement found agreement at $\kappa = 0.71$, which is substantial but not high, and the recomputed delta of −0.4188 sits inside the original interval. That does not prove the labels are right; it proves they are stable enough that the loss is not an artefact of label drift.

### What was tried first and why it did not work

Two approaches were genuinely attempted before the factorial replay, and recording them is part of the case's honesty.

The first was **rolling back the ranker to `v6` alone**, attempted as the obvious minimal reversal at 03:05 UTC on 2026-10-20. It failed for a reason recorded as assumption `A-02`: `v6` was not feature-compatible with index `v3`, because one input feature had been renamed in the serving vector when `v7` shipped. The serving path returned zeros for the renamed feature, and the replay-equivalent measurement after the rollback showed 0.6512 — better than 0.4707, far worse than 0.8935. The rollback was reversed within nine minutes. This was a real failure of a real attempted mitigation, and its lesson is that model rollback is only safe when the feature contract is versioned independently of the model artefact.

The second was **investigating the query parser**, on the hypothesis that the query-side builder was stripping hyphens and thereby mangling MPNs. This hypothesis was partly right and was pursued for roughly two hours. It was abandoned as the primary explanation when a direct token-stream comparison showed that under `v2` the parser's hyphen handling was not merely tolerated but irrelevant: `v2`'s keyword field never exposed a token stream at all, so the parser's behaviour had been invisible for two years. The parser was not the cause; it was the mechanism by which an index-side change became a retrieval-side failure. That distinction changed the fix from "patch the parser" to "restore the exact field and add a contract test", which is a different and smaller change.

### Where the conclusion is weakest

Three weaknesses, stated plainly. The judged set is stale and single-assessor, so the *level* 0.8935 may not describe today's part-number queries even though the *difference* is robust. The replay uses a snapshot index rather than live serving, so it cannot capture cache effects, partial-shard behaviour, or the interaction between the two front ends; the discrepancy between replay and production is unknown and is an open question in Section 14b. And the counterfactual "`v6` with `v3` would have cost only 0.1948" is a single run with no replication; the number is exactly reproducible given the artefacts, but its generalisation to a different query mix is not established. None of these weaknesses threatens the decision, which rests on the mechanism rather than on the third decimal. All three would matter to a claim like "we lost exactly $13,958 of contact cost", which is why Section 11 labels that figure an order-of-magnitude estimate rather than a measurement.

## 7. The Finding

The release cost the part-number stratum 0.4228 NDCG@10 — from 0.8935 to 0.4707, a 47.32% relative loss, with a paired 95% interval of [−0.4622, −0.3834] on 400 judged queries — and cut exact-hit-in-top-3 from 96.50% to 31.25%, a 65.25 percentage-point collapse whose two-proportion $z$ is −19.2. The cause is a joint failure that neither artefact would have produced alone: index `v3`'s analyser change on the `mpn` field removed the single-token exact match that retrieval and the `mpn_exact` feature both depended on, contributing −0.1948, while ranker `v7` was trained on click logs in which 71.4% of the part-number click weight came from a misattributed accessory carousel, contributing −0.0145 on its own; the interaction between the two accounts for −0.2135, or 50.5% of the total loss. The exploratory stratum, 72.9% of judged queries, moved by +0.0009 with an interval bracketing zero, so the release did not harm the dominant population and no rollback of both artefacts is justified on quality grounds.

The evidence scope is bounded and must be stated with the finding rather than after it. All figures are **ILLUSTRATIVE** scenario values recomputed from the documented inputs; the judged set is four months stale, single-assessor, and under-samples the affected stratum by a factor of two; the replay runs on snapshot artefacts rather than live serving; and no online experiment was conducted, because operations vetoed testing on a population that was actively failing. The finding is a *diagnosis* and not a *causal estimate from a controlled trial*: the 2×2 decomposition establishes that the two artefacts jointly explain the observed metric, but with one run per cell it cannot separate the mechanism's contribution from any other difference between the artefacts.

The main boundary is that only the exact-match, hyphenated-MPN subclass of part-number search was affected, and within that subclass the damage is concentrated where accessories are a plausible cross-sell. Non-hyphenated MPNs were damaged slightly more (−0.4301 on that subpopulation), and specification queries with attribute-based intent were essentially untouched. A different catalogue — one without a large accessory attach-rate on exact-part queries, or one whose part numbers are unhyphenated alphanumeric strings — would show a smaller interaction and a proportionally cleaner index-side story.

The next justified action is therefore not a rollback but a two-part fix executed inside the same change window: restore the `mpn` field to a keyword-style exact-matchable configuration with a build-time analyser-contract assertion, and add a hard monotone constraint in the ranker's decision layer that an exact MPN match never ranks below position 3. The first restores retrieval; the second converts a learned preference back into a business rule and would have contained the loss to roughly 0.19 rather than 0.42. Both changes are code-and-configuration only, so they fit the two-hour window without a schema migration.

## 8. System/Solution Design

The design has two distinct purposes and they must not be confused. The *incident response* is a two-hour, reversible change to restore quality. The *durable rebuild* is a set of structural changes that make this class of failure detectable and preventable. This section specifies both, labels which is which, and states plainly that neither was executed in this illustrative scenario beyond the documented attempted rollback.

### 8a. Offline / Training / Batch Path

The batch path trains the ranker, rebuilds the index, and generates the evaluation artefacts. Its failure is where this incident originated, so the design changes concentrate there.

```text
[catalogue snapshot + price/stock feed]
        |
        v
[attribute normalisation] --> [MPN canonicalisation] --> [contract check: MPN is
        |                              |                  single-token, exact]
        |                              v                        |
        |                    [index build: vN]                   | FAIL -> block release
        |                              |                        v
        +--> [click + impression logs] |              [analyser contract test]
                    |                  |                        |
                    v                  v                        |
        [label construction]    [index snapshot registry]       |
        - exclude module != organic                             |
        - position-bias weights                                 |
        - carousel impressions dropped                          |
                    |                                           |
                    v                                           v
        [feature extraction: 74 features] -------------> [training set]
                    |                                           |
                    v                                           v
        [feature contract version]  -----------------> [ranker training]
                                                                |
                                                                v
                                              [evaluation gates: per-stratum]
                                                                |
                                            FAIL -> block   |  PASS
                                                                v
                                                    [model + index registry]
                                                                |
                                                                v
                                              [release train: paired manifest]
```

Three additions to the batch path matter, and each is justified by a specific failure observed in this incident rather than by general good practice.

**Label construction with an explicit module filter.** The current pipeline selects `module is null` as organic. The fix inverts the default: only `module = 'organic'` is a positive training label, and any click whose `module` is null is dropped rather than assumed organic. A null is an unknown, not a category. The pipeline additionally joins the impression record to the click on `impression_id` and asserts one-to-one cardinality; a click that matches two impressions is a schema violation and quarantines the session rather than the click. This change alone would have prevented `v7` from learning the accessory preference, and it is cheap because the `module` field already exists in the impression schema from 2026-08 — only the click path omitted it.

**Analyser contract test as a release gate.** For every field with an exact-match requirement, a unit test tokenises a fixture set of at least 200 real values from the catalogue and asserts that each produces exactly one token, byte-identical modulo case folding, to what a query-side normaliser produces. The fixture set must include hyphenated, dotted, slashed and alphanumeric MPNs, because those are the shapes that break a standard analyser. The test runs in the index-build pipeline; a failure blocks the index snapshot from entering the registry. This is the check whose absence allowed a 0.06% size saving to destroy 47% of a stratum's quality, and it costs milliseconds.

**Paired release manifest.** The model and index ship as a single manifest with two independently versioned entries and a declared compatibility statement, plus a matrix of validated pairs. The manifest records, for each validated pair, the offline metrics on every stratum. A release that changes either artefact without a validated pairing fails a schema check. This does not prevent regression; it prevents *unattributable* regression, because a reviewer can see immediately which artefact moved and which cells of the matrix were re-measured.

**Per-stratum evaluation gates.** The gate that let this ship evaluated aggregate NDCG@10 only, with a −0.005 tolerance, and the release measured −0.0023 on that aggregate at the time of shipping (a smaller number than the −0.0081 later measured on production-weighted traffic, because the shipping gate used a different query sample). Replacing it requires three changes: report every stratum, set a per-stratum tolerance proportional to its judged-sample precision, and add a hard gate on the exact-match constraint that is not a metric at all. The hard gate is the important one: exact-hit-in-top-3 for exact-MPN queries must be ≥ 95% or the release is blocked, regardless of aggregate NDCG. A constraint expressed as a metric can be traded off; a constraint expressed as a gate cannot.

### 8b. Online / Serving / Real-Time Path

The online path serves one query. Its failure modes are latency, staleness, partial shards and — the one that mattered here — silent feature degradation.

```text
[query text + locale + front_end + session_id]
        |
        v
[normalise: case, whitespace, unicode NFKC] --> [classify stratum]
        |                                              |
        v                                              v
[parse: tokens, phrases, MPN candidate detection]  [per-stratum policy]
        |                                              |
        v                                              v
[retrieve: BM25 over fields, top-100] -------> [candidate set]
        |                                              |
        | (shard timeout -> partial + flag)            v
        |                                    [feature extraction: 74]
        |                                              |
        |                              [feature-health assertion:
        |                               null-rate, constant-value,
        |                               schema version] --> FAIL -> fallback
        |                                              |
        v                                              v
[exact-match detector] -- hit --> [constraint layer:        [ranker score]
        |                            force rank <= 3]            |
        |                                 |                       |
        +---------------------------------+---------------------->+
                                          |
                                          v
                              [final ordering, top-10 rendered]
                                          |
                                          v
                        [log impression: module='organic' explicit]
                                          |
                                          v
                              [response: 10 results + flags]
```

Four serving-side elements carry the diagnosis and the fix.

**Feature-health assertion.** Before ranking, the service samples the serving vector and asserts three invariants per feature: the null-rate is below its historical band, the value is not constant within the request, and the schema version matches the model's declared expectation. A binary feature that has gone to a constant zero is the exact signature of this incident, and a per-request check on "does any candidate have `mpn_exact = 1` when an MPN was detected in the query" would have fired on the first affected request. On failure the service degrades to a rule-based ordering — exact match first, then lexical score — and flags the response. That fallback is deliberately boring and deliberately not the ranker, because the ranker is what is suspect.

**Exact-match constraint layer.** This sits *after* ranking and cannot be learned away. If a candidate's canonical MPN equals the query's canonical MPN, that candidate is placed at position 1 unless another candidate is also an exact match, in which case the ranker orders among the exact matches only. The generalisation is a monotone rule: exact-match > same-manufacturer-compatible > cross-manufacturer-equivalent > everything else, with the ranker ordering within each tier. This converts a statistical preference into a deterministic contract, which is what a product promise requires. It also bounds the blast radius of any future ranking regression on this stratum: the worst case becomes "the ranker mis-orders the non-exact tail", not "the exact part is invisible".

**Explicit module in the impression log.** Every rendered result carries `module ∈ {organic, accessory_carousel, sponsored}` as a required field with a schema-level non-null constraint, validated in the client and re-validated server-side. An event missing it is rejected rather than stored with a null, so the ambiguity is pushed to the producer where it can be fixed. Rejection is preferable to silent null because a dropped impression is a visible metric drop, while a null impression is an invisible data defect.

**Cache key composed from the full ordering-relevant input.** The serving cache is keyed on query text plus index version plus ranker version plus locale plus the constraint-layer version. The pre-incident key was query text alone with a six-hour TTL, which meant a rollback could not be observed for popular queries until the cache turned over. Section 9 quantifies that recovery curve; Section 12 records it as a distinct failure mode because it briefly made a correct fix look ineffective.

### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| `DEC-01` | Fix index and ranker together in one window, no full rollback | Full rollback; ranker-only; index-only | 2×2 replay: single rollbacks leave 0.6512 / 0.6987 | Higher change risk in one window | If the fix regresses the exploratory stratum | Any exploratory-stratum movement beyond its band |
| `DEC-02` | Restore `mpn` to a keyword-style exact field | Keep standard analyser and patch the feature extractor | Feature is 0 for 96.8% of candidates; retrieval also broken | Reverts 21.6 MB of a 6.5 GiB saving | Catalogue grows MPN-like free-text needing tokenisation | New field semantics for `mpn` |
| `DEC-03` | Hard monotone constraint layer instead of a learned preference | Up-weight exact-match samples in training | Interaction term −0.2135 proves the model cannot be trusted to hold the promise | Constraint can over-promote an exact match with no stock | Multi-variant part numbers where exact ≠ best | Product changes the exact-match promise |
| `DEC-04` | Drop null-module clicks rather than infer organic | Infer organic from position; keep null and down-weight | 71.4% click-weight contamination | Loses 12.1% of click signal | Loss of signal degrades head-query quality | Carousel instrumentation fixed at source |
| `DEC-05` | Per-stratum release gates plus a hard exact-hit gate | Aggregate gate with tighter tolerance | Aggregate moved −0.0081 inside ±0.0100 noise | Slower releases; more gate maintenance | Sparse strata have wide intervals and weak gates | Judged set re-curated per stratum |
| `DEC-06` | Cache key includes index and ranker version | Shorter TTL; purge on deploy | 6-hour TTL, 74% cache hit rate on head queries | Lower hit rate after every deploy | Query-text-only caching is safe only if ordering never changes | Serving cost becomes dominant |
| `DEC-07` | No online experiment during the incident | A/B test on affected population | Operations veto; ethical constraint on failing users | Weaker causal identification | Diagnosis rests on replay, not randomisation | Any future investigation where users are not actively harmed |

`DEC-01` is the decision the whole case turns on, and its alternative is genuinely defensible. A relevance lead under time pressure at 03:00 could reasonably roll back both artefacts, restore quality within the window, and conduct the post-mortem afterwards. That path succeeds at the operational objective and fails at the learning objective: it destroys the ability to attribute, which means the same defect ships again the next time someone rebuilds the index. The tradeoff is stated honestly in the table — the chosen path carries more change risk inside a two-hour window than a rollback does. The mitigation is that both changes are configuration and code, both are reversible, and the rollback remains available if the canary degrades.

`DEC-03` deserves separate defence because "add a hard constraint" is often the wrong answer. Constraints hide model failures, and a team that adds constraints for every regression ends up with a ranker whose output is mostly overridden. The test for whether a constraint is appropriate is whether the requirement is a *product promise* or a *statistical preference*. Exact part-number matching is a promise: the buyer typed a specific identifier and expects that identifier. Accessory attach-rate is a preference: it should be learned, and it should lose to the promise. The constraint is scoped to exact MPN equality only — a narrow, inspectable predicate — and everything else remains the ranker's judgement. A constraint on "results should be relevant" would be unjustifiable; a constraint on "an exact identifier match ranks first" is a specification.

### 8d. Architecture artefacts and operational readiness

| Area | Design position |
|---|---|
| Interfaces | Query API accepts `query_text`, `front_end`, `locale`, `session_id`; returns ten results plus `flags` (partial, degraded, constraint-applied). Impression event schema requires `module` non-null and `impression_id` unique; violation rejects the event at ingestion |
| Data/state | Index snapshots immutable and versioned; model artefacts immutable and versioned; the pairing lives in a manifest. Cache entries are derived state and may be discarded at any time without correctness loss |
| Reliability | SLI: fraction of queries returning HTTP 200 with a `results` array of ≥1 within 300 ms at the 95th percentile, measured at the load balancer over a rolling 28 days. Target 99.5%; error budget consumed by both errors and degradation flags |
| Resilience | Per-shard timeout 40 ms with partial-result flag; ranker timeout 60 ms with fallback to lexical ordering; circuit breaker opens after 5 consecutive ranker failures and stays open 30 s; degraded responses are served rather than errors |
| Recovery | Index rollback is a snapshot pointer swap, seconds; model rollback requires a compatible feature contract, which is why `A-02` mattered. Replay tests exercise both directions monthly. Cache purge is part of the deploy procedure, not an afterthought |
| Observability | Per-stratum relevance, per-feature null-rate and constant-value rate, exact-hit-in-top-3, contact-rate z-score on a 3-day trailing window, index lag, cache hit rate by version. Each has a named owner and a documented first diagnostic step |
| Deployment | Index and model ship as a paired manifest; canary at 5% of part-number traffic for 30 minutes with automatic rollback on exact-hit dropping below 95%; full rollout only after canary passes both the constraint gate and per-stratum NDCG |
| Security | Index field-level access controls restrict `mpn` to the search service identity; ranker weights are not writable by the merchandising role; configuration history is append-only and audited |
| Economics | Serving cost dominated by ranker inference; index storage cost trivial relative to its size because the failing change saved 21.6 MB of 34.0 GiB. The relevant economics is *evaluation* cost, sized below |
| Human workflow | Relevance lead owns ranking quality; merchandising owns the exact-match promise; operations owns the contact-rate signal; any of the three may open a rollback, but only the relevance lead and platform lead jointly may execute one |

**Worked capacity estimate for the monitoring change.** The incident's cost was dominated by detection delay, not by any computational inefficiency, so the design's sizing question is what a per-stratum monitor costs. The monitor recomputes NDCG@10 per stratum on the judged set nightly and computes a contact-rate z-score hourly on a rolling 3-day window.

```text
FORMULA [R-05]: C_eval = n_judged * (t_retrieve + t_rank) * r_replays
WHERE: C_eval evaluation cost [core-seconds per run];
       n_judged judged queries evaluated [queries];
       t_retrieve average retrieval time [s/query];
       t_rank average ranking time [s/query];
       r_replays number of (index, model) cells replayed.
CONSTRAINT: single-threaded equivalent; ignores harness overhead, I/O
            contention and parallel speedup; gives an order of magnitude.
NUMERICAL EXAMPLE: n_judged = 20000, t_retrieve = 0.045 s,
                   t_rank = 0.020 s, r_replays = 1;
                   C_eval = 20000 * 0.065 * 1 = 1300 core-seconds
                          = 21.7 core-minutes per nightly run.
INTERPRETATION: at 8-way parallelism the nightly job takes under 3 minutes,
                which is why per-stratum gating has no meaningful cost.
```

```text
FORMULA [R-06]: C_hourly = q_hourly * p_sample * c_metric
WHERE: C_hourly hourly monitoring compute [core-seconds/h];
       q_hourly queries in the hour [queries/h];
       p_sample fraction sampled [fraction];
       c_metric cost per sampled query of stratum assignment and
                exact-hit evaluation [core-s/query].
CONSTRAINT: excludes storage and query cost in the warehouse; excludes
            on-call labour, which dominates the true cost.
NUMERICAL EXAMPLE: q_hourly = 20000, p_sample = 0.05, c_metric = 0.0004;
                   C_hourly = 20000 * 0.05 * 0.0004 = 0.4 core-seconds/h
                            = 9.6 core-seconds/day.
INTERPRETATION: the incremental compute is negligible; the change is an
                alerting and ownership change, not a capacity change.
```

Both estimates say the same thing in different units: the monitoring was never expensive, and the reason it did not exist was that nobody had defined the stratum-specific SLI. That is an organisational finding, and it is the one Section 13 generalises. Note also what these estimates deliberately exclude — on-call labour, which at even one hour of engineer attention per detected incident vastly exceeds the annual compute cost of the monitor. Costing the compute and ignoring the labour is the standard way to conclude, wrongly, that monitoring is too expensive.
