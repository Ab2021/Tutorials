---
case_id: "CASE-03-002"
case_title: "Implement Top-K Stream Processing"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing / streaming aggregation"
primary_pattern: "bounded-stream-ranking"
secondary_patterns: ["sketch-error-contract", "deterministic-tie-ordering"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "streaming platform lead"
source_index: "03_dsa_and_computational_thinking/Case_Studies/INDEX.md"
review_trigger: "Change to key cardinality, skew, window length, publication cadence, shard count, memory budget, or the accepted rank-flip rate"
---

# Implement Top-K Stream Processing

**Evidence declaration.** This is an ILLUSTRATIVE teaching case. The stream, the key catalogue, the skew parameter, the shard count, the memory budget and every latency or bandwidth constant below are constructed so that one decision can be audited. No company deployment, no measured incident and no vendor benchmark is reported. **Implementation stage:** proposed. **Editorial status:** draft; the error contract, the tie-ordering rule and the publication protocol are design artefacts that have not been operated.

All arithmetic here is *derived* from the declared inputs and was recomputed line by line during authoring (the recomputations are shown, not summarised). Derived arithmetic is still not evidence about a real system: the arrival model is a fluid-plus-Zipf approximation, no hash function was executed against real keys, and no counter was allocated on real hardware. Where a quantity is a per-operation cost rather than an identity, it is flagged as an assumed constant at the point of use.

This version replaces an earlier draft whose interview block repeated one sentence twelve times and whose walkthrough contained a single queue formula. The topic, the identifier and the unit role are unchanged; the substance is new. Nothing in the earlier draft is carried forward as evidence.

## 1. The One-Line Framing

Given **one publication tick — a 60-second boundary at which a 300-second event-time window closes — and the sketch and counter state that survive from it**, decide **whether to publish a top-20 snapshot, hold the previously published snapshot, or publish with a provisional flag that forbids downstream commitments**, for the **streaming platform lead**, so that **the published ranking is reproducible from the retained state and its error is smaller than the gap between adjacent ranks**, subject to **a hard per-window memory budget of 64 MiB and a 60-second decision deadline**, compared with **recomputing the ranking by full re-sort of the raw window**.

That is a ranking-publication decision, taken 1,440 times per day, with a decision that is cheap to take and expensive to retract. It is not a modelling decision about traffic: the traffic model only fixes the *gap* between adjacent ranks, and the gap is the quantity that decides whether any approximate structure may be trusted to order the output at all.

The unit of analysis is the **published snapshot**, not the event, not the key and not the window. An event-level view hides the fact that one wrong snapshot can be committed against downstream money (a promoted listing, a bonus payout, a capacity plan). A window-level view hides the boundary: the interesting failures happen at the moment a window closes and two nearly-equal counts are ordered.

Three reusable patterns carry the case:

| Pattern | What it fixes | What it cannot fix |
|---|---|---|
| Bounded stream ranking | Whether the answer can be produced inside a memory budget | Whether the budget is large enough to resolve the ranks that matter |
| Sketch error contract | The maximum distance between an estimate and the truth | Enumeration of candidates the sketch never stored |
| Deterministic tie ordering | That two replays of the same state emit the same bytes | That the state itself was faithfully retained |

**Where the obvious framing fails.** The intuitive design is "keep a hash map of counts per key and sort at the end". Four defects make that design wrong here, and the case exists to separate them.

First, the exact map's size is set by *distinct keys in the window*, not by the event rate, so it grows with window length rather than with traffic; the same 9,600 events per second supports a 24 MiB map at five minutes and a 384 MiB map at six hours. Second, replacing the map with a sketch trades that growth for a *bounded but non-zero* error, and an error bound is useless until it is compared with the rank gap, which is a property of the skew and of how deep into the tail the publication reaches. Third, a per-shard design that keeps `K/S` candidates per shard instead of `K` silently drops globally winning keys, and the drop is invisible in the output because the published list still has the right length. Fourth, any comparator that is not a total order makes two replays of identical state disagree, which turns an audit into an argument.

**The decisive quantity.** Everything in the analysis reduces to one number per rank boundary:

```text
gap(r) = f(r) − f(r + 1)
```

where `f(r)` is the expected count of the key at rank `r` in one window. The gap is what separates "the sketch can order these two keys" from "the sketch is guessing". At rank 20 of the constructed workload the gap is 569.35 counts; at rank 500 it is 0.81 counts. A single sketch depth cannot serve both, and that asymmetry — not throughput — is what makes top-K a design problem rather than a library call.

**Interview cue.** A reader should be able to say in thirty seconds: "count per key per trailing window, keep `K` candidates per shard so the merge is complete, size the approximating structure so that twice its additive error is below the gap at the published depth, order ties by key identifier, and publish a provisional flag rather than a guess when the gap check fails."

**Acceptance for this opening.** Input, output, beneficiary, binding constraint, baseline and evidence basis are identifiable above; the three pattern names are the ones used by this unit's cheat sheet rather than invented here.

## 2. Context & Background

The system is a metered event pipeline that scores a product-analytics tenant's search terms. Every user search emits one event carrying a term identifier, an event timestamp, a tenant identifier and a payload of request metadata. The pipeline must publish, for the platform's demand dashboard, the twenty most-searched terms over a trailing five-minute window, refreshed every minute, and the same ranking feeds an automated "trending" slot that a merchandising team treats as an editorial input.

The decision owner is the streaming platform lead, who owns the ranking contract: what a published list means, at what error, at what cadence, and who may act on it. Other stakeholders matter because they consume the number rather than the mechanism.

| Context field | Illustrative content |
|---|---|
| Product | Demand-analytics dashboard plus an automated trending slot |
| Workflow before the tick | Collector batches events for up to 200 ms; router extracts the key and assigns a shard; shard state is updated |
| Workflow at the tick | Window closes at an event-time boundary; shard top-K lists are merged; gap check runs; snapshot is published or held |
| Workflow after publication | Dashboard renders; trending slot may change editorial ordering; both surfaces cache for 60 s |
| Decision owner | Streaming platform lead (ranking contract, error budget, publication authority) |
| Other stakeholders | Merchandising (consumes trending), tenant success (explains rank changes), finance (payouts tied to trending placement), on-call (paging on snapshot failures) |
| Baseline | Nightly batch re-sort of the raw event store, published once per day |
| Scale (illustrative) | 9,600 events/s mean, 18,000 events/s peak, 1.9M catalogue keys, 12 shards, one region |
| Decision cadence | 1,440 publication ticks per day at 60-second spacing |
| Decision deadline | Snapshot must be publishable within 2,000 ms of window close |
| Cost of a wrong publish | A trending slot change is a merchandising action; retracting it costs an editorial cycle, not a transaction rollback |
| Cost of a held publish | Dashboard shows a stale window; on-call gets a stale-snapshot alert after two consecutive holds |
| Evidence basis | Constructed workload; no production trace, no measured cardinality, no benchmark |

**Why this is not a sorting problem.** The obvious comparison is "sort the window, take twenty". That comparison is correct and unusable: it needs every key's exact count, which means retaining 66,356 entries per shard at five-minute granularity and 424,109 per shard at one-hour granularity in the constructed model. Sorting is also the wrong cost model — a comparison sort spends `log2 n` comparisons per element to answer a question whose answer depends only on the top of the distribution. The case therefore holds sorting as the *baseline oracle* rather than as the candidate design, which is a deliberate inversion of the usual framing: the expensive exact path is retained precisely because it is the only thing that can adjudicate a disputed snapshot.

**The skew premise.** Real search traffic is not uniform over a catalogue, and the whole analysis depends on that. The constructed model uses a Zipf law with exponent 1.05 over 1,900,000 keys, normalised by the harmonic-like sum `H = 10.873682`. Three consequences follow immediately and are used throughout: the top key carries 9.20% of all events, the top twenty carry 31.12%, and 53,022 of the 66,356 distinct keys seen per shard per window are seen exactly once. A design that is correct for uniform keys is not automatically correct here, and a design tuned to this skew does not transfer to a uniform-keyed workload — Section 6 shows both directions.

**What the baseline really costs.** A nightly re-sort is affordable because it runs once; the same machinery at 1,440 ticks per day is 1,440 full scans. The case's cost analysis (Section 6, block C-05) makes this explicit with a break-even point, because "batch is too slow" is a claim that should be quantified rather than asserted.

**Takeaway of Section 2:** context fixes three parameters that no amount of code quality can repair — the window length, the publication cadence, and the skew that sets the rank gaps.

## 3. The Business Question

| Field | Contract |
|---|---|
| Question | At this tick, may the team publish the computed top-20 snapshot? |
| Owner | Streaming platform lead |
| Trigger | A window closes and a candidate ranking exists in shard state |
| Actions | publish, hold (republish previous snapshot), publish with provisional flag |
| Objective | Maximise reproducible, error-bounded rankings per tick without exceeding the memory budget |
| Hard constraint | No publication whose error exceeds the gap at the published depth |
| Secondary constraint | Two replays of the same retained state must emit identical bytes |
| Scope | One trailing 300-second window, one region, one ranking |
| Out of scope | Personalisation, per-tenant rankings, revenue-weighted ranking |
| Baseline | Full re-sort of the raw window, once per day |
| Escalation | Missing shard state, gap check failure twice in a row, or a duplicate rate above 1% |

**The three actions are not cosmetic.** *Publish* means the ranking may drive the trending slot for the next minute. *Hold* means the last published ranking stays visible with its original timestamp; it does not mean "quietly keep an old number" — the timestamp is displayed, and a second consecutive hold raises a stale-snapshot alert. *Provisional* means the ranking is visible to humans as an estimate, tagged, and the automated slot is frozen. The distinction exists because the cheapest correct action under uncertainty is usually neither "publish" nor "nothing".

**What the question is not.** It is not "what is the true top twenty?". That question has an exact answer obtainable by replay, and it is the wrong question at the tick because the replay costs 0.1536 s of scan per rebuild but cannot run before the deadline while also serving the live path. The publishable question is "is *this* ranking within the error contract?", which is answerable from retained state in microseconds.

**Proxy metric warning.** Rank agreement with an offline re-sort is the natural metric and it is a proxy: a snapshot can agree on nineteen of twenty positions and still be unusable if the disagreement sits at rank 1, because the trending slot acts on rank 1 only. The case therefore scores by *weighted* position error, and Section 9 states the weighting as a contract rather than a preference.

**The authority boundary.** Software computes the candidate and the gap check; software does not decide that a failed gap check may be published anyway. The owner may grant a standing exception for a *named* depth (for example, "depth 20 may publish when the gap margin is at least 1.5") and that exception is versioned configuration, not judgement at the tick. This is the same authority pattern the unit's rate-limiter case uses: the mechanism is automatic, the exception is signed.

## 4. Assumptions & Constraints

Each assumption below is stated with the observation that would falsify it and the safe action when it is falsified. An assumption without a falsification test is a hope; the table is the audit surface for the whole case.

| ID | Assumption | Why it must hold | Verification | Safe action if violated |
|---|---|---|---|---|
| A-01 | Key extraction is a pure function of the event body | A key that depends on ambient state makes replays disagree | Hash 1,000 fixed events in two processes and compare key sets | Freeze publication; replay after the extractor is pinned |
| A-02 | Events carry a stable identifier usable for deduplication | At-least-once delivery inflates counts one-sidedly | Compare distinct identifiers with total events per window | Hold; publish the previous snapshot |
| A-03 | Event time exists and arrival lateness is bounded in practice | The window is event-time, so unbounded lateness means unbounded correction | Measure the lateness distribution on the retained sample | Widen the allowed lateness or switch the window to processing time with a stated caveat |
| A-04 | Catalogue size stays below 1.9M keys in the planning horizon | Cardinality drives the exact-map decision | Compare daily distinct-key counts | Re-derive the memory crossover before publishing |
| A-05 | Traffic skew is at least as concentrated as s = 1.05 | Rank gaps shrink toward zero under flat traffic | Fit the exponent on a recent window and compare gaps | Switch to exact counting or publish a wider error band |
| A-06 | The shard hash is stable across restarts and versions | An unstable hash splits one key across shards and halves its count | Assert fixed key-to-shard mappings in a test vector file | Repartition only by an explicit re-shard operation |
| A-07 | Shard count is fixed for the life of a window | Mid-window re-sharding double counts or loses state | Compare configured and observed shard sets at close | Hold until the window that straddles the change expires |
| A-08 | Counters are 64-bit signed integers | 32-bit counters overflow on multi-day cumulative rankings | Unit test the counter type and the accumulation path | Reset the accumulation and re-derive from the log |
| A-09 | The watermark advances monotonically | Non-monotone watermarks re-open closed windows | Assert monotonicity on the watermark stream | Drop the regressing watermark and alert |
| A-10 | Exactly one writer owns a shard per window | Two writers double-count the same event | Compare two writers' totals for one shard | Elect a single writer or deduplicate by identifier |
| A-11 | Per-window memory budget is 64 MiB across all shards | The exact map and the sketch differ by an order of magnitude | Compare high-water residency with the budget | Reduce window length or raise the budget with sign-off |
| A-12 | A 60-second cadence is acceptable to consumers | Cadence sets how long a wrong snapshot is visible | Confirm with the consuming team in writing | Change cadence, then re-derive cost per tick |

**Constraint C-1: bounded state.** Every window's state must be released when the window closes. A design that keeps history "just in case" converts a streaming problem into a storage problem and destroys the cost argument.

**Constraint C-2: no irreversible downstream action on an unverified ranking.** The trending slot consumes the ranking as an editorial signal; it must be frozen when the snapshot is provisional. This is a policy constraint, not a technical one, and it is the reason the three-action contract exists.

**Constraint C-3: the decision deadline of 2,000 ms.** Merging 12 shard lists cannot approach this budget in any plausible design, so the constraint is not binding on the merge; it is binding on any design that triggers a full recount at the tick. The recount path is 0.1536 s for a 138.24 MB projection, so a *single* recount fits and a retried recount does not. This asymmetry drives the retry policy in Section 8.

**Constraint C-4: reproducibility without the raw log.** The audit must be answerable from retained state alone, because the raw log is retained for a shorter period than the ranking history. This forces the snapshot to carry its own provenance (configuration hash, shard set, counter depth) rather than relying on being able to re-derive it later.

## 5. Data, Tools & Method

### 5a Data Specification

| Element | Specification |
|---|---|
| Grain | One event folded into one (shard, key) counter for one window |
| Event fields used at decision time | `event_id`, `event_time`, `term_id`, `tenant_id`, `collector_batch_id` |
| Event fields deliberately excluded | Session identifiers, user identifiers, query text, device metadata |
| Population | All search events in the trailing 300 s of event time, all tenants, one region |
| Cut-off timestamp | Window close at the event-time boundary; events with `event_time` beyond it belong to the next window |
| Keys | `term_id` (catalogue identifier), `shard_id` (hash of `term_id` mod 12), `window_id` |
| Units | Counts in events; memory in bytes; time in seconds; rates in events per second |
| Window convention | Half-open `[t0, t0 + 300)` in event time, buckets of 1 s |
| Publication grain | One ranked list of twenty `term_id` values with counts and a total event count |
| Not available at decision time | Events later than the watermark; exact counts of keys the sketch did not retain; per-tenant breakdowns; revenue weights |
| Retained for audit | Snapshot record, shard candidate lists, configuration hash, counter depth, total event count |

Two exclusions need to be stated rather than implied. **Catalogue attributes** (category, language, brand) are joined after publication by the dashboard, so a term whose catalogue row is missing still ranks; the join failure is a display defect, not a ranking defect, and mixing the two produces a false accuracy metric. **Tenant identity** is retained on the event but not in the ranking state, because per-tenant state multiplies memory by the tenant count and no consumer of this ranking needs it.

The *not-available* row is the one that decides the case. At the tick, the exact count of an arbitrary key is not available unless the key is in the exact map or the sketch; the sketch's estimate is available for tracked keys and a floor estimate for untracked ones. Everything the publish decision can use must therefore be expressible as a statement about estimates and their bounds, which is why Section 6 spends its effort on the bound rather than on the estimator's implementation.

### 5b Tools & Stack

| Component | Role | Version discipline |
|---|---|---|
| Collector | Batches events for at most 200 ms, assigns arrival time | Batch window in configuration, hashed into the snapshot |
| Key extractor | Maps event body to `term_id`, validates the catalogue identifier | Pinned pure function with a test vector file |
| Router | `shard_id = stable_hash(term_id) mod S` | Hash algorithm and `S` recorded in the snapshot |
| Window state | Per-shard sketch (or exact map) keyed by `term_id` | Depth and structure type recorded in the snapshot |
| Candidate heap | Per-shard top-`K` heap ordered by count then key | Comparator version recorded in the snapshot |
| Merger | `K × S` candidate merge with a global heap | Deterministic; no floating point |
| Gap checker | Compares sketch bound with the rank gap at the published depth | Threshold is versioned configuration |
| Snapshot store | Append-only records with idempotency on `(window_id, config_hash)` | Immutable; the only durable authority |

The stack is deliberately dull. No component is a database, no component coordinates with another, and the only shared state between ticks is the snapshot store. That choice is what makes the failure modes in Section 12 diagnosable: each one can be reproduced by replaying a single shard against a single window, without reconstructing a distributed conversation.

### 5c Method Summary

The method is a five-step loop per tick, with a fixed order that matters:

1. **Close** the window at the event-time boundary and stop accepting events for it.
2. **Merge** the `S` per-shard candidate lists into one ranked list of twenty, using the total order defined below.
3. **Check** the error bound against the rank gap at the published depth and at one rank beyond it.
4. **Decide** publish, hold, or provisional, and record the reason code with the numbers that produced it.
5. **Release** the window state and retain only the snapshot record and the shard candidate lists.

**The total order.** Two keys are ordered by count descending, then by `term_id` ascending as an unsigned integer. The second component is not decoration: it is what makes the output a function of the retained state rather than of hash iteration order. Every structure in the pipeline — sketch, heap, merge — must use this comparator, and Section 9's determinism test exists to catch the case where one of them silently does not.

**Method-selection rationale.** Three candidate methods were considered and are compared explicitly rather than dismissed:

| Method | Correctness | State per window | Enumerates candidates | Where it fails |
|---|---|---|---|---|
| Exact map per shard | Exact | Proportional to distinct keys | Yes | Memory grows with window length |
| Space-Saving sketch | Bounded overestimate | Fixed, set by depth | Yes | Error depends on the key, not on the request |
| Count-min sketch | Bounded one-sided error | Fixed, set by error and confidence | No | Cannot report which keys are large |

The third row is the trap. A count-min sketch answers "what is the estimate for this key?" and never "which keys should I look at?", so it needs a candidate generator in front of it — which is a second structure with its own error. Using it alone produces a design that estimates nothing, because it has nothing to estimate.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** Every published snapshot records: the window identifier, the configuration hash (shard count, sketch depth, comparator version, gap threshold), the total event count of the window, the per-shard candidate lists with their estimates, the merge order, and the decision with its reason code. A snapshot without a configuration hash is not auditable, because the same input under a different depth produces a different answer and the record would not say which.

**Quality gates before state enters the window.** Reject events with a missing `term_id`; reject events whose `event_time` is more than 300 s behind the watermark (they belong to a closed window and would double count); reject duplicate `event_id` values within the batch; reject events whose extractor version is not the pinned one. Each rejection increments a counted reason, and a rejection rate above 1% of the batch holds publication, because a rejected event is a missing count and missing counts are one-sided.

**Exclusions.** The ranking excludes terms on a suppression list (legal takedowns, internal test terms). Exclusion is applied **before** counting, not after, so that the published total event count matches the sum of the published counts' population. Applying it after counting creates a ranking whose displayed percentages do not add to the displayed total, which is the most common way a leaderboard loses credibility.

**Leakage.** There is no label and no model, so classical leakage does not apply; the analogous defect is *future information in the retained state*. Two concrete leaks are excluded by construction: the window state may not be updated with events stamped after the window close, and the gap threshold may not be tuned on the window it is judging. The second is subtler and is why the threshold is configuration with a version, not a per-window computation.

**Sampling.** No sampling is used in the counts. A design that samples events to save CPU must scale every count by the inverse sampling probability, and the resulting non-integer counts break the total order's tie-breaking unless the comparator is defined on the scaled value. That is a real design, and it is out of scope here with a stated reason rather than left ambiguous.

### 5e Reproducibility & Method-Selection Ledger

| Artifact | Purpose | Accessibility | Limitation |
|---|---|---|---|
| Model-M1 | Zipf-s=1.05 key generator over 1.9M keys | Constructed, deterministic seed | Not a traffic trace |
| Oracle-O1 | Exact per-key count from a full window replay | Constructed; runnable in minutes | Validates only ranking logic, not capacity |
| Oracle-O2 | Brute-force merge from exact per-shard maps | Constructed | Shares the comparator under test |
| Test-T1 | Fixed key-to-shard vector file | Committed with the code | Catches hash drift, not hash quality |
| Test-T2 | Two-process, three-replay determinism check | Constructed | Detects nondeterminism, not incorrectness |
| Config-C1 | Depth, shard count, threshold, comparator version | Versioned | Requires change control |

The two oracles differ in what they can refute. **O1** can refute a published count; it cannot refute a published *order* produced by a comparator that is consistently wrong, because the replay uses the same comparator. **O2** closes that gap only if the merge is written independently of the production merger, which is why the ledger lists them separately: an oracle that shares code with the system under test is a regression test, not an oracle.

**Reproducibility statement.** Given `Model-M1`, the configuration hash and the comparator, the full pipeline is a pure function of the seed. That is a property of the construction, and it is what allows Section 9's checks to be *designed* rather than merely described. It is not a claim that a production pipeline is pure: production adds arrival order, retries and clock skew, and each of those appears as a failure mechanism in Section 12.
