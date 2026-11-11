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

## 6. Analysis Walkthrough

The following derived quantities use ILLUSTRATIVE inputs only; they are design arithmetic rather than measurements.

```text
FORMULA [TK-01]: N = lambda * W
WHERE: N [events]; lambda [events/s]; W [s].
CONSTRAINT: constant mean arrival rate over the window; no sampling or loss.
NUMERICAL EXAMPLE: lambda=9,600 events/s; W=300 s; N=9,600*300=2,880,000 events.
INTERPRETATION: this is event volume, not distinct-key cardinality.
```

Independent check: 9,600 events/s multiplied by 5 minutes multiplied by 60 s/min equals 2,880,000 events.

```text
FORMULA [TK-02]: E = ceil(N / m)
WHERE: E [events/cell] is an additive sketch scale; m [cells] is retained capacity.
CONSTRAINT: this simplified scale is illustrative; a real sketch needs its documented bound and confidence.
NUMERICAL EXAMPLE: N=2,880,000; m=24,000; E=ceil(120)=120 events.
INTERPRETATION: a 120-event one-sided uncertainty cannot settle a gap smaller than 240 events.
```

Boundary: if collision behavior violates the chosen sketch's assumptions, this scale is not an error certificate.

```text
FORMULA [TK-03]: G = f(20) - f(21)
WHERE: G [events] is the boundary gap; f(r) [events] is the exact count at rank r.
CONSTRAINT: ranks use the same total comparator and the same closed window.
NUMERICAL EXAMPLE: f(20)=1,842; f(21)=1,511; G=1,842-1,511=331 events.
INTERPRETATION: two-sided uncertainty below 331 events is required to prove this ordering.
```

Sensitivity: if the gap falls to 180 events while the uncertainty remains 120 events on each side, 240 exceeds 180 and the boundary is unresolved.

```text
FORMULA [TK-04]: M = S * K * b
WHERE: M [bytes]; S [shards]; K [candidates/shard]; b [bytes/candidate].
CONSTRAINT: payload-only sizing; heaps, object headers, and dedup state are excluded.
NUMERICAL EXAMPLE: S=12; K=80; b=48 bytes; M=12*80*48=46,080 bytes.
INTERPRETATION: candidate transport is small, but it is not the whole memory budget.
```

Independent check: 960 candidates times 48 bytes per candidate equals 46,080 bytes, approximately 45 KiB.

**Walkthrough control: window membership.** Assign an event only when its event time is inside the half-open interval; processing arrival is not membership.
Invariant: Each accepted event has exactly one window_id.
Diagnostic: inspect event_time, watermark, and assigned window_id before accepting the snapshot.
Decision consequence: window membership changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise window membership beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for window membership with the window identifier and configuration hash.
Recovery: when window membership is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate window membership against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: window membership is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: late-event policy.** Route an event arriving after close to a correction ledger rather than reopening retained state invisibly.
Invariant: A closed snapshot remains immutable.
Diagnostic: inspect late event count and correction-ledger key before accepting the snapshot.
Decision consequence: late-event policy changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise late-event policy beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for late-event policy with the window identifier and configuration hash.
Recovery: when late-event policy is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate late-event policy against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: late-event policy is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: watermark advance.** Close only after the watermark crosses the end boundary plus the stated lateness allowance.
Invariant: Watermarks advance monotonically.
Diagnostic: inspect watermark sequence and allowed lateness before accepting the snapshot.
Decision consequence: watermark advance changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise watermark advance beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for watermark advance with the window identifier and configuration hash.
Recovery: when watermark advance is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate watermark advance against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: watermark advance is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: deduplication scope.** Deduplicate event_id within the window before incrementing a counter, including replayed collector batches.
Invariant: One logical event contributes at most one count.
Diagnostic: inspect event_id cardinality versus accepted-event total before accepting the snapshot.
Decision consequence: deduplication scope changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise deduplication scope beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for deduplication scope with the window identifier and configuration hash.
Recovery: when deduplication scope is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate deduplication scope against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: deduplication scope is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: stable key extraction.** Normalize term_id before hashing so equivalent encodings do not become separate candidates.
Invariant: The extractor is pure and version-pinned.
Diagnostic: inspect extractor version and normalization rejects before accepting the snapshot.
Decision consequence: stable key extraction changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise stable key extraction beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for stable key extraction with the window identifier and configuration hash.
Recovery: when stable key extraction is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate stable key extraction against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: stable key extraction is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: shard routing.** Route every occurrence of a key through the same stable hash during a window.
Invariant: A key has one shard owner per window.
Diagnostic: inspect key-to-shard test-vector result before accepting the snapshot.
Decision consequence: shard routing changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise shard routing beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for shard routing with the window identifier and configuration hash.
Recovery: when shard routing is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate shard routing against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: shard routing is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: candidate completeness.** Keep enough shard candidates to ensure a globally winning key can reach the merger.
Invariant: No candidate is discarded solely because another shard is quiet.
Diagnostic: inspect per-shard candidate cutoff and omitted-count bound before accepting the snapshot.
Decision consequence: candidate completeness changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise candidate completeness beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for candidate completeness with the window identifier and configuration hash.
Recovery: when candidate completeness is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate candidate completeness against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: candidate completeness is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: heap admission.** Replace the heap minimum only when the incoming pair wins under count-descending then key-ascending order.
Invariant: The heap root is the worst retained candidate.
Diagnostic: inspect heap root before and after admission before accepting the snapshot.
Decision consequence: heap admission changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise heap admission beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for heap admission with the window identifier and configuration hash.
Recovery: when heap admission is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate heap admission against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: heap admission is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: tie handling.** Use the unsigned term_id as a deterministic secondary order at every sort and merge boundary.
Invariant: Equal counts produce a unique byte ordering.
Diagnostic: inspect serialized ordered key sequence before accepting the snapshot.
Decision consequence: tie handling changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise tie handling beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for tie handling with the window identifier and configuration hash.
Recovery: when tie handling is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate tie handling against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: tie handling is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: sketch interpretation.** Treat a sketch estimate as an estimate with a stated additive error, not as an exact count.
Invariant: The error parameter is carried into publication logic.
Diagnostic: inspect depth, width, and bound value before accepting the snapshot.
Decision consequence: sketch interpretation changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise sketch interpretation beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for sketch interpretation with the window identifier and configuration hash.
Recovery: when sketch interpretation is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate sketch interpretation against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: sketch interpretation is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: rank-gap gate.** Compare the lower-versus-upper possibility at ranks 20 and 21 before authorising automation.
Invariant: An unresolved boundary cannot drive the trending slot.
Diagnostic: inspect rank-20 and rank-21 intervals before accepting the snapshot.
Decision consequence: rank-gap gate changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise rank-gap gate beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for rank-gap gate with the window identifier and configuration hash.
Recovery: when rank-gap gate is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate rank-gap gate against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: rank-gap gate is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: merge strategy.** Merge sorted shard lists with a global heap instead of repeatedly sorting their concatenation.
Invariant: The merger consumes a deterministic candidate multiset.
Diagnostic: inspect candidate count and merge comparator version before accepting the snapshot.
Decision consequence: merge strategy changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise merge strategy beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for merge strategy with the window identifier and configuration hash.
Recovery: when merge strategy is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate merge strategy against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: merge strategy is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: state release.** Release mutable window state only after the immutable decision record is committed.
Invariant: Recovery never depends on reclaimed counters.
Diagnostic: inspect commit acknowledgement before release before accepting the snapshot.
Decision consequence: state release changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise state release beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for state release with the window identifier and configuration hash.
Recovery: when state release is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate state release against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: state release is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: snapshot idempotency.** Key snapshot writes by window_id and configuration hash, then reject conflicting payloads.
Invariant: A retried close cannot create two authorities.
Diagnostic: inspect idempotency key and payload digest before accepting the snapshot.
Decision consequence: snapshot idempotency changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise snapshot idempotency beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for snapshot idempotency with the window identifier and configuration hash.
Recovery: when snapshot idempotency is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate snapshot idempotency against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: snapshot idempotency is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: clock separation.** Store event-time close and processing-time publication separately.
Invariant: Latency is never mistaken for window membership.
Diagnostic: inspect close timestamp and commit timestamp before accepting the snapshot.
Decision consequence: clock separation changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise clock separation beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for clock separation with the window identifier and configuration hash.
Recovery: when clock separation is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate clock separation against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: clock separation is a ranking-contract concern, not evidence of an observed production result.

**Walkthrough control: capacity headroom.** Reserve memory for heap metadata, dedup state, and serialization rather than budgeting only counters.
Invariant: The 64 MiB cap applies to resident state.
Diagnostic: inspect high-water bytes by component before accepting the snapshot.
Decision consequence: capacity headroom changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise capacity headroom beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for capacity headroom with the window identifier and configuration hash.
Recovery: when capacity headroom is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate capacity headroom against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: capacity headroom is a ranking-contract concern, not evidence of an observed production result.

## 7. The Finding

The proposed finding is conditional: an exact or bounded-error top-20 may be published only when its candidate set is complete, all shards attest, the uncertainty intervals do not overlap at the twentieth boundary, and the deterministic digest is reproducible. These conditions have not been executed on a production stream.

**Finding condition: window membership.** Assign an event only when its event time is inside the half-open interval; processing arrival is not membership.
Invariant: Each accepted event has exactly one window_id.
Diagnostic: inspect event_time, watermark, and assigned window_id before accepting the snapshot.
Decision consequence: window membership changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise window membership beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for window membership with the window identifier and configuration hash.
Recovery: when window membership is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate window membership against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: window membership is a ranking-contract concern, not evidence of an observed production result.

**Finding condition: late-event policy.** Route an event arriving after close to a correction ledger rather than reopening retained state invisibly.
Invariant: A closed snapshot remains immutable.
Diagnostic: inspect late event count and correction-ledger key before accepting the snapshot.
Decision consequence: late-event policy changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise late-event policy beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for late-event policy with the window identifier and configuration hash.
Recovery: when late-event policy is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate late-event policy against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: late-event policy is a ranking-contract concern, not evidence of an observed production result.

**Finding condition: watermark advance.** Close only after the watermark crosses the end boundary plus the stated lateness allowance.
Invariant: Watermarks advance monotonically.
Diagnostic: inspect watermark sequence and allowed lateness before accepting the snapshot.
Decision consequence: watermark advance changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise watermark advance beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for watermark advance with the window identifier and configuration hash.
Recovery: when watermark advance is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate watermark advance against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: watermark advance is a ranking-contract concern, not evidence of an observed production result.

**Finding condition: deduplication scope.** Deduplicate event_id within the window before incrementing a counter, including replayed collector batches.
Invariant: One logical event contributes at most one count.
Diagnostic: inspect event_id cardinality versus accepted-event total before accepting the snapshot.
Decision consequence: deduplication scope changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise deduplication scope beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for deduplication scope with the window identifier and configuration hash.
Recovery: when deduplication scope is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate deduplication scope against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: deduplication scope is a ranking-contract concern, not evidence of an observed production result.

**Finding condition: stable key extraction.** Normalize term_id before hashing so equivalent encodings do not become separate candidates.
Invariant: The extractor is pure and version-pinned.
Diagnostic: inspect extractor version and normalization rejects before accepting the snapshot.
Decision consequence: stable key extraction changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise stable key extraction beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for stable key extraction with the window identifier and configuration hash.
Recovery: when stable key extraction is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate stable key extraction against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: stable key extraction is a ranking-contract concern, not evidence of an observed production result.

**Finding condition: shard routing.** Route every occurrence of a key through the same stable hash during a window.
Invariant: A key has one shard owner per window.
Diagnostic: inspect key-to-shard test-vector result before accepting the snapshot.
Decision consequence: shard routing changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise shard routing beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for shard routing with the window identifier and configuration hash.
Recovery: when shard routing is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate shard routing against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: shard routing is a ranking-contract concern, not evidence of an observed production result.

**Finding condition: candidate completeness.** Keep enough shard candidates to ensure a globally winning key can reach the merger.
Invariant: No candidate is discarded solely because another shard is quiet.
Diagnostic: inspect per-shard candidate cutoff and omitted-count bound before accepting the snapshot.
Decision consequence: candidate completeness changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise candidate completeness beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for candidate completeness with the window identifier and configuration hash.
Recovery: when candidate completeness is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate candidate completeness against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: candidate completeness is a ranking-contract concern, not evidence of an observed production result.

**Finding condition: heap admission.** Replace the heap minimum only when the incoming pair wins under count-descending then key-ascending order.
Invariant: The heap root is the worst retained candidate.
Diagnostic: inspect heap root before and after admission before accepting the snapshot.
Decision consequence: heap admission changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise heap admission beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for heap admission with the window identifier and configuration hash.
Recovery: when heap admission is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate heap admission against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: heap admission is a ranking-contract concern, not evidence of an observed production result.

## 8. System/Solution Design

**Input-to-output architecture.** Collector validation → event-time/dedup gate → stable partition router → per-shard counters and candidate heap → close barrier → global merge → error and completeness gate → immutable snapshot store → dashboard plus automation fence. The arrow denotes data dependency, not a claim that a deployed service exists.

**Design component: input envelope.** Validate event_id, event_time, term_id, and schema version at ingress.
Invariant: Invalid events never mutate ranking state.
Diagnostic: inspect rejection reason counters before accepting the snapshot.
Decision consequence: input envelope changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise input envelope beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for input envelope with the window identifier and configuration hash.
Recovery: when input envelope is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate input envelope against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: input envelope is a ranking-contract concern, not evidence of an observed production result.

**Design component: event-time bucketing.** Place accepted events into one-second event-time buckets to make expiry explicit.
Invariant: Bucket expiration is deterministic.
Diagnostic: inspect bucket index versus watermark before accepting the snapshot.
Decision consequence: event-time bucketing changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise event-time bucketing beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for event-time bucketing with the window identifier and configuration hash.
Recovery: when event-time bucketing is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate event-time bucketing against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: event-time bucketing is a ranking-contract concern, not evidence of an observed production result.

**Design component: dedup store.** Use a bounded identifier store whose retention matches the allowed lateness, not the full history.
Invariant: Duplicate detection covers the correction horizon.
Diagnostic: inspect dedup eviction age before accepting the snapshot.
Decision consequence: dedup store changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise dedup store beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for dedup store with the window identifier and configuration hash.
Recovery: when dedup store is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate dedup store against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: dedup store is a ranking-contract concern, not evidence of an observed production result.

**Design component: counter engine.** Use exact counters for retained candidates and sketches only where their error contract is recorded.
Invariant: Counter type is declared per field.
Diagnostic: inspect counter implementation tag before accepting the snapshot.
Decision consequence: counter engine changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise counter engine beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for counter engine with the window identifier and configuration hash.
Recovery: when counter engine is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate counter engine against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: counter engine is a ranking-contract concern, not evidence of an observed production result.

**Design component: per-shard heap.** Maintain a bounded heap with the same comparator used at publication.
Invariant: Local ordering is compatible with global ordering.
Diagnostic: inspect comparator hash before accepting the snapshot.
Decision consequence: per-shard heap changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise per-shard heap beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for per-shard heap with the window identifier and configuration hash.
Recovery: when per-shard heap is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate per-shard heap against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: per-shard heap is a ranking-contract concern, not evidence of an observed production result.

**Design component: candidate transport.** Transmit key, estimate, lower bound, upper bound, and shard provenance together.
Invariant: The merger receives uncertainty with each candidate.
Diagnostic: inspect candidate schema checksum before accepting the snapshot.
Decision consequence: candidate transport changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise candidate transport beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for candidate transport with the window identifier and configuration hash.
Recovery: when candidate transport is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate candidate transport against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: candidate transport is a ranking-contract concern, not evidence of an observed production result.

**Design component: global merger.** Deduplicate candidates by key before ranking because one key may appear in recovery records.
Invariant: Each key has one merged representation.
Diagnostic: inspect merge duplicate counter before accepting the snapshot.
Decision consequence: global merger changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise global merger beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for global merger with the window identifier and configuration hash.
Recovery: when global merger is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate global merger against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: global merger is a ranking-contract concern, not evidence of an observed production result.

**Design component: decision gate.** Apply schema, shard-completeness, error-gap, and idempotency gates in a fixed order.
Invariant: A failed earlier gate cannot be hidden by a later pass.
Diagnostic: inspect ordered reason-code trace before accepting the snapshot.
Decision consequence: decision gate changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise decision gate beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for decision gate with the window identifier and configuration hash.
Recovery: when decision gate is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate decision gate against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: decision gate is a ranking-contract concern, not evidence of an observed production result.

**Design component: snapshot serializer.** Encode counts as integers and identifiers canonically; never serialize heap iteration order.
Invariant: Equal state yields identical bytes.
Diagnostic: inspect digest across replays before accepting the snapshot.
Decision consequence: snapshot serializer changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise snapshot serializer beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for snapshot serializer with the window identifier and configuration hash.
Recovery: when snapshot serializer is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate snapshot serializer against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: snapshot serializer is a ranking-contract concern, not evidence of an observed production result.

**Design component: publication channel.** Publish an immutable snapshot reference, not mutable ranking state.
Invariant: Consumers can retrieve the audited object.
Diagnostic: inspect reference resolution success before accepting the snapshot.
Decision consequence: publication channel changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise publication channel beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for publication channel with the window identifier and configuration hash.
Recovery: when publication channel is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate publication channel against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: publication channel is a ranking-contract concern, not evidence of an observed production result.

**Design component: automation fence.** Allow the dashboard to show provisional data while blocking the automated trending action.
Invariant: Human visibility is separated from automated commitment.
Diagnostic: inspect fence status in consumer request before accepting the snapshot.
Decision consequence: automation fence changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise automation fence beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for automation fence with the window identifier and configuration hash.
Recovery: when automation fence is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate automation fence against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: automation fence is a ranking-contract concern, not evidence of an observed production result.

**Design component: correction stream.** Emit late-event corrections as a new version with linkage to the affected closed window.
Invariant: Corrections never overwrite the original authority.
Diagnostic: inspect supersedes reference before accepting the snapshot.
Decision consequence: correction stream changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise correction stream beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for correction stream with the window identifier and configuration hash.
Recovery: when correction stream is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate correction stream against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: correction stream is a ranking-contract concern, not evidence of an observed production result.

**Design component: configuration registry.** Version shard count, comparator, lateness, K, and error tolerance as one reviewed bundle.
Invariant: A snapshot names exactly one contract.
Diagnostic: inspect configuration digest before accepting the snapshot.
Decision consequence: configuration registry changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise configuration registry beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for configuration registry with the window identifier and configuration hash.
Recovery: when configuration registry is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate configuration registry against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: configuration registry is a ranking-contract concern, not evidence of an observed production result.

**Design component: recovery checkpoint.** Persist enough state to resume a window or abandon it deterministically after a crash.
Invariant: Recovery chooses replay or hold explicitly.
Diagnostic: inspect checkpoint age and completeness before accepting the snapshot.
Decision consequence: recovery checkpoint changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise recovery checkpoint beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for recovery checkpoint with the window identifier and configuration hash.
Recovery: when recovery checkpoint is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate recovery checkpoint against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: recovery checkpoint is a ranking-contract concern, not evidence of an observed production result.

**Design component: backpressure boundary.** Throttle ingress before state exhaustion rather than letting allocation failure decide the policy.
Invariant: Overload produces an observable admission decision.
Diagnostic: inspect queue depth and rejection rate before accepting the snapshot.
Decision consequence: backpressure boundary changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise backpressure boundary beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for backpressure boundary with the window identifier and configuration hash.
Recovery: when backpressure boundary is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate backpressure boundary against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: backpressure boundary is a ranking-contract concern, not evidence of an observed production result.

**Design component: partition migration.** Start a new epoch for shard-count changes and prevent an epoch from spanning a window.
Invariant: No key is counted under two hash domains.
Diagnostic: inspect epoch id at window close before accepting the snapshot.
Decision consequence: partition migration changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise partition migration beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for partition migration with the window identifier and configuration hash.
Recovery: when partition migration is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate partition migration against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: partition migration is a ranking-contract concern, not evidence of an observed production result.

**Design component: retention policy.** Retain snapshot provenance longer than raw mutable state so audits remain possible.
Invariant: Auditing does not require resurrecting a heap.
Diagnostic: inspect retention class and expiry job before accepting the snapshot.
Decision consequence: retention policy changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise retention policy beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for retention policy with the window identifier and configuration hash.
Recovery: when retention policy is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate retention policy against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: retention policy is a ranking-contract concern, not evidence of an observed production result.

## 9. Evaluation & Validation

Evaluation is a proposed protocol, not an executed experiment. The numerator, denominator, population, window, comparator, guardrails, uncertainty, and status are specified so a later operator cannot turn a convenient sample into a success claim.

**Validation contract: rank-set agreement.** Compute the numerator as keys shared with the exact oracle top-20 and denominator as 20.
Invariant: The population is closed windows selected before replay.
Diagnostic: inspect set overlap and oracle version before accepting the snapshot.
Decision consequence: rank-set agreement changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise rank-set agreement beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for rank-set agreement with the window identifier and configuration hash.
Recovery: when rank-set agreement is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate rank-set agreement against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: rank-set agreement is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: weighted position error.** Weight a position difference by 1/rank so rank-one disagreement is more visible than rank-twenty disagreement.
Invariant: The comparator baseline is the exact replay.
Diagnostic: inspect per-rank displacement vector before accepting the snapshot.
Decision consequence: weighted position error changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise weighted position error beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for weighted position error with the window identifier and configuration hash.
Recovery: when weighted position error is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate weighted position error against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: weighted position error is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: publication eligibility.** Measure eligible ticks divided by all scheduled ticks, excluding none after collection begins.
Invariant: A hold is a denominator event, not missing data.
Diagnostic: inspect scheduled tick count and decision code before accepting the snapshot.
Decision consequence: publication eligibility changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise publication eligibility beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for publication eligibility with the window identifier and configuration hash.
Recovery: when publication eligibility is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate publication eligibility against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: publication eligibility is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: staleness guardrail.** Measure wall-clock age of the last accepted snapshot at consumer read time.
Invariant: Two consecutive holds trigger an alert.
Diagnostic: inspect snapshot age and hold streak before accepting the snapshot.
Decision consequence: staleness guardrail changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise staleness guardrail beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for staleness guardrail with the window identifier and configuration hash.
Recovery: when staleness guardrail is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate staleness guardrail against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: staleness guardrail is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: duplicate guardrail.** Measure rejected duplicate event_ids divided by received event_ids in each window.
Invariant: The threshold is a policy setting, not an accuracy score.
Diagnostic: inspect duplicate numerator and total arrivals before accepting the snapshot.
Decision consequence: duplicate guardrail changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise duplicate guardrail beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for duplicate guardrail with the window identifier and configuration hash.
Recovery: when duplicate guardrail is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate duplicate guardrail against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: duplicate guardrail is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: late-data guardrail.** Measure events arriving after the watermark divided by all valid arrivals.
Invariant: Late rate is reported separately from invalid-schema rate.
Diagnostic: inspect lateness histogram before accepting the snapshot.
Decision consequence: late-data guardrail changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise late-data guardrail beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for late-data guardrail with the window identifier and configuration hash.
Recovery: when late-data guardrail is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate late-data guardrail against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: late-data guardrail is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: memory guardrail.** Compare peak resident bytes with 64 MiB including allocator and serialization headroom.
Invariant: A mean memory reading cannot certify a cap.
Diagnostic: inspect per-component high-water marks before accepting the snapshot.
Decision consequence: memory guardrail changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise memory guardrail beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for memory guardrail with the window identifier and configuration hash.
Recovery: when memory guardrail is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate memory guardrail against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: memory guardrail is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: deadline guardrail.** Measure close-to-commit elapsed milliseconds against 2,000 ms at every tick.
Invariant: Processing time does not alter event-time membership.
Diagnostic: inspect close and commit timestamps before accepting the snapshot.
Decision consequence: deadline guardrail changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise deadline guardrail beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for deadline guardrail with the window identifier and configuration hash.
Recovery: when deadline guardrail is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate deadline guardrail against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: deadline guardrail is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: determinism check.** Replay identical retained state in independent processes and compare bytes, not just displayed rows.
Invariant: The sample includes tied counts and empty shards.
Diagnostic: inspect snapshot digest pair before accepting the snapshot.
Decision consequence: determinism check changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise determinism check beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for determinism check with the window identifier and configuration hash.
Recovery: when determinism check is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate determinism check against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: determinism check is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: partition check.** Compare the configured shard set with the set contributing to a close.
Invariant: A missing shard invalidates the output.
Diagnostic: inspect expected and observed shard ids before accepting the snapshot.
Decision consequence: partition check changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise partition check beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for partition check with the window identifier and configuration hash.
Recovery: when partition check is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate partition check against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: partition check is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: uncertainty status.** Report whether each metric is illustrative, executed, or not-executed.
Invariant: No constructed oracle result is represented as a live outcome.
Diagnostic: inspect evidence label before accepting the snapshot.
Decision consequence: uncertainty status changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise uncertainty status beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for uncertainty status with the window identifier and configuration hash.
Recovery: when uncertainty status is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate uncertainty status against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: uncertainty status is a ranking-contract concern, not evidence of an observed production result.

**Validation contract: sensitivity sweep.** Vary skew, K, window length, lateness, and cardinality one at a time in a constructed analysis.
Invariant: A decision boundary is recorded with its assumptions.
Diagnostic: inspect parameter grid and crossing point before accepting the snapshot.
Decision consequence: sensitivity sweep changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise sensitivity sweep beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for sensitivity sweep with the window identifier and configuration hash.
Recovery: when sensitivity sweep is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate sensitivity sweep against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: sensitivity sweep is a ranking-contract concern, not evidence of an observed production result.

## 10. Action Taken

No operational deployment was performed for this illustrative case. The recommended action is to adopt the following decision record only after the named test evidence exists.

**Operational action: change approval.** Require a reviewed configuration digest before changing K, error tolerance, or shard count.
Invariant: Unreviewed configuration cannot publish.
Diagnostic: inspect approval reference before accepting the snapshot.
Decision consequence: change approval changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise change approval beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for change approval with the window identifier and configuration hash.
Recovery: when change approval is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate change approval against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: change approval is a ranking-contract concern, not evidence of an observed production result.

**Operational action: pre-close checklist.** Check watermark, shard membership, memory headroom, and comparator version before merge.
Invariant: Close gates are evaluated from durable inputs.
Diagnostic: inspect pre-close checklist record before accepting the snapshot.
Decision consequence: pre-close checklist changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise pre-close checklist beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for pre-close checklist with the window identifier and configuration hash.
Recovery: when pre-close checklist is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate pre-close checklist against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: pre-close checklist is a ranking-contract concern, not evidence of an observed production result.

**Operational action: hold behavior.** Republish the last accepted snapshot with its original timestamp and a current hold reason.
Invariant: A hold never masquerades as fresh data.
Diagnostic: inspect stale marker and reason code before accepting the snapshot.
Decision consequence: hold behavior changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise hold behavior beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for hold behavior with the window identifier and configuration hash.
Recovery: when hold behavior is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate hold behavior against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: hold behavior is a ranking-contract concern, not evidence of an observed production result.

**Operational action: provisional behavior.** Expose the candidate to humans with uncertainty while the automation fence remains closed.
Invariant: Provisional output cannot trigger ranking-dependent action.
Diagnostic: inspect consumer fence decision before accepting the snapshot.
Decision consequence: provisional behavior changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise provisional behavior beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for provisional behavior with the window identifier and configuration hash.
Recovery: when provisional behavior is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate provisional behavior against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: provisional behavior is a ranking-contract concern, not evidence of an observed production result.

**Operational action: replay escalation.** Use the exact oracle only as an asynchronous adjudication path after a disputed close.
Invariant: Replay does not retroactively alter the original record.
Diagnostic: inspect replay request and result linkage before accepting the snapshot.
Decision consequence: replay escalation changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise replay escalation beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for replay escalation with the window identifier and configuration hash.
Recovery: when replay escalation is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate replay escalation against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: replay escalation is a ranking-contract concern, not evidence of an observed production result.

**Operational action: on-call handoff.** Page after two holds, a missing shard, or idempotency conflict with snapshot evidence attached.
Invariant: Pages name a mechanism, not a vague accuracy alarm.
Diagnostic: inspect page reason and evidence reference before accepting the snapshot.
Decision consequence: on-call handoff changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise on-call handoff beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for on-call handoff with the window identifier and configuration hash.
Recovery: when on-call handoff is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate on-call handoff against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: on-call handoff is a ranking-contract concern, not evidence of an observed production result.

**Operational action: runbook ownership.** Assign the platform lead authority for contract changes and on-call authority for recovery execution.
Invariant: No operator invents an error tolerance during an incident.
Diagnostic: inspect owner and escalation timestamp before accepting the snapshot.
Decision consequence: runbook ownership changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise runbook ownership beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for runbook ownership with the window identifier and configuration hash.
Recovery: when runbook ownership is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate runbook ownership against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: runbook ownership is a ranking-contract concern, not evidence of an observed production result.

**Operational action: retrospective record.** Classify each failure as data, state, partition, comparator, or capacity before remediation.
Invariant: Corrective work targets a discriminated cause.
Diagnostic: inspect incident classification before accepting the snapshot.
Decision consequence: retrospective record changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise retrospective record beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for retrospective record with the window identifier and configuration hash.
Recovery: when retrospective record is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate retrospective record against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: retrospective record is a ranking-contract concern, not evidence of an observed production result.

## 11. Outcome & Results

**Result status: not-measured.** No company stream, live snapshot, latency distribution, cost invoice, or accuracy result was supplied or collected. The numerical examples are ILLUSTRATIVE derivations and demonstrate only the arithmetic under their stated assumptions.

**Outcome interpretation: change approval.** Require a reviewed configuration digest before changing K, error tolerance, or shard count.
Invariant: Unreviewed configuration cannot publish.
Diagnostic: inspect approval reference before accepting the snapshot.
Decision consequence: change approval changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise change approval beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for change approval with the window identifier and configuration hash.
Recovery: when change approval is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate change approval against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: change approval is a ranking-contract concern, not evidence of an observed production result.

**Outcome interpretation: pre-close checklist.** Check watermark, shard membership, memory headroom, and comparator version before merge.
Invariant: Close gates are evaluated from durable inputs.
Diagnostic: inspect pre-close checklist record before accepting the snapshot.
Decision consequence: pre-close checklist changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise pre-close checklist beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for pre-close checklist with the window identifier and configuration hash.
Recovery: when pre-close checklist is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate pre-close checklist against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: pre-close checklist is a ranking-contract concern, not evidence of an observed production result.

**Outcome interpretation: hold behavior.** Republish the last accepted snapshot with its original timestamp and a current hold reason.
Invariant: A hold never masquerades as fresh data.
Diagnostic: inspect stale marker and reason code before accepting the snapshot.
Decision consequence: hold behavior changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise hold behavior beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for hold behavior with the window identifier and configuration hash.
Recovery: when hold behavior is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate hold behavior against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: hold behavior is a ranking-contract concern, not evidence of an observed production result.

**Outcome interpretation: provisional behavior.** Expose the candidate to humans with uncertainty while the automation fence remains closed.
Invariant: Provisional output cannot trigger ranking-dependent action.
Diagnostic: inspect consumer fence decision before accepting the snapshot.
Decision consequence: provisional behavior changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise provisional behavior beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for provisional behavior with the window identifier and configuration hash.
Recovery: when provisional behavior is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate provisional behavior against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: provisional behavior is a ranking-contract concern, not evidence of an observed production result.

**Outcome interpretation: replay escalation.** Use the exact oracle only as an asynchronous adjudication path after a disputed close.
Invariant: Replay does not retroactively alter the original record.
Diagnostic: inspect replay request and result linkage before accepting the snapshot.
Decision consequence: replay escalation changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise replay escalation beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for replay escalation with the window identifier and configuration hash.
Recovery: when replay escalation is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate replay escalation against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: replay escalation is a ranking-contract concern, not evidence of an observed production result.

**Outcome interpretation: on-call handoff.** Page after two holds, a missing shard, or idempotency conflict with snapshot evidence attached.
Invariant: Pages name a mechanism, not a vague accuracy alarm.
Diagnostic: inspect page reason and evidence reference before accepting the snapshot.
Decision consequence: on-call handoff changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise on-call handoff beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for on-call handoff with the window identifier and configuration hash.
Recovery: when on-call handoff is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate on-call handoff against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: on-call handoff is a ranking-contract concern, not evidence of an observed production result.

**Outcome interpretation: runbook ownership.** Assign the platform lead authority for contract changes and on-call authority for recovery execution.
Invariant: No operator invents an error tolerance during an incident.
Diagnostic: inspect owner and escalation timestamp before accepting the snapshot.
Decision consequence: runbook ownership changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise runbook ownership beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for runbook ownership with the window identifier and configuration hash.
Recovery: when runbook ownership is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate runbook ownership against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: runbook ownership is a ranking-contract concern, not evidence of an observed production result.

**Outcome interpretation: retrospective record.** Classify each failure as data, state, partition, comparator, or capacity before remediation.
Invariant: Corrective work targets a discriminated cause.
Diagnostic: inspect incident classification before accepting the snapshot.
Decision consequence: retrospective record changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise retrospective record beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for retrospective record with the window identifier and configuration hash.
Recovery: when retrospective record is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate retrospective record against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: retrospective record is a ranking-contract concern, not evidence of an observed production result.

## 12. Failure Modes & What Went Wrong

The following are considered mechanisms and diagnostic controls, not reports of failures in a real system.

**Failure mechanism: duplicate delivery.** At-least-once transport repeats a valid event after a retry.
Invariant: The duplicate-id rate rises while schema rejects remain flat.
Diagnostic: inspect dedup before increment and compare batch identifiers. before accepting the snapshot.
Decision consequence: duplicate delivery changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise duplicate delivery beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for duplicate delivery with the window identifier and configuration hash.
Recovery: when duplicate delivery is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate duplicate delivery against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: duplicate delivery is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: late arrival.** An event arrives after the watermark although its event time belongs to the closed window.
Invariant: Lateness histogram has a tail beyond allowance.
Diagnostic: inspect write a correction version; do not mutate the old snapshot. before accepting the snapshot.
Decision consequence: late arrival changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise late arrival beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for late arrival with the window identifier and configuration hash.
Recovery: when late arrival is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate late arrival against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: late arrival is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: watermark regression.** A source emits an earlier watermark after recovery.
Invariant: The monotonicity assertion fails before window assignment.
Diagnostic: inspect reject the regression and investigate source epoch. before accepting the snapshot.
Decision consequence: watermark regression changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise watermark regression beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for watermark regression with the window identifier and configuration hash.
Recovery: when watermark regression is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate watermark regression against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: watermark regression is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: missing shard.** One shard fails close or its state is unavailable.
Invariant: Observed shard ids differ from configured ids.
Diagnostic: inspect hold; never merge eleven shards as if twelve contributed. before accepting the snapshot.
Decision consequence: missing shard changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise missing shard beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for missing shard with the window identifier and configuration hash.
Recovery: when missing shard is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate missing shard against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: missing shard is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: split ownership.** Two workers process the same partition during a lease race.
Invariant: Two writer identities report overlapping sequence ranges.
Diagnostic: inspect fence writers with epoch tokens and replay safely. before accepting the snapshot.
Decision consequence: split ownership changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise split ownership beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for split ownership with the window identifier and configuration hash.
Recovery: when split ownership is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate split ownership against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: split ownership is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: hash drift.** A changed hash library or seed routes a key to another shard after restart.
Invariant: Fixed test vectors disagree with current routing.
Diagnostic: inspect freeze publication for straddling windows and start a new epoch. before accepting the snapshot.
Decision consequence: hash drift changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise hash drift beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for hash drift with the window identifier and configuration hash.
Recovery: when hash drift is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate hash drift against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: hash drift is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: comparator drift.** One component uses lexical identifiers while another uses unsigned numeric order.
Invariant: Replay rows match counts but serialized order differs.
Diagnostic: inspect pin a shared comparator specification and digest. before accepting the snapshot.
Decision consequence: comparator drift changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise comparator drift beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for comparator drift with the window identifier and configuration hash.
Recovery: when comparator drift is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate comparator drift against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: comparator drift is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: heap eviction error.** The heap admits an item using the wrong less-than direction.
Invariant: The root improves when a worse item arrives.
Diagnostic: inspect assert heap invariant after each mutation in test builds. before accepting the snapshot.
Decision consequence: heap eviction error changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise heap eviction error beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for heap eviction error with the window identifier and configuration hash.
Recovery: when heap eviction error is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate heap eviction error against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: heap eviction error is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: candidate truncation.** A shard retains too few candidates for a global winner to reach merge.
Invariant: Exact oracle finds a winner absent from every shard list.
Diagnostic: inspect raise local retention or use a proven candidate bound. before accepting the snapshot.
Decision consequence: candidate truncation changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise candidate truncation beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for candidate truncation with the window identifier and configuration hash.
Recovery: when candidate truncation is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate candidate truncation against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: candidate truncation is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: sketch collision.** Colliding keys inflate an estimate enough to cross a rank boundary.
Invariant: Estimate interval overlaps its neighbor despite point-order.
Diagnostic: inspect hold or provisional when intervals overlap. before accepting the snapshot.
Decision consequence: sketch collision changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise sketch collision beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for sketch collision with the window identifier and configuration hash.
Recovery: when sketch collision is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate sketch collision against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: sketch collision is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: counter overflow.** A narrow counter wraps during an unexpectedly long accumulation.
Invariant: Count becomes negative or decreases without expiry.
Diagnostic: inspect use checked 64-bit arithmetic and reset by epoch. before accepting the snapshot.
Decision consequence: counter overflow changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise counter overflow beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for counter overflow with the window identifier and configuration hash.
Recovery: when counter overflow is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate counter overflow against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: counter overflow is a ranking-contract concern, not evidence of an observed production result.

**Failure mechanism: dedup expiry.** Identifier state is evicted before a delayed retry arrives.
Invariant: Duplicate spikes correlate with dedup eviction age.
Diagnostic: inspect retain identifiers through the lateness and retry horizon. before accepting the snapshot.
Decision consequence: dedup expiry changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise dedup expiry beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for dedup expiry with the window identifier and configuration hash.
Recovery: when dedup expiry is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate dedup expiry against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: dedup expiry is a ranking-contract concern, not evidence of an observed production result.
## 13. Transferable Lesson

**Transferable lesson: rank gaps govern feasibility.** A fixed memory structure is safe only relative to the depth-specific separation it must resolve.
Invariant: Error bounds must be compared to a decision boundary.
Diagnostic: inspect rank-gap calculation before accepting the snapshot.
Decision consequence: rank gaps govern feasibility changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise rank gaps govern feasibility beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for rank gaps govern feasibility with the window identifier and configuration hash.
Recovery: when rank gaps govern feasibility is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate rank gaps govern feasibility against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: rank gaps govern feasibility is a ranking-contract concern, not evidence of an observed production result.

**Transferable lesson: enumeration is separate.** A frequency estimator does not necessarily identify which keys deserve estimation.
Invariant: Candidate generation has its own correctness contract.
Diagnostic: inspect candidate recall evidence before accepting the snapshot.
Decision consequence: enumeration is separate changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise enumeration is separate beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for enumeration is separate with the window identifier and configuration hash.
Recovery: when enumeration is separate is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate enumeration is separate against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: enumeration is separate is a ranking-contract concern, not evidence of an observed production result.

**Transferable lesson: idempotency is semantic.** Retry safety requires a stable decision identity, not merely a network retry counter.
Invariant: A snapshot is an authority record.
Diagnostic: inspect window-plus-config key before accepting the snapshot.
Decision consequence: idempotency is semantic changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise idempotency is semantic beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for idempotency is semantic with the window identifier and configuration hash.
Recovery: when idempotency is semantic is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate idempotency is semantic against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: idempotency is semantic is a ranking-contract concern, not evidence of an observed production result.

**Transferable lesson: time has two axes.** Event time chooses membership while processing time chooses operational latency.
Invariant: Never use one timestamp as both.
Diagnostic: inspect dual timestamp fields before accepting the snapshot.
Decision consequence: time has two axes changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise time has two axes beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for time has two axes with the window identifier and configuration hash.
Recovery: when time has two axes is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate time has two axes against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: time has two axes is a ranking-contract concern, not evidence of an observed production result.

**Transferable lesson: capacity is compositional.** Heaps, deduplication, metadata, checkpoints, and serialization compete with counters for memory.
Invariant: Budget the total resident state.
Diagnostic: inspect component high-water table before accepting the snapshot.
Decision consequence: capacity is compositional changes the publication action rather than merely a dashboard annotation.
Boundary: do not generalise capacity is compositional beyond the declared event-time window and configured comparator.
Audit record: retain the relevant value for capacity is compositional with the window identifier and configuration hash.
Recovery: when capacity is compositional is violated, preserve the previous accepted snapshot and emit an explicit reason code.
Cost note: evaluate capacity is compositional against the 64 MiB state ceiling and the 2,000 ms close deadline.
Transfer: capacity is compositional is a ranking-contract concern, not evidence of an observed production result.
## 14. Connections & Cross-References

This case connects bounded heaps to streaming state, deterministic comparators to reproducible systems, and error intervals to decision policy. It does not establish a cross-case production dependency.

### Interview pack

**Question 1: Why is a count-min sketch alone insufficient for top-K publication?**
Answer: It estimates supplied keys but does not enumerate candidate keys.
Reasoning: Separate candidate discovery from frequency estimation.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 1, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Claim that querying every catalogue key is free.
Scoring: 2 points for naming enumeration and error.
Answer support: question 1 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 2: Compute window event volume for 18,000 events/s over 300 s.**
Answer: 5,400,000 events because 18,000 times 300 equals 5,400,000.
Reasoning: State why this is not cardinality.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 2, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Answering 54,000,000 by adding a zero.
Scoring: 2 points for units and arithmetic.
Answer support: question 2 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 3: A rank-20 count is 1,842 and rank-21 count is 1,511; what is the gap?**
Answer: 331 events because subtraction uses the same closed window.
Reasoning: Explain why a 400-event combined uncertainty fails.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 3, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Comparing either count to zero instead.
Scoring: 2 points for subtraction and contract.
Answer support: question 3 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 4: A shard is absent at close. Should the merger publish eleven-shard output?**
Answer: No; it must hold because candidate completeness is unknown.
Reasoning: Describe the alert evidence.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 4, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Treating zero output as a valid shard result.
Scoring: 2 points for the hold decision.
Answer support: question 4 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 5: A duplicate retry has the same event_id but later processing time. What key controls counting?**
Answer: event_id within the event-time window controls the logical event.
Reasoning: Discuss dedup retention horizon.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 5, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Using processing time to make it distinct.
Scoring: 2 points for separating time axes.
Answer support: question 5 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 6: When may a provisional ranking be shown?**
Answer: When humans may inspect uncertainty but automated action remains fenced.
Reasoning: Name the required consumer signal.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 6, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Allowing the trending slot to consume it.
Scoring: 2 points for the automation boundary.
Answer support: question 6 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 7: Compare 12 shards times 80 candidates times 48 bytes.**
Answer: 46,080 bytes, or about 45 KiB payload-only.
Reasoning: List excluded overhead.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 7, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Calling it the complete 64 MiB budget.
Scoring: 2 points for multiplication and limitation.
Answer support: question 7 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 8: How does a comparator bug appear if counts are correct?**
Answer: Replays can contain the same keys and counts but different serialized order.
Reasoning: Propose a byte-digest test.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 8, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Checking only set overlap.
Scoring: 2 points for a discriminating test.
Answer support: question 8 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 9: What happens when combined uncertainty is 240 events and the boundary gap is 180 events?**
Answer: The boundary is unresolved, so hold or provisional applies.
Reasoning: State the action fence.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 9, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Publishing because the point estimates differ.
Scoring: 2 points for interval reasoning.
Answer support: question 9 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 10: Why version the shard count with the snapshot?**
Answer: A hash domain change changes key ownership and replay meaning.
Reasoning: Explain epoch separation.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 10, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Updating the count in place mid-window.
Scoring: 2 points for partition reasoning.
Answer support: question 10 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 11: Choose an audit key for a snapshot retry.**
Answer: Use window_id plus configuration hash and reject a conflicting payload.
Reasoning: Explain why a random request id fails.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 11, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Using only processing timestamp.
Scoring: 2 points for idempotency.
Answer support: question 11 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 12: A late event belongs to a closed window. What is the correct correction model?**
Answer: Emit a linked correction version rather than mutate the closed authority.
Reasoning: Discuss consumer visibility.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 12, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Silently incrementing old heap state.
Scoring: 2 points for immutability.
Answer support: question 12 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 13: Which metric catches a backlog of held snapshots?**
Answer: Consumer-visible snapshot age and consecutive hold count.
Reasoning: Set a page threshold.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 13, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Using top-K set agreement alone.
Scoring: 2 points for operational relevance.
Answer support: question 13 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

**Question 14: How does flatter traffic change the design?**
Answer: It shrinks adjacent rank gaps and can invalidate an approximate error budget.
Reasoning: Choose exact counting or a stricter hold policy.
Follow-up: state the changed constraint and its decision effect.
Acceptable alternative: for question 14, a method with an explicit candidate and uncertainty contract earns credit.
Red flag: Assuming skew is fixed because it was configured.
Scoring: 2 points for sensitivity.
Answer support: question 14 is grounded in the relevant invariant and ILLUSTRATIVE derivation in Sections 6–12.

## 15. Quick Reference Card

| Field | Decision-ready statement |
|---|---|
| Decision grain | One closed 300-second event-time snapshot. |
| Owner | Streaming platform lead. |
| Output | Ordered top-20 keys, counts, bounds, provenance, and action. |
| Comparator | Count descending; unsigned key ascending. |
| Publish gate | Complete shards, valid state, resolved rank-20 boundary, idempotent write. |
| Hold gate | Missing shard, unresolved uncertainty, state conflict, or contract breach. |
| Provisional gate | Human display allowed while automation fence stays closed. |
| Evidence | ILLUSTRATIVE inputs; not-measured operational outcome. |
| Review trigger | Changes to cardinality, lateness, K, partitioning, or error tolerance. |
| Recovery | Preserve prior accepted snapshot and record reason code. |

