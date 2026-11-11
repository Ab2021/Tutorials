---
case_id: "CASE-03-002"
case_title: "Implement Top-K Stream Processing"
case_version: "3.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "streaming-systems"
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



**Operational evidence boundary.** The state described here is a proposed design, not a claim that any named platform has run it.

**Decision inputs retained at close.** The closer receives a watermark, a membership policy version, a shard manifest, and each shard's signed candidate payload.

**Decision inputs deliberately absent.** The closer does not query future events, mutable catalogue labels, or a later configuration revision.

**Reason-code vocabulary.** `READY` means all gates passed; `GAP_UNRESOLVED` means intervals overlap; `SHARD_MISSING` means no complete manifest exists.

**Audit question one.** Can an auditor recompute the candidate order from the retained candidate payloads and comparator version?

**Audit question two.** Can an auditor identify whether a late event was excluded, corrected, or included before the immutable close?

**Audit question three.** Can an auditor show that a duplicate delivery did not create a second increment?

**Selection ledger conclusion.** Exact maps are the reference when cardinality fits; Space-Saving is the candidate generator when bounded state is necessary; Count-Min is a companion estimator rather than a standalone top-K extractor.

**Method boundary.** A sketch bound certifies an estimate under its stated algorithmic assumptions; it does not certify that business consumers should act when rank gaps are smaller than the uncertainty.

**Reproducibility boundary.** Stable hashes and integer comparators make replay deterministic only if the same accepted-event set is reconstructed.

**Security boundary.** Snapshot payloads need integrity protection in a real deployment; this illustrative case does not specify an authentication protocol.

**Closing takeaway for Section 5.** The data contract, selection ledger, and retained evidence define what may be asserted at a publication tick.
## 6. Analysis Walkthrough

All numerical inputs in this section are ILLUSTRATIVE teaching inputs. Each result is a design check, not an observed capacity or accuracy result.

**Notation before calculation.** Let `N` be accepted events per 300-second window in events.

Let `lambda` be mean input rate in events per second.

Let `K` be publication depth in keys, with `K=20` for this case.

Let `S` be the number of key-owned shards, with `S=12`.

Let `epsilon` be a dimensionless additive-error fraction for a Count-Min estimate.

Let `delta` be the probability bound parameter used when choosing Count-Min depth.

Let `w` be sketch width in counters per row, and `d` be sketch depth in rows.

Let `L` be allowed lateness in seconds, and let `T_p` be processing time.

Let `T_e` be event time, the timestamp used for window membership.

The term “estimate” below always means an upper-biased Count-Min estimate unless a lower bound is explicitly named.

The term “candidate” means a key retained by a candidate generator, not a key known to be globally top-K.

```text
FORMULA [TK-01]: N = lambda * W
WHERE: N [events/window]; lambda [events/s]; W [s/window].
CONSTRAINT: ILLUSTRATIVE constant mean rate; accepted events only; no sampling.
NUMERICAL EXAMPLE: lambda=9,600 events/s; W=300 s;
                   N=9,600*300=2,880,000 events/window.
INTERPRETATION: volume sets update work and sketch error scale, not the number of distinct keys.
```

Dimension check: events/s multiplied by seconds produces events.

Independent arithmetic check: five minutes contains `5*60=300` seconds, and `9,600*300=2,880,000`.

Limit case: when `lambda=0`, the formula yields zero events and there is no meaningful rank boundary to certify.

```text
FORMULA [TK-02]: w = ceil(e / epsilon); d = ceil(ln(1/delta));
                  B = w * d * 8
WHERE: w [counters/row]; d [rows]; B [bytes]; e=2.7182818;
       epsilon dimensionless; delta dimensionless; each counter is 8 bytes.
CONSTRAINT: ILLUSTRATIVE Count-Min sizing with independent hash rows;
            it bounds point-query overestimate, not candidate enumeration.
NUMERICAL EXAMPLE: epsilon=0.00004; delta=0.01;
                   w=ceil(2.7182818/0.00004)=ceil(67,957.045)=67,958;
                   d=ceil(ln(100))=ceil(4.60517)=5;
                   B=67,958*5*8=2,718,320 bytes=2.592 MiB.
INTERPRETATION: the counter array fits well inside 64 MiB, but a separate candidate generator remains required.
```

Dimension check: counters times rows times bytes per counter produces bytes.

Independent conversion: `2,718,320 / 1,048,576 = 2.592` MiB to three decimals.

Boundary: correlated hash rows or adversarial keys can invalidate the claimed probability behavior; a hash-family review is then required.

```text
FORMULA [TK-03]: E = epsilon * N; certify if 2E < G
WHERE: E [events] is one-key additive uncertainty; G [events] is
       f(K)-f(K+1), the exact boundary gap for the target window.
CONSTRAINT: ILLUSTRATIVE conservative two-estimate comparison; both point
            estimates may overestimate by at most E; candidate coverage holds.
NUMERICAL EXAMPLE: epsilon=0.00004; N=2,880,000;
                   E=0.00004*2,880,000=115.2 events;
                   G=331 events; 2E=230.4 events < 331 events.
INTERPRETATION: the constructed rank-20 boundary has 100.6 events of conservative separation margin.
```

Independent check: `331-230.4=100.6` events.

Sensitivity: if `G=180`, then `230.4>180`; the design must hold or mark provisional rather than assert rank 20.

Counterexample: this comparison cannot certify a list if the true 21st key never reaches the candidate generator.

```text
FORMULA [TK-04]: C = S * H; M_c = C * b
WHERE: C [candidate records]; S [shards]; H [candidates/shard];
       b [bytes/candidate record]; M_c [bytes].
CONSTRAINT: ILLUSTRATIVE fixed-width serialized records; heap and network headers excluded.
NUMERICAL EXAMPLE: S=12; H=80; b=48 bytes;
                   C=12*80=960 records;
                   M_c=960*48=46,080 bytes=45 KiB.
INTERPRETATION: candidate transport is small, so retaining more candidates is usually cheaper than reopening a window.
```

Independent conversion: `46,080 / 1,024 = 45` KiB exactly.

Limit case: if keys are split across shards by event rather than key, `H=K` can miss a globally frequent key whose partial counts never dominate a shard.

```text
FORMULA [TK-05]: R = W + L; B_bucket = R * S * q
WHERE: R [s] is mutable retention horizon; W [s] window duration;
       L [s] allowed lateness; S [shards]; q [bytes/shard-second bucket].
CONSTRAINT: one-second event-time buckets; no overlapping-window duplication;
            bucket payload q includes counters already selected for retention.
NUMERICAL EXAMPLE: W=300 s; L=90 s; S=12; q=6,144 bytes;
                   R=300+90=390 s;
                   B_bucket=390*12*6,144=28,753,920 bytes=27.421875 MiB.
INTERPRETATION: retaining late-data correction buckets consumes material state even before sketch and heap overhead.
```

Independent check: `390*12=4,680` shard-seconds, then `4,680*6,144=28,753,920` bytes.

Boundary: a late-data policy with unbounded `L` has unbounded retained mutable history and cannot meet a fixed memory contract.

```text
FORMULA [TK-06]: M_total = B + B_bucket + M_c + M_meta
WHERE: M_total [bytes]; B sketch bytes; B_bucket bucket bytes;
       M_c candidate bytes; M_meta metadata and serialization reserve [bytes].
CONSTRAINT: components are resident simultaneously at close; no allocator overhead beyond M_meta.
NUMERICAL EXAMPLE: B=2,718,320; B_bucket=28,753,920;
                   M_c=46,080; M_meta=8,388,608;
                   M_total=39,906,928 bytes=38.058 MiB < 64 MiB.
INTERPRETATION: the illustrative plan leaves 25.942 MiB of headroom, not 64 MiB of free memory.
```

Independent check: the byte sum is `2,718,320+28,753,920+46,080+8,388,608=39,906,928`.

Sensitivity: doubling `L` from 90 to 180 seconds adds another `90*12*6,144=6,635,520` bytes, reducing headroom to 19.614 MiB.

**Worked event-time trace.** Consider window `W17=[10:00:00,10:05:00)` in event time.

At processing time 10:05:02, event `a` has event time 10:04:58 and is accepted because the watermark is 10:04:55.

At processing time 10:05:15, the watermark reaches 10:05:00, but the configured lateness interval keeps correction buckets mutable until 10:06:30.

At processing time 10:05:35, event `b` has event time 10:04:59 and is accepted into `W17` because it arrives within the 90-second allowance.

At processing time 10:06:31, event `c` has the same event time and is late beyond the stated allowance.

Event `c` is appended to a correction ledger with `window_id=W17`; it does not silently mutate the snapshot already published for W17.

If policy permits a correction, a new record references the original snapshot and has a distinct correction sequence.

If policy forbids correction, the event is recorded as excluded-late so that downstream users can distinguish omission from loss.

The close decision uses event-time membership, while close latency uses processing time; neither timestamp substitutes for the other.

**Worked partition trace.** Suppose stable hash assigns keys `p`, `q`, and `r` to shards 2, 7, and 2 respectively.

All occurrences of `p` in W17 must go to shard 2, including retry deliveries.

A retry of an already accepted producer sequence is recognized before the sketch update, so it creates zero additional count.

Shards emit candidate records keyed by `(window_id, shard_id, configuration_hash)`.

The closer rejects a manifest with 11 records when configuration expects 12 shards.

The closer merges only records bearing the same window and configuration hash; mixing versions is a failed precondition, not a tie.

The global comparator is descending estimated count, then ascending unsigned `term_id`.

This comparator is used for heap eviction, shard list sort, merge, and serialized snapshot order.

**Heap versus sketch selection.** An exact map plus a min-heap is correct when every distinct key fits in memory.

Its map retains count for every observed key, while its heap limits the output-selection cost to `O(log K)` per candidate update.

The heap does not reduce the map's cardinality cost because the map still needs each key's exact count.

A Space-Saving summary keeps a fixed number of counters and emits candidates directly, but its counts include replacement error.

A Count-Min sketch uses fixed counters and yields point estimates, but it cannot enumerate high-frequency keys without another structure.

A hybrid can use Space-Saving for candidates and Count-Min for a conservative query bound, provided their distinct error meanings remain visible.

The design in this case chooses a bounded candidate summary plus a point-estimate sketch only after proving both the candidate and rank-bound contracts.

An exact-map fallback is selected whenever the observed cardinality and memory estimate fit inside the 64 MiB ceiling with reserve.

**Candidate completeness rule.** A shard-local top-20 is not generally sufficient for a global top-20 when a key can be represented on multiple shards.

Key-based partitioning avoids that failure because each key has exactly one owner during the window.

Under key-based partitioning, retaining each shard's local top-K is sufficient for global top-K because a globally top-K key must be in its owning shard's top-K.

This sufficiency statement fails during re-sharding if the same logical key can exist under two owners.

The configuration hash therefore includes partition count, hash function identifier, and re-shard epoch.

**Idempotency rule.** Input idempotency and output idempotency solve different duplicate paths.

Input idempotency rejects a previously accepted source sequence before state mutation.

Output idempotency makes repeated snapshot commits of identical `(window_id, configuration_hash)` return the same authority record.

A conflicting payload under an existing output key is corruption evidence and must be rejected rather than “last write wins.”

A producer sequence is only safe for input idempotency when the producer identifier and sequence are stable across its own restarts.

If that premise cannot be met, the system needs an event identifier with a bounded retention method and must re-budget memory.

## 7. The Finding

**Proposed finding.** Publish an automated top-20 only when the window manifest is complete, partition ownership is stable, candidate coverage is satisfied, duplicate rejection is healthy, and the rank-20 margin exceeds the configured uncertainty bound.

The selected default is a bounded candidate summary plus a Count-Min point-estimation path, with an exact-map mode available when distinct-key memory fits.

This is a conditional design decision, not a statement that the configuration has been executed on a production workload.

For the illustrative 9,600 events/s workload, a 67,958-by-5 counter array consumes 2.592 MiB and provides a 115.2-event per-key error scale.

With a constructed 331-event boundary gap, the conservative two-estimate margin is positive by 100.6 events.

That arithmetic supports a publication gate; it does not prove the assumed gap will occur on future windows.

The 64 MiB budget is met in the stated accounting model at 38.058 MiB, leaving 25.942 MiB before unmodelled allocator behavior.

The operational conclusion is to measure actual high-water state and hold publication on a budget breach rather than infer safety from this paper budget.

**Decision table.**

| Gate | Evidence at close | Action when true | Action when false |
|---|---|---|---|
| Manifest | 12 matching shard attestations | Continue merge | Hold previous snapshot |
| Ownership | Single re-shard epoch and stable hash | Continue merge | Hold and drain boundary window |
| Dedup | Duplicate ratio at or below configured bound | Continue merge | Hold and investigate source replay |
| Candidate coverage | Candidate generator has its documented guarantee | Continue merge | Provisional only for humans |
| Rank margin | `2E < G` at K/K+1 | Permit automation | Provisional or hold |
| Memory | Resident high-water below 64 MiB | Retain until close | Shed mode or hold |
| Commit | Snapshot key absent or byte-identical | Publish once | Reject conflicting payload |

A provisional snapshot is useful only when its interface visibly carries `provisional=true`, its reason code, and the event-time window.

Automation reads only snapshots with `automation_eligible=true`; it must not infer eligibility from a nonempty list.

A held snapshot retains its original event-time end and publication timestamp, so users see staleness rather than a disguised new observation.

The finding deliberately declines to promise exact rank positions below the certified boundary.

If the business needs exact order through rank 500, the error budget must be re-derived at that depth or the system must use an exact approach.

## 8. System/Solution Design

**Input-to-output path.** Ingress validation → event-time assignment → source-sequence deduplication → stable key partitioning → per-shard summary update → watermark close barrier → manifest validation → candidate merge → rank and capacity gates → immutable snapshot commit → dashboard and automation consumers.

Each arrow represents a data dependency in the proposed design, not a claim about a deployed product.

**Ingress validation.** Require an event identifier or stable producer-sequence pair, event time, key, schema version, and tenant scope before counting.

Malformed keys are quarantined with a reason code because mapping them to a common placeholder would create an artificial heavy hitter.

Event time outside configured clock-skew bounds is recorded separately from late arrival because a bad clock and normal network delay require different correction.

The validator must not consult future catalogue state when normalizing a key, or replay determinism is lost.

**Window manager.** Store events in one-second event-time buckets for the stated 300-second trailing window.

Advance the watermark monotonically from observed source progress under a documented policy.

Keep buckets mutable through `L=90` seconds after their nominal end so an eligible late event can alter the not-yet-final result.

Seal a bucket after the lateness allowance, then retain only correction metadata required by policy.

A watermark regression is rejected and paged because reopening already-closed state makes previous snapshots ambiguous.

**Partitioner.** Compute `shard_id=stable_hash(term_id, hash_version) mod S` at ingestion.

Persist `hash_version`, `S`, and re-shard epoch with every shard manifest.

Do not change any of these values within a window; route a planned change through a clean boundary plus a drain interval.

Key ownership is chosen over round-robin event routing because global key counts then have a single source of truth.

Hot keys remain a load-skew risk under key ownership, so the design observes per-shard update rate and does not hide the trade-off.

**Per-shard state.** Maintain candidate-summary state, optional Count-Min counters, source progress, and a bounded diagnostic sample.

The candidate summary is responsible for enumeration; the point-estimate sketch is responsible for a stated bound on queried candidates.

The heap used to serialize candidates stores the same total comparator as the merge.

Store counters as unsigned 64-bit quantities or fail before overflow; silently wrapping reverses order.

Record state high-water bytes by component rather than reporting one process-resident number with unknown attribution.

**Close barrier.** At close, each shard writes an immutable candidate payload with count estimate, error metadata, total accepted events, watermark, and digest.

The manifest is complete only when it has exactly one payload for every expected `(window_id, shard_id, configuration_hash)`.

A duplicate payload with the same digest is a harmless retry; a duplicate with a different digest is a correctness incident.

The closer never infers a missing shard's zero contribution, because a missing shard can contain the winning key.

**Merge.** Deserialize each sorted shard list using the pinned comparator.

Push the first candidate from each list into a global heap ordered by count descending and key ascending.

Pop one candidate, emit it if it is not already emitted, then push the next record from that shard list.

Continue until K unique key records are emitted or the candidate lists are exhausted.

Uniqueness should be trivial under strict key ownership; retaining the check makes an accidental double-owner detectable rather than silent.

**Rank gate.** Compare a lower confidence quantity for rank K with an upper confidence quantity for rank K+1 under the actual summary's documented semantics.

Do not substitute the last emitted candidate's estimated count for a proof that no omitted key can exceed it.

Persist the two compared values, the bound version, and the decision threshold alongside the action.

A failed rank gate permits dashboard visibility only if the UI labels uncertainty and the automation fence stays closed.

**Snapshot commit.** Use `(window_id, configuration_hash, correction_sequence)` as the durable authority key.

The first accepted payload becomes immutable for that key.

A retry with byte-identical payload returns the same record identifier.

A retry with a different digest raises an integrity alert and cannot overwrite the authority record.

Store event-time bounds, processing-time commit time, shard manifest digest, candidate digest, comparator version, decision, and reason code.

**Consumer contract.** Dashboard clients display the window end, action, estimated counts, and provisional or held condition.

Automation clients require `automation_eligible=true` and reject any missing eligibility field.

Consumer cache TTL must not be treated as the event-time window; it only controls display freshness.

A downstream recommender that needs per-tenant ranks cannot reuse this global aggregate without a separate state budget and privacy review.

**Recovery path.** On process restart, reload the latest durable bucket checkpoint and source progress before accepting more input.

Replay only source sequences greater than the durable progress marker; this avoids reapplying acknowledged updates.

If source progress is unavailable, switch the affected window to hold rather than claim exactly-once behavior.

Rebuild a lost candidate summary from retained bucket data only when the recovery deadline still permits a validated close.

If reconstruction exceeds the 2,000 ms close deadline, publish the explicitly stale previous snapshot and emit a recovery reason code.

**Operational readiness.** Configuration changes require a dry-run manifest check, capacity recalculation, comparator compatibility review, and an explicit effective window.

An on-call runbook needs separate actions for missing shard, late-data surge, memory pressure, digest conflict, and rank-gate failure.

A dashboard label alone is not a runbook action; each alert must identify whether data can be replayed and whether automation is fenced.

The minimum recovery drill is a forced shard restart during an open window followed by digest comparison against an independent replay.

## 9. Evaluation & Validation

No production evaluation has been executed; the following is a proposed validation plan using constructed streams and an independent exact oracle.

**Primary correctness metric.** Exact top-K set recall has numerator “keys present in both the published K-set and oracle K-set.”

Its denominator is K oracle keys for the same event-time window and comparator version.

Its population is all completed constructed windows after the configured watermark, not only windows that passed publication gates.

Its window is one 300-second event-time interval evaluated at each 60-second publication cadence.

Its comparator is an exact per-key replay with independently implemented merge and comparator code.

The guardrail is zero false `automation_eligible` snapshots when the oracle shows an incorrect top-K membership.

The uncertainty statement is that finite generated scenarios cannot bound unrepresented production distributions.

Status: not executed.

**Order metric.** Boundary-order accuracy has numerator “windows where oracle rank 20 precedes oracle rank 21 in the same order emitted.”

Its denominator is windows whose oracle ranks 20 and 21 have unequal counts after deterministic tie-breaking.

Its population excludes empty windows because no twentieth rank exists there.

Its comparator uses event-time membership and the same key normalization as the candidate path, while the merge implementation remains independent.

The guardrail is that a tie resolved by key order is not counted as an estimator success; it is tested as determinism.

Status: not executed.

**Freshness metric.** On-time snapshot fraction has numerator “eligible snapshots committed within 2,000 ms of watermark close.”

Its denominator is all windows with complete input manifests during the test run.

Its population includes held and provisional decisions so latency is not flattered by excluding failures.

The comparator is the 2,000 ms service objective measured from close-barrier observation to durable commit acknowledgement.

The guardrail is no missing-manifest window may be counted as on-time by committing an empty list.

Status: not executed.

**Resource metric.** State headroom has numerator `64 MiB - observed component-attributed high-water bytes`.

Its denominator is one process/window state image, and its unit is MiB rather than a percentage that obscures absolute risk.

The population includes the highest-rate and highest-cardinality constructed scenarios separately because they stress different allocations.

The comparator is the 64 MiB hard budget stated in this case.

The guardrail is allocation accounting must include checkpoint serialization and dedup state, not merely sketch counters.

Status: not executed.

**Determinism test.** Feed an identical generated event multiset through two process instances with permuted arrival order allowed by the same watermark schedule.

Compare the serialized snapshot digest, candidate order, action, and reason code.

A difference caused only by processing-time metadata is excluded from the digest comparison and separately reported.

A difference in key order at equal counts is a comparator defect even when the set metric is perfect.

Status: proposed, not executed.

**Late-data test.** Generate an event at each offset from `-1` second through `L+1` seconds around the watermark allowance.

Verify that accepted late events affect the mutable window, while excluded late events are recorded in the correction path.

Verify that a correction record cannot overwrite the original authority key.

The comparator is a hand-calculated event-time ledger for the small trace.

Status: proposed, not executed.

**Failure injection test.** Omit one shard payload at close and assert that no automation-eligible snapshot is committed.

Inject a byte-different duplicate payload for the same manifest identity and assert that commit fails closed.

Restart a shard after its checkpoint but before close and compare recovery output to the exact oracle.

Drive one shard above its memory allocation and assert that the capacity reason code takes precedence over a rank-gate pass.

Status: proposed, not executed.

**Acceptance threshold.** The design is acceptable for pilot implementation only if every false automation-eligible result is zero across the declared test matrix, digest determinism holds, and observed state stays within budget with stated reserve.

This is a pre-deployment criterion, not a claimed result.
## 10. Action Taken

**Proposed action, not completed deployment.** Implement the design in an isolated test harness before connecting it to dashboard or automation consumers.

First, freeze the ranking contract: window interval, allowed lateness, key normalization, deterministic comparator, publication actions, and authority key.

Second, implement the exact oracle independently from the bounded-state pipeline.

Third, generate constructed workloads covering concentrated keys, nearly uniform keys, ties, duplicates, late events, shard loss, and recovery.

Fourth, make the candidate generator and point-estimator selectable by configuration with versioned semantics rather than silently changing algorithm under the same snapshot identity.

Fifth, implement the close barrier and manifest validation before optimizing the merge, because a fast merge of incomplete state is still wrong.

Sixth, attach component-level memory accounting to every candidate design and reject a configuration that budgets only its sketch array.

Seventh, create the automation fence so a provisional or held snapshot cannot be consumed as a normal ranking.

Eighth, run the proposed validation matrix and record results as measured only after execution.

Ninth, require a change review for `K`, window length, lateness, shard count, key hash, candidate capacity, counter width, or rank-margin policy.

The action does not include automatic correction of late data because correction semantics require consumer agreement about whether historical ranks may change.

The action does not include per-tenant rankings because that changes cardinality, privacy, and capacity assumptions rather than being a simple filter.

**Implementation sequence and exit evidence.**

| Step | Deliverable | Exit evidence | Decision if it fails |
|---|---|---|---|
| 1 | Comparator library | Equal-count key test vectors serialize identically | Stop before state implementation |
| 2 | Exact oracle | Small hand ledger matches replay output | Repair oracle independently |
| 3 | Event-time buckets | Boundary events follow half-open interval | Fix membership logic |
| 4 | Dedup gate | Replayed source sequence adds zero count | Hold input acceptance |
| 5 | Key partitioner | Every test key has one owner per epoch | Reject re-shard plan |
| 6 | Candidate summary | Candidate guarantee documented and tested | Use exact-map mode |
| 7 | Close barrier | Missing shard blocks eligible commit | Fix manifest handling |
| 8 | Rank gate | Overlapping interval blocks automation | Tune only through reviewed config |
| 9 | Snapshot store | Byte-identical retry is idempotent | Fix authority-key semantics |
| 10 | Recovery drill | Restart digest equals oracle digest | Block pilot |

**Rollout constraints.** A shadow run may compute snapshots but cannot alter the trending slot.

A pilot may expose an automation-eligible flag only after the exact oracle has completed for the corresponding constructed workload.

A consumer migration must use the explicit eligibility field and cannot use count magnitude as a proxy for confidence.

A rollback is configuration-driven: fence automation, retain prior published snapshot, preserve the failing manifest and checkpoint, then replay offline.

A rollback does not delete the failed record because the record is needed to explain the decision and improve the test matrix.

## 11. Outcome & Results

**Outcome status: not measured.** No implementation, pilot, production traffic, latency measurement, memory measurement, accuracy result, or business impact is claimed in this case.

The completed outcome of this manuscript is a proposed decision contract and a testable design specification.

The design predicts that bounded summary state can meet the illustrative 64 MiB budget while preserving a visible rank-confidence gate.

That prediction depends on the constructed traffic rate, cardinality behavior, late-data retention model, and hash assumptions.

The design does not predict that a real workload will have a 331-event rank-20 gap.

The design does not predict that all production duplicate paths have stable source sequences.

The design does not predict that a Count-Min parameterization alone can enumerate the correct keys.

**Result record to populate only after execution.**

| Result field | Required future observation | Current value |
|---|---|---|
| Top-20 membership errors | Oracle mismatch count and window identifiers | not-measured |
| Incorrect eligible publishes | Count, severity, and cause | not-measured |
| Commit freshness | Close-to-commit latency distribution | not-measured |
| State high-water | Per-component byte distribution | not-measured |
| Late-event handling | Accepted, corrected, and excluded counts | not-measured |
| Duplicate handling | Duplicate attempts and rejected increments | not-measured |
| Recovery fidelity | Digest comparison after forced restart | not-measured |
| Consumer behavior | Eligibility-fence acceptance evidence | not-measured |

**Interpretation discipline.** A passing constructed test proves only the tested invariant under generated inputs.

A successful shadow run would establish observability and replay behavior for that environment, not business benefit.

A production measurement would need an explicit observation window, population, comparator, missing-data treatment, and uncertainty statement.

A dashboard click-through change would not establish ranking correctness because user behavior has additional causes.

A lower memory reading would not establish safety unless it includes the close-time peak and recovery overlap.

A zero duplicate count would not establish idempotency unless duplicate injection was attempted.

The honest next conclusion is therefore “ready for specified validation,” not “successful stream-processing implementation.”

## 12. Failure Modes & What Went Wrong

The mechanisms below are considered failure scenarios, not observed incidents.

### F-01: Event-time and processing-time were conflated

**Symptom.** A snapshot includes an event that arrived before close but belongs to the next event-time window.

**Competing causes.** The window assignment may use processing time, the event clock may be wrong, or the watermark may be advanced prematurely.

**Discriminating check.** Inspect the event timestamp, processing timestamp, assigned `window_id`, watermark history, and configured interval for the same event.

**Correction.** Recompute membership from immutable event time and use the correction path if the prior snapshot was already committed.

**Prevention.** Keep event-time bounds and processing-time commit time as distinct serialized fields.

**Verification.** Boundary tests at exactly start, exactly end, one millisecond before end, and one millisecond after end match the half-open contract.

### F-02: Late data reopened a published window invisibly

**Symptom.** Two consumers retrieve different counts for what appears to be the same window identifier.

**Competing causes.** Mutable state was reused after commit, a correction overwrote the authority record, or cache invalidation served mixed versions.

**Discriminating check.** Compare snapshot keys, correction sequence, payload digests, and commit chronology.

**Correction.** Restore the original immutable snapshot and emit an explicit correction record with a new sequence.

**Prevention.** Enforce append-only authority records and prevent consumers from selecting “latest” without correction semantics.

**Verification.** A late-event injection produces either no mutation or a separately addressable correction, never an overwritten original.

### F-03: A retry inflated one key's count

**Symptom.** One term rises sharply after collector recovery while total traffic does not show a commensurate increase.

**Competing causes.** Source sequence was reset, the dedup horizon expired, or two writers owned the same partition.

**Discriminating check.** Compare producer sequence ranges, dedup rejects, shard lease history, and event identifiers around recovery.

**Correction.** Fence the duplicate writer, restore source progress, and replay the affected interval from an authoritative log if available.

**Prevention.** Make source progress durable before acknowledging the batch and test producer restart semantics.

**Verification.** Replaying an acknowledged batch changes neither per-key counts nor snapshot digest.

### F-04: Candidate enumeration missed a true heavy hitter

**Symptom.** A Count-Min query estimates a key as large when queried offline, but the key never appears in the published candidate list.

**Competing causes.** Count-Min was used without enumeration, candidate capacity was too small, or ownership split the key across shards.

**Discriminating check.** Compare oracle heavy hitters to per-shard retained candidates and inspect ownership by re-shard epoch.

**Correction.** Use a candidate-producing summary, increase retained capacity based on a documented bound, or switch to exact maps.

**Prevention.** Treat candidate coverage as a separate gate from point-query error.

**Verification.** Oracle top-K membership recall remains one across the declared adversarial candidate test set.

### F-05: A hash or shard change split key ownership

**Symptom.** The same normalized key occurs in two shard payloads for one window.

**Competing causes.** Hash-version rollout was non-atomic, shard count changed mid-window, or normalizers differed by process.

**Discriminating check.** Inspect configuration hashes, hash test vectors, shard manifests, and normalization versions.

**Correction.** Hold the straddling window, drain old ownership, and begin a new window under one explicit epoch.

**Prevention.** Apply partition changes only at a declared clean boundary with compatibility tests.

**Verification.** Every key in a replay maps to exactly one owner for each accepted window.

### F-06: Equal counts emitted in nondeterministic order

**Symptom.** Replays have the same K-set and counts but different serialized ordering.

**Competing causes.** Heap comparator omits key order, map iteration leaks into output, or signed and unsigned key comparisons differ.

**Discriminating check.** Feed equal counts with high-bit key identifiers through every sort and merge boundary.

**Correction.** Adopt one count-descending, unsigned-key-ascending comparator and version it.

**Prevention.** Test byte digests, not just unordered set equality.

**Verification.** Two independent process runs emit identical ordered bytes for the same retained state.

### F-07: Sketch error was interpreted as an exact confidence interval

**Symptom.** The system publishes a rank because estimated counts differ, even though their possible errors overlap.

**Competing causes.** The gate ignores error, one-sided overestimate was treated as symmetric, or the bound was sized for a different N.

**Discriminating check.** Recompute E from actual accepted event total and compare documented upper and lower quantities at K and K+1.

**Correction.** Apply the correct estimator-specific gate and mark the list provisional when the order remains unresolved.

**Prevention.** Persist algorithm type, parameters, and bound derivation with every snapshot.

**Verification.** Synthetic near-tie cases never set `automation_eligible=true` when the conservative intervals overlap.

### F-08: Memory budget excluded recovery and metadata

**Symptom.** Steady-state counters fit, but close or restart causes resident memory to exceed 64 MiB.

**Competing causes.** Serialization buffers were omitted, dedup state grew, allocator overhead rose, or late buckets overlapped during recovery.

**Discriminating check.** Attribute high-water bytes by counter array, buckets, heap, manifests, dedup, serialization, and recovery copies.

**Correction.** Reduce retention or window size, increase capacity with approval, or choose a more compact representation.

**Prevention.** Budget peak concurrent components rather than a steady-state algorithm object.

**Verification.** Fault-injected restart and close tests stay below the ceiling with a recorded reserve.

### F-09: A missing shard was treated as zero traffic

**Symptom.** A list publishes on time but lacks terms that the oracle finds in the absent shard's range.

**Competing causes.** Timeout fallback inserted zeros, manifest cardinality was not checked, or a stale payload was accepted under a new configuration.

**Discriminating check.** Validate expected shard identifiers, window identifier, and configuration hash before merge.

**Correction.** Hold the snapshot and preserve available payloads for replay; do not invent missing counts.

**Prevention.** Make complete-manifest validation a hard prerequisite for eligibility.

**Verification.** Deliberately removing one payload produces a hold reason and no eligible commit.

### F-10: Output retries created conflicting authorities

**Symptom.** Two snapshot records share a window label but contain different candidate digests.

**Competing causes.** The authority key omitted configuration, retry writer recomputed with changed state, or storage allowed upsert overwrite.

**Discriminating check.** Query by `(window_id, configuration_hash, correction_sequence)` and compare stored payload digests.

**Correction.** Reject the second payload, fence its writer, and investigate the state divergence before any correction.

**Prevention.** Use create-if-absent semantics with digest comparison on retry.

**Verification.** A duplicate commit is idempotent only when serialized bytes match exactly.

## 13. Transferable Lesson

Top-K streaming is not solved by naming a heap or a sketch; it is solved by stating which state is complete enough to authorize an action.

The first transferable lesson is to separate enumeration from estimation.

A structure that estimates a named key may still be unable to discover the names worth estimating.

The second lesson is to compare approximation error with the business boundary, not with a visually impressive global error percentage.

A small error can still reverse an actionable tie, while a larger error can be harmless far above the cutoff.

The third lesson is to partition by the invariant needed for correctness.

Key ownership makes local candidate sufficiency easier to prove; event routing may spread load but introduces aggregation obligations.

The fourth lesson is to keep event time, processing time, and correction version distinct.

Temporal ambiguity becomes a data-contract failure long before it becomes a performance problem.

The fifth lesson is that idempotency has an input side and an output side.

Rejecting duplicate input does not prevent duplicate authorities, and idempotent output does not undo double-counted state.

The sixth lesson is to design the hold path as carefully as the success path.

A hold with visible staleness and retained evidence is safer than an apparently fresh unsupported ranking.

The seventh lesson is to reserve capacity for recovery, serialization, and late-data retention, not only the nominal algorithm.

The eighth lesson is to treat deterministic tie ordering as correctness rather than presentation.

Stable output makes auditing, caching, retry comparison, and interview reasoning tractable.

These lessons transfer to rate limiting, fraud counters, alert ranking, online feature aggregates, and any windowed decision with a small published frontier.

**Practical transfer checks.**

Before adopting a top-K summary, name the action that changes when rank K and rank K+1 exchange places.

If no action changes, a less frequent batch report may be simpler and more honest.

If the action changes at rank one, evaluate a stronger margin near rank one rather than averaging errors across the list.

If different consumers require different K values, compute the deepest required boundary rather than assuming a shallow proof transfers downward.

A thresholded alert can reuse the same gate logic by comparing a lower count bound to the action threshold.

A quota system can reuse source-progress idempotency but must also define whether retries consume quota.

A fraud leaderboard can reuse deterministic order but needs a stronger correction policy because an appeal may depend on historical rank.

A monitoring leaderboard can tolerate provisional display more readily when automation is not coupled to the rank.

A recommender candidate list may value recall above rank order, so its validation metric should not copy this case's exact-boundary gate blindly.

A billing use case normally requires exactness or a separately approved reconciliation path because approximate overcount has financial meaning.

For any consumer, encode the selected action in the snapshot rather than asking consumers to reconstruct it from raw metrics.

A reason code must be stable enough for alert grouping; free-text explanations are useful supplements, not a machine contract.

Window identifiers should include unambiguous event-time bounds, not only a sequence number that needs external context.

Configuration identity should cover every setting capable of changing emitted bytes or the correctness proof.

Algorithm migration should write a new configuration identity even if the list happens to be identical for one window.

A comparator change is a semantic migration because it can change which tied key receives a scarce position.

Capacity planning should distinguish per-shard state from coordinator state because increasing shard count moves pressure between them.

Increasing shards can lower update load while increasing manifest size, candidate transport, and coordination failure surface.

Increasing K increases heap and candidate output cost, but its dominant effect may instead be the smaller gap deeper in the distribution.

Increasing allowed lateness improves event-time completeness only if the source watermark itself remains trustworthy.

Increasing a sketch width lowers its error scale but cannot repair omitted candidates or incorrect membership.

Increasing sketch depth changes a confidence parameter and consumes memory, but does not shrink an additive bound determined by width.

An exact fallback should have an explicit trigger based on observed cardinality and a measured map-overhead model.

A fallback that activates after memory exhaustion is a failure path, not a capacity strategy.

The oracle should avoid sharing the production parser whenever possible, otherwise the same normalization defect may agree with itself.

An independent oracle still needs the same written business contract for membership and ties; independence does not mean ambiguity.

Generated workloads should include rank ties because production-like skew generators can accidentally hide comparator errors.

Generated workloads should include collision pressure because benign random keys do not test a hash selection boundary.

Generated workloads should include source restart because simple duplicate records do not test lost-progress behavior.

A chaos test that kills a shard without checking consumer fencing proves availability behavior, not correctness behavior.

A retained manifest lets investigators distinguish computation failure from input incompleteness after the fact.

A retained candidate list is insufficient for forensic replay if it lacks the summary parameters that gave those candidates meaning.

A retained configuration hash is insufficient if its underlying configuration document is not archived or resolvable.

A dashboard should show estimated counts as estimates; hiding approximation behind integer formatting turns a technical caveat into misinformation.

A dashboard should show staleness in event-time terms because processing-time freshness can hide an old underlying window.

An automation client should reject unknown decision statuses so a future status cannot become an accidental publish.

A correction client should key cache entries by correction sequence as well as window identifier when revisions are supported.

A stalled watermark can produce apparently low traffic because events remain buffered; observe watermark age separately from input rate.

A fast watermark can produce superficially fresh snapshots by excluding slow sources; observe source participation separately from close latency.

A memory alarm should carry the component attribution, because counter-array growth and late-bucket growth imply different remedies.

A rank-margin alarm should carry the competing keys and bounds, because a generic “low confidence” alert is not actionable.

A digest-conflict alarm should preserve both payloads under restricted access for comparison rather than logging only a hash mismatch.

The safest degraded state is usually stale-but-labelled data, not a newly computed incomplete ranking.

A stale snapshot policy needs a maximum age, after which absence is safer than continued display.

A hold decision is an output with consumers, observability, and service-level expectations; it is not an exception to product design.

The final transfer test is whether an engineer can explain, for a single snapshot, why every included event counted once and why no excluded uncertainty drove automation.

## 14. Connections & Cross-References

**Related method: exact top-K with a map and min-heap.** Use it when observed distinct cardinality plus overhead fits the state budget and exactness is required.

**Related method: Space-Saving heavy hitters.** Use it when candidate enumeration must remain bounded and replacement error can be carried into publication logic.

**Related method: Count-Min sketch.** Use it for point-frequency estimates with bounded one-sided error, never as the only answer to “which keys are top?”

**Related method: reservoir sampling.** It can estimate distributional properties but is not a direct replacement for exact heavy-hitter membership guarantees.

**Connection to rate limiting.** Both rate limiting and top-K need a time-bucket contract, bounded state, and a safe response when state evidence is incomplete.

**Open question: adversarial keys.** A public or user-controlled key space may require stronger hash hardening and an adversarial collision evaluation.

**Open question: correction policy.** Consumers must decide whether a later corrected rank may trigger reversal of a prior editorial action.

**Open question: multi-region aggregation.** Region-level top-K lists cannot generally be merged into global top-K without retaining enough candidate coverage from every region.

### Interview pack

#### Q1. A 300-second window receives 12,000 accepted events/s. How many events enter the window?

**Answer.** `12,000*300=3,600,000` events.

**Reasoning.** Rate in events/s multiplied by duration in seconds yields events; this is volume, not distinct keys.

**Follow-up.** Explain why 3.6 million events does not imply 3.6 million map entries.

**Acceptable alternatives.** Converting five minutes to 300 seconds before multiplication is equivalent.

**Red flag.** Reporting 60,000 events by multiplying by five without converting minutes to seconds.

**Scoring.** 2 points: one for units and formula, one for correct arithmetic.

**Answer support.** Formula TK-01 and its dimension check.

#### Q2. With `epsilon=0.00004` and `N=2,880,000`, what is the additive error scale?

**Answer.** `E=0.00004*2,880,000=115.2` events.

**Reasoning.** Count-Min's illustrative additive scale grows with total accepted events.

**Follow-up.** State why a 115.2-event scale does not itself prove a rank ordering.

**Acceptable alternatives.** Saying 116 events after conservative upward rounding is acceptable if the gate uses that rounded value consistently.

**Red flag.** Dividing N by epsilon or treating epsilon as 0.04.

**Scoring.** 2 points: calculation and interpretation.

**Answer support.** Formula TK-03.

#### Q3. The rank-20/rank-21 gap is 180 events under the preceding error scale. May automation publish?

**Answer.** No; the conservative comparison uses `2E=230.4`, which exceeds 180.

**Reasoning.** Two estimates can have uncertainty in the ordering comparison, so the gap must exceed the selected combined bound.

**Follow-up.** Name a safe action: hold or human-visible provisional with automation fenced.

**Acceptable alternatives.** An estimator-specific interval test is acceptable if it is derived from the actual summary semantics.

**Red flag.** Publishing because 180 is greater than one 115.2-event error scale.

**Scoring.** 3 points: calculate, compare, and choose a safe action.

**Answer support.** Formula TK-03 sensitivity.

#### Q4. Why cannot a Count-Min sketch alone answer “which twenty keys are largest?”

**Answer.** It estimates a count for a key supplied to it but does not enumerate the keys that deserve querying.

**Reasoning.** Candidate discovery and point estimation are separate algorithmic obligations.

**Follow-up.** Select a companion: a bounded heavy-hitter summary or an exact candidate map.

**Acceptable alternatives.** A full external candidate catalogue scan is valid only if its cost and completeness are explicitly budgeted.

**Red flag.** Saying the sketch stores all keys because it has counters.

**Scoring.** 2 points: enumeration distinction and viable companion.

**Answer support.** Section 5c method comparison.

#### Q5. A key appears in two shard payloads for one window. Diagnose the highest-priority issue.

**Answer.** Treat it as a partition-ownership violation until proven otherwise.

**Reasoning.** Stable key ownership should give exactly one source of truth per key per window.

**Follow-up.** Inspect re-shard epoch, hash version, and normalization version before merging.

**Acceptable alternatives.** A deliberately replicated design is acceptable only if it carries an explicit aggregation rule.

**Red flag.** Deduplicating by choosing the larger count without investigating the split.

**Scoring.** 3 points: identify invariant, checks, and safe response.

**Answer support.** F-05 and the partition trace.

#### Q6. Why is a shard-local top-20 sufficient under strict key partitioning?

**Answer.** A globally top-20 key has one owning shard; if it were below that shard's local top-20, twenty local keys would outrank it and it could not be globally top-20.

**Reasoning.** The proof depends on one owner and the same total comparator locally and globally.

**Follow-up.** State when the proof fails: a key split across shards or inconsistent comparator.

**Acceptable alternatives.** Retaining more than 20 candidates is safe but may be needed for other uncertainty reasons.

**Red flag.** Claiming local top-20 is sufficient when events for one key are round-robin routed.

**Scoring.** 3 points: proof idea, assumptions, failure boundary.

**Answer support.** Candidate completeness rule.

#### Q7. Calculate candidate payload bytes for 16 shards, 100 candidates per shard, and 48 bytes per record.

**Answer.** `16*100*48=76,800` bytes, or 75 KiB.

**Reasoning.** There are 1,600 records, each with a fixed serialized size.

**Follow-up.** Identify one excluded cost: heap object overhead or serialization buffers.

**Acceptable alternatives.** Reporting 76.8 kB decimal is acceptable when explicitly labelled decimal rather than KiB.

**Red flag.** Calling this the complete memory budget.

**Scoring.** 2 points: correct product and a stated exclusion.

**Answer support.** Formula TK-04.

#### Q8. A retry carries the same source sequence after a process restart. What must happen before state update?

**Answer.** The dedup gate must recognize the sequence as acknowledged and reject the increment.

**Reasoning.** Input idempotency must precede sketch and heap mutation; output idempotency cannot undo double counting.

**Follow-up.** What if sequence identity resets? Fence the source or use a stable event identifier with re-budgeted retention.

**Acceptable alternatives.** Transactional source offsets are valid when they prove the same monotonic ownership property.

**Red flag.** Counting it and relying on a later snapshot retry key.

**Scoring.** 3 points: ordering, distinction, recovery answer.

**Answer support.** Idempotency rule and F-03.

#### Q9. A late event arrives 91 seconds after the window's allowed close, where `L=90` seconds. Which timestamp determines its treatment?

**Answer.** Event time determines membership, while the watermark and configured 90-second allowance determine whether its state may still mutate.

**Reasoning.** Processing arrival alone cannot assign event-time membership.

**Follow-up.** Give the safe disposition: append correction metadata or exclude-late according to declared policy.

**Acceptable alternatives.** A revised snapshot is acceptable only with a new correction sequence and explicit consumer semantics.

**Red flag.** Reopening the original immutable snapshot without a correction identity.

**Scoring.** 3 points: clock distinction, policy, immutable authority.

**Answer support.** Worked event-time trace and F-02.

#### Q10. A manifest has eleven of twelve expected shards, but the available eleven agree on their top-20. Should it publish?

**Answer.** No; hold because the missing shard can contain a globally winning key.

**Reasoning.** Agreement among present shards is not evidence that absent state is zero.

**Follow-up.** State what to retain: available payloads, expected manifest, timeout evidence, and reason code.

**Acceptable alternatives.** A separately authorized degraded mode may show a clearly incomplete human-only view, not automation eligibility.

**Red flag.** Filling the missing shard with zero counts to meet the deadline.

**Scoring.** 2 points: fail-closed choice and rationale.

**Answer support.** Manifest gate and F-09.

#### Q11. Which action handles a byte-identical retry of an already committed snapshot?

**Answer.** Return the existing authority record for the same `(window_id, configuration_hash, correction_sequence)` key.

**Reasoning.** The retry does not create a new decision when its serialized evidence is identical.

**Follow-up.** Explain the different-digest case: reject and investigate rather than overwrite.

**Acceptable alternatives.** Content-addressed storage is acceptable if it preserves the same authority semantics.

**Red flag.** Last-write-wins upsert.

**Scoring.** 2 points: idempotent key and conflict handling.

**Answer support.** Snapshot commit design and F-10.

#### Q12. The workload becomes nearly uniform and the rank-20 gap shrinks. What design choice should be revisited first?

**Answer.** Re-derive the error-versus-gap gate and consider exact counting or a larger summary before changing heap code.

**Reasoning.** Rank separability, not raw event throughput, determines whether the approximation is decision-safe.

**Follow-up.** Name a measured input needed for that review: observed boundary-gap distribution by window.

**Acceptable alternatives.** Reducing K is acceptable only if the business requirement permits a shallower publication.

**Red flag.** Assuming the original sketch width remains safe because memory has not changed.

**Scoring.** 3 points: identify binding constraint, evidence, and a valid response.

**Answer support.** Finding and Formula TK-03.


**Connection to selection algorithms.** Batch quickselect finds a cutoff efficiently but still assumes all candidate counts are materialized for its input.

**Connection to distributed snapshots.** The close barrier resembles a consistency boundary: the system must name exactly which shard states contributed to one decision.

**Connection to approximate query processing.** Error bounds are useful only when their confidence and direction match the action that consumes them.

**Connection to observability.** A digest is not an explanation by itself; paired dimensions, versions, and reason codes turn it into diagnostic evidence.

**Reading sequence.** Learn exact map-plus-heap invariants before applying bounded summaries, because the exact path provides the oracle for approximation failures.

**Interview assessment note.** A strong answer names assumptions before selecting an algorithm and treats a failed precondition as a different action, not as a smaller score.

**Boundary of this case.** It does not specify tenant isolation, privacy policy, cryptographic hashing, transport encryption, or storage retention guarantees.

**Review prompt.** Revisit every derived number when changing the traffic model, because accuracy and capacity scale through different variables.

**Evidence prompt.** Label generated replay outcomes as simulated and production observations as observed; neither label should be inherited from an algorithm name.

**Final cross-reference takeaway.** The reusable asset is the gate sequence: establish membership, completeness, uncertainty, capacity, and authority before publication.

## 15. Quick Reference Card

| Field | Quick reference |
|---|---|
| Decision | Publish, hold, or human-visible provisional top-20 snapshot at a 60-second tick. |
| Owner | Streaming platform lead owns ranking contract and exception policy. |
| Input grain | One accepted event keyed by normalized `term_id`, event time, and stable source identity. |
| Output grain | One immutable event-time window snapshot with ordered keys, estimates, action, and evidence digest. |
| Window | Trailing 300-second half-open event-time interval with 90-second allowed lateness in this illustrative design. |
| Comparator | Count descending; unsigned `term_id` ascending for every tie. |
| Candidate rule | Enumeration is separate from point estimation; retain a candidate-producing summary or exact map. |
| Partition rule | One stable key owner per window; reject mixed hash versions or re-shard epochs. |
| Input idempotency | Reject acknowledged producer sequence before state mutation. |
| Output idempotency | Reuse the existing record only for byte-identical snapshot retry under the authority key. |
| Approximation gate | For the illustrative conservative check, require `2*epsilon*N < f(K)-f(K+1)` plus candidate coverage. |
| Illustrative sizing | `N=2,880,000`, `epsilon=0.00004`, `E=115.2`, `G=331`, so `2E=230.4<G`. |
| Sketch sizing | `w=67,958`, `d=5`, 64-bit counters: 2,718,320 bytes or 2.592 MiB. |
| Candidate transport | 12 shards × 80 candidates × 48 bytes = 46,080 bytes or 45 KiB. |
| Late-data retention | 300-second window plus 90-second allowance retains 390 seconds of mutable bucket horizon. |
| Total illustrative state | 39,906,928 bytes or 38.058 MiB, leaving 25.942 MiB under a 64 MiB cap before unmodelled overhead. |
| Publish gate | Complete manifest, stable ownership, healthy dedup, candidate coverage, margin pass, memory pass, idempotent commit. |
| Hold trigger | Missing shard, conflicting payload, capacity breach, unresolved rank boundary, or unavailable recovery evidence. |
| Provisional meaning | Human-visible uncertainty only; `automation_eligible=false` remains mandatory. |
| Observability | Record watermark, accepted total, late count, duplicate rejects, manifest, rank margin, component memory, digest, reason code. |
| Recovery | Restore checkpoint and source progress; replay only unacknowledged input; hold if fidelity or deadline cannot be proven. |
| Evidence | ILLUSTRATIVE design arithmetic; no company deployment or measured result is claimed. |
| Stage | Proposed validation design; not executed. |
| Review trigger | Change K, window, lateness, cardinality, input rate, shard ownership, hash, candidate capacity, memory cap, or rank policy. |
