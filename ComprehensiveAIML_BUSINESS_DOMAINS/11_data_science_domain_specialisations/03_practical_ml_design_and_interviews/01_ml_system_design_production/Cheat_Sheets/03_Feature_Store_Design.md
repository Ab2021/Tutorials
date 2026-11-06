# 1. File Header

# Feature Store Design: Time, Consistency, and Operations

> **Purpose:** Design reproducible historical features and bounded-freshness online reads.
> **Audience:** ML engineers, data engineers, and system-design interview candidates.
> **Prerequisites:** SQL joins and windows, basic distributed systems, model serving, event-time processing, and supervised-learning leakage.
> **Outcome:** Given entity events, visibility times, and prediction cutoffs, derive a safe historical join, select serving semantics, and diagnose mismatches.
> **Exclusions:** Model training algorithms, full streaming-engine internals, and vendor-specific API guarantees.
> **Level:** 3 — advanced / production.
> **Version:** 0.1; draft, unreviewed; no production implementation or measurement claimed.
> **Evidence:** DERIVED invariants; ILLUSTRATIVE scenarios and numbers; HEURISTIC design choices; cited primary documentation for background only.
> **Retrieval cue:** Start at Section 4 for temporal correctness, Section 8 for incidents, or Section 11 for interviews.
> **Source mapping:** `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/03_Feature_Store_Design.md` is a directory-tree topic identifier, not a verified accessible source document.

A feature store is an optional platform for defining, materializing, discovering, and serving reusable model inputs.
Its value depends on the time, entity, version, and availability contracts, not on the database label.
An ordinary SQL pipeline and logged serving vectors are credible baselines for a small use case.
The reader should prefer the smallest system that satisfies reproducibility and serving constraints.

**Takeaway:** First specify what a prediction could have known; then choose infrastructure.

# 2. Concept Map / Mental Model

```text
Source events -----> event-time transformation -----> versioned feature history
      |                         |                            |
      |                         +-----> online materialize   +--> historical join
      |                                      |                    with label cutoffs
      v                                      v                          |
availability + lineage                entity-key lookup                 v
      |                                      |                    training rows
      +-------------------------> parity/age monitoring <------- serving logs
                                             |
                                             v
                                 prediction + fallback policy
```

Arrows represent data or operational feedback, not a claim of causation.
The offline join reconstructs the feature state eligible at each decision cutoff.
The online lookup returns state actually materialized by the serving deadline.
Those states agree only if transformation, input availability, revision and publication semantics align.
A feature timestamp alone does not prove the feature was visible to a prediction service.

**Takeaway:** Historical reconstruction and online serving are two views of one explicitly versioned contract.

# 3. Foundational Definitions

## 3.1 Vocabulary and operational boundaries

- **Entity:** a business object identified by a typed, stable key, such as `(tenant_id, account_id)`.
- Key namespace and tenant are part of identity; `42` from two tenants is not the same account.
- A feature value belongs to a particular entity grain and definition version.
- **Event time:** when the underlying event occurred, measured as a UTC instant.
- **Ingestion time:** when an input reached an ingestion boundary, not necessarily when it became usable.
- **Availability time:** when a record or feature revision first became available to the relevant consumer.
- **Decision time:** instant at which the system must decide, rather than label creation time.
- **Cutoff:** maximum permitted information time for the prediction, possibly before the decision time.
- **Feature time:** the logical time to which the value refers, such as a completed window end.
- **Publication lag:** time between logical feature time and availability to its consumer.
- **Point-in-time join:** per-example retrieval of a value satisfying temporal eligibility, identity, and version constraints.
- **Late arrival:** event received after its event time; it is not necessarily erroneous.
- **Backfill:** historical recomputation; a later backfill cannot retroactively make data visible at an earlier decision.
- **Revision:** a new published assertion about a historical feature state; retain its publication time.
- **Freshness age:** decision or read time minus the appropriate feature reference time.
- **TTL:** configured expiry/retention policy; it is not proof that a value was fresh when read.
- **Online store:** service optimized for low-latency current-state lookup, with implementation-dependent consistency.
- **Offline store:** durable historical data usable for training and analysis, not automatically leakage-safe.
- **Parity:** agreement under a defined comparison context; exact equality may be inappropriate for floats.
- **Missing:** a typed state with a reason, not automatically numeric zero.
- **Watermark:** a processing-system estimate or progress marker; its exact guarantees depend on the engine.
- **Lineage:** records identifying inputs, transformation, versions, run, and publication for an output.
- **Label maturity:** time until outcome information is observable and adjudicated for evaluation.

The feature key, label key, timestamps, and model version must have a documented serialization.
Business-day windows require an explicit timezone and calendar rather than a silent UTC-day assumption.
An event can be valid historically but unavailable to the historical serving system.
A corrected value can be more accurate today while still being invalid as yesterday's model input.
A value can be recently written but logically stale if the source itself stopped advancing.
An online missing value may reflect a new entity, expiry, permission failure, or ingestion incident.
Missingness can be predictive, but it must be reproduced consistently in training.

**Takeaway:** Entity, event time, and availability time answer different questions and must not be conflated.

# 4. Core Formulas with Full Derivation

## 4.1 Base case: one entity and immutable values

Assume a single tenant, one feature version, immutable values, and synchronized UTC timestamps.
Let `t` be the decision cutoff, in seconds or an equivalent timestamp domain.
Each row `r` has entity `e_r`, feature time `v_r`, availability time `a_r`, and value `x_r`.
A historical prediction for entity `e` is allowed to use only rows already effective and available at `t`.
The latest eligible logical value is selected; an empty eligible set produces explicit missingness.

```text
FORMULA [F-01]: eligible(r,e,t) := (e_r=e) AND (v_r<=t) AND (a_r<=t)
               r* := argmax eligible rows by v_r
WHERE: e and e_r are typed entity keys; v_r, a_r, t are UTC instants.
CONSTRAINT: immutable rows, one definition version, unique latest v_r; empty set is missing.
NUMERICAL EXAMPLE: t=10:00; row A=(v 09:50,a 09:55); row B=(v 09:59,a 10:02).
INTERPRETATION: A is eligible and B is not, so A is selected despite B's later event time.
```

Derivation: availability is a necessary precondition for historical knowledge.
Logical time prevents a feature representing a future period from entering the past.
Maximizing among only eligible rows selects the freshest eligible logical state.
If publication always occurs after logical time, the availability predicate implies the logical-time predicate.
Keeping both predicates documents the intended contract and protects against unusual upstream timestamps.

## 4.2 General case: versions, revisions, freshness, ties

Extend each row with definition version `d_r`, revision sequence `q_r`, and tenant-qualified entity.
Let `d` be the model's pinned definition version and `H` the maximum feature age in seconds.
Eligibility additionally requires matching version and `t-v_r <= H`.
For records with the same feature time, select the latest available revision, with a deterministic final tie-break.
One deterministic order is descending `(v_r, a_r, q_r)` after all eligibility filters.
The order assumes larger revision sequence means later authority when timestamps tie.
If a later publication may retract an earlier value, tombstones must participate in this ordering.
Tombstones must not be filtered before ranking, or an old value can be resurrected.

```text
FORMULA [F-02]: C={r: e_r=e, d_r=d, v_r<=t, a_r<=t, 0<=t-v_r<=H}
                r*=argmax_(r in C) (v_r,a_r,q_r)
WHERE: C is the finite candidate set; H is a nonnegative duration; q_r is ordered revision ID.
CONSTRAINT: entity identity is historically valid, versions are immutable, order is total,
            tombstones and availability for the actual serving boundary are retained.
NUMERICAL EXAMPLE: H=15 min, t=10:00, v=09:40 -> age 20 min -> excluded.
INTERPRETATION: return the latest eligible revision, or missing if none exists.
```

The age expression subtracts instants to produce a duration, so its units match `H`.
With one version, no age limit, and immutable unique logical times, F-02 reduces to F-01.
`a_r` must represent availability at the serving boundary being reconstructed.
A raw event's availability is not the availability time of an aggregate computed from it.
When actual historical publication records do not exist, a retrospective join can be only an approximation.
A conservative fixed lag is an assumption, not proof that every historical online write completed.

## 4.3 Variants and mechanisms

| Variant | Changed condition | Appropriate use | Avoid when |
|---|---|---|---|
| Event snapshot | One value per logical time | Slowly changing attributes | Overwrites erase publication history |
| Window aggregate | Inputs bounded by event and publication times | Counts or sums available at cutoff | Recomputed with late events as if already visible |
| Logged serving vector | Actual request inputs recorded | Training-serving reproducibility | Logging is incomplete, biased, or disallowed |
| Current online lookup | Latest materialized state at read | Low-latency inference | Historical point-in-time reconstruction |
| Fixed publication lag | Use only periods older than assumed lag | Conservative approximation with evidence | Lag is unbounded or historical outages matter |
| Bitemporal history | Retain valid and publication times | Corrections and audit | No record of original publication exists |

For a fixed publication lag `L`, use the window ending no later than `t-L` only if the delivery contract supports that lag.
Lag estimation must include processing, validation, replication, and materialization into the relevant read path.
A percentile lag is not a hard upper bound, and outages invalidate a naive fixed-lag guarantee.
Exact as-served training requires the actual availability or serving log, not merely a guessed lag.
For a count in a trailing interval, specify whether endpoints are `[t-W,t)` or another convention.
An event at the right endpoint is excluded from `[t-W,t)`, avoiding accidental inclusion of the current transaction.
If records are corrected or deleted, reconstruct the state of each event as known at the cutoff.
The aggregation then uses only visible event revisions whose event times lie in the window.
The aggregate itself must have been computed and available by the decision, unless it is computed on demand.

### State transition and correctness sketch

Input is a feature revision with tenant, entity, definition, logical time, publication time, and sequence.
Output is a current entity value plus durable, queryable history.
A writer validates schema and authorization, and records a durable revision or tombstone.
The online materializer accepts a revision only if its declared ordering outranks the stored state.
A duplicate with the same identity and payload is idempotent; conflicting duplicates are quarantined.
A later-arriving older logical value must not overwrite a newer one in a latest-value store.
The offline history retains both revisions rather than mutating away their original visibility.
The historical query filters by cutoff before ranking, so no later publication may win.
The process terminates after scanning the finite candidate set and applying a deterministic ranking.
On retry after a crash, durable identifiers and conditional updates prevent repeated side effects.
This argument assumes atomic or otherwise correctly fenced writes at the materialization boundary.
Retries alone do not provide exactly-once end-to-end processing.

## 4.4 Limits and counterexamples

If `H=0`, only a row with logical time exactly equal to the cutoff can pass.
If no row passes the filters, the correct result is missing, not the latest future record.
If `H` is very large, age filtering ceases to protect against obsolete features.
If publication is unboundedly late, a fixed lag cannot guarantee temporal correctness.
If clocks have unknown error, a timestamp comparison alone cannot prove real-world ordering.
If entity IDs are recycled, joining on the bare ID can transfer another person's history.
If only today's corrected snapshot exists, no algorithm can infer the actual prior values without additional evidence.
If a value was published at the exact cutoff, inclusion depends on the agreed ordering and timestamp precision.
In this sheet equality is eligible, but production must use a defensible tie convention and clock discipline.
If the model requests a previous definition version, newest-definition values are not a substitute.

**Takeaway:** Select after filtering by both meaning-time and actual visibility, with deterministic version and revision rules.

# 5. Worked Numerical Example

## 5.1 Illustrative fraud decision

All numbers and timestamps below are invented to teach the mechanics, not observed production evidence.
A fraud model scores account A at `10:00 UTC` with cutoff equal to that time.
The feature is the number of earlier transactions in the trailing 30 minutes.
The interval is `[09:30,10:00)` and excludes the transaction being scored.
The online model uses a materialized count with a maximum logical age of 15 minutes.
The model pins feature definition `v1`, and no identity changes occur in the example.

| Input event | Account | Event UTC | Available to aggregate UTC | In window? | Known by 10:00? |
|---|---|---|---|---|---|
| E1 | A | 09:35 | 09:36 | yes | yes |
| E2 | A | 09:48 | 09:52 | yes | yes |
| E3 | A | 09:58 | 10:03 | yes | no |
| E4 | A | 09:29 | 09:30 | no | yes |
| E5 | B | 09:55 | 09:56 | yes | yes, wrong entity |

The count that could be computed from available raw events at the cutoff is two: E1 and E2.
This count is not automatically the count that was actually served.
A materialized feature must also have reached the online reader by the cutoff.

| Revision | Entity | Definition | Feature time | Published online | Sequence | Value |
|---|---|---|---|---|---:|---:|
| R1 | A | v1 | 09:50 | 09:51 | 1 | 2 |
| R2 | A | v1 | 09:59 | 10:02 | 1 | 2 |
| R3 | A | v1 | 09:59 | 10:05 | 2 | 3 |
| R4 | B | v1 | 09:59 | 09:59 | 1 | 1 |
| R5 | A | v2 | 09:59 | 09:59 | 1 | 7 |

R1 is eligible: its entity and version match, its publication precedes cutoff, and its age is 10 minutes.
R2 and R3 are excluded because their publication times exceed 10:00.
R4 is excluded because it belongs to B; R5 is excluded because it is version v2.
Applying F-02 returns R1, so the as-served feature is 2 if the publication log accurately reflects the read path.
The cutoff's 15-minute maximum age is satisfied because `10:00 - 09:50 = 10 minutes`.
The count of two from event-level reasoning independently agrees with R1 for this constructed example.
The agreement is not guaranteed in general: a new event may be known but not yet materialized online.

At 10:06, assume a new score with the same 15-minute age limit and no other changes.
R3 is now available and outranks R2 on the same logical time by later publication and sequence.
Its age is seven minutes and its value is three, reflecting late event E3.
The difference from the 10:00 score is not evidence that the earlier score was wrong.
Selecting R3 at 10:00 would introduce a revision that did not yet exist for the serving reader.

## 5.2 Sensitivity and wrong paths

If the maximum allowed age is five minutes at 10:00, R1 is too old and the result is missing.
The system must use the explicitly approved missing-feature policy rather than reach into the future.
If R2 was actually published at 09:59:30, R2 becomes eligible and its logical age is one minute.
If publication timestamps only describe a warehouse write, they cannot prove online visibility.
If the online service read a replica behind the publication marker, the result may differ from the reconstructed state.
If E2 was available at 10:01 rather than 09:52, the raw-event count at 10:00 becomes one.
That change does not by itself establish what R1 contained; the actual feature revision is authoritative for as-served replay.
A join based only on feature time would choose R2 or R3 and leak a future publication.
A join using a final warehouse snapshot could return three and make offline evaluation optimistic.
A join on account without tenant could select another customer's count even with perfect timestamps.
A join using current definition v2 would silently change the meaning of the training input.

## 5.3 Production interpretation

One example has five input events and five revisions; a production dataset can contain billions of each.
Partition pruning, versioned manifests, and entity/time clustering reduce the amount of scanned history.
The correct query plan depends on data layout and the engine; no measured throughput is claimed here.
Late data, online replication delay, clock skew, and replay can make event-derived and served values differ.
Sampling logged feature vectors provides an independent comparison, subject to logging coverage and privacy.
A high historical match rate does not establish model lift, fairness, or correct label maturity.

**Takeaway:** The correct 10:00 value is 2, selected from what was actually published then, not from the eventual corrected state.

# 6. Visual / Geometric Intuition

```text
UTC time ->    09:50       09:51        09:59       10:00       10:02       10:05
R1             logical ---- published  ------------------------> eligible at cutoff
R2                                      logical ---- cutoff ---- published
R3                                      logical ---- cutoff --------------- published
Prediction                                          ^
                                                    decision cutoff
```

Horizontal distance represents elapsed time, not relative importance.
The logical feature time places the value on the domain timeline.
The publication marker places it on the knowledge timeline.
A row must lie on the permitted side of both boundaries.
After a correction, an older decision retains the earlier knowledge state.
At a later decision, the correction may be selected if it remains within the age bound.

## 6.1 Two-dimensional eligibility

```text
                         publication time a
                                ^
                                |   future knowledge: exclude
                          t  ---+-----------------------------
                                | eligible region | future logical time
                                |                 |
                                +-----------------+----------> feature time v
                                                  t
```

A candidate belongs to the lower-left region including the boundaries under this sheet's convention.
An entity mismatch, wrong version, tombstone, or age violation can still exclude a point in that region.
The plot is not a dependency graph and does not show how a feature was computed.
A raw-event eligibility region and an aggregate-publication region refer to different objects.

**Takeaway:** A single timeline cannot express both valid time and knowledge time.

# 7. Comparison / Selection Table

| Approach | Preconditions | Guarantee and boundary | Cost | Prefer when | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Batch SQL plus logged vectors | Manageable use cases and logging consent | Actual recorded values if logs are complete | Engineering and warehouse scans | Small team and few models | Strict online sharing requires unavailable serving path | HEURISTIC |
| Offline historical tables | Retained revisions and timestamps | As-of reconstruction under contract | History storage and joins | Training and audit | Source loses publication times | DERIVED |
| Online key-value materialization | Stable key and latency budget | Implementation-dependent current reads | Replicas, writes, operations | Low-latency feature lookup | Historical replay is the only requirement | HEURISTIC |
| On-demand source computation | Reachable sources and bounded execution | Freshness depends on all sources | Request fan-out and source load | Few cheap, real-time signals | Source latency or availability exceeds deadline | HEURISTIC |
| Shared feature platform | Governance and reuse justify operations | Shared contracts if enforced | Platform team and migration | Many consumers share features | Simpler baseline meets needs | HEURISTIC |
| Hybrid batch plus streaming | Defined merge and correction rules | Batch context plus recent signals | Two paths and consistency controls | Mixed refresh requirements | Team cannot support two-path recovery | HEURISTIC |

Apply hard privacy, availability, and temporal-correctness constraints before ranking alternatives.
If one service and offline predictions meet the objective, start with versioned SQL and observable logs.
If online latency requires materialization, add an online store with an explicit freshness and fallback contract.
If publication-history reconstruction is impossible, log actual serving values or collect the missing history first.
If batch features suffice for some inputs, do not force every feature through a real-time pipeline.
If freshness and latency conflict, define the permitted stale or missing behavior with the model owner.
Comparative latency, dollar cost, and model benefit are NOT MEASURED here.
Benchmark with identical feature definitions, request keys, dataset distributions, concurrency, and failure injections.
Record hardware, software versions, warmup, sample counts, confidence intervals, and percentile definitions.
Measure successful useful predictions as well as reads and infrastructure consumption.

**Takeaway:** Platform complexity is justified by a demonstrated contract and reuse need, not by the phrase feature store.

# 8. Failure Modes & Edge Cases

All following scenarios are hypothetical diagnostic exercises, not reported incidents.
Each separates plausible causes, discriminating signals, bounded actions, and follow-up verification.

## FS-01: Future publication

- Symptom: Offline AUC jumps after a warehouse backfill.
- Hypothesis A: A corrected feature was published after scoring.
- Hypothesis B: Labels leaked through a separate join.
- Distinguishing check A: Inspect publication timestamps on sampled matched rows.
- Distinguishing check B: Compare label-table columns and decision cutoffs.
- Bounded correction: Filter on original serving visibility and retain revisions.
- Verification: Replay original request vectors and inspect every selected timestamp.
- Mitigation risk: A conservative lag may increase missingness.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-02: Late raw event

- Symptom: Recomputed count exceeds logged value.
- Hypothesis A: Event arrived after the original decision.
- Hypothesis B: Online materializer was delayed even after ingestion.
- Distinguishing check A: Compare event ingestion and online publication times.
- Distinguishing check B: Compare request log with materialization audit.
- Bounded correction: Replay using both input and aggregate availability.
- Verification: Recompute sample with cutoff-bounded raw inputs and compare logs.
- Mitigation risk: Discarding late events permanently harms later predictions.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-03: Cross-tenant match

- Symptom: A feature appears under another customer.
- Hypothesis A: The join omitted tenant from a composite key.
- Hypothesis B: Source mapped multiple tenants to one global identity.
- Distinguishing check A: Trace tenant and entity on source and output rows.
- Distinguishing check B: Inspect the identity mapping effective at the cutoff.
- Bounded correction: Add tenant to every key and rebuild affected data.
- Verification: Run cross-tenant adversarial key-collision examples.
- Mitigation risk: Adding a key changes storage layout and migration needs.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-04: Recycled identity

- Symptom: Old history attaches to a new account.
- Hypothesis A: An identifier was reused after closure.
- Hypothesis B: A merge rule joined distinct account lifetimes.
- Distinguishing check A: Inspect account lifecycle intervals for the identifier.
- Distinguishing check B: Inspect identity merge and split history.
- Bounded correction: Include a stable incarnation ID or valid interval.
- Verification: Test an ID reused across non-overlapping lifetimes.
- Mitigation risk: Identity migration can break references to historical entities.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-05: Wrong entity grain

- Symptom: A user feature varies by transaction unexpectedly.
- Hypothesis A: The aggregation was grouped by transaction ID.
- Hypothesis B: A one-to-many join duplicated user rows.
- Distinguishing check A: Inspect grouping columns and distinct keys.
- Distinguishing check B: Measure row counts before and after each join.
- Bounded correction: Define one canonical feature grain and preaggregate children.
- Verification: Check uniqueness per entity and logical time.
- Mitigation risk: Preaggregation can hide needed transaction-level context.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-06: Version mismatch

- Symptom: Online values differ from training despite equal event inputs.
- Hypothesis A: Serving uses a newer definition.
- Hypothesis B: The transformation implementation differs across paths.
- Distinguishing check A: Inspect the model manifest and returned feature definition.
- Distinguishing check B: Compare transformation artifact hashes and test vectors.
- Bounded correction: Pin both paths to the approved version and migrate explicitly.
- Verification: Replay shared contract fixtures against both paths.
- Mitigation risk: Rolling back the model may require restoring an older materialization.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-07: Stale source

- Symptom: Write timestamps are recent but feature age keeps growing.
- Hypothesis A: A periodic job republishes old source data.
- Hypothesis B: Source event production has stopped.
- Distinguishing check A: Compare logical feature time with online write time.
- Distinguishing check B: Inspect upstream source progress and last event time.
- Bounded correction: Alert on source age as well as publication age.
- Verification: Inject a stopped source while continuing materialization.
- Mitigation risk: Suppressing old writes does not repair upstream availability.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-08: Expired value

- Symptom: Online lookup returns missing after expiry.
- Hypothesis A: TTL is shorter than the model's allowed age.
- Hypothesis B: The feature has not been refreshed due to a failed job.
- Distinguishing check A: Compare configured TTL with refresh cadence and age policy.
- Distinguishing check B: Inspect job status and the last successful output.
- Bounded correction: Align retention with policy and repair the failed producer.
- Verification: Test expiry at boundary with a controlled clock.
- Mitigation risk: Extending TTL can serve dangerously stale data.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-09: Default masking

- Symptom: The model sees many zeros during an outage.
- Hypothesis A: A missing-value fallback hides store failures.
- Hypothesis B: Zero is a valid value in the source population.
- Distinguishing check A: Break down missing reason and default-applied rate.
- Distinguishing check B: Compare valid zero values to fallback-injected zeros.
- Bounded correction: Expose missing flags and choose an approved fallback.
- Verification: Inject lookup failures and inspect returned reason and model behavior.
- Mitigation risk: A default can shift model calibration.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-10: Duplicate input

- Symptom: A count doubles after a retry.
- Hypothesis A: An event was delivered more than once.
- Hypothesis B: A legitimate second event reused a business attribute.
- Distinguishing check A: Count distinct stable event IDs within the window.
- Distinguishing check B: Compare source offsets and payloads for suspect events.
- Bounded correction: Deduplicate by a source-guaranteed event identity.
- Verification: Replay duplicate and distinct same-value events.
- Mitigation risk: An overly broad deduplication key removes legitimate events.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-11: Out-of-order write

- Symptom: Online value moves backward after replay.
- Hypothesis A: A delayed old logical revision overwrote the newer state.
- Hypothesis B: A valid correction was ordered by an inappropriate key.
- Distinguishing check A: Compare logical time, revision order, and arrival time.
- Distinguishing check B: Inspect update conditions and correction semantics.
- Bounded correction: Fence writes by the approved total order.
- Verification: Deliver old, new, duplicate, and corrected revisions in varied orders.
- Mitigation risk: Ordering by event time alone can discard a valid correction.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-12: Partial materialization

- Symptom: Some features are new and others old in one request.
- Hypothesis A: Feature groups update independently.
- Hypothesis B: One shard or region is delayed.
- Distinguishing check A: Inspect feature group revision IDs on the response.
- Distinguishing check B: Compare per-shard materialization watermarks.
- Bounded correction: Publish a snapshot marker or expose per-feature freshness.
- Verification: Test reads across a staged multi-group update.
- Mitigation risk: Atomic snapshots can increase latency and operational cost.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-13: Read replica lag

- Symptom: Warehouse history shows a value that online did not return.
- Hypothesis A: Publication timestamp marks the primary write only.
- Hypothesis B: A cache contains an older version.
- Distinguishing check A: Trace write acknowledgement and read replica visibility.
- Distinguishing check B: Compare bypass-cache reads with normal reads.
- Bounded correction: Record availability at the actual read boundary or use serving logs.
- Verification: Inject controlled replication lag and compare both paths.
- Mitigation risk: Stronger consistency may consume latency budget.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-14: Backfill overwrite

- Symptom: Historical validation changes each time it is rerun.
- Hypothesis A: A backfill replaced prior revisions.
- Hypothesis B: The query reads an unpinned dataset snapshot.
- Distinguishing check A: Check immutable revision history and its publication times.
- Distinguishing check B: Inspect dataset manifest and table snapshot identifiers.
- Bounded correction: Retain old knowledge states and pin dataset snapshots.
- Verification: Recreate the same dataset twice from the same manifest.
- Mitigation risk: Retaining history increases storage and deletion obligations.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-15: Time-zone error

- Symptom: Daily features shift around daylight-saving changes.
- Hypothesis A: Business-local days were treated as UTC days.
- Hypothesis B: Ambiguous local timestamps were parsed inconsistently.
- Distinguishing check A: Inspect conversion at DST transitions and entity timezone.
- Distinguishing check B: Test repeated and missing local clock hours.
- Bounded correction: Use explicit timezone-aware calendar boundaries.
- Verification: Compare known transition-day intervals to expected UTC instants.
- Mitigation risk: Changing day boundaries changes feature meaning and needs versioning.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-16: Boundary error

- Symptom: A current transaction enters its own prior-count feature.
- Hypothesis A: The right endpoint was inclusive.
- Hypothesis B: The current event was pre-aggregated into an online value.
- Distinguishing check A: Run an event exactly at the cutoff.
- Distinguishing check B: Compare serving sequence with aggregate update order.
- Bounded correction: Adopt a documented half-open window and update ordering.
- Verification: Test both endpoints and the decision event explicitly.
- Mitigation risk: Excluding a timestamp can omit concurrent legitimate events without a tie rule.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-17: Clock skew

- Symptom: Some publications appear to precede their source events.
- Hypothesis A: Producer and store clocks are unsynchronized.
- Hypothesis B: The feature has a legitimate logical time after publication.
- Distinguishing check A: Compare independent clock-offset telemetry and timestamps.
- Distinguishing check B: Check documented meaning of each timestamp.
- Bounded correction: Use a reliable ordering source and conservative uncertainty margin.
- Verification: Inject measured positive and negative skew in replay.
- Mitigation risk: Excessive margins unnecessarily reduce available data.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-18: Precision collision

- Symptom: Several revisions share identical timestamps.
- Hypothesis A: Timestamp granularity is too coarse.
- Hypothesis B: The source emitted duplicate sequence identifiers.
- Distinguishing check A: Inspect raw timestamp precision and sequence monotonicity.
- Distinguishing check B: Compare payload hashes for equal revision IDs.
- Bounded correction: Add a stable ordered tie-break and reject conflicts.
- Verification: Run equal-timestamp revisions in randomized arrival order.
- Mitigation risk: A synthetic sequence must be stable across replays.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-19: Tombstone resurrection

- Symptom: Deleted values reappear in historical joins.
- Hypothesis A: The query filtered tombstones before ranking.
- Hypothesis B: An old materializer retry restored a deleted row.
- Distinguishing check A: Inspect ranked candidates including tombstones.
- Distinguishing check B: Check write fences and replay logs for deleted keys.
- Bounded correction: Rank tombstones with values and return missing when they win.
- Verification: Replay a delete followed by a delayed older update.
- Mitigation risk: Deletion policy may require physical erasure beyond a logical tombstone.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-20: Null semantics

- Symptom: A missing feature appears as an observed zero.
- Hypothesis A: A SQL expression coalesced null too early.
- Hypothesis B: The source encodes unknown as zero.
- Distinguishing check A: Compare pre-transform null rates to post-transform values.
- Distinguishing check B: Inspect source documentation and unknown-value codes.
- Bounded correction: Preserve a typed missing reason and impute explicitly.
- Verification: Run null, zero, and absent-key fixtures end to end.
- Mitigation risk: Adding a missingness indicator changes model input schema.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-21: Schema drift

- Symptom: A model receives malformed feature vectors.
- Hypothesis A: A producer changed field type without compatibility checks.
- Hypothesis B: A consumer decoded a different schema version.
- Distinguishing check A: Compare producer schema and model-pinned schema IDs.
- Distinguishing check B: Inspect serialization errors and field-order tests.
- Bounded correction: Reject incompatible writes and release a versioned contract.
- Verification: Exercise old and new clients against compatibility fixtures.
- Mitigation risk: Permissive casting can conceal semantic changes.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-22: Unit mismatch

- Symptom: Amounts differ by a factor of 100.
- Hypothesis A: A producer switched dollars to cents.
- Hypothesis B: A conversion was applied twice.
- Distinguishing check A: Trace units and source field lineage end to end.
- Distinguishing check B: Inspect distribution ratios and conversion code.
- Bounded correction: Define units in the schema and use a new version for changes.
- Verification: Test known amounts and round-trip conversion.
- Mitigation risk: A distribution ratio alone cannot identify the root cause.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-23: Categorical vocabulary

- Symptom: Unknown-category rate spikes after deployment.
- Hypothesis A: Online encoding uses a different vocabulary.
- Hypothesis B: New categories actually appeared in production.
- Distinguishing check A: Compare vocabulary hashes and unknowns by producer version.
- Distinguishing check B: Inspect raw category frequencies and onboarding events.
- Bounded correction: Pin vocabulary artifact and provide trained unknown handling.
- Verification: Replay a fixed category suite on both pipelines.
- Mitigation risk: A new vocabulary may require retraining the model.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-24: Floating-point parity

- Symptom: Small differences appear in offline and online values.
- Hypothesis A: Aggregation order changed rounding.
- Hypothesis B: The underlying input sets or window boundaries differ.
- Distinguishing check A: Measure absolute and relative error against defined tolerance.
- Distinguishing check B: Compare event IDs and boundary decisions before numbers.
- Bounded correction: Use consistent precision and tolerance appropriate to the feature.
- Verification: Run deterministic small cases and high-dynamic-range cases.
- Mitigation risk: A loose tolerance can hide a material semantic error.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-25: Hot entity

- Symptom: One key experiences severe read or write latency.
- Hypothesis A: A single tenant or entity dominates traffic.
- Hypothesis B: An unrelated partition or network failure affects the shard.
- Distinguishing check A: Compare per-key and per-shard load distributions.
- Distinguishing check B: Compare affected keys on the same physical shard.
- Bounded correction: Cache or isolate the hot key without violating freshness.
- Verification: Load-test the skewed distribution and examine stale reads.
- Mitigation risk: Caching can increase staleness and invalidate revocations.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-26: Wide vector

- Symptom: Serving latency rises as features are added.
- Hypothesis A: Payload serialization and network transfer dominate.
- Hypothesis B: The store is slow due to backend contention.
- Distinguishing check A: Break down serialization, network, and store latency by payload size.
- Distinguishing check B: Compare identical payloads against an unloaded backend.
- Bounded correction: Remove unused features or batch appropriate lookups.
- Verification: Measure end-to-end deadline attainment after the change.
- Mitigation risk: Removing features can affect model quality.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-27: Fan-out timeout

- Symptom: A request misses its prediction deadline.
- Hypothesis A: Multiple sequential feature calls consume the budget.
- Hypothesis B: One downstream dependency has high tail latency.
- Distinguishing check A: Trace call graph and per-dependency deadlines.
- Distinguishing check B: Compare aggregate timeout with individual slow spans.
- Bounded correction: Batch independent keys and set explicit deadline propagation.
- Verification: Inject a slow dependency and verify fallback within deadline.
- Mitigation risk: Parallel fan-out can amplify backend load.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-28: Retry storm

- Symptom: Read or write traffic grows during partial failure.
- Hypothesis A: Clients retry synchronously without backoff.
- Hypothesis B: A genuine workload surge coincides with the incident.
- Distinguishing check A: Measure attempt-to-original-request ratio and retry timing.
- Distinguishing check B: Compare client rates to external request arrivals.
- Bounded correction: Bound retries with jitter and enforce admission control.
- Verification: Inject failures and measure amplification and recovery.
- Mitigation risk: Restricting retries can expose errors sooner to clients.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-29: Consumer lag

- Symptom: Feature freshness falls during peak traffic.
- Hypothesis A: Input rate exceeds processing capacity.
- Hypothesis B: A poisoned event repeatedly blocks a partition.
- Distinguishing check A: Compare input and processed rates across partitions.
- Distinguishing check B: Inspect retry loops and partition-specific offsets.
- Bounded correction: Restore spare capacity or quarantine the malformed record.
- Verification: Measure lag drain and validate quarantined data handling.
- Mitigation risk: Quarantining without reconciliation can silently lose information.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-30: Checkpoint recovery

- Symptom: Values repeat or regress after processor restart.
- Hypothesis A: Checkpoint state and sink commit are inconsistent.
- Hypothesis B: A replay uses a changed transformation version.
- Distinguishing check A: Compare recovered offsets, state version, and sink revisions.
- Distinguishing check B: Inspect deployment timeline and artifact hashes.
- Bounded correction: Use a compatible recovery protocol and idempotent sink writes.
- Verification: Kill and restart during writes and inspect final state.
- Mitigation risk: A consistent checkpoint alone does not guarantee an external sink transaction.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-31: Backfill collision

- Symptom: Live online values regress during historical rebuild.
- Hypothesis A: Backfill writes share the live namespace.
- Hypothesis B: Historical events have greater ingestion times than current events.
- Distinguishing check A: Inspect writer identity and logical revision times.
- Distinguishing check B: Compare materializer ordering keys with intended semantics.
- Bounded correction: Isolate backfill output and merge through a guarded promotion.
- Verification: Run backfill alongside live updates in an isolated environment.
- Mitigation risk: Rebuilding only offline history may leave online state uncorrected.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-32: Batch gap

- Symptom: A daily feature is missing for one partition.
- Hypothesis A: A scheduled run failed before publishing.
- Hypothesis B: The partition exists but represents an empty population.
- Distinguishing check A: Inspect orchestration state and output manifest.
- Distinguishing check B: Compare expected partition counts and source completeness.
- Bounded correction: Rerun the missing partition or publish an explicit empty result.
- Verification: Check the manifest across the repaired interval.
- Mitigation risk: Blind reruns can duplicate outputs without idempotency.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-33: Training snapshot drift

- Symptom: The same training manifest produces different data.
- Hypothesis A: Referenced tables were mutable.
- Hypothesis B: A nondeterministic tie-break selects different revisions.
- Distinguishing check A: Compare snapshot identifiers and physical files.
- Distinguishing check B: Re-run tie cases with stable unique ordering.
- Bounded correction: Pin immutable inputs and define deterministic ranking.
- Verification: Rebuild in two independent executions and compare checksums.
- Mitigation risk: Checksums prove byte equality, not absence of leakage.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-34: Incorrect label cutoff

- Symptom: Offline metrics are unusually optimistic.
- Hypothesis A: The join uses label creation rather than decision time.
- Hypothesis B: Labels incorporate outcomes after the intended horizon.
- Distinguishing check A: Compare label timestamps with logged prediction timestamps.
- Distinguishing check B: Review outcome horizon and maturity policy.
- Bounded correction: Use the actual decision cutoff and separate label maturity.
- Verification: Rebuild a temporally held-out dataset and inspect feature candidates.
- Mitigation risk: Moving cutoff earlier may increase missingness.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-35: Immature labels

- Symptom: Recent evaluation shows artificial performance decline.
- Hypothesis A: Outcomes have not had time to mature.
- Hypothesis B: The model genuinely degraded on recent cases.
- Distinguishing check A: Plot label completion by cohort age and prediction date.
- Distinguishing check B: Compare equally mature cohorts and model versions.
- Bounded correction: Evaluate cohorts after the agreed maturity window.
- Verification: Recompute metrics as cohorts mature and track revision.
- Mitigation risk: Waiting for maturity delays incident detection.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-36: Selection bias

- Symptom: Training parity looks good but serving failures are missed.
- Hypothesis A: Only successful requests were logged.
- Hypothesis B: The missing logs came from a different model version.
- Distinguishing check A: Compare request counts to log counts by status and version.
- Distinguishing check B: Audit sampling and logging failure paths.
- Bounded correction: Log failure reasons with a documented sampling policy.
- Verification: Reconcile logs to independent request counters.
- Mitigation risk: Logging identifiers can create privacy and retention obligations.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-37: Privacy deletion

- Symptom: An erased entity still appears in a derived dataset.
- Hypothesis A: Deletion reached the source but not derived copies.
- Hypothesis B: A key mapping links the person through another identifier.
- Distinguishing check A: Trace lineage through training sets, caches, and online stores.
- Distinguishing check B: Inspect identity resolution and deletion ledger.
- Bounded correction: Propagate deletion to all required copies under the applicable policy.
- Verification: Run a deletion drill and audit all declared destinations.
- Mitigation risk: Physical erasure may conflict with reproducibility; document governance resolution.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-38: Access leak

- Symptom: An unauthorized model can read restricted features.
- Hypothesis A: Permissions were attached to a broad feature group.
- Hypothesis B: A service account has excessive cross-tenant privileges.
- Distinguishing check A: Test access at feature, tenant, and principal boundaries.
- Distinguishing check B: Inspect effective role bindings and audit records.
- Bounded correction: Apply least privilege and isolate sensitive views.
- Verification: Run allowed and denied access tests under real service identities.
- Mitigation risk: Overly broad restrictions may break legitimate consumers.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-39: Missing lineage

- Symptom: A model owner cannot reproduce a suspicious input.
- Hypothesis A: Transformation artifact and source snapshot were not recorded.
- Hypothesis B: Lineage exists but is keyed by an unstable request ID.
- Distinguishing check A: Inspect the model manifest and feature revision metadata.
- Distinguishing check B: Trace one request through durable correlation IDs.
- Bounded correction: Record artifact, source snapshot, revision, and model linkage.
- Verification: Reproduce a sampled input from the recorded lineage.
- Mitigation risk: Retaining identifiers longer than needed increases privacy exposure.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-40: Online region outage

- Symptom: A region's predictions default all features.
- Hypothesis A: The local store is unavailable.
- Hypothesis B: Cross-region failover is blocked or too slow.
- Distinguishing check A: Compare regional health, routing, and remote reachability.
- Distinguishing check B: Inspect regional replica freshness and failover configuration.
- Bounded correction: Apply approved regional fallback and restore the local path.
- Verification: Exercise a regional failure with freshness and deadline assertions.
- Mitigation risk: Cross-region reads can violate residency or latency constraints.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-41: Cold entity

- Symptom: New customers receive no behavioral features.
- Hypothesis A: No history exists for the entity.
- Hypothesis B: An ingestion defect prevents new entity registration.
- Distinguishing check A: Compare new-entity age with lookup and ingestion status.
- Distinguishing check B: Inspect unrelated new keys and source event registration.
- Bounded correction: Use trained cold-start features and a missingness flag.
- Verification: Evaluate cold-start cohorts separately from established entities.
- Mitigation risk: A generic default can disadvantage specific cohorts.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-42: Feature drift

- Symptom: The distribution of a feature changes sharply.
- Hypothesis A: A source or transformation changed.
- Hypothesis B: The actual population or business behavior changed.
- Distinguishing check A: Segment by producer version and transformation release.
- Distinguishing check B: Compare independent source distributions and population mix.
- Bounded correction: Roll back a broken transformation or investigate legitimate change.
- Verification: Recompute distributions on matched cohorts after correction.
- Mitigation risk: Automatic rollback can hide a real and important shift.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-43: Feature leakage by proxy

- Symptom: A feature predicts the label almost perfectly.
- Hypothesis A: An input is populated only after the outcome.
- Hypothesis B: A legitimate strong predictor exists before the decision.
- Distinguishing check A: Trace input creation and availability against cutoff.
- Distinguishing check B: Test predictive performance on strictly time-separated cohorts.
- Bounded correction: Remove post-outcome information or use as-known revisions.
- Verification: Inspect availability and outcome timing for sampled rows.
- Mitigation risk: Deleting a legitimately predictive pre-decision feature may reduce utility.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-44: Entity merge

- Symptom: Historical values change after accounts are merged.
- Hypothesis A: The join uses today's identity graph.
- Hypothesis B: The source backfilled its mappings without valid-time history.
- Distinguishing check A: Compare identity graph versions at decision time.
- Distinguishing check B: Trace merge effective and publication timestamps.
- Bounded correction: Use an as-of identity mapping with recorded visibility.
- Verification: Replay a merge that happened after a past decision.
- Mitigation risk: Applying historical identity can complicate current-user aggregation.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-45: Feature deletion

- Symptom: A retired feature is still consumed by one model.
- Hypothesis A: Ownership and model dependencies were not inventoried.
- Hypothesis B: An untracked batch consumer bypasses the catalog.
- Distinguishing check A: Search model manifests and read audit logs.
- Distinguishing check B: Inspect scheduled batch queries and data-access records.
- Bounded correction: Deprecate with a migration deadline and explicit consumer acknowledgements.
- Verification: Disable in a staging environment and monitor dependency failures.
- Mitigation risk: A read log may miss rare or dormant consumers.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-46: Model rollback

- Symptom: A rollback fails because old feature values are unavailable.
- Hypothesis A: Old definitions were deleted too early.
- Hypothesis B: The old model expects a different serialization.
- Distinguishing check A: Inspect version retention and online namespace availability.
- Distinguishing check B: Compare model manifest against current schema artifacts.
- Bounded correction: Retain compatible definitions for the rollback window.
- Verification: Rehearse rollback with the real previous model artifact.
- Mitigation risk: Long retention increases cost and governance burden.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-47: SLO denominator

- Symptom: Freshness appears healthy despite widespread missing reads.
- Hypothesis A: The metric excludes missing results.
- Hypothesis B: The freshness threshold uses write time instead of logical time.
- Distinguishing check A: Compare all eligible requests with age-metric sample count.
- Distinguishing check B: Compare logical timestamps with store update timestamps.
- Bounded correction: Count missing and stale outcomes explicitly in the SLO.
- Verification: Inject missing and stale reads and verify metric changes.
- Mitigation risk: Combining distinct failure classes can obscure the remedy.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-48: Distribution skew

- Symptom: A feature's online and offline histograms diverge.
- Hypothesis A: Different populations are being compared.
- Hypothesis B: The transformations truly differ for identical requests.
- Distinguishing check A: Match cohorts by request, time, entity, and version.
- Distinguishing check B: Replay identical request vectors across both paths.
- Bounded correction: Repair the path mismatch or redefine the population comparison.
- Verification: Track matched-pair discrepancy and missingness.
- Mitigation risk: A histogram match can conceal row-level swaps.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-49: Permission timeout

- Symptom: Store errors look like missing entities.
- Hypothesis A: Authorization failure is mapped to not-found.
- Hypothesis B: The key genuinely does not exist.
- Distinguishing check A: Inspect error codes and authorization audit events.
- Distinguishing check B: Test a known present key with allowed and denied credentials.
- Bounded correction: Preserve typed denial and availability statuses.
- Verification: Verify monitoring separates not-found, denied, and timeout.
- Mitigation risk: Exposing detailed denial reasons to clients may leak information.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-50: Cache invalidation

- Symptom: Old values persist after a correction.
- Hypothesis A: Cache invalidation was lost.
- Hypothesis B: Cache TTL exceeds the correction freshness requirement.
- Distinguishing check A: Compare cache version to authoritative store version.
- Distinguishing check B: Inspect invalidation delivery and expiry timestamps.
- Bounded correction: Use version-aware invalidation or a bounded TTL.
- Verification: Correct a value and observe propagation to all readers.
- Mitigation risk: Frequent invalidations can overload the cache or backing store.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-51: Outlier contamination

- Symptom: A rolling sum becomes implausibly large.
- Hypothesis A: A malformed event has an incorrect unit.
- Hypothesis B: A real rare event legitimately has a high value.
- Distinguishing check A: Trace the sum to contributing event IDs and units.
- Distinguishing check B: Compare the event against independent source records.
- Bounded correction: Quarantine invalid records and publish an auditable correction.
- Verification: Recompute the window and compare pre- and post-correction revisions.
- Mitigation risk: Clipping without domain evidence can erase real risk signals.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-52: Window expiry

- Symptom: Counts remain high after old events should leave.
- Hypothesis A: The processor never schedules expiration for idle keys.
- Hypothesis B: A longer window definition was deployed.
- Distinguishing check A: Inspect state timers and last update for quiet entities.
- Distinguishing check B: Compare deployed definition and expected window duration.
- Bounded correction: Schedule event-time or processing-time expiry consistent with the contract.
- Verification: Advance time without new events and verify count decreases.
- Mitigation risk: Timer semantics under watermark stalls must be explicitly tested.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

## FS-53: Storage growth

- Symptom: Historical feature storage grows unexpectedly.
- Hypothesis A: Every retry writes a new physical duplicate.
- Hypothesis B: Legitimate corrections and retention grew.
- Distinguishing check A: Compare unique revision IDs with stored row counts.
- Distinguishing check B: Break down bytes by feature, version, tenant, and age.
- Bounded correction: Deduplicate safely and apply governed retention.
- Verification: Measure logical and physical counts after compaction.
- Mitigation risk: Compaction must not discard historical visibility needed for replay.
- Provenance: hypothetical diagnostic; no incident or measurement is asserted.

**Takeaway:** Diagnose the mechanism before applying a fix; preserve evidence and verify the specific invariant restored.

# 9. Complexity & Cost Analysis

## 9a. Operation models

Let `N` be the number of historical feature revisions, `M` the number of labeled examples, and `K` the number of feature groups requested per example.
Let `C` be the total number of eligible-or-nearby candidates examined by a join plan.
Let `S` be the number of bytes in the stored history, including payload but excluding unspecified overhead.
The following are logical models, not promises about a particular warehouse, index, or service.

| Operation | Time / work | Extra space | Important assumptions and bottleneck |
|---|---|---|---|
| Naive historical join | Up to O(MN) comparisons per group | Depends on plan and output | Full cross comparison; no indexing or pruning |
| Sorted merge | O(M log M + N log N + M + N) before revision handling | Sort-dependent | Single key/time order, suitable join semantics and sorting |
| Indexed lookup | O(log N + c) per lookup under balanced-index assumptions | O(N) index entries | `c` candidates inspected; actual distributed I/O varies |
| Online lookup | Expected O(1) in a local hash table | O(number of active keys) | Does not include RPC, network, cache misses, or failures |
| Replay | O(N) record applications plus state-store overhead | Working state plus retained input | Every relevant revision must be read at least once |
| Full feature vector | Roughly K lookups absent batching | O(K) response fields | Grouping and transport change request count and bytes |

A warehouse engine may repartition by entity and sort candidates, adding network shuffle and temporary storage.
A hot key creates skew even when average data volume per partition looks small.
Adding an index speeds some reads but consumes write capacity and storage.
Historical joins can emit many candidates before ranking, so monitor intermediate cardinality, not just final row count.
A serving store's local algorithmic complexity does not establish its distributed p99 latency.
Replay cost depends on checkpoint frequency, source retention, data skew, and sink write capacity.
For windowed streaming features, state size depends on active keys, window design, event rate, and allowed lateness.
A compact running sum is not sufficient if the system must retract old individual events without auxiliary state.

## 9b. Illustrative workload sizing

Assume 10 million active entity keys and 40 bytes of raw feature payload per key.
Assume three full replicas, with no compression, metadata, indexes, logs, or allocator overhead in the calculation.
Assume 2,000 predictions per second and one 40-byte payload returned per prediction.
These values are hypothetical; the actual system, implementation, hardware, and cloud prices are unspecified.

```text
FORMULA [F-03]: S_payload = E * b * R
WHERE: E active entities [entities], b payload [bytes/entity], R replica count [dimensionless].
CONSTRAINT: all keys have equal payload; replicas each store all keys; no storage overhead.
NUMERICAL EXAMPLE: E=10,000,000; b=40; R=3 -> 1,200,000,000 bytes.
INTERPRETATION: replicated payload is 1.2 GB decimal, not required provisioned capacity.
```

Derivation: one copy contains `E*b` bytes; multiplying by the number of full copies counts stored payload across replicas.
One replica contains `10,000,000 * 40 = 400,000,000 bytes`.
Three replicas contain `400,000,000 * 3 = 1,200,000,000 bytes`.
Dividing by `1,000,000,000 bytes/GB` gives `1.2 GB` decimal.
Dividing by `1,073,741,824 bytes/GiB` gives approximately `1.118 GiB`.
The independent reverse check is `1.2 GB * 1,000,000,000 bytes/GB = 1,200,000,000 bytes`.
The entity unit cancels and replica count is dimensionless.
Index pages, row headers, keys, encryption, memory fragmentation, and temporary copies add real capacity.
A 40-byte value also does not imply a 40-byte network message.
If every key instead holds 80 bytes, payload doubles to 2.4 GB under the same replication assumption.
If only one replica contains a given key, the full-replication assumption fails and the calculation must be changed.

For the network example, request rate multiplied by response payload gives logical response payload throughput.

```text
FORMULA [F-04]: B = Q * b
WHERE: B response payload [bytes/s], Q successful reads [reads/s], b [bytes/read].
CONSTRAINT: fixed response size; no protocol, key, request, retransmission, or replica-traffic overhead.
NUMERICAL EXAMPLE: Q=2,000 reads/s; b=40 bytes/read -> B=80,000 bytes/s.
INTERPRETATION: this is logical response payload only, not provisioned network bandwidth.
```

The units cancel as `reads/s * bytes/read = bytes/s`.
Multiplying `80,000 bytes/s` by eight gives `640,000 bits/s` of payload.
A tenfold peak in request rate, with unchanged payload, produces `800,000 bytes/s` of payload.
The calculation excludes reads for retries, misses, prefetch, internal replication, and TLS overhead.
It does not imply the system can serve that rate or meet any latency percentile.
Measure request size distribution, peak concurrency, hit rate, actual payloads, and resource saturation.

## 9c. Economics and operational budgets

A cost comparison should specify currency, date, workload period, and included line items.
No price is asserted in this example because no cloud, region, or purchase arrangement is specified.
Count warehouse scan and compute, stream processing, online storage and replicas, network, logs, and operations labor.
Include training and backfill compute rather than charging only request-time infrastructure.
Include incident response, access review, feature ownership, and migration costs as appropriate.
Track cost per successful prediction with all required features, not just cost per attempted lookup.
Define numerator and denominator for the same time period and population.
A cheap request that returns unusable defaults is not equivalent to a successful feature read.
A lower resource bill does not prove a better product outcome if model quality or availability worsens.
Set a per-request deadline before allocating internal time budgets.
Reserve time for model execution, authorization, serialization, network, and a fallback decision.
Do not add independently measured component p99 values and call the result an end-to-end p99.
Measure end-to-end percentile under representative concurrency and fault conditions.

## 9d. Monitoring contract

Define feature availability as requests returning an acceptable, authorized value divided by eligible requests.
Define missingness separately by no-history, expiration, rejected schema, timeout, and denied access.
Define freshness age using the feature logical time and the actual decision or read time.
Report the fraction of eligible requests exceeding the age threshold, counting missingness separately.
Publish the sampling method, window, dimensions, and treatment of null timestamps.
Monitor source progress, processing lag, materialization lag, and read age as distinct measurements.
Record per-feature and per-model impact so a benign stale feature does not hide a critical outage.
Compare training and serving with matched request vectors where logging permits it.
Monitor numeric differences with a specified tolerance and categorical differences with exact semantics.
A mismatch rate denominator must include both matched and missing comparisons when appropriate.
Trace deadlines and errors across the request path using privacy-preserving correlation identifiers.
Set alerts based on consumer impact and error-budget policy rather than arbitrary values in this sheet.
A freshness alarm says that a contract failed; it does not identify which upstream stage caused the failure.

**Takeaway:** Model complexity and byte arithmetic help scope capacity, but only representative measurement supports an SLO or price claim.

# 10. Key Theorems / Results Reference

No formal feature-store theorem is claimed; the following are conditional design results derived in this sheet.

| Result | Required conditions | Practical implication | Does not imply |
|---|---|---|---|
| F-01 eligibility | Accurate entity and availability times | Filtering excludes known future publications | No other leakage exists |
| F-02 deterministic selection | Complete revision history and total ordering | The same eligible set selects the same row | The warehouse knows the exact online replica state |
| Logged-vector replay | Complete, correctly linked serving logs | Reproduce the recorded feature vector | Reproduce the entire decision environment |
| Payload arithmetic | Fixed payload and full replicas | Calculate payload lower bound | Capacity, cost, or achievable performance |
| Conditional idempotency | Stable IDs and atomic fenced update | A duplicate cannot change accepted state | Every external side effect happens exactly once |

Sources below describe concepts and implementation-specific behavior; their exact product semantics must be verified for a chosen release.
They do not establish that this illustrative design is deployed or that its numbers were measured.

- Feast documentation, [Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins): background on historical feature retrieval; verify release-specific details before implementing.
- Feast documentation, [Feature retrieval](https://docs.feast.dev/getting-started/concepts/feature-retrieval): background on online and historical retrieval; product APIs can change.
- Google Cloud, [Dataflow: watermarks and late data](https://cloud.google.com/dataflow/docs/concepts/streaming-pipelines#watermarks_and_late_data): background on streaming time and late data; not a universal engine guarantee.
- Apache Kafka, [Design documentation](https://kafka.apache.org/documentation/#design): source for Kafka-specific delivery and durability design; not proof of application-level exactly-once effects.
- Sources are cited as pointers, not as independently verified current guarantees; review and version verification remain open.

**Takeaway:** A correct local invariant holds only for the timestamps, histories, and storage semantics actually supplied.

# 11. Interview / Exam Questions

The following questions are constructed exercises, not claims about any employer's interview.
A score of 0 means incorrect, 1 partial, 2 correct with assumptions, and 3 a defended tradeoff.

## Level 1 — Definition / Recall

### Q01 [L1] — Availability versus event time

- Prompt: An event occurred at 09:58 and reached the relevant feature service at 10:02; can a 10:00 prediction use it?
- Clarification: Assume no separate earlier copy was visible.
- Answer: No: event time is in the past but availability is later than the decision.
- Defensible alternative: If the event is synchronously supplied in the request before the decision, that separate input may be valid.
- Follow-up: What if the warehouse received it at 09:59 but online received it at 10:02?
- Follow-up answer: The online historical prediction still cannot use it; warehouse availability is a different boundary.
- Red flag: Treating event time alone as evidence of visibility leaks future information.
- Scoring: use the 0–3 rubric above.
- Answer support: Sections 3–4, F-01.

### Q02 [L1] — Feature store purpose

- Prompt: Does using a feature store automatically prevent training-serving skew?
- Clarification: Assume the platform permits multiple feature definitions and pipelines.
- Answer: No: consistency depends on shared definitions, event availability, version pins, and matching read semantics.
- Defensible alternative: A shared implementation and logged serving values reduce some mismatch risks but do not prove all inputs agree.
- Follow-up: What evidence would you request?
- Follow-up answer: Compare matched serving vectors with point-in-time offline values and investigate missing comparisons.
- Red flag: Equating a product name with an end-to-end correctness guarantee is a category error.
- Scoring: use the 0–3 rubric above.
- Answer support: Sections 2 and 7.

### Q03 [L1] — Entity grain

- Prompt: Why is account ID alone sometimes an unsafe feature key?
- Clarification: Assume IDs are scoped to tenants or can be reused.
- Answer: The same ID can identify different entities; include tenant and, if necessary, an incarnation or validity interval.
- Defensible alternative: A globally unique, never-reused surrogate can work if its contract is enforced.
- Follow-up: How does an account merge affect the answer?
- Follow-up answer: Use the identity relationship valid and visible at the decision rather than today’s mapping.
- Red flag: A correct time join over an incorrect identity still leaks another entity’s information.
- Scoring: use the 0–3 rubric above.
- Answer support: Sections 3 and 8 FS-03–04.

## Level 2 — Derivation / Apply

### Q04 [L2] — Temporal trace

- Prompt: At 10:00, rows are (09:50,09:51,value 2) and (09:59,10:02,value 3), ordered as logical time then publication time. Which is selected?
- Clarification: Same entity and definition; age bound 15 minutes.
- Answer: The first row is eligible and age 10 minutes; the second was not published, so the result is 2.
- Defensible alternative: If no online publication log exists, return an approximation with that limitation instead of claiming exact replay.
- Follow-up: What if the age bound becomes five minutes?
- Follow-up answer: The first row expires and the second remains future, so the result is missing.
- Red flag: Selecting value 3 by maximum logical time ignores publication lag.
- Scoring: use the 0–3 rubric above.
- Answer support: Sections 4–5, F-02.

### Q05 [L2] — Count derivation

- Prompt: A trailing 30-minute window ending at 10:00 excludes its right endpoint; eligible event times are 09:30, 09:59, and 10:00. What is the count?
- Clarification: Assume all three events were visible before the decision and share the correct entity.
- Answer: The interval [09:30,10:00) contains the first two events, so the count is 2.
- Defensible alternative: Another endpoint convention is defensible only if specified and consistently implemented.
- Follow-up: If 09:59 arrives at 10:01?
- Follow-up answer: The count known at 10:00 becomes 1.
- Red flag: Including the scoring event at 10:00 may leak decision-time information.
- Scoring: use the 0–3 rubric above.
- Answer support: Sections 4.3 and 5.

### Q06 [L2] — Replica sizing

- Prompt: How many payload bytes are stored for ten million entities with 40 bytes each and three complete replicas?
- Clarification: Exclude all metadata, indexes, and compression.
- Answer: 10,000,000 × 40 × 3 = 1,200,000,000 bytes, or 1.2 GB decimal.
- Defensible alternative: A sharded non-full replica layout requires a different replication factor.
- Follow-up: What if the payload doubles?
- Follow-up answer: Payload doubles to 2,400,000,000 bytes, before overhead.
- Red flag: Calling this capacity requirement ignores keys, overhead, headroom, and workload.
- Scoring: use the 0–3 rubric above.
- Answer support: Section 9, F-03.

### Q07 [L2] — Revision ordering

- Prompt: Two revisions share logical time 09:59, published at 10:02 and 10:05; which can a 10:03 historical prediction use?
- Clarification: Both match the entity and model definition and satisfy the age bound.
- Answer: Only the 10:02 revision is available by 10:03.
- Defensible alternative: At 10:06, the later revision wins if the specified sequence agrees with publication ordering.
- Follow-up: What if the 10:05 revision is a tombstone?
- Follow-up answer: At 10:06 rank the tombstone and return missing; do not resurrect the older value.
- Red flag: Filtering deletes before ranking changes historical state.
- Scoring: use the 0–3 rubric above.
- Answer support: Section 4.2.

## Level 3 — Judgment / Design / Debug

### Q08 [L3] — Selection tradeoff

- Prompt: A single daily model needs reproducible training data but no online inference; should you build an online feature store?
- Clarification: The team can use versioned SQL and snapshot manifests.
- Answer: Start with the simpler offline baseline if it meets correctness, security, and operational needs.
- Defensible alternative: A shared platform may be justified by other consumers or governance requirements.
- Follow-up: What changes if low-latency shared inference is introduced?
- Follow-up answer: Evaluate materialization, lookup deadlines, freshness, fallback, and the added operating burden.
- Red flag: Buying or building infrastructure before a requirement does not establish value.
- Scoring: use the 0–3 rubric above.
- Answer support: Section 7.

### Q09 [L3] — Ambiguous stale signal

- Prompt: A freshness alert fires while online writes remain frequent. Give two competing explanations and a discriminating check.
- Clarification: Freshness uses logical feature time, not write time.
- Answer: The source may be stale while old data is repeatedly republished, or a transformation may emit an old logical timestamp.
- Defensible alternative: A monitoring bug is also possible; compare raw source progress with transformation and publication times.
- Follow-up: What if the source is healthy but a single shard is delayed?
- Follow-up answer: Compare per-shard offsets, queue lag, and recent logical timestamps; isolate the affected partition.
- Red flag: Restarting everything without measuring the stages can destroy evidence and extend impact.
- Scoring: use the 0–3 rubric above.
- Answer support: Section 8 FS-07 and FS-29.

### Q10 [L3] — Recovery and ordering

- Prompt: A delayed replay writes an older feature after a newer live update. How should the current state behave?
- Clarification: Assume both belong to the same definition and the ordering contract is valid.
- Answer: The old revision should not replace the higher-ranked live state; use an atomic conditional update and retain history.
- Defensible alternative: A correction may legitimately replace an old logical value if its declared revision semantics authorize it.
- Follow-up: What if the writer crashes after sending the update?
- Follow-up answer: Retry using a stable revision ID and a fenced idempotent sink operation.
- Red flag: Processing exactly once in one component is not proof that all external effects are exactly once.
- Scoring: use the 0–3 rubric above.
- Answer support: Sections 4.3 and 8 FS-11.

### Q11 [L3] — Parity verification

- Prompt: How would you test a claim that offline and online features agree?
- Clarification: Assume consent permits sampled request-vector logging.
- Answer: Match request ID, entity, cutoff, definition, revision, and model; compare values, missingness, and typed error reasons.
- Defensible alternative: When logs are unavailable, replay controlled fixtures and clearly state that production as-served parity is unproven.
- Follow-up: How should numeric values be compared?
- Follow-up answer: Use a justified absolute or relative tolerance, and separately verify identical input sets and window boundaries.
- Red flag: Similar histograms do not prove row-level equality.
- Scoring: use the 0–3 rubric above.
- Answer support: Sections 8 FS-24 and FS-48.

### Q12 [L3] — Unavailable publication history

- Prompt: A warehouse contains only corrected latest snapshots and no original online logs. Can you reconstruct exact historical serving values?
- Clarification: No other audit source exists.
- Answer: No: multiple distinct historical states are consistent with the same latest snapshot, so exact reconstruction is not identifiable.
- Defensible alternative: A conservative lag or approximate replay may support limited analysis if limitations are documented.
- Follow-up: How can future reproducibility improve?
- Follow-up answer: Record publication revisions or sampled as-served vectors with version, identity, cutoff, and governance controls.
- Red flag: Pretending a fixed lag is proof of visibility conceals uncertainty.
- Scoring: use the 0–3 rubric above.
- Answer support: Sections 4.2 and 4.4.

### Q13 [L3] — Privacy and audit

- Prompt: A deletion request affects source events and previously materialized feature vectors. What must the design address?
- Clarification: Assume the applicable policy requires deletion from identified derived copies.
- Answer: Trace identity and lineage into online stores, caches, history, logs, and governed datasets; propagate and audit deletion.
- Defensible alternative: Exact treatment of backups and model artifacts requires the applicable legal and organizational policy.
- Follow-up: What tension exists with reproducibility?
- Follow-up answer: An immutable replay dataset may conflict with erasure; document and apply the approved governance resolution.
- Red flag: A tombstone in one online table does not prove all copies were erased.
- Scoring: use the 0–3 rubric above.
- Answer support: Section 8 FS-37.

**Takeaway:** Strong answers state the temporal contract, justify the selected mechanism, and identify what evidence is missing.

# 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Feature-store design for temporally correct training and bounded-freshness serving |
| Core formula | Choose maximum `(feature time, publication time, revision)` among matching entity/version rows with both times at or before cutoff and age within the limit; tombstones participate. |
| Key assumption | Publication time reflects visibility to the actual consumer and revision history is retained. |
| Complexity | Naive historical join O(MN); indexed lookup O(log N + c) under the assumptions in Section 9. |
| Use when | Shared definitions, historical replay, or low-latency serving justify platform operations. |
| Avoid when | Simpler versioned SQL meets the need, or historical visibility cannot be established and exact replay is claimed. |
| Go deeper | Model monitoring and drift are planned in this collection; no authored handoff is asserted. |
| Guardrail | Monitor missing, stale, denied, and failed reads separately; pin definitions and test matched vectors. |
| Evidence | Illustrative numbers; no measured deployment; draft and unreviewed. |
