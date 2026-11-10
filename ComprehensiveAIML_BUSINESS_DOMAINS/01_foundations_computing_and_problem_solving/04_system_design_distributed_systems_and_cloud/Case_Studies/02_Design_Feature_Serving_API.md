---
case_id: "CASE-04-002"
case_title: "Design a Feature-Serving API"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing"
primary_pattern: "point-in-time-serving"
secondary_patterns: ["correctness-contract", "tail-latency-budget"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "ML platform lead"
source_index: "04_system_design_distributed_systems_and_cloud/Case_Studies/INDEX.md"
review_trigger: "Any change to feature definitions, label delay, latency SLO, or model family"
---

# Design a Feature-Serving API

**Evidence declaration:** This is an ILLUSTRATIVE teaching case. The query volume, entity
cardinalities, feature byte sizes, hit rates, and every computed consequence below are constructed to
make one design decision auditable. No company architecture is described, no vendor benchmark is quoted,
and nothing here has been operated. Arithmetic is shown with intermediate values so a reader can check
it; wherever an input is a planning assumption rather than a measurement, the value carries the label
ILLUSTRATIVE at the point of use. **Implementation stage:** proposed. **Editorial status:** draft;
source verification and publication review remain OPEN.

## 1. The One-Line Framing

A feature-serving API is not a cache in front of a database; it is a **time machine with a deadline**.
Its job is to return, for a given entity and a given decision moment, exactly the feature values that the
training pipeline would have computed from data available before that moment — and to do so inside a
latency budget that the online system cannot extend.

Those two requirements pull in opposite directions and the tension is the entire design. Correctness
pushes toward reading history at a precise timestamp, which is expensive and stateful. Latency pushes
toward reading the newest value from memory, which is fast and, for training, wrong in a way that
inflates offline metrics and deflates online ones. The case's decision is therefore not "which store" but
**where the timestamp boundary lives and what is forbidden to cross it**.

The one-line decision: **treat the request timestamp as an argument, not as an ambient property.** Every
read carries an explicit `as_of` instant; the online path passes `now`, the training path passes
`label_time`; the serving layer never substitutes one for the other, and a logit computed with a
different `as_of` than the one recorded is a defect that fails a check rather than a judgement call.
Everything else — the cache, the fan-out, the tail, the storage plan — is downstream of that sentence.

## 2. Context & Background

The organisation has four models in production behind one ranking surface: a propensity model for
promotional placement, a churn-risk model for retention outreach, a fraud-score model for a checkout
path, and a relevance model for search ordering. All four consume entity features — user, item,
merchant, session — and all four are retrained on a schedule. They share a serving tier because
maintaining four serving tiers means maintaining four definitions of "user's 30-day order count", and
four definitions eventually become four different numbers.

The existing tier is a read-through cache over a batch-computed feature table refreshed nightly. That
design has three properties that look acceptable and are not. First, online and offline see different
data: the online path reads the newest nightly snapshot while the training pipeline reads a
point-in-time join, so the model is trained on a distribution it never meets. Second, the nightly refresh
means intraday features are unavailable, which forces every feature to be expressible as a daily
aggregate — a real constraint that quietly kills session-level signals. Third, the cache has no
coalescing, so a hot item under a traffic burst issues one origin read per request, which is the
mechanism that turns a good hit rate into an origin outage.

The migration context matters because the platform lead cannot start from a blank page. Four models are
live, their training pipelines are owned by three different teams, and two of those teams compute
features with SQL that has never been reviewed for temporal correctness. A design that requires every
consumer to change at once will not ship; a design that lets consumers keep their semantics while the
tier changes underneath will ship and will also preserve the bug. The lead must therefore choose which
consumers migrate first, and the criterion is the label delay: a model whose label arrives 30 days later
is far more exposed to leakage than one whose label is available in the same session.

Scale is a planning scenario, chosen to keep arithmetic checkable. The tier serves 250,000,000 requests
per day (ILLUSTRATIVE), which is a mean of 2,893 requests per second and a peak of 8,000 per second
(ILLUSTRATIVE, a 2.77 peak-to-mean ratio). Each request resolves 12 entity groups and reads about 240
feature values, so a peak second touches 96,000 entity lookups and 1,920,000 feature values. The entity
population is 2,000,000 users and 4,000,000 items, which is small enough that the working set can be
reasoned about exactly rather than sampled.

The latency commitment is a p99 of 50 milliseconds at the facade for the checkout model, because that is
what the payment flow can absorb without moving its own p99; the other three models have 120
milliseconds. There is a 0.1 percent error budget on 30 days, which sounds generous until it is
converted: 30 days is 43,200 minutes, so 0.1 percent is 43.2 minutes of complete unavailability — and
the interesting design question is how much of that budget the fallback path consumes without anyone
noticing.

Ownership is split three ways. The ML platform lead owns the serving tier and the latency SLO. Feature
owners — one per model family — own the definitions and the backfill semantics. A data-governance
group owns the retention and access rules, and it is the group that will object to serving a raw event
stream if the events carry personal data. Each of these three can break correctness alone: a feature
owner can redefine a window without changing the offline query, the platform lead can widen a TTL past a
label boundary, and governance can force a field to be dropped that a model silently depends on.

Prior incidents constrain the design. A churn model was retrained on a point-in-time join that used
`updated_at` rather than the event timestamp, so churned users' "days since last order" was computed from
the row's last modification — which for churned users was the churn event itself. The model learned that
a large value of that feature meant churn, because the feature was written by the churn process. Offline
AUC rose; online the model was useless. Separately, a relevance model's serving tier returned features
for a different item than the one requested on 0.02 percent of requests because a cache key omitted the
model version, so two versions shared an entry and one read the other's layout. Both incidents are
temporal-identity bugs, and both are invisible to every metric except a direct comparison.

**Takeaway:** four models, one shared tier, a nightly-refresh incumbent with three specific defects, a
2.77 peak-to-mean ratio, and a 43.2-minute monthly error budget.

## 3. The Business Question

| Field | Contract |
|---|---|
| Question | Should the tier move to explicit `as_of` serving with a point-in-time store, and which consumer migrates first? |
| Owner | ML platform lead (tier), feature owners (definitions), governance (retention) |
| Trigger | Any retrain where offline and online metric divergence exceeds 3 points, or a latency SLO breach |
| Actions | (a) keep nightly snapshot serving; (b) explicit `as_of` with a point-in-time store for all models; (c) explicit `as_of` for batch features only, keep snapshots for real-time features; (d) build a second tier for the checkout model only |
| Deadline | Before the next quarterly retrain, so the divergence is measurable on a comparable basis |
| Objective | Maximise agreement between offline training data and online served values at fixed latency SLO |
| Hard constraint | p99 ≤ 50 ms at the facade for checkout; ≤ 120 ms elsewhere; error budget 0.1 percent per 30 days |
| Soft constraint | Serving cost per million requests must not exceed 1.5× the incumbent |
| Scope | 250,000,000 requests/day, 12 entity groups per request, 240 feature values per request |
| Out of scope | Model architecture, training compute, feature discovery, label definition |

The four actions are not equally reversible, and that asymmetry is why the decision is staged. Option (a)
is the status quo and requires nothing. Option (d) builds a parallel tier for one model, which is fast to
deliver and creates a permanent second definition of every shared feature — the failure the shared tier
exists to prevent. Option (b) is the full migration and is the only one that eliminates the two
definitions. Option (c) is a hybrid that keeps the tier small but concedes that real-time features will
drift between training and serving, which is acceptable only if the drift is measured rather than
assumed.

The objective needs to be stated as agreement, not as accuracy, because accuracy is not measurable at
decision time. **Agreement** is the share of (entity, feature, `as_of`) triples where the value the
training pipeline materialised equals the value the serving tier returns for the same triple, allowing a
declared tolerance for floating-point and aggregation order. That metric is computable by replaying a
sample of logged requests through both paths, and it is the metric the lead commits to. Accuracy would
require labels; agreement requires only the two implementations, which exist.

Two secondary metrics constrain it. **Freshness** is the age distribution of served values relative to
the event time of the underlying fact; it is not the same as cache TTL, because a value can be cached
for 60 seconds while the fact it represents is three hours old. **Coverage** is the share of requested
features that resolve to a value rather than a default, and it interacts with agreement in a way that is
easy to hide: returning a default when the store is empty improves latency and degrades agreement, so
agreement must be computed over resolved values and coverage reported separately.

The question is falsifiable in a specific way. If, on a replay of one week of production requests, the
`as_of` path shows agreement at or above 0.999 while the snapshot path shows agreement at or above 0.995
for the same week, then the snapshot path is good enough for these features and option (a) stands. The
0.995 threshold is not arbitrary: it is the level at which a 3-point offline/online divergence is
unlikely, given the observed sensitivity in §6.5.

## 4. Assumptions & Constraints

- **A-01 — Every feature request can name an `as_of` instant.** Online passes request start time;
  training passes `label_time`. Verification: inspect the facade signature; an implicit `now` is a
  contract violation. If violated: no migration, because an implicit clock cannot be overridden.
- **A-02 — Feature values are keyed by (entity_type, entity_id, feature_name, event_time).** Verification:
  inspect the store schema and confirm `event_time` is part of the key rather than a payload column. If
  violated: point-in-time reads require a scan, and the latency target is unreachable.
- **A-03 — Label time is known and is not the label creation time.** Verification: compare `label_time`
  with the business fact time for a sample; the gap is the label delay. If unknown: treat the model as
  leakage-suspect and exclude it from the first migration wave.
- **A-04 — Entity cardinality is stable within a factor of two over the planning horizon.** Verification:
  weekly count of distinct entity IDs. If violated: the memory plan in §6.2 is invalid and the store
  shards must be re-cut.
- **A-05 — Per-entity feature bytes are bounded by the entity type.** Verification: p99 row size per
  entity type from the store. If a type exceeds its budget by 20 percent: drop the lowest-value features
  rather than increase memory, because memory growth is unbounded otherwise.
- **A-06 — Real-time features can tolerate a bounded staleness at serving time.** Verification: measure
  the age distribution, not the TTL (§6.3). If p99 age exceeds the model's retraining window: exclude
  that feature from real-time use.
- **A-07 — The training pipeline can be rewritten to consume the same API.** This is the assumption that
  most often fails, because offline teams have their own SQL and their own semantics. Verification:
  confirm the team can call the batch API for a backfill of one week. If not: agreement cannot be
  measured, and any migration claim is unfalsifiable.
- **A-08 — Governance permits storing raw events for the retention window.** Verification: written
  retention decision naming the fields. If not: the point-in-time store keeps only aggregates, and
  features requiring raw events are out of scope for wave one.

Constraints, stated numerically so compliance is checkable:

| ID | Constraint | Value | Why this value |
|---|---|---|---|
| C-01 | Checkout facade p99 | ≤ 50 ms | Payment flow p99 budget; measured end to end |
| C-02 | Other models p99 | ≤ 120 ms | Ranking surfaces can absorb a slower response |
| C-03 | Agreement (replay) | ≥ 0.999 | Below this, drift is indistinguishable from leakage |
| C-04 | Coverage | ≥ 0.995 for non-nullable features | Defaults silently change model inputs |
| C-05 | Freshness p99 (real-time features) | ≤ 30 s | Bounded by the shortest meaningful session signal |
| C-06 | Error budget | 0.1 percent per 30 days | 43.2 minutes; shared with all fallback paths |
| C-07 | Cost ratio vs incumbent | ≤ 1.5× per million requests | Single-digit-percent margin is not available here |
| C-08 | Entity row size p99 | ≤ 1,900 B user, 640 B item | Memory plan derives from these |

C-03 is the constraint that forces the architecture. A naive read-through cache scores high on latency
and low on agreement because it returns the newest value regardless of `as_of`; a naive point-in-time
scan scores high on agreement and fails C-01. The design has to satisfy both, which means two stores: one
optimised for the current value under a known staleness bound, one optimised for historical lookup at a
timestamp. Option (c) in §3 is precisely the compromise of using the first for everything.

C-05 is stated as an age, and the distinction from TTL is the single most useful clarification in the
case. A feature store with a 60-second TTL and a 30-second refresh produces ages between 0 and 90
seconds; a store with the same TTL and a 3-hour refresh produces ages between 0 and 3 hours. Reporting
"TTL 60s" as freshness is a category error that has concealed real drift in practice, and §6.3 computes
the difference explicitly.

**Takeaway:** eight assumptions with probes; eight numeric constraints; agreement ≥ 0.999 is the
constraint that forces a two-store design, and freshness must be measured as age, not as TTL.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is a **feature read**: one (entity_type, entity_id, feature_name, `as_of`) tuple.
Not a request, not a response, not a row. A request fans out into 12 entity groups × about 20 features,
so a peak second contains 1,920,000 feature reads, and agreement must be measured at this grain because
a request-level agreement figure hides the fan-out: eleven groups can agree while the twelfth, the one
that drives the score, is wrong.

| Field | Type | Units / domain | Available at decision | Notes |
|---|---|---|---|---|
| `request_id` | uuid | — | yes | One per facade call; joins to the response log |
| `as_of` | timestamp | UTC, ms | yes | Explicit argument; online = request start, training = label_time |
| `entity_type` | enum | user, item, merchant, session | yes | Determines the store shard and row budget |
| `entity_id` | string | — | yes | Stable; not the cache key alone (§12.4) |
| `feature_name` | string | versioned | yes | Version change is a new feature, not an edit |
| `feature_version` | int | — | yes | Must be part of the caching key |
| `event_time` | timestamp | UTC, ms | yes | When the fact occurred; the ordering key in the store |
| `value` | float / int / string | per definition | yes | With declared range and null policy |
| `refresh_time` | timestamp | UTC, ms | yes | When the value was materialised; drives age |
| `ttl_ms` | int | ms | yes | Serving admission only; never used for training |
| `source_dataset` | string | — | yes | Lineage pointer for the offline replay |

Deliberately **not** available at decision time, and the exclusions matter more than the inclusions:

- The request's outcome. Whether the ranking converted is decided after serving; using it to gate a
  feature read is leakage with a one-request delay.
- Any aggregate computed over a window that ends after `as_of`. This is the whole point of the case, and
  the enforcement is structural: the store's query interface rejects an end bound greater than `as_of`.
- The label, or any derivative of it, including a churn flag written back by the churn process. §2's
  first incident is exactly this.
- Another model's online score. Sharing a score across models creates a dependency graph that makes an
  agreement evaluation attributive in principle and unattributable in practice.
- The current cache state. Cache state is an optimisation artefact; recording it in the decision log
  would make a replay non-reproducible, because the replay's cache starts cold.

The cut-off is `as_of` on the *event* time axis, with an additional constraint on `refresh_time`: a value
is admitted only if `refresh_time <= as_of` for training reads. That second inequality is what stops a
nightly job from writing a value computed at 02:00 with an `event_time` of 01:00 into a training read at
01:30. Without it, a feature can be "point-in-time correct" on event time while being physically
impossible, which is the subtlest form of leakage in this domain.

Grain interaction with label delay: for a model whose label arrives 24 hours after the decision, the
training row at `label_time = t` may use features up to `t` and must not use anything between `t` and
`t + 24h`. For a model with a 30-day label delay, the forbidden interval is 30 days wide, and the
practical consequence is that features whose `refresh_time` lags by days are *acceptable* for the first
model and *unusable* for the second. Label delay, not feature type, is what segments the migration.

Units: latency in milliseconds at every level, because the SLO is at 50 ms and a seconds-based internal
unit would round the interesting region away. Memory in bytes with a stated overhead factor. Rates per
second for capacity and per day for budgets. Timestamps in UTC milliseconds — not microseconds, because
the source systems emit milliseconds and converting creates a class of off-by-1000 defects that are
invisible until a point-in-time join excludes the boundary row.

### 5b. Tools & Stack

Responsibilities, not products:

- **Feature registry:** the authoritative definitions — name, version, entity type, window, unit, null
  policy, owner, and the offline query that materialises it. The registry is the only place a definition
  lives; a definition in a notebook is a definition that will diverge.
- **Online store:** a sharded key-value store holding the current value per (entity, feature_version)
  with `refresh_time`. Optimised for single-key reads at low latency, with a row budget per entity type.
- **Point-in-time store:** an append-only store keyed by (entity, feature_version, event_time) supporting
  "value in force at `as_of`", i.e. the last row with `event_time <= as_of`. This is a range query on a
  sorted key, not a scan, which is why A-02 is a hard assumption.
- **Batch materialiser:** writes both stores from the same definition, so the two cannot diverge by
  construction. Any feature written to only one store is a feature with two implementations.
- **Request facade:** resolves entity IDs, fans out to shards with a per-request deadline, merges, and
  returns values plus an `as_of` echo. The deadline is per request, not per shard, which is what makes
  the tail analysis in §6.4 meaningful.
- **Cache:** per-shard, keyed by (entity_id, feature_version), with TTL and request coalescing. Coalescing
  is not an optimisation; without it a hot entity under a burst becomes an origin amplification.
- **Decision log:** one record per request with `as_of`, the feature versions resolved, per-shard latency,
  cache outcome, and any defaults substituted. This is the input to the replay that computes agreement.
- **Replay harness:** takes a sample of decision-log records, calls the batch API for the same triples,
  and reports agreement per feature, per entity type, and per consumer.

The registry-plus-two-stores shape is what makes the agreement metric computable. If online and offline
read different stores with different semantics, agreement is a comparison of two unrelated systems;
with this shape, agreement is a comparison of one definition evaluated twice, which is a much narrower
question and therefore a much more useful metric.

### 5c. Method Summary

Four steps, each producing an artifact.

1. **Decompose the request into group-level reads and compute the request-level miss probability** from
  the per-group hit rate. This is a small arithmetic step with a large consequence: a 0.93 per-group hit
  rate produces a 58.1 percent request-level miss share at 12 groups, which is not what "93 percent hit
  rate" suggests to anyone reading a dashboard. The distinction that matters for capacity is different:
  the expected number of origin group reads per request is `k(1−h) = 12 × 0.07 = 0.84`, so at 8,000
  requests/s the origin absorbs 6,720 group reads/s while 58.1 percent of requests — 4,651 per second —
  are exposed to at least one origin read and therefore to its latency.
2. **Size memory from the working set and the hit rate, using the sharing property of the access
  distribution.** The required cache fraction is `h^(1/(1−α))` for a Zipf(α) popularity curve; at
  α = 0.9 and h = 0.93 this is 48.4 percent of entities, not 93 percent.
3. **Compute the request-level tail from the group-level tail**, then decide whether hedging buys enough
  to spend the load. The p99 of a maximum of 12 independent group latencies is far worse than the group
  p99, and that is the number the SLO binds on.
4. **Quantify leakage as the product of affected rows and effect size**, then use the product to order the
  migration waves by label delay.

The method does not fit a model or estimate an effect from data, because the case has no data. It derives
consequences from stated assumptions and marks each consequence as conditional on its input. That is the
honest form for a design case: the arithmetic is exact, the inputs are illustrative, and the conclusion
is therefore a conditional statement that a reader can re-run with their own numbers.

### 5d. Provenance, Quality, Exclusions, Leakage

All inputs are constructed. The per-group hit rate 0.93, the Zipf exponent 0.9, the group-latency median
5 ms with p99 18 ms, and the label delay 24 hours are ILLUSTRATIVE planning values in plausible ranges,
not measurements. The 50 ms SLO is a construction chosen to make the tail arithmetic bind; a real
checkout path's budget is usually tighter and would make the same arithmetic bind harder.

Quality threats:

- **Independence across groups.** The tail computation assumes the 12 group latencies are independent.
  In reality they share a network path, a connection pool, and often a shard, so a network jitter event
  correlates them and the true request p99 is worse than the independent model predicts. Handling: §6.4
  computes both the independent case and the perfect-correlation case, and the design must survive the
  latter.
- **Stationary popularity.** The Zipf exponent is assumed constant. Trending items break stationarity
  exactly when a cache most needs to work, so the memory plan is a lower bound rather than an estimate.
- **The hit rate is treated as a property of the store.** It is a property of the *access distribution
  and the cache size together*; quoting a hit rate without the size is meaningless, which is why §6.2
  inverts the relationship.
- **Agreement is measured on a sample.** A sample of 10,000 requests per day at 250 million requests per
  day detects a 0.001 defect with high probability but cannot detect a defect concentrated in 0.01
  percent of entities. Handling: stratify the replay by entity popularity decile so that the head and
  tail are both represented.

Exclusions: model training, hyperparameter selection, feature selection, label construction, and the
ranking policy itself. The case is about the boundary between training and serving, and every exclusion
is chosen to keep that boundary the only moving part.

Leakage controls, stated as mechanisms rather than intentions:

1. The store's query API takes `as_of` as a required argument with no default. A default of `now` is a
   silent leak, and removing the default is the cheapest possible control.
2. The batch materialiser refuses to write a row whose `refresh_time` exceeds its `event_time` plus a
   declared materialisation lag; a violation is a hard failure, not a warning.
3. The replay harness compares `as_of` values rather than trusting the caller, so a training job that
   passes `now` produces a disagreement rather than a plausible score.
4. The registry requires each feature to declare its window and its availability lag, and a feature whose
   window exceeds its availability lag is flagged at registration — the structural version of "don't use
   data you wouldn't have had".

### 5e. Reproducibility & Method-Selection Ledger

Every number in §6 and §11 is derived from this table. Changing an input changes the results
proportionally or through a stated formula, and each row names which.

| Input | Symbol | Value | Unit | Status |
|---|---|---|---|---|
| Peak request rate | Q | 8,000 | requests/s | ILLUSTRATIVE |
| Daily requests | D | 250,000,000 | requests/day | ILLUSTRATIVE |
| Entity groups per request | k | 12 | groups/request | ILLUSTRATIVE |
| Feature values per request | n | 240 | values/request | ILLUSTRATIVE |
| Per-group hit rate | h | 0.93 | fraction | ILLUSTRATIVE |
| Popularity exponent | α | 0.9 | dimensionless | ILLUSTRATIVE |
| User entities | U | 2,000,000 | entities | ILLUSTRATIVE |
| Item entities | I | 4,000,000 | entities | ILLUSTRATIVE |
| User entity row size p99 | r_u | 1,900 | bytes | ILLUSTRATIVE |
| Item entity row size p99 | r_i | 640 | bytes | ILLUSTRATIVE |
| Cache overhead factor | φ | 1.45 | dimensionless | ILLUSTRATIVE |
| Group latency median | M | 5 | ms | ILLUSTRATIVE |
| Group latency p99 | P99_g | 18 | ms | ILLUSTRATIVE |
| Label delay | L | 24 | hours | ILLUSTRATIVE |
| Training rows | R | 40,000,000 | rows | ILLUSTRATIVE |
| Refresh interval (real-time) | T_r | 30 | s | ILLUSTRATIVE |
| Batch TTL | T_b | 60 | s | ILLUSTRATIVE |
| Error budget | β | 0.001 | fraction/30 days | DERIVED (0.1 percent) |
| Daily active model calls | — | 4 | models | ILLUSTRATIVE |

Method selection:

| Choice made | Alternative rejected | Reason |
|---|---|---|
| Explicit `as_of` argument | Implicit `now` | An implicit clock cannot be overridden by a training replay |
| Two stores, one definition | Snapshot for everything | A snapshot cannot answer a historical query without a separate history |
| Per-request deadline | Per-shard deadline | Per-shard deadlines sum to 12× the budget under partial failure |
| Request-level p99 | Group-level p99 | A request fails if any of 12 groups is slow; group p99 is not the SLO |
| Zipf inversion for memory | Hit-rate × entity count | That formula assumes uniform popularity and over-provisions by 2× |
| Agreement on replay | Offline AUC | AUC cannot distinguish leakage from signal; agreement can |
| Wave ordering by label delay | Wave ordering by volume | Delay determines leakage exposure; volume determines cost, which is recoverable |
| Coalescing in the cache | Request collapse at the facade | Facade-level collapse misses the shard-level hot key |

**Takeaway:** the grain is one feature read with an explicit `as_of`; request-level miss share,
Zipf-inverted memory, request-level tail, and leakage product are the four quantities the method
computes, and each is conditional on one labelled input.

## 6. Analysis Walkthrough

Ten derivations. Each has the five-part anatomy, and the sixth one is the boundary case where the
previous formula stops being valid. All inputs are from §5e.

### 6.1 Per-group hit rate is not request hit rate

```
FORMULA   P_all_hit = h^k
          P_req_miss = 1 - h^k
          origin_reads_per_request = k * (1 - h)
          origin_reads_per_s = Q * k * (1 - h)

WHERE     h = per-group cache hit rate (fraction)
          k = entity groups resolved per request (groups/request)
          Q = peak request rate (requests/s)
          P_all_hit assumes independence across the k groups, which holds when the
          groups are distinct entities and fails when they share a shard (see 6.10).

CONSTRAINT  The two quantities answer different questions and must not be
          interchanged: P_req_miss sets the share of requests exposed to origin
          latency (a tail question), while origin_reads_per_s sets origin load
          (a capacity question). A design that optimises only the second can still
          breach the latency SLO.

NUMERICAL EXAMPLE  h = 0.93, k = 12, Q = 8,000 requests/s
          h^k = 0.93^12 = 0.418596
          P_req_miss = 1 - 0.418596 = 0.581404  (58.1 percent of requests)
          origin_reads_per_request = 12 * 0.07 = 0.84 group reads/request
          origin_reads_per_s = 8,000 * 0.84 = 6,720 group reads/s
          requests exposed per s = 8,000 * 0.5814 = 4,651 requests/s
          per-group origin load = 8,000 * 0.07 = 560 reads/s per group

INTERPRETATION  A dashboard reading "93 percent hit rate" invites the conclusion that
          7 percent of traffic touches the origin. In fact 58.1 percent of requests
          touch it at least once, because a request is a conjunction of 12
          independent coin flips. The origin's *load* is 6,720 reads/s, which is only
          7 percent of the 96,000 group lookups the tier performs, so the origin is
          lightly loaded and heavily latency-exposed — the opposite of what "7 percent
          miss rate" implies. Capacity planning for the origin and latency planning
          for the facade therefore read different numbers from the same tier.
```

That distinction has an operational consequence worth stating: adding origin capacity will not improve
the p99, because the origin is not saturated. Raising the per-group hit rate will, and the amount of
memory required to do so is the subject of the next derivation.

### 6.2 Memory required for a hit rate, under a skewed access distribution

```
FORMULA   For a Zipf(alpha) popularity distribution and LRU with cache fraction c,
          the hit rate is h(c) = c^(1 - alpha)   (alpha != 1)
          Inverting: c_required = h^(1 / (1 - alpha))
          bytes = c_required * sum_e (N_e * r_e) * phi

WHERE     alpha = Zipf exponent of the popularity curve (dimensionless)
          c = fraction of distinct entities resident in the cache (0 < c < 1)
          N_e = entity count for entity type e (entities)
          r_e = p99 row size for entity type e (bytes/entity)
          phi = overhead factor for allocator, indexes, replicas (dimensionless)

CONSTRAINT  Valid for alpha < 1 and c small relative to 1; the approximation breaks
          when alpha >= 1 (the integral diverges) and when the cache is large enough
          to hold everything, where h tends to 1 faster than the formula predicts.

NUMERICAL EXAMPLE  alpha = 0.9, so 1 - alpha = 0.1 and the exponent is 10
          c_required = 0.93^10 = 0.483982  (48.4 percent of distinct entities)
          users: 0.483982 * 2,000,000 * 1,900 B = 1,839,132,767 B = 1.839 GB
          items: 0.483982 * 4,000,000 *   640 B = 1,238,994,706 B = 1.239 GB
          total payload = 3.078 GB
          with phi = 1.45: 4.463 GB = 4.157 GiB
          naive alternative: 0.93 * (2,000,000*1,900 + 4,000,000*640) = 5.915 GB
            payload, * 1.45 = 8.576 GB = 7.987 GiB, i.e. 1.92x the correct plan

INTERPRETATION  The naive formula treats 93 percent as 93 percent of entities, which
          over-provisions by 1.92x. The correct reading is that 48.4 percent of
          entities serve 93 percent of references, because the top decile of a
          Zipf(0.9) curve carries 79.4 percent of requests (0.1^0.1 = 0.7943).
          The 4.16 GiB plan is affordable; the 7.99 GiB plan is affordable too, but
          it is affordable for the wrong reason and will be mis-sized again at the
          next capacity review when the entity count doubles.
```

Sensitivity worth carrying: raising h from 0.93 to 0.97 costs 1.52× the memory
(`0.97^10 / 0.93^10 = 0.7374 / 0.4840 = 1.524`, so 4.69 GB of payload and 6.80 GB with overhead = 6.34
GiB, a gain of 2.18 GiB over the 4.16 GiB plan) and cuts origin group reads from 6,720/s to 2,880/s — a
57.1 percent reduction. The trade is 2.18 GiB of memory for 3,840 origin reads/s, which is a good trade if
the origin is the constraint and a wasteful one if it is not. §6.1 shows the origin is lightly loaded, so
the honest conclusion is that 0.93 is sufficient and the extra memory buys headroom rather than
correctness.

### 6.3 Freshness is an age distribution, not a TTL

```
FORMULA   age = as_of - fact_time
          For batch features with uniform refresh phase and TTL T_b:
            P(age > u) = u / T_b              for 0 <= u <= T_b
          For real-time features with refresh interval I ~ F:
            P(age > u) = P(I > u)
          P(served_stale) = w_b * P(age > u | batch) + w_r * P(age > u | real-time)

WHERE     T_b = batch TTL (s), u = the freshness bound being tested (s)
          w_b, w_r = share of served values from batch and real-time paths
          I = refresh interval (s), modelled lognormal with median m_I and p99 I_99

CONSTRAINT  The TTL is an admission bound, not an age bound. Age includes the
          refresh interval and the materialisation lag, so age_max = T_b + I_max,
          and no TTL setting can deliver a bound tighter than the refresh interval.

NUMERICAL EXAMPLE  T_b = 60 s, u = 30 s, w_b = 0.93, w_r = 0.07
          batch: P(age > 30) = 30/60 = 0.5
          real-time: I with median 12 s and p99 45 s
            sigma_I = ln(45/12) / z_0.99 = 1.3218 / 2.3263 = 0.5682
            P(I > 30) = 1 - Phi( (ln 30 - ln 12) / 0.5682 )
                      = 1 - Phi(0.9163 / 0.5682) = 1 - Phi(1.6126) = 0.0534
          P(served_stale) = 0.93*0.5 + 0.07*0.0534 = 0.465 + 0.0037 = 0.4687
          max age = 60 + 45 = 105 s; mean age = 30 + 12 = 42 s

INTERPRETATION  A tier advertising "TTL 60 s, refresh 30 s" serves values older than
          30 s for 46.9 percent of reads, because a batch value up to 60 s old is
          admitted by the TTL and half of a uniform phase exceeds 30 s. The
          real-time path contributes only 0.37 percentage points of that. So the
          freshness problem is entirely a batch-path problem, and the fix is a
          shorter batch refresh — not a faster real-time stream.
          Fixing the batch refresh to 30 s (TTL 30 s) makes batch age uniform on
          0..30, so P(batch age > 30) = 0 at the boundary and
          P(served_stale) = 0.93*0 + 0.07*0.0534 = 0.0037, a 125.4x reduction
          (0.4687 / 0.0037). Read volume is unchanged; the cost is halving the
          materialisation pass interval, i.e. 2x writer CPU on that path.
```

That 125× figure is the reason the freshness metric is reported as an age histogram rather than as a
TTL. The histogram makes the batch path's dominance visible; the TTL makes it invisible, because both
paths are described by the same 60-second number that neither path actually delivers.

### 6.4 The request tail is not the group tail

```
FORMULA   P(request_latency <= t) = F(t)^k        (independent groups)
          Solve F(t_p) = (1 - p)^(1/k) for the request-level quantile p
          Perfect correlation: P(request_latency <= t) = F(t)

WHERE     F = CDF of a single group's latency
          k = groups per request
          t_p = request latency at quantile p (ms)
          The lognormal fit uses median M and p99 P99_g:
            sigma = ln(P99_g / M) / z_0.99

CONSTRAINT  Independence is the optimistic assumption. Shared network, shared
          connections, and shared shards correlate group latencies, so the true
          request tail lies between the independent curve and the perfectly
          correlated curve; the design must satisfy the SLO under the worse of the
          two, which is the independent curve at high quantiles.

NUMERICAL EXAMPLE  M = 5 ms, P99_g = 18 ms, k = 12
          sigma = ln(18/5) / 2.3263 = 1.2809 / 2.3263 = 0.5506
          Check the group p99: 5 * exp(2.3263*0.5506) = 18.0 ms ✓
          P(all 12 groups <= 18 ms) = 0.99^12 = 0.8864, so the group p99 is only the
            request p88.6 — the request p99 is a different, larger number.
          For the request p99: F(t) = 0.99^(1/12) = 0.999164
            z = Phi^-1(0.999164) = 3.1440
            t_99 = 5 * exp(3.1440 * 0.5506) = 5 * 5.6471 = 28.24 ms
          Request p999: F(t) = 0.999^(1/12) = 0.9999166, z = 3.7558
            t_999 = 5 * exp(3.7558*0.5506) = 39.74 ms

INTERPRETATION  A 12-way fan-out multiplies the tail: 18 ms at group p99 becomes
          28.24 ms at request p99, a 57 percent increase from fan-out alone. Against
          a 50 ms SLO that leaves 21.8 ms for gateway, serialisation, and merge.
          The same arithmetic at the request p999 gives 39.74 ms, which leaves
          10.3 ms — so the SLO is satisfied at p99 with room and at p999 barely.
          This is why the deadline must be per request with a budget split, rather
          than per shard: a per-shard 18 ms deadline lets one slow shard consume
          36 percent of the request budget.
```

Hedging is the standard remedy and its cost is computable. Sending a duplicate request to a second
replica when the first has not answered by the group p95 (12.37 ms) adds 5 percent load and changes the
request p99 to the p99 of a minimum of two draws: `(1-F(t))^2 = 0.01` gives `F(t) = 0.9`, so
`t = 5*exp(1.2816*0.5506) = 10.13 ms`. The request p99 falls from 28.24 ms to about 10.13 ms for 5
percent more group load — a good trade when the SLO is tight and a bad one when the origin is already the
constraint (§6.1 says it is not).

### 6.5 Leakage as a product: affected rows × effect size

```
FORMULA   leaked_rows = R * s_leak
          positive_share_in_leaked = (R * s_leak * p_leak) / (R * p)
          exposure = leaked_rows * delta_metric

WHERE     R = training rows (rows)
          s_leak = share of rows containing a value computed after label_time
          p = overall positive rate; p_leak = positive rate among leaked rows
          delta_metric = metric change attributable to the leak (metric units)
          exposure has units of "metric-points × rows", which is the right way to
          weight a defect that is invisible in a row count but visible in a metric.

CONSTRAINT  p_leak > p is the mechanism. If the leak affects positives and negatives
          equally it cannot inflate a ranking metric, because it adds the same
          information to both classes. The dangerous case is a leak written by the
          outcome process itself, where p_leak >> p by construction.

NUMERICAL EXAMPLE  R = 40,000,000, s_leak = 0.22, p = 0.04, p_leak = 0.15
          leaked_rows = 40,000,000 * 0.22 = 8,800,000 rows
          positives total = 40,000,000 * 0.04 = 1,600,000
          positives inside leaked rows = 8,800,000 * 0.15 = 1,320,000
          positive share in leaked rows = 1,320,000 / 1,600,000 = 0.825
          rows-to-positives ratio = 8,800,000 / 1,320,000 = 6.67 rows per leaked positive
          exposure at delta = 0.06 AUC: 8,800,000 * 0.06 = 528,000 AUC-point-rows

INTERPRETATION  22 percent of rows contain 82.5 percent of the positives. That is
          the whole mechanism: a feature written by the churn process is
          near-perfectly informative about churn, so contaminated rows are
          overwhelmingly positive and AUC rises from about 0.72 to about 0.78.
          The row share makes the defect look minor in a data-quality report
          ("22 percent null-ish"), while the positive share makes it decisive in a
          model. Any leakage assessment that reports the row share alone is
          reporting the wrong denominator.
```

This is also the answer to a common review objection: "only a fifth of the rows are affected, so the
effect must be small." The effect on a ranking metric is governed by the affected positives, and a leak
correlated with the label concentrates there. §12.1 gives the check that detects it from the store side
rather than from a metric.

### 6.6 Where the independence assumption breaks: heterogeneous consumers

```
FORMULA   P_all_hit = sum_j w_j * h_j^k
          h_mean = sum_j w_j * h_j
          modelled_P_all_hit = h_mean^k

WHERE     w_j = share of traffic from consumer (or entity class) j
          h_j = per-group hit rate experienced by consumer j
          The mixture formula is exact under independence within each consumer and
          is the boundary case for 6.1's single-h formula.

CONSTRAINT  h_mean^k is a *lower* bound on the mixture, and the gap grows with the
          variance of h_j. Using it therefore over-estimates request misses, which
          is safe for capacity and unsafe for latency: it can hide a small
          subpopulation with a very low hit rate whose requests are almost always
          slow.

NUMERICAL EXAMPLE  two consumers: w_1 = 0.8 with h_1 = 0.96; w_2 = 0.2 with h_2 = 0.60
          h_1^12 = 0.96^12 = 0.612710
          h_2^12 = 0.60^12 = 0.002177
          P_all_hit = 0.8*0.612710 + 0.2*0.002177 = 0.490168 + 0.000435 = 0.490603
          P_req_miss = 0.509397 (50.9 percent)
          h_mean = 0.8*0.96 + 0.2*0.60 = 0.888
          modelled P_all_hit = 0.888^12 = 0.240412 -> modelled miss 0.759588
          over-prediction = 0.759588 / 0.509397 = 1.491x

INTERPRETATION  The mean-based model says 76 percent of requests miss; the truth is
          50.9 percent. So far the aggregate model is conservative. The dangerous
          part is the invisible subpopulation: consumer 2 has a 0.4 percent all-hit
          rate, meaning 99.6 percent of its requests touch the origin, and it is
          20 percent of traffic. In aggregate that looks like "the tier works";
          per consumer it is an outage that the SLO reports as a p99 breach with no
          obvious cause. The correction is to report P_req_miss per consumer and
          per entity class, never only in aggregate.
```

This is the boundary where §6.1's formula breaks, and it breaks in the direction that hides an incident.
The general lesson: any metric computed as `f(mean)` rather than `mean(f)` is wrong by a factor that
depends on dispersion, and the dispersion is exactly where operational problems live.

### 6.7 Error budget consumption by the fallback path

```
FORMULA   budget_minutes = beta * days * 1,440
          fallback_requests = D * s_fb
          effective_errors = D * s_fb * s_flip
          budget_requests = D * beta

WHERE     beta = error budget (fraction per period)
          D = daily requests (requests/day)
          s_fb = share of requests served by the fallback (defaults, not errors)
          s_flip = share of fallback-served requests whose decision changes
          A fallback request counts against the budget only if it changes the
          output materially; counting every fallback would exhaust the budget on
          behaviour that is invisible to users.

CONSTRAINT  effective_errors <= budget_requests, and the fallback share must be
          bounded by a circuit breaker, because s_fb is unbounded during a store
          outage while s_flip is not.

NUMERICAL EXAMPLE  beta = 0.001, 30 days = 43,200 minutes, so budget = 43.2 minutes
          D = 250,000,000 requests/day
          budget_requests = 250,000 * 0.001 = 250,000 requests/day
          s_fb = 0.005, s_flip = 0.20
          fallback_requests = 1,250,000
          effective_errors = 1,250,000 * 0.20 = 250,000
          ratio to budget = 250,000 / 250,000 = 1.00

INTERPRETATION  A 0.5 percent fallback rate with a 20 percent decision-flip rate
          consumes exactly 100 percent of the error budget, with zero margin, and
          the fallback is *not an outage*: no request failed, no alert fired, and
          every response was 200 OK. This is the most under-appreciated accounting
          fact in serving tiers — the fallback path is the error budget's largest
          silent consumer. The design consequence is a fallback ceiling of 0.2
          percent, so that a full fallback episode consumes 40 percent of budget and
          leaves room for an actual incident in the same month.
```

### 6.8 Cost per million requests, cache versus origin

```
FORMULA   cpu_origin = Q * k * (1 - h) * c_o
          cpu_cache  = Q * k * h * c_c
          cost_day   = (cpu_origin + cpu_cache) * price_vcpu_hour * 24
          cost_per_M = cost_day / (D / 1,000,000)

WHERE     c_o = CPU seconds per origin read (s/read)
          c_c = CPU seconds per cache read (s/read)
          price_vcpu_hour = price of one vCPU-hour (USD)
          Q*k*(1-h) is the origin read rate from 6.1; Q*k*h is the cache read rate.

CONSTRAINT  cost_per_M must not exceed 1.5x the incumbent (C-07), and the ratio is
          dominated by c_o/c_c, not by the request count. Raising h trades origin
          CPU for cache CPU at a ratio of c_o/c_c = 10, so it is nearly free.

NUMERICAL EXAMPLE  Q = 8,000, k = 12, h = 0.93, c_o = 0.0015 s, c_c = 0.00015 s,
          price = 0.12 USD/vCPU-hour
          group reads/s = 96,000
          origin reads/s = 6,720 -> cpu_origin = 6,720 * 0.0015 = 10.08 vCPU
          cache reads/s = 89,280 -> cpu_cache = 89,280 * 0.00015 = 13.39 vCPU
          total = 23.47 vCPU
          cost_day = 23.47 * 0.12 * 24 = 67.59 USD/day
          cost_per_M = 67.59 / 250 = 0.2704 USD per million requests

INTERPRETATION  The cache costs more CPU than the origin it protects (13.39 vs 10.08
          vCPU), because every request is a cache read while only 7 percent reach
          the origin. That flips the usual intuition that caching saves compute: at a
          10x per-read CPU ratio and a 93 percent hit rate, the ratio of cache CPU to
          origin CPU is h/((1-h)*10) = 0.93/0.70 = 1.33, so caching *adds* 33 percent
          more CPU than it removes. What caching buys is latency, not cost — and the
          cost model must say so or a reviewer will approve a redesign on a false
          premise.
```

### 6.9 The refresh_time inequality and the physically impossible read

```
FORMULA   admissible(as_of) = { rows : event_time <= as_of AND refresh_time <= as_of }
          contaminated_share = min(1, backfill_duration / read_span)

WHERE     event_time = when the fact occurred
          refresh_time = when the row was written to the store
          backfill_duration = wall-clock time for the materialiser to finish a pass
          read_span = interval during which training reads use that pass

CONSTRAINT  Both inequalities are required. A store that enforces only event_time
          admits a row whose value was computed after the read, which is not merely
          leakage: it is causality violation, and it is undetectable from the value.

NUMERICAL EXAMPLE  materialiser pass writes 2,000,000 user rows from 02:00 to 03:00
          a training read at 02:30 has as_of = 02:30
          rows written after 02:30 = 50 percent of the pass
          contaminated_share = 1.00 * 0.5 ... precisely, the share of that pass
            visible at 02:30 is 0.5, so 0.5 of the pass's rows are ahead of the read
            and 1,000,000 rows must be excluded; a store enforcing only event_time
            includes them.
          over a 30-day feature window, the forbidden interval is L = 24 h, so the
            overlap fraction is 24 / (30*24) = 0.0333 (3.33 percent of the window)
          a single pass therefore contaminates 3.33 percent of a 30-day aggregate's
            inputs, which is small per row and decisive for a monotone window
            feature such as "days since last order"

INTERPRETATION  The 50 percent figure is the honest measure of exposure during a
          backfill, and it is why point-in-time stores separate the two keys: a
          read-time filter on refresh_time is cheap, and a read-time filter on
          event_time alone systematically admits exactly the rows that were written
          by the process whose outcome the model is predicting.
```

### 6.10 Coalescing: one hot key, 400 requests per second

```
FORMULA   concurrent_fetches = lambda_key * W_fetch
          origin_reads_without_coalescing = lambda_key
          origin_reads_with_coalescing = lambda_key / concurrent_fetches
          reduction = concurrent_fetches

WHERE     lambda_key = request rate for a single cache key (requests/s)
          W_fetch = origin fetch latency (s)
          Under coalescing, all in-flight requests for a key share one fetch, so the
          fetch rate is bounded by 1/W_fetch rather than by lambda_key.

CONSTRAINT  Coalescing requires a per-key single-flight structure; a global lock
          would serialise unrelated keys and add its own queueing delay. It also
          requires a bounded wait, because a follower must not wait longer than the
          request deadline.

NUMERICAL EXAMPLE  lambda_key = 400 requests/s, W_fetch = 0.012 s
          concurrent_fetches = 400 * 0.012 = 4.8
          without coalescing: 400 origin reads/s for one key
          with coalescing: 400 / 4.8 = 83.3 origin reads/s
          reduction = 4.8x, and the origin sees 1/0.012 = 83.3 reads/s ceiling per key
          at 8,000 total requests/s, a single key at 5 percent of traffic is
            400 requests/s; twenty such keys would put 8,000 reads/s on the origin
            without coalescing versus 1,667 with it.

INTERPRETATION  Coalescing converts a per-request origin read into a per-fetch-interval
          origin read, and the ratio is exactly the number of requests that arrive
          during one fetch. The failure mode it prevents is not a cold cache but a
          hot key: a trending item at 400 requests/s against a 12 ms origin turns a
          93 percent hit rate into 400 origin reads/s on one shard, which is the
          classic cache stampede and is invisible in the aggregate hit rate because
          the aggregate is computed over all keys.
```

**Takeaway:** 12-way fan-out turns a 0.93 group hit rate into 58.1 percent of requests touching the
origin; a Zipf(0.9) inversion needs 48.4 percent of entities for that hit rate, not 93 percent;
freshness is an age histogram where the batch path contributes 99.2 percent of staleness; the request
p99 is 28.24 ms from an 18 ms group p99; and homogeneous hit-rate models hide a 20-percent subpopulation
whose all-hit rate is 0.4 percent.

## 7. The Finding

**Finding 1 — Agreement, not latency, is the binding constraint, and it is decided by the request
signature.** A read-through cache can meet any latency target by returning the newest value, which is
exactly the wrong answer for training. The single change that fixes this is making `as_of` a required
parameter with no default; with it, an incorrect caller fails a comparison instead of producing a
plausible score. Without it, no amount of store sophistication helps, because the training job that
passes `now` cannot be distinguished from one that passes `label_time`.

**Finding 2 — The request-level miss share is an order of magnitude larger than the group-level miss
rate, and the two numbers drive different plans.** At 0.93 per group across 12 groups, 58.1 percent of
requests touch the origin while only 0.84 group reads per request hit it. So the origin is at 7 percent
of its lookup capacity while more than half of all requests are exposed to origin latency (§6.1). Any
capacity review that reads the hit rate and concludes "the origin is fine" is asking a capacity question
and answering a latency question.

**Finding 3 — Memory sized from a hit rate under a skew assumption is 1.92× smaller than memory sized
from the same hit rate treated as a uniform share of entities.** The Zipf inversion gives 48.4 percent of
entities for a 0.93 hit rate; the naive formula gives 93 percent. The correct answer depends on the
popularity exponent, which means the memory plan must state α and must be re-derived when the traffic mix
changes (§6.2). A plan without α is not a plan.

**Finding 4 — Freshness must be reported as age, because TTL is compatible with any age.** A 60-second
TTL with a 30-second refresh delivers 46.9 percent of reads older than 30 seconds; the real-time path
contributes 0.37 points of that. Tightening the batch refresh to 30 seconds reduces staleness 125× and
costs only writer CPU (§6.3). A freshness dashboard built on TTL would have shown a healthy 60 seconds
throughout.

**Finding 5 — The error budget is consumed mostly by the fallback path, silently.** A 0.5 percent
fallback rate with a 20 percent decision-flip rate consumes 100 percent of the monthly budget while
every response is a 200 OK and no alert fires (§6.7). The budget therefore needs a fallback ceiling as a
first-class guardrail, because the visible failure modes are bounded and the invisible one is not.

**Finding 6 — The 12-way fan-out costs 57 percent of the latency budget before any component is slow.**
An 18 ms group p99 becomes a 28.24 ms request p99 by arithmetic alone, leaving 21.8 ms of a 50 ms SLO for
gateway, serialisation, and merge (§6.4). This is why the facade needs a per-request deadline with a
budget split and not a per-shard timeout: twelve independent 18 ms timeouts compose into an unbounded
request.

The findings share a structure worth naming, because it generalises beyond feature serving. Each is a
case where a **mean or a single value hides a distribution that the decision actually depends on**: mean
hit rate hides the per-group conjunction, mean popularity hides the head, TTL hides the age distribution,
mean latency hides the maximum over fan-out, mean fallback rate hides the flip rate. Serving tiers
concentrate these because they are almost always fan-out systems with a latency contract, and fan-out
converts any per-item distribution into a worst-case statistic.

What the findings do not establish is that the incumbent tier is unusable. If the four models' features
are all daily aggregates and their labels are all 30 days delayed, then the snapshot path's agreement may
be high and the migration is not worth its risk. That case is specific and testable, and §9 states the
test.

## 8. System/Solution Design

### 8.1 Two stores, one definition

```
registry (definitions: name, version, entity, window, lag, owner, offline query)
   |
   +-- batch materialiser (single writer, writes BOTH stores)
         |-- online store:  key (entity_id, feature_version)            -> value, refresh_time
         |-- PIT store:     key (entity_id, feature_version, event_time) -> value
   |
facade (as_of required)
   |-- resolve entity ids
   |-- per-group: cache lookup -> online store -> PIT store (only when as_of < now - grace)
   |-- per-request deadline + budget split + hedge policy
   |-- merge, echo as_of, log decision record
```

The materialiser being a single writer for both stores is the load-bearing detail. If two code paths
write the two stores, they will disagree, and the disagreement will be discovered as an agreement-metric
drop with no attributable cause. One writer means the only way the stores differ is by a read-time
filter, which is a small and enumerable set of causes.

### 8.2 Read path decision table

| Condition | Path | Latency budget | Why |
|---|---|---|---|
| `as_of` within 5 s of now, value resident | cache | 2 ms | Online traffic, hot entity |
| `as_of` within 5 s of now, value absent | online store | 8 ms | Current value is the right answer |
| `as_of` older than 5 s, feature is append-only | PIT store, key seek | 15 ms | History required |
| `as_of` older than 5 s, feature is a daily aggregate | PIT store at day boundary | 15 ms | Aggregate is immutable after the day closes |
| `as_of` in the future | reject, 400 | — | Future reads are a caller bug, not a default |
| store unavailable | default value, counted as fallback | 1 ms | Only admissible with the ceiling in §6.7 |
| PIT row missing, feature non-nullable | fail the request | — | A default would silently change the model input |

The last row is the one that draws objections and is the one worth defending. A non-nullable feature with
no PIT row means the caller asked for a moment before the feature existed; returning a default would make
the training row look valid while silently substituting a value with different statistics. Failing the
request makes the defect visible at the caller, which is where it can be fixed. Nullable features — those
whose definition includes "no history" as a real category — take the default and record it.

The three-second grace before switching from online to PIT is a design choice with a consequence: a read
within the grace window can be served from the online store even though the online value's `refresh_time`
may exceed `as_of`. That is acceptable for online traffic (where `as_of = now` by construction) and
unacceptable for training, so the grace is keyed on the caller identity: the training caller sets
`strict_pit = true` and the grace is zero. Making the caller declare its intent is cheaper than inferring
it from latency.

### 8.3 Cache key and version discipline

The cache key is `(entity_id, feature_version, shard_hint)`. Three fields, each load-bearing:

- `entity_id` alone was the incumbent's key and produced §2's second incident, where two model versions
  read each other's entry because a version-dependent feature changed its byte layout.
- `feature_version` is a monotonically increasing integer assigned by the registry. A definition change
  is a new version, never an edit, because an edit invalidates every cached value silently and makes a
  replay non-reproducible.
- `shard_hint` is included because two entity types can share an ID space; without it, an item and a user
  with the same numeric ID collide. This is cheap to fix at key construction and expensive to find in
  production, because the symptom is a small percentage of inexplicable values rather than an error.

The cache stores the value with its `refresh_time` so that a served value can report its age. A cache
entry without its timestamp cannot support §6.3's metric, and a cache that cannot report age will end up
having its TTL quoted as freshness by the operations dashboard.

### 8.4 Deadline and budget split

The facade has one deadline per request, derived from the SLO: 50 ms for checkout, 120 ms for others. It
splits that budget as follows, in milliseconds, for the 50 ms case:

| Stage | Budget | Rationale |
|---|---|---|
| entity resolution (cached mapping) | 3 | Usually a hit; a miss falls back to direct ID |
| group fan-out (parallel, all 12) | 30 | §6.4's request p99 is 28.24 ms at group p99 18 ms |
| merge and serialise | 6 | Fixed cost, measured at p99 |
| network in/out allowance | 8 | Client to facade, includes one retransmit |
| reserve | 3 | Absorbs one stage exceeding its own budget |
| total | 50 | Sum equals the SLO, not 50 plus slack |

The budget is allocated so that the sum equals the SLO rather than exceeding it, which is the discipline
most implementations miss: budgets are usually stated per stage and never summed, so the sum silently
exceeds the contract by 20 to 40 percent and the reserve is negative in practice. Writing the sum in the
table makes the violation visible at review time.

The fan-out budget is a deadline, not a timeout: when the deadline passes, the facade returns whatever
groups have answered plus defaults for the rest, records the substitution, and reports the request as
degraded. A timeout would fail the request; a deadline degrades it, which is the correct behaviour when
a default is admissible and a failure is not. The distinction is why §8.2's last row matters — for
non-nullable features the deadline must fail rather than degrade.

### 8.5 Hedging policy

Hedging is enabled only where the SLO binds and only under a load ceiling:

- Trigger at the group p95 (12.37 ms for the fitted distribution), not at the p99, because the p99 is
  already past the point where a hedge can rescue the request.
- Cancel the straggler on first success; keep the connection for reuse.
- Cap hedged requests at 3 percent of group traffic globally and 10 percent per shard, so that hedging
  cannot amplify a slow-shard problem into an origin overload. §6.4 computes 5 percent load at a p95
  trigger; the 3 percent cap is the bound the origin plan tolerates.
- Disable hedging entirely when the origin's queue depth exceeds its configured threshold, because
  hedging assumes the second replica is not slow for the same reason as the first, and during a
  correlated slowdown it is.

The cancellation detail is where hedging implementations leak capacity: a hedge that is not cancelled
leaves an in-flight origin read that completes and occupies a slot, so the load multiplier becomes the
retry rate rather than the trigger rate. Recording both the trigger count and the cancelled count in the
decision log makes the difference auditable.

### 8.6 What is deliberately not built

- **No cross-model score sharing.** A shared score creates a dependency that makes an agreement failure
  unattributable, and it makes a retrain of one model depend on the release of another.
- **No online feature computation in the facade.** Computing a feature at request time makes the training
  pipeline's job impossible to replicate, because the offline implementation will differ. Real-time
  features come from a stream that writes both stores, so the definition stays single.
- **No automatic backfill on schema change.** A backfill rewrites history, and a point-in-time store whose
  history is rewritten cannot be used for a replay. Schema changes create a new feature version with a
  start date and leave old rows untouched.
- **No per-request cache bypass flag for training.** A training replay uses the batch API, never the
  facade, so that a cache cannot influence an agreement measurement.

**Takeaway:** one writer for two stores, `as_of` required with no default, a three-field cache key,
a budget that sums exactly to the SLO, and a hedge that is capped and cancelled.

## 9. Evaluation & Validation

### 9.1 Metrics with all six attributes

| Metric | Numerator | Denominator | Population | Window | Comparator | Guardrail |
|---|---|---|---|---|---|---|
| Agreement | feature reads where served value = batch value within tolerance | feature reads resolved on both paths | replay sample, stratified by popularity decile | daily, 7-day trailing | ≥ 0.999 (C-03) | If < 0.999: block the next retrain |
| Coverage | requested features resolving to a value | requested features | all consumers | daily | ≥ 0.995 non-nullable (C-04) | If < 0.995: fail the request, alert |
| Freshness p99 | reads whose age exceeds 30 s | all reads of real-time features | real-time features only | hourly | ≤ 30 s (C-05) | If > 30 s: flag the feature owner, not the tier |
| Facade p99 | requests exceeding 50 ms | requests | checkout consumer | rolling 5 min | ≤ 50 ms (C-01) | If > 50 ms for 10 min: shed non-checkout load |
| Fallback share | requests with any default substituted | requests | all consumers | rolling 5 min | ≤ 0.002 | If > 0.002: trip the breaker, stop hedging |
| Attribution lag | decision-log rows whose `as_of` differs from the recomputed value | sampled rows | replay | daily | 0 | Any: the log is not capturing the argument |
| Origin queue depth | median depth | — | per shard | 1 min | ≤ 20 | If > 20: disable hedging, widen TTL |

The agreement denominator is deliberately restricted to reads resolved on both paths. A feature that
resolves online and defaults offline is a coverage failure, and mixing the two would let a coverage
collapse masquerade as an agreement improvement: the agreement numerator would drop, but so would the
denominator, and the ratio could stay flat or rise. Reporting the two separately is what makes each
interpretable.

### 9.2 Validation plan

| Validation | Design | Executed? | Would falsify |
|---|---|---|---|
| One-week agreement replay | 10,000 requests/day stratified by popularity decile, both paths, compare per feature | no — not executed | Finding 1: if snapshot agreement ≥ 0.995 on all features, the migration is unnecessary |
| Backfill contamination audit | For each registered feature, check `refresh_time ≤ event_time + lag` over 30 days | no — not executed | §6.9: a feature violating the inequality is leakage-suspect regardless of its agreement |
| Tail measurement under fan-out | Instrument group latency, fit lognormal, compare predicted request p99 at 28.24 ms with measured | no — not executed | §6.4: if measured request p99 exceeds the prediction, groups are correlated |
| Fallback flip-rate measurement | Serve defaults to 0.5 percent of traffic deliberately, measure decision change rate | no — not executed | §6.7: if the flip rate is 2 percent rather than 20 percent, the budget math is 10x more forgiving |
| Hedging load test | Enable hedges at p95 trigger under 8,000/s synthetic load, measure added origin load | no — not executed | §6.4's hedge claim: if added load exceeds 5 percent, cancel is not working |

The backfill contamination audit is the highest-value validation and the cheapest: it is a SQL check over
existing tables, it requires no new infrastructure, and it detects the defect that §2's first incident
exhibited. A team that runs only this audit before any migration will find leakage that no metric
revealed, because §6.5 shows a leak can improve AUC while making the model useless online.

### 9.3 Uncertainty

**The popularity exponent.** α = 0.9 produces a 48.4 percent cache requirement. At α = 0.8 the requirement
is `0.93^5 = 0.6957` (69.6 percent of entities, 6.42 GB with overhead, versus 4.46 GB), and at α = 1.1 the
formula's integral diverges and the correct statement is that a finite cache cannot reach 0.93 without a
tail policy, so the memory plan is undefined rather than larger. The α estimate must therefore come with a
confidence interval and the plan must state the α it assumes. A point estimate of α with no interval is
the weakest link in the whole memory argument.

**The independence of group latencies.** §6.4's 28.24 ms assumes independence; §6.6 shows the analogous
assumption failing for hit rates. If groups share a shard, the request p99 moves toward the group p99
(18 ms) in the correlated limit only when all groups are on the same shard, which is pathological; the
realistic case sits between. The design response is to measure the request p99 directly rather than to
trust either model, and to size the reserve from the measurement.

**The flip rate.** §6.7 turns entirely on `s_flip = 0.20`, which is a construction. At 0.02 the fallback
consumes 10 percent of budget and the ceiling could be 2 percent rather than 0.2 percent. The measurement
(deliberate defaults on 0.5 percent of traffic) is cheap and should precede the guardrail's setting,
because a guardrail set from a guessed flip rate will either fire constantly or never.

### 9.4 What is not measured and cannot be

Model accuracy in production is not measurable here: it needs labels, and labels arrive after the label
delay. The case therefore evaluates agreement, freshness, coverage, latency, and budget — all measurable
without labels — and explicitly does not claim that higher agreement produces a better model. The
connection between agreement and accuracy is a hypothesis with a plausible mechanism (a model trained on
the distribution it will meet) and no evidence in this document. A reader who needs that evidence must
run an A/B test after cutover, which is out of scope for a design case and would take longer than the
migration itself.

**Takeaway:** agreement is the primary control and is measurable without labels; coverage must be reported
separately or it will hide inside agreement; nothing here was executed, and the cheapest validation is a
`refresh_time` audit over existing tables.

## 10. Action Taken

The lead commits to option (c) extended into option (b) over three waves, ordered by label delay rather than
by model size or request volume. Eighteen weeks, with a hard gate between waves.

**Weeks 1–3: the audit that requires no new infrastructure.** Run the `refresh_time ≤ event_time + lag`
check over every registered feature for 30 days, and produce a per-feature contamination report. Add the
required `as_of` parameter to the facade *without* changing behaviour — online callers pass `now`
explicitly, which is a no-op, and training callers pass their label time. Success criterion: every
feature carries a declared `availability_lag`; every facade call site passes `as_of` explicitly; the
audit names the features whose writes violate the inequality. This wave is deliberately non-invasive,
because it produces the evidence that decides whether the other waves are worth their risk.

**Weeks 4–7: build the PIT store for the highest-delay model.** The churn model has a 30-day label delay,
so its training rows are exposed to a 30-day forbidden interval and its features are the most likely to
be contaminated (§6.5's mechanism is a churn-process write). Build the PIT store for the four features
that the audit flagged, write both stores from one materialiser, and switch the churn training pipeline to
the batch API. Success criterion: agreement ≥ 0.999 on a 10,000-row replay, and the churn model's offline
AUC *falls* — a fall is the expected and desired direction, because it means the leak is gone.

That last criterion is the one that needs stating explicitly in the plan, because a falling offline metric
during a correctness migration looks like a regression to anyone who is not told in advance. The lead
writes it into the migration record: the churn model's AUC is expected to drop by 0.03 to 0.06, the drop
is not a defect, and no one is to "fix" it by reverting.

**Weeks 8–12: the relevance model, with the cache key change.** The relevance model's features are
item-level aggregates with short windows, so its PIT requirement is a day-boundary seek rather than an
event-level seek. Migrate it, and simultaneously change the cache key to include `feature_version` and
`shard_hint` — the fix for §2's second incident. Success criterion: zero cross-version reads in a
10-million-request replay, measured by asserting that the version in the key equals the version returned.
This wave carries the highest serving risk because it touches the hot path for the largest consumer, so it
is gated behind the two earlier waves' agreement evidence.

**Weeks 13–15: the fraud model, with the strictest agreement bar.** The fraud model's label delay is
minutes, so its forbidden interval is narrow and its features are naturally near-real-time. It migrates
last not because it is risky but because its low delay means the incumbent snapshot path was least wrong
for it: the expected benefit is smallest, and the lead wants the two highest-benefit migrations proven
before spending the fraud team's time. Success criterion: agreement ≥ 0.9995, one order tighter than the
platform standard, because a fraud score's inputs are audited individually during disputes.

**Weeks 16–18: the propensity model and the incumbent's retirement.** Propensity has a 7-day label delay
and a large feature set; it migrates with the same bar as relevance. Then remove the nightly snapshot
path. Success criterion: the snapshot path's read count reaches zero before deletion, confirmed by a
week of zero traffic rather than by a flag being flipped — a common error is deleting a path that a
forgotten batch job still calls.

Two decisions are explicitly *not* made in this plan, and their absence is deliberate. First, the
materialiser's pass interval is not changed in wave one even though §6.3 shows it dominates freshness;
changing refresh cadence and store topology in the same release would make a staleness movement
unattributable. Second, the hedging policy is not enabled until the tail is measured, because §6.4's
predicted 28.24 ms is a fit, not a measurement, and enabling a hedge against a wrong model adds load for
no benefit.

The cost of the plan is bounded and stated: 4.16 GiB of cache memory at α = 0.9, about 67.6 USD/day of
compute at the planning rate, one new store with a write path, and roughly 26 engineer-weeks across three
teams. The cost is dominated by the third item and the third item is the one the case argues is
unavoidable: two stores with one writer, or two definitions and an unfalsifiable agreement metric.

**Takeaway:** audit before building, migrate the highest-label-delay model first, expect and pre-register a
falling offline metric, and delete nothing until a week of zero traffic confirms it.

## 11. Outcome & Results

Nothing was deployed, so no operational outcome exists. What follows is the set of computed consequences
of the committed design and the set of readings that would constitute success.

### 11.1 Computed consequences

| Quantity | Value | Derivation | Status |
|---|---|---|---|
| Group lookups at peak | 96,000/s | 8,000 × 12 (§6.1) | ILLUSTRATIVE |
| Requests touching origin | 58.1 percent | 1 − 0.93^12 (§6.1) | ILLUSTRATIVE |
| Origin group reads | 6,720/s | 8,000 × 12 × 0.07 (§6.1) | ILLUSTRATIVE |
| Cache memory at α = 0.9 | 4.16 GiB | Zipf inversion (§6.2) | ILLUSTRATIVE |
| Naive memory plan | 7.99 GiB | Uniform-share formula (§6.2) | ILLUSTRATIVE |
| Served stale at u = 30 s | 46.9 percent | Age mixture (§6.3) | ILLUSTRATIVE |
| Request p99 from group p99 | 28.24 ms | Lognormal max of 12 (§6.4) | ILLUSTRATIVE |
| Request p999 | 39.74 ms | Same fit at k = 12 (§6.4) | ILLUSTRATIVE |
| Hedged request p99 | 10.13 ms | Min of 2 draws (§6.4) | ILLUSTRATIVE |
| Leaked training rows | 8,800,000 | 40M × 0.22 (§6.5) | ILLUSTRATIVE |
| Positives inside leaked rows | 82.5 percent | 1.32M / 1.6M (§6.5) | ILLUSTRATIVE |
| Fallback budget consumption | 100 percent | 1.25M × 0.20 (§6.7) | ILLUSTRATIVE |
| Cost per million requests | 0.2704 USD | 23.47 vCPU × 0.12 × 24 / 250 (§6.8) | ILLUSTRATIVE |
| Coalescing reduction on a hot key | 4.8× | 400/s × 12 ms (§6.10) | ILLUSTRATIVE |

The two figures that change decisions are 28.24 ms and 100 percent. The first says a 50 ms SLO with a
12-way fan-out has 21.8 ms for everything else, which rules out any design that adds a synchronous hop.
The second says the fallback ceiling must exist before cutover, not after, because the budget can be
consumed by behaviour that never appears in an error rate.

### 11.2 Success readings and their horizons

- **Agreement ≥ 0.999 per migrated model at the replay that gates its wave.** Readable immediately from the
  replay; no lag. A failure here stops the wave rather than prompting a tuning exercise.
- **Coverage ≥ 0.995 for non-nullable features within 48 hours of a store change.** Readable from the
  decision log. A coverage failure is a store bug or a materialiser lag, and both are diagnosable in a day.
- **Facade p99 at or below 50 ms for checkout on a rolling 5-minute window, from day one.** Readable
  immediately; a breach during a wave rolls that wave back via the flag.
- **Fallback share at or below 0.2 percent, from day one.** This is a guardrail rather than a target, and
  its purpose is to keep the invisible budget consumer bounded.
- **Offline metric movement in the expected direction — downward for churn — at the first retrain after
  each wave.** Readable after the label delay, so up to 30 days after wave two. Any other direction is
  investigated before the next wave.

### 11.3 What the numbers do not show

The case does not show that agreement improves revenue, conversion, or fraud loss. It shows that agreement
is measurable, that the incumbent path's agreement is unknown, and that a specific defect (leakage via a
process-written feature) can raise offline AUC by 0.06 while making a model useless. The step from
"agreement is 0.999" to "the model is better" is a hypothesis. The mechanism is credible — training on the
distribution you will serve is a precondition for calibration — but calibration is not measured here and
cannot be without labels.

The latency figures are also a fit rather than a measurement. The lognormal with median 5 ms and p99 18 ms
reproduces its own inputs exactly, which is a check on the arithmetic and not on the model. If real group
latency has a heavier tail than lognormal (which serving systems often do, because a retransmit or a GC
pause produces a discrete jump), the request p99 will exceed 28.24 ms and the 21.8 ms reserve will be
smaller than stated. §9.2 lists the measurement that resolves this, and it is deliberately placed before
the hedging decision.

**Takeaway:** the fan-out tail and the silent fallback consumption are the two computed results that change
the design; the accuracy benefit is a hypothesis, and the latency model is a fit whose tail must be
measured before it is relied on.

## 12. Failure Modes & What Went Wrong

Eight mechanisms. Each has competing causes and a check that discriminates using only the decision log and
the store, because during an incident the model team is not available.

### 12.1 Leakage through a process-written feature

**Mechanism.** A feature is materialised from a table whose row is updated by the outcome process. "Days
since last order" computed from `updated_at` is the canonical instance: when a user churns, the churn
process touches the row, so the feature records a value that only exists because the outcome happened.
Offline, the feature is near-perfectly predictive; online, it is meaningless because at serving time the
churn has not occurred.

**Competing causes.** (a) Genuine distribution shift between training and serving; (b) a serving bug that
returns the wrong entity's value; (c) the process-written leak. All three produce the same headline: offline
metric good, online metric bad.

**Discriminating check.** Compare the feature's value distribution at `as_of = label_time` with its value
distribution at `as_of = now` for the same entities in the same state. A process-written feature shifts
because the process has not run for serving-time entities. A distribution shift (a) moves both
distributions together. A serving bug (b) produces values that belong to other entities, which the
entity-ID check in the response log identifies. The check is cheap and does not need a model.

**Correction and prevention.** Compute from the event timestamp only, enforce `refresh_time ≤ event_time +
lag` at write time as a hard failure, and require each feature to declare its availability lag at
registration. Verification: the audit in §9.2 run over all features, with a violation count of zero.

### 12.2 Training-serving skew from a window definition that differs by an epsilon

**Mechanism.** The offline query uses `event_time > as_of - interval '30 days'` and the online
materialiser uses `>=`, or one uses a day-aligned boundary and the other a rolling one. The difference is
one row for most features, which is invisible in an aggregate and decisive for a feature like "number of
orders in 30 days" at a boundary.

**Competing causes.** (a) The epsilon in the window comparison; (b) a timezone shift making the 30-day
boundary land one day earlier offline; (c) genuinely different data because the offline query reads a
replica that lags. All three produce a small systematic disagreement concentrated in a subset of entities.

**Discriminating check.** Compute agreement per feature and plot the distribution of `served − batch`
rather than its mean. An epsilon difference produces disagreement concentrated at integer values (an
off-by-one count). A timezone shift produces disagreement concentrated on entities whose local midnight
differs from UTC midnight. A replica lag produces disagreement concentrated on entities with recent
activity, i.e. the ones whose rows changed within the lag window. The three have different shapes, so a
histogram separates them without touching the code.

**Correction and prevention.** Generate both implementations from one registry definition so the
comparison operator is not duplicated; forbid hand-written window SQL in consumer repositories; and align
all windows to the store's event-time grain rather than to the caller's local calendar. Verification: a
boundary test at exactly 30 days for an entity with an event at the boundary, asserting both paths agree.

### 12.3 Cache stampede on a trending entity

**Mechanism.** A trending item's cache entry expires and 400 requests per second for that key all miss,
each issuing an origin read. The origin sees 400 reads/s for one key, the shard saturates, and latency rises
for every entity on that shard.

**Competing causes.** (a) A shard-level capacity problem unrelated to the key; (b) the stampede itself;
(c) a key-design defect causing the entry to be missed even when resident. All three appear as "one shard
is slow".

**Discriminating check.** Plot origin reads per key for the slow shard. A stampede shows one key with reads
equal to the request rate for that key. A capacity problem shows uniform reads across keys with high
latency. A key-design defect shows reads for keys that should hit, i.e. the read rate exceeds the miss rate
implied by entry age — and the presence of the same logical entity under two key forms confirms it.

**Correction and prevention.** Single-flight per key with a bounded follower wait (§6.10), a TTL jitter of
±10 percent so a cohort of keys does not expire together, and a background refresh for keys in the top
popularity decile so the hottest entries never expire on a request. Verification: a synthetic 400/s load
on one cold key must produce at most `1/W_fetch` origin reads (83/s at 12 ms), not 400/s.

### 12.4 `as_of` substitution: the silent `now`

**Mechanism.** A training job calls the facade without an `as_of`, a default fills in `now`, and the
resulting rows are indistinguishable from correct ones. Every downstream metric is plausible, the model
scores well offline, and the defect is only visible as an online/offline divergence that nobody can
attribute.

**Competing causes.** (a) The substitution; (b) a legitimately different feature definition between the
two paths; (c) a timezone or window epsilon (§12.2). All three produce disagreement, but only one of them
produces disagreement that scales with label delay.

**Discriminating check.** Group disagreement by label-delay bucket. Substitution (a) produces disagreement
that grows with label delay, because a longer delay means a larger gap between `as_of` and `now`. A
definition difference (b) produces disagreement independent of delay. A window epsilon (c) produces a small
constant disagreement. The check requires nothing but the log and the label delay, both of which exist.

**Correction and prevention.** Remove the default from the signature so a missing argument fails to compile
or throws at the boundary; log the `as_of` the caller supplied separately from the one the store applied;
and have the replay harness assert the two are equal. Verification: a test that calls the facade without
`as_of` must fail, on every language binding, which is why the check belongs in the client library rather
than in review discipline.

### 12.5 Materialiser writing the online store and the PIT store inconsistently

**Mechanism.** Two writers, or one writer with a partial failure between the two writes. The online store
has a value the PIT store lacks, so online serving returns a value that a replay cannot reproduce. The
defect is invisible online (the value is right) and visible only as an unexplainable agreement gap.

**Competing causes.** (a) A partial write failure; (b) a definition difference (§12.2); (c) a read-time
filter admitting a row the writer excluded. All three produce "the replay disagrees but the served value
looks fine".

**Discriminating check.** For a disagreed feature read, query both stores at the same key and compare
`refresh_time`. If the online store has a row and the PIT store does not, it is a write failure (a). If
both have rows with different values, it is a definition or ordering difference (b). If the PIT store has
the row but the read-time filter excluded it, the decision log's exclusion reason identifies (c) — which is
why the log must record exclusions, not only values.

**Correction and prevention.** Single writer, write PIT first and online second, and treat a PIT write
success with an online write failure as a retryable state rather than a success; make the PIT store
authoritative for replay so that an online/PIT divergence is detectable by construction. Verification: an
injected failure between the two writes must leave the online store without the value, so that the next
read repairs from PIT.

### 12.6 Tail amplification by fan-out under correlation

**Mechanism.** §6.4 assumes independent group latencies. When groups share a network path or a shard, a
jitter event slows all of them together, and the request p99 approaches the group p99 *plus* the shared
delay rather than the lognormal prediction. The design's 21.8 ms reserve evaporates.

**Competing causes.** (a) A single slow group (a straggler); (b) correlated slowness across groups;
(c) a facade-level serialisation defect that makes the groups effectively sequential. All three produce a
p99 breach.

**Discriminating check.** Compute the correlation of per-group latencies within each request from the
decision log. A straggler shows one group slow and eleven fast, so the within-request variance is high.
Correlated slowness shows all groups slow together, so within-request variance is low and between-request
variance is high. A serialisation defect shows group latencies that are each fine but whose sum exceeds the
request latency, which is arithmetic evidence of sequencing.

**Correction and prevention.** Partition entity groups across shards so that a request's 12 groups do not
share one shard; cap per-request group count or split a fan-out into two rounds with a partial merge; and
measure the request p99 directly rather than deriving it, so a correlation change is visible. Verification:
a load test with induced shard-level jitter must show a request p99 within 20 percent of the prediction, or
the correlation assumption fails and the reserve must be resized.

### 12.7 Error-budget exhaustion by a healthy-looking fallback

**Mechanism.** §6.7's mechanism: a store read timeout causes defaults, every response is a 200, no alert
fires, and 0.5 percent of traffic gets a default for hours. The budget is consumed by a path that never
appears in the error rate.

**Competing causes.** (a) A genuine degradation of the model's inputs, i.e. the defaults are harmless;
(b) a flip rate higher than assumed; (c) the fallback share being unbounded. All three produce the same
absent signal.

**Discriminating check.** Measure the decision-flip rate directly by serving defaults to a small deliberate
sample and comparing the model's output with and without the default. That converts an assumption into a
measurement and distinguishes (b) from the others. For (c), plot the fallback share over time: a rising
share with a flat error rate is the signature, and it is visible only if the share is a first-class metric.

**Correction and prevention.** Make fallback share a guardrail with a 0.2 percent ceiling, trip a breaker
at the ceiling so degradation cannot spread, and record every substituted field in the decision log so a
dispute can be adjudicated. Verification: a deliberate store outage must trip the breaker within one
minute, and the fallback share must never exceed the ceiling.

### 12.8 Feature-version drift after a definition change

**Mechanism.** A feature owner edits a definition in place — changing a window from 7 to 14 days — without
bumping the version. The materialiser writes new values under the old version, the cache holds old values
under the same key, and the training pipeline reads a mixture. The model is trained on two definitions
simultaneously.

**Competing causes.** (a) The in-place edit; (b) a backfill that rewrote history with the new definition,
so the mixture is uniform but wrong; (c) a cache key omitting the version, so old and new values share an
entry. All three produce a bimodal value distribution.

**Discriminating check.** Plot the feature's value distribution by day. An in-place edit produces a step
change at the release time, with old values persisting in cache for up to one TTL. A backfill produces a
uniform change with no step. A cache-key omission produces a mixture within a single day, with values
disagreeing for the same entity across replicas — checkable by querying two replicas for the same key.

**Correction and prevention.** Make definitions immutable: a change is a new version with a start date, and
the registry rejects in-place edits; include `feature_version` in the cache key (§8.3); and have the
replay harness assert that the version in the request equals the version in the response. Verification: an
attempted in-place edit must be rejected by the registry, and a cache read must never return a value whose
version differs from the requested one.

**Takeaway:** eight mechanisms; the discriminating checks turn on whether disagreement scales with label
delay (the signature of `as_of` substitution), on within-request latency variance (straggler versus
correlation), and on whether a value distribution changes as a step, a mixture, or not at all.

## 13. Transferable Lesson

**Any system that serves a stored answer to a caller who has already moved past it needs the caller's clock
to be an argument.** Feature serving is the case where this is most visible — training asks for the past,
online asks for now — but the same shape appears in price lookup (the price at order time), inventory
reservation (stock at reservation time), policy evaluation (the policy version in force when the event
occurred), and document retrieval (the document as it was). In each, the naive implementation reads "the
current value" and the correct implementation reads "the value in force at `as_of`", and the difference is
invisible until someone compares two paths.

The transferable artefacts are four and each is cheap:

1. `as_of` as a required parameter with no default, in the client library rather than in review discipline.
2. A write-time inequality (`refresh_time ≤ event_time + lag`) enforced as a failure, which catches
   process-written leakage before it reaches a model.
3. One writer for every store that serves the same definition, so two paths cannot diverge by construction.
4. An agreement metric computed by replay at the grain of the atomic read, not at the grain of the request.

The third artefact is the one teams resist, because two writers are locally simpler. The resistance is
misplaced: the cost of one writer is a materialiser that knows both store schemas, and the cost of two
writers is an agreement defect that is not attributable to either writer.

A second transferable point concerns fan-out and metrics. The case's most useful arithmetic is that a
0.93 per-group hit rate is a 58.1 percent request miss share at 12 groups. The general rule: **for any
conjunctive system, the per-item metric is not the system metric, and the gap grows with the number of
conjuncts.** The same arithmetic explains why a 99.9 percent per-service availability yields 98.8 percent
for twelve chained services (`0.999^12 = 0.98806`), why a per-shard p99 is not a query p99, and why a
per-item cache hit rate is not a page hit rate. A reviewer who asks "how many conjuncts" will never be
surprised by a tail.

A third point is about measurement design. The case reports agreement, coverage, freshness, and fallback
share as four separate metrics because each hides inside another if merged: coverage hides inside
agreement (defaults look like values), freshness hides inside coverage (a stale value is still a value),
and fallback share hides inside agreement (a defaulted read can accidentally agree). The rule is that a
metric must be reported wherever a substitution occurs, because a substitution is exactly the thing that
makes a ratio look better than the system is.

Finally, the case transfers a scheduling lesson. Migration was ordered by label delay, not by volume or by
team enthusiasm. Label delay is the multiplier on leakage exposure, and it is also the variable that
determines how long a mistake stays invisible: a 30-day-delay model will not reveal its defect for 30 days.
**Migrate first where the forbidden interval is widest, because that is where the defect is largest and
where waiting costs the most.**

**Takeaway:** make the caller's clock an argument, enforce the write-time inequality, use one writer per
definition, measure agreement at the atomic read grain, remember that conjunction multiplies tails, and
order migrations by label delay.

## 14. Connections & Cross-References

Within this unit, the notification-platform case is the closest structural sibling: it also separates a
decision record from an effect, and its idempotency key plays the role that `as_of` plays here — both are
the argument without which the system cannot be replayed. The difference is instructive. The notification
case's correctness question is *identity* (has this effect already happened), while this case's is
*temporality* (what did the system know). A platform that confuses the two either deduplicates reads that
should be distinct or replays effects that should be idempotent.

Outside the unit, four areas supply the machinery. **Database theory** supplies the point-in-time read as a
range query on a sorted key, and the distinction between event time and processing time is the standard
bitemporal distinction; a reader who knows the difference between valid time and transaction time will
recognise both inequalities in §5a. **Statistics** supplies the lognormal fit and the max-of-k quantile
inversion in §6.4, and the concentration argument in §6.5 is a mixing-distribution argument. **Cache theory**
supplies the Zipf inversion in §6.2; the formula is a special case of the general result that hit rate under
skew is governed by the tail exponent, not by the cache size as a fraction of the working set.
**ML engineering** supplies the leakage taxonomy: label leakage, temporal leakage, and train-serve skew are
three distinct failures and §12 has one mechanism for each (§12.1, §12.4, §12.2).

Three contrasts are worth holding. A **CDN** also caches under skew and also faces stampedes, but it has no
correctness contract beyond freshness, so its hit rate can be optimised freely — this case cannot, because
a higher hit rate achieved by ignoring `as_of` is a correctness regression. A **materialised view** also
serves precomputed answers, but it is refreshed as a whole and has no per-call clock, so the two-store
split has no analogue. A **streaming aggregator** also provides low-latency values, but its output is the
current value by construction and cannot answer a historical query at all, which is why the PIT store cannot
be replaced by a stream no matter how fast the stream is.

For a reader extending this case, the highest-value addition is a measured group-latency distribution from
a real system, because every latency conclusion rests on a fitted lognormal that reproduces its own inputs.
The second addition is a measured flip rate for the fallback path, because the error-budget conclusion turns
entirely on a constructed 0.20. Either would convert a conditional statement into evidence.

### Interview pack

Fifteen questions. Four require computation, five require diagnosis, six require judgement; each answer
stands alone.

**Q1. Why must `as_of` be a required parameter rather than defaulting to `now`?**

- **Answer:** Because a default makes the training caller's mistake indistinguishable from correct
  behaviour. If `as_of` defaults to `now`, a training job that forgets the argument produces plausible rows
  and the only symptom is an unexplained offline/online divergence. With no default, the omission is a
  failure at the boundary, which is the cheapest possible detection point and the only one that scales past
  a handful of callers.
- **Reasoning:** Correctness controls work by making the defective path impossible or loud; a default makes
  it possible and silent. The cost of removing the default is one migration of call sites, which is bounded
  and one-time.
- **Follow-up:** What if a caller genuinely wants `now`? *(Answer: it passes `now` explicitly; the point is
  that the choice is visible in the code and in the log, not that `now` is forbidden.)*
- **Red flag:** "Defaults are convenient, and training should just be careful."
- **Scoring:** 3 for the silent-versus-loud argument and the log consequence; 2 for "explicit is better";
  1 for "documentation would fix it"; 0 for keeping the default.

**Q2. A tier shows a 93 percent group hit rate across 12 groups per request. What share of requests touch
the origin, and why is the answer surprising?**

- **Answer:** `1 − 0.93^12 = 1 − 0.4186 = 0.5814`, so 58.1 percent of requests touch the origin at least
  once. It is surprising because 7 percent per group reads as "7 percent of traffic", but a request is a
  conjunction of twelve independent draws, and the probability that none misses decays geometrically. The
  origin's *load* is still small — 0.84 group reads per request, 6,720/s at 8,000 requests/s — so the
  origin is lightly loaded and widely latency-exposed.
- **Reasoning:** Two different questions are being answered by one number: exposure (a tail question) and
  load (a capacity question). Conjunction changes the first by an order of magnitude and leaves the second
  linear.
- **Follow-up:** At 20 groups, what happens? *(Answer: `1 − 0.93^20 = 0.766`, so three-quarters of requests
  touch the origin while load is 1.4 reads per request.)*
- **Red flag:** "7 percent of requests miss."
- **Scoring:** 3 for 58.1 percent, the mechanism, and the load/exposure distinction; 2 for the number; 1
  for "more than 7 percent"; 0 for 7 percent.

**Q3. How much memory does a 0.93 hit rate need for 2M users at 1,900 B and 4M items at 640 B, with
α = 0.9 and a 1.45 overhead?**

- **Answer:** The cache fraction is `0.93^(1/(1−0.9)) = 0.93^10 = 0.48398`, so 48.4 percent of entities.
  Payload is `0.48398 × (2,000,000 × 1,900 + 4,000,000 × 640) = 0.48398 × 6.36e9 = 3.078 GB`, and with the
  overhead 4.463 GB = 4.157 GiB. Using 93 percent of entities instead gives 5.915 GB payload and 8.576 GB
  over-provisioned, which is 1.92× too much.
- **Reasoning:** Under Zipf(α) with α < 1, the hit rate is `c^(1−α)`, so the fraction of entities needed is
  a power of the hit rate with exponent `1/(1−α)`. At α = 0.9 the exponent is 10, which turns 0.93 into
  0.484 — the head of the distribution carries the references.
- **Follow-up:** What if α is estimated at 0.8? *(Answer: the exponent becomes 5, `0.93^5 = 0.6957`, so
  69.6 percent of entities and 6.42 GB — the answer is highly sensitive to α, so α must be reported with a
  confidence interval.)*
- **Red flag:** "93 percent of entities, obviously."
- **Scoring:** 3 for 4.16 GiB with the exponent shown; 2 for the right structure with an arithmetic slip;
  1 for the naive answer; 0 for "memory is cheap".

**Q4. Your dashboard says TTL 60 s and refresh 30 s. What fraction of reads is older than 30 s?**

- **Answer:** About 46.9 percent, and almost all of it is the batch path. A batch value with a uniform
  refresh phase and a 60-second TTL has age uniform on 0–60, so half of it exceeds 30 seconds; at 93 percent
  of traffic that contributes 0.465. The real-time path with median 12 s and p99 45 s contributes
  `0.07 × 0.0534 = 0.0037`. Total 0.4687. The maximum age is `60 + 45 = 105 s`, which no TTL setting below
  60 can fix.
- **Reasoning:** TTL bounds admission age, not total age; total age includes the refresh interval and
  materialisation lag. Reporting TTL as freshness is a category error that is invisible precisely because
  TTL is a plausible-looking number.
- **Follow-up:** How do you cut staleness by two orders of magnitude cheaply? *(Answer: refresh the batch
  path every 30 s instead of 60, which makes batch age uniform on 0–30 and drops the share to 0.37 percent;
  the cost is writer CPU, not read capacity.)*
- **Red flag:** "TTL 60 s means at most 60 s stale", offered as a bound rather than a maximum.
- **Scoring:** 3 for 46.9 percent with the batch/real-time split; 2 for the right idea without the split; 1
  for "60 seconds maximum"; 0 for "TTL is freshness".

**Q5. Group p99 is 18 ms and a request fans out to 12 groups. What is the request p99?**

- **Answer:** About 28.24 ms, assuming independent lognormal group latencies with a 5 ms median. The fit
  gives `σ = ln(18/5)/2.3263 = 0.5506`; the request p99 requires each group at `0.99^(1/12) = 0.999164`,
  whose z-score is 3.144, giving `5 × exp(3.144 × 0.5506) = 28.24 ms`. The group p99 is only the request
  p88.6 (`0.99^12 = 0.8864`), which is why quoting it against the SLO is a category error.
- **Reasoning:** A request succeeds only if all 12 groups succeed within budget, so the request quantile
  inverts the product of the group CDFs. The tail of a maximum is worse than the tail of any component.
- **Follow-up:** What does the request p999 look like? *(Answer: `0.999^(1/12) = 0.9999166`, z = 3.756,
  giving 39.74 ms — within a 50 ms SLO but with only 10.3 ms of margin.)*
- **Red flag:** "The request p99 is the group p99, since they're parallel."
- **Scoring:** 3 for 28.24 ms with the inversion shown; 2 for "worse than 18 ms" with reasoning; 1 for 18
  ms; 0 for "parallel, so it's the max of 12 which is still 18".

**Q6. 22 percent of training rows contain a feature written after the label time. How bad is that?**

- **Answer:** Bad enough to be decisive, and the row share is the wrong denominator. If the leak is
  correlated with the label — say 15 percent positive among leaked rows versus 4 percent overall — then
  8.8 million leaked rows contain 1.32 million of the 1.6 million positives, i.e. 82.5 percent of all
  positives. The leak therefore determines the model's decision boundary while a data-quality report shows
  a fifth of rows affected.
- **Reasoning:** A ranking metric responds to the affected positives, not to the affected rows. When the
  leak is written by the outcome process, affected rows are near-perfectly positive, so the two shares
  diverge by a factor of about 4.
- **Follow-up:** What is the direct check, without labels? *(Answer: enforce `refresh_time ≤ event_time +
  lag` at write time and audit it; the violation is a structural fact, detectable with no model.)*
- **Red flag:** "22 percent is small, the effect will wash out."
- **Scoring:** 3 for the positive-share computation and the mechanism; 2 for "leaks correlate with the
  label"; 1 for "it biases AUC"; 0 for "small share, small effect".

**Q7. A fallback path serves defaults for 0.5 percent of traffic and changes 20 percent of decisions. What
does that cost, and why does nobody notice?**

- **Answer:** `250,000,000 × 0.005 × 0.20 = 250,000` effective errors per day against a budget of
  `250,000,000 × 0.001 = 250,000`, so exactly 100 percent of the monthly error budget with zero margin.
  Nobody notices because every response is a 200, no request failed, and the error-rate metric — which
  counts failures — reads zero. The consumption is only visible if fallback share and flip rate are
  measured and multiplied.
- **Reasoning:** Error budgets are usually defined on failed requests, and a degraded response is not a
  failed request. The budget therefore needs a second input: the share of requests whose output changed
  materially.
- **Follow-up:** What ceiling prevents this? *(Answer: cap fallback share at 0.2 percent so a full episode
  consumes 40 percent of budget, and trip a breaker at the ceiling.)*
- **Red flag:** "Fallbacks are cheap, they keep availability up."
- **Scoring:** 3 for the 250,000 computation and the 200-OK invisibility; 2 for the multiplication without
  the invisibility argument; 1 for "fallbacks degrade quality"; 0 for "fallbacks are free".

**Q8. Group latencies within a request are all slow, but each individually is within its p99. Diagnose.**

- **Answer:** Correlated slowness, not stragglers. The signature is low within-request variance and high
  between-request variance: all 12 groups are slowed by a shared cause (a network path, a shard, an
  instance hosting multiple group partitions). The request p99 then sits near the group p99 plus the shared
  delay, worse than the independent model predicts, and the reserve implied by 28.24 ms is not real.
- **Reasoning:** Fan-out converts a per-item distribution into a worst-case statistic only under
  independence. Correlation makes the worst case common, so the effective number of independent draws is
  closer to 1 than to 12.
- **Follow-up:** What design change reduces correlation? *(Answer: partition a request's groups across
  shards so they do not share an instance, and split a 12-way fan-out into two rounds if necessary.)*
- **Red flag:** "Twelve slow groups means twelve stragglers."
- **Scoring:** 3 for correlation with the variance signature; 2 for correlation without the signature; 1 for
  "add more replicas"; 0 for "retry each group".

**Q9. Two workers write the online store and the PIT store. What breaks, and what is the cheap fix?**

- **Answer:** A partial failure leaves the online store with a value the PIT store lacks, so serving returns
  something the replay cannot reproduce; the symptom is an unattributable agreement gap while every served
  value looks correct. The cheap fix is not a distributed transaction: it is one writer for both stores,
  writing PIT first and online second, so the only divergence possible is "online missing", which the next
  read repairs from PIT.
- **Reasoning:** Making one store authoritative for replay converts an attribution problem into a
  detectable asymmetry. Two-writer designs cannot be repaired by ordering alone because the writers do not
  share an ordering.
- **Follow-up:** What makes "online missing" safe? *(Answer: the read path falls back to PIT for a value
  whose `as_of` is within the grace window, and records the fallback so the repair rate is visible.)*
- **Red flag:** "Wrap both writes in a transaction" without saying what happens when it fails.
- **Scoring:** 3 for one writer plus the ordering and repair path; 2 for "single writer"; 1 for "add
  retries"; 0 for a two-phase commit proposal with no failure analysis.

**Q10. Why does the case insist agreement is measured at the feature read rather than at the request?**

- **Answer:** Because a request-level agreement figure averages over 240 reads, so a defect confined to the
  one group that drives the score is diluted to invisibility. At 240 reads per request, a feature that is
  wrong for 1 percent of reads produces request-level disagreement of `1 − 0.99^240 = 0.91`, which is
  visible, but a feature wrong for 0.01 percent produces `1 − 0.9999^240 = 0.024` — small enough to be
  mistaken for noise, and concentrated in exactly the tail entities that matter. Measuring at the read also
  attributes the defect to a feature owner, which a request-level number cannot.
- **Reasoning:** The grain of measurement must match the grain of accountability. Reads have feature names;
  requests have only callers.
- **Follow-up:** How do you keep the read-level report small enough to read? *(Answer: stratify by
  popularity decile and report per feature; the deciles preserve tail visibility while bounding the row
  count.)*
- **Red flag:** "Request agreement is what the user experiences."
- **Scoring:** 3 for dilution plus attribution; 2 for attribution alone; 1 for "granular is better"; 0 for
  request-level only.

**Q11. A feature's distribution shifts abruptly at a release boundary. Give three causes and separate
them.**

- **Answer:** (a) An in-place definition edit under the same version, which shifts at release and leaves old
  cached values for up to one TTL; (b) a backfill that rewrote history with the new definition, which shifts
  uniformly with no step at release; (c) a cache key missing `feature_version`, which produces a within-day
  mixture rather than a clean step. Separate them by the shape: step at release = (a), uniform shift = (b),
  same-entity disagreement across replicas = (c).
- **Reasoning:** Each cause leaves a different footprint in the joint distribution of value and time, so a
  histogram resolves what a single number cannot.
- **Follow-up:** How do you prevent (a)? *(Answer: make definitions immutable, reject in-place edits at the
  registry, and require a new version with a start date.)*
- **Red flag:** "Roll back the release" — treats every case as (a).
- **Scoring:** 3 for three causes and the shapes; 2 for two causes; 1 for one; 0 for "it's a bad deploy".

**Q12. Order the four models' migrations and defend the order.**

- **Answer:** Churn (30-day label delay) first, then relevance, then propensity (7 days), then fraud
  (minutes). Label delay determines the width of the forbidden interval and therefore the size of the
  leakage defect and the time before a mistake becomes visible. Churn is also the model where §2's
  process-written leak is most likely, so it produces the largest expected benefit and the earliest signal.
  Fraud goes last because its short delay made the incumbent path least wrong for it, so the benefit is
  smallest.
- **Reasoning:** A migration is a sequence of experiments ordered by expected effect and by how quickly a
  wrong result is detectable. Label delay is the multiplier on both.
- **Follow-up:** What if the churn model's offline AUC drops 0.05? *(Answer: that is the expected direction;
  pre-register it so nobody reverts, and validate with an online test after the label delay elapses.)*
- **Red flag:** "Start with the smallest model to reduce risk" — volume is not blast radius.
- **Scoring:** 3 for the order plus the delay multiplier; 2 for the order with a weaker reason; 1 for volume
  ordering; 0 for "migrate all at once".

**Q13. Your store read times out and the facade returns defaults. What must the log record so that this is
auditable?**

- **Answer:** The `as_of` supplied, the entity and feature version requested, the value returned, whether it
  was a value or a default, the substitution reason, and the store's outcome. Without the substitution flag
  a defaulted read is indistinguishable from a real value that happens to equal the default, and without
  `as_of` the replay cannot reconstruct what was asked. The log must also record exclusions (a PIT row
  filtered by `refresh_time`), because otherwise a filtered read looks like a missing one.
- **Reasoning:** Auditability requires that the difference between "the value was X" and "we chose X
  because we could not find out" survives into the record. A log that records only outputs loses that
  distinction permanently.
- **Follow-up:** What metric does the flag enable? *(Answer: fallback share, which is the guardrail in
  §6.7 and the only visible signal of the invisible budget consumer.)*
- **Red flag:** "Log the response; the store's own metrics show the timeout."
- **Scoring:** 3 for the substitution flag plus `as_of` plus exclusions; 2 for the flag alone; 1 for
  "log everything"; 0 for logging only values.

**Q14. What is the difference between freshness and TTL, and why does the distinction matter operationally?**

- **Answer:** TTL bounds how long a value may be admitted after it was materialised; freshness is the age of
  the underlying fact at serve time, which is materialisation time minus fact time plus the time since
  materialisation plus the TTL phase. A 60-second TTL on top of a 60-second batch refresh yields ages up to
  120 seconds, so the TTL is 60 and the freshness bound is 120. Operationally it matters because a
  freshness incident sends the team to the cache configuration, while the actual cause is usually the
  materialiser's pass interval — which is a writer-side knob and a different owner.
- **Reasoning:** Two different clocks are being conflated: how long a computed value may live, and how old
  the fact it describes is. A TTL cannot compensate for a slow writer, and no TTL setting can deliver an age
  bound tighter than the refresh interval.
- **Follow-up:** How would you report freshness so the owner is obvious? *(Answer: an age histogram split by
  path, batch versus real-time, because the split immediately shows which writer to call.)*
- **Red flag:** "Freshness is the TTL."
- **Scoring:** 3 for the two-clock explanation and the owner consequence; 2 for the definition; 1 for
  "freshness means recent"; 0 for treating them as synonyms.

**Q15. Why does the design refuse to fail a request when a nullable feature has no PIT row, but fail it
when a non-nullable one does?**

- **Answer:** Because the two cases have different semantics. For a nullable feature, "no history" is a real
  state that the definition includes and the model was trained on, so a default is a correct answer and
  failing the request would be wrong. For a non-nullable feature, no row means the caller asked for a moment
  before the feature existed or the store is incomplete; substituting a default silently changes the input
  distribution, so failing surfaces the defect at the caller where it can be fixed.
- **Reasoning:** The distinction is whether absence is information or a bug. A default is only admissible
  when the training data contained the same absence in the same proportion; otherwise it is a distribution
  shift applied at serving time.
- **Follow-up:** How do you know a nullable feature's default rate matches training? *(Answer: compare the
  served default share with the training null share per feature; a divergence is a store or coverage
  defect.)*
- **Red flag:** "Always default, availability first."
- **Scoring:** 3 for the semantics plus the distribution argument; 2 for the semantics; 1 for "it depends";
  0 for "always default" or "always fail".

**Takeaway:** the pack's five computations each invert something a dashboard reports as a single number, and
its five diagnoses each separate causes by the shape of a distribution rather than by a summary statistic.

## 15. Quick Reference Card

| Item | Value or rule | Why it matters |
|---|---|---|
| Clock contract | `as_of` required, no default, logged separately from the applied value | A silent `now` is undetectable except as a delay-scaled divergence |
| Hit rate → request exposure | `1 − h^k`; 0.93 at k = 12 gives 58.1 percent | Conjunction multiplies misses; load stays linear at 0.84 reads/request |
| Memory under skew | `c = h^(1/(1−α))`; 0.93 at α = 0.9 gives 48.4 percent, 4.16 GiB | Uniform-share sizing over-provisions 1.92× |
| Freshness | measure age, not TTL; max age = TTL + refresh | 46.9 percent of reads exceed 30 s at TTL 60 / refresh 30 |
| Fan-out tail | request p99 = group p99 inverted by `(1−p)^(1/k)`; 18 ms → 28.24 ms | Leaves 21.8 ms of a 50 ms SLO for everything else |
| Fan-out p999 | 39.74 ms at the same fit | The SLO holds at p99 with room and at p999 barely |
| Leakage metric | affected positives, not affected rows; 22 percent rows = 82.5 percent positives | Row-share reporting hides the decisive defect |
| Write-time invariant | `refresh_time ≤ event_time + lag`, hard failure | Catches process-written leakage with no model |
| Cache key | `(entity_id, feature_version, shard_hint)` | Omitting version or entity type produces silent wrong values |
| Fallback guardrail | share ≤ 0.2 percent; 0.5 percent × 0.20 flip = 100 percent of budget | A 200-OK degradation is the budget's largest silent consumer |
| Store topology | one writer, PIT first then online, PIT authoritative for replay | Two writers make agreement unattributable |
| Latency budget | stages sum to the SLO, with a stated reserve | Per-stage budgets that are never summed exceed the contract |
| Migration order | by label delay, widest forbidden interval first | Delay sets defect size and time-to-detection |
| Primary control | agreement ≥ 0.999 at the feature-read grain, stratified by popularity decile | Request-level agreement dilutes a single-group defect |

The card compresses §5e, §6, and §8; where it disagrees with a section, the section carries the validity
condition and wins. Two entries are most often missed in review: the clock contract's *logged separately*
clause and the fallback ceiling. Both are cheap to add before cutover and expensive to add after, because
neither produces a visible failure when it is absent.
