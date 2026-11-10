---
title: "Feature Store Design: Time, Consistency, and Operations"
topic: "Feature store design — point-in-time correctness, freshness contracts, and online/offline parity"
collection: "practical-ml-design-and-interviews"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Feature Store Design: Time, Consistency, and Operations

> **Purpose:** Given entity events, visibility timestamps, and a prediction cutoff, derive a historical
> join that a model could legally have used, then choose serving semantics that keep the online
> vector close to that historical definition.
> **Audience:** ML engineers, data engineers, platform architects, and system-design interview candidates.
> **Prerequisites:** SQL joins and window functions; the difference between event time and processing
> time; basic queueing and cache reasoning; supervised-learning leakage and train/serve skew.
> **Outcome:** You can write an availability-filtered point-in-time join, size a grace period and a
> TTL from stated penalties, budget online latency across feature views, and diagnose a parity
> break to one of four named layers.
> **Exclusions:** Training algorithms, streaming-engine operator internals, vendor API guarantees,
> and data-governance policy text. This sheet also does not argue that any particular product is
> required; the cheapest correct pipeline is often a scheduled job plus a key-value table.
> **Level:** advanced / production.
> **Evidence:** DERIVED invariants with arithmetic shown inline; ILLUSTRATIVE scenario numbers
> (a card-transaction ledger, a 25M-entity account base) that no real system produced; HEURISTIC
> operating ranges flagged as such. No deployment, incident, or vendor measurement is claimed.
> **Retrieval cue:** Section 4 for the derivations, Section 5 for one fully traced entity,
> Section 8 for diagnosis, Section 11 for interviews.

## 1. File Header

This sheet exists because "we have a feature store" is not a design. A store is a set of tables and
an API; the design is the answer to five questions that outlive any vendor choice: which rows a
prediction was allowed to see, which version of the definition produced them, how stale a served
value is permitted to be, what happens when the same entity appears twice with different keys, and
how you will prove that the vector served at request time equals the vector used at training time.

Every one of those questions is answerable with arithmetic, and each answer changes the physical
layout. A wrong answer to the first question does not produce a crash. It produces a model with a
respectable offline AUC and a production lift near zero, discovered weeks later, and the bug is
invisible in code review because the SQL looks like ordinary SQL.

The sheet therefore treats the historical join as the primary artifact and the online table as a
cache of it. If you invert that ordering — build the low-latency table first and reason about
history afterwards — you will encode whatever the stream happened to contain at backfill time and
spend the rest of the project reconciling two definitions of one feature.

**What this sheet claims.** Temporal invariants that follow from the definition of a prediction
(the entity's state at a cutoff cannot depend on events the entity had not yet produced, nor on rows
the pipeline had not yet ingested). Cost and freshness relations that follow from queueing and
survival arguments. Diagnosis procedures that separate four layers of mismatch by observable
signatures.

**What it does not claim.** That any named product implements these semantics in a particular way.
That the illustrative ledger below resembles a real card portfolio. That a grace period chosen here
is appropriate for a different penalty structure. Numbers are arithmetic demonstrations, not
benchmarks.

**Notation conventions used throughout.** Event time is written `t_e`, ingestion or visibility time
`t_i`, effective time for slowly-changing attributes `t_v`, and the prediction cutoff `t_c`. All
timestamps are UTC instants unless a section explicitly discusses local calendars. Windows are
half-open, `[t_c - W, t_c)`, so a row landing exactly on a boundary belongs to exactly one window.
Aggregates are over rows, not over time buckets, unless a bucketed definition is stated. Currency is
nominal USD in the illustrative scenarios. Every formula states the units of its inputs, because the
most common production defect in this area is a unit mismatch that a type system will not catch.

**Reading paths.** A platform engineer should read Sections 2, 4, 7, and 9 to decide what to build.
A data engineer owning the join should read Sections 3, 4, and 5 and then reproduce the ledger by
hand. An ML engineer debugging a lift failure should read Sections 8 and 6 in that order. An
interview candidate should read Section 4 for the derivations and Section 11 for the question bank,
then return to Section 8 because diagnosis questions dominate senior loops.

**Maintenance trigger.** Revise this sheet when any of the following changes: the definition of the
prediction cutoff (for example a shift from batch scoring to synchronous scoring), the arrival-time
distribution of source events, the retention policy of the online table, or the tie-break rule used
for simultaneous corrections. Each of those invalidates either the join or the freshness arithmetic,
and the version field in the frontmatter should move with the change.

**Relationship to the unit.** This is the third sheet of the unit. Sheet 01 fixes the decision
contract and the metric; this sheet supplies the inputs that the contract assumes are reproducible.
Sheet 02 of the unit's case pair shows the same invariants inside an incident narrative. When the
two disagree, this sheet's derivation wins and the case is the application.

## 2. Concept Map / Mental Model

The system has one job: given `(entity_key, t_c)`, return the vector the model would have needed,
computed only from information that existed at `t_c`. Both the training path and the serving path
are implementations of that one function, and the interesting failures are differences between the
two implementations.

```text
 raw events            typed, deduplicated events         feature definitions
 (append-only)         (entity, t_e, t_i, payload)        (versioned SQL/py)
      |                          |                                 |
      |  ingest                   |  window + aggregate             |
      v                          v                                 v
 +---------+            +-------------------+          +-----------------------+
 | landing |----------->| event history     |--------->| historical feature    |
 |  table  |            | (partition by day)|          | table (entity, t_c,   |
 +---------+            +-------------------+          | feature_version, val) |
                               |                       +-----------------------+
                               |                                   |
                               | materialize latest                | point-in-time
                               v                                   v
                        +--------------+                  +------------------+
                        | online table |                  | training set     |
                        | (key -> vec) |                  | (X at t_c, y)    |
                        +--------------+                  +------------------+
                               |
                               v
                        serving API ----> model ----> decision
                               |
                               +----> serving log (what was actually sent)
```

Read the diagram as two consumers of one history, never as two pipelines. The moment the historical
feature table and the online table are computed by different code paths, you have two definitions
of the same column, and no amount of testing will keep them identical under change. The standard
resolution is that the online table is a projection of the latest row of the historical table, and
the historical table is produced by one versioned definition.

**The four contracts.** A usable feature store publishes four things, and each one is a promise that
an upstream team can break:

1. **Time contract** — which timestamps exist, what each means, and what the join filters on. This
   is the only contract that prevents leakage, and it is the one most often left implicit.
2. **Entity contract** — the key, its type and normalization, how entities merge and split, and what
   happens to history after a merge. Silent key changes duplicate history and halve every aggregate.
3. **Value contract** — the definition text, its version, the unit, the null semantics, and the
   behavior on missing input. A feature that returns `0` for "no history" and `0` for "spent
   nothing" has merged two populations.
4. **Freshness contract** — the maximum age of a served value, what is returned when the age is
   exceeded, and how the age is measured. Without a stated action at the boundary, "TTL" is a
   comment rather than a policy.

**The join is the product.** Treat the point-in-time join as the deliverable and the store as
plumbing. A team that can produce, for any entity and cutoff, the exact row set that was visible can
rebuild any table in the system. A team that cannot, but owns a fast key-value tier, has a demo.

**Three questions before adopting anything.** First, is the decision actually latency-bound? If
predictions are produced hourly in a batch and consumed by a queue, a nightly table and an indexed
SQL read are sufficient and a low-latency tier adds an operational surface with no user-visible
benefit. Second, does the model need history at all, or only the current state? Current-state-only
models need no point-in-time join and therefore no feature store; they need a well-defined snapshot
table. Third, who owns the definitions? A store without an owner converts every schema question into
a cross-team negotiation and definitions drift even when the platform is healthy.

**Where the failure surface lives.** Four layers, and they fail with distinguishable signatures:

- Layer 1, event capture: rows never arrive, arrive twice, or arrive with wrong timestamps.
- Layer 2, window and aggregation: the filter is right but the function is wrong (non-additive
  aggregates, deduplication, null handling).
- Layer 3, materialization: the online value is behind, ahead, or from a different partition.
- Layer 4, consumption: the model, the request builder, or the fallback path uses a different unit,
  default, or feature order than the definition.

Layer 4 is the cheapest to fix and the most common. Layer 1 is the most expensive and the easiest to
detect with a lag histogram. Layer 2 is the one that survives review because the SQL is plausible.

**The duality that must hold.** For any entity and cutoff sampled from production traffic, the
historical table value and the value that was served must agree within a stated tolerance. This is
not a nice-to-have test; it is the definition of correctness for the pair of paths, and Section 4.2
turns it into a measurement with a tolerance and a sample size.

**Cost shape of each layer.** Raw history grows with events per day times retention. The historical
feature table grows with entities times cutoffs materialized, which is usually far smaller because
you materialize only the cutoffs you train on. The online table grows with entities times bytes per
vector and is bounded by memory prices. The serving log grows with predictions and is the only table
that must retain the exact vector, so it is the one that quietly becomes the largest.

**A mental model that survives contact with production.** Think of the system as a photograph
archive, not a live feed. Each prediction is a photograph taken at `t_c`; the job is to develop that
photograph from negatives that were in the building at `t_c`. A negative that arrives by courier the
next morning cannot appear in yesterday's photograph even though it depicts yesterday. Most leakage
bugs are exactly that courier problem, and most are introduced by a join that filters on the
depicted moment instead of the arrival moment.

## 3. Foundational Definitions

Each definition below states what the term means operationally, why a designer cares, and the
concrete symptom when the term is used loosely. Terms are ordered so that later entries depend only
on earlier ones.

**Entity.** The unit about which features are computed: a card, an account, a merchant, a device, a
session, or a user. *Why it matters:* the entity determines the key, the fan-out of the join, and
the meaning of "the same entity over time." *Symptom when vague:* two teams compute the same named
feature at user grain and at account grain, then average across a training set that mixes both.

**Entity key.** The concrete column used for joining, with an explicit type, normalization, and
namespace. *Why it matters:* `id = 12345` in two source systems is a coincidence, not a join.
*Symptom:* aggregates computed on the wrong population, typically visible as a bimodal distribution
where you expect one mode.

**Event time (`t_e`).** The instant the represented fact occurred in the world, as reported by the
source. *Why:* windows, labels, and point-in-time filters are all expressed in event time.
*Symptom:* a timezone-naive timestamp stored as UTC shifts every window by hours and quietly moves
boundary rows between adjacent days.

**Ingestion time (`t_i`).** The instant the pipeline first durably observed the row. Also called
processing or arrival time. *Why:* it is the only timestamp that tells you what was knowable.
*Symptom:* if `t_i` is populated from the source record rather than from the ingest boundary, you
cannot reconstruct visibility and the availability filter becomes decorative.

**Effective time (`t_v`).** For slowly-changing attributes such as plan tier or risk band, the
instant from which a value applies, plus an end bound. *Why:* joining today's tier onto last
quarter's transactions reverses causality. *Symptom:* a feature that appears predictive offline and
useless online, because the attribute changed in ways the training set could not have anticipated.

**Prediction cutoff (`t_c`).** The instant at which the decision is made and no later information is
permitted. *Why:* it defines the half-open window `[t_c - W, t_c)` and the filter `t_i <= t_c`.
*Symptom:* cutoffs derived from the label date rather than from the decision date shift by the
label maturity horizon, which is exactly the leakage Section 4.1 quantifies.

**Point-in-time join.** A join that, for each `(entity, t_c)`, selects from history only rows with
`t_e < t_c` and `t_i <= t_c`, then aggregates. *Why:* it is the mechanism that makes offline
evaluation honest. *Symptom when implemented as a plain join:* every row in the window is used
regardless of arrival, which inflates recent activity precisely for the entities with the most
interesting recent behaviour.

**Availability filter.** The `t_i <= t_c` half of the join, sometimes expressed as a grace period
`t_i <= t_c - G`. *Why:* it converts "was the event inside the window" into "was the event visible."
*Symptom when omitted:* the aggregate includes rows produced after the decision, and the leak
concentrates on late-reporting sources, which are usually the ones correlated with fraud or failure.

**Grace period (`G`).** A deliberate exclusion band before `t_c` used when the pipeline cannot
guarantee that all rows with `t_i <= t_c` were actually visible by `t_c`. *Why:* it trades a known
amount of information loss for a bound on silent leakage. *Symptom when set by folklore:* either
most of the recent signal is discarded, or the leak persists because `G` is smaller than the tail of
the arrival-lag distribution.

**Label window.** The interval after `t_c` in which the outcome is observed, for example
`[t_c, t_c + 30 days)`. *Why:* it must be disjoint from the feature window in the sense that no
feature may reference the labelled outcome. *Symptom:* a feature named after a post-decision process
— dispute filed, chargeback raised — that correlates beautifully and cannot exist at serving time.

**Freshness.** The age of the value a consumer receives, `t_now - t_i` of the row that produced it.
*Why:* freshness is a per-read property, not a per-table property, and it differs between a hot
entity written seconds ago and a cold entity written a month ago. *Symptom:* a single table-level
"freshness" number that hides a long tail, so the SLA holds on average and fails on the entities the
model cares about.

**Staleness bound / TTL.** The maximum age beyond which a value is refused or replaced by a default.
*Why:* with a refusal policy the bound determines the null rate, which Section 4.3 derives from the
arrival process. *Symptom:* TTL chosen to fit memory rather than to fit the survival curve, producing
a null rate that pushes the model onto its fallback branch for a large population.

**Materialization.** The act of computing feature values ahead of request time and writing them to a
read-optimized store. *Why:* it converts compute cost into storage and latency. *Symptom:* partial
materialization after a failure leaves some partitions fresh and others days old with no marker in
the value, so consumers cannot tell which they received.

**Backfill.** Recomputing historical feature rows, typically after a definition change or a bug fix.
*Why:* backfill is where non-idempotent pipelines corrupt training data. *Symptom:* a re-run that
doubles a count because the write is an append rather than an upsert keyed by entity and cutoff.

**Training-serving skew.** Any systematic difference between the distribution of a feature in
training and at serving. *Why:* it degrades a model silently, with no error and no crash.
*Symptom:* offline AUC stable and high while online precision falls; the parity check in Section 7
localizes it to a layer.

**Feature drift.** Change over time in the feature's distribution in production, as distinct from
skew between paths. *Why:* drift is an input-distribution problem and skew is an implementation
problem; the remedies are different. *Symptom:* parity checks pass, model performance decays,
monitoring on the feature marginal shows movement.

**Serving log.** The record of the exact vector sent to the model, with its timestamp, version, and
request identity. *Why:* it is the only ground truth for diagnosing parity after the fact, and it is
the substrate for training on production distribution. *Symptom when absent:* every skew
investigation becomes a comparison of two systems reconstructed from code, which is where the
reasoning error happened in the first place.

**Feature version.** An immutable identifier for a definition plus its dependencies. *Why:* a
training set is only reproducible if it names versions, not names. *Symptom:* a model retrained six
months later on "the same feature" that has changed meaning twice, with no record of which meaning
the original model learned.

**Cardinality and hot keys.** The number of distinct entities, and the concentration of traffic on a
few of them. *Why:* online memory is proportional to cardinality and tail latency is governed by the
hottest key. *Symptom:* a single celebrity merchant serializes reads on one shard while aggregate
utilization stays low.

**Slowly-changing dimension (SCD).** An attribute table where values change over time and history is
retained with validity intervals. *Why:* it supplies `t_v`, and without it the join has only two
timestamps when it needs three. *Symptom:* as-of joins silently use the current row, which is the
most common form of effective-time leakage.

**Non-decomposable aggregate.** A statistic that cannot be maintained from a fixed-size summary plus
one new row: medians, percentiles, distinct counts, ranks, min and max over a window that expires
rows. *Why:* it determines whether incremental materialization is possible or a full window re-read
is required. *Symptom:* a "real-time" percentile that actually reflects the last batch window, and a
backfill that changes historical values after the fact.

**Idempotent write.** A write that produces the same stored state when applied twice. *Why:* retries
are unavoidable, and at-least-once delivery is the normal contract. *Symptom:* duplicates inflate
counts and sums, and the inflation is proportional to the retry rate, so it looks like a gradual
drift rather than a bug.

Now that the vocabulary is fixed, the derivations can be stated precisely: everything that follows
is a consequence of the availability filter and of the arithmetic of windows.

## 4. Core Formulas with Full Derivation

The section derives four results that a design review will ask about: what a legal historical
aggregate is, how to measure agreement between the two paths, how large a grace period or TTL
should be, and where each derivation stops being true. Each result uses the same anatomy —
FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION — and ends with its validity
condition, because the validity condition is what an interviewer probes.

### 4.1 Base case

**FORMULA.**

```text
f(e, t_c) = AGG { x_r  :  r in H,
                        r.entity   = e,
                        t_c - W   <= t_e(r) <  t_c,
                        t_i(r)    <= t_c }

with the deterministic-lag special case  L(r) = t_i(r) - t_e(r)  constant:

        t_i(r) <= t_c   <=>   t_e(r) <= t_c - L
   so   f(e, t_c) = AGG { x_r : t_c - W <= t_e(r) <= t_c - L }
```

**WHERE.**

- `e` — entity key, a normalized string or integer with an explicit namespace.
- `t_c` — prediction cutoff, a UTC instant. Window is half-open: `[t_c - W, t_c)`.
- `W` — window length in hours; `W > 0`. For the examples `W = 168` h (7 days).
- `t_e(r)` — event time of row `r`, UTC instant. `t_i(r)` — ingestion time, UTC instant.
- `L(r)` — lag in hours; `L(r) = t_i(r) - t_e(r) >= 0` by construction of the ingest boundary.
- `x_r` — the aggregated quantity; for a spend sum, USD with 2 decimals in the source.
- `H` — the event history table, append-only, deduplicated on `(source_id, t_e, amount_minor)`.
- `AGG` — `SUM`, `COUNT`, `COUNT DISTINCT`, `MAX`, or a windowed statistic. Section 4.4 shows
  why the choice changes everything about backfill.

**CONSTRAINT.** The aggregate must be a function of the *visible* set, and the visible set must be
monotone in `t_c`: for `t_c1 < t_c2`, every row legal at `t_c1` is legal at `t_c2` unless it has
aged out of the window. Monotonicity is what makes the historical table reproducible; if a row can
become legal *and then illegal* for a later cutoff, the definition is not point-in-time.

**NUMERICAL EXAMPLE (ILLUSTRATIVE).** Take `W = 168` h and a fixed lag `L = 26` h. The usable span
of event times is `168 - 26 = 142` h, which is `142/168 = 84.52%` of the nominal window. With a
uniform event rate of `lambda = 40/24 = 1.6667` events/hour, the nominal window contains
`1.6667 * 168 = 280.0` events, and the legal window contains `1.6667 * 142 = 236.7` events. So a
naive join that ignores `t_i` over-counts by `43.3` events, i.e. `280.0/236.7 = 1.183` times the
correct count, or `+18.3%` on every entity with a uniform rate.

Now let the lag be exponential with mean `mu = 8.656` h — chosen so the median lag is 6 h, since
`mu = 6 / ln 2`. The expected visible fraction is

```text
E[visible] = lambda * INTEGRAL_0^W (1 - exp(-u/mu)) du
           = lambda * [ W - mu * (1 - exp(-W/mu)) ]
           = 1.6667 * [168 - 8.656 * (1 - exp(-19.41))]
           = 1.6667 * [168 - 8.656 * (1 - 0.0000)]
           = 1.6667 * 159.344
           = 265.6 events
```

Against 280 nominal, the loss is `(280 - 265.6)/280 = 5.15%`, not the 15.5% a fixed-lag reading
would suggest. Both numbers are correct for their own lag model, and the gap between them is the
first quantitative lesson: *the lag distribution, not its maximum or its mean, sets the loss.*

**INTERPRETATION.** The availability filter is not an optimization; it is part of the definition of
the feature. Its cost is measured in information discarded, and that cost is governed by how much
probability mass of the lag distribution exceeds zero inside the window. A team that skips the
filter does not get 18% more signal; it gets 18% of its aggregate computed from the future, and the
excess concentrates on exactly the entities whose late-arriving rows are interesting.

**VALIDITY CONDITION.** The base case holds when (a) `t_i` is recorded at the durable ingest
boundary, not copied from the source payload; (b) the event history is deduplicated before
aggregation; (c) `AGG` is decomposable so that the value is a function of the row set alone; and
(d) `t_e` and `t_c` share one timezone and one calendar. Fail any of the four and the arithmetic
above describes a different quantity than the one you computed.

### 4.2 General case

**FORMULA.** Define the parity residual for entity `e` and cutoff `t_c` as the relative difference
between the historical value and the value actually served, and test it against a tolerance on a
sample of production keys:

```text
delta(e, t_c) = ( v_hist(e, t_c) - v_serve(e, t_c) ) / max(|v_hist(e, t_c)|, eps)

PASS  <=>  share{ |delta| <= tau } >= 1 - alpha

sample size for a mismatch-rate estimate p with absolute precision eps_p:
        n = p (1 - p) (z_{1-a/2} / eps_p)^2
```

**WHERE.**

- `v_hist` — value from the historical feature table at the same `(e, t_c)`.
- `v_serve` — value returned by the online read, captured in the serving log, same request.
- `eps` — a floor on the denominator, e.g. `eps = 1.0` USD for a spend feature, so that a
  difference from 0.00 to 0.01 does not produce a residual of infinity.
- `tau` — tolerance, a dimensionless relative bound; `alpha` — acceptable share of violations.
- `p` — true mismatch rate; `z_{1-a/2}` — normal quantile, 1.959964 at 95% two-sided.
- `n` — number of sampled keys, integer; `eps_p` — half-width of the confidence interval.

**CONSTRAINT.** The residuals must be computed on the *same* cutoff, not merely the same calendar
day, and the sampling frame must be production traffic rather than a curated entity list. If you
sample entities that all traded yesterday, you test the materialization path and never the
staleness path.

**NUMERICAL EXAMPLE (ILLUSTRATIVE).** Suppose a smoke test on 20,000 production keys finds 400
mismatches, so `p_hat = 400/20,000 = 0.02`. The 95% confidence half-width is
`1.959964 * sqrt(0.02*0.98/20000) = 1.959964 * sqrt(0.00000098) = 1.959964 * 0.000990 = 0.00194`,
so the interval is `[1.81%, 2.19%]`. To resolve that rate to `+/- 0.5%` absolute you need

```text
n = 0.02 * 0.98 * (1.959964 / 0.005)^2
  = 0.0196 * 153,664.0
  = 3,011.8  ->  3,012 keys
```

and to resolve it to `+/- 0.2%` you need `0.0196 * (979.98)^2 = 18,824` keys. The factor of 6.25
in sample size for a 2.5x tighter interval is just `(0.005/0.002)^2 = 6.25`; the point is that
"we sampled a thousand requests and they all matched" cannot distinguish a 2% defect from a
0.5% defect.

Now translate a residual into a modelling consequence. Take a perfectly separating binary
indicator: present with probability 0.95 for positives and 0.05 for negatives. Its single-feature
AUC is `0.5 * (TPR + TNR) = 0.5 * (0.95 + 0.95) = 0.95`. A parity defect that flips 5% of the
indicator in one direction moves this to somewhere between 0.90 and 0.95 depending on which class
is corrupted — a change that is visible but not catastrophic. The reason parity errors are treated
as severe is not that they destroy AUC; it is that they destroy *calibration* on the subpopulation
where the online value is stale, and the decision threshold was tuned on the historical marginal.

**INTERPRETATION.** Parity is a measurable invariant with a tolerance, a population, and a sample
size — not a slogan. State `tau`, `alpha`, `eps`, the sampling frame, and the frequency, and the
check becomes a regression test that a platform team can own. Leave any of them unspecified and
the check degenerates into a dashboard that everyone has learned to ignore.

**VALIDITY CONDITION.** Valid when the historical table and the online table are produced from the
same definition version for the compared keys; when the residual is computed per entity rather than
on aggregates (aggregate comparison cancels errors of opposite sign and will pass while every
individual value is wrong); and when `eps` is chosen per feature unit so that near-zero values do
not dominate the violation count.

### 4.3 Variants

Three variants cover almost all design questions: choosing a grace period when lag is uncertain,
choosing a TTL for the online table, and choosing an aggregation granularity for cost.

**FORMULA (grace period).** With leak cost `alpha` per event exposed after the cutoff, information
cost `beta` per hour of excluded recency, and an exponential lag of mean `mu`:

```text
C(G) = alpha * E * exp(-G/mu) + beta * G

dC/dG = -(alpha*E/mu) * exp(-G/mu) + beta = 0
  =>   G* = mu * ln( alpha * E / ( beta * mu ) )
```

**WHERE.** `E` — expected number of events per entity-window (dimensionless count, 280 above);
`alpha` — cost per leaked event, in the same notional unit as `beta`; `beta` — cost per hour of
recency excluded, so `beta*G` and the first term share units; `G` — grace in hours, `G >= 0`;
`mu` — lag mean in hours. Use `G* = 0` when the logarithm is negative, i.e. when
`alpha*E <= beta*mu`.

**NUMERICAL EXAMPLE (ILLUSTRATIVE).** `E = 280`, `mu = 8.656`, `alpha = 0.05`, `beta = 0.02`:

```text
G* = 8.656 * ln( 0.05 * 280 / (0.02 * 8.656) )
   = 8.656 * ln( 14.0 / 0.17312 )
   = 8.656 * ln( 80.87 )
   = 8.656 * 4.3928
   = 38.02 h

check:  C(38) = 0.05*280*exp(-38/8.656) + 0.02*38
              = 14.0*0.01245 + 0.76
              = 0.1743 + 0.76 = 0.934
        C(24) = 14.0*0.0625 + 0.48 = 0.875 + 0.48 = 1.355
        C(48) = 14.0*0.00391 + 0.96 = 0.0547 + 0.96 = 1.015
```

The curve is flat near the optimum — 38 h costs 0.934, 48 h costs 1.015, a 8.7% penalty for a
26% longer grace — which is why arguing about the exact value is less useful than agreeing on
`alpha` and `beta`. Disagreement about `G*` is almost always disagreement about the two costs.

**FORMULA (TTL and null rate).** If online values for an entity are refreshed by a Poisson process
with rate `r` per hour and reads occur at arbitrary times independent of refresh, the probability
that a read finds no live entry under TTL `T` hours is `exp(-r*T)`, and the null share is

```text
null_share(T) = exp(-r * T)
```

`r` in refreshes/hour, `T` in hours, both positive; result dimensionless. At `r = 0.5/day = 0.02083/h`:
`T = 7 d` gives `exp(-0.5*7) = exp(-3.5) = 0.0302`, so a **3.02%** null rate; `T = 14 d` gives
`exp(-7) = 0.00091`, so **0.09%**, for double the memory. If instead entities refresh only when
they transact and 25% of entities are dormant, `r = 0.25/day` and `T = 7 d` gives
`exp(-1.75) = 17.4%` nulls — the same TTL with a different arrival process is not the same
availability policy.

**FORMULA (fan-out cost).** With `N` entities, `B` bytes per entity per view, and `V` views, the
online footprint is `N*B*V` and the read amplification at `Q` requests per second is `Q*V`
point lookups per second. For `N = 25e6`, `B = 120`, `V = 6`: `25e6 * 120 * 6 = 18.0 GB` of values
before replication and index overhead; at `Q = 3,000` req/s the tier must sustain
`18,000` lookups/s, which is roughly one moderately sized node at `20,000` lookups/s, i.e. no
headroom. Cutting `V` from 6 to 3 halves both memory and lookup rate and is a design change worth
more than any cache tuning.

**INTERPRETATION.** Each variant converts a qualitative design debate into one number to agree on:
`alpha/beta` for grace, `r` for TTL, `V` for footprint. Publish those three parameters next to the
feature definitions and the debates stop recurring.

**VALIDITY CONDITION.** The grace derivation assumes exponential lag and a linear recency cost; a
heavy-tailed lag invalidates it (Section 4.4). The TTL derivation assumes independent Poisson
refresh, which fails after a partial outage — post-incident refreshes are correlated, so the null
rate spikes far above the formula for the duration of the catch-up. The fan-out formula counts
values only; add replicas, per-key overhead, and allocator slack before quoting a memory bill.

### 4.4 Limits and counterexamples

**Counterexample 1 — non-decomposable aggregates break immutability.** Suppose the feature is
"distinct merchants transacted with in 30 days." A row with `t_e = day 0` and `t_i = day 3` is
legal at every cutoff from day 3 onward, and it can change the value at each of those cutoffs if
its merchant is otherwise absent. With hourly cutoffs, one late row can revise
`30 * 24 = 720` historical cells. The formula in 4.1 is still correct — the aggregate is a function
of the visible set — but the operational assumption that historical feature rows are immutable
after write is false. Any pipeline that appends feature snapshots and never revisits them will
serve stale distinct-counts forever, and any pipeline that rewrites them needs a defined
supersede rule.

```text
affected cutoffs for a late row r  =  { t_c : t_c >= t_i(r) and t_c - W < t_e(r) }
                                   =  t_c in [ t_i(r), t_e(r) + W )
count with hourly cutoffs         =  W - L   (hours),  = 142 above
```

**Counterexample 2 — a copied ingestion timestamp destroys the filter.** If `t_i` is set from the
source record's own "received" field, and the source backfills three days of records after an
outage, those records carry `t_i` values in the past. The filter `t_i <= t_c` then admits rows that
were physically absent at `t_c`, and the leak is proportional to the backfill size, not to the lag
distribution. The discriminating check is structural rather than statistical: compare the
distribution of `t_i - ingest_wallclock`; a healthy pipeline has a tight, non-negative
distribution. Any mass at values below the deployment date of the ingest schema indicates copied
timestamps.

**Counterexample 3 — Poisson thresholds on a mixture.** A monitoring rule that alerts when a
count exceeds `mean + 4*sqrt(mean)` assumes Poisson variance. Take a population that is 90% light
users at `0.2` events/day and 10% heavy users at `20` events/day, over `W = 7` days:

```text
mean count  = 0.9*(0.2*7) + 0.1*(20*7) = 0.9*1.4 + 0.1*140.0 = 1.26 + 14.00 = 15.26
E[lambda*W] = 15.26
Var(lambda*W) = 0.9*(1.4 - 15.26)^2 + 0.1*(140.0 - 15.26)^2
              = 0.9*192.10 + 0.1*15,560.1
              = 172.89 + 1,556.01 = 1,728.90
Var(count)  = E[Var|lambda] + Var(lambda*W) = 15.26 + 1,728.90 = 1,744.16
sd(count)   = 41.76          vs Poisson sd = sqrt(15.26) = 3.91
variance/mean = 114.3
```

A Poisson rule would flag any entity above `15.26 + 4*3.91 = 30.9` events; that is roughly the
median of the heavy segment, so the rule alerts on most of the 10% heavy tail and nothing else.
The dispersion index of 114 is the diagnostic: compute `Var/mean` per segment before writing a
threshold. Alert on a per-segment quantile or on a robust z-score against the segment's own
distribution, never on a global Poisson band.

**Counterexample 4 — heavy tails invalidate the exponential grace.** With lag modelled as Pareto
with shape `k = 1.5` and scale `x_m = 0.5` h, the mean is
`x_m*k/(k-1) = 0.5*1.5/0.5 = 1.5` h, yet `P(L > 168) = (0.5/168)^1.5 = 0.0001624`, i.e. `1.62`
rows per ten thousand. With `1.4e6` events/day that is still about `227` rows per day whose
lag exceeds the entire feature window. Those rows are not merely lost to the window; they arrive
after the cutoffs at which they would have mattered, so a grace period tuned on the mean
under-protects them by construction, and the entities affected are the ones whose upstream feed
was degraded — usually a specific merchant or processor, which makes the leak cluster in a
subpopulation instead of spreading evenly.

**Counterexample 5 — window boundary off-by-one changes the count.** With a half-open window
`[t_c - W, t_c)` and events exactly hourly, the count per entity is `W - 1`, not `W`. For
`W = 168` that is a 0.6% discrepancy, invisible per entity and systematic across the table.
Worse, mixing a closed convention in one feature and a half-open convention in another makes the
ratio of two features biased by a factor that grows as `W` shrinks; at `W = 24` the same
off-by-one is 4.2%.

**INTERPRETATION.** The derivations in 4.1 through 4.3 are correct only under their stated
assumptions, and production violates them in four recurring ways: aggregates that are not
decomposable, timestamps that were copied rather than recorded, distributions that are mixtures
disguised as single populations, and conventions that differ between features. Each violation is
detectable by a specific, cheap check — supersede rules, ingestion-clock skew, per-segment
dispersion, and a boundary unit test.

**VALIDITY CONDITION FOR THE WHOLE SECTION.** Every formula here presupposes that a legal row set
is well defined for `(e, t_c)`. That presupposition fails when entity keys are recycled (a card
number reissued to a different account), when entity history is merged or split without a
mapping table, or when the cutoff is expressed in a local calendar whose offset changed inside the
window. In those three cases the correct repair is not a better formula but an entity-resolution
table and a stored UTC cutoff, and no amount of arithmetic on the aggregate will recover the
missing distinction.

## 5. Worked Numerical Example

The scenario is one entity, 13 events, and two cutoffs. Every number below is ILLUSTRATIVE and was
recomputed from the ledger; no operational system produced it.

### 5.1 The setup

Feature definition, version `card_spend_7d_v2`:

```text
feature card_spend_7d  (entity: card_id, unit: USD, window W = 168 h)
  filter  t_e >= t_c - 168h  AND  t_e < t_c  AND  t_i <= t_c
  value   SUM(amount_usd)
  null    NULL when zero rows are visible (never 0.00)
```

Entity `C-4471`. The source table is append-only with columns `card_id`, `ts_event`, `amount_usd`,
and `lag_h`, where `lag_h = t_i - t_e` is measured at the ingest boundary — not copied from a
source-supplied field. Cutoff `t_c = 2026-03-07T00:00Z`.

| # | ts_event (UTC) | amount USD | lag_h | `t_i` (event + lag) | in window `[02-28T00, 03-07T00)` | visible at `t_c` |
|---|---|---|---|---|---|---|
| 1 | 2026-02-27 21:10 | 12.40 | 0.30 | 02-27 21:28 | no (before window) | yes |
| 2 | 2026-02-28 08:05 | 88.10 | 0.15 | 02-28 08:14 | yes | yes |
| 3 | 2026-03-01 09:15 | 40.00 | 0.10 | 03-01 09:21 | yes | yes |
| 4 | 2026-03-02 18:40 | 125.50 | 0.20 | 03-02 18:52 | yes | yes |
| 5 | 2026-03-03 22:55 | 7.60 | 1.05 | 03-03 23:58 | yes | yes |
| 6 | 2026-03-04 11:05 | 62.25 | 0.30 | 03-04 11:23 | yes | yes |
| 7 | 2026-03-04 19:20 | 33.80 | 31.5 | 03-06 02:50 | yes | yes (02:50 < cutoff) |
| 8 | 2026-03-05 22:30 | 18.75 | 26.0 | 03-07 00:30 | yes | **no** (after cutoff) |
| 9 | 2026-03-06 08:00 | 210.00 | 0.07 | 03-06 08:04 | yes | yes |
| 10 | 2026-03-06 12:45 | 9.90 | 2.40 | 03-06 15:09 | yes | yes |
| 11 | 2026-03-06 23:50 | 54.30 | 0.05 | 03-06 23:53 | yes | yes |
| 12 | 2026-03-07 06:12 | 19.99 | 0.05 | 03-07 06:15 | no (after cutoff) | no |
| 13 | 2026-03-08 14:10 | 75.00 | 0.10 | 03-08 14:16 | no (after cutoff) | no |

Row 7 is the instructive one: a 31.5-hour lag, but it landed at 02:50 on 03-06, which is before
the cutoff, so it *is* legal. Long lag does not imply invisible; only `t_i > t_c` does. A designer
who filters on "lag <= some maximum" instead of on `t_i <= t_c` would wrongly drop row 7 and keep
row 8 — inverting the correct decision on both.

### 5.2 The four computed values

```text
A  naive sum, all 13 rows, no filters
   = 12.40 + 88.10 + 40.00 + 125.50 + 7.60 + 62.25 + 33.80 + 18.75
     + 210.00 + 9.90 + 54.30 + 19.99 + 75.00
   = 757.59 USD          (13 rows, mean 58.28 USD)

B  event-window sum, filter on t_e only  (rows 2..11)
   = 757.59 - 12.40 (row 1) - 19.99 (row 12) - 75.00 (row 13)
   = 650.20 USD          (10 rows, mean 65.02 USD)

C  point-in-time sum, filter on t_e AND t_i  (rows 2..7, 9..11)
   = 650.20 - 18.75 (row 8)
   = 631.45 USD          (9 rows, mean 70.16 USD)      <- correct value

D  contamination of each wrong answer, relative to C
   (B - C)/C = 18.75/631.45 = 2.97%      late-row leakage left by a t_e-only join
   (A - C)/C = 126.14/631.45 = 19.98%    naive join, future rows included
   (A - B)/B = 107.39/650.20 = 16.52%    future rows plus the aged-out row 1
```

The three gaps are different defects. `A - B` mixes two causes: rows 12 and 13 had not happened
(a genuine future leak of `19.99 + 75.00 = 94.99`) and row 1 had aged out (an overcount of
`12.40`). `B - C` is the pure arrival leak. A team that "adds a window filter" fixes 16.52% of the
contamination and, if it stops there, ships a model trained on the remaining 2.97% plus whatever
other entities contribute on top.

Row-count and amount-count views disagree, which is a useful sanity constraint:
`1/10 = 10.0%` of window rows were late, but those rows carry only `8.08%` of window dollars
(`18.75/650.20`). Late rows here are smaller than average, so a count-based leak estimate
overstates the dollar impact by `10.0/8.08 = 1.24x`. Neither statistic is "the" leak; state which
one you measured.

### 5.3 Independent re-check by a different route

Enumerate and sum the visible rows directly, then confirm by subtraction:

```text
visible rows = 2, 3, 4, 5, 6, 7, 9, 10, 11        (nine rows)

 88.10
+40.00  = 128.10
+125.50 = 253.60
+  7.60 = 261.20
+ 62.25 = 323.45
+ 33.80 = 357.25
+210.00 = 567.25
+  9.90 = 577.15
+ 54.30 = 631.45        matches C by direct sum ✓

cross-check via subtraction:
  window sum 650.20 - excluded row 8 (18.75) = 631.45 ✓
  all rows   757.59 - (12.40 + 19.99 + 75.00 + 18.75)
            = 757.59 - 126.14 = 631.45 ✓
```

Three routes agree, and the partial sum after row 7 (`357.25`) is the number the grace-period
variant in 5.4 will reuse. The re-check is not decoration: the first draft of this example
subtracted the late row twice and produced 597.65, which no single route reproduces. Enumerate,
then subtract; when the two disagree, the visible set is miscounted, and the error is almost never
in the arithmetic.

### 5.4 Sensitivity: what changes the answer

| Change | Mechanism | Recomputed value | Rows visible |
|---|---|---|---|
| baseline `W = 168`, `G = 0` | reference | 631.45 | 9 |
| `W = 72 h` | window becomes `[03-04T00, 03-07T00)`; rows 2–5 leave | 370.25 | 5 |
| grace `G = 26 h` | also requires `t_e <= 2026-03-05T22:00`, dropping rows 9–11 | 357.25 | 6 |
| `W = 336 h` | window widens to 14 days; row 1 enters | 643.85 | 10 |
| `log1p` then sum | monotone transform, same rows | 34.6454 | 9 |
| count instead of sum | row count, not dollars | 9 | 9 |
| cutoff `2026-03-06T00:00Z` | window `[02-27T00, 03-06T00)`; rows 6,7 late | 335.85 | 6 |

Two of these deserve arithmetic. **Grace 26 h** applies `t_e <= t_c - 26h = 2026-03-05T22:00`, which
removes rows 9 (`08:00`, 210.00), 10 (`12:45`, 9.90) and 11 (`23:50`, 54.30) — `274.20` USD in
total — from the nine visible rows. `631.45 - 274.20 = 357.25`, so this grace discards 43.4% of the
feature to protect against a lag tail it already handled correctly. Row 7 at `19:20` survives
because it is before the band, which is the point of using `t_e` and not `t_i` in the grace rule:
the grace is a policy about *when things happened*, while the availability filter is a fact about
*when they arrived*.

**Count versus sum:** nine rows are visible, so the count feature is `9`, and the two aggregates
differ only in unit — dollars versus rows — not in row set. Any divergence between a count and a
sum feature on the same window is a null-handling difference, not a filtering difference.

**Cutoff moved back one day:** window `[02-27T00, 03-06T00)` contains rows 1–8 for
`12.40 + 88.10 + 40.00 + 125.50 + 7.60 + 62.25 + 33.80 + 18.75 = 388.40`. At that cutoff both late
rows are invisible (row 7 becomes visible at 03-06 02:50, row 8 at 03-07 00:30), so the visible sum
is `388.40 - 52.55 = 335.85` and the leak rate is `52.55/335.85 = 15.65%` — 5.3 times the 2.97% at
the later cutoff, with the same feature definition and the same ledger.

That last figure is the design fact worth memorizing: **the leak rate is a property of the cutoff
relative to the lag mass, not of the feature.** A backtest whose cutoffs sit close behind the last
source update shows heavy contamination; one whose cutoffs sit a month back shows almost none. Both
can be run on the same week of data, and the difference in measured leakage between them explains
most "the model worked in backtest" regressions.

### 5.5 Recomputing the aggregate two ways in SQL

```sql
-- Route 1: point-in-time, explicit and auditable
SELECT l.card_id, l.t_c, SUM(e.amount_usd) AS card_spend_7d
FROM   labels l
LEFT   JOIN events e
  ON   e.card_id = l.card_id
 AND   e.ts_event >= l.t_c - INTERVAL '168 hours'
 AND   e.ts_event <  l.t_c
 AND   e.ts_ingest <= l.t_c
GROUP  BY l.card_id, l.t_c;

-- Route 2: row-count check on the same join; must agree with route 1's non-null rows
SELECT l.card_id, l.t_c, COUNT(e.amount_usd) AS n_rows
FROM   labels l
LEFT   JOIN events e
  ON   e.card_id = l.card_id
 AND   e.ts_event >= l.t_c - INTERVAL '168 hours'
 AND   e.ts_event <  l.t_c
 AND   e.ts_ingest <= l.t_c
GROUP  BY l.card_id, l.t_c;
```

Three details carry the correctness. The `LEFT JOIN` keeps entities with no visible history so
that the null policy is exercised rather than silently dropping them — an `INNER JOIN` here is the
single most common cause of "the training set has 3% fewer rows than the label table and we do not
know which 3%". `COUNT(e.amount_usd)` rather than `COUNT(*)` counts non-null amounts, so a
nullable amount column does not inflate the row count. And both routes must use the *same* filter
clause text; duplicating it in two queries invites the drift that Section 8 catalogs as the
"definition forked during review" failure.

## 6. Visual / Geometric Intuition

### 6.1 The window as a shifted strip

Draw time as an axis with the cutoff as a vertical line. A naive join uses the strip
`[t_c - W, t_c)`. The point-in-time join uses the same strip, minus a wedge at the right whose
width varies per row — the set of rows whose lag pushes their visibility past `t_c`.

```text
  t_e axis  ----------------------------------------------> t_c
  |----------------- W = 168 h -----------------|----------|
  [            usable span 142 h for L = 26 h  ][ lag 26 h ]
  |<----- rows a PIT join can use ------------>|<-- excluded -->
                                               \_ not yet visible at t_c _/

  a row at event time u = t_c - s, where s = age in hours:
     legal iff s >= L,  contributing iff  L <= s < W
     so it feeds exactly the cutoffs on the age band  [L, W)
```

The last line is the geometric content: a row of fixed lag `L` feeds `W - L` hours' worth of
cutoffs and is dead weight for the remaining `L` hours of its nominal window. For a degenerate lag
distribution the lost fraction is exactly `L/W`.

### 6.2 The leakage triangle

For a fixed window, mass splits into three regions relative to the cutoff and the lag boundary:

```text
                          t_c
                           |
    true feature     |  A  |  B  |     A: t_e < t_c - G, t_i <= t_c    (kept)
    (legal rows)     |     |     |     B: t_e in [t_c-G, t_c)          (discarded)
                     |-----+-----|     C: t_e < t_c, t_i > t_c         (leak if used)
                     |  C  |     |     D: t_e >= t_c                   (future leak)
                     +-----+-----+
                        D (future)
```

`B` shrinks as `G` shrinks and `C` grows as `G` shrinks; the two are locked in trade. `D` is
removable outright by any correct `t_e < t_c` filter, which is why future leakage is a beginner
error and arrival leakage is not: `C` survives a filter that looks right.

### 6.3 Latency budget as a bar

```text
|<-- 30 ms network+gateway -->|<- 10 auth ->|<- 15 model ->|<-- 65 ms features -->|
0                            30             40             55                   120
```

The 65 ms segment covers the slowest view, and sequential fetches add. Six views at 5 ms each are
30 ms serial but roughly 6 ms parallel (slowest reply plus fan-out coordination), which decides
whether a 120 ms budget holds. A percentile over 90 days is never 5 ms; it is a cache hit or a miss,
and the miss path needs its own line in the budget or a precomputed fallback value.

### 6.4 Staleness as a survival curve

```text
  share of
  reads
  1.00 |**************************************
  0.97 |                                   *    TTL 7 d, refresh 0.5/day -> 3.0% null
  0.83 |                             *          TTL 7 d, refresh 0.25/day -> 17.4% null
  0.50 |                    *
  0.00 +---------+---------+---------+-------->  age (days)
       0         2         4         6       8
```

Same TTL, different refresh rate, a 5.8x difference in null share. The curve makes visible that
"TTL = 7 days" is not a freshness policy without the arrival rate `r` and the action taken at the
boundary.

### 6.5 Cardinality versus QPS

```text
  QPS
  1e5 |                     *   hot-key / cache problem
      |            *
  1e4 |     *                   <- 25M entities x 6 views, 18k lookups/s
      |  *
  1e2 +--*------*------*------*----> entities
        1e4    1e6    1e8    1e10
```

`1e4` entities at `1e5` QPS is a caching problem; `1e9` entities at `1e2` QPS is a storage and
rebuild problem. Identical API, different engineering, different interview answer.

### 6.6 Where the four failure layers sit

```text
 events --> [L1 capture] --> [L2 window/agg] --> [L3 materialize] --> [L4 consume] --> decision
               |                    |                    |                   |
         lag histogram        row-set diff          parity residual      unit/default diff
         duplicate rate       count agreement       staleness hist.      feature-order diff
```

Each layer has one cheap observable that localizes a mismatch, ordered by collection cost: L4 is
free (read the serving log), L3 is one join, L1 and L2 need retained history. Diagnose in that
order and stop at the first layer that fails.

## 7. Comparison / Selection Table

### 7.1 Historical join strategies

| Strategy | Mechanism | Correct under late rows | Cost | Use when | Fails when |
|---|---|---|---|---|---|
| Plain equi-join + window filter | `t_e in [t_c-W, t_c)` | no | lowest | prototyping only | any late-arriving source exists |
| Availability-filtered join | adds `t_i <= t_c` | yes | one extra predicate, needs `t_i` stored | default choice | `t_i` was copied from source |
| Grace-windowed join | adds `t_i <= t_c - G` | yes, with bounded leak | loses `G` of recency | pipeline cannot guarantee visibility at `t_c` | `G` smaller than lag tail |
| As-of join on SCD | latest row with `t_v <= t_c` | yes for attributes | interval table + index | plan tier, risk band, merchant category | validity intervals overlap |
| Snapshot-per-cutoff table | precomputed rows per `(entity, cutoff)` | yes by construction | storage = entities × cutoffs | fixed training cadence, hourly cutoffs | cutoffs are ad hoc |
| Stream-state join | state at `t_c` from operator state | only if state is event-time and watermarked | highest complexity | sub-second freshness required | watermarks lag; state not queryable at arbitrary `t_c` |

Read the "fails when" column as the design decision. The availability-filtered join is correct in
most cases and costs one stored column; the grace variant exists to compensate for a pipeline that
cannot be trusted to have landed everything by `t_c`, and the snapshot table trades storage for
elimination of the join at training time. The two strategies that look attractive because they are
fast — plain join and stream state — are exactly the two whose failure conditions are ordinary
rather than exotic.

### 7.2 Online serving strategies

| Strategy | Freshness | Read latency (p99) | Cost driver | Consistency with history | Failure signature |
|---|---|---|---|---|---|
| Read-through to source DB | seconds | 20–200 ms | source DB load | none unless same SQL | source schema change breaks serving |
| Precomputed KV table, no TTL | as of last write | 1–5 ms | memory = N × B × V | good | silent infinite staleness |
| Precomputed KV table, TTL + default | bounded by TTL | 1–5 ms | memory + miss amplification | good within tolerance | null spike on cold entities |
| KV + on-miss compute | bounded, high variance | 5–500 ms | compute on miss | good | p99 collapse on correlated misses |
| Stream-maintained state | sub-second | 2–10 ms | cluster + state size | good if same operator code | restart loses state; rebuild is hours |
| Cached request assembly | TTL of cache | <1 ms on hit | cache tier | as of history run | stampede after invalidation |

The row that matters for most teams is the third: a precomputed table with a TTL and an explicit
default. It is cheap, it is diagnosable, and the only design work is stating the TTL from a null-rate
budget (Section 4.3) rather than from memory pressure. The fourth row is where p99 numbers die: on-miss
compute is fine on average and catastrophic when the miss set is correlated, which it always is after
a deploy that adds an entity segment.

### 7.3 Where to compute: batch, micro-batch, or streaming

| Property | Batch (hourly/daily) | Micro-batch (1–5 min) | Streaming (per event) |
|---|---|---|---|
| Decomposable aggregates (sum, count, max) | trivial | trivial | trivial |
| Non-decomposable (p50, p99, distinct, rank) | full re-read, cheap per run | expensive per run | requires sketch or window store |
| Cost per day at `1.4e6` events | one job, low | 288–1440 job starts | always-on cluster |
| Reproducibility | high (idempotent overwrite) | high with upsert key | low (state is not replayable at arbitrary `t_c`) |
| Leakage risk | low | low | high: no `t_c` boundary |
| Freshness of served value | hours | minutes | seconds |
| Operational surface | scheduler + warehouse | scheduler + smaller warehouse | cluster, checkpoints, watermark policy |

The asymmetry that decides most cases: streaming buys seconds of freshness and pays with the loss
of arbitrary-cutoff reproducibility. A model that is retrained monthly and scored hourly gains
nothing from sub-second features, and a feature store that cannot reproduce `t_c` cannot be used to
build the training set for its own features. Choose the coarsest cadence that meets the decision
latency, and keep the historical path batch even when the online path streams.

### 7.4 Choosing the entity key

| Candidate key | Join safety | History stability | Cardinality | Known hazard |
|---|---|---|---|---|
| Account number | high within one bank | stable | 1 per account | reissued numbers inherit prior history |
| Card PAN | high | unstable (reissue, replacement) | 1 per plastic | recycled PANs merge two customers |
| Tokenized PAN | high | unstable on token rotation | 1 per token | rotation splits history at each reissue |
| Customer ID | medium (shared accounts) | stable | 1 per person | one customer, many cards; spend aggregates double-count |
| Device fingerprint | low | unstable | inflated by resets | one user, many devices; one device, many users |
| Composite (account + product) | highest | stable | larger | join must carry both columns everywhere |

The default answer in an interview is "use the primary key of the system of record and never the
display identifier," and the reason is the third and fifth rows: any key that can be reissued,
rotated, or reset will silently attach one entity's history to another. When you must use a
composite key, say so explicitly and carry it through the online schema, because the most common
multi-team defect is one service joining on the account and another on the card.

### 7.5 Freshness versus correctness by decision type

| Decision | Tolerable staleness | Leakage tolerance | Recommended design |
|---|---|---|---|
| Real-time fraud authorization | seconds | ~0 (must be exact at `t_c`) | stream state or read-through, no grace, strict `t_i <= t_c` |
| Credit limit increase (daily) | hours | low | daily batch, `t_i <= t_c`, no grace |
| Churn propensity (weekly) | days | low | weekly snapshot table per cutoff |
| Next-best-offer (session) | minutes | moderate | micro-batch + on-miss default |
| Collections prioritization (daily) | a day | moderate (label is 30 d) | daily batch, grace `G` = p99 lag |
| Marketing lookalike (monthly) | weeks | low (no per-user action) | monthly snapshot, backfill freely |

The first row is the only one where "seconds of staleness" and "zero leakage" both bind, and it is
why streaming designs exist at all. The last row is the one where all the machinery is waste: no
per-user decision, no freshness requirement, and a monthly retrain. Most organizations have one
row-1 use case and twenty row-6 use cases, and build the platform for row 1.

### 7.6 Build, buy, or neither

| Option | Real cost | Real benefit | Hidden obligation |
|---|---|---|---|
| Scheduled SQL + indexed table | engineer-days, near-zero infra | reproducibility, auditability | on-call for the scheduler |
| Warehouse + materialized views | storage + warehouse credits | reuse, one definition | warehouse locking, backfill cost |
| Open-source store | cluster + upgrades + schema migration | online tier, discovery UI | version upgrade path, community support |
| Commercial store | licence + integration | vendor support, connectors | data egress, definition ownership ambiguity |
| Do nothing (model reads source directly) | zero platform | fastest to first model | serving latency, source coupling, no history |

The "do nothing" row is a legitimate choice for a single model with a batch decision, and the
contract's stance is that it should be considered rather than dismissed. A platform earns its cost
when the second and third models need the same aggregates with the same time semantics; until then
the store is infrastructure in search of a problem, and the obligations in the right-hand column are
what the team actually signs up for.

## 8. Failure Modes & Edge Cases

Each entry states the mechanism, the two or three competing causes that produce a similar symptom,
the discriminating check that separates them, the correction, and the guard that prevents
recurrence. The discriminating check is the part that matters in production: symptoms overlap
heavily and the checks are chosen to be cheap and mutually exclusive.

### FM-01 Future rows in the training set

**Mechanism.** The join filters on event time only, so rows created after `t_c` enter the aggregate.
Symptom: offline AUC much higher than any plausible online result; the gap widens as `W` shrinks
because recent rows dominate a short window.

**Competing causes.** (a) future rows via `t_e`; (b) label leakage, where a feature encodes the
outcome itself; (c) train/test split by random row rather than by time, so test rows share
entity-history with train rows.

**Discriminating check.** Recompute one feature with the correct filter and compare per-row values
on 1,000 entities: if values change, the join is at fault (a); if values match but AUC stays high,
inspect the top feature importances for post-decision columns (b); if both are clean, plot
performance by test-set month — a flat curve with a time split indicates (c) is not the cause, a
decaying curve indicates it is.

**Correction.** Add `t_i <= t_c`; re-derive the training set; retrain; report the AUC gap between
the leaky and clean runs, because the size of the gap is the size of the bug.

**Prevention.** A test fixture with a known late row whose inclusion changes the value; the test
fails if the join drops the filter.

**Verification.** For a sample of cutoffs, the count of rows with `t_i > t_c` in the joined set is
exactly zero.

### FM-02 Copied ingestion timestamps

**Mechanism.** `t_i` is populated from a source field (the upstream's own receive time, or a
backfilled default) rather than from the ingest boundary. The filter exists but admits rows that
were physically absent, so the leak reappears and is invisible in the SQL.

**Competing causes.** (a) copied `t_i`; (b) clock skew between ingest hosts; (c) a backfill job that
writes correct `t_e` but stamps `t_i = now()` for every replayed row.

**Discriminating check.** Plot `t_i - ingest_wallclock` per ingestion hour. Case (a): mass at
negative values, or a suspiciously round distribution copying the source. Case (b): a bounded,
host-dependent offset under a minute. Case (c): a single spike at the backfill's start time with
`t_i` clustered inside one hour while `t_e` spans months.

**Correction.** Record `t_i` at the boundary, keep the source timestamp as a separate column named
for what it is, and re-derive history from the raw landing table where it was retained.

**Prevention.** Schema constraint that `t_i >= t_e` and `t_i <= ingest_clock + 5 min`, enforced at
write time; a dashboard of the `t_i - wallclock` distribution as a first-class metric.

**Verification.** Zero rows violate the two constraints; the wallclock-offset histogram has no mass
below zero.

### FM-03 Non-idempotent backfill doubles counts

**Mechanism.** The feature write is an append (or a partitioned insert without overwrite), so a
re-run adds a second copy of every row for the reprocessed range. Sums double; counts double;
averages stay correct, which is what makes it hard to see.

**Competing causes.** (a) non-idempotent write; (b) genuine duplicate events at the source
(retried client submissions); (c) a join fan-out where the dimension table has two rows per key.

**Discriminating check.** Compare the mean and the count for the same population before and after
the backfill. Case (a) and (b): count rises, mean unchanged. Case (c): both count and mean change,
and the ratio of counts equals the fan-out factor. Then check whether the affected range matches
the backfill window exactly — (a) aligns to the job's range, (b) does not.

**Correction.** Rewrite as an upsert keyed by `(entity, t_c, feature_version)` or as a partition
overwrite; deduplicate on a stable natural key that includes the amount.

**Prevention.** Make the job declarative over a closed interval and assert row counts against the
source count for that interval before publishing.

**Verification.** Run the job twice; the table checksum is identical after both runs.

### FM-04 Partial materialization after a failed run

**Mechanism.** The job writes some partitions and fails; consumers read fresh values for the
succeeded range and day-old values elsewhere, with nothing in the value marking which is which.

**Competing causes.** (a) partial write; (b) clock skew causing one partition to be stamped an hour
early so it looks stale; (c) a consumer reading a replica that has not caught up.

**Discriminating check.** Group by partition and compute the max `t_i` in each. Case (a): a
contiguous range of partitions shares an old max, with a sharp boundary. Case (b): exactly one
partition is off by a constant, and the offset is under an hour. Case (c): the same partition shows
different max `t_i` on two reader instances.

**Correction.** Write to a staging table and swap atomically; stamp a `materialized_at` metadata row
per partition that consumers can read.

**Prevention.** Publish a freshness marker with every partition and have the serving layer refuse
values older than the contract, rather than silently serving them.

**Verification.** For every partition, `materialized_at` is within the contract window, and the
serving layer's refusal counter is zero in steady state.

### FM-05 Unit and scale mismatch between paths

**Mechanism.** Training divides by 100 (dollars) and serving does not (cents), or one path applies
`log1p` during the definition and the other applies it at the model input. All values are non-zero
and plausible; the model consumes a distribution shifted by a constant factor.

**Competing causes.** (a) unit mismatch; (b) transform applied twice; (c) model input order
different from the training column order, so features are permuted rather than rescaled.

**Discriminating check.** Compare the per-feature mean and standard deviation between the training
set and the serving log, per feature, as a table. Case (a): one feature's ratio is exactly 100 (or
a round power of ten) and the shape is unchanged. Case (b): the ratio is feature-specific and the
shape is compressed for a transformed feature only. Case (c): the means belong to the wrong
features — feature `i`'s serving mean equals feature `j`'s training mean.

**Correction.** Fix the definition to emit one canonical unit and document it; assert at the API
boundary; regenerate the training set rather than rescaling it in place.

**Prevention.** A schema-level unit annotation and a startup assertion comparing the first request's
vector against a stored reference vector within tolerance.

**Verification.** The per-feature mean ratio between the serving log and the latest training set is
within 1.05 for every feature, over at least one full traffic cycle.

### FM-06 Null versus zero default

**Mechanism.** The definition returns `0` for "no visible rows," so entities with no history and
entities with genuine zero activity become indistinguishable. The model learns from a mode that
does not exist at serving time, or the reverse.

**Competing causes.** (a) default chosen differently in the two paths; (b) `SUM` over an empty set
returning null in one engine and zero in another; (c) an `INNER JOIN` dropping entities that would
have produced the default.

**Discriminating check.** Count entities with the feature exactly at the default value, as a share
of population, in training and in the serving log. Case (a): shares differ materially. Case (b):
the training path's share is exactly the share of entities with no rows, and the serving path's is
zero. Case (c): the training set's row count is below the label table's, and the missing entities
are all cold-start.

**Correction.** Emit `NULL`, carry an explicit `is_missing` indicator, and let the model consume
both; never let the storage layer coerce.

**Prevention.** Contract test asserting that a cold-start entity returns `NULL` and that the
training builder emits one row per label row.

**Verification.** Training rows equal label rows; null share by segment matches between paths.

### FM-07 Hot-key latency collapse

**Mechanism.** One entity receives a large share of requests (a major merchant, a test account used
by load tests, a single institutional client). Its partition serializes reads, so p99 rises while
average CPU stays low.

**Competing causes.** (a) hot key; (b) an on-miss compute path triggered for cold entities; (c)
connection-pool exhaustion at the client rather than the store.

**Discriminating check.** Compute p99 latency as a function of request rate for the top-10 keys
separately. Case (a): only top keys degrade, and degradation scales with that key's rate. Case (b):
degrades with the share of *unique* keys, not with rate. Case (c): latency is uniform across keys
and both paths, and client queue depth shows saturation.

**Correction.** Replicate hot keys with a read-through cache keyed by hash plus a local copy; or
precompute hot-key vectors into a separate tier.

**Prevention.** Monitor per-key request concentration as a metric; alert when the top 0.01% of keys
exceeds a share threshold.

**Verification.** p99 for top-10 keys is within 2x of p99 for the median key at peak.

### FM-08 Non-decomposable feature silently rewritten

**Mechanism.** A percentile or distinct count is appended as a snapshot and never revised, so late
rows never reach history; or it is recomputed and older snapshots change, breaking the training set
already used for a model.

**Competing causes.** (a) snapshot append without supersede; (b) recompute-and-overwrite without a
version bump; (c) sketch precision too low, producing drift that looks like a revision.

**Discriminating check.** Recompute the feature for a fixed `t_c` two months apart. Case (a): the
value never changes even though late rows exist — verify by counting late rows for that entity.
Case (b): the value changes and no version changed. Case (c): the value changes by a small amount
and the change is bounded by the sketch's error parameter.

**Correction.** Version the definition, supersede under an explicit rule, and freeze the version
used for any training set that produced a deployed model.

**Prevention.** Feature rows carry `feature_version` and `computed_at`; training manifests record
the version; a change in either for the same `(entity, t_c)` is an error worth an alert.

**Verification.** For each training manifest, recomputing with the recorded version reproduces the
manifest's checksum.

### FM-09 Entity merge and split

**Mechanism.** Two accounts merge into one, or one is split, without a mapping table. History stays
under the old keys, so the merged entity's aggregate covers only part of its activity and every
window-based ratio shifts.

**Competing causes.** (a) missing mapping at merge; (b) key normalization change (case, whitespace,
leading zeros) applied to new data only; (c) recycled identifiers.

**Discriminating check.** Count distinct keys per real-world entity for a sample. Case (a): the
count jumps on merge dates. Case (b): a step change in cardinality on a deploy date, with old and
new formats both present. Case (c): a key's history contains two disjoint activity clusters
separated by a long gap, and the newer cluster starts with a different customer's first purchase.

**Correction.** Introduce an entity-resolution table mapping source keys to a canonical entity with
validity intervals; join through it and rewrite history once.

**Prevention.** Enforce normalization in a single ingestion function; record merges as events rather
than as mutations.

**Verification.** For a sample of merged entities, the aggregate computed via canonical key equals
the sum of the aggregates via source keys over the same window.

### FM-10 Watermark advanced past late rows

**Mechanism.** A streaming aggregation emits its result when a watermark passes, and rows arriving
afterwards are dropped rather than emitting a correction. The online value is therefore correct at
emission time and permanently low afterwards.

**Competing causes.** (a) dropped late rows; (b) an allowed-lateness window shorter than the lag
tail; (c) a restart that re-emits from the last checkpoint and overwrites with older state.

**Discriminating check.** Count late rows per window and compare with the count of corrections
emitted. Case (a): late rows exist, corrections are zero. Case (b): corrections exist but only for
rows within the allowed lateness; the excess is dropped. Case (c): the value moves *backwards* on a
restart date.

**Correction.** Emit retractions or idempotent corrections keyed by `(entity, window)`; make the
downstream upsert, not append.

**Prevention.** Set allowed lateness from the measured p99 lag; alert when drops exceed a rate
threshold.

**Verification.** Sum of emitted corrections equals the count of rows arriving after their window's
watermark, for a controlled replay.

### FM-11 Timezone and calendar drift

**Mechanism.** A local-calendar window ("last 7 days" in a market's timezone) is implemented as UTC
arithmetic, or a DST transition makes one day 23 or 25 hours inside a window, so boundary rows move
between cutoffs.

**Competing causes.** (a) timezone mismatch; (b) DST within the window; (c) leap-second or
epoch-unit confusion (milliseconds interpreted as seconds).

**Discriminating check.** Diff the row set between a UTC window and a local window for one market on
a normal week versus a DST week. Case (a): a constant offset of the market's UTC offset, so counts
differ on every week. Case (b): counts agree on normal weeks and differ by a few rows on the
transition weeks only. Case (c): all timestamps are off by exactly a factor of 1000.

**Correction.** Store UTC instants, compute windows in UTC, and use local calendars only for
presentation and for business rules that are genuinely calendar-based — which then need explicit
handling of the 23- and 25-hour days.

**Prevention.** Unit test with fixtures around both DST transitions, and a type that makes
timezone-naive timestamps unrepresentable.

**Verification.** Boundary tests on transition dates produce the documented counts.

### FM-12 Definition forked during review

**Mechanism.** The SQL reviewed and approved differs from the SQL deployed — an extra `OR`, a
changed `COALESCE`, a different window — because the definition lives in a query editor and the
deployment carries a copy.

**Competing causes.** (a) forked definition; (b) an override at the model layer (a scaler fitted on
the wrong split); (c) a serving-side clamp that truncates values above a bound.

**Discriminating check.** Extract the definition hash from the deployed artifact and compare with
the hash of the approved version; independently, compute the feature from the definition and
compare with the deployed value. Case (a): hashes differ. Case (b): hashes match and differences
appear only after the model layer, so raw values agree. Case (c): hashes match, raw values agree,
but the top of the distribution is truncated at exactly the clamp value.

**Correction.** Define once in a versioned artifact, generate both the batch and online
implementations from it, and record the hash in the serving log for every request.

**Prevention.** The serving log carries the definition hash; a monitoring job alerts when the hash
changes without a corresponding training-set regeneration.

**Verification.** For 100% of requests in a day, the logged hash equals the deployed artifact's hash.
## 9. Complexity & Cost Analysis

### 9.1 Cost model with all inputs stated

Take an ILLUSTRATIVE platform with these parameters, all chosen to be round and plausible rather
than measured:

```text
N  = 25,000,000 entities          (cards)
V  = 6 feature views              (spend, count, velocity, merchant-diversity, tenor, risk-band)
B  = 120 bytes per entity per view   (8 features x 12 bytes + key + version + timestamp)
E  = 1,400,000 events/day
W  = 168 h window
R  = 2 online replicas
P_mem = 0.05 USD per GB-hour of provisioned memory   (ILLUSTRATIVE)
```

### 9.2 Online footprint

```text
raw values      = N * B * V = 25e6 * 120 * 6 = 18.0e9 bytes = 18.0 GB
with 2 replicas = 36.0 GB
add 30% for per-key overhead, allocator slack, and index:
                36.0 * 1.30 = 46.8 GB
monthly memory  = 46.8 GB * 0.05 USD/GB-h * 730 h = 1,708 USD   (ILLUSTRATIVE)

halving V to 3:  25e6 * 120 * 3 = 9.0 GB raw, 23.4 GB provisioned, 854 USD/month
```

The lesson is not the absolute number but the 2x: view count is the dominant lever, and it is a
design decision made once and rarely revisited. Every "just one more feature view" is a recurring
monthly cost plus a permanent increment to read fan-out.

### 9.3 Write throughput

```text
average = E * V / 86,400 = 1.4e6 * 6 / 86,400 = 97.2 writes/s
peak    = 3x average (ILLUSTRATIVE diurnal factor) = 291.7 writes/s
```

Under 300 writes/s is a single-node workload; the interesting part is not the rate but the
*ordering*: writes for one entity must be applied in a way that a stale update cannot overwrite a
newer one. That is a versioning problem (compare-and-set on `t_i`, or last-write-wins with a
monotonic stamp), and it is where the real engineering is at even trivial rates.

### 9.4 History and rebuild cost

```text
event history, 1 year, V=6 materialized views:
  rows  = 1.4e6 * 365 * 6 = 3.07e9 rows
  bytes = 3.07e9 * 90 B   = 276 GB     (before compression; columnar typically 3-5x better)

snapshot table for hourly cutoffs over 90 days:
  cells = 25e6 entities * 2,160 cutoffs = 5.4e10 cells
  bytes = 5.4e10 * 8 B  = 432 GB
  scan for one full rebuild at 1e6 rows/s: 1.26e8 rows / 1e6 = 126 s
```

The snapshot table looks affordable at 432 GB and becomes a problem the moment cutoffs are
per-request rather than hourly — the cell count is `entities × cutoffs`, so ad hoc scoring at
arbitrary `t_c` cannot be served from a snapshot table. That is the precise boundary between
"precompute" and "join on demand," and it is set by the cutoff cardinality, not the row count.

### 9.5 Latency budget arithmetic

End-to-end p99 targets compose multiplicatively in the tail, not additively. With six views each at
per-view p99 = 5 ms, fetched in parallel and assumed independent:

```text
P(at least one view exceeds 5 ms) = 1 - 0.99^6 = 1 - 0.941480 = 0.05852
so the fan-out's own p99 is worse than 5 ms; it sits where  1-(1-q)^6 = 0.01
per-view requirement for a fan-out p99 of 5 ms:  q = 0.99^(1/6) = 0.998331
i.e. each view must be at p99.83 for six independent views to meet p99 at 5 ms

sequential 6 views at 5 ms each            = 30 ms
parallel, independent                      ~ 5 ms + 1-2 ms coordination
parallel, shared slow dependency            ~ the dependency's latency, all views wait together
budget available for features at 120 ms end-to-end = 65 ms
```

Independence is the assumption that fails: a network event or a busy store delays all six views at
once, so a real fan-out's p99 is bounded below by the shared dependency's p99 and no per-view target
fixes that. Measure the fan-out; do not infer it from per-view numbers.

The budget arithmetic is the reason to cap `V` and to fetch in parallel. It is also the reason a
percentile feature must be a cache read: a 90-day percentile over 1.26e8 rows cannot be computed in
5 ms, so either it is precomputed (and the budget applies to the read) or it is unavailable at that
latency.

### 9.6 Cost of the parity check

```text
sample of n = 3,012 keys, one historical query + one online read each
historical query: partition-pruned scan of 168 h for 3,012 keys
  rows touched ~ 3,012 * 280 events = 843,360 rows -> sub-second on a columnar store
online reads: 3,012 point lookups at 3,000 req/s capacity = 1.0 s of capacity
total: minutes of engineer-attention per run, negligible compute
```

That the check is this cheap and this diagnostic is the whole argument for running it on every
definition change and nightly. The expensive part is retaining history long enough to query it,
which is a retention decision worth making explicitly.

### 9.7 Complexity classes

| Operation | Time | Space | Dominant term |
|---|---|---|---|
| Single online read | O(1) | O(1) | one round trip + one deserialization |
| Online read across `V` views | O(V) latency (parallel) or O(V·l) serial | O(V·B) | fan-out coordination |
| Materialize latest for one entity | O(window rows) or O(1) with incremental state | O(1) incremental | decomposability of the aggregate |
| Point-in-time join for one cutoff | O(rows in window) | O(rows in window) | window cardinality |
| Build training set for `C` cutoffs | O(C · rows per window) naive; O(rows) with a sorted sweep | O(C · entities) | cutoff count |
| Full history rebuild | O(total history rows) | O(output cells) | retention volume |
| Non-decomposable recompute | O(window rows) per cutoff | O(window distinct values) | cannot be incrementalized exactly |

The "sorted sweep" row is the standard optimization: sort events and cutoffs together by time and
maintain a rolling aggregate, so a training set over `C` cutoffs costs one pass rather than `C`
passes. It works for decomposable aggregates and fails for percentiles over a window that expires
rows — which is the algorithmic reason percentile features are usually bucketed approximations.

## 10. Key Theorems / Results Reference

The results below are the ones worth being able to state and justify in a design review. They are
elementary, but each one is a decision in disguise.

### R-01 Monotonicity of the legal row set

**Statement.** For fixed `e` and `W`, if `t_c1 <= t_c2` then every row legal at `t_c1` is legal at
`t_c2` unless it has left the window (`t_e < t_c2 - W`).

**Justification.** The filter is a conjunction of two half-open upper bounds and one lower bound:
`t_e < t_c` and `t_i <= t_c` both relax as `t_c` grows; `t_e >= t_c - W` is the only term that can
fail later. So legality is monotone up to window expiry.

**Consequence.** Historical feature values can be computed once and carried forward; a row never
becomes illegal for a reason other than expiry. This is what permits incremental materialization
and what makes a snapshot table meaningful.

**Where it breaks.** With an entity key that is rewritten (merge), or a definition that references
"current" attributes, monotonicity fails and precomputation must be invalidated rather than
extended.

### R-02 The leak bound for a fixed lag

**Statement.** With constant lag `L` and window `W > L`, the fraction of nominal window mass that
is invisible at `t_c` is `L/W`; with `L >= W` the feature is always empty.

**Justification.** The visible band is `[L, W)` in age, of width `W - L`; the nominal band is
`[0, W)`. Uniform rate gives the ratio `(W - L)/W`.

**Consequence.** Under a constant-lag source, leakage from a `t_e`-only join is exactly `L/W` of
the aggregate, which is computable before writing any code. For the illustrative `W = 168`,
`L = 26`: `26/168 = 15.48%`. (The worked example shows 2.97% because its lag is not constant —
one row of 31.5 h that is nonetheless visible. The two numbers are not in conflict; the formula
bounds the mass that *could* be late, not the mass that is.)

**Where it breaks.** Non-constant lag: the bound is the expected value of `(L_i)/W` over the rows,
not the ratio of means.

### R-03 Grace period optimum under exponential lag

**Statement.** Minimizing `alpha*E*exp(-G/mu) + beta*G` gives `G* = mu*ln(alpha*E/(beta*mu))`, and
`G* = 0` when `alpha*E <= beta*mu`.

**Justification.** First-order condition; the second derivative is
`-(alpha*E/mu^2)*exp(-G/mu) < 0`, so the stationary point is a minimum. Convexity holds because the
exponential is convex and the linear term is convex.

**Consequence.** The optimum scales with `mu` (better pipelines want shorter grace) and
logarithmically with the cost ratio. A 10x cost ratio change moves `G*` by `mu*ln 10`, not by 10x.

**Where it breaks.** Heavy-tailed lag (Section 4.4, Pareto): the exponential tail underestimates the
mass beyond `G`, so the model under-protects exactly the rows the grace exists to exclude.

### R-04 TTL null rate

**Statement.** Under Poisson refresh at rate `r` and reads independent of refresh, the share of reads
with no live value at TTL `T` is `exp(-r*T)`.

**Justification.** Inter-arrival times are exponential; the gap from an arbitrary read to the
previous refresh exceeds `T` with probability `exp(-r*T)`.

**Consequence.** Null rate is exponential in the product `r*T`, so doubling `T` squares the
"survival" factor. Setting `T` from an availability target means solving
`T = -ln(target_null)/r`.

**Where it breaks.** Correlated refreshes after outages, and entities whose refresh depends on
activity (dormant customers), both of which make the Poisson assumption wrong in the direction that
increases nulls.

### R-05 Fan-out tail inflation

**Statement.** For `V` independent views each with per-view tail probability `q` of exceeding
latency `x`, the fan-out exceeds `x` with probability `1 - (1 - q)^V`.

**Justification.** Union of independent exceedance events.

**Consequence.** Tail targets must be set per view: for a fan-out p99 with `V = 6`, per-view
`p = 0.99^(1/6) = 0.99833`.

**Where it breaks.** Correlated latency (shared network, shared store) makes the union bound
pessimistic — the true fan-out tail is often dominated by the shared dependency, so the per-view
target is necessary but not sufficient.

### R-06 Non-decomposability of distinct counts and quantiles

**Statement.** Distinct counts and exact quantiles over a sliding window cannot be maintained in
`O(1)` space per entity; exact maintenance requires `O(distinct)` or `O(window rows)` state.

**Justification.** The exact distinct count of a set requires distinguishing `2^d` possible subsets;
exact quantiles require the full order statistic. Both need space proportional to the input in the
worst case.

**Consequence.** A "real-time distinct merchants" or "rolling p95" feature is approximate by
construction, and the approximation must be documented with its error bound. Snapshotted exact
values will diverge from any recomputation, so the versioning rule from FM-08 applies.

**Where it breaks.** For small bounded domains (a merchant-category count over 20 categories), an
exact count is a bitmap of fixed size and is genuinely `O(1)`; the claim is about unbounded
domains.

### R-07 The sampling floor for parity testing

**Statement.** Detecting a mismatch rate `p` to absolute precision `eps` at confidence `1-a`
requires `n >= p(1-p)(z/eps)^2` samples, which for small `p` is approximately
`n >= p(z/eps)^2`.

**Justification.** Normal approximation to the binomial standard error.

**Consequence.** Verifying a 2% defect to `+/-0.5%` needs about 3,012 samples; verifying 0.1% to
`+/-0.05%` needs about 15,366. Small-defect verification is cheap enough to run nightly, which is
why "we cannot afford parity testing" is not a defensible position.

**Where it breaks.** Non-independent samples (repeated requests for one entity) inflate the
effective variance and require clustering by entity in the interval, or a larger `n`.

### R-08 Little's law for the online tier

**Statement.** `L = lambda * W_q` where `L` is concurrent requests, `lambda` arrival rate,
`W_q` time in system.

**Justification.** Conservation of requests in a stationary system.

**Consequence.** At `18,000` lookups/s and `5` ms service time, concurrency is
`18,000 * 0.005 = 90` in flight. That number sizes connection pools and thread counts, and it
explains why a pool of 20 serializes under load even though each call is fast.

**Where it breaks.** Non-stationary traffic (bursts), where instantaneous concurrency exceeds the
average by the burst factor and pools sized on the mean queue.

### R-09 Cost of a definition change

**Statement.** Changing a definition invalidates every historical value produced by it; the rebuild
cost is proportional to the number of `(entity, cutoff)` cells the training set spans, not to the
number of features.

**Justification.** Each cell is a function of the definition, so a change to the function changes
all cells.

**Consequence.** Definition changes should be batched, versioned, and scheduled against the retrain
cadence; a per-feature rebuild triggered by a whitespace fix is a recurring cost that never appears
in the design document.

**Where it breaks.** If the training set is small (few cutoffs, sampled entities), rebuilds are
cheap and versioning discipline matters less — which is also why small teams get away with practices
that break at scale.

### R-10 Train/serve parity as a consistency requirement

**Statement.** If the historical and online paths are implementations of different functions, the
difference is not bounded by any test; if they are implementations of the same function with
independent errors, the difference is bounded by the sum of the per-path error rates.

**Justification.** Two distinct functions can differ arbitrarily on inputs outside the tested set;
one function with two implementations differs only where an implementation is wrong.

**Consequence.** Generate both paths from one definition and test the implementations against each
other. The requirement is architectural, not statistical: no sampling plan detects a divergence
that only occurs on an input slice you never sample.

**Where it breaks.** Where generation is impossible — a streaming engine and a SQL warehouse cannot
share an implementation for every operator — the substitute is a shared test suite of golden
vectors covering the operator set, which bounds the divergence to operators not covered.
## 11. Interview / Exam Questions

Thirteen questions, ordered by difficulty. At least half require arithmetic, selection, or diagnosis;
the answers are written to be given aloud in under two minutes.

**Q1 (recall). Define a point-in-time-correct join in one sentence, then name the two timestamps it
must filter on.**

A point-in-time join returns, for each entity and cutoff, an aggregate over only those history rows
whose event occurred before the cutoff and whose ingestion was visible at the cutoff. It filters on
`t_e < t_c` and `t_i <= t_c`. The first prevents future facts from entering; the second prevents
facts that had occurred but were not yet observable. Either filter alone leaves leakage, and the
second is the one that is silently omitted.

**Q2 (calculation). A feature uses a 7-day window and the source has a constant 26-hour lag. What
fraction of the nominal window is unusable, and what does that do to a count feature?**

The usable band is `168 - 26 = 142` hours, so `26/168 = 15.48%` of the window is unusable. A count
feature over a uniform event rate of `40/24 = 1.6667` events/hour drops from `280.0` to `236.7`,
a shortfall of `43.3` events. A sum feature drops by the same 15.48% if amounts are independent of
time, but not otherwise: if recent amounts are larger, the sum shortfall exceeds 15.48% while the
count shortfall stays at 15.48%.

**Q3 (diagnosis). Offline AUC 0.91, online precision at the operating threshold is barely above the
base rate, and the serving log shows the feature distributions are similar. Where do you look?**

Similar distribution marginals with a performance gap points at the *joint* rather than the marginal:
feature correlations or the calibration of the score, not the feature values. Check three things in
order: whether the parity residual per entity passes (if marginals match, per-entity residuals
should too, and a failure there means offsetting errors within the population); whether the score's
calibration curve on the serving population differs from training, which indicates prior shift
rather than skew; and whether the threshold was carried over from a population with a different base
rate. A base rate of 2% in training and 0.6% in production moves precision at a fixed threshold by a
factor of roughly three even with a perfect model.

**Q4 (calculation). You need to verify a suspected 2% parity defect to plus or minus 0.5% absolute
at 95% confidence. How many keys, and how does that change for 0.2%?**

`n = p(1-p)(z/eps)^2 = 0.02 * 0.98 * (1.959964/0.005)^2 = 0.0196 * 153,664 = 3,012`. For `eps =
0.002`: `0.0196 * (979.98)^2 = 18,824`. The ratio is `(0.005/0.002)^2 = 6.25`, so a 2.5x tighter
interval costs 6.25x the sample. Both are affordable nightly, which is the argument for making the
check a scheduled test rather than a one-off investigation.

**Q5 (selection). You need "number of distinct merchants in the last 30 days" served at p99 under
5 ms. Which of these do you choose: exact snapshot, exact recompute on read, or a sketch, and what
do you tell the model owner?**

A sketch (HyperLogLog-class) maintained incrementally, or a precomputed bucketed approximation. Exact
recompute on read cannot meet 5 ms at window cardinalities in the millions, and exact snapshots
cannot be revised when late rows arrive. Tell the model owner the error bound explicitly — typically
a relative standard error near `1.04/sqrt(m)` for `m` registers — and version the sketch parameters,
because changing `m` changes the value for the same inputs and invalidates any training set built
with the old parameter.

**Q6 (calculation). Six feature views, each independently at p99 = 5 ms, fetched in parallel. What is
the fan-out p99, and what per-view target meets a 5 ms fan-out p99?**

`1 - 0.99^6 = 5.85%` of fan-outs exceed 5 ms, so the fan-out p99 is above 5 ms. For the fan-out at
p99 = 5 ms with independence, each view needs `0.99^(1/6) = 0.99833`, i.e. p99.83. State the
independence caveat: shared dependencies correlate the tails, so the real answer is dominated by the
slowest shared hop and no per-view target compensates.

**Q7 (judgment). Your team wants a streaming feature store for a monthly-retrained churn model
scored daily. Argue the other side.**

Sub-second freshness buys nothing when the decision cadence is daily and the label matures over 30
days; the model consumes whatever the vector was at scoring time, and a one-hour-old vector is
indistinguishable in effect from a one-second-old one. The costs are real: streaming state is not
replayable at arbitrary cutoffs, so the training set cannot be built from the same code path; the
cluster adds an on-call surface; and late rows require retraction handling that a daily batch job
gets for free by recomputing. The defensible design is a daily snapshot table with an
availability-filtered join, and streaming only for the one real-time decision in the portfolio.

**Q8 (diagnosis). Counts doubled after a weekend backfill but averages are unchanged. Give the most
likely cause, one alternative, and the check that separates them.**

Most likely a non-idempotent write: the backfill appended rather than upserting, so every reprocessed
row exists twice. Count scales by 2, sum by 2, mean unchanged — exactly the observed pattern. The
alternative is genuine source duplicates, but those do not align to the backfill's time range. The
check: does the affected range's boundary coincide with the job's declared interval, and does a
second re-run double again? A second doubling is decisive for non-idempotency.

**Q9 (calculation). An entity refreshes every 12 hours like clockwork and the TTL is 7 days. What is
the null rate, and why does the Poisson formula not apply?**

Deterministic refresh at `r = 2/day` with `T = 7` days: the Poisson formula gives `exp(-14)`, which
is meaningless here because the gap between refresh and read is bounded well below 7 days. The true
null rate is 0 for any read that occurs after the first refresh cycle, so the TTL is inert. The
Poisson model applies when refresh times are unpredictable relative to read times; the diagnostic is
the coefficient of variation of the refresh interval, near 0 for clockwork and near 1 for Poisson.

**Q10 (judgment). Two teams compute `spend_7d` with different definitions. How do you resolve it
without a redesign?**

Pick the definition that matches the decision's information set at `t_c` — usually the
availability-filtered one — and publish it as a versioned artifact with a hash. Rename the other
rather than deleting it, because a live model may depend on it: `spend_7d_eventtime` is an honest
name for the version without the availability filter. Then reconcile: compute both on the same
cutoffs, report the per-entity differences and the population share where they differ by more than
the tolerance. The reconciliation table is what convinces the owning team, not the argument about
correctness.

**Q11 (calculation). A 25M-entity base with 6 views at 120 bytes each, two replicas, 30% overhead.
What is the provisioned memory, and what does the third view cost monthly at 0.05 USD per GB-hour?**

`25e6 * 120 * 6 = 18.0 GB` raw, `36.0 GB` with two replicas, `46.8 GB` with 30% overhead, and
`46.8 * 0.05 * 730 = 1,708 USD/month`. The third view is `25e6*120*1 = 3.0 GB` raw, `7.8 GB`
provisioned, `285 USD/month`. Per-view cost is linear and permanent, which is why view count is a
design budget item rather than an afterthought.

**Q12 (diagnosis). A feature's value changes for a fixed `(entity, t_c)` when recomputed two months
later, and no version changed. What are the candidate causes and how do you test them?**

Three candidates: the aggregate is non-decomposable and the recompute now sees later-arriving rows
(distinct count, percentile); the entity key mapping changed so a different row set is selected
under the same key; or the definition's dependencies changed silently — a referenced dimension
table was updated in place without versioning. Tests in order: compute the row set's checksum before
and after (a change means key mapping or filtering, candidates two and three); if the row set is
identical, the aggregate must be the cause, which confirms a non-decomposable feature being
recomputed over a superset. Then check whether the affected entities are exactly those with late
rows.

**Q13 (judgment). When would you deliberately *not* build a feature store?**

When there is one model, a batch decision cadence, and no second consumer in sight. A scheduled job
that writes a wide table and a model that reads it is reproducible, auditable, and has no cluster to
operate; the store's value is reuse across models and a latency tier, and neither exists yet.
Building it early buys a schema you will change, an on-call rotation you do not need, and a
definition-ownership argument before there is any dispute to settle. Revisit when the second model
needs the same aggregate with the same time semantics, which is the moment the reuse argument
becomes concrete.

## 12. Quick Reference Card

| Item | Rule | Number / form |
|---|---|---|
| Legal row set | filter event time *and* ingestion time | `t_c - W <= t_e < t_c` and `t_i <= t_c` |
| Constant-lag leak bound | fraction of window unusable | `L/W`; for `W=168, L=26`: 15.48% |
| Exponential-lag loss | expected visible events | `lambda*[W - mu*(1 - e^{-W/mu})]`; 5.15% at `mu=8.656` |
| Grace optimum | minimize leak + recency cost | `G* = mu*ln(alpha*E/(beta*mu))`; 38.0 h at the worked costs |
| TTL null rate | Poisson refresh | `exp(-r*T)`; 3.02% at `r=0.5/day, T=7d` |
| Parity sample size | binomial precision | `p(1-p)(1.96/eps)^2`; 3,012 at `p=2%, eps=0.5%` |
| Online footprint | entities x bytes x views | `N*B*V`; 18.0 GB raw for 25M x 120 B x 6 |
| Fan-out tail | per-view target for a fan-out tail | `p_fanout^(1/V)`; p99.83 per view for p99 at `V=6` |
| Diagnosis order | cheapest observable first | L4 units/defaults -> L3 parity -> L2 row set -> L1 lag |

Validity conditions that travel with the numbers: the leak bound assumes constant lag and a uniform
event rate; the grace optimum assumes exponential lag and a linear recency cost; the TTL formula
assumes independent Poisson refresh and fails after outages; the footprint formula counts values
only, excluding replicas beyond those stated, per-key overhead, and allocator slack; the sample-size
formula assumes independent samples, so cluster by entity when requests repeat. Every figure in this
card is ILLUSTRATIVE arithmetic from Section 5 and Section 9, not a measurement of any system.
