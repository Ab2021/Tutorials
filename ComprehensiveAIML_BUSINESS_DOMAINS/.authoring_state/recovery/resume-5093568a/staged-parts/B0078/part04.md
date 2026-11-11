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
