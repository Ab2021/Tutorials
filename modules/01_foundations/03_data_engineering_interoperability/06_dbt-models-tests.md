> Module 03 — Data Engineering and Interoperability
> Topic 06 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 03.01, 03.04

# dbt Models, Tests, and Documentation

---

## 1. Why this topic exists

Aldergate Surgical distributes 4,100 item numbers through six distribution centres. Its planning
pipeline is a dbt project: 214 models, 512 data tests, a documented directed acyclic graph (DAG),
and a green build every morning at 04:15. For eleven months the build passed. Then, in March,
safety stock for the highest-revenue item doubled, the buyer over-ordered by six weeks of cover,
and the excess sat in a distribution centre with a fourteen-month shelf life.

Nobody had changed the model. The item master had gained a second row for 312 item-locations: a
location-specific override had been added as a new version rather than as a new attribute. The
`unique` test sat on `item_sk`, the warehouse surrogate key, which was unique by construction. The
natural key — item, location, version start date — was never tested. The `relationships` test from
the fact to the item dimension checked that each key existed, not that it matched one row. The
join fanned out, on-hand doubled for those 312 item-locations, and every downstream aggregate was
arithmetically correct on doubled inventory.

This is the subject of this topic. dbt — originally "data build tool" — gives a supply chain team
three things that are genuinely hard to obtain otherwise: a model graph that is also the
dependency graph, tests that execute as part of the build, and documentation that lives beside the
code that it describes. It gives none of them for free, and it does not give the one thing that
matters most: a statement of what the numbers mean.

Four claims structure everything that follows.

1. **The model DAG is an executable map of the supply chain data model.** Its nodes are grains and
   its edges are dependencies. If the graph claims a dependency the business does not have, one of
   the two is wrong, and the build will not tell you which.
2. **A test is a falsification attempt, not a proof.** A green suite is evidence that a named set
   of failures is absent, not evidence that the plan is correct.
3. **Materialisation is a cost, latency, and consistency decision.** A view, a table, and an
   incremental model answer the same question with different bills and different failure modes.
4. **Documentation and exposures are part of the contract.** An undocumented mart is a private
   opinion that happens to be queryable.

The failure above is not a dbt failure. dbt did exactly what it was told. It is a modelling failure
that dbt made cheap to express and impossible to see, because the tool reports the status of
assertions, not the status of the world.

Three confusions recur in projects that are technically well built and analytically unreliable:

- **Schema tests mistaken for semantic tests.** `not_null` on a price proves a price exists, not
  that it is the price on the contract. No generic test reaches the second.
- **Freshness mistaken for completeness.** A feed that writes one row and stops is fresh; so is a
  feed missing one site. Freshness is a lower bound on what arrived, not a statement about gaps.
- **A green build mistaken for a good plan.** Tests run against a build-time snapshot. The plan is
  a decision made afterwards, against assumptions that may already be false.

**Why this belongs in a data engineering module.** Topic 03.01 argued that a platform needs
layers, contracts, and prohibitions at each boundary; 03.04 supplied the SQL patterns that carry
temporal and additive semantics. This topic is where those arguments become executable. A layered
architecture that is not encoded in a DAG is a diagram. A grain declaration that is not enforced by
a test is a comment. dbt's contribution is to turn both into something that fails loudly when
violated, and that contribution is narrower than the marketing suggests.

**Scope.** This is not a dbt tutorial. It assumes you can write a select statement and that you
have read 03.01 and 03.04. It does not cover ingestion (03.03), orchestration (03.07), or the wider
quality and lineage architecture (03.11). It concentrates on the decisions that determine whether
a dbt project describes a supply chain or merely reproduces its tables.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the model DAG for a named planning mart from source to exposure, labelling the grain
   at each node and the test type at each edge, and **identify** which edges encode business
   assumptions rather than physical dependencies.
2. **Write** a one-sentence grain declaration and the corresponding key tests for a snapshot fact
   and a movement fact, and **explain** why a surrogate-key uniqueness test does not enforce either.
3. **Choose** a materialisation for a named model from its query pattern, rebuild cost, freshness
   requirement, and downstream fan-out, and **defend** the choice with an estimated cost.
4. **Design** an incremental model for a snapshot fact and for a movement fact, including the
   `unique_key`, the strategy, the lookback window, and the repair procedure for late data.
5. **Map** five supply chain invariants — key uniqueness, referential integrity, non-negative
   on-hand, conservation of units, and temporal validity — to generic or singular tests, and
   **state** what each test cannot catch.
6. **Derive** source freshness thresholds from the load schedule and the planning cutoff rather
   than from intuition, and **explain** why freshness is not completeness.
7. **Document** a mart so that its grain, owner, cadence, and known limits are readable without
   access to the SQL, and **declare** its consumers as exposures.
8. **Critique** a green test suite and **name** at least five semantic failures it would not
   detect.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a left-to-right graph in which each column is a layer, each box is a model, and
each arrow is a place where the grain can change. Tests are attached to arrows, not to boxes,
because the arrow is where a join can multiply rows and where a key can stop being unique.

```text
   SOURCES              STAGING               INTERMEDIATE            MARTS                EXPOSURES
   (immutable)          (1:1 with source)     (join, re-grain)        (business-conformed)

  +------------+  [F]  +----------------+ [U N] +-----------------+ [R] +-------------+ [C] +------------+
  | erp.item   |------>| stg_erp__items |------>| int_item_       |---->| dim_item    |---->| Demand     |
  | erp.loc    |       | stg_erp__locs  |       | location_       |     | dim_location|     | plan       |
  +------------+       +----------------+       | enriched        |     +-------------+     | dashboard  |
                                                 +-----------------+            |          +------------+
  +------------+  [F]  +----------------+ [U N] +-----------------+ [A R]        |
  | wms.stock  |------>| stg_wms__      |------>| int_stock_      |---->+-------------+ [C]
  | _snapshot  |       | stock_snapshot |       | weekly_grain    |     | fct_stock_  |----> (same
  +------------+       +----------------+       +-----------------+     | weekly      |      exposure)
                                                                          +-------------+
  +------------+  [F]  +----------------+ [U N] +-----------------+ [S]        ^
  | erp.move   |------>| stg_erp__      |------>| int_movement_   |------------+
  | _ledger    |       | movements      |       | signed_units    |
  +------------+       +----------------+       +-----------------+

  [F] source freshness        [U] unique on the declared grain key
  [N] not_null on key parts   [R] relationships to the as-of dimension
  [A] accepted_values         [S] singular invariant (conservation, non-negative)
  [C] contract on a public mart
```

Read the picture left to right and the layer logic is straightforward. Read it as a set of
claims and it becomes more interesting. Each arrow says "this model is computable from that one".
Each test says "this claim must remain true after the arrow is traversed". The two are different
kinds of statement, and most projects encode only the first.

Six properties follow, and they explain most of what goes wrong.

**Property 1 — a node is a grain, not a table.** A model called `int_stock_weekly_grain` is a
promise that one row means one item, one location, one week. The table is how the promise is
stored. When the grain is violated, the table still exists, the build still succeeds, and every
downstream sum is wrong by the fan-out factor.

**Property 2 — an edge is an assumption.** The edge from `fct_stock_weekly` to `dim_item` assumes
that exactly one dimension version matches each fact row. That assumption is temporal: it depends
on the join predicate being an as-of join on the business date, not an equality join on the
natural key. The DAG draws the edge identically in both cases. Only a test distinguishes them.

**Property 3 — tests belong where grain changes.** A uniqueness test on a staging model is nearly
free and nearly useless: staging is one-to-one with a source, so the source's primary key is
already unique. The same test on the intermediate model after a join is where it earns its place,
because the join is what can multiply rows.

**Property 4 — materialisation changes what is stored, not what is true.** A view computes at read
time, a table computes at build time, and an incremental model computes part of the answer and
reuses the rest. All three can produce the same correct result on a given day. They differ in when
they can be wrong, how much they cost to be right, and how quickly a late correction propagates.

**Property 5 — an incremental model encodes a claim that the past is settled.** The claim is that
rows older than the lookback window will not change. In a supply chain that claim is routinely
false: stock counts are restated, movements are posted late, case packs change, and item masters
are corrected weeks after the effective date. The lookback window is the size of the bet you are
making on that claim.

**Property 6 — a green build certifies assertions, not decisions.** The suite can tell you that
the key is unique, the foreign key resolves, and the balance did not go negative. It cannot tell
you that the allocation rule matches commercial policy, that the demand signal is consumption
rather than shipment, or that the service target is the one in the contract.

### 3.1 The three layers and what each is allowed to know

dbt's own structure guidance converges on three layers, moving from source-conformed to
business-conformed data. The value of the convention is not the names. It is the prohibition that
each layer carries.

| Layer | Prefix | Allowed to know | Not allowed | Materialisation |
|---|---|---|---|---|
| Staging | `stg_` | One source table: rename, cast, categorise, normalise nulls | Joins, aggregation, business rules | View |
| Intermediate | `int_` | Several staging models; re-graining, joining, isolating complex logic | Being consumed by BI directly | Ephemeral or view in a private schema |
| Marts | `fct_`, `dim_` | Business entities, conformed dimensions, facts at a declared grain | Reaching back to raw sources | Table or incremental |

The prohibition is the point. A staging model that joins is a model whose grain nobody can state,
because it has two sources and therefore two possible keys. An intermediate model consumed by a
dashboard is a private detail that has acquired customers, so it can no longer be refactored. A
mart that reads a raw source has bypassed every test on the way in. The names themselves are a
convention, not a dbt requirement; projects that treat the convention as optional usually discover
the cost six months later, when the DAG has 400 models and no one can say which are safe to change.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| dbt | A transformation framework that compiles SQL and Jinja into warehouse DDL and DML | Called an orchestrator or an ingestion tool | Software version |
| Model | One select statement in a `.sql` file, compiled to a relation | Any script that produces a table | One file, one relation |
| Source | A declaration of a raw table that dbt reads but does not own | Any table referenced with `ref()` | `schema.table` |
| Snapshot (dbt) | Built-in slowly changing dimension type 2 capture of a mutable source | Confused with a snapshot fact table | Row version with validity timestamps |
| Staging model | One-to-one model over a source; rename, cast, categorise | A place for joins and aggregation | One row per source row |
| Intermediate model | Purpose-built model that re-grains or joins staging models | A second mart layer | Model |
| Mart model | Business-conformed fact or dimension for consumers | Any table a dashboard reads | Model |
| DAG | Directed acyclic graph of `ref()` and `source()` dependencies | Assumed to equal the business process map | Nodes and edges |
| `ref()` | Jinja function resolving a model dependency and its relation name | Replaced by a hard-coded schema and table | Function |
| Materialisation | How a model is persisted in the warehouse | A style preference | view, table, incremental, ephemeral, materialized view |
| View | A stored query evaluated at read time | Assumed to be free to query | No stored rows |
| Incremental | A persistent table to which only new or changed rows are added | The default choice for every mart | Rows processed per run |
| Ephemeral | A common table expression interpolated into downstream models | Used for logic shared by many models | No database object |
| `unique_key` | The column or columns that identify a row for incremental updates | Set to a surrogate key that is unique by construction | Column list |
| Incremental strategy | The adapter-specific method: append, merge, delete+insert, insert_overwrite, microbatch | Assumed portable across warehouses | Enum |
| Full refresh | Rebuild the model from scratch | Run casually against production history | Flag |
| Lookback window | Trailing period reprocessed to absorb late-arriving data | Set to zero or to a round number | Days or weeks |
| Late-arriving data | Facts whose business date precedes the last processed watermark | Assumed not to exist | Rows or percentage |
| Watermark | The maximum business date already processed | Confused with the maximum load timestamp | Timestamp |
| Data test | A SQL assertion that returns failing rows; zero rows means pass | Called a unit test | Rows returned |
| Generic test | A parameterised, reusable test such as `unique` or `relationships` | Assumed to validate business meaning | Test macro |
| Singular test | A one-off SQL file under `tests/` | Used for logic that should be reusable | File |
| Unit test | A test of transformation logic against mocked input and expected output, before materialisation | Expected to detect source drift | Test case |
| `accepted_values` | Generic test restricting a column to an enumerated domain | Applied to a list that changes every week | Test |
| `relationships` | Generic test asserting each child value exists in a parent | Assumed to enforce cardinality or temporal validity | Test |
| Composite uniqueness | A test that the combination of columns is unique | Replaced by `unique` on a single surrogate | Test |
| Severity | Whether a test failure warns or errors | Everything set to warn | Enum |
| Source freshness | Comparison of `max(loaded_at)` against thresholds | Treated as a completeness check | Hours |
| `warn_after` / `error_after` | Freshness thresholds with a count and a period | Chosen by feel rather than from the schedule | Duration |
| Exposure | Metadata declaring an external consumer of a model | A dashboard link pasted into a wiki | YAML node |
| Contract | Enforced column names and data types, checked before the model builds | Assumed to validate values | YAML |
| Lineage | The upstream and downstream graph of data dependencies | Assumed to include BI and ingestion outside dbt | Nodes and edges |
| Grain | The exact meaning of one row | Stated as the table name | Declaration |
| Referential integrity | Every child key resolves to a parent | Equated with uniqueness | Constraint |
| Invariant | A property that must hold for every row or period | A key performance indicator target | Predicate |

**Generic test versus unit test.** A generic data test runs after materialisation and asks whether
the *data* satisfies a predicate. A unit test runs before materialisation and asks whether the
*SQL* transforms a known input into a known output. The first detects a change in the world; the
second a change in your logic. A project with 500 data tests and no unit tests can still deploy a
broken join and discover it only when the source exercises the broken branch; a project with unit
tests only can be correct in logic and blind to a feed that stopped.

**Contract versus test.** A contract is a structural promise checked before the build: these
columns exist with these types. A test is a content promise checked after: this column has no
nulls, this key is unique. Contracts fail fast and cheaply; tests fail late and informatively.
Neither validates that the numbers are meaningful.

**Freshness versus completeness.** Freshness is the age of the newest row, `now - max(loaded_at)`.
Completeness is the fraction of expected rows or partitions that arrived. A feed that loads one row
per hour is fresh every hour and complete almost never; a feed missing one of six sites is fresh
and 83% complete. Section 5 step 6 adds the check freshness cannot provide.

## 5. The mechanics, step by step

The sequence below follows the order in which the decisions constrain each other. You cannot
choose a materialisation before you know the query pattern; you cannot know the query pattern
before you know the grain; you cannot know the grain before you know which decision the mart
serves. Most projects begin in the middle of this list, with a table that already exists, and
spend the rest of their life discovering which upstream decision was never made.

### Phase A — Fix the grain and the invariant before the SQL

**Step 1. Name the decision the mart serves.**

Write the decision, the owner, and the cadence in the model documentation: "Monday replenishment
for 3,940 active item-locations, owned by inventory planning, refreshed before the Monday 06:00
run." A mart with no named decision has no defensible grain, because grain is defined by what the
consumer needs to distinguish. Without it the model accretes columns from every request until it
has three grains, distinguishable only by which columns are null.

**Step 2. Write the grain as one sentence.**

"One row per item, per location, per ISO week; `closing_on_hand` is the balance at the end of the
week; `opening_on_hand` is the balance at the start." If the sentence needs an "and" joining two
row meanings, the model has two grains. Skipping this is why uniqueness tests get written against
the surrogate key, which proves nothing (section 1).

**Step 3. Write the invariants before the SQL.**

Four classes, each with an owner and a consequence:

| Class | Example | Test form | If violated |
|---|---|---|---|
| Key | item, location, week is unique | Composite uniqueness | Fan-out doubles every downstream sum |
| Domain | `on_hand >= 0`; status in a known set | Singular or `accepted_values` | Impossible states enter the plan |
| Referential | Every item exists in the item master as of the business date | `relationships`, plus a temporal singular test | Orphan rows drop silently in an inner join |
| Algebraic | Opening equals prior closing; movement equals the change in snapshot | Singular conservation test | Units appear or vanish; reconciliation never closes |

The invariants belong in the model's YAML description, not in a ticket. A test that is not written
down as an invariant is a test that will be deleted by the next person who finds it inconvenient.

### Phase B — Declare sources and freshness

**Step 4. Declare every raw table as a source.**

A source declaration does three things: it puts the raw table in the lineage graph, it attaches
freshness configuration to a named object, and it makes a schema change upstream fail at parse or
run time rather than silently producing an empty model. Referencing a raw table with `ref()` or a
hard-coded name forfeits all three.

**Step 5. Derive freshness thresholds from the schedule and the cutoff.**

Freshness thresholds are not chosen. They are computed: `healthy age at cutoff = cutoff time -
expected load completion`, `warn_after = healthy age + allowed slack`, and `error_after =
warn_after + escalation margin`.

Worked: a warehouse management system (WMS) snapshot completes at 01:30, the planning cutoff is
05:30, and a fifteen-minute delay is tolerable. The healthy age is four hours, so `warn_after` is
4 h 15 min and `error_after` is 4 h 30 min. A failed load produces an age of about 28 hours at the
next cutoff, which exceeds the error threshold by a wide margin. Skipping this is how thresholds
end up at 24 or 48 hours because those numbers feel safe: they catch a catastrophic outage and
miss a single missed nightly load, which is the failure that actually occurs (section 6.5).

**Step 6. Add a completeness check, because freshness cannot provide one.**

Freshness asks for the maximum load timestamp. A job that writes twelve rows and crashes is fresh.
Add a singular test that counts expected partitions for the current load date — six sites, four
warehouses, one row per site — and returns a row when the count is short. This is the cheapest
test in the project and one of the most valuable.

### Phase C — Build staging models

**Step 7. One staging model per source table, with no joins.**

Rename to the project's vocabulary, cast every column explicitly, normalise nulls, standardise
codes, and stop. Business logic in staging cannot be reused and cannot be tested at a stable grain.
Casting is not cosmetic: joining a string identifier to an integer either errors or silently drops
rows depending on the warehouse, and casting a timestamp without a time zone produces a value that
means different instants in different sessions. State the target type and time zone in the model
description.

**Step 8. Materialise staging as views.**

A view over a source costs nothing to build and always reflects the newest rows. The cost appears
at query time, which is correct here because staging models are small and lightly transformed.
Materialising staging as tables pays a rebuild on every run for a result discarded before any
consumer sees it.

### Phase D — Build intermediate models

**Step 9. Use intermediate models to re-grain and to isolate logic.**

An intermediate model exists because a transformation is used in more than one mart, or because
the re-grain deserves its own tests. Name it with a verb:
`int_stock_snapshot_regrained_to_week`, not `int_stock_2`. Materialise it as ephemeral when it is
small and used once or twice, and as a view in a private schema when recomputing it in three
downstream models would dominate the run.

**Step 10. Put the uniqueness test on the output grain, not the input key.**

This is the single highest-value test placement decision in the project. When an intermediate
model joins a fact to a type 2 dimension, the input keys can both be unique while the output is
not. The test that catches the failure is composite uniqueness on the output grain. The test that
misses it is `unique` on either input surrogate key.

### Phase E — Choose marts and materialisations

**Step 11. Classify each mart by query pattern and rebuild cost.**

| Materialisation | Stored object | Build cost | Query cost | Freshness | Typical use |
|---|---|---|---|---|---|
| View | None | Near zero | Full recompute per read | Current | Staging, light transforms |
| Table | Full copy | Full rebuild every run | Low | As of last run | Marts read by BI |
| Incremental | Persistent table | Only new or changed rows | Low | As of last run, subject to lookback | Large event facts |
| Ephemeral | None | Pushed downstream | Downstream | Current | Small shared logic |
| Materialised view | Warehouse-managed | Refresh policy | Low | Policy-dependent | Warehouse-specific |

**Step 12. Apply the default, then invert it where the consumer demands it.**

dbt's documented advice is to start with views, move to tables when queries become slow, and move
to incremental when builds become slow. That advice is sound for exploratory models and wrong for
a mart that a dashboard queries every fifteen minutes. A view over a 3.8-million-row join is not a
model with a query cost; it is a query cost with a model attached. For consumer-facing marts, the
order is reversed: choose the materialisation from the read pattern, then justify it with the
build cost.

The defensible arithmetic has two terms, both of which must be estimated before the choice:
`annual build credits = build minutes / 60 x credit rate x runs per year`, and the query-cost
difference between the view and the table, given the read frequency and rows scanned per query. A
decision justified only by build time ignores read amplification; a decision justified only by
query speed ignores the rebuild cost of a table nobody reads between builds.

### Phase F — Design incremental models

**Step 13. Classify the fact before choosing the strategy.**

Snapshot facts and movement facts fail differently and need different incremental designs.

| Fact | Grain | Correction pattern | Strategy | Key |
|---|---|---|---|---|
| Inventory snapshot | Item-location-week | Restatements of past weeks | Merge or delete+insert | Item, location, week |
| Movement ledger | One row per movement event | Late postings with past business dates | Merge with lookback | Movement identifier |
| Order line | One row per order line | Amendments and cancellations | Merge | Order line identifier |
| Shipment event | One row per scan | Late scans and re-scans | Append or merge | Event identifier |

**Step 14. Choose the strategy from the adapter's actual capabilities.**

`merge` is a row-level upsert and scans the destination, which is appropriate for moderate tables
and expensive on very large ones. `delete+insert` removes matching keys and inserts the new set,
which suits a key not guaranteed unique at source. `insert_overwrite` replaces partitions and
ignores `unique_key`, so it is correct only when the partition is the unit of correctness.
`microbatch` processes independent event-time batches and is designed for large time-series facts
and backfills. Adapter support differs, so verify the strategy against the warehouse in use rather
than assume it from a blog post.

**Step 15. Set the lookback from the observed lateness distribution.**

Compute the distribution of `business_date` minus `load_date` for the movement feed, and take a
high percentile. If the 99th percentile is eleven days, a seven-day lookback leaves roughly 1% of
rows permanently unseen. Round up to fourteen days. Then add a monthly full refresh as a repair
for the tail beyond the percentile, because no finite lookback covers the maximum.

Write the arithmetic next to the configuration: `rows rescanned = item-locations x lookback
weeks`, `rescan fraction = rows rescanned / total rows`, and `uncovered fraction = P(lateness >
lookback)`. For 24,600 item-locations and a two-week lookback, 49,200 rows are rescanned per run
out of 3,837,600, or 1.28%. The cost is negligible; the protection is not.

**Step 16. Test idempotency, because incremental models fail by accumulation.**

Run the incremental model twice on unchanged input. The row count and a checksum over the grain
and measure columns must be identical. A model that fails this test will drift silently, and the
drift will be attributed to the data rather than to the model. The practical form is a continuous
integration job that builds the model incrementally and then compares it with a full-refresh
build of the same model on the same input.

*If skipped:* duplicate rows accumulate at a rate proportional to retries and overlapping runs,
and no uniqueness test catches them if the key includes an ingestion timestamp.

### Phase G — Encode invariants as tests

**Step 17. Map each invariant to a test type, and state the blind spot.**

| Invariant | Test type | Example | Catches | Does not catch |
|---|---|---|---|---|
| Key uniqueness | Generic composite | `unique_combination_of_columns` on item, location, week | Fan-out, duplicate load | Wrong choice of grain |
| Key completeness | Generic | `not_null` on each key part | Orphan or unkeyed rows | Wrong key |
| Domain | Generic | `accepted_values` on status | New or misspelled codes | Illegal transitions between codes |
| Referential | Generic | `relationships` to the dimension | Orphan foreign keys | Joining the wrong version |
| Temporal referential | Singular | Exactly one dimension version per fact date | Version gaps and overlaps | Missing versions upstream |
| Non-negative | Singular | `on_hand >= 0` | Sign errors, bad reversals | Overstock, wrong item |
| Conservation | Singular | Movement equals the change in snapshot | Conversion and posting errors | Errors present in both sides |
| Idempotency | Singular or CI | Double-run equality | Non-deterministic incremental logic | Logic errors that are deterministic |
| Freshness | Source freshness | Age of `max(loaded_at)` | Stalled feed | Partial load |
| Completeness | Singular | Expected partition count | Partial load | Wrong values in arrived rows |

**Step 18. Test every grain change, and test the dimension's versions separately.**

Two tests are needed for the section 1 failure, and neither is a substitute for the other. The
fact table needs composite uniqueness on its grain, which fails when the join fans out. The
dimension needs composite uniqueness on item, location, and version start date, plus a singular
test that no two versions for the same item-location overlap in validity. The `relationships`
test passes in both cases, because the foreign key does resolve — to two rows.

**Step 19. Make referential tests temporal.**

A `relationships` test with no date predicate answers "does this item exist today". The supply
chain question is "did this item exist on the business date of the fact". Where the dimension is
type 2, write a singular test that joins on the as-of predicate and returns facts with no matching
version or with more than one. The generic test is a cheap first line; the temporal test is the
one that protects the semantic.

**Step 20. Write singular tests for algebraic invariants.**

Conservation, non-negativity, and monotonicity cannot be expressed as column predicates. They
compare two derived objects or two periods. Section 7 gives a full example for conservation of
units. The design rules are: state the sign convention, state the grain, state the tolerance
(zero for integer units), state the treatment of missing rows, and state what common-mode error
the test cannot see.

**Step 21. Set severity deliberately.**

A test at `warn` severity reports and continues. A test at `error` blocks downstream models when
the project is built with `dbt build`. Setting a uniqueness test to warn is a decision to publish
duplicates with a note attached. That may be the right decision during a migration; it must be
recorded as a decision with an expiry date, not left as a default.

### Phase H — Documentation, exposures, and contracts

**Step 22. Document the four things a reader cannot derive from the SQL.**

The grain, the owner, the refresh cadence, and the known limits. Column descriptions are useful
but derivable from names and types; the four items above are not. Put them in the model's YAML
description and in a docs block when the explanation exceeds a paragraph.

**Step 23. Declare every external consumer as an exposure.**

An exposure is metadata only: it creates no SQL and no database object. It appears as a terminal
node in the DAG, which makes the consumer visible in lineage and selectable in commands such as
`dbt build --select +exposure:demand_plan`. It also makes the dependency explicit when a model is
about to be deprecated. An undocumented dashboard is a consumer that cannot be warned.

**Step 24. Put contracts on public marts, and know what they enforce.**

A contract with `enforced: true` requires every column to be declared with a name and a data
type, and dbt checks the model's output against the contract before building it. Contracts support
table, view, and incremental materialisations, with limitations, and constraint enforcement varies
by warehouse: some platforms enforce only `not_null` and treat the rest as metadata. A contract is
a structural guarantee, not a content guarantee. It will not tell you that the price is wrong.

### Phase I — Operate the project

**Step 25. Build with `dbt build`, not with `dbt run` followed by `dbt test`.**

`dbt build` interleaves tests with model builds and skips downstream models when an upstream test
fails. `dbt run` followed by `dbt test` builds every model first and discovers the failure
afterwards, so bad data has already reached the mart. The difference is a guard versus a
post-mortem.

**Step 26. Select by state in continuous integration, run the full suite nightly, and write down
what the suite does not check.**

Selecting modified models and their children keeps pull-request checks fast, but a model that was
not modified is not retested, so a source change or a rare data path can escape. Run the full
suite on a schedule and publish the distinction. Then keep a short README section listing the
unverified semantic claims: the allocation rule, the demand proxy, the service-target definition,
the conversion-factor authority, and the customer hierarchy. That list is the honest boundary of
the green build.

## 6. Worked example

Aldergate Surgical, introduced in section 1, is used throughout. All figures are illustrative but
internally consistent. The project is small enough to reason about by hand and large enough that
the materialisation choice matters.

### 6.1 The project

| Model | Layer | Grain | Materialisation | Rows |
|---|---|---|---|---:|
| `stg_erp__items` | Staging | One row per item version | View | 9,400 |
| `stg_erp__locations` | Staging | One row per location | View | 62 |
| `stg_wms__stock_snapshot` | Staging | One row per item-location-day | View | 24,600 per day |
| `stg_erp__movements` | Staging | One row per movement | View | 1.42m per year |
| `int_item_location_enriched` | Intermediate | One row per item-location-version | Ephemeral | 26,100 |
| `int_stock_snapshot_weekly` | Intermediate | One row per item-location-week | View | 3,837,600 |
| `dim_item` | Mart | One row per item-version | Table | 9,400 |
| `dim_location` | Mart | One row per location | Table | 62 |
| `fct_stock_weekly` | Mart | One row per item-location-week | Incremental | 3,837,600 |
| `fct_inventory_movement` | Mart | One row per movement | Incremental | 4.26m |

The arithmetic behind the largest number: 24,600 item-location combinations, including inactive
ones, multiplied by 156 weeks of history, equals 3,837,600 snapshot rows. The movement ledger
holds 1.42 million rows per year over three years, or 4.26 million.

### 6.2 What the DAG claimed and what the business meant

The DAG contained an edge from `fct_stock_weekly` to `dim_item`. Read as a physical dependency,
the edge is correct: the fact needs item attributes. Read as a semantic claim, it asserts that
each fact row matches exactly one item version — the version in force on the week being reported.
The model implemented an equality join on the natural key, so the edge asserted something stronger
and false: that the natural key is unique in the dimension.

Two tests were attached. `relationships` checked that `item_sk` existed in `dim_item` and passed,
because both duplicate rows had valid surrogate keys. `unique` on the fact's `item_sk` also
passed, because the fan-out produced two fact rows with two different surrogates. Neither test was
on the grain.

### 6.3 Incident 1 — the fan-out

A location-specific override was loaded as a new item version for 312 item-locations, with the
same validity start as the existing version. The dimension gained 312 duplicate versions. The
fact join multiplied those rows by two for every week in which both versions were valid, which
was nine weeks.

Duplicated fact rows were `312 x 9 x 2 = 5,616`, or 0.146% of the fact table. The share is small
and the consequence was not. The duplicated rows were concentrated on high-revenue items, and the
plan aggregates by item across locations, so doubled on-hand propagated into safety stock, reorder
points, and purchase orders.

The tests that would have caught it:

| Test | Target | Result on the incident |
|---|---|---|
| Composite uniqueness on item, location, week | `fct_stock_weekly` | Fails with 5,616 rows |
| Composite uniqueness on item, location, valid_from | `dim_item` | Fails with 312 rows |
| Overlapping-validity singular test | `dim_item` | Fails with 312 pairs |
| `unique` on `item_sk` | `fct_stock_weekly` | Passes |
| `relationships` on `item_sk` | `fct_stock_weekly` | Passes |

The last two rows are the lesson: the tests were not badly written, they were attached to the
wrong claim.

### 6.4 Incident 2 — the conversion

Case pack for a high-volume item changed from 12 eaches to 10 eaches, effective 1 February. The
change was recorded in the item master on 19 February, as a new version with a valid-from date of
1 February. The movement model converted cases to eaches using the pack size in force on the
recording date, which for the first eighteen days of February was still 12. The warehouse
snapshot recorded eaches directly, because the WMS counts what it scans.

```text
cases issued, 1-19 February            1,860
recorded issues at 12 eaches/case     22,320 eaches
true issues at 10 eaches/case         18,600 eaches
overstatement of issues                3,720 eaches
recorded closing on hand              3,720 eaches too low
```

The conservation test compares the change in the snapshot to the net movement, per
item-location-week, and returns every row where the two disagree. It returned 47 item-locations
with a total residual of 3,720 eaches. Nineteen of them had crossed zero and produced false
stock-out alerts, which triggered emergency transfers at an average expedite cost of $640 per
transfer.

The test worked only because the two sides had independent provenance. The snapshot came from the
WMS in eaches; the movement came from the enterprise resource planning (ERP) system in cases,
converted using the item master. Had both sides been derived from the same conversion logic, the
test would have passed while both were wrong. This is the common-mode limitation described in
section 11, and it is why the conservation test must be paired with an independent control such as
a physical cycle count.

### 6.5 Incident 3 — the cutoff

The WMS snapshot was expected to complete at 01:30 each day. The planning job started at 05:30.
Freshness was configured at `warn_after` 24 hours and `error_after` 48 hours, values chosen because
they "seemed safe".

On 3 March the load failed. At 05:30 the newest row was from 2 March at 01:34, an age of 27 hours
56 minutes. The check emitted a warning and the job continued. The plan ran on inventory positions
that were one day stale, for a category with a daily replenishment cadence.

The thresholds that would have caught it are derived in section 13.2: a healthy age of 4 hours at
cutoff, `warn_after` 4 h 15 min, and `error_after` 4 h 30 min, against a missed-load age of 27 h
56 min. The general lesson is that a freshness threshold is a statement about the schedule, not
about tolerance for stale data. Setting it to 24 hours encodes the belief that a one-day-old
position is acceptable, which may be true for a monthly plan and is false for a daily one.

### 6.6 The materialisation arithmetic

`fct_stock_weekly` is the model where the choice is contested. The numbers below assume a
warehouse billed at 2.0 credits per hour and a credit price of $3.00, both illustrative.

| Quantity | Full refresh | Incremental, 2-week lookback |
|---|---:|---:|
| Build time | 41.0 min | 4.2 min |
| Credits per run | 1.367 | 0.140 |
| Runs per year (5 per week) | 260 | 260 |
| Credits per year | 355.3 | 36.4 |
| Cost per year at $3.00 | $1,066 | $109 |
| Rows processed per run | 3,837,600 | 49,200 |
| Build window before cutoff | 41.0 min | 4.2 min |

The credit saving is about 319 credits, or roughly $957 per year. That alone does not justify the
complexity: twenty hours a year of build and maintenance at a loaded rate of $110 per hour costs
$2,200, more than twice the saving. The justification is the build window. Moving from 41 minutes
to 4.2 minutes creates 36.8 minutes of margin before a hard cutoff, and margin absorbs a late
source, a retried job, or a warehouse queue.

The cost the table omits matters most: the lookback. A two-week window rescans 1.28% of the table
per run, which is cheap, but 1% of movements arrive more than eleven days late and a tail arrives
up to five weeks late. Those rows stay invisible until a monthly full refresh repairs them, so the
mart can be wrong for up to thirty days in a way no test detects unless it runs on the repair.

### 6.7 What a green suite would still not have caught

After the three incidents were fixed, the suite had 548 tests and was green for six months. It
would still not have caught three things. **The allocation rule** used the current customer master
rather than the version in force on the order date: every key unique, every foreign key resolved,
every balance conserved, and the allocation commercially wrong. **The demand proxy** measured
shipments from the distribution centre, not end-customer consumption, so conservation was
satisfied by comparing two views of the same shipment data. **The service target** measured on-time
in full against an internal promise date while the contract specified the customer's requested
date; both were computed correctly and answered different questions.

These are the failures dbt does not solve, and they are the normal content of a planning review. A
project that documents them as unverified claims is honest; one that presents a green suite as
assurance is not.

## 7. Code, query, or calculation

The single code block below is a singular test that encodes conservation of units. It is chosen
over a model definition because the model is the easy part and the invariant is the part that
takes judgement; the surrounding prose states what the SQL cannot: sign convention, grain,
tolerance, null treatment, and blind spot.

### 7.1 The conservation test

```sql
-- tests/assert_units_conserved.sql: snapshot change = net movements per week
with movement as (
    select item_id, location_id, week_start,
           sum(case movement_type
                   when 'receipt'    then  units_qty
                   when 'issue'      then -units_qty
                   when 'adjustment' then  units_qty
               end) as net_units
    from {{ ref('fct_inventory_movement') }}
    group by 1, 2, 3
),
snapshot as (
    select item_id, location_id, week_start,
           closing_on_hand - opening_on_hand as observed_change
    from {{ ref('fct_inventory_snapshot_weekly') }}
)
select s.item_id, s.location_id, s.week_start,
       s.observed_change, coalesce(m.net_units, 0) as net_units,
       s.observed_change - coalesce(m.net_units, 0) as residual
from snapshot s
left join movement m
    on  s.item_id = m.item_id
    and s.location_id = m.location_id
    and s.week_start = m.week_start
where s.observed_change <> coalesce(m.net_units, 0)
```

### 7.2 The decisions the SQL does not state

| Decision | Choice made here | Why | What would change it |
|---|---|---|---|
| Sign convention | Receipt positive, issue negative, adjustment signed | A single signed sum avoids double negation | A source that stores issues as positive with a separate direction column |
| Grain | Item, location, week | Matches the snapshot fact and the planning decision | Daily planning would require a day grain |
| Tolerance | Zero | Units are integers; there is no rounding error to absorb | Weight or volume measures would need a stated tolerance |
| Null treatment | Missing movement becomes zero | A week with no movement and no change is consistent | A week with no movement and a change is a failure and is returned |
| Left join direction | Snapshot is preserved | A movement with no snapshot is a different defect and needs its own test | Both directions should be tested in practice |
| Common-mode error | Not detected | Both sides could share a conversion bug | An independent physical count or source trial balance |

The last row is the important one. The test compares two derived objects, so a shared upstream
error cancels: it is a consistency check, not a truth check. In section 6.4 it worked because the
snapshot was recorded in eaches by the warehouse system while the movement was converted from
cases by the ERP path. Had both been converted by the same model, the residual would have been
zero and the inventory would still have been wrong.

> **Note:** The test is written against mart models rather than staging models on purpose. A
> staging-level comparison would inherit the same conversion logic on both sides and would test
> almost nothing. Mart-level comparison at least crosses two independently built lineages.

### 7.3 The tests that are not shown, and why

The block above deliberately contains one test. Two generic tests would sit in YAML beside the
model, and they are best described rather than printed, because their value is in the argument,
not the syntax.

| Test | Configuration | What it proves | What it misses |
|---|---|---|---|
| Composite uniqueness | Item, location, week on `fct_stock_weekly` | The join did not fan out | That the grain is the one the planner needs |
| Composite uniqueness | Item, location, valid-from on `dim_item` | The dimension has one version per start date | Overlapping ranges with different starts |
| `relationships` | `item_sk` to `dim_item.item_sk` | The key resolves | That it resolves to exactly one version |
| `not_null` | Each of item, location, week | No unkeyed rows | That the key is semantically correct |

The pair of composite uniqueness tests is what the section 1 project lacked. It did have
`relationships` on `item_sk`, and the fourth column of the table explains why that passed.

### 7.4 Reading a failure

A failing conservation test returns rows, not a message. Triage in order: a residual constant
across weeks for one item-location is an opening-balance or conversion error; a residual
proportional to volume is a conversion factor; one confined to a single week is a late posting or
a cut-off difference; residuals in opposite directions suggest a join fan-out rather than a
quantity error. Which side is authoritative is a data-ownership question, not a dbt question.

## 8. Data, standards, and regulatory context

dbt itself is not a standard. It is a tool with conventions, and the standards that matter are the
ones it does not supply: identifiers, units, time representation, and record retention. A dbt
project that is internally consistent can still encode a unit that no trading partner recognises.

### 8.1 Standards that touch a dbt project

| Standard or body | What it constrains | Where it touches the project | Grade |
|---|---|---|---|
| ISO/IEC 9075 (SQL) | Query semantics | Dialect portability; dbt compiles to the warehouse dialect, not to ANSI SQL | B |
| ISO 8601 | Date and time representation | UTC timestamps, week boundaries, validity intervals | B |
| UN/CEFACT Recommendation 20 | Unit of measure codes | `uom_code` in staging and conversion tables | B |
| UCUM | Computable unit syntax | Unit-aware conversion; rarely implemented in SQL marts | B |
| GS1 General Specifications | GTIN, GLN, and SSCC identifiers | Item and location join keys; see 03.09 | B |
| OpenLineage | Lineage event schema | Cross-tool lineage beyond dbt's own manifest | C |
| Semantic Versioning | Version labels | `_v1`, `_v2` model versions and contract changes | C |
| Open Data Contract Standard | Contract metadata | Exported contracts and ownership | C |
| Audit and record-retention rules | Retention of records supporting reporting | Snapshot history and validity timestamps; jurisdiction-specific | B |

Two entries need qualification. OpenLineage is an open specification with uneven adoption; a dbt
project's lineage is complete only within dbt, and the BI tool, the ingestion tool, and the
orchestrator each hold a piece. Open Data Contract Standard is a community specification, not a
regulated standard, and its use in dbt is through export rather than native enforcement.

### 8.2 What dbt does not standardise

- **Grain vocabulary.** One project's `fct_stock_weekly` and another's may both be weekly and mean
  different weeks: ISO week, Sunday-start week, or a 4-4-5 retail calendar.
- **Metric definitions.** On-time in full, fill rate, and forecast accuracy have no canonical
  numerator. dbt can test that a column is populated; it cannot test that the definition matches
  the contract.
- **Unit conversion authority.** A conversion factor is a business fact with an effective date and
  an owner. dbt can test that the conversion preserves units; it cannot test that the factor is
  the one in the supplier agreement.
- **Calendar conventions.** Fiscal calendars, holiday calendars, and time zones are project
  choices that leak into every date key.
- **Entity resolution.** Which supplier records are the same supplier is decided upstream, in
  master data management (03.10), not in a mart.
- **Allocation policy.** Priority rules, fairness rules, and contractual entitlements are
  commercial decisions encoded in SQL. A test can check that units are conserved; it cannot check
  that the rule is fair. The absence of a standard is not a defect in dbt but a boundary; the
  defect is treating the tool's internal consistency as evidence that the semantics are settled.

### 8.3 Retention and reproducibility

Snapshot history is the part of a dbt project most exposed to record-retention obligations. A
`--full-refresh` that rebuilds a snapshot from a source with thirty days of history destroys the
older history permanently. If the snapshot is a record of belief rather than a recomputation, it
must be archived, with a materialisation and a backup policy that reflect that. Retention periods
vary by jurisdiction; listed companies commonly retain records supporting financial statements
for seven years, and sector rules can be longer.

## 9. Industry practice and case evidence

### 9.1 Where the conventions came from

dbt was built by Fishtown Analytics, now dbt Labs, as an open-source SQL-first transformation tool
aimed at analysts rather than platform engineers. The layer conventions and the `stg_`, `int_`,
`fct_`, and `dim_` prefixes came from consulting practice and are documented as best practice
rather than enforced by the software. That origin explains both the tool's legibility and its
gaps: nothing requires a grain declaration, a contract, or a semantic test. The conventions are
widely imitated, which makes them look like a standard; they constrain the name, not the meaning.

### 9.2 What the vendor documentation claims

| Claim | Source | Grade | Comment |
|---|---|---|---|
| Views are cheap to build and always current | dbt documentation, materialisations page | D | True mechanically; omits read amplification |
| Tables are fast to query but rebuilt every run | dbt documentation, materialisations page | D | Accurate and appropriately hedged |
| Incremental models should not be the starting point | dbt documentation, materialisations page | D | Explicit advice against premature incremental |
| Data tests return failing rows; zero rows pass | dbt documentation, data tests page | B | This is documented behaviour, verifiable by inspection |
| Contracts fail the build before materialisation | dbt documentation, model contracts page | B | Documented preflight behaviour |
| Freshness thresholds require a count and a period | dbt documentation, freshness reference | B | Documented configuration semantics |

The grades reflect what kind of claim each is. A statement about how the software behaves is
verifiable and graded B. A statement about which behaviour is better for your project is vendor
guidance and graded D, however sensible it is.

### 9.3 What practitioners report

The critical literature on dbt is thin, recent, and mostly practitioner commentary rather than
controlled evidence.

- **Tests validate schema, not business meaning.** A widely cited practitioner essay argues that
  the four built-in tests confirm structure and cannot confirm that a cost calculation is correct,
  and that projects with green suites still produce wrong numbers. Grade E as evidence; the
  logical claim matches sections 6.7 and 11.
- **Tests catch a fraction of incidents.** A vendor blog asserts that dbt tests alone miss
  freshness, volume, and schema drift, and offers a numeric estimate without a published method.
  Do not quote the number; the qualitative point — tests are point-in-time, monitoring is
  continuous — is correct. Grade E.
- **Alert fatigue is real.** dbt Labs' own guidance concedes that over-testing produces alert
  fatigue and recommends severity discipline and fewer business-relevant tests. Grade D.

### 9.4 Codified practice

Two artifacts are worth knowing because they turn conventions into checklists.

**dbt-project-evaluator** is a package of governance rules. Its documented checks include public
models without contracts, public models without documentation, and exposures that depend on
private models. These are structural checks, not semantic ones, but they catch the common drift
where a model becomes a de facto public interface without anyone deciding that it should.

**Elementary** is an open-source observability package and command-line tool. It loads dbt run
results and test outcomes into warehouse tables through an `on-run-end` hook, adds anomaly tests
for volume, freshness, and schema change, and sends alerts to Slack or Microsoft Teams. It is
vendor-maintained and graded D. Its value is that it watches between builds, which is precisely
where a passing test suite is silent.

### 9.5 What the evidence does not support

There is no controlled study showing that dbt reduces planning errors, and no audited case
attributing a specific inventory failure to a dbt test gap. The claims that survive are
mechanical: a test returns failing rows, a contract fails before materialisation, an incremental
model processes a bounded set of rows. Claims about organisational outcomes are anecdote. Use dbt
because the mechanics are sound and the failure modes are visible, not because a green build
proves a plan right.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Green suite, wrong plan | Tests all pass; planners distrust the number | Tests encode schema, not semantics | Confident bad decisions | Publish an explicit list of unverified claims; add domain review |
| Uniqueness test on a surrogate key | Test passes; row counts jump after a join | Surrogate is unique by construction | Silent fan-out | Test composite natural grain at every grain change |
| `relationships` mistaken for cardinality | Foreign key test passes; totals double | Existence is not uniqueness | Doubled inventory, revenue, demand | Add composite uniqueness on the fact grain and an overlap test on the dimension |
| Non-temporal referential join | Historical facts attributed to current attributes | Equality join on natural key | Restated history, leakage in backtests | As-of join plus a temporal singular test (03.04) |
| Freshness configured but not run | Configuration exists; nobody reads the output | Check is optional in the job | Stale plan presented as current | Make freshness a blocking step before the build |
| Freshness thresholds set by feel | Warn at 24 h, error at 48 h | Threshold not derived from the schedule | Missed nightly load passes | Derive from cutoff and expected load time |
| Freshness treated as completeness | Feed is fresh; one site is missing | `max(loaded_at)` ignores row counts | Partial data in the plan | Add expected-partition count tests |
| Incremental lookback too short | Residuals appear weeks later and vanish after a refresh | Late rows fall behind the watermark | Persistent understatement | Set lookback from the 99th percentile; add monthly full refresh |
| Full refresh destroys history | Snapshot rebuilt from a 30-day source | Source retention shorter than mart history | Permanent loss of belief history | `full_refresh: false`; archive separately |
| Incremental model not idempotent | Duplicate rows after a retry | Append or overlapping windows without a key | Silent accumulation | Double-run equality test in CI |
| Contract declared, not enforced | Column added without a build failure | Constraints treated as metadata by the warehouse | Downstream breakage | Verify adapter enforcement; test content separately |
| Exposures undocumented | Dashboards change without warning | Consumers invisible in the DAG | Deprecation breaks reports | Declare exposures and select from them in CI |
| Alert fatigue | Hundreds of tests; warnings ignored | Severity inflation; tests on every column | Real failures missed | Tier tests; delete low-value ones; error on keys |
| Tests run after models | Bad data reaches the mart before failing | `dbt run` then `dbt test` | Downstream contamination | Use `dbt build` |
| Selection skips unmodified models | PR checks fast; nightly failures | State-based selection | Undetected regressions | Full suite nightly; publish the distinction |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Source timestamps are trustworthy | Freshness and incrementality depend on them | Clock skew, time zone errors, backdated loads | Compare source and warehouse clocks; audit offsets |
| `loaded_at` is monotonic per load | Watermarks advance correctly | Out-of-order or reprocessed loads | Check for decreasing maxima |
| Business keys are stable | Joins and relationships resolve | Re-keying after a merger or migration | Monitor key churn per period |
| The lookback exceeds the lateness tail | Incremental rows are complete | Month-end postings, batch corrections | Compare with full refresh monthly |
| The warehouse enforces contract constraints | Contracts protect downstream | Adapters that treat constraints as metadata | Test enforcement per adapter |
| Tests run on data representative of production | Passes mean something | Skewed samples, empty partitions | Track row counts alongside results |
| One dialect is used | SQL semantics are stable | Multi-warehouse or migration projects | Pin dialect; test in each target |
| The grain is the decision's grain | Aggregates are meaningful | Consumer needs a different cut | Interview the consumer; check query patterns |
| A passing test is evidence about the future | The suite is used as a guard | Structural breaks, regime change | Re-baseline tests after known breaks |

Four limits are worth stating plainly.

**dbt does not validate semantics.** It can prove a column has no nulls, a key is unique, a
foreign key resolves, and two derived quantities agree. It cannot prove the column means what its
name says, that the key is the right key, that the foreign key points at the right version, or
that the quantity is the one the business needs. Those are human judgements; the tool makes them
explicit and testable, not automatically true.

**A green suite is not a correct plan.** Tests are existential over the rows present at build
time; they inherit the model's assumptions, so a shared wrong assumption is invisible; and the
plan is a decision made after the build, against objectives no data test encodes.

**The DAG is a dependency graph, not a process model.** It records which model reads which, not
who owns the decision, which business event the model represents, or which organisational boundary
the data crosses. Those live in documentation and exposures, and are the part most often missing.

**The tooling is not the architecture.** A project can have perfect layer names, contracts, and
500 tests and still be an unmaintainable collection of models with no owner. Architecture is the
set of prohibitions and named decisions; dbt enforces only the subset writable as a predicate.

## 12. Exercises

Eight exercises, ordered by the size of the decision they support; each gives a deliverable, an
acceptance test, and a time estimate for a practitioner working alone.

### Exercise 1 — Draw the DAG from the mart backwards (45 minutes)

Choose a planning mart you can inspect, or use the Aldergate project in section 6. Draw its DAG
from exposure back to source, labelling the grain at every node.

**Deliverable:** one page with boxes, arrows, grain labels, and materialisations.

**Acceptance criteria:** at least eight nodes; every arrow labelled with the grain change it
causes; at least three edges identified as carrying a business assumption rather than a physical
dependency; one edge marked as untested.
### Exercise 2 — Declare the grain and the key tests (60 minutes)

For `fct_stock_weekly` and `fct_inventory_movement`, write the grain sentence and the tests that
enforce it.

**Deliverable:** a two-column table of model and grain sentence, plus a test list naming the test
type and target column set for each.

**Acceptance criteria:** each grain sentence is one sentence with no "and" joining two row
meanings; each key has a composite uniqueness test and a not-null test per component; the
dimension has a separate version-uniqueness test and an overlap test; each test states what it
does not catch.
### Exercise 3 — Design the incremental model (75 minutes)

For a weekly snapshot fact of 3,837,600 rows over 24,600 item-locations, design the incremental
configuration. Late movement arrivals have a median of 1 day, a 95th percentile of 6 days, a 99th
percentile of 11 days, and a maximum of 34 days. The model runs five times per week.

**Deliverable:** the `unique_key`, the strategy, the lookback in weeks with the percentile it
covers, the rows rescanned per run, the uncovered fraction, and a repair procedure.

**Acceptance criteria:** the lookback is justified against the stated percentiles, not chosen for
convenience; the rescan fraction is computed; the repair procedure states its cadence and cost;
the design names the failure it cannot prevent.
### Exercise 4 — Write the conservation test (45 minutes)

Write a singular test asserting that closing on-hand equals opening plus receipts minus issues
plus adjustments, per item-location-week, for a movement feed that records issues as positive
quantities with a direction column rather than signed values.

**Deliverable:** one SQL file of at most 30 lines plus a five-line header comment stating sign
convention, grain, tolerance, null treatment, and blind spot.

**Acceptance criteria:** the sign convention is explicit; the test returns identified, sized rows;
missing movements are treated deliberately; the comment names one error the test cannot catch.
### Exercise 5 — Derive freshness thresholds (40 minutes)

A source completes at 01:30 daily. The plan cutoff is 05:30. A 15-minute delay is tolerable.
Compute thresholds and add a completeness check.

**Deliverable:** the arithmetic, the `warn_after` and `error_after` values in hours and minutes,
and a singular test definition for expected partition count.

**Acceptance criteria:** the derivation shows the healthy age at cutoff; the failed-load age is
compared with both thresholds; the completeness check names the expected entity count and the
load date it filters on; the answer explains why freshness alone is insufficient.
### Exercise 6 — Cost the materialisation choice (50 minutes)

Using the section 6.6 figures, compute the annual credit and currency difference between full
refresh and incremental, then decide whether the incremental model is justified.

**Deliverable:** a table of build time, credits per run, annual credits, annual cost, rows
processed, and build window, plus a short recommendation.

**Acceptance criteria:** both credit and window effects are computed; maintenance cost is
estimated and compared with the credit saving; the recommendation names the benefit that actually
justifies the choice and the condition under which it would reverse.
### Exercise 7 — Document the contract (40 minutes)

Write the YAML documentation and exposure declaration for `fct_stock_weekly` and its demand-plan
consumer.

**Deliverable:** a YAML fragment with model description, grain, owner, cadence, known limits, and
one exposure with name, type, maturity, owner, and dependency.

**Acceptance criteria:** the description contains the four non-derivable facts (grain, owner,
cadence, limits); the exposure names a real consumer and a real owner; the known-limits list
includes at least one semantic limitation; the exposure would allow a `+exposure:` selection.
### Exercise 8 — Red-team a green suite (60 minutes)

Take a project with a large passing suite, or use the 548-test state described in section 6.7.
List ten questions the suite cannot answer about the plan.

**Deliverable:** ten questions, each mapped to the invariant or semantic claim it exposes, each
with the cheapest test or review that would narrow the gap.

**Acceptance criteria:** at least four questions are about definitions rather than data; at least
two are about time (vintage, cutoff, version); each proposed check is cheaper than the failure it
would prevent, or the answer says why it is not.
## 13. Self-check answers

### 13.1 Exercise 3 arithmetic

A weekly grain with 24,600 item-locations produces 24,600 rows per week.

```text
total rows                 = 24,600 x 156 = 3,837,600
two-week lookback rescan   = 24,600 x 2   = 49,200 rows per run
rescan fraction            = 49,200 / 3,837,600 = 1.28%

p99 lateness = 11 days -> round up to 14 days (two weeks)
uncovered by lookback = P(lateness > 14 days) < 1%
uncovered rows per year ~ 0.01 x 1,420,000 = 14,200 movement rows
```

A five-week lookback would cover the observed maximum of 34 days, at 123,000 rows rescanned per
run, or 3.2% of the table. That is still cheap, but it increases the chance that a retried run
reprocesses a large window and that a mid-window logic change is applied unevenly. The defensible
design is a two-week lookback for routine runs plus a monthly full refresh that repairs the tail.
State the residual exposure: up to thirty days of wrong values for rows later than fourteen days,
bounded by the repair cadence. The design still cannot prevent a late row that arrives after the
monthly refresh and is never followed by another, which is why the repair needs a trailing
sixty-day range rather than the current month.

### 13.2 Exercise 5 arithmetic

```text
healthy age at cutoff = 05:30 - 01:30 = 4 h 00 min
allowed slack         = 15 min
warn_after            = 4 h 15 min
error_after           = 4 h 30 min

age of a missed load at the next cutoff = 24 h + 4 h = 28 h
28 h exceeds error_after by 23 h 30 min
```

On 3 March the observed age was 27 h 56 min, which is the missed-load case less four minutes.
Both thresholds fire. The previous configuration of 24 hours and 48 hours produced a warning at
27 h 56 min and no error, which is why the plan ran.

The completeness check is a separate assertion:

```text
expected partitions = 6 sites x 1 load date
test returns a row when count(distinct site_id) <> 6
for rows where load_date = current_date
```

The two checks fail independently. A feed can be fresh and incomplete, or stale and complete.
Neither substitutes for the other.

### 13.3 Exercise 6 arithmetic

```text
full refresh per run   = 41.0 / 60 x 2.0 = 1.3667 credits
incremental per run    =  4.2 / 60 x 2.0 = 0.1400 credits

runs per year          = 5 x 52 = 260
full annual credits    = 1.3667 x 260 = 355.3 credits
incremental annual     = 0.1400 x 260 =  36.4 credits
saving                 = 318.9 credits
saving in currency     = 318.9 x $3.00 = $956.70

build window saved     = 41.0 - 4.2 = 36.8 min per run
annual window saved    = 36.8 x 260 = 9,568 min = 159.5 h
```

The credit saving is small relative to the maintenance cost. At twenty hours per year and a loaded
rate of $110 per hour, maintenance is $2,200, which exceeds the $957 saving. The justification must
therefore be the 159.5 hours of schedule margin and the reduced chance of missing a hard cutoff.
If a missed cutoff costs $15,000 in expedites once a quarter, avoiding one such event per year is
worth more than the entire credit saving. If the cutoff has never been missed and there is no
queue contention, the incremental model is not justified on these numbers, and the honest answer
is to keep the table. The decision reverses if the table grows by an order of magnitude, if the
queue becomes contended, or if the late-arrival tail shortens.

### 13.4 Rubrics for the open exercises

A strong answer names the claim being tested, the grain at which it is tested, and the blind spot
the test cannot see. A weak answer lists tools or restates a definition.

- **Exercise 1.** Strong: eight or more nodes, grain at every node, three edges marked as business
  assumptions, one untested edge named with the test that would close it. Weak: boxes in order,
  grain only on marts, every arrow treated as a data dependency.
- **Exercise 2.** Strong: a grain sentence that decides whether two rows are duplicates; composite
  uniqueness on the fact plus a version key and overlap test on the dimension. Weak: "one row per
  item per week" with an undefined week; `unique` on a surrogate.
- **Exercise 4.** Strong: sign convention explicit, zero tolerance for integer units,
  snapshot-preserving join, common-mode blind spot named. Weak: unsigned sums, tolerance hiding
  rounding, or two objects from the same conversion model compared and called validation.
- **Exercise 7.** Strong: grain, owner, cadence, and at least one semantic limit; a real consumer
  with a `ref()` dependency. Weak: column lists, an ownerless dashboard link, or a description
  that restates the model name.
- **Exercise 8.** Strong: separates definitional questions from data questions and proposes checks
  cheaper than the failure, including review where a test is the wrong instrument. Weak: only data
  quality issues, one test each.

## 14. Interview questions

**1. "What does a dbt test actually assert?"**

A strong answer says that a data test is a query returning failing rows; zero rows means pass. It
distinguishes generic tests, which are parameterised predicates on columns, from singular tests,
which are arbitrary SQL assertions, and from unit tests, which compare mocked input to expected
output before materialisation. A weak answer says "it checks the data is correct", which confuses
a predicate with a proof.

**2. "What is the difference between a data test and a unit test in dbt?"**

A strong answer places them on either side of materialisation: a unit test runs before the model
is built and tests the SQL against fixed inputs; a data test runs after and tests the data against
a predicate. It notes that unit tests arrived in dbt v1.8 and that the YAML key was renamed from
`tests:` to `data_tests:` to make the distinction explicit. A weak answer treats them as synonyms
or says unit tests are for Python.

**3. "How do you choose a materialisation?"**

A strong answer starts from the read pattern and the rebuild cost, computes annual build credits
and the query amplification of a view, and notes that dbt's own advice is to start with views and
move to tables and then incremental as measured costs demand. It adds that consumer-facing marts
often invert the default. A weak answer says "tables for marts, views for staging" without a cost
or a condition.

**4. "When is an incremental model the wrong choice?"**

A strong answer names at least three conditions: when the table is small enough that a rebuild is
cheap, when late-arriving data is common enough that the lookback cannot cover it, and when the
whole team cannot afford the maintenance and idempotency testing. It also notes that incremental
models hide logic errors because a full refresh is rarely run. A weak answer says incremental is
always better for large tables, which ignores correctness.

**5. "What does source freshness measure, and what does it miss?"**

A strong answer says it measures the age of `max(loaded_at)` against warn and error thresholds. It
misses completeness, because a partial load can be fresh, and it misses correctness, because the
rows can be wrong. A weak answer treats freshness as a data quality guarantee.

**6. "How would you test conservation of units?"**

A strong answer compares the change in a snapshot to the net of signed movements at the same
grain, states the sign convention and zero tolerance for integers, and names the common-mode
blind spot: if both sides share a conversion error, the test passes. It adds an independent
control such as a physical count. A weak answer sums quantities and compares totals without a
grain, which can hide offsetting errors.

**7. "What belongs in a model contract, and what does it not enforce?"**

A strong answer says a contract declares column names and data types and is checked before the
model builds; constraints such as not-null may be enforced depending on the warehouse. It does not
enforce values, business meaning, or uniqueness on every platform. A weak answer says a contract
is a test, which conflates structure with content.

**8. "Why are exposures part of the data model?"**

A strong answer says an exposure makes an external consumer visible in the DAG, enables selection
such as `+exposure:name`, and creates a warning point before a model is deprecated. It is metadata
only and creates no object. A weak answer describes it as a dashboard link.

**9. "A project has a green suite and the plan is wrong. How?"**

A strong answer gives mechanisms: tests encode schema rather than semantics, a shared upstream
assumption is invisible to every test built on it, the grain is wrong but consistently wrong, or
the definition of the metric differs from the contract. It gives an example such as demand
measured as shipments. A weak answer says the tests must be badly written, which is the least
likely explanation.

**10. "What would you delete from a 512-test suite?"**

A strong answer removes tests that restate a type or a source primary key, tests on columns no
consumer uses, and warnings that have never been actioned. It keeps and escalates key uniqueness,
temporal referential integrity, conservation, and freshness. It measures the suite's runtime and
failure rate before and after. A weak answer deletes tests to make the build faster without
distinguishing coverage from noise.

**11. "How do you test that an incremental model is idempotent?"**

A strong answer builds the model twice on unchanged input and compares row count and a checksum
over the grain and measures, or compares the incremental build with a full-refresh build of the
same model in continuous integration. It notes that a uniqueness test will not catch duplicates
when the key includes an ingestion timestamp. A weak answer says the primary key test covers it.

## 15. Key terms

- **dbt** — SQL and Jinja transformation framework that compiles models into warehouse objects.
- **Model** — one select statement compiled to a relation; the unit of the DAG.
- **Source** — a declared raw table that dbt reads but does not own.
- **Staging model** — one-to-one model over a source; rename, cast, categorise, no joins.
- **Intermediate model** — purpose-built re-grain or join, not exposed to consumers.
- **Mart** — business-conformed fact or dimension consumed by BI or another service.
- **DAG** — the dependency graph formed by `ref()` and `source()` calls.
- **Materialisation** — how a model is persisted: view, table, incremental, ephemeral, or
  materialised view.
- **Incremental model** — a persistent table to which only new or changed rows are added.
- **`unique_key`** — the column set that identifies a row for incremental updates.
- **Lookback window** — the trailing period reprocessed to absorb late-arriving data.
- **Full refresh** — rebuilding a model from scratch, with the risk of destroying history.
- **Data test** — a SQL assertion that returns failing rows; zero rows means pass.
- **Generic test** — a reusable parameterised test such as `unique` or `relationships`.
- **Singular test** — a one-off SQL assertion stored under `tests/`.
- **Unit test** — a mocked-input, expected-output test of transformation logic before build.
- **`accepted_values`** — a generic test restricting a column to an enumerated domain.
- **Composite uniqueness** — a test that a combination of columns is unique, expressing a grain.
- **Severity** — whether a failing test warns or errors and blocks downstream models.
- **Source freshness** — the age of the newest loaded row compared with thresholds.
- **Exposure** — metadata declaring an external consumer of a model.
- **Contract** — enforced column names and data types checked before materialisation.
- **Invariant** — a property that must hold for every row or period, such as conservation.
- **Grain** — the exact meaning of one row, stated as a sentence.

## 16. Sources

**Peer-reviewed and academic**

- Kimball, R., and Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to
  Dimensional Modeling*, 3rd edition. Wiley. Source for grain, additive and semi-additive
  measures, and slowly changing dimensions; not specific to dbt. Grade A for the modelling
  concepts.
- Reis, J., and Housley, M. (2022). *Fundamentals of Data Engineering*. O'Reilly. Background on
  batch and incremental processing and on the separation of storage and compute. Grade B as a
  practitioner reference.

**Standards and official**

- ISO/IEC 9075, *Information technology — Database languages — SQL*.
- ISO 8601, *Date and time — Representations for information interchange*.
- UN/CEFACT Recommendation 20, *Codes for Units of Measure Used in International Trade*.
- Unified Code for Units of Measure (UCUM), specification and implementation guide.
- GS1, *General Specifications*, current edition, including GTIN, GLN, and SSCC.
- OpenLineage specification, OpenLineage project.
- Semantic Versioning 2.0.0, semver.org.
- Open Data Contract Standard, Bitol project.

**Vendor documentation (Grade D unless the claim is behavioural, in which case B)**

- dbt Labs, "Materializations", dbt Developer Hub, accessed November 2026.
  https://docs.getdbt.com/docs/build/materializations
- dbt Labs, "How we structure our dbt projects" (overview, staging, intermediate), accessed
  November 2026. https://docs.getdbt.com/best-practices/how-we-structure/1-guide-overview
- dbt Labs, "How we style our dbt models", accessed November 2026.
  https://docs.getdbt.com/best-practices/how-we-style/1-how-we-style-our-dbt-models
- dbt Labs, "Add data tests to your DAG", accessed November 2026.
  https://docs.getdbt.com/docs/build/data-tests
- dbt Labs, "Unit tests", accessed November 2026. https://docs.getdbt.com/docs/build/unit-tests
- dbt Labs, "About incremental models", "About incremental strategy", and "About microbatch
  incremental models", accessed November 2026.
  https://docs.getdbt.com/docs/build/incremental-models-overview
- dbt Labs, "Add sources to your DAG" and freshness reference, accessed November 2026.
  https://docs.getdbt.com/docs/build/sources
- dbt Labs, "Model contracts", accessed November 2026.
  https://docs.getdbt.com/docs/mesh/govern/model-contracts
- dbt Labs, "Test smarter not harder" and "Announcing unit testing", dbt Developer Blog, accessed
  November 2026. https://docs.getdbt.com/blog/test-smarter-not-harder
- dbt Labs, "dbt-project-evaluator" governance rules, accessed November 2026.
  https://dbt-labs.github.io/dbt-project-evaluator/main/rules/governance/

**Practitioner and critical commentary (Grade E)**

- Adam Analytics, "dbt Changed Data Engineering. Here Is What It Got Wrong.", accessed November
  2026. https://adam-analytics.com/dbt-changed-data-engineering-here-is-what-it-got-wrong/
- Dalanio, "Why dbt Tests Alone Cannot Catch Freshness, Volume, or Schema Drift", accessed
  November 2026. https://dalanio.com/blog/dbt-tests-freshness-gap/
- Architect Knowledge Library, "Transformation: dbt", accessed November 2026.
  https://erikevenson.github.io/architect/providers/dbt/transformation/

These are opinion and vendor-adjacent sources. The qualitative mechanisms they describe are
consistent with the documentation; their numeric claims are not accompanied by published methods
and are not repeated here.

**Tooling**

- Elementary, open-source data observability for dbt, GitHub repository and documentation,
  accessed November 2026. https://github.com/elementary-data/elementary/ and
  https://docs.elementary-data.com/

**Note on citations.** No controlled study or audited case was found that measures the effect of
dbt testing on planning outcomes. Claims in section 9 that sound like outcome claims are
attributed and graded E; where the text states a mechanism, it is verifiable from documentation or
from the SQL itself.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite: [`01_layered-architecture.md`](01_layered-architecture.md)
- Prerequisite: [`04_sql-supply-chain-patterns.md`](04_sql-supply-chain-patterns.md)
- Sibling: [`05_python-structure-tests.md`](05_python-structure-tests.md)
- Sibling: [`07_orchestration.md`](07_orchestration.md)
- Planned sibling: `11_quality-quarantine-lineage.md` (not yet written; failing loudly, quarantine,
  versioning, and lineage)
- Quality dimensions used throughout: [`../02_supply_chain_data_landscape/08_data-quality-dimensions.md`](../02_supply_chain_data_landscape/08_data-quality-dimensions.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
