> Module 03 — Data Engineering and Interoperability
> Topic 01 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 02

# Layered Architecture and the Medallion Pattern

---

## 1. Why this topic exists

At 06:00 a replenishment planner opens a workbench and sees an inventory position of 2,926 units
for a fast-moving item at the Leeds distribution centre (DC). The warehouse manager, on the same
morning, counts 3,180. Both numbers were produced by correct queries against governed tables. The
meeting resolves the difference by seniority, and the order is placed on whichever number the more
senior person prefers.

The gap is 254 units, 8.0% of the physical count. It is not a query defect. It is a boundary
defect. The warehouse management system (WMS) writes its inventory snapshot at 23:00. Put-away
transactions continue until 23:30. The number the planner sees is the warehouse as it was before
the last half hour of work, and nobody wrote down that the snapshot cut-off and the posting
cut-off were different. The pipeline is faithful, fast, and wrong about the question.

This topic is about the machinery that would have caught that. A layered architecture is not a
folder convention. It is a set of boundaries at which one party promises something to another,
with a test that can fail and an owner who is accountable when it does. The layers are the places
where those promises live.

Three claims organise what follows.

1. **A layer is defined by what it may not do.** The useful description of a raw layer is not
   "stores raw data" but "never overwrites, never types, never rejects." Prohibitions are what
   make a layer predictable, and predictability is what makes the next layer's promise credible.
2. **The boundary is the unit of design, not the layer.** Teams argue about whether there should
   be three layers or five. The argument that matters is what crosses each boundary: which keys,
   which units, which timestamp, which completeness guarantee, which owner.
3. **Replay is the test of whether the architecture exists.** If a corrected source file cannot be
   reprocessed for an arbitrary past date and produce a reproducible result, the layers are
   decoration. Replay is the property that turns a pipeline into evidence.

The vocabulary in this topic is largely vendor vocabulary. Bronze, silver, and gold are names
popularised by Databricks, whose documentation describes the medallion pattern as "a recommended
best practice but not a requirement" (page dated 29 October 2026). The names are useful because
they are widely understood. They are dangerous because they suggest that naming a layer creates
its guarantees. It does not. A folder called `gold` with no tests is a folder.

**Why this matters to a data scientist specifically.** Every model in this program consumes a
mart. The mart's grain decides what the model can learn; its freshness decides what the model can
influence; its key policy decides whether a customer is one customer or three; its as-of semantics
decide whether a backtest is honest. A model trained on a mart built directly from a source, with
no conformed layer, inherits every entity-resolution error and every unit drift in the source and
cannot be repaired downstream. The most expensive modelling failures in supply chain analytics
are usually architecture failures that surface late.

The previous topic in Module 02 chose a source of record for each field of a decision. This topic
decides where that source lands, what may happen to it, and who promises what at each step. Topic
11 returns to the quality machinery that enforces those promises.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** a four-layer stack (raw, staging, conformed, mart) for a named pipeline, labelling the
   contract and the owner at every boundary.
2. **State** what each layer may and may not do, and **justify** each prohibition by the failure it
   prevents rather than by convention.
3. **Map** the four-layer model onto the medallion names, and **identify** the boundary that the
   three-name vocabulary hides.
4. **Design** an immutable raw layer and a replay procedure that reproduces a past planning run
   from declared inputs.
5. **Distinguish** a planning mart from an analytics warehouse on grain, state, cadence, and
   write behaviour, and **size** a refresh window against a planning cut-off.
6. **Diagnose** a wrong number by walking backwards through layers until a layer stops
   reproducing, and **name** the boundary at which the defect entered.
7. **Critique** the "one big table" architecture and **distinguish** it from a wide mart published
   at the end of a layered pipeline.
8. **Write** a boundary contract with a producer, consumer, promise, test, freshness target, and
   breach action, and **explain** why the contract is not the schema.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a stack of four layers, with a contract drawn as a horizontal line at every
boundary, and an owner named on each line. Data moves upward. Promises move downward. The number
of layers is negotiable; the existence of a written promise at each boundary is not.

```text
        CONSUMERS   planners | forecast models | reporting | auditors | regulators
             ^
             |  CONSUMPTION CONTRACT
             |  grain, freshness, as-of semantics, deprecation notice
             |
   +-----------------------+
   |  MART                 |  owner: analytics engineering, with the decision owner
   |  decision-grain facts |  may:  aggregate, join across domains, publish, index
   |  dims, plan versions  |  may not: invent keys, read a source, hide a filter
   +-----------------------+
             ^
             |  PUBLICATION CONTRACT
             |  grain declaration, additivity, test results, version, access class
             |
   +-----------------------+
   |  CONFORMED            |  owner: data engineering plus master data management
   |  canonical keys       |  may:  resolve entities, version history, convert units, apply rules once
   |  history, as-of, dates|  may not: aggregate to a decision, contain presentation logic
   +-----------------------+
             ^
             |  INTEGRATION CONTRACT
             |  key map, unit base, history policy, business-rule registry, effective dates
             |
   +-----------------------+
   |  STAGING              |  owner: data engineering, aligned to one source system
   |  one model per source |  may:  rename, cast, dedupe, flatten, flag, quarantine
   |  table, typed, deduped|  may not: join across sources, aggregate, apply business rules
   +-----------------------+
             ^
             |  INGESTION CONTRACT
             |  schema version, file naming, arrival window, completeness signal
             |
   +-----------------------+
   |  RAW                  |  owner: platform and ingestion engineering
   |  immutable, append-   |  may:  append, partition by ingest time, record metadata
   |  only, source form    |  may not: clean, type, dedupe, reject, overwrite
   +-----------------------+
             ^
        SOURCES   order systems | stock systems | freight | partner files | till data | portals
```

The four layers do not map one-to-one onto the three medallion names. That mismatch is the most
useful thing in this section.

| Four-layer role | Medallion name | dbt convention | Data Vault equivalent | What the name hides |
|---|---|---|---|---|
| Raw | Bronze | Sources | Staging area (persistent) | That raw is evidence of receipt, not truth |
| Staging | Silver (first half) | Staging models | Raw vault loading | That cleaning and integration are two jobs |
| Conformed | Silver (second half) | Intermediate models | Raw vault plus business vault | That entity resolution is the expensive part |
| Mart | Gold | Marts | Information mart | That a mart is a product with a decision owner |

**The hidden boundary.** Medallion's "silver" contains two distinct responsibilities: making one
source trustworthy on its own terms (typing, deduplication, unit normalisation) and making several
sources agree on entities, keys, and time (integration). The first is source-conformed. The second
is business-conformed. They have different tests, different owners, and different failure modes. A
team that collapses them builds staging models that join across systems, and then cannot say
which source caused a bad key. Splitting silver into staging and conformed is the single most
consequential refinement of the medallion vocabulary.

Five properties follow.

**Property 1 — Refinement is monotonic.** No mart row may exist without a conformed parent, no
conformed row without a staging parent, no staging row without a raw parent. "Parent" means a
traceable record, not a similar value. This is the property the query in section 7 tests. When it
fails, the architecture is a set of copies rather than a chain of evidence.

**Property 2 — Each boundary is a promise with an owner.** A boundary without an owner is a
suggestion. The ingestion contract promises that a file arriving by 02:00 will be readable and
complete. The integration contract promises that a case-pack change takes effect on its effective
date, not on the date the extract ran. The publication contract promises that the mart's grain is
item-location-day and that the freshness timestamp means "all source records received by 02:00."

**Property 3 — Raw is evidence, not truth.** Databricks' documentation calls bronze the "single
source of truth" because it preserves source fidelity. That phrasing is defensible for fidelity
and misleading for meaning. The raw layer is the immutable record of what the company received and
when. If the source later restates, raw holds both versions. It is the best evidence of what was
said, not a statement about what is true.

**Property 4 — The number of layers is an argument, not a fact.** Three, four, and five are all
defensible. What is not defensible is a layer that has no prohibition, no owner, and no test. A
team with four well-contracted layers will outperform a team with seven folders and no promises,
every time.

**Property 5 — A mart is a product, not a dump.** The mart exists for a decision with an actor, a
cadence, and a cut-off. If nobody can name the decision, the table is an extract with a schema.
Planning marts and analytics warehouses are different products with different obligations, and
section 6.6 treats that difference as the design constraint it is.

### 3.1 The one big table, stated precisely

A **one big table (OBT)** is a single wide denormalised table containing facts and their
dimensions side by side, joined once at build time. As a *published mart*, it is a legitimate
performance and usability choice: one table, no joins for the consumer, fast scans, easy feature
generation. As an *architecture*, it is a failure, because it has no upstream to replay from and
no boundary at which a definition can be tested.

| Property | OBT as a mart | OBT as the architecture |
|---|---|---|
| Where it sits | Top of a layered pipeline | The only layer |
| Source of truth | Raw and conformed upstream | The table itself |
| Change to one definition | Edit one model, rebuild | Rebuild everything, with no reference |
| Replay | Deterministic from raw | Impossible; last build overwrote the inputs |
| Entity resolution | Done once in conformed | Baked into the join, invisible |
| Test placement | At every boundary plus the mart | At the end, after the damage |
| Unit drift | Caught in staging | Propagates into history |
| Who can answer "why" | Lineage to raw | Nobody |

The distinction is not stylistic. An OBT published from conformed data can be rebuilt; an OBT that
is the only table cannot, because the joins were never captured as reproducible logic and the
source extracts it replaced are gone.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Layer | A named stage in a data pipeline with an explicit grain, prohibition set, owner, and retention rule | Used to mean a folder or database schema | Layer name plus version |
| Raw layer | The immutable, append-only copy of source records exactly as received, with ingestion metadata | Called "the clean layer" or "the source of truth" | Ingest partition, usually date plus batch identifier |
| Bronze | The medallion name for the raw layer | Assumed to guarantee immutability because of the name | Layer name only; no formal specification |
| Staging layer | Source-aligned models that type, rename, deduplicate, and flatten one source table at a time | Used for cross-source joins and business logic | One model per source table |
| Silver | The medallion name for cleaned and enriched data, covering both staging and integration work | Treated as one homogeneous layer with one owner | Layer name only |
| Conformed layer | The integrated layer where entities are resolved, keys are canonical, units are normalised, and history is versioned | Confused with "cleaned" or "deduplicated" | Canonical key plus effective date |
| Gold | The medallion name for business-level, consumption-ready tables | Assumed to mean "tested" or "governed" | Layer name only |
| Mart | A table or schema at the grain of a decision, with declared additivity, freshness, and access class | Used for any table a team finds convenient | Decision grain plus version |
| Planning mart | A mart that feeds a scheduling or replenishment decision, carries plan versions, and is refreshed against a decision cut-off | Treated as a report or a copy of the warehouse | Item-location-period, with plan run identifier |
| Analytics warehouse | A read-mostly store of historical, integrated data for exploration, reporting, and model training | Assumed to be usable as a planning input | Query grain varies; history in years |
| Operational data store (ODS) | A subject-oriented, volatile store of recent operational data, usually close to source latency | Confused with a data warehouse or a staging area | Current plus short history, days to weeks |
| Planning data store (PDS) | The planning application's own store of plan inputs and plan outputs, often partitioned by plan identifier | Treated as a general-purpose warehouse | Plan identifier plus scenario identifier |
| Data lake | A storage repository holding raw data in native formats, usually object storage, with schema applied on read | Treated as a synonym for a data platform | Object path plus format |
| Lakehouse | An architecture combining object storage, open table formats, and warehouse-style management | Treated as a product rather than a pattern | Table format plus catalog |
| Medallion architecture | A multi-hop design pattern using bronze, silver, and gold layers of increasing refinement | Treated as a standard or a certification | Pattern name only |
| One big table (OBT) | A single wide denormalised table joining facts to dimensions at build time | Treated as an architecture rather than a mart pattern | Row grain of the widest input |
| Star schema | A fact table joined to denormalised dimension tables by surrogate keys | Confused with any schema containing a fact table | Fact grain plus dimension keys |
| Conformed dimension | A dimension table with consistent keys, attributes, and definitions across marts | Assumed to mean "copied everywhere" | Dimension name plus version |
| Bus matrix | A grid of business processes against conformed dimensions, used to plan integration | Treated as a project plan | Process by dimension |
| Third normal form (3NF) | A relational design in which non-key attributes depend on the key, the whole key, and nothing but the key | Used to mean "any normalised schema" | Normal form level |
| Data Vault | A modelling approach using hubs, links, and satellites, with raw and business vault layers | Treated as a competitor to dimensional modelling rather than a complementary integration layer | Model release version |
| Hub | A Data Vault table holding a unique list of business keys for one entity | Confused with a dimension table | Business key plus load timestamp |
| Link | A Data Vault table holding relationships among hubs, including transaction relationships | Confused with a fact table | Link key plus load timestamp |
| Satellite | A Data Vault table holding historised descriptive attributes for a hub or link | Confused with a slowly changing dimension | Parent key plus load timestamp |
| Business vault | The Data Vault layer holding soft business rules, calculations, and performance constructs | Treated as a presentation layer | Rule set version |
| Slowly changing dimension (SCD) type 2 | A dimension design that adds a new row for each attribute change, with effective and expiry dates | Applied without an as-of join, which defeats it | Effective-from and effective-to dates |
| Grain | The exact set of entities and time span that one row represents | Described as "daily data" without naming the entities | Entity list plus time span |
| Additivity | The property that a measure can be summed across a stated dimension without distortion | Assumed for ratios, prices, and inventory snapshots | Additive, semi-additive, non-additive |
| Idempotent | A process that produces the same result whether run once or many times | Confused with "safe to retry" without a key | Run identifier plus target key |
| Replay | Reprocessing historical inputs to reproduce a past output | Used to mean "rerun the job on today's data" | Source partition range plus code version |
| Immutability | The property that an existing record is never modified in place | Confused with "never deleted" | Record identifier plus version |
| Append-only | A write pattern that adds records and never updates or deletes existing ones | Assumed to imply immutability of the storage system | Ingestion timestamp plus sequence |
| Snapshot | A copy of state at a point in time, usually losing intra-period change detail | Treated as a backup or as a transaction history | Snapshot date plus scope |
| Data contract | A versioned, testable agreement between a data producer and its consumers | Treated as a schema or a wiki page | Contract version plus effective date |
| Freshness | The age of the oldest source record reflected in a dataset at a stated time | Reported as job completion time | Minutes or hours, measured at p95 (95th percentile) |
| Lineage | The technical graph of inputs, transformations, and outputs | Confused with business meaning | Node and edge identifiers |
| Quarantine | The isolation of records that fail validation, with the failure reason retained | Deleted after inspection | Quarantine table plus reason code |
| Cut-off | The time at which a decision's inputs are frozen | Confused with job start time | Local time plus time zone |
| Deterministic build | A transformation whose output depends only on declared inputs and code version | Assumed because the query looks simple | Input manifest plus code commit |
| Tombstone | A record marking the deletion of an entity without removing its history | Treated as a physical delete | Entity key plus deletion timestamp |

Six of these deserve expansion, because the distinctions are routinely lost.

**Raw and bronze.** In practice these are often the same storage bucket. The distinction is
functional. Files arrive in a landing area; raw is the parsed, partitioned, metadata-enriched record
of arrival; bronze is the vendor's name for that role. A team can have a landing area and no raw
layer, if files are overwritten or deleted after loading. That team cannot replay.

**Staging versus conformed.** Staging is source-conformed: it makes one source usable on its own
terms. Conformed is business-conformed: it makes several sources agree. The dbt convention of
staging models that never join across sources is the same contract in tooling form. When a team
puts a join between the order system and the item master into a staging model, it has moved the
integration boundary without moving the tests or the owner.

**Snapshot versus transaction.** A snapshot records state at a time; a transaction records a
change. Summing snapshots over days multiplies inventory by the number of days. Summing
transactions over days gives flow. The two belong in different fact tables with different
additivity declarations, and mixing them is one of the most common mart defects in supply chain.

**Data contract versus schema.** A schema says which columns exist and what types they have. A
contract says who produces them, for whom, by when, with what completeness, under what change
process, and what happens when a promise is broken. A schema can be generated from a table. A
contract has to be agreed.

**Planning mart versus analytics warehouse.** The planning mart is an input to a decision with a
clock. It must be fresh at the cut-off, it must carry the plan version that was used, and it must
be writable enough to hold scenarios. The analytics warehouse is an input to learning. It must be
long, flexible, and read-mostly. A company that serves both from one table will fail one of them,
usually the planning one, because the warehouse's nightly completion time is set by reporting
convenience rather than by the planning clock.

## 5. The mechanics, step by step

The procedure has eight steps and one precondition. The precondition is a decision with a clock.
Without it, every later choice is arbitrary. The order matters because each step constrains the
next: the raw contract determines what replay is possible, and replay determines what the mart can
promise.

### 5.1 Step 0 — Fix the decision, its grain, and its cut-off

Write the decision as an action with a cadence, a cut-off, and a tolerance. The tolerance is the
smallest input error that would change the action, derived in Module 02, Topic 08. It sets how much
freshness and completeness the architecture must buy.

| Decision | Grain | Cut-off | Tolerance | What lateness costs |
|---|---|---|---|---|
| Daily DC replenishment | Item x DC x day | 06:00 local | 3% on inventory position | Order misses the outbound truck |
| Weekly supply review | Product family x month | Friday 12:00 | 5% on supply | Review deferred a week |
| Monthly sales and operations planning (S&OP) | Family x month | Working day 3 | 5% on demand and supply | Commitment made on stale plan |
| Annual network design | Lane x year | Fixed project date | 10% on volume | Scenario invalid, decision delayed a quarter |
| Control-tower exception | Shipment x event | 15 minutes | Event-level exactness | Intervention after the window closes |

The cut-off is not the job start time. It is the time at which inputs are frozen. A pipeline that
starts at 02:00 and reads a source that continues to change until 05:00 has no cut-off; it has a
race.

### 5.2 Step 1 — Write the layer charter

The charter is the document that makes the architecture real. It states, for each layer, what the
layer may do, what it may not do, who owns it, how long it is kept, and which tests must pass
before data may leave it.

| Layer | May do | May not do | Owner role | Retention | Mandatory tests before promotion |
|---|---|---|---|---|---|
| Raw | Append records; partition by ingest date and batch; store source metadata; retain every version received | Overwrite; delete on correction; type or clean values; reject a record at the door; apply business rules | Ingestion engineering | 13–25 months, longer where regulation requires | File completeness against manifest; checksum; arrival timestamp recorded |
| Staging | Rename; cast; flatten nested structures; deduplicate on a declared key; normalise units; flag invalid rows into quarantine | Join across source systems; aggregate to a business grain; apply pricing, allocation, or eligibility rules; overwrite history | Data engineering, source-aligned | 25–37 months | Row count against raw; key uniqueness; type conformance; quarantine rate |
| Conformed | Resolve entities; assign surrogate keys; implement SCD type 2; convert to base units and base currency; apply documented business rules once; expose as-of joins | Aggregate to a decision grain; contain dashboard logic; read from a mart; invent a key without a mapping table | Data engineering plus master data management (MDM) | 3–7 years | Referential integrity; one current row per entity; effective-date coverage; unit reconciliation |
| Mart | Aggregate; join across domains; publish at decision grain; index for performance; expose plan versions and as-of timestamps | Read a source directly; invent keys; hide a filter that changes the population; publish without a grain declaration | Analytics engineering with the decision owner | 3–7 years for history; current plus horizon for planning | Grain uniqueness; additivity; freshness; reconciliation to conformed totals |

Three prohibitions carry most of the value, and each maps to a failure.

- **Raw may not overwrite.** If a corrected file replaces the original, the company loses the
  ability to answer what it knew when. The correction should append a new partition with a new
  batch identifier and a reason code.
- **Staging may not join across sources.** A staging model that joins orders to the item master
  has made an integration decision inside a layer that has no integration tests. When the join
  produces duplicates, nobody can tell whether the order feed or the master caused it.
- **A mart may not read a source.** A mart that queries a source system directly creates a second,
  untested path to the decision. The path will drift from the conformed path, and the two will
  disagree at the worst moment.

### 5.3 Step 2 — Land raw without interpreting

Raw ingestion has one job: capture the record and the circumstances of its arrival. The record
keeps the source's field names, types as strings where the source is unstable, and every row,
including the malformed ones. The circumstances are metadata columns, added by the loader, not by
the source.

| Metadata column | What it records | Why it is needed |
|---|---|---|
| `_ingest_ts` | When the platform received the record | Separates event time from processing time |
| `_ingest_date` | Partition key, usually the Coordinated Universal Time (UTC) date of receipt | Enables replay by arrival cohort |
| `_source_file` | File name or application programming interface (API) cursor | Traces a row back to the delivered artefact |
| `_batch_id` | Identifier of the load attempt | Distinguishes a retry from a correction |
| `_schema_version` | Version of the source schema observed at load | Explains why a column is null |
| `_record_hash` | Hash of the source record | Detects exact duplicates without parsing |
| `_is_deleted` | Tombstone flag from the source, if supplied | Preserves deletions without physical removal |

Four rules govern this step.

1. **Do not reject at the door.** A record that fails a type check is still evidence that the
   source sent it. Rejecting it at ingestion destroys the evidence and makes the failure
   invisible. Accept it, flag it, and quarantine it in staging.
2. **Do not trust the source's timestamp as the arrival time.** Event and processing time differ.
   A pipeline that orders by event time alone drops records that arrive after a watermark.
3. **Partition by arrival, not only by event date.** A correction to a January shipment delivered
   in March belongs in a March partition. Partitioning only by event date forces an overwrite of
   January, which is exactly the operation raw must not perform.
4. **Record the manifest.** The file count and row count the source promised, and the count
   received, are the completeness signal the ingestion contract needs.

> **Note:** Raw storage is cheap relative to the cost of an unreproducible decision, but it is not
> free. A mid-size retailer at 10 million order lines per day and roughly 500 bytes per record
> produces about 5 GB per day before compression, 1.8 TB per year, and about 365 GB per year at
> 5:1 columnar compression. Retention policy is a cost decision, and it should be made explicitly
> rather than by a lifecycle rule nobody reviewed.

### 5.4 Step 3 — Stage one model per source table

Staging is where the source becomes usable without becoming integrated. The dbt project
conventions are a good statement of the prohibition: staging models are materialised as views,
there is one staging model per source table, they select only from declared sources, and they do
not join or aggregate except through an explicitly named base model. The naming convention
`stg_<source>__<entity>s` is not cosmetic; it makes the one-to-one relationship visible in the
file tree.

The work permitted in staging is narrow and mechanical.

- Rename columns to the organisation's vocabulary, keeping a mapping to the source name.
- Cast types explicitly, with a policy for failed casts that routes the row to quarantine.
- Flatten nested structures, preserving the path to each field.
- Deduplicate on a declared natural key, with a deterministic ranking.
- Normalise units to the base unit declared in the conformed layer, not to a unit invented locally.
- Flag invalid rows with a reason code and write them to a quarantine table.

Deduplication deserves its own rule because it is where replay usually breaks. A deduplication
that ranks by `source_updated_at` alone is non-deterministic when two rows share a timestamp. The
ranking must include a tie-breaker that is stable across runs, such as the ingestion timestamp
plus the source file name, and the choice must be documented. A deduplication that returns a
different survivor on a rerun makes every downstream number non-reproducible.

| Deduplication input | Deterministic? | Failure it causes |
|---|---|---|
| `row_number()` over source key only | No | Arbitrary survivor; counts change between runs |
| `row_number()` over key order by source update time | No when timestamps tie | Silent row flips; mart totals drift |
| `row_number()` over key order by update time, ingest time, file name | Yes | None, if the source time is trustworthy |
| `qualify row_number()` with a hash tie-breaker | Yes | Stable, but the survivor may not be the latest business version |

### 5.5 Step 4 — Conform: keys, units, time, and entities

The conformed layer is where the expensive, irreversible decisions are made. It is also the layer
most often skipped, because staging looks like progress and marts look like delivery. Skipping it
moves entity resolution and unit conversion into each mart, where they are duplicated and
untested.

**Keys.** Every entity gets a surrogate key and at least one business key with a mapping table.
The mapping table is the audit trail: it says which source identifiers were merged, when, under
which survivorship rule, and by whom. Without it, a merged customer cannot be unmerged.

**Units.** One base unit per dimension: eaches for count, kilograms for mass, litres for volume,
the company's reporting currency for money. Conversion factors live in a versioned reference
table with effective dates, not in a query expression. A case-pack change is a row in that table,
not an edit to a model.

**Time.** Two clocks are stored: valid time (when the fact was true) and transaction time (when
the warehouse held it). The conformed layer exposes an as-of join that takes a timestamp and
returns the version current at that timestamp. This is the mechanism that lets a backtest use what
was known at the decision, rather than what is believed now.

**Entities.** Entity resolution applies survivorship rules: which name, address, and identifier
survive when two records are judged to be the same entity. The rules are business rules, not
technical ones, and they belong in the rule registry with an owner. A conformed layer that merges
suppliers by fuzzy name match without a review queue will eventually merge two real suppliers and
misstate spend.

> **Note:** The conformed layer is not a normalised enterprise model in the Inmon sense, although
> it can be. It is the layer where cross-source agreement is established. A star schema can serve
> as the conformed layer if its dimensions are conformed and its history is versioned. The test is
> not the shape of the tables; it is whether every mart can join to the same keys and get the same
> answer.

### 5.6 Step 5 — Publish marts at the decision grain

A mart is defined by a grain declaration, an additivity declaration, a freshness statement, and an
access class. The declaration is short and it is the most useful documentation in the pipeline.

| Declaration | Example | What it prevents |
|---|---|---|
| Grain | One row per item x DC x day, including days with zero movement | Double counting from event-only rows |
| Additivity | On-hand is semi-additive over time and additive over locations; receipts are fully additive | Summing inventory across days |
| Population | Active items only, where active means at least one receipt in 90 days | Silent changes in denominator |
| Freshness | All source records received by 02:00; published by 04:30 | "Real-time" claims that mean "last night" |
| As-of | `valid_from` and `valid_to` on item attributes; mart row carries `known_at` | Backtests on restated attributes |
| Access | Planner role: read; planning service: read and write scenarios | Mart used as a transaction system |
| Deprecation | Superseded tables carry a removal date and a successor | Two marts with different answers |

Marts should be few. A useful discipline is one mart per decision, not one mart per team. When two
teams need the same entity at different grains, they need two marts that share conformed inputs,
not one mart that tries to serve both.

### 5.7 Step 6 — Make replay a first-class operation

Replay is the ability to reprocess a declared range of raw partitions with a declared code version
and produce a result that matches the original, or an explained difference. It is the operational
proof that the layers are real.

| Requirement | Implementation | Test |
|---|---|---|
| Immutable inputs | Raw partitions never overwritten; corrections append | Attempt to update a raw partition fails |
| Pinned code | Every run records the commit or model version | Two runs of the same version match |
| Pinned reference data | Calendar, foreign exchange (FX) rates, unit factors, and item master versions are inputs, not ambient state | Change a factor, replay, see the expected difference |
| No wall-clock dependence | No `current_date`, `now()`, or `rand()` in transformation logic | Freeze the clock; output unchanged |
| Idempotent writes | Merge on the natural key or replace the partition | Run twice; row count unchanged |
| Declared backfill range | A backfill names start and end partitions and a reason | Backfill log is queryable |
| Reconciled output | Replay compares key totals to the original run | Difference is zero or explained |

The common failures are mundane. A model filters `where order_date >= current_date - 365`, so a
replay in 2027 silently drops a year. A dimension assigns a surrogate key with a sequence that
restarts, so a replay reassigns keys and breaks every downstream reference. A deduplication ranks
by a timestamp with ties, so a replay picks a different survivor. None of these is visible in a
code review that reads for intent rather than for determinism.

### 5.8 Step 7 — Attach contracts and service levels to every boundary

A contract is written at the boundary, not inside the layer. It names the producer, the consumer,
the promise, the test, the measurement window, and the action on breach. The schema is one clause
of the contract, not the whole of it.

| Boundary | Producer | Consumer | Promise | Test | Breach action |
|---|---|---|---|---|---|
| Source to raw | Source system owner | Ingestion engineering | File by 02:00, 14 of 14 DCs, schema version declared | Arrival time, file count, row count against manifest | Load what arrived, flag the gap, notify the source owner |
| Raw to staging | Ingestion engineering | Data engineering | Every raw row represented once, quarantine reasons recorded | Raw-to-staging row reconciliation, quarantine rate | Block promotion; investigate the source extract |
| Staging to conformed | Data engineering | MDM and analytics engineering | Canonical keys, base units, SCD type 2 coverage from the declared start date | Referential integrity, one current row per entity, unit totals | Block publication; the defect is in integration |
| Conformed to mart | Analytics engineering | Decision owner | Decision grain, declared additivity, freshness by 04:30 | Grain uniqueness, reconciliation to conformed totals, freshness p95 | Publish with a defect flag or withhold; never silently |
| Mart to consumer | Decision owner | Planner, model, report | As-of semantics, population definition, deprecation notice | Consumer-side reconciliation sample, access audit | Deprecation notice; rollback to prior version |

Two rules make contracts enforceable. First, the test must run automatically at the boundary, not
in a quarterly review. Second, the breach action must be pre-agreed, because the moment of breach
is the worst time to negotiate. A contract whose breach action is "escalate" has not specified an
action.

### 5.9 Step 8 — Operate the layers as products

Layers decay. A source changes a column name, a mart adds a filter, a staging model acquires a
join "just this once." The operating rhythm is what keeps the prohibitions alive.

- **Daily:** freshness and completeness checks at each boundary; quarantine rate review.
- **Weekly:** reconciliation of mart totals to conformed totals; review of any manual override.
- **Monthly:** schema-drift report against declared contracts; access review; storage and cost review.
- **Quarterly:** layer charter review; deletion of unused marts; replay drill on a random past week.
- **On change:** contract version bump, consumer notification, deprecation window.

The replay drill is the most valuable and least common. Pick a week from six months ago, replay it
from raw with the recorded code version, and compare. A team that can do this has an architecture.
A team that cannot has a set of tables that happen to be updated on a schedule.

## 6. Worked example

Calder Health Supply distributes medical consumables to hospitals from 14 regional DCs. It runs a
daily replenishment decision at 06:00, plans twelve weeks ahead, and operates an enterprise
resource planning (ERP) system, a WMS, a transportation management system (TMS), and an electronic
data interchange (EDI) feed of hospital inventory reports. All figures are illustrative and
internally consistent.

### 6.1 The decision and the layer volumes

| Element | Value |
|---|---|
| Decision | Daily replenishment order from each DC to its hospital customers |
| Grain | Item x DC x day; planning horizon 12 weeks |
| Cut-off | 06:00 local; plan run starts 02:15 |
| Active items | 6,200 |
| Active DCs | 14 |
| Order tolerance | 3% on inventory position; 1 pallet rounding |
| Unit cost | EUR 11.40 per case |
| Annual holding cost | 22% of unit cost |

Monthly layer volumes after the pipeline was rebuilt:

| Layer | Rows per month | Storage per month | Retention | Owner |
|---|---:|---:|---|---|
| Raw | 41.0 million | 42 GB Parquet | 25 months | Ingestion engineering |
| Staging | 38.4 million | 31 GB | 37 months | Data engineering |
| Conformed | 12.1 million | 9 GB | 7 years | Data engineering and MDM |
| Mart | 1.04 million | 1.2 GB | 7 years plus horizon | Analytics engineering |

The mart row count follows from the grain: 6,200 items x 14 DCs x 12 weeks = 1,041,600 rows. The
number is worth checking because it is the first evidence that the grain declaration is true. A
mart with 40 million rows at that grain has a duplicate problem, not a bigger business.

### 6.2 Three defects, three boundaries

The rebuild began after a quarter in which planners reconciled inventory by hand every week. Three
defects were found, and each entered at a different boundary.

| Symptom | Where it first appeared | Layer where the cause lived | Mechanism |
|---|---|---|---|
| Inventory 8.1% below the morning count | Mart | Raw | WMS snapshot at 23:00; 41% of put-away postings land between 23:00 and 23:30 |
| Demand doubled for 18 weeks of history | Conformed test | Staging | Current case pack applied to all history after a pack change from 24 to 12 |
| 4,312 duplicate shipment rows | Mart | Staging | Deduplication ordered by update time with ties, no tie-breaker |

The pattern is the point: the symptom appeared at the top in two cases and in the middle in one.
Only the boundary audit found the cause, and the fix belonged to a different owner each time.

### 6.3 Tracing the inventory gap

The WMS writes a full inventory snapshot at 23:00. Put-away transactions post to the WMS until
23:30, when the shift closes. The staging model reads the 23:00 snapshot. The mart computes
inventory position as on-hand plus in-transit minus backorders, using the snapshot as on-hand.

For the Leeds DC on the observed day:

| Line | Units | Source |
|---|---:|---|
| Snapshot on-hand at 23:00 | 2,926 | WMS snapshot |
| Put-away posted 23:00–23:30 | 254 | WMS transaction log |
| Physical count at 06:00 | 3,180 | Cycle count |
| In-transit | 0 | TMS |
| Backorders | 0 | ERP |
| Mart inventory position | 2,926 | Mart |
| Understatement | 254 | Arithmetic |
| Understatement as share of count | 8.0% | 254 / 3,180 |

Over 30 days the average understatement was 7.9%, with a range from 3.1% to 13.4%. The range
matters more than the average: the error is largest on days with late receiving, which are also
the days when the inventory position matters most.

Two fixes were possible. The first was to move the snapshot to 23:45, after posting completes. The
second was to keep the snapshot and add a `pending_putaway_units` field from the transaction log,
so that the mart could add it explicitly. The second was chosen because it preserves the snapshot's
meaning and makes the assumption visible in the schema.

After replaying 90 days with the added field, the average gap fell from 7.9% to 0.4%, and 11 of 14
daily stock-out alerts disappeared. The remaining three were genuine: two items with supplier
delays and one with a recording error at the DC. The alerts had been 79% false, and the planners
had learned to ignore them. That is the second-order cost of a boundary defect: it destroys the
credibility of the alerting system, not only the accuracy of the number.

### 6.4 Tracing the case-pack error

Item CTH-4471 changed case pack from 24 to 12 on 1 July 2026. The ERP item master holds the
current pack only; there is no effective-dated history in the source. The original staging model
read the current pack and multiplied every historical case quantity by it.

| Period | Order quantity (cases) | Pack used | Eaches computed | True eaches | Error |
|---|---:|---:|---:|---:|---:|
| Jan–Jun 2026 | 1,200 per week | 12 | 14,400 per week | 28,800 per week | -50% |
| Jul–Dec 2026 | 1,200 per week | 12 | 14,400 per week | 14,400 per week | 0% |

The error was not visible in the mart, because the mart's forecast model learned a level and the
level was internally consistent. It became visible when a planner compared a January purchase
order in eaches to the ERP's archive and found a factor of two. Eighteen weeks of history had been
halved, and the safety stock computed from that history was set too low for the true demand.

The fix was structural: a pack-history table in conformed, with effective dates, sourced from the
ERP's change log and, where the log was incomplete, from purchase-order history. The staging model
no longer reads the current pack. The conformed model performs an as-of join on the order date.
The test is a reconciliation of eaches to cases by month, which fails loudly if a pack version is
missing.

The cost was mostly planner time: 14 planners spent about three hours a week for 18 weeks
reconciling orders, which is 756 hours, or roughly EUR 41,600 at EUR 55 per hour. The inventory
effect was smaller and harder to attribute, so the planner time is the defensible number for the
business case.

### 6.5 The boundary contract that would have caught both

The rebuilt pipeline wrote one contract per boundary. The WMS boundary contract is reproduced
below in the form the team used.

| Clause | Content |
|---|---|
| Producer | WMS product owner, logistics technology |
| Consumer | Ingestion engineering, then data engineering |
| Scope | 14 DCs; item x location x day snapshot; transaction log for put-away and picks |
| Schema | 11 columns, versioned; changes require 10 working days notice |
| Arrival | Snapshot complete by 23:45; transaction log by 00:15 |
| Completeness | 14 of 14 DCs; row count within 0.5% of the prior four-week median |
| Quality | No negative on-hand; every transaction references a valid location |
| Freshness measured | p95 arrival time over 30 days, reported weekly |
| Breach action | Load what arrived; flag the missing DC; notify the source owner; planning run proceeds with a defect flag on the DC |
| Review | Quarterly, or on any schema or scheduling change |

The contract does not say "data must be accurate." It says what will arrive, by when, with what
completeness, and what happens if it does not. That is the difference between a contract and a
wish.

### 6.6 Planning mart versus analytics warehouse

Calder also runs a warehouse for finance and category reporting. The two systems share conformed
inputs and nothing else, and the reason is the clock.

| Property | Planning mart | Analytics warehouse |
|---|---|---|
| Primary use | Daily replenishment and 12-week supply plan | Reporting, cohort analysis, model training |
| Grain | Item x DC x day, plus plan version and scenario | Flexible; often item x customer x week |
| Write pattern | Read-write; planners save scenarios and overrides | Read-mostly; analysts query, scheduled jobs load |
| History | Current plus 12-week horizon, plus 24 months for audit | 5–7 years |
| Refresh | Against the 06:00 cut-off; p95 by 04:30 | Nightly; complete by 07:30 |
| State | Stateful: open orders, firm commitments, frozen horizon | Stateless snapshots of history |
| Tests | Keys, units, grain, reconciliations to WMS and ERP | Aggregation checks, finance reconciliations |
| Failure cost | Wrong order today | Wrong conclusion this quarter |
| Owner | Demand planning with analytics engineering | Finance and commercial analytics |

The nightly warehouse finishes at 07:30, ninety minutes after the planning cut-off. It therefore
cannot be the planning mart's source, however good its data is. This is not a technology
limitation. It is a scheduling decision made for reporting convenience, and it is the most common
reason a technically sound warehouse fails to support planning.

The planning mart's refresh budget, measured in minutes:

| Step | Duration | Depends on |
|---|---:|---|
| Collection from ERP, WMS, TMS, EDI | 95 | Source availability; collection is serialised |
| Staging transforms | 25 | Collection |
| Conformed transforms | 40 | Staging |
| Mart build | 20 | Conformed |
| Plan engine | 35 | Mart |
| Publish and notify | 5 | Plan engine |
| **Critical path** | **220** | |
| **Available window, 02:15 to 06:00** | **225** | |
| **Slack** | **5** | 2.2% of the window |

Five minutes of slack on a 225-minute window is not a margin; it is a coincidence. One late EDI
file breaches the cut-off. The team's options, in order of cost, are to move the collection
earlier, to incrementalise the conformed transforms, cutting that step from 40 to 15 minutes
(which reduces the critical path to 195 minutes and raises slack to 30 minutes, 13.3%), or to add
net-change collection during the day. The arithmetic of
each option belongs in the design review, not in an incident review at 06:05.

## 7. Code, query, or calculation

One query, because the concept is simple and the query makes it testable. It counts orphan rows at
each boundary: rows in a layer with no parent in the layer below. A non-zero count at any boundary
means the chain of evidence is broken, and the number in the mart cannot be traced to a record
that arrived.

```sql
-- Boundary audit: every row must have a parent in the layer below.
select 'mart_without_conformed' as check_name, count(*) as n
from mart.inventory_position_daily m
left join conformed.item_location_day c
  on  m.item_key = c.item_key
  and m.location_key = c.location_key
  and m.snapshot_date = c.snapshot_date
where c.item_key is null
union all
select 'conformed_without_staging', count(*)
from conformed.item_location_day c
left join staging.wms_inventory_snapshot s
  on c.source_record_id = s.source_record_id
where s.source_record_id is null
union all
select 'staging_without_raw', count(*)
from staging.wms_inventory_snapshot s
left join raw.wms_inventory_snapshot r
  on  s.source_record_id = r.source_record_id
  and s.ingest_date = r.ingest_date
where r.source_record_id is null;
```

Expected output on a healthy pipeline is three rows, each with `n = 0`. On the Calder pipeline
before the rebuild the output was `mart_without_conformed = 0`, `conformed_without_staging =
18,204`, and `staging_without_raw = 0`: the conformed layer had been rebuilt from a source extract
whose rows were not retained in raw, so 18,204 rows had no evidence behind them.

Three checks before trusting the result.

- **Check the join key, not just the count.** A join on a nullable key returns no match for every
  null, producing false orphans. Confirm that the key is non-null and unique on the parent side.
- **Check the date predicate.** A snapshot date join is correct for a daily snapshot and wrong for
  a transaction table, where the key must include the event timestamp. A wrong predicate produces
  orphans that are an artefact of the query.
- **Run it after a replay, not only after a load.** A replay that deletes and rebuilds a partition
  can create orphans that a normal load never would. The audit belongs in the post-replay
  checklist, not only in the daily job.

## 8. Data, standards, and regulatory context

Layering is mostly convention. Three areas are not: contracts, temporal representation, and
retention. Each has a standard that constrains what a layer may promise.

### 8.1 Contracts and table formats

| Standard or specification | What it defines | Relevance to layering | Grade |
|---|---|---|---|
| Open Data Contract Standard (ODCS) v3.2.0, Bitol / LF AI and Data | Contract sections for schema, quality, service levels, ownership, and support | Gives the boundary contract a machine-readable form | C |
| Data contracts as practised by Jones (2023) and Sanderson (2022) | An agreed interface between data producers and consumers, with change management | Establishes the producer's obligation at the ingestion boundary | C |
| Delta Lake transaction log | Atomicity, consistency, isolation, and durability (ACID) table semantics over object storage, with time travel | Makes an append-only raw layer practical at scale | A for the paper, D for product documentation |
| Apache Iceberg table specification | Open table format with snapshots, schema evolution, and partition evolution | Portable alternative for the same role | C |
| Apache Hudi | Table format with upsert and incremental pull support | Alternative for change-heavy sources | C |

The table formats matter to this topic for one reason: they make snapshot isolation and time
travel available at the storage layer. That does not create a raw layer, but it removes the main
technical excuse for overwriting one. A team that has time travel and still overwrites has a
process problem.

### 8.2 Temporal and provenance standards

| Specification | What it defines | Relevance | Grade |
|---|---|---|---|
| Structured Query Language (SQL):2011 system-versioned tables | Standard syntax for transaction-time tables with system-maintained periods | The portable way to keep history in a relational warehouse | B |
| World Wide Web Consortium (W3C) PROV-DM, the PROV Data Model (Recommendation, 2013) | Provenance as entities, activities, and agents, with derivation relations | Makes the raw-to-mart chain auditable as a graph | B |
| W3C PROV-O | An ontology for the same model | Machine-readable provenance for a catalogue | B |
| International Organization for Standardization (ISO) 8000 | Frameworks for data quality management and exchange | Vocabulary for the quality clauses in a contract | B |

PROV answers "where did this value come from and through what process." It does not answer "is it
correct," and it does not replace a contract. Its practical value is that it forces the pipeline
to record the agent and the activity, not only the input and output tables. A lineage graph
without agents tells you which job ran; it does not tell you who was responsible.

### 8.3 Retention, erasure, and the immutability conflict

Immutability and privacy law point in opposite directions. A raw layer that never deletes is
excellent for replay and problematic when a data subject requests erasure, or when a retention
limit expires.

| Requirement | Jurisdiction | Effect on the raw layer | Grade |
|---|---|---|---|
| Right to erasure, General Data Protection Regulation (GDPR) Article 17 | European Union | Personal data must be erased on a valid request, subject to legal exceptions | B |
| Records retention for financial reporting | Varies; typically 5–10 years | Transaction records must be kept, which supports append-only raw | B |
| Product traceability retention | Sector-specific, for example medicinal products | Lot and event records must be retained for defined periods | B |
| Data minimisation, GDPR Article 5 | European Union | Personal data collected must be limited to what is necessary | B |

The workable pattern is to separate the immutable event record from the personal data it
references. The raw layer keeps the event and a pseudonymous key; a separate restricted table
holds the identifiable data, with its own retention and deletion process. A deletion writes a
tombstone and removes the identifiable record; the event remains, and replay still works because
the event never depended on the name.

A raw layer that stores names, addresses, and identifiers in the same row as the order event
cannot satisfy both requirements. The team will either refuse erasure, in breach of the law, or
delete the row, in breach of replay. Neither is an architecture; both are the absence of one.

> **Note:** This is a design pattern, not legal advice. Retention periods and erasure exceptions
> vary by jurisdiction and sector, and the lawful basis for processing determines what may be
> kept. Confirm the design with counsel before relying on it.

### 8.4 What regulation does not require

Regulation constrains what data must exist, be retained, and be traceable. It does not require a
medallion architecture, a specific number of layers, a data vault, or a particular table format.
An auditor asks whether a reported number can be traced to supporting records and whether the
records were retained. A three-layer pipeline with complete lineage satisfies that test. A
seven-layer pipeline with no lineage does not.

The practical consequence: justify architecture by decision quality, cost, and operability, not by
compliance. Compliance sets a floor of provenance, retention, access control, and reconstructability;
it does not set the shape of the pipeline.

### 8.5 What is convention rather than requirement

- Bronze, silver, and gold are vendor vocabulary, not a standard. The Databricks page that
  describes them calls the pattern "a recommended best practice but not a requirement."
- Three layers is a convention. Two well-contracted layers can be better than five poorly
  contracted ones.
- "Single source of truth" is a slogan. The defensible form is one source of record per field per
  decision context, as Module 02 argued.
- A data vault is a modelling choice, not a maturity level. It solves integration and audit
  problems at the cost of more tables and more loading logic.
- A one big table mart is a legitimate consumption pattern. It is not an architecture.
- Daily refresh is a habit inherited from batch reporting. Planning cadences and source
  capabilities, not reporting convenience, should set it.

## 9. Industry practice and case evidence

The evidence on layering is uneven. There are vendor documents that define the vocabulary, a small
peer-reviewed literature on the underlying storage and modelling questions, a large practitioner
literature of varying quality, and very little published measurement of whether layered
architectures improve supply chain decisions. This section separates those.

### 9.1 What the vendor documentation claims

Databricks' medallion page is the canonical statement of the pattern. It defines bronze as "raw,
unvalidated data" in original source formats, appended incrementally and retained for auditing;
silver as "validated, cleaned, and enriched" data with deduplication, schema enforcement, and
joins; and gold as "highly refined views" for analytics and reporting.

The page also makes three claims that are worth separating from the pattern itself. It calls
bronze the "single source of truth," which is true for fidelity and misleading for meaning. It says
the architecture guarantees atomicity, consistency, isolation, and durability, which is a property
of the underlying table format, not of the layer design. And it says the pattern is "a recommended
best practice but not a requirement," which is the most honest sentence on the page.

None of this is evidence that the pattern improves outcomes; it is a vendor's description. Grade D,
dated 29 October 2026.

### 9.2 What the critics say

The critique literature is recent, practitioner-written, and consistent in its complaints. The
strongest statement is Bellemare's "The End of the Bronze Age" (InfoQ, 29 January 2025), which
argues that all multi-hop architectures, medallion included, are expensive, slow, and hard to
reuse. Its five criticisms are specific, and each is worth testing against a real pipeline.

The proposed alternative is to shift cleaning and standardisation upstream, into data products
owned by the teams that understand the source, with contracts defining schemas, evolution, and
service levels.

| Criticism | What it gets right | What it overstates |
|---|---|---|
| Repeated copying is expensive | True at scale; each hop has storage and compute cost | A conformed layer often replaces dozens of mart-level joins, so the net cost can fall |
| Bronze is fragile | True when bronze is an unmanaged dump | A raw layer with a schema registry and manifest is not fragile |
| Quality cannot be restored downstream | True if the refined copy is the only copy | It is false when raw is retained and replay is tested |
| Consumers bear the burden | True of pull-based warehouses | Contracts and products change the burden, not the layer count |
| Operational reuse is poor | True for batch-only pipelines | Streaming and reverse loading are separate choices, not consequences of layering |

Grade E for the essay as evidence of practice, although its mechanism claims are worth taking
seriously. The critique is a useful stress test, not a measurement.

### 9.3 What survives the critique

Three claims survive, and they are the ones this topic builds on.

**The number of hops is a cost, not a virtue.** Every layer must earn its keep with a test, an
owner, and a decision it serves. A conformed layer that only renames columns is a hop without a
promise.

**The boundary matters more than the layer count.** Bellemare's alternative, data products with
contracts, is an argument for stronger boundaries, not for no boundaries. A two-layer pipeline
with a testable ingestion contract and a testable consumption contract is more defensible than a
five-layer pipeline with neither.

**Raw fidelity is not optional.** The critics and the vendors agree on this, from opposite
directions. The vendor calls bronze the source of truth; the critic says source fidelity cannot be
reconstructed downstream. Both imply that the raw record must be kept and must be replayable.

### 9.4 The older lineage: Inmon, Kimball, and Data Vault

The medallion pattern is a renaming of problems that predate it by three decades.

| Approach | Core structure | Integration mechanism | Where the layers sit |
|---|---|---|---|
| Inmon, top-down | Staging, then a normalised enterprise warehouse, then dependent data marts | Enterprise data model and 3NF | Staging is separate; marts are derived after the warehouse |
| Kimball, bottom-up | Staging, then dimensional marts | Conformed dimensions and facts, planned with a bus matrix | Staging is off-limits to users; the warehouse is the union of marts |
| Data Vault 2.0 | Staging, raw vault, business vault, information mart | Hubs, links, and satellites; business rules in the business vault | Raw vault is insert-only and auditable; business vault is where soft rules live |

Inmon's staging area is a clean, transient copy, not an immutable raw layer. Kimball's staging is
explicitly not for user queries. Data Vault's raw vault is the closest of the three to the raw
layer described here: insert-only, source-oriented, and auditable, with hubs for business keys,
links for relationships, and satellites for historised attributes. The business vault then plays
the role of the conformed layer, and the information mart plays the role of the mart.

The lineage matters because it shows that the hard problems are not new. Entity resolution,
effective dating, unit normalisation, and replay were solved, partially, before object storage
existed. A team that adopts medallion names without the older discipline reinvents the problems
and skips the solutions.

### 9.5 dbt conventions as a de facto contract

dbt Labs' project structure guidance is the most widely copied statement of layer boundaries in
modern analytics engineering. Its three model layers map onto the four-layer model as follows.

| dbt layer | Materialisation | Rule | Four-layer equivalent |
|---|---|---|---|
| Staging | View | One model per source table; no joins or aggregations; only layer that selects from sources | Staging |
| Intermediate | Ephemeral or view | Purpose-built transformation; not exposed to end users; verb-named | Conformed, in progress |
| Marts | Table or incremental | Business-defined entity; wide and denormalised; tested | Mart |

The convention is not a standard, and it is not evidence that it improves outcomes. It is evidence
of what a large vendor's users have found workable, which is a weaker and still useful claim.
Grade D.

The instructive part is the prohibition: only staging models select from sources. That single rule
prevents the most common architecture failure, a mart that reaches around the pipeline to a source
table. A team can adopt the rule without adopting dbt.

### 9.6 Planning data stores: evidence from planning applications

Supply chain planning applications have had their own layered stores for decades, and their
documentation is unusually explicit about refresh behaviour.

Oracle's supply chain planning documentation describes a collection process that pulls source data
into staging tables, then loads a planning data repository. It distinguishes **targeted
collection**, a complete refresh of selected entities usually run overnight, from **net change
collection**, which picks up only records changed since the last successful run and is used one to
a few times per day. It states that only one collection can run at a time, that certain entities
can only be collected in targeted mode, and that continuous collection can switch between modes
using a threshold, with intervals as short as 20 minutes.

The same documentation describes the storage layers as an applications data store, an operational
data store, and a planning data store, with the planning store segregated by plan identifier. This
is a planning-specific architecture: the store is partitioned by the decision it serves, not by
the history it accumulates.

Three lessons transfer.

1. **Planning data has a cadence, and the cadence is a design parameter.** Overnight plus net
   change is a deliberate compromise between completeness and freshness, not a default.
2. **Collection is serialised by the source.** One collection at a time is a constraint that
   planning pipelines must design around, and it is why the critical path in section 6.6 is
   dominated by collection.
3. **Plan identity is part of the grain.** A planning store that does not separate plan versions
   cannot reproduce a decision. The analytics warehouse seldom needs this; the planning mart
   almost always does.

Grade D: vendor product documentation, dated 2024–2026 across the versions consulted, describing
one vendor's implementation rather than a general law.

### 9.7 Data contracts: from practice to literature

The data contract idea emerged from industry practice and is only beginning to appear in academic
literature. Andrew Jones dates the pattern to 2021 and defines a contract as an agreed interface
between the generators of data and its consumers. Chad Sanderson's 2022 essay popularised the
framing of contracts as interface-like agreements between service owners and data consumers, with a
change-management workflow and enforcement in the producer's deployment process.

A 2025 systematic gray literature review (Wasser et al., BMSD 2025) notes that academic literature
on data contracts is scarce, that the concept emerged from industry, and that definitions
converge on formal agreements that define and enforce terms for exchanging data between producers
and consumers. That is a fair summary of the state of the evidence: a coherent practice with weak
empirical support.

For this topic, the contract is the mechanism that makes a boundary real. Whether it is implemented
as a configuration file, a Protobuf schema, a dbt model contract, or a service-level agreement matters less
than three properties: it is versioned, it is tested automatically, and it has a named owner on both
sides.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Layer names without contracts | A `gold` schema with no tests | Naming is mistaken for guarantee | Consumers trust untested tables | Publish the layer charter with tests |
| Mart reads a source directly | Two paths to the same number | Bypass of staging and conformed | Numbers diverge silently | Deny source access to mart roles |
| Raw overwritten on correction | A corrected file replaces yesterday's | The loader treats raw as a staging area | Replay and audit both impossible | Append with a new batch identifier |
| Staging joins across systems | Entity logic inside `stg_` models | Integration without integration tests | Duplicate keys, unattributable errors | Move joins to conformed |
| Unit or pack drift | Case and each totals disagree by a round factor | Current attribute applied to history | Demand off by the pack ratio | Effective-dated conversion table |
| Snapshot and transaction mixed | Inventory summed across days | Additivity not declared | Inventory multiplied by day count | Separate fact tables, declared additivity |
| Non-deterministic dedupe | Row counts change on rerun | Ties broken arbitrarily | Backtests irreproducible | Stable tie-breaker in the ranking |
| Backfill not idempotent | Duplicate rows after reprocessing | Append without merge or partition replace | Counts inflate after every fix | Merge on natural key or replace partition |
| Planning mart fed by nightly warehouse | Plan starts before the warehouse finishes | Reporting schedule sets planning freshness | Orders placed on stale positions | Separate planning feed and cut-off |
| Plan versions overwritten | Cannot reproduce last week's plan | The mart holds only the current plan | Audit and learning both fail | Plan run identifier in the grain |
| Bronze as unmanaged dump | No schema registry, no manifest | Ingestion accepts anything, records nothing | Fragile to source change | Schema version and arrival manifest |
| OBT as the only table | No raw or conformed upstream | Joins baked in, inputs discarded | No replay, no provenance | Demote OBT to a published mart |
| Freshness declared, not measured | Service-level agreement (SLA) says 04:30; nobody knows p95 | Declaration treated as performance | Breaches discovered by users | Measure arrival and publication p95 |
| Ownerless boundary | "The platform team owns it" | Diffuse accountability | No one fixes a breach | Named producer and consumer roles |
| Erasure by deleting raw | Privacy request removes order rows | Immutability and privacy conflated | Illegal retention or broken replay | Separate event and identifiable data |
| Layer proliferation | Platinum, diamond, and vault folders | Quality conflated with purpose | Cost without added tests | Cap layers at the number of promises |

Three deserve a note.

**Layer names without contracts.** This is the parent failure: renaming a schema is cheap and
writing a contract is not. The test is simple: ask what the layer may not do. If the answer is
"nothing in particular," the layer is a folder.

**Snapshot and transaction mixed.** Inventory is semi-additive: additive across locations,
non-additive across time. A mart that sums daily snapshots over a month reports inventory as if
the company held thirty days of stock at once. The error is large, plausible, and usually found by
finance.

**Planning mart fed by the nightly warehouse.** The failure is organisational, not technical. The
warehouse completes when reporting needs it to complete. Planning needs it three hours earlier.
Unless the planning cut-off is an explicit requirement on the warehouse schedule, planning will
lose, and the planners will maintain a spreadsheet that does not.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The decision has a stable grain and cut-off | Every layer's promise derives from it | Decision redefined mid-quarter | Ask what action changes if the number moves |
| Source delivery is observable | Completeness needs a manifest or count | Feeds with no file-level acknowledgement | Compare received rows to source counts |
| Raw storage is affordable for the retention period | Replay depends on retained inputs | Volume growth, privacy constraints | Cost per TB-month against decision value |
| Transformation logic is deterministic | Replay must reproduce results | Wall-clock filters, random keys, unpinned reference data | Freeze clock and replay a past partition |
| Business keys are stable enough to resolve | Conformed keys depend on them | Rebranding, mergers, pack changes | Count merges and reversals per quarter |
| Reference data is versioned | Unit and calendar changes must be as-of | Factors edited in place | Change a factor, replay, inspect difference |
| Contracts can be enforced | A promise without a test is advice | Producer has no incentive or mandate | Count contract tests actually failing builds |
| Consumers accept as-of semantics | Historical answers must be reproducible | Users expect "latest" everywhere | Compare as-of and current query results |
| Deletion can be separated from the event | Immutability and erasure must coexist | Identifiers embedded in event rows | Attempt an erasure and a replay |
| The team owns the layers operationally | Decay is continuous | Handover to a team without runbooks | Replay drill after a staff change |

Two limits are worth stating. The four-layer model is a design vocabulary, not a maturity scale;
a two-layer pipeline with strong contracts can be better than a five-layer pipeline with weak
ones. And the model assumes the organisation can name an owner for each boundary. In practice the
hardest part is not the technology but finding the person who will accept the promise, which is
why the contract table names roles rather than teams.

## 12. Exercises

**Exercise 1 — Draw the stack (design, 45 minutes).** Choose a pipeline you know and draw its four
layers with the contract and owner at each boundary.
Deliverable: one diagram plus a boundary table.
Acceptance: four layers with prohibitions; every boundary has a promise, a test, and an owner; one
boundary is marked as the weakest and justified.

**Exercise 2 — Write the layer charter (design, 50 minutes).** For the same pipeline, complete the
may, may-not, owner, retention, and test columns.
Deliverable: a four-row charter.
Acceptance: every prohibition cites the failure it prevents; retention is justified, not defaulted;
at least one test is a reconciliation rather than a null check.

**Exercise 3 — Write a boundary contract (design, 60 minutes).** Write the contract for the
boundary you marked weakest in Exercise 1.
Deliverable: a one-page contract with producer, consumer, scope, schema, arrival, completeness,
quality, measurement, and breach action.
Acceptance: the breach action is an action, not an escalation; freshness is a measured p95;
one clause addresses schema change notice.

**Exercise 4 — Trace a wrong number (analysis, 50 minutes).** Take a number that two reports
disagree about, or construct one from the Calder case, and walk backwards layer by layer.
Deliverable: a trace table with value at each layer, the test that should have caught it, and the
boundary where the defect entered.
Acceptance: the cause is located at a boundary, not a person; at least one alternative boundary is
ruled out with evidence; the fix names an owner.

**Exercise 5 — Replay determinism audit (analysis, 60 minutes).** Inspect a pipeline for wall-clock
filters, unpinned reference data, arbitrary deduplication, and non-idempotent writes. Quantify one.
Deliverable: a findings table with the defect, the affected rows, and the corrected logic.
Acceptance: at least one defect is reproduced by two runs; the row impact is counted, not estimated;
the fix is testable in the build.

**Exercise 6 — Critique the medallion (critique, 45 minutes).** Read the Databricks medallion page
and one critique. Write a two-page assessment for a team about to adopt the pattern.
Deliverable: a claim-by-claim table separating what the vendor asserts, what the critic asserts, and
what is supported.
Acceptance: vendor claims are graded and dated; at least one criticism is accepted and one is
rejected with a mechanism; the recommendation names the contracts, not the folder names.

**Exercise 7 — OBT decision memo (design, 40 minutes).** A team proposes replacing the pipeline
with one wide table. Write the memo.
Deliverable: the conditions under which the proposal is acceptable, the conditions under which it is
not, and the migration path.
Acceptance: distinguishes OBT as mart from OBT as architecture; names what is lost (replay,
provenance, entity resolution); states a reversible first step.

**Exercise 8 — Refresh budget (calculation, 55 minutes).** Using the Calder timings, compute the
critical path, slack, and the effect of incremental conformed transforms and net-change collection.
Deliverable: a table of scenarios with minutes and slack as a percentage of the window.
Acceptance: arithmetic shown; the bottleneck identified; a recommendation with its operational
condition.

**Exercise 9 — Freshness measurement plan (design, 45 minutes).** Design the measurement that
would tell you whether a boundary's freshness promise is being met.
Deliverable: the event definition, the timestamp pairs, the percentile, the window, and the
reporting cadence.
Acceptance: arrival time measured at the platform, not the job scheduler; p95 rather than mean; a
breach threshold tied to the decision cut-off.

## 13. Self-check answers

**Exercise 8 arithmetic.** The available window is 02:15 to 06:00, which is 3 hours 45 minutes, or
225 minutes.

Sequential critical path: 95 collection + 25 staging + 40 conformed + 20 mart + 35 plan engine + 5
publish = 220 minutes. Slack = 225 - 220 = 5 minutes, which is 5 / 225 = 2.2% of the window.

With conformed transforms incrementalised from 40 to 15 minutes, the path is 95 + 25 + 15 + 20 +
35 + 5 = 195 minutes. Slack = 225 - 195 = 30 minutes, or 13.3%. The saving comes from processing
only changed partitions, not from running dependent steps at the same time.

With net-change collection at 35 minutes on 80% of days and targeted collection at 95 minutes on
20%, and the existing sequential transforms, the net-change path is 35 + 25 + 40 + 20 + 35 + 5 =
160 minutes, slack 65 minutes (28.9%). Combining net change with incremental conformed gives 35 +
25 + 15 + 20 + 35 + 5 = 135 minutes, slack 90 minutes (40.0%). The targeted path remains 195 minutes
with incremental conformed, slack 30 minutes (13.3%), so the p95 is governed by the targeted days. A plan
that quotes the average has hidden the days that breach.

If streaming reduced collection to 15 minutes on all days, with incremental conformed the path would
be 15 + 25 + 15 + 20 + 35 + 5 = 115 minutes, slack 110 minutes (48.9%). The cost of streaming must be
compared with the value of the 105 minutes it releases against the sequential baseline, not with
the average collection time.

The design conclusion: incrementalising conformed buys 25 minutes; net-change collection buys 60;
the two together buy 85; streaming plus incremental conformed buys 105. Only the targeted-day path sets the p95.

**Exercise 5 arithmetic.** A pipeline processes 12.1 million conformed rows per month. Deduplication
orders by `source_updated_at` only. Inspection finds 4,312 groups with tied timestamps, which is
4,312 / 12,100,000 = 0.036% of rows. Two runs select different survivors in 1,907 of those groups,
44.2% of the tied groups. Downstream, the conformed layer had 18,204 rows with no staging parent,
which is 0.150% of monthly rows; the orphan count is larger than the tie count because a flipped
survivor can orphan every child row that referenced the previous key. After adding ingestion
timestamp and file name as tie-breakers, two consecutive runs produced identical row counts and
zero orphans. The percentage is small and the audit consequence is not: a 0.036% instability
invalidated a quarterly backtest.

**Rubric for Exercise 1.** A strong diagram names the prohibition at each layer and places the
planning mart's cut-off on the consumption contract. It marks the raw-to-staging boundary as the
weakest when the source has no manifest, and says so. A weak diagram draws four boxes labelled
bronze, silver, gold, and "reporting," with no owners and no tests.

**Rubric for Exercise 2.** A strong charter pairs every prohibition with the failure it prevents
and places a reconciliation test at the conformed boundary. A weak charter restates the layer names
and copies retention defaults.

**Rubric for Exercise 3.** A strong contract states a measurable arrival time, a completeness
definition, a schema-change notice period, and a breach action that the producer can execute
without a meeting. It distinguishes a late file from a missing file. A weak contract lists column
names and says data will be "accurate and timely."

**Rubric for Exercise 4.** A strong trace shows the value at each layer, rules out at least one
boundary with a test result, and names the boundary where the defect entered. A weak trace stops at
the layer where the number first looks wrong.

**Rubric for Exercise 6.** A strong assessment accepts the cost-of-copying criticism, rejects the
claim that quality cannot be restored when raw is retained and replay is tested, and concludes that
the pattern is a default rather than a mandate. A weak assessment either repeats the vendor page or
repeats the critique without testing either against a specific pipeline.

**Rubric for Exercise 7.** A strong memo accepts an OBT as a mart when raw, staging, and conformed
remain upstream, and rejects it as the sole architecture because replay and entity resolution are
lost. It proposes a reversible first step: publish the OBT alongside the existing mart, reconcile
for one month, then retire the mart. A weak memo argues about query speed only.

**Rubric for Exercise 9.** A strong plan measures arrival at the platform boundary and publication
at the consumer boundary, uses p95 over a 30-day window, and ties the threshold to the decision
cut-off. It reports the count of days that breached, not only the percentage. A weak plan measures
job success and calls it freshness.

## 14. Interview questions

**1. "Why have layers at all?"** A strong answer gives change isolation, replay, test placement, and
accountability at boundaries, and notes that the number of layers is negotiable while the promises
are not. A weak answer says layers organise data.

**2. "What is the difference between raw, bronze, and staging?"** A strong answer separates the
record of arrival (raw or bronze) from the source-aligned cleaning layer (staging), and explains
that a landing zone without retention is not raw. A weak answer treats the three names as synonyms.

**3. "Is bronze the source of truth?"** A strong answer says it is the immutable record of what was
received, which is evidence of fidelity, not of truth; if the source restates, raw holds both
versions. A weak answer repeats the vendor phrase.

**4. "When is the medallion pattern the wrong choice?"** A strong answer says when hops add cost
without adding a test or an owner, when the source is already a governed product with a contract,
or when operational latency dominates. It still keeps raw. A weak answer says never, or says the
pattern is obsolete.

**5. "What belongs in conformed rather than in a mart?"** A strong answer puts entity resolution,
surrogate keys, unit conversion, effective dating, and business rules applied once. It keeps
aggregation and presentation in the mart. A weak answer says "cleaned data."

**6. "How do you make a backfill safe?"** A strong answer names immutable inputs, a pinned code
version, deterministic logic, idempotent writes, and a reconciliation against the original run. A
weak answer says rerun the job.

**7. "How is a planning mart different from an analytics warehouse?"** A strong answer contrasts the
decision clock, current-plus-horizon history, plan versions, read-write scenarios, and freshness
against long history, read-mostly access, and flexible grain. A weak answer says one is smaller.

**8. "What is in a data contract?"** A strong answer lists producer, consumer, schema, arrival,
completeness, quality, measurement, change process, and breach action, and says the schema is one
clause. A weak answer describes a table definition.

**9. "How would you detect that a layer boundary has been breached?"** A strong answer names an
automated test at the boundary, such as an orphan count, a unit reconciliation, or a freshness p95,
and a breach action. A weak answer says monitor data quality.

**10. "When is a one big table acceptable?"** A strong answer accepts it as a published mart on top
of a layered pipeline and rejects it as the only layer, naming replay and entity resolution as what
is lost. A weak answer says wide tables are always faster and therefore better.

**11. "What is the most common cause of a wrong number in a warehouse?"** A strong answer points to
a boundary assumption that was never written down: a cut-off, a pack version, a join key, or a
population filter. A weak answer blames bad source data.

## 15. Key terms

- **Layer** — a stage with a grain, prohibitions, owner, and retention rule.
- **Raw** — the immutable, append-only record of what arrived, with ingestion metadata.
- **Bronze** — the vendor name for the raw role.
- **Staging** — source-aligned typing, deduplication, and unit normalisation.
- **Conformed** — integrated entities, canonical keys, effective dates, and business rules applied once.
- **Silver** — the vendor name covering both staging and integration.
- **Mart** — a table at a decision grain with declared additivity and freshness.
- **Gold** — the vendor name for consumption-ready tables.
- **Planning mart** — a mart serving a decision clock, with plan versions and scenarios.
- **Analytics warehouse** — long-history, read-mostly storage for reporting and learning.
- **ODS** — operational data store, recent volatile data close to source.
- **PDS** — planning data store, partitioned by plan and scenario.
- **Medallion** — the bronze, silver, gold naming convention.
- **One big table** — a wide denormalised table; a mart pattern, not an architecture.
- **Conformed dimension** — a dimension with consistent keys and definitions across marts.
- **Grain** — the entities and time span one row represents.
- **Additivity** — whether a measure can be summed across a dimension.
- **SCD type 2** — a new dimension row per attribute change, with effective dates.
- **Surrogate key** — a system-generated key stable across source changes.
- **Data Vault** — hubs, links, and satellites for auditable integration.
- **Raw vault** — insert-only, source-oriented Data Vault layer.
- **Business vault** — soft rules and calculations above the raw vault.
- **Idempotent** — same result whether run once or many times.
- **Replay** — reprocessing past inputs to reproduce a past output.
- **Immutability** — existing records are never modified in place.
- **Data contract** — a versioned, tested producer-consumer agreement at a boundary.
- **Freshness** — the age of the oldest source record reflected at a stated time.
- **Lineage** — the technical graph from inputs through transformations to outputs.

## 16. Sources

**Peer-reviewed and academic**

- Armbrust, M., Ghodsi, A., Xin, R., and Zaharia, M. (2021). "Lakehouse: A New Generation of Open
  Platforms that Unify Data Warehousing and Advanced Analytics." CIDR 2021. (A)
- Armbrust, M., et al. (2020). "Delta Lake: High-Performance ACID Table Storage over Cloud Object
  Stores." *PVLDB*, 13(12), 3411–3424. (A)
- Wasser, Kumara, Monsieur, Van den Heuvel, and Tamburri (2025). "Data Contracts in Data Mesh: A
  Systematic Gray Literature Review." BMSD 2025, Springer LNBIP 559. (A-)
- Snodgrass, R. T. (1999). *Developing Time-Oriented Database Applications in SQL*. Morgan
  Kaufmann. Valid-time and transaction-time patterns. (A)

**Books and canonical frameworks**

- Inmon, W. H. (1992; 4th ed. 2005). *Building the Data Warehouse*. Wiley. Top-down normalised
  enterprise warehouse. (C)
- Kimball, R., and Ross, M. (2013). *The Data Warehouse Toolkit*, 3rd ed. Wiley. Dimensional
  modelling and conformed dimensions. (C)
- Kimball Group (1997). "A Dimensional Modeling Manifesto." Bus matrix and conformed dimensions. (C)
- Hultgren, H. (2019). *Data Vault Modeling Guide*, v3. Hubs, links, and satellites. (C)
- Jones, A. (2023). *Driving Data Quality with Data Contracts*. Packt. Contracts as agreed
  interfaces. (C)

**Standards and specifications**

- SQL:2011, system-versioned tables. Transaction-time history in standard SQL. (B)
- W3C (2013). *PROV-DM: The PROV Data Model*, Recommendation. https://www.w3.org/TR/prov-dm/ (B)
- W3C, *PROV-O: The PROV Ontology*. (B)
- Open Data Contract Standard (ODCS) v3.2.0, Bitol / LF AI and Data. (C)
- ISO 8000, data quality frameworks. (B)
- Regulation (EU) 2016/679, General Data Protection Regulation, Article 17 and Article 5. (B)

**Vendor and product documentation**

- Databricks, "What is the medallion lakehouse architecture?" Last updated 29 October 2026.
  https://docs.databricks.com/aws/en/lakehouse/medallion (D)
- Databricks, "What Is a Lakehouse?" Blog, 30 January 2020. (D)
- dbt Labs, "How we structure our dbt projects," staging, intermediate, and marts guides,
  accessed 2 November 2026. https://docs.getdbt.com/best-practices/how-we-structure/1-guide-overview (D)
- Oracle, "Data Collection Types for Supply Chain Planning," Fusion Cloud 26a. Targeted and
  net-change collection. (D)
- Oracle, "Collection Job Set," Fusion Cloud 24d. Serialised collection and scheduling. (D)
- Amazon Web Services (AWS) Big Data Blog, "Design and build a Data Vault model in Amazon Redshift,"
  accessed 2 November 2026. (D)

**Practitioner critique, press, and analysis**

- Bellemare, A. (2025). "The End of the Bronze Age: Rethinking the Medallion Architecture." InfoQ,
  29 January 2025. https://www.infoq.com/articles/rethinking-medallion-architecture/ (E)
- Modern Data 101 (2025). "Data Products: A Case Against Medallion Architecture," February 2025. (E)
- Matterbeam (2025). "Beyond the Medallion: Rethinking Data Architecture from First Principles,"
  October 2025. (E)
- Wessely, B. (2024). "The Case Against Centralized Medallion Architecture." Medium / Towards Data
  Science, December 2024. (E)
- Sanderson, C. (2022). "The Rise of Data Contracts." Data Products Substack. (E)
- Sanderson, C., and Freeman, M. *Data Contracts*. O'Reilly. (C)

**Note on grading.** Grades follow Module 28: A peer-reviewed or audited, B regulatory or official,
C framework or case, D vendor documentation, E press or essay. Gray literature reviews are A-.
Vendor pages are dated at the version consulted; figures are as the sources state them.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite topic: [`../02_supply_chain_data_landscape/10_source-selection-trust.md`](../02_supply_chain_data_landscape/10_source-selection-trust.md)
- Module 02 map: [`../02_supply_chain_data_landscape/_index.md`](../02_supply_chain_data_landscape/_index.md)
- Units and calendars: [`../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Next topic (not yet written): `02_batch-microbatch-streaming.md`
- Later topic (not yet written): `04_sql-supply-chain-patterns.md`
- Later topic (not yet written): `11_quality-quarantine-lineage.md`
- Return to the part map: [`../_index.md`](../_index.md)
