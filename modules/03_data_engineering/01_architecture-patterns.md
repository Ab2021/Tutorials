# Architecture Patterns

> Module 03 — Data Engineering and Interoperability
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 02; basic SQL
> Version 2026.3

## 1. Why this topic exists

Architecture is the set of decisions that are expensive to reverse. A team can change a SQL query in an
hour and a dashboard in a day. Changing where protected health information (PHI) is stored, which table
format the organization standardizes on, or whether imaging lives in the same platform as claims takes
quarters and, in a hospital, a committee. The architecture question is therefore not "which product is
best" but "which constraints are we accepting for the next three years."

The failures are concrete and familiar.

- A research group lands extracts on a network share, transforms them in place, and cannot reconstruct
  the original when a mapping bug is found eleven months later. The raw files were overwritten. The
  publication is retracted, not because the science was wrong, but because the provenance is gone.
- A hospital buys a warehouse for structured data, then discovers that its 40 TB of imaging cannot be
  queried there without a second platform. Two governance models, two identity systems, two audit logs,
  and a reconciliation job that nobody owns.
- A consortium centralizes records from six countries into one bucket. The transfer is legal in four
  jurisdictions and not in the other two. The project pauses for a year while a federated design is
  retrofitted.
- A team streams every bedside monitor waveform into the lake "for future use." Storage grows by
  petabytes, 97% of it never read, and the annual bill exceeds the analytics budget that justified the
  platform.
- A pipeline is rebuilt on a new engine. The logic is identical, but the engine's null ordering and
  timestamp semantics differ. Cohort counts move by 3%. Nobody can prove which number is right because
  the old environment was deleted.

Each failure is an architecture failure rather than a coding failure. The code did what it was told. The
platform made the wrong guarantee.

This topic gives you a vocabulary for those guarantees and a procedure for choosing among them. It sits
first in the module because every later topic inherits its assumptions: the SQL topic assumes a layered
model, the orchestration topic assumes idempotent partitions, the FHIR and DICOM topics assume an
immutable raw zone, and the security topic assumes a trust boundary that architecture defines.

> **Note:** This is not a vendor comparison. Products change version numbers every quarter. The
> durable content is the set of trade-offs: schema-on-write against schema-on-read, centralized against
> federated, batch against streaming, one platform against several. Those trade-offs survive rebranding.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a proposed platform as warehouse, lake, lakehouse, federated, streaming, or trusted
   research environment (TRE), and state the guarantee each pattern does and does not provide.
   Assessed in Exercise 1 and self-check item 1.
2. **Map** a clinical data source inventory onto a zone model with explicit contracts at each boundary.
   Assessed in Exercise 3.
3. **Compute** annual storage and compute cost for two architecture options from a volume table, with
   units and assumptions stated. Assessed in Exercise 4 and self-check item 4.
4. **Choose** an open table format for a stated workload and justify the choice against concurrency,
   governance, and ecosystem constraints. Assessed in Exercise 5.
5. **Design** a batch or streaming ingestion path whose latency matches the clinical decision latency.
   Assessed in Exercise 6.
6. **Critique** an architecture diagram for missing governance planes, hidden data movement, and
   single points of failure. Assessed in Exercise 7.
7. **Write** an architecture decision record (ADR) that states the decision, the alternatives, the
   consequences, and the condition under which the decision should be revisited. Assessed in Exercise 8.
8. **Distinguish** a regulatory requirement from an architectural convention, and explain why the
   medallion pattern is the latter. Assessed in self-check item 7.

## 3. The core mental model

Think of the platform as four planes, not as a stack of products. Data moves through three of them and
controls cut across all four.

```text
+-----------------------------------------------------------------------------------+
|  GOVERNANCE PLANE                                                                  |
|  identity | access policy | catalog | lineage | audit | retention | residency     |
+-----------------------------------------------------------------------------------+
        |                    |                      |                     |
+-------+--------+  +--------+---------+  +---------+--------+  +---------+---------+
| INGESTION      |  | STORAGE ZONES    |  | COMPUTE              |  | CONSUMPTION       |
| batch, stream, |  | raw / bronze     |  | SQL engine, Spark,   |  | BI, notebooks,    |
| API, file drop |->| standardized /   |->| Python, dbt, ML      |->| models, APIs,     |
| CDC, FHIR      |  | silver           |  |                      |  | regulatory output |
| DICOM, HL7 v2  |  | curated / gold   |  |                      |  |                   |
+----------------+  +------------------+  +----------------------+  +-------------------+
        |                    |                      |                     |
        +--------------------+----------------------+---------------------+
                             |
                    +--------+---------+
                    | OPERATIONS PLANE |
                    | orchestration |   |
                    | observability |   |
                    | cost | on-call   |
                    +------------------+
```

Read the diagram in two directions.

**Left to right** is the data path. A source is acquired, landed without modification, standardized,
curated, and consumed. Each arrow is a boundary, and each boundary is a contract: a schema, an owner, a
quality expectation, and a retention rule. If a boundary has no contract, it is not an architecture; it
is a folder.

**Top to bottom** is the control path. Governance is not a layer at the end. Identity, access, lineage,
and audit must be enforceable at every boundary. A platform where the raw zone is governed and the
curated zone is a CSV export on a laptop has one governed zone and one uncontrolled copy.

The operations plane is drawn at the bottom because it is usually omitted from architecture diagrams and
is the first thing to fail. Orchestration, observability, cost control, and on-call ownership are
architecture, not afterthoughts.

### 3.1 The three guarantees that matter

Every pattern in this topic is a different answer to three questions.

1. **What is the unit of change?** In a warehouse, a row or a partition. In a lake, a file. In a
   lakehouse, a snapshot of a table. In a streaming system, an event. The unit of change determines
   whether you can roll back, replay, and audit.
2. **When is the schema enforced?** Schema-on-write enforces at load; schema-on-read enforces at query.
   Clinical feeds change without notice, so the choice determines whether a new field breaks ingestion
   or breaks a report three weeks later.
3. **Where does computation happen relative to the data?** Centralized compute moves data to a cluster.
   Federated compute moves the query to the data. The choice is driven by privacy, residency, and
   egress cost, not by query speed.

A useful discipline is to write those three answers on one page before drawing any boxes. If two
stakeholders give different answers, the architecture debate is really a requirements debate.

### 3.2 Why the medallion pattern is a convention

The bronze/silver/gold naming comes from Databricks documentation, which describes bronze as raw,
silver as validated and cleaned, and gold as enriched for business use. It is a useful default because
it forces separation between landing, standardization, and consumption. It is not a standard, not a
regulation, and not a required shape. A team that has one source, one consumer, and a one-week
deadline may reasonably collapse silver and gold into a single model. A team with a regulated product
may need four zones, with a validated-but-not-curated layer between silver and gold.

The pattern's value is the contracts it encourages, not the names. Rename bronze to `landing`, silver to
`conformed`, and gold to `marts`, and the architecture is the same. Keep the names and skip the
contracts, and the architecture is theater.

### 3.3 Batch and streaming are one axis, not two religions

```text
Event time:      t0 ---- t1 ---- t2 ---- t3 ---- t4 ---- t5
                  |       |       |       |       |
Arrival:         t1      t2      t3      t5      t6
                  \       \       \       \       \
Batch run:        [----------- 02:00 -----------]
Streaming:        .  .  .  .  .  .  .  .  .  .  .  .

Late event at t4 arrives at t6. Batch sees it in the next run.
Streaming must decide whether to update a window already emitted.
```

The axis is latency, not technology. A batch job that runs every five minutes is a streaming system with
coarse windows. A streaming job that reprocesses the full history nightly is a batch system with a
complicated scheduler. What matters is whether the pipeline can answer the clinical question in time
and whether it can correctly handle an event that arrives after its window closed.

> **Note:** The clinical rule is that pipeline latency must be less than or equal to decision latency.
> A sepsis alert evaluated every twelve hours is a different product from one evaluated every fifteen
> minutes, even if both run the same model. Architecture encodes that product decision.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Data warehouse | Analytics store with schema-on-write, usually columnar, optimized for SQL over structured data | Called "the database" when it is the transactional system | — |
| Data lake | Store of raw files with schema-on-read, accepting any modality | Treated as a garbage dump with no catalog | — |
| Lakehouse | Lake storage plus an open table format providing transactions, schema, and versioning | Assumed to be a single product | — |
| Medallion architecture | Bronze, silver, gold layering of increasing data quality | Treated as a standard or a compliance requirement | — |
| Zone | A logical storage area with an owner, contract, and retention rule | Used interchangeably with "schema" | — |
| Open table format | A specification for table metadata and file layout on object storage, readable by multiple engines | Confused with a file format such as Parquet | Delta Lake, Apache Iceberg, Apache Hudi |
| ACID | Atomicity, consistency, isolation, durability: transaction guarantees | Assumed to imply data quality | — |
| Schema-on-write | Structure enforced when data are loaded | Assumed to prevent all bad data | — |
| Schema-on-read | Structure applied when data are queried | Assumed to be schema-free | — |
| ETL | Extract, transform, load: transformation before landing | Used for any pipeline | — |
| ELT | Extract, load, transform: transformation after landing | Assumed to be always cheaper | — |
| OLTP | Online transaction processing, optimized for many small writes | Confused with analytics workloads | — |
| OLAP | Online analytical processing, optimized for large scans and aggregation | — | — |
| CDC | Change data capture: publishing row-level changes from a source system | Confused with a nightly full extract | — |
| Event time | When the clinical event occurred | Confused with load time | ISO 8601 with offset |
| Processing time | When the platform processed the event | Used as a substitute for event time in features | ISO 8601 with offset |
| Watermark | A threshold declaring how late an event may arrive before a window closes | Treated as a guarantee of completeness | Duration |
| Idempotency | Re-running an operation produces the same state | Confused with "the job succeeds on retry" | — |
| Data contract | Agreed schema, semantics, quality thresholds, and ownership between producer and consumer | Treated as a document rather than an executable check | — |
| Data product | A dataset with an owner, interface, quality guarantees, and consumers | Any table with a name | — |
| Data mesh | Organizational pattern delegating ownership of data products to domains | Used as a synonym for a technology | — |
| Federated analytics | Running queries at each site and combining results rather than centralizing records | Confused with a shared dashboard | — |
| TRE | Trusted research environment: controlled workspace where approved analysts use sensitive data | Confused with a virtual private network | — |
| Object storage | Durable, inexpensive storage addressed by key, with eventual consistency semantics | Assumed to be a file system | Amazon S3, Azure Blob, Google Cloud Storage |
| Compute-storage separation | Independent scaling of processing and durable storage | Assumed to reduce cost automatically | — |
| Partition | A subset of a table, usually by date or key, that can be written and replaced independently | Used loosely for any grouping | — |
| Snapshot | An immutable view of a table at a point in time | Confused with a backup | — |
| Catalog | Registry of datasets, schemas, owners, and locations | Confused with a data dictionary | — |
| Lineage | Graph of how datasets derive from other datasets and jobs | Confused with a list of tables | — |
| Data residency | Legal requirement that data remain in a jurisdiction | Confused with latency | Country or region |
| Egress | Data leaving a network or cloud region, often billed | Ignored in cost models | GB or TB |
| Hot/warm/cold tier | Storage classes with different retrieval cost and latency | Applied without a retrieval model | — |
| Latency | Time from event to availability for a decision | Confused with runtime of a job | Seconds, minutes, hours |
| Throughput | Volume processed per unit time | Confused with latency | GB/s, events/s |
| Blast radius | Scope of damage when a component fails | Used only for security incidents | — |
| ADR | Architecture decision record: a short document capturing a decision and its consequences | Treated as a design document | — |

Two definitions deserve expansion because they are the source of most disagreement.

**Lakehouse.** The term describes an architecture, not a product: object storage holds the data, an open
table format supplies transactions and metadata, and one or more compute engines read it. Delta Lake and
Apache Iceberg are the two formats a biomedical team is most likely to encounter. The guarantee you
actually get depends on the format version and the engine's support for it, not on the label.

**Data contract.** In dbt, a model contract is an executable declaration of column names and data
types, enforced at build time. In an architecture, a contract is broader: it includes semantic meaning,
allowed nulls, accepted units, freshness, and an owner. A dbt contract that enforces types but not units
will happily load milligrams into a column documented as grams. Section 8 returns to this gap.

## 5. The mechanics, step by step

The procedure below produces an architecture decision, not a diagram. It is written for a team that has
a source inventory and a budget. Each step states its input, output, and the reason it cannot be skipped.

### Step 1 — Write the decision statement

Input: a request such as "we need a platform for research and quality reporting."
Output: one sentence naming the decision, the deadline, and the owner.

Example: "By 30 June 2027, we will consolidate structured EHR, claims, and imaging metadata into a
single governed platform for research cohorts and quality dashboards, owned by the data platform team."

Reason: without a decision statement, every subsequent discussion becomes a preference. The statement
also defines what is out of scope. Imaging pixels may be in scope for linkage but out of scope for
storage consolidation, which changes the cost model by an order of magnitude.

### Step 2 — Inventory sources by modality, volume, and change rate

Input: interface inventory, storage metrics, vendor contracts.
Output: a table with source, modality, annual volume, rows or objects per year, update pattern, and
clinical owner.

Do not accept "about a terabyte" as an answer. Ask for the size of the last full extract and the number
of files. A source that emits 40 million small JSON files per year has a metadata problem that a source
emitting four Parquet files does not. Record the change rate separately: a laboratory feed that sends
every result twice has twice the nominal volume and a deduplication requirement.

### Step 3 — Classify each source by decision latency

Input: the source inventory and the clinical or business decisions it supports.
Output: each source labeled real-time (seconds), near-real-time (minutes), operational (hours), or
analytical (days).

The label must come from the decision, not from the source's technical capability. A monitor emits data
at 1 Hz; the decision "alert on arrhythmia" may need seconds, while the decision "compute length of
stay" needs days. The same source can feed two paths with different latency. Architecture should
separate them rather than forcing the fastest requirement onto every consumer.

### Step 4 — Identify the regulated and contractual constraints

Input: data use agreements (DUAs), business associate agreements (BAAs), institutional review board
(IRB) determinations, jurisdictional privacy law, and payer contracts.
Output: a constraint register with the obligation, the data it applies to, and the evidence required.

Typical entries: PHI may not leave the covered entity's cloud region; genomic data may not be
re-identified; a payer contract prohibits use for marketing; a European Union (EU) data set is subject to
the General Data Protection Regulation (GDPR) and cannot be transferred to the United States without a
mechanism. These constraints eliminate architectures before cost is considered. Doing this step after
the design is the most expensive ordering mistake in the procedure.

### Step 5 — Decide the trust boundaries

Input: the constraint register and the identity architecture.
Output: a diagram of which users, services, and networks may access which zone, and where encryption
keys are held.

A trust boundary is where the answer to "who can read this" changes. The boundary between the raw zone
and the curated zone is usually not a trust boundary if both live in the same account with the same
administrators. The boundary between the covered entity and a research TRE usually is. Mark boundaries
before choosing products; products are chosen to enforce boundaries, not to define them.

### Step 6 — Choose the storage model per modality

Input: the modality inventory.
Output: a mapping from modality to storage service and file format.

Structured records: columnar files (Parquet) in an open table format, or a warehouse table. Documents:
object storage with extracted text in a searchable store. Images: DICOM Part 10 files in object storage
with metadata indexed in a database, or a picture archiving and communication system (PACS) for clinical
use. Waveforms: vendor-neutral formats such as WFDB or EDF, or Parquet for derived features. Genomics:
compressed FASTA/FASTQ, BAM/CRAM, and Variant Call Format (VCF), with indexes.

The common mistake is forcing every modality into one format. A DICOM file is not improved by being
base64-encoded into a JSON column. Store the original object and index its metadata.

### Step 7 — Select the table format

Input: the engine landscape, concurrency requirements, and the governance model.
Output: a format decision with a version and a compatibility statement.

Ask four questions. Which engines must read the table? Does the organization need multi-writer
concurrency? Does it need row-level deletes or updates? Does it need time travel and audit? Delta Lake
and Iceberg both answer yes to most, with different ecosystem strengths. Record the protocol version:
as of 2026, Delta Lake's protocol ceiling is reader version 3 and writer version 7 with table features,
and Iceberg's production specification is version 3, with version 4 in development. A format decision
without a version is not a decision, because the guarantees differ by version.

### Step 8 — Decide where compute runs

Input: data location, query patterns, cost constraints, and residency.
Output: a compute topology: one regional engine, per-domain engines, or federated query.

Centralized compute is simpler to govern and usually faster for joins across domains. Federated compute
avoids moving regulated data but pushes complexity into query planning and result harmonization.
A hybrid is common: centralize structured data under one governance model, federate imaging and
genomics where volumes and consent differ.

### Step 9 — Define the zones and their contracts

Input: steps 2 through 8.
Output: a zone table with name, purpose, allowed writers, allowed readers, retention, and quality gate.

A minimal contract for the raw zone: append-only, original bytes preserved, checksum recorded, source
metadata recorded, no user edits, retention set by policy. For the standardized zone: one row per
clinical event per source, codes mapped with a versioned vocabulary, units converted to a canonical
unit with the original retained, duplicates flagged but not silently removed. For the curated zone:
documented grain, tested joins, stable identifiers, and a published data dictionary.

### Step 10 — Specify the ingestion mechanism per source

Input: source interface documentation and the latency classification.
Output: a mechanism per source: file drop, REST API, FHIR bulk export, HL7 v2 feed, DICOM store,
database CDC, or streaming topic.

Match the mechanism to the source's guarantees. A vendor that offers a nightly secure file transfer
protocol (SFTP) drop does not have a real-time API, regardless of the brochure. A FHIR server that
supports `$export` should be extracted with the bulk operation rather than by paging every resource
individually. An HL7 v2 interface that sends ADT (admission, discharge, transfer) messages is an event
stream even if it arrives over a TCP socket, not a file.

### Step 11 — Design the raw landing contract

Input: the mechanism decisions.
Output: a landing layout and a manifest format.

Land each extract under a path that encodes source, extract time, and a content hash, for example
`raw/fhir/observations/extract_date=2026-10-24/run_id=.../part-0001.ndjson`. Write a manifest with row
count, byte count, checksum, source endpoint, and retrieval timestamp. Never overwrite an earlier
extract. If the source sends a corrected file, land it as a new extract and let the standardized zone
choose the latest valid record.

### Step 12 — Define identity and deduplication strategy

Input: patient identifiers available at each source.
Output: an enterprise identifier strategy and a duplicate policy.

Options: use the source's medical record number (MRN) within a facility; use an enterprise master
patient index (EMPI) identifier across facilities; derive a pseudonym with a keyed hash; or link
probabilistically. Each has consequences for privacy and for cohort stability. Decide where the
crosswalk lives, who can read it, and how it is versioned. A crosswalk is often the most sensitive table
in the platform.

### Step 13 — Design the standardized model

Input: the source models and the target terminology.
Output: a conformed schema, typically OMOP CDM-shaped for observational health data, with mapping
tables.

Separate terminology mapping from feature logic. Mapping SNOMED CT concepts to standard OMOP concepts is
reusable across every cohort; a sepsis definition is not. Store unmapped codes rather than dropping
them, and produce a mapping-loss report. A 2% unmapped rate on laboratory tests can remove an entire
assay from a cohort without any error being raised.

### Step 14 — Design the curation layer

Input: the standardized model and the consumer requirements.
Output: cohort tables, feature tables, and label tables with documented grain and refresh cadence.

Curation is where clinical logic lives. It must be versioned and tested like software. Each table needs
a definition document that states inclusion criteria, index date, lookback window, and exclusions. The
same definition must produce the same count on a re-run; if it does not, the difference must be
explained by a stated input change, not by nondeterminism.

### Step 15 — Design the consumption interfaces

Input: consumer list.
Output: interfaces: SQL views, APIs, extracts, notebooks, or model endpoints, each with an access
pattern and a service level.

Do not let consumers read the raw zone "just in case." Every consumer should read a curated interface
with a documented contract. If a data scientist needs raw data, that is a signal that the curated layer
is missing a feature, not a reason to widen access.

### Step 16 — Specify the operations plane

Input: the pipeline inventory.
Output: orchestration model, monitoring signals, on-call ownership, and a runbook per pipeline.

At minimum: a scheduler with retry and backfill semantics, freshness and volume monitors, schema-change
detection, lineage capture, cost dashboards, and an incident process. Assign an owner to every pipeline.
A pipeline without an owner is a future outage with no one to call.

### Step 17 — Model cost with units

Input: volumes from step 2, storage tiers, compute pricing, and egress.
Output: annual cost range for at least two options, with assumptions listed.

Separate storage from compute from egress. Storage is usually smaller than teams expect; compute and
egress are usually larger. Model the retrieval pattern for cold data: archive storage is cheap until
someone restores 200 TB. Include the cost of the governance plane (catalog, identity, monitoring),
which is often 10-20% of infrastructure cost and almost always omitted from vendor estimates.

### Step 18 — Write the ADR and set review triggers

Input: all prior steps.
Output: a two-page decision record with context, decision, alternatives, consequences, and review
triggers.

Review triggers are conditions, not dates: "revisit if imaging volume exceeds 500 TB per year," "revisit
if a second writer requires concurrent updates," "revisit if a regulator requires in-country storage."
A date-based review decays into a calendar item; a condition-based review fires when the assumption
breaks.

### Step 19 — Walk the design through three failure scenarios

Input: the proposed architecture.
Output: a written account of what happens to ingestion, cohort counts, and consumers under each failure.

Scenario one: the laboratory feed changes a unit from mg/dL to mmol/L without notice. The standardized
zone should reject the range, route the row to a quarantine table, and raise an alert. If the design has
no quarantine path, the values are coerced and the cohort silently changes.

Scenario two: the source system reprocesses yesterday's encounters and resends 40,000 messages. The
ingestion path should be idempotent on the source key. If it appends, the encounter table doubles for
that day and every downstream rate is wrong.

Scenario three: the object storage credentials expire for six hours. The pipeline should fail at the
ingestion boundary with a clear error, not produce an empty partition that looks like a day with no
patients. An empty partition is more dangerous than a failed job because it passes row-count checks that
only look for growth.

Reason: architecture is judged under failure, not under the happy path. The three scenarios map to the
three guarantees in Section 3.1: schema, identity, and availability.

### Step 20 — Sequence the delivery

Input: the full design and the decision deadline.
Output: a phased plan with a thin vertical slice first.

Deliver one source, one zone path, one consumer, and the operations plane for that path before adding
the second source. A vertical slice proves the contracts, the identity model, and the deployment
mechanism. A horizontal build that lands all sources before any consumer exists produces a platform that
cannot be evaluated and accumulates unmapped data.

Reason: the most common architecture failure is not the wrong choice but the incomplete build. A
lakehouse with no catalog and no tests is a lake with extra steps. Sequencing is part of the design.

### Step 21 — Test the design against a second site

Input: the architecture and a second site's source inventory.
Output: a delta list of assumptions that do not hold at the second site.

Take the smallest site and the most different site. Compare identifiers, code systems, extract
mechanisms, latency, and governance. A design that assumes one enterprise master patient index (EMPI)
and one FHIR release will fail at a site that has neither.

Reason: multi-site architecture is where the abstract design meets the second implementation. The delta
list is cheaper to produce now than after the first site is live.

### Step 22 — Decide what happens when a modality does not fit

Input: a modality whose volume, format, or governance does not match the platform.
Output: a documented exception with an interface, not a silent workaround.

Options: keep the modality in its native system and index its metadata; mirror a research copy into
object storage; or exclude it from the platform and link by identifier. Each option needs an owner, a
linkage key, and a freshness expectation.

Reason: forcing genomics or whole-slide imaging into a structured warehouse produces expensive,
unusable tables. The exception is a legitimate architecture decision if it is documented and governed.

### Step 23 — Record the edge-case decisions

Input: the design review.
Output: a decision table that states what the architecture does for each edge case.

| Edge case | Decision | Where it is enforced | Consequence if missed |
|---|---|---|---|
| Late-arriving record after partition close | Reprocess trailing 7 days; mark `data_through` | Orchestration and gold contract | Understated counts |
| Duplicate or superseded source record | Keep latest by `recorded_at`, retain prior version | Silver dedup rule | Inflated cohort |
| Null in a required field | Quarantine row with reason code | Silver validation | Silent exclusion |
| Sentinel value (`-1`, `9999`, `1900-01-01`) | Map to null with a documented rule | Silver validation | Impossible features |
| Identifier collision across sites | Join on `(system, value)`, not value alone | Identity crosswalk | Merged patients |
| Clock skew or missing timezone | Store UTC, flag unknown offset | Ingestion contract | Misordered events |
| Partial write or partial commit | Transactional partition replace | Load task | Corrupted partition |
| Schema drift in a source | Absorb in bronze, enforce contract in silver | Zone contract | Downstream break |
| Deletion at the source | Tombstone, not hard delete | Silver policy | Stale rows persist |
| Backfill overlapping a live run | Partition lock; one writer | Orchestration | Interleaved output |

Reason: architecture is defined by its behavior at the edges. A design that states only the happy path
leaves each team to invent its own exception handling, and the exceptions diverge.

### Branch point A — One source, one consumer, one week

Collapse the zones. Land the file, transform it in a versioned script, write one curated table, and
document the mapping. The full procedure would delay a decision that does not need it. Keep the raw file
and the script; those two artifacts preserve the ability to reconstruct the result.

### Branch point B — Multi-site research with incompatible governance

Do not centralize raw records. Use a common data model, a distributed analysis package, and a result
aggregation protocol. The architecture becomes a governance and statistics problem: sites must agree on
definitions, and the analysis must be executable at each site. Plan for heterogeneity in coding,
coverage, and data quality rather than assuming a common schema solves them.

### Branch point C — Real-time clinical decision support

Separate the streaming path from the analytical path. The streaming path needs low latency, ordered
processing per patient, and a defined behavior for late events. The analytical path needs completeness
and reproducibility. Forcing one pipeline to serve both produces a system that is neither timely nor
reproducible. Feed the analytical path from the same raw event log, but do not make the alert depend on
a nightly batch.

## 6. Worked example

A five-hospital system, 220 outpatient clinics, and a research institute are choosing a platform for
three workloads: a heart-failure cohort, a quality dashboard, and an imaging-linked research registry.
The decision statement is to consolidate structured data and imaging metadata by June 2027.

### 6.1 Source inventory

| Source | Modality | Annual volume | Objects per year | Update pattern |
|---|---|---:|---:|---|
| EHR structured extract | Parquet/CSV | 1.8 TB | 310 million rows | Nightly full plus incremental |
| Claims | CSV | 0.42 TB | 48 million rows | Monthly |
| Laboratory interface | HL7 v2 | 0.09 TB | 26 million messages | Near-real-time |
| FHIR API | JSON | 0.35 TB | 12 million resources | On demand and bulk export |
| Imaging | DICOM | 43.2 TB | 180,000 studies at 240 MB | Continuous |
| Genomics | CRAM/VCF | 108 TB | 1,200 genomes at 90 GB | Batch |
| Wearables | Parquet | 0.25 TB | 2,000 patients at 1 Hz | Streaming |

The arithmetic behind two rows:

- Imaging: 180,000 studies/year × 240 MB/study = 43,200,000 MB = 43,200 GB = 43.2 TB/year.
- Wearables: 2,000 patients × 86,400 seconds/day × 4 bytes/sample = 691,200,000 bytes/day =
  691.2 MB/day. Over 365 days: 252,288 MB = 252.3 GB = 0.25 TB/year, before compression.

### 6.2 Retention and tiering

Policy: structured and imaging metadata retained seven years; pixels retained seven years with the
first 90 days hot; genomics retained three years, first 30 days hot.

Seven-year raw accumulation:

| Source | 7-year volume | Calculation |
|---|---:|---|
| EHR structured | 12.6 TB | 1.8 × 7 |
| Claims | 2.94 TB | 0.42 × 7 |
| Laboratory | 0.63 TB | 0.09 × 7 |
| FHIR | 2.45 TB | 0.35 × 7 |
| Imaging | 302.4 TB | 43.2 × 7 |
| Genomics (3 years) | 324.0 TB | 108 × 3 |
| Wearables | 1.75 TB | 0.25 × 7 |
| **Total** | **646.77 TB** | sum |

Hot storage at 90 days for imaging and 30 days for genomics, plus all structured data:

- Imaging hot: 43.2 TB × 90/365 = 10.65 TB.
- Genomics hot: 108 TB × 30/365 = 8.88 TB.
- Structured hot: 1.8 + 0.42 + 0.09 + 0.35 + 0.25 = 2.91 TB.
- Hot total: 10.65 + 8.88 + 2.91 = 22.44 TB.

Archive total: 646.77 − 22.44 = 624.33 TB.

At an illustrative 23 USD per TB-month for hot object storage and 4 USD per TB-month for archive:

- Hot: 22.44 × 23 = 516.12 USD/month.
- Archive: 624.33 × 4 = 2,497.32 USD/month.
- Total: 3,013.42 USD/month = 36,161.00 USD/year.

The storage number is not the decision driver. A single 40-hour compute cluster at 4 credits/hour and
6 USD/credit costs 960 USD for one run; twelve such runs a month cost more than the archive tier for the
entire imaging history.

### 6.3 Two options

**Option A: warehouse for structured data, separate PACS and genomics archive.** Lowest change risk for
the clinical teams. Three governance models, three identity integrations, and no cross-modality query
without an export.

**Option B: lakehouse with one catalog, imaging metadata in open tables, pixels in object storage.**
Higher initial integration cost. One identity model and one lineage graph across structured and imaging
metadata. Requires a table format decision and an engine that supports it.

### 6.4 Decision

Choose Option B for structured data and imaging metadata, keep pixels in the existing PACS for clinical
use and mirror research copies into object storage, and keep genomics in a dedicated archive with
metadata registered in the catalog. Record the decision and three review triggers: imaging metadata
above 500 TB, a second concurrent writer requirement, or a residency requirement for genomics.

The reasoning is that the expensive part is not storage but the number of governance models. Three
platforms means three access reviews, three audit logs, and three places a de-identification failure can
occur. Option B reduces that to two, and the remaining split is justified by pixel volume and clinical
workflow.

### 6.5 What the example does not prove

The cost figures are illustrative and must be replaced with current contract pricing. The volume
estimates are point estimates; a sensitivity analysis at ±40% changes the archive cost but not the
architectural conclusion. The decision depends on the organization's ability to operate a lakehouse,
which is an staffing assumption, not a technical one.

### 6.6 Pilot schema and row counts

Before committing to Option B, the team runs the cohort on one hospital's 2024 data. The pilot schema is
the contract that the full platform must reproduce.

| Zone | Table | Grain | Key | Rows |
|---|---|---|---|---:|
| Bronze | `encounter_raw` | One row per source encounter message | `(extract_id, source_file, sha256)` | 4,812,336 |
| Silver | `encounter` | One row per encounter after deduplication | `(source_system, source_record_id)` | 4,788,964 |
| Silver | `person` | One row per person | `person_id` | 412,880 |
| Gold | `hf_cohort` | One row per person with a first HF admission | `person_id` | 18,442 |
| Gold | `hf_features` | One row per cohort person | `person_id` | 18,442 |

The silver count is not the raw count minus duplicates alone. It is raw minus duplicates minus rows
quarantined for a temporal violation:

- Raw: 4,812,336.
- Exact duplicates: 22,224.
- Rows with discharge before admission: 1,148.
- Silver: 4,812,336 − 22,224 − 1,148 = 4,788,964.

### 6.7 Reconciliation from source to cohort

| Stage | Rows in | Removed | Reason | Rows out |
|---|---:|---:|---|---:|
| Raw landing | 4,812,336 | 0 | — | 4,812,336 |
| Deduplicate | 4,812,336 | 22,224 | Same source key, later transmission | 4,790,112 |
| Temporal validation | 4,790,112 | 1,148 | Discharge before admission | 4,788,964 |
| Person spine | 4,788,964 | — | Distinct persons | 412,880 |
| HF diagnosis in 2024 | 412,880 | 391,074 | No qualifying diagnosis | 21,806 |
| Prevalent exclusion | 21,806 | 3,364 | HF diagnosis in prior 365 days | 18,442 |
| Feature build | 18,442 | 0 | — | 18,442 |

The cohort is 18,442 / 412,880 = 4.47% of persons with an encounter. The duplicate rate is 22,224 /
4,812,336 = 0.4618%. Both numbers are recorded in the pilot report, because the platform's silver layer
must reproduce them after migration.

### 6.8 Second run and reproducibility check

The pilot is run twice from the same raw snapshot with the same configuration fingerprint.

| Metric | Run 1 | Run 2 | Expected |
|---|---:|---:|---|
| Raw rows | 4,812,336 | 4,812,336 | Equal |
| Duplicates removed | 22,224 | 22,224 | Equal |
| Silver rows | 4,788,964 | 4,788,964 | Equal |
| Cohort persons | 18,442 | 18,442 | Equal |
| Cohort root hash | `6f2a...c91d` | `6f2a...c91d` | Equal |
| Runtime | 38 min | 41 min | Within 20% |

A difference in any count is a defect, not noise. The usual causes are a nondeterministic window
ordering, a `now()` predicate, or a source extract that changed between runs. The runtime difference is
expected and is not a reproducibility failure.

### 6.9 Two diagnostic checks

The pilot report includes checks that would reveal a wrong answer before it reaches a committee.

**Check A: duplicate rate.** Observed 22,224 / 4,812,336 = 0.4618%. Threshold 0.5%. Pass. A rate above
0.5% indicates a retransmitting interface or a broken deduplication key.

**Check B: orphan encounter rate.** Observed 3,832 / 4,788,964 = 0.0800%. Threshold 0.05%. Fail. The
orphans are encounters whose `person_id` is absent from the person extract, which means the two extracts
were taken at different times. The fix is to extract persons and encounters in the same snapshot or to
quarantine the orphans and report them. Without this check, the cohort would silently exclude 3,832
encounters, and the exclusion would look like a clinical difference.

**Check C: birth-date precision.** Observed 1,207 / 412,880 = 0.2924% of persons with a year-only birth
date. Threshold 0.5%. Pass, with the approximation recorded. If the rate were 8%, the age threshold would
be too imprecise to defend.

### 6.10 What would reverse the decision

The decision is recorded with four reversal conditions, each observable:

1. Imaging metadata exceeds 500 TB, at which point the metadata index and the pixel archive must be
   separated for cost reasons.
2. A second pipeline requires concurrent writes to the same table, at which point the table format and
   partition lock must be revisited.
3. A regulator requires genomic data to remain in-country, at which point the single-catalog assumption
   fails.
4. The platform team cannot staff on-call for two engines, at which point a managed warehouse for
   structured data becomes the lower-risk option.

A decision without reversal conditions is a preference. These four are the conditions under which the
preference would change.

## 7. Code, query, or calculation

### 7.1 Python cost model (standard library, executable)

```python
"""Illustrative storage-tier cost model. Python 3.11+, standard library only."""
from dataclasses import dataclass

HOT_USD_PER_TB_MONTH = 23.0
ARCHIVE_USD_PER_TB_MONTH = 4.0


@dataclass(frozen=True)
class Source:
    name: str
    annual_tb: float
    retention_years: int
    hot_days: int


SOURCES = [
    Source("ehr", 1.80, 7, 365),
    Source("claims", 0.42, 7, 365),
    Source("lab", 0.09, 7, 365),
    Source("fhir", 0.35, 7, 365),
    Source("imaging", 43.20, 7, 90),
    Source("genomics", 108.00, 3, 30),
    Source("wearables", 0.25, 7, 365),
]


def tier_costs(sources: list[Source]) -> dict[str, float]:
    hot_tb = 0.0
    archive_tb = 0.0
    for s in sources:
        retained = s.annual_tb * s.retention_years
        hot = min(retained, s.annual_tb * s.hot_days / 365)
        hot_tb += hot
        archive_tb += retained - hot
    hot_month = hot_tb * HOT_USD_PER_TB_MONTH
    archive_month = archive_tb * ARCHIVE_USD_PER_TB_MONTH
    return {
        "hot_tb": round(hot_tb, 2),
        "archive_tb": round(archive_tb, 2),
        "monthly_usd": round(hot_month + archive_month, 2),
        "annual_usd": round((hot_month + archive_month) * 12, 2),
    }


if __name__ == "__main__":
    result = tier_costs(SOURCES)
    for key, value in result.items():
        print(f"{key}: {value}")
```

Expected output:

```text
hot_tb: 22.44
archive_tb: 624.33
monthly_usd: 3013.42
annual_usd: 36161.0
```

The model assumes linear annual growth, no compression benefit, and no retrieval charges. Each is a
deliberate simplification; the sensitivity analysis in Section 6.5 covers the first.

### 7.2 PostgreSQL 18 zone DDL

PostgreSQL 18 (released 25 September 2025) is the target engine. Schemas stand in for zones. The DDL
shows the contract at each boundary: primary keys, check constraints, and comments that state the
allowed writer.

```sql
-- PostgreSQL 18
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE bronze.encounter_raw (
    extract_id        uuid        NOT NULL,
    source_system     text        NOT NULL,
    source_file       text        NOT NULL,
    payload           jsonb       NOT NULL,
    sha256            char(64)    NOT NULL,
    retrieved_at      timestamptz NOT NULL,
    PRIMARY KEY (extract_id, source_file, sha256)
);
COMMENT ON TABLE bronze.encounter_raw IS
  'Append-only. Writer: ingestion service only. No updates or deletes.';

CREATE TABLE silver.encounter (
    encounter_id      bigint      PRIMARY KEY,
    person_id         bigint      NOT NULL,
    admit_at          timestamptz NOT NULL,
    discharge_at      timestamptz,
    encounter_type    text        NOT NULL,
    source_system     text        NOT NULL,
    source_record_id  text        NOT NULL,
    loaded_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT encounter_time_order
        CHECK (discharge_at IS NULL OR discharge_at >= admit_at)
);
CREATE UNIQUE INDEX encounter_source_key
    ON silver.encounter (source_system, source_record_id);

CREATE TABLE gold.hf_cohort (
    person_id         bigint      PRIMARY KEY,
    index_encounter_id bigint     NOT NULL REFERENCES silver.encounter (encounter_id),
    index_date        date        NOT NULL,
    age_at_index      integer     NOT NULL CHECK (age_at_index BETWEEN 18 AND 120),
    definition_version text       NOT NULL
);
```

Two constraints carry most of the value. The temporal check rejects a discharge before admission, which
is a common interface error. The unique index on the source key makes a duplicate delivery fail loudly
instead of doubling the cohort.

### 7.3 SQLite zone simulation (executable)

SQLite 3.50.4 does not have schemas, but `ATTACH` can simulate zones and the referential and temporal
checks are enforceable.

```python
import sqlite3

con = sqlite3.connect(":memory:")
con.execute("ATTACH DATABASE ':memory:' AS bronze")
con.execute("ATTACH DATABASE ':memory:' AS silver")
con.execute("""
CREATE TABLE bronze.encounter_raw (
    extract_id TEXT, source_system TEXT, payload TEXT, sha256 TEXT,
    PRIMARY KEY (extract_id, sha256)
)""")
con.execute("""
CREATE TABLE silver.encounter (
    encounter_id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL,
    admit_at TEXT NOT NULL,
    discharge_at TEXT,
    CHECK (discharge_at IS NULL OR discharge_at >= admit_at)
)""")

con.execute("INSERT INTO bronze.encounter_raw VALUES (?,?,?,?)",
            ("e1", "epic", '{"enc": 1}', "a" * 64))
rows = [(1, 100, "2026-01-02T08:00:00Z", "2026-01-05T12:00:00Z"),
        (2, 101, "2026-01-03T09:00:00Z", None)]
con.executemany("INSERT INTO silver.encounter VALUES (?,?,?,?)", rows)
try:
    con.execute("INSERT INTO silver.encounter VALUES (?,?,?,?)",
                (3, 102, "2026-02-01T10:00:00Z", "2026-01-31T10:00:00Z"))
except sqlite3.IntegrityError as exc:
    print("rejected:", exc)

print("raw rows:", con.execute("SELECT count(*) FROM bronze.encounter_raw").fetchone()[0])
print("silver rows:", con.execute("SELECT count(*) FROM silver.encounter").fetchone()[0])
```

Expected output:

```text
rejected: CHECK constraint failed: discharge_at IS NULL OR discharge_at >= admit_at
raw rows: 1
silver rows: 2
```

## 8. Data, standards, and regulatory context

Architecture is constrained by standards that say what data mean and by regulations that say how they
may be handled. The two are often conflated. This section separates them.

### 8.1 Standards that constrain the data model

| Standard | Version and status | What it constrains | What it does not |
|---|---|---|---|
| HL7 FHIR | R5, version 5.0.0, published 26 March 2023; R4 4.0.1 remains widely deployed | Resource structure, references, search, bulk export | Analytics schema, cohort logic |
| HL7 v2 | v2.5.1, v2.8, v2.9 depending on interface | Message segments and trigger events | Persistence model |
| OMOP CDM | v5.4 (5.4.3, August 2025); v5.5 released | Observational analytics schema and vocabulary | Source-system semantics |
| DICOM | PS3 base 2026d plus Final Text supplements; ISO 12052:2026 | Imaging objects, metadata, de-identification profiles | Research archive layout |
| LOINC | Current release, updated twice yearly | Laboratory and clinical observation codes | Unit conversion |
| SNOMED CT | National release, updated periodically | Clinical concepts | Billing codes |
| UCUM | Specification for units | Machine-readable units | Which unit is clinically preferred |
| ICD-10-CM | Fiscal year 2026 code set in the United States | Diagnosis coding for billing | Research phenotype definitions |

The binding force differs. FHIR R5 is a standard for trial use in most of its content, with normative
sections inherited from R4; a vendor claiming "FHIR R5 compliant" is claiming conformance to a set of
capability statements, not to the whole specification. OMOP CDM is a community specification, not a
regulation; its authority comes from the tools and studies that depend on it. DICOM is a standard
published by the National Electrical Manufacturers Association (NEMA) and adopted as ISO 12052; its
conformance requirements attach to service classes and profiles, not to an edition number.

### 8.2 Regulations that constrain the architecture

| Instrument | Requirement relevant to architecture | Architectural consequence |
|---|---|---|
| HIPAA Privacy Rule, 45 CFR 164.514 | De-identification standard; safe harbor or expert determination | Determines whether a zone may hold identifiable data and who may access it |
| HIPAA Security Rule, 45 CFR 164.308-312 | Administrative, physical, and technical safeguards for electronic PHI | Access control, audit controls, integrity, transmission security |
| HIPAA Breach Notification Rule, 45 CFR 164.400-414 | Notification duties after a breach | Drives encryption and detection investment |
| 21 CFR Part 11 | Audit trails for electronic records in regulated submissions | Append-only history and attributable changes |
| GDPR (EU) 2016/679 | Lawful basis, data minimization, transfers, data subject rights | Residency and deletion design |
| ONC HTI-1 final rule | Certification of health IT, including decision support transparency | Constrains certified products, not platform topology |

The HIPAA Security Rule does not name a lakehouse, a warehouse, or a cloud region. It requires access
control, audit controls, integrity, and transmission security, and it leaves the mechanism to the
covered entity. A platform that encrypts at rest but has one shared administrator account fails the
access-control requirement regardless of its table format. Conversely, a modest PostgreSQL instance with
per-role grants, audit logging, and encryption can satisfy the rule for a small workload.

ONC's HTI-1 final rule is frequently cited in architecture discussions and rarely relevant to them. It
establishes certification criteria for health information technology, including the Decision Support
Intervention criterion at 45 CFR 170.315(b)(11), with algorithm transparency and risk-management
attestations. It constrains products that seek certification; it does not require a particular data
architecture. Cite it when the question is "can this product be certified," not when the question is
"where should the data live."

### 8.3 What the standards leave open

None of the standards specifies:

- How many zones a platform should have.
- Whether transformation happens before or after landing.
- Which table format to use.
- How long to retain data, beyond legal minima.
- How to resolve patient identity across systems.
- How to model late-arriving data.

Those are engineering decisions, and they are the substance of this topic. A team that can cite FHIR and
OMOP but cannot state its deduplication policy has memorized standards without building an architecture.

## 9. Industry practice and case evidence

### 9.1 The medallion pattern in vendor documentation

Databricks documents the medallion architecture as bronze (raw), silver (validated and cleaned), and gold
(enriched for business analysis), and recommends storing bronze fields as strings or semi-structured
types to absorb source schema changes, cleaning and deduplicating in silver, and aggregating in gold.
That guidance is documented practice from one vendor, not an industry standard. Its useful content is
the separation of concerns and the advice to absorb schema variation at the edge; its unstated
assumption is that the organization operates a lakehouse with a compatible table format.

### 9.2 Open table formats in production

Delta Lake 4.x and Apache Iceberg 1.x are both in production use across large analytics estates. Delta
Lake's protocol is versioned by reader and writer levels; the current ceiling is reader 3 and writer 7
with table features, and the specification is maintained in the `delta-io/delta` repository rather than
in a standards body. Iceberg's specification is published at iceberg.apache.org, with version 3 stable
and version 4 in development. The practical consequence for a biomedical team is that "we use an open
format" is not sufficient: the reader and writer versions determine which engines can read the table and
which features, such as deletion vectors or row lineage, are available.

### 9.3 Federated and trusted environments

OpenSAFELY is a United Kingdom platform that lets approved researchers run code against primary-care
data inside a secure environment rather than extracting records. The documented model combines code
review, audit, and controlled execution. All of Us provides a cloud research workbench with tiered
access to participant data, making access tier and consent part of the product rather than a
post-hoc control. Both are evidence that architecture can reduce data movement while preserving
research utility; neither demonstrates that federation removes the need for harmonization. Sites still
differ in coding, coverage, and data quality.

### 9.4 Multi-site common data models

OHDSI's OMOP CDM is the most widely used common data model for observational health research. Sites
transform local data into the model and run standardized analyses, often without centralizing records.
The documented benefit is comparability of methods and definitions. The documented cost is mapping
loss: source concepts that do not map to a standard concept are either dropped or retained as
source-value rows, and the rate varies by site. An architecture that assumes a common model produces
comparable cohorts without checking mapping loss is making an inference, not a measurement.

### 9.5 What is inference

The following statements are inferences from the evidence above, not documented facts:

- That a lakehouse reduces total cost of ownership for a mid-sized hospital. The cost depends on
  operations staffing and query patterns; published vendor case studies are selected for success.
- That a single catalog improves research reproducibility. It improves discoverability and lineage; the
  reproducibility also depends on code and environment versioning.
- That federation is cheaper than centralization. It avoids egress and duplication but adds
  coordination, site engineering, and analysis time.

Label these as inferences when they appear in a business case.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Transforming raw data in place | No immutable copy exists; a mapping bug cannot be reversed | ETL writes over the landing file | Irreproducible results; retraction risk | Append-only bronze with checksums |
| Zone sprawl | Six schemas with overlapping tables and no owners | Teams create private copies | Inconsistent definitions; access review failure | Catalog with ownership and deprecation |
| Hidden data movement | Egress line item grows faster than volume | Engineers copy data to laptops or personal buckets | Privacy breach; uncontrolled copies | Network controls; sanctioned workspaces |
| Schema drift absorbed silently | New columns appear without review; nulls increase | Schema-on-read tolerates changes | Reports break weeks later; cohort shifts | Schema contracts and drift alerts |
| Format choice without version | "We use Iceberg" with no spec version | Features differ by version | Engine incompatibility at the worst time | Record reader/writer versions in ADR |
| Cost model omits compute | Storage estimate dominates the business case | Vendor pricing emphasizes storage | Budget overrun in month three | Model storage, compute, egress, and people |
| Streaming everything | Topics exist for data no decision uses | "We might need it later" | Cost and operational burden | Retain raw only where a use case exists |
| One administrator account | Shared credentials in a pipeline | Convenience during build | Audit trail cannot attribute actions | Per-service identity; least privilege |
| Federated design without harmonization | Sites return different cohort sizes | No common definitions or mapping tests | Results not comparable | Common model plus distributed quality checks |
| No rollback plan | Restore has never been tested | Backups exist but are unverified | Extended outage after corruption | Tested restore; table snapshots |
| Latency mismatch | Alert refreshes slower than the clinical action | Batch architecture serving real-time decision | Product is clinically useless | Separate streaming path |
| Governance as a final gate | Access requests take weeks and bypassed via exports | Governance not embedded in the platform | Shadow IT; compliance exposure | Policy as code; self-service within bounds |

Three red flags deserve emphasis because they predict failure before any incident.

**The diagram has no owner labels.** If nobody can say who owns the silver zone, nobody will fix it at
2 a.m. Ownership is not a RACI artifact; it is an operational requirement.

**The cost model has one number.** A single annual total hides the structure. Ask for storage, compute,
egress, and personnel separately. If the vendor cannot decompose it, the estimate is marketing.

**The architecture has no deletion path.** Retention is described as "we keep everything." A platform
with no deletion mechanism cannot honor a data use agreement that requires deletion, and it cannot
respond to a subject-access or erasure request.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Source volumes are stable or grow linearly | Cost and capacity planning | A new imaging protocol or genomic assay multiplies volume | Compare 12-month growth against the model quarterly |
| The organization can operate the chosen platform | Staffing and on-call are part of architecture | A two-person team inherits a multi-engine lakehouse | Run a failure drill; measure time to restore |
| Raw data can be retained for the policy period | Reprocessing and audit depend on it | A DUA requires deletion after 12 months | Reconcile retention policy with storage lifecycle rules |
| A common data model preserves clinical meaning | Multi-site comparability | Mapping loss concentrates in the cohort's defining codes | Mapping-loss report by domain and site |
| Network egress is acceptable | Centralized analysis | Residency law or cost makes transfer infeasible | Legal review before design, not after |
| The table format's features are supported by every engine | One copy, many readers | A critical engine lags the spec version | Compatibility matrix per engine version |
| Latency requirements are known | Batch versus streaming decision | A clinical service changes its alerting expectation | Review decision latency with clinical owners |
| Identity resolution is stable | Cohort reproducibility | An EMPI change merges two patients | Version the crosswalk; test cohort count deltas |
| Governance tooling scales to the catalog size | Access review and lineage | Thousands of tables with no stewardship | Sample 20 tables and trace ownership |

The largest limit is organizational. Architecture patterns assume an operating model: who approves a new
dataset, who responds to a freshness alert, who pays for storage. A technically correct design handed to
an organization without those roles will degrade into unowned tables and copied extracts. The procedure
in Section 5 assigns owners at every step for that reason.

A second limit is that this topic treats modalities in isolation. Real platforms have cross-modal
linkage: an imaging study linked to an encounter linked to a genomic sample. The linkage keys and their
governance are often harder than any single modality's storage. Section 6 chooses to keep pixels in a
PACS and mirror research copies, which preserves clinical workflow but creates two copies to govern.
That trade-off is stated, not solved.

## 12. Exercises

### Exercise 1 — Classify six architectures (recall, 20 minutes)

Deliverable: a one-page table with six rows (warehouse, lake, lakehouse, federated, streaming, TRE),
each with the guarantee it provides, the guarantee it does not provide, and one biomedical workload that
fits.

Acceptance criteria: each row names a specific guarantee (for example, "schema enforced at write" or
"raw records do not leave the site"); no row uses "better" or "more modern"; the workload is specific
enough to identify a modality and a decision.

### Exercise 2 — Read a vendor architecture diagram (critique, 30 minutes)

Use any public reference architecture from a cloud or data vendor. Deliverable: a marked-up diagram and
a 300-word critique identifying the governance plane, the missing operations plane, and one implied
data movement that the diagram does not show.

Acceptance criteria: at least three specific omissions; at least one cost implication; no claim that a
product is inherently non-compliant.

### Exercise 3 — Zone contracts (application, 45 minutes)

Given the seven sources in Section 6.1, deliverable: a zone table with raw, standardized, and curated
rows for each source, including allowed writers, readers, retention, and one quality gate per boundary.

Acceptance criteria: every boundary has a named owner role; the raw zone is append-only for all seven
sources; at least one source has a different retention in raw and curated zones with a stated reason.

### Exercise 4 — Cost model (calculation, 60 minutes)

Change the hot-storage periods in Section 7.1 to 30 days for imaging and 14 days for genomics, and add a
retrieval of 20 TB from archive once per year at 90 USD per TB. Deliverable: the modified script and the
new annual total, with the arithmetic shown.

Acceptance criteria: the code runs; the hot and archive totals reconcile to the retained total; the
retrieval charge is listed separately; assumptions are stated.

### Exercise 5 — Table format decision (design, 45 minutes)

A team has PostgreSQL for serving, Spark for transformation, and a requirement for concurrent writes
from two pipelines. Deliverable: a one-page ADR choosing Delta Lake or Iceberg, naming the spec version,
the engine versions, and the feature that satisfies concurrent writes.

Acceptance criteria: the ADR states a version, not just a product; it names a fallback if the feature is
unsupported; it records the condition that would reverse the decision.

### Exercise 6 — Latency design (design, 40 minutes)

A sepsis alert must fire within 15 minutes of a lactate result; a monthly quality report uses the same
data. Deliverable: a diagram with two paths from the laboratory interface, the storage each writes, and
the consistency guarantee between them.

Acceptance criteria: the streaming path has a stated late-event policy; the analytical path is
reproducible from raw; neither path reads the other's intermediate state.

### Exercise 7 — Failure injection (critique, 45 minutes)

Take the Section 6 architecture and remove the raw zone. Deliverable: a 400-word analysis of what becomes
impossible, with three concrete scenarios.

Acceptance criteria: at least one scenario involves a code-mapping error; at least one involves a
regulatory audit; the analysis distinguishes "harder" from "impossible."

### Exercise 8 — Write the ADR (design, 60 minutes)

Deliverable: a two-page ADR for your own organization or a fictional one, following the structure in
Section 5 step 18: context, decision, alternatives, consequences, review triggers.

Acceptance criteria: at least two alternatives; consequences include negative ones; review triggers are
conditions, not dates; the decision statement fits in one sentence.

## 13. Self-check answers

**1. What guarantee does a data warehouse provide that a lake does not?**

Schema-on-write: the structure is enforced when data are loaded, so downstream queries can rely on
column types. A lake applies schema at read time, which tolerates variation but defers failures. The
warehouse does not thereby guarantee correct values, only declared structure.

**2. Why is the medallion pattern not a compliance requirement?**

It is a vendor-documented design convention. HIPAA, GDPR, and ONC certification criteria do not name
bronze, silver, or gold. Compliance depends on safeguards and on de-identification status, not on layer
names. A platform can be compliant with two zones or non-compliant with four.

**3. What are the three guarantees that distinguish architectures?**

The unit of change (row, file, snapshot, event), when the schema is enforced (write or read), and where
computation runs relative to data (centralized or federated). These determine rollback, failure timing,
and privacy respectively.

**4. Recompute the hot and archive totals for a 30-day imaging hot tier.**

Imaging hot: 43.2 × 30/365 = 3.5507 TB. Genomics hot at 30 days remains 8.8767 TB. Structured hot
remains 2.91 TB. Hot total = 2.91 + 3.5507 + 8.8767 = 15.3374 TB. Retained total is unchanged at
646.77 TB, so archive = 646.77 − 15.3374 = 631.4326 TB. Monthly cost = 15.3374 × 23 + 631.4326 × 4 =
352.7601 + 2,525.7304 = 2,878.4905 USD; annual = 34,541.89 USD. The saving against the Section 6 model
is 36,161.00 − 34,541.89 = 1,619.11 USD per year, or 4.48%.

**5. Why does a table format decision need a version?**

Because guarantees are versioned. Delta Lake's reader/writer protocol levels and Iceberg's spec versions
determine which features exist and which engines can read them. Two organizations "using Iceberg" may
have different deletion semantics if one is on version 2 and the other on version 3.

**6. When should a team collapse the zones?**

When there is one source, one consumer, and a short delivery window, provided the raw bytes and the
transformation code are retained. The purpose of the zones is reproducibility and separation of
concerns; if those are preserved by other means, the layers are overhead.

**7. What is the difference between a regulatory requirement and an architectural convention?**

A regulatory requirement is binding law or a certification condition with an enforcement mechanism. An
architectural convention is a documented practice that may be useful but carries no legal force. "Encrypt
ePHI at rest" is a safeguard requirement; "use three zones" is a convention.

**8. Why is a single annual cost number a red flag?**

Because it hides the structure. Storage, compute, egress, and personnel scale differently and are
controlled by different decisions. A single number cannot be sensitivity-tested, and it usually excludes
the governance and operations costs that determine the real total.

**9. What does federation not solve?**

Heterogeneity. Sites still differ in coding, coverage, data quality, and patient mix. A common data model
and a distributed analysis package make results comparable in method, not identical in meaning. Mapping
loss must be measured at each site.

**10. What is the first question to ask about a new source?**

What decision does it support, and how quickly must that decision be made. The answer determines
latency, storage, and whether the source belongs on the streaming path at all.

## 14. Interview questions

**1. "Design a platform for a hospital with EHR, claims, and imaging."**

A strong answer starts with the decision and the constraints, not the products. It inventories volumes
and latency, separates pixels from metadata, names the governance model, chooses one catalog, and states
a cost range with assumptions. A weak answer lists cloud services in a stack diagram and never asks who
owns the data.

**2. "Warehouse or lakehouse?"**

Strong: states the workload, the concurrency requirement, the engine landscape, and the operational
capacity, then chooses with a fallback. Weak: declares one "modern" and the other "legacy."

**3. "How do you handle schema drift?"**

Strong: absorb variation at landing, enforce contracts at the standardized boundary, alert on new or
missing columns, and version the contract. Weak: "schema-on-read handles it," which describes tolerance,
not control.

**4. "Where does de-identification happen?"**

Strong: distinguishes the identifiable zone from the de-identified zone, states the method (safe harbor
or expert determination), and explains that de-identification is a property of a dataset release, not a
folder. Weak: "we mask names in the warehouse."

**5. "How do you decide batch versus streaming?"**

Strong: derives latency from the clinical decision and separates the real-time path from the analytical
path. Weak: picks streaming because it is newer, or batch because it is simpler, without a decision
latency.

**6. "What would you put in the raw zone?"**

Strong: original bytes, checksum, source metadata, retrieval time, and an append-only policy. Weak:
"cleaned data so it is usable," which destroys the ability to reprocess.

**7. "How do you cost this?"**

Strong: separates storage tiers, compute, egress, and personnel; states retrieval assumptions; gives a
range. Weak: multiplies terabytes by one storage price.

**8. "When would you federate?"**

Strong: residency, consent, or contractual limits on transfer, plus an honest account of harmonization
cost. Weak: "for privacy," without explaining what is computed at the site and how results are combined.

**9. "What is an open table format, and why does it matter?"**

Strong: describes metadata plus data files, ACID, time travel, and multi-engine reads, and names a
version. Weak: "it is open source," which is not the property that matters.

**10. "How do you prevent shadow copies?"**

Strong: sanctioned workspaces, egress monitoring, access reviews, and a curated layer that meets the
need. Weak: a policy document with no technical enforcement.

**11. "What is the biggest architecture risk in a research platform?"**

Strong: unowned data and irreproducible transformations, with a specific failure scenario. Weak: "scale,"
without a mechanism.

**12. "How do you review an architecture decision two years later?"**

Strong: condition-based triggers recorded in the ADR, plus metrics that reveal when an assumption
changed. Weak: "we review it annually," with no signal.

## 15. Key terms

- **ACID** — transaction guarantees: atomicity, consistency, isolation, durability.
- **ADR** — architecture decision record; a short, dated decision document.
- **Bronze** — raw landing zone in the medallion convention.
- **CDC** — change data capture from a source system.
- **Compute-storage separation** — independent scaling of processing and durable storage.
- **Data contract** — executable or documented agreement on schema, semantics, and quality.
- **Data lake** — raw, schema-on-read storage for any modality.
- **Data mesh** — domain ownership of data products.
- **Data product** — dataset with an owner, interface, and guarantees.
- **Data residency** — legal constraint on where data may be stored.
- **Data warehouse** — schema-on-write analytics store.
- **Delta Lake** — open table format with a versioned reader/writer protocol.
- **Egress** — data leaving a network or region, often billed.
- **ELT / ETL** — transform after load versus before load.
- **Event time** — when the clinical event occurred.
- **Federated analytics** — analysis at the data location, results combined centrally.
- **Gold** — curated consumption zone in the medallion convention.
- **Iceberg** — open table format maintained by the Apache Software Foundation.
- **Idempotency** — re-running an operation produces the same state.
- **Lakehouse** — lake storage plus an open table format with transactions.
- **Latency** — time from event to decision availability.
- **Lineage** — graph of dataset and job dependencies.
- **Medallion architecture** — bronze/silver/gold layering convention.
- **Object storage** — key-addressed durable storage such as S3.
- **OLAP / OLTP** — analytical versus transactional processing.
- **Partition** — independently writable subset of a table.
- **PHI** — protected health information under HIPAA.
- **Schema-on-read / schema-on-write** — when structure is enforced.
- **Silver** — standardized zone in the medallion convention.
- **Snapshot** — immutable table state at a point in time.
- **TRE** — trusted research environment.
- **Watermark** — lateness threshold for closing a streaming window.
- **Zone** — storage area with an owner, contract, and retention.

## 16. Sources

All URLs accessed 25 October 2026.

1. Databricks. "What is the medallion lakehouse architecture?" https://docs.databricks.com/aws/en/lakehouse/medallion
2. Delta Lake. "Delta Lake Protocol" (PROTOCOL.md), `delta-io/delta` repository. https://github.com/delta-io/delta/blob/master/PROTOCOL.md
3. Delta Lake. "How does Delta Lake manage feature compatibility?" https://docs.delta.io/versioning/
4. Apache Iceberg. "Spec." https://iceberg.apache.org/spec/
5. Apache Iceberg. "Releases." https://iceberg.apache.org/releases/
6. HL7. "FHIR v5.0.0 (R5)." https://hl7.org/fhir/R5/
7. HL7. "FHIR Versions." https://hl7.org/fhir/versions.html
8. OHDSI. "OMOP CDM v5.4." https://ohdsi.github.io/CommonDataModel/cdm54.html
9. OHDSI. "CommonDataModel releases." https://github.com/OHDSI/CommonDataModel/releases
10. NEMA. "DICOM Current Edition." https://www.dicomstandard.org/current
11. NEMA. "PS3.1" (2026d). https://dicom.nema.org/medical/dicom/current/output/html/part01.html
12. ISO. "ISO 12052:2026." https://www.iso.org/standard/86414.html
13. HHS. "Guidance Regarding Methods for De-identification of PHI." https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
14. eCFR. "45 CFR 164.514." https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
15. eCFR. "45 CFR 164.312." https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
16. ONC. "HTI-1 Final Rule." https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
17. OpenSAFELY. https://www.opensafely.org/
18. All of Us Research Program. https://allofus.nih.gov/
19. Johnson, A. E. W. et al. "MIMIC-IV, a freely accessible electronic health record dataset." Scientific Data, 2023. https://doi.org/10.1038/s41597-022-01899-x
20. PostgreSQL Global Development Group. "Release 18." https://www.postgresql.org/docs/current/release-18.html

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_sql-for-clinical-data.md`](02_sql-for-clinical-data.md)
- Related sibling: [`07_data-versioning-and-lineage.md`](07_data-versioning-and-lineage.md)
- Related sibling: [`09_security-engineering.md`](09_security-engineering.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — see Lab 3.1 (cohort pipeline) and Lab 3.6 (incident simulation).
