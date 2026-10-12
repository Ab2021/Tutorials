# Choosing a Database — The Complete Decision Guide

> **TL;DR** — Database choice is not a preference, it is a derivation. Every answer follows from five questions asked in order: **what are the access patterns?** (lookup by key, range scan, join, traversal, similarity, scan-and-aggregate), **what consistency is required?** (linearizable, read-your-writes, bounded staleness, eventual), **what is the scale?** (writes/sec, reads/sec, total size, growth), **how complex are the queries and will they change?** (fixed and known vs ad-hoc and evolving), and **what is the operational burden and cost you can carry?** Answer those and the choice is nearly forced. The transcript's shorthand is a good starting point — file/image/video → blob storage, AI similarity → vector DB, structured relationships → SQL, flexible high-scale → NoSQL, simple key-value → Redis, time series → Cassandra — but the shorthand is the *conclusion*, not the reasoning. Polyglot persistence is usually right, and its cost is real: every additional store is another thing to operate, back up, secure, and keep consistent.

> **60-second interview answer:** "I derive the choice rather than picking a favourite. First the access pattern: if it is a point lookup by a known key at very high QPS, that is key-value; if it is a lookup plus range scans and joins with changing queries, that is relational; if it is a variable-depth traversal, that is a graph; if it is similarity, that is a vector index. Then consistency: if a partial write is a business incident — money, inventory, bookings — it is relational with ACID, full stop. Then scale, which tells me whether one node is enough or whether I need to shard. Then how much operational burden the team can carry, because a self-managed Cassandra cluster is a different commitment from a managed Postgres. In practice I end up polyglot: Postgres as the transactional source of truth, Redis for caching and sessions, object storage for blobs, a search engine or vector index as a derived read model fed by change data capture, and an analytical store fed the same way. And I would say explicitly that each additional store is a real operational cost, not a free optimisation."

---

## 1. Definition & mental model

**Plain English.** Choosing a database means mapping your workload's *shape* onto a storage engine's *guarantees and scaling model*. It is a matching problem, not a ranking problem — there is no best database, only a best fit for a described workload.

**The analogy: choosing a vehicle for a delivery business.**

You do not ask "what is the best vehicle?" You ask: what am I carrying, how far, how often, how fast, and on what roads?

- **A bicycle** — a key-value store. Fast for short trips with one small parcel, useless for a piano.
- **A van** — a relational database. Carries structured loads, handles a mixed route, parks anywhere, and everybody knows how to drive it. Most deliveries are van deliveries.
- **A freight train** — a wide-column store. Enormous tonnage on a fixed route. Brilliant if your cargo fits the route; useless if you need to change where it goes.
- **A crane** — a graph database. The only thing that can lift a specific awkward shape, and you would not use it to deliver letters.
- **A pipeline** — object storage. Moves bulk fluid cheaply and continuously; you cannot take one bucket out mid-flow and edit it.
- **A taxi rank** — Redis. Instant, everywhere, but nothing is stored permanently.
- **A warehouse with a forklift** — an OLAP store. Answers big questions about everything you have, slowly, but never on the critical path.

**The one-line version:** pick the vehicle that matches the cargo, the route and the frequency — and expect to run a fleet.

---

## 2. Why it exists — the problem it solves

### The five questions, in order

```
1. ACCESS PATTERN
   point lookup by key?           → key-value
   lookup + range + join?          → relational
   append + time-ordered reads?    → wide-column
   variable-depth traversal?       → graph
   similarity / nearest neighbour? → vector
   scan and aggregate?             → OLAP
   whole large blob?               → object storage

2. CONSISTENCY
   linearizable / ACID?            → relational, or a strongly consistent mode
   read-your-writes?               → primary routing or a quorum read
   bounded staleness (say, 60 s)?  → any cache or replica
   eventual?                       → any NoSQL store

3. SCALE
   writes/sec, reads/sec, total size, growth rate
   → one node? → replicas? → shard?

4. QUERY COMPLEXITY AND STABILITY
   fixed and known up front?       → wide-column is viable
   ad-hoc and will change?         → relational, or a search engine

5. OPS BURDEN AND COST
   who operates it? what is the budget? what is the team's experience?
```

**Why asking in this order matters:** candidates usually start with scale ("we need a NoSQL database") when the access pattern and the consistency requirement are what actually decide. A workload with 5,000 writes/sec and a need for ACID is a relational problem; a workload with 500,000 writes/sec and an append-only, key-ordered access pattern is a wide-column problem. Same word — "database" — completely different shape.

### What goes wrong without a framework

| Failure | Consequence |
|---|---|
| "We'll use MongoDB because it scales" | No join story, no cross-document transaction story, and now a migration |
| "Postgres can't handle it" | Usually it can, with an index and a cache — and the team already knows it |
| "We need a graph database" | Often a two-hop query with an index would do |
| "Just put it in Redis" | Durability expectations unmet; the cache becomes an accidental source of truth |
| "We'll shard it" | Before caching and indexing were ever tried |
| "One database for everything" | Analytics queries take down the OLTP path; full-text search is a `LIKE '%x%'` |
| **Ten databases** | Ten systems to operate, back up, secure, keep consistent, and hire for |

---

## 3. Internal mechanics — the decision process

### 3.1 The master decision table

| Database type | Data model | Query strength | Consistency | Write scale | Read scale | Best for | Worst for |
|---|---|---|---|---|---|---|---|
| **Relational (Postgres, MySQL)** | Tables, rows, typed columns | Arbitrary joins, aggregations, ad-hoc | ACID | 1k–5k TPS single node | Replicas | Transactions, relationships, evolving queries, reporting | Write-heavy single-key loads; polymorphic schemas |
| **Distributed SQL (CockroachDB, Spanner, TiDB)** | Relational, sharded | Joins across shards | Serializable | 10k–100k+ TPS | Horizontally | Global transactions with relational semantics | Cost; cross-region write latency |
| **Key-value in-memory (Redis)** | Key → value/structures | Key only, plus structure ops | None (async) | 100k+ ops/sec | 100k+ ops/sec | Caching, sessions, rate limits, leaderboards | Durability; datasets larger than RAM |
| **Key-value persistent (DynamoDB, Riak)** | Key → item | Key (and sort key) | Tunable | Hundreds of thousands | Horizontally | Massive-scale lookups with known keys | Ad-hoc queries, joins |
| **Document (MongoDB)** | JSON documents | Fields, nested, some joins | Tunable (majority for strong) | Tens of thousands | Horizontally, sharded | Flexible per-record shape; one entity per read | Multi-entity transactions; deep joins |
| **Wide-column (Cassandra, HBase)** | `(partition, clustering) → columns` | Row key range | Tunable | **Very high, multi-region** | Horizontally | Time series, IoT, events, chat, write-heavy append | Ad-hoc queries; aggregations; small clusters |
| **Graph (Neo4j)** | Nodes + edges | Variable-depth traversal | ACID within a cluster | Low | Replicas | Recommendations, fraud, "how are these connected" | Write scaling; scans; aggregation |
| **Search (OpenSearch)** | Documents + inverted index | Full text, faceting, relevance | Eventual | High ingest | Horizontally | Text search, log analytics, hybrid retrieval | Being the source of truth; transactions |
| **OLAP columnar (ClickHouse, BigQuery)** | Columns | Aggregation over billions of rows | Eventual | High batch ingest | High throughput scans | Analytics, dashboards, reporting | Point updates; per-row access |
| **Time-series (InfluxDB, TimescaleDB)** | Timestamped points | Time-range + downsampling | Tunable | Very high | Horizontally | Metrics, monitoring, sensor data | General-purpose querying |
| **Vector (Milvus, pgvector, Qdrant)** | Embeddings + metadata | Nearest neighbour | Eventual | Moderate | Horizontally | Semantic search, RAG, recommendation candidates | Source of truth; exact-match lookups |
| **Object (S3)** | Immutable blobs | Key lookup + listing | Strong (single region) | Very high | High through CDN | Media, model weights, backups, data lakes | In-place edits, low latency, queries |
| **Block (EBS)** | Raw volumes | Filesystem | Strong | Depends on the FS | Local | Database storage, VM disks | Sharing; scale |
| **File (EFS, NFS)** | Hierarchical files | Path | Strong | Shared | Local-ish | Shared config, legacy apps, ML checkpoints | Massive concurrency |

### 3.2 The worked examples — "design X → choose Y because Z"

These are the answers to rehearse. Each one names the primary store, the derived stores, and the reason.

**1. Bank account transfers**
→ **PostgreSQL.** Because a transfer must be atomic across two rows and a partial write is a business incident — money vanishing. Row locks acquired in a consistent order prevent deadlocks, and `SERIALIZABLE` prevents write skew. No cache on the balance itself; a cache would let a user see a stale balance and make a decision on it. Derived stores: an append-only ledger table for audit, and a warehouse for reporting.

**2. Social feed (Instagram)**
→ **PostgreSQL or a wide-column store as the source of truth for posts, Redis for the precomputed feed, object storage + CDN for media.** Because reads outnumber writes 100:1, so the feed is precomputed on write and read from Redis in a single round trip. Feeds tolerate a few seconds of staleness, so no strong consistency is needed. The celebrity problem means fan-out on read for accounts above a follower threshold ([see Instagram Feed](../../Case-Studies/HLD/04-instagram-news-feed.md)).

**3. Chat messages (WhatsApp)**
→ **Wide-column (Cassandra) or a sharded relational store**, partitioned by conversation. Because the access pattern is "the last N messages in this conversation, in order" — a single-partition, time-ordered slice, which is precisely what a wide-column store is built for. Reliability matters more than latency: a message must never be lost, so it is persisted before the delivery ack ([see WhatsApp](../../Case-Studies/HLD/02-whatsapp-chat-system.md)).

**4. Product catalogue with per-category attributes**
→ **PostgreSQL with `jsonb` and a GIN index**, unless the query pattern demands full-text search. Because the structured parts (price, SKU, stock, seller) benefit from constraints and joins, and the flexible parts go in `jsonb`. A migration to a document store is only justified if the schema is *genuinely* unbounded per product, which it usually is not.

**5. Full-text search across 200 M documents**
→ **OpenSearch/Elasticsearch, fed by CDC from the source of truth.** Because relevance ranking, tokenisation, stemming, faceting and highlighting are a search-engine problem, and `LIKE '%term%'` is not a solution. The search index is *derived*: it can be rebuilt from the source, so it is not the authority.

**6. "Drivers near me" at 1 M location updates/sec**
→ **A wide-column store or Redis geospatial for live positions, with the durable trip data in relational.** Because the update rate is too high for a relational primary and the query is a geospatial range — Redis `GEOADD`/`GEOSEARCH` or PostGIS depending on volume ([see Uber](../../Case-Studies/HLD/06-uber-ride-hailing.md)).

**7. Metrics and monitoring**
→ **Prometheus/InfluxDB/TimescaleDB.** Because the workload is append-only timestamped points with TTL and downsampling — a time-series store does this natively, and a relational store does it expensively.

**8. RAG over internal documents**
→ **Object storage for the documents, a vector DB (often `pgvector`) for the index, and a metadata store for permissions and text.** Because the documents are the authority and the index is derived and rebuildable ([see Vector DBs](03-vector-databases.md)).

**9. Sessions and rate limiting**
→ **Redis.** Because the access is a point lookup by a session or user ID at very high QPS, and losing a session on a restart is acceptable (users re-login), while the latency would be unacceptable from a disk store.

**10. Analytics: "revenue by region by day"**
→ **A columnar warehouse (ClickHouse, BigQuery, Snowflake), fed by CDC.** Because the query scans billions of rows and reads two columns — the definition of a columnar workload — and running it on the OLTP primary would take down production.

**11. Payment ledger / audit trail**
→ **An append-only relational table, or Kafka with long retention plus a warehouse.** Because the requirement is immutability and completeness, not update performance. Never `UPDATE` a ledger row.

**12. Feature store for ML serving**
→ **Redis or a low-latency key-value store for online features; a warehouse or object storage for offline training features.** Because the online path needs single-digit-millisecond lookups by entity ID, and the offline path needs large scans.

### 3.3 Polyglot persistence — and what it really costs

The pattern: one **system of record** that owns the truth, plus **derived stores** that are rebuilt from it.

```
                    ┌──────────────────────────────┐
                    │  SYSTEM OF RECORD            │
                    │  Postgres (ACID, authority)  │
                    └──────────────┬───────────────┘
                          CDC (WAL/binlog → Kafka)
       ┌───────────────┬───────────┼───────────┬────────────────┐
       ▼               ▼           ▼           ▼                ▼
   Redis cache    Search index  Vector index  Warehouse     Cache invalidation
   (read model)   (read model)  (read model)  (analytics)   (eviction events)
```

**The rules that make polyglot work:**
1. **Exactly one system of record per entity.** If two stores both claim authority, they will diverge and nobody will know which is right.
2. **Derived stores are rebuildable.** If you cannot rebuild it from the source, it is a second source of truth in disguise.
3. **Derive via CDC, not application code.** The database emits the event, so every writer — including jobs and migrations — keeps the derived stores correct ([see Cache Invalidation](../03-scaling/04-cache-invalidation-and-consistency.md) §3.5).
4. **Accept the lag and bound it.** Every derived store has a staleness window; say what it is.
5. **Each store costs an on-call rotation.** Five stores means five sets of backups, upgrades, failure modes, and dashboards.

| Cost of adding a store | Detail |
|---|---|
| Operational | Backups, upgrades, failover, monitoring, capacity planning |
| Consistency | One more place to keep in sync, with its own lag and failure modes |
| Cognitive | The team must know two query languages and two failure models |
| Hiring | Finding people who know Cassandra *and* Postgres *and* OpenSearch |
| Cost | Cross-store data transfer, duplicate storage, per-request pricing |
| Migration risk | The more stores, the harder any schema change becomes |

**The honest advice for interviews:** introduce a second store when a specific, named pressure demands it (write throughput, text relevance, similarity, analytics scans), and say what that pressure is. "We'll use polyglot persistence" without naming the pressure sounds like resume-driven development.

### 3.4 Migration between databases — the safe procedure

```
Phase 0  PROVE THE NEED
  Benchmark the current store with realistic data volume and query mix.
  Most "we need to migrate" problems are a missing index or an N+1.

Phase 1  DEFINE THE TARGET SCHEMA
  Model for the new store's access patterns (query-first for wide-column).
  Do not transliterate the old schema.

Phase 2  DUAL WRITE + BACKFILL
  Write to both stores. Backfill history with a throttled, resumable job.
  Old store stays authoritative; the new one is a shadow.

Phase 3  SHADOW READ + COMPARE
  Read from both, compare results, log mismatches. Serve the old value.
  This is where you find the semantic differences (nulls, ordering, types).

Phase 4  GRADUAL CUTOVER
  Flip reads per key range or per tenant. Watch latency and error rates.
  Keep the old store writable for rollback.

Phase 5  DECOMMISSION
  After a soak period measured in weeks, stop dual-writing and remove
  the old store. Keep the export for a defined retention window.
```

**The semantic differences that break migrations, and that nobody plans for:**
- **Collation and ordering** — `ORDER BY` on strings differs by collation, so pagination results change.
- **NULL semantics** — `NOT IN` with NULLs, and whether an empty string equals NULL.
- **Type coercion** — implicit casts that worked in one engine fail in another.
- **Transaction scope** — code that relied on a multi-row transaction now spans two stores.
- **Case sensitivity** of identifiers.
- **Isolation level defaults** — MySQL defaults to repeatable read, Postgres to read committed; a migration between them changes the anomalies you get.
- **Auto-increment vs distributed IDs** — sequences collide once you leave a single node ([see Sharding](../03-scaling/05-sharding-replication-and-consistency.md) §3.12).

**The honest alternative, which is often correct:** migrate *a workload*, not the database. Move search to a search engine, analytics to a warehouse, and leave the transactional data where it is. That is the polyglot answer and it has a fraction of the risk.

### 3.5 The anti-patterns

| Anti-pattern | Why it fails | Do this instead |
|---|---|---|
| **Search engine as the primary DB** | No transactions, no durability guarantees you can rely on, reindexing is a rebuild, no referential integrity | Keep the authority in a relational store; sync to the search engine via CDC |
| **Database as a queue** | Polling tables burns IO; `SELECT ... FOR UPDATE SKIP LOCKED` works but does not scale; no fan-out, no replay, no ordering guarantees across consumers | Kafka/SQS/RabbitMQ |
| **Redis as the source of truth** | Memory-resident with async durability; a failover loses the write window; eviction can silently delete data | Redis as a cache; the authority in Postgres/DynamoDB |
| **A vector DB without a metadata store** | No text, no permissions, no way to reindex, and the ID map breaks on rebuild | A relational metadata store as the authority ([see Vector DBs](03-vector-databases.md) §6.7) |
| **Object storage as a filesystem** | No append, no rename, no directories, listing is paginated | A filesystem, or accept the semantics ([see Blob Storage](04-blob-and-object-storage.md) §6.8) |
| **Analytics on the OLTP primary** | Buffer-pool eviction, lock contention, a single query can take the site down | A replica or a warehouse fed by CDC |
| **Sharding before caching and indexing** | The problem was a missing index; now you also have a distributed system | Walk the escalation ladder ([see Sharding](../03-scaling/05-sharding-replication-and-consistency.md) §2) |
| **A KV store for data you need to query by attribute** | You cannot; every new query is a full scan | Relational or document store |
| **A graph DB for a one-hop lookup** | Overkill; an index does it | Relational with an index |
| **One database for everything** | Each workload gets a compromise that fits none of them | Polyglot, with a named pressure per store |
| **A database per microservice with no source of truth** | Every service's copy diverges; nobody can answer "what is true?" | An owning service per entity; others derive |
| **Choosing the store before the access pattern** | The store dictates the model, and you discover it does not fit | Model the queries, then choose |
| **Ignoring the ops burden** | A self-managed Cassandra cluster with no one who knows it | Managed services, or a store the team already runs |

---

## 4. Variants & taxonomy

| Family | Members | One-line role |
|---|---|---|
| **Relational** | Postgres, MySQL, SQL Server, Oracle | Transactions, relationships, ad-hoc queries |
| **Distributed SQL** | Spanner, CockroachDB, YugabyteDB, TiDB | Relational guarantees at write scale |
| **In-memory KV** | Redis, Memcached, Valkey | Speed: cache, session, counters |
| **Persistent KV** | DynamoDB, Riak, etcd | Point lookups at unbounded scale |
| **Document** | MongoDB, Couchbase, Firestore, DocumentDB | Flexible shape, one entity per read |
| **Wide-column** | Cassandra, ScyllaDB, HBase, Bigtable | Append-heavy, time-ordered, multi-region writes |
| **Graph** | Neo4j, Neptune, JanusGraph | Variable-depth traversal |
| **Search** | Elasticsearch, OpenSearch, Meilisearch, Typesense | Relevance, faceting, log analytics |
| **OLAP columnar** | ClickHouse, BigQuery, Snowflake, Redshift, DuckDB | Aggregation over billions of rows |
| **Time-series** | InfluxDB, TimescaleDB, Prometheus, QuestDB | Timestamped metrics with TTL and downsampling |
| **Vector** | pgvector, Milvus, Qdrant, Weaviate, Pinecone, FAISS | Nearest-neighbour over embeddings |
| **Object** | S3, GCS, Azure Blob, MinIO, R2 | Immutable blobs at scale |
| **Block** | EBS, Persistent Disk, local NVMe | Volumes for databases and VMs |
| **File** | EFS, NFS, SMB, CIFS | Shared hierarchical files |
| **Wide-column analytics** | Bigtable, HBase | Sparse, huge, key-range analytical tables |
| **Streaming store** | Kafka, Pulsar, Kinesis | Ordered, replayable logs |

---

## 5. Pros / Cons

| Decision | Pros | Cons |
|---|---|---|
| **One relational database for everything** | Simplest to operate; ACID everywhere; one skill set; joins work | A bad fit for very high write rates, text relevance, similarity, and analytics scans |
| **Polyglot from the start** | Each workload gets the right tool | N systems to operate, sync, secure and back up; consistency across them |
| **Managed service** | No backups/patching/failover to build; scales elastically | Cost at scale; less control; lock-in |
| **Self-hosted** | Cost control; full tunability; data residency | You operate backups, failover, upgrades, and capacity — forever |
| **Sharding** | Write and storage scale | Cross-shard joins, distributed transactions, rebalancing, ops surface |
| **Caching in front** | Order-of-magnitude read offload; cheap | Invalidation complexity ([see Cache Invalidation](../03-scaling/04-cache-invalidation-and-consistency.md)) |
| **Denormalizing** | Fast reads; no joins | App-maintained invariants that can drift |
| **Deriving via CDC** | Every writer keeps derived stores correct | A pipeline to run; lag to monitor |
| **Staying on Postgres** | 90% of workloads fit; the team knows it; extensions cover JSON, geospatial, vectors, time series | Eventually a genuine ceiling on writes |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 "Postgres cannot scale" is usually false

The most common mis-diagnosis. Before migrating, check, in order:
1. **Missing indexes** on the hot queries → often 10–100×.
2. **N+1 queries** from an ORM → a 100× reduction in query count ([see SQL](01-sql-deep-dive.md) §3.6).
3. **`SELECT *`, `OFFSET` pagination, unbounded `IN` lists.**
4. **Stale statistics** producing catastrophic plans.
5. **No caching** for the hot 1% of reads.
6. **Connection exhaustion**, which looks like a database problem and is not.
7. **Bloat and long transactions** blocking vacuum.

A modern Postgres on good hardware with an index and a cache handles tens of thousands of QPS. The migration to a NoSQL store is frequently an expensive way to avoid adding an index.

### 6.2 The reverse error: forcing a bad fit

Equally common: keeping a workload in Postgres that genuinely does not belong there.
- **1M writes/sec of sensor readings** in a table with 20 indexes → the wide-column store was correct.
- **Full-text search over 200M documents** with `LIKE '%x%'` → a search engine was correct.
- **`SELECT sum(x) FROM big` over a billion rows on the OLTP primary** → a warehouse was correct.
- **Storing 50 GB of video in bytea columns** → object storage was correct.
- **Building semantic search with trigram similarity** → a vector index was correct.

**The signal:** if you are fighting the engine's grain — adding 20 indexes, writing application-level joins, materialising everything — you are in the wrong store.

### 6.3 The consistency requirement is per-operation, not per-database

A system is not "eventually consistent". Operations are.

```
Same application, different guarantees:
  payment amount         → strongly consistent, read from the primary
  account balance shown  → read-your-writes for the owner
  "10,000 customers"     → eventually consistent is fine
  recommendation feed    → minutes of staleness is invisible
  seat availability      → must be checked at the moment of reservation
```

**State this in every design answer.** "The database is eventually consistent" is imprecise; "this read is eventually consistent with a 60-second bound, and this one is a strongly consistent read from the primary" is a design.

### 6.4 The cache that became the database

A team adopts Redis for speed, then starts treating it as the authority: sessions that cannot be lost, the only copy of a cart, a queue of pending jobs. Then a failover happens and the data is gone.

**The rule:** if you would be upset to lose it, it does not belong *only* in the cache. Write to the authority first, then the cache; or accept the loss explicitly and design for it. Specifically: a cart should be written to the database on every mutation (Redis is a read accelerator), sessions should be re-authenticatable, and any queue must be durable.

### 6.5 Derived-store drift

The search index, the vector index, the warehouse and the cache all drift from the source of truth. Silent, gradual, and the failure is invisible — the search results are *slightly* wrong.

**Detection:** periodic reconciliation jobs that compare counts and checksums between the source and the derived store; alerting on CDC consumer lag; and sampled correctness checks in the derived store ("does every document in the index have a matching row?").
**Recovery:** because the derived store is rebuildable, the fix is a backfill — which is only possible if you kept the design rebuildable. **That is the whole reason for the rule.**

### 6.6 Choosing the store before the access pattern

The most consequential process error. Once a team has chosen Cassandra, the data is modelled for Cassandra, and discovering in month three that the product needs a join is a re-architecture, not a query change.

**The discipline:** write down every query with its predicate, ordering and expected volume *before* choosing. A store that cannot serve the list you wrote down is the wrong store, no matter how fashionable it is.

### 6.7 Cost surprises

- **Egress** dominates object storage bills ([see Blob Storage](04-blob-and-object-storage.md) §3.8).
- **Per-request pricing** punishes scans: a DynamoDB `Scan` bills for every item read.
- **Cross-AZ traffic** between an application and its database is billable in most clouds — a chatty ORM can generate a meaningful transfer bill.
- **Provisioned capacity left idle** — a provisioned DynamoDB table or a large RDS instance that is sized for peak.
- **Storage-class minimums** — archiving too early costs more than storing.
- **Managed vector stores bill per pod-hour**, so an idle index is a line item.
- **Duplicated data in derived stores** — the warehouse holds a copy of everything; that is the cost of the analytical capability.

### 6.8 Operational burden is the real differentiator

Two designs can serve the same workload; one needs three people and one needs none.

| Burden | Postgres (managed) | Cassandra (self-managed) |
|---|---|---|
| Backups | Automated | You build and test them |
| Failover | Automated | You configure and game-day it |
| Scaling | Resize or add replicas | Add nodes, rebalance, repair |
| Upgrades | Managed window | Rolling restart with version compatibility |
| Hard failures | Support ticket | You debug it at 3 a.m. |
| Expertise | Everywhere | Rare and expensive |

**For most teams the honest answer is a managed relational database plus a managed cache.** Say that and follow it with the pressure that would justify the next store.

### 6.9 Other traps

- **Premature sharding** — adding a distributed system to solve a single-node problem.
- **Choosing by benchmark** — vendor benchmarks use favourable workloads; benchmark your own query mix.
- **Ignoring the write amplification of indexes** — adding indexes to make reads fast can halve write throughput.
- **Forgetting the reindex cost** of a vector or search store when the embedding model or analyzer changes.
- **No source of truth across microservices** — each service's local store diverges; the system cannot answer "what is true?"
- **Migration without a rollback** — the old store must remain writable through the soak period.
- **Underestimating cardinality** — a "small" dimension table with 500M rows is not small.
- **Assuming a managed service removes the need for a schema** — it removes the *enforcement*, not the need.
- **Time-series in a relational table without partitioning** — the table becomes unmanageable, and every query scans further back than necessary.

---

## 7. When to use / When NOT to use

### The one-page rule set

```
IF the operation must be atomic across multiple records
   → RELATIONAL (ACID), or a distributed SQL database if it must also shard

IF the access is a point lookup by a known key at very high QPS
   → KEY-VALUE (Redis for speed, DynamoDB for volume and durability)

IF one entity is read and written as a whole with a varying shape
   → DOCUMENT

IF writes are append-only, ordered by time, and read as "recent N for a key"
   → WIDE-COLUMN

IF the query is a variable-depth traversal
   → GRAPH

IF the query is "find things similar to this"
   → VECTOR

IF the query is relevance-ranked text search
   → SEARCH ENGINE (derived, never the authority)

IF the query aggregates over billions of rows
   → COLUMNAR OLAP (derived)

IF the payload is a large opaque blob read whole
   → OBJECT STORAGE

IF you need a filesystem, locking, or append
   → FILE or BLOCK storage
```

### When NOT to introduce a new store

- **When an index would fix it.** Most performance complaints.
- **When a cache would fix it.** Read-heavy workloads with a hot subset.
- **When a read replica would fix it.** Read scaling on an existing relational store.
- **When partitioning a table would fix it.** Manageability without a new system.
- **When the team cannot operate it.** An unoperated database is a future outage.
- **When you cannot describe the pressure it relieves in one sentence.** Then it is fashion.
- **When you have no reconciliation plan.** Derived stores drift.

### Decision table by workload

| Workload | Primary | Derived | Why |
|---|---|---|---|
| Payments | Postgres | Ledger table, warehouse | ACID is non-negotiable |
| Inventory | Postgres | Redis for display counts | Conditional updates prevent oversell |
| User accounts | Postgres | Redis sessions | Integrity and uniqueness |
| Sessions | Redis | — | Point lookups at high QPS; loss is tolerable |
| Product catalogue | Postgres (+`jsonb`) | Search engine, Redis cache | Structure plus flexibility |
| Cart | Postgres (authority) | Redis (read accelerator) | Must survive a cache flush |
| Orders | Postgres | Warehouse, search | Transactions plus reporting |
| Chat messages | Cassandra / sharded Postgres | Redis presence | Time-ordered, single-partition reads |
| Social feed | Postgres/cassandra (posts) | Redis (precomputed feed) | Read-heavy, staleness tolerant |
| Media | Object storage | CDN | Blobs, immutable, egress-managed |
| Search | Search engine | — | Derived from the authority |
| Logs | Object storage (compacted) | ClickHouse/Loki | Cheap retention, fast queries |
| Metrics | Time-series DB | Downsampled rollups | Append-heavy, TTL, downsampling |
| Analytics | Columnar warehouse | — | Scans over billions of rows |
| RAG index | Vector DB | Metadata in Postgres | Index is derived; metadata is authority |
| Recommendations | Vector DB + Redis | — | Candidate generation then ranking |
| Geospatial matching | Redis geo / PostGIS | Wide-column for history | Range query at high update rate |
| Feature store (online) | Redis / KV | — | Millisecond lookups by entity |
| Feature store (offline) | Warehouse / object storage | — | Large scans for training |
| Audit log | Append-only table or Kafka | Warehouse | Immutability and completeness |
| Config / service discovery | etcd / Consul / ZooKeeper | — | Strongly consistent KV with watch |

---

## 8. Numbers, defaults & tuning

| Quantity | Value | Reasoning |
|---|---|---|
| Single-node relational reads | 5,000–15,000 QPS | The number behind the caching argument |
| Single-node relational writes | 1,000–5,000 TPS | Depends on fsync and index count |
| Redis ops/sec (single node) | 100,000+ | Single-threaded but in-memory |
| DynamoDB per-partition | ~1,000 WCU, ~3,000 RCU, 10 GB | The scaling unit |
| Cassandra write throughput | 10,000+/sec/node | Linear with nodes |
| Postgres practical single-node ceiling | ~1–2 TB, or when the working set exceeds RAM | The trigger to partition or shard |
| Replication lag threshold | Alert > 5 s | Normal < 100 ms |
| Cache hit ratio target | ≥ 90% for a read-heavy system | Below that, the database is still the bottleneck |
| Cache offload formula | `db_capacity / read_QPS` | 5,000 / 50,000 → need 90% |
| Typical derived-store lag | CDC to cache 10 ms–2 s; to warehouse minutes | Bound each one explicitly |
| Polyglot store count | 3–5 in a mature system | Each beyond that multiplies ops burden |
| Migration dual-write period | Weeks, not days | Soak before decommissioning |
| Reconciliation cadence | Daily counts + checksums | Detect drift before it matters |
| Small-object threshold for tiering | 128 KB | Below it, request costs dominate |
| Connection pool per instance | 5–20 | Times 500 instances needs PgBouncer |
| Typical object storage egress | ~$0.09/GB | Usually the largest single line item |
| Vector index memory | 1.5–2× raw vectors (HNSW) | Sizing the node |
| Search index size vs source | Often 0.5–2× the source data | Plan the cluster |
| Warehouse storage vs source | 1–3× with columnar compression | Usually *smaller* than row-store |
| Time-series retention | Raw 15–90 days, rollups 1–2 years | The standard tiering |
| Kafka retention for CDC | ≥ 7 days | Must exceed the longest consumer outage |
| Benchmark realism | Use your query mix and your data volume | Vendor benchmarks mislead |

---

## 9. Interview Q&A

**Q1 (easy). How do you choose a database?**
By deriving it from the workload rather than picking a favourite. I ask five things in order: what the access pattern is — point lookup, range scan, join, traversal, similarity, or aggregate; what consistency is required; what the scale is in writes, reads and total size; how complex the queries are and whether they will change; and what operational burden the team can carry. The answers usually force the choice. For example, an operation that must be atomic across records means relational regardless of anything else; a point lookup by key at 500,000 QPS means key-value; an append-only time-ordered workload means wide-column.

**Q2 (easy). What is polyglot persistence and is it a good idea?**
It is using several database technologies in one system, each chosen for the workload it fits — typically one relational system of record plus derived stores: a cache, a search index, a vector index, a warehouse, all fed by change data capture from the source. It is usually the right architecture for a mature system because each workload genuinely has a different shape. But the cost is real and I would state it: every additional store is another thing to back up, upgrade, secure, monitor, and hire for, and one more place where the data can drift out of sync. So I introduce a store only when I can name the specific pressure it relieves in one sentence.

**Q3 (easy). When would you use a cache versus a database?**
A cache is for data that is expensive to compute or fetch and is read far more often than it changes. It sits in front of the database, absorbs the repeated reads, and is disposable — if it is lost, you refill it from the database. The database is the authority: anything you would be upset to lose lives there. The failure mode I would call out is a cache that quietly becomes the source of truth — a cart that only exists in Redis, or a session that cannot be recreated. The rule is: if losing it would be a problem, it belongs in the database, and the cache is only a read accelerator.

**Q4 (medium). Walk me through choosing a store for a chat application.**
The access pattern decides it: the dominant query is "the last N messages in this conversation, in order". That is a single-partition, time-ordered slice, which is exactly what a wide-column store is built for — partition by conversation ID, cluster by a time-ordered message ID, query the top of the partition. The second query, "messages from this sender in this conversation", becomes a second table denormalized for that pattern. The consistency requirement is read-your-writes for the sender, so I would route the sender's own reads with a quorum or to the primary. And what is not in the wide-column store is presence and connection state, which is ephemeral and belongs in Redis. The reason I would not use Postgres here is that the write volume at WhatsApp scale exceeds one node — but at a smaller scale Postgres partitioned by conversation would be entirely reasonable, and I would say that.

**Q5 (medium). When is a relational database the wrong choice?**
When one of the four pressures is genuinely present. Very high write throughput beyond one node — millions of writes per second of sensor data or events. A genuinely polymorphic schema where records differ so much that most columns are null for most rows. Relevance-ranked full-text search, which needs an inverted index and scoring. Similarity search over embeddings. And analytical scans over billions of rows, where a row store reads every column for every row. What I would be careful about is the common mis-diagnosis: most "Postgres cannot handle this" claims turn out to be a missing index, an N+1 query, stale statistics, or connection exhaustion, so I would exhaust those before migrating.

**Q6 (medium). What is the difference between a system of record and a derived store?**
The system of record is the authority — if it is wrong, the system is wrong, and it is where writes land first. A derived store is a copy shaped for a particular read pattern: a cache, a search index, a vector index, a warehouse. The defining property is that a derived store can be rebuilt from the system of record. If you cannot rebuild it, it is a second source of truth in disguise, and eventually the two will disagree and nobody will be able to say which is right. That is why I drive derived stores from change data capture rather than application code — the database emits the event, so a cron job or a manual update keeps the derived store correct too, and I can always backfill.

**Q7 (medium). How would you migrate from one database to another?**
Five phases and no shortcuts. Prove the need first with a realistic benchmark, because most migrations are avoiding an index. Then model the target schema for the new store's access patterns rather than transliterating the old one. Then dual-write to both while backfilling history with a throttled, resumable job, with the old store still authoritative. Then shadow-read from both and compare results, serving the old value while I find the semantic differences — collation, null handling, implicit casts, transaction scope, isolation defaults. Then cut over gradually, per tenant or per key range, keeping the old store writable for rollback, and only decommission after a soak measured in weeks. And often the right answer is to migrate a workload rather than the database — move search to a search engine and analytics to a warehouse and leave the transactional data where it is.

**Q8 (hard, testing §6). The team says Postgres cannot handle their scale. How do you evaluate that?**
I would insist on measuring before migrating, because the claim is usually wrong. I would look at the top queries by total time in `pg_stat_statements`, not by mean time; check for missing indexes with `EXPLAIN ANALYZE`; check for N+1 query patterns by counting queries per request; look for `SELECT *`, `OFFSET` pagination and unbounded `IN` lists; check whether statistics are stale after a bulk load; look at whether there is any caching in front of the hot 1% of reads; and check `pg_stat_activity` for connection exhaustion, which presents as a database problem and is an application configuration problem. Then I would check bloat and long transactions blocking vacuum. In my experience, indexes plus a cache plus PgBouncer takes a system much further than people expect, and the migration would have been an expensive way to avoid adding an index.

**Q9 (hard, testing §6). What goes wrong when a derived store drifts?**
It fails silently, which is why it is dangerous. The search index misses a document, the vector index has vectors for deleted documents, the warehouse is missing yesterday's rows, the cache has values the database no longer agrees with. Nothing errors — results are just slightly wrong, and users notice before the metrics do. So I build for it in three ways. Alert on CDC consumer lag, because a stalled consumer is the usual cause. Run periodic reconciliation — row counts and checksums between source and derived — to catch drift that lag monitoring misses. And make sure the derived store is genuinely rebuildable, so the remedy is a backfill rather than a data-loss incident. The specific failure I would guard against is a consumer that throws on one event and skips it, or a retention window that expires before a long consumer outage ends, so it resumes from the head and silently skips everything in between.

**Q10 (hard, testing §6). A team is treating Redis as the source of truth for shopping carts. What is the risk and what do you do?**
The risk is that Redis is memory-resident with asynchronous replication, so a failover loses the last writes, memory pressure can evict keys silently under the right policy, and a restart without persistence loses everything. A cart is a business asset — losing it loses revenue — so it should not live only in a cache. The fix is to make the database the authority and Redis the read accelerator: write the cart to Postgres on every mutation and update Redis at the same time, so a Redis loss means a slower read rather than a lost cart. If the write volume makes that expensive, I would accept a short debounce on the database write, but I would state the loss window explicitly rather than pretending the cache is durable. The general principle is that anything you would be upset to lose does not belong only in the cache.

**Q11 (hard, testing §6). When would you refuse to add a new database to a system?**
When I cannot name the pressure it relieves in one sentence, when an index or a cache or a replica or partitioning would solve the actual problem, or when the team cannot operate it. A database that nobody can back up, fail over, or upgrade is a future outage with a scheduled date. I would also push back when there is no reconciliation plan, because a derived store without drift detection is a slow correctness bug. And I would push back when the store would become a second source of truth — if two systems both claim authority over an entity, they will diverge and no one will be able to say which is right. The cost I would articulate is not the licence or the infrastructure; it is the on-call rotation, the backup and restore drills, the upgrades, and the hiring.

**Q12 (hard). How do you decide between a relational database and a document store?**
By asking whether the records genuinely have different shapes and whether the relationships matter. If the entities are polymorphic — the transcript's social-profile example, where one user has a name, age, city, followers, interests and devices and another has a completely different field set — a rigid schema is awkward and a document store fits naturally. If the relationships matter and I need joins, constraints and transactions across records, relational is correct. In practice the middle ground covers most real cases: Postgres with a `jsonb` column and a GIN index handles structured fields with flexible extras, keeps the referential integrity, and avoids a second database entirely. I would only move to a dedicated document store when the flexible part is the majority of the data and the access pattern is one-document-per-read.

**Q13 (hard). What is the escalation ladder before you shard?**
Indexes and query fixes first, which are usually worth ten to a hundred times on the same hardware. Then a cache, which offloads eighty to ninety-five percent of reads. Then vertical scaling, for a quick two to four times. Then read replicas, for read scaling and failover. Then partitioning the largest tables, which is much cheaper than sharding and requires no application routing. Then sharding, and only when writes or storage genuinely exceed one node. And finally polyglot persistence, moving a specific workload to a store shaped for it. Most systems never need step six, and skipping to it is one of the most common mistakes I see — you acquire a distributed system, cross-shard joins, distributed transactions and rebalancing, to solve a problem that a composite index would have fixed.

**Q14 (hard). How do you keep five stores consistent?**
I do not try to make them consistent; I make them derived. One store is the authority per entity, and everything else is a projection built from change data capture, each with an explicitly stated staleness bound. That way consistency is not a distributed-transaction problem — it is a lag problem, which is bounded and observable. Concretely: the database writes its WAL, a CDC connector publishes to Kafka partitioned by entity ID so ordering is per entity, and each consumer idempotently applies changes to its own store. I alert on consumer lag, run daily reconciliation to catch drift, and because every derived store is rebuildable, the remedy for any divergence is a backfill rather than a reconciliation of two authorities. And if two stores both need to accept writes for the same entity, I have made a design error — one of them has to stop being authoritative.

**Q15 (hard). A startup has three engineers and a product that needs search, analytics and transactions. What do you recommend?**
Managed Postgres as the system of record, because it covers transactions, relationships, ad-hoc queries, and — with `jsonb`, PostGIS and `pgvector` — flexible schemas, geospatial and vector search well enough for the first couple of orders of magnitude. Add Redis as a managed cache when reads get hot. For full-text search, start with Postgres full-text search, which is genuinely capable, and only move to OpenSearch when relevance quality becomes the bottleneck. For analytics, do not add a warehouse until someone is asking questions that hurt production — and then stream to a columnar store rather than querying the primary. The reason is operational: three engineers cannot run five systems, and each additional store is a rotation, a backup strategy, and a failure mode. I would write down the pressure that would trigger each addition, so the decision to add a store is a considered one rather than a reflex.

---

## 10. Common mistakes candidates make

1. **Picking the store before describing the access pattern.**
2. **Saying "we'll use NoSQL because it scales"** without naming which family or which pressure.
3. **Using one database for everything** — analytics on the OLTP primary, `LIKE '%x%'` for search.
4. **Introducing a new store when an index or a cache would do.**
5. **Not stating consistency per operation** — "eventually consistent" for the whole system.
6. **Forgetting the operational burden** — recommending self-managed Cassandra to a three-person team.
7. **Missing the system-of-record rule** — two stores both claiming authority.
8. **Treating the search or vector index as the source of truth.**
9. **Using Redis as the source of truth** for data that must survive a restart.
10. **Using a database as a queue** instead of Kafka or SQS.
11. **Storing large blobs in database rows.**
12. **Sharding before caching, indexing and replicas.**
13. **No reconciliation or CDC-lag monitoring** for derived stores.
14. **Ignoring cost** — egress, per-request pricing, cross-AZ traffic.
15. **Skipping the framework** and answering with a technology preference.

---

## 11. Cheat-sheet summary

| Concept | One-line truth |
|---|---|
| The framework | Access pattern → consistency → scale → query stability → ops burden |
| Access patterns | Lookup, range, join, traversal, similarity, aggregate, blob |
| Relational | Transactions, relationships, ad-hoc queries, evolving schema |
| Distributed SQL | Relational guarantees at write scale, at a latency and cost price |
| Key-value (memory) | Redis: sessions, cache, rate limits, leaderboards |
| Key-value (persistent) | DynamoDB: point lookups at unbounded scale |
| Document | Flexible shape, one entity per read |
| Wide-column | Append-heavy, time-ordered, single-partition reads, multi-region writes |
| Graph | Variable-depth traversal; reads scale, writes do not |
| Search engine | Relevance and faceting — **derived, never the authority** |
| OLAP columnar | Aggregation over billions of rows — derived from CDC |
| Time-series | Append-only points with TTL and downsampling |
| Vector | Nearest neighbour over embeddings — derived and rebuildable |
| Object storage | Immutable blobs; egress dominates the bill |
| System of record | Exactly one per entity; everything else is derived |
| Derived store rule | If you cannot rebuild it, it is a source of truth in disguise |
| Derive via | CDC from the WAL/binlog, partitioned by entity ID |
| Staleness | State a bound per store: 60 s for a cache, minutes for a warehouse |
| Reconciliation | Daily counts and checksums; alert on consumer lag |
| Polyglot cost | Backups, upgrades, on-call, hiring, drift — per store |
| Escalation ladder | Index → cache → vertical → replicas → partition → shard → polyglot |
| Postgres extends to | `jsonb`, PostGIS, `pgvector`, full-text, time-series |
| Migrate a workload | Move search and analytics out; leave transactions in place |
| Migration phases | Prove → model → dual-write → shadow-read → cut over → decommission |
| Semantic traps | Collation, NULLs, casts, isolation defaults, sequence collisions |
| Anti-patterns | Search as a DB, DB as a queue, Redis as truth, vectors without metadata, S3 as a filesystem |
| The one-line test | If you cannot name the pressure in a sentence, do not add the store |

**"If the interviewer says X → answer Y"**

| Interviewer says | You say |
|---|---|
| "Which database?" | "Derive it: access pattern, then consistency, then scale, then query stability, then ops" |
| "SQL or NoSQL?" | "Which pressure, and here is what I give up — transactions, joins, or query freedom" |
| "Postgres won't scale" | "Measure first: index, N+1, statistics, cache, connections — then decide" |
| "Money transfer" | "Relational with ACID, locks in a consistent order, no cache on the balance" |
| "Chat history" | "Wide-column, partition by conversation, cluster by time ID descending" |
| "Full-text search" | "A search engine fed by CDC — derived, never the authority" |
| "Analytics" | "Columnar warehouse fed by CDC; never the OLTP primary" |
| "Semantic search" | "Vector index plus a relational metadata store; the index is rebuildable" |
| "Sessions" | "Redis — point lookups at high QPS and loss is tolerable" |
| "Shopping cart" | "Postgres as the authority, Redis as a read accelerator" |
| "Polyglot?" | "Only with a named pressure per store, and a CDC path plus reconciliation" |
| "Should we shard?" | "Walk the ladder first — index, cache, vertical, replicas, partition" |

---

## 12. Cross-links

- [SQL Deep Dive](01-sql-deep-dive.md) — the default store and the four reasons to leave it
- [NoSQL Deep Dive](02-nosql-deep-dive.md) — the four families and the partition-key discipline
- [Vector Databases](03-vector-databases.md) — the derived index for semantic search
- [Blob and Object Storage](04-blob-and-object-storage.md) — where the large payloads live
- [Sharding, Replication and Consistency](../03-scaling/05-sharding-replication-and-consistency.md) — the escalation ladder and when to shard
- [Cache Invalidation and Consistency](../03-scaling/04-cache-invalidation-and-consistency.md) — CDC, the outbox, and derived-store correctness
- [Caching Strategies and Layers](../03-scaling/02-caching-strategies-and-layers.md) — the cache layer in front of every store
- [Redis Deep Dive](../03-scaling/03-redis-deep-dive.md) — the in-memory store and why it is not the truth
- [CDN Deep Dive](../02-networking/02-cdn-deep-dive.md) — the read path for object storage
- [WhatsApp Chat System](../../Case-Studies/HLD/02-whatsapp-chat-system.md) · [Instagram News Feed](../../Case-Studies/HLD/04-instagram-news-feed.md) · [Uber Ride Hailing](../../Case-Studies/HLD/06-uber-ride-hailing.md) · [BookMyShow](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md) — worked applications of this framework
- [Scaling, Caching and DB Cheatsheet](../../Cheat-Sheets/03-scaling-caching-db-cheatsheet.md)
