# SQL Deep Dive — Relations, Indexes, Transactions and the Query Planner

> **TL;DR** — A relational database is not "the default database"; it is a specific set of guarantees — a declarative query language, ACID transactions, and a query planner that turns *what you want* into *how to get it*. Almost every SQL performance problem in an interview is one of five things: a missing or wrong index, a query that cannot use the index you have, an N+1 query pattern, lock contention, or a connection-pool limit. Almost every correctness problem is an isolation-level problem. The B-tree is why reads are fast and why writes get slower as the table grows; MVCC is why readers do not block writers and why you get `serialization_failure`; and the planner is why the same query is fast for one tenant and slow for another.

> **60-second interview answer:** "I reach for a relational database when the data is structured, the relationships matter, and I need ACID transactions — money, inventory, bookings, anything where a partial write is a business incident. The internals that matter are the B-tree index, which makes range and equality lookups logarithmic but makes every write pay for every index; MVCC, which gives readers a consistent snapshot without blocking writers; and the planner, which chooses between an index scan, a sequential scan and a join strategy using statistics. In practice my first optimisations are always the same: find the missing index with `EXPLAIN ANALYZE`, kill the N+1 with a join or a batched `IN` with eager loading, and put PgBouncer in front because connection-per-instance does not scale."

---

## 1. Definition & mental model

**Plain English.** A relational database stores data in tables of rows and columns with a declared schema, enforces constraints (types, foreign keys, uniqueness), lets you query it declaratively, and gives you transactions that are atomic, consistent, isolated and durable. Underneath, it is a storage engine with a write-ahead log, a buffer pool, a B-tree index structure, and a concurrency-control mechanism (MVCC or locking).

**The analogy: a well-run public records office.**

- **Tables** are the ledgers — deeds, births, marriages — each with fixed columns.
- **The schema** is the rule that a deed must have a parcel number, an owner and a date, and nothing else goes in.
- **Constraints** are the clerks who refuse to file a deed for a parcel that does not exist.
- **Indexes** are the card catalogues: instead of reading every ledger page to find "parcel 42", you go to the catalogue, get a page number, and jump. But every time you file a new deed you must also update every catalogue that references it — which is why more indexes means slower writes.
- **Transactions** are a clerk working on a bundle of filings: either the whole bundle is filed or none of it is.
- **MVCC** is the photocopier: a reader gets a snapshot of the records as of the moment they arrived, so a clerk filing new documents does not block them.
- **The planner** is the senior clerk who decides, per request, whether to use the catalogue or just read the ledger start to finish — and sometimes makes the wrong call because it thinks there are only 1,000 deeds when there are 10 million.

**The one-line version:** SQL is a *declarative* language over a *transactional*, *schema-enforced* store whose performance is entirely determined by index design and planner statistics.

---

## 2. Why it exists — the problem it solves

### What breaks without a relational database

| Need | Without a relational DB |
|---|---|
| "Give me all orders over 100 in the last week, with the customer's name" | Manual application-side joins across files; every new query is new code |
| "Debit A and credit B, or do neither" | Partial writes: money leaves one account and never arrives |
| "Two people must not book the same seat" | Application-level locking that does not survive a second process |
| "The email must be unique" | Duplicates appear under concurrency |
| "Add a field" | Every record must be rewritten or the reader must handle both formats |
| "Report by month" | Full scans of files in application code |

### The specific properties

- **Declarative queries** mean the *what* is separated from the *how*. You write `WHERE status = 'active'`; the planner decides index or scan. This is what makes a new query cost minutes instead of days.
- **ACID** — the property the transcript names first, with the money-transfer example: Anam sends Priya 500 rupees; either both the −500 and the +500 happen, or neither does. Atomicity is what prevents money from vanishing.
- **Constraints as data integrity** — foreign keys, unique constraints, `CHECK`, `NOT NULL`. The database, not the application, guarantees them.
- **A powerful join model** — relationships between normalized entities with no data duplication, and joins executed by the engine with statistics and several algorithms.
- **Consistency of reads** — repeatable reads, snapshot isolation, and serializable when needed.

### What it costs you

- **Schema migration is a real project.** Adding a column with a default on a 500M-row table locks.
- **Horizontal write scaling is hard.** A relational database is fundamentally a single-writer system unless you shard, and sharding breaks joins and constraints ([see Sharding](../03-scaling/05-sharding-replication-and-consistency.md)).
- **The planner can be wrong**, and then a query that was 20 ms becomes 20 s.
- **Operational burden** — vacuum/bloat in Postgres, replication, connection management, backups.

---

## 3. Internal mechanics — how it actually works

### 3.1 The storage engine: pages, the buffer pool and the WAL

```
Table  →  heap file  →  8 KB pages (Postgres) / 16 KB pages (InnoDB)
                          [ page header | tuple pointers | tuples | free space ]

Buffer pool (shared_buffers, default 128 MB in Postgres; innodb_buffer_pool_size, 128 MB in MySQL)
   ↑ the hot set of pages lives here — this is why "indexes fit in RAM" is the whole game

WAL / redo log
   every change is appended sequentially BEFORE the page is modified
   → durability without random writes
   → replication stream (physical) and CDC (logical) come from here
```

**The write path, precisely:**

```
1. BEGIN
2. UPDATE accounts SET bal = bal - 500 WHERE id = 1;
   → the page containing row id=1 is pinned in the buffer pool
   → if not present, it is read from disk
   → the tuple is modified in memory (MVCC creates a NEW tuple version)
   → a WAL record is written describing the change
3. UPDATE accounts SET bal = bal + 500 WHERE id = 2;   (same page? maybe)
4. COMMIT
   → the WAL is flushed to disk (fsync)  ← this is the durability point
   → write latency ≈ fsync latency (~0.1–10 ms depending on storage)
   → the dirty data pages are flushed LATER by the background writer/checkpointer
```

**The key insight:** commits are fast because they are *sequential WAL appends*, not random page writes. But a single fsync per commit caps you at roughly 100–1,000 transactions/sec on a spinning disk, and ~5,000–20,000 on NVMe. **Group commit** batches concurrent commits into one fsync. `synchronous_commit = off` (Postgres) removes the fsync and risks losing the last ~200 ms of commits on a crash — an explicit latency/durability trade worth naming.

### 3.2 MVCC: how readers stop blocking writers

```
Row id=1, bal=500
  tuple v1: xmin=100, xmax=0                ← visible to everyone
UPDATE accounts SET bal=400 WHERE id=1;      (txid 201)
  tuple v1: xmin=100, xmax=201               ← deleted as of txid 201
  tuple v2: xmin=201, xmax=0, bal=400        ← visible to txids ≥ 201

A reader with a snapshot taken at txid 150 sees v1 (bal=500).
A reader with a snapshot at txid 250 sees v2 (bal=400).
Neither blocks the other.
```

| Consequence | Detail |
|---|---|
| **Readers never block writers** | The headline benefit |
| **Old versions accumulate** | Dead tuples must be reclaimed — Postgres `VACUUM`, MySQL purge thread |
| **Bloat** | A table that is updated constantly can be 5–10× its logical size without vacuum |
| **`UPDATE` = delete + insert** | Every update writes a new tuple and every index on the table gets a new entry |
| **Transaction ID wraparound** | Postgres txids are 32-bit; ~2 billion transactions then wrap. Autovacuum must run or the database refuses writes — an outage mode worth naming |

**Why `UPDATE` being a delete+insert matters:** updating a row that has 6 indexes writes 6 new index entries and marks 6 old ones dead. High-index-count tables therefore update slowly. This is the concrete reason "don't index everything".

### 3.3 Indexes: B-tree mechanics

```
                    [ 40 | 80 ]
                   /     |      \
        [10|20|30]   [50|60|70]   [90|100]
        /  |  |  \    ...
     leaf pages hold (key, heap-tuple-pointer) pairs, sorted, doubly linked
```

**Properties:**
- Height is typically **3–4 levels** for millions of rows (fan-out of hundreds per page). So a lookup is 3–4 page reads, and with the top levels cached, often 1.
- Leaf pages are **singly/doubly linked** — that is what makes `BETWEEN` and `ORDER BY` on the indexed column cheap (a range scan walks the leaf chain).
- Insert into a random position causes a **page split**; a sequential insert appends to the rightmost leaf. This is why a random UUID primary key is expensive and a time-ordered one is not.

**Index types:**

| Type | Structure | Good for | Notes |
|---|---|---|---|
| **B-tree** | Balanced tree | `=`, `<`, `>`, `BETWEEN`, `ORDER BY`, `LIKE 'abc%'` | The default; 95% of indexes |
| **Hash** | Hash table | `=` only | No ranges, no ordering; rarely worth it over B-tree |
| **GIN** | Inverted index | `jsonb`, arrays, full-text (`tsvector`), `pg_trgm` | Slower to write, excellent for containment |
| **GiST** | Generalized search tree | Geometry, ranges, nearest-neighbour | Used by PostGIS |
| **BRIN** | Min/max per block range | Huge, naturally-ordered tables (time series) | Tiny (kilobytes) but only useful when data is clustered |
| **SP-GiST** | Space-partitioned | Points, quadtree-like data | Niche |
| **Covering (`INCLUDE`)** | B-tree + extra payload columns | Index-only scans without a heap fetch | Postgres 11+; the biggest single-line win for hot reads |

**Index flavours:**

| Flavour | Meaning | When |
|---|---|---|
| **Simple** | One column | Rarely sufficient alone |
| **Composite** | `(a, b, c)` | Multi-column predicates and ordering |
| **Covering** | `(a) INCLUDE (b, c)` | Avoid the heap fetch entirely |
| **Partial** | `WHERE status = 'active'` | Small, fast, when you only ever query a subset |
| **Expression / functional** | `ON users (lower(email))` | The predicate must match the expression exactly |
| **Unique** | Enforces uniqueness | Also an index |
| **Clustered (MySQL/InnoDB)** | The table *is* the primary-key B-tree | Secondary indexes store the PK, so a secondary lookup is two B-tree walks |

**The leftmost-prefix rule for composite indexes** — the single most tested index fact:

```
Index on (tenant_id, status, created_at)

WHERE tenant_id = 7 AND status = 'active'                     ✅ full use
WHERE tenant_id = 7                                            ✅ uses the first column
WHERE tenant_id = 7 AND created_at > now() - interval '1 day'  ⚠️ skips status;
                                                                   created_at usable
                                                                   only as a filter
WHERE status = 'active'                                        ❌ cannot use the index
WHERE created_at > X                                            ❌ cannot use the index
WHERE tenant_id = 7 AND status = 'active' ORDER BY created_at   ✅ index also
                                                                   supplies the order
```

**Rule:** put equality predicates first, then the range/sort column last. An index on `(a, b)` will not serve a query on `b` alone — but a separate index on `b` will.

**Selectivity** decides whether an index is worth using at all:

```
selectivity = distinct_values / total_rows

gender (2 values)            → 0.000002  → terrible index; a scan is faster
email (10M values)           → 1.0       → perfect index
status (5 values, 90% 'done') → 0.0000005 → the index is useless for 'done',
                                            great if you add a partial index
                                            WHERE status='pending'
```

**The planner's choice:** if a predicate matches more than roughly **5–10% of the table**, a sequential scan is usually cheaper than an index scan plus random heap fetches. This is why "add an index" does not always help — and why a query that is index-driven for a small tenant becomes a scan for a large one.

**Index-only scans** are the exception: with a covering index and a well-vacuumed table (visibility map), the planner can answer from the index alone and skip the heap. They are dramatically faster and are the main reason to use `INCLUDE`.

### 3.4 The planner: EXPLAIN and join algorithms

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT c.name, count(*) AS orders
FROM customers c JOIN orders o ON o.customer_id = c.id
WHERE o.created_at > now() - interval '7 days'
GROUP BY c.name
ORDER BY orders DESC LIMIT 10;
```

**What to look for, in order:**
1. **`Seq Scan` on a big table with a selective filter** → missing index, or the predicate is not sargable.
2. **`rows=1` actual vs `rows=100000` estimated** → stale statistics. `ANALYZE`, or raise the statistics target.
3. **`Nested Loop` with a large outer row count** → the classic N+1-shaped plan; consider a hash join.
4. **`Sort` with `external merge Disk`** → `work_mem` too small; the sort spilled to disk.
5. **`Rows Removed by Filter: 900000`** → the index got you close but the filter is doing the work.
6. **`Buffers: shared read=...`** → how much came from disk vs cache.

**The three join algorithms:**

| Algorithm | How | Cost model | When the planner picks it |
|---|---|---|---|
| **Nested loop** | For each outer row, probe the inner | O(N × M) without an index on the inner; O(N × log M) with one | Small outer side, indexed inner side |
| **Hash join** | Build a hash table of the smaller side, probe with the larger | O(N + M), needs `work_mem` | Medium-to-large equi-joins with no useful ordering |
| **Merge join** | Sort both sides, merge | O(N log N + M log M), or free if both sides are already index-ordered | Both sides large and already sorted, or a range join |

**Join ordering** is planned with statistics; the planner enumerates orders up to `join_collapse_limit` (default 8) relations, then falls back to greedy. A query with 15 joins can get a bad plan because the search space was truncated.

**Non-sargable predicates** — the classic index killers:

```sql
WHERE DATE(created_at) = '2026-09-27'          ❌ function on the column
WHERE created_at >= '2026-09-27'
  AND created_at <  '2026-09-28'                ✅ sargable range

WHERE lower(email) = 'a@b.com'                  ❌ unless you have an index
                                                   on lower(email)
WHERE email = 'a@b.com'                         ✅ if email is stored lowercased

WHERE id::text = '42'                           ❌ cast defeats the index
WHERE name LIKE '%smith%'                       ❌ leading wildcard: no index
WHERE name LIKE 'smith%'                        ✅ prefix: B-tree works
```

### 3.5 Transactions, locking and deadlocks

**Row-level locks (Postgres):** `FOR UPDATE` (exclusive, blocks other writers and `FOR UPDATE`), `FOR NO KEY UPDATE`, `FOR SHARE`, `FOR KEY SHARE`. Plain `SELECT` takes no lock at all under MVCC.

**The four lock modes that matter in an interview:**
- **Row lock** — the default for `UPDATE`/`DELETE`. `SELECT ... FOR UPDATE` takes it explicitly and holds it until commit.
- **Table lock** — `LOCK TABLE`, and implicitly by DDL (`ALTER TABLE`). The killer: an `ALTER TABLE` waiting behind a long transaction blocks *every subsequent query* on that table (the lock queue is FIFO). Fix: `lock_timeout` on DDL so it fails fast and retries.
- **Gap lock (MySQL/InnoDB)** — locks the gaps between index values to prevent phantoms under repeatable read. Under contention it produces deadlocks that Postgres, using SSI instead, does not have.
- **Advisory lock** — application-defined (`pg_advisory_lock`); a database-backed mutex for things like "only one worker runs this job".

**Deadlock, exactly:**

```
Tx A: UPDATE accounts SET bal=bal-100 WHERE id=1;   -- holds row 1
Tx B: UPDATE accounts SET bal=bal+100 WHERE id=2;   -- holds row 2
Tx A: UPDATE accounts SET bal=bal+100 WHERE id=2;   -- waits on B
Tx B: UPDATE accounts SET bal=bal-100 WHERE id=1;   -- waits on A   → cycle!
→ the database detects the cycle in ~1 s (deadlock_timeout) and kills one
   → ERROR: deadlock detected
```

**Fixes:** acquire locks in a **consistent order** (always lower id first — that is why the money-transfer pattern is `ORDER BY id`), keep transactions short, avoid `SELECT ... FOR UPDATE` followed by user interaction, and **retry on deadlock** (it is a normal, expected error, not a bug).

**Serialization failures** under Postgres `SERIALIZABLE`/`REPEATABLE READ` are different: the transaction is aborted because it could not be serialized. The application must retry. A production system at serializable isolation *must* have retry logic; without it, error rates spike under load.

### 3.6 The N+1 problem

```python
# ❌ N+1: 1 query for the list + N for the details
orders = db.query("SELECT * FROM orders WHERE customer_id = %s", cid)   # 1
for o in orders:                                                        # N
    o.customer = db.query("SELECT * FROM customers WHERE id = %s", o.customer_id)

# 100 orders → 101 round trips → 101 × 0.5 ms network + query = ~150 ms

# ✅ One join
rows = db.query("""
    SELECT o.*, c.name, c.email
    FROM orders o JOIN customers c ON c.id = o.customer_id
    WHERE o.customer_id = %s
""", cid)

# ✅ Or two queries and join in the application (often better — no row explosion)
orders = db.query("SELECT * FROM orders WHERE customer_id = %s", cid)
ids = {o.customer_id for o in orders}
customers = db.query("SELECT * FROM customers WHERE id = ANY(%s)", list(ids))
by_id = {c.id: c for c in customers}
```

**Why it is so common:** almost every ORM's lazy loading does it by default. Django's `select_related` (join) and `prefetch_related` (second query), Rails' `includes`, Hibernate's `JOIN FETCH` and `@BatchSize`, SQLAlchemy's `joinedload`/`selectinload`. **In an interview, saying "I would check the ORM's eager-loading configuration and the query log" is a strong answer** — N+1 is found by counting queries per request, not by reading code.

**The related trap:** the join fix can explode rows (1 order × 10 line items = 10 rows), which inflates the payload. The two-query + in-application-join approach avoids both the N+1 and the row explosion.

### 3.7 Normalization and denormalization

| Form | Rule | What it prevents |
|---|---|---|
| **1NF** | Atomic values; no repeating groups | Multi-value columns like `tags = "a,b,c"` |
| **2NF** | No partial dependency on part of a composite key | Order-line storing the product name |
| **3NF** | No transitive dependency (non-key → non-key) | `orders.zip` and `orders.city` when `zip → city` |
| **BCNF** | Every determinant is a candidate key | Rare edge cases |

**Practical guidance:** normalize to 3NF for the **write path** (correctness, no update anomalies), denormalize for the **read path** (avoid joins and aggregation at query time). The transcript's framing is the right one: SQL because the data is structured and the *relationships matter*.

**Denormalization techniques, cheapest first:**
1. **A cached derived column** (`order_total` maintained by a trigger) — fast reads, write amplification.
2. **A materialized view** — refresh on a schedule or concurrently.
3. **A read model / CQRS projection** — a table shaped exactly like the API response.
4. **An external read store** — Redis or a search index built from CDC.

**The cost of denormalization is always the same:** an invariant that the database no longer enforces must now be enforced by application code, a trigger, or a background job — and it can drift.

### 3.8 Partitioning

Splitting one logical table into physical partitions *within one database* — cheaper than sharding, and often the right answer first.

```sql
-- Declarative range partitioning by month (Postgres 10+)
CREATE TABLE events (id bigserial, created_at timestamptz NOT NULL, payload jsonb)
PARTITION BY RANGE (created_at);

CREATE TABLE events_2026_09 PARTITION OF events
  FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
```

| Type | Use | Example |
|---|---|---|
| **Range** | Time series, archival | `events_2026_09` |
| **List** | Enumerated sets | `orders_us`, `orders_eu` |
| **Hash** | Even spread when no natural range | `PARTITION BY HASH (user_id)` with 16 partitions |

**What you get:** partition pruning (a query on one month touches one partition), cheap bulk deletion (`DROP TABLE events_2025_01` instead of a 50M-row `DELETE`), smaller indexes per partition, and parallel maintenance.
**What you do not get:** it does not spread writes across machines, and a query without the partition key touches every partition.

**The difference to state clearly:** partitioning = one database, many tables, for manageability. Sharding = many databases, for write and storage scale.

### 3.9 Connection pooling — PgBouncer

```
Without a pooler:   500 app instances × 20 connections = 10,000 Postgres backends
                    × ~10 MB each = 100 GB RAM, plus context-switch death

With PgBouncer:     500 instances × 20 client connections → PgBouncer
                    → 50 actual Postgres connections
```

| Pool mode | Behaviour | Constraint |
|---|---|---|
| **Session** | One server connection per client for the whole session | Safe; no real multiplexing benefit |
| **Transaction** | A server connection is assigned per transaction | **The default for web apps**; but prepared statements, `SET`, advisory locks and `LISTEN` do not survive between transactions |
| **Statement** | Per statement | Only safe without multi-statement transactions |

**Numbers:** Postgres `max_connections` default 100; each backend costs ~5–10 MB plus work memory. The classic sizing rule is `connections ≈ (core_count × 2) + effective_spindle_count` — on modern NVMe, often **50–200 is plenty**, and going beyond that *reduces* throughput because of contention on internal locks.

**The failure mode nobody mentions:** a pooler introduces a queue. If all 50 server connections are busy, requests wait — and a slow query turns into a queue of waiting clients, which turns into timeouts, which turns into retries, which makes the queue worse. **Always set `query_wait_timeout` and `statement_timeout`.**

### 3.10 Read replicas and read routing

```
Writes ─────▶ Primary ──WAL──▶ Replica 1 (analytics)
                       └─────▶ Replica 2 (read traffic)
                       └─────▶ Replica 3 (failover candidate)
```

**Routing rules:**
- Reads that tolerate staleness → replica.
- Reads immediately after a write by the same user → primary (read-your-writes), or check the replica's replay LSN.
- Analytics/reporting → a dedicated replica, so a long query does not stall replay on the read replica ([see Sharding & Replication](../03-scaling/05-sharding-replication-and-consistency.md) §6.6).
- Transactions with writes → always primary.

**Never** point an application write at a replica. Modern managed replicas reject writes, but a misconfigured one silently diverges and then fails on promotion.

### 3.11 OLTP vs OLAP

| | OLTP | OLAP |
|---|---|---|
| Workload | Many small reads/writes | Few huge scans/aggregations |
| Rows touched per query | 1–100 | 10⁶–10⁹ |
| Storage layout | **Row-oriented** | **Column-oriented** |
| Indexes | Many B-trees | Sparse/zone maps, sort keys |
| Compression | Low | High (same-typed adjacent values) |
| Concurrency | Thousands of short transactions | Tens of long queries |
| Examples | Postgres, MySQL, Aurora | Snowflake, BigQuery, Redshift, ClickHouse, DuckDB |
| Bottleneck | Random IO, lock contention, fsync | Sequential IO, CPU, memory bandwidth |

**Row vs column, concretely:** `SELECT sum(amount) FROM orders WHERE month = 'Sep'` over 100M rows. Row-oriented reads every column of every row (`SELECT *` cost) — hundreds of GB. Column-oriented reads only the `amount` and `month` columns — a few GB, and those compress 5–20×. That is the entire difference.

**The practical rule:** never run analytics on the OLTP primary. Stream changes to a warehouse (CDC → Kafka → warehouse, or a managed connector) and query there. A single unindexed analytical query on a production table can take the site down via lock contention and buffer-pool eviction.

### 3.12 Materialized views

```sql
CREATE MATERIALIZED VIEW mv_daily_revenue AS
SELECT date_trunc('day', created_at) AS day, sum(amount) AS revenue
FROM orders GROUP BY 1;

-- Non-blocking refresh (needs a unique index)
CREATE UNIQUE INDEX ON mv_daily_revenue (day);
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_revenue;
```

- **What it is:** a query result stored as a table. Reads are instant.
- **What it costs:** staleness (until the next refresh) and refresh cost (a full recompute unless incremental).
- **`CONCURRENTLY`** does not lock readers but is slower and requires a unique index; a plain refresh takes an `ACCESS EXCLUSIVE` lock and blocks reads.
- **When to use:** dashboards with a tolerance of minutes, precomputed aggregates over large tables.
- **Alternative:** maintain the aggregate incrementally with triggers or a write-back counter ([see Cache Invalidation](../03-scaling/04-cache-invalidation-and-consistency.md)).

---

## 4. Variants & taxonomy

| Variant | Description |
|---|---|
| **Postgres** | MVCC, rich types (jsonb, arrays, ranges), extensions, GIN/GiST/BRIN |
| **MySQL / InnoDB** | Clustered PK index, gap locks, `pt-online-schema-change`/`gh-ost` for DDL |
| **Aurora / RDS** | Managed; a shared storage layer with 6-way replication; reader endpoints |
| **SQLite** | Embedded; WAL mode; single-writer; perfect for local/edge |
| **CockroachDB / YugabyteDB / TiDB** | Distributed SQL — Raft per range, horizontal write scaling, serializable by default |
| **Spanner** | Globally consistent via TrueTime; ~7 ms commit wait |
| **Vitess / Citus** | Sharding middleware in front of MySQL / Postgres |
| **DuckDB** | In-process columnar OLAP — the "SQLite of analytics" |
| **ClickHouse** | Columnar OLAP, MergeTree, extremely fast aggregations |
| **Snowflake / BigQuery / Redshift** | Cloud warehouses, separated storage and compute |
| **Postgres + `pgvector`** | Vectors in the relational database ([see Vector DBs](03-vector-databases.md)) |
| **Foreign data wrappers** | Query another database through Postgres (`postgres_fdw`, `file_fdw`) |

---

## 5. Pros / Cons

| Aspect | Pros | Cons |
|---|---|---|
| **Schema** | Guarantees shape; catches bugs at write time; self-documenting | Migrations on large tables are risky and slow |
| **ACID** | Correct by construction for money and inventory | Lock contention; serialization failures need retries |
| **Joins** | No data duplication; flexible queries | Cross-shard joins impossible once you shard |
| **Indexes** | Millisecond lookups; rich types (partial, covering, expression) | Every index slows every write; misdesigned indexes are worse than none |
| **Planner** | Adapts to data distribution; no manual plan hints needed | Bad statistics → catastrophic plans; plan regressions after upgrades |
| **MVCC** | Readers never block writers | Bloat, vacuum burden, txid wraparound risk |
| **Scaling reads** | Replicas are easy | Lag; read-your-writes problems |
| **Scaling writes** | Distributed SQL exists (Cockroach, Spanner) | Complex, expensive, and cross-region writes cost milliseconds |
| **Operational maturity** | 40 years of tooling, backup, monitoring | Connection management, vacuum, bloat, failover all need care |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 Stale statistics and the plan that flips

The planner uses `pg_statistic`, updated by autovacuum/`ANALYZE`. After a bulk load of 10M rows, statistics are stale: the planner thinks a table has 1,000 rows, chooses a nested loop, and a query that took 20 ms takes 40 s.

**Symptoms:** `EXPLAIN` estimated rows differ wildly from actual; a query that is fast in staging and slow in production; a plan that changes after a deploy.

**Fixes:** `ANALYZE` after a bulk load; `ALTER TABLE t ALTER COLUMN c SET STATISTICS 500` for an important skewed column; `default_statistics_target` default 100; and **plan regression testing** — capture plans for critical queries in CI (e.g. `pg_plan_guards` or a golden `EXPLAIN` diff).

### 6.2 The "add an index" failure

Adding an index can make things *worse*:
- **Low selectivity** — an index on `status` where 90% of rows are `'done'` costs the planner an extra structure to maintain and is never chosen.
- **Redundant indexes** — `(a)` and `(a, b)`; the second serves every query the first does. Drop the first.
- **Too many indexes on a write-heavy table** — 8 indexes means 8 B-tree insertions per row inserted, and 8 new entries per update (MVCC). Write throughput can fall 3–5×.
- **`CREATE INDEX` without `CONCURRENTLY`** locks writes on the table. Always `CREATE INDEX CONCURRENTLY` in production (it is slower and can fail leaving an invalid index — check `pg_index.indisvalid`).
- **Unused indexes** — find them: `SELECT * FROM pg_stat_user_indexes WHERE idx_scan = 0;`

### 6.3 Lock contention and the DDL lock queue

```
Session 1: BEGIN; SELECT * FROM orders WHERE id=1 FOR UPDATE;   (holds a row lock,
                                                                and the transaction
                                                                stays open)
Session 2: ALTER TABLE orders ADD COLUMN note text;             (waits for the
                                                                ACCESS EXCLUSIVE lock)
Session 3: SELECT * FROM orders WHERE id=2;                     ← BLOCKED BEHIND THE DDL
```

**This is the classic production incident:** one long-running transaction plus one `ALTER TABLE` blocks *all* traffic on the table, including plain `SELECT`s, because the lock queue is FIFO.

**Fixes:** always set `lock_timeout` before DDL (`SET lock_timeout = '2s'`), retry if it fails; kill long idle-in-transaction sessions (`idle_in_transaction_session_timeout`, default 0 — set it to 60 s); use `gh-ost`/`pt-online-schema-change` for MySQL; use `ADD COLUMN` without a default rewrite where the version allows.

### 6.4 Deadlocks and serialization failures

Covered in §3.5. The interview-critical point: **both are normal, expected errors and must be retried**, with a bounded retry count and jitter. A `deadlock detected` error reaching the user is an application bug, not a database bug.

### 6.5 N+1 and the row-explosion counter-bug

Covered in §3.6. Additional trap: fixing N+1 with a join can multiply rows, and an `IN` list of 50,000 IDs is itself a problem — batch it into chunks of ~1,000, or use a temporary table / `VALUES` join.

### 6.6 Connection exhaustion

```
500 instances × 20 connections = 10,000  >  max_connections = 500
→ FATAL: sorry, too many clients already
```

**The runaway pattern:** one slow query holds connections; the pool fills; new requests wait for a connection; they time out and retry; the retries consume the pool; the database is fine but the application is down. **This is a self-inflicted outage and it is extremely common.**

**Fixes:** PgBouncer in transaction mode; small per-instance pools; `statement_timeout` and `query_wait_timeout` set; a circuit breaker in front of the database client; and dashboards on `pg_stat_activity` count by state.

### 6.7 Bloat, vacuum and transaction-ID wraparound

- **Bloat:** dead tuples accumulate; a 100 GB table can be 400 GB on disk. Detect with `pg_stat_user_tables.n_dead_tup` and `pgstattuple`. Fix with `VACUUM (FULL)` (takes an exclusive lock — avoid) or `pg_repack` (online).
- **Long-running transactions block vacuum** — they hold a snapshot, so vacuum cannot remove tuples newer than that snapshot. A single `idle in transaction` session can freeze cleanup for the whole database. **This is the single most damaging Postgres anti-pattern.**
- **Wraparound:** at ~2 billion transactions without a vacuum, Postgres refuses new writes to protect the data. The error is unmistakable and the fix (`VACUUM FREEZE`) is slow. Monitor `age(datfrozenxid)` and alert at 1.5 billion.

### 6.8 Replica lag and the reads that follow writes

Covered in §3.10 and in [Sharding & Replication](../03-scaling/05-sharding-replication-and-consistency.md). The specific SQL-level symptom: "I saved my profile and it went back to the old value when I reloaded."

### 6.9 Hot rows and the counter anti-pattern

`UPDATE counters SET n = n + 1 WHERE id = 1` on a single hot row serializes every increment on a row lock. At 10,000 increments/sec on one row, throughput collapses.

**Fixes:** shard the counter into N rows (`counter_1..counter_10`) and sum on read; batch increments in memory and flush every second; use a Redis `INCR` and write back periodically ([see Caching](../03-scaling/02-caching-strategies-and-layers.md)); or use an append-only event table and aggregate (a "counter as a log").

### 6.10 Other failure modes worth naming

- **`SELECT *`** — reads unnecessary columns, breaks covering indexes, breaks when the schema changes, and inflates the network payload.
- **`OFFSET 100000 LIMIT 20`** — the database reads and discards 100,000 rows. Use keyset pagination (`WHERE (created_at, id) < (?, ?) ORDER BY created_at DESC, id DESC LIMIT 20`).
- **Unbounded `IN` lists and unbounded `DELETE`s** — a `DELETE FROM logs WHERE created_at < X` on 50M rows takes an `ACCESS EXCLUSIVE`-adjacent lock, generates enormous WAL, and holds a snapshot that blocks vacuum. Batch it.
- **`COUNT(*)` on a large table** — a full index scan; use an approximate count (`pg_class.reltuples`) or a maintained counter.
- **Implicit casts** — `WHERE varchar_col = 42` may prevent index use.
- **Time-zone confusion** — mixing `timestamp` and `timestamptz`. Always store `timestamptz` (UTC) and convert at the edges.
- **Floating point for money** — use `numeric`/`DECIMAL`, never `float`. `0.1 + 0.2 != 0.3`.
- **`NOT IN` with NULLs** — returns no rows if the subquery contains a NULL. A classic silent-wrong-answer bug.
- **Foreign keys without an index on the child column** — every parent delete scans the child table, and every child insert checks the parent (which is indexed). Missing the child-side index is a common oversight.
- **`ORDER BY random() LIMIT 1`** — sorts the whole table.
- **Migrations that rewrite the table** — `ALTER COLUMN TYPE`, adding a `NOT NULL` with a default on an old Postgres, changing a column's collation.

---

## 7. When to use / When NOT to use

### Use SQL when
- The data is structured and relationships matter (the transcript's framing).
- Transactions are required — money, inventory, bookings, entitlements.
- You need ad-hoc, evolving queries and reporting.
- Constraints must be enforced by the database (uniqueness, referential integrity).
- The dataset fits comfortably on one node (up to a few TB), or you can shard it.
- You need strong consistency and read-your-writes.

### Do NOT use SQL when
- The write volume exceeds a single node and you cannot shard on a good key.
- The schema is genuinely polymorphic and varies per entity (documents, event payloads).
- You need single-digit-millisecond KV access at millions of QPS.
- The data is a graph of deep traversals (friends-of-friends-of-friends).
- You need full-text search relevance ranking over hundreds of millions of documents.
- The workload is analytical scans over billions of rows — use a columnar store.

### Decision table

| Requirement | Answer |
|---|---|
| Money transfer between accounts | **SQL**, single transaction, locks acquired in a consistent order |
| Seat booking | **SQL** with row locks or a conditional update; or Redis holds + a SQL write |
| User profile | **SQL** (small, relational) + a cache |
| Chat message history | Wide-column or SQL partitioned by conversation + time |
| Product catalogue with flexible attributes | SQL with `jsonb` + GIN, or a document store |
| Full-text search over 200M documents | **Search engine** (OpenSearch) fed from SQL by CDC |
| "Total revenue by region by day" | **OLAP** — warehouse or ClickHouse, fed by CDC |
| Session data at 100k writes/sec | **Redis** |
| Feed of posts from people you follow | Precomputed feed in Redis/cache + SQL for the source of truth |
| Event log, 1M writes/sec | Kafka → columnar store; not a row-store OLTP database |
| Recommendation embeddings | **Vector DB** or `pgvector` if it is small ([see Vector DBs](03-vector-databases.md)) |
| Uniqueness of a username across shards | A dedicated uniqueness index shard |
| Reporting on production data without impacting it | A dedicated analytics replica, or a warehouse |
| Geospatial "drivers within 2 km" | PostGIS / S2 / H3, depending on write volume |

---

## 8. Numbers, defaults & tuning

| Quantity | Default / typical | Reasoning |
|---|---|---|
| Postgres page size | 8 KB | The unit of IO and caching |
| InnoDB page size | 16 KB | Same role |
| B-tree height | 3–4 levels for 10⁸ rows | 3–4 page reads per lookup |
| Planner seq-scan threshold | ~5–10% selectivity | Below that an index scan wins |
| `shared_buffers` (Postgres) | 25% of RAM (tuned) | Default 128 MB is far too small |
| `innodb_buffer_pool_size` | 70–80% of RAM | The single biggest MySQL knob |
| `effective_cache_size` | ~50–75% of total RAM | A hint for the planner, not an allocation |
| `work_mem` (Postgres) | 4 MB default; 32–64 MB for analytics | Sorts and hash joins spill beyond it |
| `maintenance_work_mem` | 64 MB default; 1–2 GB for index builds | Speeds up `CREATE INDEX` and `VACUUM` |
| `random_page_cost` | 4.0 default; **1.1 on SSD** | The most common Postgres mis-tuning |
| `default_statistics_target` | 100 | Raise to 500 for skewed important columns |
| Autovacuum scale factor | 0.2 (20%) in older versions; 0.05 modern | How much churn triggers a vacuum |
| `idle_in_transaction_session_timeout` | 0 (unlimited) — **set to 60 s** | Prevents vacuum-blocking sessions |
| `statement_timeout` | 0 — set 30 s OLTP / 5 min analytics | Prevents runaway queries |
| `lock_timeout` | 0 — set 1–5 s before DDL | Prevents the DDL lock queue outage |
| `deadlock_timeout` | 1 s | How long before cycle detection runs |
| `max_connections` (Postgres) | 100 | ~10 MB per backend |
| PgBouncer `default_pool_size` | 20 | Per database/user pair |
| PgBouncer `query_wait_timeout` | 120 s | Cap on pool queue time |
| Pool sizing rule | `cores × 2 + spindles` | Often 50–200, not thousands |
| Index write cost | ~1 extra B-tree insert per index per write | 8 indexes ≈ 3–5× slower inserts |
| `CREATE INDEX CONCURRENTLY` | 2–3× slower, no write lock | Always use it in production |
| Replica lag alert | > 5 s p99 | Normal is < 100 ms |
| Batch `DELETE`/`IN` chunk | 1,000–10,000 rows | Bound lock duration and WAL |
| Bulk-load `ANALYZE` | Immediately after load | Or the first queries get catastrophic plans |
| Txid wraparound warning | `age(datfrozenxid)` > 1.5 B | Postgres refuses writes near 2 B |
| Keyset page size | 20–100 rows | Avoids `OFFSET` scans |
| Money type | `numeric(19,4)` | Never `float` |
| Timestamps | `timestamptz` (UTC) | Avoids DST and ambiguity bugs |

---

## 9. Interview Q&A

**Q1 (easy). Why would you choose SQL over NoSQL?**
When the data is structured, the relationships matter, and transactions are important. The transcript's framing is exactly right: SQL holds ACID properties, and if a transaction is important — a transfer, an order, a booking — a relational database handles it well. If I need to join two tables, enforce a foreign key, or guarantee that a uniqueness constraint holds under concurrency, SQL does that and a NoSQL store usually does not. I would move away from SQL when write volume exceeds one node, when the schema is genuinely polymorphic per record, or when the workload is analytical scans.

**Q2 (easy). Explain ACID.**
Atomicity: the whole transaction completes or none of it does — if Anam sends Priya 500 rupees, either both the debit and the credit happen or neither does, and money cannot vanish. Consistency: the database moves from one valid state to another, with constraints holding. Isolation: concurrent transactions do not interfere with each other. Durability: once committed, the data survives a crash — in practice because the change is written to a write-ahead log and fsynced before the commit returns.

**Q3 (easy). What is an index and what does it cost?**
An index is a separate data structure — usually a B-tree — that maps column values to row locations, so a lookup is logarithmic instead of a full scan. In Postgres it is a few levels deep for hundreds of millions of rows, so a lookup is three or four page reads and usually fewer with the upper levels cached. The cost is on writes: every `INSERT` and `UPDATE` must update every index on the table, and because Postgres uses MVCC, an update writes new entries into every index rather than modifying existing ones. Eight indexes on a write-heavy table can make inserts three to five times slower, so indexing is a read/write trade, not a free win.

**Q4 (medium). What is the leftmost-prefix rule?**
A composite index on `(a, b, c)` can be used for predicates on `a`, on `(a, b)`, and on `(a, b, c)`, but not for `b` alone or `c` alone, because the index is sorted by `a` first. So I order the columns by equality predicates first and the range or sort column last — an index on `(tenant_id, status, created_at)` serves `WHERE tenant_id=? AND status=? ORDER BY created_at` perfectly, supplying both the filter and the ordering, but would not help a query filtering only on `status`, which would need its own index.

**Q5 (medium). Walk me through `EXPLAIN ANALYZE` and what you look for.**
I run it with `ANALYZE` and `BUFFERS` and read it bottom-up, because the inner nodes execute first. The things I look for, in order: a sequential scan on a large table with a selective filter, which means a missing index or a non-sargable predicate; a large gap between estimated and actual rows, which means stale statistics; a nested loop with a large outer row count, which is the N+1-shaped plan; a sort that spilled to disk because `work_mem` is too small; and a large `Rows Removed by Filter`, which means the index is not doing the selective work.

**Q6 (medium). What is the N+1 problem and how do you fix it?**
It is when one query fetches a list and then a separate query runs per item to fetch related data — 100 orders becomes 101 round trips. Almost always it comes from an ORM's lazy loading. I fix it either with an explicit join, or more often with two queries and a join in application code, which avoids both the N+1 and the row explosion a join can cause. In practice you find it by counting queries per request rather than by reading code, so I would enable query logging or an APM tool and look at the query count per endpoint, then use the ORM's eager loading — `select_related` in Django, `includes` in Rails, `joinedload` in SQLAlchemy.

**Q7 (medium). Explain how MVCC works and what it costs.**
Rather than overwriting a row, an update creates a new version of the tuple and marks the old one as deleted as of the updating transaction. Each transaction reads a snapshot, so it sees the versions visible at its start. The benefit is that readers never block writers and writers never block readers — which is why plain `SELECT` takes no locks. The costs are that dead tuples accumulate and must be vacuumed, that tables become bloated if vacuum cannot keep up, that an update is effectively a delete plus an insert so every index gets a new entry, and that Postgres transaction IDs are 32-bit, so a database that goes too long without vacuum hits wraparound and refuses writes.

**Q8 (hard). Why does `SELECT ... FOR UPDATE` cause deadlocks and how do you prevent them?**
A deadlock needs a cycle: transaction A locks row 1 and wants row 2, transaction B locks row 2 and wants row 1. Both wait forever. The database detects the cycle after `deadlock_timeout` — one second in Postgres — and kills one transaction with `deadlock detected`. I prevent it by acquiring locks in a consistent global order, which is why a money transfer between two accounts takes the lower account ID first; by keeping transactions short so lock windows are small; by never holding a transaction open across user interaction or an external API call; and by retrying on deadlock, because it is a normal expected error rather than a bug.

**Q9 (hard). Your primary is at 100% CPU. Walk me through the diagnosis.**
First I separate reads from writes. I check `pg_stat_statements` for the top queries by total time — not by mean time, because a 200 ms query called 50,000 times matters more than a 5 s query called twice. I check `pg_stat_activity` for long-running and `idle in transaction` sessions. If it is reads, I look for missing indexes via `EXPLAIN ANALYZE` on the top queries, then consider a replica for offload and a cache for the hottest reads. If it is writes, I look at the index count, because every index costs an insert; at the WAL/fsync rate; at lock contention; and at whether a batch job or a migration is running. And I check `n_dead_tup` and autovacuum, because a table that has bloated can make every scan far more expensive than it should be.

**Q10 (hard, testing §6). After a bulk data load, one query that took 20 ms now takes 40 s. Why?**
Almost certainly stale statistics. The planner uses `pg_statistic`, and after inserting millions of rows those are out of date, so it may believe the table has 1,000 rows and choose a nested loop instead of a hash join — turning an O(N+M) plan into an O(N×M) one. The tell is in `EXPLAIN ANALYZE`: the estimated row count is orders of magnitude off the actual. The fix is `ANALYZE`, and going forward I would run `ANALYZE` as part of any bulk-load job, and I would consider raising the statistics target on columns with skewed distributions.

**Q11 (hard, testing §6). An `ALTER TABLE` on the orders table caused a site-wide outage. Explain how.**
Because the lock queue is FIFO. The `ALTER TABLE` needs an `ACCESS EXCLUSIVE` lock, and it will not be granted while any other transaction holds a conflicting lock on the table. So if there is one long-running transaction — or worse, one `idle in transaction` session — the `ALTER` waits. And because the queue is first-come-first-served, every subsequent `SELECT` on that table queues *behind* the `ALTER`, so plain reads block. One slow transaction plus one DDL statement takes down all traffic to that table. The fixes are to set `lock_timeout` before any DDL so it fails fast and can be retried, to set `idle_in_transaction_session_timeout` so an abandoned transaction cannot hold a snapshot forever, and for large tables to use an online schema change tool like `gh-ost` in MySQL.

**Q12 (hard, testing §6). Reads are timing out and the database looks healthy. What is happening?**
That is almost certainly connection exhaustion rather than a database problem. If 500 application instances each hold a pool of 20 connections, that is 10,000 connections against a database configured for 500, and each backend costs memory and context switching — so throughput collapses even though no individual query is slow. Or the pattern is a queue: one slow query holds connections, the pool fills, new requests wait, they time out at the client and retry, and the retries consume the pool. The database is fine; the application took itself down. The fixes are PgBouncer in transaction pooling mode, a much smaller per-instance pool — the rule of thumb is roughly twice the core count, which is usually 50 to 200, not thousands — and hard `statement_timeout` and `query_wait_timeout` limits so nothing can hold a connection indefinitely.

**Q13 (hard). How would you handle a counter that gets 10,000 increments per second?**
Not with `UPDATE counters SET n = n + 1 WHERE id = 1`, because every increment takes a row lock on the same row and serializes. I would pick one of three approaches. Shard the counter across N rows and sum on read — twenty rows turns one hot row into twenty. Or batch in the application: accumulate increments in memory and flush every second, which trades a small loss window for a 100× reduction in writes. Or maintain the counter in Redis with `INCR` and flush to the database periodically, which is the standard approach for likes and views, accepting that the counter is approximate and eventually durable. And for the truly write-heavy case, I would make it an append-only event table and aggregate, which is what a streaming pipeline would do.

**Q14 (hard). Compare normalizing and denormalizing.**
Normalizing to third normal form removes duplication and update anomalies, so the write path is correct by construction — that is why the relational model exists. But it puts joins on the read path, and joins over large tables are expensive. Denormalization copies data to make reads cheap: a precomputed `order_total`, a materialized view, or a CQRS read model shaped exactly like the API response. The cost is always the same — an invariant the database used to enforce must now be maintained by application code, a trigger, or a background job, and it can drift when someone writes to the table through a path you forgot about. So my rule is normalize for the write path, denormalize for the read path, and drive the read model from change data capture so every writer keeps it correct.

**Q15 (hard). When would you reach for a materialized view?**
When a dashboard or an aggregate query is expensive over a large table and a few minutes of staleness is acceptable. I create the view, put a unique index on it so I can refresh `CONCURRENTLY` without blocking readers, and refresh on a schedule or on an event. The trade-off I would state explicitly is staleness versus cost: a plain refresh takes an exclusive lock and blocks reads, and `CONCURRENTLY` does not lock but is slower and needs the unique index. If the data must be near-real-time, a materialized view is the wrong tool and I would maintain an incremental counter or a projection instead.

---

## 10. Common mistakes candidates make

1. **Saying "SQL for structured data, NoSQL for unstructured"** and stopping. It is about transactions, relationships, access patterns and scale.
2. **Listing indexes as a free optimisation.** Every index costs writes.
3. **Forgetting the leftmost-prefix rule** when proposing a composite index.
4. **Proposing `CREATE INDEX` without `CONCURRENTLY`** on a live table.
5. **Ignoring the N+1 problem** or claiming joins always fix it.
6. **Not knowing MVCC** — then "readers block writers" and a wrong deadlock story.
7. **Saying a deadlock is a bug.** It is expected and must be retried.
8. **Forgetting connection pooling** in a design with hundreds of app instances.
9. **Running analytics on the OLTP primary.**
10. **Using `OFFSET` pagination** at scale without mentioning keyset pagination.
11. **Saying "just shard it"** before exhausting caching, indexing and partitioning ([see Sharding](../03-scaling/05-sharding-replication-and-consistency.md)).
12. **Not mentioning `lock_timeout`/`statement_timeout`** in the operational story.
13. **Forgetting that a long transaction blocks vacuum** and causes bloat.
14. **Assuming a replica is always safe to read from** — read-your-writes breaks.
15. **Not knowing when the planner will ignore your index** (low selectivity, non-sargable predicates).

---

## 11. Cheat-sheet summary

| Concept | One-line truth |
|---|---|
| Why SQL | Structured data, relationships, complex queries, ACID |
| ACID | Atomicity (all or nothing), Consistency, Isolation, Durability |
| WAL | Sequential append, fsync at commit; fast commits, durable |
| `synchronous_commit = off` | ~200 ms loss window for latency |
| MVCC | Update = new tuple version; readers never block writers |
| MVCC cost | Bloat, vacuum, txid wraparound, index churn on update |
| B-tree | 3–4 levels for 10⁸ rows; leaf chain gives free ranges |
| Leftmost prefix | `(a,b,c)` serves `a`, `(a,b)`, `(a,b,c)` — not `b` alone |
| Column order | Equalities first, range/sort last |
| Covering index | `INCLUDE` → index-only scan, no heap fetch |
| Partial index | `WHERE status='pending'` — small and fast |
| Expression index | Must match the expression exactly (`lower(email)`) |
| Low selectivity | The planner scans instead of using the index (~5–10% rule) |
| Non-sargable | `DATE(col)=`, `col::text=`, `LIKE '%x'` defeat the index |
| Random UUID PK | Page splits, poor locality; use UUIDv7/ULID/Snowflake |
| N+1 | One query per row; fix with eager loading or a batched second query |
| Join algorithms | Nested loop (small outer), hash (large equi), merge (pre-sorted) |
| `EXPLAIN ANALYZE` | Check est vs actual rows, seq scans, disk sorts, rows removed |
| Stale statistics | After a bulk load → catastrophic plans; run `ANALYZE` |
| Deadlock | Lock-order cycles; detect at 1 s; **retry** |
| Serialization failure | Expected under serializable; must retry |
| Lock queue FIFO | One long transaction + one DDL = all reads block |
| `idle_in_transaction` | Blocks vacuum and locks; set the timeout to 60 s |
| Connection pool | `cores × 2 + spindles`, often 50–200; PgBouncer transaction mode |
| Connection exhaustion | App-level outage; pool fills, timeouts, retries, collapse |
| Partitioning | One DB, many tables — pruning, cheap `DROP`, per-partition indexes |
| Sharding | Many DBs — write scale; breaks joins and unique constraints |
| OLTP vs OLAP | Row-oriented vs columnar; never mix them on one node |
| Materialized view | Precomputed; `CONCURRENTLY` to avoid blocking reads |
| Keyset pagination | `WHERE (t,id) < (?,?)` instead of `OFFSET 100000` |
| Hot row counter | Shard the counter, batch, Redis `INCR`, or an event log |
| Money | `numeric`, never `float` |
| Timestamps | `timestamptz`, UTC at rest |

**"If the interviewer says X → answer Y"**

| Interviewer says | You say |
|---|---|
| "SQL or NoSQL?" | "Relationships and transactions → SQL; write scale or polymorphic schema → NoSQL; here is why for this case" |
| "This query is slow" | "`EXPLAIN ANALYZE` first: seq scan, est-vs-actual rows, disk sort, rows removed by filter" |
| "Add an index" | "Which predicate, what selectivity, and what write cost — and `CONCURRENTLY`" |
| "Reads are timing out but the DB looks fine" | "Connection exhaustion — PgBouncer, smaller pools, `statement_timeout`" |
| "Traffic grew 10×" | "Cache, replica, partition, then shard — in that order" |
| "Two users booked the same seat" | "Transaction with row locks or a conditional update; check the isolation level" |
| "The site went down during a migration" | "DDL lock queue behind a long transaction — `lock_timeout` and online schema change" |
| "How do you scale writes?" | "Distributed SQL or sharding; replicas do not help writes" |
| "Reporting is slowing production" | "A dedicated analytics replica or a warehouse fed by CDC" |
| "Why is the table 4× its size?" | "Bloat from MVCC; a long transaction is blocking vacuum" |

---

## 12. Cross-links

- [NoSQL Deep Dive](02-nosql-deep-dive.md) — the alternative when SQL's guarantees are the wrong fit
- [Vector Databases](03-vector-databases.md) — `pgvector` versus dedicated vector stores
- [Blob and Object Storage](04-blob-and-object-storage.md) — where the large binaries live
- [Choosing a Database](05-choosing-a-database-decision-guide.md) — the decision framework
- [Sharding, Replication and Consistency](../03-scaling/05-sharding-replication-and-consistency.md) — replication lag, quorums, sagas
- [Cache Invalidation](../03-scaling/04-cache-invalidation-and-consistency.md) — CDC from the WAL, and the replica race
- [Caching Strategies and Layers](../03-scaling/02-caching-strategies-and-layers.md) — offloading reads before scaling the database
- [BookMyShow Seat Booking](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md) — row locks, double-booking, the payment saga
- [Uber Ride Hailing](../../Case-Studies/HLD/06-uber-ride-hailing.md) — geospatial indexes and hot partitions
- [Scaling, Caching and DB Cheatsheet](../../Cheat-Sheets/03-scaling-caching-db-cheatsheet.md)
