# Scaling, Caching & Databases Cheatsheet

> **TL;DR** — One revision page for the scaling half of a system-design interview. Every number, default, decision rule and failure mode from [load balancers](../HLD/03-scaling/01-load-balancers.md), [caching](../HLD/03-scaling/02-caching-strategies-and-layers.md), [Redis](../HLD/03-scaling/03-redis-deep-dive.md), [invalidation](../HLD/03-scaling/04-cache-invalidation-and-consistency.md), [sharding and replication](../HLD/03-scaling/05-sharding-replication-and-consistency.md), and the four database families. If you have two minutes, read **Decision rules** (§7). If you have one, read the **X → Y** table at the end.

---

## 1. Load balancing

### 1.1 Algorithms

| Algorithm | How it picks | Right for | Fails when |
|---|---|---|---|
| **Round robin** | Next in rotation | Homogeneous servers, uniform requests | Request cost varies, or servers differ |
| **Weighted RR** | Rotation with weights | Heterogeneous capacity; canary shifts | Weights go stale as machine health drifts |
| **Least connections** | Fewest active connections | Variable request cost — **the sensible default** | Connections are long-lived and cheap (WebSocket) |
| **Least response time** | Fewest connections + lowest measured latency | Latency-sensitive | The latency measure is noisy or has feedback |
| **IP hash** | `hash(client_ip)` | Affinity without a shared store | NAT concentrates thousands of users on one backend |
| **Consistent hashing** | Ring with virtual nodes | Cache affinity; cheap membership change | No range behaviour; needs virtual nodes for balance |
| **Power of two choices** | Two random, pick the less loaded | No reliable global state | Requires a load signal; ties break by luck |
| **Random** | Uniform | Stateless, low stakes | High variance at small N |

**Say this:** "Least connections, because request cost varies. Consistent hashing when a backend holds state I cannot move."

### 1.2 L4 versus L7

| | L4 | L7 |
|---|---|---|
| Sees | IP, port, TCP/UDP | Full HTTP |
| Routes by | Address and port | Path, host, header, cookie |
| TLS | Passthrough or none | Terminates and can re-encrypt |
| Speed | Very high, protocol-agnostic | Lower per request, protocol-aware |
| Retries | Cannot | Can, for idempotent requests |
| Health checks | TCP connect only | Application-level |
| Use | Edge absorption, non-HTTP protocols | Routing, policy, rate limits, canaries |

**Production shape:** L4 at the edge for throughput and DDoS absorption, L7 behind it for policy. Or one L7 product with a kernel-bypass fast path.

### 1.3 Health checks

| Kind | Checks | Use | Danger |
|---|---|---|---|
| **Liveness** | The process is alive | Restart me | Restart loops if it depends on a flaky dependency |
| **Readiness** | Can serve traffic (dependencies reachable) | Remove from rotation | **Too deep, and one DB blip empties the pool** |
| **Deep / synthetic** | A real transaction | Catch subtle breakage | Expensive; must not run at request rate |

**The rules:** three consecutive failures to remove, two successes to re-add (hysteresis, to survive GC pauses); a short timeout (1–2 s) so a hung instance is removed fast; and **never let a dependency's health determine the fleet's health** — degrade, do not empty the pool.

### 1.4 Draining and connection reuse

| Concern | Rule |
|---|---|
| Deploy / scale-in | Deregister → wait for in-flight requests → then terminate. 30–60 s drain is typical |
| Keep-alive pinning | Long-lived client connections defeat round robin — cap requests per connection (e.g. 1,000) or connection age (e.g. 5 min) |
| Idle timeouts | **Client timeout < LB timeout < backend timeout**, or you get resets on idle-then-active connections |
| Slow start | A newly added instance should receive gradually increasing traffic, not a full share, while caches warm |

### 1.5 Failure modes

| Failure | Cause | Defence |
|---|---|---|
| Uneven distribution | Round robin + variable cost, or keep-alive pinning | Least connections; cap connection lifetime; check the metric that matches the bottleneck, not CPU |
| Health-check flapping | Aggressive thresholds, GC pauses | Hysteresis, short timeouts, and a shallow readiness check |
| Thundering herd on recovery | All clients reconnect at once after an outage | **Jittered** reconnect backoff; 1.3× the peak capacity as headroom |
| Full-fleet removal | A deep health check on a shared dependency | Dependency-tolerant readiness; circuit-break inside the app |
| LB as SPOF | One instance, one AZ | Multiple instances, multi-AZ, and DNS as a coarse second level |
| Hot backend | Sticky sessions, or a shard-affine workload | Externalise session state; consistent hashing with virtual nodes |
| Cross-AZ cost | Load balancing across AZs | Prefer same-AZ routing, with cross-AZ as failover — cross-AZ traffic is billed |

---

## 2. Caching

### 2.1 The write/read pattern matrix

| Pattern | Read | Write | Best for | Cost |
|---|---|---|---|---|
| **Cache-aside** | Check cache → miss → DB → populate | DB write + **invalidate** | Almost everything. **The default** | First read after a write is slow |
| **Read-through** | The cache loads on a miss | — | Clean app code | The cache needs DB awareness |
| **Write-through** | Cache is always warm | Cache + DB synchronously | Read-latency-critical, small hot set | Writes pay both latencies; the cache is on the write path |
| **Write-behind** | Fast | Cache, flush async | Metrics, counters, telemetry | **Loses writes on a crash** |
| **Refresh-ahead** | Always warm | Background refresh before expiry | A known hot set | Wasted refreshes for keys nobody reads |
| **Write-around** | Miss → DB | DB only, cache bypassed | Write-heavy, rarely re-read | Nothing cached when it is read later |

**Say this:** "Cache-aside with delete-on-write, because a cache failure degrades my latency rather than my correctness."

### 2.2 Cache layers

| Layer | Latency | Size | Hit ratio | Invalidation |
|---|---|---|---|---|
| **Browser** | 0 ms | Small | High for repeat visits | Header-driven; you cannot purge it |
| **CDN / edge** | 10–30 ms | Huge | 80–95% static | Purge (not atomic) or versioned URLs |
| **In-process (application)** | ~0.01 ms | Tiny per instance (MBs) | High for hot keys | TTL only; staleness bounded by the TTL |
| **Distributed (Redis/Memcached)** | 0.5–2 ms | Large | 80–95% | Explicit delete, pub/sub, or CDC |
| **Database buffer pool** | ~0.1 ms | GBs | Very high | Automatic |
| **Materialised view / derived table** | Query-time | Varies | — | Rebuild or CDC |

**The rule:** every layer you add costs an invalidation path. Add a layer only if it removes enough latency or load to justify that, and prefer the *closest* layer — the in-process cache removes the network hop rather than distributing it.

### 2.3 Eviction policies

| Policy | Evicts | Right for | Weakness |
|---|---|---|---|
| **LRU** | Least recently used | General purpose — the default | A sequential scan evicts the hot set |
| **LFU** | Least frequently used | Stable popularity distribution | Stale popularity; a formerly-hot key lingers |
| **W-TinyLFU** | Window + frequency sketch | **Production reality** — Redis's `allkeys-lfu` and Caffeine | More complex to reason about |
| **FIFO** | Oldest inserted | Cheap approximation | Ignores access entirely |
| **TTL** | Nearest expiry | Time-bounded data | Not a workload-aware policy |
| **Random** | At random | Simple; surprisingly decent | No guarantee |
| **Segmented LRU (2Q)** | Probation + protected | Scan-resistant, simple to implement | Two more knobs |

**Redis eviction settings:** `maxmemory` sets the ceiling; `maxmemory-policy` chooses the rule. `noeviction` is the default and returns an error when full — **if you are using Redis as a cache, you almost certainly want `allkeys-lru` or `allkeys-lfu`.** `volatile-*` policies evict only keys with a TTL, which silently stops working if you forget to set TTLs.

### 2.4 Failure modes

| Failure | What happens | Defence |
|---|---|---|
| **Cache stampede / dogpile** | One hot key expires; N requests hit the DB simultaneously | Single-flight (bounded wait), TTL jitter, XFetch, refresh-ahead |
| **Avalanche** | Many keys expire at the same instant (deploy, restart) | **TTL jitter ±10%**; warm the cache before serving |
| **Cache penetration** | Requests for keys that do not exist bypass the cache every time | Negative caching with a short TTL; a bloom filter |
| **Hot key** | One key saturates one shard | In-process cache → key replication with a suffix → replicas → reshaped data model |
| **Stale repopulation race** | A late reader writes an old value after a delete | **Version-stamped writes** (the only closed fix); delayed double-delete as a mitigation |
| **Cache node loss** | 1/N of keys become misses; a hot key lands on one DB row | Replicas per shard; consistent hashing; single-flight at the origin; load shedding |
| **Cache outage** | Every request hits the DB, which was sized for a 90% hit ratio | **Decide in advance**: serve degraded, shed load, or fail closed. Never let the DB absorb it |
| **Big key** | One 10 MB value blocks the Redis event loop | Split into hashes; cap value size (~100 KB); monitor key size |
| **Key explosion** | Unbounded key space grows memory without bound | Prefix + TTL for every key; monitor `DBSIZE` and memory growth |

### 2.5 Sizing arithmetic

```
Working set      = hot records × record size
                 e.g. 10 M active users × 4 KB profile = 40 GB

Hit ratio needed = 1 − (cache capacity / working set)     [roughly, LRU]
                 = 1 − (40 GB / 400 GB) = 90%

Redis memory     = data + overhead (typically ×1.2–1.5 for the allocator and metadata)
                 + replication buffers + a COW fork buffer during RDB save
```

**The number that bites:** a Redis instance doing `BGSAVE` forks, and copy-on-write during the fork can need up to the full dataset size again in free memory. Leave **50% headroom** on a Redis instance that persists.

---

## 3. Redis

### 3.1 Data structures

| Structure | Operations | Use |
|---|---|---|
| String | `GET/SET/INCR/APPEND/SETNX` | Counters, cached blobs, locks, bitmap-like flags |
| Hash | `HSET/HGET/HINCRBY/HDEL` | Objects with individually updated fields; small objects (Redis stores small hashes very compactly) |
| List | `LPUSH/RPOP/LRANGE/LTRIM` | Queues, stacks, bounded recent-item lists |
| Set | `SADD/SISMEMBER/SINTER/SUNION` | Membership, dedupe, "who follows both" |
| Sorted set | `ZADD/ZRANGEBYSCORE/ZRANK` | Leaderboards, rate limiters, priority queues, geo (geohash scores) |
| Stream | `XADD/XREADGROUP/XACK` | An append-only log with consumer groups — a modest Kafka |
| HyperLogLog | `PFADD/PFCOUNT` | Approximate distinct counts, **0.81% error, 12 KB fixed** |
| Bitmap | `SETBIT/GETBIT/BITCOUNT` | Per-user daily flags, DAU, presence |
| Bitfield | `BITFIELD` | Packed counters in one key |

### 3.2 Persistence

| Mode | How | Durability | Cost |
|---|---|---|---|
| **RDB** | Point-in-time snapshot, forked | Lose everything since the last snapshot (default: every 5 min if 100 keys changed) | Fork + COW memory spike; a compact file, fast restart |
| **AOF (everysec)** | Append every write, fsync every second | Lose ≤1 s | Larger files; fsync cost |
| **AOF (always)** | fsync every write | Near-zero loss | **Latency per write** — often a 10× throughput hit |
| **AOF (no)** | Let the OS flush | Up to 30 s | Fast, unsafe |
| **RDB + AOF** | Both | Best of both, at restart cost | Recommended when data matters |
| **None** | Cache only | Total loss on restart, and that is fine | Right for a pure cache |

**AOF rewrite:** `BGREWRITEAOF` compacts the log; it forks, so the COW memory spike applies here too. Configure `auto-aof-rewrite-percentage` and `auto-aof-rewrite-min-size`.

### 3.3 Replication and HA

| Aspect | Detail |
|---|---|
| **Replication** | Asynchronous by default. `WAIT numreplicas timeout` gives a synchronous-ish acknowledgement, but it does not make failover lossless |
| **Failover** | Sentinel (a quorum of Sentinels promotes a replica; typically 3 or 5 Sentinels) or Redis Cluster's gossip |
| **Cluster** | 16,384 hash slots; keys map by CRC16 mod 16384; resharding moves slots |
| **Hash tags** | `{user:1}:a` and `{user:1}:b` land in the same slot — needed for multi-key operations |
| **MULTI/EXEC** | Transactions are queued and executed atomically, **but there is no rollback on a runtime error** |
| **Lua** | Atomic, and can read-modify-write — the correct tool for compare-and-set |
| **Pipelining** | Batching reduces RTT from N round trips to 1 — the cheapest throughput win |
| **Pub/sub** | Fire-and-forget; at-most-once; subscribers that are offline miss messages (`Streams` if you need delivery) |

### 3.4 The consistency truth

**Redis replication is asynchronous. On failover, the last writes acknowledged to clients can disappear.** Everything follows from that:

- A lock can be lost → two holders → **never use a Redis lock as a correctness mechanism**, only as a contention reducer. Correctness needs a fencing token or a compare-and-set in the authoritative store.
- An idempotency key can be lost → a duplicate. Persist idempotency records to a durable store, or accept the duplicate path.
- A rate-limit counter can go backwards → a burst. Acceptable.
- A cache entry can be lost → a miss. Fine.

**The two failure mechanisms worth naming:**
1. **The stale-holder problem** — a client paused (GC, VM migration) past the lock's expiry resumes and writes as if it still holds it. No fencing token, no defence.
2. **The blocked-write problem** — a client holding a lock does a long operation and the lock expires underneath it; every subsequent design should assume this happens.

### 3.5 Failure modes

| Failure | Cause | Defence |
|---|---|---|
| **Hot key** | One key dominates traffic | In-process cache, key replication, shard the logical key |
| **Big key** | One value is MBs | Cap value size; use a hash and read fields; split the key |
| **Slow query blocking the event loop** | `KEYS`, `SMEMBERS` on a huge set, a Lua script over many keys | `SCAN` not `KEYS`; bound every script; put an index on the client side |
| **Memory full, `noeviction`** | Default policy plus no memory plan | Set `maxmemory` and `allkeys-lru` for caches; alert at 80% |
| **Fork stall during BGSAVE** | COW page copying on a large dataset | Keep instances under ~25 GB with 50% headroom; use replicas for saving |
| **Cluster resharding disruption** | Moving many slots at once | Move a few slots at a time; use `MIGRATE` with pipelining |
| **Multi-key op across slots** | No hash tag | Use `{tag}` to co-locate, or restructure |
| **Split brain in Sentinel** | A partitioned old primary still accepting writes | `min-replicas-to-write` plus a quorum; fencing at the application layer |
| **Pub/sub message loss** | Subscriber was reconnecting | Use Streams with consumer groups and `XACK` |

---

## 4. Cache invalidation

### 4.1 The four strategies

| Strategy | Mechanism | Latency of staleness | Use when |
|---|---|---|---|
| **TTL** | Expire | Up to the TTL | A bounded staleness window is acceptable — the cheapest and most robust |
| **Explicit delete on write** | The writer deletes the key | Milliseconds | One write path, strong freshness needed |
| **Versioned keys** | The key contains a version; a write bumps the version | Instant and total | Derived caches, whole-collection invalidations |
| **Event-driven (CDC)** | Consume the WAL/binlog → invalidate | Tens to hundreds of ms | Many writers, or writes outside the app |
| **Pub/sub fan-out** | The writer publishes an invalidation to all app instances | Milliseconds, fan-out | In-process caches that must be invalidated together |

### 4.2 Which to pick

| Situation | Choice |
|---|---|
| One service, one write path | **Delete on write + TTL backstop** |
| Many writers, or DB edits outside the app | **CDC-driven invalidation** |
| Whole-collection changes (a product's price list) | **Versioned keys or surrogate keys** |
| In-process caches across many instances | **Pub/sub, or a very short TTL** |
| The value is derived and expensive | **A derived table maintained by CDC** — explicit and rebuildable |
| The value must be exactly fresh | Do not cache it; or cache with a version check on read |

### 4.3 The stale-repopulation race, drawn

```
R: read DB → V1 ──────────────────────────────► SET cache = V1    ← STALE, forever
W:          DB = V2   DEL cache
                     ↑ the delete lands here
```

| Mitigation | Closes the race? |
|---|---|
| Delayed double-delete (delete, sleep 500 ms, delete) | No — narrows the window |
| Delete **before** the DB write instead of after | No — widens it |
| **Version-stamped cache write (CAS)** | **Yes** — a stale writer's CAS fails |
| CDC-ordered invalidation | Yes, if the stream is ordered per key |
| Disable caching for the value | Yes, trivially |

### 4.4 Failure modes

| Failure | Cause | Defence |
|---|---|---|
| **Missed invalidation** | A write path that forgot to invalidate, or a key the writer does not know about | CDC; a lint rule that every write has an invalidation; versioned keys |
| **Over-invalidation** | Invalidating a broad set on every write | Finer keys; accept a slightly longer TTL |
| **Invalidation storms** | One write invalidating 100,000 keys | Version the collection rather than deleting entries; debounce |
| **Delete-of-delete (key not present)** | Harmless but noisy | Idempotent by design — no action needed |
| **TTL never set** | Unbounded growth and permanent staleness | Enforce a TTL in the cache client, not by convention |
| **Cross-region invalidation lag** | The invalidation is regional | Replicate the invalidation channel; or keep the TTL short enough to bound the divergence |
| **Thundering herd after invalidate** | Every instance misses at once | Single-flight; the invalidation is a *delete*, and the repopulation is collapsed |

---

## 5. Sharding and replication

### 5.1 When to shard — in order

```
1. Index and rewrite the query                    ← almost always the real fix
2. Add a cache
3. Add read replicas
4. Batch and pool writes
5. Partition the table inside one database          ← declarative partitioning
6. Vertical scale the primary
7. SHARD
```

**Triggers that justify step 7:** write throughput saturated on one primary after batching and pooling; the dataset no longer fits one machine; the hot working set no longer fits in memory; or blast-radius/tenant isolation is a requirement.

### 5.2 Shard-key criteria

| Criterion | Good | Bad |
|---|---|---|
| **Cardinality** | `user_id`, `tenant_id` | `country` (200 values) |
| **Access balance** | Uniform traffic per shard | A key where one value is 60% of traffic |
| **Query alignment** | The common query filters on it | The common query is a cross-shard scan |
| **Immutability** | Never changes | `status`, which changes constantly |
| **Not monotonic** | Hashed or high-entropy | `created_at`, auto-increment — all writes to the last shard |
| **Co-location** | Entities read together share it | Entities joined across shards |

### 5.3 Sharding strategies

| Strategy | Distribution | Range scans | Resharding cost |
|---|---|---|---|
| Range | Can be skewed | Single-shard ✅ | Move ranges |
| Hash mod N | Even | Fan-out ❌ | **Moves almost everything** when N changes |
| Consistent hashing | Even with virtual nodes | Fan-out ❌ | Moves ~1/N ✅ |
| Directory / lookup | Flexible | Depends | Just a mapping change ✅ |
| Geographic | By region | Within a region | Per region |
| Composite (tenant, then time) | Good | Within a tenant ✅ | Per tenant |
| **Fixed logical shards (1,024–4,096) mapped to physical nodes** | Good | Depends | **Rebalance a mapping** ✅✅ |

**Say this:** "Fix the number of logical shards high — 1,024 or 4,096 — and map them onto physical nodes. Then resharding is a mapping change, not a data migration."

### 5.4 Cross-shard operations

| Operation | Difficulty |
|---|---|
| Point read by shard key | Trivial |
| Query by a non-shard key | Fan-out to all shards, then merge |
| Aggregation (`COUNT`, `SUM`) | Scatter-gather, or a pre-aggregated rollup table |
| Sort + limit across shards | Each shard returns its top-K, then a merge — **this is not a global ORDER BY unless K is ≥ the global limit** |
| Transaction across shards | Two-phase commit (slow, blocking) or a saga with compensation |
| Unique constraint across shards | Needs a separate uniqueness service or a global index |
| Auto-increment ID across shards | Per-shard offsets, or Snowflake, or a range allocator |
| Join across shards | Do it in the application, or co-locate by denormalising |

### 5.5 Replication topologies

| Topology | Writes | Consistency | Failure |
|---|---|---|---|
| **Single-leader** | One primary | Strong on the primary | Primary is a bottleneck and a failover event |
| **Multi-leader** | Any leader | Conflicts need resolution | Divergent histories; last-write-wins loses data |
| **Leaderless (quorum)** | Any replica | Tunable via W/R | Needs read repair and hinted handoff |
| **Sync** | Primary waits for replicas | No loss on failover | Latency and availability tied to replicas |
| **Semi-sync** | Waits for one of N | One durable copy | Still a latency cost |
| **Async** | No wait | **Lose the replication lag on failover** | The failover loss window = the lag |

### 5.6 Reading replication lag

| Database | Mechanism |
|---|---|
| **Postgres** | `pg_stat_replication` → `write_lag`, `flush_lag`, `replay_lag`; `pg_last_xact_replay_timestamp()` on the replica |
| **MySQL** | `SHOW SLAVE STATUS` → `Seconds_Behind_Master` (**frequently misleading**), `Relay_Log_Pos` |
| **The reliable way** | Write a heartbeat timestamp on the primary every second and read the newest on the replica |

**Acting on lag:** route read-your-writes traffic to the primary for 1–5 s after a write; shed non-critical reads; find the cause (a long query on the replica, a large write transaction, single-threaded replay, a network bottleneck).

### 5.7 Consistency ladder

```
Eventual ── consistent prefix ── monotonic reads ── read-your-writes
        ── monotonic writes ── LINEARIZABLE
```

| Level | Guarantee | Cost |
|---|---|---|
| Eventual | Replicas converge | Cheapest |
| Consistent prefix | Never see an effect before its cause | Cheap |
| Monotonic reads | Never go backwards in time | Session-pinned replica |
| Read-your-writes | You see your own writes | Route to primary briefly |
| Monotonic writes | Your writes apply in issue order | Session affinity |
| Linearizable | A single global order, atomically | Coordination on every operation — the most expensive |

**The point to make:** these are **per-session** guarantees, and they cover almost every real user expectation at a fraction of the cost of global linearizability.

### 5.8 CAP and PACELC

| System shape | CAP under partition | PACELC else-branch |
|---|---|---|
| Single-leader RDBMS | **C** — refuse writes on the minority side | **E→C** — synchronous replication costs latency |
| Dynamo / Cassandra | **A** — accept writes, reconcile later | **E→L** — quorum tuning trades latency for consistency |
| Redis (async replication) | **A** — the replica promotes, losing the tail | **E→L** — the default is latency over durability |
| Spanner / CockroachDB | **C** — quorum, minority unavailable | **E→C** — pays latency for linearizability |
| A cache | **A** — it is derived data | **E→L** — always |

**Say this:** "CAP is per-operation, not per-system. In one ride-hailing system: driver locations are PA/EL — available and lossy; the trip state machine is PC/EC — consistent, and I pay the latency for it."

### 5.9 Failure modes

| Failure | Symptom | Defence |
|---|---|---|
| **Hot shard** | One shard at 100%, others idle | Isolate the entity; split the key; cache the read path; reshard last |
| **Split brain** | Two primaries accepting writes | Quorum-based election **plus fencing tokens**; `min-replicas-to-write` |
| **Failover data loss** | Committed writes vanish | The loss window = the replication lag — monitor it; use semi-sync for money |
| **Resharding corruption** | Different results from old and new | The **shadow-read comparison** phase; explicit type/collation checks |
| **Cross-shard fan-out** | A query hits every shard | Pre-aggregated rollups; a secondary index store; change the shard key |
| **Sequential scan across shards** | Latency = the slowest shard | Bound per-shard timeouts; hedge requests to a second replica |
| **Replica lag spike** | Users see stale data | Read-your-writes routing; alert on lag as a leading indicator |
| **Cascading replica failure** | Replicas replicating from replicas amplify lag | Replicate from the primary, or from a small number of intermediates |

---

## 6. The four database families

### 6.1 Side-by-side

| | **Relational (SQL)** | **Key-value** | **Wide-column** | **Document** | **Graph** | **Columnar / OLAP** | **Vector** |
|---|---|---|---|---|---|---|---|
| Model | Tables, rows, relations | Key → opaque value | Rows keyed by partition + clustering | JSON documents | Nodes + edges | Columns | Embeddings + metadata |
| Query | SQL, joins, ad hoc | `GET`/`PUT` | CQL, range within partition | Document queries, indexes | Traversals | SQL over columns | ANN top-k |
| Consistency | Strong (single node) | Tunable | Tunable per query | Tunable | Varies | Strong | Per-index |
| Scale | Vertical + read replicas | Horizontal, excellent | Horizontal, excellent | Horizontal (sharded) | Hard to shard | Horizontal for reads | Horizontal with sharding |
| Joins | Native | None | None | Limited | Native (traversal) | Limited | None |
| Best at | Transactions, flexible queries | Massive simple lookups | Time series, high write volume | Flexible schemas | Deep relationships | Aggregations over billions of rows | Similarity search |
| Examples | Postgres, MySQL, Aurora | Redis, DynamoDB, Memcached | Cassandra, HBase, Bigtable | MongoDB, Couchbase, Firestore | Neo4j, Neptune | ClickHouse, BigQuery, Redshift, DuckDB | Pinecone, Qdrant, pgvector, Milvus, Weaviate |
| The catch | Sharding is manual and painful | No queries beyond the key | Query-first modelling; no ad hoc | Weak multi-document transactions | Sharding traversals is genuinely hard | Not for OLTP writes | Memory-bound; filtering is hard |

### 6.2 Query-first modelling — the one rule

**Write down the queries before choosing the store.** For a key-value or wide-column store this is not advice, it is the design: you cannot add a query later without a new table and a backfill. The concrete questions:

1. What are the reads, exactly (not the entities)?
2. Which reads are on the hot path, and what is the volume?
3. What consistency does each read and write need?
4. What is the write volume and shape (point writes or bulk)?
5. How stable is the query set — will it change next quarter?
6. How much operational burden can the team carry (a single Postgres is not the same operational commitment as a Cassandra ring)?

### 6.3 Indexing

| Index | Best for | Cost |
|---|---|---|
| **B-tree** | Equality and range on any sortable type | Write amplification on every modification |
| **Hash** | Equality only | No ranges at all |
| **GIN** | JSONB, arrays, full text — "contains these elements" | Large and slow to update |
| **GiST** | Geometry, ranges, nearest-neighbour | Approximate |
| **BRIN** | Very large tables with natural clustering (append-only time series) | Tiny index; only useful when the data is physically ordered |
| **Partial** | A hot subset (`WHERE status = 'open'`) | Only helps if the query matches the predicate |
| **Covering / INCLUDE** | Return the row without touching the table | Bigger index |
| **Composite** | Multi-column predicates | Leftmost-prefix rule applies |
| **Expression** | Queries the base index cannot serve (`LOWER(email)`) | Must match the expression exactly |

**Rules that carry the most weight:**

- Composite index column order follows the queries: **equality columns first, then the range column, then covering columns.**
- **Leftmost prefix:** `(a, b, c)` serves `a`, `a,b`, `a,b,c` — not `b`, and not `c`.
- **Selectivity:** an index on a column matching >5–10% of rows rarely wins against a scan.
- **Non-sargable:** a function, a cast, a leading wildcard, or arithmetic on the column discards the index. `WHERE YEAR(ts) = 2026` scans; `WHERE ts >= '2026-01-01' AND ts < '2027-01-01'` seeks.
- **Every index costs writes.** Six indexes on a write-heavy table is a self-inflicted wound.
- **Index the foreign key** — many databases do not do it for you, and the cascade delete or the join will scan.

### 6.4 Transactions and isolation

| Level | Dirty read | Non-repeatable read | Phantom read |
|---|---|---|---|
| Read uncommitted | Possible | Possible | Possible |
| Read committed | No | Possible | Possible |
| Repeatable read | No | No | **Possible in the standard; MySQL's default prevents most** |
| Serializable | No | No | No |

**Postgres defaults to read committed; MySQL/InnoDB defaults to repeatable read.** The traps: a long-running repeatable-read transaction holds a snapshot, causing bloat and vacuum pressure; `SELECT FOR UPDATE` locks rows and a forgotten `COMMIT` blocks everything behind it; and a lost update (read-modify-write without a lock or a conditional update) is the bug that atomic `UPDATE ... SET x = x + 1 WHERE ...` prevents.

**Optimistic vs pessimistic:** optimistic (a version column, retry on conflict) wins when conflicts are rare; pessimistic (`SELECT FOR UPDATE`) wins when they are frequent and retries are expensive. In a high-contention system, the third option — a **conditional atomic update** — beats both, because there is no transaction to retry.

### 6.5 Distributed IDs

| Scheme | Size | Sortable | Coordination | Trap |
|---|---|---|---|---|
| UUIDv4 | 128-bit | No | None | Random B-tree inserts → page splits and bloat |
| UUIDv7 / ULID | 128-bit | **Yes** | None | Leaks creation time; 2× a bigint |
| Snowflake | 64-bit | Yes | Machine ID assignment | **Clock skew produces duplicates** — refuse to issue until the clock catches up |
| DB sequence | 64-bit | Yes | One database | A bottleneck and a cross-shard coordination point |
| Range/segment allocator | 64-bit | Yes | A coordinator per block | IDs jump across machines; a coordinator dependency |
| Per-shard offset | 64-bit | Yes | Configured, not coordinated | Fixed capacity per shard; requires a shard count up front |

**Say this:** "Snowflake for a sortable 64-bit ID, with a logical clock or a refuse-to-issue rule on clock skew. UUIDv7 when I want zero coordination and index locality."

### 6.6 Queues versus logs

| | **Traditional queue** (SQS, RabbitMQ) | **Log** (Kafka, Kinesis, Pulsar) |
|---|---|---|
| On ack | The message is deleted | The offset advances; the message is retained |
| Ordering | Per message group at best | **Per partition**, guaranteed |
| Replay | No | Yes — seek to any offset |
| Consumers | Competing consumers, one message each | Consumer groups; multiple groups each read everything |
| Retention | Until consumed (or a TTL) | Time or size based, independent of consumption |
| Throughput | High | Very high, sequential I/O |
| Best for | Task distribution, work queues | Event streaming, event sourcing, fan-out, replay |
| The catch | No replay; the DLQ is where messages go to die | **The partition key decides what is ordered together** — a design decision, not a config detail |

**Reliability rules that apply to both:** ack **after** processing, never before; consumers must be **idempotent**, because at-least-once delivery means duplicates are certain; a **DLQ** after bounded attempts, with alerting and a replay tool; **exponential backoff with jitter**; and the **poison pill** — one message that always fails — must be routed to the DLQ rather than retried forever, blocking the partition.

### 6.7 Resilience patterns

| Pattern | What it does | Without it |
|---|---|---|
| **Timeout** | Bounds every network call | An unbounded resource commitment; one slow dependency exhausts the pool |
| **Retry with backoff + jitter** | Recovers from transient failures | A synchronised retry storm turns a blip into an outage |
| **Circuit breaker** | Fails fast after a failure threshold; half-open probes | The caller's threads all block on a dead dependency, and the caller dies too |
| **Bulkhead** | Separate pools per dependency | One slow dependency starves all the others |
| **Load shedding** | Drop low-priority work | Everything degrades, including the critical path |
| **Idempotency key** | Makes retries safe | A retried payment charges twice |
| **Dead-letter queue** | Contains the unprocessable | A poison pill blocks the queue forever |
| **Backpressure** | The consumer signals the producer to slow | The queue grows without bound until memory is gone |
| **Hedged request** | Send a second copy after p99; take the first | Tail latency is the tail |

**Circuit breaker defaults worth naming:** open after ~50% failures over a minimum of ~20 calls in a window; open for ~30 s; then a few half-open probes; close again after ~5 consecutive successes.

### 6.8 Failure modes by family

| Family | Characteristic failure | Defence |
|---|---|---|
| Relational | A hot row, a long transaction, lock contention, bloat | Batch updates, short transactions, `SKIP LOCKED` queues, autovacuum tuning |
| Key-value | Hot partition, unbounded growth, no query flexibility | Bucket suffixes, TTLs, denormalised secondary tables |
| Wide-column | Large partitions, tombstone storms, repair pressure | Cap the partition size; avoid delete-heavy workloads; tune compaction |
| Document | Unbounded document growth, weak multi-doc transactions | Cap the document; model the transaction boundary as the document |
| Graph | Supernodes (a node with millions of edges) | Cap the fan-out; split the supernode; precompute |
| Columnar | Too many small parts, unbounded cardinality in the sort key | Batch inserts; tune the merge; choose the sort key by the filter |
| Vector | Memory exhaustion, filter-induced recall collapse | Quantisation, DiskANN, make the filter a partition key |

---

## 7. Decision rules

| Question | Answer |
|---|---|
| Which LB algorithm? | Least connections; consistent hashing when there is state to preserve |
| Shard or not? | Not until index → cache → replicas → partitioning → vertical scale are exhausted |
| Which shard key? | High cardinality, even **access**, query-aligned, immutable, not monotonic |
| How many shards? | Fix 1,024–4,096 logical, map to physical |
| Cache or not? | Only if the hit ratio exceeds ~80% and staleness is tolerable |
| Which cache pattern? | Cache-aside with delete-on-write and a TTL backstop |
| Delete or update on write? | Delete — it is idempotent; update loses under concurrency |
| How to close the stale race? | Version-stamped cache writes; everything else is mitigation |
| TTL length? | The longest staleness the product tolerates, with ±10% jitter |
| How to handle a stampede? | Single-flight with a bounded wait, plus jitter, plus XFetch for hot keys |
| How to handle a hot key? | In-process cache first; then key replication; then shard the logical value |
| Redis persistence? | `no` if it is a cache; `AOF everysec` or RDB+AOF if data matters |
| Redis as a lock? | Only as an optimisation — correctness needs fencing or a DB CAS |
| Redis eviction policy? | `allkeys-lru` or `allkeys-lfu` for a cache, never the `noeviction` default |
| SQL or NoSQL? | Query-first. Postgres is the default and is usually right |
| Which index? | B-tree by default; GIN for JSON/arrays/text; BRIN for clustered append-only |
| Index column order? | Equality columns, then range, then covering |
| How to find a slow query? | Compare plans: the same plan means data/resources, a new plan means statistics/threshold |
| Reshard with no downtime? | Dual-write → backfill → **shadow-read and verify** → cut over per range |
| Split brain defence? | Quorum election **plus** fencing tokens |
| Queue or log? | Log when you need replay, ordering per key, or multiple independent consumers |
| Sync or async replication? | Async for reads, semi-sync or sync for money |
| Cache down: fail open or closed? | Open for public reads, closed for auth and payments — decide in advance |
| Retry a payment? | Only with an idempotency key — a timeout may mean it already succeeded |

---

## 8. Numbers to have memorised

| Thing | Number |
|---|---|
| Availability: 99.9% / 99.99% | 43 min / 4.3 min per month |
| Peak-to-average multiplier | ×2 to ×5 |
| B-tree depth at 10⁸ rows | 3–4 levels |
| Index useful above this selectivity | Below 5–10% of rows |
| Target cache hit ratio | 80%+ |
| TTL jitter | ±10% |
| Single-flight bounded wait | 50–100 ms, then fall through |
| XFetch beta | 1.0 |
| Negative-cache TTL | 60–300 s (or the SOA minimum) |
| In-process cache TTL | 1–10 s |
| Read-your-writes window | 1–5 s on the primary |
| Redis `maxmemory` headroom for persistence | ~50% free (COW during fork) |
| Redis instance size before fork becomes painful | ~25 GB |
| Redis cluster hash slots | 16,384 (CRC16) |
| Redis Sentinel quorum | 3 or 5 Sentinels |
| Redis key-size guidance | Keep values under ~100 KB |
| Postgres `synchronous_commit = off` loss window | ~200 ms |
| MySQL semi-sync fallback timeout | 10,000 ms |
| Quorum | `W + R > N`; N=3, W=2, R=2 |
| Replication lag alert threshold | 1 s warning, 5 s page |
| Logical shards | 1,024–4,096 |
| Consistent-hashing virtual nodes | 100–1,000 per node |
| DynamoDB partition ceiling | ~1,000 WCU / 3,000 RCU / 10 GB |
| Cassandra partition guidance | < 100 MB, < 100,000 rows |
| Wide-column tombstone grace period | 10 days default (`gc_grace_seconds`) |
| Snowflake ID | 41-bit ts + 10-bit machine + 12-bit sequence; ~4 M/sec/machine |
| Vector storage | 1,536 dims × 4 B = 6 KB/vector |
| HNSW index RAM overhead | ×1.5–2.0 the raw vectors |
| Vector `efSearch` start | 10 × k |
| Kafka default retention | 7 days (`retention.ms=604800000`) |
| Kafka consumer default `max.poll.interval.ms` | 300,000 ms |
| Idempotency key TTL | 24 h |
| Circuit breaker open threshold | ~50% failures over 20+ calls, ~30 s open |
| Timeout defaults worth setting | DB query 1–5 s; internal RPC 1–2 s; external API 2–5 s |
| Connection pool size rule | Roughly `2 × cores + effective_spindle_count`, then measure — oversized pools are worse than small ones |
| DLQ after N attempts | 3–5, then alert and replay manually |
| Retry backoff with jitter | Base 100–500 ms, exponential, ±20% jitter, 3–5 attempts |

---

## 9. Cheat-sheet summary

| Topic | One-line truth |
|---|---|
| LB default | Least connections; consistent hashing when there is state |
| L4 vs L7 | Packets versus HTTP; L4 at the edge, L7 for policy |
| Health checks | Shallow liveness, dependency-tolerant readiness, hysteresis |
| Draining | Deregister, wait 30–60 s, then terminate |
| Idle timeouts | Client < LB < backend, or you get resets |
| Cache-aside | The default; a cache failure degrades latency, not correctness |
| Delete-on-write | Delete, never update — it is idempotent and order-independent |
| Stale repopulation | Only a version-stamped cache write closes the race |
| Stampede | Single-flight + TTL jitter + XFetch + refresh-ahead |
| Avalanche | TTL jitter; warm before serving |
| Penetration | Negative caching; bloom filter |
| Hot key | In-process cache first, then replication, then reshape the data |
| Eviction | LRU by default; LFU/W-TinyLFU for stable popularity; **never `noeviction` for a cache** |
| Cache outage | Decide fail-open or fail-closed before the incident |
| Redis persistence | `no` for a cache; `AOF everysec` when data matters; a fork needs COW headroom |
| Redis locks | A contention reducer, never a correctness mechanism |
| Redis async replication | Failover loses the tail — locks, counters, idempotency records included |
| Redis slow ops | `SCAN` not `KEYS`; bound every Lua script; cap value size |
| Invalidation | TTL is cheapest; delete-on-write is freshest; CDC catches unknown writers |
| Resharding | Dual-write → backfill → shadow-read → cut over per range |
| Shard key | Cardinality, access balance, query alignment, immutability, not monotonic |
| Cross-shard limits | Joins, global sort, unique constraints and transactions are all hard |
| Replication | Async loses the lag on failover; sync costs latency and availability |
| Split brain | Quorum election plus fencing tokens |
| Consistency | Session guarantees are cheap; linearizability is not |
| CAP | Per operation; PACELC's else-branch is the latency trade |
| SQL vs NoSQL | Query-first; Postgres is the default |
| Index | Equality then range then covering; leftmost prefix; selectivity > 5–10% |
| Non-sargable | Functions, casts and leading wildcards discard the index |
| Isolation | Postgres read committed, MySQL repeatable read; know your default |
| Lost update | `UPDATE ... SET x = x + 1` beats read-modify-write |
| Distributed IDs | Snowflake for sortable 64-bit; UUIDv7 over UUIDv4 for locality |
| Queue vs log | Log when you need replay and per-key ordering |
| Reliability | Ack after processing; idempotent consumers; DLQ; jittered retries |
| Circuit breaker | Timeouts + bulkheads + bounded retries + load shedding, or it cannot save you |

**"If the interviewer says X → answer Y"**

| Interviewer says | You say |
|---|---|
| "Round robin across the fleet" | "Only if requests are uniform — least connections, and cap connection lifetime so keep-alive does not pin" |
| "One instance reports unhealthy" | "Three failures with hysteresis, and the readiness check must not depend on a shared dependency" |
| "Traffic is uneven" | "Check keep-alive pinning first, then whether the metric I am looking at matches the bottleneck" |
| "Add a cache" | "Cache-aside with delete-on-write, a TTL backstop, and versioned entries" |
| "The cache is stale" | "Missed invalidation, the repopulation race, or a TTL too long for the change rate?" |
| "All keys expired at once" | "No TTL jitter — a restart or a deploy synchronised them" |
| "The DB fell over when Redis died" | "The DB was sized for the cache's hit ratio. Fail-open with load shedding, or serve degraded" |
| "One key is hot" | "In-process cache first — it removes the hop instead of spreading it" |
| "Use Redis for the lock" | "Only as an optimisation. Async replication loses the lock on failover — I need fencing or a DB CAS" |
| "Shard it" | "Only after index, cache, replicas, partitioning and vertical scale. And the key is chosen by access pattern" |
| "Which shard key?" | "High cardinality, even access, query-aligned, immutable, not monotonic" |
| "How many shards?" | "Fix 1,024–4,096 logical and map them to physical nodes, so resharding is a mapping change" |
| "Reshard with no downtime" | "Dual-write, backfill, shadow-read and verify, cut over per range" |
| "A shard is hot" | "Isolate the entity, split the key, or cache the read path — a reshard is last" |
| "Two primaries after the partition" | "Quorum election plus fencing tokens, so the deposed primary cannot write" |
| "We lost writes on failover" | "Async replication — the loss window is exactly the replication lag" |
| "CAP?" | "Per operation. PACELC's else-branch is the latency-versus-consistency trade in normal operation" |
| "Consistency or availability?" | "For locations, availability; for the trip state machine, consistency. Say which operation" |
| "SQL or NoSQL?" | "Query-first. Postgres is the default; NoSQL when the access pattern is the product" |
| "Mongo will scale it" | "Only if the document is the transaction boundary and the access path. Otherwise I fight the model" |
| "The query got slow overnight" | "Compare plans: same plan means data or resources, a changed plan means statistics or a size threshold" |
| "Add an index" | "Which query, which column order — equality then range — and what does it cost on the write path?" |
| "Why is the index not used?" | "Non-sargable: a function, a cast, or a leading wildcard" |
| "Analytics on the primary" | "It evicts the hot working set — CDC to a columnar store instead" |
| "Use UUIDs as the primary key" | "UUIDv4 destroys B-tree locality. UUIDv7 or ULID if I want no coordination" |
| "Use a queue" | "Queue for task distribution; a log if I need replay, per-key ordering, or multiple consumer groups" |
| "The consumer processed it twice" | "At-least-once delivery — the consumer must be idempotent, keyed on the message's dedupe key" |
| "One message keeps failing" | "It is a poison pill — bounded attempts then the DLQ, not an infinite retry loop" |
| "Retry the payment" | "Only with an idempotency key — a timeout may mean it already succeeded" |
| "The downstream is slow so we are slow" | "No timeout, no bulkhead, no breaker — add all three, and shed low-priority load" |
| "Trade-offs?" | "Consistency, latency, cost, operational complexity — name which one this choice spends" |

---

## 10. Cross-links

- [Load Balancers](../HLD/03-scaling/01-load-balancers.md)
- [Caching Strategies and Layers](../HLD/03-scaling/02-caching-strategies-and-layers.md)
- [Redis Deep Dive](../HLD/03-scaling/03-redis-deep-dive.md)
- [Cache Invalidation and Consistency](../HLD/03-scaling/04-cache-invalidation-and-consistency.md)
- [Sharding, Replication and Consistency](../HLD/03-scaling/05-sharding-replication-and-consistency.md)
- [SQL Deep Dive](../HLD/04-data/01-sql-deep-dive.md)
- [NoSQL Deep Dive](../HLD/04-data/02-nosql-deep-dive.md)
- [Vector Databases](../HLD/04-data/03-vector-databases.md)
- [Blob and Object Storage](../HLD/04-data/04-blob-and-object-storage.md)
- [Choosing a Database](../HLD/04-data/05-choosing-a-database-decision-guide.md)
- [Scaling, Caching and DB Questions](../Interview-Questions/03-scaling-caching-db-questions.md)
- [Foundations and Estimation Cheatsheet](01-foundations-and-estimation-cheatsheet.md)
- [Networking Cheatsheet](02-networking-cheatsheet.md)
