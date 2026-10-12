# Redis Deep Dive — Data Structures, Persistence, Cluster, Locks and Hot Keys

> **TL;DR** — Redis is a single-threaded, in-memory data-structure server. That one sentence explains almost everything about it: commands run one at a time so every command is atomic and there are no race conditions inside a single command, but one slow command (`KEYS *`, a 10M-element `DEL`, a big `ZRANGE`) blocks every other client. You use it as a cache, a session store, a rate limiter, a leaderboard, a queue, a distributed lock, and a semantic cache for LLMs — because its data structures do the work server-side. You do **not** use it as the source of truth, because memory is expensive, persistence is not fsync-by-default, and the failure modes (fork/COW stalls, replication lag, failover data loss, big keys) are all consequences of the single-threaded, memory-bound design.

> **60-second interview answer:** "Redis is an in-memory data-structure server with a single-threaded event loop, which is why every individual command is atomic and why one slow command blocks everything. It gives me strings, hashes, lists, sets, sorted sets, bitmaps, HyperLogLog, streams and geospatial indexes, plus Lua scripting for multi-step atomic logic — which is how I build a distributed token-bucket rate limiter. For durability I use AOF with `everysec` if I care about at most a second of loss, or RDB snapshots if I accept more; neither makes it a database. For scale I use Redis Cluster with 16,384 hash slots and hash tags when a script needs multiple keys in one slot. For correctness I am careful about the operational failure modes: a fork for RDB can stall the event loop, `DEL` on a big key blocks, failover with async replication loses writes, and a single hot key or big key can take down the whole instance."

---

## 1. Definition & mental model

**Plain English.** Redis keeps data in RAM and gives you server-side data structures to operate on it. It is not "a database that happens to be fast"; it is a set of concurrent-safe data structures that you reach over a network.

**The analogy.** The transcript's fridge: your butter is in the fridge. Without it, every pat of butter means a trip to the grocery store. The fridge is fast, close, and small; the store is slow, far, and holds everything. You do not build a house out of what fits in your fridge.

A better analogy for the *engineering* properties: **Redis is a single checkout lane in a shop where the cashier can do anything you ask, one customer at a time.**

- Because there is one cashier, two customers can never both take the last item — every operation is atomic. That is why `INCR` is a correct rate limiter and `SET NX` is a correct lock.
- Because there is one lane, if one customer asks the cashier to count 10 million items, everyone behind them waits. That is why `KEYS *` in production is a self-inflicted outage.
- You can hand the cashier a scripted list of instructions (Lua) and they will do the whole list without serving anyone else in between — atomic multi-step logic.
- The shop is small, so you keep the popular items near the front (eviction policy) and the full inventory in the warehouse (the database).

**What Redis is not:** a general-purpose database. The transcript's framing — *"it is temporary storage; it uses memory itself as the cache"* — captures the default use case: session management, rate limiting, leaderboards, RAG caches. The correct mental model is **"authoritative for nothing, fast for everything."**

---

## 2. Why it exists — the problem it solves

1. **A disk-backed database cannot serve hot reads at 100k+ QPS.** Redis answers a `GET` in 0.2–1 ms with a single-threaded loop and no query planner, no lock manager, no WAL fsync per statement.
2. **Shared state across stateless app instances.** Sessions, rate-limit counters, feature flags, distributed locks, presence — all need one consistent view across N instances, and in-process memory cannot provide it.
3. **Server-side data structures eliminate round trips and races.** A leaderboard is one `ZADD` and one `ZREVRANGE`, not "read the top 100, sort them in the app, write back". A "get or set with expiry" is one atomic command.
4. **Atomicity without transactions.** `INCR`, `SETNX`, `ZADD`, `HINCRBY` are all atomic on their own; Lua gives atomicity across several. This removes whole classes of distributed-systems bugs.
5. **Pub/sub and Streams** give lightweight messaging without a Kafka cluster.
6. **Expiry as a first-class feature.** `EXPIRE` everywhere makes cache semantics trivial.
7. **Very high throughput per node with low operational complexity** compared to a database cluster.

### What breaks without it

- Every session lookup is a database query.
- Rate limiting becomes either per-instance (wrong) or a database write per request (expensive).
- Leaderboards, counters and presence require expensive aggregate queries or fragile in-memory state.
- Distributed locks have no reliable primitive.
- Cache invalidation between instances has no fast channel.

---

## 3. Internal mechanics — how it actually works

### 3.1 Data structures

| Type | Commands | Internal encoding | Use cases | Notes |
|---|---|---|---|---|
| **String** | `GET` `SET` `INCR` `DECR` `SETEX` `SETNX` `GETRANGE` `APPEND` `MSET` | `int`, `embstr` (≤44 bytes), `raw`, `sds` | Cache values, counters, locks, JSON blobs, bitmaps (via `SETBIT`) | Max value 512 MB — never approach it |
| **Hash** | `HSET` `HGET` `HINCRBY` `HGETALL` `HMGET` `HRANDFIELD` | `listpack` (small) → `hashtable` | Objects with fields (a user record), counters per field | Field-level atomic updates; `HGETALL` on a huge hash blocks |
| **List** | `LPUSH` `RPUSH` `LPOP` `BRPOP` `LRANGE` `LINSERT` `LTRIM` | `listpack` → `quicklist` | Simple queues, activity feeds, recent-N buffered with `LTRIM` | `BRPOP` gives blocking queue semantics; `LRANGE 0 -1` on a big list is a footgun |
| **Set** | `SADD` `SREM` `SISMEMBER` `SINTER` `SUNION` `SCARD` `SRANDMEMBER` | `intset` → `listpack` → `hashtable` | Tags, followers, unique visitors, set algebra (mutual friends = `SINTER`) | `SMEMBERS` on a large set blocks; use `SSCAN` |
| **Sorted set (ZSet)** | `ZADD` `ZINCRBY` `ZRANGE` `ZRANGEBYSCORE` `ZREVRANGE` `ZRANK` `ZREMRANGEBYSCORE` | `listpack` → `skiplist` + hashtable | **Leaderboards**, rate-limit sliding logs, priority queues, time-ordered indexes, geo (via geohash score) | Skiplist gives O(log N) insert and range; the workhorse structure |
| **Bitmap** | `SETBIT` `GETBIT` `BITCOUNT` `BITOP` `BITPOS` | String | Daily active users (1 bit per user per day), feature flags per user, presence | 1M users = 125 KB per day; `BITCOUNT` for the DAU count |
| **HyperLogLog** | `PFADD` `PFCOUNT` `PFMERGE` | Sparse/dense HLL | Approximate unique counts (unique visitors, distinct queries) | **0.81% standard error**, fixed 12 KB max; cannot list members or remove one |
| **Stream** | `XADD` `XREAD` `XREADGROUP` `XACK` `XPENDING` `XCLAIM` | Radix tree | Append-only event log with consumer groups, replay, and ack | Redis's answer to Kafka-lite; capped with `MAXLEN ~ N` |
| **Geospatial** | `GEOADD` `GEOSEARCH` `GEODIST` | ZSet with geohash scores | "Drivers near me", store locators | Radius queries with `ASC` ordering; 52-bit geohash precision |
| **Bitfield** | `BITFIELD` | String | Compact counters, packed state | Arbitrary-width integer fields |
| **JSON** | `JSON.SET` `JSON.GET` `JSON.ARRAPPEND` (RedisJSON module) | Module | Document storage with path access | Not in stock Redis |
| **Vector** | `FT.CREATE` `FT.SEARCH` with `KNN` (RediSearch module) | Module (HNSW / FLAT) | Semantic cache, small-scale RAG retrieval | Ties Redis to a vector-DB role |

**Sizing intuition:** a Redis string has roughly 50–100 bytes of overhead per key (key SDS, dict entry, robj, expiry entry, allocator overhead). So 10M keys of 100 bytes is not 1 GB — it is closer to 2–3 GB. Always measure with `MEMORY USAGE` and `INFO memory`, never estimate from the value size alone.

### 3.2 The single-threaded event loop

```
      ┌─────────────── Redis process ───────────────┐
      │  epoll/select event loop (1 thread)         │
      │   ├── accept connection                     │
      │   ├── read command from socket              │
      │   ├── parse command                         │
      │   ├── execute command  ←───────────────────┐│
      │   ├── append to AOF buffer / repl backlog  ││
      │   └── write reply to socket                ││
      │                                            ││
      │  background threads: lazy free,             ││
      │  fsync (AOF everysec), close_file           ││
      │  IO threads (Redis 6+, offloads read/write) ┘│
      └─────────────────────────────────────────────┘
```

**Consequences that matter in production:**

| Property | Implication |
|---|---|
| One command at a time | `INCR`, `SETNX`, `SADD` are atomic — no application-level locking needed for single commands |
| No query planner or per-query parallelism | One instance cannot use more than one core for command execution |
| One slow command blocks everything | `KEYS *`, `SMEMBERS` on 1M elements, `DEL` on a huge key, `SORT` on a big list, an expensive Lua script |
| O(N) commands must be avoided on big collections | Use `SCAN`/`HSCAN`/`SSCAN`, `UNLINK`, `ZREMRANGEBYSCORE` in bounded chunks |
| Pipelining is a big win | 100 commands in one round trip instead of 100 round trips |
| Memory is the ceiling | No disk overflow for the working set (though `maxmemory` eviction or swap will silently destroy performance) |
| CPU-bound on TLS | TLS encryption happens on the main thread; TLS + high throughput has a real cost |

**Pipelines vs transactions vs Lua:**

| Mechanism | Atomic? | Round trips | Can branch? | Use for |
|---|---|---|---|---|
| Individual commands | Per command | 1 per command | n/a | Anything single-key |
| **Pipelining** | **No** | 1 for N commands | n/a | Bulk loads, batch reads/writes where atomicity is not needed |
| `MULTI`/`EXEC` | Yes (**queued then executed together**) | 1 (+1 for WATCH) | **No** | Simple atomic batches; `INCR`+`EXPIRE` |
| `WATCH`/`EXEC` (optimistic) | Yes, with retry | 2+ | n/a | Read-modify-write where contention is low |
| **Lua script / `EVAL`** | **Yes** | 1 | **Yes** | Rate limiters, conditional updates, anything needing logic |

**Why `MULTI`/`EXEC` is not enough for a token bucket:** you cannot decide *inside* the transaction whether to decrement — `MULTI` has no `IF`. `WATCH` works but under high contention every transaction fails and retries, collapsing throughput exactly when you need it. Lua gives atomicity and branching in one round trip.

```lua
-- atomic conditional decrement, safe under concurrency
local v = tonumber(redis.call('GET', KEYS[1]) or '0')
if v <= 0 then
  return 0
end
redis.call('DECR', KEYS[1])
return 1
```

**Lua rules to remember:** scripts are atomic and block everything while running; keep them O(1)-ish and never call `KEYS`/`SMEMBERS` inside; use `redis.call` (errors propagate) vs `redis.pcall` (errors returned); all keys must be passed in `KEYS[]` (Redis 7 enforces this and warns/fails otherwise); a script that touches multiple keys in Cluster must use **hash tags** so they map to one slot; scripts are cached by SHA1, so you send `EVALSHA` after the first `SCRIPT LOAD` (and handle `NOSCRIPT` by re-loading).

### 3.3 Key expiry

Two mechanisms, both needed:

- **Lazy expiration:** on access, if the key is past its TTL, delete it and return nil.
- **Active expiration:** a background cycle samples keys with TTLs (20 keys per cycle, repeated; if >25% are expired it loops) and deletes them, so expired keys do not consume memory forever.

**Consequences:**
- Memory for expired keys is reclaimed some time after expiry, not exactly at expiry. Do not assume exact memory reclamation.
- A key can exist past its TTL if it is never sampled; readers still get nil (lazy path).
- **Millions of keys expiring at the same instant** (a batch job writing with identical TTLs) creates an expiration spike that burns CPU and can cause latency spikes. **Jitter the TTL.**
- Under a replica, expiry is driven by the **primary** (the primary sends `DEL` to replicas), so a replica can serve a logically-expired value if it is disconnected from the primary — replicas do not expire on their own for correctness reasons.
- **Expiry does not fire events reliably for `EXPIRE` on keys later overwritten** — for cache invalidation, use explicit deletes or keyspace notifications, not expiry, where correctness matters.

**Keyspace notifications** (`notify-keyspace-events`) let you subscribe to `expired`/`del`/`set` events. Fire-and-forget, and disabled by default because it costs CPU. Useful for "when this cache entry expires, kick off a refresh", not for correctness-critical invalidation.

### 3.4 Persistence: RDB, AOF, hybrid

| | **RDB (snapshot)** | **AOF (append-only file)** | **Hybrid (RDB preamble + AOF)** |
|---|---|---|---|
| What | Point-in-time binary dump of the whole dataset | Log of every write command | RDB base + AOF tail |
| Trigger | `save` rules (`900 1`, `300 10`, `60 10000`) or `BGSAVE` | `appendfsync` policy | Both |
| Data loss on crash | Everything since the last snapshot (minutes) | ≤ 1 s (`everysec`) or less (`always`) | ≤ 1 s |
| Fork cost | **Yes — forks, and COW can double memory** | Fork only for AOF rewrite | Both |
| Restart time | Fast (load a compact binary) | Slow without a rewrite (replay the whole log) | Fast |
| File size | Small | Large until rewritten | Moderate |
| Best for | Caches, fast restarts, backups | Data you would rather not lose | Modern default (`aof-use-rdb-preamble yes`) |

**`appendfsync` options and what they really mean:**

| Value | Behaviour | Loss window |
|---|---|---|
| `always` | fsync after every write | ~zero (but 10–100× slower writes) |
| `everysec` | fsync once per second in a background thread | **up to 1 second** — the practical default |
| `no` | Let the OS flush (~30 s) | Up to 30 s, and worse under load |

**The fork/COW problem — the most important operational detail about Redis persistence.** `BGSAVE` and AOF rewrite `fork()` the process; the child writes while the parent keeps serving. Memory is shared copy-on-write, but **every write by the parent after the fork copies the affected page**, so:

- Real memory can approach 2× the dataset during a snapshot under a write-heavy load.
- The `fork()` itself can block the event loop for milliseconds to **hundreds of milliseconds** on a large dataset (page-table copy is O(memory)).
- A 32 GB instance can stall 200–500 ms on fork. At 50k QPS that is 25,000 requests stalled.
- **Mitigations:** keep instances smaller (fragment by workload), avoid huge datasets per instance, use `no-appendfsync-on-rewrite yes`, disable automatic RDB in favour of replicas that snapshot, and monitor `latest_fork_usec`.

**The practical answer: "Redis is not a database because it loses the last second of writes on a crash and because a fork can stall the whole instance."** Even with `appendonly yes`.

### 3.5 Replication

```
Primary ────replication stream────▶ Replica 1
        └───replication stream────▶ Replica 2
```

- Replication is **asynchronous by default**. The primary acknowledges the write to the client before replicas have it.
- **`psync` / partial resync:** a replica that reconnects sends the replication offset and the primary replays from the backlog if it can (`repl-backlog-size`, default 1 MB). If the gap exceeds the backlog, a **full resync** happens (a fresh RDB) — expensive.
- **`WAIT numreplicas timeout`** blocks the client until N replicas acknowledge. It reduces (does not eliminate) the loss window and costs latency. `WAIT 1 100` is a common setting for "at least one replica has it".
- **`min-replicas-to-write` / `min-replicas-max-lag`** make the primary refuse writes if fewer than N replicas are connected and lagging less than X seconds — a safety valve against silent data loss.
- **Replica-read scaling:** `replicas` can serve reads, but they are eventually consistent and can serve stale data (the cache-repopulation race in the caching file). Redis Cluster also allows `READONLY` reads from replicas.
- **Chained replication** (replica of a replica) reduces primary fan-out but increases lag.

### 3.6 Redis Cluster

- **16,384 hash slots** distributed across primaries. `CRC16(key) mod 16384` determines the slot.
- **Hash tags:** `{user1001}.profile` and `{user1001}.sessions` both hash only on `user1001`, so they land in the same slot — required for multi-key commands and Lua scripts.
- **`MOVED`** — the client asked the wrong node and is redirected permanently; the client must update its slot map.
- **`ASK`** — a slot is mid-migration; the client should send the command to the target node *this once* without updating its map.
- **Cluster topology limits:** at least 3 primaries (and by convention 6 nodes total for 3 primaries + 3 replicas); no multi-key operations across slots; `SELECT` (multiple databases) is not supported; pub/sub in cluster is broadcast across all nodes (expensive).
- **Resharding:** `redis-cli --cluster reshard` moves slots; each moved slot's keys are migrated, causing brief `ASK` redirects. Large keys take time; reshard during low traffic.
- **Failure detection:** nodes gossip; a primary is marked `FAIL` on `cluster-node-timeout` (default **15,000 ms**) with quorum agreement, and a replica is promoted. During the election window (seconds), writes to that slot fail with `CLUSTERDOWN`.
- **Split brain protection:** `min-replicas-to-write` helps; a partitioned old primary will refuse writes if it cannot see replicas.

**Sentinel vs Cluster:**

| | **Sentinel** | **Cluster** |
|---|---|---|
| Purpose | High availability for a non-sharded primary + replicas | Sharding + HA |
| Data size | One node's memory | Sum of all primaries' memory |
| Multi-key across shards | Yes (single keyspace) | No (same slot only, or hash tags) |
| Failover | Sentinel quorum elects a replica; clients discover via Sentinel | Cluster gossip + replica promotion |
| Complexity | Lower | Higher (slot maps, resharding, client support) |
| Use when | Data fits in one node; you want simple HA | Data exceeds one node or you need write scaling |

**Redis Cluster does not give you "more memory per key" or cross-shard transactions.** It gives you more total memory and more write throughput. If your dataset fits comfortably on one node with headroom, **Sentinel is the better operational choice** — fewer moving parts, no cross-slot restrictions.

### 3.7 Memory management and eviction

```
maxmemory 16gb
maxmemory-policy allkeys-lru
maxmemory-samples 5          # samples per eviction decision (raise to 10 for better accuracy)
```

| Policy | Behaviour | Use when |
|---|---|---|
| `noeviction` | Writes error with OOM; reads still work | Data you cannot lose (then it is not a cache) |
| `allkeys-lru` | Evict the least recently used key among all keys | **Pure cache — the standard choice** |
| `allkeys-lfu` | Evict the least frequently used | Skewed workloads with a stable hot set; better than LRU |
| `allkeys-random` | Evict a random key | Uniform access patterns |
| `volatile-lru` / `volatile-lfu` / `volatile-ttl` | Only consider keys **with a TTL** | Mixed use — but **dangerous**: if no keys have TTLs, nothing is evictable and writes fail |
| `volatile-random` | Random among keys with TTLs | Rarely useful |

**Rules:**
- **Set `maxmemory` to 60–75% of the machine's RAM**, not 100%, so a fork during snapshot does not trigger the OOM killer.
- **Alert at 80% of `maxmemory`.** Above that, eviction starts and hit ratio silently degrades before any error appears.
- **`allkeys-lru` with `maxmemory-samples 10`** is a good default; the sampling makes LRU approximate, and 5 samples is the default.
- Understand that **eviction is a slow hit-ratio leak, not an event**. A cache that is 10% too small loses far more than 10% of hit ratio when the working set has a long tail.

### 3.8 Pub/Sub vs Streams

| | **Pub/Sub** | **Streams** |
|---|---|---|
| Model | Fire-and-forget broadcast | Append-only log with consumer groups |
| Delivery | At-most-once — lost if no subscriber is connected | At-least-once with `XACK`, pending list and `XCLAIM` |
| Replay | No | Yes, from any ID |
| Persistence | None | Stored in memory (and in RDB/AOF) |
| Consumers | All subscribers get every message | Consumer groups split messages; each group member gets a subset |
| Backpressure | None — slow consumers lose messages | Consumer lag visible via `XPENDING` |
| Cluster | Broadcast to all nodes | Sharded with the key |
| Use for | Cache invalidation broadcasts, config-change notifications, live dashboards | Work queues, event sourcing lite, task handoff, anything where a lost message is a bug |
| Use Kafka instead when | — | You need durability across broker restarts, long retention, or high partition counts |

**The trap:** using pub/sub for work distribution and losing messages when a consumer restarts. If losing the message is a bug, use Streams (or Kafka). If it is a notification ("your config changed, re-read it"), pub/sub is fine — but combine it with a TTL or a periodic re-read so a missed message self-heals.

### 3.9 Distributed locks

**The correct minimal implementation:**

```
SET lock:resource <unique-token> NX PX 30000
```

- `NX` — only if it does not exist (mutual exclusion).
- `PX 30000` — auto-expiry so a crashed holder does not deadlock the resource forever.
- `<unique-token>` — a random value (e.g. a UUID) so the release is safe.

**Release — atomically, checking the token:**

```lua
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
else
  return 0
end
```

Without the token check, a lock can be stolen: A acquires, A's TTL expires during a long GC pause, B acquires, A finishes and deletes B's lock — and now both are in the critical section. This is the single most common distributed-lock bug.

**The deeper problem: a lock with a TTL does not give you correctness.** Even with the token check, this sequence is unsafe:

```
t0  A acquires lock (TTL 30 s)
t1  A pauses (GC / VM freeze / network) for 35 s
t2  Lock expires
t3  B acquires the lock
t4  A wakes up and writes to the resource  ← both A and B in the critical section
```

**Fencing tokens are the real fix:** every lock acquisition returns a monotonically increasing token, and the *resource itself* rejects a write with a token lower than the highest it has seen. That requires the resource to participate — which is why a database with a conditional update is often a better lock than Redis.

**The Redlock controversy (know this for interviews).** Redlock is an algorithm that acquires a lock on N independent Redis primaries (usually 5) and considers the lock held if a majority ack within a time bound. Its critics (most famously Martin Kleppmann) argue it is unsafe for correctness because:
- It depends on **bounded clock drift** across nodes, which is not a safe assumption (a process can pause arbitrarily, and NTP can step a clock).
- It offers **no fencing**, so a paused holder can still corrupt a shared resource.
- It is **not needed for efficiency locks** (where a rare double-execution is tolerable) and **not sufficient for correctness locks**.
- The author's (antirez) rebuttal: with a bounded clock drift assumption and no long pauses, it holds. The debate is unresolved in the sense that both are right about different assumptions.

**What to say in an interview:** "I use `SET NX PX` with a unique token and Lua-based conditional release for *efficiency* locks where a rare double-execution is tolerable. For *correctness* locks, I use a fencing token and make the resource reject stale tokens, or I put the critical section in the database as a conditional update. I would not use Redlock as the sole guarantee for a financial operation."

### 3.10 Redis as a rate limiter

Redis is the natural home for distributed rate limiting, and the token-bucket Lua script belongs in your head (full version in [the gateway file](../02-networking/04-api-gateway-and-rate-limiting.md)). Key points specific to Redis:

- **Use `redis.call('TIME')` inside the script** rather than the caller's clock, so clock skew across app instances does not corrupt the window.
- **Cluster + multi-key = hash tags.** `rl:{user42}:tokens`.
- **Hot keys:** a global rate limit is one key; shard it (`rl:global:{0..9}`) and sum, or approximate with local counters.
- **Fixed-window `INCR` + `EXPIRE` needs care:** if the process dies between the two calls, the key never expires. Use `SET key 1 EX W NX` first, or do both in Lua.
- **Memory:** each counter is ~100 bytes; 10M active clients with a token bucket hash is roughly 2–3 GB. Plan for it.

### 3.11 Redis as a semantic cache

```
1. Embed the user's query                      (embedding model)
2. FT.SEARCH idx "*=>[KNN 1 @vec $q AS score]"  (RediSearch vector index)
3. If score ≥ threshold (0.93 cosine): return the stored answer
4. Else: call the LLM, then store query + answer + embedding with a TTL
```

Design rules specific to Redis:
- **Namespace by tenant and permission scope.** `sem:{tenant}:{model}:{hash}` — a shared namespace across tenants is a data-leak incident.
- **Include the model version in the key.** A different model gives different answers for the same prompt; mixing them in one index serves inconsistent answers.
- **TTL 1–24 h, and a hard invalidation on corpus change** (a document update must purge affected entries — tag them and delete by tag).
- **Threshold is a quality dial.** 0.90 is risky for factual answers; 0.95 is safe but saves less. Measure the false-hit rate against a labelled set before shipping.
- **Cap the index size** and evict, or it becomes the largest thing in your Redis.

### 3.12 Key design and namespacing

```
# Good
user:{1001}:profile          → Hash with fields
sess:{token}                 → String/JSON, TTL = session lifetime
rl:{user42}:tokens           → Hash for the token bucket
lb:game:2026:daily           → ZSet leaderboard
cache:v2:product:{sku}       → schema-versioned cache entry
sem:{tenant7}:gpt4:{hash}    → semantic cache entry

# Bad
u1001                        → no namespace, no structure
user:1001:profile:json:v2:cache  → key bloat; every byte × 10M keys is real
product:1001                 → no schema version; a deploy changing the format breaks
```

**Rules:**
- **Colon-delimited hierarchy**, so `SCAN MATCH user:{1001}:*` works and humans can read it.
- **Hash tags `{}` around the shard-determining part** if you use Cluster and need multi-key operations.
- **Schema version in the key** (`cache:v2:...`) so a format change is a new namespace rather than a wave of deserialisation errors.
- **Include the tenant/scope** for anything multi-tenant.
- **Keep keys short** but not cryptic — at 10M keys, 20 bytes of key is 200 MB.
- **Never use unbounded user input in a key** without validation (a 10 KB key per request is a memory attack).

### 3.13 Memory sizing

```
total_memory ≈ Σ over keys ( key_size + value_size + ~50–100 bytes overhead )
             × 1.5–2.0 (fragmentation + copy-on-write headroom for forks)

Example: 10M cache entries, 40-byte keys, 500-byte values
  raw  = 10M × 540 B           = 5.4 GB
  + overhead (dict, expiry, allocator ~30%)  ≈ 7 GB
  × 1.5 COW/replication headroom             ≈ 10.5 GB
  → provision a 16 GB instance, maxmemory 12gb
```

For **Redis Cluster**, multiply the per-shard requirement and account for the fact that the largest shard determines when you run out of memory — so you need more headroom than the average suggests. Use `redis-cli --bigkeys` and `MEMORY DOCTOR` regularly.

---

## 4. Variants & taxonomy

| Variant | Description |
|---|---|
| **Redis OSS** | The open-source server |
| **Valkey** | The Linux Foundation fork of Redis 7.2 after the 2024 licence change (BSD); drop-in compatible |
| **Redis Stack** | OSS + modules: RediSearch, RedisJSON, RedisTimeSeries, RedisBloom |
| **Redis Enterprise** | Commercial: active-active geo-replication, flash tiering, higher density |
| **KeyDB** | Multi-threaded fork (acquired by Snap) |
| **Dragonfly** | Multi-threaded, memory-efficient reimplementation |
| **AWS ElastiCache / MemoryDB** | Managed Redis / durable Redis |
| **Azure Cache for Redis / GCP Memorystore** | Managed equivalents |
| **Upstash** | Serverless Redis with per-request pricing |
| **Memcached** | Simpler, multi-threaded, no persistence, no data structures — still used for pure object caching at very high throughput |
| **Sentinel topology** | HA for a single shard |
| **Cluster topology** | Sharding + HA |
| **Active-active geo** | Multi-region writes (Enterprise) |
| **Redis as a queue (Lists/Streams)** | Lightweight messaging |
| **Redis with a vector index** | Semantic cache / small RAG store |
| **Redis as a lock service** | `SET NX PX` + fencing |
| **Redis as a leaderboard** | ZSet, the canonical use |

**Redis vs Memcached, since it comes up:**

| | Redis | Memcached |
|---|---|---|
| Data structures | Rich | Strings/objects only |
| Threading | Single-threaded command execution | Multi-threaded |
| Persistence | RDB/AOF | None |
| Replication | Yes | No (client-side sharding) |
| Max value | 512 MB | 1 MB default |
| Best for | Caches with logic — counters, locks, leaderboards, queues | Pure, very high throughput object caching with simple needs |

---

## 5. Pros / Cons

| Pros | Cons |
|---|---|
| Sub-millisecond operations, 100k–200k ops/sec/node | Memory is 10–100× more expensive per GB than disk |
| Rich server-side data structures reduce round trips | Single-threaded: one slow command stalls all clients |
| Atomic single commands and Lua scripting | No cross-slot transactions in Cluster |
| TTL everywhere; simple expiry semantics | Expiry is not exact; memory reclaim lags |
| Replication, Sentinel, Cluster all built in | Async replication means failover can lose writes |
| Very simple operations compared to a database cluster | Persistence is not fsync-by-default; RDB/AOF forks stall the loop |
| Pub/sub and Streams give lightweight messaging | Pub/sub is at-most-once |
| Excellent for caching, sessions, counters, locks, rate limits | Big keys and hot keys are catastrophic and easy to create |
| Semantic caching cuts LLM cost 20–60% | As a system of record it is the wrong tool: lossy, expensive, single-threaded |
| Ecosystem, tooling and managed offerings everywhere | Cluster resharding and slot management add real operational complexity |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 Fork / copy-on-write latency spikes

**Symptom:** periodic p99 latency spikes every few minutes, correlated with `BGSAVE` in the log, with no obvious cause in the application.

**Cause:** `fork()` on a large dataset copies page tables (O(memory)) and blocks the event loop; then COW doubles the pages touched by writes, so memory can approach 2× and the allocator works harder.

**Numbers:** a 16 GB instance can stall for 100–300 ms at fork; a 64 GB instance for 500 ms–1 s. At 50k QPS, a 300 ms stall is 15,000 queued requests.

**Fixes:**
- Smaller instances (shard the data) rather than one giant one.
- Disable automatic RDB (`save ""`) and take backups from a replica instead.
- `no-appendfsync-on-rewrite yes`.
- Use `repl-diskless-sync yes` to avoid the child writing a big RDB to disk.
- Monitor `latest_fork_usec` and `rdb_last_cow_size`; alert on both.
- Prefer AOF-only with `everysec`, which forks less often.
- **On a latency-sensitive workload, never co-locate a snapshotting Redis with anything else on the same host.**

### 6.2 Big keys

A single key with a huge value or a huge collection:

| Symptom | Cause |
|---|---|
| Intermittent multi-second stalls | `DEL`/`EXPIRE` on a 10M-element set (freed synchronously) |
| Network saturation | `HGETALL`/`SMEMBERS`/`LRANGE 0 -1` returning 100 MB in one reply |
| Memory imbalance in Cluster | One slot/shard far larger than others |
| Slow replication | Every full sync ships the big key |

**Detection:** `redis-cli --bigkeys`, `MEMORY USAGE key`, `redis-cli --memkeys`.
**Fixes:** `UNLINK` (async free) instead of `DEL`; split into chunks; use `HSCAN`/`SSCAN` instead of `HGETALL`/`SMEMBERS`; cap collection sizes with `LTRIM`/`ZREMRANGEBYRANK`; set `lazyfree-lazy-expire yes` and `lazyfree-lazy-user-del yes`.

### 6.3 Hot keys

One key absorbing a disproportionate share of commands. A single Redis instance handles ~100k–200k ops/sec; one key at 500k QPS is a hard ceiling regardless of cluster size, because a key lives on exactly one slot on one node.

**Detection:** `redis-cli --hotkeys` (needs `allkeys-lfu`), `MONITOR` (in dev only — it is a performance disaster), or `INFO commandstats` plus per-key sampling.
**Fixes:** in-process L1 cache for the hot key with a short TTL; Redis **read replicas** for read-heavy hot keys; key sharding (`hotkey:{0..N}` with random suffix); for counters, shard and sum. For a *write* hot key, shard and reconcile asynchronously.

### 6.4 Replication lag and failover data loss

```
t0  Client → primary: SET x = 2      (primary acks immediately)
t1  Primary crashes before replicating
t2  Sentinel/Cluster promotes a replica that still has x = 1
→ the acknowledged write is lost
```

- **The window is real** with async replication — typically single-digit milliseconds, but unbounded under network trouble or a loaded replica.
- **`WAIT 1 100`** narrows it at the cost of a round trip to a replica per write.
- **`min-replicas-to-write 1` `min-replicas-max-lag 10`** makes the primary refuse writes when replication is unhealthy — a good safety valve.
- **Split brain:** a partitioned old primary continues accepting writes for `cluster-node-timeout` (default 15 s), and those writes are lost when it is demoted. Reduce the timeout at the cost of more false failovers.
- **Implication for design:** never put data in Redis that you cannot reconstruct. If a lost write is a business incident, the write belongs in the database, with Redis as a cache in front.

### 6.5 `maxmemory` reached: silent degradation or hard errors

- With `noeviction`: writes start failing with `OOM command not allowed when used memory > 'maxmemory'`. Reads still work. The application may treat this as a generic error.
- With an eviction policy: writes succeed, but entries are evicted immediately after being written ("evict-write" churn), so hit ratio collapses while every dashboard shows healthy throughput.
- **Set `maxmemory` to ~60–75% of RAM, alert at 80%, and watch `evicted_keys` as a first-class metric.** A rising eviction rate with a flat hit ratio means you are just under the cliff.

### 6.6 The `KEYS`/`SMEMBERS`/`HGETALL` outage

```
KEYS user:*        → O(N) over the whole keyspace, blocks the event loop entirely
SMEMBERS big      → O(N) over a million members
HGETALL big       → serialises megabytes into one reply
ZRANGE key 0 -1   → same
SORT big          → O(N log N)
```

On a 10M-key instance, `KEYS *` can block for **seconds** — every other client times out.

**Fixes:** `SCAN` with a `MATCH` pattern and a `COUNT` hint, iterating in application code; `SSCAN`/`HSCAN`/`ZSCAN` for collections; `SORT` only on bounded sets; and ideally **never** enumerate keys in production — maintain an index set instead. Redis 7 added `OBJECT ENCODING`/`DEBUG` tooling and better `SCAN` guidance but the rule is unchanged: **the number of O(N) commands you may run in production is zero.**

### 6.7 Connection storms and client issues

- Each app instance holds a connection pool; a Redis failover causes every client to reconnect at once. A 500-instance fleet × 20 connections = 10,000 simultaneous reconnects against a freshly promoted replica.
- **Fix:** exponential backoff with jitter in the client, small pools (10–50 per instance is plenty for a single-threaded server), `tcp-keepalive` and `timeout` settings so dead connections are reaped.
- **The `CLIENT LIST`/`CLIENT KILL` trap:** an output-buffer-bloated client (a slow consumer of a pub/sub or Stream) can consume unbounded memory; bound it with `client-output-buffer-limit`.
- **Blocked clients:** `BLPOP`/`BRPOP`/`XREAD BLOCK` hold a connection and a blocked-client slot; `blocked_clients` in `INFO` is a metric worth watching.

### 6.8 Lua script hazards

- **A non-deterministic script breaks replication** (in Redis < 5 the script was replicated; in modern Redis effects are replicated, which fixes most of this — but still avoid time- and randomness-dependent scripts, or use `redis.replicate_commands()`).
- **A long-running script blocks everyone** and cannot be killed except with `SCRIPT KILL` (and only if it has not written) or `SHUTDOWN NOSAVE`. Keep scripts bounded.
- **Cluster:** all keys must be in one slot → hash tags. Otherwise `CROSSSLOT Keys in request don't hash to the same slot`.
- **`NOSCRIPT` errors** after a failover or restart: scripts are cached in memory and lost. Handle by re-loading.
- **Errors mid-script do not roll back** — Redis has no transactions in the rollback sense. A script that writes then fails leaves the earlier writes in place. Write scripts so that partial application is safe.

### 6.9 Cluster-specific failure modes

- **`CLUSTERDOWN The cluster is down`** — slots uncovered (a primary failed and no replica was promoted within `cluster-node-timeout`). Writes to those slots fail.
- **`MOVED` loops** with a stale client slot map — some clients handle this badly; check your client library's cluster support.
- **Resharding under load** produces `ASK` redirects and can time out slow clients; do it in small slot batches at low traffic.
- **No `SELECT`:** you get one keyspace; namespace with prefixes.
- **Pub/sub broadcast cost** grows with the number of nodes.
- **Key skew across slots:** with `hash tags` you *choose* to co-locate, which is exactly how you create a hot slot. `{user1001}:*` for a celebrity user puts all their keys on one node.

### 6.10 Other traps worth naming

- **Using Redis as the source of truth.** It is lossy on failover, memory-bound, and single-threaded. If it cannot be recomputed or reconstructed, it belongs in a database.
- **Storing sessions with no TTL.** Sessions accumulate forever; combine an absolute TTL with a sliding refresh, and cap the number of sessions per user.
- **`FLUSHALL` in production.** Obvious, still happens, and with a cold cache the database takes 100% of traffic.
- **Assuming `EXPIRE` gives exact invalidation.** It gives a staleness bound.
- **Not setting `maxmemory`.** Redis will happily use all the machine's RAM and then the OOM killer arrives — usually killing Redis and something else.
- **TLS overhead.** TLS is processed on the main thread; at high command rates it is a measurable cost. Use it where required, but know the price.
- **`SAVE` (blocking) instead of `BGSAVE`.** `SAVE` blocks the entire server for the duration of the snapshot.
- **Memory fragmentation.** `mem_fragmentation_ratio > 1.5` means the allocator is holding memory it cannot return; restart or enable `activedefrag`.
- **`INFO keyspace` showing millions of keys with no TTL** on a cache instance — a sign someone is using Redis as a store.

---

## 7. When to use / When NOT to use

### Use Redis when
- You need a **cache** with TTL and eviction.
- You need **shared, low-latency state** (sessions, rate limits, feature flags, presence).
- You need a **counter** or **leaderboard** with atomic increments and ranked reads.
- You need a **distributed lock** for an efficiency-only critical section.
- You need **lightweight messaging** (Streams for at-least-once, pub/sub for notifications).
- You need a **sliding-window rate limiter** with atomicity.
- You need a **semantic cache** for LLM responses (with RediSearch).
- You need a **small vector index** colocated with cache data.
- You need **approximate unique counts** (HyperLogLog) or **bitmap analytics**.
- You need a **queue** that is simpler than Kafka and you accept the durability trade-off.

### Do NOT use Redis when
- The data **must not be lost** — it is not a system of record.
- The dataset is **larger than memory** and cannot be evicted.
- You need **complex queries, joins, or secondary indexes** — that is a database's job.
- You need **multi-key transactions across shards** — Cluster forbids it.
- You need **durable, replayable messaging with long retention** — use Kafka.
- You need **analytical queries** over large data — use an OLAP store.
- You need **strong consistency for a correctness-critical decision** — use the database with a conditional write.

### Decision table

| Requirement | Answer |
|---|---|
| Cache 100k-QPS hot reads | Redis cache-aside, `allkeys-lru`, TTL + jitter |
| Sessions across 50 app instances | Redis string/JSON with TTL, sliding refresh |
| Global rate limit | Token bucket in Lua, hash-tagged, sharded if hot |
| Leaderboard of 10M players | ZSet; shard by time period if it grows unbounded |
| "Drivers near me" | `GEOADD` + `GEOSEARCH`, or a proper geospatial index at large scale |
| Unique visitors per day | Bitmap (exact, 125 KB per million) or HyperLogLog (0.81% error, 12 KB) |
| Task queue with retries | Streams + consumer groups, or Kafka if durability matters |
| Distributed lock for a cron job | `SET NX PX` + token; tolerate rare double-runs |
| Distributed lock for money movement | Database conditional update with a fencing token, not Redis |
| Repeated LLM questions | Semantic cache in RediSearch, per-tenant, threshold ≥ 0.93 |
| Dataset larger than RAM | A database, not Redis |
| Cross-shard atomic update | A database transaction, not Redis Cluster |

---

## 8. Numbers, defaults & tuning

| Quantity | Default / typical | Notes |
|---|---|---|
| `maxmemory` | 0 (unlimited!) | **Always set it** — 60–75% of RAM |
| `maxmemory-policy` | `noeviction` | Set to `allkeys-lru` for a cache |
| `maxmemory-samples` | 5 | Raise to 10 for more accurate LRU |
| Ops/sec per node | ~100k–200k | Single-threaded command execution |
| GET/SET latency | 0.2–1 ms in-region | Network-dominated |
| Pipelining throughput gain | 5–10× fewer round trips | Batch bulk operations |
| Value size limit | 512 MB | Never approach; keep values ≪ 100 KB |
| String encoding thresholds | `embstr` ≤ 44 bytes | Larger → `raw` |
| Hash/List/Set → hashtable thresholds | ~128 entries / 64-byte values (configurable) | Beyond these, memory per element rises sharply |
| Cluster hash slots | 16,384 | `CRC16(key) mod 16384` |
| Cluster minimum | 3 primaries | 6 nodes for 3 + 3 replicas |
| `cluster-node-timeout` | 15,000 ms | Failover detection; lower = faster but flappier |
| `repl-backlog-size` | 1 MB | Raise (e.g. 64 MB) to avoid full resyncs |
| `repl-timeout` | 60 s | Replica considered dead beyond this |
| `appendfsync` | `everysec` | Loss window ≤ 1 s |
| AOF rewrite trigger | 100% growth since last rewrite, min 64 MB | Tunable |
| `save` rules | `900 1`, `300 10`, `60 10000` | Disable (`save ""`) if a replica snapshots |
| RDB fork stall | 100 ms (16 GB) to ~1 s (64 GB) | Monitor `latest_fork_usec` |
| COW memory growth | Up to 2× dataset under writes | Headroom matters during snapshots |
| `slowlog-log-slower-than` | 10,000 µs | Log commands above 10 ms |
| `slowlog-max-len` | 128 | Raise to 1,000 in production |
| `timeout` (idle client) | 0 (never) | Set to 300 in busy fleets |
| `tcp-keepalive` | 300 s | Detect dead connections |
| `lazyfree-lazy-expire` | no | `yes` to avoid expiry stalls on big keys |
| `lazyfree-lazy-user-del` | no | `yes` so `DEL` behaves like `UNLINK` |
| Sentinel quorum | `(N/2)+1` | 3 Sentinels, quorum 2 |
| Sentinel `down-after-milliseconds` | 30,000 ms | Subjective-down threshold |
| Connection pool per app instance | 10–50 | More is not better on a single-threaded server |
| Key overhead | ~50–100 bytes per key | × 10M keys = 0.5–1 GB before values |
| HyperLogLog error | 0.81% | 12 KB max per key |
| Semantic cache threshold | 0.90–0.95 cosine | Below 0.90 you serve wrong answers |
| Fragmentation ratio alert | > 1.5 | Allocator cannot return memory |
| Memory alert | 80% of `maxmemory` | Before eviction starts degrading hit ratio |
| `WAIT` for durability | `WAIT 1 100` | Narrows the failover loss window |

---

## 9. Interview Q&A

**Q1 (easy). What is Redis and what do you use it for?**
Redis is an in-memory data-structure server. It keeps everything in RAM and provides strings, hashes, lists, sets, sorted sets, bitmaps, HyperLogLog, streams and geospatial indexes, plus Lua scripting and TTLs on everything. I use it as a cache, for shared session state across stateless app instances, for rate limiting, for leaderboards and counters, for distributed locks, for lightweight messaging, and for a semantic cache in front of an LLM. It is not the source of truth for anything — the transcript's framing is right: it is temporary storage that uses memory as the cache.

**Q2 (easy). Why is Redis so fast?**
Three reasons. Everything is in memory, so there is no disk I/O on the read path. It is single-threaded with an event loop, so there is no lock contention, no context switching and no thread coordination. And it has efficient data structures and encodings — listpacks for small collections, skiplists for sorted sets, and specialised encodings for small integers and short strings. The single-threaded design is also why every individual command is atomic, which is a correctness benefit, not just a performance one.

**Q3 (easy). What is the difference between RDB and AOF persistence?**
RDB is a point-in-time snapshot of the whole dataset — fast to load, compact, but you lose everything since the last snapshot. AOF logs every write and replays it on restart — much smaller loss window, up to one second with `appendfsync everysec`, but slower to load and larger on disk unless it is rewritten. Modern Redis defaults to a hybrid: an RDB preamble with an AOF tail, which gives fast restarts plus a small loss window. Neither makes Redis a database, because the loss window is non-zero and a fork for snapshotting can stall the instance.

**Q4 (medium). Explain the single-threaded event loop and its implications.**
Commands are executed one at a time on one thread. The benefit is that a single command like `INCR` or `SETNX` is atomic without any locking, which is why Redis is a correct rate limiter and lock primitive. The cost is that a slow command blocks every other client, so O(N) commands like `KEYS`, `SMEMBERS` on a large set, `HGETALL` on a large hash, or `DEL` on a huge key are production outages waiting to happen. It also means one instance cannot scale beyond one core for command execution, so scaling is by sharding, not by adding cores. In Redis 6 and later there are IO threads for socket reads and writes, but command execution is still single-threaded.

**Q5 (medium). How do you build a distributed rate limiter in Redis?**
A token bucket implemented in Lua. The bucket holds a token count and a last-refill timestamp; on each request the script refills based on elapsed time up to the bucket capacity, checks whether there is a token, decrements, and returns allow or deny. It has to be Lua rather than `MULTI`/`EXEC` because `MULTI` cannot branch — you cannot decide inside a transaction whether to decrement — and `WATCH`/`EXEC` needs retries that collapse under contention. Lua runs atomically and can branch, in one round trip. Two Redis-specific details: I read time with `redis.call('TIME')` inside the script rather than trusting the caller's clock, so clock skew between app instances does not corrupt the window; and in Cluster I use hash tags so the keys land in one slot.

**Q6 (medium). How do you implement a distributed lock correctly?**
`SET lock:resource <random-token> NX PX 30000` to acquire — `NX` for mutual exclusion and `PX` so a crashed holder does not deadlock the resource forever. To release, a Lua script that checks the value equals my token before deleting, because otherwise a lock can be stolen: if my TTL expires during a long GC pause, someone else acquires, and my unconditional delete then removes *their* lock. And I am explicit that this gives me an *efficiency* lock, not a correctness guarantee, because a holder can still be paused past its TTL and write into the critical section. For correctness I use a fencing token and make the resource reject stale tokens, or I do the critical section as a conditional update in the database.

**Q7 (medium). What is Redis Cluster and when do you use it?**
Cluster shards the keyspace across primaries using 16,384 hash slots, with replicas per primary for failover. It gives you more total memory and more write throughput than a single node, plus high availability. The constraints matter: multi-key operations must be in the same slot, so you use hash tags to force co-location; there is no `SELECT` for multiple databases; and you need at least three primaries. I would choose Sentinel instead if my dataset fits on one node with headroom, because it is much simpler and has no cross-slot restrictions. Cluster is for when you genuinely exceed one node or need to shard writes.

**Q8 (hard). Explain the fork and copy-on-write problem and how you mitigate it.**
When Redis snapshots with `BGSAVE` or rewrites the AOF, it forks. The fork itself copies page tables, which is O(memory), so on a 32 GB instance it can block the event loop for two to five hundred milliseconds. After the fork, memory is shared copy-on-write, but every write the parent makes copies the affected page, so real memory can approach twice the dataset under a write-heavy load. Both of those are production incidents on a latency-sensitive service. I mitigate by keeping instances smaller rather than one giant one, disabling automatic RDB saves and taking backups from a replica, setting `no-appendfsync-on-rewrite`, using diskless replication, and monitoring `latest_fork_usec` and `rdb_last_cow_size` as first-class metrics.

**Q9 (hard). What is a hot key and how do you deal with it?**
A single key taking a disproportionate share of traffic. Because a key lives on exactly one slot on one node, and a node handles roughly 100k–200k operations per second, one key at 500k QPS is a hard ceiling no amount of clustering fixes. My first fix is an in-process L1 cache in each application instance with a short TTL, which removes most of the Redis traffic entirely. For reads I can add Redis read replicas. For writes I shard the key — `counter:{0..9}` incremented across shards and summed on read, or random suffixes for a read-mostly hot key. And I would detect it with `--hotkeys` under an LFU policy rather than discovering it during an incident.

**Q10 (hard). Redis goes down entirely. What happens to your system?**
If Redis is only a cache, the system degrades: every read goes to the database, which almost certainly cannot take it, so I need load shedding or a degraded mode — that is the honest answer, and it is why I size the database for some multiple of the cached load, not for zero. If Redis holds sessions, users are logged out unless I have a fallback signing mechanism. If it holds rate-limit counters, my limiter must fail open or fall back to a conservative local limit. If it holds locks, in-flight critical sections may overlap. The design rule is that anything I cannot afford to lose on a Redis restart belongs in the database, and I test the failure rather than assuming it.

**Q11 (hard, testing §6). Someone ran `KEYS *` in production and everything timed out. Why, and how do you prevent it?**
`KEYS` is O(N) over the entire keyspace and it runs on the single command thread, so on a ten-million-key instance it blocks every other client for as long as it takes to scan — potentially seconds. It also builds the whole result in memory. Prevention: never enumerate keys in production. If I need to find keys, use `SCAN` with a `MATCH` pattern and iterate in application code, which is incremental and non-blocking, or better, maintain an explicit index — a set of the keys in a namespace — so I never need to enumerate. For collections, `SSCAN`/`HSCAN`/`ZSCAN` instead of `SMEMBERS`/`HGETALL`/`ZRANGE 0 -1`. I would also rename or disable the command in production with `rename-command KEYS ""`, which is a cheap and effective guard.

**Q12 (hard, testing §6). Redis failover lost acknowledged writes. How is that possible and how do you prevent it?**
Replication is asynchronous by default: the primary acknowledges a write to the client and then streams it to replicas. If the primary dies before a replica receives it, and a replica is promoted, the acknowledged write is gone. The window is usually single-digit milliseconds but is unbounded under network trouble or a loaded replica. I can narrow it with `WAIT 1 100`, which blocks the client until one replica acknowledges, at the cost of a round trip; and with `min-replicas-to-write 1` and `min-replicas-max-lag 10`, which make the primary refuse writes when replication is unhealthy. There is also a split-brain version: a partitioned old primary keeps accepting writes for `cluster-node-timeout`, default fifteen seconds, and those writes are lost when it is demoted. The design conclusion is that Redis is not a system of record — if a lost write is a business incident, the write belongs in the database.

**Q13 (hard, testing §6). Your Redis memory is at 100% but throughput looks fine. What is happening?**
With an eviction policy set, writes keep succeeding because Redis evicts to make room, so nothing errors and throughput looks healthy. But the cache is now thrashing — it is evicting entries faster than they are reused, so the hit ratio collapses and the database sees the full load, which is where the incident actually shows up. With `noeviction`, the other thing happens: writes start failing with `OOM command not allowed when used memory > maxmemory` and reads still work, so the failure is one-sided and confusing. In both cases the fixes are the same: alert at 80% of `maxmemory`, never set `maxmemory` to the whole machine, measure the real working-set size rather than guessing, and consider LFU or W-TinyLFU rather than LRU if there is a stable hot set among a lot of cold data.

**Q14 (hard). When would you choose Sentinel over Cluster, and Memcached over Redis?**
Sentinel over Cluster when the dataset fits comfortably on one node with the headroom for a fork, because Cluster's cross-slot restrictions and resharding complexity are a real cost that buys nothing if you do not need the sharding. Cluster only when you genuinely exceed one node's memory or need to shard writes. Memcached over Redis when you need a pure object cache with no data structures, no persistence and no replication, and you want multi-threaded scaling within a single instance — Memcached uses multiple cores where Redis uses one, and it is simpler. But Redis's data structures are usually worth the single-thread constraint, because a leaderboard or a rate limiter in Memcached means pulling the data into the application and losing atomicity.

**Q15 (hard). How would you use Redis in an LLM application?**
Four distinct roles. First, a semantic cache: embed the query, do a KNN search over a RediSearch vector index, and if the nearest cached query is above about 0.93 cosine similarity return the stored answer without calling the model — that typically cuts LLM calls by 20 to 60% for support and FAQ workloads. Second, a rate limiter and token budget with a token bucket in Lua, keyed per tenant, with reserve-then-settle on token counts. Third, session and conversation state with TTLs, so the app tier stays stateless. Fourth, a cheap distributed lock or idempotency key for request deduplication. The two things I am careful about: the semantic cache must be namespaced per tenant and per permission scope, because a shared one leaks data between customers; and the model version must be in the key, because a different model gives different answers for the same prompt.

---

## 10. Common mistakes candidates make

1. **Calling Redis a database.** It is not; say why (loss window, memory cost, single-threaded).
2. **Not mentioning the single-threaded loop** — it explains atomicity, blocking, and why you shard instead of adding cores.
3. **Using `MULTI`/`EXEC` for logic that needs branching.** Say Lua.
4. **Describing a lock without the token check and without fencing tokens.**
5. **Ignoring `maxmemory` and eviction.** A cache without a policy is an OOM incident.
6. **Forgetting `allkeys-lru` vs the `noeviction` default.**
7. **Not knowing `UNLINK` vs `DEL`** for big keys.
8. **Using pub/sub for work that must not be lost.** Streams or Kafka.
9. **Cluster without hash tags** — a `CROSSSLOT` error waiting to happen.
10. **Not mentioning hash slots (16,384) or MOVED/ASK** when discussing Cluster.
11. **Ignoring fork/COW stalls** — the most common real Redis latency incident.
12. **Not mentioning failover data loss** with async replication.
13. **Saying "Redis is fast so we can store everything there."** Memory cost is real.
14. **Not knowing the semantic-cache per-tenant scope requirement** in an AI-flavoured answer.
15. **Skipping the hot-key and big-key discussion**, both of which are guaranteed interview probes once you name Redis.

---

## 11. Cheat-sheet summary

| Concept | One-line truth |
|---|---|
| Redis | In-memory data-structure server; single-threaded command loop |
| Atomicity | Every single command is atomic; Lua gives multi-step atomicity |
| Blocking | One slow command blocks every client — zero O(N) commands in production |
| String | Cache values, counters, locks; 512 MB limit; `embstr` ≤ 44 bytes |
| Hash | Field-level atomic updates on an object |
| List | Simple queue, recent-N buffer (`LTRIM`) |
| Set | Tags, unique visitors, set algebra (`SINTER`) |
| Sorted set | **Leaderboards**, sliding-window logs, priority queues, geohash scores |
| Bitmap | DAU (1 bit/user/day; 1M users = 125 KB) |
| HyperLogLog | Approximate distinct counts, 0.81% error, 12 KB fixed |
| Stream | Log with consumer groups; at-least-once with `XACK` |
| Geospatial | `GEOADD` + `GEOSEARCH`, backed by a ZSet |
| Pub/sub | Fire-and-forget, at-most-once — notifications only |
| Expiry | Lazy + active sampling; memory reclaim lags; **jitter your TTLs** |
| RDB | Snapshot; fast load; loses everything since the last save |
| AOF | Command log; `everysec` loses ≤ 1 s |
| Hybrid | RDB preamble + AOF tail; modern default |
| Fork/COW | 100 ms–1 s stall on large datasets; up to 2× memory during snapshots |
| Replication | **Asynchronous** — failover can lose acknowledged writes |
| `WAIT` | `WAIT 1 100` narrows the loss window |
| Cluster | 16,384 slots; MOVED/ASK; hash tags for multi-key; 3+ primaries |
| Sentinel | HA for a single shard; simpler than Cluster |
| `cluster-node-timeout` | 15,000 ms default; split-brain window |
| `maxmemory` | 60–75% of RAM; alert at 80% |
| `allkeys-lru` | The right policy for a pure cache |
| `noeviction` | Writes fail with OOM — correct for non-cache data |
| Big key | Blocks the loop on `DEL`/`HGETALL`; use `UNLINK` and `SCAN` |
| Hot key | One key, one node, ~150k ops/s ceiling; fix with L1, replicas, sharding |
| Lock | `SET NX PX` + token check in Lua; **fencing tokens for correctness** |
| Redlock | Contested; unsafe for correctness locks without fencing |
| Lua | Atomic with branching; Cluster needs hash tags; `NOSCRIPT` after failover |
| Rate limiter | Token bucket in Lua, `redis.call('TIME')`, hash-tagged keys |
| Semantic cache | RediSearch KNN, ≥ 0.93 cosine, **per-tenant**, model version in the key |
| Key design | `v2:user:{1001}:profile` — namespace, hash tag, schema version |
| Memory estimate | (key + value + ~75 B) × keys × 1.5–2.0 headroom |
| Connection pool | 10–50 per app instance; backoff + jitter on reconnect |

**"If the interviewer says X → answer Y"**

| Interviewer says | You say |
|---|---|
| "Add a cache" | "Redis cache-aside, `allkeys-lru`, TTL + jitter, `maxmemory` at 70% of RAM" |
| "Distributed rate limiter" | "Token bucket in Lua with `redis.call('TIME')`, hash-tagged keys, 429 + Retry-After" |
| "Distributed lock" | "`SET NX PX` + token-checked Lua release; fencing tokens if correctness matters" |
| "Redis is slow every few minutes" | "Fork/COW from a snapshot — smaller instances, `save ""`, monitor `latest_fork_usec`" |
| "Redis lost data on failover" | "Async replication; `WAIT 1 100` and `min-replicas-to-write`, or move it to the DB" |
| "Scale Redis" | "Replicas for reads; Cluster with hash tags for write scale; L1 for hot keys" |
| "Redis is out of memory" | "Eviction is thrashing the hit ratio; measure the working set and set `maxmemory` with headroom" |
| "Cache LLM answers" | "RediSearch semantic cache, per-tenant namespace, model version in the key, threshold ≥ 0.93" |

---

## 12. Cross-links

- [Caching Strategies and Layers](02-caching-strategies-and-layers.md) — where Redis sits in the cache hierarchy
- [Cache Invalidation and Consistency](04-cache-invalidation-and-consistency.md) — stampede, single-flight, and pub/sub invalidation
- [Load Balancers](01-load-balancers.md) — consistent hashing and why session state belongs in Redis
- [Sharding, Replication and Consistency](05-sharding-replication-and-consistency.md) — replication lag, quorums, and Redis Cluster contrasted with database sharding
- [API Gateway and Rate Limiting](../02-networking/04-api-gateway-and-rate-limiting.md) — the rate-limit service that runs on these primitives
- [Forward and Reverse Proxy](../02-networking/03-forward-and-reverse-proxy.md) — connection pools and long-lived WebSocket state
- [Vector Databases](../04-data/03-vector-databases.md) — RediSearch KNN versus a dedicated vector DB
- [SQL Deep Dive](../04-data/01-sql-deep-dive.md) — what a real system of record guarantees
- [WhatsApp Chat System](../../Case-Studies/HLD/02-whatsapp-chat-system.md) — presence and connection registry in Redis
- [Instagram News Feed](../../Case-Studies/HLD/04-instagram-news-feed.md) — feed caching in Redis at scale
- [Scaling, Caching and DB Cheatsheet](../../Cheat-Sheets/03-scaling-caching-db-cheatsheet.md)
