# Scaling, Caching & Databases — 45 Interview Questions with Model Answers

> **TL;DR** — This is the drill file for the scaling half of a system-design interview. Every question has a model answer written the way you would say it, plus a **`Probe:`** follow-up and a **`Weak:` vs `Strong:`** contrast where it matters. The topics that generate the most signal are **caching and invalidation** (because everyone can say "add a cache" and almost nobody can explain the stale-repopulation race), **sharding and replication** (because choosing a shard key is where designs are actually made or broken), **CAP and PACELC** (because the correct answer is "it is per-operation, not per-system"), and **SQL vs NoSQL** (because the right answer is an access pattern, not a preference). Read it once end to end, then live in the pushback questions marked **P**.

---

## How to use this file

| Symbol | Meaning |
|---|---|
| **Q / A** | The question and a model answer you could say out loud |
| **Probe** | The follow-up that tests depth — the interviewer's second question |
| **Weak / Strong** | Calibration contrast |
| **Level** | E = easy, M = medium, H = hard, P = pushback |

**The three rules for every answer:** name the **number**, name the **failure mode**, and name the **trade**. A design answer without a trade is a description, and descriptions are not what is being graded.

---

## 1. Load balancing (Q1–Q5)

### Q1. L4 versus L7 load balancing. — E

**A:** An L4 load balancer operates on the TCP or UDP layer — it forwards packets based on IP and port without understanding the payload. It is fast, protocol-agnostic, and it can handle arbitrary traffic including non-HTTP. Its limitation is that it cannot make routing decisions based on anything in the request, so it cannot route by path, header or cookie, and it cannot do TLS termination with content awareness unless it also acts at L7. An L7 load balancer parses HTTP, so it can route by path, host, header or cookie, terminate TLS, retry idempotent requests, do health checks at the application level, and apply rate limits. The cost is that it does more work per request and it must understand the protocol.

The practical answer: both, in layers. An L4 balancer at the edge for raw throughput and DDoS absorption, L7 behind it for routing and policy. Or a single L7 balancer with an L4 fast path for the bulk traffic.

### Q2. Compare load-balancing algorithms. — M

**A:**

| Algorithm | How | Use when |
|---|---|---|
| **Round robin** | Rotate through servers | Servers are homogeneous and requests are uniform |
| **Weighted round robin** | Rotate with weights | Servers differ in capacity; also for canary weight shifts |
| **Least connections** | Send to the server with the fewest active connections | Requests vary in cost — the default for most APIs |
| **Least response time** | Fewest connections and lowest latency | Latency-sensitive, when latency is measured reliably |
| **IP hash** | Hash the client IP | Session affinity without a shared store; breaks on NAT and on membership change |
| **Consistent hashing** | Hash onto a ring with virtual nodes | Cache affinity; only a fraction of keys move when membership changes |
| **Random with two choices** | Pick two at random, choose the less loaded | Power-of-two-choices; nearly optimal without global state |

**The default I would pick:** least connections for general APIs, because request cost varies and least-connections adapts; consistent hashing when there is per-server state to preserve; two-random-choices when the balancer has no reliable global view.

**Probe: why is round robin bad for an API with variable request cost?** Because it counts requests, not work. One server can get ten cheap requests while another gets ten expensive ones, and the second saturates while the first idles.

### Q3. What are health checks, and how do you avoid making things worse? — M

**A:** A health check is a periodic probe of a backend that removes it from rotation when it fails. The traps: (1) **shallow health checks** — "the process responds to `/health`" — pass even when the service cannot reach its database, so traffic keeps going to a broken instance. (2) **Deep health checks that are too deep** — if the health check depends on the database, and the database has a blip, every instance reports unhealthy simultaneously and the load balancer removes everything; the service goes from degraded to fully down. (3) **Too-aggressive thresholds** — a single failed probe removing an instance causes flapping under GC pauses or transient network noise. So: a shallow liveness check for "restart me", a readiness check that includes critical dependencies but with a **short timeout and a tolerant threshold**, and hysteresis — three consecutive failures to remove, two successes to add back.

**Strong:** "The health check must not be able to take down the whole fleet, which means it should reflect whether this instance can serve traffic *better than the alternative* — not whether the world is perfect."

### Q4. How does session affinity work, and when is it a mistake? — M

**A:** Affinity routes all of a client's requests to the same backend, either by hashing the client IP or by setting a cookie the balancer reads. It is needed when the backend holds per-session state in memory. It is usually a mistake because it undermines the reason to have multiple backends: a hot client concentrates load on one instance, a rebalance logs users out, and during a deploy the draining instance loses its sessions. The right answer is almost always to externalise the session — Redis, or a signed cookie carrying the session — so affinity becomes unnecessary and any instance can serve any request. Keep affinity only for genuinely stateful protocols like WebSocket, and even there, treat the connection as the affinity, not the user.

### Q5. A load balancer is distributing unevenly. Diagnose it. — P

**A:** I would check, in order. (1) **Is the algorithm the problem?** Round robin over heterogeneous instance types or over requests with very different costs produces unevenness by construction. (2) **Are the health checks flapping?** An instance being added and removed repeatedly gets a burst of traffic when it returns. (3) **Is there connection reuse?** With keep-alive, a long-lived client connection stays pinned to one backend for its lifetime, so a few very active clients can skew the distribution badly — the fix is a maximum connection age or a maximum requests per connection. (4) **Is the metric misleading?** CPU per instance says nothing if the work is not CPU-bound; look at the metric that reflects the bottleneck — event-loop lag, connection count, or p99 latency. (5) **Is the balancer itself overloaded?** A balancer at its connection limit drops or delays; check its own saturation.

**Strong:** "The most common cause I have seen is keep-alive pinning: the balancer is doing exactly what it was told, and the connections are just too long-lived for round robin to mean anything."

---

## 2. Caching and Redis (Q6–Q15)

### Q6. What is the difference between cache-aside and write-through? — M

**A:**

| Pattern | Read path | Write path | Trade |
|---|---|---|---|
| **Cache-aside (lazy loading)** | Check cache; on a miss, read the DB and populate | Write the DB; **invalidate** the cache | Simple and resilient to cache failure; the first read after a write is slow |
| **Write-through** | Always a cache hit after the first write | Write the cache and the DB synchronously | Reads are always fast; every write pays both latencies and the cache becomes a critical path for writes |
| **Write-behind (write-back)** | Fast | Write the cache, flush to the DB asynchronously | Very fast writes; **data loss on a crash before the flush** |
| **Read-through** | The cache itself loads on a miss | — | Cleaner application code; the cache needs DB awareness |

**What I would choose:** cache-aside with explicit invalidation for almost everything, because a cache failure degrades to slow rather than to broken. Write-through where the read latency is the binding constraint and the data is small. Write-behind only where losing a few seconds of writes is genuinely acceptable — metrics, view counts, telemetry.

**Probe: why not write-behind for everything?** Because the durability guarantee drops from "committed" to "probably" without the client knowing. For anything a user expects to be saved, that is a correctness bug, not a performance trade.

### Q7. Explain the stale-repopulation race. — H

**A:** It is the race that makes delete-on-write fail.

```
t0  Reader R sees a cache miss and reads the DB, getting the OLD value V1
t1  Writer W updates the DB to V2 and DELETES the cache key
t2  R finishes and SETs the cache to V1     ← the stale value, now cached
    → the cache holds stale data until the TTL expires
```

The window is the time between the reader's database read and its cache write, which under load can be tens of milliseconds — plenty of opportunity. The fixes, in order of robustness: **version-stamped writes** (store a version with the value and use compare-and-set so a stale writer cannot overwrite a newer entry — this closes the race completely); **delayed double-delete** (delete, wait 300–500 ms, delete again, which catches the common case but not all of it); **the CDC/invalidation-log approach**, where the writer publishes an invalidation that the cache consumes in order; and, for read-your-writes at the application level, routing a user's reads to the primary for a few seconds after they write.

**Strong:** "Delayed double-delete is a mitigation, not a fix. The only way to close the race is to make the cache write conditional on a version, so a late writer cannot clobber a newer value."

### Q8. Why is `SETNX`-style distributed locking dangerous? — H

**A:** A distributed lock in Redis is only as reliable as Redis, and Redis with asynchronous replication can lose writes on failover — so a client can hold a lock that the new primary does not know about, and a second client can acquire the same lock. Redlock — acquiring a majority across independent Redis nodes — was designed to address this and is genuinely contested: a client can be paused (a long GC, a VM migration) past the lock's expiry, resume, and write while believing it still holds the lock. There is no fencing token in plain Redlock, so a stale holder's write is not rejected.

**The practical advice:** treat a Redis lock as an *optimisation that reduces contention*, never as a correctness mechanism. The correctness must come from a fencing token (a monotonically increasing number that the storage layer checks) or from a conditional write in the authoritative store. And in most designs you do not need the lock at all — a compare-and-set on the data itself is simpler and cannot be wrong ([see BookMyShow](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) §3.5).

### Q9. A cache node dies. What happens? — M

**A:** If the cache is a sharded cluster and one node dies, the keys it held are gone. Three consequences. First, the missing keys become misses, and every miss goes to the database — so the database sees a step increase in load proportional to the dead node's share of the keyspace, which with 10 shards is a 10% increase. That is usually survivable. Second — and this is the dangerous one — if the evicted keys include a hot key, the *entire* load for that key concentrates on one database row or one query, and that can be fatal. Third, if the missing keys are the same keys that millions of users request simultaneously, you get a stampede on exactly those keys ([see §Q12](#q12-explain-cache-stampede-and-how-you-prevent-it)).

**Mitigations:** replicas per shard so a node failure is a failover rather than a data loss; consistent hashing so only 1/N of the keys move when the topology changes; a request-collapsing layer at the origin (single-flight) so a stampede after a failure cannot reach the database en masse; and load-shedding at the origin as a last resort, returning a degraded response rather than falling over.

### Q10. Redis data structures and what you use them for. — M

**A:**

| Structure | Use |
|---|---|
| **String** | Simple values, counters (`INCR`), cached serialised objects, distributed locks (`SET NX EX`) |
| **Hash** | Objects with fields you update individually — a user profile, a session — saving bandwidth versus a whole-object rewrite |
| **List** | Queues and stacks (`LPUSH`/`RPOP`), recent-item feeds, bounded logs (`LTRIM`) |
| **Set** | Membership, deduplication, intersections — "who follows both" |
| **Sorted set (ZSET)** | Leaderboards, rate limiters, priority queues, anything ranked by score — `ZADD`/`ZRANGEBYSCORE` |
| **HyperLogLog** | Approximate unique counts (0.81% standard error) with 12 KB regardless of cardinality — unique visitors |
| **Bitmap** | Per-user boolean state — daily active users, feature flags, presence |
| **Stream** | An append-only log with consumer groups — a lightweight Kafka for modest volumes |
| **Geospatial (a ZSET with geohash scores)** | `GEOADD`/`GEOSEARCH` for nearby queries |

**Probe: when is a Redis Stream the wrong choice?** When you need long retention, high partition counts, replay from an arbitrary offset, or strong durability — that is Kafka. A Stream is right for modest volume with simple consumer-group semantics and no separate infrastructure.

### Q11. What is the single-flight pattern? — M

**A:** When a hot key expires or is missing, thousands of concurrent requests all miss and all recompute the same value, hammering the database with the same expensive query. Single-flight collapses them: the first request acquires a per-key lock and does the work; the others wait for that result rather than duplicating it. Implementation is a per-key mutex in-process (Go's `singleflight` package is the canonical example), or a Redis `SET NX` key with a short TTL across instances. The important refinement: **bound the wait**. A request should wait a short, fixed time (say 50–100 ms) and then, if the leader is still working, fall through and compute for itself rather than waiting indefinitely — because a slow leader must not become the whole system's latency.

**Strong:** "Single-flight with a bounded wait and a fall-through. Unbounded waiting trades a database stampede for a latency cliff, which is not obviously better."

### Q12. Explain cache stampede and how you prevent it. — M

**A:** A stampede — also called dogpile or thundering herd — is when a popular cached value expires or is invalidated and the simultaneous misses all hit the origin. With 100,000 requests/sec for one key, a naive expiry produces 100,000 database queries in the same instant.

Four defences, and the good answer gives more than one:
1. **Single-flight** — collapse the concurrent misses into one computation.
2. **Jittered TTLs** — add a random ±10% to expiry so keys do not all expire at the same moment; this prevents the *synchronised* stampede across many keys, which is the avalanche case.
3. **Probabilistic early expiration (XFetch)** — each reader, as the value approaches expiry, recomputes with a probability that rises as the deadline nears, so one reader refreshes before the value dies. The standard formula uses a `beta` of about 1.0 and the recomputation time; the effect is that the refresh happens early and only once.
4. **Refresh-ahead** — a background job refreshes hot keys before they expire, so nothing ever misses. This only works for a known hot set.

**Probe: how is this different from a hot key?** A stampede is about *time* — everything missing at once. A hot key is about *concentration* — one key getting disproportionate traffic even when it is cached. They need different fixes: single-flight and jitter for the stampede, key replication and local in-process caches for the hot key.

### Q13. Hot key — one key gets 1 M requests/sec. What do you do? — H

**A:** First, recognise that the cache hit ratio is fine and the problem is that one Redis shard holds the key, so all 1 M requests hit a single core. The fixes, in order:
1. **Local in-process cache** in each application instance, with a short TTL (1–10 s). This removes the network hop entirely and is by far the most effective fix — 1 M requests/sec becomes a memory read. The cost is bounded staleness of a few seconds.
2. **Key replication with a random suffix** — store `key:0` through `key:9` and have each reader pick one at random, spreading the load across shards. Writes must update all ten, which is fine for read-mostly keys.
3. **Read replicas for that shard.**
4. **Put a CDN in front** if the key corresponds to a public HTTP response.
5. **Change the data model** so the hot key does not exist — for a counter, shard it into N counters and sum on read ([see Redis Deep Dive](../HLD/03-scaling/03-redis-deep-dive.md) §6.10).

**Strong:** "The local in-process cache is the fix I would reach for first, because it eliminates the network hop rather than distributing it, and for a read-mostly value a few seconds of staleness is almost always acceptable."

### Q14. When should you NOT add a cache? — H

**A:** When the data changes faster than the cache can be invalidated and the cost of a stale read is high — for example an account balance that gates a spend, or a seat's availability at the moment of booking. When the hit ratio would be low, because a cache below roughly 80% hit ratio adds a hop and an invalidation burden without saving much. When the underlying query is already fast — if it is 2 ms, a cache that costs 1 ms of network round trip and introduces invalidation complexity is a bad trade. When correctness is more valuable than latency and there is no versioning mechanism. And when you would be caching to avoid fixing the real problem, which is usually a missing index or an N+1 query.

**Strong:** "The question I ask is 'what breaks if this value is stale for 30 seconds?' If the answer is 'nothing', cache it. If the answer is 'someone gets double-charged', do not."

### Q15. Your cache and database disagree. How do you find out which is wrong? — P

**A:** First, establish which one is authoritative — in every design I would describe, the database is, and the cache is derived. So a disagreement is always "the cache is stale", and the question becomes why. Then: is it a missed invalidation (the writer did not delete, or the delete failed), a race (the stale-repopulation race from Q7), a TTL that is too long for the change rate, or a key that is not being invalidated at all because the writer does not know it exists — which is extremely common, since a value is often cached under a key the writer never computes. The diagnostic is to add a version to the cached value and log when the version read differs from the version in the store; that turns "the cache is wrong sometimes" into a rate and a specific key.

**Strong:** "The structural fix is versioned cache entries, because then a stale writer physically cannot overwrite a newer value, and the disagreement becomes impossible rather than merely rare."

---

## 3. Invalidation and consistency (Q16–Q22)

### Q16. Delete-on-write or update-on-write? — M

**A:** Delete. Update-on-write is tempting because the next reader gets a warm cache, but it loses under concurrency: two writers updating the cache out of order leave the older value cached, with no way to detect it. Deleting is idempotent — two deletes in either order leave the key absent, and the next reader repopulates from the authoritative store. The cost is a guaranteed miss on the next read, which is a latency blip rather than a correctness problem.

**The refinement worth stating:** delete-on-write has the stale-repopulation race (Q7), so the full answer is "delete, plus a version-stamped write so a late reader cannot resurrect a stale value". That combination is the correct one.

### Q17. What is event-driven invalidation, and when is it worth it? — H

**A:** Instead of application code deleting cache keys, the database's change stream — the WAL in Postgres, the binlog in MySQL, DynamoDB Streams — is consumed and turned into invalidation events. The advantages are real: it catches changes made by any writer, including direct database edits and migrations, not just the code paths that remember to invalidate; it decouples the writer from the cache; and because the stream is ordered per key, it can eliminate the out-of-order race. The costs are also real: it is another pipeline to operate, it introduces lag (typically tens to hundreds of milliseconds), and it requires the change stream to be enabled and retained.

**When it is worth it:** when there are many writers to the same entity, when data is changed outside the application, or when a derived store must be kept in sync ([see Cache Invalidation](../HLD/03-scaling/04-cache-invalidation-and-consistency.md) §3.4). For a single service with a single write path, direct invalidation is simpler and sufficient.

### Q18. Explain read-your-writes and how you implement it. — M

**A:** Read-your-writes is the guarantee that after you write, your own subsequent reads reflect the write. It breaks in two ways: a cache that has not been invalidated, and a read replica that has not yet applied the change. Implementation: for a bounded window after a write — typically 1–5 seconds — route that user's reads to the primary or to a session-pinned replica, keyed on the user ID from the session. The alternative is to record the write's log position and require a replica to have reached it before serving (a "monotonic read" token). The pragmatic and widely-used approach is the time window, because replication lag is small in normal operation and the window only needs to cover the tail.

**Probe: what does this cost?** Some fraction of reads go to the primary, which reduces the read-scaling benefit. Keeping the window small and applying it only to the user who wrote keeps that fraction low — a few percent at most.

### Q19. What consistency levels exist between "strong" and "eventual"? — H

**A:** A rough ladder, weakest to strongest:
- **Eventual** — replicas converge given no new writes.
- **Consistent prefix** — reads never see a write out of order (you never see an effect before its cause).
- **Monotonic reads** — a read never goes backwards; once you have seen a value, you never see an older one.
- **Read-your-writes** — you always see your own writes.
- **Monotonic writes** — a process's writes are applied in the order it issued them.
- **Linearizable** — every operation appears to take effect atomically at a single point between its invocation and response; the strongest and the most expensive.

The useful point: these are **per-session** guarantees, and they are much cheaper than global linearizability while covering almost every real user expectation. A user does not need everyone to see their post instantly; they need to see it themselves, and to not see it disappear and reappear.

### Q20. CAP and PACELC — give the correct answer. — M

**A:** CAP says that under a network partition you must choose consistency or availability; without a partition you can have both. The common misuse is treating it as a per-system label, which is wrong — it is per-operation and even per-request. The same database can be strongly consistent for a balance check and eventually consistent for a feed, and the correct design chooses per operation.

PACELC extends it usefully: if there is a Partition, choose Availability or Consistency; **Else** (in normal operation) choose Latency or Consistency. That second half is the one people forget, and it is the half that matters day to day — most of the time there is no partition, and the trade is that stronger consistency costs latency because it requires more coordination. So the answer to "is this system AP or CP?" is "which operation, and under what condition?"

**Strong:** "For a ride-hailing system: driver locations are PA/EL — available and low-latency, lossy — while the trip state machine is PC/EC — consistent and correct, and I will pay the latency." That answer shows CAP applied as a design tool rather than a vocabulary test.

### Q21. What is a quorum, and what does `W + R > N` actually buy you? — H

**A:** With N replicas, a write acknowledged by W and a read served by R: if `W + R > N`, every read quorum intersects every write quorum, so at least one replica in the read set has the latest write, and the read can return the newest value by comparing versions. With N=3, W=2, R=2 satisfies it. It buys **strong consistency without a leader**, at the cost of latency (you must wait for two replicas) and availability (you need two of three up). Setting W=1, R=1 gives speed and availability but reads can be stale. `W + R ≤ N` gives eventual consistency.

The caveat that matters: `W + R > N` guarantees you *see* the latest value, but it does not guarantee that concurrent writes are resolved consistently unless the resolution rule is deterministic — hence vector clocks or last-write-wins with a defined tiebreak. And a "sloppy quorum" with hinted handoff breaks the intersection guarantee during a partition, trading consistency for availability.

### Q22. How do you measure replication lag, and what do you do when it grows? — M

**A:** In Postgres, `pg_stat_replication` on the primary shows each standby's `replay_lag` and `write_lag`, and `pg_last_xact_replay_timestamp()` on the replica gives the timestamp of the last applied transaction, from which you compute the lag in seconds. In MySQL, `SHOW SLAVE STATUS` gives `Seconds_Behind_Master`, with the caveat that it is frequently null or misleading. The more reliable measure is a heartbeat: write a timestamped row to a table on the primary every second and read the newest one on the replica — the difference is the lag, and it works regardless of write volume.

When lag grows: route read-your-writes traffic to the primary (Q18); shed non-critical reads; check what is causing it — a long-running query on the replica, a big write transaction on the primary, a network bottleneck, or single-threaded replay (MySQL's classic problem, addressed by parallel replication, though it only parallelises within a transaction's dependency constraints); and alert on lag as a leading indicator, because by the time users notice staleness, the lag is already large.

---

## 4. Sharding and replication (Q23–Q29)

### Q23. When do you shard, and what comes first? — M

**A:** Shard last. The order is: optimise the query and add the right index; add a cache; add read replicas; partition large tables inside one database (declarative partitioning); vertically scale the primary. Sharding is the option with the highest operational cost — cross-shard queries, distributed transactions, resharding, and rebalancing — so it should be reached only when the write throughput or the data volume genuinely exceeds a single primary, or when the blast-radius benefit of isolating tenants is worth it on its own.

**Probe: how do you know you have exhausted the earlier options?** The primary's write throughput is saturated after batching and connection pooling; or the dataset no longer fits on one machine's storage or in its memory for the hot working set; or a single tenant's load is degrading everyone and isolation is needed. Each is a concrete, measurable trigger rather than a feeling.

### Q24. How do you choose a shard key? — H

**A:** The criteria, and the answer should list them: **high cardinality**, so shards are many and small; **even access distribution**, which is not the same as even key distribution — a key can be uniform in count and extremely skewed in traffic; **query alignment**, so the common query touches one shard rather than fanning out to all of them; **immutability**, because changing a shard key means moving the row; **not monotonic**, because a timestamp or auto-increment key sends all writes to the last shard; and **co-location**, so entities that are read together land on the same shard.

**Probe: give a shard key disaster.** Sharding by `user_id` for a social app and then needing "all posts in the last hour" — that query fans out to every shard. Or sharding a time-series by timestamp, so all writes hit one shard and the others idle. Or sharding by country, when one country is 60% of the traffic. The pattern in all three is: the shard key was chosen for storage balance and not for the access pattern.

### Q25. Compare sharding strategies. — M

**A:**

| Strategy | How | Strength | Weakness |
|---|---|---|---|
| **Range** | Ranges of the key (A–F, G–M) | Range scans are single-shard; easy to reason about | Monotonic keys create a hot shard; uneven ranges |
| **Hash** | `hash(key) mod N` | Even distribution | Range queries fan out; resharding moves almost everything |
| **Consistent hashing** | Keys and nodes on a ring, 100–1,000 virtual nodes per node | Adding a node moves only ~1/N of keys | Still no range scans; virtual nodes needed for balance |
| **Directory / lookup** | A service maps key → shard | Fully flexible; supports moves | The directory is a dependency and a potential bottleneck |
| **Geographic** | Shard by region | Data residency; local latency | Uneven regions; cross-region operations are hard |
| **Composite** | Shard by tenant, then by time within the tenant | Isolation plus locality; the common production answer | More complex routing |
| **Fixed logical shards** | 1,024–4,096 logical shards mapped to physical nodes | Moving a logical shard is fast; rebundling is easy | Requires the mapping layer; logical shards can be uneven |

**The production answer:** hash or composite for distribution, but **fix the number of logical shards high and map them to physical nodes** so that resharding is a mapping change rather than a data movement. That is the design that makes the "how do you reshard" question easy.

### Q26. How do you reshard without downtime? — H

**A:** Four phases. (1) **Dual-write** — write to both the old and new shards, with the new one authoritative for the moved range's reads only after a backfill. (2) **Backfill** — copy historical data in the background, tracked by a position marker so it can resume. (3) **Shadow-read and verify** — read from both, compare, and log mismatches until they reach zero; this is the phase people skip and it is the one that catches the bugs. (4) **Cutover per range** — move reads to the new shard one range at a time, so a failure affects one range rather than everything, and keep the dual-write until the range is stable before decommissioning the old copy. The semantic traps during a migration are collation differences, NULL ordering, type coercion, transaction-scope changes and sequence collisions — all of which produce silently different results rather than errors.

**Strong:** "The shadow-read comparison phase is the part that makes this safe. Without it you are migrating data and hoping."

### Q27. Compare replication topologies. — M

**A:**

| Topology | How | Use | Cost |
|---|---|---|---|
| **Single-leader** | All writes to one primary, replicated to followers | The default; simple and correct | The primary is a write bottleneck and a failover risk |
| **Multi-leader** | Writes accepted at multiple leaders, conflicts resolved | Multi-region writes, offline-capable clients | Conflict resolution is genuinely hard — you need a defined rule |
| **Leaderless** | Any replica accepts writes; quorums | High availability, no failover step | Read repair, hinted handoff, and weaker guarantees |
| **Synchronous** | The primary waits for replicas | No data loss on failover | Write latency and availability depend on replicas |
| **Semi-synchronous** | Wait for one replica of N | A middle ground — one durable copy | Still a latency cost; the "one" must be reachable |
| **Asynchronous** | Do not wait | Fast writes | **You lose the tail on failover** |

**The trade to state:** synchronous replication converts a durability guarantee into a latency and availability cost. In Postgres, `synchronous_commit = remote_apply` means the standby has applied the transaction, which is the strongest and slowest; `on` means it has been flushed to the standby's WAL, which is the usual choice; `off` risks losing roughly the last 200 ms on a crash even without a failover.

### Q28. What is split brain, and how do you prevent it? — H

**A:** Split brain is when a partition leaves two nodes both believing they are the primary. Both accept writes; the two histories diverge; when the partition heals, one must be discarded, losing data. Prevention requires a quorum-based election — the new primary must be elected by a majority, so a minority partition cannot elect one, which is what Raft and Paxos provide. And critically, **fencing**: even with an election, the old primary may still be running and accepting writes without knowing it has been deposed. The defence is a fencing token — a monotonically increasing number issued with the leader's lease — and every write to shared storage carries it, so the storage layer rejects a write whose token is older than the last one it saw. That is what makes "the old primary cannot corrupt the data" true rather than hopeful.

**Probe: what is the failover data-loss window with async replication?** The replication lag at the moment of failure — anything the old primary committed but had not yet shipped is gone. That is why the failover loss window is exactly the replication lag, and why monitoring lag is not just a read-staleness concern.

### Q29. A shard is hot. What do you do? — P

**A:** First, confirm it is the shard and not the node — if the hot shard shares a node with cold shards, the fix may be rebalancing the mapping rather than splitting the key. Then, in order of increasing cost: (1) **isolate the hot entity** onto its own shard or its own node, so it cannot degrade its neighbours; (2) **split the key** — for a counter, shard into N sub-counters summed on read; for a hot tenant, split their data by a secondary dimension; (3) **add a cache in front of the hot shard's reads**, which is usually the real fix because a hot shard is often a hot *read* path; (4) **replicate the hot shard for reads**; (5) **change the shard key**, which is a reshard and therefore last.

**Strong:** "The most common cause of a hot shard is that the shard key was chosen for storage balance, not access balance — and the cheapest fix is usually a cache in front of the read path, not a reshard."

---

## 5. SQL, indexes and schema (Q30–Q36)

### Q30. How does an index actually make a query faster? — M

**A:** By turning a scan into a search. Without an index, finding the rows matching a predicate means reading every row — the whole table. With a B-tree index, the database descends a tree of typically 3–4 levels even for hundreds of millions of rows, which is a handful of page reads, and then follows the leaf-level linked list for range scans. The costs are on the write side: every insert, update and delete must maintain the index, so each index adds write amplification and storage. And an index on a low-selectivity column is nearly useless, because if the predicate matches 40% of the table the planner will correctly choose a sequential scan anyway — the practical threshold is around 5–10%.

**Probe: why is a B-tree 3–4 levels for 10⁸ rows?** Because each page holds hundreds of keys; with a branching factor of ~500, three levels address 500³ = 125 million entries. That is why the height barely grows with the table, and why index lookups stay fast while table scans do not.

### Q31. What is the leftmost-prefix rule? — M

**A:** In a composite index on `(a, b, c)`, the index is sorted by `a`, then `b` within `a`, then `c` within `(a, b)`. So a query can use the index if it constrains a prefix: `WHERE a = ?`, or `a = ? AND b = ?`, or all three. It cannot use it efficiently for `WHERE b = ?` alone, or `WHERE c = ?`, because the leading column is unconstrained and the index's ordering says nothing about `b` globally. Range conditions behave the same way: `a = ? AND b > ?` uses both, but `a > ? AND b = ?` uses `a` for the range and cannot use `b` for the seek.

**Strong:** "The practical consequence is that column order in a composite index is a design decision driven by the queries, not by the schema — the equality columns first, then the range column, then any covering columns in an `INCLUDE`."

### Q32. How do you read `EXPLAIN ANALYZE`? — M

**A:** I check six things, in order. (1) **Estimated versus actual rows** — a large discrepancy means stale statistics (`ANALYZE`) or a correlation the planner does not know about, and every downstream cost estimate is then wrong. (2) **The scan type** — a sequential scan on a large table with a selective predicate is usually a missing index or a non-sargable predicate. (3) **The join algorithm** — nested loop is right for small outer sets, hash join for large unsorted sets, merge join for already-sorted large sets; a nested loop over a large outer set is the classic bad plan. (4) **Buffers** — `EXPLAIN (ANALYZE, BUFFERS)` shows shared hits versus reads, which distinguishes "the query is slow because it does work" from "the query is slow because it is going to disk". (5) **The estimated cost of the top node** versus the actual time, to see whether the planner's model matches reality. (6) **Sort and aggregate nodes** — an external sort (spilling to disk) or a large hash aggregate tells you memory settings (`work_mem`) are too small, or that the query is returning far more rows than it should.

### Q33. What makes a predicate non-sargable? — H

**A:** Anything that prevents the database from using the index ordering to seek. The common cases: wrapping the indexed column in a function — `WHERE YEAR(created_at) = 2026` cannot use an index on `created_at`, but `WHERE created_at >= '2026-01-01' AND created_at < '2027-01-01'` can; a leading wildcard — `LIKE '%foo'` cannot use a B-tree, though `LIKE 'foo%'` can; an implicit type conversion, which is the sneakiest, because `WHERE user_id = '123'` on an integer column in some databases forces a cast that discards the index; arithmetic on the column; and `OR` across different columns, which often prevents index use unless the database can rewrite it as a union. The fixes are expression indexes, a rewritten predicate, or a partial index that matches the actual query shape.

### Q34. When would you denormalise? — M

**A:** When the join cost or the query complexity has become the bottleneck, and the read volume justifies the write-side complexity. The specific situations: a hot read path with an expensive multi-table join on every request; a feed or timeline that needs a materialised ordering; a reporting query that would otherwise scan. The costs are real and must be stated: redundancy, so an update has to touch multiple places; the possibility of inconsistency, especially when a denormalised copy is updated by a different code path; and greater storage. The safest form is a **materialised view or a derived table maintained by CDC**, because then the denormalisation is explicit, rebuildable, and derivable from the source of truth — rather than being an extra column that application code must remember to keep in sync.

### Q35. Compare OLTP and OLAP storage. — M

**A:**

| | OLTP (row store) | OLAP (columnar) |
|---|---|---|
| Layout | Rows stored together | Columns stored together |
| Optimised for | Point reads and writes of whole rows | Scans and aggregations over few columns of many rows |
| Compression | Poor — mixed types per page | Excellent — same type adjacent, and often sorted so run-length and delta encoding work |
| Typical query | `SELECT * FROM orders WHERE id = ?` | `SELECT region, SUM(amount) FROM orders GROUP BY region` |
| Write pattern | Many small writes | Bulk loads; updates are expensive |
| Examples | Postgres, MySQL, Aurora | ClickHouse, BigQuery, Redshift, DuckDB |

The reason columnar wins for analytics: a query touching 3 of 50 columns reads only those 3 columns from disk, whereas a row store reads all 50. Combined with compression ratios of 5–10× and vectorised execution, the difference is often two orders of magnitude. And the reason you do not run analytics on the OLTP primary is that a large analytical scan evicts the hot working set from the buffer pool and degrades the transactional workload — which is why CDC into a warehouse is the standard pattern.

### Q36. Your query got slow overnight. Nothing changed in the code. — P

**A:** Nothing changed in the code, so something changed in the data, the statistics, or the environment. In order of likelihood: (1) **the data grew past a threshold** — a plan that was a nested loop over 1,000 rows is fine and catastrophic over 1 million, and the planner flipped to a bad plan as the table crossed a size boundary; (2) **statistics went stale**, so the planner's row estimates are wrong and it chose a bad plan — `ANALYZE` fixes it, and autovacuum tuning or a manual analyze after bulk loads prevents it; (3) **a parameter-sensitive plan** — the same prepared statement with a different parameter value needs a different plan, and a generic plan that was cached is wrong for the new value (this is the classic "plan regression" and the fix is plan hinting, `plan_cache_mode`, or splitting the query); (4) **bloat** — heavy update traffic leaves dead tuples, so the table has far more pages than it needs and scans read more; (5) **a lock queue** — a long-running transaction holds a lock and everything queues behind it, which looks like the query being slow rather than blocked; (6) **resource contention** — another workload started on the same host.

**Strong:** "The single most useful artifact is a comparison of the plan now against the plan from before, because it converts 'it got slow' into either 'the same plan got slower', which is data or resources, or 'the plan changed', which is statistics or a threshold."

---

## 6. NoSQL, vectors and object storage (Q37–Q42)

### Q37. SQL or NoSQL — how do you actually decide? — M

**A:** By the access pattern, not by preference. The questions in order: what are the queries (not the entities — the queries)? What consistency does each operation need? What is the write and data volume? How stable is the query set? And how much operational burden can the team carry?

The honest framing: a relational database with the right indexes handles the overwhelming majority of workloads, including most of what people reach for NoSQL for — Postgres with `jsonb` and GIN indexes covers flexible schemas, and it is the correct default. NoSQL earns its place when a specific access pattern is the whole product and the relational model's generality is pure overhead: a key-value lookup at extreme volume with a predictable key; a wide-column time-series where the partition is the access path; a document store where the document boundary is the unit of consistency and the query. The failure mode of the opposite kind is equally real — a team chooses a relational database for a document-shaped workload and then fights the schema, or chooses a key-value store and then needs a query the key cannot answer.

**Strong:** "Query-first modelling: write down the queries, then choose the store that answers them. If you cannot write down the queries, you are not ready to choose a database."

### Q38. Why is a "hot partition" a problem in DynamoDB and Cassandra? — M

**A:** Because the partition is both the unit of storage and the unit of throughput, and a partition lives on one node. In DynamoDB, a partition supports roughly 1,000 write capacity units and 3,000 read capacity units per second and about 10 GB — so if one partition key receives more traffic than that, the requests throttle regardless of how much total capacity the table has. In Cassandra the practical ceilings are around 100 MB and 100,000 rows per partition; a larger partition causes compaction pressure, repair problems, and slow reads.

The fixes: choose a high-cardinality partition key; add a **bucket suffix** to spread a hot key across partitions (`date` for time-series, a hash for a counters key); isolate the notorious hot entities onto their own table or shard; and in DynamoDB, be careful with global secondary indexes, because the GSI's partition key is a new partition-key choice and can reintroduce the same problem.

### Q39. Explain vector search and where it fits in a system. — M

**A:** A vector database stores embeddings — dense float arrays — and answers "which vectors are most similar to this one", which is the retrieval step in RAG and the candidate-generation step in recommendations. The mechanics: embeddings from a model, an index built for approximate nearest-neighbour search (HNSW is the common default; IVF-PQ for large corpora with memory constraints), a distance metric (cosine for normalised text embeddings), and a query embedding that is searched against the index for the top-k. The scaling reality: 1,536 dimensions at 4 bytes per float is 6 KB per vector, so 100 million vectors is 600 GB before the index, which is why quantisation — int8, or binary — and disk-based indexes like DiskANN exist. The index typically costs 1.5–2× the raw vectors in RAM for HNSW, which is why memory, not compute, is the binding constraint.

**Probe: the vectors are only half the data. What else do you need?** Metadata. Retrieval almost always needs a filter — by tenant, by date, by permission, by document type — and naive post-filtering can collapse the result set to nothing. Pre-filtering on a non-indexed attribute is 10–100× slower. The robust answer is to make the filter a partition key, so the search is scoped to a shard rather than filtered after the fact ([see Vector Databases](../HLD/04-data/03-vector-databases.md) §3.8).

### Q40. Your RAG system returns irrelevant results. Where do you look? — H

**A:** In order of impact. (1) **Retrieval quality before generation** — measure recall@k against a labelled set; if the right document is not in the top-k, no prompt engineering will save it. (2) **Chunking** — chunks that are too small lose context, too large dilute the embedding; 200–1,000 tokens with 10–20% overlap is a reasonable starting point, but the structure matters more — splitting mid-sentence or mid-table is worse than splitting on document structure. (3) **The embedding model** — mismatched models between indexing and querying is a hard bug, and a model that is not trained on the domain may simply not separate the concepts you care about. (4) **The metric** — cosine on un-normalised vectors behaves differently from dot product; mixing them silently degrades results. (5) **Missing reranking** — retrieving 100 candidates and reranking with a cross-encoder to the top 5 typically improves precision noticeably. (6) **The filter problem** — a metadata filter applied after retrieval can empty the results, or a filter applied before can exclude everything.

**Strong:** "The diagnostic order is retrieval first, because generation quality is bounded by retrieval quality — and recall@k is the metric that tells you which one you have a problem with."

### Q41. Block, file, or object storage — how do you choose? — E

**A:** It depends on the access pattern. **Block storage** presents a raw device — a volume you format with a filesystem — and gives the lowest latency and the most control; use it for a database, a boot volume, anything needing random access in place. **File storage** presents a shared filesystem with POSIX semantics, so many machines can mount the same directory; use it for shared content, legacy applications and lift-and-shift workloads. **Object storage** presents a flat namespace of immutable objects addressed by key, accessed over HTTP; use it for anything write-once-read-many at a large scale — media, backups, model weights, logs, data lake. Object storage is far cheaper per gigabyte and effectively unlimited, at the cost of higher per-operation latency and no in-place mutation.

**Probe: why is object storage hard to use as a filesystem?** No rename (a rename is a copy plus a delete), no in-place append (an append is a new object), no partial reads for a partial object unless you use range requests, eventual consistency historically, and per-request costs that punish small-file workloads. Every "S3 as a filesystem" system reimplements a metadata layer to get around exactly these.

### Q42. When do you move data to a cheaper storage tier? — M

**A:** When the access frequency does not justify the price. The mechanics: object storage tiers trade a lower storage price for an earlier-deletion penalty and a retrieval cost — infrequent access tiers typically require a 30-day minimum, and archive tiers 90 or 180 days, with retrieval times from milliseconds to hours. The transition threshold is normally driven by object size, because the per-object transition cost is fixed and on a small object it exceeds the storage saving — around 128 KB is the usual break-even. So lifecycle policies transition only objects above that size after N days without access. And the honest point is that most data is cold: in a large media or analytics system the overwhelming majority of bytes are never re-read, which is why tiering is not an optimisation but the only way the bill is payable.

---

## 7. Distributed IDs, queues and resilience (Q43–Q45)

### Q43. How do you generate unique IDs in a distributed system? — M

**A:** Four options with distinct trades.

| Approach | How | Trade |
|---|---|---|
| **UUIDv4** | Random 128-bit | No coordination, but random insertion order destroys B-tree locality and it is twice the size of a bigint |
| **UUIDv7 / ULID** | Timestamp prefix plus randomness | Sortable and index-friendly, no coordination; leaks creation time; still 128-bit |
| **Snowflake** | 41 bits timestamp + 10 bits machine + 12 bits sequence | 64-bit, sortable, ~4 M IDs/sec per machine; needs clock synchronisation and machine-ID assignment |
| **Database sequence** | `nextval` | Simple and monotonic within one database; a bottleneck and a coordination point across shards |
| **Segment / range allocation** | A coordinator hands out blocks of IDs | Cheap and fast; a coordinator dependency and IDs jump across machines |

The traps: **clock skew** breaks Snowflake — a machine whose clock jumps backwards can issue duplicate IDs, and the standard fix is to refuse to issue IDs until the clock catches up or to use a logical clock; **UUIDv4 as a primary key in a B-tree** produces random inserts, which causes page splits and index bloat, which is why UUIDv7 exists; and a **sequence across shards** requires either a global allocator or per-shard prefixes.

### Q44. When do you use a queue, and how do you make consumers reliable? — M

**A:** A queue when the producer should not wait for the consumer: to absorb a burst, to decouple availability, to retry independently, or to fan out to multiple consumers. Reliability comes from four things: **acknowledge after processing, not before**, so a consumer crash re-delivers rather than loses; **idempotent processing**, because at-least-once delivery means duplicates are certain, typically via a dedupe key stored with a TTL; **a dead-letter queue after bounded attempts**, with alerting and a replay tool, so failures are visible rather than silent; and **retry with exponential backoff and jitter**, so a recovering downstream is not immediately knocked over again.

**Probe: how is Kafka different from a traditional queue?** Kafka is a partitioned, ordered, retained log, not a queue that deletes on acknowledgement. Consumers track their own offset, which means messages can be replayed, multiple consumer groups can read the same topic independently, and ordering is guaranteed per partition but not across partitions. That ordering constraint is why the partition key is a real design decision — it decides what is ordered together.

### Q45. Explain circuit breakers, and how do you avoid a cascade. — H

**A:** A circuit breaker wraps a call to a dependency and, after a threshold of failures, stops calling it and fails fast instead. Three states: closed (calls pass), open (calls fail immediately without attempting), half-open (a limited number of probes to test recovery). Without it, a slow dependency consumes all the caller's threads or connections waiting on timeouts, and the caller becomes unavailable too — that is the cascade. The accompanying mechanisms that matter as much as the breaker: **timeouts on every network call**, because a call with no timeout is an unbounded resource commitment; **bulkheads**, separate thread or connection pools per dependency so one slow dependency cannot exhaust the pool used by others; **bounded retries with jitter**, because an unbounded retry storm turns a blip into an outage; and **load shedding**, dropping low-priority work rather than degrading all of it.

**Probe: if you retry a payment call three times and it timed out each time, what is the risk?** The payment may have succeeded each time. That is why the retry must carry an idempotency key, so a re-attempt returns the original result rather than charging again — retries and idempotency are the same design decision seen from two sides ([see Notification System](../Case-Studies/HLD/03-notification-system.md) §3.5).

---

## 8. Rapid-fire numbers

| Thing | Number | Note |
|---|---|---|
| 1 M requests/day | ~12 QPS | The fastest estimate to have memorised |
| Peak multiplier | ×2 to ×5 | Diurnal, over the average |
| 99.9% availability | 43 min/month | 99.99% is 4.3 min/month |
| B-tree height for 10⁸ rows | 3–4 levels | Branching factor ~500 |
| Seq-scan threshold | 5–10% selectivity | Below it an index wins, above it a scan |
| Default cache hit ratio to aim for | 80%+ | Below that the cache costs more than it saves |
| Redis `maxmemory` eviction | `allkeys-lru` or `volatile-lru` | Know which you have set |
| Single-flight bounded wait | 50–100 ms then fall through | Unbounded waiting is a latency cliff |
| XFetch `beta` | 1.0 | Probabilistic early expiry |
| TTL jitter | ±10% | Prevents synchronised expiry |
| Read-your-writes window | 1–5 s | Route to the primary |
| Postgres async commit loss window | ~200 ms | `synchronous_commit = off` |
| `rpl_semi_sync_master_timeout` | 10,000 ms | MySQL falls back to async after this |
| Quorum | `W + R > N` | N=3, W=2, R=2 |
| DynamoDB per partition | ~1,000 WCU / 3,000 RCU / 10 GB | The hot-partition ceiling |
| Cassandra partition ceiling | < 100 MB, < 100k rows | Practical guidance |
| Logical shards | 1,024–4,096 | Fixed, then mapped to physical nodes |
| Consistent-hashing virtual nodes | 100–1,000 per node | Needed for balance |
| Snowflake ID | 41b ts + 10b machine + 12b seq | ~4 M IDs/sec/machine |
| ULID / UUIDv7 | 128-bit, time-ordered | Index-friendly, unlike UUIDv4 |
| Vector storage | 1,536 dims × 4 B = 6 KB | 100 M vectors ≈ 600 GB raw |
| HNSW index overhead | ×1.5–2.0 in RAM | Memory is the binding constraint |
| `efSearch` starting point | 10 × k | `efSearch < k` is a bug |
| Object tiering break-even | ~128 KB | Per-object transition cost |
| Infrequent-access minimum | 30 days | Archive tiers 90–180 days |
| Idempotency key TTL | 24 h | Matches typical client retry windows |
| Circuit breaker threshold | ~50% failure over 20+ calls | Then open, with a half-open probe |
| Every network call needs | A timeout | No timeout is an unbounded commitment |

---

## 9. Cheat-sheet summary

| Concept | One-line truth |
|---|---|
| Load balancing default | Least connections; consistent hashing when there is per-server state |
| Health checks | Shallow for liveness, dependency-aware for readiness, with hysteresis |
| Session affinity | A smell; externalise the session instead |
| Cache-aside | The default — degrades to slow, not broken, when the cache fails |
| Delete-on-write | Delete, not update; two deletes in any order leave the key absent |
| Stale repopulation | Delete alone loses; version-stamped writes close the race |
| Stampede | Single-flight + TTL jitter + XFetch + refresh-ahead |
| Hot key | Local in-process cache first, then key replication, then replicas |
| Redis lock | An optimisation, never a correctness mechanism — use fencing or CAS |
| Invalidation by CDC | Catches writers your code does not know about; costs a pipeline |
| Consistency ladder | Eventual → prefix → monotonic → read-your-writes → linearizable |
| CAP | Per operation, not per system; PACELC adds the latency-vs-consistency trade |
| Quorum | `W + R > N` intersects; sloppy quorums do not |
| Shard last | Index → cache → replicas → partition → shard |
| Shard key | Cardinality, access balance, query alignment, immutability, not monotonic |
| Resharding | Dual-write → backfill → shadow-read → cutover per range |
| Split brain | Quorum election plus fencing tokens |
| Failover loss | Exactly the replication lag |
| Index cost | Read speed paid for in write amplification and storage |
| Leftmost prefix | Composite indexes only serve prefix constraints |
| Non-sargable | Functions, casts, and leading wildcards discard the index |
| OLTP vs OLAP | Row store for points, columnar for scans; never analytics on the primary |
| SQL vs NoSQL | Query-first; Postgres is the default and usually right |
| Vector search | Memory-bound; filtering is the hard part, so make it the partition |
| Storage tiers | ~128 KB break-even, 30/90/180-day minimums |
| Distributed IDs | Snowflake for sortable 64-bit; UUIDv7 over UUIDv4 for index locality |
| Queues | Ack after processing, idempotent consumers, DLQ, jittered retries |
| Circuit breakers | Fail fast + timeouts + bulkheads + bounded retries + load shedding |

**"If the interviewer says X → answer Y"**

| Interviewer says | You say |
|---|---|
| "Add a cache" | "Cache-aside with delete-on-write plus versioned entries, and a single-flight miss path" |
| "Cache is stale" | "Is it a missed invalidation, the repopulation race, or a TTL too long for the change rate?" |
| "Cache node died" | "1/N of keys miss; the risk is the hot key that lands on one database row" |
| "One key is hot" | "In-process local cache first — it removes the hop rather than spreading it" |
| "Shard the database" | "Only after index, cache, replicas and partitioning; then pick the key by access pattern" |
| "Choose a shard key" | "High cardinality, even *access*, query-aligned, immutable, not monotonic" |
| "Reshard with no downtime" | "Dual-write, backfill, shadow-read and verify, cut over per range" |
| "Split brain" | "Quorum election plus fencing tokens, so the old primary cannot write" |
| "CAP?" | "Per operation — and PACELC's else-branch is the latency-versus-consistency trade" |
| "SQL or NoSQL?" | "Query-first. Postgres is the default; NoSQL when the access pattern *is* the product" |
| "Query got slow overnight" | "Compare plans: same plan means data or resources; a new plan means statistics or a size threshold" |
| "Vector search?" | "Memory-bound; the hard part is the metadata filter — make it the partition key" |
| "Retry the payment" | "Only with an idempotency key — a timeout may mean it already succeeded" |
| "Trade-offs?" | "Every scaling choice trades consistency, latency, cost, or operational complexity — name which" |

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
- [Networking and API Questions](02-networking-and-api-questions.md)
- [Scaling, Caching and DB Cheatsheet](../Cheat-Sheets/03-scaling-caching-db-cheatsheet.md)
