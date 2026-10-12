# Case Study — URL Shortener (The Flagship Worked Design)

> **Source of truth:** *AI System Design Interview | Most Asked FAANG System Design Question* (the complete URL shortener walkthrough: requirements → estimation → architecture → tool diagram → trade-offs).

> **Why this is the flagship case study:** it is the canonical "medium difficulty" question that every interviewer asks first, and it contains one instance of nearly every concept in this knowledge base — caching, CDN, load balancing, rate limiting, NoSQL, replication, async event pipelines and the encoding trade-off. Master it and you have a template for the rest.

---

## TL;DR

A URL shortener maps a long URL to a short code and redirects on lookup. Underneath, it is an **extremely read-heavy key-value system**: at 100M DAU the target load is ~1,000 writes/sec and ~100,000 reads/sec — a **100:1 read:write ratio** — which makes the entire architecture a **cache in front of a key-value store**, not a database problem. The short code comes from encoding (base62 is the default answer; sequential IDs are predictable, hashing needs collision handling, distributed IDs avoid coordination). Analytics must be **asynchronous** — the redirect path cannot wait on a click counter, so click events go through Kafka to a stream processor and an analytics store. The hard parts are: collisions in code generation, uniqueness for custom aliases (needs a conditional write), cache stampede on a hot key, and the 301-vs-302 decision that determines whether your click analytics work at all.

> **60-second interview answer:** "I'd clarify requirements first — create, redirect, custom aliases, expiration, click tracking, delete — and the non-functional numbers: 100M DAU, 1,000 writes/sec, 100,000 redirects/sec, 99.9% availability, codes unique and unpredictable. That's 86.4M URLs/day and a 100:1 read:write ratio, so this is a cache-first design: Redis in front of a key-value store, sharded by short code. I'd generate codes by base62-encoding a distributed ID — 7 characters gives 3.5 trillion codes — with a conditional write to catch collisions, and a Bloom filter or pre-generated pool to avoid a read-before-write on the hot path. The redirect service reads Redis, falls through to NoSQL on a miss, and fires a click event to Kafka so analytics never blocks the redirect. Reads go to replicas; the write path stays on the primary. And I'd use 302 rather than 301 unless analytics don't matter, because browsers cache 301s permanently."

---

## Step 1 — Clarify requirements

### Functional requirements (the six, verbatim from the walkthrough)

| # | Requirement | What it forces into the design |
|---|---|---|
| 1 | **Create a short URL from a long URL** | Code generation, collision handling, write path |
| 2 | **Redirect a short URL to the original long URL** | The read path — this is the hot path |
| 3 | **Custom alias support** | **Uniqueness check** — requires a consistency guarantee, not eventual |
| 4 | **URL expiration** | An `expiresAt` field; check on read; no redirect if expired |
| 5 | **Track click count + analytics** | An event pipeline, separate from the redirect path |
| 6 | **Delete or disable URLs** | Soft-delete flag, or TTL-based removal |

### The clarifying questions to ask (and what each one decides)

| Question | What it decides |
|---|---|
| How many DAU? | Scale of everything — assume 100M |
| What's the read:write ratio? | Whether the design is cache-first (100:1 → yes) |
| Do we need custom aliases? | Whether strong consistency enters the write path |
| Does a URL expire, and who sets it? | Schema, cleanup job, read-time validation |
| Do we need click analytics, and how fresh? | Whether an async pipeline exists (yes; seconds of lag is fine) |
| Must the short code be unpredictable? | Rules out pure sequential IDs (enumeration attack) |
| How long do we store mappings? | ~90 TB over 5 years; tiering policy |
| Is a redirect allowed to be eventually consistent? | Almost always yes — huge design freedom |
| Do we need to prevent abuse (spam, phishing)? | Rate limiting, domain blocklists, scanning |

> **The one that changes everything:** *"Must the short code be unpredictable?"* If yes, sequential IDs are out — anyone could enumerate `sho.rt/1`, `sho.rt/2`, and scrape every URL in the system. The walkthrough states it explicitly: **the short code must be unique and unpredictable.**

### Non-functional requirements

| Category | Target | Source |
|---|---|---|
| **Scalability** | 100M DAU; scale to 1B users | Walkthrough |
| **Write throughput** | 1,000 URL creations/sec | Walkthrough |
| **Read throughput** | 100,000 redirects/sec | Walkthrough |
| **Availability** | 99.9%+ | Walkthrough (states 99%) |
| **Code property** | Unique **and unpredictable** | Walkthrough |
| **Latency** | Redirect p99 < 100 ms (assume) | Reasonable default |
| **Durability** | No mapping lost once created | Implied |
| **Read consistency** | Eventual is fine for redirects | Implied by read path |
| **Write consistency** | **Strong for custom aliases** | Walkthrough: "requires a consistency check" |

> **Note the tension:** the walkthrough explicitly says custom alias creation *"requires a consistency check as well"* — two users must not get the same alias. That single sentence splits the consistency model: the **generated-code path** can tolerate eventual consistency, but the **custom-alias path cannot**. Naming that split out loud is a strong seniority signal.

---

## Step 2 — Capacity estimation

### The walkthrough's arithmetic

```
Writes:  1,000 URL creations/sec
         × 86,400 s/day
         = 86,400,000 ≈ 86.4 MILLION URLs per day

Reads:   100,000 redirects/sec
         × 86,400
         = 8.64 BILLION redirects per day

Ratio:   100 : 1   (read-heavy)
```

### Full estimate with storage

| Quantity | Calculation | Result |
|---|---|---|
| Writes/day | 1,000 × 86,400 | **86.4M URLs/day** |
| Write QPS | given | **1,000/sec** |
| Peak write QPS | ×3 | **3,000/sec** |
| Reads/day | 100,000 × 86,400 | **8.64B redirects/day** |
| Read QPS | given | **100,000/sec** |
| Peak read QPS | ×3 | **300,000/sec** |
| **Read:write ratio** | 100,000 ÷ 1,000 | **100:1** |
| Record size (URL + code + metadata) | ~500 B | |
| Storage/day | 86.4M × 500 B | **~43 GB/day** |
| Storage/year | 43 GB × 365 | **~15.8 TB/year** |
| 5-year raw | 15.8 TB × 5 | **~79 TB** |
| With index overhead (1.5×) | 79 × 1.5 | **~118 TB** |
| With 3× replication | 118 × 3 | **~355 TB** |
| Bandwidth (reads) | 100,000 × 500 B | **~50 MB/s** (150 MB/s peak) |
| Code space | 62⁷ | **3.5 × 10¹² codes** |

### The architectural conclusions — the part that scores

| Number | Conclusion |
|---|---|
| **100:1 read:write** | This is a **read-path problem**. Optimise the redirect, not the creation. |
| **300,000 peak reads/sec vs a DB's ~5,000 QPS** | A cache is **mandatory** — the gap is **60×**. Not "nice to have." |
| **3,000 peak writes/sec** | One primary absorbs it — **no sharding needed for writes at launch** |
| **~355 TB over 5 years (replicated)** | Needs distributed storage + a tiering/expiry policy |
| **50 MB/s** | Bandwidth is a non-issue; no CDN strictly needed for bandwidth (see caveat below) |
| **3.5T code space vs ~158M URLs over 5 years** | Collision probability is negligible — roughly 1 in 22,000, so collision handling is a correctness safeguard, not a scaling concern |
| **~80/20 access distribution** | 20% of links get 80% of clicks → the cache hit rate will be very high |

### Why the CDN appears (and the honest caveat)

The walkthrough's architecture puts a CDN first, reasoning: *"What if you are in Australia and using an Indian server? Does it make sense to go to the Indian server for every request?"* — CDNs cache locally and serve from the edge.

**The honest engineering read:** for a URL shortener the payload is **tiny** (a redirect is a few hundred bytes of headers), so a CDN is not a bandwidth play. It is a **latency** play — terminating the redirect at the edge avoids a cross-continent round trip for a mobile user. It also absorbs bot traffic and repeated clicks on the same link. **Say this distinction out loud** — it demonstrates you understand *why* a component is there, not just that it appears in a reference architecture. And note the caveat: a CDN caching a **302** response changes your click-analytics fidelity, because edge-cached responses may never reach your origin.

---

## Step 3 — API design

```
POST /v1/urls
  Body:    { "longUrl": "https://example.com/very/long/path?q=1",
             "customAlias": "my-link",        // optional
             "expiresAt": "2027-01-01T00:00:00Z" }  // optional
  Returns: 201 { "shortUrl": "https://sho.rt/aB3xK9",
                 "shortCode": "aB3xK9",
                 "expiresAt": "..." }
  Errors:  400 invalid URL · 409 alias taken · 429 rate limited

GET /{shortCode}
  Returns: 302 Location: https://example.com/very/long/path?q=1
  Errors:  404 unknown code · 410 gone (expired/deleted)

GET /v1/urls/{shortCode}
  Returns: 200 { longUrl, createdAt, expiresAt, clickCount, status }
  (owner-only — for management)

GET /v1/urls/{shortCode}/analytics?from=&to=
  Returns: 200 { clicks, uniqueVisitors, referrers, countries, timeseries }

DELETE /v1/urls/{shortCode}
  Returns: 204 No Content
```

### The API decisions that matter

| Decision | Choice | Why |
|---|---|---|
| **Redirect status code** | **302 Found** (default) | Not cached permanently by browsers → **click analytics work**. **301** would be cached forever, losing repeat-click data. Use 301 only if analytics don't matter. |
| **Custom alias conflict** | **409 Conflict** | Distinct from 400 — signals a valid request that can't be satisfied |
| **Expired link** | **410 Gone** | Semantically correct: it existed and is permanently gone. A 404 would be wrong. |
| **Analytics endpoint** | Separate, owner-authenticated | Keeps the write path clean and the data access-controlled |
| **Delete** | Soft delete + 204 | Enables undo and preserves analytics history |
| **Rate limit response** | 429 + `Retry-After` | Standard; prevents abuse |
| **Auth** | API key / bearer for creation; anonymous for redirects | Redirects must be public and fast |

> **The 301 vs 302 answer is a classic probe.** Say: *"I'd use 302 by default because 301 is cached permanently by browsers and the CDN, which destroys repeat-click analytics — and this system's requirement #5 is click tracking. If analytics didn't matter, 301 would be better for cacheability and load reduction."*

---

## Step 4 — High-level design

### The architecture (the walkthrough's own diagram, rendered)

```
                                                  ┌──────────────┐
  ┌────────┐        ┌─────┐        ┌──────────────┐│  Monitoring  │
  │ Client │──HTTPS▶│ CDN │───────▶│Load Balancer │└──────┬───────┘
  │(browser│        │edge │        │              │       │
  │ mobile)│        │cache│        └──────┬───────┘       │
  └────────┘        └─────┘               │               │
                                          ▼               │
                                  ┌───────────────┐       │
                                  │  API Gateway  │       │
                                  └───────┬───────┘       │
                                          ▼               │
                                  ┌───────────────┐       │
                                  │ Rate Limiter  │       │
                                  └───────┬───────┘       │
                        ┌─────────────────┴──────────────┐│
                        ▼                                ▼│
              ┌──────────────────┐            ┌───────────────────┐
              │  URL Service     │            │ Redirect Service  │
              │ (create/shorten) │            │   (the hot path)  │
              └────────┬─────────┘            └─────────┬─────────┘
                       │                                │
                       │                    ┌───────────┴───────────┐
                       │                    ▼                       ▼
                       │            ┌───────────────┐      ┌────────────────┐
                       │            │ Redis Cache   │      │  Kafka         │
                       │            │  (LRU + TTL)  │      │ click events   │
                       │            └───────┬───────┘      └───────┬────────┘
                       │                    │ cache miss           ▼
                       │                    ▼              ┌────────────────┐
                       │            ┌───────────────┐      │  Stream        │
                       └───────────▶│  NoSQL Store  │      │  Processor     │
                        write       │ (key = code)  │      └───────┬────────┘
                                    └───────┬───────┘              ▼
                                            │              ┌────────────────┐
                                            ▼              │ Analytics DB   │
                                    ┌───────────────┐      │ (click counts, │
                                    │ Read Replica  │      │  referrers)    │
                                    │ (+ Redis      │      └────────────────┘
                                    │  replica)     │
                                    └───────────────┘
```

### Component-by-component justification

| Component | Why it exists | The number behind it |
|---|---|---|
| **CDN** | Terminate the redirect at the edge; serve a global audience with low latency | Users in Australia hitting an India server; 100k reads/sec |
| **Load balancer** | Distribute traffic across stateless redirect instances | 1,000–10M concurrent requests |
| **API gateway** | Authentication, routing, quota enforcement, request validation | Central policy point |
| **Rate limiter** | Stop abuse — one client sending 10M requests/sec | Walkthrough: "if one person sends 10 million requests in a second, that's just wrong" |
| **URL service** | The write path: generate code, check collision, persist | 3,000 peak writes/sec |
| **Redirect service** | The read path: lookup, validate expiry, return 302 | **300,000 peak reads/sec — the whole system's centre of gravity** |
| **Redis cache** | Absorb the reads | DB handles ~5,000 QPS vs 100,000 needed → 60× gap |
| **Redis replica** | Fault tolerance on the cache tier | Avoid a cache-tier single point of failure |
| **NoSQL store** | Persist mappings; pure key lookup; horizontal scale | ~355 TB replicated over 5 years |
| **Read replica** | Read scaling and failover for the store | Read-heavy workload |
| **Kafka** | Decouple click events from the redirect path | Redirects must never wait on analytics |
| **Stream processor** | Aggregate, deduplicate and enrich click events | Real-time aggregation |
| **Analytics DB** | Store counts, referrers, geography, time series | Different query pattern from the mapping store |
| **Monitoring** | Observe APIs, redirects, streaming, Kafka | SLO tracking |

### The two design ideas that carry the most weight

**1. The read path never touches the database in the common case.**

```
Redirect request
   │
   ├─▶ Redis HIT  (~95% of the time)  → 302 in ~1 ms   ← the designed-for case
   │
   └─▶ Redis MISS
         ├─▶ NoSQL lookup → populate Redis → 302 in ~10 ms   ← the exception
         └─▶ Not found → 404
```

Every millisecond saved here is multiplied by 300,000 requests/second at peak.

**2. Analytics is not on the critical path.**

```
WRONG:  Redirect ──▶ increment counter in DB ──▶ return 302      ← DB write per click
RIGHT:  Redirect ──▶ fire-and-forget event to Kafka ──▶ 302      ← async, non-blocking
                        │
                        └─▶ stream processor ──▶ analytics DB (seconds later)
```

The walkthrough is explicit: clicks go to Kafka because *"the user keeps clicking repeatedly"* — a click *stream* — and the redirect must not wait. If analytics were synchronous, every click would become a write, and the write load would jump from 3,000/sec (creations) to **103,000/sec** (creations + clicks), a 34× increase that would destroy the design. **Say that number** — it is the argument for the queue.

---

## Step 5 — Deep dive & trade-offs

## Deep dive A — Short code generation (the signature trade-off)

The walkthrough names the four options explicitly: **hashing, base62, distributed IDs, sequential IDs.**

### The four approaches, compared

| Approach | How it works | Pros | Cons | Use when |
|---|---|---|---|---|
| **Sequential ID + base62** | Auto-increment counter → base62 encode | Simplest; no collisions ever; short codes as short as possible | **Predictable → enumerable** (security hole); single global counter is a **bottleneck and SPOF**; leaks business volume | Internal tools; unpredictability irrelevant |
| **Hash (MD5/SHA-256) + base62 prefix** | Hash the long URL, take the first 7 base62 chars | Deterministic — the same URL maps to the same code (dedup for free); stateless, no coordination | **Collisions** must be handled; codes are fixed-length and non-optimal; a hashing cost per write | When deduplicating duplicate long URLs is valuable |
| **Distributed ID (Snowflake)** + base62 | Timestamp + machine ID + sequence → 64-bit ID → base62 | **No coordination**; no collisions; roughly time-ordered; unpredictable enough | Needs clock sync; longer IDs; requires machine-ID allocation | **The default production answer at scale** |
| **Pre-generated key pool** | A service pre-generates unused codes into a pool | Zero collision checks on the hot path; fastest writes | An extra service and store; must handle pool exhaustion | When write latency is critical |
| **Random 7-char base62** | Draw 7 random chars | Trivially unpredictable; stateless | Needs a collision check (read-before-write) on every create | Small scale; simplicity preferred |

### The code-space math

```
Alphabet:   a–z A–Z 0–9          = 62 characters
Length 6:   62⁶ = 56,800,235,584       ≈ 5.7 × 10¹⁰
Length 7:   62⁷ = 3,521,614,606,208    ≈ 3.5 × 10¹²   ← the standard choice
Length 8:   62⁸ ≈ 2.2 × 10¹⁴
```

**With 86.4M URLs/day the system creates ~158M URLs in 5 years.** Against a 7-character space of 3.5 × 10¹²:

```
Collision probability per insert ≈ 158M ÷ 3.5T ≈ 1 in 22,000
```

That is small but **not zero** — and at 86.4M inserts/day, a 1-in-22,000 rate means roughly **4,000 collisions per day**. So collision handling is not optional; it is a routine path.

### Collision handling strategies

| Strategy | Mechanism | Cost | Verdict |
|---|---|---|---|
| **Check-then-insert** | Read the code; if taken, regenerate | One extra read per write; race-prone | Simple but slow and racy |
| **Conditional write** ✅ | `INSERT ... IF NOT EXISTS` on the code; on failure regenerate | One round trip; **race-free** | **The correct answer** |
| **Bloom filter front** | Probabilistic check before the read | False positives cause needless regeneration; no false negatives | Good optimisation, not a correctness mechanism |
| **Pre-generated pool** | Draw from a pool of known-unused codes | Pool service + exhaustion handling | Fastest |

**The sentence to say:** *"I'd generate a code from a distributed ID encoded in base62, then do a conditional write keyed on the code — if it fails, regenerate and retry. That's race-free without a read-before-write on the hot path, and with 7 characters the collision rate is roughly 1 in 22,000, so retries are rare but must be handled."*

### Unpredictability — why sequential IDs fail the requirement

The walkthrough's requirement is explicit: the short code must be **unique and unpredictable**. With sequential base62 codes, an attacker can enumerate `sho.rt/1`, `sho.rt/2`, `sho.rt/3`… and scrape every URL in the system — including private documents, invite links and unlisted content. That is a **security requirement that eliminates a whole design**, and identifying it is a strong signal.

**Mitigation if you must use sequential IDs:** encrypt the ID before encoding (e.g. a Feistel network or a format-preserving cipher), which keeps codes short and coordination-free while making them unpredictable.

### Custom alias — where strong consistency enters

The walkthrough says the alias *"must be unique"* and *"requires a consistency check."*

```
CREATE with alias "my-link"
   │
   ├─▶ Conditional write: INSERT IF NOT EXISTS (code = "my-link")
   │
   ├─▶ Success → 201 Created
   └─▶ Failure → 409 Conflict  ← do NOT silently generate a different alias
```

**The key point:** this is the **one path in the entire system that needs strong consistency**. Everywhere else eventual consistency is fine. Say it: *"The generated-code path can be eventually consistent — the code is unique by construction — but the custom alias path cannot, because two users could simultaneously claim the same alias. That's a conditional write against the primary, not a replica."*

---

## Deep dive B — The cache (why 60×, and how it fails)

### Why it is mandatory, not optional

```
Required reads:        100,000 /sec   (300,000 at peak)
Single DB capacity:    ~5,000 /sec
Gap:                   60× (peak: 120×)
```

No amount of database tuning closes a 60× gap. The cache is structural.

### Cache design decisions

| Decision | Choice | Reasoning |
|---|---|---|
| **Eviction policy** | **LRU** | Link popularity follows a power law; recency predicts the next hit. LFU is better if some links stay hot for months. |
| **TTL** | **24 h–7 days**, or none | Mappings are immutable once created (except expiry/delete). A long TTL is safe and maximises hit rate. |
| **What to cache** | `code → longUrl` (+ `expiresAt`, `status`) | The whole hot lookup in one value |
| **Negative caching** | Cache 404s briefly (60 s) | Prevents repeated misses on nonexistent codes — a common attack pattern |
| **Sizing** | Hot 20% of a day's traffic | At 86.4M new URLs/day and 20% hot, a few hundred GB covers the working set. Use an LRU and size for the hot set, not for everything. |
| **Consistency** | Cache-aside; invalidate on delete/expiry | Mappings are write-once, so invalidation is rare — a big simplification |
| **Stampede control** | Single-flight / request coalescing + jittered TTLs | A newly viral link is the classic thundering-herd case |

### The failure modes of this cache — and the answers

| Failure | Why it's severe here | Mitigation |
|---|---|---|
| **Cache stampede on a viral link** | One link goes viral; 50,000 requests/sec miss simultaneously and all hit the DB | Single-flight locking (one loader per key), probabilistic early expiry, jittered TTLs |
| **Hot key** | One celebrity link dominates a Redis shard | Replicate the hot key across nodes, or add a local in-process cache with a short TTL |
| **Cache down** | 60× the DB's capacity arrives at once → the DB falls over → total outage | Circuit breaker; **serve stale** if possible; aggressive load shedding; keep the DB's own TTL semantics |
| **Redis node lost** | A shard's worth of traffic misses | **Redis replicas** (the walkthrough adds one explicitly) + shard rebalancing |
| **Cold cache after a restart** | Full DB load on recovery | Warm-up, or gradual traffic ramp (connection draining) |
| **Stale redirect after deletion** | A deleted URL still redirects | Invalidate on delete; short negative TTL; accept a small propagation window |

> **The best sentence in this deep dive:** *"A cache absorbing 95% of reads means the DB sees 5,000 QPS instead of 100,000 — but it also means if the cache tier dies, the DB instantly receives 20× its capacity. So the cache is simultaneously the reason the system works and the largest correlated-failure risk, which is why I'd put a circuit breaker and load shedding between them."*

---

## Deep dive C — Storage choice

### Why NoSQL (as the walkthrough specifies)

| Requirement | SQL | NoSQL (key-value) |
|---|---|---|
| **Access pattern** | — | Pure `code → longUrl` lookup. **No joins, no ad-hoc queries** |
| **Scale** | Vertical + read replicas only | **Horizontal sharding built in** |
| **Write throughput** | ~5,000/sec realistically | **Tens of thousands/sec per node** |
| **Schema** | Rigid; migrations | Flexible — add fields like `expiresAt` freely |
| **Consistency** | Strong by default | Tunable (quorum) — exactly what custom aliases need |

**Conclusion:** the workload is a key-value lookup at 355 TB. **NoSQL is correct**, and the reason is the *access pattern*, not fashion.

### Schema

```
Table: url_mappings
  Partition key: shortCode        ← the access pattern
  Attributes:
    longUrl, createdAt, expiresAt, status, ownerId, clickCount
```

**Why `shortCode` is the partition key:** it is the only lookup path, and it distributes uniformly — base62 codes generated from IDs or random draws spread naturally across shards, avoiding hot partitions. **Contrast with partitioning on `userId`**, which would create hot partitions for high-volume users and would require a secondary index for the actual lookup path.

### The read-replica caveat

The walkthrough adds a read replica for fault tolerance. **The subtlety worth naming:** a read replica introduces **replication lag**, which creates the classic trap — a user creates a short URL, and an immediate redirect fails with 404 because the replica hasn't caught up. This is the **read-your-writes** problem.

**Mitigations:**
- Route the *creator's* first reads to the primary (sticky or explicit).
- Cache the new mapping immediately on creation (a write-through on create).
- Accept it — the practical window is milliseconds and the user is unlikely to redirect their own link instantly.

> **Say this out loud.** Naming read-your-writes here shows you know the difference between "reads go to replicas" and "all reads can go to replicas."

---

## Deep dive D — Analytics pipeline

```
Redirect Service
      │  (fire-and-forget, non-blocking, must never fail the redirect)
      ▼
   Kafka topic: click-events
   { code, timestamp, ip, userAgent, referrer, country }
      │
      ▼
   Stream Processor        ← aggregates, deduplicates, enriches
      │
      ├──▶ Analytics DB    (counts, referrers, geography, time series)
      └──▶ Monitoring      (top links, error rates, traffic anomalies)
```

### Design decisions

| Decision | Why |
|---|---|
| **Fire-and-forget from the redirect** | The redirect must succeed even if Kafka is down. Analytics is best-effort; the redirect is not. |
| **Kafka, not a direct DB write** | Buffers bursts; decouples; enables multiple consumers (billing, abuse detection, dashboards) |
| **Separate analytics store** | Different access pattern (aggregations, time series) from the mapping store (key lookups). Don't pollute the hot store. |
| **At-least-once + idempotent aggregation** | Kafka can duplicate; the counting logic must tolerate it |
| **Approximate counts acceptable?** | For public click counts, an approximate counter (HyperLogLog for unique visitors) is often fine and vastly cheaper |
| **Lag is acceptable** | Seconds of lag on analytics is invisible to users; the walkthrough treats it as a background pipeline |

### The counter-scaling trap

A naive design increments a counter per click. At 100,000 clicks/sec, that is **100,000 write QPS on the counter record** — a catastrophic hot row. Solutions: aggregate in the stream processor and write periodically (e.g. every 10 seconds), or shard counters and sum on read, or use probabilistic structures.

---

## Deep dive E — Scaling, availability and operations

| Dimension | Approach |
|---|---|
| **Read scaling** | Redis cluster (sharded) + read replicas of the store |
| **Write scaling** | Sharding by `shortCode`; the primary handles 3,000/sec comfortably |
| **Cache scaling** | Redis Cluster with consistent hashing; add shards as the working set grows |
| **Global distribution** | Multi-region read replicas + edge CDN; writes to a home region (or multi-master with conflict-free keys, since codes are globally unique by construction) |
| **Availability** | Multi-AZ Redis and store; health checks; automated failover; Redis replicas |
| **Rate limiting** | Token bucket per API key and per IP; stricter on creation than on redirect |
| **Abuse** | Domain blocklists, malware scanning on create, negative caching of bad codes |
| **Monitoring** | Redirect latency p50/p95/p99, cache hit rate, Kafka lag, creation error rate, 404/410 rates |
| **Key metrics** | **Cache hit rate** (the single most important number), p99 redirect latency, DB QPS, Kafka consumer lag |
| **Storage lifecycle** | TTL-based expiry; archive cold mappings; delete per retention policy |
| **Cost levers** | Cache hit rate (directly reduces DB cost), TTL tuning, tiered storage, edge caching |

### Failure mode matrix

| Failure | Blast radius | Mitigation |
|---|---|---|
| Redis shard down | ~1/n of reads miss → DB surge | Redis replicas, circuit breaker, load shedding |
| Full cache outage | **DB overwhelmed → total outage** | Circuit breaker + shedding + serve-stale; this is the #1 risk |
| NoSQL primary down | Writes fail | Automated failover to standby; queue writes |
| Read replica lag | Creator sees 404 | Route creator reads to primary; write-through cache on create |
| Kafka down | Analytics lost, **redirects fine** | Buffer locally; best-effort by design |
| CDN outage | Latency up, origin load up | Origin can absorb at reduced performance (degradation ladder) |
| Rate limiter failure | Abuse possible, or legitimate users blocked | Fail-open with monitoring (prefer abuse over outage) — state the trade-off |

---

## The trade-off summary table

| Decision | Alternatives | Chosen | Buys | Costs |
|---|---|---|---|---|
| **Code generation** | Sequential, hash, distributed ID, random | **Distributed ID + base62** | No coordination, unpredictable, no collisions | Machine-ID management, longer IDs |
| **Redirect status** | 301, 302 | **302** | Click analytics work | Less aggressive caching; higher origin load |
| **Storage** | SQL, NoSQL | **NoSQL** | Horizontal scale, key-value fit, flexible schema | No joins/ad-hoc queries, eventual consistency |
| **Read path** | DB direct, cache-aside | **Cache-aside Redis** | 60× capacity, 50 ms → 1 ms | Staleness, correlated-failure risk |
| **Analytics** | Sync DB write, Kafka | **Kafka async** | Redirect unaffected by analytics; burst absorption | Eventual consistency, pipeline ops burden |
| **Cache eviction** | LRU, LFU, TTL-only | **LRU + long TTL** | Best hit rate for power-law popularity | Hot-key risk, memory cost |
| **Alias consistency** | Eventual, strong | **Strong (conditional write)** | Correct uniqueness | Write latency; primary-bound |

---

## Interview Q&A

**Q1. What is the read:write ratio and why does it drive the design?**
100,000 redirects/sec against 1,000 creations/sec — 100:1. That makes this a read-path problem, so the entire architecture is a cache in front of a key-value store, and the redirect path is the centre of gravity. The write path is almost incidental.

**Q2. Why do you need a cache? Prove it.**
At 100,000 read QPS — 300,000 at peak — a single database handling ~5,000 QPS is off by 60–120×. That gap cannot be closed by tuning, so the cache is structural, not an optimisation.

**Q3. How do you generate the short code?**
I'd base62-encode a distributed ID (Snowflake-style: timestamp + machine ID + sequence). Seven base62 characters give 3.5 trillion codes, so against ~158M URLs over five years the collision rate is about 1 in 22,000. No coordination between servers, and codes are unpredictable.

**Q4. Why not sequential IDs?**
Two reasons. First, they're predictable — anyone can enumerate `sho.rt/1`, `/2`, `/3` and scrape every URL, which violates the "unpredictable" requirement. Second, a single global counter is a bottleneck and a single point of failure. If I needed short sequential-looking codes, I'd encrypt the ID before encoding.

**Q5. How do you handle collisions?**
A conditional write keyed on the code — `INSERT IF NOT EXISTS`. On failure, regenerate and retry. That's race-free without a read-before-write on the hot path.

**Q6. 301 or 302?**
302 by default. 301 is cached permanently by browsers and CDNs, which means repeat clicks never reach the origin — and click tracking is a stated requirement. If analytics didn't matter, 301 would reduce origin load. I'd also note the CDN interaction: an edge-cached redirect can't be counted.

**Q7. Why NoSQL and not SQL?**
The access pattern is a pure key lookup — `code → longUrl` — with no joins and no ad-hoc queries, at ~355 TB replicated. That's exactly what a horizontally sharded key-value store is for. SQL would need sharding bolted on and would struggle with the write throughput.

**Q8. What's the partition key and why?**
`shortCode`. It's the only lookup path, and codes distribute uniformly across shards, so there are no hot partitions. Partitioning by `userId` would create hot partitions and wouldn't serve the actual query.

**Q9. How does click analytics work without slowing down redirects?**
The redirect service fires an event to Kafka and returns immediately — fire-and-forget, never blocking, never failing the redirect. A stream processor aggregates and writes to a separate analytics store. If analytics were synchronous, the write path would jump from 3,000/sec to 103,000/sec, a 34× increase that would break the design.

**Q10. What's the biggest risk in this design?**
Cache-tier failure. The cache absorbs 95% of reads, so if it dies the database instantly receives 20× its capacity and falls over — the cache is both why the system works and its largest correlated-failure risk. Mitigations: Redis replicas, a circuit breaker, load shedding, and serving stale where possible.

**Q11. Why the CDN? The payload is tiny.**
It's a latency play, not a bandwidth play. Users in Australia hitting an India server pay a cross-continent round trip for a few hundred bytes of redirect; terminating at the edge removes that. It also absorbs bot traffic and repeat clicks on popular links. The caveat is that an edge-cached 302 changes analytics fidelity.

**Q12. What about custom aliases?**
That's the one path needing strong consistency. Two users could claim the same alias simultaneously, so it's a conditional write against the primary with a 409 on conflict — never a silent regeneration. Everywhere else, eventual consistency is fine.

**Q13. How does this scale to 1 billion users?**
Shard the store by `shortCode`, scale the Redis cluster with consistent hashing, add read replicas, and go multi-region with edge caching and regional read replicas. Codes are globally unique by construction, so multi-region writes are conflict-free — a real advantage of the distributed-ID scheme.

**Q14. *Pushback:* "Why not just use SQL with a big cache?"**
It could work at moderate scale, but the write throughput and the 355 TB target push toward a sharded store, and I'd be adding sharding to SQL manually. Given the access pattern is pure key lookup, NoSQL is the lower-complexity answer. If the system needed rich querying over links — say, analytics inside the same store — SQL would become more attractive.

**Q15. *Pushback:* "Your collision math says 4,000 collisions a day — isn't that a lot?"**
It's about 1 in 22,000, and each is handled by a regenerate-and-retry inside the same request, so it's invisible to users. The point of the number is that collision handling must exist and must be race-free — which is why it's a conditional write rather than a read-then-write.

**Q16. How would you estimate the cache size?**
Link popularity is a power law, so roughly 20% of links take 80% of clicks. I'd size for the hot working set over a day, not for all URLs. At 86.4M creations/day and a 500-byte record, caching the hot 20% is on the order of tens of GB per day of hot data — comfortably in memory across a Redis cluster, and the hit rate should be well above 90%.

---

## Cheat-sheet summary

**Requirements**

| Functional (6) | Non-functional |
|---|---|
| Create · Redirect · Custom alias · Expiration · Click tracking · Delete | 100M DAU · 1k writes/s · 100k reads/s · 99.9% · unique **and** unpredictable · scale to 1B |

**Numbers**

| | |
|---|---|
| Writes | 1,000/s → 86.4M/day |
| Reads | 100,000/s → 8.64B/day |
| Ratio | **100:1 read-heavy** |
| Cache gap | 100k vs 5k = **60×** |
| Storage | 43 GB/day → 15.8 TB/yr → **~355 TB over 5 yrs replicated** |
| Code space | 62⁷ = **3.5 × 10¹²** |
| Collision rate | ~1 in 22,000 (~4k/day) |

**Architecture**

```
Client → CDN → LB → API Gateway → Rate Limiter
                                     ├──▶ URL Service ──▶ NoSQL (+replica)
                                     └──▶ Redirect Service ──▶ Redis (+replica)
                                                              └─ miss ─▶ NoSQL
                                              └─▶ Kafka ─▶ Stream ─▶ Analytics DB
```

**Key decisions**

| Decision | Answer |
|---|---|
| Code generation | Distributed ID → base62, conditional write for collisions |
| Redirect code | **302**, not 301 (analytics) |
| Storage | NoSQL, partition key = `shortCode` |
| Read path | Cache-aside Redis, LRU, long TTL, single-flight |
| Analytics | Kafka, fire-and-forget, never blocks |
| Alias consistency | Strong — conditional write, 409 on conflict |
| Biggest risk | **Cache tier failure → DB overwhelmed** |
| Key metric | **Cache hit rate** |

---

## Cross-links

- Framework used: [Five-step interview framework](../../HLD/01-foundations/04-five-step-interview-framework.md) · [Capacity estimation](../../HLD/01-foundations/05-capacity-estimation-playbook.md)
- Components deep-dived: [CDN](../../HLD/02-networking/02-cdn-deep-dive.md) · [API gateway & rate limiting](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md) · [Load balancers](../../HLD/03-scaling/01-load-balancers.md) · [Caching strategies](../../HLD/03-scaling/02-caching-strategies-and-layers.md) · [Redis](../../HLD/03-scaling/03-redis-deep-dive.md) · [Cache invalidation](../../HLD/03-scaling/04-cache-invalidation-and-consistency.md) · [Sharding & replication](../../HLD/03-scaling/05-sharding-replication-and-consistency.md) · [NoSQL](../../HLD/04-data/02-nosql-deep-dive.md) · [Choosing a database](../../HLD/04-data/05-choosing-a-database-decision-guide.md)
- Other case studies: [WhatsApp](02-whatsapp-chat-system.md) · [Notification system](03-notification-system.md) · [Instagram feed](04-instagram-news-feed.md) · [BookMyShow](07-bookmyshow-seat-booking.md)
- Operations: [Observability](../../Operations/02-observability-and-monitoring.md) · [SLOs & error budgets](../../Operations/03-slos-error-budgets-and-incident-response.md) · [Reliability patterns](../../Operations/07-reliability-patterns.md)
- Drill: [Foundations & capacity questions](../../Interview-Questions/01-foundations-and-capacity-questions.md)
