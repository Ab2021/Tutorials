# Capacity Estimation Playbook — The Numbers That Justify Your Architecture

> **Source of truth:** *System Design Interview Prep | Day 1 — Foundations, Framework & Capacity Estimation* (Section 3: "Capacity Estimation") and *AI System Design Interview | Most Asked FAANG System Design Question* (URL shortener estimation walkthrough).

---

## TL;DR

Capacity estimation is back-of-the-envelope arithmetic that answers one question: **how big does this system need to be?** It exists for a single reason — **estimation justifies architecture decisions.** Saying "I'll use Redis" invites the cross-question "why do you need a cache?" Saying "read QPS is 50,000 and a single DB handles 5,000, so a cache must absorb 90% of reads" ends the debate. The whole method is five formulas: `requests/day = DAU × req/user/day`, `QPS = requests/day ÷ 86,400`, `peak = QPS × 3`, `storage = requests/day × record size`, `bandwidth = QPS × record size`. Master six anchors (10⁵ seconds/day, 12 QPS per 1M req/day, 1 GB per 1 KB × 1M, 3× peak, 5,000 QPS per DB, replication multiplies storage) and you can size any system in ninety seconds.

> **60-second interview answer:** "I estimate to make my architecture defensible, not to be precise. I take DAU times requests per user per day to get requests per day, divide by 86,400 — I use 10⁵ for mental math — multiply by three for peak, then compute storage as requests per day times record size and bandwidth as QPS times record size. The anchors I carry: one million requests a day is about 12 QPS, one kilobyte across a million records is a gigabyte, a single database handles roughly five thousand read QPS, and replication multiplies storage by the replica count. Then — and this is the part that scores — I turn the numbers into architecture: for a messaging system at 50 million DAU I get 5,000 write QPS averaging 15,000 at peak and 180 TB a year, so a single database cannot hold it and I need a write queue plus a sharded store."

---

## 1. Definition & mental model

### What it is

**Capacity estimation** is the practice of computing, from user-facing assumptions, the load, storage and bandwidth a system must handle — using arithmetic simple enough to do on a whiteboard in under five minutes.

### What it is *not*

| It is NOT | It IS |
|---|---|
| Precise forecasting | Order-of-magnitude sizing |
| A substitute for benchmarking | An input to the *decision* to benchmark |
| A performance model | A scale model |
| Something the interviewer expects to be exact | Something the interviewer expects to be **explicit about its assumptions** |

### The mental model: sizing a warehouse

You are not asked to compute the exact number of boxes. You are asked: *do we need a garage or a distribution centre?* The answer determines whether you rent a van or build a rail spur. Getting the answer wrong in either direction is expensive — too small and it collapses under load, too large and you've burned the budget on idle capacity.

Estimation is the same. You are choosing between **one database and a sharded cluster**, between **a cache and none**, between **a single region and three**. Those are the decisions; the arithmetic exists to point at them.

### The one-sentence purpose

> **Estimation justifies architecture decisions.**

Everything below serves that sentence.

---

## 2. Why it exists — the problem it solves

### The Day 1 framing, verbatim reasoning

> *"Without estimation, 'I will use Redis and caching' invites an immediate cross-question: 'Why? How do you know you need a cache?'"*

Two candidates, same design, different outcomes:

| ❌ Weak candidate | ✅ Strong candidate |
|---|---|
| "I will use a cache." | "Read QPS is **50,000/sec** and a single DB can handle about **5,000 QPS**. We need a cache in front of the DB to absorb **90% of reads**." |
| Interviewer immediately cross-questions. Candidate has no answer. | The interviewer cannot cross-question the first time, and if they do, there is a proper answer for why. |

**The structural difference:** the second answer converts a *preference* into an *engineering argument*. That is precisely the transition from junior to senior signal.

### The four things estimation buys you

| Benefit | Mechanism |
|---|---|
| **1. It blocks cross-questions** | Every component now has a number behind it |
| **2. It makes components derivable** | Each box on the diagram traces to an arithmetic result |
| **3. It exposes the real bottleneck** | Candidates optimise the wrong axis; numbers reveal which matters |
| **4. It signals seniority** | This is literally what staff engineers do in design reviews — back the proposal with the numbers |

Benefit #3 deserves emphasis. Most candidates *assume* the bottleneck (usually bandwidth, which is almost never it). The arithmetic usually reveals it to be **write throughput**, **storage volume**, **fan-out count**, or **hot keys** — none of which are bandwidth.

### Without estimation, what breaks

| Symptom in the interview | Root cause |
|---|---|
| "Why do you need that?" and you can't answer | No number behind the component |
| You spend time optimising a non-bottleneck | Never identified the bottleneck quantitatively |
| Your design can't survive "what about 10× traffic?" | Never established the headroom |
| You can't answer "what does this cost?" | Never converted requests to resource units |
| Your design is the same for a 1k-user tool and a 100M-user product | Never sized it at all |

---

## 3. Internal mechanics

## 3.1 The five-formula template

```
STEP 1   Requests per day   = DAU × requests per user per day
STEP 2   QPS                = requests per day ÷ 86,400      (≈ ÷ 10⁵)
STEP 3   Peak QPS           = QPS × 3
STEP 4   Storage            = requests per day × size per record
STEP 5   Bandwidth          = QPS × size per record
```

### Read the template as a pipeline

```
  DAU ──× req/user/day──▶ requests/day ──÷ 86400──▶ QPS ──× 3──▶ PEAK QPS
                                │                                  │
                                × record size                      × record size
                                ▼                                  ▼
                            STORAGE/day                        BANDWIDTH
                                │
                                × 365 × retention
                                ▼
                          TOTAL STORAGE
```

### Each formula, and what it decides

| Formula | Produces | The architecture decision it drives |
|---|---|---|
| **1. Requests/day** | Work volume | Retention policy, partitioning key choice |
| **2. QPS** | Sustained load | Number of servers; whether one DB suffices |
| **3. Peak QPS** | Burst capacity | Autoscaling headroom; queue presence; load-shedding thresholds |
| **4. Storage** | Data volume | Single node vs sharded cluster; hot vs cold tiering |
| **5. Bandwidth** | Network load | CDN requirement; egress cost; region placement |

### Storage — the extended form (usually needed)

```
STORAGE PER DAY   = requests/day × record size
ANNUAL STORAGE    = daily storage × 365
RAW TOTAL         = annual × retention years
REAL TOTAL        = raw total × index & overhead (1.3–2×)
REPLICATED TOTAL  = real total × replica count (typically 3×)
```

**Most candidates forget the last two multiplications.** Both matter: indexes commonly add 30–100%, and three-way replication triples the footprint. 180 TB/year becomes ~540 TB/year replicated — which changes the answer from "a big server" to "a distributed store."

---

## 3.2 The anchors — memorise these

| Anchor | Value | Where it's used |
|---|---|---|
| Seconds per day | **86,400 ≈ 10⁵** | Every QPS conversion |
| **1 million requests/day** | **≈ 12 QPS** | The single most useful conversion |
| **1 billion requests/day** | **≈ 12,000 QPS** | |
| **1 KB × 1 million records** | **1 GB** | Converts counts to bytes instantly |
| **1 KB × 1 billion records** | **1 TB** | |
| Peak multiplier | **3×** | Burst capacity |
| Single DB read capacity | **≈ 5,000 QPS** | The justification anchor for caching |
| Single DB write capacity | **≈ 1,000–5,000 QPS** | The justification anchor for sharding/queues |
| 1 server (typical app) | **≈ 1,000 RPS** | Server count estimation |
| 1 Redis node | **≈ 100,000+ ops/sec** | Cache sizing |
| 1 Kafka broker | **≈ 100,000+ msgs/sec** (batched) | Queue sizing |
| 1 node disk | **≈ 1–10 TB** | Storage node count |

### The availability anchors (adjacent, but appear in the same conversation)

| Target | Downtime/year | Downtime/month |
|---|---|---|
| 99% | 3.65 days | 7.3 hours |
| 99.9% | 8.77 hours | 43.8 minutes |
| 99.99% | **52.6 minutes** | **~26 seconds** |
| 99.999% | 5.26 minutes | ~2.6 seconds |

### The bandwidth anchors

| Scale | Equivalent |
|---|---|
| 1 Mbps | ~125 KB/s |
| 1 Gbps | ~125 MB/s |
| 10 Gbps | ~1.25 GB/s |
| 1 TB/month | ~0.4 MB/s sustained |

### Unit conversions you must not fumble

| From | To | Factor |
|---|---|---|
| KB → MB | ÷ 1,024 (≈ ÷1,000) | |
| MB → GB | ÷ 1,024 | |
| GB → TB | ÷ 1,024 | |
| TB → PB | ÷ 1,024 | |
| 1 day | 86,400 s | |
| 1 month | ~2.6M s | |
| 1 year | ~31.5M s | |

---

## 3.3 Worked example A — the Day 1 messaging system

**Assumptions extracted in step 1 (clarify):** 50 million DAU · each user sends 10 messages/day · each message ≈ 1 KB.

| Step | Calculation | Result | What it tells you |
|---|---|---|---|
| 1. Messages/day | 50M × 10 | **500M messages/day** | 500 million writes is enormous |
| 2. Write QPS | 500M ÷ 10⁵ | **5,000 writes/sec** | Average write load |
| 3. Peak QPS | 5,000 × 3 | **15,000 writes/sec** | Design for this |
| 4. Storage/day | 500M × 1 KB | **500 GB/day** | → **~180 TB/year** |
| 5. Bandwidth | 5,000 × 1 KB | **5 MB/s** (15 MB/s peak) | Network is fine |

### Now the architectural conclusions — the part that scores

| Number | Conclusion | Component it introduces |
|---|---|---|
| **15,000 peak writes/sec** | A single database **cannot** absorb this | Sharding and/or a write queue |
| **180 TB/year** | Distributed storage, not one MySQL server | Sharded store + cold tiering |
| **500M messages/day fan-out** | Burst delivery must not block the write path | Message queue (Kafka) |
| **5 MB/s** | Bandwidth is *not* the bottleneck | No CDN needed for this path |

> **The sentence to say out loud:** *"So the numbers tell me the bottleneck is write throughput and storage volume, not bandwidth — which is why I'd introduce a message queue and a sharded store rather than a bigger network pipe."*

That sentence is worth more than five minutes of drawing, because it demonstrates you designed *from evidence*.

> **A note on the transcript's own numbers — read this if you are following along with the video.** The extracted Day 1 transcript is *numerically garbled in two places* in this worked example, and the table above deliberately uses the correct figures instead:
>
> | Step | What the transcript renders as | What is actually correct | Why |
> |---|---|---|---|
> | Peak QPS | *"500 × 3, i.e. 50,000 writes/sec"* ([46:52]–[47:02]) | **15,000** (5,000 × 3) | 500 × 3 is 1,500, and the source had itself just derived 5,000 average QPS one line earlier. The "×3 peak" rule is not in doubt — only the transcription of the multiplicand. |
> | Bandwidth | *"500 × 1 KB = 1 MB per second"* ([47:30]) | **5 MB/s average, 15 MB/s peak** | Bandwidth is QPS × record size, not messages/day × record size. 500 × 1 KB is 500 KB/s and is dimensionally the wrong input anyway. |
>
> The transcription is demonstrably noisy on exactly these tokens elsewhere too — it renders "500 million" as "50 100 million" and "10⁵" as "10 की पावर 5". So treat this as **transcription noise around a correct method**, not as an error in the method. The method — DAU × requests/user/day, ÷ 86,400 (use 10⁵), × 3 for peak, × record size for storage and bandwidth — is stated consistently and is what you should take away. If the video's numbers disagree with this table, trust the table, and be aware the *storage* figures (500 GB/day, ~180 TB/year) come through correctly in both.

### Extended version (with the multipliers candidates forget)

```
Daily raw:        500 GB/day
Annual raw:       500 GB × 365            = 182.5 TB/year
With indexes:     182.5 TB × 1.5          ≈ 274 TB/year
With 3× repl:     274 TB × 3              ≈ 821 TB/year
5-year retention: 821 TB × 5              ≈ 4.1 PB
```

**4.1 petabytes changes the conversation entirely** — from "how big a server?" to "what's the tiering policy, and does anyone actually need a five-year-old message hot?" That question is a strong seniority signal.

---

## 3.4 Worked example B — the URL shortener (from the FAANG-question video)

**Assumptions:** 100M URL creations/day · 100:1 read:write ratio · ~500-byte records · 5-year retention.

| Step | Calculation | Result |
|---|---|---|
| 1. Writes/day | 100M | **100M writes/day** |
| 2. Write QPS | 100M ÷ 10⁵ | **~1,000 writes/sec** |
| 3. Peak write QPS | 1,000 × 3 | **~3,000/sec** |
| 4. Reads/day | 100M × 100 | **10B reads/day** |
| 5. Read QPS | 10B ÷ 10⁵ | **~100,000 reads/sec** |
| 6. Peak read QPS | 100,000 × 3 | **~300,000/sec** |
| 7. Storage/day | 100M × 500 B | **50 GB/day** |
| 8. Annual storage | 50 GB × 365 | **~18 TB/year** |
| 9. Total (5 yr) | 18 TB × 5 | **~90 TB** |
| 10. Bandwidth | 100,000 × 500 B | **~50 MB/s** (150 MB/s peak) |

### The conclusions

| Number | Conclusion |
|---|---|
| **300,000 peak reads/sec** | A single DB (5,000 QPS) is off by **60×** → cache is mandatory, not optional |
| **100:1 read:write** | The design is a **cache-first** design; writes are almost incidental |
| **~3,000 peak writes/sec** | One primary can absorb the writes → no sharding needed at launch |
| **~90 TB over 5 years** | One large node *could* hold it, but replication + indexes push it toward a small cluster |
| **150 MB/s peak** | Meaningful but not extreme — no CDN strictly required; cache at the edge is a latency play, not a bandwidth play |
| **Short code space** | 62⁷ ≈ 3.5 × 10¹² codes — ample for 10¹¹ URLs with negligible collision risk |

> **The derived design in one sentence:** *"100:1 read-heavy with 300,000 peak reads means the entire architecture is a cache in front of a key-value store; writes at 3,000/sec need no sharding, and 90 TB over five years means one replicated primary plus a read replica is enough at launch, with a sharding trigger at roughly 10× growth."*

---

## 3.5 Worked example C — a social feed (fan-out, the hard case)

**Assumptions:** 200M DAU · each user reads the feed 10×/day · each user posts 0.5×/day · average fan-out 200 followers · post ≈ 2 KB.

| Step | Calculation | Result | Implication |
|---|---|---|---|
| Feed reads/day | 200M × 10 | 2B reads/day | |
| Feed read QPS | 2B ÷ 10⁵ | **20,000/sec** | Peak 60,000 → cache |
| Posts/day | 200M × 0.5 | 100M posts/day | |
| Post write QPS | 100M ÷ 10⁵ | **1,000/sec** | Trivial for a DB |
| **Fan-out writes/day** | 100M × 200 | **20B timeline writes/day** | |
| **Fan-out write QPS** | 20B ÷ 10⁵ | **200,000/sec** | **20× what the DB can do** |
| Storage (posts) | 100M × 2 KB | 200 GB/day → **73 TB/yr** | |
| Storage (fan-out) | 20B × 100 B | **2 TB/day → 730 TB/yr** | **The dominant cost** |

### The conclusion this reveals

The *post* write path is trivial (1,000 QPS). The *fan-out* write path is 200,000 QPS. **The fan-out is 200× the problem the naive analysis would have found.**

This is the canonical example of estimation exposing the real bottleneck:

| Naive analysis | Real analysis |
|---|---|
| "1,000 posts/sec — easy, one DB." | "200,000 fan-out writes/sec and 730 TB/year — this *is* the design." |
| Misses the entire hard part | Explains why the **hybrid push/pull** model exists: push for normal users, pull for celebrities, because pushing to 50M followers of one account is impossible |

**The number is the argument for hybrid fan-out.** Without it, "hybrid push/pull" sounds like a memorised pattern. With it, it's a derived conclusion.

---

## 3.6 The estimation-to-architecture mapping table

This is the table that converts arithmetic into design. Keep it in your head.

| If the number is… | Then you need… | Because |
|---|---|---|
| Read QPS > 5,000 | **Cache** | A single DB can't absorb it |
| Read:write > 10:1 | **Cache-first design** | Reads dominate; optimise the read path |
| Write QPS > 5,000 | **Sharding and/or a write queue** | A single primary saturates |
| Storage > ~10 TB | **Sharded/distributed storage** | One node can't hold it comfortably |
| Storage > ~1 TB/year growth | **Tiering + retention policy** | Cost control |
| Fan-out > 1,000 per event | **Async queue + hybrid push/pull** | Synchronous fan-out is impossible |
| Peak:avg > 5:1 | **Queue to absorb bursts** | Provisioning for peak is wasteful |
| Bandwidth > 1 GB/s | **CDN** | Egress cost and origin protection |
| Global users on a latency path | **Multi-region / edge** | Speed of light is not negotiable |
| p99 target < 100 ms cross-continent | **Replication + edge caching** | Physics requires proximity |
| Any number × 3 replicas | **Multiply the storage estimate** | Replication is not free |

---

## 3.7 The "professional answer" pattern

The structural template that turns a number into a senior-sounding sentence:

```
"I measured [METRIC] at [NUMBER], and [RESOURCE] handles about [CAPACITY],
 so we need [COMPONENT] to [EFFECT], bringing [METRIC] from [BEFORE] to [AFTER]."
```

### Instantiations

| Domain | The sentence |
|---|---|
| **Cache** | "Read QPS is 50,000 and a single DB handles 5,000, so a cache absorbing 90% of reads is required, bringing latency from 50 ms to 5 ms." |
| **Sharding** | "Write volume reaches 500 GB/day, 180 TB/year, which no single node holds, so we shard on `user_id` to spread both storage and writes." |
| **Queue** | "Peak write QPS is 15,000 versus 5,000 average, so a queue absorbs the 3× burst and decouples delivery from persistence." |
| **CDN** | "Egress is 150 MB/s at peak and dominated by the same popular objects, so a CDN terminates the top 80% at the edge and cuts origin bandwidth to 30 MB/s." |
| **Multi-region** | "Users in three continents with a 200 ms p99 target cannot be served from one region — speed of light alone costs ~150 ms — so we deploy read replicas per region." |
| **Replication** | "180 TB/year raw becomes ~820 TB with indexes and 3× replication, so I'd set a 12-month hot tier and archive the rest." |

**Say the sentence.** It is the highest-ROI thirty seconds in the interview.

---

## 4. Variants & taxonomy

### By system type — what to estimate

| System type | Primary axis | Secondary axis | The number that decides the design |
|---|---|---|---|
| **Read-heavy (URL shortener, feed, product catalogue)** | Read QPS | Cache hit rate | Read QPS ÷ 5,000 → cache size |
| **Write-heavy (logging, metrics, clickstream, IoT)** | Write QPS | Storage growth | Write QPS ÷ 5,000 → shard count |
| **Fan-out (social, notifications)** | Fan-out writes | Storage of the fan-out | Fan-out QPS vs DB capacity |
| **Media (YouTube, Instagram)** | Storage + bandwidth | Transcoding compute | TB/day → object store; GB/s → CDN |
| **Real-time (chat, ride-hailing, gaming)** | Concurrent connections | Message rate | Connections ÷ per-node capacity |
| **Search (autocomplete, search)** | QPS + latency p99 | Index size | QPS → replica count; p99 → index in RAM |
| **Transactional (bookings, payments)** | Write QPS + consistency | Lock contention | QPS → sharding; consistency → DB choice |
| **Analytics (OLAP)** | Data volume scanned | Query complexity | TB → columnar store + partitioning |
| **ML training** | Samples × features | GPU-hours | Data → object store; GPU-hours → cluster |
| **LLM inference** | Tokens/sec | KV-cache memory | Tokens/sec → GPU count; KV cache → VRAM |

### By deliverable — what the estimate produces

| You're asked | You produce |
|---|---|
| "How many servers?" | Peak QPS ÷ per-server RPS, × 1.5 for headroom |
| "How much storage?" | Daily × retention × overhead × replicas |
| "How many shards?" | Total storage ÷ per-node capacity, or peak QPS ÷ per-shard QPS |
| "What's the cost?" | Resource units × unit price + egress + storage |
| "Will it fit in RAM?" | Index size vs node memory — this decides the whole search design |
| "How many GPUs?" | Tokens/sec ÷ per-GPU throughput (batch-dependent) |

### LLM/GenAI estimation — the 2026 additions

| Quantity | Formula | Notes |
|---|---|---|
| **Prompt tokens/day** | requests/day × avg prompt tokens | |
| **Completion tokens/day** | requests/day × avg completion tokens | Usually the cost driver |
| **Token throughput** | tokens/sec required at peak | Sizes the GPU fleet |
| **KV-cache memory** | 2 × layers × heads × head_dim × seq_len × bytes × batch | Grows linearly with concurrency and context |
| **Embedding storage** | chunks × dims × 4 bytes (fp32) | 1M chunks × 1536 dims × 4 B ≈ 6 GB raw; ~1.5 GB with int8 |
| **Vector index overhead** | raw vectors × 1.2–2× | HNSW graphs add substantial memory |
| **Retrieval QPS** | user QPS × top_k × ... | Sizes the vector DB |
| **Cost per query** | (prompt tokens × in-price + completion × out-price) + retrieval + infra | The number that decides caching/routing |
| **Prompt cache saving** | cached tokens × (in-price − cache-price) | Often 50–90% of input cost |

**The LLM-specific insight:** in classical systems the bottlenecks are QPS and storage. In LLM systems the same arithmetic applies, but **cost per query** is a first-class design constraint alongside latency — because a 2× token-efficiency win is often a 2× margin win.

---

## 5. Pros / Cons

### Pros of estimation

| Pros | Cons / caveats |
|---|---|
| Makes every component defensible | Precision is illusory — all figures are ±3× |
| Exposes the real bottleneck (usually not bandwidth) | Takes 5 minutes that some interviewers want spent elsewhere |
| Converts preferences into engineering arguments | Garbage assumptions produce confident garbage |
| Enables cost and capacity conversations | Easy to fumble units and lose credibility |
| Signals seniority — it's design-review behaviour | Can become a rabbit hole if you chase exactness |
| Reusable across every problem (same five formulas) | Doesn't capture tail behaviour, hot keys, or dependency chains |
| Catches missing multipliers (indexes, replicas) | |

### Pros / Cons of each formula

| Formula | Pros | Cons / risks |
|---|---|---|
| **Requests/day** | Trivial to compute; anchors the whole chain | Depends entirely on the assumption you invented |
| **QPS (÷86400)** | Instantly comparable to hardware capacity | Averages hide bursts — must apply peak |
| **Peak (×3)** | Sizes for reality | The 3× is a heuristic; diurnal/weekly/event spikes differ |
| **Storage** | Drives the biggest cost decision | Forgetting indexes (1.3–2×) and replication (3×) understates by 6× |
| **Bandwidth** | Rules out non-bottlenecks fast | Rarely the bottleneck; easy to over-engineer around |

### Pros / Cons of *how* you present it

| Approach | Pros | Cons |
|---|---|---|
| **Write it on the board** | Interviewer follows; you can point back | Slower |
| **Narrate it aloud** | Fast; conversational | Interviewer may lose the thread |
| **Both (board + narration)** ✅ | Best of both; the artefact persists | Requires practice |
| **State assumptions, then compute** ✅ | Challengeable assumptions = good engineering | — |
| **Silent mental math** ❌ | — | Invisible; looks like guessing; no credit |

---

## 6. Exceptions, edge cases & failure modes

| Exception / edge case | Why it breaks the template | How to handle |
|---|---|---|
| **Uniform-traffic assumption is false** | Real traffic is diurnal; a 3× peak may be 10× | State "3× peak, and I'd validate against observed p99"; design queues to absorb the delta |
| **Hot keys / celebrity problem** | Averages hide a single key taking 30% of traffic | Estimate the *hot partition*: one celebrity with 50M followers makes fan-out 50M writes for one event |
| **Fan-out multiplies writes** | Post QPS × followers ≠ post QPS | Always estimate the *derived* write count, not just the input |
| **Replication multiplies storage** | 3× copies is the default reality | Multiply; state the factor aloud |
| **Indexes and overhead** | Indexes add 30–100% | Use 1.5× as a default and say so |
| **Write amplification (LSM stores)** | Compaction rewrites data ~10× | Relevant for Cassandra/RocksDB-based sizing |
| **Retention is usually unbounded in the prompt** | "Store all messages" → petabytes | Propose a tiering policy: hot 12 months, cold archive, or TTL |
| **Growth rate unknown** | Today's 100M users is next year's 300M | Estimate 3-year growth; ask the interviewer |
| **Geographic distribution** | Same QPS across three regions ≠ one region | Per-region QPS; data residency may forbid consolidation |
| **Caching changes the effective QPS** | The DB doesn't see all reads | Estimate *cache-miss* QPS against the DB (e.g. 10% of 100k = 10k) |
| **Batch/bulk operations** | One "request" may be 1M rows | Estimate rows, not just API calls |
| **Retries amplify load** | A degraded service retries → 2–3× traffic at the worst moment | Design retry budgets; note the amplification |
| **Long-tail storage vs request asymmetry** | 90% of requests hit 1% of data | Estimate both the hot set and the total |
| **Connection counts ≠ QPS** | 10M concurrent WebSocket connections is not a QPS problem | Estimate connections ÷ per-node connection capacity |
| **Token costs scale with output, latency with input** | LLMs break the classic model | Estimate separately; prefill is compute-bound, decode is memory-bandwidth-bound |
| **Unit errors** | KB vs Kb, GB vs Gb (8×), ÷1000 vs ÷1024 | State units explicitly every time |
| **The interviewer disputes an assumption** | Your chain now "looks" broken | Don't defend blindly: "if DAU is 10× that, the conclusion shifts from one shard to four — here's the crossover" |
| **The system is genuinely small** | Estimation is over-engineering | Say so: "at 10k requests/day this is one service and one DB; the interesting part is elsewhere" |
| **You need a decision before the math finishes** | Time pressure | Estimate the dominant term only (usually peak QPS or storage) and state the rest as "similar order" |

### The three assumptions that most often wreck an estimate

1. **Forgetting fan-out.** The write count is rarely what the input suggests.
2. **Forgetting replication + indexes.** Understates storage by 3–6×.
3. **Designing for the average.** Designing for 5,000 QPS when peak is 15,000 is designing for failure.

---

## 7. When to use / When NOT to use

### When to use full estimation

| ✅ Use it when… | Why |
|---|---|
| Any scale-sensitive system design question | The core use case |
| The design hinges on cache vs no-cache, shard vs no-shard | The numbers *are* the decision |
| You're asked about cost or capacity | Directly the question |
| Migrating or replanning infrastructure | Sizing the target |
| Writing a design doc or RFC | Standard practice |
| Choosing a database or storage tier | Volume vs per-node capacity |
| An interviewer asks "will this scale?" | The literal question |

### When NOT to use it

| ❌ Don't use it when… | Do this instead |
|---|---|
| The question is a definition ("what is CAP?") | Answer directly |
| The question is LLD ("implement LRU") | Complexity analysis, not capacity |
| The scale is obviously trivial | Say so in one line and move on |
| You're mid-incident | Mitigate first |
| You have 20 minutes and haven't drawn anything | State two numbers and draw |
| The interviewer explicitly says "skip the math" | Obey; note you'd normally estimate |

### Depth calibration

| Available time | Estimation depth |
|---|---|
| 15 min total | State DAU and QPS in one breath; draw |
| 30 min | Full five formulas, dominant axis only |
| 45 min | Full five formulas + storage multipliers + one conclusion sentence |
| 60+ min | Add cost model, growth projection, hot-key analysis, cache-miss QPS |

---

## 8. Numbers, defaults & tuning

### Defaults to reach for when the interviewer won't supply them

| Parameter | Default | Rationale |
|---|---|---|
| DAU | **1M (small) / 10M (medium) / 100M (large)** | Pick and state it |
| Requests per user per day | **1–10** | 10 for social, 1 for utility |
| Read:write ratio | **100:1 (feed/URL) · 10:1 (typical) · 1:1 (chat)** | Ask; guess 10:1 |
| Record size — text post | **1–2 KB** | |
| Record size — URL mapping | **~500 B** | |
| Record size — user profile | **~1–10 KB** | |
| Record size — image | **~200 KB–2 MB** | |
| Record size — video (1 min) | **~5–50 MB** | |
| Peak multiplier | **3×** | State it |
| Replication factor | **3×** | Standard durability |
| Index/overhead factor | **1.5×** | Conservative |
| Retention | **1–5 years** | Ask |
| Per-server RPS | **~1,000** | Modern app server |
| Per-DB read QPS | **~5,000** | The anchor |
| Per-DB write QPS | **~1,000–5,000** | |
| Per-Redis node | **~100,000 ops/s** | |
| Per-node storage | **1–10 TB** | |
| 3-year growth | **2–3×** | |
| Availability target | **99.99%** | Standard for consumer |

### Tuning your delivery

| Do | Don't |
|---|---|
| Write assumptions **before** numbers | Compute silently |
| Use 10⁵ openly: "roughly 10⁵ seconds a day" | Precisely divide by 86,400 in your head |
| Round aggressively (5,000 not 5,047) | Give false precision |
| State units every time (GB not "gigs") | Assume they're implied |
| Finish with a conclusion sentence | Stop at the number |
| Flag which axis is the bottleneck | Treat all five as equally important |

---

## 9. Interview Q&A

**Q1. Why estimate at all? Isn't it just guessing?**
It's not guessing — it's making assumptions explicit and deriving consequences from them. The purpose is to justify architecture: "read QPS is 50,000 and one DB handles 5,000, so a cache is required" is an argument; "I'll use a cache" is a preference. The interviewer knows it's approximate; the assumptions being explicit is what makes it engineering.

**Q2. Give me the template.**
Requests per day = DAU × requests per user per day. QPS = requests per day ÷ 86,400. Peak = QPS × 3. Storage = requests per day × record size. Bandwidth = QPS × record size.

**Q3. Estimate a messaging system: 50M DAU, 10 messages each, 1 KB per message.**
500M messages/day → 5,000 write QPS → 15,000 at peak. Storage 500 GB/day → 180 TB/year. Bandwidth ~5 MB/s average. Conclusion: write throughput and storage are the bottlenecks, so a queue and a sharded store are required — a single primary saturates at peak.

**Q4. Do you include replication in your storage numbers?**
Yes, and indexes too. 180 TB/year raw becomes roughly 270 TB with index overhead at 1.5× and about 820 TB with three-way replication. That changes the decision from "one big node" to "a sharded cluster with a tiering policy."

**Q5. What's the most useful conversion in the interview?**
One million requests per day ≈ 12 QPS. It lets me convert between product-scale numbers and hardware-scale numbers instantly.

**Q6. What is typically the bottleneck?**
Usually not bandwidth — that's the one candidates assume and it's rarely binding. The real bottlenecks are write throughput, fan-out volume, storage growth, hot keys, or the tail latency of a dependency chain.

**Q7. Walk me through the URL shortener numbers.**
100M creations/day is ~1,000 write QPS, 3,000 at peak. At 100:1 read:write that's 10B reads/day, ~100,000 read QPS, 300,000 at peak. Storage is 50 GB/day, ~18 TB/year, ~90 TB over five years. The conclusion: 300,000 peak reads versus a 5,000-QPS database is a 60× gap, so caching isn't optional — the design is a cache in front of a key-value store. Writes at 3,000/sec need no sharding yet.

**Q8. How do you estimate a system with fan-out?**
You estimate the *derived* write count, not the input. A feed with 100M posts/day and 200 average followers produces 20B timeline writes/day — 200,000 QPS, which is 20–40× a database's capacity. That number is the argument for hybrid push/pull fan-out and an async queue; without it, "hybrid fan-out" is just a memorised pattern.

**Q9. What changes for an LLM system?**
The same formulas apply to requests, but I add two axes: tokens per request, which sizes the GPU fleet, and cost per query, which becomes a first-class design constraint alongside latency. KV-cache memory grows linearly with concurrency and context length, embedding storage is chunks × dimensions × 4 bytes plus 20–100% for the index, and prompt caching can remove 50–90% of input token cost — so the estimate directly drives a routing and caching architecture.

**Q10. *Pushback:* "Your numbers are off by a factor of two."**
Probably — they're order-of-magnitude estimates with stated assumptions. What matters is whether the *conclusion* is robust. At 2× or 0.5×, does the design change? If read QPS is 100,000 or 50,000, the cache is still mandatory. If it were 4,000 instead, the cache would be optional — so let me state the crossover, which is the useful engineering answer.

**Q11. *Pushback:* "You're optimising for a bottleneck you invented."**
That's why I stated the assumption before computing. I chose 3× peak and a 100:1 read:write ratio; if the real ratio is 10:1, read QPS drops 10× and the design shifts from cache-mandatory to cache-optional. Naming the crossover makes the assumption testable rather than arbitrary.

**Q12. What do you do when you don't know a record size?**
Reason from an anchor. A short URL record is a URL, a code and a timestamp — hundreds of bytes. A text post is a paragraph — a couple of kilobytes. An image is hundreds of kilobytes to a few megabytes. I state the guess and note it only affects storage, not QPS.

**Q13. How do you estimate server count?**
Peak QPS ÷ per-server capacity, times roughly 1.5 for headroom. At 300,000 peak read QPS with ~1,000 RPS per app server, that's 300 servers, or 450 with headroom — but a cache absorbing 90% of reads cuts the app-tier requirement substantially, which is the more important architectural point.

**Q14. When would you skip estimation?**
When the system is obviously small — an internal tool at thousands of requests a day — I'd say so in one sentence and spend the time on the interesting part. I'd also compress it if time is short: state DAU and QPS in one breath and draw.

**Q15. What's the single most common estimation mistake?**
Forgetting derived volume — fan-out, retries, batch expansion. Candidates estimate the input and stop. The second most common is forgetting replication and index multipliers, which understates storage by 3–6×.

---

## 10. Common mistakes candidates make

1. **Skipping estimation entirely**, then getting demolished by "why do you need that?"
2. **Estimating silently.** It's a visible artefact; narrate it and write it on the board.
3. **Designing for the average** instead of peak.
4. **Forgetting the 3× peak multiplier** — or applying it without saying so.
5. **Forgetting replication** — 3× the storage.
6. **Forgetting indexes and overhead** — another 1.3–2×.
7. **Estimating input volume but not derived volume** — fan-out, retries, batch.
8. **Assuming bandwidth is the bottleneck.** It almost never is.
9. **Chasing precision** instead of order of magnitude.
10. **Losing units** — KB vs Kb, GB vs Gb, 1000 vs 1024.
11. **Stopping at the number** instead of stating the architectural conclusion.
12. **Guessing DAU silently** rather than stating it as an assumption and inviting correction.
13. **Ignoring growth** — sizing for today's traffic.
14. **Ignoring cost** — never converting resources to money when cost matters.
15. **Spending 12 minutes on arithmetic** and leaving no time to design.
16. **Not knowing the crossover point** where the design would change.
17. **Treating all five formulas as equally important** instead of identifying the binding axis.

---

## 11. Cheat-sheet summary

**The five formulas**

```
requests/day = DAU × req/user/day
QPS          = requests/day ÷ 86,400   (≈ ÷ 10⁵)
peak QPS     = QPS × 3
storage      = requests/day × record size
bandwidth    = QPS × record size
```

**Storage, extended**

```
annual    = daily × 365
indexed   = annual × 1.5
replicated= indexed × 3
total     = replicated × retention years
```

**Anchors**

| Anchor | Value |
|---|---|
| Seconds/day | 86,400 ≈ 10⁵ |
| 1M req/day | ≈ 12 QPS |
| 1B req/day | ≈ 12,000 QPS |
| 1 KB × 1M | 1 GB |
| 1 KB × 1B | 1 TB |
| Peak | 3× average |
| DB reads | ~5,000 QPS |
| DB writes | ~1,000–5,000 QPS |
| App server | ~1,000 RPS |
| Redis node | ~100,000 ops/s |
| Node storage | 1–10 TB |
| Replication | 3× |
| Index overhead | 1.5× |

**Availability**

| Target | Down/year | Down/month |
|---|---|---|
| 99% | 3.65 days | 7.3 h |
| 99.9% | 8.77 h | 43.8 min |
| 99.99% | 52.6 min | ~26 s |
| 99.999% | 5.26 min | ~2.6 s |

**Estimate → architecture**

| Number | Need |
|---|---|
| Read QPS > 5k | Cache |
| Read:write > 10:1 | Cache-first |
| Write QPS > 5k | Shard / queue |
| Storage > 10 TB | Distributed store |
| Fan-out > 1k | Async queue + hybrid |
| Peak:avg > 5:1 | Burst queue |
| Bandwidth > 1 GB/s | CDN |

**The professional sentence**

> *"I measured [METRIC] at [NUMBER], and [RESOURCE] handles about [CAPACITY], so we need [COMPONENT] to [EFFECT], bringing [METRIC] from [BEFORE] to [AFTER]."*

**Rules of delivery:** state assumptions first · write it on the board · round aggressively · say units · finish with the architectural conclusion · name the crossover point.

---

## 12. Cross-links

- Previous: [The five-step interview framework](04-five-step-interview-framework.md)
- Next: [Availability numbers & SLOs](06-availability-numbers-and-slos.md)
- Estimation in action: [URL shortener](../../Case-Studies/HLD/01-url-shortener.md) · [Instagram feed](../../Case-Studies/HLD/04-instagram-news-feed.md) · [YouTube](../../Case-Studies/HLD/05-youtube-upload-and-recommendation.md) · [Uber](../../Case-Studies/HLD/06-uber-ride-hailing.md)
- Where these numbers get consumed: [Load balancers](../03-scaling/01-load-balancers.md) · [Caching strategies](../03-scaling/02-caching-strategies-and-layers.md) · [Sharding & replication](../03-scaling/05-sharding-replication-and-consistency.md) · [Choosing a database](../04-data/05-choosing-a-database-decision-guide.md)
- Capacity in production: [Capacity planning & autoscaling](../../Operations/05-capacity-planning-and-autoscaling.md) · [Cost & FinOps](../../Operations/06-cost-and-finops.md)
- Drill: [Foundations & capacity questions](../../Interview-Questions/01-foundations-and-capacity-questions.md) · [Cheat sheet](../../Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md)
