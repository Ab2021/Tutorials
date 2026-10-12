# The Five-Step Interview Framework — Any Question, Any Company

> **Source of truth:** *System Design Interview Prep | Day 1 — Foundations, Framework & Capacity Estimation* (Section 2: "Interview Framework").

---

## TL;DR

There are exactly five steps to answer **any** system design question, for **any** company: **(1) Clarify requirements → (2) Capacity estimation → (3) Define the API → (4) High-level design → (5) Deep dive + trade-offs.** Never start drawing. "Design Twitter" could mean one hundred different systems, and if you guess the wrong scope you have wasted the entire interview. Smart clarifying questions *are* the signal — they show the interviewer how you think before you act. The framework works because it is problem-agnostic: it converts an open-ended prompt into a structure you can execute under pressure.

> **60-second interview answer:** "I follow five steps. First I clarify requirements — functional and non-functional — because 'design Twitter' could mean the feed, the tweet path, the follow graph or the notifications, and guessing wrong wastes the interview. Second, capacity estimation, because numbers justify architecture: I can't say 'I'll add a cache' without knowing read QPS versus what a database can absorb. Third, I define the API endpoints and their payloads, because that forces me to be precise about what the system actually does. Fourth, I draw the high-level design. Fifth, I go deep on one or two areas the interviewer picks and state the trade-offs explicitly."

---

## 1. Definition & mental model

### The five steps

| # | Step | Question it answers | Artefact you produce |
|---|---|---|---|
| **1** | **Clarify requirements** | *What are we actually building, and how well must it perform?* | A written list of functional + non-functional requirements |
| **2** | **Capacity estimation** | *How big does this need to be?* | QPS, peak QPS, storage, bandwidth numbers |
| **3** | **Define the API** | *What are the contracts?* | Endpoints, request bodies, responses |
| **4** | **High-level design** | *What components exist and how do they connect?* | Architecture diagram with labelled arrows |
| **5** | **Deep dive + trade-offs** | *What are the hard parts, and what did you give up?* | Defended choices, stated sacrifices |

### The mental model: the architect's checklist

Think of it as a pre-flight checklist. Pilots do not improvise the pre-flight sequence, because under pressure people skip steps — and the skipped step is always the one that mattered. The five steps are the same: a fixed order that guarantees you never hand over an aircraft with the fuel caps open.

**Why the order is load-bearing:**

```
   1. REQUIREMENTS ──▶ determines WHAT you build
          │
          ▼
   2. ESTIMATION ────▶ determines HOW BIG it must be
          │
          ▼
   3. API ───────────▶ determines WHAT the contracts are
          │
          ▼
   4. HLD ───────────▶ determines WHICH components exist
          │
          ▼
   5. DEEP DIVE ─────▶ determines WHETHER it actually holds up
```

If you skip to step 4, steps 1–3 are missing and every box you draw is unjustified. If you skip step 1, you may be designing the wrong system entirely.

### The promise (and the honest caveat)

Day 1 states that following these five steps clears 90–95% of interviews. The caveat worth internalising: the framework guarantees *structure*, not *knowledge*. It tells you where to put each piece of knowledge you have. It cannot supply the knowledge — which is why the rest of this knowledge base exists.

---

## 2. Why it exists — the problem it solves

The framework exists to solve five specific, predictable interview failures:

| Failure without the framework | What the framework does |
|---|---|
| You start drawing immediately and design the wrong system | Step 1 forces scope agreement *before* any diagram |
| Your design has components with no justification, and the first cross-question destroys it | Step 2 produces the numbers that make each component derivable |
| You hand-wave about "the API" without knowing what it does | Step 3 forces concrete contracts, which exposes unclear thinking |
| You run out of time with an incomplete design | The steps are time-boxed and each produces a durable artefact |
| You never say what you gave up, so your design looks like a preference | Step 5 makes the sacrifice explicit and mandatory |

And the reason it is *five* steps rather than three or seven: each step is a distinct **mode of thinking** — eliciting, quantifying, specifying, synthesising, defending. Switching modes deliberately keeps you from doing all five badly at once.

---

## 3. Internal mechanics — each step in detail

## Step 1 — Clarify requirements

> **Never start drawing. Always ask questions first.**

Day 1 is emphatic: *"Never start drawing. Always ask the question first. Do not immediately start drawing."*

**Why it matters (verbatim reasoning):** *"Because 'Design Twitter' could mean 100 different systems. If you guess the wrong scope, you waste the whole interview. Interviewers love it when you ask smart clarifying questions — it shows how you think before you act."*

### The two categories — always split your questions this way

#### 1a. Functional requirements — *what the system does*

User-facing behaviour. Features. What is **in** and what is **out**.

**Worked example (Day 1) — a Twitter/X-like system:**

| Clarifying question | Why you ask it |
|---|---|
| Can users send text messages to each other? | Defines whether there's a DM system at all |
| Can users see message history? | Determines storage + pagination + retention |
| Can users create groups? | Group fan-out is a completely different write path |
| Do we need video calls? | **Explicitly out of scope** — says so to protect the time budget |

> **The rule:** *Define the scope — what is in, and what is out. Keep the design focused and manage time.*

Saying "and I'm explicitly excluding video calls, live streaming and payments from this design" is a power move. It shows you understand scope is a choice.

#### 1b. Non-functional requirements — *how well it must perform, in technical terms*

Day 1's framing: *"you are not talking about users, you are talking in a technical way."*

**Worked example (Day 1) — a messaging system:**

| Non-functional requirement | The number or guarantee |
|---|---|
| Scale | 50 million daily active users |
| Latency | messages delivered within **500 ms** |
| Durability | messages **must never be lost** |
| Geography | system must work globally — India, USA, EU |

Notice that each one is *measurable*. That is the point: non-functional requirements are the inputs to step 2.

### The standard clarifying question bank

Ask 6–10 of these. Which ones you ask signals seniority.

| Question | What it decides |
|---|---|
| How many users? DAU and MAU? | The scale of everything |
| Is this **read-heavy** or **write-heavy**? | Whether the design is cache-centric or write-path-centric |
| Should it be **globally distributed**, and why? | Multi-region or single-region |
| What **consistency** do we need — strong or eventual? | The database and replication strategy |
| What is the **latency target** for the critical operation? | Whether caching/CDN/edge is required |
| What is the **availability target**? | Redundancy level, and its cost |
| Can we tolerate **data loss** for any period of time? | Durability strategy, RPO |
| Is this a **new system or an extension of an existing one**? | Whether you must preserve existing interfaces |
| Is **exactly-once** semantics required, or is at-least-once with dedup fine? | Idempotency design |
| Are there **compliance/regulatory** constraints (PII, residency, audit)? | Storage location, encryption, retention |
| What is the **cost envelope**? | How much redundancy is affordable |

### The contrast that makes the point (Day 1's own example)

**Prompt: "Design a notification system."**

| ❌ Bad candidate | ✅ Good candidate |
|---|---|
| Immediately draws: `App → DB → Push Service`. | Asks first: |
| Guesses the scope, and the scope *might be wrong.* | • What types of notification — **push, email, SMS, or all three**? |
| | • How many users — **1 million**? |
| | • Is **delivery speed critical**, or is a few seconds' delay acceptable? |
| | • Do we need to **track whether the user opened** it? |
| | • Is there a **new system to build, or an existing one to extend**? |

Every one of those questions changes the architecture. Push vs SMS changes your provider integrations; open-tracking adds a whole event pipeline; extending an existing system constrains your interfaces. Guessing any of them wrong wastes the interview.

### Step 1 output (write it on the board)

```
FUNCTIONAL                        NON-FUNCTIONAL
• Create short URL from long URL  • 100M DAU
• Redirect short URL → long URL   • 1k URL creations/sec (writes)
• Custom alias (unique)           • 100k redirects/sec (reads)
• URL expiration                  • 99.99% availability
• Track click count               • short code must be unique & unpredictable
• Delete/disable URL              • scale to 1B total URLs
```

*(This is the actual requirement list from the URL shortener video — see the [full case study](../../Case-Studies/HLD/01-url-shortener.md).)*

---

## Step 2 — Capacity estimation

> **Calculating how big the system needs to be — in numbers. Back-of-the-envelope calculations.**

### Why it matters — the single most important paragraph

Day 1 states it exactly:

> **Estimation justifies architecture decisions.** Without estimation, "I will use Redis and caching" invites an immediate cross-question: *"Why? How do you know you need a cache?"*

**The two answers, side by side:**

| ❌ Weak | ✅ Strong |
|---|---|
| "I will use a cache." | "Read QPS is **50,000/sec** and a single DB can handle about **5,000 QPS**. We need a cache in front of the DB to absorb **90% of reads**." |
| Interviewer immediately cross-questions, and you have no answer. | The interviewer cannot cross-question the first time, and if they do, you have a proper answer for *why*. |

This is the moment estimation converts you from someone with opinions into someone with an engineering argument.

### The estimation template (memorise this)

```
STEP 1   Requests per day   = DAU × requests per user per day
STEP 2   QPS                = requests per day ÷ 86,400      (≈ ÷ 10⁵)
STEP 3   Peak QPS           = QPS × 3
STEP 4   Storage            = requests per day × size per record
STEP 5   Bandwidth          = QPS × size per record
```

| Constant | Value | Why |
|---|---|---|
| Seconds in a day | 86,400 ≈ **10⁵** | Use 10⁵ for mental math; interviewers accept the approximation |
| Peak multiplier | **3×** | Traffic is not uniform across the day |
| 1 million requests/day | ≈ **12 QPS** | The most useful conversion in the interview |
| 1 billion requests/day | ≈ **12,000 QPS** | |
| 1 KB × 1M records | **1 GB** | Converts counts to bytes instantly |
| Single DB read capacity | ≈ **5,000 QPS** | The justification anchor |

### Worked example — the Day 1 message system

**Given (from clarifying questions):** 50 million DAU · each user sends 10 messages/day · each message ≈ 1 KB.

| Step | Calculation | Result | What it tells you |
|---|---|---|---|
| 1. Messages/day | 50M × 10 | **500M messages/day** | The unit of work |
| 2. Write QPS | 500M ÷ 10⁵ | **5,000 writes/sec** | Average write load |
| 3. Peak QPS | 5,000 × 3 | **15,000 writes/sec** | Design for this, not the average |
| 4. Storage/day | 500M × 1 KB | **500 GB/day** | → **~180 TB/year** |
| 5. Bandwidth | 5,000 × 1 KB | **5 MB/s** (15 MB/s at peak) | Network sizing |

**Now the architectural conclusions — this is the part that scores:**

| Number | Conclusion |
|---|---|
| 15,000 peak writes/sec | A single database **cannot** handle this → need sharding and/or a write queue |
| 180 TB/year | **Distributed storage**, not one MySQL server → sharding, tiered storage |
| 500M messages/day fan-out to recipients | A **message queue** to absorb the write burst and decouple delivery |
| 5 MB/s sustained | Bandwidth is *not* the bottleneck — storage and write throughput are |

> **The sentence to say:** *"So the numbers tell me the bottleneck is write throughput and storage volume, not bandwidth — which is why I'd introduce a message queue and a sharded store rather than a bigger network pipe."*

### Why this is worth the 5 minutes

1. **It blocks cross-questions.** You have the answer ready.
2. **It makes components derivable.** Each box on your diagram traces to a number.
3. **It exposes the real bottleneck.** Candidates routinely optimise the wrong axis; the numbers reveal which one matters.
4. **It signals seniority.** Estimation is what staff engineers do in design reviews.

### Estimation edge cases to mention

- **Read:write asymmetry.** A feed might be 100:1 read-heavy; a logging system 1:100 write-heavy. State which.
- **Growth.** 180 TB/year means 1.8 PB in ten years — does retention policy change that?
- **Replication multiplies storage.** 3× replication turns 180 TB into 540 TB/year. Say it.
- **Indexes and overhead.** Real storage is 1.3–2× raw payload.
- **Peak isn't uniform.** Diurnal, weekly and event-driven spikes (a launch, a festival, a breaking-news moment) can exceed 3×.

---

## Step 3 — Define the API

> **Before drawing the boxes, define the contracts: what are the endpoints, what goes in, what comes out.**

Day 1's reasoning: *"Why? Because it forces you to think clearly about what the system actually does. It also shows the interviewer you think in interfaces, not just components."*

### Worked example (Day 1) — the message system API

```
POST /v1/messages
Body:    { "senderId": "...", "receiverId": "...", "content": "..." }
Returns: { "messageId": "...", "timestamp": "...", "status": "SENT" }

GET /v1/conversations/{conversationId}/messages?cursor=&limit=
Returns: [ { messageId, senderId, content, timestamp, status }, ... ]

GET /v1/users/{userId}/messages?cursor=&limit=
Returns: [ ... ]
```

And the URL shortener equivalents:

```
POST /v1/urls
Body:    { "longUrl": "...", "customAlias": "...", "expiresAt": "..." }
Returns: { "shortUrl": "https://sho.rt/aB3xK9", "shortCode": "aB3xK9" }

GET /{shortCode}                   → 302 redirect to the long URL (or 410 if expired)
GET /v1/urls/{shortCode}/analytics → { clicks, uniqueVisitors, referrers }
DELETE /v1/urls/{shortCode}        → 204
```

### What a good API section contains

| Element | Why the interviewer cares |
|---|---|
| **Resource naming** | REST semantics; plural nouns; no verbs in paths |
| **HTTP methods** | GET idempotent & cacheable; POST creates; PUT replaces; PATCH updates; DELETE removes |
| **Status codes** | 200/201/204 ok; 400/401/403/404/409/422/429 client; 500/502/503/504 server |
| **Request/response schemas** | Proves you know the shape of the data |
| **Idempotency** | `Idempotency-Key` header for POSTs that create resources (payments, messages) |
| **Pagination** | Cursor-based, not offset-based, for large or changing datasets |
| **Versioning** | `/v1/` in path; how you'd evolve without breaking clients |
| **Auth** | Bearer token / API key / OAuth — at minimum, acknowledge it |
| **Rate limits** | What you'll return when exceeded: 429 + `Retry-After` |

### The hidden benefit

Designing the API often **reveals a requirement you missed**. If you cannot write the endpoint cleanly, the model behind it is unclear. Day 1 makes this point directly: *"It forces you to think clearly."*

---

## Step 4 — High-level design

> **Now you draw.** Components and connections, with the protocol labelled on every arrow.

Day 1's template example (message system):

```
┌────────────┐      ┌───────────────┐      ┌────────────┐
│ Mobile app │─────▶│ Load balancer │─────▶│ App servers│
│  (client)  │ HTTPS│               │ HTTP │  (stateless)
└────────────┘      └───────────────┘      └──────┬─────┘
                                                  │
                        ┌─────────────────────────┼─────────────┐
                        ▼                         ▼             ▼
                 ┌────────────┐          ┌──────────────┐  ┌───────────┐
                 │   Cache    │          │  Primary DB  │  │Read replica│
                 │  (Redis)   │          │  (writes)    │  │  (reads)  │
                 └────────────┘          └──────┬───────┘  └───────────┘
                                                │ replication
                                                └──────────────▶
```

That is the entire shape of the step: client → load balancer → stateless app servers → cache and database, with a replica for read scaling. Nearly every product HLD is a variation of this skeleton, and knowing the skeleton frees your attention for what is *different* about this problem.

### The rule for this step

**Every box must be derived, not declared.** The derivation sentence:

> "Read QPS is 50,000 and a single primary handles ~5,000, so I need a cache in front of the DB. Write QPS is 5,000, which one primary can absorb for now — I'd flag sharding as the trigger when writes exceed roughly 5,000–10,000/s."

### Checklist before you move on

- [ ] Every component traces to a requirement or a number.
- [ ] Every arrow has a protocol/label (HTTPS, gRPC, SQL, Kafka topic, replication stream).
- [ ] Client, entry point, compute, cache, storage, and read/write split are all present.
- [ ] Async work is represented (queues/Kafka) where the write path is bursty.
- [ ] You know what happens when each box dies (feeds directly into step 5).

---

## Step 5 — Deep dive + trade-offs

> **The interviewer picks one or two areas and says "go deeper". You defend your choices and state what you gave up.**

Day 1: *"The interviewer picks one or two areas and says go deeper. Now you defend your choices. You talk about the trade-offs — the sacrifices every decision needs."*

### The "wrong way vs right way" pattern — memorise these two examples

| Component | ❌ Wrong way | ✅ Right way |
|---|---|---|
| **Cache** | "I will use a cache." | "**I'll use Redis as a cache because read QPS is 50,000 and our DB can only handle 5,000.** The cache absorbs **90% of reads** and brings latency down **from 50 ms to 5 ms**. The trade-off is staleness — I'd accept a 1-second TTL for feed data, but for balance updates I'd bypass the cache entirely." |
| **Database** | "I will use NoSQL." | "**I'll use Cassandra here because our access pattern is simple key-value lookups and we need horizontal scaling across 180 TB.** If we needed complex queries and joins I'd consider Postgres instead. The trade-off is that Cassandra gives up strong consistency and ad-hoc query flexibility in exchange for write throughput and linear scale-out." |

**The structural difference:** the right-hand answers contain **(a) the specific technology, (b) the numeric reason, (c) the benefit, and (d) the cost.** Every trade-off answer should have all four.

### What "go deeper" usually means

| The interviewer says… | What they want |
|---|---|
| "How does this scale to 10×?" | Sharding, partitioning, autoscaling, hot-key handling |
| "What if Redis dies?" | Failure analysis: stampede, circuit breaker, local cache, degradation |
| "How do you guarantee messages aren't lost?" | Durability: WAL, replication acks, idempotent consumers, DLQs |
| "Where does consistency break?" | Replication lag, read-your-writes, cache staleness, conflict resolution |
| "How would you handle this globally?" | Multi-region, data residency, latency, conflict resolution |
| "What does this cost?" | Cost per request, cache hit rate economics, tiering, egress |
| "Why not use X instead?" | The comparison the interviewer pre-loaded — know the alternatives |

### The trade-off table you should produce

| Decision | Buys | Costs | Chosen because |
|---|---|---|---|
| Redis cache on read path | p99 50 ms → 5 ms; DB load ÷10 | Staleness ≤1 s; new failure mode | Read:write is 100:1; staleness invisible to users |
| Kafka for click events | Absorbs write bursts; decouples analytics | Eventual consistency; DLQ ops | Analytics can lag seconds; must not block redirects |
| Cassandra for URL mappings | Linear write scale; 180 TB+ | No joins; eventual consistency | Access pattern is pure key lookup |
| Read replica | Read throughput ↑ | Replication lag | Read-heavy |

### The close

End with an explicit statement of what you sacrificed:

> "So the design trades a small amount of read staleness for a large latency and cost win, and it accepts eventual consistency in analytics while keeping writes strongly consistent. If the consistency requirement were tightened, the changes would be: drop the cache on that path, use quorum reads, and accept higher p99."

That final sentence — *what would change if the requirement changed* — is the highest-signal thing you can say in the whole interview.

---

## 4. Variants & taxonomy

### Adjusted frameworks for different question shapes

| Question shape | Framework adjustment |
|---|---|
| **Classical product design** | Use all five steps verbatim |
| **LLD / mechanism** | Steps become: clarify semantics → identify entities → interfaces → data structures + complexity → concurrency + edge cases |
| **Scaling an existing system** | Step 1 becomes "what's the current architecture and the observed symptom"; step 4 becomes "what changes, and what stays" |
| **ML system design** | Add to step 4: data pipeline, training/serving split, feature store, evaluation loop; add to step 5: drift, retraining cadence, offline/online metric divergence |
| **AI/LLM system design** | Step 2 gains token/embedding cost estimation; step 4 gains retrieval, guardrails, eval and gateway layers; step 5 gains prompt/model/index versioning and cost-per-query |
| **Migration design** | Step 5 becomes the entire interview: dual-write, shadow reads, backfill, cutover, rollback |
| **Cost-reduction design** | Step 2 gains a full cost model; step 5 is optimisation levers ranked by impact/effort |
| **Debugging / incident design** | Step 1 = symptom + blast radius; step 2 = timeline; step 4 = hypotheses ranked; step 5 = mitigation then root cause |

### Time-boxing variants

| Total time | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---|---|---|---|---|
| 30 min | 5 | 4 | 3 | 10 | 8 |
| 45 min | 6 | 5 | 5 | 15 | 14 |
| 60 min | 8 | 6 | 6 | 20 | 20 |
| 90 min (with coding) | 8 | 6 | 6 | 20 | 25 + 25 code |

**Adaptive rule:** if you have fewer than 30 minutes, compress steps 2 and 3 (state the numbers as you draw) but **never** skip step 1.

---

## 5. Pros / Cons

### Pros of the framework

| Pros | Cons / caveats |
|---|---|
| Problem-agnostic — works for any prompt, any company, any domain | Structure without knowledge still fails; it organises, it doesn't inform |
| Prevents the #1 failure (drawing the wrong system) | Can feel mechanical if delivered without genuine reasoning |
| Produces durable artefacts at every step (requirements, numbers, API, diagram, trade-offs) | Time-boxing is your responsibility; the framework doesn't enforce it |
| Makes components derivable and cross-questions survivable | Over-applying it to a knowledge question ("what is CAP?") wastes time |
| Scores points for communication, not just correctness | Interviewers who want free-form exploration may find it rigid |
| Directly mirrors real design-review practice | |
| Extends naturally to ML/AI/LLM systems with extra axes | |

### Pros / Cons of each step

| Step | Pros | Cons / risks |
|---|---|---|
| **1. Clarify** | Prevents wrong-scope disaster; high seniority signal | Over-asking wastes time — cap at 6–10 questions |
| **2. Estimate** | Blocks cross-questions; exposes the real bottleneck | Precision paralysis; wrong units; forgetting replication multipliers |
| **3. API** | Forces clarity; exposes missing requirements | Over-designing the API in a scaling-focused round |
| **4. HLD** | Concrete artefact the interviewer can follow | Undeived boxes; unlabelled arrows; happy-path only |
| **5. Deep dive** | Where seniority is actually assessed | Running out of time; asserting instead of justifying |

---

## 6. Exceptions, edge cases & failure modes

| Exception / failure mode | Symptom | How to handle |
|---|---|---|
| **You ask too many questions** | 15 minutes gone, no design on the board | Cap at 6–10 focused questions; state assumptions and move |
| **The interviewer won't answer a question** | "You decide." | Pick a value, say it aloud as an assumption, and note it's revisitable: "I'll assume 10M DAU; tell me if that changes." |
| **The prompt is a knowledge question, not a design question** | "What's the difference between SQL and NoSQL?" | Answer it directly, then *bridge*: "and here's how that choice shows up in a design…" |
| **The prompt is LLD** | "Implement an LRU cache" | Switch to the LLD framework — see [HLD vs LLD](03-hld-vs-lld.md) |
| **You realise mid-step-4 that a requirement was wrong** | The design doesn't fit | Say so and go back: "Revisiting my assumption about X, I need to change Y." Recovering visibly is a positive signal. |
| **The interviewer changes the constraints** | "Now make it work in three regions." | Re-run steps 2 and 4 for the delta only; don't redraw everything |
| **You finish in 25 of 45 minutes** | Awkward silence | Pre-empt with the trade-off close, then offer: "I can go deeper on the failure modes, the cost model, or the data model — which is most useful?" |
| **Estimation numbers are challenged** | "Why 3× peak?" | Defend the assumption and show sensitivity: "If traffic is spikier it's 5×; either way the conclusion — a single DB can't absorb it — holds." |
| **The system is small** | "Design a URL shortener for an internal team." | Say the scale out loud and right-size: "At 1k requests/day this is one service and one database; most of the interesting design is in the encoding and collision handling." |
| **You're asked to design something you don't know** | "Design a CRDT-based collaborative editor." | Use the framework and be honest about depth: clarify requirements, state what you'd research, design what you can. Never bluff — follow-ups expose it. |
| **The interviewer is silent** | No feedback either way | Narrate everything; silence is not disapproval, it's often evaluation. Check in every 5 minutes: "Shall I go deeper here or move on?" |
| **Time runs out before step 5** | You never stated trade-offs | At 5 minutes remaining, stop and do the trade-off close. An unfinished diagram with a stated trade-off beats a finished diagram with none. |

### The genuine exception: when to break the order

The framework is a default, not a law. Two legitimate breaks:

1. **Migration/evolution questions** — start with the *current* architecture (step 4 for the as-is state), because the whole question is about the delta.
2. **Very small systems** — steps 1–3 can be compressed into a single opening sentence, because the design is obvious and the interesting part is elsewhere.

Announce the break: *"For this one I'll start from the existing design, then work out what has to change."*

---

## 7. When to use / When NOT to use

### When to apply the full framework

| Situation | Apply? |
|---|---|
| Any product/system design prompt | ✅ Full five steps |
| Any "scale this" or "fix this" prompt | ✅ (steps 1 and 3 compressed) |
| ML/AI/LLM system design | ✅ With the extra axes from §4 |
| Design-review or architecture discussion at work | ✅ It's literally the same conversation |
| Take-home design documents | ✅ Use as the document outline |

### When NOT to apply it

| Situation | Why not | Do this instead |
|---|---|---|
| "What is CAP theorem?" | It's a knowledge question | Answer concisely, then bridge to a design example |
| "Implement LRU cache" | It's LLD | Use the LLD framework |
| "Estimate storage for 100M users" | It's a *sub-step*, not a full design | Just do step 2 |
| "Review this code" | Different discipline | Code review frameworks |
| A live incident ("the site is down") | Time-critical | Mitigate first; framework afterwards for the post-mortem |
| A trivial internal tool | Over-engineering | Right-size and say so |

---

## 8. Numbers, defaults & tuning

| Parameter | Default | Notes |
|---|---|---|
| Clarifying questions | **6–10** | Fewer looks careless; more wastes time |
| Functional requirements listed | **4–6** | The URL shortener video lists six |
| Non-functional requirements listed | **4–6** | Scale, latency, durability, geography, availability, consistency |
| Estimation time | **≤5 min** for a 45-min interview | State assumptions, don't agonise |
| Peak multiplier | **3×** | Say it explicitly so it can be challenged |
| Seconds/day approximation | **10⁵** | 86,400 exact; 10⁵ for mental math |
| "Single DB" read anchor | **5,000 QPS** | The justification figure used in Day 1 |
| API endpoints sketched | **2–5** | One write, one read, one analytics/admin |
| Components in HLD | **6–15 boxes** | Each derived, each labelled |
| Deep-dive areas | **2** | Usually the chosen ones by the interviewer; have 3 ready |
| Trade-off statements | **≥3** | Each with buys / costs / chosen-because |
| Time to leave for trade-offs | **Last 5 min** | Protect this even if the diagram is unfinished |

---

## 9. Interview Q&A

**Q1. What are the five steps?**
Clarify requirements (functional and non-functional), capacity estimation, define the API, high-level design, then deep dive plus trade-offs.

**Q2. Why can't you just start drawing?**
Because "design Twitter" could mean a hundred different systems — the feed, the tweet path, the follow graph, search, or notifications. Guessing the wrong scope wastes the entire interview. Clarifying first also demonstrates how you think before you act, which is exactly what's being assessed.

**Q3. Give an example of a clarifying question that changes the whole design.**
For a notification system: "Is delivery speed critical, or is a few seconds' delay acceptable?" If seconds are acceptable, I can use a durable queue with batching, which is far cheaper and more resilient. If sub-second delivery is required, I need a different path with persistent connections and no batch delay. Same product name, different architecture.

**Q4. Why is capacity estimation worth five minutes?**
Because it makes architecture decisions defensible. Without it, "I'll add a cache" invites "why?" and I have nothing. With it: "read QPS is 50,000 and a DB handles ~5,000, so a cache is required to absorb 90% of reads." The number ends the debate and shows I design from evidence.

**Q5. Walk me through the estimation template.**
DAU × requests per user per day = requests per day. Divide by 86,400 (or 10⁵) for QPS. Multiply by 3 for peak. Storage = requests per day × record size. Bandwidth = QPS × record size. 1 million requests/day ≈ 12 QPS is the conversion I use most.

**Q6. Estimate a messaging system with 50M DAU sending 10 messages each.**
500M messages/day → 500M ÷ 10⁵ = 5,000 write QPS → peak 15,000/s. Storage at 1 KB/message = 500 GB/day ≈ 180 TB/year. Bandwidth ≈ 5 MB/s average, 15 MB/s at peak. Conclusion: write throughput and storage are the bottlenecks, so I need a queue and a sharded store — a single DB can't absorb 15,000 writes/s. *(The Day 1 transcript renders the peak as 50,000/s and the bandwidth as 1 MB/s — transcription noise on a correct method; see the note in [05](05-capacity-estimation-playbook.md) §3.3.)*

**Q7. Why define the API before drawing components?**
Because it forces clarity about what the system actually does, and it shows the interviewer you think in interfaces, not just boxes. It also surfaces requirements you missed — if you can't write the endpoint cleanly, the data model behind it is unclear.

**Q8. What makes a good trade-off statement?**
Four elements: the specific choice, the numeric reason, the benefit, and the cost. "I'll use Redis as a cache because read QPS is 50,000 and the DB handles 5,000; it absorbs 90% of reads and cuts p99 from 50 ms to 5 ms, at the cost of up to one second of staleness, which is invisible on a feed."

**Q9. What does "go deeper" mean and how do you prepare for it?**
It means the interviewer wants to see depth on one or two areas. I prepare by having three areas ready — usually the write path, the failure modes, and the cost — and by knowing the numbers for each. I also have the alternatives ready, because "why not X?" is the standard probe.

**Q10. How do you handle a prompt you've never seen?**
The framework is the answer. I don't need to have seen the product; I need to elicit requirements, quantify, specify, synthesise and defend. The primitives (load balancer, cache, queue, replica, shard) are reusable across every problem.

**Q11. *Pushback:* "Isn't five steps too rigid for a conversation?"**
It's an order of operations, not a script. The steps guarantee I don't skip something under pressure, but within each step it's a conversation — I ask, listen, adjust. If the interviewer redirects me mid-way, I follow them; the framework is there to return to.

**Q12. *Pushback:* "Your estimation is rough. Isn't that unprofessional?"**
The interviewer knows it's approximate math — that's expected and accepted. What matters is that the assumptions are explicit and that the *conclusion* is robust to them. I say "assuming 3× peak" so it can be challenged, and I check that the architectural conclusion holds at 2× or 5×.

**Q13. *Pushback:* "You spent 10 minutes clarifying. Was that a good use of time?"**
Yes, if the questions changed the design. In this case they determined scope (I excluded video calls), write vs read balance, the consistency requirement and the availability target. Ten minutes to avoid designing the wrong system is a trade I'd make every time — though I'd cap it at six to ten questions.

**Q14. How does the framework change for an AI/LLM system?**
Steps 1–3 are identical. Step 2 gains token cost and embedding storage estimation. Step 4 gains a retrieval layer, an inference/gateway layer, guardrails and an evaluation pipeline. Step 5 gains prompt/model/index versioning, and cost-per-query becomes a first-class trade-off alongside latency and quality.

**Q15. What's the single most common way candidates fail this framework?**
Skipping step 1. Everything downstream — wrong scope, unjustified components, incoherent API — traces back to drawing before asking.

---

## 10. Common mistakes candidates make

1. **Starting to draw before asking anything.** The framework's #1 rule.
2. **Asking 20 questions.** Clarification becomes procrastination.
3. **Asking questions the interviewer can't answer** ("what's your budget?") instead of making and stating assumptions.
4. **Skipping estimation.** Guarantees a "why do you need that?" demolition.
5. **Doing estimation silently.** It's a visible artefact; narrate it and write the numbers down.
6. **Forgetting the peak multiplier.** Designing for average load is designing for failure.
7. **Forgetting that replication multiplies storage.**
8. **Skipping the API step entirely.** Loses an easy clarity signal.
9. **Declaring components instead of deriving them.**
10. **Unlabelled arrows.** The interviewer can't tell HTTP from a replication stream.
11. **Designing only the happy path.** No failure discussion.
12. **Saying "I'll use Redis" / "I'll use NoSQL" without a reason.** The exact anti-pattern Day 1 calls out.
13. **Running out of time without trade-offs.** Protect the last five minutes.
14. **Not adapting when the interviewer changes constraints.** Redraw the delta, not the world.
15. **Treating the framework as a monologue.** It's a conversation; check in every few minutes.

---

## 11. Cheat-sheet summary

| Step | Time (45 min) | Core question | You must produce | The trap |
|---|---|---|---|---|
| **1. Clarify** | 6 min | What are we building, and how well? | Functional *and* non-functional requirement list | Drawing before asking |
| **2. Estimate** | 5 min | How big does it need to be? | QPS, peak QPS, storage, bandwidth | No numbers → no justification |
| **3. API** | 5 min | What are the contracts? | Endpoints + payloads + status codes | Skipping it |
| **4. HLD** | 15 min | What components, connected how? | Labelled architecture diagram | Declared, not derived, boxes |
| **5. Deep dive** | 14 min | What's hard, what did you give up? | 2 deep dives + ≥3 trade-offs | No stated sacrifice |

**Estimation template (memorise):**

```
requests/day = DAU × req/user/day
QPS          = requests/day ÷ 86,400  (≈ ÷ 10⁵)
peak QPS     = QPS × 3
storage      = requests/day × record size
bandwidth    = QPS × record size
```

**Key conversions:** 10⁵ s/day · 1M req/day ≈ 12 QPS · 1B req/day ≈ 12,000 QPS · 1 KB × 1M = 1 GB · single DB ≈ 5,000 read QPS.

**The clarify split:** functional = *what it does* (user-facing features; explicitly state what's **out**). Non-functional = *how well, technically* (DAU, latency, durability, geography, consistency).

**The trade-off formula:** *"I'll use [X] because [number]. It gives me [benefit], at the cost of [sacrifice]."*

**The close:** *"If the requirement changed to [Y], I would change [Z] and accept [new cost]."*

---

## 12. Cross-links

- Previous: [HLD vs LLD](03-hld-vs-lld.md)
- Next: [Capacity estimation playbook](05-capacity-estimation-playbook.md) · [Availability numbers & SLOs](06-availability-numbers-and-slos.md)
- Framework in action: [URL shortener](../../Case-Studies/HLD/01-url-shortener.md) · [WhatsApp](../../Case-Studies/HLD/02-whatsapp-chat-system.md) · [Notification system](../../Case-Studies/HLD/03-notification-system.md) · [Production RAG](../../Case-Studies/HLD/09-production-rag-case-study.md)
- Foundations this framework rests on: [What is system design](01-what-is-system-design-and-why.md) · [Six goals & trade-offs](02-six-goals-and-tradeoffs.md)
- Question banks to drill: [Foundations & capacity](../../Interview-Questions/01-foundations-and-capacity-questions.md)
