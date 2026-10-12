# HLD vs LLD — The Two Shapes of Design Question

> **Source of truth:** *System Design Interview Prep | Day 1 — Foundations, Framework & Capacity Estimation* (the Instagram HLD example and the parking-lot LLD example).

---

## TL;DR

There are exactly **two types of system design question**, and knowing which one you're in determines everything about how you answer. **HLD (High-Level Design)** is the big picture: *which components exist and how they are connected* — boxes, labelled arrows, protocols, capacity numbers. **LLD (Low-Level Design)** is the internals of a single component: *classes, methods, interfaces and data structures* — the object-oriented design you already do when you write a class with a constructor and methods. Getting this wrong is fatal in both directions: drawing a class diagram for "design Uber", or an architecture diagram for "design a parking lot", both fail immediately.

> **60-second interview answer:** "There are two kinds of design question. HLD asks *what components exist and how they connect* — I'd draw a client, an API gateway, services, caches, queues and a database, with the protocol labelled on every arrow, plus capacity numbers. LLD asks *how one component is built* — classes, methods, interfaces, data structures, concurrency. If you want the architecture I'll start with requirements and estimation; if you want the class-level design I'll start with the data structures and the class diagram."

---

## 1. Definition & mental model

### HLD — the big picture

> **HLD decides which components exist and how they are connected.**

You answer: *what are the boxes, and what do the arrows between them carry?*

**Worked example from Day 1 — "Design Instagram":**

```
┌───────┐   HTTPS   ┌─────────────┐   HTTP   ┌────────────────┐
│ User  │ ────────▶ │ API Gateway │ ───────▶ │ Upload/User    │
│ phone │           │  (routing,  │          │ Service        │
└───────┘           │   auth,     │          └───────┬────────┘
                    │   limits)   │                  │ S3 PUT
                    └──────┬──────┘                  ▼
                           │ HTTP              ┌──────────────┐
                           ▼                   │  S3 bucket   │
                    ┌─────────────┐            │  (media)     │
                    │ Feed        │            └──────┬───────┘
                    │ Service     │                   │ CDN pull
                    └──────┬──────┘                   ▼
                           │                    ┌──────────────┐
                           ▼                    │     CDN      │
                    ┌─────────────┐             └──────────────┘
                    │ Redis cache │
                    └──────┬──────┘
                           │ cache miss
                           ▼
                    ┌─────────────┐
                    │  Database   │
                    └─────────────┘
```

Note what that diagram *is*: components (gateway, services, S3, Redis, DB), connections (arrows), and the protocol/operation on each arrow (HTTPS, HTTP, S3 PUT, CDN pull, cache miss). Nothing about classes.

**What an HLD answer must contain:**

| Element | Example |
|---|---|
| Components | API gateway, upload service, feed service, S3, Redis, Postgres |
| Connections + protocol labels | `HTTPS`, `gRPC`, `Kafka topic: media.uploaded`, `SQL`, replication stream |
| Capacity numbers | 100M DAU, 1k writes/s, 100k reads/s, 40 TB/year |
| API contracts | `POST /v1/media`, `GET /v1/feed?cursor=` |
| Data model | Tables/collections, keys, indexes |
| Failure behaviour | What happens when Redis, S3, or a whole region dies |
| Trade-offs | Why fan-out-on-write, why Redis over Memcached, why S3 over a DB blob |

### LLD — the internals

> **LLD decides how one component is built: its classes, methods, interfaces and data structures.**

It is the OOP thinking you use when you write a class. Day 1's phrasing: *"when you don't look at OOP concepts, when you simply come to a class and build it."*

**Worked example from Day 1 — "Design a parking lot":**

```java
class ParkingLot {
    private List<Floor> floors;

    ParkingSpot findAvailableSpot(VehicleType type);
    Ticket      reserveSpot(ParkingSpot spot, Vehicle vehicle);
    void        releaseSpot(Ticket ticket);
}

class Floor {
    private int id;
    private List<ParkingSpot> spots;
}

class ParkingSpot {
    private String id;
    private boolean occupied;
    private VehicleType type;   // BIKE, CAR, TRUCK
}
```

Now you are deciding class boundaries, method signatures, encapsulation, and object interaction.

**The two famous LLD interview questions:**

1. **Design a parking lot** — classes, `findAvailableSpot`, `reserveSpot`, `releaseSpot`, floor/spot hierarchy, pricing strategy, extensibility for new vehicle types.
2. **Implement an LRU cache** — data structures (hashmap + doubly linked list), `get`/`put` in O(1), eviction, then thread safety.

**What an LLD answer must contain:**

| Element | Example |
|---|---|
| Class diagram | `ParkingLot` → `Floor` → `ParkingSpot` |
| Interfaces / abstract types | `PaymentStrategy`, `SpotAllocator` |
| Method signatures | `Ticket reserveSpot(ParkingSpot, Vehicle)` |
| Data structures + complexity | HashMap + DLL → O(1) get/put |
| Design patterns | Strategy, Factory, Observer, Singleton — applied, not name-dropped |
| Concurrency handling | Locks, atomics, optimistic versioning |
| Extensibility | "To add EV charging I'd add a subclass without touching `ParkingLot`" |
| Edge cases | Lot full, double-release, concurrent reserve on the last spot |

### The mental model: blueprint vs wiring diagram

- **HLD is a city map.** It shows the roads, the districts and how traffic flows between them. It does not show the plumbing inside a building.
- **LLD is the building's floor plan.** It shows every room, door and load-bearing wall inside one building on that map.

You need both to build a city, but they are different documents used by different people at different times.

---

## 2. Why it exists — the problem it solves

The HLD/LLD split exists because **the two questions test genuinely different competencies**, and companies need both:

| Competency | Tested by | Why it matters on the job |
|---|---|---|
| Scale thinking, system decomposition, trade-off reasoning, cost | HLD | Staff/principal engineers decide architecture |
| Correctness, data structures, abstraction, concurrency, extensibility | LLD | Any engineer writes code that others maintain |

Without the distinction:

- Candidates **answer the wrong question** and fail for reasons unrelated to skill.
- Candidates **spend the whole time on one axis** — either hand-waving architecture with no correctness thinking, or writing classes for a system that needs sharding.
- Interviewers cannot **calibrate level** — HLD performance maps to seniority/scope; LLD maps to craft.

It also solves a specific interview-day problem: **time allocation**. An HLD answer has a completely different clock from an LLD answer.

| | HLD | LLD |
|---|---|---|
| Time split (45 min) | 5 min clarify · 5 min estimate · 5 min API · 15 min HLD · 10 min deep dive · 5 min trade-offs | 5 min clarify semantics · 10 min class design · 15 min core algorithm/data structure · 10 min edge cases + concurrency · 5 min extensibility |
| Where you must not over-invest | Class-level detail | Component-level scaling |

---

## 3. Internal mechanics — how each is actually built

### 3.1 Building an HLD, step by step

```
1. CLARIFY        functional + non-functional requirements
        │
        ▼
2. ESTIMATE       DAU → QPS → peak → storage → bandwidth
        │
        ▼
3. API            endpoints, request/response shapes
        │
        ▼
4. COMPONENTS     choose boxes, justify each against a goal
        │
        ▼
5. WIRE           draw labelled arrows; name the protocol
        │
        ▼
6. FAILURE        what happens when each box dies
        │
        ▼
7. TRADE-OFFS     what you gave up, and why it's acceptable
```

**Step 4 is where candidates go wrong.** Components must be *derived*, never listed. The derivation always looks like:

> "Read QPS is 50,000 and a single primary handles about 5,000, so I need something in front of the DB → a cache. Write QPS is 1,000 with 500 GB/day of growth, which one primary can absorb for now, but I'd flag sharding as the milestone when writes exceed ~5k/s."

Every box traces back to a number. That is the whole discipline.

### 3.2 Building an LLD, step by step

```
1. CLARIFY SEMANTICS   what are the exact operations and their guarantees?
        │
        ▼
2. IDENTIFY ENTITIES   nouns → classes; verbs → methods
        │
        ▼
3. DEFINE INTERFACES   method signatures, return types, exceptions
        │
        ▼
4. CHOOSE STRUCTURES   what data structure backs each operation?
        │
        ▼
5. COMPLEXITY          time/space per operation — prove O(1), O(log n)…
        │
        ▼
6. CONCURRENCY         locks, atomics, versioning, thread safety
        │
        ▼
7. EXTENSIBILITY       patterns + "how would you add X?"
        │
        ▼
8. EDGE CASES          empty, full, duplicate, concurrent, failure
```

**Worked micro-example — the LRU cache LLD:**

| Step | Answer |
|---|---|
| Clarify | Fixed capacity; `get(key)` returns value or -1; `put(key, value)` evicts least-recently-used. O(1) both. Thread-safe. |
| Entities | `Node(key, value, prev, next)`, `LRUCache(capacity, map, head, tail)` |
| Structures | `HashMap<Key, Node>` for O(1) lookup + doubly linked list for O(1) recency reorder |
| Complexity | `get` O(1), `put` O(1), space O(capacity) |
| Why DLL not array | Array removal is O(n); DLL gives O(1) unlink given the node |
| Concurrency | `ReentrantLock` around mutations, or `ConcurrentHashMap` + `synchronized` on list ops |
| Extensibility | Swap eviction into an interface → LFU/2Q become drop-in |
| Edge cases | capacity 0, capacity 1, get on missing key, put on existing key (update + move to front), eviction when full |

> **Where this goes now that the repo has an LLD pillar.** This eight-step shape *is* [the machine-coding framework](../../LLD/06-machine-coding-interview-framework.md), stated in full there. The LRU/LFU design sketched above has a complete treatment — classes, code, locks, the "add LFU" follow-up — in [LLD/problems/02](../../LLD/problems/02-lru-lfu-cache.md). This file stays the *boundary* file: it tells you which round you are in. The [`LLD/`](../../LLD/) pillar is where you go once you know.

### 3.3 The overlap: where HLD and LLD meet

Some questions sit on the boundary and the interviewer will pull you both ways:

| Question | Starts as | Pulls into |
|---|---|---|
| Design a URL shortener | HLD | Base62 encoding collisions, hash function choice (LLD-ish) |
| Design a rate limiter | HLD (where does it live?) | Token-bucket implementation, Redis Lua atomicity (LLD) |
| Design a seat-booking system | HLD | Row locking, optimistic concurrency, hold-expiry (LLD) |
| Design a notification system | HLD | Retry state machine, dedup keys (LLD) |
| Design an LLM agent | HLD | Tool schemas, loop termination, state machine (LLD) |
| Design a chat app | HLD | Message ordering, sequence numbers, delivery receipts (LLD) |

**The move:** when the boundary is fuzzy, ask. *"I can take this at the architecture level — components, protocols, scaling — or go down to the class and algorithm level for the trickiest part. Which would be more useful?"* This is a strong signal, not a weakness.

---

## 4. Variants & taxonomy

### 4.1 The full taxonomy of design rounds

| Round | Shape | Deliverable | Typical time | Who runs it |
|---|---|---|---|---|
| **System design (HLD)** | Product-level | Architecture diagram + numbers + trade-offs | 45–60 min | Senior/Staff engineer, EM |
| **Low-level design (LLD)** | Component-level | Class diagram + interfaces + complexity | 45–60 min | Senior engineer |
| **Machine coding** | Working code | Runnable program with tests | 90–120 min | Senior engineer |
| **ML system design** | Data + model + serving | Pipeline diagram + metrics + serving plan | 45–60 min | ML engineer/lead |
| **AI/GenAI system design** | LLM-centric | RAG/agent architecture + evals + cost | 45–60 min | AI engineer/architect |
| **Frontend system design** | UI-centric | Component tree + state + caching + perf | 45–60 min | Frontend lead |
| **Data engineering design** | Pipeline-centric | Ingestion + warehouse + orchestration + lineage | 45–60 min | Data platform lead |
| **Security design review** | Threat-centric | Trust boundaries + threat model | 30–45 min | Security architect |

### 4.2 HLD sub-variants

| Sub-type | Focus | Example prompt |
|---|---|---|
| **Product design** | Full feature end-to-end | "Design Instagram" |
| **Infrastructure design** | A reusable primitive | "Design a distributed rate limiter" |
| **Scaling an existing system** | Diagnose + evolve | "Our API p99 went from 80 ms to 3 s. Fix it." |
| **Migration design** | Move without downtime | "Migrate from MySQL to Cassandra with zero downtime" |
| **Cost/performance redesign** | Optimise an existing design | "Cut this pipeline's cost by 50%" |
| **Reliability design** | Failure-driven | "Make this pipeline survive a full region outage" |

### 4.3 LLD sub-variants

| Sub-type | Focus | Example prompt |
|---|---|---|
| **Classic OOP design** | Entities, relationships | "Design a parking lot / elevator / ATM" |
| **Data structure implementation** | Algorithmic + concurrency | "Implement LRU / LFU / a thread-safe queue" |
| **API/interface design** | Contracts and extensibility | "Design a payment gateway SDK" |
| **State machine design** | Lifecycle correctness | "Design an order lifecycle with cancellations" |
| **Concurrency design** | Correctness under threads | "Design a seat-locking service" |

---

## 5. Pros / Cons

### HLD

| Pros | Cons |
|---|---|
| Tests judgement, decomposition and trade-off reasoning — the highest-value senior signal | Open-ended; no single "correct" answer makes self-assessment hard |
| Problem-agnostic framework transfers to every product | Rewards talking; quiet deep thinkers underperform |
| Directly reflects real architecture-review work | Easy to hand-wave and be caught out on a cross-question |
| Numbers make arguments objective | Estimation is approximate and can be judged harshly if unstated assumptions |
| Now extended to AI/ML/LLM systems — same framework | AI versions require additional domain depth (evals, cost/token, serving) |

### LLD

| Pros | Cons |
|---|---|
| Objective: complexity analysis and correctness are falsifiable | Narrower signal about large-scale thinking |
| Directly tests code craft and maintainability instincts | Can be memorised (LRU, parking lot are famous) |
| Extensibility/pattern questions reveal real design taste | Time pressure makes it easy to leave an incomplete class diagram |
| Concurrency questions surface genuinely senior engineers | Some candidates with strong architecture skills underperform here |
| Easy to self-assess and improve with practice | Pattern name-dropping can substitute for pattern understanding |

### The meta-point

| | HLD | LLD |
|---|---|---|
| Signals | Seniority, scope, judgement | Craft, correctness, precision |
| Fails if | You hand-wave or skip requirements | You can't produce a concrete, correct structure |
| Improves by | Drawing 10–15 designs by hand + estimating each | Implementing LRU/LFU/rate limiter/bookmyshow from scratch |

---

## 6. Exceptions, edge cases & failure modes

| Exception / failure mode | Symptom | Fix |
|---|---|---|
| **Wrong shape detected** | You draw classes for "design WhatsApp"; or an architecture for "implement LFU" | Ask early: "architecture or class-level design?" |
| **HLD with no numbers** | Every box is an assertion | Estimate first; derive components from numbers |
| **HLD with no failure handling** | Design works only on the happy path | Add a "what if this box dies?" pass before trade-offs |
| **LLD with no complexity analysis** | You chose a data structure but never justified it | State time/space per operation, always |
| **LLD with no concurrency story** | "It's a hashmap" — race conditions unaddressed | Name the lock or atomic; discuss contention |
| **LLD that isn't extensible** | Adding a vehicle type requires editing five classes | Introduce an interface at the extension point |
| **Mixing the two without signalling** | You jump between scaling and method signatures | Narrate the switch: "now going one level down" |
| **The interviewer escalates deliberately** | "Now design the database schema for that service" | Treat it as a signal to descend; deliver LLD for that component only |
| **The interviewer de-escalates** | "Zoom out — how does this scale to 100M users?" | Return to HLD; do not defend the class design |
| **Famous-question over-fitting** | "Design a parking lot" answered with a memorised blog solution | Clarify requirements; the interviewer will have added a twist |
| **Boundary question (rate limiter, seat booking)** | You don't know which level is wanted | Cover both, briefly, then ask which to deepen |
| **Frontend/ML/AI variant surprises you** | You deliver generic backend HLD | Know the variant's extra axes: UI state/caching, or data/eval/serving |
| **LLD without edge cases** | Works for the demo path only | Enumerate: empty, full, duplicate, concurrent, partial failure |

### The genuine exception: when there is no distinction

For very small prompts the two collapse:

- "Design a URL shortener for 1,000 users" → you can deliver the whole thing (HLD + the Base62 function) in 20 minutes.
- "Design an in-memory key-value store" → almost pure LLD.

Say so: *"At this scale the architecture is one service and one database, so most of the interesting design is at the class level — shall I focus there?"*

---

## 7. When to use / When NOT to use

### Deciding which shape you're in

| Prompt signal | Shape |
|---|---|
| Names a **product** ("design Twitter", "design Uber") | HLD |
| Contains **"at scale"**, "millions of users", "globally" | HLD |
| Names a **mechanism or data structure** ("LRU", "rate limiter in-process", "seat lock") | LLD |
| Says **"classes"**, "interfaces", "design patterns", "OOP" | LLD |
| Asks for **"the architecture"** / "components" / "diagram" | HLD |
| Asks to **implement** or **code** | LLD / machine coding |
| Asks for "endpoints", "schema", "SLA", "cost" | HLD |
| Asks for "complexity", "thread-safe", "extensible" | LLD |

### When NOT to go deep

**In HLD, do not:**
- Write method signatures.
- Argue about hash function internals (unless asked).
- Discuss GC or memory layout.
- Name design patterns.

**In LLD, do not:**
- Design multi-region replication.
- Estimate QPS or storage.
- Draw CDN/load-balancer topology.
- Discuss shard keys.

Both are correct in their place; using one in the other reads as a failure to read the room.

### What the source actually teaches about LLD — and where the rest comes from

This matters for how you use the repo, and it is stated plainly here rather than buried:

| The source (Day 1) does this | The repo adds |
|---|---|
| Defines HLD and LLD precisely — *"you are deciding what components exist and how they are connected"* versus *"one component and code structure kind of thing"* | — |
| Walks through **one** LLD example: the parking lot. `ParkingLot` → `List<Floor>`; `Floor` → `List<ParkingSpot>`; `findAvailableSpot() / reserveSpot() / releaseSpot()` — about eighty seconds of material | A full parking-lot problem file, plus 15 more |
| States the LLD syllabus by name: *"OOPS patterns, coding problems… LRU? LFU? … hotel management system? Booking?"* | [OOP/SOLID](../../LLD/02-oop-principles-and-solid.md), [design patterns](../../LLD/03-design-patterns.md), [concurrency](../../LLD/04-concurrency-and-threading-in-lld.md), [UML](../../LLD/05-uml-and-class-diagrams.md), and the 16-problem set |
| Says LLD is *"ये सबसे ज़्यादा इंपॉर्टेंट है आपका"* — the most important part for you | Takes that seriously and builds the pillar out |

**The honest summary:** the source identifies LLD as the highest-value part of the interview and then teaches one example of it. Everything in [`LLD/`](../../LLD/) beyond the parking lot is **supplementary** — standard interview canon, written to fill the gap the source names. Every LLD file says so at the top. Nothing in this file or that pillar should be attributed to the transcript beyond what the table above credits.

---

## 8. Numbers, defaults & tuning

| Dimension | HLD | LLD |
|---|---|---|
| Time budget (45 min) | Clarify 5 · Estimate 5 · API 5 · HLD 15 · Deep dive 10 · Trade-offs 5 | Clarify 5 · Entities 5 · Interfaces 5 · Structures/algorithm 15 · Concurrency 5 · Edge cases 5 · Extensibility 5 |
| Diagram elements | 6–15 boxes, each labelled with protocol | 4–10 classes, each with 2–6 members |
| Numbers required | QPS, peak, storage, bandwidth, availability target | Time/space complexity per operation |
| Depth of API | Endpoint + payload sketch | Method signatures + exceptions |
| Failure discussion | Node/region/dependency outage | Race conditions, invalid state, resource exhaustion |
| Minimum Q&A you must survive | "Why this component?", "What if it dies?", "What did you give up?" | "Why this data structure?", "Is it thread-safe?", "How would you extend it?" |
| Hand-drawn practice target | 10–15 designs | 5–10 implementations (LRU, LFU, rate limiter, booking, parking lot, elevator) |

---

## 9. Interview Q&A

**Q1. What's the difference between HLD and LLD?**
HLD decides which components exist and how they're connected — boxes, protocols, capacity numbers, failure behaviour. LLD decides how a single component is implemented — classes, interfaces, data structures, complexity and concurrency.

**Q2. How do you know which one you're being asked?**
By the shape of the prompt. Products and "at scale" mean HLD. Mechanisms, data structures, "classes", "interfaces" or "implement" mean LLD. If it's ambiguous I ask: "architecture or class-level design?"

**Q3. Walk me through designing Instagram at a high level.**
Clarify (feed? upload? stories? followers?) → estimate (DAU → QPS → storage: images dominate storage, reads dominate QPS) → API (`POST /media`, `GET /feed`) → components: API gateway, upload service, feed service, blob store for media, CDN for delivery, Redis for feed cache, DB for metadata → wire with labelled arrows → failure: CDN miss falls back to origin, Redis down falls back to DB with a circuit breaker → trade-offs: fan-out-on-write for read speed at the cost of write amplification and the celebrity problem.

**Q4. Walk me through designing a parking lot.**
Clarify (multi-floor? vehicle types? pricing? reservations?) → entities: `ParkingLot`, `Floor`, `ParkingSpot`, `Ticket`, `Vehicle` → interfaces: `findAvailableSpot(type)`, `reserveSpot(spot, vehicle)`, `releaseSpot(ticket)` → structures: per-floor lists of spots with an index by type for O(1) availability lookup → concurrency: a lock per floor or atomic compare-and-set on spot occupancy → extensibility: `PricingStrategy` and `SpotAllocator` interfaces → edge cases: lot full, double release, two threads reserving the last spot.

**Q5. Why does HLD need capacity estimation but LLD doesn't?**
Because HLD's whole purpose is proving the design survives scale, and scale is numeric. LLD's purpose is proving correctness and efficiency, which is measured in time/space complexity, not QPS.

**Q6. Where do you draw the line on detail in an HLD round?**
I go down a level exactly when the interesting engineering is there. For a URL shortener I'll go deep on Base62 encoding and collision handling because that's the core. For "design Uber" I'll go deep on geospatial indexing and matching, not on the payment SDK's class hierarchy.

**Q7. Can a system design interview contain both?**
Yes, commonly. You do HLD and then the interviewer picks one component and says "go deeper" — that's the deep-dive step of the five-step framework, and it's often effectively LLD on a single service.

**Q8. What's the biggest HLD mistake you see?**
Listing components instead of deriving them. Every box should trace back to a requirement or a computed number. "I'll add Redis" is a preference; "read QPS is 50k and the DB handles 5k, so a cache is required" is an engineering decision.

**Q9. What's the biggest LLD mistake?**
Choosing a data structure without stating the complexity, and ignoring concurrency. Both are what the interviewer is actually testing.

**Q10. *Pushback:* "Isn't LLD just coding?"**
No. Coding is producing working code. LLD is deciding the *shape* of the code — class boundaries, interfaces, extensibility, and where the complexity budget goes — before you write it. Machine coding rounds are separate and are about producing a runnable artefact.

**Q11. *Pushback:* "You covered both levels. Which is more important for a senior role?"**
HLD carries more weight for senior/staff levels because it maps to scope and judgement. But a senior candidate who can't produce a clean class design or explain an O(1) LRU raises doubts about craft. I'd expect strength in both, with HLD carrying the level signal.

**Q12. How would you design a distributed rate limiter — HLD or LLD?**
Both, and that's typical of boundary questions. HLD: where it sits (API gateway edge vs service sidecar), per-key counters in Redis, and what happens when Redis is unavailable. LLD: the token-bucket or sliding-window-counter algorithm, and the atomic Redis Lua script that avoids the read-modify-write race. I'd state which I'm covering first.

**Q13. What about ML and AI system design — which shape?**
Primarily HLD with extra axes: data ingestion and versioning, training/eval loops, model serving and scaling, drift detection, and cost per inference. But there's often a genuine LLD component — the agent loop and its state machine, or the retriever's fusion and reranking logic.

**Q14. How do you practise each?**
HLD: draw 10–15 designs by hand, each with a full estimation pass and a written trade-off list — the course's own prescription. LLD: implement the classics from scratch — LRU, LFU, rate limiter, seat booking, parking lot, elevator — and state complexity and concurrency for each.

**Q15. How do you end an HLD answer?**
With trade-offs and explicit sacrifices: "I chose fan-out-on-write to make reads fast, which costs write amplification and hurts hot users; if the celebrity case dominates I'd switch to a hybrid push/pull model." Ending on the sacrifice proves the design is a decision, not a diagram.

---

## 10. Common mistakes candidates make

1. **Not identifying which shape the question is.** The single most avoidable failure.
2. **Drawing classes in an HLD round.** Reads as junior.
3. **Designing multi-region in an LLD round.** Reads as off-topic.
4. **HLD without numbers.** Components become opinions.
5. **HLD without failure handling.** Happy-path-only designs get picked apart.
6. **LLD without complexity analysis.** The data structure choice is unjustified.
7. **LLD without concurrency.** Thread safety is what separates senior from mid.
8. **LLD without extensibility.** Interviewers always ask "how would you add X?".
9. **Memorising famous questions.** The interviewer adds a twist; only method transfers.
10. **Over-investing in one level.** 40 minutes of HLD leaves no time for the deep dive.
11. **Not narrating transitions between levels.** The interviewer loses the thread.
12. **Never converging.** No completed diagram on the board at the end.
13. **Skipping the trade-off close** in HLD, or the edge-case close in LLD.

---

## 11. Cheat-sheet summary

| | **HLD** | **LLD** |
|---|---|---|
| Question | What components exist & how do they connect? | How is one component built? |
| Deliverable | Architecture diagram + numbers + trade-offs | Class diagram + interfaces + complexity |
| Prompt shape | "Design Twitter", "at scale", "globally" | "Implement LRU", "classes", "thread-safe", "extensible" |
| First step | Clarify requirements | Clarify semantics |
| Second step | Capacity estimation | Identify entities (nouns → classes) |
| Core artefact | Labelled arrows with protocols | Method signatures + data structures |
| Proof of quality | Numbers justify every box | Complexity justifies every structure |
| Failure discussion | Node/region/dependency outage | Race conditions, invalid state, exhaustion |
| Closing move | State the sacrifice | State the extension point + edge cases |
| Typical time | 45–60 min | 45–60 min |
| Practise by | 10–15 hand-drawn designs with estimations | Implementing LRU/LFU/limiter/booking from scratch |

**Boundary questions to be ready for:** URL shortener (Base62), rate limiter (token bucket in Redis), seat booking (locks), agent design (tool schemas + loop), chat (message ordering).

**The one-liner:** *"Big picture with components and protocols is HLD; classes, methods and data structures is LLD — tell me which you want and I'll aim there."*

---

## 12. Cross-links

- Previous: [Six goals & trade-offs](02-six-goals-and-tradeoffs.md)
- Next: [The five-step interview framework](04-five-step-interview-framework.md)
- **The LLD pillar this file hands off to:** [What is LLD and how to answer it](../../LLD/01-what-is-lld-and-how-to-answer.md) → [machine-coding framework](../../LLD/06-machine-coding-interview-framework.md) → [the 16 problems](../../LLD/problems/)
- Full worked HLD: [URL shortener case study](../../Case-Studies/HLD/01-url-shortener.md) · [Instagram feed](../../Case-Studies/HLD/04-instagram-news-feed.md) · [WhatsApp](../../Case-Studies/HLD/02-whatsapp-chat-system.md)
- Systems that exist at *both* levels (read the pair to see the boundary move): rate limiter — [HLD](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md) / [LLD](../../LLD/problems/11-rate-limiter.md) · LRU — [here §3.2](#32-building-an-lld-step-by-step) / [LLD](../../LLD/problems/02-lru-lfu-cache.md) · seat booking — [HLD](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md) / [LLD](../../LLD/problems/04-bookmyshow-seat-locking.md)
- Bridge case studies (an HLD design taken down to classes): [URL shortener LLD](../../Case-Studies/LLD/01-url-shortener-lld.md) · [Uber matching LLD](../../Case-Studies/LLD/03-uber-matching-lld.md)
- Revise: [LLD cheat sheet](../../Cheat-Sheets/07-lld-cheatsheet.md) · [LLD question bank](../../Interview-Questions/08-lld-questions.md)
- AI-flavoured design: [Agentic system design](../05-ai-systems/06-agentic-system-design.md) · [Production RAG](../05-ai-systems/01-production-rag-system.md)
