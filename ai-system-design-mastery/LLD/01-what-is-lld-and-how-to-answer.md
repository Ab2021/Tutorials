# What Is LLD and How to Answer It — The Anchor File

> **Source of truth:** *System Design Interview Prep | Day 1 — Foundations, Framework & Capacity Estimation*, **timestamps [25:54]–[27:38]**. That segment is the source's **only** treatment of low-level design. It defines HLD and LLD, gives the Instagram HLD example, gives the parking-lot LLD example, and names the reasons interviewers ask. **It does not teach OOP, SOLID, design patterns, UML or concurrency.** Everything in this pillar beyond the parking-lot sketch and the definitions is standard industry practice, and every file in this pillar says so explicitly. Do not mistake this file's depth for transcript grounding — the transcript is a signpost, not a textbook.

---

## TL;DR

**HLD decides what components exist and how they are connected. LLD decides how one component is built — its classes, its methods, how those methods interact, and the OOP concepts applied to them.** The source phrase is *"when you don't look at OOPS concepts, you come to a class — it means one component and code structure kind of thing."* That is the whole distinction: HLD is a flow-chart of boxes and labelled arrows; LLD is a class diagram, a set of method signatures, and working code. The interview test is not "can you write a class" — it is **can you think at scale level, can you make trade-offs with proper reasoning, can you communicate clearly under pressure, and do you know the right tools for the right job.** LLD is where craft is measured, because structure and correctness are falsifiable in a way that architecture is not.

> **60-second interview answer:** "HLD is when you're deciding what components exist and how they're connected — boxes, labelled arrows, protocols, a flow-chart. LLD is when you stop looking at the big picture and come down to a class: what classes exist, what methods they have, how the objects interact, and which OOP concepts apply. The classic example is a parking lot — you'd have a `ParkingLot` holding a list of floors, each `Floor` holding a list of spots, with `findAvailableSpot`, `reserveSpot` and `releaseSpot` on the floor. Interviewers ask LLD because they want to know whether you can think at the code level as well as the scale level, whether you can make and defend trade-offs, whether you communicate clearly under pressure, and whether you pick the right tool for the job. I'll clarify the semantics first, identify the entities, define the interfaces, then write the code and handle concurrency."

---

## 1. Definition & mental model

### 1.1 The source definition, verbatim

The Day 1 transcript draws the line at **[25:54]–[27:38]**. Reproduced as the source states it:

**High-level design (HLD)** is *"when you are deciding what components exist and how they are connected."*

The source's own framing of what that looks like on a whiteboard: the **big picture** — components, connections, **boxes and labelled arrows**, *"a flow-chart kind of thing."*

The source's worked HLD example: **"Design Instagram."** The transcript's own chain of boxes is:

```
user phone  →  API gateway  →  upload  →  feed service  →  S3 bucket  →  Redis cache
```

Note what that chain is: six components, five connections, and nothing about a single class. The *whole* of HLD is deciding which of those boxes exist and what each arrow carries.

**Low-level design (LLD)** is *"when you don't look at OOPS concepts... you come to a class... it means one component and code structure kind of thing — like how would you write the classes for the parking-lot system, and you draw a class diagram, define the methods, apply the OOP concept."*

Three phrases in that sentence carry the definition:

| Phrase | What it means in practice |
|---|---|
| *"one component and code structure"* | LLD zooms into a single box from the HLD diagram and asks how it is built |
| *"you draw a class diagram, define the methods"* | The deliverable is a class diagram plus method signatures |
| *"apply the OOP concept"* | Encapsulation, inheritance, polymorphism, abstraction are the tools |

And the closing sentence of the definition: *"Then you are deciding the classes, methods, and how they interact."*

**So LLD involves classes, methods and constructors** — the transcript names all three explicitly.

### 1.2 The source's LLD worked example — the parking lot, exactly as given

The transcript sketches this at **[26:48]–[27:38]**, and it is reproduced here in the shape the source gives it, before any of this pillar's own elaboration.

**The hierarchy as stated:**

- class `ParkingLot` → has **Floors**, held *"in the form of a List"*
- class `Floor` → **`findAvailableSpot()`, `reserveSpot()`, `releaseSpot()`**
- class `Floor` → has **Spots** (a List)
- class `ParkingSpot`

**Drawn as a diagram — read the note under it, because one box is not the source's:**

```
                    ┌──────────────────────┐
                    │      ParkingLot      │
                    │──────────────────────│
                    │ - floors: List<Floor>│
                    │──────────────────────│
                    │ + findAvailableSpot()│
                    │ + reserveSpot()      │
                    │ + releaseSpot()      │
                    └──────────┬───────────┘
                               │  1 : N   ("in the form of a List")
                               ▼
                    ┌──────────────────────┐
                    │        Floor         │
                    │──────────────────────│
                    │ - id: int            │
                    │ - spots: List<Spot>  │
                    │──────────────────────│
                    │ + findAvailableSpot()│
                    │ + reserveSpot()      │
                    │ + releaseSpot()      │
                    └──────────┬───────────┘
                               │  1 : N   (a List)
                               ▼
                    ┌──────────────────────┐
                    │    ParkingSpot       │
                    │──────────────────────│
                    │ - id: String         │
                    │ - occupied: boolean  │
                    │ - type: VehicleType  │
                    └──────────────────────┘
```

> **Fidelity note — what is the source's and what is not.** The transcript attaches `findAvailableSpot()`, `reserveSpot()` and `releaseSpot()` to **`Floor`**, and only to `Floor`. The identical three methods on `ParkingLot` in the diagram above are *this file's addition*: a lot-level convenience that delegates to the floors in turn. It is a sensible design and almost certainly the one the instructor would draw next, but the transcript does not state it, and a reader quoting the source should not claim it does. Everything else in the diagram — the `List<Floor>` composition, the `List<Spot>` on `Floor`, the three method names, and the class names themselves — is the source's own.

**Reproduced as code** — the source's structure, written out so it compiles as a design sketch:

```java
enum VehicleType { BIKE, CAR, TRUCK }

class ParkingSpot {
    private final String id;
    private boolean occupied;
    private final VehicleType type;

    ParkingSpot(String id, VehicleType type) {   // constructor — named in the source
        this.id = id;
        this.type = type;
        this.occupied = false;
    }

    boolean isAvailable()      { return !occupied; }
    void    occupy()           { this.occupied = true; }
    void    vacate()           { this.occupied = false; }
    String  getId()            { return id; }
    VehicleType getType()      { return type; }
}

class Floor {
    private final int id;
    private final List<ParkingSpot> spots;       // a List — as the source says

    Floor(int id, List<ParkingSpot> spots) {
        this.id = id;
        this.spots = spots;
    }

    ParkingSpot findAvailableSpot(VehicleType type) {
        for (ParkingSpot spot : spots) {
            if (spot.isAvailable() && spot.getType() == type) {
                return spot;
            }
        }
        return null;                              // floor full for that type
    }

    Ticket reserveSpot(ParkingSpot spot, Vehicle vehicle) {
        if (!spot.isAvailable()) return null;
        spot.occupy();
        return new Ticket(spot, vehicle, System.currentTimeMillis());
    }

    void releaseSpot(Ticket ticket) {
        ticket.getSpot().vacate();
    }
}

class ParkingLot {
    private final List<Floor> floors;             // "in the form of a List"

    ParkingLot(List<Floor> floors) { this.floors = floors; }

    ParkingSpot findAvailableSpot(VehicleType type) {
        for (Floor floor : floors) {
            ParkingSpot spot = floor.findAvailableSpot(type);
            if (spot != null) return spot;
        }
        return null;
    }

    Ticket reserveSpot(VehicleType type, Vehicle vehicle) {
        SpotAllocator allocator = new NearestFirstAllocator();
        ParkingSpot spot = allocator.allocate(this, type);
        if (spot == null) throw new ParkingFullException();
        for (Floor floor : floors) {
            Ticket t = floor.reserveSpot(spot, vehicle);
            if (t != null) return t;
        }
        throw new IllegalStateException("allocator returned a spot no floor owns");
    }

    void releaseSpot(Ticket ticket) { /* route to the owning floor */ }
}
```

**Two honest observations about the source sketch**, because they matter more than the sketch itself:

1. **The source puts `findAvailableSpot`/`reserveSpot`/`releaseSpot` on `Floor`, and the ASCII it describes also places them on `ParkingLot`.** Both readings appear in the transcript. The resolution used throughout this pillar: `ParkingLot` exposes the *system-facing* operations and delegates to the floor that owns the spot. That is one defensible reading, not a transcript fact — say so if asked.
2. **The source's sketch has no concurrency, no pricing, no gates, no tickets, no vehicle hierarchy.** Those are what the rest of this pillar adds, and they are standard industry practice. See [problems/01-parking-lot.md](problems/01-parking-lot.md) for the full build-out.

### 1.3 The mental model: city map versus floor plan

- **HLD is a city map.** Roads, districts, how traffic flows between them. It does not show the plumbing inside a building.
- **LLD is one building's floor plan.** Every room, door and load-bearing wall inside a single building on that map.

Both are needed to build a city. They are different documents, used by different people, at different times. Drawing one when the other was asked for is an instant, unrecoverable fail.

**A second mental model, closer to the code:** HLD is *"what services exist and who calls whom"*; LLD is *"what objects exist and who owns whom."* The arrows in HLD carry protocols. The arrows in LLD carry ownership, inheritance and references.

### 1.4 What LLD is *not*

| LLD is not… | Why the confusion happens | The distinction |
|---|---|---|
| **HLD with more detail** | Both are "design" | HLD's unit is the *component*; LLD's unit is the *class*. Adding more boxes to an HLD diagram never produces LLD |
| **Machine coding** | Both involve code | LLD produces a *design*; machine coding produces a *running program with tests*. See §1.5 |
| **"Just coding"** | Both end in a function body | Coding is producing working code. LLD is deciding the *shape* of the code before writing it — boundaries, interfaces, extension points, where the complexity budget goes |
| **Algorithm puzzles** | "Implement LRU" sounds like LeetCode | An algorithm answer gives a function. An LLD answer gives `class LRUCache` with an interface, a complexity proof, a thread-safety story and an extension point |
| **Pattern name-dropping** | Patterns are the vocabulary | Naming a pattern without saying what it costs you is worse than not naming it. Every pattern in [03-design-patterns.md](03-design-patterns.md) is paired with its cost |
| **Micro-optimisation** | LLD cares about complexity | LLD cares about *asymptotic* complexity and correctness, not about shaving a branch |

### 1.5 LLD versus machine coding versus HLD — the three rounds, side by side

This distinction is the single most useful thing in this file, because candidates lose offers by preparing for the wrong one.

| | **HLD round** | **LLD round** | **Machine coding round** |
|---|---|---|---|
| Also called | System design, architecture round | Low-level design, object-oriented design, "design a parking lot" | Coding round, "implement X", pair-programming round |
| Prompt shape | "Design Instagram", "design Uber at scale" | "Design a parking lot", "design an elevator", "design BookMyShow's booking flow" | "Implement an LRU cache", "build a rate limiter", "write a thread-safe queue" |
| Deliverable | Architecture diagram + capacity numbers + API + trade-offs | Class diagram + interfaces + method signatures + complexity + concurrency plan | **Compiling, running code** with tests and edge-case handling |
| Primary artefact | Boxes and labelled arrows | Classes and relationships | A repository that runs |
| Time | 45–60 min | 45–60 min | 90–120 min |
| Code written | Little or none | Sketches and signatures; sometimes core methods | Everything |
| Evaluated on | Judgement, decomposition, trade-offs, scale | Structure, abstraction, correctness, extensibility | Correctness, working code, tests, code quality, time management |
| Typical interviewer | Senior/staff engineer, EM | Senior engineer | Senior engineer |
| The classic failure | Hand-waving; no numbers; no failure handling | No complexity analysis; no concurrency story; no extension point | Doesn't compile; no tests; runs out of time on edge cases |
| Preparation | 10–15 hand-drawn designs with estimations | 8–12 class designs, written out | 5–10 implementations, run and tested |

**The relationship that matters:** LLD and machine coding share the same subject matter, and differ only in how far you carry it. If the interviewer says *"design an LRU cache"*, that is LLD. If they say *"now code it, and I'll add a twist at the end"*, that is machine coding. **Prepare for LLD and machine coding together** — design first, then implement — because the machine-coding round opens with a design conversation anyway.

### 1.6 What LLD and HLD share

Both use the same opening move: **never start drawing, clarify first.** The HLD framework's step 1 is requirements; the LLD framework's step 1 is *semantics*. Both are the same discipline — establish the contract before producing the artefact.

And both close the same way: **state what you gave up.** An LLD answer that never names a cost ("I chose a lock per floor, which serialises reservations within a floor but avoids a global bottleneck") is an incomplete answer. See [06-machine-coding-interview-framework.md](06-machine-coding-interview-framework.md) for the full method.

---

## 2. Why it exists — the problem it solves

### 2.1 Why interviewers ask, quoted from the source

The transcript states the four things the interviewer is testing. Verbatim, the interviewer wants to know:

> *"can you think at scale level, not just for 10 users but 10 million users; can you make the trade-offs with proper reasoning; can you communicate clearly under pressure; and how do you know the right tools for the right job."*

Four competencies, and it is worth being precise about which of them LLD tests:

| The source's competency | What HLD tests it with | What LLD tests it with |
|---|---|---|
| **Think at scale level** | QPS, storage, sharding, multi-region | Complexity at scale: O(1) vs O(n) per operation; lock contention under 10,000 concurrent reservers; memory per object × 10M objects |
| **Make trade-offs with proper reasoning** | Cache vs no cache; SQL vs NoSQL; push vs pull | Array vs linked list; lock vs optimistic versioning; inheritance vs composition; immutable vs mutable |
| **Communicate clearly under pressure** | Narrating a diagram while estimating | Narrating a class design while writing signatures, and answering "why not X?" mid-sentence |
| **Know the right tool for the right job** | Redis vs Kafka vs Cassandra | Strategy vs state pattern; mutex vs atomic; interface vs abstract class |

**The scale clause is the one candidates miss.** *"Not just for 10 users but 10 million users"* applies to LLD too. A per-object lock on a `ParkingSpot` is fine for 10 users and produces a lock convoy at 10 million. Complexity analysis *is* scale thinking, expressed in a different unit.

### 2.2 The five interview failures LLD exists to catch

| Failure | What it looks like | Why LLD catches it |
|---|---|---|
| **No abstraction** | Every class knows about every other class; adding a vehicle type edits five files | The "how would you add X?" question has no good answer |
| **Unjustified data structure** | "I'll use a list" with no complexity stated | The follow-up "why not a hash map?" exposes that it was a guess |
| **No concurrency awareness** | "It's just a boolean" in a system with 10,000 concurrent users | The race is trivially constructible in the interview |
| **Wrong granularity** | A `GodClass` with 40 methods, or 60 anemic classes | The interviewer asks who owns a piece of state and the answer is unclear |
| **Memorised answers** | A perfect parking-lot answer, then a collapse when the twist arrives ("now add EV charging and dynamic pricing") | Twists are designed to defeat memorisation; only method transfers |

### 2.3 What breaks without LLD in a real engineering organisation

This is the part that makes LLD worth an entire interview round rather than a whiteboard curiosity:

- **Interface churn.** Without an agreed class boundary, every team working on the same service renegotiates the contract. LLD is how the contract gets fixed before five people write against it.
- **Untestable code.** Code with no seams — no injected dependency, no interface at the boundary — cannot be unit tested. LLD is where the seams are decided.
- **Extension by editing.** A system where every new feature edits existing classes accumulates risk linearly with features. The extension point is an LLD decision.
- **Concurrency bugs shipped.** Race conditions are nearly free to prevent at design time and extremely expensive to find in production. LLD is the last cheap moment.
- **Silent cost.** A per-object lock, an `O(n)` scan inside a hot loop, an N+1 query hidden behind a "getter" — none of these are visible in an architecture diagram. They are visible in a class diagram.

---

## 3. Internal mechanics — how an LLD answer is actually built

### 3.1 The shape of a complete LLD answer

```
   1. CLARIFY SEMANTICS ──▶ exact operations + their guarantees + what is OUT
            │
            ▼
   2. IDENTIFY ENTITIES ──▶ nouns → classes; verbs → methods
            │
            ▼
   3. RELATIONSHIPS ──────▶ class diagram; ownership, cardinality, inheritance
            │
            ▼
   4. INTERFACES ─────────▶ method signatures, return types, exceptions
            │
            ▼
   5. STRUCTURES ─────────▶ which data structure backs each operation
            │
            ▼
   6. COMPLEXITY ─────────▶ time and space per operation, proven not asserted
            │
            ▼
   7. CONCURRENCY ────────▶ what breaks under threads; the lock or atomic
            │
            ▼
   8. CODE ───────────────▶ the core methods, written out
            │
            ▼
   9. EXTENSIBILITY ──────▶ the extension point; "now add X"
            │
            ▼
  10. EDGE CASES ─────────▶ empty, full, duplicate, concurrent, partial failure
            │
            ▼
  11. TRADE-OFFS ─────────▶ what each choice cost
```

Steps 1–4 are the design. Steps 5–7 are the proof. Steps 8–11 are the defence. **All eleven are mandatory**; the weighting changes with the interviewer's interest.

### 3.2 The full worked example in brief — the parking lot

The transcript gives the skeleton. Here is what the *complete* answer looks like, at the level of detail an LLD round expects. The unelaborated version is in §1.2; the full version is [problems/01-parking-lot.md](problems/01-parking-lot.md).

```
   1. CLARIFY       Multi-floor? Vehicle types? Sizes? Pricing? Reservations?
                    Gates? Payment? Assume: multi-floor, 3 vehicle types,
                    3 spot sizes, hourly pricing, no advance reservation,
                    entry/exit gates, card payment.
   2. ENTITIES      ParkingLot, Floor, ParkingSpot, Vehicle (+ subclasses),
                    Ticket, EntryGate, ExitGate, Payment, VehicleType,
                    SpotSize, PricingStrategy, SpotAllocator
   3. RELATIONSHIPS ParkingLot 1─N Floor 1─N ParkingSpot
                    Ticket 1─1 ParkingSpot, 1─1 Vehicle
                    ParkingLot 1─N EntryGate, 1─N ExitGate
   4. INTERFACES    findAvailableSpot(type) → ParkingSpot
                    reserveSpot(spot, vehicle) → Ticket
                    releaseSpot(ticket) → Payment
                    PricingStrategy.calculate(ticket, now) → Money
                    SpotAllocator.allocate(lot, type) → ParkingSpot
   5. STRUCTURES    Per-floor List<ParkingSpot>; per-floor free-count by type
                    (HashMap<SpotSize,Integer>) for O(1) "is anything free?"
                    HashMap<String,Ticket> for O(1) ticket lookup at exit
   6. COMPLEXITY    find O(k) worst case, O(1) with free-lists;
                    reserve O(1); release O(1); space O(spots)
   7. CONCURRENCY   Two threads reserving the last spot → double-sell.
                    Fix: per-spot atomic CAS, or a per-floor lock, or an
                    atomic free-list. Discuss contention.
   8. CODE          ParkingLot, Floor, ParkingSpot, Ticket, strategies
   9. EXTENSIBILITY Add SpotAllocator → nearest-first vs cheapest-first
                    Add PricingStrategy → hourly vs flat vs surge
                    Add VehicleType/SpotSize → EV spot with a charger
  10. EDGE CASES    Lot full; double release; ticket for another lot;
                    lost ticket; overstay; vehicle larger than the spot;
                    concurrent reserve; spot goes out of service
  11. TRADE-OFFS    Per-floor lock: simple, serialises a floor.
                    Per-spot CAS: scalable, needs retry loops.
                    Central free-list: O(1) and fast, but one hot lock.
```

**Read the trade-off line and notice that it contains no pattern names.** Patterns appear because a problem needs them, not to decorate an answer:

| Requirement | Pattern that falls out | Not because |
|---|---|---|
| Pricing varies by vehicle, time and demand | **Strategy** (`PricingStrategy`) | "Strategy is a good pattern" |
| Adding EV charging shouldn't touch `ParkingLot` | **Factory** for spot/vehicle creation | — |
| Gates must be notified when a spot frees up | **Observer** (`ParkingLot` publishes occupancy events) | — |
| Exactly one `ParkingLot` per deployment | **Singleton**, with the caveat that it hurts testability | — |
| A spot has states: FREE → HELD → OCCUPIED → OUT_OF_SERVICE | **State** | — |

### 3.3 What the interviewer is actually scoring

Six axes, roughly equal weight. Candidates who prepare only for axis 1 fail.

| Axis | The scoring question | How it is probed |
|---|---|---|
| **1. Structure** | Are the classes at the right granularity? Is ownership clear? | "Who owns the ticket — the spot or the lot?" |
| **2. Abstraction** | Are the extension points in the right place? | "Now add X" — and watching how many files you say you'd edit |
| **3. Correctness** | Does the design handle the states and the invalid transitions? | "Can a released ticket be released again?" |
| **4. Efficiency** | Is the complexity stated and justified? | "Why a linked list? Why not an array?" |
| **5. Concurrency** | Is the race understood and the fix named? | "Two users select the same seat at the same instant. Then what?" |
| **6. Communication** | Is the reasoning audible while you write? | Silence while staring at the board; refusing to commit to a choice |

**Axis 6 is the one candidates discount most, and it is the one the source names directly** — *"can you communicate clearly under pressure."* Narrate every decision as you make it. An interviewer who cannot follow your reasoning cannot score axes 1–5.

### 3.4 Time budgets

**LLD round, 45 minutes (the common case):**

| Phase | Minutes | Output |
|---|---|---|
| Clarify semantics | 5 | Explicit operation list, guarantees, out-of-scope list |
| Entities + relationships | 5 | Noun list → classes; the class diagram skeleton |
| Interfaces + signatures | 5 | Method signatures with return types and exceptions |
| Data structures + complexity | 10 | Which structure backs each operation, with big-O |
| Core code | 10 | The 2–4 methods that carry the design |
| Concurrency | 5 | The race, the fix, the contention cost |
| Edge cases + extensibility + trade-offs | 5 | The closing section |

**LLD round, 60 minutes:** add 5 minutes to clarify, 5 to code, 5 to extensibility ("now add X" rounds are common at the 60-minute length).

**Machine coding round, 90–120 minutes:**

| Phase | Minutes | Output |
|---|---|---|
| Clarify + design | 15 | Everything above, spoken and sketched |
| Skeleton + interfaces | 15 | Classes and signatures, compiling |
| Happy path | 30 | It works for the simple case |
| Edge cases + validation | 15 | Bad input, empty, full, duplicate |
| Tests | 10 | A handful of real assertions |
| Concurrency (if asked) | 10 | The lock or atomic, plus one test |
| Demo + trade-offs | 5 | Run it, then explain the sacrifices |

**The rule that protects you: never spend more than 15 minutes of a 45-minute round without a class diagram on the board.** An unfinished diagram plus a stated trade-off beats a finished diagram with no trade-off discussion — exactly as in HLD.

### 3.5 A first-pass answer framework

The full method is [06-machine-coding-interview-framework.md](06-machine-coding-interview-framework.md). The compressed version, usable on the spot:

```
0:00  "Let me clarify the semantics first."
      → 5–8 questions. Write the operations on the board as a list.
0:05  "Here are the entities." Nouns → classes. Write them down.
0:10  "Here are the relationships." Draw the class diagram.
0:15  "Here are the interfaces." Write signatures, not bodies.
0:20  "Here's the data structure behind each operation, and the complexity."
0:30  "Now the code for the core method(s)."
0:40  "Here's the concurrency problem and the fix."
0:44  "Edge cases, and how I'd add X."
```

**Say the transitions out loud.** *"Now going one level down to the data structure"* is a signal that you are running a method, not improvising.

---

## 4. Variants & taxonomy

### 4.1 The five LLD question shapes

Recognising the shape tells you where to spend the time.

| Sub-type | Focus | Prompt examples | Where the time goes |
|---|---|---|---|
| **Classic OOP design** | Entities, relationships, patterns | Parking lot, elevator, ATM, library, hotel, Splitwise, chess | Entities + class diagram + extensibility |
| **Data structure implementation** | Algorithm + concurrency | LRU, LFU, rate limiter, thread-safe queue, TTL cache, bloom filter | Data structures + complexity + thread safety |
| **API / interface design** | Contracts and evolution | Payment gateway SDK, notification provider SDK, plugin system | Interfaces + versioning + backward compatibility |
| **State machine design** | Lifecycle correctness | Order lifecycle, ticket lifecycle, document workflow, circuit breaker | States + transitions + invalid transitions + idempotency |
| **Concurrency design** | Correctness under threads | Seat-lock service, inventory counter, driver dispatch, transfer between accounts | The race + the fix + contention + failure of the fix |

**Most real prompts are a blend.** BookMyShow is classic OOP design *wrapped around* a concurrency design *plus* a data structure implementation (the LRU cache in the same interview). Expect to switch sub-type mid-round and announce it.

### 4.2 The same system at three levels of magnification

This table is the single best illustration of the HLD/LLD boundary, because it is the *same system* at three altitudes. It is also why "design BookMyShow" is a boundary question.

| Altitude | Artefact | BookMyShow example |
|---|---|---|
| **HLD** | Components + labelled arrows | CDN → API gateway → booking service → seat-hold store → payment service → Kafka → notification service; Redis for the seat map; Postgres for bookings |
| **LLD** | Classes + methods | `Show`, `Screen`, `Seat`, `SeatHold`, `Booking`, `Payment`, `SeatLockManager`, `HoldExpirySweeper`, `PaymentSaga` — with `SeatLockManager.tryHold(showId, seats, userId, ttl)` |
| **Machine coding** | Running code | `SeatLockManager` implemented with a `dict` + `threading.RLock` + a TTL sweeper thread, with tests for the concurrent case |

**The move when the boundary is ambiguous:** ask. *"I can take this at the class level — entities, interfaces, the locking model — or go all the way to running code. Which is more useful?"* That question is a strength, not a hedge.

### 4.3 LLD sub-variants by domain

| Domain | Recurring LLD problems | The recurring hard part |
|---|---|---|
| **Physical resources** | Parking lot, elevator, hotel, library, ATM | Resource state + concurrent acquisition |
| **Financial** | Splitwise, ATM, ledger, order book | Exact arithmetic, idempotency, invariants |
| **Inventory / booking** | BookMyShow, airline, restaurant, hotel | Double-booking under contention |
| **Data infrastructure** | LRU/LFU, rate limiter, KV store, message queue | Complexity + thread safety |
| **Orchestration** | Elevator scheduling, ride dispatch, job scheduler | Scheduling policy + starvation |
| **AI-adjacent (emerging)** | Agent loop state machine, tool registry, prompt/query builder, guardrail chain, retrieval pipeline | Interfaces + composition + failure handling |

**The AI-adjacent row is real and growing.** An agent loop is a state machine; a tool registry is a factory + adapter; a guardrail pipeline is a chain of responsibility or a decorator chain; a prompt builder is a builder pattern. The vocabulary transfers intact — see [03-design-patterns.md](03-design-patterns.md) for each mapping.

### 4.4 Language variants

The design is language-independent; the *idioms* are not. Say which you are using and why.

| Language | Constructor idiom | Interface idiom | Concurrency primitive you reach for |
|---|---|---|---|
| **Java** | `new`, overloaded constructors, `final` fields | `interface` (default methods), `abstract class` | `synchronized`, `ReentrantLock`, `AtomicInteger`, `ConcurrentHashMap` |
| **Python** | `__init__`, dataclasses | ABCs, `Protocol` (structural) | `threading.Lock`, `queue.Queue`, GIL caveat — see §5 |
| **C++** | constructors, RAII, `= default` | pure-virtual base classes | `std::mutex`, `std::atomic`, `std::lock_guard` |
| **Go** | struct literal + `NewX()` constructors | implicit interfaces | channels, `sync.Mutex`, `sync/atomic` |
| **C#** | constructors, object initialisers | `interface`, `abstract` | `lock`, `Interlocked`, `ConcurrentDictionary` |
| **TypeScript** | `constructor`, parameter properties | `interface` (structural) | single-threaded; concurrency is at the DB/queue |

**Python note that matters in interviews:** in CPython the GIL makes `x += 1` non-atomic across bytecodes but also prevents true parallel execution of pure-Python code. A `threading.Lock` is still required for correctness, and the interviewer will be pleased if you name the GIL rather than hand-wave. This pillar's problem files are written in Python, with the concurrency section naming the GIL explicitly. See [04-concurrency-and-threading-in-lld.md](04-concurrency-and-threading-in-lld.md).

---

## 5. Pros / Cons

### 5.1 Pros and cons of LLD as an interview round

| Pros | Cons |
|---|---|
| Falsifiable — complexity and correctness have right answers, so feedback is actionable | Narrower signal about large-scale thinking than HLD |
| Tests real craft: structure, abstraction, concurrency | The classics are memorisable; a memorised answer collapses on the twist |
| Concurrency questions surface genuinely senior engineers | Candidates strong on architecture but weak on code underperform here |
| Easy to self-assess and to practise alone | Time pressure makes an incomplete class diagram easy |
| Directly reflects the daily work of writing maintainable code | Pattern name-dropping can substitute for pattern understanding in the candidate's own head |
| The extension question ("now add X") is hard to fake | Requires knowing the domain vocabulary (states, invariants, contention) to even start |

### 5.2 The meta-comparison

| | HLD | LLD |
|---|---|---|
| Signals | Seniority, scope, judgement | Craft, correctness, precision |
| Fails if | You hand-wave, or skip requirements | You cannot produce a concrete, correct structure |
| Improves by | 10–15 hand-drawn designs, each with an estimation pass | Implementing 8–12 designs from scratch, with complexity and concurrency stated |
| The question that ends it | "What did you give up?" | "Now add X" |

### 5.3 Pros and cons of the classes-first approach to LLD

There is a real methodological choice here, and the pros/cons are honest:

| Approach | Pros | Cons |
|---|---|---|
| **Entities first** (nouns → classes, then relationships, then methods) | Mirrors how the domain is naturally described; the class diagram emerges cleanly; good for classic OOP problems | Can produce anemic classes with behaviour bolted on later |
| **Operations first** (list the operations and their guarantees, then derive the classes that own them) | Guarantees every class earns its existence; behaviour lands in the right class; better for concurrency and data-structure problems | Slower to start; can miss domain entities the interviewer expects to hear named |
| **Data-structure first** (identify the hot operation, choose the structure, then wrap it in classes) | Correct by construction for the LRU/LFU/rate-limiter family | Wrong for domain problems where no single operation dominates |

**The recommended blend, used throughout this pillar:** operations first (5 minutes), then entities (5 minutes), then check that every entity owns at least one operation. That last check is what prevents anemic classes.

---

## 6. Exceptions, edge cases & failure modes

This is the section to read slowly, because almost every LLD interview is lost here rather than in the happy path.

### 6.1 Wrong shape detected — the fatal error

| Symptom | Why it happens | The fix |
|---|---|---|
| You draw an architecture diagram for "design a parking lot" | The prompt says "design", and "design" reads as architecture | Ask: *"Do you want the class-level design or the system architecture?"* |
| You write class diagrams for "design Instagram" | The prompt names a product | Products are HLD. Only the "go deeper" step descends a level |
| You implement an LRU cache as a LeetCode function | "Implement" reads as algorithm | An LLD answer adds the interface, complexity, thread safety and extension point |
| You design multi-region replication in an LLD round | You default to your strongest material | In LLD, do **not** design multi-region, estimate QPS, or draw load-balancer topology |
| The interviewer deliberately escalates | "Now design the schema for that service" | Descend; deliver LLD for that one component only |
| The interviewer deliberately de-escalates | "Zoom out — how does this work at 100M users?" | Return to HLD. Do not defend the class design |

### 6.2 The LLD-specific failure modes

| Exception / failure mode | Symptom | How to handle |
|---|---|---|
| **No complexity analysis** | You chose a list and never said why | State time and space per operation for every public method. Always |
| **No concurrency story** | "It's a hash map" | Name the lock or the atomic and say what it serialises |
| **Not extensible** | Adding a vehicle type edits five classes | Introduce an interface at the extension point and show that the new case is a new class |
| **Anemic classes** | `ParkingLot` has all the logic; `Floor` and `ParkingSpot` are structs | Give each entity the operations that touch its own state |
| **God class** | One class with 40 methods and 12 fields | Split by axis of change, not by "size" |
| **Circular dependency** | `Floor` needs `ParkingLot` needs `Floor` | Introduce a third object (a manager) or push the reference down to one direction |
| **Over-engineering** | An interface for a single implementation that will never change | Two implementations is the honest threshold. One is speculation |
| **Pattern name-dropping** | "I'd use the Strategy pattern here" with no problem stated | State the problem first, then the pattern, then the cost. See §6.4 |
| **Memorised answer meets a twist** | A perfect parking-lot design collapses on "now add dynamic pricing" | Only the method transfers. Clarify, then re-derive |
| **Silent thinking** | Two minutes of staring at the board | Narrate continuously; the reasoning is what is being scored |
| **No stated trade-offs** | Every choice sounds like the only option | Close with the four-part form: choice, reason, benefit, cost |
| **Time overrun on the code** | 30 of 45 minutes gone writing bodies | Write signatures and 2–4 critical methods. Never write getters or boilerplate |
| **Ignoring validation** | `reserveSpot` on an occupied spot silently succeeds | Enumerate invalid transitions; every one of them needs a defined answer |
| **Unbounded state** | Holds, tickets, sessions accumulate forever | Every long-lived structure needs a TTL or a bounded size |

### 6.3 The failure modes of *your own design*, which you should name before being asked

For any LLD design, these six always apply. Say which ones bite and what you did:

```
1. UNBOUNDED GROWTH     which collection can grow without limit?
                        → tickets map, holds, audit log, sessions
                        → fix: TTL, cap, or move to a store with a retention policy

2. CONTENTION           what is the hottest lock or the hottest row?
                        → a single seat row at a hot show; a per-lot lock
                        → fix: finer-grained locks, sharding, optimistic retries

3. PARTIAL FAILURE      the operation that has two steps and the second can fail
                        → hold succeeded, payment failed; reserve succeeded, ticket write failed
                        → fix: idempotency key + compensating action + a saga

4. INVALID TRANSITION   the state machine's illegal moves
                        → release twice; hold a booked seat; book an expired hold
                        → fix: explicit states and a transition guard on every mutator

5. RESOURCE EXHAUSTION  thread pool, connection pool, memory
                        → a sweeper that holds a lock for the whole scan
                        → fix: incremental sweeping in bounded batches

6. CLOCK AND TIMEOUT    TTL measured against whose clock?
                        → two app instances disagreeing on "now"
                        → fix: one clock authority; measure TTLs on the store side
```

### 6.4 Pattern misuse — the honest exceptions

Patterns are covered in [03-design-patterns.md](03-design-patterns.md), but the failure modes belong here because they are LLD failure modes:

| Misuse | Why it is wrong | What to do instead |
|---|---|---|
| **Singleton everywhere** | Global mutable state is the enemy of testing and of concurrency. A `ParkingLot` singleton cannot be instantiated twice in a test | Pass the instance in (dependency injection); make it a singleton only at the composition root |
| **Inheritance for reuse** | `Floor extends ParkingLot` because it needed one helper method | Composition, or a free function. See [02-oop-principles-and-solid.md](02-oop-principles-and-solid.md) |
| **Decorator on a value type** | Wrapping an `int` in three decorators to add a fee | Add the fee in the pricing strategy |
| **Observer for synchronous control flow** | Notifying a listener that must complete before the caller proceeds | Call the method. Observer is for decoupling, not for ordering |
| **Factory for one product** | A `ParkingSpotFactory` that always returns `ParkingSpot` | A constructor |
| **Strategy with one strategy** | An interface with one implementation | The concrete class, until the second implementation is real |
| **State pattern for two states** | A full state machine for a boolean | The boolean |

**The threshold rule:** introduce the abstraction when the *second* implementation is real or imminent, not when it is imaginable. In an interview, say both: *"I'd add the interface here because the interviewer is about to ask for a second allocation policy — in production I'd wait until the second policy actually existed."* That answer demonstrates both design skill and restraint, and restraint is the scarcer signal.

### 6.5 Boundary-case exceptions

| Case | The issue | Handling |
|---|---|---|
| **The prompt is genuinely both** | "Design a ride-hailing service" — the interviewer wants the matching algorithm's classes and the fleet architecture | Cover both briefly, then ask which to deepen. Announce the switch |
| **The prompt is a knowledge question** | "What's the difference between an interface and an abstract class?" | Answer directly, then bridge: *"and here's where that shows up in a design…"* |
| **The prompt is a pure algorithm** | "Reverse a linked list" | It is not LLD. Answer it, then move on |
| **You realise mid-way that the model is wrong** | The class you called `Seat` actually needs to be `SeatAssignment` scoped to a show | Say so and refactor visibly. Recovering in public is a positive signal |
| **The interviewer rejects your structure** | "Why is `Ticket` owned by `Spot`?" | Do not defend reflexively. Ask what property they want, then re-derive. Changing your mind on evidence is senior behaviour |
| **The language is constrained** | "Do it in Java" and you have been practising Python | The structure is identical; the syntax is lookup. Say the design, write the skeleton, and note the idiom you would use |
| **You run out of time before concurrency** | 40 minutes gone | Stop and do the concurrency section verbally in 3 minutes. A named race and a named fix beats an unfinished method body |

---

## 7. When to use / When NOT to use

### 7.1 Which shape is being asked for

| Prompt signal | Shape |
|---|---|
| Names a **product** ("design Twitter", "design Uber") | HLD |
| Contains **"at scale"**, "millions of users", "globally" | HLD |
| Asks for **"the architecture"**, "components", "diagram", "endpoints", "SLA", "cost" | HLD |
| Names a **mechanism or data structure** ("LRU", "rate limiter in-process", "seat lock") | LLD |
| Says **"classes"**, "interfaces", "design patterns", "OOP", "extensible" | LLD |
| Asks for **"complexity"**, "thread-safe", "refactor" | LLD |
| Asks to **implement**, **code**, or **build** | LLD → machine coding |

### 7.2 When to apply the full LLD method

| Situation | Apply? |
|---|---|
| "Design a parking lot / elevator / ATM / library / hotel / Splitwise" | ✅ Full method |
| "Implement an LRU / LFU cache" | ✅ Full method, weighted to structures and concurrency |
| "Design a seat-locking service" | ✅ Full method, weighted to concurrency |
| "Design a payment gateway SDK" | ✅ Weighted to interfaces and versioning |
| "Design an order lifecycle" | ✅ Weighted to the state machine and idempotency |
| The deep-dive step of an HLD round | ✅ Abbreviated — the semantics are already established |
| A machine-coding round's opening conversation | ✅ It is the design phase of the coding round |

### 7.3 When NOT to apply it

| Situation | Why not | Do instead |
|---|---|---|
| "Design Instagram at scale" | It is HLD | Five-step framework: [04-five-step-interview-framework.md](../HLD/01-foundations/04-five-step-interview-framework.md) |
| "Estimate storage for 100M users" | A sub-step of HLD | Capacity estimation playbook |
| "What is CAP theorem?" | Knowledge question | Answer, then bridge to a design |
| "Debug this production incident" | Time-critical | Mitigate first; the framework afterwards |
| "Review this pull request" | Different discipline | Code review practice |
| A trivia internal tool | Over-engineering | Right-size and say so |

### 7.4 What NOT to do in an LLD round

- Do **not** design multi-region replication.
- Do **not** estimate QPS or storage — unless the concurrency discussion needs a number, in which case use a *contention* number ("10,000 concurrent reservers on one show"), not a capacity number.
- Do **not** draw CDN or load-balancer topology.
- Do **not** discuss shard keys, unless the data structure you chose *is* the sharding decision.
- Do **not** write getters, setters and boilerplate code.

Both levels are correct in their place. Using one in the other reads as a failure to read the room, and it is the most avoidable failure in the whole interview.

---

## 8. Numbers, defaults & tuning

### 8.1 Round structure defaults

| Parameter | Default | Notes |
|---|---|---|
| Total LLD round length | **45–60 min** | 60 increasingly common for senior |
| Clarify semantics | **5 min** (45-min round) | 8 min if 60 |
| Entities + relationships | **5 min** | Write nouns down; draw as you go |
| Interfaces | **5 min** | Signatures only |
| Data structures + complexity | **10 min** | The proof section |
| Core code | **10 min** | 2–4 methods, never boilerplate |
| Concurrency | **5 min** | The race, the fix, the contention |
| Edge cases + extensibility + trade-offs | **5 min** | Protect this |
| Clarifying questions to ask | **5–8** | Fewer looks careless; more wastes the clock |
| Classes in a good LLD diagram | **5–10** | Below 5 means missing entities; above 12 means over-decomposition |
| Methods per class | **2–6** | Public-facing; excluding accessors |
| Members per class | **1–4** | More suggests the class is doing two jobs |
| Extension-point interfaces | **1–3** | One per axis of *likely* change, not per axis of imaginable change |
| Trade-off statements | **≥3** | Each in four-part form |
| Time before a diagram must exist | **≤15 min** | An unfinished diagram plus trade-offs beats a finished diagram without |

### 8.2 Complexity anchors to quote

| Structure | Access | Insert | Delete | Search | When it wins |
|---|---|---|---|---|---|
| Dynamic array / list | O(1) | O(1) amortised | O(n) | O(n) | Iteration dominates; order matters; n is small |
| Hash map | O(1) avg | O(1) avg | O(1) avg | O(1) avg | Key lookup dominates; order irrelevant |
| Balanced BST / sorted map | O(log n) | O(log n) | O(log n) | O(log n) | Ordered iteration + lookup |
| Doubly linked list | O(n) | O(1) | O(1) given node | O(n) | O(1) reordering given a node — the LRU backbone |
| Heap | O(1) peek | O(log n) | O(log n) | O(n) | You need the min/max repeatedly |
| Skip list | O(log n) | O(log n) | O(log n) | O(log n) | Ordering + concurrency-friendly (Redis ZSet) |
| Trie | O(k) | O(k) | O(k) | O(k) | Prefix search — autocomplete, routing tables |
| Ring buffer | O(1) | O(1) | O(1) | O(n) | Fixed-capacity recent-N — sliding windows |

### 8.3 Contention anchors — the numbers that make an LLD concurrency answer concrete

| Quantity | Typical figure | Why it matters |
|---|---|---|
| Single mutex, uncontended acquire | **~20 ns** | Free when nobody contends |
| Single mutex, contended acquire | **µs–ms** | Serialisation + context switch + queueing |
| Mutex throughput ceiling | **~1–10 M lock/unlock per second** | One lock is one serialisation point |
| Practical threads per critical section | **4–16** | Beyond this, contention destroys the benefit |
| Database row lock hold time | **1–20 ms** | Why long transactions are dangerous |
| Optimistic retry cost | **1 round trip per collision** | Retry storms collapse throughput exactly when load is highest |
| Redis single-key ops/sec | **~100k–200k** | One key is one slot on one node |
| Redis `SET NX` lock latency | **0.2–1 ms** | The cheap distributed mutex |
| Thread pool size (CPU-bound) | **≈ #cores** | More threads do not add throughput |
| Thread pool size (I/O-bound) | **#cores × (1 + wait/service)** | Little's Law |
| Connection pool per instance | **10–50** | More is not better |

### 8.4 Defaults to state out loud when the interviewer will not answer

| Question | Default to assume | Say it as |
|---|---|---|
| "How many users?" | 1 M DAU for an LLD problem | "I'll assume a million users — the class design is what changes with scale, so tell me if that's wrong" |
| "How many concurrent operations on one entity?" | 10,000 | The contention number is what drives the locking choice |
| "Single process or distributed?" | **Start single-process, then say what changes** | "I'll design it in-process and then say which parts move to a shared store" |
| "How long is a hold / lease / TTL?" | 5–10 minutes | The standard for a user-facing checkout step |
| "What happens to expired state?" | A sweeper plus lazy validation | Both are needed; see §6.3 |
| "Exactly-once or at-least-once?" | At-least-once + idempotency keys | Cheaper and more honest than exactly-once |
| "Money or not?" | If money, strong consistency and idempotency are non-negotiable | Say it before being asked |
| "Which language?" | The one you will actually write in | Do **not** write pseudo-code that is neither |

---

## 9. Interview Q&A

**Q1 (easy). What is LLD, in one sentence?**
LLD is deciding how one component is built — what classes exist, what methods they have, how the objects interact, and which OOP concepts apply. The source's phrasing is that you stop looking at the big picture and come down to a class: one component and its code structure.

**Q2 (easy). What is the difference between HLD and LLD?**
HLD decides what components exist and how they are connected — boxes, labelled arrows, a flow-chart. LLD decides the internals of one of those boxes — classes, methods, constructors, how they interact. HLD's arrow carries a protocol; LLD's arrow carries an ownership or inheritance relationship.

**Q3 (easy). Why do interviewers ask LLD questions?**
The source names four reasons: they want to know whether you can think at scale level rather than just for ten users, whether you can make trade-offs with proper reasoning, whether you can communicate clearly under pressure, and whether you know the right tool for the right job. Added to that: LLD is falsifiable, so it tests craft — structure, abstraction, correctness and concurrency — in a way that architecture does not.

**Q4 (easy). Name the LLD counterpart of the Instagram example.**
The parking lot. A `ParkingLot` holds floors in a List; each `Floor` holds spots in a List and exposes `findAvailableSpot()`, `reserveSpot()` and `releaseSpot()`; each `ParkingSpot` knows its id, its occupancy and its vehicle type. The source uses it to show what "one component and code structure" means.

**Q5 (medium). Walk me through designing a parking lot.**
Clarify: multi-floor, vehicle types, spot sizes, pricing model, gates, reservations, payment. Entities: `ParkingLot`, `Floor`, `ParkingSpot`, `Vehicle`, `Ticket`, `EntryGate`, `ExitGate`. Relationships: lot 1–N floor, floor 1–N spot, ticket 1–1 spot and 1–1 vehicle. Interfaces: `findAvailableSpot(type)`, `reserveSpot(spot, vehicle)`, `releaseSpot(ticket)`, plus `PricingStrategy.calculate(ticket, now)` and `SpotAllocator.allocate(lot, type)`. Structures: a list of spots per floor, plus a per-floor free-count keyed by size so "is anything free?" is O(1); a ticket map for O(1) exit lookup. Complexity: reserve and release O(1); find O(k) worst case or O(1) with free-lists. Concurrency: two threads reserving the last spot double-sell it — fix with a per-spot compare-and-set, or a per-floor lock, and discuss the contention that each buys. Extensibility: `SpotAllocator` and `PricingStrategy` are interfaces, so nearest-first vs cheapest-first and hourly vs surge are new classes. Edge cases: lot full, double release, lost ticket, overstay, ticket from another lot. Trade-offs: per-floor locks are simple and serialise a floor; per-spot CAS scales but needs retry loops.

**Q6 (medium). Why does LLD care about complexity when HLD cares about QPS?**
Because both are answers to "will this survive scale?" expressed in different units. HLD counts requests per second; LLD counts operations per call. An O(n) scan inside a method that runs 10,000 times a second is 10,000 × n operations per second — the same scale problem, discovered at a different altitude.

**Q7 (medium). How do you decide where to put a method — on `ParkingLot` or on `Floor`?**
The rule I use is that an operation belongs on the object whose state it mutates. Reserving a specific spot mutates the spot and needs the floor's free-list, so it belongs on `Floor`, with `ParkingLot` delegating because it is the system-facing entry point. If a method needs to touch state in three classes, that is a signal that either ownership is wrong or a third coordinator object is missing.

**Q8 (medium). What is the difference between an LLD round and a machine-coding round?**
Same subject, different deliverable. LLD produces a design: a class diagram, interfaces, method signatures, complexity and a concurrency plan — and little or no code. Machine coding produces a running program with tests. The design conversation still happens at the start of a machine-coding round, which is why you prepare for both together.

**Q9 (medium). How do you handle the "now add X" follow-up?**
I first check whether my design already absorbs it. If I have an interface at the right extension point, adding X is a new class plus one line of wiring, and I say that explicitly — "this is a new `PricingStrategy`; `ParkingLot` does not change." If the extension cuts across my structure, I say so honestly and describe the refactor, because admitting that a design needs to change under a new requirement is a correct answer and pretending otherwise is not.

**Q10 (medium). What does a good class diagram contain, and what does it leave out?**
It contains the classes, their key members, the relationships with cardinality, and the interfaces marked as interfaces. It leaves out getters and setters, private helpers, framework types, and anything that does not carry a design decision. Five to ten classes is the right size; more than twelve usually means I have decomposed by size rather than by responsibility.

**Q11 (hard). *Pushback:* "Isn't LLD just coding — why is it a separate interview round?"**
Coding is producing working code. LLD is deciding the shape of the code before writing it: boundaries, interfaces, extension points, where the complexity budget goes, and what happens under concurrency. Those decisions are made in the first fifteen minutes of writing a class and they determine how expensive the next two years of changes are. The distinction is visible in the interview too — a candidate can write correct code with terrible class boundaries, and that is exactly what an LLD round is designed to expose.

**Q12 (hard). *Pushback:* "You named Strategy, Factory and Observer. Aren't you just name-dropping patterns?"**
No, and I try to make that checkable. Each one answers a specific requirement I stated. Pricing varies by vehicle, time and demand, so pricing must be substitutable at runtime — that is Strategy, and its cost is an extra indirection and a wiring point. Adding an EV spot type must not require editing `ParkingLot`, so creation goes through a Factory, and its cost is a layer of indirection for a construction that is otherwise trivial. Gates must learn that a spot freed up, and I do not want the lot to know about gates, so the lot publishes and gates subscribe — that is Observer, and its cost is that the notification is asynchronous and can be lost, which is why it is not used for anything correctness-critical. If a pattern does not come with a problem and a cost, it is decoration.

**Q13 (hard). *Pushback:* "Your design has a per-floor lock. What happens at 10 million users?"**
Then the floor lock is the bottleneck and I would change the granularity. The progression is: a single lot-wide lock (simplest, worst contention), a per-floor lock (good when floors are independent), per-spot atomic compare-and-set on the occupancy flag (fine-grained, needs retry handling), and finally an atomic free-list per spot size so allocation does not scan. Past a single process I move the reservation primitive to a store that can do an atomic conditional write — Redis `SET NX` with a TTL, or a database unique constraint — and I make the in-process lock an optimisation rather than the correctness guarantee. The point is that the *design* of the lock is what changes with scale, which is the same scale thinking HLD asks for.

**Q14 (hard). How does LLD change for an AI/LLM system?**
The vocabulary transfers almost intact. An agent loop is a state machine — states are `THINK`, `ACT`, `OBSERVE`, `DONE`, with guards on every transition and a hard iteration cap. A tool registry is a Factory plus an Adapter: one interface `Tool.run(args)`, one adapter per external API. A guardrail pipeline is a chain — input checks, then retrieval trust checks, then output checks, then action checks — which is a Chain of Responsibility or a Decorator chain depending on whether a stage can short-circuit. A prompt or query builder is the Builder pattern when the prompt has many optional parts, which it does. A semantic cache is a Strategy over an eviction policy. What is genuinely new is that correctness is probabilistic, so the "edge cases" section gains an evaluation dimension — you cannot enumerate the wrong answers, you measure them.

**Q15 (hard). You have 45 minutes and the prompt is a boundary case — "design BookMyShow". What do you do first?**
I ask which level is wanted, and I make the question concrete rather than vague: *"I can take this at the architecture level — gateway, booking service, seat store, payment, queue, cache — or at the class level, which is where the interesting part is: the seat-hold model, the locking strategy, and the race between two users confirming the same seat. Which is more useful?"* If the interviewer says "you decide", I do a compressed HLD in five minutes — enough to locate the components — and then spend the remaining time on the seat-hold LLD, because that is where the engineering is. I announce the switch and keep both on the board so the interviewer can redirect me.

**Q16 (medium). What is the most common LLD mistake you see?**
Choosing a data structure without stating the complexity, and ignoring concurrency. Both are the things the round exists to measure. A candidate who says "I'll use a hash map plus a doubly linked list, which gives O(1) get, put and eviction, and here is the lock that makes it thread-safe" has answered the question. A candidate who says "I'll use a hash map" has not started.

---

## 10. Common mistakes candidates make

1. **Not identifying which shape the question is.** The single most avoidable failure in the whole interview.
2. **Drawing classes in an HLD round.** Reads as junior.
3. **Designing multi-region in an LLD round.** Reads as off-topic.
4. **Starting to draw before clarifying semantics.** The LLD equivalent of skipping requirements.
5. **Never stating complexity.** The data structure choice is unjustified without it.
6. **Never mentioning concurrency.** Thread safety is what separates senior from mid.
7. **No extension point.** The interviewer will ask "how would you add X?", and it is always asked.
8. **Anemic classes** — one class holds all the behaviour and the rest are structs.
9. **God classes** — decomposing by size instead of by axis of change.
10. **Interfaces with one implementation, for a change that will never happen.** The honest threshold is two.
11. **Pattern name-dropping** with no problem and no cost attached.
12. **Writing boilerplate code** — getters, setters, `toString` — while the clock runs.
13. **Not handling invalid transitions.** Release twice, book an expired hold, reserve an occupied spot.
14. **Ignoring unbounded growth.** Every long-lived collection needs a TTL or a cap.
15. **Memorising the famous questions.** The interviewer's twist defeats the memorised answer; only the method transfers.
16. **Silent thinking.** The reasoning is half the score, and it is only scored if it is audible.
17. **Not narrating the level transitions.** The interviewer loses the thread when you drop from architecture to method signatures without saying so.
18. **Skipping the trade-off close.** An LLD answer without a stated sacrifice reads as a preference.

---

## 11. Cheat-sheet summary

| | |
|---|---|
| **Source** | Day 1 transcript, [25:54]–[27:38] — the only LLD segment in the collated source |
| **HLD** | *"deciding what components exist and how they are connected"* — boxes, labelled arrows, a flow-chart |
| **HLD example** | Instagram: user phone → API gateway → upload → feed service → S3 bucket → Redis cache |
| **LLD** | *"one component and code structure"* — classes, methods, constructors, *"the classes, methods, and how they interact"* |
| **LLD example** | Parking lot: `ParkingLot` → `List<Floor>` → `Floor` (`findAvailableSpot`, `reserveSpot`, `releaseSpot`) → `List<ParkingSpot>` → `ParkingSpot` |
| **Why they ask** | Scale thinking · trade-offs with reasoning · communicating under pressure · the right tool for the right job |
| **The test** | Not "can you write a class" — can you structure, abstract, prove and extend |
| **Round shapes** | HLD (architecture) · LLD (class design) · machine coding (running code + tests) |
| **Method** | Clarify → entities → relationships → interfaces → structures → complexity → concurrency → code → extensibility → edge cases → trade-offs |
| **The proof** | Complexity justifies every structure; the lock justifies every mutation |
| **The close** | Name the extension point, then state what each choice cost |
| **The failure** | A design that is not extensible and has no concurrency story, delivered silently |

**The one-liner:** *"HLD is boxes and labelled arrows; LLD is classes, methods and how they interact — and in LLD the complexity proof and the concurrency story are what is actually being scored."*

**"If the interviewer says X → do Y"**

| Interviewer says | You do |
|---|---|
| "Design a parking lot" | Entities → class diagram → interfaces → complexity → the concurrent-reserve race → "now add X" |
| "Implement an LRU cache" | Hash map + doubly linked list with sentinels → O(1) proof → lock striping → TTL variant |
| "Design Instagram" | HLD. Five-step framework. No classes |
| "Go deeper on the booking service" | Now LLD: entities, interfaces, the seat-hold state machine, the race |
| "Now add EV charging" | "That's a new `SpotSize` and a new `SpotAllocator` — `ParkingLot` does not change" |
| "Is it thread-safe?" | Name the race, name the primitive, name what it serialises, name the contention cost |
| "Why that data structure?" | Complexity per operation, and the alternative you rejected with its cost |
| "You have five minutes left" | Stop coding. Do the edge cases, the extension and the trade-offs, out loud |

---

## 12. Cross-links

- Next in this pillar: [OOP principles and SOLID](02-oop-principles-and-solid.md) · [Design patterns](03-design-patterns.md) · [Concurrency and threading in LLD](04-concurrency-and-threading-in-lld.md) · [UML and class diagrams](05-uml-and-class-diagrams.md) · [The machine-coding interview framework](06-machine-coding-interview-framework.md)
- The source's own HLD/LLD split, from the same transcript segment: [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md)
- The HLD counterpart method: [The five-step interview framework](../HLD/01-foundations/04-five-step-interview-framework.md)
- Worked problems in this pillar: [Parking lot](problems/01-parking-lot.md) · [LRU and LFU cache](problems/02-lru-lfu-cache.md) · [Hotel management](problems/03-hotel-management.md) · [BookMyShow seat locking](problems/04-bookmyshow-seat-locking.md) · [Elevator system](problems/05-elevator-system.md) · [Splitwise](problems/06-splitwise.md) · [ATM machine](problems/07-atm-machine.md) · [Library management](problems/08-library-management.md)
- The boundary problem worked at both levels: [BookMyShow seat booking case study](../Case-Studies/HLD/07-bookmyshow-seat-booking.md)
- Foundations this rests on: [What is system design and why](../HLD/01-foundations/01-what-is-system-design-and-why.md) · [Six goals and trade-offs](../HLD/01-foundations/02-six-goals-and-tradeoffs.md)
- Revise here: [Foundations and estimation cheat sheet](../Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md)
- Drill here: [Foundations and capacity questions](../Interview-Questions/01-foundations-and-capacity-questions.md)

> **Grounding note.** This file's definition, the Instagram chain, the parking-lot hierarchy and the four interviewer motivations are all taken from the source transcript at [25:54]–[27:38]. The method, the time budgets, the failure-mode taxonomy, the complexity table and the Q&A are standard industry practice assembled for this knowledge base. The reference repositories contain no OOP or LLD material, and the transcript names LLD topics (LRU/LFU, hotel management, BookMyShow seat locking) **without teaching them in depth**. Treat the transcript as the source of the *distinction* and this pillar as the source of the *practice*.
