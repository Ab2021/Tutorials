# UML and Class Diagrams — Drawing Legibly in Ten Minutes

> **Grounding note.** The source transcript names diagrams at **[25:54]–[27:38]**, and this is one of the few places it gives something concrete: LLD is *"when you don't look at OOPS concepts... you come to a class... you draw a class diagram, define the methods, apply the OOP concept."* It also describes the parking lot as a hierarchy of `ParkingLot` → `Floor` → `ParkingSpot`, held *"in the form of a List"*. **That is the whole of the transcript's UML teaching: draw a class diagram, define the methods.** It never names a diagram type beyond the class diagram, never mentions sequence or state diagrams, notation, cardinality, or visibility markers. Everything below beyond "draw a class diagram with methods" is standard industry practice, and the notation conventions come from the UML specification rather than from the course.

---

## TL;DR

A UML diagram in an interview is **a communication device with a ten-minute budget, not an artefact.** You draw four things, in this order: a **class diagram** (what exists and how it is connected), a **sequence diagram** for the one flow that matters (what happens, in what order), a **state diagram** if something has a lifecycle (which transitions are legal), and occasionally an **activity diagram** for a complex conditional flow. The notation that actually matters is small: three relationship types (association, aggregation, composition), cardinality at the ends of a line, and the ability to distinguish a concrete class from an abstract one and an interface. Everything else in UML — component diagrams, deployment diagrams, timing diagrams, the seventeen diagram types — is out of scope for an LLD round. The failure mode is not drawing too few diagrams; it is drawing an illegible one, drawing a class diagram where a sequence diagram was needed, or spending twenty minutes on notation instead of nine on design. **The interviewer is reading your diagram while you talk: if they cannot follow it, the design does not exist.**

> **60-second interview answer:** "I draw three diagrams and I keep them small. A class diagram to establish the entities and the relationships — boxes with the key fields and the methods that matter, with composition marked with a filled diamond and cardinality at the ends. A sequence diagram for the critical flow, because ordering is where concurrency bugs live, and it is the one diagram that makes 'two users, one seat' visible on the whiteboard. A state diagram for anything with a lifecycle, since the legal transitions are a design decision I want to be auditable. I keep each to roughly five to nine boxes, I write the method signatures rather than full fields, and I deliberately omit constructors and getters, because they add no information and cost time. Then I say what I left out and why."

---

## 1. Definition & mental model

### 1.1 What a diagram is for, in an interview

| Purpose | The diagram | The interviewer's question it answers |
|---|---|---|
| **Establish vocabulary** | Class diagram | "What are the nouns, and can I trust that you picked the right ones?" |
| **Show ordering** | Sequence diagram | "Do you know what happens in what order, and where the race is?" |
| **Show legality** | State diagram | "Do you know which transitions are illegal?" |
| **Show branching** | Activity diagram | "Can you handle the conditional paths without an `if` jungle?" |

**The mental model:** a diagram is a *shared whiteboard that both of you point at*. Its value is in the conversation, not in the picture. A perfect diagram nobody can follow is worth zero; a rough box-and-arrow sketch that anchors a good discussion is worth a lot.

### 1.2 The ten-minute budget

An LLD round is 45 minutes and the design part is roughly 25. Within that, drawing should never exceed a third.

| Phase | Time | What you draw |
|---|---|---|
| Requirements and entities | 8–10 min | Nothing, or a scratch list. **Do not start drawing yet** |
| Class diagram | 5–7 min | 6–10 boxes, relationships, cardinality, key methods only |
| Sequence diagram (the critical flow) | 4–5 min | 4–6 participants, 6–12 messages, the race annotated |
| State diagram (if there is a lifecycle) | 2–3 min | 3–6 states, transitions, illegal transitions marked |
| Extend, trade-offs, Q&A | 10–15 min | Annotations on the existing diagram, not a new one |

**The single most common timing error:** starting to draw at minute two, before the requirements are clear, and then redrawing everything at minute twenty. Say the requirements out loud first.

### 1.3 What to draw, and what to omit

| Always include | Deliberately omit | Why |
|---|---|---|
| Class name | Constructors | They carry no design information; the entity's identity is given |
| Key fields (2–4 per class) | Every field | The ones that carry invariants matter; the rest is noise |
| Public methods that carry behaviour | Getters, setters, `__repr__` | They are implementation, not design |
| Relationships with cardinality | Every relationship | Show the ones you will discuss |
| Abstract/interface marking | Visibility markers on every member | `+`/`−` matters for one example, not all |
| The concurrency annotation | Types of every parameter | `reserve(spot, vehicle)` reads fine without `: Ticket` on every arrow |

**The rule of thumb:** if a reader cannot tell your design from the diagram plus ninety seconds of speech, the diagram has too much on it.

---

## 2. Why it exists — the problem it solves

### 2.1 What breaks without a diagram

- **The interviewer cannot audit the design.** A verbal answer cannot be checked. A diagram lets them point at the box where the race lives.
- **You cannot see your own mistake.** The two-users-one-seat race is *invisible* in prose and obvious in a sequence diagram with two lifelines.
- **Scope is unmanaged.** A diagram makes "how many classes is this?" answerable, so you notice when the design has grown to forty.
- **The follow-up is expensive.** "Now add a second pricing model" is a one-line change to a diagram and a five-minute re-derivation in prose.
- **You lose the thread.** Under pressure, candidates forget their own entities. The diagram is the external memory.

### 2.2 Why the class diagram comes first

It fixes the vocabulary. Every subsequent sentence — "the allocator picks a spot, the lot creates the ticket" — only parses if both of you already agree what `SpotAllocator` and `Ticket` are. The class diagram is a naming ceremony that happens to look like a picture.

**And the sequence diagram is where the design actually gets tested.** Candidates who draw only a class diagram reliably miss the concurrency question, because the class diagram cannot express "these two things happen at the same time".

### 2.3 The honest counter-argument

**In a machine-coding round with a running IDE, nobody wants a diagram.** The code is the diagram, and drawing boxes wastes time you need for tests. The position to take: *"in a pure LLD discussion I lead with a class diagram; in a machine-coding round I would sketch the five entities for thirty seconds and then start coding, because the code is the artefact being judged."* Knowing *which round you are in* is part of the answer — see [the machine-coding framework](06-machine-coding-interview-framework.md).

---

## 3. Internal mechanics — the notation that matters

### 3.1 The class box

```
  ┌─────────────────────────────────────────┐
  │            ParkingLot                   │  ← name (italic if abstract)
  ├─────────────────────────────────────────┤
  │ - lot_id: str                           │  ← fields:  - private, + public, # protected
  │ - floors: List[Floor]                   │
  │ - pricing: PricingStrategy              │  ← type is an interface → composition coming
  ├─────────────────────────────────────────┤
  │ + reserve(vehicle) : Ticket             │  ← methods with key signatures only
  │ + release(ticket)  : Money              │
  │ + available(size)  : int                │
  └─────────────────────────────────────────┘

  ┌─────────────────────────────────────────┐
  │           «interface»                   │  ← stereotype for an interface
  │          PricingStrategy                │
  ├─────────────────────────────────────────┤
  │ + calculate(ticket, exit_at) : Money    │
  └─────────────────────────────────────────┘

  ┌─────────────────────────────────────────┐
  │              Vehicle                    │  ← italic name = abstract class
  │            {abstract}                   │
  ├─────────────────────────────────────────┤
  │ # plate: str                            │
  │ # size: SpotSize                        │
  └─────────────────────────────────────────┘
```

**Visibility markers.** `+` public, `−` private, `#` protected, `~` package. In an interview, use `−` and `+` and skip the rest. Writing them for every member wastes time; writing them for one class demonstrates you know the convention.

**Abstract versus concrete versus interface.** This distinction is the one that carries design meaning, because it tells the reader what can be instantiated and what must be implemented:

| Notation | Means | Python equivalent |
|---|---|---|
| *Italic* class name | Abstract class | `class Vehicle(ABC)` |
| `{abstract}` on a method | Abstract method | `@abstractmethod` |
| `«interface»` stereotype | Interface — a contract, no state | `class PricingStrategy(ABC)` with only abstract methods |
| Plain name | Concrete, instantiable | `class Car(Vehicle)` |

### 3.2 Relationships — the five that matter

This is where candidates lose marks, because the difference between aggregation and composition is a *semantic* difference about ownership and lifetime, and interviewers probe it.

```
  ASSOCIATION  ──────────────►        "knows about / uses"
                                       A ParkingLot knows a PricingStrategy.
                                       Both live independently.

  AGGREGATION  ◇─────────────►        "has, but the parts outlive the whole"
   (hollow diamond)                    A Campus has ParkingLots. Delete the Campus;
                                       the lots still exist.

  COMPOSITION  ◆─────────────►        "owns; the parts die with the whole"
   (filled diamond)                    A ParkingLot is composed of Floors, and a
                                       Floor of ParkingSpots. Delete the lot and
                                       its floors and spots are meaningless.

  INHERITANCE  ──────────────▷        "is-a"
   (hollow triangle)                   Car ──▷ Vehicle.  Use sparingly (see 02).

  REALIZATION  - - - - - - - ▷        "implements an interface"
   (dashed + hollow triangle)          HourlyPricing - -▷ PricingStrategy
```

**Reading direction matters and is often drawn backwards.** The diamond sits at the *owner* end, and the arrow (for association) points at the *known* object. `ParkingLot ◆── Floor` reads "a ParkingLot is composed of Floors".

**The test an interviewer will apply:** *"If I delete the parent, does the child still make sense?"* Yes → aggregation. No → composition. `Campus`/`ParkingLot` is aggregation (the lots exist without the campus). `ParkingLot`/`Floor` is composition (a floor with no lot is unmoored). Say that test out loud; it is more convincing than the notation.

### 3.3 Cardinality

Notation: `1`, `0..1`, `*` (zero or more), `1..*` (one or more), `n`, `0..n`.

```
  ParkingLot  1 ◆──────── 1..* Floor      one lot has one or more floors
  Floor       1 ◆──────── 0..* ParkingSpot  a floor has zero or more spots
  ParkingSpot 0..1 ─────── 0..1 Ticket      a spot has at most one active ticket
  Booking     *  ────────  1    User        many bookings per user
  Show        1  ────────  *    Seat        a show has many seats
```

**Why the multiplicities matter, and this is the interview point:** **the cardinality is where the invariant lives.**

- `ParkingSpot 0..1 ──── 0..1 Ticket` says a spot has *at most one* active ticket. That is the double-booking invariant, drawn.
- `ParkingSpot 1 ──── * Ticket` would say a spot may have many tickets — either a bug, or a deliberate history model where old tickets are retained but only one is active. **Which one it is is a design decision, and drawing it forces you to make it.**
- `Reservation * ──── 1 Room` over a date range is the overbooking question, and it is invisible without the date-range constraint noted beside the relation.

**So cardinality is not decoration.** It is the place where a reader checks whether you know what "one" means. Marking it is worth the three seconds per line.

### 3.4 The sequence diagram

```
  ┌────────┐    ┌────────────┐   ┌──────────┐   ┌───────────┐   ┌───────┐
  │ User1  │    │ BookingAPI │   │ SeatHold │   │  Payment  │   │  DB   │
  └───┬────┘    └─────┬──────┘   └────┬─────┘   └─────┬─────┘   └───┬───┘
      │ POST /book    │               │               │             │
      │──────────────►│               │               │             │
      │               │ hold(seats)   │               │             │
      │               │──────────────►│               │             │
      │               │               │ CAS: WHERE    │             │
      │               │               │ status=FREE   │             │
      │               │               │─────────────────────────────►│
      │               │               │ rowcount=1 ✓  │             │
      │               │               │◄─────────────────────────────│
      │               │ hold_id       │               │             │
      │               │◄──────────────│               │             │
      │               │ charge(amount, idem_key)      │             │
      │               │──────────────────────────────►│             │
      │               │               │      charge_id│             │
      │               │◄──────────────────────────────│             │
      │               │ INSERT booking (idem_key UNIQUE)            │
      │               │─────────────────────────────────────────────►│
      │               │ confirm(hold_id)              │             │
      │               │──────────────►│               │             │
      │               │               │ status=BOOKED │             │
      │               │               │─────────────────────────────►│
      │  201 Created  │               │               │             │
      │◄──────────────│               │               │             │

   ── THE RACE, drawn on the same diagram ──
  ┌────────┐                ┌────────────┐   ┌──────────┐
  │ User2  │                │ BookingAPI │   │ SeatHold │
  └───┬────┘                └─────┬──────┘   └────┬─────┘
      │ POST /book (same seat)    │               │
      │──────────────────────────►│               │
      │                           │ hold(seats)   │
      │                           │──────────────►│
      │                           │               │ CAS: WHERE status=FREE
      │                           │               │ → rowcount = 0  ✗
      │                           │ 409 Conflict  │
      │                           │◄──────────────│
      │◄──────────────────────────│               │
```

**Two things a sequence diagram makes visible that prose does not:**

1. **Ordering.** The hold must precede the charge. If it does not, you have charged for a seat you may not get — the single worst failure in this system.
2. **The simultaneous path.** Drawing both users on one diagram is how you *show* the race rather than assert it. This is the highest-value thirty seconds of drawing in an LLD round.

### 3.5 The state diagram

```
                ┌──────────────────────────────────────────┐
                │                                          │
                ▼                                          │
          ┌──────────┐   hold(user)   ┌──────────┐  confirm  │
          │   FREE   │───────────────►│   HELD   │──────────►┌──────────┐
          └──────────┘                └──────────┘           │  BOOKED  │
                ▲                          │                └──────────┘
                │                          │ release()             │
                │                          │  OR  TTL expiry       │ cancel()
                │                          ▼                       ▼
                │                    ┌──────────┐          ┌────────────┐
                └────────────────────│ (back to │          │ CANCELLED  │
                   compensations     │   FREE)  │          └────────────┘
                └────────────────────└──────────┘

  ILLEGAL (must be impossible, not merely unhandled):
    FREE   --confirm()-->  ✗   confirmed what?
    BOOKED --release()-->  ✗   booked seats are released by cancel(), not release()
    HELD   --hold(other)--> ✗  already held by someone else
```

**The convention that earns marks:** put the *illegal* transitions on the diagram, crossed out, with the reason. It shows you thought about invariants and not just the happy path.

### 3.6 The activity diagram, rarely needed

```
        ┌─────────┐
        │ START   │
        └────┬────┘
             ▼
      ╱──────────────╲
     ╱ spots free?    ╲──── no ──► ┌──────────────┐
     ╲                ╱            │ return 409   │
      ╲──────┬───────╱             └──────┬───────┘
             │ yes                        │
             ▼                            │
      ╱──────────────╲                    │
     ╱ payment ok?    ╲──── no ──► release hold ──┤
     ╲                ╱                          │
      ╲──────┬───────╱                           │
             │ yes                              │
             ▼                                  │
      ┌──────────────┐                          │
      │ issue ticket │                          │
      └──────┬───────┘                          │
             └──────────────► ┌────────┐ ◄───────┘
                              │  END   │
                              └────────┘
```

**Use it for one thing only:** a flow with three or more conditional branches, where the compensating actions on each failure path are the interesting part. It makes "what do I undo on each failure" visible. For two branches, write it in prose.

### 3.7 A complete worked class diagram

The parking lot, with the notation applied and the concurrency-relevant cardinality marked.

```
        ┌──────────────────────────────┐
        │          ParkingLot          │
        ├──────────────────────────────┤
        │ - lot_id: str                │
        │ - floors: List[Floor]        │
        │ - pricing: PricingStrategy   │
        │ - allocator: SpotAllocator   │
        ├──────────────────────────────┤
        │ + reserve(v: Vehicle):Ticket │
        │ + release(t: Ticket):Money   │
        │ + available(s: SpotSize):int │
        └───────┬──────────────┬───────┘
                │ 1            │ 1
                │              │
                │ ◆ 1..*       │ ────►  ┌─────────────────────┐
                │              │        │ «interface»         │
                ▼              │        │ PricingStrategy     │
        ┌──────────────────┐   │        ├─────────────────────┤
        │      Floor       │   │        │ + calculate(t, at)  │
        ├──────────────────┤   │        └──────────▲──────────┘
        │ - floor_id: int  │   │                   │
        │ - spots: List    │   │        ┌──────────┴──────────┐
        ├──────────────────┤   │        │ HourlyPricing       │
        │ + first_free(s)  │   │        │ SurgePricing        │
        └────────┬─────────┘   │        └─────────────────────┘
                 │ 1           │
                 │ ◆ 0..*      │ ────►  ┌─────────────────────┐
                 ▼             │        │ «interface»         │
        ┌──────────────────┐   │        │ SpotAllocator       │
        │   ParkingSpot    │   │        ├─────────────────────┤
        ├──────────────────┤   │        │ + choose(floor,s)   │
        │ - spot_id: str   │   │        └──────────▲──────────┘
        │ - size: SpotSize │   │                   │
        │ - status: Status │   │        ┌──────────┴──────────┐
        ├──────────────────┤   │        │ NearestFirst        │
        │ + try_hold(u,ttl)│   │        │ BalancedAllocator   │
        │ + confirm()      │   │        └─────────────────────┘
        │ + release()      │   │
        └────────┬─────────┘   │
                 │ 0..1        │
                 │ (at most one ACTIVE ticket — this is the
                 │  double-booking invariant, drawn)
                 ▼
        ┌──────────────────┐          ┌─────────────────────┐
        │      Ticket      │ 0..*     │      Vehicle        │ {abstract}
        ├──────────────────┤─────────►├─────────────────────┤
        │ - ticket_id: str │  1       │ # plate: str        │
        │ - entry_at: float│          │ # size: SpotSize    │
        │ - amount: Money  │          └──────────▲──────────┘
        └──────────────────┘                     │
                                     ┌───────────┼───────────┐
                                     │           │           │
                                  ┌──────┐   ┌──────┐   ┌───────┐
                                  │ Bike │   │ Car  │   │ Truck │
                                  └──────┘   └──────┘   └───────┘
```

**What this one diagram already tells the interviewer:**

- `pricing` and `allocator` are interfaces held by `ParkingLot` → Strategy, two axes of change isolated.
- `floors` and `spots` are filled-diamond compositions → lifetime tied to the parent.
- `ParkingSpot 0..1 ──── 0..1 Ticket` → exactly one active ticket per spot. The invariant is drawn.
- `Vehicle` is abstract with three concrete subtypes → polymorphism at the call site.
- `ParkingLot.reserve` returns a `Ticket`, so the caller never sees allocation.
- Nothing about concurrency yet — which is *why* the sequence diagram is drawn next.

---

## 4. Variants & taxonomy

### 4.1 The UML diagram types, and which matter in an LLD round

| Diagram | Shows | LLD relevance | Draw it when |
|---|---|---|---|
| **Class** | Types, attributes, methods, relationships | **Essential** | Always — first |
| **Sequence** | Messages between participants over time | **Essential** | Whenever there is a flow with ordering or concurrency |
| **State machine** | States and legal transitions | **High** | Anything with a lifecycle: seats, orders, rooms, tickets |
| **Activity** | Control flow with branches and joins | **Medium** | Three or more conditional branches, or compensating actions |
| **Object** | A snapshot of instances | Low | Almost never; a sequence diagram does it better |
| **Use case** | Actors and goals | Low | Requirements phase, if ever |
| **Component** | Deployable units and interfaces | HLD, not LLD | At the HLD boundary — see [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md) |
| **Deployment** | Nodes and artefacts | HLD | Not in an LLD round |
| **Package** | Module grouping | Low | Only if asked about module structure |
| **Communication** | Same info as a sequence diagram, laid out by object | Low | Never in preference to a sequence diagram |
| **Timing** | State changes against a clock | Very low | Only for a hard real-time system |
| **Interaction overview** | Activity diagram of sequence diagrams | Never | — |
| **Composite structure** | Internal parts of a class | Never | — |
| **Profile** | UML extensions | Never | — |

**The headline:** of the fourteen diagram types in UML 2, **four** are worth knowing for an LLD round, and the class and sequence diagrams carry almost all the value. Say that if asked — it shows you know the scope rather than that you forgot the others.

### 4.2 Association variants

| Variant | Notation | Meaning | Example |
|---|---|---|---|
| **Unidirectional association** | `A ────► B` | A knows B; B does not know A | `Ticket → ParkingSpot` |
| **Bidirectional association** | `A ──── B` | Both know each other | `User ↔ Booking` — usually a smell; prefer unidirectional |
| **Aggregation** | `A ◇─── B` | A has B; B can outlive A | `Campus ◇─ ParkingLot` |
| **Composition** | `A ◆─── B` | A owns B; B dies with A | `ParkingLot ◆─ Floor` |
| **Qualified association** | A box on the end | Indexed lookup rather than a list | `ParkingLot ─[lot_id]─ Floor` — implies a map |
| **Dependency** | `A - - -► B` | A uses B transiently, holds no reference | `TicketService - -► Clock` |
| **Realization** | `A - - ▷ B` | A implements interface B | `HourlyPricing - -▷ PricingStrategy` |
| **Generalization** | `A ──▷ B` | A inherits B | `Car ──▷ Vehicle` |

**Dependency is under-used and worth knowing.** A service that takes a `Clock` as a method parameter, rather than storing it, has a *dependency*, not an association. Drawing it as a dashed arrow is more accurate and shows you distinguish the two — and the distinction matters, because an association means lifetime coupling and a dependency does not.

### 4.3 Two styles of class diagram, and when each is right

| Style | What it shows | Use when |
|---|---|---|
| **Entity style** | Fields prominent; methods minimal | The question is about data model, invariants, cardinality — most LLD rounds |
| **Behaviour style** | Methods prominent; fields minimal | The question is about behaviour, interfaces, polymorphism — pattern questions |
| **Both** | Fields on entities, methods on services | The default for a full design |

**In a 45-minute round, draw entities with fields and interfaces with methods.** A `ParkingSpot` box should show `status`, `size`, `spot_id` and the three state methods. A `PricingStrategy` box should show only `calculate`. That contrast communicates which things hold state and which hold behaviour — which is the entire SOLID story, drawn.

### 4.4 Alternative notations you may see

| Notation | Where | Note |
|---|---|---|
| **C4** (context, container, component, code) | Modern architecture docs | Level 4 is a class diagram; C4 is better for HLD. See [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md) |
| **CRC cards** (Class, Responsibility, Collaboration) | Agile design sessions | Excellent for the entity-identification phase; no notation to learn |
| **Mermaid / PlantUML** | Text-to-diagram | Fine if you are drawing on a laptop; useless on a whiteboard |
| **Plain boxes and arrows** | Most actual interviews | **This is what you will do. The notation above just makes it unambiguous** |

**The honest position:** in practice, interviews are boxes and arrows with a few labels, and the labels are doing the work. What the notation buys you is that `◆` versus `◇` and `0..1` versus `1..*` say something *precise* in one character, so you do not have to spend a sentence on it.

---

## 5. Pros / Cons

| Diagram | Pros | Cons | Cost of not drawing it |
|---|---|---|---|
| **Class** | Fixes vocabulary; shows relationships and cardinality; exposes missing entities | Can be drawn too early and redrawn; encourages field-listing; cannot express time | The interviewer cannot audit the design; scope is unmanaged |
| **Sequence** | Makes ordering and concurrency visible; shows the race; reveals which calls can fail | Verbose if drawn fully; encourages drawing the happy path only | The race is never surfaced; ordering bugs stay hidden |
| **State** | Makes illegal transitions explicit; forces the lifecycle question | One class per state looks heavy on a diagram; timeout transitions are awkward to draw | Illegal transitions are only discovered during coding |
| **Activity** | Shows compensating actions on each failure path; handles many branches | Sprawls quickly; low information per unit of space | Failure paths are hand-waved |
| **Entity-style class** | Data and invariants prominent | Hides behaviour | Invariants and cardinality go unstated |
| **Behaviour-style class** | Polymorphism and interfaces prominent | Hides state | Looks stateless when it is not — a concurrency blind spot |

**The overall cost of drawing at all:** 10–15 minutes of a 45-minute round. That is the real price, and it is why the budget table in §1.2 exists. Drawing is worth it because it makes the *rest* of the round faster — the follow-ups are cheap when the entities already exist on the board.

---

## 6. Exceptions, edge cases & failure modes

### 6.1 The fourteen ways diagrams go wrong

| # | Failure | Symptom | Fix |
|---|---|---|---|
| 1 | **Drawing before understanding** | Redrawing everything at minute twenty | Requirements out loud first, then draw |
| 2 | **Illegible** | The interviewer cannot read the box names | Write the names large; abbreviate consistently; talk while you draw |
| 3 | **Too many boxes** | Fifteen classes, no time to discuss any | 6–10. Group the rest verbally |
| 4 | **Every field and getter** | The box is full and says nothing | Key fields only; no getters, no constructors |
| 5 | **No cardinality** | `A ──── B` with no numbers | Every relationship gets `1`, `0..1`, `*` or `1..*` |
| 6 | **Arrows drawn backwards** | The diamond at the wrong end | Diamond at the owner; the test is "delete the parent, does the child survive?" |
| 7 | **Aggregation and composition confused** | A filled diamond where lifetime is independent | Ask the delete test out loud |
| 8 | **No sequence diagram** | The race is never mentioned | Always draw the critical flow |
| 9 | **Sequence diagram shows only the happy path** | Failure and conflict paths invisible | Draw the competing path on the same diagram |
| 10 | **Inheritance where composition belongs** | `Floor ──▷ ParkingLot` | Use the three-question inheritance test — see [OOP principles](02-oop-principles-and-solid.md) |
| 11 | **Interfaces drawn as classes** | A concrete-looking box for `PricingStrategy` | Use `«interface»`, or explain it verbally |
| 12 | **The diagram and the code disagree** | Two different names for one entity | Name it once, then say it the same way every time |
| 13 | **No concurrency annotation** | Nothing marks the boundary | Mark the critical section and the atomic step on the sequence diagram |
| 14 | **Drawing a new diagram for a follow-up** | Time evaporates | Annotate the existing diagram; follow-ups are diffs, not redraws |

### 6.2 The follow-up, drawn as a diff

"How would you support multiple lots?" is a follow-up you should answer by *pointing*, not redrawing.

```
  BEFORE:                            AFTER (one new line, one new box):

  ┌────────────┐                     ┌────────────┐
  │ ParkingLot │                     │  Campus    │
  └─────┬──────┘                     └─────┬──────┘
        │ 1                                │ 1
        │                                  │ ◇ 1..*
        ▼                                  ▼
   ◆ 1..* Floor                     ┌────────────┐       ← aggregation: a lot
                                    │ ParkingLot │          outlives a campus
                                    └─────┬──────┘
                                          │ ◆ 1..*
                                          ▼
                                       Floor

  Everything below ParkingLot is UNCHANGED. The change is one
  aggregate above it, and `reserve()` now routes by lot.
```

**The reason to draw it this way:** it demonstrates that the extension is *additive*. A candidate who says "the design already supports this, here is the one line" is showing the payoff of the composition, and that is exactly what the follow-up is testing. See [problems/01-parking-lot.md](problems/01-parking-lot.md) for the full treatment.

### 6.3 The state diagram's timeout edge case

A transition triggered by *time* rather than by a call is the one state-machine idea candidates miss, and it does not fit the notation cleanly.

```
   ┌────────┐   hold    ┌────────┐  confirm   ┌────────┐
   │  FREE  │──────────►│  HELD  │───────────►│ BOOKED │
   └────────┘           └───┬────┘            └────────┘
        ▲                   │
        │                   │  release()
        │                   │  ─ OR ─
        │                   │  ⏱ TTL expiry  ← a transition with NO caller
        └───────────────────┘

  ⏱ expiry is special:
    - It fires while nothing is happening, so no method call triggers it.
    - It must be evaluated LAZILY on every read, so the answer is never wrong.
    - It must also be swept PROACTIVELY, so expired holds do not accumulate.
  Draw the ⏱ and say both halves out loud. This is the detail that
  distinguishes a seat-booking design that works from one that leaks seats.
```

### 6.4 When a diagram is the wrong tool

| Situation | Why not a diagram | Do instead |
|---|---|---|
| **Machine-coding round with a running IDE** | The code is the artefact | Sketch five entities for 30 seconds, then code |
| **One class, two methods** | A diagram is more work than the code | Write the code |
| **The problem is algorithmic, not structural** | There are no entities, just an invariant | State the complexity and the data structure |
| **The interviewer asks a direct question** | They want the answer, not a picture | Answer it |
| **You are out of time** | An unfinished diagram is worse than a complete sentence | Describe it verbally |
| **The system is genuinely simple** | Boxes add ceremony | Prose, with the entities named |

### 6.5 The illegibility failure mode, specifically

This is the most common and the most damaging, and it has a concrete cause: **candidates draw in silence.** They face the board, write, and then turn around to explain a diagram the interviewer stopped following two minutes ago.

**The fix is behavioural, not technical:**

1. **Talk while you draw.** "I am putting `Floor` under `ParkingLot` with a filled diamond, meaning the floors are owned by the lot. Cardinality is one to many."
2. **Name each box as you write it.** The name is what the rest of the conversation uses.
3. **Write large, and leave space.** Annotating later needs room. Candidates who fill the board have nowhere to add the answer to the follow-up.
4. **Cap the box count out loud.** "Six entities, and I am leaving out the payment classes to save time." This tells the interviewer what you deliberately omitted, so they do not read it as an oversight.
5. **Draw the second user.** In a concurrency-relevant flow, adding the competing lifeline is thirty seconds and signals more than any code you could write.

---

## 7. When to use / When NOT to use

### 7.1 Use a diagram when

| Situation | Diagram |
|---|---|
| Standard LLD round, 45 minutes, discussion-based | Class, then sequence |
| Anything with a lifecycle: bookings, orders, rooms, tickets | Add a state diagram |
| The concurrency question is likely | Sequence diagram with the competing path |
| A follow-up asks for an extension | Annotate the existing class diagram |
| Three or more failure branches with compensations | Activity diagram |
| The interviewer is confused about relationships | Re-draw only the relationships, larger |

### 7.2 Do not diagram when

| Situation | Do instead |
|---|---|
| Machine coding with an IDE | Sketch entities for 30 seconds, then code |
| One or two classes | Write the code |
| The problem is algorithmic | State the invariant and the complexity |
| You have under 10 minutes | Describe verbally, name the entities |
| A verbal answer was asked for | Answer it |
| The design is already understood by both of you | Move to the concurrency and failure discussion |
| You would have to redraw to answer a follow-up | Point at the existing diagram |

### 7.3 The drawing order, as a checklist

1. **Entities** — from the requirements, named in domain language. Do not draw yet.
2. **Class diagram** — boxes, fields, key methods.
3. **Relationships** — arrows, with cardinality, checked with the delete test.
4. **Interfaces** — mark them `«interface»`, and say why each exists.
5. **Sequence diagram** — the one critical flow, including the failure path.
6. **The competing path** — the second user, on the same diagram.
7. **State diagram** — if there is a lifecycle, with illegal transitions marked.
8. **Concurrency annotation** — mark the critical section and the atomic step.
9. **Say what you omitted** — and why.

---

## 8. Numbers, defaults & tuning

| Parameter | Default | Why |
|---|---|---|
| **Boxes in a class diagram** | **6–10** | More cannot be discussed in the time available |
| **Fields per class** | 2–4 | The ones carrying invariants |
| **Methods per class** | 2–4 | The behaviour-carrying ones |
| **Relationships drawn** | 4–8 | The ones you will actually discuss |
| **Participants in a sequence diagram** | 4–6 | More lifelines do not fit legibly on a whiteboard |
| **Messages per sequence diagram** | 6–12 | Beyond that, split into two flows or describe one verbally |
| **States in a state diagram** | 3–6 | Fewer → a boolean; more → split the entity |
| **Time on the class diagram** | 5–7 min | Of a 45-minute round |
| **Time on the sequence diagram** | 4–5 min | Including the competing path |
| **Total drawing time** | ≤ 15 min | A third of the round, at most |
| **Box width on a whiteboard** | ~10–12 cm | Fits the name and four members legibly |
| **Text height** | ≥ 3 cm for class names | Legible from two metres |
| **Space reserved for annotations** | ~25 % of the board | Follow-ups are answered by annotating |
| **Abbreviations used consistently** | Define on first use | `Lot`, `Spot`, `Tkt` — but define them |
| **Redraws permitted** | 0 after the class diagram | Annotate instead; a redraw costs 5+ minutes |

---

## 9. Interview Q&A

**Q1 (easy). Which UML diagrams do you actually use in an LLD round?**
Four: class, sequence, state machine, and occasionally activity. The class diagram fixes the vocabulary and the relationships; the sequence diagram is where ordering and concurrency become visible; the state diagram makes illegal transitions explicit for anything with a lifecycle; the activity diagram only when there are three or more conditional branches with compensating actions. UML has fourteen diagram types and the rest — component, deployment, timing, composite structure — are either HLD-level or irrelevant to this round.

**Q2 (easy). What is the difference between aggregation and composition?**
Both mean "has", and the difference is lifetime. In composition the part cannot outlive the whole — a `Floor` is meaningless without its `ParkingLot`, and the diamond is filled. In aggregation the part can outlive the whole — a `ParkingLot` inside a `Campus` still exists if the campus is deleted, and the diamond is hollow. The test I apply out loud is: delete the parent, does the child still make sense? Yes is aggregation, no is composition.

**Q3 (easy). Why draw a sequence diagram at all if you have the class diagram?**
Because a class diagram cannot express time, and ordering is where the bugs are. In the seat booking flow, the class diagram shows `SeatHold`, `Payment` and `Booking` and says nothing about whether the hold precedes the charge. The sequence diagram makes that ordering explicit — and it is the only diagram on which the two-users-one-seat race is visible, because you can draw both lifelines side by side and see the second one's conditional update return zero rows.

**Q4 (medium). What does cardinality tell you?**
It is where the invariants live. `ParkingSpot 0..1 ──── 0..1 Ticket` says a spot has at most one active ticket — that is the double-booking invariant, drawn. `ParkingSpot 1 ──── * Ticket` would say a spot can have many tickets at once, which is either a bug or a deliberate history model where only one is active, and drawing it forces you to decide which. So marking cardinality is not decoration; it is the three seconds that make the reader check whether you know what "one" means.

**Q5 (medium). How many classes should the diagram have?**
Six to ten. More than that and there is no time to discuss any of them, and the interviewer cannot hold the design in their head while you talk. When the design needs more — and the parking lot does, once you add payment and gates — I group them: I draw the six that carry the design and say out loud that the payment and notification classes are off-diagram because they are adapters over external systems with no interesting structure. Stating the omission is important, because otherwise it reads as an oversight.

**Q6 (medium). Do you write every field and method on the class box?**
No. Two to four fields that carry invariants, and two to four methods that carry behaviour. I deliberately omit constructors and getters, because they add no design information and they cost time I would rather spend on the concurrency discussion. I do include method signatures for interface boxes, because the signature is the contract and the contract is the design. So `PricingStrategy` shows `calculate(ticket, exit_at): Money` and nothing else, while `ParkingSpot` shows its three state methods and its status field.

**Q7 (medium). What is the difference between an association and a dependency?**
An association means a lasting reference — the object holds it and it lives for the object's lifetime. A dependency is transient: the object uses it during one method and holds no reference afterwards. A service that takes a `Clock` as a constructor argument has an association; one that takes it as a method parameter has a dependency. Drawing the distinction matters because an association is a lifetime coupling, and in a design review that is a thing people want to challenge.

**Q8 (medium). How do you draw an interface versus an abstract class?**
An interface gets the `«interface»` stereotype above the name, or a lollipop if you prefer, and it contains only abstract operations and constants. An abstract class gets an italic name and may contain concrete methods — and the presence of an abstract method is marked `{abstract}`. The distinction I care about in the design, and would say out loud, is that an abstract class shares *mechanism* and an interface declares *capability*, so `Vehicle` is an abstract class because `Bike` and `Car` share plate and size handling, while `PricingStrategy` is an interface because there is nothing to share — see [OOP principles](02-oop-principles-and-solid.md).

**Q9 (medium). How do you show concurrency on a diagram?**
Two ways. On the sequence diagram, by drawing the competing path — the second user's lifeline with the same conditional update returning zero rows — which makes the race a picture rather than an assertion. And as an annotation: I mark the critical section with a bracket and write the atomic step next to it, so the reader can see that the boundary of atomicity is one store operation and not the whole flow. Beyond that, I would not try to encode locking semantics into UML notation; it is clearer to say it in a sentence next to the arrow.

**Q10 (hard). *Pushback:* "Isn't UML a waste of time? I'd rather just write the code."**
In a machine-coding round, you are right and I would do the same — five entities sketched in thirty seconds, then code, because the code is the artefact being judged. In a discussion-based LLD round there is no code, so the diagram *is* the artefact, and without it the interviewer has nothing to audit and I have no external memory to keep my own entities consistent over forty-five minutes. The budget is what makes it defensible: about twelve minutes of drawing in a forty-five-minute round, and it makes the remaining thirty faster because the follow-ups become annotations rather than re-derivations.

**Q11 (hard). *Pushback:* "Your class diagram shows `PricingStrategy` with two implementations. Isn't that premature abstraction?"**
There are two justifications and I would lead with the testability one, because it is true immediately: a `FixedPricing` implementation makes the fee assertion exact rather than time-dependent, so the abstraction pays for itself in the test suite before any second production model exists. The second is that the immediate follow-up here is surge or weekend pricing, so the interface is about to earn itself anyway. If neither were true — one implementation, no test double, no second model coming — I would draw the concrete class and note that I would extract the interface when the second implementation landed, which is the honest threshold.

**Q12 (hard). *Pushback:* "You drew the state machine but the seat also has a 'hold expired' state. Where is it?"**
It is not a state, and that is a deliberate choice worth explaining. Expiry is a *transition* from HELD back to FREE, not a fourth state, because nothing behaves differently while a hold is expiring — there is no operation that is legal in EXPIRING but illegal in HELD. If I made it a state, every operation would need to handle it and it would buy nothing. So I draw it as a transition with a clock symbol, and I annotate the two implementation halves: lazily evaluated on every read so the answer is never wrong, and swept proactively by a background job so expired holds do not accumulate. One is correctness, the other is hygiene, and both are needed.

**Q13 (medium). What do you do when the interviewer asks for an extension mid-round?**
Annotate the existing diagram rather than redrawing. A follow-up is a diff. "Now support multiple lots" is one new box above `ParkingLot` with an aggregation arrow, and I would point at the fact that everything below `ParkingLot` is unchanged — which demonstrates that the extension is additive and that the original composition was the right choice. Redrawing costs five minutes and communicates less, because the interviewer cannot see the diff.

**Q14 (medium). How do you draw legibly under pressure?**
The technique is behavioural more than graphical: talk while drawing, name each box as I write it, write the class names large, and leave about a quarter of the board free for annotations. The failure I am avoiding is drawing in silence — facing the board for two minutes and then turning around to explain a diagram the interviewer stopped following. The other rule is capping the box count out loud, so the deliberate omissions are not read as oversights.

**Q15 (medium). Should you draw a diagram in a machine-coding round?**
Barely. Thirty seconds, five entity names on a corner of the whiteboard, and then code — because in that round the code is the artefact and every minute spent drawing is a minute not spent on tests and edge cases. I would say that explicitly: "I will sketch the five entities so we share vocabulary, then start coding, since the code is what you are judging." The exception is if the interviewer asks for a design discussion, in which case it is an LLD round wearing a machine-coding label, and I should treat it as one.

**Q16 (hard). Why draw the second user's lifeline rather than just describing the race?**
Because the picture is checkable and the description is not. If I say "there is a race where two users both read the seat as free", the interviewer has to take my word that I understand the interleaving. If I draw both lifelines and mark where the first conditional update returns one row and the second returns zero, they can see the exact window, and they can also see that my fix closes precisely that window and not a wider one. It is maybe thirty seconds of drawing, and it is the highest-value thirty seconds in the round, because the concurrency question is the one most LLD rounds are built around.

---

## 10. Common mistakes candidates make

1. **Drawing before the requirements are clear**, then redrawing everything at minute twenty.
2. **Drawing in silence.** The interviewer stops following and does not say so.
3. **Fifteen boxes** and no time to discuss any of them.
4. **Every field, getter and constructor** on every box.
5. **No cardinality** on any relationship.
6. **The diamond at the wrong end**, or aggregation drawn as composition.
7. **No sequence diagram**, so the critical flow and the race never surface.
8. **A sequence diagram with only the happy path.**
9. **Inheritance where composition belongs**, drawn as a hollow triangle.
10. **Interfaces drawn as concrete classes**, so the reader cannot tell what is substitutable.
11. **Name drift** — `Spot` in the diagram, `Slot` in the code, `Space` in the explanation.
12. **No concurrency annotation** anywhere on the diagram.
13. **Redrawing for a follow-up** instead of annotating.
14. **Filling the whole board**, leaving nowhere for the follow-up answer.
15. **Not saying what was omitted**, so a deliberate omission reads as an oversight.
16. **Drawing a state diagram as a flowchart** — boxes with no notion of illegal transitions.
17. **Spending more than a third of the round drawing.**
18. **Never drawing at all** in a discussion-based round, leaving the design unauditable.

---

## 11. Cheat-sheet summary

**Notation reference**

| Symbol | Meaning |
|---|---|
| `+` `−` `#` | public / private / protected |
| *Italic name* | abstract class |
| `{abstract}` | abstract method |
| `«interface»` | interface |
| `A ────► B` | unidirectional association (A knows B) |
| `A ──── B` | bidirectional association |
| `A ◇─── B` | aggregation — B outlives A |
| `A ◆─── B` | composition — B dies with A |
| `A ──▷ B` | inheritance (A is-a B) |
| `A - -▷ B` | realization (A implements B) |
| `A - -► B` | dependency (transient use) |
| `1`, `0..1`, `*`, `1..*` | cardinality at the ends |
| `─[key]─` | qualified association (indexed lookup) |

**The four diagrams and their budgets**

| Diagram | Answers | Boxes | Time |
|---|---|---|---|
| Class | What exists, how connected | 6–10 | 5–7 min |
| Sequence | What happens, in what order | 4–6 lifelines | 4–5 min |
| State | Which transitions are legal | 3–6 states | 2–3 min |
| Activity | Which branches, undoing what | — | 0–3 min |

**What to omit**

Constructors · getters/setters · every field · every parameter type · visibility markers on every member · component and deployment diagrams · anything you cannot discuss in the time left.

**The three checks**

1. **Delete the parent — does the child survive?** Yes → aggregation. No → composition.
2. **Delete the parent — is the child still meaningful?** Yes → you do not need inheritance.
3. **Can the interviewer see the race?** If not, draw the second lifeline.

**"If the interviewer says X → do Y"**

| Interviewer says | You do |
|---|---|
| "Walk me through the design" | Class diagram, naming each box as you write it |
| "What happens when a user books?" | Sequence diagram, then mark the atomic step on it |
| "Two users, same seat?" | Draw the second lifeline; mark `rowcount == 0` |
| "Now add multiple lots" | Annotate: one box above `ParkingLot`, aggregation arrow, say nothing below changed |
| "What are the states?" | State diagram, with illegal transitions crossed out and the reason given |
| "Why is that a composition?" | "Delete the lot — the floors are meaningless. That is composition." |
| "Isn't UML a waste of time?" | "In a coding round yes; here the diagram is the artefact, and the budget is twelve minutes" |
| "How many classes is this?" | "Six on the board; payment and notification are adapters I am leaving off to save time" |

---

## 12. Cross-links

- Previous in this pillar: [Concurrency and threading in LLD](04-concurrency-and-threading-in-lld.md) · [Design patterns](03-design-patterns.md) · [OOP principles and SOLID](02-oop-principles-and-solid.md) · [What is LLD and how to answer it](01-what-is-lld-and-how-to-answer.md)
- Next: [The machine-coding interview framework](06-machine-coding-interview-framework.md)
- The diagrams in running designs: [Parking lot](problems/01-parking-lot.md) · [LRU and LFU cache](problems/02-lru-lfu-cache.md) · [Hotel management](problems/03-hotel-management.md) · [BookMyShow seat locking](problems/04-bookmyshow-seat-locking.md) · [Elevator system](problems/05-elevator-system.md) · [Splitwise](problems/06-splitwise.md) · [ATM machine](problems/07-atm-machine.md) · [Library management](problems/08-library-management.md)
- Where the state diagram is the design: [BookMyShow seat booking](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) · [hotel management](problems/03-hotel-management.md)
- Where the class diagram meets the component diagram: [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md)
- The framework that uses these diagrams: [The five-step interview framework](../HLD/01-foundations/04-five-step-interview-framework.md)
- Practice here: [Foundations and capacity interview questions](../Interview-Questions/01-foundations-and-capacity-questions.md) · [Behavioral and project questions](../Interview-Questions/07-behavioral-and-project-questions.md)
