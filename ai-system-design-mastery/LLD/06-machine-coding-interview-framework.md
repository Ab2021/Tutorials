# The Machine-Coding and LLD Interview Framework — Nine Steps, Any Problem

> **Grounding note.** The source transcript frames this pillar at **[25:54]–[27:38]** — *"you come to a class... you draw a class diagram, define the methods, apply the OOP concept. Then you are deciding the classes, methods, and how they interact."* It names the parking lot as the worked example. **It does not give a step-by-step framework, a time budget, a testing strategy, or an extension strategy.** The nine steps below are standard industry practice assembled to counterpart the five-step HLD framework in [HLD/01-foundations/04-five-step-interview-framework.md](../HLD/01-foundations/04-five-step-interview-framework.md); the *shape* of a stepwise method mirrors that file's approach, and the parking-lot worked example is the one the transcript actually names.

---

## TL;DR

The HLD framework is five steps: **clarify → estimate → API → high-level design → deep dive**. The LLD counterpart is nine, because LLD has no capacity estimation to do and two things HLD does not: **code** and **tests**.

**1. Clarify requirements** (5 min) — scope the entity set, and get the defaults you will otherwise invent silently.
**2. Identify core entities** (5 min) — the nouns, with a responsibility each.
**3. Relationships and class diagram** (6 min) — cardinality, composition versus aggregation, the invariants drawn.
**4. Interfaces and abstract types** (4 min) — find the axes of change before writing code, not after.
**5. Write the code** (10–20 min) — working, complete, no placeholders, core path first.
**6. Handle concurrency** (5 min) — the interleaving that breaks it, and the atomic step that closes it.
**7. Extend** (5 min) — "now add X" is the payoff test for step 4, answered by pointing at the diagram.
**8. Test** (5 min) — the golden path, the boundary, and the invariant that must never be violated.
**9. State trade-offs** (3 min) — what you chose, what it buys, what it costs, what you would do differently at 10× the load.

The framework's value is not that it is the only correct order. It is that **under pressure, a fixed order stops you from skipping the two steps candidates always skip: concurrency and tests.** The single biggest LLD failure is a beautiful class diagram with two users booking the same seat and no test that would catch it.

> **60-second interview answer:** "I run nine steps. I clarify the scope — which entities are in and which are out — and get the interviewer to agree on defaults so I do not invent them silently. I name the core entities with a responsibility each, then draw the relationships with cardinality, because that is where the invariants live. Before writing code I identify the axes of change and put interfaces on them, so the 'now add a second pricing model' follow-up is a new class rather than a refactor. Then I write the core path in working code, handling the concurrency story explicitly — for the parking lot that is the allocation race — and I test the golden path, a boundary, and the invariant that must never break. Finally I state the trade-offs and the cost of each choice. The two steps people skip are concurrency and tests, and those are the two the round is scored on."

---

## 1. Definition & mental model

### 1.1 The nine steps, and what each produces

| # | Step | Produces | Time (45-min round) | Skipped by candidates |
|---|---|---|---|---|
| 1 | Clarify requirements | An agreed scope and a defaults list | 5 min | Rarely — everyone asks something |
| 2 | Identify core entities | 5–9 nouns, each with one responsibility | 5 min | Sometimes |
| 3 | Relationships and class diagram | Cardinality, composition, invariants | 6 min | Sometimes |
| 4 | Interfaces and abstract types | The axes of change, isolated | 4 min | **Often** |
| 5 | Write the code | Working core path | 12 min | In a discussion round, **often** |
| 6 | Handle concurrency | The race, the atomic step, the recovery | 5 min | **Very often** |
| 7 | Extend | The follow-up, answered by annotating | 5 min | It is asked, not chosen |
| 8 | Test | Golden path, boundary, invariant | 4 min | **Very often** |
| 9 | State trade-offs | A table of choice / buys / costs | 3 min | Sometimes |

### 1.2 The counterpart to the five-step HLD framework

| HLD step | LLD counterpart | Why the change |
|---|---|---|
| 1. Clarify requirements | **1. Clarify requirements** | Identical — and the most important step in both |
| 2. Capacity estimation | **— none —** | LLD does not size machines. The equivalent is "how many concurrent actors per entity", which belongs in step 6 |
| 3. Define the API | **4. Interfaces and abstract types** | The API becomes the internal contracts: the interfaces between components rather than the endpoint surface |
| 4. High-level design | **2. Entities + 3. Class diagram** | The components become classes, and the connections become relationships with cardinality |
| 5. Deep dive + trade-offs | **5–9. Code, concurrency, extend, test, trade-offs** | LLD's deep dive is code, and it has three extra deliverables: concurrency, tests, extension |

**The mental model:** HLD asks *"what boxes, and how do they talk?"* LLD asks *"what classes, what methods, and what happens when two threads arrive at once?"* The framework difference is that LLD must end in **something that runs** and **something that proves it correct**.

### 1.3 The one rule that governs the whole framework

**Requirements before entities, entities before relationships, relationships before code, code before optimisation, and concurrency before you think you are done.**

Every step's output is the next step's input, and skipping forward is what produces the classic failure: code written for entities that were never agreed on, with no relationship cardinality, so nobody notices that a spot can have two active tickets until the interviewer asks about it.

### 1.4 The honest caveat

**No interviewer is scoring you on following a framework.** They are scoring the design. The framework is scaffolding for *you*, to keep the two skipped steps from being skipped. If you naturally cover concurrency and tests, the framework has done its job and should become invisible. If the interviewer pulls you to a different order — "before you go on, how would you handle X?" — follow them; the framework is not a script to defend.

---

## 2. Why it exists — the problem it solves

### 2.1 What breaks without a framework

| Failure | Cause | Consequence |
|---|---|---|
| **A beautiful design that double-books** | Concurrency skipped | The one thing the round is testing is missing |
| **A refactor on the follow-up** | Interfaces not identified in step 4 | "Add a second pricing model" becomes a rewrite instead of a class |
| **Entities discovered mid-code** | Step 2 skipped | Half the code is written, then three classes are renamed |
| **Scope creep** | Step 1 skipped | 25 classes, none finished, no time for the deep dive |
| **No evidence of correctness** | Step 8 skipped | The interviewer has to take your word that it works |
| **Trade-offs asserted, not reasoned** | Step 9 skipped | "It is more scalable" with no cost stated |
| **Silent assumptions** | Defaults not agreed in step 1 | You and the interviewer are designing different systems |

### 2.2 Why the concurrency step is the scored one

Because it is the hardest thing to retrofit and the easiest thing to forget. In every domain in this pillar — parking spots, cinema seats, hotel rooms, ATM cash, library copies — there is a **finite resource that two actors can grab at the same instant**, and the design is only as good as its answer to that. The framework's job at step 6 is to make sure you *arrive* there with time left.

### 2.3 Why the test step is the credibility step

A candidate who says "and the test I would write is: two threads with a barrier, assert exactly one winner" has demonstrated that they know the concurrency fix is real, not aspirational. It takes four minutes and it converts "I would use a conditional update" from an assertion into a design. See [concurrency §6.6](04-concurrency-and-threading-in-lld.md) for the technique.

---

## 3. Internal mechanics — each step in detail

### Step 1 — Clarify requirements (5 minutes)

**Goal:** agree on what is in scope, and get the interviewer to choose the defaults you would otherwise invent.

**Two categories of question, always split this way:**

| Category | Purpose | Examples for a parking lot |
|---|---|---|
| **Scope** | What exists in this design | "Single lot or multiple? Do I model payment, or is it out of scope? Do I handle EV charging?" |
| **Constraint** | The rule that shapes the design | "One vehicle per spot? Can a spot be reserved in advance, or only on arrival? Is pricing flat or time-based?" |

**The standard clarifying question bank for LLD:**

1. **Which entities are in scope?** (the highest-value question — it sets the size of the answer)
2. **What is the primary operation?** (book, reserve, allocate, dispense, settle)
3. **Is there a lifecycle?** (does anything have states and illegal transitions?)
4. **Is there a finite shared resource?** (this predicts the concurrency step)
5. **Does anything expire?** (holds, leases, sessions — predicts TTL design)
6. **How many actors per resource?** (one user per seat, or many? predicts the contention strategy)
7. **Is money involved?** (forces integer minor units and idempotency)
8. **What is the read/write ratio?** (predicts caching and whether reads may be stale)
9. **What are the extension axes?** (pricing, allocation, notification — predicts step 4's interfaces)
10. **What is explicitly out of scope?** (the question that saves twenty minutes)

**The defaults to assume when the interviewer says "you decide":**

| Ambiguity | Default to state out loud |
|---|---|
| Scope of entities | Core resource + owner + lifecycle + one policy axis. Payment and notification are adapters |
| Multi-tenancy | Single instance first; say how it extends |
| Persistence | In-memory for the design; name the store for the concurrency mechanism |
| Concurrency level | "Assume high contention on the hot resource, so I optimise for that" |
| Money | Integer minor units, one currency |
| Time | Injected `Clock`, so expiry is testable |
| Expiry | A TTL, with lazy check plus a sweeper |
| Pricing | One flat model first, behind an interface |
| Failure handling | Idempotency key; retry with backoff; compensating release |
| Scale target | "Design for 10× today's load without a rewrite" |

**Step 1 output — write it on the board:**

```
  SCOPE:      ParkingLot, Floor, ParkingSpot, Vehicle, Ticket
  OUT:        Payment, gate hardware, notification (adapters, named only)
  PRIMARY OP: reserve(vehicle) -> Ticket   |   release(ticket) -> Money
  LIFECYCLE:  Spot: FREE -> HELD -> OCCUPIED -> FREE
  SHARED RES: the spot          <- concurrency lives here
  EXPIRY:     reservation hold, 15 min
  MONEY:      integer minor units
  EXTEND:     pricing model, allocation policy
```

**Why writing this down matters:** it is the contract. When the follow-up comes, you point at `EXTEND` and say "that is the axis I isolated in step 4".

---

### Step 2 — Identify core entities (5 minutes)

**Goal:** the nouns, each with exactly one responsibility.

**The method:** list every noun in the requirement statement, then apply three filters.

| Filter | Question | Example |
|---|---|---|
| **Is it an entity or an attribute?** | Does it have its own identity and lifecycle? | A `SpotSize` is an enum, not a class. A `Spot` is an entity |
| **Is it a value or a reference?** | Is it immutable and interchangeable? | `Money`, `TimeRange`, `SpotSize` are values — frozen dataclasses |
| **Is it in scope?** | Did we agree to it in step 1? | Payment is out; name it as an adapter and move on |

**The responsibility table — the artefact this step produces:**

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `ParkingLot` | Orchestrate reserve and release across floors | Floors, pricing, allocator | Allocate a specific spot; compute a fee |
| `Floor` | Hold spots and find one | Spots | Price; issue tickets |
| `ParkingSpot` | Own its occupancy state and transitions | status, held_by, expires_at | Decide who may hold it; price itself |
| `Vehicle` | Carry identity and size | plate, size | Anything else |
| `Ticket` | Record one parking session | spot, vehicle, entry, exit | Compute its own fee |
| `PricingStrategy` | Compute a fee | rate table | Know about floors or spots |
| `SpotAllocator` | Choose a spot from candidates | policy | Mutate the spot |
| `SpotState` | Define the legal transitions | — | Store per-spot data |

**The tests for a good entity list:**

1. **Every entity has one reason to change.** `ParkingLot` that also computes fees has two — split it (SRP).
2. **No entity is a bag of getters.** An entity with no behaviour is a data structure; that is sometimes right, but say so.
3. **Every entity is a noun your partner in the conversation already used.** No invented nouns at this stage.
4. **Count is 5–9.** Under 5 means you missed the lifecycle; over 9 means you included adapters.

**The most common miss: the state object.** Candidates list `ParkingSpot` with a `status` string and no notion that the *transitions* are the design. That omission is what step 6 later exposes.

---

### Step 3 — Relationships and the class diagram (6 minutes)

**Goal:** connect the entities, with cardinality, and let the diagram state the invariants.

Draw the class diagram from [UML §3.7](05-uml-and-class-diagrams.md) — 6–10 boxes, filled diamonds for composition, `«interface»` for the policies, and **cardinality on every line**.

**The invariants the cardinality states:**

| Relationship | Cardinality | Invariant it states |
|---|---|---|
| `ParkingLot ◆── Floor` | `1 → 1..*` | Floors are owned; a lot always has at least one |
| `Floor ◆── ParkingSpot` | `1 → 0..*` | A floor may be full; spots die with the floor |
| `ParkingSpot ── Ticket` | `0..1 → 0..1` | **At most one active ticket per spot — the double-booking invariant** |
| `Ticket ── Vehicle` | `* → 1` | A ticket has exactly one vehicle |
| `ParkingLot ── PricingStrategy` | `1 → 1` | One pricing policy at a time, replaceable |
| `ParkingLot ── SpotAllocator` | `1 → 1` | Same |

**The three checks to run out loud:**

1. **Delete the parent — does the child survive?** `Floor` without a `ParkingLot` does not → composition. `ParkingLot` without a `Campus` does → aggregation.
2. **Is the arrow the right direction?** `Ticket → ParkingSpot` (the ticket knows its spot) and not the reverse, unless a spot needs to know its ticket.
3. **Does the multiplicity say what I mean?** `0..1 → 0..1` on the spot/ticket line is the invariant. Say it: "a spot can have at most one active ticket — that is the thing I must never violate."

**What to leave off, and say so:** payment, gates, notification. "These are adapters over external systems with no interesting structure; I am leaving them off to spend the time on the concurrency."

---

### Step 4 — Interfaces and abstract types (4 minutes)

**Goal:** find the axes of change *before* writing code, so the follow-up is a new class rather than a refactor.

**The method:** for each entity, ask what varies independently, then put an interface on that seam.

| Question | Seam found | Interface |
|---|---|---|
| What policy could be replaced? | How a fee is computed | `PricingStrategy` |
| What algorithm could differ? | Which spot gets picked | `SpotAllocator` |
| What will the interviewer ask me to add? | A second pricing model, a second allocation policy | (both of the above) |
| What has alternative representations? | Nothing yet — do not force one | — |
| What is external and swappable? | The store, the clock, the payment gateway | `SpotRepository`, `Clock`, adapters |

**The three tests before creating an interface:**

1. **Is there a second implementation, real or imminent?** Pricing: yes, the follow-up is coming. Gate hardware: no, so no interface.
2. **Is there a test double?** A `FixedPricing` makes the fee assertion exact — that alone pays for the abstraction.
3. **Would the concrete class be simpler?** If both answers are no, do not create the interface, and say that you considered it.

**The inheritance question, applied here:**

| Candidate | Test | Verdict |
|---|---|---|
| `Bike`, `Car`, `Truck` extend `Vehicle` | Is-a in every context; honour every postcondition; vary for the same reason (size) | **Inheritance** |
| `Floor` extends `ParkingLot` | A floor is not a lot; a lot's methods are meaningless on a floor | **Composition, not inheritance** |
| `HourlyPricing` extends `PricingStrategy` | Different mechanism per variant; nothing shared | **Realization (interface)** |
| `ParkingLot` extends `BaseRepository` | Different reason to change | **Composition** |

This step is four minutes and it is the one candidates skip. **Skipping it is why the follow-up becomes a refactor**, and the follow-up always comes. See [OOP principles](02-oop-principles-and-solid.md) and [design patterns §4.3.2](03-design-patterns.md).

---

### Step 5 — Write the code (10–20 minutes)

**Goal:** the core path, working, complete, no placeholders.

**The rules:**

| Rule | Why |
|---|---|
| **State the language out loud** | "I will write Python; the design is language-independent" — the interviewer may prefer Java or Go, and it is their call |
| **Core path first** | `reserve` and `release` before edge cases. Get something that runs |
| **No placeholders** | `...` or `pass  # TODO` reads as "I did not finish". Write a real implementation, or say explicitly that a class is out of scope |
| **Signatures first, then bodies** | Write all the method signatures, then fill them in — you catch design errors before writing 80 lines |
| **Enums over strings** | `SpotSize.SMALL` not `"small"`; a typo in a string is a runtime bug |
| **Frozen dataclasses for values** | `Money` as `amount_minor: int` — never a float |
| **Inject `Clock`, time and the store** | Testability, and it is what makes step 8 possible |
| **Comment the *why*, not the *what*`** | "CAS: the WHERE clause is the decision" is worth a line; "increment i" is not |

```python
# The core path, complete and runnable. Note what each line buys:
#   - enum over string           → a typo is impossible
#   - frozen Money               → thread-safe by construction, no float
#   - injected Clock             → expiry is testable without sleep()
#   - injected strategy/allocator→ step 7's follow-up is a new class
#   - status transition inside try_hold → the invariant has one home

class SpotStatus(Enum):
    FREE = "FREE"
    HELD = "HELD"
    OCCUPIED = "OCCUPIED"


@dataclass(frozen=True)
class Money:
    amount_minor: int
    currency: str = "INR"


class ParkingLot:
    def __init__(self, lot_id: str, floors: list[Floor],
                 pricing: PricingStrategy, allocator: SpotAllocator,
                 clock: Clock, hold_ttl: int = 900) -> None:
        self._lot_id = lot_id
        self._floors = floors
        self._pricing = pricing
        self._allocator = allocator
        self._clock = clock
        self._hold_ttl = hold_ttl

    def reserve(self, vehicle: Vehicle, idempotency_key: str) -> Ticket:
        spot = self._allocator.choose(self._floors, vehicle.size)
        if spot is None:
            raise LotFullError(f"no {vehicle.size.name} spot available")

        if not spot.try_hold(vehicle.plate, self._clock.now(), self._hold_ttl,
                             idempotency_key):
            # Someone won the race between choose() and try_hold(). choose() was
            # only a HINT; try_hold() is the atomic decision. Retry once with a
            # re-chosen spot before giving up.
            spot = self._allocator.choose(self._floors, vehicle.size)
            if spot is None or not spot.try_hold(vehicle.plate, self._clock.now(),
                                                 self._hold_ttl, idempotency_key):
                raise SpotTakenError("spot taken concurrently; please retry")

        return Ticket.issue(ticket_id=uuid.uuid4().hex, spot=spot, vehicle=vehicle,
                            entry_at=self._clock.now(), hold_ttl=self._hold_ttl)

    def release(self, ticket: Ticket) -> Money:
        fee = self._pricing.calculate(ticket, exit_at=self._clock.now())
        ticket.spot.release()
        ticket.close(exit_at=self._clock.now(), fee=fee)
        return fee
```

**What to say while writing it:** narrate the *decisions*, not the syntax. "I am injecting the clock so the expiry test does not need a sleep." The interviewer cannot read your mind, and the narration is where the senior signal lives.

**What to do when you run out of time:** say so and prioritise. "I have eight minutes left; I will finish `reserve` including the race, and describe `release` rather than write it." Stating the plan is far better than running out silently.

---

### Step 6 — Handle concurrency (5 minutes)

**Goal:** name the interleaving, close the window, and handle what happens after a crash.

**The procedure:**

1. **Name the shared mutable resource.** "The spot's status."
2. **Write the interleaving.** Two actors, one timeline, the exact failure.
3. **Identify the atomic step that closes the window.** One conditional update, or one lock around the state transition.
4. **State the cost.** Throughput on the contended resource.
5. **Handle the second-order problems.** Deadlock (lock order), retry (idempotency key), crash (TTL plus lazy expiry plus sweeper).

```
  THE INTERLEAVING (draw it if there is a board)
  ── A: choose spot 7 (FREE) ─── try_hold: read FREE ─── write HELD ── ✓
  ── B: choose spot 7 (FREE) ─────── try_hold: read FREE ─── write HELD ── ✓   ✗ BOTH

  THE FIX — the whole decision, in the WHERE clause
  UPDATE parking_spots
     SET status = 'HELD', held_by = %s, hold_expires_at = ...
   WHERE spot_id = %s AND status = 'FREE'
  rowcount == 1  → I won.   rowcount == 0 → someone else did. No lock held.

  THE RECOVERY
  - Idempotency key on the unique index → a retry cannot double-hold.
  - TTL on the hold → a crashed holder cannot strand the spot.
  - Lazy expiry on read (correctness) + sweeper (hygiene).
```

**The five questions the interviewer will ask, in order:** What is the race? What closes it? What does that cost? What if the request is retried? What if the holder crashes? **Have all five answers ready.** The full treatment is in [concurrency in LLD](04-concurrency-and-threading-in-lld.md) and, applied end to end, [problems/04-bookmyshow-seat-locking.md](problems/04-bookmyshow-seat-locking.md).

**The distinction to state explicitly:** `choose()` is a hint; `try_hold()` is the decision. Candidates who treat allocation as authoritative are the ones who double-book, because between choosing and holding, another actor can act.

---

### Step 7 — Extend ("now add X") (5 minutes)

**Goal:** answer the follow-up by *pointing*, and demonstrate that the design anticipated it.

**The standard follow-ups, by domain:**

| Domain | The follow-up | If step 4 was done, the answer is |
|---|---|---|
| Parking lot | "A second pricing model" | A new `PricingStrategy`; `ParkingLot` unchanged |
| Parking lot | "Multiple lots" | One `Campus` above `ParkingLot` with aggregation; nothing below changes |
| BookMyShow | "Seats together, not scattered" | `allocate(n)` on the allocator interface; the caller is unchanged |
| Hotel | "Seasonal pricing" | A new `PricingStrategy`, or a rate table parameterising the existing one |
| Elevator | "A different scheduling policy" | A new `SchedulingStrategy` |
| Splitwise | "A different split rule" | A new `SplitStrategy` |
| ATM | "A second denomination algorithm" | A new `DispenseStrategy` |
| Library | "A different fine policy" | A new `FineStrategy` |

**The three-part answer to any extension:**

1. **Point at the seam.** "That is the axis I isolated in step 4 — the `PricingStrategy` interface."
2. **Name the new class and its one method.** "`SurgePricing.calculate(ticket, exit_at)` — it reads demand and multiplies."
3. **Say what is unchanged.** "`ParkingLot.reserve` does not change at all. That is the OCP payoff — one new file, zero edits."

**If the extension does *not* fit the seams you built, say so honestly.** "That does not fit; it changes the entity model. I would add a `Reservation` entity between `Vehicle` and `Spot` and re-derive the cardinality." A candidate who admits the extension needs a change is more credible than one who claims everything is already handled.

**Annotate, never redraw.** The follow-up is a diff, and pointing at an unchanged part of the diagram is the whole demonstration.

---

### Step 8 — Test (5 minutes)

**Goal:** three tests, named, that between them make the design credible.

| Test | What it proves | Example |
|---|---|---|
| **Golden path** | The core flow works | Reserve a car, assert a ticket with a FREE spot now HELD |
| **Boundary** | The edges behave | The last spot; a lot with zero spots; a release at exactly the TTL |
| **Invariant** | The thing that must never break | Two threads, one spot, exactly one winner |

```python
def test_golden_path_reserve_and_release() -> None:
    clock = FakeClock()
    lot = build_lot(clock=clock, cars=1)
    ticket = lot.reserve(Car("KA-01-1234"), idempotency_key="k1")
    assert ticket.spot.status is SpotStatus.HELD
    assert lot.available(SpotSize.MEDIUM) == 0

    clock.advance(3600)
    fee = lot.release(ticket)
    assert fee == Money(3600_00)                      # 1 hour at the rate
    assert lot.available(SpotSize.MEDIUM) == 1


def test_boundary_hold_expires_exactly_at_ttl() -> None:
    """No sleep() anywhere: the Clock makes this exact and instant."""
    clock = FakeClock()
    lot = build_lot(clock=clock, cars=1)
    lot.reserve(Car("KA-01-1234"), idempotency_key="k1")

    clock.advance(899.999)
    assert lot.available(SpotSize.MEDIUM) == 0        # still held

    clock.advance(0.001)
    assert lot.available(SpotSize.MEDIUM) == 1        # lazy expiry, exactly at TTL


def test_invariant_two_threads_never_both_hold_one_spot() -> None:
    """The test that proves the concurrency fix is real, not aspirational."""
    clock = RealClock()
    lot = build_lot(clock=clock, cars=1)
    winners: list[str] = []
    barrier = threading.Barrier(2)

    def attempt(plate: str) -> None:
        barrier.wait()                                 # maximise the overlap
        try:
            lot.reserve(Car(plate), idempotency_key=plate)
            winners.append(plate)
        except SpotTakenError:
            pass

    threads = [threading.Thread(target=attempt, args=(p,))
               for p in ("KA-01-1111", "KA-01-2222")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, f"double booking: {winners}"
```

**The four properties to mention:**

1. **A `Clock`, so there is no `sleep`.** Expiry tests are exact and instant.
2. **The invariant is asserted where it can break**, not three steps later.
3. **The stress test is a backstop, not the evidence** — it can pass by luck, so the barrier maximises overlap but the real proof is the atomic step.
4. **A test double makes the abstraction pay for itself.** `FixedPricing(100)` is why `PricingStrategy` exists even before the second model.

---

### Step 9 — State trade-offs (3 minutes)

**Goal:** close with a table, and name what you would do differently at 10× load.

| Choice | Buys | Costs |
|---|---|---|
| `PricingStrategy` interface | New model = one new class, no edits | An interface and a wiring point |
| CAS instead of a row lock | No blocking, no deadlock, scales to any contention | One statement per resource; manual partial-failure handling |
| In-memory store | Simple, fast, testable | Not shared across pods — the design needs a store-level CAS in production |
| Allocation as a hint | Simple, no lock held across allocation | A retry path when the choice is stale |
| TTL on holds | Crashed holders cannot strand a spot | A slow user loses a seat mid-checkout |
| Composite `Floor`/`Spot` | Multi-lot extension is free | A deeper tree to traverse |
| `Clock` injection | Deterministic, instant expiry tests | Every time read must be injected |

**The close, in three sentences:** "The design is X. The thing I would revisit at 10× load is Y, because Z. The thing I deliberately left out is payment, because it is an adapter over an external system with no interesting structure."

---

## 4. Variants & taxonomy

### 4.1 Adjusted frameworks for different question shapes

| Shape | Emphasis | Steps to compress | Steps to expand |
|---|---|---|---|
| **Classic OOP design** (parking lot, elevator) | Entities and relationships | 5 (less code) | 2, 3, 6 |
| **Data structure implementation** (LRU/LFU) | Correctness and complexity | 2, 3, 9 | 5, 8 |
| **API / interface design** | Contracts and errors | 3, 6 | 1, 4, 9 |
| **State machine design** (ATM, vending) | The transition table | 3, 7 | 2, 5, 6 |
| **Concurrency design** (seat locking) | The atomic step | 2, 3 | 6, 8 |
| **Machine coding** (running code) | It works, and it is tested | 1, 2, 3 (30 seconds each) | 5, 8 |

### 4.2 Time-boxing variants

| Budget | Format | Allocation |
|---|---|---|
| **45 min** | Discussion LLD | 5 / 5 / 6 / 4 / 12 / 5 / 5 / 4 / 3 — the table in §1.1 |
| **60 min** | Discussion LLD, deeper | 6 / 6 / 8 / 5 / 15 / 7 / 6 / 5 / 4 |
| **90 min** | Machine coding | 5 / 5 / 5 / 3 / 45 / 8 / — / 15 / 4 |
| **120 min** | Machine coding with tests | 6 / 6 / 6 / 4 / 60 / 10 / — / 20 / 6 |
| **20 min** | Rapid LLD screen | 3 / 3 / 4 / 3 / 4 / 2 / 1 / — / — |
| **15 min** | Verbal only | 2 / 3 / 4 / 2 / — / 3 / 1 / — / — |

### 4.3 The machine-coding variant, specifically

Machine coding changes the framework's centre of gravity. Diagrams shrink to thirty seconds; code and tests expand dramatically.

| Step | Discussion round | Machine-coding round |
|---|---|---|
| 1. Clarify | 5 min | 3 min — get scope, then go |
| 2. Entities | 5 min, on the board | 2 min, listed in a comment |
| 3. Class diagram | 6 min, drawn | 1 min, sketched in a corner |
| 4. Interfaces | 4 min, discussed | 2 min — decide, then write them |
| 5. Code | 12 min, core path | 40–60 min — this is the round |
| 6. Concurrency | 5 min, designed | 8 min, implemented and tested |
| 7. Extend | 5 min, verbal | Only if asked |
| 8. Test | 4 min, three named tests | 15–20 min, actually written and run |
| 9. Trade-offs | 3 min | 4 min |

**The machine-coding rule that differs most:** **write the test before the feature where the feature is the concurrency fix.** A passing invariant test is the evidence; everything else is assertion.

---

## 5. Pros / Cons

### 5.1 Pros of the framework

- **Prevents the two skipped steps.** Concurrency and tests are the ones that vanish under pressure, and a fixed order protects them.
- **Makes the follow-up cheap.** Step 4 turns "add a second pricing model" from a refactor into a new class.
- **Produces an artefact the interviewer can audit.** Entities, cardinality and a diagram are checkable; prose is not.
- **Manages time.** Nine steps with budgets is a schedule, and running out of time is a plan failure, not a knowledge failure.
- **Survives nerves.** A fixed order is something to do when you cannot think of what to do next.

### 5.2 Cons, and the trade-offs

| Cost | When it bites |
|---|---|
| **Rigid if applied mechanically** | The interviewer has a different order in mind; following the script over them is a mistake |
| **Diagrams can eat the round** | 15 minutes of drawing in a 45-minute round is the ceiling, and candidates exceed it |
| **Step 4 can produce speculative interfaces** | Two interfaces where one was needed; the tests in step 4 guard against it |
| **A framework answer can sound rehearsed** | The narration must be about *this* design, not the framework's names |
| **Machine coding does not fit the discussion timings** | Using the 45-minute shape in a 90-minute coding round wastes the only resource that matters |
| **It can crowd out the actual insight** | The framework is scaffolding; the design is the deliverable |

**Pros and cons of each step**

| Step | Strongest pro | Strongest con |
|---|---|---|
| 1. Clarify | Sets scope; surfaces the hidden constraint | Feels like stalling if it runs past 5 minutes |
| 2. Entities | Fixes vocabulary; smallest change cost | Over-listing produces 15 entities |
| 3. Diagram | Draws the invariants; auditable | Can sprawl if the box count is not capped |
| 4. Interfaces | Makes the follow-up free | Speculative abstraction if the tests are skipped |
| 5. Code | The only thing that proves it runs | Time sink; candidates write boilerplate |
| 6. Concurrency | The scored step | Most often skipped, and hardest to retrofit |
| 7. Extend | Demonstrates the design's payoff | Can be answered dishonestly ("it already handles that") |
| 8. Test | Converts claims into evidence | Skipped; and `sleep`-based tests prove nothing |
| 9. Trade-offs | The senior signal: costs, not just benefits | Boils down to "it is more scalable" if unprepared |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 The eighteen LLD failure modes, by step

| Step | Failure | Fix |
|---|---|---|
| 1 | No clarifying questions at all | Ask three: scope, lifecycle, shared resource |
| 1 | Twenty minutes of questions | Cap it at 5; state defaults for the rest |
| 1 | Silent assumptions | Write the defaults on the board |
| 2 | Entities as a bag of getters | Give each one responsibility and one reason to change |
| 2 | Attributes promoted to entities | `SpotSize` is an enum; `Money` is a value |
| 2 | Adapters in the core list | Name them separately as out of scope |
| 3 | No cardinality | Mark every line; that is where the invariants are |
| 3 | The diamond at the wrong end | Apply the delete test out loud |
| 3 | 15 boxes | Cap at 10; group the rest |
| 4 | No interfaces → the follow-up is a refactor | Find the axes in step 4 |
| 4 | Speculative interfaces | Two implementations or a test double, or skip it |
| 4 | Inheritance where composition belongs | The three-question test |
| 5 | Placeholders and `TODO`s | Write it, or say it is out of scope |
| 5 | Strings where enums belong | A typo in a string is a runtime bug |
| 5 | Floats for money | Integer minor units, always |
| 6 | No concurrency section | The scored step |
| 6 | Allocation treated as authoritative | `choose()` is a hint; `try_hold()` is the decision |
| 6 | No recovery | TTL, lazy expiry, sweeper, idempotency key |
| 7 | A redraw instead of an annotation | The follow-up is a diff |
| 8 | `sleep`-based tests | A `Clock` |
| 8 | No invariant test | Two threads, one resource, exactly one winner |
| 9 | Benefits with no costs | Every choice has a cost column |

### 6.2 The genuine exception: when to break the order

| Situation | Break the order how |
|---|---|
| The interviewer names the hard part up front ("the interesting bit is the locking") | Jump to step 6, then backfill entities |
| The problem is a data structure (LRU) | Compress 2–4 to two minutes; spend it on correctness and complexity |
| The problem is an API design | Steps 4 and 9 dominate; 2 and 3 are trivial |
| Machine coding | Steps 1–4 get 7 minutes total |
| The interviewer is clearly time-pressured | Say what you would do, in order, in two sentences, then do the most interesting part |
| They ask a direct question mid-framework | Answer it, then resume |
| You realise the design is wrong at step 6 | Say so, annotate the diagram, and re-derive only the affected part |

### 6.3 Running out of time: a triage plan

| Time left | Do this |
|---|---|
| 15 min | Finish the code for the core path; state concurrency verbally; name the tests |
| 10 min | Finish `reserve` including the race; describe `release`; name the three tests |
| 5 min | Stop coding. State the concurrency fix and the three tests verbally. Close with trade-offs |
| 2 min | One sentence: "the design is X, the concurrency fix is Y, the test is Z, the cost is W" |

**Stating the plan beats running out silently.** "Eight minutes left — I will finish the race and name the tests rather than write them" is a senior move.

### 6.4 The failure that looks like success

**A clean, complete, well-diagrammed design with no concurrency section and no tests.** It reads as competent and scores as incomplete, because those are the two things the round exists to measure. If you take one thing from this file: **when you think you are finished, you are at step 5 of 9.**

---

## 7. When to use / When NOT to use

### 7.1 Apply the full nine steps when

| Situation | Why |
|---|---|
| A standard 45–60 minute discussion LLD round | The timings are built for it |
| The problem has entities, a lifecycle and a shared resource | Every step has real work to do |
| You are nervous or stuck | The order is a checklist to fall back on |
| The interviewer says "design X" with no further guidance | The framework is the structure they expect |
| You have practised the problem | The framework keeps you from skipping to the answer |

### 7.2 Do not apply it mechanically when

| Situation | Do instead |
|---|---|
| Machine coding with an IDE | Compress 1–4 into 7 minutes; code is the deliverable |
| The interviewer names the interesting part | Go there first |
| A 15-minute screen | Clarify, entities, concurrency, trade-offs |
| The problem is algorithmic, not structural | State the invariant, the data structure, the complexity |
| They ask one specific question | Answer it |
| You have already solved this exact problem | Lead with the answer, then justify |
| The interviewer wants to talk about one trade-off | Have the conversation; do not narrate a framework |

### 7.3 The three-question pre-flight

Before starting any LLD answer, answer these in your head:

1. **Is there a finite shared resource two actors can grab at once?** (→ step 6 is the scored step)
2. **What will they ask me to add afterwards?** (→ step 4 must isolate that axis)
3. **What am I explicitly leaving out?** (→ step 1 and step 9 must both say it)

---

## 8. Numbers, defaults & tuning

| Parameter | Default | Why |
|---|---|---|
| **Clarifying questions** | 5–8, capped at 5 minutes | Fewer means silent assumptions; more is stalling |
| **Core entities** | 5–9 | Under 5 missed the lifecycle; over 9 included adapters |
| **Class diagram boxes** | 6–10 | More cannot be discussed in the time |
| **Methods per class** | 2–4 | The behaviour-carrying ones |
| **Interfaces introduced** | 1–3 | One per genuine axis of change |
| **Implementations before an interface** | 2 (or 1 + a test double) | Otherwise it is speculation |
| **Time on diagrams** | ≤ 15 min of 45 | A third of the round |
| **Time on concurrency** | 5 min minimum | The scored step |
| **Tests named** | 3 (golden, boundary, invariant) | Below three, the design is unproven |
| **Time-box, 45-min discussion** | 5/5/6/4/12/5/5/4/3 | The table in §1.1 |
| **Time-box, 90-min machine coding** | 5/5/5/3/45/8/—/15/4 | Code dominates |
| **Code written in a discussion round** | The core path only | Edge cases described, not written |
| **Hold TTL (parking)** | 15 min | Longer than a cinema hold; the car is already there |
| **Hold TTL (seat)** | 8–10 min | Bounded by the payment flow |
| **Retry attempts on a conflict** | 3 | Beyond that, the design is wrong |
| **Idempotency key retention** | ≥ the hold TTL; 24 h for payments | A retry outside the window must not double-charge |
| **Stated "at 10×" answer** | Always give one | The scaling question is nearly always asked |

---

## 9. Interview Q&A

**Q1 (easy). What is the difference between an LLD round and a machine-coding round?**
The deliverable. An LLD round is a design discussion: you produce entities, relationships, interfaces and a concurrency story, and you may write the core path but nobody runs it. A machine-coding round produces working, tested code — the design is judged through the code, and the diagram shrinks to a thirty-second sketch. Both use the same nine steps, with radically different time allocations: in a 45-minute LLD round, code gets 12 minutes; in a 90-minute machine-coding round, it gets 45 and tests get 15 more.

**Q2 (easy). How long should you spend clarifying requirements?**
About five minutes, and no more than eight. The two things you must get out of it are scope — which entities are in and which are out — and the constraint that shapes the design, usually whether there is a finite shared resource and whether anything expires. Everything else you can state as an assumption, out loud, and move on. Twenty minutes of questions is a failure mode of its own: it looks like stalling and it eats the deep dive.

**Q3 (easy). Which step do candidates skip most often?**
Concurrency and testing, and those are the two the round is actually scored on. A candidate will produce a clean class diagram with correct cardinality and never mention that two users can hold the same spot, because nothing in the entity model forces the question. That is why the framework puts concurrency at step 6 and tests at step 8 with explicit time budgets — under pressure, an unordered approach drops both.

**Q4 (medium). Why identify interfaces before writing code?**
Because the follow-up is coming, and the follow-up is always on the axis of change: "now add a second pricing model", "now support a different allocation policy", "now add seasonal rates". If the interface exists, the answer is one new class and zero edits to `ParkingLot` — the Open/Closed payoff, visible on the whiteboard. If it does not, the answer is a refactor, and the interviewer sees that you did not anticipate the most obvious extension of your own design. It is four minutes in step 4 to save ten in step 7.

**Q5 (medium). Walk me through designing a parking lot.**
Scope: lot, floors, spots, vehicles, tickets — payment and gate hardware named as out of scope. Entities with one responsibility each: the lot orchestrates, the floor holds spots, the spot owns its own state, the vehicle carries identity and size, the ticket records a session, and pricing and allocation are policies. Relationships: the lot is composed of floors, floors of spots, and a spot has at most one active ticket — that last cardinality is the double-booking invariant, drawn. Interfaces on the two axes of change, pricing and allocation, plus a `Clock` for testable expiry. Then the code: `reserve` chooses a spot as a hint and holds it atomically. Then the race — two vehicles, one spot, both read FREE — closed by a conditional update whose `WHERE status='FREE'` is the decision, with a TTL and an idempotency key for recovery. Then the tests: reserve and release, expiry exactly at the TTL, and the invariant that two threads never both win. Then the trade-offs, and what changes at 10×.

**Q6 (medium). What does "core path first" mean in practice?**
Write the two or three operations that carry the design — for the parking lot, `reserve` and `release` — end to end and working, before touching edge cases. It means the interviewer sees a complete design early, and every later addition is an improvement rather than a hope. The alternative, which candidates fall into, is writing helper classes and enums and value objects for fifteen minutes and never reaching the operation that is being tested. Signatures first, then the core bodies, then edge cases, then optimisations.

**Q7 (medium). How do you decide whether to create an interface?**
Two tests, and I apply them in this order. First, is there a second implementation, real or imminent? For pricing, the follow-up is coming, so yes. For gate hardware, no, so it stays concrete. Second, is there a test double that makes the abstraction pay for itself? A `FixedPricing` makes the fee assertion exact instead of time-dependent, which is a benefit today even before the second model exists. If both are no — one implementation, no double, no second coming — I use the concrete class and say that I considered the interface and rejected it. That sentence scores better than creating it speculatively.

**Q8 (medium). *Pushback:* "Your allocation strategy chooses a spot, then you try to hold it. Isn't that a race?"**
Yes, and the design says so explicitly: `choose()` is a *hint*, and `try_hold()` is the decision. The hint exists because scanning for a free spot is cheap and it makes the common case one attempt; the atomic step is the conditional update inside `try_hold`, and its `rowcount` decides the winner. On the rare occasion the hint is stale, `try_hold` fails and the caller re-chooses and retries once before reporting "spot taken, please retry". If I removed the hint entirely I would need a single statement that both finds and claims — which is possible in SQL with a subquery in the `UPDATE`, and I would mention it as the alternative, at the cost of readability.

**Q9 (medium). *Pushback:* "Why are you writing code in a design round? Isn't the diagram enough?"**
Because for the parts that carry the design, code is unambiguous in a way a diagram is not. The cardinality `0..1 ──── 0..1` states the invariant, and the conditional `UPDATE ... WHERE status = 'FREE'` states how it is enforced — the diagram cannot show the `WHERE` clause, and the `WHERE` clause is the concurrency design. I keep it proportionate: in a 45-minute discussion round, about twelve minutes of code on the core path only, with edge cases described rather than implemented. In a machine-coding round the ratio inverts and code is most of the round.

**Q10 (medium). How do you handle the "now add X" follow-up?**
By pointing, not redrawing. Most follow-ups land on the axes I isolated in step 4, so the answer is three sentences: point at the interface, name the new class and its one method, and say what is unchanged. "That is the `PricingStrategy` seam — a new `SurgePricing.calculate` — and `ParkingLot.reserve` does not change at all, which is the Open/Closed payoff." If the extension genuinely does not fit — "now support reserving a spot in advance with a start time" changes the entity model — I say so and describe the change: a `Reservation` entity between vehicle and spot, and the cardinality re-derived. Claiming everything is already handled is worse than admitting the extension needs work.

**Q11 (hard). *Pushback:* "You said the design handles concurrency, but I see no locks anywhere."**
There are no locks by design, and that is the point. The concurrency control is the `WHERE status = 'FREE'` clause in the conditional update: the database's row-level atomicity decides the winner in a single statement, the loser gets `rowcount == 0` immediately, and nobody ever blocks. A row lock would also be correct — `SELECT ... FOR UPDATE` — but it holds a transaction open, it can deadlock if two bookings lock rows in different orders, and on a hot resource it queues and produces lock-wait timeouts that surface as 500s. So I chose the design that never blocks. The costs are real and I would state them: it is one statement per resource, so a multi-spot reservation needs a transaction or compensating releases, and I lose the ability to explain *why* it failed without a second read.

**Q12 (hard). *Pushback:* "Your tests use a fake clock. Doesn't that mean you never tested real expiry?"**
It means I tested the *logic* of expiry deterministically, which is the part where bugs live: the boundary at exactly the TTL, the lazy check on read, the interaction with the sweeper. A `sleep`-based test would exercise the same code path, take ten minutes of wall-clock time, and fail intermittently on a loaded CI machine — which is strictly worse evidence, not better. What the fake clock does *not* cover is the real clock's behaviour, so I would add one small integration test against the real store with a deliberately tiny TTL — a second or two — that proves the store's own expiry mechanism fires. The fake clock for logic, one real test for the integration, and no `sleep` in the suite.

**Q13 (hard). *Pushback:* "You have `PricingStrategy` as an interface with one implementation. Isn't that speculative?"**
There are two justifications and both apply here. The testability one is true today: `FixedPricing(minor=100)` makes the fee assertion exact rather than dependent on a rate table, so the abstraction pays for itself in the test suite before any second production model exists. The extension one is imminent: the standard follow-up in this problem is a second pricing model, so the interface is about to have two implementations. If neither held — one implementation, no test double, no second coming — I would use the concrete class and say I had considered and rejected the interface. That is the honest threshold.

**Q14 (hard). What would you do differently if this were a 90-minute machine-coding round?**
I would compress steps 1 to 4 into about seven minutes: two minutes of clarifying questions, a two-minute entity list in a comment at the top of the file, a one-minute sketch of five boxes in the corner of the board, and two minutes to decide the interfaces. Then forty-five minutes of code and fifteen to twenty minutes of tests, because in that round the code and the passing tests *are* the deliverable — the interviewer reads the design through them. The one thing I would not compress is the concurrency test: I would write the invariant test early, as a failing test, and then make it pass, because that is the clearest possible evidence that the fix is real.

**Q15 (medium). How do you close an LLD round?**
With a trade-off table and three sentences. The table is choice, what it buys, what it costs — and stating the costs is what makes it a design rather than a sales pitch. Then: "the design is X; the thing I would revisit at 10× load is Y, because Z; the thing I deliberately left out is payment, because it is an adapter with no interesting structure." The explicit omission matters, because otherwise it reads as something you forgot rather than something you scoped out.

---

## 10. Common mistakes candidates make

1. **Skipping clarifying questions** and inventing requirements silently.
2. **Twenty minutes of questions**, eating the deep dive.
3. **Starting to code before the entities are agreed.**
4. **Over-listing entities** — fifteen classes, none discussed.
5. **No cardinality** on the relationships, hiding the invariant.
6. **Skipping step 4**, so the follow-up becomes a refactor.
7. **Speculative interfaces** with one implementation and no test double.
8. **Placeholders and `TODO`s** in the code.
9. **Floats for money.**
10. **Strings where enums belong.**
11. **Treatment of allocation as authoritative** rather than as a hint.
12. **No concurrency section at all.**
13. **No recovery story** — no TTL, no idempotency key, no sweeper.
14. **Redrawing the diagram** for a follow-up instead of annotating.
15. **No tests**, or tests that `sleep`.
16. **Benefits with no costs** in the trade-off table.
17. **Not stating what was left out of scope.**
18. **Thinking you are finished at step 5 of 9.**

---

## 11. Cheat-sheet summary

**The nine steps**

| # | Step | Output | 45-min budget |
|---|---|---|---|
| 1 | Clarify requirements | Scope + defaults, on the board | 5 min |
| 2 | Identify core entities | 5–9 nouns, one responsibility each | 5 min |
| 3 | Relationships + class diagram | Cardinality, composition, invariants | 6 min |
| 4 | Interfaces / abstract types | The axes of change, isolated | 4 min |
| 5 | Write the code | Working core path | 12 min |
| 6 | Handle concurrency | Race, atomic step, recovery | 5 min |
| 7 | Extend | The follow-up, answered by pointing | 5 min |
| 8 | Test | Golden, boundary, invariant | 4 min |
| 9 | State trade-offs | Choice / buys / costs | 3 min |

**The three questions to ask first**

1. Which entities are in scope, and which are explicitly out?
2. Is there a finite shared resource two actors can grab at once?
3. What will you ask me to add afterwards?

**The two steps nobody may skip**

- **Concurrency** — name the interleaving, close the window, handle the crash.
- **Tests** — golden path, boundary, and the invariant that must never break. With a `Clock`, never a `sleep`.

**The standard follow-ups, and the seam that answers them**

| Follow-up | Seam | The answer |
|---|---|---|
| Second pricing model | `PricingStrategy` | One new class; the lot unchanged |
| Second allocation policy | `SpotAllocator` | One new class |
| Multiple lots | Composition above `ParkingLot` | One new aggregate; nothing below changes |
| Contiguous seats | `allocate(n)` on the allocator | The signature widens; the caller is unchanged |
| Seasonal rates | A rate table parameterising the strategy | A config change, or one subclass |
| Different split rule (Splitwise) | `SplitStrategy` | One new class |

**The trade-off table to close with**

| Choice | Buys | Costs |
|---|---|---|
| Interface on the axis of change | The follow-up is a new class | An interface, a wiring point |
| CAS over a row lock | No blocking, no deadlock | One statement per resource |
| TTL on holds | Crashed holders cannot strand it | A slow user loses the resource |
| Injected `Clock` | Instant, exact expiry tests | Every time read must be injected |
| Allocation as a hint | Cheap common case | A retry path when stale |

**"If the interviewer says X → do Y"**

| Interviewer says | You do |
|---|---|
| "Design a parking lot" | The nine steps; write the scope on the board |
| "You decide" | State the defaults out loud, then proceed |
| "What about two users at once?" | Draw the interleaving, state the atomic step, state the cost |
| "Now add a second pricing model" | Point at the seam; one new class; nothing else changes |
| "Show me the code" | Core path only; narrate the decisions |
| "How would you test it?" | Three tests; a `Clock`, a barrier, the invariant |
| "Would that work across pods?" | Not with in-process locks — the atomicity belongs in the store |
| "What would you change at 10×?" | One named change, with the reason |

---

## 12. Cross-links

- Previous in this pillar: [UML and class diagrams](05-uml-and-class-diagrams.md) · [Concurrency and threading in LLD](04-concurrency-and-threading-in-lld.md) · [Design patterns](03-design-patterns.md) · [OOP principles and SOLID](02-oop-principles-and-solid.md) · [What is LLD and how to answer it](01-what-is-lld-and-how-to-answer.md)
- **The HLD counterpart, same shape:** [The five-step interview framework](../HLD/01-foundations/04-five-step-interview-framework.md) · [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md)
- The framework applied, end to end: [Parking lot](problems/01-parking-lot.md) · [LRU and LFU cache](problems/02-lru-lfu-cache.md) · [Hotel management](problems/03-hotel-management.md) · [BookMyShow seat locking](problems/04-bookmyshow-seat-locking.md) · [Elevator system](problems/05-elevator-system.md) · [Splitwise](problems/06-splitwise.md) · [ATM machine](problems/07-atm-machine.md) · [Library management](problems/08-library-management.md)
- The step that gets skipped: [Concurrency §4, the locking taxonomy](04-concurrency-and-threading-in-lld.md) · [BookMyShow seat booking](../Case-Studies/HLD/07-bookmyshow-seat-booking.md)
- The diagrams: [UML §3, notation and worked class diagram](05-uml-and-class-diagrams.md)
- Practice: [Foundations and capacity questions](../Interview-Questions/01-foundations-and-capacity-questions.md) · [Behavioral and project questions](../Interview-Questions/07-behavioral-and-project-questions.md) · [Capstone and mock interview](../00-meta/03-capstone-project-and-mock-interview.md)
- Revision: [Foundations and estimation cheat sheet](../Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md)
