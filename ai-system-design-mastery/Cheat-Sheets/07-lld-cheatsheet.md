# Low-Level Design Cheatsheet

**How to use this** — The LLD reference: the HLD/LLD distinction, the answer framework, SOLID, the patterns table, concurrency primitives, the classic problems one line each, UML notation, and the mistakes that cost offers. The three things to carry into any LLD interview: **the interviewer is testing whether your design survives the second requirement they are about to add**, **an LLD answer without a seam and a failure mode is a drawing rather than a design**, and **when you name a gap before they find it, it reads as judgement — when they find it, it reads as an oversight**.

---

## 1. HLD versus LLD — the distinction in one block

```
  HLD  ─ the SYSTEM  ──────────────────────────────────────────
        boxes, arrows, capacity, protocols, data stores
        "how many machines, which database, what happens
         when a region dies"
        graded on: judgement under scale and failure

  LLD  ─ the CODE  ────────────────────────────────────────────
        classes, interfaces, invariants, state machines
        "who owns this state, what is the legal transition,
         what breaks when two threads do it at once"
        graded on: modelling, coupling, correctness

  THE BRIDGE ─ the same system, one level down ────────────────
        take ONE box from the HLD and open it
        the HLD's constraints become the LLD's requirements
        do NOT re-derive the HLD — cite it and descend
```

| | HLD | LLD |
|---|---|---|
| **Unit** | a service, a store, a queue | a class, an interface, a state |
| **Question** | where does this run and how much | who owns this and what is legal |
| **Failure** | a region, a provider, a partition | a race, an illegal transition, a leak |
| **Numbers** | QPS, storage, latency, cost | complexity, thread count, retry budget |
| **Change** | add a replica, reshard, add a cache | add a subclass, add a row to a table |
| **Signal** | the trade-off you chose and why | the seam you left and the gap you named |

**The one sentence that carries the whole distinction:** the HLD decides *what must be true* of the system at scale; the LLD decides *who is responsible* for each of those truths in code — and the handoff between them is a table mapping each HLD component to the classes that implement it.

---

## 2. The LLD answer framework

**The 12 sections, in the order they buy you the most credit:**

```
  1  REQUIREMENTS      5 min, written.  Questions to ask + defaults to assume.
                        Pin: the core ops, the requirement most likely to
                        grow an "and also", the scale, the OUT-OF-SCOPE list.
  2  ENTITIES          which nouns have identity + a lifecycle?
                        the rest are VALUES. Say which is which, and why.
  3  DIAGRAM           entities + relationships + THE SEAM + dependency direction.
                        Not getters. Not every field. The seam must be visible.
  4  INTERFACES        the abstract classes, the enums, and the CONTRACTS
                        in behaviour terms ("never throws", not just "returns X")
  5  CODE              complete and RUNNABLE. If it does not run, it is prose.
  6  CONCURRENCY       what is shared, where the races are, what is NOT closed
  7  EXTENSIBILITY     the change → what you touch → what you must NOT touch
  8  FAILURE MODES     a table: failure → the class that handles it → still open
  9  PATTERNS          used AND rejected, each with the reason
 10  TRADE-OFFS        decision | alternative | what it costs | why this way
 11  Q&A               the second question, and the pushback
 12  CHEAT SHEET       one paragraph you could say walking out of the room
```

**The five moves that earn the most credit per minute:**

1. **Write the main flow as commented steps before implementing it.** It is where the interviewer sees the design rather than the typing, and it is where you catch the ordering problem (a dropped item must not consume budget) while it is still a comment.
2. **Name the seam out loud.** "This is the sentence most likely to grow an 'and also', so it goes behind an interface." That single sentence is the difference between a correct design and a senior one.
3. **Make it run.** A narrow working path with three passing checks beats a wide surface with nothing verified.
4. **Name your own gap before they find it.** "The dispatcher has no lease, so two workers can send one row twice — it is two columns and a visibility timeout."
5. **Cut your own speculative abstraction.** "If this is one lot with 200 spots, `PricingPolicy` is a file with one implementation and I would delete it."

---

## 3. SOLID in one table

| Letter | The rule in one line | The real-world example | The cost of over-applying it |
|---|---|---|---|
| **S** — Single Responsibility | One class, one **audience** — one reason to change | A `Notification` that formats, sends *and* records has three audiences: product, infra, reporting | Ten tiny classes that all change together is the same responsibility scattered |
| **O** — Open/Closed | Add behaviour by **adding code**, not editing working code | `ClassificationTable`: a new provider error code is a new **row**, and the classifier is untouched | An interface for a set that turns out to be closed is a file existing to be navigated past |
| **L** — Liskov Substitution | A subtype must be usable as the base **without the caller knowing** | A `Channel` documented as "never throws" and a `SlackChannel` that throws on 429 — a **strengthened guarantee** breaks every caller | Chasing it with defensive checks everywhere instead of fixing the subtype |
| **I** — Interface Segregation | No client is forced to implement what it does not use | A `Channel` with `schedule`/`cancel` where a webhook channel can only send — three `NotImplementedError` bodies | One-method interfaces for everything, so nothing composes |
| **D** — Dependency Inversion | Both policy and detail depend on an **abstraction the policy owns** | `Sender` takes `provider: Callable` rather than constructing a Firebase client — and that is what makes the tests run in milliseconds | Injecting a concrete class you also import is a parameter, not inversion |

**The honest meta-answer about SOLID:** every letter is a way to make **the second thing** cheap. If you cannot name the second thing, the abstraction is a guess — and a wrongly-guessed seam is worse than a missing one, because a seam can hide an ordering the design depends on (see §9, mistake 7).

**And the three that are not in SOLID but matter as much:** **DRY** is about knowledge, not text — two identical lines that change for different reasons are not duplication. **YAGNI** is the counterweight to OCP and it wins more often than candidates expect. **Composition over inheritance** — the test is "does the subclass change what the object **is**, or what it **does**?" What it does is a collaborator.

---

## 4. Patterns — name · intent · use here · cost

| Pattern | Intent | Use here | Cost |
|---|---|---|---|
| **Value Object** | A type described entirely by its data, frozen and interchangeable | `SendOutcome`, `Cursor`, `Money`, `RenderedMessage`, `Square` | A new object per change rather than a mutation; a real cost only in tight loops |
| **Entity** | Identity that persists through change | `Notification`, `Ticket`, `Reservation`, `Order` | Mutation must be funnelled through invariant-enforcing methods |
| **Strategy** | A rule varies and the caller must not know which runs | `Channel`, `PricingPolicy`, `QuietHoursPolicy`, `SchedulingPolicy` | One implementation and no test double means an interface that pays for nothing |
| **State** | The legal transitions are the design | Notification lifecycle, elevator motion, ATM flow, order status | An explicit machine for something that could be three booleans — only worth it when the transitions are the risk |
| **Template Method** | A shared spine with one varying step | `_ProviderChannel`: provider call, classify, return; subclasses fill `kind` and an allow-list | Inheritance, so it cannot be composed with another axis |
| **Ports & Adapters** | Every external system enters as an injected dependency | provider callables, `Clock`, the tz offset resolver, the event publisher | One constructor parameter per port; a port with one adapter and no fake is ceremony |
| **Repository** | Hide the index, the query and the locking | `NotificationStateStore`, `SeatStore`, `DedupeStore` | An abstraction over what is often a single table — worth it when locking must be hidden |
| **Facade** | One simple door onto a complicated subsystem | `NotificationApi.submit`, `ParkingLot.park` | Can become a god object if it accumulates policy instead of delegating it |
| **Atomic Claim** *(not GoF, and the most valuable one)* | Check-and-effect as **one** operation that returns ownership | dedupe, idempotency, seat hold, token bucket, driver assignment | The API must return an owner, not a boolean — and every caller must handle "somebody else has it" |
| **Null Object** | A neutral default instead of a `None` every caller checks | Permissive prefs for an unknown user; a limiter with no reserved bucket | Hides a missing configuration that might deserve to be loud |
| **Composite** | One thing that is many things | A request aggregating notifications aggregating attempts | Aggregate direction (max vs min) is a decision, and the wrong one is silent |

**Rejected, and the reason is the answer** — these come up in interviews and "I did not use it" is only half a sentence:

| Pattern | Rejected because |
|---|---|
| **Singleton** | Test isolation, hidden dependencies, and "exactly one" belongs to the wiring rather than the type |
| **Observer** | On a critical path, a subscriber that throws must not be able to fail the publisher — so the side effect is a direct call with a non-throwing contract |
| **Decorator** | For failover or retries it hides *what happened*: a `FailoverChannel` is where "failover" quietly becomes double-sending, and the caller's retry and the decorator's failover are now independent |
| **Builder** | When exactly one class constructs the object, a Builder is a way to skip the invariants that class enforces |
| **Chain of Responsibility** | A chain lets a link decide not to call the next — which is how a "pre-send hook" swallows a `SUPPRESS_TOKEN` and a dead token is never deleted |
| **Factory** | One product, one caller, no variation: indirection with no payoff |

---

## 5. Concurrency primitives

| Primitive | What it is for | The failure when it is wrong |
|---|---|---|
| **Mutex / `Lock`** | Mutual exclusion over a critical section | Held across I/O, it turns a slow provider into a stalled process |
| **`RLock`** | Reentrant: the same thread can re-acquire | A plain `Lock` deadlocks the **same thread** when a composed read method calls another read method on the same object |
| **Atomic claim** | Check-and-effect as one operation returning ownership | The `if not seen: mark()` window — the single most common shipped race |
| **Compare-and-set** | Optimistic update, retry on conflict | Under contention the retry loop livelocks; needs a backoff and a bound |
| **Read/write lock** | Many readers, one writer | Wrong for anything where the "read" mutates — an LRU `get` moves a node, so its read path serialises anyway |
| **Semaphore** | Bound concurrent access to N | Leaked permits become a permanent capacity reduction |
| **Lease / visibility timeout** | A worker claims work for a bounded time | No lease means two workers send the same row; a lease with no reclaim means a crash loses the work |
| **Immutable value** | No lock at all | Retrofitting immutability after callers mutate is a refactor across every call site |
| **Partition / confinement** | One owner per key | Costs throughput, and one stuck key blocks its whole partition |
| **Atomic counter** | `+=` across threads | `i += 1` is load, add, store — the GIL makes bytecodes atomic, not sequences |

**The four questions to answer about any class you just drew:**

1. **What is shared and mutable?** Not "is there a lock" — which fields can two threads write.
2. **Where is the check-then-act?** Dedupe, idempotency, rate limiting and "is this seat taken" are one race with four nouns.
3. **Is anything held across I/O?** The rule: no lock across a provider call, which is why a limiter returns a **wait time** rather than blocking.
4. **What ordering do you promise, and what does it cost?** Global ordering is a fiction; per-key ordering is a partition; per-entity ordering is a sequence number.

---

## 6. The classic problems — one line each

| Problem | The core modelling decision | The seam | The failure that must be handled |
|---|---|---|---|
| **Parking lot** | `Ticket` is the entity, `Spot` is a contended resource with a state — **not** `vehicle.spot` | `PricingPolicy`, `SpotAllocator` | The search-then-claim race on a free spot |
| **LRU / LFU** | LRU: map + doubly-linked list, and `get` is a **write** (move to front) | Eviction policy | LFU's cumulative counter lets a cold old key evict a hot new one — needs ageing |
| **Hotel management** | **A room type is not a room** — book the type, assign the room at check-in | `RatePlan`, the room-assignment step | The half-open interval: checkout and check-in on the same day do **not** conflict |
| **BookMyShow seats** | An atomic conditional update on the seat row; the affected-row count **is** the answer | `SeatAllocator`, hold-expiry policy | The hold expiring mid-payment — choose the failure (refund vs double-book) and write it down |
| **Elevator** | A **state machine**, not a scheduler; the invariants are the safety property | `SchedulingPolicy` | A request arriving mid-transition — it must be an event, not a field write |
| **Chess** | Movement is per-piece; **legality is per-position** (does this leave my king in check) | `Piece.moves`, the rule layer | Re-deriving undo from the rules — store the delta instead |
| **Splitwise** | A `Group` owns `Expense`s; balances are **derived**, and deriving is where the simplification lives | `SplitStrategy`, `SettlementStrategy` | Floating-point money, and the balance that must sum to zero |
| **ATM** | A persisted state machine plus a **two-phase dispense** | the transaction flow | An irreversible action (dispense) and a reversible one (debit) — do the reversible one first, idempotently |
| **Library management** | A `Loan` links a copy to a member; **a title is not a copy** | `FinePolicy`, `SearchStrategy` | The reservation queue versus a walk-in borrower |
| **Vending machine** | A state machine over `(item, payment)` with an explicit `REFUNDING` state | `PaymentStrategy` | Payment taken and the item not dispensed — that state must exist |
| **Rate limiter** | Token bucket for bursts, sliding-window counter for exactness | the algorithm, and the return type | The fixed-window boundary burst: 200 requests in two seconds against a limit of 100/minute |
| **Logger framework** | Logger decides *whether*, appender decides *where*, formatter decides *what it looks like* | `Appender`, `Formatter` | An unbounded async queue — decide block / drop / drop-with-a-counter |
| **Pub-sub** | At-least-once **plus** a deduplicating consumer; neither half is a guarantee alone | the broker, the consumer's idempotency key | The key must come from the **message**, not the delivery, or redelivery is a duplicate effect |
| **Tic-tac-toe** | The board is a value; the win check is a function of the board | `Player` / move source (human, AI) | A win check that misses the diagonal, and mutable board state shared between players |
| **Snake and ladder** | The board is a fixed value map; a `Game` owns the turn state | `DiceStrategy`, `BoardLayout` | The overshoot rule at the last square, and the exact-landing requirement |
| **Food delivery** | **Push versus pull** — who has the better information and who bears a bad match | the offer cascade, the ranking | Two assignments for one order when a timeout races an in-flight acceptance |

---

## 7. UML notation — the five things worth getting right

```
  ─────────────▶   association            open arrow; "knows about"
  ◆────────────▶   composition             filled diamond; the part CANNOT exist
                                           without the whole (Order ◆─ LineItem)
  ◇────────────▶   aggregation             hollow diamond; the part exists
                                           independently (Lot ◇─ Vehicle)
  ─────────────▷   inheritance             hollow triangle; "is-a"
  ┄┄┄┄┄┄┄┄┄┄┄┄▷   realisation             dashed + hollow triangle; "implements"
  ──────────────   dependency               dashed open arrow; "uses, transiently"

  MULTIPLICITY GOES ON THE END YOU ARE COUNTING
      Lot  1 ──────────── *  Spot          one lot has many spots
      Ticket 1 ────────── 1  Spot          one ticket holds one spot
      User  * ──────────── *  User         a follow graph — model it as an
                                           association OBJECT (FollowGraph),
                                           never as two mutual collections
```

**What the diagram must show:** entities, relationships with multiplicity, the interfaces that are the seams, and the **direction of the dependencies**. **What it must leave out:** getters, setters, every field type, constructors, and most of the inheritance.

**The test of a diagram:** *where does the next requirement attach?* If the seam is not visible on the page, redraw it rather than adding boxes.

**One box doing real work** is worth more than ten boxes doing none — in the notification bridge, the `DecisionService.decide` box lists its **eight numbered steps** inside it, because that box *is* the design.

---

## 8. The numbers worth knowing

| Quantity | Value | Why it matters in LLD |
|---|---|---|
| Retry attempts before DLQ | 5 | Bounds the cost of a terminal error classified as retryable |
| Jitter | ±20% minimum | Without it, a provider recovery is a synchronised second storm |
| Idempotency key TTL | 24 h | Covers a long producer retry |
| Dedupe TTL | minutes (chat) / days (orders) | Matches the window in which a repeat is meaningful |
| Quiet hours | 22:00–08:00 local | And the bypass is an **event-type allow-list**, not a priority claim |
| Per-user marketing cap | 5/day | The count is per **local** day, so the key survives a DST overlap |
| OTP validity | 5–10 min | A hard window — fail fast rather than delay |
| Webhook response budget | < 100 ms | Providers disable slow endpoints, and then you get nothing |
| Push payload limit | 4 KB (FCM/APNs) | Never a token, never a full account number |
| SMS cost | ~$0.0075 each | 5 retries × 5 M messages = $187,500 in one incident |
| Token hygiene | delete after 90 days, or on `410`/`UNREGISTERED` | A stale token is a notification that can never arrive |
| Booking hold | ~10 min | Must extend on payment start, or the sweeper eats a paying customer |

**The three LLD-native numbers**, which have no unit but matter more: the retry budget (attempts, not elapsed time), the complexity of the hot operation (O(1) claim, not O(n) scan), and the **number of classes you would have to open to answer a question about the design**.

---

## 9. Fifteen mistakes that cost offers

| # | The mistake | The fix |
|---|---|---|
| 1 | **A noun list instead of a model.** Every noun becomes a class, including the ones with no lifecycle | Three questions per noun: does it have an id, must it survive a restart, does it change state |
| 2 | **`vehicle.spot = s3`** — occupancy on the wrong object | State goes on the thing being **contended for**, not the thing passing through |
| 3 | **Check-then-act.** `if seat.available: seat.book()` | One atomic claim whose result *is* the answer |
| 4 | **A boolean from a limiter.** It makes "reject" the only expressible outcome | Return a **wait time**; the caller decides delay versus fail-fast |
| 5 | **Every failure retried.** A dead token retried five times | Classify failures as data: retryable / terminal / suppress-token |
| 6 | **`datetime.now()` inlined.** Five classes, five untestable behaviours | Inject the clock; time is an input to every interesting behaviour |
| 7 | **A seam that hides an ordering.** Splitting eight ordered checks into four collaborators makes the order a property of nothing | Keep the sequence in one readable method; extract the collaborators *inside* the steps |
| 8 | **A relative retry time.** "Now + 5 minutes" computed twice, on two workers | An **absolute** `scheduled_at`; clock skew is permanent |
| 9 | **A string where an enum belongs** — `if state == "deliverd"` is a silent no-op | Enum for the closed set your code branches on; strings for the open set that arrives from outside; a table between them |
| 10 | **Two rules for one question.** A `get` that returns deleted rows while `get_many` filters them | One rule, enforced in one place, with a docstring saying so |
| 11 | **A singleton for the store.** "Exactly one" as a fact about the type | Inject it; "exactly one" belongs to the wiring, where it can change |
| 12 | **Not knowing what happens when two threads run your method.** "It's thread-safe" with no argument | Name the shared state, the check-then-act, and the race you did **not** close |
| 13 | **A speculative interface.** One implementation, no test double, no named second | Name the change it makes cheap, or delete it |
| 14 | **Claiming a guarantee you cannot keep.** "Exactly-once delivery", "ordered notifications" | At-least-once plus dedupe; ordering is per-key and it costs throughput |
| 15 | **Silence about the gap.** Hoping they will not ask about the lease / the TTL / the hot key | Name it first: "two columns and a visibility timeout" |

**The two that are really the same mistake:** 7 and 12. Both are a design that reads well in a diagram and loses the thing it exists to get right — an ordering, or a race. Every other mistake on this list is recoverable in the interview; those two are the ones that come back in production.

---

## 10. The close — what to say when you have five minutes left

```
  1  Make the main path RUN end to end, narrow rather than wide.
  2  Write the checks for what runs — three passing beats nothing verified.
  3  Say what is MISSING and where it would go.
  4  Do NOT start anything new.
  5  Close with: "the two gaps I would fix first are X and Y, and
     what I would ask you is which requirement you think I have
     not accounted for."
```

**And the seven sentences that do the most work in an LLD round:**

1. "The core operations are A and B; everything else serves them."
2. "This requirement is the one most likely to grow an 'and also', so it goes behind an interface."
3. "Here is the main flow as steps, before I implement it."
4. "The order of these checks is the design: a dropped item must not consume budget."
5. "The race I have not closed is X, and closing it is a visibility timeout."
6. "If this is one lot with 200 spots, I would delete `PricingPolicy` and say why."
7. "That is a real gap. Two columns, and it does not touch anything else."

---

## If the interviewer says X → answer Y

| They say | You say |
|---|---|
| "Walk me through the classes" | "The entities are … and the seams are … — I will start with the flow through the facade" |
| "Is it thread-safe?" | "Check-and-effect pairs are single operations; no lock across I/O; and here is the race I have *not* closed" |
| "How would you extend it?" | Name the seam, the change it makes cheap, and one change it does **not** accommodate |
| "What if the requirements change?" | "I would put it behind X, because X is the sentence most likely to grow an 'and also'" |
| "Abstract class or interface?" | Interface when implementers share nothing; abstract class when they share a spine and vary in one place |
| "Why not a Singleton?" | "Test isolation, hidden dependencies, and 'exactly one' belongs to the wiring" |
| "Why not just retry?" | "Three outcomes, not two: retrying a dead token burns shared capacity and never deletes it" |
| "Isn't that over-engineered?" | Cut the speculative parts yourself; keep the atomic claim and the injected clock |
| "Why not use a library?" | Name the library, then say what writing it once taught you about what it does |
| "You have too many classes" | "How many would I have to open to answer a design question? Eight — the rest are values and enums" |
| "It doesn't handle X" | Behind an existing seam / a real gap with a cost / out of scope with what changes it |
| "How do you test it?" | "Time and the external system are inputs — a manual clock and a scripted provider" |
| "What did you not do?" | Name the two gaps before they find them |
| "Any questions?" | "What is the one requirement you think I have not accounted for?" |

---

## Cross-links

**The problems, worked in full:** [Parking lot](../LLD/problems/01-parking-lot.md) · [LRU/LFU cache](../LLD/problems/02-lru-lfu-cache.md) · [Hotel management](../LLD/problems/03-hotel-management.md) · [BookMyShow seat locking](../LLD/problems/04-bookmyshow-seat-locking.md) · [Elevator system](../LLD/problems/05-elevator-system.md) · [Splitwise](../LLD/problems/06-splitwise.md) · [ATM machine](../LLD/problems/07-atm-machine.md) · [Library management](../LLD/problems/08-library-management.md) · [Chess game](../LLD/problems/09-chess-game.md) · [Vending machine](../LLD/problems/10-vending-machine.md) · [Rate limiter](../LLD/problems/11-rate-limiter.md) · [Logger framework](../LLD/problems/12-logger-framework.md) · [Pub-sub system](../LLD/problems/13-pub-sub-system.md) · [Tic-tac-toe](../LLD/problems/14-tic-tac-toe.md) · [Snake and ladder](../LLD/problems/15-snake-and-ladder.md) · [Food delivery](../LLD/problems/16-food-delivery.md)

**The bridges — HLD systems opened one level down:** [URL shortener LLD](../Case-Studies/LLD/01-url-shortener-lld.md) · [WhatsApp LLD](../Case-Studies/LLD/02-whatsapp-lld.md) · [Uber matching LLD](../Case-Studies/LLD/03-uber-matching-lld.md) · [Instagram feed LLD](../Case-Studies/LLD/04-instagram-feed-lld.md) · [Notification system LLD](../Case-Studies/LLD/05-notification-system-lld.md)

**The HLDs they descend from:** [URL shortener](../Case-Studies/HLD/01-url-shortener.md) · [WhatsApp](../Case-Studies/HLD/02-whatsapp-chat-system.md) · [Notification system](../Case-Studies/HLD/03-notification-system.md) · [Instagram news feed](../Case-Studies/HLD/04-instagram-news-feed.md) · [Uber ride-hailing](../Case-Studies/HLD/06-uber-ride-hailing.md) · [BookMyShow seat booking](../Case-Studies/HLD/07-bookmyshow-seat-booking.md)

**The question bank:** [LLD and OOP — 50 questions](../Interview-Questions/08-lld-questions.md)

**Other cheat sheets:** [Foundations and estimation](01-foundations-and-estimation-cheatsheet.md) · [Networking](02-networking-cheatsheet.md) · [Scaling, caching and databases](03-scaling-caching-db-cheatsheet.md) · [RAG and LLMs](04-rag-and-llm-cheatsheet.md) · [Agentic AI](05-agentic-ai-cheatsheet.md) · [Operationalisation](06-operationalisation-cheatsheet.md)

---

## Grounding note

**What is grounded.** The cross-links above are real files in this repository. The techniques, defaults and numbers in §6 and §8 come from the worked problems in `LLD/problems/` and from the bridges in `Case-Studies/LLD/` — where the notification rows in particular (five attempts, ±20% jitter, 22:00–08:00 quiet hours, the event-type bypass allow-list, the 24-hour idempotency TTL, the 90-day token hygiene, the < 100 ms webhook budget) are the HLD's own tuning table, and where the assertions that back them execute and pass.

**What is not grounded.** The reference repositories this knowledge base was assembled from contain **no OOP or LLD material at all** — no pattern catalogue, no class diagrams, no machine-coding transcripts, and no UML. The source material names low-level design and the classic problems without teaching them. Everything here — the framework, the SOLID table, the pattern costs, the misuse list, the one-liners — is **constructed** from first principles and from the implementations in this repository. Where a claim in the linked files is checkable it is checked by executed assertions; where it is a judgement it is presented as one, with the alternative and its cost stated beside it.

**The honest way to use this sheet.** It is a revision aid, not a substitute. One line per classic problem cannot teach a seat-locking design that needs six hundred — follow the links when you have the time and use this when you have twenty minutes. And rehearse §10 and the last table out loud: the difference between knowing this material and *saying* it under time pressure is the thing being tested.
