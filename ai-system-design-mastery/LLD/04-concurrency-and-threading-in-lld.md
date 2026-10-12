# Concurrency and Threading in LLD

> **Grounding note.** The source transcript names concurrency in exactly one place, at **[9:15]–[9:18]**: *"how is BookMyShow built? how do you do seat locking? how does concurrent booking happen?"* It asks the question and does not answer it. It never names a lock, a race condition, optimistic versus pessimistic locking, or a thread pool. **Everything in this file is standard industry practice.** What the transcript *does* establish is that concurrency is the thing the interviewer is waiting to see — that a candidate who designs clean classes and then discovers two users booking the same seat is the candidate who fails. That framing is transcript-grounded; the mechanisms below are not.

---

## TL;DR

Concurrency in LLD is one question asked five ways: **what can two actors do to the same piece of state at the same time, and what happens if they interleave badly?** A **race condition** is when the answer depends on timing. **Atomicity** is the property that a multi-step operation appears to happen all at once. **Mutual exclusion** is how you buy atomicity, and the price is contention. The four thread-safety strategies, in order of preference, are **immutability** (no shared mutable state at all), **confinement** (state is owned by one thread), **stateless design** (no state to race on), and **synchronisation** (a lock, which is correct but slow and deadlock-prone). The four locking designs — pessimistic, optimistic, atomic compare-and-swap, and external/store-level — are a spectrum from "block everyone" to "retry the loser", and the right choice follows from the *contention rate*, not from taste. In an interview, the sentence that scores is: **"Here is the interleaving that breaks it, here is the fix, and here is what the fix costs in throughput."** Naming a lock without naming the interleaving is a guess.

> **60-second interview answer:** "Concurrency in an LLD round is about finding the shared mutable state and proving it cannot be corrupted. In the seat-booking design, the shared state is the seat's status, and the failing interleaving is two users both reading 'FREE' and both proceeding. There are four ways to fix it, and they differ in where the atomicity lives: a pessimistic row lock in the database (`SELECT ... FOR UPDATE`), an optimistic version column that fails the loser's update, an atomic compare-and-swap in Redis or a database conditional update, or a single-threaded per-show actor that serialises all bookings for that show. I would choose based on contention: high contention on a hot show favours the atomic conditional update, because it never blocks and the loser retries immediately. Then I would handle the second-order problems — lock ordering to avoid deadlock, a timeout plus idempotency key so a retried request cannot double-charge, and lazy-plus-sweeper expiry so a crashed holder cannot strand a seat forever. And I would say what I am giving up: a global lock is the simplest correct answer and it is the one that caps throughput."

---

## 1. Definition & mental model

### 1.1 The vocabulary, precisely

| Term | Precise meaning | The seat example |
|---|---|---|
| **Race condition** | The result depends on the interleaving of operations | Both users' bookings succeed; both hold seat A12 |
| **Critical section** | Code that must not interleave with itself | read-status → check-free → write-held |
| **Atomicity** | An operation appears indivisible to other observers | The seat goes FREE→HELD in one indivisible step |
| **Mutual exclusion** | At most one actor is inside the critical section at a time | The row lock, or the CAS |
| **Visibility** | A write by one thread is observed by another | Thread 2 must not see a stale cached status |
| **Ordering** | Operations happen in a defined order | Hold must precede charge, not the reverse |
| **Contention** | How often actors want the same resource at once | Ten thousand users, one show, one seat |
| **Deadlock** | Two actors each hold what the other needs; neither progresses | Booking holds seat A then waits for B; cancel holds B then waits for A |
| **Livelock** | Both keep retrying and neither progresses | Two optimistic updates failing each other forever |
| **Starvation** | One actor never gets the resource | A low-priority refund never acquires the lock |

### 1.2 The mental model: shared mutable state is the enemy

**The single organising idea:** concurrency bugs require *shared* and *mutable*. Remove either and the bug is impossible.

```
                    Is the state SHARED?
                    ┌──────────┴──────────┐
                   NO                    YES
                    │                     │
             No race possible.    Is the state MUTABLE?
             Replicate freely.    ┌───────┴───────┐
                                 NO              YES
                                  │               │
                          No race possible.   ⚠ HERE BE RACES
                          Share freely.       Pick a strategy (§4)
```

So the design question is never "where do I put the lock?" It is: **"can I make this state immutable, or unshared, so that I do not need the lock at all?"** An interviewer will reward that question far more than a lock.

**Applied to the seat map**, in the order you should consider them:

1. **Make it immutable?** A booking record is immutable once written; the *seat status* is not, because it must change. So this fails on the seat — but succeeds on the booking, which is why bookings are append-only rows, not mutable rows.
2. **Make it confined?** A given show's seats are owned by the process that owns the show. This is the "actor per show" design: all writes for show X go through one actor. It is correct and simple, and it caps per-show throughput at one core.
3. **Make the design stateless?** The pricing calculation, the fee computation, the availability *read* — all stateless, all safe. Much of the system should be here.
4. **Else synchronise.** The seat's status transition needs a lock, and now you choose *which kind* (§4).

### 1.3 The three altitudes of the same problem

The same race appears at three levels, and interviewers move between them:

| Altitude | Shared state is | The "lock" is | Where it lives in this repo |
|---|---|---|---|
| **In-process / threads** | An object field or a dict | A `threading.Lock`, an atomic, or immutability | This file, §3–§4 |
| **In-database** | A row or a counter | `SELECT ... FOR UPDATE`, a conditional `UPDATE`, a unique constraint | [problems/04-bookmyshow-seat-locking.md](problems/04-bookmyshow-seat-locking.md) |
| **Distributed** | Data in Redis, or across services | `SET NX`, a Lua script, a distributed lock, or a consensus protocol | [Redis deep dive](../HLD/03-scaling/03-redis-deep-dive.md) · [BookMyShow](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) |

**The crucial insight to state:** the *mechanisms* differ but the *reasoning* is identical. Every level is answering "what is the atomic step, and who wins the tie?" If you can reason about one, you can reason about all three — which is exactly what the interviewer is testing.

### 1.4 What concurrency in LLD is *not*

| Not this | Why the confusion exists | The distinction |
|---|---|---|
| **An HLD scaling discussion** | Both mention "concurrent users" | HLD asks "how many machines"; LLD asks "which two operations may interleave" |
| **A language feature tour** | `async`/`await` is everywhere | `async` is a *concurrency model*, not a correctness guarantee. Blocking inside `async` is still a race, and a worse one |
| **Just "add a lock"** | Locks are the obvious answer | A lock is the *last* resort; immutability and confinement are preferred |
| **Only about performance** | "Thread pools make it fast" | The primary goal is **correctness**. A fast wrong answer is worse than a slow right one |
| **Only about databases** | Transactions feel like the answer | An in-process counter has the same problem with no database in sight |

---

## 2. Why it exists — the problem it solves

### 2.1 What breaks without a concurrency story

| Failure | Mechanism | Real consequence |
|---|---|---|
| **Double booking** | Two readers both see FREE and both write HELD | One seat, two customers, one angry support ticket |
| **Double charge** | A retried request is not idempotent | Customer charged twice; refund plus a support cost |
| **Lost update** | Read-modify-write without atomicity | A counter shows 43 instead of 71 |
| **Overbooking beyond policy** | Availability check not atomic with the reservation | The hotel sells 102 rooms in a 100-room property |
| **Deadlock** | Two multi-resource operations in opposite orders | A request hangs until a timeout; the user sees a spinner |
| **Stranded resource** | A holder crashes without releasing | The seat is held forever; nobody can book it |
| **Thundering herd** | A cache entry expires; ten thousand requests hit the database | The database falls over at the moment of highest demand |
| **Split brain** | Two nodes each believe they hold the lock | Two writers, corrupted state |

### 2.2 Why the interviewer asks this

Because it is the **hardest thing to retrofit**. A candidate can fix a missing class in a minute; discovering on the whiteboard that the entire booking flow needs re-architecting around a single atomic step is the failure the round is designed to expose. The signal is not "do you know what a mutex is" — it is:

- **Do you find the shared state unprompted?** The candidate who says "here is where two users collide" before being asked scores highest.
- **Do you reason about interleaving rather than naming a tool?** "Two reads, then two writes" beats "I'd use a mutex".
- **Do you handle the second-order problems?** Deadlock, timeout, retry, idempotency, and stranded-hold recovery.
- **Do you know the cost?** A global lock is correct and slow — saying so is the answer.

### 2.3 The honest counter-argument

**Most systems do not have a concurrency problem at the class level, and adding locks to code that has none is a pure cost.** If a service is stateless, or if its mutable state is per-request, there is nothing to synchronise, and a lock there is theatre. The position to take: *"this class is stateless so it is safe; the shared state is the seat status in the store, so the atomicity belongs there, not here."* **That sentence demonstrates more concurrency understanding than any lock you could add.**

---

## 3. Internal mechanics

### 3.1 The anatomy of a race condition

The canonical failure, in the seat-booking case, shown as an interleaving. The code looks correct and is not:

```python
# The BROKEN version. Reads, decides, writes — three steps, no atomicity.
def hold_seat_broken(seat_id: str, user_id: str) -> bool:
    seat = store.get(seat_id)              # (1) READ
    if seat.status != "FREE":              # (2) DECIDE
        return False
    seat.status = "HELD"                   # (3) WRITE
    seat.held_by = user_id
    store.save(seat)
    return True
```

```
  TIME   THREAD A (user 1, seat A12)        THREAD B (user 2, seat A12)
  ────   ─────────────────────────────       ─────────────────────────────
   t1    READ  seat A12 → status = FREE
   t2                                        READ  seat A12 → status = FREE
   t3    DECIDE: status == FREE → proceed
   t4                                        DECIDE: status == FREE → proceed
   t5    WRITE status = HELD, held_by = U1
   t6                                        WRITE status = HELD, held_by = U2
   t7    return True (believes it holds)     return True (believes it holds)
  ────   ─────────────────────────────       ─────────────────────────────
  RESULT: seat A12 is held by U2. Both users were told they got it.
          U1 pays. U1 arrives at the cinema. U1 has no seat.
```

**The rule this teaches:** the window is *between the read and the write*. Every race in this file is a variation on widening or closing that window. `check-then-act` and `read-modify-write` are the two shapes; both are broken by default.

### 3.2 The critical section, and why it must be small

Once you know where the window is, the fix is to make read-decide-write indistinguishable from the outside. The naive fix — wrap the whole function in one lock — is correct and expensive:

```python
class LockedSeatStore:
    """Correct. And the lock is held across I/O, which is the mistake that turns
    a correctness fix into a throughput outage."""

    def __init__(self, backing: "SeatRepository") -> None:
        self._backing = backing
        self._lock = threading.Lock()          # ONE lock for ALL seats

    def hold_seat(self, seat_id: str, user_id: str) -> bool:
        with self._lock:                        # ← every seat serialised on one mutex
            seat = self._backing.get(seat_id)   # I/O inside the lock
            if seat.status != "FREE":
                return False
            seat.status = "HELD"
            seat.held_by = user_id
            self._backing.save(seat)            # I/O inside the lock — the killer
            return True
```

Two problems, and they must be named separately in an interview:

1. **Scope is too wide.** Seats A12 and B07 do not conflict, yet this lock serialises them. Throughput is now one booking at a time for the entire venue.
2. **The lock is held across I/O.** A 5 ms database write inside the lock caps throughput at roughly 200 bookings/second *per process*, no matter how many cores are idle. This is the single most common concurrency mistake in interview answers.

**The fix for scope: lock striping.** Partition the key space into a fixed number of locks so that unrelated keys rarely contend:

```python
class StripedSeatStore:
    """STRIPING: 64 locks instead of 1. Seat A12 and seat B07 almost certainly
    hash to different stripes, so they proceed in parallel. Same seats still
    serialise, which is exactly what correctness requires."""

    def __init__(self, backing: "SeatRepository", stripes: int = 64) -> None:
        self._backing = backing
        self._locks = [threading.Lock() for _ in range(stripes)]

    def _lock_for(self, seat_id: str) -> threading.Lock:
        bucket = int(hashlib.sha256(seat_id.encode()).hexdigest()[:8], 16) % len(self._locks)
        return self._locks[bucket]

    def hold_seat(self, seat_id: str, user_id: str) -> bool:
        lock = self._lock_for(seat_id)
        with lock:
            seat = self._backing.get(seat_id)
            if seat.status != "FREE":
                return False
            seat.status = "HELD"
            seat.held_by = user_id
            self._backing.save(seat)
            return True
```

**Striping is still wrong across processes.** Two pods each hold their own lock array and will happily double-book. That is why the real answer for seat locking puts the atomicity in the *store* — a conditional update or a `SET NX` — and this in-process discussion exists to show that you know why (§4.4). See [problems/04-bookmyshow-seat-locking.md](problems/04-bookmyshow-seat-locking.md) for the store-level design.

### 3.3 Atomicity: making the whole step indivisible

An atomic primitive is one the hardware or the store guarantees cannot be observed half-done. There are only a handful, and knowing which one you are relying on is the whole skill.

| Primitive | Guarantee | Where it lives |
|---|---|---|
| **CPU atomic read/write** (aligned word) | A single load or store is not torn | Everywhere, implicitly |
| **Test-and-set / compare-and-swap (CAS)** | Compare and swap, indivisibly | CPU instructions, `threading.Lock` internals, Redis `WATCH`/Lua, SQL conditional update |
| **Fetch-and-add** | Increment and return the previous value, indivisibly | Counters, sequence generation |
| **Mutex / semaphore** | Mutual exclusion over a region | `threading.Lock`, `asyncio.Lock`, DB row lock |
| **Transaction** | A group of operations is all-or-nothing, with isolation | Databases |
| **Single-writer actor** | Only one thread ever touches this state | A queue plus a dedicated consumer |

**The interview-relevant point:** *`SELECT ... FOR UPDATE` gives you a mutex; a conditional `UPDATE ... WHERE status = 'FREE'` gives you a CAS.* They are the same correctness with very different failure behaviour, and choosing between them is §4.

```python
# CAS in the database: the atomic step IS the update's WHERE clause.
# Rowcount 1 → this caller won. Rowcount 0 → someone else did. No lock held.
def hold_seat_cas(conn: "Connection", seat_id: str, user_id: str,
                  hold_id: str, ttl_seconds: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE seats
               SET status = 'HELD',
                   held_by = %s,
                   hold_id = %s,
                   hold_expires_at = now() + make_interval(secs => %s),
                   version = version + 1
             WHERE seat_id = %s
               AND status = 'FREE'
            """,
            (user_id, hold_id, ttl_seconds, seat_id),
        )
        return cur.rowcount == 1     # ← the entire concurrency control, in one line
```

### 3.4 Memory visibility and the ordering trap

Mutual exclusion buys you two things, and candidates usually mention only the first:

1. **Atomicity** — the region does not interleave.
2. **Visibility and ordering** — writes made inside the lock are visible to the next holder, and do not appear reordered.

**Why this matters even in Python.** The GIL makes individual bytecode operations atomic, which tempts people into "I do not need a lock for a simple increment." That is wrong: `counter += 1` is a *load, add, store* — three bytecodes — and the GIL is released between them. The GIL guarantees no torn memory, not no lost updates.

```python
import threading

counter = 0

def naive_increment(n: int) -> None:
    """BROKEN even under the GIL: `counter += 1` is LOAD, ADD, STORE, and the
    interpreter may switch threads between them."""
    global counter
    for _ in range(n):
        counter += 1


def locked_increment(n: int, lock: threading.Lock) -> None:
    global counter
    for _ in range(n):
        with lock:                 # atomic AND visible AND ordered
            counter += 1
```

**Run it:** four threads × 100 000 increments with the naive version yields something like 250 000–390 000 instead of 400 000, differently each run. That nondeterminism *is* the definition of a race.

### 3.5 The four thread-safety strategies, ranked

Every thread-safety answer in an interview should walk these in order, because the order is the judgement being tested.

| Rank | Strategy | Idea | Cost | Use when |
|---|---|---|---|---|
| **1** | **Immutability** | The object cannot change after construction | Allocation, and you must restructure updates as replacements | Value objects, money, tickets, booking records, config |
| **2** | **Confinement** | State is owned by exactly one thread | A queue hop; the owner is a throughput ceiling | Per-show actors, single-writer consumers, connection pools |
| **3** | **Statelessness** | There is no state to race on | Nothing — this is free | Pricing calculators, validators, formatters, most services |
| **4** | **Synchronisation** | A lock or an atomic primitive protects the state | Contention, deadlock risk, held-across-I/O bugs | Only when 1–3 are genuinely impossible |

**The sentence to say:** *"I would prefer immutability or confinement here; a lock is what I fall back to when the state genuinely must be shared and must change, and then I keep the critical section to a single store operation."*

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Money:
    """STRATEGY 1 — IMMUTABLE. Frozen, so it cannot be raced on. Arithmetic
    returns a new instance. Nothing to lock, ever."""
    amount_minor: int
    currency: str

    def __add__(self, other: "Money") -> "Money":
        if other.currency != self.currency:
            raise ValueError(f"currency mismatch: {self.currency} vs {other.currency}")
        return Money(self.amount_minor + other.amount_minor, self.currency)

    def __mul__(self, factor: int) -> "Money":
        return Money(self.amount_minor * factor, self.currency)


class PricingCalculator:
    """STRATEGY 3 — STATELESS. No fields, therefore no race, therefore no lock.
    Note that this is not a compromise: it is strictly better than a locked
    stateful version, at zero cost."""

    def __init__(self, rates: "RateTable") -> None:   # injected, and itself immutable
        self._rates = rates

    def price(self, show_id: str, seats: list[str], now: float) -> Money:
        subtotal = Money(0, "INR")
        for seat in seats:
            subtotal = subtotal + self._rates.base_for(show_id, seat)
        return subtotal * self._rates.multiplier_at(now)
```

### 3.6 Contention: why the metrics matter more than the mechanism

Contention is not "many users". It is **how often two actors want the same key at the same instant**. The distinction drives the entire choice of locking design.

```
   LOW contention                        HIGH contention
   (1000 shows, spread out)              (one blockbuster, one hot seat)
        │                                      │
        ▼                                      ▼
   Pessimistic locks are fine            Pessimistic locks queue and time out
   (rarely collide)                      Optimistic retries thrash (livelock)
        │                                      │
        ▼                                      ▼
   Simplicity wins:                     Atomic conditional update wins:
   SELECT ... FOR UPDATE                 one statement, no blocking, loser
   is readable and correct               gets rowcount 0 and a clear answer
```

The measured quantity is the **conflict rate**: the fraction of attempts that fail because someone else won. Rule of thumb from practice: below ~1 % conflict, any design works; 1–10 % favours optimistic with a bounded retry; above ~10 % favours an atomic update or a queue, and pessimistic locks start producing timeouts that look like outages.

**Say the numbers out loud.** "At 1 % conflict, two users rarely collide, so I would take the simpler pessimistic lock. At a hot show, thousands contend for the same few seats, and I want the atomic conditional update because nobody ever blocks and the loser gets an immediate, honest 'taken'." That is the answer.

---

## 4. Variants & taxonomy

### 4.1 Pessimistic locking

**Idea:** acquire exclusive access *before* reading, so nobody else can read-then-write.

```python
def hold_seat_pessimistic(conn: "Connection", seat_id: str, user_id: str,
                          hold_id: str, ttl_seconds: int) -> bool:
    """SELECT ... FOR UPDATE. The row is locked; a competing transaction BLOCKS
    here until this one commits or rolls back."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, held_by FROM seats WHERE seat_id = %s FOR UPDATE",
            (seat_id,),
        )
        row = cur.fetchone()
        if row is None or row[0] != "FREE":
            return False
        cur.execute(
            """
            UPDATE seats
               SET status = 'HELD', held_by = %s, hold_id = %s,
                   hold_expires_at = now() + make_interval(secs => %s),
                   version = version + 1
             WHERE seat_id = %s
            """,
            (user_id, hold_id, ttl_seconds, seat_id),
        )
        return True
```

| Buys | Costs |
|---|---|
| Simple to reason about; the whole block is serialised | The lock is held across the whole transaction, so a slow step blocks everyone |
| No retry logic; the loser just waits and then sees the truth | Queueing under contention; lock-wait timeouts that surface as 500s |
| Works for multi-row operations (all seats in one booking) | Deadlock risk if two bookings lock rows in different orders (§6.3) |
| Correct across processes and pods | Connection-pool exhaustion: waiting transactions each hold a connection |

**Multi-seat bookings need a fixed lock order.** If booking one locks A12 then A13 and booking two locks A13 then A12, they deadlock. **Always `ORDER BY seat_id` before locking.** That single line is the most valuable deadlock fix in this file.

### 4.2 Optimistic locking

**Idea:** read freely, and let the *write* fail if someone changed the row since. A version column detects the conflict.

```python
class OptimisticSeatStore:
    """Read a version, then update WHERE version = <the version you read>.
    Rowcount 0 means someone beat you — retry, or report 'taken'."""

    MAX_ATTEMPTS = 3

    def __init__(self, pool: "ConnectionPool") -> None:
        self._pool = pool

    def hold_seat(self, seat_id: str, user_id: str, hold_id: str,
                  ttl_seconds: int) -> "HoldOutcome":
        for attempt in range(self.MAX_ATTEMPTS):
            with self._pool.acquire() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT status, version FROM seats WHERE seat_id = %s", (seat_id,)
                )
                row = cur.fetchone()
                if row is None:
                    return HoldOutcome(ok=False, reason="no_such_seat")
                status, version = row
                if status != "FREE":
                    return HoldOutcome(ok=False, reason="taken")

                cur.execute(
                    """
                    UPDATE seats
                       SET status = 'HELD', held_by = %s, hold_id = %s,
                           hold_expires_at = now() + make_interval(secs => %s),
                           version = version + 1
                     WHERE seat_id = %s AND version = %s
                    """,
                    (user_id, hold_id, ttl_seconds, seat_id, version),
                )
                if cur.rowcount == 1:
                    conn.commit()
                    return HoldOutcome(ok=True, reason="held")
                conn.rollback()          # someone else won; loop and re-read

        return HoldOutcome(ok=False, reason="contention")   # bounded: never infinite
```

| Buys | Costs |
|---|---|
| No locks held; readers never block | Must retry, so the caller needs a loop and a bound |
| Excellent at low conflict; near-lock-free | Under high conflict the retries thrash — livelock in disguise |
| Simple to add to an existing table (one column) | Wasteful work: a losing attempt did the whole read for nothing |
| No lock-wait timeout, no deadlock from this path | Wrong for multi-row operations where all rows must be free |

**The bound is mandatory.** `MAX_ATTEMPTS = 3` plus a fallback is the difference between an optimistic design and a livelock.

### 4.3 Atomic compare-and-swap, with no read at all

**Idea:** collapse read-decide-write into a single statement. The database's row-level atomicity *is* the concurrency control, and no lock is ever held by the application.

The `hold_seat_cas` function in §3.3 is this design. Its properties are worth stating separately because it is the usual best answer for seat locking:

| Buys | Costs |
|---|---|
| One round trip; no lock held, no lock wait, no deadlock | All-or-nothing: you cannot CAS-hold five seats in one statement without a transaction |
| The loser gets an immediate, unambiguous `rowcount == 0` | You lose the ability to explain *why* it failed without a follow-up read |
| Scales to arbitrary contention — nobody queues | Requires a well-chosen `WHERE` clause; a wrong predicate silently allows double-booking |
| Works identically in a database or in Redis (`SET NX`) | Partial-failure handling is on you: 3 of 5 seats held needs a compensating release |

**The Redis equivalent**, for the hot-path hold that must not touch the database:

```python
def hold_seats_redis(redis: "Redis", show_id: str, seats: list[str], user_id: str,
                     hold_id: str, ttl_ms: int) -> list[str]:
    """Lua runs ATOMICALLY on Redis. Either every seat is held or none is —
    which the naive 'loop and SET NX each seat' version cannot promise.

    KEYS = seat keys; ARGV[1] = holder value; ARGV[2] = ttl ms
    """
    script = """
    local holder = ARGV[1]
    local ttl = tonumber(ARGV[2])
    for i = 1, #KEYS do
      local current = redis.call('GET', KEYS[i])
      if current and current ~= holder then
        return {0, i}                    -- conflict; report WHICH seat (1-based index)
      end
    end
    for i = 1, #KEYS do
      redis.call('SET', KEYS[i], holder, 'PX', ttl)
    end
    return {1, 0}
    """
    keys = [f"seat:{show_id}:{seat}" for seat in seats]
    ok, index = redis.eval(script, len(keys), *keys, hold_id, ttl_ms)
    if ok == 1:
        return []
    return [seats[index - 1]]            # the conflicting seat, for the UI
```

**Why this shape matters:** the check-all loop and the set-all loop are inside one Lua script, so no other client can interleave between them. Doing the same thing with a Python loop of `SET NX` calls would hold four of five seats and then fail on the fifth — the partial-hold bug, which is worse than a plain failure.

### 4.4 Where the atomicity lives: the decision table

| Design | Atomicity provided by | Blocks? | Fails how | Best at |
|---|---|---|---|---|
| **Pessimistic** (`FOR UPDATE`) | A row lock in the database | Yes | Lock-wait timeout | Multi-row, low contention, correctness-critical |
| **Optimistic** (version column) | A conditional update after a read | No | Retries, bounded | Low conflict, simple tables |
| **CAS** (conditional update, no read) | The store's single-statement atomicity | No | `rowcount == 0`, immediate | **High contention on a single row** — seat locking |
| **Redis Lua / `SET NX`** | Redis's single-threaded execution | No | Immediate conflict report | Hot path, must not touch the DB, TTL built in |
| **Unique constraint** | The database index | No | Constraint violation | "Only one of these may exist" — one hold per seat |
| **Single-writer actor** | One thread owns the state | Yes (a queue) | Queue depth / timeout | Per-show serialisation, simple mental model |
| **Distributed lock** (Redlock, etc.) | A lease on an external store | Yes | Lease expiry, clock skew | Cross-service mutual exclusion — use sparingly |

**The unique-constraint variant deserves a mention because it is under-used.** A partial unique index on `(show_id, seat_id) WHERE status IN ('HELD','BOOKED')` makes double-booking *unrepresentable* rather than merely detected. That is a stronger guarantee than any application-level check, and it is one `CREATE INDEX` away.

### 4.5 Thread pools and executors

A pool is a fixed set of worker threads plus a bounded queue. It exists because thread creation is expensive and unbounded concurrency is a denial-of-service against your own service.

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

class HoldExecutor:
    """A bounded pool for I/O-bound hold attempts. Three parameters matter and
    all three are interview points: size, queue depth, and what happens when
    the queue is full.
    """

    def __init__(self, workers: int = 32, max_queue: int = 512,
                 per_task_timeout: float = 2.0) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="hold",
        )
        self._semaphore = threading.BoundedSemaphore(max_queue)
        self._timeout = per_task_timeout

    def submit(self, fn, *args, **kwargs):
        # BOUNDED ADMISSION. Without this, an unbounded queue absorbs the
        # overload, latency grows without limit, and every request times out
        # together — the classic queue-collapse failure.
        if not self._semaphore.acquire(blocking=False):
            raise OverloadedError("hold queue is full; shed this request")
        try:
            future = self._pool.submit(fn, *args, **kwargs)
        except BaseException:
            self._semaphore.release()
            raise

        def _release(_f):
            self._semaphore.release()

        future.add_done_callback(_release)
        return future

    def run_with_timeout(self, fn, *args, **kwargs):
        future = self.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeout:
            # The task is still running and CANNOT be cancelled safely. Cancelling
            # an in-flight hold does not undo a write that already happened, so the
            # caller must treat the outcome as UNKNOWN and reconcile by idempotency key.
            future.cancel()
            raise HoldOutcomeUnknown(
                "hold timed out; reconcile by idempotency key before retrying"
            )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)
```

**The four sizing rules:**

| Parameter | Rule | Why |
|---|---|---|
| **Pool size, I/O-bound** | 2–4× the number of cores | Threads block on I/O; more threads means more in-flight work |
| **Pool size, CPU-bound** | ≈ the number of cores | More threads than cores adds context-switching and buys nothing |
| **Queue depth** | Bounded, and small | An unbounded queue converts an overload into a latency collapse |
| **Full-queue policy** | Shed (reject fast) or block briefly | Absorbing is the worst option: it fails everyone slowly |

**And the async pitfall, which is the same failure wearing different clothes:**

```python
async def hold_seat_async_BROKEN(store, seat_id, user_id) -> bool:
    """BLOCKING CALL INSIDE ASYNC. time.sleep and any synchronous database driver
    block the EVENT LOOP THREAD, so every other coroutine on this loop stalls.
    The service looks hung while one request does a slow query."""
    time.sleep(0.05)                          # ← stalls the entire event loop
    return store.hold_sync(seat_id, user_id)  # ← synchronous driver, same problem


async def hold_seat_async_FIXED(store, seat_id, user_id) -> bool:
    """Two correct fixes, and knowing which to use is the point:
    (a) use an async driver and await it, or
    (b) offload the blocking call to a thread pool."""
    await asyncio.sleep(0.05)                          # (a) yields the loop
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(                       # (b) offload
        None, store.hold_sync, seat_id, user_id
    )
```

**The rule:** in async code, *anything that blocks is a bug*. `time.sleep`, `requests`, a synchronous database driver, a synchronous file read, a CPU-heavy loop. The event loop has one thread; blocking it blocks everyone.

---

## 5. Pros / Cons

### 5.1 Thread-safety strategies

| Strategy | Pros | Cons | Use when |
|---|---|---|---|
| **Immutability** | No locks; safe to share and cache; easy to reason about; safe to publish | Allocation cost; updates become replacements; not usable for a hot counter | Value objects, money, config, booking records, events |
| **Confinement** | No locks; simple mental model; trivially correct | Throughput capped at one owner; a queue hop adds latency; failure needs recovery | Per-key serialisation, single-writer consumers |
| **Stateless** | Free; no reasoning required; scales horizontally | Only applies where there is no state — cannot be forced | Calculators, validators, formatters, HTTP handlers |
| **Synchronisation** | Works for genuinely shared mutable state; familiar | Contention; deadlock; held-across-I/O; visibility subtleties | Only when the first three are impossible |

### 5.2 Locking designs

| Design | Pros | Cons | Pick it when |
|---|---|---|---|
| **Pessimistic** | Simple; the block is serialised; good for multi-row | Lock held across the transaction; waits; deadlock risk; pool exhaustion | Multi-row operations, low contention |
| **Optimistic** | No locks; readers never block; cheap to add | Retries; wasted work; livelock under high conflict | Low conflict, single-row, user-initiated |
| **CAS / conditional update** | No locks; no retries; immediate failure; scales to any contention | One statement per resource; partial failure is yours; needs a correct predicate | **Hot single-row contention — seats** |
| **Unique constraint** | Makes the bad state unrepresentable | Coarse; error handling via exceptions | "Only one may exist" invariants |
| **Single-writer actor** | Trivially correct; natural fit for per-key ordering | One owner's throughput; queue is a new failure mode | Per-show or per-account serialisation |
| **Distributed lock** | Works across services | Lease expiry, clock skew, split brain; slow | Genuinely cross-service mutual exclusion |

### 5.3 What the choice costs, stated plainly

| You choose | You pay |
|---|---|
| A global lock | Throughput — one operation at a time across all keys |
| A row lock | Latency for the loser, plus a held connection, plus deadlock risk |
| Optimistic retry | Wasted work, plus a retry loop, plus a bound you must choose |
| A CAS | The inability to explain failure in the same round trip, and manual partial-failure handling |
| A distributed lock | A dependency on an external system, and a lease-expiry correctness cliff |
| An actor | A throughput ceiling of one core per key, plus a queue to operate |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 The classic races, and how each is fixed

| Race | Shape | Fix |
|---|---|---|
| **Check-then-act** | Read status, check free, write held | CAS, or a lock around the check and the write |
| **Read-modify-write** | `counter += 1` | Atomic increment, or a lock |
| **Lost update** | Two writers overwrite each other | Optimistic version column, or a lock |
| **Write skew** | Two transactions each read a consistent state and write a state that together violate an invariant | Serializable isolation, or materialise the conflict (a constraint, a lock on a shared row) |
| **Phantom** | A new row appears between a check and an act | Range lock, or a unique constraint |
| **Partial hold** | Four of five seats held, then failure on the fifth | All-or-nothing via Lua, or a transaction, or a compensating release |
| **Double submit** | The user clicks twice, or a retry fires | Idempotency key with a unique constraint |

**Write skew is worth a sentence of its own**, because it is the one candidates miss. In a hotel: two bookings each check "are there at least 2 rooms free?", each sees 2, each books 1, and the property is overbooked — yet *no individual write conflicts*, so optimistic locking does not catch it and neither does a per-row lock. The fixes are: lock a shared row (the property's booking counter row), or a constraint that counts, or serialisable isolation. **Naming write skew in an interview is a strong signal**, because it shows you know that row-level conflict detection is not sufficient for invariant-level correctness.

```python
def book_room_write_skew_SAFE(conn: "Connection", property_id: str,
                              room_id: str, guest_id: str) -> bool:
    """Fix for write skew: serialise on a SHARED row — the property — so two
    bookings for the same property cannot both read-then-write concurrently."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT overbooking_limit FROM properties WHERE property_id = %s FOR UPDATE",
            (property_id,),
        )
        limit = cur.fetchone()[0]
        cur.execute(
            "SELECT count(*) FROM reservations WHERE property_id = %s "
            "AND status IN ('CONFIRMED','PENDING')",
            (property_id,),
        )
        confirmed = cur.fetchone()[0]
        if confirmed + 1 > limit:
            return False
        cur.execute(
            "INSERT INTO reservations (property_id, room_id, guest_id, status) "
            "VALUES (%s, %s, %s, 'PENDING')",
            (property_id, room_id, guest_id),
        )
        return True
```

### 6.2 Deadlock

Four conditions must all hold (Coffman): mutual exclusion, hold-and-wait, no preemption, circular wait. **Break any one and deadlock is impossible** — and the practical fix is almost always the fourth.

```python
def lock_seats_in_order_BROKEN(conn, seats: list[str]) -> None:
    """BROKEN: locks in the order given. Booking A locks A12 then A13; booking B
    locks A13 then A12. Each holds one and waits for the other. Deadlock."""
    with conn.cursor() as cur:
        for seat in seats:
            cur.execute("SELECT * FROM seats WHERE seat_id = %s FOR UPDATE", (seat,))


def lock_seats_in_order_SAFE(conn, seats: list[str]) -> None:
    """SAFE: a total order on the lock keys. Both transactions now want A12
    first, so one waits for the other instead of circling."""
    with conn.cursor() as cur:
        for seat in sorted(set(seats)):          # ← the entire fix
            cur.execute("SELECT * FROM seats WHERE seat_id = %s FOR UPDATE", (seat,))
```

| Condition | How to break it | Cost |
|---|---|---|
| **Mutual exclusion** | Make the resource immutable or shared-read | Often impossible |
| **Hold-and-wait** | Acquire everything at once, or nothing | Hard with multi-step flows |
| **No preemption** | Use lock timeouts, so a holder is forced to release | A timeout can leave partial state |
| **Circular wait** | **A global lock order** | Free, and almost always the right fix |

**Also worth saying:** a lock timeout is not a fix, it is a mitigation. It converts an infinite hang into an error, and the error must be handled — usually by rolling back and retrying idempotently.

### 6.3 Livelock and starvation

| Failure | What it looks like | Fix |
|---|---|---|
| **Livelock** | Requests keep retrying and keep losing; CPU is busy, throughput is zero | Bounded retries plus exponential backoff plus jitter; then fall back to an atomic update or a queue |
| **Starvation** | One class of request never wins | Fair queuing, or an ageing priority |
| **Priority inversion** | A high-priority task waits on a low-priority one holding a lock | Priority inheritance, or avoid priorities inside critical sections |
| **Thundering herd** | A popular key's lock is contended by thousands at once | Backoff with jitter; a queue; a request-coalescing cache |

```python
import random
import time


def retry_with_jitter(fn, attempts: int = 5, base: float = 0.01) -> object:
    """Exponential backoff WITH JITTER. Without the jitter, every loser retries
    at the same instant and they collide again — synchronised retries turn a
    brief conflict into a sustained one."""
    for attempt in range(attempts):
        try:
            return fn()
        except TransientConflict:
            if attempt == attempts - 1:
                raise
            delay = base * (2 ** attempt)
            time.sleep(delay * (0.5 + random.random()))     # jitter: ±50 %
    raise AssertionError("unreachable")
```

### 6.4 Stranded holds and crash recovery

**The failure nobody raises unprompted, and the one that matters most in production:** a process holds a seat and dies. The lock is gone but the seat's status is `HELD` forever.

| Mechanism | How it works | Cost |
|---|---|---|
| **TTL on the hold** | The hold expires and the seat returns to `FREE` | The honest holder must confirm before the TTL |
| **Lazy expiry on read** | Any read that sees an expired hold treats the seat as free | Correctness only; the expired row lingers |
| **Proactive sweeper** | A background job returns expired holds to `FREE` | Extra work; needs to be idempotent and safe to run concurrently |
| **Heartbeat / lease renewal** | The holder renews while alive; no renewal means dead | More traffic; a slow holder loses its own hold |

**Both lazy and proactive expiry are needed, and the reason is worth stating:** lazy expiry makes the *answer correct* (nobody is told a seat is taken when its hold has expired); the sweeper makes the *state tidy* (the rows and Redis keys do not accumulate). One is correctness, the other is hygiene — and if you only have the sweeper, there is a window where reads are wrong; if you only have lazy expiry, expired holds pile up forever.

### 6.5 The GIL, precisely

| Claim | True? | Reality |
|---|---|---|
| "The GIL makes Python thread-safe" | **No** | It makes individual bytecodes non-torn. `+=` is three bytecodes and still races |
| "The GIL makes threads useless" | **No** | I/O-bound threads work well: the GIL is released during I/O |
| "The GIL makes threads useless for CPU work" | **Mostly yes** | CPU-bound work needs processes (`multiprocessing`) or native extensions that release the GIL |
| "The GIL will be removed and this stops mattering" | **Partly** | Free-threaded builds exist, which makes *correct* design more important, not less — the accidental protection disappears |
| "`asyncio` avoids the GIL" | **No** | `asyncio` is one thread on one core. It avoids *blocking*, not the GIL |

**The interview-safe formulation:** *"In Python, `threading` is right for I/O-bound work and `multiprocessing` for CPU-bound work. But the LLD answer should be language-independent: the shared mutable state is the problem, and the fix is immutability, confinement, or an atomic store operation — regardless of which runtime we are in."*

### 6.6 Testing concurrency

Concurrency bugs do not reproduce on demand, which is why the test strategy is a design decision, not an afterthought.

| Technique | What it catches | Cost |
|---|---|---|
| **Deterministic single-threaded tests** | Logic, state transitions, invariants | Cannot catch a race at all |
| **A `Clock` abstraction** | TTL and expiry logic, exactly and instantly | Requires injecting time everywhere — worth it |
| **Forced interleaving** (a barrier or a hook between the read and the write) | The exact race, reliably | Test-only seam; must not exist in production code paths |
| **Stress test** (N threads × M iterations, assert the invariant) | Races probabilistically | Nondeterministic; slow; can pass by luck |
| **Invariant assertion after every operation** | Double-booking, overbooking, negative counts | Needs a cheap, checkable invariant |

```python
class Clock:
    """Inject time. Every TTL, expiry and lease in the design reads from here,
    so tests advance time instead of sleeping."""

    def now(self) -> float:
        raise NotImplementedError


class RealClock(Clock):
    def now(self) -> float:
        return time.time()


class FakeClock(Clock):
    """Tests call advance(600) and the hold expires instantly and exactly."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time does not run backwards")
        self._now += seconds


def test_hold_expires_exactly_at_ttl() -> None:
    clock = FakeClock()
    store = SeatStore(clock=clock, hold_ttl=600)
    assert store.hold("A12", "user-1") is True

    clock.advance(599.999)
    assert store.is_available("A12") is False      # still held

    clock.advance(0.001)                            # exactly at the boundary
    assert store.is_available("A12") is True        # lazy expiry made it correct

    # And no sleep() anywhere: the test runs in microseconds, deterministically.


def test_two_threads_cannot_both_hold_the_same_seat() -> None:
    """The stress test. It asserts the INVARIANT, and it is allowed to be
    probabilistic — it is a backstop, not the primary evidence."""
    store = SeatStore(clock=RealClock(), hold_ttl=600)
    winners: list[int] = []
    barrier = threading.Barrier(2)

    def attempt(i: int) -> None:
        barrier.wait()                     # maximise the overlap: both start together
        if store.hold("A12", f"user-{i}"):
            winners.append(i)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1, f"double booking: winners={winners}"
```

**The `Clock` abstraction is the single highest-value testing decision in this file.** It converts every TTL, expiry, hold-timeout and lease test from a slow, flaky `sleep()` into an exact, instant assertion. Mention it in the interview.

### 6.7 The ten failure modes, in one table

| # | Failure | Symptom | Fix |
|---|---|---|---|
| 1 | Check-then-act race | Double booking | CAS or lock |
| 2 | Lost update | Wrong counter | Atomic increment or version |
| 3 | Write skew | Invariant violated with no row conflict | Lock a shared row; serialisable isolation |
| 4 | Deadlock | Requests hang | Global lock order; timeouts |
| 5 | Livelock | Retries thrash, no progress | Bounded retries; backoff with jitter |
| 6 | Starvation | One class never wins | Fair queue; ageing |
| 7 | Lock held across I/O | Throughput collapse | Shrink the critical section to one store op |
| 8 | Unbounded pool queue | Latency collapse under overload | Bounded queue; shed on full |
| 9 | Stranded hold after a crash | Seat unusable forever | TTL plus lazy expiry plus sweeper |
| 10 | Blocking inside `async` | Whole event loop stalls | Async driver, or offload to a thread |

---

## 7. When to use / When NOT to use

### 7.1 When each mechanism is right

| Situation | Use | Why |
|---|---|---|
| Value object, config, event, money | **Immutability** | Nothing to lock, safe to cache and share |
| A calculator, validator, formatter, HTTP handler | **Statelessness** | Free correctness; scales horizontally |
| Per-show or per-account ordering matters | **Confinement** (an actor, or a per-key lock) | One owner gives free ordering |
| Multi-row invariant (all seats in a booking must be free) | **Pessimistic lock in a fixed order**, or a transaction | Row-level all-or-nothing |
| Single hot row, very high contention | **CAS / conditional update** | No blocking, no retries, immediate answer |
| Low conflict, user-initiated edit | **Optimistic version** | No locks, no blocking, simple |
| "Only one of these may exist" | **Unique constraint** | Unrepresentable, not merely detected |
| Hot path that must not touch the database | **Redis Lua** | Atomic, TTL built in, no DB load |
| Genuinely cross-service mutual exclusion | **Distributed lock**, with a lease | The only option — and use it sparingly |

### 7.2 When NOT to lock at all

| Situation | Why not | Do instead |
|---|---|---|
| The class has no fields | There is no shared state | Nothing — say "this class is stateless, so it is safe" |
| Every request gets its own object | Not shared | Nothing |
| The state is only written by one thread | Not shared | Confinement, and document the ownership |
| The "shared" state is a read-only snapshot | Not mutable | Immutability |
| The lock would be held across a network call | Throughput collapse | Move the atomicity into the store |
| Two locks, in an order you have not chosen | Deadlock | A global order, or one lock, or CAS |
| A lock around a whole request handler | Serialises the entire service | Narrow the critical section to the state that actually races |
| You are locking to "be safe" | Cost with no named problem | Name the interleaving first, or do not lock |

### 7.3 The three-question test before adding any lock

1. **What two operations interleave badly?** Write the timeline. If you cannot, there is no race and the lock is theatre.
2. **Can I remove the sharing instead?** Immutability, confinement or statelessness beats every lock.
3. **How hot is this key?** Low contention → pessimistic is fine. High contention → CAS or a queue. And what is the cost I am accepting?

---

## 8. Numbers, defaults & tuning

| Parameter | Default | Why |
|---|---|---|
| **Hold TTL (seat, show)** | **8–10 minutes** | Long enough to pay; short enough to reclaim. Below ~5 min users lose seats mid-payment |
| **Hold TTL (hotel room)** | 15–30 minutes | More form fields, more deliberation |
| **Lock wait timeout** | 2–5 s | Below 1 s produces spurious failures; above 10 s the user has already given up |
| **Optimistic retry attempts** | **3** | Beyond 3, the conflict rate means you picked the wrong design |
| **Backoff base** | 10 ms, doubling | Fast first retry, then back off quickly |
| **Backoff jitter** | ±50 % | Prevents synchronised retries |
| **Stripe count (in-process)** | 64–256 | Power of two; the goal is that unrelated keys nearly never collide |
| **Pool size, I/O-bound** | 2–4× cores | Threads block on I/O |
| **Pool size, CPU-bound** | ≈ cores | More threads buy context-switching, not throughput |
| **Pool queue depth** | Bounded, ~2–4× pool size | An unbounded queue converts overload into latency collapse |
| **Per-task timeout** | 2 s for a hold | Must be shorter than the user-facing request timeout |
| **Sweeper interval** | 1–5 minutes, or 10 % of the TTL | Cheap; also protects against a stalled sweeper |
| **Sweeper batch size** | 500–1000 rows | Bounded work per tick; avoids a long-held lock |
| **Idempotency key retention** | ≥ the hold TTL, plus 24 h for payments | A retry outside the window must not double-charge |
| **Conflict-rate thresholds** | <1 % any design · 1–10 % optimistic · >10 % CAS or queue | The rule that picks the design |
| **Circuit-breaker threshold on the store** | 5 failures in 10 s → open | A locked-up store must not take down the API |
| **Read-replica staleness budget** | 100 ms–1 s | Availability *reads* may be stale; holds may not — see §6.7 |

---

## 9. Interview Q&A

**Q1 (easy). What is a race condition?**
Any situation where the correctness of the result depends on the interleaving of concurrent operations. The two shapes are check-then-act and read-modify-write. In the seat design it is: two users both read the seat as FREE, both decide to proceed, both write HELD, and one of them is wrong. The fix always involves making read-decide-write atomic, or removing the sharing entirely.

**Q2 (easy). What does atomic mean here?**
That the operation is indivisible from every other observer's point of view — nobody can see it half-done. In practice, atomicity comes from a small set of primitives: a mutex, a compare-and-swap, a database transaction, a single conditional `UPDATE`, or a Lua script in Redis. The design question is which primitive you are relying on and whether it really covers the whole critical section.

**Q3 (easy). What is the difference between a mutex and a semaphore?**
A mutex allows exactly one holder — it is mutual exclusion. A semaphore allows up to N holders, so it is a concurrency limiter as well as a gate. In an LLD answer I mostly want a mutex, plus a bounded semaphore when I need to cap in-flight work, as in the thread-pool admission control.

**Q4 (medium). How do you make a class thread-safe?**
I walk four strategies in order. Immutability — make the object frozen so there is nothing to race on. Confinement — the state is owned by one thread, so no lock is needed. Statelessness — no fields, therefore no race, which is free. And only then synchronisation, where I keep the critical section as small as possible, ideally one store operation, and never hold a lock across I/O. Most interview answers jump straight to the fourth, and the first three are usually better.

**Q5 (medium). Optimistic versus pessimistic locking — how do you choose?**
By the conflict rate, not by preference. Pessimistic locks the row before reading, so the loser blocks — simple and correct, but the lock is held across the whole transaction, it can deadlock, and it consumes a connection while waiting. Optimistic reads freely and fails the write if the row changed under a version column — no blocking, but the loser repeats work, so it needs a bounded retry loop. Below about 1 % conflict either is fine and I would take the simpler pessimistic one. Between 1 % and 10 % optimistic wins. Above 10 % — a hot show — I want neither, because pessimistic locks queue and time out while optimistic retries thrash. I want an atomic conditional update, which never blocks and never retries.

**Q6 (medium). Explain compare-and-swap in this design.**
It collapses read-decide-write into a single statement whose `WHERE` clause is the decision: `UPDATE seats SET status='HELD' WHERE seat_id=%s AND status='FREE'`, and `rowcount == 1` means I won. There is no lock held by the application, no lock wait, no deadlock from this path, and the loser gets an immediate unambiguous answer. The costs are that it is one statement per resource, so a five-seat booking needs either a transaction or a compensating release for partial failure, and that I lose the ability to explain *why* it failed without a follow-up read.

**Q7 (medium). What is deadlock, and how do you prevent it?**
Two actors each hold a resource the other needs, so neither progresses. Four conditions must all hold — mutual exclusion, hold-and-wait, no preemption, circular wait — and breaking any one prevents it. The practical fix is almost always the fourth: impose a global order on the lock keys. In the seat design, if every booking sorts its seats and locks them in `seat_id` order, two overlapping bookings queue rather than circle. That one line is the fix. Lock timeouts are a mitigation, not a fix: they turn a hang into an error, and the error still has to be handled.

**Q8 (medium). What is an idempotency key, and why does it matter here?**
A client-supplied identifier that makes a retried request have the same effect as the original. It matters because a retry is unavoidable: a timeout does not mean the operation did not happen — it means the outcome is unknown. If the client retries a hold or a charge without a key, it can hold two seats or charge twice. With a key, the store records it under a unique constraint, and a second request with the same key returns the original result instead of doing the work again. The key has to be stored for at least as long as a retry is possible; for payments I keep it for 24 hours.

**Q9 (hard). *Pushback:* "Why not just put one big lock around the whole booking flow? It is obviously correct."**
It is correct, and I would say so — the problem is what it costs. It serialises every booking in the entire venue on one mutex, and worse, the booking flow includes a payment call, so the lock is held across a network round trip for hundreds of milliseconds. That caps the whole system at a handful of bookings per second, no matter how many cores or pods are idle. It also does not survive the multi-instance case at all: two pods have two locks and will happily double-book. So I use it as the correctness baseline I reason against, and then move the atomicity down to the store, where one conditional update gives the same guarantee without the contention.

**Q10 (hard). *Pushback:* "Your Redis hold uses SET NX per seat in a loop. Isn't that racy?"**
Yes, as stated, and that is exactly why the design uses a Lua script instead. A Python loop of `SET NX` calls is not atomic as a group: it can hold four of five seats and then fail on the fifth, which is worse than failing outright because it leaves seats stranded and a partial hold to clean up. Redis executes a Lua script atomically on its single thread, so the check-all loop and the set-all loop happen with no other client interleaved. The script returns which seat conflicted so the UI can explain it. And I would still add a TTL to every key, so a crashed holder cannot strand a seat.

**Q11 (hard). *Pushback:* "If you use a TTL, a slow user loses their seat mid-payment. Isn't that a correctness bug?"**
It is a real trade-off and I would state it as one rather than pretend it away. The TTL bounds how long a seat can be held by someone who is not going to buy it; without it, one user who abandons the checkout page blocks that seat forever, which is worse for everyone. The mitigations are: make the TTL comfortably longer than the median checkout — eight to ten minutes for a seat, fifteen to thirty for a hotel room; extend the hold while the user is actively working in the checkout, which is a heartbeat rather than a new hold; and make the confirm step check expiry *and* fail cleanly with a clear message, so the user is told "your hold expired, the seat is gone" rather than being charged for nothing. That last one is the part that absolutely must not be wrong: never take money for a seat you cannot give.

**Q12 (hard). *Pushback:* "You said the read path can be stale. Why is that acceptable?"**
Because I am separating two different guarantees. Availability reads — how many seats are left, showtimes, prices — may be served from a cache or a replica that is up to a second stale, because the consequence of being wrong is a user seeing a seat that is taken and then being told so at hold time. Holds may not be stale, because the consequence is a double booking. So the staleness budget is a per-operation decision, not a system-wide one, and I would say the number out loud: about a second for the seat map, zero for the hold. That is the same reasoning as the BookMyShow case study's "the display can be stale; the hold cannot".

**Q13 (medium). How do you test concurrency?**
Not with `sleep`. First, extract time behind a `Clock` interface so TTL and expiry tests advance a fake clock and assert exactly at the boundary — deterministic and instant. Second, force the interleaving with a barrier so both threads reach the critical section together, which turns a probabilistic race into a reliable failure. Third, a stress test with N threads asserting the invariant — that it never happens that two users hold the same seat — which is a backstop rather than primary evidence, because it can pass by luck. Fourth, and most valuable, assert the invariant cheaply after every operation, so a violation is caught at the point it occurs rather than three steps later.

**Q14 (medium). What is the difference between concurrency and parallelism?**
Concurrency is about structure — multiple tasks making progress, possibly on one core, interleaved. Parallelism is about execution — multiple tasks literally running at the same instant on different cores. The distinction matters because they have different failure modes: concurrency introduces races and deadlocks even on a single core, which is why `asyncio` is not safe by default, and parallelism introduces contention on shared hardware, which is why a lock can be slower than the work it protects. Both are present in this system, and neither is fixed by adding threads.

**Q15 (hard). Where does concurrency live in the parking lot design?**
Three places, and the interesting thing is that only one of them is genuinely hard. First, the spot allocation: `find_available` then `reserve` is a check-then-act race, and the fix is that the reserve step must be an atomic conditional update on the spot's status — the find step is only a hint. Second, the occupancy counters used by the gate display: these are read-modify-write, but a slightly wrong counter only means a display is briefly off, and it self-heals, so I would use an atomic increment and accept eventual consistency there rather than locking. Third, the ticket sequence number, which must be unique — that is a database sequence or an atomic counter, not a lock. Naming which of the three actually needs strong consistency, and which can be eventually consistent, is the point of the question.

**Q16 (medium). When would you not lock at all?**
Whenever the state is not both shared and mutable. If the class is stateless — a pricing calculator, a validator — there is nothing to lock, and adding a lock would be worse than useless: it costs contention and signals that I did not identify the real shared state. If the state is immutable, same answer. If the state is confined to one thread, same answer, and I would document the ownership rather than add a lock "to be safe". The only genuinely shared mutable state in the seat design is the seat's status in the store, so that is where all the concurrency control belongs.

---

## 10. Common mistakes candidates make

1. **Naming a lock without naming the interleaving.** The lock is the conclusion; the timeline is the reasoning.
2. **Holding a lock across I/O.** The single biggest throughput mistake. Keep the critical section to one store operation.
3. **Adding a lock to a stateless class.** Cost with no benefit, and it hides that the real shared state was never found.
4. **One global lock for all keys.** Seat A12 and B07 do not conflict; striping or CAS fixes it.
5. **Assuming the GIL makes Python thread-safe.** `+=` is three bytecodes and still races.
6. **Blocking inside `async`.** `time.sleep`, `requests`, a synchronous driver — the whole event loop stalls.
7. **Unbounded retries under contention.** Livelock. Bound them, back off, add jitter, then fall back.
8. **No idempotency key.** A retry after a timeout double-books and double-charges.
9. **Forgetting the crash case.** A held seat with no TTL is stranded forever.
10. **Deadlock via multi-row locking in an arbitrary order.** Sort the keys.
11. **An unbounded thread-pool queue.** Converts overload into a latency collapse.
12. **Treating a lock timeout as a fix.** It converts a hang into an error that still needs handling.
13. **Ignoring partial failure in a multi-seat hold.** Four of five seats held is worse than none.
14. **Testing with `sleep`.** Slow, flaky, and it still does not prove the race is absent. Use a `Clock`.
15. **Confusing in-process locks with distributed correctness.** Two pods have two locks.
16. **Not knowing which reads may be stale.** Availability reads yes; holds no — and say the number.
17. **Assuming optimistic locking catches write skew.** It does not: no row conflicts, yet the invariant breaks.
18. **Never saying what the lock costs.** The cost is what makes the choice a design rather than a reflex.

---

## 11. Cheat-sheet summary

**The one-question test**

> **What two operations interleave badly, and what is the atomic step that closes the window?**

**Thread-safety strategies, in preference order**

| # | Strategy | Cost | When |
|---|---|---|---|
| 1 | Immutability | Allocation | Value objects, money, config, events |
| 2 | Confinement | One owner's throughput | Per-key ordering |
| 3 | Statelessness | Free | Calculators, validators, handlers |
| 4 | Synchronisation | Contention, deadlock | Only when 1–3 are impossible |

**Locking designs**

| Design | Atomicity from | Blocks | Fails | Best at |
|---|---|---|---|---|
| Pessimistic | Row lock (`FOR UPDATE`) | Yes | Timeout | Multi-row, low contention |
| Optimistic | Version column | No | Bounded retries | Low conflict, single row |
| **CAS** | One conditional `UPDATE` | No | `rowcount == 0` | **Hot single-row — seats** |
| Redis Lua | Redis single-threaded | No | Immediate conflict | Hot path, TTL built in |
| Unique constraint | Index | No | Exception | "Only one may exist" |
| Actor | One thread owns it | Queue | Queue depth | Per-show serialisation |
| Distributed lock | External lease | Yes | Lease expiry | Cross-service, sparingly |

**Failure modes and fixes**

| Failure | Fix |
|---|---|
| Check-then-act | CAS, or a lock narrowed to the state |
| Lost update | Atomic increment, or a version column |
| Write skew | Lock a *shared* row; serialisable isolation |
| Deadlock | Global lock order (sort the keys); timeouts as mitigation |
| Livelock | Bounded retries, exponential backoff, jitter |
| Starvation | Fair queue, ageing |
| Held across I/O | Shrink the critical section |
| Unbounded queue | Bound it; shed when full |
| Stranded hold | TTL + lazy expiry + sweeper |
| Blocking in `async` | Async driver, or offload to a thread |

**Numbers**

| Parameter | Value |
|---|---|
| Seat hold TTL | 8–10 min |
| Hotel room hold TTL | 15–30 min |
| Lock wait timeout | 2–5 s |
| Optimistic retries | 3 |
| Backoff | 10 ms base, ×2, ±50 % jitter |
| Stripes | 64–256 |
| Pool, I/O-bound | 2–4× cores |
| Pool, CPU-bound | ≈ cores |
| Sweeper interval | 1–5 min |
| Idempotency retention | ≥ TTL; 24 h for payments |
| Conflict rate | <1 % any · 1–10 % optimistic · >10 % CAS/queue |

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "Two users book the same seat. What happens?" | "Check-then-act race. Both read FREE, both write HELD. The fix is a single conditional update — `WHERE status='FREE'` — and `rowcount` decides the winner." |
| "Just add a lock." | "A row lock works but blocks; on a hot show it queues and times out. CAS never blocks and never retries — I'd rather have that." |
| "What if the user never pays?" | "TTL on the hold, lazy expiry on read for correctness, plus a sweeper for hygiene." |
| "What if the request times out?" | "The outcome is unknown, not failed. Idempotency key, so a retry returns the original result rather than double-booking." |
| "Multiple seats in one booking?" | "Pessimistic in a fixed key order, or a transaction, or an all-or-nothing Lua script. Never a loop of independent `SET NX` — that leaves partial holds." |
| "Does this work across pods?" | "Not with in-process locks — two pods have two locks. The atomicity has to live in the store." |
| "How do you test it?" | "A `Clock` for expiry, a barrier to force the interleaving, and an invariant assertion after every operation. No `sleep`." |
| "What does this cost you?" | "Throughput on the contended key. CAS costs one round trip and no queueing; a global lock costs everything." |

---

## 12. Cross-links

- Previous in this pillar: [Design patterns](03-design-patterns.md) · [OOP principles and SOLID](02-oop-principles-and-solid.md) · [What is LLD and how to answer it](01-what-is-lld-and-how-to-answer.md)
- Next: [UML and class diagrams](05-uml-and-class-diagrams.md) · [The machine-coding interview framework](06-machine-coding-interview-framework.md)
- **The bridge this file exists for:** [BookMyShow seat locking](problems/04-bookmyshow-seat-locking.md) — the full worked concurrency design: holds with TTL, the two-users-one-seat race, DB-level versus Redis-level locking, and idempotency keys. The *distributed* treatment lives in [Case-Studies/HLD/07-bookmyshow-seat-booking.md](../Case-Studies/HLD/07-bookmyshow-seat-booking.md), especially its "Concurrency control: the four options" and "Hold expiry: TTL plus sweeper" sections; this file owns the class-level reasoning.
- Concurrency in the other problems: [Parking lot spot allocation](problems/01-parking-lot.md) · [LRU/LFU cache thread-safety and lock striping](problems/02-lru-lfu-cache.md) · [Hotel double-booking race](problems/03-hotel-management.md) · [Elevator concurrency](problems/05-elevator-system.md) · [Splitwise balance races](problems/06-splitwise.md) · [ATM cash dispensing](problems/07-atm-machine.md) · [Library copy reservation](problems/08-library-management.md)
- Distributed locking and atomic primitives: [Redis deep dive](../HLD/03-scaling/03-redis-deep-dive.md) (single-threaded execution, Lua, `SET NX`, TTL) · [Caching strategies and layers](../HLD/03-scaling/02-caching-strategies-and-layers.md) · [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md)
- Reliability and failure handling around locks: [Reliability patterns](../Operations/07-reliability-patterns.md) · [Observability](../Operations/02-observability-and-monitoring.md)
- The boundary these mechanisms live below: [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md)
