# LLD Problem: BookMyShow Seat Locking

> **Grounding note.** The **roadmap transcript (T1)** asks this question and does not answer it. At **[9:15]–[9:18]**, in the syllabus list for the Low-Level Design module that begins at [9:09], it asks *"how is BookMyShow built? how do you do seat locking? how does concurrent booking happen?"* — three questions, no model, no code, no discussion. **[9:15]–[9:18] is the whole of the transcript's contribution** — and note that this is the *roadmap* video naming a topic, not the Day 1 lecture, which never mentions seat booking at all. So [concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md) opens with the same citation for the same reason. **Everything below is standard industry practice.** This file is the one that answers those three questions directly, and it is worth saying why they are three questions rather than one: seat locking is a *mechanism*, concurrent booking is a *race*, and "how is BookMyShow built" is the *model the mechanism sits in* — and a candidate who can only talk about the lock has answered the middle one.
>
> **The boundary with the case study, stated up front.** [BookMyShow seat booking](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md) is the **system** treatment: capacity estimation, the API surface, the high-level design, the four concurrency options weighed as architecture, the waiting room, seat-map cache staleness, Redis failover, and the numbers for a hot show. It even contains two "LLD half" digressions (an LRU cache and an LFU cache). **This file is the class-and-code treatment**: the seat model, the state machine, the hold store interface and its two implementations, the exact interleaving at statement level, idempotency keys, and the sweeper. Read that one for *where the lock lives in a fleet*; read this one for *what the lock is around*.

---

## 1. Requirement clarification

### 1.1 Questions to ask

| # | Question | Why it matters | Default if they will not say |
|---|---|---|---|
| 1 | What is being booked — a seat, a set of seats, or a whole row? | A set must be held all-or-nothing, which changes the atomicity requirement | A set: book N seats in one transaction |
| 2 | Who picks the seats — the user, or "best available"? | Auto-assign needs a *search* under the lock, which is more work than a check | User picks; auto-assign is an extension (§7.4) |
| 3 | Is a hold required before payment, or can payment be first? | Determines whether TTL and release exist at all | Hold first, then payment |
| 4 | How long is a hold? | The core tuning number, and it trades abandonment against blocking | 8 minutes |
| 5 | What happens when the hold expires mid-payment? | The nastiest real case, and the one that decides the design | Grace: the hold is not stolen until payment resolves (§7.3) |
| 6 | **Is overbooking ever acceptable?** | **For seats it is not.** Decides whether the answer is a count or a per-seat invariant | Never — one seat, one ticket, and a refusal is better than a double sale |
| 7 | Are seats individually priced? | Determines whether a hold has an amount attached | Yes: recliner, premium, balcony tiers |
| 8 | Single show or an event with many shows? | Whether the model is `Show` or `Event → Show` | A `Show`; an event is a `Show` with more fields |
| 9 | Redis, or the database, or both? | The mechanism decision — and the honest answer is both | Both: Redis for the hold's speed, the DB for the durable truth |
| 10 | What is the write rate at the peak? | Whether a single row-lock survives | A hot show: 10 000+ concurrent attempts, so per-show partitioning |

### 1.2 Defaults to state out loud

```
  MODEL:      Venue → Screen → Seat (laid out in rows) ; Movie → Show → ShowSeat
              A ShowSeat is the per-show state of a physical Seat. The Seat is
              immutable geometry; the ShowSeat is mutable availability. That
              split is the model, and it is what lets one seat map serve every
              show (§5.1).

  LOCKING:    a HOLD is a TTL'd lease on a set of ShowSeats, taken before
              payment, keyed by an idempotency key.
  HOLDS TTL:  8 minutes, extended once on request.
  ATOMICITY:  all-or-nothing over the whole seat set. Never partial.
  NO OVERSELL: an invariant, not a policy. One seat, one CONFIRMED booking.
  DUPLICATE:  payment retries must be idempotent — the same key, once.
  OUT:        the catalogue, search, recommendations, the waiting room, the
              seat-map CDN, refunds, dynamic pricing.
```

**The one sentence that frames the problem:** *"Never sell the same seat twice" is a hard invariant, and the entire design exists to hold it while two people click the same seat in the same millisecond.*

---

## 2. Core entities & responsibilities

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `Venue` | A building | name, city | Know about screens |
| `Screen` | A hall within a venue | id, rows, seat layout | Know about shows or prices |
| `Seat` | Immutable physical geometry | row, number, tier | Hold state — that is per show |
| `SeatTier` | A price band | name, surcharge | Know which seats are in it |
| `Movie` | A title | id, name, duration | Know about shows |
| `Show` | One screening | id, screen, start time, price map | Hold seats; the store does that |
| `ShowSeat` | **The per-show state of one seat** | seat ref, `SeatStatus`, held-by, hold expiry | Know about money |
| `SeatStatus` | The lifecycle of a seat for one show | an enum | — |
| `SeatHold` | A lease on a set of seats | key, show, seat ids, expiry, owner | Take payment |
| `HoldStore` *(interface)* | Atomically hold, release and confirm | — | Decide prices or policy |
| `RedisHoldStore` | A fast, TTL'd hold store | Redis keys + a Lua script | Be the durable truth |
| `DbHoldStore` | The durable, transactional store | rows + `SELECT … FOR UPDATE` | Be fast under a hot show |
| `Booking` | A confirmed sale | id, hold key, seats, amount | Hold seats |
| `PaymentGateway` *(interface)* | Charge, and report an outcome | — | Know about seats |
| `BookingService` | Orchestrate hold → pay → confirm | the saga | Implement the stores |
| `Clock` *(interface)* | Supply monotonic time | — | — |

**The single most important modelling decision, stated plainly: `Seat` and `ShowSeat` are different objects.** A `Seat` is geometry — row H, number 12, a recliner — and it never changes. Whether *that seat is taken for the 7 pm show on Friday* is a `ShowSeat`, and it changes constantly. Collapsing them into one mutable `Seat` is the mistake that makes a seat map impossible to cache, because every show would invalidate every seat. **This split is what makes the read path cheap and the write path small**, and it is worth leading with.

---

## 3. Class diagram

```
   ┌───────────────┐      ┌────────────────┐       ┌───────────────┐
   │    Venue      │1    *│    Screen      │1     *│     Seat      │
   ├───────────────┤─────▶├────────────────┤──────▶├───────────────┤
   │ id, name      │      │ id, name       │       │ id            │
   │ city          │      │ rows, seats    │       │ row, number   │
   └───────────────┘      └───────┬────────┘       │ tier          │  ← IMMUTABLE
                                  │ 1              └───────┬───────┘     geometry
                                  │                        │ 1
                                  │ *                      │
                          ┌───────▼────────┐               │
                          │     Show       │1             *│
                          ├────────────────┤───────────────┐│
                          │ id             │               ││
                          │ screen         │      ┌────────▼▼──────────┐
                          │ starts_at      │      │     ShowSeat       │
                          │ base_price     │      ├────────────────────┤
                          └───────┬────────┘      │ show_id            │  ← MUTABLE
                                  │ 1             │ seat_id            │    per-show
                                  │               │ status:SeatStatus  │    state
                                  │ *             │ held_by: str|None  │
                          ┌───────▼───────────────┤ hold_expires_at    │
                          │      SeatHold         │ version: int       │
                          ├───────────────────────┴────────────────────┘
                          │ hold_key: str      ← the IDEMPOTENCY key
                          │ show_id: str
                          │ seat_ids: frozenset[str]
                          │ owner_id: str
                          │ expires_at: float
                          │ state: HoldState
                          └───────────────────────┘
                                    ▲
                                    │ created by / read by
   ┌────────────────────────────────┴──────────────────────────────┐
   │                       «interface» HoldStore                   │
   ├───────────────────────────────────────────────────────────────┤
   │ + acquire(show_id, seat_ids, owner_id, ttl, key) -> SeatHold  │  ← ATOMIC
   │ + confirm(show_id, seat_ids, key) -> Booking                 │
   │ + release(show_id, seat_ids, key) -> bool                    │
   │ + status(show_id, seat_ids) -> dict[str, SeatStatus]         │
   └──────────┬────────────────────────────────────┬───────────────┘
              │ realization                        │ realization
   ┌──────────▼───────────┐            ┌───────────▼──────────────┐
   │   RedisHoldStore     │            │      DbHoldStore         │
   ├──────────────────────┤            ├──────────────────────────┤
   │ Lua: check-all then  │            │ BEGIN; SELECT … FOR      │
   │ set-all, atomically  │            │ UPDATE; UPDATE; COMMIT   │
   └──────────────────────┘            └──────────────────────────┘
```

**What the diagram states:**

- `Seat` (immutable geometry) is separated from `ShowSeat` (mutable per-show state) — the split that makes the read path cacheable.
- `SeatHold` carries a `hold_key`, which **is** the idempotency key. Drawing it shows the retry design is in the model, not bolted on.
- `HoldStore` is an interface with **two realizations**, and the diagram makes the DB-versus-Redis choice a Strategy rather than a fork in the code.
- `SeatStatus` per `ShowSeat`, not per `Seat`.

---

## 4. Interfaces and enums

```python
from abc import ABC, abstractmethod
from enum import Enum


class SeatStatus(Enum):
    """The lifecycle of ONE seat for ONE show. Five states, and each exists
    because collapsing it into a boolean loses a distinction the business
    needs."""

    AVAILABLE = "available"      # sellable
    HELD = "held"                # leased by a hold, expiring
    BOOKED = "booked"            # sold. Terminal for this show.
    BLOCKED = "blocked"          # house seats, a broken seat, a camera position
    UNAVAILABLE = "unavailable"  # not in the layout for this show


class HoldState(Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"      # converted into a Booking
    RELEASED = "released"        # given back, explicitly or by expiry
    EXPIRED = "expired"          # given back by time


class SeatError(RuntimeError):
    """Base class for domain failures."""


class SeatTakenError(SeatError):
    """At least one requested seat is not available. Carries WHICH, because
    the UI must highlight the conflicting seats rather than fail mutely."""

    def __init__(self, conflicts: "list[str]") -> None:
        self.conflicts = conflicts
        super().__init__(f"{len(conflicts)} seat(s) unavailable: {conflicts}")


class HoldNotFoundError(SeatError):
    pass


class HoldExpiredError(SeatError):
    pass


class IdempotencyConflict(SeatError):
    """The same idempotency key was used with DIFFERENT parameters. That is a
    client bug, and silently returning the first result would hide a real
    mistake — a caller reusing a key for a different seat set."""


class PaymentGateway(ABC):
    """Note the three-valued outcome. `charge` does NOT return a bool, because
    'did it fail?' is not the question — 'did it fail, or do I not know?' is,
    and the difference decides whether the seats may be released."""

    @abstractmethod
    def charge(self, idempotency_key: str, amount_minor: int,
               currency: str) -> "PaymentOutcome": ...

    @abstractmethod
    def resolve(self, idempotency_key: str) -> "PaymentOutcome":
        """Ask what actually happened. Called for UNKNOWN outcomes only."""


class PaymentOutcome(Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"          # timeout, connection reset: MUST NOT release


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Monotonic seconds. Not wall-clock: a hold that expires early or
        late because NTP stepped the clock is a real, shipped bug."""
```

**Three-valued payment is the design point most candidates miss.** A boolean outcome forces you to guess on a timeout, and both guesses are wrong: releasing the seats risks selling them twice if the charge succeeded, and keeping them strands inventory if it did not. `UNKNOWN` says "do not decide yet" — keep the hold, poll, and let a reconciliation job settle it. That is `§7.3`.

---

## 5. Complete working code

**Language note:** Python 3.11+, standard library only. Every block is complete.

### 5.1 The seat model — the split that makes the read path cheap

```python
class SeatTier(Enum):
    """A price band. An enum rather than a class because there are a fixed,
    small number of them, and a class would invite a per-seat price override
    that revenue management does not want."""

    BALCONY = "balcony"
    STANDARD = "standard"
    PREMIUM = "premium"
    RECLINER = "recliner"


class Seat:
    """IMMUTABLE GEOMETRY. Row H, seat 12, a recliner. This object is the same
    for every show of every film for the lifetime of the building.

    Because it never changes, the seat MAP is cacheable at the CDN (§7.6),
    and the only per-show state is a separate, small object. Making Seat
    mutable would invalidate a cached map on every booking."""

    __slots__ = ("seat_id", "row", "number", "tier")

    def __init__(self, screen_id: str, row: str, number: int, tier: SeatTier) -> None:
        if number <= 0:
            raise ValueError("a seat number is positive")
        if not row:
            raise ValueError("a seat needs a row label")
        self.seat_id = f"{screen_id}:{row}{number}"
        self.row = row
        self.number = number
        self.tier = tier

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Seat):
            return NotImplemented
        return self.seat_id == other.seat_id

    def __hash__(self) -> int:
        return hash(self.seat_id)

    def __repr__(self) -> str:
        return f"Seat({self.seat_id}, {self.tier.value})"


class Screen:
    """A hall. Holds the layout, which is geometry and never changes."""

    def __init__(self, screen_id: str, name: str) -> None:
        self.screen_id = screen_id
        self.name = name
        self._seats: dict[str, Seat] = {}

    def add_seat(self, row: str, number: int, tier: SeatTier) -> Seat:
        seat = Seat(self.screen_id, row, number, tier)
        if seat.seat_id in self._seats:
            raise ValueError(f"duplicate seat {seat.seat_id}")
        self._seats[seat.seat_id] = seat
        return seat

    @property
    def seats(self) -> "dict[str, Seat]":
        return dict(self._seats)

    def neighbours(self, seat_id: str, count: int) -> "list[str]":
        """`count` contiguous seats in the same row starting at seat_id.
        Contiguity is a layout fact, so it lives with the layout."""
        seat = self._seats.get(seat_id)
        if seat is None:
            raise SeatError(f"unknown seat {seat_id}")
        row_seats = sorted(
            (s for s in self._seats.values() if s.row == seat.row),
            key=lambda s: s.number,
        )
        index = next(i for i, s in enumerate(row_seats) if s.seat_id == seat_id)
        window = row_seats[index:index + count]
        if len(window) != count:
            raise SeatError(f"only {len(window)} seat(s) from {seat_id} to the row end")
        for previous, current in zip(window, window[1:]):
            if current.number != previous.number + 1:
                raise SeatError(f"seats {previous.seat_id} and {current.seat_id} are not adjacent")
        return [s.seat_id for s in window]


class ShowSeat:
    """THE MUTABLE PER-SHOW STATE. One row per (show, seat). This is the object
    the entire locking design is about.

    `version` exists for the optimistic path: the compare-and-swap writes
    `WHERE version = :expected`, and the rowcount says whether you won."""

    __slots__ = ("show_id", "seat_id", "status", "held_by", "hold_expires_at",
                 "hold_key", "booking_id", "version")

    def __init__(self, show_id: str, seat_id: str,
                 status: SeatStatus = SeatStatus.AVAILABLE) -> None:
        self.show_id = show_id
        self.seat_id = seat_id
        self.status = status
        self.held_by: "str | None" = None
        self.hold_expires_at: "float | None" = None
        self.hold_key: "str | None" = None
        self.booking_id: "str | None" = None
        self.version = 0

    def is_available(self, now: float) -> bool:
        """LAZY EXPIRY: a hold whose deadline has passed is available, whatever
        the stored status says. This is what makes expiry correct without
        depending on a background job running on time."""
        if self.status is SeatStatus.AVAILABLE:
            return True
        if self.status is SeatStatus.HELD:
            if self.hold_expires_at is not None and now >= self.hold_expires_at:
                return True
        return False

    def is_held_by(self, hold_key: str, now: float) -> bool:
        """Whether THIS key owns a live hold. The check that makes retries
        idempotent: a retried acquire finds its own hold and succeeds."""
        return (
            self.status is SeatStatus.HELD
            and self.hold_key == hold_key
            and self.hold_expires_at is not None
            and now < self.hold_expires_at
        )

    def apply_hold(self, hold_key: str, owner_id: str, expires_at: float) -> None:
        self.status = SeatStatus.HELD
        self.hold_key = hold_key
        self.held_by = owner_id
        self.hold_expires_at = expires_at
        self.version += 1

    def apply_expiry(self) -> None:
        """Clear an expired hold. Idempotent: clearing an already-clear seat
        is a no-op, because a sweeper and a lazy read can race."""
        if self.status is SeatStatus.HELD:
            self.status = SeatStatus.AVAILABLE
        self.hold_key = None
        self.held_by = None
        self.hold_expires_at = None
        self.version += 1

    def apply_confirm(self, booking_id: str) -> None:
        if self.status is SeatStatus.BOOKED:
            raise SeatError(f"{self.seat_id} is already booked for {self.show_id}")
        self.status = SeatStatus.BOOKED
        self.booking_id = booking_id
        self.hold_key = None
        self.held_by = None
        self.hold_expires_at = None
        self.version += 1

    def apply_release(self) -> None:
        self.status = SeatStatus.AVAILABLE
        self.hold_key = None
        self.held_by = None
        self.hold_expires_at = None
        self.version += 1

    def __repr__(self) -> str:
        return f"ShowSeat({self.show_id}, {self.seat_id}, {self.status.value})"


class Show:
    """One screening. Owns the per-show seat state and the price map.

    Note the price is per SHOW and per TIER, not per seat: a seat's price is
    a property of the tier and the show, which is why `price_of` is a method
    on Show rather than a field on Seat."""

    def __init__(self, show_id: str, movie_title: str, screen: Screen,
                 starts_at: float, base_price_minor: int) -> None:
        self.show_id = show_id
        self.movie_title = movie_title
        self.screen = screen
        self.starts_at = starts_at
        self.base_price_minor = base_price_minor
        self._tier_surcharge: dict[SeatTier, int] = {}
        self._seats: dict[str, ShowSeat] = {
            seat_id: ShowSeat(show_id, seat_id) for seat_id in screen.seats
        }

    def set_surcharge(self, tier: SeatTier, surcharge_minor: int) -> None:
        if surcharge_minor < 0:
            raise ValueError("a surcharge cannot be negative")
        self._tier_surcharge[tier] = surcharge_minor

    def price_of(self, seat_id: str) -> int:
        seat = self.screen.seats.get(seat_id)
        if seat is None:
            raise SeatError(f"unknown seat {seat_id}")
        return self.base_price_minor + self._tier_surcharge.get(seat.tier, 0)

    def total_for(self, seat_ids: "list[str]") -> int:
        return sum(self.price_of(sid) for sid in seat_ids)

    @property
    def seats(self) -> "dict[str, ShowSeat]":
        return self._seats

    def seat(self, seat_id: str) -> ShowSeat:
        try:
            return self._seats[seat_id]
        except KeyError:
            raise SeatError(f"seat {seat_id} is not in show {self.show_id}") from None
```

### 5.2 The hold store — the interface where atomicity is promised

```python
class HoldStore(ABC):
    """The contract. Read the docstrings: they are the specification the two
    implementations must meet, and they are deliberately written so that a
    correct implementation is possible in both Redis and SQL."""

    @abstractmethod
    def acquire(self, show_id: str, seat_ids: "list[str]", owner_id: str,
                ttl_seconds: float, hold_key: str) -> "SeatHold":
        """ATOMIC. Either every seat is held by this call, or none is.

        MUST be idempotent on `hold_key`:
          - the same key, the same seats, an active hold  → return that hold
          - the same key, DIFFERENT seats                 → IdempotencyConflict

        MUST raise SeatTakenError listing EVERY conflicting seat, not the
        first — the UI highlights them all, and a first-only answer makes the
        user retry repeatedly.
        """

    @abstractmethod
    def confirm(self, show_id: str, seat_ids: "list[str]", hold_key: str,
                booking_id: str) -> None:
        """Convert a hold into a booking. ATOMIC over the seat set.

        MUST be idempotent: confirming an already-confirmed booking with the
        same id returns quietly. This is what makes a retried payment callback
        safe.
        """

    @abstractmethod
    def release(self, show_id: str, seat_ids: "list[str]", hold_key: str) -> bool:
        """Give the seats back. Idempotent: releasing a released hold returns
        False rather than raising, because a sweeper and a cancel can race."""

    @abstractmethod
    def status(self, show_id: str, seat_ids: "list[str]") -> "dict[str, SeatStatus]":
        """The read path. Must apply lazy expiry so a stale HELD reads as
        AVAILABLE without waiting for the sweeper."""


class SeatHold:
    """A lease. Note it stores the KEY, so a retried acquire can be recognised,
    and `expires_at`, so expiry is a property of the hold rather than of each
    seat — one deadline to reason about instead of N."""

    __slots__ = ("hold_key", "show_id", "seat_ids", "owner_id", "expires_at", "state")

    def __init__(self, hold_key: str, show_id: str, seat_ids: "list[str]",
                 owner_id: str, expires_at: float) -> None:
        self.hold_key = hold_key
        self.show_id = show_id
        self.seat_ids = list(seat_ids)
        self.owner_id = owner_id
        self.expires_at = expires_at
        self.state = HoldState.ACTIVE

    def has_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def __repr__(self) -> str:
        return (f"SeatHold({self.hold_key}, {self.show_id}, "
                f"{len(self.seat_ids)} seat(s), {self.state.value})")
```

### 5.3 The in-memory store — the reference implementation, where the race is visible

```python
import threading


class InMemoryHoldStore(HoldStore):
    """The reference implementation, and the one to write on a whiteboard
    because the race is visible in the code.

    LOCK GRANULARITY: one lock PER SHOW. Reasoning:
      - one global lock    → every show in the country serialises. Wrong.
      - one lock per seat  → cannot hold a SET atomically, because the set
                             spans seats. Wrong for this model.
      - one lock per show  → different shows never contend, and a seat set is
                             atomic within a show. RIGHT, and it is also what
                             makes the hot-show problem (§6.4) visible rather
                             than hidden.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._shows: dict[str, Show] = {}
        self._holds: dict[str, SeatHold] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def register(self, show: Show) -> None:
        with self._registry_lock:
            self._shows[show.show_id] = show
            self._locks.setdefault(show.show_id, threading.Lock())

    def _lock_for(self, show_id: str) -> threading.Lock:
        with self._registry_lock:
            lock = self._locks.get(show_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[show_id] = lock
            return lock

    def _show(self, show_id: str) -> Show:
        try:
            return self._shows[show_id]
        except KeyError:
            raise SeatError(f"unknown show {show_id}") from None

    # ---- acquire: the whole race lives in these twenty lines -----------
    def acquire(self, show_id: str, seat_ids: "list[str]", owner_id: str,
                ttl_seconds: float, hold_key: str) -> SeatHold:
        if not seat_ids:
            raise SeatError("a hold must name at least one seat")
        if ttl_seconds <= 0:
            raise SeatError("a hold must have a positive TTL")
        if len(set(seat_ids)) != len(seat_ids):
            raise SeatError("duplicate seat ids in the request")

        show = self._show(show_id)

        with self._lock_for(show_id):                 # ← ONE critical section
            now = self._clock.now()

            # --- idempotency FIRST, before any availability judgement ---
            existing = self._holds.get(hold_key)
            if existing is not None:
                if set(existing.seat_ids) != set(seat_ids):
                    raise IdempotencyConflict(
                        f"key {hold_key} was used for {sorted(existing.seat_ids)} "
                        f"and is now being used for {sorted(seat_ids)}"
                    )
                if existing.state is HoldState.CONFIRMED:
                    return existing                    # already booked: a no-op
                if existing.state is HoldState.ACTIVE and not existing.has_expired(now):
                    return existing                    # still live: return it
                # Expired or released: fall through and re-acquire below.

            # --- check ALL seats, collect EVERY conflict ---------------
            conflicts: list[str] = []
            for seat_id in seat_ids:
                show_seat = show.seat(seat_id)
                if show_seat.is_held_by(hold_key, now):
                    continue                           # our own hold: fine
                if not show_seat.is_available(now):
                    conflicts.append(seat_id)
            if conflicts:
                # Raise BEFORE mutating anything. This ordering is the reason
                # a partial hold is impossible: nothing is written until
                # everything has been checked.
                raise SeatTakenError(conflicts)

            # --- write ALL seats. Nothing above can have failed. -------
            expires_at = now + ttl_seconds
            for seat_id in seat_ids:
                show.seat(seat_id).apply_hold(hold_key, owner_id, expires_at)

            hold = SeatHold(hold_key, show_id, seat_ids, owner_id, expires_at)
            self._holds[hold_key] = hold
            return hold

    def confirm(self, show_id: str, seat_ids: "list[str]", hold_key: str,
                booking_id: str) -> None:
        show = self._show(show_id)
        with self._lock_for(show_id):
            now = self._clock.now()
            hold = self._holds.get(hold_key)
            if hold is None:
                raise HoldNotFoundError(f"no hold {hold_key}")

            # Idempotent: a retried payment callback must succeed quietly.
            if hold.state is HoldState.CONFIRMED:
                return
            if hold.state in (HoldState.RELEASED, HoldState.EXPIRED):
                raise HoldExpiredError(
                    f"hold {hold_key} is {hold.state.value}; the seats were given back"
                )
            if hold.has_expired(now):
                hold.state = HoldState.EXPIRED
                for seat_id in hold.seat_ids:
                    show.seat(seat_id).apply_expiry()
                raise HoldExpiredError(f"hold {hold_key} expired at {hold.expires_at}")

            for seat_id in seat_ids:
                show_seat = show.seat(seat_id)
                if show_seat.status is SeatStatus.BOOKED:
                    raise SeatError(f"{seat_id} is already booked")
                show_seat.apply_confirm(booking_id)
            hold.state = HoldState.CONFIRMED

    def release(self, show_id: str, seat_ids: "list[str]", hold_key: str) -> bool:
        show = self._show(show_id)
        with self._lock_for(show_id):
            hold = self._holds.get(hold_key)
            if hold is None:
                return False
            if hold.state is HoldState.CONFIRMED:
                return False                           # booked seats are not released
            if hold.state is HoldState.RELEASED:
                return False                           # idempotent
            for seat_id in hold.seat_ids:
                show.seat(seat_id).apply_release()
            hold.state = HoldState.RELEASED
            return True

    def status(self, show_id: str, seat_ids: "list[str]") -> "dict[str, SeatStatus]":
        show = self._show(show_id)
        now = self._clock.now()
        result: dict[str, SeatStatus] = {}
        with self._lock_for(show_id):
            for seat_id in seat_ids:
                show_seat = show.seat(seat_id)
                # LAZY EXPIRY on the read path: a stale HELD reads as
                # AVAILABLE, so the UI is right even if the sweeper is late.
                if show_seat.status is SeatStatus.HELD and not show_seat.is_available(now):
                    result[seat_id] = SeatStatus.HELD
                elif show_seat.status is SeatStatus.HELD:
                    result[seat_id] = SeatStatus.AVAILABLE
                else:
                    result[seat_id] = show_seat.status
        return result

    def sweep_expired(self, limit: int = 1_000) -> int:
        """Hygiene, not correctness — lazy expiry already makes an expired hold
        invisible. The sweeper exists so that HELD rows do not accumulate and
        so that downstream consumers of `SeatStatus` see a clean table."""
        now = self._clock.now()
        swept = 0
        with self._registry_lock:
            shows = list(self._shows.values())
            holds = list(self._holds.values())

        for hold in holds:
            if swept >= limit:
                break
            if hold.state is not HoldState.ACTIVE or not hold.has_expired(now):
                continue
            show = next((s for s in shows if s.show_id == hold.show_id), None)
            if show is None:
                continue
            with self._lock_for(hold.show_id):
                if hold.state is not HoldState.ACTIVE:
                    continue                           # another thread got it
                for seat_id in hold.seat_ids:
                    show.seat(seat_id).apply_expiry()
                hold.state = HoldState.EXPIRED
                swept += 1
        return swept
```

**Read `acquire` again and notice the shape:** idempotency check, then check-all collecting every conflict, then raise before writing, then write-all. **There is no path in which some seats are held and others are not** — not because the code is careful, but because the raise happens before the first mutation. That ordering is the atomicity, and it is worth saying exactly that in the interview.

### 5.4 The service — the saga, and the case that decides the design

```
   HOLD ──▶ PAY ──▶ CONFIRM
     │       │         │
     │       │         └── SUCCEEDED → confirm the hold → Booking
     │       │
     │       ├── FAILED  → release the hold → seats free immediately
     │       │
     │       └── UNKNOWN → DO NOTHING. Keep the hold. Poll `resolve`.
     │                     Releasing risks selling the seat twice;
     │                     confirming risks a free ticket. Wait.
     │
     └── expired before payment even started → HoldExpiredError, no charge
```

```python
class BookingService:
    """The orchestration. Note what it does NOT do: no locking, no seat state,
    no price arithmetic. It sequences, and the sequencing is the design."""

    def __init__(self, holds: HoldStore, payments: PaymentGateway,
                 clock: Clock, hold_ttl_seconds: float = 480.0,
                 max_extension_seconds: float = 120.0) -> None:
        self._holds = holds
        self._payments = payments
        self._clock = clock
        self._hold_ttl = hold_ttl_seconds
        self._max_extension = max_extension_seconds
        self._bookings: dict[str, "Booking"] = {}
        self._booking_lock = threading.Lock()
        self._next_booking = 0

    def _new_booking_id(self) -> str:
        with self._booking_lock:
            self._next_booking += 1
            return f"BKG-{self._next_booking:08d}"

    def hold_seats(self, show: Show, seat_ids: "list[str]", owner_id: str,
                   hold_key: str, extend: bool = False) -> SeatHold:
        """Take or extend a hold. `extend` is deliberately bounded: an
        unbounded extension is an indefinite lock, and the whole point of the
        TTL is that no user can freeze inventory forever."""
        ttl = self._hold_ttl
        if extend:
            ttl = min(ttl * 2, ttl + self._max_extension)
        return self._holds.acquire(show.show_id, seat_ids, owner_id, ttl, hold_key)

    def book(self, show: Show, seat_ids: "list[str]", owner_id: str,
             hold_key: str) -> "Booking":
        """The full flow. The order of operations here is the answer to
        'how does concurrent booking happen' — and note the payment comes
        AFTER the hold, never before.

        Paying first would mean charging a user who then loses the race, and a
        refund on every lost race. Holding first means the worst case is a
        hold nobody paid for, which expires on its own.
        """
        hold = self._holds.acquire(show.show_id, seat_ids, owner_id,
                                   self._hold_ttl, hold_key)

        amount = show.total_for(seat_ids)
        booking_id = self._new_booking_id()

        outcome = self._payments.charge(hold_key, amount, "INR")

        if outcome is PaymentOutcome.SUCCEEDED:
            self._holds.confirm(show.show_id, seat_ids, hold_key, booking_id)
            booking = Booking(booking_id, hold_key, show.show_id, list(seat_ids), amount)
            with self._booking_lock:
                self._bookings[booking_id] = booking
            return booking

        if outcome is PaymentOutcome.FAILED:
            # Safe to release: the gateway told us it did not charge.
            self._holds.release(show.show_id, seat_ids, hold_key)
            raise SeatError(f"payment failed for {hold_key}")

        # UNKNOWN. The seats stay held. A reconciliation job calls
        # settle_unknown() until the gateway gives a verdict, and the hold's
        # TTL is the backstop if the gateway never does.
        raise PaymentUnknownError(
            f"payment outcome for {hold_key} is unknown; the hold is retained "
            f"and must be settled, not released"
        )

    def settle_unknown(self, show: Show, seat_ids: "list[str]", hold_key: str,
                       owner_id: str) -> "Booking | None":
        """Called by the reconciliation job for holds in the UNKNOWN state.
        Re-asking is safe because BOTH the gateway call and the confirm are
        idempotent on the same key."""
        outcome = self._payments.resolve(hold_key)
        if outcome is PaymentOutcome.SUCCEEDED:
            booking_id = self._new_booking_id()
            self._holds.confirm(show.show_id, seat_ids, hold_key, booking_id)
            booking = Booking(booking_id, hold_key, show.show_id,
                              list(seat_ids), show.total_for(seat_ids))
            with self._booking_lock:
                self._bookings[booking_id] = booking
            return booking
        if outcome is PaymentOutcome.FAILED:
            self._holds.release(show.show_id, seat_ids, hold_key)
            return None
        return None                                # still unknown: try again later

    def cancel(self, show: Show, seat_ids: "list[str]", hold_key: str) -> bool:
        return self._holds.release(show.show_id, seat_ids, hold_key)


class PaymentUnknownError(SeatError):
    """Deliberately NOT a failure. It means 'wait', and a caller that treats it
    as a failure and releases the seats is the bug this class exists to stop."""


class Booking:
    __slots__ = ("booking_id", "hold_key", "show_id", "seat_ids", "amount_minor")

    def __init__(self, booking_id: str, hold_key: str, show_id: str,
                 seat_ids: "list[str]", amount_minor: int) -> None:
        self.booking_id = booking_id
        self.hold_key = hold_key
        self.show_id = show_id
        self.seat_ids = seat_ids
        self.amount_minor = amount_minor

    def __repr__(self) -> str:
        return (f"Booking({self.booking_id}, {self.show_id}, "
                f"{len(self.seat_ids)} seat(s), {self.amount_minor})")
```

### 5.5 It runs

```python
def demo() -> None:
    screen = Screen("SCR-1", "Audi 1")
    for row in ("A", "B", "C"):
        for number in range(1, 9):
            tier = SeatTier.RECLINER if row == "A" else (
                SeatTier.PREMIUM if row == "B" else SeatTier.STANDARD
            )
            screen.add_seat(row, number, tier)

    show = Show("SHW-1", "Interstellar", screen, starts_at=1_800_000_000.0,
                base_price_minor=20000)
    show.set_surcharge(SeatTier.PREMIUM, 5000)
    show.set_surcharge(SeatTier.RECLINER, 15000)

    clock = ManualClock()
    store = InMemoryHoldStore(clock)
    store.register(show)
    payments = FakePaymentGateway(PaymentOutcome.SUCCEEDED)
    service = BookingService(store, payments, clock)

    seats = ["SCR-1:H5", "SCR-1:H6"]

    hold = service.hold_seats(show, seats, owner_id="user-42", hold_key="idem-abc")
    print(hold)                                             # 2 seats, active

    # A retry with the same key and the same seats returns the SAME hold.
    again = service.hold_seats(show, seats, owner_id="user-42", hold_key="idem-abc")
    print(again is hold)                                    # True — idempotent

    # Someone else wants H6.
    try:
        store.acquire(show.show_id, ["SCR-1:H6"], "user-99", 480.0, "idem-xyz")
    except SeatTakenError as exc:
        print("conflict:", exc.conflicts)                    # ['SCR-1:H6']

    booking = service.book(show, seats, owner_id="user-42", hold_key="idem-abc")
    print(booking)
    print(store.status(show.show_id, seats))                 # both BOOKED

    # The clock passes the deadline: an untouched hold becomes available again.
    clock.advance(500.0)
    store.acquire(show.show_id, ["SCR-1:H5"], "user-7", 480.0, "idem-late")
    print(store.status(show.show_id, seats))
    # H5 is HELD by the new owner; H6 is still BOOKED, because a confirmed
    # booking is NOT affected by time — only a hold is.

    print("swept:", store.sweep_expired())


class ManualClock(Clock):
    """Deterministic time. This is why every expiry test in this file runs in
    microseconds instead of sleeping for eight minutes."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class FakePaymentGateway(PaymentGateway):
    def __init__(self, outcome: PaymentOutcome) -> None:
        self.outcome = outcome
        self.charges: list[tuple[str, int]] = []

    def charge(self, idempotency_key: str, amount_minor: int, currency: str) -> PaymentOutcome:
        # A real gateway dedupes on the key; the fake does too, because the
        # idempotency contract is what is being tested.
        for key, _ in self.charges:
            if key == idempotency_key:
                return self.outcome
        self.charges.append((idempotency_key, amount_minor))
        return self.outcome

    def resolve(self, idempotency_key: str) -> PaymentOutcome:
        return self.outcome


if __name__ == "__main__":
    demo()
```

**The last line of the demonstration is the one that matters.** After the clock jumps past the deadline, `H5` — whose hold lapsed — is available again, while `H6` stays `BOOKED`. **Time expires a hold; it never expires a sale.** An implementation that clears `BOOKED` seats on a timer is selling the same seat twice, and it is a mistake that follows directly from conflating a lease with a sale.

### 5.6 The Redis implementation — atomicity via Lua

```python
class RedisHoldStore(HoldStore):
    """The fast path. The correctness argument is entirely in the Lua scripts:
    Redis runs a script atomically, so check-all-then-set-all inside one script
    is a single critical section, exactly like the in-memory lock.

    Why the script and not a transaction: a WATCH/MULTI/EXEC retry loop needs
    the client to refetch and retry, and under a hot show the retry rate is
    what kills it. A script is one round trip and cannot interleave.

    The key layout, and the reason for each part:
      show:{show_id}:seat:{seat_id}     a hash: status, owner, expiry, hold_key
      show:{show_id}:seats              a set of every seat id, for the sweeper
      hold:{hold_key}                   a hash: the seats this key owns
                                        (a TTL on this key is the BACKSTOP; the
                                         per-seat deadline is the real clock)
    """

    ACQUIRE_LUA = """
    -- KEYS: [1..n] the seat keys, [n+1] the hold key
    -- ARGV: [1] the hold_key, [2] the owner, [3] the expiry epoch seconds,
    --       [4] a per-seat TTL in seconds, [5] the seat-key TTL in seconds
    local hold_key = ARGV[1]
    local owner    = ARGV[2]
    local expires  = tonumber(ARGV[3])
    local hold_ttl = tonumber(ARGV[4])
    local seat_ttl = tonumber(ARGV[5])

    -- Is this key already holding? Idempotency, checked first.
    local existing = redis.call('HGET', KEYS[#KEYS], 'seats')
    if existing then
      local wanted = table.concat(ARGV[6], ',')
      if existing ~= wanted then
        return redis.error_reply('IDEMPOTENCY_CONFLICT')
      end
      return redis.status_reply('EXISTING')
    end

    -- Check ALL seats, collecting EVERY conflict. Nothing is written yet.
    local conflicts = {}
    for i = 1, #KEYS - 1 do
      local status = redis.call('HGET', KEYS[i], 'status')
      local seat_expires = tonumber(redis.call('HGET', KEYS[i], 'expires') or '0')
      local seat_owner = redis.call('HGET', KEYS[i], 'hold_key')
      -- Available, or held by US, or held past its deadline (lazy expiry).
      if status == 'BOOKED' or status == 'BLOCKED' then
        table.insert(conflicts, KEYS[i])
      elseif status == 'HELD' and seat_owner ~= hold_key and seat_expires > tonumber(ARGV[7]) then
        table.insert(conflicts, KEYS[i])
      end
    end
    if #conflicts > 0 then
      return redis.error_reply('SEAT_TAKEN:' .. table.concat(conflicts, ','))
    end

    -- Write ALL seats. Every check passed, so this cannot fail part-way.
    for i = 1, #KEYS - 1 do
      redis.call('HSET', KEYS[i], 'status', 'HELD', 'owner', owner,
                 'hold_key', hold_key, 'expires', expires)
      redis.call('EXPIRE', KEYS[i], seat_ttl)
    end
    redis.call('HSET', KEYS[#KEYS], 'seats', table.concat(ARGV[6], ','),
               'owner', owner, 'expires', expires)
    redis.call('EXPIRE', KEYS[#KEYS], hold_ttl)
    return redis.status_reply('OK')
    """

    CONFIRM_LUA = """
    -- KEYS: [1..n] the seat keys, [n+1] the hold key
    -- ARGV: [1] the hold_key, [2] the booking id, [3] the seat-key TTL
    local hold_key = ARGV[1]
    local booking  = ARGV[2]

    -- Idempotent: already confirmed with this booking -> succeed quietly.
    local booked = redis.call('HGET', KEYS[#KEYS], 'booking_id')
    if booked and booked == booking then
      return redis.status_reply('ALREADY')
    end

    for i = 1, #KEYS - 1 do
      local status = redis.call('HGET', KEYS[i], 'status')
      local seat_owner = redis.call('HGET', KEYS[i], 'hold_key')
      if status ~= 'HELD' or seat_owner ~= hold_key then
        return redis.error_reply('HOLD_LOST:' .. KEYS[i])
      end
    end
    for i = 1, #KEYS - 1 do
      redis.call('HSET', KEYS[i], 'status', 'BOOKED', 'booking_id', booking)
      redis.call('HDEL', KEYS[i], 'hold_key', 'owner', 'expires')
      -- NOTE: the seat key now has NO TTL. A booking is durable, and giving
      -- it a TTL is how a sold seat silently becomes available again.
      redis.call('PERSIST', KEYS[i])
    end
    redis.call('HSET', KEYS[#KEYS], 'booking_id', booking)
    return redis.status_reply('OK')
    """

    def __init__(self, client: object, clock: Clock,
                 seat_key_ttl_seconds: float = 86_400.0,
                 max_script_retries: int = 3) -> None:
        self._redis = client
        self._clock = clock
        self._seat_ttl = seat_key_ttl_seconds
        self._max_retries = max_script_retries

    def _seat_keys(self, show_id: str, seat_ids: "list[str]") -> "list[str]":
        return [f"show:{show_id}:seat:{seat_id}" for seat_id in seat_ids]

    def acquire(self, show_id: str, seat_ids: "list[str]", owner_id: str,
                ttl_seconds: float, hold_key: str) -> SeatHold:
        import hashlib

        if not seat_ids:
            raise SeatError("a hold must name at least one seat")
        # A deterministic seat order, so the "same key, same seats" comparison
        # in the script is a string equality rather than a set comparison in Lua.
        ordered = sorted(set(seat_ids))
        if len(ordered) != len(seat_ids):
            raise SeatError("duplicate seat ids in the request")

        now = self._clock.now()
        keys = self._seat_keys(show_id, ordered) + [f"hold:{hold_key}"]
        argv = [
            hold_key, owner_id, now + ttl_seconds,
            ttl_seconds + 60.0, self._seat_ttl,
            ",".join(ordered), str(now),
        ]

        last_error: "Exception | None" = None
        for _ in range(self._max_retries):
            try:
                self._redis.eval(self.ACQUIRE_LUA, len(keys), *keys, *argv)
                return SeatHold(hold_key, show_id, ordered, owner_id, now + ttl_seconds)
            except Exception as exc:                       # noqa: BLE001
                message = str(exc)
                if "SEAT_TAKEN:" in message:
                    conflicts = message.split("SEAT_TAKEN:", 1)[1].split(",")
                    clean = [c.rsplit(":", 1)[-1] for c in conflicts]
                    raise SeatTakenError(clean) from None
                if "IDEMPOTENCY_CONFLICT" in message:
                    raise IdempotencyConflict(
                        f"key {hold_key} was reused for a different seat set"
                    ) from None
                last_error = exc                           # a transient Redis error
        raise SeatError(f"hold failed after {self._max_retries} attempts: {last_error}")

    def confirm(self, show_id: str, seat_ids: "list[str]", hold_key: str,
                booking_id: str) -> None:
        ordered = sorted(set(seat_ids))
        keys = self._seat_keys(show_id, ordered) + [f"hold:{hold_key}"]
        try:
            self._redis.eval(self.CONFIRM_LUA, len(keys), *keys,
                             hold_key, booking_id, self._seat_ttl)
        except Exception as exc:                           # noqa: BLE001
            message = str(exc)
            if "HOLD_LOST:" in message:
                raise HoldExpiredError(
                    f"hold {hold_key} is no longer live; the seats were given back"
                ) from None
            raise

    def release(self, show_id: str, seat_ids: "list[str]", hold_key: str) -> bool:
        """Idempotent by construction: only seats whose `hold_key` matches are
        cleared, and only HELD seats are touched. A confirmed seat is left
        alone, which is the property that matters."""
        ordered = sorted(set(seat_ids))
        released = 0
        for key in self._seat_keys(show_id, ordered):
            status = self._redis.hget(key, "status")
            owner = self._redis.hget(key, "hold_key")
            if status == "HELD" and owner == hold_key:
                self._redis.hset(key, mapping={"status": "AVAILABLE"})
                self._redis.hdel(key, "hold_key", "owner", "expires")
                released += 1
        if released:
            self._redis.delete(f"hold:{hold_key}")
        return released > 0

    def status(self, show_id: str, seat_ids: "list[str]") -> "dict[str, SeatStatus]":
        now = self._clock.now()
        result: dict[str, SeatStatus] = {}
        for seat_id in seat_ids:
            key = f"show:{show_id}:seat:{seat_id}"
            status = self._redis.hget(key, "status")
            if status is None:
                result[seat_id] = SeatStatus.UNAVAILABLE
            elif status == "HELD":
                expires = float(self._redis.hget(key, "expires") or 0.0)
                # Lazy expiry, same as the in-memory store.
                result[seat_id] = (SeatStatus.HELD if now < expires
                                   else SeatStatus.AVAILABLE)
            else:
                result[seat_id] = SeatStatus(status)
        return result
```

**Two details in the Lua worth defending.** First, the `PERSIST` on confirm: a booked seat must **not** carry a TTL, because a TTL is how a sold seat silently becomes available again — and that is a live bug in systems that set a blanket expiry on every seat key. Second, the `hold:{key}` key has its own TTL as a **backstop**: if the per-seat cleanup ever misses, the hold record itself still goes away, so a leak is bounded rather than permanent.

### 5.7 The database implementation — the durable truth

```python
class DbHoldStore(HoldStore):
    """The durable path. Slower per operation, and the one that survives a
    Redis restart — which matters, because losing every hold in a failover
    means every in-flight customer loses their seats.

    PESSIMISTIC by default: `SELECT … FOR UPDATE` on the seat rows, in a
    transaction. Correct, and the row locks are held for the transaction's
    duration, so the seat set must be SMALL (§6.4).

    The locking ORDER is not incidental. `ORDER BY seat_id` is what prevents a
    deadlock between two overlapping set requests — the same sorted-acquisition
    rule as the night striping in the hotel problem.
    """

    ACQUIRE_SQL = """
        SELECT seat_id, status, hold_key, hold_expires_at, version
          FROM show_seat
         WHERE show_id = %(show_id)s
           AND seat_id = ANY(%(seat_ids)s)
         ORDER BY seat_id
           FOR UPDATE
    """

    def __init__(self, cursor_factory: "Callable[[], object]",
                 clock: Clock) -> None:
        self._cursor_factory = cursor_factory
        self._clock = clock

    def acquire(self, show_id: str, seat_ids: "list[str]", owner_id: str,
                ttl_seconds: float, hold_key: str) -> SeatHold:
        if not seat_ids:
            raise SeatError("a hold must name at least one seat")
        ordered = sorted(set(seat_ids))
        if len(ordered) != len(seat_ids):
            raise SeatError("duplicate seat ids in the request")
        now = self._clock.now()

        with self._cursor_factory() as cursor:
            # One transaction. The FOR UPDATE locks are released on COMMIT, so
            # the entire check-and-write is inside one critical section —
            # across processes, which a threading.Lock cannot be.
            rows = cursor.execute(
                self.ACQUIRE_SQL, {"show_id": show_id, "seat_ids": ordered}
            ).fetchall()

            if len(rows) != len(ordered):
                raise SeatError("one or more seats are not in this show")

            # Idempotency first.
            existing = [r for r in rows if r["hold_key"] == hold_key]
            if len(existing) == len(rows):
                return SeatHold(hold_key, show_id, ordered, owner_id, now + ttl_seconds)

            conflicts = [
                r["seat_id"] for r in rows
                if r["status"] in ("BOOKED", "BLOCKED")
                or (r["status"] == "HELD"
                    and r["hold_key"] != hold_key
                    and (r["hold_expires_at"] is None or r["hold_expires_at"] > now))
            ]
            if conflicts:
                # Roll back by doing nothing: the transaction has written
                # nothing yet, so raising here leaves the rows untouched.
                raise SeatTakenError(sorted(conflicts))

            cursor.execute(
                """
                UPDATE show_seat
                   SET status = 'HELD', hold_key = %(hold_key)s,
                       held_by = %(owner_id)s, hold_expires_at = %(expires)s,
                       version = version + 1
                 WHERE show_id = %(show_id)s
                   AND seat_id = ANY(%(seat_ids)s)
                """,
                {
                    "hold_key": hold_key,
                    "owner_id": owner_id,
                    "expires": now + ttl_seconds,
                    "show_id": show_id,
                    "seat_ids": ordered,
                },
            )
        return SeatHold(hold_key, show_id, ordered, owner_id, now + ttl_seconds)

    def confirm(self, show_id: str, seat_ids: "list[str]", hold_key: str,
                booking_id: str) -> None:
        ordered = sorted(set(seat_ids))
        now = self._clock.now()
        with self._cursor_factory() as cursor:
            rows = cursor.execute(
                self.ACQUIRE_SQL, {"show_id": show_id, "seat_ids": ordered}
            ).fetchall()

            if all(r["status"] == "BOOKED" for r in rows):
                return                                     # idempotent
            if not all(r["status"] == "HELD" and r["hold_key"] == hold_key
                       for r in rows):
                raise HoldExpiredError(f"hold {hold_key} is not live on every seat")

            cursor.execute(
                """
                UPDATE show_seat
                   SET status = 'BOOKED', booking_id = %(booking_id)s,
                       hold_key = NULL, held_by = NULL, hold_expires_at = NULL,
                       version = version + 1
                 WHERE show_id = %(show_id)s
                   AND seat_id = ANY(%(seat_ids)s)
                """,
                {"booking_id": booking_id, "show_id": show_id, "seat_ids": ordered},
            )

    def release(self, show_id: str, seat_ids: "list[str]", hold_key: str) -> bool:
        ordered = sorted(set(seat_ids))
        with self._cursor_factory() as cursor:
            result = cursor.execute(
                """
                UPDATE show_seat
                   SET status = 'AVAILABLE', hold_key = NULL, held_by = NULL,
                       hold_expires_at = NULL, version = version + 1
                 WHERE show_id = %(show_id)s
                   AND seat_id = ANY(%(seat_ids)s)
                   AND status = 'HELD'
                   AND hold_key = %(hold_key)s
                """,
                {"show_id": show_id, "seat_ids": ordered, "hold_key": hold_key},
            )
            return result.rowcount > 0

    def status(self, show_id: str, seat_ids: "list[str]") -> "dict[str, SeatStatus]":
        now = self._clock.now()
        with self._cursor_factory() as cursor:
            rows = cursor.execute(
                """
                SELECT seat_id, status, hold_expires_at
                  FROM show_seat
                 WHERE show_id = %(show_id)s AND seat_id = ANY(%(seat_ids)s)
                """,
                {"show_id": show_id, "seat_ids": sorted(set(seat_ids))},
            ).fetchall()

        result: dict[str, SeatStatus] = {}
        for row in rows:
            status = row["status"]
            if status == "HELD":
                expires = row["hold_expires_at"]
                # Lazy expiry in SQL: the read path never trusts a stale HELD.
                if expires is not None and expires <= now:
                    result[row["seat_id"]] = SeatStatus.AVAILABLE
                    continue
            result[row["seat_id"]] = SeatStatus(status)
        return result

    def hold_seats(
        self, show_id: str, seat_ids: "list[str]", owner_id: str,
        ttl_seconds: float, hold_key: str,
    ) -> SeatHold:
        """Alias kept so the two implementations are interchangeable behind the
        interface; the service only ever calls `acquire`."""
        return self.acquire(show_id, seat_ids, owner_id, ttl_seconds, hold_key)
```

**The optimistic variant, for when the row locks contend:**

```python
class DbOptimisticHoldStore(DbHoldStore):
    """Same table, no FOR UPDATE. The UPDATE carries the condition, so the
    database decides the winner and the rowcount reports it.

    Use this when the seat set is large or the show is hot enough that row
    locks queue. The cost is retries AND the partial-hold hazard: a multi-row
    optimistic update can succeed on the first row and fail on the third, so
    it MUST run inside a transaction. Without one, the compensating UPDATE is
    mandatory and is easy to get wrong."""

    MAX_ATTEMPTS = 5

    def acquire(self, show_id: str, seat_ids: "list[str]", owner_id: str,
                ttl_seconds: float, hold_key: str) -> SeatHold:
        ordered = sorted(set(seat_ids))
        now = self._clock.now()
        expires = now + ttl_seconds

        for attempt in range(self.MAX_ATTEMPTS):
            with self._cursor_factory() as cursor:
                # BEGIN … the cursor context is the transaction boundary.
                rows = cursor.execute(
                    """
                    SELECT seat_id, status, hold_key, hold_expires_at, version
                      FROM show_seat
                     WHERE show_id = %(show_id)s AND seat_id = ANY(%(seat_ids)s)
                    """,
                    {"show_id": show_id, "seat_ids": ordered},
                ).fetchall()
                if len(rows) != len(ordered):
                    raise SeatError("one or more seats are not in this show")

                if all(r["hold_key"] == hold_key for r in rows):
                    return SeatHold(hold_key, show_id, ordered, owner_id, expires)

                conflicts = [
                    r["seat_id"] for r in rows
                    if r["status"] in ("BOOKED", "BLOCKED")
                    or (r["status"] == "HELD"
                        and r["hold_key"] != hold_key
                        and (r["hold_expires_at"] is None or r["hold_expires_at"] > now))
                ]
                if conflicts:
                    raise SeatTakenError(sorted(conflicts))

                versions = {r["seat_id"]: r["version"] for r in rows}
                won_every_row = True
                for seat_id in ordered:
                    result = cursor.execute(
                        """
                        UPDATE show_seat
                           SET status = 'HELD', hold_key = %(hold_key)s,
                               held_by = %(owner_id)s, hold_expires_at = %(expires)s,
                               version = version + 1
                         WHERE show_id = %(show_id)s AND seat_id = %(seat_id)s
                           AND version = %(version)s
                        """,
                        {
                            "hold_key": hold_key, "owner_id": owner_id,
                            "expires": expires, "show_id": show_id,
                            "seat_id": seat_id, "version": versions[seat_id],
                        },
                    )
                    if result.rowcount != 1:
                        won_every_row = False
                        break                     # the transaction rolls back
                if won_every_row:
                    return SeatHold(hold_key, show_id, ordered, owner_id, expires)

            if attempt < self.MAX_ATTEMPTS - 1:
                # Exponential backoff with jitter: without jitter, N losers
                # retry in lockstep and collide again, forever. This is the
                # thundering herd, in miniature.
                import random
                delay = min(0.05 * (2 ** attempt), 0.5)
                time.sleep(delay * (0.5 + random.random()))

        raise SeatError(f"could not acquire after {self.MAX_ATTEMPTS} attempts")
```

### 5.8 The sweeper

```python
class HoldSweeper:
    """The proactive half of expiry. Correctness does not depend on it — lazy
    expiry already hides an expired hold — but it exists for three reasons:

      1. A HELD row that nobody reads again is still a row. Unbounded growth.
      2. Downstream consumers (analytics, the seat-map projection) read status
         directly and would see stale HELDs.
      3. A seat released on time can be sold again within the same minute, and
         on a hot show that minute is revenue.

    It must be IDEMPOTENT and SAFE TO RUN CONCURRENTLY: two sweepers on two
    instances is normal, so every operation checks the state it expects before
    mutating it.
    """

    def __init__(self, store: "HoldStore", clock: Clock,
                 interval_seconds: float = 15.0, batch_limit: int = 1_000) -> None:
        self._store = store
        self._clock = clock
        self._interval = interval_seconds
        self._batch = batch_limit
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None

    def sweep_once(self) -> int:
        sweep = getattr(self._store, "sweep_expired", None)
        if sweep is None:
            return 0
        return int(sweep(self._batch))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="hold-sweeper", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.sweep_once()
            except Exception:                              # noqa: BLE001
                # A sweeper failure must NEVER kill the thread: expiry is a
                # hygiene job, and a dead sweeper means unbounded growth that
                # nobody notices for weeks. Log and continue.
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
```

---

## 6. Concurrency and thread-safety

### 6.1 The exact race, at statement level

Two users, one seat, the same instant. This is the interleaving the whole file exists to prevent.

```
   t   Thread A (user-42, seat H6)              Thread B (user-99, seat H6)
   │
   1   acquire(H6)
   2     read H6: status=AVAILABLE
   3                                            acquire(H6)
   4                                              read H6: status=AVAILABLE
   5     is_available → True
   6                                              is_available → True
   7     apply_hold(H6, "key-A")                 ┐
   8     return hold-A                           │  no conflict is ever
   9                                              │  observed, because both
  10                                              │  reads happened before
  11                                              │  either write
  12                                              ┘
  13                                            apply_hold(H6, "key-B")
  14                                            return hold-B

   RESULT: TWO ACTIVE HOLDS ON ONE SEAT.
           Both users proceed to payment. Both payments succeed. One is
           refunded after a support call — or both are confirmed and one
           customer arrives to find someone in their seat.

   This is check-then-act: steps 2–6 are the check, steps 7 and 13 are the
   act, and the window between them is where the second thread lives.
```

**Three ways to close the window, and one that does not:**

| Approach | How | When it is right |
|---|---|---|
| **Pessimistic** | `SELECT … FOR UPDATE` on the seat rows, in a transaction | Small seat sets; the default durable path |
| **Atomic script** | All the checks and all the writes in one Lua script | Redis; the hot path; one round trip, no retry loop |
| **Optimistic CAS** | `UPDATE … WHERE version = :v`; rowcount decides; retry on loss | Large seat sets, low collision rate |
| **In-process lock** | One lock per show | One process only. Correct there, and useless across instances |
| **Check, then lock** *(the bug)* | Take the lock only for the write | Never. It is exactly the interleaving above |

### 6.2 Why per-show and not per-seat

```
   PER-SEAT LOCKS:   lock(H5); lock(H6); ... then check and write each
     → The set is NOT atomic. Thread A can hold H5 while thread B takes H6,
       and both requests fail — a livelock of mutual partial holds.
     → Correct only with sorted acquisition AND a release-everything path,
       which is more code than one lock and no faster for realistic set sizes.

   PER-SHOW LOCK:    lock(show); check all; write all
     → Atomic by construction. Different shows never contend.
     → The cost: a hot show serialises. See §6.4.

   PER-SEAT, SORTED: the compromise, if sets are large
     → Acquire in sorted seat order, always. Two overlapping requests take
       locks in the same order, so one waits rather than deadlocking.
       This is the same rule as the night striping in the hotel problem and
       the account ordering in Splitwise.
```

### 6.3 The invariant, stated so it can be tested

**For every (show, seat), at most one hold is live and at most one booking exists.**

Three consequences worth spelling out, because each is a test and each is a thing candidates get wrong:

1. **At most one live hold.** A second acquire must fail with the conflicting seat listed.
2. **A booking outlives a hold.** `BOOKED` is not affected by time — see the last line of §5.5.
3. **Expiry is monotone.** Once a hold expires, that key can never be confirmed. It can re-acquire, but only through `acquire`, which is a new decision.

### 6.4 The hot-show problem, and the honest answer

A blockbuster's first show: 10 000 concurrent requests for one show's seats, and a per-show lock serialises all of them. Each critical section is short — a few dict lookups — but at 10 000 requests it is a queue.

**Three mitigations, in increasing order of cost:**

- **Partition by seat.** Shard the seat set into `R` groups, each with its own lock. A request touching seats in several groups takes those locks in sorted group order. Contention drops by a factor of `R`, at the cost of a multi-lock acquire and the sorted-order discipline.
- **Shard the show.** Split one screening across `K` logical "sections", each with its own store shard — the same trick as the sharded cache. The seat map is presented as one screen; the state is partitioned.
- **A waiting room.** This is what the case study describes and it is the real production answer: put a virtual queue in front so that at most *N* users can be in the seat-selection path at once. **The lock is then never contended, because the queue does the serialising.** That is an architecture answer, and it belongs in the case study — but naming it here shows you know where your class design stops.

**Say this out loud:** *"A per-show lock is the correct unit for a normal show. For a hot show the lock is not the problem to solve — the problem is 10 000 people in the selection path, and the fix is a waiting room, not a cleverer lock."*

### 6.5 Tests

```python
def test_two_threads_cannot_hold_the_same_seat() -> None:
    """THE test. Two threads, one seat, released together by a barrier."""
    screen = Screen("SCR-1", "Audi 1")
    screen.add_seat("H", 6, SeatTier.STANDARD)
    show = Show("SHW-1", "Interstellar", screen, 1_800_000_000.0, 20000)
    clock = ManualClock()
    store = InMemoryHoldStore(clock)
    store.register(show)

    barrier = threading.Barrier(2)
    outcomes: list[object] = []
    lock = threading.Lock()

    def attempt(owner: str, key: str) -> None:
        barrier.wait()
        try:
            hold = store.acquire(show.show_id, ["SCR-1:H6"], owner, 480.0, key)
            with lock:
                outcomes.append(hold.hold_key)
        except SeatTakenError as exc:
            with lock:
                outcomes.append(exc)

    threads = [
        threading.Thread(target=attempt, args=("user-42", "key-A")),
        threading.Thread(target=attempt, args=("user-99", "key-B")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [o for o in outcomes if isinstance(o, str)]
    assert len(winners) == 1, f"expected exactly one hold, got {outcomes}"
    assert show.seat("SCR-1:H6").status is SeatStatus.HELD


def test_acquire_is_idempotent_on_the_same_key() -> None:
    clock = ManualClock()
    store, show = _single_seat_show(clock)
    first = store.acquire(show.show_id, ["SCR-1:H6"], "user-42", 480.0, "key-A")
    second = store.acquire(show.show_id, ["SCR-1:H6"], "user-42", 480.0, "key-A")
    assert first is not second
    assert first.expires_at == second.expires_at      # the SAME lease, not a new one


def test_same_key_different_seats_is_a_conflict() -> None:
    clock = ManualClock()
    store, show = _two_seat_show(clock)
    store.acquire(show.show_id, ["SCR-1:H6"], "user-42", 480.0, "key-A")
    with pytest.raises(IdempotencyConflict):
        store.acquire(show.show_id, ["SCR-1:H7"], "user-42", 480.0, "key-A")


def test_every_conflicting_seat_is_reported() -> None:
    """The UI highlights all of them; a first-only answer causes retry loops."""
    clock = ManualClock()
    store, show = _two_seat_show(clock)
    store.acquire(show.show_id, ["SCR-1:H6"], "user-99", 480.0, "key-B")
    with pytest.raises(SeatTakenError) as info:
        store.acquire(show.show_id, ["SCR-1:H6", "SCR-1:H7"], "user-42", 480.0, "key-A")
    assert info.value.conflicts == ["SCR-1:H6"]


def test_a_failed_hold_leaves_no_partial_state() -> None:
    """The all-or-nothing property, asserted on the seats themselves."""
    clock = ManualClock()
    store, show = _two_seat_show(clock)
    store.acquire(show.show_id, ["SCR-1:H7"], "user-99", 480.0, "key-B")
    with pytest.raises(SeatTakenError):
        store.acquire(show.show_id, ["SCR-1:H6", "SCR-1:H7"], "user-42", 480.0, "key-A")
    assert show.seat("SCR-1:H6").status is SeatStatus.AVAILABLE   # NOT held
    assert show.seat("SCR-1:H7").hold_key == "key-B"


def test_hold_expires_exactly_at_the_ttl() -> None:
    clock = ManualClock()
    store, show = _single_seat_show(clock)
    store.acquire(show.show_id, ["SCR-1:H6"], "user-42", 480.0, "key-A")
    clock.advance(479.999)
    assert store.status(show.show_id, ["SCR-1:H6"])["SCR-1:H6"] is SeatStatus.HELD
    clock.advance(0.001)                                  # exactly at the deadline
    assert store.status(show.show_id, ["SCR-1:H6"])["SCR-1:H6"] is SeatStatus.AVAILABLE


def test_a_booking_is_not_expired_by_time() -> None:
    """A lease expires; a sale does not. The distinction the design turns on."""
    clock = ManualClock()
    store, show = _single_seat_show(clock)
    store.acquire(show.show_id, ["SCR-1:H6"], "user-42", 480.0, "key-A")
    store.confirm(show.show_id, ["SCR-1:H6"], "key-A", "BKG-1")
    clock.advance(86_400.0)                               # a day later
    assert store.status(show.show_id, ["SCR-1:H6"])["SCR-1:H6"] is SeatStatus.BOOKED
    assert store.sweep_expired() == 0


def test_confirm_is_idempotent() -> None:
    clock = ManualClock()
    store, show = _single_seat_show(clock)
    store.acquire(show.show_id, ["SCR-1:H6"], "user-42", 480.0, "key-A")
    store.confirm(show.show_id, ["SCR-1:H6"], "key-A", "BKG-1")
    store.confirm(show.show_id, ["SCR-1:H6"], "key-A", "BKG-1")   # no exception
    assert show.seat("SCR-1:H6").booking_id == "BKG-1"


def test_unknown_payment_retains_the_hold() -> None:
    """The case the three-valued outcome exists for."""
    clock = ManualClock()
    store, show = _two_seat_show(clock)
    service = BookingService(store, FakePaymentGateway(PaymentOutcome.UNKNOWN), clock)
    with pytest.raises(PaymentUnknownError):
        service.book(show, ["SCR-1:H6"], "user-42", "key-A")
    # NOT released: releasing here risks selling the seat twice.
    assert store.status(show.show_id, ["SCR-1:H6"])["SCR-1:H6"] is SeatStatus.HELD


def test_sweeper_clears_expired_holds_and_is_idempotent() -> None:
    clock = ManualClock()
    store, show = _single_seat_show(clock)
    store.acquire(show.show_id, ["SCR-1:H6"], "user-42", 480.0, "key-A")
    clock.advance(481.0)
    assert store.sweep_expired() == 1
    assert store.sweep_expired() == 0                     # second pass: nothing
    assert show.seat("SCR-1:H6").status is SeatStatus.AVAILABLE
```

**`_single_seat_show` and `_two_seat_show` are three-line helpers** that build a screen, a show and a registered store; they exist so each test states its own point rather than restating the fixture.

---

## 7. Extensibility — the standard follow-ups

### 7.1 "Extend the hold"

Extending is not "reset the deadline": it is a new decision under the same lock. The bound matters — unbounded extension is an indefinite lock, and a user who can extend forever can hold the front row through the whole sale.

```python
def extend(self, show: Show, seat_ids: "list[str]", hold_key: str,
           extra_seconds: float, max_total_seconds: float = 900.0) -> SeatHold:
    """Bounded extension. Note the three refusals, each deliberate:
      - the hold must be live: an expired hold is re-acquired, not extended
      - the total must stay under the cap, or a user locks seats forever
      - the seats must still say HELD by this key, checked under the lock
    """
    with self._holds._lock_for(show.show_id):
        now = self._clock.now()
        hold = self._holds._holds.get(hold_key)
        if hold is None or hold.state is not HoldState.ACTIVE:
            raise HoldNotFoundError(f"hold {hold_key} is not active")
        if hold.has_expired(now):
            raise HoldExpiredError(f"hold {hold_key} expired at {hold.expires_at}")
        current = hold.expires_at - now
        if current + extra_seconds > max_total_seconds:
            raise SeatError(
                f"extension would exceed the {max_total_seconds:.0f}s cap "
                f"({current:.0f}s remaining)"
            )
        new_expiry = hold.expires_at + extra_seconds
        for seat_id in hold.seat_ids:
            show.seat(seat_id).hold_expires_at = new_expiry
        hold.expires_at = new_expiry
        return hold
```

### 7.2 "Best available" auto-assignment

The follow-up: the user says "2 seats, don't care where". This is a **search under the lock**, which is materially more expensive than a check — and the honest answer is that it should search a bounded candidate set rather than the whole screen.

```python
def find_best_available(self, show: Show, count: int, tier: SeatTier,
                        preference: str = "centre") -> "list[str]":
    """Search seats, then hold ATOMICALLY. The search is outside the lock
    (a hint); the hold re-checks under it. Note the retry loop: a hint can be
    invalidated by the time the hold runs, and that is expected, not an error.
    """
    if count <= 0:
        raise SeatError("must request at least one seat")

    for _ in range(3):                                    # a bounded number of tries
        hint = _search_hint(show, count, tier, self._clock.now(), preference)
        if hint is None:
            raise SeatError(f"no {count} contiguous {tier.value} seat(s) available")
        try:
            return self._holds.acquire(
                show.show_id, hint, "auto", self._hold_ttl,
                f"auto:{show.show_id}:{hash(tuple(hint)) & 0xFFFFFFFF}",
            ).seat_ids
        except SeatTakenError:
            continue                                      # the hint was stale
    raise SeatError(f"could not place {count} seat(s) after 3 attempts")


def _search_hint(show: Show, count: int, tier: SeatTier,
                 now: float, preference: str = "centre") -> "list[str] | None":
    """Prefer the centre of a row: real cinemas price and sell that way, and a
    naive first-fit fills the aisles first, which nobody wants.

    This runs OUTSIDE the lock and is therefore only a hint. It must be cheap
    and it must be safe to be wrong — the hold re-checks everything under the
    lock, and a stale hint is an expected outcome rather than an error.
    """
    rows = sorted({seat.row for seat in show.screen.seats.values()})
    for row in rows:
        row_seats = sorted(
            (seat for seat in show.screen.seats.values()
             if seat.row == row and seat.tier is tier),
            key=lambda seat: abs(seat.number - 5),        # distance from centre
        )
        for start in row_seats:
            try:
                window = show.screen.neighbours(start.seat_id, count)
            except SeatError:
                continue                                  # ran off the row end
            if all(show.seat(sid).is_available(now) for sid in window):
                return window
    return None
```

### 7.3 "The hold expires while the payment is in flight"

**The nastiest real case, and the one that decides the design.** A user takes a hold, taps pay, and the gateway takes longer than the remaining TTL. Three plausible policies, and only one is right:

| Policy | Consequence |
|---|---|
| Release the seats when the TTL passes | The payment may succeed afterwards. You refund and the customer is angry — and if the seats were resold, someone else is sitting there |
| Confirm regardless of expiry | The seat may have been resold. Two tickets, one seat |
| **Grace period** | **Keep the hold for `grace` seconds past its deadline while a payment is in flight; the sweeper skips holds marked in-flight.** Correct, and it costs only the grace window of blocked inventory |

```python
class HoldSweeper:
    """The grace period, in the sweeper. A hold marked `payment_in_flight` is
    NOT swept until the grace window passes, and the grace window is bounded —
    otherwise a crashed payment marks the hold in-flight forever."""

    GRACE_SECONDS = 120.0

    def sweep_expired(self, now: float, limit: int = 1_000) -> int:
        swept = 0
        for key, hold in list(self._holds.items()):
            if swept >= limit:
                break
            if hold.state is not HoldState.ACTIVE:
                continue
            deadline = hold.expires_at
            if hold.payment_in_flight:
                deadline += self.GRACE_SECONDS
            if now < deadline:
                continue
            self.release(key)
            swept += 1
        return swept
```

**The line to say:** *"The TTL is not the deadline for a payment already in flight. Releasing a seat whose payment may still succeed is how you sell one seat twice — so an in-flight payment buys a bounded grace window, and the reconciliation job settles the outcome."*

### 7.4 "Seat-map caching"

The read path is far hotter than the write path, and the split from §5.1 is what makes it cacheable. The geometry is immutable, so **the layout is a CDN asset**; only the per-show status is dynamic, and it is a small payload. The failure mode is staleness: a user sees a seat as free, taps it, and the hold fails. **That is acceptable and must be designed for** — the hold is the authority, the map is a hint, and the UI must handle a loss gracefully rather than treating a stale map as a promise.

### 7.5 "Group bookings and corporate blocks"

A corporate block wants a *bundle*: hold 50 seats and convert them one at a time, keeping the block intact. That is a different shape from a per-user hold — the block has an owner who is not the eventual guest, and the conversion is partial. **The design change is that the block is an entity with its own state**, and the seats hold against the block rather than against a single hold key. Worth naming as a distinct model rather than pretending a bigger `seat_ids` list covers it.

---

## 8. Failure modes and edge cases

### 8.1 Failure modes

| # | Failure | Consequence | Handling |
|---|---|---|---|
| 1 | Check-then-act across two critical sections | **Same seat held twice** | One critical section: check-all then write-all |
| 2 | Partial hold when one seat in a set is taken | Stranded seats; a set nobody can book | Raise before the first mutation (§5.3) |
| 3 | Retried acquire creates a second hold | Two leases on one seat | Idempotency on `hold_key`, checked first |
| 4 | Same key reused for different seats | Silent wrong result | `IdempotencyConflict` — refuse loudly |
| 5 | Retried payment callback confirms twice | Double booking, or a crash | `confirm` is idempotent on the booking id |
| 6 | `UNKNOWN` payment treated as failure | Seats released, then charged: sold twice | Three-valued outcome; hold retained (§5.4) |
| 7 | Sweeper clears a `BOOKED` seat | **A sold seat becomes available** | Only `HELD` seats expire; asserted in a test |
| 8 | A confirm races the sweeper | The hold is confirmed and swept | `confirm` re-checks liveness under the lock; a confirmed hold is skipped |
| 9 | A seat key carries a blanket TTL in Redis | A booked seat expires | `PERSIST` on confirm (§5.6) |
| 10 | Two overlapping set holds, unsorted locks | Deadlock | Sorted acquisition: `ORDER BY seat_id` / sorted group order |
| 11 | Optimistic multi-row update without a transaction | A stay with holes: some seats held, some not | Transaction envelope; compensating update as the fallback |
| 12 | Redis failover loses the holds | In-flight customers lose their seats | The DB is the durable truth; holds are rebuilt from `show_seat` rows |
| 13 | Hold expiry mid-payment | Refund after a successful charge | The grace window (§7.3) |
| 14 | Thundering herd when a hold is released | The origin is flattened | Retry backoff with jitter; a waiting room at the architecture level |
| 15 | Duplicate seat ids in one request | Two holds on one seat from one call | Rejected: `len(set(...)) != len(...)` |
| 16 | An empty seat list | A hold on nothing, then a charge for nothing | Rejected at the store |
| 17 | `ttl_seconds` of zero or negative | A hold that expires before it is returned | Rejected at the store |
| 18 | Seat id not in this show | A hold on a seat that does not exist | Checked against the show's seat map |
| 19 | Unbounded hold extension | One user freezes the front row indefinitely | `max_total_seconds` cap (§7.1) |
| 20 | A dead sweeper thread | Unbounded `HELD` rows; nobody notices for weeks | The loop swallows exceptions and continues (§5.8) |

### 8.2 Edge cases worth naming

- **A one-seat hold.** The degenerate case, and the one every test uses because it is the clearest.
- **The whole row.** A hold of every seat in a row is legal; a hold of every seat in the *screen* should probably be refused as a bulk-buy pattern — but that is a policy decision, and it belongs in a policy object rather than in the store.
- **The same seat requested twice in one call.** Refused, not deduplicated: silently deduplicating hides a client bug.
- **A hold that expires and is re-acquired by the same key.** Allowed — it is a new decision, and the seats may be gone.
- **Confirm on a released hold.** Refused with `HoldExpiredError`; the seats are gone and confirming would be a double sale.
- **Release after confirm.** Returns `False`, changes nothing — booked seats are never released by a hold operation.
- **A show whose start time has passed.** Not in this file's scope, but the store should refuse to hold seats for a show that has begun; the check belongs in the service, against the `Clock`.
- **Clock skew between instances.** Why the store uses a monotonic clock and why the DB variant compares against its own `now` rather than trusting a client timestamp.
- **Two sweepers on two instances.** Legal; every operation is idempotent and checks the state it expects.

### 8.3 The numbers

| Parameter | Value | Why |
|---|---|---|
| Hold TTL | 8 minutes (480 s) | Long enough to pay, short enough to recycle |
| Max total hold | 15 minutes | A bounded extension cap |
| Payment grace | 120 s past the deadline | An in-flight payment is not a failed one |
| Sweeper interval | 15 s | Hygiene; correctness does not depend on it |
| Sweeper batch | 1 000 holds | Bounded so a sweep never blocks a booking |
| Optimistic attempts | 3–5 | With exponential backoff and jitter |
| Retry backoff | 50 ms doubling, capped at 500 ms | Plus jitter, or retries synchronise |
| Lock granularity | Per show | Per seat cannot hold a set atomically |
| Redis seat-key TTL | 24 h | A backstop; `PERSIST` on confirm |
| Seat-map cache TTL | 5–15 s | Stale is acceptable; the hold is the authority |

### 8.4 Complexity

| Operation | Cost | Note |
|---|---|---|
| `acquire`, N seats | O(N) under the show lock | One pass to check, one to write |
| `status`, N seats | O(N) | Lazy expiry per seat |
| `confirm`, N seats | O(N) | Same |
| `sweep_expired` | O(holds) bounded by the batch | Idempotent, concurrent-safe |
| Redis `acquire` | O(N), one round trip | The Lua script is atomic |
| DB `acquire` | O(N + log S) | N row locks; the transaction is the cost |

**The line to memorise:** *"Every store operation is O(number of seats in the request) — that is the point of naming the seat set. The cost is not the algorithm, it is the lock hold time, and that is why the seat set must be bounded and the sweeper batched."*

---

## 9. Design patterns used, and why

| Pattern | Where | Why here | What it costs |
|---|---|---|---|
| **State** | `SeatStatus` on `ShowSeat`, with `apply_hold` / `apply_confirm` / `apply_release` / `apply_expiry` | Five states with different legal moves; centralising the transitions is what stops a `BOOKED` seat being released | A table or a set of guarded mutators to maintain |
| **Strategy** | `HoldStore` → `InMemoryHoldStore`, `RedisHoldStore`, `DbHoldStore` | The storage mechanism is the axis that varies; the service must not know which | An interface, three implementations, a wiring point |
| **Repository** | `HoldStore` as a whole | The service talks to a contract, not to a database | An indirection between the domain and the store |
| **Facade** | `BookingService` | Callers get `hold_seats` / `book` / `cancel`, not a store, a gateway and a clock | A layer that must not accumulate state |
| **Saga** *(choreography, not a GoF pattern)* | hold → pay → confirm, with compensation on failure | A distributed transaction across two systems that cannot share one. The compensation is `release` | Explicit failure paths, and a reconciliation job |
| **Idempotency key as part of the model** | `SeatHold.hold_key`, `acquire`, `confirm` | Retries are certain, not hypothetical — a user double-taps, a gateway retries | Every mutating operation must accept and honour a key |
| **Observer** *(rejected)* | — | Considered for seat-map updates; rejected because the map is polled and cached, and a push channel per user per show is a fan-out problem, not an LLD one | — |
| **Singleton** *(rejected)* | — | A global seat store breaks multi-show testing and multi-instance deployment | — |

**State and Saga are the two to defend.** State, because the `BOOKED`-is-not-expired-by-time property is a transition rule and belongs in one place rather than in every caller. Saga, because the honest framing of this problem is that **you cannot have a transaction spanning a payment gateway**, so the design is a sequence with compensations and a reconciliation job for the outcome you cannot know.

---

## 10. Trade-offs

| Choice | Buys | Costs |
|---|---|---|
| `Seat` split from `ShowSeat` | A cacheable immutable layout; a small mutable per-show set | Two objects and a join on every read |
| Per-show lock | Atomic over a set; different shows never contend | A hot show serialises — fixed by architecture, not by the lock |
| All-or-nothing holds | No stranded seats | A large request fails entirely if one seat is taken |
| TTL on every hold | No indefinite locks; automatic recovery | A user mid-payment can lose the seats — mitigated by the grace window |
| Three-valued payment outcome | No double sale on a timeout | A reconciliation job and an `UNKNOWN` state to operate |
| Idempotency keys on everything mutating | Retries are safe | Every caller must generate and carry a key |
| Lazy expiry **and** a sweeper | Correct immediately; a clean table | Two code paths that must agree |
| Redis for holds, the DB for truth | Speed and durability | Two stores to keep consistent, and a failover case |
| `FOR UPDATE` pessimistic by default | Correct, no retries | Row locks held for the transaction; wrong for large sets |
| Sorted lock acquisition | No deadlock | A sort on every multi-seat acquire |
| Confirm `PERSIST`s the Redis key | A sold seat cannot expire | A booking's keys never expire — hence the DB as truth |
| Grace window for in-flight payments | No double sale on a slow gateway | Up to 120 s of blocked inventory per stalled payment |
| Bounded extension | No indefinite lock | A slow user is cut off — which is the intent |

**What changes at 10×:** the lock stops being the answer. At a normal show the per-show lock is uncontended and correct; at a hot show 10 000 users in the selection path is a queueing problem, and the fix is a waiting room in front of the selection flow so the lock is never contended. The class design does not change — `HoldStore` gains a sharded implementation, and the queue lives in front of `BookingService`. **The case study owns that argument**; the class design owns the seat set, the states and the store contract.

---

## 11. Interview Q&A

**Q1 (easy). Why are `Seat` and `ShowSeat` separate objects?**
Because one is immutable geometry and the other is mutable per-show state. Row H seat 12 is a recliner for the life of the building; whether it is free for the 7 pm show on Friday changes constantly. Splitting them means the seat layout is a cacheable asset that never invalidates, and only a small per-show record changes on a booking. Collapsing them makes every booking invalidate the seat map.

**Q2 (easy). What is a hold, and why is there one?**
A TTL'd lease on a set of seats, taken before payment. It exists because payment takes time, and without a lease two users would pay for the same seat and one would need a refund. The TTL exists because a lease without one is permanent — a user who abandons a checkout would freeze those seats forever.

**Q3 (easy). How long should a hold be?**
Eight minutes. It is a tuning number rather than a rule: long enough that a normal payment completes, short enough that an abandoned checkout recycles the seats while the show is still sellable. On a hot show you would shorten it, because the cost of a blocked seat is higher; on a slow one you would lengthen it, because abandonment costs more than contention.

**Q4 (medium). Walk me through the exact race.**
Two threads call `acquire` for the same seat. Both read its status as `AVAILABLE` before either writes, both conclude it is available, and both write a hold. The reads happen before either write, so no conflict is ever observed. It is check-then-act, and the fix is to put the check and the write in one critical section — check *all* the seats, and only then write, so a failure cannot leave a partial hold.

**Q5 (medium). Why per-show locking rather than per-seat?**
Because the request is for a set. With per-seat locks, thread A can hold H5 while thread B holds H6, and both requests fail — a mutual partial hold. The set is not atomic unless one lock covers it. Per-show is the right unit: different shows never contend, and a set within a show is atomic by construction. The cost is that a hot show serialises, which is a real problem with an architectural answer rather than a locking one.

**Q6 (medium). How do you make it idempotent?**
Every mutating operation takes a key, and the first thing it does is check whether that key already has a result. `acquire` with a live hold for the same key returns that hold rather than creating a second; with the same key and *different* seats it raises, because silently returning the first result would hide a client bug. `confirm` with the same booking id returns quietly. `release` returns `False` for an already-released hold. Retries are certain — a user double-taps, a gateway retries a callback — so idempotency is part of the model, which is why the key is a field on `SeatHold`.

**Q7 (medium). Why does a hold expire but a booking not?**
Because a hold is a lease and a booking is a sale. A lease is a temporary reservation of a right; a sale transfers it. Clearing a `BOOKED` seat on a timer is exactly the bug that sells one seat twice, and it follows directly from treating the two as one state with an expiry. In the code, expiry only ever touches `HELD`, and there is a test asserting that a day after a confirm the seat is still `BOOKED` and the sweeper returns zero.

**Q8 (medium). Redis or the database for the hold?**
Both, and for different reasons. Redis is the hot path: one Lua script does the check-all and the write-all atomically in a single round trip with no retry loop, which is what makes it fast under a hot show. The database is the durable truth, and it is where the design survives a Redis failover — losing every hold means every in-flight customer loses their seats. So the hold lives in Redis and the `show_seat` rows are authoritative; a failover rebuilds what it can from the rows. `HoldStore` is an interface precisely so this is a wiring decision rather than a fork in the service.

**Q9 (medium). *Pushback:* "Why not just make the seat rows unique on (show, seat) and let the database reject the second insert?"**
That constraint is necessary and not sufficient, and the difference matters. A uniqueness constraint on `(show_id, seat_id)` in a `booking` table prevents two *bookings* for one seat — good. But the hold is not a booking, it is a lease with a TTL, and a unique constraint cannot express "unique among live holds". So you would need a partial index on unexpired holds, which is possible in Postgres and is genuinely a good belt-and-braces measure. It does not replace the atomic check-and-write, because a constraint gives you an error rather than a decision — you still need to know *which* seats conflicted to tell the user, and you still need the same behaviour for a set of seats rather than one.

**Q10 (medium). *Pushback:* "Payment succeeded but the confirm timed out. What do you do?"**
Confirm again with the same booking id, because confirm is idempotent — the retry either finds the booking already made and returns quietly, or makes it. This is the reason the booking id is generated *before* the confirm call rather than by it: the id has to exist on the retry path, so it cannot be a value the store invents. The general shape is that every step after the first must be safely repeatable with the same key, and that is a design constraint on the whole flow, not a patch on one call.

**Q11 (hard). *Pushback:* "Your `UNKNOWN` outcome leaves the seats held and the customer waiting. Isn't that worse than just releasing them?"**
Releasing is worse, and the asymmetry is the whole argument. If you release and the charge succeeded, you have sold a seat you no longer control — the seat gets resold and two people hold a ticket for it, which is a customer-service disaster with a legal edge. If you keep the hold and the charge failed, you have blocked a seat for at most the grace window and then released it. So the two errors are not symmetric: one is a double sale and one is a short delay. That is why the outcome is three-valued rather than a boolean, and why the reconciliation job exists to settle the unknown cases rather than the request path guessing.

**Q12 (hard). *Pushback:* "The per-show lock serialises a hot show. Doesn't that make the design wrong for a blockbuster?"**
The lock is right and the throughput is an architecture problem. At a normal show the lock is uncontended and correct. At a blockbuster, 10 000 people in the seat-selection path is a queueing problem, and no cleverer lock fixes it — you can partition the seat set into groups with sorted group acquisition, or shard the show, and both reduce contention by a constant factor while adding real complexity. The real fix is a waiting room that admits a bounded number of users into the selection path, so the lock is never contended in the first place. That is what production does, and it is the case study's argument rather than the class design's.

**Q13 (hard). *Pushback:* "In your Redis script, the seat key has a TTL. Doesn't that mean a booked seat eventually expires?"**
It would, and that is why `confirm` calls `PERSIST` on every seat key. The per-seat TTL exists as a backstop so that a leaked hold is bounded rather than permanent, and the moment a hold converts into a booking the TTL must be removed, because a sale does not expire. This is a genuine trap: applying a blanket expiry to every seat key is a common implementation and it silently makes sold seats available again a day later. The test for it is straightforward — confirm a booking, advance the clock by a day, assert the seat still reads `BOOKED`.

**Q14 (medium). How do you test concurrency here?**
A barrier test is the core: two threads, one seat, both released at the same instant by a `threading.Barrier`, asserting exactly one hold succeeded and one got `SeatTakenError`. Then the invariant tests: a failed multi-seat acquire leaves *no* seat held — that is the partial-hold assertion, and it checks the seats rather than the return value. Then the idempotency tests, which are sequential and easy: the same key twice returns the same lease; the same key with different seats raises. And every expiry test uses a manual clock, so the eight-minute TTL is tested in microseconds rather than by sleeping.

**Q15 (hard). What would you do differently at 10× the traffic?**
Nothing structural in the seat model or the store contract. What changes is where the serialisation happens: a waiting room in front of the selection flow so the lock is uncontended, Redis sharded by show with a shard count sized to the peak, and the sweeper partitioned so one instance does not sweep the whole estate. The `HoldStore` interface is what makes that a wiring change rather than a rewrite, which is the payoff for having made the store a Strategy in the first place. Beyond that, the interesting problems become the ones the case study covers — cache staleness, hot keys, the herd at release time — and none of them change the classes.

---

## 12. Cheat-sheet summary

**The model**

```
  Seat       = immutable geometry (row H, seat 12, recliner)
  ShowSeat   = mutable per-show state (AVAILABLE | HELD | BOOKED | BLOCKED | UNAVAILABLE)
  SeatHold   = a TTL'd lease on a SET of seats, keyed by an idempotency key
  Booking    = a confirmed sale. Terminal. NOT affected by time.

  THE SPLIT IS THE DESIGN: geometry is cacheable; only per-show state changes.
```

**The flow**

```
  HOLD ──▶ PAY ──▶ CONFIRM
    │       │
    │       ├── FAILED   → release immediately (the gateway said so)
    │       └── UNKNOWN  → DO NOTHING. Keep the hold. Reconcile later.
    │                       Releasing risks selling the seat twice.
    └── expired → HoldExpiredError, no charge
```

**The race, and the fix**

```
  BROKEN: read status → decide → write        ← two threads both read AVAILABLE
  FIXED:  lock(show) → check ALL → write ALL  ← one critical section

  Check ALL first, raise BEFORE the first write. Then a partial hold is
  unrepresentable rather than merely unlikely.
```

**Locking choices**

| Granularity | Verdict |
|---|---|
| One global lock | Every show in the country serialises. Wrong. |
| **One per show** | **Atomic over a set; shows never contend. Right.** |
| One per seat | Cannot hold a set atomically → mutual partial holds. Wrong. |
| Sorted per-seat / per-group | The compromise for large sets; sorted acquisition prevents deadlock |

**The invariants to assert**

1. Every `(show, seat)` has **at most one live hold** and **at most one booking**.
2. **A booking never expires.** Only `HELD` is touched by time.
3. A failed multi-seat acquire leaves **no** seat held.
4. `acquire` / `confirm` / `release` are **idempotent** on their key.
5. A confirmed Redis seat key has **no TTL** (`PERSIST`).

**Redis versus the database**

| | Redis | Database |
|---|---|---|
| Atomicity | A Lua script: one round trip, no retry loop | A transaction with `SELECT … FOR UPDATE` |
| Speed | The hot path | Slower per operation |
| Durability | Lost on a failover | **The durable truth** |
| Read path | Lazy expiry via the stored deadline | Lazy expiry via the deadline column |
| Confirm | `HSET status=BOOKED` + **`PERSIST`** | `UPDATE … SET status='BOOKED'` |

**The numbers**

| Parameter | Value |
|---|---|
| Hold TTL | 8 min (480 s) |
| Max total hold | 15 min |
| Payment grace | 120 s past the deadline |
| Sweeper interval / batch | 15 s / 1 000 |
| Optimistic attempts | 3–5, exponential backoff **with jitter** |
| Lock granularity | Per show |
| Redis seat-key TTL | 24 h — a backstop; `PERSIST` on confirm |
| Seat-map cache TTL | 5–15 s, stale is acceptable |

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "Two users, one seat" | "Check-then-act. One critical section: check all, write all, raise before the first write." |
| "Why not per-seat locks?" | "A set is not atomic under per-seat locks — mutual partial holds." |
| "Redis or DB?" | "Both: Redis is the hot path via Lua, the DB is the truth that survives a failover." |
| "Payment timed out" | "Keep the hold. `UNKNOWN`, not failure — releasing risks a double sale." |
| "Make it idempotent" | "A key on every mutating call; the hold key is a field on `SeatHold`, checked first." |
| "Hold expires mid-payment" | "A bounded grace window; the sweeper skips in-flight payments." |
| "Hot show, lock contended" | "Not a lock problem — a waiting room. The class design does not change." |
| "Prove it" | "Barrier test for two threads on one seat; partial-hold assertion; booking-outlives-TTL test." |

---

## Cross-links

- **The system treatment of this exact problem** — capacity, API, the four concurrency options as architecture, the waiting room, seat-map staleness, Redis failover, the numbers for a hot show: [BookMyShow seat booking](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md)
- The race mechanics this file assumes, and the citation at **[9:15]–[9:18]**: [Concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md)
- The same race over a *point* resource: [Parking lot](01-parking-lot.md) · over a *range* resource: [Hotel management](03-hotel-management.md) · and the data structure the seat map caches into: [LRU and LFU cache](02-lru-lfu-cache.md)
- The class concepts behind this design: [OOP principles and SOLID](../02-oop-principles-and-solid.md) · [Design patterns](../03-design-patterns.md) · [UML and class diagrams](../05-uml-and-class-diagrams.md) · [The machine-coding framework](../06-machine-coding-interview-framework.md) · [What is LLD](../01-what-is-lld-and-how-to-answer.md)
- Where the class view ends: [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md) · [Redis deep dive](../../HLD/03-scaling/03-redis-deep-dive.md) · [Caching strategies and layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md) · [Cache invalidation and consistency](../../HLD/03-scaling/04-cache-invalidation-and-consistency.md)
- Other problems in this pillar: [Elevator system](05-elevator-system.md) · [Splitwise](06-splitwise.md) · [ATM machine](07-atm-machine.md) · [Library management](08-library-management.md)
