# LLD Problem: Hotel Management System

> **Grounding note.** The **roadmap transcript (T1)** names this problem and nothing more. At **[9:22]**, in the syllabus list for the Low-Level Design module that begins at [9:09] (*"after that we'll start Low Level Design"*), it asks *"how is a hotel management system built? what is booking?"* — and that is the entire treatment. There is no hotel model, no date logic, no code, and no answer. **Everything below is standard industry practice**, and the two things that make this problem worth asking are both mine to justify rather than the transcript's to supply: the **half-open date-range interval**, which is the single decision the whole design turns on, and the **double-booking race**, which is the same class of bug as the parking-lot hold in [the parking-lot problem](01-parking-lot.md) but with a *range* instead of a *point*, which makes it materially harder. For the general mechanics of the race — locks, CAS, optimistic versus pessimistic, lock ordering — read [concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md) first; this file assumes it and spends its budget on the domain.

---

## 1. Requirement clarification

### 1.1 Questions to ask

| # | Question | Why it matters | Default if they will not say |
|---|---|---|---|
| 1 | Is a booking for **one room** or can it span several? | A multi-room booking needs an all-or-nothing hold; a single-room one does not | Single room, with the multi-room extension in §7.2 |
| 2 | **Are dates inclusive?** Is a checkout on the 10th free on the 10th? | **This is the question that decides the whole design** | Half-open `[check_in, check_out)` — the checkout night is free |
| 3 | Can a guest book a room type without picking a room number? | Decides whether you assign on booking or at check-in | Assign at check-in; inventory is held against the *type* |
| 4 | Overbooking allowed? | Real hotels oversell deliberately; that is a policy, not a bug | Configurable per room type, default 0 % |
| 5 | Cancellation and refund policy? | Determines whether a cancelled booking frees inventory immediately or on a schedule | Free cancellation up to 24 h before check-in; the extension in §7.5 |
| 6 | Does housekeeping or maintenance block inventory? | A room out of service is not sellable — a fact the calendar must know | Yes: `OUT_OF_SERVICE` rooms are excluded from availability |
| 7 | Pricing: flat, or seasonal, or dynamic? | Determines whether price is a property of the room type or a function of the dates | Seasonal, with weekday/weekend and length-of-stay modifiers |
| 8 | Timezone? | A hotel has a local date; the server may not | The property is the source of truth for dates; store dates, not instants |
| 9 | Multi-property? | Whether there is a `Property` layer above `Room` | Single property, with the multi-property extension in §7.4 |
| 10 | What happens on a no-show? | Determines whether a reservation can expire without an explicit cancel | The extension in §7.6 |

### 1.2 Defaults to state out loud

```
  DATES:      half-open [check_in, check_out). A stay from the 5th to the 8th
              occupies the nights of the 5th, 6th and 7th. The 8th is free for
              the next guest. THIS IS THE CENTRAL DECISION (§5.1).
  INVENTORY:  held against the ROOM TYPE, not a room number. A room number is
              assigned at check-in. This is how real hotels work, and it is why
              an upgrade is possible without changing the booking.
  SCOPE:      one property, N room types, M rooms, date-ranged reservations.
  PRICING:    per-night rate, integer minor units, never float.
  THREADING:  the reserve path is atomic — see §6. This is the hard part.
  OUT:        payments, refunds, channel/OTA sync, loyalty, housekeeping rota,
              the guest-facing UI.
```

**The one sentence that frames the problem:** *"Availability is not a property of a room — it is a property of a room **and a date range**. Everything hard about this problem comes from that 'and'."*

---

## 2. Core entities & responsibilities

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `DateRange` | Represent a half-open span of nights | start, end | Know about rooms or money |
| `RoomType` | A class of room: size, bed, base rate | code, capacity, base rate | Know which rooms exist |
| `Room` | A physical, numbered room | number, type, housekeeping state | Know about reservations |
| `HousekeepingState` | Whether a room can be sold | an enum | Know about dates |
| `Guest` | A person | id, name, contact | Hold bookings; bookings hold guests |
| `Reservation` | A commitment: *this* guest, *this* room type, *these* dates | id, guest, type, span, status, rate | Decide availability |
| `ReservationStatus` | The lifecycle of the commitment | an enum | — |
| `RatePlan` *(interface)* | Price a stay | — | Check availability |
| `RoomCalendar` | The non-overlapping intervals per room | sorted disjoint intervals | Price or take payment |
| `AvailabilityService` | Answer "is a room of type T free on [a,b)?" **atomically** | a calendar per room, a lock per room type | Price the stay |
| `Hotel` | Orchestrate reserve / check-in / check-out / cancel | the services | Implement them |
| `Clock` *(interface)* | Supply "today" and "now" | — | — |

**Two decisions to state, because they are the design:**

**First, inventory is held against the *type*, not the room.** A guest books "a Deluxe room for the 5th to the 8th", not "room 412". The room number is assigned at check-in. This is not a simplification — it is what real hotels do, and it buys three things: an upgrade is free (assign a better room at check-in), housekeeping can shuffle rooms, and the availability query is against a count rather than a specific unit. **The cost is that availability is a count, not a boolean**, and a count is much harder to reason about under concurrency than a single unit's state.

**Second, `Reservation` does not know its room number until check-in.** A nullable `room_id` that is set at check-in. An interviewer who asks "what if I want to guarantee room 412?" gets the honest answer: that is a different product (a room-specific booking), it needs a per-room calendar instead of a per-type count, and it is a legitimate follow-up rather than a requirement.

---

## 3. Class diagram

```
   ┌────────────────────┐        ┌─────────────────────┐
   │     DateRange      │        │     RoomType        │
   ├────────────────────┤        ├─────────────────────┤
   │ - start: date      │        │ - code: str         │
   │ - end: date        │        │ - capacity: int     │
   ├────────────────────┤        │ - base_rate: Money  │
   │ + nights() -> int  │        │ - total_rooms: int  │
   │ + overlaps(o)      │        │ - overbook_pct: int │
   │ + contains(d)      │        └──────────┬──────────┘
   │ + dates() -> list  │                   │ 1
   └─────────┬──────────┘                   │
             │ *                            │ *
             │                    ┌─────────▼──────────┐
             │                    │       Room         │
             │                    ├────────────────────┤
             │                    │ - number: str      │
             │                    │ - type: RoomType   │
             │                    │ - housekeeping     │
             │                    └────────────────────┘
             │
             │  ┌──────────────────────┐         ┌───────────────────────┐
             └─▶│     Reservation      │         │   ReservationStatus   │
                ├──────────────────────┤         ├───────────────────────┤
                │ - id: str            │────────▶│ PENDING               │
                │ - guest: Guest       │ status  │ CONFIRMED             │
                │ - type: RoomType     │         │ CHECKED_IN            │
                │ - span: DateRange    │         │ CHECKED_OUT           │
                │ - room: Room | None  │         │ CANCELLED             │
                │ - total: Money       │         │ NO_SHOW               │
                │ - created_at: float  │         └───────────────────────┘
                └──────────────────────┘
                          ▲
                          │ *  (the reservations that exist)
                          │
   ┌──────────────────────┴──────────────────────────────────┐
   │                  AvailabilityService                    │
   ├─────────────────────────────────────────────────────────┤
   │ - calendars: dict[str, RoomCalendar]   # room_id → cal  │
   │ - locks: dict[str, Lock]               # room_TYPE → lk │
   ├─────────────────────────────────────────────────────────┤
   │ + available_count(type, span) -> int                    │
   │ + reserve(type, span) -> str | None   ← ATOMIC hold     │
   │ + release(room_id, span) -> None                        │
   └─────────────────────────────────────────────────────────┘
                          ▲
                          │ used by
   ┌──────────────────────┴──────────────────────────────────┐
   │                        Hotel                            │
   ├─────────────────────────────────────────────────────────┤
   │ + book(guest, type, span) -> Reservation                │
   │ + check_in(reservation_id, room_id) -> Reservation      │
   │ + check_out(reservation_id) -> Money                    │
   │ + cancel(reservation_id) -> Money                       │
   └─────────────────────────────────────────────────────────┘
```

**What the diagram states:**

- `Reservation` holds a `DateRange` by composition — a reservation without dates is meaningless.
- `Reservation.room` is **nullable**, and that nullability is the design: the type is the booking, the room is an assignment.
- `AvailabilityService` has **one lock per room type**, drawn as a map. That is where atomicity lives, and putting it in the diagram is how you show you thought about it.
- `Hotel` orchestrates and owns no state of its own — it is a facade over services.

---

## 4. Interfaces and enums

```python
from abc import ABC, abstractmethod
from datetime import date, timedelta
from enum import Enum


class ReservationStatus(Enum):
    """The lifecycle. Transitions are enforced, not decorative — see §5.4."""

    PENDING = "pending"            # held, awaiting confirmation
    CONFIRMED = "confirmed"        # committed, occupies inventory
    CHECKED_IN = "checked_in"      # the guest is in the room
    CHECKED_OUT = "checked_out"    # terminal, inventory freed
    CANCELLED = "cancelled"        # terminal, inventory freed
    NO_SHOW = "no_show"            # terminal, inventory freed


class HousekeepingStatus(Enum):
    """Whether a room can be sold or occupied. Note this is an OPERATIONAL
    state, orthogonal to the reservation state — a room can be CLEAN while a
    reservation is CONFIRMED, and OCCUPIED while one is CHECKED_IN."""

    VACANT_CLEAN = "vacant_clean"
    VACANT_DIRTY = "vacant_dirty"
    OCCUPIED = "occupied"
    OUT_OF_SERVICE = "out_of_service"      # not sellable at all


class BookingError(RuntimeError):
    """Base for domain failures. Distinct from a programming error."""


class NoAvailabilityError(BookingError):
    pass


class InvalidTransitionError(BookingError):
    pass


class RatePlan(ABC):
    """The pricing axis. Prices a stay; never looks at availability."""

    @abstractmethod
    def price(self, room_type: "RoomType", span: "DateRange") -> "Money":
        """Total for the whole stay, in minor units."""


class Clock(ABC):
    """Injected so that 'is this date in the past?' and 'has the hold expired?'
    are testable without touching the system clock or sleeping."""

    @abstractmethod
    def today(self) -> date: ...

    @abstractmethod
    def now(self) -> float:
        """Monotonic seconds. For hold expiry, not for dates."""
```

**Why time needs two methods and not one.** Dates and durations are different things. `today()` answers "is this booking in the past?" and must be the *property's local date* — a hotel in Auckland and a server in Virginia disagree about what day it is. `now()` answers "has this hold expired?" and must be a monotonic clock, because a wall clock can jump backwards and a hold that expires early or late is a real bug. **Collapsing these into one `now()` returning a datetime is a common and consequential mistake**, and it is worth naming.

---

## 5. Complete working code

**Language note:** Python 3.11+, standard library only. The design is language-neutral; in Java or Go the same shapes appear with an explicit lock per room type. Every block below is complete.

### 5.1 The date range — the decision the whole design rests on

```
  HALF-OPEN [check_in, check_out).  A stay 5th → 8th:

     night of 5th   night of 6th   night of 7th   night of 8th
     ├──────────────┼──────────────┼──────────────┤
     [5, 8) covers  ↑              ↑              ↑  and STOPS here

     Guest A departs on the 8th. Guest B arrives on the 8th.
     No overlap: A's range is [5, 8), B's is [8, 11). 8 < 8 is false.
                                              ↑ this is the whole trick

  OVERLAP TEST for half-open ranges:
     a.start < b.end  AND  b.start < a.end
  Two conditions. No special cases. No off-by-one on the boundary night.
```

```python
class DateRange:
    """A half-open night range: [start, end).

    Why half-open and not inclusive: with inclusive dates, 'overlap' needs
    special handling for the checkout night, every query needs a +1/-1 dance,
    and the classic bug — a guest arriving on the day another departs being
    told the room is taken — appears. Half-open makes the checkout night
    simply not part of the range, and the overlap test a clean two-clause
    comparison. This is the same convention as Python slicing, and the same
    reason: the boundary belongs to exactly one side.
    """

    __slots__ = ("start", "end")

    def __init__(self, start: date, end: date) -> None:
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        self.start = start
        self.end = end

    def nights(self) -> int:
        return (self.end - self.start).days

    def is_empty(self) -> bool:
        """A zero-night stay. Legal to construct, never legal to book."""
        return self.start == self.end

    def contains_night(self, night: date) -> bool:
        """Half-open: the checkout date is NOT contained."""
        return self.start <= night < self.end

    def overlaps(self, other: "DateRange") -> bool:
        """The two-clause test. Read it as: I start before you finish, and you
        start before I finish."""
        return self.start < other.end and other.start < self.end

    def nights_iter(self) -> "list[date]":
        return [self.start + timedelta(days=i) for i in range(self.nights())]

    def shift(self, days: int) -> "DateRange":
        delta = timedelta(days=days)
        return DateRange(self.start + delta, self.end + delta)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DateRange):
            return NotImplemented
        return (self.start, self.end) == (other.start, other.end)

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __repr__(self) -> str:
        return f"DateRange({self.start.isoformat()}, {self.end.isoformat()})"


def merge_ranges(ranges: "list[DateRange]") -> "list[DateRange]":
    """Union of a set of ranges, for the housekeeping/maintenance block-out.
    Sorted sweep; adjacent ranges merge (an end equal to the next start is
    adjacent under half-open semantics, so it merges)."""
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: r.start)
    merged = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        if current.start <= last.end:                     # <= not < : adjacency merges
            if current.end > last.end:
                merged[-1] = DateRange(last.start, current.end)
        else:
            merged.append(current)
    return merged
```

**Why `overlaps` is two clauses and not three.** Under inclusive dates the test needs a boundary clause, and getting it wrong is *the* classic hotel-booking bug — two guests confirmed for the same room on the turnover night. Under half-open dates there is nothing to get wrong, because the boundary night belongs to exactly one range. **This is worth saying out loud in the interview**, because it is the difference between knowing the convention and knowing why it exists.

### 5.2 The calendar — non-overlapping intervals, found in O(log n)

**The naive availability check:** scan every reservation for the room and test overlap. O(reservations). For a room with three years of history that is thousands of tests per query, and the query runs on every search.

**The good structure:** per room, a list of intervals kept **sorted by start and provably disjoint**. Because they are disjoint, a `bisect` on the start dates narrows the candidates to **two** — which makes the check O(log n) with a tiny constant.

```python
import bisect


class RoomCalendar:
    """Non-overlapping intervals for ONE room, sorted by start date.

    The invariant that makes this fast: intervals never overlap. Given that,
    a bisect to the insertion point leaves at most two candidates — the one
    starting at or before the new start, and the one starting at or after it.
    Every later interval starts at or after the candidate's END, so it cannot
    overlap either. O(log n) with two comparisons.
    """

    def __init__(self, room_id: str) -> None:
        self.room_id = room_id
        self._intervals: list[DateRange] = []

    def __len__(self) -> int:
        return len(self._intervals)

    def _candidates(self, span: DateRange) -> "list[DateRange]":
        starts = [r.start for r in self._intervals]
        index = bisect.bisect_left(starts, span.start)
        out: list[DateRange] = []
        if index > 0:
            out.append(self._intervals[index - 1])
        if index < len(self._intervals):
            out.append(self._intervals[index])
        return out

    def conflicts(self, span: DateRange) -> "DateRange | None":
        for candidate in self._candidates(span):
            if candidate.overlaps(span):
                return candidate
        return None

    def is_free(self, span: DateRange) -> bool:
        return self.conflicts(span) is None

    def add(self, span: DateRange) -> None:
        if self.conflicts(span) is not None:
            raise BookingError(f"room {self.room_id} is not free for {span}")
        starts = [r.start for r in self._intervals]
        index = bisect.bisect_left(starts, span.start)
        self._intervals.insert(index, span)

    def remove(self, span: DateRange) -> bool:
        for i, existing in enumerate(self._intervals):
            if existing == span:
                del self._intervals[i]
                return True
        return False

    def blocked_nights(self) -> int:
        return sum(r.nights() for r in self._intervals)

    def check_invariants(self) -> None:
        """Disjoint and sorted. Worth asserting: a violation here means a
        double booking, which is the failure this entire design exists to
        prevent."""
        for i in range(1, len(self._intervals)):
            previous = self._intervals[i - 1]
            current = self._intervals[i]
            if previous.end > current.start:
                raise BookingError(
                    f"overlapping intervals on room {self.room_id}: "
                    f"{previous} and {current}"
                )
```

**Why a per-room calendar at all, when inventory is held against the type?** Because the *held* inventory is a count per type, but the *assigned* inventory is per room. The count is what you sell; the calendar is what you assign from at check-in and what blocks a room for maintenance. Keeping both is the honest design, and the relationship between them — the count is derived from the calendars, subject to the overbooking allowance — is the thing to explain.

### 5.3 Availability as a count, and the atomic hold

```python
class AvailabilityService:
    """The atomic core. Answers 'how many rooms of type T are free on [a,b)?'
    and performs the check-and-hold as ONE operation.

    The lock granularity is per ROOM TYPE, not global and not per room:
      - global      → every booking in the hotel serialises. Wrong.
      - per room    → cannot answer a type-level count atomically, because the
                      count spans rooms. Wrong for this model.
      - per type    → two bookings for different types never contend, and a
                      booking for one type is atomic. Right.
    """

    def __init__(self, rooms: "list[Room]") -> None:
        self._rooms_by_type: dict[str, list[Room]] = {}
        for room in rooms:
            self._rooms_by_type.setdefault(room.type.code, []).append(room)
        self._calendars: dict[str, RoomCalendar] = {
            room.number: RoomCalendar(room.number) for room in rooms
        }
        self._blocks: dict[str, list[DateRange]] = {room.number: [] for room in rooms}
        self._locks: dict[str, threading.Lock] = {
            code: threading.Lock() for code in self._rooms_by_type
        }

    def _sellable_rooms(self, room_type: RoomType) -> "list[Room]":
        """OUT_OF_SERVICE rooms are never sellable, whatever the calendar says."""
        return [
            room for room in self._rooms_by_type.get(room_type.code, [])
            if room.housekeeping is not HousekeepingStatus.OUT_OF_SERVICE
        ]

    def _free_rooms(self, room_type: RoomType, span: DateRange) -> "list[Room]":
        free: list[Room] = []
        for room in self._sellable_rooms(room_type):
            calendar = self._calendars[room.number]
            if calendar.is_free(span) and not any(b.overlaps(span) for b in self._blocks[room.number]):
                free.append(room)
        return free

    def available_count(self, room_type: RoomType, span: DateRange) -> int:
        """Advisory: a fast answer for the search page. Not a guarantee —
        something can be booked between this call and the hold."""
        if span.is_empty() or span.nights() <= 0:
            return 0
        with self._locks[room_type.code]:
            free = len(self._free_rooms(room_type, span))
            # The overbooking allowance: the hotel deliberately sells up to
            # overbook_pct more than it has, betting on cancellations. It is a
            # POLICY, exposed as a number, not an accident.
            allowance = int(len(self._sellable_rooms(room_type)) * room_type.overbook_pct / 100)
            return free + allowance

    def hold(self, room_type: RoomType, span: DateRange) -> "Room | None":
        """Check-and-take, ATOMICALLY. Returns the room reserved, or None.

        THE CRITICAL PROPERTY: the free-room search and the calendar write
        happen under the SAME lock acquisition. Splitting them — search, then
        lock, then write — is the double-booking bug, and it is exactly the
        check-then-act race from the concurrency file.
        """
        if span.is_empty() or span.nights() <= 0:
            raise BookingError("a stay must be at least one night")

        with self._locks[room_type.code]:
            free = self._free_rooms(room_type, span)
            if free:
                chosen = free[0]
                self._calendars[chosen.number].add(span)
                return chosen

            # Nothing free. Does the overbooking allowance permit one more?
            sellable = self._sellable_rooms(room_type)
            current = sum(
                1 for room in sellable
                if self._calendars[room.number].conflicts(span) is not None
            )
            allowance = int(len(sellable) * room_type.overbook_pct / 100)
            if current < len(sellable) + allowance:
                # Oversold deliberately: no room yet, assigned at check-in.
                return OVERSOLD
            return None

    def release(self, room_number: str, span: DateRange) -> bool:
        """Free the calendar entry. Idempotent: a second release returns False
        rather than raising, because a retried cancel must not corrupt state."""
        room = next(
            (r for rooms in self._rooms_by_type.values() for r in rooms
             if r.number == room_number),
            None,
        )
        if room is None:
            raise BookingError(f"unknown room {room_number}")
        with self._locks[room.type.code]:
            return self._calendars[room_number].remove(span)

    def block(self, room_number: str, span: DateRange, reason: str) -> None:
        """Maintenance. The room is not sellable for these nights."""
        room = next(
            (r for rooms in self._rooms_by_type.values() for r in rooms
             if r.number == room_number),
            None,
        )
        if room is None:
            raise BookingError(f"unknown room {room_number}")
        with self._locks[room.type.code]:
            if self._calendars[room_number].conflicts(span) is not None:
                raise BookingError(
                    f"room {room_number} is occupied for part of {span}; "
                    f"move the guest before blocking it"
                )
            self._blocks[room_number].append(span)

    def total_nights_sold(self, room_type: RoomType, span: DateRange) -> int:
        """Occupancy over a range — the number a revenue manager actually wants.
        Counts NIGHTS, not reservations: a 5-night stay is 5 room-nights."""
        with self._locks[room_type.code]:
            return sum(
                self._calendars[room.number].blocked_nights()
                for room in self._rooms_by_type.get(room_type.code, [])
            )
```

**`OVERSOLD` is a sentinel, not a room** — the reservation exists but no room number is assigned. Making that explicit in the return type is better than returning `None` and having the caller guess whether the booking succeeded.

```python
class _Oversold:
    """No room assigned yet: the overbooking allowance permitted the sale."""

    _instance = None

    def __new__(cls) -> "_Oversold":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "OVERSOLD"


OVERSOLD = _Oversold()
```

### 5.4 The reservation state machine

```
                    ┌───────────┐
                    │  PENDING  │   held, awaiting confirmation
                    └─────┬─────┘
              confirm ────┤──── cancel ──────────────┐
                          ▼                          │
                    ┌───────────┐                    │
        ┌──────────▶│ CONFIRMED │── cancel ──────────┤
        │           └─────┬─────┘                    │
        │                 │ check_in                 │
        │                 ▼                          ▼
        │           ┌────────────┐            ┌─────────────┐
        │           │ CHECKED_IN │            │  CANCELLED  │  terminal
        │           └──────┬─────┘            └─────────────┘
        │                  │ check_out
        │  no_show         ▼
        │           ┌─────────────┐          ┌───────────┐
        └───────────│ CHECKED_OUT │          │  NO_SHOW  │  terminal
     (only from     └─────────────┘          └───────────┘
      CONFIRMED)         terminal

  ILLEGAL, and each must be refused with a reason:
    CHECKED_OUT → CHECKED_IN    the stay ended; a new booking, not a transition
    CANCELLED   → anything      terminal is terminal
    PENDING     → CHECKED_IN    cannot occupy an unconfirmed booking
    NO_SHOW     → CHECKED_IN    the window closed
```

```python
class Reservation:
    """A commitment. Note two things that are easy to get wrong:

    1. The transitions are a TABLE, not scattered if-statements. A table is
       reviewable, testable, and impossible to leave a gap in.
    2. `check_out` FREES INVENTORY, and freeing is the operation people forget.
       A booking that never releases its nights silently shrinks the hotel.
    """

    ALLOWED: dict[ReservationStatus, frozenset[ReservationStatus]] = {
        ReservationStatus.PENDING: frozenset({
            ReservationStatus.CONFIRMED,
            ReservationStatus.CANCELLED,
        }),
        ReservationStatus.CONFIRMED: frozenset({
            ReservationStatus.CHECKED_IN,
            ReservationStatus.CANCELLED,
            ReservationStatus.NO_SHOW,
        }),
        ReservationStatus.CHECKED_IN: frozenset({ReservationStatus.CHECKED_OUT}),
        ReservationStatus.CHECKED_OUT: frozenset(),
        ReservationStatus.CANCELLED: frozenset(),
        ReservationStatus.NO_SHOW: frozenset(),
    }

    def __init__(self, reservation_id: str, guest: "Guest", room_type: RoomType,
                 span: DateRange, total: Money, room_number: "str | None" = None,
                 created_at: float = 0.0) -> None:
        self.id = reservation_id
        self.guest = guest
        self.room_type = room_type
        self.span = span
        self.total = total
        self.room_number = room_number
        self.created_at = created_at
        self._status = ReservationStatus.PENDING

    @property
    def status(self) -> ReservationStatus:
        return self._status

    def is_terminal(self) -> bool:
        return not self.ALLOWED[self._status]

    def occupies_inventory(self) -> bool:
        """Which states hold nights. CANCELLED and NO_SHOW do not — that is the
        whole reason they exist as distinct states rather than a generic 'done'."""
        return self._status in (
            ReservationStatus.PENDING,
            ReservationStatus.CONFIRMED,
            ReservationStatus.CHECKED_IN,
        )

    def transition(self, target: ReservationStatus) -> None:
        allowed = self.ALLOWED[self._status]
        if target not in allowed:
            raise InvalidTransitionError(
                f"cannot move a reservation from {self._status.value} to "
                f"{target.value}; allowed: "
                f"{sorted(s.value for s in allowed) or 'none (terminal state)'}"
            )
        self._status = target

    def nights(self) -> int:
        return self.span.nights()

    def __repr__(self) -> str:
        return (f"Reservation({self.id}, {self.guest.name}, "
                f"{self.room_type.code}, {self.span}, {self._status.value})")
```

### 5.5 Pricing — a function of the dates, not of the room

```python
class Money:
    """Integer minor units. Never a float: 0.1 + 0.2 != 0.3, and a hotel that
    loses a cent per booking loses real money at scale. This is the same
    design used in the parking-lot problem."""

    __slots__ = ("minor", "currency")

    def __init__(self, minor: int, currency: str = "USD") -> None:
        if not isinstance(minor, int):
            raise TypeError("Money takes integer minor units, not float")
        self.minor = minor
        self.currency = currency

    def __add__(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(self.minor + other.minor, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(self.minor - other.minor, self.currency)

    def __mul__(self, factor: float) -> "Money":
        if factor < 0:
            raise ValueError("cannot multiply money by a negative factor")
        # Round half-up at the POINT OF MULTIPLICATION, once, so a total is
        # never the sum of separately rounded line items. Rounding twice is
        # how invoices stop adding up.
        return Money(int(self.minor * factor + 0.5), self.currency)

    def _assert_same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError(f"currency mismatch: {self.currency} vs {other.currency}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return (self.minor, self.currency) == (other.minor, other.currency)

    def __repr__(self) -> str:
        sign = "-" if self.minor < 0 else ""
        return f"{sign}{self.currency} {abs(self.minor) // 100}.{abs(self.minor) % 100:02d}"


class Season:
    __slots__ = ("span", "multiplier", "label")

    def __init__(self, span: DateRange, multiplier: float, label: str) -> None:
        if multiplier <= 0:
            raise ValueError("a season multiplier must be positive")
        self.span = span
        self.multiplier = multiplier
        self.label = label


class SeasonalRatePlan(RatePlan):
    """The rate is a property of the NIGHT, not of the stay: a stay spanning a
    weekend and a holiday has three different nightly rates. Pricing per night
    and summing is the only correct approach — pricing the whole stay from the
    check-in date's season gets the total wrong.

    Modifiers, applied per night:
      season multiplier       holidays and high season
      weekend surcharge       Fri and Sat nights
      length-of-stay discount 7+ nights, applied to the STAY not the night
    """

    WEEKEND_DAYS = frozenset({4, 5})       # Friday, Saturday (Monday = 0)
    LONG_STAY_NIGHTS = 7
    LONG_STAY_DISCOUNT = 0.10

    def __init__(self, seasons: "list[Season]", weekend_surcharge: float = 0.0) -> None:
        if weekend_surcharge < 0:
            raise ValueError("a weekend surcharge cannot be negative")
        self._seasons = list(seasons)
        self._weekend_surcharge = weekend_surcharge

    def _multiplier_for(self, night: date) -> float:
        multiplier = 1.0
        for season in self._seasons:
            if season.span.contains_night(night):
                multiplier *= season.multiplier     # overlapping seasons compound
        if night.weekday() in self.WEEKEND_DAYS:
            multiplier *= (1.0 + self._weekend_surcharge)
        return multiplier

    def nightly_breakdown(self, room_type: RoomType, span: DateRange) -> "list[tuple[date, Money]]":
        """Exposed because a guest-facing quote shows the breakdown, and a
        revenue manager reconciles against it. A total alone is not enough."""
        return [
            (night, room_type.base_rate * self._multiplier_for(night))
            for night in span.nights_iter()
        ]

    def price(self, room_type: RoomType, span: DateRange) -> Money:
        if span.nights() <= 0:
            raise BookingError("cannot price a zero-night stay")
        total = Money(0, room_type.base_rate.currency)
        for _, nightly in self.nightly_breakdown(room_type, span):
            total = total + nightly
        if span.nights() >= self.LONG_STAY_NIGHTS:
            total = total * (1.0 - self.LONG_STAY_DISCOUNT)
        return total


class FlatRatePlan(RatePlan):
    """One rate per night regardless of date. The simple case, kept because a
    small property genuinely wants it and because a plan with no seasons is
    the easiest thing to test the rest of the system against."""

    def price(self, room_type: RoomType, span: DateRange) -> Money:
        if span.nights() <= 0:
            raise BookingError("cannot price a zero-night stay")
        return room_type.base_rate * float(span.nights())
```

### 5.6 The hotel — orchestration, and the guard that saves the day

```python
class RoomType:
    __slots__ = ("code", "description", "capacity", "base_rate", "overbook_pct")

    def __init__(self, code: str, description: str, capacity: int,
                 base_rate: Money, overbook_pct: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 <= overbook_pct <= 20:
            raise ValueError("overbook_pct outside the sane range 0-20")
        self.code = code
        self.description = description
        self.capacity = capacity
        self.base_rate = base_rate
        self.overbook_pct = overbook_pct

    def __repr__(self) -> str:
        return f"RoomType({self.code}, {self.base_rate})"


class Room:
    __slots__ = ("number", "type", "housekeeping")

    def __init__(self, number: str, room_type: RoomType,
                 housekeeping: HousekeepingStatus = HousekeepingStatus.VACANT_CLEAN) -> None:
        self.number = number
        self.type = room_type
        self.housekeeping = housekeeping

    def __repr__(self) -> str:
        return f"Room({self.number}, {self.type.code}, {self.housekeeping.value})"


class Guest:
    __slots__ = ("guest_id", "name", "email")

    def __init__(self, guest_id: str, name: str, email: str) -> None:
        self.guest_id = guest_id
        self.name = name
        self.email = email

    def __repr__(self) -> str:
        return f"Guest({self.guest_id}, {self.name})"


class Hotel:
    """Orchestration only. Owns the reservation table and delegates
    availability, pricing and assignment. A facade, not a god object — note
    every method below is a sequence of calls into something else."""

    MIN_NIGHTS = 1
    MAX_NIGHTS = 30
    HORIZON_DAYS = 730                    # two years out is plenty

    def __init__(self, name: str, rooms: "list[Room]", rate_plan: RatePlan,
                 clock: Clock) -> None:
        self.name = name
        self._rate_plan = rate_plan
        self._clock = clock
        self._availability = AvailabilityService(rooms)
        self._rooms = {room.number: room for room in rooms}
        self._by_type: dict[str, list[Room]] = {}
        for room in rooms:
            self._by_type.setdefault(room.type.code, []).append(room)
        self._reservations: dict[str, Reservation] = {}
        self._next_id = 0
        self._id_lock = threading.Lock()
        self.audit: list[str] = []

    # ---- helpers ------------------------------------------------------
    def _new_id(self) -> str:
        with self._id_lock:
            self._next_id += 1
            return f"RES-{self._next_id:06d}"

    def _validate_span(self, span: DateRange) -> None:
        """Every rule about dates, in ONE place. Scattering these across the
        callers is how a system ends up accepting a booking for last Tuesday."""
        if span.is_empty():
            raise BookingError("a reservation must be at least one night")
        if span.nights() < self.MIN_NIGHTS:
            raise BookingError(f"minimum stay is {self.MIN_NIGHTS} night(s)")
        if span.nights() > self.MAX_NIGHTS:
            raise BookingError(
                f"maximum stay is {self.MAX_NIGHTS} nights; {span.nights()} requested"
            )
        today = self._clock.today()
        if span.start < today:
            raise BookingError(f"check-in {span.start} is in the past (today is {today})")
        horizon = today + timedelta(days=self.HORIZON_DAYS)
        if span.end > horizon:
            raise BookingError(f"check-out beyond the booking horizon {horizon}")

    # ---- the four operations -----------------------------------------
    def quote(self, room_type: RoomType, span: DateRange) -> Money:
        """Price without holding anything. Note this does NOT check
        availability — a quote for a sold-out room is still a valid quote."""
        self._validate_span(span)
        return self._rate_plan.price(room_type, span)

    def book(self, guest: Guest, room_type: RoomType, span: DateRange) -> Reservation:
        """The hard path. Order matters and each step is deliberate:

          1. validate the dates            cheap, no lock, fails fast
          2. price the stay                no lock
          3. HOLD atomically               the critical section
          4. create the reservation        after the hold, never before
          5. confirm                       only once everything succeeded

        Reversing 3 and 4 — creating the reservation then holding — leaves
        orphan reservations on every failure. Holding first means the worst
        case is a held night with no reservation, which a sweeper reclaims.
        """
        self._validate_span(span)
        total = self._rate_plan.price(room_type, span)

        held = self._availability.hold(room_type, span)
        if held is None:
            raise NoAvailabilityError(
                f"no {room_type.code} available for {span} "
                f"({span.nights()} night(s))"
            )

        room_number = None if held is OVERSOLD else held.number
        reservation = Reservation(
            reservation_id=self._new_id(),
            guest=guest,
            room_type=room_type,
            span=span,
            total=total,
            room_number=room_number,
            created_at=self._clock.now(),
        )
        try:
            reservation.transition(ReservationStatus.CONFIRMED)
        except InvalidTransitionError:
            # Cannot happen from PENDING, but if the state table ever changes
            # the inventory must not leak. Release and re-raise.
            if room_number is not None:
                self._availability.release(room_number, span)
            raise

        self._reservations[reservation.id] = reservation
        self.audit.append(f"booked {reservation.id} for {guest.name} ({span})")
        return reservation

    def check_in(self, reservation_id: str, room_number: "str | None" = None) -> Reservation:
        """Assign a room and occupy it.

        THE GUARD: if the reservation has no room number (it was oversold) and
        no room is offered, we must not check the guest in — that is the
        walk-the-guest scenario, and it needs a human decision, not a default.
        """
        reservation = self._get(reservation_id)
        if reservation.status is not ReservationStatus.CONFIRMED:
            raise InvalidTransitionError(
                f"cannot check in a reservation in state {reservation.status.value}"
            )

        if room_number is None:
            free = self._availability._free_rooms(reservation.room_type, reservation.span)
            if not free:
                raise NoAvailabilityError(
                    f"{reservation.id} is oversold and no {reservation.room_type.code} "
                    f"is free; walk the guest or upgrade — do not check in"
                )
            room_number = free[0].number

        room = self._rooms.get(room_number)
        if room is None:
            raise BookingError(f"unknown room {room_number}")
        if room.housekeeping is HousekeepingStatus.OUT_OF_SERVICE:
            raise BookingError(f"room {room_number} is out of service")

        # If the room is not the one held, take it now. The held room (if any)
        # must be released, or the hotel silently loses a sellable night.
        if reservation.room_number is not None and reservation.room_number != room_number:
            self._availability.release(reservation.room_number, reservation.span)
            self._availability._calendars[room_number].add(reservation.span)

        reservation.room_number = room_number
        reservation.transition(ReservationStatus.CHECKED_IN)
        room.housekeeping = HousekeepingStatus.OCCUPIED
        self.audit.append(f"checked in {reservation.id} to room {room_number}")
        return reservation

    def check_out(self, reservation_id: str) -> Money:
        """End the stay and FREE THE NIGHTS. The release is the part that is
        easy to forget and expensive to omit."""
        reservation = self._get(reservation_id)
        room_number = reservation.room_number
        reservation.transition(ReservationStatus.CHECKED_OUT)
        if room_number is not None:
            self._availability.release(room_number, reservation.span)
            self._rooms[room_number].housekeeping = HousekeepingStatus.VACANT_DIRTY
        self.audit.append(f"checked out {reservation.id}; folio {reservation.total}")
        return reservation.total

    def cancel(self, reservation_id: str) -> Money:
        """Cancel and release. Returns the refund, which the cancellation policy
        decides — note the policy is NOT in this method; it is a separate
        decision that depends on the clock (§7.5)."""
        reservation = self._get(reservation_id)
        if reservation.is_terminal():
            raise InvalidTransitionError(
                f"{reservation.id} is already {reservation.status.value}"
            )
        reservation.transition(ReservationStatus.CANCELLED)
        if reservation.room_number is not None:
            self._availability.release(reservation.room_number, reservation.span)
        self.audit.append(f"cancelled {reservation.id}")
        return reservation.total                        # no-penalty default

    def mark_no_show(self, reservation_id: str) -> None:
        """The guest never arrived. Frees inventory without a refund."""
        reservation = self._get(reservation_id)
        reservation.transition(ReservationStatus.NO_SHOW)
        if reservation.room_number is not None:
            self._availability.release(reservation.room_number, reservation.span)
        self.audit.append(f"no-show {reservation.id}")

    def _get(self, reservation_id: str) -> Reservation:
        try:
            return self._reservations[reservation_id]
        except KeyError:
            raise BookingError(f"unknown reservation {reservation_id}") from None

    def search(self, span: DateRange, guests: int = 1) -> "list[RoomType]":
        """Which types can take this stay. Advisory only — the hold decides."""
        self._validate_span(span)
        results: list[RoomType] = []
        for code, rooms in self._by_type.items():
            room_type = rooms[0].type
            if room_type.capacity < guests:
                continue
            if self._availability.available_count(room_type, span) > 0:
                results.append(room_type)
        return results
```

### 5.7 It runs

```python
def demo() -> None:
    from datetime import date as _date

    standard = RoomType("STD", "Standard queen", 2, Money(12000))
    deluxe = RoomType("DLX", "Deluxe king", 2, Money(18500), overbook_pct=10)

    rooms = [
        Room("101", standard), Room("102", standard), Room("103", standard),
        Room("201", deluxe), Room("202", deluxe),
        Room("203", deluxe, HousekeepingStatus.OUT_OF_SERVICE),
    ]

    summer = Season(DateRange(_date(2026, 6, 1), _date(2026, 9, 1)), 1.35, "summer")
    holidays = Season(DateRange(_date(2026, 12, 20), _date(2027, 1, 3)), 1.60, "holidays")
    plan = SeasonalRatePlan([summer, holidays], weekend_surcharge=0.15)

    hotel = Hotel("Grand Meridian", rooms, plan, RealClock())

    stay = DateRange(_date(2026, 7, 10), _date(2026, 7, 14))     # 4 nights, summer
    alice = Guest("G-1", "Alice", "alice@example.com")
    bob = Guest("G-2", "Bob", "bob@example.com")

    print("quote:", hotel.quote(deluxe, stay))                   # USD 999.00ish
    reservation = hotel.book(alice, deluxe, stay)
    print(reservation)
    print("room:", reservation.room_number)                      # 201

    print("available deluxe:", hotel._availability.available_count(deluxe, stay))
    second = hotel.book(bob, deluxe, stay)
    print("second booking room:", second.room_number)            # 202

    # A stay that touches the first one's checkout day is FINE.
    turnover = DateRange(_date(2026, 7, 14), _date(2026, 7, 16))
    third = hotel.book(alice, deluxe, turnover)
    print("turnover booking:", third.room_number)                # 201 again — correct

    hotel.check_in(reservation.id)
    print("housekeeping:", rooms[3].housekeeping.value)          # occupied
    folio = hotel.check_out(reservation.id)
    print("folio:", folio)
    print("housekeeping after checkout:", rooms[3].housekeeping.value)   # vacant_dirty
    print("deluxe available again:", hotel._availability.available_count(deluxe, stay))


class RealClock(Clock):
    def today(self) -> date:
        return date.today()

    def now(self) -> float:
        return time.monotonic()


if __name__ == "__main__":
    demo()
```

**The `turnover` booking in the demonstration is the point of the whole file.** It books the same room on the day the previous guest checks out, and it succeeds — because `[10, 14)` and `[14, 16)` do not overlap. Under inclusive dates that booking would be rejected, and a hotel that rejects it loses a night's revenue on every single stay.

---

## 6. Concurrency and thread-safety

### 6.1 The double-booking race

Two guests, one room type with one room left, the same dates, one instant.

```
   t   Thread A (Alice, DLX [Jul 10, Jul 14))     Thread B (Bob, same)
   │
   1   enter hold()
   2   _free_rooms() → [201]      ← sees room 201 free
   3                                              enter hold()
   4                                              _free_rooms() → [201]
   5                                                 ← ALSO sees 201 free
   6   calendars["201"].add(span)
   7                                              calendars["201"].add(span)
   8   return room 201
   9                                              return room 201

   RESULT: two confirmed reservations for room 201 on the same nights.
           The hotel has sold one room twice. At check-in, one guest walks.
```

**Why this is worse than the parking-lot race.** In the parking lot, the resource is a *point* — one spot, one instant. Here the resource is a *range*, so the same room can legitimately be sold to two guests whose ranges do not overlap. The atomicity requirement is therefore not "only one holder per spot" but **"for any room, the set of held ranges is pairwise non-overlapping"** — a stronger and much less obvious invariant. It cannot be enforced by a per-room boolean; it needs the whole range checked and written together.

### 6.2 The fix, and the three levels of it

```python
# ---- LEVEL 1: the naive fix, and WHY it is wrong here -----------------
def hold_naive(self, room_type, span):
    free = self._free_rooms(room_type, span)      # ← lock NOT held
    if not free:
        return None
    with self._locks[room_type.code]:             # ← too late
        self._calendars[free[0].number].add(span)
        return free[0]

# The check and the act are in DIFFERENT critical sections. Between them the
# world changes. This is check-then-act, verbatim — and it is the bug that
# ships, because it looks like it takes a lock.


# ---- LEVEL 2: the correct fix, per-type lock around check AND act -----
def hold(self, room_type, span):
    with self._locks[room_type.code]:             # lock FIRST
        free = self._free_rooms(room_type, span)  # check
        if not free:
            return None
        self._calendars[free[0].number].add(span) # act — same critical section
        return free[0]

# Correct. The cost: every booking for a room type serialises. For a hotel
# that is fine — a property with 300 rooms and 6 types does not have a booking
# throughput problem, and the critical section is microseconds.


# ---- LEVEL 3: if the lock is contended, a per-NIGHT stripe ------------
class StripedCalendarLock:
    """Lock on the NIGHTS, not the room type. Two bookings for the same type
    with disjoint dates take different night locks and never contend — which
    is the common case in a hotel, where most dates are not the same date.

    Acquire in SORTED night order, always. That is the deadlock fix from the
    concurrency file: two multi-night bookings that overlap partially will
    both want a set of nights, and without a global order they deadlock.
    """

    def __init__(self, stripe_count: int = 256) -> None:
        self._stripes = [threading.Lock() for _ in range(stripe_count)]
        self._count = stripe_count

    def _index(self, night: date) -> int:
        return night.toordinal() % self._count

    @contextlib.contextmanager
    def for_span(self, span: DateRange) -> "Iterator[None]":
        indices = sorted({self._index(n) for n in span.nights_iter()})
        acquired: list[threading.Lock] = []
        try:
            for index in indices:
                lock = self._stripes[index]
                lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
```

### 6.3 The optimistic alternative

When the critical section is expensive — a multi-room group booking, or a channel-manager sync where the whole thing is a network round trip — pessimistic locking wastes throughput. The optimistic form writes conditionally and retries.

```python
class OptimisticAvailability:
    """CAS on a per-room-per-night occupancy row. The UPDATE carries the
    condition, so the database decides — and the rowcount tells you whether
    you won. This is real SQL, and it is the version that works across
    processes, which a threading.Lock does not."""

    HOLD_SQL = """
        UPDATE room_night
           SET reservation_id = %(reservation_id)s,
               version        = version + 1
         WHERE room_id       = %(room_id)s
           AND night         = %(night)s
           AND reservation_id IS NULL
           AND version       = %(version)s
    """

    def hold_cas(self, cursor, room_id: str, span: DateRange,
                 reservation_id: str, max_attempts: int = 3) -> bool:
        for attempt in range(max_attempts):
            rows = cursor.execute(
                "SELECT night, version FROM room_night "
                "WHERE room_id = %s AND night = ANY(%s)",
                (room_id, span.nights_iter()),
            ).fetchall()
            if len(rows) != span.nights():
                return False                    # missing rows: not sellable
            if any(row["reservation_id"] is not None for row in rows):
                continue                        # taken; refetch and retry

            version_by_night = {row["night"]: row["version"] for row in rows}
            won = True
            for night in span.nights_iter():
                cursor.execute(
                    self.HOLD_SQL,
                    {
                        "room_id": room_id,
                        "night": night,
                        "reservation_id": reservation_id,
                        "version": version_by_night[night],
                    },
                )
                if cursor.rowcount != 1:
                    won = False
                    # Roll back the nights already taken in THIS attempt, or a
                    # failed attempt leaks a partial hold — a stay that starts
                    # but never ends. This is the bug that makes optimistic
                    # locking dangerous without a transaction.
                    cursor.execute(
                        "UPDATE room_night SET reservation_id = NULL, version = version + 1 "
                        "WHERE room_id = %s AND night = ANY(%s) AND reservation_id = %s",
                        (room_id, span.nights_iter(), reservation_id),
                    )
                    break
            if won:
                return True
        return False
```

**The failure mode written into that comment is the one that matters:** a partially-applied optimistic hold leaves a stay with holes. A transaction is the correct envelope; the compensating update above is the fallback when the store has no transactions. **Either way, the partial hold must be named**, because it is the failure a candidate is most likely to miss.

### 6.4 The decision table

| Situation | Approach | Why |
|---|---|---|
| Single process, one property | Per-type `Lock` around check-and-act | Simple, correct, uncontended at hotel volumes |
| Many dates, few collisions | Night striping, sorted acquisition | Disjoint stays never contend; the deadlock fix is the sort |
| Multi-process or multi-service | Optimistic CAS in the database, or a transaction | A process-local lock cannot span processes |
| Multi-room group booking | Pessimistic, sorted by room id | All-or-nothing across rooms; sorting prevents deadlock |
| High-contention flash sale | Queue, then hold | Serialising on a lock thrashes; a queue makes the ordering explicit |
| Read-side availability search | Advisory only, no lock | It is a hint; the hold is the decision |

### 6.5 Tests that prove the atomicity

```python
def test_two_threads_cannot_both_book_the_last_room() -> None:
    """THE test. Two threads, one free room, same dates, both released at once."""
    room_type = RoomType("STD", "Standard", 2, Money(10000))
    hotel = Hotel("Test", [Room("101", room_type)], FlatRatePlan(), FakeClock())
    stay = DateRange(date(2026, 8, 1), date(2026, 8, 3))

    barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[str] = []
    lock = threading.Lock()

    def book(name: str) -> None:
        guest = Guest(name, name, f"{name}@example.com")
        barrier.wait()                                    # fire together
        try:
            reservation = hotel.book(guest, room_type, stay)
            with lock:
                results.append(reservation.id)
        except NoAvailabilityError as exc:
            with lock:
                results.append(exc)

    threads = [threading.Thread(target=book, args=(n,)) for n in ("alice", "bob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    success = [r for r in results if isinstance(r, str)]
    assert len(success) == 1, f"expected exactly one booking, got {results}"
    assert len(hotel._availability._calendars["101"]) == 1
    hotel._availability._calendars["101"].check_invariants()


def test_non_overlapping_stays_share_a_room() -> None:
    """The invariant is 'no OVERLAP', not 'one holder' — the difference from
    the parking-lot problem, asserted directly."""
    room_type = RoomType("STD", "Standard", 2, Money(10000))
    hotel = Hotel("Test", [Room("101", room_type)], FlatRatePlan(), FakeClock())
    alice, bob = Guest("G1", "Alice", "a@x.com"), Guest("G2", "Bob", "b@x.com")

    first = hotel.book(alice, room_type, DateRange(date(2026, 8, 1), date(2026, 8, 5)))
    # Bob arrives the day Alice leaves. Same room, no conflict.
    second = hotel.book(bob, room_type, DateRange(date(2026, 8, 5), date(2026, 8, 7)))
    assert first.room_number == second.room_number == "101"
    hotel._availability._calendars["101"].check_invariants()


def test_partial_overlap_is_refused() -> None:
    room_type = RoomType("STD", "Standard", 2, Money(10000))
    hotel = Hotel("Test", [Room("101", room_type)], FlatRatePlan(), FakeClock())
    hotel.book(Guest("G1", "Alice", "a@x.com"), room_type,
               DateRange(date(2026, 8, 1), date(2026, 8, 5)))
    with pytest.raises(NoAvailabilityError):
        hotel.book(Guest("G2", "Bob", "b@x.com"), room_type,
                   DateRange(date(2026, 8, 4), date(2026, 8, 6)))   # overlaps the 4th


def test_cancel_frees_the_nights() -> None:
    room_type = RoomType("STD", "Standard", 2, Money(10000))
    hotel = Hotel("Test", [Room("101", room_type)], FlatRatePlan(), FakeClock())
    stay = DateRange(date(2026, 8, 1), date(2026, 8, 3))
    reservation = hotel.book(Guest("G1", "Alice", "a@x.com"), room_type, stay)
    assert hotel._availability.available_count(room_type, stay) == 0
    hotel.cancel(reservation.id)
    assert hotel._availability.available_count(room_type, stay) == 1


def test_overlap_test_boundaries() -> None:
    """Every boundary case, because this is where the two-clause test lives."""
    a = DateRange(date(2026, 8, 1), date(2026, 8, 5))
    assert DateRange(date(2026, 8, 5), date(2026, 8, 7)).overlaps(a) is False  # turnover
    assert DateRange(date(2026, 7, 30), date(2026, 8, 1)).overlaps(a) is False # arrival day
    assert DateRange(date(2026, 8, 4), date(2026, 8, 6)).overlaps(a) is True   # one night
    assert DateRange(date(2026, 8, 2), date(2026, 8, 3)).overlaps(a) is True   # inside
    assert DateRange(date(2026, 7, 1), date(2026, 9, 1)).overlaps(a) is True   # contains
    assert a.overlaps(a) is True                                               # itself
    assert DateRange(date(2026, 8, 1), date(2026, 8, 1)).overlaps(a) is False  # empty
```

---

## 7. Extensibility — the standard follow-ups

### 7.1 "Housekeeping must block a room"

Already in the design as `AvailabilityService.block()` and the `OUT_OF_SERVICE` exclusion, plus the `block()` guard that refuses to block an occupied room. The interesting part is the interaction: **a room going out of service is not a booking operation, it is an inventory operation**, and it must fail loudly if it would strand a guest rather than silently removing a night someone has paid for.

### 7.2 "Book several rooms together"

The follow-up: a group booking for three rooms, all-or-nothing.

```python
def book_group(self, guest: Guest, room_type: RoomType, span: DateRange,
               count: int) -> "list[Reservation]":
    """ALL-OR-NOTHING across N rooms. The two ways to get this wrong:

      1. Hold rooms one at a time and stop on failure without releasing —
         the partial hold, which strands nights nobody can book.
      2. Hold in arbitrary order and deadlock against a concurrent group.

    The fix for both: hold all under ONE critical section, and on any failure
    release everything already taken. Sorting is unnecessary under a single
    lock, but required the moment you move to per-room locks.
    """
    if count <= 0:
        raise BookingError("a group booking needs at least one room")

    with self._availability._locks[room_type.code]:        # ONE section
        taken: list[Room] = []
        for _ in range(count):
            free = self._availability._free_rooms(room_type, span)
            if not free:
                for room in taken:                          # release everything
                    self._availability._calendars[room.number].remove(span)
                raise NoAvailabilityError(
                    f"only {len(taken)} of {count} {room_type.code} rooms available "
                    f"for {span}"
                )
            chosen = free[0]
            self._availability._calendars[chosen.number].add(span)
            taken.append(chosen)

    reservations: list[Reservation] = []
    for room in taken:
        reservation = Reservation(
            reservation_id=self._new_id(), guest=guest, room_type=room_type,
            span=span, total=self._rate_plan.price(room_type, span),
            room_number=room.number, created_at=self._clock.now(),
        )
        reservation.transition(ReservationStatus.CONFIRMED)
        self._reservations[reservation.id] = reservation
        reservations.append(reservation)
    return reservations
```

### 7.3 "Add a waitlist"

When sold out, offer the guest a place in a queue and notify on a cancellation. The design question is *what happens when the notification is ignored* — a waitlist entry that never expires starves everyone behind it. **The answer is a TTL on the offer, exactly like the seat hold** in [BookMyShow seat locking](04-bookmyshow-seat-locking.md): the freed nights are held for the first waitlisted guest for N minutes, then offered to the next.

### 7.4 "Multiple properties"

The change is structural, not cosmetic: dates become property-local (a hotel in Auckland and one in Los Angeles do not share a "today"), inventory is partitioned by property, and the search becomes a cross-property query. **The class that changes most is `Clock`** — a single `today()` is wrong the moment there are two timezones, and `PropertyClock(property)` replacing it is the fix.

### 7.5 "Cancellation policy with refunds"

```python
class CancellationPolicy(ABC):
    @abstractmethod
    def refund(self, reservation: Reservation, now: float) -> Money: ...


class TieredCancellationPolicy(CancellationPolicy):
    """Free until 24 h before check-in, then one night's charge, then nothing.
    The tiers are data, not code — a revenue manager changes them without a
    deployment, which is the whole point of making the policy an object."""

    TIERS: "list[tuple[float, float]]" = [       # (hours_before, refund_fraction)
        (168.0, 1.00),                            # a week out: full refund
        (24.0, 0.50),
        (0.0, 0.00),
    ]

    def refund(self, reservation: Reservation, now: float) -> Money:
        hours_before = self._hours_until_check_in(reservation, now)
        for threshold, fraction in self.TIERS:
            if hours_before >= threshold:
                return reservation.total * fraction
        return Money(0, reservation.total.currency)

    def _hours_until_check_in(self, reservation: Reservation, now: float) -> float:
        return max(0.0, reservation.span.start.toordinal() - _ordinal_today(now)) * 24.0
```

### 7.6 "No-show handling"

Already a terminal state with its own transition. The design questions are the ones candidates miss: **who marks it** (a nightly job that sweeps `CONFIRMED` reservations whose check-in date has passed), **does it refund** (no, that is the point of a no-show fee), and **does it free the remaining nights** (yes — and it must free *all* of them, not just the first, because a no-show means the whole stay is gone).

---

## 8. Failure modes and edge cases

### 8.1 Failure modes

| # | Failure | Consequence | Handling |
|---|---|---|---|
| 1 | Check-and-act split across two critical sections | **Double booking** — two guests, one room | One lock around both (§6.2) |
| 2 | Partial multi-room hold on failure | Stranded nights nobody can book | Release everything on any failure (§7.2) |
| 3 | Checkout does not release the nights | The hotel silently shrinks | `check_out` releases; asserted in a test |
| 4 | Cancel is not idempotent | A retried cancel corrupts state | `release` returns `False`; the reservation state machine refuses a second cancel |
| 5 | Timezone: server date vs property date | A booking for "today" is rejected as past | Store dates; the property supplies `today()` |
| 6 | Wall-clock for hold expiry | A clock jump expires a hold early | `time.monotonic()` for durations |
| 7 | Inclusive dates with a half-open overlap test | Off-by-one on every turnover night | Half-open everywhere; one test per boundary |
| 8 | Float money | Drift; invoices that do not add up | Integer minor units, round once |
| 9 | Rounding each nightly rate then summing | Total differs from the guest's expectation | Round at the point of multiplication, once |
| 10 | Out-of-service room sold | A guest with no room | `_sellable_rooms` excludes it |
| 11 | Out-of-service applied to an occupied room | A guest stranded mid-stay | `block()` refuses unless the room is free |
| 12 | Overbooking beyond the allowance | Unbounded walk-the-guest | `overbook_pct` validated at construction, checked under the lock |
| 13 | Oversold reservation checked in with no room | A guest at the desk with no room | The `check_in` guard raises rather than assigning a default |
| 14 | Availability query cached and trusted | A stale "yes" sold as a guarantee | Documented as advisory; the hold decides |
| 15 | Reservation created before the hold | Orphan reservations on every failure | Hold first, create second, release on failure |
| 16 | Concurrent multi-night holds, unsorted lock order | Deadlock | Sort nights; acquire in order (§6.2) |

### 8.2 Edge cases worth naming

- **Zero-night stay.** `DateRange(d, d)` is constructible and must never be bookable. `is_empty()` guards it.
- **Checkout on the same date as another's check-in.** Legal, and it is the case the half-open convention exists for.
- **A stay that is entirely inside another.** Refused — it overlaps.
- **A stay that contains another.** Refused.
- **A stay of exactly one night** on a date another stay departs. Legal.
- **Booking in the past.** Refused by `_validate_span`.
- **A stay longer than 30 nights.** Refused — a long-stay booking is a tenancy, a different product.
- **The 29th of February.** `timedelta` handles it; a hand-rolled date arithmetic does not, which is why date arithmetic should never be hand-rolled.
- **DST transitions.** Because the model stores *dates* and counts *nights*, not hours, DST is invisible. **This is a benefit of the date-based model and worth pointing out** — a model built on instants would break on the night the clocks change.
- **A room type with zero rooms.** `available_count` returns the overbooking allowance; harmless but worth blocking at construction.
- **Cancelling a reservation that was checked out.** Refused: terminal is terminal.
- **A guest cancelling a stay already begun.** Legal, and the refund policy decides — it should not be free.

### 8.3 Complexity, and what actually costs time

| Operation | Cost | Note |
|---|---|---|
| `DateRange.overlaps` | O(1) | Two comparisons |
| `RoomCalendar.conflicts` | **O(log n)** | Bisect, then two candidates — because intervals are disjoint |
| Naive availability scan | O(reservations) per room | The version to replace |
| `available_count` for a type | O(rooms × log n) | Under the type lock |
| `hold` | O(rooms × log n) | Same, plus one insert |
| Pricing a stay | O(nights) | Must be per-night: seasons change mid-stay |
| `search` across types | O(types × rooms × log n) | Usually 6 types, so trivial |

**The line to memorise:** *"Availability is O(rooms × log n) because each room's calendar is disjoint and sorted, so a bisect narrows the overlap check to two candidates. The naive scan is O(reservations) per room, and the difference matters because the availability query runs on every search, not once per booking."*

---

## 9. Design patterns used, and why

| Pattern | Where | Why here | What it costs |
|---|---|---|---|
| **State** | `Reservation` + `ALLOWED` table | The lifecycle has six states and terminal ones; a table makes the legal moves reviewable, and an illegal move fails with a reason | A table to maintain; a state class per state would be overkill at six |
| **Strategy** | `RatePlan` → `SeasonalRatePlan`, `FlatRatePlan`; `CancellationPolicy` | Pricing and refunds are the axes that vary by property and by season; the follow-ups are exactly these | An interface and a wiring point |
| **Facade** | `Hotel` | Callers get `book` / `check_in` / `check_out`, not five services | A layer that must not grow state |
| **Repository** *(implied)* | `AvailabilityService` | The calendar store is behind one interface; swapping the in-memory store for SQL is a change in one class | An indirection |
| **Strategy** | The lock granularity choice (§6.4) | Per-type vs striped vs optimistic is a choice made once, at construction | Must be understood, not copied |
| **Singleton** *(rejected)* | — | A global `Hotel` breaks multi-property and every test | — |
| **Template Method** *(rejected)* | — | Considered for a `booking()` skeleton; rejected because the steps are already one method and the variability is in `RatePlan` | — |

**The State pattern is the one to justify.** With six states and a handful of transitions, an `ALLOWED` dict is clearer than six classes — but the *pattern* is still State, because the transition legality is centralised and data-driven rather than scattered. **Knowing when the lightweight form of a pattern is correct is a stronger signal than implementing the heavyweight form**, and this is a good place to say so.

---

## 10. Trade-offs

| Choice | Buys | Costs |
|---|---|---|
| Half-open dates | The turnover night works with no special case; overlap is two clauses | Every developer must know the convention — so it goes in the class docstring |
| Inventory held by type, not by room | Free upgrades; housekeeping flexibility; a count-based query | Availability is a count, which is harder under concurrency than a boolean |
| Nullable `room_number` | The oversell case is representable rather than a lie | Every consumer must handle `None` |
| Per-type lock | Correct and simple; types do not contend | Bookings for one type serialise — fine at hotel volumes |
| Night striping | Disjoint stays never contend | Sorted acquisition required to avoid deadlock |
| Optimistic CAS | Scales across processes | Retries; and a partial hold needs a transaction or compensation |
| Per-room sorted calendars | O(log n) conflict checks | Two structures (count + calendars) to keep consistent |
| Overbooking as an explicit number | Deliberate policy, reviewable, set per type | Must be enforced under the lock, and it must be small |
| Per-night pricing | Correct totals across seasons and weekends | O(nights) rather than O(1); needs a breakdown for the quote |
| Money as integer minor units | Exact arithmetic | Conversion at every boundary with the outside world |
| `Clock` with `today()` and `now()` | Testable, timezone-correct | Two methods where one would "do" |
| Advisory availability search | Fast, no lock | Callers must not treat it as a guarantee — documentation debt |

**What changes at 10×:** not the data structures. A hotel is small — hundreds of rooms, thousands of bookings a day — and the per-type lock is nowhere near contention. At 10× the changes are that availability search moves to a read replica or a denormalised count table, the reservation store becomes a database with real transactions, and the OTA channel sync becomes the hard problem. **The in-memory class design is the same at 10×; what changes is where the lock lives, from a `threading.Lock` to a database transaction.**

---

## 11. Interview Q&A

**Q1 (easy). What is the core data model?**
A `Reservation` binds a `Guest`, a `RoomType` and a `DateRange`. Rooms are assigned at check-in, not at booking, so the booking is against a type. Availability is a per-room calendar of non-overlapping date ranges, and the whole design turns on one convention: date ranges are half-open, `[check_in, check_out)`.

**Q2 (easy). Why half-open dates?**
Because the checkout night belongs to exactly one side of the boundary — the next guest's. A stay from the 5th to the 8th occupies the nights of the 5th, 6th and 7th, and the 8th is free. That makes the overlap test two clauses with no special cases: `a.start < b.end and b.start < a.end`. With inclusive dates the same test needs a boundary clause, and getting it wrong produces the classic bug where a guest arriving on a departure day is told the room is taken — which costs a night's revenue on every turnover.

**Q3 (easy). How do you check availability?**
For a room type and a date range, count the rooms of that type whose calendar has no conflicting interval and which are not out of service — plus the overbooking allowance. The naive version scans every reservation for the room; the good version keeps each room's intervals sorted and disjoint, so a bisect narrows the conflict check to two candidates and makes it O(log n) instead of O(reservations). That matters because availability runs on every search, not once per booking.

**Q4 (medium). Walk me through the double-booking race.**
Two threads check availability for the same room type and dates, both see the same free room because neither has taken a lock yet, and both write it into the calendar. Two reservations, one room, same nights. It is check-then-act: the check and the act are in different critical sections, and the window between them is where the second thread gets in.

**Q5 (medium). How do you fix it?**
Put the check and the act under one lock acquisition. The lock granularity is the design decision: a global lock serialises every booking in the hotel, and a per-room lock cannot atomically answer a type-level count, so the right granularity here is per room type. Within the critical section, find a free room and add the interval — no gap. The critical section is microseconds, and a hotel's booking volume is nothing, so contention is a non-issue.

**Q6 (medium). Why is this harder than the parking-lot problem?**
Because the resource is a range, not a point. The parking lot's invariant is "at most one holder per spot", which a per-spot flag enforces. Here the invariant is "for any room, the held ranges are pairwise non-overlapping" — the same room is legitimately sold to two guests whose stays do not overlap. So the atomicity requirement is over the whole range, not over a single unit, and a per-room boolean is not merely insufficient, it is wrong.

**Q7 (medium). What happens when a room goes out of service?**
It is excluded from the sellable set, so no new booking can take it. The guard that matters is the other direction: `block()` refuses to take a room out of service if it has a conflicting reservation, because that would strand a guest who has paid. Taking a room out of service is an inventory operation, not a booking, and it should fail loudly rather than silently remove a night.

**Q8 (medium). How do you handle a group booking for three rooms?**
All-or-nothing under one critical section. Hold all three, and on any failure release everything already taken — the partial hold is the failure to design against, because it strands nights that nobody can book. The alternative, holding one at a time with a lock per room, requires acquiring in a global order to avoid deadlock against a concurrent group booking; under a single type lock that is unnecessary.

**Q9 (medium). *Pushback:* "Why not hold inventory per room from the start, so a reservation names a room?"**
Because that makes the product worse and the model harder. Holding per room means the guest books room 412, so an upgrade requires changing the booking, housekeeping cannot shuffle rooms, and the availability query becomes per-room rather than per-type — a count, which is what you actually sell. Real hotels hold by type and assign at check-in. And if a customer genuinely wants a guaranteed specific room, that is a different product with a per-room calendar, which I would build as a separate path rather than forcing it into this one.

**Q10 (medium). *Pushback:* "You have two structures, a count and a calendar per room. Isn't that duplication?"**
They answer different questions and the relationship is explicit: the calendars are the source of truth, and the count is derived from them under the room-type lock. The count is what you sell — "three Deluxes free" — and the calendars are what you assign from and what maintenance blocks. Deriving the count on every query is what keeps them from drifting; a cached count would be a second source of truth and a consistency bug waiting to happen. If the derivation became hot, I would cache it with an explicit invalidation path, not by treating it as authoritative.

**Q11 (hard). *Pushback:* "Your per-type lock serialises every booking for the most popular room type. Isn't that a bottleneck?"**
At hotel volumes, no: a property with 300 rooms does not have a booking throughput problem, and the critical section is a bisect plus an insert. But if it did contend, the fix is night striping — lock on the nights rather than the type, so two stays with disjoint dates take different locks and never contend, which is the common case. Acquiring the night locks in sorted order is mandatory, because two stays that partially overlap will both want multiple nights and unsorted acquisition deadlocks. And beyond one process, neither helps: the lock has to become a database transaction or a CAS on per-night rows.

**Q12 (hard). What breaks when there are multiple properties?**
The `Clock`, first and most. A single `today()` is wrong as soon as two properties are in different timezones, because "is this booking in the past?" has a different answer in Auckland and Los Angeles. Dates become property-local, inventory is partitioned by property, and the search becomes a cross-property query that has to fan out and merge. The class design barely changes — `PropertyClock(property)` and a `property_id` on the inventory — which is a sign the original separation was right.

**Q13 (medium). How do you test this?**
Four layers. The boundary tests on `overlaps`: turnover, arrival day, one-night overlap, containment, self-overlap, empty range — because that is where the two-clause test lives and where the off-by-one would be. The transition tests for every legal and every illegal state move. The invariant test: after any sequence of operations, each room's intervals are sorted and disjoint — `check_invariants` is the assertion. And the concurrency test: two threads, one free room, the same dates, released together by a barrier, asserting exactly one booking succeeded. That last one is the only test that proves the atomicity claim.

**Q14 (medium). How is pricing modelled, and why not a single rate?**
The rate is a property of the night, not of the stay. A stay that spans a weekend and a holiday has three different nightly rates, so pricing per night and summing is the only correct approach — pricing the whole stay from the check-in date's season gets the total wrong. Seasons compound if they overlap, weekend days get a surcharge, and a length-of-stay discount applies to the stay rather than to a night. Money is integer minor units and rounding happens once, at multiplication, so a total is never the sum of separately rounded line items.

**Q15 (hard). What would you do differently at 10×?**
Nothing in the class design. A hotel is small — the per-type lock is nowhere near contention. What changes is where the state lives and where the lock lives: the reservation store becomes a database with real transactions rather than a dict with a `threading.Lock`, the availability search moves to a read replica or a denormalised count table so it does not compete with the booking path, and the hard problem becomes channel synchronisation with the OTAs, where two systems both believe they own the inventory and a distributed lock or an idempotent reconciliation is the only answer. Those are all in the HLD files, and knowing the boundary is the point.

---

## 12. Cheat-sheet summary

**The model**

```
  Guest ──┐
          ├──▶ Reservation ──▶ RoomType (1) ──▶ Room (*)
  DateRange ┘        │
                     └──▶ room_number: Room | None   (assigned at CHECK-IN)

  Inventory is held against the TYPE. The room number is an assignment.
```

**The convention that everything rests on**

```
  HALF-OPEN [check_in, check_out)
  overlap:  a.start < b.end  AND  b.start < a.end     ← two clauses, no special case
  turnover: [5,8) and [8,11) do NOT overlap. The checkout night is sellable.

  WHY: with inclusive dates, the overlap test needs a boundary clause, and the
  classic bug — refusing a guest who arrives on a departure day — costs a
  night's revenue on every stay.
```

**Availability complexity**

| Approach | Cost |
|---|---|
| Scan all reservations per room | O(reservations) |
| **Sorted disjoint intervals per room + bisect** | **O(log n)**, two candidates |
| Naive check-then-act | correct-looking, **double-books** |
| Per-type lock around check AND act | **correct** |

**The race, and the fix**

```
  BROKEN:  free = find_free();  with lock: calendar.add(...)   ← two sections
  FIXED:   with lock: free = find_free(); calendar.add(...)    ← one section

  INVARIANT: for any room, the held ranges are pairwise non-overlapping.
  Not "one holder per room" — the same room is legitimately sold twice,
  on disjoint dates. That is the difference from the parking-lot problem.
```

**The state machine**

```
  PENDING → CONFIRMED → CHECKED_IN → CHECKED_OUT
     │           │  └──▶ NO_SHOW
     └───────────┴──▶ CANCELLED            (terminal: CANCELLED, CHECKED_OUT, NO_SHOW)

  EVERY terminal transition that leaves CHECKED_IN out of the picture must
  RELEASE THE NIGHTS. A booking that never releases silently shrinks the hotel.
```

**The numbers**

| Parameter | Value |
|---|---|
| Booking horizon | 730 days |
| Min / max stay | 1 / 30 nights |
| Weekend days | Friday, Saturday |
| Long-stay discount | 7+ nights, 10 % |
| Overbooking allowance | 0–20 %, per room type, default 0 |
| Lock granularity | Per room type; night striping if contended |
| Lock hold time | Microseconds — bisect + insert |
| Cancellation tiers | 168 h full, 24 h half, then none |
| Availability search | Advisory; the hold decides |

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "How do you model dates?" | "Half-open `[check_in, check_out)` — the checkout night is free, and overlap is two clauses." |
| "Two users, same room, same dates" | "Check-then-act. Check and act go in one critical section, locked per room type." |
| "Why not per room?" | "The count is per type, and a per-room lock cannot answer a count atomically." |
| "Make it multi-room" | "All-or-nothing under one lock; release everything on any failure — the partial hold is the bug to avoid." |
| "Is it O(1)?" | "O(log n) per room: disjoint sorted intervals, so a bisect leaves two candidates." |
| "Make it multi-property" | "The `Clock` changes first — `today()` is property-local. Inventory partitions; the class design barely moves." |
| "Add a waitlist" | "An offer TTL, exactly like a seat hold — an entry that never expires starves the queue." |
| "Prove it" | "Boundary tests on `overlaps`, the disjointness invariant after every operation, and a barrier test for two threads on the last room." |

---

## Cross-links

- The race mechanics this file assumes: [Concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md) · [Design patterns](../03-design-patterns.md) · [UML and class diagrams](../05-uml-and-class-diagrams.md) · [OOP principles and SOLID](../02-oop-principles-and-solid.md) · [The machine-coding framework](../06-machine-coding-interview-framework.md) · [What is LLD](../01-what-is-lld-and-how-to-answer.md)
- The sibling problem with the same race and a point resource instead of a range: [Parking lot](01-parking-lot.md) · and the TTL-hold variant: [BookMyShow seat locking](04-bookmyshow-seat-locking.md)
- Where the class view ends and the system view begins: [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md) · [BookMyShow seat booking as a system](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md)
- Storing reservations durably and querying them: [SQL deep dive](../../HLD/04-data/01-sql-deep-dive.md) · [Choosing a database](../../HLD/04-data/05-choosing-a-database-decision-guide.md) · [Caching strategies and layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md)
- The interview framework this problem is meant to be answered inside: [The five-step HLD framework](../../HLD/01-foundations/04-five-step-interview-framework.md)
- Other problems in this pillar: [LRU and LFU cache](02-lru-lfu-cache.md) · [Elevator system](05-elevator-system.md) · [Splitwise](06-splitwise.md) · [ATM machine](07-atm-machine.md) · [Library management](08-library-management.md)
