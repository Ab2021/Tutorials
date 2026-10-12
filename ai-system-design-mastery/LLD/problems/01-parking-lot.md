# LLD Problem: Parking Lot

> **Grounding note — this is the one problem the source transcript actually names.** At **[26:48]–[27:38]** it gives the structure directly: class `ParkingLot` holds Floors *"in the form of a List"*; `Floor` has `findAvailableSpot()`, `reserveSpot()`, `releaseSpot()`; class `Floor` holds Spots in a List; and there is a class `ParkingSpot`. Everything the transcript does not say — vehicle types, spot sizes, pricing, allocation policy, tickets, gates, or any concurrency treatment — is standard industry practice, clearly marked as such below. §1.1 reproduces the transcript's structure first, then §1.2 builds it out. The companion concept files for this problem are [What is LLD](../01-what-is-lld-and-how-to-answer.md), [OOP and SOLID](../02-oop-principles-and-solid.md), [Design patterns](../03-design-patterns.md), [Concurrency](../04-concurrency-and-threading-in-lld.md) and [UML](../05-uml-and-class-diagrams.md).

---

## 1. Requirement clarification

### 1.1 What the transcript gives, exactly

```
  ParkingLot
      │  has Floors, held "in the form of a List"
      ▼
   Floor   ── findAvailableSpot()   [26:48 – 27:38]
      │    ── reserveSpot()
      │    ── releaseSpot()
      ▼  has Spots, "in the form of a List"
  ParkingSpot

  Plus: "you draw a class diagram, define the methods, apply the OOP concept"
```

**One genuine ambiguity in the sketch, and how this file resolves it.** The transcript's three methods are attributed to `Floor`, but a real system needs those operations on the lot as well — a caller does not want to know which floor to ask. Rather than guess the transcript's intent, this file states a defensible reading and uses it consistently:

> **Reading adopted here:** `ParkingLot` exposes the system-facing operations (`reserve`, `release`) and delegates to the floor that owns the spot; `Floor` exposes `find_available`, `reserve_spot` and `release_spot` as the *internal* operations over its own list. This keeps the transcript's three method names where it puts them, and gives the lot the public surface a caller needs. It is one defensible reading, not a transcript fact.

### 1.2 The questions to ask

| # | Question | Why it matters | Default if they will not say |
|---|---|---|---|
| 1 | Single lot, or multiple sites? | Decides whether an aggregate above `ParkingLot` is needed | Single lot; say how it extends |
| 2 | Which entities are in scope? | Sets the size of the answer | Lot, Floor, Spot, Vehicle, Ticket, Pricing, Allocation |
| 3 | Are different vehicle sizes handled? | Forces the size-matching rule | Yes: SMALL / MEDIUM / LARGE, matched exactly or upward |
| 4 | Can a spot be reserved in advance? | Decides whether there is a hold with a TTL | No advance booking; a short hold during payment only |
| 5 | Is payment in scope? | The largest single scope decision | Out of scope as an entity; a `PricingStrategy` and a `Money` value are in |
| 6 | Flat pricing or time-based? | Decides the pricing interface's shape | Time-based, with a flat implementation for tests |
| 7 | How many entry and exit gates? | Decides whether gates are entities or observers | One entry and one exit gate, modelled as observers |
| 8 | Do you need to find your car later? | Decides whether a ticket lookup index exists | Yes, by plate and by ticket id |
| 9 | What about EV charging, valet, reserved spots? | The standard extension axes | Out of scope, named as extensions |
| 10 | How many concurrent entries at peak? | Predicts the contention strategy | High contention on the *hot* size class; optimise for it |

### 1.3 Defaults to state out loud

```
  SCOPE IN:   ParkingLot, Floor, ParkingSpot, Vehicle (Bike|Car|Truck),
              Ticket, PricingStrategy, SpotAllocator, Clock
  SCOPE OUT:  payment gateway, gate hardware, notification, mobile app.
              Named as adapters with no interesting structure.
  SIZES:      SMALL < MEDIUM < LARGE. A vehicle may take its own size
              or larger. A bike may take a car spot; a truck may not.
  LIFECYCLE:  FREE -> HELD -> OCCUPIED -> FREE   (HELD expires back to FREE)
  SHARED RES: the ParkingSpot. Concurrency lives in its status transition.
  PRICING:    time-based, per size class, integer minor units.
  MONEY:      integer minor units. Never a float.
  TIME:       injected Clock, so expiry tests need no sleep().
  FAILURE:    reservation hold with a TTL + idempotency key.
  EXTEND:     pricing model, allocation policy, multiple lots, EV spots.
```

**The three questions that actually shape this design:** whether the spot's status transition is atomic (it is — §6), whether pricing is a policy (it is — §5), and whether the lot knows about gates (it does not — §5, Observer).

---

## 2. Core entities & responsibilities

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `ParkingLot` | Orchestrate reserve and release across floors | floors, pricing, allocator, clock | Pick a specific spot; compute a fee; know about gates |
| `Floor` | Hold spots; find, reserve and release one | an ordered list of spots | Price; issue tickets; know about other floors |
| `ParkingSpot` | Own its occupancy state and legal transitions | spot_id, size, status, held_by, hold_expires_at | Decide who may hold it; price itself; know about vehicles |
| `Vehicle` *(abstract)* | Carry identity and size | plate, size | Anything else |
| `Bike` / `Car` / `Truck` | Bind a concrete size | — | — |
| `Ticket` | Record one parking session | ticket_id, spot, vehicle, entry_at, exit_at, fee | Compute its own fee; mutate the spot |
| `PricingStrategy` *(interface)* | Compute a fee for a ticket | a rate table | Know about floors, spots or allocation |
| `SpotAllocator` *(interface)* | Choose a spot from candidates | a policy | Mutate the spot |
| `SpotRepository` *(interface)* | Provide an atomic compare-and-set on spot status | the store | Decide anything |
| `Clock` *(interface)* | Supply the current time | — | — |
| `EntryGate` / `ExitGate` *(observer)* | React to occupancy changes | a display | Drive the lot's logic |

**Why `Ticket` does not compute its own fee:** it would need the rate table, so it would have two reasons to change — the session it records and the pricing policy. That is the SRP split, and it is what makes `PricingStrategy` replaceable. See [OOP principles §5](../02-oop-principles-and-solid.md).

**Why `SpotRepository` exists separately from `ParkingSpot`:** the spot is a domain object that knows its own transitions; the repository is what makes the *transition atomic* against other actors. Separating them is what allows the same `ParkingSpot` logic to sit on an in-memory store in the design and a database in production — the class-level treatment in §5–§6, and the store-level one in [BookMyShow seat locking](04-bookmyshow-seat-locking.md).

---

## 3. Class diagram

```
                     ┌──────────────────────────────┐
                     │          ParkingLot          │
                     ├──────────────────────────────┤
                     │ - lot_id: str                │
                     │ - floors: list[Floor]        │
                     │ - pricing: PricingStrategy   │
                     │ - allocator: SpotAllocator   │
                     │ - repo: SpotRepository       │
                     │ - clock: Clock               │
                     ├──────────────────────────────┤
                     │ + reserve(v, key): Ticket    │
                     │ + release(t): Money          │
                     │ + available(size): int       │
                     └───┬────────────┬─────────┬───┘
                         │ 1          │ 1       │ 1
                         │ ◆ 1..*     │         │
                         ▼            │         │
                 ┌───────────────┐    │         │
                 │     Floor     │    │         │
                 ├───────────────┤    │         │
                 │ - floor_id:int│    │         │
                 │ - spots: list │    │         │
                 ├───────────────┤    │         │
                 │ +find_avail() │    │         │
                 │ +reserve_spot()    │         │
                 │ +release_spot()    │         │
                 └───────┬───────┘    │         │
                         │ 1          │         │
                         │ ◆ 0..*     │         │
                         ▼            │         │
                 ┌───────────────┐    │         │
                 │  ParkingSpot  │    │         │
                 ├───────────────┤    │         │
                 │ - spot_id:str │    │         │
                 │ - size:Size   │    │         │
                 │ - status:St   │    │         │
                 │ - held_by:str │    │         │
                 │ - expires:flt │    │         │
                 ├───────────────┤    │         │
                 │ +try_hold()   │    │         │
                 │ +confirm()    │    │         │
                 │ +release()    │    │         │
                 │ +is_avail()   │    │         │
                 └───────┬───────┘    │         │
                         │ 0..1       │         │
                         │ at most ONE active ticket —        │
                         │ the double-booking invariant       │
                         ▼            │         │
                 ┌───────────────┐    │         │
                 │    Ticket     │    │         │
                 ├───────────────┤    │         │
                 │ - ticket_id   │    │         │
                 │ - entry_at    │    │         │
                 │ - exit_at     │    │         │
                 │ - fee: Money  │    │         │
                 ├───────────────┤    │         │
                 │ + close(at,fee)    │         │
                 │ + is_open()   │    │         │
                 └───────┬───────┘    │         │
                         │ *          │         │
                         │ 1          │         │
                         ▼            │         │
                 ┌───────────────┐    │         │
                 │    Vehicle    │    │         │
                 │   {abstract}  │    │         │
                 ├───────────────┤    │         │
                 │ # plate: str  │    │         │
                 │ # size: Size  │    │         │
                 └───────┬───────┘    │         │
                         │            │         │
            ┌────────────┼────────┐   │         │
            │            │        │   │         │
        ┌───────┐   ┌───────┐ ┌───────┐         │
        │ Bike  │   │  Car  │ │ Truck │         │
        └───────┘   └───────┘ └───────┘         │
                                               │
        ┌──────────────────────────┐           │
        │ «interface»              │◄──────────┘
        │ PricingStrategy          │  association: the lot knows a policy
        ├──────────────────────────┤
        │ +calculate(t, at): Money │
        └────────────▲─────────────┘
                     │ realization
        ┌────────────┴──────────────┬──────────────────┐
        │ HourlyPricing             │ FlatPricing      │
        └───────────────────────────┴──────────────────┘

        ┌──────────────────────────┐
        │ «interface»              │◄── association
        │ SpotAllocator            │
        ├──────────────────────────┤
        │ +choose(floors, size)    │
        └────────────▲─────────────┘
                     │
        ┌────────────┴──────────────┬──────────────────┐
        │ NearestFirstAllocator     │ BalancedAllocator│
        └───────────────────────────┴──────────────────┘

        ┌──────────────────────────┐
        │ «interface»              │◄── association
        │ SpotRepository           │
        ├──────────────────────────┤
        │ +cas_hold(spot_id, ...)  │   ← the atomic step
        │ +cas_release(spot_id,..) │
        │ +find(spot_id): Spot     │
        └────────────▲─────────────┘
                     │
        ┌────────────┴──────────────┬──────────────────┐
        │ InMemorySpotRepository    │ SqlSpotRepository│
        └───────────────────────────┴──────────────────┘
```

**What the diagram already states:**

- `ParkingLot ◆── Floor ◆── ParkingSpot` — filled diamonds, so a spot cannot outlive its floor.
- `ParkingSpot 0..1 ──── 0..1 Ticket` — **the double-booking invariant, drawn.**
- Three `«interface»` boxes, all *associations* from the lot: pricing, allocation, repository. Three axes of change isolated before any code was written.
- `Vehicle` is abstract with three concrete subclasses, so the size-matching rule is a polymorphic call.
- **No gate in the diagram** — deliberately out of scope, and said so.

**The invariants, written out:**

| # | Invariant | Where it is enforced |
|---|---|---|
| I1 | A spot has at most one active ticket | `cas_hold` + the `0..1` cardinality |
| I2 | A vehicle occupies a spot of its size or larger | `SpotAllocator.choose` |
| I3 | A spot in `HELD` past its TTL is treated as `FREE` | `is_available` (lazy) + `sweep` (proactive) |
| I4 | A released spot returns to `FREE` | `release_spot` |
| I5 | A ticket's fee is computed once, at release | `Ticket.close` |
| I6 | An occupancy count is never negative | `Floor.available` computes rather than stores |

**I6 is a design choice worth defending:** the floor does *not* keep a running count, because a counter is a read-modify-write race that would need its own lock for a value that is only used for display. It computes instead, and the gate display accepts a slightly stale view. See §6.6.

---

## 4. Interfaces, abstract classes and enums

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class VehicleSize(Enum):
    """Ordered: comparison by enum member order lets `>=` express
    'this spot is big enough'. SMALL < MEDIUM < LARGE."""
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


class SpotStatus(Enum):
    FREE = "FREE"
    HELD = "HELD"          # held for a specific vehicle, expiring
    OCCUPIED = "OCCUPIED"  # a vehicle is physically parked


@dataclass(frozen=True)
class Money:
    """IMMUTABLE. Integer minor units — never a float. Frozen, so it is
    thread-safe by construction and safe to share across threads."""
    amount_minor: int
    currency: str = "INR"

    def __add__(self, other: "Money") -> "Money":
        if other.currency != self.currency:
            raise ValueError(f"currency mismatch: {self.currency} vs {other.currency}")
        return Money(self.amount_minor + other.amount_minor, self.currency)

    def __mul__(self, factor: int) -> "Money":
        if factor < 0:
            raise ValueError("cannot multiply money by a negative factor")
        return Money(self.amount_minor * factor, self.currency)

    def __str__(self) -> str:
        return f"{self.amount_minor / 100:.2f} {self.currency}"


class Clock(ABC):
    """Time is injected so every TTL and expiry test is deterministic and
    instant. This is the single highest-value testability decision here."""

    @abstractmethod
    def now(self) -> float: ...


class PricingStrategy(ABC):
    """Strategy. The axis that varies: how a fee is computed."""

    @abstractmethod
    def calculate(self, ticket: "Ticket", exit_at: float) -> Money: ...


class SpotAllocator(ABC):
    """Strategy. The axis that varies: which spot gets chosen."""

    @abstractmethod
    def choose(self, floors: list["Floor"], size: VehicleSize) -> "ParkingSpot | None": ...


class SpotRepository(ABC):
    """The seam that makes the status transition ATOMIC. This is the concurrency
    design: the implementation must guarantee compare-and-set semantics."""

    @abstractmethod
    def cas_hold(self, spot_id: str, plate: str, expires_at: float,
                 idempotency_key: str) -> bool:
        """Atomically: if the spot is FREE or its hold has expired,
        set it HELD for `plate` until `expires_at`. Return True iff this
        caller made that change."""

    @abstractmethod
    def cas_occupy(self, spot_id: str, plate: str) -> bool:
        """Atomically promote a hold by `plate` to OCCUPIED."""

    @abstractmethod
    def cas_release(self, spot_id: str, plate: str) -> bool:
        """Atomically return a spot to FREE if it is held/occupied by `plate`."""

    @abstractmethod
    def get(self, spot_id: str) -> "ParkingSpot": ...


class RepositoryError(RuntimeError):
    """Base for store failures, so callers never catch a vendor exception."""


class ConcurrencyConflict(RepositoryError):
    """The compare-and-set lost. Expected, not exceptional."""


class LotFullError(RuntimeError):
    pass


class SpotTakenError(RuntimeError):
    pass


class InvalidTransition(RuntimeError):
    pass
```

**Why the abstract/interface split falls where it does:** `Vehicle` is an *abstract class* because `Bike`, `Car` and `Truck` share the plate and size handling and differ only in the constant — shared mechanism. `PricingStrategy`, `SpotAllocator`, `SpotRepository` and `Clock` are *interfaces* because nothing is shared and what matters is the contract. See [OOP principles §8](../02-oop-principles-and-solid.md).

---

## 5. Complete working code

**Language note:** Python 3.11+, standard library only. The design is language-independent; interview languages vary, and Java or Go would express the same structure with `interface` and `struct` respectively. No third-party imports, and no placeholders.

### 5.1 Value types and implementations

```python
import uuid
from dataclasses import dataclass, field


class RealClock(Clock):
    def now(self) -> float:
        return time.time()


class FakeClock(Clock):
    """Tests advance this instead of sleeping."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time does not run backwards")
        self._now += seconds


class Vehicle(ABC):
    """Abstract base: shares plate handling, differs only in size."""

    def __init__(self, plate: str) -> None:
        if not plate or not plate.strip():
            raise ValueError("plate must be non-empty")
        self._plate = plate.strip().upper()

    @property
    def plate(self) -> str:
        return self._plate

    @property
    @abstractmethod
    def size(self) -> VehicleSize: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._plate})"


class Bike(Vehicle):
    @property
    def size(self) -> VehicleSize:
        return VehicleSize.SMALL


class Car(Vehicle):
    @property
    def size(self) -> VehicleSize:
        return VehicleSize.MEDIUM


class Truck(Vehicle):
    @property
    def size(self) -> VehicleSize:
        return VehicleSize.LARGE


class HourlyPricing(PricingStrategy):
    """Time-based, per size class, with a grace period and a daily cap."""

    def __init__(self, hourly_minor: dict[VehicleSize, int],
                 grace_minutes: int = 15, daily_cap_minor: int | None = None) -> None:
        self._rates = dict(hourly_minor)
        self._grace = grace_minutes
        self._cap = daily_cap_minor

    def calculate(self, ticket: "Ticket", exit_at: float) -> Money:
        if exit_at < ticket.entry_at:
            raise ValueError("exit cannot precede entry")
        minutes = int((exit_at - ticket.entry_at) // 60)
        billable = max(0, minutes - self._grace)
        hours = -(-billable // 60)                       # ceiling division
        rate = self._rates[ticket.vehicle.size]
        amount = rate * hours
        if self._cap is not None:
            amount = min(amount, self._cap)
        return Money(amount)


class FlatPricing(PricingStrategy):
    """One flat fee. Its real job is the test double that makes fee assertions
    exact — the second justification for the interface existing at all."""

    def __init__(self, amount_minor: int) -> None:
        self._amount = amount_minor

    def calculate(self, ticket: "Ticket", exit_at: float) -> Money:
        return Money(self._amount)


class NearestFirstAllocator(SpotAllocator):
    """Walk floors in order, take the first spot of the right size,
    preferring the exact size over a larger one so big spots stay free."""

    def choose(self, floors: list["Floor"], size: VehicleSize) -> "ParkingSpot | None":
        for floor in floors:
            exact = floor.find_available(size, exact=True)
            if exact is not None:
                return exact
        for floor in floors:                             # second pass: upgrades
            larger = floor.find_available(size, exact=False)
            if larger is not None:
                return larger
        return None


class BalancedAllocator(SpotAllocator):
    """Fill by occupancy ratio so no single floor becomes a favourite."""

    def choose(self, floors: list["Floor"], size: VehicleSize) -> "ParkingSpot | None":
        candidates: list[tuple[float, Floor]] = []
        for floor in floors:
            spot = floor.find_available(size, exact=False)
            if spot is not None:
                candidates.append((floor.occupancy_ratio(), floor))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: pair[0])
        return candidates[0][1].find_available(size, exact=False)
```

### 5.2 The entity classes

```python
class ParkingSpot:
    """Owns its state and the LEGAL TRANSITIONS. It never decides who may
    hold it — the repository's compare-and-set does that, atomically."""

    def __init__(self, spot_id: str, size: VehicleSize) -> None:
        if not spot_id:
            raise ValueError("spot_id is required")
        self._spot_id = spot_id
        self._size = size
        self._status = SpotStatus.FREE
        self._held_by: str | None = None
        self._hold_expires_at: float | None = None
        self._occupant: str | None = None

    # ---- reads -------------------------------------------------------
    @property
    def spot_id(self) -> str:
        return self._spot_id

    @property
    def size(self) -> VehicleSize:
        return self._size

    @property
    def status(self) -> SpotStatus:
        return self._status

    @property
    def occupied_by(self) -> str | None:
        return self._occupant

    def fits(self, size: VehicleSize) -> bool:
        """A vehicle takes its own size or larger. Enum order makes this one
        comparison instead of a lookup table."""
        return self._size.value >= size.value

    def is_available(self, now: float) -> bool:
        """LAZY EXPIRY. An expired hold is available immediately, without a
        sweeper having run — this is the half that makes the answer CORRECT."""
        if self._status is SpotStatus.FREE:
            return True
        if self._status is SpotStatus.HELD:
            return self._hold_expires_at is not None and now >= self._hold_expires_at
        return False                                     # OCCUPIED

    @property
    def held_by(self) -> str | None:
        return self._held_by

    @property
    def hold_expires_at(self) -> float | None:
        return self._hold_expires_at

    # ---- transitions -------------------------------------------------
    def apply_hold(self, plate: str, now: float, ttl: int) -> None:
        """Called by the repository ONLY, after it has won the compare-and-set."""
        if not self.is_available(now):
            raise InvalidTransition(f"spot {self._spot_id} is not available")
        self._status = SpotStatus.HELD
        self._held_by = plate
        self._hold_expires_at = now + ttl
        self._occupant = None

    def apply_occupy(self, plate: str) -> None:
        if self._status is SpotStatus.OCCUPIED:
            return                                       # idempotent
        if self._status is SpotStatus.HELD and self._held_by != plate:
            raise InvalidTransition(
                f"spot {self._spot_id} is held by {self._held_by}, not {plate}")
        self._status = SpotStatus.OCCUPIED
        self._occupant = plate
        self._held_by = None
        self._hold_expires_at = None

    def apply_release(self, plate: str) -> None:
        owner = self._occupant or self._held_by
        if owner is None:
            if self._status is SpotStatus.FREE:
                return                                   # idempotent release
            raise InvalidTransition(f"spot {self._spot_id} has no holder")
        if owner != plate:
            raise InvalidTransition(
                f"spot {self._spot_id} is held by {owner}, not {plate}")
        self._status = SpotStatus.FREE
        self._held_by = None
        self._occupant = None
        self._hold_expires_at = None

    def __repr__(self) -> str:
        return f"Spot({self._spot_id}, {self._size.name}, {self._status.value})"


class Floor:
    """Holds spots. The transcript's three methods live here, as the internal
    operations over this floor's own list."""

    def __init__(self, floor_id: int, spots: list[ParkingSpot]) -> None:
        if not spots:
            raise ValueError(f"floor {floor_id} has no spots")
        if len({s.spot_id for s in spots}) != len(spots):
            raise ValueError("duplicate spot_id on one floor")
        self._floor_id = floor_id
        self._spots = list(spots)

    @property
    def floor_id(self) -> int:
        return self._floor_id

    @property
    def spots(self) -> tuple[ParkingSpot, ...]:
        """A tuple, not the internal list: callers cannot mutate floor state."""
        return tuple(self._spots)

    def find_available(self, size: VehicleSize, exact: bool, now: float) -> ParkingSpot | None:
        """The transcript's findAvailableSpot(). `exact` controls whether a
        larger spot may be used."""
        for spot in self._spots:
            if not spot.is_available(now):
                continue
            if (spot.size is size) if exact else spot.fits(size):
                return spot
        return None

    def available_count(self, size: VehicleSize, now: float) -> int:
        return sum(1 for s in self._spots if s.is_available(now) and s.fits(size))

    def capacity(self, size: VehicleSize) -> int:
        return sum(1 for s in self._spots if s.fits(size))

    def occupancy_ratio(self) -> float:
        if not self._spots:
            return 0.0
        busy = sum(1 for s in self._spots if s.status is not SpotStatus.FREE)
        return busy / len(self._spots)

    def release_spot(self, spot_id: str, plate: str) -> None:
        """The transcript's releaseSpot(), as an internal operation."""
        for spot in self._spots:
            if spot.spot_id == spot_id:
                spot.apply_release(plate)
                return
        raise KeyError(f"no spot {spot_id} on floor {self._floor_id}")

    def __repr__(self) -> str:
        return f"Floor({self._floor_id}, {len(self._spots)} spots)"


class Ticket:
    """Records one session. It does NOT compute its own fee — that is the
    PricingStrategy's job, and the split is what makes pricing replaceable."""

    def __init__(self, ticket_id: str, spot: ParkingSpot, vehicle: Vehicle,
                 entry_at: float, hold_expires_at: float,
                 idempotency_key: str) -> None:
        self._ticket_id = ticket_id
        self._spot = spot
        self._vehicle = vehicle
        self._entry_at = entry_at
        self._hold_expires_at = hold_expires_at
        self._idempotency_key = idempotency_key
        self._exit_at: float | None = None
        self._fee: Money | None = None

    @property
    def ticket_id(self) -> str:
        return self._ticket_id

    @property
    def spot(self) -> ParkingSpot:
        return self._spot

    @property
    def vehicle(self) -> Vehicle:
        return self._vehicle

    @property
    def entry_at(self) -> float:
        return self._entry_at

    @property
    def hold_expires_at(self) -> float:
        return self._hold_expires_at

    @property
    def idempotency_key(self) -> str:
        return self._idempotency_key

    @property
    def fee(self) -> Money | None:
        return self._fee

    def is_open(self) -> bool:
        return self._exit_at is None

    def close(self, exit_at: float, fee: Money) -> None:
        if self._exit_at is not None:
            return                                       # idempotent close
        if exit_at < self._entry_at:
            raise ValueError("exit cannot precede entry")
        self._exit_at = exit_at
        self._fee = fee

    def __repr__(self) -> str:
        state = "OPEN" if self.is_open() else f"CLOSED {self._fee}"
        return f"Ticket({self._ticket_id}, {self._vehicle.plate}, {state})"
```

### 5.3 The repository — where atomicity lives

```python
class InMemorySpotRepository(SpotRepository):
    """The compare-and-set. In the design this is a dict guarded by ONE lock per
    spot; in production it is `UPDATE ... WHERE status='FREE'` or a Redis Lua
    script, and the SEMANTICS are identical.

    The lock is per-spot and held for microseconds, never across I/O. That is
    the whole reason the critical section is here and not around the caller.
    """

    def __init__(self, spots: dict[str, ParkingSpot], clock: Clock) -> None:
        self._spots = spots
        self._clock = clock
        self._locks: dict[str, threading.Lock] = {sid: threading.Lock() for sid in spots}
        self._global = threading.Lock()                  # guards idempotency records
        self._idempotency: dict[str, str] = {}           # key -> spot_id

    def _lock_for(self, spot_id: str) -> threading.Lock:
        try:
            return self._locks[spot_id]
        except KeyError:
            raise KeyError(f"unknown spot {spot_id}") from None

    def cas_hold(self, spot_id: str, plate: str, expires_at: float,
                 idempotency_key: str) -> bool:
        # IDEMPOTENCY FIRST: a retry of a request that already succeeded returns
        # the original outcome instead of doing the work again. This is what makes
        # a timed-out client safe to retry.
        with self._global:
            already = self._idempotency.get(idempotency_key)
        if already is not None:
            return already == spot_id                    # same spot → same result

        spot = self._spots[spot_id]
        now = self._clock.now()
        with self._lock_for(spot_id):                    # ← the critical section
            if not spot.is_available(now):
                return False
            spot.apply_hold(plate, now, int(expires_at - now))
        with self._global:
            self._idempotency[idempotency_key] = spot_id
        return True

    def cas_occupy(self, spot_id: str, plate: str) -> bool:
        spot = self._spots[spot_id]
        with self._lock_for(spot_id):
            if spot.status is SpotStatus.OCCUPIED and spot.occupied_by == plate:
                return True                              # idempotent
            if spot.status is SpotStatus.HELD and spot.held_by == plate:
                spot.apply_occupy(plate)
                return True
            if spot.status is SpotStatus.HELD and self._clock.now() >= (
                    spot.hold_expires_at or 0.0):
                return False                             # hold expired; re-reserve
            return False

    def cas_release(self, spot_id: str, plate: str) -> bool:
        spot = self._spots[spot_id]
        with self._lock_for(spot_id):
            try:
                spot.apply_release(plate)
            except InvalidTransition:
                return False
            return True

    def get(self, spot_id: str) -> ParkingSpot:
        return self._spots[spot_id]

    # ---- hygiene: the PROACTIVE half of expiry -----------------------
    def sweep_expired_holds(self, limit: int = 500) -> int:
        """Returns expired holds to FREE so they do not accumulate. Correctness
        comes from `is_available`'s lazy check; this is hygiene. Both are needed:
        lazy alone leaks rows, a sweeper alone has a wrong window."""
        now = self._clock.now()
        cleared = 0
        for spot_id, spot in self._spots.items():
            if cleared >= limit:
                break
            with self._lock_for(spot_id):
                if (spot.status is SpotStatus.HELD
                        and spot.hold_expires_at is not None
                        and now >= spot.hold_expires_at):
                    spot.apply_release(spot.held_by or "")
                    cleared += 1
        return cleared
```

**The single most important line in this file** is `if not spot.is_available(now): return False` *inside* the per-spot lock. Everything before it — `choose`, the availability display, the gate count — is a hint. That line is the decision.

### 5.4 The lot, the gates, and the composition root

```python
class OccupancyEvent:
    """`__slots__` because these are created per transition and never extended."""
    __slots__ = ("lot_id", "floor_id", "spot_id", "size", "status", "at")

    def __init__(self, lot_id: str, floor_id: int, spot_id: str,
                 size: VehicleSize, status: SpotStatus, at: float) -> None:
        self.lot_id = lot_id
        self.floor_id = floor_id
        self.spot_id = spot_id
        self.size = size
        self.status = status
        self.at = at


class OccupancyObserver(ABC):
    @abstractmethod
    def on_occupancy_changed(self, event: OccupancyEvent) -> None: ...


class OccupancyPublisher:
    """Observer. The lot knows observers exist; it knows nothing about them."""

    def __init__(self) -> None:
        self._observers: list[OccupancyObserver] = []
        self._lock = threading.Lock()

    def subscribe(self, observer: OccupancyObserver) -> "Callable[[], None]":
        with self._lock:
            self._observers.append(observer)
        return lambda: self._unsubscribe(observer)

    def _unsubscribe(self, observer: OccupancyObserver) -> None:
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def publish(self, event: OccupancyEvent) -> None:
        with self._lock:
            snapshot = list(self._observers)             # snapshot; notify outside
        for observer in snapshot:
            try:
                observer.on_occupancy_changed(event)
            except Exception:
                # ISOLATION: one bad display must not break the lot's state change
                # and must not stop the other observers.
                logging.getLogger(__name__).exception(
                    "observer %s failed", type(observer).__name__)


class GateDisplay(OccupancyObserver):
    """A display is eventually consistent by design — see §6.6."""

    def __init__(self, lot: "ParkingLot") -> None:
        self._lot = lot
        self._shown: dict[VehicleSize, int] = {}

    def on_occupancy_changed(self, event: OccupancyEvent) -> None:
        self.refresh()

    def refresh(self) -> None:
        for size in VehicleSize:
            self._shown[size] = self._lot.available(size)

    def shown(self, size: VehicleSize) -> int:
        return self._shown.get(size, 0)


class FloorMapProjection(OccupancyObserver):
    """Keeps a denormalised view for the app's map. Idempotent, so a duplicate
    notification is harmless — which is what makes best-effort delivery OK here."""

    def __init__(self) -> None:
        self._free: dict[str, set[str]] = {}

    def on_occupancy_changed(self, event: OccupancyEvent) -> None:
        bucket = self._free.setdefault(event.spot_id, set())
        if event.status is SpotStatus.FREE:
            bucket.add(event.spot_id)
        else:
            bucket.discard(event.spot_id)


class ParkingLot:
    """Orchestrates. The public surface a caller needs; every decision is
    delegated to a policy, the repository, or the spot itself."""

    def __init__(self, lot_id: str, floors: list[Floor], pricing: PricingStrategy,
                 allocator: SpotAllocator, repo: SpotRepository, clock: Clock,
                 hold_ttl: int = 900) -> None:
        if not floors:
            raise ValueError("a lot needs at least one floor")
        self._lot_id = lot_id
        self._floors = list(floors)
        self._pricing = pricing
        self._allocator = allocator
        self._repo = repo
        self._clock = clock
        self._hold_ttl = hold_ttl
        self._publisher = OccupancyPublisher()
        self._floor_of: dict[str, Floor] = {
            spot.spot_id: floor for floor in floors for spot in floor.spots
        }

    # ---- wiring ------------------------------------------------------
    def subscribe(self, observer: OccupancyObserver) -> "Callable[[], None]":
        return self._publisher.subscribe(observer)

    # ---- reads -------------------------------------------------------
    def available(self, size: VehicleSize) -> int:
        now = self._clock.now()
        return sum(floor.available_count(size, now) for floor in self._floors)

    def find_ticket_spot(self, plate: str) -> ParkingSpot | None:
        for floor in self._floors:
            for spot in floor.spots:
                if spot.occupied_by == plate:
                    return spot
        return None

    # ---- the two public operations -----------------------------------
    def reserve(self, vehicle: Vehicle, idempotency_key: str) -> Ticket:
        """`choose` is a HINT; `cas_hold` is the DECISION. Between the two,
        another actor can take the spot, and the retry below handles it."""
        if not idempotency_key:
            raise ValueError("idempotency_key is required for a safe retry")

        for attempt in range(2):                         # one retry, then fail
            spot = self._allocator.choose(self._floors, vehicle.size)
            if spot is None:
                raise LotFullError(
                    f"no {vehicle.size.name} spot available in lot {self._lot_id}")
            expires_at = self._clock.now() + self._hold_ttl
            if self._repo.cas_hold(spot.spot_id, vehicle.plate, expires_at,
                                   idempotency_key):
                ticket = Ticket(ticket_id=uuid.uuid4().hex, spot=spot,
                                vehicle=vehicle, entry_at=self._clock.now(),
                                hold_expires_at=expires_at,
                                idempotency_key=idempotency_key)
                self._publish(spot, SpotStatus.HELD)
                return ticket
            # lost the race; loop and re-choose (the hint was stale)

        raise SpotTakenError(
            "the chosen spot was taken concurrently; please retry")

    def check_in(self, ticket: Ticket) -> None:
        """The vehicle physically arrives: promote the hold to OCCUPIED."""
        if not self._repo.cas_occupy(ticket.spot.spot_id, ticket.vehicle.plate):
            raise SpotTakenError(
                f"hold on {ticket.spot.spot_id} expired before check-in")
        self._publish(ticket.spot, SpotStatus.OCCUPIED)

    def release(self, ticket: Ticket) -> Money:
        """Exit: compute the fee, close the ticket, free the spot. Order matters —
        the fee is computed BEFORE the spot is freed, so a pricing failure does
        not lose the spot."""
        if not ticket.is_open():
            return ticket.fee or Money(0)                # idempotent exit
        exit_at = self._clock.now()
        fee = self._pricing.calculate(ticket, exit_at)
        released = self._repo.cas_release(ticket.spot.spot_id, ticket.vehicle.plate)
        if not released:
            # Someone else already released it, or the plate does not match.
            # The ticket still closes: the session is over either way.
            logging.getLogger(__name__).warning(
                "release lost the race for spot %s", ticket.spot.spot_id)
        ticket.close(exit_at=exit_at, fee=fee)
        self._publish(ticket.spot, SpotStatus.FREE)
        return fee

    def sweep(self, limit: int = 500) -> int:
        if isinstance(self._repo, InMemorySpotRepository):
            return self._repo.sweep_expired_holds(limit)
        return 0

    def _publish(self, spot: ParkingSpot, status: SpotStatus) -> None:
        floor = self._floor_of.get(spot.spot_id)
        if floor is None:
            return
        self._publisher.publish(OccupancyEvent(
            lot_id=self._lot_id, floor_id=floor.floor_id, spot_id=spot.spot_id,
            size=spot.size, status=status, at=self._clock.now()))
```

### 5.5 The composition root

```python
def build_lot(clock: Clock, *, floors: list[dict[VehicleSize, int]] | None = None,
              pricing: PricingStrategy | None = None,
              allocator: SpotAllocator | None = None,
              hold_ttl: int = 900) -> tuple[ParkingLot, InMemorySpotRepository]:
    """One place where everything is wired. Nothing constructs its own
    dependencies — that is what makes every collaborator swappable in a test."""
    layout = floors or [
        {VehicleSize.SMALL: 20, VehicleSize.MEDIUM: 40, VehicleSize.LARGE: 10},
        {VehicleSize.SMALL: 20, VehicleSize.MEDIUM: 40, VehicleSize.LARGE: 10},
    ]
    built: list[Floor] = []
    by_id: dict[str, ParkingSpot] = {}
    for index, spec in enumerate(layout):
        spots: list[ParkingSpot] = []
        for size, count in spec.items():
            for n in range(count):
                spot = ParkingSpot(spot_id=f"F{index}-{size.name}-{n:03d}", size=size)
                spots.append(spot)
                by_id[spot.spot_id] = spot
        built.append(Floor(floor_id=index, spots=spots))

    repo = InMemorySpotRepository(spots=by_id, clock=clock)
    lot = ParkingLot(
        lot_id="LOT-1", floors=built,
        pricing=pricing or HourlyPricing(
            hourly_minor={VehicleSize.SMALL: 1000, VehicleSize.MEDIUM: 2000,
                          VehicleSize.LARGE: 3500},
            grace_minutes=15, daily_cap_minor=20000),
        allocator=allocator or NearestFirstAllocator(),
        repo=repo, clock=clock, hold_ttl=hold_ttl)
    return lot, repo
```

### 5.6 It runs — a demonstration

```python
def demo() -> None:
    clock = FakeClock()
    lot, _repo = build_lot(clock)
    lot.subscribe(GateDisplay(lot))

    car = Car("ka-01-ab-1234")
    ticket = lot.reserve(car, idempotency_key="req-1")
    print(ticket)                                  # Ticket(..., KA-01-AB-1234, OPEN)
    print(ticket.spot)                             # Spot(F0-MEDIUM-000, MEDIUM, HELD)

    lot.check_in(ticket)
    clock.advance(3600 + 60)                       # 1 h 1 min
    fee = lot.release(ticket)
    print(fee)                                     # 40.00 INR  (1 billable hour)
    print(lot.available(VehicleSize.MEDIUM))       # 79


if __name__ == "__main__":
    demo()
```

---

## 6. Concurrency and thread-safety

### 6.1 The shared mutable state

Exactly one thing: **a spot's status.** Everything else is either immutable (`Money`, `Vehicle`), stateless (`HourlyPricing`, `NearestFirstAllocator`), or derived (`Floor.available` computes rather than stores). Being able to say that precisely is most of the concurrency answer.

### 6.2 The exact race

```
  TIME   THREAD A (KA-01-1111)                THREAD B (KA-01-2222)
  ────   ──────────────────────────           ──────────────────────────
   t1    choose() → spot F0-M-000 (FREE)
   t2                                         choose() → spot F0-M-000 (FREE)
   t3    cas_hold: is_available? TRUE
   t4                                         cas_hold: is_available? TRUE
   t5    apply_hold(plate=A)
   t6                                         apply_hold(plate=B)
   t7    return Ticket(spot, A) ✓             return Ticket(spot, B) ✓
  ────   ──────────────────────────           ──────────────────────────
  RESULT: one spot, two tickets, two cars, one angry customer.
```

This is check-then-act: the window is between `is_available` and `apply_hold`.

### 6.3 What breaks under concurrency, and the fix

| # | What breaks | Fix | Where |
|---|---|---|---|
| 1 | Two vehicles hold one spot | `is_available` and `apply_hold` inside one per-spot lock | `cas_hold` |
| 2 | A retried request holds two spots | Idempotency key map, checked before the lock | `cas_hold` |
| 3 | A crashed holder strands a spot | TTL on the hold, checked lazily | `is_available` |
| 4 | Expired holds accumulate forever | Proactive `sweep_expired_holds` | `sweep` |
| 5 | One global lock caps throughput | Per-spot lock, not one lock for the lot | `_locks` dict |
| 6 | An observer throwing breaks the transition | Per-observer isolation; notify outside the lock | `OccupancyPublisher.publish` |
| 7 | Deadlock across two locks | Lock order is always idempotency-map → spot; never the reverse | `cas_hold` |
| 8 | The gate count is a read-modify-write race | Do not store it; compute it | `Floor.available_count` |
| 9 | A display shows a spot that is gone | Accepted: display is eventually consistent | §6.6 |
| 10 | Two releases for one ticket | `Ticket.close` and `apply_release` are both idempotent | `Ticket.close` |

### 6.4 Why per-spot locks and not one global lock

```python
# BROKEN ON THROUGHPUT (correct, but serialises everything):
with self._global_lock:
    if not spot.is_available(now): return False
    spot.apply_hold(...)

# THE DESIGN: one lock per spot, held for microseconds, never across I/O.
with self._locks[spot_id]:
    if not spot.is_available(now): return False
    spot.apply_hold(...)
```

A lot with 1 000 spots gets 1 000 locks; two vehicles competing for different spots never contend. The cost is 1 000 lock objects — about 50 KB — which is nothing against the throughput.

**And the honest limitation:** this only works within one process. Two pods have two lock dicts and will double-book. In production the atomicity moves into the store — `UPDATE parking_spots SET status='HELD' WHERE spot_id=%s AND status='FREE'`, checking `rowcount` — and `InMemorySpotRepository` is the design-time stand-in with identical semantics. That is the bridge to [BookMyShow seat locking §6](04-bookmyshow-seat-locking.md).

### 6.5 Lock ordering and deadlock

Only two locks are ever held, and always in the same order: the idempotency map first (a brief `with` block that is released before the spot lock is taken), then the spot lock. No path takes them in the other order, so circular wait is impossible. **If a future change needs two spots held at once — a multi-spot reservation — the fix is to sort by `spot_id` before locking.** See [concurrency §6.2](../04-concurrency-and-threading-in-lld.md).

### 6.6 What is deliberately eventually consistent

| Data | Consistency | Why it is acceptable |
|---|---|---|
| Gate display count | Eventual, ~seconds | A stale count shows "2 free" for a spot that just went; the customer is told at reserve time |
| Floor map projection | Eventual | Same |
| `available()` read | Snapshot | It is a hint for the UI; `cas_hold` is the truth |
| Spot status | **Strong** | This is the one that must never be wrong |
| Ticket fee | Strong | Computed once, at release |

**Say this out loud in the interview.** "The display can be stale; the hold cannot" is the sentence that shows you understand which guarantee each operation needs — and it is the same reasoning as the BookMyShow case study.

### 6.7 Testing concurrency

```python
def test_two_threads_never_both_hold_one_spot() -> None:
    """The invariant test. The barrier maximises overlap so the race is
    reachable; the assertion is the invariant, not a retry count."""
    clock = FakeClock()
    lot, _ = build_lot(clock, floors=[{VehicleSize.MEDIUM: 1}])
    winners: list[str] = []
    errors: list[str] = []
    barrier = threading.Barrier(2)

    def attempt(plate: str) -> None:
        barrier.wait()
        try:
            lot.reserve(Car(plate), idempotency_key=f"key-{plate}")
            winners.append(plate)
        except SpotTakenError:
            pass
        except Exception as exc:                          # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=attempt, args=(p,))
               for p in ("KA-01-1111", "KA-01-2222")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected errors: {errors}"
    assert len(winners) == 1, f"double booking: {winners}"


def test_retried_reserve_with_same_key_is_idempotent() -> None:
    clock = FakeClock()
    lot, _ = build_lot(clock, floors=[{VehicleSize.MEDIUM: 2}])
    first = lot.reserve(Car("KA-01-1111"), idempotency_key="same-key")
    second = lot.reserve(Car("KA-01-1111"), idempotency_key="same-key")
    assert first.spot.spot_id == second.spot.spot_id     # same spot, not a second one
    assert lot.available(VehicleSize.MEDIUM) == 1        # only one consumed


def test_hold_expires_exactly_at_ttl() -> None:
    clock = FakeClock()
    lot, _ = build_lot(clock, floors=[{VehicleSize.MEDIUM: 1}], hold_ttl=900)
    lot.reserve(Car("KA-01-1111"), idempotency_key="k1")

    clock.advance(899.999)
    assert lot.available(VehicleSize.MEDIUM) == 0        # still held

    clock.advance(0.001)                                 # exactly at TTL
    assert lot.available(VehicleSize.MEDIUM) == 1        # lazy expiry, no sweeper


def test_sweeper_clears_expired_holds_and_is_idempotent() -> None:
    clock = FakeClock()
    lot, _ = build_lot(clock, floors=[{VehicleSize.MEDIUM: 3}], hold_ttl=100)
    lot.reserve(Car("KA-01-1111"), idempotency_key="k1")
    lot.reserve(Car("KA-01-2222"), idempotency_key="k2")

    clock.advance(150)
    assert lot.sweep() == 2
    assert lot.sweep() == 0                              # nothing left to clear
    assert lot.available(VehicleSize.MEDIUM) == 3


def test_pricing_uses_integer_minor_units() -> None:
    clock = FakeClock()
    lot, _ = build_lot(clock, floors=[{VehicleSize.MEDIUM: 1}],
                       pricing=FlatPricing(1999))
    ticket = lot.reserve(Car("KA-01-1111"), idempotency_key="k1")
    clock.advance(10_000)
    fee = lot.release(ticket)
    assert fee.amount_minor == 1999                      # exact; no float drift
    assert isinstance(fee.amount_minor, int)


def test_a_failing_observer_does_not_break_the_reservation() -> None:
    clock = FakeClock()
    lot, _ = build_lot(clock, floors=[{VehicleSize.MEDIUM: 1}])

    class Exploding(OccupancyObserver):
        def on_occupancy_changed(self, event: OccupancyEvent) -> None:
            raise RuntimeError("display is on fire")

    lot.subscribe(Exploding())
    ticket = lot.reserve(Car("KA-01-1111"), idempotency_key="k1")
    assert ticket.spot.status is SpotStatus.HELD         # the transition survived


def test_release_is_idempotent() -> None:
    clock = FakeClock()
    lot, _ = build_lot(clock, floors=[{VehicleSize.MEDIUM: 1}],
                       pricing=FlatPricing(500))
    ticket = lot.reserve(Car("KA-01-1111"), idempotency_key="k1")
    lot.check_in(ticket)
    clock.advance(60)
    first = lot.release(ticket)
    second = lot.release(ticket)                          # a duplicate exit request
    assert first == second == Money(500)
    assert lot.available(VehicleSize.MEDIUM) == 1
```

**No `sleep` anywhere.** The `FakeClock` makes every expiry test exact and instant, and the barrier makes the race reachable rather than hoped for.

---

## 7. Extensibility — the standard follow-ups

### 7.1 A second pricing model

**The follow-up:** "Now add weekend and surge pricing."

**The answer:** a new `PricingStrategy`. `ParkingLot` does not change at all.

```python
class SurgePricing(PricingStrategy):
    """Composes an existing strategy and multiplies it. Note that this is
    COMPOSITION over inheritance — SurgePricing reuses HourlyPricing rather
    than extending it, so a change to the base rate logic flows through."""

    def __init__(self, base: PricingStrategy, demand: "Callable[[float], float]",
                 cap_multiplier: float = 3.0) -> None:
        self._base = base
        self._demand = demand
        self._cap = cap_multiplier

    def calculate(self, ticket: "Ticket", exit_at: float) -> Money:
        base = self._base.calculate(ticket, exit_at)
        multiplier = min(self._demand(exit_at), self._cap)
        return base * round(multiplier)                  # integer minor units only


# Wiring change only:
lot = ParkingLot(..., pricing=SurgePricing(
    base=HourlyPricing({VehicleSize.SMALL: 1000, VehicleSize.MEDIUM: 2000,
                        VehicleSize.LARGE: 3500}),
    demand=lambda at: 2.0 if is_weekend(at) else 1.0), ...)
```

**One new class, one wiring line, zero edits to `ParkingLot`.** That is the Open/Closed payoff and it is the whole point of step 4 of the [framework](../06-machine-coding-interview-framework.md).

### 7.2 Multiple lots

**The follow-up:** "Now support a campus with several lots."

**The answer:** one aggregate above `ParkingLot`. Everything below is unchanged.

```python
class Campus:
    """AGGREGATION, not composition: a lot outlives a campus. The node interface
    from Composite — `available` and `choose_lot` — is all the campus needs."""

    def __init__(self, campus_id: str, lots: list[ParkingLot]) -> None:
        if not lots:
            raise ValueError("a campus needs at least one lot")
        self._campus_id = campus_id
        self._lots = list(lots)

    def available(self, size: VehicleSize) -> int:
        return sum(lot.available(size) for lot in self._lots)

    def reserve(self, vehicle: Vehicle, idempotency_key: str,
                lot_id: str | None = None) -> Ticket:
        """Route by lot_id, or fall back to the first lot with capacity."""
        if lot_id is not None:
            for lot in self._lots:
                if lot.lot_id == lot_id:
                    return lot.reserve(vehicle, f"{idempotency_key}:{lot_id}")
            raise KeyError(f"no lot {lot_id} in campus {self._campus_id}")
        for lot in self._lots:
            if lot.available(vehicle.size) > 0:
                try:
                    return lot.reserve(vehicle, f"{idempotency_key}:{lot.lot_id}")
                except SpotTakenError:
                    continue                             # try the next lot
        raise LotFullError("no campus lot has capacity")
```

**The change is one new class.** Nothing below `ParkingLot` is touched, which is exactly what the composition in §3 bought. Also note the idempotency key is *namespaced by lot* — a subtle but important detail, because the same key retried against a different lot must not be treated as a duplicate.

### 7.3 EV spots

**The follow-up:** "Now some spots have chargers."

**The answer:** a `SpotFeature` set on the spot, and a `fits` that considers it. **Do not** create `EvSpot extends ParkingSpot` — the spot's *behaviour* is identical and only its suitability changes, so this is a capability, not a subtype.

```python
class SpotFeature(Enum):
    EV_CHARGER = "EV_CHARGER"
    DISABLED_BAY = "DISABLED_BAY"
    COVERED = "COVERED"


class FeatureAwareAllocator(SpotAllocator):
    """Wraps any allocator and filters by a required feature set."""

    def __init__(self, inner: SpotAllocator) -> None:
        self._inner = inner

    def choose(self, floors: list[Floor], size: VehicleSize,
               required: frozenset[SpotFeature] = frozenset()) -> ParkingSpot | None:
        if not required:
            return self._inner.choose(floors, size)
        for floor in floors:
            for spot in floor.spots:
                if spot.is_available(0.0) and spot.fits(size) and required <= spot.features:
                    return spot
        return None
```

### 7.4 Reserved and valet spots

**The follow-up:** "Some spots are reserved for monthly subscribers."

The answer is a **reservation entity over the spot's state machine**, not a new spot type — because the spot's occupancy logic is unchanged; only *who may hold it* differs.

```python
class AllocationPolicy(ABC):
    """A second axis of change, composed with SpotAllocator rather than merged
    into it: the allocator picks WHERE, the policy decides WHETHER."""

    @abstractmethod
    def may_use(self, spot: ParkingSpot, vehicle: Vehicle, now: float) -> bool: ...


class SubscriberOnlyPolicy(AllocationPolicy):
    def __init__(self, reserved_ids: frozenset[str], is_subscriber: "Callable[[str], bool]") -> None:
        self._reserved = reserved_ids
        self._is_subscriber = is_subscriber

    def may_use(self, spot: ParkingSpot, vehicle: Vehicle, now: float) -> bool:
        if spot.spot_id not in self._reserved:
            return True
        return self._is_subscriber(vehicle.plate)
```

### 7.5 Advance booking

**The follow-up:** "Let me reserve a spot for tomorrow at 9am."

**This one does not fit the existing seams, and the honest answer is to say so.** The change is a `Reservation` entity between `Vehicle` and `Spot` with a start and end time, and the cardinality becomes *time-dependent*:

```python
@dataclass(frozen=True)
class TimeRange:
    """Immutable — shared across threads, so it must be. The overlap test is
    the hard part of any booking system; see hotel management §7."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("end must be after start")

    def overlaps(self, other: "TimeRange") -> bool:
        """Half-open intervals [start, end): back-to-back bookings that touch
        do NOT overlap, which is what you want (exit at 10:00, enter at 10:00)."""
        return self.start < other.end and other.start < self.end


class Reservation:
    def __init__(self, reservation_id: str, spot: ParkingSpot, vehicle: Vehicle,
                 window: TimeRange) -> None:
        self._reservation_id = reservation_id
        self._spot = spot
        self._vehicle = vehicle
        self._window = window

    @property
    def window(self) -> TimeRange:
        return self._window
```

The invariant becomes "no two reservations on one spot overlap", which is enforced by a conditional insert with a range check in the store — **not** by the `0..1` cardinality. Saying "this extension changes the entity model, and here is how" is a better answer than claiming the current design already handles it.

---

## 8. Failure modes and edge cases

### 8.1 The failure-mode table

| # | Failure | Consequence | Handling |
|---|---|---|---|
| 1 | Two vehicles hold one spot | Double booking | Per-spot CAS (§6.3) |
| 2 | Retried request consumes a second spot | Inventory leak | Idempotency key, checked first |
| 3 | Holder crashes after holding | Spot stranded forever | TTL + lazy expiry + sweeper |
| 4 | TTL expires mid-payment | Customer pays for nothing | Confirm checks expiry; never charge for a lost spot (§8.3) |
| 5 | Lot full | No ticket | `LotFullError` with the size class named |
| 6 | A bike takes a truck spot | Large spots consumed | `NearestFirstAllocator` two-pass: exact size first |
| 7 | Release with a wrong plate | Wrong person's spot freed | `apply_release` checks the plate |
| 8 | Duplicate exit request | Two fees | `Ticket.close` and `cas_release` are idempotent |
| 9 | Pricing throws during release | Spot freed but no ticket closed, or vice versa | Fee computed *before* release; failure aborts the exit, spot stays |
| 10 | A display observer throws | Reservation fails | Per-observer isolation, notify outside the lock |
| 11 | Clock skew between pods | Early or late expiry | TTL is short and forgiving; expiry is a hint, not a contract |
| 12 | `available()` says 1 free, reserve fails | Customer confusion | Accepted — the read is a hint, and the UI says "checking…" |
| 13 | A spot is physically blocked but marked FREE | Ticket for a blocked spot | `mark_out_of_service` transitions it to a non-FREE state (§8.4) |
| 14 | Fee computed for a negative duration | Negative money | `calculate` rejects exit < entry |
| 15 | Currency mismatch in arithmetic | Meaningless total | `Money.__add__` raises |
| 16 | Very long stay overflows the daily cap | Wrong fee | Cap applied per 24 h, not per session (§8.5) |

### 8.2 Boundary cases worth naming

- **A lot with one spot.** The concurrency test uses this deliberately — it makes the race reachable every time.
- **A floor with zero spots of a requested size.** `find_available` returns `None`, and `choose` moves to the next floor.
- **Release exactly at the hold TTL.** `is_available` uses `>=`, so the boundary is defined: at exactly the TTL the spot is free. Tests assert both sides (899.999 s and 900.000 s).
- **Release of a ticket that was never checked in.** Legal — the hold is released and the fee is charged for the elapsed time.
- **A zero-length session.** Within the grace period, so the fee is zero.
- **A session longer than the daily cap.** The cap applies per 24-hour period, not per session.

### 8.3 The worst failure in this system, and how it is prevented

**A customer charged for a spot they did not get.** The ordering in `reserve` and `release` is chosen to make this impossible:

- `reserve` holds the spot **before** any payment is attempted, and payment is out of scope in this file precisely so the ordering is: hold → (pay) → check in. A failed hold means no payment attempt at all.
- `release` computes the fee **before** freeing the spot, so a pricing error aborts the exit and leaves the spot's state consistent with the open ticket.
- The hold's TTL is what bounds the window: if a customer's payment takes longer than the TTL, `check_in` fails with a clear error, and the correct behaviour is to tell the customer and refund — **never** to check them into a spot that is no longer theirs.

```python
def check_in(self, ticket: Ticket) -> None:
    """The correctness gate. If the hold expired, this fails LOUDLY rather than
    seating a vehicle in a spot that now belongs to someone else."""
    if self._clock.now() >= ticket.hold_expires_at:
        raise HoldExpiredError(
            f"hold on {ticket.spot.spot_id} expired at {ticket.hold_expires_at}; "
            "the spot may have been reallocated")
    if not self._repo.cas_occupy(ticket.spot.spot_id, ticket.vehicle.plate):
        raise SpotTakenError(
            f"spot {ticket.spot.spot_id} is no longer held by {ticket.vehicle.plate}")
    self._publish(ticket.spot, SpotStatus.OCCUPIED)
```

### 8.4 Out-of-service spots

A spot can be physically blocked — a spill, a repair, a snow pile. This is a **state**, not a deletion, because the spot still exists and its history matters.

```python
class OutOfServiceState:
    """The spot is not deletable and not FREE. Modelled as a status so that
    `find_available` skips it and the capacity maths stays honest."""

    def mark_out_of_service(self, spot_id: str, reason: str, until: float) -> None:
        spot = self._repo.get(spot_id)
        if spot.status is not SpotStatus.FREE:
            raise InvalidTransition("cannot take an occupied spot out of service")
        spot.apply_out_of_service(reason, until)
```

Adding `OUT_OF_SERVICE` to `SpotStatus` is a change that ripples: `is_available` must return `False`, `available_count` must exclude it, and `capacity` must decide whether it counts. **This is the honest cost of adding a state, and naming the ripple is the answer** — see [design patterns §6.12](../03-design-patterns.md) on forgotten states. The alternative, a separate `out_of_service_until` timestamp checked alongside the status, avoids the ripple at the cost of two fields that must be kept consistent.

### 8.5 Pricing edge cases

| Edge case | Behaviour |
|---|---|
| Exit before entry | `ValueError` — a clock problem, not a pricing one |
| Session inside the grace period | Zero fee |
| Partial hour | Ceiling — a 61-minute stay bills 2 hours |
| Multi-day stay | Daily cap per 24 h period, not per session |
| Lost ticket | A separate flat "lost ticket" charge, out of scope here |
| Currency mismatch | `Money.__add__` raises rather than silently adding |

---

## 9. Design patterns used, and why

| Pattern | Where | Why here | What it costs |
|---|---|---|---|
| **Strategy** | `PricingStrategy`, `SpotAllocator` | Two axes vary independently of the lot's structure; the follow-up is exactly these | An interface, N classes, a wiring point |
| **Factory Method** | `build_lot` | One place constructs the object graph; callers never name concretes | A function that must know every concrete class |
| **Observer** | `OccupancyPublisher` → `GateDisplay`, `FloorMapProjection` | The lot must not know which displays and projections exist | Unordered, lossy delivery; leaks if not unsubscribed; isolation needed |
| **Composite** | `Floor` holds spots; `Campus` holds lots — the same node interface | The multi-lot extension is free | A deeper tree to traverse |
| **Adapter** | `SpotRepository` over the store | The store's error model and key syntax stop at the boundary | A class per adaptor |
| **State** | `SpotStatus` and the `apply_*` transitions | Illegal transitions raise rather than corrupt | A status enum change ripples |
| **Template Method** *(rejected)* | — | Considered for `reserve`/`release` sharing a skeleton with the hotel flow; rejected because the sequence differs enough | — |
| **Singleton** *(rejected)* | — | `ParkingLot` as a singleton would break the multi-lot follow-up and every test | — |

**The two rejections matter as much as the inclusions.** Saying "I considered a Singleton for the lot and rejected it because the multi-lot extension and the test suite both need more than one instance" demonstrates judgement; see [design patterns §4.1.1](../03-design-patterns.md).

**The Observer scope decision, stated precisely:** in-process Observer is right for gate displays and map projections because a missed notification self-heals — the display refreshes, the projection is rebuilt. If the lot ever had to *guarantee* a notification, that path would need a durable queue with idempotent consumers, not a callback list.

---

## 10. Trade-offs

| Choice | Buys | Costs |
|---|---|---|
| `PricingStrategy` interface | New model is one class; the lot is untouched | An interface, a class, a wiring point |
| `SpotAllocator` interface | Allocation policy is swappable | Same |
| Per-spot lock over one global lock | Unrelated spots never contend; throughput scales with spots | One lock object per spot; still not cross-process |
| CAS semantics in the repository | No blocking, no deadlock, immediate failure | One statement per spot; multi-spot needs a transaction |
| `choose` as a hint | Cheap common case; no lock held across allocation | A retry path when the hint is stale |
| TTL on holds | A crashed holder cannot strand a spot | A slow customer loses a spot mid-payment |
| Lazy expiry **and** a sweeper | Correct answers immediately; no accumulation | Both paths to test and keep consistent |
| `Floor.available` computes | No counter race; no lock on the read path | O(spots) per read; fine at 1 000 spots, not at 1 000 000 |
| Observer for gates | The lot is decoupled from displays | Eventual consistency on the display; a leak if not unsubscribed |
| Idempotency key | Safe retries after a timeout | Key storage with a retention window |
| Fee computed before release | A pricing failure cannot lose the spot | A brief window where the ticket is open and the spot is held |
| Gates out of scope | Time spent on the concurrency design | A follow-up may ask for the hardware model |
| Python with no framework | Design is visible; nothing hidden | No persistence or HTTP to point at |
| Integer minor units | No floating-point drift, ever | Formatting and conversion at the edges |
| Immutable `Money` | Thread-safe by construction | Allocation on every arithmetic operation |

**What changes at 10× load:** the in-memory repository becomes a database or Redis, and **every lock in this file disappears** — replaced by `UPDATE ... WHERE status='FREE'` or a Lua script with the same semantics. The class design does not change at all, which is the point of putting the atomicity behind `SpotRepository`.

---

## 11. Interview Q&A

**Q1 (easy). Walk me through the design.**
Five entities: `ParkingLot`, `Floor`, `ParkingSpot`, `Vehicle` and `Ticket`, plus three interfaces for the axes that vary — `PricingStrategy`, `SpotAllocator` and `SpotRepository` — and an injected `Clock`. The lot is composed of floors, floors of spots, and a spot has at most one active ticket, which is the double-booking invariant drawn as cardinality. `reserve` asks the allocator for a candidate, then holds it atomically through the repository using compare-and-set semantics; `release` computes the fee first, then frees the spot and closes the ticket. Concurrency lives in exactly one place: the spot's status transition.

**Q2 (easy). How do you match a vehicle to a spot?**
By size, with an ordering: SMALL, MEDIUM, LARGE, where a vehicle may take its own size or larger. Because the enum is ordered, `spot.size.value >= vehicle.size.value` expresses the rule in one comparison. The allocator does two passes — exact-size spots first, then larger ones — so a bike does not consume a truck spot while smaller spots are free. That two-pass rule is small and it is the difference between a lot that packs well and one that does not.

**Q3 (easy). What is the spot's lifecycle?**
FREE → HELD → OCCUPIED → FREE, with HELD expiring back to FREE. The transitions are methods on the spot that raise on an illegal move, so `confirm` on a `FREE` spot or `release` by the wrong plate is impossible rather than merely unhandled. The interesting one is the HELD → FREE expiry, which is triggered by time rather than by a call, so it is checked lazily on every read for correctness and swept proactively for hygiene.

**Q4 (medium). Two cars arrive at the same instant for the last spot. What happens?**
Both call `choose`, both get the same candidate — because `choose` reads availability and is only a hint. Both then attempt `cas_hold`, which reads and writes under one per-spot lock. One wins and gets a ticket; the other's `cas_hold` returns false, `reserve` re-chooses and retries once, finds nothing, and raises `SpotTakenError` with a clear message. The critical point is that the decision is inside the lock, and `choose` is outside it — treating `choose` as authoritative is the classic double-booking bug.

**Q5 (medium). Why is `PricingStrategy` an interface when you only have one implementation?**
Two reasons. The testability one is true immediately: `FlatPricing(100)` makes the fee assertion exact instead of dependent on a rate table and a grace period, so the abstraction earns its keep in the test suite today. The extension one is that the standard follow-up here is surge or weekend pricing, and with the interface that is one new class and no edits to `ParkingLot`. If neither were true I would use the concrete class and say that I considered and rejected the interface.

**Q6 (medium). What does the gate display do, and how consistent is it?**
It subscribes to occupancy changes and refreshes a count. It is deliberately eventually consistent — a display may briefly show a spot that has just gone. That is acceptable because the display is advisory: the customer learns the truth at reserve time, when `cas_hold` decides. I would say the distinction explicitly, because it is the same reasoning as "the seat map may be stale, the hold may not be": each operation has its own consistency requirement, and only one of them needs to be strong.

**Q7 (medium). What happens if a customer reserves a spot and never arrives?**
The hold has a TTL — fifteen minutes here, longer than a cinema hold because the car is already at the lot. When it expires, `is_available` treats the spot as free immediately, so the next customer can take it even before any background job runs; the sweeper exists to tidy the state so expired holds do not accumulate. That is the two-halves design: lazy expiry is correctness, the sweeper is hygiene, and both are needed.

**Q8 (medium). *Pushback:* "Your `choose` then `cas_hold` is a race. Why not just lock the whole lot?"**
It is a race, and the design says so — `choose` is a hint and `cas_hold` is the decision. Locking the whole lot would be correct and would serialise every reservation in a 1 000-spot lot through one mutex, so a customer entering at gate A waits behind a customer entering at gate Z. Instead there is one lock per spot, held for microseconds and never across I/O, so two vehicles competing for different spots never contend. The cost is 1 000 lock objects, which is nothing. And it only works in one process, so in production the same semantics move into the store as a conditional update.

**Q9 (medium). *Pushback:* "Isn't the Observer indirection overkill for two displays?"**
For two displays, arguably yes — a direct call would be simpler. What the indirection buys is that the lot does not accumulate knowledge of its consumers: today two displays and a map projection, tomorrow an analytics stream and a mobile push. Each of those would otherwise be a new line in the lot and a new reason for the lot to change. The costs are real and I would state them: delivery is unordered and best-effort, an observer that throws must be isolated so it cannot break the state transition, and an observer that is never unsubscribed leaks. I subscribe with a returned disposer for exactly that reason.

**Q10 (hard). *Pushback:* "This only works in one process. Isn't the whole concurrency design useless?"**
It is exactly one process, and the design is built so that does not matter. `SpotRepository` is the seam: `InMemorySpotRepository` implements compare-and-set semantics with a per-spot lock, and the production implementation implements the *same* semantics with `UPDATE parking_spots SET status='HELD' WHERE spot_id=%s AND status='FREE'` and a rowcount check. Both satisfy the same contract, the class design above is unchanged, and the difference is where the atomicity physically lives. That is why I put the repository behind an interface instead of calling a store directly — the concurrency mechanism is the one thing most likely to change between the interview and production.

**Q11 (hard). *Pushback:* "If the TTL expires while the customer is paying, they lose the spot. Isn't that a bug?"**
It is a real trade-off, not a bug, and it has three mitigations. The TTL is set comfortably longer than the median transaction — fifteen minutes here. Expiry is checked at check-in, and the check-in failure is loud and specific rather than silent, so the customer is told the spot is gone rather than being seated in someone else's space. And the ordering means the customer is never charged for a spot they did not get: the hold precedes payment, and check-in fails before anything is charged. The alternative — no TTL — means one abandoned reservation blocks a spot forever, which is worse for every other customer.

**Q12 (hard). What would you do differently at 10× the load?**
The class design does not change; the store does. The in-memory repository and every lock in this file are replaced by a database or Redis with the same compare-and-set semantics, and the hot path becomes a single conditional statement per spot. Beyond that: the availability read moves to a cache with a short TTL, because it is a hint and a one-second-stale count is fine; the sweeper becomes a distributed job with a lease so two pods do not sweep the same rows; and the pricing calculation, being stateless, is trivially replicable. The one thing I would watch is the hot size class — if MEDIUM spots are the contention point, the conditional update never blocks so it scales, but the *allocation* scan becomes O(spots) per attempt and wants an index.

**Q13 (medium). How would you test this?**
Three kinds, and none of them uses `sleep`. First, the golden path: reserve, check in, advance the fake clock, release, assert the exact fee in integer minor units. Second, the boundaries: expiry at exactly the TTL with assertions on both sides of it, the last spot in a lot, a release by the wrong plate, a duplicate release. Third, the invariant, which is the one that matters: two threads with a barrier, one spot, and an assertion that exactly one wins — plus an idempotency test proving a retried key does not consume a second spot, and an isolation test proving a throwing observer does not break the reservation.

**Q14 (medium). What is out of scope, and why?**
Payment, gate hardware, notification and the mobile app. Payment is out of scope as an *entity* because it is an adapter over an external system with no interesting structure — but its ordering constraint is not out of scope, and the design enforces it: hold before charge, and never charge for a spot you cannot give. Gate hardware is out of scope because the barrier's control protocol is a driver concern, and the lot's behaviour is unaffected by how the barrier is told to lift. Notification is an Observer with an eventual-consistency guarantee. Saying what is out of scope, and why, is worth more than a half-built payment class.

**Q15 (hard). Why does `Floor.available` compute rather than keep a counter?**
Because a counter is a read-modify-write race that would need its own lock, for a value that is only used for display and allocation hints. If it were wrong by one for a second, nothing of consequence happens; the customer is told the truth at hold time. Computing it is O(spots per floor) — trivial at a thousand spots, and it makes an entire class of bug impossible. If the lot grew to hundreds of thousands of spots I would revisit it with an atomic counter and accept the eventual consistency explicitly, which is exactly the trade I would state.

---

## 12. Cheat-sheet summary

**The design in one screen**

```
  ParkingLot ──◆ 1..* Floor ──◆ 0..* ParkingSpot ──0..1── 0..1 Ticket ──*── 1 Vehicle
       │                                              (the double-booking invariant)
       ├── PricingStrategy   «interface»  → HourlyPricing | FlatPricing | SurgePricing
       ├── SpotAllocator     «interface»  → NearestFirst | Balanced | FeatureAware
       ├── SpotRepository    «interface»  → InMemory | Sql | Redis   ← the ATOMIC step
       └── Clock             «interface»  → RealClock | FakeClock
```

**Core operations**

| Operation | Steps |
|---|---|
| `reserve(v, key)` | choose *(hint)* → `cas_hold` *(decision)* → on failure re-choose, retry once → `Ticket` |
| `check_in(t)` | Check the hold has not expired → `cas_occupy` → OCCUPIED |
| `release(t)` | Compute the fee → `cas_release` → `Ticket.close` → `Money` |
| `sweep()` | Return expired holds to FREE — hygiene, not correctness |

**Invariants**

| # | Invariant | Enforced by |
|---|---|---|
| I1 | One active ticket per spot | `cas_hold` + the `0..1` cardinality |
| I2 | A vehicle takes its own size or larger | `ParkingSpot.fits` + the allocator's two passes |
| I3 | An expired hold is free immediately | `is_available` (lazy) + `sweep` (proactive) |
| I4 | A released spot is FREE | `apply_release` with the plate check |
| I5 | A fee is computed once | `Ticket.close`, idempotent |
| I6 | No negative occupancy | `available` computes, never stores |

**Concurrency, in five lines**

```
  RACE:      two actors read FREE, both write HELD
  FIX:       is_available + apply_hold inside ONE per-spot lock (CAS semantics)
  SCOPE:     per-spot, never global; held for microseconds; never across I/O
  RECOVERY:  TTL + lazy expiry + sweeper + idempotency key
  CROSS-POD: the lock moves into the store — UPDATE ... WHERE status='FREE'
```

**Numbers**

| Parameter | Value |
|---|---|
| Hold TTL | 15 min (longer than a cinema hold — the car is already there) |
| Grace period | 15 min free |
| Retry on a lost race | 1 |
| Sweeper interval | 1–5 min, batch 500 |
| Lock scope | One per spot |
| Fee rounding | Ceiling, per started hour; daily cap per 24 h |
| Money | Integer minor units |

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "Two cars, one spot?" | "`choose` is a hint, `cas_hold` is the decision — one per-spot lock, and the loser retries and fails cleanly." |
| "Why not one lock?" | "Per-spot so unrelated spots never contend; global would serialise the whole lot." |
| "Cross-pod?" | "The semantics move into the store as a conditional update — `SpotRepository` is the seam." |
| "Now add surge pricing" | "A new `PricingStrategy`; `ParkingLot` is untouched — that is the Open/Closed payoff." |
| "Now add lots" | "One `Campus` above the lot with an aggregation arrow; nothing below changes." |
| "Add EV spots" | "A feature set on the spot and a filtering allocator — not a subclass, because the behaviour is identical." |
| "Reserve for tomorrow" | "That changes the entity model — a `Reservation` with a `TimeRange`, and the invariant becomes non-overlap." |
| "How do you test it?" | "A `FakeClock`, a barrier for the invariant, and idempotent-release and observer-isolation tests. No `sleep`." |

---

## Cross-links

- The concept files behind this design: [What is LLD](../01-what-is-lld-and-how-to-answer.md) · [OOP principles and SOLID](../02-oop-principles-and-solid.md) · [Design patterns](../03-design-patterns.md) · [Concurrency in LLD](../04-concurrency-and-threading-in-lld.md) · [UML and class diagrams](../05-uml-and-class-diagrams.md) · [The machine-coding framework](../06-machine-coding-interview-framework.md)
- The same concurrency pattern, solved at the store and distributed level: [BookMyShow seat locking](04-bookmyshow-seat-locking.md) · [BookMyShow seat booking case study](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md) · [Redis deep dive](../../HLD/03-scaling/03-redis-deep-dive.md)
- The same date-range problem, harder: [Hotel management](03-hotel-management.md)
- Other problems in this pillar: [LRU and LFU cache](02-lru-lfu-cache.md) · [Elevator system](05-elevator-system.md) · [Splitwise](06-splitwise.md) · [ATM machine](07-atm-machine.md) · [Library management](08-library-management.md)
- The HLD view of the same system boundary: [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md) · [The five-step framework](../../HLD/01-foundations/04-five-step-interview-framework.md)
