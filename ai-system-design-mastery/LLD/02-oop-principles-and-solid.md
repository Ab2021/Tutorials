# OOP Principles and SOLID — The Four Pillars, Five Principles, and When Each Fails

> **Grounding note.** The source transcript names OOP concepts only once, at **[25:54]–[27:38]**: LLD is when *"you come to a class... and you draw a class diagram, define the methods, apply the OOP concept."* It does not teach encapsulation, inheritance, polymorphism, abstraction or SOLID. The reference repositories contain no OOP or LLD material. **Everything in this file is standard industry practice**, and the examples are drawn from the systems this repository already documents so that the theory lands somewhere concrete. The one thing the transcript does supply — the parking-lot class hierarchy — is used as the running example where it fits.

---

## TL;DR

The four pillars are **encapsulation** (hide state behind behaviour, so an invariant can never be violated from outside), **abstraction** (expose *what*, hide *how*), **inheritance** (share a contract or a representation up a hierarchy), and **polymorphism** (one call site, many implementations — the mechanism that makes the other three useful at runtime). SOLID is five rules that keep a codebase changeable: **S**ingle Responsibility, **O**pen/Closed, **L**iskov Substitution, **I**nterface Segregation, **D**ependency Inversion. The single most valuable sentence in LLD is: **prefer composition over inheritance**, and the second is **abstract the axis of change, not the axis of imagination**. In an interview, pillars and principles are not vocabulary to recite — they are the justification for a class boundary. Every boundary you draw should trace to one of them, and every abstraction you introduce should have a stated cost.

> **60-second interview answer:** "The four pillars are encapsulation, abstraction, inheritance and polymorphism. Encapsulation means an object's invariants are enforced by its own methods, so no caller can put it in a bad state. Abstraction means a caller depends on `PricingStrategy.calculate(...)` and not on how surge pricing is computed. Inheritance shares a contract or a common representation; polymorphism is what lets one call site work for many implementations. SOLID is five rules that keep the code changeable — one reason to change per class, open for extension closed for modification, subtypes that are genuinely substitutable, interfaces that are small and role-specific, and dependencies on abstractions rather than concrete classes. And I default to composition over inheritance, because inheritance is a permanent, compile-time, one-parent coupling, and most of the time what I actually want is a collaborator I can swap."

---

## 1. Definition & mental model

### 1.1 The four pillars

| Pillar | Plain English | The test that proves you have it |
|---|---|---|
| **Encapsulation** | State is private; behaviour is public; the object cannot be put into an invalid state from outside | Can any caller produce an illegal state by setting a field directly? If yes, it is not encapsulated |
| **Abstraction** | A caller depends on *what* is offered, never on *how* it is done | Can you replace the implementation without touching a single caller? |
| **Inheritance** | A subtype shares a contract (interface) or a representation (base class) with its parent | Does the subtype genuinely *is-a* the parent, in every context where the parent is used? |
| **Polymorphism** | One call site dispatches to many implementations at runtime | Can you add a new implementation without editing the call site? |

**The relationship between them, because they are not four independent ideas:**

```
   ABSTRACTION      defines the contract the caller sees
        │
        ▼
   ENCAPSULATION    hides the implementation behind that contract,
                    so the contract cannot be bypassed
        │
        ▼
   INHERITANCE      one way to provide alternative implementations
        │            (composition is the other, better way)
        ▼
   POLYMORPHISM     the call site works for every implementation,
                    including ones written after it
```

Encapsulation without abstraction is a private field nobody can use. Abstraction without polymorphism is a header file with one implementation — useful for documentation, useless for change. Polymorphism without encapsulation leaks the implementation you were abstracting. **They are one mechanism described at four levels.**

### 1.2 The analogy

**Encapsulation is a vending machine.** You put money in and press a button. You cannot reach in and take the item, and you cannot reach in and add items. The machine enforces the invariants (money in before item out) precisely because you cannot touch its internals.

**Abstraction is the button panel.** You press "A4". The label is what you depend on. Whether A4's mechanism is a spiral, a robotic arm or a person behind the wall is not your problem.

**Inheritance is a family recipe.** The base recipe says "add the spices". Every variant adds its own spices. That works fine — until a variant needs to *omit* the spices, and now you are fighting the recipe.

**Polymorphism is a universal power socket.** The socket does not know what you plug into it. Every plug that satisfies the shape works — including ones designed after the socket was installed.

**Composition over inheritance, as an image:** inheriting from a base class is *welding* two pieces of metal together — rigid, permanent, hard to separate. Composing with a collaborator is *bolting* — the same strength where it matters, and separable when you need to change one part.

### 1.3 What OOP is *not*

| Not this | Why the confusion exists | The distinction |
|---|---|---|
| **Bundling data with methods** | That is what a class syntax looks like | That is a struct with functions. Encapsulation requires *invariants enforced*, not *fields grouped* |
| **A class hierarchy as deep as possible** | Textbooks show animal taxonomies | Deep hierarchies are the most expensive thing in OOP. Two levels is usually the ceiling |
| **`getX()` and `setX()` on every field** | "Encapsulation" is taught as "make fields private, add accessors" | Accessors that expose every field are encapsulation theatre. Real encapsulation exposes *behaviour* |
| **Inheritance as code reuse** | The first thing inheritance is used for | Inheriting to reuse one method is the classic anti-pattern. Compose, or extract a free function |
| **Patterns as the goal** | Patterns get named in interviews | A pattern is a *consequence* of a requirement, never the requirement |

---

## 2. Why it exists — the problem it solves

### 2.1 The five problems OOP exists to solve

| Problem without OOP | The mechanism that solves it |
|---|---|
| An object can be put into an invalid state by any caller, arbitrarily far from its code | **Encapsulation**: state is mutated only through methods that enforce invariants |
| Every caller must know the implementation, so a change breaks all of them | **Abstraction**: callers depend on a contract that changes far less often |
| One `if/elif` chain grows with every new case, and edits the same function forever | **Polymorphism**: a new case is a new class; the dispatch site never changes |
| Shared code is copy-pasted, so a bug fix must be applied in twelve places | **Inheritance** (for genuine is-a) or **composition** (for everything else) |
| The design's seams are invisible, so it cannot be tested | Abstraction creates the seam: an interface can be substituted with a test double |

### 2.2 What breaks without SOLID — five concrete symptoms

These are the symptoms to watch for in your own designs, and they are what the interviewer is probing when they ask "how would you add X?"

| Principle violated | The symptom you can see | The real-world cost |
|---|---|---|
| **SRP** | One class changes for five unrelated reasons; every team's PR touches it | Merge conflicts, and a change for billing risk breaking a seating rule |
| **OCP** | Every new feature edits a central `switch` or `if/elif` | Linear risk growth: feature N has N-1 chances to break something |
| **LSP** | A subtype that throws `NotImplementedError` on a parent method; callers must check the concrete type | Polymorphism stops working; callers get `isinstance` checks |
| **ISP** | A fat interface with 20 methods, where each implementation implements 3 and stubs 17 | Every interface change breaks every implementation |
| **DIP** | A service instantiates `new RedisClient()` in its constructor | Untestable without a live Redis; cannot swap the client |

### 2.3 The honest caveat

**SOLID is heuristics, not laws.** Each principle has a real cost — indirection, more files, more wiring — and applying all five maximally produces a codebase of one-method classes and four layers of interfaces that is *harder* to change than the naive version. The correct reading of SOLID is: **each violation is a cost you may choose to pay knowingly.** A `switch` over three vehicle types that has not changed in four years is fine. An interface with one implementation may be fine. What is not fine is not knowing which one you are paying.

This matters in interviews: interviewers reward "I would not abstract this yet" as much as "I would introduce an interface here", and the second answer without the first reads as over-engineering.

---

## 3. Internal mechanics — the four pillars, in code

The running example is the parking lot from the source transcript **[26:48]–[27:38]**, extended where the pillar needs it. All code is Python; the design is language-independent.

### 3.1 Encapsulation

**Naive — no encapsulation.** Every invariant is a comment hoping to be respected:

```python
class ParkingSpot:
    def __init__(self, spot_id, size):
        self.spot_id = spot_id
        self.size = size
        self.occupied = False          # public: any caller can flip this
        self.held_by = None            # public
        self.held_until = None         # public
```

Three callers can now produce three invalid states:

```python
spot.occupied = True               # occupied with no vehicle, no ticket
spot.held_by = "user-7"            # held while also occupied
spot.occupied = False              # freed while a ticket is still open
```

**Encapsulated — invariants enforced by the object itself:**

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class SpotSize(Enum):
    SMALL = "SMALL"      # bike
    MEDIUM = "MEDIUM"    # car
    LARGE = "LARGE"      # truck


class SpotState(Enum):
    FREE = "FREE"
    HELD = "HELD"
    OCCUPIED = "OCCUPIED"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class SpotInvariantError(RuntimeError):
    """Raised when an operation would violate a parking-spot invariant."""


@dataclass
class ParkingSpot:
    """A single parking spot.

    Invariants enforced by this class, not by its callers:
      I1. state is one of FREE, HELD, OCCUPIED, OUT_OF_SERVICE
      I2. only a FREE spot can be held
      I3. only a HELD spot can be converted to OCCUPIED
      I4. only a HELD or OCCUPIED spot can be freed
      I5. held_until is set iff state is HELD
    """

    spot_id: str
    size: SpotSize
    _state: SpotState = field(default=SpotState.FREE, repr=False)
    _held_by: Optional[str] = field(default=None, repr=False)
    _held_until: Optional[float] = field(default=None, repr=False)

    # ---- read-only views: the outside world can look, never touch ----
    @property
    def state(self) -> SpotState:
        self._expire_hold_if_due()
        return self._state

    @property
    def is_available(self) -> bool:
        return self.state is SpotState.FREE

    @property
    def held_by(self) -> Optional[str]:
        self._expire_hold_if_due()
        return self._held_by

    # ---- the only mutators, each guarding an invariant ----
    def hold(self, user_id: str, ttl_seconds: float, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._expire_hold_if_due(now)
        if self._state is SpotState.OUT_OF_SERVICE:
            raise SpotInvariantError(f"{self.spot_id} is out of service")
        if self._state is not SpotState.FREE:
            raise SpotInvariantError(
                f"{self.spot_id} cannot be held from state {self._state.value}"
            )
        self._state = SpotState.HELD
        self._held_by = user_id
        self._held_until = now + ttl_seconds

    def occupy(self) -> None:
        self._expire_hold_if_due()
        if self._state is not SpotState.HELD:
            raise SpotInvariantError(
                f"{self.spot_id} cannot be occupied from state {self._state.value}"
            )
        self._state = SpotState.OCCUPIED
        self._held_by = None
        self._held_until = None

    def release(self) -> None:
        if self._state not in (SpotState.HELD, SpotState.OCCUPIED):
            raise SpotInvariantError(
                f"{self.spot_id} cannot be released from state {self._state.value}"
            )
        self._state = SpotState.FREE
        self._held_by = None
        self._held_until = None

    def take_out_of_service(self) -> None:
        self._expire_hold_if_due()
        if self._state is SpotState.OCCUPIED:
            raise SpotInvariantError(f"{self.spot_id} is occupied")
        self._state = SpotState.OUT_OF_SERVICE
        self._held_by = None
        self._held_until = None

    # ---- private helper: the TTL is enforced inside the object ----
    def _expire_hold_if_due(self, now: Optional[float] = None) -> None:
        if self._state is not SpotState.HELD:
            return
        now = time.monotonic() if now is None else now
        if self._held_until is not None and now >= self._held_until:
            self._state = SpotState.FREE
            self._held_by = None
            self._held_until = None
```

**What encapsulation bought, concretely:**

- The four invalid states above are now **impossible to express**. `hold()` on an occupied spot raises; `occupy()` on a free spot raises; `release()` on a free spot raises.
- The TTL is enforced *inside the object*. No caller can forget to check expiry, because the read path checks it. (This is the "self-healing invariant" pattern — the object repairs itself on observation.)
- Every invalid transition has a **defined, testable answer**: an exception with a message naming the state pair.

**The design rule that falls out of this:** *an object owns its invariants.* If enforcing an invariant requires the caller to remember something, the invariant is not encapsulated — it is a convention.

### 3.2 Abstraction

**Abstraction is the choice of what the caller is allowed to depend on.** The test: can you replace the implementation without editing a caller?

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    spot_id: str
    vehicle_plate: str
    entry_time: float
    lot_id: str


class PricingStrategy(ABC):
    """The abstraction the lot depends on.

    The lot knows: a price is a function of a ticket and a clock.
    The lot does NOT know: surge multipliers, tier tables, weekend rules,
    tax, or how money is rounded. None of that leaks across this boundary.
    """

    @abstractmethod
    def calculate(self, ticket: Ticket, exit_time: float) -> int:
        """Return the fee in the smallest currency unit (paise/cents)."""

    @abstractmethod
    def describe(self) -> str:
        """A human-readable name, used on the receipt and in logs."""


class HourlyPricing(PricingStrategy):
    def __init__(self, rate_per_hour_minor: int, free_minutes: int = 15) -> None:
        self._rate = rate_per_hour_minor
        self._free_minutes = free_minutes

    def calculate(self, ticket: Ticket, exit_time: float) -> int:
        seconds = max(0.0, exit_time - ticket.entry_time)
        billable = max(0.0, seconds - self._free_minutes * 60)
        hours = -(-billable // 3600)          # ceiling division: part-hour bills as an hour
        return int(hours) * self._rate

    def describe(self) -> str:
        return f"Hourly @ {self._rate} minor units/hour"


class FlatPricing(PricingStrategy):
    def __init__(self, flat_minor: int) -> None:
        self._flat = flat_minor

    def calculate(self, ticket: Ticket, exit_time: float) -> int:
        return self._flat

    def describe(self) -> str:
        return f"Flat {self._flat}"
```

**Notice what is *not* in the abstraction.** `PricingStrategy` says nothing about surge, discounts, tax or currency. Those are implementation decisions. If they had been in the interface, every future strategy would carry them whether they needed them or not — which is the interface-segregation failure.

**Abstraction is not free.** `PricingStrategy` costs you an indirection, a file, and a wiring point in `ParkingLot`. The justification is that a second pricing model is *imminent* (the interviewer will ask for it, and real lots have several). If neither were true, `HourlyPricing` alone would be correct and the interface would be speculation. Say that trade-off out loud — see §7.

### 3.3 Inheritance — and when it is the wrong tool

**The transcript's own hierarchy uses inheritance for entities.** Here it is extended to vehicles, which is the case inheritance is genuinely good at (a truck *is* a vehicle, in every context):

```python
class VehicleType(Enum):
    BIKE = "BIKE"
    CAR = "CAR"
    TRUCK = "TRUCK"


class Vehicle(ABC):
    """Base contract. Subclasses supply what differs; nothing else."""

    def __init__(self, plate: str) -> None:
        if not plate or not plate.strip():
            raise ValueError("plate is required")
        self._plate = plate.strip().upper()

    @property
    def plate(self) -> str:
        return self._plate

    @property
    @abstractmethod
    def vehicle_type(self) -> VehicleType: ...

    @property
    @abstractmethod
    def required_spot_size(self) -> SpotSize: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._plate})"


class Bike(Vehicle):
    @property
    def vehicle_type(self) -> VehicleType:
        return VehicleType.BIKE

    @property
    def required_spot_size(self) -> SpotSize:
        return SpotSize.SMALL


class Car(Vehicle):
    @property
    def vehicle_type(self) -> VehicleType:
        return VehicleType.CAR

    @property
    def required_spot_size(self) -> SpotSize:
        return SpotSize.MEDIUM


class Truck(Vehicle):
    @property
    def vehicle_type(self) -> VehicleType:
        return VehicleType.TRUCK

    @property
    def required_spot_size(self) -> SpotSize:
        return SpotSize.LARGE
```

**Why this inheritance is legitimate:**

1. **It is a genuine is-a.** Every `Car` is a `Vehicle` in every context, forever. There is no context in which the substitution fails.
2. **The base class is abstract and thin.** It supplies one behaviour (plate normalisation) and two contracts. There is no behaviour for a subclass to accidentally break.
3. **New types are additive.** Adding `ElectricCar` adds one class and edits nothing — Open/Closed satisfied.
4. **There is a substitutable contract.** Any function taking a `Vehicle` works with any subtype.

**Now the same problem, done badly — the two classic inheritance failures:**

```python
# FAILURE 1 — inheritance for reuse. A Floor is NOT a ParkingLot.
class ParkingLot:
    def __init__(self) -> None:
        self._spots: list[ParkingSpot] = []

    def add_spot(self, spot: ParkingSpot) -> None:
        self._spots.append(spot)

    def count_free(self) -> int:
        return sum(1 for s in self._spots if s.is_available)


class Floor(ParkingLot):                # inherits count_free() ... and everything else
    def __init__(self, floor_id: int) -> None:
        super().__init__()
        self.floor_id = floor_id


# Consequence: Floor now has add_spot, count_free, AND can be passed anywhere a
# ParkingLot is expected. Callers of a ParkingLot that receive a Floor get a
# silently-different meaning of "the lot's spots".
```

```python
# FAILURE 2 — the subtype that cannot honour the contract.
class Vehicle(ABC):
    @abstractmethod
    def required_spot_size(self) -> SpotSize: ...


class Bicycle(Vehicle):
    @property
    def required_spot_size(self) -> SpotSize:
        raise NotImplementedError("bicycles do not need a parking spot")
        # ^ Liskov violation. The base contract promised a size for EVERY vehicle.
        #   A caller iterating vehicles and calling required_spot_size() now crashes.
```

**The correct fix for both is composition:**

```python
class Floor:
    """A floor COMPOSES spot storage; it does not inherit it.

    It reuses the *behaviour* it needs (a spot collection with a free count
    per size) without acquiring the identity of a ParkingLot.
    """

    def __init__(self, floor_id: int, spots: list[ParkingSpot]) -> None:
        self._floor_id = floor_id
        self._spots = list(spots)
        self._free_by_size: dict[SpotSize, int] = {}
        for spot in self._spots:
            if spot.is_available:
                self._free_by_size[spot.size] = self._free_by_size.get(spot.size, 0) + 1

    @property
    def floor_id(self) -> int:
        return self._floor_id

    def find_available_spot(self, size: SpotSize) -> ParkingSpot | None:
        for spot in self._spots:
            if spot.size is size and spot.is_available:
                return spot
        return None

    def free_count(self, size: SpotSize) -> int:
        return self._free_by_size.get(size, 0)

    def release(self, spot: ParkingSpot) -> None:
        spot.release()
        self._free_by_size[spot.size] = self._free_by_size.get(spot.size, 0) + 1


class VehicleSizeResolver:
    """A small collaborator, INJECTED rather than inherited.

    Inheritance would have meant `class Bicycle(Vehicle)`. Composition lets a
    bicycle exist as a vehicle with no spot requirement at all — the resolver
    answers the question instead of the object promising an answer it lacks.
    """

    _SIZE_BY_TYPE = {
        VehicleType.BIKE: SpotSize.SMALL,
        VehicleType.CAR: SpotSize.MEDIUM,
        VehicleType.TRUCK: SpotSize.LARGE,
    }

    def size_for(self, vehicle_type: VehicleType) -> SpotSize | None:
        return self._SIZE_BY_TYPE.get(vehicle_type)


class Bicycle(Vehicle):
    """No required spot size — it is not asked, because the resolver is asked."""

    def __init__(self, plate: str) -> None:
        super().__init__(plate)

    @property
    def vehicle_type(self) -> VehicleType:
        return VehicleType.BIKE

    @property
    def required_spot_size(self) -> SpotSize:
        return SpotSize.SMALL        # honest: it does use the small racks
```

**The three-question test before writing `class B extends A`:**

```
1. IS-A TEST
   Is every B usable everywhere an A is, with no caller needing to check
   the concrete type? If any caller must ask "is this really an A?",
   the answer is no.

2. CONTRACT TEST
   Can B honour EVERY postcondition the base class promises, including the
   ones it does not need? If B must weaken a precondition, throw on a base
   method, or return a sentinel, the answer is no.

3. CHANGE-AXIS TEST
   Do A and B vary for the same reason? If A changes when pricing changes
   and B changes when seating changes, they are different axes and
   inheritance couples them forever. The answer is no.
```

**Two of three "no" means composition.** This is not a guideline to be weighed — it is a rule that prevents the two most expensive code structures in OOP (the deep hierarchy and the broken substitution).

### 3.4 Polymorphism

**Subtype polymorphism** is what makes the previous sections pay off. One call site, many implementations, and the call site never changes:

```python
class ParkingLot:
    def __init__(
        self,
        lot_id: str,
        floors: list[Floor],
        allocator: "SpotAllocator",
        pricing: PricingStrategy,
    ) -> None:
        self._lot_id = lot_id
        self._floors = floors
        self._allocator = allocator      # injected: polymorphic behaviour
        self._pricing = pricing          # injected: polymorphic behaviour
        self._open_tickets: dict[str, Ticket] = {}

    def reserve(self, vehicle: Vehicle, now: float) -> Ticket:
        spot = self._allocator.allocate(self._floors, vehicle.required_spot_size)
        if spot is None:
            raise LotFullError(f"no {vehicle.required_spot_size.value} spot available")
        spot.occupy()
        ticket = Ticket(
            ticket_id=uuid.uuid4().hex,
            spot_id=spot.spot_id,
            vehicle_plate=vehicle.plate,
            entry_time=now,
            lot_id=self._lot_id,
        )
        self._open_tickets[ticket.ticket_id] = ticket
        return ticket

    def release(self, ticket_id: str, now: float) -> int:
        ticket = self._open_tickets.pop(ticket_id, None)
        if ticket is None:
            raise UnknownTicketError(ticket_id)
        fee = self._pricing.calculate(ticket, now)          # ← the polymorphic call
        spot = self._find_spot(ticket.spot_id)
        spot.release()
        self._floor_of(spot).release(spot)
        return fee

    # helpers omitted from earlier listings
    def _find_spot(self, spot_id: str) -> ParkingSpot:
        for floor in self._floors:
            for spot in floor._spots:
                if spot.spot_id == spot_id:
                    return spot
        raise KeyError(spot_id)

    def _floor_of(self, spot: ParkingSpot) -> Floor:
        for floor in self._floors:
            if spot in floor._spots:
                return floor
        raise KeyError(spot.spot_id)
```

```python
class SpotAllocator(ABC):
    @abstractmethod
    def allocate(self, floors: list[Floor], size: SpotSize) -> ParkingSpot | None: ...


class NearestFirstAllocator(SpotAllocator):
    """Lowest floor first — minimises walking for the driver."""

    def allocate(self, floors: list[Floor], size: SpotSize) -> ParkingSpot | None:
        for floor in floors:
            spot = floor.find_available_spot(size)
            if spot is not None:
                return spot
        return None


class BalancedAllocator(SpotAllocator):
    """Spreads occupancy so no floor fills up first — better for throughput
    at exit time, because exits are not funnelled through one floor."""

    def allocate(self, floors: list[Floor], size: SpotSize) -> ParkingSpot | None:
        candidates = [f for f in floors if f.free_count(size) > 0]
        if not candidates:
            return None
        best = max(candidates, key=lambda f: f.free_count(size))
        return best.find_available_spot(size)
```

**The demonstration of polymorphism's value:** adding `BalancedAllocator` required **zero edits** to `ParkingLot`, `Floor` or `ParkingSpot`. The call site `self._allocator.allocate(...)` is unchanged. That is Open/Closed, delivered by polymorphism and dependency inversion together.

**The other three kinds of polymorphism**, which candidates should be able to name:

| Kind | Mechanism | Example | When it is the right tool |
|---|---|---|---|
| **Subtype** (runtime dispatch) | Virtual method override | `PricingStrategy.calculate` | Multiple interchangeable behaviours |
| **Parametric** (generics) | Type parameters | `list[ParkingSpot]`, `Cache[K, V]` | Container/algorithm reuse with type safety |
| **Ad-hoc** (overloading / duck typing) | Same name, different signatures | `reserve(vehicle)`, `reserve(vehicle, now)` | Convenience APIs; careful — it hides which one ran |
| **Coercion** | Implicit conversion | `int` where `float` expected | Numeric code; risky in money code — see §6.7 |

---

## 4. SOLID — each principle with a violation, a fix, and a use from this repo

### 4.1 S — Single Responsibility Principle

> **A class should have one reason to change — one axis of change, one actor it answers to.**

**Violation** — a booking service that changes for pricing, notification, persistence and audit reasons:

```python
class BookingService:
    def book(self, user_id: str, show_id: str, seats: list[str]) -> dict:
        # (a) validates seat availability          → changes when seating rules change
        # (b) computes the price                    → changes when pricing changes
        # (c) writes to the database                → changes when the schema changes
        # (d) sends a confirmation email            → changes when the provider changes
        # (e) writes an audit record                → changes when compliance changes
        rows = self._db.execute("SELECT status FROM seats WHERE show_id = %s", (show_id,))
        for seat in seats:
            if rows[seat] != "free":
                raise ValueError(f"{seat} is taken")
        total = sum(self._prices[show_id][seat] for seat in seats)
        self._db.execute("INSERT INTO bookings (user_id, show_id) VALUES (%s, %s)",
                         (user_id, show_id))
        self._mailer.send(user_id, f"Booked {seats} for {total}")
        self._audit.write("booking", user_id, show_id, seats)
        return {"total": total, "seats": seats}
```

Five actors, five reasons to change, one file, and every team's PR touches it. That is the SRP violation, stated in the terms that matter: **not "the class is big" but "the class has five different people who can demand a change."**

**Fix** — split by axis of change, and let each class depend on the next through an interface:

```python
class SeatAvailabilityChecker:
    def ensure_available(self, show_id: str, seats: list[str]) -> None: ...


class PriceCalculator:
    def total(self, show_id: str, seats: list[str], tier: str) -> int: ...


class BookingRepository:
    def save(self, booking: "Booking") -> None: ...


class ConfirmationNotifier:
    def send(self, booking: "Booking") -> None: ...


class AuditLog:
    def record(self, event: str, payload: dict) -> None: ...


class BookingService:
    """Orchestration only. It composes collaborators; it does not implement them."""

    def __init__(
        self,
        availability: SeatAvailabilityChecker,
        pricing: PriceCalculator,
        repo: BookingRepository,
        notifier: ConfirmationNotifier,
        audit: AuditLog,
    ) -> None:
        self._availability = availability
        self._pricing = pricing
        self._repo = repo
        self._notifier = notifier
        self._audit = audit

    def book(self, user_id: str, show_id: str, seats: list[str]) -> "Booking":
        self._availability.ensure_available(show_id, seats)
        amount = self._pricing.total(show_id, seats, tier="STANDARD")
        booking = Booking.create(user_id, show_id, seats, amount)
        self._repo.save(booking)
        self._audit.record("booking.created", {"booking_id": booking.id})
        self._notifier.send(booking)
        return booking
```

**Use from this repo.** The notification system ([Case-Studies/HLD/03-notification-system.md](../Case-Studies/HLD/03-notification-system.md)) separates *template rendering*, *channel selection*, *provider dispatch* and *delivery tracking*. Those are four actors: marketing changes templates, product changes channel rules, ops changes providers, analytics changes tracking. In the LLD, that is four classes.

**The honest caveat.** SRP pushed to its limit gives one-method classes and a 200-line orchestrator. The real rule is **cohesion**: things that change together stay together. `Ticket` holding its id, spot, vehicle and entry time is one responsibility — it is the record of an occupancy — even though it has four fields.

### 4.2 O — Open/Closed Principle

> **Open for extension, closed for modification.** You should be able to add behaviour by adding code, not by editing existing code.

**Violation** — the `if/elif` ladder that grows forever:

```python
def calculate_fee(vehicle_type: str, hours: float) -> int:
    if vehicle_type == "BIKE":
        return int(hours * 1000)
    elif vehicle_type == "CAR":
        return int(hours * 2500)
    elif vehicle_type == "TRUCK":
        return int(hours * 5000)
    # adding EV means editing THIS function — and re-testing all of it
    else:
        raise ValueError(vehicle_type)
```

**Fix** — the polymorphism version, which adds cases without editing:

```python
class VehiclePricing(PricingStrategy):
    """One class per vehicle class. Adding a class adds a case."""

    _RATE_BY_TYPE = {
        VehicleType.BIKE: 1000,
        VehicleType.CAR: 2500,
        VehicleType.TRUCK: 5000,
    }

    def calculate(self, ticket: Ticket, exit_time: float) -> int:
        vehicle_type = self._lookup_vehicle_type(ticket.vehicle_plate)
        rate = self._RATE_BY_TYPE[vehicle_type]
        hours = max(1, -(-(exit_time - ticket.entry_time) // 3600))
        return int(hours) * rate

    def _lookup_vehicle_type(self, plate: str) -> VehicleType:
        raise NotImplementedError   # supplied by the repository in a real system

    def describe(self) -> str:
        return "Per-vehicle-class hourly"
```

**Use from this repo.** The guardrail pipeline ([HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md)) has four layers — input, content trust, output, action. Adding a new guardrail (say, a PII redactor) should add a stage, not edit the pipeline. In the LLD that is a list of `GuardrailStage` implementations, each with `check(payload) -> Verdict`, registered into a chain. The pipeline is closed; the stage list is open.

**The honest caveat — and the one interviewers reward.** OCP is the principle most often over-applied. **If the `elif` ladder has three cases and no fourth is coming, the ladder is correct.** OCP's cost is indirection: an interface, N classes, and a registry. The right formulation is: *"abstract the axis that actually changes, not the axis you can imagine changing."* In the parking lot, vehicle classes change (EV, oversized); so does pricing (surge, weekend, loyalty). Those get interfaces. The gate hardware does not change, so gates do not get a factory.

### 4.3 L — Liskov Substitution Principle

> **Objects of a subtype must be usable wherever the supertype is expected, without the caller knowing.**

**Violation** — the classic "a square is a rectangle" family, translated to this repo:

```python
class SeatHold:
    """Base: a hold may be extended while it is active."""

    def __init__(self) -> None:
        self._expires_at: float = 0.0
        self._extension_count: int = 0

    def extend(self, seconds: float) -> None:
        self._expires_at += seconds
        self._extension_count += 1


class NonExtendableHold(SeatHold):
    """Prevents extension by breaking the base contract."""

    def extend(self, seconds: float) -> None:
        raise NotImplementedError("this hold cannot be extended")
```

Any caller iterating `list[SeatHold]` and extending each one now crashes on some elements. The polymorphism is broken and the caller must add `isinstance` checks — which means the abstraction has failed at its only job.

**Fix** — model the difference in the type, so no subtype has to refuse:

```python
class Hold(ABC):
    """The contract everything can honour."""

    @property
    @abstractmethod
    def expires_at(self) -> float: ...

    @abstractmethod
    def is_active(self, now: float) -> bool: ...


class ExtendableHold(Hold):
    def __init__(self, ttl_seconds: float, max_extensions: int = 2) -> None:
        self._expires_at = time.monotonic() + ttl_seconds
        self._max_extensions = max_extensions
        self._extensions = 0

    @property
    def expires_at(self) -> float:
        return self._expires_at

    def is_active(self, now: float) -> bool:
        return now < self._expires_at

    def extend(self, seconds: float) -> bool:
        """Return False when the extension budget is spent — no exception,
        because 'cannot extend further' is a normal outcome, not a violation."""
        if self._extensions >= self._max_extensions:
            return False
        self._expires_at += seconds
        self._extensions += 1
        return True


class FixedHold(Hold):
    """Honours the contract exactly; simply has no extend() in its type."""

    def __init__(self, ttl_seconds: float) -> None:
        self._expires_at = time.monotonic() + ttl_seconds

    @property
    def expires_at(self) -> float:
        return self._expires_at

    def is_active(self, now: float) -> bool:
        return now < self._expires_at
```

Callers of `Hold` now work for both. Callers who need extension take `ExtendableHold` explicitly — a *compile-time* distinction rather than a runtime surprise.

**Use from this repo.** The cache hierarchy: an `EvictionPolicy` interface with `LRU`, `LFU` and `TTL` implementations ([problems/02-lru-lfu-cache.md](problems/02-lru-lfu-cache.md)). Any of them must be usable wherever `EvictionPolicy` is expected. The Liskov trap: a `TTLPolicy` that silently never evicts anything because nothing has a TTL set. It satisfies the interface and breaks the *expectation*. The fix is a precondition: `EvictionPolicy.on_insert(key, ttl)`, plus a documented postcondition that after `on_insert`, `on_access` and `on_evict` are consistent. If a policy cannot evict (no keys have TTLs), it must say so at construction — not silently degrade.

**The three Liskov checks to run before shipping a subtype:**

```
1. Can the subtype honour every POSTCONDITION of the base?
   (A cache that never evicts fails "size never exceeds capacity".)
2. Does the subtype WEAKEN any PRECONDITION of the base?
   (Tightening is allowed; loosening is not.)
3. Does the subtype CHANGE the meaning of an inherited operation for
   callers who only know the base type?
   (Yes → the hierarchy is wrong, not the subtype.)
```

**The rule candidates state that interviewers like:** *"a subtype may strengthen postconditions and weaken preconditions, never the reverse."* That is the precise formulation of LSP, and it is checkable.

### 4.4 I — Interface Segregation Principle

> **No client should be forced to depend on methods it does not use. Many small role interfaces beat one fat one.**

**Violation** — the fat "God interface":

```python
class IPaymentProcessor(ABC):
    @abstractmethod
    def charge(self, amount: int, token: str) -> str: ...
    @abstractmethod
    def refund(self, charge_id: str, amount: int) -> str: ...
    @abstractmethod
    def tokenize_card(self, pan: str) -> str: ...
    @abstractmethod
    def settle_batch(self, ids: list[str]) -> None: ...
    @abstractmethod
    def webhook_handler(self, payload: dict) -> None: ...
    @abstractmethod
    def reconcile(self, date: str) -> dict: ...
    @abstractmethod
    def generate_report(self, month: str) -> bytes: ...
```

A UPI implementation now stubs `tokenize_card`, `settle_batch` and `generate_report`. Every implementation carries six irrelevant methods, and every interface change breaks all of them.

**Fix** — split by role, and compose:

```python
class ChargeProcessor(ABC):
    @abstractmethod
    def charge(self, amount_minor: int, token: str, idempotency_key: str) -> str: ...


class RefundProcessor(ABC):
    @abstractmethod
    def refund(self, charge_id: str, amount_minor: int, idempotency_key: str) -> str: ...


class CardTokenizer(ABC):
    @abstractmethod
    def tokenize(self, pan: str) -> str: ...


class BatchSettler(ABC):
    @abstractmethod
    def settle(self, charge_ids: list[str]) -> "SettlementBatch": ...


class PaymentGateway(ChargeProcessor, RefundProcessor, ABC):
    """A gateway must at least charge and refund. Everything else is optional
    and is declared by implementing the role interface separately."""


class CardGateway(PaymentGateway, CardTokenizer, BatchSettler):
    def charge(self, amount_minor: int, token: str, idempotency_key: str) -> str: ...
    def refund(self, charge_id: str, amount_minor: int, idempotency_key: str) -> str: ...
    def tokenize(self, pan: str) -> str: ...
    def settle(self, charge_ids: list[str]) -> "SettlementBatch": ...


class UpiGateway(PaymentGateway):
    """Only what UPI actually supports. No stubbed methods at all."""

    def charge(self, amount_minor: int, token: str, idempotency_key: str) -> str: ...
    def refund(self, charge_id: str, amount_minor: int, idempotency_key: str) -> str: ...
```

**Use from this repo.** The MCP and tool-use layer ([HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md)). A tool protocol is where ISP earns its keep: a read-only search tool should not be forced to implement `write`, `delete` and `subscribe`. Split into `ReadTool`, `WriteTool`, `StreamingTool`, and let each tool implement only the roles it genuinely has. The alternative — one `Tool` interface with `read()`, `write()`, `delete()` — means every tool declares capabilities it does not have, and the agent cannot tell a real capability from a stub.

**The honest caveat.** ISP's cost is more interfaces and more `isinstance`-free capability checks. The threshold: **split when an implementation would otherwise stub two or more methods.** One stub is a judgement call; two or more means the interface is carrying more than one role.

### 4.5 D — Dependency Inversion Principle

> **High-level policy should not depend on low-level detail. Both should depend on abstractions.**

**Violation** — the service that hard-codes its dependencies:

```python
class BookingService:
    def __init__(self) -> None:
        self._redis = Redis(host="prod-redis.internal", port=6379)   # ← concrete, configured, global
        self._db = PostgresConnection("postgres://prod/booking")     # ← concrete
        self._mailer = SmtpMailer("smtp.prod.internal")              # ← concrete

    def book(self, user_id: str, show_id: str, seats: list[str]) -> "Booking":
        self._redis.set(f"lock:{user_id}", show_id, ex=300)      # ← needs Redis
        self._db.execute("INSERT INTO bookings (user_id, show_id) VALUES (%s, %s)",
                         (user_id, show_id))                     # ← needs Postgres
        self._mailer.send(user_id, f"Booked {seats} for {show_id}")   # ← needs SMTP
        return Booking(user_id, show_id, seats)
```

Three consequences, all fatal in an interview:

- **Untestable.** Any test of `book()` needs a live Redis, a live Postgres and an SMTP server.
- **Unswappable.** Changing the mail provider edits the service.
- **Unusable in a second context.** You cannot run this against a staging Redis.

**Fix** — invert: the service declares what it needs; the composition root supplies it:

```python
class SeatHoldStore(ABC):
    """The abstraction the high-level policy depends on."""

    @abstractmethod
    def try_hold(self, show_id: str, seats: list[str], user_id: str,
                 ttl_seconds: int, idempotency_key: str) -> "HoldResult": ...

    @abstractmethod
    def confirm(self, hold_id: str, idempotency_key: str) -> "HoldResult": ...

    @abstractmethod
    def release(self, hold_id: str) -> None: ...


class BookingService:
    """Depends only on abstractions. It has never heard of Redis or Postgres."""

    def __init__(
        self,
        holds: SeatHoldStore,
        bookings: "BookingRepository",
        payments: "ChargeProcessor",
        clock: "Clock",
    ) -> None:
        self._holds = holds
        self._bookings = bookings
        self._payments = payments
        self._clock = clock

    def book(self, user_id: str, show_id: str, seats: list[str],
             idempotency_key: str) -> "Booking":
        result = self._holds.try_hold(show_id, seats, user_id, 600, idempotency_key)
        if not result.ok:
            raise SeatsUnavailableError(result.conflicting_seats)
        charge_id = self._payments.charge(result.amount_minor, user_id, idempotency_key)
        booking = Booking.create(
            show_id=show_id, seats=seats, user_id=user_id,
            hold_id=result.hold_id, charge_id=charge_id, now=self._clock.now(),
        )
        self._bookings.save(booking)
        self._holds.confirm(result.hold_id, idempotency_key)
        return booking


# --- The composition root: the ONLY place that knows concrete classes. ---
def build_booking_service(env: str) -> BookingService:
    if env == "test":
        return BookingService(
            holds=InMemorySeatHoldStore(),
            bookings=InMemoryBookingRepository(),
            payments=FakeChargeProcessor(),
            clock=FakeClock(),
        )
    return BookingService(
        holds=RedisSeatHoldStore(Redis.from_env()),
        bookings=PostgresBookingRepository(engine_from_env()),
        payments=StripeChargeProcessor.from_env(),
        clock=SystemClock(),
    )
```

**Use from this repo.** Every system in this knowledge base has a composition root. In the LLM serving layer ([HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md)), the gateway depends on a `ModelProvider` abstraction — so adding a second model vendor, or a mock provider for evaluation, is a new implementation rather than an edit to the gateway. In the RAG pipeline, the retriever is an abstraction so the vector store can be swapped without touching the generation path.

**DIP is also what makes the concurrency discussion testable.** A `Clock` abstraction is what lets you write a deterministic test for TTL expiry without `sleep()`. That single abstraction is the difference between a flaky test and a fast one — see [04-concurrency-and-threading-in-lld.md](04-concurrency-and-threading-in-lld.md).

---

## 5. Composition over inheritance

### 5.1 The comparison, concretely

The same requirement — a parking lot that supports different allocation policies, different pricing and different gate hardware — done two ways.

**Inheritance-first (the trap):**

```
                      ┌──────────────────┐
                      │    ParkingLot    │
                      └────────┬─────────┘
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
    │HourlyParkingLot │ │FlatParkingLot│ │SurgeParkingLot   │
    └────────┬────────┘ └──────┬───────┘ └────────┬─────────┘
             │                 │                  │
             ▼                 ▼                  ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌───────────────────┐
    │HourlyNearestLot  │ │FlatNearestLot    │ │SurgeNearestLot    │
    └──────────────────┘ └──────────────────┘ └───────────────────┘
       ...and an explosion of combinations with each new axis
```

Three pricing models × three allocators = nine classes, and the tenth feature is 27. Every leaf duplicates the lot's core behaviour. A bug fixed in one leaf is a bug in eight others.

**Composition-first (the fix):**

```
                      ┌──────────────────┐
                      │    ParkingLot    │
                      │──────────────────│
                      │ - floors         │
                      │ - allocator  ────┼────▶ «interface» SpotAllocator
                      │ - pricing    ────┼────▶ «interface» PricingStrategy
                      │ - gatePolicy ────┼────▶ «interface» GatePolicy
                      └──────────────────┘              ▲
                                                        │
                              ┌─────────────────────────┼──────────────────┐
                              │                         │                  │
                    NearestFirstAllocator      BalancedAllocator    CheapestFirst
                              │
                    HourlyPricing / FlatPricing / SurgePricing  (same pattern)

    Three independent axes ⇒ 3 + 3 + 3 = 9 small classes, and combinations
    are made by INJECTION, not by subclassing. Adding a fourth allocator
    adds ONE class.
```

### 5.2 The rule table

| Use **inheritance** when | Use **composition** when |
|---|---|
| The relationship is a genuine is-a in every context | The relationship is has-a, uses-a, or is-implemented-in-terms-of |
| The base class is abstract and thin (a contract + at most one small shared mechanism) | The base class would be concrete or would carry state the subtype does not want |
| The variant axis is naturally a taxonomy | The variant axis is a policy, a strategy, or a collaborator |
| Subtypes vary only in *what they compute*, not in *which parts they have* | Subtypes would differ in which collaborators they use |
| You want compile-time substitutability with no wiring | You want runtime substitutability, and testability via injection |
| There are ≤ 2 levels of hierarchy | Any depth would exceed 2 |

### 5.3 Mixins and traits — the middle ground

Some languages (Python, Ruby, Scala, TypeScript via declaration merging) allow behaviour to be shared without a single-inheritance hierarchy. This is worth naming because it shows you know the space:

| Mechanism | Shared what | Cost |
|---|---|---|
| **Abstract base class** | Contract + optional default methods | Single parent only; couples change axes |
| **Interface / Protocol** | Contract only | No code reuse; a second implementation must be written |
| **Mixin / trait** | Behaviour only, composed linearly | Method-resolution-order surprises; state sharing is subtle |
| **Delegation** | Behaviour, explicitly forwarded | Boilerplate, but fully visible — no magic |
| **Composition + injection** | Behaviour, at runtime, swappable | A wiring point, and one more indirection |

**In an interview:** "I'd use an abstract base for the contract, and composition for anything that can vary independently. I'd reach for a mixin only if I needed to share a behaviour across unrelated types, and I'd prefer delegation because the forwarding is explicit rather than resolved by MRO."

---

## 6. Interface versus abstract class

### 6.1 The decision table

| Dimension | Interface (pure contract) | Abstract class |
|---|---|---|
| What it can contain | Signatures only (plus, in Java/C#, default/static methods) | Signatures **and** implementation **and** state (fields) |
| Multiple inheritance | Yes — implement many | No — extend one (single inheritance in Java/C#) |
| Instance state | None | Yes |
| Constructor | No | Yes — named explicitly in the source's LLD definition |
| Versioning | Adding a method breaks all implementers (unless a default is provided) | Adding a concrete method is backward-compatible |
| The relationship it expresses | **Can-do** / **plays-the-role-of** | **Is-a**, with shared representation |
| Use when | You are defining a role with several unrelated implementers | You are defining a family with shared mechanism |

### 6.2 The rule

> **Interfaces for capability, abstract classes for shared mechanism.**

**In the parking lot:**

| Type | Which | Why |
|---|---|---|
| `PricingStrategy` | **Interface** | Unrelated implementers (hourly, flat, surge, loyalty-aware). No shared state. |
| `SpotAllocator` | **Interface** | Unrelated strategies. No shared state. |
| `Vehicle` | **Abstract class** | A shared mechanism (plate normalisation) plus a contract. A genuine family. |
| `PaymentMethod` | **Interface** | Card, UPI, wallet, netbanking — no shared state at all. |
| `GuardrailStage` | **Interface** | Input check, PII redactor, toxicity check — unrelated mechanisms. |
| `BaseRepository[T]` | **Abstract class** | Shared mechanism: connection handling, retry, transaction wrapper. |

### 6.3 The worked decision, in code

```python
from abc import ABC, abstractmethod


# INTERFACE — pure contract, no state, unrelated implementers
class PaymentMethod(ABC):
    @abstractmethod
    def authorize(self, amount_minor: int, idempotency_key: str) -> str: ...

    @abstractmethod
    def capture(self, auth_id: str, idempotency_key: str) -> str: ...

    @abstractmethod
    def void(self, auth_id: str) -> None: ...


# ABSTRACT CLASS — contract + shared mechanism + state
class BaseRepository(ABC):
    """Subclasses get connection handling and retry for free; they supply
    only the SQL-specific parts. This is the ONLY reason to use an abstract
    class here: there is real shared mechanism."""

    def __init__(self, pool: "ConnectionPool", max_retries: int = 3) -> None:
        self._pool = pool
        self._max_retries = max_retries

    def _execute(self, sql: str, params: tuple) -> list[tuple]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with self._pool.acquire() as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        return cur.fetchall()
            except TransientDbError as exc:      # retry only transient failures
                last_error = exc
                time.sleep(0.05 * (2 ** attempt))  # exponential backoff
        raise last_error if last_error else RuntimeError("unreachable")

    @abstractmethod
    def _table(self) -> str: ...


class BookingRepository(BaseRepository):
    def _table(self) -> str:
        return "bookings"

    def save(self, booking: "Booking") -> None:
        self._execute(
            "INSERT INTO bookings (id, show_id, user_id, amount_minor) "
            "VALUES (%s, %s, %s, %s)",
            (booking.id, booking.show_id, booking.user_id, booking.amount_minor),
        )
```

**Why not an interface here?** Because `_execute` is real shared mechanism: connection acquisition, retry with backoff, cursor management. An interface would force every repository to re-implement all of it.

**Why not an abstract class for `PaymentMethod`?** Because card, UPI and wallet share no state and no mechanism. An abstract class would be a class with three abstract methods and nothing else — an interface with extra ceremony and single-inheritance cost.

### 6.4 The Python-specific note

Python blurs the line, and being precise about how is a small seniority signal:

| Python mechanism | Equivalent to | When to use |
|---|---|---|
| `abc.ABC` + `@abstractmethod` | Interface or abstract class | When you want explicit inheritance and instantiation to fail on incomplete implementations |
| `typing.Protocol` (structural) | Interface, duck-typed | When you want a contract *without* requiring inheritance — "anything with this shape works" |
| Plain duck typing | Interface, implicit | Small internal code; risks silent mismatches |
| `@dataclass` | A value/record class | Data carriers with no behaviour beyond invariants |

**The interview-honest position:** "In Python I'd use an ABC when I want implementers to opt in and want the runtime to reject an incomplete class; I'd use a `Protocol` when I want a contract checkable by a type checker without forcing a class hierarchy — which is exactly what you want for a plugin or tool interface, because a third-party tool author should not have to import my base class."

---

## 7. Exceptions, edge cases & failure modes

This is the section that separates a recital from a design. Each of these is a real failure with a real fix.

### 7.1 The deep hierarchy

**Symptom:** five levels of inheritance; a method is defined on level 3 and overridden twice; the fragment method problem (each level implements part of one method).

**Why it happens:** each new distinction seems to deserve a class.

**Fix:** flatten to two levels — an abstract contract plus concrete leaves — and move shared mechanism into a collaborator. **Depth is the cost; breadth is cheap.** Adding a fourth subclass to a two-level hierarchy costs one file; adding a fourth level costs a mental model.

### 7.2 The fragile base class

**Symptom:** a base-class change breaks a subclass in a distant module with no test coverage.

**Why it happens:** the subclass depended on an implementation detail of the base (a private method's behaviour, the order of calls) rather than its contract.

**Fix:** make the base class's extension points explicit — a small set of `protected` hooks with documented contracts — and mark everything else `private`/`_`. In Python, use the single-underscore convention and, for anything you truly mean to be closed, `__`-name-mangled privates. The deeper fix is composition: a subclass that needs base internals is a sign the base is doing two jobs.

### 7.3 The broken substitution (LSP in production)

**Symptom:** `if isinstance(x, SpecialCase)` appears in a caller.

**Why it happens:** a subtype could not honour the contract, so the caller compensates.

**Fix:** the presence of an `isinstance`/`typeof` check against a hierarchy is the definitive signal that the hierarchy is wrong. Either the special case is a different *role* (split the interface, ISP) or the hierarchy should be composition (a flag or a collaborator, not a subclass).

### 7.4 The interface with one implementation

**Symptom:** `IFooService` and `FooServiceImpl`, where `IFooService` has one implementer, forever.

**Why it happens:** "we might need to swap it later", or a code standard from a different era.

**The honest answer:** one implementation is speculation. Two is real. **But** — and this is what makes the answer senior — an interface with one implementation is *justified* when it exists to make a dependency injectable (DIP), i.e. when the test double counts as the second implementation. `SeatHoldStore` with one Redis implementation is justified because `InMemorySeatHoldStore` exists in tests. `IFooService` with one implementation and no test double is ceremony.

**In the interview:** *"There's one production implementation and one test double, so the abstraction pays for itself. Without the test double I'd use the concrete class until a second implementation was real."*

### 7.5 Encapsulation leaks

Five ways encapsulation is defeated while looking intact:

| Leak | Example | Fix |
|---|---|---|
| **Mutable collection returned by reference** | `def spots(self): return self._spots` — the caller can `append` | Return a copy, a tuple, or a read-only view |
| **Mutable object returned by reference** | Returning an internal `Ticket` the caller can mutate | Return an immutable copy or a frozen dataclass |
| **Frozen dataclass with a mutable field** | `@dataclass(frozen=True)` holding a `list` | Use a `tuple`, or copy on the way in and out |
| **Setter on every field** | `set_state()` defeats the state machine entirely | Expose operations (`hold`, `occupy`), never state setters |
| **Public attribute convention** | Python's lack of enforcement means `spot._state = ...` works | Accept it, and use the *test* as the enforcement: assert the invariants hold after every operation |

**The Python position to state:** "Python does not enforce privacy, so encapsulation here is a *design* guarantee backed by tests rather than a compiler guarantee. I still keep state private and expose operations, because the design intent is what a reviewer reads."

### 7.6 The money-and-float failure

**Symptom:** a ₹0.01 discrepancy in a settlement, or a total that is one rupee off.

**Cause:** `float` arithmetic. `0.1 + 0.2 != 0.3` in binary floating point, and the error compounds over thousands of line items.

**Fix:** integers in the smallest currency unit (paise, cents) throughout, or a decimal type with explicit rounding. Never a float in a money path. All the code in this pillar uses `amount_minor: int` for exactly this reason.

**The same failure class in the AI repo:** token counts and cost-per-query are money. A float per-token cost multiplied by a billion tokens accumulates real error. Use integer micro-units.

### 7.7 Inheritance and the constructor problem

**Symptom:** a subclass whose constructor must call `super().__init__()` with arguments it only partly understands; a base constructor that calls an overridable method before the subclass is initialised.

**Fix:** keep base constructors trivial (assign fields, validate), never call overridable methods from a constructor, and prefer dependency injection over base-class setup. The classic failure — a base constructor calling `self.load_config()`, which the subclass overrides and which needs subclass fields that do not exist yet — is a bug that appears only in production.

### 7.8 `ABC` with `abstractmethod` still callable

**Symptom:** an abstract method has a body, and a subclass calls it via `super().method()`, silently running the "template" behaviour.

**Fix:** in Python, if the body is intended as a template, name it as such and document it; if it is intended as a contract, make the body `raise NotImplementedError` or leave it as a docstring-only `...`. Confusing the two produces the "abstract method that quietly does something" bug.

### 7.9 Over-abstraction — the counter-failure

**Symptom:** a 400-line change to add one field, spread over nine files and four interfaces.

**Cause:** every axis of imaginable change was abstracted.

**Fix:** delete abstractions with one implementation and no test double, inline interfaces whose only implementer is the caller, and merge classes that always change together. **The rule: an abstraction must be paid for by either a second real implementation or a test double.** If neither exists, it is cost without benefit.

### 7.10 SOLID dogma — the meta-failure

**Symptom:** the design is defensible principle-by-principle and nobody can find anything.

**Cause:** each principle was applied maximally without weighing its cost.

**The position to take in an interview:**

> "I apply SOLID where there is a real axis of change. In this design, pricing and allocation get interfaces because both will genuinely vary — and the interviewer is about to ask for a second policy. The gate hardware does not get a factory, because gate hardware does not change. I'd rather ship a three-case `if` that is easy to read than a factory for three cases that never change."

That answer demonstrates both design knowledge and restraint, and restraint is the scarcer signal.

---

## 8. When to use / When NOT to use

### 8.1 Encapsulation

| Use it | Do not over-use it |
|---|---|
| Whenever an object has an invariant (state machine, bounds, non-negativity) | For a pure data-transfer object with no invariants — a dataclass with public fields is honest |
| Whenever state is mutated from several places | When the "invariant" is a convention nobody can violate meaningfully |
| Whenever a TTL, a capacity or a version must be maintained | When the accessors would outnumber the behaviour 3:1 |

### 8.2 Inheritance

| Use it | Do not use it |
|---|---|
| A genuine is-a that passes the three-question test | To reuse one or two methods |
| An abstract contract with several leaves | To model a policy or a strategy |
| A family with real shared mechanism (an abstract base) | To add a field or a flag to an existing class |
| ≤ 2 levels of depth | When the subtype would stub or throw |

### 8.3 Interfaces

| Use them | Do not use them |
|---|---|
| Two or more real implementations exist, or a test double does | One implementation with no test double |
| The implementation is a boundary to an external system (DB, payment, LLM provider) | The type is a value object with no behaviour |
| A policy varies at runtime | The variation is compile-time-only and there is one variant |
| Role-based capability (ISP split) | A "God interface" that every implementer partially stubs |

### 8.4 Design patterns

Covered fully in [03-design-patterns.md](03-design-patterns.md), but the OOP-level rule: **introduce the pattern when the problem exists, name it after, and always state the cost.** A pattern named before the problem is decoration.

### 8.5 The decision table

| Situation | Answer |
|---|---|
| An object has a state machine | Encapsulate: private state, guarded mutators, defined invalid-transition errors |
| A policy varies (pricing, allocation, eviction, rate limiting) | Interface + injection (Strategy) |
| A family shares real mechanism | Abstract base class, ≤ 2 levels |
| Two unrelated types share a behaviour | Mixin or delegation; prefer delegation |
| A subtype cannot honour the parent contract | Composition, or split the interface |
| An implementation must be swappable per environment | Interface + a composition root |
| One implementation, no test double | Concrete class |

---

## 9. Numbers, defaults & tuning

| Parameter | Default | Notes |
|---|---|---|
| Inheritance depth | **≤ 2 levels** | Contract + leaves. Depth is the cost |
| Subclasses per base | **2–8** | Above 8, look for a missing axis (a policy, a flag, a collaborator) |
| Methods per interface | **1–4** | Above 4, consider an ISP split |
| Interface implementers before introducing it | **2** (a real one + a test double) | One with no double is speculation |
| Fields per class | **1–5** | More suggests two responsibilities |
| Public methods per class | **2–6** | Excluding properties and accessors |
| Cyclomatic complexity per method | **≤ 10** | Above that, extract a policy or a method |
| Method length | **≤ 30 lines** | A guideline, not a rule; a long method with one clear job is fine |
| Class length | **≤ 300 lines** | Above that, look for a second axis of change |
| `if/elif` cases before polymorphism | **3** | Three cases that have never changed are fine as a ladder |
| Collaborators injected per class | **1–5** | Above 5, the class may be an orchestrator that should be split |
| Money type | **integer minor units** | Never float. Ever |
| Boolean parameters in a signature | **0–1** | Two booleans in one signature is four behaviours in one method |
| Composition ratio | **≥ 80% composition, ≤ 20% inheritance** | A rough health metric for a codebase |
| LSP checks before shipping a subtype | **3** | Postconditions honoured, preconditions not weakened, meaning unchanged |
| Lines changed to add one feature | **≤ 3 files** | Above that, the extension point is missing |

---

## 10. Interview Q&A

**Q1 (easy). Name the four pillars of OOP.**
Encapsulation, abstraction, inheritance and polymorphism. Encapsulation means the object enforces its own invariants; abstraction means callers depend on what is offered, not how; inheritance shares a contract or representation up a hierarchy; polymorphism lets one call site work for many implementations.

**Q2 (easy). What is encapsulation, beyond "private fields"?**
It is the property that no caller can put the object into an invalid state. Private fields with public setters on every field is encapsulation theatre. The test: can a caller, from outside, produce a state the object's own methods would reject? If yes, it is not encapsulated. In the parking lot, `hold()` on an occupied spot must raise, and that must be impossible to bypass.

**Q3 (easy). What is the difference between an interface and an abstract class?**
An interface is a pure contract with no state and multiple implementers allowed; an abstract class can carry state, implementation and a constructor, and you can extend only one. The rule I use: interfaces for capability, abstract classes for shared mechanism. In the parking lot, `PricingStrategy` is an interface because the strategies share nothing; `BaseRepository` is an abstract class because connection handling and retry are real shared mechanism.

**Q4 (medium). Explain SOLID in one line each.**
SRP: one reason to change per class — one actor it answers to. OCP: you can add behaviour by adding code rather than editing existing code. LSP: a subtype must be usable wherever the supertype is expected, without the caller knowing. ISP: no client should be forced to depend on methods it does not use. DIP: high-level policy depends on abstractions, not on concrete low-level detail.

**Q5 (medium). Show me an SRP violation and how you would fix it.**
A `BookingService` that validates availability, computes the price, writes to the database, sends an email and writes an audit record. Five reasons to change — seating rules, pricing, schema, provider, compliance — and five different people who can demand a change. The fix is to split into `SeatAvailabilityChecker`, `PriceCalculator`, `BookingRepository`, `ConfirmationNotifier` and `AuditLog`, and let `BookingService` become an orchestrator that composes them. The point is that the class then changes when the *orchestration* changes, which is rarely.

**Q6 (medium). Give me an LSP violation and the fix.**
A base `Hold` class promising an `extend(seconds)` operation, with a subclass that raises `NotImplementedError` because it cannot be extended. Any caller iterating holds and extending each one now needs an `isinstance` check — so the polymorphism has failed. The fix is to model the difference in the type: a `Hold` contract with `expires_at` and `is_active`, and a separate `ExtendableHold` that adds `extend` — and for the extension limit, return `False` rather than raising, because "budget exhausted" is a normal outcome and not a contract violation. A subtype may strengthen postconditions and weaken preconditions, never the reverse.

**Q7 (medium). Why prefer composition over inheritance?**
Because inheritance is a permanent, compile-time, single-parent coupling that ties two classes to the same axis of change forever, and it exposes the base class's implementation to the subclass. Composition is a runtime relationship you can swap, mock and reconfigure. Three specific costs of inheritance: you get everything the base has whether you want it or not; the subtype can break the base's contract and the compiler will not notice; and a base-class change ripples into every subclass. I use inheritance only where there is a genuine is-a that passes three checks: substitutable in every context, able to honour every postcondition, and varying for the same reason as the parent.

**Q8 (medium). When would you introduce an interface for a single implementation?**
When the abstraction is what makes the class testable — that is, when a test double counts as the second implementation. `SeatHoldStore` with one Redis implementation is justified because `InMemorySeatHoldStore` exists for tests, and because a production incident might mean swapping to a database-backed store. Without a test double and without a plausible second implementation, I would use the concrete class. "We might swap it later" is speculation, and speculation has a cost.

**Q9 (hard). *Pushback:* "SOLID sounds like dogma. Isn't following it blindly bad design?"**
Yes, and that is the correct reading. Each principle has a real cost — indirection, more files, more wiring — and applying all five maximally produces a codebase of one-method classes that is harder to change than the naive version. I treat each violation as a cost I may choose to pay knowingly. A three-case `if/elif` over vehicle types that has not changed in four years is correct; refactoring it into a strategy pattern would be over-engineering. The failure mode I actually watch for is the opposite: an axis of change that *is* real and has no abstraction, so every new case edits a central function and re-tests all of it. So my rule is: abstract the axis that actually changes, not the axis I can imagine changing.

**Q10 (hard). *Pushback:* "You used an abstract base class for `Vehicle`. Isn't that just inheritance, which you said to avoid?"**
It is inheritance, and it passes my three tests. First, it is a genuine is-a: every `Car` is a `Vehicle` in every context, and no caller ever needs to check the concrete type. Second, the base is abstract and thin — it supplies one small shared mechanism (plate normalisation) and two contracts, so there is no representation for a subclass to accidentally depend on. Third, the axis of change is the same: all vehicles change when the vehicle taxonomy changes. And it is two levels deep with no further subclassing planned. What I would *not* do is make `Floor` extend `ParkingLot` to reuse `count_free()` — that fails the is-a test and it acquires an identity it should not have.

**Q11 (hard). *Pushback:* "Isn't an interface for `PricingStrategy` premature when there is only one pricing model?"**
There is currently one production model, but I introduced the interface for two reasons I can defend. First, a second model is imminent — the standard follow-up in this problem is surge or weekend pricing, and real lots always have more than one. Second, and more importantly, the interface is what lets me write a deterministic test: a `FixedPricing(100)` strategy makes fee assertions exact instead of time-dependent, so the abstraction pays for itself in the test suite even before the second production model exists. If neither of those were true, I would use the concrete `HourlyPricing` and introduce the interface when the second model actually landed.

**Q12 (hard). What is the relationship between SOLID and design patterns?**
Patterns are the *mechanisms* that deliver the principles. Strategy and dependency injection deliver OCP and DIP. Adapter and Facade deliver ISP and DIP at a boundary. Observer delivers OCP for the notification axis. Template Method delivers OCP inside a family with shared mechanism. Chain of Responsibility delivers OCP for a sequence of stages. So I do not design "with patterns" — I notice which axis of change is real, choose the principle that governs it, and the pattern falls out. If I name a pattern without naming the problem and the cost, I am decorating.

**Q13 (medium). How does OOP change for an AI/LLM system?**
The vocabulary transfers. An agent loop is a state machine with states like `THINK`, `ACT`, `OBSERVE`, `DONE`, with guards on every transition and a hard iteration cap — encapsulation of a lifecycle. A tool registry is a Factory plus an Adapter: one `Tool.run(args)` interface, one adapter per external API, which is ISP in practice because a read-only search tool should not implement `write`. A guardrail pipeline is a Chain of Responsibility or a Decorator chain, and which one depends on whether a stage can short-circuit. A prompt builder is the Builder pattern, because a prompt has many optional parts. What is genuinely new is that correctness is probabilistic: the edge-case section gains an evaluation dimension, because you cannot enumerate wrong answers — you measure them.

**Q14 (medium). What is a fragile base class and how do you avoid it?**
A base class whose implementation changes break subclasses in distant modules. It happens when a subclass depends on an implementation detail rather than the contract. I avoid it by keeping the base thin, making extension points explicit and documented, marking everything else private, and preferring composition whenever a subclass needs the base's internals — because that need is itself the signal that the base is doing two jobs.

**Q15 (hard). You see `isinstance(x, SpecialSeat)` in a caller. What does that tell you?**
It tells me the hierarchy is wrong. Either the special case is a different role and should be a separate interface — that is ISP — or the difference should be a collaborator or a flag rather than a subclass, which is composition. The presence of a type check against a hierarchy is the most reliable signal that polymorphism has failed at its only job, which is letting the call site stay unchanged.

**Q16 (medium). What is the difference between cohesion and coupling, and how do they relate to SOLID?**
Cohesion is how strongly the members of one class belong together; coupling is how many other classes a class depends on. Good design is high cohesion, low coupling. SRP and ISP raise cohesion by making each unit's job narrow; DIP and OCP lower coupling by replacing concrete dependencies with abstractions. The real test in a design review is cohesion, not class size — `Ticket` with four fields and no behaviour beyond its invariants is a perfectly cohesive record even though it is small, while a `Manager` class with forty unrelated methods is incoherent however well it is organised.

**Q17 (medium). Where does the four-pillar vocabulary actually get used in an interview?**
As justification for boundaries. When I place `reserveSpot` on `Floor` rather than on `ParkingLot`, I say: the operation mutates the floor's own state and needs its free-list, so encapsulation puts it there; `ParkingLot` delegates because it is the system-facing entry point. When I introduce `PricingStrategy`, I say: pricing varies independently of everything else, so I abstract that axis and inject it — that is OCP and DIP together. The pillars are not a recital; they are the reason each line of the class diagram is where it is.

---

## 11. Common mistakes candidates make

1. **Reciting the pillars and principles without applying them.** The vocabulary is worth nothing unless it justifies a boundary.
2. **Inheritance for code reuse.** The single most common OOP error, and the most expensive.
3. **A hierarchy deeper than two levels.** Depth is the cost; breadth is cheap.
4. **A subtype that throws on a parent method.** That is an LSP violation, and the caller will end up with an `isinstance` check.
5. **Interfaces with one implementation and no test double.** Speculation with a maintenance cost.
6. **A "God interface" with 15 methods.** Every implementer stubs most of them.
7. **Getters and setters on every field**, calling it encapsulation.
8. **Returning internal mutable collections by reference.** The invariant is bypassed in one line.
9. **`float` in a money path.** Use integer minor units.
10. **Abstracting the axis of imagination instead of the axis of change.** The classic over-engineering failure, and interviewers mark it.
11. **Not saying what each abstraction costs.** An interface, a class, and a wiring point are real costs; name them.
12. **Composition mentioned but not used.** If a class has a collaborator it constructs itself with `new`, DIP has not happened.
13. **Confusing the SRP "one reason to change" test with "small"**. The test is *who can demand a change*, not line count.
14. **Applying a pattern before the problem exists.** The pattern is a consequence, never the requirement.
15. **Never mentioning testability.** DIP's most concrete payoff is a deterministic test — the `Clock` abstraction that removes `sleep()` from the test suite.
16. **Treating SOLID as a scorecard.** Five principles applied maximally produce worse code than three applied where they matter.

---

## 12. Cheat-sheet summary

**The four pillars**

| Pillar | One-line truth | The test |
|---|---|---|
| Encapsulation | The object enforces its own invariants | Can a caller create an illegal state? |
| Abstraction | Callers depend on *what*, never *how* | Can you swap the implementation without editing callers? |
| Inheritance | Shares a contract (interface) or representation (base class) | Is it a genuine is-a in every context? |
| Polymorphism | One call site, many implementations | Can you add an implementation without editing the call site? |

**SOLID**

| Letter | Principle | The symptom of violation | The fix |
|---|---|---|---|
| **S** | One reason to change | Five teams' PRs touch one class | Split by axis of change; orchestrate |
| **O** | Open for extension, closed for modification | A growing `if/elif` ladder | Polymorphism at the varying axis |
| **L** | Subtypes must be substitutable | A subtype throwing `NotImplementedError` | Model the difference in the type; split the contract |
| **I** | No client depends on methods it does not use | Implementations stubbing half an interface | Role interfaces; compose them |
| **D** | Depend on abstractions | `new RedisClient()` inside a service | Inject; keep a composition root |

**Interfaces vs abstract classes**

| | Interface | Abstract class |
|---|---|---|
| Contains | Contract only | Contract + implementation + state |
| Multiple | Yes | No (one parent) |
| Relationship | Can-do / plays-the-role-of | Is-a, with shared mechanism |
| Use when | Several unrelated implementers | A family with real shared mechanism |

**The three-question inheritance test** — a genuine is-a in every context · honours every base postcondition · varies for the same reason as the parent. **Two "no"s means composition.**

**The rule of thumb** — ≤ 2 levels of inheritance · 1–4 methods per interface · 2 implementers (one real, one test double) before introducing an interface · abstract the axis that changes, not the axis you can imagine changing · integer minor units for money, never float.

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "Why is that a separate class?" | "It changes for a different reason — SRP. Pricing changes when the business changes; seating changes when the venue changes." |
| "How would you add a new pricing model?" | "A new `PricingStrategy`; `ParkingLot` does not change. That is OCP through polymorphism and injection." |
| "Why an interface and not a base class?" | "No shared state or mechanism — unrelated implementers. Interfaces for capability, abstract classes for shared mechanism." |
| "Isn't that over-engineered?" | "If the second implementation weren't imminent I'd use the concrete class. Here the axis of change is real." |
| "Why not inherit?" | "It fails the is-a test and it would couple two different axes of change. Composition — a collaborator I inject." |
| "Is it testable?" | "Yes — the collaborators are injected, so the test substitutes fakes and the clock abstraction removes `sleep()`." |

---

## 13. Cross-links

- Previous in this pillar: [What is LLD and how to answer it](01-what-is-lld-and-how-to-answer.md)
- Next: [Design patterns](03-design-patterns.md) · [Concurrency and threading in LLD](04-concurrency-and-threading-in-lld.md) · [UML and class diagrams](05-uml-and-class-diagrams.md) · [The machine-coding interview framework](06-machine-coding-interview-framework.md)
- Where these principles become diagrams: [UML and class diagrams](05-uml-and-class-diagrams.md)
- Where these principles become running code: [Parking lot](problems/01-parking-lot.md) · [LRU and LFU cache](problems/02-lru-lfu-cache.md) · [Splitwise](problems/06-splitwise.md)
- Repo systems used as examples here: [Notification system](../Case-Studies/HLD/03-notification-system.md) (SRP across render/select/dispatch/track) · [BookMyShow seat booking](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) (LSP on holds, DIP on the hold store) · [Guardrails and safety](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) (OCP across guardrail stages) · [LLM serving and gateway](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) (DIP on the model provider) · [MCP, tool use and multi-agent](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) (ISP across tool roles)
- The HLD/LLD boundary these principles live below: [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md)
- Drill here: [Foundations and capacity questions](../Interview-Questions/01-foundations-and-capacity-questions.md)
- Revise here: [Foundations and estimation cheat sheet](../Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md)
