# LLD Problem: Elevator System

> **Grounding note.** The source transcript does not mention this problem at all — not in the LLD question list at **[9:22]**, not in the parking-lot walkthrough at **[25:54]–[27:38]**, nowhere. **The entire file is standard industry practice**, and the elevator problem is in every LLD list for one reason worth stating up front: it is the rare problem where the *scheduling algorithm* is the design, and where a candidate who writes correct-looking code can still produce a system that is obviously wrong to any human who has waited for a lift. It is also the best problem in this pillar for practising the **extension** step, because "now add a second lift" turns a single-object problem into a genuine multi-agent one with shared state, and "now add a destination panel" inverts the entire input model. For the shared-state mechanics this file assumes — locks, ordering, the difference between a lease and a sale — read [concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md).

---

## 1. Requirement clarification

### 1.1 Questions to ask

| # | Question | Why it matters | Default if they will not say |
|---|---|---|---|
| 1 | How many lifts, and how many floors? | Decides whether this is a single-object simulation or a multi-agent one | One lift, 10 floors; the multi-lift case is the key extension (§7.2) |
| 2 | **Up and down buttons, or a destination panel?** | **Inverts the entire input model** — see §1.3 | Up/down buttons; the panel is the extension in §7.4 |
| 3 | What is the scheduling goal? | Waiting time, ride time, or energy — and they conflict | Minimise the worst-case wait, then the average |
| 4 | Is a request served on the way, or is it a queue? | Determines whether direction is sticky | Served on the way: a SCAN-style sweep |
| 5 | Do we model the physical state — doors, weight, motor? | Realism versus scope; the follow-ups usually want it | Yes: doors and weight, because they gate the state machine |
| 6 | Do we model time and speed? | Determines whether the demo is a simulation or a state walk | A `Clock`; one floor per tick, doors cost two ticks |
| 7 | Does the lift ever become unavailable? | Maintenance is a real state and it changes the scheduler | The extension in §7.5 |
| 8 | Do we need emergency/fire mode? | It overrides everything, including direction | Named as an override (§7.6), not modelled in full |
| 9 | What is the concurrency model? | Buttons are pressed from many places at once | Thread-safe; presses from any floor and inside the car |
| 10 | Do we need the display and the door-open button? | Both are real inputs that affect state | Yes, both — they are cheap and they reveal a state bug |

### 1.2 Defaults to state out loud

```
  BUILDING:   10 floors (0..9), 1 lift, up/down call buttons per floor plus
              one button per floor inside the car.
  SCHEDULING: SCAN ("elevator algorithm") — the car keeps moving in its
              current direction until there is nothing left ahead, then
              reverses. This is what real lifts do, and the reason is worth
              stating: it bounds the worst-case wait, and it is what stops a
              lift oscillating between two floors while a third starves.
  STOPS:      a car only stops at a floor it can serve IN ITS CURRENT
              DIRECTION. An up-moving car does not answer a down call.
  TIME:       one tick per floor travelled, two ticks for a door cycle.
  THREADING:  the controller is shared; presses arrive concurrently.
  OUT:        the motor, the cables, the shaft, physics, the building's
              power, the fire alarm wiring.
```

### 1.3 The question that changes everything

```
   UP/DOWN BUTTONS                     DESTINATION PANEL
   ─────────────────                   ─────────────────
   One call per floor per direction.    The passenger selects the FLOOR
   The lift does not know where the     before entering. So the system knows
   passenger is going until they        the destination at request time,
   press a button INSIDE the car.       before the doors open.

   Assignment: a call is served by      Assignment: the panel can group
   whichever car arrives first.         passengers going the same way and
                                        send them to the SAME car.

   Scheduling is reactive: stop when    Scheduling is planned: the full trip
   a call is ahead of you.              is known, so cars can be routed.

   Consequence: a naive implementation  Consequence: the model needs a
   has the "up car stops for a down     destination on the request, and the
   call" bug, because the call carries  scheduler becomes a batch problem.
   no destination.
```

**Say which model you are building.** They produce different classes — with a panel, `Request` carries a `destination`, and the "serve in the current direction" rule becomes "serve if the destination is ahead". Getting this wrong after the interviewer assumed the panel is the single most common way this problem goes sideways.

---

## 2. Core entities & responsibilities

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `Direction` | Which way, or idle | an enum | — |
| `DoorState` | Open, closed, or moving | an enum | — |
| `Call` | An external request: *a floor, a direction* | floor, direction, timestamp | Know which car answers |
| `CarRequest` | An internal request: *a floor, from inside* | floor, timestamp | Know direction — it is the car's |
| `ElevatorCar` | One lift: its position, doors, load, and its stop set | floor, direction, doors, weight, stops | Choose which call to answer |
| `Scheduler` *(interface)* | Choose the next stop for a car | — | Move the car; drive the doors |
| `ScanScheduler` | SCAN: keep direction, serve ahead, reverse when empty | — | — |
| `NearestCarScheduler` | Assign a call to the best car (multi-car) | — | — |
| `ElevatorController` | Accept calls, tick the world, dispatch | the cars, the scheduler, the pending calls | Implement scheduling |
| `Clock` *(interface)* | Supply ticks | — | — |

**Two decisions, stated because they are the design:**

**First, a `Call` does not know which car will answer it.** It is a *floor and a direction*, and assignment is the scheduler's job. Letting a call hold a car reference bakes the assignment into the request, and then the multi-car extension requires changing the request class.

**Second, the car owns its own stop set.** The scheduler's job is to compute a next stop; the car's job is to move toward it and to know which floors it will stop at. **The car never asks the controller what to do** — that inversion is what makes the single-lift case a local problem and the multi-lift case a coordination problem rather than a rewrite.

---

## 3. Class diagram

```
   ┌──────────────────┐      ┌──────────────────┐      ┌────────────────┐
   │   Direction      │      │   DoorState      │      │    Call        │
   ├──────────────────┤      ├──────────────────┤      ├────────────────┤
   │ UP               │      │ CLOSED           │      │ floor: int     │
   │ DOWN             │      │ OPENING          │      │ direction      │  ← external
   │ IDLE             │      │ OPEN             │      │ created_at     │
   └──────────────────┘      │ CLOSING          │      └───────┬────────┘
                             └──────────────────┘              │ *
                                                               │
                                                               │ pending
   ┌───────────────────────────────────────────────────────────▼──────────┐
   │                        ElevatorController                            │
   ├──────────────────────────────────────────────────────────────────────┤
   │ - cars: list[ElevatorCar]                                            │
   │ - scheduler: Scheduler                                               │
   │ - pending: list[Call]                    ← unassigned external calls │
   │ - lock: RLock                                                        │
   ├──────────────────────────────────────────────────────────────────────┤
   │ + press_call(floor, direction) -> None                               │
   │ + tick() -> list[Event]                  ← advances the whole world  │
   │ + status() -> list[CarStatus]                                        │
   └───────────────┬──────────────────────────────────┬───────────────────┘
                   │ 1..*                             │ uses
                   ▼                                  ▼
   ┌───────────────────────────────┐      ┌───────────────────────────────┐
   │        ElevatorCar            │      │     «interface» Scheduler     │
   ├───────────────────────────────┤      ├───────────────────────────────┤
   │ - car_id: str                 │      │ + next_stop(car) -> int|None  │
   │ - floor: int                  │      │ + assign(call, cars) -> car   │
   │ - direction: Direction        │      └───────────▲───────────────────┘
   │ - doors: DoorState            │                  │
   │ - stops: set[int]             │      ┌───────────┴────────────┐
   │ - weight_kg: float            │      │                        │
   │ - capacity_kg: float          │  ┌───┴────────────┐  ┌────────┴──────────┐
   │ - door_ticks_left: int        │  │  ScanScheduler │  │NearestCarSched.   │
   ├───────────────────────────────┤  ├────────────────┤  ├───────────────────┤
   │ + add_stop(floor) -> bool     │  │ keeps dir;     │  │ assigns a call to │
   │ + step() -> list[Event]       │  │ serves ahead;  │  │ the car with the  │
   │ + can_serve(call) -> bool     │  │ reverses when  │  │ lowest cost       │
   │ + is_full() -> bool           │  │ nothing ahead  │  └───────────────────┘
   └───────────────────────────────┘  └────────────────┘
```

**What the diagram states:**

- `Call` is separate from `CarRequest` and carries a direction; the car's stops are a plain `set[int]`, because a car stop has no direction of its own — the car's direction supplies it.
- `Scheduler` is an interface with `next_stop` **and** `assign`. Single-lift code uses only the first; the multi-car extension is the second, already in the contract.
- `ElevatorController` owns the lock and the pending calls; the cars own their own movement.
- The controller's `tick()` returns `Event`s — the display and the door button are consumers of state changes rather than pollers.

---

## 4. Interfaces and enums

```python
from abc import ABC, abstractmethod
from enum import Enum


class Direction(Enum):
    """IDLE is not the absence of a direction — it is a state with its own
    behaviour. A car that is IDLE at floor 3 with no stops is waiting to be
    assigned; a car that is UP at floor 3 is committed and cannot be turned
    around cheaply. Collapsing IDLE into UP loses that distinction, and the
    multi-car scheduler needs it."""

    UP = 1
    DOWN = -1
    IDLE = 0

    def opposite(self) -> "Direction":
        if self is Direction.UP:
            return Direction.DOWN
        if self is Direction.DOWN:
            return Direction.UP
        return Direction.IDLE


class DoorState(Enum):
    """Four states, not two. OPENING and CLOSING are real, and they matter
    because a door mid-cycle must not be interrupted — the classic bug is a
    door reversing while a passenger is in the doorway."""

    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"

    def is_moving(self) -> bool:
        return self in (DoorState.OPENING, DoorState.CLOSING)


class EventKind(Enum):
    ARRIVED = "arrived"
    DOORS_OPENED = "doors_opened"
    DOORS_CLOSED = "doors_closed"
    STOP_ADDED = "stop_added"
    CALL_ASSIGNED = "call_assigned"
    OVERLOADED = "overloaded"
    REVERSED = "reversed"


class Event:
    """An immutable fact about something that happened. `tick()` returns these
    so a display, a logger and a test can all consume the same stream, rather
    than each polling the car and inferring what changed."""

    __slots__ = ("kind", "car_id", "floor", "detail")

    def __init__(self, kind: EventKind, car_id: str, floor: int, detail: str = "") -> None:
        self.kind = kind
        self.car_id = car_id
        self.floor = floor
        self.detail = detail

    def __repr__(self) -> str:
        suffix = f" {self.detail}" if self.detail else ""
        return f"[{self.car_id}@{self.floor}] {self.kind.value}{suffix}"


class Call:
    """An EXTERNAL request: someone on a floor pressed up or down.

    It deliberately does NOT hold a car reference. Assignment is the
    scheduler's decision, and a call that names its car cannot be reassigned
    when a better car becomes free — which is the whole point of having a
    scheduler at all."""

    __slots__ = ("floor", "direction", "created_at")

    def __init__(self, floor: int, direction: Direction, created_at: float) -> None:
        if direction is Direction.IDLE:
            raise ValueError("a call must have a direction; IDLE is not a call")
        self.floor = floor
        self.direction = direction
        self.created_at = created_at

    def wait_time(self, now: float) -> float:
        return now - self.created_at

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Call):
            return NotImplemented
        return (self.floor, self.direction) == (other.floor, other.direction)

    def __hash__(self) -> int:
        return hash((self.floor, self.direction))

    def __repr__(self) -> str:
        return f"Call(floor={self.floor}, {self.direction.name})"


class Scheduler(ABC):
    """The axis that varies. Two responsibilities, and both are in the
    interface from the start because the multi-car extension needs `assign`
    and adding it later would change every call site."""

    @abstractmethod
    def next_stop(self, car: "ElevatorCar", pending: "list[Call]") -> "int | None":
        """The floor this car should move toward next, or None to stay put.
        Called once per tick, per car, under the controller's lock."""

    @abstractmethod
    def assign(self, call: "Call", cars: "list[ElevatorCar]") -> "ElevatorCar":
        """Which car should answer this call. With one car this is trivial;
        with many it is the whole problem."""

    @abstractmethod
    def should_reverse(self, car: "ElevatorCar", pending: "list[Call]") -> bool:
        """Whether the car has nothing left ahead and should turn around."""


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Monotonic seconds. Used for wait-time metrics, which is why a
        wall-clock is wrong: an NTP step would report a negative wait."""
```

**Why `next_stop` takes `pending` as an argument rather than reading it:** the scheduler is stateless. All the state lives on the cars and in the controller. **A stateless scheduler is testable in isolation and safe to share**, and if a future policy needs state — a learned model, a rate limiter — it can hold it in its own instance without the controller knowing.

---

## 5. Complete working code

**Language note:** Python 3.11+, standard library only. Every block is complete.

### 5.1 The car — the state machine, and the door rule

```
   DOOR CYCLE, and why the intermediate states exist:

     CLOSED ──open──▶ OPENING ──(2 ticks)──▶ OPEN ──close──▶ CLOSING ──(2 ticks)──▶ CLOSED
                          ▲                                                          │
                          └──────────────── can be reversed ─────────────────────────┘
                                        ONLY from CLOSING

   RULE: a door mid-cycle cannot be interrupted. The one exception is a
   reversal out of CLOSING — reopening because someone pressed the button or
   stepped into the doorway. That is a real safety behaviour, and modelling
   it is what stops a passenger being caught.

   RULE: the car does not move with the doors not CLOSED. This is the single
   most important invariant in the whole file — a lift that moves with its
   doors open is not a bug, it is a catastrophe.
```

```python
class ElevatorCar:
    """ONE lift. Owns its position, its direction, its doors and its stops.

    The car does NOT decide where to go. It is told a next stop by the
    scheduler and it moves toward it. That inversion is what makes the
    multi-car case a coordination problem rather than a rewrite.
    """

    DOOR_CYCLE_TICKS = 2          # ticks to go from CLOSED to OPEN, or back
    DOOR_OPEN_TICKS = 4           # how long the doors stay open
    TICKS_PER_FLOOR = 1

    def __init__(self, car_id: str, top_floor: int,
                 capacity_kg: float = 800.0) -> None:
        if top_floor < 1:
            raise ValueError("a building needs at least two floors")
        if capacity_kg <= 0:
            raise ValueError("capacity must be positive")
        self.car_id = car_id
        self.top_floor = top_floor
        self.capacity_kg = capacity_kg

        self.floor = 0
        self.direction = Direction.IDLE
        self.doors = DoorState.CLOSED
        self.stops: set[int] = set()
        self.weight_kg = 0.0

        self._door_ticks_left = 0
        self._door_target = DoorState.CLOSED
        self._open_hold_ticks = 0

    # ---- the safety invariants, as a checkable method ------------------
    def _assert_invariants(self) -> None:
        if not 0 <= self.floor <= self.top_floor:
            raise RuntimeError(f"{self.car_id} is at floor {self.floor}, outside 0..{self.top_floor}")
        if self.doors is not DoorState.CLOSED and self.direction is not Direction.IDLE:
            # A car may hold a direction with the doors open (it is stopping
            # here on its way somewhere), but it must not MOVE. The check
            # that it does not move is in step(), which never advances the
            # floor unless the doors are CLOSED.
            pass
        for stop in self.stops:
            if not 0 <= stop <= self.top_floor:
                raise RuntimeError(f"{self.car_id} has a stop at invalid floor {stop}")

    # ---- stops --------------------------------------------------------
    def add_stop(self, floor: int) -> bool:
        if not 0 <= floor <= self.top_floor:
            raise ValueError(f"floor {floor} is outside 0..{self.top_floor}")
        if floor in self.stops:
            return False
        self.stops.add(floor)
        return True

    def remove_stop(self, floor: int) -> bool:
        if floor in self.stops:
            self.stops.remove(floor)
            return True
        return False

    def has_stops_ahead(self) -> bool:
        """Is there anything in the direction of travel, or — when idle — at
        all? This is the question the reversal rule is built on."""
        if self.direction is Direction.UP:
            return any(stop > self.floor for stop in self.stops)
        if self.direction is Direction.DOWN:
            return any(stop < self.floor for stop in self.stops)
        return bool(self.stops)

    def floors_ahead(self) -> "list[int]":
        """The stops in the direction of travel, NEAREST FIRST. Nearest-first
        is not cosmetic: it is what makes the car stop in order rather than
        overshooting to the furthest floor and coming back."""
        if self.direction is Direction.UP:
            return sorted(s for s in self.stops if s >= self.floor)
        if self.direction is Direction.DOWN:
            return sorted((s for s in self.stops if s <= self.floor), reverse=True)
        return []

    def can_serve(self, call: Call) -> bool:
        """Whether this car can answer an external call WITHOUT reversing.

        THE RULE THAT MATTERS: an up-moving car does not answer a down call,
        and a car does not answer a call behind it. The exception is a car
        with nothing ahead — an idle car, or one that has finished its sweep —
        which may reverse and take it. Getting this wrong produces the bug
        every user has experienced: a lift that stops on your floor going the
        wrong way, opens, and leaves."""
        if self.is_full():
            return False
        if self.direction is Direction.IDLE:
            return True
        if self.direction is call.direction:
            if call.direction is Direction.UP:
                return call.floor >= self.floor
            return call.floor <= self.floor
        # Opposite direction: only if there is nothing left ahead of us.
        return not self.has_stops_ahead()

    # ---- load ---------------------------------------------------------
    def is_full(self) -> bool:
        return self.weight_kg >= self.capacity_kg

    def board(self, weight_kg: float) -> bool:
        """Returns False if the car is over capacity, and does NOT board. The
        alternative — boarding and then refusing to move — is what produces a
        lift that sits there with its doors open and a full load."""
        if weight_kg <= 0:
            raise ValueError("a passenger has positive weight")
        if self.weight_kg + weight_kg > self.capacity_kg:
            return False
        self.weight_kg += weight_kg
        return True

    def alight(self, weight_kg: float) -> None:
        self.weight_kg = max(0.0, self.weight_kg - weight_kg)

    # ---- doors --------------------------------------------------------
    def open_doors(self) -> bool:
        """Start a door cycle. Refused unless the car is stationary — a door
        opening while the car moves is the failure the whole state machine
        exists to prevent."""
        if self.doors is DoorState.OPEN or self.doors is DoorState.OPENING:
            self._open_hold_ticks = self.DOOR_OPEN_TICKS      # re-hold, not restart
            return True
        if self.doors is DoorState.CLOSING:
            # The reversal: a passenger in the doorway. Legal, and the reason
            # CLOSING is a state rather than an instantaneous transition.
            self._door_target = DoorState.OPEN
            self._door_ticks_left = self.DOOR_CYCLE_TICKS
            self.doors = DoorState.OPENING
            return True
        self._door_target = DoorState.OPEN
        self._door_ticks_left = self.DOOR_CYCLE_TICKS
        self.doors = DoorState.OPENING
        return True

    def close_doors(self) -> None:
        """Never closes while the car is overloaded. A car that closes and
        then cannot move is worse than one that stays open."""
        if self.is_full():
            return
        if self.doors is DoorState.CLOSED or self.doors is DoorState.CLOSING:
            return
        self._door_target = DoorState.CLOSED
        self._door_ticks_left = self.DOOR_CYCLE_TICKS
        self.doors = DoorState.CLOSING

    def _step_doors(self) -> "list[Event]":
        events: list[Event] = []
        if self.doors is DoorState.OPEN:
            if self._open_hold_ticks > 0:
                self._open_hold_ticks -= 1
            else:
                self.close_doors()
            return events

        if not self.doors.is_moving():
            return events

        self._door_ticks_left -= 1
        if self._door_ticks_left > 0:
            return events

        self.doors = self._door_target
        if self.doors is DoorState.OPEN:
            self._open_hold_ticks = self.DOOR_OPEN_TICKS
            events.append(Event(EventKind.DOORS_OPENED, self.car_id, self.floor))
        else:
            events.append(Event(EventKind.DOORS_CLOSED, self.car_id, self.floor))
        return events

    # ---- one tick of the world ----------------------------------------
    def step(self) -> "list[Event]":
        """Advance this car by ONE tick. Order matters and is not arbitrary:

          1. doors  — they may open, close, or reverse
          2. serve  — if we are at a stop and the doors are open, we are here
          3. move   — ONLY if the doors are fully CLOSED. This is the safety
                      invariant, and it is enforced by the guard, not by
                      convention.
        """
        events: list[Event] = []

        events.extend(self._step_doors())

        if self.floor in self.stops and self.doors is DoorState.CLOSED:
            self.remove_stop(self.floor)
            self.open_doors()
            events.append(Event(EventKind.ARRIVED, self.car_id, self.floor))

        if self.doors is not DoorState.CLOSED:
            self._assert_invariants()
            return events                       # cannot move with doors open

        # Nothing ahead in this direction and nothing to do: become IDLE.
        if not self.has_stops_ahead():
            if self.stops:
                # Stops remain, but all behind us. The scheduler decides
                # whether to reverse; the car merely reports that it cannot
                # continue forward.
                pass
            elif self.direction is not Direction.IDLE:
                self.direction = Direction.IDLE
                events.append(Event(EventKind.REVERSED, self.car_id, self.floor,
                                    "now idle"))

        self._assert_invariants()
        return events

    def move_one_floor(self, direction: Direction) -> "list[Event]":
        """Called by the controller ONLY after the scheduler has chosen a stop.
        Kept separate from step() so that the decision and the motion are not
        the same method — the car must never decide to move on its own."""
        events: list[Event] = []
        if self.doors is not DoorState.CLOSED:
            raise RuntimeError(
                f"{self.car_id} cannot move with doors {self.doors.value}"
            )
        if direction is Direction.IDLE:
            raise ValueError("cannot move in the IDLE direction")

        target = self.floor + direction.value
        if not 0 <= target <= self.top_floor:
            raise RuntimeError(f"{self.car_id} would move outside the shaft")

        self.floor = target
        self.direction = direction
        events.append(Event(EventKind.ARRIVED, self.car_id, self.floor, "moving"))
        return events

    def status(self) -> "dict[str, object]":
        return {
            "car_id": self.car_id,
            "floor": self.floor,
            "direction": self.direction.name,
            "doors": self.doors.value,
            "stops": sorted(self.stops),
            "weight_kg": round(self.weight_kg, 1),
        }

    def __repr__(self) -> str:
        return (f"ElevatorCar({self.car_id}, floor={self.floor}, "
                f"{self.direction.name}, doors={self.doors.value}, "
                f"stops={sorted(self.stops)})")
```

**The `move_one_floor` / `step` split is the design.** `step()` handles everything the car owns — doors, arrivals, direction housekeeping. `move_one_floor` is called only by the controller, after the scheduler has decided. **The car never decides to move**, and that separation is what makes the scheduler swappable and the single-car to multi-car change a configuration rather than a rewrite.

### 5.2 The scheduler — SCAN, and why it is the right default

```
   SCAN, drawn. The car is at 3 going UP. Stops {1, 5, 8}. A call arrives at 6, down.

     floor  9 ┤
            8 ┤  █  stop (ahead, same direction)      ← served
            7 ┤
            6 ┤  ○  call DOWN                        ← NOT served: wrong direction
            5 ┤  █  stop (ahead)                     ← served
            4 ┤
            3 ┤  ▲  car, going UP
            2 ┤
            1 ┤  █  stop (BEHIND)                    ← served AFTER reversing
            0 ┤

   The car goes 3 → 5 → 8 → (nothing ahead) → reverses → 6 → 1.

   NOTE what happened at 6: it was skipped on the way up and served on the
   way down. That is correct — a passenger going down is not helped by an
   up-moving car that stops and then continues up. It is also the behaviour
   users expect, which is not a coincidence: SCAN is what real lifts do.

   WHY NOT NEAREST-FIRST (SSTF): it starves. A car at floor 0 with calls at
   1 and 9 serves 1, then 9 is the nearest, then 1 again, forever — and the
   passenger at floor 5 waits indefinitely. This is the same starvation
   argument as the LFU cold-start in the cache problem, and it is worth
   naming: a greedy policy looks better on average and has an unbounded tail.
```

```python
class ScanScheduler(Scheduler):
    """SCAN — the elevator algorithm. The default, and it is the default for a
    reason: it bounds the worst-case wait, because a car sweeps the whole
    shaft in one direction before reversing. No request can be skipped twice.

    The cost is the average: a passenger just behind the car waits for a full
    sweep. That is the trade every building makes, and it is why the
    interviewer asking "why not nearest-first?" has a real answer.
    """

    def next_stop(self, car: ElevatorCar, pending: "list[Call]") -> "int | None":
        if car.is_full():
            # A full car does not stop for new passengers. It still serves the
            # passengers already inside, so it heads to its own nearest stop.
            return car.floors_ahead()[0] if car.floors_ahead() else None

        ahead = car.floors_ahead()
        if ahead:
            return ahead[0]

        # Nothing ahead. Is there a call we could pick up by reversing?
        if car.direction is Direction.IDLE:
            servable = [call for call in pending if car.can_serve(call)]
            if servable:
                return min(servable, key=lambda c: abs(c.floor - car.floor)).floor
            return None

        opposite = car.direction.opposite()
        behind = [
            call for call in pending
            if call.direction is opposite
            and (call.floor > car.floor if opposite is Direction.UP
                 else call.floor < car.floor)
        ]
        if behind:
            return min(behind, key=lambda c: abs(c.floor - car.floor)).floor
        return None

    def should_reverse(self, car: ElevatorCar, pending: "list[Call]") -> bool:
        if car.has_stops_ahead():
            return False
        if car.direction is Direction.IDLE:
            return False
        opposite = car.direction.opposite()
        return any(call.direction is opposite for call in pending)

    def assign(self, call: Call, cars: "list[ElevatorCar]") -> ElevatorCar:
        """With one car this is a formality. With many, the rule here is the
        simplest sane one: prefer an idle car, then the nearest car that can
        serve the call without reversing, then the nearest car at all."""
        if not cars:
            raise RuntimeError("no cars to assign to")

        def cost(car: ElevatorCar) -> "tuple[int, int]":
            can_serve_without_reversing = 0 if car.can_serve(call) else 1
            distance = abs(car.floor - call.floor)
            # A car that would have to reverse is worse even if it is closer,
            # because reversing strands its existing passengers. The tuple
            # makes that lexicographic.
            return (can_serve_without_reversing, distance)

        return min(cars, key=cost)


class NearestCarScheduler(ScanScheduler):
    """The variant worth being able to write, and worth being able to argue
    against. It assigns on distance alone — a strictly better average wait and
    a strictly worse tail, because a car is repeatedly diverted from its
    existing passengers.

    Kept in the file because the interviewer's follow-up is usually "why not
    just send the nearest car?", and the answer is this class: it is five
    lines, and its cost is that the passengers already inside a diverted car
    never arrive.
    """

    def assign(self, call: Call, cars: "list[ElevatorCar]") -> ElevatorCar:
        if not cars:
            raise RuntimeError("no cars to assign to")
        return min(cars, key=lambda car: (car.is_full(), abs(car.floor - call.floor)))
```

### 5.3 The controller — accept, dispatch, tick

```python
import threading


class ElevatorController:
    """Accepts presses from anywhere, assigns them, and advances the world.

    THE CONCURRENCY MODEL, stated because it is the interesting part:
      - A press can arrive from any floor and from inside any car, at any time,
        from any thread. So `press_call` and `press_floor` take the lock.
      - `tick()` takes the SAME lock, and holds it for the whole world step.
        That makes a tick atomic with respect to presses — a press either
        lands before the tick or after it, never during.
      - The lock is held for a few microseconds of pure computation. This is
        the "short critical section" case, and it is why one lock is fine here
        where one lock on a cache was not: there is no I/O inside it.
    """

    def __init__(self, cars: "list[ElevatorCar]", scheduler: Scheduler,
                 clock: Clock, top_floor: int | None = None) -> None:
        if not cars:
            raise ValueError("a controller needs at least one car")
        self._cars = list(cars)
        self._scheduler = scheduler
        self._clock = clock
        self._top_floor = top_floor if top_floor is not None else cars[0].top_floor
        self._pending: list[Call] = []
        self._served: list[Call] = []
        self._lock = threading.RLock()
        self._events: list[Event] = []

    @property
    def pending_calls(self) -> "list[Call]":
        with self._lock:
            return list(self._pending)

    @property
    def cars(self) -> "list[ElevatorCar]":
        return list(self._cars)

    def _validate_floor(self, floor: int) -> None:
        if not 0 <= floor <= self._top_floor:
            raise ValueError(f"floor {floor} is outside 0..{self._top_floor}")

    # ---- external presses ---------------------------------------------
    def press_call(self, floor: int, direction: Direction) -> None:
        """Someone in the corridor pressed a button. Note that the button
        being already lit is NOT an error — real buttons are idempotent, and a
        duplicate press must not create a duplicate call that the car then
        serves twice."""
        self._validate_floor(floor)
        if direction is Direction.IDLE:
            raise ValueError("a corridor call needs a direction")
        if floor == 0 and direction is Direction.DOWN:
            raise ValueError("cannot call down from the bottom floor")
        if floor == self._top_floor and direction is Direction.UP:
            raise ValueError("cannot call up from the top floor")

        with self._lock:
            call = Call(floor, direction, self._clock.now())
            if call in self._pending:
                return                               # the light is already on
            self._pending.append(call)
            car = self._scheduler.assign(call, self._cars)
            car.add_stop(floor)
            self._events.append(Event(EventKind.CALL_ASSIGNED, car.car_id, floor,
                                      f"{direction.name} call"))

    def press_floor(self, car_id: str, floor: int) -> None:
        """Someone inside a car pressed a floor button. This one carries no
        direction: the car's own direction decides how it is served."""
        self._validate_floor(floor)
        with self._lock:
            car = self._car(car_id)
            if car.add_stop(floor):
                self._events.append(Event(EventKind.STOP_ADDED, car_id, floor,
                                          "car call"))

    def press_door_open(self, car_id: str) -> None:
        with self._lock:
            self._car(car_id).open_doors()

    def press_door_close(self, car_id: str) -> None:
        with self._lock:
            self._car(car_id).close_doors()

    def _car(self, car_id: str) -> ElevatorCar:
        for car in self._cars:
            if car.car_id == car_id:
                return car
        raise KeyError(f"unknown car {car_id}")

    def board(self, car_id: str, weight_kg: float) -> bool:
        """Boarding is not a press — it changes the load, and the load decides
        whether the car answers further calls."""
        with self._lock:
            return self._car(car_id).board(weight_kg)

    # ---- the world tick -----------------------------------------------
    def tick(self) -> "list[Event]":
        """Advance the whole building by one tick. This is the method the
        whole design exists to make correct, and the order is deliberate:

          1. cars step          — doors, arrivals, direction housekeeping
          2. retire served calls — a call whose floor the car is now AT is done
          3. reverse if needed   — the SCAN turn, decided per car
          4. dispatch            — move each car one floor toward its next stop
        """
        with self._lock:
            events: list[Event] = []

            for car in self._cars:
                events.extend(car.step())

            # 2. A pending call is served once a car is at its floor with the
            #    doors not closed. Retiring it here — rather than when the
            #    stop is removed — is what keeps `pending` and the cars' stop
            #    sets consistent.
            still_pending: list[Call] = []
            for call in self._pending:
                serving = [
                    car for car in self._cars
                    if car.floor == call.floor and car.doors is not DoorState.CLOSED
                ]
                if serving:
                    self._served.append(call)
                else:
                    still_pending.append(call)
            self._pending = still_pending

            # 3. Reverse a car that has nothing left ahead and calls behind.
            for car in self._cars:
                if self._scheduler.should_reverse(car, self._pending):
                    car.direction = car.direction.opposite()
                    events.append(Event(EventKind.REVERSED, car.car_id, car.floor,
                                        f"now {car.direction.name}"))

            # 4. Dispatch: one floor per tick, and ONLY if the doors are closed.
            for car in self._cars:
                target = self._scheduler.next_stop(car, self._pending)
                if target is None or target == car.floor:
                    continue
                if car.doors is not DoorState.CLOSED:
                    continue                          # doors win; the car waits
                direction = Direction.UP if target > car.floor else Direction.DOWN
                if car.direction is not Direction.IDLE and direction is not car.direction:
                    continue                          # do not reverse mid-sweep
                events.extend(car.move_one_floor(direction))

            self._events.extend(events)
            return events

    def run_until_idle(self, max_ticks: int = 10_000) -> int:
        """Drive the simulation until nothing is pending and no car has stops.
        A BOUNDED loop: an unbounded `while True` in a simulation is how a
        test hangs forever when a bug makes a car oscillate, and a bounded
        loop turns that hang into a failure with a tick count."""
        ticks = 0
        while ticks < max_ticks:
            with self._lock:
                busy = bool(self._pending) or any(car.stops for car in self._cars)
            if not busy:
                return ticks
            self.tick()
            ticks += 1
        raise RuntimeError(
            f"did not settle within {max_ticks} ticks; pending={self._pending}, "
            f"cars={self._cars}"
        )

    # ---- observability -------------------------------------------------
    def status(self) -> "dict[str, object]":
        with self._lock:
            return {
                "cars": [car.status() for car in self._cars],
                "pending": [repr(call) for call in self._pending],
                "served": len(self._served),
            }

    def wait_statistics(self) -> "dict[str, float]":
        """Worst-case wait is the number a building manager cares about, and it
        is the number SCAN is chosen to bound. Reporting the average alone is
        how a scheduling change that improves the mean and doubles the tail
        gets shipped."""
        with self._lock:
            if not self._served:
                return {"count": 0.0, "avg": 0.0, "max": 0.0}
            waits = [call.wait_time(self._clock.now()) for call in self._served]
            return {
                "count": float(len(waits)),
                "avg": sum(waits) / len(waits),
                "max": max(waits),
            }
```

**The `run_until_idle` bound is a small detail with a real payoff.** A lift simulation that oscillates — a car that reverses, then reverses again, forever — hangs an unbounded loop. A tick cap turns that into `RuntimeError: did not settle within 10000 ticks` alongside the state, which is a bug report rather than a timeout.

### 5.4 It runs

```python
class ManualClock(Clock):
    """Deterministic ticks. Every test in this file runs in microseconds."""

    def __init__(self) -> None:
        self._ticks = 0.0

    def now(self) -> float:
        return self._ticks

    def tick(self) -> None:
        self._ticks += 1.0


def demo() -> None:
    clock = ManualClock()
    car = ElevatorCar("L1", top_floor=9)
    controller = ElevatorController([car], ScanScheduler(), clock)

    # Two people: one at 5 going up, one at 7 going down.
    controller.press_call(5, Direction.UP)
    controller.press_call(7, Direction.DOWN)

    print("after presses:", car)              # stops={5, 7}, direction IDLE

    ticks = 0
    while ticks < 40:
        car_was = (car.floor, car.direction.name, car.doors.value, sorted(car.stops))
        events = controller.tick()
        clock.tick()
        ticks += 1
        car_now = (car.floor, car.direction.name, car.doors.value, sorted(car.stops))
        if car_now != car_was or events:
            print(f"t{ticks:02d}", car_now, [e.kind.value for e in events])
        if not car.stops and not controller.pending_calls:
            break

    print("settled after", ticks, "ticks")
    print(controller.status())
    print(controller.wait_statistics())
    # The car serves 5 on the way up, then reverses at 7 — the DOWN call — and
    # the DOWN passenger is served on the way down. That is SCAN working.

    # A doorway reversal: the doors are closing and someone presses open.
    car2 = ElevatorCar("L2", top_floor=9)
    car2.add_stop(0)
    for _ in range(10):
        car2.step()
        if car2.doors is DoorState.CLOSING:
            car2.open_doors()
            print("reversed the doors:", car2.doors.value)
            break


if __name__ == "__main__":
    demo()
```

### 5.5 The multi-car upgrade, in code

```python
def build_building(car_count: int, top_floor: int = 9,
                   clock: "Clock | None" = None) -> ElevatorController:
    """The composition root. One scheduler instance SHARED across cars, because
    `assign` must see all of them — a per-car scheduler cannot answer "which
    car should take this call", which is the entire multi-car question."""
    if car_count < 1:
        raise ValueError("a building needs at least one lift")
    clk = clock if clock is not None else ManualClock()
    cars = [
        ElevatorCar(f"L{i + 1}", top_floor=top_floor)
        for i in range(car_count)
    ]
    return ElevatorController(cars, ScanScheduler(), clk, top_floor=top_floor)
```

**The single design change for multiple cars is the `assign` policy, and the `Scheduler` interface already had it.** Nothing on `ElevatorCar` changes, nothing on `ElevatorController` changes except that it holds more cars. **That is the payoff for putting the car's movement decisions in the scheduler from the start**, and it is the thing to point at when the interviewer says "now add a second lift".

---

## 6. Concurrency and thread-safety

### 6.1 What is shared, and what races

| # | Race | Consequence | Fix |
|---|---|---|---|
| 1 | Two threads press the same call | Two `Call`s for one floor; the car stops twice | `Call` is value-equal; `press_call` deduplicates under the lock |
| 2 | A press lands mid-tick | The car's `stops` is mutated while the scheduler iterates it | `RLock` around the whole tick and every press |
| 3 | Two threads `tick()` concurrently | The car moves two floors in one tick; doors open while moving | `tick()` holds the lock; the controller is single-drivered |
| 4 | Boarding races the door close | A passenger boards a car that is already moving | Boarding takes the lock; `move_one_floor` refuses with the doors not `CLOSED` |
| 5 | Weight updated without a lock | Two passengers each fit, together they overload | `board` is atomic under the lock |
| 6 | `stops` mutated while `floors_ahead` sorts it | `RuntimeError: set changed size during iteration` | The scheduler runs inside the tick's lock |
| 7 | The door-open button during `CLOSING` | The reversal is lost; a passenger is caught | `open_doors` handles `CLOSING` explicitly — the reversal is modelled |
| 8 | Two cars assigned the same call | Both stop; one arrives with nobody waiting | `assign` returns one car; the stop is added to that car only |
| 9 | The pending list mutated while the scheduler reads it | Skipped or double-served calls | One lock over the whole tick |
| 10 | `run_until_idle` reads `stops` while a tick mutates it | A false "idle" and an early exit | The busy check is inside the lock |

### 6.2 Why one lock is right here, and was wrong for the cache

This is the comparison worth making explicitly, because a candidate who learned "one global lock is bad" from the caching problem will over-engineer this one.

```
   CACHE                                     ELEVATOR
   ─────                                     ────────
   Called thousands of times a second        Ticked ~10 times a second
   I/O inside the critical section           Pure computation inside it
   get() MUTATES (recency) so reads          tick() is genuinely exclusive:
     cannot share, and the lock is held        the world advances once
     for real work
   → shard                                   → one lock is correct

   The rule is not "one lock is bad". The rule is: a lock is bad when it is
   held across work that does not need it, or when unrelated keys contend
   on it. Here every operation touches the whole world, and there is no I/O,
   so a single lock is the honest answer — and saying WHY is the point.
```

### 6.3 Testing the invariants

```python
def test_doors_closed_before_moving() -> None:
    """THE safety invariant. Drive a car hard and assert it never moves with
    its doors not closed."""
    clock = ManualClock()
    car = ElevatorCar("L1", top_floor=9)
    controller = ElevatorController([car], ScanScheduler(), clock)

    for floor in (3, 7, 1, 9, 0, 5):
        controller.press_floor("L1", floor)
        while car.stops:
            previous_floor = car.floor
            previous_doors = car.doors
            controller.tick()
            clock.tick()
            if car.floor != previous_floor:
                assert previous_doors is DoorState.CLOSED, (
                    f"car moved from {previous_floor} to {car.floor} "
                    f"with doors {previous_doors.value}"
                )


def test_an_up_car_does_not_answer_a_down_call_behind_it() -> None:
    """The rule that stops the 'stops going the wrong way' bug."""
    car = ElevatorCar("L1", top_floor=9)
    car.floor = 5
    car.direction = Direction.UP
    car.add_stop(8)

    assert car.can_serve(Call(7, Direction.UP, 0.0)) is True    # ahead, same way
    assert car.can_serve(Call(7, Direction.DOWN, 0.0)) is False # ahead, wrong way
    assert car.can_serve(Call(3, Direction.UP, 0.0)) is False   # behind
    assert car.can_serve(Call(3, Direction.DOWN, 0.0)) is False # behind


def test_an_idle_car_answers_anything() -> None:
    car = ElevatorCar("L1", top_floor=9)
    car.floor = 5
    assert car.can_serve(Call(0, Direction.UP, 0.0)) is True
    assert car.can_serve(Call(9, Direction.DOWN, 0.0)) is True


def test_a_full_car_does_not_stop_for_new_calls() -> None:
    car = ElevatorCar("L1", top_floor=9, capacity_kg=200.0)
    car.floor = 3
    car.direction = Direction.UP
    assert car.board(150.0) is True
    assert car.can_serve(Call(5, Direction.UP, 0.0)) is True
    assert car.board(100.0) is False                  # would overload
    assert car.weight_kg == 150.0                     # unchanged
    car.board(50.0)
    assert car.is_full() is True
    assert car.can_serve(Call(5, Direction.UP, 0.0)) is False


def test_a_car_will_not_close_its_doors_while_overloaded() -> None:
    car = ElevatorCar("L1", top_floor=9, capacity_kg=100.0)
    car.board(100.0)
    car.open_doors()
    car.close_doors()
    assert car.doors is not DoorState.CLOSING


def test_a_closing_door_can_be_reversed() -> None:
    """The doorway reversal — the safety behaviour that stops a passenger
    being caught. CLOSING exists as a state for exactly this."""
    car = ElevatorCar("L1", top_floor=9)
    car.open_doors()
    for _ in range(10):
        car.step()
        if car.doors is DoorState.CLOSING:
            break
    assert car.doors is DoorState.CLOSING
    car.open_doors()
    assert car.doors is DoorState.OPENING
    for _ in range(10):
        car.step()
        if car.doors is DoorState.OPEN:
            break
    assert car.doors is DoorState.OPEN


def test_scan_does_not_starve_a_call_behind_the_car() -> None:
    """The property SCAN exists for: every request is served within one full
    sweep. Nearest-first has no such bound, and this test is the difference."""
    clock = ManualClock()
    controller = build_building(1, top_floor=9, clock=clock)
    car = controller.cars[0]

    controller.press_call(1, Direction.UP)
    controller.press_call(9, Direction.DOWN)
    controller.press_call(5, Direction.UP)

    ticks = controller.run_until_idle(max_ticks=200)
    assert not controller.pending_calls
    assert not car.stops
    # 5 is behind the car once it heads to 1, and it is still served within
    # the same sweep rather than deferred to a second lap.
    assert ticks < 60, f"took {ticks} ticks; a sweep of 10 floors is ~30"


def test_concurrent_presses_never_lose_a_call() -> None:
    clock = ManualClock()
    controller = build_building(2, top_floor=9, clock=clock)

    barriers = threading.Barrier(8)
    errors: list[str] = []

    def presser(worker: int) -> None:
        barriers.wait()
        try:
            for i in range(50):
                floor = (worker * 7 + i) % 10
                controller.press_call(floor, Direction.UP if floor < 9 else Direction.DOWN)
        except Exception as exc:                          # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=presser, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    controller.run_until_idle(max_ticks=5_000)
    assert not controller.pending_calls
    for car in controller.cars:
        car._assert_invariants()
        assert not car.stops
```

---

## 7. Extensibility — the standard follow-ups

### 7.1 "Make the scheduler pluggable and compare them"

Already an interface. The comparison is the interesting follow-up, and it is measurable because `wait_statistics` reports the maximum as well as the mean.

```python
def compare_schedulers(calls: "list[Call]", car_count: int = 1) -> "dict[str, dict[str, float]]":
    """A benchmark harness. Note it rebuilds the world per policy rather than
    trying to reuse one — a scheduler comparison run on dirty state is a
    comparison of nothing."""
    results: "dict[str, dict[str, float]]" = {}
    for name, scheduler in (("scan", ScanScheduler()), ("nearest", NearestCarScheduler())):
        clock = ManualClock()
        cars = [ElevatorCar(f"L{i + 1}", top_floor=9) for i in range(car_count)]
        controller = ElevatorController(cars, scheduler, clock)
        for call in calls:
            controller.press_call(call.floor, call.direction)
        try:
            controller.run_until_idle(max_ticks=5_000)
        except RuntimeError as exc:
            # A policy that never settles is a RESULT, not a crash. Reporting
            # it lets the comparison table say "scan: max 34s / nearest: did
            # not settle" instead of aborting the benchmark.
            results[name] = {"count": 0.0, "avg": 0.0, "max": float("inf"),
                             "did_not_settle": 1.0}
            continue
        results[name] = controller.wait_statistics()
    return results
```

**The expected result, stated so it can be checked:** nearest-first wins on the average and loses on the maximum, and on a single car with mixed traffic it can lose on both because it keeps reversing. **Knowing which number matters is the answer** — a building's complaint is always about the worst wait, not the mean.

### 7.2 "Now add a second lift"

Discussed in §5.5: the change is the `assign` policy, and nothing else. Three assignment policies worth being able to name:

- **Nearest car** — best average, worst tail, diverts cars with passengers.
- **Idle car first, then nearest that can serve** — what `ScanScheduler.assign` does. Simple, and it keeps busy cars on their existing route.
- **Zoning / sectoring** — in a tall building, cars own a floor range and hand off at a transfer floor. This is what a 40-floor tower actually does, and **the class change is that a `Car` gains a `zone` and `assign` filters by it**. Worth naming because it is the answer for a building the single-shaft model cannot serve, and it shows the model has a scale limit.

### 7.3 "Add a passenger and a boarding model"

The follow-up: passengers with destinations, and a car that must know whether to stop.

```python
class Passenger:
    """A person, with an origin, a destination and a weight. Adding this makes
    the simulation measurable in the units the business cares about — a
    passenger's total time, not a call's wait time — and the two differ,
    because a passenger's journey includes the ride, not just the wait."""

    __slots__ = ("passenger_id", "origin", "destination", "weight_kg", "boarded_at",
                 "arrived_at")

    def __init__(self, passenger_id: str, origin: int, destination: int,
                 weight_kg: float = 75.0) -> None:
        if origin == destination:
            raise ValueError("a passenger going nowhere is not a passenger")
        if weight_kg <= 0:
            raise ValueError("a passenger has positive weight")
        self.passenger_id = passenger_id
        self.origin = origin
        self.destination = destination
        self.weight_kg = weight_kg
        self.boarded_at: "float | None" = None
        self.arrived_at: "float | None" = None

    @property
    def direction(self) -> Direction:
        return Direction.UP if self.destination > self.origin else Direction.DOWN

    def total_time(self) -> "float | None":
        if self.arrived_at is None or self.boarded_at is None:
            return None
        return self.arrived_at - self.boarded_at

    def __repr__(self) -> str:
        return f"Passenger({self.passenger_id}, {self.origin}→{self.destination})"
```

**The measurable difference is the point:** a scheduler that minimises call wait can increase passenger total time, because it stops more often on the way. Reporting both is the honest evaluation, and it is the same lesson as `Q5` in the caching problem — a heuristic is judged by the metric you actually care about, and choosing the metric is part of the design.

### 7.4 "Add a destination panel"

The inversion from §1.3. The changes are three, and each is small:

```python
class DestinationRequest:
    """A request from a destination panel. Note it carries the DESTINATION at
    request time — which is the entire difference from a corridor call, and it
    is what lets the panel group passengers going the same way into one car."""

    __slots__ = ("origin", "destination", "created_at")

    def __init__(self, origin: int, destination: int, created_at: float) -> None:
        if origin == destination:
            raise ValueError("origin and destination must differ")
        self.origin = origin
        self.destination = destination
        self.created_at = created_at

    @property
    def direction(self) -> Direction:
        return Direction.UP if self.destination > self.origin else Direction.DOWN


class DestinationScheduler(ScanScheduler):
    """The panel-aware policy. Because the destination is known, the "serve in
    the current direction" rule becomes "serve if the DESTINATION is ahead" —
    and a passenger going the wrong way for the car is never assigned at all,
    rather than being assigned and then skipped.

    The gain is real: grouping same-direction passengers cuts the number of
    stops, which is the dominant term in ride time. The cost is that the
    passenger must choose a floor before entering, so a change of mind means
    leaving the car and re-requesting.
    """

    def next_stop(self, car: ElevatorCar, pending: "list[Call]") -> "int | None":
        # The panel's requests are already stops on the car that was assigned
        # them, so the car's own stop set is the whole answer. The interesting
        # work happens in `assign`, not here.
        ahead = car.floors_ahead()
        if ahead:
            return ahead[0]
        return super().next_stop(car, pending)

    def assign(self, call: Call, cars: "list[ElevatorCar]") -> ElevatorCar:
        return super().assign(call, cars)
```

**The honest note:** with a panel, the *assignment* is the design and the car's own scheduling is nearly trivial, because the full trip is known. That is the inverse of the corridor-call model, where assignment is nearly trivial and the sweep rule is the design. **Saying which one is doing the work is the answer to "what changes with a destination panel?"**

### 7.5 "A lift goes out of service"

```python
class CarAvailability(Enum):
    IN_SERVICE = "in_service"
    OUT_OF_SERVICE = "out_of_service"
    FIRE_SERVICE = "fire_service"


def _assignable(cars: "list[ElevatorCar]") -> "list[ElevatorCar]":
    """The filter that must be applied in `assign` AND in `next_stop`'s
    callers. A car out of service must not be assigned new calls, and must not
    be counted when deciding whether a call can be served."""
    return [car for car in cars if getattr(car, "availability", None) is None
            or car.availability is CarAvailability.IN_SERVICE]
```

**Two rules worth stating.** First, a car going out of service must **finish its current passengers** or hand them off — dropping them mid-journey is worse than a longer wait. Second, if *all* cars are out of service, the controller must refuse new calls with a clear error rather than queue forever — **the queue-forever failure is the one that surprises people**, because the simulation looks healthy while nothing happens.

### 7.6 "Fire service mode"

The override. In fire service the car ignores all calls, returns to the ground floor, parks with its doors open, and is under the fire brigade's control. **The design point is that this is an override, not a policy** — it is checked before the scheduler is consulted, and it must be able to interrupt a sweep mid-flight. Worth naming as an override rather than pretending it is another `Scheduler` implementation, because a scheduler can be swapped at any tick and an override must be immediate.

---

## 8. Failure modes and edge cases

### 8.1 Failure modes

| # | Failure | Consequence | Handling |
|---|---|---|---|
| 1 | The car moves with the doors not `CLOSED` | **Catastrophic** — a passenger is crushed | `move_one_floor` raises; `tick` skips the move; asserted in a test |
| 2 | A door reversal during `CLOSING` is lost | A passenger is caught | `open_doors` handles `CLOSING` explicitly |
| 3 | An up car stops for a down call | The "stops going the wrong way" bug | `can_serve` refuses; the call is served after the sweep reverses |
| 4 | Nearest-first starves a floor | Unbounded wait | SCAN; the sweep bound is the whole point |
| 5 | A full car stops for new passengers | It stops, cannot board, and moves on — the wait was wasted | `can_serve` returns `False` when `is_full()` |
| 6 | Doors close while overloaded | A car that sits with doors shut and cannot move | `close_doors` refuses when full |
| 7 | Two cars assigned one call | Two stops, one passenger | `assign` returns exactly one car |
| 8 | A call pressed twice creates two calls | The car stops twice for one passenger | `Call` is value-equal; `press_call` deduplicates |
| 9 | The call list never drains | A phantom call; a car that returns forever | Retiring a call only when a car is at its floor with the doors not closed — and the invariant that every pending call has a corresponding stop |
| 10 | A pending call is retired with no car serving it | The passenger waits forever | The retire check requires a car at the floor |
| 11 | A car oscillates between two floors | An infinite loop; a hung simulation | `run_until_idle` is bounded and raises with the state |
| 12 | A call at the top floor going up | An impossible call | Rejected: there is nothing above |
| 13 | A call at the bottom floor going down | An impossible call | Rejected |
| 14 | A corridor press mid-tick | A corrupted stop set | One `RLock` over every press and the whole tick |
| 15 | Boarding races the door close | A passenger boards a moving car | `board` and `move_one_floor` both take/require the lock and the door state |
| 16 | A car out of service still assigned | Nobody comes | The assignable filter, applied in `assign` |
| 17 | Every car out of service | Calls queue forever, silently | Refuse with an error rather than queue |
| 18 | The car moves outside the shaft | An index error, or worse a wraparound | `move_one_floor` refuses past either end |
| 19 | A stop added for an invalid floor | A car that never arrives | `add_stop` validates against `top_floor` |
| 20 | An unbounded `while True` simulation | A hung test | A tick cap with a state dump |

### 8.2 Edge cases worth naming

- **One floor building.** `top_floor = 0` is rejected: there is nothing to schedule.
- **Two floors.** The degenerate case, and a good test: calls up from 0 and down from 1 only.
- **A call for the floor the car is already at, in the direction it is already going.** It should be served immediately, not queued.
- **A call at the car's floor in the opposite direction.** The car must not answer it while committed, but must answer it after reversing — and the car should not need to move to do so.
- **A passenger pressing both up and down.** Two calls, both legitimate.
- **A passenger pressing their own origin floor inside the car.** A no-op stop; the car opens its doors where it already is.
- **A car with no stops and no calls.** `IDLE`; the controller should report it and not drive it.
- **A stop added while the car is at that floor with the doors open.** Served immediately; the stop should not linger.
- **Weight exactly at capacity.** Boarding succeeds and `is_full()` is true; the next passenger is refused. The boundary is `>=`, not `>`.
- **A passenger alighting takes the weight below capacity.** `can_serve` becomes true again on the next tick — the state is recomputed, not cached.

### 8.3 Complexity

| Operation | Cost | Note |
|---|---|---|
| `press_call` | O(cars) | The dedupe scan, plus `assign` |
| `press_floor` | O(1) | A set insert |
| `tick`, single car | O(stops log stops) | The `floors_ahead` sort |
| `tick`, C cars | O(C × stops log stops) | Per car |
| `assign`, naive | O(C) | A linear scan with a cost function |
| `run_until_idle` | O(ticks × C × stops log stops) | Ticks are bounded by the sweep length |
| A full sweep, F floors | ~2F ticks plus door time | 10 floors ≈ 30 ticks |

**The line to memorise:** *"A tick is O(C × S log S) where S is the stops on a car — the sort is what makes it stop in order. The scheduling policy is O(C) per assignment, and at a handful of cars and a few hundred floors a linear scan is the right answer; a heap per car would be the answer at a scale lifts do not reach."*

---

## 9. Design patterns used, and why

| Pattern | Where | Why here | What it costs |
|---|---|---|---|
| **State** | `ElevatorCar` with `DoorState` and the guarded transitions | The door cycle has four states with reversals and safety rules; centralising them is what stops an illegal transition | A state machine to reason about — and no shortcut, because this one is safety-critical |
| **Strategy** | `Scheduler` → `ScanScheduler`, `NearestCarScheduler`, `DestinationScheduler` | The policy is the axis that varies, and the follow-up is always "what if you scheduled differently?" | An interface and a shared instance |
| **Observer** *(light)* | `Event` stream from `tick()` | The display, the logger and the tests consume one stream instead of polling and inferring | Consumers must handle the events they do not care about |
| **Mediator** | `ElevatorController` | Cars do not talk to one another; the controller coordinates, which is what stops an N² car-to-car protocol | A central point that must not become a god object |
| **Command** *(rejected)* | — | Considered for presses as replayable objects; rejected because a `Call` is already an immutable value and replaying presses through a queue buys nothing here | — |
| **Singleton** *(rejected)* | — | A global controller breaks multi-building tests and the two-building comparison | — |

**Mediator is the one to name.** With multiple cars, the alternative to a mediator is cars negotiating with each other — and that is an N² protocol with a deadlock risk. **One coordinator with one lock is the correct shape for a building**, and the honest note is that it would not be at a scale where the cars were geographically distributed, which is not a building.

---

## 10. Trade-offs

| Choice | Buys | Costs |
|---|---|---|
| SCAN over nearest-first | A bounded worst-case wait; no starvation | A worse average wait than a greedy policy |
| One lock over the whole world | Correct and simple; no I/O inside it | One tick at a time, which is correct by definition for a simulation |
| The car owns its stop set | The single-car case is local; the car does not query the controller | The controller must keep `pending` consistent with the cars' stops |
| `Call` does not name a car | Reassignment when a better car frees up | An assignment step on every press |
| Doors as a four-state machine | The reversal is modelled, and a passenger is not caught | More states, more tests — and it is not optional |
| Movement only via `move_one_floor` | The safety invariant is enforceable in one place | The car cannot move itself, so the controller must always drive it |
| Weights gate new stops | No wasted stops on a full car | A near-full car may refuse a passenger who would have fitted after an alighting |
| A bounded `run_until_idle` | A hung simulation becomes a failure with state | A legitimate long run must raise the cap |
| Two numbers in `wait_statistics` | A policy change that helps the mean and hurts the tail is visible | Nobody reads the second number unless you make them |
| `Scheduler` stateless | Testable in isolation; safe to share across cars | A stateful policy must hold its own state explicitly |

**What changes at 10×:** the model. Ten cars in one shaft is not a scheduling problem, it is a zoning problem — cars own floor ranges and hand off at transfer floors, which is what a 40-floor tower does. At 10× the floors and the cars the class design's limit is the single shared lock, and the honest answer is that a building of that size partitions into zones, each with its own controller, and the "system" becomes a set of controllers with a handoff protocol. **The class design does not survive 10× unchanged, and saying so is better than pretending it does.**

---

## 11. Interview Q&A

**Q1 (easy). What are the core classes?**
`ElevatorCar` owns its floor, direction, doors, stops and load. `Call` is an external request — a floor and a direction — and deliberately does not name a car. `CarRequest` is an internal press, which has no direction of its own because the car's direction supplies it. `Scheduler` decides the next stop and which car answers a call. `ElevatorController` accepts presses, assigns them and ticks the world.

**Q2 (easy). What is the scheduling algorithm, and why that one?**
SCAN — the elevator algorithm. The car keeps moving in its current direction until it has nothing left ahead, then reverses. The reason is that it bounds the worst-case wait: a car sweeps the whole shaft before turning, so no request can be skipped twice. Nearest-first looks better on average and has an unbounded tail — a car oscillating between two nearby floors starves a third indefinitely, which is the same starvation argument as LFU's cold-start in the cache problem.

**Q3 (easy). Why do the doors have four states instead of open and closed?**
Because `OPENING` and `CLOSING` are real, and the reversal out of `CLOSING` is a safety behaviour. If someone steps into the doorway while the doors are closing, the doors must reopen — and that is only expressible if `CLOSING` is a state you can transition out of. With a boolean you either reverse an instantaneous transition, which you cannot, or you ignore the passenger, which you must not.

**Q4 (medium). Walk me through the invariant that matters most.**
A car never moves unless its doors are fully `CLOSED`. It is enforced in one place — `move_one_floor` raises if the doors are not `CLOSED`, and `tick` skips the move rather than raising — and it has a test that drives a car through many floors and asserts that every floor change happened with the doors closed. It matters more than any other rule in the file, because the consequence is not a bug but an injury, and it is why the door state machine is not optional.

**Q5 (medium). An up-moving car gets a down call ahead of it. What happens?**
It does not answer it. `can_serve` returns false for a call whose direction differs from the car's, unless the car has nothing ahead of it. The call stays pending, the car finishes its sweep, reverses, and picks it up on the way down. The alternative — stopping for it — is the bug every user has experienced: a lift arrives, opens its doors, and continues upward, and the passenger going down has learned nothing. The rule is also why the wait is bounded rather than arbitrary.

**Q6 (medium). How do you handle a full car?**
`can_serve` returns false when `is_full()`, so the car does not add new stops and does not waste a stop on passengers it cannot take. `close_doors` refuses to close while overloaded, which prevents the car that sits with its doors shut and cannot move. And `board` returns false rather than boarding and then refusing to move — the failure where a full car closes on a passenger's foot.

**Q7 (medium). How is the controller made thread-safe?**
One `RLock` around every press and around the whole `tick()`. A press can arrive from any floor and any car at any moment, so the lock must cover the pending list, the cars' stop sets and the scheduler's read of both. It is the right call here, unlike the caching problem, because the critical section is pure computation with no I/O and because every operation genuinely touches the shared world — there is no unrelated key to contend on.

**Q8 (medium). *Pushback:* "You learned one global lock is bad in the caching problem. Why is it fine here?"**
Because the rule was never "one lock is bad" — it was "a lock held across work that does not need it, or contended by unrelated keys, is bad". In a cache, `get` mutates the recency order so the lock must be exclusive and it is held for real work, and thousands of unrelated keys all queue on the same mutex. Here a tick is ten times a second, it is pure arithmetic, and every operation touches the entire building. Sharding a lift controller would mean two threads advancing the same shaft independently, which is not an optimisation, it is a bug.

**Q9 (medium). *Pushback:* "Why not just send the nearest car?"**
Because nearest-first measures the wrong thing. It minimises the distance for the new call and ignores the passengers already in the car being diverted, so a car on its way to floor 9 with passengers aboard gets turned around for a call at floor 2, and the people inside never arrive. The `assign` policy I would use prefers an idle car, then the nearest car that can serve the call *without reversing*, and only then the nearest car at all — a lexicographic cost rather than a scalar distance. I keep a `NearestCarScheduler` in the design specifically so the comparison is testable, and the expected result is a better average and a worse maximum.

**Q10 (medium). *Pushback:* "Your call-retiring logic checks whether any car is at the floor with doors not closed. Isn't that fragile?"**
It is the honest rule rather than a fragile one, and the alternative is worse. A call is satisfied when a car has actually arrived and opened — not when a stop is queued, because a queued stop can be dropped, and not when the car reaches the floor, because it may pass through without stopping. The fragility people worry about is a call being retired by a car that was merely passing, which is why the door state is part of the condition. The real safeguard is an invariant: every pending call has a corresponding stop on exactly one car, and a test asserts that after settling both are empty.

**Q11 (hard). *Pushback:* "What if the call list never drains because of a bug? Your simulation would hang."**
That is exactly why `run_until_idle` has a tick cap and why it raises with the pending calls and the car states attached, rather than looping. An unbounded `while True` in a simulation turns a logic bug into a timeout with no information. With the cap, a car that oscillates between two floors produces `RuntimeError: did not settle within 10000 ticks; pending=[Call(floor=7, UP)], cars=[ElevatorCar(L1, floor=4, UP, stops=[5, 5])]` — which names the bug directly.

**Q12 (hard). What changes with a destination panel?**
The input model inverts, and with it which part of the design does the work. With corridor buttons, the assignment is nearly trivial and the sweep rule is the design. With a panel, the destination is known at request time, so passengers going the same way can be grouped into one car, and the *assignment* becomes the design while the car's own scheduling is nearly trivial. Concretely: `Request` carries a destination, the "serve in my direction" rule becomes "serve if the destination is ahead", and a passenger going the wrong way is never assigned rather than assigned and skipped. The gain is fewer stops, which is the dominant term in ride time; the cost is that the passenger must commit to a floor before entering.

**Q13 (hard). What would you do differently at 10× — say 40 floors and 10 cars?**
The single-shaft model stops being right, and the change is zoning rather than a faster scheduler. In a tall building cars own floor ranges and hand off at transfer floors, which means `ElevatorCar` gains a zone and `assign` filters by it — and the single shared lock becomes one controller per zone with a handoff protocol between them. That is a bigger change than the model can absorb unchanged, and I would say so rather than claim the class design scales. What does scale unchanged is the car itself: the door machine, the safety invariant and the stop set are the same in a 40-floor tower.

**Q14 (medium). How do you test a scheduler?**
Two levels. Unit: `can_serve` for every combination of car direction, call direction and relative position — that is a small truth table and it is where the "stops going the wrong way" rule lives. And simulation: press a mixed set of calls, run to idle under a bounded tick count, then assert three things — nothing pending, no car has stops, and the tick count is within the sweep bound. That last assertion is the SCAN property, and it is what a nearest-first scheduler would fail on a crafted workload.

**Q15 (medium). What is the biggest modelling mistake in this problem?**
Letting the car decide where to go. If `ElevatorCar` picks its own next stop, the single-car case looks fine and the multi-car case has no coordinator — each car optimises locally and they collide on the same call. Splitting it so the scheduler decides and the car merely moves is what makes the multi-car extension a configuration change. The second biggest is letting a `Call` name the car that will answer it, which bakes the assignment into the request and prevents reassignment when a better car frees up.

---

## 12. Cheat-sheet summary

**The model**

```
  Call           = external: floor + direction. Does NOT name a car.
  CarRequest     = internal: floor only. The car's direction supplies the rest.
  ElevatorCar    = floor, direction, doors, stops, weight. Moves; does not decide.
  Scheduler      = next_stop(car) + assign(call, cars) + should_reverse(car)
  Controller     = presses, pending calls, ONE lock, tick()
```

**The safety invariant**

```
  A CAR NEVER MOVES WITH ITS DOORS NOT FULLY CLOSED.
  Enforced in move_one_floor (raises) and in tick (skips the move).
  Asserted by a test that drives many floors and checks every transition.

  DOOR CYCLE: CLOSED → OPENING → OPEN → CLOSING → CLOSED
                                  ▲         │
                                  └── reversal (a passenger in the doorway)
```

**The scheduling rule (SCAN)**

```
  Keep the direction. Serve only stops AHEAD in that direction.
  Nothing ahead + calls behind  →  reverse.
  An up car does NOT answer a down call. It answers it after reversing.

  WHY: it BOUNDS the worst-case wait — a full sweep cannot skip anyone twice.
  WHY NOT NEAREST-FIRST: it starves. A car oscillating between 1 and 9
  leaves floor 5 waiting forever. Better average, unbounded tail.
```

**The service-direction truth table**

| Car direction | Call direction | Call position | Serve? |
|---|---|---|---|
| UP | UP | ahead | **Yes** |
| UP | UP | behind | No |
| UP | DOWN | ahead | No — serve after reversing |
| UP | DOWN | behind | No |
| DOWN | UP | ahead | No — serve after reversing |
| DOWN | DOWN | ahead | **Yes** |
| IDLE | any | any | **Yes** |
| any | any | any, and car is full | **No** |

**The numbers**

| Parameter | Value |
|---|---|
| Floors | 10 (0..9) |
| Door cycle | 2 ticks each way |
| Doors hold open | 4 ticks |
| Ticks per floor | 1 |
| Car capacity | 800 kg, typically ~10 people |
| Full sweep | ~2 × floors + door time ≈ 30 ticks for 10 floors |
| Tick rate | ~10 Hz in the model; a real car is slower |
| `run_until_idle` cap | 10 000 ticks, and it raises rather than hanging |

**The invariants to assert**

1. The car **never moves with the doors not `CLOSED`**.
2. Every pending call has a stop on **exactly one** car.
3. After settling: **no pending calls and no stops**.
4. `floor` is always within `0..top_floor`.
5. A full car **never closes its doors** and **never accepts a new stop**.
6. One tick advances a car by **at most one floor**.

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "Which scheduling algorithm?" | "SCAN — it bounds the worst-case wait. Nearest-first starves." |
| "Why four door states?" | "Because a reversal out of `CLOSING` is a safety behaviour." |
| "An up car gets a down call" | "It does not answer it; it serves it after reversing. Otherwise it stops going the wrong way." |
| "Make it thread-safe" | "One `RLock` over presses and the whole tick. Pure computation, no I/O — unlike a cache, where the lock is the bottleneck." |
| "Now add a second lift" | "An `assign` policy. Nothing on the car changes — that is why the scheduler owned the decision from the start." |
| "Why not the nearest car?" | "It measures distance, not passenger time; a diverted car strands the people already in it." |
| "What if it never settles?" | "A bounded tick cap that raises with the pending calls and the car states attached." |
| "40 floors, 10 cars" | "Zoning, not a faster scheduler. The class design's limit is the single shared lock." |

---

## Cross-links

- The shared-state mechanics this file assumes: [Concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md)
- The starvation argument, made about a cache: [LRU and LFU cache](02-lru-lfu-cache.md) · the state-machine treatment: [Hotel management](03-hotel-management.md) · another lease-versus-sale distinction: [BookMyShow seat locking](04-bookmyshow-seat-locking.md) · and the classic first problem: [Parking lot](01-parking-lot.md)
- The class concepts behind this design: [OOP principles and SOLID](../02-oop-principles-and-solid.md) · [Design patterns](../03-design-patterns.md) · [UML and class diagrams](../05-uml-and-class-diagrams.md) · [The machine-coding framework](../06-machine-coding-interview-framework.md) · [What is LLD](../01-what-is-lld-and-how-to-answer.md)
- Where the class view ends: [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md) · [The five-step HLD framework](../../HLD/01-foundations/04-five-step-interview-framework.md)
- Other problems in this pillar: [Splitwise](06-splitwise.md) · [ATM machine](07-atm-machine.md) · [Library management](08-library-management.md)
