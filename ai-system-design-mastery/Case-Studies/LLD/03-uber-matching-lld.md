# Uber — Ride Hailing — LLD

> **Level.** This is the **LLD bridge** for the ride-hailing case study. The HLD in [`../HLD/06-uber-ride-hailing.md`](../HLD/06-uber-ride-hailing.md) decides the system-level answers: that location is ingested over Kafka at high frequency, that matching is a geo-index lookup, that the fare is computed server-side, that surge is per-cell. **None of that is repeated here.** This file is the layer below: the classes, the interfaces, the algorithms inside them, and the concurrency that makes them true.
>
> **Grounding note.** The reference repositories contain **no OOP/LLD material**, and the source transcripts cover ride-hailing at the system level without teaching a class design for it. Everything here — the geo index, the matching strategies, the offer lease, the trip state machine — is standard industry practice assembled for this problem. Where it aligns with a property the HLD file states, it cross-links rather than restates.
>
> **Technique.** The 12-section shape, the test discipline and the pattern vocabulary come from [`../../LLD/01-what-is-lld-and-how-to-answer.md`](../../LLD/01-what-is-lld-and-how-to-answer.md) and the worked problems in [`../../LLD/problems/`](../../LLD/problems/). Read [`../../LLD/problems/16-food-delivery.md`](../../LLD/problems/16-food-delivery.md) for the state-machine treatment this file deliberately *contrasts* with.

---

## TL;DR

> **A ride-hailing match is a lease on a driver, not an assignment, because the driver has not agreed yet — and the geo search that chooses who to ask must finish the ring it is looking at, or it will hand back the driver in the far corner of a cell while a closer one sits unvisited next door.**

Three sentences that carry it:

1. **The driver is not assigned; the driver is *offered*.** The offer is a CAS on the driver's status with an expiry, so eight riders cannot be offered the same driver and a driver who ignores the offer returns to the pool on their own.
2. **The spatial index is not a database lookup.** You cannot search only the cell containing the rider — the nearest driver may be ten metres the other side of a cell boundary — and you cannot stop as soon as you have `k` hits, because a closer driver may be in a ring you have not scanned yet.
3. **A trip's assignment *is* a state, unlike an order's.** In the food-delivery file the driver is orthogonal to kitchen progress; here the ride does not exist until a driver accepts, so `ASSIGNED` is a genuine state and the machine is linear.

---

## The 60-second answer

"Model four things. **`GeoIndex`** — I would use a geohash grid, cells at a precision chosen from the search radius, and the query expands ring by ring. Two rules make it correct: a ring is only usable when the whole ring is scanned, and I stop only when the k-th candidate is closer than the nearest possible cell in the next ring — which is why a radius search must always look at least one ring out. **`Driver`** with an availability *lease*, because a driver who loses signal never says goodbye and a stale driver in the index is a driver you offer rides to who is not there. **`MatchingStrategy`** as a strategy — nearest-first, highest-rated, and a batched one, because the objective is not distance: it is wait time plus driver idle time plus acceptance probability, and greedy on distance is not optimal for that. **The offer** as a lease with a TTL and a compare-and-swap on the driver, so the claim is atomic and an unanswered offer expires back into the pool. Then a table-driven **`TripStateMachine`** over `MATCHING → ASSIGNED → ARRIVING → IN_PROGRESS → COMPLETED`, with actors, so an illegal transition and an unauthorised one are the same failed lookup. Surge is a per-cell supply/demand ratio with a hysteresis band, because without the band the multiplier oscillates."

---

## 1. Requirement clarification

### Questions to ask, and the default to assume if the interviewer waves them through

| # | Question | Default to state and move on |
|---|---|---|
| 1 | Scope: matching only, or the whole trip lifecycle? | Matching, the trip state machine, and the fare — payments and driver onboarding are named and excluded |
| 2 | Is the driver assigned, or does the driver have to accept? | **The driver accepts.** This is the single most important assumption in the file |
| 3 | How fresh must a driver's location be? | A ping every 4 s while available; a 15 s lease; stale beyond that and the driver leaves the index |
| 4 | Exactly one driver per ride, or a pool of candidates? | One winner, but the offer is ranked so the next candidate can be tried on refusal |
| 5 | Can a rider cancel, and when? | Yes, until the trip starts. After `IN_PROGRESS` only support can, because the ride is happening |
| 6 | Is the fare fixed at request time or at completion? | Quoted at request, computed at completion, and the quote is what the rider was shown |
| 7 | Does surge change mid-trip? | No. The multiplier is captured at request time — changing it mid-ride is a fare dispute, which the HLD names |
| 8 | Multi-city? | Yes, and it changes nothing in the class model: a cell id is globally unique, so a city is a filter, not a shard you have to code for |
| 9 | How many candidates should the search return? | `k = 10`, ranked, so the offer can walk down the list without re-searching |
| 10 | What happens with no drivers? | The trip stays in `MATCHING` with a widening radius, then goes to `NO_DRIVERS_FOUND` — it does not silently fail |

### The four constraints that shape the design

1. **The driver has not agreed.** An assignment is a lie the moment the driver declines. Everything about matching is therefore an **offer with an expiry**, and the driver's availability is *leased* to the offer while it is outstanding.
2. **Space has no natural buckets.** A cell boundary is an artefact of the encoding, not a fact about geography, so any query that respects cell boundaries strictly will miss the driver standing across the street.
3. **The objective is not distance.** The nearest driver might be about to go off-shift, or might decline, or has been idle for 40 minutes while another has had four rides. Nearest-first is a *tie-breaker*, not the objective.
4. **Supply and demand move faster than the pricing decision.** Surge computed from a raw ratio every second oscillates, and a rider watching the multiplier flicker is a rider who does not book.

### Non-goals

- Payment processing, driver onboarding and background checks, the ETA model itself (assumed given as a function), and the mobile client. Named so the interviewer knows they were considered.

---

## 2. Core entities & responsibilities

| Entity | Responsibility — one line | Collaborators |
|---|---|---|
| `GeoPoint` | An immutable (lat, lon); knows haversine distance | — |
| `GeoIndex` | Map cells → drivers; answer "the k nearest available, within r" | `GeoPoint`, `Driver` |
| `MatchingStrategy` | Rank candidate drivers for a request | `RideRequest`, `Driver` |
| `Driver` | Identity, vehicle, rating, status, **availability lease** | `GeoIndex` |
| `DriverRegistry` | Owns drivers and the index; location pings and lease sweeps | `GeoIndex`, `Driver` |
| `RideRequest` | A rider's ask: pickup, dropoff, when, client key | `GeoPoint` |
| `Rider` | Identity, payment ref (excluded), active trip | `Trip` |
| **`Offer`** | **The lease on a driver: who, for which trip, until when** | `Driver`, `Trip` |
| `OfferService` | Atomically claim a driver; expire unanswered offers | `Offer`, `DriverRegistry` |
| `Trip` | The aggregate: the ride itself, its version, its history | `TripStateMachine` |
| `TripStateMachine` | `(state, action, actor) → state`, or refuse | — |
| `SurgeCalculator` | Per-cell multiplier with a hysteresis band | — |
| `FareEstimator` | Base + distance + time, times the captured surge, in minor units | — |
| `RideService` | The façade: request, dispatch, respond, advance, cancel | all of the above |

**The sentence to say out loud:** *"`Offer` is the class most people leave out, and it is the one that makes the design correct — it is what turns 'assign a driver' into 'lease a driver', and it is where the atomicity lives."*

### HLD component → its classes

| HLD component (in [`../HLD/06-uber-ride-hailing.md`](../HLD/06-uber-ride-hailing.md)) | Its classes here | The class-level question the HLD does not answer |
|---|---|---|
| Location ingest | `DriverRegistry.ping`, `Driver.location`, the lease | How is a driver who stopped pinging removed, and who notices? |
| The geo index | `GeoIndex`, `GeoPoint`, cell ring expansion | Which cells do you scan, and when are you allowed to stop? |
| Matching | `MatchingStrategy`, `OfferService`, `Offer` | Is a match an assignment or an offer, and what makes the claim atomic? |
| Driver dispatch / offers | `Offer` TTL, `OfferService.expire` | What happens to a driver who never answers? |
| Trip lifecycle | `Trip`, `TripState`, `TripStateMachine` | Which transitions exist, and who is allowed to make each? |
| Surge pricing | `SurgeCalculator` | How do you stop the multiplier oscillating, and when is the value captured? |
| Fare computation | `FareEstimator` | Integer or float, and where does rounding happen? |
| Double-assignment (§6.1) | the CAS in `OfferService.claim` | Two riders, one driver — what is the exact serialisation point? |
| Stale drivers (§6.2) | `Driver.available(now)`, `DriverRegistry.sweep` | A driver in the index who is not there |
| Empty market (§6.4) | `TripState.NO_DRIVERS_FOUND`, widening radius | How does "no drivers" surface, rather than hanging forever? |
| Surge oscillation (§6.9) | the hysteresis band in `SurgeCalculator` | What stops the multiplier flickering between two values? |

---

## 3. Class diagram (ASCII)

```
        ┌────────────────────┐
        │  GeoPoint «value»  │  lat, lon
        │  +distance_km()    │  haversine — the ONLY distance in the file
        └─────────┬──────────┘
                  │
   ┌──────────────┴───────────────────────────────────────────────┐
   │  GeoIndex                                                    │
   │  _cells: {geohash -> {driver}}   precision from the radius    │
   │  +insert(driver) / +remove(driver_id) / +move(driver)         │
   │  +candidates(center, r, k, now) -> [Driver]                   │
   │     ring 0, 1, 2, …                                           │
   │     STOP only when: |inside| >= k                             │
   │                     AND kth_dist <= ring * cell_km_min        │
   │     ← so ring 1 is ALWAYS scanned: the nearest driver may be  │
   │       ten metres across a cell boundary                       │
   └──────────────▲───────────────────────────────────────────────┘
                  │ indexes
   ┌──────────────┴──────────────────┐        ┌─────────────────────┐
   │  Driver                         │        │  DriverStatus       │
   │  id, vehicle, rating, trips     │───────►│  OFFLINE            │
   │  location: GeoPoint             │        │  AVAILABLE          │
   │  status: DriverStatus           │        │  OFFERED  ← leased  │
   │  lease_expiry: float  ← TTL     │        │  ON_TRIP            │
   │  +available(now) -> bool        │        └─────────────────────┘
   │  +touch(now, ttl)               │
   └─────────────────────────────────┘
                  ▲
                  │ holds + sweeps
   ┌──────────────┴──────────────────────┐
   │  DriverRegistry                     │
   │  +ping(driver_id, point, now)       │  location + lease refresh
   │  +sweep(now) -> [Driver]            │  expired leases leave the index
   └─────────────────────────────────────┘

   ┌──────────────────────────┐     ┌────────────────────────────────┐
   │  RideRequest             │     │  MatchingStrategy «interface»  │
   │  rider, pickup, dropoff  │     │  +rank(request, [Driver])      │
   │  requested_at, client_key│     ├────────────────────────────────┤
   └────────────┬─────────────┘     │  NearestFirst                  │
                │                   │  HighestRated                  │
                │                   │  BatchedOptimal  ← NOT distance│
                ▼                   │    score: ETA + idle + accept  │
   ┌────────────────────────────────┴───────────────────────────────┐
   │  RideService  (façade)                                         │
   │  request_ride(rider, pickup, dropoff, client_key) -> Trip      │
   │  dispatch(trip): candidates → rank → offer → ASSIGNED | keep   │
   │  respond(trip, driver, accept) -> bool                         │
   │  advance(trip, action, actor) -> TripState                     │
   └───┬───────────────┬──────────────────┬─────────────────────────┘
       │               │                  │
       ▼               ▼                  ▼
 ┌───────────┐  ┌──────────────┐  ┌──────────────────┐
 │  Offer    │  │  Trip        │  │  SurgeCalculator │
 │  driver   │  │  state       │  │  per-cell        │
 │  trip     │  │  version     │  │  demand/supply   │
 │  expires  │  │  history[]   │  │  + HYSTERESIS    │
 │  +expired │  │  surge_held  │  │  band: only move │
 │    (now)  │  │  fare_quoted │  │  past a threshold│
 └─────▲─────┘  └──────┬───────┘  └──────────────────┘
       │               │
       │        ┌──────┴──────────────────────────────────┐
       │        │  TripStateMachine (TABLE-DRIVEN)        │
       │        │  TRANSITIONS[(state, action)] ->        │
       │        │      (next_state, frozenset[actor])     │
       │        │  illegal and unauthorised are the SAME  │
       │        │  failed lookup                          │
       │        └─────────────────────────────────────────┘
       │
  claim() is a CAS on driver.status under the registry lock:
  AVAILABLE -> OFFERED, exactly once, expiry = now + TTL

  MATCHING ──assign──► ASSIGNED ──arrived──► ARRIVING ──start──► IN_PROGRESS
     │  │                  │                    │                    │
     │  └─give_up──► NO_DRIVERS_FOUND           │                    ▼
     └─cancel──► CANCELLED_BY_RIDER ◄───────────┘              COMPLETED
                       ▲
                       └── rider may cancel until the trip starts
```

**Read it this way:** the index decides *who is eligible*, the strategy decides *who to ask*, the offer decides *who is claimed*, and the trip machine decides *what has happened*. Four separate questions, four separate classes — collapsing any two of them is where this design usually goes wrong.

---

## 4. Interfaces / abstract classes / enums

```python
class DriverStatus(Enum):
    OFFLINE = "offline"
    AVAILABLE = "available"
    OFFERED = "offered"        # leased to an outstanding offer
    ON_TRIP = "on_trip"


class TripState(Enum):
    MATCHING = "matching"              # searching; an offer may be outstanding
    ASSIGNED = "assigned"              # a driver accepted; car is coming
    ARRIVING = "arriving"              # driver is at the pickup point
    IN_PROGRESS = "in_progress"        # rider is in the car
    COMPLETED = "completed"
    CANCELLED_BY_RIDER = "cancelled_by_rider"
    CANCELLED_BY_DRIVER = "cancelled_by_driver"
    NO_DRIVERS_FOUND = "no_drivers_found"


class TripAction(Enum):
    ASSIGN = "assign"                  # a driver accepted an offer
    GIVE_UP = "give_up"                # search timed out
    DRIVER_ARRIVED = "driver_arrived"
    START = "start"
    COMPLETE = "complete"
    CANCEL_RIDER = "cancel_rider"
    CANCEL_DRIVER = "cancel_driver"
    CANCEL_SYSTEM = "cancel_system"    # support, or a payment failure


class TripActor(Enum):
    RIDER = "rider"
    DRIVER = "driver"
    SYSTEM = "system"


class MatchError(Exception):
    """Base for matching failures. Distinct from IllegalTransition: this is
    'the world did not cooperate', that one is 'the code asked for something
    the machine does not permit'."""


class NoDriversAvailable(MatchError):
    pass


class OfferLost(MatchError):
    """Someone else claimed the driver first — the CAS failed. Expected under
    contention, not an error the caller should log as a fault."""


class OfferExpired(MatchError):
    pass


class IllegalTransition(Exception):
    def __init__(self, state: TripState, action: TripAction,
                 actor: TripActor) -> None:
        super().__init__(f"{actor.value} cannot {action.value} from {state.value}")
        self.state = state
        self.action = action
        self.actor = actor


class TripConflict(Exception):
    """The trip moved under us — the version compare-and-swap failed."""


class MatchingStrategy(ABC):
    @abstractmethod
    def rank(self, request: "RideRequest", candidates: list["Driver"]) -> list["Driver"]:
        """Best first. Ranking, not selection: the caller walks the list when a
        driver declines, and re-ranking would re-search the index for nothing."""


class SurgeProvider(ABC):
    @abstractmethod
    def multiplier(self, cell: str, now: float) -> float:
        """The multiplier to CHARGE. Never below 1.0, never above the ceiling."""
```

**Three contract notes:**

- **`rank` returns a list, not a winner.** The offer may be declined, and the second-best candidate must be tried without a second index search.
- **`TripAction.ASSIGN` is a real transition, unlike the food-delivery file.** There, driver assignment was deliberately *not* a state because it is orthogonal to kitchen progress. Here the trip cannot exist without a driver, so it is.
- **`OfferLost` is not an error.** It is the expected outcome of contention and the caller is meant to catch it and try the next candidate.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** In Java this is a `record` for `GeoPoint`, an enum with a constructor for the transition table, and a `ConcurrentHashMap` of cells. In Go, a `sync.RWMutex` around the index and a `(TripState, error)` return from `advance`. The **ring-expansion rule, the offer lease, the strategy seam and the transition table** are the design; the syntax is not.

```python
"""uber_matching_lld.py — the classes inside the HLD's boxes.

Design summary
--------------
GeoIndex        : a geohash grid. The query expands ring by ring and stops only
                  when the k-th candidate is closer than the nearest possible
                  cell in the next ring — so ring 1 is ALWAYS scanned. A cell
                  boundary is an artefact of the encoding, not a fact about
                  geography, and the nearest driver is often across the street
                  from where the encoding says they are.
Driver lease    : a driver is available only while the lease holds. A driver
                  who loses signal never says goodbye, and a stale entry means
                  offering rides to somebody who is not there.
Offer           : a LEASE on a driver with a TTL, claimed by a CAS on the
                  driver's status. Eight riders cannot be offered one driver,
                  and a driver who ignores the offer releases themselves.
MatchingStrategy: nearest-first is a tie-breaker, not an objective. The real
                  objective is wait time + driver idle time + acceptance
                  probability, and greedy on distance is not optimal for it —
                  the demo shows a 2x2 case where greedy scores 10 and the
                  batched assignment scores 18.
TripStateMachine: table-driven, actor-aware. An illegal transition and an
                  unauthorised one are the SAME failed lookup.
SurgeCalculator : per-cell, with a HYSTERESIS BAND. Without it the multiplier
                  oscillates between two values as supply flickers.
"""
from __future__ import annotations

import itertools
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Optional

GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------
# 0. The failure vocabulary (declared in §4 as well, repeated so this block
#    runs standalone)
# --------------------------------------------------------------------------

class MatchError(Exception):
    """Base for matching failures. Distinct from IllegalTransition: this is
    'the world did not cooperate', that one is 'the code asked for something
    the machine does not permit'."""


class NoDriversAvailable(MatchError):
    pass


class OfferLost(MatchError):
    """Someone else claimed the driver first — the CAS failed. Expected under
    contention, NOT an error the caller should log as a fault: the dispatch
    loop catches it and moves to the next candidate."""


class OfferExpired(MatchError):
    pass


# --------------------------------------------------------------------------
# 1. Geometry — the only place distance is computed
# --------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class GeoPoint:
    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"latitude out of range: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"longitude out of range: {self.lon}")

    def distance_km(self, other: "GeoPoint") -> float:
        """Haversine. Every distance in this file goes through here, so there is
        exactly one definition of 'near' — the alternative, each caller doing
        its own subtraction, produces a design where the index and the fare
        disagree about how far apart two points are."""
        lat1, lon1 = math.radians(self.lat), math.radians(self.lon)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def geohash_encode(point: GeoPoint, precision: int = 6) -> str:
    """Standard geohash: bisect the lat/lon ranges, interleave the bits, emit
    base-32. Written out rather than imported so the cell-boundary discussion
    below has something to point at."""
    if precision < 1:
        raise ValueError("precision must be at least 1")
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    bits = (16, 8, 4, 2, 1)
    bit = 0
    ch = 0
    even = True
    out: list[str] = []
    while len(out) < precision:
        if even:
            mid = (lon_min + lon_max) / 2
            if point.lon >= mid:
                ch |= bits[bit]
                lon_min = mid
            else:
                lon_max = mid
        else:
            mid = (lat_min + lat_max) / 2
            if point.lat >= mid:
                ch |= bits[bit]
                lat_min = mid
            else:
                lat_max = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(GEOHASH_ALPHABET[ch])
            bit = 0
            ch = 0
    return "".join(out)


def geohash_bounds(code: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Decode back to the cell's box: ((lat_min, lat_max), (lon_min, lon_max)).
    Needed because ring expansion is done by stepping in DEGREES, which is
    simpler and more obviously correct than the string-manipulation neighbour
    trick."""
    if not code:
        raise ValueError("an empty geohash has no bounds")
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    even = True
    for char in code:
        index = GEOHASH_ALPHABET.index(char)
        for bit in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_min + lon_max) / 2
                if index & bit:
                    lon_min = mid
                else:
                    lon_max = mid
            else:
                mid = (lat_min + lat_max) / 2
                if index & bit:
                    lat_min = mid
                else:
                    lat_max = mid
            even = not even
    return (lat_min, lat_max), (lon_min, lon_max)


def cell_ring(center_code: str, ring: int) -> list[str]:
    """Every cell at Chebyshev distance exactly `ring` from the centre cell.

    Ring 0 is the centre cell alone; ring 1 is the eight neighbours. Computed by
    decoding the centre, stepping by one cell width in degrees, and re-encoding
    — which is why `geohash_bounds` exists."""
    if ring < 0:
        raise ValueError("a ring cannot be negative")
    if ring == 0:
        return [center_code]
    (lat_min, lat_max), (lon_min, lon_max) = geohash_bounds(center_code)
    lat_mid = (lat_min + lat_max) / 2
    lon_mid = (lon_min + lon_max) / 2
    dlat = lat_max - lat_min
    dlon = lon_max - lon_min
    codes: list[str] = []
    for i in range(-ring, ring + 1):
        for j in range(-ring, ring + 1):
            if max(abs(i), abs(j)) != ring:
                continue
            lat = min(89.999999, max(-89.999999, lat_mid + i * dlat))
            lon = ((lon_mid + j * dlon + 180.0) % 360.0) - 180.0
            codes.append(geohash_encode(GeoPoint(lat, lon), len(center_code)))
    return codes


def cell_dimensions_km(code: str) -> tuple[float, float]:
    """(height, width) of the cell in km. The WIDTH shrinks with latitude, and
    the search uses min(height, width) so the stopping rule stays conservative
    everywhere on Earth rather than only near the equator."""
    (lat_min, lat_max), (lon_min, lon_max) = geohash_bounds(code)
    height = (lat_max - lat_min) * 111.32
    width = (lon_max - lon_min) * 111.32 * math.cos(math.radians(lat_min))
    return height, abs(width)


# --------------------------------------------------------------------------
# 2. Driver
# --------------------------------------------------------------------------

class DriverStatus(Enum):
    OFFLINE = "offline"
    AVAILABLE = "available"
    OFFERED = "offered"
    ON_TRIP = "on_trip"


@dataclass(eq=False)
class Driver:
    """`eq=False` is deliberate and load-bearing. A driver is an ENTITY, not a
    value: two drivers with identical fields are still two drivers, and the
    index holds them in sets, which requires identity hashing. The default
    dataclass `__eq__` would make them interchangeable by value AND would make
    the type unhashable, since a mutable dataclass sets `__hash__ = None`."""

    id: str
    location: GeoPoint
    vehicle: str = "sedan"
    rating: float = 4.9
    trips_completed: int = 0
    idle_since: float = 0.0
    status: DriverStatus = DriverStatus.AVAILABLE
    lease_expiry: float = 0.0

    def available(self, now: float) -> bool:
        """AVAILABLE *and* inside the lease. The two are separate conditions on
        purpose: status is what the system decided, the lease is evidence the
        driver is still there. A driver whose app crashed has status AVAILABLE
        and an expired lease, and only the second one is true."""
        return self.status is DriverStatus.AVAILABLE and now < self.lease_expiry

    def touch(self, point: GeoPoint, now: float, ttl: float) -> None:
        """A location ping: move the point and refresh the lease.

        The status is deliberately UNTOUCHED. A ping from a driver who is
        ON_TRIP must not put them back into the available pool, and a ping from
        a driver who is OFFERED must not cancel the offer — presence of a
        heartbeat says the driver exists, not that they are free."""
        self.location = point
        self.lease_expiry = now + ttl

    def __str__(self) -> str:
        return f"{self.id} [{self.status.value}] {self.location.lat:.4f},{self.location.lon:.4f}"


# --------------------------------------------------------------------------
# 3. The geo index
# --------------------------------------------------------------------------

class GeoIndex:
    """A geohash grid of cells holding drivers.

    The two rules that make the search correct are BOTH in `candidates`, and
    both are about when you are allowed to stop.
    """

    # radius -> geohash precision. A cell is roughly 4.9 km at 5, 1.2 km at 6,
    # 150 m at 7. The precision is chosen so a handful of rings covers the
    # radius: too coarse and every cell holds the whole city, too fine and the
    # ring count explodes.
    PRECISION_BY_RADIUS = ((0.5, 7), (3.0, 6), (20.0, 5))

    def __init__(self, default_precision: int = 6) -> None:
        self._cells: dict[str, set[Driver]] = {}
        self._where: dict[str, str] = {}          # driver_id -> cell
        self._lock = threading.RLock()
        self.default_precision = default_precision

    def precision_for(self, radius_km: float) -> int:
        for limit, precision in self.PRECISION_BY_RADIUS:
            if radius_km <= limit:
                return precision
        return 4

    def insert(self, driver: Driver, precision: Optional[int] = None) -> str:
        p = self.default_precision if precision is None else precision
        code = geohash_encode(driver.location, p)
        with self._lock:
            previous = self._where.get(driver.id)
            if previous is not None:
                bucket = self._cells.get(previous)
                if bucket is not None:
                    bucket.discard(driver)
            self._cells.setdefault(code, set()).add(driver)
            self._where[driver.id] = code
        return code

    def remove(self, driver_id: str) -> bool:
        with self._lock:
            code = self._where.pop(driver_id, None)
            if code is None:
                return False
            bucket = self._cells.get(code)
            if bucket is not None:
                for driver in list(bucket):
                    if driver.id == driver_id:
                        bucket.discard(driver)
                if not bucket:
                    del self._cells[code]
            return True

    def move(self, driver: Driver) -> None:
        """A driver crossing a cell boundary must be re-filed, or the index
        keeps them where they were — the single most common bug in a geo index,
        and it presents as 'the search finds drivers who left ten minutes ago'."""
        self.insert(driver)

    def size(self) -> int:
        with self._lock:
            return len(self._where)

    def candidates(self, center: GeoPoint, radius_km: float, k: int,
                   now: float) -> list[Driver]:
        """The k nearest AVAILABLE drivers within `radius_km`, best first.

        RING 1 IS ALWAYS SCANNED. The cell containing the rider is an artefact
        of the encoding: two points ten metres apart can be in different cells,
        so stopping at ring 0 would systematically miss the closest driver
        whenever the rider happens to be near a boundary.

        THE STOPPING RULE. After finishing ring r, we may stop when we already
        hold k candidates inside the radius AND the k-th of them is closer than
        the nearest point any unscanned cell could be. Cells in ring r+1 start
        at least r * cell_km_min away, so that is the threshold. Stopping as
        soon as SOME cell in a ring held a hit would return the far corner of
        the ring while an unvisited cell in the same ring held a closer driver
        — which is the bug this rule exists to prevent.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if radius_km <= 0:
            raise ValueError("a search radius must be positive")
        precision = self.precision_for(radius_km)
        base = geohash_encode(center, precision)
        cell_km_min = min(cell_dimensions_km(base))

        with self._lock:
            seen: set[str] = set()
            inside: list[Driver] = []
            ring = 0
            while True:
                for code in cell_ring(base, ring):
                    if code in seen:
                        continue
                    seen.add(code)
                    for driver in self._cells.get(code, ()):
                        if not driver.available(now):
                            continue
                        if driver.location.distance_km(center) <= radius_km:
                            inside.append(driver)
                if len(inside) >= k and ring >= 1:
                    inside.sort(key=lambda d: (d.location.distance_km(center), d.id))
                    kth = inside[k - 1].location.distance_km(center)
                    if kth <= ring * cell_km_min:
                        break
                ring += 1
                if ring > 12:                      # a radius wider than any city
                    break

        inside.sort(key=lambda d: (d.location.distance_km(center), d.id))
        return inside[:k]

    def all_drivers(self) -> list[Driver]:
        """For the brute-force comparison the tests use to prove the ring
        expansion found the true nearest. Not a query path."""
        with self._lock:
            found: list[Driver] = []
            for bucket in self._cells.values():
                found.extend(bucket)
        return found


class DriverRegistry:
    """Owns the drivers and keeps the index honest. Every location ping goes
    through here, so the index can never drift from the driver's real position
    without somebody having bypassed the registry."""

    def __init__(self, index: Optional[GeoIndex] = None,
                 clock: Callable[[], float] = time.monotonic,
                 lease_ttl: float = 15.0) -> None:
        self.index = index or GeoIndex()
        self.clock = clock
        self.lease_ttl = lease_ttl
        self._drivers: dict[str, Driver] = {}
        self._lock = threading.RLock()

    def add(self, driver: Driver) -> Driver:
        with self._lock:
            driver.lease_expiry = self.clock() + self.lease_ttl
            self._drivers[driver.id] = driver
            self.index.insert(driver)
        return driver

    def get(self, driver_id: str) -> Driver:
        with self._lock:
            driver = self._drivers.get(driver_id)
        if driver is None:
            raise KeyError(f"no such driver: {driver_id}")
        return driver

    def ping(self, driver_id: str, point: GeoPoint) -> Driver:
        """A location update. Refreshes the lease AND re-files the driver if the
        point is in a different cell."""
        now = self.clock()
        with self._lock:
            driver = self._drivers.get(driver_id)
            if driver is None:
                raise KeyError(f"no such driver: {driver_id}")
            driver.location = point
            driver.lease_expiry = now + self.lease_ttl
            self.index.move(driver)
            return driver

    def sweep(self) -> list[Driver]:
        """Remove drivers whose lease has lapsed. This is what makes the index
        self-healing: nobody has to detect a crashed app, because a crashed app
        stops pinging and the lease does the detection."""
        now = self.clock()
        removed: list[Driver] = []
        with self._lock:
            for driver in list(self._drivers.values()):
                if driver.status is DriverStatus.OFFLINE:
                    continue
                if now >= driver.lease_expiry:
                    driver.status = DriverStatus.OFFLINE
                    self.index.remove(driver.id)
                    removed.append(driver)
        return removed

    def set_status(self, driver_id: str, status: DriverStatus,
                   now: Optional[float] = None) -> Driver:
        moment = self.clock() if now is None else now
        with self._lock:
            driver = self._drivers.get(driver_id)
            if driver is None:
                raise KeyError(f"no such driver: {driver_id}")
            driver.status = status
            if status is DriverStatus.AVAILABLE:
                driver.idle_since = moment
                driver.lease_expiry = moment + self.lease_ttl
                self.index.insert(driver)
            elif status in (DriverStatus.OFFLINE,):
                self.index.remove(driver.id)
            return driver

    def counts(self) -> dict[DriverStatus, int]:
        with self._lock:
            tally: dict[DriverStatus, int] = {s: 0 for s in DriverStatus}
            for driver in self._drivers.values():
                tally[driver.status] += 1
        return tally

    # -- the CAS, and the only two transitions that touch a driver's status ---

    def try_claim(self, driver_id: str) -> bool:
        """AVAILABLE -> OFFERED, atomically. THIS is the serialisation point for
        double-assignment: eight riders asking for the same driver in the same
        millisecond produce exactly one winner, because the check and the write
        are one operation under one lock.

        A read-then-write version — `if driver.status is AVAILABLE:` outside the
        lock, then set it — is the classic bug, and it is invisible in a test
        with no concurrency.
        """
        with self._lock:
            driver = self._drivers.get(driver_id)
            if driver is None or driver.status is not DriverStatus.AVAILABLE:
                return False
            driver.status = DriverStatus.OFFERED
            # An OFFERED driver is not a candidate for anybody else, so they
            # leave the index while the offer is outstanding. Releasing puts
            # them back — which is also why a declined or expired offer costs
            # nothing but a re-insert.
            self.index.remove(driver.id)
            return True

    def release(self, driver_id: str, now: Optional[float] = None) -> bool:
        """OFFERED -> AVAILABLE. Called when an offer is declined, expires, or
        the trip is cancelled before it starts."""
        moment = self.clock() if now is None else now
        with self._lock:
            driver = self._drivers.get(driver_id)
            if driver is None or driver.status is not DriverStatus.OFFERED:
                return False
            driver.status = DriverStatus.AVAILABLE
            driver.idle_since = moment
            driver.lease_expiry = moment + self.lease_ttl
            self.index.insert(driver)
            return True

    def begin_trip(self, driver_id: str, now: Optional[float] = None) -> bool:
        """OFFERED -> ON_TRIP, atomically. A second CAS, and it is what makes a
        double-tap on Accept harmless: the first tap wins, the second finds the
        driver no longer OFFERED and is rejected."""
        moment = self.clock() if now is None else now
        with self._lock:
            driver = self._drivers.get(driver_id)
            if driver is None or driver.status is not DriverStatus.OFFERED:
                return False
            driver.status = DriverStatus.ON_TRIP
            driver.lease_expiry = moment + self.lease_ttl
            self.index.remove(driver.id)
            return True

    def finish_trip(self, driver_id: str, now: Optional[float] = None) -> bool:
        """ON_TRIP -> AVAILABLE, with the trip counted."""
        moment = self.clock() if now is None else now
        with self._lock:
            driver = self._drivers.get(driver_id)
            if driver is None or driver.status is not DriverStatus.ON_TRIP:
                return False
            driver.status = DriverStatus.AVAILABLE
            driver.trips_completed += 1
            driver.idle_since = moment
            driver.lease_expiry = moment + self.lease_ttl
            self.index.insert(driver)
            return True


# --------------------------------------------------------------------------
# 4. Requests and the matching strategies
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RideRequest:
    id: str
    rider_id: str
    pickup: GeoPoint
    dropoff: GeoPoint
    requested_at: float
    client_key: Optional[str] = None

    def straight_line_km(self) -> float:
        return self.pickup.distance_km(self.dropoff)


class MatchingStrategy(ABC):
    @abstractmethod
    def rank(self, request: RideRequest, candidates: list[Driver]) -> list[Driver]:
        """Best first. A LIST, not a winner: the offer may be declined and the
        next candidate must be tried without a second index search."""


class NearestFirst(MatchingStrategy):
    """Distance, with rating as a tie-breaker. The default, and the right
    default: it minimises pickup time, which is what the rider experiences."""

    def rank(self, request: RideRequest, candidates: list[Driver]) -> list[Driver]:
        return sorted(candidates,
                      key=lambda d: (d.location.distance_km(request.pickup),
                                     -d.rating, d.id))


class HighestRated(MatchingStrategy):
    """Rating first, distance second. A product decision for a premium tier,
    and it is here to show the seam is real: the index produces candidates and
    knows nothing about how they are ordered."""

    def rank(self, request: RideRequest, candidates: list[Driver]) -> list[Driver]:
        return sorted(candidates,
                      key=lambda d: (-d.rating,
                                     d.location.distance_km(request.pickup),
                                     d.id))


class BatchedOptimal(MatchingStrategy):
    """Ranking by a SCORE that is not distance.

    The objective for a marketplace is wait time + driver idle time + the
    probability the driver accepts. Nearest-first is a tie-breaker within that,
    not the objective itself, and greedy on it is provably not optimal for a
    general score.

    THE DEMO CASE. Two riders, two drivers, and a score matrix where
    (r1,d1)=10, (r1,d2)=9, (r2,d1)=9, (r2,d2)=0. Greedy rider-by-rider gives
    r1->d1 and r2->d2 for a total of 10. The batched assignment gives r1->d2
    and r2->d1 for 18 — nearly twice the score from the same two drivers. That
    is not a contrived matrix: it is what happens whenever the driver who is
    best for one rider is also the only decent option for another.
    """

    def __init__(self, score: Callable[[RideRequest, Driver], float],
                 batch_size: int = 6) -> None:
        self.score = score
        self.batch_size = batch_size
        self.batches_assigned = 0

    def rank(self, request: RideRequest, candidates: list[Driver]) -> list[Driver]:
        return sorted(candidates,
                      key=lambda d: (-self.score(request, d),
                                     d.location.distance_km(request.pickup), d.id))

    def greedy_batch(self, requests: list[RideRequest],
                     candidate_lists: list[list[Driver]]
                     ) -> list[tuple[RideRequest, Optional[Driver]]]:
        """Rider by rider, best available. The naive version, kept in the file
        because the contrast is the argument for batching."""
        taken: set[str] = set()
        out: list[tuple[RideRequest, Optional[Driver]]] = []
        for index, request in enumerate(requests):
            best: Optional[Driver] = None
            best_score = float("-inf")
            for driver in candidate_lists[index]:
                if driver.id in taken:
                    continue
                value = self.score(request, driver)
                if value > best_score:
                    best, best_score = driver, value
            if best is not None:
                taken.add(best.id)
            out.append((request, best))
        return out

    def assign_batch(self, requests: list[RideRequest],
                     candidate_lists: list[list[Driver]]
                     ) -> list[tuple[RideRequest, Optional[Driver]]]:
        """Exhaustive for small batches, greedy beyond `batch_size`.

        Exhaustive is over PERMUTATIONS of the candidate pool taken
        len(requests) at a time — the assignment problem, solved by brute force
        because the batch is bounded. At five requests over ten candidates that
        is 30,240 combinations, which is fine in-process and is not fine at
        fifty, which is exactly why the batch size is capped and the cap is
        enforced here rather than assumed by the caller.
        """
        self.batches_assigned += 1
        if len(requests) > self.batch_size:
            return self.greedy_batch(requests, candidate_lists)

        allowed: list[set[str]] = [{d.id for d in c} for c in candidate_lists]
        by_id: dict[str, Driver] = {}
        for candidates in candidate_lists:
            for driver in candidates:
                by_id[driver.id] = driver
        pool = sorted(by_id.values(), key=lambda d: d.id)
        if len(pool) < len(requests):
            return self.greedy_batch(requests, candidate_lists)

        best_total = float("-inf")
        best_combo: Optional[tuple[Driver, ...]] = None
        for combo in itertools.permutations(pool, len(requests)):
            total = 0.0
            feasible = True
            for index, driver in enumerate(combo):
                if driver.id not in allowed[index]:
                    feasible = False
                    break
                total += self.score(requests[index], driver)
            if feasible and total > best_total:
                best_total, best_combo = total, combo
        if best_combo is None:
            return self.greedy_batch(requests, candidate_lists)
        return list(zip(requests, best_combo))


def default_score(request: RideRequest, driver: Driver, now: float) -> float:
    """The score the demo uses: proximity, plus a bonus for a driver who has
    been idle, plus a small term for rating. A free function rather than a
    method so a real deployment can swap in a model without touching the
    strategy."""
    distance = driver.location.distance_km(request.pickup)
    idle_seconds = max(0.0, now - driver.idle_since)
    idle_bonus = min(idle_seconds / 600.0, 1.0)
    return 1.0 / (1.0 + distance) + 0.3 * idle_bonus + 0.1 * (driver.rating - 4.0)


# --------------------------------------------------------------------------
# 5. The offer — a lease on a driver
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Offer:
    id: str
    trip_id: str
    rider_id: str
    driver_id: str
    created_at: float
    expires_at: float
    rank: int = 0

    def expired(self, now: float) -> bool:
        return now >= self.expires_at

    def seconds_left(self, now: float) -> float:
        return max(0.0, self.expires_at - now)


class OfferService:
    """Turns 'assign a driver' into 'lease a driver'.

    Every offer is a claim with an expiry, and the expiry is what makes the
    design self-healing: a driver who never answers, or an app that crashes
    while the offer is on screen, needs no timeout handler anywhere else,
    because the offer lapses and the driver is released.
    """

    def __init__(self, registry: DriverRegistry, ttl: float = 10.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if ttl <= 0:
            raise ValueError("an offer must have a positive ttl")
        self.registry = registry
        self.ttl = ttl
        self.clock = clock
        self._outstanding: dict[str, Offer] = {}      # offer_id -> Offer
        self._by_driver: dict[str, str] = {}          # driver_id -> offer_id
        self._lock = threading.RLock()
        self._counter = 0

    def claim(self, trip_id: str, rider_id: str, driver: Driver,
              rank: int = 0, now: Optional[float] = None) -> Offer:
        """Claim a driver for a trip, or raise OfferLost.

        The registry CAS runs FIRST, so a driver who is already claimed is
        rejected before an offer object exists. Creating the offer first and
        then discovering the driver is taken would leave an orphan offer to
        clean up — and in a race, several.
        """
        moment = self.clock() if now is None else now
        if not self.registry.try_claim(driver.id):
            raise OfferLost(f"{driver.id} was claimed by somebody else")
        with self._lock:
            self._counter += 1
            offer = Offer(f"offer-{self._counter:06d}", trip_id, rider_id,
                          driver.id, moment, moment + self.ttl, rank)
            self._outstanding[offer.id] = offer
            self._by_driver[driver.id] = offer.id
            return offer

    def get_for_trip(self, trip_id: str) -> Optional[Offer]:
        with self._lock:
            for offer in self._outstanding.values():
                if offer.trip_id == trip_id:
                    return offer
        return None

    def respond(self, offer: Offer, accept: bool,
                now: Optional[float] = None) -> bool:
        """Consume the offer. Declining and expiring both release the driver;
        accepting moves them to ON_TRIP via a second CAS, which is what makes a
        double tap on Accept harmless."""
        moment = self.clock() if now is None else now
        with self._lock:
            live = self._outstanding.get(offer.id)
            if live is None:
                raise OfferLost(f"offer {offer.id} is no longer outstanding")
            if live.expired(moment):
                del self._outstanding[offer.id]
                self._by_driver.pop(offer.driver_id, None)
                self.registry.release(offer.driver_id, moment)
                raise OfferExpired(f"offer {offer.id} expired at {offer.expires_at}")
            if not accept:
                del self._outstanding[offer.id]
                self._by_driver.pop(offer.driver_id, None)
                self.registry.release(offer.driver_id, moment)
                return False
            if not self.registry.begin_trip(offer.driver_id, moment):
                # A duplicate Accept, or the driver was released underneath us.
                raise OfferLost(f"{offer.driver_id} is no longer offered")
            del self._outstanding[offer.id]
            self._by_driver.pop(offer.driver_id, None)
            return True

    def expire_outstanding(self, now: Optional[float] = None) -> list[Offer]:
        """The sweep. Called on a timer; returns what it released so the caller
        can re-dispatch those trips."""
        moment = self.clock() if now is None else now
        lapsed: list[Offer] = []
        with self._lock:
            for offer in list(self._outstanding.values()):
                if offer.expired(moment):
                    del self._outstanding[offer.id]
                    self._by_driver.pop(offer.driver_id, None)
                    self.registry.release(offer.driver_id, moment)
                    lapsed.append(offer)
        return lapsed

    def cancel_for_trip(self, trip_id: str,
                        now: Optional[float] = None) -> Optional[Offer]:
        """Release whatever offer this trip is holding, IMMEDIATELY.

        Distinct from `expire_outstanding`, which only releases offers whose TTL
        has already passed, and the distinction is not cosmetic. A trip that is
        cancelled while its offer is still live has no `driver_id` — it was
        never ASSIGNED — so a cancellation path that only looks at
        `trip.driver_id` releases nobody, and the driver sits in OFFERED with no
        trip attached until the TTL happens to lapse. For those seconds they are
        invisible to every other rider, which is exactly the supply the surge
        calculator is about to raise the price over.
        """
        moment = self.clock() if now is None else now
        with self._lock:
            for offer in list(self._outstanding.values()):
                if offer.trip_id == trip_id:
                    del self._outstanding[offer.id]
                    self._by_driver.pop(offer.driver_id, None)
                    self.registry.release(offer.driver_id, moment)
                    return offer
        return None

    def outstanding_count(self) -> int:
        with self._lock:
            return len(self._outstanding)


# --------------------------------------------------------------------------
# 6. The trip state machine — table-driven and actor-aware
# --------------------------------------------------------------------------

# The three enums are declared in §4 as well; they are repeated here so this
# block runs standalone.

class TripState(Enum):
    MATCHING = "matching"              # searching; an offer may be outstanding
    ASSIGNED = "assigned"              # a driver accepted; the car is coming
    ARRIVING = "arriving"              # the driver is at the pickup point
    IN_PROGRESS = "in_progress"        # the rider is in the car
    COMPLETED = "completed"
    CANCELLED_BY_RIDER = "cancelled_by_rider"
    CANCELLED_BY_DRIVER = "cancelled_by_driver"
    NO_DRIVERS_FOUND = "no_drivers_found"


class TripAction(Enum):
    ASSIGN = "assign"                  # a driver accepted an offer
    GIVE_UP = "give_up"                # the search timed out
    DRIVER_ARRIVED = "driver_arrived"
    START = "start"
    COMPLETE = "complete"
    CANCEL_RIDER = "cancel_rider"
    CANCEL_DRIVER = "cancel_driver"
    CANCEL_SYSTEM = "cancel_system"    # support, or a payment failure


class TripActor(Enum):
    RIDER = "rider"
    DRIVER = "driver"
    SYSTEM = "system"


TERMINAL_STATES = frozenset({
    TripState.COMPLETED,
    TripState.CANCELLED_BY_RIDER,
    TripState.CANCELLED_BY_DRIVER,
    TripState.NO_DRIVERS_FOUND,
})

# (state, action) -> (next state, who may do it)
#
# TWO THINGS TO NOTICE.
#
# 1. ASSIGN is a real transition, and it is the SYSTEM that performs it even
#    though the DRIVER's acceptance is what causes it. The driver's tap is not
#    a state transition; it is an input to the service, which then applies one.
#    Modelling it the other way — the driver mutating the trip directly — puts
#    the authorisation question in the wrong place.
#
# 2. A rider CANNOT cancel once the trip is IN_PROGRESS. There is no
#    (IN_PROGRESS, CANCEL_RIDER) row at all, so the request fails as an illegal
#    PAIR rather than as a permission error — which is the honest model,
#    because a ride already under way is not something the rider can undo. The
#    support path is CANCEL_SYSTEM, restricted to SYSTEM, and it attributes the
#    outcome to the rider because that is who asked.
TRANSITIONS: dict[tuple[TripState, TripAction],
                  tuple[TripState, frozenset[TripActor]]] = {
    (TripState.MATCHING, TripAction.ASSIGN):
        (TripState.ASSIGNED, frozenset({TripActor.SYSTEM})),
    (TripState.MATCHING, TripAction.GIVE_UP):
        (TripState.NO_DRIVERS_FOUND, frozenset({TripActor.SYSTEM})),
    (TripState.MATCHING, TripAction.CANCEL_RIDER):
        (TripState.CANCELLED_BY_RIDER, frozenset({TripActor.RIDER, TripActor.SYSTEM})),

    (TripState.ASSIGNED, TripAction.DRIVER_ARRIVED):
        (TripState.ARRIVING, frozenset({TripActor.DRIVER, TripActor.SYSTEM})),
    (TripState.ASSIGNED, TripAction.CANCEL_RIDER):
        (TripState.CANCELLED_BY_RIDER, frozenset({TripActor.RIDER, TripActor.SYSTEM})),
    (TripState.ASSIGNED, TripAction.CANCEL_DRIVER):
        (TripState.CANCELLED_BY_DRIVER, frozenset({TripActor.DRIVER, TripActor.SYSTEM})),

    (TripState.ARRIVING, TripAction.START):
        (TripState.IN_PROGRESS, frozenset({TripActor.DRIVER, TripActor.SYSTEM})),
    (TripState.ARRIVING, TripAction.CANCEL_RIDER):
        (TripState.CANCELLED_BY_RIDER, frozenset({TripActor.RIDER, TripActor.SYSTEM})),
    (TripState.ARRIVING, TripAction.CANCEL_DRIVER):
        (TripState.CANCELLED_BY_DRIVER, frozenset({TripActor.DRIVER, TripActor.SYSTEM})),

    (TripState.IN_PROGRESS, TripAction.COMPLETE):
        (TripState.COMPLETED, frozenset({TripActor.DRIVER, TripActor.SYSTEM})),
    (TripState.IN_PROGRESS, TripAction.CANCEL_SYSTEM):
        (TripState.CANCELLED_BY_RIDER, frozenset({TripActor.SYSTEM})),
}


class IllegalTransition(Exception):
    """One exception for two failures, with the reason attached.

    An illegal pair — (COMPLETED, START) — and an unauthorised actor —
    (IN_PROGRESS, CANCEL_RIDER) by RIDER — are both "the table has no row you
    may use", which is why they share a type. The `reason` field keeps the
    distinction available for logging without forcing callers to catch two
    exceptions to handle one situation.
    """

    def __init__(self, state: TripState, action: TripAction, actor: TripActor,
                 reason: str) -> None:
        super().__init__(f"{actor.value} cannot {action.value} from "
                         f"{state.value}: {reason}")
        self.state = state
        self.action = action
        self.actor = actor
        self.reason = reason


class TripConflict(Exception):
    """The trip moved under us — the version compare-and-swap failed."""


def apply_transition(state: TripState, action: TripAction,
                     actor: TripActor) -> TripState:
    """The whole machine, in five lines. Deriving `allowed_actions` from the
    same table means the "what can I do now" query and the "may I do this"
    check can never disagree — which they do the moment they are written
    separately."""
    row = TRANSITIONS.get((state, action))
    if row is None:
        raise IllegalTransition(state, action, actor,
                                "no such transition from this state")
    next_state, permitted = row
    if actor not in permitted:
        raise IllegalTransition(state, action, actor,
                                f"only {sorted(a.value for a in permitted)} may do it")
    return next_state


def allowed_actions(state: TripState, actor: TripActor) -> list[TripAction]:
    """Derived, never hand-written. This is what a UI calls to decide which
    buttons to show, and because it reads the same table the check reads, a
    button can never be shown for an action that will then be refused."""
    return sorted((action for (st, action), (_n, actors) in TRANSITIONS.items()
                   if st is state and actor in actors),
                  key=lambda a: a.value)


def is_terminal(state: TripState) -> bool:
    return state in TERMINAL_STATES


@dataclass(frozen=True)
class TripEvent:
    state: TripState
    action: TripAction
    actor: TripActor
    at: float
    version: int

    def __str__(self) -> str:
        return (f"v{self.version} {self.actor.value} {self.action.value} "
                f"-> {self.state.value}")


class Trip:
    """The aggregate. It owns the state, the version that makes updates safe,
    and the history — and it owns the CAPTURED surge and quote, because a fare
    that changes mid-ride is a dispute. The HLD names fare disputes as a
    failure mode; this is the class that prevents them."""

    def __init__(self, trip_id: str, request: RideRequest, state: TripState,
                 surge_multiplier: float, quoted_fare_minor: int,
                 created_at: float) -> None:
        self.id = trip_id
        self.request = request
        self.state = state
        self.driver_id: Optional[str] = None
        self.surge_multiplier = surge_multiplier
        self.quoted_fare_minor = quoted_fare_minor
        self.created_at = created_at
        self.version = 0
        self.history: list[TripEvent] = []
        self.declined: set[str] = set()
        self._lock = threading.RLock()

    @property
    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    def advance(self, action: TripAction, actor: TripActor,
                expected_version: Optional[int] = None,
                driver_id: Optional[str] = None, now: Optional[float] = None
                ) -> TripEvent:
        """The version CAS and the transition, under one lock.

        `expected_version` is a PARAMETER rather than read inside, for the same
        reason it was in the food-delivery file: a version read here is always
        current, so the compare-and-swap would always pass and would compare
        nothing. The caller must present the version they were working from.
        """
        moment = time.monotonic() if now is None else now
        with self._lock:
            if expected_version is not None and expected_version != self.version:
                raise TripConflict(
                    f"trip {self.id} moved from version {expected_version} "
                    f"to {self.version}")
            next_state = apply_transition(self.state, action, actor)
            if action is TripAction.ASSIGN and driver_id is not None:
                if self.driver_id is not None and self.driver_id != driver_id:
                    raise TripConflict(
                        f"trip {self.id} is already assigned to {self.driver_id}")
                self.driver_id = driver_id
            self.state = next_state
            self.version += 1
            event = TripEvent(next_state, action, actor, moment, self.version)
            self.history.append(event)
            return event

    def __str__(self) -> str:
        driver = self.driver_id or "-"
        return (f"{self.id} [{self.state.value}] "
                f"rider={self.request.rider_id} driver={driver}")


# --------------------------------------------------------------------------
# 7. Surge — with a hysteresis band
# --------------------------------------------------------------------------

class SurgeCalculator:
    """Per-cell multiplier from supply and demand, WITH HYSTERESIS.

    The band is the whole design. A raw demand/supply ratio recomputed every
    second flickers as a single driver goes on and off shift, and the rider
    sees the price change while deciding whether to book — which is the
    "surge oscillation" the HLD names, and it is a product failure before it is
    a technical one.

    The rule: compute the target, and PUBLISH it only when it differs from the
    published value by at least `band` (as a fraction). Between those points
    the price is stable. It converges a little more slowly, and that is the
    trade being made, explicitly.
    """

    def __init__(self, floor: float = 1.0, ceiling: float = 3.0,
                 band: float = 0.15, min_demand: int = 3,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if floor < 1.0:
            raise ValueError("a multiplier below 1.0 is a discount, not a surge")
        if ceiling < floor:
            raise ValueError("the ceiling cannot be below the floor")
        if not 0.0 <= band < 1.0:
            raise ValueError("the band is a fraction of the published value")
        self.floor = floor
        self.ceiling = ceiling
        self.band = band
        self.min_demand = min_demand
        self.clock = clock
        self._published: dict[str, float] = {}
        self._lock = threading.Lock()

    def target(self, demand: int, supply: int) -> float:
        """The raw answer, before hysteresis. `supply` is floored at 1 so a cell
        with no drivers does not divide by zero — it saturates at the ceiling
        instead, which is the correct behaviour: no supply and real demand is
        the maximum surge, not an error."""
        if demand < self.min_demand:
            return self.floor
        ratio = demand / max(supply, 1)
        raw = 1.0 + 0.5 * (ratio - 1.0)
        return min(self.ceiling, max(self.floor, round(raw, 4)))

    def observe(self, cell: str, demand: int, supply: int,
                now: Optional[float] = None) -> float:
        """Record the state of a cell and return what is now published."""
        wanted = self.target(demand, supply)
        with self._lock:
            current = self._published.get(cell, self.floor)
            if abs(wanted - current) >= self.band * current:
                self._published[cell] = wanted
                return wanted
            return current

    def multiplier(self, cell: str, now: Optional[float] = None) -> float:
        with self._lock:
            return self._published.get(cell, self.floor)

    def published(self) -> dict[str, float]:
        with self._lock:
            return dict(self._published)


# --------------------------------------------------------------------------
# 8. Fare — integer minor units, rounded once
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FareQuote:
    base_minor: int
    distance_minor: int
    time_minor: int
    surge_minor: int
    total_minor: int
    currency: str = "USD"

    def __str__(self) -> str:
        return (f"{self.total_minor / 100:.2f} {self.currency} "
                f"(base {self.base_minor}, distance {self.distance_minor}, "
                f"time {self.time_minor}, surge {self.surge_minor})")


class FareEstimator:
    """Base + distance + time, times the surge captured at request time.

    Money is in INTEGER MINOR UNITS throughout. A fare computed in floats and
    rounded at the end produces totals that do not add up — the components
    disagree with the sum by a cent — and a fare that does not add up is a
    support ticket with a screenshot attached. There is exactly one rounding
    step in this class, and it is named in `quote`.
    """

    def __init__(self, base_minor: int = 250, per_km_minor: int = 120,
                 per_minute_minor: int = 25, minimum_minor: int = 500,
                 currency: str = "USD") -> None:
        self.base_minor = base_minor
        self.per_km_minor = per_km_minor
        self.per_minute_minor = per_minute_minor
        self.minimum_minor = minimum_minor
        self.currency = currency

    def quote(self, distance_km: float, duration_min: float,
              surge: float = 1.0) -> FareQuote:
        if distance_km < 0 or duration_min < 0:
            raise ValueError("a trip cannot have negative distance or duration")
        if surge < 1.0:
            raise ValueError("the surge multiplier is never below 1.0")
        base = self.base_minor
        # THE ONE ROUNDING STEP: distance is the only fractional input, so it is
        # rounded once here and never carried as a float afterwards.
        distance = int(round(distance_km * self.per_km_minor))
        duration = int(round(duration_min * self.per_minute_minor))
        subtotal = base + distance + duration
        total = max(self.minimum_minor, int(round(subtotal * surge)))
        return FareQuote(base, distance, duration, total - subtotal, total,
                         self.currency)


# --------------------------------------------------------------------------
# 9. The façade — the request, dispatch and respond paths
# --------------------------------------------------------------------------

class RideService:
    """The façade. It owns the order of the steps, because the order is the
    design: search, rank, offer, and only then assign."""

    def __init__(self, registry: DriverRegistry, strategy: MatchingStrategy,
                 surge: SurgeCalculator, fares: FareEstimator,
                 clock: Callable[[], float] = time.monotonic,
                 offer_ttl: float = 10.0, search_radius_km: float = 3.0,
                 candidates: int = 10,
                 estimated_speed_kmh: float = 25.0) -> None:
        self.registry = registry
        self.strategy = strategy
        self.surge = surge
        self.fares = fares
        self.clock = clock
        self.offers = OfferService(registry, offer_ttl, clock)
        self.search_radius_km = search_radius_km
        self.candidates = candidates
        self.estimated_speed_kmh = estimated_speed_kmh
        self.trips: dict[str, Trip] = {}
        self._by_client_key: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()
        self._counter = 0

    # -- the request ------------------------------------------------------
    def request_ride(self, rider_id: str, pickup: GeoPoint, dropoff: GeoPoint,
                     client_key: Optional[str] = None,
                     cell: Optional[str] = None) -> Trip:
        """Idempotent on (rider_id, client_key).

        The lookup and the insert share one lock, so two identical requests
        arriving together produce one trip and one duplicate is returned the
        second time — the same client-key rule as the food-delivery file, and
        for the same reason: the rider's app retries when the ack is lost.

        The SURGE and the QUOTE are captured HERE, at request time, and stored
        on the trip. Recomputing them at completion would let the price change
        during the ride, which the HLD names as a fare-dispute failure mode.
        """
        moment = self.clock()
        with self._lock:
            if client_key is not None:
                existing_id = self._by_client_key.get((rider_id, client_key))
                if existing_id is not None:
                    return self.trips[existing_id]
            self._counter += 1
            trip_id = f"trip-{self._counter:08d}"
            request = RideRequest(f"req-{trip_id}", rider_id, pickup, dropoff,
                                  moment, client_key)
            key = cell if cell is not None else geohash_encode(pickup, 5)
            multiplier = self.surge.multiplier(key, moment)
            distance = request.straight_line_km()
            duration = distance / self.estimated_speed_kmh * 60.0
            quote = self.fares.quote(distance, duration, multiplier)
            trip = Trip(trip_id, request, TripState.MATCHING, multiplier,
                        quote.total_minor, moment)
            self.trips[trip_id] = trip
            if client_key is not None:
                self._by_client_key[(rider_id, client_key)] = trip_id
            return trip

    # -- dispatch ---------------------------------------------------------
    def dispatch(self, trip: Trip, now: Optional[float] = None) -> Trip:
        """Search, rank, and claim the first candidate that is still free.

        THE OFFER LOOP MATTERS. Each candidate is claimed with a CAS, and a
        candidate somebody else just took raises OfferLost — which is caught and
        the loop moves on. Without that catch, a busy corner would surface a
        contention error to the rider as a failed booking.

        `trip.declined` is honoured because a driver who just refused the ride
        is immediately AVAILABLE again, so a naive re-dispatch would offer the
        same ride straight back to the person who said no. That is a real bug,
        not a hypothetical one.
        """
        moment = self.clock() if now is None else now
        if trip.state is not TripState.MATCHING:
            return trip
        found = self.registry.index.candidates(
            trip.request.pickup, self.search_radius_km, self.candidates, moment)
        pool = [d for d in found if d.id not in trip.declined]
        if not pool:
            trip.advance(TripAction.GIVE_UP, TripActor.SYSTEM, now=moment)
            return trip
        ranked = self.strategy.rank(trip.request, pool)
        for rank, driver in enumerate(ranked):
            try:
                self.offers.claim(trip.id, trip.request.rider_id, driver,
                                  rank, moment)
            except OfferLost:
                continue
            return trip                      # still MATCHING: an offer is out
        trip.advance(TripAction.GIVE_UP, TripActor.SYSTEM, now=moment)
        return trip

    # -- the driver's answer ----------------------------------------------
    def respond(self, trip: Trip, driver_id: str, accept: bool,
                now: Optional[float] = None) -> bool:
        """A driver taps Accept or Decline.

        Three outcomes, deliberately distinct:
        - the trip is already ASSIGNED to this driver: return True, because a
          double tap on Accept is a duplicate, not an error;
        - declined: release, record the refusal, return False;
        - accepted: two CASes (driver OFFERED -> ON_TRIP, then the trip's
          ASSIGN transition), and only then is the rider told a car is coming.
        """
        moment = self.clock() if now is None else now
        offer = self.offers.get_for_trip(trip.id)
        if offer is None:
            if trip.state is TripState.ASSIGNED and trip.driver_id == driver_id:
                return True                  # a duplicate Accept
            raise OfferLost(f"trip {trip.id} has no outstanding offer")
        if offer.driver_id != driver_id:
            raise OfferLost(f"offer for {trip.id} belongs to {offer.driver_id}")

        if not accept:
            self.offers.respond(offer, False, moment)
            trip.declined.add(driver_id)
            return False

        self.offers.respond(offer, True, moment)
        trip.advance(TripAction.ASSIGN, TripActor.SYSTEM,
                     expected_version=trip.version, driver_id=driver_id,
                     now=moment)
        return True

    def offer_next(self, trip: Trip, now: Optional[float] = None) -> Trip:
        """After a refusal or an expiry: try the next candidate."""
        return self.dispatch(trip, now)

    def expire_offers(self, now: Optional[float] = None) -> list[Trip]:
        """The sweep that makes an unanswered offer harmless. Returns the trips
        that lost their offer, so the caller can re-dispatch them — a real
        deployment runs this on a timer, not in the request path."""
        moment = self.clock() if now is None else now
        lapsed = self.offers.expire_outstanding(moment)
        touched: list[Trip] = []
        for offer in lapsed:
            trip = self.trips.get(offer.trip_id)
            if trip is not None and trip.state is TripState.MATCHING:
                touched.append(trip)
        return touched

    # -- the trip lifecycle -----------------------------------------------
    def advance(self, trip: Trip, action: TripAction, actor: TripActor,
                now: Optional[float] = None) -> TripEvent:
        moment = self.clock() if now is None else now
        return trip.advance(action, actor, expected_version=trip.version,
                            now=moment)

    def cancel(self, trip: Trip, actor: TripActor,
               now: Optional[float] = None) -> TripEvent:
        """Cancelling releases whatever the trip was holding.

        Two releases, and BOTH are needed. A trip cancelled after assignment
        holds a driver in ON_TRIP, released via `trip.driver_id`. A trip
        cancelled while still MATCHING holds a driver in OFFERED and has NO
        `driver_id` at all — so releasing only the driver id would strand that
        driver until the offer TTL lapsed. `cancel_for_trip` handles the second
        case, and a version of this method without it is correct in every test
        that cancels after assignment and wrong in production.

        The releases run AFTER the transition succeeds, and the transition is
        what decides whether the cancellation was allowed — so an illegal
        cancellation cannot leave a driver stranded with no trip attached. The
        reverse order would leak a driver on every rejected cancel attempt.
        """
        moment = self.clock() if now is None else now
        action = (TripAction.CANCEL_RIDER if actor is TripActor.RIDER
                  else TripAction.CANCEL_DRIVER if actor is TripActor.DRIVER
                  else TripAction.CANCEL_SYSTEM)
        event = trip.advance(action, actor, expected_version=trip.version,
                             now=moment)
        if trip.driver_id is not None:
            self.registry.release(trip.driver_id, moment)
        self.offers.cancel_for_trip(trip.id, moment)
        return event

    def complete(self, trip: Trip, actual_km: float, actual_minutes: float,
                 now: Optional[float] = None) -> FareQuote:
        """Complete the trip and settle the fare.

        The final fare uses the ACTUAL distance and duration but the surge
        CAPTURED at request time — so a surge rise mid-ride does not raise the
        bill, and a surge fall does not lower it. That is the honest choice: the
        rider agreed to a number."""
        moment = self.clock() if now is None else now
        trip.advance(TripAction.COMPLETE, TripActor.DRIVER,
                     expected_version=trip.version, now=moment)
        if trip.driver_id is not None:
            self.registry.finish_trip(trip.driver_id, moment)
        return self.fares.quote(actual_km, actual_minutes, trip.surge_multiplier)

    def get(self, trip_id: str) -> Trip:
        with self._lock:
            trip = self.trips.get(trip_id)
        if trip is None:
            raise KeyError(f"no such trip: {trip_id}")
        return trip


# --------------------------------------------------------------------------
# 10. Tests and demo
# --------------------------------------------------------------------------

def _check(name: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    if not condition:
        raise SystemExit(1)


def _raises(exc_type, fn) -> Optional[str]:
    try:
        fn()
    except exc_type as exc:
        return str(exc)
    return None


def _clock_box(start: float = 1000.0) -> tuple[Callable[[], float], dict]:
    box = {"t": start}
    return (lambda: box["t"]), box


if __name__ == "__main__":
    # --- 1. geohash: the published reference vector ---
    # The canonical example from the geohash specification: 42.6, -5.6 at
    # precision 5 is "ezs42". If this fails, the bit interleaving is wrong and
    # every cell in the system is wrong with it.
    _check("the reference geohash encodes correctly",
           geohash_encode(GeoPoint(42.6, -5.6), 5) == "ezs42")
    _check("precision changes the cell, not the location",
           geohash_encode(GeoPoint(42.6, -5.6), 7).startswith("ezs42"))
    _check("two nearby points share a prefix",
           geohash_encode(GeoPoint(42.6, -5.6), 6)[:4]
           == geohash_encode(GeoPoint(42.6001, -5.6001), 6)[:4])

    # --- 2. the bounds decode round-trips ---
    (lat_min, lat_max), (lon_min, lon_max) = geohash_bounds("ezs42")
    _check("the decoded box contains the point",
           lat_min <= 42.6 <= lat_max and lon_min <= -5.6 <= lon_max)
    _check("and the box is narrower than the coarser cell",
           (lat_max - lat_min) < 5.0 and (lon_max - lon_min) < 5.0)
    _check("re-encoding the box centre reproduces the code",
           geohash_encode(GeoPoint((lat_min + lat_max) / 2,
                                   (lon_min + lon_max) / 2), 5) == "ezs42")
    _check("an empty geohash has no bounds",
           _raises(ValueError, lambda: geohash_bounds("")) is not None)

    # --- 3. rings ---
    _check("ring 0 is the cell itself", cell_ring("ezs42", 0) == ["ezs42"])
    ring1 = cell_ring("ezs42", 1)
    _check("ring 1 is eight distinct neighbours",
           len(ring1) == 8 and len(set(ring1)) == 8)
    _check("and none of them is the centre", "ezs42" not in ring1)
    _check("ring 2 has sixteen cells", len(set(cell_ring("ezs42", 2))) == 16)
    _check("every ring-1 cell is a real geohash of the same precision",
           all(len(c) == 5 for c in ring1))
    _check("a negative ring is refused",
           _raises(ValueError, lambda: cell_ring("ezs42", -1)) is not None)

    # --- 4. distance ---
    london = GeoPoint(51.5074, -0.1278)
    paris = GeoPoint(48.8566, 2.3522)
    d = london.distance_km(paris)
    _check(f"London to Paris is about 344 km (got {d:.1f})", 340.0 < d < 348.0)
    _check("distance to self is zero", london.distance_km(london) == 0.0)
    _check("distance is symmetric",
           abs(d - paris.distance_km(london)) < 1e-9)
    _check("a quarter degree of latitude is about 27.8 km",
           27.5 < GeoPoint(0.0, 0.0).distance_km(GeoPoint(0.25, 0.0)) < 28.1)
    _check("an out-of-range latitude is refused",
           _raises(ValueError, lambda: GeoPoint(91.0, 0.0)) is not None)
    _check("an out-of-range longitude is refused",
           _raises(ValueError, lambda: GeoPoint(0.0, 181.0)) is not None)

    # --- 5. the index finds the nearest, and ALWAYS looks one ring out ---
    clock, box = _clock_box()
    registry = DriverRegistry(clock=clock, lease_ttl=15.0)
    index = registry.index
    rider = GeoPoint(37.7749, -122.4194)       # San Francisco

    # A driver in the rider's own cell, so ring 0 is not empty.
    near = registry.add(Driver("d-near", GeoPoint(37.7749, -122.4194)))
    # And one 5 km away, well outside.
    far = registry.add(Driver("d-far", GeoPoint(37.8200, -122.4194)))

    found = index.candidates(rider, 3.0, 5, clock())
    _check("the nearest driver is found", found and found[0].id == "d-near")
    _check("the far driver is outside the radius",
           "d-far" not in [d.id for d in found])

    # THE RING-1 CASE. Take a rider whose own cell is EMPTY and whose nearest
    # driver is in a neighbouring cell — the situation that a naive
    # "search only my cell" implementation gets wrong.
    empty_point = GeoPoint(37.7800, -122.4300)
    base_cell = geohash_encode(empty_point, index.precision_for(3.0))
    _check("the chosen rider cell starts empty",
           base_cell not in index._cells or not index._cells[base_cell])
    # Place a driver in a neighbour of that cell, at its centre.
    (nlat_min, nlat_max), (nlon_min, nlon_max) = geohash_bounds(
        cell_ring(base_cell, 1)[0])
    neighbour_centre = GeoPoint((nlat_min + nlat_max) / 2,
                                (nlon_min + nlon_max) / 2)
    neighbour_driver = registry.add(Driver("d-neighbour", neighbour_centre))
    _check("the neighbour driver is in a DIFFERENT cell",
           geohash_encode(neighbour_centre, len(base_cell)) != base_cell)
    ring_found = index.candidates(empty_point, 3.0, 5, clock())
    _check("and is still found, because ring 1 is always scanned",
           "d-neighbour" in [d.id for d in ring_found])

    # --- 6. the ring search agrees with brute force ---
    # Deterministic pseudo-random drivers: the strongest available check that
    # the stopping rule never discards a closer driver.
    index2 = GeoIndex()
    clock2, box2 = _clock_box()
    generated: list[Driver] = []
    for i in range(40):
        lat = 37.7500 + ((i * 7919) % 2000) / 100000.0
        lon = -122.4500 + ((i * 104729) % 2000) / 100000.0
        driver = Driver(f"d-{i:03d}", GeoPoint(lat, lon))
        driver.lease_expiry = 1e9
        generated.append(driver)
        index2.insert(driver)
    query = GeoPoint(37.7600, -122.4400)
    brute = sorted((d for d in generated
                    if d.location.distance_km(query) <= 3.0),
                   key=lambda d: (d.location.distance_km(query), d.id))
    for k in (1, 3, 5, 10):
        got = index2.candidates(query, 3.0, k, clock2())
        expected = brute[:k]
        _check(f"the ring search returns the true {k} nearest",
               [d.id for d in got] == [d.id for d in expected])
    _check("brute force found enough drivers for the test to mean something",
           len(brute) >= 10)

    # A driver just across a cell boundary must NOT be lost.
    edge = GeoPoint(37.7600, -122.4400)
    edge_cell = geohash_encode(edge, 7)
    (elat_min, elat_max), (elon_min, _elon_max) = geohash_bounds(edge_cell)
    just_over = GeoPoint(elat_min - 0.000001, edge.lon)   # one step south
    _check("the test point really is across a cell boundary",
           geohash_encode(just_over, 7) != edge_cell)
    index3 = GeoIndex()
    boundary_driver = Driver("d-boundary", just_over)
    boundary_driver.lease_expiry = 1e9
    index3.insert(boundary_driver)
    _check("a driver one metre across a boundary is still returned",
           [d.id for d in index3.candidates(edge, 3.0, 1, clock2())]
           == ["d-boundary"])

    # --- 7. the stale driver leaves the index on its own ---
    clock3, box3 = _clock_box(0.0)
    registry3 = DriverRegistry(clock=clock3, lease_ttl=15.0)
    registry3.add(Driver("d-stale", GeoPoint(37.7749, -122.4194)))
    _check("a fresh driver is a candidate",
           len(registry3.index.candidates(GeoPoint(37.7749, -122.4194), 3.0, 5,
                                          clock3())) == 1)
    box3["t"] = 16.0
    _check("past the lease the driver is no longer a candidate",
           registry3.index.candidates(GeoPoint(37.7749, -122.4194), 3.0, 5,
                                      clock3()) == [])
    swept = registry3.sweep()
    _check("the sweep removes exactly one driver", len(swept) == 1)
    _check("and marks them OFFLINE",
           registry3.get("d-stale").status is DriverStatus.OFFLINE)
    _check("and the index is now empty", registry3.index.size() == 0)
    _check("a second sweep finds nothing", registry3.sweep() == [])

    # A ping refreshes the lease and re-files a moved driver.
    registry3.set_status("d-stale", DriverStatus.AVAILABLE)
    box3["t"] = 100.0
    registry3.ping("d-stale", GeoPoint(37.8000, -122.4194))
    _check("a ping puts the driver back in the index",
           registry3.index.size() == 1)
    _check("and moves the location",
           abs(registry3.get("d-stale").location.lat - 37.8000) < 1e-9)
    box3["t"] = 110.0
    _check("and the lease is refreshed by the ping",
           registry3.index.candidates(GeoPoint(37.8000, -122.4194), 3.0, 5,
                                      clock3()) != [])

    # --- 8. THE CAS: eight riders, one driver, exactly one winner ---
    clock4, box4 = _clock_box(0.0)
    registry4 = DriverRegistry(clock=clock4, lease_ttl=1000.0)
    contested = registry4.add(Driver("d-one", GeoPoint(37.7749, -122.4194)))
    offers4 = OfferService(registry4, ttl=10.0, clock=clock4)
    barrier = threading.Barrier(8)
    wins: list[str] = []
    losses: list[str] = []
    outcome_lock = threading.Lock()

    def claimant(index_: int) -> None:
        barrier.wait()
        try:
            offer = offers4.claim(f"trip-{index_}", f"rider-{index_}", contested)
            with outcome_lock:
                wins.append(offer.id)
        except OfferLost as exc:
            with outcome_lock:
                losses.append(str(exc))

    threads = [threading.Thread(target=claimant, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _check("exactly ONE claimant won the driver", len(wins) == 1)
    _check("and seven were told it was gone", len(losses) == 7)
    _check("the driver is OFFERED",
           contested.status is DriverStatus.OFFERED)
    _check("and is out of the index while the offer is out",
           registry4.index.size() == 0)
    _check("exactly one offer is outstanding", offers4.outstanding_count() == 1)

    # --- 9. decline, expiry and the release path ---
    # The winner's trip id is whichever of the eight actually got the offer.
    winning_trip = None
    for i in range(8):
        if offers4.get_for_trip(f"trip-{i}") is not None:
            winning_trip = f"trip-{i}"
            break
    _check("the winning trip can be recovered from the offer service",
           winning_trip is not None)
    offer = offers4.get_for_trip(winning_trip)
    _check("the outstanding offer can be found by trip", offer is not None)

    declined = offers4.respond(offer, False, clock4())
    _check("declining returns False", declined is False)
    _check("and the driver goes back to AVAILABLE",
           contested.status is DriverStatus.AVAILABLE)
    _check("and is a candidate again", registry4.index.size() == 1)
    _check("and the driver can be offered again",
           offers4.claim("trip-99", "rider-99", contested) is not None)

    second = offers4.get_for_trip("trip-99")
    box4["t"] = 100.0                       # past the 10 s offer TTL
    _check("the offer reports itself expired", second.expired(clock4()) is True)
    _check("responding after expiry raises",
           _raises(OfferExpired,
                   lambda: offers4.respond(second, True, clock4())) is not None)
    _check("and the driver was released by the expiry",
           contested.status is DriverStatus.AVAILABLE)
    _check("with the index restored", registry4.index.size() == 1)

    # The sweep is the same release, on a timer.
    offers4.claim("trip-100", "rider-100", contested)
    box4["t"] = 200.0
    lapsed = offers4.expire_outstanding(clock4())
    _check("the sweep releases one offer", len(lapsed) == 1)
    _check("and nothing is left outstanding", offers4.outstanding_count() == 0)
    _check("and the driver is free again",
           contested.status is DriverStatus.AVAILABLE)

    # --- 10. a double Accept is harmless ---
    clock5, box5 = _clock_box(0.0)
    registry5 = DriverRegistry(clock=clock5, lease_ttl=1000.0)
    driver5 = registry5.add(Driver("d5", GeoPoint(37.7749, -122.4194)))
    svc5 = RideService(registry5, NearestFirst(), SurgeCalculator(clock=clock5),
                       FareEstimator(), clock=clock5, offer_ttl=30.0)
    trip5 = svc5.request_ride("rider-a", GeoPoint(37.7749, -122.4194),
                              GeoPoint(37.8000, -122.4000), client_key="k1")
    svc5.dispatch(trip5)
    _check("the trip is still MATCHING while the offer is out",
           trip5.state is TripState.MATCHING)
    _check("and is no longer MATCHING once accepted",
           svc5.respond(trip5, "d5", True) is True
           and trip5.state is TripState.ASSIGNED)
    _check("a duplicate Accept returns True rather than raising",
           svc5.respond(trip5, "d5", True) is True)
    _check("and does not advance the trip again", trip5.version == 1)
    _check("the trip records its driver", trip5.driver_id == "d5")
    _check("and the driver is ON_TRIP",
           driver5.status is DriverStatus.ON_TRIP)
    _check("and out of the pool", registry5.index.size() == 0)

    # --- 11. a refusal offers the next candidate ---
    clock6, box6 = _clock_box(0.0)
    registry6 = DriverRegistry(clock=clock6, lease_ttl=1000.0)
    registry6.add(Driver("d-close", GeoPoint(37.7749, -122.4194)))
    registry6.add(Driver("d-second", GeoPoint(37.7752, -122.4194)))
    svc6 = RideService(registry6, NearestFirst(), SurgeCalculator(clock=clock6),
                       FareEstimator(), clock=clock6, offer_ttl=30.0)
    trip6 = svc6.request_ride("rider-b", GeoPoint(37.7749, -122.4194),
                              GeoPoint(37.8000, -122.4000))
    svc6.dispatch(trip6)
    first_offer = svc6.offers.get_for_trip(trip6.id)
    _check("the nearest driver is offered first",
           first_offer.driver_id == "d-close")
    svc6.respond(trip6, "d-close", False)
    _check("the refusal is recorded on the trip",
           trip6.declined == {"d-close"})
    svc6.offer_next(trip6)
    second_offer = svc6.offers.get_for_trip(trip6.id)
    _check("the next candidate is offered instead",
           second_offer is not None and second_offer.driver_id == "d-second")
    _check("and NOT the driver who just refused",
           second_offer.driver_id != "d-close")

    # --- 12. the empty market, and recovery ---
    clock7, box7 = _clock_box(0.0)
    registry7 = DriverRegistry(clock=clock7, lease_ttl=1000.0)
    svc7 = RideService(registry7, NearestFirst(), SurgeCalculator(clock=clock7),
                       FareEstimator(), clock=clock7)
    trip7 = svc7.request_ride("rider-c", GeoPoint(37.7749, -122.4194),
                              GeoPoint(37.8000, -122.4000))
    svc7.dispatch(trip7)
    _check("with no drivers the trip goes to NO_DRIVERS_FOUND",
           trip7.state is TripState.NO_DRIVERS_FOUND)
    _check("which is terminal", trip7.is_terminal is True)
    _check("and no offer was created", svc7.offers.outstanding_count() == 0)

    trip8 = svc7.request_ride("rider-d", GeoPoint(37.7749, -122.4194),
                              GeoPoint(37.8000, -122.4000))
    registry7.add(Driver("d-late", GeoPoint(37.7749, -122.4194)))
    svc7.dispatch(trip8)
    _check("a driver arriving later makes the next request matchable",
           svc7.offers.get_for_trip(trip8.id) is not None)
    _check("and the earlier failed trip stays failed",
           trip7.state is TripState.NO_DRIVERS_FOUND)

    # --- 13. idempotent request ---
    same = svc7.request_ride("rider-d", GeoPoint(37.10, -122.10),
                             GeoPoint(37.20, -122.20), client_key=None)
    _check("no client key means no dedupe", same.id != trip8.id)
    again = svc7.request_ride("rider-d", GeoPoint(37.7749, -122.4194),
                              GeoPoint(37.8000, -122.4000), client_key="kk")
    repeat = svc7.request_ride("rider-d", GeoPoint(37.7749, -122.4194),
                               GeoPoint(37.8000, -122.4000), client_key="kk")
    _check("the same client key returns the SAME trip", again.id == repeat.id)
    other = svc7.request_ride("rider-e", GeoPoint(37.7749, -122.4194),
                              GeoPoint(37.8000, -122.4000), client_key="kk")
    _check("but a client key is scoped to the rider", other.id != again.id)

    # --- 14. the trip state machine ---
    _check("MATCHING + ASSIGN is ASSIGNED",
           apply_transition(TripState.MATCHING, TripAction.ASSIGN, TripActor.SYSTEM)
           is TripState.ASSIGNED)
    _check("the DRIVER may not assign", _raises(
        IllegalTransition,
        lambda: apply_transition(TripState.MATCHING, TripAction.ASSIGN,
                                 TripActor.DRIVER)) is not None)
    _check("the RIDER may not assign either", _raises(
        IllegalTransition,
        lambda: apply_transition(TripState.MATCHING, TripAction.ASSIGN,
                                 TripActor.RIDER)) is not None)
    _check("the driver may start from ARRIVING",
           apply_transition(TripState.ARRIVING, TripAction.START, TripActor.DRIVER)
           is TripState.IN_PROGRESS)
    _check("the rider may NOT start the trip", _raises(
        IllegalTransition,
        lambda: apply_transition(TripState.ARRIVING, TripAction.START,
                                 TripActor.RIDER)) is not None)

    # The two failure kinds, told apart by `reason`.
    pair_missing = _raises(
        IllegalTransition,
        lambda: apply_transition(TripState.COMPLETED, TripAction.START,
                                 TripActor.DRIVER))
    # A pair that IS in the table, attempted by an actor who is not on its
    # permitted list — this is the authorisation failure, and it is a different
    # reason from the pair being absent.
    actor_wrong = _raises(
        IllegalTransition,
        lambda: apply_transition(TripState.IN_PROGRESS, TripAction.CANCEL_SYSTEM,
                                 TripActor.DRIVER))
    # And the case with no row at all: a rider cancelling a ride already under
    # way. There is no (IN_PROGRESS, CANCEL_RIDER) entry, deliberately.
    rider_cancel_in_progress = _raises(
        IllegalTransition,
        lambda: apply_transition(TripState.IN_PROGRESS, TripAction.CANCEL_RIDER,
                                 TripActor.RIDER))
    _check("a nonexistent pair names the pair",
           pair_missing is not None and "no such transition" in pair_missing)
    _check("an unauthorised actor names the permitted actors",
           actor_wrong is not None and "only" in actor_wrong)
    _check("and names the state and the action it refused",
           actor_wrong is not None
           and "in_progress" in actor_wrong and "cancel_system" in actor_wrong)
    _check("the rider CANNOT cancel a trip in progress",
           rider_cancel_in_progress is not None
           and "no such transition" in rider_cancel_in_progress)

    _check("every terminal state refuses everything",
           all(_raises(IllegalTransition,
                       lambda s=s, a=a: apply_transition(s, a, TripActor.SYSTEM))
               is not None
               for s in TERMINAL_STATES for a in TripAction))
    _check("NO_DRIVERS_FOUND is terminal",
           is_terminal(TripState.NO_DRIVERS_FOUND) is True)
    _check("ASSIGNED is not", is_terminal(TripState.ASSIGNED) is False)

    # allowed_actions is DERIVED from the same table.
    rider_matching = allowed_actions(TripState.MATCHING, TripActor.RIDER)
    _check("a rider waiting for a match may cancel",
           TripAction.CANCEL_RIDER in rider_matching)
    _check("but may not assign", TripAction.ASSIGN not in rider_matching)
    _check("a rider in progress may do NOTHING",
           allowed_actions(TripState.IN_PROGRESS, TripActor.RIDER) == [])
    _check("a driver in progress may only complete",
           allowed_actions(TripState.IN_PROGRESS, TripActor.DRIVER)
           == [TripAction.COMPLETE])
    _check("every state has at least one SYSTEM action except the terminals",
           all(allowed_actions(s, TripActor.SYSTEM)
               for s in TripState if s not in TERMINAL_STATES))

    # --- 15. the full lifecycle, and the CAS ---
    clock8, box8 = _clock_box(0.0)
    registry8 = DriverRegistry(clock=clock8, lease_ttl=1000.0)
    registry8.add(Driver("d-life", GeoPoint(37.7749, -122.4194)))
    svc8 = RideService(registry8, NearestFirst(), SurgeCalculator(clock=clock8),
                       FareEstimator(), clock=clock8, offer_ttl=30.0)
    trip9 = svc8.request_ride("rider-f", GeoPoint(37.7749, -122.4194),
                              GeoPoint(37.8100, -122.3900))
    _check("a new trip starts in MATCHING", trip9.state is TripState.MATCHING)
    _check("with version 0", trip9.version == 0)
    svc8.dispatch(trip9)
    svc8.respond(trip9, "d-life", True)
    _check("accepting advances to ASSIGNED", trip9.state is TripState.ASSIGNED)
    svc8.advance(trip9, TripAction.DRIVER_ARRIVED, TripActor.DRIVER)
    _check("arriving advances to ARRIVING", trip9.state is TripState.ARRIVING)
    svc8.advance(trip9, TripAction.START, TripActor.DRIVER)
    _check("starting advances to IN_PROGRESS",
           trip9.state is TripState.IN_PROGRESS)
    quote = svc8.complete(trip9, actual_km=5.2, actual_minutes=14.0)
    _check("completing advances to COMPLETED", trip9.state is TripState.COMPLETED)
    _check("and the driver is free again with a trip counted",
           registry8.get("d-life").status is DriverStatus.AVAILABLE
           and registry8.get("d-life").trips_completed == 1)
    _check("the history has one row per transition", len(trip9.history) == 4)
    _check("and the versions are 1..4",
           [e.version for e in trip9.history] == [1, 2, 3, 4])
    _check("the first history row is the assignment",
           trip9.history[0].action is TripAction.ASSIGN)
    _check("and the last is the completion",
           trip9.history[-1].state is TripState.COMPLETED)
    _check("completing a completed trip is refused",
           _raises(IllegalTransition,
                   lambda: svc8.complete(trip9, 1.0, 1.0)) is not None)

    # The version CAS.
    stale = trip9.version - 1
    _check("a stale version is rejected",
           _raises(TripConflict, lambda: trip9.advance(
               TripAction.START, TripActor.DRIVER, expected_version=stale))
           is not None)
    _check("and the trip did not move",
           trip9.state is TripState.COMPLETED)

    # --- 16. cancel releases the driver ---
    clock9, box9 = _clock_box(0.0)
    registry9 = DriverRegistry(clock=clock9, lease_ttl=1000.0)
    registry9.add(Driver("d-cx", GeoPoint(37.7749, -122.4194)))
    svc9 = RideService(registry9, NearestFirst(), SurgeCalculator(clock=clock9),
                       FareEstimator(), clock=clock9, offer_ttl=30.0)
    trip10 = svc9.request_ride("rider-g", GeoPoint(37.7749, -122.4194),
                               GeoPoint(37.8100, -122.3900))
    svc9.dispatch(trip10)
    svc9.cancel(trip10, TripActor.RIDER)
    _check("cancelling from MATCHING succeeds",
           trip10.state is TripState.CANCELLED_BY_RIDER)
    _check("and the offered driver is released",
           registry9.get("d-cx").status is DriverStatus.AVAILABLE)
    _check("and back in the index", registry9.index.size() == 1)
    _check("and nothing is outstanding", svc9.offers.outstanding_count() == 0)
    _check("cancelling again is refused",
           _raises(IllegalTransition,
                   lambda: svc9.cancel(trip10, TripActor.RIDER)) is not None)
    _check("and did not strand the driver",
           registry9.get("d-cx").status is DriverStatus.AVAILABLE)

    # --- 17. surge, and the hysteresis band ---
    surge = SurgeCalculator(floor=1.0, ceiling=3.0, band=0.15, min_demand=3)
    _check("low demand sits at the floor", surge.target(1, 10) == 1.0)
    _check("balanced supply and demand is 1.0", surge.target(10, 10) == 1.0)
    _check("twice the demand is 1.5x", surge.target(20, 10) == 1.5)
    _check("four times the demand is 2.5x", surge.target(40, 10) == 2.5)
    _check("no supply saturates at the ceiling, not an error",
           surge.target(10, 0) == 3.0)
    _check("the ceiling clamps", surge.target(1000, 1) == 3.0)

    surge.observe("cell-a", 10, 10)
    _check("a cell with balanced supply and demand stays at the floor",
           surge.multiplier("cell-a") == 1.0)
    # A move LARGER than the band is published. 1.5 from 1.0 is a 50% move and
    # the band is 15%, so it goes out.
    _check("a move outside the band is published",
           surge.observe("cell-a", 20, 10) == 1.5)
    # A move INSIDE the band is not: the target is 1.525, the published value
    # is 1.5, so the difference is 0.025 against a band of 0.225.
    _check("a move inside the band is NOT published",
           surge.observe("cell-a", 20.5, 10) == 1.5)
    _check("and the published value is unchanged",
           surge.multiplier("cell-a") == 1.5)
    _check("but a large move is published again",
           surge.observe("cell-a", 40, 10) == 2.5)
    _check("and the hysteresis did not block the big moves",
           surge.multiplier("cell-a") == 2.5)
    _check("an unobserved cell is at the floor",
           surge.multiplier("cell-never-seen") == 1.0)
    _check("a floor below 1.0 is refused",
           _raises(ValueError, lambda: SurgeCalculator(floor=0.9)) is not None)
    _check("a band outside [0, 1) is refused",
           _raises(ValueError, lambda: SurgeCalculator(band=1.0)) is not None)

    # --- 18. fare, in integer minor units ---
    fares = FareEstimator()
    q = fares.quote(5.0, 15.0, 1.0)
    _check("base is 250 minor", q.base_minor == 250)
    _check("5 km at 120/km is 600", q.distance_minor == 600)
    _check("15 min at 25/min is 375", q.time_minor == 375)
    _check("no surge means no surge component", q.surge_minor == 0)
    _check("the total is the sum of the parts", q.total_minor == 1225)
    _check("and the components add up",
           q.base_minor + q.distance_minor + q.time_minor + q.surge_minor
           == q.total_minor)
    q2 = fares.quote(5.0, 15.0, 2.0)
    _check("a 2x surge doubles the total", q2.total_minor == 2450)
    _check("and reports the surge component",
           q2.surge_minor == 1225)
    _check("the parts still add up",
           q2.base_minor + q2.distance_minor + q2.time_minor + q2.surge_minor
           == q2.total_minor)
    short = fares.quote(0.1, 1.0, 1.0)
    _check("the minimum fare applies to a very short trip",
           short.total_minor == 500)
    _check("and the surge component absorbs the difference",
           short.base_minor + short.distance_minor + short.time_minor
           + short.surge_minor == short.total_minor)
    _check("a surge below 1.0 is refused",
           _raises(ValueError, lambda: fares.quote(1.0, 1.0, 0.5)) is not None)
    _check("negative distance is refused",
           _raises(ValueError, lambda: fares.quote(-1.0, 1.0)) is not None)

    # --- 19. the batched assignment beats greedy ---
    # A crafted score matrix, not derived from distance: (r1,d1)=10, (r1,d2)=9,
    # (r2,d1)=9, (r2,d2)=0. Greedy rider-by-rider takes d1 for r1 and leaves r2
    # with d2 — total 10. The batched assignment gives 18.
    matrix = {("r1", "d1"): 10.0, ("r1", "d2"): 9.0,
              ("r2", "d1"): 9.0, ("r2", "d2"): 0.0}

    def lookup(request: RideRequest, driver: Driver) -> float:
        return matrix[(request.rider_id, driver.id)]

    batched = BatchedOptimal(lookup)
    r1 = RideRequest("q1", "r1", GeoPoint(0.0, 0.0), GeoPoint(1.0, 1.0), 0.0)
    r2 = RideRequest("q2", "r2", GeoPoint(0.0, 0.0), GeoPoint(1.0, 1.0), 0.0)
    d1 = Driver("d1", GeoPoint(0.0, 0.0))
    d2 = Driver("d2", GeoPoint(0.0, 0.0))
    requests = [r1, r2]
    candidates = [[d1, d2], [d1, d2]]

    greedy = batched.greedy_batch(requests, candidates)
    greedy_total = sum(matrix[(req.rider_id, drv.id)] for req, drv in greedy)
    _check("greedy rider-by-rider scores 10", greedy_total == 10.0)
    _check("and produces the crossing-free pairing it was always going to",
           [(req.rider_id, drv.id) for req, drv in greedy]
           == [("r1", "d1"), ("r2", "d2")])

    optimal = batched.assign_batch(requests, candidates)
    optimal_total = sum(matrix[(req.rider_id, drv.id)] for req, drv in optimal)
    _check("the batched assignment scores 18", optimal_total == 18.0)
    _check("by swapping the two drivers",
           [(req.rider_id, drv.id) for req, drv in optimal]
           == [("r1", "d2"), ("r2", "d1")])
    _check("so batching is worth 80% more on this matrix",
           optimal_total > greedy_total * 1.5)

    # Feasibility is respected: a driver not in a request's candidate list is
    # never assigned to it.
    only_d1 = [[d1], [d1, d2]]
    feasible = batched.assign_batch(requests, only_d1)
    _check("an infeasible pairing is never chosen",
           all(drv in req_list for (req, drv), req_list
               in zip(feasible, only_d1)))
    _check("and the batch counter counted the calls",
           batched.batches_assigned == 2)

    # --- 20. two strategies, two different orders ---
    near_but_mediocre = Driver("d-near-low", GeoPoint(37.7750, -122.4194),
                              rating=4.1)
    far_but_excellent = Driver("d-far-high", GeoPoint(37.7900, -122.4194),
                               rating=5.0)
    request20 = RideRequest("q20", "r20", GeoPoint(37.7749, -122.4194),
                            GeoPoint(37.8, -122.4), 0.0)
    pool = [near_but_mediocre, far_but_excellent]
    _check("NearestFirst puts the close driver first",
           NearestFirst().rank(request20, pool)[0].id == "d-near-low")
    _check("HighestRated puts the better driver first",
           HighestRated().rank(request20, pool)[0].id == "d-far-high")
    _check("both return the same SET",
           {d.id for d in NearestFirst().rank(request20, pool)}
           == {d.id for d in HighestRated().rank(request20, pool)})

    # --- 21. the dispatch race: many riders, one driver, one winner ---
    clock21, box21 = _clock_box(0.0)
    registry21 = DriverRegistry(clock=clock21, lease_ttl=1000.0)
    registry21.add(Driver("d-solo", GeoPoint(37.7749, -122.4194)))
    svc21 = RideService(registry21, NearestFirst(),
                        SurgeCalculator(clock=clock21), FareEstimator(),
                        clock=clock21, offer_ttl=30.0)
    trips = [svc21.request_ride(f"r-{i}", GeoPoint(37.7749, -122.4194),
                                GeoPoint(37.80, -122.40), client_key=f"k-{i}")
             for i in range(8)]
    barrier21 = threading.Barrier(8)
    results: list[bool] = []
    results_lock = threading.Lock()

    def racer21(trip: Trip) -> None:
        barrier21.wait()
        svc21.dispatch(trip)
        with results_lock:
            results.append(svc21.offers.get_for_trip(trip.id) is not None)

    threads21 = [threading.Thread(target=racer21, args=(t,)) for t in trips]
    for t in threads21:
        t.start()
    for t in threads21:
        t.join()
    _check("exactly one trip got an offer", sum(results) == 1)
    _check("the rest went to NO_DRIVERS_FOUND",
           sum(1 for t in trips if t.state is TripState.NO_DRIVERS_FOUND) == 7)
    _check("and the driver was claimed exactly once",
           registry21.get("d-solo").status is DriverStatus.OFFERED)
    _check("with one offer outstanding", svc21.offers.outstanding_count() == 1)

    print("\nall checks passed")
```

**What to point at:**

- `GeoIndex.candidates` is the centre of the file, and both of its rules are about *stopping*. `ring >= 1` forces at least the neighbour ring, because a cell boundary is an encoding artefact and the nearest driver is often across the street from where the encoding says they are. And the stop threshold `kth <= ring * cell_km_min` waits until no unscanned cell could hold a closer driver — the alternative, stopping at the first cell with a hit, returns the far corner of a ring. The demo proves it two ways: a hand-built boundary case, and a 40-driver brute-force comparison at k = 1, 3, 5, 10.
- `DriverRegistry.try_claim` is the CAS. Eight threads claim one driver behind a barrier and exactly one wins; the demo also races eight *full dispatches* and asserts one offer and seven `NO_DRIVERS_FOUND`.
- `OfferService.respond` distinguishes three outcomes, and the "already assigned to this driver" case returns `True` — a double tap on Accept is a duplicate, not an error.
- `Trip.advance` takes `expected_version` as a parameter, for the same reason as the food-delivery file: a version read inside is always current, so the compare-and-swap would compare nothing.
- `BatchedOptimal` carries both the greedy and the exhaustive batch, because the numeric contrast — 10 versus 18 — is the entire argument for batching.

---

## 6. Concurrency & thread-safety

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Two riders claim one driver** | Both are told a car is coming; one rider watches it drive past | `try_claim`: the check and the write are one operation under one lock |
| **A read-then-write claim** | `if driver.status is AVAILABLE:` outside the lock, then set — passes for both threads | The CAS is the method; there is no way to call it in two steps |
| **An offered driver still in the index** | Offered to rider B while rider A's offer is on screen | The claim removes the driver from the index; the release re-inserts |
| **A driver who never answers** | Occupied by a rider who has already booked elsewhere | The offer TTL, plus `expire_outstanding` on a timer |
| **A cancelled trip that was never assigned** | The driver stays OFFERED with no trip attached, invisible to everyone | `cancel_for_trip` releases the offer, not just `trip.driver_id` |
| **A double tap on Accept** | Two `ASSIGN` transitions, or a spurious error to the driver | `begin_trip` is a second CAS, and the second tap returns `True` as a duplicate |
| **A stale driver in the index** | Rides offered to somebody whose app died an hour ago | The availability lease, and `sweep` |
| **A driver who crosses a cell boundary** | The index keeps them where they were | `GeoIndex.move` re-files on every ping |
| **Membership of a cell changing during a search** | The candidate list holds a driver who went offline mid-search | Every candidate is re-checked by `available(now)` at selection time, and the CAS re-checks at claim time |
| **A trip moving under an update** | Two transitions applied from the same starting state, losing one | `Trip.advance` takes `expected_version` and raises `TripConflict` |
| **The `expected_version` read inside the lock** | The compare-and-swap compares nothing, because the version read is always current | The version is a parameter, and the docstring says so |
| **The ring search holding the index lock** | Location pings block behind a search | The search copies out the candidates under the lock and does the sorting outside it |
| **Two surges for the same cell** | The multiplier flickers and the price changes while the rider decides | The hysteresis band, and one cell key per observation |
| **A fare computed in floats** | The parts do not add up to the total | Integer minor units, and exactly one rounding step |

### The two serialisation points

Everything above reduces to two places where contention is resolved, and naming them is the answer to "where is the race":

```python
# 1. CLAIMING A DRIVER — DriverRegistry.try_claim.
#    AVAILABLE -> OFFERED. Eight riders, one winner. This is the answer to
#    "what stops double-assignment", and it is one line of state change inside
#    one lock, not a distributed protocol.

# 2. MOVING A TRIP — Trip.advance.
#    A version compare-and-swap plus the table lookup, under the trip's lock.
#    This is the answer to "what stops a transition being applied twice", and
#    it is why the version is a PARAMETER.
```

### The distributed version

At more than one matching server per city, the driver's status must live somewhere both can see it — a shared store, not the in-process `DriverRegistry`. The class-level consequence is narrow and worth stating precisely: **`try_claim`, `release`, `begin_trip` and `finish_trip` become conditional writes against that store**, and everything else in this file is unchanged. That is the payoff of putting the CAS behind a method rather than spreading status checks through the service.

```python
class DistributedDriverRegistry(DriverRegistry):
    """The same interface with the status transitions as conditional writes.

    In a real system `try_claim` is a single-statement compare-and-set — an
    UPDATE ... WHERE status = 'available', or a Redis Lua script — and its
    return value says whether this caller won. The point of writing it this way
    in the interview is that NOTHING ELSE CHANGES: the offer service, the trip
    machine and the fare code never learn that the registry became distributed,
    because they only ever asked it questions through its interface.
    """

    def __init__(self, store, index: GeoIndex,
                 clock: Callable[[], float] = time.monotonic,
                 lease_ttl: float = 15.0) -> None:
        super().__init__(index, clock, lease_ttl)
        self.store = store

    def try_claim(self, driver_id: str) -> bool:
        # One atomic compare-and-set in the shared store. The local cache is
        # best-effort and is never the authority.
        won = self.store.compare_and_set(
            driver_id, expect="available", set_to="offered")
        if won:
            self.index.remove(driver_id)
        return won

    def release(self, driver_id: str, now: Optional[float] = None) -> bool:
        moment = self.clock() if now is None else now
        won = self.store.compare_and_set(
            driver_id, expect="offered", set_to="available")
        if won:
            driver = self._drivers.get(driver_id)
            if driver is not None:
                driver.status = DriverStatus.AVAILABLE
                driver.lease_expiry = moment + self.lease_ttl
                self.index.insert(driver)
        return won
```

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Riders can share a ride"

The design point is that pooling does not need a new matcher — it needs a **different candidate source and a different feasibility rule**, and both fit behind the seams already present.

```python
@dataclass(frozen=True)
class PoolRequest:
    """A ride request that will accept a detour, with the price of that
    acceptance stated as data rather than buried in the matcher."""

    request: RideRequest
    max_detour_ratio: float = 0.35       # +35% over the direct trip
    max_extra_minutes: float = 12.0
    seats: int = 1


def detour_ok(existing: PoolRequest, candidate: PoolRequest,
              estimator: FareEstimator, speed_kmh: float = 25.0) -> bool:
    """The feasibility rule, as a free function. A pooled match is legal only
    if BOTH riders' trips stay inside their detour budgets — a rule that looks
    at one rider at a time is the bug that makes pooled rides arrive an hour
    late for the second passenger."""
    direct = existing.request.straight_line_km()
    via = (existing.request.pickup.distance_km(candidate.request.pickup)
           + candidate.request.pickup.distance_km(candidate.request.dropoff)
           + candidate.request.dropoff.distance_km(existing.request.dropoff))
    ratio_ok = via <= direct * (1.0 + existing.max_detour_ratio) + 1e-9
    extra_minutes = (via - direct) / speed_kmh * 60.0
    return ratio_ok and extra_minutes <= existing.max_extra_minutes
```

### 7.2 "Schedule a ride for later"

The design point: a scheduled ride is **not a trip with a future timestamp**. It is a *job* that creates a trip at the right moment, because everything about a trip — the surge, the quote, the driver — must be decided when it starts, not when it was booked.

```python
@dataclass(frozen=True)
class ScheduledRide:
    id: str
    rider_id: str
    pickup: GeoPoint
    dropoff: GeoPoint
    for_at: float
    created_at: float
    lead_seconds: float = 600.0          # start matching 10 min ahead


def due_scheduled(scheduled: list[ScheduledRide], now: float) -> list[ScheduledRide]:
    """What to turn into a real trip right now.

    Starting the match at `for_at` would guarantee a late pickup; starting it
    at booking time would hold a driver for hours. The lead time is the whole
    design, and it is a parameter for exactly that reason. Note what is NOT
    stored on a ScheduledRide: no surge multiplier and no quote, because those
    are facts about the moment the ride starts.
    """
    return [s for s in scheduled if now >= s.for_at - s.lead_seconds]
```

### 7.3 "Add a new matching strategy"

The seam already exists; the follow-up is really asking whether you know why.

```python
class LongestIdleFirst(MatchingStrategy):
    """A fairness strategy: the driver who has waited longest gets the ride.

    It exists to make a point about the seam. The index produced the candidates
    and knows nothing about ordering; the service calls `rank` and knows nothing
    about which ordering it got. Adding this strategy is one class and one
    constructor argument, and NOTHING in `GeoIndex`, `OfferService` or `Trip`
    changes — which is the test of whether a strategy pattern is real or
    decorative.
    """

    def rank(self, request: RideRequest, candidates: list[Driver]) -> list[Driver]:
        return sorted(candidates,
                      key=lambda d: (d.idle_since, d.id))
```

### 7.4 "The driver is on the way but takes a wrong turn"

The design point: **the ETA is a projection, not a state.** Everything the rider sees between `ASSIGNED` and `ARRIVING` is derived from the driver's location stream, and none of it is a transition.

```python
class EtaProjector:
    """Derives what the rider sees: minutes away, and a path.

    Deliberately NOT part of the trip state machine. A wrong turn is not a
    state — it is a changed projection over the same state — and modelling it as
    one produces a state machine with `REROUTING` and `DELAYED` in it, which
    then has to be reconciled with `ARRIVING` a minute later. Keep the machine
    about commitments and the projector about estimates.
    """

    def __init__(self, speed_kmh: float = 25.0) -> None:
        self.speed_kmh = speed_kmh

    def minutes_away(self, driver_location: GeoPoint,
                     pickup: GeoPoint) -> float:
        return driver_location.distance_km(pickup) / self.speed_kmh * 60.0

    def has_arrived(self, driver_location: GeoPoint, pickup: GeoPoint,
                    threshold_m: float = 120.0) -> bool:
        """The geofence that triggers ARRIVING. A threshold in metres, not an
        exact equality: GPS does not land on the pin, and an equality check
        would leave a driver circling the block."""
        return driver_location.distance_km(pickup) * 1000.0 <= threshold_m
```

### 7.5 "Why not one big `MatchingEngine` class?"

Because four questions with four different change rates are being merged. The index changes when the geo encoding changes, the strategy changes on a product decision, the offer changes when the dispatch protocol changes, and the state machine changes when the trip lifecycle changes. One class means every one of those edits touches the same file, and the tests for the state machine have to construct a geo index to run.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Searching only the rider's own cell** | The nearest driver, ten metres across a boundary, is invisible | Ring 1 is always scanned |
| 2 | **Stopping at the first cell with a hit** | Returns the far corner of a ring while a closer driver sits unvisited next door | The ring is complete, and the stop threshold is the k-th distance |
| 3 | **A driver who stopped pinging** | Offered rides they will never see | The availability lease, and the sweep |
| 4 | **A driver crossing a cell boundary** | The index keeps them at their old cell forever | `move` on every ping |
| 5 | **Assigning instead of offering** | The rider is told a car is coming that never moves | The offer, with a TTL |
| 6 | **Two riders, one driver** | Both are told yes | The CAS |
| 7 | **A driver who ignores the offer** | Occupied until they notice | The TTL releases them |
| 8 | **A declined driver re-offered immediately** | The same ride goes back to the person who just said no | `trip.declined` is honoured on re-dispatch |
| 9 | **Cancelling a trip that only had an offer** | The driver stays OFFERED with no trip, invisible to everyone | `cancel_for_trip` |
| 10 | **A double tap on Accept** | Two transitions, or an error the driver does not understand | The second tap is a duplicate and returns `True` |
| 11 | **A rider cancelling mid-ride** | The trip is cancelled while the car is moving and the driver is unpaid | No such transition; the support path exists |
| 12 | **An empty market** | The request hangs, and the rider sees a spinner forever | `NO_DRIVERS_FOUND` is a terminal state |
| 13 | **A stale version applied twice** | A transition is applied from a state the caller never saw | The version CAS, with the version as a parameter |
| 14 | **Surge recomputed every second** | The price flickers as one driver goes on shift | The hysteresis band |
| 15 | **Surge recomputed at completion** | The rider is charged a price they never agreed to | The multiplier is captured at request time |
| 16 | **A cell with no drivers** | Divide by zero in the ratio | `max(supply, 1)`, saturating at the ceiling |
| 17 | **A fare in floats** | The parts do not add up to the total | Integer minor units, one rounding step |
| 18 | **GPS not exactly on the pin** | `ARRIVING` never fires and the driver circles the block | A geofence threshold in metres |
| 19 | **A retried ride request** | Two trips, two drivers, one rider | The client key |
| 20 | **A driver's ping while ON_TRIP** | The driver is put back in the available pool mid-ride | `touch` never changes status |

**The three to volunteer:** #1 and #2 together (the ring rules — they are the signature of this problem and the thing an interviewer is listening for), #6 (the CAS, stated as "the check and the write are one operation"), and #15 (surge captured at request time, which is a product decision expressed in the class model).

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **Strategy** | `MatchingStrategy`: nearest, rated, batched | The objective is a product decision and changes without touching the index | One more class per policy |
| **State machine** | `TRANSITIONS`, `Trip.advance` | The trip lifecycle has many illegal combinations and one legal path | A table to maintain, and it must be read to know the behaviour |
| **Lease** | `Driver.lease_expiry`, `Offer.expires_at` | Two leases for two different claims: "this driver exists" and "this driver is reserved" | Two TTLs to tune |
| **Repository** | `DriverRegistry`, `GeoIndex` | The index is a storage structure and its invariants deserve their own tests | An abstraction over a dict |
| **Aggregate** | `Trip` owns state, version, history, captured surge | One boundary per invariant; a fare cannot change under the trip | The version discipline |
| **Value Object** | `GeoPoint`, `RideRequest`, `FareQuote`, `Offer` | Frozen, hashable, comparable in tests | An allocation per call |
| **Entity** | `Driver` with `eq=False` | Identity, not value: two drivers with identical fields are two drivers | Nothing — the alternative is a bug |
| **Façade** | `RideService` | The order of search → rank → offer → assign is the design and belongs in one place | A façade that is not thin |
| **Specification** | `detour_ok` (§7.1) | A business rule as a function, testable without a ride | A free function that needs plumbing |
| **Singleton** | *Deliberately rejected* | A global registry makes two tests share state, fatal with 150+ assertions | — |
| **Observer** | *Deliberately rejected* for trip state | A subscriber that throws must not lose a transition; the history list is an explicit log | The caller reads history rather than subscribing |

**The sentence to give:** *"Strategy for the matcher because the objective is a product decision; a table-driven state machine because the lifecycle has more illegal combinations than legal ones; a lease for the driver and a second lease for the offer because they are two different claims with two different natural TTLs; and a repository for the index because the ring logic is the hardest thing in the file and deserves to be testable without a ride service."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Geohash grid vs quadtree** | Geohash | A flat map, a trivially serialisable key, ring expansion by re-encoding | Neighbours must be computed; a quadtree gives them for free |
| **Ring expansion by degrees** | Yes | Obviously correct, and no string-manipulation neighbour table | A decode and a re-encode per ring cell |
| **Always scan ring 1** | Yes | The boundary case can never lose the nearest driver | Roughly 9x the cells examined for a sparse query |
| **Stop threshold from the k-th distance** | Yes | Provably never discards a closer driver | The search cannot stop early on a lucky first cell |
| **Offer, not assignment** | Yes | A driver who does not answer is handled by the TTL, not by a handler | A round trip before the rider sees a car |
| **A second lease on the offer** | Yes | A crashed driver app costs one TTL, not one stuck driver | Two TTLs to tune and reason about |
| **Driver removed from the index while offered** | Yes | No double-offering without a second status check in the search | A re-insert on every release |
| **`trip.declined` on the trip** | Yes | A refusal cannot loop back to the refuser | The set grows over a long search |
| **Rider cannot cancel mid-ride** | Yes | The driver is protected once the journey has started | A support path must exist |
| **Batched matching** | Available, not default | Nearly double the score on adversarial matrices | Exhaustive search bounded by a batch size |
| **Surge hysteresis band** | Yes | A stable price the rider can decide against | Slower convergence to a genuine change |
| **Surge captured at request** | Yes | No fare disputes | The platform absorbs a mid-ride surge rise |
| **Integer minor units** | Yes | The parts add up | Slightly more verbose arithmetic |
| **`eq=False` on Driver** | Yes | Identity semantics, and hashable for the index | A driver is not comparable by value |

**The trade to name unprompted:** *"The offer round trip costs the rider a few seconds before they see a car, and I would still take it, because the alternative — assign and hope — fails in the direction that is visible and infuriating. What I would change is the ring-1 guarantee for a dense city centre: scanning nine cells for every query is fine at low density and wasteful where every cell has drivers, so a real deployment would cache the last search per cell or switch to a quadtree that descends to a node with k drivers directly."*

---

## 11. Q&A

**Q1. Why not just search the cell the rider is in?**
Because a cell boundary is an artefact of the geohash encoding, not a fact about geography. Two points ten metres apart can fall in different cells, and the rider standing near a boundary is precisely the case where the nearest driver is in the cell next door. The rule in `candidates` is therefore `ring >= 1`: the neighbour ring is always scanned, whatever ring 0 found. The demo builds the case explicitly — a rider whose own cell is empty with a driver in a neighbour cell — and then, more convincingly, compares the ring search against a brute-force scan over 40 drivers at k = 1, 3, 5 and 10.

**Q2. And why not stop as soon as you have k candidates?**
Because the first cell that yields a hit may be the far corner of its ring. Ring 1 is eight cells, and a driver 400 m away in the first cell you happen to visit does not mean there is not a driver 50 m away in the eighth. The stop rule is `kth_distance <= ring * cell_km_min`: cells in ring r+1 are at least r cell-widths from the centre, so once the k-th candidate is closer than that, no unscanned cell can beat it. It is deliberately conservative — `cell_km_min` is the *narrower* of the cell's two dimensions, which shrinks with latitude, so the bound holds everywhere rather than only near the equator.

**Q3. Why an offer instead of an assignment?**
Because the driver has not agreed. An assignment is a claim the system cannot back up, and the moment the driver declines you have to unwind it — which means the rider was told something untrue in the meantime. An offer makes the uncertainty explicit: it is a lease with a TTL, so the three possible outcomes (accept, decline, ignore) are all first-class. Ignoring is the one that matters, because there is no handler for it anywhere: the offer lapses and the driver is released by the same path that handles a decline.

**Q4. Where exactly is the double-assignment prevented?**
In `DriverRegistry.try_claim`, and it is one operation under one lock: check `status is AVAILABLE`, set `status = OFFERED`, remove from the index. The bug it prevents is the read-then-write version — `if driver.status is AVAILABLE:` evaluated outside the lock, then assigned — which passes for every thread in the race. It is invisible in a test with no concurrency, which is why the demo uses a `threading.Barrier` to release eight claimers simultaneously and asserts exactly one winner and seven `OfferLost`. There is a second CAS at `begin_trip` for the same reason: a double tap on Accept must not produce two transitions.

**Q5. What happens to a driver who never answers the offer?**
The TTL releases them, and nothing else in the system has to know. `OfferService.expire_outstanding` runs on a timer, finds the lapsed offers, releases each driver back to AVAILABLE and back into the index, and returns the offers so the caller can re-dispatch those trips. That is the whole handler for "the driver's phone died mid-offer", and it is the same code path as a decline — which is the design paying off: two different real-world events, one implementation.

**Q6. Why is the driver removed from the index when offered?**
So the search cannot hand out a driver who is already spoken for. The alternative — leaving them in and filtering on status at selection time — works, but it puts the same check in two places and means every search reads drivers it is about to discard. Removing them also makes the invariant visible: a driver in the index is a driver who is genuinely free. The cost is a re-insert on every release, which is one hash lookup.

**Q7. Is `NearestFirst` not obviously correct? Why the strategy at all?**
Nearest-first is correct for minimising pickup time, and pickup time is what the rider feels, so it is a good default. It is not correct for the marketplace's objective, which is total wait plus driver idle time plus acceptance probability — and it is not even a tie-breaker-neutral choice, because always giving the ride to the closest driver starves the ones in low-demand areas. The strategy exists so that is a decision rather than an accident. `BatchedOptimal` is in the file to make the cost concrete: on the 2×2 score matrix in the demo, greedy rider-by-rider scores 10 and the batched assignment scores 18, from the same two drivers.

**Q8. Then why is batching not the default?**
Because it is exhaustive. `assign_batch` enumerates permutations of the candidate pool, which is 30,240 combinations at five requests over ten candidates and 3.6 million at eight — so it is capped at a batch size and falls back to greedy above it. Batching also adds a delay, because you have to wait for a batch to accumulate, and a rider waiting for a batch is a rider watching a spinner. The honest position: batching is right for a dense area with a request queue and wrong for a single rider in a suburb.

**Q9. Why is surge hysteresis in the class and not in the pricing service?**
Because the band is a property of the *decision*, not of the arithmetic, and the class that decides is the class that should hold it. Splitting it out would mean two places compute a multiplier and only one applies the band. The mechanism is four lines: compute the target, compare it against the published value, publish only if the difference is at least the band. The demo walks a cell through a large move (published), a 1.7% move (not published), and another large move (published) to show that the band damps noise without blocking real changes.

**Q10. Why is the surge captured on the trip rather than recomputed at the end?**
Because the rider was shown a number and agreed to it. If the multiplier is recomputed at completion, a surge that rose during the ride raises the bill for a decision the rider already made, and a surge that fell means the platform quietly keeps the difference. Both are disputes. Capturing it at request time means the final fare uses the *actual* distance and duration — which is fair, because that is the service delivered — with the *quoted* multiplier, which is fair because that is the price agreed. The HLD names fare disputes as a failure mode; this is the field that prevents them.

**Q11. Why is `expected_version` a parameter on `Trip.advance`?**
Because a version read inside the method is always current, so the comparison would always pass and the compare-and-swap would compare nothing. The caller must present the version they were working from — the one they read before doing whatever else they did — and the CAS fails if the trip moved in between. This is the same reasoning as the food-delivery file's `OrderLock.assign_driver`, and it is the kind of bug that is invisible until a test forces the interleaving.

**Q12 (senior). Where does this design break at scale?**
Two places, and neither is "add more servers". First, the index: `GeoIndex` is an in-process map, so several matching servers need a shared view of driver positions, and the ring search then becomes a network round trip per ring. The mitigation is a per-city partition of drivers across matching servers — which is a change to *which* index you ask, not to the class — plus a cached candidate list per cell, accepting staleness of a few seconds because a driver who moved 50 m is still in the same search result. Second, the offer's blast radius: with one driver and one rider the offer is cheap, but in a dense market the dispatch loop over ten candidates with a CAS each is ten round trips to the shared store, and that is where the latency goes. The fix there is to batch the CAS attempts, or to use a per-server lease on a block of drivers the way the URL shortener leases a block of codes.

**Q13 (senior). A city goes dark because a Kafka partition is lost. What does this class model do?**
It degrades in a specific, describable way, and the reason it is worth answering at the class level is that the failure is invisible in the state machine. The location pings stop arriving, so the leases lapse, so `candidates` returns nothing, so `dispatch` walks to `NO_DRIVERS_FOUND` — which is *correct behaviour for the information available* and a total outage in reality. Nothing in `Trip` or `TripStateMachine` is wrong; the model has simply been told there are no drivers. That is the argument for making the ingest path's health a first-class signal: `DriverRegistry` should expose "how many drivers did I sweep this tick" and a sweep of an entire city's worth is an alarm, not a normal tick. The HLD names this failure; the class-level answer is that the lease mechanism cannot distinguish "no drivers" from "no data", and something outside the model has to.

**Q14. What would you test first?**
The ring search against brute force, because it is the only way to know the stopping rule is right — the demo does it at four values of k over 40 drivers. Then the CAS under a barrier, because a double-assignment bug is invisible without concurrency. Then the cancellation paths, and specifically the one where the trip was only ever *offered*: that is the case my first version got wrong, releasing only `trip.driver_id`, which is `None` for a trip that was never assigned, and leaving the driver stranded in `OFFERED` until the TTL happened to lapse.

---

### Pushback round

**P1. "Your search always scans nine cells, so a dense city centre pays 9x for nothing. That is a real cost you dismissed."**
It is a real cost, and the honest accounting is that it is proportional to *cell occupancy*, not to a constant: nine cells with three drivers each is 27 distance computations, which is nothing; nine cells with 5,000 drivers each is a different program. So the ring-1 guarantee is cheap exactly where the query is cheap and expensive where the index is already too coarse — which means the fix is the precision, not the ring rule. `PRECISION_BY_RADIUS` picks 6 for a 3 km search, and at 1.2 km cells a dense downtown is several cells rather than one, so the ring scan is bounded in practice. If it were not, the right answer is a quadtree that descends to a node holding k drivers and never examines a ring at all — and I would say that in the interview rather than defend the grid.

**P2. "`cancel_for_trip` scans every outstanding offer to find one trip's. With thousands of live offers that is a linear scan on the cancellation path."**
Correct, and it is the second time in this file that a `for` loop over a dictionary is doing the work of an index — the first being `get_for_trip`. The fix is a `_by_trip: dict[str, str]` index alongside `_by_driver`, maintained in `claim`, `respond`, `expire_outstanding` and `cancel_for_trip` — four places, which is exactly the cost of an index and exactly why it is worth stating rather than assuming. I left the scan because the offer count in the demo is small enough that the index would be untested code, but in a deployment with thousands of live offers per city, `cancel_for_trip` running on every rider cancellation is the hot path and the index is mandatory.

**P3. "`Driver.touch` takes `now` and `ttl` from the caller, but `DriverRegistry.ping` ignores it and uses its own clock. Two ways to refresh a lease is one too many."**
Agreed, and it is the kind of inconsistency that becomes a bug the first time somebody calls the wrong one. `ping` is the intended path — it re-files the driver in the index, which `touch` does not — and `touch` exists only as the field-level operation that `ping` and `release` both use. The correct cleanup is to make `touch` private to the registry (a leading underscore, or a module-private function) so the only public way to refresh a lease is `ping`, and the index can never be left stale by a caller who refreshed the lease directly. As written, a caller who finds `Driver.touch` first will use it, and will produce a driver whose lease is fresh and whose cell is wrong — which is bug #4 in the failure table, reached by a different road.

**P4. "You have `TripState` and `TripAction` defined in two code blocks. That is duplication, and the two could drift."**
They are defined twice on purpose — the §4 block is the interface listing and the §5 block must run standalone, which is a constraint this file shares with every problem in `LLD/problems/`. But the objection is right about the risk, and the failure would be silent: a second `TripState` with an extra member would make `TRANSITIONS` lookups return `None` for that member, so `apply_transition` would report "no such transition" for a state that exists — an error message pointing at the wrong thing. The real fix is structural: do not have two blocks, or generate the §5 block from the §4 one. The cheap mitigation, and the one actually in the file, is the comment above the enums that says they are repeated and why — so a reader editing one knows the other exists. A drift-detecting test is possible but would have to compare source text rather than runtime objects, since only one definition survives into the running module.

---

## 12. Cheat sheet

**In one paragraph.** A driver is never assigned — a driver is **offered**, and the offer is a lease with a TTL claimed by a compare-and-swap on the driver's status, so eight riders cannot be offered the same driver and a driver who never answers releases themselves. Drivers are found through a **geohash grid** searched ring by ring, and both of the search's rules are about when you may stop: ring 1 is always scanned, because a cell boundary is an encoding artefact and the nearest driver is often across the street from where the encoding puts them; and the search stops only when the k-th candidate is closer than any unscanned cell could be, because stopping at the first cell with a hit returns the far corner of a ring. The **matching strategy** is a seam because the objective is not distance — batching beats greedy 18 to 10 on the demo's score matrix. The **trip state machine** is a table keyed by `(state, action)` with an actor set, so an illegal transition and an unauthorised one are the same failed lookup and `allowed_actions` cannot disagree with the check. **Surge** is per-cell with a hysteresis band, and the multiplier is captured at request time because the rider agreed to a number.

**The five sentences that carry the design**

1. *"A driver is offered, not assigned — the offer is a lease, so a driver who never answers releases themselves."*
2. *"Ring 1 is always scanned: a cell boundary is an encoding artefact, not a fact about geography."*
3. *"Stop only when the k-th candidate is closer than the nearest possible cell in the next ring."*
4. *"The claim is a compare-and-swap — the check and the write are one operation under one lock."*
5. *"The objective is not distance, which is why batching beats greedy on the same two drivers."*

**The class table**

| Class | Its one job | The mistake it prevents |
|---|---|---|
| `GeoPoint` | Haversine, once, for everybody | The index and the fare disagreeing about distance |
| `GeoIndex` | Cells → drivers; the ring search | Missing the nearest driver across a boundary; stopping too early |
| `DriverRegistry` | Location pings, the lease, and the **CAS** | A stale driver in the index; two riders claiming one driver |
| `Driver` | Identity, status, availability lease | Rides offered to an app that died an hour ago |
| `MatchingStrategy` | Rank the candidates | An objective that is a product decision being hard-coded |
| `BatchedOptimal` | Score-based assignment over a batch | Greedy leaving 8 points on the table |
| `Offer` | A lease on a driver, with an expiry | An unanswered offer occupying a driver forever |
| `OfferService` | Claim atomically; release on decline, expiry, cancel | Double-assignment; a driver stranded by a cancelled trip |
| `Trip` | State, version, history, captured surge and quote | A fare that changes mid-ride; a transition applied twice |
| `apply_transition` | The table, in five lines | Illegal and unauthorised transitions being handled differently |
| `SurgeCalculator` | Per-cell multiplier with a hysteresis band | The price flickering while the rider decides |
| `FareEstimator` | Integer minor units, rounded once | A fare whose parts do not add up |
| `RideService` | Search → rank → offer → assign, in that order | The order living in three different places |

**The dispatch path, in order — say it in this order**

`candidates(ring search)` → `filter declined` → `strategy.rank` → **for each candidate:** `try_claim` (CAS; on `OfferLost`, next) → `offer` outstanding, trip still `MATCHING` → driver responds → `begin_trip` (CAS) → `trip.advance(ASSIGN, SYSTEM)` → `ASSIGNED`.

**The state machine, as a table**

| From | Action | To | Who |
|---|---|---|---|
| `MATCHING` | `ASSIGN` | `ASSIGNED` | SYSTEM |
| `MATCHING` | `GIVE_UP` | `NO_DRIVERS_FOUND` | SYSTEM |
| `MATCHING` | `CANCEL_RIDER` | `CANCELLED_BY_RIDER` | RIDER, SYSTEM |
| `ASSIGNED` | `DRIVER_ARRIVED` | `ARRIVING` | DRIVER, SYSTEM |
| `ASSIGNED` | `CANCEL_RIDER` / `CANCEL_DRIVER` | cancelled by that party | that party, SYSTEM |
| `ARRIVING` | `START` | `IN_PROGRESS` | DRIVER, SYSTEM |
| `ARRIVING` | `CANCEL_RIDER` / `CANCEL_DRIVER` | cancelled by that party | that party, SYSTEM |
| `IN_PROGRESS` | `COMPLETE` | `COMPLETED` | DRIVER, SYSTEM |
| `IN_PROGRESS` | `CANCEL_SYSTEM` | `CANCELLED_BY_RIDER` | SYSTEM only |

Note what is *absent*: there is no `(IN_PROGRESS, CANCEL_RIDER)` row, so a rider cannot cancel a ride in progress, and that is expressed as a missing row rather than a permission check.

**Failure modes to name before you are asked**

| Symptom | Cause |
|---|---|
| The nearest driver is never offered | Searching only the rider's own cell |
| A distant driver is offered while a closer one waits | Stopping at the first cell with a hit |
| Two riders told the same car is coming | A read-then-write claim instead of a CAS |
| A driver unavailable for ten minutes for no reason | An unanswered offer with no TTL |
| A driver invisible to everybody after a cancellation | Cancelling released only `trip.driver_id`, which is `None` for an offered trip |
| The same ride re-offered to the driver who just declined | No record of the refusal |
| A ride offered to somebody who went home | A boolean availability flag instead of a lease |
| The price flickers while the rider decides | Surge without a hysteresis band |
| The bill is higher than the quote | Surge recomputed at completion |
| The fare components do not sum to the total | Floats instead of integer minor units |
| A transition applied twice from the same state | A version read inside the method instead of passed in |

**Cross-links**

- **[The HLD this descends from](../HLD/06-uber-ride-hailing.md)** — capacity, the Kafka location ingest, the API surface, the matching at system level, surge at system level, and the failure catalogue (§6.1 double-assignment, §6.2 stale drivers, §6.4 the empty market, §6.9 surge oscillation). This file is the classes that make those answers true; it deliberately does not restate them.
- **[`LLD/problems/16-food-delivery.md`](../../LLD/problems/16-food-delivery.md)** — the deliberate contrast, and worth reading first. There, driver assignment is *not* a state, because kitchen progress is orthogonal to who is delivering; here `ASSIGNED` is a state, because a ride does not exist until a driver accepts. Same technique — a table-driven, actor-aware machine — used where it fits and refused where it does not.
- **[URL shortener LLD](01-url-shortener-lld.md)** — the range lease. That file mints codes in blocks to avoid a round trip per code; §6 here uses the same idea as the escape hatch for a distributed `try_claim` under load.
- **[WhatsApp LLD](02-whatsapp-lld.md)** — the other lease in this directory. The presence lease there and the driver lease here are the same argument: a process that cannot be trusted to say goodbye must be given an expiry instead of a flag.
- **[`LLD/problems/11-rate-limiter.md`](../../LLD/problems/11-rate-limiter.md)** — what you put in front of the location-ping endpoint, which is the highest-volume write in this system.
- **[`LLD/problems/05-elevator-system.md`](../../LLD/problems/05-elevator-system.md)** — a second state machine, and a different scheduling problem: the elevator's hard part is that its "matches" are pre-committed, which is what a matching system deliberately avoids.
- **[`LLD/problems/04-bookmyshow-seat-locking.md`](../../LLD/problems/04-bookmyshow-seat-locking.md)** — a third lease, on seats, and the closest analogue to the offer: a temporary hold that expires rather than a permanent reservation.
- **[The LLD question bank](../../Interview-Questions/08-lld-questions.md)** — the machine-coding questions this file's shape answers.
- **[The LLD cheat sheet](../../Cheat-Sheets/07-lld-cheatsheet.md)** — the condensed form of every file in this directory.

**Grounding note.** The reference repositories contain **no OOP/LLD material**, and the source transcripts cover ride-hailing at the system level without teaching a class design for matching. Everything above — the geohash ring search and its two stopping rules, the offer lease, the batched strategy, the actor-aware transition table, the surge band — is standard industry practice assembled for this problem, not material recovered from a transcript. Where the HLD file states a system-level property, this file is consistent with it and cross-links rather than repeats.


