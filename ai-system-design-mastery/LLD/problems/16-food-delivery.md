# LLD 16 — Food Delivery (Order Lifecycle)

> **Grounding note.** The reference repositories contain **no OOP/LLD material**; the source transcripts do not cover food delivery at the class level. This file is written from first principles and is not transcript-grounded. The answer shape follows the house template at [LLD/01 What is LLD and How to Answer](../01-what-is-lld-and-how-to-answer.md). The *system* view of this domain — dispatch at scale, geo-sharding, surge, the courier marketplace — belongs to [Uber Ride-Hailing](../../Case-Studies/HLD/06-uber-ride-hailing.md) at HLD level and to the [Uber Matching LLD](../../Case-Studies/LLD/03-uber-matching-lld.md) one level down; this file owns the **order aggregate**: its state machine, its money, its cancellation policy and the assignment race.

> **TL;DR** — Food delivery is the best LLD problem for one reason: **not every fact about an order is a state.** Kitchen progress (`PLACED → ACCEPTED → PREPARING → READY`) is a state machine; driver assignment is *orthogonal to it* — a driver can be assigned while the food is still being cooked, and forcing both onto one linear enum is the classic mistake in this problem. So the design is a table-driven `OrderState` machine whose rows say *(from-state, action) → (to-state, who may do it)*, plus a separately guarded `assign_driver` with a **version check** because two drivers accepting the same order is the race that matters. Around it: `Money` in **integer minor units** (never a float), a `LineItem` that snapshots the **price at add-to-cart time** so a menu edit cannot change an order already placed, an idempotency key on `place_order` so a double-tap cannot place two orders, and a `CancellationPolicy` strategy because "can I cancel and what do I get back?" is a policy that changes between markets.

> **60-second interview answer:** "An `Order` aggregate with a table-driven state machine: rows keyed by *(from-state, action)* mapping to a target state and the set of actors allowed to do it, so an illegal transition and an unauthorised transition are the same lookup. Driver assignment is deliberately **not** a state — it is a guarded field with a version check, because it can happen concurrently with the kitchen and because two drivers accepting the same order must resolve to exactly one. `Money` is integers in minor units, `LineItem` snapshots the unit price when it is added to the cart, `place_order` is idempotent on a client key, and cancellation is a policy object rather than an `if` ladder. The interesting tests are the race — eight drivers, one order, exactly one winner — and the transition table, exercised exhaustively."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **Which lifecycle states?** | The shape of the state machine | `PLACED → ACCEPTED → PREPARING → READY → PICKED_UP → DELIVERED`, plus `CANCELLED` and `REJECTED` |
| 2 | **Who may do what?** | Whether the machine is actor-aware | Yes — each transition names the actors allowed to trigger it |
| 3 | **Is driver assignment a state?** | The single most important modelling decision here | No — it is a guarded field, orthogonal to the kitchen |
| 4 | **Money type?** | Whether the design is financially correct | Integer minor units with a currency code; floats are rejected |
| 5 | **Can the menu change under a placed order?** | Whether prices are snapshotted | Snapshot the unit price at add-to-cart; the order is immutable in price |
| 6 | **What happens if the restaurant does not respond?** | Whether there is an accept timeout | A system-driven auto-cancel after a deadline |
| 7 | **Cancellation and refunds?** | Whether there is a policy object | A tiered policy: free before acceptance, fee after, no refund after pickup |
| 8 | **Double-tap on "Place order"?** | Idempotency | A client-supplied idempotency key; the same key returns the same order |
| 9 | **How is a driver chosen?** | Whether assignment is a strategy | A strategy port: nearest, least-loaded, batched |
| 10 | **One process or many?** | Whether the version check needs a lock | One aggregate per order with a lock; a distributed version uses a compare-and-set in the store |

### The defaults to assume, stated plainly

> "I'll assume a table-driven order state machine that is **actor-aware**, driver assignment as a **guarded field rather than a state**, `Money` as **integer minor units**, **price snapshots** on the cart and the order, an **idempotency key** on placement, a **tiered cancellation policy** as a strategy, an **accept timeout** driven by the system actor, and a **driver-assignment port**. Does that match, or do you want the lifecycle first and the rest as follow-ups?"

### Non-goals

- No real payments, no real geo-index (the [Uber matching LLD](../../Case-Studies/LLD/03-uber-matching-lld.md) owns the geo side).
- No persistence, no HTTP API, no UI.
- No surge pricing or courier incentives — pricing here is the order total, not the marketplace.

### The four constraints that shape the design

1. **Kitchen progress and driver assignment are orthogonal.** One linear enum cannot express "cooking and a driver is on the way".
2. **Every illegal move must be rejected for two different reasons** — wrong state, or wrong actor — and both must be checked in one place.
3. **Money must never be a float.** `0.1 + 0.2 != 0.3` is not a curiosity when it is somebody's total.
4. **Placement and assignment are both idempotency problems.** A double-tap and two drivers racing are the same class of bug, and both need an explicit guard.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `Money` | Frozen value object | Integer minor units + currency; add, subtract, multiply, compare | Formatting locale rules |
| `MenuItem` | Frozen value object | `id`, `name`, `price`, `available` | Cart state |
| `Menu` | Entity | The restaurant's items; `get`, `available_items` | Availability policy (sold-out windows) |
| `LineItem` | Frozen value object | `menu_item_id`, `name`, `unit_price` **snapshot**, `quantity`, `line_total` | Pricing rules |
| `Cart` | Mutable entity | Lines, add/remove/set-quantity, `subtotal`, `snapshot()` | Who pays, fees, taxes |
| `PricingStrategy` (ABC) | Strategy | `price(cart, distance_km, tip) -> PricingBreakdown` | Refunds |
| `PricingBreakdown` | Frozen value object | Subtotal, delivery fee, service fee, tax, tip, discount, total | — |
| `OrderActor` | Enum | `CUSTOMER`, `RESTAURANT`, `DRIVER`, `SYSTEM` | — |
| `OrderState` | Enum | The kitchen/delivery lifecycle; `is_terminal` | The transitions |
| `OrderAction` | Enum | `ACCEPT`, `REJECT`, `START_PREPARING`, `MARK_READY`, `PICK_UP`, `DELIVER`, `CANCEL` | — |
| `TRANSITIONS` | Table | `(state, action) -> (state, allowed actors)` — the machine | Domain preconditions |
| `OrderEvent` | Frozen value object | One history row: from, to, action, actor, at, note | — |
| `Order` | Aggregate root | State, lines, driver, history, version; `transition`, `assign_driver` | Choosing a driver, computing money |
| `OrderConflict` / `IllegalTransition` | Exceptions | Wrong actor/state; already assigned or stale version | — |
| `CancellationPolicy` (ABC) | Strategy | `evaluate(order) -> RefundDecision` | Executing the refund |
| `RefundDecision` | Frozen value object | Refundable amount, fee kept, the reason | — |
| `Driver` | Entity | `id`, `location`, `active_orders`, `available` | Assignment rules |
| `DriverAssigner` (ABC) | Strategy | `rank(order, drivers) -> list[Driver]` | Mutating the order |
| `DispatchService` | Service | The assignment **race**: version-checked, exactly one winner | ETA |
| `EtaEstimator` (ABC) | Port | `estimate(order, driver) -> timedelta` | — |
| `OrderStore` | Repository | Idempotency keys, orders by id | Business rules |
| `DeliveryService` | Façade | `place_order`, `accept`, `assign`, `pick_up`, `deliver`, `cancel` | — |

**The split to say out loud:** *"`Order` owns the state machine and the history; it does not own pricing, does not choose a driver, and does not decide refunds. Those are three strategies injected into the service, because each of them changes for a business reason rather than a modelling one."*

---

## 3. Class diagram (ASCII)

```
 ┌────────────┐   ┌──────────────┐   ┌────────────────────────┐
 │  Money     │   │  MenuItem    │   │      LineItem          │
 │ «value»    │   │ «value»      │   │ «value»                │
 │ minor:int  │   │ id, name     │   │ menu_item_id, name     │
 │ currency   │   │ price: Money │   │ unit_price: Money  ←   │
 │ +add +sub  │   │ available    │   │   SNAPSHOT at add time │
 │ +mul +cmp  │   └──────┬───────┘   │ quantity, line_total   │
 └────────────┘          │           └────────────────────────┘
        ▲                │                    ▲
        │           ┌────┴─────┐              │ produces
        │           │  Menu    │       ┌──────┴───────┐
        │           │ items{}  │──────►│    Cart      │
        │           └──────────┘  add  │ lines[]      │
        │                              │ +add(item,q) │
        │                              │ +snapshot()  │
        │                              └──────┬───────┘
        │                                     │ priced by
        │                      ┌──────────────┴───────────────┐
        │                      │   «interface» PricingStrategy │
        │                      │ +price(cart, km, tip)         │
        │                      └──────────────▲───────────────┘
        │                                     │
        │                          ┌──────────┴──────────┐
        │                          │  StandardPricing    │
        │                          │  fee · svc rate ·   │
        │                          │  tax rate           │
        │                          └─────────────────────┘
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                              Order  (aggregate root)                      │
│───────────────────────────────────────────────────────────────────────────│
│ id · customer_id · restaurant_id · lines[] · pricing : PricingBreakdown   │
│ state : OrderState        ← the KITCHEN / LIFECYCLE machine               │
│ driver_id : Optional[str] ← NOT a state: orthogonal, separately guarded   │
│ version : int             ← the compare-and-set token                     │
│ history : [OrderEvent]                                                    │
│───────────────────────────────────────────────────────────────────────────│
│ +transition(action, actor) -> OrderEvent    (table + guard)               │
│ +assign_driver(driver, expected_version)    (CAS: exactly one winner)     │
│ +allowed_actions(actor) -> [OrderAction]    (the same table, per actor)   │
│ +can_cancel() +is_terminal()                                              │
└───────────┬───────────────────────────────────────────┬───────────────────┘
            │ consults the table                        │ guarded by
            ▼                                           ▼
  ┌──────────────────────────────┐        ┌────────────────────────────────┐
  │ TRANSITIONS                  │        │ OrderLock                      │
  │ (state, action) ->           │        │  makes the version CHECK and   │
  │   (state, {actors})          │        │  the MUTATION atomic — a CAS   │
  │                              │        │  without atomicity is a race   │
  │ PLACED  +ACCEPT   → ACCEPTED │        │  that passes code review       │
  │ PLACED  +CANCEL   → CANCELLED│        └────────────────────────────────┘
  │ ACCEPTED+START_PREP→PREPARING│
  │ PREPARING+MARK_READY→READY   │        ┌────────────────────────────────┐
  │ READY   +PICK_UP  → PICKED_UP│        │  DispatchService               │
  │ PICKED_UP+DELIVER → DELIVERED│        │  rank via DriverAssigner, then │
  │ …                            │        │  assign under the lock         │
  └──────────────────────────────┘        └───────────────▲────────────────┘
                                                          │
                                          ┌───────────────┴────────────────┐
                                          │  «interface» DriverAssigner    │
                                          │ +rank(order, drivers)          │
                                          └───────▲──────────────┬─────────┘
                                     ┌────────────┴───┐   ┌──────┴──────────┐
                                     │ NearestAssigner│   │LeastLoaded      │
                                     └────────────────┘   └─────────────────┘

 ┌────────────────────────────┐        ┌──────────────────────────────────┐
 │  «interface» Cancellation  │        │        DeliveryService  (façade) │
 │      Policy                │        │──────────────────────────────────│
 │ +evaluate(order)           │        │ +place_order(key, cart, ...)     │
 └─────────▲──────────────────┘        │   idempotent on `key`            │
           │                           │ +accept/assign/pick_up/deliver   │
 ┌─────────┴──────────────────┐        │ +cancel(order_id, actor)         │
 │ TieredCancellationPolicy   │◄───────│                                  │
 │ free · fee · no refund     │        └──────────────────────────────────┘
 └────────────────────────────┘
```

**Read it this way:** `Order` composes values and consults one table. Everything that varies for a business reason — pricing, assignment ranking, refunds, ETAs — is a strategy injected into the service, not a branch inside the aggregate.

---

## 4. Interfaces / abstract classes / enums

```python
class PricingStrategy(ABC):
    @abstractmethod
    def price(self, cart: Cart, distance_km: float, tip: Money) -> PricingBreakdown: ...


class CancellationPolicy(ABC):
    @abstractmethod
    def evaluate(self, order: "Order") -> RefundDecision:
        """Pure: decide the refund, do not perform it. The same split as the
        rate limiter's Decision and snake-and-ladder's MoveResolution."""


class DriverAssigner(ABC):
    @abstractmethod
    def rank(self, order: "Order", drivers: list["Driver"]) -> list["Driver"]:
        """Order the candidates. It RANKS; it does not assign — because
        assigning is the race, and the race belongs in one place."""


class EtaEstimator(ABC):
    @abstractmethod
    def estimate(self, order: "Order", driver: Optional["Driver"]) -> int:
        """Minutes. A port, so a test asserts an ETA instead of waiting for one."""
```

**Four contract notes worth saying out loud:**

- **`rank` and `assign` are separate.** A ranker that also assigns would put the race in every strategy; keeping `rank` pure means there is exactly one place where "exactly one driver wins" is enforced.
- **`evaluate` is pure.** "What would the refund be?" and "perform the refund" are different operations with different failure modes, and only the second needs a payment provider.
- **`OrderState` is not a single linear enum for everything.** Kitchen progress is a state; driver assignment is a field.
- **`OrderActor` is part of the transition key**, not a check bolted on afterwards. That is what makes "who may do this" a table lookup rather than a ladder of `if`s.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** Interview languages vary — in Java this is an `EnumMap` of transitions, a `record` for `Money` and a `ConcurrentHashMap` for the store; in Go, a map keyed by a struct and an `sync.Mutex`; in C++, `virtual` strategies and `std::map`. The **table-driven actor-aware machine, the orthogonal assignment, and the version-checked race** are the design; the syntax is not. Python 3.10+ for `X | None`.

```python
"""food_delivery.py — an order aggregate with an actor-aware state machine.

Design summary
--------------
Money              : integer MINOR UNITS. Floats are not accepted anywhere, and
                     a test proves 0.1+0.2 == 0.3 when the unit is a cent.
LineItem / Cart    : the unit price is SNAPSHOT when the item is added, so a
                     menu edit after the fact cannot change a placed order.
Order              : the aggregate. `state` is the kitchen/delivery lifecycle;
                     `driver_id` is NOT a state — a driver can be assigned while
                     the food is still cooking, so the two are orthogonal and
                     forcing them into one enum is the classic mistake here.
TRANSITIONS        : (from_state, action) -> (to_state, {actors allowed}). An
                     illegal transition and an unauthorised one are the SAME
                     lookup, which is why the actor is part of the key.
assign_driver      : a guarded field with a VERSION check — the compare-and-set
                     that makes two racing drivers resolve to one winner.
OrderLock          : makes that check and the mutation atomic. A version check
                     without atomicity is a race that passes code review.
CancellationPolicy : a strategy, because "what do I get back?" changes by market.
DeliveryService    : the façade; `place_order` is idempotent on a client key.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Optional


# --------------------------------------------------------------------------
# 1. Money — integers only
# --------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Money:
    """Integer minor units plus a currency.

    There is no float path in or out. `Money.from_units(0.1) + Money.from_units(0.2)`
    is exactly 0.30 because the arithmetic happens in cents, and the demo shows
    the float version failing on the identical inputs — which is the whole
    argument for this class existing.
    """

    minor: int
    currency: str = "USD"

    @staticmethod
    def zero(currency: str = "USD") -> "Money":
        return Money(0, currency)

    @staticmethod
    def from_units(units: float, currency: str = "USD") -> "Money":
        """Only for parsing input. Rounds half-up to the nearest minor unit."""
        # 0.1 + 0.2 in binary floating point is 0.30000000000000004, so this
        # rounds rather than truncates: int(0.29 * 100) is 28, not 29.
        return Money(int(round(units * 100)), currency)

    @staticmethod
    def from_str(text: str, currency: str = "USD") -> "Money":
        """Parse "12.34" without touching a float at all."""
        negative = text.strip().startswith("-")
        body = text.strip().lstrip("-")
        if "." in body:
            whole, _, frac = body.partition(".")
        else:
            whole, frac = body, ""
        frac = (frac + "00")[:2]
        if not whole.isdigit() or not frac.isdigit():
            raise ValueError(f"cannot parse money: {text!r}")
        minor = int(whole) * 100 + int(frac)
        return Money(-minor if negative else minor, currency)

    # -- arithmetic --------------------------------------------------------
    def _check(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError(f"currency mismatch: {self.currency} vs {other.currency}")

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.minor + other.minor, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.minor - other.minor, self.currency)

    def __mul__(self, factor: int) -> "Money":
        if not isinstance(factor, int):
            raise TypeError("multiply money by an integer quantity, not a float")
        return Money(self.minor * factor, self.currency)

    __rmul__ = __mul__

    def percent(self, rate: float) -> "Money":
        """The one place a float is allowed, and it is rounded immediately."""
        return Money(int(round(self.minor * rate)), self.currency)

    def allocate(self, weights: Iterable[int]) -> list["Money"]:
        """Split without losing a minor unit: the remainder goes to the first
        shares, one unit each. Naive rounding loses cents, and lost cents are
        how a refund reconciliation fails."""
        ratios = list(weights)
        total = sum(ratios)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        shares = [self.minor * r // total for r in ratios]
        remainder = self.minor - sum(shares)
        for i in range(remainder):
            shares[i % len(shares)] += 1
        return [Money(s, self.currency) for s in shares]

    def __neg__(self) -> "Money":
        return Money(-self.minor, self.currency)

    @property
    def is_zero(self) -> bool:
        return self.minor == 0

    @property
    def units(self) -> float:
        """For display only. Never feed this back into arithmetic."""
        return self.minor / 100

    def __str__(self) -> str:
        sign = "-" if self.minor < 0 else ""
        minor = abs(self.minor)
        return f"{sign}{self.currency} {minor // 100}.{minor % 100:02d}"


# --------------------------------------------------------------------------
# 2. Menu, cart, and the price snapshot
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MenuItem:
    id: str
    name: str
    price: Money
    available: bool = True


class Menu:
    def __init__(self, restaurant_id: str, items: Iterable[MenuItem]) -> None:
        self.restaurant_id = restaurant_id
        self.items: dict[str, MenuItem] = {}
        for item in items:
            if item.id in self.items:
                raise ValueError(f"duplicate menu item id: {item.id}")
            self.items[item.id] = item

    def get(self, item_id: str) -> MenuItem:
        try:
            return self.items[item_id]
        except KeyError:
            raise KeyError(f"{item_id} is not on the menu") from None

    def set_available(self, item_id: str, available: bool) -> None:
        """Restaurants mark items sold out mid-service. Because the cart
        snapshots prices, this cannot retroactively change an order."""
        item = self.get(item_id)
        self.items[item_id] = replace(item, available=available)

    def available_items(self) -> list[MenuItem]:
        return [i for i in self.items.values() if i.available]


class ItemUnavailable(Exception):
    pass


@dataclass(frozen=True)
class LineItem:
    menu_item_id: str
    name: str
    unit_price: Money          # SNAPSHOT, not a reference to the menu
    quantity: int

    @property
    def line_total(self) -> Money:
        return self.unit_price * self.quantity


class Cart:
    """The cart snapshots the price when an item is added. The alternative —
    reading the price at checkout — means a menu edit between add and checkout
    silently changes what the customer is charged, which is both a bug and, in
    most jurisdictions, not legal."""

    def __init__(self, restaurant_id: str, currency: str = "USD") -> None:
        self.restaurant_id = restaurant_id
        self.currency = currency
        self.lines: list[LineItem] = []

    def add(self, menu: Menu, item_id: str, quantity: int = 1) -> LineItem:
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        item = menu.get(item_id)
        if menu.restaurant_id != self.restaurant_id:
            raise ValueError("that item belongs to a different restaurant")
        if not item.available:
            raise ItemUnavailable(f"{item.name} is sold out")
        for index, line in enumerate(self.lines):
            if line.menu_item_id == item_id:
                merged = replace(line, quantity=line.quantity + quantity)
                self.lines[index] = merged
                return merged
        line = LineItem(item_id, item.name, item.price, quantity)
        self.lines.append(line)
        return line

    def set_quantity(self, item_id: str, quantity: int) -> None:
        if quantity <= 0:
            self.remove(item_id)
            return
        for index, line in enumerate(self.lines):
            if line.menu_item_id == item_id:
                self.lines[index] = replace(line, quantity=quantity)
                return
        raise KeyError(f"{item_id} is not in the cart")

    def remove(self, item_id: str) -> None:
        self.lines = [l for l in self.lines if l.menu_item_id != item_id]

    @property
    def subtotal(self) -> Money:
        total = Money.zero(self.currency)
        for line in self.lines:
            total = total + line.line_total
        return total

    @property
    def is_empty(self) -> bool:
        return not self.lines

    def snapshot(self) -> tuple[LineItem, ...]:
        """An immutable copy. The order holds this, so later cart edits cannot
        reach back into a placed order."""
        return tuple(self.lines)


# --------------------------------------------------------------------------
# 3. Pricing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PricingBreakdown:
    subtotal: Money
    delivery_fee: Money
    service_fee: Money
    tax: Money
    tip: Money
    discount: Money
    total: Money

    def __str__(self) -> str:
        return (f"sub {self.subtotal} + del {self.delivery_fee} + svc {self.service_fee} "
                f"+ tax {self.tax} + tip {self.tip} - disc {self.discount} "
                f"= {self.total}")


class PricingStrategy(ABC):
    @abstractmethod
    def price(self, cart: Cart, distance_km: float, tip: Money) -> PricingBreakdown: ...


class StandardPricing(PricingStrategy):
    def __init__(self, base_delivery: Money, per_km: Money,
                 service_rate: float, tax_rate: float) -> None:
        self.base_delivery = base_delivery
        self.per_km = per_km
        self.service_rate = service_rate
        self.tax_rate = tax_rate

    def price(self, cart: Cart, distance_km: float, tip: Money) -> PricingBreakdown:
        subtotal = cart.subtotal
        # Per-km is held in MINOR UNITS, so the only fractional step is the
        # multiplication by a fractional distance. Round ONCE, here, and never
        # carry the result as a float.
        delivery = self.base_delivery + Money(
            int(round(distance_km * self.per_km.minor)), self.base_delivery.currency)
        service = subtotal.percent(self.service_rate)
        # Tax applies to the food and the service fee, not to the tip — the
        # order of these four lines is a policy decision, so they are spelled
        # out rather than folded into one expression.
        tax = (subtotal + service).percent(self.tax_rate)
        discount = Money.zero(subtotal.currency)
        total = subtotal + delivery + service + tax + tip - discount
        return PricingBreakdown(subtotal, delivery, service, tax, tip, discount, total)


# --------------------------------------------------------------------------
# 4. The order state machine
# --------------------------------------------------------------------------

class OrderActor(Enum):
    CUSTOMER = "customer"
    RESTAURANT = "restaurant"
    DRIVER = "driver"
    SYSTEM = "system"          # timeouts, auto-cancels, reconciliation


class OrderState(Enum):
    PLACED = "placed"
    ACCEPTED = "accepted"
    PREPARING = "preparing"
    READY = "ready"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderState.DELIVERED, OrderState.CANCELLED,
                        OrderState.REJECTED)


class OrderAction(Enum):
    CREATE = "create"          # the origin row only; it has no row in the table
    ACCEPT = "accept"
    REJECT = "reject"
    START_PREPARING = "start_preparing"
    MARK_READY = "mark_ready"
    PICK_UP = "pick_up"
    DELIVER = "deliver"
    CANCEL = "cancel"


# The machine, as data. An illegal transition and an UNAUTHORISED one are the
# same lookup, because the actor is part of the key.
TRANSITIONS: dict[tuple[OrderState, OrderAction],
                  tuple[OrderState, frozenset[OrderActor]]] = {
    (OrderState.PLACED, OrderAction.ACCEPT):
        (OrderState.ACCEPTED, frozenset({OrderActor.RESTAURANT})),
    (OrderState.PLACED, OrderAction.REJECT):
        (OrderState.REJECTED, frozenset({OrderActor.RESTAURANT})),
    (OrderState.PLACED, OrderAction.CANCEL):
        (OrderState.CANCELLED, frozenset({OrderActor.CUSTOMER, OrderActor.SYSTEM})),

    (OrderState.ACCEPTED, OrderAction.START_PREPARING):
        (OrderState.PREPARING, frozenset({OrderActor.RESTAURANT})),
    (OrderState.ACCEPTED, OrderAction.CANCEL):
        (OrderState.CANCELLED, frozenset({OrderActor.CUSTOMER, OrderActor.SYSTEM})),

    (OrderState.PREPARING, OrderAction.MARK_READY):
        (OrderState.READY, frozenset({OrderActor.RESTAURANT})),
    (OrderState.PREPARING, OrderAction.CANCEL):
        # Once the kitchen has started, the customer may still cancel, but the
        # CancellationPolicy decides what comes back. The machine says WHO may
        # act; the policy says WHAT IT COSTS. Those are different questions.
        (OrderState.CANCELLED,
         frozenset({OrderActor.CUSTOMER, OrderActor.RESTAURANT, OrderActor.SYSTEM})),

    (OrderState.READY, OrderAction.PICK_UP):
        (OrderState.PICKED_UP, frozenset({OrderActor.DRIVER})),
    (OrderState.READY, OrderAction.CANCEL):
        # Only the system cancels a ready order — the food exists and is going
        # to waste, so it is a support action rather than a self-service one.
        (OrderState.CANCELLED, frozenset({OrderActor.SYSTEM})),

    (OrderState.PICKED_UP, OrderAction.DELIVER):
        (OrderState.DELIVERED, frozenset({OrderActor.DRIVER})),
}


class IllegalTransition(Exception):
    """The (state, action) pair is not in the machine, or the actor is not
    allowed to perform it."""


class OrderConflict(Exception):
    """The order changed under us: a stale version, or a field already set."""


@dataclass(frozen=True)
class OrderEvent:
    index: int
    from_state: OrderState
    to_state: OrderState
    action: OrderAction
    actor: OrderActor
    at: float
    note: str = ""

    def __str__(self) -> str:
        return (f"#{self.index} {self.from_state.value} --{self.action.value}"
                f"({self.actor.value})--> {self.to_state.value}"
                + (f" [{self.note}]" if self.note else ""))


@dataclass(frozen=True)
class RefundDecision:
    refund: Money
    fee_kept: Money
    reason: str

    def __str__(self) -> str:
        return f"refund {self.refund} (kept {self.fee_kept}): {self.reason}"


class CancellationPolicy(ABC):
    @abstractmethod
    def evaluate(self, order: "Order") -> RefundDecision: ...


class TieredCancellationPolicy(CancellationPolicy):
    """Free before the restaurant accepts, a fee once it has, nothing back
    once the driver has the food. The thresholds are constructor arguments
    because they differ by market — and a hard-coded `if` ladder is exactly
    the thing that cannot be localised."""

    def __init__(self, fee_after_accept: Optional[Money] = None) -> None:
        self.fee_after_accept = fee_after_accept

    def evaluate(self, order: "Order") -> RefundDecision:
        paid = order.pricing.total
        zero = Money.zero(paid.currency)

        if order.state in (OrderState.PLACED, OrderState.REJECTED):
            return RefundDecision(paid, zero, "cancelled before the restaurant accepted")
        if order.state is OrderState.ACCEPTED:
            fee = self.fee_after_accept if self.fee_after_accept is not None \
                else order.pricing.delivery_fee
            if fee.minor > paid.minor:
                fee = paid
            return RefundDecision(paid - fee, fee,
                                  "cancelled after acceptance: delivery fee retained")
        if order.state is OrderState.PREPARING:
            # Half the food value back: the kitchen has started but the meal is
            # not made. A deliberate policy, not a formula.
            half = order.pricing.subtotal.percent(0.5)
            refund = paid - half
            if refund.minor < 0:
                refund = zero
            return RefundDecision(refund, half, "cancelled mid-preparation: 50% retained")
        if order.state in (OrderState.READY, OrderState.PICKED_UP, OrderState.DELIVERED):
            return RefundDecision(zero, paid, "food prepared or delivered: no refund")
        if order.state is OrderState.CANCELLED:
            return RefundDecision(zero, zero, "already cancelled")
        return RefundDecision(zero, paid, f"no policy for {order.state.value}")


# --------------------------------------------------------------------------
# 5. The Order aggregate
# --------------------------------------------------------------------------

class Order:
    def __init__(self, order_id: str, customer_id: str, restaurant_id: str,
                 lines: tuple[LineItem, ...], pricing: PricingBreakdown,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not lines:
            raise ValueError("an order needs at least one line")
        self.id = order_id
        self.customer_id = customer_id
        self.restaurant_id = restaurant_id
        self.lines = lines
        self.pricing = pricing
        self.state = OrderState.PLACED
        self.driver_id: Optional[str] = None      # NOT a state — see §1
        self.version = 0
        self.history: list[OrderEvent] = []
        self.clock = clock
        self.accept_deadline: Optional[float] = None
        # The origin row. It is deliberately NOT a transition: there is no
        # (state, CREATE) row in the table, because an order is not created
        # FROM a state — the history simply starts here.
        self._record(OrderState.PLACED, OrderState.PLACED, OrderAction.CREATE,
                     OrderActor.CUSTOMER, note="created")

    # -- history -----------------------------------------------------------
    def _record(self, from_state: OrderState, to_state: OrderState,
                action: OrderAction, actor: OrderActor, note: str = "") -> OrderEvent:
        event = OrderEvent(len(self.history), from_state, to_state, action,
                           actor, self.clock(), note)
        self.history.append(event)
        return event

    # -- the machine -------------------------------------------------------
    def transition(self, action: OrderAction, actor: OrderActor,
                   note: str = "") -> OrderEvent:
        """One lookup, two failure modes: the pair is not in the table, or the
        actor is not in the pair's allowed set."""
        before = self.state
        row = TRANSITIONS.get((before, action))
        if row is None:
            raise IllegalTransition(
                f"{action.value} is not legal from {before.value}")
        target, allowed = row
        if actor not in allowed:
            raise IllegalTransition(
                f"{actor.value} may not {action.value} an order in {before.value} "
                f"(allowed: {', '.join(sorted(a.value for a in allowed))})")
        self.state = target
        self.version += 1
        return self._record(before, target, action, actor, note)

    def allowed_actions(self, actor: OrderActor) -> list[OrderAction]:
        """Derived from the same table, so a UI cannot offer an action the
        aggregate would reject — the classic 'the button was there but the
        server said no' bug."""
        return [action for (state, action), (_, allowed) in TRANSITIONS.items()
                if state is self.state and actor in allowed]

    def can_accept(self) -> bool:
        return (self.state, OrderAction.ACCEPT) in TRANSITIONS

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    # -- assignment: a guarded field, not a state --------------------------
    def assign_driver(self, driver: "Driver", expected_version: int) -> None:
        """The compare-and-set.

        Three guards, and all three matter: a stale version (somebody else
        changed the order), an already-assigned order (the race between two
        drivers), and a terminal order (a driver accepting a cancelled order).
        The version check MUST be atomic with the mutation — see OrderLock.
        """
        if self.version != expected_version:
            raise OrderConflict(
                f"order {self.id} moved from version {expected_version} "
                f"to {self.version}")
        if self.driver_id is not None:
            raise OrderConflict(
                f"order {self.id} is already assigned to {self.driver_id}")
        if self.is_terminal:
            raise OrderConflict(f"order {self.id} is {self.state.value}")
        self.driver_id = driver.id
        self.version += 1
        self._record(self.state, self.state, OrderAction.ACCEPT, OrderActor.DRIVER,
                     note=f"driver {driver.id} assigned")

    def unassign_driver(self, expected_version: int, note: str = "") -> None:
        """Reassignment after a driver drops out. Same CAS, opposite direction."""
        if self.version != expected_version:
            raise OrderConflict("stale version")
        if self.driver_id is None:
            raise OrderConflict("no driver assigned")
        if self.is_terminal:
            raise OrderConflict("the order is finished")
        previous = self.driver_id
        self.driver_id = None
        self.version += 1
        self._record(self.state, self.state, OrderAction.CANCEL, OrderActor.SYSTEM,
                     note=note or f"driver {previous} unassigned")

    # -- the guard the table cannot express --------------------------------
    def pick_up(self, actor: OrderActor) -> OrderEvent:
        """PICK_UP needs a driver AND a driver must be on the order.

        The table says where and who; it cannot say 'and there must be a
        driver assigned', because that is a fact about another field. This is
        what guards are for, and keeping them in one method is why the table
        stays a table instead of becoming a matrix of special cases.
        """
        if self.driver_id is None:
            raise OrderConflict("an order cannot be picked up with no driver assigned")
        if actor is not OrderActor.DRIVER:
            raise IllegalTransition("only a driver picks up an order")
        return self.transition(OrderAction.PICK_UP, actor,
                               note=f"picked up by {self.driver_id}")

    # -- timeout -----------------------------------------------------------
    def arm_accept_timeout(self, timeout_seconds: float) -> None:
        self.accept_deadline = self.clock() + timeout_seconds

    def check_accept_timeout(self) -> Optional[OrderEvent]:
        """Called by the system actor. A restaurant that does not respond must
        not hold the order open forever, and the deadline lives on the order
        so a sweeper can find it."""
        if self.accept_deadline is None or self.state is not OrderState.PLACED:
            return None
        if self.clock() < self.accept_deadline:
            return None
        return self.transition(OrderAction.CANCEL, OrderActor.SYSTEM,
                               note="restaurant did not respond in time")

    def render(self) -> str:
        lines = [f"order {self.id} [{self.state.value}] v{self.version}",
                 f"  driver: {self.driver_id or '-'}",
                 f"  {self.pricing}"]
        for event in self.history:
            lines.append(f"  {event}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 6. Assignment, drivers and the race
# --------------------------------------------------------------------------

@dataclass
class Driver:
    id: str
    distance_km: float = 0.0
    active_orders: int = 0
    available: bool = True

    def __str__(self) -> str:
        return f"{self.id}({self.distance_km}km, {self.active_orders} active)"


class DriverAssigner(ABC):
    @abstractmethod
    def rank(self, order: Order, drivers: list[Driver]) -> list[Driver]: ...


class NearestAssigner(DriverAssigner):
    def rank(self, order: Order, drivers: list[Driver]) -> list[Driver]:
        return sorted([d for d in drivers if d.available],
                      key=lambda d: (d.distance_km, d.active_orders, d.id))


class LeastLoadedAssigner(DriverAssigner):
    def rank(self, order: Order, drivers: list[Driver]) -> list[Driver]:
        return sorted([d for d in drivers if d.available],
                      key=lambda d: (d.active_orders, d.distance_km, d.id))


class DispatchService:
    """Owns the race. The ranker says who to try; this says who WON, and it is
    the only place that decides — because two rankers racing to assign is the
    same bug in two places."""

    def __init__(self, assigner: DriverAssigner,
                 lock_factory: Optional[Callable[[Order], "OrderLock"]] = None) -> None:
        self.assigner = assigner
        self._lock_factory = lock_factory if lock_factory is not None \
            else lambda order: OrderLock(order)

    def offer(self, order: Order, drivers: list[Driver],
              expected_version: Optional[int] = None) -> list[tuple[Driver, bool]]:
        """Offer the order to candidates in rank order and let the FIRST one
        win. In production this is a push to each driver's device and the
        acceptance arrives as a callback — the shape is identical, and the
        version check is what makes a late second acceptance fail rather than
        double-assign.

        `expected_version` defaults to the version the server sees now, which
        is right for a server-initiated push. Every offer in one round must
        pass the SAME version, so the second acceptance is rejected as stale
        rather than succeeding against a version it never saw.
        """
        if expected_version is None:
            expected_version = order.version
        lock = self._lock_factory(order)
        results: list[tuple[Driver, bool]] = []
        for driver in self.assigner.rank(order, drivers):
            try:
                lock.assign_driver(driver, expected_version)
                results.append((driver, True))
                break                       # one winner, and we stop offering
            except OrderConflict:
                results.append((driver, False))
        return results

    def accept(self, order: Order, driver: Driver, lock: "OrderLock",
               expected_version: int) -> bool:
        """The callback path: a driver taps Accept, presenting the version that
        came with the offer. Returns True for the winner and False for every
        loser, which is what the device should show as 'someone else took it'.

        `expected_version` is a parameter rather than `order.version` read in
        here, because reading it here would defeat the entire CAS: the driver
        must present the version they were OFFERED, not whatever the order
        happens to be at now. A version read at call time is always current,
        so it always passes the first guard — which is the bug this parameter
        exists to prevent.
        """
        try:
            lock.assign_driver(driver, expected_version)
            return True
        except OrderConflict:
            return False


class OrderLock:
    """Makes the version CHECK and the MUTATION atomic.

    This is the point that is easy to miss: a compare-and-set is only a
    compare-and-set if nothing can interleave between the compare and the set.
    Two threads that both read version 3, both find it equal, and both write
    have produced two winners — and the code looks correct.
    """

    def __init__(self, order: Order) -> None:
        self.order = order
        self._lock = threading.Lock()

    def assign_driver(self, driver: Driver, expected_version: int) -> None:
        with self._lock:
            self.order.assign_driver(driver, expected_version)

    def transition(self, action: OrderAction, actor: OrderActor,
                   note: str = "") -> OrderEvent:
        with self._lock:
            return self.order.transition(action, actor, note)

    def pick_up(self, actor: OrderActor) -> OrderEvent:
        """The guard runs INSIDE the lock. Checking 'is a driver assigned' and
        then transitioning outside it would be a check-then-act race: an
        unassign could land in between, and the order would be picked up with
        no driver — the exact impossible state the guard exists to prevent."""
        with self._lock:
            return self.order.pick_up(actor)

    def snapshot(self) -> tuple:
        with self._lock:
            return (self.order.state, self.order.driver_id, self.order.version)


class EtaEstimator(ABC):
    @abstractmethod
    def estimate(self, order: Order, driver: Optional[Driver]) -> int: ...


class SimpleEtaEstimator(EtaEstimator):
    """Preparation time, plus travel, plus a pickup buffer. A port, so a test
    asserts an ETA instead of waiting for one — the same argument as the
    injected clock in [the rate limiter](11-rate-limiter.md)."""

    def __init__(self, prep_minutes: int = 15, speed_km_per_min: float = 0.5,
                 pickup_buffer: int = 3) -> None:
        self.prep_minutes = prep_minutes
        self.speed = speed_km_per_min
        self.buffer = pickup_buffer

    def estimate(self, order: Order, driver: Optional[Driver]) -> int:
        if order.state in (OrderState.DELIVERED, OrderState.CANCELLED,
                           OrderState.REJECTED):
            return 0
        if order.state is OrderState.READY:
            prep = 0
        elif order.state in (OrderState.PICKED_UP,):
            prep = 0
        else:
            prep = self.prep_minutes
        travel = 0 if driver is None else int(driver.distance_km / self.speed)
        if order.state is OrderState.PICKED_UP:
            buffer = 0
        else:
            buffer = self.buffer
        return prep + travel + buffer


# --------------------------------------------------------------------------
# 7. Store and façade
# --------------------------------------------------------------------------

class OrderStore:
    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._idempotency: dict[str, str] = {}          # client key -> order id
        self._sequence = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        with self._lock:
            self._sequence += 1
            return f"ord-{self._sequence:06d}"

    def put(self, order: Order) -> None:
        with self._lock:
            self._orders[order.id] = order

    def get(self, order_id: str) -> Order:
        with self._lock:
            try:
                return self._orders[order_id]
            except KeyError:
                raise KeyError(f"no such order: {order_id}") from None

    def find_by_key(self, key: str) -> Optional[Order]:
        with self._lock:
            order_id = self._idempotency.get(key)
            return self._orders[order_id] if order_id else None

    def remember(self, key: str, order_id: str) -> None:
        with self._lock:
            self._idempotency[key] = order_id


class DeliveryService:
    """The façade. Every method is one of: idempotent on a key, or a CAS."""

    def __init__(self, store: OrderStore, pricing: PricingStrategy,
                 policy: CancellationPolicy, dispatch: DispatchService,
                 eta: EtaEstimator,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.pricing = pricing
        self.policy = policy
        self.dispatch = dispatch
        self.eta = eta
        self.clock = clock
        self._place_lock = threading.Lock()
        self._locks: dict[str, OrderLock] = {}

    def lock_for(self, order: Order) -> OrderLock:
        with self._place_lock:
            if order.id not in self._locks:
                self._locks[order.id] = OrderLock(order)
            return self._locks[order.id]

    # -- placement ---------------------------------------------------------
    def place_order(self, idempotency_key: str, customer_id: str, cart: Cart,
                    distance_km: float, tip: Optional[Money] = None,
                    accept_timeout: float = 120.0) -> Order:
        """Idempotent on the client key.

        The check and the insert share ONE lock, for the same reason the
        assignment CAS does: two threads that both find no existing order and
        both create one have produced two orders from one tap.
        """
        if not idempotency_key:
            raise ValueError("an idempotency key is required")
        if cart.is_empty:
            raise ValueError("cannot place an empty order")

        with self._place_lock:
            existing = self.store.find_by_key(idempotency_key)
            if existing is not None:
                return existing
            total_tip = tip if tip is not None else Money.zero(cart.currency)
            pricing = self.pricing.price(cart, distance_km, total_tip)
            order = Order(self.store.next_id(), customer_id, cart.restaurant_id,
                          cart.snapshot(), pricing, clock=self.clock)
            order.arm_accept_timeout(accept_timeout)
            self.store.put(order)
            self.store.remember(idempotency_key, order.id)
            return order

    # -- lifecycle ---------------------------------------------------------
    def accept(self, order_id: str, restaurant_id: str) -> OrderEvent:
        order = self.store.get(order_id)
        if order.restaurant_id != restaurant_id:
            raise IllegalTransition("that order belongs to another restaurant")
        event = self.lock_for(order).transition(OrderAction.ACCEPT, OrderActor.RESTAURANT)
        order.accept_deadline = None          # accepted, so the deadline is moot
        return event

    def start_preparing(self, order_id: str) -> OrderEvent:
        order = self.store.get(order_id)
        return self.lock_for(order).transition(OrderAction.START_PREPARING,
                                               OrderActor.RESTAURANT)

    def mark_ready(self, order_id: str) -> OrderEvent:
        order = self.store.get(order_id)
        return self.lock_for(order).transition(OrderAction.MARK_READY,
                                               OrderActor.RESTAURANT)

    def assign(self, order_id: str, drivers: list[Driver]) -> list[tuple[Driver, bool]]:
        order = self.store.get(order_id)
        return self.dispatch.offer(order, drivers)

    def pick_up(self, order_id: str, driver_id: str) -> OrderEvent:
        """Two different rejections, deliberately kept distinct: a driver who is
        not THIS order's driver is an authorisation failure, while an order with
        no driver at all fails the aggregate's precondition guard."""
        order = self.store.get(order_id)
        if order.driver_id is not None and order.driver_id != driver_id:
            raise IllegalTransition(f"{driver_id} is not assigned to {order.id}")
        return self.lock_for(order).pick_up(OrderActor.DRIVER)

    def deliver(self, order_id: str, driver_id: str) -> OrderEvent:
        order = self.store.get(order_id)
        if order.driver_id != driver_id:
            raise IllegalTransition(f"{driver_id} is not assigned to {order.id}")
        return self.lock_for(order).transition(OrderAction.DELIVER, OrderActor.DRIVER)

    # -- cancellation ------------------------------------------------------
    def cancel(self, order_id: str, actor: OrderActor) -> tuple[OrderEvent, RefundDecision]:
        """The machine says who may cancel; the policy says what it costs. Note
        the policy is evaluated BEFORE the transition, because after it the
        state is CANCELLED and the policy can no longer tell where it came
        from — a subtle ordering that is worth stating out loud."""
        order = self.store.get(order_id)
        decision = self.policy.evaluate(order)
        event = self.lock_for(order).transition(OrderAction.CANCEL, actor,
                                                note=str(decision))
        return event, decision

    def sweep_timeouts(self, order_ids: Iterable[str]) -> list[OrderEvent]:
        """The system actor's job, driven by a scheduler in production and by a
        test directly."""
        fired = []
        for order_id in order_ids:
            order = self.store.get(order_id)
            event = order.check_accept_timeout()
            if event is not None:
                fired.append(event)
        return fired


# --------------------------------------------------------------------------
# 8. Tests and demo — correctness proved, not asserted
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


def _menu() -> Menu:
    return Menu("rest-1", [
        MenuItem("pizza", "Margherita", Money.from_str("12.50")),
        MenuItem("wings", "Chicken Wings", Money.from_str("8.25")),
        MenuItem("coke", "Cola", Money.from_str("2.00")),
        MenuItem("special", "Sold Out Special", Money.from_str("99.00"), available=False),
    ])


def _service() -> DeliveryService:
    return DeliveryService(
        OrderStore(),
        StandardPricing(Money.from_str("2.00"), Money.from_str("0.75"), 0.10, 0.08),
        TieredCancellationPolicy(),
        DispatchService(NearestAssigner()),
        SimpleEtaEstimator(),
    )


if __name__ == "__main__":
    # --- 1. money is integers, and the float version is demonstrably wrong ---
    a = Money.from_str("0.10")
    b = Money.from_str("0.20")
    _check("integer money adds exactly", (a + b) == Money.from_str("0.30"))
    _check("and the float version does not", 0.1 + 0.2 != 0.3)
    _check("from_units rounds rather than truncating",
           Money.from_units(0.29) == Money.from_str("0.29"))
    _check("money parses without a float", Money.from_str("12.34").minor == 1234)
    _check("and negative amounts", Money.from_str("-4.05").minor == -405)
    _check("multiplication is by an integer quantity",
           (Money.from_str("12.50") * 3) == Money.from_str("37.50"))
    _check("multiplying by a float is refused",
           _raises(TypeError, lambda: Money.from_str("1.00") * 1.5) is not None)
    _check("currency mismatch is refused",
           _raises(ValueError, lambda: Money(100, "USD") + Money(100, "EUR")) is not None)
    _check("money formats to two places", str(Money(5, "USD")) == "USD 0.05")
    _check("and pads correctly", str(Money(1234, "USD")) == "USD 12.34")

    # The allocation case: three ways with a remainder that naive rounding loses.
    shares = Money(100, "USD").allocate([1, 1, 1])
    _check("allocation loses no minor units",
           sum(s.minor for s in shares) == 100)
    _check("and the remainder goes to the first share",
           [s.minor for s in shares] == [34, 33, 33])

    # --- 2. cart, snapshot and pricing ---
    menu = _menu()
    cart = Cart("rest-1")
    cart.add(menu, "pizza", 2)
    cart.add(menu, "coke")
    _check("the subtotal is exact", cart.subtotal == Money.from_str("27.00"))
    _check("adding the same item merges the line",
           cart.add(menu, "pizza", 1).quantity == 3 and len(cart.lines) == 2)
    cart.set_quantity("pizza", 2)
    _check("set_quantity replaces", cart.subtotal == Money.from_str("27.00"))
    _check("a sold-out item cannot be added",
           _raises(ItemUnavailable, lambda: cart.add(menu, "special")) is not None)
    _check("an unknown item cannot be added",
           _raises(KeyError, lambda: cart.add(menu, "sushi")) is not None)
    _check("an item from another restaurant is refused",
           _raises(ValueError,
                   lambda: cart.add(Menu("rest-2", [MenuItem("x", "X", Money(100))]), "x"))
           is not None)
    _check("a zero quantity removes the line", (cart.set_quantity("coke", 0) or True)
           and len(cart.lines) == 1)

    # THE price snapshot: change the menu after adding to the cart.
    before = cart.subtotal
    menu.items["pizza"] = MenuItem("pizza", "Margherita", Money.from_str("19.99"))
    _check("a menu price change does NOT change the cart", cart.subtotal == before)

    svc = _service()
    order = svc.place_order("key-1", "cust-1", cart, distance_km=4.0,
                            tip=Money.from_str("3.00"))
    _check("the order total is subtotal + fees + tax + tip",
           order.pricing.total ==
           order.pricing.subtotal + order.pricing.delivery_fee +
           order.pricing.service_fee + order.pricing.tax + order.pricing.tip)
    _check("the subtotal is the snapshotted price, not the new menu price",
           order.pricing.subtotal == before)
    _check("the delivery fee scales with distance",
           order.pricing.delivery_fee == Money.from_str("2.00") + Money.from_str("3.00"))
    _check("the subtotal is the two pizzas left in the cart",
           order.pricing.subtotal == Money.from_str("25.00"))
    _check("the service fee is 10% of the food",
           order.pricing.service_fee == order.pricing.subtotal.percent(0.10))
    _check("the tip is not taxed", order.pricing.tax ==
           (order.pricing.subtotal + order.pricing.service_fee).percent(0.08))
    _check("the order starts PLACED with no driver",
           order.state is OrderState.PLACED and order.driver_id is None)

    # --- 3. idempotent placement ---
    same = svc.place_order("key-1", "cust-1", cart, distance_km=4.0,
                           tip=Money.from_str("3.00"))
    _check("the same key returns the SAME order", same is order)
    other = svc.place_order("key-2", "cust-1", cart, distance_km=4.0)
    _check("a different key creates a new order", other is not order)
    _check("an empty cart cannot be ordered",
           _raises(ValueError,
                   lambda: svc.place_order("key-3", "c", Cart("rest-1"), 1.0))
           is not None)
    _check("a missing key is refused",
           _raises(ValueError, lambda: svc.place_order("", "c", cart, 1.0)) is not None)

    # --- 4. the state machine: legal transitions ---
    svc.accept(order.id, "rest-1")
    _check("the restaurant can accept a placed order",
           order.state is OrderState.ACCEPTED)
    svc.start_preparing(order.id)
    _check("preparation can start", order.state is OrderState.PREPARING)
    svc.mark_ready(order.id)
    _check("the order can be marked ready", order.state is OrderState.READY)
    _check("accepting cleared the timeout deadline", order.accept_deadline is None)

    # --- 5. the machine rejects illegal transitions AND wrong actors ---
    svc2 = _service()
    cart2 = Cart("rest-1")
    cart2.add(menu, "wings", 1)
    o2 = svc2.place_order("k", "cust-2", cart2, 2.0)
    msg = _raises(IllegalTransition, lambda: svc2.mark_ready(o2.id))
    _check("a restaurant cannot mark a PLACED order ready",
           msg is not None and "not legal from placed" in msg)
    msg = _raises(IllegalTransition,
                  lambda: svc2.lock_for(o2).transition(OrderAction.ACCEPT,
                                                       OrderActor.CUSTOMER))
    _check("a customer cannot accept their own order",
           msg is not None and "may not" in msg and "restaurant" in msg)
    _check("a customer CAN cancel a placed order",
           OrderAction.CANCEL in o2.allowed_actions(OrderActor.CUSTOMER))
    _check("a driver can do nothing at all on a placed order",
           o2.allowed_actions(OrderActor.DRIVER) == [])

    # Now move it to ACCEPTED, where START_PREPARING IS a row in the table — so
    # the rejection below is the ACTOR guard rather than a missing row. The two
    # failure modes are distinct and both are worth asserting separately.
    svc2.accept(o2.id, "rest-1")
    msg = _raises(IllegalTransition,
                  lambda: svc2.lock_for(o2).transition(OrderAction.START_PREPARING,
                                                       OrderActor.DRIVER))
    _check("a driver cannot start preparation even where the row exists",
           msg is not None and "may not" in msg and "restaurant" in msg)
    _check("but the restaurant can",
           OrderAction.START_PREPARING in o2.allowed_actions(OrderActor.RESTAURANT))

    # --- 6. the full happy path, and pick-up needs a driver ---
    svc3 = _service()
    c3 = Cart("rest-1")
    c3.add(menu, "wings", 2)
    o3 = svc3.place_order("k3", "cust-3", c3, 3.0)
    svc3.accept(o3.id, "rest-1")
    svc3.start_preparing(o3.id)
    svc3.mark_ready(o3.id)
    msg = _raises(OrderConflict, lambda: svc3.pick_up(o3.id, "d-1"))
    _check("an order cannot be picked up with no driver assigned",
           msg is not None and "no driver assigned" in msg)

    drivers = [Driver("d-far", 9.0), Driver("d-near", 1.2), Driver("d-mid", 3.5)]
    results = svc3.assign(o3.id, drivers)
    _check("the nearest available driver gets the order",
           results[0][0].id == "d-near" and results[0][1] is True)
    _check("the order records the driver",
           o3.driver_id == "d-near" and o3.state is OrderState.READY)
    _check("and the kitchen state did not change", o3.state is OrderState.READY)
    msg = _raises(IllegalTransition, lambda: svc3.pick_up(o3.id, "d-far"))
    _check("a driver who is not assigned cannot pick up",
           msg is not None and "not assigned" in msg)
    svc3.pick_up(o3.id, "d-near")
    _check("the assigned driver picks up", o3.state is OrderState.PICKED_UP)
    svc3.deliver(o3.id, "d-near")
    _check("and delivers", o3.state is OrderState.DELIVERED)
    _check("the order is terminal", o3.is_terminal)
    _check("no further driver actions are possible",
           o3.allowed_actions(OrderActor.DRIVER) == [])
    msg = _raises(IllegalTransition,
                  lambda: svc3.lock_for(o3).transition(OrderAction.CANCEL,
                                                       OrderActor.SYSTEM))
    _check("a delivered order cannot be cancelled",
           msg is not None and "not legal from delivered" in msg)

    # --- 7. THE race: many drivers, one order, exactly one winner ---
    svc4 = _service()
    c4 = Cart("rest-1")
    c4.add(menu, "coke", 1)
    o4 = svc4.place_order("k4", "cust-4", c4, 1.0)
    svc4.accept(o4.id, "rest-1")
    svc4.start_preparing(o4.id)
    svc4.mark_ready(o4.id)

    lock4 = svc4.lock_for(o4)
    contenders = [Driver(f"d-{i}", 1.0 + i) for i in range(8)]
    expected_version = o4.version
    barrier = threading.Barrier(len(contenders))
    outcomes: list[bool] = []
    outcomes_lock = threading.Lock()

    def race(driver: Driver) -> None:
        barrier.wait()                       # release all threads together
        # Every driver presents the version that came with their offer — the
        # same version for all eight. One wins and bumps the version; the other
        # seven fail on the STALE guard, not on the already-assigned one.
        won = svc4.dispatch.accept(o4, driver, lock4, expected_version)
        with outcomes_lock:
            outcomes.append(won)

    threads = [threading.Thread(target=race, args=(d,)) for d in contenders]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _check("exactly one of eight racing drivers wins", outcomes.count(True) == 1)
    _check("and seven lose", outcomes.count(False) == 7)
    _check("the order has exactly one driver",
           o4.driver_id is not None and o4.driver_id in {d.id for d in contenders})

    # A second attempt after the race loses on the stale-version guard.
    late = Driver("d-late", 0.1)
    _check("a late acceptance fails",
           svc4.dispatch.accept(o4, late, lock4, expected_version) is False)
    _check("and the loser is not recorded on the order", o4.driver_id != "d-late")

    # The same guard fires for a driver who somehow presented a FUTURE version.
    _check("an acceptance against a future version also fails",
           svc4.dispatch.accept(o4, late, lock4, o4.version + 5) is False)

    # And a stale version is rejected directly, which is the other half of the CAS.
    msg = _raises(OrderConflict,
                  lambda: o4.assign_driver(Driver("d-stale"), expected_version))
    _check("a stale version is rejected",
           msg is not None and "moved from version" in msg)

    # --- 8. reassignment after a driver drops out ---
    v = o4.version
    o4.unassign_driver(v, note="driver went offline")
    _check("the driver can be unassigned", o4.driver_id is None)
    res = svc4.dispatch.offer(o4, [Driver("d-new", 0.5)])
    _check("and the order can be reassigned",
           res[0][1] is True and o4.driver_id == "d-new")
    _check("the history records both assignments",
           sum(1 for e in o4.history if "assigned" in e.note and "unassigned" not in e.note)
           == 2)

    # --- 9. cancellation policy, at each state ---
    def cancelled_at(action_state: OrderState) -> tuple[Order, RefundDecision]:
        s = _service()
        # A FRESH menu, not the shared `menu` — the snapshot test above mutated
        # that one's pizza to 19.99, and a fixture that silently depends on
        # another test's side effect is how a suite becomes order-dependent.
        fresh = _menu()
        c = Cart("rest-1")
        c.add(fresh, "pizza", 1)
        o = s.place_order("kc", "cust", c, 2.0, tip=Money.from_str("2.00"))
        if action_state in (OrderState.ACCEPTED, OrderState.PREPARING, OrderState.READY):
            s.accept(o.id, "rest-1")
        if action_state in (OrderState.PREPARING, OrderState.READY):
            s.start_preparing(o.id)
        if action_state is OrderState.READY:
            s.mark_ready(o.id)
        # The actor is part of the transition key, so cancelling a READY order
        # as the customer is not merely refused by the policy — it is not in
        # the table at all. Only the system may do it.
        actor = OrderActor.SYSTEM if action_state is OrderState.READY \
            else OrderActor.CUSTOMER
        _, decision = s.cancel(o.id, actor)
        return o, decision

    o_placed, d_placed = cancelled_at(OrderState.PLACED)
    _check("cancelling before acceptance refunds the whole total",
           d_placed.refund == o_placed.pricing.total and d_placed.fee_kept.is_zero)
    _check("and that total is the one computed at placement",
           d_placed.refund == Money.from_str("20.35"))
    _check("the reason names the stage",
           "before the restaurant accepted" in d_placed.reason)

    o_accepted, d_accepted = cancelled_at(OrderState.ACCEPTED)
    _check("cancelling after acceptance keeps the delivery fee",
           d_accepted.fee_kept == o_accepted.pricing.delivery_fee)
    _check("which is really charged, not zero",
           d_accepted.fee_kept == Money.from_str("3.50"))
    _check("so the refund is smaller than the total",
           d_accepted.refund.minor < o_accepted.pricing.total.minor)
    _check("and the refund plus the fee is the total",
           (d_accepted.refund + d_accepted.fee_kept) == o_accepted.pricing.total)

    o_prep, d_prep = cancelled_at(OrderState.PREPARING)
    _check("cancelling mid-preparation keeps half the food value",
           d_prep.fee_kept == o_prep.pricing.subtotal.percent(0.5))
    _check("which is less than the whole total",
           d_prep.refund.minor < o_prep.pricing.total.minor)
    _check("and still more than the delivery fee alone",
           d_prep.refund.minor > o_prep.pricing.delivery_fee.minor)

    o_ready, d_ready = cancelled_at(OrderState.READY)
    _check("a ready order refunds nothing",
           d_ready.refund.is_zero and d_ready.fee_kept == o_ready.pricing.total)
    _check("and says so", "no refund" in d_ready.reason)

    # The policy is evaluated BEFORE the transition, so the reason survives it.
    svc5 = _service()
    c5 = Cart("rest-1")
    c5.add(menu, "coke", 1)
    o5 = svc5.place_order("k5", "cust-5", c5, 1.0)
    svc5.accept(o5.id, "rest-1")
    event5, decision5 = svc5.cancel(o5.id, OrderActor.CUSTOMER)
    _check("the cancel event carries the refund reason in its note",
           "delivery fee retained" in event5.note)
    _check("and the order is CANCELLED", o5.state is OrderState.CANCELLED)
    _check("cancelling twice is refused",
           _raises(IllegalTransition,
                   lambda: svc5.cancel(o5.id, OrderActor.CUSTOMER)) is not None)
    _check("the system can cancel a ready order",
           OrderActor.SYSTEM in TRANSITIONS[(OrderState.READY, OrderAction.CANCEL)][1])
    _check("but a customer cannot cancel a ready order",
           OrderActor.CUSTOMER not in TRANSITIONS[(OrderState.READY, OrderAction.CANCEL)][1])

    # --- 10. the accept timeout, driven by the system actor ---
    now = {"t": 0.0}
    svc6 = DeliveryService(OrderStore(),
                           StandardPricing(Money.from_str("2.00"), Money.from_str("0.75"),
                                           0.10, 0.08),
                           TieredCancellationPolicy(),
                           DispatchService(NearestAssigner()),
                           SimpleEtaEstimator(), clock=lambda: now["t"])
    c6 = Cart("rest-1")
    c6.add(menu, "coke", 1)
    o6 = svc6.place_order("k6", "cust-6", c6, 1.0, accept_timeout=120.0)
    _check("the deadline is armed",
           o6.accept_deadline == 120.0 and o6.state is OrderState.PLACED)
    now["t"] = 60.0
    _check("nothing fires before the deadline", svc6.sweep_timeouts([o6.id]) == [])
    now["t"] = 121.0
    fired = svc6.sweep_timeouts([o6.id])
    _check("the sweep cancels a silent order", len(fired) == 1)
    _check("with the system as the actor", fired[0].actor is OrderActor.SYSTEM)
    _check("and the order is CANCELLED", o6.state is OrderState.CANCELLED)
    _check("a second sweep is a no-op", svc6.sweep_timeouts([o6.id]) == [])

    # An accepted order never times out, even if the clock runs on.
    c6b = Cart("rest-1")
    c6b.add(menu, "coke", 1)
    now["t"] = 200.0
    o6b = svc6.place_order("k6b", "cust-6", c6b, 1.0, accept_timeout=120.0)
    svc6.accept(o6b.id, "rest-1")
    now["t"] = 10_000.0
    _check("an accepted order is not swept", svc6.sweep_timeouts([o6b.id]) == [])

    # --- 11. ETA ---
    eta = SimpleEtaEstimator(prep_minutes=15, speed_km_per_min=0.5, pickup_buffer=3)
    c7 = Cart("rest-1")
    c7.add(menu, "coke", 1)
    s7 = _service()
    o7 = s7.place_order("k7", "c", c7, 1.0)
    _check("a placed order with no driver is prep + buffer",
           eta.estimate(o7, None) == 18)
    s7.accept(o7.id, "rest-1")
    s7.start_preparing(o7.id)
    s7.mark_ready(o7.id)
    _check("a ready order drops the prep time", eta.estimate(o7, None) == 3)
    _check("a driver 4km away adds 8 minutes",
           eta.estimate(o7, Driver("d", 4.0)) == 11)
    s7.assign(o7.id, [Driver("d-eta", 2.0)])
    s7.pick_up(o7.id, "d-eta")
    _check("a picked-up order has no prep and no buffer",
           eta.estimate(o7, Driver("d", 2.0)) == 4)
    s7.deliver(o7.id, "d-eta")
    _check("a delivered order has no ETA", eta.estimate(o7, None) == 0)

    # --- 12. the history is a complete, ordered audit ---
    _check("every transition is recorded",
           len(o7.history) == 7)    # created, accept, prep, ready, assign, pickup, deliver
    _check("with contiguous indices",
           [e.index for e in o7.history] == list(range(7)))
    _check("each row's to_state matches the next row's from_state",
           all(o7.history[i].to_state is o7.history[i + 1].from_state
               for i in range(len(o7.history) - 1)))
    _check("the first row is the origin, not a transition",
           o7.history[0].action is OrderAction.CREATE
           and (o7.history[0].from_state, o7.history[0].action) not in TRANSITIONS)
    _check("the last row is DELIVERED",
           o7.history[-1].to_state is OrderState.DELIVERED)
    pickup_row = next(e for e in o7.history if e.action is OrderAction.PICK_UP)
    _check("and the actor on the pickup row is the driver",
           pickup_row.actor is OrderActor.DRIVER)
    _check("the assignment row did not change the kitchen state",
           next(e for e in o7.history if "assigned" in e.note).from_state
           is OrderState.READY
           and next(e for e in o7.history if "assigned" in e.note).to_state
           is OrderState.READY)

    # --- 13. the store ---
    _check("an unknown order id raises",
           _raises(KeyError, lambda: s7.store.get("ord-999999")) is not None)
    _check("the idempotency key maps to the right order",
           s7.store.find_by_key("k7") is o7)
    _check("an unknown key returns None", s7.store.find_by_key("nope") is None)
    _check("ids are unique",
           len({s7.store.next_id() for _ in range(50)}) == 50)

    # --- 14. the two assignment strategies rank differently ---
    candidates = [Driver("a", distance_km=0.5, active_orders=5),
                  Driver("b", distance_km=3.0, active_orders=0),
                  Driver("c", distance_km=1.0, active_orders=2),
                  Driver("off", distance_km=0.1, available=False)]
    near = [d.id for d in NearestAssigner().rank(o7, candidates)]
    load = [d.id for d in LeastLoadedAssigner().rank(o7, candidates)]
    _check("the nearest assigner ignores load", near == ["a", "c", "b"])
    _check("the least-loaded assigner ignores distance", load == ["b", "c", "a"])
    _check("both exclude unavailable drivers",
           "off" not in near and "off" not in load)

    # --- 15. the table itself is exhaustively checked ---
    terminal_states = [s for s in OrderState if s.is_terminal]
    _check("terminal states have no outgoing transitions",
           all((s, a) not in TRANSITIONS for s in terminal_states
               for a in OrderAction))
    _check("every non-terminal state can be cancelled by the system",
           all((s, OrderAction.CANCEL) in TRANSITIONS
               for s in OrderState
               if not s.is_terminal and s is not OrderState.PICKED_UP))
    _check("a picked-up order cannot be cancelled at all",
           (OrderState.PICKED_UP, OrderAction.CANCEL) not in TRANSITIONS)
    _check("every transition target is a real state",
           all(target in OrderState for target, _ in TRANSITIONS.values()))
    _check("every transition has at least one allowed actor",
           all(len(allowed) >= 1 for _, allowed in TRANSITIONS.values()))
    _check("the customer can never reach PICKED_UP",
           all(OrderActor.CUSTOMER not in allowed
               for (s, a), (_, allowed) in TRANSITIONS.items()
               if a is OrderAction.PICK_UP))

    print("\nall checks passed")
```

**What to point at:**

- `Order.driver_id` is a **field, not a state**, with the comment saying why: a driver can be assigned while the food cooks, so one linear enum cannot express it.
- `TRANSITIONS` is keyed by `(state, action)` and the value carries **the set of allowed actors** — so "illegal" and "unauthorised" are one lookup, and `allowed_actions(actor)` is derived from the same table.
- `Order.assign_driver` checks **version, then already-assigned, then terminal** — and `OrderLock` makes the version check atomic with the mutation. The demo runs eight threads at a barrier and asserts exactly one winner.
- `pick_up` is the **guard the table cannot express**: it needs an assigned driver, which is a fact about another field.
- `DeliveryService.cancel` evaluates the policy **before** the transition, because afterwards the state is `CANCELLED` and the policy can no longer tell where it came from.
- `Money.allocate` distributes the remainder one unit at a time — naive rounding loses cents.

---

## 6. Concurrency & thread-safety

### What breaks

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Two drivers accepting one order** | Both read version 3, both pass the check, both write — **two winners**, and the code looks correct | `OrderLock`: the compare and the set in one critical section |
| **Two taps on "Place order"** | Two orders, one payment attempt each | The idempotency check and the insert share one lock; the same key returns the same order |
| **Cancel racing a pick-up** | The order is cancelled and picked up, and a driver is carrying food for a cancelled order | Both are transitions through the same per-order lock, so one of them loses |
| **Restaurant accept racing the timeout sweep** | Both accept and cancel fire, and the order ends in an impossible combination | Same per-order lock; the loser sees a state it did not expect |
| **A cart edited while an order is placed** | The order's lines change after placement | `Cart.snapshot()` returns an immutable tuple and the order holds that |
| **A menu edited after add-to-cart** | The price changes retroactively | The snapshot is taken at `add`, not at checkout |
| **A shared `Order` read while it mutates** | A UI renders a half-applied transition, or a history row with no state change | `OrderLock.snapshot()` — read under the same lock the writes use |
| **A shared `OrderStore` sequence** | Two orders get the same id | `OrderStore.next_id()` takes the store lock |
| **A lock per operation instead of per order** | The cancel lock and the pick-up lock are different objects, so they do not exclude each other | One lock per order id, held in the service's map |
| **A distributed deployment** | An in-process lock does not span two servers | The version becomes a store-level compare-and-set — the same three guards, enforced by the database |

**The one to volunteer:** the **version check that is not atomic**. It is the concurrency point specific to this problem, and it is the one that survives code review: `if order.version == v: order.driver_id = d.id` reads correctly and is wrong. The demo proves it — eight threads released together at a barrier, exactly one success — and the fix is one lock around the pair, not a cleverer check.

### The CAS in a distributed setting

```python
class StoreBackedOrderRepository:
    """The same three guards, enforced by the store instead of a mutex.

    In-process the lock is a `threading.Lock`; across servers the equivalence is
    a conditional write — `UPDATE orders SET driver_id = ?, version = version + 1
    WHERE id = ? AND version = ? AND driver_id IS NULL AND state NOT IN (...)`.
    The row count tells you whether you won, and the guards move from Python
    into the WHERE clause. The design does not change; only the mechanism does.
    """

    def __init__(self, table) -> None:
        self.table = table

    def assign_driver(self, order_id: str, driver_id: str, expected_version: int) -> bool:
        updated = self.table.compare_and_set(
            key=order_id,
            expected_version=expected_version,
            require={"driver_id": None},
            set_to={"driver_id": driver_id},
        )
        return updated == 1        # 0 means somebody else won the race
```

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Split the state into two orthogonal dimensions"

The honest version of the modelling point, when the interviewer pushes on it:

```python
class KitchenState(Enum):
    PLACED = "placed"
    ACCEPTED = "accepted"
    PREPARING = "preparing"
    READY = "ready"
    REJECTED = "rejected"


class FulfilmentState(Enum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"


class TwoDimensionOrder(Order):
    """Kitchen progress and fulfilment progress are independent: a driver can
    be assigned while the food is still cooking, and a kitchen can be READY
    while nobody has accepted the delivery.

    The single enum in section 5 collapses the second dimension into a field
    because that is the simpler correct model for this rule set. The version
    below is what you build when the rules start caring — for example when a
    driver may only be assigned once the restaurant has ACCEPTED, which is a
    constraint BETWEEN the two dimensions and cannot be expressed in either.
    """
```

The general principle: **when two facts vary independently, model them independently** — the moment you find yourself writing `ASSIGNED_AND_PREPARING` you have made the state space the cross product of two things that are not, in fact, coupled.

### 7.2 "Split the bill"

`Money.allocate` already does the arithmetic; the follow-up is where the shares come from:

```python
def split_evenly(order: Order, ways: int) -> list[Money]:
    if ways < 1:
        raise ValueError("cannot split between fewer than one payer")
    return order.pricing.total.allocate([1] * ways)


def split_by_items(order: Order, payers: dict[str, list[str]]) -> dict[str, Money]:
    """Each payer's share is their own items plus an equal slice of the fees.
    The allocation is the interesting part: the fees must be split with the
    remainder distributed rather than rounded away, or the shares do not sum to
    the total and the payment reconciliation fails by a cent."""
    if not payers:
        raise ValueError("there is nobody to split between")
    currency = order.pricing.total.currency
    zero = Money.zero(currency)
    shares: dict[str, Money] = {name: zero for name in payers}
    for name, item_ids in payers.items():
        for line in order.lines:
            if line.menu_item_id in item_ids:
                shares[name] = shares[name] + line.line_total
    items_total = zero
    for share in shares.values():
        items_total = items_total + share
    fees = order.pricing.total - items_total
    for name, extra in zip(shares, fees.allocate([1] * len(shares))):
        shares[name] = shares[name] + extra
    return shares
```

The invariant to assert: **the shares sum to the total, exactly** — which is only true because `allocate` distributes the remainder rather than rounding.

### 7.3 "The restaurant goes offline mid-service"

This is the timeout, generalised into a recovery policy:

```python
class RestaurantOfflinePolicy(ABC):
    @abstractmethod
    def handle(self, order: Order, service: "DeliveryService") -> str: ...


class CancelAndReassign(RestaurantOfflinePolicy):
    def handle(self, order: Order, service: "DeliveryService") -> str:
        if order.state in (OrderState.PLACED, OrderState.ACCEPTED):
            service.cancel(order.id, OrderActor.SYSTEM)
            return "cancelled and refunded"
        if order.state is OrderState.PREPARING:
            return "keep waiting: food may already be cooking"
        return "no action"


class ReassignToSibling(RestaurantOfflinePolicy):
    """The chain has another branch that can make the same items."""

    def __init__(self, siblings: dict[str, str]) -> None:
        self.siblings = siblings

    def handle(self, order: Order, service: "DeliveryService") -> str:
        sibling = self.siblings.get(order.restaurant_id)
        if sibling is None:
            return CancelAndReassign().handle(order, service)
        return f"reassignment to {sibling} is a new order with the same lines"
```

The design note: **reassignment is not a mutation of the order** — it is a new order carrying the same `lines` snapshot. Mutating `restaurant_id` would break the history, because the earlier history rows name a restaurant that no longer applies.

### 7.4 "Driver drops out after pickup"

The state is `PICKED_UP`, so the order cannot be reassigned by `unassign_driver` (it allows any non-terminal state, which includes `PICKED_UP` — worth reviewing). The correct handling is an incident, not a state transition:

```python
class DeliveryIncident(ABC):
    @abstractmethod
    def resolve(self, order: Order) -> RefundDecision: ...


class DriverLostAfterPickup(DeliveryIncident):
    """The food has left the restaurant, so the order cannot be silently
    cancelled and refunded — that would be a policy that pays out for every
    lost delivery. The incident is recorded, the customer is refunded in full
    as a goodwill gesture, and the cost is attributed to the courier
    operation. The design point is that the refund comes from a DIFFERENT
    object than the CancellationPolicy, because it is a different decision with
    a different owner."""

    def resolve(self, order: Order) -> RefundDecision:
        paid = order.pricing.total
        return RefundDecision(paid, Money.zero(paid.currency),
                              "driver lost after pickup: full goodwill refund, "
                              "cost attributed to courier operations")
```

### 7.5 "Expose per-actor views"

`allowed_actions(actor)` already gives the actions; a view composes it with what each actor may see:

```python
@dataclass(frozen=True)
class OrderView:
    order_id: str
    state: OrderState
    actions: tuple[OrderAction, ...]
    payload: dict


def view_for(order: Order, actor: OrderActor) -> OrderView:
    """The driver must not see the customer's tip breakdown before delivery,
    and the customer must not see the courier's identity before pickup. The
    projection is a function of the actor, which is why it is not a method on
    Order — the aggregate should not know about visibility rules any more than
    it knows about refunds."""
    payload: dict = {"state": order.state.value}
    if actor is OrderActor.RESTAURANT:
        payload["lines"] = [(l.name, l.quantity) for l in order.lines]
    elif actor is OrderActor.DRIVER:
        payload["pickup"] = order.restaurant_id
        payload["delivery_eta"] = None
    elif actor is OrderActor.CUSTOMER:
        payload["total"] = str(order.pricing.total)
        payload["driver"] = order.driver_id if order.state in (
            OrderState.PICKED_UP, OrderState.DELIVERED) else None
    return OrderView(order.id, order.state,
                     tuple(order.allowed_actions(actor)), payload)
```

### 7.6 "Batch orders to one driver"

Assignment ranking extended to a set of orders — the design point is that a **batch assignment is a matching problem, not a per-order choice**, so it does not belong in `DriverAssigner`:

```python
class BatchMatcher(ABC):
    @abstractmethod
    def match(self, orders: list[Order], drivers: list[Driver]) -> list[tuple[str, str]]:
        """Return (order_id, driver_id) pairs. Ranking per order and then
        assigning greedily is the wrong shape: two orders can both rank the
        same driver first, and the greedy result depends on the loop order.
        A matcher decides the whole set at once so the outcome is stable."""


class GreedyNearestMatcher(BatchMatcher):
    def match(self, orders: list[Order], drivers: list[Driver]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        taken: set[str] = set()
        # Stable order: sort by distance so the result does not depend on the
        # order the orders arrived in.
        candidates = sorted(
            ((o, d) for o in orders for d in drivers if d.available),
            key=lambda pair: (pair[1].distance_km, pair[0].id, pair[1].id))
        assigned_orders: set[str] = set()
        for order, driver in candidates:
            if order.id in assigned_orders or driver.id in taken:
                continue
            pairs.append((order.id, driver.id))
            assigned_orders.add(order.id)
            taken.add(driver.id)
        return pairs
```

### 7.7 "Why did my order cost that?"

`PricingBreakdown` already holds the answer, so the audit is a renderer — and the reason it is a separate object from `Money` is that a total without its components cannot be explained:

```python
def explain_pricing(breakdown: PricingBreakdown) -> list[str]:
    return [
        f"items                 {breakdown.subtotal}",
        f"delivery              {breakdown.delivery_fee}",
        f"service               {breakdown.service_fee}",
        f"tax                   {breakdown.tax}",
        f"tip                   {breakdown.tip}",
        f"discount             -{breakdown.discount}" if not breakdown.discount.is_zero
        else "discount               none",
        f"TOTAL                 {breakdown.total}",
    ]
```

The invariant worth asserting in a test: the lines **sum to the total**, so no fee can be added without appearing in the breakdown.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Money as a float** | `0.1 + 0.2` totals off by a hundredth of a cent, and reconciliation fails monthly | Integer minor units; `from_units` rounds half-up |
| 2 | **Taxing the tip** | The customer pays tax on a gratuity, which is wrong in most jurisdictions | Tax applies to food plus service fee; the tip is added after |
| 3 | **Reading the price at checkout** | A menu edit between add and checkout changes the charge | The unit price is snapshotted at `add` |
| 4 | **A mutable lines list on the order** | A later cart edit reaches into a placed order | `Cart.snapshot()` returns a tuple |
| 5 | **Driver assignment as a state** | `ASSIGNED_AND_PREPARING`, then `ASSIGNED_AND_READY` — the state space is a cross product | A field orthogonal to the kitchen state |
| 6 | **Two drivers, no CAS** | Both are assigned; two couriers arrive and one is sent away unpaid | Version-checked assignment under one lock |
| 7 | **A CAS without atomicity** | The check and the write interleave; two winners, and the code reads correctly | `OrderLock` around both |
| 8 | **A double-tap on Place order** | Two orders and two charges | The idempotency key, checked and inserted under one lock |
| 9 | **Idempotency key but no lock around the insert** | Both threads find no existing order and both create one | The check and the insert share a lock |
| 10 | **Cancel racing pick-up** | A driver carries food for a cancelled order | Both are transitions through the per-order lock |
| 11 | **Pick-up with no driver** | The order is picked up by nobody and sits in `PICKED_UP` forever | The guard: `pick_up` requires an assigned driver |
| 12 | **The policy evaluated after the transition** | The state is already `CANCELLED`, so the policy cannot tell where it came from and refunds the wrong amount | Evaluate first, transition second |
| 13 | **A restaurant that never responds** | The order is held open indefinitely and the customer waits | `arm_accept_timeout` plus a system sweep |
| 14 | **A second sweep on an already-cancelled order** | The order is cancelled twice and refunded twice | The sweep checks the state; a cancelled order has no deadline path |
| 15 | **Reassigning a `PICKED_UP` order** | The food is in a courier's bag and the order jumps to a new driver | `unassign_driver` allows it, which is a **deliberate review point** — the incident policy owns this case instead |
| 16 | **Round-robin split losing a cent** | The shares do not sum to the total | `allocate` distributes the remainder one unit at a time |
| 17 | **Currency mixing** | Dollars added to euros silently | Currency is checked on every operation |
| 18 | **An empty cart ordered** | A zero-value order is created and charged | Placement refuses an empty cart |
| 19 | **A menu item from another restaurant** | The cart mixes two restaurants and one kitchen gets the order | The cart validates the restaurant on `add` |
| 20 | **A stale version after an unrelated change** | A driver's acceptance fails because someone edited the tip | Deliberate — the CAS is on the whole aggregate, not on one field; a narrower check would need per-field versions |

**The three to volunteer:** #1 (money as integers), #7 (the CAS that is not atomic) and #5 (assignment as a field rather than a state). The first is the one that loses real money, the second is the concurrency bug that survives review, and the third is the modelling insight that the whole file is built around.

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **State (table-driven)** | `TRANSITIONS` | The machine is data, so it is exhaustively testable and the actor is part of the key | Domain guards that need another field live outside the table |
| **Strategy** | `PricingStrategy`, `CancellationPolicy`, `DriverAssigner`, `EtaEstimator` | Four things that change for business reasons, not modelling reasons | Four injected collaborators in the service |
| **Value Object** | `Money`, `LineItem`, `PricingBreakdown`, `OrderEvent`, `RefundDecision` | Immutable, hashable, safe to log and to compare in tests | One allocation per line and per event |
| **Aggregate root** | `Order` | One consistency boundary, one version, one history | The aggregate grows; pricing and refunds were deliberately pushed out |
| **Repository** | `OrderStore` | Idempotency keys and id lookup in one place | A second map to keep in step |
| **Façade** | `DeliveryService` | Callers get one object and five verbs | The façade must not accumulate rules — it only coordinates |
| **Ports & Adapters** | `EtaEstimator`, `DriverAssigner`, the injected `clock` | A test asserts an ETA instead of waiting for one | An interface over something small |
| **Observer** | *Deliberately absent* | The history is an audit log, not an event bus; a real notification fan-out belongs outside the aggregate | A separate notifier if real-time push is needed |
| **Singleton** | *Deliberately rejected* | A global store makes two tests interfere | — |

**The sentence to give:** *"State for the lifecycle because the machine is data and can be tested exhaustively; Strategy four times because pricing, refunds, ranking and ETAs each change for a business reason; Value Object for the money and the history rows. I would explicitly not put notifications in the aggregate — the history is an audit log, and an event bus inside the consistency boundary is how you get a transition that succeeded but a notification that fired twice."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Table-driven transitions** | Yes | Exhaustive testing; illegal and unauthorised are one lookup | Guards needing other fields sit outside the table |
| **Actor in the transition key** | Yes | Authorisation cannot be forgotten | A wider key, and a table to read |
| **Assignment as a field** | Yes | The kitchen and fulfilment vary independently | A guard on `pick_up` that the table cannot express |
| **Version-checked CAS** | Yes | Two racing drivers resolve to one winner, provably | A version on every mutation, and a lock per order |
| **`Money` in minor units** | Yes | Exact arithmetic; `allocate` distributes the remainder | A little ceremony at the boundaries where input is decimal |
| **Price snapshot at add-to-cart** | Yes | A menu edit cannot change a placed order | The cart cannot honour a "price drop before checkout" without an explicit rule |
| **Idempotency key on placement** | Yes | A double-tap is one order | The client must generate and persist a key |
| **Policy evaluated before the transition** | Yes | The refund reason survives the state change | An ordering requirement that is easy to invert |
| **`PricingBreakdown` rather than a total** | Yes | Every total is explainable | More fields to thread through |
| **Accept timeout on the order** | Yes | A sweeper can find silent orders | A deadline field and a scheduler |
| **`EtaEstimator` as a port** | Yes | Testable ETAs | An interface over arithmetic |
| **No observer in the aggregate** | Yes | The consistency boundary stays clean | Real-time push needs a separate notifier |
| **In-process lock, not a distributed CAS** | Yes, with the store path shown | Simplicity for a single process | The distributed version must be implemented before scaling out — the design does not change, the mechanism does |

**The trade to name unprompted:** *"Snapshotted prices mean a customer whose item goes on sale between add-to-cart and checkout pays the old price, and I would defend that: the alternative is a price that changes under a customer after they have committed, which is worse in every way, and honouring a *drop* specifically is a deliberate feature rather than the default behaviour of a reference."*

---

## 11. Q&A

**Q1. What is the hardest part of this problem?**
The modelling decision, not the code: **driver assignment is not a state**. Kitchen progress and fulfilment progress vary independently — a driver can be assigned while the food is still cooking, and the kitchen can be ready while no driver has accepted. Force both onto one enum and the state space becomes their cross product, you find yourself writing `ASSIGNED_AND_PREPARING`, and every new fact about an order multiplies the enum again. Once that decision is right, the rest is a table and four strategies. The second hardest part is the race, and it is only hard because the wrong version of it looks correct.

**Q2. Why is the transition table keyed by `(state, action)` with the actors in the value?**
Because "can this happen?" and "may this person do it?" are the same question asked twice, and separating them is how one of the two checks gets forgotten. With the actor in the key, an illegal transition and an unauthorised one are both a single failed lookup — the demo asserts both messages — and `allowed_actions(actor)` is derived from the same table, so a UI cannot offer a button the aggregate would reject. The cost is a slightly wider key and a table that has to be read rather than skimmed, and I would pay it for the fact that authorisation cannot be left out of a new transition.

**Q3. Why is `driver_id` a field rather than a state?**
Because it varies independently of the kitchen. A driver accepts while the food is `PREPARING`, the food becomes `READY` while the driver is en route, and neither change implies the other. Modelling it as a state means either the enum doubles — `PREPARING_ASSIGNED`, `READY_ASSIGNED`, `PREPARING_UNASSIGNED` — or the assignment is squeezed into a single `ASSIGNED` state that loses the kitchen position entirely. As a field it composes freely, and the one thing it cannot express is a *constraint between the two dimensions*, which is exactly what the `pick_up` guard is for: pick-up requires both `READY` and a driver.

**Q4. Walk through the assignment race and its fix.**
Eight drivers tap Accept on the same order. Each reads the order's version, checks it matches, checks no driver is set, and writes. Without a lock, two of them can interleave between the check and the write: both see version 3 and `driver_id is None`, both write, and both consider themselves the winner — and the code reads as obviously correct. The fix is that the compare and the set happen in one critical section, which is `OrderLock`; the demo runs exactly that scenario with a `threading.Barrier` and asserts one success and seven failures. The third guard matters too: a terminal order cannot be assigned, so a driver cannot accept an order that was cancelled while the offer was in flight.

**Q5. Why integer minor units and not a `Decimal` or a float?**
A float is simply wrong — the demo shows `0.1 + 0.2 != 0.3` for the inputs that matter — and `Decimal` is correct but brings a context, a precision setting and a rounding mode that have to be configured consistently everywhere and can be changed globally by an unrelated library. Integers remove the question: the arithmetic is exact, the only rounding is at the boundaries, and `percent` rounds once at the point of use. The one subtlety is that division needs care — hence `allocate`, which distributes the remainder one minor unit at a time rather than rounding each share, because rounded shares do not sum to the total and a reconciliation that is off by a cent is a failed reconciliation.

**Q6. How does the idempotency key work, and what is the trap?**
The client generates a key per checkout attempt and sends it with the request; the server looks the key up, returns the existing order if there is one, and otherwise creates one and records the mapping. The trap is that the lookup and the insert must be in the same critical section — two threads that both find no existing order and both create one have produced two orders from one tap, which is the exact failure the key was supposed to prevent. It is the same shape as the assignment CAS: the guard and the mutation must be atomic, and a guard that is merely adjacent to the mutation is not a guard.

**Q7. Why is the cancellation policy evaluated before the transition?**
Because after the transition the state is `CANCELLED` and the policy can no longer tell where the order came from — a `CANCELLED` order that was cancelled after acceptance and one that was cancelled after pickup are identical, and the refund differs. So the ordering is load-bearing, and the demo tests it by asserting that the event's note carries the reason, which is only possible if the decision was made first. The deeper point is that "what does this cost?" is a question about the state you were in, and any object that is asked after a state change has lost the information it needs.

**Q8. How do you handle a restaurant that never responds?**
An accept deadline is armed on the order at placement, and a system-actor sweep calls `check_accept_timeout`, which fires a `CANCEL` by `SYSTEM` if the deadline has passed and the order is still `PLACED`. Two details matter. The deadline lives **on the order** rather than in a scheduler's memory, so a sweeper can find the candidates by querying rather than by holding a timer per order. And the sweep is idempotent: a second sweep on a cancelled order does nothing, because `check_accept_timeout` returns `None` unless the state is `PLACED`. Both are in the demo, driven by an injected clock so the test does not sleep.

**Q9. What stops an order from being picked up without a driver, or cancelled mid-flight?**
Two different mechanisms, and separating them is the point. `pick_up` is a **guard**: the transition table says `READY + PICK_UP → PICKED_UP` by a driver, but it cannot say "and a driver must be assigned", because that is a fact about another field. So the guard lives in one method and the table stays a table. `PICKED_UP` having no `CANCEL` row at all is a **table** decision: once the food is in a bag, self-service cancellation is over and the remaining case — a driver who disappears — is an incident with its own policy and its own cost attribution, not a transition.

**Q10 (senior). Where does this design break at scale?**
Three places. The **in-process lock becomes a store-level compare-and-set**: the three guards move from Python into a `WHERE` clause, and the design does not change but the mechanism must, because a `threading.Lock` means nothing across two servers. The **`Order` aggregate becomes too big to load for every read**: a driver's device polling an order should not deserialise the history, so the read path needs projections — which is why `view_for` exists as a function of the actor rather than a method on the aggregate. And the **`_locks` map grows without bound** — one lock per order, held forever, is a memory leak; the real version needs a lock registry with eviction, which is the same problem as the [rate limiter's bounded key space](11-rate-limiter.md) §8.

**Q11 (senior). If two services both write the order, what changes?**
The version check stops being a lock and becomes the concurrency control of the store, which means the version must be incremented by the store rather than by the aggregate — otherwise two servers both read version 3 and both write version 4, and one of the writes is silently lost. That also forces a decision the in-process version lets you defer: **what is the consistency boundary?** If the kitchen state and the driver assignment are written by different services, the version becomes a shared token that either serialises them, which couples the two services, or it does not, which means the two dimensions genuinely are independent and should be stored as such. The single version is the simplest thing that works and it is also the thing that eventually has to be split.

**Q12. What would you test first, and what would you not bother testing?**
First the two things that lose money or produce impossible states: the race — many threads, one winner — and the money arithmetic including `allocate`'s remainder. Then the transition table **exhaustively**, because it is data and a loop over every `(state, action, actor)` triple is cheaper and more complete than any set of scenarios; the demo does exactly that for the terminal states and the cancel rows. What I would not test is the history's timestamps or the exact wording of a reason string — those are assertions that fail on a refactor that changed nothing, and the same reasoning appears in the [logger framework](12-logger-framework.md) about asserting on structure rather than prose.

---

### Pushback round

**P1. "You have four strategy interfaces and an aggregate for what is, at heart, a CRUD app with a status column."**
The status column is the smallest part of it, and the four strategies are the parts that actually differ between deployments. Pricing and cancellation differ by market — a hard-coded `if` ladder for refunds is precisely the thing that cannot be localised, and I have seen that movie. Ranking differs by supply density. The ETA is a port so a test does not have to wait. If the requirement really were "one market, one fee schedule, no refunds", then yes, the status column plus a pricing function is the right answer and I would write that. The design is sized for the follow-ups, and the first follow-up on this problem is always the refund policy.

**P2. "Your version check is on the whole aggregate, so a driver's acceptance fails if anybody changed anything — that is too coarse."**
That is a real cost and I would not hide it. The alternative is per-field versions, which means the concurrency control now encodes a belief about which fields are independent — and the moment two fields are in fact coupled, a per-field version permits an interleaving the whole-aggregate version forbids, silently. Coarse is the safe direction, and the practical objection is weaker than it looks: the operations that change an order between an offer and an acceptance are rare, and the cost of one is a retry with a fresh version, which the dispatch loop already handles by trying the next-ranked driver. If contention were measurably a problem I would narrow it deliberately, having decided which fields are genuinely independent — which is the same modelling question as the orthogonal-dimensions discussion.

**P3. "`unassign_driver` permits unassigning a `PICKED_UP` order, which you also list as a failure mode. That is a contradiction."**
It is a deliberate seam and I should have said so in the code as clearly as I say it here: the guard is `is_terminal`, so `PICKED_UP` is unassignable, and the incident policy in §7.4 is what handles that case. The tension is real — an order in `PICKED_UP` with no driver is an impossible state, so allowing the transition creates one, and forbidding it means the incident path has no way to record what happened. My answer is that the incident is a **separate object** that does not mutate the order's driver field: it records the incident, refunds, and attributes the cost, and the order stays as it was with a note. If I were tightening the code, `unassign_driver` would refuse `PICKED_UP` explicitly — and that is the change I would make first on review.

**P4. "Your `view_for` returns the driver's identity to the customer only after pickup, but the customer needs to know a driver is assigned."**
Correct, and the distinction the code misses is between **the order's internal driver id and the courier's publicly displayed identity**. The customer needs "a driver is on the way" from the moment of assignment, and needs the name and the vehicle only once the food is moving — which is a two-level projection rather than a boolean. As written, `payload["driver"]` is `None` between assignment and pickup, so the customer sees "no driver" for the whole preparation window, which is misleading rather than merely terse. The fix is to project an assignment status separately from an identity, and it is a good example of a visibility rule having more states than a flag can carry.

---

## 12. Cheat-sheet summary + cross-links

| Concept | One-line truth |
|---|---|
| The shape | `Order` (aggregate) + a transition **table** + four strategies (pricing, refunds, ranking, ETA) + a façade |
| The key modelling insight | **Driver assignment is not a state** — it is orthogonal to kitchen progress |
| The tell | If you are writing `ASSIGNED_AND_PREPARING`, you have cross-producted two independent things |
| The table's key | `(from_state, action) → (to_state, {allowed actors})` — illegal and unauthorised are one lookup |
| Derived from the table | `allowed_actions(actor)`, so a UI cannot offer what the aggregate would reject |
| What the table cannot say | "…and a driver must be assigned": a **guard**, in one method, for facts about other fields |
| Money | **Integer minor units.** `0.1 + 0.2 != 0.3` is not a curiosity when it is a total |
| Splitting money | `allocate` distributes the remainder one unit at a time; rounded shares do not sum |
| Tax order | Food + service fee, then tip added after — a policy, spelled out line by line |
| Price snapshot | At **add-to-cart**, not at checkout; the order holds an immutable lines tuple |
| Idempotency | A client key; the **lookup and the insert share one lock** |
| The race that matters | N drivers, one order, **exactly one winner** — `threading.Barrier` in the demo |
| Why a CAS is not enough | The check and the write must be **atomic**; adjacent is not atomic |
| The three assignment guards | Stale version · already assigned · terminal state |
| Cancellation | The **machine** says who may cancel; the **policy** says what it costs |
| The ordering trap | Evaluate the policy **before** the transition, or `CANCELLED` loses where it came from |
| Refund tiers | Free before acceptance · delivery fee after · 50% mid-prep · nothing once ready |
| Accept timeout | A deadline **on the order** + a system sweep; idempotent by checking the state |
| `PICKED_UP` has no cancel row | Deliberate: self-service cancellation ends once the food is in a bag |
| A lost driver after pickup | An **incident** with its own policy, not a transition — different owner, different cost |
| Scale-out | The lock becomes a store-level CAS; the guards move into the `WHERE` clause |
| Scale-out leak | One lock per order, forever, is a memory leak — the lock registry needs eviction |
| Patterns | State (table), Strategy ×4, Value Object, Aggregate, Repository, Façade, Ports & Adapters |
| Rejected | Observer in the aggregate (an event bus inside a consistency boundary), Singleton |

**The three sentences to memorise**

1. *"Driver assignment is not a state — it is orthogonal to kitchen progress, and cross-producting them is what produces `ASSIGNED_AND_PREPARING`."*
2. *"The transition table maps (state, action) to a target and the allowed actors, so an illegal transition and an unauthorised one are the same failed lookup."*
3. *"Money is integer minor units, and the assignment CAS needs a lock — a version check that is merely adjacent to the write is a race that passes code review."*

### Cross-links

- **The LLD answer framework this file follows, and the HLD/LLD boundary:** [LLD/01 What is LLD and How to Answer](../01-what-is-lld-and-how-to-answer.md)
- **The system view of this domain at HLD level — dispatch, geo-sharding, the courier marketplace:** [Uber Ride-Hailing](../../Case-Studies/HLD/06-uber-ride-hailing.md)
- **One level down on the matching problem — geo index, strategies, trip state machine:** [Uber Matching LLD](../../Case-Studies/LLD/03-uber-matching-lld.md)
- **The same "pure decision returned as a value" split:** [Rate Limiter](11-rate-limiter.md) §5 (`Decision`) · [Snake and Ladder](15-snake-and-ladder.md) §5 (`MoveResolution`)
- **A table-driven state machine with a history, and actor-like guards:** [Vending Machine](10-vending-machine.md) §5 (states and transitions) · §7.4 (payment idempotency)
- **Money-like exactness and bounded maps at scale:** [Rate Limiter](11-rate-limiter.md) §8 (bounded key space, the same eviction problem as the lock registry)
- **Injected clocks and ports as the testability seam:** [Rate Limiter](11-rate-limiter.md) §5 (`Clock`) · [Snake and Ladder](15-snake-and-ladder.md) §5 (`Dice`) · §6 here (`EtaEstimator`, `clock`)
- **Idempotency on a write path, in a different system:** [URL Shortener LLD](../../Case-Studies/LLD/01-url-shortener-lld.md) (idempotent short-code creation) · [Notification System LLD](../../Case-Studies/LLD/05-notification-system-lld.md) (deduplication)
- **Asserting on structure rather than prose, in tests:** [Logger Framework](12-logger-framework.md) §11 Q12
- **Cheat sheet, pattern tables and the classic-problems list:** [LLD Cheatsheet](../../Cheat-Sheets/07-lld-cheatsheet.md)
- **Question bank:** [LLD Questions](../../Interview-Questions/08-lld-questions.md)
