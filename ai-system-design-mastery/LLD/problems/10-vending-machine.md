# LLD 10 — Vending Machine

> **Grounding note.** The reference repositories for this knowledge base contain **no OOP/LLD material**, and the source transcripts name the classic LLD problems (parking lot, LRU, elevator, vending machine) **without teaching them in depth**. This file is built from first principles plus the framing in [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md). It is not transcript-grounded and does not claim to be.

> **TL;DR** — The vending machine looks like a beginner problem and is actually a **change-making problem wearing a state machine**. The three ideas that carry the design are: (1) **money is integers in minor units and nothing else** — a float cent will eventually break a ledger; (2) **a State pattern** (`Idle → HasFunds → Dispensing`) because the legality of "insert coin", "select item" and "refund" changes with the phase, and a boolean soup cannot express that; (3) **change-making against a finite coin bank** — greedy fails the moment a denomination is exhausted, so the change algorithm must be a bounded search over the actual inventory of coins, not a table of denominations. The single strongest answer in the interview is: *"the customer's own inserted coins are part of the change pool, and if change cannot be made the sale must be refused rather than dispensed at a loss."*

> **60-second interview answer:** "I'd clarify first: which denominations, does it accept notes, is there a card reader, does it give change, does it ever need exact change? Assuming coins and notes, change given, one customer at a time. The entities are `Product` and `Slot` (a slot holds a product and a count), `Inventory` as the collection, a `CashStore` that holds the coin bank **and** answers `make_change(amount)`, a `PaymentMethod` strategy with a cash implementation and a card implementation, a `VendingMachineState` hierarchy for `Idle`, `HasFunds` and `Dispensing`, and a `VendingMachine` facade that owns those. Money is stored as integer cents. The two hard parts are the state machine — because `insert_money` is legal in `Idle` but nonsensical in `Dispensing` — and change-making, which is not a lookup: it is a bounded backtracking search over the coins the machine actually holds, because greedy fails when the bank runs out of a denomination. And the sale has to be refused when change cannot be made, which means the change check must happen *before* dispensing, not after."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **Which denominations?** | The `Coin` / `Note` enums and the change algorithm's branching factor | Coins 5/10/25/100c plus a 500c note; make it a configurable set |
| 2 | **Does it accept notes?** | Whether the escrow must handle a denomination larger than any item | Yes — and this is what makes change-making hard, because a 500c note against a 65c item needs 435c of change |
| 3 | **Does it give change?** | Whether `make_change` exists at all — and this is the question that decides the difficulty | Yes |
| 4 | **Card / contactless?** | Whether `PaymentMethod` needs to be a Strategy on day one | "I'll design one payment method and make it pluggable" — and then show the card subclass |
| 5 | **Can the machine run out of change?** | Whether there is an exact-change mode and a `CannotMakeChange` path | Yes — a real machine does, and handling it is the senior signal |
| 6 | **One customer at a time, or a shared/remote machine?** | The concurrency model | One physical customer; but make the core thread-safe because a remote API in front of it is a two-line change |
| 7 | **Admin operations: restock, collect cash, set prices?** | Whether there is an admin surface and whether prices are mutable | Yes — `restock`, `collect_cash`, `set_price` |
| 8 | **Refund / cancel mid-transaction?** | Whether the escrow is separable from the bank | Yes — refund returns exactly what was inserted |
| 9 | **Inventory hardware: does a slot jam?** | Whether `Dispensing` can fail and must roll back | Yes — a jam is the canonical failure, and the rollback is the interesting part |
| 10 | **Does it need to report telemetry (low stock, cash level)?** | Whether there is an Observer | Yes — a `MachineObserver` for restock alerts |

### The defaults to assume, stated plainly

> "I'll assume: **four coin denominations plus notes, change is given, one cash payment method made pluggable for card, an admin surface for restock and cash collection, exact-change mode when the bank is short, and a jam as a recoverable failure**. Is that the shape you want, or should I simplify to coins only?"

### Non-goals

- No physical hardware driver, no coin-acceptor protocol — I model the *interface* (`Dispenser`, `CoinAcceptor`) and fake it.
- No UI beyond a small ASCII display string.
- No distributed fleet management — that is HLD (see [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md)).

### The four requirements that actually constrain the design

1. **Money must be exact.** Integers in minor units. Every currency bug traces back to a float.
2. **The same input is legal in one state and illegal in another.** Inserting a coin mid-dispense is a no-op at best and a double-count at worst.
3. **Change-making is bounded by the coin bank, not by arithmetic.** `65 - 60 = 5` is arithmetic; "do I have a 5c coin left?" is the real question.
4. **The customer's inserted coins become the machine's coins.** So the change pool at decision time is *bank + escrow*, and that is the subtlety most candidates miss.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `Money` | Convention | Every amount is an `int` in minor units; a formatting helper only | Arithmetic — it is just an int |
| `Coin` | `IntEnum` | The coin denominations and their cent values | Counting |
| `Note` | `IntEnum` | The note denominations | Counting |
| `Product` | Immutable value | SKU, display name, price in cents | Stock |
| `Slot` | Entity | A code (`A1`), the product in it, the quantity, the capacity | Money |
| `Inventory` | Collection | Slots by code; add/remove; low-stock query | Prices |
| `CashStore` | Entity | The counts of each coin/note; `make_change`, `deposit`, `withdraw`, `total` | Prices, products |
| `PaymentMethod` (ABC) | Strategy interface | `balance`, `add_funds`, `charge`, `refund`, `change_due` | Stock |
| `CashPayment` | Strategy impl | Escrow of inserted coins and notes; refund returns them | The bank |
| `CardPayment` | Strategy impl | Charge exactly the price via a terminal port; change is always zero | Coins |
| `CardTerminal` (Protocol) | Port | `authorize(amount) -> bool`, `void(ref)` — the hardware seam | Business rules |
| `VendingMachineState` (ABC) | State interface | What `insert_money` / `select` / `refund` mean **in this phase** | Where the money is stored |
| `IdleState`, `HasFundsState`, `DispensingState`, `OutOfServiceState` | State impls | Transition rules and the legal operations per phase | — |
| `Dispenser` (Protocol) | Port | `dispense(slot) -> bool` — the hardware seam that can fail (jam) | Stock accounting |
| `PurchaseResult` | Immutable value | `status`, `product`, `change` — the outcome of a vend | — |
| `VendingMachine` | Facade / aggregate root | Owns inventory, bank, state, payment; exposes `insert_money`, `select`, `refund`, `restock`, `collect_cash` | The rules themselves |
| `MachineObserver` (Protocol) | Observer | Notified on low stock, low change, jams | — |

**The one-line split to say out loud:** *"`VendingMachine` is a facade over four collaborators — inventory, cash bank, payment strategy and state. It holds no rules of its own."*

---

## 3. Class diagram (ASCII)

```
                    ┌──────────────────────────────┐
                    │        VendingMachine        │  (Facade / Aggregate root)
                    │──────────────────────────────│
                    │ - inventory : Inventory      │
                    │ - bank      : CashStore      │
                    │ - payment   : PaymentMethod  │
                    │ - state     : VendingMachine │
                    │                State         │
                    │ - dispenser : Dispenser      │
                    │ - observers : list           │
                    │ - lock      : RLock          │
                    │──────────────────────────────│
                    │ + insert_money(coin)         │
                    │ + select(code) PurchaseResult│
                    │ + refund() dict[Coin,int]    │
                    │ + restock(code, qty)         │
                    │ + collect_cash() int         │
                    │ + set_state(s)   (package)   │
                    └───┬───────┬─────────┬────────┘
                        │       │         │
        ┌───────────────┘       │         └───────────────┐
        ▼                       ▼                         ▼
┌───────────────┐   ┌─────────────────────┐   ┌───────────────────────┐
│  Inventory    │   │ «interface»         │   │ «interface»           │
│───────────────│   │ VendingMachineState │   │ PaymentMethod         │
│ slots: dict   │   │─────────────────────│   │───────────────────────│
│───────────────│   │ +insert_money(m,c)  │   │ +balance : int        │
│ +slot(code)   │   │ +select(m,c) Result │   │ +add_funds(amount)    │
│ +add(code,n)  │   │ +refund(m) dict     │   │ +charge(amount) bool  │
│ +remove(code) │   │ +restock(m,c,n)     │   │ +refund() int         │
│ +low_stock()  │   └──────▲──────────────┘   │ +change_due(price) int│
└───────┬───────┘          │                  └───────▲───────────────┘
        │                  │ extends                  │ implements
        ▼                  │                          ├──────────────┐
┌───────────────┐  ┌───────┴────┬──────────┬─────────┐│              │
│     Slot      │  │            │          │         ││              │
│───────────────│┌─┴──────┐┌────┴────┐┌────┴──────┐┌─┴┴───────────┐┌─┴──────────┐
│ code: str     ││ Idle   ││HasFunds ││Dispensing ││ CashPayment  ││ CardPayment│
│ product: Prod ││ State  ││ State   ││ State     ││ (escrow of   ││ (terminal  │
│ quantity: int │└────────┘└─────────┘└───────────┘│  inserted    ││  port)     │
│ capacity: int │                                   │  coins)      │└─────┬──────┘
└───────────────┘                                   └──────┬───────┘      │
                                                           │ uses         │ uses
                                              ┌────────────▼───────┐  ┌───▼────────────┐
                                              │    CashStore       │  │ «interface»    │
                                              │────────────────────│  │ CardTerminal   │
                                              │ counts: dict       │  │ +authorize(amt)│
                                              │────────────────────│  │ +void(ref)     │
                                              │ +make_change(amt)  │  └────────────────┘
                                              │   -> dict|None     │
                                              │ +deposit(coins)    │
                                              │ +withdraw(coins)   │
                                              │ +total() int       │
                                              └────────────────────┘

        ┌───────────────────┐        ┌────────────────────┐
        │ «interface»       │        │  PurchaseResult    │
        │ Dispenser         │        │────────────────────│
        │ +dispense(slot)   │        │ status: VendStatus │
        │   -> bool  (jam)  │        │ product: Product?  │
        └───────────────────┘        │ change: dict       │
                                     └────────────────────┘
```

**Read the diagram this way:** two **Strategy** interfaces (`PaymentMethod`, and the change algorithm inside `CashStore`), one **State** hierarchy, one **Facade** (`VendingMachine`), one **Observer**, and two **ports** (`CardTerminal`, `Dispenser`) that isolate the hardware.

---

## 4. Interfaces / abstract classes / enums

```python
class Coin(IntEnum):
    FIVE = 5
    TEN = 10
    TWENTY_FIVE = 25
    ONE_DOLLAR = 100


class Note(IntEnum):
    FIVE_DOLLARS = 500
    TEN_DOLLARS = 1000


class VendStatus(Enum):
    SUCCESS = "success"
    SOLD_OUT = "sold_out"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CANNOT_MAKE_CHANGE = "cannot_make_change"
    JAMMED = "jammed"
    INVALID_SLOT = "invalid_slot"
    OUT_OF_SERVICE = "out_of_service"


class PaymentMethod(ABC):
    """Strategy. Cash and card differ in *everything* except this contract.

    `collect` / `restore` are the rollback pair: `collect` takes the funds out
    of the customer's hands when a sale commits, and `restore` puts them back
    if the sale has to be undone (a jam). Keeping them separate from
    `charge` is what makes the rollback path expressible for both a coin
    escrow and a card authorisation.
    """

    @property
    @abstractmethod
    def balance(self) -> int: ...

    @abstractmethod
    def sufficient(self, amount: int) -> bool:
        """Can this method plausibly cover `amount`? A card always says yes —
        it is the authorisation that may decline."""

    @abstractmethod
    def add_funds(self, amount: int) -> None: ...

    @abstractmethod
    def charge(self, amount: int) -> bool: ...

    @abstractmethod
    def collect(self) -> dict[int, int]:
        """Take the customer's funds (cash: the coin escrow). Card: empty."""

    @abstractmethod
    def restore(self, funds: dict[int, int]) -> None:
        """Undo a `collect` — return the funds to the customer's credit, or
        void the card authorisation."""

    @abstractmethod
    def change_due(self, price: int) -> int:
        """How much the machine owes the customer after charging `price`."""


class VendingMachineState(ABC):
    @abstractmethod
    def insert_money(self, machine: "VendingMachine", amount: int) -> None: ...

    @abstractmethod
    def select(self, machine: "VendingMachine", code: str) -> "PurchaseResult": ...

    @abstractmethod
    def refund(self, machine: "VendingMachine") -> dict[int, int]: ...

    @abstractmethod
    def restock(self, machine: "VendingMachine", code: str, quantity: int) -> None: ...


class CardTerminal(Protocol):
    """Port: the only thing that knows about the payment network."""

    def authorize(self, amount: int) -> tuple[bool, str]: ...
    def void(self, reference: str) -> None: ...


class Dispenser(Protocol):
    """Port: hardware. Can fail, and failure must be recoverable."""

    def dispense(self, slot: "Slot") -> bool: ...


class MachineObserver(Protocol):
    def on_low_stock(self, machine: "VendingMachine", slot: "Slot") -> None: ...
    def on_low_change(self, machine: "VendingMachine", coin: "Coin") -> None: ...
    def on_jam(self, machine: "VendingMachine", slot: "Slot") -> None: ...
```

**Why `PaymentMethod` is a Strategy and not an `if is_card:`:** the card path shares *nothing* with the cash path — no escrow, no change, no coin bank, a different failure model (network decline versus physical jam) and a different refund mechanism (a void versus a coin return). One interface, two entirely different implementations, is the textbook case for the pattern. Do **not** introduce it until the second implementation exists — but the card reader is a near-certain follow-up, so it is worth mentioning that you would add it then.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** Interview languages vary. In Java this is `abstract class` + `enum` + `synchronized`; in C++ the state objects would be `std::unique_ptr`; in Go you would use interfaces and a `sync.Mutex`. The design does not change. Python 3.10+ because of `IntEnum` arithmetic and `|` type unions.

```python
"""vending_machine.py — a complete, runnable vending machine.

Design summary
--------------
Money          : int cents, always. Never float.
CashStore      : the coin bank + make_change() via bounded backtracking.
PaymentMethod  : Strategy — CashPayment (escrow) / CardPayment (terminal port).
...State       : State pattern — Idle / HasFunds / Dispensing / OutOfService.
VendingMachine : Facade owning inventory, bank, payment and state.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Protocol


# --------------------------------------------------------------------------
# 1. Money — integers in minor units, and nothing else
# --------------------------------------------------------------------------

class Coin(IntEnum):
    FIVE = 5
    TEN = 10
    TWENTY_FIVE = 25
    ONE_DOLLAR = 100


class Note(IntEnum):
    FIVE_DOLLARS = 500
    TEN_DOLLARS = 1000


# Every accepted denomination, coins and notes, as plain ints.
ACCEPTED: tuple[int, ...] = tuple(int(c) for c in Coin) + tuple(int(n) for n in Note)


def format_money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100}.{cents % 100:02d}"


# --------------------------------------------------------------------------
# 2. Products, slots, inventory
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price: int  # cents

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("price must be positive cents")


class Slot:
    def __init__(self, code: str, product: Product, quantity: int = 0, capacity: int = 10) -> None:
        if not code:
            raise ValueError("slot code required")
        if quantity < 0 or quantity > capacity:
            raise ValueError("quantity out of range")
        self.code = code
        self.product = product
        self.quantity = quantity
        self.capacity = capacity

    @property
    def is_empty(self) -> bool:
        return self.quantity == 0

    @property
    def is_low(self) -> bool:
        return self.quantity <= max(1, self.capacity // 5)

    def __repr__(self) -> str:
        return f"<Slot {self.code} {self.product.name} x{self.quantity}>"


class Inventory:
    def __init__(self, slots: Optional[dict[str, Slot]] = None) -> None:
        self._slots: dict[str, Slot] = dict(slots or {})

    def slot(self, code: str) -> Slot:
        try:
            return self._slots[code]
        except KeyError as exc:
            raise KeyError(f"no such slot: {code!r}") from exc

    def add(self, code: str, quantity: int = 1) -> int:
        slot = self.slot(code)
        room = slot.capacity - slot.quantity
        added = min(room, quantity)
        slot.quantity += added
        return added

    def remove_one(self, code: str) -> None:
        slot = self.slot(code)
        if slot.is_empty:
            raise KeyError(f"slot {code} is empty")
        slot.quantity -= 1

    def codes(self) -> list[str]:
        return sorted(self._slots)

    def low_stock(self) -> list[Slot]:
        return [s for s in self._slots.values() if s.is_low]

    def __iter__(self):
        return iter(self._slots.values())


# --------------------------------------------------------------------------
# 3. CashStore — the coin bank and the change algorithm
# --------------------------------------------------------------------------

class CashStore:
    """Holds counts of every denomination and answers `make_change`.

    The change algorithm is a bounded backtracking search, NOT greedy.
    Greedy fails on the *bank*, not on the arithmetic:

        bank = {25: 1, 10: 3, 5: 0},  want 30c
          greedy : takes the 25 (largest first), remainder 5, and there is
                   no 5c coin, so nothing further can help -> FAIL
          correct: 10 + 10 + 10 = 30c                        -> SUCCEEDS

    Greedy commits to the largest denomination before it knows whether the
    remainder is payable, and it has no way to back out. The search below
    tries every count of each denomination from the maximum down to zero and
    recurses, so it finds a combination whenever one exists.
    """

    def __init__(self, counts: Optional[dict[int, int]] = None) -> None:
        self._counts: dict[int, int] = {value: 0 for value in ACCEPTED}
        if counts:
            for value, qty in counts.items():
                if value not in self._counts:
                    raise ValueError(f"unsupported denomination: {value}")
                if qty < 0:
                    raise ValueError("negative quantity")
                self._counts[value] = qty

    # ---- queries ----------------------------------------------------------

    def count(self, value: int) -> int:
        return self._counts.get(value, 0)

    def total(self) -> int:
        return sum(value * qty for value, qty in self._counts.items())

    def snapshot(self) -> dict[int, int]:
        return {v: q for v, q in self._counts.items() if q}

    # ---- mutation ---------------------------------------------------------

    def deposit(self, cash: dict[int, int]) -> None:
        for value, qty in cash.items():
            if qty < 0:
                raise ValueError("negative deposit")
            self._counts[value] = self._counts.get(value, 0) + qty

    def withdraw(self, cash: dict[int, int]) -> None:
        for value, qty in cash.items():
            if self._counts.get(value, 0) < qty:
                raise ValueError(f"insufficient {value}c coins")
        for value, qty in cash.items():
            self._counts[value] -= qty

    def collect_all(self) -> dict[int, int]:
        """Admin: empty the bank. Refuses if it cannot be emptied atomically."""
        taken = self.snapshot()
        self._counts = {value: 0 for value in ACCEPTED}
        return taken

    # ---- change making ----------------------------------------------------

    def make_change(self, amount: int, extra: Optional[dict[int, int]] = None) -> Optional[dict[int, int]]:
        """Return a denomination->count map summing exactly to `amount`, or None.

        `extra` lets the caller add the customer's just-inserted coins to the
        pool, which is correct: those coins are physically in the machine.
        """
        if amount < 0:
            raise ValueError("negative change")
        if amount == 0:
            return {}
        pool = dict(self._counts)
        if extra:
            for value, qty in extra.items():
                pool[value] = pool.get(value, 0) + qty

        denominations = sorted(pool, reverse=True)
        # Prune with a suffix-sum bound: the largest reachable total from
        # denominations[i:] given the pool. Without this the search is
        # exponential on pathological inputs.
        suffix_max = [0] * (len(denominations) + 1)
        for i in range(len(denominations) - 1, -1, -1):
            suffix_max[i] = suffix_max[i + 1] + denominations[i] * pool[denominations[i]]

        return self._search(amount, denominations, 0, pool, suffix_max)

    @staticmethod
    def _search(
        amount: int,
        denominations: list[int],
        index: int,
        pool: dict[int, int],
        suffix_max: list[int],
    ) -> Optional[dict[int, int]]:
        if amount == 0:
            return {}
        if index >= len(denominations) or suffix_max[index] < amount:
            return None  # provably impossible from here

        value = denominations[index]
        available = min(pool[value], amount // value)
        for take in range(available, -1, -1):
            remainder = amount - take * value
            if remainder and suffix_max[index + 1] < remainder:
                continue  # prune this branch
            sub = CashStore._search(remainder, denominations, index + 1, pool, suffix_max)
            if sub is not None:
                if take:
                    sub[value] = take
                return sub
        return None

    def can_make_change(self, amount: int, extra: Optional[dict[int, int]] = None) -> bool:
        return self.make_change(amount, extra) is not None


# --------------------------------------------------------------------------
# 4. Payment strategies
# --------------------------------------------------------------------------

class PaymentMethod(ABC):
    """Cash and card differ in everything except this contract.

    `collect` / `restore` are the rollback pair: `collect` takes the funds out
    of the customer's hands when a sale commits, and `restore` puts them back
    if the sale has to be undone (a jam). Keeping them separate from `charge`
    is what makes the rollback expressible for both a coin escrow and a card
    authorisation.
    """

    @property
    @abstractmethod
    def balance(self) -> int: ...

    @abstractmethod
    def sufficient(self, amount: int) -> bool:
        """Can this method plausibly cover `amount`? A card always says yes —
        it is the authorisation that may decline."""

    @abstractmethod
    def add_funds(self, amount: int) -> None: ...

    @abstractmethod
    def charge(self, amount: int) -> bool: ...

    @abstractmethod
    def collect(self) -> dict[int, int]:
        """Take the customer's funds (cash: the coin escrow). Card: empty."""

    @abstractmethod
    def restore(self, funds: dict[int, int]) -> None:
        """Undo a `collect` — restore the customer's credit, or void the card."""

    @abstractmethod
    def change_due(self, price: int) -> int: ...


class CashPayment(PaymentMethod):
    """Escrow: inserted coins are held aside until the sale commits or refunds."""

    def __init__(self) -> None:
        self._escrow: dict[int, int] = {}

    @property
    def balance(self) -> int:
        return sum(value * qty for value, qty in self._escrow.items())

    @property
    def escrow(self) -> dict[int, int]:
        return dict(self._escrow)

    def sufficient(self, amount: int) -> bool:
        return self.balance >= amount

    def add_funds(self, amount: int) -> None:
        if amount not in ACCEPTED:
            raise ValueError(f"unsupported denomination: {amount}")
        self._escrow[amount] = self._escrow.get(amount, 0) + 1

    def charge(self, amount: int) -> bool:
        return self.balance >= amount

    def collect(self) -> dict[int, int]:
        taken = dict(self._escrow)
        self._escrow = {}
        return taken

    def restore(self, funds: dict[int, int]) -> None:
        for value, qty in funds.items():
            self._escrow[value] = self._escrow.get(value, 0) + qty

    def change_due(self, price: int) -> int:
        return self.balance - price


class CardPayment(PaymentMethod):
    """Charges exactly the price. There is no such thing as card change."""

    def __init__(self, terminal: "CardTerminal") -> None:
        self._terminal = terminal
        self._reference: Optional[str] = None
        self._charged = 0

    @property
    def balance(self) -> int:
        return 0

    def sufficient(self, amount: int) -> bool:
        # A card is not a stored-value medium: the authorisation decides.
        return True

    def add_funds(self, amount: int) -> None:
        raise NotImplementedError("a card is not topped up at the machine")

    def charge(self, amount: int) -> bool:
        approved, reference = self._terminal.authorize(amount)
        if approved:
            self._reference = reference
            self._charged = amount
        return approved

    def collect(self) -> dict[int, int]:
        return {}  # nothing physical to take

    def restore(self, funds: dict[int, int]) -> None:
        if self._reference is not None:
            self._terminal.void(self._reference)
            self._reference = None
            self._charged = 0

    def change_due(self, price: int) -> int:
        return 0


class FakeTerminal:
    """A test double for the port. Approves everything under a limit."""

    def __init__(self, limit: int = 10_000) -> None:
        self.limit = limit
        self._counter = 0

    def authorize(self, amount: int) -> tuple[bool, str]:
        self._counter += 1
        reference = f"ref-{self._counter}"
        return (amount <= self.limit), reference

    def void(self, reference: str) -> None:
        return None


# --------------------------------------------------------------------------
# 5. Result value object
# --------------------------------------------------------------------------

class VendStatus(Enum):
    SUCCESS = "success"
    SOLD_OUT = "sold_out"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CANNOT_MAKE_CHANGE = "cannot_make_change"
    JAMMED = "jammed"
    INVALID_SLOT = "invalid_slot"
    OUT_OF_SERVICE = "out_of_service"


@dataclass(frozen=True)
class PurchaseResult:
    status: VendStatus
    product: Optional[Product] = None
    change: dict[int, int] = field(default_factory=dict)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is VendStatus.SUCCESS


# --------------------------------------------------------------------------
# 6. Hardware ports (faked)
# --------------------------------------------------------------------------

class Dispenser(Protocol):
    def dispense(self, slot: Slot) -> bool: ...


class PerfectDispenser:
    def dispense(self, slot: Slot) -> bool:
        return True


class FlakyDispenser:
    """Jams on a designated slot, so the rollback path is exercisable."""

    def __init__(self, jam_code: str) -> None:
        self.jam_code = jam_code

    def dispense(self, slot: Slot) -> bool:
        return slot.code != self.jam_code


class MachineObserver(Protocol):
    def on_low_stock(self, machine: "VendingMachine", slot: Slot) -> None: ...
    def on_low_change(self, machine: "VendingMachine", value: int) -> None: ...
    def on_jam(self, machine: "VendingMachine", slot: Slot) -> None: ...


# --------------------------------------------------------------------------
# 7. The VendingMachine facade
# --------------------------------------------------------------------------

class VendingMachine:
    def __init__(
        self,
        inventory: Inventory,
        bank: Optional[CashStore] = None,
        dispenser: Optional[Dispenser] = None,
        observers: Optional[list[MachineObserver]] = None,
    ) -> None:
        self.inventory = inventory
        self.bank = bank if bank is not None else CashStore()
        self.dispenser = dispenser if dispenser is not None else PerfectDispenser()
        self.observers = list(observers or [])

        self._payment: PaymentMethod = CashPayment()
        self._state: VendingMachineState = IdleState()
        self._lock = threading.RLock()
        self._display = "INSERT COIN"

    # ---- collaborators the states reach through ---------------------------

    @property
    def payment(self) -> PaymentMethod:
        return self._payment

    @property
    def state(self) -> "VendingMachineState":
        return self._state

    @property
    def display(self) -> str:
        return self._display

    def set_display(self, text: str) -> None:
        self._display = text

    def set_state(self, state: "VendingMachineState") -> None:
        self._state = state

    def set_payment(self, payment: PaymentMethod) -> None:
        self._payment = payment

    # ---- public API (all locked) ------------------------------------------

    def insert_money(self, amount: int) -> None:
        with self._lock:
            self._state.insert_money(self, amount)

    def select(self, code: str) -> PurchaseResult:
        with self._lock:
            return self._state.select(self, code)

    def refund(self) -> dict[int, int]:
        with self._lock:
            return self._state.refund(self)

    def restock(self, code: str, quantity: int) -> None:
        with self._lock:
            self._state.restock(self, code, quantity)

    def begin_card_payment(self, terminal: "CardTerminal") -> None:
        """The card reader is pressed: switch the payment strategy and move to
        HasFundsState, because 'funds are available' is now true by strategy."""
        with self._lock:
            self.set_payment(CardPayment(terminal))
            self.set_state(HasFundsState())
            self.set_display("PRESENT CARD")

    def collect_cash(self) -> dict[int, int]:
        with self._lock:
            if self._payment.balance:
                raise RuntimeError("cannot collect cash during a transaction")
            return self.bank.collect_all()

    def set_price(self, code: str, price: int) -> None:
        with self._lock:
            slot = self.inventory.slot(code)
            slot.product = Product(slot.product.sku, slot.product.name, price)

    def service(self) -> None:
        with self._lock:
            self._payment.restore(self._payment.collect())  # cancel cleanly
            self._reset_payment()
            self.set_state(OutOfServiceState())
            self.set_display("OUT OF SERVICE")

    def restore(self) -> None:
        with self._lock:
            self.set_state(IdleState())
            self.set_display("INSERT COIN")

    # ---- the vend algorithm, called by HasFundsState ----------------------

    def _vend(self, slot: Slot) -> PurchaseResult:
        """The whole transaction, in the only correct order.

        Order matters:
          1. stock check                (cheap, no side effects)
          2. funds check                (cheap)
          3. change affordability check (BEFORE any irreversible action)
          4. commit the payment         (escrow -> bank, change -> customer)
          5. dispense  <-- the only irreversible step
          6. roll back if the dispense fails
        """
        if slot.is_empty:
            return PurchaseResult(VendStatus.SOLD_OUT, detail=f"{slot.code} is empty")

        price = slot.product.price          # read ONCE; never re-read mid-vend
        if not self._payment.sufficient(price):
            return PurchaseResult(
                VendStatus.INSUFFICIENT_FUNDS,
                detail=f"need {format_money(price)}, have {format_money(self._payment.balance)}",
            )

        change_due = self._payment.change_due(price)

        # The customer's inserted coins join the pool: they are in the machine.
        extra = self._cash_escrow() or None
        change = self.bank.make_change(change_due, extra) if change_due else {}
        if change is None:
            return PurchaseResult(
                VendStatus.CANNOT_MAKE_CHANGE,
                detail=f"cannot make {format_money(change_due)} in change",
            )

        # ---- commit ----
        # charge() validates/authorises; collect() takes physical custody.
        # Charge MUST come first: for cash, collect() empties the escrow and
        # would make the balance check fail.
        if not self._payment.charge(price):
            return PurchaseResult(VendStatus.INSUFFICIENT_FUNDS, detail="payment declined")
        escrow = self._payment.collect()

        if escrow:
            self.bank.deposit(escrow)             # escrow becomes banked cash
        if change:
            self.bank.withdraw(change)            # then the change leaves the bank

        self.set_state(DispensingState())
        delivered = self.dispenser.dispense(slot)

        if not delivered:
            # Roll back to EXACTLY the pre-vend state:
            #   bank  : undo the withdrawal, then undo the deposit
            #   credit: the customer's coins return to escrow (or the card is voided)
            if change:
                self.bank.deposit(change)
            if escrow:
                self.bank.withdraw(escrow)
            self._payment.restore(escrow)
            self._reset_payment()
            # The customer's credit was restored, so the machine must be in a
            # state where they can retry or take it back — NOT Idle, which
            # would silently swallow the refund.
            self.set_state(HasFundsState() if self._payment.balance else IdleState())
            self.set_display("JAMMED - SEE STAFF")
            for observer in self.observers:
                observer.on_jam(self, slot)
            return PurchaseResult(VendStatus.JAMMED, detail=f"{slot.code} jammed")

        self.inventory.remove_one(slot.code)      # stock moves only on success
        self._reset_payment()
        self.set_state(IdleState())
        self.set_display(format_money(price))
        self._notify_stock(slot)
        return PurchaseResult(VendStatus.SUCCESS, product=slot.product, change=change)

    def _cash_escrow(self) -> dict[int, int]:
        return self._payment.escrow if isinstance(self._payment, CashPayment) else {}

    def _reset_payment(self) -> None:
        """Return to a clean cash acceptor — UNLESS the customer still has a
        credit. After a jam rollback the funds have been restored to escrow,
        and replacing the payment object would silently destroy them."""
        if self._payment.balance:
            return
        self.set_payment(CashPayment())

    def _notify_stock(self, slot: Slot) -> None:
        if slot.is_low:
            for observer in self.observers:
                observer.on_low_stock(self, slot)


# --------------------------------------------------------------------------
# 8. States
# --------------------------------------------------------------------------

class VendingMachineState(ABC):
    @abstractmethod
    def insert_money(self, machine: VendingMachine, amount: int) -> None: ...

    @abstractmethod
    def select(self, machine: VendingMachine, code: str) -> PurchaseResult: ...

    @abstractmethod
    def refund(self, machine: VendingMachine) -> dict[int, int]: ...

    @abstractmethod
    def restock(self, machine: VendingMachine, code: str, quantity: int) -> None: ...


class IdleState(VendingMachineState):
    """No funds inserted. The only legal money operation is an insert."""

    def insert_money(self, machine: VendingMachine, amount: int) -> None:
        if not isinstance(machine.payment, CashPayment):
            raise RuntimeError("this machine does not accept cash right now")
        machine.payment.add_funds(amount)
        machine.set_display(f"CREDIT {format_money(machine.payment.balance)}")
        machine.set_state(HasFundsState())

    def select(self, machine: VendingMachine, code: str) -> PurchaseResult:
        return PurchaseResult(VendStatus.INSUFFICIENT_FUNDS, detail="insert money first")

    def refund(self, machine: VendingMachine) -> dict[int, int]:
        return {}  # nothing to refund

    def restock(self, machine: VendingMachine, code: str, quantity: int) -> None:
        added = machine.inventory.add(code, quantity)
        machine.set_display(f"RESTOCKED {code} +{added}")


class HasFundsState(VendingMachineState):
    """Funds are in escrow. The customer may add more, select, or refund."""

    def insert_money(self, machine: VendingMachine, amount: int) -> None:
        machine.payment.add_funds(amount)
        machine.set_display(f"CREDIT {format_money(machine.payment.balance)}")

    def select(self, machine: VendingMachine, code: str) -> PurchaseResult:
        try:
            slot = machine.inventory.slot(code)
        except KeyError:
            return PurchaseResult(VendStatus.INVALID_SLOT, detail=f"no slot {code}")
        result = machine._vend(slot)
        if not result.ok:
            machine.set_display(result.status.value.upper().replace("_", " "))
            # A failed vend keeps the customer's funds in escrow for a retry.
        return result

    def refund(self, machine: VendingMachine) -> dict[int, int]:
        returned = machine.payment.collect()
        machine._reset_payment()
        machine.set_state(IdleState())
        machine.set_display("INSERT COIN")
        return returned

    def restock(self, machine: VendingMachine, code: str, quantity: int) -> None:
        raise RuntimeError("cannot restock during a transaction")


class DispensingState(VendingMachineState):
    """Transient. Only reached inside `_vend`; blocks all customer input."""

    def insert_money(self, machine: VendingMachine, amount: int) -> None:
        raise RuntimeError("please wait, dispensing")

    def select(self, machine: VendingMachine, code: str) -> PurchaseResult:
        raise RuntimeError("please wait, dispensing")

    def refund(self, machine: VendingMachine) -> dict[int, int]:
        raise RuntimeError("please wait, dispensing")

    def restock(self, machine: VendingMachine, code: str, quantity: int) -> None:
        raise RuntimeError("please wait, dispensing")


class OutOfServiceState(VendingMachineState):
    def insert_money(self, machine: VendingMachine, amount: int) -> None:
        raise RuntimeError("out of service")

    def select(self, machine: VendingMachine, code: str) -> PurchaseResult:
        return PurchaseResult(VendStatus.OUT_OF_SERVICE)

    def refund(self, machine: VendingMachine) -> dict[int, int]:
        return {}

    def restock(self, machine: VendingMachine, code: str, quantity: int) -> None:
        added = machine.inventory.add(code, quantity)
        machine.set_display(f"RESTOCKED {code} +{added}")


# --------------------------------------------------------------------------
# 9. Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    def fresh_inventory() -> Inventory:
        """A new Inventory per machine. Sharing one would be the exact
        shared-mutable-state bug the trade-offs section warns about."""
        return Inventory({
            "A1": Slot("A1", Product("SKU-COLA", "Cola", 65), quantity=10, capacity=10),
            "A2": Slot("A2", Product("SKU-CHIPS", "Chips", 50), quantity=2, capacity=10),
        })

    print("--- vend 65c with a 100c coin ---")
    machine = VendingMachine(fresh_inventory(), CashStore({5: 10, 10: 10, 25: 4, 100: 2}))
    machine.insert_money(100)
    result = machine.select("A1")
    print(result.status.value, "change:", {format_money(k): v for k, v in result.change.items()})
    print("bank:", {format_money(k): v for k, v in machine.bank.snapshot().items()})

    print("--- insufficient funds ---")
    machine2 = VendingMachine(fresh_inventory(), CashStore({5: 1, 10: 1}))
    machine2.insert_money(25)
    print(machine2.select("A1").status.value)

    print("--- cannot make change ---")
    machine3 = VendingMachine(fresh_inventory(), CashStore({100: 1}))  # no small coins
    machine3.insert_money(100)
    print(machine3.select("A1").status.value)  # needs 35c of change, has none
    print("refund returns:", {format_money(k): v for k, v in machine3.refund().items()})

    print("--- jam rollback ---")
    machine4 = VendingMachine(fresh_inventory(), CashStore({5: 10, 10: 10}), FlakyDispenser("A2"))
    machine4.insert_money(100)
    jam = machine4.select("A2")
    print(jam.status.value,
          "| stock intact:", machine4.inventory.slot("A2").quantity,
          "| bank restored:", {format_money(k): v for k, v in machine4.bank.snapshot().items()},
          "| credit returned:", {format_money(k): v for k, v in machine4.refund().items()})

    print("--- card payment ---")
    machine5 = VendingMachine(fresh_inventory(), CashStore())
    machine5.begin_card_payment(FakeTerminal())
    card_result = machine5.select("A1")
    print(card_result.status.value, "| change owed on a card:", card_result.change)

    print("--- state guard ---")
    machine5.set_state(DispensingState())
    try:
        machine5.insert_money(25)
    except RuntimeError as exc:
        print("refused mid-dispense:", exc)

    print("--- greedy would fail, the search does not ---")
    awkward = CashStore({25: 1, 10: 3, 5: 0})
    print("make 30c from {25:1, 10:3, 5:0}:", awkward.make_change(30))
```

**What to point at:**

- The **`_vend` ordering comment** — the change affordability check happens *before* the payment commits. Reverse it and you have charged a customer for an item you cannot make change for.
- The **`charge` before `collect`** ordering with its comment: `collect` empties the cash escrow, so calling it first makes the balance check fail. This is exactly the kind of ordering bug that a passing test suite misses and an audit catches.
- `CashStore.make_change` takes an `extra` pool, which is how the customer's coins participate. This is the detail that shows you have thought about a real machine.
- `_search` has a **suffix-max prune**. Without it, a pathological bank makes the search exponential; with it, four denominations are effectively instant.
- `_reset_payment` **refuses to discard a live credit**. A jam restores the customer's funds to escrow, and a naive "start the next customer clean" would destroy them — which is why the jam branch ends in `HasFundsState`, not `IdleState`.
- The jam path **rolls back three things**: the change withdrawn, the escrow deposit, and the stock (which was never decremented, because decrement happens only after a successful dispense). The demo prints the bank before and after so the conservation is visible.

---

## 6. Concurrency & thread-safety

### What breaks

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Two threads select the last item** | Both read `quantity == 1`, both pass the stock check, both dispense — one item, two sales | Decrement **inside** the lock, and only after a successful dispense; better, reserve first then release on failure |
| **Check-then-act on change** | Thread A checks the bank can make 35c; thread B's vend consumes those coins; A withdraws coins that are gone | `make_change` and `withdraw` must be one atomic decision — do the search against a *reserved* pool, or hold the mutex across both |
| **Insert during dispense** | A coin acceptor fires an interrupt while the machine is dispensing; the credit is counted twice or lost | The `DispensingState` refuses every input (already in the code), **and** the acceptor queue is drained only in `Idle`/`HasFunds` |
| **Refund racing a vend** | The refund returns coins while the vend is mid-charge | One re-entrant lock over the whole public API — which the `VendingMachine` facade already does |
| **`collect_cash` during a transaction** | The bank is emptied while change is owed | `collect_cash` refuses if the escrow is non-empty (already in the code) |
| **Observer called under the lock** | A telemetry observer that makes a network call stalls every vending operation | Notify outside the lock, or push to a queue |
| **Two machines sharing one `CashStore`** | A shared bank object means one machine's change is another's loss | A `CashStore` is owned by exactly one machine; construct it inside `VendingMachine` if a caller might pass a shared one |

### The reservation pattern, which is the correct fix for the first two hazards

```python
    # EXCERPT — a method of the VendingMachine class defined in §5, indented
    # because it is a class member. The complete runnable class is in §5 above.
    def _reserve_change(self, amount: int, extra: dict[int, int] | None) -> dict[int, int] | None:
        """Search AND remove in one step, so the change cannot be stolen
        between the decision and the withdrawal."""
        change = self.bank.make_change(amount, extra)
        if change is None:
            return None
        self.bank.withdraw(change)   # inside the same lock as the search
        return change
```

**The three sentences to say:**

1. *"The facade holds one re-entrant lock over the whole public API, because every operation is a read-modify-write on shared state — stock and coins."*
2. *"The two real races are double-dispense on the last item and change being stolen between the check and the withdrawal. Both are fixed by making the decision and the mutation one critical section."*
3. *"The `DispensingState` is also a concurrency device: it refuses input, so a coin interrupt during a dispense cannot corrupt the credit."*

**Do not build a distributed vending machine.** If the interviewer pushes toward "a fleet of machines with a central server", the answer is: the machine is authoritative for its own coins and stock (it is a physical device), the server is a reporting and pricing layer, and this is an HLD question — cross-link [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md).

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Add a card reader"

Already in the design: implement `CardPayment` against the `CardTerminal` port. Two things that differ and must be handled deliberately:

- `change_due(price)` returns **0** — a card charge is exact, so the change-making path is skipped entirely.
- The **decline** path is a new failure mode with a different rollback: no coins to return, but an authorization to void.

Because `PaymentMethod` was an interface from the start, `VendingMachine`, the states and `CashStore` are untouched. Say that: *"the payment strategy is the only thing that changed."*

### 7.2 "Support multiple payment methods in sequence" (part card, part cash)

Do not add a `MultiPayment` boolean. Compose:

```python
class SplitPayment(PaymentMethod):
    """Falls through a list of methods in order."""

    def __init__(self, methods: list[PaymentMethod]) -> None:
        self._methods = methods

    @property
    def balance(self) -> int:
        return sum(m.balance for m in self._methods)

    def sufficient(self, amount: int) -> bool:
        return self.balance >= amount

    def add_funds(self, amount: int) -> None:
        for method in self._methods:
            try:
                method.add_funds(amount)
                return
            except NotImplementedError:
                continue
        raise ValueError("no method accepts a top-up")

    def charge(self, amount: int) -> bool:
        remaining = amount
        charged: list[tuple[PaymentMethod, int]] = []
        for method in self._methods:
            take = min(method.balance, remaining)
            if take and method.charge(take):
                charged.append((method, take))
                remaining -= take
        if remaining:
            for method, take in reversed(charged):
                method.restore({})          # each method undoes its own charge
            return False
        return True

    def collect(self) -> dict[int, int]:
        collected: dict[int, int] = {}
        for method in self._methods:
            for value, qty in method.collect().items():
                collected[value] = collected.get(value, 0) + qty
        return collected

    def restore(self, funds: dict[int, int]) -> None:
        for method in self._methods:
            method.restore(funds)

    def change_due(self, price: int) -> int:
        return max(0, self.balance - price)
```

The design point: **compose strategies; do not add booleans to the facade.** Note the honest caveat — `restore(funds)` broadcasting the whole coin map to every sub-method is only correct when at most one sub-method holds a cash escrow, which is true for the "cash then card" ordering this exists for. A general split-across-two-cash-escrows design needs per-method attribution, and that is a real limitation, not a hidden one.

### 7.3 "Exact-change mode"

When the bank cannot make change for a denomination, real machines refuse that denomination rather than refusing the sale. Two designs:

```python
    # EXCERPT — a method of the VendingMachine class defined in §5, indented
    # because it is a class member. The complete runnable class is in §5 above.
    def accepts(self, amount: int) -> bool:
        """Refuse a denomination the machine cannot break, before it is inserted."""
        if not self.bank.snapshot():
            return True                      # empty bank: it keeps what it takes
        for slot in self.inventory:
            if slot.is_empty:
                continue
            owed = amount - slot.product.price
            if owed > 0 and not self.bank.can_make_change(owed, {amount: 1}):
                return False
        return True
```

That is the honest answer: the check is "can I break this note against *any* item I might sell?" — the alternative (refusing the sale after insertion) is a worse customer experience and a support ticket.

### 7.4 "Telemetry: low stock, low change, jams"

`MachineObserver` is already the seam. Push to a queue, not to the network:

```python
class RestockAlerter:
    def __init__(self, sink): self.sink = sink

    def on_low_stock(self, machine, slot):
        self.sink.emit({"event": "low_stock", "machine": id(machine),
                        "slot": slot.code, "quantity": slot.quantity})

    def on_low_change(self, machine, value):
        self.sink.emit({"event": "low_change", "denomination": value})

    def on_jam(self, machine, slot):
        self.sink.emit({"event": "jam", "slot": slot.code, "priority": "p1"})
```

### 7.5 "Dynamic / time-based pricing"

Prices become a function of context, not a field:

```python
class PricingStrategy(Protocol):
    def price_of(self, slot: Slot, at: float) -> int: ...


class HappyHourPricing:
    def __init__(self, base: PricingStrategy, discount_pct: int, window: tuple[int, int]) -> None:
        self.base, self.discount_pct, self.window = base, discount_pct, window

    def price_of(self, slot: Slot, at: float) -> int:
        hour = int(at) % 24
        price = self.base.price_of(slot, at)
        if self.window[0] <= hour < self.window[1]:
            return price * (100 - self.discount_pct) // 100
        return price
```

Note the integer division and the explicit rounding direction — a pricing strategy that produces fractional cents is a bug, not a rounding detail.

### 7.6 "Persistence and audit"

Every vend is an immutable `PurchaseResult`; the audit log is a list of them plus the bank deltas. `PurchaseResult` being frozen is what makes this free.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Float money** | `0.65 - 0.60 == 0.050000000000000044` — change is wrong | Integers in cents; formatting is presentation |
| 2 | **Greedy change with an exhausted denomination** | Fails to make change it could make | Bounded backtracking search over the actual bank |
| 3 | **Change check after dispensing** | The item is gone and the machine cannot pay the change | Check affordability **before** committing the payment |
| 4 | **Inserting coins mid-dispense** | Credit is double-counted | `DispensingState` refuses input |
| 5 | **Selecting while idle with no funds** | Dispenses on credit | `IdleState.select` returns `INSUFFICIENT_FUNDS` |
| 6 | **Selecting a sold-out slot** | Decrements to −1 | Stock check first, and `remove_one` asserts |
| 7 | **Refund mid-transaction** | Bank and escrow both change; money is created | Refund returns exactly the escrow and returns the state to `Idle` |
| 8 | **Vend fails after charging (jam)** | Customer is charged, item is not delivered | Roll back to the exact pre-vend state: re-deposit the change, withdraw the escrow, restore the credit, **stay in `HasFundsState`**, page an observer |
| 8b | **Jam rollback resets the payment object** | The restored credit is silently destroyed by "start the next customer clean" | `_reset_payment` refuses to replace a payment that still has a balance |
| 9 | **Change includes the customer's own coins** | A 100c note against a 65c item fails on an empty bank | The escrow joins the change pool |
| 10 | **Price changed mid-transaction** | The customer pays the new price for the decision they made at the old one | Read the price once, at select time, and use that value throughout `_vend` |
| 11 | **`collect_cash` while funds are in escrow** | The bank is emptied and the change cannot be paid | Refuse if `payment.balance > 0` |
| 12 | **Admin restock above capacity** | Quantity exceeds the physical slot | `Inventory.add` clamps to capacity and returns what was actually added |
| 13 | **Unknown denomination** | Silently accepted, breaking `total()` | `CashPayment.add_funds` validates against the accepted set |
| 14 | **Two threads on the last item** | Two sales from one item | Lock + decrement after the dispense; reserve-first if dispensing is slow |
| 15 | **Change requested as 0** | The search returns `None` for a valid case | `make_change(0)` returns `{}` explicitly |
| 16 | **Negative/zero price** | A free item, or a credit | `Product.__post_init__` rejects non-positive prices |
| 17 | **Card decline after a partial authorization** | The customer is charged for nothing | `charge` returns `False` and the caller makes no state change |
| 18 | **Out-of-service machine taking coins** | Accepts money it cannot refund | `OutOfServiceState.insert_money` raises |

**The two to volunteer:** #3 (check change before charging) and #9 (the escrow joins the change pool). They are the two that show you have modelled a *physical* machine rather than a vending-shaped calculator.

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **State** | `IdleState` / `HasFundsState` / `DispensingState` / `OutOfServiceState` | The *same call* means different things per phase: `insert_money` is legal in `Idle`, an increment in `HasFunds`, and an error in `Dispensing`. A boolean like `is_dispensing` cannot express "refund is illegal now" without a stack of guards | One class per phase and an indirection on every call; overkill below about three states |
| **Strategy** | `PaymentMethod` (cash/card/split) | The cash and card paths share no implementation — different escrow, different change, different failure and refund models | Only justified once the second implementation exists; introducing it for one method is speculative generality |
| **Strategy (again)** | `PricingStrategy` | Time-based and loyalty pricing are policies, not machine behaviour | Prices stop being inspectable fields |
| **Facade** | `VendingMachine` | Gives callers `insert_money` / `select` / `refund` and hides inventory, bank, payment and state coordination | Can degrade into a god object; keep the *rules* in the collaborators |
| **Observer** | `MachineObserver` | Restock alerts and jam paging are someone else's concern | Notification order is unspecified; a throwing observer must not break a vend |
| **Ports & Adapters** | `CardTerminal`, `Dispenser` | The hardware and the payment network are the two things that change independently of the rules, and the two things you cannot run in a test | An extra interface per seam |
| **Value Object** | `Product`, `PurchaseResult` (frozen dataclasses) | Hashable, safe to log, safe to hand across threads — and `PurchaseResult` is the audit record for free | Immutability means replacing rather than mutating, which is why `set_price` rebuilds the `Product` |
| **Singleton** | *Deliberately rejected* for the machine and for `CashStore` | `CashStore` must be per-machine, and a global machine would make tests order-dependent | — |

**The pattern answer to give:** *"State for the phases, Strategy for payment and pricing, Facade for the public API, Observer for telemetry, Ports for the hardware. I would explicitly **not** make the machine or the cash store a singleton — a cash store is per-machine state, and a global one is a test-ordering bug waiting to happen."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Money representation** | `int` cents everywhere, format at the edges | Exact arithmetic; no float drift; trivially serialisable | Every display site must remember to format |
| **Change algorithm** | Bounded backtracking with a suffix-max prune | Correct against a finite, lopsided bank | More code than greedy; needs a test for the pathological case |
| **Greedy change** | Not used | Ten lines, and correct when the bank is well stocked | Silently fails to give change it could give — a real revenue and support cost |
| **Change check placement** | Before the payment commits | The machine never charges for a sale it cannot complete | A second bank lookup per vend |
| **Escrow as part of the pool** | Yes | A machine with a coin-heavy bank but no small coins still works | The change search must include coins not yet banked |
| **Stock decrement timing** | After a successful dispense | A jam does not lose inventory | Briefly oversells; fix with a reservation if dispensing is slow |
| **Lock granularity** | One re-entrant lock on the whole facade | Every operation is atomic; the reasoning is trivial | Throughput is one vend at a time — fine, that is the physics |
| **State as objects vs an enum** | Objects | Transitions and per-phase legality live in one place; adding a phase is a class | More classes; an enum plus a switch is smaller for two states |
| **`PaymentMethod` as a Strategy up front** | Yes, with one cash implementation | The card follow-up is additive | Speculative until card exists — say that you would add it with the second method |
| **Observer notification** | Outside the lock (documented intent) | A slow telemetry sink cannot stall a vend | Ordering across observers is unspecified |
| **Hardware as ports** | `Dispenser`, `CardTerminal` protocols | The whole machine is testable with no hardware, including the jam path | One interface per device |

**The trade to name unprompted:** *"I am paying for a correct change algorithm with roughly forty lines, and the alternative — greedy change — is the single most common bug in candidate vending-machine implementations, because it is only correct when the bank happens to be well stocked."*

---

## 11. Q&A

**Q1. Why is the change algorithm not greedy?**
Because greedy optimises for "fewest coins" and the real constraint is "fewest coins **that I actually have**". Greedy walks denominations largest-first and commits to taking as many as it can; when a smaller denomination is exhausted, it has already spent the budget on a larger coin and cannot back out. A bounded backtracking search tries the larger coin counts from the maximum down to zero and recurses, so it finds the combination when one exists. With four denominations and a suffix-max prune, the search is effectively instant — the correctness costs nothing in practice.

**Q2. What is the suffix-max prune doing?**
It computes, for each suffix of the denomination list, the largest total those denominations could possibly contribute given the current bank. When the remaining amount exceeds that bound, the whole branch is abandoned without recursing. Without it, the recursion explores every combination down to the smallest coin and degrades toward exponential on a large bank; with it, the search is pruned to the reachable region.

**Q3. Why does `make_change` take an `extra` parameter?**
Because the customer's inserted coins are physically inside the machine and are available to hand straight back as change. A machine with a 100c coin in escrow and nothing in the bank must be able to sell a 65c item. Passing the escrow into the search models reality; ignoring it produces the false "cannot make change" that frustrates customers in the real world.

**Q4. Why does the change check happen before the payment commits?**
Because dispensing is irreversible from the machine's point of view. If you charge first and then discover you cannot make change, you have taken the customer's money for a sale you must now unwind — which means a refund path that returns coins you have already put in the bank. Checking first makes that situation impossible: the machine refuses the sale and the customer's funds stay in escrow for a retry or a refund. Stating the ordering explicitly is the point of the `_vend` comment.

**Q5. Where is the State pattern earning its keep?**
`insert_money` has three different meanings: in `Idle` it opens a transaction; in `HasFunds` it increments the credit; in `Dispensing` it must be refused, because a coin arriving mid-dispense would corrupt the accounting or be credited twice. `refund` is similar — legal in `HasFunds`, meaningless in `Idle`, illegal in `Dispensing`. Modelling that with booleans means three guards on every method and a combinatoric testing burden; modelling it with states means the refusal is a method that raises, and adding a phase is a new class.

**Q6. Why store money as integer cents?**
Because binary floating point cannot represent 0.10 exactly, so repeated arithmetic accumulates error and a change calculation can be off by a cent — which is not a rounding detail, it is a wrong answer that shows up in an audit. Integers in minor units are exact, comparable, hashable and cheap. Formatting happens only at the display boundary, in one function, so there is exactly one place where the representation becomes human-readable.

**Q7. What happens if the machine jams after the customer has paid?**
The vend rolls back to exactly the pre-vend state, which is four undo steps in a specific order: re-deposit the change that was withdrawn, withdraw the escrow that was deposited, restore the customer's credit, and leave the stock undecremented — which is automatic, because the decrement happens only after a successful dispense. The machine then lands in `HasFundsState`, not `Idle`, because the customer genuinely has a credit and must be able to retry or take it back; `_reset_payment` deliberately refuses to replace a payment object that still has a balance. The honest caveat: the customer's coins were physically accepted, so "returning funds" means either dispensing them from the bank or recording a credit — a real machine does the latter and settles on the next visit.

**Q8. Why is `DispensingState` a separate state at all, given that `_vend` is synchronous?**
Two reasons. First, the hardware is not synchronous in reality — a dispense is a motor command with a timeout, and the machine must refuse input for the duration. Second, it makes the illegal-input case explicitly representable and testable: you can construct the state and assert that `insert_money` raises, which is a one-line test for a class of concurrency bug that is otherwise very hard to reproduce.

**Q9. How do you make this thread-safe?**
One re-entrant lock on the facade covering the entire public API, because every operation is a read-modify-write on shared state — the stock count and the coin bank. The two races that matter are double-dispense on the last item and change being stolen between the affordability check and the withdrawal; both are fixed by making the decision and the mutation a single critical section. Observers are notified outside the lock. And I would say what I am *not* doing: a global lock across a fleet of machines, because each machine is a physically independent device.

**Q10. Why does `CardPayment.balance` return zero?**
Because a card is not a stored-value medium — there is nothing "in" it at the machine. The card path charges exactly the price in one authorization, so its `change_due` is always zero and the change-making machinery is skipped entirely. Modelling the card as a balance would force the vend path to handle "card change", which does not exist. The interface accommodates both because the *contract* is "what do I owe you afterwards", and for a card the answer is nothing.

**Q11 (senior). Your `CashStore` is per-machine. Why not make it a shared service with the central fleet server?**
Because the coins are physically in one machine. A central coin ledger cannot make change: the change must come out of the same chassis the customer is standing in front of. The central server's correct role is pricing, telemetry and restocking — the fleet is a *reporting* layer, and the machine is authoritative for its own cash and stock. Making the cash store shared would be a distributed-consistency problem with no benefit, and it would break the moment a machine went offline with a customer's money in escrow.

**Q12 (senior). The vend path charges, then dispenses. What if the process crashes between the two?**
The customer is charged and no item is dispensed. This is the classic two-phase problem and the honest answer is that it needs a **write-ahead record**: persist an intent record (transaction id, slot, amount, change) *before* the dispense, and on restart reconcile — if the record is uncommitted, refund; if committed, do not re-dispense. That is the same discipline as a payment provider's idempotency key and a database's transaction log, and the same shape as the ack-after-persist rule in [the WhatsApp design](../../Case-Studies/HLD/02-whatsapp-chat-system.md) §3.4. For an in-process LLD answer I would say the crash window exists, name the fix, and note that a single-chassis vending machine has a much smaller window than a distributed system.

**Q13. How would you test this without hardware?**
Three seams make it fully testable: `Dispenser` is a protocol, so I inject a `FlakyDispenser` that jams a chosen slot; `CardTerminal` is a protocol, so I inject `FakeTerminal` with an authorization limit; and `CashStore` takes an explicit count map, so I can construct a bank that cannot make change on purpose. Those three injections cover the interesting paths — jam, decline, and change failure — with no hardware and no network. That is the practical argument for the ports.

**Q14. What is the display for, and why does the machine own it?**
It is the only user-visible state, and it is a *rendering* of the state, not the state itself. The machine owns a display string and updates it in the state transitions because the display is a function of the phase: "INSERT COIN" when idle, the credit when funds are in, the failure reason after a refused vend. The design point is that no business rule reads the display — it is write-only from the machine's perspective, which is what keeps it from becoming a hidden state variable.

**Q15. Where would a bug most likely hide in this code?**
In the interaction between the escrow and the bank. The escrow is a `dict` of denominations inside `CashPayment`, the bank is a `dict` of counts inside `CashStore`, and the vend moves value between them in a specific order: deposit the escrow, then withdraw the change. A bug that deposits twice, or that withdraws before depositing, produces a bank that disagrees with the physical cash and is invisible until the machine is audited. That is why `collect_cash` refuses during a transaction and why the rollback path re-deposits the change explicitly — the invariant is *"bank + escrow is conserved across a vend."* I would write that invariant as an assertion in tests.

**Q16. How would you extend this to a coffee machine with recipes and consumables?**
Two new entity families and one new collaborator. A `Recipe` is a bill of materials — `{espresso: 18g, milk: 120ml}` — and a `ConsumableStore` mirrors `CashStore` for milk, beans and cups, with the same "can I afford this?" question. The vend path gains a step: check consumable affordability before dispensing, exactly as it checks change affordability, because running out of milk mid-pour is the jam of the coffee world. The `Product` gains a recipe reference, and the state machine is unchanged. That the state machine and the payment strategy survive untouched is the evidence the seams were drawn in the right place.

---

### Pushback round

**P1. "This is over-engineered for a vending machine. A single class with a handful of methods would do."**
For a machine that takes one coin, has four slots and gives no change, that is exactly right and I would say so. The design earns its classes at three points: change-making against a finite bank, a phase machine where the same call is legal and illegal depending on state, and a payment path that must be swappable for a card reader. If none of those three were in scope I would write one class and say the extra structure is not justified. I would rather name the threshold than defend structure that the requirements do not need.

**P2. "Your `_vend` is a 40-line method — that is a long method, and it lives on the facade, which you said holds no rules."**
That is a fair hit and the honest answer is that `_vend` is the one place where the transaction's *ordering* is the rule, and ordering is inherently sequential. I would improve it by extracting the rollback into a private `_rollback(escrow, change)` method and the affordability checks into `_can_vend(slot, price) -> VendStatus | None`, which drops `_vend` to about fifteen lines of straight-line orchestration. I would **not** split the ordering itself across objects, because the whole point is that the sequence is atomic and visible in one place — an event bus or a chain of handler objects would hide the ordering that is the design.

**P3. "You made `PaymentMethod` a Strategy but you only have one real implementation. That is speculative generality."**
Correct, and I said so in the trade-offs table. My defence is that card payment is not a hypothetical extension for a vending machine — it is the standard second requirement, and it is the one that changes the change-making path entirely (to nothing). Given a genuinely single-method world I would inline the cash logic into the machine and extract the interface at the moment the second method appears, which is about a fifteen-minute refactor. I would rather state the trigger condition than defend the abstraction on principle.

**P4. "Why not make `VendingMachine` a Singleton? There is physically one machine."**
Because "one" is a property of the deployment, not the design, and a singleton makes the class untestable in sequence — every test shares state, so test order matters and a failure in one test corrupts the next. I can get the same operational guarantee by constructing exactly one instance in `main` and injecting it, which is what the demo does. The physical-singleton argument also breaks immediately during development, when you want three machines in one process. Same reasoning as rejecting a singleton `RuleEngine` in [the chess design](09-chess-game.md) §9.

**P5. "Your rollback re-deposits the change and refunds the payment, but the customer's coins were physically accepted. The machine's cash no longer matches reality."**
That is the deepest objection in this problem and it is right. The physical truth is that the coins are in the hopper; the ledger now has to represent a debt. The correct model is that the rollback does **not** attempt to give the coins back through the bank — it records an outstanding credit and the machine either dispenses from the bank or the customer is refunded by staff. I would represent that with an explicit `CreditRecord` rather than pretending the bank is unchanged, and I would reconcile at the next `collect_cash`. The alternative — physically ejecting the coins — is a hardware operation the acceptor may not support mid-cycle. I would say that rather than claim the ledger is clean.

---

## 12. Cheat-sheet summary + cross-links

| Concept | One-line truth |
|---|---|
| Money | **Integer cents. Never float.** Format only at the display boundary |
| Change making | Bounded **backtracking search over the actual bank**, not greedy over denominations |
| The prune | Suffix-max bound; without it the search degrades toward exponential |
| The escrow rule | The customer's inserted coins **join the change pool** — they are physically in the machine |
| Vend ordering | Stock → funds → **change affordability** → commit → withdraw → dispense |
| Irreversible step | Dispensing is last; the stock decrement happens *after* it succeeds |
| Jam rollback | Refund funds, restore change to the bank, do not decrement stock, notify an observer |
| State pattern | `Idle` → `HasFunds` → `Dispensing`; `insert_money` is legal, an increment, and an error across them |
| Why not booleans | A flag cannot express "refund is illegal in this phase" without a guard per method |
| Payment | `PaymentMethod` Strategy; card's `change_due` is always **0** |
| Split payment | Compose strategies; never add a boolean to the facade |
| Exact-change mode | Refuse the *denomination* before insertion, not the sale after |
| Concurrency | One re-entrant lock on the facade; every op is a read-modify-write |
| The two races | Double-dispense on the last item; change stolen between check and withdrawal |
| Invariant | **bank + escrow is conserved across a vend** — assert it in tests |
| `collect_cash` | Refuses while funds are in escrow |
| Hardware seams | `Dispenser` and `CardTerminal` ports — jam and decline become testable |
| Persistence gap | A crash between charge and dispense needs a write-ahead intent record |
| Patterns | State, Strategy (payment + pricing), Facade, Observer, Ports & Adapters, Value Object |
| Rejected | Singleton for the machine or the cash store |

**The three sentences to memorise**

1. *"Money is integer cents, and change is a bounded search over the coins I actually have — not greedy over the denominations."*
2. *"Check that I can make change **before** I charge, because dispensing is irreversible."*
3. *"`insert_money` means three different things in three phases — that is the State pattern, and the `Dispensing` state is also the concurrency guard."*

### Cross-links

- **The LLD framing:** [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md)
- **Same family (state machines and finite resources):** [Chess Game](09-chess-game.md) — State + Command · [Snake and Ladder](15-snake-and-ladder.md) — turn-driven transitions · [Rate Limiter](11-rate-limiter.md) — the token budget as a finite resource
- **The distributed version of "finite resource with a shared ledger"** (and why the machine must stay authoritative): [Redis Deep Dive](../../HLD/03-scaling/03-redis-deep-dive.md) §3.9 · [Rate Limiter](11-rate-limiter.md)
- **Idempotency and the two-phase problem:** [URL Shortener LLD](../../Case-Studies/LLD/01-url-shortener-lld.md) · [Notification System LLD](../../Case-Studies/LLD/05-notification-system-lld.md)
- **Pattern catalogue and concurrency primitives:** [LLD Cheatsheet](../../Cheat-Sheets/07-lld-cheatsheet.md)
- **Question bank:** [LLD Questions](../../Interview-Questions/08-lld-questions.md) — Part E covers the vending machine, Part G is the pushback round
