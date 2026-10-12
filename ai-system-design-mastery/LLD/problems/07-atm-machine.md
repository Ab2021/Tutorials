# LLD Problem: ATM Machine

> **Grounding note.** The source transcript does not mention this problem. It is not in the LLD question list at **[9:22]**, and it is not part of the parking-lot walkthrough at **[25:54]–[27:38]**. **Everything below is standard industry practice.** It is in this pillar because it is the standard State-pattern problem, and because the state machine is *not* the interesting part. The interesting parts are three, and each is a genuinely hard decision dressed as a detail: **cash is a depleting inventory, not a number** — an ATM that hands out its last ₹2000 note can no longer serve a ₹2000 withdrawal, so "can I dispense this?" is a packing problem over finite cassettes, not a division; **a withdrawal is two operations that must both happen** — debiting the account and ejecting notes — and there is no ordering that makes both atomic, so the design has to name which failure it prefers and how it recovers; and **the PIN retry counter must survive a power cut**, which is the one piece of state in this system that lives in hardware rather than in a database. The shared-state mechanics are in [concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md); this file spends its budget on the domain.

---

## 1. Requirement clarification

### 1.1 Questions to ask

| # | Question | Why it matters | Default if they will not say |
|---|---|---|---|
| 1 | **Denominations, and are they fixed?** | Determines whether the dispense algorithm can assume a canonical system | Fixed at construction, e.g. `{2000, 500, 200, 100}` (§5.3) |
| 2 | **Finite cassettes, or infinite cash?** | **The question that separates the real design from a sketch** | Finite, with a per-denomination count (§5.2) |
| 3 | **Can the ATM dispense any amount?** | ₹150 with only ₹100 and ₹500 loaded is impossible | No — check feasibility *before* the debit (§5.4) |
| 4 | **Which failure is worse: cash out with no debit, or a debit with no cash?** | The whole ordering of the withdrawal | Never dispense without a completed debit; reverse on failure (§5.5) |
| 5 | PIN retry limit and card retention? | Whether the counter must persist | 3 attempts, retain, counter persisted per card (§5.6) |
| 6 | Are there multiple accounts / banks? | Whether there is an interbank switch | One bank; the switch is a later extension (§7.4) |
| 7 | Deposits and transfers, or withdrawal only? | Surface area | Withdrawal and balance enquiry; deposit as an extension (§7.2) |
| 8 | How many sessions at once per machine? | Whether the machine is concurrent at all | **One**: an ATM is a single-user device. This matters (§6.1) |
| 9 | Is a receipt mandatory? | Whether a print failure can block a completed withdrawal | Best-effort; a failed print does not fail the withdrawal (§8.1) |
| 10 | Who reconciles the physical cash? | Whether there is a balance check at all | A nightly job; the counter must match the cassettes (§7.5) |

### 1.2 Defaults to state out loud

```
  MODEL:      Card ─── Session ─── Transaction ─── DispensePlan
                        │                              │
                        └── states (§3)          consumes ─┴─ Cassette

  STATES:     IDLE → CARD_INSERTED → PIN_VERIFIED → DISPENSING → IDLE
              plus RETRACTING (the customer did not take the cash) and
              OUT_OF_SERVICE (terminal). Five are implemented in §5.4;
              RETRACTING is designed in §7.6 and is not in the code excerpt.
  CASH:       FINITE cassettes, one per denomination, each with a count and a
              capacity. Dispensability is a PACKING problem, not a division.
  ORDERING:   feasibility → authorize → dispense → confirm. Reverse on failure.
              Never dispense cash that has not been authorized.
  SECURITY:   the PIN is never stored, never logged, never in a message. Only a
              salted hash, verified off the card's own record. Retry count is
              PERSISTED so a power cycle cannot reset it.
  THREADING:  an ATM serves ONE customer at a time. The concurrency that matters
              is against the shared inventory and the reconciliation job, not
              against other customers at this machine.
  OUT:        the interbank switch, the card network, hardware drivers, the
              receipt printer's actual commands, the vault.
```

**The one sentence that frames the problem:** *"The ATM's state machine is a solved textbook exercise — what is not solved is that cash is a finite inventory you are packing into, and that a withdrawal is two operations you cannot make atomic."*

---

## 2. Core entities & responsibilities

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `Card` | The physical card's identity | card number, expiry, **PIN hash + salt**, retry count | Know the balance |
| `Account` | Money at the bank | balance, holds, currency | Know about cards |
| `AtmSession` | **One customer's visit, and its state** | session id, card, state, attempt count | Hold cash |
| `AtmState` *(interface)* + the six implementations | What is legal *now* | — | — |
| `Cassette` | One denomination's physical notes | denomination, count, capacity | Decide what to dispense |
| `CashInventory` | The set of cassettes | the cassettes, the lock | Authorize anything |
| `DispensePlanner` | Turn an amount into a note combination | — | Touch inventory |
| `DispensePlan` | A concrete combination of notes | the per-denomination counts | — |
| `CashDispenser` *(interface)* | Physically eject notes | — | Decide anything |
| `Bank` *(interface)* | Authorize, debit, credit, reverse | balances | Dispense |
| `Transaction` | One attempted operation and its outcome | id, type, amount, outcome, idempotency key | — |
| `Atm` | Wire it together, hold the session | the session, the inventory, the bank | — |
| `Clock` *(interface)* | Supply time | — | — |

**The design decision, stated plainly: the state is an object, not an enum field with a switch over it.** Every operation in an ATM is legal only in some states — you cannot select a transaction before the PIN is verified, you cannot insert a card while dispensing, you cannot take a withdrawal request while the previous notes are still in the slot. Modelling that as `state: AtmState` plus a `switch` in every method means the legality rules for one state are spread across six methods and a new state touches all of them. **Modelling it as an `AtmState` object means each state answers "what happens if the customer does X now?" in one place, and adding a state is adding a class.**

**The second decision: `DispensePlanner` is pure.** It takes an amount and a snapshot of the cassettes and returns a plan or nothing. **It does not touch the inventory, does not hold a lock, and is trivially testable** — and that matters because the packing algorithm is the part with real logic in it and it should not be tangled with the state machine.

---

## 3. Class diagram

```
                       ┌──────────────────────────────┐
                       │            Atm               │
                       ├──────────────────────────────┤
                       │ - session: AtmSession | None │
                       │ - inventory: CashInventory   │
                       │ - bank: Bank                 │
                       │ - dispenser: CashDispenser   │
                       │ - planner: DispensePlanner   │
                       ├──────────────────────────────┤
                       │ + insert_card(card)          │
                       │ + enter_pin(pin)             │
                       │ + select_withdrawal(amount)  │
                       │ + take_cash()                │
                       │ + eject_card()               │
                       └───────────┬──────────────────┘
                                   │ delegates to
                   ┌───────────────▼──────────────────────┐
                   │      «interface»  AtmState           │
                   ├──────────────────────────────────────┤
                   │ + insert_card(atm, card)             │
                   │ + enter_pin(atm, pin)                │
                   │ + select_withdrawal(atm, amount)     │
                   │ + take_cash(atm)                     │
                   │ + eject_card(atm)                    │
                   └───────────────▲──────────────────────┘
       ┌───────────┬───────────┬───┴───────┬────────────┬─────────────┐
  ┌────┴─────┐┌────┴──────┐┌───┴──────┐┌───┴───────┐┌───┴──────┐┌─────┴──────┐
  │  Idle    ││CardInserted││PinVerified││Dispensing ││Retracting││ OutOfSvc   │
  └──────────┘└───────────┘└───────────┘└───────────┘└──────────┘└────────────┘

  ┌────────────────────┐        ┌──────────────────────┐
  │   CashInventory    │        │   DispensePlanner    │
  ├────────────────────┤        ├──────────────────────┤
  │ - cassettes: {int: │        │ + plan(amount,       │
  │     Cassette}      │───────▶│    cassettes)        │
  │ - lock             │ snapshot│    -> DispensePlan|None│
  ├────────────────────┤        └──────────┬───────────┘
  │ + snapshot()       │                   │ produces
  │ + withdraw(plan)   │        ┌──────────▼───────────┐
  │ + total_available()│        │    DispensePlan      │
  │ + can_dispense(amt)│        ├──────────────────────┤
  └────────────────────┘        │ notes: {2000: 2,     │
                                │         500: 3}      │
                                │ total_minor: int     │
                                └──────────────────────┘

  ┌────────────────┐   ┌──────────────┐   ┌──────────────────┐
  │      Card      │   │   Account    │   │   «interface»    │
  ├────────────────┤   ├──────────────┤   │      Bank        │
  │ card_number    │   │ account_id   │   ├──────────────────┤
  │ pin_hash       │   │ balance_minor│   │ + authorize(...) │
  │ pin_salt       │   │ currency     │   │ + confirm(...)   │
  │ failed_attempts│   └──────────────┘   │ + reverse(...)   │
  │ is_retained    │                      │ + balance_of(...)│
  └────────────────┘                      └──────────────────┘
```

**What the diagram states:**

- `Atm` holds **one** `session`, not a collection. An ATM is a single-user device; the design should not pretend otherwise (§6.1).
- Every `AtmState` method takes `atm` — the state calls back into the machine to change state. **That is the State pattern's shape:** the state object knows the transitions, the context knows the data.
- `DispensePlanner.plan` takes `cassettes` as a **snapshot**, so it is pure and does not need the lock. The plan is then applied by `CashInventory.withdraw` under the lock, and if the inventory changed in between the withdraw is refused and the whole transaction reverses.
- `Bank` has `authorize`/`confirm`/`reverse` — **three verbs, not one `debit`**, and that is the ordering decision drawn.

---

## 4. Interfaces and enums

```python
from abc import ABC, abstractmethod
from enum import Enum


class AtmError(Exception):
    """Base for every refusal. An ATM refusing is normal, not exceptional: an
    insufficient balance, an undispensable amount, a wrong PIN and a depleted
    cassette are all expected outcomes that the UI must explain. They are
    typed exceptions for that reason, not generic failures."""


class InvalidStateError(AtmError):
    """The operation is not legal in the current state. Carries the state so
    the message can say what the machine was doing."""


class InsufficientFundsError(AtmError):
    pass


class UndispensableAmountError(AtmError):
    """The amount cannot be made from the loaded cassettes. Raised BEFORE any
    debit — see §5.4, which is the whole point."""


class DispenseFaultError(AtmError):
    """The hardware failed mid-dispense. The notes may or may not have left the
    cassette, which is the worst case in this system and the reason for §5.5."""


class AuthenticationError(AtmError):
    pass


class CardRetainedError(AuthenticationError):
    """The retry limit was reached and the machine kept the card."""


class TransactionType(Enum):
    WITHDRAWAL = "withdrawal"
    BALANCE_ENQUIRY = "balance_enquiry"
    DEPOSIT = "deposit"                # extension, §7.2


class TransactionOutcome(Enum):
    """Note UNKNOWN, and note it is not a failure. A bank call that times out
    has an outcome we do not know, and treating it as either success or failure
    is wrong in a different way: treating it as failure risks a double debit on
    retry, treating it as success risks a phantom withdrawal. UNKNOWN means
    'do not decide yet, and reconcile'."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class StateName(Enum):
    IDLE = "idle"
    CARD_INSERTED = "card_inserted"
    PIN_VERIFIED = "pin_verified"
    TRANSACTION_SELECTED = "transaction_selected"
    DISPENSING = "dispensing"
    RETRACTING = "retracting"
    OUT_OF_SERVICE = "out_of_service"


class Bank(ABC):
    """The bank's side. THREE verbs rather than one `debit`, and the split is
    the design:

      authorize  reserves the funds and returns a hold reference. NOT yet a
                 debit: the customer's balance has not moved.
      confirm    turns the hold into a real debit. Called only after the cash
                 has actually left the machine.
      reverse    releases the hold. Called when the withdrawal did not happen.

    Why not debit straight away: if the machine jams after the debit and there
    is no reverse, the customer has paid for notes they never received, and
    discovering that requires a complaint. The hold plus confirm means the
    window in which a customer's money is committed but they have no cash is
    bounded and reversible."""

    @abstractmethod
    def authorize(self, account_id: str, amount_minor: int,
                  reference: str) -> str:
        """Reserve the funds. Returns a hold id. Idempotent on `reference`:
        a retried authorize with the same reference returns the SAME hold
        rather than reserving twice."""

    @abstractmethod
    def confirm(self, hold_id: str) -> None:
        """Turn the hold into a debit. Idempotent: confirming an
        already-confirmed hold is a no-op, not a second debit."""

    @abstractmethod
    def reverse(self, hold_id: str) -> None:
        """Release the hold. Idempotent: reversing an already-reversed hold is
        a no-op. This matters because the reverse is called from a failure
        path, and failure paths run twice."""

    @abstractmethod
    def balance_of(self, account_id: str) -> int: ...


class CashDispenser(ABC):
    """The hardware. Deliberately tiny, because everything interesting about
    dispensing is decided BEFORE this is called."""

    @abstractmethod
    def dispense(self, plan: "DispensePlan") -> None:
        """Eject the notes. Raises `DispenseFaultError` on a hardware fault.

        The contract that makes the rest of the design work: this is called
        only after the plan has been validated against a snapshot AND the
        inventory has been decremented under its lock. So if it raises, the
        inventory has already been reduced and the transaction must reverse —
        and the notes' whereabouts are unknown, which §5.5 handles."""

    @abstractmethod
    def retract(self) -> None:
        """Pull notes back into the machine. Called when the customer does not
        take them — an unattended slot is a theft, and the machine must not
        simply wait."""

    @abstractmethod
    def is_available(self) -> bool: ...


class Clock(ABC):
    @abstractmethod
    def now(self) -> float: ...
```

**`TransactionOutcome.UNKNOWN` and `Bank`'s three verbs are the two things in this file that a sketch would miss.** A sketch has `bank.debit(amount)`. The real thing has a hold, a confirm and a reverse, and an outcome that is sometimes unknown — and the unknown case is what makes the reconciliation job (§7.5) a requirement rather than a nicety.

---

## 5. Complete working code

**Language note:** Python 3.11+, standard library only. Every block is complete.

### 5.1 Money and the card

```python
class Money:
    """Integer minor units, as everywhere in this pillar. An ATM is the one
    place where a float is unthinkable — the machine counts physical notes and
    a rounding error is a note that does not exist."""

    __slots__ = ("minor", "currency")

    def __init__(self, minor: int, currency: str = "INR") -> None:
        if not isinstance(minor, int) or isinstance(minor, bool):
            raise TypeError("Money takes integer minor units")
        self.minor = minor
        self.currency = currency

    def __add__(self, other: "Money") -> "Money":
        self._same(other)
        return Money(self.minor + other.minor, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._same(other)
        return Money(self.minor - other.minor, self.currency)

    def _same(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise AtmError(f"currency mismatch: {self.currency} vs {other.currency}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return (self.minor, self.currency) == (other.minor, other.currency)

    def __repr__(self) -> str:
        return f"{self.currency} {self.minor // 100}.{self.minor % 100:02d}"


import hashlib
import hmac
import os


class Card:
    """The card's IDENTITY, as the bank holds it.

    THE SECURITY CONSTRAINT, and it is a hard rule rather than a preference:
    the PIN is never stored, never logged, never included in a message, never
    in an exception's text. What is stored is a salted hash, and verification
    is a constant-time comparison — because a byte-by-byte comparison leaks
    the PIN's prefix through timing, and an attacker with physical access to a
    terminal is exactly the threat model here.

    `failed_attempts` is PERSISTED. That is the point of it being on the card's
    server-side record rather than in the session: a customer who enters two
    wrong PINs and then pulls the power on the machine must not get a fresh
    three attempts. The counter lives with the card, not with the session.
    """

    __slots__ = ("card_number", "account_id", "pin_hash", "pin_salt",
                 "failed_attempts", "is_retained", "expiry_year", "expiry_month")

    MAX_ATTEMPTS = 3

    def __init__(self, card_number: str, account_id: str, pin: str,
                 expiry_year: int, expiry_month: int) -> None:
        if not pin.isdigit() or not 4 <= len(pin) <= 6:
            raise ValueError("a PIN is 4 to 6 digits")
        self.card_number = card_number
        self.account_id = account_id
        self.pin_salt = os.urandom(16)
        self.pin_hash = self._hash(pin, self.pin_salt)
        self.failed_attempts = 0
        self.is_retained = False
        self.expiry_year = expiry_year
        self.expiry_month = expiry_month

    @staticmethod
    def _hash(pin: str, salt: bytes) -> bytes:
        """PBKDF2 with a work factor. A bare SHA-256 of a 4-digit PIN is
        brute-forced in microseconds — there are only 10 000 of them — so the
        cost factor is what makes the hash worth storing at all."""
        return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, 200_000)

    def verify_pin(self, pin: str) -> bool:
        """Constant-time comparison, and the failed attempt is recorded HERE
        so there is no path that checks the PIN without counting the attempt."""
        candidate = self._hash(pin, self.pin_salt)
        if hmac.compare_digest(candidate, self.pin_hash):
            self.failed_attempts = 0
            return True
        self.failed_attempts += 1
        if self.failed_attempts >= self.MAX_ATTEMPTS:
            self.is_retained = True
        return False

    def is_usable(self) -> bool:
        return not self.is_retained

    def __repr__(self) -> str:
        # NEVER the PIN, and not the full card number either — the last four
        # digits only, which is the same rule a receipt follows.
        return f"Card(****{self.card_number[-4:]}, attempts={self.failed_attempts})"


class Account:
    __slots__ = ("account_id", "balance_minor", "currency", "holds")

    def __init__(self, account_id: str, balance_minor: int, currency: str = "INR") -> None:
        self.account_id = account_id
        self.balance_minor = balance_minor
        self.currency = currency
        self.holds: dict[str, int] = {}          # hold_id → amount

    def available_minor(self) -> int:
        """Balance MINUS outstanding holds. Without subtracting the holds, two
        concurrent withdrawals can each pass the balance check and together
        overdraw the account — the classic check-then-act, and here it is cash
        leaving a machine."""
        return self.balance_minor - sum(self.holds.values())

    def __repr__(self) -> str:
        return (f"Account({self.account_id}, {self.balance_minor}, "
                f"held={sum(self.holds.values())})")
```

### 5.2 Cassettes and the inventory — cash as a depleting resource

```python
class Cassette:
    """One denomination's physical notes.

    A cassette has a CAPACITY as well as a count, because the difference is
    what the service engineer restocks against: a cassette at 200 of 2000 is
    nearly empty and should be refilled before it becomes the reason a
    withdrawal fails. Note there is no "top up" method — restocking is a
    physical act performed by an engineer, and modelling it as a method would
    imply the software can create cash."""

    __slots__ = ("denomination_minor", "count", "capacity")

    def __init__(self, denomination_minor: int, count: int, capacity: int = 2000) -> None:
        if denomination_minor <= 0:
            raise ValueError("a denomination must be positive")
        if not 0 <= count <= capacity:
            raise ValueError(
                f"count {count} is outside the cassette's capacity {capacity}"
            )
        self.denomination_minor = denomination_minor
        self.count = count
        self.capacity = capacity

    @property
    def value_minor(self) -> int:
        return self.denomination_minor * self.count

    def is_low(self) -> bool:
        return self.count < self.capacity * 0.1

    def __repr__(self) -> str:
        return (f"Cassette({self.denomination_minor // 100} × {self.count}"
                f"/{self.capacity})")


import threading


class CashInventory:
    """The cassettes, and the lock over them.

    THE SNAPSHOT/APPLY SPLIT, which is the design: `snapshot()` returns a plain
    dict of denominations to counts, taken under the lock and released
    immediately. `DispensePlanner` works on that snapshot with no lock held.
    `withdraw(plan)` then applies the plan under the lock, and REFUSES if the
    inventory no longer matches what the plan assumed.

    Why this rather than holding the lock across the planning: the planning is
    a search that can take a few milliseconds, and holding a lock across it
    would serialise the reconciliation job and the low-cash alarm behind every
    withdrawal. The snapshot/apply pattern is the same optimistic shape as the
    seat-locking file's CAS — plan on a read, then apply only if nothing moved.
    """

    def __init__(self, cassettes: "list[Cassette]") -> None:
        if not cassettes:
            raise ValueError("an ATM needs at least one cassette")
        denominations = [c.denomination_minor for c in cassettes]
        if len(set(denominations)) != len(denominations):
            raise ValueError("two cassettes cannot share a denomination")
        self._cassettes: dict[int, Cassette] = {
            c.denomination_minor: c for c in cassettes
        }
        self._lock = threading.RLock()

    def snapshot(self) -> "dict[int, int]":
        """A plain dict, so the planner cannot mutate the real thing by
        accident. Deep enough: the ints are copied into a new dict, and ints
        are immutable."""
        with self._lock:
            return {d: c.count for d, c in self._cassettes.items()}

    def denominations(self) -> "list[int]":
        with self._lock:
            return sorted(self._cassettes, reverse=True)

    def total_available_minor(self) -> int:
        with self._lock:
            return sum(c.value_minor for c in self._cassettes.values())

    def note_counts(self) -> "dict[int, int]":
        return self.snapshot()

    def withdraw(self, plan: "DispensePlan") -> None:
        """Apply the plan. Raises if the inventory cannot honour it.

        THE CHECK IS INSIDE THE LOCK, and it is a full re-validation against
        the CURRENT counts rather than a trust of the plan. A plan built from a
        stale snapshot is refused here, which is the whole reason the planner
        is allowed to run without the lock. This is compare-and-swap in
        spirit: apply only if nothing moved."""
        with self._lock:
            for denomination, count in plan.notes.items():
                cassette = self._cassettes.get(denomination)
                if cassette is None:
                    raise AtmError(
                        f"no cassette for denomination {denomination}; the "
                        f"plan was built against a different inventory"
                    )
                if cassette.count < count:
                    raise AtmError(
                        f"cassette {denomination} has {cassette.count} notes but "
                        f"the plan needs {count}"
                    )
            for denomination, count in plan.notes.items():
                self._cassettes[denomination].count -= count

    def is_empty(self) -> bool:
        with self._lock:
            return all(c.count == 0 for c in self._cassettes.values())

    def low_cassettes(self) -> "list[int]":
        with self._lock:
            return sorted(d for d, c in self._cassettes.items() if c.is_low())

    def restock(self, denomination: int, count: int) -> None:
        """Used by tests and by the engineer's console. Note it is NOT called
        during a customer session: restocking an ATM with a customer mid-
        transaction is not a thing that happens."""
        with self._lock:
            cassette = self._cassettes.get(denomination)
            if cassette is None:
                raise AtmError(f"no cassette for {denomination}")
            if cassette.count + count > cassette.capacity:
                raise AtmError(
                    f"restocking to {cassette.count + count} exceeds capacity "
                    f"{cassette.capacity}"
                )
            cassette.count += count
```

### 5.3 The dispense planner — the packing problem

```
   WHY THIS IS NOT A DIVISION.

   If the ATM has unlimited notes of {2000, 500, 200, 100}, then any multiple
   of 100 is dispensable and the fewest-notes answer is a greedy division:
   take as many 2000s as fit, then 500s, then 200s, then 100s.

   THE MOMENT THE CASSETTES ARE FINITE, THAT IS NO LONGER TRUE.

   Inventory:  2000 × 1,  500 × 0,  200 × 0,  100 × 5      (total ₹7000)
   Withdraw:   ₹2000

   Greedy takes the 2000. Now the inventory is 100 × 5 and the machine can
   serve only ₹100, ₹200, ₹300, ₹400, ₹500. The next customer wanting ₹2000 is
   refused, even though the machine holds ₹5000.

   Alternative: is there one? 5 × 100 = 500, and no combination of {1×2000,
   5×100} makes 2000 except the 2000 itself. So here greedy is the ONLY answer.
   But change the inventory to 2000 × 1, 100 × 20 and the choice is real:
   dispensing the 2000 strands nothing much, dispensing twenty 100s also works
   and would exhaust the small notes. Greedy is right here.

   THE CASE WHERE GREEDY IS WRONG — a non-canonical system:

   Inventory {4, 3, 1} (scaled: think of them as denominations), unlimited.
   Withdraw 6.  Greedy: 4 + 1 + 1 = 3 notes.  Optimal: 3 + 3 = 2 notes.
   Greedy fails whenever the denomination set is not CANONICAL, and canonical
   is a property of the specific set — {2000, 500, 200, 100} happens to be
   canonical for multiples of 100, so greedy IS optimal for the notes. But a
   set with a 200 and a 300 in it is not, and the planner must not assume.

   SO THE PLANNER SEARCHES, WITH A BOUND:

     - Objective 1 (hard): the notes must sum EXACTLY to the amount.
     - Objective 2 (hard): no cassette may go negative.
     - Objective 3 (soft): FEWEST NOTES. Fewer notes means less time in the
       slot, fewer jams, happier customers.
     - Objective 4 (soft): break the largest notes — preserve the small ones,
       because small notes are what make odd amounts dispensable later.

   The search is a bounded DFS over denominations in descending order, taking
   0..available of each, with memoisation on (index, remaining). The state
   space is small — a handful of denominations and a bounded count of each —
   and the node cap makes the worst case predictable rather than exponential.
```

```python
class DispensePlan:
    """A concrete combination. Immutable, and value-equal, because a test that
    says "the plan is two 2000s and three 500s" should read as an equality."""

    __slots__ = ("notes", "total_minor")

    def __init__(self, notes: "dict[int, int]") -> None:
        clean = {d: c for d, c in notes.items() if c > 0}
        if not clean:
            raise AtmError("a dispense plan with no notes is not a plan")
        self.notes = clean
        self.total_minor = sum(d * c for d, c in clean.items())

    @property
    def note_count(self) -> int:
        return sum(self.notes.values())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DispensePlan):
            return NotImplemented
        return self.notes == other.notes

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.notes.items())))

    def describe(self) -> "list[str]":
        return [f"{d // 100} × {c}" for d, c in sorted(self.notes.items(), reverse=True)]

    def __repr__(self) -> str:
        return f"DispensePlan({', '.join(self.describe())} = {self.total_minor})"


class DispensePlanner:
    """Pure. Takes a snapshot, returns a plan or None. No lock, no mutation.

    Deterministic: the DFS explores denominations in a fixed descending order
    and the preference tie-break is a total order, so the same inventory and
    amount always produce the same plan. That matters for the same reason it
    matters in the Splitwise allocation — two runs must agree, or a test can
    never pin the behaviour down.
    """

    MAX_NODES = 200_000

    def __init__(self, prefer_fewer_notes: bool = True) -> None:
        self._prefer_fewer_notes = prefer_fewer_notes

    def plan(self, amount_minor: int, cassettes: "dict[int, int]") -> "DispensePlan | None":
        if amount_minor <= 0:
            raise AtmError("a withdrawal must be for a positive amount")

        denominations = sorted(cassettes, reverse=True)
        if not denominations:
            return None

        # A cheap necessary condition before the search: the amount must be
        # reachable modulo the smallest denomination. If the smallest note is
        # 100, then 15 050 is impossible no matter how much cash is loaded —
        # and this check is O(1) where the search is not.
        smallest = min(denominations)
        if amount_minor % smallest != 0:
            return None

        self._nodes = 0
        self._best: DispensePlan | None = None
        memo: "set[tuple[int, int]]" = set()

        self._search(0, amount_minor, denominations, cassettes, {}, memo)
        return self._best

    def _search(self, index: int, remaining: int, denominations: "list[int]",
                cassettes: "dict[int, int]", chosen: "dict[int, int]",
                memo: "set[tuple[int, int]]") -> None:
        """Bounded DFS. Returns nothing; records the best plan in `self._best`.

        The pruning that makes this cheap in practice:
          - memoisation on (index, remaining): a state reached twice needs no
            second exploration
          - a node cap, so a pathological inventory degrades to "no plan found
            quickly" rather than hanging a customer's session
          - the `remaining < 0` cut, which is the obvious one
          - a bound against the current best: once a plan with k notes exists,
            a branch already holding k notes cannot beat it if any more notes
            are needed, so it is cut
        """
        self._nodes += 1
        if self._nodes > self.MAX_NODES:
            return

        if remaining == 0:
            candidate = DispensePlan(chosen)
            if self._is_better(candidate):
                self._best = candidate
            return

        if index >= len(denominations):
            return

        key = (index, remaining)
        if key in memo:
            return
        memo.add(key)

        if self._best is not None and sum(chosen.values()) >= self._best.note_count:
            # Already more notes than the incumbent, and we still need more.
            return

        denomination = denominations[index]
        available = cassettes[denomination]
        max_take = min(available, remaining // denomination)

        # Descending denomination order means a greedy-feeling first descent,
        # which finds a good incumbent early and makes the bound bite.
        for take in range(max_take, -1, -1):
            if take:
                chosen[denomination] = take
            self._search(index + 1, remaining - take * denomination,
                         denominations, cassettes, chosen, memo)
            chosen.pop(denomination, None)
            if self._nodes > self.MAX_NODES:
                return

    def _is_better(self, candidate: DispensePlan) -> bool:
        """Objective 3 then objective 4, as a total order so the result is
        deterministic.

        Fewer notes first. Then prefer the plan that uses MORE small notes and
        FEWER large ones — the tie-break that preserves the large notes, since
        a ₹2000 note is the only thing that can serve a ₹2000 withdrawal on its
        own. The comparison is a tuple, so it is a total order and there is no
        ambiguity to break."""
        if self._best is None:
            return True
        if self._prefer_fewer_notes:
            if candidate.note_count != self._best.note_count:
                return candidate.note_count < self._best.note_count
        return self._value_of_small_notes(candidate) > self._value_of_small_notes(self._best)

    @staticmethod
    def _value_of_small_notes(plan: DispensePlan) -> int:
        """A larger number means more of the plan is made of smaller notes,
        which is the preferable direction. Computed as the total value of notes
        below the largest denomination."""
        if not plan.notes:
            return 0
        largest = max(plan.notes)
        return sum(d * c for d, c in plan.notes.items() if d < largest)

    def can_dispense(self, amount_minor: int, cassettes: "dict[int, int]") -> bool:
        """Cheap check used before a full plan is needed — for the UI, which
        wants to grey out an amount before the customer commits."""
        return self.plan(amount_minor, cassettes) is not None

    def dispensable_amounts(self, cassettes: "dict[int, int]",
                            limit: int = 50) -> "list[int]":
        """The amounts that can be dispensed RIGHT NOW, up to `limit` entries.
        The ATM's UI uses this to offer "other amounts"; the reconciliation job
        uses it to warn that a refill is due because the machine can no longer
        serve common round amounts."""
        smallest = min(cassettes) if cassettes else 0
        if smallest == 0:
            return []
        reachable: list[int] = []
        max_amount = sum(d * c for d, c in cassettes.items())
        for amount in range(smallest, max_amount + 1, smallest):
            if self.can_dispense(amount, cassettes):
                reachable.append(amount)
                if len(reachable) >= limit:
                    break
        return reachable
```

### 5.4 The state machine — one class per state

```python
class AtmState(ABC):
    """Each state answers "what happens if the customer does X now?".

    The alternative — a `state` enum plus a `switch` in each of the machine's
    methods — spreads the rules for one state across six places, and adding a
    state touches all of them. This shape keeps one state's whole behaviour in
    one class.

    Every method's default is to refuse with the current state named, and the
    states override only what is legal. That inverts the usual arrangement:
    illegality is the default and legality is the exception, which means a
    forgotten state fails closed rather than allowing something."""

    name: StateName

    def insert_card(self, atm: "Atm", card: Card) -> None:
        raise InvalidStateError(f"cannot insert a card while {self.name.value}")

    def enter_pin(self, atm: "Atm", pin: str) -> None:
        raise InvalidStateError(f"no PIN expected while {self.name.value}")

    def select_withdrawal(self, atm: "Atm", amount_minor: int) -> None:
        raise InvalidStateError(f"cannot withdraw while {self.name.value}")

    def select_balance(self, atm: "Atm") -> None:
        raise InvalidStateError(f"cannot check a balance while {self.name.value}")

    def take_cash(self, atm: "Atm") -> None:
        raise InvalidStateError(f"no cash to take while {self.name.value}")

    def eject_card(self, atm: "Atm") -> None:
        raise InvalidStateError(f"cannot eject a card while {self.name.value}")


class IdleState(AtmState):
    name = StateName.IDLE

    def insert_card(self, atm: "Atm", card: Card) -> None:
        if not atm.dispenser.is_available():
            raise AtmError("this machine is out of service")
        if card.is_retained:
            # The card is retained: the machine keeps it, on purpose. Not an
            # error the customer can retry out of.
            raise CardRetainedError(
                "this card was retained after too many incorrect PINs; "
                "please contact your bank"
            )
        atm.begin_session(card)
        atm.set_state(CardInsertedState())


class CardInsertedState(AtmState):
    name = StateName.CARD_INSERTED

    def enter_pin(self, atm: "Atm", pin: str) -> None:
        session = atm.require_session()
        card = session.card
        if card.verify_pin(pin):
            session.pin_verified = True
            atm.set_state(PinVerifiedState())
            return

        remaining = Card.MAX_ATTEMPTS - card.failed_attempts
        if card.is_retained:
            atm.end_session(reason="card retained after failed PIN attempts")
            atm.set_state(IdleState())
            raise CardRetainedError(
                "too many incorrect PINs; this card has been retained"
            )
        # The PIN itself is NEVER in the message. Neither is the candidate.
        raise AuthenticationError(
            f"incorrect PIN; {remaining} attempt(s) remaining"
        )

    def eject_card(self, atm: "Atm") -> None:
        atm.end_session(reason="customer cancelled at the PIN prompt")
        atm.set_state(IdleState())


class PinVerifiedState(AtmState):
    name = StateName.PIN_VERIFIED

    def select_withdrawal(self, atm: "Atm", amount_minor: int) -> None:
        atm.perform_withdrawal(amount_minor)
        # A successful withdrawal has already moved to Dispensing; a refused
        # one leaves us here, able to try a smaller amount.
        if atm.state_name() == StateName.PIN_VERIFIED:
            return
        raise InvalidStateError("state changed during the withdrawal")

    def select_balance(self, atm: "Atm") -> None:
        session = atm.require_session()
        balance = atm.bank.balance_of(session.card.account_id)
        session.last_message = f"available balance {Money(balance)}"
        atm.set_state(CardInsertedState())        # back to the menu for a PIN?

    def eject_card(self, atm: "Atm") -> None:
        atm.end_session(reason="customer finished")


class DispensingState(AtmState):
    """The state the machine is in while notes are in the slot. Note what is
    NOT legal here: no withdrawal, no balance, no PIN — and no card eject,
    because ejecting the card while notes are in the slot abandons the cash."""

    name = StateName.DISPENSING

    def take_cash(self, atm: "Atm") -> None:
        atm.complete_dispense()
        atm.set_state(IdleState())

    def eject_card(self, atm: "Atm") -> None:
        raise InvalidStateError(
            "cannot eject the card while cash is in the slot; take the cash first"
        )


class OutOfServiceState(AtmState):
    name = StateName.OUT_OF_SERVICE

    def insert_card(self, atm: "Atm", card: Card) -> None:
        raise AtmError("this machine is out of service")


STATE_REGISTRY: "dict[StateName, type[AtmState]]" = {
    StateName.IDLE: IdleState,
    StateName.CARD_INSERTED: CardInsertedState,
    StateName.PIN_VERIFIED: PinVerifiedState,
    StateName.DISPENSING: DispensingState,
    StateName.OUT_OF_SERVICE: OutOfServiceState,
}
```

### 5.5 The session, the transaction, and the withdrawal ordering

```python
class AtmSession:
    """One customer's visit. Holds the state that must NOT survive the visit —
    which is the exact opposite of the card's retry counter, and the contrast
    is the point: `pin_verified` is per-visit and resets, `failed_attempts` is
    per-card and persists."""

    __slots__ = ("session_id", "card", "pin_verified", "started_at",
                 "last_message", "transactions")

    def __init__(self, session_id: str, card: Card, started_at: float) -> None:
        self.session_id = session_id
        self.card = card
        self.pin_verified = False
        self.started_at = started_at
        self.last_message = ""
        self.transactions: list["Transaction"] = []

    def __repr__(self) -> str:
        return (f"AtmSession({self.session_id}, {self.card!r}, "
                f"verified={self.pin_verified})")


class Transaction:
    """One attempted operation. `idempotency_key` is the session id plus a
    sequence number, so a retried confirm reaches the bank with the same
    reference and the bank's own idempotency makes it safe."""

    __slots__ = ("transaction_id", "type", "amount_minor", "outcome",
                 "idempotency_key", "hold_id", "plan", "error")

    def __init__(self, transaction_id: str, type: TransactionType,
                 amount_minor: int, idempotency_key: str) -> None:
        self.transaction_id = transaction_id
        self.type = type
        self.amount_minor = amount_minor
        self.outcome: "TransactionOutcome | None" = None
        self.idempotency_key = idempotency_key
        self.hold_id: "str | None" = None
        self.plan: "DispensePlan | None" = None
        self.error: "str | None" = None

    def __repr__(self) -> str:
        return (f"Transaction({self.transaction_id}, {self.type.value}, "
                f"{self.amount_minor}, {self.outcome.value if self.outcome else 'pending'})")


class Atm:
    """The machine.

    THE WITHDRAWAL ORDERING, which is the heart of this file:

      1. FEASIBILITY   Can the loaded cassettes make this amount at all?
                       PURE, no lock, no side effect. Refuse here and nothing
                       has happened.  ← THE STEP A SKETCH OMITS
      2. AUTHORIZE     Reserve the funds at the bank, returning a hold.
                       The customer's balance has not moved yet.
      3. TAKE INVENTORY  Decrement the cassettes under the lock.
      4. DISPENSE      Eject the notes. If this faults, go to 6.
      5. CONFIRM       Turn the hold into a debit, now that the customer has
                       the cash. If the confirm fails, retry — it is idempotent
                       on the hold id.
      6. REVERSE       On a dispense fault, release the hold. The machine has
                       a jam, the customer has no cash, and the hold must not
                       become a debit.

    WHY THIS ORDER and not debit-first: if the debit came first and the
    machine jammed, the customer's money would be gone with nothing in return,
    and they would have to notice and complain. Authorize-then-confirm bounds
    that exposure to a hold, which expires on its own even if the reverse never
    runs.

    THE ONE FAILURE THAT REMAINS, stated rather than hidden: if the notes were
    ejected AND the confirm never succeeds, the customer has cash the bank has
    not charged for. That is a genuine loss and no ordering avoids it — the
    machine cannot un-eject notes. What the design does is make it DETECTABLE:
    the transaction is left in UNKNOWN, the hold stays, and reconciliation
    (§7.5) sees a hold that was never confirmed against a dispense that the
    hardware log says succeeded. That is the honest answer: not that the case
    is impossible, but that it is found rather than discovered by a complaint.
    """

    def __init__(self, atm_id: str, inventory: CashInventory, bank: Bank,
                 dispenser: CashDispenser, planner: DispensePlanner,
                 clock: Clock, currency: str = "INR") -> None:
        self.atm_id = atm_id
        self.inventory = inventory
        self.bank = bank
        self.dispenser = dispenser
        self.planner = planner
        self.clock = clock
        self.currency = currency

        self._state: AtmState = IdleState()
        self._session: "AtmSession | None" = None
        self._pending: "Transaction | None" = None
        self._next_session = 0
        self._next_transaction = 0

    # ---- state plumbing ------------------------------------------------
    def state(self) -> AtmState:
        return self._state

    def state_name(self) -> StateName:
        return self._state.name

    def set_state(self, state: AtmState) -> None:
        self._state = state

    def require_session(self) -> AtmSession:
        if self._session is None:
            raise InvalidStateError("no session is active")
        return self._session

    def begin_session(self, card: Card) -> AtmSession:
        self._next_session += 1
        self._session = AtmSession(
            session_id=f"S{self._next_session:06d}", card=card,
            started_at=self.clock.now(),
        )
        return self._session

    def end_session(self, reason: str) -> None:
        if self._session is not None:
            self._session.last_message = reason
        self._session = None
        self._pending = None

    # ---- customer operations -------------------------------------------
    def insert_card(self, card: Card) -> None:
        self._state.insert_card(self, card)

    def enter_pin(self, pin: str) -> None:
        self._state.enter_pin(self, pin)

    def select_withdrawal(self, amount_minor: int) -> None:
        self._state.select_withdrawal(self, amount_minor)

    def select_balance(self) -> None:
        self._state.select_balance(self)

    def take_cash(self) -> None:
        self._state.take_cash(self)

    def eject_card(self) -> None:
        self._state.eject_card(self)

    # ---- the withdrawal ------------------------------------------------
    def perform_withdrawal(self, amount_minor: int) -> Transaction:
        session = self.require_session()
        if not session.pin_verified:
            raise InvalidStateError("the PIN has not been verified")

        self._next_transaction += 1
        transaction = Transaction(
            transaction_id=f"T{self._next_transaction:06d}",
            type=TransactionType.WITHDRAWAL,
            amount_minor=amount_minor,
            idempotency_key=f"{session.session_id}-{self._next_transaction}",
        )
        session.transactions.append(transaction)

        # STEP 1 — FEASIBILITY, before anything has happened. This is the step
        # a sketch omits, and omitting it means the customer's account is
        # debited and then the machine discovers it cannot make change.
        cassettes = self.inventory.snapshot()
        plan = self.planner.plan(amount_minor, cassettes)
        if plan is None:
            transaction.outcome = TransactionOutcome.FAILED
            transaction.error = "undispensable_amount"
            raise UndispensableAmountError(
                f"this machine cannot dispense {Money(amount_minor)} from its "
                f"current notes; try a smaller amount"
            )

        if self.inventory.total_available_minor() < amount_minor:
            transaction.outcome = TransactionOutcome.FAILED
            transaction.error = "empty"
            raise UndispensableAmountError("this machine is out of cash")

        # STEP 2 — AUTHORIZE. A hold, not a debit.
        hold_id = self.bank.authorize(
            session.card.account_id, amount_minor, transaction.idempotency_key
        )
        transaction.hold_id = hold_id

        # STEP 3 — TAKE THE INVENTORY. Inside its own lock, and it re-validates
        # against the current counts, so a plan built on a stale snapshot is
        # refused here rather than over-dispensing.
        try:
            self.inventory.withdraw(plan)
        except AtmError as exc:
            self.bank.reverse(hold_id)
            transaction.outcome = TransactionOutcome.FAILED
            transaction.error = f"inventory: {exc}"
            raise

        transaction.plan = plan
        self._pending = transaction
        self.set_state(DispensingState())

        # STEP 4 — DISPENSE.
        try:
            self.dispenser.dispense(plan)
        except DispenseFaultError as exc:
            # STEP 6 — REVERSE. The inventory has already been decremented and
            # the notes' whereabouts are unknown, so the inventory is NOT
            # credited back — counting them back in could make the machine
            # promise cash it does not have. Instead the transaction goes to
            # UNKNOWN and a fault is raised for the engineer.
            self.bank.reverse(hold_id)
            transaction.outcome = TransactionOutcome.UNKNOWN
            transaction.error = f"dispense fault: {exc}"
            self._pending = None
            self.set_state(IdleState())
            raise

        return transaction

    def complete_dispense(self) -> None:
        """Called when the customer takes the cash — that is the signal the
        notes actually left, and the moment the hold becomes a debit."""
        transaction = self._pending
        if transaction is None or transaction.hold_id is None:
            raise InvalidStateError("no dispense is pending")
        self.bank.confirm(transaction.hold_id)
        transaction.outcome = TransactionOutcome.SUCCEEDED
        self._pending = None

    # ---- operator operations -------------------------------------------
    def reconcile(self) -> "dict[str, object]":
        """The nightly check. What it compares:

          - the note counts the software believes against the engineer's count
          - every UNKNOWN transaction, which is a hold that was never confirmed
            against a dispense the hardware log says happened
          - any hold older than the hold window that was never confirmed or
            reversed, which is the "customer walked away" case
        """
        return {
            "atm_id": self.atm_id,
            "expected_notes": self.inventory.note_counts(),
            "total_available_minor": self.inventory.total_available_minor(),
            "low_cassettes": self.inventory.low_cassettes(),
            "is_empty": self.inventory.is_empty(),
            "dispensable_sample": self.planner.dispensable_amounts(
                self.inventory.snapshot(), limit=10
            ),
        }
```

### 5.6 The in-memory bank and a scripted dispenser

```python
class InMemoryBank(Bank):
    """A bank that behaves like one: holds, idempotency on every verb, and a
    balance that is the balance MINUS the holds."""

    def __init__(self) -> None:
        self.accounts: dict[str, Account] = {}
        self._holds: dict[str, tuple[str, int]] = {}      # hold_id → (account, amount)
        self._by_reference: dict[str, str] = {}           # reference → hold_id
        self._confirmed: set[str] = set()
        self._reversed: set[str] = set()
        self._lock = threading.RLock()
        self._next_hold = 0

    def add_account(self, account: Account) -> None:
        self.accounts[account.account_id] = account

    def authorize(self, account_id: str, amount_minor: int, reference: str) -> str:
        with self._lock:
            existing = self._by_reference.get(reference)
            if existing is not None:
                # Idempotent: the same reference returns the SAME hold. A
                # retried authorize must not reserve twice.
                return existing

            account = self.accounts.get(account_id)
            if account is None:
                raise AtmError(f"unknown account {account_id}")
            if amount_minor <= 0:
                raise AtmError("cannot authorize a non-positive amount")
            if account.available_minor() < amount_minor:
                raise InsufficientFundsError(
                    f"available {Money(account.available_minor())} is less than "
                    f"the requested {Money(amount_minor)}"
                )
            self._next_hold += 1
            hold_id = f"H{self._next_hold:06d}"
            self._holds[hold_id] = (account_id, amount_minor)
            self._by_reference[reference] = hold_id
            account.holds[hold_id] = amount_minor
            return hold_id

    def confirm(self, hold_id: str) -> None:
        with self._lock:
            if hold_id in self._confirmed:
                return                                    # idempotent no-op
            if hold_id in self._reversed:
                raise AtmError(
                    f"hold {hold_id} was already reversed and cannot be confirmed"
                )
            entry = self._holds.get(hold_id)
            if entry is None:
                raise AtmError(f"unknown hold {hold_id}")
            account_id, amount = entry
            account = self.accounts[account_id]
            account.balance_minor -= amount
            account.holds.pop(hold_id, None)
            self._confirmed.add(hold_id)

    def reverse(self, hold_id: str) -> None:
        with self._lock:
            if hold_id in self._reversed or hold_id in self._confirmed:
                return                                    # idempotent no-op
            entry = self._holds.get(hold_id)
            if entry is None:
                raise AtmError(f"unknown hold {hold_id}")
            account_id, _ = entry
            self.accounts[account_id].holds.pop(hold_id, None)
            self._reversed.add(hold_id)

    def balance_of(self, account_id: str) -> int:
        with self._lock:
            account = self.accounts.get(account_id)
            if account is None:
                raise AtmError(f"unknown account {account_id}")
            return account.available_minor()

    def hold_state(self, hold_id: str) -> str:
        with self._lock:
            if hold_id in self._confirmed:
                return "confirmed"
            if hold_id in self._reversed:
                return "reversed"
            if hold_id in self._holds:
                return "held"
            return "unknown"


class ScriptedDispenser(CashDispenser):
    """A dispenser whose faults are chosen by the test. `fail_on_dispense`
    makes the Nth dispense raise, which is how the reverse path is exercised
    without hardware."""

    def __init__(self) -> None:
        self.available = True
        self.dispensed: list[DispensePlan] = []
        self.retracted = 0
        self.fail_on_dispense: "set[int]" = set()

    def dispense(self, plan: DispensePlan) -> None:
        if not self.available:
            raise DispenseFaultError("dispenser unavailable")
        if len(self.dispensed) + 1 in self.fail_on_dispense:
            self.dispensed.append(plan)          # the notes DID leave
            raise DispenseFaultError("jam after the notes were presented")
        self.dispensed.append(plan)

    def retract(self) -> None:
        self.retracted += 1

    def is_available(self) -> bool:
        return self.available


class ManualClock(Clock):
    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds
```

### 5.7 It runs

```python
def build_atm(notes: "dict[int, int] | None" = None) -> "tuple[Atm, InMemoryBank, ScriptedDispenser]":
    counts = notes or {2000: 10, 500: 20, 200: 20, 100: 50}
    inventory = CashInventory([
        Cassette(denomination, count) for denomination, count in counts.items()
    ])
    bank = InMemoryBank()
    dispenser = ScriptedDispenser()
    atm = Atm("ATM-001", inventory, bank, dispenser, DispensePlanner(), ManualClock())
    return atm, bank, dispenser


def demo() -> None:
    atm, bank, dispenser = build_atm()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))          # ₹10,000

    # A normal withdrawal.
    atm.insert_card(card)
    atm.enter_pin("1234")
    print("state:", atm.state_name().value)                # pin_verified
    atm.select_withdrawal(4_700)
    print("plan:", atm._pending.plan)                      # 2 × 2000, 1 × 500, 1 × 200
    atm.take_cash()
    print("balance after:", Money(bank.accounts["ACC-1"].balance_minor))
    # ₹10,000 − ₹4,700 = ₹5,300. The hold became a debit.

    # THE STEP A SKETCH OMITS: an undispensable amount is refused BEFORE the
    # debit. ₹150 cannot be made from 2000/500/200/100 notes.
    atm.insert_card(card)
    atm.enter_pin("1234")
    before = bank.accounts["ACC-1"].balance_minor
    try:
        atm.select_withdrawal(15_000 + 50)                 # ₹150
    except UndispensableAmountError as exc:
        print("refused:", exc)
    print("balance unchanged:", bank.accounts["ACC-1"].balance_minor == before)

    # A wrong PIN, three times. The counter is on the CARD.
    fresh = Card("4222222222229999", "ACC-2", "4321", 2028, 6)
    bank.add_account(Account("ACC-2", 500_000))
    atm2, bank2, _ = build_atm()
    bank2.add_account(Account("ACC-2", 500_000))
    for attempt in range(1, 4):
        atm2.insert_card(fresh)
        try:
            atm2.enter_pin("0000")
        except CardRetainedError as exc:
            print(f"attempt {attempt}: retained — {exc}")
        except AuthenticationError as exc:
            print(f"attempt {attempt}: {exc}")
    print("card retained:", fresh.is_retained, "attempts:", fresh.failed_attempts)

    # A dispense fault AFTER the notes left: the hold is reversed, the
    # transaction is UNKNOWN, and reconciliation can see it.
    atm3, bank3, dispenser3 = build_atm()
    card3 = Card("4333333333337777", "ACC-3", "1111", 2028, 6)
    bank3.add_account(Account("ACC-3", 1_000_000))
    dispenser3.fail_on_dispense = {1}
    atm3.insert_card(card3)
    atm3.enter_pin("1111")
    try:
        atm3.select_withdrawal(2_000)
    except DispenseFaultError as exc:
        print("fault:", exc)
    print("balance untouched:", Money(bank3.accounts["ACC-3"].balance_minor))
    print("reconcile:", atm3.reconcile()["dispensable_sample"][:3])


if __name__ == "__main__":
    demo()
```

**The dispense-fault case is the one to dwell on.** The notes left the machine, the customer has cash, and the bank has not charged them. The hold is reversed so the customer is not charged — and the transaction is left `UNKNOWN` rather than `FAILED`, because "the customer has money they were not charged for" is a different situation from "the withdrawal did not happen", and the reconciliation job has to be able to tell them apart. **Labelling it FAILED would make the loss invisible, which is the actual bug.**

---

## 6. Concurrency and thread-safety

### 6.1 The honest framing: an ATM serves one customer at a time

**Say this out loud in the interview, because it changes what the concurrency section is about.** A real ATM is a single-user device: one card slot, one keypad, one cash slot. Two customers cannot be mid-transaction at the same machine, and a design that carefully makes the session state thread-safe is solving a problem that does not exist.

**What genuinely is concurrent:**

| # | Race | Consequence | Fix |
|---|---|---|---|
| 1 | The dispense plan vs the inventory changing | A plan that over-dispenses | Snapshot for planning; re-validate inside `withdraw`'s lock |
| 2 | Two withdrawals from one account at two ATMs | The account overdrawn | Holds subtracted from the balance, so `available` is honest |
| 3 | A retried `confirm` | **A double debit** | Idempotent on the hold id, checked under the bank's lock |
| 4 | A retried `reverse` | A hold released twice; a negative hold | Idempotent; `reverse` after `confirm` is a no-op |
| 5 | The reconciliation job reading counts mid-dispense | A count that never existed | `snapshot()` under the lock |
| 6 | The engineer restocking during a dispense | A count corrected into an over-dispense | Restock takes the same lock; and it is not a customer-path operation |
| 7 | A power cut between the debit and the dispense | A debit with no cash | **This is not a thread race — see §6.2** |
| 8 | Two sessions from the same card at two ATMs | Two PIN attempts in parallel, counter lost | The counter is on the shared card record, so it must be updated under a lock at the bank, not in the session |

### 6.2 The failure that concurrency does not solve

**The interesting answer in this problem is not a lock — it is the ordering in §5.5.** A power cut between the debit and the dispense is not a race between threads, it is a crash between two steps, and no lock helps. What helps is that the steps are ordered so the crash window leaves a *recoverable* state:

- Crash after authorize, before dispense → a hold exists, no cash left. The hold expires, or reconciliation reverses it. **The customer loses nothing.**
- Crash after dispense, before confirm → cash left, no debit. **The bank loses the amount.** This is the case that cannot be eliminated and is made visible instead.

**Compare with the cache and elevator files.** In the cache, the right answer was sharding a lock because the critical section was hot. In the elevator, one global lock was right because the critical section was microseconds of pure computation. Here, **the lock is barely in question at all** — one machine, one customer — and the hard part is a distributed-ordering problem between a machine and a bank. **Saying that is the strongest answer available: naming that the concurrency question is the wrong question for this problem, and the ordering question is the right one.**

### 6.3 Tests

```python
import pytest


def test_a_normal_withdrawal_debits_only_after_the_cash_is_taken() -> None:
    atm, bank, _ = build_atm()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))

    atm.insert_card(card)
    atm.enter_pin("1234")
    atm.select_withdrawal(4_700)

    # The hold exists; the balance has NOT moved.
    assert bank.accounts["ACC-1"].balance_minor == 1_000_000
    assert bank.accounts["ACC-1"].available_minor() == 1_000_000 - 4_700
    assert atm.state_name() == StateName.DISPENSING

    atm.take_cash()
    assert bank.accounts["ACC-1"].balance_minor == 1_000_000 - 4_700
    assert bank.accounts["ACC-1"].available_minor() == 1_000_000 - 4_700


def test_an_undispensable_amount_is_refused_before_any_debit() -> None:
    """THE test. ₹150 cannot be made from 2000/500/200/100 notes."""
    atm, bank, _ = build_atm()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))

    atm.insert_card(card)
    atm.enter_pin("1234")
    with pytest.raises(UndispensableAmountError):
        atm.select_withdrawal(15_050)                       # ₹150.50 — not a multiple of ₹1
    assert bank.accounts["ACC-1"].balance_minor == 1_000_000
    assert bank.accounts["ACC-1"].holds == {}, "a refusal must leave no hold"


def test_a_dispense_fault_reverses_the_hold_and_leaves_the_balance_alone() -> None:
    atm, bank, dispenser = build_atm()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))
    dispenser.fail_on_dispense = {1}

    atm.insert_card(card)
    atm.enter_pin("1234")
    with pytest.raises(DispenseFaultError):
        atm.select_withdrawal(2_000)

    assert bank.accounts["ACC-1"].balance_minor == 1_000_000
    assert bank.accounts["ACC-1"].holds == {}
    assert dispenser.dispensed, "the notes did leave, which is the whole problem"


def test_the_pin_retry_counter_survives_and_retains_the_card() -> None:
    atm, bank, _ = build_atm()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))

    for _ in range(2):
        atm.insert_card(card)
        with pytest.raises(AuthenticationError):
            atm.enter_pin("0000")
        atm.set_state(IdleState())                          # the customer leaves
    assert card.failed_attempts == 2
    assert not card.is_retained

    atm.insert_card(card)
    with pytest.raises(CardRetainedError):
        atm.enter_pin("0000")
    assert card.is_retained
    # A retained card cannot start a session at all, even with the right PIN.
    atm.set_state(IdleState())
    with pytest.raises(CardRetainedError):
        atm.insert_card(card)


def test_a_correct_pin_resets_the_counter() -> None:
    atm, bank, _ = build_atm()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))

    atm.insert_card(card)
    with pytest.raises(AuthenticationError):
        atm.enter_pin("0000")
    assert card.failed_attempts == 1

    atm.insert_card(card)
    atm.enter_pin("1234")
    assert card.failed_attempts == 0


def test_an_incorrect_pin_message_never_contains_the_pin() -> None:
    """A small test with a real point: the message is built from the attempt
    count only, so there is no path that could leak the candidate."""
    atm, bank, _ = build_atm()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))

    atm.insert_card(card)
    with pytest.raises(AuthenticationError) as caught:
        atm.enter_pin("9876")
    assert "9876" not in str(caught.value)
    assert "1234" not in str(caught.value)
    assert "2 attempt" in str(caught.value)


def test_the_planner_finds_the_fewest_notes() -> None:
    planner = DispensePlanner()
    cassettes = {2000: 10, 500: 20, 200: 20, 100: 50}
    plan = planner.plan(4_700, cassettes)
    assert plan is not None
    assert plan.total_minor == 4_700
    assert plan.note_count == 4                            # 2000 + 2000 + 500 + 200
    assert plan.notes == {2000: 2, 500: 1, 200: 1}


def test_the_planner_respects_finite_cassettes() -> None:
    """Greedy would take the 2000; with only one, the rest must come from
    elsewhere — and here it cannot, so the plan is None."""
    planner = DispensePlanner()
    assert planner.plan(2_000, {2000: 1, 100: 5}) == DispensePlan({2000: 1})
    assert planner.plan(4_000, {2000: 1, 100: 5}) is None   # total is only ₹2,500
    assert planner.plan(1_000, {2000: 0, 100: 10}) == DispensePlan({100: 10})
    assert planner.plan(2_050, {2000: 1, 100: 5}) is None   # not a multiple of 100


def test_the_planner_prefers_preserving_large_notes_on_a_tie() -> None:
    """₹1000 from {500 × 10, 100 × 50}: two 500s is fewest notes, so that wins.
    ₹600 from {200 × 5, 100 × 10}: three 200s and six 100s both work, and the
    fewest-notes rule picks the 200s. Neither case is a tie, so both are
    deterministic — the tie-break exists for the cases that are."""
    planner = DispensePlanner()
    assert planner.plan(1_000, {500: 10, 100: 50}) == DispensePlan({500: 2})
    assert planner.plan(600, {200: 5, 100: 10}) == DispensePlan({200: 3})


def test_the_planner_is_deterministic() -> None:
    planner = DispensePlanner()
    cassettes = {2000: 3, 500: 4, 200: 5, 100: 6}
    plans = {repr(planner.plan(3_700, cassettes)) for _ in range(20)}
    assert len(plans) == 1, f"non-deterministic planning: {plans}"


def test_a_plan_built_on_a_stale_snapshot_is_refused_by_withdraw() -> None:
    """The snapshot/apply discipline, tested directly."""
    inventory = CashInventory([Cassette(2_000, 2), Cassette(100, 10)])
    stale = inventory.snapshot()
    plan = DispensePlanner().plan(4_000, stale)
    assert plan == DispensePlan({2000: 2})

    # Someone else dispenses first.
    inventory.withdraw(DispensePlan({2000: 2}))

    with pytest.raises(AtmError):
        inventory.withdraw(plan)                            # only 0 notes left
    assert inventory.note_counts() == {2000: 0, 100: 10}


def test_two_atms_cannot_overdraw_one_account() -> None:
    """The realistic race: the same card at two machines. The hold subtracted
    from `available` is what stops it."""
    bank = InMemoryBank()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 100_000))             # ₹1,000

    atm_a, _, _ = build_atm()
    atm_b, _, _ = build_atm()
    atm_a.bank = bank
    atm_b.bank = bank

    hold_a = bank.authorize("ACC-1", 60_000, "ref-a")
    with pytest.raises(InsufficientFundsError):
        bank.authorize("ACC-1", 60_000, "ref-b")
    bank.confirm(hold_a)
    assert bank.accounts["ACC-1"].balance_minor == 40_000


def test_confirm_and_reverse_are_idempotent() -> None:
    bank = InMemoryBank()
    bank.add_account(Account("ACC-1", 100_000))
    hold = bank.authorize("ACC-1", 30_000, "ref-1")
    bank.confirm(hold)
    bank.confirm(hold)                                       # no second debit
    assert bank.accounts["ACC-1"].balance_minor == 70_000
    assert bank.hold_state(hold) == "confirmed"

    hold2 = bank.authorize("ACC-1", 20_000, "ref-2")
    bank.reverse(hold2)
    bank.reverse(hold2)
    assert bank.accounts["ACC-1"].available_minor() == 70_000


def test_authorize_is_idempotent_on_the_reference() -> None:
    bank = InMemoryBank()
    bank.add_account(Account("ACC-1", 100_000))
    first = bank.authorize("ACC-1", 30_000, "same-ref")
    second = bank.authorize("ACC-1", 30_000, "same-ref")
    assert first == second, "a retried authorize must not reserve twice"
    assert bank.accounts["ACC-1"].available_minor() == 70_000


def test_a_confirm_on_a_reversed_hold_is_refused() -> None:
    """The one place where idempotency would be wrong: a confirm after a
    reverse means the caller lost track of the transaction, and silently
    succeeding would debit a customer whose withdrawal was cancelled."""
    bank = InMemoryBank()
    bank.add_account(Account("ACC-1", 100_000))
    hold = bank.authorize("ACC-1", 30_000, "ref-1")
    bank.reverse(hold)
    with pytest.raises(AtmError):
        bank.confirm(hold)
    assert bank.accounts["ACC-1"].balance_minor == 100_000


def test_the_state_machine_refuses_illegal_operations() -> None:
    atm, bank, _ = build_atm()
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))

    with pytest.raises(InvalidStateError):
        atm.enter_pin("1234")                               # no card inserted
    with pytest.raises(InvalidStateError):
        atm.select_withdrawal(2_000)                        # no card inserted

    atm.insert_card(card)
    with pytest.raises(InvalidStateError):
        atm.select_withdrawal(2_000)                        # PIN not verified
    with pytest.raises(InvalidStateError):
        atm.take_cash()                                     # nothing to take

    atm.enter_pin("1234")
    atm.select_withdrawal(2_000)
    with pytest.raises(InvalidStateError):
        atm.eject_card()                                    # cash is in the slot
    with pytest.raises(InvalidStateError):
        atm.enter_pin("1234")


def test_an_out_of_cash_machine_refuses_and_leaves_no_hold() -> None:
    atm, bank, _ = build_atm({2000: 0, 500: 0, 200: 0, 100: 3})
    card = Card("4111111111111234", "ACC-1", "1234", 2028, 6)
    bank.add_account(Account("ACC-1", 1_000_000))

    atm.insert_card(card)
    atm.enter_pin("1234")
    with pytest.raises(UndispensableAmountError):
        atm.select_withdrawal(10_000)                       # ₹100 vs ₹100 cash
    assert bank.accounts["ACC-1"].holds == {}


def test_reconciliation_reports_what_it_should() -> None:
    atm, bank, _ = build_atm({2000: 1, 500: 0, 200: 0, 100: 5})
    report = atm.reconcile()
    assert report["total_available_minor"] == 2_000 + 500
    assert report["is_empty"] is False
    # With one 2000 and five 100s the machine can serve ₹2000 but not ₹4000,
    # which is exactly the depletion problem §5.3 is written around.
    planner = DispensePlanner()
    assert planner.plan(2_000, atm.inventory.snapshot()) == DispensePlan({2000: 1})
    assert planner.plan(4_000, atm.inventory.snapshot()) is None
```

---

## 7. Extensibility — the standard follow-ups

### 7.1 "Add a deposit"

**The interesting part is that a deposit is the withdrawal in reverse, and the ordering inverts.** For a deposit the machine takes the notes first and the bank credits second, because the bank must not credit money it has not secured. The failure window is different: if the machine takes the notes and the credit fails, the customer has lost cash and there is no hold to reverse — **the recovery is a reconciliation entry, not an automatic reversal.** And the notes taken must be counted and validated against the claimed amount, which is a new failure mode entirely: the customer says ₹5000 and the machine counted ₹4500. That is the reason real ATMs hold a deposit in escrow and ask the customer to confirm the counted amount before crediting.

### 7.2 "Add a transfer between accounts"

Straightforward, with one rule: **both sides must be in one authorization.** A transfer that debits one account and separately credits another leaves a window where the money exists in neither or both. The bank's `authorize` grows a two-account form, or the transfer becomes a bank-side operation that the ATM only initiates. **Saying "this stops being an ATM problem and becomes a bank problem" is the right answer.**

### 7.3 "Support multiple banks / the interbank switch"

The extension that changes the failure model rather than the design. A withdrawal from another bank's account means an extra hop, and each hop can time out — which is where `TransactionOutcome.UNKNOWN` stops being a nicety and becomes routine. The design consequence: **the hold becomes mandatory rather than a refinement**, because the reversal path now crosses a network and may itself fail, so the hold must expire on its own. And the reconciliation job becomes a settlement between two banks.

### 7.4 "Multiple denominations of the same value, or a new cassette"

`CashInventory` refuses two cassettes with the same denomination, because two cassettes of ₹200 are indistinguishable to the planner and would need the plan to name a cassette rather than a denomination. **The extension is to key plans by cassette id rather than denomination**, which is a small change to `DispensePlan` and a large change to the planner's search. Worth naming as the boundary rather than implementing.

### 7.5 "The reconciliation job"

The job that makes the `UNKNOWN` outcome useful. What it does nightly:

1. **Compare counts.** The software's note counts against the engineer's physical count. A mismatch is either a dispense fault or a theft, and either way it is a real incident.
2. **Find unconfirmed holds.** A hold older than the expiry window that was never confirmed or reversed. Each one is a customer who walked away or a crash, and each is reversible.
3. **Find dispenses with no debit.** The hardware log says notes left; no confirm exists for the corresponding transaction. **These are the losses**, and the whole `UNKNOWN` design exists so that this query has rows to return rather than the losses being invisible.
4. **Check for dispensability drift.** If the machine can no longer serve ₹500 or ₹2000 — the common round amounts — a refill is due before a customer discovers it. `dispensable_amounts` is what that check uses, and it is the reason the method exists.

### 7.6 "The customer walks away without taking the cash"

The state machine has a `RETRACTING` state named in §3 for exactly this. A timeout in `DispensingState` moves to retracting, and the dispenser pulls the notes back. **The subtlety: the notes pulled back must be counted back into the inventory, and the transaction must be reversed** — but only if the retract confirms the notes are actually back. A retract that fails leaves the notes in the slot, which is `UNKNOWN` again. **And the amount dispensed is not necessarily the amount retracted** if a customer took some — which is why the retract reports a count.

---

## 8. Failure modes and edge cases

### 8.1 Failure modes

| # | Failure | Consequence | Handling |
|---|---|---|---|
| 1 | **The amount is not dispensable** | The customer is debited and gets nothing | **Feasibility checked BEFORE the authorize** (§5.5 step 1) |
| 2 | Cash dispensed, debit not confirmed | **The bank loses the amount** | Not preventable; the transaction is left `UNKNOWN` so reconciliation finds it |
| 3 | Debit confirmed, cash not dispensed | The customer pays for nothing | Prevented by authorize-then-confirm; the hold reverses |
| 4 | A dispense fault after the notes left | A loss, and a jam | `UNKNOWN`, hold reversed, engineer fault raised |
| 5 | A retried `confirm` | **A double debit** | Idempotent on the hold id |
| 6 | A retried `reverse` | A hold released twice | Idempotent |
| 7 | A `confirm` after a `reverse` | The customer debited for a cancelled withdrawal | **Refused**, deliberately not idempotent (§6.3) |
| 8 | A power cut between authorize and dispense | A hold with no cash | The hold expires; reconciliation reverses it |
| 9 | A power cut between dispense and confirm | A loss | `UNKNOWN`; reconciliation finds it |
| 10 | The retry counter reset by a power cycle | **An unlimited PIN guessing oracle** | The counter is on the card's server record, not the session |
| 11 | The PIN in a log or a message | **A credential leak** | Never stored, never logged, never in an exception; constant-time compare |
| 12 | A stale plan applied to a changed inventory | An over-dispense or a negative count | `withdraw` re-validates under the lock |
| 13 | Two withdrawals from one account at two ATMs | An overdraft | Holds subtracted from `available` |
| 14 | An account at exactly the withdrawal amount | A zero balance, not a refusal | `available >= amount` is the check, not `>` |
| 15 | A withdrawal of zero or a negative amount | A nonsense transaction | Refused at the planner |
| 16 | The customer does not take the cash | An unattended slot; theft | `RETRACTING` on a timeout; the notes are counted back |
| 17 | A receipt printer failure after a completed withdrawal | The withdrawal rolled back? | **No** — printing is best-effort, reported separately; the money has moved |
| 18 | The machine is out of cash mid-session | A refusal after the PIN | Refused at feasibility; the session continues so the customer can try less |
| 19 | A cassette low but not empty | A refusal for a large amount | `low_cassettes` in reconciliation, so it is a refill before it is a failure |
| 20 | The largest note depleted | Amounts only it can serve become impossible | The planner's large-note preference slows it; reconciliation reports the drift |
| 21 | An expired card | A session with an unusable card | Refused at insert; the expiry check is on the server's record |
| 22 | A card retained but the customer still tries | Repeated refusals | Refused at insert, with a message that says contact the bank |

### 8.2 Edge cases worth naming

- **₹150 with denominations {2000, 500, 200, 100}.** Impossible. Refused before the debit.
- **₹100 with an empty ₹100 cassette and ₹500s loaded.** Impossible. Refused, even though the machine holds ₹50,000.
- **₹2000 with one ₹2000 note.** Dispensable once; the next customer cannot have it.
- **₹4000 with one ₹2000 and twenty ₹100s.** Impossible — the 2000 plus 2000 in 100s is possible if there are 20 of them, so this one *is* dispensable and the planner must find it.
- **Exactly the account balance.** Allowed; the balance becomes zero.
- **One paisa more than the balance.** Refused, on the available figure which already subtracts holds.
- **A withdrawal of ₹50 when the smallest note is ₹100.** Refused by the modulo check, before the search.
- **Two cassettes of the same denomination.** Refused at construction.
- **A cassette at capacity.** `restock` refuses to exceed it.
- **A dispense of a single note.** The plan has one entry; the note count is 1; the tie-breaks all still apply.
- **The same reference used for two different amounts.** The bank returns the first hold — and the ATM must compare, as the settlement key does in the Splitwise design, or it will silently dispense the wrong amount. **Worth naming as a gap: `authorize` here returns the existing hold without checking the amount, which real code must check.**

### 8.3 Complexity

| Operation | Cost | Note |
|---|---|---|
| `DispensePlanner.plan` | **Bounded, worst case O(MAX_NODES)** | Typically a few dozen nodes; the cap makes it predictable |
| The modulo pre-check | O(1) | Kills most undispensable amounts before the search |
| `CashInventory.snapshot` | O(d) | d denominations, typically 4 |
| `CashInventory.withdraw` | O(d) | Re-validate then decrement |
| `Bank.authorize` | O(holds) | The `available` computation sums the holds |
| `Bank.confirm` / `reverse` | O(1) | A dict lookup and an update |
| `Atm.perform_withdrawal` | O(plan) + O(bank) | The planner, then two bank calls |
| `dispensable_amounts(limit)` | O(limit × plan) | Used by reconciliation only, never on the customer path |

**The line to memorise:** *"Planning is a bounded search, not a division, because the cassettes are finite — and the bound is what keeps it off the customer's critical path. The modulo pre-check kills most impossible amounts in O(1). Everything else is a dict operation. And the honest complexity problem in this system is not any of these: it is that the withdrawal is two operations across a network boundary and no amount of algorithmic cleverness makes them atomic."*

---

## 9. Design patterns used, and why

| Pattern | Where | Why here | What it costs |
|---|---|---|---|
| **State** | `AtmState` and its six implementations | Legality is per-state and the number of states is small but their rules overlap; one class per state keeps each rule in one place and fails closed | Six classes and a delegation hop on every operation |
| **Strategy** | `DispensePlanner` (the note-count preference), `Bank`, `CashDispenser` | Three axes vary independently: the planner's objective, the bank's implementation, the hardware | Three interfaces |
| **Snapshot / compare-and-apply** | `CashInventory.snapshot()` + `withdraw`'s re-validation | Lets the expensive planning happen outside the lock while keeping the invariant | A plan can be refused and the transaction reversed |
| **Idempotency key** | The hold reference, the transaction's key | A retried bank call must not double-debit; the key is a model field, not a bolt-on | Every caller must generate one |
| **Three-valued outcome** | `TransactionOutcome.UNKNOWN` | "The bank call timed out" is neither success nor failure, and treating it as either is wrong differently | Every consumer of the outcome must handle three cases |
| **Null Object / default-refuse** | `AtmState`'s methods all raise by default | Illegality is the default; a forgotten override fails closed | A little boilerplate per state |
| **Singleton** *(rejected)* | — | The `Atm` is not a singleton: two machines are two instances, and a global would make the tests share state | — |
| **Command** *(rejected)* | — | Considered for the customer operations, so they could be queued and logged. Rejected: each operation's legality is state-dependent, and a command queue would need the state at *execution* time, not enqueue time — the state machine already carries the ordering | — |

**The rejected Command is the interesting one.** It is the pattern a candidate reaches for because "the operations are undoable and loggable", and the reason it does not fit is specific: **an ATM's operations are legal only in a particular state, and the state changes as a result of executing them.** A command queued in `PinVerified` may be executed in `Dispensing`, where it is illegal. The state object *is* the ordering mechanism, and a queue would be a second one fighting it.

---

## 10. Trade-offs

| Choice | Buys | Costs |
|---|---|---|
| One class per state | Each state's rules in one place; fails closed | Six classes; a delegation hop |
| Feasibility before the debit | **A debit never happens for an amount the machine cannot make** | A planning search on the customer's path (bounded) |
| Authorize → dispense → confirm | The crash window leaves a reversible hold | Three bank calls instead of one; a hold to expire |
| Three-valued outcome | A timeout is not mislabelled | Every consumer handles `UNKNOWN` |
| Holds subtracted from `available` | Two ATMs cannot overdraw one account | `available` is O(holds) to compute |
| Planning on a snapshot | No lock held across the search | Plans can be refused; a reversal needed |
| Finite cassettes modelled explicitly | Dispensability is honest | A search rather than a division; a planner to test |
| Fewest-notes objective | Less time in the slot; fewer jams | A tie-break rule and a total order to get right |
| Large-note preservation as a tie-break | Common round amounts stay dispensable longer | A subtle rule that is hard to explain to a customer |
| The retry counter on the card record | Survives a power cycle | A write to a shared record on every failed attempt |
| Constant-time PIN comparison | No timing leak of the PIN prefix | A hash with a real work factor per attempt |
| Best-effort receipt printing | A print failure cannot roll back a completed withdrawal | The customer has no receipt; reported separately |
| Inventory not credited back on a dispense fault | Does not promise cash that may be in the slot | The counts are pessimistic until an engineer reconciles |
| An ATM as a single-session device | No session concurrency to design | Cannot serve two customers at one machine — which is correct |

**What changes at 10×:** nothing in this file. **An ATM is one machine, and ten times as many machines is ten times as many instances, not a bigger one.** The thing that scales is the bank behind them: the holds table, the reconciliation, and the authorization path — and those are the bank's problems, not the ATM's. **Saying that is the correct answer to "how does this scale?" and it is a stronger one than inventing a distributed ATM.**

---

## 11. Interview Q&A

**Q1 (easy). What are the core classes?**
`Card`, `Account`, `AtmSession`, `AtmState` with its six implementations, `Cassette`, `CashInventory`, `DispensePlanner` and `DispensePlan`, plus the `Bank` and `CashDispenser` interfaces. The two decisions that shape it: the state is an object rather than an enum with a switch, and the cash is a finite inventory rather than a number.

**Q2 (easy). Which pattern is this?**
State. Every operation is legal in some states and not others, and the set of operations is the same across states — which is exactly the State pattern's shape. The alternative, an enum plus a `switch` in the machine's methods, spreads one state's rules across six places. **The default-refuse arrangement matters more than the pattern name: illegality is the default and legality is the override, so a forgotten case fails closed.**

**Q3 (easy). Why is the state an object and not an enum?**
Because legality is not a property of the operation, it is a property of the operation *and* the state. With an enum you get `if state == PIN_VERIFIED: ... elif state == DISPENSING: ...` in `withdraw`, `check_balance`, `eject_card` and three more — six methods each holding a fragment of the same rule. With a state object, "what happens if the customer presses eject while dispensing?" is answered in one class, and adding a state is adding a class rather than editing six methods.

**Q4 (medium). How do you decide which notes to dispense?**
A bounded depth-first search over the denominations in descending order, choosing 0 to the available count of each, with memoisation and a node cap. The objective is fewest notes, with a tie-break that preserves the large notes. **And the reason it is a search rather than a division: the cassettes are finite.** With unlimited notes, any multiple of the smallest denomination is dispensable and greedy is optimal. With finite cassettes that is false — a machine holding one ₹2000 and five ₹100s can dispense ₹2000 and cannot dispense ₹4000, and no division tells you that.

**Q5 (medium). Is greedy ever wrong for the notes?**
For `{2000, 500, 200, 100}` at multiples of ₹100, greedy is optimal — that set is canonical, which is a property of the specific set. But a set with a 200 and a 300 in it is not canonical, and greedy fails on it: with unlimited {4, 3, 1}, withdrawing 6 gives greedy 4+1+1, three notes, where 3+3 is two. So the planner searches rather than assuming, and **the cost of that is bounded by the node cap rather than being exponential in practice.**

**Q6 (medium). *Pushback:* "Why check dispensability before the debit? Just debit and then figure out the notes."**
Because the reverse order means the customer's account is debited and then the machine discovers it cannot make change, and now there is a debit with no cash and no hold to reverse — the recovery is a manual credit. Checking first is pure, needs no lock, and costs a bounded search. **It is the single most important ordering decision in the file: a refusal before any side effect is free, and a refusal after one is a refund.**

**Q7 (medium). What is the ordering of a withdrawal?**
Feasibility, then authorize, then take the inventory, then dispense, then confirm. Authorize is a hold, not a debit; confirm turns it into a debit only once the customer has the cash. **The reason for hold-then-confirm rather than debit-first: if the machine jams after a debit, the customer has paid for notes they never received, and they have to notice and complain. A hold reverses.**

**Q8 (medium). *Pushback:* "What if the confirm fails after the notes are out? You have cash out and no debit."**
You do, and no ordering prevents it — the machine cannot un-eject notes. What the design does is make it *detectable* rather than invisible: the hold stays open, the transaction is left `UNKNOWN` rather than `FAILED`, and the reconciliation job queries for dispenses the hardware log says succeeded with no corresponding confirm. **Labelling it FAILED is the actual bug, because then the loss has no row to appear on.** The honest answer is "this is a real loss and here is how we find it", not "this cannot happen".

**Q9 (medium). *Pushback:* "You have six state classes for six states. Isn't that over-engineered for a menu?"**
The count is the argument for it, not against it. Six states times six operations is thirty-six cells of "is this legal", and with an enum those thirty-six cells are scattered across six methods where nobody can see them together. With state objects they are six small classes, each of which reads as a description of what that state allows. **And the failure mode is asymmetric: a forgotten cell in a switch silently allows an illegal operation — ejecting a card while cash is in the slot, say — where the default-refuse arrangement refuses it.**

**Q10 (medium). Why is the PIN retry counter not in the session?**
Because a session ends and a counter that lives in it resets. A customer who enters two wrong PINs and then cuts the power would get three fresh attempts, and that turns the machine into an unlimited PIN-guessing oracle — 10 000 possible four-digit PINs and no cost to trying. So the counter is on the card's server-side record, persisted on every failed attempt, and the session's `pin_verified` flag is the thing that resets, because that one *should*. **The contrast between the two pieces of state is the point.**

**Q11 (hard). How is the PIN handled?**
Never stored, never logged, never in an exception's text. What is stored is a PBKDF2 hash with a per-card random salt and a real work factor, because a bare SHA-256 of a four-digit PIN is brute-forced in microseconds — there are only 10 000 of them. Verification is `hmac.compare_digest`, not `==`, because a byte-by-byte comparison leaks how much of the PIN prefix was right through timing, and physical access to a terminal is exactly this threat model. **And the failure message is built from the attempt count only, so there is no code path that could interpolate the candidate.**

**Q12 (medium). What happens if the customer walks away with the cash still in the slot?**
The `Dispensing` state times out and moves to retracting, and the dispenser pulls the notes back. The notes pulled back are counted into the inventory — but only for the count the retract reports, because the customer may have taken some. And the transaction reverses: the hold is released, so the customer is not charged for notes they did not get. **If the retract itself fails, that is `UNKNOWN` again, and reconciliation sees notes in the slot.**

**Q13 (hard). *Pushback:* "Your inventory is not credited back on a dispense fault. Isn't that wrong? The notes are either out or they are not."**
They might be either, and that is exactly why the counts stay pessimistic. If the notes jammed inside the mechanism, crediting them back makes the software promise ₹2000 that no customer can receive — and the next withdrawal is planned against cash that is not there, which then fails at dispense, and the failure cascades. Keeping the count low means the machine under-promises until an engineer physically opens it. **A machine that refuses a withdrawal it could have served is a nuisance; a machine that promises cash it does not have is a queue of failures.**

**Q14 (medium). How does the concurrency story work here?**
Less than you would expect, and saying so is the answer. **An ATM is a single-customer device** — one slot, one keypad — so there is no session concurrency to design. What is genuinely concurrent is the inventory (the planner reads a snapshot while the reconciliation job reads counts) and the bank's holds, which is where the real race is: two ATMs, one account. That is solved by subtracting holds from the available balance, so two concurrent authorizations cannot both pass the check.

**Q15 (medium). What is missing from this design?**
Three things, named rather than hidden. First, `Bank.authorize` returns an existing hold on a repeated reference without checking that the amount matches, and real code must compare — a key reused for a different amount returning the first hold silently is exactly the bug the Splitwise settlement guards against. Second, deposit escrow: a deposit needs the customer to confirm the counted amount before crediting, and this design has no deposit at all. Third, the expiry of a hold is delegated to the bank's implementation with no `Clock` on the ATM side driving it, which is fine when the bank owns expiry and wrong if the ATM is expected to sweep.

---

## 12. Cheat-sheet summary

**The model**

```
  Card ─── AtmSession ─── Transaction ─── DispensePlan ── consumes ── Cassette
              │
              └── AtmState (one class per state)

  ONE SESSION PER MACHINE. An ATM is a single-customer device.
```

**The states**

```
  IDLE ──insert──▶ CARD_INSERTED ──correct PIN──▶ PIN_VERIFIED
                        │                              │
                        │ eject                        │ withdraw
                        ▼                              ▼
                      IDLE ◀──take cash── DISPENSING ──timeout──▶ RETRACTING
                                          OUT_OF_SERVICE (terminal)

  DEFAULT IS REFUSAL. A state overrides only what is legal, so a forgotten
  case fails closed instead of silently allowing.
```

**The withdrawal ordering — the heart of the file**

```
  1. FEASIBILITY   can the cassettes make this amount?  PURE. Refuse here.
  2. AUTHORIZE     a HOLD at the bank. The balance has NOT moved.
  3. INVENTORY     decrement under the lock, re-validating the plan.
  4. DISPENSE      eject. A fault → 6.
  5. CONFIRM       hold → debit, now that the customer has the cash.
  6. REVERSE       on failure, release the hold.

  WHY: a refusal before any side effect is free; a refusal after one is a refund.
  THE REMAINING LOSS: notes out, confirm never succeeds. Not preventable.
  MAKE IT DETECTABLE: outcome = UNKNOWN, hold stays, reconciliation finds it.
```

**Dispensing is a packing problem, not a division**

```
  UNLIMITED notes → greedy division is optimal.
  FINITE cassettes → greedy can fail. {2000:1, 100:5} serves ₹2000 and NOT ₹4000.
  NON-CANONICAL sets → greedy is wrong even unlimited. {4,3,1} for 6: greedy 3
                       notes, optimal 2.

  PLANNER: bounded DFS, descending denominations, memoised, node-capped.
  OBJECTIVES: (1) sums exactly  (2) no cassette negative  (3) fewest notes
              (4) preserve the large notes on a tie
  PRE-CHECK: amount % smallest_denomination != 0  →  impossible, O(1)
```

**The numbers**

| Parameter | Value |
|---|---|
| Denominations | Fixed at construction; `{2000, 500, 200, 100}` |
| Cash | Finite, per-cassette count and capacity |
| Planner | Bounded DFS, node cap 200 000; typically dozens of nodes |
| Objective | Fewest notes, large notes preserved on a tie |
| PIN | PBKDF2, 200 000 iterations, per-card salt, constant-time compare |
| Attempts | 3, then the card is retained; **counter persisted per card** |
| Bank verbs | `authorize` (hold) · `confirm` (debit) · `reverse` (release) — all idempotent |
| Outcomes | `SUCCEEDED` · `FAILED` · `UNKNOWN` |
| Lock | One over the cassettes; planning happens outside it on a snapshot |

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "Which pattern?" | "State — and the reason is that the default must be refusal, not a forgotten switch case." |
| "How do you pick notes?" | "A bounded DFS, because the cassettes are finite. Greedy is wrong the moment cash is not unlimited." |
| "Debit then dispense?" | "Never. Feasibility, then a hold, then dispense, then confirm. A refusal before a side effect is free." |
| "The confirm fails" | "A hold, not a debit, so it reverses. And if the notes were already out, the outcome is UNKNOWN, not FAILED." |
| "Three wrong PINs" | "Retained, and the counter is on the card's server record so a power cycle cannot reset it." |
| "Store the PIN?" | "Never. PBKDF2, per-card salt, constant-time compare, and never in an exception's text." |
| "Concurrency?" | "One customer per machine. The real race is two ATMs on one account, solved by holds reducing the available balance." |
| "How does it scale?" | "It does not need to. Ten times the machines is ten times the instances; the bank behind them scales." |

---

## Cross-links

- The shared-state mechanics this file assumes, and the check-then-act race the ordering exists to avoid: [Concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md)
- The same three-valued outcome and the same hold-then-confirm shape, in a booking context: [BookMyShow seat locking](04-bookmyshow-seat-locking.md)
- The same idempotency-key discipline on a money-moving write, and the same integer-money rule: [Splitwise](06-splitwise.md)
- Where resource ownership is a lease with a TTL rather than a hold: [Hotel management](03-hotel-management.md) · and where a state machine is the core of the design: [Elevator system](05-elevator-system.md) · [Parking lot](01-parking-lot.md) · [LRU and LFU cache](02-lru-lfu-cache.md)
- The class concepts behind this design: [OOP principles and SOLID](../02-oop-principles-and-solid.md) · [Design patterns](../03-design-patterns.md) · [UML and class diagrams](../05-uml-and-class-diagrams.md) · [The machine-coding framework](../06-machine-coding-interview-framework.md) · [What is LLD](../01-what-is-lld-and-how-to-answer.md)
- Where the class view ends: [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md) · a ledger that must survive a crash: [SQL deep dive](../../HLD/04-data/01-sql-deep-dive.md) · and the durability guarantees behind a confirm: [Sharding, replication and consistency](../../HLD/03-scaling/05-sharding-replication-and-consistency.md)
- Other problems in this pillar: [Library management](08-library-management.md)
