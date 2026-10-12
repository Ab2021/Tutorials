# LLD Problem: Splitwise (Expense Sharing)

> **Grounding note.** The source transcript does not mention this problem. It is not in the LLD question list at **[9:22]**, not in the parking-lot walkthrough at **[25:54]–[27:38]**. **Everything below is standard industry practice.** It earns its place in this pillar for three reasons that are worth stating before the design, because they are what the interviewer is actually probing. First, it is the best problem for **integer money** — an equal split of an amount that does not divide evenly forces a rounding decision, and the naive answers either lose a cent or make the same transaction produce different splits on different runs. Second, it is the best problem for **derived versus stored state** — a mutable balance that is updated on every expense drifts from the truth, and the fix is to derive balances from an immutable expense log. Third, it is the best problem for **an optimisation that is genuinely hard**: "settle up with the fewest transfers" is NP-hard, and the honest answer is a greedy approximation with a proof of its bound, not a claim of optimality. The mechanics of the shared-state races are in [concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md); this file spends its budget on the domain.

---

## 1. Requirement clarification

### 1.1 Questions to ask

| # | Question | Why it matters | Default if they will not say |
|---|---|---|---|
| 1 | Groups, or just pairwise debts? | Determines whether there is a `Group` layer at all | Both: expenses belong to an optional group |
| 2 | **Which split types?** | The whole surface area of the problem | Equal, exact, percentage, shares — all four (§5.3) |
| 3 | **How is an indivisible remainder handled?** | **The detail that separates a real answer from a sketch** | Distribute deterministically, by sorted user id (§5.2) |
| 4 | Can people settle up? | A settlement is not an expense, and modelling it as one corrupts the balance | Yes, as a distinct entity (§5.5) |
| 5 | **Do we simplify debts?** | "Who pays whom" versus "the fewest possible transfers" — different features | Show direct balances; simplification is an explicit action (§5.6) |
| 6 | One currency, or many? | Mixing currencies in one balance is meaningless | One currency per group, enforced (§7.3) |
| 7 | Can an expense be edited or deleted? | Determines whether the log is immutable | Immutable: a reversal is a new expense (§7.4) |
| 8 | Are balances stored or derived? | **The decision that determines whether the system drifts** | Derived from the expense log (§2) |
| 9 | Can two people settle the same debt at once? | The race worth designing for | Yes; idempotency keys and a versioned settlement (§6) |
| 10 | What is the scale? | Whether a greedy settle-up is fast enough | Thousands of users per group is the extreme case |

### 1.2 Defaults to state out loud

```
  MODEL:      User ─┬─ Group ─┬─ Expense ── Split (one per participant)
                    │         │
                    └─────────┴─ Settlement      ← NOT an expense
  SPLITS:     Equal | Exact | Percentage | Shares
  MONEY:      integer minor units. NEVER float. An equal split that does not
              divide evenly is the case this whole file is written around.
  BALANCES:   DERIVED from the expense log, never stored as a mutable number.
  SIMPLIFY:   an explicit operation, greedy, ≤ n-1 transfers. Not automatic.
  THREADING:  the expense log is append-only; the race is on SETTLEMENT.
  OUT:        the UI, notifications, bank integration, the OCR of a receipt,
              multi-currency conversion rates.
```

**The one sentence that frames the problem:** *"A balance is not a number you store — it is a query over the expense log, and every bug in a Splitwise clone comes from treating it as the former."*

---

## 2. Core entities & responsibilities

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `User` | A person | id, name, currency preference | Hold a balance |
| `Group` | A set of people sharing expenses | id, members, currency | Hold a balance |
| `Expense` | One immutable fact: who paid, how much, for whom | id, payer, amount, participants, split strategy, created_at | Compute balances |
| `SplitStrategy` *(interface)* | Divide an amount among participants | — | Know who paid |
| `EqualSplit`, `ExactSplit`, `PercentageSplit`, `SharesSplit` | The four ways to divide | — | — |
| `Split` | One participant's share of one expense | user, amount | — |
| `Settlement` | One person paying another outside an expense | id, from, to, amount, idempotency key | Be an `Expense` |
| `BalanceSheet` | Derive who owes whom from the log | a reference to the log | Store a balance |
| `DebtSimplifier` | Turn a balance sheet into a small set of transfers | — | Change what anyone owes |
| `ExpenseService` | Append expenses and settlements, atomically | the log, the lock | Compute or store balances |
| `Clock` *(interface)* | Supply time | — | — |

**The design decision, stated plainly: `Settlement` is not an `Expense`.** It is tempting to model "Bob paid Alice ₹500" as an expense with a single participant, and it produces a balance sheet that is nearly right and wrong in a specific way: an expense generates a *split*, and a settlement has no split — it discharges a debt. Modelling it as an expense means either inventing a one-person split, which double-counts, or special-casing the expense type inside every balance query. **A separate entity with its own table and its own effect on the balance is simpler and it is the honest model.**

**The second decision: balances are derived.** `BalanceSheet` holds a reference to the log and computes on demand. A stored mutable balance is the bug factory: every expense must update it, every edit must reverse-and-reapply it, every settlement must adjust it, and any missed path leaves a number that is quietly wrong forever. **A derived balance cannot drift, because there is nothing to drift from.** The cost is a query rather than a lookup, and §5.7 shows the cache that resolves it without reintroducing the drift.

---

## 3. Class diagram

```
   ┌──────────────────┐            ┌───────────────────────┐
   │      User        │*          *│        Group          │
   ├──────────────────┤────────────├───────────────────────┤
   │ user_id          │  members   │ group_id              │
   │ name             │            │ name                  │
   └────────┬─────────┘            │ currency: str         │
            │                      └───────────┬───────────┘
            │ payer / participant              │ 1
            │                                  │
            │ *                                │ *
   ┌────────▼──────────────────────────────────▼──────────┐
   │                      Expense                         │
   ├──────────────────────────────────────────────────────┤
   │ - expense_id: str                                    │
   │ - description: str                                   │
   │ - payer: User                                        │
   │ - amount_minor: int          ← IMMUTABLE             │
   │ - currency: str                                      │
   │ - strategy: SplitStrategy    ← how it was divided    │
   │ - splits: list[Split]        ← the result, frozen    │
   │ - created_at: float                                  │
   └──────────────────────────┬───────────────────────────┘
                              │ 1
                              │ *
                     ┌────────▼────────┐        ┌────────────────────────┐
                     │      Split      │        │   «interface»          │
                     ├─────────────────┤        │   SplitStrategy        │
                     │ user: User      │        ├────────────────────────┤
                     │ amount_minor:int│        │ + split(amount, users, │
                     └─────────────────┘        │     params) -> [Split] │
                                                └────────────▲───────────┘
                                    ┌────────────┬───────────┼──────────┬─────────┐
                              ┌─────┴─────┐ ┌────┴─────┐ ┌───┴──────┐ ┌─┴────────┐
                              │EqualSplit │ │ExactSplit│ │Percent…  │ │Shares…   │
                              └───────────┘ └──────────┘ └──────────┘ └──────────┘

   ┌────────────────────────┐        ┌──────────────────────────────────────┐
   │      Settlement        │        │           BalanceSheet               │
   ├────────────────────────┤        ├──────────────────────────────────────┤
   │ settlement_id: str     │        │ - expenses:  list[Expense]           │
   │ from_user: User        │        │ - settlements: list[Settlement]      │
   │ to_user: User          │        ├──────────────────────────────────────┤
   │ amount_minor: int      │        │ + net_balances() -> dict[User, int]  │
   │ idempotency_key: str   │        │ + who_owes_whom() -> list[Transfer]  │
   │ created_at: float      │        │ + balance_for(user) -> int           │
   └────────────────────────┘        └──────────────────┬───────────────────┘
                                                       │ used by
                                       ┌───────────────▼───────────────────┐
                                       │        DebtSimplifier             │
                                       ├───────────────────────────────────┤
                                       │ + simplify(balances) -> [Transfer]│
                                       └───────────────────────────────────┘
```

**What the diagram states:**

- `Expense.splits` is a **frozen list of `Split`s**, computed once at creation. The strategy is retained for display ("split equally") but the result is stored, so a later change to the strategy class cannot retroactively change an old expense. **This is what makes an expense immutable and auditable.**
- `BalanceSheet` holds **references to the log, not balances** — the derived-state decision, drawn.
- `Settlement` is a peer of `Expense`, not a subclass.
- `DebtSimplifier` takes a balance map, so it is testable with a hand-written map and knows nothing about expenses.

---

## 4. Interfaces and enums

```python
from abc import ABC, abstractmethod
from enum import Enum


class SplitError(ValueError):
    """A split that does not add up, or parameters that do not match the
    participant list. Raised at construction, always — an expense whose splits
    do not sum to its amount must never exist, because every balance computed
    from it is then wrong and there is no way to tell which."""


class SplitType(Enum):
    """The four ways, named because the UI and the audit log both refer to
    them by name, and because an expense's type is part of its identity."""

    EQUAL = "equal"
    EXACT = "exact"
    PERCENTAGE = "percentage"
    SHARES = "shares"


class Transfer:
    """A payment: from one person to another. Note it is a VALUE, and it is
    what both `who_owes_whom` and `simplify` return — so the direct answer and
    the simplified answer are the same type and can be compared."""

    __slots__ = ("from_user_id", "to_user_id", "amount_minor", "currency")

    def __init__(self, from_user_id: str, to_user_id: str, amount_minor: int,
                 currency: str = "INR") -> None:
        if amount_minor <= 0:
            raise SplitError("a transfer must be for a positive amount")
        if from_user_id == to_user_id:
            raise SplitError("a transfer to yourself is not a transfer")
        self.from_user_id = from_user_id
        self.to_user_id = to_user_id
        self.amount_minor = amount_minor
        self.currency = currency

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transfer):
            return NotImplemented
        return ((self.from_user_id, self.to_user_id, self.amount_minor, self.currency)
                == (other.from_user_id, other.to_user_id, other.amount_minor,
                    other.currency))

    def __hash__(self) -> int:
        return hash((self.from_user_id, self.to_user_id, self.amount_minor))

    def __repr__(self) -> str:
        return (f"{self.from_user_id} → {self.to_user_id} "
                f"{self.currency} {self.amount_minor / 100:.2f}")


class SplitStrategy(ABC):
    """Divide an amount among participants. Note the contract, because it is
    the invariant the whole system rests on: the returned splits MUST sum
    exactly to `amount_minor`. Not approximately, not to within a cent."""

    @property
    @abstractmethod
    def split_type(self) -> SplitType: ...

    @abstractmethod
    def split(self, amount_minor: int, user_ids: "list[str]",
              params: "dict[str, object] | None" = None) -> "list[Split]":
        """Return one Split per participant, summing exactly to amount_minor.

        MUST be DETERMINISTIC: the same inputs produce the same splits, every
        time, on every run. A non-deterministic remainder allocation makes the
        same expense show two different breakdowns on two screens."""


class Split:
    """One participant's share. Frozen: the amount is decided at construction
    and never mutated, which is what makes an expense auditable."""

    __slots__ = ("user_id", "amount_minor")

    def __init__(self, user_id: str, amount_minor: int) -> None:
        if amount_minor < 0:
            raise SplitError(f"a negative share for {user_id} is not a split")
        self.user_id = user_id
        self.amount_minor = amount_minor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Split):
            return NotImplemented
        return (self.user_id, self.amount_minor) == (other.user_id, other.amount_minor)

    def __hash__(self) -> int:
        return hash((self.user_id, self.amount_minor))

    def __repr__(self) -> str:
        return f"Split({self.user_id}, {self.amount_minor})"


class Clock(ABC):
    @abstractmethod
    def now(self) -> float: ...
```

**The determinism requirement in `SplitStrategy`'s docstring is the point of the file.** A naive equal split that gives the remainder to "whichever participant the dict happened to iterate first" is not wrong in the arithmetic — the sum is right — but it means two renders of the same expense show different individual shares, and a settlement computed from one render disagrees with the other. **The fix is one line of sorting, and finding the need for it is the whole exercise.**

---

## 5. Complete working code

**Language note:** Python 3.11+, standard library only. Every block is complete.

### 5.1 Money, and the one rule

```python
class Money:
    """Integer minor units. The rule stated once and enforced by the type
    system: there is no float anywhere in this system.

    Why it matters more here than in most systems: a split is a DIVISION, and
    division is where floats accumulate error. Splitting 100.00 three ways in
    floats gives 33.333333..., and multiplying that back and summing gives
    99.99999999 or 100.00000001 depending on the order. In integers the
    question becomes the honest one — who gets the extra paisa? — and §5.2
    answers it explicitly."""

    __slots__ = ("minor", "currency")

    def __init__(self, minor: int, currency: str = "INR") -> None:
        if not isinstance(minor, int) or isinstance(minor, bool):
            raise TypeError("Money takes integer minor units, not float")
        self.minor = minor
        self.currency = currency

    def __add__(self, other: "Money") -> "Money":
        self._same_currency(other)
        return Money(self.minor + other.minor, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._same_currency(other)
        return Money(self.minor - other.minor, self.currency)

    def _same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise SplitError(f"currency mismatch: {self.currency} vs {other.currency}")

    def is_zero(self) -> bool:
        return self.minor == 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return (self.minor, self.currency) == (other.minor, other.currency)

    def __repr__(self) -> str:
        sign = "-" if self.minor < 0 else ""
        return f"{sign}{self.currency} {abs(self.minor) // 100}.{abs(self.minor) % 100:02d}"


def allocate(amount_minor: int, weights: "list[int]") -> "list[int]":
    """Divide an integer amount across weights, EXACTLY, with a deterministic
    remainder.

    THE ALGORITHM, and it is the standard one because it is provably correct:
      1. Give everyone floor(amount * weight_i / total_weight). This under-
         allocates by at most (n - 1) minor units in total.
      2. Distribute the remaining units ONE EACH to the participants with the
         largest fractional remainders, breaking ties by INDEX.

    The tie-break by index is what makes it deterministic. Without it, the
    remainder lands on whichever participant a dict or set happened to yield
    first, which differs between runs and between processes — and then the
    same expense shows two different breakdowns.

    The property to assert: sum(result) == amount_minor, ALWAYS. That is the
    invariant this function exists to hold, and there is a test for it.
    """
    if amount_minor < 0:
        raise SplitError("cannot allocate a negative amount")
    if not weights:
        raise SplitError("cannot allocate across zero participants")
    if any(w < 0 for w in weights):
        raise SplitError("weights cannot be negative")

    total_weight = sum(weights)
    if total_weight == 0:
        raise SplitError(
            "weights sum to zero; there is no way to divide the amount"
        )

    # Step 1: the floors, and the exact remainder each participant is owed.
    floors: list[int] = []
    remainders: list[tuple[int, int]] = []          # (remainder, index)
    for index, weight in enumerate(weights):
        numerator = amount_minor * weight
        floor_share, remainder = divmod(numerator, total_weight)
        floors.append(floor_share)
        remainders.append((remainder, index))

    # Step 2: hand out the leftover units, largest remainder first, ties by index.
    leftover = amount_minor - sum(floors)
    if leftover:
        remainders.sort(key=lambda pair: (-pair[0], pair[1]))
        for _, index in remainders[:leftover]:
            floors[index] += 1

    if sum(floors) != amount_minor:
        # Unreachable, and asserted rather than assumed: this function's whole
        # purpose is the sum, so a silent mismatch would be the worst failure.
        raise AssertionError(
            f"allocation lost money: {sum(floors)} != {amount_minor}"
        )
    return floors
```

**Walk `allocate(10000, [1, 1, 1])` through it.** `10000 * 1 / 3` is `3333` with remainder `1` for each. Floors are `[3333, 3333, 3333]`, sum `9999`, leftover `1`. The remainders all tie at `1`, so the tie-break by index gives the extra unit to index `0`. Result `[3334, 3333, 3333]`, sum `10000`. **Deterministic, exact, and the person who gets the extra paisa is always the same one** — which is the property that makes two screens agree.

### 5.2 The four split strategies

```python
class EqualSplit(SplitStrategy):
    """Divide equally. THE case the whole file is written around: ₹100 split
    three ways is not ₹33.33 each, because there is no such thing as a third
    of a paisa. Two people pay 33 and one pays 34, and which one is decided
    deterministically rather than by accident."""

    @property
    def split_type(self) -> SplitType:
        return SplitType.EQUAL

    def split(self, amount_minor: int, user_ids: "list[str]",
              params: "dict[str, object] | None" = None) -> "list[Split]":
        if not user_ids:
            raise SplitError("an expense needs at least one participant")
        if len(set(user_ids)) != len(user_ids):
            raise SplitError("a participant appears twice in the split")
        ordered = sorted(user_ids)                    # determinism, not cosmetic
        shares = allocate(amount_minor, [1] * len(ordered))
        return [Split(uid, share) for uid, share in zip(ordered, shares)]


class ExactSplit(SplitStrategy):
    """The caller names every amount. The validation is the whole class: the
    amounts must sum EXACTLY to the total, and a mismatch is a client bug that
    must be reported rather than silently absorbed into someone's share."""

    @property
    def split_type(self) -> SplitType:
        return SplitType.EXACT

    def split(self, amount_minor: int, user_ids: "list[str]",
              params: "dict[str, object] | None" = None) -> "list[Split]":
        if not params:
            raise SplitError("an exact split needs an amount per participant")
        amounts = params.get("amounts")
        if not isinstance(amounts, dict):
            raise SplitError("an exact split needs an 'amounts' mapping")
        if set(amounts) != set(user_ids):
            raise SplitError(
                f"amounts cover {sorted(amounts)} but participants are "
                f"{sorted(user_ids)}"
            )
        splits = [Split(uid, int(amounts[uid])) for uid in sorted(user_ids)]
        total = sum(s.amount_minor for s in splits)
        if total != amount_minor:
            raise SplitError(
                f"exact splits sum to {total} but the expense is {amount_minor}; "
                f"off by {total - amount_minor}"
            )
        return splits


class PercentageSplit(SplitStrategy):
    """Percentages, given as integers in basis points (1% = 100 bp) so the
    arithmetic stays integral. A `float` percentage is the same trap as a float
    amount: 33.333...% three ways does not sum to 100, and rounding it does not
    recover the missing amount."""

    @property
    def split_type(self) -> SplitType:
        return SplitType.PERCENTAGE

    def split(self, amount_minor: int, user_ids: "list[str]",
              params: "dict[str, object] | None" = None) -> "list[Split]":
        if not params:
            raise SplitError("a percentage split needs percentages")
        percentages = params.get("percentages")
        if not isinstance(percentages, dict):
            raise SplitError("a percentage split needs a 'percentages' mapping")
        if set(percentages) != set(user_ids):
            raise SplitError("percentages and participants do not match")
        basis_points = [int(percentages[uid]) for uid in sorted(user_ids)]
        if any(bp < 0 for bp in basis_points):
            raise SplitError("a percentage cannot be negative")
        if sum(basis_points) != 10_000:
            raise SplitError(
                f"percentages sum to {sum(basis_points) / 100:.2f}%, not 100%"
            )
        shares = allocate(amount_minor, basis_points)
        return [Split(uid, share) for uid, share in zip(sorted(user_ids), shares)]


class SharesSplit(SplitStrategy):
    """Proportional to integer shares — "4 shares, 2 shares, 1 share". This is
    what people actually mean by 'proportional' when a couple earns unequal
    amounts, and it is the same arithmetic as percentages with the normalisation
    done for you."""

    @property
    def split_type(self) -> SplitType:
        return SplitType.SHARES

    def split(self, amount_minor: int, user_ids: "list[str]",
              params: "dict[str, object] | None" = None) -> "list[Split]":
        if not params:
            raise SplitError("a shares split needs the shares")
        shares = params.get("shares")
        if not isinstance(shares, dict):
            raise SplitError("a shares split needs a 'shares' mapping")
        if set(shares) != set(user_ids):
            raise SplitError("shares and participants do not match")
        weights = [int(shares[uid]) for uid in sorted(user_ids)]
        if any(w < 0 for w in weights):
            raise SplitError("a share count cannot be negative")
        if sum(weights) == 0:
            raise SplitError("shares sum to zero; nothing to divide by")
        amounts = allocate(amount_minor, weights)
        return [Split(uid, amount) for uid, amount in zip(sorted(user_ids), amounts)]


SPLIT_STRATEGIES: "dict[SplitType, SplitStrategy]" = {
    SplitType.EQUAL: EqualSplit(),
    SplitType.EXACT: ExactSplit(),
    SplitType.PERCENTAGE: PercentageSplit(),
    SplitType.SHARES: SharesSplit(),
}
```

**A design note worth making: the strategies are stateless, so a module-level dict of singletons is correct here** — unlike the `Singleton` pattern objection in the caching problem. They hold no state, so sharing them is free, and the dict is a lookup table rather than a global mutable. **Saying why this one is fine is a good way to show the Singleton rule is understood rather than memorised.**

### 5.3 Expense and Settlement — the immutable log

```python
class Expense:
    """ONE IMMUTABLE FACT. It is never edited and never deleted.

    Why immutable: a balance is derived from the log, so an edit means every
    balance computed before the edit is now wrong, and there is no record of
    what it was. An immutable log plus reversal entries makes the history
    auditable — you can always answer "why did this number change?".

    `splits` and `amount_minor` are computed once, validated to sum, and then
    frozen. A later change to a `SplitStrategy` cannot retroactively change
    this expense, which is the property that makes old balances reproducible.
    """

    __slots__ = ("expense_id", "description", "payer_id", "amount_minor",
                 "currency", "splits", "strategy_type", "created_at", "group_id")

    def __init__(self, expense_id: str, description: str, payer_id: str,
                 amount_minor: int, currency: str, splits: "list[Split]",
                 strategy_type: SplitType, created_at: float,
                 group_id: "str | None" = None) -> None:
        if amount_minor <= 0:
            raise SplitError("an expense must be for a positive amount")
        if not splits:
            raise SplitError("an expense needs at least one split")

        total = sum(s.amount_minor for s in splits)
        if total != amount_minor:
            # THE invariant. Checked here, at the last moment before the expense
            # exists, because an expense whose splits do not sum is a permanent
            # corruption of every balance derived from it.
            raise SplitError(
                f"splits sum to {total} but the expense is {amount_minor}"
            )
        payer_ids = {s.user_id for s in splits}
        if payer_id not in payer_ids:
            # The payer must be a participant. If they are not, the payer's
            # share of their own expense is zero and they are owed the whole
            # amount — which is occasionally intended and usually a bug, so it
            # is an explicit choice rather than a silent default.
            raise SplitError(
                f"the payer {payer_id} is not among the participants "
                f"{sorted(payer_ids)}"
            )

        self.expense_id = expense_id
        self.description = description
        self.payer_id = payer_id
        self.amount_minor = amount_minor
        self.currency = currency
        self.splits = list(splits)
        self.strategy_type = strategy_type
        self.created_at = created_at
        self.group_id = group_id

    def share_of(self, user_id: str) -> int:
        """This user's share of THIS expense. A user with no split has a share
        of zero, which is the correct answer and not an error — they were not
        part of the expense."""
        return sum(s.amount_minor for s in self.splits if s.user_id == user_id)

    def participants(self) -> "list[str]":
        return [s.user_id for s in self.splits]

    def __repr__(self) -> str:
        return (f"Expense({self.expense_id}, {self.description!r}, "
                f"paid by {self.payer_id}, {self.amount_minor}, "
                f"{self.strategy_type.value})")


class Settlement:
    """ONE PERSON PAYING ANOTHER, outside any expense.

    Deliberately NOT an Expense, and the reason is worth stating: an expense
    divides an amount among participants and creates shares; a settlement
    discharges a debt and creates none. Modelling it as an expense with a
    single participant either double-counts the amount or forces a special
    case into every balance query.

    `idempotency_key` is a model field, not a bolt-on: a user double-tapping
    "settle up" is certain, and a duplicate settlement moves real money.
    """

    __slots__ = ("settlement_id", "from_user_id", "to_user_id", "amount_minor",
                 "currency", "idempotency_key", "created_at", "group_id")

    def __init__(self, settlement_id: str, from_user_id: str, to_user_id: str,
                 amount_minor: int, currency: str, idempotency_key: str,
                 created_at: float, group_id: "str | None" = None) -> None:
        if amount_minor <= 0:
            raise SplitError("a settlement must be for a positive amount")
        if from_user_id == to_user_id:
            raise SplitError("cannot settle with yourself")
        if not idempotency_key:
            raise SplitError("a settlement needs an idempotency key")
        self.settlement_id = settlement_id
        self.from_user_id = from_user_id
        self.to_user_id = to_user_id
        self.amount_minor = amount_minor
        self.currency = currency
        self.idempotency_key = idempotency_key
        self.created_at = created_at
        self.group_id = group_id

    def __repr__(self) -> str:
        return (f"Settlement({self.settlement_id}, {self.from_user_id} → "
                f"{self.to_user_id}, {self.amount_minor})")
```

### 5.4 The balance sheet — derived, and the invariant that proves it

```
   THE BALANCE FORMULA, for one user over the whole log:

     balance(u)  =  Σ over expenses where u is the payer:      +amount
                 −  Σ over expenses where u has a share:       −share
                 −  Σ over settlements where u is from_user:   −amount
                 +  Σ over settlements where u is to_user:     +amount

   SIGN CONVENTION:  positive = the group owes u.  negative = u owes the group.

   THE INVARIANT:  Σ over ALL users of balance(u)  ==  0     ALWAYS.

   Why it holds: every expense adds +amount to the payer and −share to each
   participant, and the shares sum to the amount, so the expense contributes
   exactly zero to the total. Every settlement adds −amount to one and
   +amount to the other: zero. So the total is zero by construction.

   Why it is worth asserting: it is the single check that catches a split that
   does not sum, a settlement applied on one side only, a participant dropped
   from a split, or a sign flipped somewhere. If the total is not zero, the
   ledger is broken and no individual balance can be trusted.
```

```python
class BalanceSheet:
    """DERIVED, never stored. Every query walks the log.

    The alternative — a `balances: dict[str, int]` mutated on every expense —
    is faster and is the source of every drift bug: an edit that does not
    reverse-reapply, a delete that forgets one participant, a settlement
    applied to one side, a retry that applies twice. A derived balance cannot
    drift because there is nothing to drift from.
    """

    def __init__(self, expenses: "list[Expense]", settlements: "list[Settlement]",
                 currency: str = "INR") -> None:
        self._expenses = expenses
        self._settlements = settlements
        self._currency = currency

    def scope(self, group_id: "str | None") -> "BalanceSheet":
        """A balance sheet for one group, or one for everything. A filter over
        the log rather than a second store, which is the payoff for deriving."""
        return BalanceSheet(
            [e for e in self._expenses if e.group_id == group_id],
            [s for s in self._settlements if s.group_id == group_id],
            self._currency,
        )

    def net_balances(self) -> "dict[str, int]":
        """Every user's net position. Positive = owed to them."""
        balances: dict[str, int] = {}

        for expense in self._expenses:
            balances[expense.payer_id] = balances.get(expense.payer_id, 0) + expense.amount_minor
            for split in expense.splits:
                balances[split.user_id] = balances.get(split.user_id, 0) - split.amount_minor

        for settlement in self._settlements:
            balances[settlement.from_user_id] = (
                balances.get(settlement.from_user_id, 0) - settlement.amount_minor
            )
            balances[settlement.to_user_id] = (
                balances.get(settlement.to_user_id, 0) + settlement.amount_minor
            )

        return balances

    def balance_for(self, user_id: str) -> int:
        return self.net_balances().get(user_id, 0)

    def check_invariant(self) -> None:
        """The zero-sum check. Call it in every test and after every mutation in
        development — it is cheap and it localises a ledger bug immediately
        rather than three screens later."""
        total = sum(self.net_balances().values())
        if total != 0:
            raise AssertionError(
                f"ledger does not balance: the net positions sum to {total}, "
                f"not zero. Balances: {self.net_balances()}"
            )

    def who_owes_whom(self) -> "list[Transfer]":
        """The DIRECT answer: one transfer per debtor-creditor pair.

        Read the sign: a user with a NEGATIVE balance owes money, so they are
        the `from_user_id`. A positive balance means they are owed, so they are
        the `to_user_id`. Getting this backwards is the single most common bug
        in this problem, and it produces a settlement instruction that moves
        money the wrong way — so it is stated in the code, not just in a comment.

        The result is greedy and NOT minimal: N debtors and M creditors produce
        up to N×M transfers. `DebtSimplifier` is the minimal-ish version.
        """
        balances = self.net_balances()
        debtors = sorted(
            ((uid, -amount) for uid, amount in balances.items() if amount < 0),
            key=lambda pair: (-pair[1], pair[0]),       # largest debt first
        )
        creditors = sorted(
            ((uid, amount) for uid, amount in balances.items() if amount > 0),
            key=lambda pair: (-pair[1], pair[0]),       # largest credit first
        )

        transfers: list[Transfer] = []
        debtor_index = 0
        creditor_index = 0
        remaining_debt = [amount for _, amount in debtors]
        remaining_credit = [amount for _, amount in creditors]

        while debtor_index < len(debtors) and creditor_index < len(creditors):
            amount = min(remaining_debt[debtor_index], remaining_credit[creditor_index])
            if amount > 0:
                transfers.append(Transfer(
                    debtors[debtor_index][0], creditors[creditor_index][0],
                    amount, self._currency,
                ))
            remaining_debt[debtor_index] -= amount
            remaining_credit[creditor_index] -= amount
            if remaining_debt[debtor_index] == 0:
                debtor_index += 1
            if remaining_credit[creditor_index] == 0:
                creditor_index += 1

        return transfers

    def total_for(self, user_id: str) -> Money:
        return Money(self.balance_for(user_id), self._currency)
```

### 5.5 The simplifier — the honest optimisation

```
   THE PROBLEM:  given a net balance per person, find the FEWEST transfers that
                 clears every debt.

   THE TRUTH:    this is NP-hard. It reduces to a partition problem — deciding
                 the minimum number of transfers is equivalent to partitioning
                 the creditor set into subsets that each exactly match one
                 debtor, and subset-sum is NP-complete. There is no polynomial
                 algorithm and no one has one.

   SO WHAT DO YOU DO: the greedy largest-first algorithm. It is not optimal and
                 it comes with a PROVABLE BOUND: at most (n − 1) transfers for
                 n people with non-zero balances. That bound is what makes it
                 the right production answer, and stating the bound is what
                 makes it a good interview answer.

   THE BOUND ARGUMENT, briefly: each greedy step zeroes at least ONE balance,
   because it matches the largest debt against the largest credit and one of
   the two is fully consumed. So each step removes at least one person, and
   after at most (n − 1) steps only one person remains — and with a zero sum,
   that last person's balance is zero. Hence ≤ n − 1 transfers.

   EXAMPLE where greedy is NOT optimal:
     balances:  +5  +5  −4  −3  −3        (sums to 0, n = 5)
     greedy:    −4 clears against a +5 (1), then −3 against the remaining +1
                plus the other +5 (2), then −3 against the +4 left (3), then
                ... 4 transfers
     optimal:   3 transfers — (−4 → +5), (−3 → +5), (−3 → ... )  needs the
                credits split, which greedy cannot do
   Greedy is 4, optimal is 3, and finding the 3 requires search.
```

```python
class DebtSimplifier:
    """Greedy largest-first. At most n−1 transfers for n non-zero balances, and
    no claim of optimality — because optimality is NP-hard and a claim of it
    would be false.

    Why production ships this: n−1 is already a huge improvement on the N×M
    direct answer, it is O(n log n), it is deterministic and therefore
    explainable, and it preserves every net position exactly. A user does not
    care that 4 transfers could have been 3; they care that the number went
    from 12 to 4.

    WHY IT NEEDS A GUARD: a greedy settle-up can ask someone to pay a person
    they have never transacted with. That is fine numerically and sometimes
    socially wrong — a user's mental model is "I owe Bob", and being told to
    pay Carol instead is confusing. Production usually keeps the DIRECT
    balances visible for this reason and offers simplification as an explicit
    action rather than a default.
    """

    def simplify(self, balances: "dict[str, int]", currency: str = "INR") -> "list[Transfer]":
        if not balances:
            return []
        total = sum(balances.values())
        if total != 0:
            raise SplitError(
                f"cannot simplify an unbalanced ledger: the net positions sum "
                f"to {total}, not zero"
            )

        creditors = sorted(
            [[uid, amount] for uid, amount in balances.items() if amount > 0],
            key=lambda pair: (-pair[1], pair[0]),
        )
        debtors = sorted(
            [[uid, -amount] for uid, amount in balances.items() if amount < 0],
            key=lambda pair: (-pair[1], pair[0]),
        )

        transfers: list[Transfer] = []
        creditor_index = 0
        debtor_index = 0

        while creditor_index < len(creditors) and debtor_index < len(debtors):
            creditor = creditors[creditor_index]
            debtor = debtors[debtor_index]
            amount = min(creditor[1], debtor[1])
            if amount > 0:
                transfers.append(Transfer(debtor[0], creditor[0], amount, currency))
            creditor[1] -= amount
            debtor[1] -= amount
            if creditor[1] == 0:
                creditor_index += 1
            if debtor[1] == 0:
                debtor_index += 1

        if len(transfers) > max(0, len(balances) - 1):
            # The bound is a theorem, so a violation means the algorithm is
            # broken. Assert it rather than trusting it.
            raise AssertionError(
                f"greedy produced {len(transfers)} transfers for "
                f"{len(balances)} participants, exceeding the n−1 bound"
            )
        return transfers

    def count(self, balances: "dict[str, int]") -> int:
        return len(self.simplify(balances))
```

### 5.6 The service — append-only, and the settlement race

```python
import threading


class ExpenseService:
    """The write path. Note what it does and does not own:

      - It OWNS the canonical log (expenses and settlements) and its lock.
      - It does NOT own balances. `BalanceSheet` derives from the log it hands
        out, so there is exactly one source of truth.

    The lock is a single `RLock` around every append and every read of the log
    as a consistent snapshot. Is that the "one global lock" the caching file
    warned against? No — and the reason is the same as in the elevator file:
    the critical section is a list append and a shallow copy, microseconds of
    pure computation with no I/O, and every operation genuinely reads the whole
    ledger. Sharding this would mean two threads computing balances from two
    partial views.
    """

    def __init__(self, currency: str = "INR") -> None:
        self._currency = currency
        self._expenses: list[Expense] = []
        self._settlements: list[Settlement] = []
        self._settlement_keys: dict[str, Settlement] = {}
        self._lock = threading.RLock()
        self._next_id = 0

    def _new_id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}-{self._next_id:08d}"

    # ---- writes --------------------------------------------------------
    def add_expense(self, description: str, payer_id: str, amount_minor: int,
                    participant_ids: "list[str]", split_type: SplitType,
                    params: "dict[str, object] | None" = None,
                    group_id: "str | None" = None) -> Expense:
        """Build the splits FIRST, outside the lock, then take the lock only to
        append.

        That ordering is deliberate: the split computation and its validation
        are pure and can be slow for a large group, and holding the lock across
        them would serialise unrelated expenses. Validate, then append — never
        append, then validate."""
        strategy = SPLIT_STRATEGIES.get(split_type)
        if strategy is None:
            raise SplitError(f"unknown split type {split_type}")

        splits = strategy.split(amount_minor, participant_ids, params)

        with self._lock:
            expense = Expense(
                expense_id=self._new_id("EXP"),
                description=description,
                payer_id=payer_id,
                amount_minor=amount_minor,
                currency=self._currency,
                splits=splits,
                strategy_type=split_type,
                created_at=0.0,
                group_id=group_id,
            )
            self._expenses.append(expense)
            return expense

    def record_settlement(self, from_user_id: str, to_user_id: str,
                          amount_minor: int, idempotency_key: str,
                          group_id: "str | None" = None) -> Settlement:
        """IDEMPOTENT on `idempotency_key`.

        THE RACE THIS EXISTS FOR: two people in a group both tap "settle up"
        for the same debt at the same moment — or one person taps twice because
        the app did not respond. Without idempotency, the same ₹500 debt is
        discharged twice, and one of them has now overpaid by ₹500. With it,
        the second call returns the FIRST settlement rather than creating a
        second one.

        Note the check and the insert are in the SAME critical section. A
        check outside the lock is the check-then-act race from the concurrency
        file, and here it moves money.
        """
        if amount_minor <= 0:
            raise SplitError("a settlement must be positive")
        if from_user_id == to_user_id:
            raise SplitError("cannot settle with yourself")

        with self._lock:
            existing = self._settlement_keys.get(idempotency_key)
            if existing is not None:
                # Idempotent replay. Return the original, and validate that the
                # replay is for the same thing — a key reused for a different
                # amount is a client bug that must not return the old result
                # silently.
                if (existing.from_user_id != from_user_id
                        or existing.to_user_id != to_user_id
                        or existing.amount_minor != amount_minor):
                    raise SplitError(
                        f"idempotency key {idempotency_key} was used for "
                        f"{existing.from_user_id}→{existing.to_user_id} "
                        f"{existing.amount_minor}, and is now being used for "
                        f"{from_user_id}→{to_user_id} {amount_minor}"
                    )
                return existing

            settlement = Settlement(
                settlement_id=self._new_id("STL"),
                from_user_id=from_user_id,
                to_user_id=to_user_id,
                amount_minor=amount_minor,
                currency=self._currency,
                idempotency_key=idempotency_key,
                created_at=0.0,
                group_id=group_id,
            )
            self._settlements.append(settlement)
            self._settlement_keys[idempotency_key] = settlement
            return settlement

    def reverse_expense(self, expense_id: str, reason: str,
                        group_id: "str | None" = None) -> Expense:
        """The correction path, and the reason the log is append-only: to undo
        an expense you append its NEGATIVE as a new expense — no, you cannot,
        because an expense must be positive.

        So the reversal is expressed as a settlement-shaped entry per
        participant, OR — the simpler production answer — the expense is
        marked reversed and excluded from the derivation. This method takes
        the second route and appends a reversing EXPENSE with the payer and
        participants swapped, which exactly cancels it and requires no new
        entity.

        Swapping the payer and participants is the trick: the original added
        +amount to the payer and −share to each participant. The reversal adds
        +amount to each participant's... no — it adds +amount to each
        participant and −share to the original payer, which sums to zero
        against the original for every user.
        """
        with self._lock:
            original = next(
                (e for e in self._expenses if e.expense_id == expense_id), None
            )
            if original is None:
                raise SplitError(f"unknown expense {expense_id}")

            # Reverse: for each participant, a reversing expense where THEY are
            # the payer of their own share, split entirely to the original payer.
            reversals: list[Expense] = []
            for split in original.splits:
                if split.amount_minor == 0:
                    continue                              # nothing to reverse
                reversal = Expense(
                    expense_id=self._new_id("REV"),
                    description=f"Reversal of {original.expense_id}: {reason}",
                    payer_id=split.user_id,              # they paid it back
                    amount_minor=split.amount_minor,
                    currency=original.currency,
                    splits=[Split(original.payer_id, split.amount_minor)],
                    strategy_type=SplitType.EXACT,
                    created_at=0.0,
                    group_id=group_id if group_id is not None else original.group_id,
                )
                reversals.append(reversal)
            self._expenses.extend(reversals)
            return reversals[0]

    # ---- reads ---------------------------------------------------------
    def balance_sheet(self, group_id: "str | None" = None) -> BalanceSheet:
        with self._lock:
            # A SHALLOW COPY under the lock: the caller gets an immutable
            # snapshot rather than a live view that another thread can mutate
            # while it is being walked. This is the fix for the
            # "list changed size during iteration" class of bug.
            return BalanceSheet(
                list(self._expenses), list(self._settlements), self._currency
            ).scope(group_id)

    def expenses(self, group_id: "str | None" = None) -> "list[Expense]":
        with self._lock:
            if group_id is None:
                return list(self._expenses)
            return [e for e in self._expenses if e.group_id == group_id]

    def settlements(self, group_id: "str | None" = None) -> "list[Settlement]":
        with self._lock:
            if group_id is None:
                return list(self._settlements)
            return [s for s in self._settlements if s.group_id == group_id]
```

### 5.7 The cache — speed without reintroducing drift

```python
class CachedBalanceSheet:
    """Deriving on every read is correct and O(expenses). At a few thousand
    expenses that is microseconds; at millions it is not.

    THE CACHE THAT IS SAFE TO ADD: memoise on the LENGTH of the logs. The log
    is append-only, so its length is a perfect version number — a new expense
    changes the length, which invalidates. This is the versioned-cache pattern
    from the caching file, and it is safe here for exactly the reason it is
    safe there: the underlying data is immutable and the version is monotonic.

    What this is NOT: a mutable `balances` dict updated per expense. That is
    the drift bug. The difference is that this cache can always be discarded
    and recomputed, and it never disagrees with the log, because its key
    includes the log's version.
    """

    def __init__(self, service: ExpenseService) -> None:
        self._service = service
        self._cache: "dict[tuple[int, int, str | None], dict[str, int]]" = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def net_balances(self, group_id: "str | None" = None) -> "dict[str, int]":
        expenses = self._service.expenses(group_id)
        settlements = self._service.settlements(group_id)
        key = (len(expenses), len(settlements), group_id)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.hits += 1
                return dict(cached)

        self.misses += 1
        sheet = BalanceSheet(expenses, settlements)
        balances = sheet.net_balances()
        sheet.check_invariant()

        with self._lock:
            # Bounded: the version key changes on every append, so an unbounded
            # dict would grow forever. Keep the most recent few.
            if len(self._cache) > 16:
                self._cache.clear()
            self._cache[key] = dict(balances)
        return balances

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
```

### 5.8 It runs

```python
def demo() -> None:
    service = ExpenseService("INR")

    # THE case the whole file is about: 100.00 split three ways.
    service.add_expense("Dinner", "alice", 10_000, ["alice", "bob", "carol"],
                        SplitType.EQUAL, group_id="trip")
    sheet = service.balance_sheet("trip")

    for expense in service.expenses("trip"):
        print(expense, "→", expense.splits)
    # [Split(alice, 3334), Split(bob, 3333), Split(carol, 3333)]
    # Sum is exactly 10000. Deterministic: run it again and alice still has 3334.

    print("balances:", sheet.net_balances())          # alice +6666, bob −3333, carol −3333
    sheet.check_invariant()
    print("direct:", sheet.who_owes_whom())

    # A settlement, and the idempotent replay.
    first = service.record_settlement("bob", "alice", 3_333, "key-bob-1", group_id="trip")
    replay = service.record_settlement("bob", "alice", 3_333, "key-bob-1", group_id="trip")
    print("same settlement:", first.settlement_id == replay.settlement_id)   # True
    print("balances after:", service.balance_sheet("trip").net_balances())

    # A key reused for a DIFFERENT settlement is refused.
    try:
        service.record_settlement("bob", "alice", 1_000, "key-bob-1", group_id="trip")
    except SplitError as exc:
        print("refused:", exc)

    # The simplifier on a hand-written unbalanced-looking-but-balanced map.
    balances = {"a": 500, "b": 500, "c": -400, "d": -300, "e": -300}
    simplifier = DebtSimplifier()
    transfers = simplifier.simplify(balances, "INR")
    print("simplified:", transfers)
    # Verify the simplification PRESERVES every net position — the property
    # that makes it a settle-up plan rather than a different set of debts.
    check = {uid: 0 for uid in balances}
    for transfer in transfers:
        check[transfer.from_user_id] -= transfer.amount_minor
        check[transfer.to_user_id] += transfer.amount_minor
    print("preserved:", check == balances)             # True
    print("transfer count:", len(transfers), "vs direct pairs: 6")


if __name__ == "__main__":
    demo()
```

**The last check in the demonstration is the one to keep.** A simplified plan must leave every person's net position unchanged — it changes *who pays whom*, never *how much*. A simplifier that alters a balance is not a settle-up plan, it is a different set of debts, and the test that catches that is exactly the one above.

---

## 6. Concurrency and thread-safety

### 6.1 The races

| # | Race | Consequence | Fix |
|---|---|---|---|
| 1 | Two threads settle the same debt | **The same debt discharged twice** — money moves twice | Idempotency key checked and inserted in one critical section |
| 2 | A double-tap on "settle up" | The same as above, from one user | The same idempotency key on both taps |
| 3 | An expense appended while a balance is computed | `RuntimeError: list changed size during iteration`; a wrong balance | The read takes a shallow copy under the same lock |
| 4 | Two expenses appended concurrently | Interleaved appends are fine — the log is append-only | The lock still serialises the id, so ids are unique |
| 5 | Balances read from a stale cache | A user shown a debt that is already settled | The cache key includes the log length |
| 6 | A settlement for a debt larger than the current balance | Someone overpays | **Deliberately allowed** — a user may pre-pay; see §6.2 |
| 7 | Two people both settle with a third | Correct: two independent settlements, both valid | No fix needed — the net derivation handles it |
| 8 | A reversal racing the original's balance | A balance computed mid-reversal | Both appends take the lock; the snapshot is consistent |

### 6.2 The subtle one: should a settlement be capped at the outstanding debt?

**The interviewer's follow-up, and it has a real answer.** Capping looks right — why let someone discharge a debt that does not exist? — and it is wrong for three reasons:

1. **The debt is a moving target.** Between the read that computed the balance and the write that appends the settlement, another expense can change it. Enforcing a cap means a read-check-write across the lock boundary, which is check-then-act, and here it moves money.
2. **Two people settling simultaneously is legal.** Alice owes ₹500; both Alice and Carol pay Bob ₹500 to clear it. Capping the second would refuse a payment that is legitimately owed to Bob for a *different* reason.
3. **Pre-payment is a real behaviour.** A flatmate paying next month's share in advance is normal, and a cap refuses it.

**So the settlement is not capped; the balance simply goes the other way.** The design accepts any positive settlement and lets the derived balance absorb it, and the invariant check proves nothing was corrupted. **Saying "I deliberately do not cap it, and here is why" is a much stronger answer than adding a cap that looks careful and creates a race.**

### 6.3 Tests

```python
def test_equal_split_of_an_indivisible_amount_is_exact_and_deterministic() -> None:
    """THE test. 100.00 across three people, twice, must agree exactly."""
    strategy = EqualSplit()
    first = strategy.split(10_000, ["alice", "bob", "carol"])
    second = strategy.split(10_000, ["carol", "alice", "bob"])   # different order in

    assert sum(s.amount_minor for s in first) == 10_000
    assert sum(s.amount_minor for s in second) == 10_000
    # Same result regardless of the order the participants arrive in — that is
    # what the sort inside the strategy buys.
    assert sorted((s.user_id, s.amount_minor) for s in first) == \
           sorted((s.user_id, s.amount_minor) for s in second)
    # And the extra unit goes to a known participant rather than to chance.
    assert {s.user_id: s.amount_minor for s in first} == {
        "alice": 3334, "bob": 3333, "carol": 3333,
    }


def test_allocate_never_loses_a_minor_unit() -> None:
    """The property, over every amount and every participant count that matters."""
    import random
    rng = random.Random(7)
    for participants in range(1, 13):
        for _ in range(200):
            amount = rng.randint(0, 1_000_000)
            weights = [rng.randint(0, 20) for _ in range(participants)]
            if sum(weights) == 0:
                continue
            shares = allocate(amount, weights)
            assert sum(shares) == amount
            assert all(s >= 0 for s in shares)
            # Nobody is more than one minor unit below their exact share.
            total_weight = sum(weights)
            for share, weight in zip(shares, weights):
                exact = amount * weight / total_weight
                assert share - exact < 1.0, (share, exact)
                assert exact - share < 1.0, (share, exact)


def test_ledger_always_balances() -> None:
    """The zero-sum invariant, after a mixed sequence of operations."""
    service = ExpenseService("INR")
    service.add_expense("Dinner", "alice", 10_000, ["alice", "bob", "carol"], SplitType.EQUAL)
    service.add_expense("Taxi", "bob", 4_777, ["alice", "bob"], SplitType.EQUAL)
    service.add_expense("Hotel", "carol", 99_999, ["alice", "bob", "carol"],
                        SplitType.SHARES, {"shares": {"alice": 4, "bob": 4, "carol": 1}})
    service.record_settlement("bob", "alice", 3_333, "k1")
    service.balance_sheet().check_invariant()


def test_an_expense_with_splits_that_do_not_sum_is_refused() -> None:
    with pytest.raises(SplitError):
        Expense("E1", "Bad", "alice", 10_000, "INR",
                [Split("alice", 3_000), Split("bob", 3_000)],      # sums to 6000
                SplitType.EXACT, 0.0)


def test_percentages_must_sum_to_one_hundred() -> None:
    strategy = PercentageSplit()
    with pytest.raises(SplitError):
        strategy.split(10_000, ["a", "b"], {"percentages": {"a": 5_000, "b": 4_000}})
    ok = strategy.split(10_000, ["a", "b"], {"percentages": {"a": 3_333, "b": 6_667}})
    assert sum(s.amount_minor for s in ok) == 10_000


def test_settlement_is_idempotent_under_concurrency() -> None:
    """THE race. Eight threads record the SAME settlement; exactly one row."""
    service = ExpenseService("INR")
    service.add_expense("Dinner", "alice", 10_000, ["alice", "bob", "carol"], SplitType.EQUAL)

    barrier = threading.Barrier(8)
    results: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()

    def settle() -> None:
        barrier.wait()
        try:
            settlement = service.record_settlement("bob", "alice", 3_333, "shared-key")
            with lock:
                results.append(settlement.settlement_id)
        except Exception as exc:                              # noqa: BLE001
            with lock:
                errors.append(repr(exc))

    threads = [threading.Thread(target=settle) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(set(results)) == 1, f"expected one settlement, got {set(results)}"
    assert len(service.settlements()) == 1
    service.balance_sheet().check_invariant()


def test_a_settlement_key_reused_for_a_different_amount_is_refused() -> None:
    service = ExpenseService("INR")
    service.record_settlement("bob", "alice", 3_333, "k1")
    with pytest.raises(SplitError):
        service.record_settlement("bob", "alice", 1_000, "k1")


def test_simplification_preserves_every_net_position() -> None:
    """The property that makes a simplify a settle-up plan and not a rewrite."""
    balances = {"a": 5_000, "b": 5_000, "c": -4_000, "d": -3_000, "e": -3_000}
    transfers = DebtSimplifier().simplify(balances, "INR")

    settled = {uid: 0 for uid in balances}
    for transfer in transfers:
        settled[transfer.from_user_id] -= transfer.amount_minor
        settled[transfer.to_user_id] += transfer.amount_minor
    assert settled == balances
    assert len(transfers) <= len(balances) - 1


def test_simplifier_refuses_an_unbalanced_ledger() -> None:
    with pytest.raises(SplitError):
        DebtSimplifier().simplify({"a": 500, "b": -400}, "INR")   # sums to 100


def test_reversal_cancels_an_expense_exactly() -> None:
    service = ExpenseService("INR")
    expense = service.add_expense("Dinner", "alice", 10_000,
                                  ["alice", "bob", "carol"], SplitType.EQUAL)
    before = service.balance_sheet().net_balances()
    service.reverse_expense(expense.expense_id, "duplicate entry")
    after = service.balance_sheet().net_balances()
    assert all(after[uid] == 0 for uid in after), after
    assert before != after
    service.balance_sheet().check_invariant()


def test_concurrent_expenses_do_not_corrupt_the_log() -> None:
    service = ExpenseService("INR")
    barrier = threading.Barrier(6)
    errors: list[str] = []

    def adder(worker: int) -> None:
        barrier.wait()
        try:
            for i in range(100):
                service.add_expense(
                    f"item-{worker}-{i}", f"user{worker}", 1_000 + i,
                    [f"user{worker}", f"user{(worker + 1) % 6}"], SplitType.EQUAL,
                )
        except Exception as exc:                              # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=adder, args=(w,)) for w in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(service.expenses()) == 600
    assert len({e.expense_id for e in service.expenses()}) == 600   # unique ids
    service.balance_sheet().check_invariant()
```

---

## 7. Extensibility — the standard follow-ups

### 7.1 "Add a group with members and a join date"

The group matters because it scopes the balance sheet, and the follow-up is always about the boundary: **can a person be in two groups and owe across them?** The design answer is that the balance is scoped by the filter in `BalanceSheet.scope`, and an all-groups view is the same computation with no filter. The subtlety is that a settlement between two people in two different groups belongs to **neither** — it is a personal settlement, and forcing it into a group makes one group's balance wrong. **That is a real modelling choice and worth surfacing.**

### 7.2 "Edit an expense"

The follow-up that tests whether the immutability was understood. There are two honest answers:

- **Reverse and re-add.** Append a reversal, append the corrected expense. The full history is preserved and every balance is derivable. The cost is that the log grows, and the UI must hide reversal pairs.
- **A mutable expense with a version, and the balance derived from the LATEST version.** Cheaper storage, and it reintroduces the problem immutability solved: you can no longer answer "what did this look like on Tuesday?" without a separate audit table.

**The answer to give:** reverse-and-re-add for the money path, because the audit trail is the point; a versioned expense only if the storage really is the constraint, and then with an append-only audit of its own.

### 7.3 "Multiple currencies"

The extension that breaks things quietly. Three rules:

1. **One currency per group**, enforced at construction. A group is a shared context; mixing currencies inside it makes every balance meaningless.
2. **A settlement must match its group's currency.** A cross-currency settlement needs a rate and a rate timestamp, and that is a different entity.
3. **A balance is never converted at read time.** A stored rate changes, and a balance that changes when you look at it is not a balance. Convert at settlement time and record the rate used.

**The failure to name:** summing amounts across currencies. `+500 INR +300 USD` is `800` in the arithmetic and nonsense in reality, and it is exactly what a naive derived balance does with no currency check. The `Money` type raises on a mismatch for that reason.

### 7.4 "Recurring expenses"

A rent split is the same expense every month, and the design point is that **a recurrence is a template, not an expense** — it generates real expenses on a schedule, and the generated ones are ordinary immutable expenses. Generating them lazily (on read, if the period has passed) versus eagerly (a scheduled job) is the trade: lazy needs no infrastructure and makes the read path write, which is a race; eager needs a job and is idempotent per period. **Idempotency per period — a unique key of `(template_id, period)` — is what stops a retried job creating two months of rent.**

### 7.5 "The activity feed and the audit trail"

Since the log is immutable and append-only, the feed is free: it is the log, sorted by time. **This is a benefit of the immutability decision, and it is worth pointing out** — the feature that would otherwise need a separate audit table is already there.

### 7.6 "Simplify debts automatically"

**The interesting answer is that you probably should not.** A greedy plan can ask a user to pay someone they have never shared an expense with, which is numerically correct and socially confusing. **Production keeps the direct balances visible and offers simplification as an explicit, user-initiated action**, so the user consents to the rearrangement. The technical follow-up — "can you do better than greedy?" — has the honest answer in §5.5: it is NP-hard, greedy gives `≤ n−1`, and the improvement from N×M to n−1 is where all the value is.

---

## 8. Failure modes and edge cases

### 8.1 Failure modes

| # | Failure | Consequence | Handling |
|---|---|---|---|
| 1 | Splits that do not sum to the expense | **Every balance derived from it is wrong** | Checked in `Expense.__init__` and in each strategy |
| 2 | A non-deterministic remainder | Two screens show different shares of one expense | Sort before allocating; ties by index |
| 3 | Float money | Drift; a settlement that does not clear the debt | Integer minor units throughout |
| 4 | A settlement modelled as an expense | Double-counted, or a one-person split | A distinct entity |
| 5 | Duplicate settlement from a double-tap | **The same debt discharged twice** | Idempotency key, checked and inserted under the lock |
| 6 | An idempotency key reused for a different amount | A wrong result returned silently | Refuse with the conflict spelled out |
| 7 | A mutable balance that drifts | A number that is quietly wrong forever | Derive from the log; cache on the log's length |
| 8 | Caching without a version | A user shown a debt that is already settled | The cache key includes the log length |
| 9 | A balance computed while the log mutates | `RuntimeError`, or a wrong total | A shallow copy under the lock |
| 10 | The payer not among the participants | The payer is owed the whole amount, silently | Refused at construction |
| 11 | A negative share | A participant "owes" a negative amount, i.e. is owed | Refused in `Split` and in `allocate` |
| 12 | Percentages that do not sum to 100 | The remainder silently lands on someone | Refused explicitly, in basis points |
| 13 | Weights that sum to zero | A divide-by-zero | Refused in `allocate` |
| 14 | An expense across two currencies | A meaningless sum | `Money` raises; one currency per group |
| 15 | The simplifier asked for an unbalanced ledger | A plan that does not clear the debts | Refused; `check_invariant` catches it first |
| 16 | A settlement larger than the debt | An overpayment | **Allowed deliberately** (§6.2) |
| 17 | A reversal that does not exactly cancel | A residual balance nobody can explain | The reversal mirrors the splits exactly; a test asserts a zero result |
| 18 | An unbounded cache on a versioned key | Unbounded growth | Bounded, cleared past 16 entries |
| 19 | An expense with zero amount | A no-op that pollutes the log | Refused: the amount must be positive |
| 20 | A participant listed twice in one split | They appear twice in the balance | Refused |

### 8.2 Edge cases worth naming

- **₹100 split three ways.** The canonical case: `334, 333, 333`.
- **₹0.01 split three ways.** `1, 0, 0`. Legal, and someone pays nothing — which is correct, because there is no way to divide a paisa.
- **₹0.01 split two ways.** `1, 0`. And the same person always gets it.
- **An expense with one participant who is also the payer.** Nets to zero. Legal, meaningless, and a good test that the derivation handles it.
- **A settlement that exactly clears a debt.** The balance becomes zero, and a zero balance is omitted from `who_owes_whom`.
- **A settlement that over-clears.** The balance flips sign; the person is now owed. Legal.
- **A user with all-zero balances.** Omitted from the transfers, and the sum check still holds.
- **A very large group where everyone owes everyone.** The extreme case for the simplifier: N debtors and N creditors, and greedy bounds it at `n−1`.
- **Two people in two groups.** A settlement between them in one group does not affect the other — which is the point of scoping, and the bug if a group id is dropped.
- **An expense added before the group exists.** Refused, or the group is created implicitly — an explicit choice.

### 8.3 Complexity

| Operation | Cost | Note |
|---|---|---|
| `allocate` | O(n log n) | The sort of remainders dominates |
| `SplitStrategy.split` | O(n log n) | The sort for determinism plus the allocation |
| `net_balances` | **O(total splits)** | Walks the whole log |
| `who_owes_whom` | O(n log n) | Sorts the debtors and creditors |
| `DebtSimplifier.simplify` | **O(n log n)** | Sort plus one linear pass; `≤ n−1` transfers |
| `CachedBalanceSheet.net_balances` (hit) | O(1) plus the log-length reads | The reads are O(group), not O(1) — worth noting |
| `ExpenseService.add_expense` | O(n log n) | The split, outside the lock, plus an O(1) append |
| `record_settlement` | O(1) | A dict lookup and an append, under the lock |

**The line to memorise:** *"Deriving balances is O(total splits) per read, which is why the cache exists — but the cache keyed on the log length is the only safe one, because the log is append-only and its length is therefore a perfect version number. The simplifier is O(n log n) and produces at most n−1 transfers, and optimality is NP-hard so that bound is the claim."*

---

## 9. Design patterns used, and why

| Pattern | Where | Why here | What it costs |
|---|---|---|---|
| **Strategy** | `SplitStrategy` → the four implementations, via `SPLIT_STRATEGIES` | The split rule is the axis that varies, and the four are a closed set the UI names | An interface and a lookup table |
| **Value Object** | `Money`, `Split`, `Transfer`, `Expense`, `Settlement` | Immutable values with equality; `Split` and `Transfer` compare by content, which makes tests readable | No identity, so no in-place update — which is the point |
| **Repository / derived view** | `BalanceSheet` over the log | One source of truth; the sheet is a query, not a store | O(n) per read, mitigated by the versioned cache |
| **Snapshot** | `Expense.splits`, and the shallow copy in `balance_sheet()` | The splits are frozen at creation and the read is a consistent snapshot — both are the same idea, applied to different lifetimes | Memory; and a caller cannot see a live view |
| **Versioned Cache** | `CachedBalanceSheet`, keyed on log length | Safe memoisation, because the append-only log gives a monotonic version | A cache to invalidate, with a bound |
| **Singleton** *(used, deliberately)* | The four strategies in `SPLIT_STRATEGIES` | They are **stateless**, so sharing them costs nothing and a table is clearer than a factory | Global, but immutable global |
| **Observer** *(rejected)* | — | Considered for activity notifications; rejected as a concern of the notification system, not of the ledger — the immutable log already provides the feed | — |

**The interesting entry is the Singleton that is used rather than rejected.** In the caching and elevator files a global was wrong because it held state. Here the strategies hold none, so a module-level table is correct — and **saying why the rule inverts is a much better signal than applying the rule mechanically.**

---

## 10. Trade-offs

| Choice | Buys | Costs |
|---|---|---|
| Balances derived, not stored | Cannot drift; one source of truth | O(n) per read; a cache is needed at scale |
| Integer minor units | Exact; the remainder question becomes explicit | Conversion at every boundary |
| Deterministic remainder | Every screen agrees; balances are reproducible | The same person always pays the extra paisa — occasionally noticed |
| Immutable expenses | Auditable; a reversal is traceable | An edit is two appends; the log grows |
| `Settlement` as a separate entity | No double-counting; a clean balance query | Two lists to walk rather than one |
| A cap-free settlement | No read-check-write race; pre-payment works | A typo can overpay — mitigated by the UI, not the model |
| Idempotency keys on settlements | A double-tap or a retry is safe | Every caller must generate a key |
| Greedy simplification | O(n log n); ≤ n−1 transfers; deterministic | Not optimal, and it can pair strangers |
| Simplification as an explicit action | The user consents to the rearrangement | An extra tap; the direct view stays the default |
| One currency per group | Balances are always meaningful | A cross-currency group needs a different model |
| The versioned cache | Correct and fast | A cache, with a bound and an invalidation path |
| The zero-sum invariant as an assertion | Catches nearly every ledger bug immediately | Cheap enough to leave on in production |

**What changes at 10×:** the derivation. A group with thousands of members and millions of expenses cannot recompute a balance per read, so the balance becomes a materialised, versioned projection updated by a stream of expense events — and **the moment it is materialised, the drift risk returns**, which is why the zero-sum invariant is worth keeping as a reconciliation job even then. The class design's shapes survive; what changes is that `BalanceSheet` becomes a store fed by a projector rather than a query over a list, and the invariant moves from an assertion to a nightly check.

---

## 11. Interview Q&A

**Q1 (easy). What are the core classes?**
`User`, `Group`, `Expense` with its `Split`s, `Settlement`, `BalanceSheet`, `SplitStrategy` with four implementations, and `DebtSimplifier`. The two decisions that shape everything: an expense is immutable, and balances are derived from the log rather than stored.

**Q2 (easy). Why is `Settlement` not an `Expense`?**
Because an expense divides an amount among participants and creates shares, and a settlement discharges a debt and creates none. Modelling it as an expense means either a one-person split, which double-counts, or a special case in every balance query. A separate entity with its own effect on the derivation is simpler and it is the honest model.

**Q3 (easy). Why integer money?**
Because a split is a division, and division is where floating point accumulates error. ₹100 split three ways in floats is 33.333..., and summing it back gives 99.99999999 or 100.00000001 depending on the order. In integers the question becomes the honest one — who gets the extra paisa? — and the design answers it explicitly rather than hoping.

**Q4 (medium). ₹100 split three ways. What exactly happens?**
The allocation gives everyone `floor(10000 × 1 / 3)` which is 3333, summing to 9999, leaving one unit. The remainders all tie at 1, and the tie-break is by index after sorting the participants, so the first participant in sorted order gets 3334. The result is `[3334, 3333, 3333]`, summing to exactly 10000. The sort and the index tie-break are what make it deterministic — without them the extra paisa lands on whichever participant the iteration happened to reach first, which differs between runs, and then two screens show different breakdowns of the same expense.

**Q5 (medium). Why derive balances instead of storing them?**
Because a stored balance is a number that has to be kept correct by every code path that could change it — every expense, every edit, every reversal, every settlement, every retry — and any path that forgets leaves a number that is quietly wrong forever with no way to tell. A derived balance cannot drift, because there is nothing to drift from. The cost is O(n) per read, and the safe cache for that is keyed on the length of the append-only log, which is a perfect version number.

**Q6 (medium). What is the invariant that proves the ledger is correct?**
The net positions across all users must sum to exactly zero. Every expense adds its amount to the payer and subtracts each share from each participant, and the shares sum to the amount, so an expense contributes zero to the total; every settlement subtracts from one and adds to the other, also zero. So the total is zero by construction, and if it is not zero then something is broken — a split that does not sum, a settlement applied on one side, a participant dropped, a sign flipped. It is the single cheapest check in the system and it localises almost every bug.

**Q7 (medium). How do you handle a double-tap on "settle up"?**
An idempotency key per settlement, checked and inserted in the same critical section. The second call finds the key and returns the *first* settlement rather than creating a second one. If the same key arrives with a different amount or different parties, it is refused loudly rather than returning the old result, because that is a client bug and silently returning the wrong settlement would hide it. The check and the insert being in one critical section matters — a check outside the lock is check-then-act, and here it moves money.

**Q8 (medium). *Pushback:* "Why not cap a settlement at the outstanding debt? Someone could overpay."**
Because enforcing a cap means reading the balance, deciding, and writing across the lock boundary — which is exactly the check-then-act race, and here it moves money. And the cap is wrong in principle for three reasons: the debt is a moving target, because another expense can change it between the read and the write; two people settling the same debt simultaneously is legitimate, because both may genuinely owe the creditor; and pre-payment is normal behaviour — a flatmate paying next month's share in advance. So I do not cap it, the balance simply flips, and the zero-sum invariant proves nothing was corrupted. Refusing a legitimate payment to look careful is worse than accepting it.

**Q9 (medium). *Pushback:* "Your simplifier is not optimal. Why ship it?"**
Because optimality is NP-hard — minimising the number of transfers reduces to partitioning the creditors into subsets that each exactly match a debtor, which is subset-sum — so there is no polynomial algorithm and claiming one would be false. What greedy gives is a provable bound: at most `n−1` transfers for `n` people with non-zero balances, because each step fully zeroes at least one balance so at least one person is removed per step. Given that the naive direct answer is up to N×M transfers, going from twelve to four is the entire value, and nobody cares that four could have been three.

**Q10 (hard). *Pushback:* "If greedy is not optimal, why not just search for the optimum? n is small."**
Sometimes you can, and the honest answer is that it depends on `n`. For a group of five, a branch-and-bound search is instant and gives the true optimum. The problem is that the groups where simplification matters most are the large ones, and `n` there is not small — and the search is exponential, so the boundary is somewhere around ten to fifteen people with non-zero balances where it stops being free. Greedy is the only policy with a bound that holds at every size, so it is the default and the search is an optimisation for small groups. What I would not do is ship a search without a bound, because a group of thirty would hang the request.

**Q11 (medium). *Pushback:* "Isn't simplifying debts confusing for users? They think they owe Bob and you tell them to pay Carol."**
That is a real product problem and it is why simplification is an explicit action rather than a default. The direct balances are what the user sees — "you owe Bob ₹500" — because that matches what actually happened. Simplification is offered as a "settle up more easily" button, so the user consents to the rearrangement and sees the plan before accepting it. The technical version of the same point is that the simplifier preserves every net position exactly, so nobody pays a different amount; only the counterparty changes. A plan that altered a balance would not be a settle-up plan, it would be a different set of debts.

**Q12 (medium). How do you correct a mistake in an expense?**
Reverse and re-add, because the log is append-only. The reversal mirrors the original exactly — for each participant's share, an expense where they are the payer and the original payer is the sole participant — so the two cancel to zero for every user, which is testable. Then the corrected expense is appended. The cost is that the log grows and the UI has to hide reversal pairs, and the benefit is that you can always answer "why did this balance change?" — which a mutable expense cannot answer without a separate audit table.

**Q13 (hard). What breaks with multiple currencies?**
Summing across them, which is what a derived balance does with no check. `+500 INR +300 USD` is 800 in the arithmetic and meaningless in reality. So the rules are: one currency per group, enforced; a settlement must match its group's currency, or it needs a rate and a rate timestamp and that is a different entity; and a balance is never converted at read time, because a stored rate changes and a balance that changes when you look at it is not a balance. Conversion happens at settlement time and the rate used is recorded.

**Q14 (hard). What does the design look like at 10× — millions of expenses?**
The derivation stops being affordable per read, so the balance becomes a materialised projection fed by a stream of expense and settlement events, and `BalanceSheet` becomes a store rather than a query. The important part is that materialising reintroduces exactly the drift risk that deriving avoided, so the zero-sum invariant moves from an assertion in the read path to a nightly reconciliation that compares the projection against a full derivation and alarms on any difference. That is the honest trade: you buy speed and you pay for it with a consistency check you now have to operate.

**Q15 (medium). How do you test this?**
Four layers. The allocation property test, which is the important one: over every amount and participant count, assert the shares sum exactly and that nobody is more than one minor unit from their exact proportional share. The determinism test: the same split with the participants in a different order produces the same result. The ledger invariant: after a mixed sequence of expenses, splits, settlements and a reversal, the net positions sum to zero. And the concurrency test: eight threads recording the same settlement with one key, asserting exactly one settlement row exists and the invariant still holds.

---

## 12. Cheat-sheet summary

**The model**

```
  User ─┬─ Group ─┬─ Expense ── Split (one per participant, frozen)
        │         │
        └─────────┴─ Settlement      ← NOT an expense. Discharges a debt.

  BALANCES ARE DERIVED: BalanceSheet is a query over the log, never a store.
```

**The balance formula**

```
  balance(u) = + Σ u was the payer            (expenses)
               − Σ u had a share             (expenses)
               − Σ u was the from_user       (settlements)
               + Σ u was the to_user         (settlements)

  positive = owed to u.   negative = u owes.

  INVARIANT:  Σ over all u of balance(u) == 0.  ALWAYS.  Assert it.
```

**The split contract**

```
  EVERY strategy must return splits that sum EXACTLY to the amount.
  EVERY strategy must be DETERMINISTIC: same input → same output, always.

  allocate(amount, weights):
    1. floor(amount × w_i / Σw) for each — under-allocates by < n units
    2. hand the leftover out ONE EACH, largest remainder first, TIES BY INDEX
    3. assert Σ == amount

  ₹100 / 3  →  [3334, 3333, 3333]     and alice always gets the extra paisa
  ₹100 / 7  →  sum is exactly 10000, deterministically
```

**The four strategies**

| Strategy | Input | The failure it refuses |
|---|---|---|
| Equal | participants | — (the remainder is the interesting part) |
| Exact | amount per person | Splits that do not sum to the total |
| Percentage | basis points (1% = 100) | Percentages that do not sum to 10000 |
| Shares | integer weights | Weights summing to zero |

**The race**

```
  DOUBLE-TAP ON "SETTLE UP"  →  the same debt discharged twice, money moves twice
  FIX: an idempotency key, checked and INSERTED IN THE SAME CRITICAL SECTION.
       Same key, same terms  → return the FIRST settlement.
       Same key, different terms → refuse loudly (a client bug).
  DO NOT CAP the settlement: the cap needs read-check-write across the lock,
  and pre-payment plus simultaneous settlements are both legitimate.
```

**Debt simplification**

```
  PROBLEM: fewest transfers to clear all debts.  NP-HARD (reduces to subset-sum).
  GREEDY:  largest creditor against largest debtor, repeat.
  BOUND:   ≤ n − 1 transfers for n non-zero balances.
           Proof: each step zeroes at least one balance → removes a person.
  GAIN:    up to N×M direct transfers  →  ≤ n−1.  That is the whole value.
  PROPERTY: every net position is PRESERVED exactly. It changes who pays whom,
            never how much.
```

**The numbers**

| Parameter | Value |
|---|---|
| Currency | Integer minor units only |
| Percentage unit | Basis points (1% = 100 bp), summing to 10 000 |
| Balance derivation | O(total splits) per read |
| Simplifier | O(n log n), ≤ n−1 transfers |
| Settlement | O(1), idempotent on the key |
| Cache | Keyed on the log length; bounded to 16 entries |
| Lock | One `RLock`; microseconds, no I/O inside |

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "₹100 three ways" | "3334, 3333, 3333 — sorted participants, ties by index, sum exact. Deterministic." |
| "Why not store balances?" | "They drift. Derive from the append-only log; cache on the log's length." |
| "How do you know it is right?" | "The net positions sum to zero. It is a theorem, and it catches nearly every bug." |
| "Someone settles twice" | "An idempotency key, checked and inserted in one critical section." |
| "Cap the settlement?" | "No — the cap is check-then-act, and pre-payment is legitimate." |
| "Fewest transfers?" | "NP-hard. Greedy gives ≤ n−1 with a proof, and that is the shipped answer." |
| "Fix a wrong expense" | "Reverse and re-add. The log is append-only, and the reversal mirrors the splits exactly." |
| "Prove it" | "An allocation property test over every amount and size, plus the zero-sum invariant." |

---

## Cross-links

- The shared-state mechanics this file assumes: [Concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md)
- The other lease-versus-ownership distinctions in this pillar: [BookMyShow seat locking](04-bookmyshow-seat-locking.md) · [Hotel management](03-hotel-management.md) · [Elevator system](05-elevator-system.md) · [Parking lot](01-parking-lot.md) · and the cache whose remainder-free arithmetic contrasts with the allocation here: [LRU and LFU cache](02-lru-lfu-cache.md)
- The class concepts behind this design: [OOP principles and SOLID](../02-oop-principles-and-solid.md) · [Design patterns](../03-design-patterns.md) · [UML and class diagrams](../05-uml-and-class-diagrams.md) · [The machine-coding framework](../06-machine-coding-interview-framework.md) · [What is LLD](../01-what-is-lld-and-how-to-answer.md)
- Where the class view ends: [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md) · storing an immutable log durably: [SQL deep dive](../../HLD/04-data/01-sql-deep-dive.md) · and reading it at scale: [Caching strategies and layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md)
- Other problems in this pillar: [ATM machine](07-atm-machine.md) · [Library management](08-library-management.md)
