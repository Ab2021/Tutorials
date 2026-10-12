# LLD Problem: Library Management System

> **Grounding note.** The source transcript does not mention this problem. It is not in the LLD question list at **[9:22]**, and it is not part of the parking-lot walkthrough at **[25:54]–[27:38]**. **Everything below is standard industry practice.** It is in this pillar because it is the problem where **the modelling distinction is the whole answer**: a `Book` is a bibliographic record — "Clean Code, 1st edition, by Robert Martin" — and a `BookCopy` is a physical object with an accession number, a shelf, and a condition. Almost every candidate models one class called `Book` with an `is_available` flag, and that model cannot answer "how many copies do we have?", cannot represent two copies of the same title with different conditions, cannot attach a loan to *the copy in your hand* rather than to the title, and cannot survive a lost copy. **Getting that split right is worth more than every algorithm in the file.** Beyond it, two smaller decisions carry real weight: the due date is a **date in library-local terms computed over open days**, not a wall-clock timestamp plus a duration, and an overdue fine is **assessed once at return and frozen**, not continuously recomputed. The shared-state mechanics are in [concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md); this file spends its budget on the domain.

---

## 1. Requirement clarification

### 1.1 Questions to ask

| # | Question | Why it matters | Default if they will not say |
|---|---|---|---|
| 1 | **Does the library hold multiple copies of a title?** | **The question that decides the whole model** | Yes — `Book` and `BookCopy` are separate (§2) |
| 2 | **How long is a loan, and does it depend on who borrows?** | Whether there is a policy strategy | Yes: student 14 days, faculty 30, guest 7 (§5.3) |
| 3 | **Is the due date counted in calendar days or open days?** | A library closed for a week should not fine everyone for that week | Open days, computed from a calendar (§5.4) |
| 4 | **Fines: per day, with a grace period and a cap?** | The money rules are the ones people argue about | Grace 2 open days, ₹5/open day, capped at the book's price |
| 5 | **Can a loan be renewed?** | And the interaction: not if someone is waiting | Yes, up to 2 renewals, refused if a reservation is waiting (§7.1) |
| 6 | **Reservations on a title whose copies are all out?** | Whether there is a queue at all | Yes, FIFO, with a TTL once the copy is ready (§5.6) |
| 7 | **What happens when a copy is lost?** | Whether a copy can leave the catalog | Charged the replacement cost; the copy moves to `LOST`, never deleted (§7.4) |
| 8 | Can a member borrow two copies of the same title? | A rule that is easy to forget and produces a silly outcome | No — at most one copy of a given title per member (§5.5) |
| 9 | Is the availability count stored or derived? | **The performance decision** | Derived is the truth; a denormalized counter is a reconciled cache (§2) |
| 10 | Who is notified when a reserved book comes back? | Whether there is a notification path | A notification port; the class design does not own delivery (§7.5) |

### 1.2 Defaults to state out loud

```
  MODEL:      Book (bibliographic) ──1:N── BookCopy (physical)
                  │                          │
                  │                          └── 1:1 at most ── Loan ── Member
                  └── Reservation queue (per TITLE, not per copy)

  AVAILABLE:  a count of COPIES, derived from copy statuses. "Is this book
              available?" is the wrong question — "how many copies are on the
              shelf?" is the right one.
  DATES:      due dates are DATES, counted in OPEN days from a calendar, not
              timestamps plus a duration (§5.4).
  FINES:      assessed ONCE at return, then FROZEN. A fine that changes when
              you look at it is not a fine.
  OVERDUE:    DERIVED from due_on < today. Not a stored state (§5.5).
  THREADING:  per-BOOK lock. The race is two members borrowing the last copy,
              and a return racing a reservation.
  OUT:        the notification transport, the barcode scanner, the catalogue
              search index, the acquisitions budget, the fine payment gateway.
```

**The one sentence that frames the problem:** *"A library does not lend books, it lends copies — and every design that starts with one `Book` class with an `is_available` flag has to be rewritten the moment there are two copies of anything."*

---

## 2. Core entities & responsibilities

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `Book` | The bibliographic record | isbn, title, author, subject, replacement cost | Know how many copies exist |
| `BookCopy` | **One physical object** | accession number, book, status, shelf, condition | Know who has it |
| `CopyStatus` *(enum)* | What a copy's situation is right now | — | — |
| `Member` | A borrower | member id, name, type, status | Hold a fine balance directly |
| `Loan` | One copy out to one member | copy, member, borrowed_on, due_on, returned_on, renewals | Decide the rules |
| `LoanPolicy` *(interface)* + three | The rules per member type | loan days, max loans, max renewals, fine rate, grace | — |
| `Reservation` | One member's claim on a **title** | book, member, placed_at, state, expires_at | Name a copy |
| `ReservationQueue` | The FIFO per title | the waiting reservations | Allocate copies |
| `FineAssessment` | One frozen fine on one loan | loan, amount, assessed_on, paid | Recompute itself |
| `LibraryCalendar` | Which days the library is open | closed dates | — |
| `LoanLedger` | The append-only record of loans and fines | loans, assessments, payments | — |
| `Catalog` | Books and copies | both collections | Lend |
| `LendingService` | Borrow, return, renew — the write path | the locks | Compute balances |
| `Clock` *(interface)* | `today()` and `now()` | — | — |

**The design decision, stated plainly: `Book` and `BookCopy` are separate entities, and the loan attaches to the copy.** The consequences run through everything:

- **Availability is a count of copies**, not a flag on a book. A library with three copies and one on loan has two available, and only a per-copy model can say so.
- **Two copies of the same title can differ.** One is a reference copy that cannot leave the building; one is water-damaged; one is a large-print edition. A single `Book` class cannot hold that.
- **A loan is of an object, not a concept.** When a member returns a book, the librarian scans the accession number — the loan closes against *that copy*. With a single-class model there is nothing to attach the return to, and the system cannot tell which of the three copies came back.
- **A lost copy is a copy, not a title.** Losing one of three copies does not make the title unavailable. The copy moves to `LOST`, is excluded from the available count, and stays in the history so the closed loans that reference it still resolve.

**The second decision, and it is a genuine tension rather than a clean answer: availability is derived, but the catalogue wants it fast.** Search results show "3 of 5 available", and that must render in milliseconds for a page of results. So:

- The **truth is derived** — a count over `BookCopy.status`.
- A **denormalized `available_count` on `Book` is a cache**, and it must be maintained inside the same critical section that changes a copy's status, never recomputed later by a job.
- **`check_invariants` reconciles it.** The counter is allowed to exist; it is not allowed to disagree.

**This is a case where the "derive everything" purism from [Splitwise](06-splitwise.md) has to bend**, and saying why is better than applying the rule mechanically: there, a balance was read once per screen; here, an availability count is read once per search result and there may be a hundred results per page.

---

## 3. Class diagram

```
        ┌─────────────────────────┐
        │          Book           │        (bibliographic record)
        ├─────────────────────────┤
        │ isbn: str               │
        │ title: str              │
        │ author: str             │
        │ replacement_minor: int  │
        │ available_count: int    │ ← CACHE, reconciled (§2)
        └───────────┬─────────────┘
                    │ 1
                    │ N
        ┌───────────▼─────────────┐         ┌────────────────────┐
        │        BookCopy         │         │   «enum»           │
        ├─────────────────────────┤         │   CopyStatus       │
        │ accession: str      ────┼────────▶├────────────────────┤
        │ book: Book              │         │ AVAILABLE          │
        │ status: CopyStatus      │         │ ON_LOAN            │
        │ shelf: str              │         │ ON_HOLD_SHELF      │
        │ condition: str          │         │ IN_REPAIR          │
        └───────────┬─────────────┘         │ LOST               │
                    │ 0..1 active loan      │ WITHDRAWN          │
                    │                       └────────────────────┘
        ┌───────────▼─────────────┐         ┌────────────────────┐
        │         Loan            │         │      Member        │
        ├─────────────────────────┤         ├────────────────────┤
        │ loan_id: str            │        *│ member_id: str     │
        │ copy: BookCopy          │─────────│ name: str          │
        │ member: Member          │    1    │ policy: LoanPolicy │
        │ borrowed_on: date       │         │ status: MemberStatus│
        │ due_on: date            │         └────────────────────┘
        │ returned_on: date|None  │
        │ renewals: int           │         ┌────────────────────┐
        │ state: LoanState        │         │ «interface»        │
        ├─────────────────────────┤         │   LoanPolicy       │
        │ + is_overdue(today)     │         ├────────────────────┤
        │ + open_days_late(...)   │         │ loan_days          │
        └─────────────────────────┘         │ max_loans          │
                                            │ max_renewals       │
        ┌─────────────────────────┐         │ fine_per_day_minor │
        │       Reservation       │         │ grace_open_days    │
        ├─────────────────────────┤         │ + may_borrow(...)  │
        │ reservation_id: str     │         └─────────▲──────────┘
        │ book: Book              │                   │
        │ member: Member          │      ┌────────────┼────────────┐
        │ placed_at: float        │ ┌────┴─────┐┌─────┴────┐┌──────┴─────┐
        │ state: ReservationState │ │ Student  ││ Faculty  ││   Guest    │
        │ expires_at: float|None  │ └──────────┘└──────────┘└────────────┘
        └─────────────────────────┘
                    │
        ┌───────────▼─────────────┐
        │    ReservationQueue     │   one per Book, FIFO
        ├─────────────────────────┤
        │ + enqueue(reservation)  │
        │ + peek() -> Reservation │
        │ + advance()             │
        └─────────────────────────┘

        ┌─────────────────────────┐   ┌──────────────────────────────┐
        │     FineAssessment      │   │       LendingService         │
        ├─────────────────────────┤   ├──────────────────────────────┤
        │ loan: Loan              │   │ - catalog: Catalog           │
        │ amount_minor: int       │   │ - calendar: LibraryCalendar  │
        │ assessed_on: date       │   │ - clock: Clock               │
        │ paid: bool              │   │ - locks: per-book            │
        └─────────────────────────┘   │ + borrow(member, isbn)       │
                                      │ + return(accession)          │
                                      │ + renew(loan_id)             │
                                      │ + reserve(member, isbn)      │
                                      │ + sweep_expired_holds()      │
                                      └──────────────────────────────┘
```

**What the diagram states:**

- `BookCopy` has **at most one active loan** (`0..1`), which is the invariant that makes the whole model work: a copy is either on the shelf, on loan, on the hold shelf, or out of circulation — never two of those.
- `Reservation` points at a **`Book`**, not a `BookCopy`. **This is the subtle part:** a member reserves a *title* ("I want Clean Code"), and which physical copy they get is decided at collection time, exactly as a hotel booking is against a room *type* rather than a room number. **The cross-link is genuine and worth making in the interview.**
- `Member.policy` is a `LoanPolicy`, so the rules per member type are data attached to the member rather than a `switch` in the lending code.
- `available_count` is on `Book` and is **marked as a cache in the diagram itself**, which is where a reader will notice it.
- `Loan` holds `borrowed_on`/`due_on`/`returned_on` as **dates**, not floats. That is deliberate and §5.4 argues it.

---

## 4. Interfaces and enums

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class LibraryError(Exception):
    """A refused operation. A library refusing a loan is normal — the member is
    at their limit, the copy is on loan, they have unpaid fines — so these are
    typed and explainable, not generic failures."""


class NoCopyAvailableError(LibraryError):
    pass


class MemberBlockedError(LibraryError):
    pass


class LoanLimitError(LibraryError):
    pass


class NotRenewableError(LibraryError):
    pass


class LoanNotFoundError(LibraryError):
    pass


class CopyStatus(Enum):
    """Where a physical copy is. Note the states are mutually exclusive, which
    is what makes 'at most one active loan' enforceable rather than aspirational.

    ON_HOLD_SHELF is the state that a returned copy enters when a reservation is
    waiting for it: it is physically on the shelf, and it is NOT available to
    anyone else. Skipping this state — going straight to AVAILABLE and notifying
    the member — is the bug where the reserved copy is borrowed by a walk-in
    before the member arrives."""

    AVAILABLE = "available"
    ON_LOAN = "on_loan"
    ON_HOLD_SHELF = "on_hold_shelf"
    IN_REPAIR = "in_repair"
    LOST = "lost"
    WITHDRAWN = "withdrawn"

    def counts_as_available(self) -> bool:
        return self is CopyStatus.AVAILABLE


class LoanState(Enum):
    """Note what is NOT here: OVERDUE.

    Deliberately absent, and it is the same argument as `TransactionOutcome` in
    the ATM file: an overdue loan is DERIVED from `due_on < today`, not a stored
    state. Storing it means a nightly job must flip every loan on the right day,
    and any bug in that job — a missed run, a wrong timezone, a leap day —
    leaves loans in a state that is simply false. A derived `is_overdue()` cannot
    be wrong because there is nothing to keep in sync."""

    ACTIVE = "active"
    RETURNED = "returned"
    LOST = "lost"


class ReservationState(Enum):
    """WAITING → READY → FULFILLED, or WAITING/READY → EXPIRED/CANCELLED.

    READY means a copy is physically on the hold shelf for this member and the
    clock is running. A READY reservation that is not collected expires and the
    copy goes to the next person in the queue — the lease-with-a-TTL shape from
    the seat-holding design, applied to a shelf."""

    WAITING = "waiting"
    READY = "ready"
    FULFILLED = "fulfilled"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    def is_open(self) -> bool:
        return self in (ReservationState.WAITING, ReservationState.READY)


class MemberStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"      # unpaid fines past a threshold, or misconduct
    CLOSED = "closed"


class Clock(ABC):
    """TWO methods, and the split is the point — the same split as in the hotel
    design, and collapsing it is the same mistake.

      today()  the LIBRARY'S calendar date. Used for due dates, for overdue
               arithmetic, and for fines. Local, and NOT derived from a UTC
               timestamp, because a library in one city on one calendar needs
               one date and "what time is it in UTC" is a different question.
      now()    a monotonic instant. Used for the hold-shelf TTL and for
               reservation ordering, where a duration is what matters and the
               calendar is irrelevant.

    Collapsing them into one `now()` that returns a datetime is how a system
    ends up with due dates that shift by a day for members in a different
    timezone, and with hold TTLs that break when the clocks go forward."""

    @abstractmethod
    def today(self) -> date: ...

    @abstractmethod
    def now(self) -> float: ...


class LibraryCalendar:
    """Which days the library is open.

    THE REASON THIS EXISTS, and it is the decision most candidates miss: a loan
    period of "14 days" means fourteen OPEN days. A library closed for a
    week-long holiday and charging every member a week of fines for it is a
    customer-service disaster and an obvious defect. So the due date is computed
    by walking forward fourteen open days, and the overtime is counted in open
    days too.

    A real calendar also has half-days, exam-period extended hours and
    branch-specific closures. The model here is the simplest thing that is
    correct: a set of closed dates, one per branch. Everything richer is a
    change to `is_open` and nothing else."""

    def __init__(self, closed_dates: "set[date] | None" = None,
                 weekend_closed: bool = False) -> None:
        self._closed = set(closed_dates or ())
        self._weekend_closed = weekend_closed

    def is_open(self, day: date) -> bool:
        if day in self._closed:
            return False
        if self._weekend_closed and day.weekday() >= 5:      # Sat=5, Sun=6
            return False
        return True

    def add_open_days(self, start: date, count: int) -> date:
        """The `count`-th open day strictly after `start`.

        A loan taken on an open day with a 14-open-day term is due on the 14th
        open day after it — so if the library is closed the next day, the due
        date moves out by one. Walking rather than adding a timedelta is the
        whole point of this class."""
        if count < 0:
            raise LibraryError("an open-day count cannot be negative")
        day = start
        remaining = count
        while remaining > 0:
            day += timedelta(days=1)
            if self.is_open(day):
                remaining -= 1
        return day

    def open_days_between(self, start: date, end: date) -> int:
        """Open days in the half-open interval [start, end).

        HALF-OPEN, and the same decision as the hotel's date ranges: a copy due
        on the 10th and returned on the 10th is NOT late, so the interval that
        matters is [due_on, returned_on) and its length is the number of days
        charged. Using a closed interval charges everyone who returns on time."""
        if end < start:
            raise LibraryError("cannot count open days backwards")
        count = 0
        day = start
        while day < end:
            if self.is_open(day):
                count += 1
            day += timedelta(days=1)
        return count

    def next_open_day(self, start: date) -> date:
        return self.add_open_days(start, 1)


class LoanPolicy(ABC):
    """The rules for one class of member. Everything that varies between a
    student and a professor is here, which is what keeps the lending code free
    of `if member.type == 'faculty'`."""

    name: str

    def __init__(self, loan_days: int, max_loans: int, max_renewals: int,
                 fine_per_open_day_minor: int, grace_open_days: int) -> None:
        self.loan_days = loan_days
        self.max_loans = max_loans
        self.max_renewals = max_renewals
        self.fine_per_open_day_minor = fine_per_open_day_minor
        self.grace_open_days = grace_open_days

    def may_borrow(self, member: "Member", active_loans: int,
                   unpaid_fine_minor: int) -> None:
        """Raise if this member may not borrow right now. A no-op on success,
        which is the shape that makes the rules composable."""
        if member.status is not MemberStatus.ACTIVE:
            raise MemberBlockedError(
                f"member {member.member_id} is {member.status.value}"
            )
        if active_loans >= self.max_loans:
            raise LoanLimitError(
                f"{self.name} members may hold at most {self.max_loans} loans; "
                f"this member holds {active_loans}"
            )

    @abstractmethod
    def fine_cap_minor(self, copy: "BookCopy") -> int:
        """The most this member can be fined for one overdue copy. Faculty
        policies usually cap at the replacement cost; a guest policy may cap
        lower so a tourist is never billed for a book they cannot return."""


class StudentPolicy(LoanPolicy):
    name = "student"

    def __init__(self) -> None:
        super().__init__(loan_days=14, max_loans=4, max_renewals=2,
                         fine_per_open_day_minor=500, grace_open_days=2)

    def fine_cap_minor(self, copy: "BookCopy") -> int:
        return copy.book.replacement_minor


class FacultyPolicy(LoanPolicy):
    name = "faculty"

    def __init__(self) -> None:
        super().__init__(loan_days=30, max_loans=10, max_renewals=5,
                         fine_per_open_day_minor=200, grace_open_days=7)

    def fine_cap_minor(self, copy: "BookCopy") -> int:
        return copy.book.replacement_minor


class GuestPolicy(LoanPolicy):
    name = "guest"

    def __init__(self) -> None:
        super().__init__(loan_days=7, max_loans=2, max_renewals=0,
                         fine_per_open_day_minor=1_000, grace_open_days=0)

    def fine_cap_minor(self, copy: "BookCopy") -> int:
        # A guest is capped low on purpose: they may leave the city, and a fine
        # they cannot pay and a book they cannot return is a worse outcome than
        # writing the copy off.
        return min(copy.book.replacement_minor, 20_000)
```

**`LibraryCalendar` and the `LoanPolicy` split are the two things a sketch omits.** A sketch computes `due_on = borrowed_on + timedelta(days=14)` and puts `if member.is_faculty` in the lending method. The first is wrong every time the library closes, and the second is wrong the moment a fourth member type arrives.

---

## 5. Complete working code

**Language note:** Python 3.11+, standard library only. Every block is complete.

### 5.1 Money and the catalogue

```python
from datetime import datetime, timezone


class Money:
    """Integer minor units, as everywhere in this pillar. A fine is money and a
    float fine is a float balance in a ledger, which is the same mistake."""

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
            raise LibraryError(f"currency mismatch: {self.currency} vs {other.currency}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return (self.minor, self.currency) == (other.minor, other.currency)

    def __repr__(self) -> str:
        return f"{self.currency} {self.minor // 100}.{self.minor % 100:02d}"


class Book:
    """The BIBLIOGRAPHIC record. There is one of these per title, and it is
    shared by every physical copy.

    `available_count` is a CACHE. The truth is a count over the copies' statuses
    and this field must agree with it — `check_invariants` is what proves that,
    and the reason the cache exists at all is that catalogue search renders
    availability for a hundred results per page."""

    __slots__ = ("isbn", "title", "author", "replacement_minor", "available_count")

    def __init__(self, isbn: str, title: str, author: str,
                 replacement_minor: int) -> None:
        if not isbn:
            raise LibraryError("a book needs an ISBN")
        if replacement_minor <= 0:
            raise LibraryError("a replacement cost must be positive")
        self.isbn = isbn
        self.title = title
        self.author = author
        self.replacement_minor = replacement_minor
        self.available_count = 0

    def __repr__(self) -> str:
        return f"Book({self.isbn}, {self.title!r} by {self.author})"


class BookCopy:
    """ONE PHYSICAL OBJECT. The accession number is its identity, and it is what
    the librarian scans.

    Note what this class does NOT have: a `borrower` field. The borrower lives
    on the `Loan`, because a copy needs to know whether it is on loan, not who
    has it — asking "who has this copy?" is a lookup on the loans, and putting
    the member here would mean two places know the same fact and can disagree."""

    __slots__ = ("accession", "book", "status", "shelf", "condition")

    def __init__(self, accession: str, book: Book, shelf: str = "A-01",
                 condition: str = "good") -> None:
        if not accession:
            raise LibraryError("a copy needs an accession number")
        self.accession = accession
        self.book = book
        self.status = CopyStatus.AVAILABLE
        self.shelf = shelf
        self.condition = condition

    def is_borrowable(self) -> bool:
        """Available to a walk-in borrower. Note ON_HOLD_SHELF is excluded: the
        copy is physically present and it is promised to someone else."""
        return self.status is CopyStatus.AVAILABLE

    def is_in_circulation(self) -> bool:
        """Whether it can ever be lent again. A lost or withdrawn copy cannot."""
        return self.status not in (CopyStatus.LOST, CopyStatus.WITHDRAWN)

    def __repr__(self) -> str:
        return f"BookCopy({self.accession}, {self.book.isbn}, {self.status.value})"


class Catalog:
    """Books and copies. Note `availability` computes from the copies — the
    derived truth — and `sync_available_count` writes the cache."""

    def __init__(self) -> None:
        self._books: dict[str, Book] = {}                 # isbn → Book
        self._copies: dict[str, BookCopy] = {}            # accession → BookCopy
        self._copies_of: dict[str, list[str]] = {}        # isbn → [accession]

    def add_book(self, book: Book) -> None:
        if book.isbn in self._books:
            raise LibraryError(f"{book.isbn} is already catalogued")
        self._books[book.isbn] = book
        self._copies_of[book.isbn] = []

    def add_copy(self, copy: BookCopy) -> None:
        if copy.accession in self._copies:
            raise LibraryError(f"accession {copy.accession} already exists")
        if copy.book.isbn not in self._books:
            raise LibraryError(f"{copy.book.isbn} is not catalogued")
        self._copies[copy.accession] = copy
        self._copies_of[copy.book.isbn].append(copy.accession)
        self.sync_available_count(copy.book.isbn)

    def book(self, isbn: str) -> Book:
        book = self._books.get(isbn)
        if book is None:
            raise LibraryError(f"unknown ISBN {isbn}")
        return book

    def copy(self, accession: str) -> BookCopy:
        copy = self._copies.get(accession)
        if copy is None:
            raise LibraryError(f"unknown accession {accession}")
        return copy

    def copies_of(self, isbn: str) -> "list[BookCopy]":
        return [self._copies[a] for a in self._copies_of.get(isbn, [])]

    def availability(self, isbn: str) -> int:
        """THE DERIVED TRUTH. Walks the copies every time."""
        return sum(1 for c in self.copies_of(isbn) if c.is_borrowable())

    def total_copies(self, isbn: str) -> int:
        return len(self._copies_of.get(isbn, []))

    def sync_available_count(self, isbn: str) -> int:
        """Write the cache. Call it inside the same critical section that changed
        a copy's status — never from a background job, because a job that runs
        later is a window in which the cache is wrong."""
        count = self.availability(isbn)
        self._books[isbn].available_count = count
        return count

    def find_borrowable_copy(self, isbn: str) -> "BookCopy | None":
        """Any available copy. The choice is deliberately unspecified beyond
        'available' — a real library prefers the copy already on the shelf the
        member is standing at, and a copy in better condition. Those are
        tie-breaks on this list, not a different algorithm."""
        for accession in self._copies_of.get(isbn, []):
            copy = self._copies[accession]
            if copy.is_borrowable():
                return copy
        return None

    def check_invariants(self) -> None:
        """The cache against the truth, for every title. Cheap, and it catches
        the one bug this cache can have."""
        for isbn, book in self._books.items():
            truth = self.availability(isbn)
            if book.available_count != truth:
                raise AssertionError(
                    f"{isbn}: cached available_count is {book.available_count} "
                    f"but {truth} copies are actually available"
                )
```

### 5.2 Members and the loan ledger

```python
class Member:
    __slots__ = ("member_id", "name", "policy", "status", "joined_on")

    def __init__(self, member_id: str, name: str, policy: LoanPolicy,
                 joined_on: date) -> None:
        self.member_id = member_id
        self.name = name
        self.policy = policy
        self.status = MemberStatus.ACTIVE
        self.joined_on = joined_on

    def __repr__(self) -> str:
        return f"Member({self.member_id}, {self.name!r}, {self.policy.name})"


class Loan:
    """ONE COPY OUT TO ONE MEMBER.

    The dates are `date`s, not instants, and §5.4 argues why. `due_on` is
    computed at borrow time from the calendar's open days — it is NOT
    `borrowed_on + timedelta(loan_days)`.

    `is_overdue` is derived rather than stored, which is why `LoanState` has no
    OVERDUE member: a stored flag needs a job to flip it and every bug in that
    job leaves a loan lying about its own status."""

    __slots__ = ("loan_id", "copy", "member", "borrowed_on", "due_on",
                 "returned_on", "renewals", "state")

    def __init__(self, loan_id: str, copy: BookCopy, member: Member,
                 borrowed_on: date, due_on: date) -> None:
        if due_on < borrowed_on:
            raise LibraryError("a due date cannot precede the loan")
        self.loan_id = loan_id
        self.copy = copy
        self.member = member
        self.borrowed_on = borrowed_on
        self.due_on = due_on
        self.returned_on: "date | None" = None
        self.renewals = 0
        self.state = LoanState.ACTIVE

    def is_overdue(self, today: date) -> bool:
        """DERIVED. An active loan is overdue when today is strictly past the
        due date — so a copy returned on its due date is never overdue."""
        return self.state is LoanState.ACTIVE and today > self.due_on

    def chargeable_open_days(self, as_of: date, calendar: LibraryCalendar,
                             grace_open_days: int) -> int:
        """Open days late, minus the grace period, floored at zero.

        THE HALF-OPEN INTERVAL IS THE WHOLE CORRECTNESS ARGUMENT HERE: the
        interval counted is [due_on, as_of), so a copy due on the 10th and
        returned on the 10th charges zero days. A closed interval would charge
        every member who returned on time, which is the single most common bug
        in a library system and also the most infuriating one."""
        if as_of <= self.due_on:
            return 0
        late = calendar.open_days_between(self.due_on, as_of)
        return max(0, late - grace_open_days)

    def __repr__(self) -> str:
        return (f"Loan({self.loan_id}, {self.copy.accession}, "
                f"{self.member.member_id}, due {self.due_on}, {self.state.value})")


class FineAssessment:
    """ONE FROZEN FINE.

    ASSESSED ONCE, AT RETURN, AND NEVER RECOMPUTED. This is the decision worth
    arguing in the interview: a fine computed on demand from `due_on` and today
    changes every day it goes unpaid, which means the member sees one number,
    the statement shows another, and an appeal has no figure to appeal against.
    Freezing it at the moment of return makes the fine an immutable fact — the
    same argument as an immutable expense in [Splitwise](06-splitwise.md), and
    it is not a coincidence that both are money.

    `assessed_on` records the date the assessment was made, which is the return
    date. It is not the date it was paid, and it is not a timestamp."""

    __slots__ = ("assessment_id", "loan", "amount_minor", "assessed_on",
                 "open_days_charged", "paid_on")

    def __init__(self, assessment_id: str, loan: Loan, amount_minor: int,
                 assessed_on: date, open_days_charged: int) -> None:
        if amount_minor < 0:
            raise LibraryError("a fine cannot be negative")
        self.assessment_id = assessment_id
        self.loan = loan
        self.amount_minor = amount_minor
        self.assessed_on = assessed_on
        self.open_days_charged = open_days_charged
        self.paid_on: "date | None" = None

    def is_paid(self) -> bool:
        return self.paid_on is not None

    def __repr__(self) -> str:
        state = "paid" if self.is_paid() else "unpaid"
        return (f"FineAssessment({self.assessment_id}, "
                f"{Money(self.amount_minor)}, {self.open_days_charged} open days, "
                f"{state})")


class LoanLedger:
    """The append-only record: loans, fine assessments, payments.

    A member's outstanding balance is DERIVED from the unpaid assessments minus
    the payments — the same shape as the Splitwise balance sheet, and for the
    same reason. A mutable `member.outstanding_fine` field drifts; a query over
    an append-only ledger cannot."""

    def __init__(self) -> None:
        self.loans: list[Loan] = []
        self.assessments: list[FineAssessment] = []
        self.payments: list[tuple[str, date, int]] = []     # (member_id, on, minor)

    def record_loan(self, loan: Loan) -> None:
        self.loans.append(loan)

    def record_assessment(self, assessment: FineAssessment) -> None:
        self.assessments.append(assessment)

    def record_payment(self, member_id: str, on: date, amount_minor: int) -> None:
        if amount_minor <= 0:
            raise LibraryError("a payment must be positive")
        self.payments.append((member_id, on, amount_minor))

    def active_loans_of(self, member_id: str) -> "list[Loan]":
        return [loan for loan in self.loans
                if loan.member.member_id == member_id
                and loan.state is LoanState.ACTIVE]

    def active_loan_of_copy(self, accession: str) -> "Loan | None":
        for loan in self.loans:
            if loan.copy.accession == accession and loan.state is LoanState.ACTIVE:
                return loan
        return None

    def title_on_loan_to(self, member_id: str, isbn: str) -> bool:
        """The 'one copy of a title per member' rule, checked as a query."""
        return any(
            loan.member.member_id == member_id
            and loan.copy.book.isbn == isbn
            and loan.state is LoanState.ACTIVE
            for loan in self.loans
        )

    def unpaid_total(self, member_id: str) -> int:
        owed = sum(a.amount_minor for a in self.assessments
                   if a.loan.member.member_id == member_id and not a.is_paid())
        paid = sum(amount for mid, _, amount in self.payments if mid == member_id)
        # A member may pay in advance; the balance floors at zero for the
        # purposes of blocking, and the credit is retained.
        return max(0, owed - paid)

    def credit_of(self, member_id: str) -> int:
        owed = sum(a.amount_minor for a in self.assessments
                   if a.loan.member.member_id == member_id and not a.is_paid())
        paid = sum(amount for mid, _, amount in self.payments if mid == member_id)
        return max(0, paid - owed)

    def check_invariants(self) -> None:
        """Every active loan's copy must be ON_LOAN, and every copy that is
        ON_LOAN must have exactly one active loan. This is the invariant that
        makes the copy-status model trustworthy."""
        for loan in self.loans:
            if loan.state is LoanState.ACTIVE:
                if loan.copy.status is not CopyStatus.ON_LOAN:
                    raise AssertionError(
                        f"active loan {loan.loan_id} but copy "
                        f"{loan.copy.accession} is {loan.copy.status.value}"
                    )
        for loan in self.loans:
            if loan.state is LoanState.ACTIVE:
                same = [other for other in self.loans
                        if other.copy.accession == loan.copy.accession
                        and other.state is LoanState.ACTIVE]
                if len(same) != 1:
                    raise AssertionError(
                        f"copy {loan.copy.accession} has {len(same)} active loans"
                    )
```

### 5.3 The reservation queue

```python
class Reservation:
    """A member's claim on a TITLE. Note `book`, not `copy` — which physical
    copy they collect is decided when one comes back, exactly as a hotel
    reservation is against a room type and the room number is assigned at
    check-in. Trying to reserve a specific copy means the reservation has to be
    re-pointed every time the wrong copy comes back first."""

    __slots__ = ("reservation_id", "book", "member", "placed_at", "state",
                 "expires_at", "ready_accession")

    def __init__(self, reservation_id: str, book: Book, member: Member,
                 placed_at: float) -> None:
        self.reservation_id = reservation_id
        self.book = book
        self.member = member
        self.placed_at = placed_at
        self.state = ReservationState.WAITING
        self.expires_at: "float | None" = None
        self.ready_accession: "str | None" = None

    def is_expired(self, now: float) -> bool:
        """DERIVED, and only once the copy is actually on the shelf. A WAITING
        reservation has no TTL — the hold clock starts when the library can
        actually hand something over, not when the member asks. Expiring a
        waiting reservation would punish a member for the library being slow,
        which is the opposite of what the queue is for."""
        return (self.state is ReservationState.READY
                and self.expires_at is not None
                and now >= self.expires_at)

    def __repr__(self) -> str:
        return (f"Reservation({self.reservation_id}, {self.book.isbn}, "
                f"{self.member.member_id}, {self.state.value})")


class ReservationQueue:
    """One per title, FIFO.

    FIFO rather than any cleverness, and the reason is fairness rather than
    simplicity: a member who placed a reservation first can see their position
    in the app, and a queue that reorders — for a faculty member, say — makes
    that position a lie. Priority borrowing is a real policy at real libraries
    and it is implemented as a separate queue or an explicit priority field, not
    as an implicit reordering."""

    def __init__(self, book: Book) -> None:
        self.book = book
        self._queue: list[Reservation] = []

    def enqueue(self, reservation: Reservation) -> int:
        """Returns the 1-based position, computed AFTER the insert so the member
        is told the truth about where they are."""
        if reservation.book.isbn != self.book.isbn:
            raise LibraryError("a reservation is being queued on the wrong title")
        if any(r.member.member_id == reservation.member.member_id
               and r.state.is_open() for r in self._queue):
            raise LibraryError(
                f"{reservation.member.member_id} already holds a reservation "
                f"on {self.book.isbn}"
            )
        self._queue.append(reservation)
        return self.position_of(reservation)

    def position_of(self, reservation: Reservation) -> int:
        open_ones = [r for r in self._queue if r.state.is_open()]
        for index, candidate in enumerate(open_ones, start=1):
            if candidate.reservation_id == reservation.reservation_id:
                return index
        raise LibraryError(f"{reservation.reservation_id} is not in this queue")

    def peek(self) -> "Reservation | None":
        """The next member who should get a copy: the earliest WAITING one.
        A READY reservation is skipped because that member already has a copy
        on the hold shelf — handing them a second would be a duplicate."""
        for reservation in self._queue:
            if reservation.state is ReservationState.WAITING:
                return reservation
        return None

    def waiting_count(self) -> int:
        return sum(1 for r in self._queue if r.state is ReservationState.WAITING)

    def has_waiting(self) -> bool:
        return self.peek() is not None

    def open_reservations(self) -> "list[Reservation]":
        return [r for r in self._queue if r.state.is_open()]

    def snapshot(self) -> "list[Reservation]":
        """For the caller to report positions without holding the lock."""
        return list(self._queue)
```

**`peek()` skipping `READY` reservations is a small detail with a real consequence.** If it returned the first open reservation regardless of state, a member whose copy is already on the hold shelf would be handed the next returned copy as well, and the member behind them would wait forever. **A queue whose head can block on a member who has already been served is a starvation bug, and it is worth naming.**

### 5.4 The lending service

```python
import threading


class LendingService:
    """The write path. Borrow, return, renew, reserve, sweep.

    LOCK GRANULARITY: ONE LOCK PER BOOK (isbn). Why not one global lock: the
    whole point of a library is that borrowing is concurrent — a hundred members
    at the desk at once, and a global lock serialises all of them behind every
    catalogue write. Why not one lock per copy: because the rules that matter
    are per TITLE, not per copy — the "one copy per member per title" check, the
    reservation queue, and the choice of which copy to hand over all span every
    copy of a title, and a per-copy lock cannot hold that set atomically. The
    result would be two members each taking a different copy of the same title
    and both passing a per-copy check.

    This is the same reasoning as the per-show lock in
    [BookMyShow](04-bookmyshow-seat-locking.md) and the per-room-type lock in
    [Hotel management](03-hotel-management.md): the lock is sized to the unit
    the rule spans, not to the unit being changed.

    LOCK ORDERING, when more than one book is involved (a return that advances a
    queue on a different title, or a batch operation): acquire by sorted ISBN,
    exactly as the seat design orders by seat id.
    """

    def __init__(self, catalog: Catalog, ledger: LoanLedger,
                 calendar: LibraryCalendar, clock: Clock,
                 hold_ttl_seconds: float = 3 * 24 * 60 * 60) -> None:
        self.catalog = catalog
        self.ledger = ledger
        self.calendar = calendar
        self.clock = clock
        self.hold_ttl_seconds = hold_ttl_seconds
        self._queues: dict[str, ReservationQueue] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._next_loan = 0
        self._next_reservation = 0

    # ---- lock plumbing -------------------------------------------------
    def _lock_for(self, isbn: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._locks.get(isbn)
            if lock is None:
                lock = threading.RLock()
                self._locks[isbn] = lock
            return lock

    def _locked_books(self, isbns: "list[str]") -> "list[threading.RLock]":
        """One lock per distinct title, acquired in SORTED order. This is the
        deadlock rule; acquiring in the caller's order is how a deadlock gets
        built from two correct-looking operations."""
        locks = [self._lock_for(isbn) for isbn in sorted(set(isbns))]
        for lock in locks:
            lock.acquire()
        return locks

    def _release(self, locks: "list[threading.RLock]") -> None:
        for lock in reversed(locks):
            lock.release()

    def queue_for(self, isbn: str) -> ReservationQueue:
        """Lazily created, and only ever created under the title's lock or the
        guard — a lazily created dict entry is otherwise a race between two
        threads that both find nothing and both insert."""
        with self._lock_for(isbn):
            queue = self._queues.get(isbn)
            if queue is None:
                queue = ReservationQueue(self.catalog.book(isbn))
                self._queues[isbn] = queue
            return queue

    # ---- borrow --------------------------------------------------------
    def borrow(self, member: Member, isbn: str) -> Loan:
        """THE RACE THIS EXISTS FOR: two members at two desks both try to borrow
        the last copy. Both read `availability == 1`, both find the same copy
        available, both write — and one copy has two loans.

        The fix is that the read, the decision and the write are all inside one
        critical section for the title. There is no check outside the lock."""
        locks = self._locked_books([isbn])
        try:
            book = self.catalog.book(isbn)

            # Rule 1: the member may borrow at all.
            active = self.ledger.active_loans_of(member.member_id)
            member.policy.may_borrow(member, len(active),
                                     self.ledger.unpaid_total(member.member_id))

            # Rule 2: one copy of a given title per member. Checked here, in the
            # same critical section, because it is a count over this member's
            # loans and another concurrent borrow of the same title would
            # otherwise slip past it.
            if self.ledger.title_on_loan_to(member.member_id, isbn):
                raise LoanLimitError(
                    f"{member.member_id} already has a copy of {isbn} on loan"
                )

            # Rule 3: a copy must exist. Note the queue is consulted FIRST: if
            # this member has a READY reservation, their copy is on the hold
            # shelf and they must be given THAT one, not any available copy —
            # otherwise a walk-in could take the very copy held for them and the
            # ready reservation would sit there until it expired.
            copy = self._copy_for(member, isbn)
            if copy is None:
                raise NoCopyAvailableError(
                    f"no copy of {isbn} is available; "
                    f"{self.queue_for(isbn).waiting_count()} member(s) waiting"
                )

            borrowed_on = self.clock.today()
            due_on = self.calendar.add_open_days(borrowed_on,
                                                 member.policy.loan_days)
            self._next_loan += 1
            loan = Loan(
                loan_id=f"L{self._next_loan:08d}", copy=copy, member=member,
                borrowed_on=borrowed_on, due_on=due_on,
            )

            copy.status = CopyStatus.ON_LOAN
            self.catalog.sync_available_count(isbn)
            self.ledger.record_loan(loan)

            # The member's own READY reservation is now fulfilled.
            self._fulfil_reservation_for(member, isbn, copy)
            return loan
        finally:
            self._release(locks)

    def _copy_for(self, member: Member, isbn: str) -> "BookCopy | None":
        """The member's own held copy first, then any available copy."""
        for reservation in self.queue_for(isbn).open_reservations():
            if (reservation.member.member_id == member.member_id
                    and reservation.state is ReservationState.READY
                    and reservation.ready_accession is not None):
                copy = self.catalog.copy(reservation.ready_accession)
                if copy.status is CopyStatus.ON_HOLD_SHELF:
                    return copy
        return self.catalog.find_borrowable_copy(isbn)

    def _fulfil_reservation_for(self, member: Member, isbn: str,
                                copy: BookCopy) -> None:
        for reservation in self.queue_for(isbn).snapshot():
            if (reservation.member.member_id == member.member_id
                    and reservation.state is ReservationState.READY
                    and reservation.ready_accession == copy.accession):
                reservation.state = ReservationState.FULFILLED
                return

    # ---- return --------------------------------------------------------
    def return_copy(self, accession: str, on: "date | None" = None) -> "Loan":
        """THE SECOND HALF OF THE RACE: a return and a reservation.

        A returned copy has three possible destinations, and getting the order
        wrong is the bug:
          1. A reservation is WAITING → the copy goes to ON_HOLD_SHELF for that
             member and the hold clock starts. NOT to AVAILABLE.
          2. No reservation → AVAILABLE and the cache is synced.
        The failure mode of getting it wrong is specific and bad: the copy is
        marked available, a walk-in borrows it ten seconds later, and the member
        who has been waiting three weeks is told their book came and went.
        """
        copy = self.catalog.copy(accession)
        isbn = copy.book.isbn

        locks = self._locked_books([isbn])
        try:
            loan = self.ledger.active_loan_of_copy(accession)
            if loan is None:
                raise LoanNotFoundError(f"{accession} is not on loan")

            if loan.copy.status is not CopyStatus.ON_LOAN:
                raise LibraryError(
                    f"loan {loan.loan_id} is active but copy {accession} is "
                    f"{loan.copy.status.value}; the copy status and the loan "
                    f"disagree"
                )

            returned_on = on or self.clock.today()
            if returned_on < loan.borrowed_on:
                raise LibraryError("a copy cannot be returned before it was lent")

            loan.returned_on = returned_on
            loan.state = LoanState.RETURNED

            # The fine is assessed HERE, once, and frozen (§5.5).
            assessment = self._assess_fine(loan, returned_on)

            # Destination 1: a waiting reservation gets the hold shelf.
            queue = self.queue_for(isbn)
            next_member = queue.peek()
            if next_member is not None:
                copy.status = CopyStatus.ON_HOLD_SHELF
                next_member.state = ReservationState.READY
                next_member.ready_accession = accession
                next_member.expires_at = self.clock.now() + self.hold_ttl_seconds
            else:
                # Destination 2: back to the shelf.
                copy.status = CopyStatus.AVAILABLE

            self.catalog.sync_available_count(isbn)
            return loan
        finally:
            self._release(locks)

    def _assess_fine(self, loan: Loan, returned_on: date) -> "FineAssessment | None":
        """Compute, cap, freeze. Returns None when nothing is owed, which is the
        common case and should not create a zero-value row — a ledger full of
        zero fines makes the real ones hard to find."""
        charged_days = loan.chargeable_open_days(
            returned_on, self.calendar, loan.member.policy.grace_open_days
        )
        if charged_days <= 0:
            return None
        raw = charged_days * loan.member.policy.fine_per_open_day_minor
        capped = min(raw, loan.member.policy.fine_cap_minor(loan.copy))
        if capped <= 0:
            # No row for a zero fine. A ledger full of zero-value assessments
            # makes the real ones hard to find, and it would also mean every
            # return created a row whether or not anything was owed.
            return None
        assessment = FineAssessment(
            assessment_id=f"F{len(self.ledger.assessments) + 1:08d}",
            loan=loan, amount_minor=capped, assessed_on=returned_on,
            open_days_charged=charged_days,
        )
        self.ledger.record_assessment(assessment)
        return assessment

    # ---- renew ---------------------------------------------------------
    def renew(self, loan_id: str) -> Loan:
        """THE INTERACTION WORTH KNOWING: a renewal is refused when someone else
        is WAITING for the title. That single rule is what makes the reservation
        queue meaningful — without it, a member can hold a popular book
        indefinitely by renewing, and the queue never advances, and the feature
        the library advertises does not work.

        Note the check is `has_waiting()` and not `waiting_count() > 0`: they are
        the same thing, and the named method says which intent it encodes.
        """
        loans = [loan for loan in self.ledger.loans if loan.loan_id == loan_id]
        if not loans:
            raise LoanNotFoundError(f"unknown loan {loan_id}")
        loan = loans[0]
        isbn = loan.copy.book.isbn

        locks = self._locked_books([isbn])
        try:
            if loan.state is not LoanState.ACTIVE:
                raise NotRenewableError(f"loan {loan_id} is {loan.state.value}")

            policy = loan.member.policy
            if loan.renewals >= policy.max_renewals:
                raise NotRenewableError(
                    f"{policy.name} members may renew {policy.max_renewals} "
                    f"time(s); this loan has been renewed {loan.renewals}"
                )
            if self.queue_for(isbn).has_waiting():
                raise NotRenewableError(
                    f"{self.queue_for(isbn).waiting_count()} member(s) are "
                    f"waiting for {isbn}; this loan cannot be renewed"
                )

            today = self.clock.today()
            if today > loan.due_on:
                # Renewing an overdue loan is refused, and the reason is the
                # fine: a fine was already earned for the days already late, and
                # a renewal would silently reset the clock and forgive it.
                raise NotRenewableError(
                    f"loan {loan_id} was due {loan.due_on}; return it and "
                    f"settle the fine before borrowing again"
                )

            loan.due_on = self.calendar.add_open_days(today, policy.loan_days)
            loan.renewals += 1
            return loan
        finally:
            self._release(locks)

    # ---- reserve -------------------------------------------------------
    def reserve(self, member: Member, isbn: str) -> Reservation:
        locks = self._locked_books([isbn])
        try:
            book = self.catalog.book(isbn)
            self._next_reservation += 1
            reservation = Reservation(
                reservation_id=f"R{self._next_reservation:08d}", book=book,
                member=member, placed_at=self.clock.now(),
            )

            # A reservation on a title with a free copy is allowed and it is a
            # deliberate choice: it holds the copy off the shelf so the member
            # can collect it. That is what a "request" button does at a real
            # library. The alternative — refusing, and telling them to come and
            # get it — loses the copy in the meantime.
            queue = self.queue_for(isbn)
            queue.enqueue(reservation)

            if self.catalog.availability(isbn) > 0:
                copy = self.catalog.find_borrowable_copy(isbn)
                if copy is not None:
                    copy.status = CopyStatus.ON_HOLD_SHELF
                    reservation.state = ReservationState.READY
                    reservation.ready_accession = copy.accession
                    reservation.expires_at = self.clock.now() + self.hold_ttl_seconds
                    self.catalog.sync_available_count(isbn)

            return reservation
        finally:
            self._release(locks)

    # ---- the sweeper ---------------------------------------------------
    def sweep_expired_holds(self, batch_limit: int = 100) -> int:
        """Move expired READY reservations out and hand the copy to the next
        person. Bounded, and callable from a scheduler.

        LAZY PLUS SWEEPER, the same pattern as the seat-hold expiry: every read
        treats an expired reservation as expired (lazy), and this job reclaims
        the copies so they do not sit on the hold shelf forever. The lazy part is
        what makes correctness independent of the job running; the sweeper is
        what makes the shelf space come back."""
        swept = 0
        now = self.clock.now()

        for isbn in sorted(self._queues):
            if swept >= batch_limit:
                break
            locks = self._locked_books([isbn])
            try:
                queue = self._queues[isbn]
                for reservation in queue.open_reservations():
                    if swept >= batch_limit:
                        break
                    if not reservation.is_expired(now):
                        continue

                    reservation.state = ReservationState.EXPIRED
                    accession = reservation.ready_accession
                    if accession is not None:
                        copy = self.catalog.copy(accession)
                        if copy.status is CopyStatus.ON_HOLD_SHELF:
                            # The copy goes to the next WAITING member, or back
                            # to the shelf. Note it does NOT go to AVAILABLE
                            # first — that window is the bug.
                            following = queue.peek()
                            if following is not None:
                                copy.status = CopyStatus.ON_HOLD_SHELF
                                following.state = ReservationState.READY
                                following.ready_accession = accession
                                following.expires_at = now + self.hold_ttl_seconds
                            else:
                                copy.status = CopyStatus.AVAILABLE
                            self.catalog.sync_available_count(isbn)
                    swept += 1
            finally:
                self._release(locks)
        return swept
```

### 5.5 The fine arithmetic, walked

```
   A LOAN UNDER StudentPolicy: loan_days=14, fine=₹5/open day, grace=2.

   borrowed_on = Fri 2026-03-06          closed Sundays (an explicit date set)
   due_on      = the 14th OPEN day after it

   Walk it:  Mar 7 Sat(1), 8 Sun closed, 9(2), 10(3), 11(4), 12(5), 13(6),
             14 Sat(7), 15 Sun closed, 16(8), 17(9), 18(10), 19(11), 20(12),
             21 Sat(13), 22 Sun closed, 23(14)
   So  due_on = Mon 2026-03-23.

   Note what a naive `borrowed_on + timedelta(days=14)` gives: Sat 2026-03-20.
   That is three days EARLIER than the truth, and every member borrowing on a
   Friday would be fined for a weekend plus the Sunday the library was shut.
   This is why the calendar exists and why the due date is a walk, not an add.

   And note the closure on Sun 2026-03-22 lands INSIDE the loan window, so it
   pushes the due date out by one — which is exactly the behaviour a member
   expects and the reason the arithmetic cannot be a constant.

   NOW THE FINES, with due_on = Mon 2026-03-23 and grace = 2 open days:

   returned_on        open days in [due, returned)   minus grace   charged
   ───────────────────────────────────────────────────────────────────────
   2026-03-23 (due)              0                       0            0
   2026-03-24 (Tue)              1                       0            0
   2026-03-25 (Wed)              2                       0            0
   2026-03-26 (Thu)              3                       1        ₹5
   2026-03-27 (Fri)              4                       2       ₹10
   2026-03-28 (Sat)              5                       3       ₹15
   2026-03-30 (Mon)    ← Sunday is NOT counted
                                  6                       4       ₹20
   2026-04-15                    20                      18       ₹90
                                                  capped at the replacement cost

   THREE THINGS TO NOTICE, and each is a question the interviewer may ask:
     1. Returning ON the due date charges nothing, because the interval is
        HALF-OPEN. A closed interval charges everybody who returns on time.
     2. Sunday costs nothing. A member is not fined for a day they could not
        have returned the book.
     3. The grace period is counted in OPEN days too, so a "2-day grace" that
        spans a Sunday is still two days of actual opportunity to return it.

   AND THE FINE IS FROZEN HERE. It is not recomputed tomorrow. If it were, a
   member who returned the book late and has not paid would see a number that
   grows every day they look at it — and there would be no figure to appeal.
```

### 5.6 It runs

```python
class ManualClock(Clock):
    def __init__(self, day: date, instant: float = 1_000_000.0) -> None:
        self._day = day
        self._instant = instant

    def today(self) -> date:
        return self._day

    def now(self) -> float:
        return self._instant

    def set_day(self, day: date) -> None:
        self._day = day

    def advance_seconds(self, seconds: float) -> None:
        self._instant += seconds


def build_library() -> "tuple[Catalog, LoanLedger, LendingService, ManualClock]":
    catalog = Catalog()
    ledger = LoanLedger()

    for isbn, title, author, price in [
        ("978-0132350884", "Clean Code", "Robert C. Martin", 250_000),
        ("978-0201633610", "Design Patterns", "Gamma et al.", 400_000),
        ("978-0134685991", "Effective Java", "Joshua Bloch", 300_000),
    ]:
        catalog.add_book(Book(isbn, title, author, price))

    # THREE copies of Clean Code, which is the point of the Book/BookCopy split.
    catalog.add_copy(BookCopy("A0001", catalog.book("978-0132350884"), "A-01"))
    catalog.add_copy(BookCopy("A0002", catalog.book("978-0132350884"), "A-01",
                              condition="water-damaged"))
    catalog.add_copy(BookCopy("A0003", catalog.book("978-0132350884"), "A-02"))
    catalog.add_copy(BookCopy("B0001", catalog.book("978-0201633610"), "B-01"))
    catalog.add_copy(BookCopy("C0001", catalog.book("978-0134685991"), "C-01"))

    # Closed on Sundays in March, which is the calendar §5.5's worked example
    # uses — so the demo below prints the same due date the arithmetic section
    # derives, and the two can be read together.
    calendar = LibraryCalendar(closed_dates={
        date(2026, 3, 8), date(2026, 3, 15), date(2026, 3, 22), date(2026, 3, 29),
    })
    clock = ManualClock(date(2026, 3, 6))
    service = LendingService(catalog, ledger, calendar, clock)
    return catalog, ledger, service, clock


def demo() -> None:
    catalog, ledger, service, clock = build_library()

    alice = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    bob = Member("M002", "Bob", StudentPolicy(), date(2025, 8, 1))
    carol = Member("M003", "Carol", StudentPolicy(), date(2025, 8, 1))

    print("available copies of Clean Code:",
          catalog.availability("978-0132350884"))                # 3

    # Alice borrows, and the count is the CACHE synced from the truth.
    loan_a = service.borrow(alice, "978-0132350884")
    print("Alice:", loan_a, "| due:", loan_a.due_on, "| available:",
          catalog.availability("978-0132350884"))                # 2
    catalog.check_invariants()

    # Alice cannot take a SECOND copy of the same title.
    try:
        service.borrow(alice, "978-0132350884")
    except LoanLimitError as exc:
        print("refused:", exc)

    # Bob and Carol take the other two. Now nothing is available.
    service.borrow(bob, "978-0132350884")
    service.borrow(carol, "978-0132350884")
    print("available now:", catalog.availability("978-0132350884"))   # 0

    # A fourth member reserves, and cannot borrow.
    dave = Member("M004", "Dave", StudentPolicy(), date(2025, 8, 1))
    try:
        service.borrow(dave, "978-0132350884")
    except NoCopyAvailableError as exc:
        print("refused:", exc)
    reservation = service.reserve(dave, "978-0132350884")
    print("Dave reserved, position",
          service.queue_for("978-0132350884").position_of(reservation))

    # Alice renews — allowed, because nobody is WAITING yet? No: Dave IS waiting,
    # so the renewal is refused. This is §7.1's interaction.
    try:
        service.renew(loan_a.loan_id)
    except NotRenewableError as exc:
        print("renewal refused:", exc)

    # Alice returns. The copy goes ON THE HOLD SHELF for Dave, NOT to available.
    service.return_copy("A0001")
    print("copy A0001 status:",
          catalog.copy("A0001").status.value)                    # on_hold_shelf
    print("available (still 0):",
          catalog.availability("978-0132350884"))                # 0
    catalog.check_invariants()

    # Dave collects it.
    loan_d = service.borrow(dave, "978-0132350884")
    print("Dave:", loan_d, "| reservation state:",
          reservation.state.value)                               # fulfilled

    # The fine path: Carol keeps her copy past the due date.
    clock.set_day(date(2026, 5, 1))
    service.return_copy("A0003")
    assessment = ledger.assessments[-1]
    print("Carol's fine:", assessment, "=", Money(assessment.amount_minor))
    ledger.check_invariants()


if __name__ == "__main__":
    demo()
```

**The line in the demonstration that matters most is `copy A0001 status: on_hold_shelf` followed by `available (still 0)`.** The copy is physically on the shelf and the count says zero, which is correct: it is promised to Dave. **A design that marks it available and then notifies Dave has created a window in which anybody can take it, and that window is where the feature fails.**

---

## 6. Concurrency and thread-safety

### 6.1 The races

| # | Race | Consequence | Fix |
|---|---|---|---|
| 1 | **Two members borrow the last copy** | **One copy, two loans** | Per-title lock; the read and the write share one critical section |
| 2 | A return racing a reservation | The copy marked available; a walk-in takes it; the waiting member loses it | The return's destination is decided inside the same lock (§5.4) |
| 3 | A renewal racing a new reservation | The renewal succeeds after the queue formed | Both take the title lock; renewal checks `has_waiting` inside it |
| 4 | Two members reserving at once | Two positions that are both "first" | The queue is mutated under the title lock |
| 5 | The sweeper expiring a hold while the member is at the desk | The copy is handed to the next person mid-collection | The sweeper takes the title lock; the borrow checks `state` and `expires_at` |
| 6 | Two borrows of *different* titles | Independent; no interaction | Different locks — this is why per-title granularity is worth it |
| 7 | A return of one title whose queue advances another | A lock-ordering deadlock | `_locked_books` acquires by sorted ISBN |
| 8 | The `available_count` cache vs the truth | A search result that lies | Synced inside the same critical section, never by a job |
| 9 | Two threads lazily creating the same queue | Two queues for one title; reservations split across them | The queue is created under the title's lock |
| 10 | A member's active-loan count read outside the lock | Two borrows both passing the limit check | The count is read inside the title lock of the title being borrowed |
| 11 | A lost copy still marked `ON_LOAN` | A loan that can never be returned; a phantom overdue | The `LOST` transition closes the loan in the same critical section (§7.4) |
| 12 | A fine assessed twice on one return | **A double charge** | The return checks the loan is `ACTIVE` first; a returned loan has no active loan |

**Race 10 is the honest gap in the per-title design, and it is worth stating.** A member's loan limit spans *all* titles, and a per-title lock cannot hold that set atomically. Two concurrent borrows of two different titles can both read `active_loans == max_loans - 1` and both proceed, leaving the member one over the limit. **The consequence is benign — one extra book, not a corrupt ledger — and it is the standard trade: per-title locking for throughput, and the limit enforced as a soft rule.** The alternative, a per-member lock, would have to be held alongside the title lock and would recreate the deadlock ordering problem for a rule whose violation is a book too many.

**Say that out loud rather than pretending the lock covers everything.**

### 6.2 The invariant that proves the model

```
   FOR EVERY ACTIVE LOAN:  loan.copy.status IS ON_LOAN
   FOR EVERY COPY ON_LOAN: EXACTLY ONE active loan references it
   FOR EVERY TITLE:        book.available_count == count of AVAILABLE copies

   These three are what make the model trustworthy, and `check_invariants` on
   both the catalogue and the ledger asserts all three. The first two catch the
   borrowing race; the third catches the cache drifting from the truth.

   Note what is NOT an invariant: "a copy has at most one reservation". It does
   not, and it should not — a title can have twenty reservations, and the copy
   that comes back goes to exactly one of them, chosen by the queue.
```

### 6.3 Tests

```python
import pytest


def test_the_book_and_copy_split_is_load_bearing() -> None:
    """Three copies of one title, with different conditions, and availability
    that is a count rather than a flag."""
    catalog, _, _, _ = build_library()
    assert catalog.total_copies("978-0132350884") == 3
    assert catalog.availability("978-0132350884") == 3

    damaged = catalog.copy("A0002")
    assert damaged.condition == "water-damaged"
    damaged.status = CopyStatus.IN_REPAIR
    catalog.sync_available_count("978-0132350884")
    assert catalog.availability("978-0132350884") == 2
    assert catalog.book("978-0132350884").available_count == 2
    catalog.check_invariants()


def test_the_due_date_is_counted_in_open_days() -> None:
    """The decision most candidates get wrong: 14 OPEN days, not
    borrowed_on + 14."""
    calendar = LibraryCalendar(closed_dates={date(2026, 3, 9), date(2026, 3, 10)})
    borrowed = date(2026, 3, 6)                       # Friday
    # Two closed days early in the window push the due date out by two.
    assert calendar.add_open_days(borrowed, 14) == date(2026, 3, 22)
    # And the naive answer is two days earlier, which is the bug.
    assert borrowed + timedelta(days=14) == date(2026, 3, 20)


def test_returning_on_the_due_date_charges_nothing() -> None:
    """THE half-open interval test. A closed interval fails this, and failing
    this means fining every member who returns on time."""
    calendar = LibraryCalendar()
    policy = StudentPolicy()
    due = date(2026, 3, 23)

    member = Member("M001", "Alice", policy, date(2025, 8, 1))
    book = Book("X", "T", "A", 250_000)
    copy = BookCopy("A1", book)
    loan = Loan("L1", copy, member, date(2026, 3, 9), due)

    assert loan.chargeable_open_days(due, calendar, policy.grace_open_days) == 0
    assert not loan.is_overdue(due)
    assert loan.is_overdue(date(2026, 3, 24))


def test_the_grace_period_is_counted_in_open_days() -> None:
    calendar = LibraryCalendar(closed_dates={date(2026, 3, 29)})
    policy = StudentPolicy()                          # grace = 2 open days
    member = Member("M001", "Alice", policy, date(2025, 8, 1))
    loan = Loan("L1", BookCopy("A1", Book("X", "T", "A", 250_000)), member,
                date(2026, 3, 9), date(2026, 3, 23))

    # Thu and Fri are within grace.
    assert loan.chargeable_open_days(date(2026, 3, 26), calendar, 2) == 1
    # Sat 28th is a 5th open day; Sun 29th is closed and must not count.
    assert loan.chargeable_open_days(date(2026, 3, 28), calendar, 2) == 3
    assert loan.chargeable_open_days(date(2026, 3, 30), calendar, 2) == 4


def test_a_fine_is_assessed_once_and_frozen() -> None:
    """THE money test: returning, then advancing the clock a month, must not
    change the fine."""
    catalog, ledger, service, clock = build_library()
    member = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    loan = service.borrow(member, "978-0132350884")

    clock.set_day(loan.due_on + timedelta(days=10))
    service.return_copy(loan.copy.accession)
    assessment = ledger.assessments[-1]
    frozen = assessment.amount_minor

    clock.set_day(loan.due_on + timedelta(days=80))
    assert assessment.amount_minor == frozen, "the fine must not grow"
    assert assessment.assessed_on == loan.due_on + timedelta(days=10)


def test_a_fine_is_capped_at_the_replacement_cost() -> None:
    catalog, ledger, service, clock = build_library()
    member = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    loan = service.borrow(member, "978-0132350884")

    clock.set_day(loan.due_on + timedelta(days=3650))     # ten years late
    service.return_copy(loan.copy.accession)
    assessment = ledger.assessments[-1]
    assert assessment.amount_minor == loan.copy.book.replacement_minor


def test_two_threads_cannot_borrow_the_last_copy() -> None:
    """THE race. One copy, eight members, eight threads, one loan."""
    catalog = Catalog()
    catalog.add_book(Book("X", "Title", "Author", 100_000))
    catalog.add_copy(BookCopy("A1", catalog.book("X")))
    ledger = LoanLedger()
    clock = ManualClock(date(2026, 3, 6))
    service = LendingService(catalog, ledger, LibraryCalendar(), clock)

    members = [Member(f"M{i:03d}", f"Member {i}", StudentPolicy(),
                      date(2025, 8, 1)) for i in range(8)]
    barrier = threading.Barrier(8)
    successes: list[str] = []
    failures: list[str] = []
    guard = threading.Lock()

    def attempt(member: Member) -> None:
        barrier.wait()
        try:
            loan = service.borrow(member, "X")
            with guard:
                successes.append(loan.loan_id)
        except LibraryError as exc:
            with guard:
                failures.append(type(exc).__name__)

    threads = [threading.Thread(target=attempt, args=(m,)) for m in members]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 1, f"expected one loan, got {successes}"
    assert len(failures) == 7
    ledger.check_invariants()
    catalog.check_invariants()
    assert catalog.availability("X") == 0


def test_a_returned_copy_goes_to_the_hold_shelf_not_to_available() -> None:
    """THE second race. The copy must never be briefly available."""
    catalog, ledger, service, clock = build_library()
    alice = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    dave = Member("M004", "Dave", StudentPolicy(), date(2025, 8, 1))

    # Take the only copy of Design Patterns.
    loan = service.borrow(alice, "978-0201633610")
    reservation = service.reserve(dave, "978-0201633610")
    assert reservation.state is ReservationState.WAITING

    service.return_copy(loan.copy.accession)

    copy = catalog.copy("B0001")
    assert copy.status is CopyStatus.ON_HOLD_SHELF, "not available, not on loan"
    assert catalog.availability("978-0201633610") == 0
    assert reservation.state is ReservationState.READY
    assert reservation.ready_accession == "B0001"
    assert reservation.expires_at is not None

    # And Dave gets THAT copy, not an arbitrary one.
    loan_d = service.borrow(dave, "978-0201633610")
    assert loan_d.copy.accession == "B0001"
    assert reservation.state is ReservationState.FULFILLED


def test_an_expired_hold_advances_the_queue() -> None:
    catalog, ledger, service, clock = build_library()
    alice = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    dave = Member("M004", "Dave", StudentPolicy(), date(2025, 8, 1))
    erin = Member("M005", "Erin", StudentPolicy(), date(2025, 8, 1))

    loan = service.borrow(alice, "978-0201633610")
    first = service.reserve(dave, "978-0201633610")
    second = service.reserve(erin, "978-0201633610")
    service.return_copy(loan.copy.accession)
    assert first.state is ReservationState.READY

    # Dave never collects.
    clock.advance_seconds(service.hold_ttl_seconds + 1)
    assert service.sweep_expired_holds() == 1

    assert first.state is ReservationState.EXPIRED
    assert second.state is ReservationState.READY, "the queue advanced"
    assert second.ready_accession == "B0001"
    assert catalog.copy("B0001").status is CopyStatus.ON_HOLD_SHELF
    # Idempotent: sweeping again does nothing.
    assert service.sweep_expired_holds() == 0


def test_a_renewal_is_refused_when_someone_is_waiting() -> None:
    catalog, ledger, service, clock = build_library()
    alice = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    dave = Member("M004", "Dave", StudentPolicy(), date(2025, 8, 1))

    loan = service.borrow(alice, "978-0201633610")
    service.renew(loan.loan_id)                       # fine, nobody waiting
    assert loan.renewals == 1

    service.reserve(dave, "978-0201633610")
    with pytest.raises(NotRenewableError):
        service.renew(loan.loan_id)
    assert loan.renewals == 1, "a refused renewal must not increment"


def test_a_member_cannot_hold_two_copies_of_one_title() -> None:
    catalog, ledger, service, clock = build_library()
    alice = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    service.borrow(alice, "978-0132350884")
    with pytest.raises(LoanLimitError):
        service.borrow(alice, "978-0132350884")
    # But a different title is fine.
    service.borrow(alice, "978-0201633610")
    assert len(ledger.active_loans_of("M001")) == 2


def test_a_member_at_their_loan_limit_is_refused() -> None:
    catalog = Catalog()
    ledger = LoanLedger()
    clock = ManualClock(date(2026, 3, 6))
    service = LendingService(catalog, ledger, LibraryCalendar(), clock)

    for index in range(6):
        isbn = f"ISBN-{index}"
        catalog.add_book(Book(isbn, f"Title {index}", "Author", 100_000))
        catalog.add_copy(BookCopy(f"A{index}", catalog.book(isbn)))

    member = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    for index in range(4):                            # StudentPolicy max_loans = 4
        service.borrow(member, f"ISBN-{index}")
    with pytest.raises(LoanLimitError):
        service.borrow(member, "ISBN-4")
    assert len(ledger.active_loans_of("M001")) == 4


def test_a_suspended_member_cannot_borrow() -> None:
    catalog, ledger, service, clock = build_library()
    member = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    member.status = MemberStatus.SUSPENDED
    with pytest.raises(MemberBlockedError):
        service.borrow(member, "978-0132350884")


def test_a_loan_and_its_copy_status_never_disagree() -> None:
    """The invariant, after a mixed sequence."""
    catalog, ledger, service, clock = build_library()
    alice = Member("M001", "Alice", StudentPolicy(), date(2025, 8, 1))
    bob = Member("M002", "Bob", StudentPolicy(), date(2025, 8, 1))

    loan_a = service.borrow(alice, "978-0132350884")
    loan_b = service.borrow(bob, "978-0132350884")
    service.return_copy(loan_a.copy.accession)
    service.return_copy(loan_b.copy.accession)

    ledger.check_invariants()
    catalog.check_invariants()
    assert catalog.availability("978-0132350884") == 3
    assert all(loan.state is LoanState.RETURNED
               for loan in ledger.loans)


def test_concurrent_mixed_operations_keep_every_invariant() -> None:
    """Six members borrowing, returning and reserving different titles at once.
    Every invariant must hold afterwards."""
    catalog = Catalog()
    ledger = LoanLedger()
    for index in range(4):
        isbn = f"ISBN-{index}"
        catalog.add_book(Book(isbn, f"Title {index}", "Author", 100_000))
        for copy_index in range(3):
            catalog.add_copy(BookCopy(f"A{index}{copy_index}", catalog.book(isbn)))

    clock = ManualClock(date(2026, 3, 6))
    service = LendingService(catalog, ledger, LibraryCalendar(), clock)
    members = [Member(f"M{i:03d}", f"Member {i}", StudentPolicy(),
                      date(2025, 8, 1)) for i in range(6)]

    errors: list[str] = []
    guard = threading.Lock()
    barrier = threading.Barrier(6 * 12)

    def worker(member_index: int, step: int) -> None:
        member = members[member_index]
        isbn = f"ISBN-{step % 4}"
        barrier.wait()
        try:
            if step % 3 == 0:
                try:
                    service.borrow(member, isbn)
                except LibraryError:
                    pass                               # expected: no copy
            elif step % 3 == 1:
                try:
                    service.reserve(member, isbn)
                except LibraryError:
                    pass
            else:
                loans = ledger.active_loans_of(member.member_id)
                if loans:
                    service.return_copy(loans[0].copy.accession)
        except Exception as exc:                        # noqa: BLE001
            with guard:
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(m, s))
               for m in range(6) for s in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    ledger.check_invariants()
    catalog.check_invariants()
```

---

## 7. Extensibility — the standard follow-ups

### 7.1 "Add renewals" — and the interaction that makes them interesting

Built into §5.4, and the follow-up is the reason. **A renewal must be refused when someone is waiting for the title.** That one rule is what makes the reservation queue mean anything: without it, a member renews indefinitely, the queue never advances, and the library advertises a feature that does not work. **The follow-up is usually "what if the renewal is refused — what do you tell the member?"**, and the honest answer is that the refusal message names the waiting count, because a member who is told "3 people are waiting" accepts it and a member who is told "not renewable" phones the desk.

### 7.2 "Add an inter-library loan"

A title the branch does not hold, borrowed from another branch. **The design consequence is that the copy belongs to another `Catalog`**, so a `BookCopy` grows an owning branch and the loan grows a *lending* branch and a *holding* branch — and the fine policy becomes a question of whose policy applies. **That is a real decision: the borrowing member's policy, because they are the one who might return it late.** Saying that is the whole answer.

### 7.3 "Add e-books"

The extension that breaks the model, and saying so is the answer. An e-book has **no copies** — a licence permits N concurrent readers, and the "copy" is a licence seat. So `BookCopy` becomes `Licence` with an expiry date and a concurrent-reader count, `CopyStatus.ON_LOAN` becomes "a seat is taken", and the return is an explicit release rather than a physical event. **The insight worth stating: a licence that expires means the library can LOSE access to a title it has lent, which has no physical analogue, and the member's loan may have to be terminated early by the library rather than by the member.** That is a genuinely new failure mode and naming it is the answer.

### 7.4 "A lost copy"

```python
def mark_lost(self, accession: str, on: "date | None" = None) -> FineAssessment:
    """THE transitions that must all happen together:
       - the copy moves to LOST
       - the active loan closes as LOST, not RETURNED
       - the member is charged the replacement cost
       - the available count is synced
       - the reservation queue advances, because the copy will never come back

    Note the charge is the REPLACEMENT COST, capped by the member's policy, and
    it is not an overdue fine — the two are different assessments on the same
    loan, and conflating them is a real bug: a member who loses a book and
    returns it a year later owes the replacement cost AND nothing else, because
    the fine stops accruing the moment the copy is declared lost.
    """
```

**The follow-up is "what if the member finds it and returns it a year later?"** The copy transitions `LOST → AVAILABLE`, the replacement charge is refunded or credited, and **no overdue fine accrues for the lost period** — because the member was already charged for the book. **Charging both is the double-charge bug, and it is the most likely thing an interviewer will probe.**

### 7.5 "Notify members when their book is ready"

A `NotificationPort` interface with a single `send(member, message)` method, and **the class design does not own delivery** — which is the correct answer to give. The point worth making: **the notification is triggered inside the return's critical section** — where the state change happens — and the *send* is what should be deferred, because a network call inside a lock is the mistake the caching file warns about. So the queue state changes under the lock, a message is appended to a dispatch list, and the send happens after the lock is released.

### 7.6 "Barcodes, and a self-checkout machine"

The extension that turns this into a real deployment. **A self-checkout is a second writer on the same catalogue**, which changes nothing in the class design — the per-title lock is already the right granularity — and everything in the deployment: the machine must be able to fail and retry, so `borrow` needs an idempotency key of `(accession, member, day)` so a re-scan does not create a second loan. **That is the same idempotency discipline as the ATM's hold reference and the Splitwise settlement key, and saying so links the three.**

### 7.7 "Search, and the availability count"

The extension where the cache earns its existence. Search must render availability for a page of results, so the derived count is too slow — and **the answer is that `available_count` stays a cache, synced inside the critical section, plus a periodic full reconciliation** because a cache on a mutable count can drift in a way a cache on an append-only log cannot (the contrast with Splitwise's versioned cache is exact and worth drawing: there the version was free, here it is not).

---

## 8. Failure modes and edge cases

### 8.1 Failure modes

| # | Failure | Consequence | Handling |
|---|---|---|---|
| 1 | **One `Book` class with an `is_available` flag** | Cannot answer "how many copies", cannot attach a return to a copy | `Book` and `BookCopy` are separate (§2) |
| 2 | **Two members borrow the last copy** | One copy, two loans | Reads and writes in one per-title critical section |
| 3 | **A returned copy marked available while a reservation waits** | A walk-in takes the held copy; the waiting member loses it | The copy goes `ON_HOLD_SHELF` inside the return's lock |
| 4 | `due_on = borrowed_on + timedelta(days=n)` | Every member fined for days the library was shut | `add_open_days` over a calendar |
| 5 | **A closed interval for lateness** | Everyone who returns on time is fined | Half-open `[due_on, returned_on)` |
| 6 | A fine recomputed on read | A number that grows while the member looks at it; nothing to appeal | Assessed once at return, then frozen |
| 7 | A fine assessed twice for one return | **A double charge** | The return requires the loan to be `ACTIVE` first |
| 8 | A fine with no cap | A ₹200 book generating a ₹10,000 fine | Capped at the replacement cost, per policy |
| 9 | A renewal when someone is waiting | The queue never advances; the advertised feature does not work | `NotRenewableError` on `has_waiting` |
| 10 | A renewal of an overdue loan | The earned fine silently forgiven | Refused; return it and settle first |
| 11 | A renewal incrementing after a refusal | A member loses a renewal they never got | The increment is after every check |
| 12 | A lost copy still `ON_LOAN` | A loan that can never be returned; phantom overdues | `LOST` closes the loan in the same critical section |
| 13 | A lost copy charged AND fined | **A double charge** | Declaring it lost stops the fine; the two are exclusive |
| 14 | A returned lost copy not refunded | The member paid for a book they gave back | `LOST → AVAILABLE` credits the replacement charge |
| 15 | A reservation holding a copy forever | The shelf fills with uncollected holds | A TTL from when the copy becomes ready, plus a sweeper |
| 16 | A WAITING reservation with a TTL | Punishing a member for the library being slow | The TTL starts when the copy is ready, never before |
| 17 | The queue head is a READY reservation | Starvation: the queue never advances past them | `peek()` skips non-WAITING |
| 18 | Two reservations by one member on a title | Two places in the queue for one person | Refused on enqueue |
| 19 | A reservation on a title with a free copy | Ambiguous: hold it or leave it | Held, deliberately — it is what a request button does (§5.4) |
| 20 | Two copies of one title to one member | A member with two identical books | Refused: at most one per title per member |
| 21 | The `available_count` cache drifting | A search result that lies | Synced in the critical section; `check_invariants` reconciles |
| 22 | Two threads lazily creating one queue | Reservations split across two queues | Created under the title's lock |
| 23 | Returning an accession that is not on loan | A phantom return | `LoanNotFoundError` |
| 24 | A return dated before the loan | Negative lateness | Refused |
| 25 | Deleting a book with loan history | Closed loans referencing a missing book | Nothing is deleted; copies move to `WITHDRAWN` |
| 26 | A hold TTL expiring while the member is at the desk | The copy handed to the next person mid-collection | The sweeper and the borrow share the title lock |
| 27 | A guest's uncapped fine they cannot pay | An uncollectable debt and a stranded copy | A guest cap well below the replacement cost |
| 28 | A member's loan limit exceeded by concurrent borrows of different titles | One book too many | **Accepted** — soft rule, per-title locking (§6.1 race 10) |

### 8.2 Edge cases worth naming

- **Three copies, one title, one borrowed.** Availability is 2, and the cached count agrees.
- **Returning on the due date.** Zero charge, because the interval is half-open.
- **Returning during a library closure.** The closure day is not counted; a member who could not physically return it is not fined.
- **A due date landing on a closed day.** It never does — `add_open_days` only lands on open days.
- **A grace period spanning a closure.** Still two *open* days of grace.
- **A fine capped by the replacement cost.** A cheap book caps fast, which is correct.
- **A member paying in advance.** The credit is retained; `unpaid_total` floors at zero; `credit_of` reports it.
- **The last copy reserved while available.** The copy is held, which removes it from the walk-in pool. Intentional, and worth stating.
- **Two members reserving at the same instant.** Positions are assigned under the lock; exactly one is first.
- **A reservation whose member is suspended before collection.** The hold expires on its TTL, and the sweeper advances the queue. **Worth naming: the design does not check the member's status at collection time, and a suspended member can still collect a held copy** — which is a real gap and the fix is a status check in `_copy_for`.
- **A copy returned by someone who is not the borrower.** A friend returns it. Allowed — the return is against the copy, not the member — and this is a benefit of the model rather than an oversight.
- **A renewal extending into a closure.** `add_open_days` from today, so the new due date is also an open day.
- **A title with zero copies.** Availability 0, and a reservation is still accepted — which is correct, it is how a "please acquire this" request works.

### 8.3 Complexity

| Operation | Cost | Note |
|---|---|---|
| `add_open_days(start, n)` | O(n) | Walks day by day; n is 7–30, so this is trivial |
| `open_days_between(start, end)` | O(days) | Bounded by the lateness, not by the collection |
| `Catalog.availability(isbn)` | O(copies of the title) | The derived truth |
| `Catalog.find_borrowable_copy(isbn)` | O(copies of the title) | First available wins |
| `Catalog.check_invariants` | **O(books × copies)** | Development and reconciliation only |
| `LoanLedger.active_loans_of(member)` | O(loans) | Linear scan; a real system indexes by member |
| `title_on_loan_to(member, isbn)` | O(loans) | Same |
| `unpaid_total(member)` | O(assessments + payments) | Derived, as it should be |
| `ReservationQueue.enqueue` | O(queue) | The duplicate check scans; a real system indexes |
| `ReservationQueue.peek` | O(queue) | Linear scan for the first WAITING |
| `borrow` | **O(copies + loans) under the title lock** | The lock scope is what matters |
| `return_copy` | O(copies + queue) under the title lock | |
| `sweep_expired_holds` | O(titles × queue), bounded by `batch_limit` | |

**The line to memorise:** *"Every derivation is O(copies) or O(loans) and they are all small and local, which is what makes deriving the truth affordable — the cache on `available_count` exists because search renders it a hundred times per page, and it is synced inside the critical section rather than by a job. The locks are per title because the rules span every copy of a title, and the one rule that is not covered — a member's total loan count — is accepted as a soft rule rather than paid for with a second lock and a deadlock ordering."*

---

## 9. Design patterns used, and why

| Pattern | Where | Why here | What it costs |
|---|---|---|---|
| **Strategy** | `LoanPolicy` → Student, Faculty, Guest | The rules per member type are the axis that varies, and a fourth type must not edit the lending code | An interface and a class per policy |
| **Value Object / record** | `Book`, `BookCopy`, `Loan`, `Reservation`, `FineAssessment` | These are records with identity and no behaviour beyond their own predicates | No in-place mutation of the money fields, which is the point |
| **Repository** | `Catalog`, `LoanLedger` | One place that owns the collections; everything else asks | A wider interface than a direct list |
| **Derived view + reconciled cache** | `Catalog.availability` (truth) and `Book.available_count` (cache) | Search needs it fast; correctness needs it derived | A cache with a `check_invariants` |
| **Queue** | `ReservationQueue` | Fairness, and visible positions | O(n) scans |
| **Lease with a TTL** | A `READY` reservation's `expires_at` plus the sweeper | A hold on a physical copy must expire or the shelf fills | Lazy expiry plus a sweeper |
| **Idempotency key** | *(designed, §7.6)* for self-checkout | A re-scan must not create a second loan | Every caller generates one |
| **Observer** *(rejected)* | — | Considered for notifying the next member. Rejected because there is exactly one interested party and the notification must be sent **after** the lock is released, so a synchronous observer is the wrong shape — a dispatch list is | — |
| **State** *(rejected)* | — | Considered for the copy's status. Rejected because the transitions are data, not behaviour: a copy's status does not decide what happens next, the lending service does. `LoanState` is an enum and `is_overdue` is derived | — |

**Both rejections are worth stating, because they are the two patterns a candidate reaches for on reflex.** Observer is wrong because the timing is wrong — an observer that fires inside the critical section does network I/O under a lock. State is wrong because **a state pattern is for when the state decides the transitions, and here the copy's status is a fact that the service reads, not an agent that acts.** The ATM file uses State for exactly the opposite reason, and the contrast is the lesson.

---

## 10. Trade-offs

| Choice | Buys | Costs |
|---|---|---|
| `Book` and `BookCopy` separate | Availability is a count; copies differ; returns attach to an object; a lost copy is representable | Two collections and a join |
| Availability derived + a synced cache | Search is fast and the truth is derivable | A cache with an invariant check |
| Per-title locking | Borrowing is concurrent; the rules span the title | A lock per title; a soft limit on total loans |
| Open-day due dates | A closure does not fine everyone | A calendar to maintain; O(n) date walks |
| Half-open lateness interval | Returning on time is never fined | One more thing to get right |
| Fines assessed and frozen | A number the member can act on and appeal | An assessment row per late return |
| Fines capped at replacement cost | No absurd debts | Less deterrence than an uncapped fine |
| Overdue derived, not stored | No job to flip it; cannot be wrong | Every consumer must pass `today` |
| Reservation against the title | The returned copy goes to whoever is next | An accession assigned at collection |
| A TTL only once READY | Members are not punished for library slowness | Uncollected ready holds occupy the shelf until swept |
| FIFO with no reordering | The position shown is the truth | Faculty priority needs a separate queue |
| Renewal refused when waiting | The queue advances; the feature works | A member must return and re-borrow |
| A lost copy is never deleted | History resolves; the count stays honest | The catalogue accumulates `LOST` rows |
| Rejecting Observer and State | No I/O under a lock; no state objects that hold no behaviour | The service is a little more procedural |
| Accepting the loan-limit race | No second lock, no ordering problem | A member can occasionally hold one book too many |

**What changes at 10×:** two things, and only two. The **catalogue search** moves to an index and the availability count becomes a genuinely materialised projection, updated from a stream of status-change events, with the reconciliation becoming a scheduled job rather than a development assertion. And the **loan and fine tables** become the large ones, so the O(loans) scans become indexed lookups by member and by accession. **The class design does not change at all** — which is the point to make, because a candidate who starts redesigning the entities at 10× has misdiagnosed where the scale lands.

---

## 11. Interview Q&A

**Q1 (easy). What are the core classes?**
`Book` and `BookCopy` separately, `Member`, `Loan`, `Reservation` with a per-title `ReservationQueue`, `FineAssessment`, `LoanPolicy` with three implementations, a `LibraryCalendar`, a `LoanLedger`, and a `Catalog`. The decision that shapes everything is that a book and a physical copy are different things.

**Q2 (easy). Why are `Book` and `BookCopy` separate classes?**
Because they are different kinds of thing. A `Book` is a bibliographic record — one per title — and a `BookCopy` is a physical object with an accession number, a shelf, and a condition. Four consequences follow immediately: availability is a count of copies rather than a flag; two copies of a title can differ in condition or lending rules; a returned book attaches to the copy that came back rather than to the title; and a lost copy is representable without making the title unavailable. **Almost every wrong answer to this problem starts with one `Book` class with an `is_available` flag, and it has to be rewritten the moment there are two copies of anything.**

**Q3 (easy). Why is the reservation against a title and not a copy?**
Because the member does not care which physical copy they get, and because any specific copy they named might not be the one that comes back first. Reserving the title and assigning the accession at collection means the returned copy goes to whoever is next, which is what the queue is for. **It is the same shape as a hotel reservation against a room type with the room number assigned at check-in** — and the reason is the same: the thing the customer wants is the capability, not the specific instance.

**Q4 (medium). How do you compute the due date?**
Fourteen **open** days from the borrow date, walking a library calendar — not `borrowed_on + timedelta(days=14)`. If the library is closed for a two-day holiday in the loan window, the naive answer is two days early and every member is fined for days they could not have returned the book. The calendar is a set of closed dates and an `add_open_days` that walks forward. **It is a small class, and it is the difference between a library system members trust and one that generates a complaint per closure.**

**Q5 (medium). How do you compute an overdue fine?**
Count the **open** days in the half-open interval `[due_on, returned_on)`, subtract the grace period's open days, multiply by the per-day rate, and cap at the replacement cost. **The half-open interval is the correctness argument: a copy due on the 10th and returned on the 10th charges zero days. A closed interval charges every member who returns on time, which is both the most common bug in a library system and the most infuriating.** And the count is in open days for the same reason the due date is — a member is not fined for a day the library was shut.

**Q6 (medium). *Pushback:* "Why not compute the fine on demand? It is the same formula."**
Because a fine computed on demand from `due_on` and today changes every day it goes unpaid. The member sees one number in the app, a statement a week later shows another, and an appeal has no figure to appeal against. **Assessing it once at return and freezing it makes the fine an immutable fact** — the same argument as an immutable expense in the Splitwise design, and it is not a coincidence that both are money. It also means the assessment can record the return date it was based on, which is what makes an appeal resolvable.

**Q7 (medium). What is the race in this system?**
Two members borrowing the last copy of a title. Both read availability as one, both find the same copy available, both write, and one copy has two loans. The fix is that the read, the decision and the write are all inside one critical section for that title. **And the second race is subtler: a copy returned while a reservation is waiting.** The copy must go to `ON_HOLD_SHELF` for the waiting member — not to `AVAILABLE` with a notification — because the available window is where a walk-in takes the copy the waiting member has been queued three weeks for.

**Q8 (medium). *Pushback:* "Why lock per title? Why not per copy, since that is what you are changing?"**
Because the rules span the title, not the copy. The "one copy per member per title" check, the reservation queue, and the choice of which copy to hand over all have to see every copy of the title atomically. With a per-copy lock, two members can each take a different copy of the same title and both pass a check that was supposed to allow only one — the check and the act are on different things. **And why not one global lock: a library's entire point is that borrowing is concurrent, and a global lock serialises a hundred members at the desk behind every catalogue write.** The lock is sized to the unit the rule spans.

**Q9 (medium). *Pushback:* "Your per-title lock does not cover a member's total loan limit. Two concurrent borrows of different titles can both pass it."**
Correct, and I accept it deliberately. A member's loan limit spans all titles, and a per-title lock cannot hold that set atomically. The consequence is that a member can end up one book over the limit — a benign outcome, not a corrupt ledger. Covering it properly means a per-member lock held alongside the title lock, which reintroduces the deadlock-ordering problem for a rule whose violation is one extra book. **The right shape is per-title locking for throughput, and the total limit enforced as a soft rule with a periodic report, not hardened into a second lock.** If the interviewer insists the limit must be hard, then it becomes a single transaction in a database with row locks on the member and the copies, and the ordering rule returns.

**Q10 (medium). Is availability stored or derived?**
**Both, deliberately, and the honest answer names which is which.** The truth is derived — a count over the copies' statuses — because a stored flag drifts. But catalogue search renders availability for a hundred results per page, so a denormalized `available_count` on `Book` is a cache, and it is written inside the same critical section that changed a copy's status, never by a background job. `check_invariants` proves the two agree. **This is where the derive-everything rule bends, and the reason it bends is that the read frequency is different: a Splitwise balance is read once per screen, an availability count once per search result.**

**Q11 (hard). *Pushback:* "A returned copy goes on a hold shelf. Isn't that just available with a reservation?"**
No, and the difference is a member losing a book. `ON_HOLD_SHELF` means the copy is physically present and **not** available to anyone else — it is promised. If the return marks it `AVAILABLE` and separately notifies the waiting member, there is a window of minutes or hours in which a walk-in at the desk can borrow the very copy held for someone who has been queued for three weeks. **The window is the bug, and it is unrepresentable if the state is a single exclusive status.** It is also why the state machine has that status at all: without it, the promise has nowhere to live.

**Q12 (medium). How do holds expire?**
A `READY` reservation gets an expiry when the copy reaches the hold shelf, and the sweeper reclaims it — moving the copy to the next WAITING member, or back to available if nobody is waiting. **The important detail is that the TTL starts when the copy is ready, not when the member placed the reservation.** A `WAITING` reservation has no TTL, because the member is waiting on the library rather than the other way round, and expiring them for the library being slow is the opposite of what the queue is for. Lazy check plus a sweeper, the same shape as the seat hold.

**Q13 (medium). What happens when a copy is lost?**
The copy moves to `LOST`, the active loan closes as `LOST` rather than `RETURNED`, the member is charged the replacement cost capped by their policy, the available count is synced, and the reservation queue advances because the copy is never coming back. **The one to get right: the replacement charge and an overdue fine are mutually exclusive.** Declaring a copy lost stops the fine from accruing — otherwise a member who loses a book is charged for the book *and* fined for every day of the year they did not return it, which is a double charge and the most likely thing an interviewer probes. If they later find it and return it, the copy goes `LOST → AVAILABLE` and the replacement charge is credited.

**Q14 (hard). *Pushback:* "Your reservation on a title with a free copy removes that copy from the walk-in pool. Isn't that hostile to the people standing in the library?"**
It is a deliberate trade, and it is what a request button does at a real library. A member who reserves a title that is on the shelf is asking for it to be held so they can collect it on the way home. The alternative — refusing and saying "it is on the shelf, come and get it" — means they arrive to find it gone, which is worse. **What makes the trade survivable is the TTL: the hold expires in a few days and the copy comes back to the pool if they never collect.** Without the TTL, holding shelf copies for absent members really would be hostile, which is why the two features are designed together rather than separately.

**Q15 (medium). How do you test this?**
Four layers. The date arithmetic, which is where the subtle bugs are: `add_open_days` over a closure, and the half-open lateness interval at its boundary — returning on the due date charges zero. The fine freeze: return a book, advance the clock a year, assert the assessment has not changed. The invariant checks after every operation: every active loan's copy is `ON_LOAN`, every `ON_LOAN` copy has exactly one active loan, and every cached count matches the derived one. And the concurrency tests: eight threads borrowing one copy with a barrier, asserting exactly one loan; and a mixed workload of borrows, returns and reservations across several titles, asserting every invariant still holds at the end.

---

## 12. Cheat-sheet summary

**The model**

```
  Book (bibliographic) ──1:N── BookCopy (physical) ──0..1── Loan ── Member
      │
      └── ReservationQueue (per TITLE, FIFO)

  A library lends COPIES, not books. One `Book` class with an `is_available`
  flag cannot answer "how many copies?" and has to be rewritten.
```

**Copy status — mutually exclusive, and that is the point**

```
  AVAILABLE · ON_LOAN · ON_HOLD_SHELF · IN_REPAIR · LOST · WITHDRAWN

  ON_HOLD_SHELF is the state that makes the promise representable: the copy is
  physically present and NOT available. Skipping it — going straight to
  AVAILABLE and notifying the member — is the bug where a walk-in takes the
  copy a member has been queued three weeks for.
```

**Dates: open days, not calendar days**

```
  due_on = add_open_days(borrowed_on, policy.loan_days)   ← NOT + timedelta!

  Fine = open_days_between(due_on, returned_on)   ← HALF-OPEN [due, returned)
                       − grace_open_days
                       × fine_per_open_day
         capped at min(replacement_cost, policy cap)

  Returning ON the due date charges 0. A closed interval fines everyone who
  returns on time — the most common and most infuriating bug in this domain.
  A closure day costs the member nothing.
```

**The rules, per policy**

| | Student | Faculty | Guest |
|---|---|---|---|
| Loan days (open) | 14 | 30 | 7 |
| Max loans | 4 | 10 | 2 |
| Max renewals | 2 | 5 | 0 |
| Fine / open day | ₹5 | ₹2 | ₹10 |
| Grace (open days) | 2 | 7 | 0 |
| Cap | replacement cost | replacement cost | ₹200 |

**The two races**

```
  1. TWO MEMBERS, LAST COPY
     Both read availability == 1, both write → one copy, two loans.
     FIX: read + decide + write in ONE critical section for the TITLE.

  2. A RETURN WHILE A RESERVATION WAITS
     Marking it AVAILABLE is a window in which a walk-in takes it.
     FIX: the return decides the destination inside the same lock:
          WAITING reservation → ON_HOLD_SHELF + a TTL, for THAT member
          nobody waiting      → AVAILABLE + sync the cache

  LOCK GRANULARITY: per TITLE (the rules span the copies), acquired by sorted
  ISBN when more than one title is involved.
```

**Reservations**

```
  WAITING ──copy returned──▶ READY ──collected──▶ FULFILLED
     │                        │
     │ no TTL                 │ TTL from when the copy became ready
     └────────────────────────┴──▶ EXPIRED (sweeper advances the queue)

  peek() returns the earliest WAITING — NOT the first open one, or a READY
  head starves the queue.
  A renewal is REFUSED when has_waiting(). That rule is what makes the queue
  mean anything.
```

**Money**

```
  FineAssessment: assessed ONCE at return, FROZEN. Never recomputed.
  A fine that changes while the member looks at it is not a fine.
  The member's balance is DERIVED from unpaid assessments minus payments.
  Lost copy: replacement charge and overdue fine are MUTUALLY EXCLUSIVE.
```

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "Model the book" | "Two classes: `Book` and `BookCopy`. Availability is a count of copies, and the loan attaches to a copy." |
| "Is it available?" | "The wrong question. How many copies are on the shelf?" |
| "Compute the due date" | "Fourteen OPEN days over a calendar, not `+ timedelta`. Otherwise a closure fines everyone." |
| "Compute the fine" | "Open days in the half-open `[due_on, returned_on)`, minus grace, capped. Returning on time charges zero." |
| "Recompute the fine later?" | "No. Assessed once at return and frozen, or the member sees a number that grows while they look at it." |
| "Two members, last copy" | "Per-title lock; read and write in one critical section." |
| "A copy comes back with a reservation" | "`ON_HOLD_SHELF` for that member, not `AVAILABLE`. The available window is where the copy is stolen." |
| "Renew while someone waits?" | "Refused. Without that rule the queue never advances and the feature does not work." |
| "Reserve a title or a copy?" | "The title. The accession is assigned at collection — the same as a hotel room type." |
| "Concurrency gaps?" | "The per-title lock does not cover a member's total loan limit. Accepted as a soft rule, and I will say why." |

---

## Cross-links

- The shared-state mechanics this file assumes, and the lock-granularity reasoning: [Concurrency and threading in LLD](../04-concurrency-and-threading-in-lld.md)
- The same per-unit lock sizing: [BookMyShow seat locking](04-bookmyshow-seat-locking.md) · [Hotel management](03-hotel-management.md) · and the same lease-with-a-TTL plus sweeper: [LRU and LFU cache](02-lru-lfu-cache.md) · [Elevator system](05-elevator-system.md) · [Parking lot](01-parking-lot.md)
- The same frozen-assessment money discipline and the same derived-balance ledger: [Splitwise](06-splitwise.md) · and the same state-machine-versus-derived-status contrast: [ATM machine](07-atm-machine.md)
- The class concepts behind this design: [OOP principles and SOLID](../02-oop-principles-and-solid.md) · [Design patterns](../03-design-patterns.md) · [UML and class diagrams](../05-uml-and-class-diagrams.md) · [The machine-coding framework](../06-machine-coding-interview-framework.md) · [What is LLD](../01-what-is-lld-and-how-to-answer.md)
- Where the class view ends: [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md) · the durable store behind the ledger: [SQL deep dive](../../HLD/04-data/01-sql-deep-dive.md) · and where the catalogue search lives at scale: [Choosing a database](../../HLD/04-data/05-choosing-a-database-decision-guide.md) · [Caching strategies and layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md)
