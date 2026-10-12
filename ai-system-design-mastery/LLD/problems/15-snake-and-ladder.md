# LLD 15 — Snake and Ladder

> **Grounding note.** The reference repositories contain **no OOP/LLD material**; the source transcripts do not cover snake and ladder as a design problem. This file is written from first principles and is not transcript-grounded. The answer shape follows the house template at [LLD/01 What is LLD and How to Answer](../01-what-is-lld-and-how-to-answer.md).

> **TL;DR** — Snake and ladder is not a hard *game*; it is a hard *configuration* problem. The whole design turns on four decisions: the **board is validated data, not code** (a jump graph with cycles, duplicates, a jump onto the last square and a ladder that lands on a snake are all silent bugs that make the game unplayable or instantly won); the **dice is an injected port** (without it the game is untestable, and every interview answer that hard-codes `random.randint` is untestable); **move resolution is a pure function** `resolve(position, roll) -> MoveResolution` separate from the mutation, so the three win-rule variants and the chaining question are unit-testable in isolation; and the **turn manager owns the extra-roll-on-six rule together with the three-sixes forfeit**, because without the pair the game can fail to terminate — a liveness property, not a nicety. The sentence that wins the interview: *"a six granting another roll is not a rule about dice, it is a rule about the turn, and it needs a bound or the game may never end."*

> **60-second interview answer:** "A `Board` holding a validated `Jump` graph, a `Dice` port with a random implementation and a scripted one, a `MoveResolver` that is pure — it takes the board, the position and the roll and returns a resolution with the ladder and snake hops recorded — a `WinRule` strategy covering the three variants people actually ask for (exact landing, overshoot wins, bounce back), and a `TurnManager` that owns turn order, the extra roll on a six and the three-sixes forfeit. The `Game` orchestrates and emits a `RollOutcome` per roll, which makes the whole game replayable from a log. I would validate the board at construction: no duplicate jump starts, no cycles, no jump from or onto the last square, and no chains unless the game asks for them."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **Board size and the jump layout?** | Whether the board is hard-coded or configuration | Configuration data, validated; 100 squares by default |
| 2 | **Where does the layout come from?** | Constructor, file, or a random generator | Constructor for tests; a loader is an extension |
| 3 | **Exact landing, overshoot wins, or bounce back?** | The win rule — three real variants | Exact landing, as a strategy so the others are one class |
| 4 | **Does a six grant another roll?** | Whether turns are "one roll" or "roll until you don't roll a six" | Yes, with the three-sixes forfeit attached |
| 5 | **Can a ladder land on a snake?** | Whether jump resolution chains, and whether validation forbids it | Forbid it at validation; resolve chains defensively anyway |
| 6 | **How many players, and can one sit out?** | Turn order and skipping finished players | Two or more, round-robin, finished players skipped |
| 7 | **How is the dice injected?** | Whether the game is testable at all | A `Dice` port; a scripted implementation for tests |
| 8 | **Is there a turn or game limit?** | Whether the game can run forever | A turn cap, because a loaded dice can loop |
| 9 | **Do you need the roll history?** | Whether a `RollOutcome` log exists | Yes — it makes the game replayable and debuggable |
| 10 | **One game or many at once?** | Locking and dice sharing | One game per object, one lock per game, one dice per game |

### The defaults to assume, stated plainly

> "I'll assume a **configurable board** validated at construction, an **injected dice port** with a scripted implementation so the game is deterministic under test, **exact landing** as the win rule with the other two variants as interchangeable strategies, **an extra roll on a six with a three-sixes forfeit**, jump chaining **forbidden by validation but defended in the resolver**, and a per-roll `RollOutcome` log. Does that match, or do you want the minimal 100-square version first?"

### Non-goals

- No network play, no persistence, no UI beyond a text renderer.
- No AI — there is no decision to make; the only player action is "roll".

### The four constraints that shape the design

1. **The board must be validated, not trusted.** A cycle hangs the game; a jump onto the last square wins instantly; a duplicate start silently drops a snake. These are configuration bugs that look like gameplay.
2. **The dice must be a port.** Otherwise the game cannot be tested, and "the game is a class" is a claim you cannot demonstrate.
3. **Move resolution must be pure and separate from mutation.** Chaining, overshoot and bounce-back are exactly where the bugs are, and they must be testable without running a game.
4. **The extra-roll rule needs a bound.** "Roll again on a six" makes the number of rolls per turn unbounded; the three-sixes forfeit is what makes it terminate.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `Jump` | Frozen value object | `start`, `end`; `is_ladder` / `is_snake` | Whether the jump is legal *in context* |
| `Board` | Validated configuration | `size`, the jump map, `destination`, `validate()` | Movement rules |
| `BoardConfigError` | Exception | Names the exact defect and the offending jump | — |
| `Dice` (ABC) | Port | `roll() -> int` | Turn rules |
| `RandomDice` | Implementation | Uniform 1..sides from an injected `random.Random` | — |
| `ScriptedDice` | Implementation | A fixed list of rolls, then a default — the test double | — |
| `LoadedDice` | Implementation | Always the same value; the liveness adversary | — |
| `Player` | Mutable entity | `name`, `position` | Whose turn it is |
| `MoveOutcome` | Enum | `MOVED`, `OVERSHOT`, `BOUNCED`, `WON` | — |
| `MoveResolution` | Frozen value object | From, roll, landing, hops, final, outcome | Applying itself |
| `WinRule` (ABC) | Strategy | `resolve(board, position, roll) -> MoveResolution` — **pure** | Mutating a player |
| `ExactLandingRule` | Implementation | Must land on the last square | — |
| `OvershootWinsRule` | Implementation | Passing the end wins | — |
| `BounceBackRule` | Implementation | Reflect off the end | — |
| `TurnManager` | Policy | Order, skipping, extra roll on six, three-sixes forfeit | Rolling |
| `RollOutcome` | Frozen value object | One row of the game log | — |
| `Game` | Aggregate root | Orchestrates a roll, applies a resolution, emits outcomes | Rules, dice, I/O |
| `GameResult` | Frozen value object | Winner, turn count, the log | — |

**The split to say out loud:** *"`WinRule` resolves a move but does not apply it. That is the key seam: resolution is a pure function of (board, position, roll), so the three variants and the chaining question are unit tests rather than twenty-minute simulations."*

---

## 3. Class diagram (ASCII)

```
   ┌───────────────┐         ┌──────────────────────────────────────┐
   │   Jump        │         │              Board                   │
   │ «value»       │◄────────│──────────────────────────────────────│
   │ start, end    │  many   │ size : int                           │
   │ +is_ladder    │         │ jumps : {start: Jump}                │
   │ +is_snake     │         │──────────────────────────────────────│
   └───────────────┘         │ +destination(pos) -> int             │
                             │ +validate() -> None   ← THE HARD PART │
   ┌───────────────┐         └──────────────────────────────────────┘
   │ «exception»   │                        ▲ raises
   │ BoardConfig   │────────────────────────┘
   │ Error         │   duplicate start · cycle · off-board ·
   └───────────────┘   jump onto the last square · chained jump
        │
        │  ┌──────────────┐        ┌───────────────────────────────┐
        └─►│   «interface»│        │        MoveResolution         │
           │     Dice     │        │ «value, pure output»          │
           │──────────────│        │ from, roll, landing, hops,    │
           │ +roll() ->int│        │ final, outcome                │
           └──────▲───────┘        └───────────────────────────────┘
                  │                              ▲ returns
   ┌──────────────┼───────────┐                  │
   │              │           │      ┌───────────┴───────────────┐
┌──┴───────┐ ┌────┴──────┐ ┌──┴─────┐│      «interface»          │
│Random    │ │Scripted   │ │Loaded  ││        WinRule            │
│Dice      │ │Dice       │ │Dice    ││───────────────────────────│
│rng       │ │rolls[]    │ │value   ││ +resolve(board, position, │
└──────────┘ └───────────┘ └────────┘│    roll) -> MoveResolution│
                                     └───────────▲───────────────┘
                    ┌────────────────┬───────────┴────────┬──────────────┐
              ┌─────┴──────┐  ┌──────┴───────┐  ┌──────────┴─────────┐
              │ExactLanding│  │OvershootWins │  │BounceBackRule      │
              │(default)   │  │              │  │                    │
              └────────────┘  └──────────────┘  └────────────────────┘

┌───────────────────────────────┐        ┌──────────────────────────────┐
│      TurnManager              │        │            Game              │
│───────────────────────────────│        │──────────────────────────────│
│ order : [Player]              │◄───────│ board, dice, rule, turn_mgr  │
│ index : int                   │  uses  │ players, log, turn_cap       │
│ streak : int (consecutive 6s) │        │──────────────────────────────│
│ turn_start_pos : {name: int}  │        │ +step() -> RollOutcome        │
│───────────────────────────────│        │ +play() -> GameResult         │
│ +current() -> Player          │        │ -_apply(resolution)           │
│ +advance()                    │        │  emits ONE RollOutcome per    │
│ +note_roll(roll) -> Decision  │        │  roll — the game log         │
│   EXTRA · NEXT · FORFEIT      │        └──────────────────────────────┘
└───────────────────────────────┘                     │ produces
                                                      ▼
                                          ┌──────────────────────────┐
                                          │      RollOutcome         │
                                          │ «value» player, roll,    │
                                          │ from, to, hops, outcome, │
                                          │ extra_turn, forfeited,   │
                                          │ won                      │
                                          │  → replayable log        │
                                          └──────────────────────────┘
```

**Read it this way:** `Game` composes a `Board` (validated data), a `Dice` (a port), a `WinRule` (a pure strategy) and a `TurnManager` (the turn policy). The only object that mutates anything is `Game`, and it emits a `RollOutcome` for every roll.

---

## 4. Interfaces / abstract classes / enums

```python
class Dice(ABC):
    @abstractmethod
    def roll(self) -> int:
        """Return a value in 1..sides inclusive."""


class WinRule(ABC):
    @abstractmethod
    def resolve(self, board: Board, position: int, roll: int) -> "MoveResolution":
        """PURE: compute the whole move — landing, hops and outcome — without
        touching any player. Everything that varies between rule variants
        lives behind this one method."""


class Decision(Enum):
    EXTRA = "extra"     # roll again (a six)
    NEXT = "next"       # turn passes
    FORFEIT = "forfeit" # three sixes: the turn's moves are undone
```

**Three contract notes worth saying out loud:**

- **`resolve` is pure.** It returns a value describing the move; it does not move anybody. That is what makes overshoot, bounce-back and chaining unit-testable without a game loop.
- **`Dice.roll` is a port, not a call to `random`.** The scripted implementation is not a testing convenience, it is the thing that makes the design demonstrable.
- **`Decision` is an enum, not a boolean.** "Do you roll again?" is not a yes/no question — the third case, a forfeit that *undoes* the turn's moves, has to be representable or the three-sixes rule cannot be expressed.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** Interview languages vary — in Java this is an interface plus a `HashMap<Integer, Jump>`; in Go, interfaces and a map; in C++, `virtual` and `std::map`. The **validated board, the dice port and the pure resolver** are the design; the syntax is not. Python 3.10+ for `X | None`.

```python
"""snake_and_ladder.py — a validated board, an injected dice, a pure resolver.

Design summary
--------------
Board              : validated CONFIGURATION. validate() rejects the five
                     defects that make a board unplayable — duplicate jump
                     starts, cycles, jumps off the board, a jump onto the last
                     square (an instant win), and a chained jump (a ladder
                     whose landing is a snake's head).
Dice (port)        : Random / Scripted / Loaded. Scripted is what makes the
                     whole game deterministic under test; Loaded is the
                     adversary that proves the turn cap matters.
WinRule (strategy) : resolve(board, position, roll) -> MoveResolution is PURE.
                     ExactLanding / OvershootWins / BounceBack are the three
                     variants interviewers ask for, and each is one class.
TurnManager        : order, skipping finished players, the extra roll on a
                     six, and the three-sixes forfeit that bounds it.
Game               : the only object that mutates anything. Emits one
                     RollOutcome per roll, so a game is replayable from a log.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Optional


# --------------------------------------------------------------------------
# 1. Values and configuration
# --------------------------------------------------------------------------

class BoardConfigError(Exception):
    """Raised by Board.validate(). The message names the defect AND the
    offending jump, because a configuration error without the offending entry
    is a support ticket."""


@dataclass(frozen=True)
class Jump:
    start: int
    end: int

    @property
    def is_ladder(self) -> bool:
        return self.end > self.start

    @property
    def is_snake(self) -> bool:
        return self.end < self.start

    def __str__(self) -> str:
        kind = "ladder" if self.is_ladder else "snake"
        return f"{kind} {self.start}->{self.end}"


class Board:
    """Squares 1..size. Position 0 means "not on the board yet", which is the
    start before the first roll.

    The jumps are DATA. `validate()` is the interesting method: the five
    defects below are the bugs that make a game look broken rather than
    crash, which is the worst kind.
    """

    def __init__(self, size: int = 100, jumps: Optional[Iterable[Jump]] = None) -> None:
        if size < 2:
            raise BoardConfigError("a board needs at least two squares")
        self.size = size
        self.jumps: dict[int, Jump] = {}
        for jump in jumps or ():
            if jump.start in self.jumps:
                raise BoardConfigError(
                    f"two jumps start at {jump.start}: "
                    f"{self.jumps[jump.start]} and {jump}")
            self.jumps[jump.start] = jump

    # -- queries -----------------------------------------------------------
    def destination(self, position: int) -> int:
        """One hop. The resolver applies this repeatedly to handle chains."""
        jump = self.jumps.get(position)
        return jump.end if jump is not None else position

    def on_board(self, position: int) -> bool:
        return 0 <= position <= self.size

    # -- THE validation ----------------------------------------------------
    def validate(self, allow_chains: bool = False) -> "Board":
        """Reject every defect that produces a silently broken game.

        Returns self so a caller can write `Board(100, jumps).validate()`.
        """
        for start, jump in self.jumps.items():
            # 1. Both ends must be on the board, and a jump must actually move.
            if not self.on_board(start) or start == 0:
                raise BoardConfigError(f"{jump}: {start} is off the board")
            if not self.on_board(jump.end) or jump.end == 0:
                raise BoardConfigError(f"{jump}: {jump.end} is off the board")

            # 2. A jump onto the LAST square is an instant win that skips the
            #    whole game — a ladder from square 2 to 100 ends the game on
            #    the first roll.
            if jump.end == self.size:
                raise BoardConfigError(
                    f"{jump}: a jump onto the last square wins instantly")

            # 3. A jump OFF the last square is unreachable in a correct game,
            #    but it indicates the layout was generated against another size.
            if start == self.size:
                raise BoardConfigError(
                    f"{jump}: nothing can start on the final square")

            if jump.end == jump.start:
                raise BoardConfigError(f"{jump}: a jump that does not move")

        # 4. Cycles — a chain that returns to where it started. This is the one
        #    that HANGS the game, so it is checked as a graph walk rather than
        #    per jump, and it is checked BEFORE the chain rule because a
        #    two-jump cycle is also a chain and "cycle" is the more useful
        #    message.
        for start in self.jumps:
            seen: set[int] = set()
            node = start
            while node in self.jumps:
                if node in seen:
                    raise BoardConfigError(
                        f"the jumps form a cycle through square {node}")
                seen.add(node)
                node = self.jumps[node].end

        # 5. Chains — a ladder whose landing is also a jump start. Allowed
        #    only if the game explicitly asks for them.
        if not allow_chains:
            for start, jump in self.jumps.items():
                if jump.end in self.jumps:
                    raise BoardConfigError(
                        f"{jump}: lands on {self.jumps[jump.end]}, so the move "
                        f"would chain; pass allow_chains=True if that is intended")
        return self

    def describe(self) -> str:
        ladders = sorted((j for j in self.jumps.values() if j.is_ladder),
                         key=lambda j: j.start)
        snakes = sorted((j for j in self.jumps.values() if j.is_snake),
                        key=lambda j: j.start)
        return (f"board 1..{self.size}: {len(ladders)} ladders, "
                f"{len(snakes)} snakes")


# --------------------------------------------------------------------------
# 2. The dice — a port, not a call to random
# --------------------------------------------------------------------------

class Dice(ABC):
    def __init__(self, sides: int = 6) -> None:
        if sides < 2:
            raise ValueError("a die needs at least two sides")
        self.sides = sides

    @abstractmethod
    def roll(self) -> int: ...


class RandomDice(Dice):
    """The production implementation. The RNG is injected so a test can fix
    the sequence at the source rather than mocking the dice."""

    def __init__(self, sides: int = 6, rng: Optional[random.Random] = None) -> None:
        super().__init__(sides)
        self.rng = rng if rng is not None else random.Random()

    def roll(self) -> int:
        return self.rng.randint(1, self.sides)


class ScriptedDice(Dice):
    """The test double, and the reason the whole design is demonstrable: a
    fixed list of rolls makes an entire game deterministic."""

    def __init__(self, rolls: Iterable[int], sides: int = 6,
                 default: Optional[int] = None) -> None:
        super().__init__(sides)
        self.rolls = list(rolls)
        self.default = default if default is not None else sides
        self.index = 0

    def roll(self) -> int:
        if self.index < len(self.rolls):
            value = self.rolls[self.index]
            self.index += 1
        else:
            value = self.default
        if not 1 <= value <= self.sides:
            raise ValueError(f"scripted roll {value} is outside 1..{self.sides}")
        return value


class LoadedDice(Dice):
    """Always the same value. With sixes and an extra-roll rule this is the
    adversary that makes the game non-terminating, which is why the turn cap
    and the three-sixes forfeit exist."""

    def __init__(self, value: int = 6, sides: int = 6) -> None:
        super().__init__(sides)
        self.value = value

    def roll(self) -> int:
        return self.value


# --------------------------------------------------------------------------
# 3. Move resolution — pure, and where the variants live
# --------------------------------------------------------------------------

class MoveOutcome(Enum):
    MOVED = "moved"          # advanced, possibly via jumps
    OVERSHOT = "overshot"    # the roll went past the end and did not move
    BOUNCED = "bounced"      # the roll reflected off the end
    WON = "won"


@dataclass(frozen=True)
class MoveResolution:
    position: int
    roll: int
    landing: int                          # before any jump
    hops: tuple[Jump, ...] = ()           # the ladders and snakes taken
    final: int = 0
    outcome: MoveOutcome = MoveOutcome.MOVED

    @property
    def moved(self) -> bool:
        return self.final != self.position

    def __str__(self) -> str:
        journey = " -> ".join(str(h.end) for h in self.hops)
        via = f" via {journey}" if journey else ""
        return (f"{self.position} +{self.roll} = {self.landing}{via} "
                f"=> {self.final} ({self.outcome.value})")


class WinRule(ABC):
    @abstractmethod
    def resolve(self, board: Board, position: int, roll: int) -> MoveResolution:
        """PURE. Compute the move; do not move anybody."""

    # -- shared helper -----------------------------------------------------
    def _apply_jumps(self, board: Board, landing: int,
                     max_hops: int = 8) -> tuple[int, tuple[Jump, ...]]:
        """Walk the jump graph from the landing square.

        Validation forbids chains and cycles, so a correct board needs at most
        one hop — but the resolver defends anyway, because a board built
        without validate() is exactly the case that hangs, and `max_hops` turns
        an infinite loop into a reported anomaly.
        """
        hops: list[Jump] = []
        node = landing
        for _ in range(max_hops):
            jump = board.jumps.get(node)
            if jump is None:
                return node, tuple(hops)
            hops.append(jump)
            node = jump.end
        raise BoardConfigError(
            f"jump chain from {landing} exceeded {max_hops} hops; "
            f"call Board.validate()")


class ExactLandingRule(WinRule):
    """The default: you must land exactly on the last square. Nothing else
    counts as a win, and an overshoot does not move you at all."""

    def resolve(self, board: Board, position: int, roll: int) -> MoveResolution:
        target = position + roll
        if target > board.size:
            return MoveResolution(position, roll, target,
                                  outcome=MoveOutcome.OVERSHOT, final=position)
        if target == board.size:
            return MoveResolution(position, roll, target,
                                  outcome=MoveOutcome.WON, final=target)
        final, hops = self._apply_jumps(board, target)
        outcome = MoveOutcome.WON if final == board.size else MoveOutcome.MOVED
        return MoveResolution(position, roll, target, hops, final, outcome)


class OvershootWinsRule(WinRule):
    """Any roll that reaches or passes the last square wins. Jumps are ignored
    on the winning move, because the game is over."""

    def resolve(self, board: Board, position: int, roll: int) -> MoveResolution:
        target = position + roll
        if target >= board.size:
            return MoveResolution(position, roll, target,
                                  outcome=MoveOutcome.WON, final=board.size)
        final, hops = self._apply_jumps(board, target)
        outcome = MoveOutcome.WON if final == board.size else MoveOutcome.MOVED
        return MoveResolution(position, roll, target, hops, final, outcome)


class BounceBackRule(WinRule):
    """The roll reflects off the last square: from 98 with a 5, you land on 97
    (98 -> 100 is 2, the remaining 3 comes back to 97).

    A SINGLE reflection is not a correct implementation. It is right for a d6
    on a 100-square board, and wrong the moment the roll can overshoot by more
    than the board length: from 8 with a roll of 25 on a size-10 board, one
    reflection gives 10 - (33 - 10) = -13, which is off the board entirely.
    The correct model is a triangular wave, which is what _fold computes.
    """

    def resolve(self, board: Board, position: int, roll: int) -> MoveResolution:
        target = position + roll
        if target == board.size:
            return MoveResolution(position, roll, target,
                                  outcome=MoveOutcome.WON, final=target)
        if target > board.size:
            folded = _fold(target, board.size)
            # Standing on the last square ends the game even when you arrived
            # by reflection. The alternative — a bounce that is not a win —
            # parks a player on 100 who then overshoots on every later roll and
            # can never finish.
            if folded == board.size:
                return MoveResolution(position, roll, target,
                                      outcome=MoveOutcome.WON, final=folded)
            final, hops = self._apply_jumps(board, folded)
            outcome = MoveOutcome.WON if final == board.size else MoveOutcome.BOUNCED
            return MoveResolution(position, roll, target, hops, final, outcome)
        final, hops = self._apply_jumps(board, target)
        outcome = MoveOutcome.WON if final == board.size else MoveOutcome.MOVED
        return MoveResolution(position, roll, target, hops, final, outcome)


def _fold(position: int, size: int) -> int:
    """Fold a position on an infinite track onto squares 1..size, reversing at
    each end — the triangular wave a bounce-back rule describes.

    From 8 with a roll of 25 on a size-10 board: forward 8->10 (2 steps), back
    10->1 (9 steps), forward 1->10 (9), back 10->5 (5) — 25 steps lands on 5.
    The modular form below gives the same answer without the loop.
    """
    if size < 2:
        raise BoardConfigError("cannot fold onto a board smaller than 2")
    span = size - 1
    offset = (position - 1) % (2 * span)
    return 1 + (offset if offset <= span else 2 * span - offset)


# --------------------------------------------------------------------------
# 4. Turn policy
# --------------------------------------------------------------------------

class Decision(Enum):
    EXTRA = "extra"
    NEXT = "next"
    FORFEIT = "forfeit"


@dataclass
class Player:
    name: str
    position: int = 0
    finished: bool = False

    def __str__(self) -> str:
        return f"{self.name}@{self.position}"


class TurnManager:
    """Order, skipping, and the two rules about sixes.

    The extra roll on a six is NOT a rule about dice; it makes the number of
    rolls in a turn unbounded. The three-sixes forfeit is what re-establishes
    the bound, and the pair has to live together or the rule set is unsound.
    """

    def __init__(self, order: list[Player], max_consecutive_sixes: int = 3) -> None:
        if not order:
            raise ValueError("a game needs at least one player")
        self.order = order
        self.index = 0
        self.max_consecutive_sixes = max_consecutive_sixes
        self.streak = 0
        self.turn_number = 1

    def current(self) -> Player:
        """Advance past finished players. If every player is finished this
        raises, which cannot happen while a game stops at the first winner —
        but it is the guard that turns a stuck loop into an error."""
        for _ in range(len(self.order) + 1):
            player = self.order[self.index]
            if not player.finished:
                return player
            self.index = (self.index + 1) % len(self.order)
        raise RuntimeError("every player is finished")

    def advance(self) -> None:
        self.streak = 0
        self.turn_number += 1
        self.index = (self.index + 1) % len(self.order)

    def note_roll(self, roll: int) -> Decision:
        """The single place the six rules live."""
        if roll != 6:
            self.streak = 0
            return Decision.NEXT
        self.streak += 1
        if self.streak >= self.max_consecutive_sixes:
            return Decision.FORFEIT
        return Decision.EXTRA


# --------------------------------------------------------------------------
# 5. The log and the game
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RollOutcome:
    """One row of the game log. A list of these replays the entire game, which
    is what makes a failure report actionable."""

    turn: int
    player: str
    roll: int
    from_position: int
    to_position: int
    hops: tuple[Jump, ...]
    outcome: MoveOutcome
    decision: Decision
    forfeited: bool = False
    won: bool = False

    def __str__(self) -> str:
        via = " " + " ".join(str(h) for h in self.hops) if self.hops else ""
        flag = " WON" if self.won else (" FORFEIT" if self.forfeited else "")
        return (f"t{self.turn} {self.player} rolled {self.roll}: "
                f"{self.from_position}->{self.to_position}{via} "
                f"[{self.outcome.value}/{self.decision.value}]{flag}")


@dataclass(frozen=True)
class GameResult:
    winner: Optional[Player]
    turns: int
    log: tuple[RollOutcome, ...]

    @property
    def truncated(self) -> bool:
        """True when the game hit the cap rather than producing a winner."""
        return self.winner is None


class Game:
    def __init__(self, board: Board, dice: Dice, players: list[Player],
                 rule: Optional[WinRule] = None,
                 turn_manager: Optional[TurnManager] = None,
                 turn_cap: int = 10_000,
                 validate_board: bool = True) -> None:
        if validate_board:
            board.validate()
        self.board = board
        self.dice = dice
        self.rule = rule if rule is not None else ExactLandingRule()
        self.turns = turn_manager if turn_manager is not None else TurnManager(players)
        self.players = self.turns.order
        self.turn_cap = turn_cap
        self.log: list[RollOutcome] = []
        self.winner: Optional[Player] = None
        self._turn_start: dict[str, int] = {}

    # -- one roll ----------------------------------------------------------
    def step(self) -> RollOutcome:
        """Exactly one die roll, one resolution and one log row. `play()` is a
        loop over this, and so is any interactive UI — which is why the
        per-roll step is public and the loop is not the only entry point."""
        player = self.turns.current()

        # Remember where the turn began, so a three-sixes forfeit can undo the
        # whole turn rather than just ending it.
        if self.turns.streak == 0:
            self._turn_start[player.name] = player.position

        roll = self.dice.roll()
        decision = self.turns.note_roll(roll)
        before = player.position

        if decision is Decision.FORFEIT:
            # The classic rule: three sixes in a turn and the turn's moves are
            # undone. Note this UNDOES, which is why it cannot be a boolean.
            player.position = self._turn_start.get(player.name, before)
            row = RollOutcome(self.turns.turn_number, player.name, roll,
                              before, player.position, (), MoveOutcome.MOVED,
                              decision, forfeited=True)
            self.log.append(row)
            self.turns.advance()
            return row

        resolution = self.rule.resolve(self.board, before, roll)

        if resolution.outcome is MoveOutcome.OVERSHOT:
            player.position = resolution.final        # unchanged, by the rule
        else:
            player.position = resolution.final

        won = resolution.outcome is MoveOutcome.WON and player.position == self.board.size
        if won:
            player.finished = True
            # Track the winner HERE rather than reading it from the turn
            # manager afterwards. current() skips finished players, so asking
            # it "whose turn is it?" after the win returns the NEXT player —
            # a bug that reports the loser as the winner.
            self.winner = player

        row = RollOutcome(self.turns.turn_number, player.name, roll, before,
                          player.position, resolution.hops, resolution.outcome,
                          decision, won=won)
        self.log.append(row)

        if not won:
            if decision is Decision.NEXT:
                self.turns.advance()
            # EXTRA: the same player rolls again; the streak persists so the
            # three-sixes rule can fire on the third.
        return row

    # -- the whole game ----------------------------------------------------
    def play(self) -> GameResult:
        """Loop until somebody wins or the cap is reached. The cap is not
        decoration: with a loaded dice and an extra-roll rule, a game can run
        forever, and a test that hangs is worse than a test that fails."""
        rolls = 0
        while rolls < self.turn_cap:
            row = self.step()
            rolls += 1
            if row.won:
                return GameResult(self.winner, self.turns.turn_number,
                                  tuple(self.log))
        return GameResult(None, self.turns.turn_number, tuple(self.log))

    def render(self) -> str:
        return "\n".join(str(p) for p in self.players)


# --------------------------------------------------------------------------
# 6. Tests and demo — correctness proved, not asserted
# --------------------------------------------------------------------------

def _check(name: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    if not condition:
        raise SystemExit(1)


def _raises(fn) -> Optional[str]:
    """Return the BoardConfigError message, or None if nothing was raised."""
    try:
        fn()
    except BoardConfigError as exc:
        return str(exc)
    return None


def _raises_value(fn) -> Optional[str]:
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    return None


def _raises_runtime(fn) -> Optional[str]:
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    return None


def _group_by_turn(log: Iterable[RollOutcome]) -> list[tuple[int, list[RollOutcome]]]:
    """Consecutive rows sharing a turn number. Used to prove the extra-roll
    rule is bounded: no turn may contain more rolls than the six limit."""
    groups: list[tuple[int, list[RollOutcome]]] = []
    for row in log:
        if groups and groups[-1][0] == row.turn:
            groups[-1][1].append(row)
        else:
            groups.append((row.turn, [row]))
    return groups


if __name__ == "__main__":
    # --- 1. the five board defects are all rejected ---
    msg = _raises(lambda: Board(100, [Jump(10, 30), Jump(10, 40)]))
    _check("a duplicate jump start is rejected", msg is not None and "two jumps" in msg)

    msg = _raises(lambda: Board(100, [Jump(10, 30), Jump(30, 10)]).validate())
    _check("a cycle is rejected", msg is not None and "cycle" in msg)

    msg = _raises(lambda: Board(100, [Jump(10, 101)]).validate())
    _check("a jump off the board is rejected", msg is not None and "101 is off" in msg)

    msg = _raises(lambda: Board(100, [Jump(10, 100)]).validate())
    _check("a jump onto the last square is rejected",
           msg is not None and "instantly" in msg)

    msg = _raises(lambda: Board(100, [Jump(10, 20), Jump(20, 5)]).validate())
    _check("a chained jump is rejected by default",
           msg is not None and "chain" in msg)
    _check("but is allowed when the game asks for chains",
           Board(100, [Jump(10, 20), Jump(20, 5)]).validate(allow_chains=True)
           is not None)

    msg = _raises(lambda: Board(100, [Jump(50, 50)]).validate())
    _check("a jump that does not move is rejected",
           msg is not None and "does not move" in msg)

    msg = _raises(lambda: Board(100, [Jump(100, 50)]).validate())
    _check("a jump starting on the final square is rejected",
           msg is not None and "final square" in msg)

    # --- 2. a valid board, and the two directions ---
    board = Board(100, [
        Jump(1, 38), Jump(4, 14), Jump(9, 31), Jump(21, 42), Jump(28, 84),
        Jump(36, 44), Jump(51, 67), Jump(71, 91), Jump(80, 100 - 1),
        Jump(98, 78), Jump(95, 75), Jump(93, 73), Jump(87, 24), Jump(64, 60),
        Jump(62, 19), Jump(49, 11), Jump(48, 26), Jump(16, 6),
    ]).validate()
    _check("a realistic 100-square board validates", board.size == 100)
    _check("the ladder at 28 goes up", board.destination(28) == 84)
    _check("the snake at 87 goes down", board.destination(87) == 24)
    _check("a square with no jump stays put", board.destination(50) == 50)
    _check("the description counts both kinds",
           "ladders" in board.describe() and "snakes" in board.describe())

    # --- 3. ExactLanding: a ladder lifts, a snake bites ---
    rule = ExactLandingRule()
    r = rule.resolve(board, 26, 2)                     # 28 is a ladder to 84
    _check("landing on a ladder climbs it",
           r.landing == 28 and r.final == 84 and len(r.hops) == 1)
    _check("the hop is recorded for the log", str(r.hops[0]) == "ladder 28->84")
    r = rule.resolve(board, 85, 2)                     # 87 is a snake to 24
    _check("landing on a snake slides down",
           r.landing == 87 and r.final == 24 and r.hops[0].is_snake)

    # --- 4. ExactLanding: overshoot does not move you ---
    r = rule.resolve(board, 97, 5)                     # 102 > 100
    _check("an overshoot does not move the player",
           r.outcome is MoveOutcome.OVERSHOT and r.final == 97)
    r = rule.resolve(board, 95, 5)                     # exactly 100
    _check("landing exactly on 100 wins", r.outcome is MoveOutcome.WON and r.final == 100)

    # --- 5. the other two win rules over the identical board ---
    over = OvershootWinsRule()
    r = over.resolve(board, 97, 5)
    _check("OvershootWins wins on the same roll that ExactLanding refuses",
           r.outcome is MoveOutcome.WON and r.final == 100)

    bounce = BounceBackRule()
    plain = Board(10)                                  # no jumps, so the
    r = bounce.resolve(plain, 8, 5)                    # arithmetic is visible
    _check("BounceBack reflects off the end",
           r.outcome is MoveOutcome.BOUNCED and r.final == 7)
    r = bounce.resolve(plain, 8, 4)                    # 12 -> reflect to 8
    _check("a reflection can return you to where you started",
           r.outcome is MoveOutcome.BOUNCED and r.final == 8)
    r = bounce.resolve(plain, 8, 2)                    # exactly 10
    _check("an exact landing with BounceBack still wins",
           r.outcome is MoveOutcome.WON and r.final == 10)
    r = bounce.resolve(plain, 9, 3)                    # 12 -> reflect to 8
    _check("bounce-back arithmetic is symmetric",
           r.final == 8 and r.outcome is MoveOutcome.BOUNCED)
    # The boundary a single reflection gets wrong: a roll that overshoots by
    # more than the board length. One reflection gives 10 - 23 = -13.
    r = bounce.resolve(plain, 8, 25)                   # 33 -> back to 5
    _check("a roll longer than the board still lands on the board",
           r.final == 5 and r.outcome is MoveOutcome.BOUNCED)
    _check("_fold agrees with walking the track by hand",
           [_fold(p, 10) for p in (9, 10, 11, 19, 20, 21, 29)] ==
           [9, 10, 9, 1, 2, 3, 9])

    # --- 6. all three rules agree when there is no overshoot ---
    for name, rl in (("exact", ExactLandingRule()), ("overshoot", OvershootWinsRule()),
                     ("bounce", BounceBackRule())):
        r = rl.resolve(board, 10, 3)
        _check(f"{name}: a plain move of 3 from 10 lands on 13", r.final == 13)

    # --- 7. the resolver defends against a chain even though validation does not ---
    chain_board = Board(100, [Jump(10, 20), Jump(20, 5)], )
    # Building it is fine; only validate() rejects the chain. Resolve anyway.
    r = ExactLandingRule().resolve(chain_board, 8, 2)   # 10 -> 20 -> 5
    _check("an unvalidated chain still resolves to its end",
           r.final == 5 and len(r.hops) == 2)

    # A cycle cannot be resolved; max_hops turns the hang into an error.
    cyclic = Board(100)
    cyclic.jumps = {10: Jump(10, 20), 20: Jump(20, 10)}   # bypasses __init__
    msg = _raises(lambda: ExactLandingRule().resolve(cyclic, 8, 2))
    _check("a resolver hitting a cycle raises instead of hanging",
           msg is not None and "exceeded" in msg)

    # --- 8. the dice port ---
    ordered = ScriptedDice([3, 1, 4])
    _check("ScriptedDice returns the script in order",
           [ordered.roll() for _ in range(3)] == [3, 1, 4])
    sd = ScriptedDice([2], default=5)
    sd.roll()
    _check("ScriptedDice falls back to its default", sd.roll() == 5)
    _check("RandomDice stays in range",
           all(1 <= RandomDice(6, random.Random(i)).roll() <= 6 for i in range(50)))
    _check("a die with one side is rejected",
           _raises_value(lambda: ScriptedDice([1], sides=1)) is not None)

    # --- 9. the six rules ---
    players = [Player("A"), Player("B")]
    tm = TurnManager(players)
    _check("a non-six passes the turn", tm.note_roll(3) is Decision.NEXT)
    _check("a six grants another roll", tm.note_roll(6) is Decision.EXTRA)
    _check("two sixes still grant another roll", tm.note_roll(6) is Decision.EXTRA)
    _check("the third six forfeits", tm.note_roll(6) is Decision.FORFEIT)
    tm.advance()
    _check("advancing clears the streak", tm.streak == 0)
    _check("a six after a non-six restarts the count",
           tm.note_roll(6) is Decision.EXTRA and tm.streak == 1)

    # --- 10. a deterministic game, driven by a scripted dice ---
    def fresh_game(script, rule=None, cap=10_000):
        return Game(Board(30, [Jump(3, 12), Jump(17, 5), Jump(22, 28)]),
                    ScriptedDice(script, default=2),
                    [Player("A"), Player("B")],
                    rule=rule, turn_cap=cap)

    game = fresh_game([3, 1, 6, 2])
    r0 = game.step()
    _check("A rolls 3 and lands on the ladder at 3", r0.to_position == 12)
    r1 = game.step()
    _check("B rolls 1 and moves to 1", r1.to_position == 1)
    r2 = game.step()
    _check("A's six grants another roll",
           r2.roll == 6 and r2.decision is Decision.EXTRA)
    _check("and the turn does not advance", game.turns.current().name == "A")
    r3 = game.step()
    _check("A's second roll moves A again", r3.to_position == 20)
    _check("then the turn passes to B", game.turns.current().name == "B")

    # --- 11. a full game with a scripted dice is reproducible ---
    def run(script) -> tuple:
        g = fresh_game(script)
        result = g.play()
        return (result.winner.name if result.winner else None,
                len(result.log), result.turns)

    a = run([4, 3, 5, 2, 6, 1, 3, 4, 2, 5, 1, 3])
    b = run([4, 3, 5, 2, 6, 1, 3, 4, 2, 5, 1, 3])
    c = run([5, 5, 5])
    _check("the same script gives the same game", a == b)
    _check("the scripted game produces a winner", a[0] is not None)
    _check("and a non-empty log", a[1] > 0)
    _check("a different script gives a different game", c != a)

    # --- 12. the whole log is replayable from its rolls ---
    g = fresh_game([4, 3, 5, 2, 6, 1, 3, 4, 2, 5, 1, 3])
    result = g.play()
    _check("exactly one row is marked won", sum(1 for r in result.log if r.won) == 1)
    _check("the winner's row is a WON outcome",
           [r.outcome for r in result.log if r.won] == [MoveOutcome.WON])
    _check("the winner is standing on the last square",
           result.winner is not None and result.winner.position == g.board.size)
    _check("the winner is marked finished", result.winner.finished is True)
    _check("the reported winner is not the player whose turn comes next",
           result.winner is not g.turns.current())
    _check("finished is set only on the winner",
           sum(1 for p in g.players if p.finished) == 1)

    # Feed the logged rolls back through a fresh game. This is the payoff for
    # logging the roll and not just the destination: the game reproduces from
    # its own log, which is what makes a bug report actionable.
    logged_rolls = [row.roll for row in result.log]
    replay = fresh_game(logged_rolls).play()
    _check("re-feeding the logged rolls reproduces the game exactly",
           [(r.turn, r.player, r.from_position, r.to_position, r.outcome)
            for r in replay.log] ==
           [(r.turn, r.player, r.from_position, r.to_position, r.outcome)
            for r in result.log])

    # --- 13. the three-sixes forfeit undoes the turn ---
    g = Game(Board(30), ScriptedDice([6, 6, 6, 3, 3, 3], default=3),
             [Player("A"), Player("B")])
    r1 = g.step()          # A rolls 6 -> moves 6, extra roll
    r2 = g.step()          # A rolls 6 -> moves to 12, extra roll
    r3 = g.step()          # A rolls 6 -> FORFEIT, back to 0
    _check("the first two sixes moved A", r1.to_position == 6 and r2.to_position == 12)
    _check("the third six forfeits the turn", r3.forfeited is True)
    _check("and the turn's moves are undone",
           r3.to_position == 0 and g.players[0].position == 0)
    _check("the turn passes after the forfeit",
           g.turns.current().name == "B")

    # --- 14. with a loaded dice the game does not terminate, and the cap says so ---
    stuck = Game(Board(30), LoadedDice(6), [Player("A"), Player("B")],
                 turn_cap=40)
    result = stuck.play()
    _check("a permanently-loaded dice game hits the cap instead of hanging",
           result.truncated and len(result.log) == 40)
    _check("the forfeits fired along the way",
           sum(1 for r in result.log if r.forfeited) > 0)

    # The forfeit is what bounds an honest dice too: without it, a run of sixes
    # would be unbounded. With it, at most max_consecutive_sixes rolls per turn.
    longest = max(len(list(group)) for _, group in _group_by_turn(result.log))
    _check("no turn exceeds the three-six bound", longest <= 3)

    # --- 15. finished players are skipped ---
    order = [Player("A"), Player("B", finished=True), Player("C")]
    tm2 = TurnManager(order)
    _check("a finished player is skipped", tm2.current().name == "A")
    tm2.advance()
    _check("and the next live player is chosen", tm2.current().name == "C")
    _check("a game with every player finished raises rather than looping",
           _raises_runtime(lambda: TurnManager([Player("Z", finished=True)]).current())
           is not None)

    # --- 16. the board validator accepts an empty jump list ---
    _check("a board with no jumps is valid",
           Board(100).validate().size == 100)
    _check("a one-square board is rejected",
           _raises(lambda: Board(1)) is not None)

    # --- 17. an interactive caller can drive one roll at a time ---
    g = fresh_game([2, 2, 2, 2, 2, 2])
    first = g.step()
    _check("step() returns one row and appends exactly one row",
           len(g.log) == 1 and g.log[0] is first)
    _check("the row names the turn and the player",
           first.turn == 1 and first.player == "A")
    _check("render lists both players", "A@" in g.render() and "B@" in g.render())

    print("\nall checks passed")
```

**What to point at:**

- `Board.validate()` — the five defects, each with the offending jump in the message. A cycle is a graph walk, not a per-jump check, because only the walk finds it.
- `WinRule.resolve` is **pure**: it returns a `MoveResolution` and moves nobody, so the three variants are tested against the same board in four lines each.
- `Decision.FORFEIT` is a third case, not a boolean, because the three-sixes rule **undoes** the turn's moves — a boolean cannot express that.
- `TurnManager` owns both six rules together, with the comment saying why: the extra roll is unbounded without the forfeit.
- `Game.play` has a `turn_cap` and returns `GameResult(truncated=True)` rather than hanging against a loaded dice. The demo shows it firing.
- `_apply_jumps` has a `max_hops` guard that turns a cycle from a hang into a `BoardConfigError`, even though `validate()` already forbids one.

---

## 6. Concurrency & thread-safety

### What breaks

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Two threads rolling one game** | Two dice rolls land on the same player, the turn index moves twice, and the log has two rows numbered the same turn | One lock per game around `step()` — the whole aggregate |
| **A shared dice across games** | A shared `random.Random` is not a documented thread-safe API; the sequence becomes interleaved and irreproducible | One `Dice` per game, or a lock inside a shared implementation |
| **A shared `Player` list across games** | Two games mutate the same `Player` objects: positions jump between boards | Each game constructs its own `Player` instances |
| **Reading the log while a roll is in flight** | A UI iterating `game.log` sees a partially appended row, or a row for a turn whose position has not been applied | Read under the same per-game lock, or return copies |
| **An observer mutating the game** | A callback calls `step()` mid-`step()` and the log loses a turn | Emit the row after the state is applied, and keep callbacks read-only |
| **The turn cap checked outside the lock** | Two threads both pass the cap check and the game overshoots it | The cap is a per-game counter, checked and incremented inside the lock |
| **A `ScriptedDice` shared by parallel tests** | Its `index` is mutable, so the second test consumes the first test's rolls | A fresh dice per game; the scripted dice is deliberately not thread-safe, and the docstring says so |

**The one to volunteer:** the shared `random.Random`. It is the concurrency question that is specific to this problem: a `ScriptedDice` is **stateful** (`index`), so sharing one between two games silently corrupts both, and a `RandomDice` shared across games is at best non-deterministic and at worst relying on an implementation detail of the GIL.

### The per-game lock

```python
import threading


class LockedGame:
    """One lock per game around the whole aggregate.

    The lock is not per-player or per-dice: `step()` reads the board, reads and
    advances the turn index, rolls the dice, mutates a player and appends a log
    row. Any two of those interleaving produces a game whose log does not
    describe the position. So the critical section is the whole step.
    """

    def __init__(self, game: Game) -> None:
        self._game = game
        self._lock = threading.Lock()

    def step(self) -> RollOutcome:
        with self._lock:
            return self._game.step()

    def snapshot(self) -> tuple:
        """A read that is safe because it is taken under the same lock the
        writes use — not because tuple() is atomic."""
        with self._lock:
            return (tuple((p.name, p.position) for p in self._game.players),
                    tuple(self._game.log))
```

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Load the board from a file"

The board is already data, so this is a parser plus the existing validator:

```python
import json


def load_board(path: str) -> Board:
    """The validator is the point: a hand-edited layout file is exactly the
    source of the five defects validate() rejects, so the loader cannot skip
    it."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    board = Board(data["size"], [Jump(j["start"], j["end"]) for j in data["jumps"]])
    return board.validate(allow_chains=data.get("allow_chains", False))
```

A generated layout is the same shape with a generator instead of a parser — and the generator must produce a board that passes `validate()`, which is the cheapest possible test for it.

### 7.2 "A fair dice, and a custom one"

```python
class WeightedDice(Dice):
    """A non-uniform die, for a house-rules variant. `random.choices` with an
    explicit weight vector keeps the distribution in one place instead of
    being smeared through the game."""

    def __init__(self, weights: dict[int, float], sides: int = 6) -> None:
        super().__init__(sides)
        missing = set(range(1, sides + 1)) - set(weights)
        if missing:
            raise ValueError(f"weights are missing faces: {sorted(missing)}")
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        self.faces = list(weights)
        self.weights = list(weights.values())

    def roll(self) -> int:
        return random.choices(self.faces, weights=self.weights, k=1)[0]
```

The interesting consequence to name: **a biased dice changes the liveness question**. With a fair die the three-sixes rule bounds a turn at three rolls; with a loaded one, the same rule still bounds it — which is precisely why the bound belongs to `TurnManager` and not to the dice.

### 7.3 "Several dice, or a die with more sides"

```python
class SumDice(Dice):
    """Two dice, summed — a Monopoly-style variant. Composition rather than a
    new subclass of each die: the game does not care how a number arrived."""

    def __init__(self, dice: list[Dice]) -> None:
        if not dice:
            raise ValueError("SumDice needs at least one die")
        super().__init__(sum(d.sides for d in dice))
        self.dice = dice

    def roll(self) -> int:
        return sum(d.roll() for d in self.dice)
```

Note the trap: `SumDice.sides` is the sum of the dice, which is *not* the range of the result. A caller that assumes `1..sides` will validate wrongly — so `sum` of two d6 is 2..12, and the class must not pretend otherwise. That is worth saying out loud, because it is the kind of thing that passes review and then produces a wrong turn cap.

### 7.4 "Replay a game from its log"

The log is already the full record, so a replay is a parser and a re-run:

```python
def replay(board: Board, log: Iterable[RollOutcome],
           rule: Optional[WinRule] = None) -> GameResult:
    """Re-run a logged game by feeding its rolls back through a ScriptedDice.
    This is the payoff for logging outcomes rather than just positions: a bug
    report is a board, a rule and a list of rolls, and it reproduces exactly."""
    rows = list(log)
    scripted = ScriptedDice([row.roll for row in rows],
                            default=rows[-1].roll if rows else 1)
    names = []
    for row in rows:
        if row.player not in names:
            names.append(row.player)
    game = Game(board, scripted, [Player(name) for name in names], rule=rule)
    return game.play()
```

The condition on its honesty: the replay is exact only if the rule and the board are the same, and if the log contains **every** roll — which means a log that records only positions cannot be replayed at all.

### 7.5 "Teams, or a per-player board"

The `TurnManager` already owns order, so teams are a change there:

```python
class TeamTurnManager(TurnManager):
    """A team wins when any member finishes. The order is by team, and the
    member within a team rotates."""

    def __init__(self, teams: dict[str, list[Player]], **kwargs) -> None:
        super().__init__([p for members in teams.values() for p in members], **kwargs)
        self.teams = teams

    def team_of(self, player: Player) -> str:
        for team, members in self.teams.items():
            if player in members:
                return team
        raise KeyError(player.name)

    def advance(self) -> None:
        super().advance()
        # Skip to the next TEAM rather than the next player.
        start = self.team_of(self.order[self.index])
        for _ in range(len(self.order)):
            if self.team_of(self.order[self.index]) != start:
                return
            self.index = (self.index + 1) % len(self.order)
```

### 7.6 "Make the game observable, for a UI"

`Game` already produces one `RollOutcome` per roll, so the observer seam is a list of callbacks:

```python
class GameObserver(ABC):
    def on_roll(self, game: Game, row: RollOutcome) -> None: ...
    def on_finish(self, game: Game, result: GameResult) -> None: ...


class ObservableGame(Game):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._observers: list[GameObserver] = []

    def add_observer(self, observer: GameObserver) -> None:
        self._observers.append(observer)

    def step(self) -> RollOutcome:
        row = super().step()
        for observer in self._observers:
            observer.on_roll(self, row)          # after the state is applied
        return row

    def play(self) -> GameResult:
        result = super().play()
        for observer in self._observers:
            observer.on_finish(self, result)
        return result
```

The ordering rule is the same as in [tic-tac-toe](14-tic-tac-toe.md): notify **after** the state is consistent, never mid-mutation.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **A cycle in the jumps** | The game hangs on the first landing there | `validate()` walks the graph and rejects it; the resolver also has a hop limit |
| 2 | **A jump onto the last square** | The game is won on the first roll — a layout typo that looks like a feature | Rejected at validation: an instant win |
| 3 | **Two jumps starting at one square** | One silently overwrites the other in a dict | Rejected at construction, naming both |
| 4 | **A ladder landing on a snake** | The move chains, which is either intended or a bug depending on the game | Rejected by default, allowed with `allow_chains=True` |
| 5 | **A jump off the board** | An `IndexError` later, or a position beyond the end | Rejected at validation with the offending number |
| 6 | **A jump from the final square** | Unreachable, so a dead entry nobody notices | Rejected: the layout was generated against another size |
| 7 | **Overshoot with ExactLanding** | The player moves past the end and can never come back | The roll is a no-op; `MoveOutcome.OVERSHOT` says so |
| 8 | **Bounce-back reflecting exactly onto the end** | Treated as a bounce, so the player sits on 100 without winning | A reflection landing on the end is a win — the code checks it |
| 9 | **Extra roll on a six with no bound** | An unbounded number of rolls per turn; against a loaded dice the game never ends | The three-sixes forfeit bounds it, and the game still has a `turn_cap` |
| 10 | **The three-sixes forfeit implemented as a boolean** | The rule cannot undo the turn's moves, so it is silently a different rule | `Decision.FORFEIT` as a third case, with revert-to-turn-start |
| 11 | **A hard-coded `random.randint`** | The game cannot be tested; every run is different | The `Dice` port, with a scripted implementation |
| 12 | **A shared `ScriptedDice` between games** | Its `index` carries over and the second game consumes the wrong rolls | A dice per game |
| 13 | **A shared `Player` object between games** | Positions leak between boards | Construct players per game |
| 14 | **A finished player still taking turns** | The game never ends, because a winner keeps rolling | `TurnManager.current()` skips finished players |
| 14b | **Reading the winner from `current()` after the win** | The winner is marked finished, so `current()` skips them and the **loser is reported as the winner** | `Game` records the winner at the moment of the win |
| 15 | **A `SumDice.sides` used as the result range** | Two d6 summed is 2..12, not 1..12 — validation and the turn cap are wrong | `sides` is documented as the *maximum*, and range checks use the actual bounds |
| 16 | **A one-square board** | Every roll overshoots; the game is unwinnable with ExactLanding | Rejected: a board needs at least two squares |
| 17 | **The turn manager advanced twice on a six** | The player who rolled a six loses the extra roll | `step()` advances only on `Decision.NEXT` |
| 18 | **The log row appended before the mutation** | A crash mid-move leaves a row describing a move that did not happen | Mutate, then append |
| 19 | **Replaying a log that records only positions** | The rolls are lost, so the game cannot be reproduced | Log the roll, not just the destination |
| 20 | **A negative or zero roll** | A scripted or weighted dice can produce it; the position goes backwards | `ScriptedDice` validates its script, `WeightedDice` validates its weights |

**The three to volunteer:** #1 (the cycle), #2 (the jump onto the last square) and #9 (the unbounded extra roll). The first two are the configuration defects that make a board unplayable, and the third is the liveness property that interviewers rarely get to because candidates stop designing once the game "works".

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **Strategy** | `WinRule` | The three variants interviewers ask for — exact landing, overshoot wins, bounce back — are one class each over an identical board | One indirection per move, and the pure signature must be honoured |
| **Ports & Adapters** | `Dice` | The game cannot be tested without it; `ScriptedDice` is the adapter that makes a whole game deterministic | The game never calls `random` directly, which some readers find indirect until they see the test |
| **Value Object** | `Jump`, `MoveResolution`, `RollOutcome`, `GameResult` | Immutable, loggable, replayable; the log is the debugging tool | One allocation per roll |
| **Policy** | `TurnManager` | The turn rules are the thing that varies between house variants, and the six rules must live together | An extra object in the composition |
| **Façade** | `Game` | One `step()` and one `play()` hide four collaborators from a caller | The façade can accumulate logic that belongs in the collaborators — the discipline is that it only *applies* resolutions |
| **Registry / Config** | `Board` | The layout is data; `validate()` is the guard that makes data safe to accept | Validation must be remembered at every construction path |
| **Observer** | `ObservableGame` in §7.6 | A UI needs per-roll events, and the log already is one | Notification ordering |
| **Singleton** | *Deliberately rejected* | A global board or dice makes two tests interfere | — |

**The sentence to give:** *"Strategy for the win rule, Ports & Adapters for the dice, and a Policy object for the turn rules. The dice port is the one I would defend hardest, because without it none of the other claims in this design can be demonstrated — a game class that calls `random.randint` internally is a class you can only test by running it many times and checking statistical properties, which is not a test."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Board as validated data** | Yes | A layout typo is a startup error, not a mysteriously broken game | A validator to write, and every construction path must call it |
| **Five specific defects rejected** | Yes | Each names the offending jump | Some layouts a customer wants are rejected — chains — and need a flag |
| **Dice as a port** | Yes | A whole game is deterministic under test; the demo is possible | An abstraction over `randint` that looks like overkill |
| **Pure `resolve`** | Yes | The three variants and chaining are unit tests | The game must then apply the resolution itself, so mutation lives in one place |
| **Three win rules** | Yes | The follow-up is answered, not described | More surface than the default needs |
| **`Decision` as a three-case enum** | Yes | The forfeit can undo a turn | A boolean would have been simpler and wrong |
| **Three-sixes forfeit** | Yes | The extra-roll rule is bounded; the game terminates | A house rule some players do not expect |
| **`turn_cap` on top of the forfeit** | Yes | A loaded or buggy dice cannot hang a test | A truncated game is a silent non-result unless reported — hence `GameResult.truncated` |
| **`RollOutcome` log** | Yes | Replay, debugging, UI events | Memory grows with the game; a long game needs a ring buffer |
| **`max_hops` in the resolver** | Yes | A cycle is an error rather than a hang, even on an unvalidated board | A defensive limit that a correct board never reaches |
| **Per-game lock** | Yes | Many concurrent games; the aggregate stays consistent | A wrapper, and a discipline that no read bypasses it |
| **No observer in the core** | Yes | `Game` has no callback surface until a UI needs it | The extension in §7.6 duplicates a small amount of `step()` |

**The trade to name unprompted:** *"Validating the board rejects layouts that are merely unusual, and chained jumps are the concrete example — they are legal in some house variants and I have made them opt-in. That is the right default because a chain is far more often a typo than an intention, and a rejection with the offending jump in the message costs the developer thirty seconds while a silent chain costs them an afternoon."*

---

## 11. Q&A

**Q1. This is a children's game. What is there to design?**
The game is trivial and the *configuration* is not. The design work is: a board whose layout is validated data rather than code, a dice port that makes the game deterministic under test, a pure resolution function so the three win-rule variants and the chaining question are testable, and a turn policy that owns the extra-roll rule together with the bound that makes it terminate. That is four design decisions, three of which have nothing to do with the game being simple — and interviewers ask this problem precisely because a candidate who says "it's easy, I'll just write it" produces a 40-line script with eight bugs in the layout handling.

**Q2. What exactly does the validator check, and why those five?**
Duplicate jump starts, because a map silently drops one and a snake disappears; cycles, because the game hangs; jumps off the board, because a position outside the range breaks every later comparison; a jump onto the last square, because it wins on the first roll and looks like a feature; and chained jumps, because a ladder landing on a snake is ambiguous between two correct games. The first four are always bugs. The fifth is a policy, which is why it is the one with a flag. And each message names the offending jump, because a configuration error that does not tell you which entry is wrong is a support ticket rather than a fix.

**Q3. Why is the dice a class rather than a `random.randint` call?**
Because the alternative makes every claim about this design untestable. With a dice port, one scripted sequence makes an entire game deterministic: the demo can assert that A lands on the ladder at square 3 and climbs to 12, that a six grants another roll, and that a full game is reproducible across two runs — none of which is assertable against `randint`. The cost is an interface with a one-line implementation, and I would pay it every time: a game object that calls `random` internally can only be tested statistically, and a statistical test is not a test.

**Q4. Explain move resolution being pure, and what it buys.**
`resolve(board, position, roll)` returns a `MoveResolution` — the landing square, the hops taken, the final square, the outcome — and moves nobody. Everything interesting about the rules lives in that function, so the three variants are each tested against the same board in four lines, and the chaining question is tested without running a game at all. Mutation is then confined to one place in `Game`, which means the log row and the position can never disagree. The cost is that the game has to apply the resolution itself, which is a few more lines in `Game.step` — and those lines are the only place a position changes, which is a benefit I would trade for again.

**Q5. What are the three win rules, and why are they separate classes?**
Exact landing — you must land on the last square, and an overshoot is a no-op. Overshoot wins — any roll that reaches or passes the end wins. And bounce-back, where the roll reflects off the end: from 98 with a 5 you land on 97. They are separate classes because they are three different answers to the same question, and the question — "what does it mean to reach the end?" — is not the game's business. The demo runs all three against an identical board, which is also how I would answer "which one is right?": it depends on the house rules, so the design makes it a parameter rather than a rewrite.

**Q6. Why does a six grant another roll, and what breaks about it?**
It is a house rule from Ludo, and what breaks is termination. "Roll again on a six" makes the number of rolls in a turn unbounded — with a fair die the probability of a very long run is small but non-zero, and with a loaded or buggy die the game literally never ends. The fix is the three-sixes forfeit, which is the companion rule in Ludo for exactly that reason: three consecutive sixes and the turn's moves are undone. The pair has to live in the same object, because either rule alone is unsound. And `Decision.FORFEIT` has to be a third case rather than a boolean, because a forfeit *undoes* — a boolean cannot express "end the turn and roll back the position".

**Q7. Even with the forfeit, could the game fail to terminate?**
Yes: a player who never reaches the end, or a loaded die combined with a rule set that keeps granting turns. So `play()` has a `turn_cap`, and hitting it returns `GameResult(winner=None, truncated=True)` rather than a hang or a lie. The demo proves it: a permanently-loaded d6 hits the cap after exactly 40 rolls, and the forfeit fires along the way. The design point is that a *silent* truncation would be worse than the hang, because a test that hangs is obvious and a test that silently returns a non-result gets asserted against and becomes a false confidence. So truncation is a property of the result.

**Q8. What is the `RollOutcome` log for?**
It is the observability spine. One frozen row per roll — the turn, the player, the roll, the from and to positions, the hops, the outcome, the decision, and the two flags — gives three things at once: a UI has an event stream without the game knowing about the UI; a bug report is a board plus a rule plus a list of rolls and it reproduces exactly; and the tests can assert on the *sequence* rather than only the final position, which is where the interesting bugs are. The cost is memory proportional to the game length, so a very long game needs a ring buffer — worth saying, because it is the obvious follow-up.

**Q9. How do you test a game with randomness in it?**
Three layers, and only the last is statistical. First, the **pure resolver** takes a roll as an argument, so all the rule logic is tested with no randomness at all. Second, the **scripted dice** makes whole games deterministic, so an end-to-end test asserts exact positions and an exact winner. Third — and only where it matters — a **distribution test** over a real dice: a few thousand rolls and a loose bound on the mean. That ordering is deliberate: statistical tests are slow, flaky and weak, so they are reserved for the one thing that has no deterministic alternative.

**Q10 (senior). The board has 100 squares and a game averages around 30 turns each. Does that number matter to the design?**
It tells you two things. The first is that a full game is short enough that the log is small — a few hundred rows — so keeping the entire log in memory is fine, and a ring buffer is a hypothetical rather than a need. The second is that the simulation cost is low enough that the honest way to answer "is this board balanced?" is to **run a million games** rather than reason about it: the layout's fairness is a statistical property, and the design supports that directly because `RandomDice` is a port and a game is a pure function of its rolls. That is the practical payoff of the injection: a Monte Carlo balance check is a hundred games in a loop, not an instrumented copy of the game.

**Q11 (senior). Where would you expect bugs in this code?**
Four places. The **bounce-back arithmetic**: the reflection is `size - (target - size)`, and the case where the reflection lands exactly on the end is a win rather than a bounce — the demo tests both, because that is the boundary that gets it wrong. The **forfeit's revert**: the position must be restored to the *start of the turn*, not to the position before the third six, and getting that wrong changes the rule silently. The **turn advance on a six**, where advancing and granting an extra roll can both happen and the player silently loses the extra roll. And the **chain guard's interaction with validation**, where a board built without calling `validate()` is resolvable and hangs — the `max_hops` limit exists for exactly that path. All four produce a plausible game rather than a crash, which is why the tests assert on the log's structure and not just the winner.

**Q12. Why not put the win condition on the board or in the player?**
Because neither owns the question. On the board it would be a flag that every rule variant has to edit, and the board would accumulate a mode for each house variant. On the player it would be worse: the rule is about the *board's geometry*, and a player that knows how to win on this board cannot be moved to another one. Keeping it in a pure `WinRule` means the same board supports all three variants and the same rule supports any board, and it makes the rule the thing you unit-test in isolation. The same reasoning appears in [tic-tac-toe](14-tic-tac-toe.md), where the win check is a strategy for the same reason.

---

### Pushback round

**P1. "You have written a validator, three rule classes and a turn-policy object for a game that is forty lines of straightforward code."**
For the game itself that is true, and if the requirement stopped at "simulate a game" I would write the forty lines. The design is justified by three things the forty lines cannot do: they cannot be tested deterministically, because the dice is inline; they cannot answer "what about bounce-back?", because the rule is an `if`; and they accept a board layout that can hang the game or win it on the first roll, because nothing validates it. The validator is the one I would defend hardest — it is not architecture, it is the difference between a layout typo being a startup error and a layout typo being an afternoon of confusion.

**P2. "The pure resolver adds indirection. Why not just move the player inside the rule?"**
Because then the rule's tests become game tests. With the pure signature, "does bounce-back reflect off the end correctly?" is four lines with no game, no players and no dice — and the boundary case where the reflection lands exactly on the end is one of those four lines rather than a scenario you have to construct. If the rule mutated a player, every rule test would need a player, a board and a place in a turn, and the boundary cases would be the ones nobody writes. The cost is that `Game.step` has to apply the resolution, and I would rather have the mutation in exactly one method.

**P3. "`allow_chains=True` means a board can pass validation and then chain, so your validation is not a guarantee."**
Correct, and I would not claim otherwise. The flag makes chains *intentional* rather than accidental, which is the actual goal — the defect is not chaining, it is a chain nobody meant. What follows is that the resolver cannot assume a single hop, so it walks the graph with a `max_hops` limit, and the demo tests both the chain resolving correctly on a flagged board and the limit raising on a hand-built cycle. The honest summary is that validation catches the four defects that are always bugs and *flags* the fifth, and the resolver defends against all five because it can be handed a board that never went through validation.

**P4. "You said the game cannot hang, but the cap can be reached in a real game with fair dice — a player could be stuck behind an unlucky layout for ten thousand rolls."**
Ten thousand rolls is an impossible number of turns for 100 squares, so in practice the cap is a bug detector rather than a gameplay limit — a fair game finishes in tens of turns. But the criticism points at something real: `truncated` is a property of the result, and a caller must decide what to do with it. My answer is that a truncated game is a *failed* game: the correct response is to log the board and the rule set and investigate, not to declare a draw, because reaching the cap with a fair dice means either a layout where progress is impossible or a bug in the resolution. That is worth stating in the code, which is why `GameResult.truncated` exists rather than the cap silently returning `None`.

---

## 12. Cheat-sheet summary + cross-links

| Concept | One-line truth |
|---|---|
| The shape | `Board` (validated data) + `Dice` (port) + `WinRule` (pure strategy) + `TurnManager` (policy) + `Game` (aggregate) |
| The real difficulty | Not the game — the **configuration**, the **testability** and the **termination** |
| The five board defects | Duplicate start · cycle · off-board · jump onto the last square · chained jump |
| The one that hangs | A **cycle**; found by a graph walk, not a per-jump check |
| The one that wins instantly | A jump **onto the last square** — a layout typo that looks like a feature |
| The one with a flag | **Chains** (a ladder landing on a snake) — legal in some house variants |
| Every error message | Names the **offending jump**; a config error without it is a support ticket |
| The dice | A **port**, not `random.randint`. `ScriptedDice` makes a whole game deterministic |
| `LoadedDice` | The adversary that proves the turn cap and the forfeit are load-bearing |
| The resolver | **Pure** `resolve(board, position, roll) -> MoveResolution`; mutation lives only in `Game` |
| The three win rules | Exact landing (default) · overshoot wins · bounce-back |
| Bounce-back boundary | A reflection landing **exactly on the end is a win**, not a bounce |
| The six rules | Extra roll on a six **plus** the three-sixes forfeit — the pair, or the rule set is unsound |
| Why the forfeit is not boolean | It **undoes** the turn's moves → revert to turn-start position → needs a third enum case |
| Termination | The forfeit bounds turns; a `turn_cap` bounds the game; truncation is **reported**, not silent |
| Chaining defence | `max_hops` turns a cycle into an error even on a board that skipped validation |
| The log | One `RollOutcome` per roll → replayable, debuggable, and a UI event stream |
| Replay honesty | Requires **every roll** logged; a log of positions alone cannot be replayed |
| Finished players | Skipped by `TurnManager.current()`; all-finished raises rather than looping |
| `SumDice` trap | `sides` is the maximum, not the range — two d6 is 2..12 |
| Concurrency | One lock per game around the whole `step()`; a fresh `Dice` and fresh `Player`s per game |
| Patterns | Strategy, Ports & Adapters, Value Object, Policy, Façade; Singleton rejected |

**The three sentences to memorise**

1. *"The board is validated data — five defects rejected, each with the offending jump named — and a cycle is found by a graph walk because that is the one that hangs the game."*
2. *"The dice is a port, so the whole game is deterministic under test; a hard-coded `randint` makes every claim about this design unverifiable."*
3. *"An extra roll on a six is unbounded, so it must ship with the three-sixes forfeit — and the forfeit undoes a turn, which is why the decision is a three-case enum and not a boolean."*

### Cross-links

- **The LLD answer framework this file follows, and the HLD/LLD boundary:** [LLD/01 What is LLD and How to Answer](../01-what-is-lld-and-how-to-answer.md)
- **The Strategy and Value Object patterns in the catalogue:** [LLD/03 Design Patterns](../03-design-patterns.md)
- **The injected-clock/test-double argument, made the other way round:** [Rate Limiter](11-rate-limiter.md) §5 (`Clock`, `FakeClock`) · §11 Q9
- **Another pure-rule strategy so a rule set can vary without touching state:** [Tic-Tac-Toe](14-tic-tac-toe.md) §4 (`WinRule`, `MisereWinRule`) and §11 Q3
- **The same observer ordering rule — notify after the state is consistent:** [Tic-Tac-Toe](14-tic-tac-toe.md) §6 · [Vending Machine](10-vending-machine.md) §6
- **A configurable board and a validation boundary in a different problem:** [Chess Game](09-chess-game.md) §1 (board setup and promotion rules) · [Vending Machine](10-vending-machine.md) §7.1
- **Termination and liveness as a design property rather than a nicety:** [Pub/Sub System](13-pub-sub-system.md) §7.5 (poison pill, DLQ, max attempts) · [Logger Framework](12-logger-framework.md) §7.4 (rotation bounds)
- **Injecting randomness and time so a system is testable:** [Rate Limiter](11-rate-limiter.md) §6 and §11 Q9
- **Cheat sheet, pattern tables and the classic-problems list:** [LLD Cheatsheet](../../Cheat-Sheets/07-lld-cheatsheet.md)
- **Question bank:** [LLD Questions](../../Interview-Questions/08-lld-questions.md)
