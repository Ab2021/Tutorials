# LLD 14 — Tic-Tac-Toe

> **Grounding note.** The reference repositories contain **no OOP/LLD material**; the source transcripts do not cover tic-tac-toe as a design problem at all. This file is written from first principles and is not transcript-grounded. The general shape of an LLD answer — clarify, entities, interfaces, code, concurrency, extensibility, failure modes — is the house template at [LLD/01 What is LLD](../01-what-is-lld-and-how-to-answer.md).

> **TL;DR** — Tic-tac-toe looks trivial and is not: it is the smallest problem where the **win condition must be generalised** (the naive `if` of eight hard-coded lines cannot become a 4×4 board or a five-in-a-row game), where the **win check must be incremental** (only the lines through the last move can have changed), and where the **bot needs a real algorithm** (minimax with alpha-beta, and a transposition table whose correctness depends on caching only *exact* values). The design is: an immutable-looking `Move`, a `Board` that supports make/unmake so the search does not allocate, a `WinRule` strategy that generalises to *n*×*n* with *k* in a row, a `Player` interface with human, random, heuristic and minimax implementations, and a `Game` that owns turn order, status and undo history. The interview is won on the generalisation and on `is_full()`/draw handling — and on the sentence *"a bot's move is validated like any other move, because a buggy bot must not corrupt the game."*

> **60-second interview answer:** "A `Board` of `Cell`s, a `Move` value object, a `WinRule` strategy that checks whether the last move completed a run of `k` in a row in any of four directions — that generalises to any board size instead of hard-coding eight lines — and a `Player` interface with a `choose_move` method so a human, a random bot and a minimax bot are interchangeable. The `Game` owns turn order, status and a history for undo. Two things I would call out: the win check is incremental, O(k) along the last move's four axes rather than O(n²) for a full scan, and the minimax bot searches by make/unmake on the real board rather than copying it — with a transposition table that caches **only exact values**, because caching an alpha-beta cutoff bound as if it were a value is the classic bug. And the draw case is not an afterthought: the game ends when the board is full *or* a line is completed, and the check order matters."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **3×3 only, or generalised?** | Whether the win rule is eight hard-coded lines or a parameterised *k*-in-a-row scan | Generalise to *n*×*n* with *k*; the 3×3 case is then a configuration |
| 2 | **Human versus bot, or both?** | Whether `Player` is an interface with several implementations or just an input loop | Both — that is what makes `Player` a real abstraction |
| 3 | **How strong must the bot be?** | Random, heuristic, or minimax with alpha-beta | Minimax with alpha-beta and a depth limit; the heuristic tier is a nice extra |
| 4 | **Undo/redo?** | Whether the game keeps a move history and can restore status and turn | Undo at least; redo is a stack away |
| 5 | **Which rules variants?** | Misère (last to move loses), forbidden moves, a move clock | Standard rules; name the variants as extensions |
| 6 | **Single game or many concurrent games?** | Whether there is a per-game lock and a game registry | One game object per match, locked per game — the server shape |
| 7 | **Who validates a move?** | Whether the bot's output is trusted | The game validates every move, including a bot's |
| 8 | **How is it rendered?** | Whether there is an observer/renderer seam or the game prints | An observer interface, so the game has no I/O |
| 9 | **Is the bot allowed to be slow?** | Whether the search is depth-limited and whether it can be off-thread | Depth-limited; a time budget rather than a fixed depth for larger boards |
| 10 | **What about the opening symmetry?** | Whether the search exploits the 8-fold symmetry of the empty board | The transposition table catches much of it; full symmetry canonicalisation is an extension |

### The defaults to assume, stated plainly

> "I'll assume a **generalised board** — *n*×*n* with *k* in a row, defaulting to 3×3 and *k*=3 — with a `Player` interface so human and bot players are interchangeable, **minimax with alpha-beta and a transposition table** for the strong bot, a heuristic bot as a difficulty tier, undo, and an observer seam so the game does no I/O. Every move, including a bot's, is validated by the game. Does that match, or do you want the plain 3×3 first?"

### Non-goals

- No network protocol or matchmaking; the game is a single-process object.
- No UI beyond a text renderer.
- No persistence — a serialisation hook is named in the extensions, not built.

### The three constraints that shape the design

1. **The win rule must be a strategy, not eight `if` statements.** Otherwise the design cannot answer "make it 4×4".
2. **The win check must be incremental.** Only the four lines through the last move can have changed; a full scan is O(n²) per move for no reason.
3. **The search must not allocate per node.** Copying the board at every minimax node is the difference between an instant move and a slow one — make/unmake is the fix.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `Cell` | Enum | `EMPTY`, `X`, `O`; `opponent` | Turn order — the game owns that |
| `Move` | Frozen value object | `row`, `col` | Legality |
| `GameStatus` | Enum | `ONGOING`, `X_WINS`, `O_WINS`, `DRAW`; `is_over` | — |
| `GameResult` | Frozen value object | Terminal status plus the winning line, if any | — |
| `Board` | Mutable data structure | Cells, bounds, `empty_cells`, `is_full`, `key`, make/unmake | Rules |
| `WinRule` (ABC) | Strategy | `wins(board, move, symbol)` incrementally; `winning_lines(...)` for display | Turn order |
| `NKLineWinRule` | Implementation | *k* consecutive in any of four directions | — |
| `Player` (ABC) | Interface | `choose_move(board, rule, status) -> Move \| None` | Applying the move |
| `HumanPlayer` | Implementation | Reads a move from an injected input function | I/O formatting |
| `RandomPlayer` | Implementation | Uniform random empty cell | Strength |
| `HeuristicPlayer` | Implementation | Win now, block, centre, corner — the "easy" tier | Optimality |
| `MinimaxPlayer` | Implementation | Alpha-beta search with a transposition table | Rules |
| `SearchStats` | Value object | Nodes visited, table hits, depth reached | — |
| `MoveRecord` | Value object | The move, the mover, the status *before* it | Undo policy |
| `GameObserver` (ABC) | Observer | `on_move`, `on_game_over` | Rendering decisions |
| `Game` | Aggregate root | Turn order, validation, status, history, undo, observer fan-out | I/O, timing, persistence |
| `ConsoleRenderer` | Observer | Draws the board and the result | Game logic |
| `GameRegistry` | Registry | Many concurrent games, one lock each | Matchmaking |

**The split to say out loud:** *"`Board` is a data structure, `WinRule` is the rule strategy, `Player` is the decision-maker, and `Game` is the aggregate that owns turn order and validity. The reason `WinRule` is separate from `Board` is that the board does not know what a win is — the same board supports standard and misère rules with different rule objects."*

---

## 3. Class diagram (ASCII)

```
      ┌────────────┐        ┌──────────────┐        ┌────────────────┐
      │   Cell     │        │    Move      │        │  GameStatus    │
      │ «enum»     │        │ «value»      │        │  «enum»        │
      │ EMPTY X O  │        │ row, col     │        │ ONGOING        │
      │ +opponent  │        └──────┬───────┘        │ X_WINS O_WINS  │
      └────────────┘               │                │ DRAW           │
                                   │                │ +is_over       │
                                   │                └────────────────┘
┌──────────────────────────────────┴───────────────────────────────────┐
│                              Game  (aggregate root)                  │
│──────────────────────────────────────────────────────────────────────│
│ board : Board        rule : WinRule      players : {Cell: Player}    │
│ turn  : Cell         status : GameStatus history : [MoveRecord]      │
│──────────────────────────────────────────────────────────────────────│
│ +play() -> GameResult        +apply(move) -> MoveRecord              │
│ +undo() -> bool              +legal_moves() -> [Move]                │
│ +add_observer(o)             -_validate(move)                        │
│  every move — HUMAN OR BOT — goes through _validate                   │
└───┬───────────────┬─────────────────────────┬────────────────────────┘
    │ owns          │ owns                    │ notifies
    ▼               ▼                         ▼
┌─────────┐  ┌──────────────┐        ┌──────────────────────┐
│ Board   │  │  «interface» │        │   GameObserver       │
│─────────│  │  WinRule     │        │ «interface»          │
│ size    │  │──────────────│        │──────────────────────│
│ grid    │  │ +wins(board, │        │ +on_move(game, ...)  │
│─────────│  │   move, sym) │        │ +on_game_over(...)   │
│ +get/set│  │ +winning_    │        └──────────▲───────────┘
│ +copy   │  │   lines()    │                   │
│ +key    │  └──────▲───────┘          ┌────────┴─────────┐
│ +is_full│         │                  │ ConsoleRenderer  │
│ +empty_ │  ┌──────┴───────────┐      │ MemoryObserver   │
│  cells  │  │ NKLineWinRule    │      │ (tests)          │
└─────────┘  │ k : int          │      └──────────────────┘
             │ DIRECTIONS = 4   │
             │ +wins: walk the  │      ┌──────────────────────────┐
             │  last move's 4   │      │      «interface»         │
             │  axes only       │      │        Player            │
             └──────────────────┘      │──────────────────────────│
                                       │ symbol, name             │
                                       │ +choose_move(board,      │
                                       │      rule, status)       │
                                       └────────────▲─────────────┘
                        ┌───────────────┬───────────┼──────────────┬─────────────┐
                  ┌─────┴──────┐ ┌──────┴─────┐ ┌───┴────────┐ ┌───┴──────────┐
                  │HumanPlayer │ │RandomPlayer│ │Heuristic   │ │MinimaxPlayer │
                  │input_fn    │ │rng         │ │Player      │ │depth, table  │
                  │            │ │            │ │win/block/  │ │alpha-beta    │
                  │            │ │            │ │centre      │ │+ SearchStats │
                  └────────────┘ └────────────┘ └────────────┘ └──────────────┘
```

**Read it this way:** `Game` composes a `Board`, a `WinRule` and a map of `Player`s. `WinRule` is a strategy so the rule set can change without touching the board. `Player` is an interface so a human, a random bot and a search bot are interchangeable — which is also what makes the bot testable.

---

## 4. Interfaces / abstract classes / enums

```python
class WinRule(ABC):
    @abstractmethod
    def wins(self, board: "Board", move: "Move", symbol: "Cell") -> bool:
        """Did `symbol`, having just played `move`, complete a line?
        INCREMENTAL: only the lines through `move` can have changed."""

    @abstractmethod
    def winning_lines(self, board: "Board", symbol: "Cell") -> list[tuple["Move", ...]]:
        """The full scan, for highlighting. Not on the hot path."""


class Player(ABC):
    @abstractmethod
    def choose_move(self, board: "Board", rule: WinRule,
                    status: "GameStatus") -> Optional["Move"]:
        """Return a move, or None to resign. The move is NOT trusted —
        the Game validates it like any other."""


class GameObserver(ABC):
    def on_move(self, game: "Game", move: "Move", symbol: "Cell") -> None: ...
    def on_game_over(self, game: "Game", result: "GameResult") -> None: ...
```

**Three contract notes worth saying out loud:**

- **`wins` is incremental and takes the move.** A rule that scans the whole board on every move is O(n²) for information that only changed along four lines. Passing the move in is what makes the cheap version expressible.
- **`choose_move` returns `None` to resign.** A bot that cannot find a move and a human who quits are the same thing to the game, so they are the same return value.
- **The bot's move is validated by `Game`, not by the bot.** A search bug that returns an occupied cell must produce a rejected move, not a corrupted board. That is the single most important defensive boundary in the design.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** Interview languages vary. In Java this is an interface plus implementations with an `enum` board; in Go, interfaces and a `[][]Cell`; in C++, `virtual` and a `std::vector`. The **rule strategy plus the player interface** is the design; the syntax is not. Python 3.10+ for `X | None`.

```python
"""tic_tac_toe.py — a generalised n x n, k-in-a-row game with a minimax bot.

Design summary
--------------
Cell / Move / GameStatus : values; the board never holds logic.
Board                    : mutable cells + make/unmake, so the search does not
                           allocate a board per node.
WinRule (strategy)       : NKLineWinRule answers "did THIS move complete k in
                           a row?" by walking only the four axes through that
                           move. Eight hard-coded lines cannot become 4x4;
                           this can.
Player (interface)       : Human / Random / Heuristic / Minimax are
                           interchangeable, and every one of their moves is
                           validated by Game like any other.
MinimaxPlayer            : alpha-beta, with a transposition table that caches
                           ONLY exact values — caching a cutoff bound as if it
                           were a value is the classic transposition bug.
Game                     : turn order, validation, status, undo history and
                           observer fan-out. It does no I/O.
"""
from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


# --------------------------------------------------------------------------
# 1. Values
# --------------------------------------------------------------------------

class Cell(Enum):
    EMPTY = "."
    X = "X"
    O = "O"

    @property
    def opponent(self) -> "Cell":
        if self is Cell.X:
            return Cell.O
        if self is Cell.O:
            return Cell.X
        raise ValueError("EMPTY has no opponent")


@dataclass(frozen=True)
class Move:
    row: int
    col: int

    def __str__(self) -> str:
        return f"({self.row},{self.col})"


class GameStatus(Enum):
    ONGOING = "ongoing"
    X_WINS = "x_wins"
    O_WINS = "o_wins"
    DRAW = "draw"

    @property
    def is_over(self) -> bool:
        return self is not GameStatus.ONGOING

    @property
    def winner(self) -> Optional[Cell]:
        if self is GameStatus.X_WINS:
            return Cell.X
        if self is GameStatus.O_WINS:
            return Cell.O
        return None

    @staticmethod
    def for_winner(symbol: Cell) -> "GameStatus":
        return GameStatus.X_WINS if symbol is Cell.X else GameStatus.O_WINS


@dataclass(frozen=True)
class GameResult:
    status: GameStatus
    winning_line: tuple[Move, ...] = ()
    moves_played: int = 0

    @property
    def winner(self) -> Optional[Cell]:
        return self.status.winner


# --------------------------------------------------------------------------
# 2. Board — a data structure, no rules
# --------------------------------------------------------------------------

class Board:
    """Cells plus bounds. `set` is the make/unmake primitive the search uses:
    the minimax does NOT copy the board per node, it places a symbol, recurses,
    and clears the cell. On a 3x3 that is the difference between ~10^5 list
    allocations and none."""

    def __init__(self, size: int = 3, grid: Optional[list[list[Cell]]] = None) -> None:
        if size < 1:
            raise ValueError("board size must be positive")
        self.size = size
        if grid is not None:
            self._grid = [row[:] for row in grid]
        else:
            self._grid = [[Cell.EMPTY] * size for _ in range(size)]

    # -- access ------------------------------------------------------------
    def get(self, row: int, col: int) -> Cell:
        if not self.in_bounds(row, col):
            raise IndexError(f"({row},{col}) is off a {self.size}x{self.size} board")
        return self._grid[row][col]

    def set(self, move: Move, symbol: Cell) -> None:
        if not self.in_bounds(move.row, move.col):
            raise IndexError(f"{move} is off a {self.size}x{self.size} board")
        self._grid[move.row][move.col] = symbol

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_empty(self, move: Move) -> bool:
        return self.in_bounds(move.row, move.col) and \
            self._grid[move.row][move.col] is Cell.EMPTY

    # -- derived -----------------------------------------------------------
    def empty_cells(self) -> list[Move]:
        return [Move(r, c) for r in range(self.size) for c in range(self.size)
                if self._grid[r][c] is Cell.EMPTY]

    def is_full(self) -> bool:
        return all(cell is not Cell.EMPTY for row in self._grid for cell in row)

    def count(self, symbol: Cell) -> int:
        return sum(1 for row in self._grid for cell in row if cell is symbol)

    def copy(self) -> "Board":
        return Board(self.size, self._grid)

    def key(self) -> tuple:
        """A hashable position, used as the transposition-table key."""
        return tuple(cell for row in self._grid for cell in row)

    def render(self, last: Optional[Move] = None) -> str:
        lines = ["  " + " ".join(str(c) for c in range(self.size))]
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                symbol = self._grid[r][c].value
                if last is not None and last.row == r and last.col == c:
                    symbol = f"({symbol})"
                else:
                    symbol = f" {symbol} "
                cells.append(symbol)
            lines.append(f"{r} " + " ".join(cells))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


# --------------------------------------------------------------------------
# 3. The win rule — a strategy, generalised over n and k
# --------------------------------------------------------------------------

class WinRule(ABC):
    @abstractmethod
    def wins(self, board: Board, move: Move, symbol: Cell) -> bool: ...

    @abstractmethod
    def winning_lines(self, board: Board, symbol: Cell) -> list[tuple[Move, ...]]: ...


class NKLineWinRule(WinRule):
    """k consecutive symbols in any of four axes.

    3x3 with k=3 is one configuration of this class, not a special case, which
    is why "now make it 4x4" is a constructor argument.
    """

    DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

    def __init__(self, k: int = 3) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    def wins(self, board: Board, move: Move, symbol: Cell) -> bool:
        """INCREMENTAL: only the four lines through `move` can have changed,
        so this is O(k) rather than O(n^2). This is the method on the hot path
        — the bot calls it at every node of the search."""
        for dr, dc in self.DIRECTIONS:
            run = 1
            for sign in (1, -1):
                r = move.row + dr * sign
                c = move.col + dc * sign
                while board.in_bounds(r, c) and board.get(r, c) is symbol:
                    run += 1
                    r += dr * sign
                    c += dc * sign
            if run >= self.k:
                return True
        return False

    def winning_lines(self, board: Board, symbol: Cell) -> list[tuple[Move, ...]]:
        """The full scan, for highlighting the winning line in a UI. O(n^2 * k)
        and deliberately NOT used by the search."""
        found: list[tuple[Move, ...]] = []
        for r in range(board.size):
            for c in range(board.size):
                if board.get(r, c) is not symbol:
                    continue
                for dr, dc in self.DIRECTIONS:
                    # Only start a run at its first cell, or every position
                    # inside the line would report the same line again.
                    pr, pc = r - dr, c - dc
                    if board.in_bounds(pr, pc) and board.get(pr, pc) is symbol:
                        continue
                    line = []
                    rr, cc = r, c
                    while board.in_bounds(rr, cc) and board.get(rr, cc) is symbol:
                        line.append(Move(rr, cc))
                        rr += dr
                        cc += dc
                    if len(line) >= self.k:
                        found.append(tuple(line))
        return found


class MisereWinRule(NKLineWinRule):
    """Misère: completing a line makes you LOSE. Implemented by inverting the
    caller's decision — which is why `Game` asks the rule for the winner
    rather than assuming the mover won. Shown here to prove the seam exists."""

    def wins(self, board: Board, move: Move, symbol: Cell) -> bool:
        return False        # completing a line is not a win under misère


# --------------------------------------------------------------------------
# 4. Players
# --------------------------------------------------------------------------

class Player(ABC):
    def __init__(self, symbol: Cell, name: str) -> None:
        if symbol is Cell.EMPTY:
            raise ValueError("a player needs X or O")
        self.symbol = symbol
        self.name = name

    @abstractmethod
    def choose_move(self, board: Board, rule: WinRule,
                    status: GameStatus) -> Optional[Move]:
        """A move, or None to resign."""

    def __str__(self) -> str:
        return f"{self.name}({self.symbol.value})"


class HumanPlayer(Player):
    """I/O is INJECTED, so the class has no dependency on stdin and is
    testable with a scripted list of moves."""

    def __init__(self, symbol: Cell, name: str = "human",
                 input_fn: Optional[Callable[[str], str]] = None) -> None:
        super().__init__(symbol, name)
        self.input_fn = input_fn if input_fn is not None else input

    def choose_move(self, board: Board, rule: WinRule,
                    status: GameStatus) -> Optional[Move]:
        prompt = f"{self.name}, enter row and col (e.g. '1 2'), or 'q' to quit: "
        while True:
            raw = self.input_fn(prompt).strip().lower()
            if raw in ("q", "quit", "resign"):
                return None
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                continue
            try:
                move = Move(int(parts[0]), int(parts[1]))
            except ValueError:
                continue
            return move        # Game validates it; the human gets another turn


class RandomPlayer(Player):
    def __init__(self, symbol: Cell, rng: Optional[random.Random] = None) -> None:
        super().__init__(symbol, "random")
        self.rng = rng if rng is not None else random.Random()

    def choose_move(self, board: Board, rule: WinRule,
                    status: GameStatus) -> Optional[Move]:
        cells = board.empty_cells()
        return self.rng.choice(cells) if cells else None


class HeuristicPlayer(Player):
    """The "easy" tier, and the shape most candidates write first: win if you
    can, block if you must, prefer the centre, then a corner, then anything.
    It is beatable and that is the point — it is a difficulty setting, not a
    solver."""

    def __init__(self, symbol: Cell, rng: Optional[random.Random] = None) -> None:
        super().__init__(symbol, "heuristic")
        self.rng = rng if rng is not None else random.Random()

    def choose_move(self, board: Board, rule: WinRule,
                    status: GameStatus) -> Optional[Move]:
        cells = board.empty_cells()
        if not cells:
            return None

        def completes(move: Move, symbol: Cell) -> bool:
            board.set(move, symbol)
            won = rule.wins(board, move, symbol)
            board.set(move, Cell.EMPTY)
            return won

        for move in cells:                                  # win now
            if completes(move, self.symbol):
                return move
        for move in cells:                                  # block
            if completes(move, self.symbol.opponent):
                return move

        centre = Move(board.size // 2, board.size // 2)
        if centre in cells:
            return centre
        corners = [Move(0, 0), Move(0, board.size - 1),
                   Move(board.size - 1, 0), Move(board.size - 1, board.size - 1)]
        open_corners = [m for m in corners if m in cells]
        if open_corners:
            return self.rng.choice(open_corners)
        return self.rng.choice(cells)


@dataclass
class SearchStats:
    nodes: int = 0
    table_hits: int = 0
    cutoffs: int = 0
    depth: int = 0

    def __str__(self) -> str:
        return (f"nodes={self.nodes} table_hits={self.table_hits} "
                f"cutoffs={self.cutoffs} depth={self.depth}")


class MinimaxPlayer(Player):
    """Alpha-beta minimax over the real board, with make/unmake.

    The transposition table stores ONLY exact values. A node whose search was
    cut off by beta has an UPPER BOUND, not a value; caching it and returning
    it later as if it were exact is the classic transposition-table bug, and
    it produces a bot that plays a losing move in a position it "remembers".
    """

    WIN = 1.0
    LOSS = -1.0
    DRAW = 0.0

    def __init__(self, symbol: Cell, depth: Optional[int] = None,
                 use_table: bool = True, name: str = "minimax") -> None:
        super().__init__(symbol, name)
        self.depth = depth
        self.use_table = use_table
        self.stats = SearchStats()
        self._table: dict[tuple, float] = {}

    # -- the public entry point -------------------------------------------
    def choose_move(self, board: Board, rule: WinRule,
                    status: GameStatus) -> Optional[Move]:
        cells = board.empty_cells()
        if not cells:
            return None
        self.stats = SearchStats()
        self._table.clear()

        limit = self.depth
        if limit is None:
            # Search to the end of the game: the number of empty cells is the
            # deepest the search can go, and on a full-width board it will
            # visit every line. Depth-limited play needs a heuristic instead,
            # which is what _evaluate supplies.
            limit = len(cells)

        best_score = float("-inf")
        best_move = cells[0]
        alpha, beta = float("-inf"), float("inf")

        for move in self._ordered(cells, board, rule):
            board.set(move, self.symbol)                      # MAKE
            score = self._search(board, rule, self.symbol.opponent, move,
                                 self.symbol, limit - 1, alpha, beta)
            board.set(move, Cell.EMPTY)                       # UNMAKE
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        self.stats.depth = limit
        return best_move

    def _ordered(self, cells: list[Move], board: Board, rule: WinRule) -> list[Move]:
        """Try the centre first. Move ordering is what makes alpha-beta prune:
        the better the first move, the more of the tree is cut."""
        centre = Move(board.size // 2, board.size // 2)
        return sorted(cells, key=lambda m: 0 if m == centre else 1)

    # -- the search --------------------------------------------------------
    def _search(self, board: Board, rule: WinRule, to_move: Cell,
                last_move: Move, last_symbol: Cell, depth: int,
                alpha: float, beta: float) -> float:
        """Score from the perspective of self.symbol."""
        self.stats.nodes += 1

        if rule.wins(board, last_move, last_symbol):          # incremental
            # Prefer a faster win and a slower loss: `depth` is what remains,
            # so a larger depth means the result arrived sooner.
            base = self.WIN if last_symbol is self.symbol else self.LOSS
            return base * (1.0 + depth * 0.01)
        if board.is_full():
            return self.DRAW
        if depth == 0:
            return self._evaluate(board)

        key = (board.key(), to_move)
        if self.use_table:
            cached = self._table.get(key)
            if cached is not None:
                self.stats.table_hits += 1
                return cached

        alpha_orig = alpha
        if to_move is self.symbol:
            value = float("-inf")
            for move in board.empty_cells():
                board.set(move, to_move)
                value = max(value, self._search(board, rule, to_move.opponent,
                                                move, to_move, depth - 1, alpha, beta))
                board.set(move, Cell.EMPTY)
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.stats.cutoffs += 1
                    break
        else:
            value = float("inf")
            for move in board.empty_cells():
                board.set(move, to_move)
                value = min(value, self._search(board, rule, to_move.opponent,
                                                move, to_move, depth - 1, alpha, beta))
                board.set(move, Cell.EMPTY)
                beta = min(beta, value)
                if beta <= alpha:
                    self.stats.cutoffs += 1
                    break

        # Cache ONLY an exact value. A value that fell outside the original
        # window is a bound, and storing a bound as a value is the bug.
        if self.use_table and alpha_orig < value < beta:
            self._table[key] = value
        return value

    def _evaluate(self, board: Board) -> float:
        """Only reached when the search is depth-limited, which only happens
        on a board too large to solve. On 3x3 the search always reaches a
        terminal position and this never runs."""
        centre = Move(board.size // 2, board.size // 2)
        if not board.is_empty(centre):
            who = board.get(centre.row, centre.col)
            if who is self.symbol:
                return 0.1
            if who is not Cell.EMPTY:
                return -0.1
        return self.DRAW


# --------------------------------------------------------------------------
# 5. Observers
# --------------------------------------------------------------------------

class GameObserver(ABC):
    def on_move(self, game: "Game", move: Move, symbol: Cell) -> None:
        return None

    def on_game_over(self, game: "Game", result: GameResult) -> None:
        return None


class ConsoleRenderer(GameObserver):
    def __init__(self, stream: Optional[Callable[[str], None]] = None) -> None:
        self.write = stream if stream is not None else print

    def on_move(self, game: "Game", move: Move, symbol: Cell) -> None:
        self.write(f"{symbol.value} plays {move}")
        self.write(game.board.render(last=move))

    def on_game_over(self, game: "Game", result: GameResult) -> None:
        if result.winner is not None:
            self.write(f"{result.winner.value} wins on {[str(m) for m in result.winning_line]}")
        else:
            self.write("draw")


class MemoryObserver(GameObserver):
    """For tests: capture the event stream instead of printing it."""

    def __init__(self) -> None:
        self.moves: list[tuple[Move, Cell]] = []
        self.results: list[GameResult] = []

    def on_move(self, game: "Game", move: Move, symbol: Cell) -> None:
        self.moves.append((move, symbol))

    def on_game_over(self, game: "Game", result: GameResult) -> None:
        self.results.append(result)


# --------------------------------------------------------------------------
# 6. The game — turn order, validation, status, undo
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveRecord:
    move: Move
    symbol: Cell
    status_before: GameStatus
    turn_before: Cell


class IllegalMove(Exception):
    pass


class Game:
    def __init__(self, board: Optional[Board] = None, rule: Optional[WinRule] = None,
                 players: Optional[dict[Cell, Player]] = None,
                 first: Cell = Cell.X) -> None:
        self.board = board if board is not None else Board(3)
        self.rule = rule if rule is not None else NKLineWinRule(3)
        self.players: dict[Cell, Player] = players if players is not None else {}
        self.turn = first
        self.status = GameStatus.ONGOING
        self.history: list[MoveRecord] = []
        self._observers: list[GameObserver] = []
        self._last_move: Optional[Move] = None

    # -- setup -------------------------------------------------------------
    def add_observer(self, observer: GameObserver) -> None:
        self._observers.append(observer)

    def set_player(self, player: Player) -> "Game":
        self.players[player.symbol] = player
        return self

    # -- queries -----------------------------------------------------------
    @property
    def last_move(self) -> Optional[Move]:
        return self._last_move

    def legal_moves(self) -> list[Move]:
        """Empty once the game is over — this is what stops a bot from playing
        on in a finished position, and it is checked before the bot is asked
        at all."""
        if self.status.is_over:
            return []
        return self.board.empty_cells()

    def is_legal(self, move: Move) -> bool:
        return not self.status.is_over and self.board.is_empty(move)

    # -- the move path -----------------------------------------------------
    def _validate(self, move: Move) -> None:
        """Every move passes through here, INCLUDING a bot's. A search bug that
        returns an occupied or off-board cell must produce a rejected move, not
        a corrupted board."""
        if self.status.is_over:
            raise IllegalMove(f"the game is over ({self.status.value})")
        if not self.board.in_bounds(move.row, move.col):
            raise IllegalMove(f"{move} is off a {self.board.size}x{self.board.size} board")
        if not self.board.is_empty(move):
            raise IllegalMove(f"{move} is already occupied")

    def apply(self, move: Move) -> MoveRecord:
        self._validate(move)
        symbol = self.turn
        record = MoveRecord(move, symbol, self.status, self.turn)

        self.board.set(move, symbol)
        self._last_move = move
        self.history.append(record)

        # ORDER MATTERS: check the win FIRST. On a full board the last move can
        # complete a line, and testing is_full() first would report a draw in
        # a won game — the classic tic-tac-toe bug.
        if self.rule.wins(self.board, move, symbol):
            self.status = GameStatus.for_winner(symbol)
        elif self.board.is_full():
            self.status = GameStatus.DRAW
        else:
            self.turn = self.turn.opponent

        for observer in self._observers:
            observer.on_move(self, move, symbol)
        if self.status.is_over:
            result = self.result()
            for observer in self._observers:
                observer.on_game_over(self, result)
        return record

    def undo(self) -> bool:
        """Restores the board cell, the turn, the status and the last-move
        marker from the record. Deriving the status again would be wrong after
        a win, because the win check needs the move that caused it."""
        if not self.history:
            return False
        record = self.history.pop()
        self.board.set(record.move, Cell.EMPTY)
        self.status = record.status_before
        self.turn = record.turn_before
        self._last_move = self.history[-1].move if self.history else None
        return True

    def result(self) -> GameResult:
        """The terminal summary. Public because a caller that wants the winning
        line for a UI should not have to reach into the history to find it."""
        line: tuple[Move, ...] = ()
        if self.status.winner is not None and self._last_move is not None:
            for candidate in self.rule.winning_lines(self.board, self.status.winner):
                if self._last_move in candidate:
                    line = candidate
                    break
        return GameResult(self.status, line, len(self.history))

    # -- the loop ----------------------------------------------------------
    def play(self, max_moves: int = 1000) -> GameResult:
        """Ask the current player for a move, validate and apply it. A player
        that returns an illegal move is the GAME's problem, so it is handled
        here: the game does not trust the bot."""
        for _ in range(max_moves):
            if self.status.is_over:
                break
            player = self.players.get(self.turn)
            if player is None:
                raise RuntimeError(f"no player registered for {self.turn}")
            move = player.choose_move(self.board, self.rule, self.status)
            if move is None:                       # resign
                self.status = GameStatus.for_winner(self.turn.opponent)
                result = self.result()
                for observer in self._observers:
                    observer.on_game_over(self, result)
                return result
            try:
                self.apply(move)
            except IllegalMove:
                # Do not corrupt the board and do not spin: a player that
                # cannot produce a legal move loses. The alternative — asking
                # again — is an infinite loop against a buggy bot.
                self.status = GameStatus.for_winner(self.turn.opponent)
                result = self.result()
                for observer in self._observers:
                    observer.on_game_over(self, result)
                return result
        return self.result()


class TimeoutExpired(IllegalMove):
    """A move that arrives after the deadline is not a legal move. It derives
    from IllegalMove so the existing play() loop forfeits the timed-out player
    with no change to Game — but the message is specific, because "you ran out
    of time" and "that square is taken" are different things to a player."""


class TimedGame(Game):
    """A move clock. The clock is INJECTED, the same seam as the rate limiter's
    Clock, so a timeout is a unit test with a fake clock rather than a
    thirty-second wait. The deadline is enforced by the arbiter in apply(),
    not by the player honouring a budget it was handed, because a player that
    polices its own clock is a player that can decline to."""

    def __init__(self, *args, budget_seconds: float = 30.0,
                 clock: Optional[Callable[[], float]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.budget = budget_seconds
        self.clock = clock if clock is not None else time.monotonic
        self._deadline = self.clock() + self.budget

    def apply(self, move: Move) -> MoveRecord:
        if self.status.is_over:
            raise IllegalMove(f"the game is over ({self.status.value})")
        if self.clock() > self._deadline:
            self.status = GameStatus.for_winner(self.turn.opponent)
            raise TimeoutExpired(f"{self.turn.value} ran out of time")
        record = super().apply(move)
        self._deadline = self.clock() + self.budget      # restart for the next turn
        return record


# --------------------------------------------------------------------------
# 7. Many games at once
# --------------------------------------------------------------------------

class GameRegistry:
    """One lock per game, not one lock for the registry.

    A game is single-threaded BY DESIGN — the board, the turn and the history
    are one mutable aggregate, and sharing it between threads needs one lock
    around the whole aggregate, which is exactly a per-game lock. A bot's
    search can run outside the lock on a COPY, then apply under the lock, so a
    slow search never blocks another game.
    """

    def __init__(self) -> None:
        self._games: dict[str, Game] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def create(self, game_id: str, **kwargs) -> Game:
        with self._registry_lock:
            if game_id in self._games:
                raise ValueError(f"game {game_id} already exists")
            self._games[game_id] = Game(**kwargs)
            self._locks[game_id] = threading.Lock()
            return self._games[game_id]

    def lock_for(self, game_id: str) -> threading.Lock:
        with self._registry_lock:
            return self._locks[game_id]

    def play_move(self, game_id: str, move: Move) -> MoveRecord:
        with self.lock_for(game_id):
            return self._games[game_id].apply(move)

    def snapshot(self, game_id: str) -> str:
        with self.lock_for(game_id):
            return self._games[game_id].board.render()


# --------------------------------------------------------------------------
# 8. Tests and demo — correctness proved, not asserted
# --------------------------------------------------------------------------

def _check(name: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    if not condition:
        raise SystemExit(1)


def _plain_minimax_nodes(board: Board, rule: WinRule, player: MinimaxPlayer) -> int:
    """Count every node a full-width, no-cutoff search visits, so the pruning
    claim in test 11 is measured rather than asserted."""
    counter = {"n": 0}

    def walk(b: Board, to_move: Cell, last: Optional[Move], last_symbol: Cell) -> float:
        counter["n"] += 1
        if last is not None and rule.wins(b, last, last_symbol):
            return 1.0 if last_symbol is player.symbol else -1.0
        if b.is_full():
            return 0.0
        best = float("-inf") if to_move is player.symbol else float("inf")
        for m in b.empty_cells():
            b.set(m, to_move)
            score = walk(b, to_move.opponent, m, to_move)
            b.set(m, Cell.EMPTY)
            best = max(best, score) if to_move is player.symbol else min(best, score)
        return best

    for first in board.empty_cells():
        board.set(first, Cell.X)
        walk(board, Cell.O, first, Cell.X)
        board.set(first, Cell.EMPTY)
    return counter["n"]


if __name__ == "__main__":
    rule = NKLineWinRule(3)

    # --- 1. all eight winning lines are detected ---
    lines = [
        [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)],
    ]
    detected = 0
    for line in lines:
        board = Board(3)
        last = None
        for r, c in line:
            move = Move(r, c)
            board.set(move, Cell.X)
            last = move
        if rule.wins(board, last, Cell.X):
            detected += 1
    _check("all eight lines are detected on 3x3", detected == 8)

    # --- 2. a nearly-complete line is not a win ---
    board = Board(3)
    board.set(Move(0, 0), Cell.X)
    board.set(Move(0, 1), Cell.X)
    _check("two in a row is not a win", not rule.wins(board, Move(0, 1), Cell.X))

    # --- 3. the rule generalises: 4x4 with k=4 is a constructor argument ---
    rule4 = NKLineWinRule(4)
    board4 = Board(4)
    for i in range(4):
        board4.set(Move(i, i), Cell.O)
    _check("4x4 with k=4 detects a diagonal", rule4.wins(board4, Move(3, 3), Cell.O))
    board4.set(Move(2, 2), Cell.EMPTY)          # punch a gap in the diagonal
    _check("three marked with a gap is NOT a win at k=4",
           not rule4.wins(board4, Move(1, 1), Cell.O))
    _check("and the same three are NOT a win at k=3 either",
           not NKLineWinRule(3).wins(board4, Move(1, 1), Cell.O))
    board5 = Board(5)
    for i in range(5):
        board5.set(Move(0, i), Cell.X)
    _check("5x5 with k=5 detects a row", NKLineWinRule(5).wins(board5, Move(0, 4), Cell.X))

    # The same board, two rules: this is what "the rule is a strategy" means.
    board6 = Board(5)
    for i in range(3):
        board6.set(Move(0, i), Cell.X)
    _check("three in a row on 5x5 IS a win when k=3",
           NKLineWinRule(3).wins(board6, Move(0, 2), Cell.X))
    _check("and is NOT a win when k=5",
           not NKLineWinRule(5).wins(board6, Move(0, 2), Cell.X))

    # --- 4. winning_lines finds the line for highlighting ---
    board = Board(3)
    for c in range(3):
        board.set(Move(1, c), Cell.X)
    found = rule.winning_lines(board, Cell.X)
    _check("winning_lines returns exactly one line", len(found) == 1)
    _check("and it is the middle row",
           [str(m) for m in found[0]] == ["(1,0)", "(1,1)", "(1,2)"])

    # --- 5. draw detection, and the order of the two checks ---
    # X wins on the LAST move of a full board: is_full() must be checked AFTER
    # the win, or this is reported as a draw. The board below is built so that
    # only move 9 completes a line:
    #     X X X
    #     X O O
    #     O O X
    game = Game(Board(3), rule)
    for move in [Move(0, 0), Move(1, 1), Move(1, 0), Move(2, 0), Move(2, 2),
                 Move(1, 2), Move(0, 1), Move(2, 1)]:
        game.apply(move)
    _check("with eight played and no line, the game is ONGOING",
           game.status is GameStatus.ONGOING and not game.board.is_full())
    game.apply(Move(0, 2))
    _check("the last move both fills the board AND wins",
           game.board.is_full() and game.status is GameStatus.X_WINS)
    _check("the result names the winning line", len(game.result().winning_line) == 3)

    # A genuine draw: no line, board full.
    #     X O X
    #     X O O
    #     O X X
    game_draw = Game(Board(3), rule)
    for move in [Move(0, 0), Move(0, 1), Move(0, 2), Move(1, 1), Move(1, 0),
                 Move(2, 0), Move(2, 1), Move(1, 2)]:
        game_draw.apply(move)
    _check("eight played with no line is still ONGOING",
           game_draw.status is GameStatus.ONGOING)
    game_draw.apply(Move(2, 2))
    _check("a full board with no line is a DRAW",
           game_draw.status is GameStatus.DRAW)
    _check("a draw has no winning line", game_draw.result().winning_line == ())

    # --- 6. illegal moves are rejected, including a bot's ---
    g = Game(Board(3), rule)
    g.apply(Move(1, 1))
    try:
        g.apply(Move(1, 1))
        _check("an occupied cell is rejected", False)
    except IllegalMove as exc:
        _check("an occupied cell is rejected", "already occupied" in str(exc))
    try:
        g.apply(Move(5, 5))
        _check("an off-board move is rejected", False)
    except IllegalMove as exc:
        _check("an off-board move is rejected", "off a 3x3 board" in str(exc))

    class BrokenBot(Player):
        def __init__(self) -> None:
            super().__init__(Cell.O, "broken")

        def choose_move(self, board, rule, status):
            return Move(1, 1)          # always tries the same occupied cell

    g.set_player(BrokenBot())
    result = g.play()
    _check("a bot that cannot move legally loses instead of looping",
           result.status is GameStatus.X_WINS)

    # --- 7. undo restores the exact position, including after a win ---
    g2 = Game(Board(3), rule)
    for move in [Move(0, 0), Move(1, 0), Move(0, 1), Move(1, 1), Move(0, 2)]:
        g2.apply(move)
    _check("X has won", g2.status is GameStatus.X_WINS)
    key_before, turn_before = g2.board.key(), g2.turn
    g2.undo()
    _check("undo clears the winning move",
           g2.board.is_empty(Move(0, 2)) and g2.status is GameStatus.ONGOING)
    _check("undo restores the turn to the player who moved",
           g2.turn is Cell.X)
    _check("undo restores the last-move marker",
           g2.last_move == Move(1, 1))
    g2.apply(Move(0, 2))
    _check("re-playing the move restores the won position",
           g2.status is GameStatus.X_WINS and g2.board.key() == key_before)
    while g2.undo():
        pass
    _check("undoing everything empties the board",
           g2.board.key() == Board(3).key() and g2.history == [])

    # --- 8. minimax never loses to a random player ---
    losses = 0
    wins = 0
    draws = 0
    for seed in range(60):
        rng = random.Random(seed)
        bot = MinimaxPlayer(Cell.O)
        game = Game(Board(3), rule, {
            Cell.X: RandomPlayer(Cell.X, random.Random(seed * 7 + 1)),
            Cell.O: bot,
        })
        result = game.play()
        if result.status is GameStatus.X_WINS:
            losses += 1
        elif result.status is GameStatus.O_WINS:
            wins += 1
        else:
            draws += 1
    _check("minimax never loses to random across 60 games", losses == 0)
    _check("and it wins most of them", wins + draws == 60 and wins >= 40)

    # --- 9. minimax as X beats a random player quickly ---
    bot_x = MinimaxPlayer(Cell.X)
    game = Game(Board(3), rule, {
        Cell.X: bot_x,
        Cell.O: RandomPlayer(Cell.O, random.Random(11)),
    })
    result = game.play()
    _check("minimax moving first does not lose",
           result.status is not GameStatus.O_WINS)

    # --- 10. minimax versus minimax draws (perfect play) ---
    game = Game(Board(3), rule, {
        Cell.X: MinimaxPlayer(Cell.X),
        Cell.O: MinimaxPlayer(Cell.O, name="minimax-o"),
    })
    result = game.play()
    _check("perfect play on both sides is a draw", result.status is GameStatus.DRAW)
    _check("and the board is full", result.moves_played == 9)

    # --- 11. alpha-beta prunes, and the table is only an optimisation ---
    board = Board(3)
    with_cuts = MinimaxPlayer(Cell.X, use_table=False)
    with_cuts.choose_move(board, rule, GameStatus.ONGOING)
    _check("alpha-beta produces cutoffs", with_cuts.stats.cutoffs > 0)

    plain = MinimaxPlayer(Cell.X, use_table=False)
    plain_stats = _plain_minimax_nodes(board, rule, plain)
    _check("alpha-beta visits fewer nodes than a full search",
           with_cuts.stats.nodes < plain_stats)

    cached = MinimaxPlayer(Cell.X, use_table=True)
    cached.choose_move(board, rule, GameStatus.ONGOING)
    _check("the transposition table gets hits", cached.stats.table_hits > 0)
    _check("and does not change the chosen move",
           cached.choose_move(board, rule, GameStatus.ONGOING) ==
           MinimaxPlayer(Cell.X, use_table=False).choose_move(board, rule, GameStatus.ONGOING))

    # --- 12. the opening move is the centre or a corner, never an edge ---
    opener = MinimaxPlayer(Cell.X)
    best = opener.choose_move(Board(3), rule, GameStatus.ONGOING)
    _check("the optimal opening is the centre on an empty 3x3",
           best == Move(1, 1))

    # A non-losing opening after the centre is taken is a corner.
    board_c = Board(3)
    board_c.set(Move(1, 1), Cell.O)
    second = MinimaxPlayer(Cell.X).choose_move(board_c, rule, GameStatus.ONGOING)
    _check("with the centre taken, a corner is chosen",
           second in (Move(0, 0), Move(0, 2), Move(2, 0), Move(2, 2)))

    # --- 13. the heuristic tier blocks an immediate loss ---
    hb = Board(3)
    hb.set(Move(0, 0), Cell.O)
    hb.set(Move(0, 1), Cell.O)          # O threatens (0,2)
    hb.set(Move(1, 1), Cell.X)
    block = HeuristicPlayer(Cell.X, random.Random(3)).choose_move(hb, rule, GameStatus.ONGOING)
    _check("the heuristic player blocks the immediate loss", block == Move(0, 2))

    # --- 14. the observer stream is complete and ordered ---
    g3 = Game(Board(3), rule)
    mem = MemoryObserver()
    g3.add_observer(mem)
    for move in [Move(0, 0), Move(1, 0), Move(0, 1), Move(1, 1), Move(0, 2)]:
        g3.apply(move)
    _check("every move was observed", len(mem.moves) == 5)
    _check("the symbols alternate starting with X",
           [s for _, s in mem.moves] == [Cell.X, Cell.O, Cell.X, Cell.O, Cell.X])
    _check("game over fired exactly once", len(mem.results) == 1)
    _check("with the right status", mem.results[0].status is GameStatus.X_WINS)

    # --- 15. legal_moves is empty once the game is over ---
    _check("no legal moves after the win", g3.legal_moves() == [])
    _check("and is_legal is false for an empty cell", not g3.is_legal(Move(2, 2)))
    try:
        g3.apply(Move(2, 2))
        _check("a move after game over is rejected", False)
    except IllegalMove as exc:
        _check("a move after game over is rejected", "game is over" in str(exc))

    # --- 16. depth-limited play on a board too large to solve ---
    big = Board(6)
    rule5 = NKLineWinRule(5)
    limited = MinimaxPlayer(Cell.X, depth=2)
    mv = limited.choose_move(big, rule5, GameStatus.ONGOING)
    _check("a depth-limited bot still returns a legal move on 6x6",
           mv is not None and big.is_empty(mv))
    _check("and the heuristic evaluation was reached", limited.stats.depth == 2)

    # --- 17. misere is a different rule object over the same board ---
    mis = Game(Board(3), MisereWinRule(3), {
        Cell.X: HeuristicPlayer(Cell.X, random.Random(5)),
        Cell.O: HeuristicPlayer(Cell.O, random.Random(6)),
    })
    mis_result = mis.play()
    _check("misere terminates without a line win",
           mis_result.status in (GameStatus.DRAW, GameStatus.X_WINS, GameStatus.O_WINS))
    _check("the misere rule itself never reports a win",
           not MisereWinRule(3).wins(mis.board, Move(0, 0), Cell.X))

    # --- 18. the registry locks per game ---
    registry = GameRegistry()
    registry.create("g1", board=Board(3), rule=rule)
    registry.create("g2", board=Board(3), rule=rule)
    registry.play_move("g1", Move(0, 0))
    registry.play_move("g2", Move(2, 2))
    _check("two games keep independent boards",
           registry.snapshot("g1") != registry.snapshot("g2"))
    _check("the locks are distinct objects",
           registry.lock_for("g1") is not registry.lock_for("g2"))

    # --- 19. the move clock, driven by an injected fake clock (no sleeping) ---
    now = {"t": 0.0}
    timed = TimedGame(Board(3), rule, budget_seconds=5.0, clock=lambda: now["t"])
    timed.apply(Move(1, 1))                 # X moves at t=0; deadline t=5
    _check("the first move is inside the budget", timed.turn is Cell.O)
    now["t"] = 4.0                          # O thinks for four seconds
    timed.apply(Move(0, 0))                 # legal at t=4; deadline becomes t=9
    _check("a move inside the budget is applied", timed.turn is Cell.X)
    now["t"] = 100.0                        # X takes a minute
    try:
        timed.apply(Move(2, 2))
        _check("a move past the deadline is rejected", False)
    except TimeoutExpired:
        _check("a move past the deadline is rejected", True)
    _check("and the slow player forfeits on time",
           timed.status is GameStatus.O_WINS)

    print("\nall checks passed")
```

**What to point at:**

- `NKLineWinRule.wins` walks **only the four axes through the last move** — O(k), not O(n²) — and the method takes `move` precisely so the cheap version is expressible.
- `apply` checks the **win before `is_full()`**, with the comment saying why: on a full board the last move can complete a line, and the other order reports a draw in a won game. The demo builds exactly that position.
- The minimax **makes and unmakes on the real board** rather than copying it, and the transposition table stores **only exact values**, with the comment naming the bug it avoids.
- `Game._validate` is called for **every** move including a bot's, and `play()` handles an `IllegalMove` from a bot by awarding the game to the opponent — the alternative is an infinite loop against a broken bot.
- `legal_moves()` returns empty once the game is over, so a bot is never asked to move in a finished position.
- `MisereWinRule` exists to prove the `WinRule` seam is real: the same board, a different rule object.

---

## 6. Concurrency & thread-safety

### What breaks

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Two threads on one game** | The board, the turn and the history are one aggregate; two `apply` calls interleave and the turn desynchronises from the board | One lock per game around the whole aggregate — the same shape as `GameRegistry` |
| **One lock for all games** | A slow bot search in game A blocks every move in game B | A lock per game id, and a registry lock only for map lookup |
| **Bot search holding the lock** | A 6×6 depth-limited search holds the per-game lock for a second and the UI freezes | Search on a **copy** outside the lock, then re-validate and apply under the lock |
| **The copy going stale** | The board changed while the bot was searching, so its move is now illegal | Re-validate under the lock; on failure, discard the move and re-search or forfeit |
| **Rendering during a move** | The renderer reads the board mid-mutation and draws a half-applied move | Render under the same lock, from the observer callback |
| **Observers mutating the game** | An observer calls `apply`, re-entering the aggregate mid-notification | Notify after the state is fully updated, and make observers read-only by convention |
| **A read-only query during a search** | `is_full()` on the live board while the search has temporarily placed a symbol | Never share the live board with a search; the search owns a copy or the lock |
| **Two games sharing a `Board`** | `Game(board=b)` with the same `b` for two games — one board, two turn counters | Construct a board per game; the registry does |

**The one to volunteer:** the **bot search holding the lock**. It is the concurrency question that makes this problem interesting: a perfect 3×3 bot is instant, but the design must already be the shape that survives a 6×6 board with a depth limit, and that shape is *search on a copy, apply under the lock, re-validate*.

### Threading the bot safely

```python
def bot_move_safely(game: Game, player: Player, lock: threading.Lock) -> Optional[Move]:
    """Search on a copy outside the lock; apply under the lock and re-validate.
    This is the pattern that keeps a slow search from blocking the game."""
    with lock:                              # cheap: take a snapshot
        snapshot = game.board.copy()
        status = game.status
        rule = game.rule
    if status.is_over:
        return None
    move = player.choose_move(snapshot, rule, status)   # slow, no lock held
    with lock:
        if move is None or not game.is_legal(move):
            return None                     # the position changed; discard
        game.apply(move)
        return move
```

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Make it 4×4"

A constructor argument, which is the whole reason `WinRule` is a strategy:

```python
game = Game(Board(4), NKLineWinRule(4))
```

Nothing else changes. Say the interesting part: at 4×4 with k=4 the game is a **draw under perfect play** (no forced win for either side), so a user expecting a solvable game gets a different experience — that is a product consequence of a rule change, not a bug.

### 7.2 "Play as a network service"

The server must be authoritative — the client sends a move, never a board:

```python
class GameService:
    """The client sends `game_id` and a move; it never sends a board. A client
    that sends its own board state is a client that can cheat."""

    def __init__(self) -> None:
        self.registry = GameRegistry()

    def move(self, game_id: str, row: int, col: int, player_token: str) -> dict:
        lock = self.registry.lock_for(game_id)
        with lock:
            game = self.registry._games[game_id]
            expected = game.players.get(game.turn)
            if expected is None or expected.name != player_token:
                return {"error": "not your turn"}
            try:
                game.apply(Move(row, col))
            except IllegalMove as exc:
                return {"error": str(exc)}
            return {
                "status": game.status.value,
                "board": game.board.render(),
                "turn": game.turn.value,
            }
```

Two points: the **turn check is server-side** (a client claiming it is their turn means nothing), and the response carries the authoritative board rather than diffing a client's.

### 7.3 "Undo, and redo"

Undo is a stack pop; redo is a second stack. The subtlety worth naming is that **redo must be invalidated by a new move** — otherwise a player undoes, plays elsewhere, then redoes the abandoned branch and the board takes both:

```python
class UndoRedoGame(Game):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._redo: list[MoveRecord] = []

    def apply(self, move: Move) -> MoveRecord:
        self._redo.clear()          # a new move abandons the redo branch
        return super().apply(move)

    def undo(self) -> bool:
        if not self.history:
            return False
        self._redo.append(self.history[-1])
        super().undo()
        return True

    def redo(self) -> Optional[MoveRecord]:
        if not self._redo:
            return None
        move = self._redo.pop().move
        return self.apply(move)
```

### 7.4 "Difficulty tiers"

Three bots already exist, and the tiering is a policy, not a new class:

```python
def player_for_tier(tier: str, symbol: Cell, rng: random.Random) -> Player:
    if tier == "easy":
        return RandomPlayer(symbol, rng)
    if tier == "medium":
        return HeuristicPlayer(symbol, rng)
    if tier == "hard":
        return MinimaxPlayer(symbol)
    if tier == "impossible":
        return MinimaxPlayer(symbol, use_table=True)
    raise ValueError(f"unknown tier: {tier}")
```

The honest note: on 3×3, "hard" and "impossible" are the **same bot**, because perfect play is achievable. A meaningful difficulty ladder needs a larger board, or deliberate imperfection in the strong bot (play the best move with probability p, a random legal move otherwise).

### 7.5 "A move clock"

Timing is a `Game` concern, not a `Player` concern:

```python
# The class lives in section 5 (it must be part of the runnable program, not a
# fragment). What matters here is how it is CONSTRUCTED: the clock is a
# parameter, so a test drives time by hand and never sleeps.
now = {"t": 0.0}
timed = TimedGame(Board(3), NKLineWinRule(3), budget_seconds=5.0,
                  clock=lambda: now["t"])
timed.apply(Move(1, 1))          # X moves at t=0, deadline becomes t=5
now["t"] = 4.0                   # O thinks for four seconds
timed.apply(Move(0, 0))          # legal: 4 <= 5, deadline becomes t=9
now["t"] = 100.0
timed.apply(Move(2, 2))          # raises TimeoutExpired; O wins on time
```

A `Clock` port rather than `time.monotonic()` called inline is the difference between a timeout test that runs in microseconds and one that runs in thirty seconds — the identical argument as [the rate limiter's `Clock`](11-rate-limiter.md) §5, and the identical argument against a bot policing its own budget.
The design point is the seam: the clock is **injected** exactly as in [the rate limiter](11-rate-limiter.md), so a timeout is a unit test rather than a 30-second wait — and the timeout is enforced by the arbiter, not by the bot honouring its own budget.

### 7.6 "Serialise a game"

State is small enough that a snapshot is the position plus the history:

```python
def to_dict(game: Game) -> dict:
    return {
        "size": game.board.size,
        "k": getattr(game.rule, "k", 3),
        "turn": game.turn.value,
        "status": game.status.value,
        "cells": [c.value for c in game.board.key()],
        "history": [[r.move.row, r.move.col, r.symbol.value]
                    for r in game.history],
    }


def from_dict(data: dict) -> Game:
    board = Board(data["size"])
    game = Game(board, NKLineWinRule(data["k"]), first=Cell(data["turn"]))
    for row, col, _symbol in data["history"]:
        game.apply(Move(row, col))      # replay rebuilds the history properly
    return game
```

The design point: **replay rather than restore**. Reconstructing the board from the cell array and setting the status directly would lose the history, and a game restored without its history cannot be undone — so the serialised form is the move list, and the board is derived.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Win checked after `is_full()`** | The last-move win on a full board is reported as a draw | Check the win first; the demo builds this exact position |
| 2 | **Eight hard-coded lines** | Cannot answer "make it 4×4" without a rewrite | The *k*-in-a-row scan with a direction table |
| 3 | **Full-board scan per move** | O(n²) per move for information that changed along four axes | The incremental check from the last move |
| 4 | **Move on an occupied cell** | The board silently overwrites, and a piece vanishes | `_validate` rejects it before any mutation |
| 5 | **Move after the game is over** | The game continues; a second winner appears | `legal_moves()` is empty and `apply` raises |
| 6 | **A bot returning an illegal move** | The board is corrupted, or the game loops asking again forever | The game validates and forfeits the bot; never trust a move |
| 7 | **No legal moves and no winner** | A bot is asked to move and crashes on an empty list | `is_full()` before asking; `choose_move` returning `None` is a resign |
| 8 | **Copying the board per search node** | Thousands of allocations; a 6×6 search crawls | Make/unmake on the real board |
| 9 | **Caching an alpha-beta bound as a value** | The bot plays a losing move in a position it "remembers" | Cache only values from a window that did not cut off |
| 10 | **Unbounded search on a large board** | Recursion depth and minutes per move | A depth limit plus a heuristic evaluation |
| 11 | **`k` greater than the board size** | The game can never be won, and a UI waits forever for a win | It degenerates to a draw; state it, and validate `k <= 2n-1` at construction |
| 12 | **Undo after a win** | Deriving the status again misses the win, because the winning move is gone | Restore the status from the `MoveRecord` |
| 13 | **Redo after a different move** | The abandoned branch is replayed on top of the new one | A new move clears the redo stack |
| 14 | **Two players with the same symbol** | One silently replaces the other in the players map | Validate symbols are distinct at setup |
| 15 | **Non-square boards** | Row and column indices cross over; the win rule walks off the board | The board is square by construction here; a rectangular board needs separate width and height, and the demo's test 16 uses a square 6×6 |
| 16 | **A shared `Board` between two games** | One board, two turn counters; moves appear in both games | A board per game, created by the registry |
| 17 | **An observer calling back into the game** | Re-entrant mutation mid-notification | Notify after the state is consistent; observers are read-only |
| 18 | **Misère implemented by inverting the check inside `Game`** | The rule and the game each own half of the rule set | Invert inside the rule; `Game` asks the rule, never assumes |
| 19 | **The heuristic bot's lookahead mutating the board** | A half-placed symbol survives a check | `completes()` sets and clears the cell in the same method |
| 20 | **Resignation modelled as an exception** | An exception used for a normal game outcome | `None` from `choose_move` is a resign |

**The three to volunteer:** #1 (the check order), #6 (never trust a bot's move) and #9 (the transposition-table bound bug). The first is the classic tic-tac-toe bug, the second is the defensive boundary that separates a toy from a service, and the third is the one that shows you have actually written a search.

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **Strategy** | `WinRule` | The rule set is the thing that varies: *k*-in-a-row, misère, forbidden moves. Keeping it out of `Board` is what makes "make it 4×4" a constructor argument | One indirection per win check, on the search's hottest path |
| **Strategy** | `Player` | Human, random, heuristic and minimax are interchangeable, which is also what makes the game testable without a human | The interface must be honest about resign and about being untrusted |
| **Observer** | `GameObserver` | The game does no I/O; a console renderer, a test recorder and a network push are all the same seam | Fan-out must happen after the state is consistent |
| **Value Object** | `Move`, `GameResult`, `MoveRecord`, `SearchStats` | Immutable, hashable, safe to log and to put in the history | One allocation per move — trivial next to the search |
| **Memento** | `MoveRecord` in `history` | Undo needs the status and turn *before* the move, not a recomputation | The history grows with the game; it is bounded by the board size |
| **Template Method** | *Deliberately not used for `Player`* | The four players share no skeleton — one reads input, one is random, one scans, one searches. A base `choose_move` with hooks would be a forced abstraction | — |
| **Registry** | `GameRegistry` | Games are addressed by id and each needs its own lock | A second map to keep in step with the first |
| **Singleton** | *Deliberately rejected* | The game, the rule and the players are all injected, so a test builds its own | — |

**The sentence to give:** *"Strategy twice — for the rule and for the player — because those are the two axes the problem varies along. Observer for the UI seam, Memento for undo, Value Object for the moves. I would explicitly not use Template Method for the players: they share no skeleton, so a base class with hooks would be a forced abstraction, and the same reasoning appears in the [rate limiter](11-rate-limiter.md) where the four algorithms share no arithmetic."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Generalised *n*×*n*, *k*-in-a-row** | Yes | "Make it 4×4" is a constructor argument; the 3×3 case is a configuration | A direction-table scan instead of eight readable lines; slightly more code for the common case |
| **Incremental win check** | Yes | O(k) per move instead of O(n²) — and it is called at every search node | The method must take the last move, which is a slightly odd signature |
| **Mutable board with make/unmake** | Yes | No allocation in the search; the 3×3 search is instant | The board is not safe to share with a search; the search must own it |
| **Minimax with alpha-beta** | Yes | Perfect play on 3×3, provably (the demo shows 60 games without a loss) | Exponential without depth limits; needs a heuristic on larger boards |
| **Transposition table, exact values only** | Yes | Reuses work; the demo shows table hits | A subtly wrong table is worse than none — bound caching is a real bug |
| **Heuristic player kept** | Yes | A difficulty tier, and the shape most candidates write first | It is beatable, which must be an intentional product decision |
| **Undo via a Memento history** | Yes | Correct undo after a win, where recomputation would fail | Memory grows with the move count |
| **Observers for I/O** | Yes | The game is testable with no console and no mocking of `print` | Notification ordering becomes a design concern |
| **Every move validated** | Yes | A buggy bot cannot corrupt the game | One extra check per move, and a bot forfeit rule that must be documented |
| **Per-game lock** | Yes | Many concurrent games; a slow search in one does not block another | A registry map plus a lock map to keep in sync |
| **Resign as `None`** | Yes | No exceptions for a normal outcome | `None` must be handled at every call site — one place, in `play()` |
| **Replay-based serialisation** | Yes | A restored game can still be undone | Restoring is O(moves) |

**The trade to name unprompted:** *"Generalising costs readability on the 3×3 case: eight `if` statements become a direction table. I take that trade because the follow-up question is always 'now make it 4×4', and a design that cannot answer it in one line was never really a design. The place I would not generalise is the board geometry — a rectangular board is a separate change with separate off-by-one risks, and doing it speculatively would complicate the win rule for no current requirement."*

---

## 11. Q&A

**Q1. The problem is trivial. What is actually being assessed?**
Three things beyond the board. First, whether you generalise the win condition, because "make it 4×4" is the guaranteed follow-up and hard-coded lines cannot answer it. Second, whether you separate the rule from the board and the player from the game, because that separation is the difference between a script and a design. Third, whether you handle the cases everyone forgets: the win checked before the draw, a bot's move validated like anyone else's, and a draw when the board fills with no line. The minimax is the fourth — it is the part that shows whether you can write an algorithm inside a class design rather than beside it.

**Q2. Why is the win check incremental rather than a full scan?**
Because only the four lines through the last move can have changed, and everything else on the board is provably unchanged. So the check walks four axes from that move and stops — O(k) instead of O(n²). That matters because the check is not called once per game; it is called at **every node of the minimax search**, where a full scan would dominate the cost. Passing the move into `wins` is what makes the cheap version expressible at all, which is why the signature looks unusual — it takes information the rule could in principle derive, because deriving it is the expensive part.

**Q3. Why is `WinRule` a separate class rather than methods on `Board`?**
Because the board does not know what a win is, and different games over the same board have different rules. Misère — where completing a line makes you lose — uses the identical board and a different rule object, which is exactly what the demo's `MisereWinRule` shows. The same seam covers *k*-in-a-row for any *k*, forbidden-move rules and scoring rules. If the win condition lives on the board, every rule variant edits the board, and the board accumulates a flag for each game type.

**Q4. Why make/unmake rather than copying the board at each search node?**
Because the search visits hundreds of thousands of nodes and each copy is an `n²` list allocation. Make/unmake places one symbol, recurses, and clears one cell — no allocation at all, and the position is identical at every point in the tree. The cost is that the board is now **stateful and unsafe to share**: the search owns the board for its duration, and anything else that reads it concurrently sees intermediate positions. That is why the concurrency section says the search runs on a copy when the game is being served concurrently.

**Q5. What does alpha-beta actually buy, and how do you know it works?**
It prunes branches that provably cannot affect the result: once one option is better than the opponent would ever allow, the rest of the sibling options are irrelevant. It does not change the answer, only the work — so the demo asserts both, that the chosen move is identical with and without the table, and that the node count is lower than a full-width search. The size of the win depends on **move ordering**: alpha-beta is O(b^(d/2)) with perfect ordering against O(b^d) without. That is why `_ordered` tries the centre first — a good first move cuts more of the tree.

**Q6. Explain the transposition table, and the bug you avoided.**
The same position can be reached by different move orders, so the table caches a position's value keyed by the board and the side to move. The bug: with alpha-beta, a node that was searched with a window `(alpha, beta)` may return a value that is only a **bound** — if the true value is at least beta, the search stops early and returns something ≤ the truth. Storing that as if it were the exact value, and returning it later under a different window, gives a wrong answer, and the symptom is a bot that plays a losing move in a position it "remembers" having seen. The fix in the code is to store only when the returned value fell strictly inside the original window, which means the search was not cut off and the value is exact.

**Q7. How do you handle a draw?**
The status is DRAW when the board is full and no line was completed. The order matters more than the rule: **check the win first**, because on a full board the final move can complete a line, and testing `is_full()` first reports a draw in a game that was won. The demo builds that exact position — a full board where the last move completes the top row — and asserts `X_WINS`. Draws also matter for the bot: with perfect play on 3×3 the game is always a draw, which is why the minimax-versus-minimax test expects a draw and a full board rather than a winner.

**Q8. Should a bot's move be validated?**
Yes, and this is the boundary I would emphasise most. A bot is a component that can have a bug, and a search bug that returns an occupied cell must produce a rejected move rather than a board with two symbols on one square. So `Game._validate` is called for every move regardless of who produced it. The follow-up is what to do when it fails: not "ask again", because that is an infinite loop against a deterministic buggy bot; the game forfeits the bot and ends. That is a real operational decision disguised as an error path.

**Q9. How would you make it work for many games at once?**
One `Game` object per match, one lock per game. A single lock for the registry would mean a slow bot in one game blocking moves in every other game, and the registry lock is only needed for the map lookup that finds the game, not for the move itself. The interesting part is the bot: a search can take a second on a large board, so the pattern is to snapshot the board under the lock, search on the copy with no lock held, then re-acquire and **re-validate** — because the position may have changed while the search ran, and the move it produced may now be illegal.

**Q10 (senior). A 6×6 board with k=5 cannot be solved by search. What changes?**
Three things. The search needs a **depth limit**, and once it is depth-limited the terminal-only evaluation is not enough, so a heuristic takes over at the leaves — the code's `_evaluate`, which on 3×3 is never reached because the search always terminates. The second change is the resource model: a fixed depth is the wrong control because the branching factor varies with the position, so it becomes a **time budget** — search until the clock says stop, keep the best move found so far, and make sure the move returned is from a completed depth so it is not a partial result. The third is that the transposition table stops being an optimisation and becomes necessary, because the same positions recur constantly on a larger board. And the honest caveat: on a 6×6 board a depth-limited search plays a decent game and is not strong, and no amount of engineering inside this class design makes it strong — that needs a proper engine with pattern evaluation or a trained policy.

**Q11 (senior). Where would you expect a bug in this code?**
Four places. The **check order** in `apply` — win before full — is the classic and the demo targets it. The **status restoration in undo**: recomputing the status after popping the winning move gives ONGOING for a position that is genuinely won by the *previous* mover in some variants, which is why the `MoveRecord` carries `status_before`. The **transposition table's exactness condition**, which is invisible in tests that do not compare against an unpruned search — which is why the demo does exactly that comparison. And the **heuristic player's lookahead mutating the board**: `completes()` sets and clears a cell, and a missing clear would leave a phantom symbol that changes the next evaluation. The general shape is that all four produce a *plausible* game rather than a crash.

**Q12. Why not use a 2D array of strings and a list of eight winning triples?**
For a fixed 3×3 game with no follow-ups, that is the correct answer and I would say so — it is twenty lines and it is obviously right. The reason to generalise is that the follow-up "make it 4×4" arrives in essentially every interview, and the second follow-up — "now five in a row on a 6×6" — is where the fixed version stops entirely. The cost of generalising is genuinely small: a four-element direction table and a `run` counter. Where I would *not* generalise speculatively is the board geometry: a rectangular board changes the win rule's bounds handling and adds off-by-one risk, and I would only add it when asked.

**Q13. How does the observer pattern earn its place here?**
Because a game that prints is a game that cannot be tested and cannot be served. With `GameObserver`, the same `Game` drives a console renderer in a terminal, a `MemoryObserver` in a test that asserts the event stream, and a websocket push in a service — three consumers, one seam, and the game has no I/O dependency at all. The design cost is that notification order becomes a concern: observers must be notified after the state is fully consistent, and an observer that calls back into the game must be prevented or the aggregate re-enters mid-update.

**Q14. Why is a difficulty ladder hard here?**
Because 3×3 is solved: a perfect bot always draws and never loses, so "hard" and "impossible" are the same program and there is no room between them. A real ladder needs either a larger board where perfection is unattainable — so depth limits and time budgets create genuine differences — or deliberate imperfection in the strong bot: play the optimal move with probability p and a random legal move otherwise, where p is the difficulty knob. The second is what real games do, and it is worth naming because it is a design decision that inverts the obvious intuition: you make a bot *weaker* by adding randomness, not by searching less.

**Q15. What is the state-space size, and does it matter?**
The 3×3 game tree has 9! = 362,880 leaf paths and 5,478 distinct reachable positions. That is why a full solve is instant even without pruning, and why the transposition table's benefit is real but modest here — the same position recurs through different move orders, and the demo shows hits. It matters for two reasons: it tells you when the search is affordable at all, and it explains why alpha-beta's improvement is a constant factor here rather than the difference between feasible and infeasible. On a 6×6 board the same arithmetic gives you the opposite answer, which is the point of Q10.

---

### Pushback round

**P1. "You have written a full minimax and a heuristic player for a game that fits in twenty lines. That is over-engineering."**
For the game itself, the twenty-line version is right and I would write it if the requirement stopped at "play tic-tac-toe". The design is justified by the two questions that always follow — "make it 4×4" and "now add a bot" — and by the fact that the bot is where the interesting algorithm lives. But I would accept the criticism in one respect: the `HeuristicPlayer` is not needed by any requirement, it exists to show the difficulty tier and the shape most people write first. If I were writing this for production I would ship the `WinRule` strategy, the `Player` interface and the minimax, and add the heuristic only when a difficulty setting is actually requested.

**P2. "Your alpha-beta and your transposition table are both constant-factor optimisations on a board that solves instantly. Why include them?"**
On 3×3 that is exactly right — they are optimisations with no practical effect, and I say so in the trade-offs. They are there because the follow-up is a larger board, and on a larger board they stop being optimisations: a depth-limited search's quality depends on how deep it can reach in the time budget, and pruning is what buys depth. The transposition table also has a correctness dimension that is worth demonstrating precisely because it is invisible on a small board — the demo compares the table-enabled and table-disabled searches to prove they choose the same move, which is the check that catches a bound cached as a value.

**P3. "Generalising to n×n costs clarity. Eight `if` statements are readable; a direction table is not."**
The direction table is four lines of data and one loop, and I would argue it is *more* readable than eight `if` statements, because it states the rule once — *k* consecutive in any of four axes — instead of enumerating its consequences. What is genuinely lost is the ability to see the 3×3 win conditions at a glance, and that has a real cost during a whiteboard discussion. So the honest answer is that the generalisation's cost is small and its benefit is large, and if the interviewer wants the eight lines first I would write them first and then generalise when the follow-up arrives — which is a better demonstration of the design muscle than writing the general version unprompted.

**P4. "You have a `MisereWinRule` that returns `False` from `wins`. That is not misère."**
Correct, and it is the weakest code in the file — the class exists to demonstrate that the `WinRule` seam is real, and its implementation is a stub that makes the game un-winnable rather than misère. Real misère changes the *outcome* mapping: completing a line means the mover loses, which the current `Game.apply` cannot express because it maps "rule says won" to "the mover won". The right design is for the rule to return an outcome rather than a boolean — `evaluate(board, move, symbol) -> GameStatus` — so misère returns `for_winner(symbol.opponent)`. I would call that out rather than pretend the stub is complete, and it is a good example of a rule variant revealing that an interface is too narrow.

**P5. "Why does `Game.apply` return a `MoveRecord` instead of nothing?"**
Because undo needs it, and because the record is the memento the history stores. Returning it also makes `apply` useful to a caller that wants to log or acknowledge the move without re-deriving what happened. The alternative — `apply` returning nothing and `undo` reconstructing state by replaying — is O(moves) per undo and gets the status wrong after a win, because replaying without the winning move cannot tell you what the status was. The cost is one small immutable object per move, which is nothing next to the board it describes.

---

## 12. Cheat-sheet summary + cross-links

| Concept | One-line truth |
|---|---|
| The shape | `Board` (data) + `WinRule` (strategy) + `Player` (interface) + `Game` (aggregate) |
| Why the rule is separate | Misère, *k*-in-a-row and forbidden-move variants share a board and differ in the rule |
| The win check | **Incremental** — only the four axes through the last move, O(k) not O(n²) |
| The signature clue | `wins(board, move, symbol)` takes the move *because* deriving it is the expensive part |
| The classic bug | Check the **win before `is_full()`** — the last move can win on a full board |
| Draw | Board full and no line. Full-board-with-a-win is a win, not a draw |
| The search | Make/unmake on the real board — **no allocation per node** |
| Alpha-beta | Prunes provably irrelevant branches; strength depends on **move ordering** |
| The table | Cache **only exact values**; a cached cutoff bound is the classic bug |
| Complexity | 3×3: 5,478 positions, 9! leaf paths — instantly solved. Larger needs a depth limit and a heuristic |
| Difficulty ladder | 3×3 is solved, so "hard" == "impossible"; real tiers need a bigger board or deliberate randomness |
| Every move is validated | **Including a bot's.** A buggy search must not corrupt the board |
| A bad bot's move | Forfeit the bot. Never re-ask — that is an infinite loop |
| Resign | `None` from `choose_move`; not an exception |
| Undo | Memento: store `status_before` and `turn_before`; recomputation fails after a win |
| Redo | A new move **clears** the redo stack |
| Concurrency | One lock per game; search on a **copy**, apply under the lock, **re-validate** |
| Serialisation | Store the move list and **replay**; storing cells loses the history |
| Patterns | Strategy ×2, Observer, Memento, Value Object, Registry |
| Rejected | Template Method for players (no shared skeleton), Singleton |

**The three sentences to memorise**

1. *"The win rule is a strategy generalised to k-in-a-row, so 'make it 4×4' is a constructor argument — and it checks only the four axes through the last move."*
2. *"Check the win before the draw, because the last move can win on a full board."*
3. *"Every move is validated by the game, including a bot's — and a bot that cannot move legally forfeits rather than looping."*

### Cross-links

- **The LLD answer framework this file follows, and the HLD/LLD boundary:** [LLD/01 What is LLD and How to Answer](../01-what-is-lld-and-how-to-answer.md)
- **The Strategy and Observer patterns in the catalogue:** [LLD/03 Design Patterns](../03-design-patterns.md)
- **The same injected-clock seam as §7.5:** [Rate Limiter](11-rate-limiter.md) §5 (`Clock`), §7.5
- **Rejecting Template Method for classes that share no skeleton, the same reasoning:** [Rate Limiter](11-rate-limiter.md) §9 · [Logger Framework](12-logger-framework.md) §9
- **Another aggregate with per-object locking and a status machine:** [Vending Machine](10-vending-machine.md) §6 (`VendingMachineState`) · [Chess Game](09-chess-game.md) §6
- **Memento and undo in a board game, at a much larger scale:** [Chess Game](09-chess-game.md) §5 (`HistoryEntry`) and §7.3
- **Search over a game tree with alpha-beta, extended:** [Chess Game](09-chess-game.md) §7.1
- **Cheat sheet, pattern tables and the classic-problems list:** [LLD Cheatsheet](../../Cheat-Sheets/07-lld-cheatsheet.md)
- **Question bank:** [LLD Questions](../../Interview-Questions/08-lld-questions.md)
