# LLD 09 — Chess Game

> **Grounding note.** The reference repositories for this knowledge base contain **no OOP/LLD material** and the source transcripts name the classic LLD problems (parking lot, LRU, chess, elevator) **without teaching them in depth**. This file is therefore built from first principles plus the HLD/LLD distinction set out in [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md). It is not transcript-grounded, and it does not claim to be.

> **TL;DR** — A chess game is the canonical **"rule engine behind a state machine"** LLD question. The two ideas that carry the design are: (1) **polymorphism over a `Piece` hierarchy** — every piece answers "what moves can I pseudo-legally make from here?", and a single `RuleEngine` layer on top filters those through "does this leave my own king in check?"; (2) **special moves are the whole difficulty** — castling, en passant, promotion, and the threefold/fifty-move draws are where candidates lose points, because each one couples the move to *board state that is not in the grid* (castling rights, en-passant target, halfmove clock). If you model those four in the board state and make moves a **Command object** so they can be undone, the rest is bookkeeping.

> **60-second interview answer:** "I'd clarify scope first: two-player standard chess, full FIDE rules — castling, en passant, promotion, check, checkmate, stalemate, and the draw conditions — or a simplified variant? Assuming full rules, the entities are `Board` (the 8×8 grid plus castling rights, en-passant target and the halfmove clock), a `Piece` hierarchy for the six piece types, a `Move` value object that doubles as the Command for undo, a `RuleEngine` that generates legal moves and detects check, and a `Game` that owns turn order, status and history. The key design decision is the split: pieces generate **pseudo-legal** moves with no knowledge of check, and the rule engine filters them by making the move on a copy of the board and testing whether the mover's king is attacked. That keeps each piece class tiny and puts every rule that needs the whole board in one place. The hard parts are castling rights, en passant and repetition, all of which are state the board must carry explicitly."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **Two players, or an AI opponent too?** | Whether you need a `PlayerStrategy` abstraction | Two human players; AI is a stated extension (§7) |
| 2 | **Full FIDE rules, or "pieces move and you capture the king"?** | Whether you implement castling, en passant, promotion, draws | **Full rules** — but say "I'll implement the core first, then add castling, en passant, promotion" |
| 3 | **Do we need check / checkmate / stalemate detection?** | A legality filter (king in check) versus naive pseudo-legal moves | Yes — a game that lets you move into check is wrong |
| 4 | **Undo?** | Whether `Move` must be a Command carrying enough state to reverse | Yes — it is one of the most common follow-ups |
| 5 | **Draw rules: fifty-move, threefold repetition, insufficient material?** | Whether the board carries a halfmove clock and a position history | Implement fifty-move and insufficient material; threefold via a position hash |
| 6 | **Timers / clocks?** | A separate `Clock` entity, and whether the game is time-aware | No clocks unless asked; clocks are a thin addition |
| 7 | **Move notation output (SAN)?** | A `NotationFormatter` collaborator | "Out of scope, but it's a formatting concern bolted onto `Move`" |
| 8 | **Variants: Chess960, three-check, bughouse?** | Whether board geometry is hard-coded | Standard chess; note that Chess960 only changes the *start position and castling*, so the design absorbs it (§7) |
| 9 | **Single game per process, or a server running many games?** | The concurrency model | Many games; one actor/thread per game, sharded by `game_id` |
| 10 | **Persistence / replay?** | Whether positions must be serialisable (FEN/PDN) | In-memory; note FEN round-tripping as the serialisation format |

### The defaults to assume, stated plainly

> "I'll assume **two human players, full standard rules, one game at a time in the object model, with undo and the three draw conditions**. I'll model the board so that persistence and Chess960 are additive, and I'll treat the AI opponent and the network layer as pluggable strategies rather than redesigning the core. Is that the right shape?"

### Non-goals (say them, so the interviewer stops probing)

- No UI rendering beyond an ASCII dump (presentation is not LLD).
- No legal-move *search* (minimax/alpha-beta) unless the AI follow-up is asked — that is an AI concern, not a class-design concern.
- No distributed game state — that is HLD ([HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md)).

### The five requirements that actually constrain the design

1. **Legality is board-wide.** A move is legal only if the mover's king is not in check afterwards. So pieces cannot decide legality alone.
2. **Three special moves depend on hidden state.** Castling rights, the en-passant target square, and the halfmove clock are not derivable from the grid.
3. **The same position can recur.** Repetition detection needs a position key, not a move list, if it is to be cheap.
4. **Moves must be reversible.** Undo is a follow-up in almost every chess interview.
5. **Status is a function of the position.** `checkmate`, `stalemate` and the draws are computed, never stored as a flag the caller sets.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `Color` | Enum | Which side; `opponent` | Anything else |
| `PieceType` | Enum | K, Q, R, B, N, P | Movement rules |
| `Square` | Immutable value | A (file, rank) coordinate; algebraic parsing; offsets | Board knowledge |
| `Move` | Immutable value + Command | Origin, target, promotion, castle flag, en-passant flag; `uci()` | Applying itself to a board |
| `Piece` (abstract) | Abstract class | `pseudo_legal_moves(board, origin)` — **no check awareness** | Legality, board mutation |
| `King`, `Queen`, `Rook`, `Bishop`, `Knight`, `Pawn` | Concrete classes | Their own movement via shared `OffsetPiece` template | Special rules (castling, promotion target choice) |
| `Board` | Mutable state | The 8×8 grid **plus** castling rights, en-passant target, halfmove clock, side to move; `apply(move)`, `copy()` | Deciding legality |
| `RuleEngine` | Stateless service | Legal-move generation, check detection, attack detection, castling generation, status computation | Owning history |
| `GameStatus` | Enum | The terminal/transient states | — |
| `Game` | Aggregate root | Turn order, history for undo, repetition keys, listeners, status | Moving pieces itself — it delegates |
| `MoveListener` / observer fn | Interface | Be notified on every applied move (UI, notation, telemetry) | Changing the game |
| `PlayerStrategy` | Interface (extension) | Choose a move given a position — human or AI | Rule knowledge |
| `NotationFormatter` | Interface (extension) | Render a `Move` as SAN/UCI | — |

**The one-line responsibility split to say out loud:** *"Pieces know geometry; the rule engine knows legality; the board knows state; the game knows turn order and history."*

---

## 3. Class diagram (ASCII)

```
                        ┌──────────────────────┐
                        │      «enum»          │
                        │       Color          │
                        │  WHITE  BLACK        │
                        │  + opponent          │
                        └──────────┬───────────┘
                                   │ has
        ┌──────────────────────────┴───────────────────────────┐
        │                                                      │
┌───────▼────────┐                                   ┌─────────▼─────────┐
│   «abstract»   │                                   │      Board        │
│     Piece      │                                   │───────────────────│
│────────────────│                                   │ grid: 8×8 Piece?  │
│ color: Color   │                                   │ castling: set     │
│────────────────│                                   │ ep_target: Square?│
│ + symbol       │                                   │ halfmove: int     │
│ +pseudo_legal_ │                                   │ side_to_move:Color│
│    moves(b,s)  │                                   │───────────────────│
└───────▲────────┘                                   │ + piece_at(sq)    │
        │ extends                                    │ + set_piece(sq,p) │
        │                                            │ + apply(move)     │
┌───────┴───────────┐                                │ + copy()          │
│   OffsetPiece     │                                │ + fen()           │
│  (Template Method)│                                └─────────▲─────────┘
│ DIRECTIONS,SLIDING│                                          │ owns
└───────▲───────────┘                                          │
        │                                                      │
  ┌─────┴────┬────────┬─────────┬─────────┬─────────┐           │
┌─┴──┐  ┌────┴─┐ ┌────┴──┐ ┌────┴───┐ ┌───┴───┐ ┌───┴───┐      │
│King│  │Queen │ │ Rook  │ │ Bishop │ │Knight │ │ Pawn  │      │
└────┘  └──────┘ └───────┘ └────────┘ └───────┘ └───────┘      │
   ▲                                                           │
   │ castling is generated by the RuleEngine, not by King       │
   │                                                           │
┌──┴─────────────────┐        ┌───────────────────┐            │
│    RuleEngine      │───────▶│       Move        │            │
│────────────────────│ uses   │───────────────────│            │
│ +legal_moves(b,c)  │        │ origin: Square    │            │
│ +is_in_check(b,c)  │        │ target: Square    │            │
│ +is_attacked(...)  │        │ promotion:Type?   │            │
│ +status(b)         │        │ castle: Side?     │            │
└────────────────────┘        │ en_passant: bool  │            │
                              │ + uci()           │            │
                              └───────────────────┘            │
                                                               │
┌──────────────────────────────────────────────┐               │
│                    Game                      │               │
│──────────────────────────────────────────────│               │
│ board: Board                                 │───────────────┘
│ rules: RuleEngine                            │
│ history: list[HistoryEntry]                  │
│ position_keys: list[str]   (repetition)      │
│ listeners: list[Callable]                    │
│──────────────────────────────────────────────│
│ + legal_moves()                              │
│ + play(move)   ── validate → snapshot → apply│
│ + undo()                                     │
│ + resign(color)                              │
│ + status() -> GameStatus                     │
│ + ascii() -> str                             │
└───────────────┬──────────────────────────────┘
                │ notifies
        ┌───────▼────────┐        ┌──────────────────┐
        │ «interface»    │        │ «interface»      │
        │ MoveListener   │        │ PlayerStrategy   │  (extension)
        │ + on_move(...) │        │ + choose(g) Move │
        └────────────────┘        └──────────────────┘
```

**Read the diagram this way:** the `Piece` hierarchy is a **Template Method** hierarchy (`OffsetPiece` fixes the iteration, subclasses supply `DIRECTIONS` and `SLIDING`). `Move` is a **Command**. `GameStatus` is the **State** of the game. `PlayerStrategy` is a **Strategy** for the extension point.

---

## 4. Interfaces / abstract classes / enums

```python
class Color(Enum):
    WHITE = "w"
    BLACK = "b"

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.WHITE else Color.WHITE


class PieceType(Enum):
    KING = "K"
    QUEEN = "Q"
    ROOK = "R"
    BISHOP = "B"
    KNIGHT = "N"
    PAWN = "P"


class CastleSide(Enum):
    KING_SIDE = "O-O"
    QUEEN_SIDE = "O-O-O"


class GameStatus(Enum):
    ONGOING = "ongoing"
    CHECK = "check"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    DRAW_FIFTY_MOVE = "draw_fifty_move"
    DRAW_INSUFFICIENT_MATERIAL = "draw_insufficient_material"
    DRAW_THREEFOLD_REPETITION = "draw_threefold_repetition"
    RESIGNED = "resigned"


class IllegalMoveError(ValueError):
    """Raised when a move is not in the legal-move set for the position."""


class Piece(ABC):
    """A chess piece.

    The contract is deliberately narrow: a piece answers *pseudo-legal* moves
    only. It must never consult check, castling rights or the en-passant
    target, because those require whole-board knowledge.
    """

    piece_type: ClassVar[PieceType]

    def __init__(self, color: Color) -> None:
        self.color = color

    @property
    def symbol(self) -> str:
        return self.piece_type.value if self.color is Color.WHITE else self.piece_type.value.lower()

    @abstractmethod
    def pseudo_legal_moves(self, board: "Board", origin: "Square") -> list["Move"]:
        ...


class MoveListener(Protocol):
    """Observer: notified after every successfully applied move."""

    def on_move(self, game: "Game", move: "Move", status: GameStatus) -> None:
        ...


class PlayerStrategy(Protocol):
    """Extension point: a human prompt, an AI search, or a scripted replay."""

    def choose_move(self, game: "Game") -> "Move":
        ...


class NotationFormatter(Protocol):
    def format(self, game_before: "Game", move: "Move", game_after: "Game") -> str:
        ...
```

**Why an abstract `Piece` and not a `if piece_type == ...` switch:** adding a piece type (a fairy-chess piece, or a "wall" in a variant) must not require editing a 200-line `if/elif`. The abstract method makes the compiler/`ABC` enforce that every new piece supplies its movement — which is the Open/Closed Principle applied where it pays.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** Interview languages vary — Java uses `abstract class Piece` with `@Override`, C++ uses `virtual` plus a `unique_ptr` board, Go uses an interface plus a struct and no inheritance. The *shape* here (Template Method over a `Piece` hierarchy, Command for `Move`, State for status) transfers unchanged; only the syntax differs. Python is used because it is executable as written.

```python
"""chess_game.py — a complete, runnable two-player chess engine.

Design summary
--------------
Piece          : geometry only (pseudo-legal moves).
OffsetPiece    : Template Method — supplies DIRECTIONS + SLIDING.
Board          : grid + castling rights + en-passant target + clocks.
RuleEngine     : legality (check filter), attack detection, status.
Move           : immutable Command value.
Game           : turn order, history, undo, listeners, status.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, ClassVar, Iterator, Optional, Protocol


# --------------------------------------------------------------------------
# 1. Value types
# --------------------------------------------------------------------------

class Color(Enum):
    WHITE = "w"
    BLACK = "b"

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.WHITE else Color.WHITE


class PieceType(Enum):
    KING = "K"
    QUEEN = "Q"
    ROOK = "R"
    BISHOP = "B"
    KNIGHT = "N"
    PAWN = "P"


class CastleSide(Enum):
    KING_SIDE = "O-O"
    QUEEN_SIDE = "O-O-O"


class GameStatus(Enum):
    ONGOING = "ongoing"
    CHECK = "check"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    DRAW_FIFTY_MOVE = "draw_fifty_move"
    DRAW_INSUFFICIENT_MATERIAL = "draw_insufficient_material"
    DRAW_THREEFOLD_REPETITION = "draw_threefold_repetition"
    RESIGNED = "resigned"


class IllegalMoveError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Square:
    file: int  # 0..7 -> a..h
    rank: int  # 0..7 -> 1..8

    def is_on_board(self) -> bool:
        return 0 <= self.file < 8 and 0 <= self.rank < 8

    def offset(self, df: int, dr: int) -> "Square":
        return Square(self.file + df, self.rank + dr)

    @classmethod
    def parse(cls, text: str) -> "Square":
        if len(text) != 2 or text[0] not in "abcdefgh" or text[1] not in "12345678":
            raise ValueError(f"not a square: {text!r}")
        return cls(ord(text[0]) - ord("a"), int(text[1]) - 1)

    def __str__(self) -> str:
        return f"{chr(ord('a') + self.file)}{self.rank + 1}"


@dataclass(frozen=True)
class Move:
    """A Command: everything needed to describe (and later mirror) a move."""

    origin: Square
    target: Square
    promotion: Optional[PieceType] = None
    castle: Optional[CastleSide] = None
    en_passant: bool = False

    def uci(self) -> str:
        text = f"{self.origin}{self.target}"
        if self.promotion is not None:
            text += self.promotion.value.lower()
        return text

    @classmethod
    def parse_uci(cls, text: str) -> "Move":
        if len(text) not in (4, 5):
            raise ValueError(f"not a UCI move: {text!r}")
        promotion = None
        if len(text) == 5:
            promotion = {"q": PieceType.QUEEN, "r": PieceType.ROOK,
                         "b": PieceType.BISHOP, "n": PieceType.KNIGHT}[text[4]]
        return cls(Square.parse(text[0:2]), Square.parse(text[2:4]), promotion)


# --------------------------------------------------------------------------
# 2. Piece hierarchy — Template Method over movement geometry
# --------------------------------------------------------------------------

class Piece(ABC):
    piece_type: ClassVar[PieceType]

    def __init__(self, color: Color) -> None:
        self.color = color

    @property
    def symbol(self) -> str:
        return self.piece_type.value if self.color is Color.WHITE else self.piece_type.value.lower()

    @abstractmethod
    def pseudo_legal_moves(self, board: "Board", origin: Square) -> list[Move]:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.color.value}>"


class OffsetPiece(Piece):
    """Template Method: iterate DIRECTIONS, optionally sliding until blocked."""

    DIRECTIONS: ClassVar[tuple[tuple[int, int], ...]] = ()
    SLIDING: ClassVar[bool] = False

    def pseudo_legal_moves(self, board: "Board", origin: Square) -> list[Move]:
        moves: list[Move] = []
        for df, dr in self.DIRECTIONS:
            square = origin.offset(df, dr)
            while square.is_on_board():
                occupant = board.piece_at(square)
                if occupant is None:
                    moves.append(Move(origin, square))
                else:
                    if occupant.color is not self.color:
                        moves.append(Move(origin, square))
                    break
                if not self.SLIDING:
                    break
                square = square.offset(df, dr)
        return moves


ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KING_DIRS = ROOK_DIRS + BISHOP_DIRS
KNIGHT_DIRS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))


class King(OffsetPiece):
    piece_type = PieceType.KING
    DIRECTIONS = KING_DIRS
    SLIDING = False
    # Castling is NOT generated here: it needs rights and attack tests.


class Queen(OffsetPiece):
    piece_type = PieceType.QUEEN
    DIRECTIONS = KING_DIRS
    SLIDING = True


class Rook(OffsetPiece):
    piece_type = PieceType.ROOK
    DIRECTIONS = ROOK_DIRS
    SLIDING = True


class Bishop(OffsetPiece):
    piece_type = PieceType.BISHOP
    DIRECTIONS = BISHOP_DIRS
    SLIDING = True


class Knight(OffsetPiece):
    piece_type = PieceType.KNIGHT
    DIRECTIONS = KNIGHT_DIRS
    SLIDING = False


class Pawn(Piece):
    """The only piece that cannot use the offset template: direction,
    double-push, diagonal capture and en passant are all irregular."""

    piece_type = PieceType.PAWN

    def pseudo_legal_moves(self, board: "Board", origin: Square) -> list[Move]:
        moves: list[Move] = []
        forward = 1 if self.color is Color.WHITE else -1
        start_rank = 1 if self.color is Color.WHITE else 6
        last_rank = 7 if self.color is Color.WHITE else 0

        # Single push (and promotion expansion).
        one = origin.offset(0, forward)
        if one.is_on_board() and board.piece_at(one) is None:
            if one.rank == last_rank:
                for promo in (PieceType.QUEEN, PieceType.ROOK,
                              PieceType.BISHOP, PieceType.KNIGHT):
                    moves.append(Move(origin, one, promotion=promo))
            else:
                moves.append(Move(origin, one))

            # Double push, only from the start rank and only if both squares are empty.
            two = origin.offset(0, 2 * forward)
            if origin.rank == start_rank and board.piece_at(two) is None:
                moves.append(Move(origin, two))

        # Diagonal captures, including en passant.
        for df in (-1, 1):
            diag = origin.offset(df, forward)
            if not diag.is_on_board():
                continue
            occupant = board.piece_at(diag)
            if occupant is not None and occupant.color is not self.color:
                if diag.rank == last_rank:
                    for promo in (PieceType.QUEEN, PieceType.ROOK,
                                  PieceType.BISHOP, PieceType.KNIGHT):
                        moves.append(Move(origin, diag, promotion=promo))
                else:
                    moves.append(Move(origin, diag))
            elif occupant is None and board.ep_target == diag:
                moves.append(Move(origin, diag, en_passant=True))
        return moves


PIECE_CLASSES: dict[PieceType, type[Piece]] = {
    PieceType.KING: King, PieceType.QUEEN: Queen, PieceType.ROOK: Rook,
    PieceType.BISHOP: Bishop, PieceType.KNIGHT: Knight, PieceType.PAWN: Pawn,
}


def make_piece(piece_type: PieceType, color: Color) -> Piece:
    """Factory: the only place that maps an enum to a class."""
    return PIECE_CLASSES[piece_type](color)


# --------------------------------------------------------------------------
# 3. Board — the grid plus the state that is not in the grid
# --------------------------------------------------------------------------

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class Board:
    """Mutable position state.

    The three things candidates forget live here:
      * castling rights (a set of (color, side) that survives the grid)
      * the en-passant target square (only valid for exactly one ply)
      * the halfmove clock (drives the fifty-move draw)
    """

    def __init__(
        self,
        grid: list[list[Optional[Piece]]],
        side_to_move: Color,
        castling: frozenset[tuple[Color, CastleSide]],
        ep_target: Optional[Square],
        halfmove: int,
        fullmove: int,
    ) -> None:
        self.grid = grid
        self.side_to_move = side_to_move
        self.castling = castling
        self.ep_target = ep_target
        self.halfmove = halfmove
        self.fullmove = fullmove

    # ---- construction -----------------------------------------------------

    @classmethod
    def initial(cls) -> "Board":
        return cls.from_fen(START_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        placement, side, rights, ep, halfmove, fullmove = fen.split()
        grid: list[list[Optional[Piece]]] = [[None] * 8 for _ in range(8)]
        ranks = placement.split("/")
        if len(ranks) != 8:
            raise ValueError(f"bad FEN placement: {placement!r}")
        for row, rank_text in enumerate(ranks):
            file_index = 0
            for ch in rank_text:
                if ch.isdigit():
                    file_index += int(ch)
                    continue
                color = Color.WHITE if ch.isupper() else Color.BLACK
                piece_type = PieceType(ch.upper())
                # FEN lists rank 8 first; our rank 0 is rank 1.
                grid[7 - row][file_index] = make_piece(piece_type, color)
                file_index += 1
            if file_index != 8:
                raise ValueError(f"bad FEN rank: {rank_text!r}")

        castling: set[tuple[Color, CastleSide]] = set()
        if rights != "-":
            for ch in rights:
                castle_color = Color.WHITE if ch.isupper() else Color.BLACK
                castle_side = (CastleSide.KING_SIDE if ch.upper() == "K"
                               else CastleSide.QUEEN_SIDE)
                castling.add((castle_color, castle_side))

        # NOTE: `side` is the FEN side-to-move field. Do not let the castling
        # loop reuse that name — shadowing it silently flips the side to move.
        return cls(
            grid=grid,
            side_to_move=Color.WHITE if side == "w" else Color.BLACK,
            castling=frozenset(castling),
            ep_target=None if ep == "-" else Square.parse(ep),
            halfmove=int(halfmove),
            fullmove=int(fullmove),
        )

    # ---- accessors --------------------------------------------------------

    def piece_at(self, square: Square) -> Optional[Piece]:
        if not square.is_on_board():
            return None
        return self.grid[square.rank][square.file]

    def set_piece(self, square: Square, piece: Optional[Piece]) -> None:
        self.grid[square.rank][square.file] = piece

    def iter_pieces(self, color: Color) -> Iterator[tuple[Square, Piece]]:
        for rank in range(8):
            for file in range(8):
                piece = self.grid[rank][file]
                if piece is not None and piece.color is color:
                    yield Square(file, rank), piece

    def copy(self) -> "Board":
        # Pieces are immutable (colour only), so a shallow row copy is safe.
        return Board(
            grid=[row[:] for row in self.grid],
            side_to_move=self.side_to_move,
            castling=self.castling,
            ep_target=self.ep_target,
            halfmove=self.halfmove,
            fullmove=self.fullmove,
        )

    def position_key(self) -> str:
        """Repetition key: piece placement, side, rights and ep target.
        Move counters are excluded — they are not part of the position."""
        rows = []
        for rank in range(7, -1, -1):
            empty = 0
            text = ""
            for file in range(8):
                piece = self.grid[rank][file]
                if piece is None:
                    empty += 1
                    continue
                if empty:
                    text += str(empty)
                    empty = 0
                text += piece.symbol
            if empty:
                text += str(empty)
            rows.append(text)
        rights = "".join(
            sorted(
                ("K" if side is CastleSide.KING_SIDE else "Q")
                if color is Color.WHITE
                else ("k" if side is CastleSide.KING_SIDE else "q")
                for color, side in self.castling
            )
        ) or "-"
        return f"{'/'.join(rows)} {self.side_to_move.value} {rights} {self.ep_target or '-'}"

    def fen(self) -> str:
        rights = "".join(
            sorted(
                ("K" if side is CastleSide.KING_SIDE else "Q")
                if color is Color.WHITE
                else ("k" if side is CastleSide.KING_SIDE else "q")
                for color, side in self.castling
            )
        ) or "-"
        return (f"{self.position_key().split(' ')[0]} {self.side_to_move.value} "
                f"{rights} {self.ep_target or '-'} {self.halfmove} {self.fullmove}")

    # ---- mutation ---------------------------------------------------------

    def apply(self, move: Move) -> None:
        """Apply a move with NO legality check. RuleEngine gates this."""
        mover = self.piece_at(move.origin)
        if mover is None:
            raise IllegalMoveError(f"no piece on {move.origin}")

        # 1. Identify the captured piece. En passant captures a pawn that sits
        #    on the ORIGIN rank, not on the target square.
        captured_square = move.target
        if move.en_passant:
            captured_square = Square(move.target.file, move.origin.rank)
        captured = self.piece_at(captured_square)

        # 2. Remove the captured piece BEFORE placing the mover. On an ordinary
        #    capture the captured square IS the target square, so removing it
        #    afterwards would delete the piece we just moved.
        if captured is not None:
            self.set_piece(captured_square, None)

        # 3. Move the piece, expanding promotions.
        self.set_piece(move.origin, None)
        if move.promotion is not None:
            self.set_piece(move.target, make_piece(move.promotion, mover.color))
        else:
            self.set_piece(move.target, mover)

        # 4. Castling also moves a rook that the Move does not name.
        if move.castle is not None:
            rank = move.origin.rank
            if move.castle is CastleSide.KING_SIDE:
                rook_from, rook_to = Square(7, rank), Square(5, rank)
            else:
                rook_from, rook_to = Square(0, rank), Square(3, rank)
            rook = self.piece_at(rook_from)
            self.set_piece(rook_from, None)
            self.set_piece(rook_to, rook)

        # 5. Castling rights: king moves, rook moves, rook captured.
        rights = set(self.castling)
        if isinstance(mover, King):
            rights = {r for r in rights if r[0] is not mover.color}
        if isinstance(mover, Rook):
            home = self._home_side(move.origin, mover.color)
            if home is not None:
                rights.discard((mover.color, home))
        if captured is not None and isinstance(captured, Rook):
            home = self._home_side(captured_square, captured.color)
            if home is not None:
                rights.discard((captured.color, home))
        self.castling = frozenset(rights)

        # 6. En-passant target: set only after a two-square pawn push.
        if isinstance(mover, Pawn) and abs(move.target.rank - move.origin.rank) == 2:
            self.ep_target = Square(move.origin.file, (move.origin.rank + move.target.rank) // 2)
        else:
            self.ep_target = None

        # 7. Clocks.
        if isinstance(mover, Pawn) or captured is not None:
            self.halfmove = 0
        else:
            self.halfmove += 1
        if self.side_to_move is Color.BLACK:
            self.fullmove += 1
        self.side_to_move = self.side_to_move.opponent

    @staticmethod
    def _home_side(square: Square, color: Color) -> Optional[CastleSide]:
        home_rank = 0 if color is Color.WHITE else 7
        if square.rank != home_rank:
            return None
        if square.file == 7:
            return CastleSide.KING_SIDE
        if square.file == 0:
            return CastleSide.QUEEN_SIDE
        return None


# --------------------------------------------------------------------------
# 4. RuleEngine — everything that needs whole-board knowledge
# --------------------------------------------------------------------------

class RuleEngine:
    """Stateless. Legality, attacks, castling, status."""

    # ---- attack detection -------------------------------------------------

    @staticmethod
    def find_king(board: Board, color: Color) -> Optional[Square]:
        for square, piece in board.iter_pieces(color):
            if isinstance(piece, King):
                return square
        return None

    def is_attacked(self, board: Board, square: Square, by_color: Color) -> bool:
        # Pawns: a white pawn attacking `square` sits one rank below it.
        pawn_delta = -1 if by_color is Color.WHITE else 1
        for df in (-1, 1):
            candidate = square.offset(df, pawn_delta)
            piece = board.piece_at(candidate)
            if isinstance(piece, Pawn) and piece.color is by_color:
                return True

        for df, dr in KNIGHT_DIRS:
            piece = board.piece_at(square.offset(df, dr))
            if isinstance(piece, Knight) and piece.color is by_color:
                return True

        for df, dr in KING_DIRS:
            piece = board.piece_at(square.offset(df, dr))
            if isinstance(piece, King) and piece.color is by_color:
                return True

        if self._sliding_attack(board, square, ROOK_DIRS, by_color, (Rook, Queen)):
            return True
        if self._sliding_attack(board, square, BISHOP_DIRS, by_color, (Bishop, Queen)):
            return True
        return False

    @staticmethod
    def _sliding_attack(
        board: Board,
        square: Square,
        directions: tuple[tuple[int, int], ...],
        by_color: Color,
        attackers: tuple[type, ...],
    ) -> bool:
        for df, dr in directions:
            cursor = square.offset(df, dr)
            while cursor.is_on_board():
                piece = board.piece_at(cursor)
                if piece is not None:
                    if piece.color is by_color and isinstance(piece, attackers):
                        return True
                    break
                cursor = cursor.offset(df, dr)
        return False

    def is_in_check(self, board: Board, color: Color) -> bool:
        king = self.find_king(board, color)
        if king is None:
            return False
        return self.is_attacked(board, king, color.opponent)

    # ---- legal move generation -------------------------------------------

    def legal_moves(self, board: Board, color: Color) -> list[Move]:
        moves: list[Move] = []
        for square, piece in board.iter_pieces(color):
            for move in piece.pseudo_legal_moves(board, square):
                if self._leaves_king_safe(board, move, color):
                    moves.append(move)
        moves.extend(self._castling_moves(board, color))
        return moves

    def is_legal(self, board: Board, move: Move) -> bool:
        piece = board.piece_at(move.origin)
        if piece is None or piece.color is not board.side_to_move:
            return False
        if move.castle is not None:
            return move in self._castling_moves(board, piece.color)
        if move not in piece.pseudo_legal_moves(board, move.origin):
            return False
        return self._leaves_king_safe(board, move, piece.color)

    def _leaves_king_safe(self, board: Board, move: Move, color: Color) -> bool:
        trial = board.copy()
        trial.apply(move)
        return not self.is_in_check(trial, color)

    def _castling_moves(self, board: Board, color: Color) -> list[Move]:
        moves: list[Move] = []
        home_rank = 0 if color is Color.WHITE else 7
        king_home = Square(4, home_rank)
        king = board.piece_at(king_home)
        if not isinstance(king, King) or king.color is not color:
            return moves
        if self.is_in_check(board, color):
            return moves  # cannot castle out of check

        for side, (empty_files, transit, rook_file) in {
            CastleSide.KING_SIDE: ((5, 6), (5, 6), 7),
            CastleSide.QUEEN_SIDE: ((1, 2, 3), (3,), 0),
        }.items():
            if (color, side) not in board.castling:
                continue
            rook = board.piece_at(Square(rook_file, home_rank))
            if not isinstance(rook, Rook) or rook.color is not color:
                continue
            if any(board.piece_at(Square(f, home_rank)) is not None for f in empty_files):
                continue
            # The king may not pass through an attacked square.
            if any(
                self.is_attacked(board, Square(f, home_rank), color.opponent)
                for f in transit
            ):
                continue
            target_file = 6 if side is CastleSide.KING_SIDE else 2
            moves.append(Move(king_home, Square(target_file, home_rank), castle=side))
        return moves

    # ---- status -----------------------------------------------------------

    def has_insufficient_material(self, board: Board) -> bool:
        """K vs K, K+B vs K, K+N vs K, and K+B vs K+B on same-colour squares."""
        pieces = [(sq, p) for sq, p in board.iter_pieces(Color.WHITE)]
        pieces += [(sq, p) for sq, p in board.iter_pieces(Color.BLACK)]
        if any(isinstance(p, (Pawn, Rook, Queen)) for _, p in pieces):
            return False
        minors = [(sq, p) for sq, p in pieces if not isinstance(p, King)]
        if len(minors) <= 1:
            return True
        if len(minors) == 2:
            (sq_a, a), (sq_b, b) = minors
            if type(a) is not type(b) or a.color is b.color:
                return False
            if isinstance(a, Bishop):
                square_color = (sq_a.file + sq_a.rank) % 2
                return square_color == (sq_b.file + sq_b.rank) % 2
            return False
        return False

    def status(self, board: Board, position_keys: list[str]) -> GameStatus:
        color = board.side_to_move
        in_check = self.is_in_check(board, color)
        if not self.legal_moves(board, color):
            return GameStatus.CHECKMATE if in_check else GameStatus.STALEMATE
        if position_keys.count(board.position_key()) >= 3:
            return GameStatus.DRAW_THREEFOLD_REPETITION
        if board.halfmove >= 100:
            return GameStatus.DRAW_FIFTY_MOVE
        if self.has_insufficient_material(board):
            return GameStatus.DRAW_INSUFFICIENT_MATERIAL
        return GameStatus.CHECK if in_check else GameStatus.ONGOING


# --------------------------------------------------------------------------
# 5. Game — the aggregate root
# --------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    move: Move
    board_before: Board
    status_after: GameStatus


class Game:
    def __init__(self, board: Optional[Board] = None, rules: Optional[RuleEngine] = None) -> None:
        self.board = board if board is not None else Board.initial()
        self.rules = rules if rules is not None else RuleEngine()
        self.history: list[HistoryEntry] = []
        self.position_keys: list[str] = [self.board.position_key()]
        self._listeners: list[Callable[[Game, Move, GameStatus], None]] = []
        self._resigned: Optional[Color] = None

    # ---- observation ------------------------------------------------------

    def legal_moves(self) -> list[Move]:
        return self.rules.legal_moves(self.board, self.board.side_to_move)

    def status(self) -> GameStatus:
        if self._resigned is not None:
            return GameStatus.RESIGNED
        return self.rules.status(self.board, self.position_keys)

    def is_over(self) -> bool:
        return self.status() not in (GameStatus.ONGOING, GameStatus.CHECK)

    def add_listener(self, listener: Callable[[Game, Move, GameStatus], None]) -> None:
        self._listeners.append(listener)

    # ---- mutation ---------------------------------------------------------

    def play(self, move: Move) -> GameStatus:
        if self.is_over():
            raise IllegalMoveError(f"game is already over: {self.status().value}")
        if not self.rules.is_legal(self.board, move):
            raise IllegalMoveError(f"illegal move {move.uci()} in {self.board.position_key()}")
        before = self.board.copy()
        self.board.apply(move)
        self.position_keys.append(self.board.position_key())
        status = self.status()
        self.history.append(HistoryEntry(move, before, status))
        for listener in self._listeners:
            listener(self, move, status)
        return status

    def play_uci(self, text: str) -> GameStatus:
        """Convenience: accept a UCI string, matching it against legal moves
        so promotion and castling flags are filled in correctly."""
        parsed = Move.parse_uci(text)
        for candidate in self.legal_moves():
            if (candidate.origin == parsed.origin
                    and candidate.target == parsed.target
                    and candidate.promotion == parsed.promotion):
                return self.play(candidate)
        raise IllegalMoveError(f"no legal move matches {text!r}")

    def undo(self) -> Move:
        if not self.history:
            raise IllegalMoveError("nothing to undo")
        entry = self.history.pop()
        self.position_keys.pop()
        self.board = entry.board_before
        self._resigned = None
        return entry.move

    def resign(self, color: Color) -> GameStatus:
        self._resigned = color
        return self.status()

    # ---- rendering --------------------------------------------------------

    def ascii(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            cells = [self.board.grid[rank][file] for file in range(8)]
            rows.append(
                f"{rank + 1} " + " ".join(p.symbol if p else "." for p in cells)
            )
        rows.append("  a b c d e f g h")
        rows.append(f"  side: {self.board.side_to_move.value}   status: {self.status().value}")
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.ascii()


# --------------------------------------------------------------------------
# 6. Demo — Scholar's mate, which exercises check, checkmate and listeners
# --------------------------------------------------------------------------

if __name__ == "__main__":
    game = Game()
    game.add_listener(lambda g, m, s: print(f"  played {m.uci():<6} -> {s.value}"))

    for uci in ("e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"):
        game.play_uci(uci)

    print(game.ascii())
    print("final status:", game.status().value)          # checkmate
    print("history length:", len(game.history))          # 7
    game.undo()
    print("after undo:", game.status().value)            # ongoing (threat, not check)
```

**What to point at while the interviewer reads it:**

- `Pawn` is the only piece overriding `pseudo_legal_moves` — that is the template method earning its keep.
- `King` does **not** generate castling. Castling needs rights, empty squares and three attack tests, so it belongs to the `RuleEngine`. Saying this out loud is the strongest single sentence in the answer.
- `Board.apply` handles five things beyond "move the piece": capture, promotion, the castling rook, castling-rights update, and the en-passant target. It also resets/advances the halfmove clock.
- `_leaves_king_safe` makes a copy and applies. That is O(64) per candidate move, which is fine for correctness and is the honest trade named in §10.

---

## 6. Concurrency & thread-safety

### What breaks

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Two callers `play()` at once** | Both read the same `board`, both apply, one move is silently lost or a piece is duplicated | A per-game `threading.RLock` around validate-snapshot-apply, or a single-threaded actor per game |
| **Check-then-act on legality** | Thread A validates move M, thread B plays a different move, thread A applies M against the new position → an illegal position | Legality and application must be **one critical section**, never two calls |
| **Listener runs under the lock** | A listener that blocks (network, DB) holds the game lock and stalls every move | Notify **after** releasing the lock, or dispatch to a queue |
| **`undo()` racing `play()`** | The history list is mutated by two threads | The same lock; and `position_keys.pop()` must pair with `history.pop()` atomically |
| **Board sharing between games** | A copy-on-write assumption breaks when a caller passes the same `Board` to two `Game`s | Construct games from a **FEN string**, not from a shared `Board` object |
| **Global lock across all games** | A 10,000-game server serialises on one mutex; throughput collapses | Shard games by `game_id` and give each *one* lock — or one actor thread per game |

### The fix, in code

```python
import threading


class ThreadSafeGame(Game):
    """One lock per game. The lock covers *validate + apply* as a unit,
    because splitting them is the classic check-then-act race."""

    def __init__(self, board=None, rules=None):
        super().__init__(board, rules)
        self._lock = threading.RLock()

    def play(self, move):
        with self._lock:
            return super().play(move)

    def undo(self):
        with self._lock:
            return super().undo()

    def snapshot(self):
        """A consistent read for a spectator thread. Returning the *board*
        under the lock and rendering outside it keeps the lock short."""
        with self._lock:
            return self.board.copy(), self.status()
```

**The three sentences:**

1. *"One logical game is inherently sequential — two moves cannot be simultaneous — so the right model is a lock per game, not a global lock."*
2. *"The lock must span legality **and** application, because a legality check followed by an apply is a check-then-act race."*
3. *"Listeners must be invoked outside the lock. A `MoveListener` that writes to Kafka must not hold the game's mutex."*

### Optimistic alternative

If contention is genuinely high (an online platform with spectators), give the board a **version counter** and make `play` conditional:

```python
    # EXCERPT — a method of the ChessGame class defined in §5, indented because
    # it is a class member. The complete runnable class is in §5 above.
    def try_play(self, move, expected_version):
        with self._lock:
            if self.version != expected_version:
                return False          # someone else moved; caller re-reads
            self.play(move)
            self.version += 1
            return True
```

That is the same compare-and-set idea as [Uber's driver accept](../../Case-Studies/HLD/06-uber-ride-hailing.md) (§3.5), moved down a level.

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Add an AI opponent"

Add a `PlayerStrategy` implementation. The core does not change.

```python
class RandomPlayer:
    def __init__(self, rng): self.rng = rng
    def choose_move(self, game): return self.rng.choice(game.legal_moves())


class MinimaxPlayer:
    """Negamax with alpha-beta pruning. Depth 3-4 is a realistic interview answer."""

    PIECE_VALUE = {PieceType.PAWN: 100, PieceType.KNIGHT: 320, PieceType.BISHOP: 330,
                   PieceType.ROOK: 500, PieceType.QUEEN: 900, PieceType.KING: 20000}

    def __init__(self, rules, depth=4): self.rules, self.depth = rules, depth

    def evaluate(self, board, color):
        score = 0
        for sq, piece in board.iter_pieces(color):
            score += self.PIECE_VALUE[piece.piece_type]
        for sq, piece in board.iter_pieces(color.opponent):
            score -= self.PIECE_VALUE[piece.piece_type]
        return score

    def choose_move(self, game):
        _, move = self._search(game.board.copy(), game.rules, game.board.side_to_move,
                               self.depth, -10**9, 10**9, game.board.side_to_move)
        return move

    def _search(self, board, rules, to_move, depth, alpha, beta, root_color):
        if depth == 0:
            return self.evaluate(board, root_color), None
        best_move = None
        for move in rules.legal_moves(board, to_move):
            child = board.copy()
            child.apply(move)
            score, _ = self._search(child, rules, child.side_to_move, depth - 1,
                                    -beta, -alpha, root_color)
            score = -score
            if score > alpha:
                alpha, best_move = score, move
            if alpha >= beta:
                break
        sign = 1 if to_move is root_color else -1
        return sign * alpha, best_move
```

The important design point: **the strategy depends only on `legal_moves()` and `Board.copy()`** — which is why those two must exist and be cheap-ish. It also explains why `_leaves_king_safe` copying is the performance cliff for a search: at depth 4 with ~30 moves per ply, that is ~800,000 board copies. A real engine uses make/unmake or a bitboard; say so, and say you would not do it unless asked.

### 7.2 "Support Chess960"

Only three things change, and none is in the piece classes:
- `Board.from_fen` already reads arbitrary placement, so a 960 start position loads unchanged.
- Castling generation must generalise: the king's destination is still c6/g6 (`file 6`/`file 2`) and the rook's is f6/d6, but the *origin files* vary. Replace the hard-coded `rook_file = 7 or 0` with a lookup of the rook on the correct side of the king.
- `Board.apply`'s castling branch must place the rook on the file it derives, not `5`/`3` — and for queen-side 960 the rook can end up left of the king's target.

```python
    # EXCERPT — a method of the ChessGame class defined in §5, indented because
    # it is a class member. The complete runnable class is in §5 above.
    def _castling_rook_file(self, board, color, side):
        home_rank = 0 if color is Color.WHITE else 7
        king_file = self.find_king(board, color).file
        candidates = range(king_file + 1, 8) if side is CastleSide.KING_SIDE else range(king_file)
        for f in candidates:
            piece = board.piece_at(Square(f, home_rank))
            if isinstance(piece, Rook) and piece.color is color:
                return f
        return None
```

That is the whole change. It is a good answer because it shows the abstraction was drawn in the right place: **variants touch the rule engine, not the pieces.**

### 7.3 "Implement undo/redo"

Undo already exists via `HistoryEntry.board_before`. Redo is a second stack:

```python
class UndoRedoGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redo_stack: list[HistoryEntry] = []

    def play(self, move):
        status = super().play(move)
        self.redo_stack.clear()   # a new move invalidates the redo branch
        return status

    def redo(self):
        if not self.redo_stack:
            raise IllegalMoveError("nothing to redo")
        entry = self.redo_stack.pop()
        return self.play(entry.move)
```

**The design point:** you do not need to store "how to reverse a move" if you store the *previous board* — snapshotting is simpler and always correct, at O(64) memory per ply. A move-reversing `unmake` is faster and is the engine-grade answer; say both.

### 7.4 "Add a clock / time control"

A `Clock` per colour, ticked by the game loop, not by the rules:

```python
class Clock:
    def __init__(self, seconds: float):
        self.remaining = {Color.WHITE: seconds, Color.BLACK: seconds}
        self.last_started: Optional[float] = None

    def start(self, color, now): self.last_started = now
    def stop(self, color, now):
        self.remaining[color] -= now - self.last_started
        return self.remaining[color] > 0
```

Note the honesty: **a clock is a wall-clock concern, so it belongs outside the rule engine** — otherwise your chess logic becomes untestable without patching time.

### 7.5 "Serve thousands of concurrent games"

That is an HLD question wearing an LLD hat: shard `game_id → node`, one actor per game, a matchmaking service, and Redis for game discovery. Cross-link [Uber's matching design](../../Case-Studies/HLD/06-uber-ride-hailing.md) for the sharding pattern and keep the LLD core untouched.

### 7.6 "Serialise a game"

`Board.fen()` round-trips with `Board.from_fen()`, and the move list is a UCI list. Persisting `(fen, [uci moves])` is enough to replay. Note that **`fen()` deliberately excludes the move history**, so threefold repetition needs the key list persisted too — a subtle point worth raising before the interviewer does.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Moving into check** | Allows it, because the king is just another piece | Pseudo-legal moves are filtered by "does this leave my king attacked?" |
| 2 | **Castling through check** | Allows O-O with a bishop on f1's diagonal; only the destination is tested | Three tests: not in check, transit square not attacked, destination not attacked |
| 3 | **Castling after the rook moved and returned** | Allows it, because the rook is back on h1 | Castling rights are **state**, updated on rook/king moves; they do not "come back" |
| 4 | **Castling when the rook is captured on its home square** | Allows it | Capturing a rook on a1/a8/h1/h8 removes that right for the opponent |
| 5 | **En passant offered two plies later** | Allows it, because a pawn is adjacent | `ep_target` is valid for exactly one ply; it is cleared in `apply` unless a double push just happened |
| 6 | **En passant capturing the wrong pawn** | Removes the piece on the target square (which is empty) | The captured pawn is on `(target.file, origin.rank)`, not on the target square |
| 7 | **Pawn reaching the last rank with no promotion choice** | Leaves it as a pawn, or auto-queens without saying so | Generate four promotion moves and make the caller choose; under-promotion to a knight is a real winning tactic |
| 8 | **King captured** | Pseudo-legal generation allows a king capture, and the game continues with no king | Unreachable if legality is always enforced; `find_king` returning `None` must be handled defensively |
| 9 | **Stalemate reported as checkmate** | No legal moves is treated as a loss | Checkmate = no legal moves **and** in check; otherwise stalemate |
| 10 | **Threefold repetition never detected** | Comparing move lists instead of positions | Compare `position_key()` — placement, side, castling rights, ep target — **not** the move list |
| 11 | **Fifty-move rule triggered by the wrong clock** | Using `fullmove` (which counts plies/2) | Use the **halfmove** clock, reset on every pawn move and every capture; the threshold is 100 halfmoves |
| 12 | **Insufficient-material false positive** | Declaring a draw with K+B vs K+B on opposite colours | Same-colour-square bishops only; any pawn, rook or queen means material is sufficient |
| 13 | **Undo across a resignation** | Leaves `_resigned` set, so the game stays over | `undo()` clears the resignation |
| 14 | **`play()` after the game is over** | Continues happily | `is_over()` guard, raising `IllegalMoveError` |
| 15 | **UCI promotion string ignored** | `play_uci("e7e8")` matches nothing and throws a confusing error | Match against `legal_moves()`, so the four promotion candidates are distinguished by the 5th character |
| 16 | **Board copy aliasing** | `copy()` shares the inner lists → a test move corrupts the real board | Row-wise copy; pieces are immutable so sharing piece objects is safe |
| 17 | **Ambiguous SAN output** | `Nd2` when two knights can reach d2 | SAN disambiguation needs the legal-move set; put it in a `NotationFormatter`, not in `Move` |
| 18 | **Repetition across an undo** | `position_keys` and `history` desynchronise | Pop both in the same critical section |

**The two to volunteer before being asked:** #3 (castling rights are state, not derivable) and #10 (repetition is a *position* hash, not a move-list comparison). Both are the kind of thing that separates a candidate who has actually implemented chess from one who has sketched it.

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is the right tool here | What it costs |
|---|---|---|---|
| **Template Method** | `OffsetPiece` fixes the ray/jump iteration; `King`/`Queen`/`Rook`/`Bishop`/`Knight` supply `DIRECTIONS` + `SLIDING` | Five of six pieces differ only in a list of vectors. Without it, five near-identical 20-line methods | One more class in the hierarchy; `Pawn` has to opt out — which is the honest sign the template does not fit *everything* |
| **Polymorphism / Open-Closed** | `Piece` abstract method | Adding a fairy piece is a new class, not an edit to a switch | Slight indirection when debugging |
| **Command** | `Move` + `HistoryEntry` | Makes undo, redo, replay, network transport and notation all trivial: a move is data | A `Move` must carry non-obvious flags (`castle`, `en_passant`) that a naive `(from, to)` pair loses |
| **State** | `GameStatus` computed by the rule engine | Illegal transitions become unrepresentable: you cannot set `CHECKMATE` without the position proving it | Computation on every status read — cache it if it hurts |
| **Factory** | `make_piece()` + `PIECE_CLASSES` | One place maps an enum to a class; FEN loading and promotion both use it | A dict that must be kept in sync with the enum |
| **Observer** | `Game.add_listener` | UI, notation, telemetry and persistence all react to moves without the game knowing about them | Notification ordering is unspecified; a throwing listener breaks the loop unless wrapped |
| **Strategy** | `PlayerStrategy` | Human prompt, random AI and minimax AI are interchangeable | Only worth introducing once there is a second implementation — do not add it speculatively |
| **Value Object** | `Square`, `Move` frozen dataclasses | Hashable → `Move` works in sets and dicts; `move in legal_moves` is O(1)-ish and reads well | Immutability means constructing many objects in a search hot loop |
| **Singleton** | *Deliberately not used* for `RuleEngine` | It is stateless, so a module-level default is enough; a singleton would add global mutable state for nothing | — |

**The pattern answer to give:** *"Template Method for the pieces, Command for moves, State for status, Factory for piece construction, Observer for listeners. I would **not** use Singleton for the rule engine — it is stateless, so injection is free and testing stays easy."* Naming a pattern you deliberately rejected is worth more than naming a sixth one you used.

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Legality via board copy** | `board.copy()` then `apply` | Correctness is obvious; no unmake bugs; trivial to reason about | ~64 cells × ~30 moves per position — fine for play, catastrophic for a depth-4 search |
| **Make/unmake instead** | Not used | 10–100× faster search | Unmake must perfectly mirror apply for five special cases; one missed reversal corrupts the board silently |
| **Castling in the rule engine** | Yes | `King` stays a pure geometry class; rights and attack tests live where the whole board is available | The engine must know about king/rook home squares |
| **Pieces immutable (colour only)** | Yes | `Board.copy()` can share piece objects; no accidental cross-board mutation | You cannot store "has this rook moved" on the rook — which is exactly why rights live on the `Board` |
| **`Move` as a frozen value object** | Yes | Hashable; usable in sets; safe to log and send over a network | Non-obvious fields must be carried explicitly |
| **Status computed, not stored** | Yes | Impossible to have a `status` field that disagrees with the board | Recomputed on each read; cache with a dirty flag if profiling says so |
| **Position-key repetition** | FEN-like key, minus counters | Correct threefold detection with one string comparison | Must be maintained on every move and popped on undo |
| **Snapshot undo** | Store `board_before` | Always correct, 3 lines | O(64) memory per ply; a 200-move game holds 200 boards |
| **Listener notification outside the lock** | Yes | A slow listener cannot stall the game | Listener ordering across threads is unspecified |
| **Per-game lock, not a global lock** | Yes | Independent games scale independently | A caller holding two game locks can deadlock — forbid cross-game operations |
| **Full FIDE rules vs simplified** | Full | The answer survives "what about en passant?" | Roughly 3× the code of the naive version |

**The trade to name unprompted:** *"The board-copy legality check is the one place I am trading performance for obviousness. It is the right call for a game object, and it is exactly the wrong call for a search engine — which is why a real engine uses make/unmake or bitboards."*

---

## 11. Q&A

**Q1. Why is `Piece` abstract rather than an enum plus a switch?**
Because movement is the only thing that varies between pieces and it varies completely. An abstract method makes each piece self-contained, makes adding a piece type a new file, and makes the compiler enforce completeness. A switch on `piece_type` grows without bound and every new piece is an edit to a function that already works — a textbook Open/Closed violation. The one place I do switch on type is `make_piece`, the factory, and that is deliberate: it is the single mapping point.

**Q2. Why doesn't `King` generate castling moves?**
Because castling depends on state the king cannot see: whether castling rights still exist, whether the squares between are empty, and whether the king is in check, passes through check, or lands in check. Those are whole-board questions, so they belong to the `RuleEngine`. Putting them in `King` would force the piece to reach into board state and duplicate attack logic.

**Q3. How do you know a move is legal?**
Two stages. First, the piece generates **pseudo-legal** moves from its own geometry — these ignore check entirely. Second, the rule engine copies the board, applies the candidate move, and asks whether the mover's own king is now attacked. If not, the move is legal. The split is what keeps the piece classes at ten lines each.

**Q4. Where does en passant live, and what is the classic bug?**
In two places. `Pawn.pseudo_legal_moves` offers the diagonal move when the target square equals `board.ep_target`, and `Board.apply` removes the captured pawn from `(target.file, origin.rank)` — **not** from the target square, which is empty. The classic bug is removing the piece on the target square, which silently does nothing and leaves a phantom pawn that can later be captured. The second bug is not clearing `ep_target`: it is valid for exactly one ply.

**Q5. How do you stop a king from castling after its rook has moved back?**
By modelling castling rights as explicit board state — a set of `(color, side)` — rather than deriving them from where the pieces currently are. Rights are removed when the king moves, when a rook leaves its home square, and when a rook is captured **on** its home square. Crucially, they are never restored. A piece-based check would say "there is a rook on h1, so castling is fine" and would be wrong.

**Q6. How do you detect checkmate versus stalemate?**
Generate the legal moves for the side to move. If there are none and the side is in check, it is checkmate. If there are none and the side is **not** in check, it is stalemate. `checkmate` is strictly stronger than `stalemate` in the test order — testing "no moves" first and calling it checkmate is a real bug.

**Q7 (senior). How would you detect threefold repetition, and why can't you just compare the move list?**
I compare a **position key**: piece placement, side to move, castling rights and the en-passant target — everything that makes two positions "the same position" under the FIDE rule. The move list is the wrong comparison because different move sequences routinely transpose into the identical position, and the rule is about the position, not the path. Counterintuitively, the *halfmove clock and fullmove number are excluded*, because the rule ignores them.

**Q8 (senior). Why does the fifty-move rule use the halfmove clock and not `fullmove`?**
`fullmove` increments after Black's move, so it counts move *pairs*, not plies. The halfmove clock counts plies since the last **capture or pawn move** — and it is the pawn move that resets it, which is the subtle part, because a pawn move is the only thing that can permanently change the position class. The threshold is 100 halfmoves.

**Q9. How would you add undo without writing an unmake for every special move?**
Store the previous board with each history entry. Correctness is free, and the cost is one 64-cell copy per ply. If I needed engine-grade speed I would implement make/unmake, but then `unmake` has to reverse five things perfectly — capture, promotion, castling rook, castling rights and the ep target — and one missed reversal corrupts the board silently. For an LLD interview, the snapshot is the better trade; I would say both.

**Q10. Is a `Game` thread-safe? What is the race if two threads call `play()`?**
It is not, and the race is a **check-then-act**: thread A validates move M against position P, thread B plays a different legal move, and thread A then applies M against position P′ where M may be illegal. The fix is one lock per game covering *both* validation and application as a single critical section. A global lock across all games would serialise the whole server, so per-game (or per-actor) is the model. And listeners must fire outside the lock, because a listener that writes to a queue or a socket must not hold the game mutex.

**Q11. Two `Game` objects share a `Board`. What breaks?**
Everything, silently. `Board` is mutable and `copy()` is shallow on the grid *rows* only in the sense that a new outer list plus new inner lists are created — but if a caller passes the same `Board` instance to two games, they mutate the same grid. The defensive design is to construct every game from a FEN string rather than a shared object, which makes ownership unambiguous.

**Q12. How would you support Chess960 with minimal change?**
Three changes and none in the piece classes: the FEN loader already handles arbitrary placement; castling generation generalises by locating the rook on the correct side of the king instead of assuming files 0 and 7; and `Board.apply`'s castling branch places the rook using the same derivation. That the pieces need no change at all is the evidence that the abstraction boundary was drawn correctly.

**Q13. Why is `NotationFormatter` a separate collaborator rather than a `Move.to_san()` method?**
SAN is ambiguous without the full legal-move set: `Nd2` must become `Nbd2` or `Nfd2` when two knights can reach d2, and `+`/`#` suffixes require knowing the *resulting* position's check status. So the formatter needs the game before and after — a `Move` cannot compute its own SAN. Keeping it external also means the game core has no presentation dependency.

**Q14. `Move` is frozen. Why does that matter?**
Three reasons: it is hashable, so `move in legal_moves` and set-based deduplication work; it is safe to hand to a listener, log, or send over a network without defensive copying; and it makes accidental identity bugs (two "same" moves that are different objects) into equality bugs, which are far easier to find. The immutability also forces the non-obvious flags — `castle`, `en_passant`, `promotion` — to be part of the value rather than side-channel state.

**Q15 (senior). Your legality check copies the board for every candidate move. Where does that hurt, and what would you do?**
It hurts the moment anything searches: at depth 4 with ~30 moves per ply, roughly 800,000 board copies, each allocating 8 lists and 64 slots. I would first make the copy cheaper (a single 64-element flat list), then implement make/unmake with an explicit undo record, and only then consider bitboards — which is a different program, not a refactor. But I would keep the copying version as the reference implementation and test the fast one against it, because the fast path's failure mode is silent corruption.

**Q16 (senior). Your engine says a position is a draw by insufficient material. What are the cases, and the trap?**
King versus king; king and bishop versus king; king and knight versus king; and king and bishop versus king and bishop **with both bishops on the same colour square**. The trap is the last one: opposite-coloured bishops can mate, so a naive "two bishops total" test declares a draw in a position that is not dead. Also, if any pawn, rook or queen is on the board, material is sufficient by definition — a pawn can promote.

---

### Pushback round

**P1. "You have a `RuleEngine` and a `Board` — isn't that just an anemic domain model? Why not put `is_legal` on the `Board`?"**
It is a fair objection and the honest answer is that I am optimising for *change*, not for purity. `Board` is the mutable state; `RuleEngine` is a stateless function over it. Fusing them would put every rule in one large class that changes for many reasons — new piece types, new castling variants, new draw rules — and would make it impossible to swap in a different rule set (Chess960, a puzzle mode, a variant with no castling) without forking the board. The separation also makes the rule engine trivially unit-testable against FEN fixtures with no game object at all. If the interviewer pushed further and said "this is a small program, a rich `Board` would be simpler", I would agree for a 200-line toy and disagree for anything that will grow — and I would say which I am building.

**P2. "Your `_leaves_king_safe` copies the whole board for every candidate move. That is a performance bug in a game loop."**
For a two-player game it is invisible: a position has ~30 legal moves, so ~30 copies of 64 slots — microseconds, versus the ~100 ms a human takes to move. It becomes a genuine bug the moment a search appears, which is why I named the boundary explicitly: this is the reference implementation, and the search path needs make/unmake. I would rather ship an obviously-correct rule engine and pay for the search path deliberately than ship a fast unmake that silently corrupts the board in the en-passant case. I will also say the measurement: if p99 move latency exceeded a millisecond, I would profile before rewriting.

**P3. "Why not just store the board as a 64-character string or use python-chess? You are rebuilding a solved wheel."**
If the interviewer wants a working chess program, `python-chess` is the correct answer in about ten seconds and I would say so. The question is a *class design* question, and the design is the deliverable — the point is to demonstrate a piece hierarchy, a legality layer, Command undo, and the special-move state model. Using a library would also hide exactly the parts being graded: castling rights, en passant, and the pseudo-legal-versus-legal split. I would say that explicitly rather than pretend a library does not exist.

**P4. "You said the pieces are immutable, so `Board.copy()` can share them. But then how does a rook know it has moved?"**
It does not need to, and that is the design decision: "has this rook moved" is **not a property of the rook**, it is a property of the position. It is stored as the castling-rights set on the `Board`, and it is updated in `apply` by three rules — king moves, rook moves, rook captured on its home square. Making the piece mutable to hold a `has_moved` flag would introduce exactly the state that breaks `copy()`, because a shared piece object would then carry one board's history into another's. Immutable pieces plus explicit rights is the combination that makes copying safe.

**P5. "Your `status()` recomputes everything on every call. Two calls in a row, two full move generations."**
True, and it is a deliberate trade for correctness: there is no way for a cached status to disagree with the board. If it showed up in a profile, the fix is a dirty flag on the `Board` incremented by `apply` — status is recomputed when the version changes and cached otherwise. I would not add that preemptively, because the flag is one more piece of state that can go stale, and a stale status in a chess game is a wrong result, not a slow one.

---

## 12. Cheat-sheet summary + cross-links

| Concept | One-line truth |
|---|---|
| The core split | Pieces know **geometry**; the rule engine knows **legality**; the board knows **state**; the game knows **turn order** |
| Pseudo-legal | Piece-generated, check-blind — the only thing a `Piece` returns |
| Legal | Pseudo-legal + "mover's king is not attacked afterwards" |
| Castling | Generated by the **rule engine**, never by `King`; three attack tests, not one |
| Castling rights | Explicit **state**, removed on king move / rook move / rook captured at home; **never restored** |
| En passant | `ep_target` valid for exactly one ply; the captured pawn is on `(target.file, origin.rank)` |
| Promotion | Generate four moves; never auto-queen silently; under-promotion wins games |
| Checkmate vs stalemate | No legal moves **and** in check = checkmate; no legal moves and not in check = stalemate |
| Fifty-move | Halfmove clock ≥ 100; reset on **pawn move or capture**; not `fullmove` |
| Threefold | Compare a **position key** (placement, side, rights, ep), never the move list |
| Insufficient material | K-K, K+B-K, K+N-K, and same-colour-square bishops; any pawn/rook/queen means sufficient |
| Undo | Snapshot the previous board (simple, correct) or make/unmake (fast, mirror-exact) |
| Concurrency | One lock **per game**; the lock spans validate **and** apply; notify listeners outside it |
| Patterns | Template Method (pieces), Command (moves), State (status), Factory (piece construction), Observer (listeners), Strategy (AI) |
| Rejected pattern | Singleton for `RuleEngine` — it is stateless; inject it instead |
| The performance cliff | Board-copy legality is fine for play, fatal for search — name the boundary |
| Chess960 | Touches the rule engine's castling derivation only; **zero piece changes** |
| Serialisation | FEN round-trips the position; repetition needs the key list persisted too |

**The three sentences to memorise**

1. *"Pieces generate pseudo-legal moves; a rule engine filters them through the king-safety test."*
2. *"Castling, en passant and the fifty-move clock are board state that is not in the grid — that is where candidates lose points."*
3. *"Undo is a Command plus a snapshot; one lock per game spans validate and apply."*

### Cross-links

- **The LLD framing this file assumes:** [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md) — the parking-lot example, and the HLD/LLD table that this file's section shape follows
- **Same family (state machines and boards):** [Tic-Tac-Toe](14-tic-tac-toe.md) — a much smaller board and the same three-layer split · [Snake and Ladder](15-snake-and-ladder.md) — dice-driven state transitions
- **Concurrency technique reused here:** [Rate Limiter](11-rate-limiter.md) — the same "validate and mutate in one critical section" rule · [Pub-Sub](13-pub-sub-system.md) — the Observer pattern at system scale
- **Pattern catalogue and the concurrency primitives table:** [LLD Cheatsheet](../../Cheat-Sheets/07-lld-cheatsheet.md)
- **Question bank:** [LLD Questions](../../Interview-Questions/08-lld-questions.md) — Part E covers chess directly, and Part G is the pushback round
- **An HLD system with a state machine at its centre:** [Uber Ride Hailing](../../Case-Studies/HLD/06-uber-ride-hailing.md) §3.5 (trip state machine, compare-and-set transitions) · [Uber LLD bridge](../../Case-Studies/LLD/03-uber-matching-lld.md)
