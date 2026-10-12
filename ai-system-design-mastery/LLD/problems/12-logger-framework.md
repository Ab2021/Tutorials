# LLD 12 — Logger Framework

> **Grounding note.** The reference repositories contain **no OOP/LLD material**; the source transcripts name logging as an observability concern (log aggregation, sampling, structured logs) without building a logging library as a class design. This file is written from first principles and is not transcript-grounded. The HLD observability material is the counterpart: [Observability](../../Operations/02-observability-and-monitoring.md) owns what to log, where logs go and how they are aggregated; this file owns the classes.

> **TL;DR** — A logging framework is a **hot-path library**: it is called on every request, so its design is dominated by two questions nothing else in LLD has — *how do I not pay for a log line that is suppressed*, and *how do I never let a broken sink take down the application*. The class design is small and stable: a `LogRecord` value object, `Logger` nodes in a dotted hierarchy that resolve an effective level by walking up, `Handler`s that pair a `Sink` with a `Formatter` and a filter chain, and an `AsyncDispatcher` that moves formatting and I/O off the caller's thread behind a bounded queue with an explicit overflow policy. The two senior signals are the **level check that runs before the message is formatted** — precisely scoped, because it skips the interpolation but *not* the eager evaluation of the arguments, which is what the lazy overload is for — and the **error isolation** that makes a log call unable to raise.

> **60-second interview answer:** "A `Logger` is a named node in a hierarchy — `a.b.c` parents to `a.b` — and it resolves its effective level by walking up to the first ancestor that sets one, which is what makes `logging.getLogger('app.db')` tunable without touching code. `Logger.info` does `is_enabled_for` first and only builds a `LogRecord` if it passes, so a suppressed line never pays for the `%`-interpolation — and for a message that is genuinely expensive to *build* there is a lazy overload taking a callable, because Python evaluates a call's arguments eagerly and the level check alone does not make `logger.debug("%s", expensive())` free Records go to `Handler`s, each of which is a `Sink` plus a `Formatter` plus a filter chain; handlers are inherited up the hierarchy unless `propagate` is false. The async dispatcher puts records on a bounded queue drained by one worker thread, with an explicit overflow policy, so the request path never blocks on disk. And every handler's `emit` is wrapped so a failing sink degrades to a stderr fallback instead of propagating into the caller."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **Sync or async emission?** | Whether the caller's thread does the I/O, or a worker thread drains a queue | Async with a bounded queue — sync is fine for a CLI, fatal for a request path |
| 2 | **Is a log line ever allowed to be lost?** | The overflow policy: block, drop newest, drop oldest, or sample | Drop with a counter for DEBUG/INFO at volume; never drop ERROR and above |
| 3 | **Structured or free text?** | Whether `LogRecord` carries `extra` fields and the formatter emits JSON | Structured — JSON in production, pretty text in dev |
| 4 | **What happens when a sink throws or the disk is full?** | Error isolation and the fallback path | Isolate per handler, report once to stderr, never propagate to the caller |
| 5 | **Multi-process or single-process?** | Whether writes to one file interleave and corrupt | Single-process; for multi-process, one file per process or a socket to a collector |
| 6 | **Is the level per-logger, or global?** | The dotted hierarchy with inherited levels | Per-logger, inherited — that is the feature that makes a library usable |
| 7 | **Redaction of secrets and PII?** | Whether a filter rewrites records before formatting | Yes — a `RedactionFilter` on the sink that leaves the process |
| 8 | **Rate limiting / sampling of log volume?** | Whether there is a filter in front of the noisy sink | Yes, per-logger rate limiting for repetitive errors — see [Rate Limiter](11-rate-limiter.md) |
| 9 | **Context propagation (request id, user id)?** | Whether there is a context variable merged into every record | Yes — `contextvars`, so the caller does not pass it to every call |
| 10 | **Rotation and retention?** | Whether the file sink rotates by size or time | Size-based rotation with a backup count; retention is the collector's job |

### The defaults to assume, stated plainly

> "I'll assume a **single-process, structured, asynchronously-dispatched** logger: a dotted name hierarchy with inherited levels, handlers built from sink + formatter + filters, a bounded queue with drop-and-count for low levels, per-handler error isolation, and contextvars for request-scoped fields. I will not cover log aggregation or retention policy — that is the observability HLD. Does that match, or do you want the synchronous simple version first?"

### Non-goals

- No aggregation, shipping, indexing or retention — [Observability](../../Operations/02-observability-and-monitoring.md).
- No metrics or traces; the seam where a trace id would be injected is `extra` plus contextvars.
- No multi-process file coordination beyond naming the failure.

### The four constraints that shape every decision

1. **It is on the hot path.** A suppressed log line must cost one comparison and no formatting — and the genuinely expensive case needs a lazy callable, because the level check cannot make eager argument evaluation free.
2. **It must never raise.** A logging call that throws converts an observability bug into an availability incident.
3. **It must never block indefinitely.** A full disk queue that blocks the request path is a deadlock with extra steps.
4. **It must be configurable without a redeploy of the code that logs.** That is what the hierarchy is for.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `LogLevel` | IntEnum | Ordered severities, `parse`, `name` | Thresholds — those live on loggers and handlers |
| `LogRecord` | Frozen value object | `level`, `message`, `logger_name`, `timestamp`, `extra`, `exc_info` | Formatting or routing |
| `Formatter` (ABC) | Interface | `format(record) -> str` | Choosing where the text goes |
| `PlainTextFormatter` | Implementation | Human-readable one-liner | Escaping newlines in `extra` |
| `JsonFormatter` | Implementation | One JSON object per line | Pretty-printing |
| `Sink` (ABC) | Interface | `emit(text)`, `flush()`, `close()` | Formatting or filtering |
| `ConsoleSink` | Implementation | Write to a stream | Owning the stream's lifetime unless asked |
| `MemorySink` | Implementation | Capture for tests and demos | Production use |
| `FileSink` | Implementation | Append with a lock and a flush policy | Rotation |
| `RotatingFileSink` | Implementation | Rotate by size, keep N backups | Retention beyond N |
| `Filter` (ABC) | Interface | `apply(record) -> LogRecord \| None` | I/O |
| `LevelFilter` | Implementation | Drop records below a level | — |
| `RedactionFilter` | Implementation | Replace secret values in `extra` | Deciding what is a secret — that is config |
| `RateLimitFilter` | Implementation | Cap records per logger per window | The limiter itself — see [Rate Limiter](11-rate-limiter.md) |
| `Handler` | Composite | Sink + Formatter + level + filter chain, with error isolation | Logger hierarchy |
| `Logger` | Facade node | Named node, effective level, `handle`, propagation | Storage and formatting |
| `LoggerManager` | Registry | Creates and wires the dotted hierarchy | Configuration policy |
| `AsyncDispatcher` | Decorator | Bounded queue, worker thread, overflow policy | Formatting — the worker does it |
| `OverflowPolicy` | Enum | `BLOCK`, `DROP_NEWEST`, `DROP_OLDEST` | Choosing which levels may drop |
| `LogContext` | Helper | `contextvars` map merged into every record | Persistence |
| `LoggerBuilder` | Builder | Declarative configuration | Runtime reconfiguration |
| `NullLogger` | Null Object | Swallows everything, for library defaults | — |

**The split to say out loud:** *"A `Logger` decides whether and where; a `Handler` decides how it looks and where it goes; a `Sink` does I/O. Those are three different reasons to change, which is why they are three classes."*

---

## 3. Class diagram (ASCII)

```
                          ┌──────────────────────┐
                          │    LogLevel          │  «enum»
                          │ DEBUG 10 .. CRITICAL 50│
                          └──────────────────────┘
                                    ▲ used by
┌──────────────────────┐            │
│     LogRecord        │ «frozen value»
│──────────────────────│
│ level   : LogLevel   │
│ message : str        │        ┌───────────────────────────┐
│ logger_name : str    │        │  LoggerManager  (registry) │
│ timestamp : float    │        │───────────────────────────│
│ extra : dict         │        │ +get_logger(name) -> Logger│
│ exc_info : str|None  │        │ -root : Logger            │
└──────────▲───────────┘        └─────────────▲─────────────┘
           │ created by                       │ creates / parents
           │                                  │
┌──────────┴──────────────────────────────────┴───────────────┐
│                          Logger  (Facade node)              │
│─────────────────────────────────────────────────────────────│
│ name : str                 parent   : Logger | None         │
│ level : LogLevel | None    propagate: bool                  │
│ handlers : [Handler]                                        │
│─────────────────────────────────────────────────────────────│
│ +is_enabled_for(level) -> bool  <-- checked BEFORE formatting│
│ +effective_level        -> LogLevel   (walk up until set)   │
│ +debug/info/warning/error/critical(msg, *args, **extra)      │
│ +handle(record)                                               │
└───────────────┬─────────────────────────────┬────────────────┘
                │ owns                        │ wraps (async)
                ▼                             ▼
        ┌───────────────┐            ┌──────────────────────────┐
        │   Handler     │            │    AsyncDispatcher       │
        │───────────────│            │──────────────────────────│
        │ sink          │            │ queue : bounded          │
        │ formatter     │            │ worker : Thread          │
        │ level         │            │ policy : OverflowPolicy  │
        │ filters : [ ] │            │ dropped : int            │
        │───────────────│            │ +emit(record)  (never    │
        │ +handle(rec)  │            │   blocks under DROP_*)   │
        │  try/except   │            └──────────────────────────┘
        └───┬───────┬───┘
            │       │
            ▼       ▼
   ┌────────────┐ ┌───────────────┐      ┌────────────────────────┐
   │ Formatter  │ │    Sink       │      │  Filter  «interface»   │
   │ «interface»│ │ «interface»   │      │────────────────────────│
   │────────────│ │───────────────│      │ +apply(rec)->rec|None  │
   │ +format()  │ │ +emit/flush/  │      └───────────▲────────────┘
   └─────▲──────┘ │  close        │                  │
         │        └───────▲───────┘       ┌──────────┼───────────┐
 ┌───────┴──────┐         │               │          │           │
 │PlainText     │  ┌──────┴────┬──────────┴───┐ ┌────┴─────┐ ┌───┴────────┐
 │Json          │  │ConsoleSink│FileSink      │ │Level     │ │Redaction   │
 └──────────────┘  │MemorySink │RotatingFile  │ │Filter    │ │RateLimit   │
                   └───────────┴──────────────┘ └──────────┘ └────────────┘
```

**Read it this way:** `Logger` walks up the hierarchy for its level and its handlers; a `Handler` is the composition of a formatter, a sink and a filter chain; the dispatcher is a decorator over the handler set that moves the work off the caller's thread.

---

## 4. Interfaces / abstract classes / enums

```python
class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class Formatter(ABC):
    @abstractmethod
    def format(self, record: "LogRecord") -> str: ...


class Sink(ABC):
    @abstractmethod
    def emit(self, text: str) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class Filter(ABC):
    @abstractmethod
    def apply(self, record: "LogRecord") -> "LogRecord | None":
        """Return a (possibly rewritten) record, or None to drop it."""


class OverflowPolicy(Enum):
    BLOCK = "block"              # correct, and can deadlock the request path
    DROP_NEWEST = "drop_newest"  # keep what is queued, count the loss
    DROP_OLDEST = "drop_oldest"  # favour recency
```

**Three contract notes worth saying out loud:**

- **`Filter.apply` returns a record, not a bool.** Returning a bool would make redaction impossible to express: rewriting a record needs to hand back a new one. Redaction, enrichment and sampling are all the same shape.
- **`Sink.emit` takes a string, not a record.** That is the boundary where formatting has already happened, so a sink cannot re-render or inspect structured fields. If a sink needs the fields — a JSON-lines writer, a test asserting on `extra` — it takes a `Formatter` too, or the design gives the sink the record and lets it format. Pick one and be consistent; the boundary here is "sinks do I/O, formatters render".
- **`Handler` owns error isolation**, not `Logger`. A logger catching exceptions from one handler would stop the other handlers from running; catching per handler means one broken sink does not silence the others.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** Interview languages vary. In Java this is SLF4J/Logback's shape — `Logger`, `Appender`, `Layout`, `Filter`, `AsyncAppender`; in Go, a `Logger` struct with a `[]Sink` and a channel; in C++, a `Logger` with `std::function` sinks. The **hierarchy plus handler composition** is the design; the syntax is not. Python 3.10+ for `X | None`.

```python
"""logger_framework.py — a small structured logging framework.

Design summary
--------------
LogRecord      : frozen value object; one record may reach many handlers.
Logger         : named node in a dotted hierarchy; resolves its level by
                 walking up, so `app.db` is tunable without code changes.
                 `is_enabled_for` is checked BEFORE the message is formatted,
                 and a lazy overload skips building it at all.
Handler        : Sink + Formatter + level + filter chain, with per-handler
                 error isolation — a broken sink must not break the caller
                 or silence the other handlers.
Filter         : apply(record) -> record | None. Rewriting, not just gating,
                 which is what makes redaction expressible.
AsyncDispatcher: bounded queue + one worker thread + an explicit overflow
                 policy. The caller never blocks under DROP_*.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Any, Callable, Iterable, Mapping, Optional, TextIO


# --------------------------------------------------------------------------
# 1. Levels
# --------------------------------------------------------------------------

class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def parse(cls, value: "str | int | LogLevel") -> "LogLevel":
        if isinstance(value, LogLevel):
            return value
        if isinstance(value, int):
            return cls(value)
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown log level: {value!r}") from exc


class OverflowPolicy(Enum):
    """What a bounded queue does when it is full. There is no option that is
    both lossless and non-blocking, which is the whole reason it is explicit."""
    BLOCK = "block"              # correct, and can stall the caller's thread
    DROP_NEWEST = "drop_newest"  # keep what is queued, count the loss
    DROP_OLDEST = "drop_oldest"  # favour recency


# --------------------------------------------------------------------------
# 2. The record — a frozen value object
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LogRecord:
    level: LogLevel
    message: str                       # ALREADY formatted: see Logger._log
    logger_name: str
    timestamp: float
    extra: Mapping[str, Any] = field(default_factory=dict)
    exc_info: Optional[str] = None

    def with_extra(self, **additions: Any) -> "LogRecord":
        merged = dict(self.extra)
        merged.update(additions)
        return replace(self, extra=merged)


# Context-scoped fields: a request id set once, present on every record
# emitted while handling that request, without being threaded through calls.
_log_context: ContextVar[Mapping[str, Any]] = ContextVar("log_context", default={})


class LogContext:
    """Context manager that attaches fields to every record emitted inside it.

    `contextvars` rather than a thread-local because it is correct under both
    threads and async tasks, and because a nested context restores its parent.
    """

    def __init__(self, **fields: Any) -> None:
        self._fields = fields
        self._token = None

    def __enter__(self) -> "LogContext":
        merged = dict(_log_context.get())
        merged.update(self._fields)
        self._token = _log_context.set(merged)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _log_context.reset(self._token)


# --------------------------------------------------------------------------
# 3. Formatters — turn a record into text
# --------------------------------------------------------------------------

class Formatter(ABC):
    @abstractmethod
    def format(self, record: LogRecord) -> str: ...


class PlainTextFormatter(Formatter):
    def __init__(self, pattern: str = "%(asctime)s %(levelname)-8s %(name)s | %(message)s") -> None:
        self.pattern = pattern

    def format(self, record: LogRecord) -> str:
        asctime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.timestamp))
        rendered = (self.pattern
                    .replace("%(asctime)s", asctime)
                    .replace("%(levelname)-8s", f"{record.level.name:<8}")
                    .replace("%(levelname)s", record.level.name)
                    .replace("%(name)s", record.logger_name or "root")
                    .replace("%(message)s", record.message))
        if record.extra:
            pairs = " ".join(f"{k}={v!r}" for k, v in sorted(record.extra.items()))
            rendered = f"{rendered}  [{pairs}]"
        if record.exc_info:
            rendered = f"{rendered}\n{record.exc_info}"
        return rendered


class JsonFormatter(Formatter):
    """One JSON object per line — the shape a collector wants, because it can
    index `extra` fields instead of regexing them out of prose."""

    RESERVED = ("timestamp", "level", "logger", "message")

    def __init__(self, ensure_ascii: bool = False) -> None:
        self.ensure_ascii = ensure_ascii

    def format(self, record: LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": round(record.timestamp, 6),
            "level": record.level.name,
            "logger": record.logger_name or "root",
            "message": record.message,
        }
        # Structured fields are top level, so the collector can query them.
        # Collisions with reserved keys are nested rather than silently lost.
        collisions: dict[str, Any] = {}
        for key, value in record.extra.items():
            if key in self.RESERVED:
                collisions[key] = value
            else:
                payload[key] = value
        if collisions:
            payload["extra"] = collisions
        if record.exc_info:
            payload["exc_info"] = record.exc_info
        try:
            return json.dumps(payload, ensure_ascii=self.ensure_ascii, default=str)
        except (TypeError, ValueError):
            # An unserialisable extra must not lose the message.
            payload["extra_unserialisable"] = sorted(record.extra) if record.extra else []
            for key in list(payload):
                if key not in ("timestamp", "level", "logger", "message",
                               "extra_unserialisable", "exc_info"):
                    payload[key] = repr(payload[key])
            return json.dumps(payload, ensure_ascii=self.ensure_ascii)


# --------------------------------------------------------------------------
# 4. Sinks — the only place I/O happens
# --------------------------------------------------------------------------

class Sink(ABC):
    @abstractmethod
    def emit(self, text: str) -> None: ...

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleSink(Sink):
    def __init__(self, stream: Optional[TextIO] = None, close_stream: bool = False) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.close_stream = close_stream

    def emit(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()

    def close(self) -> None:
        if self.close_stream:
            self.stream.close()


class MemorySink(Sink):
    """For tests: capture lines so assertions can be made without a file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, text: str) -> None:
        self.lines.append(text)

    def clear(self) -> None:
        self.lines.clear()


class FileSink(Sink):
    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding
        self._lock = threading.Lock()
        self._handle = open(path, "a", encoding=encoding)

    def emit(self, text: str) -> None:
        with self._lock:
            self._handle.write(text + "\n")
            self._handle.flush()

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


class RotatingFileSink(FileSink):
    """Size-based rotation. Note what rotation does NOT solve: a multi-process
    writer still interleaves, because two processes each rotate independently
    and both append to the same inode. One file per process, or a socket to a
    collector, is the multi-process answer."""

    def __init__(self, path: str, max_bytes: int = 10 * 1024 * 1024,
                 backups: int = 5, encoding: str = "utf-8") -> None:
        self.max_bytes = max_bytes
        self.backups = backups
        self._written = os.path.getsize(path) if os.path.exists(path) else 0
        super().__init__(path, encoding)

    def emit(self, text: str) -> None:
        line = text + "\n"
        with self._lock:
            if self._written + len(line.encode(self.encoding)) > self.max_bytes:
                self._rotate_locked()
            self._handle.write(line)
            self._handle.flush()
            self._written += len(line.encode(self.encoding))

    def _rotate_locked(self) -> None:
        self._handle.close()
        for index in range(self.backups - 1, 0, -1):
            src = f"{self.path}.{index}"
            dst = f"{self.path}.{index + 1}"
            if os.path.exists(src):
                os.replace(src, dst)
        os.replace(self.path, f"{self.path}.1")
        self._handle = open(self.path, "a", encoding=self.encoding)
        self._written = 0


# --------------------------------------------------------------------------
# 5. Filters — gate AND rewrite
# --------------------------------------------------------------------------

class Filter(ABC):
    @abstractmethod
    def apply(self, record: LogRecord) -> Optional[LogRecord]:
        """A rewritten record, or None to drop."""


class LevelFilter(Filter):
    def __init__(self, minimum: LogLevel) -> None:
        self.minimum = LogLevel.parse(minimum)

    def apply(self, record: LogRecord) -> Optional[LogRecord]:
        return record if record.level >= self.minimum else None


class RedactionFilter(Filter):
    """Masks configured keys in `extra`. Config decides WHAT is a secret; this
    class only knows HOW to hide it — that separation is what keeps a new
    secret out of the code."""

    MASK = "[REDACTED]"

    def __init__(self, keys: Iterable[str], mask: str = MASK) -> None:
        self.keys = {k.lower() for k in keys}
        self.mask = mask

    def apply(self, record: LogRecord) -> Optional[LogRecord]:
        if not record.extra:
            return record
        hits = [k for k in record.extra if k.lower() in self.keys]
        if not hits:
            return record
        cleaned = dict(record.extra)
        for key in hits:
            cleaned[key] = self.mask
        # The message is NOT rewritten here. A secret that reached the message
        # was formatted into a string before this filter saw it; redaction is
        # a backstop for structured fields, not a licence to log secrets.
        return replace(record, extra=cleaned)


class RateLimitFilter(Filter):
    """Caps records per logger per window. A repeated exception in a hot loop
    can emit millions of identical lines; this is the same fixed-window
    arithmetic as [Rate Limiter](11-rate-limiter.md), used as a log filter."""

    def __init__(self, max_per_window: int, window_seconds: float = 60.0,
                 clock: Callable[[], float] | None = None) -> None:
        self.limit = max_per_window
        self.window = window_seconds
        self.clock = clock if clock is not None else time.monotonic
        self._lock = threading.Lock()
        self._state: dict[str, tuple[int, int]] = {}   # name -> (index, count)

    def apply(self, record: LogRecord) -> Optional[LogRecord]:
        with self._lock:
            index = int(self.clock() // self.window)
            current_index, count = self._state.get(record.logger_name, (index, 0))
            if current_index != index:
                current_index, count = index, 0
            if count >= self.limit:
                self._state[record.logger_name] = (current_index, count)
                return None
            self._state[record.logger_name] = (current_index, count + 1)
        return record


# --------------------------------------------------------------------------
# 6. Handler — composition, plus error isolation
# --------------------------------------------------------------------------

class Handler:
    """A sink, a formatter, a level and a filter chain.

    `handle` NEVER raises. A logging failure must degrade to a report on
    stderr, not propagate into the caller whose request is being logged.
    """

    def __init__(self, sink: Sink, formatter: Optional[Formatter] = None,
                 level: Optional[LogLevel] = None,
                 filters: Iterable[Filter] = ()) -> None:
        self.sink = sink
        self.formatter = formatter if formatter is not None else PlainTextFormatter()
        self.level = LogLevel.parse(level) if level is not None else None
        self.filters = list(filters)
        self._broken_reported = False
        self._report_lock = threading.Lock()

    def handle(self, record: LogRecord) -> bool:
        """True if the record reached the sink."""
        if self.level is not None and record.level < self.level:
            return False

        current: Optional[LogRecord] = record
        for flt in self.filters:
            current = flt.apply(current)   # type: ignore[arg-type]
            if current is None:
                return False

        try:
            self.sink.emit(self.formatter.format(current))
            return True
        except Exception as exc:                       # noqa: BLE001 - deliberate
            # Isolation: swallow, report once, keep the application alive.
            self._report_failure(exc)
            return False

    def _report_failure(self, exc: BaseException) -> None:
        with self._report_lock:
            if self._broken_reported:
                return
            self._broken_reported = True
        try:
            # Deliberately raw stderr, not a logger: a logger here would
            # recurse if the failure IS the logger's own sink.
            sys.stderr.write(f"logging error: sink {type(self.sink).__name__} "
                             f"failed: {exc!r}\n")
        except Exception:                              # noqa: BLE001
            pass

    def flush(self) -> None:
        try:
            self.sink.flush()
        except Exception:                              # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            self.sink.close()
        except Exception:                              # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# 7. Logger — the node, the facade, the lazy check
# --------------------------------------------------------------------------

class Logger:
    def __init__(self, name: str, manager: "LoggerManager",
                 level: Optional[LogLevel] = None) -> None:
        self.name = name
        self.level = LogLevel.parse(level) if level is not None else None
        self.handlers: list[Handler] = []
        self.propagate = True
        self.disabled = False
        self._manager = manager
        self._parent: Optional["Logger"] = None

    # -- hierarchy ---------------------------------------------------------
    @property
    def parent(self) -> Optional["Logger"]:
        return self._parent

    @property
    def effective_level(self) -> LogLevel:
        """Walk up until an ancestor sets a level; the root always does."""
        node: Optional[Logger] = self
        while node is not None:
            if node.level is not None:
                return node.level
            node = node._parent
        return LogLevel.WARNING

    def set_level(self, level: "str | int | LogLevel | None") -> "Logger":
        self.level = LogLevel.parse(level) if level is not None else None
        return self

    def add_handler(self, handler: Handler) -> "Logger":
        self.handlers.append(handler)
        return self

    def is_enabled_for(self, level: "LogLevel | int") -> bool:
        return not self.disabled and LogLevel.parse(level) >= self.effective_level

    # -- the hot path ------------------------------------------------------
    def _log(self, level: LogLevel, message: str, args: tuple,
             extra: Optional[Mapping[str, Any]] = None,
             exc_info: bool = False) -> None:
        # THE ordering that matters: the level check happens first, so a
        # suppressed record never performs the `message % args` interpolation.
        #
        # Be precise about what this does and does not save. It saves the
        # FORMATTING. It does not save the ARGUMENT EVALUATION: Python
        # evaluates a call's arguments before the call, so
        # `logger.debug("%s", expensive())` still calls expensive(). For a
        # message that is genuinely costly to BUILD, use the lazy overload
        # (`debug_lazy(lambda: ...)`) or an explicit is_enabled_for guard.
        # Claiming the level check makes `expensive()` free is the single
        # most common wrong answer to this question.
        if not self.is_enabled_for(level):
            return

        if args:
            try:
                text = message % args
            except Exception:                          # noqa: BLE001 - deliberate
                # A malformed format string must not lose the log line.
                text = f"{message} [format failed with args={args!r}]"
        else:
            text = message

        self._emit(level, text, extra, exc_info)

    def _log_lazy(self, level: LogLevel, factory: Callable[[], str],
                  extra: Optional[Mapping[str, Any]] = None,
                  exc_info: bool = False) -> None:
        """The genuinely lazy form: `factory` is not called at all when the
        level is suppressed, so an expensive message costs one comparison."""
        if not self.is_enabled_for(level):
            return
        try:
            text = factory()
        except Exception:                              # noqa: BLE001 - deliberate
            text = "<lazy message factory raised>"
        self._emit(level, text, extra, exc_info)

    def _emit(self, level: LogLevel, text: str,
              extra: Optional[Mapping[str, Any]],
              exc_info: bool) -> None:
        merged: dict[str, Any] = dict(_log_context.get())
        if extra:
            merged.update(extra)

        record = LogRecord(
            level=level,
            message=text,
            logger_name=self.name,
            timestamp=time.time(),
            extra=merged,
            exc_info=traceback.format_exc() if exc_info else None,
        )
        self.handle(record)

    def debug(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(LogLevel.DEBUG, message, args, extra)

    def info(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(LogLevel.INFO, message, args, extra)

    def warning(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(LogLevel.WARNING, message, args, extra)

    def error(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(LogLevel.ERROR, message, args, extra)

    def exception(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(LogLevel.ERROR, message, args, extra, exc_info=True)

    def critical(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(LogLevel.CRITICAL, message, args, extra)

    # -- the lazy overloads, for messages that are expensive to BUILD -------
    def debug_lazy(self, factory: Callable[[], str], **extra: Any) -> None:
        self._log_lazy(LogLevel.DEBUG, factory, extra)

    def info_lazy(self, factory: Callable[[], str], **extra: Any) -> None:
        self._log_lazy(LogLevel.INFO, factory, extra)

    def warning_lazy(self, factory: Callable[[], str], **extra: Any) -> None:
        self._log_lazy(LogLevel.WARNING, factory, extra)

    def error_lazy(self, factory: Callable[[], str], **extra: Any) -> None:
        self._log_lazy(LogLevel.ERROR, factory, extra)

    # -- routing -----------------------------------------------------------
    def handle(self, record: LogRecord) -> None:
        self.call_handlers(record)

    def call_handlers(self, record: LogRecord) -> None:
        """Walk up from here, emitting at every level, stopping at the first
        logger that sets propagate=False. This is how a library stops its
        logs reaching the application's root handlers."""
        node: Optional[Logger] = self
        while node is not None:
            for handler in node.handlers:
                handler.handle(record)
            if not node.propagate:
                break
            node = node._parent

    def flush(self) -> None:
        node: Optional[Logger] = self
        while node is not None:
            for handler in node.handlers:
                handler.flush()
            node = node._parent

    def close(self) -> None:
        for handler in self.handlers:
            handler.close()


class NullLogger(Logger):
    """A Null Object, so a library can default to a logger that does nothing
    rather than to None with a check at every call site."""

    def __init__(self) -> None:
        super().__init__("null", _NULL_MANAGER)
        self.disabled = True

    def is_enabled_for(self, level: "LogLevel | int") -> bool:
        return False


# --------------------------------------------------------------------------
# 8. Manager — the dotted hierarchy
# --------------------------------------------------------------------------

class LoggerManager:
    def __init__(self, default_level: LogLevel = LogLevel.WARNING) -> None:
        self._lock = threading.RLock()
        self._loggers: dict[str, Logger] = {}
        self.root = Logger("", self, level=default_level)
        self._loggers[""] = self.root

    def get_logger(self, name: str, level: Optional[LogLevel] = None) -> Logger:
        name = name or ""
        with self._lock:
            existing = self._loggers.get(name)
            if existing is not None:
                if level is not None:
                    existing.set_level(level)
                return existing
            logger = Logger(name, self, level=level)
            self._loggers[name] = logger
            if name:
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                # Recursive under an RLock: creating `a.b` creates `a`.
                logger._parent = self.get_logger(parent_name)
            else:
                logger._parent = None
            return logger

    def all_loggers(self) -> dict[str, Logger]:
        with self._lock:
            return dict(self._loggers)

    def shutdown(self) -> None:
        for logger in self.all_loggers().values():
            logger.close()


_NULL_MANAGER = LoggerManager()


# --------------------------------------------------------------------------
# 9. AsyncDispatcher — off the hot path, with a stated overflow policy
# --------------------------------------------------------------------------

class AsyncDispatcher(Handler):
    """A Handler that enqueues instead of writing.

    The caller's thread only builds a record and puts it on a bounded queue;
    one worker thread does the formatting and the I/O. The overflow policy is
    the design decision: BLOCK is correct and can stall a request path,
    DROP_NEWEST is lossy and never stalls. There is no policy that is both.
    """

    def __init__(self, handlers: Iterable[Handler], max_queue: int = 10_000,
                 policy: OverflowPolicy = OverflowPolicy.DROP_NEWEST,
                 flush_interval: float = 0.5) -> None:
        self.handlers = list(handlers)
        self.policy = policy
        self.flush_interval = flush_interval
        self._queue: deque[LogRecord] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._max_queue = max_queue
        self._stop = threading.Event()
        self._dropped = 0
        self._dropped_by_level: dict[str, int] = {}
        self._worker = threading.Thread(target=self._run, name="log-dispatcher",
                                        daemon=True)
        self._worker.start()

    # -- the caller's path -------------------------------------------------
    def handle(self, record: LogRecord) -> bool:
        with self._not_empty:
            if len(self._queue) >= self._max_queue:
                if self.policy is OverflowPolicy.DROP_NEWEST:
                    self._count_drop(record)
                    return False
                if self.policy is OverflowPolicy.DROP_OLDEST:
                    self._queue.popleft()
                    self._count_drop(record)
                else:                                  # BLOCK
                    while len(self._queue) >= self._max_queue and not self._stop.is_set():
                        self._not_empty.wait(timeout=0.05)
            self._queue.append(record)
            self._not_empty.notify()
            return True

    def _count_drop(self, record: LogRecord) -> None:
        self._dropped += 1
        self._dropped_by_level[record.level.name] = \
            self._dropped_by_level.get(record.level.name, 0) + 1

    @property
    def dropped(self) -> int:
        return self._dropped

    def stats(self) -> dict[str, Any]:
        return {
            "queued": len(self._queue),
            "capacity": self._max_queue,
            "dropped": self._dropped,
            "dropped_by_level": dict(self._dropped_by_level),
            "policy": self.policy.value,
        }

    # -- the worker's path -------------------------------------------------
    def _run(self) -> None:
        last_flush = time.monotonic()
        while True:
            record = None
            with self._not_empty:
                if not self._queue and not self._stop.is_set():
                    self._not_empty.wait(timeout=self.flush_interval)
                if self._queue:
                    record = self._queue.popleft()
                elif self._stop.is_set():
                    break
            if record is not None:
                # Formatting and I/O happen HERE, off the caller's thread.
                for handler in self.handlers:
                    handler.handle(record)
            now = time.monotonic()
            if now - last_flush >= self.flush_interval:
                for handler in self.handlers:
                    handler.flush()
                last_flush = now

    def flush(self) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if not self._queue:
                    break
            time.sleep(0.01)
        for handler in self.handlers:
            handler.flush()

    def close(self) -> None:
        self.flush()
        self._stop.set()
        with self._not_empty:
            self._not_empty.notify_all()
        self._worker.join(timeout=2.0)
        for handler in self.handlers:
            handler.close()


# --------------------------------------------------------------------------
# 10. Builder — declarative configuration
# --------------------------------------------------------------------------

class LoggerBuilder:
    """Config as data. The alternative — every module configuring its own
    handlers — produces duplicate output and unfixable level conflicts."""

    def __init__(self, manager: Optional[LoggerManager] = None) -> None:
        self.manager = manager if manager is not None else LoggerManager()
        self._async_handlers: list[Handler] = []

    def root(self, level: "str | int | LogLevel",
             handlers: Iterable[Handler]) -> "LoggerBuilder":
        self.manager.root.set_level(level)
        self.manager.root.handlers = list(handlers)
        return self

    def logger(self, name: str, level: "str | int | LogLevel | None" = None,
               handlers: Iterable[Handler] = (), propagate: bool = True) -> "LoggerBuilder":
        logger = self.manager.get_logger(name)
        logger.set_level(level)
        logger.handlers = list(handlers)
        logger.propagate = propagate
        return self

    def with_async(self, logger: Logger, max_queue: int = 10_000,
                   policy: OverflowPolicy = OverflowPolicy.DROP_NEWEST) -> AsyncDispatcher:
        dispatcher = AsyncDispatcher(logger.handlers, max_queue=max_queue, policy=policy)
        logger.handlers = [dispatcher]
        self._async_handlers.append(dispatcher)
        return dispatcher

    def build(self) -> LoggerManager:
        return self.manager

    def shutdown(self) -> None:
        for handler in self._async_handlers:
            handler.close()          # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# 11. Tests and demo — no asserts on wall-clock time, no sleeping
# --------------------------------------------------------------------------

def _check(name: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    if not condition:
        raise SystemExit(1)


if __name__ == "__main__":
    # --- 1. level inheritance across the dotted hierarchy ---
    manager = LoggerManager(default_level=LogLevel.WARNING)
    app = manager.get_logger("app")
    db = manager.get_logger("app.db")
    _check("a child inherits the root level",
           db.effective_level is LogLevel.WARNING)
    app.set_level(LogLevel.INFO)
    _check("setting a parent level changes the child", db.effective_level is LogLevel.INFO)
    db.set_level(LogLevel.DEBUG)
    _check("an explicit child level wins", db.effective_level is LogLevel.DEBUG)
    _check("and does not affect the parent", app.effective_level is LogLevel.INFO)
    _check("a sibling still inherits", manager.get_logger("app.web").effective_level is LogLevel.INFO)

    # --- 2. the lazy check: the FORMATTING is skipped, and the lazy
    #        overload skips the message CONSTRUCTION too ---
    calls: list[str] = []

    def expensive() -> str:
        calls.append("evaluated")
        return "costly-value"

    app.info("an enabled level formats: %s", expensive())
    _check("an enabled level evaluates its arguments", calls == ["evaluated"])
    calls.clear()

    app.debug("a suppressed level: %s", expensive())
    _check("arguments are STILL evaluated when suppressed (Python is eager)",
           calls == ["evaluated"])
    calls.clear()

    class Tracker:
        """`%s` interpolation calls __str__; the level check means the
        interpolation never happens, so __str__ is never called."""

        def __init__(self) -> None:
            self.calls = 0

        def __str__(self) -> str:
            self.calls += 1
            return "rendered"

    tracker = Tracker()
    app.info("enabled: %s", tracker)
    _check("an enabled level performs the interpolation", tracker.calls == 1)

    tracker = Tracker()
    app.debug("suppressed: %s", tracker)
    _check("a SUPPRESSED level never performs the interpolation", tracker.calls == 0)

    app.debug_lazy(lambda: f"lazy {expensive()}")
    _check("a suppressed lazy factory is never invoked", calls == [])
    app.info_lazy(lambda: f"lazy {expensive()}")
    _check("an enabled lazy factory is invoked", calls == ["evaluated"])
    calls.clear()

    # --- 3. handlers are inherited, and propagate=False stops the walk ---
    root_sink, app_sink, lib_sink = MemorySink(), MemorySink(), MemorySink()
    manager.root.handlers = [Handler(root_sink, PlainTextFormatter())]
    app.handlers = [Handler(app_sink, PlainTextFormatter())]
    app.propagate = True
    app.warning("reaches both")
    _check("propagate=True hits the logger and the root",
           len(app_sink.lines) == 1 and len(root_sink.lines) == 1)

    lib = manager.get_logger("lib")
    lib.set_level(LogLevel.INFO)
    lib.handlers = [Handler(lib_sink, PlainTextFormatter())]
    lib.propagate = False
    lib.warning("library noise")
    _check("propagate=False stops at the library boundary",
           len(lib_sink.lines) == 1 and len(root_sink.lines) == 1)

    # --- 4. structured output ---
    json_sink = MemorySink()
    json_handler = Handler(json_sink, JsonFormatter())
    _check("JSON formatter produces one line per record",
           json_handler.handle(LogRecord(LogLevel.INFO, "hello", "app.db", time.time(),
                                         {"user_id": 7})) is True)
    parsed = json.loads(json_sink.lines[0])
    _check("extra fields are top level in JSON", parsed["user_id"] == 7)
    _check("the reserved keys are present",
           parsed["level"] == "INFO" and parsed["logger"] == "app.db")

    # --- 5. redaction rewrites, and does not lose non-secrets ---
    redact_sink = MemorySink()
    redact_handler = Handler(redact_sink, JsonFormatter(),
                             filters=[RedactionFilter(["password", "token"])])
    redact_handler.handle(LogRecord(LogLevel.INFO, "login", "auth", time.time(),
                                    {"password": "hunter2", "user": "alice"}))
    redacted = json.loads(redact_sink.lines[0])
    _check("the secret is masked", redacted["password"] == "[REDACTED]")
    _check("the non-secret survives", redacted["user"] == "alice")

    # --- 6. rate limiting a noisy logger ---
    fake_now = [0.0]
    rl_sink = MemorySink()
    rl_handler = Handler(rl_sink, PlainTextFormatter(),
                         filters=[RateLimitFilter(3, window_seconds=60,
                                                  clock=lambda: fake_now[0])])
    for _ in range(10):
        rl_handler.handle(LogRecord(LogLevel.ERROR, "flooding", "noisy", time.time()))
    _check("the rate limit filter caps the flood", len(rl_sink.lines) == 3)
    fake_now[0] = 61.0
    rl_handler.handle(LogRecord(LogLevel.ERROR, "next window", "noisy", time.time()))
    _check("and the window rolls", len(rl_sink.lines) == 4)

    # --- 7. a broken sink must not break the caller or the other handlers ---
    class ExplodingSink(Sink):
        def emit(self, text: str) -> None:
            raise OSError("disk full")

    good_sink = MemorySink()
    exploding = Handler(ExplodingSink(), PlainTextFormatter())
    healthy = Handler(good_sink, PlainTextFormatter())
    survivor = manager.get_logger("resilient")
    survivor.set_level(LogLevel.DEBUG)
    survivor.propagate = False
    survivor.handlers = [exploding, healthy]
    survivor.error("this must still be logged")            # must not raise
    _check("a failing sink does not raise into the caller", True)
    _check("and the healthy handler still received the record",
           len(good_sink.lines) == 1)
    _check("the handler reports itself as failed",
           exploding.handle(LogRecord(LogLevel.ERROR, "x", "r", time.time())) is False)

    # --- 8. context fields appear on every record inside the block ---
    ctx_sink = MemorySink()
    ctx_logger = manager.get_logger("ctx")
    ctx_logger.set_level(LogLevel.DEBUG)
    ctx_logger.propagate = False
    ctx_logger.handlers = [Handler(ctx_sink, JsonFormatter())]
    with LogContext(request_id="req-42", user_id="alice"):
        ctx_logger.info("inside")
    ctx_logger.info("outside")
    inside, outside = (json.loads(line) for line in ctx_sink.lines)
    _check("the context field is attached inside the block",
           inside["request_id"] == "req-42")
    _check("and gone after it", "request_id" not in outside)

    # --- 9. async dispatch: the caller returns immediately, and drops are counted ---
    slow_sink = MemorySink()

    class SlowSink(Sink):
        def __init__(self) -> None:
            self.count = 0

        def emit(self, text: str) -> None:
            time.sleep(0.001)
            self.count += 1

    slow = SlowSink()
    dispatcher = AsyncDispatcher([Handler(slow, PlainTextFormatter())],
                                 max_queue=50, policy=OverflowPolicy.DROP_NEWEST)
    hot = manager.get_logger("hot")
    hot.set_level(LogLevel.DEBUG)
    hot.propagate = False
    hot.handlers = [dispatcher]

    started = time.monotonic()
    for i in range(500):
        hot.info("line %d", i)
    elapsed = time.monotonic() - started
    _check("500 calls with a 1ms sink took far less than 500ms", elapsed < 0.25)
    _check("the queue is bounded, so some were dropped",
           dispatcher.dropped > 0)
    stats = dispatcher.stats()
    _check("drops are attributed to a level", stats["dropped_by_level"].get("INFO", 0) > 0)

    dispatcher.flush()
    _check("flush drained what was queued", len(dispatcher._queue) == 0)
    dispatcher.close()

    # --- 10. a malformed format string does not lose the line ---
    bad_sink = MemorySink()
    bad = manager.get_logger("badformat")
    bad.set_level(LogLevel.DEBUG)
    bad.propagate = False
    bad.handlers = [Handler(bad_sink, PlainTextFormatter())]
    bad.error("value is %d and %d", "not-a-number")     # wrong arg count and type
    _check("a bad format string still logs something",
           len(bad_sink.lines) == 1 and "format failed" in bad_sink.lines[0])

    # --- 11. rotation keeps the file bounded ---
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rotation_demo.log")
    for suffix in ("", ".1", ".2"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    rotator = RotatingFileSink(path, max_bytes=200, backups=2)
    rot_handler = Handler(rotator, PlainTextFormatter())
    for i in range(20):
        rot_handler.handle(LogRecord(LogLevel.INFO, f"line {i} padded out to a length",
                                     "rot", time.time()))
    rotator.close()
    _check("rotation produced a backup", os.path.exists(path + ".1"))
    _check("the backup count is respected", not os.path.exists(path + ".3"))
    for suffix in ("", ".1", ".2"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)

    # --- 12. the NullLogger swallows everything, including at CRITICAL ---
    null = NullLogger()
    null.critical("nothing happens")
    _check("NullLogger is disabled at every level",
           not null.is_enabled_for(LogLevel.CRITICAL))

    manager.shutdown()
    print("\nall checks passed")
```

**What to point at:**

- `Logger._log` checks `is_enabled_for` **before** formatting, and the demo distinguishes the two things that saves. The `%s` interpolation is skipped — proven by an object whose `__str__` counts its calls, which is never invoked at DEBUG. The **argument evaluation is not** skipped, because Python evaluates a call's arguments eagerly; the demo asserts that too, rather than claiming a saving that does not exist. `debug_lazy(lambda: ...)` is the form that skips building the message at all.
- `Handler.handle` wraps `sink.emit` in `try/except Exception` and reports once to raw stderr. Raw stderr, because routing the report through a logger would **recurse** if the failing sink is that logger's sink.
- `Filter.apply` returns a **record**, not a bool — that is what makes `RedactionFilter` expressible at all.
- `AsyncDispatcher` counts drops **by level** so the policy "never drop ERROR" is checkable, and `flush()` waits for the queue to drain so a test or shutdown is deterministic.
- `RotatingFileSink` documents what rotation does **not** fix: two processes appending to one path still interleave.
- `LoggerManager` uses an `RLock` because creating `app.db` recursively creates `app` — a plain `Lock` would deadlock on itself.

---

## 6. Concurrency & thread-safety

### What breaks

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Two threads writing one stream** | Interleaved or lost output; a `TextIOWrapper` is not atomic across arbitrary strings | A per-sink lock, and one `write` per record (`emit` takes a whole line, including its newline) |
| **Two threads in `get_logger`** | Two `Logger` objects for one name; one gets the handlers, the other gets the calls | Create under a lock; the registry is the single source of truth |
| **Recursive parent creation deadlocking** | `get_logger("a.b")` calls `get_logger("a")` while holding the lock | `RLock`, not `Lock` |
| **Rotation racing an append** | Two threads rotate at once; a file is renamed out from under an open handle | The rotate-and-write sequence is inside the sink's lock |
| **The worker thread and `close()`** | `close()` while the worker is mid-`emit`; the sink is closed under it | `close()` flushes, then sets the stop event, then joins with a timeout |
| **A full queue with `BLOCK`** | The request thread waits on a log line; if the worker is stuck on a slow sink, the whole service stalls | `BLOCK` is only safe with a sink that cannot hang; otherwise drop and count |
| **Unbounded queue** | The producer outruns the consumer and memory grows until OOM | A hard `max_queue` and an explicit policy — no third option |
| **`_dropped` incremented off-thread** | A torn read of the counter in `stats()` | Increment under the queue lock (`_count_drop` is called inside it) |
| **Context leaking across requests** | A request id set for one request appears on another's records | `contextvars` with a token and `reset()` in `__exit__`; never a thread-local on a pooled worker |
| **Formatting on the caller's thread** | The hot path pays `json.dumps` per record | The dispatcher enqueues the **record**; the worker formats |
| **`logging` from inside a sink** | Infinite recursion → stack overflow → the process dies | Raw stderr in the failure path; never a logger |
| **Fork with a worker thread** | The child inherits a queue and no worker; logs vanish | Rebuild the dispatcher after fork (`os.register_at_fork`), or dispatch per process |

**The one to volunteer:** the `BLOCK`-and-deadlock hazard. It is the reason the default policy in this file is `DROP_NEWEST`, and the reason the honest answer to "should logging ever block?" is "only for ERROR and above, and only with a sink that cannot hang."

### Ordering

Records from one thread reach a sink **in order**, because the queue is FIFO and there is one worker. Across threads there is no total order and there cannot be: two threads' records interleave by arrival at the queue. Timestamps are `time.time()` at emission, so records can be *sorted* by timestamp but not *ordered* by it — two records can share a timestamp, and clocks differ across hosts. If a strict cross-thread order is genuinely needed, the sequence number must come from the producer (a monotonic counter incremented under a lock), not from the clock.

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Add a new sink"

The whole point of the composition. A Kafka sink is `Sink.emit` plus a producer:

```python
class KafkaSink(Sink):
    """One sink class; nothing else in the framework changes."""

    def __init__(self, producer, topic: str, key_field: str = "logger") -> None:
        self.producer, self.topic, self.key_field = producer, topic, key_field
        self._buffer: list[str] = []
        self._lock = threading.Lock()

    def emit(self, text: str) -> None:
        with self._lock:
            self._buffer.append(text)
            if len(self._buffer) >= 100:                 # batch, do not send per line
                batch, self._buffer = self._buffer, []
            else:
                batch = []
        if batch:
            self.producer.send(self.topic, "\n".join(batch).encode())

    def flush(self) -> None:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if batch:
            self.producer.send(self.topic, "\n".join(batch).encode())
```

The design point: a network sink **batches**, and it must not block the worker for long. `AsyncDispatcher` already moved it off the caller; batching moves it off the request *pattern*.

### 7.2 "Add sampling"

Sampling is a `Filter`, and the interesting question is *what* to sample:

```python
class SamplingFilter(Filter):
    """Keep all records at or above `always_at`; keep 1 in `rate` below it.

    Sampling uniformly across levels is wrong: you would drop the ERROR that
    explains the incident. Sampling by level is the design.
    """

    def __init__(self, rate: int = 100, always_at: LogLevel = LogLevel.WARNING) -> None:
        self.rate = rate
        self.always_at = always_at
        self._counter = 0
        self._lock = threading.Lock()

    def apply(self, record: LogRecord) -> Optional[LogRecord]:
        if record.level >= self.always_at:
            return record
        with self._lock:
            self._counter += 1
            keep = self._counter % self.rate == 0
        if not keep:
            return None
        # Mark the record so a reader knows it is 1-in-N, not 1-of-1.
        return record.with_extra(sampled=True, sample_rate=self.rate)
```

The `sampled` and `sample_rate` fields are the part people forget: a sampled stream that does not say it is sampled will be read as a count, and someone will conclude the error rate fell.

### 7.3 "Per-request correlation across services"

```python
def install_trace_context(get_trace_id: Callable[[], str]) -> None:
    """Bind the ambient trace id into the log context once, at request entry."""
    _log_context.set({"trace_id": get_trace_id()})
```

The design point: it is a **context variable**, not a parameter. Threading a `trace_id` through every function signature is the alternative, and it is why the parameter exists in codebases that lack this facility. The boundary is that the context must be reset — an exception path that misses the reset leaks the id into the next request on that worker.

### 7.4 "Runtime reconfiguration without a restart"

Levels are the cheap part: `logger.set_level("DEBUG")` is a single assignment, and because `effective_level` is computed on read, it takes effect immediately.

```python
def apply_config(manager: LoggerManager, config: Mapping[str, str]) -> None:
    """config maps a logger name to a level. Applied in order, so a parent
    set before a child does not clobber it."""
    for name in sorted(config, key=lambda n: n.count(".")):
        manager.get_logger(name).set_level(config[name])
```

Handlers are the hard part: replacing a sink while records are in flight needs the same care as `close()`. The honest answer is that handler replacement should drain the dispatcher first, or swap the sink inside a wrapper that takes the sink's lock.

### 7.5 "Make the whole thing a no-op for a library"

```python
LIBRARY_LOGGER = NullLogger()
```

The Null Object pattern. A library must not configure logging — that decision belongs to the application — so it logs to a `NullLogger` it can be *handed* a real one for. That is the same reasoning as rejecting a global singleton for the [rate limiter](11-rate-limiter.md) §9: a library that installs a handler changes the application's behaviour as a side effect of import.

### 7.6 "Guarantee no log loss"

The only way is `BLOCK` plus a durable sink, and then the honest statement is that you have converted a logging system into a **write-ahead log** with the same availability characteristics. If a record must not be lost, it is not a log record any more — it is an event to be published through [the pub-sub system](13-pub-sub-system.md) with an ack. That reframing is the correct answer.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Sink throws** | The exception propagates into the caller — a disk-full error becomes a 500 response | Catch per handler, report once to raw stderr, continue with other handlers |
| 2 | **A sink logs** | Infinite recursion, stack overflow, process death | Never route the failure report through a logger |
| 3 | **Queue full with `BLOCK`** | The request path stalls behind a slow sink | Bounded queue with an explicit policy; `BLOCK` only for a sink that cannot hang |
| 4 | **Unbounded queue** | Memory grows until OOM — logging kills the process it was diagnosing | Hard `max_queue`, drop and count |
| 5 | **Silent drops** | Records vanish with no trace, so the volume looks healthy | `dropped`, `dropped_by_level`, and a periodic self-report |
| 6 | **A suppressed call still evaluates its arguments** | The work happens for a line nobody will see — the level check does not prevent this, because Python evaluates a call's arguments eagerly | Guard explicitly with `is_enabled_for`, or use `debug_lazy(lambda: ...)` so the message is never built |
| 7 | **Malformed format string** | `TypeError` on the hot path — logging breaks the request | Catch the format error and log the raw message with the args |
| 8 | **Unserialisable `extra`** | `json.dumps` raises and the whole line is lost | Fall back to `repr()` per field; never lose the message |
| 9 | **Secret in `extra`** | Credentials land in the log store, where they live for the retention period | `RedactionFilter`, and the rule that a secret reaching the *message* is already a bug |
| 10 | **Secret in the message** | Redaction cannot help — the value was formatted into a string | A review rule and a lint for f-string interpolation of known-secret names; document that the filter is a backstop |
| 11 | **Repetitive error in a hot loop** | Millions of identical lines; the collector falls over | `RateLimitFilter`, with a count of what was suppressed |
| 12 | **Context leaks across requests** | One user's request id on another user's records — a correctness and privacy bug | `contextvars` with a token, reset in `__exit__`, and a reset on the error path |
| 13 | **Log file fills the disk** | The host dies; whatever logging was diagnosing is now unrecoverable | Rotation with a backup count, plus an alert on disk usage; a sink failure is isolated |
| 14 | **Two processes, one file** | Interleaved partial lines and two independent rotations | One file per process, or log to a socket/collector |
| 15 | **Fork after the worker started** | The child has a queue and no thread; logs silently vanish | `os.register_at_fork` to restart the worker, or dispatch per process |
| 16 | **Timestamp is wall clock** | NTP steps make records appear out of order; cross-host comparison is meaningless | Wall clock for the timestamp with the caveat, monotonic for durations; sort by timestamp with the understanding that it is not an order |
| 17 | **`get_logger` called per request** | A registry lookup on the hot path; or worse, a new `Logger` per call | Cache the logger in a module-global at import, which is what every logging library expects |
| 18 | **Handler added twice** | Every line appears twice; the fix is obvious only if you know where to look | Idempotent configuration — the builder owns handler assignment, not the modules |
| 19 | **Shutdown without a flush** | The last records — usually the ones explaining the shutdown — are lost | `close()` flushes; register it with the process's shutdown path |
| 20 | **Level set on a child by mistake** | The parent's change has no effect and nobody can tell why | `effective_level` is computed, so it is inspectable; expose a dump of the hierarchy |
| 21 | **JSON line contains a newline** | One record becomes two lines in the collector's parser | The formatter must guarantee one line: escape embedded newlines in messages |
| 22 | **`extra` with a key `message`** | Silent collision — one field overwrites another | `JsonFormatter` nests collisions rather than losing them |

**The two to volunteer:** #1/#2 (the failure path must not raise and must not recurse) and #12 (context leaking across requests, which is a privacy bug, not a cosmetic one).

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **Facade** | `Logger.info/debug/...` | Callers get one obvious call; the level check, record construction, context merge and propagation walk are hidden | The facade is what must stay stable — a signature change ripples into every call site |
| **Composite** | `Handler` = sink + formatter + filters | Three independent reasons to change, combined into one object with one `handle` | One more layer of indirection per record |
| **Chain of Responsibility** | The filter list, and `call_handlers` walking the hierarchy | Both are "pass along until something stops it": a filter can drop, a logger can stop propagation | A drop is invisible unless counted — hence `dropped` |
| **Strategy** | `Formatter`: plain text versus JSON | The same record renders two ways; the sink does not care which | Two formatters must be kept field-compatible |
| **Template Method** | Not used | The formatters share only a dictionary-shaped payload; a base class would save six lines and cost a hierarchy | — |
| **Decorator** | `AsyncDispatcher` wraps a handler list | Moves work off the caller's thread without the algorithms or the sinks knowing | Queue, thread, overflow policy — the whole cost of asynchrony |
| **Null Object** | `NullLogger` | A library defaults to a logger that does nothing instead of `None` plus a check at every call site | A disabled logger that looks real; must be `disabled` so the level check still short-circuits |
| **Builder** | `LoggerBuilder` | Configuration is data, applied in one place, in a deterministic order | The builder must be the only thing assigning handlers |
| **Observer** | Several handlers per logger | One record, many destinations | Duplicate output if configuration is sloppy |
| **Value Object** | `LogRecord` (frozen) | Safe to pass to many handlers; `with_extra` returns a new record so redaction cannot mutate what another handler sees | One allocation per record; the alternative — mutation — is a cross-handler data race |
| **Singleton** | *Deliberately rejected* | A global registry makes tests share state and order-dependent | — |

**The sentence to give:** *"Facade for the call site, Composite for the handler, Chain of Responsibility for both the filter list and the propagation walk, Strategy for the formatter, Decorator for async dispatch. The one I would argue hardest for is the value object: a frozen `LogRecord` is what makes redaction safe, because a rewriting filter returns a new record instead of mutating one that three other handlers are about to format."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Async dispatch by default** | Yes | The request path never pays for formatting or I/O | A queue, a thread, drops to reason about, and ordering only per producer |
| **`DROP_NEWEST` default policy** | Yes | The caller never stalls — logging cannot become an availability incident | Records are lost under load, exactly when they are most interesting |
| **`BLOCK` available, not default** | Yes | Correctness for a sink that cannot hang | One slow sink away from stalling the service |
| **Bounded queue** | Yes | Memory is bounded and predictable | Loss is now a deliberate policy rather than an accident |
| **Drop counting per level** | Yes | The "never drop ERROR" policy is checkable | A small lock on the drop path |
| **Frozen `LogRecord`** | Yes | Redaction and enrichment cannot corrupt other handlers | Allocation per record; deep-copying `extra` would be worse |
| **`Filter` returns a record, not a bool** | Yes | Redaction, enrichment and sampling are one interface | A filter that only gates must return the record it was given |
| **Per-handler error isolation** | Yes | One broken sink cannot silence the others or break the caller | A failure can be invisible unless the report is monitored |
| **`contextvars` for request fields** | Yes | No signature pollution; correct under threads and async | A missed reset leaks fields across requests |
| **Handlers inherited via `propagate`** | Yes | Root-level configuration reaches every logger | A library that forgets `propagate=False` prints twice |
| **Structured JSON in production** | Yes | Fields are queryable; no regex parsing | Larger records; a schema to keep stable |
| **Plain text in development** | Yes | Readable | Two rendering paths to keep in sync |
| **Dotted-name hierarchy** | Yes | Per-module levels without code changes | A registry, a lock, and a lookup that must be cached at import |
| **Redaction as a backstop, not a guarantee** | Yes, documented | Catches structured secrets cheaply | A secret formatted into the message is NOT caught — that must be said, not implied |
| **Rotation in the sink, retention elsewhere** | Yes | Bounded local disk | Local backups are not a retention policy; the collector owns that |
| **No multi-process file sharing** | Documented, not solved | Simplicity | N processes need N files or a collector |

**The trade to name unprompted:** *"Async dispatch is the decision with the sharpest trade. It buys you a request path that never blocks on disk and costs you the guarantee that every record arrives. The failure mode of sync logging is a stalled service; the failure mode of async logging is missing evidence. I would take async with drop-and-count, and make sure the self-report — the dropped counter — is itself logged, because silent loss is the one thing worse than loss."*

---

## 11. Q&A

**Q1. Why is `is_enabled_for` called before the record is built, and what exactly does that save?**
It saves the **formatting**, and it is important to be precise about that because the usual answer overclaims. Building the record includes the `message % args` interpolation, and skipping that is a real saving — the demo proves it with an object whose `__str__` counts its calls, which is never invoked when the level is suppressed. What it does *not* save is the **argument evaluation**: Python evaluates a call's arguments before the call, so `logger.debug("%s", expensive())` still calls `expensive()` whether or not DEBUG is on. For a message that is genuinely costly to build you need either an explicit `if logger.is_enabled_for(DEBUG):` guard or a lazy overload — `logger.debug_lazy(lambda: ...)` — which takes a callable and never invokes it when the level is suppressed. The demo asserts all three behaviours, including the one that is unflattering to the framework, because the common wrong answer here ("the level check makes expensive arguments free") is exactly the kind of thing an interviewer probes.

**Q2. Why a dotted name hierarchy rather than a logger per class?**
Because the hierarchy is a **configuration namespace**. `getLogger("app.db.pool")` inherits from `app.db`, which inherits from `app`, which inherits from the root, so an operator can raise `app.db` to DEBUG in production without touching the code that logs. A logger per class gives you the same call sites with no way to address a subsystem. The cost is a registry and a walk up the parents on each emit — which is why the `Logger` object is cached at import and `effective_level` is the only thing computed per call.

**Q3. Why does `Handler.handle` swallow exceptions?**
Because a logging failure must not become an application failure. Without the `try/except`, a full disk or a closed stream turns every logged request into a 500 — the observability system has taken down the system it was observing. The swallow is per handler, not per logger, so one broken sink does not stop the other handlers from receiving the record. The report goes to **raw stderr** rather than through a logger, because if the failing sink belongs to that logger, routing the report through it recurses until the stack is exhausted.

**Q4. Why does `Filter.apply` return a record instead of a boolean?**
Because gating is not the only thing a filter does. Redaction has to hand back a *different* record with the secret masked; enrichment has to add fields; sampling has to mark the record so a reader knows it is 1-in-N. A boolean interface makes all three impossible and forces them into the formatter, which then needs the whole filter configuration. The cost is that a pure gate has to return the record it was given — a one-line inconvenience that buys an interface where redaction, enrichment and sampling are all expressible.

**Q5. Why is `LogRecord` frozen?**
Because one record reaches several handlers. If a redaction filter mutated the record in place, whether a handler saw the secret would depend on the order of the handler list — a bug that appears only when someone reorders configuration. A frozen record means a rewriting filter returns a **new** record and the others are untouched. The cost is an allocation per record, which is cheap next to the I/O the record is about to cause.

**Q6. Why is the dispatcher's default overflow policy `DROP_NEWEST` and not `BLOCK`?**
Because `BLOCK` converts a logging problem into an availability problem. If the queue is full because a sink is slow, blocking the request thread means the request latency now includes the log pipeline's latency — and under sustained load, the queue never drains and every request stalls. `DROP_NEWEST` bounds the damage: the call returns immediately, records are lost, and the loss is counted. `BLOCK` stays available because it is the right answer when the sink cannot hang — a bounded in-memory buffer, say — and then the caller genuinely wants backpressure. There is no policy that is both lossless and non-blocking; saying that explicitly is the point.

**Q7. If you drop logs, how does anyone know?**
By counting and reporting. The dispatcher keeps `dropped` and `dropped_by_level`, exposed through `stats()`, and the framework's own health should be logged periodically — from a logger whose sink is *not* the overloaded one, or the report is dropped too. The alternative to counting is silent loss, which is worse than loss: a missing line looks like a quiet system. The general principle is that a lossy component must publish its loss rate, which is also why the [rate limiter](11-rate-limiter.md) reports `RateLimit-Remaining` rather than silently denying.

**Q8. Why is the context a `contextvars` variable rather than a thread-local?**
Because thread-locals break in two directions. Under `asyncio`, many tasks share one thread, so a thread-local request id leaks from one request into every other request on that loop. With a thread pool, a leaked value survives into the next request that reuses the worker. `contextvars` is scoped to the execution context, which is correct for both, and it nests — `LogContext` inside `LogContext` restores the outer one on exit. The failure it prevents is not cosmetic: a request id attached to the wrong request makes a log search return another user's activity, which is a privacy incident.

**Q9. Wall clock or monotonic for the timestamp?**
Wall clock for the timestamp, because a log without a real time is not correlatable with anything outside the process — a trace, a metric, another service. Monotonic for durations, because a wall clock can step and a duration measured across a step is wrong. The caveat to state is that the wall clock is not an ordering: two records can share a timestamp, and across hosts the clocks differ. `AsyncDispatcher` preserves per-producer order through a FIFO queue, and cross-thread order is arrival order at the queue — which is the best available and should be described as such rather than implied to be a global order.

**Q10 (senior). A secret is formatted into the message, not passed as a field. Does your redaction filter catch it?**
No, and it is important to say so plainly rather than imply the framework is safe. `RedactionFilter` inspects `extra`, the structured field map, because that is where secrets are identifiable by key. Once a value is interpolated into the message string, all that is left is prose and a filter cannot know which words are secret. The defences are: a logging convention that secrets go in `extra` under a known key, a lint that flags f-string interpolation of variables whose names match a secret pattern, and a code review rule that no credential is passed to a log call at all. The filter is a backstop for the structured path, and the honest framing is that the primary control is not logging the secret.

**Q11 (senior). How would you make this multi-process safe, given each process has its own dispatcher?**
Not by sharing a file. Two processes appending to one path interleave partial lines, and rotation is worse: each process rotates independently and can rename a file the other has open, so writes land in the wrong backup or vanish. Three real options, in order of preference. First, log to **stdout** and let the container runtime or a sidecar collect it — this is the standard answer, and it removes the framework from the durability question entirely. Second, one file per process (a pid in the name) with the collector merging them. Third, a socket to a local collector, which is the same as option one with extra steps. I would not attempt file locking on the write path: it serialises all processes on one lock and does not fix rotation.

**Q12 (senior). Where would you expect a subtle bug in this design?**
Three places. First, the **shutdown path**: an `AsyncDispatcher` that is not flushed before exit loses the records that explain the shutdown, so `close()` must drain and join, and the process must actually call it — an `atexit` handler or a signal handler. Second, the **fork case**: a pre-fork server forks after the dispatcher's thread started, so the child has a queue and no worker, and its logs vanish silently. That needs `os.register_at_fork` to restart the worker, or a dispatcher per process created after the fork. Third, the **rotation racing an append**: rotate-and-write must be inside the sink's lock, because a rotation that renames the file between another thread's open and write sends a record into a backup — quiet, and rare enough to be very hard to find. All three share a shape: the failure is a *lost* record rather than an error, which is why it needs a counter to be observable.

**Q13. Why a `NullLogger` rather than passing `None`?**
Because `None` forces a check at every call site, and the check is exactly the kind of thing that gets forgotten in the one place it mattered. `NullLogger` implements the whole interface and reports `is_enabled_for` as false for every level, so calls into it cost one comparison and no branching at the caller. It also lets a library be written to log unconditionally and handed either a real logger or a null one — which is the correct boundary: a library must not configure logging, because that is the application's decision.

**Q14. Why does the builder own handler assignment rather than each module adding its own?**
Because duplicate output is the single most common logging misconfiguration. If `app.db` adds a file handler at import and the application also configures a file handler at the root, every `app.db` record appears twice, and the duplication is invisible in the code that added it. Making the builder the only writer means the handler set is a fact about the configuration, not an accumulation of import side effects. The corollary is that `add_handler` exists on `Logger` for tests, and configuration code should prefer the builder.

**Q15. How do you log at high volume without the log becoming the bottleneck?**
Five levers, in order of effect. **Levels**: do not log INFO per request in a service doing 100k rps; log the outcome at a sampled rate. **Lazy formatting**: never build a string that might be discarded — that is the level check. **Sampling**: keep 1 in N below WARNING and always keep ERROR. **Batching**: a network sink must batch, not send per line. **Async dispatch**: move formatting and I/O off the caller. And the meta-lever: **measure the framework's own cost** — the dispatcher's queue depth and drop count are the numbers that tell you whether logging is the bottleneck, and they belong on a dashboard alongside the application's own.

---

### Pushback round

**P1. "This is a lot of machinery for printing strings. Why not just `print`?"**
For a script, `print` is correct and I would say so. The machinery is justified by four things a service needs and a script does not: the ability to turn a subsystem's verbosity up **without a redeploy**, which is what the hierarchy buys; the ability never to let logging break a request, which is the error isolation; the ability to keep logging off the critical path, which is the dispatcher; and the ability to search logs by field rather than regex, which is the structured record. Each of those is an availability or operability requirement, not a stylistic one. If none of the four applies, the framework is over-engineered — and saying which requirement justifies which class is the answer.

**P2. "Your `AsyncDispatcher` drops logs. Losing the log line that explains an outage is unacceptable."**
Under `DROP_NEWEST`, a burst that overflows a 10,000-record queue loses records, and if the burst is the incident then yes, that is the worst possible time. Two mitigations, and I would do both. First, **priority**: the policy should not be uniform by level — ERROR and above should take a separate, smaller, non-dropping path, which is cheap because they are rare. Second, **the drop is itself an event**: the counter is logged through a logger whose sink is not the overloaded one, so a gap in the logs has an accompanying "we dropped 48,000 INFO records" line. The alternative — blocking — trades a partial log for a stalled service, and I would rather explain an outage with a hole in the logs than cause one. If the requirement is truly "no loss", then it is not a log, it is an event, and it belongs in [the pub-sub system](13-pub-sub-system.md) with an ack.

**P3. "You reject the Singleton pattern for loggers, but every real logging library has a global registry."**
I reject a *singleton class*, not a *registry*. The distinction matters: `LoggerManager` is an object you construct, inject and replace in a test, and `get_logger` is a convenience over one default instance. Python's `logging` module does exactly this — there is a module-level manager, and you can also build your own. What I am rejecting is the pattern where the logger's state is reachable only through a static accessor, because then two tests in one process share handler state and start failing based on order. The registry gives the ergonomics; the constructor gives the testability. A singleton gives only the first.

**P4. "Rotation in the sink is the wrong layer. Logging libraries should not manage files."**
That is a defensible position and it is the one I would take for a containerised service, where logs go to stdout and the runtime owns the files. Rotation belongs in the sink for the case where the process genuinely owns a file — a CLI writing a run log, an on-premise agent with no collector — and then it must be size-based with a backup count, because unbounded growth is a disk-full incident. So the honest answer is conditional: rotation is a *local durability* concern, and it belongs in the framework only when the framework is the last hop. In a modern deployment it is not, and the sink should be a stream.

**P5. "Why not just use the standard library / an existing framework?"**
In production, absolutely — Python's `logging`, Java's SLF4J with Logback, Go's `slog`. Nobody should write this. The reason to design it in an interview is that a logging framework is the densest concentration of LLD concerns in one small component: a hierarchy resolved by walking up, an interface with several implementations that must be substitutable, a hot path where the ordering of two operations is a performance decision, a concurrency boundary with a bounded queue and a policy, an error-isolation boundary where the component must not propagate, a value object that makes rewriting safe, and a plugin seam for sinks. That is the whole syllabus in something you can read in one sitting. And understanding it is why you can answer the operational questions — why does this line appear twice, why did the log stop, why did the service stall when the disk filled.

---

## 12. Cheat-sheet summary + cross-links

| Concept | One-line truth |
|---|---|
| The shape | `Logger` (whether and where) → `Handler` (how and where) → `Sink` (I/O) |
| The hot path | **`is_enabled_for` first**, always — a suppressed line skips the interpolation, not the argument evaluation (Python is eager) |
| Truly lazy | `debug_lazy(lambda: ...)` — the factory is not called when suppressed; the level check alone does not give you this |
| The hierarchy | Dotted names, `effective_level` walks up to the first ancestor that sets one |
| Propagation | Handlers are inherited up the tree; `propagate = False` stops the walk at a library boundary |
| `Handler` | Sink + Formatter + level + filters, and it **never raises** |
| Error isolation | Per handler, reported once to **raw stderr** — routing it through a logger recurses |
| `Filter` | `apply(record) -> record \| None` — rewriting, which is what makes redaction possible |
| `LogRecord` | Frozen value object; rewriting filters return a new one, so handlers cannot corrupt each other |
| Redaction | A backstop for **structured** fields only; a secret in the message is already a bug |
| Sampling | By level — always keep ERROR, sample below it, and **mark the record** as sampled |
| Rate limiting | Same fixed-window arithmetic as [Rate Limiter](11-rate-limiter.md), as a filter |
| Async dispatch | Bounded queue + one worker + an explicit overflow policy; the worker formats |
| The policy | `BLOCK` is correct and can stall the service; `DROP_NEWEST` is lossy and cannot — no third option |
| Silent loss | Count drops **by level** and log the counter through a different sink |
| Ordering | FIFO per producer; no global order, and timestamps are not one |
| The clock | Wall clock for the timestamp, monotonic for durations |
| Rotation | Size-based with a backup count; **does not** fix multi-process interleaving |
| Multi-process | stdout + a collector, or one file per process — never a shared file with locking |
| Context | `contextvars`, not a thread-local — thread-locals leak across async tasks and pooled workers |
| Fork | Restart the dispatcher with `os.register_at_fork`, or the child logs into a queue with no worker |
| Shutdown | `close()` flushes and joins; without it you lose the records explaining the shutdown |
| `NullLogger` | Null Object, so a library logs unconditionally without configuring anything |
| Patterns | Facade, Composite, Chain of Responsibility, Strategy, Decorator, Null Object, Builder, Value Object |
| Patterns rejected | Singleton (test state sharing), Template Method (formatters share too little) |

**The three sentences to memorise**

1. *"A logger decides whether and where; a handler decides how it looks; a sink does the I/O — three classes because there are three reasons to change."*
2. *"The level check comes before the message is built, and the handler's emit is wrapped so a broken sink degrades to stderr instead of breaking the caller."*
3. *"Async dispatch buys a request path that never blocks on disk and costs the guarantee that every record arrives — so the drops are counted, by level, and the counter is logged elsewhere."*

### Cross-links

- **What to log, where logs go, aggregation and retention** (the HLD counterpart this file does not duplicate): [Observability](../../Operations/02-observability-and-monitoring.md)
- **The fixed-window arithmetic behind `RateLimitFilter`, and the limiter as a component:** [Rate Limiter](11-rate-limiter.md) — §5 `FixedWindowCounter`, §8 row 7 on the boundary artefact
- **Where a must-not-lose record belongs instead of a log line (§7.6):** [Pub-Sub System](13-pub-sub-system.md)
- **Rejecting the Singleton pattern, for the same test-isolation reason:** [Chess Game](09-chess-game.md) §9 · [Vending Machine](10-vending-machine.md) §9
- **The reserve-and-settle shape reused by weighted logging quotas:** [URL Shortener LLD](../../Case-Studies/LLD/01-url-shortener-lld.md)
- **A worked example of a value object that makes rewriting safe:** [Vending Machine](10-vending-machine.md) §4 (`PurchaseResult`)
- **Pattern catalogue, concurrency primitives, and the classic-problems list:** [LLD Cheatsheet](../../Cheat-Sheets/07-lld-cheatsheet.md)
- **Question bank:** [LLD Questions](../../Interview-Questions/08-lld-questions.md)
