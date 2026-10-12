# LLD 11 — Rate Limiter

> **Scope split — read this first.** This file owns the **class and code treatment**: the four algorithms as objects behind one interface, their state, their complexity, their thread-safety and their failure modes. The **distributed view** — where the limiter sits in a gateway, Redis + Lua, keying strategy, tiers and quotas, `429`/`Retry-After` semantics, fail-open versus fail-closed at the fleet level — is owned by [API Gateway and Rate Limiting](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md). Cross-link, do not duplicate.

> **Grounding note.** The reference repositories contain **no OOP/LLD material**; the source transcripts name the rate limiter as a topic and teach the algorithms at the HLD level (in the networking file above) without building them as classes. This file is written from first principles and is not transcript-grounded.

> **TL;DR** — Four algorithms, one interface, and the differences between them are entirely about **what state you keep per key and how much memory it costs**. A **token bucket** keeps `(tokens, last_refill)` — O(1) memory, admits bursts up to the bucket size. A **leaky bucket** keeps `(level, last_drain)` — the same O(1) memory with queue semantics, so it smooths rather than bursts. A **sliding-window log** keeps every accepted timestamp — exact, and O(limit) memory per key, which is the whole reason it is not the default. A **sliding-window counter** keeps two integers — O(1), and it is an *approximation* with a bounded error at window boundaries. The interview is won by naming that memory-versus-precision axis, by making the clock injectable so the limiter is testable without sleeping, and by stating that the real correctness problem is the **atomic read-modify-write**, not the arithmetic.

> **60-second interview answer:** "One interface — `allow(key, cost) -> Decision` — and four implementations. Token bucket for burst tolerance with O(1) state per key; leaky bucket when I want to smooth output rather than permit a burst, which is the same state with different semantics; sliding-window log when I need exactness and can pay O(limit) memory per key; sliding-window counter when I want O(1) memory and accept a small boundary error. Every one of them is a read-modify-write on per-key state, so the correctness problem is atomicity: a per-key lock in a single process, or a Lua script if the state is in Redis. The clock is injected as a protocol so the tests advance time instead of sleeping. And I would return a `Decision` carrying `limit`, `remaining` and `retry_after`, because the headers are part of the contract, not the caller's problem."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **What is being limited — requests, or some cost?** | Whether `allow` takes a `cost` (a weighted API call, LLM tokens) or is a pure counter | `cost=1` default, with the parameter present — it is the thing that is painful to add later |
| 2 | **What is the limit expressed as?** | rps, rpm, or "N per window" | "N requests per window" generalises; store `limit` + `window_seconds` |
| 3 | **Burst tolerance: is a burst allowed?** | Token bucket (yes, up to capacity) versus leaky bucket (no) | Yes, bursty clients are normal — token bucket |
| 4 | **Is the limit exact or approximate?** | Sliding-window log (exact, O(limit) memory) versus counter (approximate, O(1)) | Approximate is fine; say so and say the error bound |
| 5 | **Single process, or distributed?** | A lock in-process, or Redis + Lua | Start single-process here; name the Redis path as the scale answer |
| 6 | **Fail open or fail closed when the limiter's own store is down?** | Whether a limiter failure blocks traffic | **Fail open** for availability, with a conservative local fallback — cross-link the gateway file |
| 7 | **One limit, or several at once?** | Whether there is a composite/hierarchical limiter (per-user AND per-tenant AND global) | Several — that is the real-world shape |
| 8 | **What happens to a rejected request?** | Whether `Decision` carries `retry_after` | Reject with `retry_after` and `RateLimit-*` headers |
| 9 | **Memory ceiling for the key space?** | Whether per-key state needs eviction | Yes — cap keys and evict; an unbounded dict is a slow OOM |
| 10 | **Is the clock trusted?** | Whether time comes from `monotonic()` behind a protocol | Injected `Clock` protocol; monotonic in production, faked in tests |

### The defaults to assume, stated plainly

> "I'll assume: **a per-key limiter with a weighted cost, expressed as N per window, allowing bursts, single-process with a per-key lock, returning a `Decision` that carries the headers, with the clock injected**. I'll implement the four algorithms behind one interface and show the Redis Lua version as the distributed extension. Does that match what you want, or should I focus on one algorithm?"

### Non-goals

- No gateway routing, auth, or quota/plan management — that is [the gateway file](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md).
- No distributed coordination here beyond naming the Lua path and the keying decision.
- No metrics pipeline; `Decision` is the seam a metrics sink would consume.

### The four requirements that actually constrain the design

1. **The operation is a read-modify-write.** Every algorithm decays state, then decides, then mutates. Splitting decay from mutation is the race.
2. **Time must be injectable.** A limiter tested with `time.sleep` is a limiter with flaky CI.
3. **State per key must be bounded and evictable.** A dict keyed by client is an unbounded allocation controlled by an attacker.
4. **The decision must carry the numbers.** `limit`, `remaining` and `retry_after` are protocol, not cosmetics.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `Clock` (Protocol) | Port | `now() -> float` — monotonic seconds | Knowing the limit |
| `MonotonicClock` | Adapter | `time.monotonic()` | — |
| `FakeClock` | Test double | Advanceable time, so tests never sleep | — |
| `Decision` | Immutable value | `allowed`, `limit`, `remaining`, `retry_after`; `headers` | Applying the limit |
| `RateLimiter` (ABC) | Interface | `allow(key, cost) -> Decision` | Storage, time, or identity |
| `TokenBucket` | Implementation | Bucket state per key; lazy refill from elapsed time | Bursts beyond capacity |
| `LeakyBucket` | Implementation | Queue level per key; lazy drain | Admission beyond queue depth |
| `SlidingWindowLog` | Implementation | A timestamp deque per key | Bounding its own memory |
| `FixedWindowCounter` | Implementation | One counter per key per window | Smoothing the boundary |
| `SlidingWindowCounter` | Implementation | Two counters per key, weighted | Exact precision |
| `KeySpace` | Helper | Bounded dict with eviction and a `max_keys` ceiling | The limit policy |
| `CompositeLimiter` | Implementation | All sub-limiters must allow | Atomicity across sub-limiters (documented caveat) |
| `KeyedLocks` | Helper | Striped locks so unrelated keys do not contend | Fairness |
| `RateLimitExceeded` | Exception | Optional raise-style facade over `Decision` | — |

**The one-line split to say out loud:** *"A limiter is `allow(key, cost) -> Decision`. Everything else — bucket, log, counters — is an implementation detail of how much state it keeps per key."*

---

## 3. Class diagram (ASCII)

```
                        ┌───────────────────────┐
                        │    «interface»        │
                        │     RateLimiter       │
                        │───────────────────────│
                        │ +allow(key,cost)      │
                        │        -> Decision    │
                        └───────────▲───────────┘
                                    │ implements
     ┌───────────────┬──────────────┼───────────────┬──────────────────┐
     │               │              │               │                  │
┌────┴─────┐  ┌──────┴──────┐ ┌─────┴──────┐ ┌──────┴───────┐ ┌────────┴────────┐
│TokenBucket│ │ LeakyBucket │ │SlidingWin- │ │SlidingWindow │ │ FixedWindow     │
│──────────│ │─────────────│ │dowLog      │ │Counter       │ │ Counter         │
│rate      │ │rate         │ │────────────│ │──────────────│ │─────────────────│
│capacity  │ │capacity     │ │limit       │ │limit         │ │limit / window   │
│clock     │ │clock        │ │window      │ │window        │ │                 │
│──────────│ │─────────────│ │────────────│ │──────────────│ │─────────────────│
│state/key:│ │state/key:   │ │state/key:  │ │state/key:    │ │state/key:       │
│(tokens,  │ │(level,      │ │deque[float]│ │(index, cur,  │ │(index, count)   │
│ updated) │ │ updated)    │ │            │ │ prev)        │ │                 │
│O(1) mem  │ │O(1) mem     │ │O(limit) mem│ │O(1) mem      │ │O(1) mem         │
└────┬─────┘ └─────────────┘ └────────────┘ └──────────────┘ └─────────────────┘
     │
     │ uses                      ┌──────────────────────┐
     ├──────────────────────────▶│        Clock         │  «interface»
     │                           │  +now() -> float     │
     │                           └─────────▲────────────┘
     │                                     │
     │                        ┌────────────┴───────────┐
     │                        │ MonotonicClock         │
     │                        │ FakeClock (tests)      │
     │                        └────────────────────────┘
     │
     │ returns
     ▼
┌───────────────────┐        ┌──────────────────────┐
│    Decision       │        │     KeySpace         │  (bounded dict)
│───────────────────│        │──────────────────────│
│ allowed : bool    │        │ max_keys : int       │
│ limit   : int     │        │ +get_or_create(key)  │
│ remaining : int   │        │ +evict_oldest()      │
│ retry_after: float│        └──────────────────────┘
│ +headers          │
└───────────────────┘        ┌──────────────────────┐
                             │  CompositeLimiter    │
┌───────────────────┐        │──────────────────────│
│  KeyedLocks       │◀───────│ limiters: [ ... ]    │
│ (striped mutexes) │  used  │ (all must allow)     │
└───────────────────┘        └──────────────────────┘
```

**Read the diagram this way:** one interface, five implementations differing only in their per-key state, one injected clock, one value object, and two helpers (`KeySpace` for memory, `KeyedLocks` for contention).

---

## 4. Interfaces / abstract classes / enums

```python
class Clock(Protocol):
    def now(self) -> float:
        """Monotonic seconds. Never the wall clock."""
        ...


class RateLimiter(ABC):
    """One operation. Whether it is a bucket, a log or a counter is private."""

    @abstractmethod
    def allow(self, key: str, cost: int = 1) -> "Decision":
        """Decide and *consume* in one atomic step."""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: float = 0.0

    @property
    def headers(self) -> dict[str, str]:
        out = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(max(0, self.remaining)),
        }
        if not self.allowed:
            out["Retry-After"] = str(max(1, math.ceil(self.retry_after)))
        return out


class RateLimitExceeded(Exception):
    """Optional raise-style facade for callers that prefer exceptions."""
```

**Why `allow` and not `check` + `consume`:** a limiter that offers a separate `check` invites exactly the bug the whole component exists to prevent — a caller checks, does work, then consumes, and the check is stale. One method that decides and consumes atomically makes the race unrepresentable. If a caller genuinely needs a dry run (an admin UI showing remaining quota), that is `peek(key)`, which does not mutate and is explicitly not the admission path.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** Interview languages vary. In Java this is an interface with four implementations and `AtomicLong`/`synchronized`; in Go, an interface plus a mutex per limiter; in C++, `virtual` plus `std::mutex`. The **state per key is the design**; the syntax is not. Python 3.10+ for `X | None`.

```python
"""rate_limiter.py — four rate-limiting algorithms behind one interface.

Design summary
--------------
Clock          : injected port -> tests advance time, they never sleep.
Decision       : immutable, carries limit / remaining / retry_after.
RateLimiter    : allow(key, cost) -> Decision. Decide AND consume, atomically.
TokenBucket    : (tokens, updated) per key            O(1) memory, bursts.
LeakyBucket    : (level, updated) per key             O(1) memory, smoothing.
SlidingWindowLog       : deque of timestamps per key  O(limit) memory, exact.
SlidingWindowCounter   : (index, cur, prev) per key   O(1) memory, approx.
FixedWindowCounter     : (index, count) per key       O(1) memory, boundary bug.
KeySpace       : bounded dict, so the key space cannot be an attack vector.
KeyedLocks     : striped locks, so unrelated keys do not contend.
"""
from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol


# --------------------------------------------------------------------------
# 1. Clock — injected, so the limiter is testable without sleeping
# --------------------------------------------------------------------------

class Clock(Protocol):
    def now(self) -> float: ...


class MonotonicClock:
    """Use monotonic time. A wall-clock adjustment (NTP step, DST) must never
    grant or destroy a rate-limit budget."""

    def now(self) -> float:
        return time.monotonic()


class FakeClock:
    """Test double. `advance` is the only way time moves."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("time does not go backwards")
        self._t += seconds
        return self._t


class SteppingClock:
    """A clock that returns whatever it is told, including backwards.
    Exists only so the tests can prove the limiter is defensive."""

    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value


# --------------------------------------------------------------------------
# 2. The value object the limiter returns
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: float = 0.0

    @property
    def headers(self) -> dict[str, str]:
        out = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(max(0, self.remaining)),
        }
        if not self.allowed:
            out["Retry-After"] = str(max(1, math.ceil(self.retry_after)))
        return out

    def __bool__(self) -> bool:
        return self.allowed


class RateLimitExceeded(Exception):
    pass


# --------------------------------------------------------------------------
# 3. Helpers: a bounded key space and striped locks
# --------------------------------------------------------------------------

class KeySpace:
    """A dict with a ceiling. Without this, the limiter's own memory is a
    denial-of-service vector: an attacker sends one request per fake identity
    and the per-key state grows without bound."""

    def __init__(self, max_keys: int = 100_000) -> None:
        self.max_keys = max_keys
        self._data: dict[str, object] = {}

    def get(self, key: str) -> Optional[object]:
        return self._data.get(key)

    def put(self, key: str, value: object) -> None:
        if key not in self._data and len(self._data) >= self.max_keys:
            # Evict in insertion order: the eldest key is the least likely to
            # be currently active. A true LRU needs a linked list; this is the
            # cheap approximation and it is honest about being one.
            for eldest in next(iter(self._data)):
                del self._data[eldest]
                break
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> Iterator[str]:
        return iter(list(self._data))


class KeyedLocks:
    """Striped locks. One global lock serialises every client; one lock per
    key is an unbounded number of objects. A fixed stripe count is the
    standard middle path — unrelated keys usually land on different stripes."""

    def __init__(self, stripes: int = 64) -> None:
        self._locks = [threading.Lock() for _ in range(stripes)]

    def for_key(self, key: str) -> threading.Lock:
        return self._locks[hash(key) % len(self._locks)]


# --------------------------------------------------------------------------
# 4. The interface
# --------------------------------------------------------------------------

class RateLimiter(ABC):
    @abstractmethod
    def allow(self, key: str, cost: int = 1) -> Decision:
        """Decide AND consume, atomically."""

    def peek(self, key: str) -> int:
        """How much budget is left, WITHOUT consuming. Not the admission path.
        Overridden where the state makes it cheap."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# 5. Token bucket — O(1) memory, tolerates bursts
# --------------------------------------------------------------------------

@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucket(RateLimiter):
    """Refill `rate` tokens per second up to `capacity`.

    Lazy refill: nothing runs on a timer. Tokens are computed from the elapsed
    time at the moment of the request, which is what makes this O(1) memory
    and O(1) time with no background threads.
    """

    def __init__(self, rate: float, capacity: float, clock: Optional[Clock] = None,
                 max_keys: int = 100_000) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity < 1:
            raise ValueError("capacity must be at least one request")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.clock = clock if clock is not None else MonotonicClock()
        self._space = KeySpace(max_keys)
        self._locks = KeyedLocks()

    def _bucket(self, key: str, now: float) -> _Bucket:
        bucket = self._space.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self.capacity, updated=now)
            self._space.put(key, bucket)
        return bucket  # type: ignore[return-value]

    def allow(self, key: str, cost: int = 1) -> Decision:
        if cost < 1:
            raise ValueError("cost must be positive")
        if cost > self.capacity:
            # Asking for more than the bucket can ever hold is a permanent
            # rejection, not a temporary one. Say so instead of looping.
            return Decision(False, int(self.capacity), 0, retry_after=float("inf"))

        with self._locks.for_key(key):
            now = self.clock.now()
            bucket = self._bucket(key, now)
            elapsed = now - bucket.updated
            if elapsed > 0:
                bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.rate)
                bucket.updated = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return Decision(True, int(self.capacity), int(bucket.tokens))

            deficit = cost - bucket.tokens
            return Decision(
                False, int(self.capacity), int(bucket.tokens),
                retry_after=deficit / self.rate,
            )

    def peek(self, key: str) -> int:
        bucket = self._space.get(key)
        if bucket is None:
            return int(self.capacity)
        now = self.clock.now()
        refilled = min(self.capacity, bucket.tokens + max(0.0, now - bucket.updated) * self.rate)
        return int(refilled)


# --------------------------------------------------------------------------
# 6. Leaky bucket — O(1) memory, smooths output instead of bursting
# --------------------------------------------------------------------------

@dataclass
class _Level:
    level: float
    updated: float


class LeakyBucket(RateLimiter):
    """Queue semantics: requests enter a queue of depth `capacity` and drain
    at `rate` per second.

    Note the duality, and say it out loud: as a *meter* this is a token
    bucket. As a *queue* the accounting is "level must stay under capacity"
    instead of "tokens must stay above zero". The admission arithmetic is
    mirror-image, which is why the two are often confused.
    """

    def __init__(self, rate: float, capacity: float, clock: Optional[Clock] = None,
                 max_keys: int = 100_000) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity < 1:
            raise ValueError("capacity must be at least one request")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.clock = clock if clock is not None else MonotonicClock()
        self._space = KeySpace(max_keys)
        self._locks = KeyedLocks()

    def allow(self, key: str, cost: int = 1) -> Decision:
        if cost < 1:
            raise ValueError("cost must be positive")

        with self._locks.for_key(key):
            now = self.clock.now()
            state = self._space.get(key)
            if state is None:
                state = _Level(level=0.0, updated=now)
                self._space.put(key, state)
            state = state  # type: ignore[assignment]

            drained = max(0.0, now - state.updated) * self.rate
            level = max(0.0, state.level - drained)
            state.updated = now

            if level + cost <= self.capacity:
                state.level = level + cost
                return Decision(True, int(self.capacity), int(self.capacity - state.level))

            state.level = level
            overflow = level + cost - self.capacity
            return Decision(False, int(self.capacity), int(self.capacity - level),
                            retry_after=overflow / self.rate)

    def peek(self, key: str) -> int:
        state = self._space.get(key)
        if state is None:
            return int(self.capacity)
        now = self.clock.now()
        level = max(0.0, state.level - max(0.0, now - state.updated) * self.rate)
        return int(max(0.0, self.capacity - level))


# --------------------------------------------------------------------------
# 7. Fixed-window counter — the naive baseline, for contrast
# --------------------------------------------------------------------------

@dataclass
class _Window:
    index: int
    count: int


class FixedWindowCounter(RateLimiter):
    """One counter per window. The bug is at the boundary: with a limit of 100
    per minute, a client can send 100 at 00:00:59 and 100 at 00:01:00 — 200
    requests in a two-second span — and the limiter is perfectly happy."""

    def __init__(self, limit: int, window_seconds: float, clock: Optional[Clock] = None,
                 max_keys: int = 100_000) -> None:
        self.limit = int(limit)
        self.window = float(window_seconds)
        self.clock = clock if clock is not None else MonotonicClock()
        self._space = KeySpace(max_keys)
        self._locks = KeyedLocks()

    def allow(self, key: str, cost: int = 1) -> Decision:
        with self._locks.for_key(key):
            now = self.clock.now()
            index = int(now // self.window)
            state = self._space.get(key)
            if state is None or state.index != index:
                state = _Window(index=index, count=0)
                self._space.put(key, state)

            if state.count + cost <= self.limit:
                state.count += cost
                return Decision(True, self.limit, self.limit - state.count)

            boundary = (index + 1) * self.window
            return Decision(False, self.limit, max(0, self.limit - state.count),
                            retry_after=boundary - now)


# --------------------------------------------------------------------------
# 8. Sliding-window log — exact, and the memory is the price
# --------------------------------------------------------------------------

class SlidingWindowLog(RateLimiter):
    """Keep the timestamp of every accepted request in the window.

    Exact: there is no boundary artefact, because the window slides with the
    request rather than snapping to a grid. The cost is O(limit) floats plus
    deque overhead per active key — 100,000 limits × 100,000 keys is not a
    thing you can do, which is why this is the correctness reference and not
    the production default.
    """

    def __init__(self, limit: int, window_seconds: float, clock: Optional[Clock] = None,
                 max_keys: int = 100_000, prune_every: int = 1_000) -> None:
        self.limit = int(limit)
        self.window = float(window_seconds)
        self.clock = clock if clock is not None else MonotonicClock()
        self._space = KeySpace(max_keys)
        self._locks = KeyedLocks()
        self._prune_every = prune_every
        self._calls = 0

    def allow(self, key: str, cost: int = 1) -> Decision:
        if cost > self.limit:
            return Decision(False, self.limit, 0, retry_after=float("inf"))

        with self._locks.for_key(key):
            now = self.clock.now()
            cutoff = now - self.window
            log = self._space.get(key)
            if log is None:
                log = deque()
                self._space.put(key, log)
            log = log  # type: ignore[assignment]

            # Drop everything that has slid out of the window.
            while log and log[0] <= cutoff:
                log.popleft()

            if len(log) + cost <= self.limit:
                for _ in range(cost):
                    log.append(now)
                return Decision(True, self.limit, self.limit - len(log))

            # The retry moment is when the oldest entry leaves the window.
            retry_after = (log[0] + self.window) - now if log else 0.0
            return Decision(False, self.limit, self.limit - len(log), retry_after=retry_after)

    def peek(self, key: str) -> int:
        log = self._space.get(key)
        if log is None:
            return self.limit
        cutoff = self.clock.now() - self.window
        return max(0, self.limit - sum(1 for ts in log if ts > cutoff))

    def sweep(self) -> int:
        """Periodic maintenance: drop empty deques so the key space does not
        grow forever with keys that stopped sending traffic. Call it from a
        background task; it is deliberately NOT on the request path."""
        removed = 0
        cutoff = self.clock.now() - self.window
        for key in list(self._space.keys()):
            log = self._space.get(key)
            if log is None:
                continue
            while log and log[0] <= cutoff:
                log.popleft()
            if not log:
                del self._space._data[key]  # deliberate: this helper owns it
                removed += 1
        return removed

    def allow_and_maybe_sweep(self, key: str, cost: int = 1) -> Decision:
        decision = self.allow(key, cost)
        self._calls += 1
        if self._calls % self._prune_every == 0:
            self.sweep()
        return decision


# --------------------------------------------------------------------------
# 9. Sliding-window counter — O(1) memory, approximate
# --------------------------------------------------------------------------

@dataclass
class _Counters:
    index: int
    current: int
    previous: int


class SlidingWindowCounter(RateLimiter):
    """Two fixed windows, weighted by how far into the current one we are.

        estimate = previous * (1 - progress) + current

    It is an *approximation*: it assumes traffic in the previous window was
    spread evenly, so a client that front-loaded the previous window is
    under-counted and one that back-loaded it is over-counted. That is the
    accepted trade: O(1) memory per key and no boundary artefact big enough
    to matter. Cloudflare's published figure for this technique on real
    traffic is a ~0.003% error rate.
    """

    def __init__(self, limit: int, window_seconds: float, clock: Optional[Clock] = None,
                 max_keys: int = 100_000) -> None:
        self.limit = int(limit)
        self.window = float(window_seconds)
        self.clock = clock if clock is not None else MonotonicClock()
        self._space = KeySpace(max_keys)
        self._locks = KeyedLocks()

    def allow(self, key: str, cost: int = 1) -> Decision:
        if cost > self.limit:
            return Decision(False, self.limit, 0, retry_after=float("inf"))

        with self._locks.for_key(key):
            now = self.clock.now()
            index = int(now // self.window)
            state = self._space.get(key)
            if state is None:
                state = _Counters(index=index, current=0, previous=0)
                self._space.put(key, state)
            state = state  # type: ignore[assignment]

            if state.index != index:
                # Roll: the current window becomes the previous one. If more
                # than one window elapsed, the previous window is empty.
                state.previous = state.current if index - state.index == 1 else 0
                state.current = 0
                state.index = index

            progress = (now - index * self.window) / self.window
            estimate = state.previous * (1.0 - progress) + state.current
            remaining = self.limit - estimate

            if remaining >= cost:
                state.current += cost
                return Decision(True, self.limit, int(self.limit - estimate - cost))

            # Time until the weighted estimate drops enough is not closed-form;
            # report the moment the current window ends, which is always safe.
            boundary = (index + 1) * self.window
            return Decision(False, self.limit, int(max(0, remaining)),
                            retry_after=boundary - now)

    def peek(self, key: str) -> int:
        state = self._space.get(key)
        if state is None:
            return self.limit
        now = self.clock.now()
        index = int(now // self.window)
        if state.index != index:
            return self.limit
        progress = (now - index * self.window) / self.window
        estimate = state.previous * (1.0 - progress) + state.current
        return int(max(0, self.limit - estimate))


# --------------------------------------------------------------------------
# 10. Composite — several limits at once
# --------------------------------------------------------------------------

class CompositeLimiter(RateLimiter):
    """Every sub-limiter must allow (per-IP AND per-user AND global).

    The caveat, stated openly: if the first sub-limiter admits a request and a
    later one rejects it, the first one's budget has already been consumed.
    That is acceptable for limits that are individually approximate, and NOT
    acceptable when one of them is a billing quota. The fix is a two-phase
    reserve/commit, which needs every sub-limiter to support a rollback — a
    real cost, paid only when the quota is money.
    """

    def __init__(self, limiters: list[RateLimiter]) -> None:
        if not limiters:
            raise ValueError("a composite needs at least one limiter")
        self.limiters = list(limiters)

    def allow(self, key: str, cost: int = 1) -> Decision:
        allowed: list[Decision] = []
        for limiter in self.limiters:
            decision = limiter.allow(key, cost)
            if not decision.allowed:
                # Report the binding constraint, with the longest wait, so the
                # caller's Retry-After is the one that actually unblocks them.
                worst = max(allowed + [decision], key=lambda d: d.retry_after)
                return Decision(False, worst.limit, worst.remaining,
                                retry_after=worst.retry_after)
            allowed.append(decision)
        return Decision(
            True,
            min(d.limit for d in allowed),
            min(d.remaining for d in allowed),
        )


# --------------------------------------------------------------------------
# 11. A raise-style facade, for callers that prefer exceptions
# --------------------------------------------------------------------------

class GuardedLimiter:
    def __init__(self, limiter: RateLimiter) -> None:
        self.limiter = limiter

    def acquire(self, key: str, cost: int = 1) -> Decision:
        decision = self.limiter.allow(key, cost)
        if not decision.allowed:
            raise RateLimitExceeded(f"{key}: retry in {decision.retry_after:.3f}s")
        return decision


# --------------------------------------------------------------------------
# 12. Tests — the point of the injected clock is that none of these sleep
# --------------------------------------------------------------------------

def _check(name: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    if not condition:
        raise SystemExit(1)


if __name__ == "__main__":
    # --- token bucket: burst, then refill ---
    clock = FakeClock()
    bucket = TokenBucket(rate=10, capacity=10, clock=clock)
    _check("token bucket admits a full burst",
           all(bucket.allow("alice").allowed for _ in range(10)))
    _check("token bucket rejects the 11th", not bucket.allow("alice").allowed)
    _check("other keys are unaffected", bucket.allow("bob").allowed)
    clock.advance(0.5)
    _check("half a second buys 5 tokens",
           sum(bucket.allow("alice").allowed for _ in range(6)) == 5)

    # --- retry_after is real ---
    clock2 = FakeClock()
    b2 = TokenBucket(rate=2, capacity=2, clock=clock2)
    b2.allow("x"); b2.allow("x")
    denied = b2.allow("x")
    _check("retry_after is 0.5s at rate 2/s", abs(denied.retry_after - 0.5) < 1e-9)
    _check("Retry-After header rounds up", denied.headers["Retry-After"] == "1")

    # --- leaky bucket smooths ---
    clock3 = FakeClock()
    leaky = LeakyBucket(rate=1, capacity=2, clock=clock3)
    _check("leaky admits up to capacity",
           leaky.allow("a").allowed and leaky.allow("a").allowed)
    _check("leaky rejects the third", not leaky.allow("a").allowed)
    clock3.advance(1.0)
    _check("leaky drains one per second", leaky.allow("a").allowed)

    # --- sliding-window log is exact across the boundary ---
    clock4 = FakeClock()
    log = SlidingWindowLog(limit=3, window_seconds=10, clock=clock4)
    for _ in range(3):
        log.allow("ip")
    _check("log rejects the 4th", not log.allow("ip").allowed)
    clock4.advance(10.0)
    _check("log admits after the window slides", log.allow("ip").allowed)

    # The boundary case fixed window gets wrong:
    clock5 = FakeClock(59.0)
    fixed = FixedWindowCounter(limit=3, window_seconds=60, clock=clock5)
    for _ in range(3):
        fixed.allow("ip")
    clock5.advance(2.0)                      # now at 61s -> new window
    burst = sum(fixed.allow("ip").allowed for _ in range(3))
    _check("fixed window allows a 2x burst at the boundary", burst == 3)

    clock6 = FakeClock(59.0)
    counter = SlidingWindowCounter(limit=3, window_seconds=60, clock=clock6)
    for _ in range(3):
        counter.allow("ip")
    clock6.advance(2.0)
    _check("sliding counter dampens the boundary burst",
           sum(counter.allow("ip").allowed for _ in range(3)) < 3)

    # --- composite: the strictest limit wins ---
    clock7 = FakeClock()
    composite = CompositeLimiter([
        TokenBucket(rate=1, capacity=100, clock=clock7),
        TokenBucket(rate=1, capacity=2, clock=clock7),
    ])
    _check("composite admits up to the tightest",
           composite.allow("u").allowed and composite.allow("u").allowed)
    _check("composite rejects at the tightest", not composite.allow("u").allowed)

    # --- defensive: a backwards clock ---
    stepping = SteppingClock()
    defensive = TokenBucket(rate=10, capacity=10, clock=stepping)
    defensive.allow("k")
    stepping.value = -100.0                   # time jumps backwards
    _check("a backwards clock does not mint tokens", defensive.allow("k").allowed)
    _check("and does not crash or go negative", defensive.peek("k") >= 0)

    print("\nall checks passed")
```

**What to point at:**

- `allow` **decides and consumes in one call**. There is no `check` + `consume` pair, because that pair is the bug.
- Every algorithm computes its decay from `now - updated` **inside the lock**. Computing it outside and then mutating is the classic torn read-modify-write.
- `KeySpace` exists because per-key state is **attacker-controlled memory**. Naming that unprompted is a senior signal.
- `SlidingWindowLog.sweep()` is explicitly **off the request path**. A maintenance task that runs on every request turns O(1) admission into O(keys).
- `cost > capacity` returns `retry_after=float("inf")` rather than looping forever — a permanent rejection must not look temporary.
- The demo's last two checks use `SteppingClock` to prove a **backwards clock** neither mints tokens nor crashes.

---

## 6. Concurrency & thread-safety

### What breaks

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Torn read-modify-write** | Thread A reads `tokens=1`, thread B reads `tokens=1`, both decide "allowed", both write `tokens=0` — two admissions from one token | Do read, decay, decide and write inside one lock (the code does) |
| **Decay computed outside the lock** | The elapsed time is measured against a state that another thread has already replaced | Compute `now` and the decay inside the critical section |
| **One global lock** | Every client serialises on one mutex; at 100k QPS the limiter is the bottleneck | Striped locks (`KeyedLocks`) — unrelated keys rarely collide |
| **One lock per key object** | Unbounded lock allocation, and a lock that outlives its key | A fixed stripe count |
| **`setdefault` as a check-then-act** | `if key not in d: d[key] = ...` from two threads creates two states, and one thread's consumption is lost | Create inside the lock |
| **`KeySpace` eviction racing a reader** | The evicted state is still referenced by a thread mid-decision; the write lands on an orphan | Evict inside the same stripe lock, or accept the lost write as a rare reset (say which) |
| **`sweep()` concurrent with `allow()`** | The sweep deletes a deque a request is appending to | Sweep under the stripe lock, or tolerate it — the consequence is one lost admission |
| **Distributed: two app instances** | In-process state is per-instance, so N instances means N× the limit | Move the state to Redis and make the decay-and-consume a single Lua script |

### The distributed fix, since it is the obvious follow-up

In-process state multiplies by the instance count — the classic "limits that multiply by instance count" failure in [the gateway file](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md) §6.5. The fix is to make the whole decay-decide-consume sequence one atomic server-side operation, which is what Lua is for:

```lua
-- token_bucket.lua — atomic, and the clock comes from Redis, not the caller.
-- KEYS[1] = rl:{scope}:{key}
-- ARGV    = rate, capacity, cost, ttl_seconds
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local cost     = tonumber(ARGV[3])
local ttl      = tonumber(ARGV[4])

-- redis.call('TIME') returns {seconds, microseconds}. Use the SERVER's clock
-- so clock skew across app instances cannot corrupt the window.
local t     = redis.call('TIME')
local now   = tonumber(t[1]) + tonumber(t[2]) / 1000000

local state = redis.call('HMGET', key, 'tokens', 'updated')
local tokens  = tonumber(state[1])
local updated = tonumber(state[2])

if tokens == nil then
  tokens  = capacity
  updated = now
end

local elapsed = now - updated
if elapsed < 0 then elapsed = 0 end        -- defensive: never mint on a rewind
if elapsed > 0 then
  tokens  = math.min(capacity, tokens + elapsed * rate)
  updated = now
end

local allowed = 0
local retry   = 0
if tokens >= cost then
  tokens  = tokens - cost
  allowed = 1
else
  retry = (cost - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'updated', updated)
redis.call('EXPIRE', key, ttl)             -- idle keys clean themselves up

if allowed == 1 then
  return {1, math.floor(tokens), 0}
end
return {0, math.floor(tokens), math.ceil(retry * 1000)}
```

Four things to say about it, all of which are the reason the script looks like this:

1. **`redis.call('TIME')`, not the caller's clock.** Clock skew across app instances is a real source of wrong limits ([gateway §6.6](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md)).
2. **`EXPIRE` on every write**, so an idle key's state disappears and the Redis keyspace does not grow without bound. This is `KeySpace`'s eviction, done by Redis.
3. **Hash tags** — `rl:{tenant7}:user42` — so the key lands in one slot in Cluster, and the script never crosses slots.
4. **`elapsed < 0` guarded**, because a backwards clock must not mint tokens. Same defensive check as `SteppingClock` exercises in the in-process version.

The honest limit of the distributed version: it is **one round trip per request** and it makes Redis a hard dependency on the admission path. That is why the gateway file discusses fail-open versus fail-closed, and why the in-process striped limiter in this file is not redundant — it is the fallback when Redis is unreachable.

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Move the state to Redis"

Replace the limiter implementation, keep the interface:

```python
class RedisTokenBucket(RateLimiter):
    """Same interface, same Decision. The state moved; nothing else did."""

    SCRIPT = """..."""   # the Lua above, loaded once with SCRIPT LOAD

    def __init__(self, client, rate, capacity, scope="rl", ttl=3600):
        self.client, self.rate, self.capacity = client, rate, capacity
        self.scope, self.ttl = scope, ttl
        self._sha = client.script_load(self.SCRIPT)

    def _redis_key(self, key: str) -> str:
        # Hash tag so a Cluster script stays in one slot.
        return f"{self.scope}:{{{key}}}"

    def allow(self, key: str, cost: int = 1) -> Decision:
        allowed, remaining, retry_ms = self.client.evalsha(
            self._sha, 1, self._redis_key(key),
            self.rate, self.capacity, cost, self.ttl,
        )
        return Decision(bool(allowed), int(self.capacity), int(remaining),
                        retry_after=retry_ms / 1000.0)
```

**The sentence that scores:** *"The whole point of the interface is that the algorithm's state is private, so moving it to Redis is a new implementation, not a change to any caller."* And the caveats: handle `NOSCRIPT` after a failover by reloading, and decide fail-open versus fail-closed before you need to.

### 7.2 "Weighted cost per endpoint"

Already in the interface as `cost`. The policy — which endpoint costs what — belongs in configuration, not in the limiter:

```python
ENDPOINT_COST = {
    ("POST", "/v1/messages"): 1,
    ("GET", "/v1/feed"): 1,
    ("POST", "/v1/embeddings"): 5,
    ("POST", "/v1/completions"): 20,
}


def cost_of(method: str, path: str) -> int:
    return ENDPOINT_COST.get((method, path), 1)
```

For LLM endpoints the natural unit is **tokens, not requests**, and the cost is only known *after* the response. That needs a reserve-then-settle flow: reserve an estimate before the call, settle the actual after. Naming that is worth more than the table — it is the same reserve/commit shape that makes `CompositeLimiter` correct.

### 7.3 "Hierarchical limits: per-user, per-tenant, per-IP, global"

```python
def build_hierarchy(clock: Clock) -> RateLimiter:
    """The tightest limit wins; the composite reports the binding one."""
    return CompositeLimiter([
        TokenBucket(rate=5, capacity=10, clock=clock),        # per IP
        TokenBucket(rate=20, capacity=40, clock=clock),       # per user
        TokenBucket(rate=500, capacity=1_000, clock=clock),   # per tenant
        TokenBucket(rate=5_000, capacity=10_000, clock=clock),  # global
    ])
```

Two design points. First, the **global** limiter is a hot key in a distributed setting — it must be sharded (`rl:global:{0..9}` summed) or approximated locally, which the gateway file covers. Second, the composite's **partial-consumption caveat** is real here: a request that passes per-IP but fails the global limit has still spent its IP budget. For abuse limits that is fine; for a billing quota it is not.

### 7.4 "A dry-run / remaining-quota endpoint"

`peek(key)` is already on the interface. Say the boundary explicitly: **`peek` is not the admission path.** An admin UI shows remaining quota with `peek`; the data path uses `allow`. A limiter that exposes only `peek` + `consume` has a race by construction.

### 7.5 "Dynamic limits from configuration"

Do not make the limiter poll config. Inject a resolver:

```python
class ConfiguredLimiter:
    """The policy is a function; the mechanism is unchanged."""

    def __init__(self, resolver, build):
        self.resolver = resolver      # tenant -> (limit, window)
        self.build = build            # (limit, window) -> RateLimiter
        self._cache: dict[str, RateLimiter] = {}

    def for_tenant(self, tenant: str) -> RateLimiter:
        if tenant not in self._cache:
            self._cache[tenant] = self.build(*self.resolver(tenant))
        return self._cache[tenant]
```

### 7.6 "Coalesce duplicate rejections"

At scale, a rejected client retries immediately and each retry is a full limiter evaluation. A short-lived **deny cache** (`key -> retry_after`) collapses those:

```python
class DenyCache:
    def __init__(self, clock, default_ttl=1.0):
        self.clock, self.default_ttl = clock, default_ttl
        self._denies: dict[str, float] = {}

    def check(self, key: str) -> Optional[float]:
        until = self._denies.get(key)
        if until is None:
            return None
        if self.clock.now() >= until:
            del self._denies[key]
            return None
        return until - self.clock.now()

    def remember(self, key: str, retry_after: float) -> None:
        self._denies[key] = self.clock.now() + min(retry_after, self.default_ttl)
```

Note the cap: caching a denial for longer than a second risks serving stale denials after a limit change. Bounded staleness, not unbounded.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Two threads on one token** | Both read `tokens=1`, both admit | Read, decay, decide and write inside one lock |
| 2 | **Backwards clock** | `elapsed` is negative; naive refill *removes* tokens, or a `max(0, ...)` mistake mints them | Clamp `elapsed` to `>= 0`; use monotonic time |
| 3 | **Wall clock instead of monotonic** | An NTP step or DST change grants a fresh budget | `time.monotonic()` behind the `Clock` port |
| 4 | **`cost > capacity`** | Rejected with a finite `retry_after`, so the client retries forever | Reject with `retry_after = inf`; a permanent rejection must look permanent |
| 5 | **Unbounded key space** | One request per fake IP; the limiter OOMs | `KeySpace` with `max_keys` and eviction |
| 6 | **Key never expires after the last request** | A key that stopped sending traffic holds state forever | TTL/eviction: Redis `EXPIRE`, or `SlidingWindowLog.sweep()` offline |
| 7 | **Fixed-window boundary burst** | 2× the limit across the boundary in a few milliseconds | Sliding window, log or counter |
| 8 | **Sliding-window-log memory** | 100k limit × 100k keys = tens of GB | Use the counter; keep the log as the correctness reference and for small limits |
| 9 | **`sweep()` on the request path** | O(keys) work per request; admission latency collapses | Sweep from a background task, on a counter, never per request |
| 10 | **Sliding-counter roll with a multi-window gap** | The previous window's count is carried over from minutes ago | `previous = current if index - state.index == 1 else 0` |
| 11 | **`retry_after = 0`** | The client retries immediately and thrashes | Floor `Retry-After` at 1 second (`max(1, ceil(...))`) |
| 12 | **Composite partial consumption** | A request denied by the global limit still spent its per-IP budget | Accept for approximate limits; two-phase reserve/commit for billing quotas |
| 13 | **Distributed: N instances** | The effective limit is N × the configured limit | Move the state to Redis; one Lua script for decay-decide-consume |
| 14 | **Lua script missing after failover** | `NOSCRIPT` errors on every request | Catch `NOSCRIPT` and reload the script once |
| 15 | **Limiter store unreachable** | Requests fail closed and the whole API goes down with Redis | Choose deliberately: fail-open with a conservative local fallback, and say which |
| 16 | **`peek` used for admission** | A check-then-consume race, by design | `peek` is for display only; `allow` is the only admission path |
| 17 | **Key derived from unbounded user input** | A 10 KB header becomes a 10 KB dict key | Normalise and bound the key (hash it, or validate the shape) |
| 18 | **Cost of 0 or negative** | A free pass, or tokens minted | Validate `cost >= 1` |
| 19 | **`setdefault` outside the lock** | Two threads create two states; one thread's consumption is lost | Create inside the stripe lock |

**The two to volunteer:** #2 (backwards clock) and #13 (limits multiplying by instance count). The first shows you have thought about the arithmetic's assumptions; the second is the whole reason the distributed design exists.

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **Strategy** | The four algorithms behind `RateLimiter` | They differ only in *state and admission arithmetic*; the caller's contract is identical. Swapping token bucket for sliding counter is a one-line change | Four implementations to keep behaviourally aligned; needs a shared conformance test |
| **Value Object** | `Decision` (frozen dataclass) | Immutable, hashable, safe to log; `headers` lives with the decision instead of being recomputed at every call site | Rebuilt per request; negligible |
| **Ports & Adapters** | `Clock` | Makes the limiter testable without `sleep`, and makes "backwards clock" a unit test rather than a production incident | One indirection per time read |
| **Template Method** | Not used deliberately | The four algorithms share almost no code — the decay arithmetic differs — so a base class would be a forced abstraction | — |
| **Decorator** | `GuardedLimiter`, `DenyCache`, `CompositeLimiter` all wrap a `RateLimiter` | Behaviour added without touching an algorithm: raise instead of return, cache denials, AND several limits | Each wrapper adds a layer to the call stack and to the debugging story |
| **Facade** | `GuardedLimiter.acquire` | Callers that prefer exceptions get them without the algorithm knowing | Two ways to call the limiter; pick one per codebase |
| **Flyweight / Striped locking** | `KeyedLocks` | Bounds the number of lock objects while keeping contention low | Unrelated keys occasionally share a stripe and contend |
| **Singleton** | *Deliberately rejected* | A global limiter instance makes tests order-dependent and hides the fact that limits are per-scope | — |

**The pattern answer to give:** *"Strategy for the algorithms, a value object for the decision, a port for the clock, and Decorator for the wrapping behaviours — deny caching, exception-raising, composition. I would explicitly not use Template Method here: the algorithms share almost no arithmetic, so a base class would be a forced abstraction. And I would not make the limiter a singleton, because the scope of a limit is a composition decision, not a global."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Token bucket as the default** | Yes | O(1) memory per key, burst tolerance, trivial to reason about | Admits a burst of `capacity` instantly, which is exactly what some downstreams cannot take |
| **Leaky bucket instead** | Offered | Smooth output; the downstream sees a steady rate | No burst tolerance, so a legitimate client that batched is punished |
| **Sliding-window log** | Offered, documented as the exact reference | No boundary artefact; per-request precision | O(limit) memory per key — 100k limit × 100k keys is not deployable |
| **Sliding-window counter** | Offered | O(1) memory and a ~0.003%-class error; the production middle path | Assumes uniform traffic in the previous window, so it can under- or over-admit |
| **Fixed-window counter** | Offered as the baseline | Simplest possible; one integer | The 2× boundary burst — the bug everyone ships first |
| **`cost` in the interface from day one** | Yes | Weighted endpoints and token budgets are additive | One extra parameter nobody uses until they need it |
| **Clock injected as a port** | Yes | Tests advance time; backwards-clock is a unit test | One indirection; a caller who ignores it silently gets the real clock |
| **Striped locks over per-key locks** | Yes | Bounded lock count; low contention | Occasional false sharing between unrelated keys |
| **In-process state (this file)** | Yes, with the Redis path named | No network hop on the admission path; no Redis dependency | Limits multiply by instance count — correct only behind a single instance or with sticky routing |
| **`CompositeLimiter` without reserve/commit** | Yes, with the caveat documented | Simple, and correct for approximate abuse limits | A denied request may still have spent another limit's budget |
| **Fail-open on limiter failure** | Stated as the default | The API survives a Redis outage | An attacker who can break the limiter gets unlimited traffic — mitigate with a conservative local fallback |
| **Deny caching with a 1-second cap** | Yes | Collapses retry storms into one evaluation | Up to a second of stale denials after a limit change |

**The trade to name unprompted:** *"The memory-versus-precision axis is the whole decision. The log is exact and unaffordable at scale; the counter is affordable and approximate; the bucket is affordable and bursty. I would ship the counter or the bucket, and keep the log as the reference implementation I test the others against."*

---

## 11. Q&A

**Q1. Why one interface with four implementations rather than a configuration flag?**
Because the algorithms differ in their *state*, not in their behaviour at the seam. The caller asks one question — "may this request through?" — and the answer's shape is identical; only the memory cost and the precision differ. A `mode="sliding"` flag inside one class would put four different state layouts in one dict and make each branch reachable in every test. The interface also makes the important claim explicit: these are *substitutable*, which is why you can test the cheap one against the exact one.

**Q2. Token bucket or leaky bucket?**
They are duals, and the difference is what you promise the downstream. A token bucket promises an *average* rate with bursts up to the bucket size — good for a client that legitimately batches. A leaky bucket promises a *smooth* output rate because requests queue and drain at a fixed rate — good when the downstream is a fragile service or a third-party API with a hard per-second cap. If you cannot say which property you need, you do not yet know your downstream, and that is the real answer.

**Q3. Why is the sliding-window log not the default, given it is the exact one?**
Because it stores a timestamp per accepted request: memory is O(limit) per active key. With a limit of 100,000 and 100,000 active keys that is on the order of 10¹⁰ timestamps. It is the right tool for small limits (10 per minute per API key) and the right *reference implementation* to test the approximations against. The sliding-window counter gets you within a fraction of a percent for two integers per key, and that is the trade most systems should take.

**Q4. What exactly is wrong with a fixed-window counter?**
The window snaps to a grid, so the boundary is not enforced. With 100 requests per minute, a client sends 100 at 00:00:59 and 100 at 00:01:00 — 200 requests in roughly a second — and both windows are individually compliant. The counter is fine as a cheap first cut and wrong as a rate limiter. The sliding counter's whole purpose is to damp that artefact without paying the log's memory.

**Q5. Why is `allow` not split into `check` and `consume`?**
Because that split *is* the race. A caller checks (allowed), does work, then consumes, and between the two another request can consume the same budget. One method that decides and consumes atomically makes the interleaving unrepresentable. If a caller needs to display remaining quota, that is `peek`, which does not mutate and is explicitly not on the admission path.

**Q6. Why is the clock injected?**
Two reasons. Testing: a limiter tested with `time.sleep` has slow, flaky CI, whereas a `FakeClock` makes "advance 0.5s and expect exactly 5 tokens" an exact assertion. Correctness: time is an *input* to the algorithm, and an algorithm whose input comes from a global is one you cannot test at its boundaries — including the backwards-clock case, which is a real production failure and here is a three-line unit test.

**Q7. Why monotonic time and not the wall clock?**
Because a rate-limit budget is a duration, not a date. A wall-clock adjustment — an NTP step, a manual fix, a DST bug in a naive implementation — can move time backwards or forwards by seconds. Forwards mints budget that was never earned; backwards either destroys budget or, with a sign error in the clamp, mints it. `time.monotonic()` is guaranteed non-decreasing, and the code additionally clamps `elapsed` to `>= 0` so an injected clock that violates monotonicity still cannot mint tokens.

**Q8. Why does `cost > capacity` return `retry_after = inf`?**
Because it is a permanent rejection for that request shape. The bucket can never hold more than `capacity` tokens, so a cost above capacity will never be satisfiable no matter how long the client waits. Returning a finite `retry_after` invites the client to retry forever, which is a self-inflicted retry storm and exactly the failure mode retry-after headers are supposed to prevent. An infinite wait is honest — and it usually reveals a configuration bug, which is the point.

**Q9. Why `KeySpace` rather than a plain dict?**
Because the keys are usually client-controlled — IP, API key, user id. An attacker sends one request per fabricated identity and the limiter's own memory grows without bound. That is a denial of service against the component whose job is to prevent denial of service. `KeySpace` caps the number of keys and evicts the oldest, which bounds memory. In the Redis implementation the equivalent is `EXPIRE` on every write.

**Q10 (senior). Your `SlidingWindowLog` has a `sweep()` and an `allow_and_maybe_sweep`. Why is the sweep not in `allow`?**
Because sweeping is O(number of keys) and `allow` must be O(1). Putting it on the request path turns a constant-time admission into one that degrades with the size of the key space — the worst kind of performance bug, because it is invisible until the system is large and then it is a cliff. So the sweep runs every N calls or from a background task. The honest trade is that this makes eviction *approximate* in time: an idle key's state can linger for up to a sweep interval.

**Q11 (senior). What breaks when you run this behind ten instances?**
The limit multiplies by ten, because each instance keeps its own per-key state and each admits up to the configured limit. This is the classic "limits that multiply by instance count" failure and it is invisible in a single-instance test. The fix is to move the state to a shared store and make the entire decay-decide-consume sequence atomic there — a Lua script in Redis, keyed with a hash tag so Cluster keeps it in one slot, reading time with `redis.call('TIME')` so clock skew between app instances cannot corrupt the window. It costs one round trip per request and makes Redis a dependency on the admission path, which is why fail-open versus fail-closed becomes a design decision rather than a detail.

**Q12 (senior). When is `CompositeLimiter`'s partial consumption a correctness bug, and what is the fix?**
It is a bug when one of the composed limits represents a *resource with a cost* — a billing quota, a token budget — rather than an abuse ceiling. A request that passes the per-IP limit but fails the tenant quota has consumed IP budget for work that will never happen, so the client is charged twice for one request: once in IP budget, once in the retry. The fix is two-phase: `reserve(key, cost)` returns a reservation token, `commit(token)` finalises it, and `rollback(token)` releases it. Every sub-limiter then needs a release operation, which is why you pay for this only where the limit is money. It is the same reserve-then-settle shape as metering LLM tokens, where the cost is only known after the call.

**Q13. Where would a bug most likely hide in this code?**
In the interaction between the decay and the admission. The invariant each algorithm must maintain is that the state it writes is consistent with the state it read plus the elapsed time — and every one of them does that read-modify-write under a lock. The likely bug is a decay computed *outside* the lock, which looks harmless in a single-threaded test and produces a lost admission under concurrency. The second-likeliest is the sliding counter's window roll, specifically the multi-window gap case: if a key is idle for three windows, the "previous" count must be zeroed, not carried over. I would write a conformance test suite that runs every implementation against the same scenario list — burst, refill, backward clock, boundary, idle gap — because the interface exists precisely so that test can be shared.

**Q14. Is the limiter a singleton?**
No, and deliberately. "The" rate limiter is really "a limiter for this scope": per-IP, per-user, per-tenant, global, per-endpoint. Making it a singleton hides that scoping decision and makes tests share state, so test order starts to matter. I construct the set of limiters once at startup and inject them, which gives the same operational reality with none of the test pathology. Same reasoning as rejecting a singleton `RuleEngine` in [the chess problem](09-chess-game.md) §9.

**Q15. How would you add weighted costs and get the header right?**
`cost` is already the parameter, so the mechanism is unchanged; what is needed is the *policy*. A configuration map from (method, path) to cost, or for LLM endpoints a token count. The two design points: the natural unit for an LLM endpoint is tokens, not requests; and tokens are only known *after* the response, which forces reserve-then-settle. For the headers, `RateLimit-Remaining` should be reported in the same unit as the limit, so if the limit is in tokens the remaining must be tokens — a mixed-unit header is worse than no header.

**Q16. What is the deny cache for, and what is its risk?**
It collapses retry storms. When a client is rejected it typically retries immediately, and each retry is a full limiter evaluation — decay, lock, decide — for a request whose answer is already known. A short-lived map from key to "denied until" turns thousands of evaluations into one. The risk is staleness: if the limit is raised, the cache keeps denying for its TTL. So it must be short — about a second — and it must be a *cache*, meaning losing it is always safe. Bounded staleness, never unbounded.

---

### Pushback round

**P1. "Four implementations of the same thing is over-engineering. Pick the sliding counter and move on."**
For a single production limiter, picking one is correct and I would say so — the sliding counter is a good default. The four exist here because the *interview question* is "design a rate limiter", and the thing being assessed is whether I can articulate the memory-versus-precision axis and the burst-versus-smooth axis. That said, in a real service I would still ship two: the counter for the request path, and the log behind a flag for small-limit endpoints like a login throttle where the limits are tiny (5 per minute) and exactness at the boundary is worth the memory. Two is a justified count; four is a menu.

**P2. "You keep state in a Python dict per key. In a real service that is per-instance and therefore wrong."**
It is wrong the moment there is more than one instance, and I said that in the trade-offs and in Q11 rather than hiding it. The defence is that this is the LLD treatment — the algorithm, its state and its thread-safety — and the distributed view lives in [the gateway file](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md) and in the Lua script I included. If the interviewer wants the distributed answer *here*, I would replace `TokenBucket` with `RedisTokenBucket`, which is a new implementation of the same interface, and then the interesting conversation becomes fail-open versus fail-closed and the round-trip cost on the admission path — not the class design.

**P3. "Your `CompositeLimiter` has a real correctness hole and you shipped it anyway."**
I shipped it with the hole documented, because the alternative is more expensive and is only justified for a subset of limits. The hole is that a request may spend budget on limiters it passes before failing one it does not. For abuse ceilings — per-IP, per-user — over-counting by one on a rejected request is harmless and self-correcting: the client is being rejected anyway. For a billing quota it is a real overcharge, and there the correct answer is reserve/commit, which requires every sub-limiter to support a release. I would rather state the condition under which my design is wrong than pretend it is universally right.

**P4. "Your sliding-window counter is an approximation. Justify admitting a request that puts the client over the limit."**
The honest framing is that *every* rate limiter is an approximation of an intent, and the intent is "limit this client to roughly N per window". The fixed window is a bad approximation (2× at the boundary). The sliding counter is a good one, because it weights the previous window by how much of it is still inside the sliding view, which is exactly the right correction — the error is bounded by the assumption that the previous window's traffic was uniform. Cloudflare's measured error on production traffic is around 0.003%. The exact alternative costs O(limit) memory per key, which for most limits is a worse engineering decision than a 0.003% admission error. I would take the approximation and *measure* the error against the log implementation on my own traffic, which is the practical resolution.

**P5. "You made the clock a port but every method reads it directly. Why not pass `now` into `allow`?"**
That is a legitimate alternative and it has a real advantage: the caller can batch several limiter checks at one timestamp, which matters for `CompositeLimiter` — four sub-limiters reading the clock four times can straddle a window boundary. The reason I keep the clock injected rather than passed is that it puts the responsibility on every caller, and a caller who forgets passes `time.time()` and reintroduces the wall-clock bug the port exists to prevent. The compromise I would actually implement is both: `allow(key, cost, now=None)` where `None` means "read the injected clock", and `CompositeLimiter` passes one timestamp through to all sub-limiters. That gets the consistency without making the default unsafe.

---

## 12. Cheat-sheet summary + cross-links

| Concept | One-line truth |
|---|---|
| The interface | `allow(key, cost) -> Decision`. **Decide and consume in one call** |
| Token bucket | `(tokens, updated)` per key, lazy refill — **O(1) memory, bursts up to capacity** |
| Leaky bucket | `(level, updated)` per key, queue semantics — O(1) memory, **smooths, never bursts** |
| The duality | Token bucket as a meter *is* a leaky bucket; the difference is the promise to the downstream |
| Fixed window | One counter — simplest, and **2× burst at the boundary** |
| Sliding-window log | Deque of timestamps — **exact**, O(limit) memory per key, the reference implementation |
| Sliding-window counter | Two integers weighted by progress — **O(1) memory, ~0.003% error** |
| The axis | Memory versus precision — that is the entire design decision |
| The race | Read-modify-write on per-key state; compute the decay **inside** the lock |
| Stripe it | `KeyedLocks` — a global lock serialises every client; a lock per key is unbounded |
| Clock | **Injected port**, `monotonic()`, clamp `elapsed >= 0` so a rewind cannot mint tokens |
| `cost > capacity` | `retry_after = inf` — a permanent rejection must look permanent |
| `retry_after = 0` | Floor `Retry-After` at 1 second or clients thrash |
| Key space | Attacker-controlled memory: cap the keys, evict, `EXPIRE` in Redis |
| Sweep | **Never on the request path**; O(keys) work per request is a cliff |
| Composite | All must allow; partial consumption is fine for abuse limits, wrong for billing |
| Distributed | Limits **multiply by instance count** — Redis + Lua, hash tags, `redis.call('TIME')` |
| Lua details | `EXPIRE` every write, handle `NOSCRIPT`, one slot via hash tags |
| Fail mode | Fail-open with a local fallback, or fail-closed — **choose explicitly** |
| `peek` | For display only; never the admission path |
| Deny cache | Collapses retry storms; cap the TTL at ~1s so limits can be raised |
| Patterns | Strategy (algorithms), Decorator (guards, deny cache, composite), Port (clock), Value Object (decision) |

**The three sentences to memorise**

1. *"The four algorithms differ only in their per-key state and its memory cost — token bucket and leaky bucket are O(1) and burst or smooth, the log is exact and O(limit), the counter is O(1) and approximate."*
2. *"Every one of them is a read-modify-write, so the decay must be computed inside the lock — and the clock must be injectable so I can test the backwards-clock case."*
3. *"In-process state multiplies the limit by the instance count; the distributed fix is one Lua script that decays, decides and consumes atomically, using the Redis clock."*

### Cross-links

- **The distributed view, owned by this file's counterpart** (gateway placement, keying strategy, tiers and quotas, `429`/`Retry-After` semantics, fail-open versus fail-closed, hot keys in the limit store): [API Gateway and Rate Limiting](../../HLD/02-networking/04-api-gateway-and-rate-limiting.md) — §3.3 algorithms, §3.4 Redis + Lua, §3.5 fail-open, §3.7 keying, §6.5 limits that multiply
- **The Redis primitives used by the Lua script:** [Redis Deep Dive](../../HLD/03-scaling/03-redis-deep-dive.md) §3.10 (Redis as a rate limiter) · §3.9 (locks and why `SET NX` is the same primitive)
- **The same "finite budget with atomic consumption" shape:** [Vending Machine](10-vending-machine.md) §5 · [Pub-Sub System](13-pub-sub-system.md) (consumer-side flow control)
- **Idempotency and the reserve/settle pattern mentioned in Q12 and §7.2:** [URL Shortener LLD](../../Case-Studies/LLD/01-url-shortener-lld.md) · [Notification System LLD](../../Case-Studies/LLD/05-notification-system-lld.md)
- **Pattern catalogue, concurrency primitives table, and the classic-problems list:** [LLD Cheatsheet](../../Cheat-Sheets/07-lld-cheatsheet.md)
- **Question bank:** [LLD Questions](../../Interview-Questions/08-lld-questions.md) — Part E covers the rate limiter
