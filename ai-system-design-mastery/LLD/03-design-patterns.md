# Design Patterns — The 16 That Matter, Their Costs, and When NOT to Use Them

> **Grounding note.** The source transcript names design patterns in exactly one place, at **[25:54]–[27:38]**: LLD is *"when you don't look at OOPS concepts... you come to a class... you draw a class diagram, define the methods, apply the OOP concept."* It does not name a single pattern, and the reference repositories contain no OOP or LLD material. **Every pattern here is standard industry practice.** The "use in this repo" examples are drawn from systems this knowledge base already documents, so the theory lands somewhere concrete. Nothing in this file is transcript-grounded except the framing sentence.

---

## TL;DR

A design pattern is **a named solution to a recurring design problem, together with its cost.** It is not a goal, and it is not a decoration. There are three families: **creational** (how objects get made), **structural** (how objects are composed), and **behavioural** (how objects communicate and how responsibility is distributed). The sixteen patterns worth knowing cold are: Singleton, Factory Method, Abstract Factory, Builder, Prototype; Adapter, Decorator, Facade, Proxy, Composite; Observer, Strategy, State, Command, Template Method, Iterator, Chain of Responsibility. The single sentence that separates a senior answer from a junior one is: **"I'd introduce this pattern here because the problem is X, and it costs me Y."** A pattern named without a problem and a cost is worse than no pattern at all, because it adds indirection and demonstrates only vocabulary.

> **60-second interview answer:** "Patterns are named solutions to recurring design problems, and I use them as a vocabulary for justified structure rather than as a checklist. In the parking lot, pricing varies independently of everything else, so it becomes a Strategy that I inject — the cost is an interface and a wiring point, and the benefit is that a new pricing model edits nothing. Gate notification is Observer, because the lot should not know which gates exist. A seat's lifecycle is State, because the legal transitions are the design. A prompt with twelve optional parts is Builder. A guardrail pipeline is a Chain of Responsibility, because each stage can short-circuit. And I will say out loud when I am *not* using a pattern: a three-case `if` over vehicle types that never changes is better than a factory, and an interface with one implementation and no test double is speculation with a maintenance cost."

---

## 1. Definition & mental model

### 1.1 What a pattern actually is

A pattern has four parts, and an interview answer that omits the third and fourth is incomplete:

| Part | Meaning | Parking-lot example |
|---|---|---|
| **Name** | Shared vocabulary | "Strategy" |
| **Problem** | The recurring situation | Pricing varies by vehicle class, time of day and demand, independently of the lot's structure |
| **Solution** | The structure that resolves it | An interface `PricingStrategy` with `calculate(ticket, exit_time)`, injected into `ParkingLot` |
| **Cost** | What you paid | An interface, a wiring point, one more indirection in the exit path, and a composition root that must be told which strategy to use |

### 1.2 The mental model: patterns are a vocabulary, not a plan

Think of patterns the way a carpenter thinks of joint names. "Dovetail joint" is not the goal of building a drawer — it is the name for a way of joining two boards that is strong under one kind of load and weak under another. Naming it lets two carpenters agree in one word. **Using a dovetail where a screw would do is not craftsmanship; it is showing off.**

Three consequences:

- **Patterns are discovered, not applied.** You find the axis of change, and the pattern names what you already needed.
- **Every pattern has a counter-pattern.** The naive version is sometimes correct, and saying so scores points.
- **The patterns overlap.** Strategy and State have *identical* structure and different *intent*. Command and Strategy look alike. Decorator, Proxy and Chain of Responsibility are all "an object wrapping an object of the same interface". Understanding what distinguishes them is the actual knowledge.

### 1.3 The three families, and the question each answers

```
  CREATIONAL   "How does this object come into existence?"
       │         Singleton · Factory Method · Abstract Factory
       │         Builder · Prototype
       ▼
  STRUCTURAL   "How is this object composed with others?"
       │         Adapter · Decorator · Facade · Proxy · Composite
       ▼
  BEHAVIOURAL  "How do objects communicate, and who holds responsibility?"
                 Observer · Strategy · State · Command
                 Template Method · Iterator · Chain of Responsibility
```

### 1.4 What a pattern is *not*

| Not this | Why the confusion exists | The distinction |
|---|---|---|
| **A library or a framework** | Patterns are named like APIs | A pattern is a *shape of code you write*, not code you import |
| **A way to make code shorter** | Patterns often reduce duplication | Patterns almost always add lines. They buy *changeability*, not brevity |
| **Something you should use as many of as possible** | Interviews mention them | Code with twelve patterns in 500 lines is unreadable. Restraint is the senior signal |
| **Interchangeable with an algorithm** | "Strategy" sounds like an algorithm | An algorithm is *what is computed*; the pattern is *how the choice is structured* |
| **Language-independent in mechanism** | The names are shared | The *structure* transfers; the *idiom* does not. Go has no inheritance; Python has no `interface` keyword |

---

## 2. Why it exists — the problem it solves

### 2.1 The five problems patterns exist to solve

| Problem | Which patterns address it |
|---|---|
| A choice must be made at runtime, and the choice set grows | **Strategy**, **State**, **Factory Method**, **Abstract Factory** |
| One object must notify many others without knowing them | **Observer**, **Chain of Responsibility** |
| Construction is complex: many optional parts, or validation that must happen before the object is usable | **Builder**, **Prototype** |
| An interface does not match what you need | **Adapter**, **Facade**, **Proxy**, **Decorator** |
| A request must be turned into an object so it can be queued, logged, undone | **Command** |
| A structure is recursive (a tree of the same thing) | **Composite**, **Iterator** |
| A fixed algorithm has variable steps | **Template Method** |
| Exactly one instance must exist process-wide | **Singleton** — the most over-used pattern in the catalogue |

### 2.2 What breaks without them

- **Every new case edits a central function.** Feature N has N−1 opportunities to break something existing.
- **Objects know too much.** The lot knows about gates, email, SMS and analytics; each new channel edits the lot.
- **Construction is scattered.** A fifteen-argument constructor called from nine places, six of which pass the same defaults.
- **Nothing is substitutable.** Tests need live infrastructure because there is no seam to insert a fake.
- **Cross-cutting concerns are copy-pasted.** The retry, the metric, the log and the auth check appear in every method.

### 2.3 The honest counter-argument, which you should say out loud

**Most patterns are a deferred cost, not a free win.** Each one buys changeability with indirection: an interface, a class, a wiring point and a hop in the debugger. Applied to a problem whose axis of change is *imaginary*, the pattern is pure cost. The 1990s-era "pattern everything" culture produced codebases of one-method classes and four layers of interfaces, and the industry's reaction — "patterns are overrated" — is itself an over-correction.

**The position to take in an interview:** patterns are introduced when the *second* implementation, the *second* channel, or the *second* variation is real or imminent, and the cost is stated every time. In the parking lot, pricing and allocation qualify (the interviewer is about to ask for a second policy, and real lots have several). Gate hardware does not qualify. Say both.

---

## 3. Internal mechanics — how patterns relate to each other

The most useful diagram in this file is not any single pattern's structure — it is the map of how they overlap, because interviewers probe exactly the confusable pairs.

```
                          «interface» Handler
                                  ▲
                                  │
        ┌─────────────┬───────────┼────────────┬──────────────┐
        │             │           │            │              │
    Decorator       Proxy     Composite   ChainOfResp.    Adapter
        │             │           │            │              │
   same interface  same iface  same iface  same iface    DIFFERENT interface
   + adds behaviour + controls  + is a tree  + passes along  (translates)
                     access                   until handled  


                    «interface» Behaviour
                            ▲
              ┌─────────────┴─────────────┐
              │                           │
          Strategy                      State
    (client picks it)          (the object changes its OWN
    (stateless strategy)        behaviour, by changing its
                                reference on transition)


      CREATION PIPELINE

   Singleton ─── the instance itself is global
   Factory Method ── one method decides which concrete type
   Abstract Factory ─ several related factories, chosen as a family
   Builder ───────── step-by-step assembly of one complex object
   Prototype ─────── clone an existing instance instead of constructing
```

**The three confusable families, stated precisely:**

| Pair | Identical? | What distinguishes them |
|---|---|---|
| **Strategy vs State** | Structure is identical | **Intent.** In Strategy the *client* chooses the implementation and it usually does not change. In State the *object* changes its own behaviour by swapping its own reference on a transition. Strategy is a configuration; State is a lifecycle |
| **Decorator vs Proxy vs Chain** | All wrap the same interface | **Decorator** adds behaviour and always calls through. **Proxy** controls access and may *not* call through (a lazy proxy, a protection proxy, a caching proxy). **Chain** may handle and stop, or pass to the next |
| **Adapter vs Facade** | Both sit in front of something | **Adapter** changes an interface to *one you already require* (it makes an incompatible thing fit). **Facade** *simplifies* a subsystem into a new, easier interface that nothing previously required |
| **Factory Method vs Abstract Factory** | Both create | **Factory Method** is one method that returns one product. **Abstract Factory** is an object with several factory methods producing a *consistent family* (a `CardFactory` yields card charge + card refund + card tokenizer) |
| **Builder vs Factory** | Both create | **Factory** answers *which* type in one call. **Builder** answers *how to assemble* across several calls, with validation at the end |

---

## 4. Variants & taxonomy

### 4.1 Creational patterns

#### 4.1.1 Singleton

| | |
|---|---|
| **Intent** | Guarantee exactly one instance of a class and provide a global access point to it |
| **Problem it solves** | A genuinely process-wide resource: a configuration registry, a connection pool, a metrics sink, a lock manager |
| **Cost** | Global mutable state; hidden dependencies; test isolation breaks (two tests share one instance); concurrency needs care to make initialisation safe |

```
        ┌────────────────────────────┐
        │        Singleton           │
        │────────────────────────────│
        │ - static instance          │
        │ - private constructor      │
        │────────────────────────────│
        │ + static getInstance()     │
        └────────────────────────────┘
                     ▲
                     │  callers ask for THE instance
        ┌────────────┴────────────┐
        │  everywhere in the app  │
        └─────────────────────────┘
```

```python
import threading
from typing import Optional


class MetricsRegistry:
    """A process-wide metrics sink. Singleton, and honest about why."""

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        if cls._instance is None:
            with cls._lock:                     # double-checked locking
                if cls._instance is None:       # re-check inside the lock
                    instance = super().__new__(cls)
                    instance._counters = {}
                    instance._counters_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def increment(self, name: str, by: int = 1) -> None:
        with self._counters_lock:
            self._counters[name] = self._counters.get(name, 0) + by

    def snapshot(self) -> dict[str, int]:
        with self._counters_lock:
            return dict(self._counters)

    @classmethod
    def reset_for_tests(cls) -> None:
        """Escape hatch so tests do not share state. Without this, a Singleton
        makes the test suite order-dependent — which is the pattern's real cost."""
        with cls._lock:
            cls._instance = None
```

**Use in this repo.** The rate limiter in the API gateway ([HLD/02-networking/04-api-gateway-and-rate-limiting.md](../HLD/02-networking/04-api-gateway-and-rate-limiting.md)) holds per-instance token buckets in a process-wide registry, because the buckets must be shared across every request handler in the process. A Singleton is defensible there.

**When NOT to use it.** Almost always, in application code. A `ParkingLot` Singleton means you cannot instantiate a second lot in a test, and you cannot run two lots in one process — which the "make it multi-lot" follow-up demands. **The pattern to use instead: create one instance at the composition root and pass it in (Dependency Inversion).** You get single-instance behaviour, testability, and no global state. Say exactly that in the interview: *"I want one instance, not a global access point — so I construct it once at startup and inject it."*

#### 4.1.2 Factory Method

| | |
|---|---|
| **Intent** | Define an interface for creating an object, but let subclasses or a configuration decide which concrete type is created |
| **Problem it solves** | The caller must not depend on concrete classes; the choice depends on runtime input |
| **Cost** | One more class; the creation logic is now somewhere else, which can obscure a simple `new` |

```python
from abc import ABC, abstractmethod


class SpotFactory(ABC):
    @abstractmethod
    def create(self, spot_id: str) -> ParkingSpot: ...


class BikeSpotFactory(SpotFactory):
    def create(self, spot_id: str) -> ParkingSpot:
        return ParkingSpot(spot_id=spot_id, size=SpotSize.SMALL)


class CarSpotFactory(SpotFactory):
    def create(self, spot_id: str) -> ParkingSpot:
        return ParkingSpot(spot_id=spot_id, size=SpotSize.MEDIUM)


class EvSpotFactory(SpotFactory):
    """Adding EV spots touches NO existing class — only this one."""

    def create(self, spot_id: str) -> ParkingSpot:
        spot = ParkingSpot(spot_id=spot_id, size=SpotSize.MEDIUM)
        spot.attach_charger(Charger(kw=22))
        return spot


SPOT_FACTORIES: dict[str, SpotFactory] = {
    "BIKE": BikeSpotFactory(),
    "CAR": CarSpotFactory(),
    "EV": EvSpotFactory(),
}


def build_floor(floor_id: int, layout: dict[str, int]) -> Floor:
    spots: list[ParkingSpot] = []
    for kind, count in layout.items():
        factory = SPOT_FACTORIES[kind]
        spots.extend(factory.create(f"{floor_id}-{kind}-{i}") for i in range(count))
    return Floor(floor_id, spots)
```

**Use in this repo.** The notification system ([Case-Studies/HLD/03-notification-system.md](../Case-Studies/HLD/03-notification-system.md)) creates a `Notification` of the right channel type from a template and a channel enum. The agentic tool registry ([HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md)) creates a tool instance by name from a registry — the same shape.

**When NOT to use it.** When there is one product type and the caller knows which it wants. `ParkingSpot(spot_id, size)` is a constructor. A factory that always returns the same class is a constructor with extra steps.

#### 4.1.3 Abstract Factory

| | |
|---|---|
| **Intent** | Provide an interface for creating *families* of related objects without specifying their concrete classes |
| **Problem it solves** | Several products must be consistent with each other (all from the same vendor, all from the same environment) |
| **Cost** | A second axis of classes (one factory per family); heavyweight when the family has one product |

```python
class PaymentFactory(ABC):
    """A family: every method returns products that must be mutually consistent."""

    @abstractmethod
    def charge_processor(self) -> ChargeProcessor: ...

    @abstractmethod
    def refund_processor(self) -> RefundProcessor: ...

    @abstractmethod
    def tokenizer(self) -> CardTokenizer: ...


class StripeFactory(PaymentFactory):
    def charge_processor(self) -> ChargeProcessor: return StripeCharger()
    def refund_processor(self) -> RefundProcessor: return StripeRefunder()
    def tokenizer(self) -> CardTokenizer:      return StripeTokenizer()


class RazorpayFactory(PaymentFactory):
    def charge_processor(self) -> ChargeProcessor: return RazorpayCharger()
    def refund_processor(self) -> RefundProcessor: return RazorpayRefunder()
    def tokenizer(self) -> CardTokenizer:      return RazorpayTokenizer()


class IndianProcessorFactory(PaymentFactory):
    """A family that mixes vendors is almost always a bug: a charge created by
    one processor cannot be refunded by another. The Abstract Factory makes that
    class of bug unrepresentable."""

    def charge_processor(self) -> ChargeProcessor: return RazorpayCharger()
    def refund_processor(self) -> RefundProcessor: return RazorpayRefunder()
    def tokenizer(self) -> CardTokenizer:      return RazorpayTokenizer()
```

**Use in this repo.** The LLM serving gateway ([HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md)): a `ProviderFactory` yields a chat client, an embedding client and a tokenizer that must all come from the same vendor, because an embedding produced by one model is meaningless to another. That consistency requirement is exactly what Abstract Factory exists for.

**When NOT to use it.** When the family has one product — then it is a Factory Method wearing a costume. And when mixing is legitimate, the consistency guarantee is a constraint you did not need.

#### 4.1.4 Builder

| | |
|---|---|
| **Intent** | Construct a complex object step by step, so that the same construction process can produce different representations, and so that invalid intermediate objects cannot leak |
| **Problem it solves** | A constructor with 12 optional parameters; a construction that must validate several parts together at the end |
| **Cost** | A second class (the builder) and a `build()` that must be called; more verbose for a simple object |

```python
class QueryBuilder:
    """Builds a paginated, filtered retrieval query. Every part is optional;
    the invariants are checked once, at build() time."""

    def __init__(self) -> None:
        self._table: str | None = None
        self._filters: list[tuple[str, str, object]] = []
        self._order_by: tuple[str, bool] | None = None
        self._limit: int | None = None
        self._offset: int = 0
        self._projection: list[str] = ["*"]

    def from_table(self, table: str) -> "QueryBuilder":
        self._table = table
        return self

    def where(self, column: str, op: str, value: object) -> "QueryBuilder":
        if op not in {"=", "!=", "<", "<=", ">", ">=", "IN", "LIKE"}:
            raise ValueError(f"unsupported operator: {op}")
        self._filters.append((column, op, value))
        return self

    def order_by(self, column: str, descending: bool = False) -> "QueryBuilder":
        self._order_by = (column, descending)
        return self

    def limit(self, n: int) -> "QueryBuilder":
        if n <= 0:
            raise ValueError("limit must be positive")
        self._limit = n
        return self

    def offset(self, n: int) -> "QueryBuilder":
        if n < 0:
            raise ValueError("offset must be non-negative")
        self._offset = n
        return self

    def select(self, *columns: str) -> "QueryBuilder":
        self._projection = list(columns) or ["*"]
        return self

    def build(self) -> tuple[str, tuple]:
        """The single validation point. Everything the object needs is checked
        here, so no half-built query can ever be executed."""
        if not self._table:
            raise ValueError("from_table() is required")
        if self._limit is None:
            raise ValueError("limit() is required — unbounded queries are not allowed")
        sql = f"SELECT {', '.join(self._projection)} FROM {self._table}"
        params: list[object] = []
        if self._filters:
            clauses = []
            for column, op, value in self._filters:
                if op == "IN":
                    placeholders = ", ".join(["%s"] * len(value))   # type: ignore[arg-type]
                    clauses.append(f"{column} IN ({placeholders})")
                    params.extend(value)                            # type: ignore[arg-type]
                else:
                    clauses.append(f"{column} {op} %s")
                    params.append(value)
            sql += " WHERE " + " AND ".join(clauses)
        if self._order_by:
            column, descending = self._order_by
            sql += f" ORDER BY {column} {'DESC' if descending else 'ASC'}"
        sql += " LIMIT %s OFFSET %s"
        params.extend([self._limit, self._offset])
        return sql, tuple(params)


# usage — reads like the sentence it produces
sql, params = (
    QueryBuilder()
    .from_table("bookings")
    .where("show_id", "=", "show-42")
    .where("status", "IN", ["CONFIRMED", "PENDING"])
    .order_by("created_at", descending=True)
    .select("id", "user_id", "amount_minor")
    .limit(50)
    .offset(100)
    .build()
)
```

**Use in this repo.** The RAG pipeline's prompt construction ([HLD/05-ai-systems/01-production-rag-system.md](../HLD/05-ai-systems/01-production-rag-system.md)) — system instruction, retrieved context blocks, few-shot examples, tool schemas, output format constraints — is a builder with many optional parts and one final validation (token budget). The capacity-estimation estimation script's parameter assembly is the same shape.

**When NOT to use it.** For an object with three required fields and no optional ones. `Ticket(id, spot_id, plate, entry_time)` is a constructor; a `TicketBuilder` would be noise. The threshold in practice is **more than four optional parameters, or a cross-field validation that must happen before the object is usable.**

#### 4.1.5 Prototype

| | |
|---|---|
| **Intent** | Create new objects by cloning an existing instance rather than constructing from scratch |
| **Problem it solves** | Construction is expensive (a loaded configuration, a warmed connection, a parsed model) and the new object differs only slightly |
| **Cost** | Deep-copy versus shallow-copy is a real correctness hazard; every cloneable class must implement `clone` correctly, including nested mutable state |

```python
import copy
from dataclasses import dataclass, field


@dataclass
class PricingTemplate:
    """A pre-validated pricing configuration, cloned and tweaked per venue.
    Parsing and validating a tier table is expensive; cloning is cheap."""

    name: str
    base_rate_minor: int
    surge_multiplier: float = 1.0
    weekend_multiplier: float = 1.0
    free_minutes: int = 15
    tier_overrides: dict[str, int] = field(default_factory=dict)

    def clone(self, **overrides: object) -> "PricingTemplate":
        """DEEP copy of the mutable parts, then apply overrides. A shallow copy
        would share tier_overrides between the template and every clone — the
        classic Prototype bug, and it corrupts the template silently."""
        cloned = copy.deepcopy(self)
        for key, value in overrides.items():
            if not hasattr(cloned, key):
                raise AttributeError(f"unknown pricing field: {key}")
            setattr(cloned, key, value)
        return cloned


STANDARD_CAR = PricingTemplate(
    name="standard-car",
    base_rate_minor=2500,
    surge_multiplier=1.0,
    tier_overrides={"PREMIUM": 4000, "RECLINER": 6000},
)

# A weekend surge venue is a clone with two fields changed — no re-parsing.
WEEKEND_SURGE = STANDARD_CAR.clone(name="weekend-surge", surge_multiplier=1.5)
```

**Use in this repo.** Feature-flag and configuration objects in the deployment pipeline ([Operations/04-deployment-release-and-cicd.md](../Operations/04-deployment-release-and-cicd.md)): a base config is validated once and cloned per environment with overrides. Any ML model-serving setup where a loaded model handle is reused rather than reloaded is the same idea.

**When NOT to use it.** When construction is cheap. Cloning a five-field object is more code and more risk than a constructor call. And **be explicit about deep versus shallow** — in Python, mutable default arguments and `copy.copy` are the two most common sources of silent shared-state bugs in this pattern.

### 4.2 Structural patterns

#### 4.2.1 Adapter

| | |
|---|---|
| **Intent** | Convert the interface of a class into another interface the client expects |
| **Problem it solves** | A third-party API, a legacy component, or a vendor SDK whose shape you cannot change |
| **Cost** | One class per adaptee; the adapter can leak the adaptee's quirks (exceptions, pagination style, error codes) if written lazily |

```python
class SeatHoldStore(ABC):
    """The interface OUR code requires."""

    @abstractmethod
    def try_hold(self, show_id: str, seats: list[str], user_id: str,
                 ttl_seconds: int, idempotency_key: str) -> "HoldResult": ...


class RedisSeatHoldStore(SeatHoldStore):
    """ADAPTER — makes Redis's SET NX + TTL look like our domain operation.

    The ugliness stops here. Callers never see SET NX, never see a PX value,
    never see a Redis error code, and never handle the retry.
    """

    _HOLD_SCRIPT = """
    -- KEYS[1] = hold key, ARGV[1] = user, ARGV[2] = ttl ms, ARGV[3] = token
    if redis.call('SET', KEYS[1], ARGV[3], 'NX', 'PX', ARGV[2]) then
      return 1
    end
    return 0
    """

    def __init__(self, redis_client: "Redis", clock: "Clock") -> None:
        self._redis = redis_client
        self._clock = clock
        self._script_sha: str | None = None

    def try_hold(self, show_id: str, seats: list[str], user_id: str,
                 ttl_seconds: int, idempotency_key: str) -> "HoldResult":
        hold_id = f"hold:{show_id}:{idempotency_key}"
        token = uuid.uuid4().hex
        try:
            acquired = self._eval(self._HOLD_SCRIPT, [hold_id],
                                  [user_id, ttl_seconds * 1000, token])
        except RedisError as exc:
            raise HoldStoreUnavailable(str(exc)) from exc      # translate the error type
        if not acquired:
            return HoldResult(ok=False, hold_id=None, conflicting_seats=seats,
                              amount_minor=0)
        return HoldResult(ok=True, hold_id=token, conflicting_seats=[],
                          amount_minor=0)

    def _eval(self, script: str, keys: list[str], args: list[object]) -> object:
        if self._script_sha is None:
            self._script_sha = self._redis.script_load(script)
        try:
            return self._redis.evalsha(self._script_sha, len(keys), *keys, *args)
        except NoScriptError:                       # lost after a failover
            self._script_sha = self._redis.script_load(script)
            return self._redis.evalsha(self._script_sha, len(keys), *keys, *args)
```

**Use in this repo.** Every external boundary in this knowledge base is an Adapter: the payment gateway SDK, the LLM provider client, the vector database client, the SMS provider. The value is that the vendor's error model and pagination style stop at the adapter, so swapping vendors changes one file.

**When NOT to use it.** When you control both sides. If you can change the incompatible interface, change it. Adapter is for when you cannot.

#### 4.2.2 Decorator

| | |
|---|---|
| **Intent** | Attach additional responsibilities to an object dynamically, by wrapping it in an object with the same interface |
| **Problem it solves** | Cross-cutting behaviour that combines in any order: caching, retry, logging, metrics, auth, rate limiting |
| **Cost** | Many small objects; the wrapping order matters and is easy to get wrong; stack traces get deeper |

```python
class GuardrailStage(ABC):
    @abstractmethod
    def check(self, payload: "GuardrailPayload") -> "Verdict": ...


class StageDecorator(GuardrailStage):
    """Base decorator: same interface, holds a wrapped stage."""

    def __init__(self, inner: GuardrailStage) -> None:
        self._inner = inner


class WithLatencyBudget(StageDecorator):
    """DECORATOR — adds a deadline to any stage, without the stage knowing."""

    def __init__(self, inner: GuardrailStage, budget_ms: int) -> None:
        super().__init__(inner)
        self._budget_ms = budget_ms

    def check(self, payload: "GuardrailPayload") -> "Verdict":
        start = time.monotonic()
        try:
            return self._inner.check(payload)
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            payload.metrics.record(f"guardrail.{type(self._inner).__name__}.ms",
                                   elapsed_ms)
            if elapsed_ms > self._budget_ms:
                payload.metrics.increment("guardrail.slow_stage")


class WithAudit(StageDecorator):
    """Another decorator, same interface, independent concern."""

    def __init__(self, inner: GuardrailStage, audit: "AuditLog") -> None:
        super().__init__(inner)
        self._audit = audit

    def check(self, payload: "GuardrailPayload") -> "Verdict":
        verdict = self._inner.check(payload)
        self._audit.record("guardrail.verdict", {
            "stage": type(self._inner).__name__,
            "allowed": verdict.allowed,
            "reason": verdict.reason,
            "request_id": payload.request_id,
        })
        return verdict


# Composition, in order, decided at wiring time:
stage = WithAudit(WithLatencyBudget(PiiRedactor(), budget_ms=50), audit=audit_log)
```

**Use in this repo.** The guardrail pipeline ([HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md)) is the canonical case: the four layers are the *stages*, and latency budgets, audit logging and fail-open/fail-closed policy are the *decorators*. A retrieval pipeline's reranker, deduplicator and citation-attacher wrap the same `retrieve()` interface. HTTP middleware is the same pattern in every web framework you have used.

**When NOT to use it.** When the behaviour must be synchronous and ordering-critical in a way the wrapper cannot express, or when there is exactly one decorator that will never compose. **Decorator versus Chain of Responsibility:** a Decorator *always* calls through and combines results; a Chain may *stop*. If a stage can short-circuit the pipeline, model it as a chain, not a decorator — and say which one you chose and why.

#### 4.2.3 Facade

| | |
|---|---|
| **Intent** | Provide a unified, simplified interface to a set of interfaces in a subsystem |
| **Problem it solves** | A subsystem is complex and has many parts; callers should not have to orchestrate them |
| **Cost** | The facade can become a god class; it hides capability that power users need (so keep the subsystem accessible) |

```python
class BookingFacade:
    """One call, five subsystems. This is the API gateway's job at the HLD level
    and a facade's job at the class level."""

    def __init__(self, holds: SeatHoldStore, pricing: PricingStrategy,
                 payments: ChargeProcessor, repo: "BookingRepository",
                 notifier: "ConfirmationNotifier", clock: "Clock") -> None:
        self._holds = holds
        self._pricing = pricing
        self._payments = payments
        self._repo = repo
        self._notifier = notifier
        self._clock = clock

    def book_seats(self, user_id: str, show_id: str, seats: list[str],
                   idempotency_key: str) -> "BookingReceipt":
        """The whole multi-step flow, behind one method."""
        hold = self._holds.try_hold(show_id, seats, user_id, 600, idempotency_key)
        if not hold.ok:
            raise SeatsUnavailableError(hold.conflicting_seats)

        amount = self._pricing.calculate_for(show_id, seats)
        charge_id = self._payments.charge(amount, user_id, idempotency_key)

        booking = Booking.create(show_id=show_id, seats=seats, user_id=user_id,
                                 hold_id=hold.hold_id, charge_id=charge_id,
                                 now=self._clock.now())
        self._repo.save(booking)
        self._holds.confirm(hold.hold_id, idempotency_key)
        self._notifier.send(booking)
        return BookingReceipt.from_booking(booking)

    # The subsystem stays reachable for the rare caller that needs it:
    @property
    def holds(self) -> SeatHoldStore:
        return self._holds
```

**Use in this repo.** The API gateway ([HLD/02-networking/04-api-gateway-and-rate-limiting.md](../HLD/02-networking/04-api-gateway-and-rate-limiting.md)) is a facade over auth, rate limiting, routing, transformation and observability. A `RagPipeline.answer(question)` facade over embed, retrieve, rerank, generate, cite, guard is the same pattern.

**When NOT to use it.** When the subsystem is simple enough that callers can orchestrate it in two lines — the facade then just hides two lines and adds a file.

#### 4.2.4 Proxy

| | |
|---|---|
| **Intent** | Provide a surrogate or placeholder for another object to control access to it |
| **Problem it solves** | Lazy initialisation, access control, caching, remote calls, reference counting, copy-on-write |
| **Cost** | Extra hop; behaviour can surprise (a caching proxy that returns stale data); the proxy must be transparent or it is a different abstraction |

```python
class LazyModelProxy(ModelProvider):
    """VIRTUAL PROXY — the model is loaded on first use, not at construction.
    Startup is fast, memory is only committed when the model is actually needed."""

    def __init__(self, loader: "Callable[[], ModelProvider]") -> None:
        self._loader = loader
        self._real: ModelProvider | None = None
        self._lock = threading.Lock()

    def _resolve(self) -> ModelProvider:
        if self._real is None:
            with self._lock:
                if self._real is None:              # double-checked: one load only
                    self._real = self._loader()
        return self._real

    def complete(self, prompt: str, max_tokens: int) -> "Completion":
        return self._resolve().complete(prompt, max_tokens)

    def embed(self, text: str) -> list[float]:
        return self._resolve().embed(text)

    @property
    def is_loaded(self) -> bool:
        return self._real is not None


class CachingProviderProxy(ModelProvider):
    """PROTECTION/CACHING PROXY — serves identical prompts from a local cache.
    Transparent unless you look at the hit-rate metric."""

    def __init__(self, inner: ModelProvider, cache: "LocalLruCache", ttl: float) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl = ttl

    def complete(self, prompt: str, max_tokens: int) -> "Completion":
        key = hashlib.sha256(f"{prompt}|{max_tokens}".encode()).hexdigest()
        hit = self._cache.get(key)
        if hit is not None and hit.expires_at > time.monotonic():
            return hit.value
        result = self._inner.complete(prompt, max_tokens)
        self._cache.put(key, CachedCompletion(result, time.monotonic() + self._ttl))
        return result
```

**Use in this repo.** Redis sits in front of the database as a caching proxy on the read path ([Caching strategies and layers](../HLD/03-scaling/02-caching-strategies-and-layers.md) — see [Redis deep dive](../HLD/03-scaling/03-redis-deep-dive.md)). A semantic cache in front of an LLM is a proxy in exactly this sense. An API gateway is a proxy — it is literally its name.

**When NOT to use it.** When the "proxy" is not transparent — when callers must know whether they are talking to the proxy or the real thing. That is an Adapter or a Facade, and calling it a Proxy confuses everyone.

#### 4.2.5 Composite

| | |
|---|---|
| **Intent** | Compose objects into tree structures and treat individual objects and compositions uniformly |
| **Problem it solves** | A recursive structure where a leaf and a group must be handled by the same code: file systems, org charts, UI trees, seat maps, guardrail pipelines |
| **Cost** | Over-general interfaces — a leaf is forced to implement `add()`/`remove()` and must refuse; that refusal is an LSP smell, so split it |

```python
class ParkingNode(ABC):
    """Uniform interface: a spot and a floor are both 'something with capacity'."""

    @abstractmethod
    def capacity(self, size: SpotSize) -> int: ...

    @abstractmethod
    def find_available(self, size: SpotSize) -> ParkingSpot | None: ...


class ParkingSpot(ParkingNode):
    """LEAF."""

    def capacity(self, size: SpotSize) -> int:
        return 1 if (self.size is size and self.is_available) else 0

    def find_available(self, size: SpotSize) -> ParkingSpot | None:
        return self if (self.size is size and self.is_available) else None


class Floor(ParkingNode):
    """COMPOSITE over spots."""

    def __init__(self, floor_id: int, children: list[ParkingNode]) -> None:
        self._floor_id = floor_id
        self._children = children

    def capacity(self, size: SpotSize) -> int:
        return sum(child.capacity(size) for child in self._children)

    def find_available(self, size: SpotSize) -> ParkingSpot | None:
        for child in self._children:
            spot = child.find_available(size)
            if spot is not None:
                return spot
        return None


class ParkingLot(ParkingNode):
    """COMPOSITE over floors — and itself nests arbitrarily deep, which is what
    makes the multi-lot extension free."""

    def __init__(self, lot_id: str, children: list[ParkingNode]) -> None:
        self._lot_id = lot_id
        self._children = children

    def capacity(self, size: SpotSize) -> int:
        return sum(child.capacity(size) for child in self._children)

    def find_available(self, size: SpotSize) -> ParkingSpot | None:
        for child in self._children:
            spot = child.find_available(size)
            if spot is not None:
                return spot
        return None
```

**Use in this repo.** The RAG retrieval pipeline treats a single retriever and a fused multi-retriever pipeline uniformly. A guardrail pipeline is a composite of stages. The seat map is a composite of row → seat. And crucially: **the "now make it multi-lot" extension of the parking lot is Composite plus a name** — a `Campus` of `ParkingLot`s is the same node interface again.

**When NOT to use it.** When the structure is not genuinely recursive. Two levels of nesting is a container, not a composite — and the cost is that leaves must implement methods they cannot honour.

### 4.3 Behavioural patterns

#### 4.3.1 Observer

| | |
|---|---|
| **Intent** | Define a one-to-many dependency so that when one object changes state, all dependents are notified |
| **Problem it solves** | A producer must inform many consumers without knowing who they are or what they do |
| **Cost** | Notification order is unspecified; delivery is usually asynchronous and may be lost; an observer that throws can break the publisher unless each notification is isolated; a long-lived subject with un-removed observers leaks memory |

```python
class OccupancyEvent:
    __slots__ = ("lot_id", "floor_id", "spot_id", "size", "occurred_at")

    def __init__(self, lot_id: str, floor_id: int, spot_id: str,
                 size: SpotSize, occurred_at: float) -> None:
        self.lot_id = lot_id
        self.floor_id = floor_id
        self.spot_id = spot_id
        self.size = size
        self.occurred_at = occurred_at


class OccupancyObserver(ABC):
    @abstractmethod
    def on_occupancy_changed(self, event: OccupancyEvent) -> None: ...


class OccupancyPublisher:
    """The subject. It knows observers exist; it knows nothing about them."""

    def __init__(self) -> None:
        self._observers: list[OccupancyObserver] = []
        self._lock = threading.Lock()

    def subscribe(self, observer: OccupancyObserver) -> "Callable[[], None]":
        with self._lock:
            self._observers.append(observer)
        return lambda: self._unsubscribe(observer)      # returns a disposer

    def _unsubscribe(self, observer: OccupancyObserver) -> None:
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def publish(self, event: OccupancyEvent) -> None:
        with self._lock:
            snapshot = list(self._observers)            # copy: never iterate a live list
        for observer in snapshot:
            try:
                observer.on_occupancy_changed(event)
            except Exception:
                LOG.exception("observer failed; continuing", extra={"observer":
                             type(observer).__name__})
                # Isolation is essential: one bad observer must not stop the others,
                # and must not roll back the lot's state change.


class GateDisplay(OccupancyObserver):
    def on_occupancy_changed(self, event: OccupancyEvent) -> None:
        self._render(self._count_free(event.lot_id, event.size))


class AvailabilityCacheInvalidator(OccupancyObserver):
    def on_occupancy_changed(self, event: OccupancyEvent) -> None:
        self._cache.delete(f"avail:{event.lot_id}:{event.size}")
```

**Use in this repo.** The notification system ([Case-Studies/HLD/03-notification-system.md](../Case-Studies/HLD/03-notification-system.md)) is Observer raised to a distributed system: the event bus is the subject and channel workers are the observers, which is why the delivery guarantee has to be *at-least-once with dedup* rather than in-process fire-and-forget. The feed fan-out in [Instagram](../Case-Studies/HLD/04-instagram-news-feed.md) is the same shape with the same lesson: fan-out means the publisher does not know the consumers, and therefore the publisher cannot guarantee their delivery.

**When NOT to use it.** When the notification order matters, when the caller must not proceed until the observer has finished, or when delivery must be guaranteed. Those are synchronous call, Command, or a durable queue respectively. **Saying this out loud is the difference between knowing the pattern and being able to use it.**

#### 4.3.2 Strategy

| | |
|---|---|
| **Intent** | Define a family of interchangeable algorithms, encapsulate each, and make them substitutable at runtime |
| **Problem it solves** | A policy varies independently of the code that uses it |
| **Cost** | An interface, N classes, and a wiring point; callers must know which strategy to supply |

```python
class RateLimitStrategy(ABC):
    @abstractmethod
    def check(self, key: str, cost: int, now: float) -> "RateLimitDecision": ...


class TokenBucketStrategy(RateLimitStrategy):
    """Allows bursts up to capacity, refills smoothly. The general-purpose choice."""

    def __init__(self, store: "RedisLike", capacity: int, refill_per_sec: float) -> None:
        self._store = store
        self._capacity = capacity
        self._refill = refill_per_sec

    def check(self, key: str, cost: int, now: float) -> "RateLimitDecision":
        return self._store.eval_token_bucket(
            key, self._capacity, self._refill, cost, now
        )


class SlidingWindowStrategy(RateLimitStrategy):
    """Exact over a moving window; costs more memory (one entry per request)."""

    def __init__(self, store: "RedisLike", window_seconds: int, limit: int) -> None:
        self._store = store
        self._window = window_seconds
        self._limit = limit

    def check(self, key: str, cost: int, now: float) -> "RateLimitDecision":
        return self._store.eval_sliding_window(
            key, self._window, self._limit, cost, now
        )


class CompositeStrategy(RateLimitStrategy):
    """COMPOSITION, not inheritance: two strategies must BOTH allow."""

    def __init__(self, *strategies: RateLimitStrategy) -> None:
        self._strategies = strategies

    def check(self, key: str, cost: int, now: float) -> "RateLimitDecision":
        for strategy in self._strategies:
            decision = strategy.check(key, cost, now)
            if not decision.allowed:
                return decision          # first denial wins; report that reason
        return RateLimitDecision(allowed=True, retry_after_ms=0, reason="ok")
```

**Use in this repo.** Rate-limiting algorithms ([HLD/02-networking/04-api-gateway-and-rate-limiting.md](../HLD/02-networking/04-api-gateway-and-rate-limiting.md)) — fixed window, sliding window, token bucket, leaky bucket — are four strategies over one interface, and a real gateway composes them (a global bucket *and* a per-endpoint sliding window). Cache eviction policies ([problems/02-lru-lfu-cache.md](problems/02-lru-lfu-cache.md)) are the same shape. Pricing in the parking lot, allocation in the parking lot, and retrieval fusion in the RAG pipeline are all Strategy.

**When NOT to use it.** When there is one algorithm and no second is coming. And when the "strategy" is chosen once at startup and never changes, a `bool` or an enum switch inside one function is often clearer — the pattern's benefit (runtime substitution) is not being used.

#### 4.3.3 State

| | |
|---|---|
| **Intent** | Allow an object to alter its behaviour when its internal state changes, by delegating to a state object |
| **Problem it solves** | A lifecycle with illegal transitions, where every method must first check "am I in a valid state for this?" |
| **Cost** | One class per state; more objects; overkill for two states — and the transition table becomes hard to see if it is spread across state classes instead of being centralised |

```python
class SeatState(ABC):
    """Each state answers exactly one question: 'can this operation happen now,
    and what state do we become?' The transition table is the design."""

    @abstractmethod
    def hold(self, seat: "SeatModel", user_id: str, ttl: float) -> None: ...

    @abstractmethod
    def confirm(self, seat: "SeatModel") -> None: ...

    @abstractmethod
    def release(self, seat: "SeatModel") -> None: ...

    @abstractmethod
    def name(self) -> str: ...


class FreeState(SeatState):
    def hold(self, seat: "SeatModel", user_id: str, ttl: float) -> None:
        seat.held_by = user_id
        seat.hold_expires_at = seat.clock.now() + ttl
        seat.transition_to(HeldState())

    def confirm(self, seat: "SeatModel") -> None:
        raise InvalidTransition("cannot confirm a free seat")

    def release(self, seat: "SeatModel") -> None:
        raise InvalidTransition("cannot release a free seat")

    def name(self) -> str:
        return "FREE"


class HeldState(SeatState):
    def hold(self, seat: "SeatModel", user_id: str, ttl: float) -> None:
        if seat.held_by == user_id:                  # idempotent re-hold: extend it
            seat.hold_expires_at = seat.clock.now() + ttl
            return
        raise SeatTakenError(f"seat held by {seat.held_by}")

    def confirm(self, seat: "SeatModel") -> None:
        if seat.is_hold_expired():
            seat.transition_to(FreeState())
            raise HoldExpired("hold expired before payment confirmed")
        seat.held_by = None
        seat.transition_to(BookedState())

    def release(self, seat: "SeatModel") -> None:
        seat.held_by = None
        seat.hold_expires_at = None
        seat.transition_to(FreeState())

    def name(self) -> str:
        return "HELD"


class BookedState(SeatState):
    def hold(self, seat: "SeatModel", user_id: str, ttl: float) -> None:
        raise SeatTakenError("seat is already booked")

    def confirm(self, seat: "SeatModel") -> None:
        return                                       # idempotent: confirming again is a no-op

    def release(self, seat: "SeatModel") -> None:
        raise InvalidTransition("booked seats are released by cancellation, not release()")

    def name(self) -> str:
        return "BOOKED"


class SeatModel:
    def __init__(self, seat_id: str, clock: "Clock") -> None:
        self.seat_id = seat_id
        self.clock = clock
        self.held_by: str | None = None
        self.hold_expires_at: float | None = None
        self._state: SeatState = FreeState()

    def transition_to(self, state: SeatState) -> None:
        LOG.info("seat %s: %s → %s", self.seat_id, self._state.name(), state.name())
        self._state = state

    @property
    def state_name(self) -> str:
        return self._state.name()

    # Every public operation delegates. No if/elif chain, and no illegal
    # transition is expressible — each state simply does not offer it.
    def hold(self, user_id: str, ttl: float) -> None:
        self._state.hold(self, user_id, ttl)

    def confirm(self) -> None:
        self._state.confirm(self)

    def release(self) -> None:
        self._state.release(self)

    def is_hold_expired(self) -> bool:
        return (self.hold_expires_at is not None
                and self.clock.now() >= self.hold_expires_at)


# The transition table, readable at a glance:
#
#   STATE    hold            confirm         release
#   FREE     → HELD          ✗ error         ✗ error
#   HELD     same user: ext  → BOOKED        → FREE
#            other: error    (or expired→FREE)
#   BOOKED   ✗ error         no-op           ✗ error (use cancel())
```

**Use in this repo.** The seat-booking lifecycle ([problems/04-bookmyshow-seat-locking.md](problems/04-bookmyshow-seat-locking.md)) — FREE, HELD, BOOKED, CANCELLED — is the canonical case, and it is why the BookMyShow case study ([Case-Studies/HLD/07-bookmyshow-seat-booking.md](../Case-Studies/HLD/07-bookmyshow-seat-booking.md)) insists the display can be stale but the hold cannot: the hold is the state machine. The order lifecycle, the circuit breaker in [reliability patterns](../Operations/07-reliability-patterns.md), and the agent loop in [agentic system design](../HLD/05-ai-systems/06-agentic-system-design.md) are all State machines.

**When NOT to use it.** For two states. A boolean is correct, and a state-class hierarchy for `enabled/disabled` is over-engineering. The threshold: **when the transition table has more than three states, or when at least two operations are illegal in at least two states.** And when the transitions are more important to see at a glance than the behaviour, keep a single class with a transition table and guard methods rather than distributing states into classes.

#### 4.3.4 Command

| | |
|---|---|
| **Intent** | Encapsulate a request as an object, so requests can be parameterised, queued, logged, retried, undone |
| **Problem it solves** | An action must survive past the call that requested it: queued, persisted, retried, audited, reversed |
| **Cost** | A class per command; the undo path must be designed at the same time or it never works |

```python
class Command(ABC):
    @abstractmethod
    def execute(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...

    @abstractmethod
    def idempotency_key(self) -> str: ...


class HoldSeatsCommand(Command):
    def __init__(self, holds: SeatHoldStore, show_id: str, seats: list[str],
                 user_id: str, ttl: int, key: str) -> None:
        self._holds = holds
        self._show_id = show_id
        self._seats = seats
        self._user_id = user_id
        self._ttl = ttl
        self._key = key
        self._hold_id: str | None = None

    def execute(self) -> None:
        result = self._holds.try_hold(self._show_id, self._seats, self._user_id,
                                      self._ttl, self._key)
        if not result.ok:
            raise SeatsUnavailableError(result.conflicting_seats)
        self._hold_id = result.hold_id

    def undo(self) -> None:
        """The compensating action. Without this, the command is not reversible
        and cannot participate in a saga."""
        if self._hold_id is not None:
            self._holds.release(self._hold_id)

    def idempotency_key(self) -> str:
        return self._key


class CommandInvoker:
    """Holds the queue, the retry policy and the history. The commands know
    nothing about any of it."""

    def __init__(self, max_retries: int = 3) -> None:
        self._history: list[Command] = []
        self._max_retries = max_retries

    def run(self, command: Command) -> None:
        for attempt in range(self._max_retries):
            try:
                command.execute()
                self._history.append(command)
                return
            except TransientError:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(0.05 * 2 ** attempt)

    def rollback_all(self) -> None:
        """Undo in reverse order — the saga's compensation phase."""
        for command in reversed(self._history):
            try:
                command.undo()
            except Exception:
                LOG.exception("compensation failed; needs manual reconciliation")
        self._history.clear()
```

**Use in this repo.** The payment saga in [BookMyShow](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) — hold, then charge, then confirm, with compensation on failure — is Command plus a saga. Every job on a queue is a Command: serialising it to a topic is exactly turning a request into an object that outlives the request. Audit trails and "undo" in a collaborative editor are the same shape.

**When NOT to use it.** For a synchronous call whose result the caller needs immediately and which never needs to be retried, queued or reversed. A `Command` wrapping a method that returns a value is usually a Strategy or just a method.

#### 4.3.5 Template Method

| | |
|---|---|
| **Intent** | Define the skeleton of an algorithm in a base class, letting subclasses override specific steps without changing the structure |
| **Problem it solves** | Several variants share the same *sequence* and differ only in *steps* |
| **Cost** | Inheritance-based, so it couples subclasses to the base forever and is capped at one parent; the "fragile base class" risk applies; and the algorithm's shape is split across files |

```python
class ReservationFlow(ABC):
    """The skeleton. The ORDER is fixed here and cannot be changed by a subclass —
    which is the whole point: the sequence is the policy."""

    def reserve(self, request: "ReservationRequest") -> "ReservationResult":
        self.validate(request)                       # step 1 — subclass supplies
        if not self.is_eligible(request):            # hook — subclass may override
            return ReservationResult.rejected("not eligible")
        resource = self.acquire(request)             # step 2 — subclass supplies
        try:
            confirmation = self.persist(request, resource)   # step 3
        except Exception:
            self.compensate(request, resource)       # step 4 — subclass supplies
            raise
        self.notify(request, confirmation)           # step 5
        return ReservationResult.ok(confirmation)

    @abstractmethod
    def validate(self, request: "ReservationRequest") -> None: ...

    @abstractmethod
    def acquire(self, request: "ReservationRequest") -> "Resource": ...

    @abstractmethod
    def persist(self, request: "ReservationRequest",
                resource: "Resource") -> "Confirmation": ...

    @abstractmethod
    def compensate(self, request: "ReservationRequest", resource: "Resource") -> None: ...

    def is_eligible(self, request: "ReservationRequest") -> bool:
        """A HOOK with a default. Subclasses override only if they need to."""
        return True

    def notify(self, request: "ReservationRequest", confirmation: "Confirmation") -> None:
        """A hook with a real default implementation."""
        LOG.info("reservation confirmed: %s", confirmation.id)


class SeatReservationFlow(ReservationFlow):
    def validate(self, request: "ReservationRequest") -> None:
        if not request.seats:
            raise ValueError("at least one seat is required")
        if len(request.seats) > 10:
            raise ValueError("maximum 10 seats per booking")

    def acquire(self, request: "ReservationRequest") -> "Resource":
        hold = self._holds.try_hold(request.show_id, request.seats, request.user_id,
                                    600, request.idempotency_key)
        if not hold.ok:
            raise SeatsUnavailableError(hold.conflicting_seats)
        return hold

    def persist(self, request: "ReservationRequest", resource: "Resource") -> "Confirmation":
        return self._repo.save(Booking.from_hold(resource, request))

    def compensate(self, request: "ReservationRequest", resource: "Resource") -> None:
        self._holds.release(resource.hold_id)


class RoomReservationFlow(ReservationFlow):
    """Same skeleton, different steps. Note that the ORDER of hold/persist/
    compensate is inherited and cannot be got wrong by this subclass."""

    def validate(self, request: "ReservationRequest") -> None:
        if request.check_in >= request.check_out:
            raise ValueError("check-out must be after check-in")

    def is_eligible(self, request: "ReservationRequest") -> bool:
        return not self._blacklist.contains(request.user_id)     # overrides the hook

    def acquire(self, request: "ReservationRequest") -> "Resource": ...
    def persist(self, request: "ReservationRequest", resource: "Resource") -> "Confirmation": ...
    def compensate(self, request: "ReservationRequest", resource: "Resource") -> None: ...
```

**Use in this repo.** The reservation flow is shared verbatim between [BookMyShow seat booking](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) and [hotel management](problems/03-hotel-management.md) — hold, persist, compensate, notify, in that order. The notification delivery path (render → select channel → dispatch → record attempt → retry or dead-letter) has the same fixed skeleton with per-channel steps.

**When NOT to use it.** When the variants differ in *sequence*, not just in steps — then a separate class or a strategy is honest. And when there is only one subclass, the abstract base is not paying for itself. **The modern alternative to say out loud:** *"I could do this with Template Method, but I would probably inject the steps as strategies instead, because that avoids the inheritance coupling and lets me vary the sequence too. I'd use Template Method only if the sequence really is fixed policy."*

#### 4.3.6 Iterator

| | |
|---|---|
| **Intent** | Provide sequential access to the elements of an aggregate without exposing its internal representation |
| **Problem it solves** | Callers must traverse a collection whose internals should stay private, and multiple traversals must be independent |
| **Cost** | Minimal, and usually built into the language. The real risk is a *lazy* iterator that holds a lock, a connection or a snapshot open for its whole lifetime |

```python
class SeatMapIterator:
    """Iterates available seats in a stable, deterministic order, WITHOUT
    exposing the floor/row/seat structure or letting the caller mutate it."""

    def __init__(self, floors: list[Floor], size: SpotSize) -> None:
        self._floors = floors
        self._size = size
        self._floor_idx = 0
        self._spot_idx = 0

    def __iter__(self) -> "SeatMapIterator":
        return self

    def __next__(self) -> ParkingSpot:
        while self._floor_idx < len(self._floors):
            floor = self._floors[self._floor_idx]
            spots = floor._spots                       # internal, exposed to the
            if self._spot_idx >= len(spots):           # iterator only — never the caller
                self._floor_idx += 1
                self._spot_idx = 0
                continue
            spot = spots[self._spot_idx]
            self._spot_idx += 1
            if spot.size is self._size and spot.is_available:
                return spot
        raise StopIteration


# And the lazy-iterator hazard, named explicitly:
def stream_bookings(self) -> "Iterator[Booking]":
    """WRONG if it holds a cursor and a connection for the iteration's lifetime:
    a slow consumer keeps a database connection open and a transaction idle."""
    with self._pool.acquire() as conn:
        with conn.cursor(name="booking_stream") as cur:
            cur.execute("SELECT * FROM bookings WHERE show_id = %s", (self._show_id,))
            for row in cur:
                yield Booking.from_row(row)

def iter_bookings_batched(self) -> "Iterator[Booking]":
    """RIGHT: page through in bounded batches, releasing the connection between
    batches, so a slow consumer cannot hold a connection hostage."""
    offset = 0
    while True:
        with self._pool.acquire() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM bookings WHERE show_id = %s "
                    "ORDER BY id LIMIT %s OFFSET %s",
                    (self._show_id, self._batch, offset),
                )
                rows = cur.fetchall()
        if not rows:
            return
        for row in rows:
            yield Booking.from_row(row)
        offset += self._batch
```

**Use in this repo.** Every repository in this pillar exposes an iterator rather than its internal list. The `SCAN`-based key iteration in the [Redis deep dive](../HLD/03-scaling/03-redis-deep-dive.md) is an iterator over a keyspace precisely because the non-iterating version (`KEYS *`) is an outage.

**When NOT to use it.** When the language's built-in iteration already does the job — in Python, `__iter__` on a list suffices. The pattern is worth naming when you are iterating a *computed* or *remote* sequence, or when you need a snapshot-stable traversal.

#### 4.3.7 Chain of Responsibility

| | |
|---|---|
| **Intent** | Pass a request along a chain of handlers until one handles it |
| **Problem it solves** | A sequence of checks or handlers where each may stop, and the set and order must be configurable |
| **Cost** | A request can fall off the end unhandled (so there must be a terminal handler); order is significant and easy to get wrong; harder to debug than a straight-line function |

```python
class GuardrailStage(ABC):
    def __init__(self) -> None:
        self._next: "GuardrailStage | None" = None

    def set_next(self, nxt: "GuardrailStage") -> "GuardrailStage":
        """Returns nxt so chains can be built fluently: a.set_next(b).set_next(c)"""
        self._next = nxt
        return nxt

    @abstractmethod
    def handle(self, ctx: "GuardrailContext") -> "Verdict": ...

    def pass_to_next(self, ctx: "GuardrailContext") -> "Verdict":
        if self._next is None:
            return Verdict(allowed=True, reason="chain completed")   # terminal default
        return self._next.handle(ctx)


class InputSanitiser(GuardrailStage):
    def handle(self, ctx: "GuardrailContext") -> "Verdict":
        if ctx.user_input.strip() == "":
            return Verdict(allowed=False, reason="empty input")       # SHORT-CIRCUITS
        ctx.sanitised_input = normalise_unicode(ctx.user_input)
        return self.pass_to_next(ctx)


class PromptInjectionDetector(GuardrailStage):
    def handle(self, ctx: "GuardrailContext") -> "Verdict":
        score = self._classifier.score(ctx.sanitised_input)
        if score > self._threshold:
            return Verdict(allowed=False, reason=f"injection score {score:.2f}")
        return self.pass_to_next(ctx)


class PiiRedactor(GuardrailStage):
    def handle(self, ctx: "GuardrailContext") -> "Verdict":
        ctx.sanitised_input = self._redact(ctx.sanitised_input)       # TRANSFORMS
        return self.pass_to_next(ctx)                                 # and continues


class GroundednessChecker(GuardrailStage):
    def handle(self, ctx: "GuardrailContext") -> "Verdict":
        if ctx.retrieved_context and not self._is_grounded(ctx.answer,
                                                           ctx.retrieved_context):
            return Verdict(allowed=False, reason="answer not grounded in context")
        return self.pass_to_next(ctx)


class ActionAuthoriser(GuardrailStage):
    def __init__(self, allowed_tools: set[str]) -> None:
        super().__init__()
        self._allowed = allowed_tools

    def handle(self, ctx: "GuardrailContext") -> "Verdict":
        for call in ctx.tool_calls:
            if call.name not in self._allowed:
                return Verdict(allowed=False, reason=f"tool {call.name} not permitted")
            if call.is_destructive and not ctx.human_approved:
                return Verdict(allowed=False, reason="destructive action needs approval")
        return self.pass_to_next(ctx)


def build_guardrail_chain(human_approval_required: bool) -> GuardrailStage:
    """Order matters and is stated in one place. This function IS the design."""
    head = InputSanitiser()
    head.set_next(PromptInjectionDetector(threshold=0.7)) \
        .set_next(PiiRedactor(patterns=PII_PATTERNS)) \
        .set_next(GroundednessChecker(min_overlap=0.35)) \
        .set_next(ActionAuthoriser(allowed_tools=SAFE_TOOLS))
    return head
```

**Use in this repo.** The guardrail architecture ([HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md)) is a four-layer chain: input, content trust, output, action. The fail-open/fail-closed decision per stage is the chain's most important property and belongs in the design discussion. Middleware stacks, `try/except` ladders in disguise, and approval workflows are the same pattern.

**When NOT to use it.** When the stages are all mandatory and none can short-circuit — that is a Decorator stack or simply a list of calls in a function. **The distinction to state:** a chain *stops* (or the pipeline continues until something rejects); a decorator *always calls through and combines*.

---

## 5. Pros / Cons

### 5.1 Per-pattern: the one-line cost

| Pattern | Buys | Costs | The naive alternative that is sometimes right |
|---|---|---|---|
| **Singleton** | One instance, global access | Global mutable state; untestable; hidden dependency | Construct once at the composition root and inject |
| **Factory Method** | Caller never names a concrete class | A class, and creation logic in a second place | A constructor |
| **Abstract Factory** | Consistent product families | An extra class axis; heavy for one product | A dict of factories |
| **Builder** | Many optional parts; one validation point; readable construction | A second class and a `build()` call | A constructor with defaults, or a dataclass |
| **Prototype** | Avoids expensive re-construction | Deep/shallow copy hazards; every class needs a correct clone | A constructor |
| **Adapter** | Vendor quirks stop at one file | A class per adaptee; leaky adapters | Change the interface you control |
| **Decorator** | Cross-cutting behaviour, composed in any order | Wrapping order matters; deep stacks; harder debugging | A straight-line function with explicit steps |
| **Facade** | One call for a multi-part flow | A god class; hides capability power users need | Callers orchestrate — fine when it is two lines |
| **Proxy** | Lazy load, cache, protect, remote | A hop; surprise staleness; must stay transparent | Call the real object directly |
| **Composite** | Uniform handling of a recursive structure | Leaves must implement methods they cannot honour (LSP smell) | A container plus explicit loops |
| **Observer** | Publisher decoupled from consumers | Unordered, possibly-lossy delivery; leaks if unsubscribed; one bad observer | A direct call when the consumer is known |
| **Strategy** | Runtime-substitutable policy | An interface, N classes, a wiring point | An `if/elif` when there are ≤ 3 cases that never change |
| **State** | Illegal transitions become unrepresentable | A class per state; transition table spread out | A guard method per operation with an enum |
| **Command** | Requests that outlive the call: queue, retry, undo | A class per command; undo must be designed up front | A function call |
| **Template Method** | One fixed sequence, many variants | Inheritance coupling; one parent; base-fragility | Inject the steps as strategies |
| **Iterator** | Traversal without exposing internals | Almost free; the risk is a lazy iterator holding a resource | The language's built-in iteration |
| **Chain of Responsibility** | Configurable, ordered, short-circuiting pipeline | Can fall off the end; order-significant; hard to debug | A straight-line function with early returns |

### 5.2 What patterns do *not* buy

| Claim | Reality |
|---|---|
| "Patterns make code shorter" | They almost always add lines. They buy *changeability*, not brevity |
| "More patterns means better design" | Twelve patterns in 500 lines is unreadable. Restraint is the senior signal |
| "Patterns are language-neutral" | The *structure* transfers; the idiom does not. Go has no inheritance; Python has no `interface` keyword |
| "Patterns are the goal" | A pattern is a consequence of a requirement. If you cannot name the requirement and the cost, do not use it |
| "Strategy and State are different structures" | They are the same structure with different intent. Saying *why* you chose one is the whole answer |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 Singleton

- **Test isolation breaks.** Two tests share one instance and become order-dependent. The fix is a `reset_for_tests()` escape hatch, which is itself an admission that the pattern is fighting the test suite.
- **Hidden dependency.** Nothing in a signature reveals that the class needs the singleton, so the dependency graph lies.
- **Initialisation races.** Lazy initialisation without double-checked locking under a lock can construct two instances. In Python the GIL makes the window narrow but not empty; always lock.
- **Serialisation and cloning.** A singleton that is deserialised or cloned becomes a second instance, silently violating the guarantee.
- **Distributed singleton.** The pattern guarantees one instance *per process*. Across ten pods you have ten. For a real distributed singleton you need a leader election or a lock, not a pattern — see [Redis deep dive](../HLD/03-scaling/03-redis-deep-dive.md).

### 6.2 Factory

- **The `else: raise` branch nobody tests.** A factory with an unregistered key throws in production. Test the unknown-key path.
- **Hidden dependency direction.** The factory now knows every concrete class, which inverts the dependency arrow if the factory lives low in the stack. Keep factories at the composition root.
- **The factory that is a constructor.** If it takes the same arguments as the constructor and returns the same class, delete it.

### 6.3 Builder

- **`build()` not called.** A builder whose product is usable before `build()` leaks half-built objects. Make the product constructible only from `build()`.
- **Builder mutating shared state.** A builder reused across calls carries state forward. Make `build()` reset or make builders single-use.
- **Validation in the wrong place.** If the builder validates eagerly per-setter, cross-field invariants cannot be checked. Validate once, in `build()`.

### 6.4 Prototype

- **Shallow copy sharing mutable state.** The classic bug: cloning a template shares its `dict` or `list`, and mutating the clone corrupts the template. Always deep-copy mutable parts, and test it.
- **Cloning an object with a live resource handle.** Cloning a connection or a file handle gives you two owners of one resource. Exclude non-cloneable members explicitly.
- **`__deepcopy__` performance.** Deep-copying a large object graph can be slower than re-constructing. Measure before assuming the clone is cheaper.

### 6.5 Adapter

- **The leaky adapter.** The vendor's exception type, error code or pagination cursor appears in the domain interface. Callers now depend on the vendor. Translate errors at the boundary, always.
- **Adapter sprawl.** One adapter per method rather than one per adaptee. Group by the interface you require, not by the vendor's resource.
- **Adapter that hides a needed capability.** If two callers need different subsets of a vendor, you need two interfaces (ISP) and possibly two adapters.

### 6.6 Decorator

- **Wrapping order changes behaviour.** A retry decorator outside a transaction decorator retries the whole transaction; inside, it retries one statement. Both are sometimes right, and the wrong one is a silent bug. Document the order in one place.
- **Decorator identity confusion.** `decorated == original` is false; anything comparing identity or using `isinstance` against the concrete type breaks.
- **Deep stacks.** Eight decorators produce eight stack frames per call and an unreadable traceback. Above four or five, consider a pipeline object that holds an ordered list.

### 6.7 Facade

- **The facade becomes the whole system.** Every new feature adds a method; the facade becomes the god class the split was meant to avoid. Keep the subsystem reachable.
- **The facade hides a failure mode.** A facade that swallows a subsystem error and returns a generic failure removes the caller's ability to react. Translate, do not erase.

### 6.8 Proxy

- **A caching proxy serving stale data** with no way to bypass it. Always provide a `refresh`/`no_cache` path.
- **A lazy proxy that is not thread-safe.** Two threads both see `None` and both load. Double-checked locking, as in the code above.
- **A proxy that is not transparent** — the caller must know which object it holds. That is an Adapter or Facade; calling it a Proxy misleads the reader.

### 6.9 Composite

- **Leaves forced to implement `add`/`remove`.** A spot has no children, so `add()` must raise or no-op — an LSP violation. Split into `Component` (read operations) and `Composite` (structural operations), or keep the structural methods on the composite only.
- **Cycles.** A composite that can contain itself creates infinite recursion. Guard with a depth cap or a parent check.
- **Unbounded recursion on a deep tree.** A ten-thousand-deep composite overflows the stack on a recursive traversal. Use an explicit stack in the iterator.

### 6.10 Observer

- **Notification order is unspecified.** Any observer that depends on another having run first is a bug. Order must be explicit, and if it matters, Observer is the wrong pattern.
- **A throwing observer stops the others** unless each notification is individually isolated — as in the code above.
- **Asynchronous delivery is lossy.** In-process fire-and-forget loses the event if the process dies; across a network it is at-least-once at best. In the [notification system](../Case-Studies/HLD/03-notification-system.md), this is why delivery is a durable log with idempotent consumers, not a callback list.
- **Memory leaks.** A long-lived publisher holds strong references to observers. Use weak references or return a disposer and actually call it.
- **The observer holding a lock while notifying.** Deadlock is one line away. Snapshot the observer list under the lock, then notify outside it.

### 6.11 Strategy

- **Strategy explosion.** Twelve strategies for four concepts. Look for a parameterised strategy instead (a rate table passed to one class).
- **Stateless strategies holding state.** A `TokenBucketStrategy` that stores the bucket in an instance field is shared across all requests and is a race. State belongs in the store, not the strategy.
- **The client must know which strategy to supply.** If the client has to reason about which one to use, the selection logic belongs behind a factory.

### 6.12 State

- **The transition table becomes invisible.** Spread across eight state classes, nobody can answer "what are the legal moves?" Put the table in a comment, a test, or a single `TRANSITIONS` dict.
- **Shared state objects with mutable fields.** State objects must be stateless (or shared safely); the data lives on the context. A state object holding per-seat data is a bug.
- **Timeout transitions.** Expiry is a transition triggered by *time*, not by a method call. It must be checked lazily on read **and** swept proactively — one for correctness, one for memory. See [problems/04-bookmyshow-seat-locking.md](problems/04-bookmyshow-seat-locking.md).
- **Forgotten states.** `CANCELLED`, `REFUNDED`, `NO_SHOW` are added later and break every `if state == ...` in the codebase. Polymorphic states absorb a new state as a new class; that is the payoff.

### 6.13 Command

- **Undo designed later never works.** The compensating action must be specified when the command is written, and it must be idempotent.
- **Commands holding live references.** A command serialised to a queue cannot hold a database connection. Commands must carry ids and re-resolve dependencies on execution.
- **Unbounded history.** An undo stack with no cap grows forever. Bound it or persist it.

### 6.14 Template Method

- **Fragile base class.** A base-class change breaks subclasses in distant modules.
- **The base calls an overridable method from its constructor.** The subclass's fields do not exist yet. Never do it.
- **Subclass overriding the skeleton.** If a subclass must override the sequence, the pattern is wrong — use strategies.
- **No hook, so every subclass must implement everything.** Provide sensible defaults for optional steps, or subclasses become boilerplate.

### 6.15 Iterator

- **Concurrent modification.** Iterating a list while another thread mutates it raises or behaves arbitrarily. Snapshot, or use a concurrent structure, or lock.
- **Lazy iterator holding a resource.** A generator holding a database connection, a Redis `SCAN` cursor or a file handle for its whole lifetime. Batch it, as in the code above.
- **Infinite iterators with no termination.** `while True` generators need a consumer-side cap.

### 6.16 Chain of Responsibility

- **Falling off the end unhandled.** A request that reaches the tail with no handler is silently dropped. Always have a terminal handler with an explicit policy.
- **Order dependence.** A PII redactor before or after an injection detector gives different results. State the order as a designed artefact, not an accident of wiring.
- **Fail-open versus fail-closed.** If the injection detector times out, does the request proceed? This is the single most important decision in a guardrail chain, and it is a *policy*, not an implementation detail. See [guardrails](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md).
- **Hard to debug.** "Which stage rejected this?" needs an answer, so the chain must record the verdict of each stage it entered.

### 6.17 The meta-failure: pattern-driven design

**Symptom:** the answer is a list of patterns, and the requirements are stated afterwards to justify them.

**Fix:** requirements first, entities second, patterns last — and only for the axes that genuinely vary. The interview test: **if you cannot state the cost, you have not understood the pattern.**

---

## 7. When to use / When NOT to use

### 7.1 The decision table

| Situation | Pattern | The naive alternative that is sometimes right |
|---|---|---|
| A policy varies at runtime | Strategy | `if/elif`, when ≤ 3 cases and none changes |
| An object's behaviour depends on its own state | State | A status enum plus guards, when there are 2 states |
| Many consumers must learn of a change | Observer | A direct call, when the consumer is known and synchronous |
| A request must be queued, retried or undone | Command | A function call, when none of those is needed |
| Construction has many optional parts | Builder | A dataclass with defaults, when there are ≤ 4 optional fields |
| An interface does not match | Adapter | Change the interface, when you control it |
| Cross-cutting behaviour composes in any order | Decorator | A straight-line function, when there is one behaviour |
| A subsystem must be simplified for callers | Facade | Callers orchestrate, when it is two lines |
| Access to an object must be controlled (lazy, cached, guarded) | Proxy | Direct use, when no access control is needed |
| The structure is recursive | Composite | A container and explicit loops, for one level |
| A sequence of checks each of which may stop | Chain of Responsibility | A function with early returns, when the order is fixed |
| An algorithm's sequence is fixed, steps vary | Template Method | Injected strategies — usually better |
| Traversal must not expose internals | Iterator | The language's built-in iteration |
| Several related products must be consistent | Abstract Factory | A dict of factories, when consistency is not a constraint |
| Exactly one instance is required | **Construct once and inject** | Singleton — only for a genuine process-wide resource |

### 7.2 When NOT to use patterns at all

| Situation | Why | Do instead |
|---|---|---|
| The axis of change is imaginary | You pay indirection for nothing | The concrete class; extract later |
| There is one implementation and no test double | Speculation with a maintenance cost | The concrete class |
| There are ≤ 3 cases and none has changed | The ladder is clearer than the hierarchy | The ladder, with a test per case |
| The system is a throwaway prototype | Patterns slow exploratory code | Write it plain |
| The team does not know the pattern | An unfamiliar pattern is a maintenance liability | The simpler structure, plus a comment |
| The pattern would be applied to a value object | Value objects have no behaviour to vary | A frozen dataclass |
| The performance budget is tight and the indirection is on the hot path | A virtual dispatch per element can matter | Inline it, and say that you measured |

### 7.3 What NOT to do in an LLD round

- **Do not name a pattern before naming the problem.** Problem, then pattern, then cost.
- **Do not list patterns as an answer.** "I'd use Strategy, Factory and Observer" is not a design.
- **Do not use Abstract Factory for one product.** It is a Factory Method in costume.
- **Do not use Singleton for `ParkingLot`.** It breaks the "make it multi-lot" follow-up.
- **Do not use State for a boolean.**
- **Do not skip the cost.** The cost is the answer's credibility.

---

## 8. Numbers, defaults & tuning

| Parameter | Default | Notes |
|---|---|---|
| Patterns in a single LLD answer | **2–4** | More reads as decoration; fewer usually means an axis of change was missed |
| Patterns to name in a 45-min answer | **2–3** | One per genuine axis of change |
| `if/elif` cases before extracting a Strategy | **3** | Three cases that have never changed are fine as a ladder |
| States before extracting a State machine | **> 3**, or ≥ 2 illegal operations | Below that, a status enum with guards |
| Optional constructor params before a Builder | **> 4**, or a cross-field validation | Below that, a dataclass with defaults |
| Decorators in a stack before a pipeline object | **4–5** | Beyond that, an ordered list is clearer |
| Chain length before it needs a debug story | **> 3 stages** | Record each stage's verdict |
| Concrete implementers before introducing an interface | **2** (one real + one test double) | One with no double is speculation |
| Singleton instances per service | **0–1** | Prefer zero: construct at the composition root and inject |
| Observer notification isolation | **Per observer, try/except** | One bad observer must not stop the rest |
| Observer list handling | **Snapshot under lock, notify outside** | Prevents deadlock and concurrent-modification |
| Command history cap | **Bounded** (e.g. 100) or persisted | Otherwise unbounded growth |
| Prototype copy depth | **Deep for mutable members** | Shallow copy is the classic Prototype bug |
| Facade methods before it is a god class | **Watch at ~10** | Keep the subsystem reachable |
| Abstract Factory products | **≥ 2 per family** | One product means Factory Method |
| Pattern names in a 60-second answer | **2–3, each with a cost** | The cost is what makes it credible |

---

## 9. Interview Q&A

**Q1 (easy). What is a design pattern?**
A named solution to a recurring design problem, together with its cost. The name is vocabulary — it lets two engineers agree on a structure in one word. It is not a goal, and it is not a library. The four parts are the name, the problem, the solution and the cost, and an answer that omits the last two is a vocabulary demonstration rather than a design.

**Q2 (easy). Name the three families of patterns.**
Creational — how objects come into existence: Singleton, Factory Method, Abstract Factory, Builder, Prototype. Structural — how objects are composed: Adapter, Decorator, Facade, Proxy, Composite. Behavioural — how objects communicate and where responsibility sits: Observer, Strategy, State, Command, Template Method, Iterator, Chain of Responsibility.

**Q3 (easy). What is the difference between Strategy and State?**
The structure is identical — a context holding a reference to an interface it delegates to. The intent is different. In Strategy the *client* chooses the implementation and it usually stays put for the life of the object; it is a configuration. In State the *object* changes its own behaviour by swapping its own reference on a transition; it is a lifecycle, and the transitions are the design. In the parking lot, pricing is Strategy and a seat's FREE/HELD/BOOKED lifecycle is State.

**Q4 (medium). Where would you use Observer in the systems in this repo?**
The notification system — but I would be careful, and that care is the point. In-process Observer is the right structure for the gate displays and the availability-cache invalidator, because a missed notification is self-healing: the display refreshes, the cache entry has a TTL. It is the *wrong* answer for the notification delivery path, because there a lost event is a bug: the user never gets the email. That path needs a durable log with consumer groups and idempotent consumers, which is Observer's guarantee lifted from "best effort in process" to "at-least-once across a network". Naming that distinction is the difference between knowing the pattern and being able to choose it.

**Q5 (medium). Where would you use Strategy, and what does it cost?**
Rate limiting is the clean case: fixed window, sliding window, token bucket and leaky bucket are four algorithms behind one `check(key, cost, now)` interface, and a real gateway composes them so that both a global bucket and a per-endpoint window must allow. The cost is an interface, four classes and a wiring point, plus the fact that the caller must know which strategy to supply — which is why real systems hide the choice behind a factory. In the parking lot, pricing is Strategy for the same reason: it varies by vehicle class, time and demand, independently of everything else. I would not use it for something with one algorithm and no second one coming.

**Q6 (medium). What is the difference between Decorator and Chain of Responsibility?**
Both wrap an object of the same interface. A Decorator *always calls through* and combines its behaviour with the inner result — a latency-budget decorator measures the inner call and returns its verdict unchanged. A Chain *may stop*: an injection detector that scores above threshold returns a rejection and never calls the next stage. That difference in control flow is the whole distinction, and in the guardrail pipeline I use both — the stages are a chain because they short-circuit, and the latency budget and audit logging are decorators because they never change the verdict.

**Q7 (medium). What is the difference between Adapter and Facade?**
An Adapter makes an incompatible interface fit an interface *you already require* — the Redis seat-hold store adapts `SET NX PX` to `try_hold(...)`, and its job is to stop the vendor's error model and key syntax at the boundary. A Facade *simplifies* a subsystem into a new, easier interface that nothing previously required — a `BookingFacade.book_seats()` that orchestrates hold, price, charge, persist, confirm and notify. Adapter translates; Facade simplifies.

**Q8 (medium). Is Singleton a good idea?**
Rarely, and I would push back on it in most designs. It gives you two things — one instance and a global access point — and the second is the problem. It is global mutable state: hidden dependencies, order-dependent tests, and a guarantee that only holds within one process, so across ten pods you have ten singletons. When I genuinely want one instance, I construct it once at the composition root and inject it. I would use an actual Singleton for something like a process-wide metrics registry, and even then with a `reset_for_tests()` escape hatch so the test suite does not become order-dependent.

**Q9 (medium). Where does Builder fit in an AI system?**
Prompt and query construction. A retrieval query has a table, several optional filters, an ordering, a limit, a projection and a token budget — that is more than four optional parameters and at least one cross-field validation, which is exactly the Builder threshold. The payoff is that the query reads like the sentence it produces, and there is exactly one place where an invalid query can be rejected. The same shape appears in the retrieval pipeline's prompt assembly: system instruction, retrieved context, few-shot examples, tool schemas and output constraints, all optional, all validated once against the token budget.

**Q10 (hard). *Pushback:* "Aren't design patterns just over-engineering? I can write the parking lot without any of them."**
For a lot with one pricing model, one allocation policy and no notifications, you are right, and I would too. Patterns are a deferred cost: each one buys changeability with an interface, a class and a wiring point. So I introduce them only where the axis of change is real. In this design, pricing and allocation qualify — real lots have several of each, and the standard follow-up in this interview is exactly that. Notification qualifies because the lot should not know which gates and caches exist. Gate hardware does not qualify, so it does not get a factory. If you asked me to design a lot for a single-site valet with flat pricing, my answer would be three classes and no patterns, and I would say so.

**Q11 (hard). *Pushback:* "You named Observer for gate notification. Isn't that unreliable?"**
Yes, and that is why I scoped it. In-process Observer gives me decoupling and no ordering or delivery guarantee — the notification is fire-and-forget, an exception in one observer is isolated so it cannot break the publisher, and if the process dies the notification is lost. For gate displays and cache invalidation that is acceptable, because both self-heal: the display refreshes from the lot's state, and the cache entry has a TTL. For anything where a lost event is a bug — a payment confirmation, a booking email — I would not use Observer at all. I would use a durable log with consumer groups and idempotent consumers, and the delivery guarantee becomes at-least-once with dedup rather than best-effort. The pattern is a scope decision, not a universal answer.

**Q12 (hard). *Pushback:* "Your design has an interface for pricing with one implementation. Isn't that premature abstraction?"**
There are two justifications and I would give whichever is true. The first is a real second implementation: the immediate follow-up here is surge or weekend pricing, so the interface is about to earn itself. The second, and the one I would lead with, is testability — a `FixedPricing(100)` implementation makes the fee assertion exact instead of time-dependent, so the abstraction pays for itself in the test suite even before the second production model exists. If neither were true — one implementation, no test double, no second model coming — I would use the concrete class and extract the interface when the second implementation landed. That is the honest threshold: an abstraction is paid for by a second real implementation or by a test double.

**Q13 (hard). What pattern would you use for a seat's lifecycle, and what does it prevent?**
State. The states are FREE, HELD, BOOKED and CANCELLED, and the important thing is not the classes — it is that the *illegal transitions become unrepresentable*. In the naive version, `release()` is a method with four `if` checks, and every new state adds another check to every method. In the State version, `FreeState.confirm()` simply is not an operation that succeeds; it raises a named error. The transition table is the design, and I would write it out as a table in the answer so the interviewer can audit it. The subtlety worth mentioning is the timeout transition: hold expiry is triggered by *time*, not by a method call, so it has to be checked lazily on read for correctness and swept proactively for memory. One handles the correctness, the other handles the leak, and both are needed.

**Q14 (hard). Where does Abstract Factory earn its keep?**
When several products must be mutually consistent. In the LLM serving layer, a provider factory yields a chat client, an embedding client and a tokenizer that must all come from the same vendor — because an embedding produced by one model is meaningless to another, so mixing them is a silent quality bug. That consistency constraint is the pattern's reason to exist; the Abstract Factory makes the mixed state unrepresentable. In the payment layer the same argument holds: a charge created by one processor cannot be refunded by another. Where it does *not* earn its keep is a family with one product — that is a Factory Method wearing a costume.

**Q15 (medium). What is the most common pattern mistake you see?**
Naming a pattern without stating the problem or the cost. "I'd use the Strategy pattern here" tells the interviewer nothing about whether you understand it. "Pricing varies independently of the lot's structure, so I extract a Strategy and inject it; the cost is an interface, one more class and a wiring point, and the benefit is that a new model edits nothing" is the same design with the reasoning attached. The second version cannot be produced by memorisation, which is exactly why it scores.

**Q16 (medium). How do patterns relate to SOLID?**
Patterns are the mechanisms; SOLID is the reasoning. Strategy and dependency injection deliver Open/Closed and Dependency Inversion. Adapter and Facade deliver Interface Segregation and DIP at a boundary. Observer delivers Open/Closed for the notification axis. Template Method delivers Open/Closed inside a family with shared mechanism. Chain of Responsibility delivers OCP for a sequence of stages. So I do not design "with patterns" — I identify which axis of change is real, apply the principle that governs it, and the pattern falls out. Inverting that order is how you get pattern-driven design, which is the failure mode.

---

## 10. Common mistakes candidates make

1. **Naming a pattern without a problem or a cost.** The single most common error.
2. **Listing patterns as the answer.** A pattern list is not a design.
3. **Singleton for the main domain object.** Breaks multi-instance tests and the "make it multi-lot" follow-up.
4. **Abstract Factory for one product.** A Factory Method in costume.
5. **State for a boolean.** A status enum with guards is correct there.
6. **Builder for a four-field object.** The threshold is > 4 optional parameters or a cross-field validation.
7. **Decorator where a chain is needed** (or vice versa) — confusing "always calls through" with "may stop".
8. **Adapter that leaks the vendor's exceptions** into the domain layer.
9. **Observer for delivery that must not be lost.** Use a durable queue with idempotent consumers.
10. **Observer notifying while holding the lock** — a one-line deadlock.
11. **Prototype with a shallow copy.** The template gets corrupted by the clone.
12. **Template Method where strategies would do** — the inheritance coupling is not needed.
13. **No terminal handler in a chain.** Requests fall off the end silently.
14. **Not stating the wrapping order** for decorators or the stage order for a chain.
15. **Over-engineering** — twelve patterns in 500 lines. Restraint is the senior signal.
16. **Never saying "I would not use a pattern here."** The negative answer is the credibility.
17. **Confusing Strategy and State** under questioning, because their structures are identical.
18. **Forgetting the concurrency story for a stateless-looking pattern.** A strategy holding per-request state is a race.

---

## 11. Cheat-sheet summary

**The catalogue, one line each**

| Pattern | Family | One-line truth | Cost |
|---|---|---|---|
| **Singleton** | Creational | One instance, global access | Global state; untestable; per-process only |
| **Factory Method** | Creational | One method decides which concrete type | A class; creation logic elsewhere |
| **Abstract Factory** | Creational | Consistent product *families* | An extra class axis; heavy for one product |
| **Builder** | Creational | Step-by-step assembly, one validation point | A builder class and a `build()` call |
| **Prototype** | Creational | Clone instead of construct | Deep/shallow copy hazards |
| **Adapter** | Structural | Make an incompatible interface fit | A class per adaptee; leaky adapters |
| **Decorator** | Structural | Add behaviour, composed in any order | Order matters; deep stacks |
| **Facade** | Structural | Simplify a subsystem behind one call | God-class risk; hides capability |
| **Proxy** | Structural | Control access: lazy, cache, protect, remote | Extra hop; must stay transparent |
| **Composite** | Structural | Uniform handling of a recursive tree | Leaves must implement what they cannot honour |
| **Observer** | Behavioural | Notify many without knowing them | Unordered, lossy; leaks; one bad observer |
| **Strategy** | Behavioural | Interchangeable algorithms, chosen by the client | Interface, N classes, a wiring point |
| **State** | Behavioural | The object swaps its own behaviour on transition | A class per state; table spread out |
| **Command** | Behavioural | A request as an object: queue, retry, undo | A class per command; undo designed up front |
| **Template Method** | Behavioural | Fixed sequence, variable steps | Inheritance coupling; one parent |
| **Iterator** | Behavioural | Traverse without exposing internals | Lazy iterators holding resources |
| **Chain of Responsibility** | Behavioural | Pass along until handled | Falls off the end; order-significant |

**The confusable pairs**

| Pair | What distinguishes them |
|---|---|
| Strategy vs State | Strategy: client chooses, stays put. State: object swaps itself on a transition |
| Decorator vs Proxy vs Chain | Decorator always calls through · Proxy controls access · Chain may stop |
| Adapter vs Facade | Adapter translates to an interface you already require · Facade simplifies into a new one |
| Factory Method vs Abstract Factory | One product · a consistent family |
| Builder vs Factory | How to assemble across calls · which type in one call |
| Composite vs container | Genuinely recursive · two levels |

**Repo mappings**

| Repo system | Pattern |
|---|---|
| Notification system | Observer in process; durable log with idempotent consumers across the network |
| Rate limiting ([API gateway](../HLD/02-networking/04-api-gateway-and-rate-limiting.md)) | Strategy (four algorithms), composed |
| Seat-booking lifecycle ([BookMyShow](../Case-Studies/HLD/07-bookmyshow-seat-booking.md)) | State (FREE→HELD→BOOKED), Command for the payment saga |
| Query and prompt construction | Builder (many optional parts, one validation) |
| Guardrail pipeline | Chain of Responsibility for stages; Decorator for latency/audit |
| API gateway | Facade over auth, limits, routing, transform, observe |
| Redis in front of the DB | Proxy (caching) |
| Payment and LLM provider families | Abstract Factory (consistency) |
| External vendor SDKs | Adapter |
| Multi-lot parking, RAG retriever fusion | Composite |

**The three rules**

1. **Problem first, then pattern, then always the cost.**
2. **Abstract the axis that changes, not the axis you can imagine changing.**
3. **An abstraction is paid for by a second real implementation or a test double. Otherwise it is cost without benefit.**

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "Why Strategy here?" | "The policy varies independently — and the cost is an interface, a class and a wiring point." |
| "Why not Singleton?" | "I want one instance, not a global access point — construct once at the composition root and inject." |
| "Now add a second pricing model" | "A new `PricingStrategy`. `ParkingLot` does not change — that is OCP through polymorphism." |
| "Isn't that over-engineered?" | "If the second implementation weren't imminent I'd use the concrete class. This axis of change is real." |
| "How do the guards compose?" | "A Chain of Responsibility — each stage may short-circuit; latency and audit are decorators because they never change the verdict." |
| "What if the observer fails?" | "Each notification is isolated; the publisher's state change does not roll back. If the event must not be lost, it is a durable queue, not Observer." |

---

## 12. Cross-links

- Previous in this pillar: [OOP principles and SOLID](02-oop-principles-and-solid.md)
- Next: [Concurrency and threading in LLD](04-concurrency-and-threading-in-lld.md) · [UML and class diagrams](05-uml-and-class-diagrams.md) · [The machine-coding interview framework](06-machine-coding-interview-framework.md)
- The anchor: [What is LLD and how to answer it](01-what-is-lld-and-how-to-answer.md)
- Patterns in running code: [Parking lot](problems/01-parking-lot.md) (Strategy, Factory, Observer, Composite) · [LRU and LFU cache](problems/02-lru-lfu-cache.md) (Strategy for eviction, Proxy for TTL) · [Hotel management](problems/03-hotel-management.md) (State, Template Method) · [BookMyShow seat locking](problems/04-bookmyshow-seat-locking.md) (State, Command, Adapter) · [Elevator system](problems/05-elevator-system.md) (State, Strategy for scheduling) · [Splitwise](problems/06-splitwise.md) (Strategy for splitting) · [ATM machine](problems/07-atm-machine.md) (State, Chain of Responsibility) · [Library management](problems/08-library-management.md) (Strategy, Observer)
- Repo systems used as examples here: [Notification system](../Case-Studies/HLD/03-notification-system.md) · [Instagram news feed](../Case-Studies/HLD/04-instagram-news-feed.md) · [BookMyShow seat booking](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) · [API gateway and rate limiting](../HLD/02-networking/04-api-gateway-and-rate-limiting.md) · [Caching strategies and layers](../HLD/03-scaling/02-caching-strategies-and-layers.md) · [Redis deep dive](../HLD/03-scaling/03-redis-deep-dive.md) · [Guardrails and safety](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) · [LLM serving and gateway](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) · [Agentic system design](../HLD/05-ai-systems/06-agentic-system-design.md) · [MCP, tool use and multi-agent](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) · [Production RAG](../HLD/05-ai-systems/01-production-rag-system.md) · [Reliability patterns](../Operations/07-reliability-patterns.md)
- The boundary these patterns live below: [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md)
- Revise here: [Foundations and estimation cheat sheet](../Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md)
