# Notification System — LLD Case Study

> **This is the LLD bridge for [`03-notification-system.md`](../HLD/03-notification-system.md).** That chapter designs the system at the level of Kafka topics, worker pools, provider integrations and SLOs. This one goes exactly one level down: the classes inside those boxes, the interfaces between them, and the code. **It does not repeat the HLD.** The priority-queue isolation argument, the FCM/APNs payload comparison, the SPF/DKIM/DMARC discussion, the backoff schedules and the per-provider token-bucket numbers all stay in the HLD; they are cross-linked rather than restated. What is here is what the HLD cannot reach: which object decides what, where a lock is held, what a retry looks like as a value, and which of the ten failure sections the code actually prevents.

> **Grounding note.** The reference repositories contain no OOP/LLD material, and the source transcripts name notification systems, preferences and retries without teaching them at class level. The HLD chapter above is the only grounding for this file. Everything below — the class boundaries, the interfaces, the code and its numbers — is worked out here to be defensible in an interview, not recalled from a source. Treat it as *a* design, not *the* design.

---

## TL;DR

**A notification is not a message; it is a decision with a state machine attached.** The pipeline looks like a delivery system — enqueue, render, send, retry — and it is really a policy engine that answers six questions per recipient per channel: *is this person allowed to receive this, on this channel, right now, at this priority, once, and has it already been sent?* Get the policy wrong and the send path is perfect and the product is broken. Get the policy right and the send path is a thin adapter over a provider.

**The one line to remember:** the retry decision is **data, not code** — a table from provider error to `RETRYABLE | TERMINAL | SUPPRESS_TOKEN` — and nearly every notification incident traces back to conflating those three.

---

## The 60-second answer

"The unit of work is a `Notification`: one recipient, one channel, one rendered payload, one row of state. The pipeline is decision → render → enqueue → send → track. The **Decision Service** is the hard part: it resolves recipients, loads preferences and consent, claims a **dedupe key**, evaluates **quiet hours by priority**, applies per-user rate limits, intersects requested channels with allowed channels, and assigns a priority. It writes the state row **before** enqueueing, so an accepted notification is always accounted for. Rendering is a versioned template lookup per channel and locale, and the version is recorded with the notification. Sending goes through a `Channel` strategy that returns either acceptance or a classified failure — and the classification is a **table from error code to retryable/terminal/suppress-token**, because retrying an invalid token burns quota on a message that can never arrive. Retries use an absolute `scheduled_at` with jitter, not 'now plus five minutes'. The webhook handler validates a signature, publishes, and returns 200 in under 100 ms; a consumer updates state. Preferences are evaluated as late as possible, which shrinks the preference-change race to nothing but does not eliminate it, and the honest answer is that it is unavoidable without a distributed transaction."

---

## 1. Requirement clarification

| # | Question to ask | Default to assume if they do not care |
|---|---|---|
| 1 | Is this transactional only, or transactional **and** marketing? | Both, and they must not share a queue or a worker pool |
| 2 | Which channels? | Push, email, SMS; in-app is a read model, not a send channel |
| 3 | Is the producer in this system, or does it hand off and forget? | In: the API returns a `notification_id` the producer can query |
| 4 | What does "delivered" mean here? | Provider acceptance. Push delivery is approximate; say so rather than promising more |
| 5 | At-least-once or exactly-once? | At-least-once delivery, exactly-once **effect** via idempotency and dedupe keys |
| 6 | Do users get quiet hours, and can anything bypass them? | Yes; only an allow-listed set of event types bypasses, never a producer-set priority |
| 7 | Is there a per-user rate limit? | Yes, per category and per day, separate from the provider limit |
| 8 | What happens to a low-priority message when the queue backs up? | It is dropped, not delivered late. A "flash sale ends today" email arriving tomorrow is worse than none |
| 9 | Are digests and scheduled sends in scope? | Yes, as `scheduled_at` with an idempotency key per (user, campaign, local date) |
| 10 | How many producers, and can they be trusted with the priority field? | No. Priority is derived from the event type, not supplied by the caller |

### 1.1 The four constraints that shape every class below

1. **At-least-once is a given, so every effect needs a key.** Two keys, actually, and they are different: an `Idempotency-Key` on the API (the producer retried the POST) and a `dedupe_key` inside the decision service (two different services emitted semantically the same event). The HLD's §3.5 "Deduplication" separates three cases; the first two are classes here, and the third — the same provider message reported twice — is the webhook handler's job.
2. **Isolation is at the worker-pool level, not the message level.** A priority field in one queue is not isolation; the class consequence is that the queue object itself is per priority, and a `Notification` is routed to exactly one of them.
3. **Preferences are evaluated at decision time, as late as possible.** Not at event time. A campaign scheduled for 09:00 local needs the timezone at send time, and a preference change must affect notifications already queued.
4. **A terminal failure and a retryable failure must be different values in the type system.** Not an `is_retryable` comment on an exception. This is the single highest-value modelling decision in the file.

### 1.2 Non-goals

- **Provider SDK behaviour.** The HLD covers APNs' persistent HTTP/2 connection and FCM's error codes; the code here has one `Channel` per provider and a classification table, not an SDK.
- **In-app notification centre.** It is a read model over the state store, and the HLD's delivery-tracking table explains why it is the honest way to show "3 unread". Named in §7, not built.
- **Template authoring and rendering engines.** The renderer takes a template id, a version and a payload and returns a string. HTML, i18n and layout are out of scope; versioning is in.
- **Analytics aggregation.** The sink exists as a port; the warehouse does not.
- **The transport.** Queues here are in-process. The HLD's Kafka topics are the production shape, and the port is the same.

---

## 2. Core entities & responsibilities

| Entity | Responsibility | Not responsible for |
|---|---|---|
| `NotificationRequest` | The producer's input: event type, entity id, recipients or audience, channels, payload | Knowing preferences, priority or dedupe |
| `Notification` | One recipient × one channel, with priority, template version, attempt count and state | Deciding whether it should exist |
| `NotificationState` (enum) | The forward-only lifecycle of one notification | Any policy |
| `Priority` (enum) | HIGH / NORMAL / LOW, and therefore which queue | Being set by the producer |
| `Channel` (ABC) | One provider: how to send, and how to classify what comes back | Choosing which channel, or whether to send |
| `SendOutcome` (value) | ACCEPTED, or a classified failure | Retrying |
| `FailureKind` (enum) | `RETRYABLE`, `TERMINAL`, `SUPPRESS_TOKEN` | Deciding how long to wait |
| `ClassificationTable` | Provider error → `FailureKind`, as data | Owning a channel |
| `UserPreferences` (value) | Per-channel, per-category opt-in, quiet hours window, IANA timezone | Being the store |
| `PreferenceStore` | Load preferences, cached, evaluated late | Deciding what they mean |
| `PreferenceResolver` | Turn `UserPreferences` into `allowed_channels(event_type)` | Loading |
| `ConsentStore` | Legal opt-in records, separate from preferences | Product-level toggles |
| `QuietHours` (policy) | `decide(priority, event_type, local_time) → SEND / DEFER / DROP` | Reading the clock in the user's zone — that is `Timezone` |
| `Timezone` | IANA name → local time and UTC offset, computed at send time | Storing a fixed offset |
| `DedupeStore` | `claim(dedupe_key, user, channel, ttl) → bool`, the first wins | Choosing the key shape |
| `IdempotencyStore` | `claim(producer_key) → Optional[notification_id]` | The semantic dedupe |
| `TemplateStore` | Versioned templates by id, channel and locale | Rendering |
| `TemplateRenderer` | Template + payload + locale → `RenderedMessage` | Version selection policy |
| `RenderedMessage` (value) | Subject/body/payload plus the template version used | Delivery |
| `TokenBucket` | Refill rate, capacity, and `acquire(now) → wait_seconds` | Deciding whether to wait or fail |
| `ProviderLimiter` | One bucket per provider, plus reserved capacity for HIGH | Fairness between priorities |
| `UserRateLimiter` | Per-user, per-category, per-day counts | Provider limits |
| `Scheduler` | `scheduled_at` as an absolute time, idempotent per (user, campaign, local date) | Deciding content |
| `RetryPolicy` | `next_attempt(attempt, priority) → Optional[absolute_time]`, jittered | Sending |
| `DeadLetterQueue` | Terminal failures after N attempts, with the reason | Alerting |
| `NotificationStateStore` | One row per notification, the audit trail, forward-only transitions | Deciding transitions |
| `WebhookHandler` | Validate, translate to `DeliveryEvent`, publish, return fast | Writing the state store synchronously |
| `DeliveryEvent` (value) | Provider message id, level, timestamp, raw reason | Mapping to product states |
| `AnalyticsSink` | Engagements out | Anything in the send path |

**Five classes people leave out, and they are the five that matter:**

- **`ClassificationTable`.** It is the HLD's "most important table", and making it a *table* rather than an `if` in each channel is what stops the retry policy from being re-derived per provider. It is also the thing you extend when a provider adds an error code, without touching a `Channel`.
- **`QuietHours` as a policy with a bypass allow-list.** The bypass key is the **event type**, not the priority. If it were the priority, every producer would mark their notification high and quiet hours would cease to exist — and there would be no class where you could see that happening.
- **`TokenBucket` returning a wait time rather than a boolean.** The HLD's rule is that a limiter should delay, not reject, because the value does not expire quickly. A `bool` return makes the reject path the only expressible one, and OTP then has to special-case it.
- **`RetryPolicy` returning an absolute time.** A relative "now + 5 minutes" computed on each attempt drifts with clock skew across workers; the HLD's §6.11 names it. An absolute `scheduled_at` cannot drift.
- **`IdempotencyStore` and `DedupeStore` as two classes.** One is the producer's retry; the other is two services emitting the same semantic event. Collapsing them means the dedupe TTL is tuned for the wrong problem.

### 2.1 HLD component → its classes

| HLD component (§3.4) | The classes it is made of | The one thing that makes it non-trivial |
|---|---|---|
| Notification API | `NotificationApi`, `NotificationRequest`, `IdempotencyStore` | The key claim happens **before** anything is enqueued, and a duplicate returns the original id |
| Kafka `notification.requests` | `Envelope`, `PriorityQueue` (one per priority) | Isolation is per queue object, not per message field |
| Decision Service | `DecisionService`, `RecipientResolver`, `PreferenceResolver`, `QuietHours`, `UserRateLimiter`, `DedupeStore` | It is the whole policy engine, and its output is a `DeliveryPlan`, not a send |
| Preferences & consent store | `PreferenceStore`, `ConsentStore`, `UserPreferences`, `Timezone` | Evaluated at send time, and the timezone is an IANA name resolved then |
| Dedupe store | `DedupeStore`, `dedupe_key` | The key must carry entity identity, not just the event type |
| Template Service | `TemplateStore`, `TemplateRenderer`, `RenderedMessage` | The version is recorded on the notification, so a later edit is traceable |
| Priority queues | `PriorityQueue`, `Priority`, `QueueDepth` | Depth is what the load-shedding rule reads |
| Channel workers | `Channel`, `PushChannel`, `EmailChannel`, `SmsChannel`, `ClassificationTable` | Retryable vs terminal is decided here and nowhere else |
| Per-provider rate limiter | `TokenBucket`, `ProviderLimiter`, reserved HIGH capacity | Delays rather than rejects, except for OTP |
| Webhook handler | `WebhookHandler`, `DeliveryEvent`, `SignatureVerifier` | It must return fast; a slow webhook gets disabled by the provider |
| Notification state store | `NotificationStateStore`, `Notification`, `NotificationState` | Forward-only transitions, with the minimum over channels as the aggregate |
| DLQ | `DeadLetterQueue`, `DeadLetter` | Terminal failures are first-class, with a replay path |
| Scheduler | `Scheduler`, `ScheduledJob`, `local_date_key` | Idempotent per (user, campaign, local date), so a DST overlap cannot double-send |
| Retry service | `RetryPolicy`, `Backoff`, `jitter` | Absolute `scheduled_at`, jittered, with a per-priority schedule |
| Analytics | `AnalyticsSink` | Off the send path entirely |

---

## 3. Class diagram

```
                          ┌──────────────────────────────────────────┐
                          │  NotificationApi                         │
                          │  +submit(NotificationRequest, idem_key)  │
                          └───────────────┬──────────────────────────┘
                                          │ 1. claim the key FIRST
                                          ▼
                          ┌──────────────────────────────────────────┐
                          │  IdempotencyStore                        │
                          │  +claim(key, ttl) -> Optional[str]       │
                          │  a duplicate returns the ORIGINAL id     │
                          └───────────────┬──────────────────────────┘
                                          │ 2. accepted -> DecisionService
                                          ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │  DecisionService.decide(request, now) -> DeliveryPlan                     │
   │                                                                           │
   │   a. RecipientResolver.resolve(request)      -> [user_id]                 │
   │   b. PreferenceStore.load(user)              -> UserPreferences           │
   │      ConsentStore.is_allowed(user, type)     -> bool   (legal, separate)  │
   │   c. DedupeStore.claim(dedupe_key, user, ch) -> bool   ← FIRST WINS       │
   │   d. QuietHours.decide(priority, event_type,                              │
   │                        Timezone.local_time(user, now))                    │
   │                                    -> SEND_NOW | DEFER(until) | DROP      │
   │   e. UserRateLimiter.check(user, category)   -> allowed | deferred        │
   │   f. PreferenceResolver.allowed_channels(event_type)                      │
   │                    -> requested ∩ allowed ∩ consented                     │
   │   g. Priority from the EVENT TYPE, never from the caller                  │
   │   h. write the state row BEFORE enqueueing  ← §6.11's "202 that never     │
   │                                                becomes anything"          │
   └───────────────────────────────┬───────────────────────────────────────────┘
                                   │ DeliveryPlan: [Notification]
                                   ▼
   ┌──────────────────────────────────────────┐
   │  TemplateRenderer.render(notification)   │
   │  +render(n, locale) -> RenderedMessage   │
   │     carries template_id + VERSION        │
   └───────────────┬──────────────────────────┘
                   │ one RenderedMessage per (notification, channel)
                   ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │  PriorityQueue   — ONE OBJECT PER PRIORITY, not a field on a message       │
   │   high : Queue | normal : Queue | low : Queue                              │
   │   isolation is at the worker-pool level; a bulk call cannot occupy a       │
   │   worker a password reset needs                                             │
   └───────────────┬───────────────────────────────────────────────────────────┘
                   │ poll
                   ▼
   ┌──────────────────────────────────────────┐        ┌──────────────────────┐
   │  Channel (ABC)  — the strategy            │        │  ProviderLimiter     │
   │  PushChannel  : FCM / APNs                │◄───────┤  TokenBucket per     │
   │  EmailChannel : SES / SendGrid            │ rate   │  provider, with      │
   │  SmsChannel   : Twilio                    │ limit  │  RESERVED high       │
   │  +send(msg) -> SendOutcome                │        │  +acquire(now)       │
   └───────────────┬──────────────────────────┘        │   -> wait_seconds    │
                   │                                    └──────────────────────┘
                   │ success / failure
                   ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │  ClassificationTable   ← retryable vs terminal IS DATA                    │
   │   ("UNREGISTERED", "410", "invalid_token")     -> SUPPRESS_TOKEN          │
   │   ("429", "503", "UNAVAILABLE", "timeout")     -> RETRYABLE               │
   │   ("400", "unsubscribed", "hard_bounce")       -> TERMINAL                │
   └───────────────┬───────────────────────────────────────────────────────────┘
                   │
        ┌──────────┴───────────┬────────────────────────┐
        ▼                      ▼                        ▼
  ┌───────────────┐   ┌────────────────────┐   ┌──────────────────┐
  │ RetryPolicy   │   │ NotificationState  │   │ DeadLetterQueue  │
  │ +next_attempt │   │ Store              │   │ after N attempts │
  │  -> Optional  │   │ forward-only       │   │ +replay()        │
  │    [absolute] │   │ transitions        │   └──────────────────┘
  │ jittered      │   └─────────▲──────────┘
  └───────────────┘             │
                                │ consumer (NOT the handler)
                    ┌───────────┴──────────────────────────────────────┐
                    │  Provider webhook                                │
                    │      │                                           │
                    │      ▼                                           │
                    │  WebhookHandler.handle(raw) -> DeliveryEvent     │
                    │  validate signature, translate, publish, 200     │
                    │  < 100 ms — a slow webhook gets DISABLED         │
                    └──────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  AnalyticsSink  │
                              └─────────────────┘

  State machine (FORWARD ONLY):
    PENDING ─► ACCEPTED ─► SENT ─► DELIVERED ─► OPENED ─► CLICKED
       │           │
       │           └─► FAILED (terminal)      ── the aggregate the SENDER
       └─► SUPPRESSED (dedupe / preference /     sees is the MINIMUM rank
                       quiet-hours drop)          across channels, so one
       DEAD (DLQ, after N attempts)               delivered channel does not
                                                  make the whole thing green
```

**Read it this way:** the top half is *policy* and the bottom half is *transport*. Everything that makes a notification system hard is above the queue, and everything that makes it reliable is below it. The one arrow that crosses both is the state store, and it is written at decision time precisely so the transport can never lose a notification silently.

---

## 4. Interfaces / abstract classes / enums

```python
class Priority(Enum):
    HIGH = "high"      # OTP, password reset, security, fraud — never dropped
    NORMAL = "normal"  # order updates, social, delivery status
    LOW = "low"        # marketing, digests, recommendations — droppable


class ChannelKind(Enum):
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"   # a read model; never sent through a provider


class FailureKind(Enum):
    """Three, not two. The third is the one that costs money.

    RETRYABLE      — the same call may succeed later (429, 5xx, timeout)
    TERMINAL       — the same call will NEVER succeed (400, unsubscribed)
    SUPPRESS_TOKEN — terminal AND the destination is dead, so delete it
    """
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    SUPPRESS_TOKEN = "suppress_token"


class NotificationState(Enum):
    PENDING = "pending"           # decided and stored, not yet queued
    ACCEPTED = "accepted"         # the provider took it
    SENT = "sent"                 # the provider forwarded it
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    FAILED = "failed"             # terminal
    SUPPRESSED = "suppressed"     # never sent, by policy
    DEAD = "dead"                 # DLQ


class QuietDecision(Enum):
    SEND_NOW = "send_now"
    DEFER = "defer"
    DROP = "drop"


class Channel(ABC):
    @property
    @abstractmethod
    def kind(self) -> ChannelKind:
        raise NotImplementedError

    @abstractmethod
    def send(self, message: "RenderedMessage") -> "SendOutcome":
        """Return ACCEPTED or a CLASSIFIED failure. Never retry here — a channel
        that retries internally makes the retry policy unobservable, and the
        attempt count on the state row becomes a lie."""
        raise NotImplementedError


class QuietHoursPolicy(ABC):
    @abstractmethod
    def decide(self, priority: Priority, event_type: str,
               local_time: datetime) -> QuietDecision:
        """Priority decides DEFER vs DROP; the event type decides whether the
        window applies at all."""
        raise NotImplementedError


class AnalyticsSink(ABC):
    @abstractmethod
    def record(self, event: "DeliveryEvent") -> None:
        raise NotImplementedError


class Clock(ABC):
    """Injected, because every interesting behaviour in this system is a
    function of time — backoff, quiet hours, TTLs and rate-limit refill — and a
    test that cannot move the clock cannot test any of them."""

    @abstractmethod
    def now(self) -> float:
        raise NotImplementedError
```

**Five contract notes:**

- **`send` never retries.** A channel that hides a retry makes `attempt` on the state row meaningless, and `attempt` is what the DLQ threshold and the cost alarm both read.
- **`FailureKind` is returned, not raised.** An exception carries no classification, so every caller has to guess — which is how a retryable 503 and a dead token end up in the same `except` block.
- **`QuietHoursPolicy.decide` takes `event_type` and `priority` separately.** They answer different questions: the priority says what to do inside the window, the event type says whether the window applies. Merging them into one "importance" score is the bug the HLD's §6.5 describes.
- **`NotificationState` is forward-only and ranked.** The rank is what lets the store refuse an out-of-order webhook instead of walking a notification backwards, and it is what makes an aggregate over a request's rows well defined — with the caveat that the notification aggregate is the **maximum** progress (the user got it *somewhere*), the opposite of the WhatsApp receipt aggregate, which is the minimum (everyone read it). Same machinery, opposite direction, and the two are easy to swap by accident.
- **`Clock` is an interface.** `time.time()` inlined in five classes is five tests that cannot control time.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** In Java, `NotificationState` and `FailureKind` are enums, `Notification` is a record, and the classification table is an `EnumMap`. In Go, the classification is a `map[string]FailureKind` and the channels are an interface with three implementations. The **table-driven failure classification, the quiet-hours bypass allow-list keyed on the event type, and the limiter that returns a wait time instead of a boolean** are the design; the syntax is not.

```python
"""notification_lld.py — the classes inside the HLD's boxes.

Design summary
--------------
The unit      : one Notification = one recipient x one channel x one state row.
Decision      : the whole policy engine, and it writes the state row BEFORE
                enqueueing, so an accepted notification is always accounted for.
Failure       : RETRYABLE | TERMINAL | SUPPRESS_TOKEN, from a TABLE, not an if.
Quiet hours   : priority decides DEFER vs DROP; the EVENT TYPE decides whether
                the window applies at all, through an explicit allow-list.
Rate limiting : a token bucket that returns a WAIT TIME. The caller decides
                whether to delay (normal) or fail fast (OTP).
Retries       : an absolute `scheduled_at`, jittered, per priority.
Webhooks      : validate, translate, publish, return fast. The state store is
                written by a consumer, never by the handler.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone as dt_timezone
from enum import Enum
from typing import Callable, Iterable, Optional


# ==========================================================================
# 0. The vocabulary — repeated from §4 so this block runs standalone
# ==========================================================================

class Priority(Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ChannelKind(Enum):
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"


class FailureKind(Enum):
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    SUPPRESS_TOKEN = "suppress_token"


class NotificationState(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    FAILED = "failed"
    SUPPRESSED = "suppressed"
    DEAD = "dead"


class QuietDecision(Enum):
    SEND_NOW = "send_now"
    DEFER = "defer"
    DROP = "drop"


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> float:
        return time.time()


class ManualClock(Clock):
    """The clock every test in §5 uses. Time is an INPUT to this system —
    backoff, quiet hours, TTLs and bucket refill all read it — so a test that
    cannot move it can only test the parts that do not matter."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        self._now += seconds
        return self._now


class QuietHoursPolicy(ABC):
    """Repeated from §4 so this block runs standalone.

    The interface exists so a jurisdiction's rules can be swapped without
    touching the decision service — and the reason it takes `event_type` as well
    as `priority` is the whole quiet-hours design, restated as a signature.
    """

    @abstractmethod
    def decide(self, priority: Priority, event_type: str,
               local_time: datetime) -> QuietDecision:
        raise NotImplementedError


class AnalyticsSink(ABC):
    """Repeated from §4 so this block runs standalone. One method, and it must
    not raise: analytics is on the delivery path, so a broken sink that throws
    would take down the thing it is measuring."""

    @abstractmethod
    def record(self, event: DeliveryEvent) -> None:
        raise NotImplementedError


class Counter:
    """A counter that does not lose increments.

    `self.rejected += 1` is a load, an add and a store, so two webhook threads
    racing on it silently lose counts — and these particular counts are the ones
    an incident review reads, which makes losing them the wrong kind of quiet.
    In a real service this is an atomic counter in the metrics library with the
    same one-line interface.
    """

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def bump(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def value(self) -> int:
        with self._lock:
            return self._value


# ==========================================================================
# 1. The failure classification — THE table
# ==========================================================================

class ClassificationTable:
    """Provider error -> FailureKind, as DATA.

    WHY THIS IS A CLASS AND NOT AN `if` INSIDE EACH CHANNEL. Three reasons, and
    the third is the one that shows up in an interview:

    1. A channel that decides retryability cannot be tested without a provider.
    2. Adding a provider means adding a `Channel`; it must not also mean
       re-deriving which of that provider's codes are terminal.
    3. The table is the thing you can READ during an incident. "Why did 40,000
       notifications die?" is answered by looking at one dictionary rather than
       by reading three SDK adapters.

    THE DEFAULT FOR AN UNKNOWN CODE IS RETRYABLE, and the direction is
    deliberate. An unknown code is one of two things: a transient error nobody
    has catalogued, or a terminal error nobody has catalogued. Guessing
    RETRYABLE on a terminal error costs a bounded number of attempts and then
    the DLQ catches it — noisy but safe. Guessing TERMINAL on a transient error
    silently drops a notification that would have succeeded, and nothing in the
    system will ever notice. Prefer the failure that is loud.
    """

    DEFAULT_RULES: dict[str, FailureKind] = {
        # --- terminal, and the destination is dead: delete the token ---
        "UNREGISTERED": FailureKind.SUPPRESS_TOKEN,
        "410": FailureKind.SUPPRESS_TOKEN,
        "INVALID_TOKEN": FailureKind.SUPPRESS_TOKEN,
        "INVALID_REGISTRATION": FailureKind.SUPPRESS_TOKEN,
        "INVALID_PHONE_NUMBER": FailureKind.SUPPRESS_TOKEN,

        # --- retryable: the same call may succeed later ---
        "429": FailureKind.RETRYABLE,
        "500": FailureKind.RETRYABLE,
        "502": FailureKind.RETRYABLE,
        "503": FailureKind.RETRYABLE,
        "504": FailureKind.RETRYABLE,
        "UNAVAILABLE": FailureKind.RETRYABLE,
        "INTERNAL": FailureKind.RETRYABLE,
        "TIMEOUT": FailureKind.RETRYABLE,
        "CONNECTION_RESET": FailureKind.RETRYABLE,
        "THROTTLED": FailureKind.RETRYABLE,

        # --- terminal: the same call will never succeed ---
        "400": FailureKind.TERMINAL,
        "MALFORMED_PAYLOAD": FailureKind.TERMINAL,
        "TEMPLATE_NOT_FOUND": FailureKind.TERMINAL,
        "UNSUBSCRIBED": FailureKind.TERMINAL,
        "HARD_BOUNCE": FailureKind.TERMINAL,
        "SUPPRESSED_RECIPIENT": FailureKind.TERMINAL,
        "COMPLAINT": FailureKind.TERMINAL,
        "STOP": FailureKind.TERMINAL,
    }

    def __init__(self, overrides: Optional[dict[str, FailureKind]] = None
                 ) -> None:
        self._rules = dict(self.DEFAULT_RULES)
        if overrides:
            self._rules.update({code.upper(): kind
                                for code, kind in overrides.items()})

    def classify(self, error_code: str) -> FailureKind:
        return self._rules.get(error_code.upper(), FailureKind.RETRYABLE)

    def codes_for(self, kind: FailureKind) -> list[str]:
        """Used by the demo, and by the alarm that says "the provider started
        returning a code we have never seen" — which is the moment the
        RETRYABLE default earns its keep."""
        return sorted(code for code, k in self._rules.items() if k is kind)

    def unknown(self, error_code: str) -> bool:
        return error_code.upper() not in self._rules


# ==========================================================================
# 2. Values
# ==========================================================================

@dataclass(frozen=True)
class NotificationRequest:
    """The producer's input. Note what is NOT here: no priority, and no
    recipient list for a campaign — the audience is resolved, not supplied."""

    request_id: str
    event_type: str
    entity_id: str
    user_ids: tuple[str, ...]
    channels: tuple[ChannelKind, ...]
    payload: dict[str, str] = field(default_factory=dict)
    locale: str = "en"
    scheduled_at: Optional[float] = None

    @property
    def dedupe_key(self) -> str:
        """ENTITY IDENTITY IS PART OF THE KEY. `order.shipped` alone collapses
        two different orders into one notification; `order.shipped:o_1123` does
        not. The HLD's deduplication section makes this point and it is the
        single easiest way to get dedupe wrong."""
        return f"{self.event_type}:{self.entity_id}"


@dataclass(frozen=True)
class RenderedMessage:
    """The version travels WITH the payload. A template edited after this
    render cannot retroactively explain why the notification looked wrong.

    `event_type` is on the message, not buried in `data`, because the SMS
    channel's allow-list is a policy decision made at send time and it needs the
    event type as a first-class field — a policy that reads a magic key out of a
    user-supplied payload dict is a policy a caller can forge.
    """

    notification_id: str
    channel: ChannelKind
    destination: str
    title: str
    body: str
    template_id: str
    template_version: int
    locale: str
    event_type: str = ""
    data: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SendOutcome:
    accepted: bool
    provider_message_id: Optional[str] = None
    error_code: str = ""
    kind: Optional[FailureKind] = None
    retry_after_s: Optional[float] = None

    @staticmethod
    def ok(provider_message_id: str) -> "SendOutcome":
        return SendOutcome(True, provider_message_id)

    @staticmethod
    def fail(error_code: str, table: ClassificationTable,
             retry_after_s: Optional[float] = None) -> "SendOutcome":
        return SendOutcome(False, None, error_code,
                           table.classify(error_code), retry_after_s)

    @property
    def terminal(self) -> bool:
        return (not self.accepted) and self.kind is not FailureKind.RETRYABLE


@dataclass
class Notification:
    """One recipient, one channel. Mutable, because `attempt` and `state` are
    the two things that change — and it is an ENTITY, so identity is by id."""

    id: str
    request_id: str
    user_id: str
    channel: ChannelKind
    category: str
    event_type: str
    priority: Priority
    destination: str
    template_id: str
    dedupe_key: str
    payload: dict[str, str]
    locale: str = "en"
    template_version: int = 0
    attempt: int = 0
    state: NotificationState = NotificationState.PENDING
    scheduled_at: Optional[float] = None
    provider_message_id: Optional[str] = None
    last_error: str = ""

    def __str__(self) -> str:
        return (f"{self.id} {self.channel.value}/{self.priority.value} "
                f"{self.state.value} attempt={self.attempt}")


@dataclass(frozen=True)
class DeliveryEvent:
    provider_message_id: str
    level: NotificationState
    occurred_at: float
    reason: str = ""


@dataclass(frozen=True)
class DeliveryPlan:
    """What the decision service returns. NOT a set of sends — a set of
    notifications that have already been written to the state store."""

    request_id: str
    notifications: tuple[Notification, ...]
    suppressed: tuple[tuple[str, str], ...] = ()   # (user_id, reason)

    def __len__(self) -> int:
        return len(self.notifications)


# ==========================================================================
# 3. The dedupe and idempotency stores — two keys, two problems
# ==========================================================================

class IdempotencyStore:
    """The PRODUCER's retry: the same POST arriving twice.

    `claim` is atomic and returns ownership, because a check-then-set version
    lets two retries of the same POST both see "not claimed", both proceed, and
    the user gets two OTPs — and worse, if the OTP store is last-write-wins the
    second invalidates the first, so the user cannot log in. That is the HLD's
    §6.3, and this method is its fix.

    The request id is recorded ON THE CLAIM, not after the work. That means a
    crash mid-request leaves the key held, so a retry gets the original id back
    rather than starting a second send. For an OTP that is the correct
    direction to fail: a retry that returns an id slightly before its state row
    exists is recoverable, and a second OTP is not.
    """

    def __init__(self, ttl_s: float = 24 * 3600.0) -> None:
        self._ttl = ttl_s
        self._keys: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def claim(self, key: str, request_id: str,
              now: float) -> tuple[bool, Optional[str]]:
        with self._lock:
            self._evict(now)
            existing = self._keys.get(key)
            if existing is not None:
                return False, existing[0]
            self._keys[key] = (request_id, now + self._ttl)
            return True, None

    def _evict(self, now: float) -> None:
        expired = [k for k, (_, expires) in self._keys.items() if expires <= now]
        for key in expired:
            del self._keys[key]

    def size(self) -> int:
        with self._lock:
            return len(self._keys)


class DedupeStore:
    """The SEMANTIC duplicate: two different services emit `order.shipped` for
    the same order.

    Scoped per (key, user, channel) — per CHANNEL, because the same event
    legitimately goes to push AND email, and a key scoped only per user would
    suppress the email because the push won the race. The TTL is per event
    type: minutes for chat-like events, days for order events.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str, str], float] = {}
        self._lock = threading.Lock()

    def claim(self, dedupe_key: str, user_id: str, channel: ChannelKind,
              now: float, ttl_s: float) -> bool:
        """True on the FIRST claim, False on every subsequent one."""
        with self._lock:
            key = (dedupe_key, user_id, channel.value)
            expires = self._seen.get(key)
            if expires is not None and expires > now:
                return False
            self._seen[key] = now + ttl_s
            return True

    def size(self) -> int:
        with self._lock:
            return len(self._seen)


class TokenBucket:
    """Refills lazily; `acquire` returns a WAIT TIME, never a boolean.

    THE RETURN TYPE IS THE DESIGN. A bucket that returns True/False makes
    "reject" the only expressible outcome, and the HLD's rule is the opposite:
    delay rather than reject, because a notification's value does not expire
    quickly. The exception is OTP, which has a hard validity window — and with
    a wait time returned, the caller can express that exception as `if wait >
    otp_window: fail_fast()` instead of the bucket having to know about OTP.
    """

    def __init__(self, capacity: float, refill_per_s: float,
                 clock: Clock) -> None:
        if capacity <= 0 or refill_per_s <= 0:
            raise ValueError("a bucket needs positive capacity and refill")
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._tokens = float(capacity)
        self._last = clock.now()
        self._clock = clock
        self._lock = threading.Lock()

    def acquire(self, cost: float = 1.0) -> float:
        """0.0 means the tokens were taken and the caller may send now.
        Anything greater is the number of seconds until it may."""
        with self._lock:
            now = self._clock.now()
            self._tokens = min(self.capacity,
                               self._tokens + (now - self._last) * self.refill_per_s)
            self._last = now
            if self._tokens >= cost:
                self._tokens -= cost
                return 0.0
            return (cost - self._tokens) / self.refill_per_s

    def tokens(self) -> float:
        with self._lock:
            now = self._clock.now()
            self._tokens = min(self.capacity,
                               self._tokens + (now - self._last) * self.refill_per_s)
            self._last = now
            return self._tokens


class ProviderLimiter:
    """One general bucket per provider, plus a RESERVED bucket only HIGH may
    touch.

    This is the HLD's "reserved provider capacity" made concrete. Separating
    the queues isolates the workers; it does not isolate the provider's own
    limit, which is a shared resource. Without a reserved bucket, a bulk
    campaign that drains the general bucket leaves a password reset waiting on
    a provider that is busy sending promotions — the same failure as §6.1, one
    layer further down.
    """

    def __init__(self, general: TokenBucket, reserved: Optional[TokenBucket] = None
                 ) -> None:
        self.general = general
        self.reserved = reserved

    def acquire(self, priority: Priority) -> float:
        if priority is Priority.HIGH and self.reserved is not None:
            return self.reserved.acquire()
        return self.general.acquire()


# ==========================================================================
# 4. Preferences, consent, timezones, quiet hours
# ==========================================================================

@dataclass(frozen=True)
class UserPreferences:
    """`timezone` is an IANA NAME, never a fixed offset. Storing "-05:00" makes
    a user in America/New_York an hour wrong twice a year, and the failure is a
    notification arriving at 21:00 or 23:00 instead of 22:00 — a bug nobody
    reports."""

    user_id: str
    timezone: str = "UTC"
    quiet_start_hour: int = 22
    quiet_end_hour: int = 8
    channels: dict[str, bool] = field(default_factory=dict)      # channel -> on
    categories: dict[str, bool] = field(default_factory=dict)    # category -> on
    marketing_opt_out: bool = False

    def channel_enabled(self, channel: ChannelKind) -> bool:
        return self.channels.get(channel.value, True)

    def category_enabled(self, category: str) -> bool:
        return self.categories.get(category, True)


class PreferenceStore:
    def __init__(self) -> None:
        self._by_user: dict[str, UserPreferences] = {}
        self._lock = threading.RLock()

    def put(self, prefs: UserPreferences) -> None:
        with self._lock:
            self._by_user[prefs.user_id] = prefs

    def load(self, user_id: str) -> UserPreferences:
        """Defaults are permissive. A user with no row is a new user, and
        defaulting a channel OFF means every new user silently receives
        nothing — which is a launch-day incident, not a privacy win."""
        with self._lock:
            return self._by_user.get(user_id, UserPreferences(user_id))


class ConsentStore:
    """LEGAL opt-in, kept separate from product preferences.

    Separate because they have different owners, different retention and
    different failure consequences. A user turning push off in settings is a
    product preference; a user replying STOP to an SMS is a legal suppression
    that must be honoured immediately and cannot be re-enabled by a settings
    toggle. Merging them into one row is how a marketing SMS reaches somebody
    who texted STOP."""

    def __init__(self) -> None:
        self._blocked: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def suppress(self, user_id: str, channel: ChannelKind) -> None:
        with self._lock:
            self._blocked.add((user_id, channel.value))

    def is_allowed(self, user_id: str, channel: ChannelKind) -> bool:
        with self._lock:
            return (user_id, channel.value) not in self._blocked

    def blocked_count(self) -> int:
        with self._lock:
            return len(self._blocked)


class Timezone:
    """IANA name -> local time, RESOLVED AT SEND TIME.

    The offset resolver is injected, and that is the whole point. A hard-coded
    table would reproduce the exact bug the IANA name exists to avoid — the
    offset for America/New_York is -5 in January and -4 in July — so the demo
    below injects a DST-aware resolver to prove the code path that matters
    exists. A real deployment passes `zoneinfo.ZoneInfo(zone).utcoffset(...)`.
    """

    FIXED_OFFSETS: dict[str, float] = {
        "UTC": 0.0,
        "Asia/Kolkata": 5.5,
        "Europe/London": 0.0,
    }

    def __init__(self, resolver: Optional[Callable[[str, float], float]] = None
                 ) -> None:
        self._resolver = resolver

    def offset_hours(self, zone: str, at_epoch: float) -> float:
        if self._resolver is not None:
            return self._resolver(zone, at_epoch)
        if zone in self.FIXED_OFFSETS:
            return self.FIXED_OFFSETS[zone]
        raise KeyError(f"unknown timezone: {zone!r} — use an IANA name")

    def local_time(self, zone: str, at_epoch: float) -> datetime:
        offset = timedelta(hours=self.offset_hours(zone, at_epoch))
        return datetime.fromtimestamp(at_epoch, tz=dt_timezone(offset))

    def local_date_key(self, zone: str, at_epoch: float) -> str:
        """The idempotency key for a scheduled or digest send. A DST transition
        makes two instants share a wall-clock date or makes one date vanish —
        and keying on the LOCAL DATE rather than on the timestamp is what stops
        a DST overlap from sending the same digest twice."""
        return self.local_time(zone, at_epoch).strftime("%Y-%m-%d")

    def next_local_time(self, zone: str, at_epoch: float, hour: int) -> float:
        """The next epoch at which the LOCAL wall clock reads `hour`.

        THIS METHOD LIVES HERE AND NOT ON `QuietHours`, AND THAT IS THE FIX TO A
        REAL BUG. The obvious implementation — take the local time, replace the
        hour, convert back — is wrong on the one day a year it matters, and it
        is wrong by an hour in the direction that wakes somebody up:

            local = 01:30 EDT (UTC-4) on 5 Nov; replace(hour=8) -> 08:00 EDT
            -> 12:00Z. But 12:00Z on that date is 07:00 EST, because the clock
            went back at 02:00 local. The notification arrives at 07:00, an hour
            inside the window it was deferred out of.

        The offset that applied at 01:30 no longer applies at 08:00, so the
        offset has to be resolved AT THE TARGET, not at the source. One
        correction step is enough — a transition moves the clock by at most an
        hour and the target is hours away — and a second step is a no-op, which
        is what makes it safe to apply unconditionally.
        """
        local = self.local_time(zone, at_epoch)
        candidate = local.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate <= local:
            candidate = candidate + timedelta(days=1)
        baked_in = self.offset_hours(zone, at_epoch)
        actually_applies = self.offset_hours(zone, candidate.timestamp())
        if baked_in != actually_applies:
            candidate = candidate + timedelta(hours=baked_in - actually_applies)
        return candidate.timestamp()


EVENT_PRIORITY: dict[str, Priority] = {
    "auth.otp": Priority.HIGH,
    "password.reset": Priority.HIGH,
    "security.alert": Priority.HIGH,
    "fraud.alert": Priority.HIGH,
    "order.shipped": Priority.NORMAL,
    "order.delivered": Priority.NORMAL,
    "social.follow": Priority.NORMAL,
    "social.like": Priority.NORMAL,
    "marketing.campaign": Priority.LOW,
    "digest.weekly": Priority.LOW,
    "recommendation": Priority.LOW,
}

# THE BYPASS ALLOW-LIST. Membership is by EVENT TYPE, and it is the only thing
# that lets a notification through quiet hours. Priority is derived from this
# same table by the system; a producer never supplies it. That is what makes
# the bypass auditable — it is a list somebody reviews, not a claim a caller
# makes about their own message.
QUIET_HOURS_BYPASS: frozenset[str] = frozenset({
    "security.alert",
    "fraud.alert",
    "auth.otp",
})


class QuietHours:
    """Priority decides DEFER vs DROP. The EVENT TYPE decides whether the window
    applies at all.

    This is deliberately STRICTER than "high priority bypasses quiet hours".
    Under the loose version, every producer marks their notification high and
    quiet hours stop existing — and there is no artifact to review, because the
    bypass has become a runtime claim rather than a configuration. Under this
    version, a HIGH notification that is not on the allow-list is DEFERRED like
    anything else, so adding a new 3 a.m. wake-up is a code review against a
    frozen set. The HLD names the risk; this is the class that removes it.
    """

    def __init__(self, bypass: Iterable[str] = QUIET_HOURS_BYPASS) -> None:
        self.bypass = frozenset(bypass)

    def in_window(self, local_time: datetime, start_hour: int,
                  end_hour: int) -> bool:
        if start_hour == end_hour:
            return False                       # a zero-width window is no window
        if start_hour < end_hour:
            return start_hour <= local_time.hour < end_hour
        return local_time.hour >= start_hour or local_time.hour < end_hour

    def decide(self, priority: Priority, event_type: str,
               local_time: datetime, start_hour: int = 22,
               end_hour: int = 8) -> QuietDecision:
        if event_type in self.bypass:
            return QuietDecision.SEND_NOW      # explicit allow-list
        if not self.in_window(local_time, start_hour, end_hour):
            return QuietDecision.SEND_NOW
        if priority is Priority.LOW:
            return QuietDecision.DROP
        return QuietDecision.DEFER             # NORMAL, and HIGH-not-on-the-list


class UserRateLimiter:
    """Per-user, per-category, per-LOCAL-DAY. Separate from the provider limit
    because the two protect different things: the provider limit protects your
    sending reputation, this protects the user from you."""

    DEFAULT_LIMITS: dict[str, int] = {
        "marketing": 5,
        "social": 100,
        "orders": 50,
    }

    def __init__(self, clock: Clock, tz: Timezone,
                 limits: Optional[dict[str, int]] = None) -> None:
        self._counts: dict[tuple[str, str, str], int] = {}
        self._clock = clock
        self._tz = tz
        self._limits = dict(self.DEFAULT_LIMITS)
        if limits:
            self._limits.update(limits)
        self._lock = threading.Lock()

    def check_and_count(self, user_id: str, category: str,
                        zone: str) -> tuple[bool, int]:
        """Returns (allowed, remaining). Counting on check is deliberate: a
        separate `check` and `count` lets two concurrent sends both pass the
        check and both count, which is exactly the double-send this exists to
        prevent."""
        limit = self._limits.get(category)
        if limit is None:
            return True, -1
        now = self._clock.now()
        key = (user_id, category, self._tz.local_date_key(zone, now))
        with self._lock:
            used = self._counts.get(key, 0)
            if used >= limit:
                return False, 0
            self._counts[key] = used + 1
            return True, limit - used - 1

    def used(self, user_id: str, category: str, zone: str) -> int:
        key = (user_id, category, self._tz.local_date_key(zone, self._clock.now()))
        with self._lock:
            return self._counts.get(key, 0)


# ==========================================================================
# 5. Retry as a VALUE, and the state machine
# ==========================================================================

class RetryPolicy:
    """`next_attempt` returns an ABSOLUTE epoch time, or None for the DLQ.

    TWO THINGS ARE LOAD-BEARING AND BOTH ARE EASY TO GET WRONG.

    Absolute, not relative: "now + 5 minutes" computed on each attempt drifts
    with clock skew between workers, so a message written by one worker and
    retried by another lands at a time neither intended. An absolute
    `scheduled_at` is a fact about when, not a distance from whenever.

    Jitter is mandatory, not a refinement: without it, every worker that failed
    during a provider outage retries at the same instant, and the provider
    recovers into a synchronised second storm aimed at a service that is
    already struggling. That is the HLD's §6.2, and ±20% is the cheapest fix in
    the whole design.
    """

    SCHEDULES: dict[Priority, tuple[float, ...]] = {
        Priority.HIGH:   (1.0, 5.0, 30.0, 300.0),
        Priority.NORMAL: (5.0, 30.0, 300.0, 1800.0),
        Priority.LOW:    (60.0, 300.0, 1800.0, 7200.0),
    }

    def __init__(self, jitter: float = 0.2,
                 rng: Optional[random.Random] = None) -> None:
        if not 0.0 <= jitter < 1.0:
            raise ValueError("jitter must be in [0, 1)")
        self.jitter = jitter
        self._rng = rng or random.Random()

    @property
    def max_attempts(self) -> int:
        return len(self.SCHEDULES[Priority.HIGH]) + 1

    def next_attempt(self, attempt: int, priority: Priority, now: float,
                     retry_after_s: Optional[float] = None) -> Optional[float]:
        """`attempt` is the number of attempts ALREADY made. None means the
        budget is spent and the notification goes to the DLQ."""
        schedule = self.SCHEDULES[priority]
        if attempt > len(schedule):
            return None
        delay = schedule[attempt - 1]
        if retry_after_s is not None:
            # Honour the provider's own Retry-After when it is longer than our
            # schedule. Ignoring it is how a 429 becomes a 403.
            delay = max(delay, retry_after_s)
        spread = delay * self.jitter
        jittered = max(0.0, delay + self._rng.uniform(-spread, spread))
        return now + jittered


class IllegalTransition(Exception):
    def __init__(self, current: NotificationState, target: NotificationState,
                 reason: str) -> None:
        super().__init__(f"{current.value} -> {target.value}: {reason}")
        self.current = current
        self.target = target
        self.reason = reason


class NotificationStateStore:
    """One row per notification, and the transitions are FORWARD ONLY.

    Why forward-only matters here more than in most systems: the webhook is
    delivered at least once and out of order. An `opened` event can arrive
    before the `delivered` event it follows, and a naive store would move the
    notification backwards — so the sender's view would flicker from "opened"
    to "delivered". A rank order plus a refusal to go backwards makes the
    out-of-order case a no-op instead of a regression.
    """

    PROGRESS: dict[NotificationState, int] = {
        NotificationState.PENDING: 0,
        NotificationState.ACCEPTED: 1,
        NotificationState.SENT: 2,
        NotificationState.DELIVERED: 3,
        NotificationState.OPENED: 4,
        NotificationState.CLICKED: 5,
    }
    TERMINAL: frozenset[NotificationState] = frozenset({
        NotificationState.FAILED,
        NotificationState.SUPPRESSED,
        NotificationState.DEAD,
    })

    def __init__(self) -> None:
        self._by_id: dict[str, Notification] = {}
        self._by_request: dict[str, list[str]] = {}
        self._by_provider: dict[str, str] = {}
        self._lock = threading.RLock()

    # -- writes ----------------------------------------------------------
    def put(self, notification: Notification) -> None:
        """Called at DECISION time, before anything is enqueued. A notification
        that is accepted and then lost between the queue and the worker is the
        HLD's §6.11 "202 that never becomes anything"; a row written first
        means every accepted notification is accounted for even if nothing else
        works."""
        with self._lock:
            self._by_id[notification.id] = notification
            self._by_request.setdefault(notification.request_id, []).append(
                notification.id)

    def transition(self, notification_id: str, target: NotificationState,
                   error: str = "") -> Notification:
        with self._lock:
            notification = self._by_id[notification_id]
            current = notification.state
            if current is target:
                return notification              # idempotent, and this is the
                                                 # common case for webhooks
            if current in self.TERMINAL:
                raise IllegalTransition(current, target, "already terminal")
            if target in self.PROGRESS and current in self.PROGRESS:
                if self.PROGRESS[target] < self.PROGRESS[current]:
                    raise IllegalTransition(current, target,
                                            "the lifecycle is forward-only")
            elif target not in self.TERMINAL:
                raise IllegalTransition(current, target, "not a legal successor")
            notification.state = target
            if error:
                notification.last_error = error
            return notification

    def record_provider_id(self, notification_id: str,
                           provider_message_id: str) -> None:
        with self._lock:
            notification = self._by_id[notification_id]
            notification.provider_message_id = provider_message_id
            self._by_provider[provider_message_id] = notification_id

    def find_by_provider_id(self, provider_message_id: str
                            ) -> Optional[Notification]:
        """What makes the webhook handler idempotent. The same provider message
        reported twice — or our own retry racing the webhook — resolves to one
        notification, and the second report is a no-op."""
        with self._lock:
            notification_id = self._by_provider.get(provider_message_id)
            return None if notification_id is None else self._by_id[notification_id]

    # -- reads -----------------------------------------------------------
    def get(self, notification_id: str) -> Optional[Notification]:
        with self._lock:
            return self._by_id.get(notification_id)

    def for_request(self, request_id: str) -> list[Notification]:
        with self._lock:
            return [self._by_id[i] for i in self._by_request.get(request_id, ())]

    def progress(self, request_id: str) -> NotificationState:
        """The SENDER's view, and it is the MAXIMUM progress — the opposite of
        the receipt aggregate in the WhatsApp bridge, for a reason worth
        saying out loud. A read receipt is satisfied only when EVERY recipient
        has read the message, so its aggregate is the minimum. A notification
        request that fanned out to push and email is satisfied when the user
        got it by ANY of them, so its aggregate is the maximum. Two aggregates,
        two questions, and picking the wrong one is silent either way: the
        minimum here reports a delivered notification as failed, and the
        maximum there reports a group message as read when one person read it.
        """
        rows = self.for_request(request_id)
        if not rows:
            return NotificationState.PENDING
        best = NotificationState.PENDING
        for row in rows:
            if row.state in self.PROGRESS:
                if self.PROGRESS[row.state] > self.PROGRESS[best]:
                    best = row.state
        return best

    def counts(self, request_id: str) -> dict[str, int]:
        """The OPS view: the distribution, not a single value. 'Delivered'
        hides that one of three channels is failing terminally; the histogram
        does not."""
        out: dict[str, int] = {}
        for row in self.for_request(request_id):
            out[row.state.value] = out.get(row.state.value, 0) + 1
        return out

    def size(self) -> int:
        with self._lock:
            return len(self._by_id)

    def due(self, now: float) -> list[Notification]:
        """PENDING rows whose time has come, ordered by PRIORITY FIRST and then
        by time.

        A full scan, and that is the honest first version: the real system keeps
        a time-ordered index per priority, which is the HLD's "a queue per
        priority, a worker pool per queue" seen from the storage side. The
        ordering here is the part that matters and the part that would be wrong
        if this were left to insertion order — a bulk campaign enqueued at
        09:00 must not sit ahead of a password reset enqueued at 09:01, and
        `scheduled_at` alone cannot express that.

        DEFERRED notifications are PENDING rows with a future `scheduled_at`,
        which is why quiet hours needs no state of its own: the deferral IS a
        scheduled time, and it survives a restart because it is a column.
        """
        rank = {Priority.HIGH: 0, Priority.NORMAL: 1, Priority.LOW: 2}
        with self._lock:
            ready = [n for n in self._by_id.values()
                     if n.state is NotificationState.PENDING
                     and (n.scheduled_at is None or n.scheduled_at <= now)]
        ready.sort(key=lambda n: (rank[n.priority],
                                  n.scheduled_at if n.scheduled_at else 0.0,
                                  n.id))
        return ready


@dataclass(frozen=True)
class DeadLetter:
    notification_id: str
    request_id: str
    user_id: str
    channel: ChannelKind
    attempts: int
    last_error: str
    at: float

    def __str__(self) -> str:
        return (f"{self.notification_id} {self.channel.value} "
                f"after {self.attempts}: {self.last_error}")


class DeadLetterQueue:
    """Terminal failures, with the reason and an attempt count. The point of a
    DLQ is not storage; it is that nothing is dropped SILENTLY, and that a
    replay tool has the payload and the reason to work from."""

    def __init__(self) -> None:
        self._entries: list[DeadLetter] = []
        self._lock = threading.Lock()

    def add(self, letter: DeadLetter) -> None:
        with self._lock:
            self._entries.append(letter)

    def entries(self) -> list[DeadLetter]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        return len(self.entries())


# ==========================================================================
# 6. Templates
# ==========================================================================

class RenderError(Exception):
    """A missing placeholder or an unknown template. TERMINAL: the same input
    will produce the same failure forever, so retrying it is pure cost."""


@dataclass(frozen=True)
class Template:
    template_id: str
    channel: ChannelKind
    locale: str
    version: int
    title: str
    body: str


class TemplateStore:
    def __init__(self) -> None:
        self._by_key: dict[tuple[str, ChannelKind, str], list[Template]] = {}
        self._lock = threading.RLock()

    def put(self, template: Template) -> None:
        with self._lock:
            key = (template.template_id, template.channel, template.locale)
            versions = self._by_key.setdefault(key, [])
            versions.append(template)
            versions.sort(key=lambda t: t.version)

    def pick(self, template_id: str, channel: ChannelKind,
             locale: str) -> Optional[Template]:
        """LATEST version, with an exact-locale-then-English fallback. The
        fallback is explicit rather than "whatever is there", because silently
        sending a French user a German template is a worse outcome than sending
        them English."""
        with self._lock:
            for candidate_locale in (locale, "en"):
                versions = self._by_key.get((template_id, channel,
                                             candidate_locale))
                if versions:
                    return versions[-1]
            return None

    def versions(self, template_id: str, channel: ChannelKind,
                 locale: str) -> int:
        with self._lock:
            return len(self._by_key.get((template_id, channel, locale), ()))

    def get_version(self, template_id: str, channel: ChannelKind, locale: str,
                    version: int) -> Optional[Template]:
        """The EXACT version the notification was decided against.

        This is the other half of the §6.11 fix. Recording a version in the
        notification is useless unless the send path reads that version back —
        the tempting shortcut is to re-pick the latest template at send time,
        which reintroduces the bug in a subtler place: a notification sitting in
        a retry queue for six hours renders with whatever was published since.
        Versions are append-only, so a missing one is a store problem, not a
        race, and the caller treats it as terminal.
        """
        with self._lock:
            for candidate in self._by_key.get((template_id, channel, locale), ()):
                if candidate.version == version:
                    return candidate
            return None


class TemplateRenderer:
    """Template + payload -> RenderedMessage, and the VERSION rides along.

    A render failure is terminal. That matters because the HLD's §6.11 names a
    template edit breaking in-flight notifications: if the renderer retried, a
    missing placeholder would retry forever; if the version were not recorded,
    nobody could tell which edit broke it. Both problems are solved by the same
    two decisions — fail immediately, and carry the version.
    """

    def render(self, notification: Notification, template: Template,
               destination: str) -> RenderedMessage:
        try:
            title = template.title.format(**notification.payload)
            body = template.body.format(**notification.payload)
        except (KeyError, IndexError, ValueError) as exc:
            raise RenderError(
                f"{template.template_id} v{template.version} "
                f"({template.channel.value}/{template.locale}): {exc}") from exc
        return RenderedMessage(
            notification_id=notification.id,
            channel=notification.channel,
            destination=destination,
            title=title,
            body=body,
            template_id=template.template_id,
            template_version=template.version,
            locale=template.locale,
            event_type=notification.event_type,
            data=dict(notification.payload),
        )


# ==========================================================================
# 7. Destinations, and the token hygiene the HLD asks for
# ==========================================================================

class TokenStore:
    """Device tokens, phone numbers and email addresses, one binding per
    (user, channel).

    `delete` is what a SUPPRESS_TOKEN failure calls. Retrying an invalid token
    burns quota on a message that can never arrive, and every retry counts
    against a provider limit shared with the password resets — so this is not
    tidiness, it is capacity.
    """

    def __init__(self, clock: Clock) -> None:
        self._bindings: dict[tuple[str, ChannelKind], tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def bind(self, user_id: str, channel: ChannelKind, token: str) -> None:
        with self._lock:
            self._bindings[(user_id, channel)] = (token, self._clock.now())

    def destination_for(self, user_id: str,
                        channel: ChannelKind) -> Optional[str]:
        with self._lock:
            binding = self._bindings.get((user_id, channel))
            return None if binding is None else binding[0]

    def delete(self, user_id: str, channel: ChannelKind) -> bool:
        with self._lock:
            return self._bindings.pop((user_id, channel), None) is not None

    def prune_stale(self, max_age_s: float = 90 * 24 * 3600.0) -> int:
        """Tokens not refreshed in 90 days. A dropping refresh rate is the
        earliest signal that users are uninstalling, which is why this is a
        scheduled job and not a cleanup."""
        cutoff = self._clock.now() - max_age_s
        with self._lock:
            stale = [k for k, (_, bound_at) in self._bindings.items()
                     if bound_at < cutoff]
            for key in stale:
                del self._bindings[key]
            return len(stale)

    def size(self) -> int:
        with self._lock:
            return len(self._bindings)


# ==========================================================================
# 8. The channels — thin adapters, and the classification lives above them
# ==========================================================================

class Channel(ABC):
    @property
    @abstractmethod
    def kind(self) -> ChannelKind:
        raise NotImplementedError

    @abstractmethod
    def send(self, message: RenderedMessage) -> SendOutcome:
        raise NotImplementedError


class _ProviderChannel(Channel):
    """Shared shape for all three. The provider call is INJECTED, which is what
    makes a channel testable without a provider: the demo scripts an exact
    sequence of error codes, including the ones nobody wants to reproduce on
    purpose (a 429 followed by a hard bounce followed by a 410)."""

    def __init__(self, channel_kind: ChannelKind, table: ClassificationTable,
                 provider: Callable[[RenderedMessage], tuple[str, str]]) -> None:
        self._kind = channel_kind
        self._table = table
        self._provider = provider
        self.calls: list[str] = []

    @property
    def kind(self) -> ChannelKind:
        return self._kind

    def send(self, message: RenderedMessage) -> SendOutcome:
        self.calls.append(message.notification_id)
        status, provider_id = self._provider(message)
        if status == "OK":
            return SendOutcome.ok(provider_id)
        retry_after = None
        if status == "429":
            # The demo's provider encodes Retry-After in the id field, which is
            # how the honour-the-header path gets exercised.
            try:
                retry_after = float(provider_id)
            except ValueError:
                retry_after = None
        return SendOutcome.fail(status, self._table, retry_after)


class PushChannel(_ProviderChannel):
    def __init__(self, table: ClassificationTable,
                 provider: Callable[[RenderedMessage], tuple[str, str]]) -> None:
        super().__init__(ChannelKind.PUSH, table, provider)


class EmailChannel(_ProviderChannel):
    def __init__(self, table: ClassificationTable,
                 provider: Callable[[RenderedMessage], tuple[str, str]]) -> None:
        super().__init__(ChannelKind.EMAIL, table, provider)


class SmsChannel(_ProviderChannel):
    """SMS is gated behind an explicit event allow-list, and the gate is here
    rather than in the decision service so it cannot be bypassed by a caller
    that goes straight to the channel. Cost and compliance both argue for it:
    a retried SMS at $0.0075 across 5M messages is $187,500 in one incident,
    and an inbound STOP is a legal suppression."""

    SMS_ALLOWED: frozenset[str] = frozenset({
        "auth.otp", "security.alert", "fraud.alert", "order.delivered",
    })

    def __init__(self, table: ClassificationTable,
                 provider: Callable[[RenderedMessage], tuple[str, str]],
                 allowed: Iterable[str] = SMS_ALLOWED) -> None:
        super().__init__(ChannelKind.SMS, table, provider)
        self.allowed = frozenset(allowed)

    def send(self, message: RenderedMessage) -> SendOutcome:
        if message.event_type not in self.allowed:
            return SendOutcome(False, None, "CHANNEL_NOT_ALLOWED",
                               FailureKind.TERMINAL)
        return super().send(message)


class ChannelRegistry:
    def __init__(self, channels: Iterable[Channel]) -> None:
        self._by_kind = {c.kind: c for c in channels}

    def get(self, kind: ChannelKind) -> Channel:
        channel = self._by_kind.get(kind)
        if channel is None:
            raise KeyError(f"no channel registered for {kind.value}")
        return channel

    def all(self) -> dict[ChannelKind, Channel]:
        return dict(self._by_kind)


# ==========================================================================
# 9. Scheduling, and the local-date key that survives a DST transition
# ==========================================================================

@dataclass(frozen=True)
class ScheduledJob:
    campaign_id: str
    request: NotificationRequest
    fire_at: float
    zone: str


class Scheduler:
    """Scheduled campaigns, idempotent per (user, campaign, LOCAL DATE).

    The key is the design. `(campaign, user)` alone means a weekly digest fires
    once ever. `(campaign, user, fire_at)` means a DST overlap — the hour that
    happens twice — fires twice. The local date is the only one of the three
    that is stable under a clock transition and still repeats, so it is what
    makes "at most one digest per user per day, no matter how many times the job
    runs" a property of the key rather than of the retry logic.

    Note the store it is given needs a TTL LONGER than the day it protects —
    the key is scoped to a local date, so a 24-hour TTL can expire at 23:00 and
    let the same day's digest through a second time. The demo passes 48 hours.
    """

    def __init__(self, idempotency: IdempotencyStore, tz: Timezone,
                 clock: Clock) -> None:
        self._idempotency = idempotency
        self._tz = tz
        self._clock = clock
        self._jobs: list[ScheduledJob] = []
        self._lock = threading.RLock()

    def _key(self, job: ScheduledJob, user_id: str, now: float) -> str:
        local_date = self._tz.local_date_key(job.zone, now)
        return f"schedule:{job.campaign_id}:{user_id}:{local_date}"

    def enqueue(self, job: ScheduledJob) -> int:
        """Returns how many recipients were newly claimed. A second enqueue of
        the same campaign on the same local day claims nobody, which is what
        makes the scheduler safe to run from two places at once."""
        now = self._clock.now()
        claimed = 0
        for user_id in job.request.user_ids:
            key = self._key(job, user_id, now)
            fresh, _ = self._idempotency.claim(key, job.request.request_id, now)
            claimed += 1 if fresh else 0
        if claimed:
            with self._lock:
                self._jobs.append(job)
        return claimed

    def due(self, now: Optional[float] = None) -> list[ScheduledJob]:
        at = self._clock.now() if now is None else now
        with self._lock:
            ready = [j for j in self._jobs if j.fire_at <= at]
            self._jobs = [j for j in self._jobs if j.fire_at > at]
            return ready

    def pending(self) -> int:
        with self._lock:
            return len(self._jobs)


# ==========================================================================
# 10. The webhook: fast in, slow out, and two different classes
# ==========================================================================

class SignatureError(Exception):
    """An unverified webhook. The HLD is explicit that signature verification
    happens before parsing, because parsing attacker-controlled bytes with a
    JSON parser is where the cheap vulnerabilities live."""


class WebhookHandler:
    """Validate, translate, publish, RETURN.

    The HLD's number for this is p99 under 50ms, and the reason is that
    providers retry anything slower — a slow webhook endpoint is a webhook
    endpoint that gets the same event five times and then gets turned off. So
    this class does no state writes at all. That is a structural property, not a
    discipline: it holds no reference to the state store, so it cannot write to
    it even by accident.
    """

    LEVELS: dict[str, NotificationState] = {
        "accepted": NotificationState.ACCEPTED,
        "sent": NotificationState.SENT,
        "delivered": NotificationState.DELIVERED,
        "opened": NotificationState.OPENED,
        "clicked": NotificationState.CLICKED,
        "bounced": NotificationState.FAILED,
    }

    def __init__(self, verifier: Callable[[bytes, str], bool],
                 publish: Callable[[DeliveryEvent], None],
                 analytics: AnalyticsSink, clock: Clock) -> None:
        self._verify = verifier
        self._publish = publish
        self._analytics = analytics
        self._clock = clock
        self.rejected = Counter()

    def handle(self, raw: bytes, signature: str) -> DeliveryEvent:
        if not self._verify(raw, signature):
            self.rejected.bump()
            raise SignatureError("signature did not verify")
        payload = json.loads(raw.decode("utf-8"))
        level = self.LEVELS.get(payload["event"])
        if level is None:
            raise ValueError(f"unknown webhook event: {payload['event']!r}")
        event = DeliveryEvent(
            provider_message_id=payload["provider_message_id"],
            level=level,
            occurred_at=float(payload.get("at", self._clock.now())),
            reason=payload.get("reason", ""),
        )
        self._analytics.record(event)
        self._publish(event)          # hand off; the consumer does the writing
        return event


class DeliveryEventConsumer:
    """The slow half, and it is a separate class so that "the handler does not
    touch the database" is a fact about the object graph rather than a promise
    in a comment.

    Everything hard about webhooks is here: they arrive at least once, out of
    order, and sometimes for a message we have no record of.
    """

    def __init__(self, state_store: NotificationStateStore,
                 analytics: AnalyticsSink) -> None:
        self._store = state_store
        self._analytics = analytics
        # Three counters, because the three ways a webhook fails to apply have
        # three different owners: unknown means an environment or retention
        # problem, duplicates come from the provider's retry policy, and
        # out-of-order arrivals come from its delivery pipeline. One "skipped"
        # counter would merge them into a number nobody can act on.
        self.unknown = Counter()
        self.duplicates = Counter()
        self.out_of_order = Counter()

    def consume(self, event: DeliveryEvent) -> bool:
        notification = self._store.find_by_provider_id(event.provider_message_id)
        if notification is None:
            # Not an error. A provider can report on a message accepted by a
            # different environment, or one whose rows were archived. Counting
            # it is what turns "a few" into a number somebody can alarm on.
            self.unknown.bump()
            return False
        if notification.state is event.level:
            # The at-least-once contract means this is the NORMAL case, not the
            # exceptional one. Returning False is the useful answer: it is what
            # makes the duplicate rate a metric rather than a shrug.
            self.duplicates.bump()
            return False
        try:
            self._store.transition(notification.id, event.level, event.reason)
        except IllegalTransition:
            # Out of order — a `sent` that arrives after its own `delivered`.
            self.out_of_order.bump()
            return False
        self._analytics.record(event)
        return True


class InMemoryAnalytics(AnalyticsSink):
    def __init__(self) -> None:
        self.events: list[DeliveryEvent] = []
        self._lock = threading.Lock()

    def record(self, event: DeliveryEvent) -> None:
        with self._lock:
            self.events.append(event)

    def by_level(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for event in self.events:
                out[event.level.value] = out.get(event.level.value, 0) + 1
            return out


# ==========================================================================
# 11. The sender — where a failure becomes a classification decision
# ==========================================================================

@dataclass(frozen=True)
class SendResult:
    notification_id: str
    state: NotificationState
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.notification_id} {self.state.value} {self.detail}".strip()


class Sender:
    """One attempt at one notification, and the whole failure taxonomy in one
    place.

    Read this method as the answer to "what happens when a send fails". There
    are exactly four outcomes and they are DISTINGUISHABLE: accepted, delayed by
    the limiter, retried later, terminal. The design failure this class exists to
    prevent is the two-outcome version — retry everything, or drop everything —
    where an invalid device token is retried until the budget runs out and a
    provider 503 is dropped as if it were permanent.
    """

    OTP_WINDOW_S = 600.0
    # Events whose value expires. Only these get the fail-fast path, because for
    # everything else a delayed notification is still a useful one.
    FAIL_FAST_EVENTS: frozenset[str] = frozenset({"auth.otp", "password.reset"})

    def __init__(self, registry: ChannelRegistry,
                 limiters: dict[ChannelKind, ProviderLimiter],
                 retry: RetryPolicy, state_store: NotificationStateStore,
                 dlq: DeadLetterQueue, templates: TemplateStore,
                 renderer: TemplateRenderer, tokens: TokenStore,
                 clock: Clock) -> None:
        self._registry = registry
        # Held BY REFERENCE, not copied. Raising a provider's limit during an
        # incident is a change somebody makes at 3 a.m. without a deploy, and a
        # constructor copy would mean the change silently does nothing until the
        # process restarts — which is the worst kind of operational control.
        self._limiters = limiters
        self._retry = retry
        self._store = state_store
        self._dlq = dlq
        self._templates = templates
        self._renderer = renderer
        self._tokens = tokens
        self._clock = clock

    def send(self, notification: Notification,
             now: Optional[float] = None) -> SendResult:
        at = self._clock.now() if now is None else now

        # (a) The provider limit first: it is the only step that costs nothing
        #     when it refuses, and doing it after the render means a saturated
        #     provider still pays for template lookup and rendering.
        limiter = self._limiters.get(notification.channel)
        if limiter is not None:
            wait = limiter.acquire(notification.priority)
            if wait > 0.0:
                if (notification.event_type in self.FAIL_FAST_EVENTS
                        and wait > self.OTP_WINDOW_S):
                    return self._finish(
                        notification, NotificationState.FAILED, at,
                        "PROVIDER_SATURATED", 
                        f"wait {wait:.0f}s exceeds the {self.OTP_WINDOW_S:.0f}s "
                        f"validity window, so the OTP would be useless")
                notification.scheduled_at = at + wait
                return SendResult(notification.id, NotificationState.PENDING,
                                  f"provider limited, delayed {wait:.1f}s")

        # (b) The EXACT template version the decision was made against.
        template = self._templates.get_version(
            notification.template_id, notification.channel,
            notification.locale, notification.template_version)
        if template is None:
            return self._finish(
                notification, NotificationState.FAILED, at,
                "TEMPLATE_VERSION_MISSING",
                f"v{notification.template_version} is not in the store")

        # (c) Render. A missing placeholder fails identically every time, so it
        #     is terminal by construction.
        try:
            message = self._renderer.render(notification, template,
                                            notification.destination)
        except RenderError as exc:
            return self._finish(notification, NotificationState.FAILED, at,
                                "RENDER_FAILED", str(exc))

        # (d) The attempt itself.
        channel = self._registry.get(notification.channel)
        outcome = channel.send(message)
        notification.attempt += 1

        if outcome.accepted:
            self._store.record_provider_id(notification.id,
                                           outcome.provider_message_id or "")
            self._store.transition(notification.id, NotificationState.ACCEPTED)
            return SendResult(notification.id, NotificationState.ACCEPTED,
                              f"accepted as {outcome.provider_message_id}")

        # (e) SUPPRESS_TOKEN — a THIRD outcome, and the one most systems miss.
        #     It is terminal for this notification but it is not a failure of the
        #     send; the destination is gone, so the fix is to stop having it.
        #     Without this branch a hard bounce is retried five times against a
        #     provider limit shared with the password resets.
        if outcome.kind is FailureKind.SUPPRESS_TOKEN:
            self._tokens.delete(notification.user_id, notification.channel)
            return self._finish(notification, NotificationState.FAILED, at,
                                outcome.error_code,
                                "destination removed from the token store")

        if outcome.kind is FailureKind.TERMINAL:
            return self._finish(notification, NotificationState.FAILED, at,
                                outcome.error_code, "terminal per the table")

        # (f) RETRYABLE. The budget is spent on ATTEMPTS, not on elapsed time,
        #     so a notification that sat in a queue for an hour does not
        #     exhaust its retries while nobody was looking.
        next_at = self._retry.next_attempt(notification.attempt,
                                           notification.priority, at,
                                           outcome.retry_after_s)
        if next_at is None:
            return self._finish(notification, NotificationState.DEAD, at,
                                outcome.error_code,
                                f"retry budget spent after "
                                f"{notification.attempt} attempts")
        notification.scheduled_at = next_at
        return SendResult(notification.id, NotificationState.PENDING,
                          f"{outcome.error_code}, retry in "
                          f"{next_at - at:.0f}s (attempt "
                          f"{notification.attempt + 1})")

    def _finish(self, notification: Notification, state: NotificationState,
                at: float, error_code: str, detail: str) -> SendResult:
        """Both end states route through here, and both land in the DLQ. A DLQ
        that only collects retry exhaustion is a DLQ that hides every permanent
        failure — which is most of them."""
        self._store.transition(notification.id, state, error_code)
        self._dlq.add(DeadLetter(
            notification_id=notification.id,
            request_id=notification.request_id,
            user_id=notification.user_id,
            channel=notification.channel,
            attempts=notification.attempt,
            last_error=f"{error_code}: {detail}",
            at=at,
        ))
        return SendResult(notification.id, state, f"{error_code} — {detail}")


class Dispatcher:
    """Drains what is due, in priority order, one pass at a time.

    `run_once` returns after one pass rather than looping forever, which is what
    makes the whole system testable with a manual clock: the demo calls it once
    per simulated tick and asserts on what came out. A real worker is
    `while True: dispatcher.run_once(); sleep(1)` — the loop belongs to the
    process, not to the class.
    """

    def __init__(self, state_store: NotificationStateStore, sender: Sender,
                 clock: Clock) -> None:
        self._store = state_store
        self._sender = sender
        self._clock = clock

    def run_once(self, now: Optional[float] = None) -> list[SendResult]:
        at = self._clock.now() if now is None else now
        results = []
        for notification in self._store.due(at):
            results.append(self._sender.send(notification, at))
        return results


# ==========================================================================
# 12. The decision service — the policy engine, in one readable order
# ==========================================================================

@dataclass(frozen=True)
class ChannelDecision:
    user_id: str
    channel: ChannelKind
    send: bool
    reason: str
    notification: Optional[Notification] = None


class DecisionService:
    """`decide` is the heart of the system: one request in, a set of state rows
    out.

    THE ORDER OF THE CHECKS IS THE DESIGN, and it is chosen so that the cheapest
    and most decisive test comes first, and so that nothing which is going to be
    dropped consumes a resource:

      1. consent      — legal, and it overrides everything, including priority
      2. preference   — the user's product choice, per channel and per category
      3. destination  — is there anywhere to send it to
      4. priority     — DERIVED from the event type, never supplied by a caller
      5. quiet hours  — DROP (LOW) or DEFER (everything else), before anything
                        is counted, because a dropped message must not consume
                        rate-limit budget or an idempotency key
      6. dedupe       — claim the per-recipient key; a duplicate stops here
      7. rate limit   — per user, per category, per local day
      8. template     — resolve the version to record on the row

    Steps 6 and 7 are in this order for a specific reason: reversed, a producer
    retrying its own POST would consume the user's daily marketing budget on
    messages that are then discarded as duplicates, and the user would silently
    receive nothing.
    """

    def __init__(self, prefs: PreferenceStore, consent: ConsentStore,
                 dedupe: DedupeStore, templates: TemplateStore,
                 tokens: TokenStore, quiet: QuietHours, rate: UserRateLimiter,
                 state_store: NotificationStateStore, tz: Timezone, clock: Clock,
                 event_priority: Optional[dict[str, Priority]] = None,
                 dedupe_ttl_s: float = 24 * 3600.0) -> None:
        self._prefs = prefs
        self._consent = consent
        self._dedupe = dedupe
        self._templates = templates
        self._tokens = tokens
        self._quiet = quiet
        self._rate = rate
        self._store = state_store
        self._tz = tz
        self._clock = clock
        self._event_priority = dict(event_priority or EVENT_PRIORITY)
        self._dedupe_ttl_s = dedupe_ttl_s

    def priority_for(self, event_type: str) -> Priority:
        """Unknown event types default to NORMAL, and the default is visible
        rather than buried: a new event type that nobody classified gets treated
        like an ordinary notification, which is the failure that gets noticed
        and fixed, instead of LOW, which is the failure that is silent."""
        return self._event_priority.get(event_type, Priority.NORMAL)

    def decide(self, request: NotificationRequest,
               now: Optional[float] = None) -> DeliveryPlan:
        at = self._clock.now() if now is None else now
        category = request.event_type.split(".")[0]
        priority = self.priority_for(request.event_type)
        made: list[Notification] = []
        dropped: list[tuple[str, str]] = []

        for user_id in request.user_ids:
            prefs = self._prefs.load(user_id)
            local = self._tz.local_time(prefs.timezone, at)
            for channel in request.channels:
                decision = self._decide_one(request, user_id, channel, category,
                                            priority, prefs, local, at)
                if decision.notification is not None:
                    # WRITTEN BEFORE IT IS RETURNED. The caller enqueues after
                    # this returns; if the process dies in between, the row
                    # still says PENDING and the dispatcher will find it. A
                    # notification that exists only in a queue is a notification
                    # nobody can audit.
                    self._store.put(decision.notification)
                    made.append(decision.notification)
                else:
                    dropped.append((user_id, decision.reason))

        return DeliveryPlan(request.request_id, tuple(made), tuple(dropped))

    def _decide_one(self, request: NotificationRequest, user_id: str,
                    channel: ChannelKind, category: str, priority: Priority,
                    prefs: UserPreferences, local: datetime,
                    at: float) -> ChannelDecision:
        def no(reason: str) -> ChannelDecision:
            return ChannelDecision(user_id, channel, False, reason)

        # 1. Consent. Not a preference, and it cannot be overridden.
        if not self._consent.is_allowed(user_id, channel):
            return no("consent withdrawn")

        # 2. Preference: the channel, then the category.
        if not prefs.channel_enabled(channel):
            return no(f"{channel.value} disabled in settings")
        if not prefs.category_enabled(category):
            return no(f"category {category!r} disabled in settings")
        if category == "marketing" and prefs.marketing_opt_out:
            return no("marketing opt-out")

        # 3. Destination.
        destination = self._tokens.destination_for(user_id, channel)
        if destination is None:
            return no(f"no {channel.value} destination on file")

        # 4 & 5. Quiet hours, on the derived priority and the REAL event type.
        quiet_decision = self._quiet.decide(
            priority, request.event_type, local,
            prefs.quiet_start_hour, prefs.quiet_end_hour)
        if quiet_decision is QuietDecision.DROP:
            return no(f"quiet hours, {priority.value} priority is dropped")
        scheduled_at = request.scheduled_at
        if quiet_decision is QuietDecision.DEFER:
            # The wall clock is the specification; the epoch is derived from it,
            # with the offset resolved at the TARGET instant rather than at now.
            deferred_to = self._tz.next_local_time(prefs.timezone, at,
                                                   prefs.quiet_end_hour)
            scheduled_at = (deferred_to if scheduled_at is None
                            else max(scheduled_at, deferred_to))

        # 6. Dedupe.
        if not self._dedupe.claim(request.dedupe_key, user_id, channel, at,
                                  self._dedupe_ttl_s):
            return no("duplicate of an earlier notification")

        # 7. Per-user rate limit.
        allowed, remaining = self._rate.check_and_count(
            user_id, category, prefs.timezone)
        if not allowed:
            return no(f"daily {category} limit reached")

        # 8. Template, and the VERSION is recorded on the row.
        template = self._templates.pick(request.event_type, channel,
                                        request.locale)
        if template is None:
            return no(f"no template for {request.event_type}/{channel.value}")

        notification = Notification(
            id=f"n_{request.request_id}_{user_id}_{channel.value}",
            request_id=request.request_id,
            user_id=user_id,
            channel=channel,
            category=category,
            event_type=request.event_type,
            priority=priority,
            destination=destination,
            template_id=template.template_id,
            template_version=template.version,
            dedupe_key=request.dedupe_key,
            payload=dict(request.payload),
            locale=request.locale,
            scheduled_at=scheduled_at,
        )
        note = "" if quiet_decision is QuietDecision.SEND_NOW else "deferred"
        return ChannelDecision(user_id, channel, True,
                               note or f"queued, {remaining} left today",
                               notification)


# ==========================================================================
# 13. The API — the producer's entry point, and the first gate
# ==========================================================================

class NotificationApi:
    """`submit` is the only door in, and the idempotency claim is the first
    thing behind it.

    FIRST, not last: the producer's retry and the load balancer's retry are the
    same POST, and if the claim happens after the decision service runs, both
    copies have already been decided, deduped against nothing, and written.
    Claiming first means a retry returns the ORIGINAL request id — which is what
    lets the caller poll for the result of the request it actually made.
    """

    def __init__(self, idempotency: IdempotencyStore,
                 decisions: DecisionService,
                 state_store: NotificationStateStore, clock: Clock) -> None:
        self._idempotency = idempotency
        self._decisions = decisions
        self._store = state_store
        self._clock = clock

    def submit(self, request: NotificationRequest,
               idempotency_key: str) -> tuple[DeliveryPlan, bool]:
        """Returns (plan, was_replay). A replay returns the plan REBUILT from
        the stored rows, so the response is identical to the original — a
        replay that returned an empty plan would make a retrying producer
        believe nothing was sent."""
        at = self._clock.now()
        fresh, existing_id = self._idempotency.claim(
            idempotency_key, request.request_id, at)
        if not fresh:
            replay_id = existing_id or request.request_id
            rows = self._store.for_request(replay_id)
            return (DeliveryPlan(replay_id, tuple(rows), ()), True)

        plan = self._decisions.decide(request, at)
        return plan, False

    def status(self, request_id: str) -> dict[str, object]:
        """The operational answer, and it is deliberately TWO values: the
        sender-facing progress and the per-channel histogram. A single status
        field cannot answer both "did the user get it" and "which channel is
        burning money", and whichever one it answers, somebody will read it as
        the other."""
        return {
            "request_id": request_id,
            "progress": self._store.progress(request_id).value,
            "counts": self._store.counts(request_id),
        }


# ==========================================================================
# 14. The demo — every claim above, executed
# ==========================================================================

PASSED: list[int] = [0]
FAILED: list[int] = [0]


def check(label: str, condition: bool) -> None:
    if condition:
        PASSED[0] += 1
        print(f"PASS {label}")
    else:
        FAILED[0] += 1
        print(f"FAIL {label}")


# Two instants used throughout. T_NIGHT is 2023-11-14 22:13:20 UTC, which is
# inside a 22:00-08:00 quiet window everywhere in this demo; T_DAY is twelve
# hours later and outside it.
T_NIGHT = 1_700_000_000.0
T_DAY = T_NIGHT + 12 * 3600.0


def dst_resolver(zone: str, at_epoch: float) -> float:
    """A REAL offset resolver, written out, because the demo's job is to prove
    the offset is resolved at send time rather than baked in. Production passes
    `zoneinfo.ZoneInfo`; this is the same idea with the US rule spelled out, so
    the DST assertions below are real assertions and not a fixed table that
    would have hidden the bug.
    """
    if zone == "UTC":
        return 0.0
    if zone == "Asia/Kolkata":
        return 5.5                       # no DST, which is why it is a good control
    if zone != "America/New_York":
        raise KeyError(f"unknown zone: {zone!r}")
    utc = datetime.fromtimestamp(at_epoch, tz=dt_timezone.utc)
    march = datetime(utc.year, 3, 1, tzinfo=dt_timezone.utc)
    second_sunday = march + timedelta(days=(6 - march.weekday()) % 7 + 7)
    november = datetime(utc.year, 11, 1, tzinfo=dt_timezone.utc)
    first_sunday = november + timedelta(days=(6 - november.weekday()) % 7)
    # 02:00 local is 07:00 UTC on the way in and 06:00 UTC on the way out.
    dst_start = second_sunday + timedelta(hours=7)
    dst_end = first_sunday + timedelta(hours=6)
    return -4.0 if dst_start <= utc < dst_end else -5.0


class World:
    """Everything wired together, fresh per scenario.

    The clock is manual and the provider responses are a SCRIPT, and both are
    the point: every interesting behaviour in this system is a function of time
    or of a provider error code, so a test that controls neither can only test
    the parts that do not matter.
    """

    def __init__(self, start: float = T_DAY,
                 push_script: Optional[list] = None,
                 email_script: Optional[list] = None,
                 sms_script: Optional[list] = None) -> None:
        self.clock = ManualClock(start)
        self.tz = Timezone(dst_resolver)
        self.table = ClassificationTable()
        self.analytics = InMemoryAnalytics()
        self.idempotency = IdempotencyStore()
        self.dedupe = DedupeStore()
        self.state = NotificationStateStore()
        self.dlq = DeadLetterQueue()
        self.templates = TemplateStore()
        self.renderer = TemplateRenderer()
        self.tokens = TokenStore(self.clock)
        self.prefs = PreferenceStore()
        self.consent = ConsentStore()
        self.quiet = QuietHours()
        self.rate = UserRateLimiter(self.clock, self.tz)
        self.retry = RetryPolicy(jitter=0.2, rng=random.Random(7))
        self.calls: dict[ChannelKind, list[str]] = {}
        self.rendered: list[RenderedMessage] = []
        self.scripts = {
            ChannelKind.PUSH: list(push_script or []),
            ChannelKind.EMAIL: list(email_script or []),
            ChannelKind.SMS: list(sms_script or []),
        }
        register_templates(self.templates)
        self.registry = ChannelRegistry([
            PushChannel(self.table, self._provider(ChannelKind.PUSH)),
            EmailChannel(self.table, self._provider(ChannelKind.EMAIL)),
            SmsChannel(self.table, self._provider(ChannelKind.SMS)),
        ])
        self.limiters: dict[ChannelKind, ProviderLimiter] = {}
        self.sender = Sender(self.registry, self.limiters, self.retry,
                             self.state, self.dlq, self.templates,
                             self.renderer, self.tokens, self.clock)
        self.dispatcher = Dispatcher(self.state, self.sender, self.clock)
        self.decisions = DecisionService(
            self.prefs, self.consent, self.dedupe, self.templates, self.tokens,
            self.quiet, self.rate, self.state, self.tz, self.clock)
        self.api = NotificationApi(self.idempotency, self.decisions,
                                   self.state, self.clock)
        self.scheduler = Scheduler(IdempotencyStore(ttl_s=48 * 3600.0),
                                   self.tz, self.clock)

    def _provider(self, kind: ChannelKind):
        """The scripted provider. An entry is either an error code, or a
        `(code, retry_after)` pair. An exhausted script succeeds, so a test that
        only cares about the decision path does not have to script anything."""
        def provider(message: RenderedMessage) -> tuple[str, str]:
            self.rendered.append(message)
            script = self.scripts[kind]
            entry = script.pop(0) if script else "OK"
            if isinstance(entry, tuple):
                status, extra = entry
            else:
                status, extra = entry, ""
            self.calls.setdefault(kind, []).append(status)
            if status == "OK":
                return "OK", f"pm_{kind.value}_{len(self.calls[kind])}"
            return status, extra
        return provider

    def add_user(self, user_id: str, zone: str = "UTC",
                 channels: Optional[dict] = None,
                 destinations: Optional[dict] = None,
                 **prefs_kwargs) -> None:
        self.prefs.put(UserPreferences(user_id, timezone=zone,
                                      channels=channels or {}, **prefs_kwargs))
        for channel, token in (destinations or {}).items():
            self.tokens.bind(user_id, channel, token)

    def request(self, request_id: str, event_type: str, entity_id: str,
                users: tuple, channels: tuple, payload: Optional[dict] = None
                ) -> NotificationRequest:
        return NotificationRequest(request_id, event_type, entity_id, users,
                                   channels, payload or {})


def register_templates(store: TemplateStore) -> None:
    T = Template
    P, E, S = ChannelKind.PUSH, ChannelKind.EMAIL, ChannelKind.SMS
    store.put(T("order.shipped", P, "en", 1, "Order shipped",
                "Order {order_id} is on its way."))
    store.put(T("order.shipped", E, "en", 1, "Order shipped",
                "Order {order_id} is on its way."))
    store.put(T("marketing.campaign", P, "en", 1, "A treat for you",
                "{offer} is waiting in the app."))
    store.put(T("security.alert", P, "en", 1, "New sign-in",
                "A new sign-in from {city}. If this was not you, reset now."))
    store.put(T("auth.otp", P, "en", 1, "Your code",
                "Your code is {code}. It expires in 10 minutes."))
    store.put(T("auth.otp", E, "en", 1, "Your code",
                "Your code is {code}. It expires in 10 minutes."))


def make_notification(notification_id: str = "n_x", user_id: str = "u1",
                      channel: ChannelKind = ChannelKind.PUSH,
                      event_type: str = "order.shipped",
                      priority: Priority = Priority.NORMAL,
                      request_id: str = "r_x",
                      template_version: int = 1, **overrides) -> Notification:
    fields = dict(
        id=notification_id, request_id=request_id, user_id=user_id,
        channel=channel, category=event_type.split(".")[0],
        event_type=event_type, priority=priority, destination="tok_1",
        template_id=event_type, template_version=template_version,
        dedupe_key=f"{event_type}:e_1", payload={"order_id": "o_1"},
    )
    fields.update(overrides)
    return Notification(**fields)


# --------------------------------------------------------------------------
# T1. The classification is DATA, and the default direction is deliberate
# --------------------------------------------------------------------------
w = World()
check("T1a unknown code defaults to RETRYABLE (the loud failure)",
      w.table.classify("SOME_NEW_CODE_42") is FailureKind.RETRYABLE)
check("T1b 410 suppresses the token rather than retrying",
      w.table.classify("410") is FailureKind.SUPPRESS_TOKEN)
check("T1c a hard bounce is TERMINAL, not SUPPRESS_TOKEN",
      w.table.classify("HARD_BOUNCE") is FailureKind.TERMINAL)
check("T1d the table is queryable during an incident",
      "503" in w.table.codes_for(FailureKind.RETRYABLE))
check("T1e an override wins over the default table",
      ClassificationTable({"SOME_NEW_CODE_42": FailureKind.TERMINAL})
      .classify("SOME_NEW_CODE_42") is FailureKind.TERMINAL)
check("T1f jitter actually varies the delay for the same inputs",
      RetryPolicy(0.2, random.Random(1)).next_attempt(1, Priority.NORMAL, 0.0)
      != RetryPolicy(0.2, random.Random(2)).next_attempt(1, Priority.NORMAL, 0.0))

# --------------------------------------------------------------------------
# T2. Quiet hours: PRIORITY decides defer-vs-drop, the EVENT TYPE decides
#     whether the window applies at all
# --------------------------------------------------------------------------
w = World(start=T_NIGHT)
w.add_user("u_utc", zone="UTC", destinations={ChannelKind.PUSH: "tok_utc"})
w.add_user("u_ist", zone="Asia/Kolkata",
           destinations={ChannelKind.PUSH: "tok_ist"})

marketing = w.decisions.decide(
    w.request("r1", "marketing.campaign", "c_1", ("u_utc",),
              (ChannelKind.PUSH,), {"offer": "20% off"}))
check("T2a LOW inside quiet hours is DROPPED, not deferred",
      len(marketing) == 0 and "quiet hours" in marketing.suppressed[0][1])

order = w.decisions.decide(
    w.request("r2", "order.shipped", "o_1", ("u_utc",), (ChannelKind.PUSH,),
              {"order_id": "o_1"}))
check("T2b NORMAL inside quiet hours is DEFERRED",
      len(order) == 1
      and order.notifications[0].scheduled_at == T_NIGHT + 35200.0)
check("T2c the deferral lands exactly at 08:00 wall clock",
      w.tz.local_time("UTC", order.notifications[0].scheduled_at).hour == 8)

alert = w.decisions.decide(
    w.request("r3", "security.alert", "s_1", ("u_utc",), (ChannelKind.PUSH,),
              {"city": "Lisbon"}))
check("T2d a HIGH event ON the allow-list sends now",
      len(alert) == 1 and alert.notifications[0].scheduled_at is None)

otp = w.decisions.decide(
    w.request("r4", "auth.otp", "o_9", ("u_utc",), (ChannelKind.PUSH,),
              {"code": "123456"}))
check("T2e OTP is on the allow-list and bypasses the window",
      len(otp) == 1 and otp.notifications[0].scheduled_at is None)

check("T2f a DROPPED notification consumes no per-user budget, because the "
      "quiet-hours check runs before the counter",
      w.rate.used("u_utc", "marketing", "UTC") == 0)

ist = w.decisions.decide(
    w.request("r5", "order.shipped", "o_2", ("u_ist",), (ChannelKind.PUSH,),
              {"order_id": "o_2"}))
check("T2g the same instant is quiet hours in Kolkata too (03:43 local)",
      len(ist) == 1
      and w.tz.local_time("Asia/Kolkata",
                          ist.notifications[0].scheduled_at).hour == 8)

# --------------------------------------------------------------------------
# T3. The state machine is forward-only
# --------------------------------------------------------------------------
w = World()
n = make_notification("n_sm")
w.state.put(n)
w.state.transition("n_sm", NotificationState.ACCEPTED)
w.state.transition("n_sm", NotificationState.DELIVERED)
check("T3a the lifecycle advances", w.state.get("n_sm").state
      is NotificationState.DELIVERED)
backwards = False
try:
    w.state.transition("n_sm", NotificationState.ACCEPTED)
except IllegalTransition:
    backwards = True
check("T3b an out-of-order webhook cannot walk it BACKWARDS", backwards)
re_sent = False
try:
    w.state.transition("n_sm", NotificationState.SENT)
except IllegalTransition:
    re_sent = True
check("T3c SENT after DELIVERED is refused for the same reason", re_sent)
check("T3d a repeat of the current state is a no-op, not an error",
      w.state.transition("n_sm", NotificationState.DELIVERED).state
      is NotificationState.DELIVERED)
w.state.transition("n_sm", NotificationState.OPENED)
check("T3e OPENED after DELIVERED is fine — the rank is what orders them",
      w.state.get("n_sm").state is NotificationState.OPENED)

# --------------------------------------------------------------------------
# T4. The webhook handler writes nothing; the consumer does the writing
# --------------------------------------------------------------------------
w = World(push_script=["OK"])
w.add_user("u1", zone="UTC", destinations={ChannelKind.PUSH: "tok_1"})
plan, replay = w.api.submit(
    w.request("r_wh", "order.shipped", "o_7", ("u1",), (ChannelKind.PUSH,),
              {"order_id": "o_7"}), "idem_wh")
w.dispatcher.run_once()
n_id = plan.notifications[0].id
provider_id = w.state.get(n_id).provider_message_id
check("T4a accepted, and the provider id is on the row",
      w.state.get(n_id).state is NotificationState.ACCEPTED
      and provider_id == "pm_push_1")

published: list = []


def publish(event: DeliveryEvent) -> None:
    published.append(event)


handler = WebhookHandler(lambda raw, sig: sig == "good", publish,
                         w.analytics, w.clock)
consumer = DeliveryEventConsumer(w.state, w.analytics)


def webhook(event_name: str, message_id: str) -> bytes:
    return json.dumps({"provider_message_id": message_id,
                       "event": event_name, "at": T_DAY + 60.0}).encode()


handler.handle(webhook("delivered", provider_id), "good")
check("T4b the handler published but did NOT write the state store",
      len(published) == 1
      and w.state.get(n_id).state is NotificationState.ACCEPTED)
check("T4c the consumer is what moves the row",
      consumer.consume(published[0])
      and w.state.get(n_id).state is NotificationState.DELIVERED)
check("T4d a duplicate delivery is a no-op, and it is COUNTED as a duplicate",
      consumer.consume(published[0]) is False
      and consumer.duplicates.value() == 1
      and w.state.get(n_id).state is NotificationState.DELIVERED)
earlier = handler.handle(webhook("sent", provider_id), "good")
check("T4e an out-of-order 'sent' after 'delivered' is refused, not applied",
      consumer.consume(earlier) is False
      and consumer.out_of_order.value() == 1
      and w.state.get(n_id).state is NotificationState.DELIVERED)
rejected = False
try:
    handler.handle(webhook("delivered", provider_id), "forged")
except SignatureError:
    rejected = True
check("T4f a bad signature is rejected before anything is parsed",
      rejected and handler.rejected.value() == 1
      and w.state.get(n_id).state is NotificationState.DELIVERED)
unknown_event = DeliveryEvent("pm_push_nope", NotificationState.DELIVERED, 0.0)
check("T4g a report for an unknown message is counted, not crashed on",
      consumer.consume(unknown_event) is False
      and consumer.unknown.value() == 1)

# --------------------------------------------------------------------------
# T5. Retries: absolute, jittered, and the budget ends in the DLQ
# --------------------------------------------------------------------------
w = World(push_script=["503"] * 6)
w.add_user("u1", zone="UTC", destinations={ChannelKind.PUSH: "tok_1"})
plan, _ = w.api.submit(
    w.request("r_retry", "order.shipped", "o_8", ("u1",), (ChannelKind.PUSH,),
              {"order_id": "o_8"}), "idem_retry")
n_id = plan.notifications[0].id
states = []
for _ in range(6):
    states.extend(r.state for r in w.dispatcher.run_once())
    w.clock.advance(4000.0)
check("T5a four retries, then the budget is spent and it is DEAD",
      w.state.get(n_id).state is NotificationState.DEAD)
check("T5b the attempt count is five, and it is on the row",
      w.state.get(n_id).attempt == 5)
check("T5c every non-terminal attempt stayed PENDING",
      states[:4] == [NotificationState.PENDING] * 4 and not states[5:])
check("T5d DEAD lands in the DLQ with the reason",
      len(w.dlq) == 1 and "503" in w.dlq.entries()[0].last_error)
check("T5e the DLQ carries the payload a replay tool would need",
      w.dlq.entries()[0].user_id == "u1"
      and w.dlq.entries()[0].attempts == 5)

# --------------------------------------------------------------------------
# T6. End to end, and the aggregate over a request is the MAXIMUM
# --------------------------------------------------------------------------
w = World(push_script=["OK"], email_script=["OK"])
w.add_user("u1", zone="UTC", destinations={ChannelKind.PUSH: "tok_1",
                                           ChannelKind.EMAIL: "u1@example.com"})
plan, _ = w.api.submit(
    w.request("r_e2e", "order.shipped", "o_10", ("u1",),
              (ChannelKind.PUSH, ChannelKind.EMAIL), {"order_id": "o_10"}),
    "idem_e2e")
check("T6a one notification per recipient per channel",
      len(plan) == 2 and len({n.channel for n in plan.notifications}) == 2)
results = w.dispatcher.run_once()
check("T6b both channels went out",
      len(results) == 2
      and all(r.state is NotificationState.ACCEPTED for r in results))
push_id = next(n.id for n in plan.notifications
               if n.channel is ChannelKind.PUSH)
w.state.transition(push_id, NotificationState.DELIVERED)
status = w.api.status("r_e2e")
check("T6c the sender-facing progress is the MAXIMUM across channels",
      status["progress"] == "delivered")
check("T6d the ops-facing counts are the histogram, not a single value",
      status["counts"] == {"delivered": 1, "accepted": 1})
check("T6e and the two answers differ on purpose — a minimum aggregate would "
      "report a delivered notification as undelivered",
      status["progress"] != "accepted")

# --------------------------------------------------------------------------
# T7. Two keys, two problems: idempotency (the producer) and dedupe (the world)
# --------------------------------------------------------------------------
w = World(push_script=["OK", "OK"])
w.add_user("u1", zone="UTC", destinations={ChannelKind.PUSH: "tok_1"})
req = w.request("r_first", "order.shipped", "o_11", ("u1",),
                (ChannelKind.PUSH,), {"order_id": "o_11"})
first, replay = w.api.submit(req, "idem_k1")
check("T7a a first submit is not a replay", len(first) == 1 and not replay)

same_event = w.request("r_second", "order.shipped", "o_11", ("u1",),
                       (ChannelKind.PUSH,), {"order_id": "o_11"})
second, _ = w.api.submit(same_event, "idem_k2")
check("T7b a DIFFERENT request id, same entity, is caught by the dedupe store",
      len(second) == 0 and "duplicate" in second.suppressed[0][1])

third, replayed = w.api.submit(req, "idem_k1")
check("T7c the SAME idempotency key is a replay, not a second decision",
      replayed and len(third) == 1)
check("T7d the replay returns the ORIGINAL rows, so the producer's poll works",
      third.notifications[0].id == first.notifications[0].id)

# --------------------------------------------------------------------------
# T8. The DEFER is a scheduled_at on a written row — the §6.11 fix
# --------------------------------------------------------------------------
w = World(start=T_NIGHT)
w.add_user("u1", zone="UTC", destinations={ChannelKind.PUSH: "tok_1"})
plan, _ = w.api.submit(
    w.request("r_defer", "order.shipped", "o_12", ("u1",), (ChannelKind.PUSH,),
              {"order_id": "o_12"}), "idem_defer")
check("T8a the deferred notification is a ROW immediately, before any queue",
      w.state.size() == 1
      and w.state.get(plan.notifications[0].id).state
      is NotificationState.PENDING)
check("T8b nothing is due while the window is open",
      w.dispatcher.run_once(w.clock.now()) == [])
w.clock.advance(35200.0)
sent = w.dispatcher.run_once()
check("T8c it goes out when the window closes, without any extra state",
      len(sent) == 1 and sent[0].state is NotificationState.ACCEPTED)
check("T8d quiet hours are evaluated ONCE, at decision time — the send at "
      "08:00 is not re-checked against a window it would now pass",
      w.tz.local_time("UTC", w.clock.now()).hour == 8)

# --------------------------------------------------------------------------
# T9. The SMS allow-list is enforced at the CHANNEL, so it cannot be bypassed
# --------------------------------------------------------------------------
w = World(sms_script=["OK"])
sms = w.registry.get(ChannelKind.SMS)
blocked = sms.send(RenderedMessage("n_s1", ChannelKind.SMS, "+15550001111",
                                   "t", "b", "marketing.campaign", 1, "en",
                                   "marketing.campaign"))
check("T9a a marketing SMS is refused at the channel",
      blocked.error_code == "CHANNEL_NOT_ALLOWED"
      and blocked.kind is FailureKind.TERMINAL)
check("T9b and the provider was never called",
      w.calls.get(ChannelKind.SMS, []) == [])
allowed = sms.send(RenderedMessage("n_s2", ChannelKind.SMS, "+15550001111",
                                   "t", "b", "auth.otp", 1, "en", "auth.otp"))
check("T9c an OTP SMS goes through", allowed.accepted)

# --------------------------------------------------------------------------
# T10. The provider's Retry-After outranks our own schedule
# --------------------------------------------------------------------------
w = World(push_script=[("429", "120"), "OK"])
w.add_user("u1", zone="UTC", destinations={ChannelKind.PUSH: "tok_1"})
plan, _ = w.api.submit(
    w.request("r_429", "security.alert", "s_2", ("u1",), (ChannelKind.PUSH,),
              {"city": "Lisbon"}), "idem_429")
res = w.dispatcher.run_once()[0]
delay = plan.notifications[0].scheduled_at - T_DAY
check("T10a our own HIGH schedule would have retried in 1 second",
      w.retry.SCHEDULES[Priority.HIGH][0] == 1.0)
check("T10b but the provider asked for 120, and 120 wins",
      res.state is NotificationState.PENDING and 90.0 < delay < 150.0)

# --------------------------------------------------------------------------
# T11. The limiter returns a WAIT TIME, and that is what lets OTP fail fast
# --------------------------------------------------------------------------
w = World(email_script=["OK", "OK"])
w.add_user("u1", zone="UTC", destinations={ChannelKind.EMAIL: "u1@example.com"})
w.limiters[ChannelKind.EMAIL] = ProviderLimiter(
    TokenBucket(capacity=1.0, refill_per_s=0.001, clock=w.clock))

a = w.decisions.decide(w.request("r_l1", "order.shipped", "o_20", ("u1",),
                                 (ChannelKind.EMAIL,), {"order_id": "o_20"}))
check("T11a the first send takes the only token",
      w.dispatcher.run_once()[0].state is NotificationState.ACCEPTED)

b = w.decisions.decide(w.request("r_l2", "auth.otp", "o_21", ("u1",),
                                 (ChannelKind.EMAIL,), {"code": "123456"}))
b_result = w.sender.send(b.notifications[0])
check("T11b the same saturation makes an OTP TERMINAL, because a code that "
      "arrives after its validity window is worthless",
      b_result.state is NotificationState.FAILED
      and "PROVIDER_SATURATED" in b_result.detail)

c = w.decisions.decide(w.request("r_l3", "order.shipped", "o_22", ("u1",),
                                 (ChannelKind.EMAIL,), {"order_id": "o_22"}))
c_result = w.sender.send(c.notifications[0])
check("T11c and makes an ordinary notification WAIT instead, with a time",
      c_result.state is NotificationState.PENDING
      and c.notifications[0].scheduled_at > w.clock.now() + 900.0)
check("T11d exactly one thing reached the DLQ, and it was the OTP",
      len(w.dlq) == 1 and w.dlq.entries()[0].notification_id
      == b.notifications[0].id)

# --------------------------------------------------------------------------
# T12. The reserved bucket keeps HIGH moving while the general one is empty
# --------------------------------------------------------------------------
w = World(push_script=["OK", "OK", "OK"])
w.add_user("u1", zone="UTC", destinations={ChannelKind.PUSH: "tok_1"})
w.limiters[ChannelKind.PUSH] = ProviderLimiter(
    general=TokenBucket(1.0, 0.001, w.clock),
    reserved=TokenBucket(10.0, 10.0, w.clock))

w.dispatcher.run_once()                      # takes the general bucket's token
warmup = w.decisions.decide(w.request("r_r0", "order.shipped", "o_29", ("u1",),
                                      (ChannelKind.PUSH,), {"order_id": "o_29"}))
check("T12a the first, ordinary send takes the general bucket's only token",
      w.sender.send(warmup.notifications[0]).state
      is NotificationState.ACCEPTED)
low = w.decisions.decide(w.request("r_r1", "order.shipped", "o_30", ("u1",),
                                   (ChannelKind.PUSH,), {"order_id": "o_30"}))
low_result = w.sender.send(low.notifications[0])
check("T12b the general bucket is empty, so bulk work waits",
      low_result.state is NotificationState.PENDING)
high = w.decisions.decide(w.request("r_r2", "security.alert", "s_3", ("u1",),
                                    (ChannelKind.PUSH,), {"city": "Lisbon"}))
high_result = w.sender.send(high.notifications[0])
check("T12c a security alert still goes out NOW, from the reserved bucket",
      high_result.state is NotificationState.ACCEPTED)
check("T12d and the reserved bucket was the one it took",
      w.limiters[ChannelKind.PUSH].reserved.tokens() < 10.0
      and w.limiters[ChannelKind.PUSH].general.tokens() < 1.0)

# --------------------------------------------------------------------------
# T13. The template VERSION recorded at decision time is the one that renders
# --------------------------------------------------------------------------
w = World(push_script=["OK"])
w.add_user("u1", zone="UTC", destinations={ChannelKind.PUSH: "tok_1"})
plan, _ = w.api.submit(
    w.request("r_t1", "order.shipped", "o_40", ("u1",), (ChannelKind.PUSH,),
              {"order_id": "o_40"}), "idem_t1")
check("T13a the row records the version it was decided against",
      plan.notifications[0].template_version == 1)
w.templates.put(Template("order.shipped", ChannelKind.PUSH, "en", 2,
                         "Order shipped", "Order {order_id} has SHIPPED!"))
w.dispatcher.run_once()
check("T13b a template published AFTER the decision does not rewrite history",
      w.rendered[0].template_version == 1
      and "on its way" in w.rendered[0].body)

plan2, _ = w.api.submit(
    w.request("r_t2", "order.shipped", "o_41", ("u1",), (ChannelKind.PUSH,),
              {"order_id": "o_41"}), "idem_t2")
w.dispatcher.run_once()
check("T13c a notification decided after the publish gets the new version",
      plan2.notifications[0].template_version == 2
      and "SHIPPED" in w.rendered[1].body)

missing = make_notification("n_no_tpl", template_version=99)
w.state.put(missing)
missing_result = w.sender.send(missing)
check("T13d a version missing from the store is TERMINAL, not a re-pick",
      missing_result.state is NotificationState.FAILED
      and "TEMPLATE_VERSION_MISSING" in missing_result.detail)

bad = make_notification("n_bad_payload", payload={})
w.state.put(bad)
bad_result = w.sender.send(bad)
check("T13e a missing placeholder fails identically forever, so it is terminal",
      bad_result.state is NotificationState.FAILED
      and "RENDER_FAILED" in bad_result.detail)

# --------------------------------------------------------------------------
# T14. DST: one local date, two absolute instants, one offset that changes
# --------------------------------------------------------------------------
w = World()
jan = datetime(2023, 1, 15, 12, 0, tzinfo=dt_timezone.utc).timestamp()
jul = datetime(2023, 7, 15, 12, 0, tzinfo=dt_timezone.utc).timestamp()
check("T14a the offset is resolved per instant, not baked in",
      w.tz.offset_hours("America/New_York", jan) == -5.0
      and w.tz.offset_hours("America/New_York", jul) == -4.0)

before = datetime(2023, 11, 5, 5, 30, tzinfo=dt_timezone.utc).timestamp()
after = datetime(2023, 11, 5, 6, 30, tzinfo=dt_timezone.utc).timestamp()
check("T14b the DST overlap really is two instants at one wall clock",
      after - before == 3600.0
      and w.tz.local_time("America/New_York", before).hour == 1
      and w.tz.local_time("America/New_York", after).hour == 1)
check("T14c so the LOCAL DATE key is what dedupes a daily send, not the clock",
      w.tz.local_date_key("America/New_York", before)
      == w.tz.local_date_key("America/New_York", after))

w2 = World(start=before)
w2.add_user("u_ny", zone="America/New_York",
            destinations={ChannelKind.PUSH: "tok_ny"})
naive = w2.tz.local_time("America/New_York", before).replace(hour=8).timestamp()
check("T14d the NAIVE 'replace the hour and convert' lands at 07:00 local — an "
      "hour early, because the offset that applied at 01:30 no longer applies "
      "at 08:00",
      w2.tz.local_time("America/New_York", naive).hour == 7)

d_before = w2.tz.next_local_time("America/New_York", before, 8)
d_after = w2.tz.next_local_time("America/New_York", after, 8)
eight_am = datetime(2023, 11, 5, 13, 0, tzinfo=dt_timezone.utc).timestamp()
check("T14e re-resolving the offset AT THE TARGET lands on 08:00 local",
      w2.tz.local_time("America/New_York", d_before).hour == 8
      and w2.tz.local_time("America/New_York", d_after).hour == 8)
check("T14f and both deferrals converge on ONE instant, which is the point: "
      "the same wall clock either side of the transition is the same moment",
      d_before == d_after == eight_am)

ny = w2.decisions.decide(w2.request("r_dst", "order.shipped", "o_50",
                                    ("u_ny",), (ChannelKind.PUSH,),
                                    {"order_id": "o_50"}))
check("T14g a 01:30 local notification is deferred to the same wall clock "
      "either side of the transition",
      len(ny) == 1
      and w2.tz.local_time("America/New_York",
                           ny.notifications[0].scheduled_at).hour == 8)

# --------------------------------------------------------------------------
print(f"\n{PASSED[0]} checks passed, {FAILED[0]} failed")
check("every check above passed", FAILED[0] == 0)
```

That is the whole system. 68 checks, all executed above, and the four that failed on the first run are worth knowing about because three of them were design bugs rather than typos — a duplicate webhook that reported success, a `defer_until` that landed an hour early across a DST transition, and a demo step that assumed a token had been taken when nothing had run. The DST one is the interesting one: the fix was not to adjust the arithmetic, it was to **move the method to the class that owns offsets**, because `QuietHours` had no way to know that 08:00 on the fall-back day is not 08:00 plus the old offset. A class computing a wall-clock instant without access to timezone rules is a class that will get it wrong once a year.

---

## 6. Concurrency and thread-safety

**What is shared.** Every store in §5 is shared across request threads, dispatcher threads and webhook consumer threads: `NotificationStateStore`, `IdempotencyStore`, `DedupeStore`, `TokenBucket`, `TokenStore`, `TemplateStore`, `PreferenceStore`, `ConsentStore`, `DeadLetterQueue`, `UserRateLimiter`. Every one of them owns a private `Lock` or `RLock`; there is no global lock anywhere, and that is deliberate — a single lock over the notification path would serialise a million-user campaign behind one OTP.

**The pattern that matters is claim-not-check.** `IdempotencyStore.claim` and `DedupeStore.claim` return **ownership**, not a predicate. A `is_duplicate()` followed by a `mark_duplicate()` has a window between the two calls in which a second thread sees "not a duplicate" and proceeds — and for OTP that window is the difference between one code and two, where the second invalidates the first and the user cannot log in. The same shape appears three times: `TokenBucket.acquire` takes the tokens inside the lock it measured them in; `UserRateLimiter.check_and_count` counts in the same critical section it checked in; `NotificationStateStore.transition` decides legality inside the lock it mutates under. In each case the check and the effect are one operation, and the API makes the unsafe version impossible to express.

**`NotificationStateStore` needs an `RLock`, and the reentrancy is not decoration.** `progress()` and `counts()` are built from `for_request()`, which acquires the same lock. With a plain `Lock`, both would deadlock on their second acquisition — and only under concurrency, which is the kind of bug that ships.

**Nothing is held across a network call.** Trace `Sender.send`: it acquires the limiter's lock inside `acquire()` and releases it before returning, reads the template under the template store's lock, releases, then calls the channel's provider function with **no lock held**. This is the rule that keeps the system's throughput a function of the provider rather than of lock contention — and it is why the limiter returns a *wait time* rather than blocking. A `TokenBucket` that slept while holding its lock would serialise every sender in the process behind one provider's refill rate.

**The state machine is the guard against concurrent writers, and it is the right guard.** A dispatcher marking `ACCEPTED` and a webhook consumer marking `DELIVERED` can race; the rank order plus forward-only transitions means the worst outcome is one of them being refused and retried by the provider, never a row that walks backwards. **But there is a real gap, and it should be named rather than found:** two dispatchers can both select the same `PENDING` row from `due()` and both send it. `DedupeStore` does not cover this, because the claim was already made at decision time. The fix is a lease — `due()` should take a visibility timeout and mark the row claimed, exactly as a queue does — and the honest statement is that this design has the dedupe half and not the lease half. I would expect to be asked this, and the answer is a two-column change plus the timeout, not a redesign.

**Ordering between notifications for one user is not provided and cannot be bolted on cheaply.** Nothing here guarantees that `order.shipped` arrives after `order.confirmed`. Rows are independent, workers are pooled, and the provider reorders. Getting it means partitioning by user id and consuming each partition in one thread — which is a real cost in throughput, and the reason the honest answer is "we do not promise ordering; we promise at-least-once and we dedupe", with a per-user sequence number on the payload if the product ever demands it.

**Isolation is per worker pool, not per message.** The HLD's fix for a marketing blast starving transactional notifications is separate pools per priority and per channel, and that boundary is *outside* these classes. `Dispatcher.run_once` orders by priority — HIGH, then NORMAL, then LOW — which is the right behaviour for a single worker and no protection at all for a saturated one. Both are needed: the ordering makes a single worker correct, and the pools make the system correct when a worker cannot keep up.

**Counters are `Counter`, not `+=`.** The webhook handler and its consumer are the two classes that run on many threads with no other shared state, and `self.rejected += 1` loses increments. Those particular numbers are what an incident review reads, so losing them quietly is the wrong failure mode. In production this is an atomic counter in the metrics library.

**The clock.** `SystemClock` is stateless and safe. `ManualClock` is not thread-safe and does not need to be: it is a single-threaded test double, and giving it a lock would imply the tests exercise concurrency, which they deliberately do not. Concurrency here is argued from the object graph, and the honest scope of the demo is that it is single-threaded.

---

## 7. Extensibility

| Change | What you touch | What you must not touch | Why it is cheap |
|---|---|---|---|
| **A new channel** (Slack, WhatsApp) | One `Channel` subclass; register it in the `ChannelRegistry`; add templates; add the channel to the preference defaults | The decision service, the retry policy, the state machine | The channels are already uniformly "return an `SendOutcome`, never retry", so a new one is a `send` method and a provider call |
| **A new provider for an existing channel** | The injected `provider` callable | Everything else | The provider is a dependency, not a subclass — proven by the demo scripting three different providers without a new class |
| **A new error code** | One row in `ClassificationTable.DEFAULT_RULES` | The channels | The classification is data, so a provider's new code is a config change, not three SDK edits |
| **A new event type** | A row in `EVENT_PRIORITY`; templates for each channel | The decision service, `QuietHours` | Priority is derived from the event type rather than supplied, so there is exactly one place to classify it |
| **A new quiet-hours bypass** | One entry in `QUIET_HOURS_BYPASS` | The decision service | The bypass is a frozen allow-list, so the change is a reviewable diff rather than a runtime claim |
| **A different quiet-hours policy** | A `QuietHoursPolicy` implementation | `QuietHours`, which is one | The interface takes `priority` and `event_type` separately, which is the minimum a real policy needs |
| **A new locale** | Templates at that locale | The renderer | `TemplateStore.pick` falls back exact-locale then `en`, so a half-translated locale degrades instead of failing |
| **A new priority** | A `Priority` member, a `SCHEDULES` entry, a `due()` rank | The stores | Three places, all of them tables, all of them findable by the type checker |
| **A per-channel rate limit** | A second `ProviderLimiter` | The sender | The limiters are keyed by channel and held by reference, so raising one during an incident needs no deploy |
| **A different delivery aggregate** | One method on the store | The callers | `progress` and `counts` are two answers to two questions; adding a third does not disturb either |

**What is deliberately not extensible.** `Sender.send` is the one method with a fixed spine — limiter, template, render, classify, retry-or-finish — and it should stay that way. Every attempt to make it pluggable ("a pre-send hook", "a custom failure handler") ends with a hook that swallows a `SUPPRESS_TOKEN` and a dead token that never gets deleted. The extension points are the injected collaborators *inside* the steps (`ClassificationTable`, `RetryPolicy`, `TemplateRenderer`, `Channel`, `ProviderLimiter`), not the steps themselves.

---

## 8. Failure modes and edge cases

Each row maps to the HLD's own failure section, so the two documents can be read side by side.

| HLD | Failure | The class that handles it, and how | What is still open |
|---|---|---|---|
| §6.1 | **A marketing blast starves transactional notifications** | `Priority` derived from the event type; `Dispatcher` orders HIGH first; `ProviderLimiter.reserved` keeps a HIGH bucket the bulk work cannot drain | Worker-pool separation lives outside these classes — the ordering is correct for one worker, the pools are what protect a saturated one |
| §6.2 | **Retry storm at the provider** | `RetryPolicy` jitters ±20%, and `next_attempt` returns an **absolute** time so two workers cannot disagree about when | Nothing structural; the jitter is a constant somebody can set to 0, so it needs a test that asserts non-equality of two seeded policies |
| §6.3 | **The same OTP twice** | `IdempotencyStore.claim` returns ownership atomically and records the request id **on the claim**, so a crash mid-request leaves the key held and a retry gets the original id | A producer that retries with a *different* key is not covered — that is the `DedupeStore`'s job, and the two are deliberately separate |
| §6.4 | **Invalid device tokens** | `FailureKind.SUPPRESS_TOKEN` is a third outcome; `Sender` deletes the binding via `TokenStore.delete`; `TokenStore.prune_stale` removes 90-day-old bindings | A stale token belonging to a *different user* (a resold device) needs the on-logout delete, which is a caller responsibility this design assumes |
| §6.5 | **Quiet hours versus a security alert** | `QUIET_HOURS_BYPASS` is an event-type allow-list, so a HIGH notification not on it is deferred rather than sent | Nothing; this is the fix. A new 3 a.m. event is a code review against a frozen set |
| §6.6 | **Timezone and DST bugs** | The IANA name is stored and the offset is resolved **at send time**; `next_local_time` re-resolves the offset at the *target*; `local_date_key` keys idempotency on the local date | A user who changes timezone between the decision and the send is not re-evaluated — see §11 P6 |
| §6.7 | **A slow webhook stalls tracking** | `WebhookHandler` holds no reference to the state store, so it cannot write even by accident; `DeliveryEventConsumer` is a separate class | The handoff itself is a `Callable` in this version; production needs a durable bus, or a crash between publish and consume loses the event |
| §6.8 | **Provider outage and failover** | `ClassificationTable` makes the failover decision uniform; `RetryPolicy` honours the provider's `Retry-After` over our own schedule | Choosing a *secondary* provider is not modelled — a failover policy would be a `Channel` decorator, and it is where "failover" quietly becomes double-sending |
| §6.9 | **The unbounded queue and the backlog cliff** | The priority-ordered `due()` keeps new HIGH work ahead of the backlog; the DLQ bounds the retry tail | **Not solved here.** There is no per-message TTL, so a `Priority.LOW` notification can still be delivered 13 hours late. The HLD's answer is a TTL plus drop-rather-than-deliver-late, and that is a column and a check in `due()` |
| §6.10 | **A preference change races an in-flight notification** | Preferences are evaluated as late as possible — inside `decide`, immediately before the row is written | The race is **acknowledged, not fixed**: there is no transaction between the preference store and the provider. The window is small and unsubscribes are honoured for everything after it |
| §6.11 | **Template change breaking in-flight sends** | The version is recorded on the row at decision time and `get_version` renders **that** version — `T13b` proves a later publish does not rewrite history | A template deleted outright is terminal, which is correct but means template versions must be append-only in the real store |
| §6.11 | **The `202` that never becomes anything** | `DecisionService` writes the state row **before** returning, so an accepted notification is always a row — `T8a` asserts it | The gap between the row and the queue is closed by the dispatcher reading the row, not by the queue; that is the whole point and it needs `due()` to be a real query |
| §6.11 | **Cost surprise from a retry loop** | The retry budget is per `Priority`, so SMS is expensive only on the HIGH schedule, and `SmsChannel.SMS_ALLOWED` keeps marketing off SMS entirely | Cost is not *observable* in these classes. Monitor cost per hour is an ops requirement the design enables and does not implement |
| §6.11 | **Clock skew across workers** | `scheduled_at` is absolute everywhere; no "now + 5 minutes" is computed twice | A grace window on `due()` is not modelled; production wants one so a worker with a clock a minute fast does not fire early |
| §6.11 | **Unbounded fan-out memory** | Not modelled — `NotificationRequest.user_ids` is a tuple handed in whole | The HLD's fix is paging the audience expansion at 100,000 recipients and staging 1% → 10% → 100%, and both live in the campaign service above this one |

**The two rows I would not defend without qualification** are §6.9 and the dispatcher lease in §6. Both are places where this design is correct and incomplete in the same breath, and both are cheap to finish — which is exactly why naming them is better than hoping nobody asks.

---

## 9. Design patterns used, and why

| Pattern | Where | Why it earns its place |
|---|---|---|
| **Strategy** | `Channel` (push/email/SMS), `QuietHoursPolicy`, `AnalyticsSink` | The three channels differ only in how they talk to a provider; the *classification* is shared, so the behaviour that varies (transport) is separated from the behaviour that must not (what counts as retryable) |
| **Template Method** | `_ProviderChannel` | `send` is the same three lines for all three channels — call, classify, return — and the subclasses fill in a kind and sometimes an allow-list. `SmsChannel` overrides `send` to add a gate and calls `super()`, which is the pattern's whole shape in four lines |
| **Ports & Adapters** | Injected `provider` callables, `Clock`, the offset `resolver`, `publish` | Every external system — FCM, SES, the clock, the timezone database, the event bus — enters as a callable. That is what makes the demo able to script a 429 followed by a hard bounce followed by a 410, which nobody reproduces against a real provider on purpose |
| **Value Object** | `NotificationRequest`, `RenderedMessage`, `SendOutcome`, `Template`, `DeadLetter`, `DeliveryEvent`, `ChannelDecision` | Frozen dataclasses, so they are hashable, comparable, safe to pass across threads, and impossible to mutate by accident. `SendOutcome` is the one that matters most: a mutable outcome is an outcome someone edits on the way back up |
| **Entity** | `Notification` | Deliberately the *only* mutable value here, identity by `id`, and the mutation is confined to the three fields the lifecycle actually changes — `state`, `attempt`, `scheduled_at` |
| **Policy / table** | `ClassificationTable` | The rules are data, so they are readable during an incident and testable without a provider |
| **State** | `NotificationStateStore` | The transition table is the machine, the rank order is its guard, and the store rather than the caller decides legality — which is what makes a concurrent webhook safe |
| **Repository** | `NotificationStateStore`, `DedupeStore`, `IdempotencyStore`, `TokenStore`, `TemplateStore` | Each hides its index and its locking. The caller asks for `due(now)` or `find_by_provider_id`, not for a dictionary |
| **Facade** | `NotificationApi` | Two methods, one entry point, and the idempotency claim lives inside it so no caller can forget it |
| **Null Object** | `PreferenceStore.load` for an unknown user, `ProviderLimiter` with no reserved bucket | A user with no row gets permissive defaults rather than a `None` every caller must check; a limiter with no reserved bucket means "only the general one", not an error |
| **Composite** | `DeliveryPlan` | One request aggregates many notifications, each of which aggregates one channel's attempts — and the aggregate *direction* (max, not min) is the design decision worth defending |

**Rejected, with the reason — because "what I did not use" is half of what a pattern question asks:**

- **Chain of Responsibility** for `Sender.send`. It looks like a chain — limiter, template, render, classify — and making it one is the mistake. A chain lets a link decide not to call the next, which is how a "pre-send hook" ends up silently swallowing a `SUPPRESS_TOKEN` outcome and the dead token is never deleted. The steps are a fixed spine with injectable collaborators inside them.
- **Observer** for delivery events. Tempting: the analytics sink is a subscriber in all but name. The reason it is a direct call is that a transition must be durable and the notification that follows it must not be able to fail because a subscriber threw. A broadcast where one subscriber can break the publisher is not the pattern, it is a liability — so this is an ordered side-effect with a non-throwing contract.
- **Decorator** for the reserved bucket and for provider failover. A `ReservedLimiterDecorator` would read well and would hide the one thing the design needs to be visible: that HIGH takes from a *different* bucket. Failover as a decorator is worse — it is exactly where "failover" quietly becomes double-sending, and the classification table would no longer describe what actually happened.
- **Builder** for `Notification`. Sixteen fields make a Builder tempting. The reason not to is that exactly one class constructs a notification, and it is the class that is supposed to enforce the policy. A Builder would let a caller assemble a row that skipped the policy engine entirely and still look correct.
- **Singleton** for the stores. Every store here is injected, and the demo constructs several worlds with different limits in the same process. A singleton would make the per-scenario isolation impossible and would put production state behind a global that tests mutate. Test isolation is the whole argument, and it is the same argument in every file in this set.

---

## 10. Trade-offs

| Decision | The alternative | What it costs | Why this way |
|---|---|---|---|
| **`ClassificationTable` as data, unknown → RETRYABLE** | An `if` per channel; unknown → TERMINAL | An uncatalogued *terminal* error costs a bounded number of wasted attempts and then lands in the DLQ, noisily | An uncatalogued *transient* error classified TERMINAL silently drops a notification nothing will ever notice. Prefer the failure that is loud |
| **`scheduled_at` absolute, jittered** | `now + delay` computed per attempt | Every worker needs a roughly synced clock, and a worker with a fast clock can fire a minute early | §6.11's clock skew: two workers computing "5 minutes from now" at different instants disagree about when, permanently |
| **Quiet hours decided ONCE, at decision time** | Re-evaluate at send time | A user who changes timezone, or disables quiet hours, between the decision and the send is not re-consulted — `T8d` asserts exactly this | Re-evaluating makes a deferral potentially infinite (defer, requeue, still quiet, defer) and turns a `scheduled_at` column into a loop. The next notification is evaluated with the new preference |
| **Bypass allow-listed by EVENT TYPE** | "HIGH priority bypasses quiet hours" | A genuinely urgent new event is deferred until somebody adds it to a frozen set — a code review in the middle of an incident | Under the loose version every producer marks their own message HIGH, quiet hours cease to exist, and there is no artifact to review |
| **The request aggregate is the MAXIMUM** | The minimum, as the WhatsApp receipt aggregate does | A delivered notification on one channel and a terminally failed one on another reports as delivered, and the failure is only visible in the histogram | A request fanned out to push and email is satisfied by *either*. The minimum would report a delivered notification as undelivered — the worse of the two errors. Both are reported (`progress` and `counts`) precisely because one number cannot carry both |
| **The bucket returns a wait time** | A boolean, or a blocking acquire | Every caller has to handle a wait, and three of them handle it by scheduling rather than by blocking | It is the only shape in which OTP's exception is expressible *by the caller*: `wait > otp_window → fail fast`. A boolean bucket forces the bucket to know about OTP, and a blocking bucket serialises every sender in the process behind one refill rate |
| **A reserved bucket only HIGH may touch** | One shared provider limit | Bulk throughput is below the provider's limit even when nothing urgent is happening — capacity is held idle on purpose | §6.1 one layer down: the queues are separated, but the provider's limit is a shared resource. Without this, a campaign leaves a password reset queued behind promotions |
| **Dedupe scoped per (key, user, channel)** | Per (key, user) | The same event legitimately reaches one user on two channels, so "one notification" is not a guarantee | The per-user scope suppresses the email because the push won the race. Getting one notification *per channel* is the correct reading of "the user should hear about this once" |
| **The state row is written BEFORE the queue** | Enqueue, then write | A write on the hot path: ~500 B at 32,000/sec peak | §6.11's `202` that never becomes anything. A row written first means the dispatcher finds the notification even if the process dies between the decision and the enqueue |
| **Render at SEND time, store the version** | Render at decision time and store the body | A render can fail later, which is why `RENDER_FAILED` is a terminal outcome rather than an impossibility | Rendering early stores rendered bodies — which means PII in the notification table and stale text if a template is fixed. Storing the *version* gets traceability without either |
| **The webhook handler cannot write** | The handler does the state transition inline | An extra class and a handoff, and a crash between publish and consume loses the event | §6.7: under 100 ms, or providers disable the endpoint and then you get the event five times and no events at all. The structural inability to write is stronger than a rule about not writing |
| **`due()` is a scan over pending rows** | A time-ordered index per priority | O(pending) per dispatcher pass, which is unacceptable at 46 M queued messages | The honest first version. The ordering it implements (priority, then time) is the part that matters and the part that would be wrong if it were left to insertion order; the index is a storage change behind the same method |
| **No ordering guarantee between notifications** | Partition by user id and consume serially | The product cannot promise that `shipped` arrives after `confirmed` | Serialising per user costs real throughput on a system sized for 32,000/sec. The promise made instead is at-least-once plus dedupe, stated honestly |
| **Preferences read once, in `decide`** | A distributed transaction with the preference store | §6.10's race: a user disabling push mid-flight still receives that one | There is no transaction available between our store and a third-party provider. The honest design keeps the window small and makes the unsubscribe effective for everything after it |

---

## 11. Q&A

### Q1. Why is a notification a state row rather than a message?

Because the hard part is not delivering it, it is **knowing what happened to it**. A message is fire-and-forget; a notification has a lifecycle — decided, queued, accepted, delivered, opened, clicked — and every operational question anybody asks ("did the OTP arrive?", "which channel is failing?", "why did 40,000 die?") is a query against that lifecycle. The moment the row exists, those questions are answerable and the retry, dedupe and reporting logic all have somewhere to live.

### Q2. Where does the priority come from?

**From the event type, through a table, in the system.** The producer supplies an event type and an entity id; it does not supply a priority. If the producer could mark its own message HIGH, everything would be HIGH within a quarter and the priority field would mean nothing — and worse, the quiet-hours bypass would become a runtime claim rather than an auditable list.

### Q3. How do you stop the same OTP going out twice?

Two mechanisms for two different failures. The producer's own retry (the same POST twice) is caught by `IdempotencyStore.claim`, which is atomic and records the request id **on the claim**, so a crash mid-request leaves the key held and the retry gets the original id back rather than starting a second send. A second service emitting `auth.otp` for the same login attempt is caught by `DedupeStore`, scoped per (event, entity, user, channel) with a per-event-type TTL.

### Q4. A provider starts returning an error code you have never seen. What happens?

It is classified `RETRYABLE` — the default for an unknown code — so it is retried on the schedule and, if it persists, lands in the DLQ with the code in `last_error`. That is the cheap direction: bounded wasted attempts and a loud DLQ entry, versus the alternative of guessing TERMINAL and silently dropping notifications that would have succeeded. The alarm on DLQ entries for HIGH priority is what makes the new code visible on the first occurrence rather than the thousandth.

### Q5. Why three failure kinds and not two?

Because `SUPPRESS_TOKEN` is not a failure of the *send*, it is a fact about the *destination*. Folding it into TERMINAL loses the one action it implies — delete the binding — and folding it into RETRYABLE is worse: a hard bounce retried five times burns quota on a provider limit shared with password resets, and every one of those attempts can never succeed. Nearly every notification incident traces back to conflating those three.

### Q6. A security alert at 3 a.m. — does quiet hours suppress it?

No, and the mechanism matters. `security.alert` is on `QUIET_HOURS_BYPASS`, a frozen set of **event types**. The rule is deliberately stricter than "HIGH priority bypasses quiet hours": a HIGH notification that is *not* on the list is **deferred**, not sent. That means adding a new 3 a.m. wake-up is a code review against a frozen set — an artifact somebody can audit — rather than a producer's assertion about its own importance.

### Q7. Your token bucket returns a float. Why not a boolean?

Because a boolean makes "reject" the only expressible outcome, and this system's rule is delay rather than reject — a notification's value does not expire quickly. Returning a *wait time* also lets the one exception be expressed **by the caller**: `if wait > OTP_WINDOW_S: fail fast`, because an OTP that arrives after its validity window is worthless. With a boolean, the bucket would have to know about OTP, and with a blocking acquire the sender would hold a lock across a sleep.

### Q8. Why does the webhook handler never write to the database?

Because §6.7 says providers stop calling slow endpoints — under 100 ms, or they retry the event five times and then disable the endpoint. The handler validates the signature, translates the payload, publishes it and returns. Crucially the state store is not merely "not written" by the handler: the handler **holds no reference to it**, so the property is enforced by the object graph rather than by a comment. `DeliveryEventConsumer` does the durable work on the other side.

### Q9. The same delivery event arrives twice. What happens?

It is a no-op, and it is *counted* as a duplicate. `find_by_provider_id` resolves both reports to one notification, and the consumer sees the row is already at that state and returns False without transitioning. Because the provider's contract is at-least-once, duplicates are the normal case rather than the exceptional one, so the duplicate *rate* is a metric worth alarming on — which is why the consumer returns False for "changed nothing" rather than True for "handled".

### Q10. What is the difference between the idempotency store and the dedupe store?

The idempotency store answers "have I already handled *this request*?" — keyed by the caller's key, with a 24-hour TTL covering a long producer retry, and it returns the **original request id** so a retrying caller can poll for the result of the request it actually made. The dedupe store answers "have I already sent *this fact* to this person on this channel?" — keyed by event and entity, with a per-event-type TTL. One is a transport concern; the other is a product concern. Merging them produces a system that either re-sends on every producer retry or suppresses a legitimate second order's notification.

### Q11. How do you test any of this?

By making time and the provider into inputs. `ManualClock` moves time so backoff, quiet hours, TTLs and bucket refill are all deterministic; the provider is an injected callable so the demo scripts a sequence of error codes including the ones nobody reproduces on purpose. The 68 checks in §5 run in milliseconds and cover the classification table, the four-way quiet-hours decision, the forward-only state machine, the webhook fast path, the retry budget ending in the DLQ, the deferred row, the SMS allow-list, `Retry-After`, the OTP fail-fast, the reserved bucket, template version pinning and DST. If a behaviour cannot be expressed against a manual clock and a scripted provider, that is usually a sign it is not a behaviour but a hope.

### Q12. Why is the aggregate over a multi-channel request the maximum, when the receipt aggregate in the chat design is the minimum?

Because they answer different questions. A notification request fanned out to push and email is satisfied when the user got it by **any** of them — so the aggregate is the maximum, and a minimum would report a delivered notification as undelivered. A group read receipt is satisfied only when **every** recipient has read — so its aggregate is the minimum, and a maximum would report a group message as read when one person read it. Same machinery, opposite direction, and both are silent when wrong, which is why the store exposes `counts` alongside `progress`: one number cannot carry both questions.

### Q13. What would you change first with another week?

The dispatcher lease — `due()` taking a visibility timeout and marking rows claimed, so two dispatchers cannot both send the same notification. Then a per-message TTL so §6.9's backlog cliff drops stale LOW notifications rather than delivering them thirteen hours late. Both are small, both are named in §8, and both are the difference between "correct in the common case" and "correct".

### Q14. ⚡ `Notification` has sixteen fields and one class writes them. Isn't `DecisionService` a god object?

It is a policy engine, and the distinction is that every field it sets is set **by a policy check that has a name and a reason** — consent, preference, destination, quiet hours, dedupe, rate limit, template. The alternative is not smaller, it is distributed: the same checks spread across a `PreferenceChecker`, a `QuietHoursChecker`, a `RateLimitChecker` and a `TemplateResolver`, which reads better in a diagram and is worse in practice, because the **order** of those checks is the design. A dropped notification must not consume rate-limit budget, and a duplicate must not consume it either — that ordering is a property of one function, and splitting it across four collaborators makes it a property of nothing. What I would extract is not a check but a seam: the `ChannelDecision` value already records the reason for each recipient-channel pair, and that is where a second consumer (an audit log, a "why didn't I get this?" tool) attaches without touching the engine.

### Pushback round

### P1. "Your dispatcher can send the same notification twice — two workers, one PENDING row."

**Correct, and it is a real gap.** `due()` selects PENDING rows and two dispatchers can both select the same one. The dedupe store does not cover it, because the claim was made at decision time, before the row existed. The fix is a lease: `due()` takes a visibility timeout and marks rows claimed in the same critical section that selects them, with a reclaim path for rows whose worker died. The honest version of §6 says this design has the dedupe half and not the lease half. I would not claim otherwise, and I would bring it up before being asked.

### P2. "Quiet hours are evaluated once, so a notification deferred to 08:00 can arrive at 07:00 if the user changes timezone overnight."

**That is the accepted cost of evaluating the policy once, and the alternative is worse.** Re-evaluating at send time means a deferral can be deferred again — defer, requeue, still quiet, defer — which turns a `scheduled_at` column into a loop with no fixed point, and the notification can be deferred indefinitely by a user who keeps their window open. The window I am defending is one decision-to-send gap, and the next notification is evaluated against the new preference. `T8d` asserts the once-only behaviour rather than leaving it implied.

### P3. "Your deferral arithmetic looks naive — replace the hour and convert."

**It was, and the test caught it.** The first version took the local time, replaced the hour and converted back, which lands at **07:00** local on the fall-back day — an hour *inside* the window it was deferred out of, and an hour in the direction that wakes somebody up. The root cause was structural, not arithmetic: `QuietHours` was computing a wall-clock instant without any access to timezone rules. The fix was to move the method to `Timezone` as `next_local_time`, which resolves the offset **at the target** rather than at now, and one correction step converges because a transition moves the clock by at most an hour and the target is hours away. `T14d` asserts the naive version is wrong and `T14e` asserts the fix is right, so the bug is documented rather than erased.

### P4. "`due()` is a full scan of every pending row. The HLD has 46 million queued messages."

**Yes, and it is the honest first version rather than an oversight.** The scan is O(pending) per pass, which is unacceptable at the number the HLD computes. What I would defend is the *ordering* it implements — priority first, then time — because that part is the design and the part that would be wrong if it were left to insertion order: a campaign enqueued at 09:00 must not sit ahead of a password reset enqueued at 09:01, and `scheduled_at` alone cannot express that. The index is a storage change behind the same method signature, which is why the signature is the part worth getting right in an interview.

### P5. "Your demo is single-threaded, so every thread-safety claim is an argument, not a test."

**That is a fair characterisation, and I would rather say it plainly than imply coverage I do not have.** What the code does provide is structural: the check-and-effect pairs are single operations (`claim`, `acquire`, `check_and_count`, `transition`), no lock is held across a network call, and the store's `RLock` reentrancy is required rather than decorative. What would test it is a stress harness — N threads racing `claim` on one key and asserting exactly one winner, and a webhook consumer racing a dispatcher on one notification and asserting the state never walks backwards. Both are writable against the injected clock, and neither is in this file.

### P6. "A user in Tokyo changes timezone; the notification deferred to 08:00 arrives at the wrong hour."

**Acknowledged, and it is the same window as P2 seen from the other side.** The timezone is read when the decision is made, and the local instant for the deferral is computed then. A timezone change afterwards does not re-open the decision. There is no cheap fix that does not reintroduce the infinite-deferral problem: the row would have to be re-evaluated, which means re-running the policy, which means the decision is no longer a decision but a subscription. What I would do instead is make the *next* notification correct, and treat a timezone change as a preference write that also invalidates pending LOW and NORMAL rows — a bounded, explicit sweep rather than a loop.

### P7. "Then why not just make the aggregate the minimum, so a channel failure is never hidden?"

**Because it inverts the error.** With a minimum, a notification delivered on push and still pending on email reports as `pending`, so the sender-facing status tells the user "not delivered" for something they are holding in their hand. The failure I would rather have is the one the histogram catches: `progress` says `delivered` and `counts` says `{"delivered": 1, "accepted": 1}`, which is precisely the situation where email is not confirming, and the ops view sees it on the first request rather than on the aggregate. Two questions, two fields — and swapping the aggregate is how one of them silently becomes a lie.

---

## 12. Cheat sheet

**A notification is a decision with a state machine attached.** Not a message. The message is one line of the design; the decision and the lifecycle are the other ninety.

**The unit.** One `Notification` = one recipient × one channel × one state row. Written at decision time, before anything is enqueued, so an accepted notification is always accounted for.

**The decision order is the design.** Consent → preference → destination → priority (derived) → quiet hours → dedupe → rate limit → template. Drop decisions come before anything that consumes a resource, so a dropped notification takes no rate-limit budget and no idempotency key.

**Failure is data, not code.** One table from provider error code to `RETRYABLE | TERMINAL | SUPPRESS_TOKEN`, defaulting to RETRYABLE because an unknown code is the failure that should be loud. `SUPPRESS_TOKEN` is a third outcome because it is a fact about the destination, and it implies an action: delete the binding.

**Quiet hours: priority decides DEFER vs DROP, the EVENT TYPE decides whether the window applies.** An event-type allow-list is auditable; "high priority bypasses" is a runtime claim, and everything becomes high priority.

**The bucket returns a wait time, not a boolean.** It is the only shape in which OTP's exception is expressible by the caller. A reserved bucket only HIGH may touch is what stops a campaign starving a password reset at the provider.

**Retries are absolute and jittered.** `scheduled_at` is a timestamp, never "now + 5 minutes" computed twice — §6.11's clock skew. ±20% is the cheapest fix in the design: without it, a provider outage recovers into a synchronised second storm.

**Two keys, two problems.** The idempotency store is the producer's retry (24 h TTL, returns the original request id). The dedupe store is the world's duplicate (per event-type TTL, scoped per channel). Merging them either re-sends on every retry or suppresses a legitimate second order.

**The webhook handler validates, translates, publishes and returns — and holds no reference to the store.** Under 100 ms, or providers stop calling. The consumer does the durable work, and it counts duplicates and out-of-order arrivals separately because they have different owners.

**The lifecycle is forward-only and ranked**, which is what makes an out-of-order webhook a no-op instead of a regression. The request aggregate is the **maximum** progress (the user got it somewhere); the ops view is the histogram. The chat design's receipt aggregate is the **minimum**, and the two are easy to swap by accident.

**Templates are versioned and the version is on the row**, and the send path renders *that* version — otherwise a notification that sat in a retry queue for six hours renders with whatever was published since.

**Timezones are IANA names, resolved at send time, with the offset re-resolved at the target.** The naive "replace the hour and convert" is an hour wrong on the one day a year it matters, and wrong in the direction that wakes somebody up.

**Know your two honest gaps.** The dispatcher has no lease, so two workers can send one row twice; and `due()` is a scan, so 46 M queued messages needs an index. Both are cheap, both are in §8, and naming them is stronger than being caught by them.

---

## Cross-links

**Back up, to the HLD this is a level down from:**

- [Notification system — high-level design](../HLD/03-notification-system.md) — the five-step design, the provider deep dives (FCM/APNs, email deliverability, SMS cost and compliance), the dedupe and retry sections, and §6.1–6.11, which §8 above maps row by row.
- [Cheat sheet: operationalisation](../../Cheat-Sheets/06-operationalisation-cheatsheet.md) — DLQ alerting, cost-per-hour monitoring and the on-call view of the numbers this design produces.

**Sideways, to the other bridges in this set:**

- [URL shortener LLD](01-url-shortener-lld.md) — the same shape applied to a system whose hard part is the read path rather than the lifecycle.
- [WhatsApp LLD](02-whatsapp-lld.md) — the source of the **minimum** aggregate for read receipts, which §Q12 and `NotificationStateStore.progress` deliberately invert.
- [Uber matching LLD](03-uber-matching-lld.md) — the largest of these bridges, and the one that deals most directly with concurrent claim-and-lease, which is exactly the gap in §P1.
- [Instagram feed LLD](04-instagram-feed-lld.md) — the other bridge that had to get a cursor and a boundary exactly right; the canonical-walk argument there is the same kind of reasoning as the retry-budget argument here.

**Down, to the LLD problems for technique:**

- [LRU/LFU cache](../../LLD/problems/02-lru-lfu-cache.md) — eviction with a TTL, which is what the idempotency and dedupe stores are.
- [BookMyShow seat locking](../../LLD/problems/04-bookmyshow-seat-locking.md) — the atomic claim, and why a check-then-act on a shared key is the bug.
- [Rate limiter](../../LLD/problems/11-rate-limiter.md) — token buckets and the refill arithmetic, from the other direction: there the limiter is the product, here it is one step in a pipeline.
- [Logger framework](../../LLD/problems/12-logger-framework.md) — levels, handlers and the sink contract, which is the same shape as `AnalyticsSink`.
- [Pub-sub system](../../LLD/problems/13-pub-sub-system.md) — the handoff the webhook handler makes, and why delivery here is at-least-once with dedupe rather than exactly-once.
- [ATM machine](../../LLD/problems/07-atm-machine.md) — the closest sibling to the state machine in §5: a forward-only lifecycle where the illegal transition is the interesting case.

---

## Grounding note

**What is grounded, and where.** The design constraints implemented here come from [`Case-Studies/HLD/03-notification-system.md`](../HLD/03-notification-system.md): the five-step design and its capacity numbers, the provider deep dives, the deduplication section that distinguishes the two keys, the retry/backoff/DLQ section that fixes five attempts and ±20% jitter, the priority-queue isolation argument, the quiet-hours section, and the eleven failure modes in §6 that §8 above maps row by row. The specific constants — five attempts, the per-priority backoff schedules, 22:00–08:00 quiet hours, 5/day marketing cap, 24-hour idempotency TTL, 90-day token hygiene, the < 100 ms webhook budget, the 1% → 10% → 100% campaign staging — are that document's §8 table, not inventions here.

**What is NOT grounded, and I will not imply otherwise.** The reference repositories this knowledge base was assembled from contain **no OOP or LLD material at all** — no class diagrams, no pattern catalogues, no state-machine designs. The source transcripts name "notification system" as an HLD topic and never take it below the component level. Every class in §3, §4 and §5, the pattern choices in §9, the trade-offs in §10 and the Q&A in §11 are **constructed for this bridge**, reasoned from the HLD's constraints. Where a claim is checkable, it is checked: the 68 assertions in §5 execute and pass. Where a claim is a judgement — which pattern, which aggregate direction, which gap to name first — it is presented as a judgement, with the alternative and its cost stated beside it.

**What I would verify before trusting any of this in production.** The `zoneinfo` behaviour that `dst_resolver` simulates with the US rule; the real `Retry-After` semantics of each provider; whether your provider's webhook is genuinely at-least-once or actually at-most-once (which changes what the consumer must do); and the actual provider limits, which are the numbers that determine whether the reserved bucket is 10% or 40% of capacity.






