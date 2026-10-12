# LLD Case Study 02 — WhatsApp, One Level Down

> **Grounding note.** The reference repositories contain **no OOP/LLD material**, and the source transcripts do not cover a chat system at the class level. This file is written from first principles and is not transcript-grounded. The answer shape follows the house template at [LLD/01 What is LLD and How to Answer](../../LLD/01-what-is-lld-and-how-to-answer.md). The **system** view — the WebSocket fleet, fan-out at scale, the storage choice for messages, E2E encryption's key management, media handling — is owned by [WhatsApp Chat System (HLD)](../HLD/02-whatsapp-chat-system.md) and is **not repeated here**. This file opens its boxes: what a `SequenceGenerator` is as a class, why per-conversation ordering cannot come from timestamps, what an outbox row actually holds, and why a delivery receipt is a per-recipient matrix rather than a flag.

> **TL;DR** — The HLD said "WebSocket, a message queue per recipient, ordering per conversation, at-least-once delivery with a dedupe id". One level down that is six collaborators and one rule. The rule: **ordering is a sequence number assigned by the sender's server under a per-conversation lock, never a timestamp** — clocks across devices disagree by seconds, and a chat that reorders because a phone's clock drifted is a chat that is visibly broken. The collaborators: `Message` with a client-generated `client_msg_id` that makes a retry idempotent, a `SequenceGenerator` that hands out contiguous per-conversation numbers (a range lease when distributed), a `Conversation` that owns the lock and the membership, an `Outbox` that is a **message × recipient matrix** rather than a flag, a `DeliveryReceipt` state machine that aggregates per recipient, and a `PresenceService` whose online flag is a **lease with a TTL** so a crashed client goes offline on its own. The two genuinely hard parts are the ordering rule and the receipt aggregation — and both are hard for the same reason: they are facts about a *set* of deliveries, not about one message.

> **60-second interview answer:** "The send path is: dedupe on the client's message id, take the conversation lock, allocate the next sequence number, persist the message with that sequence, write one outbox row per recipient device, release the lock, then ack the sender with the server id and the sequence. Ordering is enforced by that sequence number and nothing else — timestamps are for display and are never the sort key. The client message id is what makes a retry safe: the sender's app generates it before the first attempt, so a resend after a lost ack returns the same message rather than a duplicate. Receipts are a matrix, not a flag: one row per (message, recipient), because 'delivered' for a group of five means five separate events and the aggregate has a state for each prefix. And presence is a lease — a TTL refreshed by heartbeats — because a client that crashes never sends a disconnect, so an online flag with no expiry is wrong the moment a phone loses signal."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **1:1 only, or groups?** | Whether the outbox fan-out is 1 row or N | Both; the group case is the interesting one |
| 2 | **Multi-device?** | Whether the outbox is per *user* or per *device* | Per device — one row per recipient device |
| 3 | **How is ordering defined?** | The single most important decision here | A per-conversation monotonic sequence number |
| 4 | **Are timestamps trusted?** | Whether ordering could use them | No — a client's clock is arbitrary |
| 5 | **Retry semantics?** | Whether there is a client message id | At-least-once with a client id for dedupe |
| 6 | **Receipts: sent, delivered, read?** | The receipt state machine | All three, and per recipient |
| 7 | **Do receipts go to everyone?** | Whether reads are aggregated | Delivered: per recipient. Read: aggregate, so one person reading does not tell the group |
| 8 | **Presence?** | Whether there is a lease and a TTL | Online is a lease with a TTL refreshed by heartbeat |
| 9 | **Offline recipients?** | Whether the outbox is a mailbox | Yes — the outbox *is* the offline mailbox |
| 10 | **Editing and deletion?** | Whether a message has a version | Out of scope, but the sequence design must not block it |

### The defaults to assume, stated plainly

> "I'll assume 1:1 and group conversations, multi-device fan-out, ordering by a **per-conversation sequence number assigned under a lock**, a **client-generated message id** for idempotent retries, an outbox that is one row per recipient device and doubles as the offline mailbox, a **per-recipient receipt matrix** with an aggregated read state, and **presence as a lease**. Does that match, or do you want the send path first and the receipts as a follow-up?"

### Non-goals

- No WebSocket framing, no protobuf, no E2E encryption — the [HLD owns those](../HLD/02-whatsapp-chat-system.md) §3.
- No media upload path, no push notification delivery.
- No persistence layer; the store is in-memory behind an interface.

### The four constraints that shape the design

1. **Ordering must not come from a clock.** Two phones disagree by seconds; a sender's server can disagree with a receiver's by more. The sort key has to be assigned once, by one authority.
2. **A retry must not create a second message.** Acks are lost, so clients resend; the dedupe key has to be generated by the *client*, because the server has not seen the message yet.
3. **"Delivered" is a fact about one recipient device, not about a message.** A group of five has five of them, and they arrive at different times or never.
4. **A crashed client never says goodbye.** An online flag with no expiry is wrong the moment a phone loses signal in a tunnel.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `ClientMessageId` | Value object | The sender's dedupe key: `(sender_id, client_id)` | Generating sequence numbers |
| `Message` | Frozen value object | `id`, `conversation_id`, `sender_id`, `seq`, `body`, `sent_at`, `client_msg_id` | Delivery state |
| `SequenceGenerator` | Service | `next(conversation_id) -> int`; **contiguous**, monotonic per conversation | Ordering the *storage* |
| `Conversation` | Aggregate | Participants, `last_seq`, the lock that makes seq allocation atomic | Receipts, presence |
| `ConversationStore` | Repository | Conversations by id, membership lookup | — |
| `OutboxRow` | Mutable entity | `(message_id, recipient_device_id, state, attempts, next_attempt_at)` | The message's content |
| `Outbox` | Repository | The matrix: one row per recipient device; also the **offline mailbox** | Deciding retry policy |
| `ReceiptState` | Enum | `PENDING → SENT → DELIVERED → READ` | — |
| `DeliveryReceipt` | Service/aggregate | Per-recipient transitions and the **aggregated** message state | Delivery itself |
| `PresenceService` | Service | `online` as a **lease**: heartbeat renews, TTL expires | Last-seen storage (a companion field) |
| `SessionRegistry` | Service | `user_id → connections` (multi-device) | Presence policy |
| `MessageStore` | Repository | Messages by `(conversation_id, seq)` and by `(sender, client_msg_id)` | Fan-out |
| `SyncService` | Service | `since(conversation_id, seq)` for a reconnecting client, and gap fill | — |
| `ChatService` | Façade | `send`, `deliver`, `mark_read`, `sync`, `history` | — |
| `DuplicateMessage` / `NotAParticipant` | Exceptions | Retry hit an existing client id; sender is not a member | — |

**The split to say out loud:** *"`Conversation` owns exactly one thing beyond membership: the lock and the counter that make sequence allocation atomic. `Message` is immutable and knows nothing about who received it. Receipts live in their own aggregate because they change after the message is frozen — a message that could mutate its own delivery state would have to be mutable, and then it is not safely cacheable or loggable."*

### HLD component → its classes

Every row is a box the [HLD](../HLD/02-whatsapp-chat-system.md) named; the right column is what that box is made of one level down.

| HLD component | Its classes here | The class-level question the HLD does not answer |
|---|---|---|
| **Message model** (HLD §3) | `Message`, `ClientMessageId`, `MessageStore` | Why the message is **immutable** and carries a client id, and why `seq` is not `sent_at` |
| **Per-conversation ordering** (HLD §3, the signature problem) | `SequenceGenerator`, `Conversation.last_seq`, the conversation lock | **Where** ordering is enforced: the seq is allocated under a lock at the sender's server; the receiver reorders by seq and detects gaps |
| **At-least-once + dedupe** (HLD §3) | `ClientMessageId`, the `(sender, client_msg_id)` index, `DuplicateMessage` | The client generates the key, not the server — because the server has not seen the message when the retry is sent |
| **Delivery to offline users / the inbox** (HLD §3) | `OutboxRow`, `Outbox`, the drain path | The outbox is a **matrix** (message × recipient device) and **is** the offline mailbox |
| **Delivery receipts** (HLD §3) | `DeliveryReceipt`, `ReceiptState`, the receipt matrix | Receipts are per recipient and *aggregate*; a read receipt to a group must not say who read |
| **Presence** (HLD §3) | `PresenceService`, the lease + TTL, `SessionRegistry` | Online is a **lease that expires**, because a crashed client never disconnects |
| **Multi-device** (HLD §3) | `SessionRegistry`, per-device `OutboxRow`s | One row per device, so a phone and a laptop advance independently |
| **Reconnection / gap fill** | `SyncService.since`, the `seq`-contiguous invariant | A gap in the receiver's sequence means "fetch the missing range", which only works because seq is contiguous |
| **Scaling out** | The range-lease `SequenceGenerator` (§6) | The per-conversation lock becomes a lease per block of sequence numbers |

---

## 3. Class diagram (ASCII)

```
  ┌──────────────────────────┐        ┌────────────────────────────────┐
  │  ClientMessageId «value» │        │  Message  «value, FROZEN»      │
  │  sender_id, client_id    │───────►│  id, conversation_id           │
  │  generated by the CLIENT │ carries│  sender_id, seq   ← NOT sent_at│
  └──────────────────────────┘        │  body, sent_at, client_msg_id  │
                                      └────────────────────────────────┘
                                                    ▲ stored in
  ┌─────────────────────────────────────────────────┴──────────────────┐
  │                          MessageStore                              │
  │  by_key:    {(conv, seq) -> Message}      ← the ordering index      │
  │  by_client: {(sender, client_id) -> Message}  ← the DEDUPE index    │
  │  +append(msg)        raises DuplicateMessage                        │
  │  +range(conv, from_seq, to_seq)                                     │
  └────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │  Conversation  (aggregate)                                         │
  │  id · participants: {user} · devices: {user -> {device}}           │
  │  last_seq: int                                                     │
  │  _lock: Lock        ← allocate_seq and append happen UNDER IT       │
  │  +allocate_seq() -> int     contiguous, monotonic, per conversation │
  │  +has(user) -> bool                                                │
  └────────────────────────────┬───────────────────────────────────────┘
                               │ uses
                               ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │  SequenceGenerator                                                 │
  │  counters: {conversation_id -> int}                                │
  │  +next(conv_id) -> int      ← THE ordering authority               │
  │  (distributed: a RANGE LEASE per conversation, see §6)             │
  └────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │  ChatService  (façade) — THE SEND PATH, in order                   │
  │  1. conversation.has(sender)            else NotAParticipant       │
  │  2. by_client lookup                    → DuplicateMessage         │
  │  3. WITH the conversation lock:                                    │
  │       seq  = allocate_seq()                                        │
  │       store.append(Message(..seq..))    ← storage is in seq order  │
  │  4. outbox.add_rows(for every recipient DEVICE)                    │
  │  5. return the message  (the ack carries id + seq)                 │
  └───┬───────────────────────────┬────────────────────┬───────────────┘
      │                           │                    │
      ▼                           ▼                    ▼
┌───────────────┐   ┌────────────────────────┐  ┌──────────────────┐
│  Outbox       │   │  DeliveryReceipt       │  │  PresenceService │
│  rows[]       │   │  matrix:               │  │  lease + TTL     │
│  (message ×   │   │   (msg, recipient) ->  │  │  +heartbeat()    │
│   recipient   │   │   ReceiptState         │  │  +is_online()    │
│   DEVICE)     │   │  forward-only          │  │  expiry is the   │
│───────────────│   │  +aggregate(msg) ->    │  │  only "offline"  │
│ +pending_for()│   │   the WEAKEST state    │  └──────────────────┘
│ +mark()       │   └────────────────────────┘
└───────────────┘
      ▲
      │ is also the OFFLINE MAILBOX: rows accumulate while the
      │ device is away and drain on reconnect

  ┌────────────────────────────┐   ┌──────────────────────────────────┐
  │  SessionRegistry           │   │  SyncService                     │
  │  user -> {connection ids}  │   │  +since(conv, seq) -> [Message]  │
  │  (multi-device fan-out)    │   │  gap = contiguous seq makes the  │
  └────────────────────────────┘   │  missing range computable        │
                                   └──────────────────────────────────┘

  ReceiptState:  PENDING → SENT → DELIVERED → READ   (forward only)
                 aggregate = the MINIMUM across recipients
```

**Read it this way:** there is exactly one ordering authority (`SequenceGenerator`, used under the conversation's lock), exactly one dedupe index (`by_client`), and the receipt matrix is separate from the immutable message because it changes after the fact.

---

## 4. Interfaces / abstract classes / enums

```python
class ReceiptState(Enum):
    PENDING = 0        # the outbox row exists, nothing sent yet
    SENT = 1           # it left this server
    DELIVERED = 2      # the recipient's DEVICE acknowledged
    READ = 3           # the recipient opened the conversation

    def __lt__(self, other: "ReceiptState") -> bool:
        return self.value < other.value


class SequenceGenerator(ABC):
    @abstractmethod
    def next(self, conversation_id: str) -> int:
        """Contiguous and monotonic PER CONVERSATION. Contiguity is load-bearing:
        a receiver that sees seq 7 and then 9 can conclude that exactly one
        message is missing, which is what makes gap fill possible. A generator
        that skipped numbers would make a gap ambiguous."""


class Outbox(ABC):
    @abstractmethod
    def add_rows(self, message: Message, recipients: list[tuple[str, str]]) -> list[OutboxRow]:
        """One row per (recipient, DEVICE). The return value matters: the caller
        needs the rows to address the sends without re-querying."""

    @abstractmethod
    def pending_for(self, device_id: str, now: float) -> list[OutboxRow]: ...
    @abstractmethod
    def mark(self, message_id: str, recipient_id: str, device_id: str,
             state: ReceiptState) -> None: ...


class PresenceService(ABC):
    @abstractmethod
    def heartbeat(self, user_id: str, ttl_seconds: float = 30.0) -> None: ...
    @abstractmethod
    def is_online(self, user_id: str) -> bool:
        """Expiry-based. There is NO explicit `go_offline` on the read path,
        because a crashed client never calls it."""
```

**Four contract notes worth saying out loud:**

- **`next` is contiguous, not merely increasing.** Contiguity is what makes a gap computable. Say this before you are asked.
- **`add_rows` is keyed by device, not by user.** Multi-device means a user is a set of destinations.
- **`mark` is forward-only.** A late "delivered" arriving after a "read" must not roll the state back — a real reordering hazard, because receipts travel over different connections.
- **`is_online` has no counterpart setter.** The absence is the design: presence is a lease, so going offline is something that *happens to* a client, not something it must remember to do.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** In Java this is a `ConcurrentHashMap` of conversations, a `record` for `Message`, and an enum with an explicit ordinal for the receipt ordering. In Go, a per-conversation `sync.Mutex` in a map and a `(Message, bool, error)` return for the dedupe. The **ordering rule, the dedupe index, the receipt matrix and the presence lease** are the design; the syntax is not.

```python
"""whatsapp_lld.py — the classes inside the HLD's boxes.

Design summary
--------------
Message            : FROZEN. It knows nothing about who received it, which is
                     why it is safe to cache and to log.
Ordering           : `seq`, allocated by SequenceGenerator under the
                     conversation's lock. NEVER `sent_at` — two phones disagree
                     by seconds, and a chat that reorders is visibly broken.
Contiguity         : seq is contiguous per conversation, so a receiver seeing
                     7 then 9 knows EXACTLY one message is missing. That is what
                     makes gap fill possible, and why a skipping generator is a
                     bug rather than a quirk.
Dedupe             : ClientMessageId is generated by the CLIENT, because the
                     server has not seen the message when the retry is sent. The
                     index key is (sender_id, client_id).
Outbox             : one row per (message, recipient DEVICE). It IS the offline
                     mailbox: rows accumulate while a device is away.
Receipts           : a per-recipient matrix, forward-only, with an aggregate
                     that is the MINIMUM across recipients — "delivered" means
                     every device has it, not that one does.
Presence           : a LEASE. A crashed client never sends a disconnect, so
                     online is defined by a refreshable expiry, not a flag.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Optional


# --------------------------------------------------------------------------
# 1. Identifiers, the receipt state, and the message
# --------------------------------------------------------------------------

class ReceiptState(Enum):
    """Ordered, and the order is the contract: a receipt only ever moves
    forward. Declared here as well as in §4 so this block runs standalone."""

    PENDING = 0        # the outbox row exists, nothing sent yet
    SENT = 1           # it left this server
    DELIVERED = 2      # the recipient's DEVICE acknowledged
    READ = 3           # the recipient opened the conversation

    def __lt__(self, other: "ReceiptState") -> bool:
        return self.value < other.value


@dataclass(frozen=True)
class ClientMessageId:
    """Generated by the SENDER'S APP before the first attempt. This is the whole
    idempotency mechanism: the server cannot generate it, because the entire
    problem is that the server may never have seen the first attempt."""

    sender_id: str
    client_id: str

    def __post_init__(self) -> None:
        if not self.client_id:
            raise ValueError("a client message id cannot be empty")

    def __str__(self) -> str:
        return f"{self.sender_id}:{self.client_id}"


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    sender_id: str
    seq: int
    body: str
    sent_at: float
    client_msg_id: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.conversation_id}#{self.seq}] {self.sender_id}: {self.body}"


class DuplicateMessage(Exception):
    """A retry whose original was already stored. It carries the ORIGINAL so the
    caller can ack the retry with the real id and seq rather than an error."""

    def __init__(self, existing: Message) -> None:
        super().__init__(f"duplicate of {existing.id} (seq {existing.seq})")
        self.existing = existing


class NotAParticipant(Exception):
    pass


# --------------------------------------------------------------------------
# 2. The message store — two indexes
# --------------------------------------------------------------------------

class MessageStore:
    """Three indexes, because there are three questions.

    by_key    : (conversation, seq)  -> Message   "give me the conversation in order"
    by_client : (sender, client_id)  -> Message   "is this a retry?"
    by_id     : message_id           -> Message   "give me this one message"

    The third is the one people forget. It looks redundant because the first
    index already holds every message — but finding one message by id through
    the ordering index is a full scan of the conversation, and the delivery
    path does exactly that lookup once per outbox row. A drain of 500 pending
    messages against a 100k-message conversation is then 50 million comparisons,
    which is a bug that shows up as latency under exactly the condition the
    drain exists to fix.
    """

    def __init__(self) -> None:
        self.by_key: dict[tuple[str, int], Message] = {}
        self.by_client: dict[tuple[str, str], Message] = {}
        self.by_id: dict[str, Message] = {}
        self._lock = threading.Lock()
        self._sequence = 0

    def next_id(self) -> str:
        with self._lock:
            self._sequence += 1
            return f"msg-{self._sequence:08d}"

    def find_by_client(self, sender_id: str, client_id: str) -> Optional[Message]:
        with self._lock:
            return self.by_client.get((sender_id, client_id))

    def find_by_id(self, message_id: str) -> Optional[Message]:
        """O(1). This is what the delivery path uses, once per outbox row."""
        with self._lock:
            return self.by_id.get(message_id)

    def append(self, message: Message) -> None:
        """Raises DuplicateMessage. The check and the write are one operation,
        for the same reason the sequence allocation is: two concurrent retries
        of the same client id must not both store."""
        with self._lock:
            key = (message.conversation_id, message.seq)
            if key in self.by_key:
                raise DuplicateMessage(self.by_key[key])
            if message.client_msg_id is not None:
                client_key = (message.sender_id, message.client_msg_id)
                if client_key in self.by_client:
                    raise DuplicateMessage(self.by_client[client_key])
                self.by_client[client_key] = message
            self.by_key[key] = message
            self.by_id[message.id] = message

    def range(self, conversation_id: str, from_seq: int, to_seq: int) -> list[Message]:
        """Sorted by seq. The caller never sorts by `sent_at` — this is the
        place that guarantees it."""
        with self._lock:
            found = [m for (conv, seq), m in self.by_key.items()
                     if conv == conversation_id and from_seq <= seq <= to_seq]
        return sorted(found, key=lambda m: m.seq)

    def last_seq(self, conversation_id: str) -> int:
        with self._lock:
            seqs = [seq for (conv, seq) in self.by_key if conv == conversation_id]
        return max(seqs) if seqs else 0

    def count(self) -> int:
        with self._lock:
            return len(self.by_key)

    def gaps(self, conversation_id: str, up_to: int) -> list[int]:
        """Which sequence numbers are MISSING between 1 and `up_to`. This only
        has a well-defined answer because seq is contiguous — with a generator
        that skipped numbers, every skip would look like a gap."""
        present = {seq for (conv, seq) in self.by_key if conv == conversation_id}
        return [n for n in range(1, up_to + 1) if n not in present]


# --------------------------------------------------------------------------
# 3. The sequence generator — the ordering authority
# --------------------------------------------------------------------------

class SequenceGenerator(ABC):
    @abstractmethod
    def next(self, conversation_id: str) -> int: ...


class LocalSequenceGenerator(SequenceGenerator):
    """One counter per conversation, contiguous from 1. This is the authority on
    order; nothing else in the system may decide it."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def next(self, conversation_id: str) -> int:
        with self._lock:
            value = self._counters.get(conversation_id, 0) + 1
            self._counters[conversation_id] = value
            return value

    def current(self, conversation_id: str) -> int:
        with self._lock:
            return self._counters.get(conversation_id, 0)


class Conversation:
    """The aggregate. Its only real responsibility beyond membership is making
    sequence allocation and the append ATOMIC TOGETHER — allocating a seq and
    then failing to append would leave a permanent hole."""

    def __init__(self, conversation_id: str, participants: Iterable[str],
                 devices: Optional[dict[str, set[str]]] = None) -> None:
        self.id = conversation_id
        self.participants: set[str] = set(participants)
        if not self.participants:
            raise ValueError("a conversation needs at least one participant")
        self.devices: dict[str, set[str]] = {
            user: set(devs) for user, devs in (devices or {}).items()
        }
        for user in self.participants:
            self.devices.setdefault(user, {f"{user}-primary"})
        self.last_seq = 0
        self._lock = threading.Lock()

    def has(self, user_id: str) -> bool:
        return user_id in self.participants

    def recipient_devices(self, sending_user: Optional[str] = None,
                          sending_device: Optional[str] = None) -> list[tuple[str, str]]:
        """The fan-out list: (user_id, device_id) pairs.

        The sender's OWN other devices ARE included — a message sent from a
        phone must appear on the sender's laptop, and forgetting that is the
        classic multi-device bug. What is excluded is the one device that sent
        it, so the message is not echoed back to the screen it came from.

        The exclusion is by `sending_device`, an actual device id, rather than
        by a naming convention. An earlier version skipped any device whose name
        looked like a default (a `-primary` suffix), which is a guess about how
        device ids are spelled: in this very file the demo devices are called
        `alice-phone` and `alice-laptop`, so the guess excluded nothing and the
        sender's own phone received its own message. Excluding the id the
        connection actually presented is a fact, not a guess.
        """
        pairs: list[tuple[str, str]] = []
        for user in sorted(self.participants):
            for device in sorted(self.devices.get(user, ())):
                if user == sending_user and device == sending_device:
                    continue
                pairs.append((user, device))
        return pairs

    def allocate_seq(self, generator: SequenceGenerator) -> int:
        with self._lock:
            seq = generator.next(self.id)
            self.last_seq = seq
            return seq

    def append_atomic(self, generator: SequenceGenerator, store: MessageStore,
                      build) -> Message:
        """Allocate the seq and append UNDER THE SAME LOCK.

        Without this, two sends can allocate seq 4 and 5 and then append in the
        order 5, 4 — the store is then in an order that disagrees with the seq,
        and every reader that trusts insertion order is wrong. Holding the lock
        across both makes storage order and sequence order the same thing."""
        with self._lock:
            seq = generator.next(self.id)
            self.last_seq = seq
            message = build(seq)
            store.append(message)
            return message


# --------------------------------------------------------------------------
# 4. The outbox — a message × recipient-device matrix
# --------------------------------------------------------------------------

@dataclass
class OutboxRow:
    message_id: str
    conversation_id: str
    recipient_id: str
    device_id: str
    state: ReceiptState = ReceiptState.PENDING
    attempts: int = 0
    next_attempt_at: float = 0.0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.message_id, self.recipient_id, self.device_id)

    def __str__(self) -> str:
        return (f"{self.message_id} -> {self.recipient_id}/{self.device_id} "
                f"[{self.state.name}]")


class Outbox:
    """One row per destination DEVICE. This is also the offline mailbox: while a
    device is away its rows simply accumulate, and a reconnect drains them.
    There is no separate mailbox structure, because there is nothing a mailbox
    would hold that a pending outbox row does not."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._rows: dict[tuple[str, str, str], OutboxRow] = {}
        self._lock = threading.Lock()
        self.clock = clock

    def add_rows(self, message: Message,
                 recipients: list[tuple[str, str]]) -> list[OutboxRow]:
        if not recipients:
            raise ValueError("a message with no recipients is a bug upstream")
        created: list[OutboxRow] = []
        with self._lock:
            for user_id, device_id in recipients:
                row = OutboxRow(message.id, message.conversation_id,
                                user_id, device_id, next_attempt_at=self.clock())
                if row.key in self._rows:
                    raise DuplicateMessage(message)   # a retried fan-out
                self._rows[row.key] = row
                created.append(row)
        return created

    def pending_for(self, device_id: str, now: Optional[float] = None) -> list[OutboxRow]:
        """What a reconnecting device must be given. Sorted by seq is the
        caller's job — this returns rows, and the CALLER sorts by the message's
        seq, never by row insertion order."""
        moment = self.clock() if now is None else now
        with self._lock:
            return [r for r in self._rows.values()
                    if r.device_id == device_id
                    and r.state is ReceiptState.PENDING
                    and r.next_attempt_at <= moment]

    def mark(self, message_id: str, recipient_id: str, device_id: str,
             state: ReceiptState, now: Optional[float] = None) -> OutboxRow:
        moment = self.clock() if now is None else now
        with self._lock:
            row = self._rows.get((message_id, recipient_id, device_id))
            if row is None:
                raise KeyError(f"no outbox row for {message_id}/{recipient_id}/{device_id}")
            if state.value <= row.state.value:
                # FORWARD ONLY. A late "delivered" arriving after a "read" must
                # not roll the state back — receipts travel over independent
                # connections and reorder all the time.
                return row
            row.state = state
            if state is ReceiptState.PENDING:
                row.attempts += 1
            return row

    def retry_later(self, message_id: str, recipient_id: str, device_id: str,
                    backoff_seconds: float) -> OutboxRow:
        with self._lock:
            row = self._rows.get((message_id, recipient_id, device_id))
            if row is None:
                raise KeyError("no such outbox row")
            row.attempts += 1
            row.next_attempt_at = self.clock() + backoff_seconds
            return row

    def for_message(self, message_id: str) -> list[OutboxRow]:
        with self._lock:
            return [r for r in self._rows.values() if r.message_id == message_id]

    def size(self) -> int:
        with self._lock:
            return len(self._rows)


# --------------------------------------------------------------------------
# 5. Receipts — per recipient, forward-only, with an aggregate
# --------------------------------------------------------------------------

class DeliveryReceipt:
    """The matrix. It is a SEPARATE aggregate from Message for one reason:
    receipts change after the message is frozen, and a mutable Message is not
    safely cacheable or loggable."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], ReceiptState] = {}
        self._lock = threading.Lock()

    def apply(self, message_id: str, recipient_id: str,
              state: ReceiptState) -> ReceiptState:
        with self._lock:
            key = (message_id, recipient_id)
            current = self._states.get(key, ReceiptState.PENDING)
            if state.value <= current.value:
                return current               # forward only, same as the outbox
            self._states[key] = state
            return state

    def state_for(self, message_id: str, recipient_id: str) -> ReceiptState:
        with self._lock:
            return self._states.get((message_id, recipient_id), ReceiptState.PENDING)

    def aggregate(self, message_id: str, recipients: Iterable[str]) -> ReceiptState:
        """The MINIMUM across recipients — "delivered" means EVERY recipient's
        device has it, not that one does. It also means the aggregate cannot
        leak WHO has read: a group where one person read and four did not shows
        "delivered" to the sender, which is the privacy behaviour you want."""
        people = list(recipients)
        if not people:
            return ReceiptState.PENDING
        with self._lock:
            states = [self._states.get((message_id, r), ReceiptState.PENDING)
                      for r in people]
        return min(states)

    def readers(self, message_id: str, recipients: Iterable[str]) -> list[str]:
        with self._lock:
            return [r for r in recipients
                    if self._states.get((message_id, r)) is ReceiptState.READ]


# --------------------------------------------------------------------------
# 6. Presence — a lease, not a flag
# --------------------------------------------------------------------------

class PresenceService:
    """Online is a LEASE with a TTL.

    The reason is blunt: a phone that loses signal, a laptop that sleeps, and a
    process that crashes all never send a disconnect. A boolean `online` flag is
    therefore wrong from the first tunnel onward — it stays true forever. A
    lease expires on its own, so the failure mode is "presence is briefly stale"
    rather than "presence is permanently wrong".
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._expiry: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self.clock = clock

    def heartbeat(self, user_id: str, ttl_seconds: float = 30.0) -> None:
        """Called by the connection layer on every inbound frame, not only on a
        dedicated ping — traffic is the best evidence of liveness."""
        if ttl_seconds <= 0:
            raise ValueError("a lease must have a positive ttl")
        now = self.clock()
        with self._lock:
            self._expiry[user_id] = now + ttl_seconds
            self._last_seen[user_id] = now

    def is_online(self, user_id: str) -> bool:
        now = self.clock()
        with self._lock:
            expiry = self._expiry.get(user_id)
            return expiry is not None and now < expiry

    def go_offline(self, user_id: str) -> None:
        """The GRACEFUL path only — an explicit disconnect. The system does not
        depend on it being called, which is the point of the lease."""
        with self._lock:
            self._expiry.pop(user_id, None)
            self._last_seen[user_id] = self.clock()

    def last_seen(self, user_id: str) -> Optional[float]:
        with self._lock:
            return self._last_seen.get(user_id)

    def online_users(self) -> list[str]:
        now = self.clock()
        with self._lock:
            return sorted(u for u, expiry in self._expiry.items() if now < expiry)


class SessionRegistry:
    """user -> the connections currently serving them. Multi-device means this
    is a SET, and it is what the outbox addresses."""

    def __init__(self) -> None:
        self._sessions: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def connect(self, user_id: str, connection_id: str) -> None:
        with self._lock:
            self._sessions.setdefault(user_id, set()).add(connection_id)

    def disconnect(self, user_id: str, connection_id: str) -> None:
        with self._lock:
            sessions = self._sessions.get(user_id)
            if sessions:
                sessions.discard(connection_id)
                if not sessions:
                    del self._sessions[user_id]

    def connections(self, user_id: str) -> set[str]:
        with self._lock:
            return set(self._sessions.get(user_id, ()))

    def is_connected(self, user_id: str) -> bool:
        return bool(self.connections(user_id))


# --------------------------------------------------------------------------
# 7. Sync — gap fill
# --------------------------------------------------------------------------

class SyncService:
    def __init__(self, store: MessageStore) -> None:
        self.store = store

    def since(self, conversation_id: str, since_seq: int,
              limit: int = 500) -> list[Message]:
        """Everything after `since_seq`, IN SEQ ORDER. A reconnecting client
        calls this with the highest seq it holds."""
        last = self.store.last_seq(conversation_id)
        if since_seq >= last:
            return []
        return self.store.range(conversation_id, since_seq + 1,
                                min(last, since_seq + limit))

    def missing(self, conversation_id: str, have: Iterable[int]) -> list[int]:
        """Which seq numbers a client is missing, given the ones it holds. Only
        answerable because seq is contiguous."""
        held = set(have)
        last = self.store.last_seq(conversation_id)
        return [n for n in range(1, last + 1) if n not in held]


# --------------------------------------------------------------------------
# 8. The façade — THE SEND PATH
# --------------------------------------------------------------------------

class ChatService:
    def __init__(self, store: MessageStore, generator: SequenceGenerator,
                 outbox: Outbox, receipts: DeliveryReceipt,
                 presence: PresenceService, sessions: SessionRegistry,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.generator = generator
        self.outbox = outbox
        self.receipts = receipts
        self.presence = presence
        self.sessions = sessions
        self.clock = clock
        self.conversations: dict[str, Conversation] = {}
        self.sync = SyncService(store)

    def create_conversation(self, conversation_id: str, participants: Iterable[str],
                            devices: Optional[dict[str, set[str]]] = None) -> Conversation:
        conversation = Conversation(conversation_id, participants, devices)
        self.conversations[conversation_id] = conversation
        return conversation

    def conversation(self, conversation_id: str) -> Conversation:
        try:
            return self.conversations[conversation_id]
        except KeyError:
            raise KeyError(f"no such conversation: {conversation_id}") from None

    def send(self, sender_id: str, conversation_id: str, body: str,
             client_msg_id: Optional[str] = None,
             device_id: Optional[str] = None) -> Message:
        """THE SEND PATH. The order of these five steps is the design.

        1. Membership, before anything is allocated.
        2. Dedupe on the CLIENT id — before allocating a seq, so a retry does
           not consume a sequence number. This is what keeps the sequence
           contiguous: a duplicate that allocated a seq and then failed the
           append would leave a permanent hole.
        3. Allocate the seq and append, under the conversation lock.
        4. Fan out one outbox row per recipient DEVICE.
        5. Return — the ack carries the server id AND the seq.

        `device_id` is the sending device, and it exists so the fan-out can
        exclude exactly that one destination. It is Optional because a
        server-originated message has no sending device, in which case nothing
        is excluded.
        """
        conversation = self.conversation(conversation_id)
        if not conversation.has(sender_id):
            raise NotAParticipant(f"{sender_id} is not in {conversation_id}")
        if not body:
            raise ValueError("an empty message is not a message")

        # 2. Dedupe FIRST.
        if client_msg_id is not None:
            existing = self.store.find_by_client(sender_id, client_msg_id)
            if existing is not None:
                raise DuplicateMessage(existing)

        # 3. Allocate + append, atomically.
        sent_at = self.clock()
        message = conversation.append_atomic(
            self.generator, self.store,
            lambda seq: Message(self.store.next_id(), conversation_id, sender_id,
                                seq, body, sent_at, client_msg_id))

        # 4. Fan out to every recipient DEVICE, including the sender's own
        #    other devices but not the device that sent it.
        recipients = conversation.recipient_devices(
            sending_user=sender_id, sending_device=device_id)
        self.outbox.add_rows(message, recipients)

        # 5. A receipt row exists implicitly for every recipient, starting at
        #    PENDING; recording it makes the aggregate well-defined immediately.
        for user_id, _device in recipients:
            self.receipts.apply(message.id, user_id, ReceiptState.PENDING)
        return message

    # -- delivery ---------------------------------------------------------
    def deliver(self, message_id: str, recipient_id: str, device_id: str) -> ReceiptState:
        row = self.outbox.mark(message_id, recipient_id, device_id,
                               ReceiptState.DELIVERED)
        return self.receipts.apply(message_id, recipient_id, ReceiptState.DELIVERED)

    def mark_read(self, message_id: str, recipient_id: str,
                  device_id: str) -> ReceiptState:
        """Opening the conversation on ANY device marks it read for that user —
        reading on a phone should not leave it unread on a laptop, because
        "read" is a fact about the person, not the device."""
        self.outbox.mark(message_id, recipient_id, device_id, ReceiptState.READ)
        return self.receipts.apply(message_id, recipient_id, ReceiptState.READ)

    def aggregate_for(self, message: Message) -> ReceiptState:
        """The aggregate the SENDER sees: every participant except themselves.

        The recipients are DERIVED here rather than passed in. Taking the
        recipient list as a parameter looks more flexible and is strictly
        worse: the caller can pass the wrong set, and a wrong set produces a
        confidently wrong answer rather than an error — omit one recipient and
        the aggregate reports "delivered" while somebody has nothing. A method
        that cannot be called incorrectly beats a method that is convenient.
        """
        conversation = self.conversation(message.conversation_id)
        others = conversation.participants - {message.sender_id}
        return self.receipts.aggregate(message.id, others)

    def aggregate_of(self, message_id: str, conversation_id: str) -> ReceiptState:
        """The same answer, from an id rather than an object — what a UI needs
        when it is rendering a message it has not loaded."""
        message = self.store.find_by_id(message_id)
        if message is None:
            raise KeyError(f"no such message: {message_id}")
        conversation = self.conversation(conversation_id)
        others = conversation.participants - {message.sender_id}
        return self.receipts.aggregate(message_id, others)

    # -- reconnect --------------------------------------------------------
    def drain(self, recipient_id: str, device_id: str) -> list[Message]:
        """What a reconnecting device must receive: its pending outbox rows,
        turned back into messages and ordered BY SEQ.

        Two things are deliberate. The message lookup goes through `by_id`, so
        draining 500 rows is 500 hash lookups rather than 500 scans — the scan
        would be quadratic in exactly the situation this method exists to fix.
        And the ordering is applied HERE, at the last moment, from the message's
        seq: never from the row's insertion order, which a dict does not
        promise, and never from `sent_at`, which is not the order.
        """
        messages: list[Message] = []
        for row in self.outbox.pending_for(device_id):
            if row.recipient_id != recipient_id:
                continue
            message = self.store.find_by_id(row.message_id)
            if message is not None:
                messages.append(message)
        messages.sort(key=lambda m: (m.conversation_id, m.seq))
        for message in messages:
            self.outbox.mark(message.id, recipient_id, device_id,
                             ReceiptState.SENT)
        return messages


# --------------------------------------------------------------------------
# 9. Tests and demo
# --------------------------------------------------------------------------

def _check(name: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    if not condition:
        raise SystemExit(1)


def _raises(exc_type, fn) -> Optional[str]:
    try:
        fn()
    except exc_type as exc:
        return str(exc)
    return None


def _service() -> tuple[ChatService, MessageStore, Outbox, DeliveryReceipt,
                        PresenceService]:
    store = MessageStore()
    outbox = Outbox()
    receipts = DeliveryReceipt()
    presence = PresenceService()
    svc = ChatService(store, LocalSequenceGenerator(), outbox, receipts,
                      presence, SessionRegistry())
    return svc, store, outbox, receipts, presence


if __name__ == "__main__":
    # --- 1. the sequence generator is contiguous and per-conversation ---
    gen = LocalSequenceGenerator()
    _check("a fresh conversation starts at 1", gen.next("c1") == 1)
    _check("and continues", [gen.next("c1"), gen.next("c1")] == [2, 3])
    _check("another conversation is independent", gen.next("c2") == 1)
    _check("counters do not interfere", gen.next("c1") == 4)
    _check("contiguity holds over many draws",
           [gen.next("c3") for _ in range(5)] == [1, 2, 3, 4, 5])

    # --- 2. sending, and the three indexes ---
    svc, store, outbox, receipts, presence = _service()
    svc.create_conversation("alice-bob", ["alice", "bob"])
    m1 = svc.send("alice", "alice-bob", "hello", client_msg_id="c-1")
    _check("the first message has seq 1", m1.seq == 1)
    _check("and a server id", m1.id.startswith("msg-"))
    _check("the store has it under (conv, seq)",
           store.by_key[("alice-bob", 1)] is m1)
    _check("and under the client id",
           store.by_client[("alice", "c-1")] is m1)
    _check("and under the message id, which is the O(1) delivery lookup",
           store.by_id[m1.id] is m1 and store.find_by_id(m1.id) is m1)
    _check("an unknown message id is None, not a KeyError",
           store.find_by_id("msg-nope") is None)
    m2 = svc.send("bob", "alice-bob", "hi back", client_msg_id="b-1")
    _check("the reply has seq 2", m2.seq == 2)

    # --- 3. ORDERING IS BY SEQ, NOT BY TIMESTAMP ---
    # Build a conversation where sent_at runs BACKWARDS as seq runs forwards —
    # exactly what happens when two phones' clocks disagree.
    now = {"t": 1000.0}
    store_t = MessageStore()
    svc_t = ChatService(store_t, LocalSequenceGenerator(), Outbox(),
                        DeliveryReceipt(), PresenceService(),
                        SessionRegistry(), clock=lambda: now["t"])
    svc_t.create_conversation("clock-skew", ["a", "b"])
    skewed: list[Message] = []
    for index, body in enumerate(["first", "second", "third"]):
        now["t"] = 1000.0 - index * 10          # the clock goes BACKWARDS
        skewed.append(svc_t.send("a", "clock-skew", body))
    _check("the seq still increments", [m.seq for m in skewed] == [1, 2, 3])
    _check("but sent_at decrements", skewed[0].sent_at > skewed[2].sent_at)
    synced = svc_t.sync.since("clock-skew", 0)
    _check("sync returns them in SEQ order despite the clock",
           [m.body for m in synced] == ["first", "second", "third"])
    # The point, stated as a contrast: a reader that trusted sent_at would show
    # this conversation BACKWARDS.
    by_clock = [m.body for m in sorted(synced, key=lambda m: m.sent_at)]
    _check("a reader that sorted by sent_at would show it backwards",
           by_clock == ["third", "second", "first"])

    # --- 4. the dedupe: a retry is not a second message ---
    msg = _raises(DuplicateMessage,
                  lambda: svc.send("alice", "alice-bob", "hello",
                                   client_msg_id="c-1"))
    _check("resending the same client id is a duplicate", msg is not None)
    _check("and the store still holds exactly two messages", store.count() == 2)
    _check("the duplicate carries the ORIGINAL so the retry can be acked",
           "seq 1" in (msg or ""))
    _check("a client id is scoped to the SENDER",
           svc.send("bob", "alice-bob", "different", client_msg_id="c-1").seq == 3)
    _check("a send with no client id is never deduped",
           svc.send("alice", "alice-bob", "again").seq == 4)

    # A duplicate must NOT consume a sequence number — that is what keeps the
    # sequence contiguous, so a gap always means a genuinely missing message.
    before = store.last_seq("alice-bob")
    _raises(DuplicateMessage, lambda: svc.send("alice", "alice-bob", "hello",
                                               client_msg_id="c-1"))
    _check("a duplicate consumes no sequence number",
           store.last_seq("alice-bob") == before)
    _check("so the sequence has no holes",
           store.gaps("alice-bob", store.last_seq("alice-bob")) == [])

    # --- 5. membership is checked before anything is allocated ---
    msg = _raises(NotAParticipant,
                  lambda: svc.send("carol", "alice-bob", "let me in"))
    _check("a non-participant cannot send", msg is not None)
    _check("and no seq was burned", store.last_seq("alice-bob") == before)
    _check("an empty body is refused",
           _raises(ValueError,
                   lambda: svc.send("alice", "alice-bob", "")) is not None)

    # --- 6. the outbox is a MATRIX keyed by device ---
    svc2, store2, outbox2, receipts2, presence2 = _service()
    svc2.create_conversation(
        "group", ["alice", "bob", "carol"],
        devices={"alice": {"alice-phone", "alice-laptop"},
                 "bob": {"bob-phone"},
                 "carol": {"carol-phone", "carol-tablet"}})
    group_msg = svc2.send("alice", "group", "hello everyone",
                          device_id="alice-phone")
    rows = outbox2.for_message(group_msg.id)
    _check("one row per recipient DEVICE, not per person", len(rows) == 4)
    _check("the sender's own laptop is included",
           ("alice", "alice-laptop") in {r.key[1:] for r in rows})
    _check("but not the sending device itself",
           ("alice", "alice-phone") not in {r.key[1:] for r in rows})
    _check("every row starts PENDING",
           all(r.state is ReceiptState.PENDING for r in rows))
    # With no sending device — a server-originated message — nothing is
    # excluded, so the count goes up by exactly one.
    server_msg = svc2.send("alice", "group", "Alice added Bob")
    _check("a message with no sending device excludes nobody",
           len(outbox2.for_message(server_msg.id)) == 5)

    # --- 7. receipts: per recipient, forward-only, aggregated ---
    _check("the aggregate starts PENDING",
           svc2.aggregate_for(group_msg) is ReceiptState.PENDING)
    svc2.deliver(group_msg.id, "bob", "bob-phone")
    _check("one delivery does not make the message delivered",
           svc2.aggregate_for(group_msg) is ReceiptState.PENDING)
    svc2.deliver(group_msg.id, "carol", "carol-phone")
    svc2.deliver(group_msg.id, "carol", "carol-tablet")
    _check("once EVERY recipient has it, so does the message",
           svc2.aggregate_for(group_msg) is ReceiptState.DELIVERED)

    # A late "delivered" after a "read" must not roll the state back.
    svc2.mark_read(group_msg.id, "bob", "bob-phone")
    _check("bob reading is recorded", 
           receipts2.state_for(group_msg.id, "bob") is ReceiptState.READ)
    svc2.deliver(group_msg.id, "bob", "bob-phone")     # a LATE re-delivery ack
    _check("a late DELIVERED does not roll a READ back",
           receipts2.state_for(group_msg.id, "bob") is ReceiptState.READ)
    _check("and the aggregate is still the weakest link",
           svc2.aggregate_for(group_msg) is ReceiptState.DELIVERED)  # carol hasn't read
    svc2.mark_read(group_msg.id, "carol", "carol-phone")
    _check("when everyone has read, so has the message",
           svc2.aggregate_for(group_msg) is ReceiptState.READ)

    # The aggregate hides WHO read — the privacy behaviour.
    _check("readers knows who read, and it is not the aggregate",
           receipts2.readers(group_msg.id, ["bob", "carol"]) == ["bob", "carol"]
           or set(receipts2.readers(group_msg.id, ["bob", "carol"])) == {"bob", "carol"})
    _check("a message with no recipients aggregates to PENDING",
           receipts2.aggregate(group_msg.id, []) is ReceiptState.PENDING)
    _check("aggregate_of answers the same question from an id",
           svc2.aggregate_of(group_msg.id, "group") is svc2.aggregate_for(group_msg))
    _check("aggregate_of on an unknown message raises",
           _raises(KeyError, lambda: svc2.aggregate_of("msg-nope", "group")) is not None)

    # --- 8. the outbox row state machine is forward-only too ---
    row = outbox2.for_message(group_msg.id)[0]
    key = row.key
    outbox2.mark(*key, ReceiptState.SENT)
    _check("a row advances PENDING -> SENT", outbox2.for_message(group_msg.id)[0].state
           is ReceiptState.SENT or
           [r for r in outbox2.for_message(group_msg.id) if r.key == key][0].state
           is ReceiptState.SENT)
    _check("marking back to PENDING is refused",
           outbox2.mark(*key, ReceiptState.PENDING).state is not ReceiptState.PENDING)

    # --- 9. the outbox IS the offline mailbox ---
    svc3, store3, outbox3, receipts3, presence3 = _service()
    svc3.create_conversation("dm", ["alice", "bob"],
                             devices={"alice": {"a1"}, "bob": {"b1"}})
    offline: list[Message] = []
    for i in range(5):
        offline.append(svc3.send("alice", "dm", f"queued {i}"))
    _check("five messages queued while bob is away",
           len(outbox3.pending_for("b1")) == 5)
    _check("and none of bob's rows are delivered yet",
           all(r.state is ReceiptState.PENDING for r in outbox3.pending_for("b1")))
    drained = svc3.drain("bob", "b1")
    _check("a reconnect drains them all", len(drained) == 5)
    _check("in SEQ order", [m.seq for m in drained] == [1, 2, 3, 4, 5])
    _check("and they are no longer pending",
           len(outbox3.pending_for("b1")) == 0)

    # --- 10. gap fill needs contiguity ---
    _check("no gaps after the drain", store3.gaps("dm", 5) == [])
    _check("sync from 2 returns 3,4,5",
           [m.seq for m in svc3.sync.since("dm", 2)] == [3, 4, 5])
    _check("sync from the last seq returns nothing",
           svc3.sync.since("dm", 5) == [])
    _check("sync from 0 returns everything",
           len(svc3.sync.since("dm", 0)) == 5)
    _check("a client missing seq 3 is told exactly that",
           svc3.sync.missing("dm", [1, 2, 4, 5]) == [3])

    # --- 11. presence is a lease, and it expires ---
    now_p = {"t": 500.0}
    presence_p = PresenceService(clock=lambda: now_p["t"])
    _check("an unknown user is offline", presence_p.is_online("alice") is False)
    presence_p.heartbeat("alice", ttl_seconds=30.0)
    _check("a heartbeat makes them online", presence_p.is_online("alice") is True)
    _check("and last_seen is recorded", presence_p.last_seen("alice") == 500.0)
    now_p["t"] = 529.0
    _check("still online just before the lease expires",
           presence_p.is_online("alice") is True)
    now_p["t"] = 531.0
    _check("OFFLINE after it expires, with no disconnect sent",
           presence_p.is_online("alice") is False)
    _check("and they are no longer in the online list",
           presence_p.online_users() == [])
    _check("last_seen survives the expiry",
           presence_p.last_seen("alice") == 500.0)
    presence_p.heartbeat("alice")
    _check("a new heartbeat brings them back", presence_p.is_online("alice") is True)
    presence_p.go_offline("alice")
    _check("a graceful disconnect takes effect immediately",
           presence_p.is_online("alice") is False)
    _check("a non-positive ttl is refused",
           _raises(ValueError, lambda: presence_p.heartbeat("x", ttl_seconds=0))
           is not None)

    # --- 12. the session registry is a set per user ---
    registry = SessionRegistry()
    registry.connect("alice", "conn-1")
    registry.connect("alice", "conn-2")
    _check("two devices means two connections",
           registry.connections("alice") == {"conn-1", "conn-2"})
    _check("and the user is connected", registry.is_connected("alice") is True)
    registry.disconnect("alice", "conn-1")
    _check("dropping one device keeps the other",
           registry.connections("alice") == {"conn-2"})
    registry.disconnect("alice", "conn-2")
    _check("dropping the last device disconnects the user",
           registry.is_connected("alice") is False)
    _check("disconnecting an unknown connection is harmless",
           registry.disconnect("nobody", "nothing") is None)

    # --- 13. THE ordering race: concurrent sends, one contiguous sequence ---
    svc4, store4, outbox4, _r4, _p4 = _service()
    svc4.create_conversation("race", ["a", "b", "c", "d", "e", "f"])
    senders = ["a", "b", "c", "d", "e", "f"]
    barrier = threading.Barrier(len(senders) * 5)
    sent: list[Message] = []
    sent_lock = threading.Lock()
    errors: list[BaseException] = []

    def racer(sender: str, index: int) -> None:
        barrier.wait()
        try:
            message = svc4.send(sender, "race", f"m{index}",
                                client_msg_id=f"{sender}-{index}",
                                device_id=f"{sender}-primary")
        except BaseException as exc:                     # noqa: BLE001
            with sent_lock:
                errors.append(exc)
            return
        with sent_lock:
            sent.append(message)

    threads = [threading.Thread(target=racer, args=(s, i))
               for s in senders for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _check("no send raised", errors == [])
    _check("all 30 sends stored", len(sent) == 30 and store4.count() == 30)
    seqs = sorted(m.seq for m in sent)
    _check("the sequence is exactly 1..30 with NO gaps", seqs == list(range(1, 31)))
    _check("and no duplicates", len(set(seqs)) == 30)
    _check("storage order and sequence order agree (no holes)",
           store4.gaps("race", 30) == [])
    _check("the generator agrees with the store",
           svc4.generator.current("race") == 30)
    _check("one outbox row per non-sender participant per message",
           outbox4.size() == 30 * 5)

    # The invariant that the atomic append exists to protect: reading back in
    # seq order gives the messages in the order they were numbered.
    readback = svc4.sync.since("race", 0, limit=100)
    _check("reading back gives all 30 in seq order",
           [m.seq for m in readback] == list(range(1, 31)))

    # --- 14. the conversation aggregate ---
    conv = svc4.conversation("race")
    _check("last_seq tracks the allocation", conv.last_seq == 30)
    _check("membership is a set", conv.has("a") and not conv.has("z"))
    _check("an unknown conversation raises",
           _raises(KeyError, lambda: svc4.conversation("nope")) is not None)
    _check("a conversation with no participants is refused",
           _raises(ValueError, lambda: Conversation("empty", [])) is not None)
    _check("a user with no devices gets a default device",
           Conversation("d", ["solo"]).devices["solo"] == {"solo-primary"})

    print("\nall checks passed")
```

**What to point at:**

- `Conversation.append_atomic` allocates the sequence and appends **under the same lock**. Allocating outside it lets two sends take seq 4 and 5 and store them in the order 5, 4 — and then every reader that trusts insertion order is wrong. The demo races 30 sends from 6 threads and asserts the sequence is exactly 1..30 with no gaps.
- `ChatService.send` **dedupes before allocating a sequence number**. A duplicate that allocated and then failed would leave a permanent hole, and a hole breaks gap fill — the demo asserts a duplicate consumes no seq.
- The clock-skew test builds a conversation where `sent_at` runs backwards as `seq` runs forwards, and asserts `sync` returns them in seq order. That is the ordering rule, demonstrated rather than asserted in prose.
- `OutboxRow.mark` and `DeliveryReceipt.apply` are both **forward-only**: the demo sends a late `DELIVERED` after a `READ` and asserts the state does not roll back.
- `PresenceService` has no setter for offline — the demo advances an injected clock past the lease and asserts the user goes offline with **no disconnect call**.

---

## 6. Concurrency & thread-safety

| Hazard | What goes wrong | Fix |
|---|---|---|
| **Two sends allocate and append out of order** | seq 4 and 5 are stored 5 then 4; readers trusting insertion order see the wrong order | `append_atomic`: one lock over allocate + append |
| **Two retries of one client id** | Two messages stored, the recipient sees it twice | The dedupe index, checked and written under the store lock |
| **A duplicate consumes a seq** | A permanent hole, and gap fill reports a message that never existed | Dedupe **before** allocation |
| **Ordering by `sent_at`** | Two phones with skewed clocks produce a visibly reordered chat | The sort key is `seq`, everywhere; `sent_at` is display only |
| **A receipt arriving out of order** | A late `DELIVERED` after a `READ` rolls the state back and the sender sees "delivered" again | Forward-only comparison in `apply` and `mark` |
| **Reading the outbox dictionary order as the delivery order** | Messages delivered in hash order rather than conversation order | Sort by the message's `seq` at the last moment, in `drain` |
| **A crashed client staying online forever** | A boolean flag with no expiry; presence is permanently wrong | The lease: expiry is the only offline path the system depends on |
| **A heartbeat per user rather than per connection** | One device closing marks the user offline while another is still connected | Presence is per user, sessions are per connection; `SessionRegistry` decides |
| **A reconnect draining while a send is in flight** | A message written between the pending query and the mark is lost or double-sent | The row's idempotent `mark`; a re-drain returns nothing for rows already advanced |
| **The sequence counter read and written non-atomically** | Two sends get the same seq, and one overwrites the other in the store | `LocalSequenceGenerator` holds a lock over read-and-increment, and `append_atomic` raises on the collision |
| **A group membership change mid-send** | The fan-out list is computed from a set being mutated | Membership is read once into a list before the fan-out |
| **Multi-process** | Two servers have two counters per conversation | A **range lease** per conversation — see below |

### The distributed sequence: a lease per conversation

```python
class LeasedSequenceGenerator(SequenceGenerator):
    """The counter, when more than one server can send to a conversation.

    Each server leases a BLOCK of sequence numbers for a conversation and hands
    them out locally with no coordination at all; it only talks to the store
    when the block is exhausted. That is the same trick as the URL shortener's
    range allocator, and it has the same cost — with a twist that is specific
    to chat.

    THE TWIST: contiguity is no longer guaranteed per conversation. If server A
    leases 1..100 and server B leases 101..200, and A dies after using 40, then
    seq 41..100 are never issued. A receiver seeing 40 then 101 sees a gap of 60
    that will NEVER be filled, so "gap means fetch the missing range" breaks:
    the client would retry forever for messages that do not exist.

    The fix is that the lease is only safe if it can be RELEASED. A lease with a
    short TTL and an explicit release on clean shutdown means an abandoned block
    is reissued, and the client's gap-fill rule becomes "a gap older than the
    lease TTL is permanent, so stop asking". That is the honest answer: the
    invariant does not survive distribution for free, and you have to choose
    between a longer lease (fewer round trips, longer false gaps) and a shorter
    one (more coordination, faster convergence).
    """

    def __init__(self, store, conversation_id: str, block_size: int = 100) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.block_size = block_size
        self._next = 0
        self._limit = 0
        self._lock = threading.Lock()

    def next(self, conversation_id: str) -> int:
        with self._lock:
            if self._next >= self._limit:
                self._next = self.store.reserve_sequence_block(
                    conversation_id, self.block_size)
                self._limit = self._next + self.block_size
            value = self._next
            self._next += 1
            return value

    def release(self) -> int:
        """Hand the unused tail of the block back. Called on clean shutdown;
        the TTL on the lease is the fallback for a crash."""
        with self._lock:
            unused = self._limit - self._next
            if unused > 0:
                self.store.release_sequence_block(self.conversation_id,
                                                  self._next, self._limit)
            self._next = self._limit = 0
            return unused
```

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Edit a message"

The design point: an edit is a **new version of the same sequence position**, not a new message, and the recipient must be able to apply it out of order.

```python
@dataclass(frozen=True)
class MessageVersion:
    message_id: str
    version: int
    body: str
    edited_at: float


class EditableMessageStore:
    """Versions live BESIDE the message, keyed by (message_id, version), and the
    message keeps only `version`. An edit never allocates a new seq — it would
    reorder the conversation if it did, since the edited message would appear at
    the end.

    Out-of-order application is the hazard: a client may receive version 3
    before version 2. It applies a version only if it is HIGHER than the one it
    holds, which is the same forward-only rule as the receipts."""

    def __init__(self) -> None:
        self._versions: dict[str, dict[int, MessageVersion]] = {}
        self._lock = threading.Lock()

    def add(self, message_id: str, body: str, edited_at: float) -> MessageVersion:
        with self._lock:
            versions = self._versions.setdefault(message_id, {})
            version = MessageVersion(message_id, len(versions) + 1, body, edited_at)
            versions[version.version] = version
            return version

    def latest(self, message_id: str) -> Optional[MessageVersion]:
        with self._lock:
            versions = self._versions.get(message_id)
            if not versions:
                return None
            return versions[max(versions)]

    def apply(self, held: Optional[MessageVersion],
              incoming: MessageVersion) -> MessageVersion:
        if held is None or incoming.version > held.version:
            return incoming
        return held
```

### 7.2 "Delete a message"

Two different operations that are always conflated, and the distinction is a design point:

```python
class DeleteScope(Enum):
    FOR_ME = "for_me"          # hide locally; the bytes remain
    FOR_EVERYONE = "for_everyone"   # a tombstone the recipients must see


def delete_message(scope: DeleteScope, store: MessageStore,
                   message_id: str) -> Optional[Message]:
    """FOR_EVERYONE is a TOMBSTONE, not a deletion, and it must ride the
    sequence like any other change — otherwise a client that was offline during
    the delete never learns about it. FOR_ME is a per-device local hide and
    never touches the store, which is why it needs no sequence number and no
    coordination."""
    if scope is DeleteScope.FOR_ME:
        return None
    found = [m for m in store.by_key.values() if m.id == message_id]
    if not found:
        return None
    original = found[0]
    tombstone = replace(original, body="")
    with store._lock:                                  # the store's own lock
        store.by_key[(original.conversation_id, original.seq)] = tombstone
    return tombstone
```

### 7.3 "Typing indicators and read receipts in real time"

The design point: these are **not messages**. They have no sequence number, must never be stored, and must never be retried.

```python
class EphemeralBus:
    """Typing, presence updates and live read receipts. Every property that
    makes a Message correct is wrong for these: they are not ordered, not
    durable, not retried, and losing one is fine because another arrives in a
    second. Routing them through the message path would burn sequence numbers
    and fill the store with junk."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set] = {}
        self._lock = threading.Lock()

    def subscribe(self, conversation_id: str, connection) -> None:
        with self._lock:
            self._subscribers.setdefault(conversation_id, set()).add(connection)

    def publish(self, conversation_id: str, event: dict) -> int:
        """Best effort. Returns how many connections it reached, and no error
        if the answer is zero — a typing indicator for an empty room is not a
        failure."""
        with self._lock:
            targets = list(self._subscribers.get(conversation_id, ()))
        delivered = 0
        for connection in targets:
            try:
                connection.send(event)
                delivered += 1
            except Exception:                          # noqa: BLE001
                continue
        return delivered
```

### 7.4 "Group membership changes"

The hard part is not adding a member — it is **where the new member's history starts**, and the answer is a sequence number:

```python
class MembershipChange(ABC):
    @abstractmethod
    def history_from(self, conversation: Conversation,
                     store: MessageStore) -> int: ...


class HistoryFromJoin(MembershipChange):
    """The new member sees messages from the seq at which they joined. This is
    the only implementable default for an unencrypted group, and it is a
    one-line rule because seq gives us the join point exactly."""

    def history_from(self, conversation: Conversation, store: MessageStore) -> int:
        return conversation.last_seq + 1


class HistoryFromBeginning(MembershipChange):
    """For a small trusted group. It requires replaying history to a new device,
    which is why it is opt-in rather than the default."""

    def history_from(self, conversation: Conversation, store: MessageStore) -> int:
        return 1


def apply_membership_change(change: MembershipChange, conversation: Conversation,
                            store: MessageStore, user_id: str) -> int:
    start = change.history_from(conversation, store)
    conversation.participants.add(user_id)
    conversation.devices.setdefault(user_id, {f"{user_id}-primary"})
    return start
```

### 7.5 "Why is the message not editable in place?"

Because immutability buys three things that are individually worth it: a frozen message can be cached without an invalidation problem, it can be appended to a log without rewriting, and it can be compared in a test by value. Editing is expressed as a **version beside the message** (§7.1) rather than a mutation of it, so the three properties survive the feature.

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Ordering by `sent_at`** | Two phones with skewed clocks reorder the chat visibly | Order by `seq`, allocated once by one authority |
| 2 | **A client resends after a lost ack** | The recipient sees the message twice | The client-generated id, deduped on `(sender, client_id)` |
| 3 | **The server generates the dedupe key** | It cannot — the retry arrives before the server has a record of the first attempt | The **client** generates it, before the first attempt |
| 4 | **A duplicate consumes a sequence number** | A permanent hole in the sequence | Dedupe before allocation |
| 5 | **Allocate and append not atomic** | Storage order disagrees with sequence order | `append_atomic` holds one lock over both |
| 6 | **A non-contiguous generator** | Every skipped number looks like a missing message, and gap fill fetches forever | Contiguity is a hard invariant; gaps mean missing |
| 7 | **A non-member sends** | The message is stored and fanned out to people who should not see it | The membership check comes first |
| 8 | **A receipt arriving out of order** | A late `DELIVERED` rolls a `READ` back | Forward-only comparison |
| 9 | **One row per recipient instead of per device** | Your laptop never shows what your phone sent | One row per device |
| 10 | **The sender's own devices excluded** | The message is missing on the sender's other device — the classic multi-device bug | Include them, minus the sending device |
| 11 | **"Delivered" when one recipient has it** | A group message shows delivered while four people have nothing | The aggregate is the **minimum** across recipients |
| 12 | **A read receipt naming who read** | A group is told exactly who has read, which is a privacy leak and a social problem | The aggregate is a state, not a list; `readers()` is for the sender's own audit only |
| 13 | **Online as a boolean** | A crashed client is online forever | A lease with a TTL |
| 14 | **Presence per connection** | Closing one device marks the user offline while another is connected | Presence is per user; sessions are per connection |
| 15 | **A gap that will never be filled** | With a leased distributed generator, a client retries for a block nobody will ever issue | The gap-fill rule becomes "permanent after the lease TTL" |
| 16 | **A message with no recipients** | A row-less message that is silently never delivered | `add_rows` raises; it is a bug upstream, not a valid state |
| 17 | **Empty body** | An empty bubble appears | Refused at the façade |
| 18 | **Typing indicators routed through the message path** | Sequence numbers burned on events that must never be stored | An ephemeral path with no seq and no retry |
| 19 | **Editing by allocating a new seq** | The edited message jumps to the end of the conversation | Versions beside the message, keyed by `(message_id, version)` |
| 20 | **A reconnect draining twice** | Duplicate delivery | `mark` is idempotent and advances the row, so a second drain finds nothing |

**The three to volunteer:** #1 (ordering is the signature problem and the clock is the trap), #4 plus #6 (the duplicate/hole interaction, which is subtle and only shows up as a client fetching forever), and #11 (the aggregate being the minimum — "delivered" is a claim about a set).

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **Value Object** | `Message`, `ClientMessageId`, `MessageVersion` | Frozen and hashable: safe to cache, safe to log, comparable in a test | An allocation per message |
| **Aggregate** | `Conversation` owns the invariant (seq allocation and append are one atomic step); `DeliveryReceipt` owns the matrix | One boundary per invariant, so the invariant cannot be broken from outside | Two aggregates to coordinate |
| **Repository** | `MessageStore`, `Outbox` | The two indexes and the matrix are storage concerns | An abstraction over a dict, worth it for the invariant tests |
| **Strategy** | `SequenceGenerator` (local vs leased), `MembershipChange` | The local and distributed generators have genuinely different guarantees | Two implementations of one interface |
| **State machine** | `ReceiptState`, forward-only | A receipt is a monotonic fact; making the monotonicity explicit prevents the late-arrival rollback | A comparison on every transition |
| **Lease** | `PresenceService` | The only correct model for a liveness claim about a process that cannot be trusted to say goodbye | A TTL to tune, and staleness bounded by it |
| **Façade** | `ChatService` | The send path is five ordered steps and belongs in one readable place | The façade is not thin — it owns the order of the steps |
| **Policy** | `DeleteScope`, `HistoryFromJoin` vs `HistoryFromBeginning` | Product decisions expressed as data | An extra object for a one-line rule |
| **Singleton** | *Deliberately rejected* | A global store makes two tests interfere, which is fatal for a file with 60+ assertions | — |
| **Observer** | *Deliberately limited* to `EphemeralBus` | A durable delivery must be an explicit outbox row, not a subscription — a subscriber that throws must not lose a message | The ephemeral and durable paths are separate |

**The sentence to give:** *"Value Object for the message because immutability is what makes caching and logging safe; two aggregates because there are two invariants — the sequence allocation and the receipt matrix — and they change at different times; a Strategy for the generator because the local and distributed versions make different guarantees about contiguity; and a lease for presence because the alternative is a flag that is wrong from the first dropped connection."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Ordering by sequence number** | Yes | Total order independent of any clock; gaps are computable | An allocation authority per conversation, and a lock on the send path |
| **Client-generated dedupe id** | Yes | Retries are safe after any lost ack | The client must persist an id across restarts |
| **Dedupe before allocation** | Yes | The sequence stays contiguous, so gaps are meaningful | The lookup is on the hot path of every send |
| **Atomic allocate + append** | Yes | Storage order equals sequence order | A lock held across two operations |
| **Immutability of `Message`** | Yes | Cacheable, loggable, comparable | Editing needs a version structure |
| **Receipts as a matrix** | Yes | Correct for groups and multi-device | One row per recipient per device — the write amplification is real |
| **Aggregate = minimum** | Yes | "Delivered" means what people think it means | A single dead device keeps a message at "sent" forever |
| **Aggregate hides who read** | Yes | No social pressure and no privacy leak | A sender cannot see individual reads through the aggregate, so a separate audit path is needed |
| **Presence as a lease** | Yes | A crashed client goes offline on its own | Presence is stale for up to the TTL |
| **Outbox doubles as the mailbox** | Yes | One structure, and the offline path is the same code as the online path | The outbox grows with offline time and needs a cap |
| **Ephemeral events off the message path** | Yes | No sequence numbers spent on typing indicators | Two paths to maintain |
| **In-process locks** | Yes, with the lease path shown | Simple and correct for one server | Contiguity needs a lease TTL before distribution |

**The trade to name unprompted:** *"The receipt matrix costs one row per recipient per device, and for a large group that is the dominant write. I would keep it, because the alternative — a flag on the message — cannot express 'three of five have it', and that is the state the UI actually shows. What I would change at scale is the aggregate: computing a minimum across a thousand recipients on every read is a fan-out of its own, so the aggregate would be maintained incrementally as a counter of how many recipients have reached each state, rather than recomputed."*

---

## 11. Q&A

**Q1. Where is per-conversation ordering actually enforced?**
In two places, and they are different jobs. On the **sender's** server, `Conversation.append_atomic` allocates the sequence number and appends under one lock — so allocation order and storage order agree, and two concurrent sends cannot take seq 4 and 5 and store them as 5, 4. On the **receiver**, the order is enforced by sorting on `seq` at the last moment: `SyncService.since` and `drain` both sort by `message.seq`, never by insertion order and never by `sent_at`. The critical property is that the sequence number is assigned exactly once, by one authority, and everything downstream treats it as the order. Timestamps are for display — the demo builds a conversation where `sent_at` runs backwards as `seq` runs forwards and asserts the read comes back in seq order.

**Q2. Why can't you just use timestamps?**
Because a clock is a local, unverifiable claim. Two phones are seconds apart after a flight; a device with a wrong timezone or a failed NTP sync can be hours out; and a *malicious* client can claim any time it likes. If the sort key is a field the sender controls, the sender controls the conversation's order. A sequence number assigned by the server under a lock is not a claim — it is a fact, and every client agrees on it because they all receive it. The second reason is gap detection: with a contiguous sequence, "I have 7 and 9, so I am missing exactly 8" is computable. With timestamps, a gap is not detectable at all, because timestamps are neither unique nor contiguous.

**Q3. Why does the client generate the dedupe id?**
Because of when the retry happens. The client sends, the network drops the *acknowledgement*, and the client resends — and the server may have stored the first attempt already, or may not have received it at all. If the server generated the key, the client would have nothing stable to send on the retry, and the server would see two unrelated requests. So the client generates `client_msg_id` **before the first attempt** and persists it until it is acked; a retry carries the same id and the server collapses it. The key is `(sender_id, client_id)`, not the id alone, because two senders can pick the same uuid — extremely unlikely with a uuid, but the composite key costs nothing and removes the assumption.

**Q4. Why is dedupe checked before the sequence is allocated?**
Because a hole in the sequence is a lie that the client cannot resolve. If a duplicate allocated seq 8 and then failed the append, the sequence would go 7, 9 — and a receiver that sees 7 then 9 correctly concludes that one message is missing and asks for it, forever, because message 8 never existed. Deduping first means a duplicate never touches the counter, so any gap is a genuinely missing message. The demo asserts exactly this: a duplicate consumes no sequence number and the conversation has no holes.

**Q5. Why is the outbox a matrix rather than a flag?**
Because delivery is a fact about a destination. A group of five with two devices each is ten destinations, and they advance independently — one phone is in a tunnel, one laptop is asleep, one person reads on their tablet. A flag on the message cannot express "three of ten have it", which is exactly the state the UI shows. So `OutboxRow` is keyed by `(message_id, recipient_id, device_id)`, and the receipt aggregate is computed from those rows. The write amplification is real: one message to a ten-device group is ten rows.

**Q6. What does "delivered" mean for a group, exactly?**
Every recipient device has acknowledged it. That is the **minimum** across recipients, not the maximum and not the count. The demo makes this concrete: one of three recipients receiving the message still leaves the aggregate at `PENDING`, because the sender's "delivered" tick means "everyone has it". The alternative — showing delivered as soon as one has it — is a UI that lies. The cost is that a single dead device pins a message at "sent" forever, and that is a real, common complaint; the honest answer is that the *display* may want a second state ("delivered to some") while the underlying aggregate stays the minimum.

**Q7. How do read receipts work, and what about privacy?**
`DeliveryReceipt.apply` is forward-only, so a `READ` can never be rolled back by a late `DELIVERED` — receipts travel over independent connections and genuinely reorder, and the demo sends a late delivery ack on a read message and asserts the state holds. For a group, the aggregate is again the minimum, which has a useful side effect: the sender sees "read" only when everyone has read, so the aggregate cannot reveal *who* read. `readers()` exists for the sender's own audit, and routing that to the group would be both a privacy leak and a social problem — the classic "why did you leave me on read" issue is a product decision that the data model should not make by accident.

**Q8. Why is presence a lease rather than a flag?**
Because the failure mode of a flag is permanent. A phone in a tunnel, a laptop that sleeps, and a crashed process all fail to send a disconnect, so a boolean `online` stays true forever and the system believes somebody is available who is not. A lease expires on its own: the client refreshes on every frame (traffic is better evidence of liveness than a dedicated ping), and if the refreshes stop the entry lapses. The consequence is that presence is deliberately **stale for up to the TTL**, which is the correct trade — a presence claim that is briefly wrong is much better than one that is permanently wrong. The demo advances an injected clock past the lease and asserts the user is offline with no disconnect having been sent.

**Q9. Why is the message immutable?**
Three properties, each worth it alone. A frozen object can be **cached** without any invalidation problem, because it never changes — which matters enormously on the read path. It can be **appended to a log** without a rewrite. And it can be **compared by value** in a test, which is why this file's assertions read as `m1.seq == 1` rather than as a mock-verification dance. Editing and deletion are then expressed as structures *beside* the message — `MessageVersion` keyed by `(message_id, version)`, tombstones that ride the sequence — rather than as mutations of it, so all three properties survive the features.

**Q10 (senior). Where does this design break at scale?**
At the sequence generator, and the break is more interesting than "use a range lease". A lease makes contiguity conditional: if a server leases 1..100, uses 40, and dies, then 41..100 are never issued, and a client that sees 40 then 101 sees a gap that will never be filled — so "gap means fetch the missing range" becomes an infinite retry. The honest fix is that a lease must be **releasable** and short-lived, and the client's gap rule becomes "a gap older than the lease TTL is permanent, stop asking". So the invariant that makes this design work in one process does not survive distribution for free, and you have to pick a TTL and accept a false-gap window. The second break is the receipt matrix: for a large group, maintaining the aggregate by scanning every recipient is a fan-out per read, so it has to become an incremental counter of how many recipients have reached each state.

**Q11 (senior). How would you handle a conversation with a million participants?**
Badly, if you kept this model. The fan-out is the problem: one message is a million outbox rows, which is a write amplification of a million, and the receipt aggregate is a minimum over a million recipients. Real systems treat this as a different problem — a broadcast channel rather than a conversation — and the class-level consequences are that the outbox becomes a **cursor per subscriber** (each device tracks its own position rather than having a row per message), and receipts stop being per recipient entirely, because a million read receipts is not a feature anybody wants. The honest answer in an interview is to say that this is where the model changes shape rather than to pretend the same classes scale by adding shards.

**Q12. What would you test first?**
The three that are invisible until they are wrong in production. The **ordering invariant** under concurrency — many threads sending, asserting the sequence is exactly 1..N with no gaps and no duplicates, which is the atomic-append property. The **dedupe interacting with the sequence** — a duplicate consumes no seq and leaves no hole, because that bug presents as a client fetching a message that never existed, which is very hard to diagnose from a bug report. And the **forward-only receipts** — a late `DELIVERED` after a `READ`. After those, the presence lease with an injected clock, because a test that sleeps for a TTL is a test that will be deleted.

---

### Pushback round

**P1. "Your `send` takes the conversation lock, so a busy conversation serialises every send. That is a bottleneck."**
It is, and it is a deliberate one: the lock covers exactly two operations — increment a counter and append to a dictionary — and neither can do I/O, so the critical section is microseconds. The alternative is to allocate and append separately, which is what breaks the ordering invariant, or to use an atomic counter and accept that storage order can disagree with sequence order, which means every reader has to sort — and then I have moved the cost from a microsecond lock to every read in the system. If a single conversation genuinely outgrew one lock, the answer is a shard per conversation rather than a weaker invariant, because one conversation's ordering is the whole point of a conversation.

**P2. "`client_msg_id` is optional, so every server-originated message — 'Alice added Bob', a bot reply, a system notice — is stored with no dedupe key at all. A retry inside your own fan-out double-posts it."**
Correct, and it is a real gap rather than a simplification. The dedupe mechanism is keyed on an id the *client* generates, and a server-originated message has no client. The fix is not to make the field required — there genuinely is no client id for "Alice added Bob" — but to have the **server generate a deterministic id for its own messages**: something like `sys:{conversation_id}:{event_type}:{event_id}`, where `event_id` is the id of the thing that caused the message. Then the same `by_client` index dedupes it, keyed on a synthetic sender of `"system"`, and a retried fan-out collapses exactly like a retried send. The reason it is worth saying out loud is that the bug it prevents is *invisible*: a duplicated system message looks like a UI glitch, and nobody traces it back to a missing idempotency key.

**P3. "Your aggregate returns the minimum, so a group message with one offline device shows 'sent' forever. Users hate that."**
They do, and the honest version of this answer distinguishes the **data** from the **display**. The aggregate — the minimum — is the correct underlying fact: "delivered" as a promise means everyone has it. But the UI should not render one state; it should render a **distribution**, which is what messaging apps actually do: one tick for sent, two for delivered-to-some, two blue for read. That is a projection over the matrix, and the matrix has exactly the information needed, because it is per recipient. So the design is not wrong, it is incomplete: the aggregate is the strictest useful summary and the UI wants a different one, and the fix is another projection function over the same rows rather than a change to the aggregate.

**P4. "Presence is a write on every frame from every connection and a read on every send. At any real scale that is a hot key per user, and you are now paying for it twice."**
Yes, and the cost is real — `heartbeat` on every inbound frame is a write amplification equal to your inbound traffic, and `is_online` on every send is a read on the same key. Two things make it acceptable. First, **presence is a UI affordance and never a routing decision**: the outbox delivers to a device whether or not the user is online, so a stale or missing presence answer changes what the app *shows* and never what it *does*. That decoupling is what lets you cache it aggressively. Second, the write can be **batched per connection** — a connection that is already marked live with, say, 20 seconds of lease left does not need to refresh on every frame, only when it crosses a threshold, which turns "write per frame" into "one write per 10 seconds per connection". The read side then gets a short-TTL cache (5–10 seconds is imperceptible for a green dot) with the same accepted staleness the lease already has. So the answer is not that the cost is imaginary; it is that presence is deliberately allowed to be approximate, which is what buys the right to cache it.

---

## 12. Cheat sheet

**In one paragraph.** A message is immutable and carries a `seq`; the `seq` is allocated by one authority per conversation, under a lock, atomically with the append, and it — never `sent_at` — is the order. Clients generate a `client_msg_id` before their first attempt so a retry collapses on `(sender, client_id)`, and the dedupe runs *before* the sequence allocation so a duplicate leaves no hole; contiguity is what makes a gap mean "a message is missing". The outbox is a matrix keyed by `(message_id, recipient, device)` and doubles as the offline mailbox. Receipts are per recipient, forward-only, and aggregate to the **minimum** — "delivered" is a claim about a set. Presence is a lease with a TTL, because a crashed client never says goodbye.

**The five sentences that carry the design**

1. *"Ordering is a sequence number assigned under a lock, never a timestamp."*
2. *"The client generates the dedupe id, because the server has not seen the message when the retry is sent."*
3. *"Dedupe runs before allocation, or a duplicate leaves a permanent hole in the sequence."*
4. *"One outbox row per recipient device — the outbox is also the offline mailbox."*
5. *"Delivered is the minimum across recipients, and presence is a lease, not a flag."*

**The class table**

| Class | Its one job | The mistake it prevents |
|---|---|---|
| `ClientMessageId` | A client-generated id, persisted until acked | A lost ack producing a duplicate message |
| `Message` | Frozen: id, conv, sender, `seq`, body, `sent_at` | Mutating a message and breaking caching/logging |
| `MessageStore` | Three indexes: order, dedupe, id | A retry stored twice; a delivery draining in O(n²) |
| `Conversation` | Membership, plus atomic allocate-and-append | Storage order disagreeing with sequence order |
| `SequenceGenerator` | Contiguous per-conversation numbering | Gaps that are indistinguishable from lost messages |
| `Outbox` / `OutboxRow` | One row per destination device; the mailbox | A device that misses what was sent while it was away |
| `DeliveryReceipt` | Per-recipient matrix, forward-only, minimum aggregate | A late ack rolling a read back; a group lying about delivery |
| `PresenceService` | A lease with a TTL | A crashed client shown as online forever |
| `SessionRegistry` | user → connections | Closing one device marking the user offline |
| `SyncService` | `since(conv, seq)` gap fill | A reconnecting client with a hole it cannot name |
| `ChatService` | The five-step send path, in order | The order of the steps being spread across the codebase |

**The send path, in order — say it in this order**

`membership` → `dedupe on (sender, client_id)` → **under the lock:** `allocate seq` + `append` → `one outbox row per recipient device` → `return (id, seq)`.

**Failure modes to name before you are asked**

| Symptom | Cause |
|---|---|
| The chat is visibly reordered | Ordering by `sent_at` |
| The recipient sees a message twice | No client-generated dedupe id |
| A client fetches a message that never existed, forever | A duplicate that consumed a sequence number |
| The sender's laptop is missing what the phone sent | A fan-out per recipient instead of per device |
| A group shows "delivered" while four people have nothing | The aggregate is a maximum or a count, not a minimum |
| Somebody is online who went home hours ago | Presence as a boolean |
| A reconnecting client has a hole it cannot describe | A non-contiguous sequence |

**Cross-links**

- **[The HLD this descends from](../HLD/02-whatsapp-chat-system.md)** — delivery guarantees, the WebSocket tier, offline storage, media, and the fan-out at scale. The HLD decides *that* the system is at-least-once with per-conversation ordering; this file is the classes that make that true.
- **[URL shortener LLD](01-url-shortener-lld.md)** — the other end of the same range-allocator idea. That file's `RangeAllocator` mints codes in blocks to avoid a round trip per code; this file's §6 uses the identical trick for sequence numbers and shows what it costs *there* — contiguity stops being free, and a gap-fill rule needs a lease TTL to stay honest.
- **[`LLD/problems/13-pub-sub-system.md`](../../LLD/problems/13-pub-sub-system.md)** — the outbox as a durable hand-off between producing and delivering, and the difference between at-most-once, at-least-once and exactly-once delivery. Read it for the *queue* mechanics that this file deliberately keeps out of the class model.
- **[`LLD/problems/16-food-delivery.md`](../../LLD/problems/16-food-delivery.md)** — the technique twin. That file's table-driven, actor-aware state machine and its client-key idempotency are the same two ideas as this file's `ReceiptState` and `ClientMessageId`, applied to orders instead of messages; if you want to see the state machine done more thoroughly, it is there.
- **[`LLD/problems/12-logger-framework.md`](../../LLD/problems/12-logger-framework.md)** — the asynchronous hand-off pattern, for the delivery worker that drains outbox rows.
- **[`LLD/problems/11-rate-limiter.md`](../../LLD/problems/11-rate-limiter.md)** — what you put in front of `send` when a client retries in a loop; the two designs meet at the retry.
- **[The LLD question bank](../../Interview-Questions/08-lld-questions.md)** — the machine-coding questions this file's shape answers.
- **[The LLD cheat sheet](../../Cheat-Sheets/07-lld-cheatsheet.md)** — the condensed form of every file in this directory.

**Grounding note.** The reference repositories contain **no OOP/LLD material**, and the source transcripts name real-time messaging without teaching a class design for it. Everything above — the sequence authority, the dedupe index, the outbox matrix, the forward-only receipt states and the presence lease — is standard industry practice assembled for this problem, not material recovered from a transcript. Where the HLD file states a system-level property, this file is consistent with it and cross-links rather than repeats.


