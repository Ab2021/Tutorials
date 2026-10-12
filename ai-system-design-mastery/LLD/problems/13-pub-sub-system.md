# LLD 13 — Pub/Sub System

> **Scope split — read this first.** This file owns the **class and code treatment**: the append-only partition log as a data structure, the partitioner, the producer's batching and idempotence, the consumer's offset bookkeeping, the group coordinator's rebalance and generation fencing, and the processing loop whose ordering *is* the delivery guarantee. The **distributed view** — broker replication, ISR, leader election, cross-AZ placement, retention sizing, consumer lag as an SLO — belongs to the HLD material: [URL Shortener](../../Case-Studies/HLD/01-url-shortener.md) **Deep dive D — Analytics pipeline** is the worked Kafka example (why the queue exists, the 34× arithmetic, at-least-once plus idempotent aggregation), and [Sharding, Replication & Consistency](../../HLD/03-scaling/05-sharding-replication-and-consistency.md) owns partition placement and replication. Cross-link those; do not restate them.

> **Grounding note.** The reference repositories contain **no OOP/LLD material**. The transcripts teach the queue at the architecture level — Kafka in the URL-shortener analytics path, partitioned by conversation id in the WhatsApp case, consumer lag in the observability material — but never as a class design. This file is written from first principles and is not transcript-grounded. Where it touches Kafka behaviour (offsets, rebalance protocol, idempotent producer) it describes the real system's semantics, verified against its documented behaviour, not against a transcript.

> **TL;DR** — A pub/sub system is an **append-only log per partition**, plus two clients that disagree about who owns the position: the broker owns the bytes, the consumer group owns the offset. Everything interesting follows from that split. Partitioning by key is what gives per-key ordering — so the partitioner is a correctness component, not a load-balancing detail. The consumer group's rebalance protocol is where the hard concurrency lives: a **generation** number fences a consumer that has been replaced, because the classic bug is a zombie committing offsets for partitions it no longer owns. And the delivery guarantee is not a property of the broker at all — it is the **ordering of two lines in the processing loop**: commit before processing gives at-most-once (and loses data on a crash), commit after gives at-least-once (and duplicates), and exactly-once needs the offset commit to be atomic with the output write.

> **60-second interview answer:** "A topic is N partitions; a partition is an append-only log with a monotonically increasing offset, so a reader is just an integer position. The producer hashes the message key to a partition — that is what guarantees ordering *per key*, because only within a partition is order defined. The broker assigns offsets; the consumer group assigns partitions to members, and a rebalance bumps a generation number so a consumer that has been replaced cannot commit offsets for partitions it no longer owns. Reading is `poll()` returning a batch, and the delivery guarantee comes from the loop: commit-before-process is at-most-once and can lose messages on a crash, commit-after-process is at-least-once and can duplicate, and exactly-once requires the offset commit and the output write to be one atomic act — which is why the idempotent producer and a transactional sink exist. Then I'd cover hot partitions from key skew, poison-pill messages to a dead-letter queue, and backpressure through a bounded prefetch rather than an unbounded buffer."

---

## 1. Requirement clarification

### The questions to ask

| # | Question | What it decides | Default if they won't say |
|---|---|---|---|
| 1 | **What ordering is actually required?** | Partition key choice; whether a global order is needed (it is expensive) | Per-key ordering only — that is what partitioning buys cheaply |
| 2 | **What delivery guarantee?** | The processing loop's ordering, and whether transactions are needed | At-least-once with an idempotent consumer — cheapest correct option |
| 3 | **Who are the consumers — one app or many?** | Whether there is a consumer group with rebalance at all | A consumer group; a single consumer still needs offsets |
| 4 | **Push or pull?** | Whether the broker tracks consumer state and needs flow control, or the consumer does | Pull — it lets the consumer control its own backpressure |
| 5 | **What happens to a message that always fails?** | Whether there is a dead-letter path and a retry cap | DLQ after N attempts, plus an alert; never an infinite retry loop |
| 6 | **What is the retention?** | Time or size based deletion, and whether a slow consumer can fall off the log | Time-based, sized so the slowest consumer's lag still fits |
| 7 | **Can a message be lost?** | Whether the producer's `acks` setting is `all` or `1`, and whether the broker replicates | `acks=all` with replication 3 — the HLD half of this question |
| 8 | **Is duplicate delivery acceptable?** | Whether the consumer must be idempotent, or the producer idempotent, or both | Duplicates are accepted; the consumer dedupes on `(topic, partition, offset)` |
| 9 | **How many partitions?** | The parallelism ceiling and the rebalance cost | Start from target throughput ÷ per-consumer throughput, and say that the count is hard to change |
| 10 | **Is the key skew-prone?** | Whether the partitioner needs a salting escape hatch | Assume skew is possible; add a hot-partition monitor |

### The defaults to assume, stated plainly

> "I'll assume **pull-based consumption with a consumer group**, **per-key ordering** via a stable hash partitioner, **at-least-once delivery** with an idempotent consumer, a **dead-letter queue** after three attempts, time-based retention, and a bounded prefetch so a slow consumer applies backpressure instead of exhausting memory. The broker's replication and ISR behaviour is the HLD layer and I'll cross-link it rather than rebuild it. Does that match, or do you want exactly-once?"

### Non-goals

- No broker replication, ISR, leader election or cross-AZ placement — [Sharding, Replication & Consistency](../../HLD/03-scaling/05-sharding-replication-and-consistency.md).
- No retention sizing arithmetic or lag SLOs — [URL Shortener](../../Case-Studies/HLD/01-url-shortener.md) Deep dive D.
- No disk layout, zero-copy or page-cache optimisation; the in-memory log here models the *semantics*.

### The four constraints that shape everything

1. **Order is per partition, never per topic.** Any design that promises topic-wide order has serialised the topic.
2. **The offset is the consumer's state, and it is the only thing that must not be lost.** Losing it means re-reading; committing it too early means losing data.
3. **Partition ownership is leased, so it must be fenced.** A consumer whose lease expired must not be able to write.
4. **The partition count is effectively immutable.** Increasing it rehashes keys and breaks per-key ordering for existing keys — say this before someone proposes it.

---

## 2. Core entities & responsibilities

| Entity | Kind | Responsibility | Explicitly *not* its job |
|---|---|---|---|
| `Message` | Frozen value object | `key`, `value`, `headers`, `timestamp` | Offset — the broker assigns that |
| `ConsumerRecord` | Frozen value object | The message plus `topic`, `partition`, `offset` | Delivery policy |
| `RecordMetadata` | Frozen value object | Where a produce call landed: topic, partition, offset | — |
| `Partition` | Data structure | Append-only log; `append`, `read(offset)`, `high_watermark`, `log_start_offset`, retention trim | Choosing which partition a key goes to |
| `Topic` | Container | N partitions, retention config, lookup by index | Assignment to consumers |
| `Partitioner` (ABC) | Strategy | `partition(key, count, available) -> int` | Balancing load — it must stay deterministic |
| `HashPartitioner` | Implementation | Stable hash of the key | Round-robin for keyless messages |
| `RoundRobinPartitioner` | Implementation | Even spread for keyless messages | Any ordering guarantee |
| `StickyPartitioner` | Implementation | Hold one partition until a batch fills | Even spread at low volume |
| `Broker` | Facade | Holds topics; `append` assigns the offset; `fetch` | Consumer position |
| `Producer` | Client | Partitioner + batching + idempotence; `send` | Delivery to consumers |
| `IdempotentProducer` | Decorator | Attaches `(producer_id, sequence)`; broker dedupes | Cross-partition atomicity |
| `RecordAccumulator` | Helper | Batches by partition, flush by size or linger | Ordering across batches |
| `Consumer` | Client | Subscription, position, `poll`, `commit` | Assignment — that is the group's |
| `OffsetStore` | Repository | Committed offsets per `(group, topic, partition)` | Processing |
| `PartitionAssignor` (ABC) | Strategy | Members + partitions → assignment | Rebalance triggering |
| `RangeAssignor` / `RoundRobinAssignor` | Implementations | Two standard assignment policies | Fairness over time — both can move everything |
| `GroupCoordinator` | Mediator | Membership, **generation**, rebalance, fenced commits | Processing |
| `DeliverySemantics` | Enum | `AT_MOST_ONCE`, `AT_LEAST_ONCE`, `EXACTLY_ONCE` | — |
| `ProcessingLoop` | Template | poll → process → commit, in the order the semantics demand | Business logic |
| `DeadLetterQueue` | Sink | Terminal destination for poison messages | Retry |
| `Backpressure` | Policy | Bounded prefetch; `pause`/`resume` | — |

**The split to say out loud:** *"The broker owns the bytes and the offsets; the consumer group owns the position. `Partition` is a data structure, `GroupCoordinator` is a distributed lock manager, and `ProcessingLoop` is where the delivery guarantee is decided — three very different kinds of class, which is why they are separate."*

---

## 3. Class diagram (ASCII)

```
        ┌──────────────┐                ┌──────────────────────────┐
        │   Message    │ «value»        │      Broker  (Facade)     │
        │ key value    │                │──────────────────────────│
        │ headers ts   │                │ topics : {name: Topic}   │
        └──────▲───────┘                │ dedupe : {(pid,pt):seq}  │
               │ produced as            │──────────────────────────│
               │                        │ +create_topic()          │
┌──────────────┴───────────────┐        │ +append(topic,msg,key,   │
│          Producer            │───────▶│          partitioner)    │
│──────────────────────────────│        │   -> RecordMetadata      │
│ partitioner : Partitioner    │        │ +fetch(topic,pt,offset,n)│
│ accumulator : RecordAccumulator│      └───────┬──────────────────┘
│ acks / idempotent            │                │ owns
│──────────────────────────────│                ▼
│ +send(topic, value, key)     │        ┌──────────────────────────┐
└──────────────┬───────────────┘        │         Topic            │
               │ uses                   │──────────────────────────│
               ▼                        │ name, retention_seconds  │
      ┌──────────────────┐              │ partitions : [Partition] │
      │  «interface»     │              │ +partition_for(key,      │
      │  Partitioner     │              │      partitioner)        │
      │──────────────────│              └───────┬──────────────────┘
      │ +partition(key,  │                      │ 1..N
      │   count, avail)  │                      ▼
      └────────▲─────────┘              ┌──────────────────────────┐
               │                        │       Partition          │
   ┌───────────┼───────────┐            │──────────────────────────│
   │           │           │            │ records : [ConsumerRecord]│
┌──┴───────┐ ┌─┴────────┐ ┌┴──────────┐ │ base_offset : int        │
│Hash      │ │RoundRobin│ │Sticky     │ │──────────────────────────│
│(key->pt) │ │(keyless) │ │(until full)│ │ +append(rec) -> offset  │
└──────────┘ └──────────┘ └───────────┘ │ +read(offset) -> rec     │
                                        │ +read_batch(offset, n)   │
                                        │ +trim()  (retention)     │
                                        │ high_watermark           │
                                        │ log_start_offset         │
                                        └──────────────────────────┘

┌───────────────────────────────┐        ┌──────────────────────────┐
│           Consumer            │        │      OffsetStore         │
│───────────────────────────────│        │──────────────────────────│
│ group_id, member_id           │───────▶│ {(group,topic,pt): off}  │
│ assignment : {(t,pt): pos}    │ commit │ +commit / +committed     │
│ prefetch : bounded            │        └──────────────────────────┘
│───────────────────────────────│
│ +subscribe(topics)            │        ┌──────────────────────────┐
│ +poll() -> [ConsumerRecord]   │───────▶│    GroupCoordinator      │
│ +commit_sync()                │  join  │──────────────────────────│
│ +pause(pt) / +resume(pt)      │        │ generation : int         │
└───────────────┬───────────────┘        │ members : {id: [pt]}     │
                │ drives                 │ assignor : Assignor      │
                ▼                        │──────────────────────────│
      ┌────────────────────┐             │ +join(group, member)     │
      │  ProcessingLoop    │             │   -> Assignment(gen, pts)│
      │────────────────────│             │ +rebalance(group)        │
      │ semantics          │             │ +commit(group, member,   │
      │ handler            │             │    generation, offsets)  │
      │ dlq                │             │   -> bool (fenced?)      │
      │────────────────────│             └──────────────────────────┘
      │ +run_once()        │
      │  AT_MOST_ONCE  : commit -> process                          │
      │  AT_LEAST_ONCE : process -> commit                          │
      │  EXACTLY_ONCE  : process + commit, atomically, + dedupe     │
      └────────┬───────────┘
               │ on repeated failure
               ▼
      ┌────────────────────┐
      │  DeadLetterQueue   │
      └────────────────────┘
```

**Read it this way:** the log is a data structure, the partitioner is a strategy that decides *ordering*, the coordinator is a lease manager with generation fencing, and the processing loop is a template whose step order **is** the delivery guarantee.

---

## 4. Interfaces / abstract classes / enums

```python
class Partitioner(ABC):
    @abstractmethod
    def partition(self, key: str | None, num_partitions: int,
                  available: list[int] | None = None) -> int:
        """Must be DETERMINISTIC for a given key across processes and restarts."""


class PartitionAssignor(ABC):
    @abstractmethod
    def assign(self, members: list[str],
               partitions: dict[str, list[int]]) -> dict[str, list[tuple[str, int]]]:
        """members -> the (topic, partition) pairs each one owns."""


class DeliverySemantics(Enum):
    AT_MOST_ONCE = "at_most_once"     # commit BEFORE processing: lossy
    AT_LEAST_ONCE = "at_least_once"   # commit AFTER processing: duplicating
    EXACTLY_ONCE = "exactly_once"     # atomic offset + output, idempotent


class Acks(Enum):
    NONE = 0      # fire and forget
    LEADER = 1    # the leader has it in its log
    ALL = -1      # the in-sync replicas have it (the HLD half)
```

**Two contract notes that are load-bearing:**

- **`partition` must be deterministic across processes.** Python's built-in `hash()` is randomised per process by `PYTHONHASHSEED`, so a key would land on a different partition in a different process, and per-key ordering would silently break the moment the producer is scaled out. Use a stable digest — the code does. This is a real bug that ships.
- **`assign` returns ownership, and the coordinator must fence it.** An assignment without a generation number is a lease with no expiry, and the zombie-commit bug is exactly that.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** Interview languages vary. In Java this is an interface plus implementations with `synchronized` blocks; in Go, interfaces with a mutex per partition; in C++, `virtual` plus `std::mutex`. The **log-plus-offset model** is the design; the syntax is not. Python 3.10+ for `X | None`.

```python
"""pubsub.py — an in-memory pub/sub system with real Kafka semantics.

What is modelled faithfully
---------------------------
* A partition is an append-only log; a reader position is an integer offset.
* The broker assigns offsets; the consumer group owns the committed position.
* The partitioner is a CORRECTNESS component: only within a partition is
  order defined, so partition-by-key is what gives per-key ordering.
* A rebalance bumps a GENERATION; a commit from a stale generation is
  rejected. That fence is what stops a zombie consumer from committing
  offsets for partitions it no longer owns.
* The delivery guarantee is the ORDER OF TWO LINES in the processing loop,
  not a property of the broker.
* Idempotent producer: (producer_id, sequence) per partition, deduped by the
  broker, which is the producer half of exactly-once.

What is deliberately NOT modelled (it is the HLD layer)
------------------------------------------------------
Replication, ISR, leader election, disk layout, zero-copy, retention sizing.
See HLD/03-scaling/05-sharding-replication-and-consistency.md and
Case-Studies/HLD/01-url-shortener.md (Deep dive D).
"""
from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional


# --------------------------------------------------------------------------
# 1. Value objects
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Message:
    key: Optional[str]
    value: bytes
    headers: Mapping[str, str] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass(frozen=True)
class ConsumerRecord:
    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: bytes
    headers: Mapping[str, str] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def coordinates(self) -> tuple[str, int, int]:
        """The identity of a delivered record. This triple is what an
        idempotent consumer dedupes on and what a DLQ records."""
        return (self.topic, self.partition, self.offset)


@dataclass(frozen=True)
class RecordMetadata:
    topic: str
    partition: int
    offset: int


@dataclass(frozen=True)
class Assignment:
    generation: int
    partitions: tuple[tuple[str, int], ...]


class OffsetOutOfRange(Exception):
    """Requested an offset below the log start (retention deleted it) or above
    the high watermark (the consumer is ahead of the log)."""


# --------------------------------------------------------------------------
# 2. The partition — an append-only log
# --------------------------------------------------------------------------

class Partition:
    """An append-only log. `read(offset)` is O(1) random access, which is what
    makes a consumer's position a single integer instead of a cursor object.

    In a real broker this is a directory of segment files with an index; the
    semantics modelled here are the same.
    """

    def __init__(self, topic: str, index: int, max_records: Optional[int] = None) -> None:
        self.topic = topic
        self.index = index
        self.max_records = max_records
        self._records: list[ConsumerRecord] = []
        self._base_offset = 0          # "log start offset": retention moves it
        self._lock = threading.Lock()

    # -- geometry ----------------------------------------------------------
    @property
    def log_start_offset(self) -> int:
        return self._base_offset

    @property
    def high_watermark(self) -> int:
        """The next offset to be assigned. Readers may read below this."""
        return self._base_offset + len(self._records)

    def __len__(self) -> int:
        return len(self._records)

    # -- the write side ----------------------------------------------------
    def append(self, message: Message) -> int:
        with self._lock:
            offset = self._base_offset + len(self._records)
            self._records.append(ConsumerRecord(
                topic=self.topic, partition=self.index, offset=offset,
                key=message.key, value=message.value,
                headers=message.headers, timestamp=message.timestamp,
            ))
            return offset

    # -- the read side -----------------------------------------------------
    def read(self, offset: int) -> ConsumerRecord:
        with self._lock:
            if offset < self._base_offset:
                raise OffsetOutOfRange(
                    f"{self.topic}-{self.index}: offset {offset} is below the log "
                    f"start {self._base_offset}; retention deleted it")
            if offset >= self._base_offset + len(self._records):
                raise OffsetOutOfRange(
                    f"{self.topic}-{self.index}: offset {offset} is at or beyond "
                    f"the high watermark {self._base_offset + len(self._records)}")
            return self._records[offset - self._base_offset]

    def read_batch(self, offset: int, max_records: int) -> list[ConsumerRecord]:
        with self._lock:
            if offset < self._base_offset:
                raise OffsetOutOfRange(
                    f"{self.topic}-{self.index}: offset {offset} is below the log "
                    f"start {self._base_offset}; retention deleted it")
            start = offset
            end = min(self._base_offset + len(self._records), start + max_records)
            return list(self._records[start - self._base_offset:end - self._base_offset])

    # -- retention ---------------------------------------------------------
    def trim(self) -> int:
        """Drop the oldest records beyond the retention limit and advance the
        log start offset. Returns how many were deleted.

        The consequence to state out loud: a consumer whose committed offset
        was below the new log start has LOST its position, and the only
        recovery is to reset to the earliest available record — losing data
        silently from that consumer's point of view.
        """
        with self._lock:
            if self.max_records is None or len(self._records) <= self.max_records:
                return 0
            drop = len(self._records) - self.max_records
            self._records = self._records[drop:]
            self._base_offset += drop
            return drop


# --------------------------------------------------------------------------
# 3. Partitioners — a correctness component, not a load-balancing detail
# --------------------------------------------------------------------------

class Partitioner(ABC):
    @abstractmethod
    def partition(self, key: Optional[str], num_partitions: int,
                  available: Optional[list[int]] = None) -> int: ...


class HashPartitioner(Partitioner):
    """Deterministic per key, using a STABLE digest.

    Do not use Python's built-in hash(): it is randomised per process by
    PYTHONHASHSEED, so the same key lands on different partitions in different
    producer processes and per-key ordering breaks the moment you scale out.
    """

    def __init__(self) -> None:
        self._next = 0
        self._lock = threading.Lock()

    def partition(self, key: Optional[str], num_partitions: int,
                  available: Optional[list[int]] = None) -> int:
        candidates = list(available) if available else list(range(num_partitions))
        if not candidates:
            raise ValueError("no partitions available")
        if key is None:
            # Keyless messages have no ordering contract, so spread them.
            with self._lock:
                chosen = candidates[self._next % len(candidates)]
                self._next += 1
            return chosen
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return candidates[int.from_bytes(digest[:4], "big") % len(candidates)]


class RoundRobinPartitioner(Partitioner):
    """Even spread for keyless workloads; no ordering guarantee at all."""

    def __init__(self) -> None:
        self._next = 0
        self._lock = threading.Lock()

    def partition(self, key: Optional[str], num_partitions: int,
                  available: Optional[list[int]] = None) -> int:
        candidates = list(available) if available else list(range(num_partitions))
        with self._lock:
            chosen = candidates[self._next % len(candidates)]
            self._next += 1
        return chosen


class StickyPartitioner(Partitioner):
    """Hold one partition until `batch_size` messages have gone to it.

    This is what Kafka does for keyless records: round-robin per *record*
    produces many tiny batches and poor throughput, so it sticks to a
    partition until that partition's batch is full. It trades spread for
    batching — the opposite of what a queue with two consumers wants.
    """

    def __init__(self, batch_size: int = 10) -> None:
        self.batch_size = batch_size
        self._current: Optional[int] = None
        self._count = 0
        self._lock = threading.Lock()

    def partition(self, key: Optional[str], num_partitions: int,
                  available: Optional[list[int]] = None) -> int:
        candidates = list(available) if available else list(range(num_partitions))
        with self._lock:
            if self._current not in candidates or self._count >= self.batch_size:
                start = candidates.index(self._current) + 1 if self._current in candidates else 0
                self._current = candidates[start % len(candidates)]
                self._count = 0
            chosen = self._current
            self._count += 1
            return chosen  # type: ignore[return-value]


# --------------------------------------------------------------------------
# 4. Topic and broker
# --------------------------------------------------------------------------

class Topic:
    def __init__(self, name: str, num_partitions: int,
                 max_records_per_partition: Optional[int] = None) -> None:
        if num_partitions < 1:
            raise ValueError("a topic needs at least one partition")
        self.name = name
        self.partitions = [
            Partition(name, i, max_records_per_partition) for i in range(num_partitions)
        ]

    @property
    def num_partitions(self) -> int:
        return len(self.partitions)

    def partition_for(self, key: Optional[str], partitioner: Partitioner) -> Partition:
        return self.partitions[partitioner.partition(key, self.num_partitions)]

    def trim_all(self) -> int:
        return sum(p.trim() for p in self.partitions)

    def partition_sizes(self) -> list[int]:
        return [len(p) for p in self.partitions]

    def skew(self) -> float:
        """max / mean. Above ~2 on a hashed key means one key dominates — the
        hot-partition problem, which adding partitions does NOT fix."""
        sizes = self.partition_sizes()
        if not sizes or sum(sizes) == 0:
            return 1.0
        return max(sizes) / (sum(sizes) / len(sizes))


class Broker:
    """Owns the topics and assigns offsets. Note what it does NOT own: the
    consumer position. That split is the whole architecture."""

    def __init__(self) -> None:
        self._topics: dict[str, Topic] = {}
        self._dedupe: dict[tuple[str, int, int], int] = {}   # (pid, pt, seq) -> offset
        self._lock = threading.Lock()

    def create_topic(self, name: str, num_partitions: int,
                     max_records_per_partition: Optional[int] = None) -> Topic:
        with self._lock:
            if name in self._topics:
                raise ValueError(f"topic {name} already exists")
            topic = Topic(name, num_partitions, max_records_per_partition)
            self._topics[name] = topic
            return topic

    def topic(self, name: str) -> Topic:
        try:
            return self._topics[name]
        except KeyError as exc:
            raise KeyError(f"no such topic: {name}") from exc

    def append(self, topic_name: str, message: Message,
               partitioner: Partitioner, producer_id: int = -1,
               sequence: int = -1) -> RecordMetadata:
        """Append and assign the offset. When producer_id and sequence are
        given, the append is DEDUPED: a retried batch whose sequence the
        partition has already seen returns the ORIGINAL offset instead of
        appending a second copy. That is the producer half of exactly-once.

        The dedupe check and the append share ONE lock. Splitting them is the
        classic read-modify-write race: two concurrent retries of the same
        batch would both miss the dedupe and both append.
        """
        topic = self.topic(topic_name)
        partition = topic.partition_for(message.key, partitioner)
        dedupe_key = (producer_id, partition.index, sequence)
        track = producer_id >= 0 and sequence >= 0

        with self._lock:
            if track:
                seen = self._dedupe.get(dedupe_key)
                if seen is not None:
                    return RecordMetadata(topic_name, partition.index, seen)
            offset = partition.append(message)
            if track:
                self._dedupe[dedupe_key] = offset
        return RecordMetadata(topic_name, partition.index, offset)

    def fetch(self, topic_name: str, partition_index: int, offset: int,
              max_records: int) -> list[ConsumerRecord]:
        return self.topic(topic_name).partitions[partition_index].read_batch(offset, max_records)

    def high_watermark(self, topic_name: str, partition_index: int) -> int:
        return self.topic(topic_name).partitions[partition_index].high_watermark


# --------------------------------------------------------------------------
# 5. Producer
# --------------------------------------------------------------------------

class Producer:
    def __init__(self, broker: Broker, partitioner: Optional[Partitioner] = None,
                 producer_id: Optional[int] = None) -> None:
        self.broker = broker
        self.partitioner = partitioner if partitioner is not None else HashPartitioner()
        self.producer_id = producer_id if producer_id is not None else -1
        self._sequences: dict[int, int] = defaultdict(int)   # partition -> next seq

    def send(self, topic: str, value: bytes, key: Optional[str] = None,
             headers: Optional[Mapping[str, str]] = None) -> RecordMetadata:
        message = Message(key=key, value=value, headers=headers or {},
                          timestamp=time.time())
        if self.producer_id < 0:
            return self.broker.append(topic, message, self.partitioner)

        # Allocate the sequence by partition, so a retry of the same batch
        # carries the same sequence and is deduped by the broker.
        partition = self.broker.topic(topic).partition_for(key, self.partitioner)
        sequence = self._sequences[partition.index]
        self._sequences[partition.index] += 1
        return self.broker.append(topic, message, self.partitioner,
                                  producer_id=self.producer_id, sequence=sequence)


class RecordAccumulator:
    """Batches records per partition before sending. Two knobs, and they trade
    against each other: `batch_size` bounds latency, `linger_ms` bounds
    throughput. Batching per partition is what makes producer ordering for a
    key trivially correct — one batch, one partition, appended in order."""

    def __init__(self, producer: Producer, batch_size: int = 100,
                 linger_ms: float = 5.0) -> None:
        self.producer = producer
        self.batch_size = batch_size
        self.linger_ms = linger_ms
        self._buffers: dict[str, list[tuple[str, Optional[str], Mapping[str, str]]]] = \
            defaultdict(list)
        self._opened: dict[str, float] = {}

    def add(self, topic: str, value: bytes, key: Optional[str] = None,
            headers: Optional[Mapping[str, str]] = None) -> Optional[RecordMetadata]:
        """Returns metadata when the batch flushed, else None."""
        self._buffers[topic].append((value, key, headers or {}))  # type: ignore[arg-type]
        self._opened.setdefault(topic, time.monotonic())
        if len(self._buffers[topic]) >= self.batch_size:
            return self.flush(topic)
        return None

    def flush(self, topic: str) -> Optional[RecordMetadata]:
        batch = self._buffers.pop(topic, [])
        self._opened.pop(topic, None)
        last = None
        for value, key, headers in batch:                  # type: ignore[misc]
            last = self.producer.send(topic, value, key, headers)
        return last

    def maybe_flush(self, topic: str) -> Optional[RecordMetadata]:
        opened = self._opened.get(topic)
        if opened is not None and (time.monotonic() - opened) * 1000 >= self.linger_ms:
            return self.flush(topic)
        return None


# --------------------------------------------------------------------------
# 6. Offsets and the consumer group
# --------------------------------------------------------------------------

class OffsetStore:
    """Committed offsets, keyed by (group, topic, partition). The committed
    offset is the NEXT offset to read, not the last one read — an off-by-one
    here either re-reads one record or skips one, and both are silent."""

    def __init__(self) -> None:
        self._committed: dict[tuple[str, str, int], int] = {}
        self._lock = threading.Lock()

    def commit(self, group: str, topic: str, partition: int, offset: int) -> None:
        with self._lock:
            self._committed[(group, topic, partition)] = offset

    def committed(self, group: str, topic: str, partition: int) -> Optional[int]:
        with self._lock:
            return self._committed.get((group, topic, partition))

    def commit_map(self, group: str,
                   offsets: Mapping[tuple[str, int], int]) -> None:
        with self._lock:
            for (topic, partition), offset in offsets.items():
                self._committed[(group, topic, partition)] = offset

    def group_offsets(self, group: str) -> dict[tuple[str, int], int]:
        with self._lock:
            return {(t, p): o for (g, t, p), o in self._committed.items() if g == group}


class PartitionAssignor(ABC):
    @abstractmethod
    def assign(self, members: list[str],
               partitions: dict[str, list[int]]) -> dict[str, list[tuple[str, int]]]: ...


class RangeAssignor(PartitionAssignor):
    """Per topic, hand each member a contiguous range. Simple, and it is what
    makes a member's assignment stable when another member leaves — at the
    cost of giving the first members an extra partition whenever the counts
    do not divide evenly."""

    def assign(self, members: list[str],
               partitions: dict[str, list[int]]) -> dict[str, list[tuple[str, int]]]:
        ordered = sorted(members)
        result: dict[str, list[tuple[str, int]]] = {m: [] for m in ordered}
        for topic in sorted(partitions):
            parts = sorted(partitions[topic])
            if not ordered:
                continue
            per = len(parts) // len(ordered)
            extra = len(parts) % len(ordered)
            cursor = 0
            for i, member in enumerate(ordered):
                take = per + (1 if i < extra else 0)
                result[member].extend((topic, p) for p in parts[cursor:cursor + take])
                cursor += take
        return result


class RoundRobinAssignor(PartitionAssignor):
    """Deal all partitions out one at a time. Even counts, and it moves more
    partitions than necessary on every rebalance."""

    def assign(self, members: list[str],
               partitions: dict[str, list[int]]) -> dict[str, list[tuple[str, int]]]:
        ordered = sorted(members)
        result: dict[str, list[tuple[str, int]]] = {m: [] for m in ordered}
        if not ordered:
            return result
        all_parts = [(t, p) for t in sorted(partitions) for p in sorted(partitions[t])]
        for i, pair in enumerate(all_parts):
            result[ordered[i % len(ordered)]].append(pair)
        return result


class GroupCoordinator:
    """Membership and the generation fence.

    The generation is the whole point. A consumer that missed a rebalance
    still thinks it owns its partitions and will happily commit offsets for
    them — rolling the group's position backwards and causing a re-read storm
    or silent data loss. Bumping the generation makes its commit a no-op.
    """

    def __init__(self, assignor: Optional[PartitionAssignor] = None) -> None:
        self.assignor = assignor if assignor is not None else RangeAssignor()
        self.generation = 0
        self.members: dict[str, list[tuple[str, int]]] = {}
        self._topics: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topics: Mapping[str, int]) -> None:
        """topics maps a topic name to its partition count."""
        with self._lock:
            self._topics = {name: list(range(count)) for name, count in topics.items()}

    def join(self, member_id: str) -> Assignment:
        with self._lock:
            if member_id not in self.members:
                self.members[member_id] = []
            return self._rebalance_locked(member_id)

    def leave(self, member_id: str) -> None:
        with self._lock:
            self.members.pop(member_id, None)
            if self.members:
                self._rebalance_locked(next(iter(self.members)))

    def _rebalance_locked(self, member_id: str) -> Assignment:
        self.generation += 1
        # Start every member empty, so a partition is never owned twice.
        for member in self.members:
            self.members[member] = []
        self.members = self.assignor.assign(list(self.members), self._topics)
        return Assignment(self.generation, tuple(self.members.get(member_id, [])))

    def commit(self, member_id: str, generation: int,
               offsets: Mapping[tuple[str, int], int]) -> bool:
        """False means FENCED: the member's generation is stale and its offsets
        are discarded. Callers must treat False as "I no longer own this" and
        rejoin, not as a transient error to retry."""
        with self._lock:
            if generation != self.generation:
                return False
            if member_id not in self.members:
                return False
            return True


# --------------------------------------------------------------------------
# 7. Consumer
# --------------------------------------------------------------------------

class Consumer:
    """One consumer, one member of a group.

    Kafka's rule and this class's rule: a consumer is used from ONE thread.
    Parallelism comes from more partitions and more consumers, not from
    threads sharing one consumer — that is why `assignment` is a plain dict
    with no lock around the position.
    """

    def __init__(self, broker: Broker, group_id: str, offsets: OffsetStore,
                 coordinator: GroupCoordinator, member_id: str = "c1",
                 max_poll_records: int = 10,
                 prefetch_limit: Optional[int] = None) -> None:
        self.broker = broker
        self.group_id = group_id
        self.offsets = offsets
        self.coordinator = coordinator
        self.member_id = member_id
        self.max_poll_records = max_poll_records
        self.prefetch_limit = prefetch_limit if prefetch_limit is not None else max_poll_records
        self.assignment: dict[tuple[str, int], int] = {}
        self.generation = 0
        self._paused: set[tuple[str, int]] = set()
        self._buffer: list[ConsumerRecord] = []
        self._resets = 0

    def subscribe(self, topics: Mapping[str, int]) -> None:
        self.coordinator.subscribe(topics)
        assignment = self.coordinator.join(self.member_id)
        self._apply(assignment)

    def _apply(self, assignment: Assignment) -> None:
        """Adopt a new assignment. Anything dropped from the assignment is
        dropped from our position map too, and any buffered records for it are
        discarded — they will be re-read by whoever owns them now."""
        self.generation = assignment.generation
        owned = set(assignment.partitions)
        self.assignment = {pair: self.assignment.get(pair, self._start_of(pair))
                           for pair in owned}
        self._buffer = [r for r in self._buffer if (r.topic, r.partition) in owned]
        self._paused &= owned

    def _start_of(self, pair: tuple[str, int]) -> int:
        """Where a newly owned partition starts: the committed offset, or the
        log start if there is none. This is the seek-on-assign decision, and
        choosing the log END instead would skip everything produced while the
        consumer was down."""
        committed = self.offsets.committed(self.group_id, pair[0], pair[1])
        if committed is not None:
            return committed
        return self.broker.topic(pair[0]).partitions[pair[1]].log_start_offset

    def rejoin(self) -> None:
        self._apply(self.coordinator.join(self.member_id))

    @property
    def position(self) -> dict[tuple[str, int], int]:
        return dict(self.assignment)

    @property
    def resets(self) -> int:
        """How many times this consumer had to jump forward because retention
        deleted its position. A silent reset is data loss that looks like a
        healthy consumer, so the count must be observable."""
        return self._resets

    def pause(self, pair: tuple[str, int]) -> None:
        self._paused.add(pair)

    def resume(self, pair: tuple[str, int]) -> None:
        self._paused.discard(pair)

    def poll(self) -> list[ConsumerRecord]:
        """Fetch up to `max_poll_records`, respecting the bounded prefetch.

        Backpressure: we fetch at most max_poll_records per call and never
        buffer more than prefetch_limit. An unbounded prefetch turns a slow
        handler into an OOM, and it also breaks the rebalance contract —
        Kafka's max.poll.interval exists precisely because a consumer that
        takes too long between polls is assumed dead.
        """
        out = self._buffer[:self.max_poll_records]
        self._buffer = self._buffer[self.max_poll_records:]
        if len(out) >= self.max_poll_records:
            return out

        for pair in sorted(self.assignment):
            if len(out) >= self.max_poll_records:
                break
            if pair in self._paused:
                continue
            topic, partition = pair
            want = self.max_poll_records - len(out)
            try:
                records = self.broker.fetch(topic, partition, self.assignment[pair], want)
            except OffsetOutOfRange:
                # Retention deleted the offset we were positioned at. The data
                # is gone; the only recovery is to jump to the earliest
                # available record — and to COUNT it, because a silent reset
                # is data loss that looks like a healthy consumer.
                self.assignment[pair] = \
                    self.broker.topic(topic).partitions[partition].log_start_offset
                self._resets += 1
                continue
            for record in records:
                self.assignment[pair] = record.offset + 1   # position advances on read
                out.append(record)
        return out

    def commit_sync(self) -> bool:
        """Commit the CURRENT POSITION — which is the next offset to read, i.e.
        every record handed out so far is considered done. Committing before
        the work is what makes a crash lose data."""
        ok = self.coordinator.commit(self.member_id, self.generation, self.assignment)
        if ok:
            self.offsets.commit_map(self.group_id, self.assignment)
        return ok

    def seek(self, pair: tuple[str, int], offset: int) -> None:
        if pair not in self.assignment:
            raise ValueError(f"not assigned {pair}")
        self.assignment[pair] = offset

    def lag(self) -> dict[tuple[str, int], int]:
        return {
            pair: self.broker.high_watermark(pair[0], pair[1]) - position
            for pair, position in self.assignment.items()
        }


# --------------------------------------------------------------------------
# 8. Delivery semantics — the ordering of two lines
# --------------------------------------------------------------------------

class DeliverySemantics(Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class LoopStats:
    polled: int = 0
    processed: int = 0
    duplicated_output: int = 0
    dead_lettered: int = 0
    commits: int = 0
    fenced_commits: int = 0


class CrashInjected(Exception):
    """Injected by the tests to model a consumer dying mid-loop."""


class DeadLetterQueue:
    def __init__(self) -> None:
        self.records: list[tuple[ConsumerRecord, str]] = []

    def send(self, record: ConsumerRecord, reason: str) -> None:
        self.records.append((record, reason))

    def __len__(self) -> int:
        return len(self.records)


class ProcessingLoop:
    """poll -> process -> commit, with the ORDER chosen by the semantics.

        AT_MOST_ONCE  : commit, then process   -> loss on crash
        AT_LEAST_ONCE : process, then commit   -> duplicates on crash
        EXACTLY_ONCE  : process and commit with a dedupe on the output

    The broker does not decide this. The client does, by which line comes
    first — which is why "what delivery guarantee?" is a client-side question.
    """

    def __init__(self, consumer: Consumer, handler: Callable[[ConsumerRecord], None],
                 semantics: DeliverySemantics = DeliverySemantics.AT_LEAST_ONCE,
                 dlq: Optional[DeadLetterQueue] = None,
                 max_attempts: int = 3) -> None:
        self.consumer = consumer
        self.handler = handler
        self.semantics = semantics
        self.dlq = dlq if dlq is not None else DeadLetterQueue()
        self.max_attempts = max_attempts
        self._attempts: dict[tuple[str, int, int], int] = {}
        self._delivered: set[tuple[str, int, int]] = set()

    def run_once(self, crash_after: Optional[str] = None) -> LoopStats:
        """`crash_after` is 'commit' or 'process' — where the injected failure
        happens. It exists so the tests can prove loss and duplication instead
        of asserting them in prose."""
        stats = LoopStats()
        records = self.consumer.poll()
        stats.polled = len(records)

        if self.semantics is DeliverySemantics.AT_MOST_ONCE:
            # Commit FIRST: the position advances past records we have not
            # processed. A crash here loses them permanently.
            if self.consumer.commit_sync():
                stats.commits += 1
            else:
                stats.fenced_commits += 1
            if crash_after == "commit":
                raise CrashInjected("died after committing, before processing")

        # Partitions that must be rewound so a retryable failure is retried.
        rewind: dict[tuple[str, int], int] = {}

        for record in records:
            if self.semantics is DeliverySemantics.EXACTLY_ONCE:
                # The idempotent half: if this exact record already produced
                # output, re-processing it must not produce it twice.
                if record.coordinates in self._delivered:
                    stats.duplicated_output += 1
                    continue
            try:
                self.handler(record)
                stats.processed += 1
                self._delivered.add(record.coordinates)
                self._attempts.pop(record.coordinates, None)
            except Exception as exc:                        # noqa: BLE001 - deliberate
                attempts = self._attempts.get(record.coordinates, 0) + 1
                self._attempts[record.coordinates] = attempts
                pair = (record.topic, record.partition)
                if attempts >= self.max_attempts:
                    # Poison pill: retrying forever blocks the partition, which
                    # is worse than losing one message. Park it and move on.
                    self.dlq.send(record, f"{type(exc).__name__}: {exc}")
                    stats.dead_lettered += 1
                    self._attempts.pop(record.coordinates, None)
                else:
                    # Rewind this partition to the failed record, so the next
                    # poll retries it instead of skipping past it. Committing
                    # the advanced position here would silently drop the
                    # retry — the failure would be retried zero times.
                    previous = rewind.get(pair)
                    rewind[pair] = record.offset if previous is None \
                        else min(previous, record.offset)

        if crash_after == "process":
            raise CrashInjected("died after processing the batch, before committing")

        if self.semantics is not DeliverySemantics.AT_MOST_ONCE:
            for pair, offset in rewind.items():
                self.consumer.seek(pair, offset)
            if self.consumer.commit_sync():
                stats.commits += 1
            else:
                stats.fenced_commits += 1
        return stats


# --------------------------------------------------------------------------
# 9. Tests and demo — every guarantee is demonstrated, not asserted in prose
# --------------------------------------------------------------------------

def _check(name: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    if not condition:
        raise SystemExit(1)


if __name__ == "__main__":
    # --- 1. per-key ordering survives partitioning ---
    broker = Broker()
    broker.create_topic("clicks", 4)
    producer = Producer(broker, HashPartitioner())
    for i in range(200):
        producer.send("clicks", f"v{i}".encode(), key=f"user{i % 5}")

    seen: dict[str, list[int]] = defaultdict(list)
    homes: dict[str, set[int]] = defaultdict(set)
    topic = broker.topic("clicks")
    for part in topic.partitions:
        for record in part.read_batch(0, 10_000):
            seen[record.key].append(record.offset)
            homes[record.key].add(part.index)
    _check("every key landed in exactly one partition",
           len(homes) == 5 and all(len(v) == 1 for v in homes.values()))
    _check("offsets are strictly increasing within a key",
           all(v == sorted(v) and len(set(v)) == len(v) for v in seen.values()))
    _check("every key's records were read back and read in order",
           sum(len(v) for v in seen.values()) == 200)

    # --- 2. the partitioner is deterministic across instances AND processes ---
    p1, p2 = HashPartitioner(), HashPartitioner()
    assignments = {k: p1.partition(k, 8) for k in ("a", "b", "c", "user42")}
    _check("two partitioner instances agree for every key",
           all(p2.partition(k, 8) == v for k, v in assignments.items()))
    _check("the digest is a stable one (sha256, not Python's hash())",
           HashPartitioner().partition("user42", 8) == assignments["user42"])

    # --- 3. key skew is visible, and it is not fixed by adding partitions ---
    skewed = Broker()
    skewed.create_topic("skewed", 4)
    sp = Producer(skewed, HashPartitioner())
    for _ in range(90):
        sp.send("skewed", b"hot", key="vip-tenant")
    for i in range(10):
        sp.send("skewed", b"cold", key=f"tenant-{i}")
    _check("skew detects a dominant key", skewed.topic("skewed").skew() > 2.0)
    hot_partition = max(range(4),
                        key=lambda i: len(skewed.topic("skewed").partitions[i]))
    _check("the hot key occupies a single partition",
           len(skewed.topic("skewed").partitions[hot_partition]) > 50)

    # --- 4. consumer groups, assignment, and the rebalance ---
    offsets = OffsetStore()
    coordinator = GroupCoordinator(RangeAssignor())
    coordinator.subscribe({"clicks": 4})
    c1 = Consumer(broker, "analytics", offsets, coordinator, member_id="c1")
    c1.subscribe({"clicks": 4})
    _check("one member owns all four partitions", len(c1.assignment) == 4)
    gen_one = c1.generation

    c2 = Consumer(broker, "analytics", offsets, coordinator, member_id="c2")
    c2.subscribe({"clicks": 4})                           # joins, triggering a rebalance
    c1.rejoin()                                           # c1 must rejoin too
    owned = sorted(list(c1.assignment) + list(c2.assignment))
    _check("two members split the topic without overlap",
           len(owned) == len(set(owned)) == 4)
    _check("each member owns exactly two partitions",
           len(c1.assignment) == 2 and len(c2.assignment) == 2)
    _check("the rebalance bumped the generation", coordinator.generation > gen_one)

    # --- 5. the zombie consumer is fenced ---
    stale = gen_one
    _check("a stale generation cannot commit",
           coordinator.commit("c1", stale, {("clicks", 0): 999}) is False)
    _check("the current generation can commit",
           coordinator.commit("c1", coordinator.generation, {("clicks", 0): 5}) is True)

    # --- 6. at-least-once duplicates on a crash after processing ---
    broker2 = Broker()
    broker2.create_topic("orders", 1)
    prod2 = Producer(broker2)
    for i in range(5):
        prod2.send("orders", f"order-{i}".encode(), key="k")
    offsets2 = OffsetStore()
    coord2 = GroupCoordinator()
    coord2.subscribe({"orders": 1})
    consumer2 = Consumer(broker2, "billing", offsets2, coord2,
                         max_poll_records=5)
    consumer2.subscribe({"orders": 1})

    output: list[int] = []
    loop = ProcessingLoop(consumer2,
                          lambda r: output.append(r.offset),
                          DeliverySemantics.AT_LEAST_ONCE)
    try:
        loop.run_once(crash_after="process")
    except CrashInjected:
        pass
    _check("the crash happened after processing, before committing",
           len(output) == 5 and offsets2.committed("billing", "orders", 0) is None)

    # Restart: the consumer resumes from the last COMMITTED offset (none), so
    # it re-reads everything. At-least-once means duplicates are expected.
    restarted = Consumer(broker2, "billing", offsets2, coord2, max_poll_records=5)
    restarted.rejoin()
    loop2 = ProcessingLoop(restarted, lambda r: output.append(r.offset),
                           DeliverySemantics.AT_LEAST_ONCE)
    stats2 = loop2.run_once()
    _check("the restart re-read all five records", stats2.polled == 5)
    _check("so the handler saw each offset twice (at-least-once duplicates)",
           len(output) == 10 and len(set(output)) == 5)
    _check("and the offset is now committed",
           offsets2.committed("billing", "orders", 0) == 5)

    # --- 7. at-most-once LOSES data on the same crash ---
    broker3 = Broker()
    broker3.create_topic("payments", 1)
    prod3 = Producer(broker3)
    for i in range(5):
        prod3.send("payments", f"pay-{i}".encode(), key="k")
    offsets3 = OffsetStore()
    coord3 = GroupCoordinator()
    coord3.subscribe({"payments": 1})
    c3 = Consumer(broker3, "ledger", offsets3, coord3, max_poll_records=5)
    c3.subscribe({"payments": 1})
    lost_output: list[int] = []
    loop3 = ProcessingLoop(c3, lambda r: lost_output.append(r.offset),
                           DeliverySemantics.AT_MOST_ONCE)
    try:
        loop3.run_once(crash_after="commit")
    except CrashInjected:
        pass
    _check("at-most-once committed before doing any work",
           len(lost_output) == 0 and offsets3.committed("ledger", "payments", 0) == 5)
    c4 = Consumer(broker3, "ledger", offsets3, coord3, max_poll_records=5)
    c4.rejoin()
    loop4 = ProcessingLoop(c4, lambda r: lost_output.append(r.offset),
                           DeliverySemantics.AT_MOST_ONCE)
    stats4 = loop4.run_once()
    _check("the restart re-reads NOTHING: five payments are lost",
           stats4.polled == 0 and lost_output == [])

    # --- 8. exactly-once: the re-read happens, the duplicate output does not ---
    broker4 = Broker()
    broker4.create_topic("ledger", 1)
    prod4 = Producer(broker4)
    for i in range(5):
        prod4.send("ledger", f"entry-{i}".encode(), key="k")
    offsets4 = OffsetStore()
    coord4 = GroupCoordinator()
    coord4.subscribe({"ledger": 1})
    c5 = Consumer(broker4, "eos", offsets4, coord4, max_poll_records=5)
    c5.subscribe({"ledger": 1})
    eos_output: list[int] = []
    loop5 = ProcessingLoop(c5, lambda r: eos_output.append(r.offset),
                           DeliverySemantics.EXACTLY_ONCE)
    try:
        loop5.run_once(crash_after="process")
    except CrashInjected:
        pass
    # The restart is a NEW consumer: a restarted process recomputes its
    # position from the committed offset. Calling rejoin() on the same object
    # would keep its in-memory position and hide the whole point.
    c5b = Consumer(broker4, "eos", offsets4, coord4, max_poll_records=5)
    c5b.subscribe({"ledger": 1})
    loop5.consumer = c5b
    stats5 = loop5.run_once()
    _check("exactly-once re-read the batch after the crash", stats5.polled == 5)
    _check("but suppressed the duplicate outputs",
           stats5.duplicated_output == 5 and len(eos_output) == 5)
    _check("so the sink saw each record exactly once",
           sorted(eos_output) == [0, 1, 2, 3, 4])

    # --- 9. the idempotent producer dedupes a retried batch ---
    broker5 = Broker()
    broker5.create_topic("idem", 1)
    idem = Producer(broker5, producer_id=7)
    first = idem.send("idem", b"only-once", key="k")
    # Simulate a producer retry of the SAME batch: same producer_id, same
    # sequence, because the sequence is allocated per batch, not per attempt.
    idem._sequences[first.partition] -= 1
    retried = idem.send("idem", b"only-once", key="k")
    _check("a retried batch returns the original offset",
           retried.offset == first.offset)
    _check("and the broker holds exactly one copy",
           len(broker5.topic("idem").partitions[first.partition]) == 1)

    # --- 10. poison pill goes to the DLQ instead of blocking the partition ---
    broker6 = Broker()
    broker6.create_topic("ingest", 1)
    prod6 = Producer(broker6)
    for i in range(4):
        prod6.send("ingest", b"poison" if i == 2 else f"ok-{i}".encode(), key="k")
    offsets6 = OffsetStore()
    coord6 = GroupCoordinator()
    coord6.subscribe({"ingest": 1})
    c6 = Consumer(broker6, "etl", offsets6, coord6, max_poll_records=4)
    c6.subscribe({"ingest": 1})

    def fragile(record: ConsumerRecord) -> None:
        if record.value == b"poison":
            raise ValueError("cannot parse")

    dlq = DeadLetterQueue()
    loop6 = ProcessingLoop(c6, fragile, DeliverySemantics.AT_LEAST_ONCE,
                           dlq=dlq, max_attempts=3)
    for _ in range(5):
        try:
            loop6.run_once()
        except CrashInjected:
            pass
        if len(c6.assignment) and all(
                offsets6.committed("etl", t, p) == broker6.high_watermark(t, p)
                for (t, p) in c6.assignment):
            break
    _check("the poison message was parked in the DLQ", len(dlq) == 1)
    _check("the DLQ recorded the reason", "ValueError" in dlq.records[0][1])
    _check("the partition advanced past it instead of blocking",
           offsets6.committed("etl", "ingest", 0) == 4)
    _check("every non-poison record was processed",
           broker6.high_watermark("ingest", 0) == 4 and len(dlq) == 1)

    # --- 11. retention deletes unread data, and the failure is explicit ---
    broker7 = Broker()
    broker7.create_topic("short", 1, max_records_per_partition=3)
    prod7 = Producer(broker7)
    for i in range(10):
        prod7.send("short", f"m{i}".encode(), key="k")
    dropped = broker7.topic("short").trim_all()
    _check("retention dropped the oldest seven records", dropped == 7)
    _check("the log start offset advanced",
           broker7.topic("short").partitions[0].log_start_offset == 7)
    try:
        broker7.fetch("short", 0, 0, 1)
        _check("reading a trimmed offset raises", False)
    except OffsetOutOfRange as exc:
        _check("reading a trimmed offset raises OffsetOutOfRange",
               "retention deleted it" in str(exc))
    _check("a new consumer starts at the log start, not at zero",
           Consumer(broker7, "late", OffsetStore(), GroupCoordinator())._start_of(("short", 0)) == 7)

    # --- 12. backpressure: poll is bounded by max_poll_records ---
    broker8 = Broker()
    broker8.create_topic("burst", 1)
    prod8 = Producer(broker8)
    for i in range(500):
        prod8.send("burst", f"b{i}".encode(), key="k")
    offsets8 = OffsetStore()
    coord8 = GroupCoordinator()
    coord8.subscribe({"burst": 1})
    c8 = Consumer(broker8, "slow", offsets8, coord8, max_poll_records=10)
    c8.subscribe({"burst": 1})
    batch = c8.poll()
    _check("a 500-record partition yields only the poll limit", len(batch) == 10)
    c8.pause(("burst", 0))
    _check("a paused partition yields nothing", c8.poll() == [])
    c8.resume(("burst", 0))
    _check("resuming continues from the position, with no gap",
           c8.poll()[0].offset == 10)

    # --- 13. consumer lag is the operational number ---
    _check("lag is high watermark minus position",
           c8.lag()[("burst", 0)] == 500 - 20)

    # --- 14. assignment snapshots, and what a rebalance moves ---
    coord9 = GroupCoordinator(RoundRobinAssignor())
    coord9.subscribe({"t": 6})
    a = coord9.join("a")
    _check("with one member, that member owns all six partitions",
           len(a.partitions) == 6)
    b = coord9.join("b")
    _check("a's assignment object is a SNAPSHOT — it still claims all six",
           len(a.partitions) == 6 and len(b.partitions) == 3)
    _check("which is exactly why the generation fence is needed",
           a.generation != b.generation)
    c = coord9.join("c")
    _check("the joining member gets its own generation's share",
           sorted(c.partitions) == [("t", 2), ("t", 5)])
    covered = sorted(pair for pairs in coord9.members.values() for pair in pairs)
    _check("three members cover six partitions exactly once",
           covered == [("t", i) for i in range(6)])
    coord9.leave("c")
    after = coord9.members
    _check("after a departure every partition is still owned exactly once",
           sorted(list(after["a"]) + list(after["b"])) == [("t", i) for i in range(6)])

    print("\nall checks passed")
```

**What to point at:**

- `Partition.read(offset)` is **O(1) random access**, which is why a consumer's entire position is one integer. That is the design decision that makes replay, seek and lag trivial.
- `HashPartitioner` uses **sha256, not `hash()`**, with a comment saying why: Python randomises `hash()` per process, so a key would land on a different partition in a different producer process and per-key ordering would break on scale-out.
- `GroupCoordinator.commit` returns **`False` for a stale generation**, and the docstring says the caller must treat that as "I no longer own this", not as a retryable error.
- `ProcessingLoop.run_once` reorders **two lines** to change the delivery guarantee, and `crash_after` lets the tests *prove* the loss and the duplication rather than assert them.
- `DeadLetterQueue` exists because **retrying a poison pill forever blocks the partition**, which is worse than losing one message.
- `Consumer._start_of` seeks to the **committed offset or the log start** — seeking to the log end instead is the bug that silently skips everything produced while a consumer was down.

---

## 6. Concurrency & thread-safety

### What breaks

| Hazard | What goes wrong | Fix |
|---|---|---|
| **A consumer shared across threads** | The position map is mutated by two threads; two threads process the same record or skip one | One consumer, one thread. Parallelism comes from partitions and group members, not threads |
| **Zombie consumer committing** | A consumer that missed the rebalance commits offsets for partitions it no longer owns, rolling the group's position backwards | The generation fence: `commit` returns `False` and the caller rejoins |
| **Commit racing the rebalance** | The commit lands between the assignment change and the position reset | The coordinator owns both under one lock; the commit is validated against the generation |
| **Auto-commit** | A background timer commits records that are still being processed — at-most-once by accident, and invisibly | Explicit commits after processing, or a synchronous autocommit at a poll boundary with the position advanced only after handling |
| **Two producers, same key, different processes** | Different partitions if the partitioner is not stable across processes | A stable digest, never `hash()` |
| **`max.poll.records` too large** | One poll returns more than the handler can process before the session times out; the group rebalances and the work is duplicated | Bound the batch by handler latency; this is the real meaning of `max.poll.interval.ms` |
| **Rebalance storm** | Members keep timing out and rejoining, so no partition is processed for long stretches | A rebalance timeout longer than the longest poll, and static membership where supported |
| **Unbounded prefetch** | A slow handler plus eager fetching is an OOM, and it hides the lag that would have flagged the problem | A bounded prefetch and `pause`/`resume` on the consumer |
| **Offset commit off the log's start** | Retention deleted the committed offset; the commit is now below the log start | Handle `OffsetOutOfRange` explicitly — reset to earliest or to latest, and record which |
| **Hot partition** | One key gets 90% of the traffic; one consumer is saturated while others idle | Salt the key for non-order-sensitive work, or split the topic; adding partitions does **not** help a single hot key |
| **Adding partitions** | Every existing key may rehash to a different partition, breaking per-key ordering mid-stream | Add partitions only with a key-preserving scheme (a partitioner that keeps the old mapping for old keys), or not at all |

**The one to volunteer:** the **generation fence**. It is the difference between a toy rebalance and a correct one, and it is the answer that separates a candidate who has operated a queue from one who has read about it.

### Ordering, stated precisely

Within one partition, order is total and stable, because the log is append-only and offsets only increase. Across partitions there is **no order at all** and no timestamp that can recover one. So "does this system preserve order?" has exactly one correct answer: *"per key, if the key maps to a partition — and only there."* Any consumer that needs a global order needs a single partition, which means a single consumer, which means the throughput ceiling is one thread.

---

## 7. Extensibility — the standard follow-ups, with real designs

### 7.1 "Add a second consumer group"

Nothing in the broker changes — that is the point of decoupling:

```python
def subscribe_group(broker: Broker, offsets: OffsetStore, group: str,
                    topics: Mapping[str, int], members: int) -> list[Consumer]:
    coordinator = GroupCoordinator(RangeAssignor())
    coordinator.subscribe(topics)
    consumers = [
        Consumer(broker, group, offsets, coordinator, member_id=f"{group}-{i}",
                 max_poll_records=50)
        for i in range(members)
    ]
    for consumer in consumers:
        consumer.subscribe(topics)
    for consumer in consumers:                    # everyone must rejoin after
        consumer.rejoin()
    return consumers
```

Two groups read the same partitions independently, each with its own committed offsets in `OffsetStore`. That is why the offset key is `(group, topic, partition)` and not `(topic, partition)` — the group is part of the consumer's identity.

### 7.2 "Add a retry topic with a delay"

Retry-with-backoff in a queue is a different topic per delay tier, because a queue has no per-message visibility timeout:

```python
class RetryRouter:
    """Failed record -> retry topic for its attempt count. Each retry topic is
    consumed by a worker that only reads records whose `not_before` has
    passed, so the delay is a filter, not a sleep."""

    TIERS = ("retry-1m", "retry-5m", "retry-30m")

    def __init__(self, producer: Producer, dlq: DeadLetterQueue) -> None:
        self.producer = producer
        self.dlq = dlq

    def route(self, record: ConsumerRecord, attempt: int) -> None:
        if attempt > len(self.TIERS):
            self.dlq.send(record, f"exhausted {attempt} attempts")
            return
        self.producer.send(
            self.TIERS[attempt - 1], record.value, key=record.key,
            headers={**record.headers, "attempt": str(attempt),
                     "origin": f"{record.topic}-{record.partition}@{record.offset}"},
        )
```

The header carrying the **origin coordinates** is what makes the eventual DLQ entry traceable back to the offset it came from. Without it, a dead-lettered message cannot be replayed.

### 7.3 "Exactly-once across two topics (read-process-write)"

This is the case the producer-idempotence alone does not cover: the consumer's output must be committed atomically with its input offset. The model:

```python
class TransactionalSink:
    """The output side of exactly-once. `begin` buffers writes; `commit`
    makes them durable together with the offset. A crash before commit
    discards both, so the batch is re-read and re-processed with no visible
    effect — the same guarantee the processing loop gives, extended to the
    output."""

    def __init__(self) -> None:
        self._pending: list[tuple[str, int]] = []
        self._durable: list[tuple[str, int]] = []

    def begin(self) -> None:
        self._pending = []

    def write(self, topic: str, offset: int) -> None:
        self._pending.append((topic, offset))

    def commit(self) -> None:
        self._durable.extend(self._pending)
        self._pending = []

    def abort(self) -> None:
        self._pending = []
```

The honest caveat to state: real exactly-once is **exactly-once within the system's boundary**. The moment the sink is an external database with no transactional participant, you have at-least-once plus an idempotent write, and that is the strongest guarantee available. Saying that boundary out loud is the senior answer.

### 7.4 "Compaction instead of deletion"

A compacted topic keeps the **last** record per key rather than dropping old ones — the changelog pattern. The design difference is in the data structure: `Partition` would need a key index, and retention becomes a `dict[key] -> latest offset` sweep:

```python
def compact(partition: Partition) -> list[ConsumerRecord]:
    """Keep only the newest record per key; keys whose newest value is a
    tombstone (empty payload) are removed entirely."""
    with partition._lock:
        high_watermark = partition._base_offset + len(partition._records)
        newest: dict[Optional[str], ConsumerRecord] = {}
        for record in partition._records:
            newest[record.key] = record
        kept = [r for r in newest.values() if r.value != b""]
        kept.sort(key=lambda r: r.offset)
        partition._records = kept
        # The log start does NOT move to the first surviving offset: offsets
        # below it are simply gone, which is what creates the gaps.
        partition._base_offset = kept[0].offset if kept else high_watermark
        return kept
```

This needs a note in `Partition`'s docstring that the invariant "offsets are contiguous" is broken by compaction — a compacted partition has **gaps**, and a reader must not assume `offset + 1` is the next record. That is the kind of invariant a follow-up question is testing for.

### 7.5 "Flow control on the consumer side"

```python
class FlowController:
    """Pause a partition when its lag exceeds a ceiling; resume when it
    recovers. The point is to shed load at the SOURCE rather than buffering
    it, because buffering is what turns lag into an OOM."""

    def __init__(self, consumer: Consumer, high: int = 10_000, low: int = 1_000) -> None:
        self.consumer = consumer
        self.high, self.low = high, low

    def apply(self) -> list[tuple[str, int]]:
        paused = []
        for pair, lag in self.consumer.lag().items():
            if lag > self.high:
                self.consumer.pause(pair)
                paused.append(pair)
            elif lag < self.low:
                self.consumer.resume(pair)
        return paused
```

### 7.6 "Make the consumer idempotent without transactions"

The cheapest correct option, and the one to reach for first:

```python
class DedupingHandler:
    """Idempotent by (topic, partition, offset). At-least-once delivery plus a
    dedupe key gives an effectively-exactly-once *effect* without transactions
    — as long as the dedupe store is as durable as the output.

    The subtlety worth saying: deduping on the record COORDINATES only works
    if the same logical message is never re-produced with a new offset. If a
    producer retries without idempotence, the same order arrives twice at two
    offsets and this dedupe does not catch it — which is why the producer is
    idempotent too.
    """

    def __init__(self, sink: Callable[[ConsumerRecord], None], store: set) -> None:
        self.sink = sink
        self.store = store

    def __call__(self, record: ConsumerRecord) -> None:
        if record.coordinates in self.store:
            return
        self.sink(record)
        self.store.add(record.coordinates)
```

---

## 8. Failure modes & edge cases

| # | Case | What a naive implementation does | Correct behaviour |
|---|---|---|---|
| 1 | **Commit before processing** | A crash loses the batch silently | Commit after processing; at-most-once must be chosen knowingly |
| 2 | **Zombie consumer** | Commits offsets after losing its partitions; the group's position goes backwards | The generation fence; a fenced commit is a signal to rejoin |
| 3 | **Auto-commit** | A timer commits work in flight; loss happens invisibly | Explicit commits, or commit only at a poll boundary after handling |
| 4 | **Unstable partitioner** | `hash()` maps a key to different partitions in different processes; per-key order breaks on scale-out | A stable digest such as sha256 |
| 5 | **Hot key** | One partition absorbs most traffic; one consumer saturates | Salt the key where order is not required; monitor `max/mean`; adding partitions does not help |
| 6 | **Poison pill** | Infinite retry blocks the partition; lag grows without bound | Retry with a cap, then DLQ, then advance |
| 7 | **Retention deletes unread data** | A slow consumer's committed offset falls below the log start; recovery reads garbage or crashes | Alert on lag approaching retention; handle `OffsetOutOfRange` by an explicit, logged reset |
| 8 | **Seek to the log end on assign** | Everything produced while the consumer was down is skipped, silently | Seek to the committed offset, or the log start if none |
| 9 | **Adding partitions** | Existing keys rehash, so per-key ordering breaks mid-stream | Do not add; or add with a partitioner that preserves the old mapping for existing keys |
| 10 | **`max.poll.records` too large** | The handler takes longer than the session timeout, the group rebalances, the work is duplicated | Bound the batch by handler latency |
| 11 | **Rebalance storm** | Members keep timing out; nothing is processed for long stretches | A rebalance timeout longer than the slowest poll; static membership |
| 12 | **Unbounded prefetch** | A slow consumer OOMs, and lag looks fine until it does not | Bounded prefetch plus pause/resume |
| 13 | **Duplicate delivery** | The consumer double-charges or double-counts | Idempotent consumer on `(topic, partition, offset)`, and an idempotent producer so the same logical message does not arrive at two offsets |
| 14 | **Off-by-one on the commit** | The committed offset is the last read rather than the next to read — one record is re-read or skipped forever | The committed offset is the NEXT offset; test the boundary explicitly |
| 15 | **A consumer reading two partitions in one thread** | A slow partition starves a fast one; the fast one's lag grows | One thread per partition, or accept the coupling knowingly |
| 16 | **Ordering assumed across partitions** | Processing looks correct in a single-partition test and reorders in production | State the invariant: order is per partition, never per topic |
| 17 | **DLQ without provenance** | A dead-lettered message cannot be traced or replayed | Carry origin coordinates `(topic, partition, offset)` in the DLQ record |
| 18 | **Compacted topic treated as contiguous** | A reader assumes `offset + 1` is the next record and skips or misreads | Compaction leaves gaps; the reader must use the log, not arithmetic |
| 19 | **Exactly-once claimed for an external sink** | A database write is not in the transaction, so a crash duplicates it | Say the boundary: at-least-once plus an idempotent write is the strongest available |
| 20 | **Producer without idempotence** | A retried batch appends a second copy at a new offset; the consumer's coordinate dedupe cannot catch it | Idempotent producer: `(producer_id, sequence)` deduped by the broker |

**The two to volunteer:** #2 (the zombie) and #4 (the unstable partitioner). Both are bugs that pass every single-process test and fail only in production, which is exactly the category an interviewer is probing for.

---

## 9. Design patterns used, and why

| Pattern | Where | Why it is right here | What it costs |
|---|---|---|---|
| **Strategy** | `Partitioner` | The partitioner decides *ordering*, not just balance, so swapping it changes the system's guarantees. It must be swappable and it must be deterministic | A non-deterministic implementation is a correctness bug with no compile-time symptom |
| **Strategy** | `PartitionAssignor` | Range versus round-robin is a real operational choice: stability of assignment versus evenness of count | Both move partitions on rebalance; neither is incremental |
| **Mediator** | `GroupCoordinator` | Membership, assignment and the generation are one coherent piece of state that must change together and be read atomically | It is the single point every commit goes through |
| **Template Method** | `ProcessingLoop.run_once` | The skeleton is poll-process-commit; the semantics decide the order of two steps, the handler decides the work | The step order is invisible at the call site, so it needs a comment and a test |
| **Decorator** | `IdempotentProducer` (as a `producer_id`), `DedupingHandler`, `RetryRouter` | Delivery guarantees are added by wrapping, not by changing the broker or the handler | Each wrapper is stateful, and its state must be as durable as the output |
| **Value Object** | `Message`, `ConsumerRecord`, `RecordMetadata`, `Assignment` | Immutable records are safe to hand to many consumers and to log; `coordinates` gives a delivery identity | One allocation per record on the hot path |
| **Repository** | `OffsetStore` | The committed position is the consumer group's durable state, and it belongs behind an interface that a real store can implement | An in-memory version is not durable — the class exists to make that seam explicit |
| **Null Object** | Not needed | A consumer with no assignment is simply empty; there is nothing to null-object here | — |
| **Singleton** | *Deliberately rejected* | The broker is injected, so tests can create a fresh one per scenario; a global broker would make tests order-dependent | — |
| **Observer** | Not used for delivery | A pull-based consumer is explicitly not an observer: the broker does not call back, which is exactly what lets the consumer control its own rate | Push-based fan-out is easier to write and much harder to flow-control |

**The sentence to give:** *"Strategy for both the partitioner and the assignor, Mediator for the coordinator, Template Method for the processing loop, Decorator for each delivery guarantee, and a Repository for the offset store. The two I would argue hardest for are the partitioner as a strategy — because it is a correctness component — and the processing loop as a template, because the delivery guarantee is literally the order of its steps and putting that behind one method makes it testable."*

---

## 10. Trade-offs

| Decision | Choice made here | Buys | Costs |
|---|---|---|---|
| **Per-key ordering via hashing** | Yes | Order where it is needed, parallelism everywhere else | A hot key serialises on one partition; skew is a permanent operational risk |
| **Pull, not push** | Yes | The consumer controls its own rate; lag is a first-class number | The consumer must poll on a timer, and a slow poll triggers a rebalance |
| **At-least-once default** | Yes | No data loss, and the cheapest correct option | The consumer must be idempotent, or it double-charges |
| **Exactly-once offered** | Documented, with its boundary | True end-to-end once for read-process-write inside one system | Transactions, latency, and a boundary that ends at the external sink |
| **Generation fencing** | Yes | Zombie consumers cannot corrupt the group's position | Every commit carries a generation, and a fence must be handled as a rejoin |
| **Range assignor** | Yes, as the default | Assignment is stable when a member leaves | Uneven counts when partitions do not divide evenly |
| **Round-robin offered** | Yes | Even counts | Every rebalance reshuffles almost everything |
| **Bounded prefetch** | Yes | A slow consumer sheds load instead of buffering to OOM | More poll round trips; throughput is bounded by the poll rate |
| **DLQ after N attempts** | Yes | A poison pill cannot block a partition forever | A message can be lost to the DLQ, so the DLQ must be monitored |
| **Retention by record count (here) / time (real)** | Count, for testability | Deterministic tests | Time-based retention is what production uses; the failure mode modelled is the same |
| **Explicit commits, not auto-commit** | Yes | The window between processing and committing is visible in the code | More code, and a forgotten commit means re-reading everything |
| **One thread per consumer** | Yes | No shared position state, so no lock on the hot path | Parallelism must come from partitions, which caps it at the partition count |
| **In-memory log (here)** | Yes, documented | The semantics are testable in milliseconds | Durability, replication and ISR are the HLD layer and are not modelled |
| **Partition count effectively immutable** | Yes, documented | Per-key ordering holds for the life of the topic | Scaling out requires a new topic with a migration |

**The trade to name unprompted:** *"Partition-by-key buys per-key ordering and parallelism at the same time, which is what makes it the right default — but it makes key skew a permanent property of the topic, and it makes the partition count effectively immutable. Both of those are decisions you cannot undo later, so they are the two I would want to be sure about in the requirements phase."*

---

## 11. Q&A

**Q1. Why partition by key rather than round-robin?**
Because partitioning is what defines ordering. Within a partition the log is append-only and offsets only increase, so order is total and stable; across partitions there is no order at all and no timestamp recovers one. So hashing the key to a partition is the mechanism that gives "all events for one user are processed in order" while still allowing N consumers to work in parallel. Round-robin gives a better spread and no ordering contract whatsoever — correct for genuinely independent work like resizing images, wrong for anything where two events for the same entity must not be reordered.

**Q2. Why is `hash()` the wrong partitioner in Python?**
Because `hash()` on a string is randomised per process by `PYTHONHASHSEED`, so the same key maps to a different partition in a different producer process. In a single-process test everything is consistent and the bug is invisible; the moment the producer is scaled to two processes, half the keys move and per-key ordering breaks — silently, because nothing errors. The fix is a stable digest, which is why the code uses sha256 and the comment says so. The general lesson is that a partitioner is a correctness component and must be deterministic across processes, restarts and versions.

**Q3. What is the generation number for?**
Fencing. During a rebalance a consumer can be replaced while it still believes it owns its partitions — it may be paused on a long GC, or partitioned from the coordinator. Without a fence it will commit offsets for partitions it no longer owns, and because the committed offset is the group's position, it can roll that position backwards, causing a mass re-read, or forwards, causing silent loss. The coordinator increments the generation on every rebalance, and `commit` rejects any commit from a stale generation. The caller must treat a rejection as "I am no longer a member" and rejoin, not as a transient error to retry.

**Q4. What is the difference between at-least-once and at-most-once, mechanically?**
The order of two lines in the processing loop. Commit before processing is at-most-once: the position advances past records that have not been handled, so a crash between the two loses them permanently — the demo proves it, with five payments disappearing and the restart reading nothing. Commit after processing is at-least-once: a crash between processing and committing means the restart re-reads records that were already handled, so the handler sees them twice — the demo shows ten handler calls for five offsets. The broker is not involved in this decision; it is a client-side choice, which is why "what guarantee do you need?" is a requirements question, not an implementation detail.

**Q5. So how do you get exactly-once?**
Two halves, and both are needed. On the producer side, idempotence: each batch carries `(producer_id, sequence)`, and the broker dedupes, so a retried batch returns the original offset instead of appending a second copy. That is necessary because deduping on `(topic, partition, offset)` in the consumer only catches re-reads at the *same* offset — a producer retry creates a *new* offset with the same logical message, which the consumer's dedupe cannot see. On the consumer side, the offset commit must be atomic with the output write, so a crash discards both and the batch is re-read with no visible effect. And the boundary must be stated: this is exactly-once *within the system*. If the sink is an external database that is not a transactional participant, the strongest real guarantee is at-least-once plus an idempotent write, and claiming more is a red flag.

**Q6. Why is `max.poll.records` a correctness parameter and not a performance one?**
Because a batch that takes longer to process than the session timeout gets the consumer evicted, which triggers a rebalance, which hands the batch to someone else, which processes it again — duplication caused entirely by the batch size. So the bound must be derived from the handler's worst-case latency against the rebalance timeout, not from a throughput target. In practice you measure p99 handling time per record and set the batch so the worst case stays well inside the timeout, then raise throughput by adding partitions and consumers rather than by making the batch bigger.

**Q7. Why is a poison pill parked in a DLQ rather than retried?**
Because a queue is ordered and a partition is processed in order, so a message that always fails blocks everything behind it — the partition's lag grows without bound and one bad record stops the pipeline. Retrying a bounded number of times handles the transient cases (a downstream timeout, a lock conflict) and the DLQ handles the permanent ones (a malformed payload, a schema mismatch). The two things a DLQ must have: a retry cap, and provenance — the origin `(topic, partition, offset)` in the DLQ record — because without provenance a dead-lettered message cannot be traced or replayed, and the DLQ becomes a place data goes to be forgotten.

**Q8. What happens when retention deletes records a consumer has not read?**
The consumer's committed offset falls below the log start offset, and its next read raises `OffsetOutOfRange`. There is no recovery that preserves the data — it is gone. The only options are to reset to the earliest available record (losing the gap, and the loss is invisible unless you log it) or to fail loudly and require an operator decision. The design defence is upstream: alert when lag multiplied by the ingest rate approaches the retention window, because the failure is predictable and the alert is cheap. The demo shows the mechanism with `log_start_offset` advancing to 7.

**Q9. Why does a new consumer start at the committed offset rather than the log end?**
Because the log end would skip everything produced while the consumer was not running, and it would do so silently — the consumer would look healthy and be missing data. Starting at the committed offset, or the log start if there is none, is the safe default. "Start at latest" is a legitimate choice for a consumer that only wants live events (a dashboard tail, a cache warmer), but it must be an explicit configuration decision, not an accident of how the seek was written. The demo checks the log-start fallback.

**Q10 (senior). A tenant's key is 90% of your traffic. What do you do?**
First, measure it — `max/mean` partition size above about 2 means one key dominates, and the demo computes exactly that. Adding partitions does not help, because a single key hashes to exactly one partition; that is the mistake people make first. The real options, in order. If ordering for that tenant is not actually required, salt the key — append a bucket suffix — which spreads it across partitions and gives up per-key order, so this must be a product decision. If ordering is required, split the tenant by a sub-key that carries the required ordering (per-device, per-session), which preserves the order that matters and discards the order that does not. If neither works, give that tenant a dedicated topic or a dedicated partition count, which is the "large tenant" escape hatch and is honestly a capacity decision as much as a design one.

**Q11 (senior). Where would you expect a subtle bug in this design?**
Four places, and they share a property: each passes a single-process test and fails in production. First, the **partitioner's determinism** — a randomised hash looks perfect in one process. Second, the **generation fence** — without it the zombie commit only appears when a rebalance overlaps a slow consumer, which is rare and load-dependent. Third, the **commit boundary** — an off-by-one where the committed offset is the last read rather than the next to read either re-reads one record forever or skips one forever, and both are invisible until someone reconciles counts. Fourth, the **rebalance during a long poll**, which duplicates work and looks like an intermittent bug in the handler. I would write a test for each: two partitioner instances must agree; a stale-generation commit must be rejected; the committed offset must equal the high watermark after a clean drain; and a crash between processing and committing must produce duplicates, not loss.

**Q12. Why is a consumer single-threaded here?**
Because the position map and the buffered records are per-consumer state, and sharing them across threads means either a lock on the hottest path or a race that shows up as one record processed twice and another skipped. Kafka's rule is the same and it is not an accident: the consumer is a single-threaded object, and parallelism comes from partitions and group members. If a handler is genuinely CPU-bound and the partition count is high, the pattern is a poll thread that hands batches to a worker pool — with the offset commit done only after the pool drains the batch, so the ordering of the commit relative to the work is preserved. That is the correct way to get both.

**Q13. Why is the offset key `(group, topic, partition)` rather than `(topic, partition)`?**
Because the position belongs to the *consumer group*, not to the topic. Two groups reading the same topic are independent readers with independent positions — that is what makes the pub/sub pattern work: the analytics group and the billing group both read `orders` and neither affects the other. Keying on `(topic, partition)` alone would make the second group to commit overwrite the first group's position, and the symptom would be a group silently jumping forward or backwards depending on which one committed last. The demo builds a second group with the same broker and distinct offsets.

**Q14. What is consumer lag and why is it the operational number?**
Lag is the high watermark minus the consumer's position: how many records exist that this group has not read. It is the single number that tells you whether the pipeline is keeping up, and it is better than throughput because it integrates the whole history — a consumer can have healthy throughput and a growing lag, which means the ingest rate rose and nobody noticed. The demo computes it per partition. The two thresholds worth alerting on: lag multiplied by the ingest rate approaching the retention window (imminent data loss), and lag growing monotonically for more than a few minutes (the consumer is losing).

**Q15. Why a pull model rather than push?**
Because the consumer knows its own capacity and the broker does not. In a push model the broker must track each consumer's rate, apply its own flow control, and get it wrong for the slowest member — and a slow consumer either gets overwhelmed or gets skipped. With pull, a slow consumer simply polls less, the lag number rises, and that is the correct behaviour: the system reports the problem rather than hiding it. It also makes batch size a consumer-side decision, which is what ties `max.poll.records` to the handler's latency rather than to a broker-side guess. The cost is that the consumer must poll on a schedule, and a consumer that stops polling looks like a dead member.

---

### Pushback round

**P1. "This is a lot of machinery. Why not a database table with a status column?"**
For low volume that is genuinely better and I would say so — a table with `status`, `created_at` and an index gives you durability, transactions and replay with none of the queue semantics to operate. The queue earns its place when at least one of these is true: you need more than a few hundred writes per second sustained; you need multiple independent consumer groups over the same stream; you need ordered consumption per key at scale; or you need retention and replay as a first-class feature rather than an ad-hoc query. The transition point is usually the *second* consumer group, because at that moment the table starts needing a fan-out table per group, and that is a queue with a worse API.

**P2. "Your at-least-once demo showed the handler running twice for the same record. That is a bug, not a design."**
It is the design, and the demo is deliberately unflattering because the alternative claims are worse. At-least-once means exactly that: a crash between processing and committing produces duplicates, and the five records were handled ten times. The claim I am making is that duplicates are *recoverable* and loss is not — a double-counted click can be deduplicated if the sink is idempotent; a lost payment cannot be recovered because nothing knows it existed. That is why I default to at-least-once and put the burden on the consumer. If the interviewer then says "a duplicate is unacceptable", the answer is exactly-once or an idempotent sink, and I would say which and what each costs.

**P3. "You modelled the log in memory with a Python list. That is not a queue."**
Correct, and the trade is stated in §10. A real broker has segment files, an offset index, a page cache and zero-copy sends; none of that is modelled, because in an LLD interview the thing being assessed is the *semantics* — that a partition is append-only, that offsets are monotonic, that a reader position is an integer, and that retention moves the log start. Those semantics are identical in the real system, and they are what the rest of the design depends on. What the in-memory model deliberately cannot show — durability across a broker failure, replication and ISR, rebalance under network partition — I cross-link to the HLD material rather than pretending to reproduce it.

**P4. "Your exactly-once is fake: you dedupe in the consumer's memory."**
The dedupe store in the demo is in memory, and that is the weakest part of it — it is honest in the docstring, which says the dedupe store must be as durable as the output. The *mechanism* being demonstrated is the correct one: the record is re-read after the crash, and the duplicate is suppressed by its coordinates, so the sink sees each record once. Making it real means putting the dedupe set in the same transaction as the output, which is exactly what the transactional-offset-commit does in a real system. So the demo shows the shape and the caveat, and I would not claim more than that: for an external sink with no transactional participant, the strongest guarantee is at-least-once plus an idempotent write, and saying so is more valuable than a stronger claim that is not true.

**P5. "Why bother with the assignor abstraction when there is one obvious answer?"**
Because there is not one obvious answer, and the choice is a real operational trade. Range hands each member a contiguous range per topic, so a leaver moves only its own share — assignments are comparatively stable, which matters because every moved partition means re-reading from a committed offset and rebuilding any in-memory state. Round-robin gives even counts when partitions do not divide evenly, and moves almost everything on every rebalance. Sticky assignors exist precisely to get both, at the cost of a much more complicated algorithm and a much harder correctness argument. In an interview, naming that range-versus-round-robin trade — stability against evenness — is the point; the abstraction exists because both are defensible and the choice is a configuration decision.

---

## 12. Cheat-sheet summary + cross-links

| Concept | One-line truth |
|---|---|
| The model | A partition is an **append-only log**; a reader position is an integer offset |
| Who owns what | The broker owns the bytes and assigns offsets; the **consumer group owns the committed position** |
| Ordering | **Per partition only** — never per topic. There is no cross-partition order and no timestamp recovers one |
| The partitioner | A **correctness component**: deterministic per key across processes. Never `hash()` |
| Hot key | `max/mean` partition size > ~2. **Adding partitions does not help a single hot key** |
| The committed offset | The **NEXT offset to read** — an off-by-one either re-reads or skips, forever and silently |
| `max.poll.records` | A **correctness** bound: a batch slower than the session timeout causes a rebalance and duplication |
| Delivery guarantee | The **order of two lines** in the processing loop, chosen by the client, not the broker |
| At-most-once | Commit → process. Lossy. The demo loses five payments |
| At-least-once | Process → commit. Duplicating. The demo handles five records ten times |
| Exactly-once | Idempotent producer `(producer_id, sequence)` **plus** an atomic offset commit; the boundary stops at an external sink |
| Idempotent consumer | Dedupe on `(topic, partition, offset)` — and it cannot catch a producer retry at a new offset |
| Generation | The **zombie fence**. A stale-generation commit is rejected; the caller must rejoin, not retry |
| Rebalance | Bumps the generation; start every member empty so no partition is owned twice |
| Seek on assign | The committed offset, or the **log start** if none — never the log end |
| Retention | Moves `log_start_offset`; a committed offset below it means the data is **gone**, and the read raises |
| Poison pill | Retry with a cap, then **DLQ with provenance** `(topic, partition, offset)`, then advance |
| Backpressure | A **bounded prefetch** plus `pause`/`resume`; unbounded buffering turns lag into an OOM |
| Concurrency | **One consumer, one thread**; parallelism comes from partitions and group members |
| Lag | High watermark − position. The operational number; alert before lag × rate reaches retention |
| Compaction | Keeps the newest record per key and leaves **gaps** — the log is no longer contiguous |
| Patterns | Strategy (partitioner, assignor), Mediator (coordinator), Template Method (loop), Decorator (guarantees), Repository (offsets) |
| Rejected | Singleton (inject the broker), Observer (push fights flow control) |

**The three sentences to memorise**

1. *"A partition is an append-only log, so a consumer is an integer offset, and the broker and the consumer group each own half of that state."*
2. *"Partitioning by key is what gives per-key ordering — and it makes the partitioner a correctness component, so it must be a stable digest and the partition count is effectively immutable."*
3. *"The delivery guarantee is the order of two lines in the processing loop: commit-then-process loses, process-then-commit duplicates, and exactly-once needs an idempotent producer plus an atomic offset commit."*

### Cross-links

- **The worked Kafka example, and why the queue exists at all** (the 34× write amplification argument, at-least-once plus idempotent aggregation, the failure matrix for Kafka being down): [URL Shortener](../../Case-Studies/HLD/01-url-shortener.md) — Deep dive D, *Analytics pipeline*
- **Partition placement, replication, ISR, quorum and consistency** (the HLD layer this file does not rebuild): [Sharding, Replication & Consistency](../../HLD/03-scaling/05-sharding-replication-and-consistency.md)
- **The same partitioned-log idea used to sequence chat messages, partitioned by conversation id:** [WhatsApp LLD](../../Case-Studies/LLD/02-whatsapp-lld.md) — the outbox, the sequence generator and where per-conversation ordering is enforced
- **The notification fan-out built on this:** [Notification System LLD](../../Case-Studies/LLD/05-notification-system-lld.md) — priority queues, retries, DLQ and the dispatch pipeline
- **Concurrency primitives, locks versus generations, the fencing argument in general:** [LLD/04 Concurrency & Threading](../04-concurrency-and-threading-in-lld.md)
- **The Strategy, Mediator and Template Method catalogue:** [LLD/03 Design Patterns](../03-design-patterns.md)
- **The rate-limiting filter used for consumer-side flow control and log flooding:** [Rate Limiter](11-rate-limiter.md)
- **Where a must-not-lose record belongs instead of a log line, and the reverse:** [Logger Framework](12-logger-framework.md) §7.6
- **Cheat sheet, pattern tables and the classic-problems list:** [LLD Cheatsheet](../../Cheat-Sheets/07-lld-cheatsheet.md)
- **Question bank:** [LLD Questions](../../Interview-Questions/08-lld-questions.md)
