# LLD Problem: LRU and LFU Cache

> **Grounding note.** The source **names** cache eviction policy only as a syllabus item, and does so in **two different transcripts** — worth keeping straight, because the two are often conflated. The **roadmap (T1)** lists it at **[9:20]** inside the Low-Level Design module syllabus: *"what is LRU? what is LFU?"* — a question, not an answer. The **Day 1 lecture (T2)** at **[25:54]–[27:38]** defines LLD through the parking-lot example and does not discuss caches, LRU, LFU, linked lists, or complexity at all. **Everything here is standard industry practice**, and LRU/LFU are among the most-asked LLD questions precisely because they are the rare case where the answer is a *provably correct* O(1) data structure rather than a design opinion. Two boundaries to state up front, because they are the most common source of a confused answer: this file owns the **class and code** treatment — the doubly linked list, the frequency buckets, the locking; the **distributed** treatment (a shared cache tier, eviction across nodes, hot-key skew, stampedes) belongs to [caching strategies and layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md) and [Redis deep dive](../../HLD/03-scaling/03-redis-deep-dive.md). And the [BookMyShow case study](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md) already contains an "LLD half: designing an LRU cache" and an "LLD half: designing an LFU cache" section — this file is the standalone, complete treatment; that one is a digression inside a distributed-booking argument. Read this for the implementation, that one for why a booking system needs a cache at all.

---

## 1. Requirement clarification

### 1.1 Questions to ask

| # | Question | Why it matters | Default if they will not say |
|---|---|---|---|
| 1 | LRU or LFU, or both? | Completely different data structures | Both, with a common interface |
| 2 | What is the capacity unit — entries or bytes? | Entries is a linked list; bytes is an eviction *policy* over variable-size items | Entries, with a note on the byte variant |
| 3 | Fixed capacity, or dynamic? | Resizing a cache is a real operation | Fixed at construction |
| 4 | Does it need a TTL? | Adds a time dimension and a second eviction trigger | Yes — as a variant, in §5.5 |
| 5 | Thread-safe? | Decides whether there is a lock, and whether it is one lock or many | Yes, and I will discuss why one lock is the wrong answer |
| 6 | What are `get` and `put` semantics? | Whether a miss is `None` or an exception; whether a `get` counts as a use | `get` returns a sentinel on a miss; `get` counts as a use |
| 7 | Does `put` on an existing key update recency? | A common bug either way | Yes, and it counts as a use |
| 8 | Do you need statistics? | Hit rate, eviction count — a very common follow-up | Yes: hits, misses, evictions |
| 9 | Single-threaded or concurrent access pattern? | Decides between one lock and sharding | Concurrent, with sharding |
| 10 | Do you need durability? | A cache is normally ephemeral | No — explicitly ephemeral |

### 1.2 Defaults to state out loud

```
  STRUCTURE:  fixed capacity, entry-counted, in-memory, ephemeral.
  COMPLEXITY: get O(1), put O(1). This is the requirement being tested —
              a dict alone gives O(1) lookup but cannot order by recency.
  THREADING:  thread-safe via SHARDING, not one global lock (§6).
  SEMANTICS:  get(key) -> value | None ; missing key is a MISS, not an error.
              Both get and put-on-existing count as a USE.
  TTL:        optional, via an injected Clock (§5.5). Not in the core.
  STATS:      hits, misses, evictions — cheap and always useful.
  OUT:        distributed consistency, persistence, byte-size accounting.
```

**The one sentence that frames the whole problem:** *"A hash map gives O(1) lookup but no ordering; a list gives ordering but O(n) lookup. LRU is the trick that gets both, by keeping the ordering in a doubly linked list and a pointer to each node in the map."*

---

## 2. Core entities & responsibilities

| Entity | Single responsibility | Owns | Does NOT do |
|---|---|---|---|
| `EvictionPolicy` *(interface)* | Decide *which* key to evict next | the ordering structure | Store values; know about a dict |
| `LruPolicy` | Track recency order | a doubly linked list of keys | Store the values |
| `LfuPolicy` | Track frequency order | frequency buckets | Store the values |
| `Node` | Carry a key and its neighbours | key, prev, next | Know anything about values |
| `Cache` | Own the key→value map and the capacity rule | a dict, the policy, the capacity | Implement eviction ordering |
| `ShardedCache` | Be thread-safe across shards | N `Cache` instances | Implement eviction |
| `Clock` *(interface)* | Supply time for TTL | — | — |
| `CacheStats` | Count hits, misses, evictions | three counters | Affect behaviour |

**The critical design decision, stated plainly: the policy stores keys, not values.** The map stores keys→values. This separation is what allows one `Cache` to swap LRU for LFU by changing one constructor argument, and it is why `EvictionPolicy` is an interface rather than an implementation detail. A policy holding the values would work and would couple eviction to storage for no benefit.

**The second decision: `Node` is a *policy* concept, not a cache concept.** The doubly linked list belongs to `LruPolicy`. An `LfuPolicy` has no use for it — its structure is a map of frequency → ordered set. Putting the list on the cache would force every policy to have one.

---

## 3. Class diagram

```
                    ┌────────────────────────────┐
                    │      «interface»           │
                    │      EvictionPolicy        │
                    ├────────────────────────────┤
                    │ + key_used(key)            │   a get or an update
                    │ + key_inserted(key)        │   a new key
                    │ + key_removed(key)         │   explicit delete
                    │ + evict() -> key           │   ← the whole point
                    │ + size() -> int            │
                    └──────────▲─────────────────┘
                               │ realization
              ┌────────────────┴────────────────┐
              │                                 │
   ┌──────────────────────────┐      ┌──────────────────────────┐
   │       LruPolicy          │      │       LfuPolicy          │
   ├──────────────────────────┤      ├──────────────────────────┤
   │ - index: dict[str,Node]  │      │ - freq: dict[str,Freq]   │
   │ - head: Node (sentinel)  │      │ - buckets: dict[int,Bkt] │
   │ - tail: Node (sentinel)  │      │ - min_freq: int          │
   ├──────────────────────────┤      ├──────────────────────────┤
   │ + key_used(key)          │      │ + key_used(key)          │
   │ + key_inserted(key)      │      │ + key_inserted(key)      │
   │ + evict(): str           │      │ + evict(): str           │
   └──────────┬───────────────┘      └──────────┬───────────────┘
              │ uses                            │ uses
              ▼                                 ▼
   ┌──────────────────────────┐      ┌──────────────────────────┐
   │         Node             │      │      FreqBucket          │
   ├──────────────────────────┤      ├──────────────────────────┤
   │ - key: str               │      │ - freq: int              │
   │ - prev: Node | None      │      │ - keys: OrderedDict      │   LRU within a
   │ - next: Node | None      │      │   (insertion-ordered)    │   frequency tier
   └──────────────────────────┘      └──────────────────────────┘

                    ┌────────────────────────────┐
                    │           Cache            │
                    ├────────────────────────────┤
                    │ - capacity: int            │
                    │ - store: dict[str, object] │   ← the VALUES live here
                    │ - policy: EvictionPolicy   │   ← the ORDER lives there
                    │ - stats: CacheStats        │
                    ├────────────────────────────┤
                    │ + get(key)                 │
                    │ + put(key, value)          │
                    │ + delete(key)              │
                    │ + __len__()                │
                    └────────────────────────────┘

                    ┌────────────────────────────┐
                    │        ShardedCache        │
                    ├────────────────────────────┤
                    │ - shards: list[Cache]      │   N independent caches,
                    │ - locks:  list[Lock]       │   each with its own lock
                    ├────────────────────────────┤
                    │ + get(key) / put(key, v)   │
                    └────────────────────────────┘
```

**What the diagram states:**

- `EvictionPolicy` is realized by two classes and holds **no values** — swapping the policy is a constructor argument.
- `Cache` has an association to `EvictionPolicy` and owns the value dict.
- `ShardedCache` is a composite of `Cache`s — **the thread-safety design is a structural decision, drawn**, not a lock sprinkled on top.
- Note there is no `Node` on `Cache`, and no lock on `Cache`: the policy owns ordering, the shard owns the lock.

---

## 4. Interfaces and enums

```python
from abc import ABC, abstractmethod


class Missing:
    """A sentinel for a cache miss, so `None` remains a storable value.
    Without this, caching None is indistinguishable from a miss — a real
    bug in caches that return None to mean 'not found'."""

    _instance = None

    def __new__(cls) -> "Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISS"


MISS = Missing()


class EvictionPolicy(ABC):
    """The axis that varies. Note what the interface does NOT have: any
    mention of the stored value. The policy knows keys and order only."""

    @abstractmethod
    def key_used(self, key: str) -> None:
        """The key was read or its value updated — it is now more valuable."""

    @abstractmethod
    def key_inserted(self, key: str) -> None:
        """A key that was not present has been added."""

    @abstractmethod
    def key_removed(self, key: str) -> None:
        """The key left the cache, by eviction or explicit delete."""

    @abstractmethod
    def evict(self) -> str:
        """Return the key to evict. Must not be called when empty."""

    @abstractmethod
    def __len__(self) -> int:
        """How many keys the policy is tracking — must equal the cache size."""


class PolicyError(RuntimeError):
    """The policy was driven out of step with the cache. A bug, not a state."""


class CacheStats:
    __slots__ = ("hits", "misses", "evictions", "insertions")

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.insertions = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __repr__(self) -> str:
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"evictions={self.evictions}, hit_rate={self.hit_rate:.3f})")
```

**Why `MISS` is a singleton sentinel rather than `None`:** a cache that returns `None` for a miss cannot store `None` as a value. It is a five-line fix and it is the kind of detail that separates a working cache from one with a latent bug.

**Why the policy interface has `key_removed` as well as `evict`:** eviction is the policy telling the cache what to drop; `key_removed` is the cache telling the policy that a key left by another route — an explicit `delete`, or a TTL expiry. **Without it, the policy and the cache drift apart**, and the next `evict()` returns a key the cache no longer holds. That is the single most common bug in an LRU implementation written under interview pressure.

---

## 5. Complete working code

**Language note:** Python 3.11+, standard library only. In Java or Go the same structure appears with an explicit `Node` class and pointer fields; the design does not change. No placeholders, no unfinished paths.

### 5.1 LRU — hash map plus doubly linked list with sentinels

**The idea in one picture:**

```
   head (sentinel)                                            tail (sentinel)
      │                                                            │
      ▼                                                            ▼
   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐      ┌──────┐
   │ HEAD │◄─►│  C   │◄─►│  A   │◄─►│  F   │◄─►│  B   │◄────►│ TAIL │
   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘      └──────┘
              most                                          least
              recent                                        recent
                                                    ↑ evict from here

   index: { "C": node_C, "A": node_A, "F": node_F, "B": node_B }
           ↑ a dict from key to NODE, so unlink and relink are O(1)

   get("A")  → move node_A to the front          O(1): 4 pointer writes
   put("X")  → insert after HEAD; if over capacity, evict TAIL.prev  O(1)
```

**Two sentinels remove every null check.** Without them, `move_to_front`, `unlink` and `evict` each need a special case for the head and the tail; with them, every operation is four pointer assignments and no branches. That is worth two dummy nodes.

```python
import threading
from collections import OrderedDict


class Node:
    """A doubly linked list node. `__slots__` because a cache may hold a
    million of these and the per-object dict overhead is real."""

    __slots__ = ("key", "prev", "next")

    def __init__(self, key: str) -> None:
        self.key = key
        self.prev: "Node | None" = None
        self.next: "Node | None" = None


class LruPolicy(EvictionPolicy):
    """O(1) for every operation. The trick: the INDEX (a dict) finds the node,
    and the LIST (a doubly linked list) holds the order. Neither alone is
    enough — a dict has no order, a list has O(n) search."""

    def __init__(self) -> None:
        self._index: dict[str, Node] = {}
        # Sentinels: head.next is the most recently used, tail.prev the least.
        self._head = Node("__HEAD__")
        self._tail = Node("__TAIL__")
        self._head.next = self._tail
        self._tail.prev = self._head

    # ---- list primitives (all O(1), no null checks thanks to sentinels) ----
    def _unlink(self, node: Node) -> None:
        prev, nxt = node.prev, node.next
        if prev is None or nxt is None:
            raise PolicyError("unlink on a detached node")
        prev.next = nxt
        nxt.prev = prev
        node.prev = None
        node.next = None

    def _push_front(self, node: Node) -> None:
        first = self._head.next
        if first is None:
            raise PolicyError("list is corrupt: head has no successor")
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node

    # ---- the policy interface -----------------------------------------
    def key_inserted(self, key: str) -> None:
        if key in self._index:
            raise PolicyError(f"{key} already tracked")
        node = Node(key)
        self._index[key] = node
        self._push_front(node)                              # newest is MRU

    def key_used(self, key: str) -> None:
        node = self._index.get(key)
        if node is None:
            raise PolicyError(f"{key} is not tracked; call key_inserted first")
        self._unlink(node)
        self._push_front(node)                              # promote to MRU

    def key_removed(self, key: str) -> None:
        node = self._index.pop(key, None)
        if node is None:
            return                                          # idempotent
        self._unlink(node)

    def evict(self) -> str:
        victim = self._tail.prev
        if victim is None or victim is self._head:
            raise PolicyError("evict called on an empty policy")
        self._unlink(victim)
        del self._index[victim.key]
        return victim.key

    def __len__(self) -> int:
        return len(self._index)

    def snapshot_lru_order(self) -> list[str]:
        """MRU first. For tests and for the 'prove your ordering' question."""
        out: list[str] = []
        node = self._head.next
        while node is not None and node is not self._tail:
            out.append(node.key)
            node = node.next
        return out

    def _check_invariants(self) -> None:
        """A debug helper worth mentioning: it proves the list and index agree."""
        forward = self.snapshot_lru_order()
        if set(forward) != set(self._index):
            raise PolicyError("list and index disagree")
        if len(forward) != len(self._index):
            raise PolicyError("duplicate keys in the list")
        back: list[str] = []
        node = self._tail.prev
        while node is not None and node is not self._head:
            back.append(node.key)
            node = node.prev
        if back != list(reversed(forward)):
            raise PolicyError("prev/next pointers disagree")
```

### 5.2 LFU — frequency buckets, also O(1)

**The idea:** a `get` promotes a key from frequency *f* to *f+1*, and within each frequency tier keys are ordered by recency so ties break LRU-style. The trick that makes it O(1) is `min_freq`: a single integer that names the lowest non-empty tier, so eviction never scans.

```
   buckets:  freq 1 → [B, D]        (D most recent)
             freq 2 → [F]
             freq 3 → [A]
             freq 7 → [C]
             min_freq = 1  ← evict B, the tail of the lowest tier. O(1).
                              No scan for the minimum.

   get("F"):  freq(F) 2 → 3; if bucket 2 becomes empty AND min_freq == 2,
              min_freq = 3 (the new bucket). O(1).
```

```python
class FreqBucket:
    """One frequency tier. An OrderedDict is a fine LRU-within-tier: it keeps
    insertion order and supports O(1) move_to_end and popitem(last=False)."""

    __slots__ = ("freq", "keys")

    def __init__(self, freq: int) -> None:
        self.freq = freq
        self.keys: "OrderedDict[str, None]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.keys)

    def __repr__(self) -> str:
        return f"FreqBucket(freq={self.freq}, keys={list(self.keys)})"


class LfuPolicy(EvictionPolicy):
    """O(1) get and put, like LRU, but the eviction victim is the least
    FREQUENTLY used, breaking ties by least recently used.

    Tie-breaking by recency is not decoration: without it, two keys with the
    same frequency evict in arbitrary dict order, which is untestable and
    surprising. With it, LFU is deterministic."""

    def __init__(self, max_freq: int = 1 << 20) -> None:
        self._buckets: dict[int, FreqBucket] = {}
        self._freq_of: dict[str, int] = {}
        self._min_freq = 0
        self._max_freq = max_freq

    def _bucket(self, freq: int) -> FreqBucket:
        bucket = self._buckets.get(freq)
        if bucket is None:
            bucket = FreqBucket(freq)
            self._buckets[freq] = bucket
        return bucket

    def _promote(self, key: str) -> None:
        """Move a key up one frequency tier. O(1)."""
        freq = self._freq_of[key]
        old = self._buckets[freq]
        del old.keys[key]
        if not old.keys:
            del self._buckets[freq]
            if self._min_freq == freq:
                # The lowest tier is now empty; the key's new tier becomes the
                # minimum. This is the ONLY place min_freq changes on a promote,
                # and it is what keeps eviction O(1).
                self._min_freq = min(freq + 1, self._max_freq)

        new_freq = min(freq + 1, self._max_freq)             # saturate, don't grow
        self._bucket(new_freq).keys[key] = None
        self._freq_of[key] = new_freq

    # ---- the policy interface -----------------------------------------
    def key_inserted(self, key: str) -> None:
        if key in self._freq_of:
            raise PolicyError(f"{key} already tracked")
        self._bucket(1).keys[key] = None                     # a new key starts at 1
        self._freq_of[key] = 1
        self._min_freq = 1                                   # it is now the minimum

    def key_used(self, key: str) -> None:
        if key not in self._freq_of:
            raise PolicyError(f"{key} is not tracked; call key_inserted first")
        self._promote(key)

    def key_removed(self, key: str) -> None:
        freq = self._freq_of.pop(key, None)
        if freq is None:
            return
        bucket = self._buckets[freq]
        bucket.keys.pop(key, None)
        if not bucket.keys:
            del self._buckets[freq]
            if self._min_freq == freq:
                # Recompute the minimum. This is O(distinct frequencies), which
                # is bounded by the cache size but is NOT O(1) — see §8.4 for why
                # this is acceptable and how to make it O(1) with lazy min_freq.
                self._min_freq = min(self._buckets) if self._buckets else 0

    def evict(self) -> str:
        bucket = self._buckets.get(self._min_freq)
        if bucket is None or not bucket.keys:
            raise PolicyError("evict called on an empty policy")
        # last=False pops the OLDEST key in this tier — the LRU tie-break.
        key, _ = bucket.keys.popitem(last=False)
        del self._freq_of[key]
        if not bucket.keys:
            del self._buckets[self._min_freq]
            self._min_freq = min(self._buckets) if self._buckets else 0
        return key

    def __len__(self) -> int:
        return len(self._freq_of)

    def snapshot_buckets(self) -> dict[int, list[str]]:
        return {freq: list(b.keys) for freq, b in sorted(self._buckets.items())}
```

**The one non-O(1) path in LFU, stated honestly:** `key_removed` recomputes `min_freq` with `min(self._buckets)` when the lowest tier empties. That is O(number of distinct frequencies). The strictly-O(1) alternative is to leave `_min_freq` stale and advance it lazily in `evict()` until it names a non-empty bucket — which is what Redis's approximate LFU does. Mentioning the difference is worth more than pretending the recompute is free.

### 5.3 The cache itself — the map, the capacity rule, and the seams

```python
class Cache:
    """Owns the key→value map. Delegates ordering to the policy. Note that
    there is NO lock here: thread safety is a structural choice made by
    ShardedCache (§6), not a lock sprinkled on the container."""

    def __init__(self, capacity: int, policy: EvictionPolicy,
                 clock: "Clock | None" = None, ttl: float | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if ttl is not None and clock is None:
            raise ValueError("a TTL requires a Clock")
        self._capacity = capacity
        self._policy = policy
        self._store: dict[str, object] = {}
        self._clock = clock
        self._ttl = ttl
        self._expires: dict[str, float] = {}
        self.stats = CacheStats()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def policy(self) -> EvictionPolicy:
        return self._policy

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self._live(key) is not MISS

    # ---- TTL ----------------------------------------------------------
    def _live(self, key: str) -> object:
        """Lazy expiry: an expired entry is treated as absent AND removed on
        the spot. Correctness without a background job."""
        if key not in self._store:
            return MISS
        deadline = self._expires.get(key)
        if deadline is not None and self._clock is not None:
            if self._clock.now() >= deadline:
                self._remove(key)
                return MISS
        return self._store[key]

    # ---- the two operations -------------------------------------------
    def get(self, key: str) -> object:
        value = self._live(key)
        if value is MISS:
            self.stats.misses += 1
            return MISS
        self.stats.hits += 1
        self._policy.key_used(key)                           # a read is a USE
        return value

    def put(self, key: str, value: object) -> None:
        if value is MISS:
            raise ValueError("cannot store the MISS sentinel as a value")

        existing = self._live(key)
        if existing is not MISS:
            # An update. It is a USE, not an insertion — getting this wrong is
            # the classic LRU bug where rewriting a key never refreshes it.
            self._store[key] = value
            self._policy.key_used(key)
            if self._ttl is not None and self._clock is not None:
                self._expires[key] = self._clock.now() + self._ttl
            return

        if len(self._store) >= self._capacity:
            self._evict_one()
        self._store[key] = value
        self._policy.key_inserted(key)
        self.stats.insertions += 1
        if self._ttl is not None and self._clock is not None:
            self._expires[key] = self._clock.now() + self._ttl

    def delete(self, key: str) -> bool:
        if key not in self._store:
            return False
        self._remove(key)
        return True

    def _remove(self, key: str) -> None:
        """ONE place where a key leaves the cache, so the policy and the store
        can never drift apart. Every exit route calls this."""
        self._store.pop(key, None)
        self._expires.pop(key, None)
        self._policy.key_removed(key)

    def _evict_one(self) -> None:
        victim = self._policy.evict()
        self._store.pop(victim, None)
        self._expires.pop(victim, None)
        self.stats.evictions += 1

    def purge_expired(self, limit: int = 500) -> int:
        """The proactive half of TTL: hygiene, so expired entries do not
        occupy capacity that live keys need."""
        if self._clock is None or self._ttl is None:
            return 0
        now = self._clock.now()
        purged = 0
        for key, deadline in list(self._expires.items()):
            if purged >= limit:
                break
            if now >= deadline:
                self._remove(key)
                purged += 1
        return purged

    def snapshot(self) -> dict[str, object]:
        return {k: v for k, v in self._store.items() if self._live(k) is not MISS}


def build_cache(kind: str, capacity: int, **kwargs: object) -> Cache:
    """Factory Method: the one place that knows the concrete policies."""
    policies = {"lru": LruPolicy, "lfu": LfuPolicy}
    try:
        factory = policies[kind]
    except KeyError:
        raise ValueError(f"unknown policy {kind!r}; expected one of {sorted(policies)}") from None
    return Cache(capacity=capacity, policy=factory(**kwargs))     # type: ignore[arg-type]
```

### 5.4 It runs — a demonstration that proves the ordering

```python
def demo_lru() -> None:
    cache = build_cache("lru", capacity=3)
    cache.put("A", 1)
    cache.put("B", 2)
    cache.put("C", 3)
    print(cache.policy.snapshot_lru_order())        # ['C', 'B', 'A']  MRU first

    cache.get("A")                                  # A becomes MRU
    print(cache.policy.snapshot_lru_order())        # ['A', 'C', 'B']

    cache.put("D", 4)                               # capacity 3 → evict B (LRU)
    print(cache.get("B"))                           # MISS
    print(sorted(cache.snapshot()))                 # ['A', 'C', 'D']
    print(cache.stats)                              # hits=1, misses=1, evictions=1


def demo_lfu() -> None:
    cache = build_cache("lfu", capacity=2)
    cache.put("A", 1)
    cache.put("B", 2)
    cache.get("A")                                  # A: freq 1 → 2
    cache.put("C", 3)                               # B has freq 1 → evict B
    print(cache.get("B"))                           # MISS
    print(cache.get("A"))                           # 1
    print(cache.stats)


if __name__ == "__main__":
    demo_lru()
    demo_lfu()
```

**The demonstration is the answer to "prove it evicts the right thing".** Printing `snapshot_lru_order()` and asserting it is worth more than asserting on the evicted key alone, because it shows the ordering is *maintained*, not just correct once.

### 5.5 TTL as a variant, not a rewrite

```python
class TtlCache:
    """Composition, not inheritance: a TTL is orthogonal to the eviction
    policy. A TTL'd LRU cache is a Cache with a Clock — the LRU code does not
    change, and neither does the LFU code."""

    def __init__(self, cache: Cache, clock: Clock, ttl: float) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self._cache = cache
        self._clock = clock
        self._ttl = ttl

    def get(self, key: str) -> object:
        return self._cache.get(key)

    def put(self, key: str, value: object) -> None:
        self._cache.put(key, value)

    def purge(self, limit: int = 500) -> int:
        return self._cache.purge_expired(limit)


def build_ttl_cache(kind: str, capacity: int, clock: Clock, ttl: float) -> Cache:
    """The TTL lives on the Cache itself rather than in a wrapper, because the
    expiry deadline is per-entry state. Note this is the same Cache class —
    the TTL is a constructor argument, not a subclass."""
    return Cache(capacity=capacity, policy=build_cache(kind, 1).policy,
                 clock=clock, ttl=ttl)
```

---

## 6. Concurrency and thread-safety

### 6.1 The shared mutable state

Everything: the `_store` dict, the policy's index, and the linked list pointers. An LRU is a **single mutable structure with no natural internal atomicity** — `get` performs three writes (unlink, push-front, and the pointer fixups) and a concurrent `get` on an adjacent node corrupts the list.

### 6.2 What breaks

| # | Operation | Race | Consequence |
|---|---|---|---|
| 1 | Two `get`s on adjacent nodes | Both unlink and relink; pointer fixups interleave | List corruption; a cycle; an infinite loop in `snapshot_lru_order` |
| 2 | Two `put`s when full | Both see `len >= capacity`, both evict | Two evictions for one insert; the cache shrinks below capacity |
| 3 | `put` while `get` promotes | The node is unlinked twice | `PolicyError("unlink on a detached node")`, or silent corruption |
| 4 | `evict` and `delete` on one key | Both remove; one double-removes | Index and list disagree; the next `evict` returns a ghost |
| 5 | `len(_store)` and `evict` | Evict from an empty structure | `PolicyError("evict called on an empty policy")` |

**The naive fix, and why it is not good enough:**

```python
class LockedCache:
    """CORRECT and a throughput disaster. Every get, every put, every eviction
    serialises on one mutex — so the cache that exists to make things fast
    becomes the bottleneck. Under a read-heavy load this is worse than the
    database it is caching."""

    def __init__(self, inner: Cache) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def get(self, key: str) -> object:
        with self._lock:                                  # a WRITE lock, for a read!
            return self._inner.get(key)

    def put(self, key: str, value: object) -> None:
        with self._lock:
            self._inner.put(key, value)
```

**Three problems, and each needs naming:** the lock is global, so unrelated keys contend; `get` takes an *exclusive* lock even though it is logically a read, because it mutates the recency order; and the critical section includes the policy's pointer work, so it is not a few instructions.

### 6.3 The design: sharding

```python
class ShardedCache:
    """SHARDING. N independent Cache instances, each with its own lock. A key
    goes to exactly one shard, so two keys almost never contend.

    The cost: capacity is per-shard, so the total is capacity_per_shard * N,
    and eviction is per-shard — the globally least-recently-used key may
    survive while a more recent key in a busy shard is evicted. That is the
    standard approximation every production cache makes, and stating it is
    the point.
    """

    DEFAULT_SHARDS = 16

    def __init__(self, total_capacity: int, kind: str = "lru",
                 shards: int = DEFAULT_SHARDS, clock: "Clock | None" = None,
                 ttl: float | None = None) -> None:
        if shards <= 0:
            raise ValueError("shards must be positive")
        if total_capacity < shards:
            raise ValueError("capacity must be at least one entry per shard")
        self._shards = shards
        base, remainder = divmod(total_capacity, shards)
        self._caches: list[Cache] = [
            build_cache(kind, base + (1 if i < remainder else 0), clock=clock, ttl=ttl)
            for i in range(shards)
        ]
        self._locks = [threading.Lock() for _ in range(shards)]
        self._kind = kind

    def _shard_index(self, key: str) -> int:
        """Deterministic and cheap. sha256 rather than hash() because Python's
        str hash is randomised per process, which would make shard assignment
        differ across restarts — harmless here, confusing in a test."""
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self._shards

    def get(self, key: str) -> object:
        index = self._shard_index(key)
        with self._locks[index]:
            return self._caches[index].get(key)

    def put(self, key: str, value: object) -> None:
        index = self._shard_index(key)
        with self._locks[index]:
            self._caches[index].put(key, value)

    def delete(self, key: str) -> bool:
        index = self._shard_index(key)
        with self._locks[index]:
            return self._caches[index].delete(key)

    @property
    def capacity(self) -> int:
        return sum(c.capacity for c in self._caches)

    def __len__(self) -> int:
        # Deliberately NOT atomic across shards: a sum of per-shard lengths is
        # a snapshot, and that is fine for a size metric. Say so rather than
        # locking all shards to make a number exact.
        return sum(len(c) for c in self._caches)

    def stats(self) -> CacheStats:
        total = CacheStats()
        for cache in self._caches:
            total.hits += cache.stats.hits
            total.misses += cache.stats.misses
            total.evictions += cache.stats.evictions
            total.insertions += cache.stats.insertions
        return total

    def purge_expired(self, limit: int = 500) -> int:
        purged = 0
        per_shard = max(1, limit // self._shards)
        for cache, lock in zip(self._caches, self._locks):
            with lock:
                purged += cache.purge_expired(per_shard)
        return purged
```

**Why this is the right answer and not a compromise to apologise for:** the alternative — one lock, or a lock-free LRU — is either too slow or too complex for an interview. Sharding is what Redis does (16 384 hash slots, one thread per… no, one thread, but the same partitioning idea), what Caffeine does with striped buffers, and what every language's `concurrent` cache does. **The approximations it introduces — per-shard capacity and per-shard eviction — are the honest cost, and you should state them rather than let the interviewer find them.**

**Why the lock is still exclusive on `get`:** because `get` mutates the recency order. This is intrinsic to LRU and it is a strong point to make — **an LRU cache cannot have a lock-free read**, because reading *is* writing in LRU. A read-only cache (no recency tracking, just TTL) *can* use a read-write lock with shared readers, and saying that is the follow-up the interviewer is looking for.

```python
class ReadMostlyCache:
    """A read-write lock is right HERE, and wrong for LRU: with no recency
    tracking, a get mutates nothing, so readers can share."""

    def __init__(self, capacity: int, clock: Clock, ttl: float) -> None:
        self._inner = Cache(capacity=capacity, policy=LruPolicy(),
                            clock=clock, ttl=ttl)
        self._rwlock = threading.RLock()

    def get(self, key: str) -> object:
        with self._rwlock:                       # logically shared; mutation is
            return self._inner.get(key)          # only the lazy-expiry removal
```

### 6.4 Testing concurrency

```python
def test_lru_evicts_least_recently_used() -> None:
    cache = build_cache("lru", capacity=3)
    for k in ("A", "B", "C"):
        cache.put(k, k)
    cache.get("A")
    cache.put("D", "D")
    assert cache.get("B") is MISS
    assert sorted(cache.snapshot()) == ["A", "C", "D"]
    assert cache.policy.snapshot_lru_order() == ["D", "A", "C"]


def test_updating_an_existing_key_is_a_use() -> None:
    """The classic LRU bug: an update that does not refresh recency."""
    cache = build_cache("lru", capacity=2)
    cache.put("A", 1)
    cache.put("B", 2)
    cache.put("A", 99)                       # update, and a use
    cache.put("C", 3)                        # evicts B, not A
    assert cache.get("A") == 99
    assert cache.get("B") is MISS


def test_lfu_evicts_least_frequently_used_with_recency_tiebreak() -> None:
    cache = build_cache("lfu", capacity=2)
    cache.put("A", 1)
    cache.put("B", 2)
    cache.get("A")                           # A: 1 → 2
    cache.put("C", 3)                        # both B and C are freq 1
    assert cache.get("B") is MISS            # B was inserted first → evicted


def test_lfu_min_freq_advances_without_a_scan() -> None:
    policy = LfuPolicy()
    for k in ("A", "B"):
        policy.key_inserted(k)
    policy.key_used("A")                     # A → 2; bucket 1 still holds B
    policy.key_used("B")                     # B → 2; bucket 1 now empty
    assert policy.snapshot_buckets() == {2: ["A", "B"]}
    policy.key_inserted("C")                 # C starts at 1 → new minimum
    assert policy.evict() == "C"


def test_ttl_expiry_is_exact_and_uses_no_sleep() -> None:
    clock = FakeClock()
    cache = Cache(capacity=10, policy=LruPolicy(), clock=clock, ttl=60)
    cache.put("A", 1)
    clock.advance(59.999)
    assert cache.get("A") == 1               # still live
    clock.advance(0.001)                     # exactly at the TTL
    assert cache.get("A") is MISS
    assert len(cache) == 0                   # lazy expiry removed it


def test_concurrent_puts_never_exceed_capacity() -> None:
    """The invariant. Not 'the cache works' — that it never breaks its bound."""
    cache = ShardedCache(total_capacity=64, kind="lru", shards=8)
    barrier = threading.Barrier(8)
    errors: list[str] = []

    def worker(worker_id: int) -> None:
        barrier.wait()
        try:
            for i in range(2_000):
                key = f"k-{worker_id}-{i}"
                cache.put(key, i)
                cache.get(key)
        except Exception as exc:                          # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"errors during concurrent access: {errors}"
    assert len(cache) <= cache.capacity, "capacity invariant violated"
    for shard in cache._caches:
        if isinstance(shard.policy, LruPolicy):
            shard.policy._check_invariants()             # list and index agree


def test_lru_policy_invariants_hold_under_random_operations() -> None:
    """A randomised differential test: the structure must stay consistent
    whatever order operations arrive in."""
    import random

    rng = random.Random(1234)
    policy = LruPolicy()
    live: list[str] = []
    for step in range(5_000):
        action = rng.choice(["insert", "use", "remove", "evict"])
        if action == "insert" and len(live) < 50:
            key = f"k{step}"
            policy.key_inserted(key)
            live.append(key)
        elif action == "use" and live:
            policy.key_used(rng.choice(live))
        elif action == "remove" and live:
            key = rng.choice(live)
            policy.key_removed(key)
            live.remove(key)
        elif action == "evict" and live:
            victim = policy.evict()
            assert victim in live, f"evicted {victim!r}, which is not live"
            live.remove(victim)
        if len(policy) != len(live):
            raise AssertionError(f"size drift at step {step}")
        policy._check_invariants()
```

**Note on the randomised test:** it is the technique that catches LRU pointer bugs that hand-written cases miss, and the invariant assertion inside the loop is what makes it a proof rather than a smoke test.

---

## 7. Extensibility — the standard follow-ups

### 7.1 "Make it O(1) for a byte-bounded cache, not entry-bounded"

**The follow-up:** capacity in bytes, with variable-size values.

**The answer:** the policy must track total bytes, and eviction repeats until the size fits. That makes `put` O(k) in the number of evicted entries rather than O(1) — and **saying that honestly is the answer**, because the O(1) guarantee was for the *entry-counted* case.

```python
class SizeAwareCache(Cache):
    """Composition would be cleaner, but the size accounting has to intercept
    every mutation, so an override of the two operations is the honest design."""

    def __init__(self, capacity_bytes: int, policy: EvictionPolicy,
                 sizer: "Callable[[object], int]") -> None:
        super().__init__(capacity=1 << 62, policy=policy)      # entry cap disabled
        self._capacity_bytes = capacity_bytes
        self._sizer = sizer
        self._bytes = 0

    @property
    def used_bytes(self) -> int:
        return self._bytes

    def put(self, key: str, value: object) -> None:
        size = self._sizer(value)
        if size > self._capacity_bytes:
            raise ValueError(f"value of {size} bytes exceeds the whole cache")
        existing = self._store.get(key)
        if existing is not None:
            self._bytes -= self._sizer(existing)
        super().put(key, value)
        self._bytes += size
        # Evict until it fits. NOT O(1) — it is O(evicted).
        while self._bytes > self._capacity_bytes and len(self._store) > 1:
            victim = self._policy.evict()
            removed = self._store.pop(victim, None)
            self._expires.pop(victim, None)
            self._bytes -= self._sizer(removed)
            self.stats.evictions += 1
```

### 7.2 "Add a second-chance / 2Q policy"

**The follow-up:** "A one-off scan is evicting my hot data." **This is the real weakness of LRU**, and naming it is the answer: a sequential scan of a large table touches every key once, and those keys become the most recently used, evicting the working set. 2Q and ARC exist for this.

```python
class TwoQueuePolicy(EvictionPolicy):
    """2Q: new keys enter a PROBATIONARY queue. Only on a SECOND access do they
    get promoted to the PROTECTED queue. A one-off scan therefore churns
    probation and never touches the protected working set.

    This is the fix for the scan problem, and the cost is one extra level of
    bookkeeping — plus the fact that a genuinely-hot key is now evicted on its
    first miss after a restart."""

    def __init__(self, protected_ratio: float = 0.8) -> None:
        if not 0.0 < protected_ratio < 1.0:
            raise ValueError("protected_ratio must be in (0, 1)")
        self._protected_ratio = protected_ratio
        self._probation = LruPolicy()
        self._protected = LruPolicy()
        self._seen: set[str] = set()

    def _rebalance(self) -> None:
        """Keep the protected queue within its share by demoting its LRU end."""
        total = len(self._probation) + len(self._protected)
        target = int(total * self._protected_ratio)
        while len(self._protected) > target and len(self._protected) > 0:
            demoted = self._protected.evict()
            self._probation.key_inserted(demoted)

    def key_inserted(self, key: str) -> None:
        self._probation.key_inserted(key)

    def key_used(self, key: str) -> None:
        if key in self._protected:
            self._protected.key_used(key)
            return
        # A second access: promote out of probation.
        self._probation.key_removed(key)
        self._protected.key_inserted(key)
        self._seen.add(key)
        self._rebalance()

    def key_removed(self, key: str) -> None:
        self._probation.key_removed(key)
        self._protected.key_removed(key)
        self._seen.discard(key)

    def evict(self) -> str:
        if len(self._probation):
            return self._probation.evict()
        if len(self._protected):
            return self._protected.evict()
        raise PolicyError("evict called on an empty policy")

    def __len__(self) -> int:
        return len(self._probation) + len(self._protected)
```

### 7.3 "Make it distributed"

**The follow-up:** "Now the cache is shared across ten services."

**This is where this file stops and the HLD files begin.** The answer is not a class change; it is a different set of problems: a shared Redis tier with its own eviction policy, hot-key skew (one key on one shard), cache stampedes (a popular key expires and ten thousand requests hit the database — fixed by a lock or by probabilistic early expiry), and consistency between the cache and the source of truth. **The right move is to say that explicitly** and point at [caching strategies and layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md) and the [Redis deep dive](../../HLD/03-scaling/03-redis-deep-dive.md), rather than pretending a `DistributedCache` class is the answer.

```python
class StampedeProtectedCache:
    """The one class-level piece that does belong here: single-flight. On a
    miss, ONE caller loads the value while the others wait for it — so a
    popular key's expiry causes one database read, not ten thousand."""

    def __init__(self, cache: Cache, loader: "Callable[[str], object]") -> None:
        self._cache = cache
        self._loader = loader
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def get_or_load(self, key: str) -> object:
        value = self._cache.get(key)
        if value is not MISS:
            return value

        with self._lock:
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                i_am_the_loader = True
            else:
                i_am_the_loader = False

        if not i_am_the_loader:
            # Someone else is loading this key. Wait for them rather than
            # issuing a duplicate load — this is single-flight.
            if not event.wait(timeout=5.0):
                raise TimeoutError(f"timed out waiting for {key} to load")
            result = self._cache.get(key)
            if result is MISS:
                raise LookupError(f"{key} failed to load")
            return result

        try:
            loaded = self._loader(key)
            self._cache.put(key, loaded)
            return loaded
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()                       # release every waiter, always
```

### 7.4 "Add statistics and a hit-rate target"

Already present as `CacheStats`, and the interesting part is the follow-up question: *what hit rate do you need?* A cache with a 60 % hit rate on a 10 ms database read costs 4 ms of average latency, plus the miss path's own overhead — which may be a net loss if the miss path is not the bottleneck. **Saying "a cache is a bet that the hit rate justifies the write path and the staleness" is the senior answer**, and it belongs with the HLD treatment.

---

## 8. Failure modes and edge cases

### 8.1 Failure modes

| # | Failure | Consequence | Handling |
|---|---|---|---|
| 1 | Policy and store drift apart | `evict()` returns a ghost key; size is wrong | **Every** exit route calls `_remove`, which calls `key_removed` |
| 2 | Update does not refresh recency | A hot key is evicted as if cold | `put` on an existing key calls `key_used` |
| 3 | Concurrent `put` when full | Two evictions for one insert | Sharding; per-shard lock covers the length check and the evict |
| 4 | Concurrent `get` corrupts the list | Cycle; infinite loop | Sharding; the lock covers the pointer work |
| 5 | `evict()` on an empty policy | Exception, or worse a ghost | Guarded; the cache only evicts when `len >= capacity` |
| 6 | `None` stored as a value | Indistinguishable from a miss | The `MISS` sentinel |
| 7 | TTL entries occupy capacity after expiry | Live keys evicted in favour of dead ones | Lazy expiry on read **and** `purge_expired` |
| 8 | Capacity zero or negative | Every `put` evicts, or a crash | Rejected at construction |
| 9 | Shard count greater than capacity | Empty shards; a divide-by-zero flavour of bug | Rejected at construction |
| 10 | LFU frequency overflow | Unbounded int growth | Saturating `max_freq` (Redis uses 8-bit counters with probabilistic increment) |
| 11 | LFU cold-start | A genuinely hot key is evicted before it has frequency | The 2Q/probation fix (§7.2); this is LFU's real weakness |
| 12 | LRU scan pollution | One sequential scan evicts the working set | 2Q or ARC (§7.2); this is LRU's real weakness |
| 13 | `hash()` randomised per process | Shard assignment changes across restarts | `sha256` for the shard index |
| 14 | Cache stampede | A popular expiry flattens the database | Single-flight (§7.3) |
| 15 | Stale value after the source changes | Wrong data served | Invalidation on write, or a short TTL — and it belongs to the writer, not the cache |

### 8.2 Edge cases worth naming

- **Capacity 1.** Every `put` of a new key evicts the only entry. Legal, and a good test.
- **Repeated `put` of the same key when full.** No eviction should occur — the key is already present.
- **`get` on a key that expired one microsecond ago.** A miss, and the entry is removed.
- **Evicting the key that was just inserted** — with capacity 1, this happens, and the code must not evict the entry it is about to need.
- **LFU where all keys have the same frequency.** The recency tie-break decides, and it must be deterministic.
- **A zero-frequency key.** Impossible: a new key starts at 1. (Some implementations start at 0, which makes `min_freq` handling harder for no benefit.)
- **A `get` on a key concurrently deleted.** A miss is the correct answer, not an exception.

### 8.3 Complexity, stated precisely

| Structure | `get` | `put` | `evict` | Space |
|---|---|---|---|---|
| **Dict alone** | O(1) | O(1) | — (no order) | O(n) |
| **List alone** | O(n) | O(1) | O(1) | O(n) |
| **LRU: dict + doubly linked list** | **O(1)** | **O(1)** | **O(1)** | O(n) — 3 pointers + a dict entry per key |
| **LFU: freq buckets + ordered sets** | **O(1)** | **O(1)** | **O(1)** | O(n) + O(distinct frequencies) |
| **LFU, `key_removed` when the min tier empties** | O(distinct freqs) | — | — | — |
| **2Q** | O(1) | O(1) | O(1) | O(n) across two lists |
| **Byte-bounded** | O(1) | **O(evicted)** | O(1) each | O(n) |

**The line to memorise:** *"LRU and LFU are both O(1) for get, put and evict; the byte-bounded variant is O(1) for get and O(evicted) for put, and the LFU minimum-recompute is O(distinct frequencies)."*

### 8.4 Why the sentinel nodes matter, concretely

Without sentinels, `move_to_front` and `evict` each need special cases:

```python
# WITHOUT sentinels — four branches, and each is an eviction bug waiting:
def move_to_front_branchy(self, node):
    if node is self.head:
        return
    if node.prev: node.prev.next = node.next
    if node.next: node.next.prev = node.prev
    if node is self.tail: self.tail = node.prev
    node.next = self.head
    node.prev = None
    if self.head: self.head.prev = node
    self.head = node
    if self.tail is None: self.tail = node

# WITH sentinels — four assignments, no branches, nothing to get wrong:
def _unlink(self, node):
    node.prev.next = node.next
    node.next.prev = node.prev
```

**The cost is two dummy nodes.** The benefit is that three of the four classic LRU pointer bugs become unrepresentable. Worth saying out loud in the interview, because it shows the reasoning behind a detail rather than a memorised implementation.

---

## 9. Design patterns used, and why

| Pattern | Where | Why here | What it costs |
|---|---|---|---|
| **Strategy** | `EvictionPolicy` → `LruPolicy`, `LfuPolicy`, `TwoQueuePolicy` | The eviction rule is the axis that varies, and the interviewer's follow-up is exactly this | An interface, N classes, a wiring point |
| **Factory Method** | `build_cache` | One place knows the concrete policies; callers name a string | A function that must know every concrete type |
| **Template Method** *(rejected)* | — | Considered for a `BaseCache` with a `get`/`put` skeleton; rejected because the policy seam already gives the variability, and inheritance would add coupling for nothing | — |
| **Decorator** *(avoided)* | — | A wrapping `LockedCache` is the obvious shape and is the wrong one — thread safety belongs in the structure, not in a wrapper (§6.2) | — |
| **Proxy** | A cache in front of a store, generally | The cache is transparent except for the hit rate | Staleness; the proxy must offer a bypass |
| **Singleton** *(rejected)* | — | A cache as a singleton breaks every test that wants a fresh one and every process that wants two | — |
| **Sentinel nodes** | Not a pattern — an idiom | Removes four branches from the hot path | Two dummy objects |

**The two rejections are the interesting part.** A `LockedCache` decorator is what most candidates propose, and explaining *why it is wrong* — a global exclusive lock on a logically-read operation, and worse, an LRU cannot have a lock-free read because reading mutates the order — is a stronger answer than the decorator would have been.

---

## 10. Trade-offs

| Choice | Buys | Costs |
|---|---|---|
| Policy holds keys, cache holds values | Swap LRU↔LFU with one constructor argument | Two structures to keep in sync, and the drift bug if `key_removed` is forgotten |
| Doubly linked list with sentinels | O(1) everything; no null branches | Two dummy nodes; O(n) space overhead of three pointers per key |
| `OrderedDict` for the LFU tier | Less code; O(1) `move_to_end` and `popitem` | Relies on an implementation detail of the language; a hand-rolled list is more portable |
| Sharding instead of one lock | Near-linear scaling; unrelated keys never contend | Per-shard capacity and per-shard eviction are approximations |
| Exclusive lock on `get` | Correct recency tracking | No concurrent reads — intrinsic to LRU, not a flaw in the implementation |
| LFU recomputes `min_freq` on removal | Simple, obviously correct | O(distinct frequencies) on that path; a lazy `min_freq` is O(1) but subtler |
| LFU saturating frequency counter | Bounded memory; no overflow | Hot keys stop separating after the cap (Redis's exact trade) |
| `MISS` sentinel | `None` is storable | One concept to explain |
| TTL as a `Cache` argument | One class, no subclass | Expiry state per entry; a second eviction trigger |
| Lazy expiry **and** purge | Correct immediately; no dead entries occupying capacity | Two code paths |
| `sha256` shard index | Stable across processes and restarts | ~1 µs per lookup vs ~50 ns for `hash()`; irrelevant against the lock |
| Rejecting capacity 0 | No degenerate behaviour | A caller cannot build an accidental no-op cache |

**What changes at 10× load:** nothing in the data structures. Capacity grows, shards grow with the core count, and the cache stops being the answer — at some point the hit rate is bounded by the working set and the fix is a bigger working set, not a better eviction policy. The class design does not change; the deployment does.

---

## 11. Interview Q&A

**Q1 (easy). Why is LRU O(1) when a list is O(n) to search?**
Because there are two structures, each doing what it is good at. A dict maps key → node, giving O(1) lookup; a doubly linked list holds the order, and because the dict gives you the *node* rather than an index, unlinking and relinking it is four pointer writes with no search. Neither structure alone gives O(1) get and O(1) evict — the combination does, and that combination is the answer to the question.

**Q2 (easy). Why the sentinel nodes?**
They remove every special case. Without them, moving a node to the front and evicting the tail each need branches for "is this the head?", "is this the tail?", "is the list empty?" — and three of the four classic LRU pointer bugs live in those branches. With a head and a tail sentinel, `unlink` is `node.prev.next = node.next; node.next.prev = node.prev` with no conditionals, and eviction is `tail.prev`. The cost is two dummy objects.

**Q3 (easy). What is the difference between LRU and LFU?**
LRU evicts the least recently used; LFU evicts the least frequently used. LRU is better when recency predicts the future — a working set that moves. LFU is better when frequency does — a small hot set that stays hot. LRU has a scan problem: one sequential pass over a large table makes every scanned key "recent" and evicts the real working set. LFU has a cold-start problem: a genuinely hot key is evicted before it has accumulated frequency. Both are O(1) with the right structure.

**Q4 (medium). How does LFU achieve O(1)?**
Three things. A map from key to its frequency, so a `get` finds the tier in O(1). A map from frequency to a bucket of keys, where the bucket is insertion-ordered so the LRU tie-break is O(1). And a single integer, `min_freq`, naming the lowest non-empty tier, so eviction never scans for the minimum. The one subtlety is maintaining `min_freq`: on a promote it only changes when the bottom tier empties, and on removal it needs a recompute — which is O(distinct frequencies), not O(1), and I would say so rather than claim otherwise.

**Q5 (medium). Why is one global lock the wrong answer?**
Three reasons. It is global, so two requests for unrelated keys contend. It is exclusive even for `get`, and it has to be, because an LRU `get` mutates the recency order — so there is no read concurrency at all. And it is held across the policy's pointer work and the dict update, so it is not a few instructions. The result is that the cache that exists to be fast becomes the bottleneck, and under a read-heavy load it can be worse than the database it is fronting. The fix is sharding: N independent caches with N locks, so unrelated keys never contend.

**Q6 (medium). What does sharding cost you?**
Two approximations. Capacity becomes per-shard, so the total is per-shard capacity times the shard count, and a key cannot overflow its shard's budget even if other shards are empty. And eviction is per-shard, so the globally least-recently-used key may survive while a more recently used key in a busy shard is evicted. Both are standard and both are worth stating, because an interviewer who finds them for you has found a gap you left.

**Q7 (medium). What is a cache stampede, and how do you fix it at the class level?**
A popular key expires, and every concurrent request for it misses at the same instant and hits the origin — so one expiry becomes ten thousand database reads. The class-level fix is single-flight: the first caller to miss marks the key as in-flight and loads it; every other caller waits on an event and then reads the loaded value. One load instead of ten thousand. At the distributed level the same idea uses a lock in Redis, or probabilistic early expiry so keys do not all expire together, and that part belongs in the caching and Redis files, not here.

**Q8 (medium). What is the classic LRU bug?**
`put` on a key that already exists not refreshing its recency. An update is a *use* — the key was just written — and if it is not promoted, a frequently rewritten key drifts to the tail and gets evicted as if it were cold. The fix is one line: detect the existing key and call `key_used` rather than `key_inserted`. It is the bug I would write a test for first, because it is invisible until a key that is being updated constantly disappears.

**Q9 (medium). *Pushback:* "Your `LockedCache` wrapper is what I would write. Why is sharding better?"**
Because the wrapper is correct and slow in a way that defeats its own purpose. It takes an exclusive lock on `get`, and it has no choice: an LRU `get` moves a node, so it is a write. That means every read serialises with every other read and every write across the whole cache. Sharding gives the same correctness with N locks, so unrelated keys never contend, and the shard index is a hash of the key so the assignment is deterministic. The cost is per-shard capacity and per-shard eviction, which are approximations I would state up front.

**Q10 (medium). *Pushback:* "Why does the policy store only keys and not values? Isn't that indirection?"**
It is indirection with a purpose: it is what makes the eviction policy swappable with one constructor argument. If the policy held the values, `LruPolicy` and `LfuPolicy` would each have to own a store, and the cache would have to hand its data over on construction and take it back on a policy change. Keeping values in the cache and order in the policy means `Cache` is one class and the policy is a Strategy — the Open/Closed payoff is swapping LRU for 2Q without touching a line of the cache.

**Q11 (hard). *Pushback:* "You said O(1), but your LFU recomputes the minimum frequency. Isn't that a lie?"**
It is a qualified O(1) and I would state the qualification rather than hide it. `get`, `put` and `evict` are all O(1), including the promote path which only touches `min_freq` when the bottom tier empties. The exception is `key_removed` when the lowest tier becomes empty: it recomputes the minimum in O(number of distinct frequencies). That is bounded by the number of distinct frequency values, which is far smaller than the cache size in practice, and it is the path Redis avoids by letting `min_freq` go stale and advancing it lazily in the eviction path. If you wanted strict O(1) on every path, that lazy version is the fix, at the cost of a subtler invariant.

**Q12 (hard). *Pushback:* "If your cache is sharded, the eviction is not really LRU. Isn't that a correctness problem?"**
It is an approximation, not a correctness problem, and the distinction matters. LRU is a *heuristic* about which key is least valuable — it is not a correctness invariant, because any key can be evicted at any time and every caller must handle a miss. Per-shard eviction means a key in a busy shard may be evicted while a slightly-less-recent key in a quiet shard survives. The hit rate drops fractionally; nothing breaks. The contrast is with something like a seat hold, where "at most one holder" *is* an invariant and an approximation would be a double booking. Knowing which of the two you are approximating is the point of the question.

**Q13 (medium). How do you test an LRU?**
Three layers. The ordering test: insert A, B, C, get A, insert D, and assert both that B is evicted *and* that the full order is now D, A, C — asserting the order rather than only the victim catches promotion bugs that a victim-only assertion misses. The boundary tests: capacity 1, updating an existing key when full, a get on a just-evicted key, a get on an expired key. And a randomised differential test that drives thousands of random inserts, uses, removes and evictions while checking the size against a model list and asserting the list/index invariants after every step — that is what catches a pointer bug in `_unlink` that hand-written cases do not reach.

**Q14 (medium). When would you use LFU over LRU?**
When the popularity distribution is stable and skewed: a small set of keys accounts for most of the traffic and stays hot for a long time, which is what a CDN's popular-object set or a product catalogue's front page looks like. LRU is better when recency is the better predictor, which is most request-path caches. The honest framing is that both are heuristics and the real answer is to measure the hit rate — which is why the stats object is in the design from the start rather than added later. And if you have the budget, 2Q or ARC gets most of LFU's benefit without its cold-start problem.

**Q15 (hard). What would you do differently at 10× the load?**
Nothing in the data structures — the complexity guarantees hold at any size. The changes are all structural: more shards sized to the core count, a larger capacity, and eventually the recognition that the cache has stopped being the answer. At some point the hit rate is bounded by the working set, and no eviction policy fixes a working set that does not fit; the fix is a bigger tier, a tiered cache with a smaller fast layer and a larger slow one, or accepting the origin read. The class design does not change at 10×; the deployment does, and knowing which of those two is being asked is most of the answer.

---

## 12. Cheat-sheet summary

**The two structures**

```
  LRU:  dict[key] -> Node   +   doubly linked list (head sentinel → MRU … LRU → tail sentinel)
        get → unlink, push_front (4 pointer writes)      O(1)
        put → insert front; if over capacity, evict tail.prev   O(1)

  LFU:  dict[key] -> freq   +   dict[freq] -> OrderedDict of keys   +   min_freq
        get → promote key freq → freq+1; update min_freq if the bottom emptied   O(1)
        evict → popitem(last=False) from bucket[min_freq]      O(1)
```

**Why they work**

| Structure | get | put | evict | Note |
|---|---|---|---|---|
| Dict alone | O(1) | O(1) | — | No order |
| List alone | O(n) | O(1) | O(1) | No lookup |
| **LRU (both)** | **O(1)** | **O(1)** | **O(1)** | Dict holds a *node* pointer, not an index |
| **LFU (buckets + min_freq)** | **O(1)** | **O(1)** | **O(1)** | Except `key_removed` min-recompute |

**The invariants to assert**

1. `len(policy) == len(cache._store)` — the policy and the store never drift.
2. List and index agree, forward and backward (`_check_invariants`).
3. `len(cache) <= capacity`, always — including under concurrency.
4. A `get` always leaves its key at the MRU end (LRU) or one tier up (LFU).
5. Every exit route — evict, delete, TTL expiry, purge — calls the policy's `key_removed`.

**The concurrency answer**

```
  WRONG:  one global exclusive lock; it makes get a write and serialises everything
  RIGHT:  sharding — N caches, N locks; unrelated keys never contend
  COST:   per-shard capacity and per-shard eviction are approximations
  NOTE:   an LRU CANNOT have a lock-free read — reading mutates the order
```

**Numbers**

| Parameter | Value |
|---|---|
| Shard count | 16–256, sized to cores |
| Lock scope | One per shard, held for the whole operation |
| `OrderedDict` vs hand-rolled list | Either; the list is more portable |
| LFU frequency cap | Saturating (Redis uses 8-bit + probabilistic increment) |
| TTL purge interval | 1–5 min, batch 500 |
| Single-flight wait timeout | 2–5 s |
| Capacity | Must be ≥ shard count |

**Variant decision table**

| Requirement | Policy |
|---|---|
| Recency predicts the future | **LRU** |
| Stable, skewed popularity | **LFU** |
| A scan must not evict the working set | **2Q / ARC** |
| Byte-bounded values | Size-aware eviction, `put` becomes O(evicted) |
| Entries expire | TTL on the cache: lazy expiry + purge |
| Shared across services | Not a class change — a shared tier (see HLD) |

**"If the interviewer says X → say Y"**

| Interviewer says | You say |
|---|---|
| "How is it O(1)?" | "Dict gives lookup, list gives order, and the dict stores the *node* so unlink is four writes." |
| "Why sentinels?" | "They remove the head/tail/empty branches, which is where the pointer bugs live." |
| "Make it thread-safe" | "Sharding — N caches and N locks. One global lock makes `get` a serialised write." |
| "Is sharded eviction still LRU?" | "It is an approximation, not a broken invariant — any key can be evicted, so callers already handle misses." |
| "Now add LFU" | "A second `EvictionPolicy`; the cache is untouched — that is why the policy holds keys and not values." |
| "A scan is evicting my hot data" | "That is LRU's real weakness — 2Q with a probationary tier fixes it." |
| "Make it distributed" | "Different problem — shared tier, hot keys, stampedes. The class design does not change." |
| "Prove it" | "Ordering assertions, not just the victim; plus a randomised differential test with the invariants checked every step." |

---

## Cross-links

- **The distributed half of this topic** (owned elsewhere, and the right place to go next): [Caching strategies and layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md) · [Redis deep dive](../../HLD/03-scaling/03-redis-deep-dive.md) · [Scaling and caching cheat sheet](../../Cheat-Sheets/03-scaling-caching-db-cheatsheet.md)
- **The existing LLD treatment inside a case study** — "designing an LRU cache" and "designing an LFU cache" as digressions in a booking argument: [BookMyShow seat booking](../../Case-Studies/HLD/07-bookmyshow-seat-booking.md)
- The class/code concepts behind this design: [OOP principles and SOLID](../02-oop-principles-and-solid.md) · [Design patterns](../03-design-patterns.md) · [Concurrency in LLD](../04-concurrency-and-threading-in-lld.md) · [UML and class diagrams](../05-uml-and-class-diagrams.md) · [The machine-coding framework](../06-machine-coding-interview-framework.md) · [What is LLD](../01-what-is-lld-and-how-to-answer.md)
- The boundary between the class view and the system view: [HLD vs LLD](../../HLD/01-foundations/03-hld-vs-lld.md)
- Other problems in this pillar: [Parking lot](01-parking-lot.md) · [Hotel management](03-hotel-management.md) · [BookMyShow seat locking](04-bookmyshow-seat-locking.md) · [Elevator system](05-elevator-system.md) · [Splitwise](06-splitwise.md) · [ATM machine](07-atm-machine.md) · [Library management](08-library-management.md)
