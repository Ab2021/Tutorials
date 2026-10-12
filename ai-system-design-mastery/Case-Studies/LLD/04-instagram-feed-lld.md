# Instagram News Feed — LLD

> **Level.** This is the **LLD bridge** for the news-feed case study. The HLD in [`../HLD/04-instagram-news-feed.md`](../HLD/04-instagram-news-feed.md) decides the system-level answers: that fan-out happens on write for most authors and on read for celebrities, that the timeline is a cache, that media goes through a separate pipeline, that ranking is a service. **None of that is repeated here.** This file is the layer below: the classes, the policies, the pipeline stages and the cursor that make those answers true.
>
> **Grounding note.** The reference repositories contain **no OOP/LLD material**, and the source transcripts cover feeds at the system level without teaching a class design for one. Everything here — the per-author fan-out policy, the merge of pushed and pulled candidates, the stage pipeline, the canonical-order cursor — is standard industry practice assembled for this problem. Where it aligns with a property the HLD states, it cross-links rather than restates.
>
> **Technique.** The 12-section shape, the test discipline and the pattern vocabulary come from [`../../LLD/01-what-is-lld-and-how-to-answer.md`](../../LLD/01-what-is-lld-and-how-to-answer.md) and the worked problems in [`../../LLD/problems/`](../../LLD/problems/). The pagination treatment deliberately contrasts with [`../../LLD/problems/04-bookmyshow-seat-locking.md`](../../LLD/problems/04-bookmyshow-seat-locking.md), which is about holding a resource rather than walking a list.

---

## TL;DR

> **The timeline is a cache, and the cursor must be over an order that does not move — which is why the cursor is (created_at, post_id) and ranking happens within the page, not across it. A cursor over a recomputed score skips and duplicates posts that a reader can see.**

Three sentences that carry it:

1. **Fan-out is per author, decided at write time, and the decision is a policy object.** A user with 50M followers is never pushed; their posts are pulled at read time and merged with the pushed timeline. The read path therefore has *two* sources, and it must dedupe between them.
2. **The materialised timeline is derived data.** It can be truncated, rebuilt and wrong — so every read validates against the source of truth for deletion, and a tombstone is what makes a deleted post disappear from a timeline that still lists it.
3. **Ranking is a pipeline of stages, not a function.** Candidate generation, filtering, scoring and truncation each change for different reasons; collapsing them into one `score(post)` is what makes a feed impossible to debug when a post shows up that nobody wanted.

---

## The 60-second answer

"Four things. **`FollowGraph`** — indexed both directions, because fan-out needs the followers of an author and the pull path needs the followees of a viewer. **`FanOutPolicy`** — consulted at post time and returning `PUSH`, `PULL` or `PUSH_TO_ACTIVE`; the threshold is a follower count and the policy is a class so it can change without touching the writer. **`TimelineStore`** — the materialised cache, entries of `(created_at, post_id)`, truncatable and rebuildable, with an explicit `remove_post` because deleting a post has to reach a timeline that already lists it. **`FeedService.get_feed`** — merge the pushed entries with the pulled candidates from followees whose posts were never fanned out, dedupe by post id, drop tombstones and seen posts, sort into the canonical order, take the page strictly after the cursor, and only then run the ranking pipeline *within the page*. That last ordering is the design: a cursor over a recomputed score skips and duplicates, because the score moves between page fetches and the cursor cannot know."

---

## 1. Requirement clarification

### Questions to ask, and the default to assume if the interviewer waves them through

| # | Question | Default to state and move on |
|---|---|---|
| 1 | Scope: the feed read path, or writing and media too? | All three, with media as an id list — the upload pipeline is named and excluded |
| 2 | Ranked or reverse-chronological? | **Both, and they are different code paths.** Chronological is the fallback; ranked is the default |
| 3 | Fan-out on write or on read? | **Per author.** Pushed below a follower threshold, pulled above it |
| 4 | Is the timeline durable? | No. It is a cache and can be rebuilt from the follow graph plus posts |
| 5 | What is the page size, and is offset or cursor pagination used? | 20, and **cursor** — offset is the bug the HLD names |
| 6 | Do deletes need to disappear from an existing timeline? | Yes, immediately for the author. This is what forces a tombstone |
| 7 | Are seen posts filtered? | Yes, and **before** pagination, or a page comes back empty while more posts exist |
| 8 | Is the ranking deterministic? | Yes, given the same inputs — and the tie-break is `post_id`, so two runs agree |
| 9 | How fresh must a new post be in a follower's feed? | Under a second for a normal author; the pull path makes a celebrity's post immediate too |
| 10 | Does the follow graph need to be consistent with the timeline? | No. A follow/unfollow race produces a briefly wrong timeline, and that is accepted |

### The four constraints that shape the design

1. **One author can break the write path.** A post from an account with 50M followers cannot produce 50M timeline writes, so the fan-out decision must be made per author at write time — and the read path must then cope with a candidate source it did not have before.
2. **The timeline is derived, so it can be wrong.** Anything that changes the underlying facts — a delete, a follow, a block — makes a materialised timeline stale, and the read path has to repair it rather than assume it.
3. **A cursor over a moving order is broken.** If the cursor's key is recomputed per request, an insert between two page fetches duplicates or skips. This is the single most common production bug in a feed, and the class model has to prevent it structurally.
4. **A feed has four concerns with four change rates.** What is eligible, what is removed, what is ordered, and how much. One method that does all four is untestable.

### Non-goals

Media upload and transcoding, the ML model behind the affinity score (assumed given as a table), ads, stories, and the client. Named so the interviewer knows they were considered.

---

## 2. Core entities & responsibilities

| Entity | Responsibility — one line | Collaborators |
|---|---|---|
| `Post` | Immutable: author, time, media ids, caption | — |
| `PostStore` | Posts by id, per author; **tombstones** on delete | `Post` |
| `FollowGraph` | Both directions: followers and following | — |
| `FanOutMode` | `PUSH` · `PULL` · `PUSH_TO_ACTIVE` | — |
| `FanOutPolicy` | Decide the mode for one author at post time | `FollowGraph` |
| `TimelineEntry` | `(created_at, post_id)` — the canonical order | — |
| `TimelineStore` | The materialised cache: push, read, truncate, remove, rebuild | `TimelineEntry` |
| `FanOutService` | Execute the policy: push to timelines, or record for pulling | `TimelineStore`, `FanOutPolicy` |
| `Stage` | One ranking concern over a candidate list | — |
| `RankingPipeline` | Ordered stages, with a deterministic tie-break | `Stage` |
| `AffinityStore` | Precomputed, slow-moving viewer↔author features | — |
| `SeenStore` | What this viewer has already been shown | — |
| `Cursor` | `(created_at, post_id)`, encoded for the wire | — |
| `FeedPage` | The page and the next cursor | `Cursor` |
| `FeedService` | The façade: merge, repair, paginate, rank | all of the above |

**The sentence to say out loud:** *"`FanOutPolicy` and `Cursor` are the two classes people leave out, and they are the two that make the design correct — the policy is what stops one account from breaking the write path, and the cursor is what stops the read path from skipping posts."*

### HLD component → its classes

| HLD component (in [`../HLD/04-instagram-news-feed.md`](../HLD/04-instagram-news-feed.md)) | Its classes here | The class-level question the HLD does not answer |
|---|---|---|
| Fan-out on write | `FanOutService.fan_out`, `TimelineStore.push` | How many writes, and what happens past the bound? |
| Fan-out on read | `FanOutService.pull_candidates` | Where do pulled candidates merge with pushed ones, and how are duplicates removed? |
| The celebrity problem (§6.1) | `FanOutPolicy`, `FanOutMode.PULL` | Where is the threshold decided, and can it change without touching the writer? |
| The timeline cache | `TimelineStore`, `TimelineEntry` | What is the entry, what is the order, and can it be rebuilt? |
| Ranking | `RankingPipeline`, `Stage`, `AffinityStore` | Which concern is each stage, and what makes two runs agree? |
| Pagination (§6.9) | `Cursor`, `FeedPage` | What is the cursor over, and what happens if the order moves? |
| Deleted posts lingering (§6.8) | `PostStore` tombstones, `TimelineStore.remove_post`, the read-path check | A timeline that already lists a deleted post |
| The N+1 hydration bug (§6.7) | `PostStore.get_many` | Batch by id, once, at the end |
| The follow/unfollow race (§6.5) | `FollowGraph` and the tolerated staleness | What is allowed to be briefly wrong? |
| The dormant user returning (§6.6) | `TimelineStore.rebuild` | An empty timeline is a rebuild, not an error |
| Hot keys (§6.4) | `TimelineStore` sharding key (§6) | One user's timeline on one partition |

---

## 3. Class diagram (ASCII)

```
                      ┌──────────────────────────┐
                      │  Post  «value, FROZEN»   │
                      │  id, author_id           │
                      │  created_at, media_ids   │
                      │  caption                 │
                      └────────────▲─────────────┘
                                   │ stored in
   ┌───────────────────────────────┴──────────────────────────────────┐
   │  PostStore                                                       │
   │  by_id: {post_id -> Post}                                        │
   │  by_author: {author_id -> [(created_at, post_id)]}   ← the PULL source
   │  tombstones: {post_id -> deleted_at}                 ← NOT a delete   │
   │  +get_many(ids) -> {id: Post}     ← the N+1 fix, one batch call    │
   │  +recent_by_author(author, since, limit)                           │
   │  +delete(post_id)                 ← a tombstone, so readers see it │
   └───────────────────────────────▲──────────────────────────────────┘

   ┌───────────────────────┐      ┌──────────────────────────────────┐
   │  FollowGraph          │      │  FanOutPolicy «interface»        │
   │  following: {u -> set}│      │  +decide(author, followers,      │
   │  followers: {u -> set}│─────►│          stats) -> FanOutMode    │
   │  (BOTH directions)    │      ├──────────────────────────────────┤
   └───────────────────────┘      │  ThresholdFanOut                 │
                                  │   followers < T  -> PUSH         │
                                  │   followers >= T -> PULL         │
                                  │   and PUSH_TO_ACTIVE for the     │
                                  │   active subset, if configured   │
                                  └──────────────┬───────────────────┘
                                                 │ consulted by
   ┌─────────────────────────────────────────────┴────────────────────┐
   │  FanOutService                                                   │
   │  +fan_out(post) -> FanOutResult                                  │
   │     PUSH : write one TimelineEntry into each follower's timeline │
   │            (bounded by max_fanout; past it, downgrade to PULL)   │
   │     PULL : write NOTHING; the post is found via by_author        │
   │  +pull_candidates(viewer, since, limit) -> [Post]                │
   └──────────────┬───────────────────────────────┬───────────────────┘
                  │                               │
                  ▼                               ▼
   ┌──────────────────────────────┐   ┌───────────────────────────────┐
   │  TimelineStore               │   │  (pull candidates come from   │
   │  {user -> [TimelineEntry]}   │   │   PostStore.by_author over    │
   │  entries are (created_at,    │   │   FollowGraph.following)      │
   │           post_id)           │   └───────────────────────────────┘
   │  SORTED DESC by that pair    │
   │  +push(user, entries)        │
   │  +read(user, after, limit)   │
   │  +remove_post(user, post_id) │  ← deletes must reach a timeline
   │  +truncate(user, keep)       │     that already lists the post
   │  +rebuild(user, entries)     │  ← the dormant user
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  FeedService.get_feed(viewer, cursor, limit)   THE READ PATH      │
   │  1. NOTE the viewer active (the signal PUSH_TO_ACTIVE reads)      │
   │  2. WALK the timeline from the cursor, in cursor order:           │
   │       hydrate each window in ONE batch   ← the N+1 fix            │
   │       drop TOMBSTONE / SEEN / OWN while walking ── not after      │
   │       STOP the moment the page is full   ← nothing over-fetched   │
   │  3. MERGE fanout.pull_candidates ── the celebrity half            │
   │       dedupe by post_id (a pushed post may also be pulled)        │
   │  4. rank WITHIN the page     ← NOT across it; see §10             │
   │  5. next_cursor = the OLDEST SERVED pair  ← not the last ranked   │
   │  6. mark SEEN what was returned   ← not what was scanned          │
   │  7. return FeedPage(items, next_cursor)                           │
   └──────────────┬───────────────────────────────────────────────────┘
                  │ runs
                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  RankingPipeline                                                 │
   │  stages, in order:                                               │
   │    EligibilityStage  is it visible to this viewer at all?        │
   │    SeenFilterStage   has this viewer seen it?                    │
   │    AffinityStage     reader↔author score, from AffinityStore     │
   │    RecencyStage      decay over age                              │
   │    MediaBoostStage   a product nudge                             │
   │    ScoreStage        combine into (score, post_id)               │
   │    TruncateStage     cut to the page size                        │
   │  DETERMINISTIC: the tie-break is post_id, so two runs agree      │
   └──────────────────────────────────────────────────────────────────┘

  Cursor = (created_at, post_id).  NOT a score.
  ─────────────────────────────────────────────
  A cursor over a recomputed score is broken: the score moves between
  page fetches, so a post can be returned twice or skipped entirely.
  (created_at, post_id) never moves — a post's creation time is fixed —
  so "strictly after this pair" is a total order that cannot drift.
```

**Read it this way:** the write path decides *where a post lands*, the read path decides *what a viewer sees*, and the cursor is the contract between two calls to the read path. The one thing that must not move is the order the cursor walks.

---

## 4. Interfaces / abstract classes / enums

```python
class FanOutMode(Enum):
    PUSH = "push"                     # write into every follower's timeline
    PULL = "pull"                     # write nothing; find it at read time
    PUSH_TO_ACTIVE = "push_to_active" # push only to followers seen recently


class FanOutPolicy(ABC):
    @abstractmethod
    def decide(self, author_id: str, follower_count: int,
               now: float) -> FanOutMode:
        """Called at POST time, once per post. Returning PULL is not a failure
        — it is the correct answer for an author whose fan-out would be
        enormous, and the read path is built to handle it."""


class Stage(ABC):
    @abstractmethod
    def apply(self, candidates: list["ScoredPost"], context: "RankContext"
              ) -> list["ScoredPost"]:
        """One concern. Stages compose in order and each is testable alone —
        which is the entire reason ranking is a pipeline rather than a
        `score(post)` function."""


class TimelineStore(ABC):
    @abstractmethod
    def push(self, user_id: str, entries: list["TimelineEntry"]) -> int:
        raise NotImplementedError

    @abstractmethod
    def read(self, user_id: str, after: Optional["Cursor"],
             limit: int) -> list["TimelineEntry"]:
        raise NotImplementedError

    @abstractmethod
    def remove_post(self, user_id: str, post_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def truncate(self, user_id: str, keep: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def rebuild(self, user_id: str, entries: list["TimelineEntry"]) -> int:
        raise NotImplementedError
```

**Four contract notes:**

- **`decide` returns PULL as a normal outcome.** The read path is built for it, so it is not an error and must not be signalled as one.
- **`remove_post` exists on the store.** A timeline that already lists a deleted post has to be repaired; a store with only `push` and `read` makes deletion impossible without a full rebuild.
- **`Stage.apply` takes and returns the whole list.** A stage can reorder, drop or annotate; narrowing it to a per-post function would forbid filtering, which is half the job.
- **`read` takes a cursor, not an offset.** An offset is a position in a list that changes under you.

---

## 5. Complete working code (Python 3.10+)

> **Language note.** In Java this is a `record` for `Post` and `TimelineEntry`, a `TreeMap` per user for the timeline, and an enum with a strategy for the policy. In Go, a `sort.Slice` over entries and a `(FeedPage, error)` return. The **per-author policy, the merge-and-repair read path, and the cursor over a fixed pair** are the design; the syntax is not.

```python
"""instagram_feed_lld.py — the classes inside the HLD's boxes.

Design summary
--------------
Fan-out         : decided PER AUTHOR at post time by a policy object. A user
                  with 50M followers is never pushed; the read path pulls
                  their posts and merges them.
The timeline    : DERIVED data. It can be truncated, rebuilt and stale, so the
                  read path repairs it rather than trusting it.
Deletes         : a tombstone in the PostStore, plus TimelineStore.remove_post.
                  A timeline that already lists a deleted post is the failure
                  the HLD names, and only an explicit removal fixes it.
The cursor      : (created_at, post_id) — an order that CANNOT move. A cursor
                  over a recomputed score duplicates and skips, because the
                  score between two page fetches is not the same score.
Pagination      : the cursor is applied BEFORE ranking. Ranking within the page
                  is a real limitation and §10 states it honestly.
Ranking         : a pipeline of stages, each one concern, each testable alone,
                  and deterministic because the tie-break is post_id.
Hydration       : one batched get_many at the end. The N+1 bug the HLD names
                  is a missing batch call, and this is where it would live.
"""
from __future__ import annotations

import base64
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional


# --------------------------------------------------------------------------
# 1. The post, and the store that keeps tombstones
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Post:
    id: str
    author_id: str
    created_at: float            # epoch seconds; the FIRST half of the cursor
    caption: str = ""
    media_ids: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.id} by {self.author_id}: {self.caption[:24]}"


class PostStore:
    """Posts by id, an ordered index per author — which IS the pull source —
    and a tombstone set.

    DELETION IS A TOMBSTONE, NOT A `del`. Removing the post from `by_id` would
    make it unhydratable, and a timeline that still lists it would silently
    return a hole — or, worse, an implementation that filters holes out would
    return a page of 19 items and no explanation. A tombstone lets the reader
    say "this is deleted" and repair the timeline as a side effect.
    """

    def __init__(self) -> None:
        self.by_id: dict[str, Post] = {}
        self.by_author: dict[str, list[tuple[float, str]]] = {}
        self.tombstones: dict[str, float] = {}
        self._lock = threading.RLock()
        self._counter = 0

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"post-{self._counter:08d}"

    def add(self, post: Post) -> Post:
        with self._lock:
            self.by_id[post.id] = post
            entries = self.by_author.setdefault(post.author_id, [])
            entries.append((post.created_at, post.id))
            entries.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
            return post

    def get(self, post_id: str) -> Optional[Post]:
        """Tombstone-aware, and deliberately so. `get_many` filters deleted
        posts, and a `get` that did not would be a second, weaker rule for the
        same question — the kind of inconsistency that produces a deleted post
        on one surface and not another. One rule: a deleted post is not readable
        through the ordinary read path. The row is still in `by_id` for
        moderation and for the tombstone's own timestamp."""
        with self._lock:
            if post_id in self.tombstones:
                return None
            return self.by_id.get(post_id)

    def get_many(self, post_ids: Iterable[str]) -> dict[str, Post]:
        """THE N+1 FIX. One call, one lock acquisition, one pass. A feed page
        of 20 posts hydrated with 20 individual `get` calls is the bug the HLD
        names, and it is invisible until the feed is slow and the post store is
        remote."""
        with self._lock:
            return {pid: self.by_id[pid] for pid in post_ids
                    if pid in self.by_id and pid not in self.tombstones}

    def is_deleted(self, post_id: str) -> bool:
        with self._lock:
            return post_id in self.tombstones

    def delete(self, post_id: str, now: float) -> bool:
        """Idempotent: deleting twice is not an error, because a retried delete
        request is normal."""
        with self._lock:
            if post_id in self.tombstones:
                return False
            if post_id not in self.by_id:
                raise KeyError(f"no such post: {post_id}")
            self.tombstones[post_id] = now
            return True

    def recent_by_author(self, author_id: str, since: float = 0.0,
                         limit: int = 50) -> list[Post]:
        """THE PULL SOURCE. Newest first, already in canonical order because
        `by_author` is maintained sorted by the same pair the cursor uses."""
        with self._lock:
            pairs = self.by_author.get(author_id, [])
            out: list[Post] = []
            for created_at, post_id in pairs:
                if created_at < since:
                    break                    # sorted, so nothing later qualifies
                if post_id in self.tombstones:
                    continue
                out.append(self.by_id[post_id])
                if len(out) >= limit:
                    break
            return out

    def author_post_count(self, author_id: str) -> int:
        with self._lock:
            return len(self.by_author.get(author_id, ()))


# --------------------------------------------------------------------------
# 2. The follow graph — indexed BOTH ways
# --------------------------------------------------------------------------

class FollowGraph:
    """Both directions, and both are load-bearing.

    `followers_of` is what fan-out needs: given an author, who must receive the
    post. `following_of` is what the pull path needs: given a viewer, whose
    posts must be fetched. A graph with only one direction forces the other
    question to be answered by a full scan.
    """

    def __init__(self) -> None:
        self._following: dict[str, set[str]] = {}
        self._followers: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self._revision = 0

    @property
    def revision(self) -> int:
        """Bumped on every change. A reader that wants to know whether the
        graph moved under it compares revisions — cheap, and it is what turns
        the follow/unfollow race into a detectable event rather than a mystery."""
        with self._lock:
            return self._revision

    def follow(self, follower_id: str, followee_id: str) -> bool:
        if follower_id == followee_id:
            raise ValueError("a user cannot follow themselves")
        with self._lock:
            outgoing = self._following.setdefault(follower_id, set())
            if followee_id in outgoing:
                return False
            outgoing.add(followee_id)
            self._followers.setdefault(followee_id, set()).add(follower_id)
            self._revision += 1
            return True

    def unfollow(self, follower_id: str, followee_id: str) -> bool:
        with self._lock:
            outgoing = self._following.get(follower_id)
            if outgoing is None or followee_id not in outgoing:
                return False
            outgoing.discard(followee_id)
            incoming = self._followers.get(followee_id)
            if incoming is not None:
                incoming.discard(follower_id)
            self._revision += 1
            return True

    def is_following(self, follower_id: str, followee_id: str) -> bool:
        with self._lock:
            return followee_id in self._following.get(follower_id, ())

    def followers_of(self, author_id: str) -> list[str]:
        with self._lock:
            return sorted(self._followers.get(author_id, ()))

    def following_of(self, viewer_id: str) -> list[str]:
        with self._lock:
            return sorted(self._following.get(viewer_id, ()))

    def follower_count(self, author_id: str) -> int:
        with self._lock:
            return len(self._followers.get(author_id, ()))

    def following_count(self, viewer_id: str) -> int:
        with self._lock:
            return len(self._following.get(viewer_id, ()))

    def stats(self) -> tuple[int, int]:
        with self._lock:
            edges = sum(len(v) for v in self._following.values())
            return len(self._following), edges


# --------------------------------------------------------------------------
# 3. Fan-out — the policy, and the service that executes it
# --------------------------------------------------------------------------

class FanOutMode(Enum):
    PUSH = "push"
    PULL = "pull"
    PUSH_TO_ACTIVE = "push_to_active"


class FanOutPolicy(ABC):
    @abstractmethod
    def decide(self, author_id: str, follower_count: int,
               now: float) -> FanOutMode:
        """Called once per post, at write time."""


class ThresholdFanOut(FanOutPolicy):
    """The whole celebrity problem, in eight lines.

    Below the threshold, push: the write amplification is bounded and the read
    path stays a single lookup. Above it, pull: the read path pays, but one post
    stops costing millions of writes. The threshold is a parameter because it is
    an operational number — it moves with the cost of a write and the cost of a
    read, and neither is a constant.
    """

    def __init__(self, push_below: int = 10_000,
                 push_to_active_below: int = 1_000_000,
                 active_window_s: float = 7 * 24 * 3600.0) -> None:
        if push_below < 0 or push_to_active_below < push_below:
            raise ValueError("thresholds must be ordered and non-negative")
        self.push_below = push_below
        self.push_to_active_below = push_to_active_below
        self.active_window_s = active_window_s

    def decide(self, author_id: str, follower_count: int,
               now: float) -> FanOutMode:
        if follower_count < self.push_below:
            return FanOutMode.PUSH
        if follower_count < self.push_to_active_below:
            return FanOutMode.PUSH_TO_ACTIVE
        return FanOutMode.PULL


@dataclass(frozen=True)
class FanOutResult:
    mode: FanOutMode
    timelines_written: int
    downgraded: bool = False     # the bound forced a PULL
    skipped_followers: int = 0   # PUSH_TO_ACTIVE wrote to the active subset

    def __str__(self) -> str:
        note = " (downgraded past the bound)" if self.downgraded else ""
        return (f"{self.mode.value}: {self.timelines_written} timelines"
                f"{note}")


class FanOutService:
    """Executes the policy and owns the bound.

    ONE SNAPSHOT OF THE FOLLOWER LIST drives three things: the count the policy
    decides on, the count the bound checks, and the list the fan-out writes to.
    That is deliberate and it was not always so. An earlier version read
    `follower_count(author)` for the decision and `followers_of(author)`
    separately for the write, which is two lock acquisitions and therefore two
    different answers — so a policy that said PUSH for 9,999 followers could
    write to a list that had grown to 200,000 in between, and the `max_fanout`
    check, having used the stale smaller number, would not stop it. The bound
    was guarding the wrong number. Reading the list once makes the decision and
    the write incapable of disagreeing.

    THE BOUND IS STILL WORTH HAVING, even though the snapshot is consistent: a
    follower list is a snapshot, and a policy bug would otherwise turn one post
    into an unbounded write storm. Past `max_fanout` the post is treated as PULL
    and marked `downgraded`, which is a signal worth alarming on rather than a
    silent fallback.
    """

    def __init__(self, graph: FollowGraph, timelines: "TimelineStore",
                 policy: FanOutPolicy, posts: PostStore,
                 max_fanout: int = 50_000,
                 active_window_s: float = 7 * 24 * 3600.0,
                 is_active: Optional[Callable[[str], bool]] = None) -> None:
        self.graph = graph
        self.timelines = timelines
        self.policy = policy
        self.posts = posts
        self.max_fanout = max_fanout
        self.active_window_s = active_window_s
        self.is_active = is_active

    def fan_out(self, post: Post, now: float) -> FanOutResult:
        # ONE read of the follower list. Everything below uses this snapshot, so
        # the decision and the write cannot disagree.
        followers = self.graph.followers_of(post.author_id)
        count = len(followers)
        mode = self.policy.decide(post.author_id, count, now)

        if mode is FanOutMode.PULL:
            return FanOutResult(mode, 0)

        if count > self.max_fanout:
            # The policy said push; the bound refuses. Say so, do not hide it.
            return FanOutResult(FanOutMode.PULL, 0, downgraded=True)

        if mode is FanOutMode.PUSH_TO_ACTIVE:
            followers = self._active_only(followers, now)

        entry = TimelineEntry(post.created_at, post.id)
        written = 0
        for follower_id in followers:
            written += self.timelines.push(follower_id, [entry])
        return FanOutResult(mode, written,
                            skipped_followers=max(0, count - len(followers)))

    def _active_only(self, followers: list[str], now: float) -> list[str]:
        """The middle mode: push only to followers who have been active inside
        the window. It bounds the write without giving up on the read path for
        everybody — at the cost of a reader who returns after a month missing
        recent posts from mid-tier accounts until their timeline is rebuilt.

        Activity arrives as a CALLABLE rather than a second store. That is a
        deliberate narrowing: the fan-out service needs one bit about a user,
        and handing it a store would give it a reason to start asking other
        questions. The caller owns the definition of "active" — sessions,
        opens, or a heartbeat — and this class never learns which.
        """
        if self.is_active is None:
            return followers
        return [f for f in followers if self.is_active(f)]

    def pull_candidates(self, viewer_id: str, since: float = 0.0,
                        per_author: int = 10) -> list[Post]:
        """The pull half of the read path: the newest posts of everybody the
        viewer follows whose posts might not have been pushed.

        It deliberately queries EVERY followee rather than only the celebrities.
        Knowing which authors are in PULL mode at read time would mean a second
        policy evaluation per author per request, and the policy's inputs move —
        an author can cross the threshold between the write and the read. A
        cheap `recent_by_author` per followee, capped, is simpler and cannot be
        wrong about which authors to ask."""
        found: list[Post] = []
        for followee in self.graph.following_of(viewer_id):
            found.extend(self.posts.recent_by_author(followee, since, per_author))
        return found


@dataclass(frozen=True, order=True)
class TimelineEntry:
    """`(created_at, post_id)`, and the ORDER OF THE FIELDS IS THE SORT ORDER.

    `created_at` alone is not a total order — two posts in the same millisecond
    would compare equal, and a cursor built on an equal pair cannot express
    "strictly after". The post id breaks the tie, and because `order=True`
    makes the tuple comparable, the dataclass IS the sort key."""
    created_at: float
    post_id: str


# --------------------------------------------------------------------------
# 4. The cursor — over an order that CANNOT move
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cursor:
    """`(created_at, post_id)`, encoded for the wire.

    WHY NOT A SCORE. A cursor over a recomputed ranking score is broken, and it
    is broken in a way that produces no error: request page 1, and the pipeline
    scores posts at time T. A new post arrives. Request page 2, and the pipeline
    scores the same posts at time T+1 — with a different score, because
    affinity and recency both moved. Now "everything with a score below this
    one" is a different set, and posts near the boundary come back twice or not
    at all. A post's creation TIME never changes, so a cursor over it is a
    total order that cannot drift. That is the whole reason it is this pair.
    """
    created_at: float
    post_id: str

    def encode(self) -> str:
        raw = f"{self.created_at:.6f}|{self.post_id}".encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def decode(token: str) -> "Cursor":
        if not token:
            raise ValueError("an empty cursor is not a cursor")
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        created, separator, post_id = raw.partition("|")
        if not separator or not post_id:
            raise ValueError(f"malformed cursor: {token!r}")
        return Cursor(float(created), post_id)

    @staticmethod
    def of(entry: TimelineEntry) -> "Cursor":
        return Cursor(entry.created_at, entry.post_id)

    def is_before(self, entry: TimelineEntry) -> bool:
        """True when `entry` is strictly OLDER than this cursor — that is,
        strictly after it in a newest-first walk. This is the predicate the
        timeline store applies, and it is the entire pagination contract."""
        return (entry.created_at, entry.post_id) < (self.created_at, self.post_id)


@dataclass(frozen=True)
class FeedPage:
    items: list[Post]
    next_cursor: Optional[Cursor]
    ranked: bool = True

    def __len__(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------
# 5. The timeline store
# --------------------------------------------------------------------------

class TimelineStore(ABC):
    """The port. Repeated from §4 so this block runs standalone — §4 is where
    the contract is argued, this is the copy the code below actually uses."""

    @abstractmethod
    def push(self, user_id: str, entries: list[TimelineEntry]) -> int:
        raise NotImplementedError

    @abstractmethod
    def read(self, user_id: str, after: Optional[Cursor],
             limit: int) -> list[TimelineEntry]:
        raise NotImplementedError

    @abstractmethod
    def remove_post(self, user_id: str, post_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def truncate(self, user_id: str, keep: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def rebuild(self, user_id: str, entries: list[TimelineEntry]) -> int:
        raise NotImplementedError


class InMemoryTimelineStore(TimelineStore):
    """The materialised cache, per user, newest first.

    CAPACITY IS THE POINT OF THE DATA STRUCTURE. A timeline is a bounded window
    of the recent past, not a history: a user with 5,000 followees produces far
    more posts than anyone scrolls. The cap is what keeps one user's timeline
    fitting in a fixed memory budget, and it is why the pull path exists even
    for pushed authors — a post older than the window has fallen out, and the
    pull path is what finds it again.
    """

    def __init__(self, capacity: int = 800) -> None:
        if capacity < 1:
            raise ValueError("a timeline must hold at least one entry")
        self.capacity = capacity
        self._by_user: dict[str, list[TimelineEntry]] = {}
        self._lock = threading.RLock()

    def push(self, user_id: str, entries: list[TimelineEntry]) -> int:
        """Merge, dedupe, sort, cap. Returns how many entries were ADDED, not
        how many were offered — a re-pushed post is a no-op, which is what
        makes the fan-out retryable."""
        with self._lock:
            current = self._by_user.setdefault(user_id, [])
            known = {(e.created_at, e.post_id) for e in current}
            added = 0
            for entry in entries:
                if (entry.created_at, entry.post_id) in known:
                    continue
                current.append(entry)
                known.add((entry.created_at, entry.post_id))
                added += 1
            if added:
                current.sort(reverse=True)
                if len(current) > self.capacity:
                    del current[self.capacity:]
            return added

    def read(self, user_id: str, after: Optional[Cursor],
             limit: int) -> list[TimelineEntry]:
        with self._lock:
            entries = self._by_user.get(user_id, [])
            if after is None:
                return list(entries[:limit])
            return [e for e in entries if after.is_before(e)][:limit]

    def remove_post(self, user_id: str, post_id: str) -> bool:
        """Deleting a post must reach a timeline that ALREADY LISTS IT. This is
        the method whose absence produces the HLD's "deleted posts lingering in
        feeds", and it cannot be replaced by a rebuild without a full fan-out
        per user."""
        with self._lock:
            entries = self._by_user.get(user_id)
            if entries is None:
                return False
            remaining = [e for e in entries if e.post_id != post_id]
            if len(remaining) == len(entries):
                return False
            self._by_user[user_id] = remaining
            return True

    def truncate(self, user_id: str, keep: int) -> int:
        with self._lock:
            entries = self._by_user.get(user_id, [])
            removed = max(0, len(entries) - keep)
            if removed:
                self._by_user[user_id] = entries[:keep]
            return removed

    def rebuild(self, user_id: str, entries: list[TimelineEntry]) -> int:
        """The dormant user (§6.6): an empty timeline is not an error, it is a
        rebuild. Replacing rather than merging is deliberate — a rebuild is the
        answer to "this cache is wrong", and merging would keep the wrongness."""
        with self._lock:
            ordered = sorted(set(entries), reverse=True)[:self.capacity]
            self._by_user[user_id] = ordered
            return len(ordered)

    def size(self, user_id: str) -> int:
        with self._lock:
            return len(self._by_user.get(user_id, ()))

    def users(self) -> list[str]:
        with self._lock:
            return sorted(self._by_user)

    def total_entries(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._by_user.values())


# --------------------------------------------------------------------------
# 6. Per-viewer state: what has been seen
# --------------------------------------------------------------------------

class SeenStore:
    """What this viewer has already been shown, and when they were last active.

    Both facts live here because they have the same shape, the same owner and
    the same retention: a per-viewer record with a timestamp. Splitting them
    into two classes would mean two caches to size, two TTLs to tune and two
    places to look when a feed is wrong.
    """

    def __init__(self, retention_s: float = 30 * 24 * 3600.0) -> None:
        self._seen: dict[str, dict[str, float]] = {}
        self._active: dict[str, float] = {}
        self._lock = threading.RLock()
        self.retention_s = retention_s

    def touch(self, user_id: str, now: float) -> None:
        with self._lock:
            self._active[user_id] = now

    def last_active(self, user_id: str) -> float:
        with self._lock:
            return self._active.get(user_id, 0.0)

    def is_active(self, user_id: str, now: float,
                  window_s: float = 7 * 24 * 3600.0) -> bool:
        return now - self.last_active(user_id) <= window_s

    def mark_seen(self, user_id: str, post_ids: Iterable[str],
                  now: float) -> int:
        with self._lock:
            bucket = self._seen.setdefault(user_id, {})
            added = 0
            for post_id in post_ids:
                if post_id not in bucket:
                    bucket[post_id] = now
                    added += 1
            return added

    def has_seen(self, user_id: str, post_id: str) -> bool:
        with self._lock:
            return post_id in self._seen.get(user_id, {})

    def seen_count(self, user_id: str) -> int:
        with self._lock:
            return len(self._seen.get(user_id, ()))

    def prune(self, user_id: str, now: float) -> int:
        """Seen-ness is not forever. A post seen 30 days ago can be shown again
        without anybody noticing, and forgetting it bounds the store — which
        matters because this map grows with every scroll."""
        cutoff = now - self.retention_s
        with self._lock:
            bucket = self._seen.get(user_id)
            if bucket is None:
                return 0
            before = len(bucket)
            self._seen[user_id] = {p: t for p, t in bucket.items() if t >= cutoff}
            return before - len(self._seen[user_id])


# --------------------------------------------------------------------------
# 7. Ranking — a pipeline of stages
# --------------------------------------------------------------------------

@dataclass
class ScoredPost:
    """A candidate in flight, carrying the post and the partial score. Mutable
    on purpose: a stage annotates what the previous stages established, and
    rebuilding the object per stage would make the pipeline quadratic in
    allocations for no benefit."""
    post: Post
    score: float = 0.0
    affinity: float = 0.0
    recency: float = 0.0
    dropped: bool = False
    reason: str = ""

    @property
    def key(self) -> tuple[float, str]:
        """THE sort key, and the tie-break is `post_id`. Without it, two posts
        with the same score would come back in whatever order the sort happened
        to produce — which differs between runs, and makes the pipeline
        untestable and the feed feel arbitrary."""
        return (self.score, self.post.id)


@dataclass(frozen=True)
class RankContext:
    viewer_id: str
    now: float
    limit: int
    affinity: dict[str, float] = field(default_factory=dict)


class Stage(ABC):
    @abstractmethod
    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        """One concern, applied to the whole list."""
        raise NotImplementedError


class EligibilityStage(Stage):
    """Is this post visible to this viewer AT ALL? Self-posts are dropped here
    rather than in the query, because a viewer's own post can reach their
    timeline by two routes (the push fan-out includes the author) and the
    dedupe by id does not catch that they should not see it twice."""

    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        for candidate in candidates:
            if candidate.post.author_id == context.viewer_id:
                candidate.dropped = True
                candidate.reason = "own post"
        return candidates


class SeenFilterStage(Stage):
    """Drop what this viewer has already been shown.

    THE ORDERING MATTERS AND IS NOT OBVIOUS. Filtering happens BEFORE the page
    is cut, in the read path, because filtering after the cut returns short
    pages — a page of 20 with 14 seen comes back as 6 items, and the client
    shows a mostly empty screen while more posts exist. The stage is here for
    the ranked path; the read path applies the same filter earlier, and §8
    names the double-application as a deliberate redundancy rather than a
    mistake."""

    def __init__(self, seen: SeenStore) -> None:
        self.seen = seen

    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        for candidate in candidates:
            if self.seen.has_seen(context.viewer_id, candidate.post.id):
                candidate.dropped = True
                candidate.reason = "already seen"
        return candidates


class AffinityStage(Stage):
    """How much this viewer cares about this author. One of the few features
    that is genuinely slow-moving, which is what makes it cacheable and what
    keeps the ranking deterministic between two page fetches."""

    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        for candidate in candidates:
            candidate.affinity = context.affinity.get(
                candidate.post.author_id, 0.0)
        return candidates


class RecencyStage(Stage):
    """A decay, not a timestamp. `0.5 ** (age / half_life)` is bounded in
    [0, 1] whatever the age, so it composes with the other terms instead of
    dominating them — a raw timestamp would swamp every other feature."""

    def __init__(self, half_life_s: float = 12 * 3600.0) -> None:
        if half_life_s <= 0:
            raise ValueError("a half-life must be positive")
        self.half_life_s = half_life_s

    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        for candidate in candidates:
            age = max(0.0, context.now - candidate.post.created_at)
            candidate.recency = 0.5 ** (age / self.half_life_s)
        return candidates


class MediaBoostStage(Stage):
    """A product nudge, isolated in a class so it can be argued about. It is
    the stage most likely to be removed, and making it a class means removing
    it is one line in a constructor rather than an edit to a scoring body."""

    def __init__(self, boost: float = 0.15) -> None:
        self.boost = boost

    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        for candidate in candidates:
            if candidate.post.media_ids:
                candidate.recency += self.boost
        return candidates


class ScoreStage(Stage):
    """Combine the annotations. The weights are the whole model, and they are
    constructor parameters because a weight is a number somebody will want to
    change on a Tuesday."""

    def __init__(self, affinity_weight: float = 1.0,
                 recency_weight: float = 0.8) -> None:
        self.affinity_weight = affinity_weight
        self.recency_weight = recency_weight

    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        for candidate in candidates:
            candidate.score = round(
                self.affinity_weight * candidate.affinity
                + self.recency_weight * candidate.recency, 6)
        return candidates


class DropStage(Stage):
    """Remove everything a previous stage flagged, in one place.

    Stages FLAG rather than remove, and this is why: a stage that removes
    cannot explain itself, and "why is my post not in my feed" is the question
    this pipeline exists to answer. Dropping at the end means the reason it was
    dropped is still attached to the object when somebody asks."""

    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        return [c for c in candidates if not c.dropped]


class TruncateStage(Stage):
    def apply(self, candidates: list[ScoredPost],
              context: RankContext) -> list[ScoredPost]:
        return candidates[:context.limit]


class RankingPipeline:
    """Stages in order, plus the sort that makes them deterministic.

    The sort happens here, between the scoring stages and the truncation, so
    every stage before it can stay order-agnostic and every stage after it can
    assume best-first."""

    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages

    def run(self, posts: list[Post], context: RankContext
            ) -> list[ScoredPost]:
        candidates = [ScoredPost(post) for post in posts]
        for stage in self.stages:
            candidates = stage.apply(candidates, context)
        candidates.sort(key=lambda c: c.key, reverse=True)
        return candidates

    @staticmethod
    def default(seen: SeenStore) -> "RankingPipeline":
        """The pipeline the demo uses, in the order that makes the flags
        meaningful: eligibility and seen-ness flag, affinity and recency
        annotate, media nudges, score combines, drop removes, truncate cuts."""
        return RankingPipeline([
            EligibilityStage(),
            SeenFilterStage(seen),
            AffinityStage(),
            RecencyStage(),
            MediaBoostStage(),
            ScoreStage(),
            DropStage(),
            TruncateStage(),
        ])


class AffinityStore:
    """Interaction counts per (viewer, author), turned into a bounded score.

    Kept deliberately dumb: counters plus a squash. A real system trains this,
    and the point of putting it behind a store is that swapping a trained model
    in later changes one class rather than the pipeline — the pipeline only
    ever asks for a dict of author -> float.

    KEYED BY VIEWER FIRST, and that is the whole reason the nesting is
    `{viewer: {author: n}}` rather than `{(viewer, author): n}`. The read is
    "everything this viewer has interacted with", which is one lookup on the
    outer key; the flat version has to iterate every counter belonging to every
    viewer on every feed request, so a single reader's page costs
    O(all interactions in the system). The nesting is the difference between a
    per-viewer cost and a global one, and it is invisible in a test with three
    users.
    """

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    def record(self, viewer_id: str, author_id: str, weight: int = 1) -> None:
        with self._lock:
            bucket = self._counts.setdefault(viewer_id, {})
            bucket[author_id] = bucket.get(author_id, 0) + weight

    def affinity(self, viewer_id: str) -> dict[str, float]:
        """`n / (n + 5)` — strictly increasing, bounded by 1, and steepest for
        the first few interactions. A linear count would let one prolific
        author dominate every feed after a weekend of scrolling; the squash is
        what keeps affinity comparable to the recency term it is summed with."""
        with self._lock:
            bucket = self._counts.get(viewer_id, {})
            return {author: count / (count + 5.0)
                    for author, count in bucket.items()}

    def tracked_authors(self, viewer_id: str) -> int:
        """How many authors this viewer carries. A user who follows 5,000
        accounts has 5,000 entries here, and the whole map is copied into a
        `RankContext` on every feed request — that is the unbounded part, and
        the fix is to keep only the top-K by count."""
        with self._lock:
            return len(self._counts.get(viewer_id, ()))


# --------------------------------------------------------------------------
# 8. The read path
# --------------------------------------------------------------------------

class FeedService:
    """THE READ PATH, and the ten steps below are the design.

    The two that people get wrong are step 6 and step 9. Step 6 is the seen
    filter applied BEFORE the cut, so a page is never returned short. Step 9 is
    the next cursor taken from the RAW FETCHED WINDOW rather than from the
    ranked, truncated list — taking it from the last item the reader saw skips
    every post that was fetched but ranked below the cut, which is the bug that
    makes a feed silently lose posts at a page boundary and never recover.
    """

    def __init__(self, posts: PostStore, graph: FollowGraph,
                 timelines: InMemoryTimelineStore, fan_out_service: FanOutService,
                 seen: SeenStore, affinity: AffinityStore,
                 pipeline: Optional[RankingPipeline] = None,
                 page_size: int = 20, overscan: int = 4) -> None:
        self.posts = posts
        self.graph = graph
        self.timelines = timelines
        self.fan_out_service = fan_out_service
        self.seen = seen
        self.affinity = affinity
        self.pipeline = pipeline or RankingPipeline.default(seen)
        self.page_size = page_size
        self.overscan = overscan

    # -- write path ------------------------------------------------------
    def publish(self, author_id: str, caption: str, now: float,
                media_ids: tuple[str, ...] = ()) -> tuple[Post, FanOutResult]:
        """Create the post, then hand it to the fan-out service. The post is
        stored FIRST: a fan-out that fails must not leave the post missing,
        because a post that exists and is not in timelines can be repaired by
        pull, and a post that does not exist cannot be repaired at all."""
        post = self.posts.add(Post(self.posts.next_id(), author_id, now,
                                   caption, media_ids))
        result = self.fan_out_service.fan_out(post, now)
        return post, result

    def delete_post(self, post_id: str, now: float) -> int:
        """Tombstone, then evict from every timeline that lists it.

        The eviction is a LINEAR SCAN over users in this implementation, and it
        is the honest shape of the problem: a real system needs a reverse index
        from post to the timelines holding it, which is exactly the index an
        incremental fan-out already has a reason to maintain. Named here rather
        than hidden, because a linear scan is what a first version looks like.
        """
        if not self.posts.delete(post_id, now):
            return 0
        evicted = 0
        for user_id in self.timelines.users():
            if self.timelines.remove_post(user_id, post_id):
                evicted += 1
        return evicted

    # -- read path -------------------------------------------------------
    def get_feed(self, viewer_id: str, cursor_token: Optional[str] = None,
                 limit: Optional[int] = None, now: float = 0.0,
                 rank: bool = True) -> FeedPage:
        """Steps 1-10. Read the numbered comments as the answer to "walk me
        through a feed request"."""
        page_size = limit or self.page_size
        cursor = Cursor.decode(cursor_token) if cursor_token else None
        context = RankContext(viewer_id, now, page_size,
                              self.affinity.affinity(viewer_id))

        # 1. The viewer is active; that is the signal the fan-out service reads
        #    to decide who gets pushed to.
        self.seen.touch(viewer_id, now)

        max_scan = page_size * self.overscan
        candidates: dict[str, Post] = {}
        walk: Optional[Cursor] = cursor
        last_examined: Optional[TimelineEntry] = None
        scanned = 0
        exhausted = False

        # 2-6. THE CANONICAL WALK, and the shape of it is the whole design.
        #
        #      We walk the timeline in cursor order and stop the moment we hold
        #      `page_size` SURVIVORS. We do not fetch a big window and then cut
        #      it, because a cut window has entries that were fetched and never
        #      served, and a cursor over creation time cannot say "come back for
        #      those" — it can only say "everything older than X". Anything
        #      fetched, unserved and newer than the returned cursor is lost
        #      silently. Stopping at the boundary means nothing is ever fetched
        #      and then abandoned: the page IS the walk.
        #
        #      `max_scan` is therefore a LIMIT ON HOW FAR WE WILL LOOK, not a
        #      window we intend to serve. A viewer who has seen everything recent
        #      would otherwise scan the whole timeline on every request.
        while len(candidates) < page_size and scanned < max_scan:
            want = max_scan - scanned
            window = self.timelines.read(viewer_id, walk, want)
            full_window = len(window) == want

            # 3. Hydrate the window in ONE call. This is the N+1 fix: up to
            #    `max_scan` posts are one lock acquisition, not `max_scan` of
            #    them, and the walk below reads from the dict.
            batch = self.posts.get_many([e.post_id for e in window])

            consumed_all = True
            for entry in window:
                scanned += 1
                walk = Cursor.of(entry)      # the cursor only moves forward
                last_examined = entry

                post = batch.get(entry.post_id)

                # 4. THE THREE THINGS THAT DROP AN ENTRY, and each is a different
                #    reason: the post is gone (tombstone), the reader has already
                #    seen it, or it is their own. All three `continue` rather than
                #    shrink the page — that is the entire reason the walk exists.
                if post is None or self.posts.is_deleted(entry.post_id):
                    continue                  # tombstoned, or gone from the store
                if self.seen.has_seen(viewer_id, entry.post_id):
                    continue                  # already shown
                if post.author_id == viewer_id:
                    continue                  # own post

                # 5. A survivor, and it counts toward the page.
                candidates[post.id] = post
                if len(candidates) >= page_size:
                    consumed_all = False
                    break

            # 6. The timeline is only exhausted when we ran out of window having
            #    consumed all of it. A short window we broke out of early still
            #    has entries in it we have not looked at.
            if consumed_all and not full_window:
                exhausted = True
                break

        # 7. Merge the PULL candidates — the second half of the per-author
        #    decision, and the reason an author with 50M followers still reaches
        #    a reader. They carry the same (created_at, post_id) coordinates as
        #    timeline entries, so the same cursor predicate is meaningful for
        #    them. Note the lower bound: `per_author` is small and re-queried on
        #    every page, so a pulled post that loses the ranking contest is
        #    reachable again only while the cursor has not passed it. That is the
        #    stated cost of pull, and §8 and §10 both name it rather than hiding
        #    it.
        for post in self.fan_out_service.pull_candidates(
                viewer_id, since=cursor.created_at if cursor else 0.0,
                per_author=self.overscan):
            if post.id in candidates:
                continue
            if cursor is not None and not cursor.is_before(
                    TimelineEntry(post.created_at, post.id)):
                continue                      # already served on an earlier page
            if self.seen.has_seen(viewer_id, post.id):
                continue
            if self.posts.is_deleted(post.id) or post.author_id == viewer_id:
                continue
            candidates[post.id] = post

        pool = list(candidates.values())

        # 8. Rank WITHIN the page. Ranking across the whole timeline is not
        #    possible: the cursor is over creation time, and a globally ranked
        #    order would need a cursor over the score, which moves between two
        #    requests. So the page is the unit of ranking, and the reader's
        #    experience is "the best of the next N", not "the best overall".
        if rank:
            context = RankContext(viewer_id, now, page_size, context.affinity)
            scored = self.pipeline.run(pool, context)
            page_posts = [c.post for c in scored]
        else:
            # The chronological path ignores the pipeline entirely, which is
            # what makes it a genuine fallback rather than a configuration flag.
            pool.sort(key=lambda p: (p.created_at, p.id), reverse=True)
            page_posts = pool[:page_size]

        # 9. THE CURSOR IS THE OLDEST POST ACTUALLY RETURNED, in canonical
        #    order. Three candidates, and only one of them is correct:
        #
        #    - The LAST RANKED item is wrong. Ranking reorders the page, so the
        #      last thing in the response has no fixed position in the cursor's
        #      order; using it as the next cursor skips everything the ranking
        #      put above it.
        #    - The LAST ENTRY EXAMINED is wrong when the page was cut. An entry
        #      examined, not served, and equal to the returned cursor is
        #      unreachable: the next request asks for entries STRICTLY older, and
        #      `read` returns nothing.
        #    - The OLDEST SERVED is correct. The next request asks for entries
        #      strictly older than the oldest thing the reader actually got.
        #
        #    `last_examined` is used only when nothing was served at all, so that
        #    a page of pure already-seen entries still makes progress instead of
        #    returning the same cursor forever.
        if exhausted and len(page_posts) >= len(pool):
            next_cursor = None               # nothing was fetched that we did not serve
        elif page_posts:
            oldest = min((p.created_at, p.id) for p in page_posts)
            next_cursor = Cursor(oldest[0], oldest[1])
        elif last_examined is not None:
            next_cursor = Cursor.of(last_examined)
        else:
            next_cursor = None

        # 10. Mark what was actually returned. Marking the whole scanned window
        #     instead would hide posts the reader never saw, and they would then
        #     never be shown.
        self.seen.mark_seen(viewer_id, [p.id for p in page_posts], now)
        return FeedPage(page_posts, next_cursor, ranked=rank)

    def rank_affinity(self, viewer_id: str, author_id: str,
                      weight: int = 1) -> None:
        """The event that feeds the ranking: the viewer engaged with an author."""
        self.affinity.record(viewer_id, author_id, weight)


# ==========================================================================
# 9. THE DEMO — every assertion below is a claim made earlier in this file
# ==========================================================================

T0 = 1_700_000_000.0
SEVEN_DAYS = 7.0 * 24.0 * 3600.0

PASSED = [0]
FAILED = [0]


def check(label: str, condition: bool) -> None:
    if condition:
        PASSED[0] += 1
        print(f"PASS {label}")
    else:
        FAILED[0] += 1
        print(f"FAIL {label}")


class World:
    """The §3 diagram, wired up. One place to build a small system, so each
    test below states only what it is testing."""

    def __init__(self, push_below: int = 10_000,
                 push_to_active_below: int = 1_000_000,
                 capacity: int = 800, max_fanout: int = 50_000,
                 page_size: int = 3, overscan: int = 2) -> None:
        self.now = T0
        self.posts = PostStore()
        self.graph = FollowGraph()
        self.timelines = InMemoryTimelineStore(capacity)
        self.seen = SeenStore()
        self.affinity = AffinityStore()
        self.policy = ThresholdFanOut(push_below, push_to_active_below)
        self.fan = FanOutService(
            self.graph, self.timelines, self.policy, self.posts,
            max_fanout=max_fanout,
            is_active=lambda uid: self.seen.is_active(uid, self.now, SEVEN_DAYS))
        self.svc = FeedService(self.posts, self.graph, self.timelines, self.fan,
                               self.seen, self.affinity,
                               page_size=page_size, overscan=overscan)

    def publish(self, author_id: str, caption: str,
                media: tuple[str, ...] = ()) -> tuple[Post, FanOutResult]:
        self.now += 60.0
        return self.svc.publish(author_id, caption, self.now, media)

    def feed(self, viewer_id: str, token: Optional[str] = None,
             rank: bool = True) -> FeedPage:
        return self.svc.get_feed(viewer_id, token, now=self.now, rank=rank)

    def walk(self, viewer_id: str, rank: bool = True,
             max_pages: int = 12) -> list[str]:
        """Follow the cursor to the end, as a client does."""
        served: list[str] = []
        token: Optional[str] = None
        for _ in range(max_pages):
            page = self.feed(viewer_id, token, rank)
            served.extend(p.id for p in page.items)
            if page.next_cursor is None:
                break
            token = page.next_cursor.encode()
        return served


print("--- 1. the per-author fan-out decision ---")

w = World()
w.graph.follow("alice", "bob")
post, result = w.publish("bob", "hello")
check("a small author is PUSHed, not pulled", result.mode is FanOutMode.PUSH)
check("and the follower's timeline is written", w.timelines.size("alice") == 1)
page = w.feed("alice")
check("so the reader's feed is one lookup", [p.id for p in page.items] == [post.id])
check("and there is no next page", page.next_cursor is None)

# A "celebrity" is just a user past the threshold. Lowering the thresholds is
# the same test at a size a unit test can build.
w = World(push_below=2, push_to_active_below=3)
for follower in ("dave", "eve", "frank"):
    w.graph.follow(follower, "celeb")
post, result = w.publish("celeb", "an announcement")
check("an author past the threshold is PULLed", result.mode is FanOutMode.PULL)
check("pull writes nothing to timelines", result.timelines_written == 0)
check("so the follower's timeline is empty", w.timelines.size("dave") == 0)
page = w.feed("dave")
check("and an empty timeline still yields a feed (the pull path)",
      [p.id for p in page.items] == [post.id])

w = World(push_below=2, push_to_active_below=10)
for follower in ("gina", "gary", "gwen"):
    w.graph.follow(follower, "hana")
w.seen.touch("gina", w.now)
post, result = w.publish("hana", "a mid-tier post")
check("the middle mode is chosen for a mid-tier author",
      result.mode is FanOutMode.PUSH_TO_ACTIVE)
check("the active follower is written", w.timelines.size("gina") == 1)
check("an inactive follower is skipped", w.timelines.size("gary") == 0)
check("and the skip is reported rather than hidden", result.skipped_followers == 2)

w = World(push_below=1000, max_fanout=2)
for follower in ("a1", "a2", "a3"):
    w.graph.follow(follower, "z")
post, result = w.publish("z", "too many followers for this bound")
check("the bound overrode the policy", result.mode is FanOutMode.PULL)
check("and said so, rather than failing silently", result.downgraded is True)
check("nothing was written", result.timelines_written == 0)


print("--- 2. the cursor ---")

c = Cursor(T0, "post-7")
check("a cursor round-trips through the wire format", Cursor.decode(c.encode()) == c)
check("and is opaque, not a readable timestamp", "1700000000" not in c.encode())
for bad in ("", "not-a-cursor", "!!!!"):
    try:
        Cursor.decode(bad)
        check(f"a malformed cursor is rejected: {bad!r}", False)
    except ValueError:
        check(f"a malformed cursor is rejected: {bad!r}", True)
check("is_before() is strict, so equal coordinates stop the walk",
      not c.is_before(TimelineEntry(T0, "post-7")))
check("and an older pair is before it", c.is_before(TimelineEntry(T0 - 1, "post-9")))


print("--- 3. pagination over an order that does not move ---")

w = World(page_size=3, overscan=2)
w.graph.follow("reader", "author")
made = [w.publish("author", f"post {i}")[0] for i in range(10)]
served = w.walk("reader")
check("every pushed post is served exactly once",
      sorted(served) == sorted(p.id for p in made))
check("with no duplicates across pages", len(served) == len(set(served)))
check("and pages descend in canonical order",
      served == [p.id for p in sorted(made, key=lambda p: p.created_at,
                                      reverse=True)])

# The property the whole design exists for: a post published BETWEEN two page
# fetches must not disturb the second page.
w = World(page_size=3, overscan=2)
w.graph.follow("reader", "author")
made = [w.publish("author", f"p{i}")[0] for i in range(6)]
page1 = w.feed("reader")
token = page1.next_cursor.encode()
intruder, _ = w.publish("author", "published between the two fetches")
page2 = w.feed("reader", token)
check("a post published between fetches does not appear below the cursor",
      intruder.id not in {p.id for p in page2.items})
check("and page two repeats nothing from page one",
      not ({p.id for p in page1.items} & {p.id for p in page2.items}))
check("while still serving the rest of the older posts",
      {p.id for p in page2.items} == {made[2].id, made[1].id, made[0].id})

# A full second page is not short, even though every post on page one is now
# "seen". This is the seen-filter-before-the-cut rule.
w = World(page_size=4, overscan=2)
w.graph.follow("reader", "author")
made = [w.publish("author", f"s{i}")[0] for i in range(12)]
page1 = w.feed("reader")
page2 = w.feed("reader", page1.next_cursor.encode())
check("the second page is full, not short", len(page2.items) == 4)
check("because seen posts are skipped while walking, not after cutting",
      not ({p.id for p in page1.items} & {p.id for p in page2.items}))

# The cursor must walk a FIXED order. A cursor over a recomputed score does
# not, and the failure is a duplicate rather than an error.
w = World(page_size=3, overscan=2)
w.graph.follow("reader", "author")
made = [w.publish("author", f"k{i}")[0] for i in range(6)]
ctx_a = RankContext("reader", w.now, 6, {})
scored_a = w.svc.pipeline.run([w.posts.get(p.id) for p in made], ctx_a)
boundary = scored_a[2].score            # the last post that fits on page one
later = w.now + 3 * 3600.0
ctx_b = RankContext("reader", later, 6, {})
scored_b = w.svc.pipeline.run([w.posts.get(p.id) for p in made], ctx_b)
score_page2 = [c.post.id for c in scored_b if c.score < boundary]
check("a SCORE cursor re-serves the last post of page one",
      scored_a[2].post.id in score_page2)
page = w.feed("reader")
check("the time cursor names a FIXED pair, so the boundary cannot drift",
      page.next_cursor == Cursor(scored_a[2].post.created_at,
                                 scored_a[2].post.id))


print("--- 4. the timeline is derived data ---")

w = World(page_size=3, overscan=2)
w.graph.follow("reader", "author")
kept = [w.publish("author", f"kept {i}")[0] for i in range(3)]
gone, _ = w.publish("author", "delete me")
check("the doomed post is in the timeline first", w.timelines.size("reader") == 4)
check("the delete evicts it", w.svc.delete_post(gone.id, w.now) == 1)
check("and leaves a tombstone", w.posts.is_deleted(gone.id))
check("so the store no longer returns it", w.posts.get(gone.id) is None)
served = w.walk("reader")
check("it never appears in the feed", gone.id not in served)
check("and the rest still do", sorted(served) == sorted(p.id for p in kept))
check("deleting twice is a retry, not an error",
      w.svc.delete_post(gone.id, w.now) == 0)

w = World(capacity=5, page_size=3, overscan=2)
w.graph.follow("reader", "author")
made = [w.publish("author", f"c{i}")[0] for i in range(9)]
check("the timeline is bounded", w.timelines.size("reader") == 5)
served = w.walk("reader")
check("a bounded timeline serves what it holds", len(served) == 5)
check("and what it holds is the newest", set(served) == {p.id for p in made[-5:]})

# The dormant reader: skipped by PUSH_TO_ACTIVE, so nothing was ever written.
w = World(push_below=1, push_to_active_below=1000, capacity=10)
for follower in ("active", "dormant"):
    w.graph.follow(follower, "author")
w.seen.touch("active", w.now)
made = [w.publish("author", f"r{i}")[0] for i in range(3)]
check("only the active follower is pushed to", w.timelines.size("active") == 3)
check("the dormant follower's timeline is empty", w.timelines.size("dormant") == 0)
page = w.feed("dormant")
check("pull still reaches them, bounded by per_author",
      {p.id for p in page.items} == {made[2].id, made[1].id})
check("and the oldest post is out of a bounded pull's reach",
      made[0].id not in {p.id for p in page.items})
check("a rebuild is what materialises it",
      w.timelines.rebuild("dormant", [TimelineEntry(p.created_at, p.id)
                                      for p in made]) == 3)
check("after which the walk reaches the post pull could not",
      made[0].id in w.walk("dormant"))


print("--- 5. ranking ---")

w = World(page_size=3, overscan=2)
w.graph.follow("reader", "author")
made = [w.publish("author", f"m{i}", ("m1",) if i in (3, 4) else ())[0]
        for i in range(6)]
scored = w.svc.pipeline.run(
    [w.posts.get(p.id) for p in made],
    RankContext("reader", w.now, 6, {}))
check("media lifts a post above newer ones",
      [c.post.id for c in scored[:2]] == [made[4].id, made[3].id])
check("and the boosted post is not the newest, so the lift came from the flag",
      scored[0].post.created_at < max(p.created_at for p in made))
check("the tie-break makes the order total and repeatable",
      [c.key for c in scored] == sorted((c.key for c in scored), reverse=True))

w = World(page_size=4, overscan=2)
w.graph.follow("reader", "author")
w.now += 60.0
same = [w.svc.publish("author", f"tie{i}", w.now, ())[0] for i in range(3)]
page = w.feed("reader")
check("three posts with identical scores are ordered by post id",
      [p.id for p in page.items] == sorted((p.id for p in same), reverse=True))

w = World(page_size=3, overscan=2)
w.graph.follow("reader", "author")
plain = [w.publish("author", f"a{i}")[0] for i in range(3)]
w.svc.rank_affinity("reader", "author", 10)
check("affinity is bounded by the squash, not the raw count",
      w.svc.affinity.affinity("reader")["author"] == 10 / 15.0)
check("and a viewer with no affinity scores zero",
      w.svc.affinity.affinity("nobody") == {})
check("the store is keyed per viewer, so one reader's page does not scan others",
      (w.svc.affinity.tracked_authors("reader"),
       w.svc.affinity.tracked_authors("nobody")) == (1, 0))


print("--- 6. chronological fallback ---")


def build_media_world() -> World:
    w = World(page_size=3, overscan=2)
    w.graph.follow("reader", "author")
    for i in range(6):
        w.publish("author", f"f{i}", ("m1",) if i in (3, 4) else ())
    return w


# Two identical systems, one asked for each path — because `get_feed` marks what
# it returns as seen, so a second call on the same system would answer from the
# NEXT page rather than the same one.
ranked_world, chrono_world = build_media_world(), build_media_world()
ranked_page = ranked_world.feed("reader", rank=True)
chrono = chrono_world.feed("reader", rank=False)
newest_three = sorted(chrono_world.posts.by_id.values(),
                      key=lambda p: (p.created_at, p.id), reverse=True)[:3]
check("the ranked page is not in creation order",
      [p.id for p in ranked_page.items] != [p.id for p in chrono.items])
check("the chronological page is the newest three, in creation order",
      [p.id for p in chrono.items] == [p.id for p in newest_three])
check("and both report which path they took",
      ranked_page.ranked is True and chrono.ranked is False)


print("--- 7. determinism ---")


def build() -> World:
    w = World(page_size=3, overscan=2)
    w.graph.follow("reader", "author")
    for i in range(6):
        w.publish("author", f"d{i}", ("m1",) if i == 3 else ())
    return w


left, right = build(), build()
page_left, page_right = left.feed("reader"), right.feed("reader")
check("two identically-built systems return the same page",
      [p.id for p in page_left.items] == [p.id for p in page_right.items])
check("and the same cursor", page_left.next_cursor == page_right.next_cursor)


print("--- 8. concurrency ---")

w = World(page_size=20, overscan=2)
for i in range(8):
    w.graph.follow(f"reader{i}", "author")


def blast(worker: int) -> None:
    for i in range(25):
        # Deliberately identical timestamps across workers, so the tie-break is
        # exercised rather than avoided.
        w.svc.publish("author", f"w{worker}-{i}", T0 + 60.0 * (i + 1), ())


threads = [threading.Thread(target=blast, args=(worker,)) for worker in range(4)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

check("every concurrent publish was stored", len(w.posts.by_id) == 100)
check("the id counter never handed out an id twice",
      w.posts.author_post_count("author") == 100)

sizes_ok, order_ok = True, True
for i in range(8):
    entries = w.timelines.read(f"reader{i}", None, 800)
    if len(entries) != 100:
        sizes_ok = False
    if list(entries) != sorted(set(entries), reverse=True):
        order_ok = False
check("every follower's timeline holds all 100 posts", sizes_ok)
check("each is sorted newest-first with no duplicates", order_ok)


print()
print(f"{PASSED[0]} checks passed, {FAILED[0]} failed")
if FAILED[0]:
    raise SystemExit(1)
```

**What the demo establishes, in the order the design was argued:**

| Block | The claim it holds the code to |
|---|---|
| 1 | The per-author decision is made at write time, and the bound overrides it loudly |
| 2 | The cursor is an opaque, strict, fixed pair — not a score and not an offset |
| 3 | Pagination is exact: nothing skipped, nothing duplicated, stable under new posts |
| 4 | The timeline is a repairable cache: tombstones evict, capacity caps, rebuild fills |
| 5 | Ranking is deterministic and bounded, with an explicit tie-break |
| 6 | The chronological path is a genuine fallback, not a flag on the same code |
| 7 | Two identical systems return identical pages |
| 8 | The stores are safe under concurrent writes, and the id counter is atomic |

---

## 6. Concurrency & thread-safety

The HLD answers "how does this scale". This section answers the question the HLD does not: **which line of this code can two threads be inside at the same time, and what happens when they are.**

### 6.1 What is shared, and what guards it

| State | Guard | Why that granularity |
|---|---|---|
| `PostStore.by_id / by_author / tombstones / _counter` | one `RLock` | Small critical sections, and `next_id` must be atomic with respect to itself or two posts share an id |
| `FollowGraph._following / _followers / _revision` | one `RLock` | The graph is read on every fan-out and every pull; a torn read of `_following` would fan out to a half-updated list |
| `InMemoryTimelineStore._by_user[user]` | one `RLock` over all users | The list is the unit: merge-sort-cap must be one atomic read-modify-write |
| `SeenStore._seen / _active` | one `RLock` | Same shape, same owner, same retention |
| `AffinityStore._counts` | one `RLock` | An increment, nothing more |
| `RankingPipeline` | **none — it is stateless** | Stages carry parameters, not state; the only shared thing they touch is the `SeenStore` |

**No global lock.** The four stores are four independent locks, which is what lets a publish to one author run while another author's feed is being read. A single coarse lock around "the app" is the version that passes every functional test and then serialises the whole service.

### 6.2 The one invariant that prevents every deadlock here

> **No lock is held across a call into another store.**

Trace `FeedService.publish`: `PostStore.add` takes and *releases* the post lock; then `FanOutService.fan_out` takes each timeline's lock, writes, and releases — one follower at a time. At no instant does the code hold the post lock and a timeline lock together, so there is no lock **ordering** to get wrong and therefore no deadlock to be possible. The same holds for `is_active`, which is called during `fan_out` with no lock held, and which itself takes the `SeenStore` lock.

Two consequences worth saying out loud:

- **The tempting optimisation that breaks it is to hold the post lock through the fan-out.** It reads as "make the post and its delivery atomic". It is wrong: a fan-out to 9,000 timelines becomes a global write barrier on the post store, so every *read* of every *other* post waits behind it. The post lock's job is to protect the post maps, not to make the write path transactional.
- **If you ever do need two locks, order them and write the order down.** The reason it is safe to skip that here is that the layering is `FeedService → {stores}`, never `store → store`.

### 6.3 The three concurrency bugs a single-threaded test cannot see

**1. A lost timeline push.** `InMemoryTimelineStore.push` is read-modify-write: it reads the list, merges, sorts, caps, and stores it back. Two authors posting to the same follower concurrently must not lose one. Because the whole sequence is inside one `RLock`, they cannot. The version that *looks* equivalent — `entries = self._by_user.get(user, [])` outside the lock, then a write back inside it — loses whichever push lands first, and it is invisible until two authors post at the same moment.

**2. A duplicated post id.** `next_id` and `add` are separate calls, so an id is allocated and *then* used. That is safe here only because the counter is monotonic and incremented under the lock: two threads necessarily get different numbers, and a gap (if a thread dies between the two calls) is harmless. The bug lives in the alternative — `len(self.by_id) + 1` as the id — where two concurrent inserts derive the same number and one silently overwrites the other.

**3. The seen filter racing itself.** Two concurrent requests from the *same* viewer with the *same* cursor both walk, both serve the same page, and both call `mark_seen`. That is benign: the read is idempotent and marking twice is a no-op. It is worth naming because the opposite assumption — "only one request per viewer" — is what leads people to add a per-viewer lock, which then serialises a reader's own feed requests for no benefit.

### 6.4 The cost of the lock discipline, stated honestly

- **The post store is the hot lock.** One dict pair for the whole service means every `add`, `get` and `is_deleted` contends. At scale this is sharded by post id (the id is a snowflake; its high bits pick the shard), and `by_author` moves to its own store. The class boundary is already right — `PostStore` is a port — so the sharded version is a second implementation, not a rewrite.
- **The seen filter takes the lock once per candidate.** `SeenFilterStage` calls `has_seen` per post, and the read path calls it per timeline entry. On a page of 20 with a 4x scan that is ~80 lock acquisitions per feed request. Each is trivial, and the fix is a `seen_many(ids) -> set[str]` batch on the store, called once per page. Named here rather than assumed away; it is the same N+1 shape the `get_many` batch in step 3 already fixes for the post store.
- **`pull_candidates` walks every followee under the graph lock, one call per followee.** `following_of` copies the list under the lock and releases; the post lookups that follow take the post lock separately. So a viewer following 5,000 accounts is 5,000 sequential lookups on the read path. That is a real number and §10 prices it.

---

## 7. Extensibility

Each row is a change somebody will actually ask for, and what it costs.

| Change | What you touch | What you do NOT touch |
|---|---|---|
| A new ranking signal (e.g. "posts from accounts you DM") | a new `Stage` subclass + one line in the pipeline list | `FeedService`, the cursor, the stores |
| Re-weight the model | two constructor arguments on `ScoreStage` | any other class |
| A new fan-out mode | a new `FanOutMode` member + a `FanOutPolicy` | `FanOutService` handles it in one branch; the read path already tolerates any mix |
| A per-post affinity instead of per-author | widen `AffinityStore.affinity` to return `dict[str, float]` keyed by post | `AffinityStage` — it already reads a dict by key, so only the key changes |
| Redis / Cassandra timelines | a new `TimelineStore` implementation | everything above the port; `InMemoryTimelineStore` stays as the test double |
| A trained ranker | replace `RankingPipeline` with one stage that calls the model | the `Stage` contract, so the pipeline can still be composed and tested stage by stage |
| A different cursor wire format (e.g. protobuf) | `Cursor.encode` / `Cursor.decode` | the cursor's *meaning*, which is the pair |
| Pagination that must resume inside a ranked order | **a different architecture** — see below | — |

### 7.1 The one extension that is not a drop-in, and why

"Rank the whole timeline and let me page through the ranking" is the most common follow-up request, and it cannot be done by adding a stage.

The reason is the cursor. A ranked order is not stable — the score of an unread post changes as `now` advances and as affinity moves — so a cursor over it drifts, and the drift is a duplicate or a skip. The three honest ways to get a globally ranked feed:

1. **Snapshot the ranking.** Materialise the ranked list per viewer with a version number, and make the cursor `(snapshot_version, position)`. Stable, and it costs a materialised ranked timeline per active viewer — which is a real write amplification and a real storage bill.
2. **Rank on a coarse time bucket.** Cursor over `(bucket, post_id)`, rank within the bucket. Pagination is stable because the bucket boundaries are fixed; the reader sees "the best of each hour". This is the pragmatic choice and it is close to what §5 does with a page-sized bucket.
3. **Freeze the features at fetch time.** Carry the page-1 feature vector in the cursor so page 3 is scored by the same model state as page 1. Correct, and the cursor becomes large and versioned.

Saying which one you would pick, and why, is a stronger answer than claiming the stage approach works.

### 7.2 Two things this design deliberately does not do

- **No reverse index from post → timelines holding it.** `delete_post` therefore scans every timeline. A real system maintains that index, and it is the same index an incremental fan-out wants for its own reasons. Named in §8 as a known cost.
- **No cross-region story.** `created_at` comes from one clock, and the cursor's correctness depends on it. Two regions with skewed clocks can produce a post whose `created_at` places it before a cursor that has already been issued, which means it is never served. The fix is a monotonic sequence allocated by a single authority (or a hybrid logical clock) as the leading component of the cursor pair — and that is a change to the cursor's *type*, not its encoding.

---

## 8. Failure modes & edge cases

| # | Symptom | Cause | How you would know | What this code does |
|---|---|---|---|---|
| 1 | A page returns 6 posts when the reader asked for 20 | Seen/deleted posts were filtered **after** the page was cut | Page sizes are consistently short for heavy users only | The walk filters *while* scanning and stops at `page_size` survivors — the short page is structurally impossible |
| 2 | The same post appears on two pages | The cursor was taken from the last **ranked** item, or from a recomputed **score** | Duplicate ids across page boundaries in client logs | The cursor is `(created_at, post_id)` from the oldest **served** post; both are fixed |
| 3 | A post is never shown to anybody | It was fetched, cut by the page size, and the cursor passed it | A post id present in the timeline and absent from every served page | The walk does not over-fetch and then cut — nothing is fetched and abandoned |
| 4 | A deleted post still appears in feeds | The timeline was written before the delete and never repaired | The post resolves to nothing when hydrated | Tombstone in `PostStore` + `TimelineStore.remove_post` + a tombstone check in both the walk and the pull merge |
| 5 | The same post arrives twice in one page | Pushed **and** pulled — the two halves of the fan-out decision both reach it | Id appears twice in one response | `candidates` is a dict keyed by post id, so the merge dedupes by construction |
| 6 | A celebrity's post never reaches a dormant follower | `PUSH_TO_ACTIVE` skipped them and the bounded pull only returns the newest `per_author` posts | Their feed is missing older posts from big accounts | The pull is bounded and the loss is stated; `rebuild` on next open is the repair (demonstrated in demo block 4) |
| 7 | Write amplification explodes when an author crosses the threshold | The policy said PUSH, and the follower count read at post time has grown | One post writes millions of timelines | `FanOutService.max_fanout` overrides the policy and sets `downgraded=True`, which is an alarmable signal rather than a silent fallback |
| 8 | Two posts share an id | `next_id` was derived from `len(by_id)` instead of a locked counter | One post overwrites another in `by_id` | The counter is monotonic under the post lock; the demo publishes 100 posts across 4 threads and asserts 100 distinct ids |
| 9 | A timeline is silently missing a post after two authors post at once | Read-modify-write on the timeline list without one lock around the whole sequence | Timeline size is low for popular followers only | The merge-sort-cap is inside a single `RLock` acquisition |
| 10 | The feed is slow, and only for users who follow a lot of accounts | 5,000 sequential `recent_by_author` lookups on the read path | Latency correlates with following count | Stated as a cost in §6.4 and §10, with the fix (batch the pull by author shard) |
| 11 | A post's `created_at` places it before an already-issued cursor | Two regions, two clocks, skew | A post that is never served in one region | Out of scope, and named in §7.2 with the fix (a monotonic sequence as the cursor's leading component) |
| 12 | A viewer's feed is empty though they follow people | Their timeline was truncated, or they were skipped while inactive, and pull is bounded | `timelines.size(viewer) == 0` with a non-zero following count | `TimelineStore.rebuild` fills it; pull covers the gap in the meantime |
| 13 | Ranking returns a different order on two identical requests | A score with no tie-break, so equal scores sort arbitrarily | Non-reproducible pages in tests | `ScoredPost.key` is `(score, post_id)`; demo block 7 asserts two identical systems agree exactly |
| 14 | The last page of a feed is an empty request | The cursor was returned when the timeline was already exhausted | A 200 with zero items at the end of every session | `next_cursor` is `None` when nothing was fetched that was not served; one empty trailing request remains possible when the walk stops on a full page — the standard fix is a `has_more` hint, and it is not in this code |

### 8.1 The two edge cases worth arguing about

**A post published while a reader is mid-scroll.** The reader is on page 3 with a cursor from page 2. A followee posts. That post is *newer* than the cursor, so it is not on page 3 — correct, and deliberate: inserting it would shift every subsequent page. The cost is that a reader who never returns to page 1 does not see it until the next session. The alternative (a "new posts" banner on top of the existing scroll) is a product decision the cursor does not have to make.

**A follow change mid-scroll.** The reader follows someone new; their older posts now qualify. They are older than the cursor, so they *will* appear on a later page — the timeline is only written for posts after the follow, but the pull path finds the followee's recent posts. If the reader *unfollows*, their posts already in the timeline stay there until they age out. Pruning on unfollow is a fan-out-shaped job over the follower's timeline, and the revision counter on `FollowGraph` is the hook for it.

---

## 9. Design patterns used, and why

| Pattern | Where | What it buys | What it costs |
|---|---|---|---|
| **Strategy** | `FanOutPolicy` behind `FanOutService`; `Stage` inside `RankingPipeline` | The two decisions that change most — who gets pushed to, and what ranks — are objects, so they are swapped and tested alone | One more indirection than an `if`; worth it the first time somebody wants a second policy |
| **Composite + Chain of Responsibility** | `RankingPipeline` — stages applied in order, each free to annotate or drop | A new signal is a new class, and the order of concerns is data rather than control flow | Stage order becomes a thing to get wrong; the pipeline constructor is now a design surface |
| **Ports & Adapters** | `TimelineStore` and `PostStore` as the write/read boundaries | The in-memory store in §5 is the test double for the Redis one; the read path never learns which | An ABC per store; the abstraction has to be right before it earns its keep |
| **Value Object** | `Post`, `TimelineEntry`, `Cursor`, `RankContext`, `FanOutResult`, `FeedPage` — all frozen dataclasses | Equality is by value, hashing works, and a `Cursor` cannot be mutated by the code that receives it | Rebuilding rather than mutating; the `frozen=True` on `TimelineEntry` is also what makes `set(entries)` legal in the demo |
| **Entity** | nothing here, deliberately | — | A post is immutable after creation, so it has no state to identify across changes. `Driver`-style `eq=False` was needed in the Uber file because a driver *changes*; a `Post` does not |
| **Facade** | `FeedService` | One entry point over four stores and a pipeline, so the API surface is two methods | The facade is where the ten-step read path lives, and it is the only place that has to know all four stores exist |
| **Memento** | `Cursor` | An opaque token that captures a read position without exposing the timeline | It is a *position* memento, not a snapshot: it captures where the reader is, not what they saw |
| **Policy + Null Object** | `FanOutPolicy`; `is_active=None` meaning "everybody is active" | The absent case is a value, not a branch at the call site | A `None` that means "no restriction" is a small trap for the next reader; the docstring on `_active_only` is the mitigation |
| **Specification** | the four predicates that drop an entry (deleted, seen, own, out-of-range) | Each reason is a named, separately testable condition | Four places to add a fifth reason; the walk keeps them together on purpose |
| **Repository** | `AffinityStore` | Ranking asks for a `dict[str, float]` and never learns whether it came from counters or a model | The interface is a dict, which is convenient and slightly lossy — a ranked list with weights would be stricter |

### 9.1 Two patterns deliberately not used

**Observer for fan-out.** The tempting shape is "`PostStore` publishes a `PostCreated` event and a fan-out subscriber handles it". It is worse here, and the reason is `FanOutResult`. A synchronous call returns `mode`, `timelines_written`, `downgraded` and `skipped_followers` to the caller that can act on them. An event makes the fan-out fire-and-forget, so the write path cannot tell the author's request whether the post actually reached anybody — and the `downgraded` alarm in §8 row 7 has nowhere to go. Events are right when the producer genuinely does not care; here it does.

**Decorator for the fan-out bound.** `FanOutService` looks like it decorates `FanOutPolicy` with a `max_fanout` check, and it would be a cleaner-looking class if it did. It is not one, because a decorator has to preserve the signature and the bound needs to return a *different* thing: `downgraded=True`. Encoding that in the mode enum alone would make "the policy chose pull" and "the bound overrode a push" indistinguishable, and those two deserve different alarms.

**Singleton.** Not for `PostStore`, `FollowGraph`, `SeenStore` or `AffinityStore`. Wired as constructor arguments, every test builds its own world — the demo constructs eleven of them — and two tests can run in the same process without interfering. A module-level `POSTS = PostStore()` makes that impossible, and it is the single most common reason a machine-coding round cannot test its own code.

---

## 10. Trade-offs

| Decision | The alternative | Why this | What it costs | Revisit when |
|---|---|---|---|---|
| Fan-out decided **per author at post time** | Push for everyone, or pull for everyone | One policy object covers both regimes, and the read path already handles any mix, so the threshold is tunable without a migration | Two code paths on the read side; the pull branch has to exist and be tested even if few authors use it | Never — this is the shape. The *thresholds* are what move |
| Cursor over **`(created_at, post_id)`** | A cursor over the ranking score, or an offset | The pair never moves, so "strictly after" is a total order; duplicates and skips become impossible rather than unlikely | The feed cannot be globally ranked; ranking is per page | When the product requires a globally ranked feed — then §7.1's snapshot or bucket design |
| The walk **stops at the page boundary** instead of over-fetching and cutting | Fetch `page_size × k`, filter, cut | Nothing is ever fetched and abandoned, so no post can be silently lost; the cursor always means "older than what you got" | Up to `max_scan` entries examined per page, and a viewer who has seen everything recent scans all of them | When the scan cost shows up as latency — then push the seen-filter down into the timeline store |
| Timeline is **derived and repairable** | Treat the timeline as the source of truth | Truncation, capacity caps and deletes all become normal operations rather than data loss | Every read path must tolerate a wrong timeline, which is why the pull merge and the tombstone checks exist | Never — the alternative is a timeline that cannot be rebuilt, which fails on the first bad deploy |
| **Tombstones** rather than row deletion | Hard delete from every store | A tombstone answers "was this deleted, and when", which a missing row cannot; and the pull path can check one flag instead of a scan | Tombstone state has to be checked in four places, and they must all agree — which is why `PostStore.get` was made tombstone-aware for consistency with `get_many` | When tombstone count becomes a storage problem; then they move to a compaction job |
| Rank **within the page** | Rank the whole timeline per request | The whole-timeline version is O(followees) per request and needs a stable cursor it cannot have | A post that loses the ranking contest inside a page is not shown on that page | When engagement data says the losses matter — then a materialised ranked timeline |
| Bounded pull (`per_author`) | Pull everything from every followee | Bounds read amplification: `followees × per_author` lookups, not `followees × their_post_count` | Older posts from large authors are unreachable until a rebuild (§8 row 6) | When readers complain about missing big-account posts; then per-author depth becomes a per-author setting |
| **Integer, monotone ids** | UUIDs | Sortable, compact, and the id is the tie-break in the cursor, so a UUID's randomness would make the tie-break meaningless | A single id authority; multi-region needs a snowflake | On the second region |
| One lock per store, **no lock held across stores** | A single service-wide lock, or holding the post lock through the fan-out | Fan-out to 9,000 timelines cannot block other authors' reads | Four locks to reason about, and the invariant is a convention rather than something the type system enforces | Never — the invariant is the design |

### 10.1 The trade-off that decides the whole file

If you take one thing away: **the cursor's order must be fixed, and everything expensive follows from that.** Ranking across the whole timeline is not available, because the only order that is stable enough to page over is creation time. The page is therefore the unit of ranking, the walk is the unit of pagination, and the cost is "the best of the next N" rather than "the best overall".

The alternative — a materialised, ranked timeline per viewer with a snapshot version — buys the better product and pays for it in write amplification, storage and a version in every cursor. Both are defensible. What is not defensible is a cursor over a recomputed score, and that is the one thing this design refuses to do.

---

## 11. Q&A

**Q1. Why not just push to everyone?**
Because the write amplification is not linear in anything you can afford. One post from an account with 50M followers is 50M timeline writes, each a read-modify-write on a bounded list. At a few thousand writes per second per node that is hours of work for one post, and the author's next post arrives long before it finishes — so the writes queue, and the queue is now the outage. Per-author fan-out is the answer: push while the number is bounded, pull when it is not, and let the read path pay for the difference on only the posts that are actually read. A pulled post costs a read only for the people who scroll to it.

**Q2. Why is the cursor `(created_at, post_id)` and not a score or an offset?**
An offset is a position in a list that changes under you: a post inserted at the top shifts everything, so offset 20 on page 2 is a different post than it was on page 1 — one duplicate and one skip per new post. A score cursor is worse in a way that produces no error at all: the score is recomputed per request from `now` and from features that move, so the boundary between "page 1" and "page 2" drifts, and posts cross it in both directions. The demo in §5 shows the exact failure: the last post on page 1 re-appears in the page-2 set one request later. `(created_at, post_id)` cannot drift, because a post's creation time never changes, and the id makes it a total order so equal timestamps still have a defined position.

**Q3. Why does the walk stop at the page boundary instead of fetching a big window and cutting it?**
Because anything fetched and not served has to be reachable on the next request, and a cursor can only say "older than this pair". If you fetch 80, serve 20, and set the cursor to the oldest *fetched* entry, the 60 you cut are newer than the cursor and can never be requested again — a silent post loss with no error, no log, and no test failure unless the test pages all the way through. Stopping the walk the moment the page is full makes the page *be* the walk: nothing is fetched and abandoned, and the cursor is the oldest thing the reader actually got. The cost is that `max_scan` is a limit on how far we look, not a buffer we intend to serve.

**Q4. Where do ranking features come from, and what changes when it is a real model?**
`AffinityStore` today, as counters plus a squash. The pipeline asks for one thing: a `dict[str, float]` of author to score, through `RankContext`. A trained ranker is one `Stage` that calls the model with the candidate and the context; every other stage, the pipeline, the cursor and both stores are untouched. The thing to say out loud is that a model makes the *score* less stable, not more, which is why the cursor must not be over it — swapping in a model does not change the pagination design, and that is the point.

**Q5. How do deletes work?**
A tombstone in `PostStore` plus `TimelineStore.remove_post` on every timeline that lists the post. Two halves are needed because the post store and the timeline are separate: tombstoning alone leaves the id in the timeline, and evicting alone lets a rebuild resurrect it. Three read paths check the tombstone — the walk, the pull merge, and `get_many` — and `get` was made tombstone-aware to match, because a `get` that returned deleted posts while `get_many` filtered them is two different answers to one question. The current eviction scans every timeline; a real system keeps a reverse index from post to the timelines holding it, which is the same index an incremental fan-out wants anyway.

**Q6. What happens when a viewer follows 5,000 accounts?**
Three effects, and they are all in the read path. Their timeline fills faster, so the capacity cap does more work — that is cheap and intended. `pull_candidates` walks every followee per page, which is 5,000 `recent_by_author` calls — bounded work per call, but 5,000 of them, and that is the real cost. And the pull is capped at `per_author` per followee, so the total is `followees × per_author` lookups regardless of how much those accounts posted. At scale the fix is to shard the pull by author and batch it, and to raise `per_author` only for the accounts that matter. The honest statement is that this is the number to watch, and §10 puts it in the trade-off table rather than pretending `recent_by_author` is free.

**Q7. Why bound the timeline, and what happens to a post that falls out?**
A timeline is a window on the recent past, not a history. Without a cap, one user following 5,000 accounts accumulates unbounded memory per reader, and the cost lands on the readers who follow the most — exactly the users you least want to degrade. With the cap, an old post falls out of the materialised timeline but is still in `PostStore`, so the pull path can find it: `recent_by_author` reads the post store, not the timeline. That is why the pull path exists even for authors whose posts are pushed, and it is the reason a truncated timeline is a performance event rather than a data-loss event.

**Q8. How do you avoid duplicates when a post is both pushed and pulled?**
`candidates` is a dict keyed by post id, and both the walk and the pull merge write into it. A pushed post that the pull also finds is one entry, by construction, and the merge is `if post.id in candidates: continue` on the pull side. The demo asserts that a post reachable both ways appears once. The alternative — deduping after ranking — puts the duplicate through the scorer twice and gives it two chances to be cut, which is a subtler bug than it looks.

**Q9. What is the actual concurrency story?**
Four stores, four locks, and one invariant: no lock is held across a call into another store. That is what makes a lock ordering unnecessary and a deadlock impossible here. The three bugs a single-threaded test cannot see are a lost timeline push (read-modify-write without one lock around the whole merge-sort-cap), a duplicated post id (`len(by_id) + 1` instead of a locked counter), and a torn `_following` read during fan-out. The demo runs four threads publishing into eight shared timelines and asserts 100 posts, 100 distinct ids, and every timeline sorted with no duplicates.

**Q10. How would you shard this?**
The timeline store is already sharded by construction — it is keyed by user, so it distributes with the user id. The post store is the one with a global lock, and it shards by post id: the id is a snowflake whose high bits pick the shard, so `get_many` becomes a scatter-gather with a small fan-out (a page of 20 posts from 20 different authors can be 20 shards; a page from one author is 1). The graph shards by the follower, since the read path is "who do I follow". The thing that does *not* shard is the id authority, which is why §7.2 names the monotonic sequence as the multi-region prerequisite.

**Q11. How do you handle a reader who has not opened the app in a month?**
Their timeline was skipped by `PUSH_TO_ACTIVE` while they were away, so it holds nothing recent, and pull is bounded — the demo shows the oldest of three posts being out of a bounded pull's reach. The repair is a rebuild on next open: read their followees' recent posts and materialise a timeline. That is a cold-start cost paid once, on a read that was going to be slow anyway, instead of a per-post cost paid by everyone. Note that `PUSH_TO_ACTIVE` is therefore a bet that dormant readers are a minority of reads; if that is wrong, the mode is wrong, and the fix is to drop to `PULL` for high-fan-out authors and let the read path carry it.

**Q12. What breaks first at 10x?**
The post store's single lock, because it is the one lock every read and every write touches. Behind it, the pull path's per-followee walk, because it scales with how many accounts a reader follows and that distribution has a long tail. Third, the timeline `remove_post` scan on delete, because it is O(users). None of the three is a design flaw — each is a known sharding or indexing job — and the reason to name them in this order is that it tells you what to instrument first rather than what to rewrite.

**Q13. ⚡ Suppose the interviewer says the feed must be globally ranked. What now?**
Then the cursor cannot be over creation time, and the design changes rather than extends. Three ways, and I would pick one out loud: materialise a ranked timeline per active viewer with a snapshot version and make the cursor `(version, position)`, which costs write amplification and storage; rank within a coarse time bucket and keep the cursor over `(bucket, post_id)`, which is stable and gives "the best of each hour"; or freeze the feature vector in the cursor so page 3 is scored by page 1's model state, which is correct and makes the cursor large and versioned. What I would not do is keep the current cursor and rank globally, because that is exactly the drift the demo shows.

**Q14. ⚡ Why not put the fan-out behind a message queue?**
Because the write path needs the answer. `FanOutService.fan_out` returns `mode`, `timelines_written`, `downgraded` and `skipped_followers`, and the `downgraded` flag is an alarm — a post the policy wanted pushed that the bound refused. Fire-and-forget removes the return value, so the overridden-push case goes silent, which is the one case most worth knowing about. A queue is right when the producer genuinely does not care about the outcome; publish does care. The compromise, if the synchronous fan-out becomes too slow, is synchronous for the decision and asynchronous for the writes — which is a different split than "async for all of it", and it keeps the flag.

**Q15. ⚡ How do you know your tests would catch a pagination regression?**
Because the test pages all the way to the end and asserts the served set equals the created set, not that page 1 looks right. A pagination bug is invisible in any single-page assertion: it shows up as a post that is never served, or served twice, and both require walking the whole feed to see. The demo does that over ten and twelve posts, asserts no duplicates, asserts the pages descend in canonical order, and separately asserts the hard case — a post published between two page fetches, which must not appear below the cursor. That last one is the regression that a "does page 2 have three items" test passes happily.

### 11.1 Pushback round

**P1. "Your fan-out holds a snapshot of the follower list. Someone who follows during the fan-out never gets the post. Isn't that a correctness hole?"**
It is a gap, and it is the correct one to accept. The alternative is to make the fan-out transactional with the follow — which means holding a lock across the graph and every timeline, which is the lock-ordering problem §6.2 exists to avoid, and it still would not be atomic across shards. The gap is self-healing: a new follower's timeline has no entry, so the read path's pull branch finds the author's recent posts on their next feed request. That is the payoff of treating the timeline as derived rather than authoritative. The property I would defend is that the failure is a *delay*, not a loss.

**P2. "`delete_post` scans every timeline in the system. You wrote the code and then wrote a paragraph admitting it is O(users). Why ship that?"**
Because the alternative is worse and the signature is already right. The alternatives are: rebuild every timeline (far more work than a scan), or do nothing and let the HLD's "deleted posts lingering in feeds" happen. A scan that evicts correctly is a slow correct answer, and it is the honest first version — the reverse index from post to timelines is a real requirement at scale, and it is the same index an incremental fan-out maintains for its own reasons, so it is not throwaway work to add it. What I would not do is hide the scan behind a method name that reads as O(1).

**P3. "Ranking within the page means a reader can see a worse post above a better one, and your own demo has a media-boosted old post jumping over a newer one. Isn't that just a broken product?"**
It is the accepted cost of a stable cursor, and it is visible rather than hidden — the demo asserts it as a positive. The reader-visible properties that come with it are worth more than perfect ordering: pages do not duplicate, posts are not skipped, and a new post arriving mid-scroll does not shift what comes next. The product-level answer is that "the best of the next twenty" is a different feed from "the best twenty overall", and if the product requires the latter, §7.1's snapshot design is the one to build. Saying that trade-off out loud is the point; the failure mode is shipping page-local ranking while the product copy says "ranked for you".

**P4. "Seen-marking means a post the reader scrolled past on page 1 is gone forever. That is a real bug."**
It is a real limitation and it comes from a genuine information gap: the server cannot distinguish "returned and read" from "returned and scrolled past", because the client only ever told it the former. Two mitigations, and I would offer both: mark seen on a dwell signal from the client rather than on render, which is a one-line API change and makes the flag mean what it says; and put a TTL on seen-ness, which `SeenStore.prune` already implements at 30 days, so the worst case is a post reappearing a month later rather than never again. What I would not do is keep an unbounded "seen" set with no expiry, because that is a store that grows with every scroll and never shrinks.

**P5. "`affinity(viewer)` copies the viewer's whole author map into every `RankContext`. For a user following 5,000 accounts that is 5,000 floats per request, built and thrown away."**
Correct, and it is the last unbounded thing in the read path. The store is keyed per viewer now, which fixed the worse version — the flat `(viewer, author)` key meant a single reader's page scanned every interaction in the system, so one page cost O(everyone). What remains is O(this viewer's authors), which for the tail is thousands. The fix is to keep only the top-K authors by count per viewer, since the squash makes author number 4,000 worth about 0.0002 of score — below the recency term's resolution — and to build the map once per request rather than per stage. Named rather than left as an exercise.

---

## 12. Cheat sheet

**The one-line answer.** The timeline is a cache; a post reaches a reader either by being written into their timeline or by being found at read time, and the fan-out decision is made per author at post time. Pagination walks an order that cannot move — `(created_at, post_id)` — and ranking happens *within* a page, never across it.

| Thing to remember | The version that is wrong |
|---|---|
| Fan-out decided per author, at post time | One global push-or-pull policy |
| `PULL` is a normal policy outcome, not an error | Signalling it as a failure |
| The bound is enforced by the service, and `downgraded` is alarmable | Trusting the policy to respect its own threshold |
| `delete` = tombstone **and** `remove_post` | Either one alone |
| The timeline is derived: truncate, rebuild, repair on read | Treating it as the source of truth |
| Cursor = `(created_at, post_id)`, opaque | A score, or an offset |
| The cursor is the oldest **served** post | The last ranked item, or the oldest fetched |
| The walk stops at the page boundary | Over-fetch a window, filter, cut |
| Seen-filter **while** walking, before the page fills | Filter after cutting, and return short pages |
| Ranking is within the page | Ranking across a cursor that cannot express it |
| Hydrate a page in one `get_many` | One `get` per post |
| Dedupe push and pull through one dict keyed by id | Dedupe after ranking |
| Tie-break every score with the post id | A score alone, and a different order per run |
| One lock per store; never hold one across another | One lock for the service, or holding the post lock through the fan-out |
| `eq=False` for entities, frozen dataclasses for values | One rule for both |

**The 60-second close.** Write path: policy decides push, pull or push-to-active; the service enforces a bound and reports a downgrade. Read path: walk the timeline in `(created_at, post_id)` order, dropping tombstones, seen posts and own posts *while* scanning, stopping at the page size; merge the bounded pull; rank within the page; hydrate in one batch; return the oldest served post's pair as the cursor. Concurrency: four locks, one snapshot per fan-out, and no lock held across a store boundary.

**Cross-links**

- The HLD this goes one level down from: [`04-instagram-news-feed.md`](../HLD/04-instagram-news-feed.md) — its §6 failure sections are the source of §8's rows.
- The other LLD bridges: [`01-url-shortener-lld.md`](./01-url-shortener-lld.md) (the cursor and the id-generation authority), [`02-whatsapp-lld.md`](./02-whatsapp-lld.md) (per-recipient state as a matrix), [`03-uber-matching-lld.md`](./03-uber-matching-lld.md) (a policy object and a bounded scan).
- Technique, not repeated here: [`../../LLD/problems/`](../../LLD/problems/) — Strategy and Composite for the pipeline, Repository for the stores, and the classic "design a news feed" walkthrough.
- The framework and the primitives in one page: [`../../Cheat-Sheets/07-lld-cheatsheet.md`](../../Cheat-Sheets/07-lld-cheatsheet.md).
- The spoken-answer drill: [`../../Interview-Questions/08-lld-questions.md`](../../Interview-Questions/08-lld-questions.md).

**Grounding note.** This file is a design written to be defensible in an interview, not a description of any production system, and nothing here is drawn from the source transcripts or the reference repositories — those cover HLD material and name LLD topics without teaching them, so the class-level design, the code and every number in it are worked out here rather than recalled from a source. The HLD chapter it bridges from is the only grounding, and where this file goes beyond it — the cursor's exact semantics, the walk's stopping rule, the lock discipline — that is the part to treat as *a* design rather than *the* design, and to argue with.

```

