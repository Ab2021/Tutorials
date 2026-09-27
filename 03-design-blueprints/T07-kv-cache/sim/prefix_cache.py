"""Prefix caching by block hash, and block sharing between sibling sequences.

Two distinct things share the same mechanism (a refcounted block) and get conflated constantly:

  * PREFIX CACHE  -- a *cache*: blocks keyed by content hash, surviving across requests,
                     evicted under pressure, re-adopted on a hash hit.
  * BLOCK SHARING -- a *binding*: several live sequences (best-of-N siblings, a fork) point at
                     the same physical prefix blocks. The refcount is the invariant; no eviction
                     policy is involved at all.

A design that implements sharing as "a cache" and a cache as "a binding" gets both wrong: the
cache leaks blocks nobody references, and the binding drops a prefix that a live sequence is
still reading.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def block_hash(parent_hash: str, tokens: tuple, extra: str = "") -> str:
    """Chain the hash: block N's key includes block N-1's hash.

    This is what makes the cache a RADIX tree rather than a bag of blocks. Hashing token
    blocks independently would let a hit match a block that is identical in isolation but sits
    under a different prefix -- which produces confident, wrong output. Chaining makes a hit
    mean "the entire prefix from position 0 matches", which is the only claim that is true.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(parent_hash.encode())
    for t in tokens:
        h.update(str(t).encode())
        h.update(b",")
    h.update(extra.encode())      # e.g. a LoRA adapter id or a multimodal hash
    return h.hexdigest()


@dataclass
class CacheEntry:
    key: str
    block: int
    n_tokens: int
    hits: int = 0
    last_used: int = 0            # logical clock, for LRU


class PrefixCache:
    """Content-addressed block cache with a radix-chained key.

    The `extra` field exists because a prefix is not identified by its tokens alone: two
    requests with identical tokens but different LoRA adapters, different image hashes, or
    different system-prompt *versions* must NOT share KV. Getting this wrong is one of the few
    KV bugs that produces silently wrong answers rather than a crash, so it is a first-class
    parameter rather than an afterthought.
    """

    def __init__(self, pool, block_size: int = 16, capacity_blocks: int | None = None):
        self.pool = pool
        self.block_size = block_size
        self.capacity = capacity_blocks if capacity_blocks is not None else pool.total
        self.entries: dict[str, CacheEntry] = {}
        self.clock = 0
        self.stats = {"hits": 0, "misses": 0, "tokens_reused": 0,
                      "tokens_prefilled": 0, "evictions": 0}

    # -- lookup ------------------------------------------------------------------------

    def lookup(self, tokens: list[int], extra: str = "") -> tuple[int, list[int]]:
        """Longest cached prefix. Returns (n_tokens_hit, [block ids to retain]).

        A partial final block is NOT reused: the last cached block may be partially filled and
        its KV for the empty slots was never computed. Reusing it would read uninitialised
        memory as if it were K/V. The hit therefore stops at the last COMPLETE block.
        """
        parent = ""
        hit_tokens = 0
        blocks: list[int] = []
        n_full = len(tokens) // self.block_size

        for i in range(n_full):
            chunk = tuple(tokens[i * self.block_size:(i + 1) * self.block_size])
            key = block_hash(parent, chunk, extra)
            e = self.entries.get(key)
            if e is None:
                break
            self.clock += 1
            e.last_used = self.clock
            e.hits += 1
            blocks.append(e.block)
            hit_tokens += self.block_size
            parent = key

        if blocks:
            self.stats["hits"] += 1
            self.stats["tokens_reused"] += hit_tokens
        else:
            self.stats["misses"] += 1
        self.stats["tokens_prefilled"] += len(tokens) - hit_tokens
        return hit_tokens, blocks

    def retain_all(self, blocks: list[int]) -> None:
        """Bind the hit blocks into a live sequence's table."""
        for b in blocks:
            self.pool.retain(b)

    # -- insert ------------------------------------------------------------------------

    def insert(self, tokens: list[int], blocks: list[int], extra: str = "") -> int:
        """Record `blocks` as the cached prefix for `tokens`. Returns entries created.

        The caller has already allocated the blocks for the live sequence, so the cache holds a
        SECOND reference to each: one for the sequence, one for the cache. That is why a
        cached block survives the sequence that created it finishing -- and why forgetting to
        release the cache's reference leaks the pool.
        """
        parent = ""
        created = 0
        n_full = len(tokens) // self.block_size
        for i in range(min(n_full, len(blocks))):
            chunk = tuple(tokens[i * self.block_size:(i + 1) * self.block_size])
            key = block_hash(parent, chunk, extra)
            if key not in self.entries:
                if len(self.entries) >= self.capacity:
                    self._evict_one()
                self.pool.retain(blocks[i])
                self.clock += 1
                self.entries[key] = CacheEntry(key=key, block=blocks[i],
                                               n_tokens=len(tokens),
                                               last_used=self.clock)
                created += 1
            parent = key
        return created

    # -- eviction ----------------------------------------------------------------------

    def _evict_one(self) -> bool:
        """LRU eviction. Returns False if nothing was evictable.

        Eviction releases only the CACHE's reference. If a live sequence still holds the block,
        the refcount stays > 0 and the block is not freed -- the sequence keeps reading correct
        KV. A cache that freed unconditionally would corrupt live requests, which is the
        classic bug this separation exists to prevent.
        """
        if not self.entries:
            return False
        victim = min(self.entries.values(), key=lambda e: e.last_used)
        self.pool.release(victim.block)
        del self.entries[victim.key]
        self.stats["evictions"] += 1
        return True

    def evict_to(self, target_blocks: int) -> int:
        """Evict LRU until at most `target_blocks` cache entries remain."""
        n = 0
        while len(self.entries) > target_blocks and self._evict_one():
            n += 1
        return n

    # -- accounting --------------------------------------------------------------------

    @property
    def retained_blocks(self) -> int:
        return len(self.entries)

    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total else 0.0

    def reuse_fraction(self) -> float:
        reused = self.stats["tokens_reused"]
        total = reused + self.stats["tokens_prefilled"]
        return reused / total if total else 0.0


# --------------------------------------------------------------------------------------
# Sibling sharing (best-of-N, forking) -- NOT a cache
# --------------------------------------------------------------------------------------

def fork(parent_blocks: list[int], pool) -> list[int]:
    """Create a sibling sequence sharing the parent's prefix blocks.

    Used by best-of-N (T05): n candidates share one prompt prefix. Without this, n candidates
    cost n x the prompt KV -- and at a 32k prompt that is the difference between n=32 fitting
    and n=32 being impossible.
    """
    for b in parent_blocks:
        pool.retain(b)
    return list(parent_blocks)


def copy_on_write(table_ref: list[int], idx: int, pool) -> int:
    """Divergence: the sibling must write token idx, so its block can no longer be shared.

    COW is the reason sharing is safe. Until two sequences actually disagree about a block's
    contents, they share it; the first writer pays for a private copy.
    """
    old = table_ref[idx]
    if pool.refcount[old] > 1:
        pool.release(old)
        new = pool.allocate()
        table_ref[idx] = new
        return new
    return old


def sharing_saving(n_siblings: int, prompt_blocks: int, gen_blocks: int) -> dict:
    """KV cost of best-of-N with and without prefix sharing.

    The result is the argument for sharing: without it, cost scales with n over the WHOLE
    sequence; with it, only the generated part scales with n.
    """
    if n_siblings < 1:
        raise ValueError("n_siblings must be >= 1")
    unshared = n_siblings * (prompt_blocks + gen_blocks)
    shared = prompt_blocks + n_siblings * gen_blocks
    return {"unshared_blocks": unshared, "shared_blocks": shared,
            "saving": 1.0 - shared / unshared if unshared else 0.0}
