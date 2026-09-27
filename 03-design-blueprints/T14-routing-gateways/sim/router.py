"""Gateway routing policies, and the difference between a precise and an approximate
view of where the KV cache actually is.

The mechanism: a router's job is not to pick the least busy box. It is to build a live
view of where the cache is, FILTER the candidate set, then RANK what is left. The corpus
describes exactly that pipeline, and describes the router keeping that view from the KV
create and evict events the engine emits. [T]

A router that keeps only the CREATE events drifts: it believes a prefix is present after
it has been evicted. The corpus names this as the approximate approach, notes it is
hash-based, and says it "has all the consistency problems". This module makes that
drift measurable. [T]

Nothing here measures hardware. It is a model of a routing policy. [D]
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Instance:
    name: str
    cache_capacity: int          # how many distinct prefixes fit in this instance's KV
    kv: list[int] = field(default_factory=list)   # LRU order, oldest first
    active: int = 0
    hits: int = 0
    misses: int = 0
    routed: int = 0
    misroutes: int = 0     # sent here on a miss while ANOTHER instance held the prefix

    def holds(self, prefix: int) -> bool:
        return prefix in self.kv

    def remember(self, prefix: int) -> None:
        if prefix in self.kv:
            self.kv.remove(prefix)
        self.kv.append(prefix)
        while len(self.kv) > self.cache_capacity:
            self.kv.pop(0)          # evict LRU -- a NORMAL lifecycle event [T]


@dataclass(frozen=True)
class Req:
    rid: int
    prefix: int
    steps: int = 1               # how long the request occupies the instance


@dataclass
class RouterView:
    """What the router BELIEVES is cached, per instance.

    `precise`  -- updated on create AND evict events, so it matches reality.
    `approx`   -- updated on create only; hash-based; never learns about eviction.
    """
    precise: bool
    belief: dict[str, set[int]] = field(default_factory=dict)

    def note_create(self, inst: str, prefix: int) -> None:
        self.belief.setdefault(inst, set()).add(prefix)

    def note_evict(self, inst: str, prefix: int) -> None:
        if self.precise:
            self.belief.setdefault(inst, set()).discard(prefix)
        # approximate routers never receive (or never apply) the evict event [T]

    def believes(self, inst: str, prefix: int) -> bool:
        return prefix in self.belief.get(inst, set())

    def accuracy(self, instances: list[Instance]) -> float:
        """Share of (instance, prefix) beliefs that match reality."""
        agree = 0
        total = 0
        for inst in instances:
            real = set(inst.kv)
            believed = self.belief.get(inst.name, set())
            total += len(believed | real)
            agree += len(believed & real)
        return agree / total if total else 1.0


# --------------------------------------------------------------------------- policies


def _mix(x: int) -> int:
    """A cheap avalanche hash, so hash-based routing is not correlated with prefix id --
    which would accidentally balance perfectly and flatter the policy. [D]"""
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 0x27D4EB2D
    x = x ^ (x >> 15)
    return x & 0x7FFFFFFF


def _pick(policy: str, r: Req, instances: list[Instance], view: RouterView,
          cursor: list[int], rng: random.Random) -> Instance:
    if policy == "round_robin":
        cursor[0] = (cursor[0] + 1) % len(instances)
        return instances[cursor[0]]

    if policy == "least_loaded":
        return min(instances, key=lambda i: (i.active, i.routed))

    if policy == "prefix_approx":
        # The corpus's description of the approximate approach: hash the request and
        # thereby "know" which instance it goes to -- no cache state consulted at all.
        # It is stable per prefix, which is why it is not useless; it simply cannot
        # adapt when the cache moves, is evicted, or when an instance is saturated. [T]
        return instances[_mix(r.prefix) % len(instances)]

    if policy == "prefix_precise":
        # FILTER: instances the router believes hold this prefix.
        candidates = [i for i in instances if view.believes(i.name, r.prefix)]
        if not candidates:
            candidates = instances
        # RANK: among the candidates, least loaded -- the corpus's token-load / active
        # request scorer, reduced here to one number. [T]
        return min(candidates, key=lambda i: (i.active, i.routed))

    raise ValueError(f"unknown policy: {policy}")


def simulate(n_instances: int = 8, n_requests: int = 2000, hot_prefixes: int = 64,
             cache_capacity: int = 24, policy: str = "prefix_precise",
             zipf_exponent: float = 1.1, seed: int = 3,
             scale_to: int = 0, scale_at_frac: float = 0.5
             ) -> tuple[list[Instance], RouterView]:
    """Prefixes are Zipf-distributed: a few are hot, most are rare.

    That distribution is what makes prefix-aware routing worth anything -- with uniform
    prefixes every instance would miss equally often. [D]
    """
    rng = random.Random(seed)
    instances = [Instance(f"i{k}", cache_capacity) for k in range(n_instances)]
    view = RouterView(precise=(policy == "prefix_precise"))
    cursor = [0]

    # Zipf-ish sampling over prefix ids
    weights = [1.0 / ((k + 1) ** zipf_exponent) for k in range(hot_prefixes)]
    total_w = sum(weights)
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / total_w
        cum.append(acc)

    def draw_prefix() -> int:
        u = rng.random()
        for k, c in enumerate(cum):
            if u <= c:
                return k
        return len(cum) - 1

    scale_at = int(n_requests * scale_at_frac) if scale_to else -1

    for rid in range(n_requests):
        if rid == scale_at:
            # A scale-out. Every index-based hash mapping now points somewhere else -- the
            # classic consistency problem the corpus names. A consistent-hash ring would
            # move only ~1/n of prefixes instead of all of them, but even then the router
            # holds no evict signal, so it cannot tell a remapped-and-cold prefix from a
            # remapped-and-still-warm one. [T] for the claim, [D] for the model of it.
            while len(instances) < scale_to:
                instances.append(Instance(f"i{len(instances)}", cache_capacity))

        r = Req(rid, draw_prefix())
        inst = _pick(policy, r, instances, view, cursor, rng)
        inst.routed += 1
        inst.active += r.steps
        hit = inst.holds(r.prefix)
        if hit:
            inst.hits += 1
        else:
            inst.misses += 1
            # MISROUTE: this instance cannot reuse the prefix, but another one could
            # have. The prefill is recomputed while the cached copy sits idle on a
            # different box. That waste is the real cost of a stale view -- it does not
            # show up in a hit-rate average per instance. [D]
            if any(o.holds(r.prefix) for o in instances if o is not inst):
                inst.misroutes += 1
            # The engine emits a create event; the router learns. [T]
            view.note_create(inst.name, r.prefix)
        # Cache the prefix (a hit refreshes LRU position). Eviction may drop something.
        before = set(inst.kv)
        inst.remember(r.prefix)
        after = set(inst.kv)
        for gone in before - after:
            # A real engine emits an evict event here. Approximate routers do not apply it.
            view.note_evict(inst.name, gone)
        inst.active -= r.steps
    return instances, view


def summarise(instances: list[Instance], view: RouterView) -> dict:
    hits = sum(i.hits for i in instances)
    misses = sum(i.misses for i in instances)
    routed = sum(i.routed for i in instances)
    loads = [i.routed for i in instances]
    mis = sum(i.misroutes for i in instances)
    mean_load = routed / len(instances) if instances else 0.0
    peak = max(loads) if loads else 0
    return {
        "hit_rate": hits / (hits + misses) if (hits + misses) else 0.0,
        "hits": hits,
        "misses": misses,
        "misroutes": mis,
        "misroute_rate": mis / (hits + misses) if (hits + misses) else 0.0,
        "routed": routed,
        "load_spread": (max(loads) / min(loads)) if loads and min(loads) > 0 else float("inf"),
        "load_imbalance": (peak / mean_load) if mean_load else 0.0,
        # The number that actually connects routing to flow control. The busiest instance
        # saturates first, so the CLUSTER's usable capacity is not n * per_instance -- it
        # is that divided by how far the peak sits above the mean. This is arithmetic on
        # the model's own output, not a measurement. [D]
        "effective_capacity": (len(instances) / (peak / mean_load)) if mean_load and peak else 0.0,
        "view_accuracy": view.accuracy(instances),
    }


POLICIES = ("round_robin", "least_loaded", "prefix_approx", "prefix_precise")


def sweep(**kw) -> list[tuple[str, dict]]:
    out = []
    for p in POLICIES:
        inst, view = simulate(policy=p, **kw)
        out.append((p, summarise(inst, view)))
    return out


def render(rows) -> str:
    lines = [f"  {'policy':<16} {'hit rate':>9} {'misroutes':>10} {'misroute %':>11} "
             f"{'peak/mean':>10} {'eff cap':>8} {'view acc':>9}"]
    lines.append("  " + "-" * 76)
    for name, s in rows:
        lines.append(f"  {name:<16} {s['hit_rate']:>9.1%} "
                     f"{s['misroutes']:>10} {s['misroute_rate']:>11.1%} "
                     f"{s['load_imbalance']:>10.2f} {s['effective_capacity']:>8.2f} "
                     f"{s['view_accuracy']:>9.1%}")
    return "\n".join(lines)
