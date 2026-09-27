"""Flow control: what a gateway does when the cluster is saturated.

The corpus is precise about the shape of this. Routing decides PLACEMENT; flow control
handles ADMISSION, and it only matters once saturation is detected. Saturation is
operator-defined -- e.g. "KV cache 80% full" or "average active requests above eight".
Once saturated, the endpoint picker queues, and the queue policy is a choice. [T]

The corpus's own read on the naive choice: first-come-first-serve "doesn't add anything
extra to your policies because it's going to slow down everything". What it suggests
instead is priority bands -- under saturation only premium traffic is dispatched. [T]

This module measures the cost of that trade, including the part that is easy to leave
unsaid: best-effort traffic is not slowed, it is starved.

Model output only. No cluster was saturated. [D]
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

PREMIUM = "premium"
BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class Arrival:
    rid: int
    step: int
    priority: str
    slo_steps: int          # the latency budget this class is promised


@dataclass
class FlowStats:
    dispatched: int = 0
    shed: int = 0
    premium_met: int = 0
    premium_total: int = 0
    be_met: int = 0
    be_total: int = 0
    wait_steps: list[int] = field(default_factory=list)

    def met_rate(self, priority: str) -> float:
        if priority == PREMIUM:
            return self.premium_met / self.premium_total if self.premium_total else 0.0
        return self.be_met / self.be_total if self.be_total else 0.0

    @property
    def mean_wait(self) -> float:
        return sum(self.wait_steps) / len(self.wait_steps) if self.wait_steps else 0.0


@dataclass
class Scenario:
    n_steps: int = 400
    arrival_rate: float = 12.0        # requests per step offered
    dispatch_capacity: int = 8        # requests per step the cluster can actually take
    premium_fraction: float = 0.25
    premium_slo: int = 4
    best_effort_slo: int = 20
    saturation_active: int = 8        # operator-defined: active requests above this [T]
    seed: int = 13


def make_arrivals(sc: Scenario) -> list[Arrival]:
    rng = random.Random(sc.seed)
    out = []
    rid = 0
    for step in range(sc.n_steps):
        n = _poisson(rng, sc.arrival_rate)
        for _ in range(n):
            prem = rng.random() < sc.premium_fraction
            out.append(Arrival(rid, step, PREMIUM if prem else BEST_EFFORT,
                               sc.premium_slo if prem else sc.best_effort_slo))
            rid += 1
    return out


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's method. Deterministic given the seed."""
    import math
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


# --------------------------------------------------------------------------- policies


def run(sc: Scenario, policy: str) -> FlowStats:
    """policy: "fcfs" | "priority_bands" | "admit_none" (no flow control at all)."""
    arrivals = make_arrivals(sc)
    by_step: dict[int, list[Arrival]] = {}
    for a in arrivals:
        by_step.setdefault(a.step, []).append(a)

    stats = FlowStats()
    queue: list[Arrival] = []
    active: list[int] = []      # finish steps

    for step in range(sc.n_steps + 200):
        # release finished work
        active = [f for f in active if f > step]
        queue.extend(by_step.get(step, []))

        # is the cluster saturated? operator-defined threshold [T]
        saturated = len(active) + len(queue) > sc.saturation_active * sc.dispatch_capacity

        capacity = sc.dispatch_capacity
        if policy == "admit_none":
            # No flow control: everything is offered to the cluster immediately. There
            # is no queue to shape, so the cluster simply falls behind.
            pass
        elif policy == "fcfs":
            # Naive default. Adds nothing: under saturation everything slows equally. [T]
            pass
        elif policy == "priority_bands":
            if saturated:
                # Under saturation, dispatch premium first; best-effort waits. Some
                # best-effort is shed rather than queued without bound.
                queue.sort(key=lambda a: (a.priority != PREMIUM, a.step))
                if len(queue) > sc.saturation_active * sc.dispatch_capacity * 4:
                    keep = queue[: sc.saturation_active * sc.dispatch_capacity * 4]
                    stats.shed += len(queue) - len(keep)
                    queue = keep
        else:
            raise ValueError(policy)

        # dispatch
        dispatched_now = queue[:capacity]
        queue = queue[capacity:]
        for a in dispatched_now:
            stats.dispatched += 1
            wait = step - a.step
            stats.wait_steps.append(wait)
            active.append(step + 1)
            if a.priority == PREMIUM:
                stats.premium_total += 1
                if wait <= a.slo_steps:
                    stats.premium_met += 1
            else:
                stats.be_total += 1
                if wait <= a.slo_steps:
                    stats.be_met += 1
        if not queue and step > sc.n_steps and not active:
            break

    # anything still queued never completed
    for a in queue:
        if a.priority == PREMIUM:
            stats.premium_total += 1
        else:
            stats.be_total += 1
    return stats


def run_all(sc: Scenario) -> list[tuple[str, FlowStats]]:
    return [(p, run(sc, p)) for p in ("admit_none", "fcfs", "priority_bands")]


def render(rows) -> str:
    lines = [f"  {'policy':<16} {'premium SLO':>12} {'best-effort SLO':>16} "
             f"{'dispatched':>11} {'shed':>6} {'mean wait':>10}"]
    lines.append("  " + "-" * 76)
    for name, s in rows:
        lines.append(f"  {name:<16} {s.met_rate(PREMIUM):>12.1%} "
                     f"{s.met_rate(BEST_EFFORT):>16.1%} {s.dispatched:>11} "
                     f"{s.shed:>6} {s.mean_wait:>10.2f}")
    return "\n".join(lines)
