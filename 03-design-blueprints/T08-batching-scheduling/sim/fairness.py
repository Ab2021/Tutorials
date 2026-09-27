"""Fairness: FCFS, priority bands, least-attained service, and turn priority.

The corpus's findings, verbatim in spirit [T] (llm-d):
  * FCFS "doesn't add anything extra to your policies because it's going to slow down everything"
  * least-attained service cut request latencies "by up to 50%... up to 2x or sometimes 3x"
  * turn priority is a different kind of mechanism that improves throughput under KV saturation
  * and the two address DIFFERENT resources: least-attained service addresses compute saturation,
    turn priority addresses KV cache saturation

That last point is the design lesson and it is easy to miss: they are not two candidates for one
slot in a config file.

THE MODEL, AND WHY IT LOOKS LIKE THIS
-------------------------------------
A dispatch policy can only matter when the work that WANTS a slot exceeds the slots available.
Everything below is arranged around that single fact:

  * slots          -- the concurrent-request limit; the scarce resource
  * max_in_flight  -- a PROGRAM's fan-out: how many of its own turns it keeps outstanding
  * hold           -- how many cycles a dispatch occupies its slot (generation time)

The corpus's "large session" is a program with a large fan-out and long turns. Because it refills
its own outstanding requests the moment any retire, it presents a permanent queue to the
scheduler -- and a naive policy then serves that queue forever while a short session that arrived
later waits behind it. That is the starvation the corpus names, and it is why fairness here is
computed per PROGRAM: a per-REQUEST policy gives the largest program the most dispatches, which is
precisely backwards.

This is an illustrative model. It reproduces the DIRECTION and the MECHANISM the corpus reports.
It does not claim to reproduce their measured magnitudes, and no corpus figure is asserted as an
output of this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TPS_PER_CYCLE = 100          # tokens a running sequence produces per cycle, for hold arithmetic


@dataclass
class AgentProgram:
    """A session: one agentic program, which issues many REQUESTS (turns).

    Fairness is computed per PROGRAM, not per request -- the corpus calls this "agentic program
    aware fairness" [T]. Getting this wrong (fairness per request) rewards the program that issues
    the most requests, which is exactly the program that needs throttling.
    """
    session_id: str
    turns: int
    tokens_per_turn: int
    max_in_flight: int = 1        # fan-out: how many turns this program keeps outstanding
    band: int = 1                 # 0 = premium, 1 = best-effort [T]
    start_cycle: int = 1          # when the session first appears
    initial_turns: int = 0        # turns already completed before we start observing
    served: int = 0               # tokens served
    completed_turns: int = 0
    issued: int = 0               # turns ever enqueued
    in_flight: int = 0
    finished_at: int | None = None

    def __post_init__(self) -> None:
        """A session may be observed MID-FLIGHT.

        The corpus's turn-priority scenario is an agent "at turn 100" [T] -- which only means
        something if the program already has 100 turns of service and a corresponding pile of
        resident KV behind it. Starting every program at zero service makes the two fairness
        policies behave identically, because neither has anything to disagree about yet.
        """
        if self.initial_turns:
            self.completed_turns = self.initial_turns
            self.issued = self.initial_turns
            self.served = self.initial_turns * self.tokens_per_turn

    @property
    def demand(self) -> int:
        return self.turns * self.tokens_per_turn

    @property
    def attained(self) -> float:
        """Least-attained service: service received / demand. The corpus's scheduling signal [T]."""
        return self.served / self.demand if self.demand else 1.0

    @property
    def remaining(self) -> int:
        return self.turns - self.completed_turns

    @property
    def hold(self) -> int:
        """Cycles a dispatched turn occupies its slot. Larger session = longer turn."""
        return max(1, round(self.tokens_per_turn / TPS_PER_CYCLE))

    def outstanding(self) -> int:
        """Turns enqueued but not yet retired: in flight plus waiting."""
        return self.issued - self.completed_turns

    @property
    def kv_tokens(self) -> int:
        """Resident KV, in context tokens: the program's accumulated conversation [T].

        An agentic session's KV GREW with every turn it completed -- the corpus's whole reason the
        retention and eviction machinery exists [T].

        RELEASED ON COMPLETION. This is not bookkeeping: it is the entire mechanism turn priority
        exploits. Finishing a near-done program hands its whole accumulated context back to the
        pool, whereas a program that is merely deprioritised keeps holding it. Forgetting this
        line makes the two fairness policies indistinguishable no matter how long the simulation
        runs, because the resource they disagree about never actually moves.
        """
        if self.finished_at is not None:
            return 0
        return self.completed_turns * self.tokens_per_turn


@dataclass
class Request:
    """One turn, once it is waiting for a slot."""
    session: str
    turn: int
    arrival: int        # the CYCLE at which this turn entered the queue
    band: int
    hold: int
    kv_charge: int = 0  # context this turn adds while it is in flight
    seq: int = 0        # global monotone enqueue counter -- tie-break only
    dispatched_at: int | None = None
    finished_at: int | None = None

    @property
    def latency(self) -> int | None:
        """Cycles from enqueue to completion. Both operands are CYCLES.

        This is the quantity the corpus reports ("reduce the request latencies by up to 50%") [T].
        Mixing clocks here -- comparing a monotone enqueue counter against a cycle number --
        produces negative latencies that still sort consistently, so the bug hides in plain sight
        and every policy comparison silently returns ~1.0.
        """
        return None if self.finished_at is None else self.finished_at - self.arrival


# --------------------------------------------------------------------------------------
# Policies -- each takes (queue, programs) and returns the queue in service order
# --------------------------------------------------------------------------------------

def fcfs_order(queue: list[Request], _programs) -> list[Request]:
    """First-come-first-served. The default, and the one the corpus says adds nothing [T].

    Sorted on the global enqueue counter and nothing else, so the policy has NO hidden preference.
    That absence is the point -- and it is also the failure: a program that continuously refills
    its own outstanding queue is continuously at the head of it.
    """
    return sorted(queue, key=lambda r: (r.seq,))


def priority_band_order(queue: list[Request], _programs) -> list[Request]:
    """Premium before best-effort [T]. Within a band, arrival order.

    The corpus's scenario: under saturation only the premium traffic is dispatched, so that
    customer-facing or interactive workloads do not suffer while batch work waits [T].
    """
    return sorted(queue, key=lambda r: (r.band, r.seq))


def least_attained_order(queue: list[Request], programs) -> list[Request]:
    """Least-attained service [T]: dispatch for the program that has received the LEAST SERVICE.

    ABSOLUTE service, not service as a fraction of demand. This distinction is the whole policy
    and getting it backwards inverts the result:

      * served / demand (proportional share) favours the program with the LARGEST demand, because a
        huge program's ratio stays near zero for a long time. It starves exactly the short sessions
        the corpus is trying to protect.
      * served, absolute, needs no knowledge of the program's size at all -- and that is precisely
        why it protects short sessions. A short session finishes before it accumulates much
        service, so it departs early; a long session keeps being pushed to the back.

    THE DIRECTION IS COUNTER-INTUITIVE. The corpus's scenario is that a large session "comes in and
    then takes all the dispatch cycles. So the shorter sessions are starving" [T]. So this protects
    the SHORT sessions from one monopolising program -- it is NOT "protect the long agent from
    short requests", which is the reverse.

    The second-order effect is the payoff: the short sessions finish fast, which frees slots, which
    lets the large one proceed too. The corpus's own summary is exactly this -- "short sessions
    finish much faster and leaving the room for larger sessions to also finish much faster" [T].
    """
    def key(r: Request):
        # served ascending -> the least-served program first; ties by enqueue order.
        return (programs[r.session].served, r.seq)
    return sorted(queue, key=key)


def turn_priority_order(queue: list[Request], programs) -> list[Request]:
    """Turn priority [T]: favour programs NEAR COMPLETION so their KV can be evicted.

    This targets a DIFFERENT resource than least-attained service: KV cache saturation, not compute
    saturation. An agent at turn 100 of 100 is likely to finish, so finishing it EVICTS its KV and
    frees memory for everyone else [T].

    Note the deliberate asymmetry: this policy is ANTI-fair. It starves the long-running program on
    purpose, because the resource being managed is memory occupancy rather than dispatch cycles.
    """
    def key(r: Request):
        return (programs[r.session].remaining, r.seq)
    return sorted(queue, key=key)


POLICIES = {
    "fcfs": fcfs_order,
    "priority_band": priority_band_order,
    "least_attained": least_attained_order,
    "turn_priority": turn_priority_order,
}


# --------------------------------------------------------------------------------------
# The flow-controlled dispatcher
# --------------------------------------------------------------------------------------

def dispatch(programs: dict[str, AgentProgram], slots: int, cycles: int = 400,
             policy: str = "fcfs", premium: set[str] | None = None,
             saturation: bool = False, kv_capacity: int = 0) -> dict:
    """Run a multi-cycle dispatch with a chosen policy.

    Each cycle: retire the finished, let every program refill its fan-out, order the waiting queue
    by the policy, and dispatch into the free slots.

    `saturation` models the corpus's flow-control trigger: when the cluster is saturated (KV ~80%
    full, or average active requests > 8 [T]) the router queues and applies the band policy, so
    only premium traffic is dispatched and the rest waits for capacity.

    `kv_capacity` is resident context tokens across the cluster (0 = unlimited). A request is
    admitted only if the total resident KV still fits after admitting it. THIS IS WHAT SEPARATES
    THE TWO FAIRNESS POLICIES: on a compute-only model, least-attained service and turn priority
    produce almost identical completion times -- because round-robin already retires short
    programs early -- and the corpus's claim that they address different resources is untestable.
    Add the memory constraint and turn priority's benefit appears, because finishing a near-done
    program RELEASES its whole accumulated KV rather than merely reshuffling dispatch.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if slots <= 0:
        raise ValueError("slots must be > 0")
    order = POLICIES[policy]
    premium = premium or set()

    # ASSIGN THE BAND FROM THE PREMIUM SET. Without this the band policy is a silent no-op: every
    # program keeps the default band of 1, `priority_band_order` degenerates to arrival order, and
    # the gated and ungated runs print IDENTICAL numbers -- which reads as "the gate does nothing"
    # rather than "the gate was never wired up".
    for p in programs.values():
        if p.session_id in premium:
            p.band = 0

    queue: list[Request] = []
    running: list[Request] = []
    requests: list[Request] = []
    enqueued = 0
    idle_total = 0
    kv_refused = 0
    kv_high_water = 0

    for cycle in range(1, cycles + 1):
        # ---- 1. retire -------------------------------------------------------------
        still: list[Request] = []
        for r in running:
            if (r.dispatched_at or 0) + r.hold <= cycle:
                r.finished_at = cycle
                p = programs[r.session]
                p.in_flight -= 1
                p.completed_turns += 1
                p.served += p.tokens_per_turn
            else:
                still.append(r)
        running = still
        free = slots - len(running)

        # ---- 2. refill fan-out -----------------------------------------------------
        # A program keeps up to `max_in_flight` turns outstanding. This is the mechanism by which
        # a large session presents a permanent queue to the scheduler.
        for p in programs.values():
            if cycle < p.start_cycle:
                continue
            while p.issued < p.turns and p.outstanding() < p.max_in_flight:
                p.issued += 1
                enqueued += 1
                r = Request(p.session_id, p.issued, cycle, p.band, p.hold,
                            kv_charge=p.tokens_per_turn, seq=enqueued)
                queue.append(r)
                requests.append(r)

        resident = sum(p.kv_tokens for p in programs.values()) + sum(r.kv_charge for r in running)
        kv_high_water = max(kv_high_water, resident)

        # ---- 3. retire programs that are entirely done -----------------------------
        for p in programs.values():
            if p.finished_at is None and p.completed_turns >= p.turns:
                p.finished_at = cycle

        if not queue and not running and not any(p.completed_turns < p.turns
                                                for p in programs.values()):
            break

        # ---- 4. order and dispatch -------------------------------------------------
        # The corpus's flow-control contract [T]: while saturated, capacity is RESERVED for
        # premium traffic, and whatever premium does not use is released back to everyone else.
        # A gate that reserved unconditionally would leave slots idle and starve best-effort
        # forever -- the request is meant to QUEUE, not to be refused.
        eligible = queue
        if saturation and premium:
            reserved = [r for r in queue if r.session in premium]
            if reserved:
                eligible = reserved

        ordered = order(eligible, programs)
        take: list[Request] = []
        for r in ordered:
            if len(take) >= free:
                break
            charge = r.kv_charge
            if kv_capacity and resident + charge > kv_capacity:
                kv_refused += 1
                continue        # the request QUEUES -- it is not dropped
            resident += charge
            take.append(r)

        idle_total += max(0, free - len(take))
        for r in take:
            queue.remove(r)
            r.dispatched_at = cycle
            programs[r.session].in_flight += 1
            running.append(r)

    latencies: dict[str, list[int]] = {sid: [] for sid in programs}
    for r in requests:
        if r.latency is not None:
            latencies[r.session].append(r.latency)

    return {"policy": policy, "slots": slots, "cycles": cycles,
            "finished_at": {sid: p.finished_at for sid, p in programs.items()},
            "served": {sid: p.served for sid, p in programs.items()},
            "attained": {sid: p.attained for sid, p in programs.items()},
            "kv_tokens": {sid: p.kv_tokens for sid, p in programs.items()},
            "kv_refused": kv_refused,
            "kv_high_water": kv_high_water,
            "latencies": latencies,
            "latency_stats": {sid: _stats(v) for sid, v in latencies.items()},
            "idle_total": idle_total}


def _stats(xs: list[int]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "p50": s[len(s) // 2], "p99": s[-1],
            "mean": sum(s) / len(s)}


def starve_report(result: dict, programs: dict[str, AgentProgram]) -> dict:
    """How badly did the SMALL programs lose, measured on REQUEST LATENCY?

    Latency, not completion cycle, is what the corpus reports ("reduce the request latencies by up
    to 50%") [T] -- a program can be slow to finish while its individual requests are each served
    promptly, and vice versa.
    """
    by_demand = sorted(programs.values(), key=lambda p: p.demand)
    small, large = by_demand[0], by_demand[-1]
    st = result["latency_stats"]
    s_mean = st.get(small.session_id, {}).get("mean")
    l_mean = st.get(large.session_id, {}).get("mean")
    return {"small_session": small.session_id, "small_mean_latency": s_mean,
            "large_session": large.session_id, "large_mean_latency": l_mean,
            "ratio": (l_mean / s_mean) if (s_mean and l_mean) else float("inf")}
