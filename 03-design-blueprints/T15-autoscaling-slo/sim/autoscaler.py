"""T15 — the autoscaling control loop, modelled end to end.

The mechanism this module exists to demonstrate: for LLM serving, the choice of
SCALING SIGNAL matters less than the choice of SCALING MECHANISM, and the mechanism is
dominated by a delay the signal cannot remove -- the cold start.

Three findings the model is built to make visible:

  1. A compressed, lagged signal (CPU utilisation) fires late; a queue- or KV-derived
     signal fires on time. The corpus's two guide files disagree about which to use, and
     the disagreement is recorded in HLD section 4 rather than resolved. [R]
  2. Even a PERFECT instantaneous signal still loses to a predictive one, because a
     replica that starts booting now serves nothing for boot_steps. That result does not
     depend on assumption 1 at all, which is why the model tests both.
  3. Booting replicas COST MONEY AND SERVE NOTHING. Any cost comparison that counts only
     ready replicas flatters reactive autoscaling by exactly the boot window.

Nothing here is a benchmark. No cluster scaled, no GPU booted, no SLO was measured. [D]
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- scenario


@dataclass
class Scenario:
    """A bursty workload against a fleet that takes time to grow."""

    n_steps: int = 600
    base_rate: float = 8.0        # requests/step before the burst
    peak_rate: float = 44.0       # requests/step during it
    burst_at: int = 180
    burst_len: int = 240
    settle_steps: int = 120       # trailing low period, tests scale-DOWN

    rep_capacity: float = 10.0    # requests/step one READY replica can serve
    boot_steps: int = 20          # cold start, in steps. 15-20s is the corpus's
                                  # best case from un-quantized base images [R]
    slo_wait_steps: int = 3       # queue wait budget
    min_reps: int = 2
    max_reps: int = 48
    cooldown_up_steps: int = 15   # anti-thrash hysteresis, scale-UP direction
    cooldown_down_steps: int = 0  # 0 means "derive from the boot time" -- see below
    seed: int = 7

    def down_cooldown(self) -> int:
        """Scale DOWN must be slower than scale UP, and specifically slower than the
        cold start. If it is not, the controller cancels replicas it has already paid
        to start: they are still booting when the signal that justified them lapses,
        and the capacity never arrives. Deriving this from boot_steps rather than
        hard-coding it is the single most valuable line in this file. [D]"""
        if self.cooldown_down_steps:
            return self.cooldown_down_steps
        return max(30, 2 * self.boot_steps)

    def rate_at(self, step: int) -> float:
        """A plateau, not a spike: a burst that lasts long enough that a control loop
        COULD track it if it reacted in time. A spike would only measure the delay."""
        if self.burst_at <= step < self.burst_at + self.burst_len:
            return self.peak_rate
        return self.base_rate


# --------------------------------------------------------------------------- state


@dataclass
class Replica:
    born: int
    ready_at: int

    def ready(self, step: int) -> bool:
        return step >= self.ready_at


@dataclass
class Result:
    policy: str
    met: int = 0
    total: int = 0
    waits: list[int] = field(default_factory=list)
    replica_steps: float = 0.0     # cost: replica-steps, READY OR BOOTING
    ready_steps: float = 0.0       # only the ones that could actually serve
    scale_events: int = 0
    peak_reps: int = 0
    peak_queue: int = 0
    unserved: int = 0

    @property
    def slo_rate(self) -> float:
        return self.met / self.total if self.total else 0.0

    @property
    def mean_wait(self) -> float:
        return sum(self.waits) / len(self.waits) if self.waits else 0.0

    @property
    def boot_waste(self) -> float:
        """Share of replica-steps spent not serving. This is the number that makes
        reactive autoscaling look worse than it does on a ready-replica count."""
        return (1.0 - self.ready_steps / self.replica_steps) if self.replica_steps else 0.0


# --------------------------------------------------------------------------- signals


def cpu_signal(outstanding: float, true_capacity: float) -> float:
    """CPU utilisation as a COMPRESSED, LAGGED function of true load.

    This is the modelling assumption the whole CPU-versus-queue comparison rests on, so
    it is stated plainly rather than buried. In LLM serving the CPU orchestrates --
    tokenisation, batching, HTTP -- while the accelerator does the work. CPU
    utilisation therefore rises slowly, saturates well below 100%, and is only loosely
    coupled to whether requests are meeting their latency budget. [D]

    Formula:  cpu = 0.15 + 0.70 * min(1, outstanding / (2.5 * true_capacity))
              outstanding = inflight + queue

    Two things this encodes. First, the signal is COMPRESSED: it tops out at 0.85 even
    when the fleet is hopelessly behind, so a 0.70 target is reached only at 2.5x
    overload. Second, and more important for the result, the HPA's own control law --
    desired = ceil(current * signal / target) -- moves GEOMETRICALLY. It approaches the
    target asymptotically rather than jumping to it, which is why an HPA far from its
    setpoint is slow even when the signal is live.

    The 2.5 and the 0.85 ceiling are assumptions, not measurements -- and run.py section
    2 runs the variant where CPU tracks load PERFECTLY, so the conclusion does not rest
    on them.
    """
    return 0.15 + 0.70 * min(1.0, outstanding / (2.5 * true_capacity))


def cpu_signal_ideal(outstanding: float, true_capacity: float) -> float:
    """The best case for the CPU signal: linear, uncompressed, no lag."""
    return min(1.0, outstanding / max(true_capacity, 1e-9))


def kv_pressure(inflight: float, true_capacity: float) -> float:
    """KV-cache pressure: the corpus's suggested signal -- "scaling based on KV Cache
    utilization rather than CPU or standard memory usage" [R].

    Unlike CPU it saturates proportionally to load, because every in-flight request
    holds KV blocks for its whole life. Modelled as linear in inflight.
    """
    return min(1.0, inflight / max(true_capacity, 1e-9))


# --------------------------------------------------------------------------- policies


@dataclass
class PolicyState:
    last_scale: int = -10_000
    scale_events: int = 0
    # The predictor's estimate is built from OBSERVED arrivals, sampled every
    # `window` steps, so the ramp it extrapolates is a trend over several samples
    # rather than the difference between two adjacent ones. A two-point ramp is a
    # pulse: it is large for exactly one window during a step change and then
    # vanishes, which makes the controller over-scale and then immediately cancel
    # the very replicas it just started booting.
    acc: float = 0.0
    acc_n: int = 0
    samples: list[float] = field(default_factory=list)
    window: int = 10
    n_samples: int = 6

    def observe(self, arrivals: int) -> None:
        self.acc += arrivals
        self.acc_n += 1
        if self.acc_n >= self.window:
            self.samples.append(self.acc / self.acc_n)
            self.acc = 0.0
            self.acc_n = 0
            if len(self.samples) > self.n_samples:
                self.samples.pop(0)

    def ramp_per_sample(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return (self.samples[-1] - self.samples[0]) / (len(self.samples) - 1)


def decide(policy: str, sc: Scenario, step: int, reps: list[Replica], queue: int,
           inflight: float, st: PolicyState) -> int:
    """Return the desired replica count for this step."""
    ready = [r for r in reps if r.ready(step)]
    true_capacity = max(len(ready), 1) * sc.rep_capacity

    if policy == "static_min":
        return sc.min_reps

    if policy == "static_peak":
        # Overprovision for the peak. The baseline every autoscaler must beat.
        return max(sc.min_reps, math.ceil(sc.peak_rate / sc.rep_capacity))

    if policy in ("cpu", "cpu_ideal"):
        sig = (cpu_signal if policy == "cpu" else cpu_signal_ideal)(inflight + queue,
                                                                    true_capacity)
        # The HPA shape the first guide file ships: target 70% utilisation. [R]
        return max(sc.min_reps, min(sc.max_reps, math.ceil(len(reps) * sig / 0.70)))

    if policy == "queue":
        # Reactive on the queue: scale to drain it at a target utilisation. Note this
        # includes replicas still BOOTING -- a real controller counts pending pods --
        # which is why it overshoots and oscillates (run.py section 3).
        want = math.ceil((queue + inflight) / (sc.rep_capacity * 0.75))
        return max(sc.min_reps, min(sc.max_reps, want))

    if policy == "queue_predict":
        # The same signal, projected forward by the boot window. Because a replica
        # started now serves nothing for boot_steps, reacting to the CURRENT queue is
        # always boot_steps too late. This is the whole point of the policy.
        if not st.samples:
            return sc.min_reps
        now = st.samples[-1]
        ramp = st.ramp_per_sample()
        # Convert the ramp to per-step units, then extrapolate across the boot window.
        projected = max(0.0, now + ramp / st.window * sc.boot_steps)
        want = math.ceil(projected * 1.15 / sc.rep_capacity)
        return max(sc.min_reps, min(sc.max_reps, want))

    raise ValueError(f"unknown policy: {policy}")


# --------------------------------------------------------------------------- the loop


def simulate(sc: Scenario, policy: str) -> Result:
    rng = random.Random(sc.seed)
    res = Result(policy=policy)
    st = PolicyState()

    reps: list[Replica] = [Replica(0, 0) for _ in range(sc.min_reps)]
    queue: list[int] = []          # arrival steps, FIFO
    n_steps = sc.burst_at + sc.burst_len + sc.settle_steps

    for step in range(n_steps):
        # ---- arrivals (the only thing a controller can actually observe)
        rate = sc.rate_at(step)
        arrived = _poisson(rng, rate)
        for _ in range(arrived):
            queue.append(step)
        st.observe(arrived)

        # ---- observe
        ready = [r for r in reps if r.ready(step)]
        inflight = min(len(queue), len(ready) * sc.rep_capacity)

        # ---- decide, with cooldown hysteresis so the loop cannot flap every step
        desired = decide(policy, sc, step, reps, len(queue), inflight, st)
        up = desired > len(reps)
        cd = sc.cooldown_up_steps if up else sc.down_cooldown()
        if step - st.last_scale >= cd:
            if up:
                for _ in range(desired - len(reps)):
                    reps.append(Replica(step, step + sc.boot_steps))
                st.scale_events += 1
                st.last_scale = step
            elif desired < len(reps):
                # Remove the newest first -- a still-booting replica is the cheapest to
                # cancel, since its cost is the least sunk.
                drop = min(len(reps) - desired, len(reps))
                for r in reps[-drop:]:
                    reps.remove(r)
                st.scale_events += 1
                st.last_scale = step

        # ---- serve
        ready = [r for r in reps if r.ready(step)]
        capacity = int(len(ready) * sc.rep_capacity)
        for _ in range(min(capacity, len(queue))):
            arrived = queue.pop(0)
            wait = step - arrived
            res.total += 1
            res.waits.append(wait)
            if wait <= sc.slo_wait_steps:
                res.met += 1

        # ---- account
        res.replica_steps += len(reps)
        res.ready_steps += len(ready)
        res.peak_reps = max(res.peak_reps, len(reps))
        res.peak_queue = max(res.peak_queue, len(queue))

    res.scale_events = st.scale_events
    res.unserved = len(queue)
    res.total += len(queue)          # requests still queued at the end never met an SLO
    return res


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's method. Deterministic given the seed."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


POLICIES = ("static_min", "static_peak", "cpu", "cpu_ideal", "queue", "queue_predict")


def run_all(sc: Scenario) -> list[Result]:
    return [simulate(sc, p) for p in POLICIES]


def render(rows: list[Result], baseline: Result | None = None) -> str:
    lines = [f"  {'policy':<15} {'SLO met':>8} {'mean wait':>10} {'replica-steps':>14} "
             f"{'boot waste':>11} {'scales':>7} {'peak q':>7}"]
    lines.append("  " + "-" * 78)
    for r in rows:
        cost = f"{r.replica_steps:,.0f}"
        if baseline is not None and baseline.replica_steps:
            cost += f" ({r.replica_steps / baseline.replica_steps:.2f}x)"
        lines.append(f"  {r.policy:<15} {r.slo_rate:>8.1%} {r.mean_wait:>10.2f} "
                     f"{cost:>14} {r.boot_waste:>11.1%} {r.scale_events:>7} "
                     f"{r.peak_queue:>7}")
    return "\n".join(lines)


def cooldown_comparison(sc: Scenario, policy: str = "queue") -> tuple[Result, Result]:
    """Naive symmetric hysteresis vs. a scale-down cooldown derived from the boot time.

    The rule this exists to demonstrate: if scale-down is as fast as scale-up, the
    controller cancels replicas that are still booting. They were paid for, they never
    served, and the capacity the signal asked for never arrives.
    """
    naive = simulate(Scenario(**{**sc.__dict__, "cooldown_down_steps": sc.cooldown_up_steps}),
                     policy)
    derived = simulate(sc, policy)
    return naive, derived


def render_cooldown(naive: Result, derived: Result, boot: int) -> str:
    lines = [f"  {'scale-down hysteresis':<28} {'SLO met':>8} {'replica-steps':>14} "
             f"{'boot waste':>11}"]
    lines.append("  " + "-" * 64)
    lines.append(f"  {'naive (same as scale-up)':<28} {naive.slo_rate:>8.1%} "
                 f"{naive.replica_steps:>14,.0f} {naive.boot_waste:>11.1%}")
    lines.append(f"  {'derived (2x boot = ' + str(2 * boot) + ')':<28} "
                 f"{derived.slo_rate:>8.1%} {derived.replica_steps:>14,.0f} "
                 f"{derived.boot_waste:>11.1%}")
    return "\n".join(lines)


def render_boot_sweep(sc: Scenario, policies=("cpu", "queue", "queue_predict")) -> str:
    """The boot delay is the dominant term, so sweep it rather than asserting it."""
    lines = [f"  {'boot steps':>11} " + " ".join(f"{p:>16}" for p in policies)]
    lines.append("  " + "-" * (13 + 17 * len(policies)))
    for boot in (5, 20, 60, 180):
        s2 = Scenario(**{**sc.__dict__, "boot_steps": boot})
        cells = []
        for p in policies:
            r = simulate(s2, p)
            cells.append(f"{r.slo_rate:>7.1%} {r.replica_steps:>7,.0f}")
        lines.append(f"  {boot:>11} " + " ".join(f"{c:>16}" for c in cells))
    lines.append(f"  {'':>11} " + " ".join(f"{'SLO':>7} {'cost':>7}" for _ in policies))
    return "\n".join(lines)
