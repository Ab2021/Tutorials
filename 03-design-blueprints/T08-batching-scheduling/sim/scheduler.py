"""Continuous batching: the scheduler loop, and why static batching wastes the GPU.

The corpus's whole point in one step of the cost ladder: continuous batching takes 100 to 42 [T].
That is not a caching trick or a kernel optimisation -- it is an ADMISSION decision. A static
batch runs until the longest sequence in it finishes; a continuous batch refills a slot the instant
any sequence finishes.

Everything else in this blueprint (chunking, fairness, preemption) layers on top of this loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Seq:
    """A sequence in flight."""
    seq_id: str
    prompt_tokens: int
    output_tokens: int
    session: str = "default"
    arrival: int = 0
    decoded: int = 0
    prefill_left: int = 0
    prefilled: bool = False
    start: int | None = None
    finish: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def done(self) -> bool:
        return self.decoded >= self.output_tokens


# --------------------------------------------------------------------------------------
# Static batching -- the baseline everyone starts with and should not keep
# --------------------------------------------------------------------------------------

def static_utilisation(lengths: list[int]) -> dict:
    """Utilisation of batched decoding when the batch waits for its longest member.

    A static batch is admitted whole and retired whole, so its wall-clock is set by the MAXIMUM
    length while the useful work is the SUM. Every short sequence in the batch is a slot that
    finished early and then idled.

    This is the entire mechanism behind the corpus's 100 -> 42 [T], and the shape of the result is
    what matters: the penalty is a function of the length DISTRIBUTION, so it is severe on real
    (skewed) traffic and mild on the uniform traffic a benchmark uses.
    """
    if not lengths:
        raise ValueError("empty batch")
    mx = max(lengths)
    return {"n": len(lengths), "max_len": mx, "mean_len": sum(lengths) / len(lengths),
            "utilisation": (sum(lengths) / len(lengths)) / mx,
            "useful_token_steps": sum(lengths), "wall_token_steps": mx * len(lengths)}


def continuous_utilisation(lengths: list[int], slots: int) -> dict:
    """Utilisation when a slot is refilled the moment it frees.

    With an infinite queue behind the batch the steady-state utilisation is 1 minus the
    end-of-queue tail, where there is nothing left to admit. That tail is the ONLY idle time a
    continuous batcher has -- which is why it is so much better than a static batch.
    """
    if slots <= 0:
        raise ValueError("slots must be > 0")
    total = sum(lengths)
    # Simulate slot occupancy: greedy, longest-processing-time-first is optimal for makespan.
    ends = [0] * slots
    for L in sorted(lengths, reverse=True):
        i = ends.index(min(ends))
        ends[i] += L
    makespan = max(ends)
    return {"n": len(lengths), "slots": slots, "makespan": makespan,
            "utilisation": total / (makespan * slots) if makespan else 0.0}


# --------------------------------------------------------------------------------------
# The continuous-batching loop
# --------------------------------------------------------------------------------------

@dataclass
class SchedulerState:
    slots: int
    running: list[Seq] = field(default_factory=list)
    waiting: list[Seq] = field(default_factory=list)
    done: list[Seq] = field(default_factory=list)
    step: int = 0
    idle_slots_total: int = 0
    admits: int = 0
    preemptions: int = 0
    full: int = 0                # max occupancy this run can reach, = min(slots, n_seqs)
    steady_steps: int = 0        # steps at which the batch was at that occupancy
    steady_tokens: int = 0       # decode tokens produced during those steps

    @property
    def free(self) -> int:
        return self.slots - len(self.running)

    def idle_fraction(self) -> float:
        total = self.step * self.slots
        return self.idle_slots_total / total if total else 0.0

    def saturated_throughput(self) -> float:
        """Decode tokens per step, counted ONLY while the batch was at full occupancy.

        The whole-run average is a different and much less useful number: it includes the
        prefill-only steps, and a deeper batch spends proportionally more steps prefilling. That
        makes the run average FALL as the batch grows, which looks like a throughput regression and
        is really a measurement artefact. The saturated rate is the one that answers "can this
        engine go faster", and the answer is that it cannot -- it sits on the ceiling.
        """
        return self.steady_tokens / self.steady_steps if self.steady_steps else 0.0


def run_continuous(seqs: list[Seq], slots: int, prefill_rate: int = 8000,
                   tps_per_slot: int = 100, step_token_budget: int = 800,
                   max_steps: int = 100000) -> SchedulerState:
    """Iteration-level admission.

    Each step: (1) admit any waiting sequence into a free slot, (2) prefill what is prefilling and
    decode what is decoding, (3) retire the finished and free their slots. Step 3 happening BEFORE
    step 1 of the next iteration is the whole trick -- a slot is never idle while work is queued.

    TWO RESOURCES, BECAUSE THE ROOFLINE HAS TWO REGIMES (see T06):

      prefill_rate       -- compute-bound. Tokens prefilled per step, shared among all sequences
                            currently prefilling. Cheap per token, but a burst of new prompts
                            competes for it.
      step_token_budget  -- bandwidth-bound. Decode tokens per step, shared among all running
                            sequences. This is the roofline ceiling.

    `step_token_budget` is what makes over-batching visible. Without it every sequence decodes at
    full speed no matter how deep the batch is -- i.e. the model assumes infinite memory bandwidth
    -- and adding slots would only ever reduce latency, so goodput could never peak and fall.

    Sequences are admitted only once `step >= arrival`, so `arrival` and `finish` are both cycle
    numbers and their difference is a real latency.

    A prefilling sequence occupies its slot without decoding, and the batch does NOT stall behind
    it -- this is the CHUNKED case. The unchunked case, where a long prefill blocks the whole batch
    for its full duration, is the head-of-line blocking that `policies.chunking_effect` prices.
    """
    if slots <= 0:
        raise ValueError("slots must be > 0")
    if prefill_rate <= 0 or step_token_budget <= 0:
        raise ValueError("prefill_rate and step_token_budget must be > 0")
    st = SchedulerState(slots=slots, full=min(slots, len(seqs)))
    st.waiting = sorted(seqs, key=lambda s: (s.arrival, s.seq_id))

    while (st.waiting or st.running) and st.step < max_steps:
        st.step += 1
        # (1) admit -- iteration level, so this happens EVERY step, not once per batch
        while st.waiting and st.free > 0:
            if st.waiting[0].arrival > st.step:
                break                       # not arrived yet; the queue is arrival-ordered
            s = st.waiting.pop(0)
            s.start = st.step
            s.prefill_left = s.prompt_tokens
            st.running.append(s)
            st.admits += 1
        st.idle_slots_total += st.free

        # (2a) prefill -- compute-bound, shared
        prefilling = [s for s in st.running if s.prefill_left > 0]
        if prefilling:
            share = max(1, prefill_rate // len(prefilling))
            for s in prefilling:
                s.prefill_left = max(0, s.prefill_left - share)
                if s.prefill_left == 0:
                    s.prefilled = True

        # (2b) decode -- bandwidth-bound, shared, and this is where over-batching bites
        decoding = [s for s in st.running if s.prefill_left == 0]
        if decoding:
            want = tps_per_slot * len(decoding)
            actual = min(want, step_token_budget)
            per, extra = divmod(actual, len(decoding))
            # Distribute the remainder instead of dropping it. With integer division alone a
            # 64-slot batch reports 768 tok/step instead of the 800 ceiling, purely because
            # 800 // 64 leaves 32 tokens unassigned -- which reads as a throughput plateau below
            # the roofline and is really an arithmetic leak.
            for i, s in enumerate(decoding):
                s.decoded += per + (1 if i < extra else 0)
            if len(st.running) == st.full:
                st.steady_steps += 1
                st.steady_tokens += actual

        # (3) retire
        still = []
        for s in st.running:
            if s.prefill_left == 0 and s.done:
                s.finish = st.step
                st.done.append(s)
            else:
                still.append(s)
        st.running = still
    return st


def static_schedule(seqs: list[Seq], slots: int, tps_per_slot: int = 100) -> SchedulerState:
    """Static batching: groups of `slots` sequences admitted TOGETHER and retired TOGETHER.

    The batch's wall-clock is its LONGEST member, and every other member is carried until then.
    A sequence that arrives after a batch has started cannot join it -- it waits for the next
    batch to form, which is the head-of-line blocking that continuous batching removes.

    This exists so the two admission disciplines can be compared on IDENTICAL traffic. Comparing
    continuous batching against a hand-computed ideal proves nothing; comparing it against the
    static scheduler you would actually have replaced is the argument.
    """
    st = SchedulerState(slots=slots, full=min(slots, len(seqs)))
    ordered = sorted(seqs, key=lambda s: (s.arrival, s.seq_id))
    step = 0
    while ordered:
        step += 1                                     # the batch forms at this step
        batch = ordered[:slots]
        del ordered[:slots]
        span = max(1, math.ceil(max(s.total_tokens for s in batch) / tps_per_slot))
        step += span - 1                              # ...and runs to its longest member
        for s in batch:
            s.start = step - span + 1
            s.finish = step
            s.decoded = s.output_tokens
            st.done.append(s)
    st.step = step
    return st


def throughput(st: SchedulerState, tps_per_slot: int = 100) -> float:
    return sum(s.decoded for s in st.done) / st.step if st.step else 0.0


def completion_stats(st: SchedulerState) -> dict:
    if not st.done:
        return {"n": 0}
    lat = sorted((s.finish or 0) - s.arrival for s in st.done)
    return {"n": len(lat), "p50": lat[len(lat) // 2], "p99": lat[-1],
            "mean": sum(lat) / len(lat), "makespan": st.step,
            "idle_fraction": st.idle_fraction()}
