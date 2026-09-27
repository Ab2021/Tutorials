"""Iteration-level (continuous) batching versus static batching.

The mechanism: a static batch is a set of sequences admitted together and retired
together, so the batch's slot count is held for as long as the LONGEST sequence in it
runs. Continuous batching makes the admission decision every iteration: a finished
sequence leaves and a waiting one takes its slot.

The metric that separates them is slot utilisation -- useful sequence-steps divided by
slot-steps paid for. Nothing here is measured; it is a model of the scheduling policy. [D]
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Seq:
    sid: int
    prompt_tokens: int
    output_tokens: int


@dataclass
class ScheduleResult:
    name: str
    slot_steps: int        # batch slots x steps, i.e. what you paid for
    useful_steps: int      # steps in which a slot was doing real work
    makespan: int          # steps from start to last completion

    @property
    def slot_utilisation(self) -> float:
        return self.useful_steps / self.slot_steps if self.slot_steps else 0.0


def make_sequences(n: int, min_out: int = 16, max_out: int = 512, seed: int = 5) -> list[Seq]:
    """Output lengths are heavy-tailed -- most answers are short, a few are very long.
    That spread is exactly what static batching pays for and continuous batching does
    not. [D]"""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # log-uniform between min_out and max_out: many short, a few long
        import math
        lo, hi = math.log(min_out), math.log(max_out)
        o = int(math.exp(rng.uniform(lo, hi)))
        out.append(Seq(i, prompt_tokens=rng.randint(200, 4000), output_tokens=o))
    return out


def static_batching(seqs: list[Seq], batch_size: int) -> ScheduleResult:
    """Admit B at a time; the batch retires when its longest member finishes."""
    slot_steps = 0
    useful = 0
    step = 0
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        longest = max(s.output_tokens for s in batch)
        slot_steps += len(batch) * longest
        useful += sum(s.output_tokens for s in batch)
        step += longest
    return ScheduleResult("static", slot_steps, useful, step)


def continuous_batching(seqs: list[Seq], batch_size: int) -> ScheduleResult:
    """Every step, retire finished sequences and fill their slots.

    A slot that becomes free is refilled on the very next iteration rather than held to
    the end of a batch. That is the whole change. [T]
    """
    pending = list(seqs)
    running: list[int] = []      # remaining output tokens per running sequence
    slot_steps = 0
    useful = 0
    step = 0
    while pending or running:
        # retire
        running = [r for r in running if r > 0]
        # fill
        while len(running) < batch_size and pending:
            running.append(pending.pop(0).output_tokens)
        if not running:
            break
        slot_steps += batch_size          # the batch slot is paid for whether used or not
        useful += len(running)
        running = [r - 1 for r in running]
        step += 1
    return ScheduleResult("continuous", slot_steps, useful, step)


def render(results: list[ScheduleResult]) -> str:
    lines = [f"  {'policy':<12} {'slot-steps':>12} {'useful':>10} {'utilisation':>12} "
             f"{'makespan':>10}"]
    lines.append("  " + "-" * 60)
    for r in results:
        lines.append(f"  {r.name:<12} {r.slot_steps:>12,} {r.useful_steps:>10,} "
                     f"{r.slot_utilisation:>12.1%} {r.makespan:>10,}")
    return "\n".join(lines)
