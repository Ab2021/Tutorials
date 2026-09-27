"""T16 — the anatomy of an agent task, in tokens and seconds.

The mechanism this module exists to demonstrate: **the agent loop does not look like a
chat request, so the things you would optimise for chat are the wrong things here.**

Three consequences, all derived rather than measured:

  1. WALL CLOCK and COST are dominated by different terms. Tool execution dominates the
     wall clock; token processing dominates the bill. Optimising TTFT makes an agent feel
     faster without making it cheaper, and capping spend slows it down in a way the
     latency dashboard cannot show. [D]

  2. CONTEXT GROWTH is the whole ballgame. An agent that re-prefills its growing
     conversation every turn processes O(N^2) prompt tokens across N turns; one that
     reuses the prefix processes O(N). The mechanism is the same one T12 models for
     sessions and T13 models for the KV connector -- here it is the dominant cost term
     rather than an optimisation. [D]

  3. FAN-OUT multiplies both. A sub-agent is a NEW session, so unless it inherits the
     parent's prefix AND the router lands it on a replica holding that prefix, it does
     not share the parent's cache. The corpus puts the ad-hoc scale at up to 100,000
     sub-agents [T]; every one of them re-prefills whatever context it was given.

Everything is in TOKENS and GPU-SECONDS. No dollar figure appears anywhere in this
module, because no vendor price is asserted by the corpus and inventing one would be a
fabricated benchmark. A reader who wants money can multiply by their own rate. [D]
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TaskShape:
    """One agent task. Defaults are a coding agent doing ~40 turns of work."""

    turns: int = 40
    system_tokens: int = 2_000        # system prompt
    tool_schema_tokens: int = 6_000   # tool definitions -- re-sent EVERY turn
    output_tokens: int = 300          # what the model writes per turn
    tool_result_tokens: int = 900     # what the tool returns per turn
    tool_latency_s: float = 2.5       # wall clock for the tool call itself
    tool_calls_per_turn: int = 1

    def context_at_turn(self, i: int) -> int:
        """Prompt size at the START of turn i (0-indexed).

        The tool schemas and system prompt are re-sent every turn -- that is what makes
        the constant term large, and it is worth noticing on its own for a long task.
        """
        return (self.system_tokens + self.tool_schema_tokens
                + i * (self.output_tokens + self.tool_result_tokens))


@dataclass(frozen=True)
class EngineParams:
    """Server-side rates. Same shape as T12/T13's parameters, so the numbers are
    comparable across blueprints -- but these are MODEL PARAMETERS, not measurements."""

    prefill_tps: float = 8_000.0      # prompt tokens/s, batched aggregate
    decode_tps: float = 800.0         # output tokens/s, batched aggregate


# --------------------------------------------------------------------------- prefill


def new_tokens_at_turn(shape: TaskShape, i: int, reuse: float) -> float:
    """Prompt tokens the GPU must actually process at turn `i`.

    `reuse` is the fraction of the ALREADY-EXISTING prefix that is cache-resident when
    the turn starts -- the thing T12's KV offload, T13's connector and T14's prefix
    routing exist to raise.

    The turn's prompt splits into two parts that behave differently:

      * the PREFIX that existed at the end of the previous turn -- cacheable in full, so
        only `(1 - reuse)` of it is recomputed;
      * the DELTA appended since then (the model's own last output, plus the tool
        result). This is new text. **No cache can hold it**, at any reuse, because it
        did not exist when the cache was written.

    That second term is why the task is superlinear in its length even at perfect reuse,
    and why `reuse=1.0` still does not make a long agent free.

      reuse = 0.0  -> every turn re-prefills the whole conversation:   O(N^2)
      reuse = 1.0  -> every turn prefills only its own delta:          O(N)
    """
    ctx = shape.context_at_turn(i)
    if i == 0:
        return float(ctx)                       # cold start: nothing to reuse
    prev = shape.context_at_turn(i - 1)
    return (1.0 - reuse) * prev + (ctx - prev)


def prefill_tokens(shape: TaskShape, reuse: float = 0.0) -> float:
    """Prompt tokens the GPU processes across the whole task."""
    return sum(new_tokens_at_turn(shape, i, reuse) for i in range(shape.turns))


def prefill_tokens_ideal(shape: TaskShape) -> float:
    """Perfect reuse. Identical to `prefill_tokens(shape, 1.0)` by construction -- kept
    as a named function because it is the lower bound the ratio is quoted against."""
    return prefill_tokens(shape, 1.0)


def decode_tokens(shape: TaskShape) -> int:
    return shape.turns * shape.output_tokens


# --------------------------------------------------------------------------- the two


def gpu_seconds(shape: TaskShape, reuse: float = 0.0,
                ep: EngineParams | None = None) -> float:
    """What the task costs the FLEET. This is the number that sizes capacity."""
    ep = ep or EngineParams()
    return prefill_tokens(shape, reuse) / ep.prefill_tps + decode_tokens(shape) / ep.decode_tps


def wall_clock_s(shape: TaskShape, reuse: float = 0.0,
                 ep: EngineParams | None = None) -> float:
    """What the USER waits for. Tool execution is in here; fleet cost is not.

    Note the asymmetry the whole blueprint turns on: `reuse` moves this number by a
    little and `gpu_seconds` by a lot, because tool latency is a large constant that no
    cache touches and prefill is a small share of the wall clock. A team optimising the
    user-visible number will not notice the cache, and a team optimising the bill will
    not notice the tool.
    """
    ep = ep or EngineParams()
    total = 0.0
    for i in range(shape.turns):
        total += new_tokens_at_turn(shape, i, reuse) / ep.prefill_tps
        total += shape.output_tokens / ep.decode_tps
        total += shape.tool_calls_per_turn * shape.tool_latency_s
    return total


def breakdown(shape: TaskShape, reuse: float = 0.0,
              ep: EngineParams | None = None) -> dict:
    """Where the wall clock goes, and where the GPU time goes. They do not agree."""
    ep = ep or EngineParams()
    tool_s = shape.turns * shape.tool_calls_per_turn * shape.tool_latency_s
    decode_s = decode_tokens(shape) / ep.decode_tps
    total_s = wall_clock_s(shape, reuse, ep)
    prefill_s = total_s - tool_s - decode_s
    g_total = gpu_seconds(shape, reuse, ep)
    g_prefill = prefill_tokens(shape, reuse) / ep.prefill_tps
    return {
        "reuse": reuse,
        "wall_total_s": total_s,
        "wall_tool_s": tool_s,
        "wall_decode_s": decode_s,
        "wall_prefill_s": prefill_s,
        "wall_tool_share": tool_s / total_s if total_s else 0.0,
        "wall_prefill_share": prefill_s / total_s if total_s else 0.0,
        "gpu_total_s": g_total,
        "gpu_prefill_s": g_prefill,
        "gpu_prefill_share": g_prefill / g_total if g_total else 0.0,
    }


# --------------------------------------------------------------------------- fan-out


def subagent_shape(shape: TaskShape, subagent_turns: int) -> TaskShape:
    """A sub-agent given a FRESH, independent prompt -- the framework default.

    It re-sends the same system prompt and the same tool schemas (that is the constant
    term the corpus's "massive tool volume and token prefill" points at [T]), and it
    builds its own history from nothing.
    """
    return replace(shape, turns=subagent_turns)


def fanout(shape: TaskShape, n_subagents: int, *, shared_prefix: bool,
           reuse: float = 0.0, ep: EngineParams | None = None,
           subagent_turns: int = 6) -> dict:
    """A parent agent spawning N sub-agents.

    The corpus's framing: sub-agents form a hierarchy, memory is shared between them in
    a scoped way, and the scoping is an open standards problem [T]. The serving
    consequence is sharp and the two cases are NOT ordered:

      shared_prefix=False  an independent prompt per sub-agent. The sub pays its own
                           cold start (system + tool schemas) and re-prefills its own
                           history every turn. This is what most frameworks do, and the
                           parent's warm cache buys nothing.
      shared_prefix=True   the sub-agent's prompt BEGINS WITH the parent's context, so
                           the parent's cached prefix is reusable -- but ONLY to the
                           extent it is actually resident (reuse) AND the router places
                           the sub-agent on a replica holding it (T14). Inheriting a cold
                           parent context is not free: you are now prefilling the
                           parent's whole history to give the child its bearings.

    So which is cheaper depends on `reuse`, and the crossover is the finding.
    """
    ep = ep or EngineParams()
    parent = gpu_seconds(shape, reuse, ep)
    parent_wall = wall_clock_s(shape, reuse, ep)

    if shared_prefix:
        # Turn 0 of the sub is the parent's final context. Later turns pay only deltas.
        parent_ctx = shape.context_at_turn(shape.turns - 1)
        sub_prefill = parent_ctx * (1.0 - reuse) + (subagent_turns - 1) * (
            shape.output_tokens + shape.tool_result_tokens)
    else:
        sub = subagent_shape(shape, subagent_turns)
        sub_prefill = prefill_tokens(sub, reuse)

    sub_decode = subagent_turns * shape.output_tokens
    per_sub = sub_prefill / ep.prefill_tps + sub_decode / ep.decode_tps
    sub_wall = (sub_prefill / ep.prefill_tps
                + sub_decode / ep.decode_tps
                + subagent_turns * shape.tool_calls_per_turn * shape.tool_latency_s)

    return {
        "shared_prefix": shared_prefix,
        "n_subagents": n_subagents,
        "parent_gpu_s": parent,
        "per_subagent_gpu_s": per_sub,
        "subagents_gpu_s": per_sub * n_subagents,
        "total_gpu_s": parent + per_sub * n_subagents,
        "multiple": (parent + per_sub * n_subagents) / parent if parent else 0.0,
        # Wall clock: sub-agents run CONCURRENTLY, so the parent's wall clock grows by
        # ONE sub-agent's, not N of them. Cost grows by N. That asymmetry is the whole
        # reason fan-out is attractive and dangerous at the same time -- it looks like
        # latency for free, and it is billed per unit of concurrency. [D]
        "wall_s": parent_wall + sub_wall,
        "parent_wall_s": parent_wall,
    }


# --------------------------------------------------------------------------- horizons


def learning_signal(trajectory_hours: float, days: int = 30) -> dict:
    """Tworek's argument, as arithmetic [T] claim, [D] derivation.

    The claim: a long agent trajectory can only be scored when it ENDS, so the number of
    gradient steps a day is bounded by how many trajectories complete in a day. The
    corpus's instance is a 12-hour trajectory yielding ~2 gradient steps/day, ~14/week,
    ~60/month.

    The generalisation is a hyperbola with no free parameter: steps/day = 24 / hours.
    This is a scheduling identity, not a measurement. What follows from it -- that
    information per token falls as roughly 1/n while cost rises as n -- is the argument
    for distilling harness capabilities into the model rather than lengthening the
    harness. [D]
    """
    per_day = 24.0 / trajectory_hours
    return {
        "trajectory_hours": trajectory_hours,
        "steps_per_day": per_day,
        "steps_per_week": per_day * 7,
        "steps_per_month": per_day * days,
    }


# --------------------------------------------------------------------------- render


def render_breakdown(b: dict) -> str:
    lines = [f"  {'where the WALL CLOCK goes':<28} {'where the GPU TIME goes':<28}",
             "  " + "-" * 33 + "  " + "-" * 33,
             f"  tool execution  {b['wall_tool_s']:>9,.0f} s "
             f"{b['wall_tool_share']:>7.1%}    prefill      {b['gpu_prefill_s']:>9,.0f} s "
             f"{b['gpu_prefill_share']:>7.1%}",
             f"  model decode    {b['wall_decode_s']:>9,.0f} s "
             f"{b['wall_decode_s'] / b['wall_total_s']:>7.1%}    "
             f"decode       {b['gpu_total_s'] - b['gpu_prefill_s']:>9,.0f} s "
             f"{1 - b['gpu_prefill_share']:>7.1%}",
             f"  prefill         {b['wall_prefill_s']:>9,.0f} s "
             f"{b['wall_prefill_share']:>7.1%}",
             f"  {'TOTAL':<15} {b['wall_total_s']:>9,.0f} s        "
             f"{'TOTAL':<12} {b['gpu_total_s']:>9,.0f} s"]
    return "\n".join(lines)


def render_growth(shape: TaskShape, ep: EngineParams) -> str:
    lines = [f"  {'turns':>6} {'no-reuse tok':>13} {'ideal tok':>11} {'ratio':>7} "
             f"{'no-reuse GPU-s':>15} {'ideal GPU-s':>12}"]
    lines.append("  " + "-" * 70)
    for n in (5, 10, 20, 40, 80, 160):
        s = replace(shape, turns=n)
        a = prefill_tokens(s, 0.0)
        b = prefill_tokens_ideal(s)
        ga = gpu_seconds(s, 0.0, ep)
        gb = gpu_seconds(s, 1.0, ep)
        lines.append(f"  {n:>6} {a:>13,.0f} {b:>11,.0f} {a / b:>6.1f}x "
                     f"{ga:>15,.1f} {gb:>12,.1f}")
    return "\n".join(lines)


def render_fanout(rows: list[dict]) -> str:
    lines = [f"  {'sub-agents':>11} {'shared':>7} {'total GPU-s':>12} {'x parent':>9} "
             f"{'wall s':>8} {'wall x parent':>14}"]
    lines.append("  " + "-" * 66)
    for r in rows:
        lines.append(f"  {r['n_subagents']:>11,} "
                     f"{'yes' if r['shared_prefix'] else 'no':>7} "
                     f"{r['total_gpu_s']:>12,.0f} {r['multiple']:>8.1f}x "
                     f"{r['wall_s']:>8,.0f} "
                     f"{r['wall_s'] / r['parent_wall_s']:>13.2f}x")
    return "\n".join(lines)


def render_learning_signal(rows: list[dict]) -> str:
    lines = [f"  {'trajectory (h)':>15} {'steps/day':>10} {'steps/week':>11} "
             f"{'steps/month':>12}"]
    lines.append("  " + "-" * 52)
    for r in rows:
        lines.append(f"  {r['trajectory_hours']:>15,.1f} {r['steps_per_day']:>10,.2f} "
                     f"{r['steps_per_week']:>11,.1f} {r['steps_per_month']:>12,.1f}")
    return "\n".join(lines)
