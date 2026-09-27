"""T16 — agentic inference: what an agent task costs, and why it is not a chat request.

Run:  python run.py

Stdlib only, no network, no GPU. Everything below is arithmetic on MODEL PARAMETERS
declared in sim/agent_cost.py -- it is not a measurement of any real system, and no
vendor price or benchmark figure appears anywhere. Where a number comes from the corpus
it is marked [T] in the prose; where it is my own derivation it is marked [D].
"""

from __future__ import annotations

from sim import agent_cost as ac

RULE = "=" * 74


def section(n: int, title: str) -> None:
    print()
    print(RULE)
    print(f"  {n}. {title}")
    print(RULE)


def main() -> None:
    shape = ac.TaskShape()
    ep = ac.EngineParams()

    print(RULE)
    print("  T16 - AGENTIC INFERENCE: the anatomy of an agent task")
    print(RULE)
    print(f"  A {shape.turns}-turn agent task. System prompt {shape.system_tokens:,} tok, "
          f"tool schemas {shape.tool_schema_tokens:,} tok,")
    print(f"  {shape.output_tokens} tok out and {shape.tool_result_tokens} tok back per turn, "
          f"{shape.tool_latency_s}s of tool latency per turn.")
    print()
    print("  This models a MECHANISM, not a machine. The rates are declared parameters,")
    print("  and every conclusion below is a direction plus an order of magnitude -- not")
    print("  a predicted latency for any particular deployment.")

    # ---------------------------------------------------------------- 1
    section(1, "The wall clock and the bill are dominated by DIFFERENT terms")
    for reuse in (0.0, 0.9):
        b = ac.breakdown(shape, reuse, ep)
        print()
        print(f"  reuse = {reuse:.0%} of the existing prefix cache-resident")
        print(ac.render_breakdown(b))
    b0 = ac.breakdown(shape, 0.0, ep)
    b9 = ac.breakdown(shape, 0.9, ep)
    print()
    print("  Both columns use the same rates and the wall-clock column sums the turns")
    print("  serially, so it treats the fleet as if it were serving this one task. Read it")
    print("  as a SHAPE, not a schedule. Within that shape:")
    print()
    print(f"  Tool execution is a CONSTANT {b0['wall_tool_s']:,.0f} s of user-visible time and")
    print("  exactly 0 s of fleet cost -- no cache, no scheduler and no parallelism")
    print("  technique in T11-T15 touches it. That constant is what everything else is")
    print("  measured against, and it is why the two dashboards diverge:")
    print()
    print(f"    {'':<22}{'cold (reuse 0%)':>17}{'warm (reuse 90%)':>19}")
    print(f"    {'tool, % of wall clock':<22}{b0['wall_tool_share']:>16.0%}"
          f"{b9['wall_tool_share']:>19.0%}")
    print(f"    {'prefill, % of wall clock':<22}{b0['wall_prefill_share']:>16.0%}"
          f"{b9['wall_prefill_share']:>19.0%}")
    print(f"    {'prefill, % of fleet bill':<22}{b0['gpu_prefill_share']:>16.0%}"
          f"{b9['gpu_prefill_share']:>19.0%}")
    print()
    print("  So the answer to 'what dominates an agent task' is not one thing -- it")
    print("  moves with the cache:")
    print(f"    at COLD reuse the token work dominates BOTH (prefill {b0['wall_prefill_share']:.0%} of")
    print(f"      the wait, {b0['gpu_prefill_share']:.0%} of the bill), so latency work and cost work are")
    print("      the same work;")
    print(f"    at WARM reuse the tool constant dominates the wait ({b9['wall_tool_share']:.0%}) while")
    print(f"      prefill still dominates the bill ({b9['gpu_prefill_share']:.0%}), so further latency")
    print("      work is invisible to the user and further cache work is invisible to")
    print("      the latency dashboard.")
    print()
    print(f"  Raising reuse 0% -> 90% cuts fleet cost {b0['gpu_total_s']:,.0f} -> "
          f"{b9['gpu_total_s']:,.0f} GPU-s "
          f"({(1 - b9['gpu_total_s'] / b0['gpu_total_s']):.0%})")
    print(f"  and the user's wait {b0['wall_total_s']:,.0f} -> {b9['wall_total_s']:,.0f} s "
          f"({(1 - b9['wall_total_s'] / b0['wall_total_s']):.0%}). Both improve -- but they improve")
    print("  by different amounts, and past the crossover only one of them keeps moving. [D]")

    # ---------------------------------------------------------------- 2
    section(2, "Context growth: O(N^2) without prefix reuse, O(N) with it")
    print(ac.render_growth(shape, ep))
    print()
    s40 = ac.TaskShape(turns=40)
    a40 = ac.prefill_tokens(s40, 0.0)
    b40 = ac.prefill_tokens_ideal(s40)
    print(f"  At 40 turns the difference is {a40 / b40:.0f}x the prompt tokens. It is not a")
    print("  constant factor: it GROWS with task length, which is exactly the regime the")
    print("  corpus is pushing agents into -- long-horizon work measured in hours. [D]")
    print()
    print("  The reason the ratio is not 40x is the delta term in new_tokens_at_turn():")
    print(f"  even at perfect reuse each turn must prefill its own {shape.output_tokens} output +")
    print(f"  {shape.tool_result_tokens} tool-result tokens, which no cache can hold because they did")
    print("  not exist when the cache was written. That is why a long agent is never free")
    print("  even on a perfect cache, and why 'just cache the prefix' is necessary but")
    print("  not sufficient.")

    # ---------------------------------------------------------------- 3
    section(3, "The tool-schema tax: a constant term that scales with TURNS, not work")
    no_schema = ac.TaskShape(tool_schema_tokens=0)
    no_sys = ac.TaskShape(system_tokens=0, tool_schema_tokens=0)
    full = ac.prefill_tokens(shape, 0.0)
    without_schema = ac.prefill_tokens(no_schema, 0.0)
    without_either = ac.prefill_tokens(no_sys, 0.0)
    print(f"  prefill tokens, 40 turns, no reuse")
    print(f"    as configured (system {shape.system_tokens:,} + tools {shape.tool_schema_tokens:,})"
          f"   {full:>12,.0f}")
    print(f"    without the tool schemas                        {without_schema:>12,.0f}")
    print(f"    without either                                  {without_either:>12,.0f}")
    print()
    print(f"  The tool schemas are {(full - without_schema) / full:.1%} of all prompt tokens "
          f"processed for this task.")
    print("  That is not a one-off: they are re-sent on every turn, so their share is a")
    print("  function of how many turns the agent takes, not of how much work it does.")
    print()
    print("  This is the number behind the panel's complaint of 'massive tool volume and")
    print("  token prefill' [T]. The serving consequence is a design rule [D]:")
    print("    - tool schemas belong in the CACHED prefix, never in the per-turn text;")
    print("    - a large tool catalogue is a fixed per-turn tax on EVERY request, so tool")
    print("      selection (retrieve the relevant schemas, don't ship all of them) is a")
    print("      cache-and-cost optimisation, not just a prompt-quality one;")
    print("    - the tax is invisible in TTFT at low load and becomes the whole bill at")
    print("      high load, because it is exactly the term batching cannot amortise away.")
    print()
    print("  Counterfactual: the same 40 turns with the schema block present costs")
    print(f"  {ac.gpu_seconds(shape, 0.0, ep):,.0f} GPU-s; without it "
          f"{ac.gpu_seconds(no_schema, 0.0, ep):,.0f} GPU-s. [D]")

    # ---------------------------------------------------------------- 4
    section(4, "What the cache is worth, as a function of how resident it is")
    print(f"  {'reuse':>7} {'prompt tok':>12} {'GPU-s':>9} {'vs ideal':>10} "
          f"{'wall s':>8} {'tool share':>11}")
    print("  " + "-" * 61)
    ideal = ac.gpu_seconds(shape, 1.0, ep)
    for r in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        b = ac.breakdown(shape, r, ep)
        print(f"  {r:>7.0%} {ac.prefill_tokens(shape, r):>12,.0f} "
              f"{b['gpu_total_s']:>9,.1f} {b['gpu_total_s'] / ideal:>9.2f}x "
              f"{b['wall_total_s']:>8,.1f} {b['wall_tool_share']:>10.1%}")
    print()
    print("  In THIS model the saving is exactly LINEAR in reuse -- check the prompt-token")
    print("  column: every 25 points of residency removes the same ~300k tokens. That is a")
    print("  property of the arithmetic (a cached token avoids the same recompute whether")
    print("  it sits early or late), and it is why no 'diminishing returns' story is told")
    print("  here: on this mechanism there are none, and inventing one would be a")
    print("  fabricated finding.")
    print()
    print("  What is NOT linear is the capability. Prefix caching is close to a discrete")
    print("  step -- you either operate a cache and a prefix-aware router or you do not --")
    print("  so the step from 0% to a working 40% is qualitatively different from 90% to")
    print("  100%, and the last points are where eviction policy (T12) and placement")
    print("  precision (T14) earn their keep. The linearity above tells you the SIZE of")
    print("  the prize; it does not tell you where the engineering is hard.")
    print()
    print("  This is the same mechanism T12 calls KV offload and T13 makes addressable")
    print("  across nodes. T16's contribution is that for agents it is not an")
    print("  optimisation on top of the design -- it is the difference between a task")
    print(f"  that costs {ac.gpu_seconds(shape, 0.0, ep):,.0f} GPU-s and one that costs "
          f"{ideal:,.0f}.")

    # ---------------------------------------------------------------- 5
    section(5, "Fan-out: cost scales with N, wall clock does not")
    counts = (8, 64, 512, 4_096, 100_000)
    for reuse in (0.0, 0.9):
        print()
        print(f"  reuse = {reuse:.0%}, sub-agents run {6} turns each")
        rows = [ac.fanout(shape, n, shared_prefix=sp, reuse=reuse, ep=ep)
                for sp in (False, True) for n in counts]
        print(ac.render_fanout(rows))
    print()
    print("  Read the two right-hand columns together. At 100,000 sub-agents -- the")
    print("  corpus's own ad-hoc figure [T] -- the wall clock is barely above one parent's")
    print("  (1.09x, and nearly all of that is one sub-agent's own tool latency), while the")
    print("  fleet does 6,106 parents' worth of work with an unshared cold cache and")
    print("  12,615 with an unshared warm one.")
    print("  Fan-out is not parallelism; it is CONCURRENCY plus a cost multiplier, and the")
    print("  multiplier is the part that appears in the cluster and not in the trace. [D]")
    print()
    print("  Note also that warm reuse makes the MULTIPLIER worse, not better -- the parent")
    print("  gets cheap and the sub-agents do not get equally cheap, so the fan-out ratio")
    print("  rises from 6,106x to 12,615x. Anything that reduces the per-turn cost of the")
    print("  parent alone will flatter the parent and magnify the fan-out.")
    print()
    print("  Now the design question: should a sub-agent inherit the parent's context?")
    print("  The answer is not 'yes' -- it depends on how long the sub-agent lives, and")
    print("  on whether the parent's prefix is resident at all.")
    print()
    print("  Per-sub-agent prefill, shared vs unshared, by sub-agent length:")
    print(f"    {'sub-agent turns':>16} {'shared, cold':>13} {'unshared, cold':>15} "
          f"{'shared, 90% warm':>18} {'unshared, 90% warm':>19}")
    print("    " + "-" * 83)
    for k in (2, 3, 4, 6, 10, 20):
        s_c = ac.fanout(shape, 1, shared_prefix=True, reuse=0.0, ep=ep, subagent_turns=k)
        u_c = ac.fanout(shape, 1, shared_prefix=False, reuse=0.0, ep=ep, subagent_turns=k)
        s_w = ac.fanout(shape, 1, shared_prefix=True, reuse=0.9, ep=ep, subagent_turns=k)
        u_w = ac.fanout(shape, 1, shared_prefix=False, reuse=0.9, ep=ep, subagent_turns=k)
        print(f"    {k:>16} {s_c['per_subagent_gpu_s']:>13,.1f} "
              f"{u_c['per_subagent_gpu_s']:>15,.1f} {s_w['per_subagent_gpu_s']:>18,.1f} "
              f"{u_w['per_subagent_gpu_s']:>19,.1f}")
    print()
    print("  A SHORT sub-agent should NOT inherit the parent's context. Handing a child the")
    print("  parent's accumulated history costs the parent's whole context to prefill, and")
    print("  a two-turn child would have cost almost nothing on its own. The crossover on a")
    print("  cold cache is at about 5.6 sub-agent turns [D] -- an algebraic identity, not a")
    print("  measurement: the parent's context grows linearly with the parent's turns while")
    print("  an unshared child's quadratic growth is still small early on.")
    print()
    print("  With a WARM cache the ordering collapses at every length: the parent's context")
    print("  is the only thing that is cached, so sharing it is cheap from the first turn.")
    print("  [D]")
    print()
    print("  So 'sub-agents should share the parent's context' is a conditional, and the")
    print("  condition is a serving-system property -- cache residency (T12/T13) and")
    print("  prefix-aware placement (T14) -- plus the sub-agent's expected lifetime. The")
    print("  corpus identifies scoped shared memory between sub-agents as an open standards")
    print("  problem [T]; this table is why it is also a capacity problem.")

    # ---------------------------------------------------------------- 6
    section(6, "The long-horizon wall: why harness capability gets internalised")
    print("  Tworek's argument [T], as arithmetic [D]: a long trajectory can only be")
    print("  scored when it ENDS, so gradient steps per day is bounded by completions")
    print("  per day, and that is a hyperbola with no free parameter.")
    print()
    hours = (0.5, 2.0, 4.0, 8.0, 12.0, 24.0)
    print(ac.render_learning_signal([ac.learning_signal(h) for h in hours]))
    print()
    print("  The corpus's instance is the 12-hour trajectory at about 2 steps/day, ~14")
    print("  per week, ~60 per month [T]. The identity reproduces it exactly, which is")
    print("  the whole point: this is a scheduling consequence, not a measurement.")
    print()
    print("  Put that beside section 2. As trajectory length grows, the tokens needed")
    print("  grow with the SQUARE of the length (or linearly at best) while the learning")
    print("  signal per token falls. Both curves point the same way [D]:")
    print()
    print("    - a harness that glues short-horizon work into long-horizon work buys")
    print("      capability at a superlinear token cost and a sublinear signal;")
    print("    - moving a capability INTO the model (internalisation) removes it from")
    print("      the per-turn prompt, which removes it from the cached prefix AND from")
    print("      the learning-signal decay;")
    print("    - so internalisation is a serving optimisation and a training")
    print("      optimisation at the same time, which is why the corpus treats the")
    print("      harness and the model as one moving boundary rather than two layers.")
    print()
    print("  The harness does not disappear. The corpus's own framing is that the")
    print("  harness is where long-horizon behaviour is COMPOSED -- plan mode, goals,")
    print("  sub-agents [T]. What changes is which capabilities are cheap enough to")
    print("  leave in the prompt.")

    print()
    print(RULE)
    print("  What this proves: an agent task is superlinear in turns, its bill and its")
    print("  latency are dominated by different terms, and fan-out multiplies the")
    print("  expensive one. None of that is visible in a chat-shaped dashboard.")
    print(RULE)
    print()


if __name__ == "__main__":
    main()
