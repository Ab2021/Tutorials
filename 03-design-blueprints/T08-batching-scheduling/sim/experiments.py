"""The seven experiments. Each is a scheduling DECISION, not a number.

Two of these (4 and 5) reproduce results the corpus reports, and one of them (5) does not come out
the way a reader might expect. Where the model and the corpus disagree in magnitude, the corpus
figure is printed beside the modelled one and the gap is stated rather than smoothed.
"""
from __future__ import annotations

import random

from .scheduler import (Seq, static_utilisation, continuous_utilisation, run_continuous,
                        static_schedule, completion_stats, throughput)
from .fairness import AgentProgram, dispatch, starve_report
from .policies import (prefill_chunk_size, chunk_plan, chunking_effect, saturation_gate,
                       goodput, batch_sweep)


def _skewed_lengths(rng, n=512, max_len=1024, shape=3.0):
    return [max(1, int(1 + (rng.random() ** shape) * (max_len - 1))) for _ in range(n)]


def exp_static_vs_continuous() -> None:
    print("\n1. STATIC vs CONTINUOUS BATCHING -- the 100 -> 42 step [T]")
    print("   corpus: 'turn on continuous batching and you are near 42' [T] LLMOps cost talk\n")

    rng = random.Random(11)
    print(f"   {'traffic':>14}  {'max len':>8}  {'mean len':>9}  {'static util':>12}  "
          f"{'continuous (8 slots)':>21}")
    print("   " + "-" * 72)
    for label, shape in (("uniform", 0.0), ("mild skew", 2.0), ("real skew", 3.0)):
        if shape == 0.0:
            lengths = [rng.randint(200, 1024) for _ in range(512)]
        else:
            lengths = _skewed_lengths(rng, shape=shape)
        st = static_utilisation(lengths)
        ct = continuous_utilisation(lengths, slots=8)
        print(f"   {label:>14}  {st['max_len']:>8,}  {st['mean_len']:>9.0f}  "
              f"{st['utilisation']:>11.1%}  {ct['utilisation']:>20.1%}")

    print("\n   READ: a static batch retires WHOLE, so its wall-clock is set by its LONGEST member")
    print("   while its useful work is the SUM. Every short sequence that finished early is a slot")
    print("   that idled until the long one caught up. Continuous batching refills the slot the")
    print("   instant ANY sequence finishes, so the only idle time is the end-of-queue tail.")
    print("   Note the shape: the penalty tracks the length DISTRIBUTION. Uniform traffic hides")
    print("   it; real (skewed) traffic exposes it. A benchmark with uniform prompt lengths will")
    print("   report a much smaller gain than production delivers -- which is why the corpus's")
    print("   100 -> 42 is measured on real traffic and not on a harness.")
    print("   WHY IT IS NOT A TUNABLE: this is an admission decision, not a parameter. There is")
    print("   no traffic shape for which waiting for the slowest sequence is the right answer.")


def exp_iteration_level_admission() -> None:
    print("\n2. ITERATION-LEVEL ADMISSION -- what 'continuous' actually means")
    print("   corpus: continuous batching 'keeps the [GPU busy]' by admitting per iteration [T]\n")

    def work():
        return [Seq(f"long{i}", 0, 900, arrival=0) for i in range(4)] + \
               [Seq(f"short{i}", 0, 60, arrival=3) for i in range(4)]

    SLOTS = 8
    print(f"   {SLOTS} slots. Four long sequences (900 output tokens each) start at step 1;")
    print(f"   four short ones (60 tokens) arrive at step 3, mid-flight. Same traffic, two")
    print(f"   admission disciplines -- static, and continuous.\n")
    print(f"   Slots are deliberately OVER-PROVISIONED: the four long sequences need only four of")
    print(f"   the eight, so there is genuine free capacity for the shorts. This is the honest")
    print(f"   setting for the comparison -- if the batch is already full, both disciplines are")
    print(f"   forced to make the shorts wait and the experiment measures nothing.\n")
    print(f"   {'discipline':>12}  {'makespan':>9}  {'short finishes':>15}  {'short waited':>13}  "
          f"{'idle':>7}")
    print("   " + "-" * 66)
    for label, runner in (("static", lambda: static_schedule(work(), SLOTS)),
                          ("continuous", lambda: run_continuous(work(), SLOTS))):
        st = runner()
        short_fin = sorted(s.finish for s in st.done if s.seq_id.startswith("short"))
        print(f"   {label:>12}  {st.step:>9}  {str(short_fin):>15}  "
              f"{min(short_fin) - 3:>13}  {st.idle_fraction():>6.1%}")

    print("\n   READ: under STATIC batching the four short sequences cannot join a batch already")
    print("   in flight, so they wait for the whole 900-token batch to retire and are then")
    print("   carried to the end of their own batch -- a 60-token job served as if it were 900.")
    print("   Four of the eight slots were never needed by the long sequences and sat idle for")
    print("   the entire batch: capacity exists, but the admission discipline cannot reach it.")
    print("   Under CONTINUOUS batching they are admitted at the next iteration and finish")
    print("   immediately, and their slots are freed for whatever arrives next.")
    print("   The makespan is UNCHANGED -- the same total decode work is done. THE ENTIRE WIN IS")
    print("   IN LATENCY AND SLOT REUSE, which is why a throughput-only dashboard shows almost")
    print("   nothing and teams conclude continuous batching 'didn't help'.")
    print("   Note also WHAT does not change: static batching is not slower in aggregate here,")
    print("   it is simply unable to use capacity it already has. That is the argument that wins")
    print("   the migration, and it is invisible on a makespan chart.")
    print("\n   WHERE THIS FAILS: admission is per-iteration, so a LONG PREFILL admitted")
    print("   mid-batch occupies its slot for the whole prefill in one step, stalling every")
    print("   decode in the batch. That is head-of-line blocking, and chunked prefill")
    print("   (experiment 3) is the fix.")
    print("   THAT FAILURE IS NOT VISIBLE ABOVE, and deliberately so: every sequence here has a")
    print("   ZERO-LENGTH PROMPT, which isolates the admission discipline and removes prefill")
    print("   from the measurement entirely. A reader who took the idle column as the whole")
    print("   story would conclude continuous batching is strictly better -- and then be")
    print("   surprised when a long prompt wrecks the ITL of everything behind it. Experiment 3")
    print("   puts prompts back in and prices exactly that.")


def exp_chunked_prefill() -> None:
    print("\n3. CHUNKED PREFILL -- trading a little TTFT for a lot of ITL stability")
    print("   formula: prefill_chunk_size ~= max_num_batched_tokens - (decode_seqs x 1)\n")

    for decode_seqs in (4, 16, 64):
        print(f"   with {decode_seqs:>3} sequences decoding: chunk size "
              f"{prefill_chunk_size(8192, decode_seqs):>5,} tokens")
    print("\n   and the trade, per prompt length (budget 8192, 16 decoding, 100 tok/s):")
    print(f"   {'prompt':>8}  {'chunk':>6}  {'chunks':>7}  {'unchunked ITL spike':>20}  "
          f"{'chunked spike':>14}  {'TTFT steps added':>17}")
    print("   " + "-" * 80)
    for row in chunking_effect([512, 2048, 8192, 32768], decode_seqs=16, budget=8192):
        print(f"   {row['prompt']:>8,}  {row['chunk']:>6,}  {row['chunks']:>7}  "
              f"{row['unchunked_itl_spike_steps']:>18} steps  "
              f"{row['chunked_itl_spike_steps']:>12} step  {row['ttft_steps_added']:>17}")

    print("\n   READ: unchunked, a 32k prompt is ONE step that takes ~328 decode-steps of time, so")
    print("   every other sequence in the batch waits that long -- one enormous ITL spike. Chunked,")
    print("   the spike is bounded at one chunk, at the cost of the prompt finishing n-1 steps")
    print("   later. Note the direction of the trade: TTFT gets slightly WORSE and ITL gets much")
    print("   BETTER. A team that measures only TTFT will conclude chunking hurt.")
    print("   EXCEPTION: if the traffic has no long prompts, chunking is pure overhead -- it adds")
    print("   scheduling steps and buys nothing. Enable it when the prompt-length distribution has")
    print("   a long tail, not by default.")


def exp_fcfs_vs_least_attained() -> None:
    print("\n4. FAIRNESS -- and the starvation direction that runs against intuition")
    print("   corpus: FCFS 'doesn't add anything extra to your policies because it's going to slow")
    print("   down everything' [T]; least-attained service cut request latencies 'up to 2x or")
    print("   sometimes 3x' [T]")
    print("   corpus scenario: 'a large session comes in and takes all the dispatch cycles, so the")
    print("   shorter sessions are starving' [T]\n")

    SLOTS = 8

    def make(fanout, big_first=True):
        """One large session that fans out, plus six short sessions arriving mid-flight."""
        big = AgentProgram("large-C", turns=60, tokens_per_turn=400, max_in_flight=fanout)
        smalls = [AgentProgram(f"small-{i}", turns=2, tokens_per_turn=100, start_cycle=5)
                  for i in range(6)]
        out = {"large-C": big} if big_first else {}
        for s in smalls:
            out[s.session_id] = s
        return out

    print(f"   {SLOTS} slots. large-C fans out to N concurrent turns of 400 tokens and refills")
    print(f"   them the instant any retire. Six 2-turn sessions arrive at cycle 5, behind it.")
    print(f"   The question is not whether they are served -- it is WHEN.\n")

    print(f"   {'C fan-out':>10}  {'policy':>15}  {'small mean lat':>14}  {'C mean lat':>11}  "
          f"{'C done @':>9}")
    print("   " + "-" * 66)
    ratios = []
    for fanout in (8, 12, 16, 24):
        row = {}
        for policy in ("fcfs", "least_attained"):
            progs = make(fanout)
            r = dispatch(progs, slots=SLOTS, cycles=400, policy=policy)
            s_stats = [r["latency_stats"][f"small-{i}"] for i in range(6)]
            s_mean = sum(s["mean"] for s in s_stats) / len(s_stats)
            row[policy] = s_mean
            print(f"   {fanout:>10}  {policy:>15}  {s_mean:>14.2f}  "
                  f"{r['latency_stats']['large-C']['mean']:>11.1f}  "
                  f"{r['finished_at']['large-C']:>9}")
        ratios.append((fanout, row["fcfs"] / row["least_attained"]))

    print("\n   small-session latency, FCFS vs least-attained, as the contention deepens:")
    for fanout, ratio in ratios:
        bar = "#" * int(round(ratio * 2))
        print(f"     fan-out {fanout:>3}:  {ratio:>5.2f}x better under least-attained  {bar}")
    print("   READ: least-attained holds the small sessions at a latency of 1 cycle regardless")
    print("   of how deep the large session's backlog gets, while FCFS's penalty grows with the")
    print("   fan-out -- 4.8x at fan-out 8, 12.8x at 24. That trend, not the constant, is the")
    print("   shape to remember: the more aggressively ONE program parallelises, the more a naive")
    print("   arrival-order policy punishes everyone else.")
    print("\n   AND LOOK AT THE LARGE SESSION IN THE TABLE ABOVE: its mean latency barely moves")
    print("   (4.2 -> 11.0) and it finishes at the same cycle either way. Serving the small")
    print("   sessions first does not cost the large one anything -- which is the corpus's own")
    print("   second-order claim, 'leaving the room for larger sessions to also finish much")
    print("   faster' [T], reproduced here as 'no worse'.")
    print("\n   HONEST NOTE ON MAGNITUDE: the corpus reports 'up to 2x or sometimes 3x' [T]. This")
    print("   model's contention is harsher and its ratios run higher -- it is NOT reproducing")
    print("   their number. Read the direction and the trend, not the constant. The model is also")
    print("   deliberately simple: FCFS here is arrival-order over a queue that one program")
    print("   refills continuously, which is the mechanism the corpus names, not a reproduction")
    print("   of their production workload.")
    print("   HONEST NOTE ON THE CORPUS FIGURE: the speaker says 'up to 50%... up to 2x or")
    print("   sometimes 3x', which are not the same claim (a 50% reduction IS 2x; 3x would be")
    print("   67%). The transcript is loose here and this blueprint does not tighten it.")
    print("\n   READ THE DIRECTION CAREFULLY -- it is the most commonly inverted result in this")
    print("   topic. The corpus's scenario has the LARGE session monopolising dispatch, so")
    print("   least-attained service protects the SMALL sessions. It is NOT 'protect the long")
    print("   agent from short requests', which is the reverse. And the policy must be keyed on")
    print("   ABSOLUTE service received, not service as a fraction of each program's demand: a")
    print("   ratio-based key is permanently smallest for the largest program and reproduces the")
    print("   very starvation it was meant to fix.")


def exp_turn_priority() -> None:
    print("\n5. TURN PRIORITY -- the 'opposite mechanism', for a DIFFERENT resource")
    print("   corpus: an agent at turn 100 'is likely to finish faster... so if you prioritize")
    print("   those agents, those agents will finish and then evict their KV cache' [T]")
    print("   corpus: 'one addresses the compute saturation, the other addresses the KV cache")
    print("   saturation' [T]\n")

    # X and Y ARE mid-session: 40 turns already served, 4 remaining, 1000 tokens each.
    # That is what makes them "near done" in the corpus's sense -- they carry 40,000 tokens of
    # resident context and are close to releasing it.
    def make():
        return {
            "near-done-X": AgentProgram("near-done-X", turns=44, tokens_per_turn=1000,
                                        initial_turns=40),
            "near-done-Y": AgentProgram("near-done-Y", turns=44, tokens_per_turn=1000,
                                        initial_turns=40),
            "long-Z": AgentProgram("long-Z", turns=40, tokens_per_turn=200),
        }

    print("   2 slots. X and Y are at turn 40 of 44 (40,000 tokens of resident KV each and")
    print("   four turns left); Z is a long-horizon program with 40 cheap turns and nothing")
    print("   served yet. Both policies can see the same information -- they disagree about")
    print("   which resource to spend the next dispatch on.\n")
    print(f"   {'KV cap':>9}  {'policy':>16}  {'X done':>7}  {'Y done':>7}  {'Z done':>7}  "
          f"{'KV freed @':>11}  {'refused':>8}")
    print("   " + "-" * 78)

    def show(cap, policy):
        progs = make()
        r = dispatch(progs, slots=2, cycles=5000, policy=policy, kv_capacity=cap)
        fa = r["finished_at"]
        fx, fy = fa["near-done-X"], fa["near-done-Y"]
        freed = max(fx, fy) if (fx and fy) else None
        label = "unlimited" if cap == 0 else f"{cap:,}"
        fmt = lambda v: "-" if v is None else str(v)   # noqa: E731
        print(f"   {label:>9}  {policy:>16}  {fmt(fx):>7}  {fmt(fy):>7}  {fmt(fa['long-Z']):>7}  "
              f"{fmt(freed):>11}  {r['kv_refused']:>8}")
        return fa, freed

    x_c, y_c = show(0, "least_attained"), show(0, "turn_priority")
    show(88000, "least_attained")
    show(88000, "turn_priority")
    show(120000, "least_attained")
    show(120000, "turn_priority")

    print("\n   THREE REGIMES, AND EACH ONE TEACHES SOMETHING DIFFERENT:")
    print(f"\n   1. NO MEMORY CONSTRAINT (cap unlimited). Least-attained serves Z first, because Z")
    print(f"      has the least service, so it holds X and Y resident until cycle {x_c[1]}; turn")
    print(f"      priority finishes X and Y and releases their context at cycle {y_c[1]}. Same")
    print(f"      total work, and Z still finishes -- but the KV sits in the pool roughly twice as")
    print(f"      long. This is the corpus's claim, and it is visible even without a cap.")
    print(f"\n   2. CAPACITY TIGHT (88,000). Least-attained DEADLOCKS: X and Y hold 40,000 tokens")
    print(f"      each while waiting their turn, Z needs room to continue, and the pool is too full")
    print(f"      to admit anyone. Nothing running means nothing retires, means nothing frees --")
    print(f"      the run never completes. Turn priority survives, because finishing X and Y hands")
    print(f"      their context back and lets Z continue.")
    print(f"      THAT IS THE REAL ARGUMENT FOR TURN PRIORITY, and it is a MEMORY argument that no")
    print(f"      amount of compute-policy tuning can produce.")
    print(f"\n   3. CAPACITY ADEQUATE (120,000). Both complete, and turn priority frees the KV at")
    print(f"      cycle 41 against 81 -- again roughly 2x -- while total completion is unchanged.")
    print("\n   WHY THE DEADLOCK IS NOT A SIMULATION ARTEFACT: a scheduler that fills KV to 100%")
    print("   and has no preemption cannot recover, because the request that would free space is")
    print("   the one that cannot be admitted. Real engines avoid this two ways, and both are")
    print("   design decisions this blueprint has to make explicitly: (a) reserve headroom -- the")
    print("   corpus's saturation gate is KV ~80% full [T], not 100%; and (b) preempt, evicting a")
    print("   resident program and recomputing or swapping its KV on return (T07's tiering).")
    print("   A fairness policy alone is not a substitute for either one.")
    print("\n   WHERE TURN PRIORITY IS HARMFUL: with no KV pressure it is pure starvation, and it")
    print("   penalises exactly the long-horizon agents that agentic workloads are made of --")
    print("   note Z finishing at 121 against 81. Gate it on KV occupancy; that is what the")
    print("   saturation signal in experiment 7 is for.")


def exp_overbatching() -> None:
    print("\n6. OVER-BATCHING -- throughput plateaus, latency never stops rising")
    print("   corpus: saturation gates are KV 80% full or average active requests > 8 [T]\n")

    seqs = [Seq(f"s{i}", 800, 200, arrival=0) for i in range(96)]

    print("   96 requests arriving together (a burst, so the curve is not distorted by a ramp),")
    print("   800 prompt + 200 output tokens each. The engine's decode ceiling is 800 tok/step.")
    print(f"   {'slots':>6}  {'run avg':>8}  {'at full batch':>13}  {'P50':>5}  {'P99':>5}  "
          f"{'idle':>7}  {'goodput @ SLO':>24}")
    print("   " + "-" * 84)
    slos = (10, 15, 25)
    rows = batch_sweep(seqs, [4, 8, 16, 32, 64, 128], slo=slos[1], slos=slos)
    for r in rows:
        gp = "  ".join(f"{r['goodput_at'][s]:>5.0%}@{s:<3}" for s in slos)
        print(f"   {r['slots']:>6}  {r['throughput']:>8,.0f}  {r['saturated_throughput']:>13,.0f}  "
              f"{r['p50']:>5}  {r['p99']:>5}  {r['idle_fraction']:>6.1%}  {gp:>24}")

    print("\n   READ THE COLUMNS SEPARATELY, because they do not peak together:")
    print("     * AT FULL BATCH the rate climbs to the engine's ceiling and then STOPS. It is")
    print("       flat from 8 slots upward, because the engine is already saturated -- so")
    print("       throughput can never tell you that the batch is too deep. It goes quiet, which")
    print("       looks like success on every dashboard.")
    print("     * THE RUN AVERAGE is a worse metric and it DECLINES, which is a measurement")
    print("       artefact rather than a regression: it counts prefill-only steps, and a deeper")
    print("       batch spends proportionally more of its life prefilling. Report the saturated")
    print("       rate; a run average will make you think you have a throughput problem when you")
    print("       have a metric problem.")
    print("     * P50/P99 rise once past the knee. Adding slots does not create capacity; it")
    print("       redistributes a fixed budget more thinly across more requests.")
    print("     * goodput peaks and falls -- IF the SLO binds at all.")

    binding = {s: r for s in slos for r in [max(rows, key=lambda x: x["goodput_at"][s])]}
    print(f"\n   peak-goodput slot count by SLO:")
    for s in slos:
        best = binding[s]
        if best["goodput_at"][s] >= 1.0:
            print(f"     SLO {s:>3}: no peak -- every batch size conforms, so goodput@SLO does")
            print(f"              not discriminate at all. AN SLO YOU ALWAYS MEET IS NOT A")
            print(f"              CONSTRAINT, IT IS DECORATION. Tighten it before reading this")
            print(f"              chart, or the chart is telling you nothing.")
        else:
            print(f"     SLO {s:>3}: peaks at {best['slots']} slots "
                  f"({best['goodput_at'][s]:.0%} conforming), "
                  f"falling to {rows[-1]['goodput_at'][s]:.0%} at 128 slots")
    print("   So 'what is the right batch size' has no answer until the SLO is fixed. Teams that")
    print("   set the SLO after seeing the latency distribution will always find it achievable,")
    print("   and will then tune the batch size against a target that no longer constrains them.")
    print("\n   EXCEPTION: for genuinely offline batch traffic with no SLO, throughput IS the right")
    print("   objective and the peak-goodput slot count is under-provisioned. The metric follows")
    print("   the workload, not the other way round.")


def exp_admission_control() -> None:
    print("\n7. ADMISSION CONTROL -- converting a latency collapse into a visible queue")
    print("   corpus: at saturation the router queues and applies priority bands, so interactive")
    print("   traffic is dispatched and the rest waits for capacity [T]\n")

    print(f"   {'kv usage':>9}  {'active':>7}  {'saturated':>10}  {'reason':>8}")
    print("   " + "-" * 40)
    for kv, active in ((0.42, 3), (0.81, 5), (0.55, 12), (0.95, 20)):
        g = saturation_gate(kv, active)
        print(f"   {kv:>9.2f}  {active:>7}  {str(g['saturated']):>10}  {g['reason']:>8}")

    print("\n   and what the gate buys, under contention, with a premium tenant present:")
    print("   premium-chat keeps 1 turn outstanding; best-effort-batch floods 8. 4 slots.\n")

    def make():
        return {
            "best-effort-batch": AgentProgram("best-effort-batch", turns=16,
                                              tokens_per_turn=100, max_in_flight=8),
            "premium-chat": AgentProgram("premium-chat", turns=4, tokens_per_turn=100,
                                         max_in_flight=1),
        }

    rows = (("ungated  (no bands, FCFS)", "fcfs", False),
            ("gated    (priority bands)", "priority_band", False),
            ("reserved (band + hard gate)", "priority_band", True))
    print(f"   {'mode':>28}  {'premium lat':>11}  {'best-effort lat':>15}  {'slots idle':>10}")
    print("   " + "-" * 72)
    for label, policy, sat in rows:
        progs = make()
        r = dispatch(progs, slots=4, cycles=400, policy=policy,
                     premium={"premium-chat"}, saturation=sat)
        pm = r["latency_stats"]["premium-chat"].get("mean", float("nan"))
        bl = r["latency_stats"]["best-effort-batch"].get("mean", float("nan"))
        print(f"   {label:>28}  {pm:>11.1f}  {bl:>15.1f}  {r['idle_total']:>10}")

    print("\n   READ: the gate does not make the system faster -- it makes the RIGHT traffic fast and")
    print("   the other traffic WAIT. That is the whole point. Ungated, the flood from best-effort")
    print("   is served first because it arrived first and it is deeper, so premium's single")
    print("   request queues behind it; everything slows down together, nothing fails, nothing")
    print("   alerts, and no capacity is ever added.")
    print("   A system that degrades gracefully into uselessness is worse than one that sheds load")
    print("   visibly -- which is why the corpus pairs the gate with a priority band [T].")
    print("   WATCH THE THIRD ROW: hard reservation protects premium best but burns slots while")
    print("   premium is not using them. Reserve-and-release (row 2) is the corpus's shape: the")
    print("   request QUEUES rather than being refused, and the queue depth is the signal that")
    print("   autoscaling (T15) needs in order to add capacity. A gate with no queue is just a")
    print("   dropped request.")
