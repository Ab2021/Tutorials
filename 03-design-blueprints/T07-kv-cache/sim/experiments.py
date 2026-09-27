"""The seven experiments. Each answers a question an operator actually asks.

No fabricated benchmarks: every corpus figure is labelled [T] and every modelled figure is
reproducible from this file. Where the model and the corpus disagree in magnitude, the corpus
number is printed alongside and the difference is stated rather than smoothed over.
"""
from __future__ import annotations

import math

from .allocator import (BLOCK_SIZE, BlockPool, BlockTable, contiguous_waste, paged_waste,
                        expected_waste, make_length_distribution, block_size_table)
from .prefix_cache import PrefixCache, sharing_saving
from .tiering import (POOLED_MEMORY, RDMA, TCP, NVME_LOCAL, breakeven, preemption_choice,
                      RetentionPolicy)
from .kvmath import MODELS, kv_table, max_concurrency, kv_quant_effect, offload_headroom, \
    recompute_cliff

GB = 1e9
MB = 1e6


def exp_paged_vs_contiguous() -> None:
    print("\n1. PAGED vs CONTIGUOUS -- why paging is not an optimisation, it is the entry fee")
    print("   corpus: contiguous allocation wastes ~60-80% of KV; paged wastes <4% [R]")
    print("   and: saturation gates are KV ~80% full, fill to 90% of VRAM [T]\n")

    rng = __import__("random").Random(7)
    print("   TWO METRICS, and conflating them is how this arithmetic goes wrong:")
    print("     ratio-of-sums  = wasted tokens / allocated tokens -- a CAPACITY figure.")
    print("                      This is what the corpus's '<4%' [R] means.")
    print("     mean-of-ratios = average per-sequence waste fraction -- a FAIRNESS figure.")
    print("                      Short sequences round up to a whole block and look terrible.\n")
    print(f"   {'skew':>6}  {'mean len':>9}  {'contig (cap)':>13}  {'paged (cap)':>12}  "
          f"{'contig (per-seq)':>17}  {'paged (per-seq)':>16}")
    print("   " + "-" * 82)
    for shape in (1.0, 2.0, 4.0, 8.0):
        lengths = make_length_distribution(rng, n=4000, max_seq_len=2048, shape=shape)
        w = expected_waste(lengths, max_seq_len=2048, block_size=BLOCK_SIZE)
        print(f"   {shape:>6.1f}  {sum(lengths) / len(lengths):>9.0f}  "
              f"{w['contiguous_agg']:>12.1%}  {w['paged_agg']:>11.2%}  "
              f"{w['contiguous_mean']:>16.1%}  {w['paged_mean']:>15.1%}")

    print("\n   READ: the CAPACITY columns are the ones that size a fleet. Contiguous allocation")
    print("   reserves max_seq_len for every sequence because contiguous KV cannot be extended in")
    print("   place, so it wastes 50-89% of the pool as traffic skews shorter -- the corpus's")
    print("   60-80% [R] is exactly this, on a realistic skew. Paged allocation wastes ~1-5%.")
    print("   The PER-SEQUENCE columns are much larger on both sides and for a different reason:")
    print("   a 20-token request rounds up to a whole 16-token block and 'wastes' 20% of itself.")
    print("   That is real, it is a fairness problem (short requests are charged an amortisation")
    print("   penalty), and it is NOT a capacity problem. Reporting only the per-sequence number")
    print("   would wrongly conclude that paging does not work.")
    print("   WHY CAPACITY MATTERS: waste is not a bill, it is a CONCURRENCY LIMIT. At 80% waste")
    print("   on 40 GB of KV you serve ~6 sequences where you could serve ~30 -- same hardware,")
    print("   same model, same latency target. A 5x throughput difference, from an allocator.")


def exp_block_size() -> None:
    print("\n2. BLOCK SIZE -- the one knob, and both sides of it")
    rng = __import__("random").Random(7)
    lengths = make_length_distribution(rng, n=4000, max_seq_len=2048, shape=4.0)

    print(f"   {'block':>6}  {'waste':>9}  {'mean table entries/seq':>24}  {'note':>18}")
    print("   " + "-" * 64)
    for row in block_size_table(lengths):
        notes = {4: "tiny blocks, big table", 16: "the common default [R]",
                 128: "waste creeping up"}
        note = notes.get(row["block_size"], "")
        print(f"   {row['block_size']:>6}  {row['waste']:>8.2%}  "
              f"{row['mean_table_entries']:>24.1f}  {note:>18}")

    print("\n   READ: at the capacity measure, waste falls monotonically as the block shrinks, and")
    print("   the block TABLE grows inversely. The table is gathered per attention kernel launch,")
    print("   so a 4-token block means ~4x the gather entries and ~4x the pointer chasing for a")
    print("   fraction of a percent of memory. 16 is the corpus-typical default [R] because it")
    print("   sits at the knee: going 16 -> 8 halves the block size and buys back about a")
    print("   percentage point, which is not worth doubling the table.")
    print("   EXCEPTION -- the one case where the default is wrong: very SHORT sequences. Every")
    print("   request rounds up to a whole block, so a 20-token prompt occupies 32 tokens of")
    print("   blocks at block size 16 (60% overhead) but 128 tokens at block size 128 (6.4x")
    print("   overhead on that request). Short prompts are the common case in chat and")
    print("   classification traffic, and they are where the block size actually costs something.")
    print("   Long sequences (64k+) do not care either way: one partial block is a rounding error.")

    print("\n   The same curve on the PER-SEQUENCE measure, to show why the two must not be mixed:")
    print(f"   {'block':>6}  {'capac. waste':>13}  {'per-seq waste':>14}")
    print("   " + "-" * 36)
    for bs in (4, 16, 128):
        agg = sum(math.ceil(s / bs) * bs for s in lengths) - sum(lengths)
        agg = agg / sum(math.ceil(s / bs) * bs for s in lengths)
        mean = sum(paged_waste(s, bs) for s in lengths) / len(lengths)
        print(f"   {bs:>6}  {agg:>12.2%}  {mean:>13.2%}")


def exp_prefix_cache() -> None:
    print("\n3. PREFIX CACHE -- the lever that turns quadratic prefill into linear")
    print("   corpus: the router consumes per-request create AND evict events and an offload")
    print("   tier with a retention API exists [T]; prefix-cache-aware routing is a whole talk [T]\n")

    pool = BlockPool(n_blocks=1024, block_size=BLOCK_SIZE)
    cache = PrefixCache(pool, block_size=BLOCK_SIZE)

    system_prompt = 1536      # a real system prompt, in tokens
    turn_delta = 192          # what each conversational turn adds
    turns = 12

    tokens = list(range(system_prompt))
    table = BlockTable("session-a", BLOCK_SIZE)
    no_cache_total = 0
    with_cache_total = 0
    print(f"   system prompt {system_prompt} tokens, {turn_delta} new tokens/turn, {turns} turns")
    print(f"\n   {'turn':>4}  {'seq len':>8}  {'cache hit':>10}  {'prefilled':>10}  "
          f"{'no-cache':>9}")
    print("   " + "-" * 48)
    for turn in range(1, turns + 1):
        tokens = tokens + [500000 + turn * 1000 + i for i in range(turn_delta)]
        hit_tokens, hit_blocks = cache.lookup(tokens)
        cache.retain_all(hit_blocks)
        need = table.needed_blocks(len(tokens)) - len(hit_blocks)
        new_blocks = [pool.allocate() for _ in range(max(0, need))]
        table.blocks = hit_blocks + new_blocks
        table.n_tokens = len(tokens)
        cache.insert(tokens, table.blocks)
        prefilled = len(tokens) - hit_tokens
        no_cache_total += len(tokens)
        with_cache_total += prefilled
        if turn <= 3 or turn == turns:
            print(f"   {turn:>4}  {len(tokens):>8}  {hit_tokens:>10}  {prefilled:>10}  "
                  f"{len(tokens):>9}")
        elif turn == 4:
            print("   ...")

    saved = 1.0 - with_cache_total / no_cache_total
    print(f"\n   totals over {turns} turns: without cache {no_cache_total:,} prompt tokens")
    print(f"                              with cache    {with_cache_total:,} prompt tokens")
    print(f"   prefill work saved: {saved:.1%}   cache hit rate: {cache.hit_rate():.0%}")
    print(f"   pool blocks used: {pool.used}/{pool.total}, cache entries: {cache.retained_blocks}")
    print("\n   READ: with a cache, turn N prefills only its DELTA -- prefill goes from quadratic")
    print("   in turn count to linear. This is the single largest serving win available for chat")
    print("   and agent traffic, and it costs one hash per block.")
    print("   WHERE IT FAILS, SILENTLY: a per-turn timestamp or a nonce in the system prompt")
    print("   invalidates from its own offset. Every request then pays the full prefill and the")
    print("   only symptom is a hit rate pinned near zero. That is why agent harnesses pin the")
    print("   stable prefix (T16) and why the router is cache-aware (T14).")
    print("   AND THE TRAP: block_hash chains the parent, so a hit claims the ENTIRE prefix")
    print("   matches. Hashing blocks independently would let a hit match a block that is")
    print("   identical in isolation but sits under a different prefix -- confident wrong output.")


def exp_sharing() -> None:
    print("\n4. BLOCK SHARING -- best-of-N without n x the prompt KV")
    print("   corpus: 20% -> 32% accuracy at ~16x critic cost [T]; n = 32 fits one batch [T]\n")
    prompt_blocks, gen_blocks = 128, 24     # 2048-token prompt, ~384 generated tokens
    budget = 12 * GB
    print(f"   prompt {prompt_blocks} blocks, generated {gen_blocks} blocks per candidate")
    print(f"   candidate-pool budget: {budget / GB:.0f} GB of HBM for KV\n")
    print(f"   {'n':>4}  {'unshared GB':>12}  {'shared GB':>10}  {'saving':>8}  "
          f"{'unshared?':>10}  {'shared?':>8}")
    print("   " + "-" * 62)
    for n in (1, 2, 4, 8, 16, 32, 64):
        s = sharing_saving(n, prompt_blocks, gen_blocks)
        un_gb = s["unshared_blocks"] * 16 * 327680 / GB
        sh_gb = s["shared_blocks"] * 16 * 327680 / GB
        print(f"   {n:>4}  {un_gb:>10.1f}GB  {sh_gb:>8.1f}GB  {s['saving']:>7.0%}  "
              f"{('yes' if un_gb <= budget / GB else 'NO'):>10}  "
              f"{('yes' if sh_gb <= budget / GB else 'NO'):>8}")

    print("\n   READ: without sharing, best-of-N cost scales with n over the WHOLE sequence. With")
    print("   it, only the GENERATED part scales -- the prompt is paid once. Read the two verdict")
    print("   columns together: at n=16 the UNshared pool overflows a 12 GB budget while the")
    print("   shared one fits at 2.7 GB, and by n=32 it is 25.5 GB against 4.7 GB. That is the")
    print("   difference between n=32 fitting and not fitting at all -- which is the whole reason")
    print("   best-of-N at n=32 is feasible (T05). The saving is not a constant: it approaches")
    print("   the prompt/generated ratio as n grows, so long-prompt best-of-N is where sharing")
    print("   matters most, and short-prompt best-of-N barely needs it.")
    print("   THE MECHANISM IS NOT A CACHE: it is a refcount. Several LIVE sequences point at the")
    print("   same blocks; the block is freed when the last reference drops. Implementing sharing")
    print("   as 'a cache with a TTL' drops prefixes that live sequences are still reading.")
    print("   COPY-ON-WRITE is what makes it safe: siblings share until they disagree.")


def exp_tiering() -> None:
    print("\n5. TIERING -- and why the MEDIUM decides whether tiering is even a win")
    print("   corpus: pooled memory < RDMA << TCP/IP [T]; ~5x TTFT improvement restoring a")
    print("   session's KV from CPU instead of recomputing [T] llm-d\n")

    spec = MODELS["llama-3.1-70b"]
    seq_len = 8000
    bpt = spec.bytes_per_token
    prefill_tps = 12000.0
    n_bytes = seq_len * bpt

    print(f"   70B fp16: {bpt:,.0f} bytes/token; a {seq_len:,}-token session holds "
          f"{n_bytes / GB:.2f} GB of KV")
    print(f"   re-prefill throughput assumed: {prefill_tps:,.0f} tokens/s "
          f"(ILLUSTRATIVE [D])\n")
    print(f"   {'medium':>15}  {'rank':>5}  {'offload (2-way)':>16}  {'recompute':>10}  "
          f"{'winner':>10}")
    print("   " + "-" * 62)
    medias = [POOLED_MEMORY, RDMA, NVME_LOCAL, TCP]
    for m in medias:
        r = breakeven(seq_len, bpt, m, prefill_tps)
        win = "offload" if r["offload_wins"] else "RECOMPUTE"
        print(f"   {m.name:>15}  {m.rank:>5}  {r['offload_s'] * 1000:>14.1f}ms  "
              f"{r['recompute_s'] * 1000:>8.1f}ms  {win:>10}")

    print("\n   READ: on TCP/IP, moving the KV takes LONGER THAN RE-PREFILLING IT. The corpus's")
    print("   ranking (pooled < RDMA << TCP/IP) is not a preference -- on a slow fabric the")
    print("   tiering is strictly worse than recompute, and a team that tiers everything has")
    print("   added latency and called it an optimisation.")
    print("   WHERE THIS FAILS: these are ILLUSTRATIVE [D] bandwidths. Substitute the operator's")
    print("   own measured fabric numbers before using the table. The ORDERING is the corpus's;")
    print("   the crossover length is entirely hardware-dependent.")

    print("\n   The same decision at PREEMPTION time -- cost, and separately, occupancy:")
    for med in (POOLED_MEMORY, RDMA, TCP):
        c = preemption_choice(seq_len, bpt, med, prefill_tps, expected_gap_s=0.05)
        print(f"     {med.name:>14}: swap {c['offload_s'] * 1000:>7.1f}ms vs recompute "
              f"{c['recompute_s'] * 1000:>6.1f}ms -> {c['choice']}")
    print("   and the occupancy question, which the cost column cannot answer:")
    for gap_s in (0.05, 5.0, 300.0):
        held = (gap_s / 300.0) * 6 * GB
        print(f"     gap {gap_s:>6.2f}s -> tier bytes held ~{held / GB:>4.2f}GB at steady state")
    print("   READ: at 8,000 tokens the swap cost is 105 ms against a 667 ms re-prefill on RDMA,")
    print("   so swap wins -- the ~5x TTFT effect the corpus reports [T], reproduced. On TCP the")
    print("   SAME decision inverts: 1.75 s of transfer to avoid a 0.67 s re-prefill.")
    print("   The gap does not change that arithmetic; it changes how many bytes the tier must")
    print("   hold. A tier that evicts before the sequence resumes pays the transfer AND the")
    print("   re-prefill, which is strictly worse than never tiering. Size occupancy first.")

    print("   Retention: the re-arrival distribution sets the TTL, not a fleet-wide constant.")
    pol = RetentionPolicy(max_retained_bytes=20 * GB, default_ttl_s=300.0)
    offered = 0.0
    for name, bytes_kv, re_arr, label in (
        ("agent-tool-call", 2.6 * GB, 0.4, "returns in ~ms"),
        ("chat-session", 0.4 * GB, 45.0, "returns in tens of s"),
        ("one-shot-batch", 5.2 * GB, 100000.0, "never returns"),
    ):
        offered += bytes_kv
        pol.create(name, bytes_kv, ttl_s=300.0)
        v = pol.value_of_retention(name, re_arr, bpt, seq_len, RDMA, prefill_tps)
        print(f"     {name:>16}: retain={str(v['retain']):>5}  ({label})")
        if not v["retain"]:
            pol.evict(name, reason="low-re-arrival")
    print(f"     events emitted for the router [T]: "
          f"{{'create': {sum(1 for e in pol.events if e['event'] == 'create')}, "
          f"'evict': {sum(1 for e in pol.events if e['event'] == 'evict')}}}")
    print(f"     bytes retained after policy: {pol.used_bytes / GB:.1f} GB "
          f"(of {offered / GB:.1f} GB offered)")
    print("   READ: a single fleet-wide TTL gets it wrong in BOTH directions at once -- a TTL")
    print("   sized for chat evicts the agent prefix that returns in milliseconds, and it holds")
    print("   the batch job's 5.2 GB that will never be read again. The per-session policy above")
    print("   does the opposite in both directions: 3.0 GB retained out of 8.2 GB offered, with")
    print("   the one-shot evicted on its re-arrival estimate rather than on a timer. That is why")
    print("   the router is fed both create AND evict events [T], not just a cache-hit count --")
    print("   a hit count alone cannot distinguish 'this tier is working' from 'this tier is full")
    print("   of sessions that will never come back'.")


def exp_capacity() -> None:
    print("\n6. CAPACITY -- the constraint that actually binds")
    print("   corpus worked example: 40e9 / (327e3 x 4000) ~= 30 concurrent sequences [T]\n")

    spec = MODELS["llama-3.1-70b"]
    print(f"   70B fp16 KV per token: 2 x {spec.n_layers} layers x {spec.n_kv_heads} KV heads x "
          f"{spec.head_dim} head_dim x 2 bytes = {spec.bytes_per_token:,.0f}")
    print(f"   ({spec.bytes_per_token / 1024:,.0f} KiB/token)\n")

    print(f"   {'n_ctx':>9}  {'KV per sequence':>17}")
    print("   " + "-" * 30)
    for row in kv_table(spec, (4096, 32768, 131072)):
        print(f"   {row['n_ctx']:>9,}  {row['GB']:>15.1f} GB")

    print(f"\n   {'HBM for KV':>11}  {'avg ctx':>8}  {'raw conc':>9}  {'block-granular':>15}")
    print("   " + "-" * 50)
    for hbm, ctx in ((40 * GB, 4000), (40 * GB, 128000), (180 * GB, 4000), (180 * GB, 128000)):
        c = max_concurrency(spec, hbm, ctx)
        print(f"   {hbm / GB:>9.0f}GB  {ctx:>8,}  {c['raw_concurrency']:>9.1f}  "
              f"{c['block_granular_concurrency']:>15}")

    print("\n   READ: at 40 GB for KV and a 4k average context, ~30 sequences -- the corpus's")
    print("   number, reproduced ([T]: 40e9 / (327e3 x 4000) ~= 30). Push the average context to")
    print("   128k and raw concurrency drops to 1.0 while the BLOCK-GRANULAR answer drops to 0: a")
    print("   single 128k sequence needs 42.9 GB of KV, more than the whole 40 GB budget. That is")
    print("   a capacity wall, not a tuning problem, and no batch size or scheduler change moves")
    print("   it. The fixes are paging plus offload (experiment 5), KV quantization, or a shorter")
    print("   context. Note also that raw and block-granular differ in general -- you cannot serve")
    print("   a fraction of a sequence, and every sequence rounds up to whole blocks.")
    print("   THE ERROR TO AVOID: passing TOTAL HBM as `hbm_for_kv`. A 70B at fp16 has 140 GB")
    print("   of weights and does not fit one 80 GB part at all; subtracting the weights is not")
    print("   a detail, it is the difference between a plan and a fiction.\n")

    print("   KV quantization: the change that moves THIS number (T10 owns the accuracy):")
    print(f"   {'dtype':>8}  {'bytes/token':>13}  {'KV @128k':>10}  {'conc multiplier':>16}")
    print("   " + "-" * 54)
    for row in kv_quant_effect(spec, 131072):
        label = {2.0: "fp16", 1.0: "fp8", 0.5: "int4"}.get(row["dtype_bytes"], "?")
        print(f"   {label:>8}  {row['bytes_per_token']:>13,.0f}  "
              f"{row['total_GB']:>8.1f} GB  {row['concurrency_multiplier']:>15.1f}x")
    print("   READ: fp8 KV doubles concurrency. Weight quantization does NOT touch this number")
    print("   at all -- it moves decode TIME. Two changes, routinely called one word.")
    print("   EXCEPTION: KV quantization is where accuracy risk lives, not weight quantization,")
    print("   because every token's K and V is quantized once and then READ back for the whole")
    print("   life of the sequence. T10 owns the accuracy question.")


def exp_cliff_and_headroom() -> None:
    print("\n7. THE RECOMPUTE CLIFF -- when to stop dropping and start tiering")
    print("   corpus: recomputation cliff at 28k input tokens [T] ROCm/WideEP")
    print("   corpus: rack-scale demo pools memory across 4 servers, one pooled box mid-rack [T]\n")

    spec = MODELS["llama-3.1-70b"]
    for ctx in (4096, 28000, 131072):
        r = recompute_cliff(spec, crossover_tokens=28000, n_ctx=ctx)
        print(f"   n_ctx {ctx:>7,}: {'PAST the cliff' if r['past_cliff'] else 'below':>14}  "
              f"-> {r['guidance']}")
    print("\n   READ: below the cliff, dropping a preempted sequence and re-prefilling it on")
    print("   resume is cheaper than the memory the KV was occupying. Past it, the re-prefill")
    print("   costs more than the memory is worth, and the engine must tier.")
    print("   EXCEPTION: 28k is a CORPUS figure on specific hardware. It is a function of")
    print("   prefill throughput and fabric bandwidth, both of which the operator can measure.")

    print("\n   Headroom: what an offload tier buys")
    for hbm, dram in ((40 * GB, 512 * GB), (180 * GB, 1024 * GB)):
        h = offload_headroom(spec, hbm, dram, avg_ctx=4000)
        print(f"   HBM {hbm / GB:>5.0f} GB -> {h['hbm_sequences']:>4} sequences; "
              f"DRAM {dram / GB:>6.0f} GB -> {h['dram_sequences']:>4} "
              f"({h['dram_multiplier']:.1f}x more)")
    print("\n   READ: the host tier holds an order of magnitude more KV than HBM, at a transfer")
    print("   cost the table in experiment 5 prices. That is the entire case for tiering: it is")
    print("   not faster than HBM, it is enormously cheaper, and it is faster than recompute.")
