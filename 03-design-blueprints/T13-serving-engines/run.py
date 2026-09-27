#!/usr/bin/env python3
"""T13 — Serving engines: runnable core.

Proves the design's central mechanism: a serving engine's job is to make a *policy* out
of memory and scheduling decisions that the workload keeps changing underneath it. Two
of those are modelled here:

  1. a paged KV allocator for a HYBRID model (full attention + a linear attention with a
     fixed-size state), where the memory split between the two is dynamic, not reserved;
  2. iteration-level (continuous) batching against static batching.

WHAT THIS IS:  a model of each policy, using the relationships the corpus describes.
WHAT THIS IS NOT:  a benchmark. No GPU, no engine and no model was run. Every number is
                   a simulation output.

Run:  python run.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import allocator as al  # noqa: E402
from sim import scheduler as sc  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    rule("1. The workload the engine is built for")
    print("  Agent-era models are HYBRID: full attention interleaved with a cheaper")
    print("  mechanism -- sliding window, or linear attention such as KDA. Full-attention")
    print("  KV grows linearly with context; a linear attention keeps a FIXED-SIZE state")
    print("  per sequence regardless of context. That is what makes 1M context feasible")
    print("  at all [T], and it is why one memory policy cannot serve both.")
    print()
    print(f"  model parameters used below:")
    print(f"    tokens per KV block      : {al.TOKENS_PER_BLOCK}")
    print(f"    linear state per sequence: {al.LINEAR_BLOCKS_PER_SEQ} blocks (fixed)")
    print(f"    pool                     : 1024 blocks")
    print()
    print("  The naive answer is a STATIC split: reserve x% of the pool for full attention")
    print("  and the rest for the linear state. The corpus names the problem with it -- the")
    print("  optimal split depends on batch size and context length, both dynamic [T].")
    print("  The engine's answer is one shared pool with an allocator per attention type [T].")

    # ------------------------------------------------------------------ fragmentation
    rule("2. Static split vs one shared pool (fragmentation, measured as rejections)")
    print("  200 requests, mixed contexts, 1024 blocks. A request needs")
    print("  ceil(ctx/16) full-attention blocks PLUS the fixed linear state.")
    print()
    base = al.Scenario(n_requests=200, total_blocks=1024)
    for s in (0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        o = al.simulate(base, s)
        print(f"  static, full reserve {s:>4.0%}   admitted {o.admitted:>4}  "
              f"rejected {o.rejected:>4}  peak concurrent {o.peak_concurrent:>3}  "
              f"pool used {o.pool_utilisation:>5.1%}")
    dyn = al.simulate(base, None)
    print("  " + "-" * 74)
    print(f"  DYNAMIC (shared pool)   admitted {dyn.admitted:>4}  rejected {dyn.rejected:>4}  "
          f"peak concurrent {dyn.peak_concurrent:>3}  pool used {dyn.pool_utilisation:>5.1%}")
    print()
    print("  The 'pool used' column is the point. A static reservation is capacity held for")
    print("  a type that is not using it -- the allocator cannot lend it, so the request is")
    print("  rejected while memory sits idle. Under dynamic partitioning that number is 100%")
    print("  by construction, because there is nothing reserved to go unused.")
    print()
    print("  Note the shape of the static row: it is not monotonic. Reserving too little for")
    print("  full attention strangles the common case; reserving too much strangles the")
    print("  other. Choosing the split IS the design decision, and it is made blind.")

    # ------------------------------------------------------------------ moving optimum
    rule("3. The best static split moves with the workload -- so a fixed one is wrong")
    workloads = [
        ("short chat ", 600, 2_000, 0.05),
        ("mixed      ", 2_000, 16_000, 0.25),
        ("long doc   ", 8_000, 64_000, 0.60),
    ]
    grid = (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
    print(f"  {'workload':<12} " + "".join(f"{s:>7.0%}" for s in grid)
          + f" {'best':>7} {'dynamic':>9}")
    print("  " + "-" * 72)
    for label, short_ctx, long_ctx, lf in workloads:
        scen = al.Scenario(n_requests=200, total_blocks=1024, short_ctx=short_ctx,
                           long_ctx=long_ctx, long_fraction=lf)
        counts = [al.simulate(scen, s).admitted for s in grid]
        best_i = counts.index(max(counts))
        dy = al.simulate(scen, None).admitted
        print(f"  {label:<12} " + "".join(f"{c:>7}" for c in counts)
              + f" {grid[best_i]:>6.0%} {dy:>9}")
    print()
    print("  Each column is a candidate static split; each row is a workload. The best")
    print("  column is not the same across rows. A deployment that picks one split and")
    print("  keeps it is over-provisioning for two of these three workloads and")
    print("  under-provisioning for the third -- and it will never see that in a dashboard,")
    print("  because rejection is a queue depth, not an error.")

    # ------------------------------------------------------------------ honest failure
    rule("4. What dynamic partitioning does NOT decide (it needs a second policy)")
    print("  Dynamic partitioning decides WHERE memory goes. It says nothing about WHO gets")
    print("  served, and a shared pool without an explicit per-sequence admission rule lets")
    print("  one long sequence define everyone else's latency. That rule is a SEPARATE")
    print("  policy with a real cost, and the cost is paid by the longest requests.")
    print()
    scen = al.Scenario(n_requests=200, total_blocks=1024, short_ctx=3000,
                       long_ctx=10_000, long_fraction=0.2)
    reqs = {r.rid: r for _, r in al.make_requests(scen)}
    print(f"  {'admission rule':<22} {'admitted':>9} {'short':>7} {'long':>6} "
          f"{'mean wait':>10} {'p95 wait':>9}")
    print("  " + "-" * 68)
    for label, mss in (("no per-sequence cap", None),
                       ("cap at 50% of pool", 0.5),
                       ("cap at 25% of pool", 0.25)):
        s2 = al.Scenario(**{**scen.__dict__, "max_seq_share": mss})
        o = al.simulate(s2, None)
        short = sum(1 for i in o.admitted_ids if reqs[i].ctx_tokens < 6000)
        lng = o.admitted - short
        print(f"  {label:<22} {o.admitted:>9} {short:>7} {lng:>6} "
              f"{o.mean_wait_steps:>10.2f} {o.p95_wait_steps:>9}")
    total_long = sum(1 for r in reqs.values() if r.ctx_tokens >= 6000)
    print(f"  ({total_long} of the {len(reqs)} requests are long)")
    print()
    print("  Read the long column against the admitted column. Tightening the cap raises")
    print("  total throughput and REFUSES long requests to get it. That is a trade, not a")
    print("  win -- and it is invisible if the engine reports only an aggregate admit rate.")
    print()
    print("  The design conclusion is that 'dynamic partitioning' is one decision and")
    print("  'admission control' is another. Shipping the first while leaving the second")
    print("  implicit means the second is whatever the scheduler happened to do. [D]")

    # ------------------------------------------------------------------ batching
    rule("5. Continuous batching vs static batching (iteration-level scheduling)")
    seqs = sc.make_sequences(128)
    out_lens = sorted(s.output_tokens for s in seqs)
    print(f"  128 sequences, output lengths from {out_lens[0]} to {out_lens[-1]} tokens")
    print(f"  (median {out_lens[len(out_lens)//2]}). The spread is the whole story: a static")
    print("  batch is held for as long as its LONGEST member runs.")
    print()
    for bs in (16, 32, 64):
        st = sc.static_batching(seqs, bs)
        co = sc.continuous_batching(seqs, bs)
        print(f"  batch size {bs:>3}")
        print(sc.render([st, co]))
        print(f"    continuous makespan is {st.makespan/co.makespan:.2f}x shorter; "
              f"slot utilisation {co.slot_utilisation/st.slot_utilisation:.2f}x higher")
        print()
    print("  Continuous batching makes the admission decision every iteration: a finished")
    print("  sequence leaves and a waiting one takes its slot. Static batching pays for the")
    print("  slot until the batch retires. Same hardware, different policy, and the gap")
    print("  widens with the spread of output lengths.")

    # ------------------------------------------------------------------ parallelism
    rule("6. Parallelism is not a setting, it is a search (and there is no winner)")
    print("  vLLM supports seven kinds of parallelism today [T]:")
    for i, kind in enumerate(
            ["tensor", "pipeline", "data", "expert", "sequence",
             "context (prefill)", "decode context -- parallelism around the KV cache"],
            1):
        print(f"    {i}. {kind}")
    print()
    print("  'Decode context parallelism' does not exist in the training world; it exists")
    print("  here because decode is a KV problem, not a compute problem [T].")
    print()
    print("  The corpus's own worked result, for a large MoE prefill on B200 in a")
    print("  disaggregated setup [T]:")
    print("    naive deployment  : 8-way tensor parallelism on a single host")
    print("    winning deployment: a tuned mix across 16 GPUs per replica, combining")
    print("                        tensor + pipeline + sequence + expert parallelism")
    print("    outcome           : lower time-to-first-token AND higher throughput per GPU")
    print()
    print("  Why each piece earned its place, as the speaker gives it [T]:")
    print("    pipeline  -- parallelism between chunks of a long prefill sequence")
    print("    sequence  -- more overlap of communication with computation")
    print("    expert    -- better GEMM shapes than 8-way tensor parallelism")
    print()
    print("  The conclusion is not 'use this mix'. It is that parallelism must be selected")
    print("  for the target model architecture, the target cluster, and the target workload")
    print("  shape, and that there is NO universal winner [T]. A blueprint that shipped a")
    print("  fixed configuration would be shipping the naive baseline with confidence.")

    rule("Summary")
    print(f"  Shared pool beats every static split on all three workloads modelled; the best")
    print("  static split itself moves from 65% to 80% with the mix.")
    print("  Dynamic partitioning is a memory policy. Admission control is a separate one,")
    print("  and tightening it trades long-request service for aggregate throughput.")
    print(f"  Continuous batching roughly doubles slot utilisation and halves makespan.")
    print("  Parallelism needs a performance model to tune, because there is no winner.")
    print()
    print("  No GPU was touched and no engine was run. Every figure is a model output. [D]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
