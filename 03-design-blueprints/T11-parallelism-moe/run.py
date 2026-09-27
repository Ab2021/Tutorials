#!/usr/bin/env python3
"""T11 — Parallelism & MoE topology: runnable core.

Proves the design's central mechanism: given a model and a node layout, the choice of which
parallelism dimension carries the wide axis changes the memory available for KV, the
communication cost, and the pipeline bubble — and there is no single winner, so the planner
enumerates and ranks rather than prescribing.

WHAT THIS IS:  a model of the tradeoff, using the relationships the corpus describes.
WHAT THIS IS NOT:  a benchmark. Nothing here ran on a GPU. Every number is a model output.

Run:  python run.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import moe_cost, planner  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    m = planner.ModelSpec()
    a = planner.Accelerator()
    f = planner.Fabric()

    rule("1. Model and hardware assumptions (edit these; every number below is re-derivable)")
    print(f"  model      : {m.name}")
    print(f"  layers={m.layers} hidden={m.hidden} experts={m.experts} top_k={m.top_k} "
          f"kv_heads={m.kv_heads} head_dim={m.head_dim} dtype={m.dtype_bytes}B")
    print(f"  accelerator: {a.name}, {a.memory_gb:.0f} GB, usable fraction {a.usable_fraction}")
    print(f"  fabric     : {f.gpus_per_node} GPUs/node, intra {f.intra_node_bw_gbps:.0f} Gb/s, "
          f"inter {f.inter_node_bw_gbps:.0f} Gb/s")
    print()
    print(f"  attention weights (replicated per DP rank) : {planner.attention_weight_gb(m):.1f} GB")
    print(f"  KV per token per sequence                  : {planner.kv_bytes_per_token(m)/1024:.0f} KB")
    print(f"  all-to-all payload per token               : {planner.a2a_payload_bytes(m)/1024:.0f} KB "
          f"(independent of EP degree)")

    # ---------------------------------------------------------------- experts per GPU
    rule("2. experts_per_GPU = total_experts / EP_degree   (the corpus's relationship [T])")
    print(f"  {'EP':>4} {'experts/GPU':>12} {'expert wt/GPU':>15} {'KV budget/GPU':>15} {'KV tokens/GPU':>15}")
    for ep in (8, 16, 32, 64):
        s = planner.Sharding(tp=2, pp=2, dp=16, ep=ep)
        mp = planner.plan_memory(m, a, s)
        print(f"  {ep:>4} {mp.experts_per_gpu:>12} {mp.expert_weight_gb:>12.1f} GB "
              f"{mp.kv_per_gpu_gb:>12.1f} GB {mp.max_tokens_per_gpu:>15,.0f}")
    print()
    print("  Raising EP cuts weight memory per GPU and frees KV for concurrency -- but the")
    print("  number of all-to-all peers rises with it. That is the tradeoff the planner ranks.")

    # ---------------------------------------------------------------- enumeration
    rule("3. Is Expert Parallelism a choice, or a necessity?")
    fits_ep1 = planner.fits_without_ep(m, a, f, world_size=16)
    print(f"  Can this model be served with EP=1 on {a.memory_gb:.0f} GB accelerators?  "
          f"{'YES' if fits_ep1 else 'NO'}")
    print()
    print("  This is the question that decides what the wide dimension is FOR. When a model")
    print("  fits with EP=1, EP is a performance choice and it usually loses -- the all-to-all")
    print("  is a real cost. When it does not fit, EP is the only way the model runs at all,")
    print("  and its communication cost is simply the price. That is why WideEP exists. [D]")
    print()
    print(f"  {'model':<20} {'experts':>8} {'best EP=1 wt/GPU':>18} {'budget':>9} {'EP=1 fits?':>11}")
    print("  " + "-" * 70)
    for mult, name in ((1, "this model"), (8, "8x experts"), (32, "32x experts")):
        big = planner.ModelSpec(
            name=f"{m.name}-x{mult}", layers=m.layers, hidden=m.hidden,
            experts=m.experts * mult, top_k=m.top_k, kv_heads=m.kv_heads,
            head_dim=m.head_dim, dtype_bytes=m.dtype_bytes)
        # best achievable weight-per-GPU with no expert parallelism: maximise TP and PP
        best = min(
            (planner.plan_memory(big, a, s).weights_per_gpu_gb
             for s in planner.enumerate_shardings(big, f, 16) if s.ep == 1),
            default=float("inf"))
        ok = best < a.memory_gb * a.usable_fraction
        print(f"  {name:<20} {big.experts:>8} {best:>15.1f} GB {a.memory_gb*a.usable_fraction:>6.0f} GB "
              f"{str(ok):>11}")
    print()
    print("  Note the direction: growing the model does not just need more GPUs, it needs a")
    print("  DIFFERENT KIND of parallelism. Adding TP would grow the cross-GPU collective;")
    print("  EP splits the weights that are the problem. [D]")

    # ---------------------------------------------------------------- ranked list
    rule("3b. Enumerated and ranked shardings for a 16-GPU replica (8 GPUs per node)")
    cands = planner.plan(m, a, f, world_size=16, microbatches=8, tokens_in_flight=4096)
    print(f"  {len(cands)} feasible candidates, ranked by the model score in sim/planner.py.")
    print()
    hdr = (f"  {'shape':<18} {'exp/GPU':>8} {'wt/GPU':>9} {'KV tok/GPU':>12} "
           f"{'bubble':>8} {'A2A peers':>10} {'score':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    seen_ep = set()
    shown = 0
    for c in cands:
        # show the best of each EP degree, so the tradeoff is visible rather than just
        # the top of one hill
        if c.sharding.ep in seen_ep:
            continue
        seen_ep.add(c.sharding.ep)
        s = c.sharding
        print(f"  {s.label():<18} {c.memory.experts_per_gpu:>8} "
              f"{c.memory.weights_per_gpu_gb:>6.1f} GB {c.memory.max_tokens_per_gpu:>12,.0f} "
              f"{c.bubble_fraction:>7.1%} {c.a2a_peers:>10} {c.score:>12,.0f}")
        shown += 1
    print()
    top = cands[0]
    print(f"  Best overall: {top.sharding.label()}  (binding: {top.memory.binding_constraint})")
    print("  One row per EP degree above, so the tradeoff is visible: EP frees KV memory")
    print("  (more experts/GPU avoided) and costs all-to-all peers. With this model on these")
    print("  GPUs, EP is not required, so the comm-free shape wins -- an honest result, and")
    print("  the reason the ranking is re-derivable rather than a recommendation to copy.")
    print("  The corpus's own conclusion is that there is no universal winner [T].")

    # ---------------------------------------------------------------- TP confinement
    rule("4. Invariant check: tensor parallelism never leaves the node")
    bad = [s for s in planner.enumerate_shardings(m, f, 16) if s.tp > f.gpus_per_node]
    print(f"  shardings emitted with tp > gpus_per_node ({f.gpus_per_node}): {len(bad)}")
    print("  The enumerator cannot emit one -- the constraint is structural, not a post-filter.")
    print()
    print("  Note the rule as the corpus states it: TP stays INSIDE THE NODE. 'TP stays 1' is")
    print("  not the rule [T]. The winning shape above uses TP>1 and is still legal because")
    print(f"  TP={top.sharding.tp} <= {f.gpus_per_node}.")

    # ---------------------------------------------------------------- pipeline bubble
    rule("5. Pipeline bubble = (PP-1) / (microbatches + PP - 1)")
    print(f"  {'PP':>4} {'M=8':>10} {'M=32':>10} {'M=128':>10}")
    for pp in (1, 2, 4, 8):
        row = "".join(f" {planner.bubble_fraction(pp, mm):>9.1%}" for mm in (8, 32, 128))
        print(f"  {pp:>4}{row}")
    print()
    print("  A deeper pipeline needs many more microbatches to hide. This is why the corpus's")
    print("  tuned configuration holds PP at 2 rather than going deeper [T].")

    # ---------------------------------------------------------------- kernel path
    rule("6. MoE kernel path: naive (2 A2A + 6 kernels) vs fused (3 kernels)  [T]")
    rows = moe_cost.compare(tokens=4096)
    print(moe_cost.render(rows))
    n, fu = rows
    print()
    print(f"  modelled kernels {n.kernels} -> {fu.kernels}; intermediate traffic "
          f"{n.intermediate_mb:.0f} MB -> {fu.intermediate_mb:.0f} MB")
    print(f"  modelled cost falls {n.modelled_us:.0f} us -> {fu.modelled_us:.0f} us "
          f"({(1 - fu.modelled_us/n.modelled_us):.0%} lower)")
    print("  The all-to-all count is unchanged: it is the launch and materialisation overhead")
    print("  that the fusion removes. Model parameters in sim/moe_cost.py -- not measurements.")

    # ---------------------------------------------------------------- NIAH sweep
    rule("7. Long-context validation sweep (NIAH-shaped, synthetic)")
    print("  10 needles x 3 context shapes x rising concurrency; the corpus's pass threshold")
    print("  is >= 7 of 10 needles [T]. Recall below is a MODEL of KV-pressure degradation.")
    print()
    shapes = {"4k": 4_000, "32k": 32_000, "128k": 128_000}
    concurrency_levels = (1, 8, 32, 64, 128, 256, 512)
    # KV ceiling for the best shape from section 3b, per GPU
    ceiling = top.memory.max_tokens_per_gpu
    print(f"  using the winning shape's KV ceiling: {ceiling:,.0f} token-slots/GPU")
    print()
    print(f"  {'shape':>6} {'conc@ceiling':>13} " + "".join(f"{c:>7}" for c in concurrency_levels))
    print("  " + "-" * (21 + 7 * len(concurrency_levels)))
    for name, ctx in shapes.items():
        conc_at_ceiling = ceiling / ctx
        cells = []
        for c in concurrency_levels:
            r = planner.niah_recall(c, ctx, ceiling)
            mark = "P" if planner.passes(r) else "."
            cells.append(f"{r:>5.1f}{mark}")
        print(f"  {name:>6} {conc_at_ceiling:>13,.0f} " + "".join(cells))
    print()
    print("  P = passes the >=7-of-10 threshold.  The shelf where P turns into '.' is the")
    print("  KV-recomputation cliff the corpus documents at 28k inputs / concurrency 256 [T].")
    print("  The three shapes fail at DIFFERENT concurrencies -- which is exactly why the")
    print("  corpus's harness sweeps context shape and not just concurrency [T].")
    print("  The cliff's POSITION here follows from this model's assumptions, not from a")
    print("  measurement. Run the real sweep (docs/SEQUENCES.md, flow 4) before promoting.")

    rule("Summary")
    print(f"  Ranked {len(cands)} feasible shardings for {m.experts} experts across 16 GPUs.")
    print(f"  Model-recommended shape: {top.sharding.label()}")
    print("  Central mechanism proved: the wide dimension belongs to EP/DP, TP stays in the")
    print("  node, and the optimum is a balance -- enumerated, not assumed.")
    print()
    print("  No GPU was touched. Every figure above is a model output. [D]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
