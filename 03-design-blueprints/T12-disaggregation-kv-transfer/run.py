#!/usr/bin/env python3
"""T12 — Disaggregation & KV transfer: runnable core.

Proves the design's central mechanism: separating prefill from decode only pays if the KV
can get from one pool to the other (and back again later) for less than it costs to
recompute — and the point where that flips is a computable number, not a preference.

WHAT THIS IS:  a model of the tradeoff, using the relationships the corpus describes.
WHAT THIS IS NOT:  a benchmark. No GPU, no RDMA NIC, no pooled-memory box was touched.
                   Every number below is a model or simulation output.

Run:  python run.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import kv_lifecycle as kl  # noqa: E402
from sim import transfer as tf  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    prefill_tps = 8000.0     # model parameter: single-pass prefill tokens/s per worker [D]
    decode_tps = 800.0       # model parameter: AGGREGATE batched decode tokens/s [D]

    rule("1. How big is the thing we are moving?")
    per_token = tf.kv_bytes_per_token()
    print(f"  KV per token per sequence = 2 x {tf.LAYERS} layers x {tf.KV_HEADS} kv_heads "
          f"x {tf.HEAD_DIM} x {tf.DTYPE_BYTES}B")
    print(f"                            = {per_token:,.0f} bytes  ({per_token/1024:.0f} KB)")
    print()
    print(f"  {'prompt':>10} {'KV size':>12}   context")
    for n in (1_000, 8_000, 32_000, 128_000):
        gb = tf.kv_bytes(n) / 1e9
        print(f"  {n:>10,} {gb:>9.2f} GB   "
              f"{'a long agent transcript' if n >= 32000 else ''}")
    print()
    print("  A 32k prompt's KV is larger than many models' weights. That is why a tier")
    print("  exists at all, and why 'just keep it in HBM' is not an answer. [D]")

    # ------------------------------------------------------------------ break-even
    rule("2. Fetch the KV back, or recompute it?  (the break-even)")
    print("  Offload + reload costs 2 x (bytes/bw + latency). Recompute costs")
    print(f"  prompt_tokens / {prefill_tps:,.0f} tokens/s. Fetch wins when recompute is larger.")
    print()
    print(f"  {'tier':<22} {'round trip 32k':>15} {'recompute 32k':>15} {'fetch wins?':>13}")
    print("  " + "-" * 68)
    nbytes = tf.kv_bytes(32_000)
    for t in tf.TIERS:
        rt = tf.round_trip_seconds(nbytes, t)
        rc = tf.recompute_seconds(32_000, prefill_tps)
        print(f"  {t.name:<22} {rt*1000:>12.0f} ms {rc*1000:>12.0f} ms "
              f"{str(tf.offload_wins(32_000, nbytes, t, prefill_tps)):>13}")
    xbw = tf.crossover_bandwidth_gbps(prefill_tps)
    print(f"  The crossover is a BANDWIDTH, not a length. Per token, fetching costs")
    print(f"  16 x bytes_per_token / bw and recomputing costs 1/{prefill_tps:,.0f} s. Setting")
    print(f"  them equal gives a tier bandwidth of {xbw:.1f} Gb/s -- independent of prompt")
    print("  length, because both sides scale linearly with tokens. [D]")
    print()
    print(f"  {'tier':<22} {'bandwidth':>12} {'vs crossover':>14} {'verdict':>10}")
    print("  " + "-" * 62)
    for t in tf.TIERS:
        verdict = "fetch" if t.bandwidth_gbps > xbw else "recompute"
        print(f"  {t.name:<22} {t.bandwidth_gbps:>9,.0f} Gb/s "
              f"{t.bandwidth_gbps/xbw:>13.1f}x {verdict:>10}")
    print()
    print("  What DOES depend on length is the fixed per-transfer latency's share, so")
    print("  small blocks are penalised twice: less bytes to amortise the latency over.")
    print(f"  {'block tokens':>12} {'round trip (pooled)':>20} {'of which latency':>17}")
    print("  " + "-" * 52)
    pooled = tf.tier("pooled")
    for bt in (16, 128, 1024, 8192):
        rt = tf.round_trip_seconds(tf.kv_bytes(bt), pooled)
        lat = tf.latency_penalty_seconds(tf.kv_bytes(bt), pooled)
        print(f"  {bt:>12,} {rt*1000:>17.1f} ms {lat/max(rt,1e-12):>16.0%}")
    print()
    print("  That is the argument for larger blocks -- and against them: fewer, larger")
    print("  blocks amortise latency but reuse at coarser granularity, so a partially")
    print("  shared prefix cannot be reused. block_tokens is a real tuning knob (LLD, section 9).")
    print()
    print("  The pooled-memory tier's provenance is a VENDOR REPORT that sharing beat RDMA")
    print("  [T]. It is ordered here on that basis. It was not measured in this program.")
    print()
    print("  The corpus reports up to 5x better TTFT when an agentic session returns after")
    print("  a pause, using CPU KV offload [T]. That is a reported result for a specific")
    print("  workload -- the break-even above is what tells you whether it applies to yours.")

    # ------------------------------------------------------------------ P:D ratio
    rule("3. The prefill:decode split tracks the workload (there is no universal ratio)")
    print("  The corpus tests 1P1D and 2P2D nightly and reports that 2P2D vs 3P1D depends")
    print("  on the workload, with prefill-heavy work favouring more prefill [T].")
    print()
    total = 16
    print(f"  A total of {total} workers to place, across five request shapes.")
    print()
    print(f"  {'workload':<26} {'prompt':>8} {'output':>8} {'pref work':>10} {'dec work':>10} "
          f"{'=> split':>10} {'imbal':>7}")
    print("  " + "-" * 84)
    mixes = [
        ("long doc, short answer", 32_000, 100),
        ("RAG / document QA", 8_000, 300),
        ("code completion", 2_000, 60),
        ("chat", 1_000, 400),
        ("generation-heavy", 500, 2_000),
    ]
    for name, ptok, otok in mixes:
        p, d = tf.pd_split(total, ptok, otok, prefill_tps, decode_tps)
        pw, dw = tf.per_request_work_s(ptok, otok, prefill_tps, decode_tps)
        imb = tf.imbalance(p, d, ptok, otok, prefill_tps, decode_tps)
        print(f"  {name:<26} {ptok:>8,} {otok:>8,} {pw:>9.2f}s {dw:>9.2f}s "
              f"{f'{p}P{d}D':>10} {imb:>7.2f}")
    print()
    print("  The recommendation runs from prefill-dominated to decode-dominated across a")
    print("  single table. A hard-coded 1P1D is right for roughly one row of it -- which is")
    print("  exactly what the corpus reports: 2P2D versus 3P1D depends on the workload, and")
    print("  prefill-heavy work wanted more prefill [T].")
    print()
    print("  Sizing on 2P4D because it looked best is NOT supported here: the corpus")
    print("  presents it as preliminary and the speaker explicitly says not to read the")
    print("  performance off those numbers [T].")

    # ------------------------------------------------------------------ lifecycle
    rule("4. KV block lifecycle: eviction is normal, THRASH is the failure")
    print("  40 sessions x 6 turns, each pausing 60s between turns -- the agentic shape")
    print("  where a session calls a tool and comes back to a context it already prefilled.")
    print("  A quarter of them leave and never return (the abandon rate). Two policies:")
    print("    recompute_only : the block is released the moment the session pauses")
    print("    retain         : the pausing session's block is pinned for a bounded TTL")
    print()
    base = kl.Scenario(n_sessions=40, turns=6, pause_s=60.0, capacity=96, pin_fraction=0.30)
    rows = kl.sweep_capacity(base, capacities=(48, 72, 96, 144, 192, 240, 360))
    print(kl.render_sweep(rows))
    print()
    print("  The delta column is not a marginal improvement. Under recompute_only the hit")
    print("  rate is STRUCTURALLY zero: the policy discards precisely the blocks a")
    print("  returning session needs. No amount of capacity fixes that -- the ceiling on")
    print("  the retain column is set by the abandon rate, since a session that never")
    print("  comes back can never register a hit.")
    print()
    print("  This is the corpus's CPU-KV-offload result in miniature: the reported ~5x")
    print("  TTFT improvement on session return [T] is available only to a design that")
    print("  retains the KV at all.")

    # ------------------------------------------------------------------ pin quota
    rule("5. More pinning is not monotonically better (retention competes for capacity)")
    print("  Fixed tight capacity (72 blocks), sweep the pin quota. Pins protect the")
    print("  sessions that come back -- and they hold capacity the ACTIVE traffic needs,")
    print("  while a departed session's pin returns nothing at all. [D]")
    print()
    print(f"  {'pin quota':>10} {'hit rate':>10} {'evictions':>10} {'thrash':>8} "
          f"{'grants':>8} {'redeemed':>9} {'pin eff':>8}")
    print("  " + "-" * 68)
    for pf in (0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80):
        sc = kl.Scenario(n_sessions=40, turns=6, pause_s=60.0, capacity=72,
                         pin_fraction=pf, policy="retain")
        st = kl.simulate(sc)
        print(f"  {pf:>9.0%} {st.hit_rate:>10.1%} {st.evictions:>10} "
              f"{st.thrash_rate:>8.1%} {st.pin_grants:>8} {st.pin_hits:>9} "
              f"{st.pin_efficiency:>8.1%}")
    print()
    print("  Read the two columns together, because they move in OPPOSITE directions:")
    print("  raising the pin quota lowers the hit rate (pins crowd out active traffic) and")
    print("  also lowers thrash (protected blocks stop being recycled). There is no single")
    print("  'right' quota -- it is a position on that curve, chosen from the measured")
    print("  pause distribution. That is why 'keep everything warm' is not a policy. [D]")
    print()
    print("  thrash_rate counts only evictions of blocks that had ALREADY proven reusable")
    print("  (uses >= 2). Evicting a never-reused block is the policy working correctly,")
    print("  so it is excluded from the alarm -- otherwise the alarm is pure noise. [D]")

    # ------------------------------------------------------------------ ledger
    rule("6. What the router must be told (the event contract)")
    sc = kl.Scenario(n_sessions=40, turns=6, pause_s=60.0, capacity=96,
                     pin_fraction=0.30, policy="retain")
    st = kl.simulate(sc)
    print(f"  accesses={st.accesses}  hits={st.hits}  misses={st.misses}")
    print(f"  creations={st.creations}  evictions={st.evictions}  "
          f"pin grants={st.pin_grants}  denials={st.pin_denials}")
    print(f"  Every one of those {st.creations} creations and {st.evictions} evictions is a")
    print("  router input [T].")
    print()
    print("  A router that sees only creations will route requests to blocks that are gone,")
    print("  and the")
    print("  only symptom is a cache hit rate that quietly collapses -- no error, no")
    print("  exception, just recomputation nobody asked for.")
    print()
    print("  The retention API that would let a caller pin blocks across a tool call is an")
    print("  NVIDIA PR in the corpus, NOT shipped [T]. The pin mechanism simulated above is")
    print("  this blueprint's internal contract, written to be satisfiable by that API later.")

    rule("Summary")
    print(f"  KV per token: {per_token/1024:.0f} KB -> {tf.kv_bytes(32000)/1e9:.1f} GB at 32k.")
    print(f"  Offload beats recompute at 32k on every tier modelled except "
          f"{sum(1 for t in tf.TIERS if not tf.offload_wins(32000, tf.kv_bytes(32000), t, prefill_tps))} of {len(tf.TIERS)}.")
    print("  The right P:D split moves with the workload -- no universal ratio.")
    print("  Eviction is a normal lifecycle stage; thrash is the failure, and it is a rate.")
    print()
    print("  No GPU, NIC or memory pool was touched. Every figure is a model output. [D]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
