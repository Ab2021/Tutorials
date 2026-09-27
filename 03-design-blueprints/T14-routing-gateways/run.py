#!/usr/bin/env python3
"""T14 — Routing and gateways: runnable core.

Proves the design's central mechanism: the router in front of an inference cluster does
two jobs that are easy to conflate, and both are policy problems rather than plumbing.

  1. PLACEMENT (routing). Not "pick the least busy box". Build a live view of where the
     KV cache is, FILTER the candidate set by it, then RANK what remains. The model shows
     the trade this creates -- cache affinity against load balance -- and shows that the
     live view is what lets you have some of both.
  2. ADMISSION (flow control). Only matters once the cluster is saturated, saturation is
     operator-defined, and the queue policy is a choice. The model measures what the
     corpus's suggested choice costs, including the part that is easy to leave unsaid.

WHAT THIS IS:  a model of each policy, using the relationships the corpus describes.
WHAT THIS IS NOT:  a benchmark. No gateway, no cluster and no engine was run. Every
                   number below is a simulation output from the modules in sim/.

Run:  python run.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import flowcontrol as fc  # noqa: E402
from sim import router as rt  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    # ----------------------------------------------------------------- the pipeline
    rule("1. What the router actually does")
    print("  The corpus is specific about the pipeline, and it is not a load balancer [T]:")
    print()
    print("     build a live view of where the cache is  ->  FILTER  ->  RANK")
    print()
    print("  The view is built from the KV events the engine emits. The corpus says the")
    print("  router consumes the per-request CREATE event AND the EVICT event [T]. Eviction")
    print("  is a normal stage of a block's lifecycle, not an error -- the same discipline")
    print("  T12 applies. A router that keeps only creates drifts: it believes a prefix is")
    print("  resident after it has been evicted.")
    print()
    print("  The corpus's own description of the cheap alternative: it is HASH-BASED, and")
    print("  it 'has all the consistency problems' [T]. Both are modelled below.")
    print()
    print("  Two responsibilities are separated throughout this file:")
    print("    routing   = PLACEMENT -- which instance serves this request")
    print("    flow ctrl = ADMISSION -- whether it is dispatched at all, once saturated")

    # ------------------------------------------------------- affinity vs balance
    rule("2. Four placement policies: the affinity / balance trade")
    print("  8 instances, 3000 requests, Zipf-distributed prefixes (a few hot, most rare).")
    print("  Zipf is the whole reason prefix routing is worth anything: with uniform")
    print("  prefixes every instance would miss equally often, and nothing would matter. [D]")
    print()
    print("  'misroute %' = share of requests sent to an instance that could NOT reuse the")
    print("  prefix while ANOTHER instance held it. The prefill is recomputed and the cached")
    print("  copy sits idle on a different box. It does not show up in a hit-rate average.")
    print()
    print("  'peak/mean' is how far the busiest instance sits above the average. The busiest")
    print("  instance saturates first, so 'eff cap' -- cluster capacity in INSTANCE")
    print("  EQUIVALENTS -- is n / (peak/mean). That is arithmetic on this model's output,")
    print("  not a measurement. [D]")
    for cap, label in ((24, "loose (24 prefixes fit per instance)"),
                       (8, "tight (8 prefixes fit per instance)")):
        print()
        print(f"  cache capacity: {label}")
        rows = rt.sweep(n_instances=8, n_requests=3000, hot_prefixes=512, cache_capacity=cap)
        print(rt.render(rows))
    print()
    print("  Read the two ends first:")
    print("    round_robin / least_loaded -- perfect balance (eff cap 8.00 of 8) and NO")
    print("      cache affinity at all. ~25-29% of requests are misrouted: a quarter of the")
    print("      traffic re-prefills something the cluster already had. Load-aware is not")
    print("      cache-aware, and the corpus's point is that load-aware alone is naive.")
    print("    prefix_precise -- the live view. Best effective capacity in the table.")
    print()
    print("  The interesting comparison is the middle two, and it does NOT go all one way.")
    print("  Hash routing posts the HIGHEST hit rate (81.8% vs 77.3%). Sticky placement is")
    print("  genuinely good for cache affinity. What it cannot do is balance: its peak")
    print("  instance carries 2.22x the mean, so only 3.61 of the 8 instances are usable")
    print("  before something saturates. The precise view gives up 4.5 points of hit rate")
    print("  and buys back 52% more usable cluster. Under saturation that is the trade that")
    print("  matters -- which is the regime the corpus is describing.")
    print()
    print("  Reported honestly: this model does NOT reproduce the corpus's precise-beats-")
    print("  approximate gap on hit rate. Here hashing wins that column outright. The")
    print("  precise view's advantage shows up in capacity, not in cache hits.")

    # --------------------------------------------------------- cluster size
    rule("3. The same trade as the cluster grows")
    print("  Varying instance count, capacity 24, same workload. This one is worth sitting")
    print("  with, because it says adding replicas is not a fix.")
    print()
    print(f"  {'n':>4} {'policy':<16} {'hit rate':>9} {'peak/mean':>10} {'eff cap':>8}")
    print("  " + "-" * 52)
    for n in (4, 8, 16, 32):
        rows = dict(rt.sweep(n_instances=n, n_requests=3000, hot_prefixes=512,
                             cache_capacity=24))
        for pol in ("prefix_approx", "prefix_precise"):
            s = rows[pol]
            print(f"  {n:>4} {pol:<16} {s['hit_rate']:>9.1%} "
                  f"{s['load_imbalance']:>10.2f} {s['effective_capacity']:>8.2f}")
    print()
    print("  Hash routing's imbalance gets WORSE with scale: 1.52x at n=4 rising to 6.20x")
    print("  at n=32, because a fixed set of hot prefixes concentrates on proportionally")
    print("  fewer boxes as the cluster grows. Adding replicas does not dilute a hot key.")
    print()
    print("  The precise router tracks it down to 5.83x -- and then stops. It cannot go")
    print("  below the imbalance the WORKLOAD has. 512 Zipf prefixes against a cache that")
    print("  holds 24 per instance is a working set far smaller than the cluster, so most")
    print("  traffic has few legitimate homes no matter how good the router is.")
    print("  Routing removes the imbalance routing caused. It cannot remove the rest.")

    # --------------------------------------------------------- scale-out churn
    rule("4. Scale-out: the consistency problem the corpus names")
    print("  One instance set change mid-run: 8 instances scale out to 10 at the halfway")
    print("  mark. An index-based hash remaps EVERY prefix when the divisor changes. [T] for")
    print("  the claim; [D] for this way of modelling it.")
    print()
    rows = rt.sweep(n_instances=8, n_requests=3000, hot_prefixes=512,
                    cache_capacity=24, scale_to=10)
    print(rt.render(rows))
    print()
    print("  Hash routing goes from 0 misroutes to 54 (1.8%) and its imbalance worsens from")
    print("  2.22x to 2.72x. The precise router stays at 0.0% misroutes and actually gains")
    print("  hit rate, because new instances emit creates like any other and the view")
    print("  absorbs them.")
    print()
    print("  State the size of this honestly: 1.8% is a SMALL effect, and it would be")
    print("  smaller still under a consistent-hash ring, which moves only ~1/n of prefixes")
    print("  instead of all of them. The durable point is not the magnitude. It is that")
    print("  even a perfect ring cannot tell the router which prefixes actually lost their")
    print("  cache, because a hash-only router holds no evict signal to tell it with.")

    # ------------------------------------------------------------- flow control
    rule("5. Flow control: what happens once the cluster is saturated")
    print("  Placement decides where a request goes. It does not decide whether the cluster")
    print("  should accept it. The corpus separates these, and puts flow control behind a")
    print("  saturation signal that is OPERATOR-DEFINED -- e.g. 'KV cache 80% full', or")
    print("  'average active requests above 8' [T]. Both thresholds are policy, not physics.")
    print()
    sc = fc.Scenario()
    print(f"  Offered load {sc.arrival_rate:.0f} req/step against a cluster that can take"
          f" {sc.dispatch_capacity}/step.")
    print(f"  Premium traffic is {sc.premium_fraction:.0%} of arrivals with a"
          f" {sc.premium_slo}-step budget; best-effort gets {sc.best_effort_slo} steps.")
    print("  Offered load is ~1.5x capacity, so the cluster is saturated for most of the")
    print("  run -- the only condition under which any of these policies differs.")
    print()
    print(fc.render(fc.run_all(sc)))
    print()
    print("  The corpus's read on the naive default is that first-come-first-serve 'doesn't")
    print("  add anything extra to your policies because it's going to slow down")
    print("  everything' [T]. The model agrees, and is blunt about why: fcfs and admit_none")
    print("  produce IDENTICAL rows. A queue with no policy is not a policy.")
    print()
    print("  What the corpus suggests instead is priority bands -- under saturation only")
    print("  premium traffic is dispatched [T]. That works, and it works spectacularly on")
    print("  the class it is meant to protect: premium SLO attainment goes from 2.9% to")
    print("  99.3%. Now the part that is easy to leave unsaid: best-effort does not get")
    print("  slower, it gets STARVED. Its attainment barely moves (11.1% -> 12.5%), 1324")
    print("  requests are shed outright, and its mean wait only looks better because the")
    print("  requests that would have waited longest were dropped rather than queued.")
    print("  Mean wait is a lying metric under a shedding policy. Count the sheds.")
    print()
    print("  This is a trade an operator is entitled to make. It is not a trade an operator")
    print("  should make by accident, which is what happens when the dashboard shows mean")
    print("  wait falling and nothing shows the sheds.")

    # ------------------------------------------------------------------- the join
    rule("6. Where the two halves meet")
    print("  Routing and flow control are usually discussed separately. They meet at")
    print("  effective capacity:")
    print()
    rows = dict(rt.sweep(n_instances=8, n_requests=3000, hot_prefixes=512, cache_capacity=24))
    for pol, note in (("round_robin", "perfectly balanced, no affinity"),
                      ("prefix_approx", "best affinity, worst balance"),
                      ("prefix_precise", "live view: both, partly")):
        s = rows[pol]
        print(f"    {pol:<16} eff cap {s['effective_capacity']:>5.2f} of 8"
              f"   hit {s['hit_rate']:>5.1%}   ({note})")
    print()
    print("  The cluster the flow controller believes it is protecting is 8 instances. Two")
    print("  of these three policies leave it with materially fewer, and the flow controller")
    print("  has no way to know: its saturation threshold is a number an operator typed in.")
    print("  A saturation signal derived from a cluster that is only 45% reachable is a")
    print("  signal about the wrong thing.")
    print()
    print("  That is the design argument for the live view, and it is not about hit rate.")

    # -------------------------------------------------------------------- summary
    rule("Summary")
    print("  Routing is placement, not load balancing: build a live view from create AND")
    print("  evict events, filter by it, then rank. Keeping only creates is the approximate")
    print("  policy the corpus warns about.")
    print("  Load-aware-only routing misroutes ~25-29% of a Zipf workload -- it re-prefills")
    print("  what the cluster already holds.")
    print("  Hash routing wins cache affinity (81.8% vs 77.3%) and loses capacity (3.61 vs")
    print("  5.48 usable instances of 8). Under saturation the second number is the one that")
    print("  decides whether the SLO holds.")
    print("  Scale-out costs a hash-only router its consistency; the effect is small (1.8%)")
    print("  and a consistent-hash ring would shrink it further, but the missing evict")
    print("  signal cannot be recovered by a better hash.")
    print("  Flow control only matters under saturation, and FCFS is indistinguishable from")
    print("  doing nothing. Priority bands take premium attainment from 2.9% to 99.3% by")
    print("  shedding best-effort traffic -- watch the sheds, not the mean wait.")
    print()
    print("  No cluster was routed to and no gateway was run. Every figure is a model")
    print("  output from sim/router.py and sim/flowcontrol.py. [D]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
