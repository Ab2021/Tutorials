"""T07 -- KV cache design blueprint: runnable core.

    python run.py

Seven experiments, stdlib-only, offline, no GPU. Each one answers a question an operator asks
about KV: how much memory paging actually saves, what block size to use, what a prefix cache is
worth, whether best-of-N fits, which transfer medium makes tiering pay, what the concurrency
ceiling is, and when to stop dropping KV and start tiering it.

Corpus figures are labelled [T] in the output; everything else is derived and reproducible from
this file. No fabricated benchmarks.
"""
from __future__ import annotations

import sys

from sim import experiments as E


def main() -> int:
    print("=" * 78)
    print("T07 -- KV CACHE: allocation, sharing, prefix caching, tiering")
    print("=" * 78)
    print("\n[corpus] 'paged -> <4% waste vs 60-80% contiguous' [R]")
    print("[corpus] 'saturation at KV 80%, fill to 90% of VRAM' [T]")
    print("[corpus] 'pooled memory < RDMA << TCP/IP' [T]")
    print("[corpus] '~5x TTFT improvement restoring a session from CPU' [T] llm-d")
    print("[corpus] 'recomputation cliff at 28k tokens' [T]")
    print("[corpus] 'a rack-scale pool across 4 servers, one pooled box mid-rack' [T]")

    E.exp_paged_vs_contiguous()
    E.exp_block_size()
    E.exp_prefix_cache()
    E.exp_sharing()
    E.exp_tiering()
    E.exp_capacity()
    E.exp_cliff_and_headroom()

    print("\n" + "=" * 78)
    print("THE ONE-PARAGRAPH SUMMARY")
    print("=" * 78)
    print("""
Paging is the entry fee, not an optimisation: without it, 60-80% of KV memory is reserved for
sequences that never reach their maximum length, and the concurrency ceiling falls by that
factor. Everything interesting is built ON TOP of the block table. Prefix caching turns a
conversation's prefill from quadratic in turn count to linear in the delta, and it fails
silently when anything in the prefix is unstable. Refcounting blocks lets n best-of-N siblings
share one prompt. Tiering extends HBM with DRAM that holds an order of magnitude more KV -- but
only on a medium fast enough that the round trip beats re-prefilling, which on TCP/IP it does
not. And the retention TTL must follow each session's re-arrival distribution, because a single
fleet-wide constant evicts the prefixes that return in milliseconds while holding the ones that
never return at all.
""".strip())
    print("\nOK -- all seven experiments completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
