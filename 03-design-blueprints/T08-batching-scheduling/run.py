"""T08 -- batching and scheduling design blueprint: runnable core.

    python run.py

Seven experiments, stdlib-only, offline, no GPU. Each one is a scheduling DECISION, not a number:
static vs continuous batching, iteration-level admission, chunked prefill, fairness under
starvation, turn priority under KV pressure, over-batching, and admission control.

Corpus figures are labelled [T] in the output; everything else is derived and reproducible from
this file. No fabricated benchmarks.
"""
from __future__ import annotations

import sys

from sim import experiments as E


def main() -> int:
    print("=" * 78)
    print("T08 -- BATCHING & SCHEDULING: continuous batching, fairness, admission control")
    print("=" * 78)
    print("\n[corpus] 'turn on continuous batching and you are near 42' [T] LLMOps cost talk")
    print("[corpus] 'a naive way is first come first serve but that doesn't add anything extra")
    print("          to your policies because it's going to slow down everything' [T] llm-d")
    print("[corpus] 'saturation at KV cache ~80% full, or average active requests > 8' [T] llm-d")
    print("[corpus] 'least-attained service reduced request latencies by up to 50%... up to 2x")
    print("          or sometimes 3x' [T] llm-d")
    print("[corpus] 'one addresses the compute saturation, the other addresses the KV cache")
    print("          saturation' [T] llm-d")

    E.exp_static_vs_continuous()
    E.exp_iteration_level_admission()
    E.exp_chunked_prefill()
    E.exp_fcfs_vs_least_attained()
    E.exp_turn_priority()
    E.exp_overbatching()
    E.exp_admission_control()

    print("\n" + "=" * 78)
    print("THE ONE-PARAGRAPH SUMMARY")
    print("=" * 78)
    print("""
Continuous batching is an admission decision, not a parameter: a static batch runs until its
longest member retires, so every short sequence in it is a slot that idled after finishing, while
a continuous batch refills a slot the instant anything finishes and retires only the end-of-queue
tail as idle. The penalty tracks the length DISTRIBUTION, which is why real traffic gains far more
than a uniform benchmark predicts. On top of that loop sit four decisions that are usually
conflated. Chunked prefill trades a little TTFT for a lot of ITL stability, and only pays off when
the prompt lengths have a long tail. Fairness must be computed per agentic PROGRAM, not per
request, or it rewards the program that issues the most requests -- the one that most needs
throttling -- and its direction is the reverse of what most people assume: least-attained service
protects the SMALL sessions from one monopolising large one, and the large one finishes faster too
because the small ones free the room. Turn priority is deliberately ANTI-fair, and that is correct,
because it manages a different resource: it finishes near-done agents so their KV is evicted. And
none of it matters past saturation, where the system stops being fast and starts being useless --
so the scheduler needs a gate that converts a latency collapse into a visible queue, because a
system that degrades gracefully into uselessness fails at nothing and therefore alerts on nothing.
""".strip())
    print("\nOK -- all seven experiments completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
