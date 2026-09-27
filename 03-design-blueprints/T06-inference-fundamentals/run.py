"""T06 -- inference fundamentals: the capacity and latency model you build first.

Run:  python run.py

Stdlib only, no network, no GPU. This module is CLOSED-FORM: there is no simulation and no RNG,
which is the difference between a model and a simulation, and why every result here is
whiteboard-reproducible from the formulas printed beside it.

WHAT THIS PROVES
  1. decode's arithmetic intensity is 0.5 FLOPs/byte at fp16 and does NOT depend on model size;
     prefill is on the other side of the machine balance point. That comparison classifies
     every workload and decides which optimisations can apply.
  2. prefill crosses from memory-bound to compute-bound at 2 x machine_balance prompt tokens
  3. attention's crossover is at n_ctx = hidden, so "long context" starts at a different length
     for each model -- it is a property of the model, not a universal number
  4. KV is ~516 KB/token on a 405B at fp16 => ~66 GB for ONE 128k sequence; GQA is why that is
     servable at all
  5. at 300 output tokens, TTFT is NOT the lever -- driving it to zero still misses a 3 s budget
  6. goodput and throughput diverge: throughput rises monotonically with batch, goodput does not

WHAT THIS IS NOT: a profiler, and not a benchmark. No latency in milliseconds is predicted,
because the corpus supplies no per-millisecond baseline to validate against. The output is a
CLASSIFICATION and a LEVER, never a millisecond.

Provenance: [T] transcript, [R] repo, [D] derived.
"""
from __future__ import annotations

from sim import experiments as ex

RULE = "=" * 78


def section(n: int, title: str) -> None:
    print()
    print(RULE)
    print(f"  {n}. {title}")
    print(RULE)


def main() -> None:
    print(RULE)
    print("  T06 - INFERENCE FUNDAMENTALS: the model you build before anything else")
    print(RULE)
    print("  Prefill is compute-bound and sets TTFT. Decode is memory-bandwidth-bound and sets")
    print("  ITL. Every serving optimisation in the other eighteen topics follows from which")
    print("  side of that line it targets.")
    print()
    print("  Model dimensions are the corpus's Llama 3.1 figures [T]. Hardware parameters are")
    print("  ILLUSTRATIVE [D] -- the corpus states no H100 peak -- and are declared in")
    print("  sim/roofline.py so every conclusion is reproducible from them.")

    section(1, "The two phases, and the one comparison that classifies them")
    ex.exp_two_phases()

    section(2, "The prefill crossover")
    ex.exp_prefill_crossover()

    section(3, "Attention is quadratic, MLP is linear")
    ex.exp_attention_quadratic()

    section(4, "The KV budget")
    ex.exp_kv_budget()

    section(5, "The latency budget, and which lever moves it")
    ex.exp_latency_budget()

    section(6, "Goodput, not throughput")
    ex.exp_goodput()

    print()
    print(RULE)
    print("  DONE. Six decisions, no predicted milliseconds.")
    print(RULE)
    print("  Carry away: prefill compute / decode bandwidth; decode intensity is 0.5 FLOPs/byte")
    print("  regardless of size; KV is 516 KB/token on a 405B; TTFT is often not the lever; and")
    print("  throughput can improve while goodput falls.")


if __name__ == "__main__":
    main()
