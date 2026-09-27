"""T09 -- speculative decoding design blueprint: runnable core.

    python run.py

Seven demonstrations, stdlib-only, offline, no GPU. Each one is a DECISION, not a number:
exactness, acceptance decay, draft cost, n-gram lookup, the batch-regime crossover, tree drafting,
and the deployment verdict that decides whether any of it is safe to turn on.

Corpus figures are labelled [T] in the output; everything else is derived and reproducible from
this file. No fabricated benchmarks.
"""
from __future__ import annotations

import sys

from sim import experiments as E


def main() -> int:
    print("=" * 78)
    print("T09 -- SPECULATIVE DECODING: exactness, acceptance, draft cost, batch regime")
    print("=" * 78)
    print("\n[corpus] MTP 'enabled more interactivity which gained about 2x improvement in")
    print("          throughput' [T] llm-d (Pravin, IBM Research)")
    print("[corpus] SGLang supports spec decoding 'from Eagle, MTP to Deep Flash' with")
    print("          'Spec V2 ... for the native spec decoding speed up' [T] Banghua Zhu")
    print("[corpus] agentic workloads are prefill dominated -- 'prefill is occupying like")
    print("          98% of the tokens' [T] llm-d")
    print("[R]     draft/verify paradigm and the Medusa/lookahead families [R]")
    print("          ai-system-design-guide 04-inference-optimization/03-speculative-decoding.md")

    E.exp_exactness()
    E.exp_acceptance_decay()
    E.exp_draft_cost()
    E.exp_ngram_prompt_lookup()
    E.exp_regime_crossover()
    E.exp_tree_vs_linear()
    E.exp_deployment_verdict()

    print("\n" + "=" * 78)
    print("THE ONE-PARAGRAPH SUMMARY")
    print("=" * 78)
    print("""
Speculative decoding is an exact sampler, not an approximation, and that exactness is bought by
one construction -- resampling from the residual max(0, p_target - p_draft) on rejection. Drop it
and the output distribution shifts by several percentage points while the text stays fluent, which
is why the only reliable test is a distribution test nobody runs. Past correctness, the method has
two inputs and one hidden third. The first is acceptance, which decays with position because each
draft token is conditioned on previous guesses, so the draft length has an optimum and the common
'alpha is constant' simplification over-predicts the gain. The second is draft cost, which varies
by an order of magnitude across the drafter families and is the bigger lever at useful draft
lengths; an n-gram drafter's cost is exactly zero, meaning a wrong guess costs nothing at all,
which makes prompt lookup free to be wrong and the best default for any workload whose output
quotes its input. The hidden third is the batch regime: verification is free only because decode is
memory-bound, so past a few hundred concurrent sequences the verify pass costs gamma+1 times as
much and speculation becomes a net loss -- which is why a service can show a speedup at p50 and a
regression at p99, and why speculative decoding is a latency tool for small batches rather than a
throughput feature. Judge it at the load you actually reach, gate it on batch size, and remember
that the workload it is most often proposed for -- long-horizon agents -- is prefill-dominated and
has the least decode to accelerate.
""".strip())
    print("\nOK -- all seven demonstrations completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
