"""T10 -- quantization design blueprint: runnable core.

    python run.py

Eight demonstrations, stdlib-only, offline, no GPU. Each one is a DECISION, not a number: the
precision->bytes->concurrency chain, quantization granularity, outlier severity, NF4 vs INT4,
AWQ's salient-channel policy, the KV frontier, the corpus's cost ladder, and the eval gate.

Corpus figures are labelled [T]/[R] in the output; everything else is derived and reproducible from
this package. No fabricated benchmarks.
"""
from __future__ import annotations

import sys

from sim import experiments as E


def main() -> int:
    print("=" * 78)
    print("T10 -- QUANTIZATION: precision, bytes, error, and what it costs")
    print("=" * 78)
    print("\n[corpus] the cost ladder -- 'Start with naive 16-bit serving one request at a time as")
    print("          100 cost units. Turn on continuous batching and you are near 42. Quantize")
    print("          to 4-bit and you reach 26. Cache the stable system prompt and you land")
    print("          around 11.' [T] Cut LLM Cost & Latency")
    print("[corpus] 'Four-bit quantization is nearly free on many models, but on some it quietly")
    print("          drops accuracy. So always rerun your evals after quantizing.' [T]")
    print("[corpus] 'AWQ and GPTQ to quantize' [T]; SGLang 'natively support 8-bit and also 4-bit")
    print("          training ... FP4 native rollout ... without any performance loss' [T]")
    print("[R]     the precision/quality table [R]")
    print("          03-training-and-adaptation/07-quantization-deep-dive.md")
    print("[R]     'allow 4x higher concurrency on the same GPU' via KV quantization [R]")

    E.exp_precision_to_concurrency()
    E.exp_granularity()
    E.exp_outlier_severity()
    E.exp_nf4_vs_int4()
    E.exp_awq_protection()
    E.exp_kv_frontier()
    E.exp_cost_ladder()
    E.exp_quality_gate()

    print("\n" + "=" * 78)
    print("THE ONE-PARAGRAPH SUMMARY")
    print("=" * 78)
    print("""
Quantization is four things wearing one name. The first is an accounting chain with no choices in
it: bits are not bytes, because a 4-bit weight with a per-group fp16 scale costs 4 + 16/group_size
bits, and the KV cache is quantized separately from the weights. That second fact is the one teams
miss -- and the two levers turn out to answer different questions, which the concurrency arithmetic
makes plain. Weight quantization is worth only about 1.2x on concurrency, because on a large part
the weights are a small share of the budget; its real role is FITTING, and for a 70B in bf16 on an
80 GB card it is the difference between a deployment and none. KV quantization is worth a
consistent 4x on concurrency at every context length, and it is the only lever that makes long
context serveable at all. The second thing is the quantizers themselves, whose error depends far
more on the tensor's dynamic range than on the bit width -- and whose aggregate SNR can even
IMPROVE as the tensor becomes more extreme, while the worst channel is destroyed outright. The
metric that improves is the one people quote; the metric that matters is the one that degrades,
the same tail-over-mean discipline as T08's goodput and T09's p99. Per-channel and group-wise
scales are flat across severity because each is fitted locally, and that -- not the bit count -- is
what makes them the default. The third is the cost ladder, whose four rungs act on different terms
of one equation: batching divides the weight term, quantization shrinks the weight and KV terms,
KV quantization shrinks only the KV term, and prefix caching removes only the prefill term. Which
rung is worth climbing is therefore a property of the workload's shape, and a ladder measured on one
shape does not transfer to another -- the corpus's own caching win is 57% on a prompt-heavy workload
and 1% on an output-heavy one. The fourth is the honest conclusion: quality loss is not a smooth
function of reconstruction error, no single exponent reconciles the corpus's own three anchors, and
the only way to know whether a quantization is safe remains what the corpus says it is -- run the
evals, and gate on the worst slice rather than the average.
""".strip())
    print("\nOK -- all eight demonstrations completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
