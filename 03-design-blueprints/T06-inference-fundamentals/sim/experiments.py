"""The six experiments. Each is a DECISION the model is asked to make, not a number it prints.

This module has no RNG: the model is closed-form. That absence is the difference between a
model and a simulation, and it is why the results are whiteboard-reproducible.
"""
from __future__ import annotations

from .roofline import (LLAMA_31, H100_CLASS, classify, intensity, flops_per_token,
                       bytes_moved, prefill_crossover, attention_mlp_ratio)
from .kv import (kv_bytes_per_token, kv_total, max_concurrency, gqa_saving,
                 kv_dtype_effect, weights_bytes, weight_replicas_needed)
from .latency import (ttft, itl, e2e, required_itl, required_output_len, goodput,
                      queue_wait, decode_throughput)

GB = 1e9


def exp_two_phases() -> None:
    print("\n1. THE TWO PHASES -- and the one comparison that classifies them")
    print("   corpus: prefill is compute-bound and sets TTFT; decode is memory-bandwidth-bound")
    print(f"   and sets ITL. [T]   machine_balance = {H100_CLASS.machine_balance:.1f} FLOPs/byte\n")

    print(f"   hardware (ILLUSTRATIVE [D]): peak {H100_CLASS.peak_flops / 1e12:.1f} TFLOPs dense, "
          f"BW {H100_CLASS.memory_bw / 1e12:.2f} TB/s")
    print(f"\n   {'model':>16}  {'phase':>8}  {'intensity':>11}  {'ratio to balance':>17}  {'bound':>14}")
    print("   " + "-" * 74)
    for name, m in LLAMA_31.items():
        for phase, nprompt in (("decode", 0), ("prefill", 2000)):
            bound, ratio = classify(m, H100_CLASS, phase, n_prompt=nprompt)
            inten = intensity(m, phase, n_prompt=nprompt)
            print(f"   {name:>16}  {phase:>8}  {inten:>11.3f}  {ratio:>17.6f}  {bound:>14}")

    vals = {name: intensity(m, "decode") for name, m in LLAMA_31.items()}
    print(f"\n   decode intensity across all three models: "
          f"{', '.join(f'{v:.3f}' for v in vals.values())}")
    d0 = list(vals.values())[0]
    print("   READ: decode's intensity is 1.0 FLOP/byte at fp16 and is IDENTICAL for 8B, 70B and")
    print("   405B. It does not depend on model size at all -- it is 2N FLOPs over 2N bytes.")
    print(f"   Every model here is memory-bound in decode by a factor of ~"
          f"{H100_CLASS.machine_balance / d0:.0f}x.")
    print("   Prefill at 2,000 prompt tokens is on the other side. THAT comparison is the whole")
    print("   roofline story for LLM inference.")


def exp_prefill_crossover() -> None:
    print("\n2. THE PREFILL CROSSOVER -- where a prompt stops being memory-bound")
    print("   prefill intensity = 2 * n_prompt / dtype_bytes = n_prompt at fp16,")
    print("   so the crossing is at n_prompt = machine_balance.\n")

    x = prefill_crossover(LLAMA_31["llama-3.1-405b"], H100_CLASS)
    print(f"   crossover prompt length: {x:,.0f} tokens")
    print(f"\n   {'n_prompt':>10}  {'intensity':>11}  {'ratio':>10}  {'bound':>14}")
    print("   " + "-" * 52)
    for p in (64, 128, 256, int(x), 512, 4096, 32768, 128000):
        bound, ratio = classify(LLAMA_31["llama-3.1-405b"], H100_CLASS, "prefill", n_prompt=p)
        print(f"   {p:>10,}  {p:>11.1f}  {ratio:>10.2f}  {bound:>14}")

    print("\n   READ: short prompts are MEMORY-bound in prefill, long prompts are COMPUTE-bound.")
    print("   This is why prefix caching helps short-prompt workloads (it removes the weight")
    print("   read) and why chunked prefill and sequence parallelism help long-prompt ones.")
    print("   A team that applies a long-context optimisation to a short-prompt workload, or")
    print("   vice versa, will measure no effect and conclude the technique does not work.")


def exp_attention_quadratic() -> None:
    print("\n3. ATTENTION IS QUADRATIC, MLP IS LINEAR -- the crossover is real")
    print("   attention ~ n_ctx^2 * d   ;   MLP ~ n_ctx * d^2   ;   ratio = n_ctx / d\n")

    for name in ("llama-3.1-8b", "llama-3.1-405b"):
        m = LLAMA_31[name]
        print(f"   {name} (hidden d = {m.hidden:,}) -- crossover at n_ctx = {m.hidden:,}")
        print(f"   {'n_ctx':>9}  {'attention / MLP':>16}  {'dominates':>12}")
        print("   " + "-" * 42)
        for n in (1024, 4096, 16384, 65536, 128000):
            r = attention_mlp_ratio(m, n)
            print(f"   {n:>9,}  {r:>16.3f}  {'attention' if r > 1 else 'MLP':>12}")
        print()

    print("   READ: the crossover is at n_ctx = hidden, and hidden differs by 4x between the 8B")
    print("   and the 405B. So 'long context' begins at a different length for each model -- the")
    print("   boundary is a property of the MODEL, not a universal number. Past it, the workload")
    print("   is a different engineering problem, which is why sparse and linear attention exist.")
    print("   A model validated at 8k and deployed at 128k fails for this reason, not for a bug.")


def exp_kv_budget() -> None:
    print("\n4. THE KV BUDGET -- the arithmetic that constrains everything else")
    m = LLAMA_31["llama-3.1-405b"]
    ctx = 128000

    per_tok = kv_bytes_per_token(m)
    print(f"   405B: 2 (K and V) x {m.n_layers} layers x {m.n_kv_heads} KV heads x "
          f"{m.head_dim} head_dim x 2 bytes")
    print(f"   = {per_tok:,.0f} bytes/token  ({per_tok / 1024:.0f} KiB)")
    print(f"   at {ctx:,} tokens: {kv_total(m, ctx) / GB:.1f} GB for ONE sequence\n")

    print(f"   GQA saving: {m.n_heads} query heads / {m.n_kv_heads} KV heads = "
          f"{gqa_saving(m.n_heads, m.n_kv_heads):.0f}x cheaper than full MHA")
    print("   READ: Llama 3.1 keeps n_kv_heads = 8 at EVERY size [T]. KV cost therefore does not")
    print("   scale with parameter count the way weights do -- which is why a 405B is servable")
    print("   at long context at all. A model without GQA would be 16x worse here.\n")

    print(f"   {'KV dtype':>9}  {'bytes/token':>13}  {'KV at 128k':>12}")
    print("   " + "-" * 38)
    for row in kv_dtype_effect(m, ctx):
        print(f"   {row['dtype']:>9}  {row['kv_bytes_per_token']:>13,.0f}  "
              f"{row['kv_total_GB']:>10.1f} GB")

    print(f"\n   weights alone: {weights_bytes(m) / GB:.0f} GB -> spans "
          f"{weight_replicas_needed(m, H100_CLASS)} parts of "
          f"{H100_CLASS.memory_capacity / GB:.0f} GB")
    conc = max_concurrency(m, H100_CLASS, ctx, weights_per_gpu=0.0)
    print(f"   max concurrency at 128k, ignoring the weight footprint: {conc}")
    print("\n   READ: weight quantization and KV quantization are DIFFERENT changes. Weight")
    print("   quantization cuts decode's bytes_moved and therefore its TIME, but does not change")
    print("   its CLASSIFICATION -- decode was memory-bound and remains memory-bound. KV")
    print("   quantization changes this table, and therefore concurrency. Conflating them is the")
    print("   standard error.")


def exp_latency_budget() -> None:
    print("\n5. THE LATENCY BUDGET -- and which lever actually moves it")
    ttft_s, itl_s, n_out = 0.200, 0.020, 300
    total = e2e(ttft_s, itl_s, n_out)
    print(f"   E2E = TTFT + ITL x N_out = {ttft_s} + {itl_s} x {n_out} = {total:.2f} s\n")

    budget = 3.0
    print(f"   target budget: {budget:.1f} s")
    print(f"   required ITL            : {required_itl(budget, ttft_s, n_out) * 1000:.1f} ms")
    print(f"   required output length  : {required_output_len(budget, ttft_s, itl_s):.0f} tokens")
    print(f"   ITL at TTFT = 0 (best possible): {required_itl(budget, 0.0, n_out) * 1000:.1f} ms")

    print(f"\n   {'scenario':>34}  {'E2E':>8}  {'meets 3 s?':>11}")
    print("   " + "-" * 58)
    cases = [
        ("baseline", ttft_s, itl_s, n_out),
        ("halve TTFT", ttft_s / 2, itl_s, n_out),
        ("halve ITL", ttft_s, itl_s / 2, n_out),
        ("halve output length", ttft_s, itl_s, n_out // 2),
        ("TTFT -> 0 (theoretical best)", 0.0, itl_s, n_out),
    ]
    for label, t, i, n in cases:
        v = e2e(t, i, n)
        print(f"   {label:>34}  {v:>7.2f}s  {'YES' if v <= budget else 'no':>11}")

    print("\n   READ: at 300 output tokens, TTFT IS NOT THE LEVER. Halving it gets 6.2 -> 6.1 s;")
    print("   driving it to ZERO still fails the budget. The lever is ITL, or output length.")
    print("   This is the single most common misdiagnosis in serving work, and the model makes")
    print("   it a one-line check rather than an experiment.")


def exp_goodput() -> None:
    print("\n6. GOODPUT, NOT THROUGHPUT -- the metric that can improve while users suffer")
    m = LLAMA_31["llama-3.1-8b"]
    wb = weights_bytes(m)
    qps = 40.0
    ttft_slo, itl_slo = 0.6, 0.030

    print(f"   SLO: TTFT <= {ttft_slo:.2f} s AND ITL <= {itl_slo * 1000:.0f} ms")
    print(f"   traffic: {qps:.0f} req/s, 300 output tokens, 1200 prompt tokens\n")
    print(f"   {'batch':>6}  {'queue wait':>11}  {'TTFT':>8}  {'ITL':>8}  {'tokens/s':>9}  "
          f"{'goodput':>8}")
    print("   " + "-" * 62)

    best = None
    for batch in (1, 2, 4, 8, 16, 32, 64):
        qw = queue_wait(batch, qps)
        t = qw + ttft(1200, 8000.0)
        i = itl(H100_CLASS.memory_bw, wb, batch, kv_slope=0.004)
        reqs = [{"ttft": t, "itl": i, "n_out": 300} for _ in range(200)]
        gp = goodput(reqs, ttft_slo, itl_slo)
        tps = decode_throughput(H100_CLASS.memory_bw, wb, batch)
        print(f"   {batch:>6}  {qw:>10.3f}s  {t:>7.3f}s  {i * 1000:>6.1f}ms  {tps:>9,.0f}  "
              f"{gp:>8.1%}")
        if best is None or gp > best[1]:
            best = (batch, gp)

    print(f"\n   peak goodput at batch = {best[0]} ({best[1]:.1%} conforming)")
    print("\n   READ: throughput rises monotonically with batch -- every row up. Goodput does NOT:")
    print("   queue wait lands directly on TTFT, and the KV/attention slope lands on ITL, so past")
    print("   some batch the SLO is missed and the extra tokens are delivered too late to count.")
    print("   A team tuning on tokens/s will, with complete internal consistency, degrade the")
    print("   user experience and read every dashboard as improving. Alert on goodput.")
