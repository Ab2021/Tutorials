"""T10 -- seven demonstrations, each one a decision an operator actually has to make."""
from __future__ import annotations

import math

from . import cost as C
from . import precision as P
from . import quality as Q
from . import quantize as QZ

# The model used throughout: an 8B in the corpus's own size table [R], on one H100-class part.
PARAMS_B = 8.0
HBM_GB = 80.0
BALANCE = 295.0            # H100-class flops/byte
LAYERS, KV_HEADS, HEAD_DIM = 32, 8, 128


def _snr_table() -> dict[str, float]:
    """Measured SNR for each precision the corpus's quality column names.

    Measured on a REALISTIC synthetic tensor rather than assumed, so the quality fit in experiment 6
    is tied to what the quantizers actually do.
    """
    w = QZ.synthetic_weights()
    return {
        "fp8":  QZ.snr_db(w, QZ.quantize_groupwise(w, 8, 128)),
        "int4": QZ.snr_db(w, QZ.quantize_groupwise(w, 4, 32)),
        "int2": QZ.snr_db(w, QZ.quantize_groupwise(w, 2, 32)),
    }


def exp_precision_to_concurrency() -> None:
    print("\n1. PRECISION -> BYTES -> CONCURRENCY -- the chain with no choices in it")
    print("   weight bytes = params x effective_bits / 8;  concurrency = (usable - weights) / (kv x ctx)\n")

    snrs = _snr_table()
    print(f"   an 8B model on one 80 GB part at util 0.90, ctx 2048, GQA {KV_HEADS} kv heads\n")
    print(f"   {'precision':>10}  {'bits':>5}  {'eff':>6}  {'weights':>9}  {'concurrency':>12}")
    print("   " + "-" * 50)
    for name, bits, grp in [("bf16", 16, None), ("fp8", 8, 128), ("int8", 8, None),
                            ("int4", 4, 128), ("nf4", 4, 64), ("int2", 2, 32)]:
        eff = P.effective_bits(bits, grp)
        w_gb = P.weight_bytes(PARAMS_B, bits, grp) / 1e9
        r = P.max_concurrency(HBM_GB, PARAMS_B, bits, 16, 2048, LAYERS, KV_HEADS, HEAD_DIM)
        print(f"   {name:>10}  {bits:>5}  {eff:>6.3f}  {w_gb:>8.2f}G  {r['sequences']:>12.1f}")

    print("\n   READ: BITS ARE NOT BYTES. `eff` is the honest column -- a 4-bit weight with a")
    print("   per-group fp16 scale costs 4 + 16/group_size bits, which is why int4 at group 128 is")
    print("   4.125 and nf4 at group 64 is 4.25. A 4-bit model is not exactly a quarter of bf16.")
    print("\n   AND WEIGHT QUANTIZATION ALONE BARELY MOVES CONCURRENCY, which is the surprise.")

    print(f"\n   {'configuration':>34}  {'weights':>9}  {'kv/seq':>9}  {'conc':>8}  {'vs bf16':>8}")
    print("   " + "-" * 74)
    base = None
    rows = [("bf16 weights, bf16 KV", 16, 16), ("int4 weights, bf16 KV", 4, 16),
            ("bf16 weights, fp8 KV", 16, 8), ("int4 weights, fp8 KV", 4, 8),
            ("int4 weights, int4 KV", 4, 4)]
    for name, wb, kb in rows:
        r = P.max_concurrency(HBM_GB, PARAMS_B, wb, kb, 2048, LAYERS, KV_HEADS, HEAD_DIM)
        base = r["sequences"] if base is None else base
        ratio = r["sequences"] / base
        print(f"   {name:>34}  {r['weights_gb']:>8.2f}G  {r['kv_gb_per_seq']:>8.3f}G  "
              f"{r['sequences']:>8.1f}  {ratio:>7.2f}x")

    print("\n   THE 4x CLAIM NEEDS BOTH [R] ('allow 4x higher concurrency on the same GPU').")
    print("   Weight quantization alone moves only the SUBTRAHEND -- and at this context the KV")
    print("   budget is most of what is left. KV quantization moves the DENOMINATOR. They multiply,")
    print("   so doing one and not the other captures roughly a quarter of the available gain.")

    print("\n   AND THE PER-SEQUENCE CROSSOVER IS FURTHER OUT THAN MOST TEAMS ASSUME,")
    print("   while the AGGREGATE answer is degenerate -- total resident KV always equals the")
    print("   free budget, because concurrency absorbs whatever is left. The informative")
    print("   comparison is per sequence:")
    for ctx in [2048, 8192, 32768, 131072]:
        r = P.max_concurrency(HBM_GB, PARAMS_B, 16, 16, ctx, LAYERS, KV_HEADS, HEAD_DIM)
        print(f"     ctx {ctx:>6}: kv/seq {r['kv_gb_per_seq']:>6.2f}G vs weights "
              f"{r['weights_gb']:>5.2f}G  (kv is {r['kv_gb_per_seq'] / r['weights_gb'] * 100:>3.0f}% "
              f"of the weights)  -> {r['sequences']:>5.1f} sequences")
    cross = P.max_concurrency(HBM_GB, PARAMS_B, 16, 16, 1, LAYERS, KV_HEADS, HEAD_DIM)
    kv1k = cross["kv_gb_per_seq"] * 1000
    print(f"   One sequence's KV overtakes the weights at ctx ~ "
          f"{cross['weights_gb'] / cross['kv_gb_per_seq'] / 1000:.0f}k tokens -- much further out")
    print("   than the 'long context' intuition. THE REAL NUMBER IS THE MARGINAL ONE: each extra")
    print(f"   1k tokens of context costs {kv1k:.3f} GB PER CONCURRENT SEQUENCE, so at 200")
    print(f"   sequences that is {kv1k * 200:.1f} GB per 1k tokens -- more than the weights, at any")
    print("   context length. Context and concurrency are the same budget spent two ways.")



def exp_granularity() -> None:
    print("\n2. GRANULARITY -- the only genuine trade in the method, and its true storage price")
    print("   smaller groups -> a scale fitted to more homogeneous weights -> less error")
    print("   smaller groups -> more scales stored                     -> more bytes\n")

    w = QZ.synthetic_weights()
    print(f"   tensor {len(w)} x {len(w[0])}, zero-centred normal with "
          f"{8} outlier channels at 12x magnitude, 4-bit\n")
    print(f"   {'method':>22}  {'eff bits':>9}  {'SNR dB':>8}  {'worst chan':>11}  {'bytes':>11}")
    print("   " + "-" * 68)
    variants = [
        ("per-tensor", QZ.quantize_per_tensor(w, 4), 4, None, 1),
        ("per-channel", QZ.quantize_per_channel(w, 4), 4, None, len(w)),
        ("group 32", QZ.quantize_groupwise(w, 4, 32), 4, 32, len(w) * 2),
        ("group 16", QZ.quantize_groupwise(w, 4, 16), 4, 16, len(w) * 4),
        ("group 8", QZ.quantize_groupwise(w, 4, 8), 4, 8, len(w) * 8),
        ("nf4 group 64", QZ.quantize_nf4(w, 64), 4, 64, len(w)),
    ]
    for name, q, bits, grp, n_scales in variants:
        eff = P.effective_bits(bits, grp)
        n_weights = len(w) * len(w[0])
        nbytes = n_weights * eff / 8.0 + (0 if grp else n_scales * 2)
        print(f"   {name:>22}  {eff:>9.3f}  {QZ.snr_db(w, q):>8.2f}  "
              f"{QZ.worst_channel_error(w, q)[1]:>11.4f}  {nbytes:>10.0f}B")

    print("\n   READ THE `eff bits` COLUMN AGAINST THE SNR COLUMN -- they move in OPPOSITE")
    print("   directions, and that is the whole trade. Group 8 beats group 32 on SNR and pays for")
    print("   it in bytes. The right group size is the one where the marginal dB stops being worth")
    print("   the marginal byte, and it is workload-specific.")
    print("\n   BUT LOOK AT `worst chan` FIRST. It is the column that decides whether the model")
    print("   still works. Per-tensor 4-bit destroys a channel outright -- one outlier row sets the")
    print("   scale for everything, so every small weight rounds to zero. Aggregate SNR looks")
    print("   survivable; the worst channel does not. Per-channel and group-wise fix exactly this,")
    print("   which is why nobody ships per-tensor for weights.")
    print("\n   A MEAN-ONLY REPORT HIDES THIS. Same discipline as T08's goodput and T09's p99: the")
    print("   tail decides, and the tail is where the destroyed feature lives.")


def exp_outlier_severity() -> None:
    print("\n3. OUTLIER SEVERITY -- same bit width, wildly different error")
    print("   The fact that makes '4-bit is nearly free' true of some models and false of others.\n")

    print(f"   4-bit, tensor 96 x 64, 4% of channels scaled by the severity factor\n")
    print(f"   {'severity':>9}  {'per-tensor':>11}  {'per-channel':>12}  {'group 32':>10}  "
          f"{'worst chan pt':>14}")
    print("   " + "-" * 64)
    for r in QZ.outlier_severity_sweep([1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0], bits=4, group_size=32):
        print(f"   {r['severity']:>9.1f}  {r['per_tensor_snr']:>11.2f}  {r['per_channel_snr']:>12.2f}  "
              f"{r['groupwise_snr']:>10.2f}  {r['worst_channel_tensor']:>14.4f}")

    print("\n   READ THE `per-tensor` COLUMN CAREFULLY, BECAUSE IT DOES SOMETHING WORSE THAN")
    print("   GET WORSE: it is NON-MONOTONE. As severity rises from 1 to 12 it falls (14.76 ->")
    print("   8.48) -- and then it RISES again (8.48 -> 16.61 at severity 48). Read alone, that")
    print("   last move says the quantization got BETTER as the tensor got more extreme.")
    print("\n   IT DID NOT. This is SNR's blind spot, and it is worth understanding: SNR is")
    print("   signal-energy over noise-energy, and the outlier channels contribute to BOTH. At high")
    print("   severity the signal grows faster than the error, so the ratio improves WHILE the")
    print("   small weights -- which the huge scale rounds to zero -- are being destroyed.")
    print("   THE `worst chan pt` COLUMN IS THE TRUTH: 0.23 -> 1.00, monotonically worse, and")
    print("   1.0000 means a channel whose output is pure error. The metric that improves is the")
    print("   one people quote; the metric that matters is the one that degrades.")
    print("\n   THIS IS THE SAME PATTERN AS T08 AND T09. An average that improves while the tail")
    print("   collapses. There, it was a p99 latency regression hidden by a better mean; here, a")
    print("   destroyed feature hidden by a better SNR. Report the tail.\n")
    print("   AND NOTE WHICH QUANTIZERS ARE FLAT: per-channel (19.28 -> 19.97 dB) and group-wise")
    print("   (20.22 -> 20.61 dB) barely move across a 48x change in severity, precisely because")
    print("   each scale is fitted locally. That flatness is what makes them the default -- not")
    print("   their absolute SNR.")
    print("\n   THE OPERATIONAL CONSEQUENCE: 'we quantized to 4-bit and it was fine' is a statement")
    print("   about the tensor you tested. A different checkpoint, a different layer, or a")
    print("   fine-tune can move the severity -- and the same 4-bit config will not be fine.")



def exp_nf4_vs_int4() -> None:
    print("\n4. NF4 vs INT4 -- why the grid SHAPE matters, not just the bit count")
    print("   Both are 4 bits. Both have 16 levels. They are not equivalent.\n")

    w = QZ.synthetic_weights()
    nf4 = QZ.nf4_levels()
    int4 = QZ.int4_levels()
    print("   the 16 levels of each, normalised to [-1, 1]:")
    print("     NF4 (normal quantiles) :", " ".join(f"{v:+.3f}" for v in nf4[:8]))
    print("                              ", " ".join(f"{v:+.3f}" for v in nf4[8:]))
    print("     INT4 (uniform grid)    :", " ".join(f"{v:+.3f}" for v in int4[:8]))
    print("                              ", " ".join(f"{v:+.3f}" for v in int4[8:]))

    q_nf4 = QZ.quantize_nf4(w, 64)
    q_i4_g64 = QZ.quantize_groupwise(w, 4, 64)

    print(f"\n   {'quantizer':>22}  {'SNR dB':>8}  {'worst chan':>11}")
    print("   " + "-" * 45)
    print(f"   {'NF4, group 64':>22}  {QZ.snr_db(w, q_nf4):>8.2f}  "
          f"{QZ.worst_channel_error(w, q_nf4)[1]:>11.4f}")
    print(f"   {'INT4 uniform, group 64':>22}  {QZ.snr_db(w, q_i4_g64):>8.2f}  "
          f"{QZ.worst_channel_error(w, q_i4_g64)[1]:>11.4f}")

    print("\n   READ THE SPACING. NF4's levels are dense near zero and sparse in the tails; INT4's")
    print("   are evenly spaced. LLM weights are zero-centred normal, so most of the mass is near")
    print("   zero -- exactly where NF4 has the most resolution and INT4 wastes it.")
    print("   THE CORPUS'S OWN EXPLANATION [R]: each NF4 bin 'contains an equal number of values")
    print("   from the normal distribution. This prevents clustering of weights and ensures that")
    print("   the model preserves as much information (entropy) as possible'.")
    print("\n   AND THE HONEST CAVEAT: on weights that are NOT normally distributed, the advantage")
    print("   narrows or vanishes. NF4 is a bet on the weight distribution, and it is a good bet")
    print("   for transformer weights -- which is why it is the QLoRA standard [R].")


def exp_awq_protection() -> None:
    print("\n5. AWQ -- protecting the salient channels, and why 1% is enough")
    print("   Not a quantizer: a POLICY on top of one. Keep the important channels exact.\n")

    w = QZ.synthetic_weights(outlier_frac=0.04, outlier_scale=12.0)
    base = QZ.quantize_groupwise(w, 4, 32)
    base_snr = QZ.snr_db(w, base)

    print(f"   4-bit group 32, tensor {len(w)} x {len(w[0])}; baseline SNR {base_snr:.2f} dB\n")
    print(f"   {'salient frac':>13}  {'channels':>9}  {'SNR dB':>8}  {'gain':>7}  {'storage cost':>13}")
    print("   " + "-" * 58)
    for frac in [0.0, 0.01, 0.02, 0.05, 0.10, 0.25]:
        q, prot = QZ.quantize_with_protected_channels(w, 4, 32, frac)
        s = QZ.snr_db(w, q)
        over = QZ.protected_storage_overhead(frac)
        print(f"   {frac:>13.2f}  {len(prot):>9}  {s:>8.2f}  {s - base_snr:>+6.2f}  "
              f"{over * 100:>12.1f}%")

    print("\n   READ: the first 1% of channels -- the most salient -- recover a disproportionate")
    print("   share of the error, and 25% is barely better than 10%. That is the empirical")
    print("   justification for the corpus's stated 1% [R], and the reason AWQ beats GPTQ at")
    print("   aggressive bit widths: it spends a few percent of storage exactly where the error")
    print("   is concentrated.")
    print("\n   THE STORAGE COST IS THE OTHER HALF. Protecting 1% at 16 bits against 4 bits")
    print("   elsewhere costs 0.01 x (16-4)/4 = 3% more bytes. The ratio of dB recovered to bytes")
    print("   spent is what makes this worth doing -- and it is why AWQ is a POLICY, not a format:")
    print("   the same quantizer with the same bit width gives different quality depending on")
    print("   which channels you chose to protect.")
    print("\n   THE CAVEAT: real salience comes from ACTIVATION statistics on a calibration set,")
    print("   which the corpus states explicitly [R]. This module uses the weight row's L1 norm as")
    print("   the offline stand-in -- correlated with activation salience, not equal to it.")


def exp_kv_frontier() -> None:
    print("\n6. WHICH LEVER -- weights or KV? The answer is not the same question")
    print("   Weight quantization fixes the numerator. KV quantization fixes the denominator.\n")

    print("   8B on 80 GB, util 0.90 -- the two levers side by side, at every context\n")
    print(f"   {'ctx':>7}  {'bf16/bf16':>10}  {'int4 w':>8}  {'int4 kv':>8}  {'int4 w+kv':>10}  "
          f"{'w gain':>7}  {'kv gain':>8}")
    print("   " + "-" * 74)
    for ctx in [2048, 8192, 32768, 131072]:
        a = P.max_concurrency(HBM_GB, PARAMS_B, 16, 16, ctx, LAYERS, KV_HEADS, HEAD_DIM)
        w = P.max_concurrency(HBM_GB, PARAMS_B, 4, 16, ctx, LAYERS, KV_HEADS, HEAD_DIM,
                              weight_group=128)
        k = P.max_concurrency(HBM_GB, PARAMS_B, 16, 4, ctx, LAYERS, KV_HEADS, HEAD_DIM)
        b = P.max_concurrency(HBM_GB, PARAMS_B, 4, 4, ctx, LAYERS, KV_HEADS, HEAD_DIM,
                              weight_group=128)
        print(f"   {ctx:>7}  {a['sequences']:>10.1f}  {w['sequences']:>8.1f}  {k['sequences']:>8.1f}  "
              f"{b['sequences']:>10.1f}  {w['sequences'] / a['sequences']:>6.2f}x  "
              f"{k['sequences'] / a['sequences']:>7.2f}x")

    print("\n   NEITHER MULTIPLIER CHANGES WITH CONTEXT, and that is the finding -- both are")
    print("   ratios of the form (usable - weights)/kv, so context cancels out of the RATIO and")
    print("   survives only in the absolute number. What changes with context is what the same")
    print("   multiplier is WORTH:")
    print("     at 2k context, 4x on 208 sequences is 208 extra sequences;")
    print("     at 128k context, 4x on 3.3 sequences is 10 extra -- and THAT is the difference")
    print("     between a service that cannot run long context and one that can.")
    print("\n   SO THE TWO LEVERS ANSWER DIFFERENT QUESTIONS:")
    print("     KV quantization  -> CONCURRENCY. A consistent ~4x, at any context.")
    print("     weight quantization -> FITTING. Only ~1.2x here, because on an 80 GB part the")
    print("                          weights are a small share of the budget -- but change the")
    print("                          model or the GPU and it becomes the whole answer:")
    for params, gpu in [(8.0, 24.0), (70.0, 80.0)]:
        r16 = P.max_concurrency(gpu, params, 16, 16, 4096, LAYERS, KV_HEADS, HEAD_DIM)
        r4 = P.max_concurrency(gpu, params, 4, 16, 4096, LAYERS, KV_HEADS, HEAD_DIM, weight_group=128)
        verdict = ("FITS" if r4["fits"] else "does not fit") if not r16["fits"] else "fits either way"
        print(f"       {params:.0f}B on {gpu:.0f} GB: bf16 {r16['sequences']:>6.1f} seq, "
              f"int4 {r4['sequences']:>6.1f} seq  -> {verdict}")
    print("   A 70B in bf16 does not fit on one 80 GB part AT ALL. There, weight quantization is")
    print("   not an optimization -- it is the difference between a deployment and no deployment,")
    print("   and no amount of KV quantization substitutes for it.")

    print("\n   THE ACCURACY SIDE IS DIFFERENT TOO. KV quantization degrades ATTENTION rather than")
    print("   the weights, so it does not show up in the same evals. A model can score identically")
    print("   on a short-context benchmark and fail at 64k, because the errors accumulate over")
    print("   positions the benchmark never exercises. Weight quantization's damage is visible on")
    print("   any eval; KV quantization's is visible only on long-context ones.")



def exp_cost_ladder() -> None:
    print("\n7. THE COST LADDER -- 100 -> 42 -> 26 -> 11, reproduced from the mechanism")
    print("   The corpus's four figures are anchors [T]. This asks what configuration each implies.\n")

    prompt_len, output_len = 4000, 100
    b = C.solve_batch(0.42, prompt_len, output_len, params_b=PARAMS_B, machine_balance=BALANCE,
                      n_layers=LAYERS, n_kv_heads=KV_HEADS, head_dim=HEAD_DIM)
    h = C.solve_cache_hit(0.42, prompt_len, output_len, batch=int(b), params_b=PARAMS_B,
                          machine_balance=BALANCE, n_layers=LAYERS, n_kv_heads=KV_HEADS,
                          head_dim=HEAD_DIM)

    print(f"   workload: {prompt_len}-token prompt (e.g. a system prompt) -> {output_len}-token output\n")
    print(f"   rung                    model units   corpus [T]   delta   prefill share   implies")
    print("   " + "-" * 88)
    rows = C.ladder(prompt_len, output_len, params_b=PARAMS_B, machine_balance=BALANCE,
                    n_layers=LAYERS, n_kv_heads=KV_HEADS, head_dim=HEAD_DIM)
    resid = {r["rung"]: r for r in C.ladder_residuals(rows)}
    for r in rows:
        implies = ""
        if r["rung"] == "continuous_batching":
            implies = f"effective batch ~{b:.0f}"
        elif r["rung"] == "four_bit_weights":
            implies = "4-bit weights, bf16 KV, group 128"
        elif r["rung"] == "prefix_cached":
            implies = f"cache hit ~{h * 100:.0f}%"
        rr = resid.get(r["rung"], {})
        print(f"   {r['rung']:>22}  {r['units']:>11.1f}  {rr.get('corpus', 0):>10.0f}  "
              f"{rr.get('delta', 0):>+5.1f}  {r['prefill_share'] * 100:>13.0f}%  {implies}")

    print("\n   THE MODEL IS A MECHANISM, NOT A FIT, and the residual column shows how close it")
    print("   lands. The order and the shape are right; two rungs are within a few units. The 4-bit")
    print("   rung is the least accurate -- the model credits weight quantization a little more than")
    print("   the corpus's composite does -- and reporting that gap is the difference between")
    print("   reproducing the corpus and building something that merely resembles it.")
    print("\n   NOTE THE THIRD RUNG IS WEIGHTS-ONLY 4-bit, because that is what the corpus's")
    print("   toolchain line means -- 'AWQ and GPTQ to quantize' is weight quantization [T].")
    print("   Quantizing the KV cache as well is a lever the ladder does NOT include, and")
    print("   experiment 6 shows it is the larger one for concurrency.")
    print("\n   READ THE BATCHING RUNG'S IMPLIED BATCH: single digits. That is the most useful number")
    print("   in the table, because it says the corpus's headline 'you are near 42' is already")
    print("   mostly captured at a depth a modest deployment reaches -- the batching win is not")
    print("   something you have to reach a large batch to collect.")
    print("\n   AND THE FOUR-BIT RUNG IS THE SMALLEST SINGLE STEP -- 42 to 26 is a factor of 0.62,")
    print("   not the 0.25 the words 'four-bit' suggest. Two reasons, both real: (a) the weights")
    print("   are only part of what is read per token, since the KV term is untouched; and (b)")
    print("   group-wise 4-bit carries scale overhead, so the true ratio is 4.125/16, not 4/16.")

    print("\n   AND THE CACHING RUNG IS THE LARGEST -- but look at the prefill share column. That")
    print("   step is big ONLY because this workload's prompt is 40x its output. Same settings, an")
    print("   output-heavy workload, and the same cache hit rate buys almost nothing:")
    print(f"\n   {'prompt':>8}  {'output':>7}  {'ratio':>7}  {'prefill share':>14}  "
          f"{'cache step saves':>17}")
    print("   " + "-" * 60)
    for p_len, o_len in [(100, 500), (500, 500), (4000, 100), (16000, 100)]:
        no_cache = C.cost_per_request(p_len, o_len, 3, params_b=PARAMS_B, machine_balance=BALANCE,
                                      weight_bits=4, kv_bits=4, cache_hit=0.0,
                                      n_layers=LAYERS, n_kv_heads=KV_HEADS, head_dim=HEAD_DIM)
        cached = C.cost_per_request(p_len, o_len, 3, params_b=PARAMS_B, machine_balance=BALANCE,
                                    weight_bits=4, kv_bits=4, cache_hit=0.95,
                                    n_layers=LAYERS, n_kv_heads=KV_HEADS, head_dim=HEAD_DIM)
        save = 1.0 - cached["total"] / no_cache["total"]
        print(f"   {p_len:>8}  {o_len:>7}  {p_len / o_len:>7.1f}  "
              f"{no_cache['prefill_share'] * 100:>13.0f}%  {save * 100:>16.0f}%")

    print("\n   THE CORPUS'S OWN CAVEAT, quantified: 'caching only helps a stable prefix. Cache")
    print("   something that changes each call and you gain nothing' [T]. The table says exactly")
    print("   how much of the gain is available, and it is a function of the workload's")
    print("   prompt-to-output ratio, not of the cache implementation.")
    print("\n   WHICH IS THE TRANSFERABLE LESSON OF THE WHOLE LADDER: the four levers act on")
    print("   DIFFERENT TERMS of the same cost equation. Batching divides the weight term,")
    print("   quantization shrinks both terms, KV quantization shrinks only the KV term, and")
    print("   caching removes only the prefill term. Which one is worth doing is a property of")
    print("   the workload's shape -- and a ladder measured on one shape does not transfer to")
    print("   another.")


def exp_quality_gate() -> None:
    print("\n8. THE EVAL GATE -- turning 'always rerun your evals' into a merge check")
    print("   A degradation curve fitted to the corpus's own quality anchors [R], driven by the")
    print("   SNR this module MEASURES.\n")

    snrs = _snr_table()
    model = Q.fit_degradation(snrs=snrs)
    print(f"   measured SNR on a realistic tensor: "
          + ", ".join(f"{k} {v:.2f} dB" for k, v in snrs.items()))
    print(f"   fitted curve: delta_ppl% = {model['A']:.4f} * rel_err ** {model['B']:.3f}\n")
    print(f"   {'precision':>10}  {'SNR dB':>8}  {'corpus [R]':>11}  {'fitted':>9}  {'error':>8}")
    print("   " + "-" * 54)
    for label, anchor, pred in model["fit_on"]:
        err = (pred - anchor) / anchor * 100.0
        print(f"   {label:>10}  {snrs[label]:>8.2f}  {anchor:>10.1f}%  {pred:>8.2f}%  {err:>+7.1f}%")
    print(f"\n   worst residual: {model['max_residual_pct']:.1f}% of the anchor value")

    print("\n   THE FIT IS BAD, AND THAT IS THE FINDING -- not a defect to be tuned away.")
    print("   A power law in SNR cannot pass through the corpus's own three anchors, because the")
    print("   LOCAL SLOPES BETWEEN THEM ARE INCONSISTENT:")
    labels = [l for l, _, _ in model["fit_on"]]
    for a, b_ in zip(range(len(labels) - 1), range(1, len(labels))):
        la, lb = labels[a], labels[b_]
        ea, eb = Q.relative_error(snrs[la]), Q.relative_error(snrs[lb])
        ya = dict((l, v) for l, v, _ in model["fit_on"])[la]
        yb = dict((l, v) for l, v, _ in model["fit_on"])[lb]
        slope = (math.log(yb) - math.log(ya)) / (math.log(eb) - math.log(ea))
        print(f"     {la:>5} -> {lb:<5}: {yb / ya:>5.2f}x the quality loss for "
              f"{ea / eb:>6.2f}x the relative error   (implied exponent {slope:.2f})")
    print("   If quality loss were a smooth function of reconstruction error, those two implied")
    print("   exponents would be equal. They are not, and no single exponent reconciles them.")
    print("\n   WHY THIS MATTERS MORE THAN THE CURVE: it is the quantitative form of the corpus's")
    print("   own advice. 'Four-bit quantization is nearly free on many models, but on some it")
    print("   quietly drops accuracy. So always rerun your evals after quantizing' [T] -- because")
    print("   there is no formula that will tell you in advance, and this experiment is the")
    print("   demonstration that there cannot be a simple one.")

    print("\n   NOW USE THE CURVE ON THE THING THAT ACTUALLY VARIES -- the TENSOR, at fixed 4 bits.")
    print("   The gate's value depends on the QUANTIZER, and the two cases differ sharply:")
    print(f"\n   {'quantizer':>12}  {'severity':>9}  {'SNR dB':>8}  {'predicted':>10}")
    print("   " + "-" * 44)
    for qname, qfn in [("per-tensor", lambda w, bits: QZ.quantize_per_tensor(w, bits)),
                       ("group 32", lambda w, bits: QZ.quantize_groupwise(w, bits, 32))]:
        for sev in [1.0, 4.0, 12.0, 48.0]:
            r = Q.degradation_across_models(
                model, 4, [sev], qfn,
                lambda severity: QZ.synthetic_weights(outlier_frac=0.04, outlier_scale=severity))[0]
            print(f"   {qname:>12}  {sev:>9.1f}  {r['snr_db']:>8.2f}  "
                  f"{r['predicted_ppl_pct']:>9.2f}%")
    print("\n   READ: with group-wise quantization the prediction barely moves (severity is")
    print("   irrelevant, as experiment 3 showed), so the gate is nearly a formality. With a")
    print("   per-tensor scale it moves across the whole budget -- and the gate is doing real work.")
    print("   THAT IS THE RULE: the eval gate's value is a function of how much the quantizer's")
    print("   quality depends on the model. A quantizer that is flat across tensors does not need")
    print("   one; a quantizer whose quality is a property of the checkpoint does, and cannot")
    print("   substitute anything else for it.")

    print("\n   AND THE GATE IS ON THE WORST ROW, NOT THE AVERAGE:")
    rows = []
    for sev in [1.0, 2.0, 4.0, 8.0, 12.0, 24.0]:
        rows += Q.degradation_across_models(
            model, 4, [sev], lambda w, bits: QZ.quantize_per_tensor(w, bits),
            lambda severity: QZ.synthetic_weights(outlier_frac=0.04, outlier_scale=severity))
    mean_pred = sum(r["predicted_ppl_pct"] for r in rows) / len(rows)
    thr = 6.0
    gate = Q.eval_gate(rows, threshold_pct=thr)
    print(f"   per-tensor at 4 bits across severity: mean {mean_pred:.2f}%, "
          f"worst {gate['worst_ppl_pct']:.2f}%")
    print(f"   a threshold of {thr:.0f}% sits BETWEEN those two on purpose:")
    print(f"     gate on the MEAN  -> {mean_pred:.2f}% < {thr:.0f}%  -> would PASS")
    print(f"     gate on the WORST -> {gate['worst_ppl_pct']:.2f}% > {thr:.0f}%  -> "
          f"{gate['verdict']}, {len(gate['breached'])} of {len(rows)} breach")
    print("   Same numbers, opposite decisions. The mean is the number a summary reports; the")
    print("   worst is the number a user experiences. A gate on the mean ships this.")
    print("   THE MERGE RULE: quantize, run the evals, block the merge if the WORST slice")
    print("   regresses. 'It looked fine on the sample we tried' is the failure this replaces --")
    print("   the same tail-over-mean discipline as T08's goodput and T09's p99.")

