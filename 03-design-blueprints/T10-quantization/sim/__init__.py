"""T10 -- quantization: precision, bytes, error, and what it costs.

Quantization is usually argued as one trade ("4 bits instead of 16, small quality loss"). It is
three separate things, and conflating them is why deployments are surprised:

  * an ACCOUNTING chain that has no choices in it -- bits are not bytes, the KV cache is quantized
    separately from the weights, and the two gains multiply rather than add      -> precision.py
  * the QUANTIZERS themselves, whose error depends far more on the tensor's dynamic range than on
    the bit width, which is why "4-bit is nearly free" is true of some models and false of others
                                                                                  -> quantize.py
  * a QUALITY model fitted to the corpus's own anchors, used to gate a merge on the WORST slice
    rather than the average                                                            -> quality.py
  * the COST LADDER (100 -> 42 -> 26 -> 11), reproduced from the mechanism so that each rung's
    implied configuration is visible                                              -> cost.py

The through-line, and the reason the four modules are separable: the four levers act on DIFFERENT
TERMS of one cost equation. Which lever is worth pulling is a property of the workload's shape.

Provenance inline: [T] corpus, [R] supporting repo, [D] derived. Stdlib only, no GPU.
"""
from .precision import (PRECISIONS, QUALITY_ANCHORS, bytes_per_weight, effective_bits, weight_bytes,
                        kv_bytes_per_token, max_concurrency, frontier)
from .quantize import (absmax_scale, quantize_symmetric, dequantize_symmetric, quantize_per_tensor,
                       quantize_per_channel, quantize_groupwise, nf4_levels, int4_levels,
                       quantize_nf4, salience, quantize_with_protected_channels,
                       protected_storage_overhead, mse, snr_db, worst_channel_error,
                       synthetic_weights, outlier_severity_sweep)
from .quality import (relative_error, fit_degradation, predicted_degradation,
                      degradation_across_models, eval_gate)
from .cost import (CORPUS_LADDER, prefill_cost_per_token, cost_per_request, ladder, solve_batch,
                   solve_cache_hit, cost_per_million_tokens)
from . import experiments

__all__ = [
    "PRECISIONS", "QUALITY_ANCHORS", "bytes_per_weight", "effective_bits", "weight_bytes",
    "kv_bytes_per_token", "max_concurrency", "frontier",
    "absmax_scale", "quantize_symmetric", "dequantize_symmetric", "quantize_per_tensor",
    "quantize_per_channel", "quantize_groupwise", "nf4_levels", "int4_levels",
    "quantize_nf4", "salience", "quantize_with_protected_channels",
    "protected_storage_overhead", "mse", "snr_db", "worst_channel_error",
    "synthetic_weights", "outlier_severity_sweep",
    "relative_error", "fit_degradation", "predicted_degradation",
    "degradation_across_models", "eval_gate",
    "CORPUS_LADDER", "prefill_cost_per_token", "cost_per_request", "ladder", "solve_batch",
    "solve_cache_hit", "cost_per_million_tokens",
    "experiments",
]
