"""Precision -> bytes -> concurrency. The only chain in this topic that has no choices in it.

Quantization is usually argued in the abstract ("4-bit is 4x smaller"). This module keeps the
accounting honest in three ways that the abstract argument gets wrong:

  1. **Bits are not bytes.** A 4-bit weight with a per-group fp16 scale costs
     `4 + 16/group_size` bits, not 4. At group_size 128 that is 4.125 -- small, but it is why a
     4-bit model is not exactly one quarter of a bf16 one.
  2. **The KV cache is quantized separately from the weights**, and in long-context serving it is
     the LARGER of the two. Quantizing weights alone buys far less concurrency than the headline
     suggests; the corpus's own claim is 4x, and it takes BOTH.
  3. **The concurrency ceiling is a division, so the gains compound.** Weight savings and KV
     savings do not add -- they multiply, because one shrinks the subtrahend and the other shrinks
     the denominator.

Provenance: [T] transcript, [R] supporting repo, [D] derived.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# The precision table
# --------------------------------------------------------------------------------------

# bytes here are STORAGE bytes for the weight tensor alone. The corpus's own table gives the
# resulting model size for an 8B model [R] (03-training-and-adaptation/07-quantization-deep-dive.md),
# which is what `weight_bytes` reproduces.
PRECISIONS: dict[str, dict] = {
    "fp32": {"bits": 32, "kind": "float", "native": False, "note": "training reference only"},
    "bf16": {"bits": 16, "kind": "float", "native": True,  "note": "serving default"},
    "fp16": {"bits": 16, "kind": "float", "native": True,  "note": "numerically wider mantissa"},
    "fp8":  {"bits": 8,  "kind": "float", "native": True,  "note": "H100/B200/4090 native [R]"},
    "int8": {"bits": 8,  "kind": "int",   "native": True,  "note": "symmetric, per-channel"},
    "int4": {"bits": 4,  "kind": "int",   "native": False, "note": "needs a scale; group-wise"},
    "nf4":  {"bits": 4,  "kind": "codebook", "native": False, "note": "equal-mass normal bins [R]"},
    "int2": {"bits": 2,  "kind": "int",   "native": False, "note": "research/specialised [R]"},
}

# The corpus's quality-loss column [R]. These are the ANCHORS the quality model in `quality.py`
# is fitted to, and their provenance matters: they are the guide's figures, not measurements.
QUALITY_ANCHORS: dict[str, float] = {"bf16": 0.0, "fp8": 0.5, "int4": 1.5, "int2": 12.5}


def bytes_per_weight(bits: int) -> float:
    return bits / 8.0


def effective_bits(bits: int, group_size: int | None = None,
                   scale_bits: int = 16, zero_point_bits: int = 0) -> float:
    """Bits per weight INCLUDING the quantization metadata. THE HONEST NUMBER.

    A group-wise 4-bit tensor stores one scale (and optionally one zero point) per group. At
    group_size 128 with an fp16 scale that is `4 + 16/128 = 4.125` bits per weight. At group_size
    32 it is `4 + 0.5 = 4.5`, i.e. a 12.5% storage penalty over the nominal 4.

    Two consequences worth internalising:
      * smaller groups quantize BETTER (more scales, less error) and cost MORE storage -- this is
        the only real trade in the method;
      * a per-TENSOR scale has zero metadata overhead and much worse error, which is why nobody
        uses it for weights (see `quantize.py`).
    """
    if group_size is None:
        return float(bits)
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    return bits + (scale_bits + zero_point_bits) / group_size


def weight_bytes(params_b: float, bits: int, group_size: int | None = None,
                 scale_bits: int = 16) -> float:
    """Weight storage in bytes for a model of this size at this precision."""
    if params_b <= 0:
        raise ValueError("params_b must be > 0")
    return params_b * 1e9 * effective_bits(bits, group_size, scale_bits) / 8.0


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, bits: int,
                       group_size: int | None = None, scale_bits: int = 16) -> float:
    """KV storage per token, K and V both.

        bytes = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_element

    GROUPED-QUERY ATTENTION IS ALREADY A QUANTIZATION. `n_kv_heads < n_heads` divides the KV size
    before any precision change, and the corpus lists both under the same heading [R]. GQA and
    KV quantization multiply, and they are the two cheapest levers on the largest term in
    long-context serving.
    """
    if min(n_layers, n_kv_heads, head_dim) <= 0:
        raise ValueError("dimensions must be positive")
    per_element = effective_bits(bits, group_size, scale_bits) / 8.0
    return 2.0 * n_layers * n_kv_heads * head_dim * per_element


def max_concurrency(hbm_gb: float, params_b: float, weight_bits: int, kv_bits: int,
                    ctx_len: int, n_layers: int = 32, n_kv_heads: int = 8, head_dim: int = 128,
                    util: float = 0.90, weight_group: int | None = None,
                    kv_group: int | None = None) -> dict:
    """How many sequences fit. THE NUMBER THAT MATTERS, and the reason to quantize at all.

    `(usable_hbm - weights) / (kv_per_token * ctx_len)`

    The numerator is why weight quantization helps; the denominator is why KV quantization helps.
    **They multiply, and that is why the corpus's 4x concurrency figure needs BOTH** -- weight
    quantization alone moves only the subtrahend.

    Returns a dict rather than a float so the caller cannot accidentally compare a weight-only
    figure with a combined one: `weights_gb`, `kv_gb_per_seq`, `usable_gb` and `sequences` all
    travel together.
    """
    if ctx_len <= 0:
        raise ValueError("ctx_len must be > 0")
    if not 0.0 < util <= 1.0:
        raise ValueError("util must be in (0, 1]")
    usable = hbm_gb * util
    w_gb = weight_bytes(params_b, weight_bits, weight_group) / 1e9
    kv_per_seq = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kv_bits, kv_group) * ctx_len / 1e9
    free = usable - w_gb
    if free <= 0:
        return {"usable_gb": usable, "weights_gb": w_gb, "kv_gb_per_seq": kv_per_seq,
                "sequences": 0.0, "fits": False,
                "note": "model does not fit -- weights alone exceed usable HBM"}
    return {"usable_gb": usable, "weights_gb": w_gb, "kv_gb_per_seq": kv_per_seq,
            "sequences": free / kv_per_seq, "fits": True, "note": ""}


def frontier(hbm_gb: float, params_b: float, weight_bits: int, kv_bits: int,
             ctx_lengths: list[int], **kw) -> list[dict]:
    """Concurrency across context lengths, with the per-sequence KV measured against the weights.

    A DEGENERATE METRIC TO AVOID: total resident KV is always exactly the free budget, because
    concurrency absorbs whatever is left. `kv_per_sequence x sequences == usable - weights` by
    construction, so "what fraction of memory is KV" is 100% at every context and says nothing.
    The informative quantity is the PER-SEQUENCE comparison -- does one sequence's KV exceed the
    weights? -- and the marginal cost of context:

        one more 1k tokens of context costs  0.131 GB per concurrent sequence

    which is the number that makes long context expensive, and the reason raising concurrency and
    raising context are the same budget spent two ways.
    """
    rows = []
    for ctx in ctx_lengths:
        r = max_concurrency(hbm_gb, params_b, weight_bits, kv_bits, ctx, **kw)
        r["ctx_len"] = ctx
        r["kv_vs_weights"] = (r["kv_gb_per_seq"] / r["weights_gb"]) if r["weights_gb"] else 0.0
        r["kv_overtakes_weights"] = r["kv_gb_per_seq"] > r["weights_gb"]
        rows.append(r)
    return rows
