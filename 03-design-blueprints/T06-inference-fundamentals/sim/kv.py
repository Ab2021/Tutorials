"""KV cache arithmetic -- the number that constrains everything else.

bytes/token = 2 (K and V) * n_layers * n_kv_heads * head_dim * dtype_bytes

The design point: `n_kv_heads` is why a 405B model is servable at all. Llama 3.1 keeps it at 8
at EVERY size [T], so KV cost does not scale with parameter count the way weights do. A model
without that property would be unservable at long context.
"""
from __future__ import annotations

from .roofline import ModelSpec, HardwareSpec

GB = 1e9


def kv_bytes_per_token(model: ModelSpec, dtype_bytes: int | None = None) -> float:
    d = model.dtype_bytes if dtype_bytes is None else dtype_bytes
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * d


def kv_total(model: ModelSpec, n_ctx: int, concurrency: int = 1,
             dtype_bytes: int | None = None) -> float:
    if n_ctx <= 0:
        raise ValueError("n_ctx must be > 0")
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    return kv_bytes_per_token(model, dtype_bytes) * n_ctx * concurrency


def weights_bytes(model: ModelSpec, dtype_bytes: int | None = None) -> float:
    d = model.dtype_bytes if dtype_bytes is None else dtype_bytes
    return model.params * d


def weight_replicas_needed(model: ModelSpec, hw: HardwareSpec) -> int:
    """How many parts the weights alone span. Usually > 1 for the big models."""
    return max(1, -(-int(weights_bytes(model)) // int(hw.memory_capacity)))


def free_after_weights(model: ModelSpec, hw: HardwareSpec,
                       weights_per_gpu: float | None = None) -> float:
    """Bytes free on ONE part after its share of the weights. May be <= 0."""
    w = weights_per_gpu if weights_per_gpu is not None else 0.0
    return hw.memory_capacity - w


def max_concurrency(model: ModelSpec, hw: HardwareSpec, n_ctx: int,
                    weights_per_gpu: float | None = None,
                    headroom_frac: float = 0.10) -> int:
    """KV-bound concurrency on one part.

    `headroom_frac` is EXPLICIT and defaults to 10%: activation memory, fragmentation and
    the CUDA context are real and are not modelled. Making it a parameter keeps the omission
    visible instead of pretending the model is exact.
    """
    per_seq = kv_total(model, n_ctx)
    usable = free_after_weights(model, hw, weights_per_gpu) * (1.0 - headroom_frac)
    if usable <= 0:
        return 0
    return int(usable // per_seq)


def gqa_saving(n_heads: int, n_kv_heads: int) -> float:
    """KV cost ratio between full multi-head attention and GQA. 128/8 = 16x for Llama 3.1."""
    if n_kv_heads <= 0:
        raise ValueError("n_kv_heads must be > 0")
    return n_heads / n_kv_heads


def kv_dtype_effect(model: ModelSpec, ctx: int = 128000) -> list[dict]:
    """Weight quantization and KV quantization are DIFFERENT changes with DIFFERENT effects.

    Weight quantisation moves decode's bytes_moved and therefore its time -- but NOT its
    classification, which was and remains memory-bound. KV quantisation changes the KV
    budget and therefore concurrency. Conflating the two is the standard error.
    """
    rows = []
    for label, d in (("fp16", 2), ("int8", 1), ("int4", 0.5)):   # bytes per KV element
        per_tok = kv_bytes_per_token(model, d)
        rows.append({
            "dtype": label,
            "kv_bytes_per_token": per_tok,
            "kv_total_GB": kv_total(model, ctx, dtype_bytes=d) / GB,
        })
    return rows
