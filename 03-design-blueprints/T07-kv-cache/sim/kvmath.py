"""KV arithmetic: bytes per token, per-sequence totals, and the concurrency ceiling.

Deliberately self-contained -- a blueprint must run without importing a sibling blueprint. The
formulas are the same as T06's by construction; T06 *classifies* a workload, this module
*plans capacity*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

GB = 1e9


@dataclass(frozen=True)
class KvSpec:
    """The four architectural numbers that determine KV cost."""
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_heads: int = 0          # for the GQA ratio only
    dtype_bytes: int = 2
    max_ctx: int = 131072

    @property
    def bytes_per_token(self) -> float:
        """2 (K and V) x n_layers x n_kv_heads x head_dim x dtype_bytes."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def gqa_ratio(self) -> float:
        return self.n_heads / self.n_kv_heads if self.n_kv_heads else 0.0

    def total(self, n_ctx: int, concurrency: int = 1) -> float:
        return self.bytes_per_token * n_ctx * concurrency


# [T] dimensions (CMU lecture 1). n_kv_heads = 8 at EVERY size -- the GQA fact.
MODELS = {
    "llama-3.1-8b":   KvSpec("llama-3.1-8b", n_layers=32,  n_kv_heads=8, head_dim=128, n_heads=32),
    "llama-3.1-70b":  KvSpec("llama-3.1-70b", n_layers=80, n_kv_heads=8, head_dim=128, n_heads=64),
    "llama-3.1-405b": KvSpec("llama-3.1-405b", n_layers=126, n_kv_heads=8, head_dim=128, n_heads=128),
}


def kv_table(spec: KvSpec, contexts: Iterable[int]) -> list[dict]:
    rows = []
    for c in contexts:
        if c > spec.max_ctx:
            raise ValueError(f"n_ctx {c} exceeds trained window {spec.max_ctx}")
        rows.append({"n_ctx": c, "GB": spec.total(c) / GB})
    return rows


def max_concurrency(spec: KvSpec, hbm_for_kv: float, avg_ctx: int,
                    block_size: int = 16) -> dict:
    """The capacity ceiling -- the number that actually binds in most deployments.

    `hbm_for_kv` is HBM after weights and activations, NOT total HBM. Passing total HBM is the
    most common capacity-planning error and it over-estimates concurrency by the weight
    footprint, which for a 405B at fp16 is an infinite error (the weights do not fit at all).

    Returns the raw division AND the block-granular truth, because the two differ: you cannot
    run a fraction of a sequence, and each sequence rounds UP to whole blocks.
    """
    if avg_ctx <= 0:
        raise ValueError("avg_ctx must be > 0")
    if hbm_for_kv <= 0:
        raise ValueError("hbm_for_kv must be > 0 -- did you subtract the weights?")
    # The corpus's form: HBM_for_KV / (bytes_per_token x avg_ctx)  [T]
    raw = hbm_for_kv / (spec.bytes_per_token * avg_ctx)
    seq_blocks = math.ceil(avg_ctx / block_size)
    blocks_available = int(hbm_for_kv // (spec.bytes_per_token * block_size))
    return {"avg_ctx": avg_ctx, "bytes_per_token": spec.bytes_per_token,
            "bytes_per_sequence": spec.bytes_per_token * avg_ctx,
            "raw_concurrency": raw, "seq_blocks": seq_blocks,
            "block_granular_concurrency": blocks_available // seq_blocks,
            "bound_by": "capacity"}


def kv_quant_effect(spec: KvSpec, n_ctx: int, dtypes=(2, 1, 0.5)) -> list[dict]:
    """fp16 -> fp8 -> int4 KV. Halves the ceiling each step; accuracy is T10's question.

    `dtypes` is bytes per element: 2 = fp16, 1 = fp8, 0.5 = int4. The multiplier column is the
    whole point -- it is the number that changes, and it is the number weight quantization does
    NOT change.
    """
    out = []
    for b in dtypes:
        out.append({"dtype_bytes": b, "bytes_per_token": spec.bytes_per_token * b / 2,
                    "total_GB": spec.total(n_ctx) * b / 2 / GB,
                    "concurrency_multiplier": 2 / b})
    return out


def offload_headroom(spec: KvSpec, hbm_for_kv: float, host_dram: float, avg_ctx: int) -> dict:
    """How many additional sequences a host-DRAM tier can hold.

    The corpus's rack-scale demo pools memory across four servers with one pooled box mid-rack
    [T]. The point of the arithmetic is the order of magnitude: DRAM holds an order more KV
    than HBM, and SSD another order again. Which tier a session belongs in is decided by its
    re-arrival time (see tiering.RetentionPolicy).
    """
    per_seq = spec.total(avg_ctx)
    return {"hbm_sequences": int(hbm_for_kv // per_seq),
            "dram_sequences": int(host_dram // per_seq),
            "dram_multiplier": host_dram / hbm_for_kv if hbm_for_kv else 0.0}


def recompute_cliff(spec: KvSpec, crossover_tokens: int, n_ctx: int) -> dict:
    """The corpus reports a recomputation cliff at 28k input tokens [T] ROCm/WideEP.

    Past that length, re-prefilling costs more than the memory the KV was occupying is worth,
    so an engine should stop dropping and start tiering. The number is corpus-reported and
    hardware-specific; this function makes the comparison explicit rather than hard-coding the
    threshold as if it were universal.
    """
    return {"crossover_tokens": crossover_tokens, "n_ctx": n_ctx,
            "past_cliff": n_ctx >= crossover_tokens,
            "kv_at_n_ctx_GB": spec.total(n_ctx) / GB,
            "guidance": ("tier, do not drop" if n_ctx >= crossover_tokens
                         else "dropping and re-prefilling is acceptable")}
