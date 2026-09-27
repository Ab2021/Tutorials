"""MoE kernel-path cost model — naive vs fused.

The corpus describes the naive MoE implementation as 2 all-to-all operations plus 6 kernels,
reduced to 3 kernels by fusing: top-k permute -> grouped GEMMs -> unpermute, with the
reduction and scaling folded in. [T]

This module models the *kernel launch and intermediate materialisation* cost of each path so
the difference is visible as a number. It does not model GPU time and does not measure
anything. [D]
"""

from __future__ import annotations

from dataclasses import dataclass

# Naive path, as described in the corpus [T]:
#   gate -> permute -> sort -> GEMM -> unpermute -> reduce -> scale  (plus 2 all-to-all)
NAIVE_KERNELS = [
    "gate_topk",
    "permute",
    "sort_by_expert",
    "grouped_gemm",
    "unpermute",
    "reduce_scale",
]

# Fused path: top-k permute -> grouped GEMMs -> unpermute, reduction and scale folded in. [T]
FUSED_KERNELS = [
    "topk_permute",
    "grouped_gemm",
    "unpermute_reduce_scale",
]

# Illustrative per-kernel overheads. These are MODEL parameters expressing "launch and
# materialisation are not free", NOT measurements. [D]
LAUNCH_US = 3.0            # fixed cost per kernel launch
MATERIALISE_US_PER_MB = 0.4  # cost of writing/reading an intermediate buffer


@dataclass
class KernelPathCost:
    name: str
    kernels: int
    a2a_ops: int
    intermediate_mb: float
    modelled_us: float


def estimate(name: str, kernels: list[str], intermediate_mb: float, a2a_ops: int) -> KernelPathCost:
    launches = len(kernels) * LAUNCH_US
    movement = intermediate_mb * MATERIALISE_US_PER_MB
    return KernelPathCost(
        name=name,
        kernels=len(kernels),
        a2a_ops=a2a_ops,
        intermediate_mb=intermediate_mb,
        modelled_us=launches + movement,
    )


def compare(tokens: int, hidden: int = 7168, dtype_bytes: int = 2, top_k: int = 8) -> list[KernelPathCost]:
    """Compare the two paths for a given number of tokens in flight.

    Intermediate buffer size is the permuted token tensor: tokens x top_k x hidden x dtype.
    The naive path materialises it more than once (permute, sort, unpermute each touch it);
    the fused path touches it once. [D]
    """
    permuted_mb = tokens * top_k * hidden * dtype_bytes / 1e6
    naive = estimate("naive (2 A2A + 6 kernels)", NAIVE_KERNELS, permuted_mb * 3.0, 2)
    fused = estimate("fused (2 A2A + 3 kernels)", FUSED_KERNELS, permuted_mb * 1.0, 2)
    return [naive, fused]


def render(rows: list[KernelPathCost]) -> str:
    w = 30
    lines = [
        f"{'path':<{w}} {'kernels':>8} {'A2A':>4} {'intermediate':>14} {'modelled':>12}",
        "-" * (w + 44),
    ]
    for r in rows:
        lines.append(
            f"{r.name:<{w}} {r.kernels:>8} {r.a2a_ops:>4} "
            f"{r.intermediate_mb:>11.1f} MB {r.modelled_us:>9.1f} us"
        )
    return "\n".join(lines)
