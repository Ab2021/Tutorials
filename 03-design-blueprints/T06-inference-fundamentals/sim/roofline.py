"""Hardware and model specs, FLOPs, bytes, arithmetic intensity, and the classification.

The one comparison that matters: `intensity` against `machine_balance`. Decode sits far below
that line; prefill sits above it. Every optimisation downstream follows from which side of the
line a workload falls on.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: float              # TOTAL parameters -- you must store every expert
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int            # GQA: may be << n_heads. Llama 3.1 keeps this at 8 at EVERY size.
    head_dim: int
    dtype_bytes: int = 2       # fp16
    active_params: float | None = None   # MoE only; decode FLOPs use this, weights use `params`

    @property
    def is_moe(self) -> bool:
        return self.active_params is not None

    @property
    def weights_bytes(self) -> float:
        return self.params * self.dtype_bytes

    @property
    def compute_params(self) -> float:
        """Parameters touched per token: active for MoE, total for dense."""
        return self.params if self.active_params is None else self.active_params

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes

    def check(self) -> list[str]:
        """Return a list of WARNINGS (not errors). Errors raise in the loader."""
        warns = []
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"{self.name}: n_heads % n_kv_heads != 0 -- GQA needs an integer ratio")
        if self.head_dim * self.n_heads != self.hidden:
            # Legitimate in real architectures; warn, never raise.
            warns.append(
                f"{self.name}: head_dim * n_heads = {self.head_dim * self.n_heads} != hidden "
                f"{self.hidden}. Legal, but check this is intended."
            )
        if self.active_params is not None and self.active_params > self.params:
            raise ValueError(f"{self.name}: active_params > params is impossible")
        return warns


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    peak_flops: float          # DENSE, at the working precision. NOT a sparse or FP4 peak.
    memory_bw: float           # bytes/s
    memory_capacity: float     # bytes

    @property
    def machine_balance(self) -> float:
        """FLOPs per byte. Above this, a kernel is compute-bound."""
        return self.peak_flops / self.memory_bw

    def assert_dense_peak(self, dense_peak: float) -> None:
        """Guard against the standard error of using a quoted sparse/low-precision peak.

        The dataclass cannot enforce this -- only the loader can -- so the guard lives here
        and fails LOUDLY rather than silently flipping a classification.
        """
        if self.peak_flops > dense_peak * 1.05:
            raise ValueError(
                f"{self.name}: peak_flops {self.peak_flops:.3e} exceeds the dense peak "
                f"{dense_peak:.3e}. Vendor sparse/low-precision peaks are 2-4x the number your "
                f"kernel achieves; using one moves machine_balance by an order of magnitude."
            )


@dataclass(frozen=True)
class TrafficSpec:
    prompt_tokens_p50: int
    prompt_tokens_p95: int     # P95 sizes the KV budget; the mean is a sanity check only
    output_tokens_p50: int
    output_tokens_p95: int
    qps: float


# --------------------------------------------------------------------------------------
# FLOPs, bytes, intensity
# --------------------------------------------------------------------------------------

def flops_per_token(model: ModelSpec, phase: str, n_prompt: int = 0) -> float:
    """The 2N rule: one multiply and one accumulate per parameter.

    Prefill:  2 * N * n_prompt
    Decode:   2 * N              (per generated token, per sequence)
    """
    N = model.compute_params
    if phase == "decode":
        return 2.0 * N
    if phase == "prefill":
        if n_prompt <= 0:
            raise ValueError("prefill requires n_prompt > 0")
        return 2.0 * N * n_prompt
    raise ValueError(f"unknown phase {phase!r}")


def bytes_moved(model: ModelSpec, phase: str, batch: int = 1, n_prompt: int = 0) -> float:
    """Weight traffic.

    The key asymmetry: DECODE READS THE WEIGHTS ONCE FOR THE WHOLE BATCH, so this does not
    grow with `batch`. That is why decode amortises so well and why continuous batching is
    the win the corpus says it is.
    """
    if batch <= 0:
        raise ValueError("batch must be > 0")
    return model.weights_bytes      # independent of batch, in both phases


def intensity(model: ModelSpec, phase: str, batch: int = 1, n_prompt: int = 0) -> float:
    return flops_per_token(model, phase, n_prompt) / bytes_moved(model, phase, batch, n_prompt)


def classify(model: ModelSpec, hw: HardwareSpec, phase: str, batch: int = 1,
             n_prompt: int = 0) -> tuple[str, float]:
    """(bound, ratio) where ratio = intensity / machine_balance.

    The MAGNITUDE matters as much as the side: a ratio of 0.001 and one of 0.9 both read
    'memory-bound' but call for completely different urgency.
    """
    ratio = intensity(model, phase, batch, n_prompt) / hw.machine_balance
    return ("compute-bound" if ratio >= 1.0 else "memory-bound"), ratio


def prefill_crossover(model: ModelSpec, hw: HardwareSpec) -> float:
    """Prompt length at which prefill crosses from memory-bound to compute-bound.

    Closed form: prefill intensity = 2 * n_prompt / dtype_bytes, so the crossing is at
        n_prompt = dtype_bytes * machine_balance / 2
    which at fp16 is just machine_balance.
    """
    return model.dtype_bytes * hw.machine_balance / 2.0


def attention_mlp_ratio(model: ModelSpec, n_ctx: int) -> float:
    """attention ~ n_ctx^2 * d  ;  MLP ~ n_ctx * d^2  ;  ratio = n_ctx / d.

    Above 1, attention dominates and the workload is a DIFFERENT engineering problem.
    """
    if n_ctx <= 0:
        raise ValueError("n_ctx must be > 0")
    return n_ctx / model.hidden


# --------------------------------------------------------------------------------------
# The corpus's Llama 3.1 dimensions [T] CMU lecture 1.
# --------------------------------------------------------------------------------------

LLAMA_31 = {
    "llama-3.1-8b": ModelSpec("llama-3.1-8b", 8e9, 32, 4096, 32, 8, 128),
    "llama-3.1-70b": ModelSpec("llama-3.1-70b", 70e9, 80, 8192, 64, 8, 128),
    "llama-3.1-405b": ModelSpec("llama-3.1-405b", 405e9, 126, 16384, 128, 8, 128),
}

# Hardware parameters are ILLUSTRATIVE [D] -- the corpus states no H100 peak. They are
# declared here so every conclusion is reproducible from them, and `assert_dense_peak`
# exists to stop a quoted sparse peak being substituted.
H100_CLASS = HardwareSpec("h100-class", peak_flops=989.5e12, memory_bw=3.35e12,
                          memory_capacity=80e9)
