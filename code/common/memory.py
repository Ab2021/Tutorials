"""
common/memory.py — VRAM, FLOPs and cost calculators for LLM fine-tuning.

Everything in this file is arithmetic you can do on a napkin, implemented so you
never have to. Run it BEFORE you rent a GPU.

Usage
-----
    python common/memory.py --model 7B --method qlora --seq-len 2048 --batch 4
    python common/memory.py --model 70B --method lora --gpus 4
    python common/memory.py --model 8B --method full --optimizer adamw --gpus 8
    python common/memory.py --table          # print the full reference table

The formulas
------------
Training memory has four components:

    M_total = M_weights + M_gradients + M_optimizer + M_activations

  * M_weights      = P * bytes_per_param
  * M_gradients    = P * grad_bytes            (only for trainable params)
  * M_optimizer    = P_trainable * optim_bytes (Adam: 8 bytes/param fp32 m+v,
                                                plus 4 if master weights kept fp32)
  * M_activations  ≈ depends on batch, seq len, layers, hidden size, and whether
                     gradient checkpointing is on (divides by ~sqrt(L) in practice,
                     commonly modelled as a large constant reduction)

Inference memory:

    M_inference = M_weights + M_kv_cache + M_activations(small) + framework_overhead

  * M_kv_cache = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_element

The 6ND and 2N rules (N = params, D = tokens):
  * Training FLOPs ≈ 6 * N * D   (forward 2ND + backward 4ND)
  * Inference FLOPs ≈ 2 * N * D  (forward only, per generated token)

Author's note: these are *engineering estimates*, accurate to roughly ±25%. Real
usage is dominated by allocator fragmentation, framework overhead (CUDA context,
cuBLAS workspaces) and kernel choice. Always leave 10-20% headroom.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Model presets: (params_in_billions, n_layers, hidden_size, n_heads, n_kv_heads)
# kv_heads < heads means Grouped-Query Attention (GQA), which shrinks the KV cache.
# --------------------------------------------------------------------------------------
MODEL_PRESETS: dict[str, dict] = {
    "0.5B":  dict(params=0.5,  layers=24,  hidden=896,  heads=14, kv_heads=2),
    "1B":    dict(params=1.1,  layers=22,  hidden=2048, heads=32, kv_heads=4),
    "1.5B":  dict(params=1.5,  layers=28,  hidden=1536, heads=12, kv_heads=2),
    "2B":    dict(params=2.0,  layers=26,  hidden=2048, heads=16, kv_heads=4),
    "3B":    dict(params=3.0,  layers=28,  hidden=3072, heads=24, kv_heads=8),
    "7B":    dict(params=7.0,  layers=32,  hidden=4096, heads=32, kv_heads=32),
    "8B":    dict(params=8.0,  layers=32,  hidden=4096, heads=32, kv_heads=8),
    "13B":   dict(params=13.0, layers=40,  hidden=5120, heads=40, kv_heads=40),
    "14B":   dict(params=14.0, layers=48,  hidden=5120, heads=40, kv_heads=8),
    "32B":   dict(params=32.0, layers=64,  hidden=5120, heads=40, kv_heads=8),
    "70B":   dict(params=70.0, layers=80,  hidden=8192, heads=64, kv_heads=8),
    "405B":  dict(params=405.0, layers=126, hidden=16384, heads=128, kv_heads=8),
}

GB = 1024 ** 3


# --------------------------------------------------------------------------------------
# Training memory
# --------------------------------------------------------------------------------------
@dataclass
class TrainPlan:
    model: str
    method: str            # full | lora | qlora | qlora-8bit
    seq_len: int
    batch: int
    grad_accum: int = 1
    gpus: int = 1
    optimizer: str = "adamw"      # adamw | adamw_8bit | paged_adamw_8bit | adafactor | sgd
    grad_checkpointing: bool = True
    lora_r: int = 16
    lora_targets: str = "all"     # all | attention
    freeze_base: bool | None = None
    extra: dict = field(default_factory=dict)

    # -- derived -----------------------------------------------------------------
    @property
    def cfg(self) -> dict:
        if self.model not in MODEL_PRESETS:
            raise SystemExit(
                f"Unknown model preset '{self.model}'. "
                f"Choose from: {', '.join(MODEL_PRESETS)}"
            )
        return MODEL_PRESETS[self.model]

    @property
    def n_params(self) -> float:
        return self.cfg["params"] * 1e9

    @property
    def trainable_params(self) -> float:
        """
        Number of parameters that receive gradients.

        LoRA with r=16 on all linear projections is empirically ~0.5-2% of the base
        model for 7B-class models; attention-only targets are ~0.3-0.5%. These are
        measured typicals from the handbook, not derivations.
        """
        p = self.n_params
        if self.method == "full":
            return p
        # LoRA parameter count scales with r; r=16 is the baseline.
        scale = self.lora_r / 16.0
        if self.lora_targets == "all":
            frac = 0.02 * scale        # ~2% at r=16
        else:
            frac = 0.005 * scale       # ~0.5% at r=16
        return max(p * frac, 1e6)

    # -- the four components -----------------------------------------------------
    def weights_gb(self) -> float:
        """Bytes held by the frozen/trainable base weights."""
        p = self.n_params
        if self.method == "full":
            bpp = {  # full FT keeps fp16/bf16 weights + often an fp32 master copy
                "adamw": 4, "adamw_8bit": 2, "paged_adamw_8bit": 2,
                "adafactor": 2, "sgd": 2,
            }.get(self.optimizer, 2)
        elif self.method == "lora":
            bpp = 2                    # base stays bf16/fp16
        else:
            bpp = 0.5                  # 4-bit NF4
            if self.method == "qlora-8bit":
                bpp = 1.0              # 8-bit
        return p * bpp / GB

    def gradients_gb(self) -> float:
        if self.method == "full":
            return self.n_params * 2 / GB      # fp16 gradients
        if self.method == "lora":
            return self.trainable_params * 2 / GB
        return self.trainable_params * 2 / GB  # QLoRA: adapters only

    def optimizer_gb(self) -> float:
        """
        Adam keeps two fp32 moments (m, v) => 8 bytes/param for trainable params.
        8-bit optimizers quantize the moments to 1 byte each => ~2 bytes/param.
        `paged_adamw_8bit` adds the same math but moves state to CPU on pressure.
        """
        if self.optimizer == "sgd":
            return self.trainable_params * 4 / GB
        if self.optimizer == "adafactor":
            # Adafactor factorizes the second moment: ~1/rank of Adam's state.
            return self.trainable_params * 0.5 / GB
        if self.optimizer.endswith("8bit"):
            return self.trainable_params * 2 / GB
        return self.trainable_params * 8 / GB

    def activations_gb(self) -> float:
        """
        Activation memory. The exact value depends on every kernel in the graph, so
        we use a calibrated empirical model rather than a derivation.

        Without checkpointing, activations scale ~linearly with
        (batch * seq_len * hidden * layers). With checkpointing, only the layer
        boundaries are stored and the rest is recomputed, cutting this dramatically
        (commonly cited as ~sqrt(L) effective, we use a conservative 1/(3*sqrt(L))).
        """
        c = self.cfg
        base = (
            self.batch
            * self.seq_len
            * c["hidden"]
            * c["layers"]
            * 2                       # bytes per activation element (bf16)
        ) / GB
        base *= 12                    # empirical multiplier for the full graph
        if self.grad_checkpointing:
            base /= (3.0 * math.sqrt(c["layers"]))
        return base

    def total_gb(self) -> float:
        return (
            self.weights_gb()
            + self.gradients_gb()
            + self.optimizer_gb()
            + self.activations_gb()
        )

    def per_gpu_gb(self, sharded: bool = True) -> float:
        """
        With ZeRO-3 / FSDP, weights + gradients + optimizer are sharded across ranks;
        activations are *not* (they are per-GPU, reduced by sequence parallelism only).
        DDP (sharded=False) replicates weights/grads/optimizer everywhere.
        """
        if not sharded or self.gpus == 1:
            return self.total_gb()
        shardable = self.weights_gb() + self.gradients_gb() + self.optimizer_gb()
        return shardable / self.gpus + self.activations_gb()

    def effective_batch_tokens(self) -> int:
        return self.batch * self.grad_accum * self.seq_len * self.gpus


# --------------------------------------------------------------------------------------
# Inference memory
# --------------------------------------------------------------------------------------
def kv_cache_gb(model: str, seq_len: int, batch: int, bytes_per_element: float = 2.0) -> float:
    """
    KV cache = 2 (K and V) * layers * kv_heads * head_dim * seq * batch * bytes.

    head_dim is hidden/heads. Note we use kv_heads, not heads: with GQA (e.g. Llama-3
    8B has 32 heads but 8 kv_heads) the cache is 4x smaller. This is why modern models
    can serve long contexts at all.
    """
    c = MODEL_PRESETS[model]
    head_dim = c["hidden"] / c["heads"]
    return (2 * c["layers"] * c["kv_heads"] * head_dim
            * seq_len * batch * bytes_per_element) / GB


def inference_gb(model: str, seq_len: int = 4096, batch: int = 1,
                 weight_bytes_per_param: float = 2.0, framework_overhead_gb: float = 1.0) -> float:
    c = MODEL_PRESETS[model]
    weights = c["params"] * 1e9 * weight_bytes_per_param / GB
    return weights + kv_cache_gb(model, seq_len, batch, weight_bytes_per_param) + framework_overhead_gb


# --------------------------------------------------------------------------------------
# Compute and cost
# --------------------------------------------------------------------------------------
def training_flops_gb(model: str, tokens: int) -> float:
    """6ND rule: forward (2ND) + backward (4ND)."""
    return 6 * MODEL_PRESETS[model]["params"] * 1e9 * tokens


def hours_for_flops(flops: float, gpu_tflops: float, mfu: float = 0.35) -> float:
    """
    Wall-clock hours. MFU (Model FLOPs Utilization) is the fraction of the GPU's peak
    that you actually achieve. 0.35-0.45 is a realistic target for well-tuned LLM
    training; naive scripts hit 0.10-0.20. This is the single biggest lever on cost.
    """
    return flops / (gpu_tflops * 1e12 * mfu * 3600)


# Rough on-demand $/hour, early-2026 ballpark. Update before relying on it.
GPU_HOURLY_USD = {
    "T4": 0.35, "A10G": 1.00, "L4": 0.80, "L40S": 1.95,
    "A100-40": 1.80, "A100-80": 2.50, "H100": 3.50, "H200": 4.50, "B200": 6.50,
}

# Dense fp16/bf16 TFLOPs (with sparsity off). Used only for the MFU estimate.
GPU_TFLOPS = {
    "T4": 65, "A10G": 125, "L4": 121, "L40S": 362,
    "A100-40": 312, "A100-80": 312, "H100": 989, "H200": 989, "B200": 2250,
}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _print_plan(p: TrainPlan) -> None:
    w, g, o, a = p.weights_gb(), p.gradients_gb(), p.optimizer_gb(), p.activations_gb()
    total = p.total_gb()
    print(f"\n{'=' * 74}")
    print(f"  {p.model}  |  method={p.method}  |  optimizer={p.optimizer}  |  "
          f"seq={p.seq_len}  batch={p.batch} x accum {p.grad_accum}")
    print(f"{'=' * 74}")
    print(f"  trainable params      {p.trainable_params / 1e9:>10.3f} B "
          f"({100 * p.trainable_params / p.n_params:.2f}% of base)")
    print(f"  weights               {w:>10.2f} GB")
    print(f"  gradients             {g:>10.2f} GB")
    print(f"  optimizer state       {o:>10.2f} GB")
    print(f"  activations           {a:>10.2f} GB"
          f"{'   (gradient checkpointing ON)' if p.grad_checkpointing else '   (checkpointing OFF)'}")
    print(f"  {'-' * 60}")
    print(f"  TOTAL (1 GPU)         {total:>10.2f} GB")
    if p.gpus > 1:
        print(f"  TOTAL / GPU (FSDP)    {p.per_gpu_gb(sharded=True):>10.2f} GB  "
              f"({p.gpus} GPUs, ZeRO-3/FSDP)")
        print(f"  TOTAL / GPU (DDP)     {p.total_gb():>10.2f} GB  (no sharding — will OOM)")
    print(f"  effective batch       {p.effective_batch_tokens():>10,} tokens/step")
    print(f"{'-' * 74}")
    print(f"  {'FITS on 24 GB (4090/3090)':<40} {'YES' if p.per_gpu_gb() <= 22 else 'no'}")
    print(f"  {'FITS on 48 GB (A6000/L40S)':<40} {'YES' if p.per_gpu_gb() <= 44 else 'no'}")
    print(f"  {'FITS on 80 GB (A100/H100)':<40} {'YES' if p.per_gpu_gb() <= 74 else 'no'}")
    print(f"{'=' * 74}\n")

    if p.per_gpu_gb() > 80:
        print("  ⚠  Exceeds a single 80GB card. Options, in order of preference:")
        print("     1. Switch full FT -> LoRA/QLoRA   (biggest win, ~10-20x)")
        print("     2. Enable gradient checkpointing  (if not already on)")
        print("     3. Shorten max_seq_length")
        print("     4. Use 8-bit/paged optimizer")
        print("     5. Shard with FSDP/DeepSpeed ZeRO-3 across N GPUs\n")


def _table() -> None:
    print("\nTraining VRAM (GB), single GPU, gradient checkpointing ON, AdamW\n")
    hdr = f"{'Model':<8}{'Full FT':>10}{'LoRA r16':>11}{'QLoRA r16':>12}{'Infer bf16':>12}{'Infer 4bit':>12}"
    print(hdr)
    print("-" * len(hdr))
    for m in ["0.5B", "1B", "1.5B", "3B", "7B", "8B", "13B", "32B", "70B"]:
        f = TrainPlan(m, "full", 2048, 1).total_gb()
        l = TrainPlan(m, "lora", 2048, 1).total_gb()
        q = TrainPlan(m, "qlora", 2048, 1).total_gb()
        i16 = inference_gb(m, 4096, 1, 2.0)
        i4 = inference_gb(m, 4096, 1, 0.5)
        print(f"{m:<8}{f:>10.1f}{l:>11.1f}{q:>12.1f}{i16:>12.1f}{i4:>12.1f}")
    print("\nNote: 'full' assumes bf16 weights + fp32 Adam states for all params.")
    print("      Real usage adds 10-20% for allocator/framework overhead. Leave headroom.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="7B", choices=list(MODEL_PRESETS))
    ap.add_argument("--method", default="qlora", choices=["full", "lora", "qlora", "qlora-8bit"])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adamw_8bit", "paged_adamw_8bit", "adafactor", "sgd"])
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=0,
                    help="Total training tokens, to estimate time and cost.")
    ap.add_argument("--gpu-type", default="A100-80", choices=list(GPU_HOURLY_USD))
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.table:
        _table()
        return

    plan = TrainPlan(
        model=args.model, method=args.method, seq_len=args.seq_len, batch=args.batch,
        grad_accum=args.grad_accum, gpus=args.gpus, optimizer=args.optimizer,
        grad_checkpointing=not args.no_grad_checkpointing, lora_r=args.lora_r,
    )
    _print_plan(plan)

    if args.tokens:
        flops = training_flops_gb(args.model, args.tokens)
        hrs = hours_for_flops(flops, GPU_TFLOPS[args.gpu_type])
        cost = hrs * GPU_HOURLY_USD[args.gpu_type] * args.gpus
        print(f"  Compute estimate for {args.tokens:,} tokens on {args.gpus}x {args.gpu_type}:")
        print(f"    FLOPs          {flops:.3e}  ({flops / 1e21:.2f} ZFLOP)")
        print(f"    Wall clock     {hrs:.2f} h/GPU  @ 35% MFU")
        print(f"    Cost           ${cost:,.2f}")
        print(f"    (MFU is the dominant unknown: at 15% MFU this costs ${cost * 35 / 15:,.2f})\n")


if __name__ == "__main__":
    main()
