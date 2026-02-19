"""
============================================================================
CONFIG.PY — Centralized Configuration for Gemma Fine-Tuning
============================================================================

PURPOSE:
    This file contains ALL hyperparameters, paths, and settings in one place.
    Instead of scattering magic numbers across files, everything lives here.
    This makes it easy to experiment with different settings.

HOW TO USE:
    from config import TrainingConfig
    cfg = TrainingConfig()
    print(cfg.model_name)  # "google/gemma-2b"

HOW TO CUSTOMIZE:
    Option 1: Edit the defaults below directly
    Option 2: Override at runtime:
        cfg = TrainingConfig(learning_rate=1e-5, num_epochs=5)
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class TrainingConfig:
    """
    Every hyperparameter for the fine-tuning pipeline, with detailed docs.
    Uses Python dataclass for clean, type-checked configuration.
    """

    # ======================================================================
    # MODEL CONFIGURATION
    # ======================================================================

    model_name: str = "google/gemma-2b"
    """
    Which Gemma model to fine-tune. Options:
      • "google/gemma-2b"   — 2 billion params, ~6 GB VRAM with QLoRA (RECOMMENDED)
      • "google/gemma-7b"   — 7 billion params, ~16 GB VRAM with QLoRA
      • "google/gemma-2b-it" — 2B instruction-tuned (already fine-tuned on instructions)
      • "google/gemma-7b-it" — 7B instruction-tuned

    For beginners: Start with "google/gemma-2b". It trains faster and needs
    less memory. You can always scale up after confirming your pipeline works.
    """

    model_revision: str = "main"
    """
    Git revision / branch of the model on Hugging Face Hub.
    "main" = latest stable release. You can pin to a specific commit hash
    for reproducibility, e.g. "a1b2c3d".
    """

    trust_remote_code: bool = False
    """
    Whether to allow running custom code from the model repository.
    Gemma doesn't need this (it's natively supported in Transformers),
    so we keep it False for security.
    """

    # ======================================================================
    # QUANTIZATION CONFIGURATION (QLoRA / BitsAndBytes)
    # ======================================================================

    use_4bit: bool = True
    """
    Enable 4-bit quantization via bitsandbytes.
    
    WHAT IT DOES:
      Normal models store each weight as a 16-bit float (2 bytes).
      4-bit quantization compresses each weight to just 4 bits (0.5 bytes).
      This cuts memory by ~4x with minimal quality loss.
    
    WHEN TO USE:
      • True  — You have limited VRAM (< 24 GB). RECOMMENDED for most users.
      • False — You have 40+ GB VRAM and want maximum quality.
    """

    bnb_4bit_compute_dtype: str = "bfloat16"
    """
    The dtype used for COMPUTATION during forward/backward pass.
    Even though weights are stored in 4-bit, computations happen in this dtype.
    
    Options:
      • "bfloat16" — Best for Ampere+ GPUs (RTX 3000/4000, A100).
                      More numerically stable than float16. RECOMMENDED.
      • "float16"  — Use if your GPU doesn't support bfloat16 (older than Ampere).
      • "float32"  — Maximum precision but 2x slower and 2x more memory.
    
    HOW TO CHECK YOUR GPU:
      Run: python -c "import torch; print(torch.cuda.get_device_capability())"
      If the result is (8, 0) or higher → use bfloat16
      If the result is (7, x) or lower  → use float16
    """

    bnb_4bit_quant_type: str = "nf4"
    """
    The quantization algorithm for compressing weights.
    
    Options:
      • "nf4"  — Normal Float 4-bit. Specifically designed for neural network
                  weights which follow a normal distribution. RECOMMENDED.
      • "fp4"  — Standard 4-bit float. Slightly faster to quantize but
                  lower quality than nf4.
    
    Technical detail: nf4 creates quantization bins that match the expected
    distribution of neural network weights, giving better accuracy.
    """

    use_double_quant: bool = True
    """
    Double quantization — quantizes the quantization constants themselves.
    
    WHAT IT DOES:
      In 4-bit quantization, each group of weights has a scaling constant
      (stored in fp16 = 2 bytes). Double quantization compresses these
      constants too, saving an additional ~0.4 GB for a 2B model.
    
    RECOMMENDATION: Always True. Free memory savings with no quality loss.
    """

    # ======================================================================
    # LoRA (Low-Rank Adaptation) CONFIGURATION
    # ======================================================================

    lora_r: int = 16
    """
    LoRA rank — the "size" of the adapter matrices.
    
    WHAT IT DOES:
      LoRA decomposes weight updates into two small matrices of rank r.
      Original weight W (d×d) → W + A(d×r) × B(r×d)
      
    IMPACT:
      • Higher r = more trainable parameters = better learning capacity
      • Lower r  = fewer parameters = faster training, less overfitting
    
    Guidelines:
      • r=8   — Minimum useful rank. Good for very small datasets (<1K samples)
      • r=16  — Sweet spot for most tasks. RECOMMENDED.
      • r=32  — Use if you have a large dataset (>50K samples) and see underfitting
      • r=64  — Rarely needed. Risk of overfitting on small datasets.
    
    Trainable params at r=16: ~2.6M out of 2B total (0.13%)
    """

    lora_alpha: int = 32
    """
    LoRA scaling factor. Controls how much the adapter affects the output.
    
    FORMULA: output = base_output + (lora_alpha / lora_r) × adapter_output
    
    RULE OF THUMB: Set lora_alpha = 2 × lora_r.
    This gives a scaling factor of 2.0, which works well empirically.
    
    If you increase lora_r, increase lora_alpha proportionally.
    """

    lora_dropout: float = 0.05
    """
    Dropout rate applied to LoRA adapter layers during training.
    
    WHAT IT DOES:
      Randomly zeroes out 5% of adapter values during each forward pass.
      This prevents overfitting by forcing the model to not rely too
      heavily on any single adapter neuron.
    
    Guidelines:
      • 0.0   — No dropout. Use if you have a very large dataset.
      • 0.05  — Light dropout. Good default. RECOMMENDED.
      • 0.1   — Moderate dropout. Use if you see overfitting (train loss
                 decreasing but val loss increasing).
      • 0.2+  — Heavy dropout. Usually too aggressive for LoRA.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    """
    Which layers to attach LoRA adapters to.
    
    Gemma's transformer has these linear layers in each attention block:
      • q_proj — Query projection (how the model "asks questions" about each token)
      • k_proj — Key projection (how each token "advertises" its content)
      • v_proj — Value projection (what information each token provides)
      • o_proj — Output projection (combines attention results)
    
    Options:
      • ["q_proj", "v_proj"]              — Minimum. Fastest but less expressive.
      • ["q_proj", "k_proj", "v_proj", "o_proj"] — RECOMMENDED. Good balance.
      • Add "gate_proj", "up_proj", "down_proj"  — Also adapts the MLP layers.
                                                     More powerful but slower.
    
    More modules = more trainable parameters = more VRAM usage.
    """

    # ======================================================================
    # TRAINING HYPERPARAMETERS
    # ======================================================================

    num_epochs: int = 3
    """
    Number of complete passes through the training dataset.
    
    Guidelines:
      • 1 epoch  — Quick test run or very large dataset (>100K samples)
      • 3 epochs — Standard for most fine-tuning tasks. RECOMMENDED.
      • 5 epochs — If model is underfitting (training loss not converging)
      • 10+ epochs — Rarely needed. High risk of overfitting.
    
    WATCH FOR: If validation loss starts increasing while training loss
    keeps decreasing, you're overfitting → reduce epochs or increase dropout.
    """

    per_device_train_batch_size: int = 4
    """
    Number of training examples processed per GPU per step.
    
    IMPACT ON MEMORY:
      Each example occupies ~(max_seq_length × hidden_size × bytes_per_param).
      Larger batch size = more VRAM usage.
    
    Guidelines:
      • 1  — Minimum. Use if you're running out of VRAM.
      • 4  — Good default for 8-16 GB VRAM. RECOMMENDED.
      • 8  — Use if you have 24+ GB VRAM.
      • 16 — Use if you have 40+ GB VRAM (A100 / A6000).
    
    If you get CUDA Out-of-Memory (OOM), reduce this FIRST.
    """

    per_device_eval_batch_size: int = 4
    """
    Batch size for validation/evaluation. Can be larger than training
    batch size because we don't need to store gradients during evaluation.
    """

    gradient_accumulation_steps: int = 4
    """
    Simulates a larger batch size without using more VRAM.
    
    EFFECTIVE BATCH SIZE = per_device_train_batch_size × gradient_accumulation_steps
    With defaults: 4 × 4 = 16 effective batch size.
    
    HOW IT WORKS:
      Instead of updating weights after every batch, it accumulates gradients
      over N batches and then does one big update. Mathematically equivalent
      to training with a larger batch size.
    
    WHY USE IT:
      • You want batch_size=16 but only have VRAM for batch_size=4
      • Larger effective batch sizes give more stable gradient estimates
    """

    learning_rate: float = 2e-4
    """
    How big of a step to take when updating weights.
    
    THE MOST IMPORTANT HYPERPARAMETER.
    
    Guidelines for LoRA fine-tuning:
      • 1e-5 — Very conservative. Use if you see instability (loss spikes).
      • 5e-5 — Conservative. Good for small datasets.
      • 2e-4 — Standard for LoRA fine-tuning. RECOMMENDED.
      • 5e-4 — Aggressive. May cause instability.
      • 1e-3 — Too high for most cases. Will likely diverge.
    
    NOTE: LoRA fine-tuning typically uses higher learning rates than
    full fine-tuning because we're only updating a tiny fraction of params.
    """

    weight_decay: float = 0.01
    """
    L2 regularization strength. Penalizes large weight values.
    
    WHAT IT DOES:
      Adds a penalty term: loss = original_loss + weight_decay × sum(weights²)
      This encourages smaller weights, which can reduce overfitting.
    
    Guidelines:
      • 0.0   — No regularization.
      • 0.01  — Light. Good default. RECOMMENDED.
      • 0.1   — Strong. Use if severely overfitting.
    """

    warmup_ratio: float = 0.03
    """
    Fraction of total training steps used for learning rate warmup.
    
    WHAT IT DOES:
      Instead of starting at the full learning_rate immediately (which can
      cause instability), the LR linearly increases from 0 to learning_rate
      over the first 3% of training steps.
    
    WHY:
      The model's gradients are very noisy at the start because the randomly
      initialized LoRA adapters produce garbage outputs. Warmup prevents
      these noisy gradients from making destructive updates.
    
    Guidelines:
      • 0.03 — Standard. RECOMMENDED for most cases.
      • 0.1  — More warmup. Use if training is unstable at the start.
    """

    lr_scheduler_type: str = "cosine"
    """
    How the learning rate changes over training.
    
    Options:
      • "linear"  — Linearly decreases from max to 0. Simple but effective.
      • "cosine"  — Follows a cosine curve. Decreases slowly at first, then
                     faster. Empirically gives slightly better results. RECOMMENDED.
      • "constant" — Never changes. Not recommended for fine-tuning.
      • "cosine_with_restarts" — Cosine with periodic resets. For very long training.
    """

    max_grad_norm: float = 1.0
    """
    Gradient clipping threshold. If the gradient norm exceeds this value,
    it's scaled down to this maximum.
    
    WHY:
      Prevents "exploding gradients" where one bad batch causes enormously
      large weight updates that destabilize training.
    
    Guidelines:
      • 0.3 — Very aggressive clipping. Use if training is very unstable.
      • 1.0 — Standard. RECOMMENDED.
      • 5.0 — Light clipping.
    """

    # ======================================================================
    # DATASET CONFIGURATION
    # ======================================================================

    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    """
    Hugging Face Hub dataset identifier.
    
    Amazon Reviews 2023 contains millions of product reviews with:
      • review text, rating (1-5), product title, category
      • Perfect for training recommendation models
    
    Alternative datasets you could try:
      • "amazon_reviews_multi" — Multilingual Amazon reviews
      • "yelp_review_full"    — Yelp business reviews
      • "imdb"                — Movie reviews (simpler task)
    """

    dataset_config: str = "raw_review_All_Beauty"
    """
    Which subset/configuration of the dataset to use.
    
    Amazon Reviews 2023 has many product categories:
      • "raw_review_All_Beauty"        — Beauty products (~370K reviews)
      • "raw_review_Electronics"       — Electronics (~20M reviews, very large)
      • "raw_review_Books"             — Books (~25M reviews)
      • "raw_review_Clothing_Shoes_and_Jewelry" — Fashion
    
    START SMALL: Use "raw_review_All_Beauty" for initial experiments.
    It's small enough to train quickly but large enough to learn from.
    """

    max_train_samples: int = 5000
    """
    Maximum number of training examples to use.
    
    WHY LIMIT:
      Full datasets have millions of rows. For initial experiments and
      learning, using a subset is much faster and shows you the pipeline works.
    
    Guidelines:
      • 1000  — Quick smoke test (10-20 minutes of training)
      • 5000  — Good for learning and experimentation. RECOMMENDED.
      • 20000 — Serious fine-tuning run
      • None  — Use all available data (can take hours/days)
    """

    max_eval_samples: int = 500
    """
    Maximum number of evaluation/validation examples.
    Usually 10-20% of training samples is fine.
    """

    max_seq_length: int = 512
    """
    Maximum number of tokens per input sequence.
    
    WHAT IT DOES:
      Any input longer than this gets truncated. Shorter inputs get padded.
    
    IMPACT:
      • VRAM usage scales QUADRATICALLY with sequence length in attention
      • 512 is a good balance between context and memory
    
    Guidelines:
      • 256  — Short inputs only. Less VRAM.
      • 512  — Good for reviews (avg ~100-200 words). RECOMMENDED.
      • 1024 — Long reviews/documents. Needs more VRAM.
      • 2048 — Gemma's max context. Very memory hungry.
    """

    validation_split: float = 0.1
    """
    Fraction of data to use for validation (10%).
    Used to monitor overfitting during training.
    """

    # ======================================================================
    # torch.compile CONFIGURATION
    # ======================================================================

    use_torch_compile: bool = True
    """
    Whether to use torch.compile() to optimize the model.
    
    WHAT torch.compile DOES:
      PyTorch normally runs in "eager mode" — it executes operations one at a
      time as Python encounters them. torch.compile() analyzes the entire
      computation graph and generates optimized fused GPU kernels.
    
    SPEEDUP: Typically 1.3x - 2x faster training after warmup.
    
    CAVEATS:
      • First few steps are SLOW (compilation overhead, ~2-5 minutes)
      • Requires Triton on Linux for the "inductor" backend
      • May not work with all PEFT operations (we handle this gracefully)
      • Not supported on Windows / macOS
    
    RECOMMENDATION:
      • True  — If you're on Linux with NVIDIA GPU. RECOMMENDED.
      • False — If compilation fails or you're on Windows/Mac.
    """

    torch_compile_backend: str = "inductor"
    """
    Which compilation backend torch.compile should use.
    
    Options:
      • "inductor"  — Default. Uses Triton to generate optimized GPU kernels.
                       Best performance. Requires the 'triton' package. RECOMMENDED.
      • "cudagraphs" — Captures entire CUDA call sequences. Less flexible but
                        can be faster for fixed-shape inputs.
      • "eager"      — No compilation (useful for debugging).
      • "aot_eager"  — Ahead-of-time tracing without optimization. For debugging.
    
    If "inductor" fails with Triton errors, try "cudagraphs" or set
    use_torch_compile=False.
    """

    torch_compile_mode: Optional[str] = "default"
    """
    Compilation aggressiveness level.
    
    Options:
      • "default"        — Balanced. Good compilation speed + runtime perf. RECOMMENDED.
      • "reduce-overhead" — Minimizes Python overhead. Better for small models.
                            Uses CUDA graphs internally.
      • "max-autotune"   — Tries many kernel variants, picks fastest.
                            Compilation is VERY slow (10-30 mins) but gives
                            best runtime performance.
    
    For fine-tuning: Start with "default". Only try "max-autotune" for
    production training runs where compilation time doesn't matter.
    """

    torch_compile_fullgraph: bool = False
    """
    Whether to require the ENTIRE model to be compilable as one graph.
    
    WHAT IT MEANS:
      • True  — torch.compile will ERROR if it encounters any operation
                 it can't compile (graph breaks). Use for maximum perf.
      • False — torch.compile will silently fall back to eager mode for
                 incompatible operations. Safer but potentially slower. RECOMMENDED.
    
    WHY False:
      Some PEFT/LoRA operations cause "graph breaks" (points where the
      compiler can't trace through). With fullgraph=False, those parts
      run in eager mode while the rest is compiled. This is the safe default.
    """

    torch_compile_dynamic: Optional[bool] = None
    """
    Whether to handle dynamic shapes (varying sequence lengths).
    
    Options:
      • None  — Let PyTorch decide automatically. RECOMMENDED.
      • True  — Explicitly tell compiler that tensor shapes may vary.
                 Adds overhead but more flexible.
      • False — Assume all shapes are fixed (same batch size, seq length).
                 Faster compilation but will recompile if shapes change.
    
    In practice, None works best because PyTorch applies heuristics.
    """

    # ======================================================================
    # OUTPUT & LOGGING
    # ======================================================================

    output_dir: str = "./outputs"
    """
    Directory where checkpoints, logs, and the final model are saved.
    Created automatically if it doesn't exist.
    """

    logging_steps: int = 10
    """
    Log training metrics (loss, learning rate, etc.) every N steps.
    Lower = more verbose logging. 10 is a good balance.
    """

    save_steps: int = 100
    """
    Save a checkpoint every N steps.
    
    Checkpoints include: model weights, optimizer state, scheduler state.
    This allows you to resume training if it crashes.
    
    Lower = more frequent saves = more disk space used.
    Set this higher (500-1000) for long training runs to save disk space.
    """

    save_total_limit: int = 3
    """
    Maximum number of checkpoints to keep on disk.
    Older checkpoints are automatically deleted when this limit is exceeded.
    3 means: keep the 3 most recent checkpoints.
    """

    eval_steps: int = 100
    """
    Run validation every N training steps.
    Validation computes loss on the held-out validation set to monitor overfitting.
    """

    report_to: str = "none"
    """
    Where to report training metrics.
    
    Options:
      • "none"    — Just print to console. RECOMMENDED for beginners.
      • "wandb"   — Send to Weights & Biases dashboard (requires wandb login).
      • "tensorboard" — Log to TensorBoard (run `tensorboard --logdir outputs`).
    """

    seed: int = 42
    """
    Random seed for reproducibility.
    Setting a fixed seed ensures you get the same results each run
    (assuming same hardware and software versions).
    """

    # ======================================================================
    # MIXED PRECISION
    # ======================================================================

    fp16: bool = False
    """
    Use float16 mixed precision training.
    
    • Use this if your GPU does NOT support bfloat16 (pre-Ampere GPUs).
    • bfloat16 is preferred over fp16 when available (more numerically stable).
    """

    bf16: bool = True
    """
    Use bfloat16 mixed precision training.
    
    RECOMMENDED for Ampere+ GPUs (RTX 3000/4000 series, A100, H100).
    If your GPU doesn't support bfloat16, set this to False and fp16 to True.
    """

    # ======================================================================
    # GRADIENT CHECKPOINTING
    # ======================================================================

    gradient_checkpointing: bool = True
    """
    Trade compute for memory: recompute activations during backward pass
    instead of storing them all in memory.
    
    MEMORY SAVINGS: Typically reduces VRAM usage by 30-50%.
    SPEED COST: ~20% slower training.
    
    RECOMMENDATION: Always True for consumer GPUs. The memory savings
    are essential. Only set False if you have abundant VRAM (80+ GB).
    """

    # ======================================================================
    # INFERENCE CONFIGURATION
    # ======================================================================

    max_new_tokens: int = 256
    """
    Maximum number of tokens to generate during inference.
    256 tokens ≈ 180-200 words, enough for a detailed recommendation.
    """

    temperature: float = 0.7
    """
    Controls randomness in generation.
    
    • 0.0 — Deterministic (always picks most likely token). Boring but safe.
    • 0.7 — Balanced creativity and coherence. RECOMMENDED.
    • 1.0 — More creative/random.
    • 1.5+ — Very random. Usually produces nonsense.
    """

    top_p: float = 0.9
    """
    Nucleus sampling: only consider tokens whose cumulative probability ≤ top_p.
    
    • 0.9 — Standard. Filters out very unlikely tokens. RECOMMENDED.
    • 0.95 — More diverse outputs.
    • 1.0 — No filtering (consider all tokens).
    """

    top_k: int = 50
    """
    Only consider the top K most likely tokens at each step.
    
    • 50 — Standard. RECOMMENDED.
    • 10 — Very focused/repetitive outputs.
    • 0  — Disable top-k filtering (use only top_p).
    """

    # ======================================================================
    # HELPER METHODS
    # ======================================================================

    @property
    def compute_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype for bitsandbytes config."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.bnb_4bit_compute_dtype, torch.bfloat16)

    @property
    def effective_batch_size(self) -> int:
        """The true batch size after accounting for gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def __str__(self) -> str:
        """Pretty-print all configuration values."""
        lines = ["\n" + "=" * 60, "  TRAINING CONFIGURATION", "=" * 60]
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                lines.append(f"  {key:40s} = {value}")
        lines.append("=" * 60)
        lines.append(f"  Effective batch size: {self.effective_batch_size}")
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)


# ============================================================================
# Quick test: Run this file directly to see all config values
# ============================================================================
if __name__ == "__main__":
    cfg = TrainingConfig()
    print(cfg)
    print("✅ Config loaded successfully!")
    print(f"   Model: {cfg.model_name}")
    print(f"   LoRA rank: {cfg.lora_r}, alpha: {cfg.lora_alpha}")
    print(f"   Learning rate: {cfg.learning_rate}")
    print(f"   Effective batch size: {cfg.effective_batch_size}")
    print(f"   torch.compile enabled: {cfg.use_torch_compile}")
    print(f"   torch.compile backend: {cfg.torch_compile_backend}")
