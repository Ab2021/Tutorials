# 8. Hyperparameter Guide — Decision Trees & Tuning Strategies

## Table of Contents
- [The Most Important Hyperparameters](#the-most-important-hyperparameters)
- [Decision Trees for Every Parameter](#decision-trees-for-every-parameter)
- [Hyperparameter Interaction Effects](#hyperparameter-interaction-effects)
- [Tuning Strategy: A Step-by-Step Plan](#tuning-strategy-a-step-by-step-plan)
- [Quick Reference Card](#quick-reference-card)

---

## The Most Important Hyperparameters

Not all hyperparameters are created equal. Here's the priority ranking:

```
Impact on Results:

HIGH IMPACT    ██████████████████████  Learning Rate
               █████████████████████   LoRA Rank (r)
               ████████████████████    Dataset Size
               ███████████████████     Number of Epochs
               ██████████████████      Target Modules

MEDIUM IMPACT  ████████████████        Batch Size (effective)
               ███████████████         Max Sequence Length
               ██████████████          LoRA Alpha
               █████████████           LoRA Dropout

LOW IMPACT     ████████████            Weight Decay
               ███████████             Warmup Ratio
               ██████████              LR Scheduler Type
               █████████               Gradient Clipping
               ████████                Quantization Type (NF4 vs FP4)
```

---

## Decision Trees for Every Parameter

### 1. Learning Rate

```
                Start with 2e-4 (default for LoRA)
                            │
               Run training for 100 steps
                            │
              What does the loss look like?
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    Loss is NaN/Inf   Loss decreasing    Loss stuck/plateau
    or spiking        smoothly           (not decreasing)
          │                 │                 │
          ▼                 ▼                 ▼
    TOO HIGH!          JUST RIGHT!       TOO LOW!
    Reduce by 4x      Keep current       Increase by 2x
    → Try 5e-5         = 2e-4            → Try 4e-4
          │                                   │
     Still bad?                          Still stuck?
          │                                   │
     Try 1e-5                            Try 1e-3
                                         (max for LoRA)
```

### Recommended Values by Scenario

| Scenario | Learning Rate | Why |
|----------|--------------|-----|
| First experiment | 2e-4 | Standard LoRA default |
| Small dataset (< 1K) | 5e-5 | Prevent overfitting |
| Large dataset (> 50K) | 2e-4 to 5e-4 | Can learn more aggressively |
| Seeing instability | 5e-5 | More conservative updates |
| Underfitting | 5e-4 | Faster learning (careful!) |

---

### 2. LoRA Rank (r)

```
        How complex is your task?
                    │
     ┌──────────────┼──────────────┐
     │              │              │
  Simple         Moderate        Complex
  (sentiment,    (recommend-     (detailed
   classify)      ation)          analysis)
     │              │              │
     ▼              ▼              ▼
   r = 8          r = 16        r = 32-64
     │              │              │
     │         How much data?      │
     │              │              │
     │    ┌─────────┼──────────┐   │
     │    │         │          │   │
     │   < 5K     5K-50K    > 50K  │
     │    │         │          │   │
     │    ▼         ▼          ▼   │
     │  r = 8    r = 16     r = 32 │
     │                             │
     └─────── Check VRAM ──────────┘
                    │
         Is it running out of memory?
                    │
            ┌───────┴───────┐
           YES              NO
            │                │
            ▼                ▼
        Reduce r         Keep current r
        by half          
```

### Memory Impact of Rank

| Rank | LoRA Params | Extra VRAM | Quality |
|------|-------------|------------|---------|
| r=4 | ~1.2M | ~0.005 GB | Basic |
| r=8 | ~2.4M | ~0.01 GB | Decent |
| **r=16** | **~4.7M** | **~0.02 GB** | **Good (default)** |
| r=32 | ~9.4M | ~0.04 GB | Better |
| r=64 | ~18.9M | ~0.08 GB | Best |

> Note: LoRA params are tiny — the rank has minimal direct VRAM impact.
> The indirect impact (gradients, optimizer states) is also small.

---

### 3. Number of Epochs

```
          How big is your dataset?
                    │
     ┌──────────────┼──────────────┐
     │              │              │
  < 1K examples  1K-20K examples  > 20K examples
     │              │              │
     ▼              ▼              ▼
   5-10 epochs    3-5 epochs     1-3 epochs
     │              │              │
     └──────────────┼──────────────┘
                    │
          Monitor validation loss
                    │
         ┌──────────┼──────────┐
         │          │          │
    Val loss     Val loss    Val loss
    still ↓      plateaus    going ↑
         │          │          │
         ▼          ▼          ▼
    Keep going!   Stop here   OVERFITTING!
    Add more     (this is     Reduce epochs
    epochs       your best)   or add more data
```

### The Overfitting Check

```
Training Loss vs. Validation Loss over epochs:

Loss ▲
     │  ╲  Training
     │   ╲ Loss          Validation
     │    ╲___           Loss
     │        ╲___    ╱───────
     │            ╲╱                    ← DIVERGENCE = Overfitting!
     │             │           
     └─────────────┴───────► Epochs
                   │
              Stop training here!
              This is your best checkpoint.
```

---

### 4. Batch Size

```
        How much GPU VRAM do you have?
                    │
     ┌──────────────┼──────────────┐
     │              │              │
   8 GB           16 GB         24+ GB
     │              │              │
     ▼              ▼              ▼
  batch=1         batch=4       batch=8
  accum=16        accum=4       accum=2
  (eff=16)        (eff=16)      (eff=16)
     │              │              │
     └──────────────┴──────────────┘
                    │
     Effective batch size should be 16-32
     for most LLM fine-tuning tasks
```

### Why Effective Batch Size Matters

```
Effective batch size too SMALL (< 4):
  - Gradients are very noisy
  - Training is unstable
  - Convergence is slow

Effective batch size SWEET SPOT (16-32):
  - Gradients are reasonably stable
  - Good convergence speed
  - Balances memory and quality

Effective batch size too LARGE (> 64):
  - Very stable but slow to converge
  - May need higher learning rate
  - Diminishing returns
```

---

### 5. Max Sequence Length

```
    What's the average length of your text?
                    │
     ┌──────────────┼──────────────┐
     │              │              │
  Short text     Medium text    Long text
  (tweets,       (reviews,      (articles,
   labels)        emails)        documents)
     │              │              │
     ▼              ▼              ▼
   128-256        512            1024-2048
     │              │              │
     └──────────────┼──────────────┘
                    │
          VRAM check: Attention is O(n²)
                    │
         ┌──────────┼──────────┐
         │          │          │
    OOM at 512    OK at 512   Lots of VRAM
         │          │          │
         ▼          ▼          ▼
    Use 256      Use 512     Try 1024
                 (default)
```

### Memory vs Sequence Length

```
Sequence Length │ Relative VRAM │ Content Coverage
───────────────┼───────────────┼──────────────────
128            │ ▓░░░░░░░░░    │ Very short snippets
256            │ ▓▓░░░░░░░░    │ Short reviews
512 (default)  │ ▓▓▓▓░░░░░░    │ Typical reviews ← Our choice
1024           │ ▓▓▓▓▓▓▓░░░    │ Long reviews + full prompts
2048           │ ▓▓▓▓▓▓▓▓▓▓    │ Very long documents
```

---

### 6. Target Modules

```
   How much VRAM is available for LoRA?
                    │
     ┌──────────────┼──────────────┐
     │              │              │
  Very tight      Normal        Plenty
  (< 8 GB)       (8-16 GB)     (> 24 GB)
     │              │              │
     ▼              ▼              ▼
  ["q_proj",    ["q_proj",    ["q_proj",
   "v_proj"]     "k_proj",     "k_proj",
                  "v_proj",     "v_proj",
                  "o_proj"]     "o_proj",
                 (DEFAULT)      "gate_proj",
                                "up_proj",
                                "down_proj"]
```

---

## Hyperparameter Interaction Effects

### Learning Rate × Batch Size

```
Key principle: When you increase batch size, increase learning rate proportionally.

batch_size=4, lr=2e-4     → baseline
batch_size=8, lr=4e-4     → scale up linearly
batch_size=16, lr=8e-4    → or use sqrt scaling: lr=2.8e-4

Why? Larger batches give smoother gradients, which can tolerate larger steps.
```

### LoRA Rank × Dataset Size

```
                    Dataset Size
                Small (<1K)    Medium (1K-20K)    Large (>20K)
              ┌───────────────┬─────────────────┬────────────────┐
  Rank Low    │ OK            │ Underfitting     │ Underfitting   │
  (r=4-8)     │ (risk:overfit)│                  │                │
              ├───────────────┼─────────────────┼────────────────┤
  Rank Med    │ Overfitting   │ ✅ SWEET SPOT    │ Good           │
  (r=16)      │               │                  │                │
              ├───────────────┼─────────────────┼────────────────┤
  Rank High   │ Overfitting   │ Good             │ ✅ SWEET SPOT  │
  (r=32-64)   │               │                  │                │
              └───────────────┴─────────────────┴────────────────┘
```

### Epochs × Data Size

```
Rule of thumb: Total tokens seen = dataset_size × epochs × avg_seq_length

Target: 1-10 million token-steps for good fine-tuning

Dataset 1K  × 5 epochs × 300 avg_len = 1.5M tokens ← minimum viable
Dataset 5K  × 3 epochs × 300 avg_len = 4.5M tokens ← our default, good
Dataset 20K × 2 epochs × 300 avg_len = 12M tokens  ← very good
```

---

## Tuning Strategy: A Step-by-Step Plan

### Phase 1: Smoke Test (5 min)
```python
# Verify pipeline works
max_train_samples = 100
max_eval_samples = 20
num_epochs = 1
lora_r = 8
learning_rate = 2e-4

# Expected: Training completes without errors
# Don't worry about quality yet
```

### Phase 2: Baseline (30 min)
```python
# Get a baseline to compare against
max_train_samples = 5000       # Default
num_epochs = 3                  # Default
lora_r = 16                     # Default
learning_rate = 2e-4            # Default

# Record: final train loss, final val loss, ROUGE scores
# This is your BASELINE — all future experiments compare to this
```

### Phase 3: Learning Rate Search (1 hour)
```python
# Try 3 learning rates, keep everything else fixed
experiments = [
    {"learning_rate": 5e-5},   # Conservative
    {"learning_rate": 2e-4},   # Default (baseline)
    {"learning_rate": 5e-4},   # Aggressive
]

# Pick the one with lowest VALIDATION loss
# (not training loss — that could be overfitting)
```

### Phase 4: Rank Search (1 hour)
```python
# Try 3 ranks, use best LR from Phase 3
experiments = [
    {"lora_r": 8,  "lora_alpha": 16},
    {"lora_r": 16, "lora_alpha": 32},   # Default
    {"lora_r": 32, "lora_alpha": 64},
]

# Pick based on val loss and ROUGE scores
```

### Phase 5: Scale Up (2-4 hours)
```python
# Use best LR and rank, scale up training
max_train_samples = 20000      # More data
num_epochs = 3-5                # More epochs
# Monitor for overfitting!
```

### Phase 6: Final Evaluation
```python
# Run evaluate.py on the TEST set (not validation!)
# This is your final, unbiased quality score
python evaluate.py --model_dir ./outputs/best_run/final_model
```

---

## Quick Reference Card

### Default Config (Start Here)

```python
# Model
model_name = "google/gemma-2b"
use_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = "bfloat16"

# LoRA
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
num_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-4
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
max_grad_norm = 1.0
weight_decay = 0.01

# Data
max_train_samples = 5000
max_seq_length = 512

# torch.compile  
use_torch_compile = True
torch_compile_backend = "inductor"
torch_compile_mode = "default"
torch_compile_fullgraph = False

# Precision
bf16 = True  # (use fp16=True if pre-Ampere GPU)
gradient_checkpointing = True
```

### If You Have Problems

| Problem | Parameter | Change |
|---------|-----------|--------|
| CUDA OOM | `per_device_train_batch_size` | 4 → 1 |
| CUDA OOM | `max_seq_length` | 512 → 256 |
| CUDA OOM | `use_torch_compile` | True → False |
| Loss is NaN | `learning_rate` | 2e-4 → 5e-5 |
| Loss is NaN | `bf16` → `fp16` | If pre-Ampere GPU |
| Overfitting | `lora_dropout` | 0.05 → 0.15 |
| Overfitting | `num_epochs` | 3 → 1 |
| Underfitting | `lora_r` | 16 → 32 |
| Underfitting | `learning_rate` | 2e-4 → 5e-4 |
| Slow training | `use_torch_compile` | False → True |
| Compile errors | `torch_compile_backend` | "inductor" → "eager" |
