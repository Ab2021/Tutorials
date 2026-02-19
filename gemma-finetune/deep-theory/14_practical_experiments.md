# 14. Practical Experiments — Hands-On Learning

## Table of Contents
- [Experiment 1: Rank Ablation Study](#experiment-1-rank-ablation-study)
- [Experiment 2: Learning Rate Sweep](#experiment-2-learning-rate-sweep)
- [Experiment 3: Target Module Comparison](#experiment-3-target-module-comparison)
- [Experiment 4: Dataset Size Impact](#experiment-4-dataset-size-impact)
- [Experiment 5: Quantization Quality Impact](#experiment-5-quantization-quality-impact)
- [Experiment 6: torch.compile Speedup Measurement](#experiment-6-torchcompile-speedup-measurement)
- [Experiment 7: Debugging a Failing Training Run](#experiment-7-debugging-a-failing-training-run)
- [Experiment 8: Prompt Format Impact](#experiment-8-prompt-format-impact)
- [Experiment 9: Inference Parameter Tuning](#experiment-9-inference-parameter-tuning)
- [Experiment 10: Full Pipeline Optimization](#experiment-10-full-pipeline-optimization)

---

## Experiment 1: Rank Ablation Study

### Goal
Measure how LoRA rank (r) affects model quality and training cost.

### Setup

```python
# Test ranks: 4, 8, 16, 32, 64
# Keep everything else fixed!

import subprocess
import json

ranks = [4, 8, 16, 32, 64]
results = {}

for r in ranks:
    print(f"\n{'='*60}")
    print(f"  Training with rank r={r}")
    print(f"{'='*60}")
    
    # Modify config
    config_override = f"""
# rank_ablation_config.py
from config import FineTuneConfig
config = FineTuneConfig()
config.lora_r = {r}
config.lora_alpha = {r * 2}  # Keep alpha/r = 2
config.max_train_samples = 2000
config.num_epochs = 3
config.run_name = f"rank_{r}"
"""
    
    # Expected results (approximate):
    # Rank 4:  ROUGE-L ≈ 0.20, Train time ≈ 45 min, Params = 0.4M
    # Rank 8:  ROUGE-L ≈ 0.28, Train time ≈ 48 min, Params = 0.8M
    # Rank 16: ROUGE-L ≈ 0.33, Train time ≈ 52 min, Params = 1.6M
    # Rank 32: ROUGE-L ≈ 0.35, Train time ≈ 60 min, Params = 3.2M
    # Rank 64: ROUGE-L ≈ 0.36, Train time ≈ 75 min, Params = 6.4M
```

### Analysis

```
Expected Pattern:
  ROUGE-L ▲
          │               ___________
    0.36  │              ╱
    0.33  │           __╱
    0.28  │         _╱
    0.20  │       _╱
          │     _╱
          │   _╱
          └────────────────────────────▶ Rank
              4    8   16   32   64

Diminishing returns after r=16:
  r=4→16: +0.13 ROUGE-L (huge improvement)
  r=16→64: +0.03 ROUGE-L (minor improvement, 4× more params)
  
Conclusion: r=16 is the sweet spot for our task.
  More rank helps, but with diminishing returns.
  
What to look for:
  - Overfitting: If val loss increases while train loss decreases at high rank,
    you're memorizing training data → reduce rank.
  - Underfitting: If both train and val loss plateau at low rank,
    the model needs more capacity → increase rank.
```

---

## Experiment 2: Learning Rate Sweep

### Goal
Find the optimal learning rate for our task.

### Setup

```python
# Test LRs: 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3
learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

# For each LR:
# 1. Train for 3 epochs with the same data
# 2. Record train loss, val loss, and ROUGE-L
# 3. Plot the training curves
```

### Expected Results

```
Loss curves for different LRs:

Loss ▲  lr=1e-3 (too high)
       │  ╱╲╱╲╱╲╱╲╱╲╱╲    ← oscillating, never converges!
       │ ╱
       │╱
       │
       │    lr=5e-4
       │   ╲
       │    ╲
       │     ╲____          ← converges but slightly aggressive
       │
       │      lr=2e-4
       │       ╲
       │        ╲
       │         ╲_____     ← converges well (our default)
       │
       │          lr=1e-5
       │           ╲
       │            ╲
       │             ╲  ╲  ╲  ╲   ← too slow, hasn't converged yet
       └──────────────────────────▶ Steps

Optimal range for LoRA fine-tuning: usually 1e-4 to 5e-4
  Too low → wastes compute, may not converge in time
  Too high → unstable training, oscillating loss
  
Quick method: Train for 100 steps at each LR, pick the one
where loss decreased the fastest WITHOUT oscillating.
```

---

## Experiment 3: Target Module Comparison

### Goal
Which attention modules benefit most from LoRA?

### Setup

```python
# Configuration A: Q + V projections only (standard)
target_modules_A = ["q_proj", "v_proj"]

# Configuration B: All attention projections
target_modules_B = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Configuration C: Attention + MLP
target_modules_C = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
```

### Expected Results

```
Config │ Target Modules       │ Trainable │ ROUGE-L │ Time
───────┼──────────────────────┼───────────┼─────────┼──────
A      │ q_proj, v_proj       │ 0.8M      │ 0.28    │ 45m
B      │ q,k,v,o_proj         │ 1.6M      │ 0.33    │ 52m
C      │ q,k,v,o + MLP        │ 2.6M      │ 0.35    │ 62m

Analysis:
  A → B: Adding K and O projections gives +0.05 ROUGE-L (worth it!)
  B → C: Adding MLP gives only +0.02 for 60% more params (marginal)
  
Recommendation:
  Default: Config B (all attention projections) — best quality/cost ratio
  Memory-constrained: Config A (Q+V only)
  Want maximum quality: Config C (all modules)
```

---

## Experiment 4: Dataset Size Impact

### Goal
How much training data do you really need?

### Setup

```python
# Vary dataset size, keep all else fixed
dataset_sizes = [100, 500, 1000, 2000, 5000, 10000]

# For each size:
# 1. Train for 3 epochs
# 2. Evaluate on SAME test set (not affected by training size)
# 3. Record ROUGE-L and training time
```

### Expected Results

```
ROUGE-L ▲
  0.40  │                            ___________
  0.35  │                       ____╱
  0.30  │                  ____╱
  0.25  │             ____╱
  0.20  │        ____╱
  0.15  │   ____╱
  0.10  │__╱
        └──────────────────────────────────────▶ Dataset Size
           100  500  1K   2K   5K   10K

Key findings:
  100 samples:  Model barely learns the format → poor quality
  500 samples:  Learns basic structure, some useful outputs
  1000 samples: Good baseline quality
  2000 samples: Solid performance (our default)
  5000+ samples: Diminishing returns (already covered most patterns)

The "diminishing returns" threshold depends on task complexity:
  Simple tasks (classification-like): ~500 samples sufficient
  Our task (structured generation): ~2000-5000 samples
  Complex tasks (multi-step reasoning): ~10,000+ samples
```

---

## Experiment 5: Quantization Quality Impact

### Goal
How much quality do we lose from 4-bit quantization?

### Setup

```python
# Compare:
# A: Full precision (fp16) — baseline, needs 24+ GB VRAM
# B: 8-bit quantization
# C: 4-bit NF4 quantization (our default)
# D: 4-bit FP4 quantization

# Same LoRA config, same data, same training
```

### Expected Results

```
Config │ Precision │ VRAM    │ ROUGE-L │ Speed  │ Quality Loss
───────┼───────────┼─────────┼─────────┼────────┼─────────────
A      │ fp16      │ 8.5 GB  │ 0.350   │ 1.0×   │ baseline
B      │ int8      │ 4.5 GB  │ 0.345   │ 0.9×   │ -1.4%
C      │ NF4       │ 3.5 GB  │ 0.335   │ 0.8×   │ -4.3%
D      │ FP4       │ 3.5 GB  │ 0.325   │ 0.8×   │ -7.1%

Key findings:
  NF4 > FP4 (NF4 is better suited for normally-distributed weights)
  4-bit has ~4-7% quality loss but HALVES the VRAM requirement
  The quality loss is acceptable for most use cases
  
If quality is critical: Use fp16 (but need more VRAM)
For most uses: NF4 (our default) — best quality/memory trade-off
```

---

## Experiment 6: torch.compile Speedup Measurement

### Goal
Measure actual speedup from torch.compile on your hardware.

### Setup

```python
import torch
import time

def benchmark_training_step(model, batch, compile_enabled):
    """Time one training step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # Warmup (compilation happens here)
    for _ in range(3):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Measure
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times)
    }

# Run:
# without_compile = benchmark_training_step(model, batch, False)
# model_compiled = torch.compile(model)
# with_compile = benchmark_training_step(model_compiled, batch, True)
# speedup = without_compile["mean"] / with_compile["mean"]
# print(f"Speedup: {speedup:.2f}×")
```

### Expected Results

```
GPU              │ Without  │ With     │ Speedup │ Compilation
─────────────────┼──────────┼──────────┼─────────┼────────────
RTX 3060         │ 2.1 s    │ 1.8 s    │ 1.17×   │ ~3 min
RTX 4090         │ 0.8 s    │ 0.55 s   │ 1.45×   │ ~2 min
A100             │ 0.5 s    │ 0.3 s    │ 1.67×   │ ~2 min
T4 (Colab)       │ 3.5 s    │ 3.0 s    │ 1.17×   │ ~4 min

Notes:
  - Speedup varies by GPU (newer GPUs benefit more)
  - First few steps are SLOW (compilation)
  - Speedup is per-step; amortize compilation over total training
  - For < 200 total steps: compilation overhead may outweigh speedup
```

---

## Experiment 7: Debugging a Failing Training Run

### Scenario: Loss is NaN After 50 Steps

```python
# Step-by-step debugging procedure:

# Step 1: Check data
print("Checking data format...")
for i, example in enumerate(train_dataset):
    text = example.get("text", "")
    if not text or len(text) < 10:
        print(f"  BAD: Example {i} is empty or too short: '{text[:50]}'")
    if len(text) > 10000:
        print(f"  WARNING: Example {i} is very long: {len(text)} chars")

# Step 2: Check for NaN in model weights
print("\nChecking model weights...")
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"  NaN found in: {name}")
    if torch.isinf(param).any():
        print(f"  Inf found in: {name}")

# Step 3: Gradient analysis
print("\nChecking gradients...")
loss = model(**batch).loss
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if torch.isnan(grad_norm):
            print(f"  NaN gradient in: {name}")
        elif grad_norm > 100:
            print(f"  EXPLODING gradient in: {name}: norm={grad_norm:.1f}")

# Step 4: Try with fp32 compute
print("\nTrying fp32 compute dtype...")
# In config: bnb_4bit_compute_dtype = "float32"
# Slower but more numerically stable

# Step 5: Reduce learning rate
print("\nTrying LR = 1e-5...")
# Sometimes NaN is caused by too-large updates
```

---

## Experiment 8: Prompt Format Impact

### Goal
Test different instruction formats and their effect on output quality.

### Setup

```python
# Format A: Simple
template_a = "Review: {review}\nRating: {rating}\nRecommendation:"

# Format B: Structured (our default)
template_b = """<start_of_turn>user
Analyze this product review and provide a recommendation.
Review: {review}
Rating: {rating}/5
<end_of_turn>
<start_of_turn>model
"""

# Format C: Detailed instructions
template_c = """<start_of_turn>user
You are a product analyst. Based on the following review and rating,
provide a structured recommendation including:
1. BUY/SKIP/CONSIDER decision
2. Key strengths and weaknesses
3. Marketing strategy suggestion

Review: {review}
Rating: {rating}/5
<end_of_turn>
<start_of_turn>model
"""
```

### Expected Results

```
Format │ Style           │ ROUGE-L │ Format Compliance │ Notes
───────┼─────────────────┼─────────┼───────────────────┼────────────
A      │ Simple          │ 0.28    │ 40%               │ Often unstructured
B      │ Structured      │ 0.33    │ 75%               │ Good balance
C      │ Detailed        │ 0.30    │ 85%               │ Longer, but precise

Insight: More detailed instructions → better FORMAT compliance
but not necessarily higher ROUGE (because the model follows
the instruction format instead of matching the reference exactly).

Recommendation:
  For training: Use Format B (structured, not too long)
  For inference: Can use Format C (more detailed instructions)
```

---

## Experiment 9: Inference Parameter Tuning

### Goal
Find the best generation parameters for output quality.

```python
# Test different temperature, top_k, top_p, repetition_penalty combos

configs = [
    {"temperature": 0.1, "top_k": 50,  "top_p": 0.95, "rep_pen": 1.0},  # Greedy-like
    {"temperature": 0.5, "top_k": 40,  "top_p": 0.90, "rep_pen": 1.1},  # Conservative
    {"temperature": 0.7, "top_k": 50,  "top_p": 0.95, "rep_pen": 1.15}, # Default
    {"temperature": 1.0, "top_k": 100, "top_p": 0.99, "rep_pen": 1.0},  # Creative
    {"temperature": 1.5, "top_k": 200, "top_p": 1.00, "rep_pen": 1.0},  # Very creative
]
```

### Expected Results

```
Config    │ Diversity │ Coherence │ ROUGE-L │ Best For
──────────┼───────────┼───────────┼─────────┼────────────────
Greedy    │ Low       │ High      │ 0.38    │ Factual/consistent
Conservative│ Low-Med  │ High      │ 0.35    │ Professional output
Default   │ Medium    │ Medium    │ 0.33    │ Balanced
Creative  │ High      │ Medium    │ 0.28    │ Varied suggestions
Very Creative│ Very high│ Low      │ 0.20    │ Brainstorming

For product recommendations:
  Use "Conservative" or "Default" → consistent, professional
  Avoid "Very Creative" → too much randomness for recommendations
```

---

## Experiment 10: Full Pipeline Optimization

### Goal
Maximize quality given a fixed GPU and time budget.

```python
# Optimization strategy:
# Time budget: 2 hours on RTX 3060 (12 GB)

# Step 1: Quick baseline (30 minutes)
#   r=16, lr=2e-4, 1000 samples, 2 epochs
#   → Baseline ROUGE-L

# Step 2: LR sweep (30 minutes)
#   Train for 200 steps with LR = [5e-5, 1e-4, 2e-4, 5e-4]
#   Pick best LR

# Step 3: Data scaling (30 minutes)
#   Use best LR, try 2000 and 5000 samples for 1 epoch
#   More data might help more than more epochs

# Step 4: Final training (30 minutes)
#   Best LR + best data size + 3 epochs
#   Full evaluation on test set

# Expected improvement: 10-20% ROUGE-L over naive defaults
```

### Optimization Checklist

```
□ Start with a smoke test (50 samples, 1 epoch) — ensure pipeline works
□ Run the baseline with default config — establish benchmark
□ Sweep learning rate (4 values, 200 steps each) — biggest impact
□ Test different data sizes (3 values) — second biggest impact
□ Try different ranks (r=8, 16, 32) — moderate impact
□ Test target modules (q+v vs all attention) — moderate impact
□ Enable torch.compile — free speedup (if GPU supports it)
□ Final run with best config — full training + evaluation
□ Save results to JSON for reproducibility

Rule of thumb:
  Learning rate > data size > rank > target modules > prompt format
  (ordered by impact on quality)
```
