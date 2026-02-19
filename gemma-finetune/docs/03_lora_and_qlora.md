# 3. LoRA and QLoRA — Parameter-Efficient Fine-Tuning

## Table of Contents
- [The Memory Problem](#the-memory-problem)
- [LoRA: The Core Idea](#lora-the-core-idea)
- [The Math Behind LoRA](#the-math-behind-lora)
- [QLoRA: Adding Quantization](#qlora-adding-quantization)
- [Which Layers to Adapt](#which-layers-to-adapt)
- [LoRA Rank: How to Choose](#lora-rank-how-to-choose)
- [LoRA Alpha: The Scaling Factor](#lora-alpha-the-scaling-factor)
- [LoRA vs Other PEFT Methods](#lora-vs-other-peft-methods)
- [Merging Adapters Back](#merging-adapters-back)

---

## The Memory Problem

To understand WHY we need LoRA, let's first understand WHY full fine-tuning is so expensive.

### Memory Breakdown for Full Fine-Tuning (Gemma-2B, fp16)

```
┌────────────────────────────────────────────────────────┐
│                FULL FINE-TUNING MEMORY                  │
│                                                        │
│  Model weights (fp16):           4 GB                  │
│    2B params × 2 bytes/param                           │
│                                                        │
│  Gradients (fp16):               4 GB                  │
│    Same size as weights (one gradient per weight)       │
│                                                        │
│  Optimizer states (AdamW):       16 GB                  │
│    First moment (fp32):  2B × 4 bytes = 8 GB           │
│    Second moment (fp32): 2B × 4 bytes = 8 GB           │
│                                                        │
│  Activations (batch=4, seq=512): 8-12 GB               │
│    Intermediate values saved for backward pass          │
│                                                        │
│  CUDA overhead:                  2 GB                   │
│  ───────────────────────────────────────                │
│  TOTAL:                          ~35-40 GB              │
└────────────────────────────────────────────────────────┘
```

**Problem:** You need a ~$15,000 A100 GPU just to fine-tune a "small" 2B model!

### Memory with QLoRA

```
┌────────────────────────────────────────────────────────┐
│                   QLoRA MEMORY                          │
│                                                        │
│  Model weights (4-bit):          1 GB                  │
│    2B params × 0.5 bytes/param (4-bit quantized)       │
│                                                        │
│  LoRA adapters (fp16):           0.01 GB               │
│    2.6M params × 2 bytes                               │
│                                                        │
│  Gradients (only LoRA, fp16):    0.01 GB               │
│    Only compute gradients for LoRA params              │
│                                                        │
│  Optimizer states (only LoRA):   0.04 GB               │
│    AdamW moments for just 2.6M params                  │
│                                                        │
│  Activations (with grad ckpt):   2-3 GB                │
│    Gradient checkpointing reduces this                 │
│                                                        │
│  CUDA overhead:                  1-2 GB                │
│  ───────────────────────────────────────                │
│  TOTAL:                          ~5-6 GB               │
└────────────────────────────────────────────────────────┘
```

**That's a 6-7× memory reduction!** Now it fits on a $300 RTX 3060.

---

## LoRA: The Core Idea

LoRA was introduced in the paper ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) by Microsoft Research (2021).

### The Key Insight

When you fine-tune a model, the weight updates (Δ_W = W_new - W_original) tend to have **low rank**. This means the updates can be decomposed into much smaller matrices.

```
Standard fine-tuning:
  W_new = W_original + ΔW        
  where ΔW is a full d×d matrix (millions of params)

LoRA's trick:
  W_new = W_original + A × B
  where A is d×r and B is r×d (r << d, so much fewer params!)
```

### Visual Explanation

```
Original Linear Layer:
  ┌─────────────────┐
  │                 │
  │  W (2048×2048)  │  = 4,194,304 params
  │  ALL TRAINABLE  │
  │                 │
  └─────────────────┘

LoRA-Adapted Layer:
  ┌─────────────────┐      ┌──────────┐   ┌──────────┐
  │                 │      │          │   │          │
  │  W (2048×2048)  │  +   │ A (2048× │ × │ B (16×   │
  │  ❄️ FROZEN ❄️    │      │    16)   │   │   2048)  │
  │                 │      │ TRAINABLE│   │ TRAINABLE│
  └─────────────────┘      └──────────┘   └──────────┘
   4,194,304 params        32,768 +        32,768 = 65,536 params
   (not updated)           (updated!)

   Memory savings: 98.4%!
```

### How It Works During a Forward Pass

```python
# Original (no LoRA):
output = input @ W                       # Simple matrix multiplication

# With LoRA:
base_output = input @ W_frozen           # Regular path (frozen, no gradients)  
lora_output = input @ A @ B              # LoRA path (tiny, trainable)
output = base_output + (alpha/r) * lora_output   # Combined
```

### Why Does This Work?

1. **Weight updates are low-rank**: Research shows that fine-tuning for specific tasks only changes weights along a few "directions" in weight space. LoRA captures these directions with rank r.

2. **Pre-trained knowledge is preserved**: Since the original weights W are frozen, the base model's language understanding remains intact.

3. **Adapters are task-specific**: You can train different LoRA adapters for different tasks and swap them in/out:
   ```
   Base Gemma + Product Reviews adapter = Product recommender
   Base Gemma + Medical adapter = Medical assistant
   Base Gemma + Legal adapter = Legal document analyzer
   ```

---

## The Math Behind LoRA

### Forward Pass

For a linear layer with input `x` and original weight `W`:

```
Standard: h = x · W

With LoRA: h = x · W + (α/r) · x · A · B

Where:
  x    ∈ ℝ^(batch × d)      — input
  W    ∈ ℝ^(d × d)          — frozen pre-trained weights
  A    ∈ ℝ^(d × r)          — LoRA down-projection (trainable)
  B    ∈ ℝ^(r × d)          — LoRA up-projection (trainable)
  α    ∈ ℝ                  — scaling hyperparameter
  r    ∈ ℤ                  — rank (bottleneck dimension)
```

### Initialization

- **A** is initialized with random Gaussian values (mean=0, std=small)
- **B** is initialized to all zeros

```
Why zero-init B?
At the start of training:
  h = x · W + (α/r) · x · A · 0 = x · W

So at initialization, LoRA has NO EFFECT on the output!
The model starts exactly as the pre-trained model, and the adapter
gradually learns task-specific adjustments during training.
```

### Backward Pass (Gradient Computation)

During backpropagation, only A and B receive gradients:

```
∂Loss/∂A = (α/r) · xᵀ · (∂Loss/∂h) · Bᵀ
∂Loss/∂B = (α/r) · Aᵀ · xᵀ · (∂Loss/∂h)
∂Loss/∂W = 0  (frozen, no gradient computed = memory saved!)
```

### Total Parameter Count (Our Setup)

```
Gemma-2B has 18 transformer layers.
Each layer has 4 attention linear layers: q_proj, k_proj, v_proj, o_proj.

For each adapted layer (d=2048, r=16):
  A: 2048 × 16 = 32,768 params
  B: 16 × 2048 = 32,768 params
  Per layer: 65,536 params

Total LoRA params:
  18 layers × 4 projections × 65,536 = 4,718,592 params
  
As % of total: 4.7M / 2,000M = 0.24%
```

---

## QLoRA: Adding Quantization

QLoRA (Quantized LoRA), introduced in ["QLoRA: Efficient Finetuning of Quantized Language Models"](https://arxiv.org/abs/2305.14314) by Dettmers et al. (2023), combines LoRA with 4-bit quantization.

### The Three Innovations of QLoRA

**1. 4-Bit NormalFloat (NF4) Quantization**
```
Standard fp16: Each weight = 16 bits = 2 bytes
NF4:           Each weight = 4 bits = 0.5 bytes

Memory: 2B × 0.5 bytes = 1 GB (vs. 4 GB in fp16)

NF4 specifically designed for normally-distributed weights
(neural network weights ~ Normal distribution)
```

**2. Double Quantization**
```
Problem: Each block of 64 weights needs a fp16 scaling constant
         2B params / 64 = 31.25M scaling constants × 2 bytes = 62.5 MB

Solution: Quantize the scaling constants too! (to 8-bit)
          31.25M × 1 byte = 31.25 MB

Savings: ~31 MB extra. Small but free!
```

**3. Paged Optimizer**
```
Problem: Optimizer states sometimes exceed GPU memory during spikes
Solution: Use CPU-GPU memory paging (like virtual memory)
          Temporarily offload optimizer states to CPU RAM
          Automatically pages back to GPU when needed
```

### QLoRA vs LoRA vs Full Fine-Tuning Quality

From the original QLoRA paper:

```
Benchmark: MMLU (Massive Multitask Language Understanding)

Method          | Accuracy | VRAM  | Relative to full FT
───────────────┼──────────┼───────┼─────────────────────
Full FT (16-bit)| 63.2%    | 40 GB | Baseline
LoRA (16-bit)   | 63.0%    | 8 GB  | -0.2% (negligible)
QLoRA (4-bit)   | 62.8%    | 6 GB  | -0.4% (negligible)

The quality loss from QLoRA vs full fine-tuning is typically < 1%
```

---

## Which Layers to Adapt

### Gemma's Linear Layers

Each transformer block in Gemma has 7 linear layers:

```
ATTENTION:
  q_proj: Query projection      (2048 → 2048)
  k_proj: Key projection        (2048 → 256)  [GQA: fewer heads]
  v_proj: Value projection      (2048 → 256)  [GQA: fewer heads]
  o_proj: Output projection     (2048 → 2048)

FEED-FORWARD NETWORK (MLP):
  gate_proj: Gate projection     (2048 → 16384)
  up_proj:   Up projection       (2048 → 16384)
  down_proj: Down projection     (16384 → 2048)
```

### Strategies for Target Module Selection

**Strategy 1: Attention Only (Our Default)**
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```
- LoRA params: ~4.7M
- Memory: Low
- Quality: Good for most tasks
- **When to use:** Default starting point. Recommended.

**Strategy 2: Attention + MLP**
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```
- LoRA params: ~10M
- Memory: Medium
- Quality: Better, especially for knowledge-heavy tasks
- **When to use:** When attention-only doesn't achieve desired quality

**Strategy 3: Minimal (Q and V only)**
```python
target_modules = ["q_proj", "v_proj"]
```
- LoRA params: ~2.4M
- Memory: Lowest
- Quality: Often surprisingly good
- **When to use:** Extremely limited VRAM, or quick experiments

**Strategy 4: All Linear Layers**
```python
target_modules = "all-linear"  # PEFT shorthand
```
- LoRA params: ~15M
- Memory: Highest
- Quality: Best
- **When to use:** Have plenty of VRAM, want maximum quality

### Decision Tree

```
                 How much VRAM do you have?
                         │
              ┌──────────┼──────────┐
              │          │          │
           < 8 GB     8-16 GB    > 16 GB
              │          │          │
              ▼          ▼          ▼
          q, v only   q,k,v,o    q,k,v,o +
                    (DEFAULT)    gate,up,down
```

---

## LoRA Rank: How to Choose

The rank `r` is the single most important LoRA hyperparameter.

### What Rank Controls

```
r = 1:  ΔW = a × bᵀ  (rank-1 update, all changes are in one direction)
r = 4:  ΔW = sum of 4 rank-1 updates (4 independent directions)
r = 16: ΔW = sum of 16 rank-1 updates (16 independent directions)
r = d:  ΔW = full-rank (equivalent to full fine-tuning!)
```

### Higher Rank = More Capacity

```
Rank  │ Trainable Params │ Capacity │ Overfitting Risk │ Speed
──────┼──────────────────┼──────────┼──────────────────┼──────
r=4   │ ~1.2M            │ Low      │ Low              │ Fast
r=8   │ ~2.4M            │ Medium   │ Low              │ Fast
r=16  │ ~4.7M            │ Good     │ Medium           │ Medium
r=32  │ ~9.4M            │ High     │ Medium-High      │ Slower
r=64  │ ~18.9M           │ Very High│ High             │ Slow
r=128 │ ~37.7M           │ Maximum  │ Very High        │ Slowest
```

### Decision Guide

| Your Situation | Recommended Rank |
|---------------|-----------------|
| Quick experiment / smoke test | r = 4 or 8 |
| Small dataset (< 1K examples) | r = 8 |
| **Standard fine-tuning** | **r = 16 (our default)** |
| Large dataset (> 50K examples) | r = 32 |
| Complex task, underfitting seen | r = 32 or 64 |
| Maximum quality, no VRAM limit | r = 64 |

### How to Tell If Rank Is Too Low or Too High

**Too Low (underfitting):**
- Training loss plateaus at a high value
- Validation loss remains high
- Fix: Increase rank

**Too High (overfitting):**
- Training loss keeps decreasing
- But validation loss INCREASES
- Fix: Decrease rank, or increase dropout

---

## LoRA Alpha: The Scaling Factor

### The Formula

```
output = base_output + (alpha / rank) × lora_output
```

The ratio `alpha/rank` determines how much the LoRA adapter influences the output.

### Common Settings

| Alpha | Rank | Ratio (α/r) | Effect |
|-------|------|-------------|--------|
| 8 | 8 | 1.0 | Standard influence |
| 16 | 16 | 1.0 | Standard influence |
| **32** | **16** | **2.0** | **Stronger influence (our default)** |
| 64 | 32 | 2.0 | Stronger influence |
| 16 | 16 | 1.0 | Conservative |

### Rule of Thumb

**Set `alpha = 2 × rank`.** This gives a scaling ratio of 2.0, which works well empirically across many tasks and models.

If you change the rank, change alpha proportionally:
```
r=8  → alpha=16
r=16 → alpha=32 (our default)
r=32 → alpha=64
r=64 → alpha=128
```

---

## LoRA vs Other PEFT Methods

| Method | How It Works | Params | Quality | Our Choice? |
|--------|-------------|--------|---------|-------------|
| **LoRA** | Low-rank adapter matrices | 0.1-1% | ⭐⭐⭐⭐ | ✅ |
| **QLoRA** | LoRA + 4-bit quantization | 0.1-1% | ⭐⭐⭐⭐ | ✅✅ |
| Prefix Tuning | Learnable input prefixes | < 0.1% | ⭐⭐⭐ | ❌ |
| P-Tuning v2 | Learnable prompts at every layer | < 0.1% | ⭐⭐⭐ | ❌ |
| IA³ | Learned activation scaling | < 0.01% | ⭐⭐⭐ | ❌ |
| AdaLoRA | Adaptive rank allocation | 0.1-1% | ⭐⭐⭐⭐ | ❌ |

**LoRA/QLoRA dominate** because:
1. Best quality-to-efficiency ratio
2. Can be merged into base model (zero inference overhead)
3. Well-supported by HuggingFace PEFT library
4. Extensively tested across many models and tasks

---

## Merging Adapters Back

After training, you have two options:

### Option A: Keep Separate (Development)
```
Base Model (4-bit) + LoRA Adapter (fp16)
                   ↓
   Applied dynamically during forward pass
```
- ✅ Can swap adapters (different tasks)
- ✅ Can compare base vs. adapted model
- ❌ Slightly more inference memory

### Option B: Merge (Production)
```python
model = model.merge_and_unload()
# Now: model.W = W_original + (α/r) × A × B
# The LoRA wrapper is removed
```
- ✅ Faster inference (no adapter overhead)
- ✅ Simpler deployment
- ❌ Can't un-merge or swap adapters
- ❌ Can't merge cleanly with 4-bit quantized model (need to dequantize first)

### When to Use Each

| Scenario | Choice |
|----------|--------|
| Experimenting with different configs | Keep separate |
| Comparing multiple fine-tuned versions | Keep separate |
| **Deploying to production** | **Merge** |
| Sharing the model | Merge |
