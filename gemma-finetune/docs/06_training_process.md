# 6. The Training Process — What Happens Under the Hood

## Table of Contents
- [Overview: The Training Loop](#overview-the-training-loop)
- [Loss Function: Cross-Entropy](#loss-function-cross-entropy)
- [Backpropagation](#backpropagation)
- [Optimizers: AdamW Deep Dive](#optimizers-adamw-deep-dive)
- [Learning Rate Schedulers](#learning-rate-schedulers)
- [Gradient Accumulation](#gradient-accumulation)
- [Gradient Checkpointing](#gradient-checkpointing)
- [Mixed Precision Training](#mixed-precision-training)
- [Gradient Clipping](#gradient-clipping)
- [Monitoring Training: What to Watch](#monitoring-training-what-to-watch)

---

## Overview: The Training Loop

Every neural network training follows this loop:

```
for epoch in range(num_epochs):          # Repeat full dataset N times
    for batch in training_data:           # Process one batch at a time
    
        # 1. FORWARD PASS
        predictions = model(batch.inputs)  # Model generates predictions
        
        # 2. COMPUTE LOSS
        loss = loss_fn(predictions, batch.targets)  # How wrong is the model?
        
        # 3. BACKWARD PASS (Backpropagation)
        loss.backward()                    # Compute gradients
        
        # 4. GRADIENT CLIPPING
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 5. OPTIMIZER STEP
        optimizer.step()                   # Update weights using gradients
        
        # 6. SCHEDULER STEP
        scheduler.step()                   # Adjust learning rate
        
        # 7. ZERO GRADIENTS
        optimizer.zero_grad()              # Reset gradients for next batch
```

### What Happens at Each Step (For Our Gemma Fine-Tuning)

```
Step 1: FORWARD PASS
  Input: "The product is great because" → tokens [651, 2857, 338, 1571, 1363]
  
  For each position, model predicts next token:
    Position 0 ("The"):     predicts probabilities for position 1
    Position 1 ("product"): predicts probabilities for position 2
    Position 2 ("is"):      predicts probabilities for position 3
    Position 3 ("great"):   predicts probabilities for position 4
    Position 4 ("because"): predicts probabilities for position 5

Step 2: COMPUTE LOSS
  At position 3, model predicted:
    P("great") = 0.15      ← correct answer
    P("bad") = 0.20        ← wrong
    P("a") = 0.10          ← wrong
    
  Cross-entropy loss = -log(0.15) = 1.90
  Better prediction → lower loss
  Perfect prediction (P=1.0) → loss = 0

Step 3: BACKWARD PASS
  Only computes gradients for LoRA parameters (2.6M)
  Skips frozen base model parameters (2B)
  
Step 5: OPTIMIZER STEP
  For each LoRA parameter:
    param = param - learning_rate × gradient
  (Actually more complex with AdamW — see below)
```

---

## Loss Function: Cross-Entropy

### What Is Cross-Entropy Loss?

For language modeling, the loss measures how well the model predicts the correct next token.

```
Given: Model's predicted probability for the correct token = p

Cross-Entropy Loss = -log(p)

Examples:
  p = 1.0 (perfect prediction):  loss = -log(1.0) = 0.0    ← best
  p = 0.5 (decent prediction):   loss = -log(0.5) = 0.69
  p = 0.1 (poor prediction):     loss = -log(0.1) = 2.30
  p = 0.01 (terrible prediction): loss = -log(0.01) = 4.60  ← worst

Lower loss = better model
```

### Over the Full Sequence

```
Training example: "The product is great"
Tokens: [The, product, is, great, <eos>]

Loss = average of per-token losses:
  loss_1 = -log(P("product" | "The"))        = 2.1
  loss_2 = -log(P("is" | "The product"))     = 0.8
  loss_3 = -log(P("great" | "The product is")) = 1.5
  loss_4 = -log(P("<eos>" | "The product is great")) = 0.3
  
  Total loss = (2.1 + 0.8 + 1.5 + 0.3) / 4 = 1.175
```

### What Good Loss Values Look Like

```
For causal language modeling (our task):

Loss > 5.0:   Model is outputting near-random predictions
Loss 3.0-5.0: Early training, model is learning basic patterns
Loss 1.5-3.0: Model is learning task-specific patterns
Loss 0.5-1.5: Model has learned well ← target for fine-tuning
Loss < 0.5:   Possibly overfitting (memorizing training data)
```

---

## Backpropagation

### The Chain Rule

Backpropagation uses calculus chain rule to compute gradients:

```
Forward:  input → Layer1 → Layer2 → Layer3 → loss
                   h1        h2       h3

Backward: 
  ∂loss/∂W3 = ∂loss/∂h3 × ∂h3/∂W3
  ∂loss/∂W2 = ∂loss/∂h3 × ∂h3/∂h2 × ∂h2/∂W2
  ∂loss/∂W1 = ∂loss/∂h3 × ∂h3/∂h2 × ∂h2/∂h1 × ∂h1/∂W1
```

### What Happens with LoRA

```
Forward:
  input → [Frozen W + LoRA A×B] → output → loss

Backward:
  ∂loss/∂A = ∂loss/∂output × ∂output/∂(A×B) × ∂(A×B)/∂A  ← Computed!
  ∂loss/∂B = ∂loss/∂output × ∂output/∂(A×B) × ∂(A×B)/∂B  ← Computed!
  ∂loss/∂W = NOT COMPUTED (frozen, requires_grad=False)    ← Skipped!

Memory savings:
  Full: Store gradients for 2B params = 4 GB
  LoRA: Store gradients for 2.6M params = 0.005 GB ← 800x less!
```

---

## Optimizers: AdamW Deep Dive

### Why Not Plain SGD?

**SGD (Stochastic Gradient Descent):**
```
param = param - learning_rate × gradient
```
Simple, but:
- ❌ All parameters use the same learning rate
- ❌ Gradients are noisy (from mini-batches), causing zigzag path
- ❌ Gets stuck in flat regions (plateaus)

### How AdamW Works

AdamW (Adam with Weight Decay) maintains TWO moving averages for each parameter:

```
For each parameter θ, AdamW tracks:

m = "momentum" = exponential moving average of gradients
    (like a heavy ball rolling — smooths out noisy gradients)
    
v = "velocity" = exponential moving average of SQUARED gradients
    (tracks how much each gradient varies)
    
At each step t:

1. Compute gradient: g_t = ∂Loss/∂θ

2. Update momentum:  m_t = β₁ × m_{t-1} + (1-β₁) × g_t
                     Think: "weighted average of recent gradients"
                     β₁ = 0.9 (default) → strong momentum

3. Update velocity:  v_t = β₂ × v_{t-1} + (1-β₂) × g_t²
                     Think: "how volatile is this gradient?"
                     β₂ = 0.999 (default) → long memory

4. Bias correction:  m̂_t = m_t / (1 - β₁ᵗ)
                     v̂_t = v_t / (1 - β₂ᵗ)
                     Fix the zero-initialization bias

5. Update parameter: θ_t = θ_{t-1} - lr × (m̂_t / (√v̂_t + ε) + λ × θ_{t-1})
                                         ↑ adaptive per-param LR  ↑ weight decay

Where:
  lr = learning rate (2e-4 in our config)
  ε = 1e-8 (prevents division by zero)
  λ = weight_decay (0.01 in our config)
```

### Why AdamW Specifically?

**Adam** (without W): Applies weight decay INSIDE the adaptive learning rate.
**AdamW**: Applies weight decay SEPARATELY (decoupled). This is mathematically correct and gives better regularization.

```
Adam:  θ = θ - lr × m̂/(√v̂ + ε) - lr × λ × θ/(√v̂ + ε)
       Weight decay is scaled by the adaptive term ← WRONG

AdamW: θ = θ - lr × m̂/(√v̂ + ε) - lr × λ × θ
       Weight decay is applied directly ← CORRECT
```

### 8-Bit Paged AdamW (Our Choice)

```python
optim="paged_adamw_8bit"
```

**8-bit:** Stores momentum (m) and velocity (v) in 8-bit instead of 32-bit.
```
Standard AdamW: m (fp32) + v (fp32) = 8 bytes per param
8-bit AdamW:    m (int8) + v (int8) = 2 bytes per param → 4× less memory!

For 2.6M LoRA params:
  Standard: 2.6M × 8 bytes = 20.8 MB
  8-bit:    2.6M × 2 bytes = 5.2 MB  (saves 15.6 MB)
```

**Paged:** If GPU runs out of memory during an optimizer step, automatically pages optimizer states to CPU RAM and back.

### Other Optimizer Choices

| Optimizer | Memory | Speed | Stability | When to Use |
|-----------|--------|-------|-----------|-------------|
| `adamw_torch` | High | Fast | ⭐⭐⭐⭐⭐ | Have plenty of VRAM |
| `adamw_8bit` | Low | Fast | ⭐⭐⭐⭐ | Limited VRAM |
| **`paged_adamw_8bit`** | **Lowest** | **Fast** | **⭐⭐⭐⭐** | **Very limited VRAM (ours)** |
| `adafactor` | Very low | Medium | ⭐⭐⭐ | Extreme memory constraints |
| `sgd` | Lowest | Fastest | ⭐⭐ | Not recommended for LLMs |

---

## Learning Rate Schedulers

### Why Not Constant Learning Rate?

```
Constant LR through training:
  Start: LR may be too high → wild gradient updates → unstable
  Middle: LR is about right → steady learning
  End: LR may be too high → overshoots minimum → can't converge precisely

Solution: Start low (warmup), peak, then decay
```

### Warmup Phase

```
Steps 1-30 (warmup_ratio=0.03 of 1000 total steps):

  Learning rate: 0 → 2e-4 (linearly increases)
  
  Why? At the start:
  - LoRA adapters are randomly initialized
  - Gradients are very noisy
  - Large learning rate would cause destructive updates
  
  Warmup lets the model "ease into" training
```

### Cosine Schedule (Our Choice)

```
Learning Rate vs. Training Step:

LR ▲
   │  ╱╲
   │ ╱  ╲
   │╱    ╲
   │      ╲
   │       ╲
   │        ╲
   │         ╲
   │          ╲╲
   │            ╲╲
   │              ╲╲___________
   └──────────────────────────► Step
   warmup    decay

Formula: lr = lr_max × 0.5 × (1 + cos(π × progress))

Where progress goes from 0 to 1 over training.
```

### Why Cosine Is Better Than Linear

```
Cosine:  Decreases slowly at first, faster in the middle, slowly at the end
Linear:  Decreases at a constant rate throughout

Cosine advantage:
  - Spends more time at moderate LR (where most learning happens)
  - Gradually approaches zero (fine-grained convergence at the end)
  - Empirically gives ~1-3% better results on LLM fine-tuning
```

### Other Scheduler Options

| Scheduler | Shape | Best For |
|-----------|-------|----------|
| `linear` | Straight line down | Simple, predictable |
| **`cosine`** | **Cosine curve** | **LLM fine-tuning (our choice)** |
| `cosine_with_restarts` | Repeating cosine | Very long training |
| `constant` | Flat line | Not recommended |
| `polynomial` | Polynomial decay | Some vision tasks |

---

## Gradient Accumulation

### The Problem

You want `effective_batch_size=16` for stable gradients, but your GPU only fits `batch_size=4`.

### The Solution

```
Without accumulation (batch_size=16, needs lots of VRAM):
  Step 1: Process 16 examples → compute gradient → update weights

With accumulation (batch_size=4, accumulation_steps=4):
  Step 1.1: Process 4 examples → compute gradient → ACCUMULATE (don't update)
  Step 1.2: Process 4 examples → compute gradient → ACCUMULATE
  Step 1.3: Process 4 examples → compute gradient → ACCUMULATE
  Step 1.4: Process 4 examples → compute gradient → ACCUMULATE → UPDATE weights!

Mathematically equivalent to batch_size=16, but only needs VRAM for 4 examples!
```

### Our Config

```python
per_device_train_batch_size = 4        # GPU processes 4 at a time
gradient_accumulation_steps = 4        # Accumulate 4 mini-batches
effective_batch_size = 4 × 4 = 16     # Same as batch_size=16
```

---

## Gradient Checkpointing

### The Memory Problem

During the forward pass, PyTorch saves all intermediate activations (hidden states) so it can compute gradients during the backward pass.

```
Forward: input → h1 → h2 → h3 → ... → h18 → output
                 ↑     ↑     ↑           ↑
                 All saved in GPU memory!
                 
For Gemma-2B with batch_size=4, seq_length=512:
  Each hidden state: 4 × 512 × 2048 × 2 bytes = 8 MB
  18 layers × multiple per layer ≈ 2-4 GB of activations!
```

### The Solution: Gradient Checkpointing

```
Instead of saving ALL activations, only save SOME (checkpoints).
During backward pass, RECOMPUTE the missing activations from checkpoints.

Standard:    Save h1, h2, h3, h4, h5, h6, ..., h18  (all saved)
Checkpointing: Save h1,     h3,         h6, ..., h18  (every 3rd)
               Recompute h2 from h1, h4 from h3, etc.

Memory: Save ~1/3 of activations → ~60% less memory
Speed:  Recomputation adds ~20% training time
```

### Our Config

```python
gradient_checkpointing = True   # Always on for consumer GPUs
# Saves ~1-2 GB of VRAM at the cost of ~20% slower training
# Essential for fitting Gemma-2B on 8 GB GPUs
```

---

## Mixed Precision Training

### What It Is

Use lower precision (16-bit) for most operations, higher precision (32-bit) only where needed:

```
┌────────────────────────────────────────────────────┐
│               MIXED PRECISION                       │
│                                                    │
│  Forward pass:  bfloat16  (fast, less memory)      │
│  Backward pass: bfloat16  (fast, less memory)      │
│  Weight update: float32   (accurate, prevents      │
│                            rounding errors)         │
│  Loss scaling:  float32   (prevents underflow)     │
│  Layer norms:   float32   (numerically sensitive)   │
└────────────────────────────────────────────────────┘
```

### Why It Works

Most neural network operations (matmul, convolution) are numerically stable even in 16-bit. Only specific operations (normalization, loss computation, weight updates) need 32-bit precision.

### Benefits

```
Full float32:     Memory: 100%    Speed: 100%
Mixed precision:  Memory: ~60%    Speed: ~150% (Tensor Cores!)

Tensor Cores: Special hardware on NVIDIA GPUs (Volta+) that does
16-bit matrix multiplication at 2x the speed of 32-bit.
```

---

## Gradient Clipping

### The Problem: Exploding Gradients

Sometimes a "bad batch" produces extremely large gradients:

```
Normal gradients:    [0.01, -0.02, 0.003, 0.015, ...]
Exploding gradients: [0.01, -0.02, 500.0, 0.015, ...]  ← one huge value!

Without clipping:
  param = param - lr × 500.0 = massive weight change!
  This can destabilize the entire model
```

### How Gradient Clipping Works

```python
max_grad_norm = 1.0  # Our setting

# 1. Compute the total gradient norm:
total_norm = sqrt(sum(grad² for all parameters))

# 2. If total_norm > max_grad_norm, scale ALL gradients down:
if total_norm > 1.0:
    scale = 1.0 / total_norm
    for param in model.parameters():
        param.grad *= scale

# Now the total gradient norm = 1.0 (or less)
# Direction preserved, magnitude limited
```

### Our Setting

```python
max_grad_norm = 1.0  # Standard value, works well for most LLM fine-tuning
```

---

## Monitoring Training: What to Watch

### Key Metrics

```
Step 100 | Loss: 2.45 | LR: 1.5e-4 | Grad Norm: 0.82 | GPU: 5.2 GB

Training Loss:
  Should DECREASE over time
  2.5 → 2.0 → 1.5 → 1.2 → ... ✅ Good
  2.5 → 2.5 → 2.6 → 2.5 → ... ❌ Not learning (try higher LR or rank)
  2.5 → 0.1 → 0.01 → 0.001     ❌ Possible overfitting

Validation Loss:
  Should ALSO decrease, tracking training loss
  If train_loss decreases but val_loss INCREASES → overfitting!

Learning Rate:
  Should follow your schedule (warmup → peak → cosine decay)

Gradient Norm:
  Should be stable, typically 0.1-2.0
  If > 5.0 consistently → training may be unstable, reduce LR
  If always at max_grad_norm → clipping is active, may need lower LR

GPU Memory:
  Should be stable after the first step
  If gradually increasing → memory leak (rare, restart training)
```

### Warning Signs

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss = NaN | LR too high, bad data | Reduce LR, check data |
| Loss doesn't decrease | LR too low, rank too low | Increase LR or rank |
| Loss increases | LR too high | Reduce LR |
| Train loss ↓, val loss ↑ | Overfitting | More data, more dropout, fewer epochs |
| Both losses plateau high | Underfitting | Higher rank, more epochs |
| CUDA OOM | Batch too large | Reduce batch size |
| Very slow steps | torch.compile compiling | Normal for first 3-5 steps |
