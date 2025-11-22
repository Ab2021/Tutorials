# Day 6: Optimization & Training Techniques
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Adam Optimizer: Why It Dominates Deep Learning

**The Gradient Descent Landscape Problem:**

Standard SGD treats all dimensions equally:
```
w_{t+1} = w_t - η × g_t
```

**Problem**: Different parameters have different curvatures!

Example loss surface:
```
Dimension 1: Steep valley (large gradients, small optimal step)
Dimension 2: Flat plateau (small gradients, large optimal step needed)

SGD with fixed η:
- Too large for dim 1 → Oscillation
- Too small for dim 2 → Slow progress
```

**Adam's Solution: Per-Parameter Adaptive Rates**

**First moment (Momentum):**
```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
```

Exponential moving average of gradients.

**Why?** Dampens oscillations, accelerates in consistent directions.

**Second moment (RMSProp):**
```
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
```

Exponential moving average of squared gradients.

**Why?** Estimates curvature → larger updates in flat dimensions, smaller in steep.

**Update rule:**
```
m̂_t = m_t / (1 - β₁^t)  # Bias correction
v̂_t = v_t / (1 - β₂^t)

w_t = w_{t-1} - η × m̂_t / (√v̂_t + ε)
```

**Bias Correction Derivation:**

Initially, m_0 = 0, v_0 = 0.

After 1 step:
```
m_1 = (1 - β₁) × g_1
```

Expected value: E[m_1] = (1 - β₁) × E[g_1]

But we want: E[m̂_1] = E[g_1]

Solution: m̂_1 = m_1 / (1 - β₁)

As t → ∞: (1 - β₁^t) → 1, so m̂_t → m_t (no correction needed).

**Why β₁=0.9, β₂=0.999?**

```
β₁ = 0.9: First moment lookback ≈ 1/(1-0.9) = 10 steps
β₂ = 0.999: Second moment lookback ≈ 1/(1-0.999) = 1000 steps
```

First moment (direction): Adapt quickly
Second moment (curvature): Estimate stably over longer horizon

**Adam vs AdamW: The L2 Regularization Bug**

**Standard L2 Regularization:**
```
L = L_task + λ/2 × ||w||²

∇L = ∇L_task + λ × w
```

**Adam with L2:**
```
g_t = ∇L_task + λ × w
m_t = β₁ × m_{t-1} + (1 - β₁) × (∇L_task + λ × w)
v_t = β₂ × v_{t-1} + (1 - β₂) × (∇L_task + λ × w)²

w_t = w_{t-1} - η × m̂_t / √v̂_t
```

**Problem**: Weight decay term (λ × w) goes through adaptive learning rate!

For parameters with large v̂_t (frequent large gradients):
- Effective weight decay ∝ 1/√v̂_t → **Reduced regularization**

For parameters with small v̂_t:
- Effective weight decay ∝ 1/√v̂_t → **Increased regularization**

**Inconsistent regularization across parameters!**

**AdamW Fix: Decoupled Weight Decay**

```python
# Step 1: Adam update (no weight decay in gradient)
g_t = ∇L_task  # Pure task gradient
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²

w_temp = w_{t-1} - η × m̂_t / √v̂_t

# Step 2: Separate weight decay
w_t = w_temp - η × λ × w_{t-1}
```

Now weight decay is **constant** across all parameters!

**Empirical Impact:**

```
Image Classification (ResNet-50):
- Adam + L2: 76.2% accuracy
- AdamW: 76.8% accuracy (+0.6%)

Language Modeling (GPT-2):
- Adam + L2: Perplexity 23.5
- AdamW: Perplexity 22.1 (-1.4)
```

Larger models → bigger difference!

### Learning Rate Warmup: Mathematical Justification

**Problem Without Warmup:**

Early training:
```
Step 1: m_1 = (1-β₁) × g_1, v_1 = (1-β₂) × g_1²
Step 2: m_2 = β₁×m_1 + (1-β₁)×g_2, v_2 = β₂×v_1 + (1-β₂)×g_2²
...
```

Estimates based on few samples → High variance!

With high learning rate + noisy estimates:
```
Large η × noisy m̂_t / √v̂_t → Wild updates → Divergence
```

**Warmup Solution:**

```
For t ≤ T_warmup:
    η_t = η_target × (t / T_warmup)

For t > T_warmup:
    η_t = cosine_schedule(t - T_warmup)
```

**Effect:**
- Small lr initially → Safe despite noisy estimates
- As m_t, v_t stabilize (more samples), increase lr
- By T_warmup, estimates reliable → Can use target lr

**Optimal Warmup Duration:**

Rule of thumb: T_warmup ≈ 1 / (1 - β₂)

For β₂ = 0.999:
```
T_warmup ≈ 1000 steps
```

Gives second moment ~63% of converged value (1 - e^{-1}).

**Transformer-Specific Warmup:**

```python
def transformer_lr(step, d_model, warmup_steps=4000):
    return d_model^{-0.5} × min(step^{-0.5}, step × warmup_steps^{-1.5})
```

**Analysis:**

During warmup (step < warmup_steps):
```
lr = d_model^{-0.5} × step × warmup_steps^{-1.5}
   = (d_model × warmup_steps^3)^{-0.5} × step
```

Linear increase!

After warmup:
```
lr = d_model^{-0.5} × step^{-0.5}
```

Inverse square root decay.

**Why inverse square root?**

Matches theory: Optimal lr ∝ 1/√t for stochastic optimization.

### Mixed Precision Training: Numerical Analysis

**FP32 vs FP16:**

```
FP32: 1 sign bit, 8 exponent bits, 23 mantissa bits
- Range: ~10^{-38} to 10^{38}
- Precision: ~7 decimal digits

FP16: 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range: ~10^{-8} to 10^{4}
- Precision: ~3 decimal digits
```

**Problem 1: Gradient Underflow**

Typical gradient magnitudes:
```
Early layers: 10^{-7} to 10^{-3}
Later layers: 10^{-2} to 10^1
```

FP16 can't represent values < 6×10^{-8} → **Underflow to zero!**

**Solution: Loss Scaling**

```python
# Scale loss by large factor before backward
loss_scaled = loss × scale_factor  # e.g., scale_factor = 2^16

# Backward (gradients also scaled)
loss_scaled.backward()

# Unscale gradients before optimizer
for p in model.parameters():
    p.grad /= scale_factor

# Now gradients in normal range!
```

**Dynamic Loss Scaling:**

```python
class GradScaler:
    def __init__(self, init_scale=2^16):
        self.scale = init_scale
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
        self.growth_interval = 2000
        self.steps_since_increase = 0
    
    def step(self, optimizer):
        # Check for inf/nan in gradients
        inf_or_nan = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    inf_or_nan = True
                    break
        
        if inf_or_nan:
            # Overflow detected → reduce scale
            self.scale *= self.backoff_factor
            self.steps_since_increase = 0
            # Skip optimizer step
        else:
            # Successful step
            # Unscale gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= self.scale
            
            optimizer.step()
            
            # Try to increase scale periodically
            self.steps_since_increase += 1
            if self.steps_since_increase >= self.growth_interval:
                self.scale *= self.growth_factor
                self.steps_since_increase = 0
```

**BFloat16 Advantage:**

```
BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
- Range: Same as FP32 (~10^{-38} to 10^{38})
- Precision: ~2 decimal digits (less than FP16)
```

**Key**: Same exponent range → No underflow/overflow issues → **No scaling needed!**

```python
# BF16 training (simpler)
with autocast(dtype=torch.bfloat16):
    loss = model(input)

loss.backward()  # No scaling!
optimizer.step()
```

**Performance Comparison:**

```
A100 GPU, LLaMA-7B training:

FP32:
- Memory: 28 GB
- Speed: 100 tokens/sec
- Accuracy: Baseline

FP16 + scaling:
- Memory: 14 GB (2× reduction)
- Speed: 180 tokens/sec (1.8× faster)
- Accuracy: -0.1% (negligible)

BF16:
- Memory: 14 GB (2× reduction)
- Speed: 190 tokens/sec (1.9× faster)
- Accuracy: -0.05% (better than FP16)
```

BF16 is best when available (A100, H100, TPU)!

### Gradient Accumulation: Memory-Compute Trade-off

**Effective Batch Size:**

```
batch_size = 8
accumulation_steps = 4

Effective batch size = 8 × 4 = 32
```

**Memory Savings:**

Forward pass memory ∝ batch_size

```
batch=32: 12 GB memory
batch=8: 3 GB memory (4× reduction!)
```

**Compute Cost:**

Must do 4 forward passes before 1 backward.

**Time Overhead:**

```
Standard (batch=32):
- 1 forward + 1 backward = 100ms

Accumulated (batch=8, accum=4):
- 4 forwards + 4 backwards = 4 × 100ms = 400ms

Overhead: 4× time for same effective batch!
```

**But**: Enables training when batch=32 doesn't fit!

**Gradient Statistics Differ:**

With BatchNorm:
```
Standard batch=32:
- Batch statistics computed over 32 samples

Accumulated batch=8×4:
- Batch statistics computed over 8 samples (4 times)
- Different statistics!
```

**Solution**: Use LayerNorm (independent of batch) or GroupNorm.

**Optimal Strategy:**

1. Find maximum batch size that fits in memory
2. If effective batch needed > max batch:
   - Use accumulation
3. Tune accumulation steps to match target effective batch

Example:
```
GPU memory: 16 GB
Max batch without OOM: 16
Target effective batch: 128

accumulation_steps = 128 / 16 = 8
```

### Summary: Modern Training Best Practices

**For Transformers (2025 standard):**

```python
# 1. Optimizer: AdamW with standard hyperparams
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,  # Will be controlled by scheduler
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# 2. Learning rate: Warmup + cosine decay
num_warmup_steps = len(train_loader) * 5  # 5 epochs warmup
num_training_steps = len(train_loader) * num_epochs

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 3. Mixed precision: BF16 if available, else FP16
scaler = GradScaler() if use_fp16 else None
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

# 4. Gradient clipping
max_grad_norm = 1.0

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward with mixed precision
        with autocast(dtype=autocast_dtype):
            outputs = model(batch)
            loss = criterion(outputs, targets)
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        scheduler.step()
```

This recipe achieves:
- Stable training (warmup, clipping, AdamW)
- Fast convergence (adaptive rates, cosine schedule)
- Memory efficiency (mixed precision)
- Good generalization (weight decay, gradient clipping)

Used by: GPT-3, LLaMA, Mistral, all modern LLM's!
