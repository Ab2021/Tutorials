# Day 2: Deep Learning Fundamentals for NLP
## Deep Dive - Internal Mechanics

### The Adam Optimizer: Why It Dominates LLM Training

**Problem with vanilla SGD:**

Consider parameter with sparse gradients (embedding for rare word):
- Most steps: gradient ≈ 0
- Occasional step: gradient = large value

SGD applies same learning rate → poor convergence.

**Adam's Solution:**

Maintains per-parameter adaptive learning rates based on gradient history.

**Full Algorithm:**

```python
# Initialization
m = 0  # First moment (mean)
v = 0  # Second moment (variance)  
β1 = 0.9  # Exponential decay for first moment
β2 = 0.999  # Exponential decay for second moment
ε = 1e-8  # Numerical stability

for t in range(1, num_steps+1):
    g = compute_gradient()
    
    # Update biased moments
    m = β1 * m + (1 - β1) * g
    v = β2 * v + (1 - β2) * g²
    
    # Bias correction (important for early steps)
    m_hat = m / (1 - β1^t)
    v_hat = v / (1 - β2^t)
    
    # Update parameters
    θ = θ - α * m_hat / (√v_hat + ε)
```

**Why Bias Correction Matters:**

Initially (t=1):
- m = (1-β1) * g = 0.1 * g (underestimates true mean!)
- Without correction: m would start near 0 regardless of g

With correction:
- m_hat = m / (1 - 0.9) = m / 0.1 = 10m (corrects the bias)

As t→∞, (1-β^t)→1, bias correction →identity.

**AdamW vs Adam:**

Standard Adam with L2 regularization:
```python
g = grad(L) + λ * w  # Add weight penalty to gradient
# Then apply Adam update with this modified gradient
```

**Problem:** Adaptive learning rate applies to regularization term too (unintended!)

**AdamW:**
```python
g = grad(L)  # Pure gradient
m, v = update_moments(g)
w = w - α * m_hat / √v_hat - λ * w  # Separate weight decay
```

**Result:** Weight decay acts consistently regardless of gradient magnitude.

### Backpropagation: The Chain Rule at Scale

**Forward Pass (computation graph):**

```
input → Linear → ReLU → Linear → Output
  x       Wx+b     a      Ua+c     y
```

**Backward Pass (reverse mode autodiff):**

Given dL/dy, compute dL/dW, dL/dU, dL/db, dL/dc.

**Chain Rule:**
```
dL/dW = dL/dy × dy/da × da/dW
```

**PyTorch Autograd:**

Every Tensor has `.grad_fn` pointing to the operation that created it.

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = y * 3

z.backward()  # Computes dz/dx
print(x.grad)  # dz/dx = d(3x²)/dx = 6x = 6.0
```

**How it works:**
1. Forward: Build computation graph (y.grad_fn = PowBackward, z.grad_fn = MulBackward)
2. Backward: Traverse graph in reverse, apply chain rule at each node
3. Accumulate gradients in .grad

**Memory Trade-off:**

Must store all intermediate activations for backward pass!

For Transformer layer:
```python
x → Attention → Add&Norm → FFN → Add&Norm → output
```

Must store: input, attention output, norm output, FFN output.

With 32 layers × batch_size × seq_len × d_model → GBs of memory!

**Gradient Checkpointing Solution:**
- Don't store intermediate activations
- Recompute them during backward pass
- Trade: 20% more compute, save 30-50% memory

### Loss Landscape and Training Dynamics

**Non-Convex Optimization:**

Neural network loss is highly non-convex (many local minima).

**Surprising Finding (LLMs):**

Local minima are often "good enough" - they generalize well!

**Why?**
- Overparameterized models (more params than data points)
- All critical points (gradient=0) tend to have similar loss values
- SGD noise helps escape poor local minima

**Gradient Noise as Regularization:**

Small batch SGD has noisy gradients.

**Effect:**
- Prevents overfitting (doesn't converge to sharp minima)
- Sharp minima → poor generalization
- Flat minima → good generalization

SGD noise biases toward flat minima!

### Layer Normalization: Why It Works for Transformers

**Batch Norm (for CNNs):**

Normalize over batch dimension and spatial dimensions.

```python
# BatchNorm: normalize over (batch, height, width), per channel
mean = x.mean(dim=[0, 2, 3])  # Shape: (channels,)
```

**Problem for NLP:**
- Variable sequence lengths
- Small batches (GPU memory constraints)
- Batch statistics unstable

**Layer Norm:**

Normalize over feature dimension, per sample, per position.

```python
# LayerNorm: normalize over features, per sample
mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch, seq_len, 1)
```

**Why it helps training:**

1. **Gradient Flow:** Normalizes inputs to each layer → gradients don't explode/vanish
2. **Learning Rate:** Can use larger LR (inputs are normalized)
3. **Invariance:** Output invariant to input scale

**Pre-Norm vs Post-Norm:**

Post-Norm (original Transformer):
```
x = LayerNorm(x + Sublayer(x))
```

Pre-Norm (modern, more stable):
```
x = x + Sublayer(LayerNorm(x))
```

Pre-Norm is more stable (gradient flows through residual path).

### Information Theory Perspective on Loss

**Cross-Entropy Loss:**

L = -Σ p(x) log q(x)

Where:
- p(x) = true distribution (one-hot for classification)
- q(x) = predicted distribution (softmax output)

**Interpretation:**

Cross-entropy measures "surprise" under q when true distribution is p.

**KL Divergence:**

CE(p, q) = H(p) + KL(p || q)

Where H(p) is entropy of true distribution.

For one-hot p, H(p) = 0, so:

CE(p, q) = KL(p || q)

Minimizing cross-entropy = minimizing KL divergence!

**Label Smoothing:**

Instead of one-hot [0, 0, 1, 0], use smoothed [0.025, 0.025, 0.9, 0.025].

**Why?**
- Reduces overconfidence  
- Better calibration
- Regularization effect
- Common in LLM training

### Gradient Accumulation: Mathematical Equivalence

**Claim:** Accumulating gradients over N batches = training with batch size N×B.

**Proof:**

Let L_i = loss on batch i.

**Standard training (large batch):**
```
g = ∇(L_1 + L_2 + ... + L_N) / N
w = w - η * g
```

**Gradient accumulation:**
```
g_1 = ∇L_1, g_2 = ∇L_2, ..., g_N = ∇L_N
g_accum = (g_1 + g_2 + ... + g_N) / N
w = w - η * g_accum
```

By linearity of derivatives:
```
∇(L_1 + L_2 + ... + L_N) = ∇L_1 + ∇L_2 + ... + ∇L_N
```

Therefore: **Mathematically equivalent!**

**Subtle Difference with BatchNorm:**

If using BatchNorm, statistics computed per mini-batch → not exactly equivalent.

Solution: Use LayerNorm or GroupNorm (independent of batch size).

### Debugging Training Dynamics

**Signs of healthy training:**

1. **Loss:** Smooth decrease, no sudden spikes
2. **Learning Rate:** Should decrease over time (cosine/linear decay)
3. **Gradient Norm:** Stable, not growing explosively
4. **Weight Norm:** Grows slowly, then stabilizes

**Pathological Behaviors:**

**Loss Spike:**
- Cause: Gradient explosion, bad batch, learning rate too high
- Fix: Gradient clipping, reduce LR, check for data issues

**Loss Plateau Early:**
- Cause: Learning rate too low, poor initialization
- Fix: Increase LR, warmup longer, check gradients

**NaN Loss:**
- Cause: Numerical overflow (often in softmax/exp)
- Fix: Mixed precision loss scaling, gradient clipping, check for inf in data

**Monitoring Script:**
```python
import torch.nn.utils as utils

for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    
    # Monitor gradient norms
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > 10.0:
        print(f"Warning: Large gradient norm {total_norm}")
    
    utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### Summary

Deep learning for NLP relies on:
- **AdamW** for stable, adaptive optimization
- **Layer Norm** for stable gradients in variable-length sequences  
- **Gradient accumulation** to simulate large batches
- **Careful monitoring** to catch training pathologies
- **Information-theoretic losses** (cross-entropy) for probabilistic predictions

Understanding these internals enables debugging and optimization of LLM training.
