# 5. Optimization Theory — From SGD to AdamW

## Table of Contents
- [The Optimization Problem](#the-optimization-problem)
- [Vanilla Gradient Descent](#vanilla-gradient-descent)
- [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
- [Momentum: Physics-Inspired Optimization](#momentum-physics-inspired-optimization)
- [Nesterov Accelerated Gradient](#nesterov-accelerated-gradient)
- [Adaptive Learning Rates: AdaGrad and RMSProp](#adaptive-learning-rates-adagrad-and-rmsprop)
- [Adam: Combining Momentum and Adaptivity](#adam-combining-momentum-and-adaptivity)
- [AdamW: Decoupled Weight Decay](#adamw-decoupled-weight-decay)
- [8-Bit Optimizers](#8-bit-optimizers)
- [Learning Rate Schedules: Theory and Practice](#learning-rate-schedules-theory-and-practice)
- [Loss Landscapes: What Does the Terrain Look Like?](#loss-landscapes-what-does-the-terrain-look-like)
- [Convergence Theory](#convergence-theory)
- [Practical: Comparing Optimizers](#practical-comparing-optimizers)

---

## The Optimization Problem

### What We're Trying to Solve

```
Find parameters θ* that minimize the loss function L:

  θ* = arg min L(θ)
           θ

Where:
  θ = all trainable parameters (LoRA weights A, B for each layer)
  L(θ) = (1/N) × Σᵢ loss(model(xᵢ; θ), yᵢ)
  
  The average loss over all training examples.
```

### Why This Is Hard

```
1. NON-CONVEX: The loss landscape has many local minima and saddle points.
   There's no guarantee of finding the global minimum.

2. HIGH-DIMENSIONAL: Even with LoRA, we have ~2.6 million parameters.
   Imagine navigating a 2.6M-dimensional landscape!

3. STOCHASTIC: We approximate the true gradient using mini-batches,
   introducing noise into our direction estimates.

4. COMPUTATIONAL COST: Each gradient computation requires a full
   forward + backward pass through the model.
```

---

## Vanilla Gradient Descent

### Algorithm

```
θₜ₊₁ = θₜ - η × ∇L(θₜ)

Where:
  η = learning rate (step size)
  ∇L(θₜ) = gradient of loss with respect to θ at current point
```

### Visual Intuition

```
Loss landscape (2D slice):

  Loss ▲
       │    ╱╲
       │   ╱  ╲
       │  ╱    ╲      ╱╲
       │ ╱      ╲    ╱  ╲
       │╱        ╲__╱    ╲___
       │              ╲___    ╲
       └──────────────────────▶ θ
       
       Start →  gradient points RIGHT (downhill)
       →  step right →  gradient points RIGHT
       →  step right →  arrive at minimum!

The gradient ∇L points in the direction of STEEPEST ASCENT.
We move in the OPPOSITE direction (subtract) to descend.
```

### Problems

```
1. LEARNING RATE IS CRITICAL:
   Too large: Overshoot and oscillate
   Too small: Crawl slowly and get stuck

   η too large:           η just right:          η too small:
     ╱╲  ×                  ╱╲                      ╱╲
    ╱× ╲  ×               ╱  ╲                    ╱  ╲
   ╱  × ╲                ╱ ×  ╲                  ╱×   ╲
   ×    ×    Diverging!      × ╲  Converging!    ×  ×  ╲  Very slow!
                              ×                    ×  ×

2. SAME LR FOR ALL PARAMETERS:
   Some parameters may need large steps, others need tiny steps.

3. FULL BATCH REQUIRED:
   Computing gradient over ALL examples is expensive.
```

---

## Stochastic Gradient Descent (SGD)

### Key Insight: Use a Random Sample

```
Instead of gradient over ALL N examples:
  ∇L = (1/N) × Σᵢ ∇lossᵢ      ← expensive!

Use gradient over ONE random example:
  ∇L̃ = ∇loss_random            ← cheap!

Or a mini-batch of B examples:
  ∇L̃ = (1/B) × Σᵢ₌₁ᴮ ∇lossᵢ  ← good trade-off

Expected value: E[∇L̃] = ∇L (unbiased estimator)
But with VARIANCE from sampling → noisy gradient
```

### SGD with Mini-Batch

```python
# SGD Algorithm
for epoch in range(num_epochs):
    shuffle(training_data)
    for batch in mini_batches(training_data, batch_size=B):
        gradient = compute_gradient(model, batch)  # Stochastic gradient
        for param in model.parameters():
            param = param - lr * gradient[param]
```

### The Noise Is Actually Helpful!

```
Noise from mini-batches:
  ❌ Makes optimization path zigzag (slower)
  ✅ Helps escape shallow local minima
  ✅ Provides implicit regularization (avoids sharp minima)
  ✅ Sharp minima → poor generalization, flat minima → good generalization

The noisy path:
        ╱╲
       ╱  ╲  local      ╱╲
      ╱ ×  ╲ minimum   ╱  ╲
     ╱   └──↗──────×──╱    │  ← SGD ESCAPES shallow local minimum!
    ╱              ╲__╱    ╲___  ← finds BETTER (flatter) minimum
```

---

## Momentum: Physics-Inspired Optimization

### The Problem SGD Has

```
Without momentum, SGD oscillates in narrow valleys:

         ╱╲
        ╱  ╲
       ╱ ×←→× ╲   ← oscillates back and forth!
      ╱  ×←→×  ╲
     ╱    ×→     ╲
                  minimum

The gradient points mostly UP the valley walls (steep),
not along the valley floor (where we want to go).
```

### Momentum Accumulates Previous Gradients

```
Algorithm:
  vₜ = β × vₜ₋₁ + ∇L(θₜ)        (accumulate velocity)
  θₜ₊₁ = θₜ - η × vₜ             (update using velocity)

Where:
  v = velocity (running average of gradients)
  β = momentum coefficient (typically 0.9)

Effect:
  Components that consistently point in the same direction ACCUMULATE
  Components that oscillate CANCEL OUT

  Without momentum:         With momentum (β=0.9):
       ╱╲                        ╱╲
      ╱  ╲                      ╱  ╲
     ╱ ×←→× ╲                 ╱ × ╲
    ╱  ×←→×  ╲               ╱  ×→ ╲
   ╱    ×     ╲             ╱    ×→→→→ ╲
                minimum              minimum  ← much faster!
```

### Exponential Moving Average

```
vₜ = β × vₜ₋₁ + (1-β) × gₜ

Expansion:
  vₜ = (1-β) × gₜ + (1-β)β × gₜ₋₁ + (1-β)β² × gₜ₋₂ + ...

The weight of a gradient from k steps ago:
  weight = (1-β) × βᵏ

For β = 0.9:
  Current gradient:  weight = 0.10  (10%)
  1 step ago:        weight = 0.09  (9%)
  2 steps ago:       weight = 0.081 (8.1%)
  10 steps ago:      weight = 0.035 (3.5%)
  50 steps ago:      weight = 0.0005 (0.05%)

Effectively averages the last ~1/(1-β) = 10 gradients.
```

---

## Nesterov Accelerated Gradient

### Look Ahead Before Computing Gradient

```
Standard momentum:
  1. Compute gradient at current position
  2. Update velocity
  3. Move

Nesterov:
  1. Take a tentative step using current velocity (look ahead)
  2. Compute gradient at the LOOK-AHEAD position
  3. Update velocity using this smarter gradient
  4. Move

This is more responsive to changes in the loss landscape:
  If we're about to overshoot, the look-ahead gradient will tell us
  to slow down BEFORE we actually overshoot.
```

---

## Adaptive Learning Rates: AdaGrad and RMSProp

### AdaGrad: Per-Parameter Learning Rates

```
Key idea: Parameters that appear frequently (large accumulated gradients)
should have SMALLER learning rates.

Algorithm:
  Gₜ = Gₜ₋₁ + gₜ²              (accumulate squared gradients)
  θₜ₊₁ = θₜ - η / √(Gₜ + ε) × gₜ

Each parameter gets its own effective learning rate: η / √(Gₜ + ε)
  - Frequently updated params → large Gₜ → smaller LR
  - Rarely updated params → small Gₜ → larger LR

Problem: Gₜ only grows → learning rate monotonically decreases → STOPS learning
```

### RMSProp: Fix AdaGrad's Decay

```
Instead of accumulating ALL squared gradients, use exponential moving average:

  vₜ = β × vₜ₋₁ + (1-β) × gₜ²    (moving average of squared gradients)
  θₜ₊₁ = θₜ - η / √(vₜ + ε) × gₜ

β = 0.999 → average of the last ~1000 squared gradients

Now the learning rate can INCREASE again if recent gradients are smaller.
Doesn't suffer from AdaGrad's monotonic decay.
```

---

## Adam: Combining Momentum and Adaptivity

### The Best of Both Worlds

```
Adam = Momentum + RMSProp

It maintains TWO moving averages:
  m = momentum of gradients (direction)        ← from Momentum
  v = momentum of SQUARED gradients (scale)    ← from RMSProp

Algorithm:
  mₜ = β₁ × mₜ₋₁ + (1-β₁) × gₜ              Step 1: Update momentum
  vₜ = β₂ × vₜ₋₁ + (1-β₂) × gₜ²             Step 2: Update velocity
  
  m̂ₜ = mₜ / (1 - β₁ᵗ)                        Step 3: Bias correction
  v̂ₜ = vₜ / (1 - β₂ᵗ)                         (important for early steps)
  
  θₜ₊₁ = θₜ - η × m̂ₜ / (√v̂ₜ + ε)            Step 4: Update

Default hyperparameters:
  β₁ = 0.9     (momentum decay, averages ~10 gradients)
  β₂ = 0.999   (velocity decay, averages ~1000 squared gradients)
  ε = 1e-8     (prevents division by zero)
  η = learning rate (task-specific, e.g., 2e-4)
```

### Why Bias Correction?

```
Problem: At t=1, m₁ = 0.9 × 0 + 0.1 × g₁ = 0.1 × g₁

The momentum estimate is BIASED toward 0 (because m₀ = 0).
The true gradient is g₁, but the estimate is 0.1 × g₁ — 10× too small!

Fix: Divide by (1 - β₁ᵗ)
  At t=1: m̂₁ = 0.1g₁ / (1 - 0.9¹) = 0.1g₁ / 0.1 = g₁  ← corrected!
  At t=2: m̂₂ = (0.09g₁ + 0.1g₂) / (1 - 0.81) = ...     ← corrected
  As t → ∞: (1 - β₁ᵗ) → 1, correction disappears
```

### The Adaptive Learning Rate

```
Per-parameter effective learning rate:

  η_effective = η / (√v̂ₜ + ε)

For a parameter with:
  - Large gradients (high variance): v̂ is large → smaller LR → cautious
  - Small gradients (low variance): v̂ is small → larger LR → aggressive
  - Consistent gradients: m̂ / √v̂ is stable → steady progress
```

---

## AdamW: Decoupled Weight Decay

### The Problem with L2 Regularization in Adam

```
L2 regularization adds a penalty for large weights:
  L_total = L_data + (λ/2) × ||θ||²

The gradient becomes:
  ∇L_total = ∇L_data + λ × θ

In standard Adam, this penalty gradient (λ × θ) goes through the
adaptive learning rate mechanism:

  θₜ₊₁ = θₜ - η × (m̂ₜ + λ × θₜ) / (√v̂ₜ + ε)

The weight decay is SCALED by the adaptive term 1/(√v̂ₜ + ε).
This means:
  - Parameters with large gradients → less weight decay
  - Parameters with small gradients → more weight decay
  
  This is NOT the intended behavior of weight decay!
```

### AdamW: The Fix

```
AdamW separates weight decay from the adaptive mechanism:

  θₜ₊₁ = θₜ - η × m̂ₜ / (√v̂ₜ + ε) - η × λ × θₜ
                ↑ adaptive gradient update    ↑ fixed weight decay

Now weight decay is applied UNIFORMLY to all parameters,
regardless of their gradient history.

This is mathematically equivalent to true weight decay
(not L2 regularization), which empirically works better.
```

### Why This Matters

```
From the paper "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019):

Experiments on various tasks:
  Adam + L2 reg:       Accuracy: 91.2%
  AdamW (decoupled):   Accuracy: 92.8%   ← better!
  
The improvement is consistent across many tasks and models.
All modern LLM training uses AdamW, not Adam.
```

---

## 8-Bit Optimizers

### Memory Cost of Optimizer States

```
For each parameter, Adam stores:
  m (first moment): same size as parameter
  v (second moment): same size as parameter

Total: 2 × number_of_parameters × bytes_per_param

Standard (fp32):
  LoRA params: 2.6M
  m + v = 2 × 2.6M × 4 bytes = 20.8 MB

8-bit (int8):
  m + v = 2 × 2.6M × 1 byte = 5.2 MB

Savings: 15.6 MB (75% less)

For full fine-tuning of Gemma-2B:
  Standard: 2 × 2B × 4 = 16 GB (just optimizer states!)
  8-bit:    2 × 2B × 1 = 4 GB
```

### How 8-Bit Quantization Works for Optimizer States

```
Standard: m and v stored in fp32 (32 bits, 4 bytes)
8-bit:    m and v stored in dynamic int8 (8 bits, 1 byte)

Each block of 2048 optimizer state values:
  1. Find absmax of the block
  2. Normalize to [-127, 127] range
  3. Round to nearest integer
  4. Store as int8 + fp32 scaling factor

Dequantize when needed:
  value_fp32 = value_int8 × (absmax / 127)

Quality loss: < 0.1% in practice (optimizer states don't need high precision)
```

---

## Learning Rate Schedules: Theory and Practice

### Why Not Just Constant LR?

```
Constant LR = 2e-4 throughout training:

Problem 1 (Early Training):
  LoRA weights are randomly initialized → gradients are noisy
  Large LR × noisy gradient = destructive updates!

Problem 2 (Late Training):
  Near the minimum, we want FINE adjustments
  But constant LR makes the same SIZE steps as before
  → oscillates around the minimum instead of converging
```

### Warmup From Theory

```
Linear warmup for the first W steps:

  lr(t) = lr_max × (t / W)    for t < W

Purpose:
  At step 1: lr = lr_max × (1/W) ≈ 0 (tiny steps)
  At step W: lr = lr_max (full learning rate)
  
  Gradually ramp up to allow the optimizer states (m, v) to
  accumulate meaningful statistics before making large updates.

Our config: warmup_ratio = 0.03
  Total steps ≈ 1000 → warmup = 30 steps
```

### Cosine Annealing Theory

```
After warmup, cosine schedule:

  lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × progress))
  
  where progress = (t - W) / (T - W), going from 0 to 1

Properties:
  At progress=0: lr = lr_max
  At progress=0.5: lr = (lr_max + lr_min) / 2 (midpoint)
  At progress=1: lr = lr_min (typically 0)

The derivative of cosine is sine:
  d(lr)/d(progress) = -0.5 × (lr_max - lr_min) × π × sin(π × progress)

  At start (progress=0): Rate of change = 0 → gentle start
  At middle (progress=0.5): Rate of change = maximum → fastest decrease
  At end (progress=1): Rate of change = 0 → gentle stop
  
  This means the LR spends more time near lr_max (productive learning)
  and more time near lr_min (fine convergence) than a linear schedule would.
```

---

## Loss Landscapes: What Does the Terrain Look Like?

### Flat vs Sharp Minima

```
Sharp minimum:                 Flat minimum:
  Loss ▲                        Loss ▲
       │  ╱╲                         │  ╱────────╲
       │ ╱  ╲                        │ ╱          ╲
       │╱    ╲                       │╱            ╲
       ╲      ╲                      ╲              ╲
       
  Sharp = small change in θ          Flat = small change in θ
  → large change in loss             → small change in loss
  
  Sharp minima: BAD for generalization
    Train set minimum might not align with test set minimum
    
  Flat minima: GOOD for generalization
    Robust to slight distribution shifts between train and test

SGD with momentum tends to find FLAT minima → better generalization
```

### Saddle Points

```
In high-dimensional spaces, saddle points are MORE COMMON than local minima!

  Loss ▲
       │   ╱╲
       │  ╱  ╲
       │ ╱    ╲
       │╱      ╲
       │        ╲╱ ← saddle point (looks like minimum in this 2D slice)
       │          ╲
       │           ╲
       
In other dimensions, it's a MAXIMUM.
Gradient = 0 but it's NOT a minimum!

SGD's noise helps escape saddle points (cannot get stuck at zero gradient).
Adam's momentum also helps (accumulated velocity carries through).
```

---

## Convergence Theory

### Convergence Rate of SGD

```
For a convex function with Lipschitz continuous gradients:

  E[L(θₜ) - L(θ*)] ≤ O(1/√t)

"Expected loss gap decreases as 1/√t"

After 100 steps: gap ≤ C/10
After 10000 steps: gap ≤ C/100

Convergence is SLOW! But this is the worst case bound.
In practice, Adam converges much faster for neural networks.
```

### Why Adam Converges Faster (In Practice)

```
Adam's effective update:  m̂ / √v̂

When the gradient is consistent:
  m̂ ≈ g (current gradient)
  v̂ ≈ g² (current gradient squared)
  Update ≈ g / √g² = ±1 (sign of gradient!)

This means Adam approximately does SIGN gradient descent:
  θₜ₊₁ = θₜ - η × sign(gₜ)

Every parameter gets the SAME magnitude update (just positive or negative).
This works surprisingly well because:
  - Large gradients don't dominate
  - Small gradients aren't ignored
  - All parameters make meaningful progress
```

---

## Practical: Comparing Optimizers

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Simple test: minimize Rosenbrock function
# f(x,y) = (1-x)² + 100(y-x²)²
# Minimum at (1, 1)

def rosenbrock(params):
    x, y = params[0], params[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

optimizers = {
    "SGD": lambda p: torch.optim.SGD(p, lr=0.001),
    "SGD+Momentum": lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
    "Adam": lambda p: torch.optim.Adam(p, lr=0.01),
    "AdamW": lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.01),
}

results = {}
for name, opt_fn in optimizers.items():
    params = torch.tensor([-1.0, 1.0], requires_grad=True)
    optimizer = opt_fn([params])
    
    trajectory = [params.detach().clone().numpy()]
    losses = []
    
    for step in range(500):
        optimizer.zero_grad()
        loss = rosenbrock(params)
        loss.backward()
        optimizer.step()
        
        trajectory.append(params.detach().clone().numpy())
        losses.append(loss.item())
    
    results[name] = {"trajectory": trajectory, "losses": losses}
    print(f"{name:20s}: Final loss = {losses[-1]:.6f}, "
          f"Final pos = ({params[0]:.4f}, {params[1]:.4f})")

# Expected output:
# SGD                : Final loss = 3.421000, Final pos = (-0.8531, 0.7324)
# SGD+Momentum       : Final loss = 0.012300, Final pos = (0.8891, 0.7894)
# Adam               : Final loss = 0.000012, Final pos = (0.9965, 0.9930)
# AdamW              : Final loss = 0.000015, Final pos = (0.9961, 0.9922)
```
