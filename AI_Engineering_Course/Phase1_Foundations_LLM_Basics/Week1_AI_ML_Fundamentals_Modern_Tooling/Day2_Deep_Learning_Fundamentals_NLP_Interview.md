# Day 2: Deep Learning Fundamentals for NLP
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain the vanishing gradient problem. How do modern architectures address it?

**Answer:**

**Vanishing Gradient Problem:**

In deep networks, gradients exponentially decay as they backpropagate through layers.

For network with L layers, gradient at layer 1:
```
∂L/∂w_1 = ∂L/∂z_L × ∂z_L/∂z_{L-1} × ... × ∂z_2/∂z_1 × ∂z_1/∂w_1
```

If each ∂z_i/∂z_{i-1} < 1 (e.g., sigmoid derivative ≤ 0.25):
```
∂L/∂w_1 ≈ (0.25)^L → 0 as L increases
```

Early layers don't learn!

**Solutions in Modern Architectures:**

1. **Residual Connections (ResNets, Transformers):**
   ```python
   x = x + F(x)  # Skip connection
   ```
   Gradient flows through identity path (∂/∂x of identity = 1).

2. **ReLU/GELU Activations:**
   - ReLU derivative = 1 (for x>0), doesn't saturate
   - Unlike sigmoid/tanh (derivatives → 0)

3. **Layer Normalization:**
   - Normalizes inputs → activations don't shrink exponentially
   - Better gradient flow

4. **Better Initialization (Kaiming/Xavier):**
   - Carefully initialize weights to preserve variance
   - Prevents activations from vanishing early

**For Transformers:**
- Residual connections around each sublayer
- Direct path from input to output
- Enables training 100+ layer models

---

#### Q2: Why is AdamW preferred over Adam for training Transformers? What's the difference?

**Answer:**

**Key Difference:** How weight decay (L2 regularization) is applied.

**Adam with L2 Regularization:**
```python
grad = ∇loss + λ * w  # Add weight penalty to gradient
# Then apply Adam's adaptive learning rate to this modified gradient
```

**Problem:** 
- Adaptive learning rate (√v_hat) applies to weight decay term
- For parameters with large historical gradients (large v_hat), weight decay is diminished
- Inconsistent regularization across parameters

**AdamW (Decoupled Weight Decay):**
```python
grad = ∇loss  # Pure gradient, no weight penalty
# Update with Adam
w_new = w - α * grad_adam
# Then apply weight decay separately
w_new = w_new - λ * w
```

**Result:**
- Weight decay is constant across all parameters (not affected by adaptive LR)
- Better regularization
- Improved generalization

**Empirical Results:**
- AdamW achieves better validation performance on Transformers
- Standard in BERT, GPT, T5 training

**Interview Follow-up:**
*Q: Can you show the AdamW update rule mathematically?*

**A:**
```
m_t = β1 * m_{t-1} + (1-β1) * g_t
v_t = β2 * v_{t-1} + (1-β2) * g_t²

w_t = w_{t-1} - α * m_t/√v_t - λ * w_{t-1}
                 ^^^^^^^^^^^^^   ^^^^^^^^^^
                  Adam update    Weight decay (decoupled)
```

 ---

#### Q3: Explain gradient accumulation. When would you use it, and what are the trade-offs?

**Answer:**

**What It Is:**

Accumulating gradients over multiple forward/backward passes before updating weights.

**Code:**
```python
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()  # Gradients accumulate in .grad
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()
```

**When to Use:**

1. **Limited GPU Memory:**
   - Model is 13B params, can only fit batch_size=1
   - Use accumulation_steps=32 → Effective batch size = 32
   - Larger effective batch → Stabler gradients

2. **Matching Published Results:**
   - Paper used batch_size=256, you have smaller GPU
   - Accumulate to match effective batch size

3. **Very Large Models:**
   - Even with optimizations, forward pass fills memory
   - Accumulate over small batches

**Trade-offs:**

**Pros:**
- Larger effective batch size without increasing memory
- Stabler gradient estimates
- Mathematically equivalent to large batch (with LayerNorm)

**Cons:**
- **More Compute:** N accumulation steps = N forward passes before one update
- **Slower Iteration:** Updates happen less frequently  
- **Not Exact with BatchNorm:** BN statistics computed per mini-batch (not full effective batch)

**Best Practices:**
- Divide loss by accumulation_steps (average, not sum)
- Track effective batch size in hyperparameters
- Use when memory-bound, not compute-bound

---

#### Q4: What is the learning rate warmup, and why is it important for Transformer training?

**Answer:**

**Learning Rate Warmup:**

Gradually increasing learning rate from small value to target value over initial training steps.

```python
# Linear warmup
if step < warmup_steps:
    lr = target_lr * (step / warmup_steps)
else:
    lr = target_lr * cosine_schedule(step - warmup_steps)
```

**Why It's Needed:**

1. **Adam's Adaptive Rates are Unstable Early:**
   - Adam maintains moving averages of gradients (m_t, v_t)
   - Early in training, few samples → poor estimates
   - Large learning rate + poor estimates = instability

2. **Random Initialization:**
   - Parameters initialized randomly
   - Initial gradients can be very large
   - Large LR + large gradients = divergence

3. **Empirical Observation:**
   - Transformers trained without warmup often diverge
   - Loss spikes, NaN values

**How It Helps:**

- Small LR initially → small updates despite noisy gradients
- As training progresses and averages stabilize, increase LR
- By end of warmup, optimization is on stable trajectory

**Typical Values:**
- Warmup steps: 4000-10000 steps (or 5-10% of total)
- After warmup: Cosine decay or constant LR

**Interview Follow-up:**
*Q: Could we just use a smaller constant learning rate instead?*

**A:** 
- Yes, but converges slower
- Warmup allows us to use higher peak LR (faster convergence)
- Best of both: stability early + speed later

---

#### Q5: Your model training suddenly shows NaN loss. How do you debug this?

**Answer:**

**Systematic Debugging:**

**Step 1: Locate the NaN**

```python
# Add hooks to find where NaN appears
def check_nan(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN in {module.__class__.__name__}")
        import pdb; pdb.set_trace()

for module in model.modules():
    module.register_forward_hook(check_nan)
```

**Step 2: Common Causes**

1. **Numerical Overflow in Softmax/Exp:**
   ```python
   # Bad: exp(large_number) = inf
   scores = torch.exp(logits)  # If logits > 88, exp(88) = inf
   
   # Good: Subtract max (log-sum-exp trick)
   scores = torch.exp(logits - logits.max(dim=-1, keepdim=True))
   ```

2. **Division by Zero:**
   ```python
   # LayerNorm without epsilon
   std = x.std(dim=-1, keepdim=True)
   normalized = x / std  # If std=0, NaN!
   
   # Fix: Add epsilon
   normalized = x / (std + 1e-5)
   ```

3. **Gradient Explosion:**
   - Check gradient norms:
   ```python
   total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
   if total_norm > 100:
       print(f"Large gradient: {total_norm}")
   ```
   - Fix: Gradient clipping

4. **Learning Rate Too High:**
   - Update too large → parameters explode
   - Fix: Reduce LR, add warmup

5. **Mixed Precision Without Loss Scaling:**
   - FP16 overflow
   - Fix: Use `GradScaler`

**Step 3: Prevention**

```python
# Check for NaN in inputs
assert not torch.isnan(input_tensor).any()

# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Use stable implementations
loss = F.cross_entropy(logits, targets)  # Numerically stable

# Monitor training
wandb.log({"loss": loss.item(), "grad_norm": total_norm})
```

**Production Horror Story:**

"Trained for 3 days, loss=2.5. Day 4, loss=NaN. No checkpoints saved!"

**Prevention:** Save checkpoints frequently, monitor loss in real-time.

---

### Production Challenges

#### Challenge 1: Training Divergence After 10k Steps

**Scenario:**
- Model trains smoothly for 10,000 steps (loss decreasing)
- Suddenly, loss spikes to 50× previous value
- Subsequent steps: loss = NaN

**Root Causes:**

1. **Gradient Explosion (Most Common):**
   - Rare data sample causes large gradients
   - No gradient clipping → weights explode

2. **Learning Rate Schedule Issue:**
   - Sudden LR increase (bug in scheduler)
   - Check: Plot LR over time

3. **Batch with Extreme Values:**
   - Outlier in data (e.g., very long sequence)
   - Causes numerical overflow

**Debugging:**
```python
# before the crash (step 9999), log:
- Gradient norms per layer
- Input statistics (min, max, mean, std)
- Learning rate
- Batch characteristics (seq lengths)
```

**Solution:**
- Add gradient clipping: `max_norm=1.0`
- Outlier filtering in data pipeline
- More aggressive warmup + slower LR increase

---

#### Challenge 2: model.eval() Gives Different Results Than model.train()

**Scenario:**
- Training loss: 2.3
- Eval loss: 4.5 (much worse!)
- Same data, different mode

**Causes:**

**1. Dropout:**
```python
# Training: Randomly drops neurons
output_train = dropout(x, p=0.1)

# Eval: No dropout (deterministic)
output_eval = x
```

Different outputs → different results (expected!).

**2. Layer Norm / Batch Norm Running Stats:**

Usually not an issue with LayerNorm, but if using custom norm:
```python
# If accumulating running stats, difference is expected
```

**3. Bugs:**
- Forgetting to call `model.eval()`
- Mismatch in how loss is computed

**Debugging:**
```python
model.eval()
with torch.no_grad():
    eval_output = model(batch)

# Loss should be higher (dropout was regularizing)
# If loss is WAY higher → bug or severe overfitting
```

**This is Often Normal:** 
- Train mode uses dropout (regularization)
- Eval mode doesn't → model more prone to overfitting on train set
- If eval loss much worse → model is overfitting

**Solution:** More regularization, more data, or reduce model capacity.

---

### Key Takeaways for Interviews

1. **Understand Core Concepts:** Vanishing gradients, optimizers, regularization
2. **Know Trade-offs:** Gradient accumulation (memory vs compute), warmup (stability vs speed)
3. **Debugging Skills:** Systematic approach to NaN, divergence, performance issues
4. **Production Awareness:** Checkpointing, monitoring, numerical stability
5. **Communication:** Explain clearly, use examples, show code when helpful
