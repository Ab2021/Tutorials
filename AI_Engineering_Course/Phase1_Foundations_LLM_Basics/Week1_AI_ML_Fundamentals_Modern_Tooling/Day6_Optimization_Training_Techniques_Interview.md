# Day 6: Optimization & Training Techniques
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain the difference between Adam and AdamW. Why is AdamW preferred for training Transformers?

**Answer:**

**Adam (Adaptive Moment Estimation):**

Standard Adam with L2 regularization:
```python
# Add weight decay to gradient
effective_grad = grad + weight_decay × weights

# Then apply Adam's adaptive updates
m = β₁ × m + (1 - β₁) × effective_grad
v = β₂ × v + (1 - β₂) × effective_grad²
weights -= lr × m / (√v + ε)
```

**Problem:**

Weight decay goes through adaptive learning rates!

For parameters with large v (frequent large gradients):
```
Update ∝ (grad + λ×w) / √v
Effective weight decay ∝ λ / √v  # Reduced!
```

For parameters with small v (infrequent/small gradients):
```
Effective weight decay ∝ λ / √tiny  # Huge!
```

**Inconsistent regularization** across parameters.

**AdamW Fix:**

Decouple weight decay from gradient:
```python
# Step 1: Adam without weight decay
m = β₁ × m + (1 - β₁) × grad  # Pure gradient
v = β₂ × v + (1 - β₂) × grad²

# Step 2: Separate weight decay
weights -= lr × (m / (√v + ε) + λ × weights)
                └──────────────┘   └────────┘
                 Adam update      Weight decay
```

**Why Prefer AdamW for Transformers:**

1. **Better Generalization:**
   ```
   BERT fine-tuning:
   - Adam + L2 reg: 88.5% accuracy
   - AdamW: 89.3% accuracy (+0.8%)
   
   GPT-2 pre-training:
   - Adam + L2: Perplexity 23.7
   - AdamW: Perplexity 22.1 (-1.6)
   ```

2. **Consistent Regularization:**
   - All parameters regularized equally
   - Embedding layers (large v) and output layers (small v) both properly regularized

3. **Simpler Hyperparameter Tuning:**
   - weight_decay and lr are decoupled
   - Can tune independently

**Interview Follow-up:**
*Q: What weight_decay value should you use?*

**A:**
- Default: 0.01 (works for most cases)
- Large models (>1B params): 0.1
- Small datasets: 0.001 (less regularization needed)
- Rule: Start with 0.01, increase if overfitting

---

#### Q2: Your model's training loss decreases but validation loss increases after epoch 5. What's happening and how do you fix it?

**Answer:**

**Diagnosis: Overfitting**

Model memorizes training data instead of learning generalizable patterns.

**Classic Overfitting Curve:**
```
Epoch 1-5:
- Train loss: 2.5 → 1.0
- Val loss: 2.4 → 0.9 (good!)

Epoch 6-10:
- Train loss: 1.0 → 0.3 (still decreasing)
- Val loss: 0.9 → 1.2 (increasing!) ← Problem
```

**Root Causes:**

1. **Model too large for dataset:**
   - 10M parameters, 1K training samples
   - Model can memorize all samples

2. **Insufficient regularization**

3. **Training too long**

**Solutions:**

**1. Early Stopping:**

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# Usage
early_stopping = EarlyStopping(patience=5)

for epoch in range(100):
    train(model)
    val_loss = validate(model)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

**2. Increase Regularization:**

```python
# Add/increase dropout
model = Transformer(
    dropout=0.2  # Increase from 0.1
)

# Increase weight decay
optimizer = AdamW(
    model.parameters(),
    weight_decay=0.1  # Increase from 0.01
)

# Add label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**3. Data Augmentation:**

```python
# For NLP: Back-translation, synonym replacement, random deletion
def augment_text(text):
    # Example: Random word deletion
    words = text.split()
    if random.random() < 0.1:  # 10% chance
        idx = random.randint(0, len(words)-1)
        words.pop(idx)
    return ' '.join(words)

# Use during training
augmented_batch = [augment_text(x) for x in batch]
```

**4. Get More Data:**

- Collect more training samples
- Use pre-trained models (transfer learning)
- Semi-supervised learning

**5. Reduce Model Capacity:**

```python
# Fewer layers
model = Transformer(num_layers=6)  # Instead of 12

# Smaller hidden dimension
model = Transformer(d_model=512)  # Instead of 1024

# Parameter sharing
```

**Production Strategy:**

```python
# 1. Monitor train vs val loss
wandb.log({"train_loss": train_loss, "val_loss": val_loss})

# 2. Alert when overfitting detected
if val_loss > best_val_loss * 1.1:  # 10% worse
    alert("Overfitting detected!")

# 3. Automatic early stopping
# 4. Checkpoint best model (by val loss, not train loss!)
if val_loss < best_val_loss:
    torch.save(model.state_dict(), "best_model.pt")
    best_val_loss = val_loss
```

---

#### Q3: Explain gradient clipping. When and how would you tune the max_norm parameter?

**Answer:**

**What It Is:**

Limit maximum gradient norm to prevent exploding gradients.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**How It Works:**

1. Compute total gradient norm:
   ```
   total_norm = √(Σ ||grad_i||²) for all parameters
   ```

2. If total_norm > max_norm:
   ```
   grad_i = grad_i × (max_norm / total_norm)
   ```
   Scale all gradients proportionally.

3. Else: Do nothing.

**When to Use:**

**1. Recurrent Networks (RNNs, LSTMs):**
   - Gradients through time can explode
   - Clipping essential for stability

**2. Very Deep Networks:**
   - Gradients accumulate through layers
   - Without residual connections, can explode

**3. Unstable Training:**
   - Loss spikes to NaN
   - Parameters becoming very large

**How to Tune max_norm:**

**Method 1: Monitor Gradient Norms**

```python
# Log gradient norms without clipping
grad_norms = []

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        
        # Compute norm before clipping
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        grad_norms.append(total_norm)
        
        # Don't clip yet, just observe
        optimizer.step()

# Analyze distribution
import numpy as np
print(f"Median: {np.median(grad_norms)}")
print(f"90th percentile: {np.percentile(grad_norms, 90)}")
print(f"99th percentile: {np.percentile(grad_norms, 99)}")
```

**Set max_norm:**
- If 99th percentile < 5: No clipping needed
- If 99th percentile 5-20: Set max_norm = 1.0 to 2.0
- If 99th percentile > 20: Set max_norm = 1.0, investigate why so large

**Method 2: Grid Search**

```python
max_norm_values = [0.5, 1.0, 2.0, 5.0, None]  # None = no clipping

for max_norm in max_norm_values:
    model = create_model()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
    
    val_loss = evaluate(model)
    print(f"max_norm={max_norm}: val_loss={val_loss}")

# Choose max_norm with best val_loss
```

**Typical Values:**

| Architecture | Recommended max_norm |
|--------------|---------------------|
| Transformers | 1.0 |
| LSTMs/GRUs | 1.0 - 5.0 |
| ResNets | Usually not needed |
| Very large LLMs (>10B) | 0.5 - 1.0 |

**Signs max_norm is Too Low:**

- Very slow convergence
- Gradient norms always at max_norm (constantly clipping)
- Can try increasing

**Signs max_norm is Too High (or not needed):**

- No gradients ever clipped
- No stability issues
- Can remove clipping

**Production Monitoring:**

```python
# Track clipping frequency
clipped_steps = 0
total_steps = 0

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    
    # Measure before clipping
    norm_before = get_total_norm(model.parameters())
    
    # Clip
    norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    total_steps += 1
    if norm_after < norm_before:
        clipped_steps += 1
    
    optimizer.step()

clip_rate = clipped_steps / total_steps
print(f"Clipping rate: {clip_rate:.2%}")

# Ideal: 1-10% (occasional clipping for outliers)
# Too high (>50%): max_norm too low or model issues
# Zero: max_norm might be unnecessary
```

---

#### Q4: You're training a transformer with mixed precision (FP16) and see NaN losses after a few hundred steps. Debug this.

**Answer:**

**Symptom:**

```
Step 100: loss = 2.34
Step 200: loss = 1.87
Step 300: loss = 1.52
Step 350: loss = NaN  ← Problem!
```

**Root Cause: FP16 Numerical Issues**

**Problem 1: Gradient Underflow**

FP16 can't represent values < 6×10^{-8}.

```python
# Typical gradient magnitudes
early_layer_grads: 10^{-7} to 10^{-4}
# In FP16: 10^{-7} → 0 (underflow!)
```

**Problem 2: Loss Overflow**

Large logits → large softmax → large loss → overflow

```python
logits = [100, 95, 90]  # Large values
exp(100) = 2.7×10^{43}  # Overflow in FP16!
```

**Debugging Steps:**

**Step 1: Identify Where NaN Appears**

```python
# Add hooks to find NaN source
def check_nan_hook(module, input, output):
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"NaN detected in {module.__class__.__name__}")
            import pdb; pdb.set_trace()

for module in model.modules():
    module.register_forward_hook(check_nan_hook)
```

**Step 2: Check Loss Scaling**

```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()

# Monitor scale factor
print(f"Current loss scale: {scaler.get_scale()}")

# If scale growing very large (>2^20): Overflow likely
# If scale decreasing rapidly: Underflow likely
```

**Solutions:**

**1. Increase Initial Loss Scale:**

```python
scaler = GradScaler(init_scale=2^10)  # Instead of default 2^16

# Smaller scale → Less overflow risk
# Trade-off: More underflow risk
```

**2. Use BFloat16 Instead:**

```python
# BF16: Same range as FP32, no overflow/underflow!
with autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# No scaler needed
loss.backward()
optimizer.step()
```

**3. Check for Numerical Instability in Model:**

```python
# Common issue: Large embeddings
# Check embedding norms
emb_norm = model.embedding.weight.norm()
if emb_norm > 10:
    print(f"Large embedding norm: {emb_norm}")
    # Solution: Normalize embeddings
    model.embedding.weight.data /= emb_norm
```

**4. Gradient Clipping (Essential with FP16):**

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

**5. Check Learning Rate:**

```python
# Learning rate might be too high for FP16
# Reduce by 0.5×
optimizer = AdamW(model.parameters(), lr=5e-4)  # Instead of 1e-3
```

**Production Prevention:**

```python
class SafeTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        self.nan_count = 0
        self.max_nan_tolerance = 5
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        with autocast():
            loss = self.model(batch)
        
        # Check for NaN
        if torch.isnan(loss):
            self.nan_count += 1
            print(f"NaN detected! Count: {self.nan_count}")
            
            if self.nan_count > self.max_nan_tolerance:
                # Switch to FP32 or BF16
                print("Too many NaNs, switching to BF16")
                self.use_amp = False
            
            return None  # Skip this step
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        
        # Check gradient norms
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        
        if total_norm > 100:
            print(f"Large gradient norm: {total_norm}")
            # Skip step if too large
            return None
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

---

#### Q5: Compare learning rate warmup, cosine decay, and step decay. When would you use each?

**Answer:**

**1. Learning Rate Warmup**

Gradually increase LR from small value to target.

```python
def warmup_lr(step, warmup_steps, target_lr):
    if step < warmup_steps:
        return target_lr * (step / warmup_steps)
    else:
        return target_lr
```

**Why:**
- Adam's adaptive rates unstable initially (few gradient samples)
- Prevents early divergence
- Standard for Transformers

**When to Use:**
- **Always** with Adam/AdamW on Transformers
- Large models
- High learning rates

**Typical warmup duration:**
- Small models: 1000-4000 steps
- Large models (GPT-3): 2000-10000 steps
- Rule: ~1-5% of total steps

**2. Cosine Decay**

```python
def cosine_decay(step, total_steps, lr_max, lr_min=0):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * step / total_steps))
```

**Characteristics:**
- Smooth decrease
- Fast decrease initially
- Slow decrease near end (fine-tuning)
- No sudden drops

**When to Use:**
- Modern Transformers (GPT, BERT, T5)
- When you know total training steps
- Want smooth convergence

**3. Step Decay**

```python
def step_decay(epoch, lr_init, decay_factor=0.1, decay_epochs=[30, 60, 90]):
    lr = lr_init
    for decay_epoch in decay_epochs:
        if epoch >= decay_epoch:
            lr *= decay_factor
    return lr
```

**Characteristics:**
- Sudden drops at specific epochs
- Simple, intuitive
- Can cause temporary instability after drop

**When to Use:**
- CNNs (ResNet, VGG)
- Classification tasks
- When training for fixed number of epochs

**Comparison Table:**

| Schedule | Smoothness | Predictability | Best For | Downsides |
|----------|------------|----------------|----------|-----------|
| Warmup | N/A (increasing) | High | Stability | Must combine with decay |
| Cosine | Very smooth | Medium (needs total steps) | Transformers | Requires knowing total steps |
| Step | Sudden drops | High | CNNs | Can cause instability |

**Recommended Combinations:**

**For Transformers:**
```python
# Warmup + Cosine
warmup_steps = 10000
total_steps = 500000

if step < warmup_steps:
    lr = target_lr * (step / warmup_steps)
else:
    lr = 0.5 * target_lr * (1 + cos(π * (step - warmup_steps) / (total_steps - warmup_steps)))
```

**For CNNs:**
```python
# No warmup, step decay
lr = 0.1  # Initial
# Decay by 10× at epochs 30, 60, 90
```

**For Fine-tuning:**
```python
# Small warmup + linear decay
warmup_steps = 100  # Short warmup
total_steps = 10000

if step < warmup_steps:
    lr = target_lr * (step / warmup_steps)
else:
    lr = target_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
```

---

### Production Challenges

**Challenge: Training Hangs After Several Hours**

**Symptom:**
- Training progresses normally for 6 hours
- Suddenly stops (no error, no progress)
- GPU utilization drops to 0%

**Root Causes:**

1. **Deadlock in DataLoader:**
   - Multi-process data loading
   - One worker hangs

2. **NaN Propagation:**
   - NaN in forward pass
   - Backward hangs (waiting for non-existent gradients)

3. **Distributed Training Issue:**
   - One GPU diverges
   - Collective operations (all_reduce) wait forever

**Solutions:**
- Timeout DataLoader: `torch.utils.data.DataLoader(..., timeout=60)`
- NaN checks: Validate loss each step
- Distributed timeouts: `torch.distributed.init_process_group(..., timeout=timedelta(seconds=1800))`

---

### Key Takeaways for Interviews

1. **Understand optimizer internals**: Adam vs AdamW mathematical difference
2. **Know when to use what**: Warmup (always for Transformers), cosine (smooth), step (CNNs)
3. **Debug systematically**: NaN → check FP16, gradients, loss scale
4. **Production awareness**: Monitoring, fallbacks, graceful degradation
5. **Regularization trade-offs**: More data > better architecture > more regularization
