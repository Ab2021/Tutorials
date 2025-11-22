# Day 6: Optimization & Training Techniques
## Core Concepts & Theory

### Learning Rate: The Most Important Hyperparameter

The learning rate η controls how much we update parameters based on gradients:

```
w_{t+1} = w_t - η × ∇L(w_t)
```

**Too small**: Slow convergence, may get stuck
**Too large**: Overshooting, divergence, oscillation
**Just right**: Fast, stable convergence

### Learning Rate Schedules

**1. Constant Learning Rate**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# lr = 1e-3 throughout training
```

**Pros**: Simple
**Cons**: May not converge to optimal solution

**2. Step Decay**

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=10,  # Decay every 10 epochs
    gamma=0.1      # Multiply lr by 0.1
)

# Epoch 0-9: lr = 1e-3
# Epoch 10-19: lr = 1e-4  
# Epoch 20-29: lr = 1e-5
```

**3. Exponential Decay**

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95  # lr *= 0.95 each epoch
)

# Smooth exponential decrease
```

**4. Cosine Annealing**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # Period
    eta_min=1e-6  # Minimum lr
)

# lr_t = eta_min + 0.5 × (eta_max - eta_min) × (1 + cos(π × t / T_max))
```

**Why cosine?**
- Smooth decrease (no sudden drops)
- Larger steps initially → Fast convergence
- Smaller steps later → Fine-tuning
- Standard for Transformers

**5. Warm Restarts (SGDR)**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,     # First restart after 10 epochs
    T_mult=2  ,  # Double period each restart
    eta_min=1e-6
)

# Periodic restarts help escape local minima
```

**6. Warmup + Decay**

```python
def get_lr(step, warmup_steps, d_model):
    """Learning rate from 'Attention Is All You Need'"""
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))
    return d_model ** (-0.5) * min(arg1, arg2)

# Increases linearly during warmup
# Decreases as inverse square root after
```

**Typical Transformer Schedule:**

```python
class TransformerLR:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### Advanced Optimizers

**SGD with Momentum**

```python
class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = {p: torch.zeros_like(p) for p in params}
    
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            # Update velocity
            self.velocity[p] = (
                self.momentum * self.velocity[p] - self.lr * p.grad
            )
            
            # Update parameters
            p.data += self.velocity[p]
```

**Intuition**: Rolling ball accumulates momentum going downhill.

**AdaGrad (Adaptive Gradient)**

```python
class AdaGrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.sum_squared_grads = {p: torch.zeros_like(p) for p in params}
    
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            # Accumulate squared gradients
            self.sum_squared_grads[p] += p.grad ** 2
            
            # Adaptive learning rate
            adapted_lr = self.lr / (torch.sqrt(self.sum_squared_grads[p]) + self.eps)
            
            # Update
            p.data -= adapted_lr * p.grad
```

**Problem**: Learning rate monotonically decreases → stops learning.

**RMSProp (Root Mean Square Propagation)**

```python
class RMSProp:
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.alpha = alpha  # Decay rate
        self.eps = eps
        self.squared_avg = {p: torch.zeros_like(p) for p in params}
    
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            # Exponential moving average of squared gradients
            self.squared_avg[p] = (
                self.alpha * self.squared_avg[p] + (1 - self.alpha) * p.grad ** 2
            )
            
            # Adaptive learning rate
            adapted_lr = self.lr / (torch.sqrt(self.squared_avg[p]) + self.eps)
            
            # Update
            p.data -= adapted_lr * p.grad
```

**Fix**: Uses moving average instead of cumulative sum.

**Adam (Adaptive Moment Estimation)**

Combines momentum + RMSProp:

```python
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {p: torch.zeros_like(p) for p in params}  # First moment
        self.v = {p: torch.zeros_like(p) for p in params}  # Second moment
        self.t = 0
    
    def step(self):
        self.t += 1
        
        for p in self.params:
            if p.grad is None:
                continue
            
            # Update biased first moment
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad
            
            # Update biased second moment
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * p.grad ** 2
            
            # Bias correction
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

**AdamW (Adam with Weight Decay)**

```python
class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0.01):
        # Same as Adam
        self.weight_decay = weight_decay
    
    def step(self):
        self.t += 1
        
        for p in self.params:
            if p.grad is None:
                continue
            
            # Adam update
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * p.grad ** 2
            
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            
            # Update with DECOUPLED weight decay
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            p.data -= self.lr * self.weight_decay * p.data  # Separate!
```

**Key difference**: Weight decay not affected by adaptive learning rate.

### Regularization Techniques

**1. L2 Regularization (Weight Decay)**

```python
loss = criterion(output, target) + lambda * sum(p.pow(2).sum() for p in model.parameters())
```

**Effect**: Penalizes large weights → simpler models → better generalization.

**2. Dropout**

```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            # Create mask
            mask = (torch.rand_like(x) > self.p).float()
            # Scale to maintain expected value
            return x * mask / (1 - self.p)
        else:
            return x  # No dropout during inference
```

**Why scaling?**
- Training: E[output] = input × (1-p)
- Without scaling at inference: E[output] = input  
- Mismatch!

**Solution**: Scale by 1/(1-p) during training OR scale by (1-p) during inference.

Modern PyTorch: Scales during training.

**3. DropConnect**

Dropout on weights instead of activations:

```python
# Randomly zero weights during training
W_dropped = W * mask
output = input @ W_dropped
```

**4. Label Smoothing**

Instead of one-hot [0, 0, 1, 0]:
```python
smoothed = [0.05, 0.05, 0.85, 0.05]  # ε = 0.05

loss = -sum(smoothed * log(predictions))
```

**Effect**: Less confident predictions → Better calibration.

**5. Gradient Noise**

Add noise to gradients:

```python
for p in model.parameters():
    if p.grad is not None:
        noise = torch.randn_like(p.grad) * sigma
        p.grad += noise
```

**Effect**: Helps escape sharp minima → Better generalization.

### Batch Normalization

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (not learned)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        # x: (batch, features)
        
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var
            )
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var
        
        # Normalization
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        
        return out
```

**Why it works:**
- Normalizes activations → stable training
- Reduces internal covariate shift
- Allows higher learning rates

**Problem for NLP**: Batch statistics unstable for variable-length sequences.

### Layer Normalization (Standard for Transformers)

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        # Normalize over hidden dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta
```

**Key difference**: Normalizes over features, not batch.

### Mixed Precision Training

**FP32 (Full Precision):**
- 32 bits per number
- High precision, slow

**FP16 (Half Precision):**
- 16 bits per number
- 2× faster, 2× less memory
- Risk: Gradients underflow/overflow

**Automatic Mixed Precision (AMP):**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward in FP16
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    
    # Unscale gradients, step optimizer
    scaler.step(optimizer)
    scaler.update()
```

**How it works:**
1. Forward/backward in FP16 (fast)
2. Scale loss by large number (e.g., 2^16) before backward
3. Gradients scaled up → avoid underflow
4. Unscale before optimizer step
5. Adjust scale factor if overflow detected

**BFloat16 (Brain Float16):**
- Same exponent range as FP32 (no scaling needed!)
- Less precision (7 bits vs 10 bits mantissa)
- Supported on A100, TPU, newer GPUs

```python
with autocast(dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, target)

loss.backward()  # No scaling needed!
optimizer.step()
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward
    output = model(batch)
    loss = criterion(output, target)
    
    # Normalize loss by accumulation steps
    loss = loss / accumulation_steps
    
    # Backward (gradients accumulate)
    loss.backward()
    
    # Update every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Effective batch size** = batch_size × accumulation_steps

### Summary

**Learning Rates:**
- Warmup + cosine decay (Transformers)
- Step decay (CNNs)
- Adaptive methods (Adam, AdamW)

**Regularization:**
- Weight decay (L2)
- Dropout (0.1 for Transformers)
- Label smoothing

**Normalization:**
- LayerNorm (NLP)
- BatchNorm (CV)

**Mixed Precision:**
- FP16 with scaling (older GPUs)
- BF16 (A100+)

**Modern Training Recipe (Transformers):**
```python
model = Transformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            output = model(batch)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    
    scheduler.step()
```

This combines: AdamW + warmup/decay + mixed precision + gradient clipping.
