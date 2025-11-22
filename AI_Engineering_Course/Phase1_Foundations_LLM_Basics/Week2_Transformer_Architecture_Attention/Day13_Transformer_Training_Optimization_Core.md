# Day 13: Transformer Training & Optimization
## Core Concepts & Theory

### The Training Loop Anatomy

Training a Transformer is notoriously unstable compared to CNNs or RNNs. It requires a specific recipe of initialization, normalization, and optimization.

**Standard Pipeline:**
1.  **Tokenization:** Convert text to integers.
2.  **Forward Pass:** Embeddings -> Layers -> Logits.
3.  **Loss Calculation:** Cross Entropy (usually with label smoothing).
4.  **Backward Pass:** Compute gradients.
5.  **Optimization:** Update weights (AdamW + Scheduler).

### 1. Initialization Schemes

Proper initialization is critical to prevent vanishing/exploding gradients in deep Transformers.

**Xavier (Glorot) Initialization:**
- Standard for tanh/sigmoid activations.
- $W \sim U(-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}})$

**He Initialization:**
- Standard for ReLU/GELU (used in Transformers).
- $W \sim N(0, \sqrt{2/n_{in}})$

**Scaled Initialization (GPT-2/3):**
- For residual layers, scale weights by $1/\sqrt{N}$ where $N$ is the number of residual layers.
- **Why?** As the network gets deeper, the variance of the residual path increases. Scaling down keeps the variance constant at initialization.

### 2. Learning Rate Schedulers

Transformers do not train well with a constant learning rate.

**The "Warmup" Phase:**
- Linearly increase LR from 0 to `max_lr` over the first few thousand steps.
- **Why?** Early gradients are noisy because parameters are random. Large updates early on can push parameters into bad local minima or cause divergence.

**The "Decay" Phase:**
- After warmup, decay the LR.
- **Cosine Decay:** Smooth curve down to `min_lr`. Most popular today.
- **Linear Decay:** Simple linear drop.
- **Inverse Square Root:** Used in original Transformer paper ($d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$).

### 3. Optimization Algorithms

**AdamW (Adam with Weight Decay):**
- The gold standard for LLMs.
- **Adam:** Adaptive Moment Estimation. Keeps running average of gradients ($m_t$) and squared gradients ($v_t$).
- **Weight Decay Fix:** In standard Adam, L2 regularization is added to the gradient. In AdamW, weight decay is decoupled and applied directly to the weights. This yields better generalization.

**Hyperparameters:**
- $\beta_1 = 0.9$
- $\beta_2 = 0.95$ (Standard is 0.999, but 0.95 is better for LLMs to handle sparse gradients)
- $\epsilon = 1e-8$
- Weight decay $\approx 0.1$

### 4. Regularization Techniques

**Dropout:**
- Applied to:
    - Output of each sub-layer (before residual add).
    - Embedding sums.
    - Attention weights.
- Rate: Typically 0.1.

**Label Smoothing:**
- Instead of hard targets (1 for correct, 0 for others), use soft targets (e.g., 0.9 for correct, 0.1 distributed among others).
- Prevents the model from becoming over-confident and improves calibration.

**Weight Decay:**
- Penalizes large weights to prevent overfitting.

### 5. Gradient Clipping

- **Problem:** Exploding gradients, especially in deep networks.
- **Solution:** Clip the global norm of the gradient vector to a maximum value (usually 1.0).
- `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

### Code: Complete Training Step

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_step(model, batch, optimizer, scheduler, criterion, device):
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # 1. Forward
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # 2. Loss
    # Reshape for CrossEntropy: (batch*seq, vocab)
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # 3. Backward
    optimizer.zero_grad()
    loss.backward()
    
    # 4. Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 5. Update
    optimizer.step()
    scheduler.step()
    
    return loss.item()

# Setup
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
```

### Summary Checklist
- [ ] **AdamW** optimizer used?
- [ ] **Warmup** steps included?
- [ ] **Gradient Clipping** enabled?
- [ ] **Weight Decay** applied correctly?
- [ ] **Pre-LN** architecture used for stability?

### Next Steps
In the Deep Dive, we will explore advanced optimization techniques like Gradient Accumulation, Mixed Precision (AMP), and ZeRO that allow training large models on limited hardware.
