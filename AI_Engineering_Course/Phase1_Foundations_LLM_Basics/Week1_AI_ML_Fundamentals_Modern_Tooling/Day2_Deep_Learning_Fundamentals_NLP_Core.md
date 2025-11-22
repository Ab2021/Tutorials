# Day 2: Deep Learning Fundamentals for NLP
## Core Concepts & Theory

### Neural Networks for Language: Foundation

Unlike computer vision where Conv

Nets excel, language poses unique challenges:
- **Variable Length**: Sentences vary from 5 to 500+ tokens
- **Sequential Nature**: Order matters ("dog bites man" ≠ "man bites dog")  
- **Long-Range Dependencies**: "The keys that were on the  table are **lost**" (subject-verb agreement across 6 tokens)

### Feed-Forward Networks: The Building Block

**Architecture:**
```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()  # Modern choice over ReLU
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)        # (batch, seq_len, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)        # (batch, seq_len, d_model)
        return self.dropout(x)
```

**Key Concepts:**

1. **Position-wise**: Applied independently to each position
2. **Expansion-Contraction**: d_model → d_ff (expand) → d_model (contract)
   - Typically d_ff = 4 × d_model
   - Provides model capacity to learn complex transformations
3. **GELU vs ReLU**: 
   - ReLU: `max(0, x)` - hard cutoff at 0
   - GELU: Smooth, probabilistic (better gradients) - standard in Transformers

### Backpropagation Through Time (BPTT)

For sequential models, we must propagate gradients through the sequence.

**Example: Simple RNN**
```
h_0 → h_1 → h_2 → h_3
      ↓     ↓     ↓
      y_1   y_2   y_3
```

Loss at each step: L = L_1 + L_2 + L_3

To compute ∂L/∂W, we need ∂L_3/∂h_1 (gradient flows backward through time).

**The Vanishing Gradient Problem:**

Gradient through T time steps:
```
∂L_T/∂h_0 = ∂L_T/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂h_0
```

Each ∂h_t/∂h_{t-1} involves weight matrix W and activation derivative.

If ||∂h_t/∂h_{t-1}|| < 1, after T steps: (0.9)^50 ≈ 0.005 (gradient vanishes!)

**Solutions:**
1. **LSTM/GRU**: Gating mechanisms maintain gradient flow
2. **Residual Connections**: Skip connections (x + f(x))
3. **Better Initialization**: Careful weight initialization
4. **Gradient Clipping**: Prevent exploding gradients

### Optimization Algorithms

**Stochastic Gradient Descent (SGD):**
```
w_{t+1} = w_t - η × ∇L(w_t)
```

Problems:
- Same learning rate for all parameters
- No notion of curvature (second derivatives)
- Can oscillate in ravines

**Adam (Adaptive Moment Estimation):**

Maintains two moving averages:
- m_t: First moment (mean of gradients)
- v_t: Second moment (variance of gradients)

```python
# Simplified Adam
m_t = β1 × m_{t-1} + (1-β1) × g_t        # Momentum
v_t = β2 × v_{t-1} + (1-β2) × g_t²       # RMSProp-like

m_hat = m_t / (1 - β1^t)                 # Bias correction
v_hat = v_t / (1 - β2^t)

w_t = w_{t-1} - η × m_hat / (√v_hat + ε)
```

**Why  Adam for LLMs:**
- Adaptive learning rates per parameter
- Handles sparse gradients (embeddings)
- Generally stable

**AdamW (Adam with Weight Decay):**

Standard L2 regularization with Adam behaves unexpectedly.

AdamW separates weight decay:
```python
w_t = w_{t-1} - η × m_hat / (√v_hat + ε) - λ × w_{t-1}
```

**Result**: Better generalization, standard for Transformers.

### Loss Functions for Language Modeling

**Causal Language Modeling (GPT):**

Predict next token given previous tokens.

```python
# Input:  "The cat sat on"
# Target: "cat sat on the"

logits = model(input_ids)  # (batch, seq_len, vocab_size)
shift_logits = logits[:, :-1, :]  # Predict next token
shift_labels = labels[:, 1:]      # Shifted targets

loss = F.cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1),
    ignore_index=pad_token_id
)
```

**Cross-Entropy Loss:**

For token at position i with true class y_i:

L = -log P(y_i | context) = -log (e^{z_{y_i}} / Σ_j e^{z_j})

Where z are logits.

**Masked Language Modeling (BERT):**

Randomly mask 15% of tokens, predict masked tokens.

```python
# Input:  "The [MASK] sat on the"
# Target: Predict "cat" at masked position

# Only compute loss on masked positions
loss = F.cross_entropy(
    logits[mask_positions],
    labels[mask_positions]
)
```

### Regularization Techniques

**1. Dropout:**

Randomly zero neurons during training (with probability p).

```python
# During training: scale by 1/(1-p)
output = dropout(x, p=0.1)  # 10% neurons → 0, rest scaled by 1/0.9

# During inference: no dropout (deterministic)
```

**Why it works:**
- Prevents co-adaptation (neurons depending on each other)
- Ensemble effect (train 2^n sub-networks)
- For Transformers: typically 0.1 (10%)

**2. Layer Normalization:**

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def  forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)   # Average over features
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

**Why for NLP (vs BatchNorm for Vision):**
- Sequence lengths vary → batch statistics unstable
- LayerNorm normalizes over features (stable)
- Enables larger learning rates

**3. Weight Decay:**

L2 regularization: Loss + λ × ||W||²

Penalizes large weights, encourages simpler models.

**Typical values:**
- AdamW: weight_decay = 0.01 to 0.1

### Training Dynamics

**Learning Rate Schedules:**

**1. Warmup:**

Start with small LR, gradually increase to target.

```python
# Linear warmup over warmup_steps
lr = target_lr * (step / warmup_steps)  # step < warmup_steps
```

**Why:** 
- Adam's adaptive rates are unstable initially (few gradient samples)
- Warmup stabilizes early training
- Typical warmup: 1-10% of total steps

**2. Cosine Annealing:**

After warmup, decrease LR following cosine curve.

```python
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π × step / total_steps))
```

**Why:**
- Smooth decrease (no sudden drops)
- Better final convergence than step decay

**3. Inverse Square Root:**

```python
lr = lr_base × min(step^{-0.5}, step × warmup_steps^{-1.5})
```

Used in original Transformer paper.

### Gradient Accumulation

**Problem:** Large models + limited GPU memory → small batch sizes → noisy gradients

**Solution:** Accumulate gradients over multiple forward passes.

```python
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # Average loss
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()       # Update weights
        optimizer.zero_grad()  # Reset gradients
```

**Effective batch size** = actual_batch × accumulation_steps

**Trade-off:**
- More gradient accumulation steps → More compute, same memory
- Larger effective batch → Stabler gradients, but diminishing returns

### Gradient Clipping

**Problem:** Exploding gradients (gradient norm >> 1) → unstable training

**Solution:** Clip gradient norm to maximum value.

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**How it works:**

If ||g|| > max_norm:
  g = g × (max_norm / ||g||)

**Why for NLP:**
- Recurrent connections can cause gradient explosion
- Transformer residual paths can accumulate
- Common max_norm: 1.0 to 5.0

### Evaluation Metrics

**Perplexity:**

Perplexity = exp(cross_entropy_loss)

Lower perplexity = better model.

Intuition: "On average, how many words could go next?"

- Perplexity 10: Model is choosing among ~10 words  
- Perplexity 100: Model is choosing among ~100 words

**Limitations:**
- Only measures likelihood, not quality
- Can't compare across different tokenizations

### Challenges in Deep Learning for NLP

**1. Long-Range Dependencies:**

RNNs struggle with dependencies > 100 tokens apart.

**Solution:** Transformers with attention (direct connections between all positions)

**2. Computational Cost:**

Language models process billion-token datasets × billion parameters.

**Optimizations:**
- Mixed precision (FP16/BF16)
- Efficient attention (Flash Attention)
- Model parallelism

**3. Data Efficiency:**

Unlike vision (millions of labeled images), NLP often has limited labeled data.

**Solutions:**
- Pre-training on unlabeled text (self-supervised)
- Transfer learning (fine-tune pre-trained models)
- Few-shot learning (GPT-3 style)

**4. Catastrophic Forgetting:**

Fine-tuning on Task B makes model forget Task A.

**Solutions:**
- Elastic Weight Consolidation
- Multi-task learning
- Parameter-efficient fine-tuning (LoRA)

### Summary

Deep learning for NLP builds on:
- **Feed-forward layers** for transformations
- **Backpropagation** for learning
- **Advanced optimizers** (AdamW) for stable training
- **Regularization** (dropout, layer norm, weight decay)
- **Careful training** (warmup, scheduling, clipping)

Modern LLMs (GPT, BERT, T5) combine these foundations with Transformer architecture to achieve state-of-the-art performance.
