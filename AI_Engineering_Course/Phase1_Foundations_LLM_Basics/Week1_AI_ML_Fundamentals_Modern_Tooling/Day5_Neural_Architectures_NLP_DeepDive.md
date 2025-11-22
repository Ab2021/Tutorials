# Day 5: Neural Architectures for NLP  
## Deep Dive - Internal Mechanics & Advanced Reasoning

### LSTM: Solving Vanishing Gradients Through Gating

**The Gradient Flow Problem in RNNs:**

Standard RNN gradient:
```
∂L/∂W = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂W
```

Each ∂h_t/∂h_{t-1} involves:
```
∂h_t/∂h_{t-1} = tanh'(W_{hh}h_{t-1} + W_{xh}x_t) × W_{hh}
```

For long sequences (T=100), this product vanishes or explodes.

**LSTM's Solution: Additive Cell State**

LSTM maintains two paths for gradients:
1. **Multiplicative path**: Through hidden states (like RNN)
2. **Additive path**: Through cell states (new!)

**Cell State Gradient:**

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

∂c_t/∂c_{t-1} = f_t  # Element-wise, no matrix multiply!
```

**Key Insight:** Forget gate f_t ∈ (0,1) controls gradient flow.

If f_t ≈ 1 (preserve memory):
```
∂c_T/∂c_0 = f_T ⊙ f_{T-1} ⊙ ... ⊙ f_1 ≈ 1
```

Gradient preserved!

If f_t ≈ 0 (forget):
```
∂c_T/∂c_0 ≈ 0
```

Gradient blocked (intentionally, to forget irrelevant past).

**Comparison:**

```python
# RNN gradient (T=100 steps)
grad_rnn = (W_hh)^100  # Matrix power → vanishes/explodes

# LSTM gradient
grad_lstm = f_100 * f_99 * ... * f_1  # Element-wise → stable!
```

**Mathematical Proof:**

Let's prove gradient doesn't vanish if forget gates ≈ 1.

```
L = loss(h_T)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

∂L/∂c_0 = ∂L/∂c_T × ∂c_T/∂c_{T-1} × ... × ∂c_1/∂c_0

Since ∂c_t/∂c_{t-1} = diag(f_t):

∂L/∂c_0 = ∂L/∂c_T × diag(f_T) × diag(f_{T-1}) × ... × diag(f_1)
        = ∂L/∂c_T × diag(f_T ⊙ f_{T-1} ⊙ ... ⊙ f_1)

If f_t ≈ 1 for all t:
∂L/∂c_0 ≈ ∂L/∂c_T  # Gradient preserved!
```

### Peephole Connections in LSTM

Standard LSTM gates don't directly see cell state. **Peephole LSTMs** add connections:

```python
class PeepholeLSTM Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xi = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_ci = nn.Parameter(torch.zeros(hidden_size))  # Peephole
        
        # Similar for f, o gates
    
    def forward(self, x, h, c):
        # Input gate with peephole
        i = torch.sigmoid(
            self.W_xi(x) + self.W_hi(h) + self.W_ci * c  # c_{t-1} peephole
        )
        
        # Forget gate with peephole
        f = torch.sigmoid(
            self.W_xf(x) + self.W_hf(h) + self.W_cf * c
        )
        
        # Cell update
        g = torch.tanh(self.W_xg(x) + self.W_hg(h))
        c_new = f * c + i * g
        
        # Output gate with peephole to NEW cell state
        o = torch.sigmoid(
            self.W_xo(x) + self.W_ho(h) + self.W_co * c_new  # c_t peephole
        )
        
        h_new = o * torch.tanh(c_new)
        return h_new, c_new
```

**Why Peepholes?**

Allow gates to directly inspect cell state → better control of what to remember/forget.

Empirically: Marginal improvement (~1-2%) on some tasks.

### GRU: Simplified Gating Mechanism

**Design Philosophy:** Combine forget and input gates into single update gate.

**Intuition:**

LSTM has separate:
- Forget gate: How much to forget from c_{t-1}
- Input gate: How much to add from new input

GRU combines: If we forget less, we add more (and vice versa).

**Update Gate:**

```
z_t = σ(W_z [h_{t-1}, x_t])

h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
     ^^^^^^^^^^^^^        ^^^^^^^^^^^
     New content          Old content

# If z_t = 1: keep all old (forget new)
# If z_t = 0: use all new (forget old)
```

**Reset Gate:**

Controls how much past hidden state to use when computing candidate:

```
r_t = σ(W_r [h_{t-1}, x_t])

h̃_t = tanh(W [r_t ⊙ h_{t-1}, x_t])
            ^^^^^^^^^^^^^^^
            Selectively reset past
```

**GRU vs LSTM Parameter Count:**

```
LSTM:
- 4 gates × (W_x + W_h) = 4 × (input×hidden + hidden×hidden)
- Parameters: 4 × (d_in × d_h + d_h × d_h)

GRU:
- 3 gates × (W_x + W_h) = 3 × (d_in × d_h + d_h × d_h)  
- Parameters: 3 × (d_in × d_h + d_h × d_h)

Reduction: 25% fewer parameters
```

**Empirical Comparison:**

Studies show:
- LSTM: Slightly better on tasks requiring precise timing
- GRU: Slightly better on tasks with less temporal structure
- Difference: Usually < 2%
- GRU trains 15-20% faster (fewer parameters)

### Bidirectional RNNs: Information Flow Analysis

**Forward RNN Information:**

```
h_t^→ = f(h_{t-1}^→, x_t)
```

Contains information from: x_1, x_2, ..., x_t (past + present)

**Backward RNN Information:**

```
h_t^← = f(h_{t+1}^←, x_t)
```

Contains information from: x_T, x_{T-1}, ..., x_t (future + present)

**Combined Representation:**

```
h_t = [h_t^→; h_t^←]
```

Contains: Past + present + future context!

**Why This Helps for NER:**

Example: "Apple announced new iPhone"

For predicting "Apple" = ORGANIZATION:
- Forward only: Sees "Apple" (ambiguous: fruit vs company?)
- Bidirectional: Sees "Apple" + "announced" + "iPhone" → Clearly company!

**Trade-off:**

Cannot use for:
- Language modeling (would see answer!)
- Real-time applications (need to wait for full sequence)

### CNN Text Classification: Receptive Field Analysis

**1D Convolution Mathematics:**

For input x of length L, filter w of size k:

```
output[i] = Σ_{j=0}^{k-1} w[j] × x[i+j] + b
```

**Receptive Field:**

A filter of size k sees k consecutive tokens.

Example:
- k=3: Sees trigrams ("not very good")
- k=5: Sees 5-grams ("this movie was really great")

**Stacking Convolutions Increases Receptive Field:**

```
Layer 1: Filter size 3 → Sees 3 tokens
Layer 2: Filter size 3 → Sees 3×3 = 5 tokens (dilated)  
Layer 3: Filter size 3 → Sees 5+2 = 7 tokens
```

Exponential growth with depth!

**Comparison with RNN:**

```
CNN (3 layers, k=3):
- Receptive field: 7 tokens
- Computation: O(L) - parallel
- Gradient path: 3 layers (short)

RNN (same depth):
- Receptive field: Entire sequence
- Computation: O(L) - sequential
- Gradient path: L steps (long, risk vanishing)
```

**Why CNNs Work for Text:**

N-gram patterns are compositional:

```
"not good" → negative (bigram)
"not very good" → still negative (trigram)
"not particularly good" → CNN with k=3 might miss!
```

Best for: Local patterns (sentiment, topic classification)
Worse for: Long dependencies (QA, translation)

### Attention Mechanism: Computational Details

**Attention Score Computation:**

Given:
- Query q (decoder hidden state): (d_h,)
- Keys K (encoder hidden states): (seq_len, d_h)

Compute scores:

```python
# Additive (Bahdanau) attention
scores = v^T tanh(W_q q + W_k K)  # (seq_len,)

# Multiplicative (Luong) attention  
scores = q^T K^T  # (seq_len,)

# Scaled dot-product (Transformer)
scores = (q^T K^T) / sqrt(d_k)  # (seq_len,)
```

**Why Scaling Factor?**

Dot product grows with dimensionality:

```
If q, k ~ N(0, 1) (standard normal):

E[q · k] = 0
Var[q · k] = d_k

As d_k increases, variance increases → softmax saturates!
```

**Saturation Problem:**

```python
scores = [10, 9, 8]  # Not too different
softmax(scores) = [0.66, 0.24, 0.09]  # Reasonable distribution

scores = [100, 90, 80]  # Scaled up
softmax(scores) = [0.9999, 0.0001, 0.0000]  # Saturated!
```

**Solution: Scale by √d_k**

```
scores_scaled = scores / sqrt(d_k)
```

Keeps variance constant regardless of dimension!

**Attention Complexity:**

Computing attention for sequence of length L:

```
Scores: Q @ K^T  → O(L^2 × d_k)
Softmax: Over L elements → O(L)
Weighted sum: α @ V → O(L × d_v)

Total: O(L^2 × d) - Quadratic in sequence length!
```

This is why long-context Transformers are expensive.

### Gradient Clipping: Theory and Practice

**Why Gradients Explode:**

In RNNs, gradient involves:

```
∂L/∂W = Σ_t ∂L/∂h_t × ( ∏_{i=t}^T ∂h_i/∂h_{i-1}) × ∂h_t/∂W
```

If ||∂h_i/∂h_{i-1}|| > 1 (e.g., ||W_hh|| > 1):

```
||∏_{i=t}^T ∂h_i/∂h_{i-1}|| ≈ ||W_hh||^{T-t} → ∞
```

**Gradient Clipping:**

```python
def clip_grad_norm(parameters, max_norm):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm
```

**Effect:**

```
Before: ||g|| = 100
After (max_norm=1): g' = g × (1/100) → ||g'|| = 1
```

Scales gradient to maximum allowed norm.

**Choosing max_norm:**

Empirical rule:
- RNNs/LSTMs: 1.0 to 5.0
- Transformers: 1.0 (less prone to explosion)

Monitor gradient norms during training to tune!

### Summary: Why Transformers Replaced RNNs

**RNN/LSTM Limitations:**

1. **Sequential**: h_t+1 depends on h_t → No parallelization
2. **Vanishing gradients**: Even with LSTMs, very long sequences (1000+ tokens) problematic
3. **Limited context**: Bottleneck in fixed-size hidden state

**Transformer Advantages:**

1. **Parallel**: All positions processed simultaneously
2. **Direct connections**: Attention connects any two positions directly
3. **No vanishing**: Residual connections + layer norm
4. **Scalable**: Can train on modern hardware (TPUs, GPUs) efficiently

**Evolution:**

```
2014: LSTM state-of-the-art
2017: Transformer introduced ("Attention Is All You Need")
2018: BERT (Transformer encoder) dominates NLP
2019: GPT-2 (Transformer decoder) shows scaling potential
2020+: Pure Transformers (GPT-3, T5, BERT variants)
```

RNNs still used for:
- Edge devices (smaller models)
- Streaming applications (online processing)
- Some specialized tasks

But Transformers dominate modern NLP!
