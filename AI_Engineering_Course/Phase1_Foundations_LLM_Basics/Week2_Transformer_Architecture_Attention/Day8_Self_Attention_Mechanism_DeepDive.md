# Day 8: Self-Attention Mechanism
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Attention as Matrix Operations: Complete Derivation

**From First Principles:**

Given sequence embeddings X ∈ ℝ^(n×d):

Step 1: **Create Q, K, V**
```
Q = XW_Q where W_Q ∈ ℝ^(d×d_k)
K = XW_K where W_K ∈ ℝ^(d×d_k)
V = XW_V where W_V ∈ ℝ^(d×d_v)
```

Step 2: **Compute Attention Scores**
```
S = QK^T / √d_k ∈ ℝ^(n×n)

S_ij = (Q_i · K_j) / √d_k = Σ_k Q_ik K_jk / √d_k
```

Step 3: **Softmax (row-wise)**
```
A = softmax(S) where A_ij = exp(S_ij) / Σ_k exp(S_ik)
```

Step 4: **Weighted Sum**
```
Output = AV ∈ ℝ^(n×d_v)

Output_i = Σ_j A_ij V_j
```

**Full Computation:**
```
Attention(X) = softmax((XW_Q)(XW_K)^T / √d_k) × (XW_V)
```

**Complexity Analysis:**

```
XW_Q: O(n × d × d_k) = O(nd²) if d_k ≈ d
QK^T: O(n × n × d_k) = O(n²d)
softmax: O(n²)
AV: O(n × n × d_v) = O(n²d)

Total: O(n²d + nd²)

For typical Transformers (n << d):
- Small sequences (n=512, d=768): nd² dominates
- Long sequences (n=4096, d=768): n²d dominates
```

**Memory:**

```
Attention matrix A: O(n²)
Gradients through attention: O(n²)

For n=2048: 4M floats × 4 bytes = 16MB per attention head
For 12 heads: 192MB just for attention!
```

### Gradient Flow Through Attention

**Forward:**
```
Y = Attention(X)
```

**Backward (simplified):**
```
∂L/∂X = (∂L/∂Y) × (∂Y/∂X)
```

**Problem:** Y depends on X through multiple paths (Q, K, V).

**Chain Rule:**
```
∂L/∂X = ∂L/∂Q × ∂Q/∂X + ∂L/∂K × ∂K/∂X + ∂L/∂V × ∂V/∂X
```

**Softmax Gradient:**

```python
# Forward
A = softmax(S)  # A_ij = exp(S_ij) / Σ_k exp(S_ik)

# Backward
∂L/∂S_ij = Σ_k (∂L/∂A_ik × ∂A_ik/∂S_ij)

# Softmax derivative:
∂A_ik/∂S_ij = {
    A_ik(1 - A_ik)  if i = j
    -A_ik A_ij      if i ≠ j
}

# Simplifies to:
∂L/∂S = A ⊙ (∂L/∂A - (∂L/∂A ⊙ A).sum(dim=-1, keepdim=True))
```

**Why Attention Has Good Gradients:**

Unlike RNNs (gradients must flow through time), attention has **direct gradient paths**:

```
RNN: ∂L/∂h_0 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂h_0
     Product of T terms → vanishing!

Attention: ∂L/∂X_i directly from ∂L/∂Y_i
     No long chain → stable gradients!
```

### Why √d_k Scaling: Deep Mathematical Reason

**Variance Analysis:**

Assume Q, K have entries ~ N(0, 1):

```
Dot product: q · k = Σ_{i=1}^{d_k} q_i k_i

E[q · k] = Σ E[q_i k_i] = 0 (q_i, k_i independent)

Var[q · k] = Σ Var[q_i k_i] = Σ E[q_i²]E[k_i²] = d_k
```

Without scaling: Dot products have variance d_k.

**Softmax Saturation:**

```python
# For large variance, most weight goes to max
scores = torch.randn(100, 100) * math.sqrt(512)  # High variance
attn = torch.softmax(scores, dim=-1)

# Typical row:
# [0.98, 0.01, 0.00, 0.00, ..., 0.01]  # Almost one-hot!

# Gradients:
# d(softmax) all goes through max element
# Other elements get tiny gradients → slow learning
```

With 1/√d_k scaling: Variance = 1, softmax in reasonable range.

**Gradient Magnitude:**

```
∂softmax/∂x_i ∝ sotmax[i] × (1 - softmax[i])

If softmax saturated (≈ 1):
∂softmax/∂x ≈ 1 × (1-1) = 0  # Gradient vanishes!

If softmax reasonable (≈ 0.3):
∂softmax/∂x ≈ 0.3 × 0.7 = 0.21  # Good gradient!
```

### Attention Matrix Patterns: What Do They Learn?

**Empirical Observations (BERT, GPT-2):**

**Early Layers (1-4):**
```
Attention pattern: Mostly local (diagonal)

"The quick brown fox" attention:
     The  quick brown  fox
The  [0.7,  0.2,  0.1, 0.0]
quick[0.2,  0.6,  0.2, 0.0]
brown[0.1,  0.2,  0.6, 0.1]
fox  [0.0,  0.1,  0.2, 0.7]

# Attends to adjacent words → Learns syntax
```

**Middle Layers (5-8):**
```
Attention pattern: Phrasal, dependency structures

"New York is" attention:
       New   York   is
New   [0.4,  0.5,  0.1]  # "New" attends to "York"
York  [0.3,  0.6,  0.1]  # Phrasal attention
is    [0.2,  0.2,  0.6]  # Self-attention

# Learns phrase boundaries, dependencies
```

**Late Layers (9-12):**
```
Attention pattern: Long-range semantic

"John went to the store. He bought milk."
              He attends to John (10 words away!)

# Learns coreference, long-range semantics
```

### Attention as Kernel Smoothing

**Connection to Kernel Methods:**

Attention can be viewed as adaptive kernel smoothing:

```
y_i = Σ_j K(x_i, x_j) v_j

where K(x_i, x_j) = softmax(q_i · k_j / √d_k)
```

**Kernel properties:**
- K(x_i, x_j) ≥ 0 (softmax is non-negative)
- Σ_j K(x_i, x_j) = 1 (softmax normalizes)

**Adaptation:** Unlike fixed kernels (Gaussian, etc.), attention kernel is **learned** and **input-dependent**!

### Efficient Attention Computation

**Standard Approach:**
```python
# Compute full attention matrix
scores = Q @ K.T / sqrt(d_k)  # (n, n)
attn = softmax(scores)        # (n, n)
output = attn @ V             # (n, d_v)

Memory: O(n²)
```

**Memory-Efficient (for inference):**
```python
# Process in chunks
chunk_size = 64
for i in range(0, n, chunk_size):
    Q_chunk = Q[i:i+chunk_size]
    scores_chunk = Q_chunk @ K.T / sqrt(d_k)
    attn_chunk = softmax(scores_chunk)
    output[i:i+chunk_size] = attn_chunk @ V

# Still O(n²) ops but reduced peak memory
```

**Flash Attention (GPU-optimized):**

Key idea: Fuse operations, minimize memory transfers.

```python
# Traditional: Many memory reads/writes
Q, K, V from HBM → GPU compute → Store attn → Read attn → Compute output

# Flash Attention: Kernel fusion
Q, K, V from HBM → Compute in shared memory → Output to HBM

# 2-4× speedup for free! (same math, better execution)
```

### Self-Attention vs Cross-Attention

**Self-Attention:**
```python
# All from same sequence
Q = K = V = X

# Each position attends to all positions
attn[i,j] = similarity(X[i], X[j])
```

**Cross-Attention:**
```python
# Q from one sequence (decoder)
# K, V from another sequence (encoder)

Q = decoder_hidden
K = V = encoder_output

# Decoder attends to encoder
attn[i,j] = similarity(decoder[i], encoder[j])
```

**Use Cases:**
- Self-attention: BERT, GPT (encoder/decoder only models)
- Cross-attention: T5, BART (encoder-decoder models)

### Attention Dropout

**Standard Dropout:**
```python
# Drop neurons
output = dropout(activation)
```

**Attention Dropout:**
```python
# Drop attention weights!
attn = softmax(Q @ K.T / sqrt(d_k))
attn = dropout(attn, p=0.1)  # Zero out 10% of attention connections
output = attn @ V
```

**Why?**
- Regularization: Prevents over-reliance on specific attention patterns
- Forces learning multiple attention strategies
- Standard in Transformers (for consistency dropout = 0.1)

### Summary: Deep Insights

1. **Mathematical Elegance:** Attention is differentiable, parallelizable kernel smoothing
2. **Gradient Flow:** Direct paths → No vanishing gradients
3. **Learned Patterns:** Different layers learn different attention (local → global)
4. **Scaling Critical:** √d_k prevents softmax saturation
5. **Quadratic Cost:** O(n²) fundamental limitation → Need efficient variants
6. **Interpretability:** Can visualize learned attention patterns

Self-attention is the foundation of modern NLP - enabling Transformers to replace RNNs/CNNs entirely!
