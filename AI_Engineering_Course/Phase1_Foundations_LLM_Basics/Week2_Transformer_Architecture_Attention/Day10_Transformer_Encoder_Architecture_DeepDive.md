# Day 10: Transformer Encoder Architecture
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Residual Connections: Gradient Flow Analysis

**Mathematical Analysis:**

Without residuals:
```
x_{l+1} = F_l(x_l)

Gradient:
∂L/∂x_0 = ∂L/∂x_L × ∏_{l=0}^{L-1} ∂F_l/∂x_l
```

If ||∂F_l/∂x_l|| < 1: Product vanishes
If ||∂F_l/∂x_l|| > 1: Product explodes

With residuals:
```
x_{l+1} = x_l + F_l(x_l)

∂x_{l+1}/∂x_l = I + ∂F_l/∂x_l

Gradient:
∂L/∂x_0 = ∂L/∂x_L × ∏_{l=0}^{L-1} (I + ∂F_l/∂x_l)
```

**Key Property:** Always has identity path!

```
∂x_L/∂x_0 = I + (gradient through transformations)
```

Even if ∂F_l/∂x_l → 0, gradient still flows through I.

**Empirical Verification:**

```python
# Measure gradient magnitudes
grads_by_layer = []

for layer_idx in range(12):
    grad = model.layers[layer_idx].input.grad.norm()
    grads_by_layer.append(grad)

# Without residuals: Exponential decay
# grads = [1.0, 0.5, 0.25, 0.125, ...] → Vanishing!

# With residuals: Stable
# grads = [1.0, 0.95, 0.92, 0.90, ...] → Stable!
```

### Layer Normalization: Why It Stabilizes Training

**Internal Covariate Shift:**

Each layer's input distribution changes as previous layers update:

```python
# Iteration 1
layer_2_input ~ N(0, 1)

# Iteration 100
layer_2_input ~ N(5, 10)  # Mean and variance shifted!

# Layer 2 must constantly adapt to new distribution
```

**Layer Norm Solution:**

```
Normalized = (x - μ) / σ

Always: μ = 0, σ = 1 (per sample)
```

Each layer sees consistent input distribution!

**Learnable Affine Transform:**

```
Output = γ × Normalized + β

γ, β learned per feature
```

Allows network to undo normalization if needed:
- If γ = σ, β = μ: Recovers original distribution
- If γ = 1, β = 0: Pure normalization

**Why Not Batch Norm for Transformers?**

Batch Norm statistics depend on batch:
```python
# Batch of sequences with different lengths
batch = [
    "The cat",           # length 2
    "A long sentence",   # length 3  
]

# Padding required
batch_padded = [
    "The cat <PAD>",     # length 3
    "A long sentence",   # length 3
]

# Batch Norm includes <PAD> in statistics → Unstable!
```

Layer Norm: Independent of batch → Stable for sequences.

### Feed-Forward Network: Why 4× Expansion?

**Standard Configuration:**

```
d_model = 512
d_ff = 2048  # 4× expansion

FFN(x) = W_2 ReLU(W_1 x + b_1) + b_2
```

**Reasoning:**

**1. Representational Capacity:**

Linear layers without expansion:
```
y = W x  # (512, 512) @ (512,) = (512,)
```

Limited: Same dimensionality → Simple transformations only

With expansion:
```
h = ReLU(W_1 x)  # (2048, 512) @ (512,) = (2048,)
y = W_2 h        # (512, 2048) @ (2048,) = (512,)
```

Intermediate 2048-dim space allows complex transformations!

**2. Parameter Count:**

```
No expansion: 512² = 262K params
4× expansion: 512×2048 + 2048×512 = 2.1M params

8× more parameters → More capacity
```

**3. Empirical Optimal:**

Ablation studies show:
- 1×: Underfitting
- 2×: Better but not optimal
- 4×: Sweet spot (performance vs cost)
- 8×: Diminishing returns

**4. Comparison to Attention:**

```
Attention params: 4 × d_model² = 1M (Q,K,V,O projections)
FFN params: 2 × d_model × d_ff = 2.1M

FFN has more parameters!
Most model capacity is in FFN, not attention
```

### Pre-Norm vs Post-Norm: Training Dynamics

**Post-Norm (Original Transformer):**

```python
def layer_post_norm(x):
    x = x + attention(x)
    x = layer_norm(x)
    x = x + ffn(x)
    x = layer_norm(x)
    return x
```

**Gradient Flow:**

```
∂L/∂x_input = ∂L/∂x_output × ∂layer_norm/∂(x + attention) × ...
```

Gradient must pass through layer_norm → Can cause instability

**Requires:** Careful learning rate warmup

**Pre-Norm (Modern):**

```python
def layer_pre_norm(x):
    x = x + attention(layer_norm(x))
    x = x + ffn(layer_norm(x))
    return x
```

**Gradient Flow:**

```
∂L/∂x_input = ∂L/∂x_output × (I + ∂attention/∂(layer_norm(x)) × ...)
```

Direct residual path BEFORE normalization → More stable!

**Empirical Comparison:**

```
Training BERT-base (12 layers):

Post-Norm:
- Requires 10K step warmup
- Diverges with lr=1e-3 without warmup
- Final loss: 2.1

Pre-Norm:
- Works with 1K step warmup
- Stable with lr=1e-3
- Final loss: 2.15 (slightly higher but trains faster)
```

Trade-off: Pre-norm more stable, post-norm slightly better final performance.

**Modern Practice:** Pre-norm for training stability (GPT-2, GPT-3, most new models)

### Encoder Self-Attention: Bidirectional Information Flow

**Key Difference from Decoder:**

Encoder: Can see all tokens (past and future)
Decoder: Can only see past tokens (causal masking)

**Information Aggregation:**

```python
# For token at position i:
# Encoder: Aggregates from all positions
output[i] = Σ_j attention_weights[i,j] × value[j]
            for j in [0, 1, 2, ..., n-1]  # All positions!

# Decoder: Aggregates only from past
output[i] = Σ_j attention_weights[i,j] × value[j]
            for j in [0, 1, ..., i]  # Only up to i
```

**Consequence:**

Encoder builds "context-aware" representations:
- Each token's embedding incorporates information from entire sequence
- Suitable for understanding tasks (classification, NER, QA)

Decoder builds "causal" representations:
- Each token only knows past
- Suitable for generation (language modeling, translation)

### BERT-Specific Architectural Details

**Three Types of Embeddings:**

```python
# 1. Token embeddings
token_emb = token_embedding(input_ids)  # (batch, seq, 768)

# 2. Position embeddings (learned, not sinusoidal)
pos_emb = position_embedding(positions)  # (batch, seq, 768)

# 3. Segment embeddings (for sentence pairs)
seg_emb = segment_embedding(segments)  # (batch, seq, 768)

# Combine
embeddings = token_emb + pos_emb + seg_emb
```

**Why Segment Embeddings?**

BERT processes sentence pairs:
```
Input: [CLS] Sentence A [SEP] Sentence B [SEP]
Segments: [0,    0,        0,    1,         1]

Segment embedding helps model distinguish:
- Tokens from sentence A (seg=0)
- Tokens from sentence B (seg=1)
```

**[CLS] Token:**

```python
# Special token at position 0
# Aggregates sequence-level information

# During pre-training:
# - Used for next sentence prediction
# - Attends to entire sequence
# - Becomes "sequence representation"

# During fine-tuning:
# - Classification: Pass [CLS] to classifier
# - QA: Also use token-level representations
```

### Computational Bottlenecks

**Profile of BERT-base Forward Pass:**

```
Total time: 100ms

Token embedding: 2ms (2%)
Position embedding: 1ms (1%)

Layer 1-12 (each ~8ms):
  - Attention: 5ms (50% of layer)
    - QKV projection: 1ms
    - Attention computation: 3ms (n² dominant for long sequences)
    - Output projection: 1ms
  - FFN: 3ms (38% of layer)
    - First linear: 1.5ms
    - Second linear: 1.5ms

Total: 2 + 1 + 12×8 = 99ms

Bottleneck for short sequences (n<512): FFN
Bottleneck for long sequences (n>1024): Attention
```

### Summary: Architectural Insights

1. **Residuals are Critical:** Enable training deep networks (12+ layers)
2. **Layer Norm > Batch Norm:** Stable for variable-length sequences
3. **FFN Has Most Parameters:** 4× expansion gives capacity
4. **Pre-Norm Preferred:** Training stability without sacrificing much performance
5. **Bidirectional Attention:** Encoder's key feature (vs decoder's causal attention)

Encoder architecture is the foundation for BERT, RoBERTa, ELECTRA, and all encoder-only models!
