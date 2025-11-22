# Day 10: Transformer Encoder Architecture
## Core Concepts & Theory

### Complete Encoder Architecture

The Transformer encoder consists of stacked layers, each with:
1. Multi-head self-attention
2. Add & Normalize
3. Feed-forward network
4. Add & Normalize

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        seq_len = x.shape[1]
        
        # Token embeddings
        token_emb = self.token_embedding(x)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(positions)
        
        # Combine and dropout
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
```

### Residual Connections

**Why They're Critical:**

```python
# Without residuals (deep network problem)
x1 = layer1(x0)
x2 = layer2(x1)
...
x12 = layer12(x11)

# Gradient: ∂L/∂x0 = ∂L/∂x12 × ∂x12/∂x11 × ... × ∂x1/∂x0
# Product of 12 terms → Vanishing gradients!

# With residuals
x1 = x0 + layer1(x0)
x2 = x1 + layer2(x1)

# Gradient has direct path: ∂x2/∂x0 = 1 + ∂layer2/∂x0
# Always has identity path → No vanishing!
```

**Implementation:**

```python
# Original output
attn_output = self.self_attn(x)

# Add residual
x = x + attn_output  # Residual connection

# Normalize
x = self.layer_norm(x)
```

### Layer Normalization

**Formula:**

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β

where:
- μ = mean across features (per sample, per position)
- σ² = variance across features
- γ, β = learned parameters
- ε = numerical stability (1e-5)
```

**Implementation:**

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta
```

**Why Layer Norm (not Batch Norm)?**

```
Batch Norm: Normalize across batch
- Problem: Batch statistics unstable for sequences
- Different sequence lengths → different statistics

Layer Norm: Normalize across features (per sample)
- Stable regardless of batch size or sequence length
- Standard for Transformers
```

### Feed-Forward Network

**Architecture:**

```python
FFN(x) = W2 × ReLU(W1 × x + b1) + b2

where:
- W1: (d_model, d_ff) typically d_ff = 4 × d_model
- W2: (d_ff, d_model)
```

**Why Large Hidden Dimension?**

```
d_model = 512, d_ff = 2048 (4×)

Intuition: FFN acts as position-wise MLP
- Projects to higher dimension (2048)
- Learns non-linear transformations
- Projects back to original dimension (512)

Similar to bottleneck in ResNet but inverted:
ResNet: Wide → Narrow → Wide
Transformer FFN: Narrow → Wide → Narrow
```

**Position-wise:**

```python
# FFN applied independently to each position
for i in range(seq_len):
    output[i] = FFN(x[i])

# Equivalent to:
output = FFN(x)  # Broadcasts across sequence

# Each position transformed independently
# No interaction between positions (that's what attention does!)
```

### Pre-Norm vs Post-Norm

**Post-Norm (Original Transformer):**

```python
x = x + attention(x)
x = layer_norm(x)
x = x + ffn(x)
x = layer_norm(x)
```

**Pre-Norm (Modern, e.g., GPT-2):**

```python
x = x + attention(layer_norm(x))
x = x + ffn(layer_norm(x))
```

**Differences:**

| Aspect | Post-Norm | Pre-Norm |
|--------|-----------|----------|
| Training Stability | Less stable (needs warmup) | More stable |
| Learning Rate | Requires careful tuning | More robust |
| Performance | Slightly better (converged) | Slightly worse |
| Training Speed | Slower (needs warmup) | Faster |

**Modern Practice:**
- GPT-2, GPT-3: Pre-norm
- BERT: Post-norm
- T5: Pre-norm

Pre-norm is now standard for training very deep models.

### Encoder Self-Attention Patterns

Different from decoder (Day 11):
- **Bidirectional**: Can attend to all positions
- **No masking**: All tokens visible
- **Purpose**: Build contextualized representations

```python
# All positions attend to all positions
# "The cat sat on the mat"

Attention matrix:
     The  cat  sat  on  the  mat
The [0.3, 0.2, 0.1, 0.1, 0.2, 0.1]  # Can see all
cat [0.2, 0.4, 0.2, 0.1, 0.0, 0.1]  
sat [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]
...

# "cat" can attend to "sat" (after) and "The" (before)
# Bidirectional context!
```

### BERT Architecture (Encoder-Only)

```python
class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, 768)
        self.position_emb = nn.Embedding(512, 768)
        self.token_type_emb = nn.Embedding(2, 768)  # For sentence pairs
        
        # 12 encoder layers
        self.encoder = TransformerEncoder(
            num_layers=12,
            d_model=768,
            num_heads=12,
            d_ff=3072
        )
        
        # Task-specific heads
        self.mlm_head = nn.Linear(768, vocab_size)  # Masked LM
        self.nsp_head = nn.Linear(768, 2)  # Next sentence prediction
    
    def forward(self, input_ids, token_type_ids=None):
        # Embeddings
        token_emb = self.token_emb(input_ids)
        pos_emb = self.position_emb(torch.arange(len(input_ids)))
        
        if token_type_ids is not None:
            type_emb = self.token_type_emb(token_type_ids)
            embeddings = token_emb + pos_emb + type_emb
        else:
            embeddings = token_emb + pos_emb
        
        # Encode
        encoder_output = self.encoder(embeddings)
        
        return encoder_output
```

### Computational Complexity

**Per Layer:**

```
Multi-head attention: O(n² × d)
FFN: O(n × d²)

For large d (768, 1024):
- Short sequences (n < 512): FFN dominates
- Long sequences (n > 1024): Attention dominates
```

**Total (N layers):**

```
Total: N × (O(n² × d) + O(n × d²))

BERT-base: 12 × (n² × 768 + n × 768²)
```

### Summary

Encoder architecture key components:
- **Multi-head attention**: Capture relationships
- **Residual connections**: Enable deep networks
- **Layer normalization**: Stabilize training
- **Feed-forward**: Position-wise transformations
- **Pre-norm**: Modern standard for stability

Used in: BERT, RoBERTa, ELECTRA, DeBERTa (encoder-only models)

Next (Day 11): Decoder architecture with masked attention
