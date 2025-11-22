# Day 9: Multi-Head Attention
## Core Concepts & Theory

### From Single to Multiple Attention Heads

**Problem with Single Attention:**

A single attention mechanism can only learn one type of relationship:

```python
# Single attention learns one pattern
attn = softmax(Q @ K.T / sqrt(d_k))  # One attention matrix

# Example: Might learn syntactic dependencies
# But misses semantic, positional, or other patterns
```

**Solution: Multi-Head Attention**

Run multiple attention mechanisms in parallel, each learning different patterns:

```python
# 8 parallel attention heads
head1 = attention(Q1, K1, V1)  # Learns local syntax
head2 = attention(Q2, K2, V2)  # Learns long-range dependencies
head3 = attention(Q3, K3, V3)  # Learns positional patterns
...
head8 = attention(Q8, K8, V8)

# Concatenate and project
output = concat(head1, ..., head8) @ W_O
```

### Multi-Head Attention Architecture

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 512 / 8 = 64 per head
        
        # Projection matrices
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Linear projections
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_O(output)
        
        return output
```

### Head Dimension Trade-off

**Total parameters: d_model × d_model (same regardless of num_heads)**

```
Single head (d_k = 512):
- W_Q, W_K, W_V: 512 × 512 each
- Total: 3 × 512² = 786K params

8 heads (d_k = 64 each):
- W_Q, W_K, W_V: 512 × 512 each (split into 8 heads)
- Total: 3 × 512² = 786K params (same!)

Key difference: How those parameters are used
```

**Why Multiple Heads Help:**

```
1 head of 512-dim: Limited to one attention pattern
8 heads of 64-dim each: Can learn 8 different patterns simultaneously

Trade-off:
+ More diverse patterns
+ Better representation learning
- Each head has less capacity (64-dim vs 512-dim)
```

**Typical Configurations:**

```
BERT-base: d_model=768, num_heads=12, d_k=64
BERT-large: d_model=1024, num_heads=16, d_k=64
GPT-2-small: d_model=768, num_heads=12, d_k=64
GPT-3: d_model=12288, num_heads=96, d_k=128
```

Pattern: d_k usually 64 or 128 (empirically optimal)

### What Different Heads Learn

**Empirical Analysis (from BERT/GPT papers):**

**Syntactic Heads:**
```
Head 1: Attends to adjacent words (local syntax)
"The cat sat" → "cat" attends to "The" and "sat"
```

**Positional Heads:**
```
Head 2: Attends to specific positions
All tokens attend to first token ([CLS])
```

**Dependency Heads:**
```
Head 3: Learns grammatical dependencies
Subject-verb agreement: "cats" → "are" (not "is")
```

**Semantic Heads:**
```
Head 4: Long-range semantic relationships
"John ... he" → "he" attends to "John" (coreference)
```

**Broadcast Heads:**
```
Head 5: Uniform attention (averaging)
All positions attend equally → Global context
```

**Redundant Heads:**
```
Some heads learn similar patterns (redundancy)
Can prune without much performance loss
```

### Computational Complexity

**Single-Head Attention:**
```
Time:  O(n² × d_model)
Memory: O(n²)
```

**Multi-Head Attention (h heads):**
```
Time: O(n² × d_model) + O(d_model²)
       └─────────────┘   └─────────┘
       h × attention     final projection
       (parallelized)

# Same as single-head! (parallel computation)

Memory: O(h × n²)  # Store attention for each head
```

**Why Same Time Complexity:**

```python
# Single head
Q_single @ K_single.T  # (n, 512) @ (512, n) = O(n² × 512)

# 8 heads
for i in range(8):
    Q_i @ K_i.T  # (n, 64) @ (64, n) = O(n² × 64)
# Total: 8 × O(n² × 64) = O(n² × 512)  # Same!

# But heads run in parallel → Same wall-clock time
```

### Concatenation and Final Projection

**After attention:**

```python
# Each head produces: (batch, seq_len, d_k)
head1_out = attention1(Q1, K1, V1)  # (batch, seq_len, 64)
head2_out = attention2(Q2, K2, V2)  # (batch, seq_len, 64)
...
head8_out = attention8(Q8, K8, V8)  # (batch, seq_len, 64)

# Concatenate along feature dimension
concat_out = torch.cat([head1_out, ..., head8_out], dim=-1)
# concat_out: (batch, seq_len, 512)  # 8 × 64 = 512

# Final linear projection
output = W_O @ concat_out
# output: (batch, seq_len, d_model)
```

**Why Final Projection?**

```
1. Mix information from all heads
2. Allow heads to interact and combine patterns
3. Project back to d_model (required for residual connection)
```

### Multi-Head vs Multi-Layer

**Multi-Head (parallel):**
```
Input → [Head1, Head2, ..., Head8] → Concat → Output
         (all simultaneously)

Learns: Different patterns at same depth
Example: Syntax + semantics + position simultaneously
```

**Multi-Layer (sequential):**
```
Input → Layer1 → Layer2 → ... → Layer12 → Output
        (one after another)

Learns: Hierarchical abstractions
Example: Layer1 (syntax) → Layer6 (phrases) → Layer12 (semantics)
```

**Modern Transformers use BOTH:**
```
12 layers × 8 heads = 96 total attention mechanisms
- 8 heads per layer (parallel, diverse patterns)
- 12 layers (sequential, hierarchical learning)
```

### Visualization Example

```python
# Attention pattern for "The cat sat on the mat"

# Head 1 (Local syntax):
     The  cat  sat  on   the  mat
The [0.7, 0.2, 0.1, 0.0, 0.0, 0.0]  # Attends locally
cat [0.3, 0.5, 0.2, 0.0, 0.0, 0.0]

# Head 2 (Long-range):
     The  cat  sat  on   the  mat
The [0.2, 0.1, 0.1, 0.1, 0.2, 0.3]  # Uniform/long-range
mat [0.2, 0.3, 0.1, 0.1, 0.2, 0.1]  # Attends to "cat"

# Head 3 (Positional):
     The  cat  sat  on   the  mat
cat [0.8, 0.1, 0.0, 0.0, 0.1, 0.0]  # Strong self-attention
mat [0.1, 0.0, 0.0, 0.0, 0.8, 0.1]

# Different heads capture different relationships!
```

### Implementation Details

**Efficient Reshaping:**

```python
# Original implementation (readable):
Q = self.W_Q(x)  # (batch, seq_len, d_model)
Q = Q.view(batch, seq_len, num_heads, d_k)
Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

# Efficient implementation (fused):
Q = self.W_Q(x).view(batch, seq_len, num_heads, d_k).transpose(1, 2)
```

**Memory-Efficient Attention:**

```python
# Standard: Materialize full attention matrix per head
attn = softmax(Q @ K.T / sqrt(d_k))  # (batch, h, n, n) in memory

# Flash Attention: Never materialize full matrix
# Compute attention on-the-fly in fused kernel
# Same result, 2-4× faster, less memory
```

### Summary

Multi-head attention enables:
- **Parallel pattern learning**: Different heads learn different relationships
- **No parameter overhead**: Same total parameters as single head
- **Computational efficiency**: Heads process in parallel
- **Interpretability**: Can analyze what each head learns
- **Robustness**: Redundancy across heads prevents overfitting to single pattern

Key hyperparameter: num_heads (typically 8-16)
Key dimension: d_k = d_model / num_heads (typically 64-128)

Next (Day 10): Complete Transformer encoder architecture
