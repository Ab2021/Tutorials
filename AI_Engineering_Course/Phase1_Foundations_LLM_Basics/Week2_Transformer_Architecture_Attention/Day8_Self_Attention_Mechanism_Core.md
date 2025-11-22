# Day 8: Self-Attention Mechanism
## Core Concepts & Theory

### The Attention Revolution

Before Transformers (2017), sequence models relied on recurrence (RNNs, LSTMs) or convolution (CNNs). The key limitation: 
- RNNs: Sequential processing (can't parallelize)
- CNNs: Fixed receptive field (limited long-range dependencies)

**Transformers solution**: Attention mechanism - direct connections between all positions.

### Scaled Dot-Product Attention

**Input:**
- Query Q: (seq_len, d_k)
- Key K: (seq_len, d_k)  
- Value V: (seq_len, d_v)

**Output:**
- Attended values: (seq_len, d_v)

**Algorithm:**

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    Returns: (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, seq_len, seq_len)
    
    # Step 2: Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 3: Softmax to get attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    # attn_weights: (batch, seq_len, seq_len)
    
    # Step 4: Weighted sum of values
    output = torch.matmul(attn_weights, V)
    # output: (batch, seq_len, d_v)
    
    return output, attn_weights
```

**Step-by-Step Example:**

```python
# Input sequence: "The cat sat"
# Embeddings: (3, 512) - 3 tokens, 512-dim each

Q = K = V = embeddings  # Self-attention (all same)

# Compute similarity between all pairs
scores = Q @ K.T / sqrt(512)

# scores:
#        The   cat   sat
# The  [[0.9,  0.3,  0.1],
# cat   [0.4,  0.8,  0.2],
# sat   [0.2,  0.5,  0.7]]

# Softmax (each row sums to 1)
attn_weights = softmax(scores)

# attn_weights:
#        The   cat   sat
# The  [[0.5,  0.3,  0.2],
# cat   [0.2,  0.6,  0.2],
# sat   [0.1,  0.4,  0.5]]

# Interpretation:
# - Token "The" attends mostly to itself (0.5) and "cat" (0.3)
# - Token "cat" attends mostly to itself (0.6)
# - Token "sat" attends to "cat" (0.4) and itself (0.5)

# Final output: weighted sum
output = attn_weights @ V
# output[0] = 0.5*V[0] + 0.3*V[1] + 0.2*V[2]  (representation of "The")
```

### Why Scaling by √d_k?

**Problem without scaling:**

```python
# For large d_k, dot products grow large
Q = torch.randn(100, 512)  # d_k = 512
K = torch.randn(100, 512)

scores = Q @ K.T  # Range: ~(-50, 50) due to √512 ≈ 22.6

softmax_scores = torch.softmax(scores, dim=-1)
# Most values ≈ 0, one value ≈ 1 (saturated!)
```

**With scaling:**

```python
scores_scaled = (Q @ K.T) / math.sqrt(512)  # Range: ~(-2.2, 2.2)

softmax_scaled = torch.softmax(scores_scaled, dim=-1)
# More uniform distribution, gradients flow better
```

**Mathematical reasoning:**

If Q, K ~ N(0, 1):
- Dot product Q·K has variance d_k
- Scaling by √d_k normalizes variance to 1
- Keeps softmax in reasonable range

### Query, Key, Value Intuition

**Analogy: Database Search**

```python
# You have a database of key-value pairs
database = {
    "cat": "a small feline animal",
    "dog": "a canine pet",
    "bird": "a flying creature"
}

# Query: "What is a cat?"
query = "cat"

# Step 1: Compute similarity between query and all keys
similarities = {
    "cat": similarity(query, "cat"),   # High
    "dog": similarity(query, "dog"),   # Medium
    "bird": similarity(query, "bird")  # Low
}

# Step 2: Softmax (normalize to probabilities)
weights = softmax(similarities)  # cat: 0.7, dog: 0.2, bird: 0.1

# Step 3: Weighted sum of values
result = (
    0.7 * "a small feline animal" +
    0.2 * "a canine pet" +
    0.1 * "a flying creature"
)
```

**In attention:**
- **Query**: What information am I looking for?
- **Key**: What information do I have?
- **Value**: The actual information content

**Self-Attention:**
- Q = K = V (all from same sequence)
- Each token queries all other tokens

**Cross-Attention:**
- Q from decoder, K and V from encoder
- Decoder tokens query encoder information

### Attention as Soft Dictionary Lookup

```python
# Hard lookup (nearest neighbor)
query = "cat"
key_match = argmax(similarity(query, all_keys))
return values[key_match]

# Soft lookup (attention)
query = "cat"
weights = softmax(similarity(query, all_keys))
return weighted_average(values, weights)
```

Soft lookup is differentiable → Can train end-to-end!

### Computational Complexity

**Time Complexity:**

```
1. Q @ K.T: O(n² × d_k)
2. Softmax: O(n²)
3. Attn @ V: O(n² × d_v)

Total: O(n² × d) where d = d_k ≈ d_v
```

**Memory Complexity:**

```
Attention matrix: O(n²)
```

**Quadratic in sequence length!**

For n = 512: 512² = 262K attention scores
For n = 2048: 2048² = 4.2M attention scores (16× more!)

This is the key bottleneck for long sequences.

### Masking in Attention

**Types of Masks:**

**1. Padding Mask:**

```python
# Sequence: "The cat <PAD> <PAD>"
# Mask out PAD tokens

sequence = torch.tensor([1, 2, 0, 0])  # 0 = PAD
pad_mask = (sequence != 0).unsqueeze(0)
# pad_mask: [[True, True, False, False]]

# In attention:
scores = scores.masked_fill(~pad_mask.unsqueeze(1), -1e9)
# Don't attend to PAD tokens
```

**2. Causal Mask (for autoregressive models):**

```python
# Prevent attending to future tokens
seq_len = 4
causal_mask = torch.tril(torch.ones(seq_len, seq_len))

# causal_mask:
# [[1, 0, 0, 0],    Token 0 can only see token 0
#  [1, 1, 0, 0],    Token 1 can see tokens 0-1
#  [1, 1, 1, 0],    Token 2 can see tokens 0-2
#  [1, 1, 1, 1]]    Token 3 can see all

scores = scores.masked_fill(causal_mask == 0, -1e9)
```

**Why -1e9?**

```python
softmax([-1e9, 3, 2]) = [0.0, 0.73, 0.27]
# -1e9 → 0 after softmax (effectively masked out)
```

### Complete Implementation

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None):
        super().__init__()
        if d_k is None:
            d_k = d_model
        
        self.d_k = d_k
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_k, d_model)  # Output back to d_model
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        
        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
```

### Attention Patterns

Different layers learn different attention patterns:

**Layer 1 (early):**
- Local attention (adjacent words)
- Syntactic patterns ("the" → "cat")

**Layer 6 (middle):**
- Phrasal attention ("New" → "York")
- Dependency relations (subject-verb)

**Layer 12 (late):**
- Long-range semantic attention
- Coreference ("he" → "John" from 50 tokens away)

### Summary

Key concepts:
- **Attention mechanism**: Direct connections between all positions
- **Q, K, V**: Query what you're looking for, Keys what's available, Values the content
- **Scaling**: Divide by √d_k for stable gradients
- **Masking**: Padding and causal masks  
- **Complexity**: O(n²) - quadratic in sequence length

Self-attention is the core building block of Transformers - enabling:
- Parallelization (no sequential dependency)
- Long-range dependencies (direct connections)
- Interpretability (can visualize attention weights)

Next (Day 9): Multi-head attention - running multiple attention layers in parallel.
