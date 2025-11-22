# Day 9: Multi-Head Attention
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Parameter Allocation: Single vs Multi-Head

**Single-Head Attention:**

```
W_Q: (d_model, d_model) = 512 × 512 = 262K params
W_K: (d_model, d_model) = 512 × 512 = 262K params
W_V: (d_model, d_model) = 512 × 512 = 262K params
Total: 786K parameters
```

**Multi-Head Attention (8 heads):**

```
W_Q: (d_model, d_model) = 512 × 512 = 262K params
W_K: (d_model, d_model) = 512 × 512 = 262K params
W_V: (d_model, d_model) = 512 × 512 = 262K params
W_O: (d_model, d_model) = 512 × 512 = 262K params
Total: 1.05M parameters (33% overhead for W_O)
```

**Key Insight:** Most parameters are the same! W_O is the only addition.

**How Parameters Are Used Differently:**

```python
# Single head: Full 512×512 matrix
Q_single = X @ W_Q  # Uses entire 512×512

# Multi-head: Same 512×512 matrix, split into 8 heads
Q_multi = X @ W_Q  # Same operation!
Q_multi = Q_multi.view(batch, seq, 8, 64)  # Reshape into 8 heads

# Equivalent to having 8 separate 512×64 matrices:
# W_Q = [W_Q1 | W_Q2 | ... | W_Q8]
```

So W_Q is implicitly 8 smaller matrices stacked!

### Head Dimension: Why d_k = 64 or 128?

**Empirical Observation:**

```
BERT-base: d_model=768, heads=12, d_k=64
GPT-2: d_model=768, heads=12, d_k=64
GPT-3: d_model=12288, heads=96, d_k=128
T5: d_model=512, heads=8, d_k=64

Pattern: d_k is almost always 64 or 128 regardless of d_model!
```

**Why Not Larger d_k?**

**1. Diminishing Returns:**

```
Express capacity with dimension d_k:
d_k=32: Can represent 32D space (limited)
d_k=64: Can represent 64D space (good for most patterns)
d_k=128: Can represent 128D space (marginal improvement)
d_k=256: Can represent 256D space (redundant for single pattern)
```

Beyond 64-128, single head doesn't learn much more.

**2. Softmax Saturation:**

Even with √d_k scaling, very large d_k can cause issues:

```python
# d_k = 512
scores = Q @ K.T / sqrt(512)  # Scale by ~22.6
# Still prone to some saturation

# d_k = 64
scores = Q @ K.T / sqrt(64)  # Scale by 8
# Better numerical stability
```

**3. Computational Efficiency:**

```
Attention FLOPs: O(n² × d_k)

d_k=64, 8 heads: 8 × (n² × 64) = n² × 512
d_k=512, 1 head: 1 × (n² × 512) = n² × 512

Same FLOPs, but 8 heads use GPU better (parallel)
```

**Optimal Strategy:**

```
Fixed d_k (64 or 128)
Increase d_model by adding more heads
```

Example scaling:
- Small: d_model=512,  heads=8,  d_k=64
- Base:  d_model=768,  heads=12, d_k=64
- Large: d_model=1024, heads=16, d_k=64
- XL:    d_model=2048, heads=32, d_k=64

### Head Specialization: Mathematical Analysis

**Why Do Heads Specialize?**

**Gradient Descent Perspective:**

Each head initialized randomly:

```python
# Head 1 randomly attends more to local patterns early in training
# Gradient: "Local attention helps prediction" → Reinforced
# Head 1 becomes local attention specialist

# Head 2 randomly attends to position 0 ([CLS])
# Gradient: "Aggregating to [CLS] helps" → Reinforced  
# Head 2 becomes positional specialist
```

Specialization emerges from random initialization + gradient descent!

**Redundancy vs Diversity:**

```
Empirical studies (BERT):
- ~30% of heads are "important" (removing them hurts performance)
- ~50% are "semi-redundant" (overlap with other heads)
- ~20% can be removed with minimal impact

Why keep redundant heads?
- Robustness: Backup if other heads fail
- Gradient flow: More paths for information
- Ensemble effect: Multiple views improve accuracy
```

### Attention Head Pruning

**Observation:** Not all heads are equally important.

**Importance Score:**

```python
def compute_head_importance(model, data):
    importances = []
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # Zero out this head's output
            with torch.no_grad():
                original_weight = model.layers[layer_idx].attn.heads[head_idx].W_O
                model.layers[layer_idx].attn.heads[head_idx].W_O = 0
            
            # Measure performance drop  
            loss_without_head = evaluate(model, data)
            
            # Restore
            model.layers[layer_idx].attn.heads[head_idx].W_O = original_weight
            
            importance = original_loss - loss_without_head
            importances.append(importance)
    
    return importances
```

**Pruning Strategy:**

```python
# Sort heads by importance
sorted_heads = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)

# Keep top 70% of heads
num_keep = int(0.7 * len(sorted_heads))
heads_to_prune = sorted_heads[num_keep:]

# Remove least important heads
for head_idx in heads_to_prune:
    prune_head(model, head_idx)

# Typical result: 30% speedup, < 1% accuracy loss!
```

### Concatenation: Why It Works

**After Attention:**

```python
# Each head outputs: (batch, seq_len, d_k)
head_outputs = [head1_out, head2_out, ..., head_h_out]

# Concatenate
concat = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, h × d_k)

# Project
output = W_O @ concat  # (batch, seq_len, d_model)
```

**Mathematical View:**

```
W_O allows cross-head interaction:

output_i = Σ_j W_O[i,j] × concat[j]
         = Σ_j W_O[i,j] × (head1[j%d_k] if j < d_k
                           else head2[j%d_k] if j < 2×d_k
                           ...)

W_O learns to mix information from all heads!
```

**Learned Mixing:**

```python
# Empirical observation of W_O patterns:
# Some rows of W_O weight head1 heavily (syntax-focused output)
# Other rows weight head4 heavily (semantic-focused output)

# W_O creates task-specific combinations of head outputs
```

### Multi-Head vs Grouped-Query Attention

**Standard Multi-Head:**

```
h heads, each with separate Q, K, V:
Memory: h × n² attention matrices
```

**Grouped-Query Attention (GQA):**

```python
# Used in LLaMA-2, Mistral
# Share K, V across groups of Q heads

num_q_heads = 32  # Query heads
num_kv_heads = 8  # Key/Value heads (4× fewer!)

# Each K,V head serves 4 Q heads
group_size = num_q_heads // num_kv_heads  # 4

for i in range(num_kv_heads):
    K_i = compute_key(x)
    V_i = compute_value(x)
    
    # Use same K,V for multiple Q heads
    for j in range(group_size):
        q_head_idx = i * group_size + j
        Q_j = compute_query(x, q_head_idx)
        output_j = attention(Q_j, K_i, V_i)
```

**Benefits:**

```
Memory: num_kv_heads × n² (vs num_q_heads × n²)
For 32 Q heads, 8 KV heads: 4× memory reduction!

Performance: Within 1-2% of full multi-head
Speed: Faster (less memory transfer)
```

Used in modern LLMs for inference efficiency!

### Attention Dropout: Where to Apply?

**Three Dropout Locations:**

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x):
        Q, K, V = self.project(x)
        
        # Option 1: Dropout on attention weights
        attn = softmax(Q @ K.T / sqrt(d_k))
        attn = self.dropout1(attn)  # ← Most common
        
        output = attn @ V
        
        # Option 2: Dropout on attention output
        output = self.dropout2(output)
        
        # Concatenate heads
        concat = concatenate(outputs)
        
        # Option 3: Dropout after final projection
        final = W_O @ concat
        final = self.dropout3(final)  # ← Also common
        
        return final
```

**Standard Practice (Transformers):**

- Dropout on attention weights: 0.1
- Dropout after final projection: 0.1  
- No dropout on attention output (redundant)

### Computational Optimization

**Fused Multi-Head Attention:**

```python
# Naive: Separate projections for each head
for i in range(num_heads):
    Q_i = W_Q_i @ X
    K_i = W_K_i @ X
    V_i = W_V_i @ X

# Optimized: Single large matrix multiply
Q_all = W_Q_all @ X  # Single matmul for all heads
K_all = W_K_all @ X
V_all = W_V_all @ X

# Then reshape
Q = Q_all.view(batch, seq, num_heads, d_k)
```

Much faster (better GPU utilization)!

This is how PyTorch/HuggingFace implement it.

### Summary: Deep Insights

1. **Same Parameters**: Multi-head uses same params as single-head (except W_O)
2. **Emergent Specialization**: Heads learn different patterns via gradient descent  
3. **Redundancy is Useful**: Robustness, ensemble effects
4. **d_k ≈ 64-128**: Empirically optimal head dimension
5. **W_O Mixes**: Final projection combines head outputs
6. **Grouped-Query**: Modern optimization (fewer KV heads)

Multi-head attention is elegant: simple modification with profound impact!
