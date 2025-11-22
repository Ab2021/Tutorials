# Day 9: Multi-Head Attention
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why use multiple attention heads instead of a single large attention head?

**Answer:**

**Single Head (d_k = 512):**
- Can only learn one attention pattern
- All 512 dimensions used for single relationship type
- Limited expressiveness

**Multiple Heads (8 × d_k = 64):**
- Can learn 8 different attention patterns simultaneously
- Different heads specialize in different relationships:
  - Head 1: Local syntax
  - Head 2: Long-range dependencies
  - Head 3: Positional patterns
  - Head 4: Semantic relationships

**Same Total Parameters:**
```
Single head: 3 × d_model² (Q, K, V projections)
8 heads: 3 × d_model² + d_model² (Q, K, V + final projection)
Only 33% overhead for W_O!
```

**Empirical Benefits:**
- Better representation learning
- Ensemble effect (multiple views)
- Robustness (redundancy across heads)
- Interpretability (can analyze each head)

---

#### Q2: You notice that removing 40% of attention heads doesn't hurt performance. Explain why and what you should do.

**Answer:**

**Why This Happens:**

1. **Redundancy:** Many heads learn similar patterns
   ```
   Empirical observation:
   - ~30% of heads are "critical"
   - ~50% are "semi-redundant"
   - ~20% can be removed with < 1% impact
   ```

2. **Gradient Flow:** During training, some heads become less important

**What To Do:**

**1. Head Pruning (Production):**
```python
# Measure importance
importances = []
for layer, head in all_heads:
    loss_without = evaluate_without_head(layer, head)
    importance = baseline_loss - loss_without
    importances.append((layer, head, importance))

# Sort and prune least important
sorted_heads = sorted(importances, key=lambda x: x[2])
prune_bottom_40_percent(sorted_heads)

# Benefits:
# - 40% less compute
# - 40% less memory
# - < 1% accuracy loss
```

**2. Distillation (Better approach):**
```python
# Train smaller model to match large model
teacher_model = large_model_with_12_heads
student_model = smaller_model_with_6_heads

# Distillation loss
loss = KL_divergence(student_output, teacher_output.detach())

# Result: 6-head model performs as well as 12-head!
```

**Interview Follow-up:**
*Q: Why not train with fewer heads from the start?*

**A:**
- More heads help during training (exploration)
- Can prune after training (exploitation)
- Redundancy aids gradient flow and robustness
- Like neural network pruning: train big, deploy small

---

#### Q3: Explain the difference between multi-head attention and multi-layer attention.

**Answer:**

**Multi-Head Attention (Parallel):**
```python
# Same layer, different heads running simultaneously
heads = [head1(x), head2(x), ..., head8(x)]  # Parallel
output = concat(heads) @ W_O

# Learns: Different patterns at same depth
# Example: Syntax + semantics + position at once
```

**Multi-Layer Attention (Sequential):**
```python
# Different layers, one after another
x1 = layer1(x0)
x2 = layer2(x1)
...
x12 = layer12(x11)

# Learns: Hierarchical abstractions
# Example: Tokens → words → phrases → sentences
```

**Key Differences:**

| Aspect | Multi-Head | Multi-Layer |
|--------|------------|-------------|
| Execution | Parallel | Sequential |
| Learns | Diverse patterns | Hierarchical features |
| Depth | Same | Increasing |
| Example | 8 heads at layer 5 | Layers 1-12 |

**Modern Transformers Use BOTH:**
```
BERT: 12 layers × 12 heads = 144 attention mechanisms
- Within layer: 12 heads learn diverse patterns (parallel)
- Across layers: 12 layers learn hierarchy (sequential)
```

---

#### Q4: Your multi-head attention is slow. How do you optimize it?

**Answer:**

**Bottlenecks:**

1. **Memory Transfers:** Moving data between GPU memory layers
2. **Multiple Projections:** Q, K, V for each head
3. **Attention Computation:** n² for each head

**Optimizations:**

**1. Fused Kernels (Flash Attention):**
```python
# Standard: Multiple GPU kernel launches
scores = Q @ K.T  # Kernel 1
attn = softmax(scores)  # Kernel 2
output = attn @ V  # Kernel 3

# Flash Attention: Fused into single kernel
output = flash_attention(Q, K, V)  # Single kernel
# 2-4× faster!
```

**2. Grouped-Query Attention:**
```python
# Standard: 32 Q, K, V heads
num_heads = 32
# Memory: 32 × n² attention matrices

# GQA: 32 Q heads, 8 K/V heads
num_q_heads = 32
num_kv_heads = 8  # 4× fewer!

# Each KV head shared by 4 Q heads
# Memory: 8 × n² (4× reduction)
# Used in LLaMA-2, Mistral
```

**3. Efficient Projections:**
```python
# Slow: Separate projections
for i in range(num_heads):
    Q_i = W_Q_i @ X

# Fast: Single large projection + reshape
Q_all = W_Q_all @ X  # One matmul
Q = Q_all.view(batch, seq, num_heads, d_k)  # Reshape
```

**4. Attention Dropout Placement:**
```python
# Slow: Dropout on large tensors
attn = softmax(scores)  # (batch, heads, n, n)
attn = dropout(attn)

# Fast: Only dropout non-zero entries
# Or: Higher dropout rate, fewer operations
```

**Performance Gains:**
```
Baseline: 100ms per forward pass

+ Flash Attention: 40ms (2.5× faster)
+ GQA (if applicable): 30ms (3.3× faster)
+ Efficient projections: 25ms (4× faster)
```

---

#### Q5: How do you visualize and interpret what different attention heads learn?

**Answer:**

**Extraction:**
```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
outputs = model(**inputs)

# attentions: tuple of (num_layers) × (batch, num_heads, seq, seq)
layer_5_head_3 = outputs.attentions[5][0, 3]  # Layer 5, Head 3
```

**Visualization Types:**

**1. Attention Heatmaps:**
```python
import seaborn as sns

sns.heatmap(layer_5_head_3.detach().numpy(),
            xticklabels=tokens, yticklabels=tokens)
plt.title('Layer 5, Head 3 Attention')
```

**2. Head Clustering:**
```python
# Compute similarity between heads
head_similarities = []
for i in range(num_heads):
    for j in range(i+1, num_heads):
        sim = cosine_similarity(head_i_patterns, head_j_patterns)
        head_similarities.append((i, j, sim))

# Cluster similar heads
# Find: Some heads learn similar patterns (redundant)
# Others are unique (specialized)
```

**3. Probing Tasks:**
```python
# Test what head learns
# E.g., Does head attend to subject of verb?

for sentence in test_set:
    subject_idx = find_subject(sentence)
    verb_idx = find_verb(sentence)
    
    # Check if verb attends to subject
    attn_weight = attention[verb_idx, subject_idx]
    
    if attn_weight > threshold:
        syntax_count += 1

# High count → Head learns syntax!
```

**Common Patterns Found:**

- **Syntactic heads**: Attend to grammatical dependencies
- **Positional heads**: Attend to first/last tokens
- **Broadcast heads**: Uniform attention (averaging)
- **Previous token heads**: Attend to token before
- **Semantic heads**: Long-range coreference

**Production Tool:**
```python
def analyze_all_heads(model, data):
    analysis = {}
    
    for layer in range(num_layers):
        for head in range(num_heads):
            # Extract patterns
            patterns = extract_attention_patterns(layer, head, data)
            
            # Classify head type
            head_type = classify_head(patterns)
            # Returns: 'syntactic', 'positional', 'semantic', etc.
            
            analysis[f'L{layer}H{head}'] = head_type
    
    return analysis

# Use for model understanding and debugging
```

---

### Production Challenges

**Challenge: Inconsistent Head Importance Across Tasks**

**Scenario:**
- Head pruning works great on Task A (sentiment)
- Same pruning fails on Task B (QA)

**Root Cause:**
Different tasks need different attention patterns.

**Solution:**
- Task-specific pruning
- Or: Keep all heads, use task-specific W_O
- Or: Train separate models (best performance)

---

### Key Takeaways

1. **Multiple heads enable specialization**: Different patterns learned simultaneously
2. **Redundancy is intentional**: Robustness, gradient flow, ensemble
3. **Pruning possible**: Can remove 30-40% heads with minimal impact
4. **Optimization critical**: Flash Attention, GQA for production speed
5. **Interpretability**: Can visualize and understand head specialization
