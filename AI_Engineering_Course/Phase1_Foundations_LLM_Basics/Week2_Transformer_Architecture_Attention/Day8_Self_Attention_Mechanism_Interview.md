# Day 8: Self-Attention Mechanism
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain scaled dot-product attention. Why do we divide by √d_k?

**Answer:**

**Scaled Dot-Product Attention Formula:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Why Divide by √d_k:**

If Q, K have entries with variance 1, their dot product has variance d_k. For large d_k (e.g., 512), dot products can be very large, causing softmax saturation.

Without scaling:
```python
scores = Q @ K.T
# Variance ≈ d_k = 512
# Softmax becomes nearly one-hot → Vanishing gradients
```

With √d_k scaling:
```python
scores = (Q @ K.T) / math.sqrt(512)
# Variance ≈ 1
# Softmax well-distributed → Good gradients
```

---

#### Q2: What's the computational complexity of self-attention? How does it scale with sequence length?

**Answer:**

**Time Complexity: O(n² × d)**

```
1. Q @ K.T: O(n² × d)
2. Softmax: O(n²)
3. Attn @ V: O(n² × d)
```

**Memory: O(n²)** for attention matrix

**Scaling:**
```
n=512:  262K values
n=1024: 1M values (4× more)
n=2048: 4M values (16× more!)
```

Quadratic scaling is the key limitation for long sequences.

---

#### Q3: Your attention weights sum to values > 1. What went wrong?

**Answer:**

**Common Mistakes:**

1. **Wrong Softmax Dimension:**
```python
# WRONG: Softmax over batch
attn = torch.softmax(scores, dim=0)

# CORRECT: Softmax over keys
attn = torch.softmax(scores, dim=-1)
```

2. **Mask Applied After Softmax:**
```python
# WRONG
attn = torch.softmax(scores, dim=-1)
attn = attn.masked_fill(mask == 0, 0)  # Rows don't sum to 1!

# CORRECT  
scores = scores.masked_fill(mask == 0, -1e9)  # Before softmax
attn = torch.softmax(scores, dim=-1)
```

---

#### Q4: Model works on length 512 during training but fails on length 1024 at inference. Why?

**Answer:**

**Most Likely: Fixed Positional Embeddings**

```python
# Training
model = Transformer(max_len=512)
pos_embedding = nn.Embedding(512, d_model)  # Fixed!

#Inference
long_input = torch.randn(1, 1024, d_model)
# ERROR: Position 513 doesn't exist!
```

**Solutions:**
1. Train with larger max_len (2048)
2. Use sinusoidal encodings (extrapolate to any length)
3. Use RoPE or relative positional encodings

---

#### Q5: How do you visualize and interpret attention weights?

**Answer:**

**Extract Attention:**
```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
outputs = model(**inputs)
attentions = outputs.attentions  # (num_layers, batch, heads, seq_len, seq_len)
```

**What to Look For:**

1. **Diagonal Pattern**: Self-attention (preserving information)
2. **Local Attention**: Learning syntax (early layers)
3. **Long-Range**: Coreference, semantics (late layers)
4. **Head Specialization**: Different heads learn different patterns

**Red Flags:**
- All uniform (not learning)
- All random (hasn't converged)
- Over-reliance on special tokens

---

### Production Challenges

**Challenge: OOM on Long Sequences**

Attention memory grows quadratically:
- 512 tokens: 150MB
- 2048 tokens: 2.4GB

**Solutions:**
1. Gradient checkpointing
2. Flash Attention
3. Sparse attention (Longformer)
4. Chunking/sliding window

---

### Key Takeaways

1. **Math matters**: Understand QK^T/√d_k and why scaling is critical
2. **O(n²) complexity**: Fundamental bottleneck for long sequences
3. **Debug systematically**: Softmax dimension, masking order
4. **Visualization helps**: Attention patterns show what model learned
5. **Production**: Memory scaling requires efficient variants
