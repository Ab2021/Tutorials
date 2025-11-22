# Day 11: Transformer Decoder Architecture
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain causal (masked) self-attention in decoders. Why is it necessary?

**Answer:**

**Causal attention** prevents tokens from attending to future tokens:

```python
# Causal mask (lower triangular)
mask = torch.tril(torch.ones(seq_len, seq_len))
# [[1, 0, 0],
#  [1, 1, 0],
#  [1, 1, 1]]

scores = scores.masked_fill(mask == 0, -1e9)
attn = softmax(scores)  # Future positions have ~0 weight
```

**Why Necessary?**

**1. Match Training and Inference:**

```
Training: Process full sequence
- "The cat sat" → predict "on"
- Must not see "on" when predicting it!

Inference: Generate left-to-right
- Have: "The cat sat"
- Predict: "on" (don't know future)

Causal mask ensures training matches inference
```

**2. Autoregressive Property:**

```python
P(sequence) = P(w1) × P(w2|w1) × P(w3|w1,w2) × ...

Each token depends only on previous tokens
Causal attention enforces this dependency structure
```

**Without Causal Mask (Wrong):**

```
Token "cat" attends to "sat" (future)
→ Learns to "cheat" using future information
→ Fails at inference (future unavailable)
```

---

#### Q2: Explain KV caching. How much speedup does it provide for generation?

**Answer:**

**Problem:**

Naive generation recomputes attention for all previous tokens:

```python
# Step 1: Process "The" → Q1, K1, V1
# Step 2: Process "The cat" → Recompute K1,V1 + compute K2,V2
# Step 3: Process "The cat sat" → Recompute K1,V1,K2,V2 + K3,V3

# O(n²) recomputation!
```

**KV Caching Solution:**

```python
# Cache Keys and Values from previous steps
kv_cache = {'keys': [], 'values': []}

for step in range(max_len):
    # Only compute K,V for NEW token
    K_new = compute_key(new_token)
    V_new = compute_value(new_token)
    
    # Cache
    kv_cache['keys'].append(K_new)
    kv_cache['values'].append(V_new)
    
    # Use full cached K,V for attention
    K_all = torch.cat(kv_cache['keys'])
    V_all = torch.cat(kv_cache['values'])
    
    # Only compute Q for new token
    Q_new = compute_query(new_token)
    attn = softmax(Q_new @ K_all.T) @ V_all
```

**Speedup:**

```
Without KV cache:
Step 50: Process 50 tokens
Compute: 50 × (K,V computation) = 50× work

With KV cache:
Step 50: Process 1 token
Compute: 1 × (K,V computation) = 1× work

Speedup: 50× for 50-token generation!
```

**Memory Cost:**

```
KV cache per layer:
2 (K+V) × seq_len × d_k × num_heads

GPT-2 (12 layers, 12 heads, d_k=64, seq_len=1024):
Memory: 2 × 1024 × 64 × 12 × 12 = 18M values × 2 bytes = 36MB

GPT-3 (96 layers, 96 heads, d_k=128, seq_len=2048):
Memory: 2 × 2048 × 128 × 96 × 96 ≈ 10GB!
```

**Interview Follow-up:**
*Q: How do you handle KV cache for batched generation?*

**A:** 
- Naive: Separate cache per batch element
- Optimized (PagedAttention): Share cache blocks for common prefixes
- Example: Batching "Translate: Hello" and "Translate: Hi" shares "Translate:" cache

---

#### Q3: Compare decoder-only (GPT) vs encoder-decoder (T5) models. When would you use each?

**Answer:**

**Decoder-Only (GPT, LLaMA):**

Architecture:
- Only causal self-attention
- Simpler: One module type

Training:
- Language modeling (predict next token)
- Unsupervised (just needs text)

Use Cases:
- General text generation
- Continuation tasks
- Few-shot learning via prompting

**Encoder-Decoder (T5, BART):**

Architecture:
- Encoder: Bidirectional attention
- Decoder: Causal attention + cross-attention
- More complex: Two modules

Training:
- Seq2seq tasks (input → output)
- Often requires paired data

Use Cases:
- Translation (explicit input/output)
- Summarization (document → summary)
- Question answering (context + question → answer)

**When to Use:**

| Task | Choice | Reasoning |
|------|--------|-----------|
| Open-ended generation | Decoder-only | Simple, flexible via prompting |
| Translation | Either | Encoder-decoder traditional, but GPT-4 works via prompting |
| Summarization | Either | T5-style if have paired data, GPT-style for zero-shot |
| Classification | Encoder-only (BERT) | Don't need generation |
| General purpose | Decoder-only | Most flexible, single architecture |

**Modern Trend:**

Decoder-only dominates (GPT-3, GPT-4, LLaMA, Mistral):
- **Simpler** (one architecture)
- **Scales better** (no encoder overhead)
- **More flexible** (handle any task via prompting)

Encoder-decoder still useful for:
- Translation (dedicated systems)
- Tasks with clear input/output structure

---

#### Q4: Your GPT model generates repetitive text. Debug and propose solutions.

**Answer:**

**Symptoms:**

```
Input: "The cat"
Output: "The cat sat on the mat on the mat on the mat on the mat..."

Repetition loop!
```

**Root Causes:**

**1. Greedy Decoding:**

```python
# Always picks most likely token
next_token = logits.argmax()

# Problem: Local optimum, no exploration
```

**Solution: Sampling**

```python
# Top-p (nucleus) sampling
probs = torch.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, 1)
```

**2. Attention Collapse:**

```python
# Check attention patterns
attn_weights = model.get_attention_weights()

# If attention focuses only on recent tokens:
# Position: 0    1    2    3    4    5
# Weights: [0.0, 0.0, 0.0, 0.1, 0.4, 0.5]
#                              ↑Recent only

# Model "forgets" earlier context → Repetition
```

**Solution:** 
- Check position embeddings (might have decayed)
- Increase context window
- Use attention regularization

**3. Low Temperature:**

```python
temperature = 0.1  # Too low!
# Makes distribution very peaked → Repetitive

# Solution: Increase temperature
temperature = 0.7  # More diversity
```

**4. Repetition Penalty:**

```python
def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    for token in set(generated_tokens):
        logits[token] /= penalty  # Penalize already-generated tokens
    return logits

# During generation
logits = model(input_ids)
logits = apply_repetition_penalty(logits, input_ids, penalty=1.2)
next_token = sample(logits)
```

**5. Inadequate Training:**

```
If model wasn't trained on diverse text:
→ Learns repetitive patterns
→ Solution: Better training data
```

**Production Solution:**

```python
def generate_with_anti_repetition(model, prompt, max_len=50):
    input_ids = tokenize(prompt)
    
    for _ in range(max_len):
        logits = model(input_ids)
        
        # Apply repetition penalty
        for token in set(input_ids[-20:]):  # Last 20 tokens
            logits[0, -1, token] /= 1.2
        
        # Temperature sampling
        probs = torch.softmax(logits[0, -1] / 0.7, dim=-1)
        
       # Top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum <= 0.9
        mask[0] = True  # Keep at least one
        
        filtered_probs = sorted_probs[mask]
        filtered_probs /= filtered_probs.sum()
        
        next_token = sorted_indices[mask][torch.multinomial(filtered_probs, 1)]
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)])
        
        if next_token == EOS_TOKEN:
            break
    
    return input_ids
```

---

#### Q5: How do you optimize decoder inference latency for production?

**Answer:**

**Baseline: GPT-2 (124M params)**

```
Latency: 100ms per token
Throughput: 10 tokens/sec
```

**Optimizations:**

**1. KV Caching:**

```
Without: 100ms/token
With: 20ms/token (5× faster)

Implementation: Built into most frameworks
```

**2. Quantization:**

```python
# INT8 or FP16
model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Result: 2-3× faster, 4× less memory
Latency: 20ms → 8ms
```

**3. Batching:**

```python
# Process multiple sequences simultaneously
batch = ["Translate: Hello", "Translate: Hi", "Translate: Bye"]

# Without batching: 3 × 8ms = 24ms total
# With batching: 12ms total (2× faster)

# Use dynamic batching (continuous batching in vLLM)
```

**4. Speculative Decoding:**

```python
# Use small "draft" model to propose tokens
# Use large "target" model to verify

draft_tokens = small_model.generate(5)  # Fast, propose 5 tokens
verified = large_model.verify(draft_tokens)  # Batch verification

# If verified: Accept all 5 (5× speedup)
# If  rejected: Fall back to standard generation
```

**5. Flash Attention / Flash Decoding:**

```
Fused kernels, optimized memory access
2-4× speedup on GPUs

Latency: 8ms → 3ms
```

**6. Model Distillation:**

```
Teacher: GPT-2 Large (774M)
Student: GPT-2 Small (124M)

Latency: 8ms → 2ms (4× faster)
Quality: 95% of teacher
```

**Combined Results:**

```
Baseline: 100ms/token

+ KV cache: 20ms
+ Quantization: 8ms
+ Flash Attention: 3ms
+ Batching (10 sequences): 0.5ms/token effective

Total: 200× speedup!
```

---

### Production Challenges

**Challenge: KV Cache Memory Explosion**

**Scenario:**
- Serving 1000 concurrent users
- Each generates 2048 tokens
- GPT-3-scale model

**Memory:**
```
Per user: 10GB KV cache
1000 users: 10TB!
```

**Solutions:**
1. PagedAttention (vLLM) - share prefixes
2. Offload to CPU/disk for inactive users
3. Limit max sequence length
4. Use smaller models (distillation)

---

### Key Takeaways

1. **Causal attention enables autoregressive generation**
2. **KV caching provides 10-50× speedup** (essential for production)
3. **Decoder-only models dominate** (simpler, more flexible)
4. **Repetition requires sampling strategies** (temperature, top-p, penalties)
5. **Production optimization** Multi-faceted (caching, quantization, batching, speculative decoding)
