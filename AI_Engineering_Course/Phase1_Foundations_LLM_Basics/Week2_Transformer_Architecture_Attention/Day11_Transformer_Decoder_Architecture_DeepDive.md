# Day 11: Transformer Decoder Architecture
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Causal Masking: Implementation Details

**Naive Implementation:**

```python
# Create O(n²) mask matrix
mask = torch.tril(torch.ones(seq_len, seq_len))

# Apply before softmax
scores = scores.masked_fill(mask == 0, -1e9)
```

**Problem:** For long sequences (n=2048), mask is 2048² = 4M elements!

**Optimized: Attention with Blocks**

```python
# Process in blocks to reduce memory
block_size = 256

for i in range(0, seq_len, block_size):
    # Process block [i:i+block_size]
    # Only materialize mask for this block
    block_mask = create_causal_mask(block_size)
    ...
```

**Flash Attention Approach:**

Never materialize full attention matrix - compute on-the-fly in fused kernel.

### KV Cache: Memory Analysis

**Without KV Cache:**

```python
# Step t: Recompute K,V for all previous tokens
# Memory: O(seq_len × d_model) each step
# Compute: O(seq_len × d_model²) each step

for t in range(max_len):
    K_all = compute_keys(tokens[:t+1])  # Recompute all!
    V_all = compute_values(tokens[:t+1])
    ...
```

**With KV Cache:**

```python
# Cache K,V from previous steps
# Memory: O(max_len × d_model) total
# Compute: O(1 × d_model²) per step

K_cache = []  # List of cached keys
V_cache = []

for t in range(max_len):
    K_new = compute_keys(tokens[t:t+1])  # Only new token!
    V_new = compute_values(tokens[t:t+1])
    
    K_cache.append(K_new)
    V_cache.append(V_new)
    
    K_all = torch.cat(K_cache, dim=1)  # Concatenate cached
    V_all = torch.cat(V_cache, dim=1)
    ...
```

**Memory Trade-off:**

```
KV cache size per layer per head:
2 (K+V) × seq_len × d_k × sizeof(float16)

For GPT-3 (175B):
- 96 layers
- 96 heads per layer
- d_k = 128
- seq_len = 2048
- float16 = 2 bytes

Total: 2 × 2048 × 128 × 2 × 96 × 96 = 9.6 GB!

For long sequences, KV cache dominates memory usage!
```

**PagedAttention (vLLM):**

Store KV cache in non-contiguous memory blocks:

```python
# Traditional KV cache: Contiguous memory
K_cache = torch.zeros(max_len, d_k)  # Pre-allocate max length

# PagedAttention: Paged memory (like virtual memory)
K_cache_pages = []  # List of pages
page_size = 16  # tokens

for t in range(max_len):
    page_idx = t // page_size
    offset = t % page_size
    
    if len(K_cache_pages) <= page_idx:
        K_cache_pages.append(torch.zeros(page_size, d_k))
    
    K_cache_pages[page_idx][offset] = compute_key(token[t])

# Benefits:
# - Only allocate pages as needed
# - Can share pages across batches (prefix caching)
# - Reduces memory fragmentation
```

### Cross-Attention: Information Flow Analysis

In encoder-decoder models (T5, BART):

```python
# Encoder: Process input (bidirectional)
encoder_output = encoder(source_tokens)

# Decoder: Generate output (causal)
for t in range(target_len):
    # Self-attention: Attend to previous decoder tokens (causal)
    decoder_t = masked_self_attention(decoder_tokens[:t+1])
    
    # Cross-attention: Attend to ALL encoder tokens
    decoder_t = cross_attention(
        query=decoder_t,
        key=encoder_output,  # No masking - can see all!
        value=encoder_output
    )
    
    next_token = predict(decoder_t)
```

**Attention Pattern Visualization:**

```
Translation: "The cat" (English) → "Le chat" (French)

Encoder (English): [The, cat]
    Bidirectional attention:
    The → [The, cat]  # Sees both
    cat → [The, cat]  # Sees both

Decoder (French): [Le, chat]
    Self-attention (causal):
    Le → [Le]              # Only sees Le
    chat → [Le, chat]      # Sees Le, chat
    
    Cross-attention (full):
    Le → [The, cat]    # Attends to "The"
    chat → [The, cat]  # Attends to "cat"
```

### GPT Pre-Norm vs BERT Post-Norm

**GPT-2 (Pre-norm):**

```python
x = x + self_attn(layer_norm(x))
x = x + ffn(layer_norm(x))
```

**Why Pre-norm for Decoder?**

1. **Training Stability:**
   ```
   Decoder trained autoregressively
   Gradients flow through many time steps
   Pre-norm more stable for long sequences
   ```

2. **Initialization:**
   ```
   At initialization:
   layer_norm(x) ≈ normalized input
   self_attn, ffn start close to zero
   
   x = x + small_perturbation
   # Residual dominates initially → Stable
   ```

3. **Layer-wise Learning Rates:**
   ```
   With pre-norm, different layers can have different effective learning rates
   Upper layers naturally get smaller updates
   ```

**BERT (Post-norm):**

```python
x = layer_norm(x + self_attn(x))
x = layer_norm(x + ffn(x))
```

**Why Post-norm for Encoder?**

1. **Better Final Performance:**
   ```
   Post-norm normalizes the output
   Provides better regularization
   Slightly better accuracy when converged
   ```

2. **Bidirectional Context:**
   ```
   Encoder processes entire sequence at once
   No autoregressive instability
   Can afford less stable but higher-performing configuration
   ```

### Autoregressive Training: Teacher Forcing

**Standard Training:**

```python
# Input: "The cat sat"
# Target: "cat sat on"

for t in range(seq_len):
    # Use ground truth as input
    decoder_input = target_tokens[:t+1]
    logits = decoder(decoder_input)
    
    # Predict next token
    loss = cross_entropy(logits[t], target_tokens[t+1])
```

**Problem: Exposure Bias**

```
Training: Always sees ground truth previous tokens
Inference: Sees model's own predictions

If model makes mistake early:
- Training: Doesn't learn to recover (never saw mistakes)
- Inference: Error compounds → Poor generation
```

**Solution 1: Scheduled Sampling**

```python
teacher_forcing_ratio = 0.8  # Start high

for t in range(seq_len):
    if random.random() < teacher_forcing_ratio:
        decoder_input = target_tokens[t]  # Ground truth
    else:
        decoder_input = model_prediction[t-1]  # Model's prediction
    
    logits = decoder(decoder_input)
    loss = cross_entropy(logits, target_tokens[t+1])

# Gradually reduce teacher_forcing_ratio during training
teacher_forcing_ratio = max(0.5, teacher_forcing_ratio * 0.99)
```

**Solution 2: Prefix Tuning**

Train model to continue from any prefix:

```python
# Randomly truncate target during training
truncate_len = random.randint(1, seq_len)
decoder_input = target_tokens[:truncate_len]

# Continue from truncation point
for t in range(truncate_len, seq_len):
    logits = decoder(decoder_input)
    decoder_input = torch.cat([decoder_input, logits.argmax(dim=-1)])
```

### Sampling Strategies for Generation

**Greedy Decoding:**

```python
next_token = torch.argmax(logits)
```

**Problem:** Repetitive, deterministic

**Top-K Sampling:**

```python
# Keep only top-k most likely tokens
top_k_logits, top_k_indices = torch.topk(logits, k=50)
probs = torch.softmax(top_k_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

**Top-P (Nucleus) Sampling:**

```python
# Keep smallest set of tokens with cumulative probability >= p
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

# Remove tokens with cumulative probability > p
sorted_indices_to_remove = cumulative_probs > p
sorted_logits[sorted_indices_to_remove] = -float('Inf')

probs = torch.softmax(sorted_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

**Temperature Scaling:**

```python
# temperature > 1: More random
# temperature < 1: More deterministic
# temperature = 1: No change

logits_scaled = logits / temperature
probs = torch.softmax(logits_scaled, dim=-1)
```

### GPT-3 Architecture Details

**Configuration:**

```
GPT-3 175B:
- Layers: 96
- d_model: 12,288
- Heads: 96
- d_k: 128 (12,288 / 96)
- d_ff: 49,152 (4× expansion)

Parameters:
- Embedding: 50257 × 12288 = 617M
- Attention (per layer): 4 × 12288² = 603M
- FFN (per layer): 2 × 12288 × 49152 = 1.2B
- Total per layer: ~1.8B
- 96 layers: ~175B total
```

**Parallelization Strategy:**

```
Tensor Parallelism: Split attention heads across GPUs
Pipeline Parallelism: Different layers on different GPUs
Data Parallelism: Different batches on different GPUs

Example (8 GPUs):
- GPU 0-1: Tensor parallel (split heads of layers 1-24)
- GPU 2-3: Layers 25-48
- GPU 4-5: Layers 49-72
- GPU 6-7: Layers 73-96
```

### Summary: Architecture Evolution

```
2017: Original Transformer (Encoder-Decoder)
      - Translation focus
      - Both encoder and decoder

2018: BERT (Encoder-Only)
      - Understanding focus
      - Bidirectional attention

2018: GPT (Decoder-Only)
      - Generation focus  
      - Causal attention

2019: GPT-2 (Decoder-Only, scaled)
      - Pre-norm for stability
      - Scaled to 1.5B parameters

2020: GPT-3 (Decoder-Only, massively scaled)
      - 175B parameters
      - Emergent abilities from scale

2023: LLaMA, Mistral (Decoder-Only, optimized)
      - GQA (Grouped-Query Attention)
      - RoPE (Rotary Position Embeddings)
      - Efficient at smaller sizes

Trend: Decoder-only dominates (simpler, scales better)
```

Decoder architecture is the foundation of modern LLMs!
