# Day 11: Transformer Decoder Architecture
## Core Concepts & Theory

### Decoder Architecture Overview

The Transformer decoder is designed for autoregressive generation:
- Predicts next token based on previous tokens
- Uses **masked self-attention** (causal attention)
- Optionally uses **cross-attention** to encoder (in encoder-decoder models)

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Masked self-attention (causal)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Cross-attention (optional, for encoder-decoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (if encoder output provided)
        if encoder_output is not None:
            cross_output = self.cross_attn(x, encoder_output, encoder_output, cross_attn_mask)
            x = self.norm2(x + self.dropout(cross_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x
```

### Causal (Masked) Self-Attention

**Key Difference from Encoder:**

Decoder can only attend to **previous tokens**, not future ones.

```python
def create_causal_mask(seq_len):
    """Create lower triangular mask for causal attention"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # mask:
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    return mask

# In attention
scores = Q @ K.T / sqrt(d_k)
scores = scores.masked_fill(causal_mask == 0, -1e9)  # Mask future
attn = softmax(scores)
```

**Why Causal?**

For autoregressive generation (language modeling):

```
Given: "The cat sat"
Predict: "on"

During training:
- Token "The" can only see "The"
- Token "cat" can only see "The", "cat"  
- Token "sat" can only see "The", "cat", "sat"

This matches inference where we generate left-to-right!
```

### Cross-Attention (Encoder-Decoder)

In encoder-decoder models (T5, BART), decoder attends to encoder output:

```python
# Cross-attention
Q = decoder_hidden  # Query from decoder
K = V = encoder_output  # Keys/Values from encoder

# No causal mask for cross-attention!
# Decoder can attend to ALL encoder positions
cross_attn = softmax(Q @ K.T / sqrt(d_k)) @ V
```

**Use Case: Machine Translation**

```
Encoder input (English): "The cat sat on the mat"
Encoder output: Contextualized representations

Decoder (generating French):
Step 1: Generate "Le" - attends to "The" via cross-attention
Step 2: Generate "chat" - attends to "cat"
Step 3: Generate "s'est" - attends to "sat"
...
```

### GPT Architecture (Decoder-Only)

Modern LLMs (GPT-2, GPT-3, LLaMA) use **decoder-only** architecture:

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model=768, num_heads=12, d_ff=3072):
        super().__init__()
        # Only masked self-attention (no cross-attention)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT uses GELU, not ReLU
            nn.Linear(d_ff, d_model)
        )
        
        # Pre-norm (GPT-2 style)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, causal_mask):
        # Pre-norm + masked self-attention
        x = x + self.self_attn(self.ln1(x), causal_mask=causal_mask)
        
        # Pre-norm + FFN
        x = x + self.ffn(self.ln2(x))
        
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, num_layers=12):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            GPTBlock(d_model) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        
        # Embeddings
        token_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len))
        x = token_emb + pos_emb
        
        # Causal mask
        causal_mask = create_causal_mask(seq_len)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final layer norm + prediction
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
```

### Autoregressive Generation

**Training:**

```python
# Input: "The cat sat on the"
# Target: "cat sat on the mat"

input_ids = tokenize("The cat sat on the")
target_ids = tokenize("cat sat on the mat")

# Forward pass
logits = model(input_ids)  # (batch, seq_len, vocab_size)

# Loss: predict next token at each position
loss = cross_entropy(logits[:, :-1], target_ids)
```

**Inference (Generation):**

```python
def generate(model, prompt, max_length=50):
    input_ids = tokenize(prompt)
    
    for _ in range(max_length):
        # Forward pass
        logits = model(input_ids)  # (1, seq_len, vocab_size)
        
        # Get next token prediction
        next_token_logits = logits[0, -1, :]  # Last position
        next_token = torch.argmax(next_token_logits)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)])
        
        # Stop if EOS token
        if next_token == EOS_TOKEN:
            break
    
    return input_ids

# Usage
prompt = "The cat"
generated = generate(model, prompt)
# Output: "The cat sat on the mat"
```

### KV Caching for Efficient Generation

**Problem:**

Naive generation recomputes attention for all previous tokens each step:

```python
# Step 1: "The" → compute attention for "The"
# Step 2: "The cat" → recompute attention for "The", compute for "cat"
# Step 3: "The cat sat" → recompute attention for "The", "cat", compute for "sat"

# Wasteful! Keys and Values don't change for previous tokens
```

**Solution: Cache Keys and Values**

```python
class GPTWithKVCache(nn.Module):
    def forward(self, x, past_kv=None):
        # past_kv: cached (Keys, Values) from previous tokens
        
        if past_kv is None:
            # First step: compute K,V for all tokens
            K, V = self.compute_kv(x)
            past_kv = (K, V)
        else:
            # Later steps: only compute K,V for new token
            K_past, V_past = past_kv
            K_new, V_new = self.compute_kv(x[:, -1:])  # Only last token
            
            # Concatenate with cached
            K = torch.cat([K_past, K_new], dim=1)
            V = torch.cat([V_past, V_new], dim=1)
            past_kv = (K, V)
        
        # Compute attention using full K, V
        Q = self.compute_q(x[:, -1:])  # Only query for new token
        attn = softmax(Q @ K.T / sqrt(d_k)) @ V
        
        return attn, past_kv

# Generation with KV cache
past_kv = None
for step in range(max_length):
    output, past_kv = model(input_ids[:, -1:], past_kv)
    # Only processes 1 token, not entire sequence!
```

**Speedup:**

```
Without KV cache:
Step 50: Process 50 tokens → 50 × attention computation

With KV cache:
Step 50: Process 1 token → 1 × attention computation

50× speedup for generation!
```

### Decoder vs Encoder Comparison

| Aspect | Encoder | Decoder |
|--------|---------|---------|
| Attention Type | Bidirectional | Causal (masked) |
| Sees Future | Yes | No |
| Use Case | Understanding (classification, NER) | Generation (language modeling) |
| Examples | BERT, RoBERTa | GPT-2, GPT-3, LLaMA |
| Cross-Attention | No | Optional (encoder-decoder only) |

### Summary

Decoder architecture enables:
- **Autoregressive generation**: Predict one token at a time
- **Causal attention**: Prevents looking into future
- **KV caching**: Efficient generation (50× speedup)
- **Flexible**: Decoder-only (GPT) or encoder-decoder (T5)

Modern LLMs are predominantly decoder-only due to simplicity and scaling properties.

Next (Day 12): Positional encodings - how Transformers understand token order
