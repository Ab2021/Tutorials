# Day 12: Positional Encodings & Embeddings
## Core Concepts & Theory

### The Permutation Invariance Problem

**Transformers are Set Processors:**
Unlike RNNs (which process sequentially) or CNNs (which process locally), the standard self-attention mechanism is **permutation invariant**.

```python
# Attention calculation
scores = Q @ K.T  # Dot product between all pairs
output = softmax(scores) @ V

# If we shuffle the input words:
# "The cat sat" -> "sat cat The"
# The attention scores between "The" and "cat" remain exactly the same!
# The output vector for "The" would be identical (just in a different position).
```

**Why this is a problem:**
Language depends heavily on order.
- "The dog bit the man" ≠ "The man bit the dog"
- Without position information, the model sees these as identical "bags of words".

**Solution:** Inject positional information into the embeddings *before* the first transformer layer.

### Absolute Positional Encodings

**1. Sinusoidal Positional Encodings (Original Transformer)**

Proposed by Vaswani et al. (2017). Uses fixed sine and cosine functions of different frequencies.

**Formula:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Where:
- `pos` is the position in the sequence (0, 1, 2...)
- `i` is the dimension index (0, 1, ..., d_model/2)
- `d_model` is the embedding dimension

**Properties:**
- **Fixed:** No parameters to learn.
- **Extrapolatable:** Can theoretically handle sequence lengths longer than seen during training (though performance degrades).
- **Relative distances:** For any fixed offset `k`, `PE(pos+k)` can be represented as a linear function of `PE(pos)`.

**Implementation:**

```python
import torch
import math

def create_sinusoidal_embeddings(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Usage
# x: (batch, seq_len, d_model)
# pe: (max_len, d_model)
x = x + pe[:x.size(1), :].to(x.device)
```

**2. Learned Positional Embeddings (BERT, GPT-2)**

Instead of fixed functions, learn a vector for each position.

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)
```

**Pros:**
- Easy to implement.
- Model learns optimal encodings for the task.

**Cons:**
- **Not extrapolatable:** Cannot handle sequences longer than `max_len` defined at training time.
- **Data hungry:** Need enough examples of each position to learn well.

### Relative Positional Encodings

Instead of "Token A is at position 5", encode "Token A is 3 positions after Token B".

**Motivation:**
In language, relative distance often matters more than absolute position. The relationship between a verb and its object depends on their distance, not whether they appear at index 5 or 500.

**Shaw et al. (2018) Approach:**
Add a learnable vector `a_{ij}` to the attention mechanism representing the distance `j-i`.

```
Attention(Q, K, V) = softmax((Q @ K.T + RelativeBias) / sqrt(d_k)) @ V
```

**T5 (Text-to-Text Transfer Transformer) Bias:**
T5 uses a simplified relative bias added to the attention logits. The bias is shared across layers but different for each head. It uses a "logarithmic bucket" scheme to handle long distances efficiently.

### Rotary Positional Embeddings (RoPE)

The modern standard (used in LLaMA, PaLM, GPT-NeoX).

**Core Idea:**
Encode absolute position by **rotating** the embedding vector.
If we rotate vector $q$ by angle $m\theta$ and vector $k$ by angle $n\theta$, their dot product depends only on the relative angle $(m-n)\theta$.

**Properties:**
- **Absolute position awareness:** via the rotation angle.
- **Relative position awareness:** Dot product naturally encodes relative distance.
- **Better extrapolation:** Generalizes to longer sequences better than learned embeddings.
- **Multiplicative:** Applied to Q and K, not added to input embeddings.

**Implementation Concept:**
```python
def apply_rotary_pos_emb(x, cos, sin):
    # x: (batch, seq_len, heads, head_dim)
    # Rotate every pair of dimensions
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

### ALiBi (Attention with Linear Biases)

**Core Idea:**
Do not add positional embeddings to the input. Instead, add a static, non-learned bias to the attention scores based on distance.

**Formula:**
```
score(q_i, k_j) = q_i · k_j - m · |i - j|
```
Where `m` is a head-specific slope fixed before training (e.g., geometric sequence like $1/2^1, 1/2^2, ...$).

**Pros:**
- **Extrapolation:** Generalizes to much longer sequences than trained on (e.g., train on 1024, inference on 16k).
- **Simple:** No parameters to learn.
- **Efficient:** Fast to compute.

### Summary Comparison

| Method | Type | Extrapolation | Used In |
| :--- | :--- | :--- | :--- |
| **Sinusoidal** | Absolute, Fixed | Moderate | Original Transformer |
| **Learned** | Absolute, Learned | Poor | BERT, GPT-2 |
| **Relative (T5)** | Relative, Learned | Good | T5 |
| **RoPE** | Hybrid, Fixed | Excellent | LLaMA, PaLM, Mistral |
| **ALiBi** | Relative, Fixed | Excellent | MPT, BLOOM |

### Why This Matters for Engineering

1.  **Context Window Limits:** Your choice of PE determines if you can extend context length at inference time.
2.  **Performance:** RoPE and ALiBi generally outperform absolute embeddings on long contexts.
3.  **Fine-tuning:** When fine-tuning on longer sequences, absolute embeddings fail; RoPE/ALiBi adapt better.

### Next Steps
In the Deep Dive, we will explore the mathematics of RoPE and why it has become the default for modern LLMs.
