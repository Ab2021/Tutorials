# Day 15.3: Multi-Head Attention & Positional Encoding - A Practical Guide

## Introduction: The Core Components of the Transformer

We have seen the high-level architecture of the Transformer and taken a deep dive into the Scaled Dot-Product Attention mechanism. Now, it's time to zoom in on the two components that truly define a Transformer block: **Multi-Head Attention** and **Positional Encoding**.

*   **Multi-Head Attention** is the workhorse. It's the engine that allows the model to weigh the importance of different tokens and build contextual representations.
*   **Positional Encoding** is the essential fix. It's the component that injects the information about word order that the self-attention mechanism otherwise lacks.

This guide will provide a final, focused, practical look at these two crucial components, solidifying their implementation details and their roles within the overall architecture.

**Today's Learning Objectives:**

1.  **Solidify the Implementation of Multi-Head Attention:** Review the from-scratch implementation, focusing on the reshaping and projection steps that enable parallel attention.
2.  **Understand the `nn.MultiheadAttention` Module:** See how PyTorch provides a single, highly optimized layer that encapsulates this entire complex process.
3.  **Revisit Positional Encodings:** Review the sine/cosine-based encoding scheme and understand its properties.
4.  **See How They Fit Together:** Trace the data flow from the initial word embedding, through the addition of positional encoding, and into the Multi-Head Attention layer.

---

## Part 1: Multi-Head Attention - A Detailed Review

Let's revisit the implementation from the previous guide, but with more detailed comments to emphasize each step.

**The Goal:** Instead of performing one big attention calculation, we split the embedding dimension into multiple "heads" and perform attention independently and in parallel for each head. This allows each head to focus on different types of relationships.

### 1.1. From-Scratch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("--- Part 1: Multi-Head Attention Review ---")

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # A single large linear layer to get Q, K, V from the input
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        # A final linear layer to project the concatenated heads back to the original dimension
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape

        # 1. Project and Reshape for Multi-Head
        qkv = self.qkv_proj(x) # Shape: [batch, seq_len, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # Shape: [batch, num_heads, seq_len, 3 * head_dim]
        q, k, v = qkv.chunk(3, dim=-1) # Split into Q, K, V. Each is [batch, num_heads, seq_len, head_dim]

        # 2. Calculate Attention Scores
        # Transpose k for dot product: [batch, num_heads, head_dim, seq_len]
        k_t = k.transpose(-2, -1)
        # scores shape: [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim)

        # 3. Apply Mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Get Attention Weights
        attention_weights = F.softmax(scores, dim=-1)

        # 5. Get Context Vector
        # context shape: [batch, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_weights, v)

        # 6. Concatenate Heads and Project
        # Reshape to combine heads: [batch, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2).contiguous()
        # [batch, seq_len, embed_dim]
        context = context.reshape(batch_size, seq_length, self.embed_dim)
        
        # 7. Final Linear Projection
        output = self.o_proj(context)

        return output
```

### 1.2. Using PyTorch's `nn.MultiheadAttention`

While the from-scratch implementation is a great learning tool, you should always use PyTorch's built-in layer in practice. It is highly optimized and handles all the reshaping and projections internally.

```python
print("\n--- Using nn.MultiheadAttention ---")

# Parameters
embed_dim = 256
num_heads = 8
batch_size = 32
seq_len = 50

# Create the built-in layer
# Note: batch_first=True is crucial for using the (batch, seq, feature) dimension order.
mha_layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

# Create dummy data
# For self-attention, the query, key, and value are all the same.
input_seq = torch.randn(batch_size, seq_len, embed_dim)

# The layer returns the output and the attention weights
output, attn_weights = mha_layer(query=input_seq, key=input_seq, value=input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}") # (batch, seq_len, seq_len)
# Note: The built-in layer averages the weights across the heads by default.
```

---

## Part 2: Positional Encoding - A Detailed Review

**The Problem:** Self-attention is permutation-invariant. It has no idea if a word is at the beginning or the end of a sentence.

**The Solution:** We create a unique vector for each position (`pos`) in the sequence and add it to the word embedding. This injects the necessary positional information.

### 2.1. The Sine/Cosine Formula

The original paper used a clever combination of sine and cosine functions:

`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))` (for even dimensions)
`PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))` (for odd dimensions)

*   `pos`: The position of the token in the sequence (0, 1, 2, ...).
*   `i`: The dimension within the embedding vector (0, 1, 2, ...).
*   `d_model`: The total dimension of the embedding.

**Why this formula?**
It has a wonderful property: for any offset `k`, `PE(pos+k)` can be represented as a linear function of `PE(pos)`. This means the model can easily learn to understand relative positions, which is more important than knowing the absolute position of a word.

### 2.2. Visualizing the Positional Encodings

Let's plot the positional encoding matrix to see what it looks like.

```python
import matplotlib.pyplot as plt
import numpy as np

print("\n--- Part 2: Visualizing Positional Encodings ---")

# We reuse the PositionalEncoding class from the first guide of this Day.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

# Create an instance
pos_encoder = PositionalEncoding(d_model=512, max_len=100)

# Get the positional encoding matrix
pe_matrix = pos_encoder.pe.squeeze(0).cpu().numpy()

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.pcolormesh(pe_matrix, cmap='viridis')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position in Sequence')
plt.title('Sine/Cosine Positional Encodings')
plt.colorbar()
plt.show()
```

**Interpretation:** Each row is the unique positional vector for a position in the sequence. The columns vary as sine and cosine waves with different frequencies, creating a unique signature for each position.

---

## Part 3: Putting It All Together - The Input to a Transformer Block

Now let's trace the full data flow for the input to the first Transformer Encoder block.

```python
print("\n--- Part 3: The Full Input Pipeline ---")

# --- 1. Parameters ---
vocab_size = 1000
embed_dim = 512
batch_size = 4
seq_len = 50

# --- 2. Input Data ---
# A batch of integer sequences
input_indices = torch.randint(1, vocab_size, (batch_size, seq_len))

# --- 3. The Layers ---
embedding_layer = nn.Embedding(vocab_size, embed_dim)
positional_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
mha_layer = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

# --- 4. The Data Flow ---
# Step a: Get word embeddings
word_embs = embedding_layer(input_indices)
print(f"Shape after Embedding Layer: {word_embs.shape}")

# Step b: Add positional encodings
final_embs = positional_encoder(word_embs)
print(f"Shape after Positional Encoding: {final_embs.shape}")

# Step c: Pass through Multi-Head Attention
# This is the first sub-layer of a Transformer block.
attn_output, _ = mha_layer(final_embs, final_embs, final_embs)
print(f"Shape after Multi-Head Attention: {attn_output.shape}")

# The rest of the Transformer block would involve the residual connection,
# layer normalization, the feed-forward network, and another residual/norm.
```

## Conclusion

Multi-Head Attention and Positional Encodings are the two pillars upon which the Transformer architecture stands. 

*   **Multi-Head Attention** provides a powerful, parallelizable mechanism for the model to learn complex relationships between all pairs of tokens in a sequence.
*   **Positional Encoding** is the simple but critical injection of order information that the attention mechanism itself lacks.

By mastering the implementation and purpose of these two components, you have gained a deep and practical understanding of the inner workings of the architecture that defines modern NLP.

## Self-Assessment Questions

1.  **Multi-Head Reshaping:** In the from-scratch Multi-Head Attention implementation, what is the purpose of the `reshape` and `permute` operations after the initial QKV projection?
2.  **`nn.MultiheadAttention`:** When using the built-in `nn.MultiheadAttention` layer for self-attention, what three arguments do you pass for `query`, `key`, and `value`?
3.  **Positional Encoding Addition:** Why are the positional encodings *added* to the word embeddings instead of being concatenated?
4.  **Learnable Positional Encodings:** The original paper used a fixed sine/cosine function. What is a simpler, alternative way to implement positional encodings that is often used in modern models?
5.  **Data Flow:** What is the very first operation that happens to a word embedding vector before it is processed by the attention mechanism in a Transformer?

