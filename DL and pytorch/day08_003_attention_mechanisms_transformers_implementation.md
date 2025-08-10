# Day 8.3: Attention Mechanisms & Transformers - A Practical Guide

## Introduction: The Bottleneck of Sequential Processing

RNNs, LSTMs, and GRUs revolutionized sequence modeling, but they have an inherent weakness: they process data **sequentially**. To understand the 100th word in a sentence, the model must process the first 99 words in order. This creates two problems:

1.  **It's slow:** The computation cannot be parallelized over the sequence length.
2.  **Information Bottleneck:** The model must compress all the information from the entire sequence into a single fixed-size hidden state vector. This can be a significant bottleneck for very long sequences.

The **Attention Mechanism** was introduced to solve this. First used in machine translation, Attention allows the model to directly look at and draw context from **all parts** of the input sequence at every step of the output generation, rather than relying on a single hidden state. This idea was so powerful that it led to the **Transformer** architecture, which dispenses with recurrence entirely and relies solely on attention.

This guide will provide a practical walkthrough of the attention mechanism and the core components of the Transformer architecture.

**Today's Learning Objectives:**

1.  **Grasp the Intuition Behind Attention:** Understand how attention allows a model to "focus" on the most relevant parts of an input sequence.
2.  **Learn the Query, Key, Value (QKV) Model:** Understand the core components of the Scaled Dot-Product Attention mechanism.
3.  **Implement a Simple Attention Layer:** Build a basic attention mechanism from scratch to see how the QKV model works in practice.
4.  **Explore the Transformer Architecture:** Learn about the main components of a Transformer block: Multi-Head Attention and the Position-wise Feed-Forward Network.
5.  **Use a `nn.TransformerEncoderLayer`:** See how to use PyTorch's built-in, highly optimized Transformer module.

--- 

## Part 1: The Attention Mechanism - A High-Level View

Imagine translating the English sentence "The cat sat on the mat" to French. When generating the French word "chat" (cat), you want the model to pay high **attention** to the English word "cat." When generating "tapis" (mat), you want it to focus on "mat."

Attention provides a way for the model to learn these focus scores automatically.

**The Process:**

1.  For each output step, the model generates a **Query** vector representing what it's currently looking for.
2.  It compares this Query to a set of **Key** vectors, where each Key corresponds to a word in the input sentence.
3.  The comparison (typically a dot product) produces a **score** for each input word, indicating its relevance to the current query.
4.  These scores are passed through a **Softmax** function to create a probability distribution, the **attention weights**. These weights sum to 1.
5.  The attention weights are used to compute a weighted sum of the **Value** vectors (which also correspond to the input words). The result is a single **context vector** that is a blend of the input information, weighted by relevance.

This context vector is then used to produce the output for the current time step.

![Attention Mechanism](https://i.imgur.com/yS4vj0M.png)

--- 

## Part 2: Implementing Scaled Dot-Product Attention

This is the specific type of attention used in the Transformer model. Let's build it from scratch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("--- Part 2: Scaled Dot-Product Attention ---")

def scaled_dot_product_attention(query, key, value):
    """
    Computes the attention scores and output.
    Args:
        query: Shape (..., seq_len_q, depth)
        key: Shape (..., seq_len_k, depth)
        value: Shape (..., seq_len_v, depth) (seq_len_k must equal seq_len_v)
    Returns:
        output: The context vector. Shape (..., seq_len_q, depth_v)
        attention_weights: The weights. Shape (..., seq_len_q, seq_len_k)
    """
    # Get the depth (embedding dimension) from the key tensor
    depth = key.shape[-1]
    
    # 1. Calculate scores: Q * K^T
    # Transpose the last two dimensions of the key tensor
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 2. Scale the scores
    # This is done to prevent the dot products from becoming too large, which can
    # push the softmax into regions with very small gradients.
    scaled_scores = scores / math.sqrt(depth)
    
    # 3. Apply Softmax to get attention weights
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    # 4. Compute the weighted sum of Value vectors
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# --- Usage Example ---
# Let's simulate a batch of sequences
batch_size = 4
seq_len = 10
embed_dim = 64

# In self-attention (used in Transformers), Q, K, and V are all the same tensor.
q = torch.randn(batch_size, seq_len, embed_dim)
k = torch.randn(batch_size, seq_len, embed_dim)
v = torch.randn(batch_size, seq_len, embed_dim)

context_vector, attention_weights = scaled_dot_product_attention(q, k, v)

print(f"Query/Key/Value shape: {q.shape}")
print(f"Output Context Vector shape: {context_vector.shape}")
print(f"Attention Weights shape: {attention_weights.shape}")
print(f"\nExample attention weights for the first word in the first batch:")
print(attention_weights[0, 0])
print(f"Sum of these weights: {attention_weights[0, 0].sum():.2f}") # Should sum to 1
```

--- 

## Part 3: The Transformer Architecture

The Transformer model, introduced in the paper "Attention Is All You Need," is built entirely from attention mechanisms.

A Transformer block consists of two main sub-layers:

1.  **Multi-Head Self-Attention:**
    *   **Self-Attention:** This is the case where the Query, Key, and Value vectors all come from the *same* input sequence. Each word in the sentence attends to every other word in the sentence (including itself) to build a contextual representation.
    *   **Multi-Head:** Instead of performing attention once, the model does it multiple times in parallel. The input Q, K, and V are first passed through separate linear layers to project them into different "representation subspaces." Attention is computed independently in each of these "heads." The results are then concatenated and passed through a final linear layer. This allows the model to jointly attend to information from different subspaces at different positions.

2.  **Position-wise Feed-Forward Network:**
    *   This is a simple two-layer MLP that is applied independently to each position (each word vector) in the sequence.

Each of these sub-layers has a residual connection around it, followed by a layer normalization. This is crucial for enabling deep stacks of Transformer blocks.

### 3.1. Using PyTorch's `nn.TransformerEncoderLayer`

Building this from scratch is a great exercise, but in practice, we use PyTorch's built-in, optimized modules.

```python
print("\n--- Part 3: The Transformer Encoder Layer ---")

# --- Parameters for a Transformer block ---
embed_dim = 512 # d_model
num_heads = 8   # Number of parallel attention heads
ff_dim = 2048   # Hidden dimension of the feed-forward network
batch_size = 32
seq_len = 100

# --- Create the Transformer Encoder Layer ---
# batch_first=True is crucial!
transformer_block = nn.TransformerEncoderLayer(
    d_model=embed_dim, 
    nhead=num_heads, 
    dim_feedforward=ff_dim, 
    batch_first=True
)

# --- Create dummy input data ---
# This would be the output of an embedding layer.
input_sequence = torch.randn(batch_size, seq_len, embed_dim)

# --- Pass the sequence through the block ---
output_sequence = transformer_block(input_sequence)

print(f"Input sequence shape: {input_sequence.shape}")
print(f"Output sequence shape: {output_sequence.shape}") # Shape is preserved

# --- Stacking multiple blocks ---
# A full Transformer Encoder is just a stack of these layers.
num_layers = 6
full_transformer_encoder = nn.TransformerEncoder(
    transformer_block, 
    num_layers=num_layers
)

final_output = full_transformer_encoder(input_sequence)
print(f"\nShape after passing through {num_layers} layers: {final_output.shape}")
```

## Conclusion: A New Foundation for Sequence Modeling

The Attention mechanism and the Transformer architecture have fundamentally changed the landscape of deep learning, particularly in NLP. By abandoning recurrence in favor of parallelizable self-attention, Transformers can process much longer sequences more effectively and have become the foundation for nearly all state-of-the-art language models, including BERT and GPT.

**Key Takeaways:**

1.  **Attention as Weighted Sum:** The core idea of attention is to compute a context vector as a weighted sum of value vectors, where the weights are dynamically computed based on query-key similarity.
2.  **QKV Model:** The Query, Key, and Value abstraction is the language used to describe attention. In self-attention, Q, K, and V are all derived from the same input sequence.
3.  **Multi-Head Attention:** Performing attention in multiple "heads" in parallel allows the model to capture different types of relationships from different representation subspaces.
4.  **Transformers are Non-Sequential:** Unlike RNNs, a Transformer processes the entire sequence at once. It has no inherent sense of word order, which is why **positional encodings** (which we saw in the Vision Transformer guide) are essential.
5.  **The Power of `torch.nn`:** PyTorch provides highly optimized, built-in modules like `nn.TransformerEncoderLayer` that make it easy to build state-of-the-art models.

With this understanding of attention and Transformers, you are now equipped to tackle the most advanced and powerful models in modern NLP.

## Self-Assessment Questions

1.  **RNN Bottleneck:** What are the two main limitations of the sequential nature of RNNs that attention helps to solve?
2.  **Q, K, V:** In the attention mechanism, what is the role of the Query? What about the Key and the Value?
3.  **Self-Attention:** What does it mean when we say a model is performing "self-attention"?
4.  **Multi-Head Attention:** What is the motivation for using Multi-Head Attention instead of just a single attention calculation?
5.  **Positional Information:** If you pass the same set of words but in a different order into a Transformer Encoder, will the output for a specific word be the same? Why is this a problem, and what is the solution?
