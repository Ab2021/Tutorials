# Day 15.2: Attention Mechanisms Deep Dive - A Practical Guide

## Introduction: The Power of Focus

Attention is arguably the most important and impactful idea in deep learning in the last decade. It allows a model to dynamically focus on the most relevant parts of its input when performing a task, rather than relying on a single, static context vector. This simple but powerful idea not only solved the bottleneck problem in Seq2Seq models but also became the foundation of the entire Transformer architecture.

This guide will provide a deep, practical dive into the mechanics of the **Scaled Dot-Product Attention** mechanism, which is the core of every Transformer. We will build it from scratch, visualize the attention weights, and then see how it is extended to **Multi-Head Attention**.

**Today's Learning Objectives:**

1.  **Master the Query, Key, Value (QKV) Abstraction:** Develop a strong intuition for what the Query, Key, and Value vectors represent.
2.  **Implement Scaled Dot-Product Attention from Scratch:** Write the code to perform the six steps of the attention calculation: projection, dot product, scaling, masking (optional), softmax, and weighted sum.
3.  **Visualize Attention Weights:** Create a heatmap of the attention matrix to see what a model is "paying attention to."
4.  **Understand Multi-Head Attention:** Grasp the motivation for performing attention in multiple "heads" in parallel and see how it's implemented.

---

## Part 1: The QKV Model - Attention as a Database Query

It's helpful to think of the attention mechanism as a fast, differentiable database retrieval system.

*   **Value (V):** This is the database itself. It's a set of vectors containing the information we want to retrieve. For self-attention, this is the set of input token embeddings.

*   **Key (K):** These are like the labels or indices for the database. Each Value vector has a corresponding Key vector. The Keys are what we search through.

*   **Query (Q):** This is the question we are asking the database. It's a vector representing what information we are currently looking for.

**The Process:**
1.  We compare our **Query** to every **Key** in the database to get a **similarity score**.
2.  We convert these scores into probabilities (**attention weights**) using a softmax.
3.  We use these weights to compute a weighted sum of all the **Values** in the database.

The result is a **context vector** that is a custom-blended mix of all the Values, specifically tailored to our Query.

---

## Part 2: Implementing Scaled Dot-Product Attention

Let's build a full attention layer from scratch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("--- Part 2: Implementing Scaled Dot-Product Attention ---")

class Attention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        """
        Args:
            embed_dim (int): The embedding dimension of the input.
            head_dim (int): The dimension of the Q, K, V vectors for this head.
        """
        super().__init__()
        self.head_dim = head_dim
        
        # --- Step 1: Create the projection layers ---
        # Linear layers to project the input into Q, K, V spaces.
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, embed_dim)
        
        # --- Step 2: Project input into Q, K, V ---
        q = self.q_proj(x) # (batch, seq_len, head_dim)
        k = self.k_proj(x) # (batch, seq_len, head_dim)
        v = self.v_proj(x) # (batch, seq_len, head_dim)
        
        # --- Step 3: Compute dot-product similarity scores ---
        # (batch, seq_len, head_dim) @ (batch, head_dim, seq_len) -> (batch, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # --- Step 4: Scale the scores ---
        # This prevents the dot products from growing too large.
        scaled_scores = scores / math.sqrt(self.head_dim)
        
        # --- Step 5: Apply Mask (Optional) ---
        # The mask is used in the decoder to prevent attending to future tokens.
        # It's a boolean tensor where `True` values will be ignored.
        if mask is not None:
            # We fill the masked positions with a very small number (-inf)
            # so that they become zero after the softmax.
            scaled_scores = scaled_scores.masked_fill(mask == 1, -1e9)
            
        # --- Step 6: Apply Softmax ---
        # This converts the scores into a probability distribution (attention weights).
        # The softmax is applied on the last dimension (the keys).
        attention_weights = F.softmax(scaled_scores, dim=-1)
        
        # --- Step 7: Compute the weighted sum of Values ---
        # (batch, seq_len, seq_len) @ (batch, seq_len, head_dim) -> (batch, seq_len, head_dim)
        context_vector = torch.matmul(attention_weights, v)
        
        return context_vector, attention_weights

# --- Usage Example ---
embed_dim = 256
head_dim = 64
seq_len = 10
batch_size = 4

# Create dummy input
input_seq = torch.randn(batch_size, seq_len, embed_dim)

# Create the attention layer
attention_layer = Attention(embed_dim, head_dim)

# Get the output
context, weights = attention_layer(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output context vector shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
```

### 2.1. Visualizing the Attention Weights

Let's create a heatmap to see what a hypothetical attention matrix might look like.

```python
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Visualizing Attention Weights ---")

# Let's look at the weights for the first sample in the batch
attention_matrix = weights[0].detach().cpu().numpy()

# Let's create some dummy tokens for labeling the axes
tokens = [f'token_{i+1}' for i in range(seq_len)]

plt.figure(figsize=(8, 6))
sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
plt.title('Self-Attention Weights Heatmap')
plt.xlabel('Keys (Words being attended to)')
plt.ylabel('Queries (Words doing the attending)')
plt.show()
```

**Interpretation:** Each row `i` shows the attention distribution for the query from token `i`. A bright spot at `(row_i, col_j)` means that when the model was processing token `i`, it paid high attention to token `j`.

---

## Part 3: Multi-Head Attention

**The Idea:** It can be beneficial for the model to attend to different parts of the sequence for different reasons. For example, one attention "head" might learn to focus on syntactic relationships, while another focuses on semantic relationships.

**Multi-Head Attention** formalizes this by running the Scaled Dot-Product Attention mechanism multiple times in parallel.

**How it Works:**
1.  The input `x` is passed through `h` different sets of linear projection layers to create `h` different sets of `(Q, K, V)` triplets.
2.  Scaled Dot-Product Attention is computed for each head in parallel, yielding `h` different context vectors.
3.  These `h` context vectors are **concatenated** together.
4.  The concatenated vector is passed through a final linear layer (`W_o`) to produce the final output.

This allows the model to jointly attend to information from different representation subspaces at different positions.

### 3.1. Implementing Multi-Head Attention

```python
print("\n--- Part 3: Multi-Head Attention ---")

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # We can use a single large linear layer for all Q, K, V projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        # A final linear layer to combine the heads
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Project to Q, K, V for all heads at once
        qkv = self.qkv_proj(x) # (batch, seq_len, 3 * embed_dim)
        
        # 2. Reshape and split into multiple heads
        # (batch, seq_len, 3 * embed_dim) -> (batch, seq_len, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # (batch, seq_len, 3, num_heads, head_dim) -> (3, batch, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3. Compute attention for all heads
        # The scaled_dot_product_attention function works with the extra head dimension.
        context, attention_weights = scaled_dot_product_attention(q, k, v)
        
        # 4. Concatenate the heads and project
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        context = context.transpose(1, 2)
        # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, embed_dim)
        context = context.reshape(batch_size, seq_len, self.embed_dim)
        
        # 5. Pass through final linear layer
        output = self.o_proj(context)
        
        return output, attention_weights

# --- Usage Example ---
multi_head_attn = MultiHeadAttention(embed_dim=256, num_heads=8)
input_seq = torch.randn(batch_size, seq_len, 256)

output, multi_head_weights = multi_head_attn(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Final output shape: {output.shape}")
print(f"Attention weights shape: {multi_head_weights.shape}") # (batch, num_heads, seq_len, seq_len)
```

## Conclusion

The attention mechanism is a powerful and flexible tool that has become a fundamental building block in modern deep learning. By learning to dynamically weigh the importance of different parts of the input, attention-based models can capture complex dependencies and relationships that are difficult for recurrent models to handle.

**Key Takeaways:**

1.  **Attention as a QKV Database:** The Query-Key-Value model provides a powerful abstraction for understanding attention.
2.  **Scaled Dot-Product Attention:** The core mechanism involves computing dot products between queries and keys to get scores, scaling, applying softmax to get weights, and computing a weighted sum of values.
3.  **Multi-Head Attention is Key:** This is the version used in Transformers. It allows the model to focus on different types of information from different representation subspaces in parallel, which is more powerful than a single attention mechanism.
4.  **Visualization is Insight:** Plotting the attention weight matrix as a heatmap is a great way to interpret and debug your model by seeing where it is "looking."

This deep understanding of the attention mechanism is the key that unlocks the inner workings of the Transformer architecture and all the state-of-the-art models that are built upon it.

## Self-Assessment Questions

1.  **QKV:** In a self-attention layer, where do the Query, Key, and Value vectors come from?
2.  **Scaling:** In Scaled Dot-Product Attention, why are the scores divided by the square root of the head dimension?
3.  **Masking:** In the context of a Transformer decoder, what is the purpose of the attention mask?
4.  **Multi-Head Attention:** What is the main motivation for using multiple attention heads instead of just one?
5.  **Output Shape:** If you have an input of shape `(32, 50, 512)` and you pass it through a Multi-Head Attention layer with `embed_dim=512` and `num_heads=8`, what will be the shape of the final output tensor?

