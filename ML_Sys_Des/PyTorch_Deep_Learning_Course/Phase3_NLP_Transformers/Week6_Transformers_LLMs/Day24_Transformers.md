# Day 24: The Transformer - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Self-Attention, Multi-Head Attention, and The Architecture

## 1. Theoretical Foundation: Attention is All You Need (2017)

RNNs are slow (Sequential).
Transformers rely entirely on **Self-Attention** to compute representations.
Parallelizable. Global Receptive Field ($O(1)$ path length).

### Scaled Dot-Product Attention
Inputs: Query ($Q$), Key ($K$), Value ($V$).
$$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) V $$
*   **Query**: What I am looking for.
*   **Key**: What I contain.
*   **Value**: What I will pass on.
*   $\sqrt{d_k}$: Scaling factor to prevent vanishing gradients in Softmax.

## 2. Multi-Head Attention

Instead of one attention head, we have $h$ heads.
Each head projects $Q, K, V$ into different subspaces.
Allows the model to focus on different things (e.g., Head 1: Syntax, Head 2: Semantics).
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h) W^O $$

## 3. The Architecture

1.  **Encoder**: Stack of layers. Each has Self-Attention + Feed Forward.
2.  **Decoder**: Stack of layers. Masked Self-Attention + Cross-Attention + Feed Forward.
3.  **Residual Connections & LayerNorm**: Add & Norm after every sub-layer.
4.  **Positional Encoding**: Injected at input.

## 4. Implementation: Self-Attention in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        
        assert (self.head_dim * heads == d_model), "Embed size needs to be div by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, d_model)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # MatMul Q * K^T
        # einsum is powerful: nqhd, nkhd -> nhqk
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.d_model ** (1/2)), dim=3)
        
        # MatMul Attn * V
        # nhqk, nvhd -> nqhd
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        return self.fc_out(out)
```

## 5. Masking

*   **Padding Mask**: Ignore `<PAD>` tokens.
*   **Look-Ahead Mask (Causal)**: In Decoder, position $t$ cannot see $t+1$. Upper triangular matrix set to $-\infty$.
