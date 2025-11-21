# Day 26: GPT & Decoder Models - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: GPT, LLaMA, and Causal Language Modeling

## 1. Theoretical Foundation: Generative Pre-trained Transformer

BERT is an Encoder (Understanding). GPT is a Decoder (Generation).
**Objective**: Causal Language Modeling (CLM).
$$ P(w_1, ..., w_T) = \prod_{t=1}^T P(w_t | w_1, ..., w_{t-1}) $$
*   Predict next token given history.
*   **Unidirectional**: Can only attend to left context.

## 2. Architecture Differences (vs BERT)

1.  **Masked Self-Attention**: The attention matrix is masked (Upper Triangular = $-\infty$) to prevent peeking at future tokens.
2.  **No Encoder**: It is a stack of Decoder layers (without Cross-Attention).
3.  **LayerNorm Position**: Pre-Norm (GPT-2/3) vs Post-Norm (BERT).

## 3. The Evolution

*   **GPT-1**: Pre-training on BooksCorpus. Fine-tuning on tasks.
*   **GPT-2**: Zero-Shot Task Transfer. "Language Models are Unsupervised Multitask Learners".
*   **GPT-3**: In-Context Learning (Few-Shot). 175B parameters.
*   **LLaMA**: Open-source, efficient, trained on more tokens (Chinchilla Optimal). Uses RoPE and SwiGLU.

## 4. Implementation: Minimal GPT in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads, max_len):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        
        self.c_attn = nn.Linear(d_model, 3 * d_model) # Q, K, V merged
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Causal Mask buffer
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channel
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        # Reshape for Multi-head
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

## 5. Generation: The Loop

```python
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # Crop context if too long
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] # Last time step
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```
