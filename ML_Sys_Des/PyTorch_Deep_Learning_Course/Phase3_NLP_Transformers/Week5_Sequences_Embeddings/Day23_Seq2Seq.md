# Day 23: Seq2Seq & Attention - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Encoder-Decoder, Alignment, and Bahdanau Attention

## 1. Theoretical Foundation: The Information Bottleneck

**Seq2Seq (Sutskever, 2014)**:
*   **Encoder**: RNN reads input $X$, produces final hidden state $h_T$ (Context Vector).
*   **Decoder**: RNN takes $h_T$, generates output $Y$.

**The Problem**:
*   $h_T$ is a fixed-size vector (e.g., 256 floats).
*   It must compress the *entire* meaning of a long sentence.
*   Information is lost (Bottleneck). Performance drops for long sequences.

## 2. The Attention Mechanism (Bahdanau, 2015)

Instead of relying on just $h_T$, allow the Decoder to look at **all** encoder states $h_1, ..., h_T$ at every step.

### How it works (Step $t$ of Decoder):
1.  **Score**: Calculate similarity between Decoder state $s_{t-1}$ and all Encoder states $h_j$.
    $$ e_{tj} = \text{score}(s_{t-1}, h_j) $$
2.  **Attention Weights**: Softmax the scores.
    $$ \alpha_{tj} = \frac{\exp(e_{tj})}{\sum \exp(e_{tk})} $$
3.  **Context Vector**: Weighted sum of encoder states.
    $$ c_t = \sum \alpha_{tj} h_j $$
4.  **Predict**: Use $c_t$ and $s_{t-1}$ to predict next word.

## 3. Implementation: Attention in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        # Linear layer to transform states before scoring
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [Batch, Dec Dim] (s_{t-1})
        # encoder_outputs: [Batch, Seq Len, Enc Dim * 2] (h_1...h_T)
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate Energy (Score)
        # Tanh(W [s; h])
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Calculate Attention (v^T * Energy)
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)
```

## 4. Types of Attention Scores

1.  **Additive (Bahdanau)**: $v^T \tanh(W_1 s + W_2 h)$.
2.  **Multiplicative (Luong)**: $s^T W h$.
3.  **Dot-Product**: $s^T h$ (Used in Transformers).
