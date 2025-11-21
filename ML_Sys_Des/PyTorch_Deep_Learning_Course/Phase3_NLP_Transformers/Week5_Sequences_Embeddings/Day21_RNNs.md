# Day 21: RNNs, LSTMs, and GRUs - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Recurrent Neural Networks, Vanishing Gradients, and Gating

## 1. Theoretical Foundation: Sequence Modeling

Feed-Forward Networks (MLPs/CNNs) assume inputs are independent.
**Sequences** (Text, Audio, Time-Series) have temporal dependencies.
$P(w_t | w_{t-1}, w_{t-2}, ...)$

### The Recurrent Neuron
Maintains a **Hidden State** $h_t$ that acts as memory.
$$ h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b) $$
$$ y_t = W_{hy} h_t + b_y $$
*   **Shared Weights**: The same $W_{xh}, W_{hh}$ are used at every time step.
*   **BPTT (Backpropagation Through Time)**: Unrolling the network over time to compute gradients.

## 2. The Vanishing Gradient Problem

In BPTT, gradients are multiplied by $W_{hh}$ at each step.
*   If eigenvalue of $W_{hh} < 1$: Gradients vanish exponentially (Short-term memory only).
*   If eigenvalue of $W_{hh} > 1$: Gradients explode (NaNs).

## 3. LSTM (Long Short-Term Memory)

Introduced by Hochreiter & Schmidhuber (1997).
Key Idea: **Cell State** $C_t$ (The "Highway" for gradients) + **Gates**.

### The Gates (Sigmoid $\sigma \in [0, 1]$)
1.  **Forget Gate ($f_t$)**: What to throw away from old memory?
2.  **Input Gate ($i_t$)**: What new info to store?
3.  **Output Gate ($o_t$)**: What to output to $h_t$?

$$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$
*   If $f_t=1$, gradients flow unchanged. Solves vanishing gradient!

## 4. GRU (Gated Recurrent Unit)

Simplified LSTM. Merges Cell/Hidden state.
1.  **Reset Gate ($r_t$)**: Ignore previous state?
2.  **Update Gate ($z_t$)**: Trade-off between old state and new candidate.

## 5. Implementation: LSTM Classifier

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # batch_first=True -> (Batch, Seq, Feat)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                            batch_first=True, dropout=0.5, bidirectional=True)
        
        # Bidirectional doubles the hidden dimension
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        # text: [Batch, Seq Len]
        embedded = self.embedding(text)
        
        # output: [Batch, Seq, Hidden*2] (All states)
        # hidden: (h_n, c_n) Last states
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concat forward and backward hidden states from last layer
        # hidden shape: [Num Layers * Num Dir, Batch, Hidden]
        hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        return self.fc(hidden_last)
```

## 6. Packing Sequences

Sequences in a batch have different lengths (padded with 0).
LSTM shouldn't process padding.
Use `pack_padded_sequence`.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 1. Sort batch by length (descending)
# 2. Pack
packed_input = pack_padded_sequence(embedded, lengths, batch_first=True)
packed_output, (hidden, cell) = lstm(packed_input)
# 3. Unpack (if needed)
output, _ = pad_packed_sequence(packed_output, batch_first=True)
```
