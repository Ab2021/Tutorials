# Day 5: Neural Architectures for NLP
## Core Concepts & Theory

### The Evolution of Sequential Models

Before Transformers dominated (2017+), the field experimented with various architectures for sequential data:

**Timeline:**
- 2010-2014: Recurrent Neural Networks (RNNs)
- 2014-2016: LSTMs and GRUs
- 2014-2017: Attention mechanisms + recurrence
- 2017+: Transformers (attention-only)

Understanding these predecessors reveals why Transformers won.

### Recurrent Neural Networks (RNNs)

**Key Idea:** Process sequence one step at a time, maintaining hidden state.

**Architecture:**

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Input to hidden
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        # Hidden to hidden (recurrence)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.W_hh.in_features, device=x.device)
        
        hidden_states = []
        for t in range(seq_len):
            # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
            h = torch.tanh(
                self.W_xh(x[:, t, :]) + self.W_hh(h) + self.bias
            )
            hidden_states.append(h)
        
        # Stack hidden states
        output = torch.stack(hidden_states, dim=1)  # (batch, seq_len, hidden_size)
        return output, h  # output, final_hidden
```

**Recurrence Formula:**

```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
```

**Key Properties:**

1. **Parameter Sharing:** Same W_xh, W_hh used at all time steps
2. **Variable Length:** Can process any sequence length
3. **Sequential Dependency:** h_t depends on h_{t-1} (inherent order)

**Problems:**

**1. Vanishing Gradients:**

Gradient flows backward through time via chain rule:

```
∂L/∂h_0 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂h_0
```

Each ∂h_t/∂h_{t-1} involves W_hh and tanh derivative (≤ 1).

For long sequences (T=100):
```
∂h_t/∂h_{t-1} = tanh'(·) × W_hh
```

If ||W_hh|| < 1 and |tanh'| < 1:
```
(W_hh)^T → 0 exponentially
```

**Result:** Gradients vanish, early time steps don't learn!

**2. Exploding Gradients:**

If ||W_hh|| > 1:
```
(W_hh)^T → ∞ exponentially
```

Gradients explode, training unstable.

**Solution:** Gradient clipping, better architectures (LSTM).

### Long Short-Term Memory (LSTM)

**Motivation:** Explicitly model long-term dependencies via gating.

**Architecture:**

LSTM has **four gates** controlling information flow:

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Combined weight matrix for all gates
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)
    
    def forward(self, x, hidden):
        h, c = hidden  # h: hidden state, c: cell state
        
        # Compute all gates in one matrix multiply (efficient!)
        gates = self.weight_ih(x) + self.weight_hh(h)
        
        # Split into four gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        #Gates
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate (candidate values)
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c_new = f * c + i * g
        
        # Update hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
```

**LSTM Equations:**

```
i_t = σ(W_xi * x_t + W_hi * h_{t-1} + b_i)  # Input gate
f_t = σ(W_xf * x_t + W_hf * h_{t-1} + b_f)  # Forget gate
g_t = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g)  # Cell candidate
o_t = σ(W_xo * x_t + W_ho * h_{t-1} + b_o)  # Output gate

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  # Update cell state
h_t = o_t ⊙ tanh(c_t)  # Update hidden state
```

**Key Innovation: Cell State c_t**

- Runs parallel to hidden state
- Has additive updates (c_t = f·c_{t-1} + i·g) instead of multiplicative
- Gradient flows through addition (no vanishing!)

**Why LSTMs Solve Vanishing Gradients:**

Gradient w.r.t. cell state:

```
∂c_t/∂c_{t-1} = f_t
```

Unlike RNN (∂h_t/∂h_{t-1} involves matrix multiply), this is element-wise!

If forget gate f_t ≈ 1 (preserve memory):
```
∂c_T/∂c_0 = f_T × f_{T-1} × ... × f_1 ≈ 1
```

Gradient preserved across long sequences!

**Gate Intuitions:**

- **Forget gate (f):** What to forget from cell state (f=0 → forget all, f=1 → keep all)
- **Input gate (i):** How much to add new candidate (i=0 → ignore new, i=1 → add fully)
- **Output gate (o):** How much cell state to expose (o=0 → hide, o=1 → expose)

**Example:**

```python
# Processing: "The cat sat on the mat"

# At "cat":
# - Forget gate: f ≈ 0.2 (forget previous subject)
# - Input gate: i ≈ 0.9 (store "cat" as new subject)
# - Cell state: stores "cat is current subject"

# At "sat":
# - Forget gate: f ≈ 0.9 (keep subject "cat")
# - Input gate: i ≈ 0.7 (add verb "sat")

# At "mat":
# - Forget gate: f ≈ 0.9 (still need subject for agreement)
# - Output gate: o ≈ 0.8 (expose for next prediction)
```

### Gated Recurrent Unit (GRU)

**Motivation:** Simpler than LSTM, similar performance.

**Architecture:**

Only **two gates** (vs LSTM's four):

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size)
    
    def forward(self, x, h):
        # Compute reset and update gates
        gi = self.weight_ih(x)
        gh = self.weight_hh(h)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        
        r = torch.sigmoid(i_r + h_r)  # Reset gate
        z = torch.sigmoid(i_z + h_z)  # Update gate
        n = torch.tanh(i_n + r * h_n)  # New gate (candidate)
        
        # Update hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new
```

**GRU Equations:**

```
r_t = σ(W_xr * x_t + W_hr * h_{t-1} + b_r)  # Reset gate
z_t = σ(W_xz * x_t + W_hz * h_{t-1} + b_z)  # Update gate
ñ_t = tanh(W_xn * x_t + W_hn * (r_t ⊙ h_{t-1}) + b_n)  # Candidate

h_t = (1 - z_t) ⊙ ñ_t + z_t ⊙ h_{t-1}  # Update hidden
```

**Key Differences from LSTM:**

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 4 (i, f, g, o) | 2 (r, z) |
| States | 2 (h, c) | 1 (h) |
| Parameters | More | Fewer |
| Performance | Slightly better | Slightly worse |
| Speed | Slower | Faster |

**When to use:**
- GRU: Smaller datasets, faster training, simpler model
- LSTM: Larger datasets, need maximum capacity

### Bidirectional RNNs

**Problem:** Forward RNN only sees past context.

Example: "The bank is closed" - to predict "closed", helpful to see "bank" (before) AND sentence end (after).

**Solution:** Process sequence in both directions.

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # Forward pass
        forward_out, _ = self.forward_lstm(x)
        
        # Backward pass (flip sequence)
        x_reversed = x.flip(dims=[1])
        backward_out, _ = self.backward_lstm(x_reversed)
        backward_out = backward_out.flip(dims=[1])  # Un-flip
        
        # Concatenate
        output = torch.cat([forward_out, backward_out], dim=-1)
        return output  # (batch, seq_len, 2*hidden_size)
```

**Advantages:**
- Richer representations (sees both directions)
- Better for tasks where future context matters (NER, POS tagging)

**Disadvantages:**
- 2× parameters and compute
- Can't use for autoregressive tasks (language modeling) - would see future!

### CNNs for Text

**Motivation:** Can we use Convolutional Neural Networks (successful in vision) for text?

**1D Convolution for Text:**

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Multiple filter sizes (e.g., [2, 3, 4] for bigrams, trigrams, 4-grams)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, text):
        # text: (batch, seq_len)
        embedded = self.embedding(text)  # (batch, seq_len, embed_dim)
        
        # Transpose for conv1d (expects channels first)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        # Apply convolutions
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # Each: (batch, num_filters, seq_len - kernel_size + 1)
        
        # Max pooling over time
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # Each: (batch, num_filters)
        
        # Concatenate
        cat = torch.cat(pooled, dim=1)  # (batch, num_filters * len(filter_sizes))
        
        # Final classification
        return self.fc(cat)
```

**How It Works:**

1. **Embedding**: Convert tokens to vectors
2. **Convolution**: Slide filters to detect n-gram patterns
   - Filter size 3 → detects trigrams
   - Each filter learns different pattern ("not good", "very bad", etc.)
3. **Max Pooling**: Extract most important feature from each filter
4. **Classification**: Use pooled features for prediction

**Advantages:**
- Fast (parallelizable, no recurrence)
- Good for classification (captures local patterns)
- Fewer parameters than RNN

**Disadvantages:**
- Fixed receptive field (filter size limits context)
- No long-range dependencies (unlike RNN/Transformer)
- Loses word order beyond filter size

**Use Cases:**
- Sentiment analysis
- Text classification
- Fast feature extraction

### Attention Mechanism (Pre-Transformer)

**Problem with Seq2Seq (Encoder-Decoder):**

Encoder compresses entire input into single vector (bottleneck!).

```
Input: "The cat sat on the mat"
Encoder → single vector h → Decoder
```

For long sequences, information loss!

**Solution; Attention**

Let decoder look at ALL encoder states, not just last one.

```python
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_size) - current decoder state
        # encoder_outputs: (batch, seq_len, hidden_size)
        
        seq_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden for each encoder output
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # (batch, seq_len, hidden_size)
        
        # Concatenate and compute energy
        energy = torch.tanh(self.attn(torch.cat([hidden_repeated, encoder_outputs], dim=2)))
        # (batch, seq_len, hidden_size)
        
        # Compute attention scores
        attention_scores = self.v(energy).squeeze(2)
        # (batch, seq_len)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # (batch, 1, hidden_size)
        
        return context.squeeze(1), attention_weights
```

**Intuition:**

At each decoding step, model "attends" to relevant input tokens.

Example (translation):
```
English: "The cat sat on the mat"
French: "Le chat [attend here]..."

When generating "s'est assis" (sat), attention focuses on "sat" in English.
```

**Why It Works:**

- No information bottleneck
- Model learns what to focus on
- Interpretable (can visualize attention weights)

### Summary: Pre-Transformer Landscape

**Sequence Models Comparison:**

| Architecture | Pros | Cons | Best For |
|--------------|------|------|----------|
| RNN | Simple, variable length | Vanishing gradients, slow | Short sequences |
| LSTM | Long dependencies, proven | Complex, slow | General sequences |
| GRU | Faster than LSTM, simpler | Slightly less capable | Resource-constrained |
| Bi-LSTM | Rich representations | 2× cost, no autoregressive | Classification, tagging |
| CNN | Fast, parallelizable | Limited context | Classification |
| Attention+RNN | No bottleneck | Still sequential (RNN) | Seq2seq tasks |

**The Transformer Revolution (2017):**

Transformers combine best of all:
- Parallelizable (like CNN)
- Long-range dependencies (like LSTM with attention)
- No vanishing gradients
- Fully attentional (no recurrence)

→ Became dominant architecture for NLP!

### Why Transformers Won

**Problems with RNNs/LSTMs:**
1. Sequential (can't parallelize)
2. Vanishing gradients (even with LSTM, long sequences hard)
3. Limited context (hard to relate token 1 to token 1000)

**Transformers Solution:**
1. Full parallelization (process all tokens simultaneously)
2. Direct connections (attention between any two tokens)
3. No gradient vanishing (residual connections, layer norm)

Modern LLMs (GPT, BERT, T5, LLaMA) are all Transformer-based!

Next (Day 8): We dive deep into Transformer architecture and self-attention.
