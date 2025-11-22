# Day 5: Neural Architectures for NLP
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain why LSTMs solve the vanishing gradient problem. What's the key difference from vanilla RNNs?

**Answer:**

**Vanilla RNN Gradient:**

```
∂h_t/∂h_{t-1} = tanh'(W_hh h_{t-1} + W_xh x_t) × W_hh
```

Through T steps:
```
∂h_T/∂h_0 = (∂h_T/∂h_{T-1}) × ... × (∂h_1/∂h_0)
           = (tanh' × W_hh)^T
```

- If ||W_hh|| < 1 and |tanh'| ≤ 1: Product vanishes
- If ||W_hh|| > 1: Product explodes

**LSTM Solution: Additive Cell State**

LSTM has two paths:
1. Hidden state h_t (like RNN)
2. **Cell state c_t** (new!)

Key equation:
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
     └─────────┘   └────────┘
     Forget old    Add new

∂c_t/∂c_{t-1} = f_t  # Element-wise multiplication, no matrix!
```

**Gradient Flow:**

```
∂c_T/∂c_0 = f_T ⊙ f_{T-1} ⊙ ... ⊙ f_1
```

- No repeated matrix multiplication!
- If f_t ≈ 1: Gradient ≈ 1 (preserved)
- If f_t ≈ 0: Gradient ≈ 0 (intentionally forgotten)

**Key Difference:**

| RNN | LSTM |
|-----|------|
| Multiplicative updates (matrix^T) | Additive updates (f·c + i·g) |
| Gradient vanishes through layers | Gradient flows via forget gates |
| No control over memory | Explicit control (gates) |

**Interview Follow-up:**
*Q: Can LSTMs still have vanishing gradients?*

**A:** Yes! If forget gates learn to be close to 0 for long sequences. But unlike RNNs, this is a **learned choice**, not a structural limitation. Model can choose to preserve gradients when needed.

---

#### Q2: You're building a sentiment classifier. Compare RNN, LSTM, CNN, and Transformer approaches. Which would you choose and why?

**Answer:**

**Task: Sentiment Classification**
- Input: Movie review (50-500 words)
- Output: Positive/Negative/Neutral

**Approach Comparison:**

**1. RNN/LSTM:**

```python
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)  # 3 classes
    
    def forward(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.lstm(embed)  # Use final hidden state
        return self.fc(h_n.squeeze(0))
```

**Pros:**
- Captures sequential dependencies
- Proven architecture
- Interpretable (can analyze hidden states)

**Cons:**
- Sequential (slow, O(L) time)
- May struggle with very long reviews (500+ tokens)
- Harder to train on GPUs (limited parallelism)

**2. CNN:**

```python
class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Multiple filter sizes for different n-grams
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 100, kernel_size=k)
            for k in [3, 4, 5]  # Trigrams, 4-grams, 5-grams
        ])
        self.fc = nn.Linear(300, 3)
    
    def forward(self, x):
        embed = self.embedding(x).permute(0, 2, 1)
        conved = [F.relu(conv(embed)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)
```

**Pros:**
- Fast (fully parallelizable)
- Good at capturing local patterns ("not bad", "very good")
- Fewer parameters than LSTM
- Easy to train

**Cons:**
- Limited receptive field (misses long-range dependencies)
- Less intuitive for sequential data

**3. Transformer (BERT fine-tuning):**

```python
from transformers import AutoModel

class SentimentTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 3)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_embed = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_embed)
```

**Pros:**
- State-of-the-art accuracy (transfer learning from pre-training)
- Handles long-range dependencies
- Bidirectional context

**Cons:**
- Overkill for simple sentiment
- Slow inference (110M parameters for BERT-base)
- Requires more data to fine-tune effectively



**My Choice: CNN for Production**

**Reasoning:**

1. **Task Characteristics:**
   - Sentiment is often local ("great movie", "terrible acting")
   - Don't need long-range dependencies for this task
   - Speed matters in production

2. **Performance vs Cost:**
   ```
   Accuracy:
   - CNN: ~87-89%
   - LSTM: ~88-90%
   - BERT: ~91-93%
   
   Inference Time (batch=32):
   - CNN: ~5ms
   - LSTM: ~20ms
   - BERT: ~100ms
   
   Model Size:
   - CNN: ~10MB
   - LSTM: ~20MB
   - BERT: ~440MB
   ```

3. **Trade-off:**
   - CNN gives 95% of BERT's accuracy at 5% of the cost!
   - For sentiment (not complex reasoning), good enough

**When I'd Choose Others:**

- **LSTM**: If order truly matters (narrative-based sentiment, sarcasm detection)
- **BERT**: If maximum accuracy required, cost not a concern (financial analysis, legal documents)

---

#### Q3: Your LSTM model trains fine but struggles at inference with sequences longer than training examples. Why, and how would you fix it?

**Answer:**

**Problem: Length Generalization Failure**

**Root Cause:**

LSTMs learn position-dependent patterns during training.

Example:
```
Training: Sequences of length 50-100 tokens
Inference: Sequence of length  500 tokens

At position 150:
- LSTM has never seen this position during training!
- Hidden state may drift
- Positional biases learned during training don't transfer
```

**Specific Issues:**

**1. Gradient Saturation:**

Even with LSTM, very long sequences can saturate:
```python
# After 500 steps, forget gates may drift toward 0 or 1
# Information loss accumulates
```

**2. Positional Bias:**

LSTM may learn:
- "Important information is usually in first 50 tokens"
- "Conclusion is in last 20 tokens"

These patterns break for longer sequences!

**3. Numerical Instability:**

```python
c_500 = f_500 * f_499 * ... * f_1 * c_0 + (accumulated updates)
```

Floating point errors accumulate!

**Solutions:**

**1. Train on Variable Lengths:**

```python
# During training, use diverse lengths
train_lengths = [50, 100, 200, 300, 400]

for batch in dataloader:
    seq_len = random.choice(train_lengths)
    # Truncate or pad to seq_len
    ...
```

Exposes model to various positions.

**2. Truncated Backpropagation Through Time (TBPTT):**

```python
# Split long sequence into chunks
chunk_size = 100
hidden = None

for i in range(0, len(sequence), chunk_size):
    chunk = sequence[i:i+chunk_size]
    output, hidden = lstm(chunk, hidden)
    
    # Detach hidden state (break gradient flow)
    hidden = (hidden[0].detach(), hidden[1].detach())
    
    loss = criterion(output, target[i:i+chunk_size])
    loss.backward()
```

Treats 500-token sequence as 5 independent 100-token sequences (with state carrying over).

**3. Segment-Level Recurrence:**

Process in overlapping segments:

```python
segment_size = 100
overlap = 20

segments = create_overlapping_segments(sequence, segment_size, overlap)

for segment in segments:
    output = lstm(segment)
    # Aggregate outputs (e.g., average)
```

**4. Switch to Transformer:**

Transformers handle arbitrary lengths better (up to context window):

```python
# BERT can handle up to 512 tokens
# Longformer, BigBird: up to 4096+ tokens
# Direct attention connections (no accumulated error)
```

**Production Fix:**

```python
class RobustLSTM:
    def __init__(self, model, max_training_len=100):
        self.model = model
        self.max_len = max_training_len
    
    def predict(self, sequence):
        if len(sequence) <= self.max_len:
            return self.model(sequence)
        
        # Process in chunks
        chunk_size = self.max_len
        outputs = []
        hidden = None
        
        for i in range(0, len(sequence), chunk_size):
            chunk = sequence[i:i+chunk_size]
            out, hidden = self.model(chunk, hidden)
            outputs.append(out)
            hidden = (hidden[0].detach(), hidden[1].detach())
        
        # Aggregate (e.g., use last output, or average)
        return outputs[-1]
```

**Key Takeaway:** Always test on sequences **longer** than training. LSTMs may not generalize!

---

#### Q4: Explain gradient clipping. When and why would you use it?

**Answer:**

**What is Gradient Clipping?**

Scaling gradients to a maximum norm to prevent exploding gradients.

```python
import torch.nn.utils as utils

# After loss.backward(), before optimizer.step()
utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**How It Works:**

1. Compute total gradient norm:
   ```
   ||g|| = sqrt(Σ_i ||g_i||^2) for all parameters
   ```

2. If ||g|| > max_norm:
   ```
   g' = g × (max_norm / ||g||)
   ```
   Scale gradient so ||g'|| = max_norm

3. Else: Do nothing

**When to Use:**

**1. Recurrent Networks (RNNs/LSTMs):**

Most common use case. Gradients through time can explode:

```python
# Without clipping
# Gradient norm: 1000 → Update too large → Parameters diverge

# With clipping (max_norm=1.0)
# Gradient norm: 1000 → Clipped to 1.0 → Stable updates
```

**2. Very Deep Networks:**

Gradients accumulate through many layers.

**3. Unstable Training:**

Signs you need clipping:
- Loss suddenly spikes to NaN or infinity
- Gradient norms > 10-100
- Parameters become very large (overflow)

**Why It Works:**

**Example:**

```python
# Bad gradient
grad = torch.tensor([100, 200, 50])  # Large gradients
||grad|| ≈ 229

# After clipping (max_norm=1.0)
grad_clipped = grad * (1.0 / 229)
grad_clipped ≈ [0.44, 0.87, 0.22]
||grad_clipped|| = 1.0

# Direction preserved, magnitude controlled!
```

**Key Insight:** Clipping preserves gradient **direction**, just limits **magnitude**.

**Choosing max_norm:**

Rule of thumb:
- Start monitoring gradient norms without clipping
- If norms regularly > 5-10: Use clipping
- Set max_norm to 90th percentile of observed norms

Typical values:
- RNNs: 1.0 - 5.0
- Transformers: 1.0 (less prone to explosion)
- Very large models: 0.5 - 1.0

**Implementation with Monitoring:**

```python
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    # Monitor before clipping
    total_norm_before = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** 0.5
    
    # Clip
    total_norm_after = utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Log if clipped significantly
    if total_norm_after < total_norm_before * 0.5:
        print(f"Clipped: {total_norm_before:.2f} → {total_norm_after:.2f}")
    
    optimizer.step()
```

**When NOT to Use:**

- Transformers (usually stable without clipping)
- Small models on standard datasets
- If you never see gradient explosion

Over-aggressive clipping can slow convergence!

---

#### Q5: You have a seq2seq model (LSTM encoder-decoder) that works well on short sequences but degrades on longer ones. Diagnose and propose solutions.

**Answer:**

**Symptoms:**

```
Short sequence (10 tokens): BLEU score 0.85
Medium sequence (50 tokens): BLEU score 0.60
Long sequence (100 tokens): BLEU score 0.30
```

**Diagnosis:**

**Problem 1: Information Bottleneck**

Standard seq2seq:
```
Encoder: x_1, ..., x_100 → Single vector h
Decoder: h → y_1, ..., y_100
```

100 tokens compressed into single vector (512-1024 dim) → Information loss!

**Test:**
```python
# Check encoder final state
h_final = encoder(long_sequence)
# If all sequences have similar h_final → bottleneck confirmed
```

**Problem 2: Attention Failures**

If using attention, it might not work well:

```python
# Compute attention weights for long sequence
attention_weights = model.get_attention(long_sequence)

# Check if attention is:
# - Too uniform (not focusing)
# - Only on recent context (forget early tokens)
```

**Problem 3: Exposure Bias**

During training: Teacher forcing (use ground truth)
During inference: Use model's own predictions

Error accumulates for long sequences!

**Solutions:**

**1. Add Attention (If Not Using):**

```python
class Seq2SeqWithAttention(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(...)
        self.decoder = nn.LSTM(...)
        self.attention = Attention()
    
    def forward(self, src, tgt):
        encoder_outputs, hidden = self.encoder(src)
        
        decoder_outputs = []
        for t in range(len(tgt)):
            # Compute attention over ALL encoder outputs
            context, attn_weights = self.attention(
                decoder_hidden, encoder_outputs
            )
            
            # Use context + decoder input
            decoder_out, decoder_hidden = self.decoder(
                torch.cat([tgt[t], context]), decoder_hidden
            )
            decoder_outputs.append(decoder_out)
        
        return torch.stack(decoder_outputs)
```

**2. Scheduled Sampling (Reduce Exposure Bias):**

```python
# Gradually transition from teacher forcing to model predictions
for epoch in range(num_epochs):
    teacher_forcing_ratio = max(0.5, 1.0 - epoch * 0.1)
    
    for t in range(seq_len):
        if random.random() < teacher_forcing_ratio:
            decoder_input = ground_truth[t]  # Teacher forcing
        else:
            decoder_input = model_output[t-1]  # Model's prediction
        
        output = decoder(decoder_input, hidden)
```

**3. Bidirectional Encoder:**

```python
# Encoder sees both directions
encoder = nn.LSTM(..., bidirectional=True)

# Richer context for decoder
```

**4. Segment Long Sequences:**

```python
# For very long sequences (200+ tokens)
# Break into segments, process separately, recombine

def hierarchical_encode(long_sequence, segment_size=50):
    segments = split_into_segments(long_sequence, segment_size)
    
    segment_encodings = []
    for segment in segments:
        encoding = encoder(segment)
        segment_encodings.append(encoding)
    
    # Higher-level encoder over segment encodings
    final_encoding = meta_encoder(torch.stack(segment_encodings))
    return final_encoding
```

**5. Switch to Transformer:**

Transformers handle long sequences better:

```python
from transformers import MarianMTModel

# Pre-trained seq2seq transformer
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Handles sequences up to 512 tokens natively
# With modifications (Longformer), up to 4096+
```

**Production Fix Priority:**

1. **Immediate**: Add attention (biggest impact)
2. **Short-term**: Scheduled sampling
3. **Long-term**: Consider Transformer migration

**Verification:**

```python
# After fixes, test on length buckets
test_lengths = [10, 25, 50, 75, 100, 150]

for length in test_lengths:
    samples = filter_by_length(test_set, length)
    score = evaluate(model, samples)
    print(f"Length {length}: Score {score}")

# Should see: More gradual degradation (not cliff)
```

---

### Production Challenges

**Challenge: RNN Inference Latency in Production**

**Scenario:**

LSTM model serves 1000 QPS, each request processes 100-token sequence.
Latency: 50ms per request (too slow!).

**Bottleneck:** Sequential processing (can't parallelize across sequence).

**Solutions:**

1. **Batch Inference**:
   ```python
   # Instead of processing requests one-by-one
   # Batch multiple requests together
   batch_size = 32
   # Amortize sequential cost
   ```

2. **Replace with CNN or Transformer** (if task allows)

3. **Quantization**: INT8 LSTM (4× smaller, 2-3× faster)

4. **Distillation**: Train smaller LSTM using large LSTM as teacher

---

### Key Takeaways for Interviews

1. **Understand gradient flow**: Vanishing/exploding, why LSTM solves it
2. **Know trade-offs**: RNN vs LSTM vs GRU vs CNN vs Transformer
3. **Production awareness**: Length generalization, latency, memory
4. **Debugging skills**: Gradient clipping, scheduled sampling, attention analysis
5. **Modern context**: Acknowledge Transformer dominance, know when RNNs still useful
