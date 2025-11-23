# Day 25: Context Management & Token Limits
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. RoPE Scaling Mathematics

**Original RoPE:**
Encodes position $m$ by rotating the embedding vector by angle $m\theta$.
$\theta_i = 10000^{-2i/d}$.
The period of rotation increases with dimension index $i$.

**The Problem:**
If trained on length $L$, the model has never seen rotation angles corresponding to positions $> L$.
Extrapolating to $2L$ fails because the high-frequency components rotate too fast and become "out of distribution".

**Linear Scaling (Interpolation):**
Map position $m'$ in the extended window $[0, L']$ to $m$ in the original window $[0, L]$.
$m = m' \times \frac{L}{L'}$.
Effectively, we "slow down" the rotation by a factor of $s = L'/L$.
$\theta_{new} = \theta / s$.
**Result:** The model sees familiar frequencies, but the resolution is coarser. Requires fine-tuning.

**NTK-Aware Scaling:**
Instead of scaling all frequencies linearly, scale high frequencies less and low frequencies more.
This preserves the high-frequency information (local attention) while allowing low-frequency components (global attention) to stretch.
**Result:** Better zero-shot extrapolation.

### 2. Conversation Memory Implementation

**Naive:** Append every message to a list. `context = "\n".join(history)`.
**Problem:** Explodes quickly.

**Buffer Window Memory:**
Keep only the last $k$ turns.
```python
history = [m1, m2, m3, m4, m5]
k = 2
window = [m4, m5]
```
**Problem:** Loses context from the start (e.g., user's name).

**Summary Memory:**
Maintain a running summary of the conversation.
```python
summary = "User asked about apples. AI explained varieties."
history = [m4, m5]
context = f"Summary: {summary}\nChat: {history}"
```

### 3. KV Cache Eviction (StreamingLLM)

**Observation:**
Attention sinks. The first token (usually `<s>`) gets massive attention scores from all subsequent tokens.
If you evict the first token (FIFO), the model collapses.

**StreamingLLM Strategy:**
Keep the **Attention Sinks** (first 4 tokens) + **Sliding Window** (last 4096 tokens).
Discard the middle.
**Result:** Infinite streaming generation with stable perplexity.

### Code: Token-Managed Memory Buffer

```python
import tiktoken

class TokenBufferMemory:
    def __init__(self, max_tokens=4000, model="gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
        self.messages = []
        
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._trim()
        
    def _trim(self):
        while self.count_tokens() > self.max_tokens:
            # Remove oldest message (index 0)
            # Ideally remove pairs (User+Assistant) to keep flow
            if len(self.messages) > 1:
                self.messages.pop(0)
            else:
                break
                
    def count_tokens(self):
        text = "".join([m["content"] for m in self.messages])
        return len(self.encoding.encode(text))
        
    def get_context(self):
        return self.messages
```
