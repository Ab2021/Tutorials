# 🤖 LLM & RAG Coding — Part 2: Generation, Decoding & Sampling
> **Difficulty:** Medium → Hard | **Focus:** Autoregressive Generation, Sampling Strategies, KV Cache
> **Flipkart Relevance:** 🔥🔥🔥 — LLM inference optimization, GenAI systems

---

## 🟡 PROBLEM 1: Greedy Decoding

### Theory
At each step, pick the token with highest probability:
$$w_t = \arg\max_{v} P(v | w_1, ..., w_{t-1})$$

**Problems:**
- Misses globally optimal sequence (greedy ≠ optimal)
- Produces repetitive, dull text
- "I like to like to like to like..."

But very fast: O(L × V) for sequence of length L, vocabulary V.

### Implementation
```python
import numpy as np
from typing import List, Optional, Callable

def greedy_decode(logits_fn: Callable, prompt_ids: List[int], 
                   max_new_tokens: int = 50, eos_id: int = 2) -> List[int]:
    """
    Greedy autoregressive decoding.
    
    logits_fn: function(token_ids) → logits (vocab_size,) for next token
    prompt_ids: initial token IDs
    max_new_tokens: max tokens to generate
    eos_id: stop generating when this token is output
    """
    generated = list(prompt_ids)
    
    for _ in range(max_new_tokens):
        # Get logits for next token (shape: vocab_size)
        logits = logits_fn(generated)
        
        # Pick highest probability token
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        
        # Stop if EOS token generated
        if next_token == eos_id:
            break
    
    return generated

# Demo (with mock logits function)
def mock_logits_fn(token_ids: List[int]) -> np.ndarray:
    """Mock LM — returns random logits based on last token."""
    vocab_size = 100
    np.random.seed(sum(token_ids) % 1000)  # Deterministic
    return np.random.randn(vocab_size)

prompt = [1, 5, 12, 7]  # Some token IDs
result = greedy_decode(mock_logits_fn, prompt, max_new_tokens=10, eos_id=2)
print(f"Greedy output: {result}")
```

---

## 🟡 PROBLEM 2: Temperature Sampling

### Theory
Temperature T controls the "creativity" of the model:

$$P_T(w_t) = \text{softmax}(z_t / T)$$

- T → 0: becomes greedy (argmax)
- T = 1: original distribution (default)
- T → ∞: uniform distribution (maximum randomness)

**Effect on logits:**
- Low T (0.1): sharpens distribution → model is very confident
- High T (2.0): flattens distribution → model explores more options

### Things to Focus On
- ✅ T < 1 for factual tasks (code, math, facts) — be precise
- ✅ T > 1 for creative tasks (stories, brainstorming) — be diverse
- ✅ Temperature only changes PROBABILITIES, not the logits order

### Implementation
```python
def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to logits before softmax."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Scale logits by temperature
    scaled = logits / temperature
    
    # Numerically stable softmax
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    return probs / probs.sum()

def temperature_sample(logits: np.ndarray, temperature: float = 1.0, 
                        seed: int = None) -> int:
    """Sample one token using temperature-scaled distribution."""
    if seed is not None:
        np.random.seed(seed)
    probs = softmax_with_temperature(logits, temperature)
    return int(np.random.choice(len(probs), p=probs))

# Demo: effect of temperature
logits = np.array([3.0, 2.0, 1.5, 0.5, 0.1])
for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"T={T:.1f}: probs={probs.round(3)}, entropy={-np.sum(probs*np.log(probs+1e-10)):.3f}")
# T=0.1: very peaked (low entropy)
# T=5.0: very flat (high entropy)
```

---

## 🟡 PROBLEM 3: Top-K Sampling

### Theory
Restrict sampling to only the K most probable tokens:

1. Sort tokens by probability (descending)
2. Keep only top K
3. Re-normalize to valid distribution
4. Sample from this restricted distribution

**Advantage over temperature-only:** Eliminates long tail of very unlikely tokens (reduces "hallucination-prone" low-probability choices while maintaining diversity in top tokens).

### Implementation
```python
def top_k_sample(logits: np.ndarray, k: int = 50, 
                  temperature: float = 1.0, seed: int = None) -> int:
    """
    Top-K sampling: restrict to K most likely tokens before sampling.
    
    logits: (vocab_size,) raw logits  
    k: number of top tokens to consider
    temperature: scaling before softmax
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Apply temperature
    logits_t = logits / temperature
    
    # Find top-K token indices
    top_k_indices = np.argpartition(logits_t, -k)[-k:]  # O(V) via partition
    top_k_logits  = logits_t[top_k_indices]
    
    # Softmax over top-K only
    top_k_logits = top_k_logits - top_k_logits.max()
    probs = np.exp(top_k_logits)
    probs = probs / probs.sum()
    
    # Sample from top-K distribution
    chosen_idx = np.random.choice(len(top_k_indices), p=probs)
    return int(top_k_indices[chosen_idx])

# Test
np.random.seed(42)
vocab_size = 1000
logits = np.random.randn(vocab_size)

samples = [top_k_sample(logits, k=50) for _ in range(1000)]
print(f"Top-K=50 sampling: unique tokens drawn = {len(set(samples))}")
samples_greedy = [int(np.argmax(logits))] * 1000
print(f"Greedy: unique tokens = {len(set(samples_greedy))}")  # Always 1
```

---

## 🔥 PROBLEM 4: Top-P (Nucleus) Sampling

### Theory
Dynamic approach — include the minimum set of tokens whose cumulative probability ≥ p:

1. Sort tokens by probability (descending)
2. Compute cumulative sum
3. Cut off at point where cumsum ≥ p
4. Re-normalize and sample from nucleus

**Why top-P over top-K?**
- Top-K: fixed K — in high-confidence situations, K might include bad tokens (flat region)
- Top-P: dynamic — adapts K to the distribution's sharpness
  - High-confidence next token (clear winner): nucleus = 1-2 tokens
  - Uncertain next token: nucleus = 20+ tokens

### Things to Focus On
- ✅ p=0.9 or p=0.95 most common in practice (used by ChatGPT)
- ✅ Often combined with temperature: apply T first, then nucleus sampling
- ✅ p=1.0: sample from full distribution; p→0: approaches greedy

### Implementation
```python
def top_p_sample(logits: np.ndarray, p: float = 0.9, 
                  temperature: float = 1.0, seed: int = None) -> int:
    """
    Nucleus (top-p) sampling.
    
    logits: (vocab_size,) raw logits
    p: cumulative probability threshold (e.g., 0.9)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Apply temperature scaling
    logits_t = logits / temperature
    
    # Convert to probabilities (sorted descending)
    logits_t = logits_t - logits_t.max()
    probs = np.exp(logits_t)
    probs = probs / probs.sum()
    
    sorted_indices = np.argsort(-probs)  # Descending sort
    sorted_probs   = probs[sorted_indices]
    
    # Cumulative sum — find nucleus boundary
    cumsum_probs = np.cumsum(sorted_probs)
    
    # Keep tokens where cumsum shifts FROM below p TO above p
    # i.e., include minimum tokens until cumsum ≥ p
    nucleus_mask = cumsum_probs <= p
    nucleus_mask[nucleus_mask.argmin()] = True  # Always include at least the crossing token
    
    # Restrict to nucleus
    nucleus_indices = sorted_indices[nucleus_mask]
    nucleus_probs   = sorted_probs[nucleus_mask]
    nucleus_probs   = nucleus_probs / nucleus_probs.sum()  # Renormalize
    
    # Sample from nucleus
    chosen = np.random.choice(len(nucleus_indices), p=nucleus_probs)
    return int(nucleus_indices[chosen])

def combined_sampling(logits: np.ndarray, temperature: float = 0.8,
                       top_k: int = 0, top_p: float = 0.9,
                       seed: int = None) -> int:
    """
    Combined sampling: temperature → top-K → top-P → sample.
    This is the standard production inference pipeline.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Temperature scaling
    logits_t = logits / temperature
    
    # Step 2: Top-K filtering (optional; k=0 means skip)
    if top_k > 0:
        top_k_threshold = np.sort(logits_t)[-top_k]  # kth largest
        logits_t = np.where(logits_t >= top_k_threshold, logits_t, -np.inf)
    
    # Step 3: Compute probabilities
    logits_t = logits_t - np.max(logits_t[logits_t != -np.inf])
    probs = np.exp(logits_t)
    probs = np.where(logits_t == -np.inf, 0, probs)
    probs = probs / probs.sum()
    
    # Step 4: Top-P nucleus filtering
    sorted_idx = np.argsort(-probs)
    sorted_p = probs[sorted_idx]
    cumsum = np.cumsum(sorted_p)
    nucleus_mask = cumsum <= top_p
    nucleus_mask[nucleus_mask.argmin()] = True
    
    nucleus_idx = sorted_idx[nucleus_mask]
    nucleus_p = probs[nucleus_idx]
    nucleus_p = nucleus_p / nucleus_p.sum()
    
    return int(np.random.choice(nucleus_idx, p=nucleus_p))

# Test: show distribution concentration
np.random.seed(42)
logits = np.random.randn(50000)  # Large vocabulary

samples_greedy = {int(np.argmax(logits))}
samples_topk = {top_k_sample(logits, k=50, seed=i) for i in range(500)}
samples_topp = {top_p_sample(logits, p=0.9, seed=i) for i in range(500)}

print(f"Greedy unique tokens: {len(samples_greedy)}")     # 1
print(f"Top-K (k=50) unique: {len(samples_topk)}")        # ≤50
print(f"Top-P (p=0.9) unique: {len(samples_topp)}")       # Dynamic
```

---

## 🔴 PROBLEM 5: Beam Search

### Theory
Beam search maintains the top-`beam_width` most likely sequences at each step.

**Algorithm:**
1. Initialize beams with prompt
2. For each step: expand each beam by V tokens → beam_width × V candidates
3. Score each candidate: log P(beam) = Σᵢ log P(wᵢ | w₁,...,wᵢ₋₁)
4. Keep top beam_width candidates → new beams
5. Stop when all beams end with EOS

**Beam width tradeoffs:**
- Width=1: greedy decoding
- Width=5: typical NLP use (BLEU optimization)
- Width=∞: exhaustive search (impractical for large vocab)

**Length normalization:**
$$\text{score} = \frac{\log P(\text{sequence})}{|\text{sequence}|^\alpha}, \quad \alpha \approx 0.6-0.7$$

### Things to Focus On
- ✅ Beam search optimizes log P (which maximizes BLEU) but may not maximize human preference
- ✅ Diverse beam search: penalize beams that are too similar to each other
- ✅ Time complexity: O(L × B × V) — expensive for large V (use top-K first)
- ✅ LLMs for chat: sampling preferred over beam (more natural); beam for translation

### Implementation
```python
def beam_search(logits_fn: Callable, prompt_ids: List[int],
                 beam_width: int = 5, max_new_tokens: int = 50,
                 eos_id: int = 2, alpha: float = 0.7) -> List[List[int]]:
    """
    Beam search decoding.
    
    logits_fn: function(token_ids) → (vocab_size,) logits for next token
    beam_width: number of beams to maintain
    alpha: length normalization factor (0=no norm, 1=full norm)
    
    Returns: top beam_width completed sequences (sorted by normalized score)
    """
    # Initialize: one beam = (log_prob, token_sequence, is_done)
    beams = [(0.0, list(prompt_ids), False)]
    completed = []
    
    for step in range(max_new_tokens):
        # If all beams are done, stop
        if all(done for _, _, done in beams):
            break
        
        candidates = []
        
        for log_prob, sequence, done in beams:
            if done:
                # Don't expand completed beams, just keep them
                candidates.append((log_prob, sequence, True))
                continue
            
            # Get logits for next token
            next_logits = logits_fn(sequence)
            
            # Compute log probabilities
            next_logits = next_logits - np.max(next_logits)
            log_probs = next_logits - np.log(np.sum(np.exp(next_logits)))
            
            # Expand: consider all vocabulary tokens
            # (In practice: pre-filter to top-K for efficiency)
            for token_id in range(len(log_probs)):
                new_log_prob = log_prob + log_probs[token_id]
                new_sequence = sequence + [token_id]
                is_done = (token_id == eos_id)
                candidates.append((new_log_prob, new_sequence, is_done))
        
        # Score with length normalization
        def normalized_score(beam):
            log_prob, seq, done = beam
            length = len(seq) - len(prompt_ids)
            return log_prob / (max(length, 1) ** alpha)
        
        # Keep top beam_width candidates
        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:beam_width]
    
    # Return sequences sorted by score
    beams.sort(key=lambda b: b[0] / max(len(b[1]) - len(prompt_ids), 1) ** alpha,
               reverse=True)
    return [seq for _, seq, _ in beams]

# Demo
def simple_logits_fn(ids: List[int]) -> np.ndarray:
    vocab_size = 20
    np.random.seed(len(ids) % 100)
    return np.random.randn(vocab_size)

prompt = [0, 1]
beams = beam_search(simple_logits_fn, prompt, beam_width=3, max_new_tokens=5, eos_id=2)
print(f"Top beam: {beams[0]}")
print(f"2nd beam: {beams[1]}")
```

---

## 🔴 PROBLEM 6: KV Cache (Key-Value Cache for Inference Optimization)

### Theory
**Without KV cache:** For each new token, recompute Q, K, V for the ENTIRE sequence.

**With KV cache:** Store previously computed K and V tensors. Only compute for the new token.

**Memory:** O(n × layers × d_model) — grows with sequence length.
**Speedup:** From O(n²) to O(n) time per generation step.

This is why: first token = slow (full prefill), subsequent tokens = fast (cached decode).

### Things to Focus On
- ✅ "Prefill" phase: process prompt → compute+cache all K, V
- ✅ "Decode" phase: generate one token at a time, only new Q × cached K, V
- ✅ KV cache grows every step → memory bottleneck for long sequences
- ✅ PagedAttention (vLLM): manage KV cache like virtual memory → higher throughput
- ✅ GQA (Grouped Query Attention): share K, V across heads → smaller KV cache

### Implementation
```python
class AttentionWithKVCache:
    """
    Self-attention with KV cache for efficient autoregressive generation.
    
    Performance comparison:
    - Without cache: O(L²) per token (recompute all K, V)
    - With cache: O(L) per token (only compute new Q; use cached K, V)
    """
    
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projection matrices
        scale = np.sqrt(2 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        # KV Cache: {layer_id: {'K': np.ndarray, 'V': np.ndarray}}
        self.kv_cache = {'K': None, 'V': None}
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, d = x.shape
        return x.reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
    
    def forward_prefill(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Prefill phase: process full prompt, build initial KV cache.
        x: (batch, prompt_len, d_model)
        """
        Q = self._split_heads(x @ self.W_Q)
        K = self._split_heads(x @ self.W_K)
        V = self._split_heads(x @ self.W_V)
        
        # Store K, V in cache
        self.kv_cache['K'] = K
        self.kv_cache['V'] = V
        
        # Full attention over prompt
        return self._attend(Q, K, V, mask)
    
    def forward_decode(self, x_new: np.ndarray) -> np.ndarray:
        """
        Decode phase: generate one new token using cached K, V.
        x_new: (batch, 1, d_model) — single new token
        """
        # Compute Q, K, V for ONLY the new token
        Q_new = self._split_heads(x_new @ self.W_Q)  # (batch, heads, 1, d_k)
        K_new = self._split_heads(x_new @ self.W_K)
        V_new = self._split_heads(x_new @ self.W_V)
        
        # Append new K, V to cache (concatenate along sequence dimension)
        K_full = np.concatenate([self.kv_cache['K'], K_new], axis=2)
        V_full = np.concatenate([self.kv_cache['V'], V_new], axis=2)
        
        # Update cache
        self.kv_cache['K'] = K_full
        self.kv_cache['V'] = V_full
        
        # Attend: new Q (length 1) × full K (full cached sequence)
        return self._attend(Q_new, K_full, V_full, mask=None)
    
    def _attend(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: np.ndarray = None) -> np.ndarray:
        d_k = Q.shape[-1]
        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        scores_max = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - scores_max)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        
        out = weights @ V
        return self._combine_heads(out) @ self.W_O

# Demonstration
d_model, n_heads = 64, 4
attn = AttentionWithKVCache(d_model, n_heads)

# Prefill: process prompt of length 10
prompt = np.random.randn(1, 10, d_model)
causal_mask = np.triu(np.ones((10, 10), dtype=bool), k=1)
out_prefill = attn.forward_prefill(prompt, mask=causal_mask)
print(f"Prefill output: {out_prefill.shape}")   # (1, 10, 64)
print(f"KV cache K: {attn.kv_cache['K'].shape}") # (1, 4, 10, 16)

# Decode: generate 5 tokens one at a time
for step in range(5):
    new_token = np.random.randn(1, 1, d_model)  # New token embedding
    out_decode = attn.forward_decode(new_token)
    print(f"Step {step+1}: new out={out_decode.shape}, cache size={attn.kv_cache['K'].shape[2]}")
# Cache grows by 1 each step ✓
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"Greedy vs beam search vs sampling — when to use each?"** → Greedy: fast debugging, simple tasks. Beam: translation (BLEU), summarization (quality). Sampling (T+top-p): chat, creative, diverse outputs. Rule of thumb: determinism needed → beam/greedy; diversity needed → sampling.
2. **"KV cache doubles memory use. How do you handle long sequences?"** → (a) Sliding window attention (only last W tokens in cache) — Mistral uses this. (b) PagedAttention (vLLM) — treat cache as paged virtual memory. (c) GQA: multiple Q heads share K, V → cache size ÷ n_heads_ratio.
3. **"Top-P p=0.9 in practice — what does that mean?"** → At each step, we include the minimum number of tokens whose combined probability ≥ 90%. If model is very confident (one token has 95% prob), nucleus = just that token. If uncertain, nucleus = many tokens.
4. **"Why is temperature applied BEFORE top-p in the combined pipeline?"** → Temperature changes the distribution shape first (sharpen/flatten). Then top-p prunes based on the TEMPERATURE-ADJUSTED probabilities. If you applied T after top-p, you'd select the nucleus at the wrong temperatures and then distort within it.
