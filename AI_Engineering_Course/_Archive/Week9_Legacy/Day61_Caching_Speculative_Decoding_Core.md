# Day 61: Caching & Speculative Decoding
## Core Concepts & Theory

### The Latency Bottleneck

**The Problem:** LLM generation is autoregressive.
- To generate token $T_N$, we need $T_1...T_{N-1}$.
- This is inherently serial. We cannot parallelize the generation of a single sequence.
- **Memory Bound:** Each step reads all weights.

**Solutions:**
1.  **Caching:** Avoid recomputing what we already know (KV Cache, Semantic Cache).
2.  **Speculative Decoding:** Break the serial dependency by guessing and verifying.

### 1. KV Cache (Key-Value Cache)

**Concept:**
- In Attention, $Q$ interacts with $K$ and $V$.
- For token $T_N$, the $K$ and $V$ for $T_1...T_{N-1}$ are the same as they were in previous steps.
- **Strategy:** Cache $K$ and $V$ vectors for all previous tokens. Only compute $Q, K, V$ for the *new* token.

**Impact:**
- Reduces complexity from $O(N^2)$ to $O(N)$ per step (compute-wise).
- **Cost:** High memory usage. (e.g., 70B model, 4k context $\approx$ 2-3GB per request).

### 2. Semantic Caching

**Concept:**
- Cache the *response* to a prompt, not just internal states.
- If a user asks "What is the capital of France?" twice, serve the second one from Redis/Vector DB.

**Mechanism:**
- **Exact Match:** Hash of the prompt.
- **Semantic Match:** Embedding similarity. If distance(new_prompt, cached_prompt) < threshold, return cached response.

**Benefit:**
- Zero latency for hits.
- Zero GPU cost for hits.

### 3. Prompt Caching (Prefix Caching)

**Concept:**
- Many requests share the same system prompt or few-shot examples (the "Prefix").
- **RadixAttention (vLLM):** Store KV cache in a radix tree.
- If a new request starts with the same tokens, reuse the cached KV blocks.

**Benefit:**
- "Zero-shot" latency for the prefix part.
- Massive savings for RAG (where context is large and reused) or Agents (shared system instructions).

### 4. Speculative Decoding (Deep Dive)

**Intuition:**
- Large models (70B) are slow. Small models (7B) are fast.
- Small models are often "right enough" for easy tokens (e.g., "The", "is", "of").
- **Idea:** Let the small model "draft" K tokens. Let the large model "verify" them in parallel.

**Algorithm:**
1.  **Draft:** Small model generates $K$ tokens ($t_1...t_K$).
2.  **Verify:** Large model processes the sequence $[prompt, t_1...t_K]$ in one forward pass.
3.  **Check:** For each $t_i$, did the large model agree? (or is $P(t_i)$ high enough?)
4.  **Accept:** Keep the prefix that matches. Discard the rest.
5.  **Bonus:** Large model predicts one extra token $t_{correct}$.

**Speedup:**
- If acceptance rate $\alpha$ is high, speedup $\approx \frac{1}{1-\alpha + \frac{1}{K}}$.
- Typical speedup: 2x - 3x.

### 5. Medusa & Lookahead Decoding

**Medusa:**
- Instead of a separate draft model, add extra "heads" to the main model.
- Heads predict $T_{N+1}, T_{N+2}, ...$ simultaneously.
- **Benefit:** No need for a separate draft model. Simpler deployment.

**Lookahead Decoding:**
- No draft model.
- Generate multiple branches (Jacobi iteration) or use n-gram features from the context itself.
- "The model drafts for itself".

### 6. Prompt Lookup Decoding

**Concept:**
- The input context often contains the answer (e.g., summarization, extraction).
- **Draft Strategy:** Search for n-gram matches in the *input prompt* to predict next tokens.
- **Verify:** Use the model to verify.
- **Benefit:** Extremely cheap draft (string matching). Very effective for RAG/Summarization.

### 7. Cache Eviction Policies

**LRU (Least Recently Used):**
- Standard policy. Evict blocks not accessed recently.

**Belady's Algorithm (Oracle):**
- Evict blocks that will be used furthest in the future. (Impossible in practice, but a benchmark).

**Attention-Based Eviction (H2O, StreamingLLM):**
- Evict tokens with low attention scores.
- Keep "Attention Sinks" (first few tokens) + "Local" tokens (recent ones).
- **Benefit:** Infinite context length with finite cache.

### 8. System Architecture for Caching

**Layers:**
1.  **L1 Cache (SRAM/GPU):** KV Cache (Prefix Caching).
2.  **L2 Cache (DRAM/CPU):** Swapped out KV blocks.
3.  **L3 Cache (Network/Redis):** Semantic Response Cache.

### Summary

**Optimization Strategy:**
1.  **KV Cache:** Mandatory for decoding.
2.  **Prefix Caching:** Mandatory for high-load systems with shared prompts.
3.  **Semantic Caching:** High ROI for repetitive queries.
4.  **Speculative Decoding:** Best way to reduce latency for single users without losing quality.

### Next Steps
In the Deep Dive, we will implement a Semantic Cache with Vector DB and a simplified Speculative Decoding verification loop.
