# Day 61: Caching & Speculative Decoding
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How does Speculative Decoding improve latency?

**Answer:**
- **Mechanism:** It uses a small "draft" model to guess $K$ tokens ahead, and a large "target" model to verify them in parallel.
- **Bottleneck Shift:** LLM decoding is memory-bound (loading weights). Verifying $K$ tokens takes roughly the same time as generating 1 token because we load the large model weights once.
- **Result:** If the draft is good, we generate $K$ tokens for the cost of 1 large-model step.
- **Trade-off:** If the draft is bad (low acceptance rate), we waste compute (drafting time) and might be slower than standard decoding.

#### Q2: What is Semantic Caching and how does it differ from exact caching?

**Answer:**
- **Exact Caching:** Key is `hash(prompt)`. Only hits if the user types the *exact* same string. Hit rate is low for natural language.
- **Semantic Caching:** Key is `embedding(prompt)`. Hits if the query is *semantically similar* (e.g., "Price of BTC" vs "Bitcoin price").
- **Implementation:** Vector Database (Redis/Milvus) stores (embedding, response) pairs. Search for nearest neighbor within a threshold (e.g., 0.9 cosine similarity).

#### Q3: Explain the concept of "Attention Sinks" in StreamingLLM.

**Answer:**
- **Observation:** In Transformer attention, the first few tokens (start-of-sequence) accumulate massive attention scores, even if they aren't semantically important.
- **Consequence:** If you evict them (e.g., in a sliding window), the model breaks (perplexity explodes).
- **Solution:** Always keep the first few tokens ("sinks") in the KV cache, plus the most recent tokens. This allows infinite streaming with finite memory.

#### Q4: What is Prefix Caching (RadixAttention)?

**Answer:**
- **Concept:** Caching the KV states of common prefixes (e.g., system prompts, few-shot examples).
- **Structure:** Stored in a Radix Tree (Trie).
- **Benefit:** When a new request shares a prefix with a previous request (or a cached node), we skip the computation for that prefix.
- **Use Case:** Multi-turn chat (history is shared), RAG (documents are shared), Agents (tools/instructions are shared).

#### Q5: Why is KV Cache quantization necessary for long contexts?

**Answer:**
- **Size:** KV cache size = $2 \times \text{layers} \times \text{hidden\_dim} \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes}$.
- **Example:** Llama-70B, 4k context, batch 1 $\approx$ 1.2GB. Batch 64 $\approx$ 80GB (exceeds A100 80GB just for cache!).
- **Solution:** Quantize KV cache to FP8 or INT4. Reduces memory by 2x-4x, allowing larger batch sizes or longer contexts.

---

### Production Challenges

#### Challenge 1: Semantic Cache Returning Stale/Wrong Data

**Scenario:** User asks "What is the price of Apple stock?" Cache returns yesterday's price because the prompt is semantically identical.
**Root Cause:** Semantic similarity doesn't capture time-sensitivity or dynamic data.
**Solution:**
- **TTL (Time To Live):** Set short expiry for dynamic topics.
- **Tagging:** Tag cache entries (e.g., "finance", "static"). Disable cache for "finance".
- **Prompt Enrichment:** Include timestamp in the prompt before embedding? (Might reduce hit rate). Better: Don't cache real-time queries.

#### Challenge 2: Speculative Decoding Slowdown

**Scenario:** Enabling speculative decoding made the model *slower*.
**Root Cause:**
- **Draft Model Mismatch:** Draft model is too weak or different vocabulary/tokenizer. Acceptance rate is near 0.
- **Overhead:** The cost of running the draft model > savings from parallel verification.
**Solution:**
- **Measure Acceptance Rate:** If $\alpha < 0.4$, disable it.
- **Tune K:** Reduce number of speculative tokens (e.g., 5 -> 3).
- **Distillation:** Train the draft model specifically to approximate the target model.

#### Challenge 3: Cache Stampede on System Restart

**Scenario:** Server restarts, Prefix Cache is empty. First few requests spike latency (recomputing system prompts).
**Root Cause:** Volatile cache.
**Solution:**
- **Warmup Script:** Replay common system prompts on startup to populate the Radix Tree.
- **Persistence:** Save/Load KV cache to disk (slow, but faster than recompute for massive prompts?). Usually warmup is preferred.

#### Challenge 4: Memory Fragmentation with Prefix Caching

**Scenario:** Radix Tree grows indefinitely, consuming all GPU memory.
**Root Cause:** Infinite unique prefixes.
**Solution:**
- **LRU Eviction:** Evict least recently used branches of the Radix Tree when memory is full.
- **Ref Counting:** Track active users of a block.

#### Challenge 5: Multi-Tenant Caching Privacy

**Scenario:** User A asks "Summarize my medical report". User B asks "Summarize medical report". Semantic cache returns User A's report.
**Root Cause:** Global cache without tenant isolation.
**Solution:**
- **Namespace/Tenant ID:** Key = `hash(tenant_id + prompt)` or filter vector search by `tenant_id`.
- **Never Cache PII:** Detect PII and bypass cache.

### System Design Scenario: Real-time Coding Assistant

**Requirement:** Low latency (<30ms) code completion.
**Design:**
1.  **Model:** Small specialized model (e.g., DeepSeek-Coder-1.3B) for draft, larger (7B) for verify? Or just a fast 7B.
2.  **Context:** File content + Cursor position.
3.  **Caching:**
    - **Prefix Cache:** Cache the file content up to the cursor line. As user types, only compute the new line.
    - **FIM (Fill-In-Middle):** Cache the suffix as well?
4.  **Speculative:** Use a tiny 100M param model for drafting.
5.  **Batching:** Continuous batching with high priority for "current line" completion.

### Summary Checklist for Production
- [ ] **KV Cache:** Enable **PagedAttention**.
- [ ] **Prefix Cache:** Enable for **Chat/Agents**.
- [ ] **Semantic Cache:** Use **Redis/VectorDB** for FAQ-style bots.
- [ ] **Speculative:** Test **acceptance rate** before enabling.
- [ ] **Eviction:** Configure **LRU** for cache management.
- [ ] **Privacy:** Ensure **tenant isolation** in semantic cache.
