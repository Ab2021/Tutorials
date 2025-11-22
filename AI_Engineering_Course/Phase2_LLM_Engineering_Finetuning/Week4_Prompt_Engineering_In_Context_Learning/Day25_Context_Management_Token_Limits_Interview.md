# Day 25: Context Management & Token Limits
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What happens if you exceed the context window of a Transformer?

**Answer:**
- **Positional Encoding Failure:** The model has no embedding for position $L+1$.
- **Crash:** Most implementations will throw an index out of bounds error.
- **Truncation:** In production, we usually truncate the input (drop the oldest messages) to fit.
- **Degradation:** Even if using ALiBi (which theoretically handles infinite length), performance degrades because the attention mechanism becomes diluted over too many tokens.

#### Q2: Explain "RoPE Scaling" and why it's needed.

**Answer:**
- **Problem:** Models pre-trained on 4k context haven't seen the high-frequency rotation components associated with positions > 4k.
- **Solution:** We interpolate the position indices. To fit 8k tokens into a 4k window, we map position $i$ to $i/2$.
- **Effect:** The model sees "slower" rotations that look like the ones it saw during training. This allows it to attend to longer distances without retraining from scratch (just fine-tuning).

#### Q3: What is the "Attention Sink" phenomenon?

**Answer:**
- **Observation:** In many LLMs, the first token (e.g., `<s>`) accumulates a massive amount of attention weight from all future tokens, even if it has no semantic meaning.
- **Reason:** The Softmax function forces attention weights to sum to 1. If the current token doesn't need to attend to anything specific, it dumps the probability mass on the first token as a "sink".
- **Implication:** You cannot simply evict the first token in a sliding window cache; you must keep it to maintain stability (StreamingLLM).

#### Q4: How does RAG help with context limits?

**Answer:**
- **Decoupling:** RAG separates "Knowledge" (Vector DB, infinite size) from "Working Memory" (Context Window, fixed size).
- **Selection:** Instead of feeding the entire knowledge base to the model, we retrieve only the top-k most relevant chunks.
- **Efficiency:** This allows answering questions about a 1GB PDF using a 4k context window.

#### Q5: Why is "Lost in the Middle" a problem for RAG?

**Answer:**
- If you retrieve 10 documents and the correct answer is in Document #5 (the middle), the model is statistically less likely to use it compared to Document #1 or #10.
- **Fix:** Re-rank the retrieved documents so the most relevant ones are placed at the start and end of the context (U-shaped placement).

---

### Production Challenges

#### Challenge 1: Handling "Infinite" Chat History

**Scenario:** A user has been chatting for 5 hours. The history is 50k tokens. Model limit is 8k.
**Solution:**
- **Summarization:** Every 10 turns, ask a cheaper model (GPT-3.5) to summarize the conversation so far. Replace the old turns with the summary.
- **Vector Memory:** Store old turns in a Vector DB. Retrieve relevant past turns based on the current user query (Long-term memory).

#### Challenge 2: Cost of Long Context

**Scenario:** You switched to GPT-4-32k. Your bill exploded.
**Analysis:**
- Cost is linear with input tokens. Sending 30k tokens for every simple "Hi" is wasteful.
- **Solution:**
    - **Dynamic Context:** Only include the full history if the user asks a question that requires it ("What did we talk about 1 hour ago?").
    - **Filtering:** Remove irrelevant system logs or boilerplate from the context.

#### Challenge 3: Latency Spikes with Long Context

**Scenario:** TTFT (Time To First Token) is 5 seconds when context is full.
**Root Cause:** Processing the prompt (Prefill) takes $O(N^2)$ or $O(N)$ compute.
**Solution:**
- **KV Caching:** If the system prompt and documents are static, cache the KV states.
- **Flash Attention:** Mandatory for long context.
- **Speculative Decoding:** Doesn't help prefill, but helps generation.

#### Challenge 4: Tokenizer Mismatch

**Scenario:** You count tokens using `len(text.split())` and truncate to 4000. The API still rejects it.
**Root Cause:** Words $\neq$ Tokens. "Hello" is 1 token, but complex strings or code might be more.
**Solution:**
- Always use the exact tokenizer for the model (e.g., `tiktoken` for OpenAI, `LlamaTokenizer` for LLaMA) to count and truncate.

### Summary Checklist for Production
- [ ] **Count:** Use `tiktoken` to count exactly.
- [ ] **Truncate:** Keep System Prompt + Summary + Last $K$ turns.
- [ ] **RAG:** Re-rank docs to put best at start/end.
- [ ] **Cache:** Use KV caching for static context.
