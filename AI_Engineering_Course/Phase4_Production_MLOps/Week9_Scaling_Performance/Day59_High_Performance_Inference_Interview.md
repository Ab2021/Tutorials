# Day 59: High-Performance Inference Serving
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between Throughput and Latency in LLM serving?

**Answer:**
- **Latency:** Time taken for a single request.
  - **TTFT (Time To First Token):** Latency to see the first character. Important for interactive chat (perceived speed).
  - **TPOT (Time Per Output Token):** Speed of generation after the first token. Important for reading speed.
- **Throughput:** Total number of tokens generated per second across all concurrent requests.
- **Trade-off:** Increasing batch size improves throughput (better GPU utilization) but may degrade latency (queuing delays).

#### Q2: Explain Continuous Batching (Orbital Batching) and why it's better than static batching.

**Answer:**
- **Static Batching:** Waits for a batch of N requests. If one request finishes early (short sequence), the GPU slot is idle/padded until the longest request finishes.
- **Continuous Batching:** Schedules at the iteration level. When a sequence finishes, it is immediately evicted, and a new request from the queue is inserted into that slot *in the next forward pass*.
- **Benefit:** Eliminates padding waste, maximizes GPU utilization, improves throughput by 10-20x.

#### Q3: How does PagedAttention solve the KV Cache fragmentation problem?

**Answer:**
- **Problem:** Traditional KV cache allocates contiguous memory for the maximum possible sequence length (e.g., 2048 tokens), even if only 100 are used. This causes massive internal fragmentation and waste.
- **Solution:** PagedAttention divides the KV cache into fixed-size blocks (e.g., 16 tokens). These blocks are stored in non-contiguous memory (like OS pages).
- **Benefit:** Only allocates blocks as needed. Near-zero waste (<4%). Allows much larger batch sizes because memory is used efficiently.

#### Q4: What is Speculative Decoding?

**Answer:**
- **Concept:** Use a small, fast "draft" model to predict the next K tokens, then use the large "target" model to verify them in parallel.
- **Why it works:** LLMs are memory-bound. Reading weights for 1 token takes same time as reading for K tokens (for verification).
- **Gain:** If draft model is accurate, we generate K tokens in 1 large-model step. 2-3x speedup.

#### Q5: When should you use Tensor Parallelism vs Pipeline Parallelism for inference?

**Answer:**
- **Tensor Parallelism (TP):** Splits layers across GPUs. Reduces latency for a single request because computation is parallelized. **Preferred for Inference** to minimize latency.
- **Pipeline Parallelism (PP):** Splits model layers across GPUs (GPU1 has layers 1-10, GPU2 has 11-20). Increases throughput but increases latency (pipeline bubbles). **Preferred for Training** or very large models where TP communication overhead is too high.

---

### Production Challenges

#### Challenge 1: High TTFT (Time To First Token) under load

**Scenario:** Users complain that the chatbot takes 5 seconds to start typing.
**Root Cause:**
- Request queue is too long.
- Prefill (processing input prompt) is compute-heavy and blocking decode steps.
**Solution:**
- **Chunked Prefill:** Split long prompts into smaller chunks so decode steps can be interleaved.
- **Scale Out:** Add more replicas.
- **Prefix Caching:** If prompts share a common system prompt, cache the KV states.

#### Challenge 2: OOM (Out of Memory) with Long Contexts

**Scenario:** Server crashes with OOM when users send 10k+ token documents.
**Root Cause:** KV cache grows linearly with sequence length. 10k tokens * huge model = massive memory.
**Solution:**
- **PagedAttention:** Ensure it's enabled (vLLM).
- **Quantized KV Cache:** Use FP8 or INT8 for KV cache (2-4x memory savings).
- **Sliding Window Attention:** Only keep last N tokens in cache (if model supports it).
- **Limit Max Context:** Strictly enforce max_model_len.

#### Challenge 3: "Cold Start" Latency

**Scenario:** First request to a new model replica takes 30 seconds.
**Root Cause:** Model weights loading from disk/S3, compiling CUDA kernels.
**Solution:**
- **Model Baking:** Bake weights into the Docker image (fast startup).
- **Warmup:** Run a dummy inference request during startup probe.
- **Provisioned Concurrency:** Keep minimum replicas active (expensive but fast).

#### Challenge 4: Throughput vs Latency Tuning

**Scenario:** Throughput is high, but individual users feel it's slow (high TPOT).
**Root Cause:** Batch size is too large. GPU is saturated.
**Solution:**
- **Max Batch Size:** Cap the maximum batch size (e.g., 64 instead of 256).
- **Tensor Parallelism:** Increase TP size (e.g., 1 GPU -> 2 GPUs) to split compute load.

#### Challenge 5: Handling Stop Sequences in Streaming

**Scenario:** Model generates the stop token (e.g., `</s>`) but also generates garbage after it before the stream ends.
**Root Cause:** Streaming logic doesn't buffer tokens to check for multi-token stop sequences.
**Solution:**
- **Token Buffering:** Buffer last N tokens in the streaming service.
- **Pattern Matching:** Check buffer against stop sequences. If match, cut stream and finish.

### System Design Scenario: Designing an LLM Gateway

**Requirement:** Build a gateway that routes requests to Llama-3-70B, Mixtral, and GPT-4.
**Key Components:**
1. **Unified API:** OpenAI-compatible endpoint.
2. **Router:** Route based on model name.
3. **Load Balancer:** Least-outstanding-requests strategy for self-hosted models.
4. **Fallback:** If self-hosted Llama fails, fallback to external API.
5. **Rate Limiting:** Token bucket algorithm per user.
6. **Caching:** Semantic cache (Redis + Vector DB) to serve identical queries instantly.

### Summary Checklist for Production
- [ ] **Engine:** Use **vLLM** or **TensorRT-LLM**.
- [ ] **Batching:** Enable **Continuous Batching**.
- [ ] **Memory:** Enable **PagedAttention**.
- [ ] **Quantization:** Use **AWQ/GPTQ** for weights, **FP8** for KV cache if needed.
- [ ] **Monitoring:** Track **TTFT, TPOT, Queue Length, GPU Utilization**.
- [ ] **Hardware:** Use **H100/A100** for best performance.
