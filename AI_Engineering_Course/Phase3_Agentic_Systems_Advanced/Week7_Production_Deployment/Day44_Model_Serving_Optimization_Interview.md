# Day 44: Model Serving & Optimization
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Compare vLLM, TGI, and TensorRT-LLM.

**Answer:**
- **vLLM:** PagedAttention, continuous batching, highest throughput. Best for most use cases. Easy to use.
- **TGI:** HuggingFace ecosystem, tensor parallelism, quantization support. Good for HF models.
- **TensorRT-LLM:** NVIDIA-optimized, FP8 support, maximum performance on H100. Complex setup.
- **Choice:** vLLM (ease + performance) > TGI (HF integration) > TensorRT-LLM (max performance).

#### Q2: What is continuous batching and why is it better than static batching?

**Answer:**
- **Static Batching:** Wait for batch to fill, process all, wait for all to complete. Late requests wait.
- **Continuous Batching:** Add/remove requests dynamically at token level. No waiting.
- **Benefits:** Lower latency (no batch wait), higher throughput (GPU always busy), 24x improvement.
- **Implementation:** vLLM, TGI use continuous batching by default.

#### Q3: Explain speculative decoding.

**Answer:**
- **Concept:** Small model drafts K tokens, large model verifies in parallel.
- **Process:** Draft → Verify → Accept correct tokens → Repeat.
- **Speedup:** 2-3x faster (typical), K× in best case.
- **Requirements:** Small model must be similar to large (e.g., LLaMA-7B → LLaMA-70B).
- **Trade-off:** Requires running two models.

#### Q4: How does PagedAttention improve memory efficiency?

**Answer:**
- **Problem:** Traditional KV cache pre-allocates max sequence length. Wastes memory for shorter sequences.
- **PagedAttention:** Allocates KV cache in blocks (pages). Only allocates as needed.
- **Sharing:** Can reuse blocks for common prefixes (prefix caching).
- **Benefit:** 2-4x memory reduction, enables higher batch sizes, higher throughput.

#### Q5: What are the key metrics for LLM serving?

**Answer:**
- **Latency:** TTFT (time to first token), TPOT (time per output token), total latency.
- **Throughput:** Requests/second, tokens/second.
- **Efficiency:** GPU utilization, memory utilization.
- **Cost:** $ per 1K tokens.
- **Quality:** User satisfaction, hallucination rate.
- **Targets:** TTFT <500ms, p95 latency <2s for interactive.

---

### Production Challenges

#### Challenge 1: Low GPU Utilization

**Scenario:** GPU utilization is 30%. Throughput is low.
**Root Cause:** Batch size too small, not enough concurrent requests.
**Solution:**
- **Increase Batch Size:** Use continuous batching to handle more concurrent requests.
- **Reduce Model Size:** Use quantization (INT8) to fit larger batches in memory.
- **Check Bottlenecks:** CPU preprocessing, network I/O might be bottlenecks.
- **Target:** GPU utilization >80%.

#### Challenge 2: High TTFT (Time to First Token)

**Scenario:** TTFT is 2 seconds. Users complain about slow start.
**Root Cause:** Long prompts require many prefill tokens before generation starts.
**Solution:**
- **Prefix Caching:** Cache KV for common prefixes (system prompts).
- **Chunked Prefill:** Process prefill in chunks, start generation earlier.
- **Smaller Model:** Use smaller model for faster prefill.
- **Target:** TTFT <500ms.

#### Challenge 3: OOM with Large Batches

**Scenario:** Increasing batch size causes OOM (out of memory).
**Root Cause:** KV cache grows with batch size and sequence length.
**Solution:**
- **PagedAttention:** Use vLLM for efficient KV cache management.
- **Quantization:** INT8 reduces memory 2x, INT4 reduces 4x.
- **Shorter Context:** Limit max sequence length (e.g., 2048 instead of 4096).
- **Larger GPU:** Upgrade to A100 80GB or H100.

#### Challenge 4: Speculative Decoding Not Speeding Up

**Scenario:** Implemented speculative decoding but no speedup.
**Root Cause:** Draft model is too different from target model. Acceptance rate is low.
**Analysis:** If acceptance rate <50%, overhead outweighs benefit.
**Solution:**
- **Better Draft Model:** Use distilled version of target model.
- **Tune K:** Reduce number of draft tokens (K=3 instead of K=5).
- **Measure Acceptance Rate:** If <50%, disable speculative decoding.

#### Challenge 5: Inconsistent Latency

**Scenario:** p50 latency is 500ms, but p99 is 5 seconds.
**Root Cause:** Long-tail requests (very long prompts or outputs).
**Solution:**
- **Input Length Limits:** Cap max input length (e.g., 2000 tokens).
- **Output Length Limits:** Cap max output length (e.g., 512 tokens).
- **Timeout:** Set request timeout (e.g., 30 seconds).
- **Separate Queues:** Route long requests to separate queue/server.

### Summary Checklist for Production
- [ ] **Framework:** Use **vLLM** for continuous batching and PagedAttention.
- [ ] **Quantization:** Enable **INT8** for 2x memory/speed improvement.
- [ ] **Batching:** Use **continuous batching** (default in vLLM/TGI).
- [ ] **Caching:** Enable **prefix caching** for common system prompts.
- [ ] **Monitoring:** Track **TTFT, TPOT, GPU utilization, throughput**.
- [ ] **Limits:** Set **max input length** and **max output length**.
- [ ] **Target:** **TTFT <500ms**, **p95 latency <2s**, **GPU util >80%**.
