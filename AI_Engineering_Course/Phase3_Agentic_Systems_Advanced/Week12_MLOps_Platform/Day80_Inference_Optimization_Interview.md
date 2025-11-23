# Day 45: Inference Optimization Techniques
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Compare GPTQ and AWQ quantization methods.

**Answer:**
- **GPTQ:** Layer-by-layer quantization minimizing reconstruction error. Fast quantization (<1 hour for 70B). Good accuracy (1-2% loss at INT4).
- **AWQ:** Activation-aware. Protects important weights (high activation magnitude) from quantization. Better accuracy than GPTQ at same bit-width. Slightly slower quantization.
- **Choice:** AWQ for best accuracy, GPTQ for speed.

#### Q2: What is Flash Attention and how does it work?

**Answer:**
- **Problem:** Standard attention uses O(N²) memory for storing attention matrix.
- **Flash Attention:** Processes attention in blocks, recomputes instead of storing. O(N) memory.
- **Method:** Tiling, online softmax, fused kernels.
- **Benefits:** 2-3x faster, 75% memory reduction for long sequences.
- **Use Case:** Long context (>2K tokens).

#### Q3: Explain prefix caching and when to use it.

**Answer:**
- **Concept:** Cache KV cache for common prefixes (e.g., system prompts). Reuse across requests.
- **Example:** System prompt "You are a helpful assistant..." is same for all users. Cache its KV once, reuse.
- **Speedup:** 2-5x for long system prompts (skip prefill for cached portion).
- **Implementation:** vLLM, TGI support this natively.
- **When:** Long system prompts (>500 tokens), high request volume.

#### Q4: What is Multi-Query Attention (MQA)?

**Answer:**
- **Standard Attention:** Q, K, V all have num_heads dimensions.
- **MQA:** Q has num_heads, K/V shared across all heads (single head).
- **KV Cache:** 8x smaller (for 8 heads).
- **Speedup:** 1.5-2x faster inference.
- **Accuracy:** Minimal loss (<1%).
- **Models:** PaLM, Falcon use MQA.

#### Q5: How do you choose between different optimization techniques?

**Answer:**
**Priority Order:**
1. **Continuous Batching:** 24x throughput (free with vLLM/TGI).
2. **Quantization (INT8):** 2x memory/speed (minimal accuracy loss).
3. **Flash Attention:** 2-3x speed for long sequences.
4. **Prefix Caching:** 2-5x for common prompts.
5. **MQA/GQA:** 1.5-2x speed (requires model architecture change).

**Decision:** Start with 1-4 (easy wins), consider 5 if training new model.

---

### Production Challenges

#### Challenge 1: Quantization Accuracy Loss

**Scenario:** Quantized model to INT4. Accuracy dropped 5% (unacceptable).
**Root Cause:** Some layers are sensitive to quantization.
**Solution:**
- **Mixed Precision:** Keep sensitive layers in FP16, quantize rest to INT4.
- **AWQ:** Use activation-aware quantization (better accuracy).
- **Higher Bits:** Use INT8 instead of INT4 (2x memory, better accuracy).
- **Calibration:** Use better calibration dataset (more representative).

#### Challenge 2: Flash Attention Not Speeding Up

**Scenario:** Implemented Flash Attention but no speedup.
**Root Cause:** Short sequences (<512 tokens). Flash Attention overhead outweighs benefit.
**Analysis:** Flash Attention benefits increase with sequence length.
**Solution:**
- **Use for Long Sequences:** Only enable for sequences >1K tokens.
- **Standard Attention for Short:** Use standard attention for <1K tokens.
- **Benchmark:** Measure actual speedup on your workload.

#### Challenge 3: Prefix Cache Misses

**Scenario:** Enabled prefix caching but hit rate is 5% (expected 50%).
**Root Cause:** System prompts vary slightly across requests.
**Example:** "You are a helpful assistant." vs "You are a helpful AI assistant."
**Solution:**
- **Normalize Prompts:** Standardize system prompts.
- **Fuzzy Matching:** Use semantic similarity instead of exact match.
- **Longer Prefixes:** Cache longer prefixes (first 1000 tokens instead of 500).

#### Challenge 4: MQA Accuracy Drop

**Scenario:** Converted model to MQA. Accuracy dropped 3%.
**Root Cause:** MQA shares K/V across heads, losing some expressiveness.
**Solution:**
- **GQA (Grouped-Query Attention):** Use groups instead of full sharing. 32 Q heads, 8 KV heads (4 groups). Better accuracy than MQA.
- **Fine-tune:** Fine-tune model after MQA conversion to recover accuracy.
- **Accept Trade-off:** 3% accuracy loss might be acceptable for 2x speedup.

#### Challenge 5: Memory Still High After Quantization

**Scenario:** Quantized to INT8 but memory usage still high.
**Root Cause:** KV cache is not quantized, still in FP16.
**Analysis:** For long sequences, KV cache dominates memory.
**Solution:**
- **KV Cache Quantization:** Quantize KV cache to INT8 (2x reduction).
- **PagedAttention:** Use vLLM for efficient KV cache management.
- **Shorter Context:** Limit max sequence length.

### Summary Checklist for Production
- [ ] **Quantization:** Use **INT8** for 2x memory/speed (AWQ for best accuracy).
- [ ] **Flash Attention:** Enable for **sequences >1K tokens**.
- [ ] **Prefix Caching:** Cache **system prompts** (standardize for high hit rate).
- [ ] **KV Cache Quantization:** Quantize to **INT8** for 2x memory reduction.
- [ ] **Continuous Batching:** Use **vLLM/TGI** (24x throughput).
- [ ] **Benchmark:** Measure **actual speedup** on your workload.
- [ ] **Monitor:** Track **accuracy, latency, memory** after each optimization.
