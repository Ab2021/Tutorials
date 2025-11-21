# Day 37: LLM Inference Optimization

> **Phase**: 4 - LLMs & GenAI
> **Week**: 8 - LLM Systems
> **Focus**: Making LLMs Fast & Cheap
> **Reading Time**: 50 mins

---

## 1. The Bottleneck: Memory Bandwidth

LLM inference is **Memory Bound**, not Compute Bound.
*   **Auto-regressive**: To generate 1 token, we must move all 70B weights from HBM (High Bandwidth Memory) to the Compute Units.
*   **Math**: A 70B model (FP16) is 140GB. A100 bandwidth is 2TB/s. Max speed = 2000/140 = 14 tokens/sec (for batch size 1).

### 1.1 KV Cache
*   **Problem**: At step $T$, we recompute attention for tokens $1 \dots T-1$. Redundant.
*   **Solution**: Cache the Key and Value matrices for past tokens in GPU memory.
*   **Cost**: KV Cache grows linearly with sequence length. Eats VRAM.

---

## 2. Advanced Optimizations

### 2.1 PagedAttention (vLLM)
*   **Problem**: KV Cache suffers from fragmentation. We reserve contiguous memory for max length (4k), but user might only type 10 words. Wasted VRAM.
*   **Solution**: Inspired by OS Virtual Memory. Break KV cache into non-contiguous "pages". Allocate pages on demand.
*   **Result**: 2-4x higher throughput (larger batch sizes).

### 2.2 Continuous Batching
*   **Problem**: Static batching waits for all requests to finish. If Request A is short and B is long, A waits for B.
*   **Solution**: Eject finished requests immediately and insert new ones into the batch at the token level.

### 2.3 Speculative Decoding
*   **Idea**: Use a tiny "Draft Model" (Llama-7B) to generate 5 tokens quickly.
*   **Verify**: Run the big "Target Model" (Llama-70B) once to verify all 5 tokens in parallel.
*   **Result**: 2-3x speedup if draft is accurate.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Time To First Token (TTFT) vs. Total Time
**Scenario**: Streaming application.
**Optimization**:
*   **TTFT**: Optimize for compute (Prefill phase is compute bound).
*   **TPOT (Time Per Output Token)**: Optimize for memory bandwidth (Decode phase).

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Why is LLM inference Memory Bound?**
> **Answer**: The arithmetic intensity (FLOPS / Bytes) of decoding is low. We perform a matrix-vector multiplication (load huge matrix, multiply by small vector). The GPU spends most of its time waiting for data to arrive from memory.

**Q2: How does Quantization (INT8/INT4) improve speed?**
> **Answer**:
> 1.  **Memory Bandwidth**: Moving 4-bit weights is 4x faster than 16-bit.
> 2.  **Capacity**: Fits larger models/batches in VRAM.
> 3.  **Compute**: INT8 tensor cores are faster (though this is secondary in memory-bound regimes).

**Q3: Explain KV Cache memory usage.**
> **Answer**: Size = $2 \times \text{Batch} \times \text{Length} \times \text{Layers} \times \text{Heads} \times \text{HeadDim} \times \text{Precision}$.
> For Llama-2-70B, a 4k context request takes ~2GB just for KV cache.

---

## 5. Further Reading
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://vllm.ai/)
- [Speculative Decoding Explained](https://arxiv.org/abs/2211.17192)
