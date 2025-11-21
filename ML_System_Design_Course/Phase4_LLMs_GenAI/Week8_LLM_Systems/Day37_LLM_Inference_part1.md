# Day 37 (Part 1): Advanced LLM Inference

> **Phase**: 6 - Deep Dive
> **Topic**: Serving LLMs
> **Focus**: PagedAttention, Speculative Decoding, and Batching
> **Reading Time**: 60 mins

---

## 1. PagedAttention (vLLM)

KV Cache fragmentation wastes 60-80% of GPU memory.

### 1.1 The Solution
*   Inspired by OS Virtual Memory.
*   Allocate KV Cache in non-contiguous "Pages" (blocks).
*   **Block Table**: Maps logical token index to physical block index.
*   **Result**: Near-zero fragmentation. Higher Batch Size. 2-4x throughput.

---

## 2. Speculative Decoding

LLMs are memory bound (reading weights). Arithmetic is cheap.

### 2.1 The Algorithm
1.  **Draft**: Small model (LLaMA-7B) generates 5 tokens quickly.
2.  **Verify**: Large model (GPT-4) checks all 5 tokens in *one* forward pass (Parallel).
3.  **Accept/Reject**: Keep matching tokens. Discard rest.
4.  **Benefit**: If Draft is accurate, we get 3-4 tokens per Large Model step.

---

## 3. Tricky Interview Questions

### Q1: Continuous Batching (Orca)?
> **Answer**:
> *   **Old**: Wait for all requests in batch to finish. (Latency = Longest request).
> *   **New**: Iteration-level scheduling.
> *   If Request A finishes, immediately insert Request C into the batch *at the next token step*.
> *   GPU is always fully utilized.

### Q2: How big is the KV Cache for LLaMA-70B?
> **Answer**:
> *   Layers $L=80$, Dim $D=8192$.
> *   KV per token = $2 \times L \times D \times 2 \text{ bytes (FP16)}$.
> *   $\approx 2.6$ MB per token.
> *   Context 4096 = 10GB per request.
> *   Batch 10 = 100GB. (Need A100 80GB x 2).

### Q3: Quantization: AWQ vs GPTQ?
> **Answer**:
> *   **GPTQ**: Quantizes weights based on Hessian (curvature). Row-wise.
> *   **AWQ**: Activation-aware. Protects "salient" weights (those with large activations) by keeping them FP16. Better accuracy.

---

## 4. Practical Edge Case: Time To First Token (TTFT) vs Tpot
*   **TTFT**: Prefill phase (Compute bound).
*   **Tpot**: Decode phase (Memory bound).
*   **Tradeoff**: Large batch size hurts TTFT (Queueing) but improves Throughput.

