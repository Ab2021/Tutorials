# Day 31 (Part 1): Advanced Transformer Internals

> **Phase**: 6 - Deep Dive
> **Topic**: The Engine of LLMs
> **Focus**: FlashAttention, RoPE, and KV Cache
> **Reading Time**: 60 mins

---

## 1. FlashAttention

Why is it 3x faster?

### 1.1 IO Awareness
*   **Bottleneck**: Reading $N \times N$ attention matrix from HBM (High Bandwidth Memory) to SRAM (Chip Cache).
*   **Trick**: Tiling. Compute attention in small blocks that fit in SRAM.
*   **Recomputation**: Don't store the huge $N \times N$ matrix for backward pass. Recompute it on the fly. (Compute is cheap, Memory IO is expensive).

---

## 2. Rotary Positional Embeddings (RoPE)

### 2.1 The Math
*   Rotate the vector pairs $(x_1, x_2)$ by an angle $m\theta$.
*   **Property**: The dot product $q^T k$ depends only on relative distance $m-n$.
*   **Extrapolation**: Generalizes better to longer sequences than absolute embeddings.

---

## 3. Tricky Interview Questions

### Q1: Why divide by $\sqrt{d_k}$ in Attention?
> **Answer**:
> *   If $q, k$ are unit variance, $q \cdot k$ has variance $d_k$.
> *   Large variance pushes Softmax into saturation (gradients $\approx 0$).
> *   Scaling brings variance back to 1.

### Q2: Multi-Query Attention (MQA) vs Multi-Head (MHA)?
> **Answer**:
> *   **MHA**: Each head has unique Q, K, V.
> *   **MQA**: All heads share same K, V. Only Q is unique.
> *   **Benefit**: Drastically reduces KV Cache size (Memory Bandwidth). Faster inference.

### Q3: Explain "Sliding Window Attention".
> **Answer**:
> *   Mistral / Longformer.
> *   Attend only to local window $W$.
> *   Complexity $O(N \times W)$ instead of $O(N^2)$.

---

## 4. Practical Edge Case: FP8 Training
*   **H100**: Supports FP8.
*   **Risk**: Precision loss.
*   **Fix**: Scaling factors for tensors to keep them in representable range.

