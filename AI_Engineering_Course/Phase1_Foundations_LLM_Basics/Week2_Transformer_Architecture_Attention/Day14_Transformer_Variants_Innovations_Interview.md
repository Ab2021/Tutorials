# Day 14: Transformer Variants & Innovations
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the computational complexity of standard Self-Attention, and why is it a problem?

**Answer:**
- **Complexity:** $O(N^2)$ in both time and memory, where $N$ is the sequence length.
- **Why:** We compute a similarity score between every pair of tokens.
- **Problem:** Doubling the sequence length quadruples the cost. For $N=100k$, this is computationally infeasible ($10^{10}$ operations). This limits standard Transformers to short contexts (typically 2k-4k).

#### Q2: Explain how Flash Attention works. Does it approximate attention?

**Answer:**
- **Approximation:** No. Flash Attention is **exact**. It produces bitwise identical results to standard attention.
- **Mechanism:** It optimizes **IO (Input/Output)** operations. It loads blocks of Q, K, V into fast SRAM, computes attention for that block, and writes only the result to slow HBM. It also uses **recomputation** in the backward pass to avoid storing the huge $N \times N$ attention matrix.
- **Benefit:** 2-4x speedup and linear $O(N)$ memory usage.

#### Q3: What is a Mixture of Experts (MoE) model? What are its advantages and disadvantages?

**Answer:**
- **Concept:** Replaces the dense Feed-Forward Network with multiple "expert" networks. A "router" selects a subset (e.g., 2) of experts to process each token.
- **Advantage:** Decouples model size (parameters) from compute cost (FLOPs). You can have a 100B parameter model where inference only costs as much as a 10B model.
- **Disadvantage:**
    - **VRAM:** All experts must be loaded into memory (high VRAM requirement).
    - **Training:** Harder to train (instability, load balancing issues).

#### Q4: What is the difference between MQA (Multi-Query) and GQA (Grouped-Query) Attention?

**Answer:**
- **MHA (Standard):** $H$ query heads, $H$ key/value heads. High KV cache memory.
- **MQA:** $H$ query heads, **1** key/value head. Lowest memory, but performance drops.
- **GQA:** $H$ query heads, **G** key/value heads (where $1 < G < H$).
- **Why GQA:** It is the "Goldilocks" zone. It offers near-MQA speed/memory benefits with near-MHA quality. Used in LLaMA-2/3.

#### Q5: How does Sliding Window Attention allow for infinite generation with finite memory?

**Answer:**
- **Mechanism:** The model only attends to the last $W$ tokens.
- **Inference:** We use a "Rolling Buffer" KV cache of size $W$. When a new token comes in, we overwrite the oldest token in the cache.
- **Result:** Memory usage is constant $O(W)$ regardless of how many tokens we generate (1 million+).

---

### Production Challenges

#### Challenge 1: Serving a 70B Model on Limited Hardware

**Scenario:** You need to serve LLaMA-70B but don't have 4x A100s.
**Solution:**
1.  **Quantization:** Use 4-bit quantization (GPTQ/AWQ). 70B @ 4-bit $\approx$ 35-40GB VRAM. Fits on 2x 3090/4090 or 1x A6000.
2.  **GQA:** LLaMA-70B uses GQA, which reduces KV cache size, allowing larger batch sizes on limited VRAM.
3.  **Offloading:** Use DeepSpeed-MII or Accelerate to offload layers to CPU/NVMe (slow, but works).

#### Challenge 2: MoE Load Balancing

**Scenario:** Training an MoE model, but the router sends 90% of tokens to Expert 1.
**Issue:** Expert 1 becomes a bottleneck (straggler), while other experts sit idle. Effective parameter count drops.
**Solution:**
- **Auxiliary Loss:** Add a loss term that minimizes the variance of the routing distribution (encourages uniform usage).
- **Capacity Buffer:** Set a max capacity per expert. If full, drop tokens (or route to second choice).
- **Jitter:** Add noise to router logits during training.

#### Challenge 3: Long Context Inference Latency

**Scenario:** RAG application with 32k context. Time-to-first-token (TTFT) is 5 seconds.
**Root Cause:** Processing the 32k prompt (prefill phase) is compute-bound.
**Solution:**
- **Flash Attention:** Essential.
- **Prefix Caching:** If the system prompt or documents are shared, cache the KV states.
- **Chunking:** Process prompt in chunks if OOM occurs (though TTFT won't improve much).

#### Challenge 4: Debugging Flash Attention

**Scenario:** Loss is NaN when enabling Flash Attention.
**Root Cause:**
- **Precision:** Flash Attention usually requires FP16 or BF16. FP32 might not be supported or optimized.
- **Masking:** Causal mask implementation in custom kernels can be tricky.
**Solution:**
- Verify input dtype (BF16 preferred).
- Check if sequence length is a multiple of block size (sometimes required).
- Compare output against standard attention for a small batch to verify correctness.

### Summary Checklist for Production
- [ ] **Inference:** Use **Flash Attention 2** + **GQA**.
- [ ] **Model Choice:** Prefer **MoE** (Mixtral) for high quality/cost ratio.
- [ ] **Long Context:** Ensure model uses **RoPE** with correct scaling.
- [ ] **Hardware:** H100/A100 are best for Flash Attn; older GPUs see less benefit.
