# Day 60: GPU Optimization & CUDA Kernels
## Core Concepts & Theory

### The Hardware Beneath

**Understanding GPUs:**
- **SMs (Streaming Multiprocessors):** The core compute units.
- **HBM (High Bandwidth Memory):** Global memory (slow, large).
- **SRAM (Shared Memory/L1 Cache):** On-chip memory (fast, small).
- **Tensor Cores:** Specialized hardware for matrix multiplication (GEMM).

**The Bottleneck:**
- **Memory Bandwidth Bound:** Most LLM operations (decoding, attention) are bound by how fast we can move data from HBM to SRAM, not by compute speed.
- **Compute Bound:** Large batch training or prefill phase can be compute bound.

### 1. Kernel Fusion

**Concept:** Combine multiple operations into a single CUDA kernel.
**Why:**
- Reduces HBM access.
- Instead of `Read A -> Op1 -> Write B -> Read B -> Op2 -> Write C`, do `Read A -> Op1 -> Op2 -> Write C`.
- **Example:** Fused Softmax, Fused LayerNorm, Fused Attention.

### 2. FlashAttention (The Game Changer)

**Problem:** Standard Attention computes `S = QK^T`, `P = softmax(S)`, `O = PV`.
- `S` and `P` are $N \times N$ matrices. For long context, they are huge and require massive HBM reads/writes.

**Solution (IO-Awareness):**
- **Tiling:** Compute attention in small blocks that fit in SRAM.
- **Recomputation:** Don't store huge intermediate matrices ($N \times N$) in HBM. Recompute them during backward pass.
- **Result:** Linear memory complexity $O(N)$ instead of quadratic $O(N^2)$. 2-4x speedup.

### 3. Triton Language

**What is it?**
- A language and compiler for writing highly efficient GPU kernels.
- Python-like syntax, compiles to PTX (CUDA assembly).
- **Benefit:** Easier than writing raw CUDA C++, but often matches or beats cuBLAS performance.
- **Key Abstraction:** Block-based programming (program for a block of data, not a single thread).

### 4. CUDA Streams & Graphs

**CUDA Streams:**
- Parallel execution queues on GPU.
- Overlap compute (Kernel 1) with memory transfer (Host-to-Device) or other compute (Kernel 2).

**CUDA Graphs:**
- Capture a sequence of kernels as a graph.
- **Benefit:** Reduces CPU launch overhead. Instead of launching 100 small kernels individually (high CPU overhead), launch the whole graph once.

### 5. Matrix Multiplication Optimization (GEMM)

**Basics:**
- $C = A \times B$
- **Naive:** 3 nested loops.
- **Optimized:** Tiling, vectorization, prefetching.

**Tensor Cores:**
- Perform $D = A \times B + C$ on $4 \times 4$ or $16 \times 16$ matrices in one instruction.
- Requires specific data layouts and precision (FP16/BF16/INT8).

### 6. Memory Coalescing

**Concept:**
- Access global memory in contiguous chunks.
- If threads in a warp access random addresses, memory bandwidth is wasted.
- **Goal:** Ensure threads 0-31 access addresses $X, X+1, ..., X+31$.

### 7. Mixed Precision Training (AMP)

**FP16 / BF16:**
- **Speed:** Tensor cores run faster in half-precision.
- **Memory:** Half the memory footprint.
- **Stability:** BF16 (Brain Floating Point) has same range as FP32, better stability than FP16.

### 8. Profiling Tools

**Nsight Systems:**
- Timeline view of CPU and GPU activities.
- Identify gaps (idle GPU) and overlapping.

**Nsight Compute:**
- Kernel-level profiling.
- Analyze occupancy, memory throughput, cache hits.

### 9. Optimization Checklist

1.  **Fuse Element-wise Ops:** (Add, Mul, Relu) -> Single Kernel.
2.  **Use FlashAttention:** Always.
3.  **Reduce Precision:** FP16/BF16/FP8.
4.  **Maximize Batch Size:** Hide memory latency with compute.
5.  **Profile:** Don't guess bottlenecks.

### 10. Future: FP8 and Hardware Transformers

**H100 Transformer Engine:**
- Automatically manages FP8 precision.
- 2x speedup over FP16.
- Dynamic scaling to preserve accuracy.

### Summary

**Optimization Hierarchy:**
1.  **Algorithm:** FlashAttention (reduce complexity).
2.  **Kernel:** Fusion (reduce memory IO).
3.  **Hardware:** Tensor Cores (specialized compute).
4.  **System:** CUDA Graphs (reduce overhead).

**Best Practices:**
- Use **Triton** for custom kernels.
- Use **PyTorch 2.0 compile** (Inductor) for automatic fusion.
- Profile with **Nsight Systems**.

### Next Steps
In the Deep Dive, we will write a custom Softmax kernel in Triton and compare it with PyTorch, and explore FlashAttention implementation details.
