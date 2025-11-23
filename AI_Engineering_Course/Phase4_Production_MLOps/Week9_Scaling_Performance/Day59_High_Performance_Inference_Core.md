# Day 59: High-Performance Inference Serving
## Core Concepts & Theory

### The Inference Challenge

**Problem:** LLMs are memory-bound and compute-intensive.
- **Memory Bound:** Loading weights and KV cache takes more time than computation.
- **Compute Intensive:** Matrix multiplications for 70B+ parameters.
- **Latency Sensitivity:** Users expect real-time responses (<50ms per token).

**Key Metrics:**
- **Time to First Token (TTFT):** Latency to start generation.
- **Time Per Output Token (TPOT):** Latency between tokens.
- **Throughput:** Tokens generated per second across all users.

### 1. Continuous Batching (Orbital Batching)

**Traditional Batching:**
- Wait for batch to fill.
- All sequences must finish before next batch starts.
- **Inefficiency:** Short sequences wait for long sequences (padding waste).

**Continuous Batching:**
- Iteration-level scheduling.
- New requests join running batch immediately.
- Finished sequences leave batch immediately.
- **Benefit:** 10-20x throughput improvement.

### 2. PagedAttention & KV Cache Management

**KV Cache Problem:**
- Key/Value states for attention grow linearly with sequence length.
- **Fragmentation:** Contiguous memory allocation leads to fragmentation.
- **Waste:** Over-allocation for max context length.

**PagedAttention (vLLM):**
- Inspired by OS virtual memory paging.
- Split KV cache into fixed-size blocks (pages).
- Blocks can be non-contiguous in physical memory.
- **Benefit:** Near-zero memory waste, larger batch sizes.

### 3. Speculative Decoding

**Concept:** Use small "draft" model to predict tokens, large "target" model to verify.

**Process:**
1. **Draft:** Small model generates K tokens (cheap).
2. **Verify:** Large model checks K tokens in parallel (1 forward pass).
3. **Accept/Reject:** Keep correct tokens, discard rest.

**Benefit:**
- 2-3x speedup for single requests.
- Exploits memory bandwidth (large model reads weights once for K tokens).

### 4. Tensor Parallelism for Inference

**Concept:** Split model across multiple GPUs to reduce latency.

**Why:**
- Model doesn't fit in one GPU memory.
- Memory bandwidth aggregation (use bandwidth of N GPUs).

**Implementation:**
- **Column Parallel:** Split weight matrices by column.
- **Row Parallel:** Split weight matrices by row.
- **All-Reduce:** Synchronize results after each layer.

### 5. Serving Frameworks

**vLLM:**
- **Key Feature:** PagedAttention, Continuous Batching.
- **Best For:** High throughput serving.
- **Engine:** Python + Custom CUDA kernels.

**Text Generation Inference (TGI) - HuggingFace:**
- **Key Feature:** Rust-based, Tensor Parallelism, Flash Attention.
- **Best For:** Production deployment, wide model support.

**TensorRT-LLM (NVIDIA):**
- **Key Feature:** Compiled kernels, FP8 support, In-flight batching.
- **Best For:** Maximum performance on NVIDIA GPUs.

**Triton Inference Server:**
- **Key Feature:** Multi-model serving, ensemble support.
- **Best For:** Enterprise infrastructure.

### 6. Attention Optimizations

**FlashAttention:**
- **Concept:** Tiling to keep attention matrix in SRAM.
- **Benefit:** Linear memory complexity, 2-4x speedup.

**Multi-Query Attention (MQA):**
- **Concept:** Share Key/Value heads across all Query heads.
- **Benefit:** Reduce KV cache size by factor of H (heads).

**Grouped-Query Attention (GQA):**
- **Concept:** Middle ground between MHA and MQA.
- **Benefit:** Good quality (like MHA) with low memory (like MQA).

### 7. Quantization for Inference

**Weight-Only Quantization:**
- **GPTQ / AWQ:** Quantize weights to 4-bit, keep activations FP16.
- **Benefit:** Reduce memory bandwidth requirement.

**KV Cache Quantization:**
- **FP8 / INT8 KV Cache:** Compress cache to fit longer context.
- **Benefit:** 2-4x longer context length.

### 8. System Architecture

**Router / Load Balancer:**
- Distributes requests to model replicas.
- Handles queueing and prioritization.

**Model Engine:**
- Manages GPU memory and scheduling.
- Executes forward passes.

**Tokenizer Service:**
- Detokenization (streaming).
- Tokenization (input processing).

### 9. Production Considerations

**Stop Sequences:**
- Handle custom stop tokens reliably.

**Streaming:**
- Server-Sent Events (SSE) for real-time output.

**Timeout Handling:**
- Graceful degradation under load.

**Concurrency Limits:**
- Max active requests to prevent OOM.

### 10. Future Trends

**Prefix Caching:**
- Cache common system prompts (RadixAttention).
- **Benefit:** Zero-latency for shared context.

**Disaggregated Serving:**
- Separate Prefill (compute-bound) and Decode (memory-bound) phases.
- Different hardware for each phase.

### Summary

**Optimization Hierarchy:**
1. **Batching:** Continuous batching (essential for throughput).
2. **Memory:** PagedAttention (essential for efficiency).
3. **Compute:** FlashAttention, Tensor Parallelism.
4. **Latency:** Speculative Decoding.
5. **Hardware:** Quantization (FP8/INT4).

**Best Practices:**
- Use **vLLM** or **TensorRT-LLM** for production.
- Enable **continuous batching**.
- Use **FP16** or **BF16** by default.
- Monitor **KV cache utilization**.

### Next Steps
In the Deep Dive, we will implement a simplified continuous batching scheduler and explore vLLM's PagedAttention mechanism.
