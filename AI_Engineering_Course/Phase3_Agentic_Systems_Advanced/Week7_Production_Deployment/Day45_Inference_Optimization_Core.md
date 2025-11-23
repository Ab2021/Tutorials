# Day 45: Inference Optimization Techniques
## Core Concepts & Theory

### Optimization Landscape

**Goals:**
- **Reduce Latency:** Faster response times.
- **Increase Throughput:** More requests per second.
- **Lower Memory:** Fit larger models or batches.
- **Reduce Cost:** Cheaper per request.

### 1. Quantization Techniques

**Weight-Only Quantization:**
- Quantize weights, keep activations in FP16.
- **INT8:** 2x memory reduction, 1.5x speedup.
- **INT4:** 4x memory reduction, 2x speedup.
- **Use Case:** Memory-bound models.

**Weight + Activation Quantization:**
- Quantize both weights and activations.
- **Speedup:** 2-3x faster than weight-only.
- **Accuracy:** Slightly lower than weight-only.

**Methods:**

**GPTQ (Post-Training Quantization):**
```
1. Calibrate on small dataset
2. Quantize weights layer-by-layer
3. Minimize reconstruction error
```
- **Accuracy:** Good (1-2% loss at INT4).
- **Speed:** Fast quantization (<1 hour for 70B).

**AWQ (Activation-Aware Weight Quantization):**
```
1. Identify important weights (high activation magnitude)
2. Protect important weights from quantization
3. Quantize remaining weights aggressively
```
- **Accuracy:** Better than GPTQ at same bit-width.
- **Trade-off:** Slightly slower quantization.

**SmoothQuant:**
```
1. Migrate difficulty from activations to weights
2. Apply per-channel scaling
3. Quantize both to INT8
```
- **Benefit:** Enables INT8 activation quantization.
- **Speedup:** 2-3x on hardware with INT8 support.

### 2. Kernel Optimization

**Flash Attention:**
- Optimized attention implementation.
- **Memory:** O(N) instead of O(N²).
- **Speed:** 2-3x faster than standard attention.
- **Method:** Tiling, recomputation instead of storing intermediate results.

**Kernel Fusion:**
- Combine multiple operations into single kernel.
- **Example:** LayerNorm + Linear → Single kernel.
- **Benefit:** Reduce memory bandwidth, faster execution.

**Custom CUDA Kernels:**
- Hand-optimized kernels for critical operations.
- **Example:** Fused MLP (GeLU + Linear).
- **Speedup:** 1.5-2x for specific operations.

### 3. Model Architecture Optimizations

**Multi-Query Attention (MQA):**
```
Standard: Q, K, V all have num_heads
MQA: Q has num_heads, K/V shared across heads
```
- **KV Cache:** 8x smaller (for 8 heads).
- **Speedup:** 1.5-2x faster inference.
- **Accuracy:** Minimal loss (<1%).

**Grouped-Query Attention (GQA):**
```
Balance between MQA and standard
Q has num_heads, K/V shared within groups
```
- **Example:** 32 Q heads, 8 KV heads (4 groups).
- **KV Cache:** 4x smaller.
- **Accuracy:** Better than MQA, close to standard.

**Sliding Window Attention:**
- Attend only to last W tokens.
- **Memory:** O(W) instead of O(N).
- **Use Case:** Long sequences (>8K tokens).

### 4. Caching Strategies

**Response Caching:**
```python
cache = {}
def generate(prompt):
    if prompt in cache:
        return cache[prompt]
    response = model.generate(prompt)
    cache[prompt] = response
    return response
```
- **Hit Rate:** 20-50% for common queries.
- **Benefit:** Instant response, zero cost.

**Prefix Caching (KV Cache Reuse):**
```
System Prompt: "You are a helpful assistant..."
User Query 1: "What is AI?"
User Query 2: "What is ML?"

Reuse KV cache for system prompt across queries
```
- **Speedup:** 2-5x for long system prompts.
- **Implementation:** vLLM, TGI support this.

**Semantic Caching:**
```
Query 1: "What is machine learning?"
Query 2: "Explain ML"
→ Same semantic meaning, return cached response
```
- **Method:** Embed queries, check similarity.
- **Hit Rate:** Higher than exact match caching.

### 5. Batching Optimizations

**Continuous Batching:**
- Add/remove requests at token level.
- **Benefit:** No waiting, 24x throughput improvement.

**Priority Batching:**
```
High Priority: Interactive users (low latency)
Low Priority: Batch processing (high throughput)

Process high priority first
```

**Adaptive Batching:**
```
if load < threshold:
    batch_size = small  # Low latency
else:
    batch_size = large  # High throughput
```

### 6. Memory Optimization

**Gradient Checkpointing (Training):**
- Recompute activations instead of storing.
- **Memory:** 2-4x reduction.
- **Speed:** 20-30% slower.

**KV Cache Quantization:**
- Quantize KV cache to INT8 or INT4.
- **Memory:** 2-4x smaller KV cache.
- **Accuracy:** Minimal loss (<1%).

**Offloading:**
- Move inactive layers to CPU/disk.
- **Use Case:** Very large models (>100B parameters).
- **Trade-off:** Slower but enables larger models.

### 7. Parallelism Strategies

**Tensor Parallelism:**
- Split layers across GPUs.
- **Use Case:** Model doesn't fit on single GPU.
- **Example:** 70B model across 4x A100.

**Pipeline Parallelism:**
- Split model into stages, each on different GPU.
- **Use Case:** Very deep models.
- **Challenge:** Bubble overhead.

**Data Parallelism:**
- Replicate model, split data.
- **Use Case:** High throughput requirements.
- **Scaling:** Linear up to network bandwidth limit.

### 8. Hardware-Specific Optimizations

**FP8 (H100):**
- 8-bit floating point.
- **Speedup:** 2x on H100 vs FP16.
- **Accuracy:** Better than INT8.

**Tensor Cores:**
- Specialized hardware for matrix multiplication.
- **Speedup:** 10-20x for supported operations.
- **Requirement:** Use supported data types (FP16, BF16, INT8).

**NVLink:**
- High-bandwidth GPU interconnect.
- **Bandwidth:** 600 GB/s (vs 64 GB/s PCIe).
- **Benefit:** Faster multi-GPU communication.

### Real-World Impact

**Quantization (INT8):**
- **Memory:** 50% reduction.
- **Speed:** 1.5-2x faster.
- **Cost:** 50% cheaper.

**Flash Attention:**
- **Memory:** 75% reduction for long sequences.
- **Speed:** 2-3x faster.

**Continuous Batching:**
- **Throughput:** 24x improvement.
- **Latency:** 50% reduction.

**Prefix Caching:**
- **Speed:** 2-5x for cached prefixes.
- **Cost:** Near-zero for cache hits.

### Summary

**Optimization Priority:**
1. **Continuous Batching:** 24x throughput (vLLM, TGI).
2. **Quantization (INT8):** 2x memory/speed.
3. **Flash Attention:** 2-3x speed for long sequences.
4. **Prefix Caching:** 2-5x for common prompts.
5. **MQA/GQA:** 1.5-2x speed, 4-8x smaller KV cache.

### Next Steps
In the Deep Dive, we will implement quantization with GPTQ/AWQ, Flash Attention, and prefix caching with complete code examples.
