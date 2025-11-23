# Day 44: Model Serving & Optimization
## Core Concepts & Theory

### Serving Framework Landscape

**Key Frameworks:**
- **vLLM:** PagedAttention, continuous batching, highest throughput.
- **TGI (Text Generation Inference):** HuggingFace's production server.
- **TensorRT-LLM:** NVIDIA's optimized serving for maximum performance.
- **Ray Serve:** Distributed serving, multi-model deployment.
- **Triton Inference Server:** Multi-framework support, dynamic batching.

### 1. vLLM Deep Dive

**Architecture:**
```
Request → Scheduler → KV Cache Manager → GPU Execution → Response
```

**Key Features:**

**PagedAttention:**
- Allocates KV cache in blocks (pages).
- **Memory Efficiency:** 2-4x better than traditional.
- **Sharing:** Reuse KV cache for common prefixes.

**Continuous Batching:**
- Add/remove requests dynamically.
- **Benefit:** No waiting for batch to fill.
- **Throughput:** 24x higher than naive batching.

**Parallel Sampling:**
- Generate multiple outputs per request.
- **Use Case:** Best-of-N sampling.

### 2. TGI (Text Generation Inference)

**Features:**

**Tensor Parallelism:**
- Split model across multiple GPUs.
- **Example:** 70B model across 4x A100 (40GB each).

**Flash Attention:**
- Optimized attention implementation.
- **Speedup:** 2-3x faster than standard attention.

**Quantization Support:**
- **bitsandbytes:** INT8, INT4.
- **GPTQ:** 4-bit quantization.
- **AWQ:** Activation-aware quantization.

**Streaming:**
- Stream tokens as generated.
- **Benefit:** Lower perceived latency.

### 3. TensorRT-LLM

**NVIDIA-Specific Optimizations:**

**FP8 Quantization:**
- 8-bit floating point (FP8).
- **Benefit:** 2x speedup on H100 GPUs.
- **Accuracy:** Minimal loss compared to FP16.

**Kernel Fusion:**
- Fuse multiple operations into single kernel.
- **Example:** LayerNorm + Attention → Single kernel.

**Multi-GPU Inference:**
- Tensor parallelism, pipeline parallelism.
- **Scaling:** Linear speedup up to 8 GPUs.

### 4. Batching Strategies

**Static Batching:**
```python
# Wait for batch to fill
batch = []
while len(batch) < batch_size:
    batch.append(wait_for_request())
process(batch)
```
- **Pros:** Simple, predictable.
- **Cons:** High latency for late arrivals.

**Dynamic Batching:**
```python
# Combine requests within time window
batch = []
deadline = time.now() + max_wait_time
while time.now() < deadline:
    if request_available():
        batch.append(get_request())
process(batch)
```
- **Pros:** Lower latency than static.
- **Cons:** Still waits for timeout.

**Continuous Batching (Iteration-Level):**
```python
# Process at token level
active_requests = []
while True:
    # Add new requests
    active_requests.extend(get_new_requests())
    
    # Generate next token for all
    next_tokens = model.generate_next_token(active_requests)
    
    # Remove completed
    active_requests = [r for r in active_requests if not r.done()]
```
- **Pros:** Lowest latency, highest throughput.
- **Cons:** Complex implementation.

### 5. KV Cache Optimization

**Problem:**
- KV cache size: `batch_size × seq_len × num_layers × hidden_dim × 2 (K+V)`
- **Example:** Batch=32, seq=2048, layers=32, hidden=4096
  - Size: 32 × 2048 × 32 × 4096 × 2 × 2 bytes = 32 GB

**Solutions:**

**PagedAttention (vLLM):**
- Allocate in blocks, share across sequences.
- **Reduction:** 2-4x memory savings.

**Multi-Query Attention (MQA):**
- Share K/V across all heads.
- **Reduction:** 8x smaller KV cache (for 8 heads).

**Grouped-Query Attention (GQA):**
- Share K/V across groups of heads.
- **Balance:** Between MQA and standard attention.

### 6. Quantization Techniques

**Post-Training Quantization (PTQ):**

**INT8:**
```python
# Quantize weights and activations to 8-bit
quantized_model = quantize_int8(model)
# Memory: 2x reduction
# Speed: 1.5-2x faster
# Accuracy: <1% loss
```

**INT4 (GPTQ):**
```python
# 4-bit quantization with calibration
quantized_model = gptq_quantize(model, calibration_data)
# Memory: 4x reduction
# Speed: 2-3x faster
# Accuracy: 1-2% loss
```

**AWQ (Activation-Aware Weight Quantization):**
- Protect important weights from quantization.
- **Accuracy:** Better than GPTQ at same bit-width.

### 7. Speculative Decoding

**Concept:**
```
Small Model (Draft) → Large Model (Verify)
```

**Process:**
1. Small model generates K tokens (draft).
2. Large model verifies in parallel.
3. Accept correct tokens, reject rest.
4. Repeat.

**Speedup:**
- **Best Case:** K× faster (all tokens accepted).
- **Typical:** 2-3× faster.

**Requirements:**
- Small model must be similar to large model.
- **Example:** LLaMA-7B drafts for LLaMA-70B.

### 8. Load Balancing

**Round Robin:**
- Distribute requests evenly across servers.
- **Simple but ignores server load.**

**Least Connections:**
- Send to server with fewest active requests.
- **Better for variable request times.**

**Weighted Round Robin:**
- Weight by server capacity (GPU type).
- **Example:** H100 gets 2x traffic vs A100.

**Consistent Hashing:**
- Route same user to same server.
- **Benefit:** KV cache reuse for multi-turn conversations.

### 9. Performance Metrics

**Throughput:**
- **Tokens/second:** Total tokens generated per second.
- **Requests/second:** Total requests processed per second.

**Latency:**
- **TTFT (Time to First Token):** Time until first token.
- **TPOT (Time Per Output Token):** Average time per token.
- **Total Latency:** TTFT + (num_tokens × TPOT).

**Efficiency:**
- **GPU Utilization:** % of time GPU is busy.
- **Memory Utilization:** % of GPU memory used.
- **Cost per 1K tokens:** $ / 1000 tokens.

### Real-World Benchmarks

**vLLM vs HuggingFace Transformers:**
- **Throughput:** 24x higher (vLLM).
- **Latency:** 3x lower (vLLM).

**TensorRT-LLM vs vLLM:**
- **Throughput:** 1.5-2x higher (TensorRT-LLM on H100).
- **Complexity:** Higher (TensorRT-LLM).

**INT8 vs FP16:**
- **Memory:** 2x reduction.
- **Speed:** 1.5-2x faster.
- **Accuracy:** <1% loss.

### Summary

**Framework Selection:**
- **vLLM:** Best for most use cases (high throughput, easy to use).
- **TGI:** Good for HuggingFace ecosystem integration.
- **TensorRT-LLM:** Maximum performance on NVIDIA GPUs (complex setup).

**Optimization Priority:**
1. **Batching:** Continuous batching (vLLM, TGI).
2. **Quantization:** INT8 for 2x memory/speed.
3. **KV Cache:** PagedAttention for memory efficiency.
4. **Speculative Decoding:** 2-3x speedup for compatible models.

### Next Steps
In the Deep Dive, we will implement production serving with vLLM, TGI, and TensorRT-LLM, including benchmarking and optimization.
