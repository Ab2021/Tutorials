# Day 169: The Mammoth in the Room: LLM Infrastructure
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 25: Large Language Model Infrastructure

---

> **🎯 Focus Area:** "Can I run Llama-70B on my laptop?" No. "Can I run it on a T4?" Maybe. "Can I run it on an A100?" Yes. **LLM Infrastructure** is strictly bound by VRAM Capacity and Memory Bandwidth.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Calculate** the VRAM requirements for any LLM (e.g., 70B Params @ FP16).
2.  **Estimate** the KV Cache size for long-context inference (32k/128k context).
3.  **Select** the correct GPU (L4 vs A10G vs A100 vs H100) based on Bandwidth economics.
4.  **Simulate** Bandwidth-Bound Inference using Python scripts.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine (CPU is fine for calculator).

### Software Environment
- `pip install torch accelerate`.

---

## 📖 Theoretical Foundation

### 1. The VRAM Formula
*   **Model Weights:** $Parameters \times BytesPerParam$.
    *   FP32 (4 bytes): 7B Params = 28GB.
    *   FP16/BF16 (2 bytes): 7B Params = 14GB.
    *   INT8 (1 byte): 7B Params = 7GB.
    *   INT4 (0.5 byte): 7B Params = 3.5GB.
*   **KV Cache:** Intermediate activation states stored to speed up generation. Grows linearly with Context Length and Batch Size.
*   **Overhead:** CUDA Kernels + PyTorch Runtime (~1-2GB).

### 2. Compute vs Bandwidth Bound
*   **Training:** Compute Bound (FLOPs). You need H100s.
*   **Inference (batch=1):** Bandwidth Bound (Memory/sec). The GPU spends 90% of time *waiting* for weights to load from HBM to SRAM. An H100 is barely faster than an A100 for batch=1.
*   **Inference (batch=128):** Compute Bound.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The VRAM Calculator

Don't guess. Calculate.

#### 📁 `src/vram_calculator.py`
```python
def calculate_vram(params_billion, precision="fp16", context_len=4096, batch_size=1):
    # 1. Weights
    bytes_per_param = {
        "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5
    }[precision]
    
    weight_vram = params_billion * 1e9 * bytes_per_param
    
    # 2. KV Cache (Simplified approximation for Llama-2 70B architecture)
    # KV Cache = 2 * n_layers * n_heads * head_dim * precision * context_len * batch_size
    # Assumptions for 70B: Layers=80, Heads=64, Dim=128
    n_layers = 80
    n_heads = 64
    head_dim = 128
    
    kv_cache_vram = 2 * n_layers * n_heads * head_dim * bytes_per_param * context_len * batch_size
    
    total_gb = (weight_vram + kv_cache_vram) / (1024**3)
    
    print(f"--- Configuration: {params_billion}B Params, {precision.upper()}, Context {context_len}, Batch {batch_size} ---")
    print(f"Model Weights: {weight_vram / (1024**3):.2f} GB")
    print(f"KV Cache:      {kv_cache_vram / (1024**3):.2f} GB")
    print(f"Total VRAM:    {total_gb:.2f} GB")
    
    return total_gb

# Examples
# Llama-3-8B on T4 (16GB) in FP16?
calculate_vram(8, "fp16") 
# Result: ~15GB. Tight fit.

# Llama-3-70B on A100 (80GB) in INT4?
calculate_vram(70, "int4")
# Result: ~35GB + KV. Easily fits.

# Llama-3-70B on A100 in FP16?
calculate_vram(70, "fp16")
# Result: ~130GB. Requires 2x A100 (Sharding).
```

### 👨‍💻 Infrastructure: Which GPU?

| GPU | VRAM | Bandwidth | Price/Hr (Spot) | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **L4** | 24GB | 300 GB/s | $0.20 | 7B Models (FP16), 13B (INT8) |
| **A10G**| 24GB | 600 GB/s | $0.80 | Low Latency Inference |
| **A100**| 80GB | 2000 GB/s| $1.50 | 70B Models, Training |
| **H100**| 80GB | 3350 GB/s| $4.00 | Massive Training |

### 👨‍💻 Core Implementation: Memory Bandwidth Simulator

Prove that Bandwidth checks bottlenecks inference.

#### 📁 `src/bandwidth_sim.py`
```python
import time
import torch

# Simulate reading 70B param model (140GB) from VRAM
model_size_bytes = 140 * (1024**3)
device_bandwidth_gbps = 2000 # A100

def simulate_inference_time(tokens_to_gen):
    # For every token generated, we must read the ENTIRE model from VRAM 
    # (unless using advanced caching optimizations, but fundamentally true for decoder)
    
    time_per_token = model_size_bytes / (device_bandwidth_gbps * 1e9)
    total_time = time_per_token * tokens_to_gen
    
    print(f"Generating {tokens_to_gen} tokens on A100:")
    print(f"Time per token: {time_per_token * 1000:.2f} ms")
    print(f"Total Time:     {total_time:.2f} s")
    print(f"Tokens/Sec:     {1/time_per_token:.2f}")

simulate_inference_time(100)
# Result: ~14 tokens/sec. This is the physical limit of the hardware for batch=1.
```

---

## 🔬 Lab Exercise: "OOM Hunter"

### Task
Predict OOM.
1.  Model: 7B. Setup: T4 GPU (16GB). Precision: FP16.
2.  Math: 14GB Weights. 2GB Overhead. Available for KV: ~0GB.
3.  **Scenario:** User sends prompt with 500 tokens.
4.  KV Cache grows.
5.  **Result:** `CUDA out of memory`.
6.  **Fix:** Use INT8 Quantization (weights=7GB). Now you have 9GB for KV Cache (Supports ~100k context).

---

## 📖 Advanced Theory: FlashAttention
The bottleneck is $O(N^2)$ memory reads in Attention.
**FlashAttention:** Tiling technique. Keeps tiles in SRAM (L1 Cache) to minimize HBM reads.
**Impact:** 2-4x speedup, 10-20x less memory for long context.
Always ensure `torch.nn.functional.scaled_dot_product_attention` is using FlashAttention backend.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Quantization is Free Lunch:** Going from FP16 to INT4 often has negligible accuracy loss for 70B+ models, but doubles speed and halves VRAM.
2.  **KV Cache is Huge:** For 128k context, the KV cache can be bigger than the model itself. Use **PagedAttention** (vLLM) to manage this.
3.  **Sharding:** If model > GPU VRAM, you must shard (Tensor Parallelism). This requires fast interconnect (NVLink) between GPUs.

### API Summary
```python
# To check VRAM usage in PyTorch
torch.cuda.memory_allocated()
torch.cuda.max_memory_allocated()
```

---

**Day 169 Complete** ✅

*Next: Day 170 - Distributing Giants - Tensor Parallelism & Pipeline Parallelism.*
