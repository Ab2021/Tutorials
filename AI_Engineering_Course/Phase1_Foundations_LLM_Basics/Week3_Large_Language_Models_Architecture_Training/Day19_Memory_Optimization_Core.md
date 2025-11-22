# Day 19: Memory Optimization
## Core Concepts & Theory

### The VRAM Bottleneck

Training LLMs is memory-bound, not compute-bound.
**Where does memory go?**
1.  **Model Weights:** 2 bytes per param (FP16).
2.  **Gradients:** 2 bytes per param.
3.  **Optimizer States:** 12 bytes per param (Adam).
4.  **Activations:** $B \times L \times H$. Scales with sequence length and batch size.

For a 1B param model:
- Static Memory (Weights+Grads+Opt): $16$ GB.
- Dynamic Memory (Activations): Can be $100$ GB+ depending on batch size.

### 1. Gradient Checkpointing (Activation Recomputation)

**Concept:** Trade compute for memory.
- **Standard:** Store all intermediate activations during forward pass to use in backward pass.
- **Checkpointing:** Store only a few "checkpoints" (e.g., input to each Transformer block).
- **Backward Pass:** When gradients are needed for a block, **re-run the forward pass** for that block using the checkpoint.

**Impact:**
- Memory: $O(N) \to O(\sqrt{N})$.
- Compute: Increases by ~33% (one extra forward pass).
- **Result:** Allows training with 3-4x larger batch sizes.

### 2. ZeRO (Zero Redundancy Optimizer)

**Concept:** In Data Parallelism, every GPU holds a full copy of everything. This is wasteful.
**ZeRO Stages:**
- **Stage 1:** Shard Optimizer States. (4x memory reduction).
- **Stage 2:** Shard Gradients. (8x memory reduction).
- **Stage 3:** Shard Model Parameters. (Linear reduction with $N_{gpus}$).

**ZeRO-Offload:**
- Offload optimizer states and update computation to **CPU RAM**.
- Allows training 13B models on a single GPU.

### 3. Mixed Precision Training (AMP)

**Concept:** Use lower precision for storage and arithmetic where possible.
- **FP32 (Master Weights):** Kept for stability.
- **FP16/BF16 (Activations/Gradients):** Used for forward/backward pass.
- **Loss Scaling:** Required for FP16 to prevent underflow. Not needed for BF16.

**Impact:**
- Memory: Halves activation/gradient memory.
- Speed: Uses Tensor Cores (up to 8x faster).

### 4. 8-bit Optimizers (bitsandbytes)

**Concept:** Store optimizer states (momentum, variance) in 8-bit instead of 32-bit.
- **Standard Adam:** 8 bytes per param (2 states * 4 bytes).
- **8-bit Adam:** 2 bytes per param.
- **Impact:** Saves 6 bytes per param. For 1B params, saves 6GB VRAM.
- **Accuracy:** Negligible drop in performance.

### Summary of Optimizations

| Technique | Saves | Cost |
| :--- | :--- | :--- |
| **Checkpointing** | Activation Memory | +33% Compute |
| **ZeRO-3** | Weight/Grad/Opt Memory | +Communication |
| **BF16** | All Memory | None (if HW supports) |
| **8-bit Adam** | Optimizer Memory | Slight CPU overhead |

### Next Steps
In the Deep Dive, we will analyze the communication overhead of ZeRO and implement Gradient Checkpointing from scratch.
