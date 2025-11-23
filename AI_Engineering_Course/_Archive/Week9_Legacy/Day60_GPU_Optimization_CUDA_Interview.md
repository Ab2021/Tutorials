# Day 60: GPU Optimization & CUDA Kernels
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between Memory Bandwidth and Compute Throughput? Which one limits LLM inference?

**Answer:**
- **Memory Bandwidth:** The speed at which data can be moved between HBM (global memory) and the compute units (SMs). Measured in GB/s (e.g., A100 has ~2000 GB/s).
- **Compute Throughput:** The speed of calculation. Measured in TFLOPS (e.g., A100 has ~312 TFLOPS for FP16).
- **LLM Inference:**
  - **Decoding Phase (Token generation):** Is **Memory Bandwidth Bound**. We load the entire model weights just to generate 1 token. Arithmetic intensity is low.
  - **Prefill Phase (Prompt processing):** Can be **Compute Bound** if batch size and sequence length are large, as we process many tokens in parallel.

#### Q2: Explain how FlashAttention speeds up training/inference.

**Answer:**
- **Standard Attention:** Computes $N \times N$ attention matrix, writes it to HBM, then reads it back for softmax and multiplication. This is slow (quadratic memory IO).
- **FlashAttention:** Uses **Tiling** to compute attention block-by-block in SRAM (fast on-chip cache). It recomputes intermediate values during backward pass instead of storing them.
- **Result:** Reduces memory access from $O(N^2)$ to $O(N)$. Speedup comes from IO reduction, not FLOP reduction.

#### Q3: What is Kernel Fusion and why is it important?

**Answer:**
- **Concept:** Combining multiple operations (e.g., MatMul + Bias + ReLU) into a single GPU kernel launch.
- **Benefit:**
  - **Reduces Memory IO:** Intermediate results (e.g., result of MatMul) stay in registers/SRAM and are immediately used for ReLU, instead of being written to HBM and read back.
  - **Reduces Launch Overhead:** 1 kernel launch instead of 3.
- **Example:** Fused LayerNorm, Fused Softmax.

#### Q4: What are Tensor Cores?

**Answer:**
- Specialized hardware units on NVIDIA GPUs designed specifically for matrix multiplication (GEMM).
- They perform the operation $D = A \times B + C$ on small matrices (e.g., $4 \times 4$) in a single clock cycle.
- They are much faster than standard CUDA cores (FP32 cores) but require specific precision (FP16/BF16/INT8) and data layout.

#### Q5: What is the purpose of CUDA Streams?

**Answer:**
- CUDA Streams allow concurrent execution of operations on the GPU.
- **Default Stream:** Serial execution.
- **Multiple Streams:** Can overlap **Compute** (Kernel A) with **Memory Transfer** (Host-to-Device) or **Compute** (Kernel B).
- **Use Case:** Loading data for the next batch while processing the current batch.

---

### Production Challenges

#### Challenge 1: Low GPU Utilization (30%)

**Scenario:** You deployed a model, but `nvidia-smi` shows only 30% utilization. Throughput is low.
**Root Cause:**
- **Small Batch Size:** Not enough work to saturate the massive parallel compute of the GPU.
- **CPU Bottleneck:** Python code / Data preprocessing is too slow, GPU is waiting for data.
- **Kernel Launch Overhead:** Too many tiny kernels.
**Solution:**
- **Increase Batch Size:** Use continuous batching.
- **Profile:** Use Nsight Systems to see if GPU is idle.
- **CUDA Graphs:** Reduce launch overhead.
- **Fast Tokenizer:** Use Rust-based tokenizer (HuggingFace Tokenizers).

#### Challenge 2: "Illegal Memory Access" Error

**Scenario:** Custom kernel crashes with illegal memory access.
**Root Cause:**
- Reading/Writing out of bounds.
- Wrong pointer arithmetic.
- **Bank Conflict:** (Performance issue, not crash, but related to memory).
**Solution:**
- **Boundary Checks:** Ensure `idx < N` in your kernel.
- **Compute Sanitizer:** Run with `compute-sanitizer python script.py` to pinpoint the error.
- **Check Strides:** Ensure you are handling non-contiguous tensors correctly.

#### Challenge 3: Training is Slow despite A100s

**Scenario:** Training speed is much lower than reported benchmarks.
**Root Cause:**
- **TF32 Disabled:** Not using TensorFloat-32 on Ampere GPUs.
- **Data Loading:** DataLoader num_workers is 0 or too low.
- **Communication:** DDP communication overhead (slow interconnect).
**Solution:**
- **Enable TF32:** `torch.backends.cuda.matmul.allow_tf32 = True`.
- **Optimize Dataloader:** Increase workers, pin memory.
- **Mixed Precision:** Ensure AMP is enabled.

#### Challenge 4: FlashAttention OOM

**Scenario:** Even with FlashAttention, you get OOM on very long sequences (e.g., 100k).
**Root Cause:**
- FlashAttention reduces quadratic memory, but linear memory (KV cache, activations) still grows.
- **Block Size:** Maybe block size is too large for SRAM? (Unlikely for standard FA).
**Solution:**
- **Gradient Checkpointing:** Trade compute for memory.
- **Activation Offloading:** Offload to CPU.
- **Sequence Parallelism:** Split sequence across GPUs.

#### Challenge 5: Triton Kernel Compilation Lag

**Scenario:** First inference request takes 5 seconds because Triton is compiling kernels.
**Root Cause:** JIT compilation.
**Solution:**
- **AOT Compilation:** Pre-compile kernels during build time.
- **Cache:** Ensure Triton cache is persistent and reused.
- **Warmup:** Run dummy data during server startup to trigger compilation.

### Summary Checklist for Production
- [ ] **Precision:** Use **BF16** (if supported) or **FP16**.
- [ ] **Attention:** Use **FlashAttention-2**.
- [ ] **Compilation:** Use **torch.compile()** (PyTorch 2.0).
- [ ] **Profiling:** Periodically check with **Nsight Systems**.
- [ ] **Utilization:** Aim for **>80% GPU utilization** via batching.
- [ ] **Overhead:** Use **CUDA Graphs** for small-batch inference.
