# Day 34: CUDA & Triton - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: GPU Programming, Optimization, and Compilers

### 1. What is a "Kernel"?
**Answer:**
*   A function that runs on the GPU.
*   Executed by many threads in parallel.

### 2. What is the difference between Global Memory and Shared Memory?
**Answer:**
*   **Global (HBM)**: Large (GBs), accessible by all, high latency, lower bandwidth.
*   **Shared (SRAM)**: Small (KBs), accessible only by threads in a block, extremely low latency, high bandwidth.
*   Optimization game is moving data from Global to Shared and keeping it there.

### 3. What is "Kernel Fusion"?
**Answer:**
*   Combining multiple operations (e.g., Add + ReLU + Multiply) into a single kernel.
*   Avoids round-trips to HBM (Memory Wall).
*   Major source of speedup in PyTorch 2.0.

### 4. What is "Triton"?
**Answer:**
*   A language and compiler for writing highly efficient GPU kernels.
*   Python syntax.
*   Abstracts away thread synchronization and shared memory management.
*   Block-based programming model (vs CUDA's thread-based).

### 5. What is "Memory Coalescing"?
**Answer:**
*   Pattern where adjacent threads read adjacent memory addresses.
*   Allows the GPU memory controller to combine multiple reads into a single transaction.
*   Crucial for bandwidth utilization.

### 6. What is a "Warp"?
**Answer:**
*   A group of 32 threads executed simultaneously by an SM (SIMT - Single Instruction Multiple Threads).
*   If threads in a warp diverge (if-else), execution is serialized (Warp Divergence).

### 7. What is "Occupancy"?
**Answer:**
*   Ratio of active warps to maximum supported warps on an SM.
*   High occupancy hides memory latency (while one warp waits for memory, another executes).

### 8. Why is Python slow for loops?
**Answer:**
*   Interpreter overhead.
*   Dynamic typing.
*   No vectorization.
*   Triton/CUDA bypasses Python completely during execution.

### 9. What is `torch.compile`?
**Answer:**
*   PyTorch 2.0 feature.
*   JIT compiles PyTorch code into optimized kernels (using Triton).
*   Reduces Python overhead and performs fusion.

### 10. What is "Tensor Cores"?
**Answer:**
*   Specialized hardware units on NVIDIA GPUs for Matrix Multiplication ($D = A \times B + C$).
*   Operate on $4 \times 4$ or $16 \times 16$ matrices in mixed precision.
*   Much faster than CUDA Cores (FP32).

### 11. What is "Bank Conflict" in Shared Memory?
**Answer:**
*   Shared memory is divided into banks.
*   If multiple threads in a warp access different addresses in the *same* bank, accesses are serialized.
*   Slows down access.

### 12. What is "PTX"?
**Answer:**
*   Parallel Thread Execution.
*   Intermediate assembly language for NVIDIA GPUs.
*   Triton compiles to PTX/Cubin.

### 13. What is "Autotuning"?
**Answer:**
*   Automatically searching for the best hyperparameters (Block Size, Warps, Stages) for a kernel.
*   Triton does this at runtime (caches the best config).

### 14. Why is `contiguous()` needed in PyTorch?
**Answer:**
*   Operations like `view` or `transpose` might change metadata (strides) without moving data.
*   Kernels often expect contiguous memory layout for coalescing.
*   `contiguous()` forces a copy to make layout match logical order.

### 15. What is "Grid" and "Block"?
**Answer:**
*   **Grid**: The collection of all blocks launched for a kernel.
*   **Block**: A group of threads that can share memory and synchronize.

### 16. What is "Pinned Memory" (Page-Locked)?
**Answer:**
*   Host (CPU) memory that cannot be swapped out to disk.
*   Allows faster DMA transfer to GPU.
*   `dataloader(pin_memory=True)`.

### 17. What is "Asynchronous Execution"?
**Answer:**
*   CUDA kernel launches are async. CPU returns immediately.
*   Allows CPU to prepare next batch while GPU computes.
*   `torch.cuda.synchronize()` waits for completion.

### 18. What is "FP8"?
**Answer:**
*   8-bit Floating Point.
*   Supported on H100.
*   Doubles throughput over FP16.

### 19. What is "Cutlass"?
**Answer:**
*   C++ template library for high-performance matrix multiplication.
*   Used as backend for some PyTorch ops.

### 20. Why is `atomicAdd` used?
**Answer:**
*   When multiple threads write to the same address (e.g., histogram, scatter_add).
*   Ensures correctness but serializes access (slow).
