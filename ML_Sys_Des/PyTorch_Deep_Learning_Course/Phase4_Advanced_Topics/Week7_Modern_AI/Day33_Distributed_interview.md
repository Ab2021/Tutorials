# Day 33: Distributed Training - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: HPC, Parallelism, and Synchronization

### 1. What is the difference between `DataParallel` (DP) and `DistributedDataParallel` (DDP)?
**Answer:**
*   **DP**: Single Process, Multi-Thread. Parameter Server model (GPU 0 gathers all gradients). Slow due to GIL and unbalanced load.
*   **DDP**: Multi-Process. Each GPU has its own process. Ring All-Reduce. No GIL bottleneck. Recommended.

### 2. What is "All-Reduce"?
**Answer:**
*   A collective operation where every node starts with a value, and ends with the sum (or average) of all values from all nodes.
*   Used to synchronize gradients in DDP.

### 3. Explain "ZeRO" optimization.
**Answer:**
*   Zero Redundancy Optimizer.
*   Removes memory redundancy in DDP by sharding Optimizer States, Gradients, and Parameters across GPUs.
*   Allows fitting models larger than single GPU memory.

### 4. What is "Model Parallelism"?
**Answer:**
*   Splitting the model itself across GPUs.
*   **Tensor Parallelism (Megatron)**: Split matrix multiplications (Row/Col wise).
*   **Pipeline Parallelism**: Put Layer 1-10 on GPU 0, Layer 11-20 on GPU 1.

### 5. What is the "Linear Scaling Rule"?
**Answer:**
*   When increasing batch size by $k$, we should roughly increase Learning Rate by $k$.
*   Because larger batches provide cleaner gradients (less noise), allowing larger steps.

### 6. What is "NCCL"?
**Answer:**
*   NVIDIA Collective Communications Library.
*   Backend for PyTorch Distributed.
*   Optimized for GPU-to-GPU communication (NVLink, PCIe, InfiniBand).

### 7. What is "Gradient Accumulation"?
**Answer:**
*   Simulating a larger batch size by running multiple forward/backward passes before one optimizer step.
*   Useful when VRAM is limited.

### 8. What is "FSDP"?
**Answer:**
*   Fully Sharded Data Parallel. PyTorch's native implementation of ZeRO-3.
*   Shards params, grads, and optimizer states.
*   Fetches params on-the-fly (All-Gather) during forward/backward.

### 9. What is "Rank" and "World Size"?
**Answer:**
*   **Rank**: ID of the current process (0 to N-1). Rank 0 is usually the master.
*   **World Size**: Total number of processes (GPUs).

### 10. Why is `DistributedSampler` needed?
**Answer:**
*   It ensures each GPU gets a *different* slice of the dataset.
*   Without it, all GPUs would process the same data (redundant).

### 11. What is "SyncBatchNorm"?
**Answer:**
*   Standard BatchNorm calculates statistics per GPU. If batch size per GPU is small (e.g., 2), stats are noisy.
*   SyncBatchNorm synchronizes stats across all GPUs to get accurate Mean/Var.
*   Slower but necessary for Detection/Segmentation.

### 12. What is "NVLink"?
**Answer:**
*   High-speed interconnect between GPUs on the same node.
*   Much faster than PCIe. Crucial for Tensor Parallelism.

### 13. What is "Checkpointing" in Distributed Training?
**Answer:**
*   Only Rank 0 should save the checkpoint (to avoid race conditions/corruption).
*   In FSDP, we need to gather all shards to save a consolidated checkpoint (or save sharded checkpoints).

### 14. What is "Pipeline Bubble"?
**Answer:**
*   In Pipeline Parallelism, GPU 1 waits for GPU 0 to finish. GPU 0 waits for GPU 1 during backward.
*   Idle time is the "bubble".
*   Reduced by Micro-batching (GPipe).

### 15. What is "Gloo" vs "NCCL"?
**Answer:**
*   **NCCL**: GPU-to-GPU. Fast. Standard for CUDA.
*   **Gloo**: CPU-to-CPU. Fallback or used for CPU distributed training.

### 16. What is "Master Address/Port"?
**Answer:**
*   Environment variables (`MASTER_ADDR`, `MASTER_PORT`) needed to coordinate processes across multiple nodes.

### 17. How does DDP handle buffers (like BatchNorm running stats)?
**Answer:**
*   Buffers are broadcast from Rank 0 to all other ranks at the start of each forward pass to ensure consistency.

### 18. What is "CPU Offloading"?
**Answer:**
*   Moving optimizer states or parameters to CPU RAM to save GPU VRAM.
*   Supported by DeepSpeed/FSDP. Slower due to PCIe transfer.

### 19. What is "Megatron-LM"?
**Answer:**
*   NVIDIA's library for training massive LLMs.
*   Pioneered Tensor Parallelism.

### 20. What is the bottleneck in Distributed Training?
**Answer:**
*   **Communication Overhead**.
*   If the network is slow (Ethernet vs InfiniBand), GPUs spend time waiting for gradients to sync (All-Reduce) instead of computing.
