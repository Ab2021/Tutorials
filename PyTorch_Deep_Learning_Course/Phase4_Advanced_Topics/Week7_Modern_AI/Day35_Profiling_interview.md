# Day 35: Profiling & Optimization - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: Performance Tuning, Hardware, and Debugging

### 1. What is the "GIL" and how does it affect training?
**Answer:**
*   Global Interpreter Lock. Prevents multiple Python threads from running simultaneously.
*   Can cause CPU bottlenecks in Dataloading.
*   Solution: Use `multiprocessing` (num_workers > 0) in Dataloader.

### 2. What is "Pin Memory"?
**Answer:**
*   Allocates host memory in a page-locked region.
*   Allows asynchronous DMA transfer to GPU.
*   Avoids CPU overhead of copying data from paged to pinned memory before transfer.

### 3. Why is `torch.cuda.synchronize()` needed for timing?
**Answer:**
*   CUDA calls are asynchronous.
*   `start = time(); model(x); end = time()` only measures launch time (microseconds).
*   Must sync before `end` to measure actual execution time.

### 4. What is "Channels Last" memory format?
**Answer:**
*   NHWC layout.
*   Optimized for Tensor Cores and Conv2d (which operate on channels).
*   Provides better memory locality for dense math.

### 5. What is "Mixed Precision" (AMP)?
**Answer:**
*   Using FP16 for math (matmul) and FP32 for accumulation/updates.
*   Tensor Cores require FP16/BF16.
*   Reduces memory bandwidth and VRAM usage.

### 6. What is "Gradient Checkpointing"?
**Answer:**
*   Technique to save VRAM.
*   Don't save intermediate activations during forward pass.
*   Recompute them during backward pass using the inputs.
*   Cost: 30% more compute. Benefit: 3-4x less memory.

### 7. What is "Kernel Launch Overhead"?
**Answer:**
*   Time taken by CPU to tell GPU to run a function (~10us).
*   If kernels are tiny (e.g., adding small tensors), overhead > execution.
*   Solution: CUDA Graphs or Fusion.

### 8. What is "CUDA Graphs"?
**Answer:**
*   Recording a sequence of kernel launches and replaying them as a single graph.
*   Eliminates CPU launch overhead.
*   Great for static input shapes.

### 9. How do you identify a Data Loading bottleneck?
**Answer:**
*   GPU utilization fluctuates or is low.
*   Profiler shows gaps between kernels.
*   `DataLoader` iterator time is high.

### 10. What is "Occupancy" in GPU?
**Answer:**
*   How many warps are active on an SM.
*   Low occupancy means GPU resources are wasted (waiting for memory).

### 11. What is "TensorBoard"?
**Answer:**
*   Visualization tool.
*   Tracks Loss, Metrics, Histograms, and Profiler traces.

### 12. What is "Quantization Aware Training" (QAT)?
**Answer:**
*   Simulating quantization effects (rounding errors) during training.
*   Allows the model to adapt to low precision.
*   Better accuracy than Post-Training Quantization.

### 13. What is "Sparsity"?
**Answer:**
*   Having many zeros in weights/activations.
*   NVIDIA Ampere supports 2:4 sparsity (2 zeros in every 4 elements) for 2x speedup.

### 14. What is "Throughput" vs "Latency"?
**Answer:**
*   **Throughput**: Samples per second (Batch size matters). Important for Training.
*   **Latency**: Time per sample (Batch size 1). Important for Real-time Inference.

### 15. What is "TorchScript"?
**Answer:**
*   Intermediate representation of PyTorch models.
*   Can be run in C++ (no Python dependency).
*   `torch.jit.trace` or `torch.jit.script`.

### 16. What is "Triton" used for in optimization?
**Answer:**
*   Writing custom fused kernels that are faster than standard PyTorch calls.
*   Used by `torch.compile`.

### 17. What is "Volatile GPU Util"?
**Answer:**
*   `nvidia-smi` shows instantaneous utilization.
*   If it jumps 0% -> 100% -> 0%, you are CPU bound.
*   Ideally should stay near 100%.

### 18. What is "Memory Fragmentation"?
**Answer:**
*   PyTorch Caching Allocator manages GPU memory.
*   Frequent alloc/free can cause fragmentation (OOM even if free memory exists).
*   `empty_cache()` clears it but is slow.

### 19. What is "TF32"?
**Answer:**
*   TensorFloat-32.
*   19-bit format (10-bit mantissa like FP16, 8-bit exponent like FP32).
*   Default on Ampere GPUs. Fast matmul with FP32-like range.

### 20. How to optimize for Inference?
**Answer:**
*   Fuse layers (Conv+BN).
*   Quantize (INT8).
*   Use TensorRT or ONNX Runtime.
*   Batch requests.
