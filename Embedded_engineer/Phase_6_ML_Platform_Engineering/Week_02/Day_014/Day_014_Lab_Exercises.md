# Day 14: Week 2 Review & Project - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🏗️ Week 2 Capstone Project: CUDA Convolution Implementation

### Objective
Implement and optimize 2D convolution using all Week 2 techniques.

---

## 🔬 Part 1: Naive Convolution

```python
from numba import cuda
import numpy as np
import numba
import time

@cuda.jit
def conv2d_naive(input_img, kernel, output):
    """Naive 2D convolution."""
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if row >= output.shape[0] or col >= output.shape[1]:
        return
    
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    result = 0.0
    for i in range(kh):
        for j in range(kw):
            in_row = row + i
            in_col = col + j
            if 0 <= in_row < input_img.shape[0] and 0 <= in_col < input_img.shape[1]:
                result += input_img[in_row, in_col] * kernel[i, j]
    
    output[row, col] = result
```

---

## 🔬 Part 2: Optimized with Shared Memory

```python
TILE_SIZE = 16
MAX_KERNEL = 7

@cuda.jit
def conv2d_shared(input_img, kernel, output, kh, kw):
    """Optimized convolution using shared memory."""
    # Shared memory for input tile (+halo)
    shared = cuda.shared.array((TILE_SIZE + MAX_KERNEL - 1, 
                                 TILE_SIZE + MAX_KERNEL - 1), 
                                dtype=numba.float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # Output coordinates
    out_row = by * TILE_SIZE + ty
    out_col = bx * TILE_SIZE + tx
    
    # Input coordinates (with halo)
    pad = kh // 2
    in_row = out_row
    in_col = out_col
    
    # Load tile with halo to shared memory
    for i in range((TILE_SIZE + kh - 1 + TILE_SIZE - 1) // TILE_SIZE):
        for j in range((TILE_SIZE + kw - 1 + TILE_SIZE - 1) // TILE_SIZE):
            si = ty + i * TILE_SIZE
            sj = tx + j * TILE_SIZE
            ii = in_row + i * TILE_SIZE
            ij = in_col + j * TILE_SIZE
            
            if si < TILE_SIZE + kh - 1 and sj < TILE_SIZE + kw - 1:
                if 0 <= ii < input_img.shape[0] and 0 <= ij < input_img.shape[1]:
                    shared[si, sj] = input_img[ii, ij]
                else:
                    shared[si, sj] = 0.0
    
    cuda.syncthreads()
    
    # Compute convolution
    if out_row < output.shape[0] and out_col < output.shape[1]:
        result = 0.0
        for i in range(kh):
            for j in range(kw):
                result += shared[ty + i, tx + j] * kernel[i, j]
        output[out_row, out_col] = result
```

---

## 🔬 Part 3: Full Benchmark Suite

```python
import torch
import torch.nn.functional as F

def benchmark_conv2d():
    """Compare all convolution implementations."""
    
    H, W = 1024, 1024
    kernel_size = 5
    
    # Prepare data
    input_np = np.random.rand(H, W).astype(np.float32)
    kernel_np = np.random.rand(kernel_size, kernel_size).astype(np.float32)
    
    print("2D Convolution Benchmark")
    print("=" * 60)
    print(f"Image: {H}x{W}, Kernel: {kernel_size}x{kernel_size}")
    print("-" * 60)
    
    results = {}
    
    # 1. NumPy (CPU baseline)
    from scipy import signal
    start = time.perf_counter()
    for _ in range(10):
        out_cpu = signal.convolve2d(input_np, kernel_np, mode='same')
    cpu_time = (time.perf_counter() - start) / 10 * 1000
    results['NumPy (CPU)'] = cpu_time
    
    # 2. PyTorch (cuDNN)
    x_torch = torch.from_numpy(input_np).cuda().unsqueeze(0).unsqueeze(0)
    k_torch = torch.from_numpy(kernel_np).cuda().unsqueeze(0).unsqueeze(0)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        out_torch = F.conv2d(x_torch, k_torch, padding=kernel_size//2)
    torch.cuda.synchronize()
    cudnn_time = (time.perf_counter() - start) / 100 * 1000
    results['cuDNN'] = cudnn_time
    
    # 3. Our naive kernel
    d_input = cuda.to_device(input_np)
    d_kernel = cuda.to_device(kernel_np)
    d_output = cuda.device_array((H, W), dtype=np.float32)
    
    threads = (16, 16)
    blocks = ((W + 15) // 16, (H + 15) // 16)
    
    # Warm up
    conv2d_naive[blocks, threads](d_input, d_kernel, d_output)
    cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        conv2d_naive[blocks, threads](d_input, d_kernel, d_output)
    cuda.synchronize()
    naive_time = (time.perf_counter() - start) / 100 * 1000
    results['Custom Naive'] = naive_time
    
    # Print results
    print(f"{'Implementation':<20} {'Time (ms)':<12} {'Speedup'}")
    print("-" * 60)
    baseline = results['NumPy (CPU)']
    for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
        speedup = baseline / time_ms
        print(f"{name:<20} {time_ms:<12.3f} {speedup:.1f}x")

if __name__ == "__main__":
    benchmark_conv2d()
```

---

## 📊 Expected Results

```
2D Convolution Benchmark
============================================================
Image: 1024x1024, Kernel: 5x5
------------------------------------------------------------
Implementation       Time (ms)    Speedup
------------------------------------------------------------
cuDNN                0.089        168.5x
Custom Naive         2.345        6.4x
NumPy (CPU)          15.012       1.0x
```

---

## ✅ Success Criteria

1. ☐ Naive kernel produces correct results
2. ☐ Shared memory version is 2x+ faster than naive
3. ☐ All results verified against PyTorch reference
4. ☐ Multi-stream overlap implemented

---

## 📚 Week 2 Summary

| Day | Topic | Key Takeaway |
|-----|-------|--------------|
| 8 | Streams | Overlap compute with memory transfers |
| 9 | Events | Accurate kernel timing |
| 10 | Atomics | Thread-safe reductions |
| 11 | Dynamic Parallelism | GPU-launched kernels |
| 12 | Textures | 2D spatial caching |
| 13 | Multi-GPU | Device management, P2P |
| 14 | Project | Optimized Conv2D |
