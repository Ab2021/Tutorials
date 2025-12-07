# Day 7: Week 1 Review & Project - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🏗️ Week 1 Capstone Project: Optimized Matrix Multiplication

### Objective
Implement and optimize GEMM (General Matrix Multiply) using all techniques learned this week.

---

## 🔬 Part 1: Naive Implementation

```python
from numba import cuda
import numpy as np
import numba
import time

@cuda.jit
def matmul_naive(A, B, C):
    """Naive matrix multiplication - one thread per output element."""
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
```

---

## 🔬 Part 2: Tiled Implementation with Shared Memory

```python
TILE_SIZE = 32

@cuda.jit
def matmul_tiled(A, B, C):
    """Tiled matrix multiplication using shared memory."""
    # Shared memory tiles
    sA = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=numba.float32)
    sB = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=numba.float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # Output element this thread computes
    row = by * TILE_SIZE + ty
    col = bx * TILE_SIZE + tx
    
    tmp = 0.0
    
    # Loop over tiles
    for t in range((A.shape[1] + TILE_SIZE - 1) // TILE_SIZE):
        # Load tiles to shared memory
        if row < A.shape[0] and t * TILE_SIZE + tx < A.shape[1]:
            sA[ty, tx] = A[row, t * TILE_SIZE + tx]
        else:
            sA[ty, tx] = 0.0
            
        if t * TILE_SIZE + ty < B.shape[0] and col < B.shape[1]:
            sB[ty, tx] = B[t * TILE_SIZE + ty, col]
        else:
            sB[ty, tx] = 0.0
        
        cuda.syncthreads()
        
        # Compute partial dot product
        for k in range(TILE_SIZE):
            tmp += sA[ty, k] * sB[k, tx]
        
        cuda.syncthreads()
    
    # Write result
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp
```

---

## 🔬 Part 3: Benchmarking Suite

```python
def benchmark_matmul():
    """Compare all implementations."""
    sizes = [256, 512, 1024, 2048]
    
    print("Matrix Multiplication Benchmark")
    print("=" * 60)
    print(f"{'Size':>6} | {'Naive (ms)':>12} | {'Tiled (ms)':>12} | {'cuBLAS (ms)':>12} | {'Speedup':>8}")
    print("-" * 60)
    
    for N in sizes:
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        
        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        d_C = cuda.device_array((N, N), dtype=np.float32)
        
        threads = (TILE_SIZE, TILE_SIZE)
        blocks = ((N + TILE_SIZE - 1) // TILE_SIZE,
                  (N + TILE_SIZE - 1) // TILE_SIZE)
        
        # Warm up
        matmul_naive[blocks, threads](d_A, d_B, d_C)
        matmul_tiled[blocks, threads](d_A, d_B, d_C)
        cuda.synchronize()
        
        iterations = 10
        
        # Naive
        start = time.perf_counter()
        for _ in range(iterations):
            matmul_naive[blocks, threads](d_A, d_B, d_C)
        cuda.synchronize()
        naive_time = (time.perf_counter() - start) / iterations * 1000
        
        # Tiled
        start = time.perf_counter()
        for _ in range(iterations):
            matmul_tiled[blocks, threads](d_A, d_B, d_C)
        cuda.synchronize()
        tiled_time = (time.perf_counter() - start) / iterations * 1000
        
        # cuBLAS (via PyTorch)
        import torch
        t_A = torch.from_numpy(A).cuda()
        t_B = torch.from_numpy(B).cuda()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            t_C = torch.mm(t_A, t_B)
        torch.cuda.synchronize()
        cublas_time = (time.perf_counter() - start) / iterations * 1000
        
        speedup = naive_time / tiled_time
        
        print(f"{N:>6} | {naive_time:>12.3f} | {tiled_time:>12.3f} | {cublas_time:>12.3f} | {speedup:>7.2f}x")
    
    print("=" * 60)

if __name__ == "__main__":
    benchmark_matmul()
```

---

## 📊 Expected Results

```
Matrix Multiplication Benchmark
============================================================
  Size |  Naive (ms) |  Tiled (ms) | cuBLAS (ms) | Speedup
------------------------------------------------------------
   256 |        2.45 |        0.89 |        0.12 |    2.75x
   512 |       18.32 |        4.21 |        0.45 |    4.35x
  1024 |      142.56 |       28.76 |        2.89 |    4.96x
  2048 |     1156.23 |      198.45 |       18.34 |    5.83x
============================================================
```

---

## ✅ Success Criteria

1. ☐ Tiled implementation is at least 3x faster than naive
2. ☐ Results match NumPy/cuBLAS reference (within floating point tolerance)
3. ☐ No memory errors (verified with compute-sanitizer)
4. ☐ Profiled with Nsight to identify remaining bottlenecks

---

## 🎯 Challenge Extensions

1. **Beginner:** Add FP16 support for Tensor Cores
2. **Intermediate:** Implement register tiling for better occupancy
3. **Advanced:** Match cuBLAS performance within 80%

---

## 📚 Week 1 Summary

| Day | Topic | Key Takeaway |
|-----|-------|--------------|
| 1 | GPU Architecture | SMs, Warps, Memory Hierarchy |
| 2 | CUDA Model | Threads, Blocks, Grids |
| 3 | Memory Management | Pinned Memory, Caching Allocator |
| 4 | Synchronization | syncthreads, Branch Divergence |
| 5 | Debugging | Profiler, compute-sanitizer |
| 6 | Shared Memory | Tiling, Bank Conflicts |
| 7 | Project | Optimized GEMM |
