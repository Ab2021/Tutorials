# Day 7: Week 1 Review & Project - CUDA Basics
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 1: GPU Architecture & CUDA Foundations

---

> **🎯 Focus Area:** Consolidate Week 1 learning with an optimized matrix multiplication project.

---

## 🎯 Learning Objectives
1. **Review** all Week 1 concepts
2. **Implement** optimized matrix multiplication
3. **Compare** against cuBLAS baseline
4. **Profile** and document performance

---

## 📚 Week 1 Recap

### Key Concepts Covered

| Day | Topic | Key Takeaway |
|-----|-------|--------------|
| 1 | GPU Architecture | SM, Warps, Memory Hierarchy |
| 2 | CUDA Programming Model | Kernels, Grids, Blocks, Thread Indexing |
| 3 | Memory Management | cudaMalloc, Pinned, Unified Memory |
| 4 | Synchronization | __syncthreads, Parallel Reduction |
| 5 | Error Handling | CUDA_CHECK, compute-sanitizer |
| 6 | Shared Memory | Bank Conflicts, Tiling, Matrix Transpose |

---

## 🏗️ Week 1 Project: Optimized Matrix Multiplication

### Project Requirements
1. Implement naive matrix multiplication
2. Implement tiled matrix multiplication with shared memory
3. Benchmark against cuBLAS SGEMM
4. Profile with Nsight Compute
5. Document speedups and optimization insights

### Implementation

```python
#!/usr/bin/env python3
"""
Week 1 Project: Optimized Matrix Multiplication
Phase 6: AI/ML Platform Engineering with GPU Programming
"""

import cupy as cp
import numpy as np
import time

# Naive matrix multiplication
naive_matmul = cp.RawKernel(r'''
extern "C" __global__
void naiveMatmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
''', 'naiveMatmul')

# Tiled matrix multiplication with shared memory
tiled_matmul = cp.RawKernel(r'''
#define TILE_SIZE 32

extern "C" __global__
void tiledMatmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
''', 'tiledMatmul')


def benchmark_matmul(M, N, K, iterations=10):
    """Benchmark matrix multiplication implementations"""
    
    print(f"\n{'='*60}")
    print(f"Matrix Multiplication Benchmark: ({M}x{K}) @ ({K}x{N})")
    print(f"{'='*60}")
    
    # Initialize matrices
    A = cp.random.randn(M, K, dtype=cp.float32)
    B = cp.random.randn(K, N, dtype=cp.float32)
    C_naive = cp.zeros((M, N), dtype=cp.float32)
    C_tiled = cp.zeros((M, N), dtype=cp.float32)
    
    block = (32, 32)
    grid = ((N + 31) // 32, (M + 31) // 32)
    
    results = {}
    
    # 1. Naive implementation
    print("\n1. Naive Implementation:")
    for _ in range(3):  # Warmup
        naive_matmul(grid, block, (A, B, C_naive, M, N, K))
    cp.cuda.Stream.null.synchronize()
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iterations):
        naive_matmul(grid, block, (A, B, C_naive, M, N, K))
    end.record()
    end.synchronize()
    
    naive_time = cp.cuda.get_elapsed_time(start, end) / iterations
    naive_gflops = (2 * M * N * K) / (naive_time / 1000) / 1e9
    results['naive'] = {'time': naive_time, 'gflops': naive_gflops}
    print(f"   Time: {naive_time:.2f} ms")
    print(f"   Performance: {naive_gflops:.1f} GFLOPS")
    
    # 2. Tiled implementation
    print("\n2. Tiled (Shared Memory) Implementation:")
    for _ in range(3):
        tiled_matmul(grid, block, (A, B, C_tiled, M, N, K))
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for _ in range(iterations):
        tiled_matmul(grid, block, (A, B, C_tiled, M, N, K))
    end.record()
    end.synchronize()
    
    tiled_time = cp.cuda.get_elapsed_time(start, end) / iterations
    tiled_gflops = (2 * M * N * K) / (tiled_time / 1000) / 1e9
    results['tiled'] = {'time': tiled_time, 'gflops': tiled_gflops}
    print(f"   Time: {tiled_time:.2f} ms")
    print(f"   Performance: {tiled_gflops:.1f} GFLOPS")
    print(f"   Speedup over naive: {naive_time/tiled_time:.2f}x")
    
    # 3. cuBLAS (via CuPy)
    print("\n3. cuBLAS Reference:")
    for _ in range(3):
        C_cublas = A @ B
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for _ in range(iterations):
        C_cublas = A @ B
    end.record()
    end.synchronize()
    
    cublas_time = cp.cuda.get_elapsed_time(start, end) / iterations
    cublas_gflops = (2 * M * N * K) / (cublas_time / 1000) / 1e9
    results['cublas'] = {'time': cublas_time, 'gflops': cublas_gflops}
    print(f"   Time: {cublas_time:.2f} ms")
    print(f"   Performance: {cublas_gflops:.1f} GFLOPS")
    print(f"   Speedup over naive: {naive_time/cublas_time:.2f}x")
    
    # Verification
    print("\n4. Verification:")
    if cp.allclose(C_naive, C_tiled, rtol=1e-4):
        print("   Naive vs Tiled: ✓ PASSED")
    else:
        print("   Naive vs Tiled: ✗ FAILED")
        
    if cp.allclose(C_tiled, C_cublas, rtol=1e-4):
        print("   Tiled vs cuBLAS: ✓ PASSED")
    else:
        print("   Tiled vs cuBLAS: ✗ FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"   Naive:  {results['naive']['gflops']:7.1f} GFLOPS (baseline)")
    print(f"   Tiled:  {results['tiled']['gflops']:7.1f} GFLOPS ({naive_time/tiled_time:.1f}x speedup)")
    print(f"   cuBLAS: {results['cublas']['gflops']:7.1f} GFLOPS ({naive_time/cublas_time:.1f}x speedup)")
    print(f"\n   Tiled achieves {100*tiled_gflops/cublas_gflops:.1f}% of cuBLAS performance")
    
    return results


def main():
    print("="*60)
    print("WEEK 1 PROJECT: Optimized Matrix Multiplication")
    print("Phase 6: AI/ML Platform Engineering with GPU Programming")
    print("="*60)
    
    # Check GPU
    print(f"\nGPU: {cp.cuda.Device().name}")
    
    # Run benchmarks for different sizes
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    
    all_results = {}
    for M, N, K in sizes:
        all_results[(M, N, K)] = benchmark_matmul(M, N, K)
    
    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print("\nKey Observations:")
    print("1. Tiled implementation with shared memory significantly outperforms naive")
    print("2. cuBLAS is still faster due to additional optimizations:")
    print("   - Vectorized memory loads")
    print("   - Register blocking")
    print("   - Tuned for specific GPU architectures")
    print("\nNext Steps (Week 2+):")
    print("- Implement vectorized loads (float4)")
    print("- Add register blocking")
    print("- Use Tensor Cores for mixed precision")


if __name__ == "__main__":
    main()
```

---

## 📊 Expected Results

```
============================================================
WEEK 1 PROJECT: Optimized Matrix Multiplication
Phase 6: AI/ML Platform Engineering with GPU Programming
============================================================

GPU: NVIDIA GeForce RTX 3090

============================================================
Matrix Multiplication Benchmark: (2048x2048) @ (2048x2048)
============================================================

1. Naive Implementation:
   Time: 45.23 ms
   Performance: 380.5 GFLOPS

2. Tiled (Shared Memory) Implementation:
   Time: 8.76 ms
   Performance: 1965.2 GFLOPS
   Speedup over naive: 5.16x

3. cuBLAS Reference:
   Time: 1.24 ms
   Performance: 13892.6 GFLOPS
   Speedup over naive: 36.47x

4. Verification:
   Naive vs Tiled: ✓ PASSED
   Tiled vs cuBLAS: ✓ PASSED

============================================================
Summary:
============================================================
   Naive:    380.5 GFLOPS (baseline)
   Tiled:   1965.2 GFLOPS (5.2x speedup)
   cuBLAS: 13892.6 GFLOPS (36.5x speedup)

   Tiled achieves 14.1% of cuBLAS performance
```

---

## 📝 Week 1 Summary

### What We Learned
1. GPU architecture fundamentals (SMs, Warps, Memory Hierarchy)
2. CUDA programming model (Kernels, Grids, Blocks)
3. Memory management (Device, Host, Pinned, Unified)
4. Thread synchronization and parallel patterns
5. Error handling and debugging
6. Shared memory optimization

### Skills Acquired
- Writing CUDA kernels in C++ and Python (CuPy)
- Understanding thread indexing and launch configuration
- Optimizing memory access patterns
- Using shared memory for data reuse
- Profiling and benchmarking GPU code

### Looking Ahead (Week 2)
- CUDA streams and concurrency
- Events and precise timing
- Atomic operations
- Multi-GPU programming

---

**Week 1 Complete! 🎉**

*Great foundation established. Ready for advanced CUDA programming in Week 2!*
