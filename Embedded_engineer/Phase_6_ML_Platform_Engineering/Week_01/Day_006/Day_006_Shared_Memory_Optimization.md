# Day 6: Shared Memory Optimization
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 1: GPU Architecture & CUDA Foundations

---

> **🎯 Focus Area:** Leveraging shared memory for dramatic performance improvements.

---

## 🎯 Learning Objectives
1. **Understand** shared memory architecture and bank conflicts
2. **Implement** tiling strategies for memory-bound kernels
3. **Optimize** matrix transpose using shared memory
4. **Analyze** occupancy and resource usage

---

## 📖 Theoretical Foundation

### Shared Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 SHARED MEMORY (per SM)                       │
├─────────────────────────────────────────────────────────────┤
│  32 Banks (32-bit wide each)                                 │
│                                                              │
│  Bank: 0  1  2  3  4  5 ... 30 31                           │
│        │  │  │  │  │  │      │  │                           │
│        ▼  ▼  ▼  ▼  ▼  ▼      ▼  ▼                           │
│     [ W0 W1 W2 W3 W4 W5 ... W30 W31 ]  Word 0-31            │
│     [ W32 W33 ... ]                     Word 32-63           │
│                                                              │
│  Stride-1 access: No conflicts (best)                        │
│  Stride-32 access: All threads hit same bank (worst!)        │
└─────────────────────────────────────────────────────────────┘
```

### Bank Conflicts

```cpp
// NO conflict: Threads access consecutive addresses
__shared__ float s[256];
s[threadIdx.x] = data[idx];  // Thread 0→s[0], Thread 1→s[1], ...

// CONFLICT: All threads hit bank 0
s[threadIdx.x * 32] = data[idx];  // All access bank 0!

// Solution: Padding
__shared__ float s[32][33];  // 33 instead of 32 to avoid conflicts
s[threadIdx.y][threadIdx.x] = data[idx];
```

---

## 💻 Implementation

### Matrix Transpose - Naive vs Optimized

```cpp
// Naive transpose: Coalesced read, strided write (bad!)
__global__ void transposeNaive(float* out, const float* in, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < N && y < N) {
        out[x * N + y] = in[y * N + x];  // Strided write!
    }
}

// Optimized: Use shared memory tile
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeOptimized(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 for padding
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile with coalesced reads
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < N && (y + i) < N) {
            tile[threadIdx.y + i][threadIdx.x] = in[(y + i) * N + x];
        }
    }
    
    __syncthreads();
    
    // Write transposed tile with coalesced writes
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // Swap block indices
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < N && (y + i) < N) {
            out[(y + i) * N + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}
```

### Python Implementation

```python
#!/usr/bin/env python3
"""Day 6: Shared Memory Optimization in Python"""

import cupy as cp
import numpy as np

# Naive transpose kernel
naive_kernel = cp.RawKernel(r'''
extern "C" __global__
void transposeNaive(float* out, const float* in, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < N && y < N) {
        out[x * N + y] = in[y * N + x];
    }
}
''', 'transposeNaive')

# Optimized with shared memory
optimized_kernel = cp.RawKernel(r'''
#define TILE_DIM 32

extern "C" __global__
void transposeOptimized(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < N && y < N) {
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}
''', 'transposeOptimized')

def benchmark_transpose():
    N = 4096
    
    A = cp.random.randn(N, N, dtype=cp.float32)
    B_naive = cp.zeros((N, N), dtype=cp.float32)
    B_opt = cp.zeros((N, N), dtype=cp.float32)
    
    block = (32, 32)
    grid = (N // 32, N // 32)
    
    # Warmup and benchmark naive
    for _ in range(3):
        naive_kernel(grid, block, (B_naive, A, N))
    cp.cuda.Stream.null.synchronize()
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    for _ in range(10):
        naive_kernel(grid, block, (B_naive, A, N))
    end.record()
    end.synchronize()
    naive_time = cp.cuda.get_elapsed_time(start, end) / 10
    
    # Benchmark optimized
    for _ in range(3):
        optimized_kernel(grid, block, (B_opt, A, N))
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for _ in range(10):
        optimized_kernel(grid, block, (B_opt, A, N))
    end.record()
    end.synchronize()
    opt_time = cp.cuda.get_elapsed_time(start, end) / 10
    
    # Bandwidth calculation
    bytes_processed = N * N * 4 * 2  # Read + Write
    naive_bw = bytes_processed / 1e9 / (naive_time / 1000)
    opt_bw = bytes_processed / 1e9 / (opt_time / 1000)
    
    print(f"Matrix Transpose ({N}x{N})")
    print(f"  Naive:     {naive_time:.2f} ms, {naive_bw:.1f} GB/s")
    print(f"  Optimized: {opt_time:.2f} ms, {opt_bw:.1f} GB/s")
    print(f"  Speedup:   {naive_time/opt_time:.2f}x")
    
    # Verify
    assert cp.allclose(B_naive, B_opt), "Results don't match!"
    assert cp.allclose(B_opt, A.T), "Transpose incorrect!"
    print("  Verification: PASSED")

if __name__ == "__main__":
    benchmark_transpose()
```

---

## 🔬 Lab Exercise: "Matrix Multiplication Tiling"

### Task
Implement tiled matrix multiplication using shared memory.

---

## 📝 Daily Summary

### Key Takeaways
1. Shared memory is ~100x faster than global memory
2. Bank conflicts reduce effective bandwidth
3. Padding (`[N][N+1]` instead of `[N][N]`) avoids conflicts
4. Tiling pattern: Load to shared → sync → compute → sync → store
5. Coalesced global memory access is still important

---

**Day 6 Complete** ✅
