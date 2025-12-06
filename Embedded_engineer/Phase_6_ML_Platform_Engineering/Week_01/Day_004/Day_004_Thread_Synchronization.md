# Day 4: Thread Synchronization & Execution
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 1: GPU Architecture & CUDA Foundations

---

> **🎯 Focus Area:** Understanding warps, synchronization barriers, and avoiding divergence.

---

## 🎯 Learning Objectives
1. **Understand** warp execution and SIMT model
2. **Use** `__syncthreads()` for block synchronization
3. **Identify** and avoid branch divergence
4. **Implement** parallel reduction pattern

---

## 📖 Theoretical Foundation

### Warp Execution Model

```
┌─────────────────────────────────────────────────────────────┐
│                      WARP (32 Threads)                       │
├─────────────────────────────────────────────────────────────┤
│  All 32 threads execute the SAME instruction simultaneously │
│                                                              │
│  Thread: 0  1  2  3  4  5  6  7 ... 28 29 30 31             │
│          │  │  │  │  │  │  │  │      │  │  │  │             │
│          ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼      ▼  ▼  ▼  ▼             │
│        [Same Instruction for All]                            │
│                                                              │
│  If threads diverge (different branches):                    │
│  - Both paths execute sequentially                           │
│  - Inactive threads are masked                               │
│  - Performance penalty!                                      │
└─────────────────────────────────────────────────────────────┘
```

### Branch Divergence Example

```cpp
// BAD: Divergence within warp
__global__ void divergentKernel(float* data) {
    int tid = threadIdx.x;
    int lane = tid % 32;  // Lane within warp
    
    if (lane < 16) {
        // Threads 0-15 execute this
        data[tid] = sinf(data[tid]);
    } else {
        // Threads 16-31 execute this (sequentially after above)
        data[tid] = cosf(data[tid]);
    }
    // Warp takes time for BOTH branches!
}

// GOOD: No divergence within warp
__global__ void nonDivergentKernel(float* data) {
    int tid = threadIdx.x;
    int warpId = tid / 32;
    
    if (warpId % 2 == 0) {
        // Entire warps 0, 2, 4, ... execute this
        data[tid] = sinf(data[tid]);
    } else {
        // Entire warps 1, 3, 5, ... execute this
        data[tid] = cosf(data[tid]);
    }
    // Each warp takes only ONE branch!
}
```

---

## 💻 Implementation

### Parallel Reduction (Classic Pattern)

```cpp
// reduction.cu - Sum reduction using shared memory
__global__ void reduceSum(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();  // Wait for all threads
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();  // Synchronize after each step
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Python Implementation

```python
#!/usr/bin/env python3
"""Day 4: Parallel Reduction with CuPy"""

import cupy as cp

reduction_kernel = cp.RawKernel(r'''
extern "C" __global__
void reduceSum(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory with bounds check
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
''', 'reduceSum')

def parallel_sum(data):
    N = len(data)
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    
    partial_sums = cp.zeros(grid_size, dtype=cp.float32)
    
    # First reduction
    reduction_kernel(
        (grid_size,), (block_size,),
        (data, partial_sums, N),
        shared_mem=block_size * 4
    )
    
    # Reduce partial sums (recursive call or final reduction)
    while len(partial_sums) > 1:
        N = len(partial_sums)
        grid_size = (N + block_size - 1) // block_size
        new_partial = cp.zeros(grid_size, dtype=cp.float32)
        
        reduction_kernel(
            (grid_size,), (block_size,),
            (partial_sums, new_partial, N),
            shared_mem=block_size * 4
        )
        partial_sums = new_partial
    
    return float(partial_sums[0])

# Test
data = cp.ones(1 << 20, dtype=cp.float32)
result = parallel_sum(data)
print(f"Sum of {len(data):,} ones = {result:,.0f} (expected: {len(data):,})")
```

---

## 🔬 Lab Exercise: "Optimized Reduction"

### Task
Compare naive reduction vs optimized versions:
1. Basic tree reduction
2. Warp-level reduction with `__shfl_down_sync`
3. Compare performance

---

## 📝 Daily Summary

### Key Takeaways
1. Warps execute in SIMT - 32 threads, one instruction
2. `__syncthreads()` synchronizes all threads in a block
3. Branch divergence causes sequential execution of both paths
4. Design algorithms to minimize divergence (warp-aligned branches)
5. Parallel reduction is a fundamental GPU pattern

---

**Day 4 Complete** ✅
