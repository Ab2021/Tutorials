# Day 10: Atomic Operations and Reductions
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 2: Advanced CUDA Programming

---

> **🎯 Focus Area:** Master thread-safe GPU operations, atomic primitives, and efficient parallel reduction algorithms for ML applications.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Understand** the need for atomic operations in parallel programming
2. **Implement** atomic operations for thread-safe data updates
3. **Build** efficient parallel reduction algorithms using warp primitives
4. **Optimize** histogram and aggregation kernels
5. **Compare** atomic-based vs reduction-based approaches
6. **Apply** these patterns to real ML workloads (loss aggregation, gradient accumulation)

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with compute capability 6.0+ (for native atomics)
- Recommended: GPU with hardware atomic support for FP32

### Software Environment
```bash
pip install cupy-cuda12x numpy

# Verify atomic support
python -c "import cupy as cp; print(f'CC: {cp.cuda.Device().compute_capability}')"
```

### Prior Knowledge
- Day 4: Thread Synchronization
- Day 8-9: Streams and Events

---

## 📖 Theoretical Foundation

### 1. The Race Condition Problem

When multiple threads modify the same memory location simultaneously, results become unpredictable:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RACE CONDITION EXAMPLE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Goal: Increment shared counter by 1000 threads                              │
│                                                                              │
│  Thread 0:           Thread 1:           Thread 2:                           │
│  ─────────           ─────────           ─────────                           │
│  1. Read counter=0   1. Read counter=0   1. Read counter=0                   │
│  2. Add 1 → 1        2. Add 1 → 1        2. Add 1 → 1                        │
│  3. Write 1          3. Write 1          3. Write 1                          │
│                                                                              │
│  Expected result: 1000                                                       │
│  Actual result: Some value between 1 and 1000 (non-deterministic!)          │
│                                                                              │
│  The Problem: READ-MODIFY-WRITE is NOT atomic!                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Atomic Operations

Atomic operations guarantee that the read-modify-write cycle is **indivisible**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ATOMIC OPERATIONS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  atomicAdd(&counter, 1):                                                     │
│                                                                              │
│  Thread 0:           Thread 1:           Thread 2:                           │
│  ─────────           ─────────           ─────────                           │
│  ┌─────────────┐     (waits)             (waits)                             │
│  │ READ: 0     │                                                             │
│  │ ADD:  1     │                                                             │
│  │ WRITE: 1    │                                                             │
│  └─────────────┘                                                             │
│       │              ┌─────────────┐     (waits)                             │
│       │              │ READ: 1     │                                         │
│       │              │ ADD:  1     │                                         │
│       │              │ WRITE: 2    │                                         │
│       │              └─────────────┘                                         │
│       │                   │              ┌─────────────┐                     │
│       │                   │              │ READ: 2     │                     │
│       │                   │              │ ADD:  1     │                     │
│       │                   │              │ WRITE: 3    │                     │
│       │                   │              └─────────────┘                     │
│       │                   │                   │                              │
│       ▼                   ▼                   ▼                              │
│                     Final Result: 3 ✓                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Available Atomic Operations

```cpp
// Arithmetic Operations
atomicAdd(address, val);      // *address += val (returns old)
atomicSub(address, val);      // *address -= val
atomicMin(address, val);      // *address = min(*address, val)
atomicMax(address, val);      // *address = max(*address, val)

// Bitwise Operations
atomicAnd(address, val);      // *address &= val
atomicOr(address, val);       // *address |= val
atomicXor(address, val);      // *address ^= val

// Exchange Operations
atomicExch(address, val);     // swap *address with val (returns old)
atomicCAS(address, compare, val);  // if (*address == compare) *address = val

// Increment/Decrement (wrapping)
atomicInc(address, val);      // (*address >= val) ? 0 : (*address + 1)
atomicDec(address, val);      // (*address == 0 || *address > val) ? val : (*address - 1)
```

### 4. Atomic Support by Data Type

| Data Type | Add | Min/Max | CAS | Notes |
|-----------|-----|---------|-----|-------|
| int32 | ✓ | ✓ | ✓ | Full support |
| uint32 | ✓ | ✓ | ✓ | Full support |
| int64 | ✓ | ✓ | ✓ | CC 3.5+ |
| uint64 | ✓ | ✓ | ✓ | CC 3.5+ |
| float32 | ✓ | ✓ | Via CAS | Native on CC 6.0+ |
| float64 | ✓ | Via CAS | ✓ | CC 6.0+ |
| float16 | ✓ | Via CAS | Via CAS | CC 7.0+ (Tensor Cores) |

### 5. Warp-Level Primitives

Modern GPUs provide warp-level operations that are much faster than atomics:

```cpp
// Warp shuffle operations (no shared memory needed)
__shfl_sync(mask, val, srcLane);        // Get val from specific lane
__shfl_up_sync(mask, val, delta);       // Get val from (laneId - delta)
__shfl_down_sync(mask, val, delta);     // Get val from (laneId + delta)
__shfl_xor_sync(mask, val, laneMask);   // Get val from (laneId ^ laneMask)

// Warp vote operations
__all_sync(mask, predicate);    // True if all lanes satisfy predicate
__any_sync(mask, predicate);    // True if any lane satisfies predicate
__ballot_sync(mask, predicate); // Bitmask of lanes satisfying predicate

// Warp match operations (CC 7.0+)
__match_any_sync(mask, val);    // Bitmask of lanes with matching val
__match_all_sync(mask, val, &pred); // True if all have same val

// Warp reduce operations (CC 8.0+)
__reduce_add_sync(mask, val);   // Sum across warp
__reduce_min_sync(mask, val);   // Min across warp
__reduce_max_sync(mask, val);   // Max across warp
```

### 6. Parallel Reduction Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TREE REDUCTION PATTERN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: [1, 2, 3, 4, 5, 6, 7, 8]                                             │
│                                                                              │
│  Step 1 (stride=1):                                                          │
│    Thread 0: 1+2=3    Thread 2: 3+4=7    Thread 4: 5+6=11   Thread 6: 7+8=15 │
│              ↓                 ↓                  ↓                   ↓      │
│             [3,    2,    7,    4,   11,    6,   15,    8]                    │
│                                                                              │
│  Step 2 (stride=2):                                                          │
│    Thread 0: 3+7=10                      Thread 4: 11+15=26                  │
│              ↓                                      ↓                        │
│            [10,    2,    7,    4,   26,    6,   15,    8]                    │
│                                                                              │
│  Step 3 (stride=4):                                                          │
│    Thread 0: 10+26=36                                                        │
│              ↓                                                               │
│            [36,    2,    7,    4,   26,    6,   15,    8]                    │
│                                                                              │
│  Result: 36 = 1+2+3+4+5+6+7+8 ✓                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Implementation

### 👨‍💻 Core Implementation

#### 📁 `src/atomics_demo.py` - Atomic Operations
```python
#!/usr/bin/env python3
"""
Day 10: Atomic Operations and Reductions
Phase 6: AI/ML Platform Engineering with GPU Programming
"""

import cupy as cp
import numpy as np
from typing import Tuple, List
import time


# ============================================================================
# ATOMIC OPERATIONS - Basic Examples
# ============================================================================

atomic_add_kernel = cp.RawKernel(r'''
extern "C" __global__
void atomicAddExample(int* counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(counter, 1);
    }
}
''', 'atomicAddExample')


atomic_histogram_kernel = cp.RawKernel(r'''
extern "C" __global__
void atomicHistogram(const int* data, int* histogram, int n, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int bin = data[tid] % num_bins;
        atomicAdd(&histogram[bin], 1);
    }
}
''', 'atomicHistogram')


atomic_minmax_kernel = cp.RawKernel(r'''
extern "C" __global__
void atomicMinMax(const float* data, float* min_val, float* max_val, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // For float, we need to use atomicCAS to implement atomicMin/Max
        // or use __float_as_int/__int_as_float trick
        
        float val = data[tid];
        
        // Atomic min (using reinterpret trick for floats)
        int* min_ptr = (int*)min_val;
        int old_int = *min_ptr;
        float old_float = __int_as_float(old_int);
        
        while (val < old_float) {
            int assumed = old_int;
            int new_val = __float_as_int(val);
            old_int = atomicCAS(min_ptr, assumed, new_val);
            if (old_int == assumed) break;
            old_float = __int_as_float(old_int);
        }
        
        // Atomic max
        int* max_ptr = (int*)max_val;
        old_int = *max_ptr;
        old_float = __int_as_float(old_int);
        
        while (val > old_float) {
            int assumed = old_int;
            int new_val = __float_as_int(val);
            old_int = atomicCAS(max_ptr, assumed, new_val);
            if (old_int == assumed) break;
            old_float = __int_as_float(old_int);
        }
    }
}
''', 'atomicMinMax')


def demonstrate_atomic_add():
    """Demonstrate basic atomic add operation."""
    print("\n" + "="*70)
    print("ATOMIC ADD DEMONSTRATION")
    print("="*70)
    
    n = 1_000_000
    counter = cp.zeros(1, dtype=cp.int32)
    
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    atomic_add_kernel((grid_size,), (block_size,), (counter, n))
    cp.cuda.Stream.null.synchronize()
    
    result = int(counter[0])
    print(f"  {n:,} threads incremented counter")
    print(f"  Final value: {result:,}")
    print(f"  Expected: {n:,}")
    print(f"  Status: {'✓ CORRECT' if result == n else '✗ ERROR'}")


def demonstrate_histogram():
    """Demonstrate atomic histogram computation."""
    print("\n" + "="*70)
    print("ATOMIC HISTOGRAM DEMONSTRATION")
    print("="*70)
    
    n = 10_000_000
    num_bins = 256
    
    # Generate random data
    data = cp.random.randint(0, 1000, size=n, dtype=cp.int32)
    histogram = cp.zeros(num_bins, dtype=cp.int32)
    
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    # Time the atomic histogram
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    atomic_histogram_kernel((grid_size,), (block_size,), (data, histogram, n, num_bins))
    end.record()
    end.synchronize()
    
    elapsed = cp.cuda.get_elapsed_time(start, end)
    
    # Verify
    histogram_sum = int(cp.sum(histogram))
    
    print(f"  Data size: {n:,} elements")
    print(f"  Number of bins: {num_bins}")
    print(f"  Time: {elapsed:.2f} ms")
    print(f"  Histogram sum: {histogram_sum:,} (expected: {n:,})")
    print(f"  Status: {'✓ CORRECT' if histogram_sum == n else '✗ ERROR'}")


# ============================================================================
# PARALLEL REDUCTION - Optimized Implementations
# ============================================================================

# Level 1: Basic reduction with shared memory
reduction_v1_kernel = cp.RawKernel(r'''
extern "C" __global__
void reduceSum_v1(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
''', 'reduceSum_v1')


# Level 2: Warp-unrolled reduction (last 32 threads don't need sync)
reduction_v2_kernel = cp.RawKernel(r'''
__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

extern "C" __global__
void reduceSum_v2(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction until warp size
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp reduction (no sync needed within warp)
    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
''', 'reduceSum_v2')


# Level 3: Using warp shuffle (modern approach)
reduction_v3_kernel = cp.RawKernel(r'''
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // One per warp
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Only first warp does final reduction
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

extern "C" __global__
void reduceSum_v3(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    val = blockReduceSum(val);
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = val;
    }
}
''', 'reduceSum_v3')


# Level 4: Grid-stride loop + warp shuffle
reduction_v4_kernel = cp.RawKernel(r'''
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

extern "C" __global__
void reduceSum_v4(const float* input, float* output, int n) {
    float sum = 0.0f;
    
    // Grid-stride loop for processing more data per thread
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        sum += input[idx];
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}
''', 'reduceSum_v4')


def benchmark_reductions():
    """Benchmark different reduction implementations."""
    print("\n" + "="*70)
    print("PARALLEL REDUCTION BENCHMARK")
    print("="*70)
    
    n = 16_000_000
    data = cp.random.randn(n, dtype=cp.float32)
    
    # Expected result
    expected = float(cp.sum(data))
    
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    results = {}
    
    # Version 1: Basic reduction
    print("\n1. Basic Reduction (shared memory):")
    output_v1 = cp.zeros(grid_size, dtype=cp.float32)
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    # Warmup
    for _ in range(3):
        output_v1.fill(0)
        reduction_v1_kernel((grid_size,), (block_size,), 
                           (data, output_v1, n), shared_mem=block_size * 4)
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for _ in range(10):
        output_v1.fill(0)
        reduction_v1_kernel((grid_size,), (block_size,), 
                           (data, output_v1, n), shared_mem=block_size * 4)
    end.record()
    end.synchronize()
    
    time_v1 = cp.cuda.get_elapsed_time(start, end) / 10
    result_v1 = float(cp.sum(output_v1))
    error_v1 = abs(result_v1 - expected) / abs(expected)
    
    print(f"   Time: {time_v1:.3f} ms")
    print(f"   Relative Error: {error_v1:.2e}")
    results['v1_basic'] = time_v1
    
    # Version 2: Warp-unrolled
    print("\n2. Warp-Unrolled Reduction:")
    output_v2 = cp.zeros(grid_size, dtype=cp.float32)
    
    for _ in range(3):
        output_v2.fill(0)
        reduction_v2_kernel((grid_size,), (block_size,), 
                           (data, output_v2, n), shared_mem=block_size * 4)
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for _ in range(10):
        output_v2.fill(0)
        reduction_v2_kernel((grid_size,), (block_size,), 
                           (data, output_v2, n), shared_mem=block_size * 4)
    end.record()
    end.synchronize()
    
    time_v2 = cp.cuda.get_elapsed_time(start, end) / 10
    result_v2 = float(cp.sum(output_v2))
    error_v2 = abs(result_v2 - expected) / abs(expected)
    
    print(f"   Time: {time_v2:.3f} ms ({time_v1/time_v2:.2f}x vs basic)")
    print(f"   Relative Error: {error_v2:.2e}")
    results['v2_unrolled'] = time_v2
    
    # Version 3: Warp shuffle
    print("\n3. Warp Shuffle Reduction:")
    output_v3 = cp.zeros(grid_size, dtype=cp.float32)
    
    for _ in range(3):
        output_v3.fill(0)
        reduction_v3_kernel((grid_size,), (block_size,), (data, output_v3, n))
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for _ in range(10):
        output_v3.fill(0)
        reduction_v3_kernel((grid_size,), (block_size,), (data, output_v3, n))
    end.record()
    end.synchronize()
    
    time_v3 = cp.cuda.get_elapsed_time(start, end) / 10
    result_v3 = float(cp.sum(output_v3))
    error_v3 = abs(result_v3 - expected) / abs(expected)
    
    print(f"   Time: {time_v3:.3f} ms ({time_v1/time_v3:.2f}x vs basic)")
    print(f"   Relative Error: {error_v3:.2e}")
    results['v3_shuffle'] = time_v3
    
    # Version 4: Grid-stride + atomic
    print("\n4. Grid-Stride + Atomic Reduction:")
    output_v4 = cp.zeros(1, dtype=cp.float32)
    
    # Use fewer blocks for grid-stride
    grid_size_v4 = min(grid_size, 1024)
    
    for _ in range(3):
        output_v4.fill(0)
        reduction_v4_kernel((grid_size_v4,), (block_size,), (data, output_v4, n))
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for _ in range(10):
        output_v4.fill(0)
        reduction_v4_kernel((grid_size_v4,), (block_size,), (data, output_v4, n))
    end.record()
    end.synchronize()
    
    time_v4 = cp.cuda.get_elapsed_time(start, end) / 10
    result_v4 = float(output_v4[0])
    error_v4 = abs(result_v4 - expected) / abs(expected)
    
    print(f"   Time: {time_v4:.3f} ms ({time_v1/time_v4:.2f}x vs basic)")
    print(f"   Relative Error: {error_v4:.2e}")
    results['v4_gridstride'] = time_v4
    
    # CuPy reference
    print("\n5. CuPy (cuBLAS) Reference:")
    
    for _ in range(3):
        _ = cp.sum(data)
    cp.cuda.Stream.null.synchronize()
    
    start.record()
    for _ in range(10):
        result_cupy = cp.sum(data)
    end.record()
    end.synchronize()
    
    time_cupy = cp.cuda.get_elapsed_time(start, end) / 10
    
    print(f"   Time: {time_cupy:.3f} ms ({time_v1/time_cupy:.2f}x vs basic)")
    results['cupy'] = time_cupy
    
    # Summary
    print("\n" + "-"*50)
    print("SUMMARY:")
    print(f"   Best custom: {min(time_v1, time_v2, time_v3, time_v4):.3f} ms")
    print(f"   CuPy/cuBLAS: {time_cupy:.3f} ms")
    print(f"   Data size: {n:,} float32 ({n * 4 / 1e6:.1f} MB)")
    
    return results


# ============================================================================
# ML-SPECIFIC PATTERNS
# ============================================================================

gradient_accumulation_kernel = cp.RawKernel(r'''
extern "C" __global__
void accumulateGradients(
    const float* local_gradients,  // [num_params]
    float* accumulated_gradients,  // [num_params]
    int num_params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_params) {
        atomicAdd(&accumulated_gradients[idx], local_gradients[idx]);
    }
}
''', 'accumulateGradients')


loss_reduction_kernel = cp.RawKernel(r'''
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__
void reduceLoss(
    const float* per_sample_loss,  // [batch_size]
    float* total_loss,
    int batch_size
) {
    float sum = 0.0f;
    
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < batch_size;
         idx += blockDim.x * gridDim.x) {
        sum += per_sample_loss[idx];
    }
    
    // Block reduce
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    sum = warpReduceSum(sum);
    if (lane == 0) shared[wid] = sum;
    __syncthreads();
    
    sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) sum = warpReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(total_loss, sum);
    }
}
''', 'reduceLoss')


def demonstrate_ml_patterns():
    """Demonstrate ML-specific atomic patterns."""
    print("\n" + "="*70)
    print("ML-SPECIFIC PATTERNS")
    print("="*70)
    
    # Gradient Accumulation (mini-batch)
    print("\n1. Gradient Accumulation:")
    
    num_params = 1_000_000
    num_micro_batches = 4
    
    accumulated = cp.zeros(num_params, dtype=cp.float32)
    
    block_size = 256
    grid_size = (num_params + block_size - 1) // block_size
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    for _ in range(num_micro_batches):
        # Simulate gradients from one micro-batch
        gradients = cp.random.randn(num_params, dtype=cp.float32) * 0.01
        gradient_accumulation_kernel((grid_size,), (block_size,), 
                                     (gradients, accumulated, num_params))
    end.record()
    end.synchronize()
    
    elapsed = cp.cuda.get_elapsed_time(start, end)
    
    print(f"   Parameters: {num_params:,}")
    print(f"   Micro-batches: {num_micro_batches}")
    print(f"   Total time: {elapsed:.3f} ms")
    print(f"   Per micro-batch: {elapsed/num_micro_batches:.3f} ms")
    
    # Loss Reduction
    print("\n2. Loss Reduction:")
    
    batch_size = 32768
    per_sample_loss = cp.abs(cp.random.randn(batch_size, dtype=cp.float32))
    total_loss = cp.zeros(1, dtype=cp.float32)
    
    grid_size = min((batch_size + 255) // 256, 256)
    
    start.record()
    loss_reduction_kernel((grid_size,), (256,), 
                          (per_sample_loss, total_loss, batch_size))
    end.record()
    end.synchronize()
    
    elapsed = cp.cuda.get_elapsed_time(start, end)
    result = float(total_loss[0])
    expected = float(cp.sum(per_sample_loss))
    
    print(f"   Batch size: {batch_size:,}")
    print(f"   Time: {elapsed:.3f} ms")
    print(f"   Result: {result:.6f}")
    print(f"   Expected: {expected:.6f}")
    print(f"   Error: {abs(result - expected):.6e}")


def main():
    """Main function running all demonstrations."""
    print("="*70)
    print("DAY 10: ATOMIC OPERATIONS AND REDUCTIONS")
    print("Phase 6: AI/ML Platform Engineering with GPU Programming")
    print("="*70)
    
    print(f"\nGPU: {cp.cuda.Device().name}")
    print(f"Compute Capability: {cp.cuda.Device().compute_capability}")
    
    demonstrate_atomic_add()
    demonstrate_histogram()
    benchmark_reductions()
    demonstrate_ml_patterns()
    
    print("\n" + "="*70)
    print("Day 10 Complete: Atomic Operations and Reductions")
    print("="*70)


if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Building a GPU-Based Aggregation Engine"

### Lab Objectives
1. Implement multiple aggregation functions (sum, min, max, mean, variance)
2. Build a reusable aggregation kernel library
3. Compare atomic vs reduction-based approaches

### Implementation

```python
#!/usr/bin/env python3
"""
Lab: GPU Aggregation Engine
Day 10: Atomic Operations and Reductions
"""

import cupy as cp
import numpy as np
from typing import Dict, Callable
from dataclasses import dataclass
from enum import Enum


class AggregationType(Enum):
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    VARIANCE = "variance"
    COUNT = "count"


@dataclass
class AggregationResult:
    """Result of an aggregation operation."""
    agg_type: AggregationType
    value: float
    time_ms: float


class GPUAggregator:
    """
    GPU-based aggregation engine.
    
    Implements efficient parallel aggregation using warp primitives.
    """
    
    def __init__(self):
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile all aggregation kernels."""
        
        self.sum_kernel = cp.RawKernel(r'''
        __device__ float warpReduceSum(float val) {
            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xffffffff, val, offset);
            return val;
        }
        
        extern "C" __global__
        void aggregateSum(const float* data, float* result, int n) {
            float sum = 0.0f;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
                sum += data[i];
            
            __shared__ float shared[32];
            int lane = threadIdx.x % 32;
            int wid = threadIdx.x / 32;
            
            sum = warpReduceSum(sum);
            if (lane == 0) shared[wid] = sum;
            __syncthreads();
            
            sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
            if (wid == 0) sum = warpReduceSum(sum);
            
            if (threadIdx.x == 0) atomicAdd(result, sum);
        }
        ''', 'aggregateSum')
        
        self.minmax_kernel = cp.RawKernel(r'''
        __device__ float warpReduceMin(float val) {
            for (int offset = 16; offset > 0; offset >>= 1)
                val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
            return val;
        }
        
        __device__ float warpReduceMax(float val) {
            for (int offset = 16; offset > 0; offset >>= 1)
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            return val;
        }
        
        extern "C" __global__
        void aggregateMinMax(const float* data, float* min_result, float* max_result, int n) {
            float local_min = 1e38f;
            float local_max = -1e38f;
            
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                float val = data[i];
                local_min = fminf(local_min, val);
                local_max = fmaxf(local_max, val);
            }
            
            __shared__ float shared_min[32];
            __shared__ float shared_max[32];
            int lane = threadIdx.x % 32;
            int wid = threadIdx.x / 32;
            
            local_min = warpReduceMin(local_min);
            local_max = warpReduceMax(local_max);
            
            if (lane == 0) {
                shared_min[wid] = local_min;
                shared_max[wid] = local_max;
            }
            __syncthreads();
            
            local_min = (threadIdx.x < blockDim.x / 32) ? shared_min[lane] : 1e38f;
            local_max = (threadIdx.x < blockDim.x / 32) ? shared_max[lane] : -1e38f;
            
            if (wid == 0) {
                local_min = warpReduceMin(local_min);
                local_max = warpReduceMax(local_max);
            }
            
            if (threadIdx.x == 0) {
                // Atomic min/max for floats using CAS
                int* min_ptr = (int*)min_result;
                int old = *min_ptr;
                while (__int_as_float(old) > local_min) {
                    old = atomicCAS(min_ptr, old, __float_as_int(local_min));
                }
                
                int* max_ptr = (int*)max_result;
                old = *max_ptr;
                while (__int_as_float(old) < local_max) {
                    old = atomicCAS(max_ptr, old, __float_as_int(local_max));
                }
            }
        }
        ''', 'aggregateMinMax')
    
    def sum(self, data: cp.ndarray) -> AggregationResult:
        """Compute sum."""
        n = len(data)
        result = cp.zeros(1, dtype=cp.float32)
        
        block_size = 256
        grid_size = min((n + block_size - 1) // block_size, 256)
        
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        self.sum_kernel((grid_size,), (block_size,), (data, result, n))
        end.record()
        end.synchronize()
        
        return AggregationResult(
            agg_type=AggregationType.SUM,
            value=float(result[0]),
            time_ms=cp.cuda.get_elapsed_time(start, end)
        )
    
    def min_max(self, data: cp.ndarray) -> Dict[str, AggregationResult]:
        """Compute min and max simultaneously."""
        n = len(data)
        min_result = cp.array([1e38], dtype=cp.float32)
        max_result = cp.array([-1e38], dtype=cp.float32)
        
        block_size = 256
        grid_size = min((n + block_size - 1) // block_size, 256)
        
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        self.minmax_kernel((grid_size,), (block_size,), 
                          (data, min_result, max_result, n))
        end.record()
        end.synchronize()
        
        elapsed = cp.cuda.get_elapsed_time(start, end)
        
        return {
            'min': AggregationResult(AggregationType.MIN, float(min_result[0]), elapsed),
            'max': AggregationResult(AggregationType.MAX, float(max_result[0]), elapsed)
        }
    
    def mean(self, data: cp.ndarray) -> AggregationResult:
        """Compute mean."""
        sum_result = self.sum(data)
        return AggregationResult(
            agg_type=AggregationType.MEAN,
            value=sum_result.value / len(data),
            time_ms=sum_result.time_ms
        )


def run_lab():
    """Run the lab exercise."""
    print("="*60)
    print("LAB: GPU Aggregation Engine")
    print("Day 10: Atomic Operations and Reductions")
    print("="*60)
    
    # Initialize
    aggregator = GPUAggregator()
    
    # Test data
    n = 10_000_000
    data = cp.random.randn(n, dtype=cp.float32)
    
    print(f"\nData: {n:,} random float32 values")
    
    # Test aggregations
    print("\n1. Sum:")
    result = aggregator.sum(data)
    expected = float(cp.sum(data))
    print(f"   Result: {result.value:.6f}")
    print(f"   Expected: {expected:.6f}")
    print(f"   Time: {result.time_ms:.3f} ms")
    
    print("\n2. Min/Max:")
    minmax = aggregator.min_max(data)
    print(f"   Min: {minmax['min'].value:.6f} (expected: {float(cp.min(data)):.6f})")
    print(f"   Max: {minmax['max'].value:.6f} (expected: {float(cp.max(data)):.6f})")
    print(f"   Time: {minmax['min'].time_ms:.3f} ms")
    
    print("\n3. Mean:")
    result = aggregator.mean(data)
    print(f"   Result: {result.value:.6f}")
    print(f"   Expected: {float(cp.mean(data)):.6f}")
    print(f"   Time: {result.time_ms:.3f} ms")
    
    print("\nLab complete!")


if __name__ == "__main__":
    run_lab()
```

---

## 📝 Daily Summary

### Key Takeaways
1. **Atomics solve race conditions** - Guarantee thread-safe memory updates
2. **Atomics are slow** - Use sparingly; prefer reductions when possible
3. **Warp primitives are fast** - `__shfl_*` avoids shared memory
4. **Tree reduction pattern** - O(log n) steps with good parallelism
5. **Grid-stride loops** - Process more data per thread
6. **ML patterns** - Gradient accumulation and loss reduction are common

### Performance Guidelines
- Use atomic for irregular access patterns (histograms)
- Use warp reductions for regular patterns (sum, max)
- Combine: warp reduce within block, atomic across blocks
- Profile to choose: atomics vs multiple kernel passes

---

**Day 10 Complete** ✅

*Next: Day 11 - Dynamic Parallelism - Launching kernels from kernels!*
