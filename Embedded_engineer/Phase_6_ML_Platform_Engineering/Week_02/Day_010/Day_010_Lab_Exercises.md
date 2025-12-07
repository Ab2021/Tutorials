# Day 10: Atomic Operations and Reductions - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Atomic Operations (Beginner)

### Objective
Use atomic operations for thread-safe updates.

### Code

```python
from numba import cuda
import numpy as np

@cuda.jit
def histogram_atomic(data, bins, hist):
    """Compute histogram using atomics."""
    idx = cuda.grid(1)
    
    if idx < data.size:
        # Find bin
        val = data[idx]
        bin_idx = int(val * bins)
        if bin_idx >= bins:
            bin_idx = bins - 1
        
        # Atomic increment
        cuda.atomic.add(hist, bin_idx, 1)

def compute_histogram():
    """Compute histogram of random data."""
    n = 1_000_000
    bins = 100
    
    # Generate data in [0, 1)
    data = np.random.rand(n).astype(np.float32)
    hist = np.zeros(bins, dtype=np.int32)
    
    d_data = cuda.to_device(data)
    d_hist = cuda.to_device(hist)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    
    histogram_atomic[blocks, threads](d_data, bins, d_hist)
    
    result = d_hist.copy_to_host()
    
    print("Histogram (first 10 bins):")
    for i in range(10):
        bar = '*' * (result[i] // 500)
        print(f"Bin {i:2d}: {result[i]:6d} {bar}")
    
    print(f"\nTotal: {result.sum()} (expected: {n})")

if __name__ == "__main__":
    compute_histogram()
```

---

## 🔬 Exercise 2: Warp-Level Reduction (Intermediate)

### Objective
Use warp shuffle for efficient reduction.

### Code

```python
from numba import cuda
import numpy as np
import numba

WARP_SIZE = 32

@cuda.jit
def warp_reduce_sum(data, result):
    """Sum using warp shuffle."""
    idx = cuda.grid(1)
    lane = cuda.threadIdx.x % WARP_SIZE
    
    # Load value
    val = 0.0
    if idx < data.size:
        val = data[idx]
    
    # Warp-level reduction using shuffle
    for offset in [16, 8, 4, 2, 1]:
        val += cuda.shfl_down_sync(0xFFFFFFFF, val, offset)
    
    # Lane 0 has the sum for this warp
    if lane == 0:
        warp_id = idx // WARP_SIZE
        if warp_id < result.size:
            result[warp_id] = val

def warp_reduce_demo():
    """Demonstrate warp reduction."""
    n = 1024
    data = np.ones(n, dtype=np.float32)
    num_warps = n // WARP_SIZE
    result = np.zeros(num_warps, dtype=np.float32)
    
    d_data = cuda.to_device(data)
    d_result = cuda.to_device(result)
    
    warp_reduce_sum[n // 256, 256](d_data, d_result)
    
    warp_sums = d_result.copy_to_host()
    total = warp_sums.sum()
    
    print(f"Warp Reduction Demo")
    print(f"=" * 40)
    print(f"Input: {n} ones")
    print(f"Warp sums: {warp_sums[:5]}...")
    print(f"Total: {total} (expected: {n})")

if __name__ == "__main__":
    warp_reduce_demo()
```

---

## 🔬 Exercise 3: Parallel Reduction Tree (Advanced)

### Objective
Implement efficient parallel sum.

### Code

```python
from numba import cuda
import numpy as np
import numba
import time

@cuda.jit
def block_reduce_sum(data, partial_sums):
    """Block-level parallel reduction."""
    shared = cuda.shared.array(256, dtype=numba.float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    idx = bid * cuda.blockDim.x + tid
    
    # Load to shared memory
    if idx < data.size:
        shared[tid] = data[idx]
    else:
        shared[tid] = 0.0
    
    cuda.syncthreads()
    
    # Tree reduction
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2
    
    # Write block result
    if tid == 0:
        partial_sums[bid] = shared[0]

def parallel_sum(data):
    """Full parallel sum."""
    n = len(data)
    threads = 256
    blocks = (n + threads - 1) // threads
    
    d_data = cuda.to_device(data)
    d_partial = cuda.device_array(blocks, dtype=np.float32)
    
    block_reduce_sum[blocks, threads](d_data, d_partial)
    
    # Sum partial results (could recurse for large arrays)
    return d_partial.copy_to_host().sum()

def benchmark_reduction():
    """Compare reduction methods."""
    n = 10_000_000
    data = np.random.rand(n).astype(np.float32)
    
    # CPU
    start = time.perf_counter()
    cpu_sum = data.sum()
    cpu_time = time.perf_counter() - start
    
    # GPU
    cuda.synchronize()
    start = time.perf_counter()
    gpu_sum = parallel_sum(data)
    cuda.synchronize()
    gpu_time = time.perf_counter() - start
    
    print(f"Reduction Benchmark ({n:,} elements)")
    print("=" * 40)
    print(f"CPU: {cpu_time*1000:.2f} ms (sum={cpu_sum:.6f})")
    print(f"GPU: {gpu_time*1000:.2f} ms (sum={gpu_sum:.6f})")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")

if __name__ == "__main__":
    benchmark_reduction()
```

---

## 🐛 Common Issues

### Issue: Race condition in histogram
**Fix:** Use `cuda.atomic.add()` for all updates

### Issue: Incorrect warp reduction
**Fix:** Ensure all threads participate in shuffle

---

## 📚 Additional Resources
- [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [Parallel Reduction](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
