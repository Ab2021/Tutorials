# Day 4: Thread Synchronization - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Understanding Warps (Beginner)

### Objective
Visualize warp execution and branch divergence.

### Code

```python
from numba import cuda
import numpy as np

@cuda.jit
def warp_demo_kernel(output, thread_info):
    """Demonstrate warp execution."""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    warp_id = tid // 32
    lane_id = tid % 32
    
    # Global index
    idx = bid * cuda.blockDim.x + tid
    
    if idx < output.size:
        # Store thread info
        thread_info[idx, 0] = tid
        thread_info[idx, 1] = warp_id
        thread_info[idx, 2] = lane_id

def visualize_warps():
    """Visualize warp structure."""
    threads_per_block = 128
    blocks = 1
    total_threads = threads_per_block * blocks
    
    output = np.zeros(total_threads, dtype=np.int32)
    thread_info = np.zeros((total_threads, 3), dtype=np.int32)
    
    d_output = cuda.to_device(output)
    d_info = cuda.to_device(thread_info)
    
    warp_demo_kernel[blocks, threads_per_block](d_output, d_info)
    
    thread_info = d_info.copy_to_host()
    
    print("Warp Structure Visualization")
    print("=" * 50)
    print(f"Block has {threads_per_block} threads = {threads_per_block//32} warps")
    print()
    
    for warp in range(threads_per_block // 32):
        mask = thread_info[:, 1] == warp
        lanes = thread_info[mask, 2]
        print(f"Warp {warp}: Lanes {lanes[0]}-{lanes[-1]} (32 threads)")

if __name__ == "__main__":
    visualize_warps()
```

---

## 🔬 Exercise 2: Barrier Synchronization (Intermediate)

### Objective
Use `__syncthreads()` for block-level synchronization.

### Code

```python
from numba import cuda
import numpy as np

@cuda.jit
def reduction_with_sync(data, result):
    """Parallel reduction using shared memory and sync."""
    # Shared memory for this block
    shared = cuda.shared.array(256, dtype=numba.float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    idx = bid * cuda.blockDim.x + tid
    
    # Load data to shared memory
    if idx < data.size:
        shared[tid] = data[idx]
    else:
        shared[tid] = 0.0
    
    # CRITICAL: Wait for all threads to load
    cuda.syncthreads()
    
    # Parallel reduction
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()  # Wait before next iteration
        stride //= 2
    
    # Thread 0 writes result
    if tid == 0:
        result[bid] = shared[0]

def parallel_sum(data):
    """Sum array using parallel reduction."""
    n = len(data)
    threads = 256
    blocks = (n + threads - 1) // threads
    
    d_data = cuda.to_device(data.astype(np.float32))
    d_result = cuda.device_array(blocks, dtype=np.float32)
    
    reduction_with_sync[blocks, threads](d_data, d_result)
    
    # Sum partial results on CPU (or do another reduction)
    return d_result.copy_to_host().sum()

if __name__ == "__main__":
    import numba
    
    data = np.ones(10000, dtype=np.float32)
    
    gpu_sum = parallel_sum(data)
    cpu_sum = data.sum()
    
    print(f"CPU sum: {cpu_sum}")
    print(f"GPU sum: {gpu_sum}")
    print(f"Match: {np.isclose(gpu_sum, cpu_sum)}")
```

---

## 🔬 Exercise 3: Branch Divergence Analysis (Advanced)

### Objective
Measure performance impact of branch divergence.

### Code

```python
from numba import cuda
import numpy as np
import time

@cuda.jit
def no_divergence_kernel(output):
    """All threads take same path."""
    idx = cuda.grid(1)
    if idx < output.size:
        # All threads do this
        output[idx] = idx * 2.0

@cuda.jit
def divergent_kernel(output):
    """Threads take different paths based on lane ID."""
    idx = cuda.grid(1)
    lane = cuda.threadIdx.x % 32
    
    if idx < output.size:
        if lane < 16:
            # First half of warp
            output[idx] = idx * 2.0
        else:
            # Second half of warp (DIVERGENCE!)
            output[idx] = idx * 3.0

def benchmark_divergence():
    """Compare divergent vs non-divergent kernels."""
    n = 10_000_000
    iterations = 100
    
    output = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    
    # Warm up
    no_divergence_kernel[blocks, threads](output)
    divergent_kernel[blocks, threads](output)
    cuda.synchronize()
    
    # Benchmark non-divergent
    start = time.perf_counter()
    for _ in range(iterations):
        no_divergence_kernel[blocks, threads](output)
    cuda.synchronize()
    no_div_time = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark divergent
    start = time.perf_counter()
    for _ in range(iterations):
        divergent_kernel[blocks, threads](output)
    cuda.synchronize()
    div_time = (time.perf_counter() - start) / iterations * 1000
    
    print("Branch Divergence Impact")
    print("=" * 40)
    print(f"No divergence: {no_div_time:.3f} ms")
    print(f"With divergence: {div_time:.3f} ms")
    print(f"Slowdown: {div_time/no_div_time:.2f}x")

if __name__ == "__main__":
    benchmark_divergence()
```

---

## 🐛 Common Errors & Fixes

### Error 1: Race condition
**Symptom:** Different results each run
**Fix:** Add `cuda.syncthreads()` before reading shared data

### Error 2: Deadlock
**Symptom:** Kernel hangs forever
**Fix:** Ensure ALL threads in block reach `syncthreads()`:
```python
# Bad - some threads might not reach sync
if condition:
    cuda.syncthreads()

# Good - all threads reach sync
cuda.syncthreads()
if condition:
    ...
```

---

## 📚 Additional Resources
- [CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
