# Day 6: Shared Memory Optimization - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Basic Shared Memory (Beginner)

### Objective
Use shared memory to speed up data reuse.

### Code

```python
from numba import cuda
import numpy as np
import numba

@cuda.jit
def shared_memory_demo(input_arr, output_arr):
    """Demonstrate shared memory usage."""
    # Declare shared memory
    shared = cuda.shared.array(256, dtype=numba.float32)
    
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    
    # Load to shared memory
    if idx < input_arr.size:
        shared[tid] = input_arr[idx]
    
    cuda.syncthreads()
    
    # Use shared memory (example: access neighbor)
    if idx < output_arr.size and tid > 0:
        output_arr[idx] = shared[tid] + shared[tid - 1]
    elif idx < output_arr.size:
        output_arr[idx] = shared[tid]

def run_shared_demo():
    """Run shared memory demonstration."""
    n = 1024
    input_arr = np.arange(n, dtype=np.float32)
    output_arr = np.zeros(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    
    d_input = cuda.to_device(input_arr)
    d_output = cuda.device_array(n, dtype=np.float32)
    
    shared_memory_demo[blocks, threads](d_input, d_output)
    
    result = d_output.copy_to_host()
    print(f"Input:  {input_arr[:10]}")
    print(f"Output: {result[:10]}")

if __name__ == "__main__":
    run_shared_demo()
```

---

## 🔬 Exercise 2: Matrix Transpose with Shared Memory (Intermediate)

### Objective
Optimize matrix transpose using shared memory tiling.

### Code

```python
from numba import cuda
import numpy as np
import numba
import time

TILE_SIZE = 32

@cuda.jit
def transpose_naive(input_mat, output_mat):
    """Naive transpose - poor memory access pattern."""
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if row < input_mat.shape[0] and col < input_mat.shape[1]:
        output_mat[col, row] = input_mat[row, col]

@cuda.jit
def transpose_shared(input_mat, output_mat):
    """Optimized transpose using shared memory."""
    # Shared memory tile (+1 to avoid bank conflicts)
    tile = cuda.shared.array((TILE_SIZE, TILE_SIZE + 1), dtype=numba.float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # Input coordinates
    row = by * TILE_SIZE + ty
    col = bx * TILE_SIZE + tx
    
    # Load tile to shared memory (coalesced read)
    if row < input_mat.shape[0] and col < input_mat.shape[1]:
        tile[ty, tx] = input_mat[row, col]
    
    cuda.syncthreads()
    
    # Output coordinates (transposed block position)
    row = bx * TILE_SIZE + ty
    col = by * TILE_SIZE + tx
    
    # Write from shared memory (coalesced write)
    if row < output_mat.shape[0] and col < output_mat.shape[1]:
        output_mat[row, col] = tile[tx, ty]

def benchmark_transpose():
    """Compare naive vs optimized transpose."""
    N = 4096
    
    input_mat = np.random.rand(N, N).astype(np.float32)
    output_naive = np.zeros((N, N), dtype=np.float32)
    output_shared = np.zeros((N, N), dtype=np.float32)
    
    d_input = cuda.to_device(input_mat)
    d_naive = cuda.device_array((N, N), dtype=np.float32)
    d_shared = cuda.device_array((N, N), dtype=np.float32)
    
    threads = (TILE_SIZE, TILE_SIZE)
    blocks = ((N + TILE_SIZE - 1) // TILE_SIZE, 
              (N + TILE_SIZE - 1) // TILE_SIZE)
    
    # Warm up
    transpose_naive[blocks, threads](d_input, d_naive)
    transpose_shared[blocks, threads](d_input, d_shared)
    cuda.synchronize()
    
    # Benchmark
    iterations = 100
    
    start = time.perf_counter()
    for _ in range(iterations):
        transpose_naive[blocks, threads](d_input, d_naive)
    cuda.synchronize()
    naive_time = (time.perf_counter() - start) / iterations * 1000
    
    start = time.perf_counter()
    for _ in range(iterations):
        transpose_shared[blocks, threads](d_input, d_shared)
    cuda.synchronize()
    shared_time = (time.perf_counter() - start) / iterations * 1000
    
    print(f"Matrix Transpose Benchmark ({N}x{N})")
    print("=" * 40)
    print(f"Naive:  {naive_time:.3f} ms")
    print(f"Shared: {shared_time:.3f} ms")
    print(f"Speedup: {naive_time/shared_time:.2f}x")
    
    # Verify correctness
    expected = input_mat.T
    result = d_shared.copy_to_host()
    print(f"Correct: {np.allclose(expected, result)}")

if __name__ == "__main__":
    benchmark_transpose()
```

---

## 🔬 Exercise 3: Bank Conflict Analysis (Advanced)

### Objective
Understand and avoid shared memory bank conflicts.

### Code

```python
from numba import cuda
import numpy as np
import numba
import time

@cuda.jit
def with_bank_conflicts(shared_data, output):
    """Access pattern that causes bank conflicts."""
    shared = cuda.shared.array(1024, dtype=numba.float32)
    tid = cuda.threadIdx.x
    
    # Column-major access -> bank conflicts!
    shared[tid] = tid
    cuda.syncthreads()
    
    if tid < 32:
        # Stride-32 access - all threads hit same bank
        val = shared[tid * 32]
        output[tid] = val

@cuda.jit  
def without_bank_conflicts(shared_data, output):
    """Access pattern that avoids bank conflicts."""
    # Add padding to avoid conflicts
    shared = cuda.shared.array(1024 + 32, dtype=numba.float32)
    tid = cuda.threadIdx.x
    
    shared[tid] = tid
    cuda.syncthreads()
    
    if tid < 32:
        # With padding, stride access avoids conflicts
        val = shared[tid * 33]  # 33 instead of 32
        output[tid] = val

def benchmark_bank_conflicts():
    """Measure impact of bank conflicts."""
    iterations = 10000
    
    d_data = cuda.device_array(1024, dtype=np.float32)
    d_output = cuda.device_array(32, dtype=np.float32)
    
    # Warm up
    with_bank_conflicts[1, 1024](d_data, d_output)
    without_bank_conflicts[1, 1024](d_data, d_output)
    cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        with_bank_conflicts[1, 1024](d_data, d_output)
    cuda.synchronize()
    conflict_time = (time.perf_counter() - start) / iterations * 1e6
    
    start = time.perf_counter()
    for _ in range(iterations):
        without_bank_conflicts[1, 1024](d_data, d_output)
    cuda.synchronize()
    no_conflict_time = (time.perf_counter() - start) / iterations * 1e6
    
    print("Bank Conflict Analysis")
    print("=" * 40)
    print(f"With conflicts:    {conflict_time:.2f} µs")
    print(f"Without conflicts: {no_conflict_time:.2f} µs")

if __name__ == "__main__":
    benchmark_bank_conflicts()
```

---

## 🐛 Common Issues

### Issue 1: Insufficient shared memory
```
numba.cuda.cudadrv.driver.CudaAPIError: Call to cuLaunchKernel results in CUDA_ERROR_LAUNCH_FAILED
```
**Fix:** Reduce shared memory size or use smaller tiles.

### Issue 2: Race condition in shared memory
**Fix:** Always call `cuda.syncthreads()` between write and read phases.

---

## 📚 Additional Resources
- [CUDA Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Bank Conflicts Explained](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
