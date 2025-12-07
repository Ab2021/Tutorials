# Day 2: CUDA Programming Model - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Thread Indexing (Beginner)

### Objective
Understand how CUDA threads are indexed within blocks and grids.

### Code

```python
import torch

def visualize_thread_indexing():
    """Visualize how threads are organized."""
    
    # Simulate a 2D grid of 2x2 blocks, each block has 4x4 threads
    grid_dim = (2, 2)
    block_dim = (4, 4)
    
    print("Thread Indexing Visualization")
    print("=" * 50)
    print(f"Grid: {grid_dim[0]}x{grid_dim[1]} blocks")
    print(f"Block: {block_dim[0]}x{block_dim[1]} threads")
    print(f"Total threads: {grid_dim[0] * grid_dim[1] * block_dim[0] * block_dim[1]}")
    print()
    
    for block_y in range(grid_dim[1]):
        for block_x in range(grid_dim[0]):
            print(f"Block ({block_x}, {block_y}):")
            for thread_y in range(block_dim[1]):
                row = []
                for thread_x in range(block_dim[0]):
                    # Global thread ID
                    global_x = block_x * block_dim[0] + thread_x
                    global_y = block_y * block_dim[1] + thread_y
                    row.append(f"({global_x},{global_y})")
                print("  " + " ".join(row))
            print()

if __name__ == "__main__":
    visualize_thread_indexing()
```

### Expected Output
```
Thread Indexing Visualization
==================================================
Grid: 2x2 blocks
Block: 4x4 threads
Total threads: 64

Block (0, 0):
  (0,0) (1,0) (2,0) (3,0)
  (0,1) (1,1) (2,1) (3,1)
  ...
```

---

## 🔬 Exercise 2: Vector Addition Kernel (Intermediate)

### Objective
Implement parallel vector addition using PyTorch's custom CUDA extension.

### Code (Using Numba for simplicity)

```python
from numba import cuda
import numpy as np
import math

@cuda.jit
def vector_add_kernel(a, b, c):
    """CUDA kernel for vector addition."""
    # Calculate global thread ID
    idx = cuda.grid(1)
    
    # Bounds check
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

def vector_add_gpu(a, b):
    """Add two vectors on GPU."""
    n = len(a)
    
    # Allocate device memory
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array(n, dtype=np.float32)
    
    # Configure kernel
    threads_per_block = 256
    blocks_per_grid = math.ceil(n / threads_per_block)
    
    # Launch kernel
    vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    
    # Copy result back
    return d_c.copy_to_host()

if __name__ == "__main__":
    # Test
    n = 1_000_000
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    
    # GPU
    c_gpu = vector_add_gpu(a, b)
    
    # CPU (for verification)
    c_cpu = a + b
    
    # Verify
    print(f"Max difference: {np.max(np.abs(c_gpu - c_cpu))}")
    print("✓ Results match!" if np.allclose(c_gpu, c_cpu) else "✗ Mismatch!")
```

---

## 🔬 Exercise 3: Grid-Stride Loops (Advanced)

### Objective
Implement efficient processing of large arrays using grid-stride loops.

### Code

```python
from numba import cuda
import numpy as np

@cuda.jit
def grid_stride_kernel(data, result):
    """Process array larger than grid size using grid-stride loop."""
    # Starting index for this thread
    start = cuda.grid(1)
    # Stride = total number of threads
    stride = cuda.gridsize(1)
    
    # Grid-stride loop
    for i in range(start, data.size, stride):
        result[i] = data[i] * 2.0  # Simple operation

def benchmark_grid_stride():
    """Compare grid-stride vs naive approach."""
    import time
    
    n = 100_000_000  # 100M elements
    data = np.ones(n, dtype=np.float32)
    result = np.zeros(n, dtype=np.float32)
    
    d_data = cuda.to_device(data)
    d_result = cuda.device_array(n, dtype=np.float32)
    
    # Fixed grid size (smaller than data)
    threads = 256
    blocks = 1024  # Only 256K threads, but processing 100M elements
    
    # Warm up
    grid_stride_kernel[blocks, threads](d_data, d_result)
    cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        grid_stride_kernel[blocks, threads](d_data, d_result)
        cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10
    
    print(f"Processed {n:,} elements in {elapsed*1000:.2f} ms")
    print(f"Throughput: {n * 4 / elapsed / 1e9:.2f} GB/s")

if __name__ == "__main__":
    benchmark_grid_stride()
```

---

## 🐛 Common Errors & Fixes

### Error 1: Kernel launch failure
```
numba.cuda.cudadrv.driver.CudaAPIError: Call to cuLaunchKernel results in CUDA_ERROR_LAUNCH_FAILED
```
**Fix:** Check array bounds in kernel. Add `if idx < array.size:` guard.

### Error 2: Wrong results
```
Max difference: 1.234
```
**Fix:** Ensure `cuda.synchronize()` before copying data back.

---

## 📚 Additional Resources
- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
