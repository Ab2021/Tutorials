# Day 2: CUDA Programming Model
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 1: GPU Architecture & CUDA Foundations

---

> **🎯 Focus Area:** Understanding kernels, grids, blocks, and thread indexing - writing your first CUDA kernel.

---

## 🎯 Learning Objectives
1. **Understand** the CUDA programming model hierarchy
2. **Write** basic CUDA kernels with proper qualifiers
3. **Calculate** thread indices for parallel computation
4. **Debug** basic kernel launch issues

---

## 📖 Theoretical Foundation

### CUDA Execution Model

```
Host Code (CPU)                    Device Code (GPU)
     │                                   │
     │    kernel<<<grid, block>>>(args)  │
     │ ───────────────────────────────▶ │
     │                                   │
     │                          ┌────────┴────────┐
     │                          │   Grid of Blocks │
     │                          │  ┌────┬────┬────┐│
     │                          │  │Blk0│Blk1│Blk2││
     │                          │  │    │    │    ││
     │                          │  └────┴────┴────┘│
     │                          └─────────────────┘
```

### Function Type Qualifiers

| Qualifier | Executed On | Callable From |
|-----------|-------------|---------------|
| `__global__` | Device (GPU) | Host or Device |
| `__device__` | Device (GPU) | Device only |
| `__host__` | Host (CPU) | Host only |

### Thread Indexing

```cpp
// 1D Grid of 1D Blocks
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D Grid of 2D Blocks
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * gridDim.x * blockDim.x + x;

// 3D Grid of 3D Blocks
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

---

## 💻 Implementation

### Vector Addition Kernel (Classic First CUDA Program)

```cpp
// vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel: runs on GPU
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check - important for non-divisible sizes
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i) * sinf(i);
        h_B[i] = cosf(i) * cosf(i);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // Ceiling division
    
    printf("Launching kernel: gridSize=%d, blockSize=%d\n", gridSize, blockSize);
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    // Verify (spot check)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(h_C[i] - (h_A[i] + h_B[i])));
    }
    printf("Max error: %f\n", maxError);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
```

### Python Equivalent with CuPy

```python
#!/usr/bin/env python3
"""Day 2: CUDA Programming Model - Vector Addition with CuPy"""

import cupy as cp
import numpy as np
import time

# Raw CUDA kernel
vector_add_kernel = cp.RawKernel(r'''
extern "C" __global__
void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
''', 'vectorAdd')

def benchmark_vector_add():
    N = 1 << 24  # 16M elements
    
    # Allocate and initialize
    A_gpu = cp.random.randn(N, dtype=cp.float32)
    B_gpu = cp.random.randn(N, dtype=cp.float32)
    C_gpu = cp.zeros(N, dtype=cp.float32)
    
    # Configure launch
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    
    # Warmup
    vector_add_kernel((grid_size,), (block_size,), (A_gpu, B_gpu, C_gpu, N))
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    vector_add_kernel((grid_size,), (block_size,), (A_gpu, B_gpu, C_gpu, N))
    end.record()
    end.synchronize()
    
    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    
    # Calculate bandwidth
    bytes_processed = N * 4 * 3  # Read A, Read B, Write C
    bandwidth_gb_s = (bytes_processed / 1e9) / (elapsed_ms / 1000)
    
    print(f"Vector Addition ({N:,} elements)")
    print(f"  Time: {elapsed_ms:.3f} ms")
    print(f"  Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    
    # Verify
    expected = A_gpu + B_gpu
    assert cp.allclose(C_gpu, expected), "Verification failed!"
    print("  Verification: PASSED")

if __name__ == "__main__":
    benchmark_vector_add()
```

---

## 🔬 Lab Exercise: "Multi-Dimensional Thread Indexing"

### Objective
Implement a 2D matrix operation (element-wise operation on a matrix)

```python
import cupy as cp

matrix_kernel = cp.RawKernel(r'''
extern "C" __global__
void matrixOp(const float* A, float* B, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        B[idx] = A[idx] * 2.0f + 1.0f;  // Simple operation
    }
}
''', 'matrixOp')

# Launch with 2D grid and blocks
rows, cols = 1024, 1024
A = cp.random.randn(rows, cols, dtype=cp.float32)
B = cp.zeros_like(A)

block = (16, 16)  # 256 threads per block
grid = ((cols + block[0] - 1) // block[0], (rows + block[1] - 1) // block[1])

matrix_kernel(grid, block, (A, B, rows, cols))
```

---

## 🐞 Debugging Tips

### Common Errors
1. **"Invalid configuration argument"** - Block size exceeds limit (max 1024)
2. **"Too many resources requested"** - Reduce block size or register usage
3. **Incorrect results** - Check bounds, verify indexing logic

### Debug Print from Kernel
```cpp
__global__ void debugKernel(int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 5) {  // Only first 5 threads
        printf("Thread %d: blockIdx=%d, threadIdx=%d\n", 
               idx, blockIdx.x, threadIdx.x);
    }
}
```

---

## 📝 Daily Summary

### Key Takeaways
1. CUDA kernels are launched with `<<<grid, block>>>` syntax
2. Use `__global__` for kernels, `__device__` for device helper functions
3. Calculate global thread index: `blockIdx.x * blockDim.x + threadIdx.x`
4. Always include bounds checking in kernels
5. Use CUDA_CHECK macro for error handling

---

**Day 2 Complete** ✅
