# Day 15: High-Performance Linear Algebra with cuBLAS
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 3: GPU Libraries

---

> **🎯 Focus Area:** Master NVIDIA's cuBLAS library for maximizing matrix operation performance and leveraging Tensor Cores.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Understand** BLAS levels (1, 2, 3) and memory layouts (column-major vs row-major).
2. **Implement** high-performance matrix operations using `cuBLAS`.
3. **Utilize** batched operations for processing multiple small matrices efficiently.
4. **Leverage** Tensor Cores automatically via library calls.
5. **Integrate** cuBLAS with C++ and Python (CuPy) applications.
6. **Benchmark** library performance against custom kernels.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Pascal or later recommended)
- Tensor Core support (Volta, Turing, Ampere, Hopper) for advanced features

### Software Environment
```bash
# Check CUDA version (cuBLAS is included)
nvcc --version

# Python environment
pip install cupy-cuda12x numpy scipy matplotlib

# Verify cuBLAS from Python
python -c "import cupy; print(cupy.cuda.cublas.get_version())"
```

### Prior Knowledge
- Day 7: Matrix Multiplication Project (Naive/Shared implementation)
- Basic Linear Algebra (Vectors, Matrices, Dot Products)

---

## 📖 Theoretical Foundation

### 1. The BLAS Standard

BLAS (Basic Linear Algebra Subprograms) is the de facto standard for low-level vector and matrix operations.

| Level | Operation Type | Complexity | Example | Description |
|-------|----------------|------------|---------|-------------|
| **Level 1** | Vector-Vector | O(N) | `y = αx + y` | Scalar scaling, dot products, vector addition (AXPY). |
| **Level 2** | Matrix-Vector | O(N²) | `y = αAx + βy` | Matrix-vector multiplication (GEMV). |
| **Level 3** | Matrix-Matrix | O(N³) | `C = αAB + βC` | Matrix-matrix multiplication (GEMM). Most compute-intensive. |

### 2. Memory Layout: Column-Major vs Row-Major

One of the biggest friction points in CUDA programming is data layout.

*   **C/C++/Python (NumPy):** **Row-major**. Consecutive elements in a row are contiguous in memory.
*   **Fortran/MATLAB/cuBLAS:** **Column-major**. Consecutive elements in a column are contiguous.

```
Row-Major (2x3 Matrix):
[ 1, 2, 3 ]
[ 4, 5, 6 ]
Memory: 1, 2, 3, 4, 5, 6

Column-Major (2x3 Matrix):
[ 1, 2, 3 ]
[ 4, 5, 6 ]
Memory: 1, 4, 2, 5, 3, 6
```

**Handling the mismatch:**
Since $C = A \times B$ is equivalent to $C^T = B^T \times A^T$, we can trick cuBLAS into computing row-major multiplications by swapping inputs and transposing the logical operation.

### 3. Tensor Cores and Math Modes

Modern Nvidia GPUs (Volta+) have Tensor Cores: specialized hardware pipelines performing $D = A \times B + C$ on mixed-precision small matrices (e.g., 16x16) in a single cycle.

*   **FP32 (Single):** Standard precision.
*   **TF32 (Tensor Float 32):** Ampere+ feature. 19-bit precision range of FP32, precision of FP16. High speed, drop-in replacement.
*   **FP16 (Half):** High speed, lower dynamic range.

cuBLAS handles these automatically if configured correctly, often yielding 8x-16x speedups over standard CUDA cores.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: C++ Low-Level API

This example demonstrates the raw C++ API for `sgemm` (Single-precision General Matrix Multiply).

#### 📁 `src/cublas_sgemm.cpp`
```cpp
/*
 * Day 15: cuBLAS SGEMM Example
 * Phase 6: AI/ML Platform Engineering
 *
 * Demonstrates basic Matrix-Matrix multiplication using cuBLAS
 * handling the Row-Major vs Column-Major difference.
 * 
 * Compile: nvcc -o cublas_sgemm cublas_sgemm.cpp -lcublas
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>

// Error checking macros
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(func) { \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void print_matrix(const float* A, int rows, int cols, const char* name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << A[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // 1. Setup - Matrix Dimensions
    // C = A * B
    // A: m x k
    // B: k x n
    // C: m x n
    int m = 3;
    int k = 4;
    int n = 2;
    
    float alpha = 1.0f;
    float beta  = 0.0f;

    std::cout << "Dimensions: M=" << m << ", K=" << k << ", N=" << n << std::endl;

    // 2. Host Data Initialization (Row-Major)
    std::vector<float> h_A = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };

    std::vector<float> h_B = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    std::vector<float> h_C(m * n, 0.0f);

    print_matrix(h_A.data(), m, k, "Host Matrix A");
    print_matrix(h_B.data(), k, n, "Host Matrix B");

    // 3. Device Allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    // 4. Memory Copy
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    // 5. cuBLAS Handle Creation
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 6. Perform SGEMM
    // Key Insight: cuBLAS expects Column-Major. 
    // We have Row-Major data A and B.
    // If we pass our Row-Major pointers to cuBLAS, it interprets them as Column-Major A^T and B^T.
    // Computing C = B * A in cuBLAS logic (which effectively does A * B in Row-Major)
    // Formula: C^T = B^T * A^T
    // Therefore, we tell cuBLAS:
    //   Do GEMM(B, A) 
    //   Output is C (which will be Column-Major C^T, which matches Row-Major C memory)
    
    // Arguments:
    // handle
    // CUBLAS_OP_N (No transpose of B "logical")
    // CUBLAS_OP_N (No transpose of A "logical")
    // n (rows of "logical result" B*A which is C^T -> rows are n)
    // m (cols of "logical result" B*A which is C^T -> cols are m)
    // k (shared dimension)
    // alpha
    // d_B (second matrix becomes first)
    // n (leading dimension of B)
    // d_A (first matrix becomes second)
    // k (leading dimension of A)
    // beta
    // d_C (result)
    // n (leading dimension of C)
    
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_B, n,
        d_A, k,
        &beta,
        d_C, n
    ));


    // 7. Retrieve Result
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // 8. Output
    print_matrix(h_C.data(), m, n, "Result Matrix C (A x B)");

    // 9. Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```

### 👨‍💻 Python Implementation: CuPy vs cuBLAS

Python's `cupy` wraps cuBLAS automatically for basic ops, but we can access raw cuBLAS handles for optimization tuning.

#### 📁 `src/cublas_optimization.py`
```python
#!/usr/bin/env python3
"""
Day 15: cuBLAS Optimization and Batched Operations
Phase 6: AI/ML Platform Engineering

Benchmarks standard matrix multiplication vs CUDA kernels vs Batched Operations.
"""

import cupy as cp
import numpy as np
import time
from cupy import cublas

# ============================================================================
# 1. NAIVE KERNEL for Comparison
# ============================================================================
naive_matmul_kernel = cp.RawKernel(r'''
extern "C" __global__
void naive_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
''', 'naive_matmul')

def run_naive_gemm(A, B):
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    
    C = cp.zeros((m, n), dtype=cp.float32)
    
    block_size = (32, 32)
    grid_size = ((n + 31) // 32, (m + 31) // 32)
    
    naive_matmul_kernel(grid_size, block_size, (A, B, C, m, k, n))
    return C

# ============================================================================
# 2. cuBLAS Standard GEMM (via CuPy's matmul)
# ============================================================================
def run_cublas_gemm(A, B):
    # CuPy internally uses cuBLAS sgemm/dgemm
    return cp.matmul(A, B)

# ============================================================================
# 3. Batched GEMM
# ============================================================================
def run_batched_gemm_loop(matrices_A, matrices_B):
    """Slow way: Loop in Python"""
    results = []
    for a, b in zip(matrices_A, matrices_B):
        results.append(cp.matmul(a, b))
    return cp.array(results)

def run_batched_gemm_native(matrices_A, matrices_B):
    """Fast way: cublas<t>gemmStridedBatched via CuPy slicing"""
    # CuPy automatically detects 3D tensors (Batch, M, K) @ (Batch, K, N)
    # and calls cublasSgemmStridedBatched
    return cp.matmul(matrices_A, matrices_B)

# ============================================================================
# 4. Benchmarking
# ============================================================================
def benchmark():
    print("=" * 60)
    print("BENCHMARK: Matrix Multiplication Methods")
    print("=" * 60)
    
    # Setup Large Matrices
    M, K, N = 4096, 4096, 4096
    print(f"Size: {M}x{K} x {K}x{N} (Single Matrix)")
    
    A = cp.random.rand(M, K, dtype=cp.float32)
    B = cp.random.rand(K, N, dtype=cp.float32)
    
    # 1. Naive Kernel
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    run_naive_gemm(A, B)
    end.record(); end.synchronize()
    print(f"Naive Kernel:       {cp.cuda.get_elapsed_time(start, end):.4f} ms")
    
    # 2. cuBLAS
    start.record()
    run_cublas_gemm(A, B)
    end.record(); end.synchronize()
    cublas_time = cp.cuda.get_elapsed_time(start, end)
    print(f"cuBLAS (Default):   {cublas_time:.4f} ms")
    
    # FLOPs calculation
    flops = 2 * M * N * K
    tflops = (flops / (cublas_time / 1000.0)) / 1e12
    print(f"cuBLAS Performance: {tflops:.2f} TFLOPS")
    
    print("\n" + "-" * 60)
    print("BATCHED OPERATIONS BENCHMARK")
    print("-" * 60)
    
    batch_size = 10000
    m, k, n = 32, 32, 32
    print(f"Configuration: {batch_size} batches of {m}x{k} x {k}x{n}")
    
    A_batch = cp.random.rand(batch_size, m, k, dtype=cp.float32)
    B_batch = cp.random.rand(batch_size, k, n, dtype=cp.float32)
    
    # 3. Python Loop
    start.record()
    # run_batched_gemm_loop(A_batch, B_batch) # Warning: Very slow, uncomment to see
    end.record(); end.synchronize()
    # print(f"Python Loop Batched: {cp.cuda.get_elapsed_time(start, end):.4f} ms")
    print(f"Python Loop Batched: (Skipped, typically >100x slower)")

    # 4. Strided Batched cuBLAS
    start.record()
    run_batched_gemm_native(A_batch, B_batch)
    end.record(); end.synchronize()
    batched_time = cp.cuda.get_elapsed_time(start, end)
    print(f"Strided Batched:    {batched_time:.4f} ms")
    
    ops_per_matrix = 2 * m * n * k
    total_ops = ops_per_matrix * batch_size
    print(f"Throughput:         {batch_size / (batched_time/1000.0):.0f} matrices/sec")

if __name__ == "__main__":
    benchmark()
```

### Advanced Topic: Tensor Core Math Modes

When initializing/using cuBLAS, you can allow it to use Tensor Cores (TF32) for FP32 inputs on Ampere GPUs. This is often default on newer CUDA versions but can be explicitly controlled.

```python
# In CuPy/Python
# Allow TF32 (Default is True on Ampere+)
cp.cuda.set_cublas_math_mode(cp.cuda.cublas.CUBLAS_TF32_TENSOR_OP_MATH)

# For strict FP32 (slower, higher precision)
# cp.cuda.set_cublas_math_mode(cp.cuda.cublas.CUBLAS_DEFAULT_MATH)
```

In C++:
```cpp
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
```

---

## 🔬 Lab Exercise: "MLP Layer forward pass with cuBLAS"

### Lab Objectives
1. Implement a Fully Connected (Dense) layer forward pass using cuBLAS.
2. Handle bias addition effectively using `cublasSgemm` (for weights) and a separate kernel or library call for bias.
3. Compare performance against a pure Python loop implementation.

### Implementation

#### 📁 `src/mlp_cublas.py`
```python
#!/usr/bin/env python3
"""
Lab: Implementing an MLP Layer with cuBLAS
Day 15
"""

import cupy as cp
import numpy as np
import time

class DenseLayerCuBLAS:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights (Xavier/Glorot)
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = cp.random.uniform(-limit, limit, (in_features, out_features), dtype=cp.float32)
        self.b = cp.zeros((out_features,), dtype=cp.float32)
        
    def forward(self, X):
        """
        X: (Batch_Size, In_Features)
        Output: (Batch_Size, Out_Features)
        Equation: Y = X * W + b
        """
        # 1. Matrix Multiplication: X * W
        # CuPy's dot/matmul uses cuBLAS
        Y = cp.matmul(X, self.W)
        
        # 2. Bias Addition
        # Broadcasting handles this efficiently, but strictly speaking
        # creating a custom kernel or using cublasSaxpy is lower level.
        # Here we rely on CuPy's broadcasting which launches an elementwise kernel.
        Y += self.b
        
        # 3. Activation (ReLU)
        # Elementwise MAX
        return cp.maximum(Y, 0)

class DenseLayerNaive:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.uniform(-1, 1, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros((out_features,), dtype=np.float32)
        
    def forward(self, X):
        # Naive NumPy Implementation
        Y = np.dot(X, self.W) + self.b
        return np.maximum(Y, 0)

def run_lab():
    print("=" * 60)
    print("LAB: MLP Forward Pass Performance")
    print("=" * 60)
    
    batch_size = 8192
    in_dim = 4096
    out_dim = 2048
    
    print(f"Network: {in_dim} -> {out_dim} | Batch: {batch_size}")
    
    # 1. CPU / NumPy
    print("\nInitializing CPU model...")
    cpu_layer = DenseLayerNaive(in_dim, out_dim)
    X_cpu = np.random.randn(batch_size, in_dim).astype(np.float32)
    
    start_cpu = time.perf_counter()
    _ = cpu_layer.forward(X_cpu)
    end_cpu = time.perf_counter()
    print(f"CPU Time: {(end_cpu - start_cpu)*1000:.2f} ms")
    
    # 2. GPU / cuBLAS
    print("\nInitializing GPU model...")
    gpu_layer = DenseLayerCuBLAS(in_dim, out_dim)
    X_gpu = cp.asarray(X_cpu) # Copy data
    
    # Warmup
    _ = gpu_layer.forward(X_gpu)
    cp.cuda.Stream.null.synchronize()
    
    start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event()
    start_gpu.record()
    Y_gpu = gpu_layer.forward(X_gpu)
    end_gpu.record(); end_gpu.synchronize()
    
    print(f"GPU Time: {cp.cuda.get_elapsed_time(start_gpu, end_gpu):.2f} ms")
    
    speedup = (end_cpu - start_cpu) * 1000 / cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print(f"Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    run_lab()
```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Don't Reinvent the Wheel:** For linear algebra, `cuBLAS` is almost always faster than handwritten kernels due to assembly-level optimizations and hardware awareness.
2.  **Memory Layout Matters:** The row-major (C++) vs column-major (cuBLAS) impedance mismatch requires careful handling (often swapping A/B and transposing).
3.  **Use Batches:** Processing one small matrix at a time is inefficient. `cublas<t>gemmBatched` utilizes the massive parallelism of the GPU.
4.  **Math Modes:** Be aware of Tensor Core capabilities (TF32/FP16) which perform matrix math significantly faster at slight precision costs.

### API Summary
```cpp
// Create Handle
cublasCreate(&handle);

// Matrix Multiply (C = alpha*A*B + beta*C)
cublasSgemm(handle, opA, opB, m, n, k, ...);

// Vector Axpy (y = alpha*x + y)
cublasSaxpy(handle, n, alpha, x, incx, y, incy);

// Destroy Handle
cublasDestroy(handle);
```

### Pitfalls to Avoid
-   Frequent `cudaMalloc`/`cudaFree` inside loops (Allocate once, reuse buffers).
-   Mixing up M, N, K dimensions in `Sgemm` calls.
-   Ignoring the leading dimension (`lda`, `ldb`, `ldc`) parameters which deal with memory stride.

---

**Day 15 Complete** ✅

*Next: Day 16 - cuDNN Deep Neural Networks - Building convolutions and RNNs with NVIDIA's deep learning primitives!*
