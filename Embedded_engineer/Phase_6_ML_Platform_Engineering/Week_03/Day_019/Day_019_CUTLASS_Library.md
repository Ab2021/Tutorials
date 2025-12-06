# Day 19: CUTLASS - Customizing Tensor Core Operations
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 3: GPU Libraries

---

> **🎯 Focus Area:** Step beyond black-box libraries and master **CUTLASS** (CUDA Templates for Linear Algebra Subroutines) to generate custom, peak-performance kernels for Deep Learning and GEMM.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the limitations of closed-source libraries like cuBLAS for custom fused operations.
2.  **Understand** the CUTLASS decomposition of matrix multiplication (Threadblock, Warp, Thread levels).
3.  **Implement** a custom GEMM kernel using CUTLASS templates.
4.  **Perform** operator fusion (e.g., GEMM + Bias + ReLU) efficiently.
5.  **Target** specific hardware features (Tensor Cores) explicitly via C++ templates.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with Tensor Cores (Volta V100, Turing T4, Ampere A100+, Ada L40/4090) required for meaningful usage.

### Software Environment
- **CUTLASS Library:** It is a header-only template library.
```bash
# Clone CUTLASS (Required for compilation)
git clone https://github.com/NVIDIA/cutlass.git
export CUTLASS_PATH=$(pwd)/cutlass
```
- **NVCC:** C++17 support required (CUDA 11.0+).

### Prior Knowledge
- Day 7: Tiling and Shared Memory.
- Day 15: GEMM Theory.
- C++ Templates (Conceptual understanding).

---

## 📖 Theoretical Foundation

### 1. Why CUTLASS?
**cuBLAS** is amazing but "Closed." If you want to compute $D = \text{ReLU}(A \times B + C)$, you typically launch two kernels:
1.  GEMM (cuBLAS) $\to$ Memory Write.
2.  ReLU (Custom) $\to$ Memory Read $\to$ Compute $\to$ Memory Write.

Writing to global memory is expensive. **CUTLASS** allows you to "Fuse" the epilogue (ReLU) directly into the register file of the GEMM kernel, saving huge bandwidth. It powers fast inference engines like **vLLM** and **TensorRT-LLM**.

### 2. The Anatomy of a High-Performance GEMM
To reach close to 100% of GPU compute peaks, a GEMM kernel is structured into a hierarchy of tiles:

1.  **Global Memory:** The big matrices $A, B, C$.
2.  **Threadblock Tile:** A large chunk (e.g., $128 \times 128$) loaded into Shared Memory.
3.  **Warp Tile:** A sub-chunk (e.g., $64 \times 64$) processed by a Warp (32 threads).
4.  **Instruction Tile (MMA):** The fundamental unit computed by Tensor Cores (e.g., $16 \times 8 \times 16$).

CUTLASS provides C++ templates for each layer, allowing you to mix and match strategies.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Basic CUTLASS GEMM

This example sets up a standard SGEMM ($C = A \times B$) using CUTLASS. While verbose, it compiles down to highly optimized machine code comparable to handwritten assembly.

#### 📁 `src/cutlass_basic.cu`
```cpp
/*
 * Day 19: Basic CUTLASS GEMM
 * Phase 6: AI/ML Platform Engineering
 *
 * Demonstrates instantiating a GEMM kernel using CUTLASS templates.
 * Compile: nvcc -o cutlass_basic cutlass_basic.cu -I$CUTLASS_PATH/include -lcuda
 */

#include <iostream>
#include <vector>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Error checking
#define CHECK_CUTLASS(status) { \
    if (status != cutlass::Status::kSuccess) { \
        std::cerr << "CUTLASS Error at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUDA(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(1); \
    } \
}

int main() {
    // 1. Define Problem Size
    int M = 512;
    int N = 512;
    int K = 512;

    long long int lda = M;
    long long int ldb = K;
    long long int ldc = M;

    // 2. Define GEMM Type using Templates
    // <ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
    using Gemm = cutlass::gemm::device::Gemm<
        float,                    // Element A
        cutlass::layout::ColumnMajor, // Layout A (Standard Linear Algebra)
        float,                    // Element B
        cutlass::layout::ColumnMajor, // Layout B
        float,                    // Element C
        cutlass::layout::ColumnMajor  // Layout C
    >;

    // 3. Allocate Memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    // Initialize (Arbitrary values)
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Configure CUTLASS Arguments
    float alpha = 1.0f;
    float beta = 0.0f;

    Gemm::Arguments args(
        {M, N, K},          // Problem size
        {d_A, lda},         // A pointer & stride
        {d_B, ldb},         // B pointer & stride
        {d_C, ldc},         // C pointer & stride
        {d_C, ldc},         // D pointer (Output C)
        {alpha, beta}       // Scalars
    );

    // 5. Initialize & Run
    Gemm gemm_op;
    
    // Query workspace
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void *d_workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    }

    CHECK_CUTLASS(gemm_op.initialize(args, d_workspace));
    CHECK_CUTLASS(gemm_op.run());

    // 6. Verify
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "CUTLASS GEMM Complete." << std::endl;
    std::cout << "C[0] = " << h_C[0] << " (Expected: " << (float)K << ")" << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    if (d_workspace) cudaFree(d_workspace);

    return 0;
}
```

### 👨‍💻 Advanced: Epilogue Fusion (The Real Power)

Standard GEMM computes $C = \alpha AB + \beta C$.
CUTLASS allows custom "Epilogues". A common pattern is **Linear + ReLU**: $D = \max(0, \alpha AB + \beta C + \text{Bias})$.

This involves changing the template arguments to the `Gemm` class.

```cpp
// Pseudocode for Fused GEMM + ReLU
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationRelu<...>;

using GemmFused = cutlass::gemm::device::Gemm<
    ...,
    EpilogueOutputOp // Pass the custom op here
>;

// When running, the kernel loads A, B, computes product, adds Bias, applies ReLU
// in shared mem/registers BEFORE writing to Global Memory C.
// Speedup = Memory Bandwidth Savings.
```

---

## 🔬 Lab Exercise: "Benchmarking Custom Kernels"

### Lab Objectives
1.  Compile the provided `cutlass_basic.cu`.
2.  Modify the code to use **Tensor Cores** (Switch element types to `cutlass::half_t`).
3.  Compare performance against `cuBLAS`.

### Steps:
1.  **Switch to FP16:**
    Change `float` to `cutlass::half_t` in the `Gemm` definition.
    Ensure your GPU supports FP16 Tensor Cores (Volta+).
2.  **Increase Size:**
    Set $M=4096, N=4096, K=4096$. Small matrices don't saturate Tensor Cores.
3.  **Benchmark:**
    Use `cudaEvent` to time the `.run()` call.
    Compare with the Day 15 `cublas_sgemm` times.

### Expected Outcome
CUTLASS should match cuBLAS performance within 95-99% for standard GEMM. However, if you add Fusion (Bias+ReLU) in a separate kernel for cuBLAS vs the Fused CUTLASS kernel, CUTLASS will win significantly (1.5x - 2x speedup on memory-bound workloads).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Templates > Libraries:** For fixed, standard math, use libraries (cuBLAS). For research/custom math (e.g., 4-bit quantization, strange activations), use Templates (CUTLASS).
2.  **Epilogue Fusion:** The killer feature. Doing math is virtually free; moving data is expensive. Fusion keeps data in registers.
3.  **Tiling:** Understanding global/shared/warp/mma hierarchical tiling is the key to GPU performance tuning.
4.  **Header-Only:** No `.so` or `.dll` to link against (mostly). Just include and compile.

### Connections
*   **Previous:** Day 15 (cuBLAS) - The standard way.
*   **Next:** Day 20 (Thrust) - The "STL" of CUDA. High-level algorithms for the rest of us.

---

**Day 19 Complete** ✅
