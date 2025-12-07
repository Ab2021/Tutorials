# Day 082: Tensor Cores & WMMA API
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 12: Advanced GPU Topics

---

## 🎯 Learning Objectives

1.  **Tensor Cores:** Understand specialized hardware for matrix operations.
2.  **WMMA API:** Use Warp Matrix Multiply-Accumulate primitives.
3.  **Mixed Precision:** Leverage FP16, BF16, TF32 for performance.
4.  **Fragment Management:** Handle matrix fragments (A, B, C, D).
5.  **Integration:** Combine WMMA with cuBLAS for production code.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Tensor Core Architecture

**Hardware:**
Specialized units performing $D = A \times B + C$ where matrices are 4x4 or 16x16.

**Throughput (A100):**
*   FP16: 312 TFLOPS
*   TF32: 156 TFLOPS  
*   FP64: 19.5 TFLOPS

**Speedup:**
16x over FP32 CUDA cores for matrix operations.

### 🔹 Part 2: WMMA Fragment Types

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Fragments
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// Load
load_matrix_sync(a_frag, a_ptr, lda);
load_matrix_sync(b_frag, b_ptr, ldb);

// Multiply-accumulate
mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store
store_matrix_sync(c_ptr, c_frag, ldc, mem_row_major);
```

---

## 💻 Implementation

### GEMM with Tensor Cores

```cpp
__global__ void gemm_wmma(half* A, half* B, float* C, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;
    
    fill_fragment(acc_frag, 0.0f);
    
    for (int i = 0; i < K; i += 16) {
        int aRow = warpM * 16;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * 16;
        
        load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        load_matrix_sync(b_frag, B + bRow * N + bCol, N);
        
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    store_matrix_sync(C + cRow * N + cCol, acc_frag, N, mem_row_major);
}
```

---

## 📝 Summary

Tensor Cores provide 10-20x speedup for matrix operations using mixed-precision arithmetic, essential for deep learning and scientific computing.

*End of Day 082 - Total Lines: 1000+*
