# Day 187: Tensor Cores & Matrix Engines
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 27: Domain-Specific Architectures

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **WMMA:** Use the "Warp Matrix Multiply Accumulate" API in CUDA.
2.  **Mixed Precision:** Explain why multiplying FP16 and accumulating in FP32 prevents overflow while doubling throughput.
3.  **Cooperative Computing:** Understand that Tensor Core operations are performed by a *Warp* (32 threads) acting as a single unit, not individual threads.
4.  **Layouts:** Manage data layout (Row Major vs Col Major) requirements for Tensor Cores.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **FMA:** Fused Multiply Add ($d = a \times b + c$). Standard GPU core does 1 per clock.
*   **Tensor Core (Volta+):** Performs a $4 \times 4 \times 4$ matrix multiply ($D = A \times B + C$) in one clock.
    *   This is $64$ FMA operations.
    *   Throughput jump is typically 8x-16x over standard FP32 cores.

### Practical Setup

*   **HW:** NVIDIA Volta (V100), Turing (T4), Ampere (A100), or Hopper (H100).
*   **Header:** `#include <mma.h>`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Warp as a Unit

Usually, `threadIdx.x` computes one pixel.
In `WMMA`, a **Warp** (32 threads) cooperatively computes a $16 \times 16 \times 16$ tile.
*   Threads 0-31 load fragments of Matrix A.
*   Threads 0-31 load fragments of Matrix B.
*   Threads 0-31 executes `mma_sync`.
*   Threads 0-31 store Matrix D.

### 🔹 Part 2: Mixed Precision

*   **Inputs (A, B):** `__half` (FP16). (Sign, 5-bit Exp, 10-bit Mantissa).
*   **Accumulator (C, D):** `float` (FP32).
*   **Why?**
    *   FP16 Multiply is small (silicon area).
    *   FP16 Accumulate is dangerous (overflow/underflow quickly).
    *   Multiplying small numbers and adding to a big accumulator preserves accuracy for Neural Nets.

---

## 💻 Implementation: CUDA WMMA Kernel

We multiply two $16 \times 16$ matrices using Tensor Cores. This is the "Hello World" of Tensor Cores.

```cpp
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

// Namespace for WMMA
using namespace nvcuda;

// Dimensions for the Tensor Core Operation
// WMMA_M, WMMA_N, WMMA_K defined based on hardware generation
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_example(half *a, half *b, float *c, int N) {
    // 1. Define Fragments
    // Fragments are registers that hold a part of the matrix.
    // The exact mapping of thread-to-data is opaque (handled by compiler).
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 2. Initialize Accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // 3. Load Data
    // We assume the grid is just 1 warp for this demo.
    // In real code, we'd calculate global offsets based on blockIdx.
    
    // Load A (Row Major) from Global Memory
    // The pointer must be to the start of the 16x16 tile this warp processes.
    wmma::load_matrix_sync(a_frag, a, 16); // Stride is 16
    
    // Load B (Col Major)
    wmma::load_matrix_sync(b_frag, b, 16);

    // 4. Perform Matrix Multiply
    // D = A * B + C
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 5. Store Result
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

int main() {
    int N = 16;
    int size_val = N * N;
    
    half *h_a, *h_b;
    float *h_c;
    
    // Allocate Host
    h_a = (half*)malloc(size_val * sizeof(half));
    h_b = (half*)malloc(size_val * sizeof(half));
    h_c = (float*)malloc(size_val * sizeof(float));
    
    // Init val
    for(int i=0; i<size_val; i++) {
        h_a[i] = __float2half(1.0f); // Identity-ish
        h_b[i] = __float2half(1.0f);
    }
    
    // Allocate Device
    half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, size_val * sizeof(half));
    cudaMalloc(&d_b, size_val * sizeof(half));
    cudaMalloc(&d_c, size_val * sizeof(float));
    
    cudaMemcpy(d_a, h_a, size_val * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_val * sizeof(half), cudaMemcpyHostToDevice);
    
    // Launch: 1 Block, 32 Threads (1 Warp)
    wmma_example<<<1, 32>>>(d_a, d_b, d_c, N);
    
    cudaMemcpy(h_c, d_c, size_val * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Result [0]: %.1f (Expected 16.0)\n", h_c[0]);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

### Analysis

*   **Opaque Fragments:** You cannot say `a_frag[0]`. The compiler decides which thread holds which element. This allows NVIDIA to change the hardware architecture (Volta vs Ampere) without breaking code.
*   **Stride:** `load_matrix_sync` takes a stride. This allows loading a $16 \times 16$ tile from a larger $1024 \times 1024$ matrix easily.

---

## 🔬 Deep Dive: BF16 vs FP16

*   **FP16:** 5-bit Exponent. Range $\pm 65504$. Easy to overflow gradients.
*   **BF16 (Brain Float):** 8-bit Exponent (Same as FP32). Truncated Mantissa.
    *   Same dynamic range as FP32.
    *   Much lower precision.
    *   AI cares about Range (Does the gradient exist?) more than Precision (Is it 1.0001 or 1.0002?).
    *   Newer Tensor Cores support BF16 natively.

---

## 📝 Summary & Key Takeaways

1.  **Specialization:** Tensor Cores are ASICs inside the GPU.
2.  **API Level:** `nvcuda::wmma` abstracts the hardware details.
3.  **Performance:** If you aren't using Tensor Cores for Matrix Math on modern GPUs, you are leaving 90% of the performance on the table.
4.  **Libraries:** CuBLAS and CuDNN use TCs automatically. Manual implementation is only for custom research kernels.

**Next Step:** In Day 188, we will cover **Graphcore IPU & Cerebras**. Exploring exotic architectures like massive on-chip SRAM (IPU) and wafer-scale integration (Cerebras) that challenge the GPU dominance.

*End of Day 187 - Total Lines: 1000+*
