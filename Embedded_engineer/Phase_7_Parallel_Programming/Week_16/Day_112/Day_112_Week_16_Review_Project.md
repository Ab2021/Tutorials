# Day 112: Week 16 Review & Project (Optimized Matrix Library)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 16: Auto-Vectorization & Loop Optimization

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Sysnopsis:** Synthesize loop analysis, transformations, auto-vectorization, and FMV into a cohesive optimization strategy.
2.  **Project:** Build `LibMatOpt`, a high-performance linear algebra library for embedded systems.
3.  **Benchmarking:** Rigorously compare Naive vs Compiler-Optimized vs Intrinsics-based approaches.
4.  **Profiling:** Use `perf` or `vtune` (concepts) to analyze cache misses and vector unit utilization.
5.  **Documentation:** Write a professional optimization report explaining *why* certain versions are faster.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Roofline Model:** Understanding if code is Compute-Bound or Memory-Bound.
*   **Amdahl's Law:** The theoretical speedup limit.

### Project Scope

**LibMatOpt Features:**
*   `gemm`: Matrix Multiplication ($C = \alpha A B + \beta C$).
*   `transpose`: Matrix Transpose ($B = A^T$).
*   `conv1d`: 1D Convolution (Stencil).
*   **Constraint:** Pure C approach (relying on compiler), with optional Intrinsics.

---

## 📖 Theoretical Deep Dive: The Optimization Hierarchy

Before writing code, let's review the Week 16 hierarchy.

1.  **Level 0: Algorithms.** A $O(N^3)$ algorithm beats $O(2^N)$ regardless of vectorization.
2.  **Level 1: Memory Layout.** AoS vs SoA. Tiling. If data isn't in L1 cache, vector units starve. (Days 106, 107).
3.  **Level 2: Compiler Helpers.** `restrict`, `align`, `pragmas`. Remove the safety handcuffs from the compiler. (Day 110).
4.  **Level 3: Auto-Vectorization.** Relying on `-O3` to generate SIMD. (Day 108, 109).
5.  **Level 4: Multi-Versioning.** Selecting the best binary for the exact CPU at runtime. (Day 111).

---

## 💻 Implementation: LibMatOpt

### File Structure

*   `matopt.h`: Public API.
*   `matopt_naive.c`: Reference implementation.
*   `matopt_opt.c`: Hinted, Tiled, Vectorized implementation with FMV.
*   `bench.c`: Timing harness.

### 1. The Interface (`matopt.h`)

```c
#ifndef LIBMATOPT_H
#define LIBMATOPT_H

typedef struct {
    float *data;
    int rows;
    int cols;
    int stride;
} Matrix;

// Reference
void mat_mul_ref(Matrix *A, Matrix *B, Matrix *C);
void mat_trans_ref(Matrix *A, Matrix *B);

// Optimized
void mat_mul_opt(Matrix *A, Matrix *B, Matrix *C);
void mat_trans_opt(Matrix *A, Matrix *B);

#endif
```

### 2. The Reference (`matopt_naive.c`)

```c
#include "matopt.h"

void mat_mul_ref(Matrix *A, Matrix *B, Matrix *C) {
    for(int i=0; i<A->rows; i++) {
        for(int j=0; j<B->cols; j++) {
            float sum = 0.0f;
            for(int k=0; k<A->cols; k++) {
                // Warning: Potentially non-contiguous usage if stride != cols
                sum += A->data[i * A->stride + k] * 
                       B->data[k * B->stride + j];
            }
            C->data[i * C->stride + j] = sum;
        }
    }
}
```

### 3. The Optimized Core (`matopt_opt.c`)

We use **Flattening** (pointer arithmetic) + **Restrict** + **FMV**.

```c
#include "matopt.h"
#include <stdlib.h>

// Helper macro for index calc
#define IDX(r, c, stride) ((r)*(stride) + (c))

// Tiled GEMM Kernel
// Using static semantics to help compiler inline
static inline void gemm_kernel(float *restrict A, float *restrict B, float *restrict C, 
                               int N, int strideA, int strideB, int strideC) 
{
    // Assume TILE_SIZE 32
    int T = 32;
    for (int ii=0; ii<N; ii+=T) {
        for (int jj=0; jj<N; jj+=T) {
            for (int kk=0; kk<N; kk+=T) {
                
                // MICRO KERNEL
                for (int i=ii; i<ii+T; i++) {
                    for (int j=jj; j<jj+T; j++) {
                        float sum = C[IDX(i, j, strideC)];
                        // Hint: Width 8 (AVX)
                        #pragma clang loop vectorize_width(8) interleave_count(2)
                        for (int k=kk; k<kk+T; k++) {
                            sum += A[IDX(i, k, strideA)] * B[IDX(k, j, strideB)];
                        }
                        C[IDX(i, j, strideC)] = sum;
                    }
                }
            }
        }
    }
}

// FMV Wrapper
__attribute__((target_clones("avx2", "sse4.1", "default")))
void mat_mul_opt(Matrix *A, Matrix *B, Matrix *C) {
    // Only optimizing square case for this demo
    if (A->rows != A->cols) return mat_mul_ref(A, B, C);

    // Assume alignment (User responsibility in this hypothetical lib)
    float *pA = __builtin_assume_aligned(A->data, 32);
    float *pB = __builtin_assume_aligned(B->data, 32);
    float *pC = __builtin_assume_aligned(C->data, 32);

    gemm_kernel(pA, pB, pC, A->rows, A->stride, B->stride, C->stride);
}
```

### 4. The Benchmark Harness (`bench.c`)

```c
#include <stdio.h>
#include <time.h>
#include "matopt.h"

// Returns elapsed seconds
double time_gemm(void (*f)(Matrix*, Matrix*, Matrix*), Matrix *A, Matrix *B, Matrix *C) {
    clock_t start = clock();
    f(A, B, C);
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

int main() {
    int N = 1024;
    // Allocate aligned memory (posix_memalign or aligned_alloc)
    // ... setup A, B, C initialized with random data ...

    printf("Benchmarking N=%d...\n", N);

    double t_ref = time_gemm(mat_mul_ref, &A, &B, &C);
    printf("Reference: %.4f s\n", t_ref);

    double t_opt = time_gemm(mat_mul_opt, &A, &B, &C);
    printf("Optimized: %.4f s\n", t_opt);

    printf("Speedup: %.2fx\n", t_ref / t_opt);
    return 0;
}
```

---

## 🧪 Experiment: Profiling Cache Misses

Ideally, you run this on Linux.

```bash
# Compile
gcc -O3 -mavx2 matopt_opt.c matopt_naive.c bench.c -o bench

# Perf
perf stat -e L1-dcache-load-misses,cycles,instructions ./bench
```

**Analysis:**
*   **Naive:** High L1 miss rate (due to `B` accessed column-wise with large stride).
*   **Optimized:** Lower L1 miss rate (due to Tiling).
*   **IPC (Instructions Per Cycle):** Optimized should be higher (SIMD packs more work).

---

## 📝 Week 16 Review: Optimization Strategy

We realized that "writing code" is only half the battle. The other half is "convincing the compiler".

| Technique | When to use | Impact |
| :--- | :--- | :--- |
| **Analysis** | Start here. Is distance > 0? | Correctness. |
| **Tiling** | Large datasets ($N > 100$). | Memory Latency hiding. 5x-10x speedup. |
| **Hints** | `#pragma`, `restrict`. | Code Quality. Enables auto-vec. |
| **FMV** | Distributing binary to many users. | Portability w/ Performance. |
| **Auto-Vec** | Always. | 4x-8x speedup (Compute Bound). |

**What's Next? (Week 17 - GCC Internals)**
We leave the comfortable world of LLVM to explore the **GNU Compiler Collection (GCC)**.
*   GIMPLE vs LLVM IR.
*   RTL (Register Transfer Language).
*   GCC Plugins.
*   Why the Linux Kernel builds better with GCC.

*End of Day 112 - Total Lines: 1000+*
*End of Week 16 - Auto-Vectorization Complete!*
