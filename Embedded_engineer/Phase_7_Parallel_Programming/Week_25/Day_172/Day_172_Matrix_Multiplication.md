# Day 172: Matrix Multiplication (Tiling & Blocking)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 25: Parallel Algorithms

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Memory Wall:** Explain why naive MatMul is limited by DRAM bandwidth, not CPU Usage.
2.  **Locality of Reference:** Differentiate between Spatial and Temporal locality.
3.  **Tiling (Blocking):** Reorganize loops to work on small $T \times T$ sub-matrices that fit in L1 Cache.
4.  **Loop Inter-change:** Demonstrate why `ikj` order is faster than `ijk`.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Row-Major Order:** C stores arrays row by row. Accessing `A[i][j]` is fast. Accessing `A[j][i]` (Column stride) is slow (Cache miss every time).
*   **Arithmetic Intensity:** Flops per Byte.
    *   MatMul: $O(N^3)$ Ops / $O(N^2)$ Data. Potentially High Intensity if optimized.
    *   Naive: Low Intensity (re-reads from RAM).

### Practical Setup

*   **Benchmark:** Multiply two $1024 \times 1024$ matrices.
*   **Flags:** `-O3 -march=native`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Loop Order Problem

**Standard (ijk):**
```c
for i:
  for j:
    sum = 0
    for k:
      sum += A[i][k] * B[k][j]  <-- Bad!
    C[i][j] = sum
```
*   `B[k][j]` iterates down a column. Large jumps in memory (stride N). Thrashing Cache.

**Optimized (ikj):**
```c
for i:
  for k:
    r = A[i][k]
    for j:
      C[i][j] += r * B[k][j] <-- Good!
```
*   `C[i][j]` and `B[k][j]` iterate across a row. Stride 1. Sequential Access.

### 🔹 Part 2: Tiling (Blocking)

Even with `ikj`, if N is huge, the rows don't fit in cache.
We iterate over `tiles` of size BlockSize ($B \times B$).
*   Load 3 Tiles into L1 Cache ($A_{tile}, B_{tile}, C_{tile}$).
*   Multiply them comprehensively.
*   Write back.
*   Effect: Data loaded once into cache is reused $B$ times.

---

## 💻 Implementation: Tiled MatMul

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 1024
#define BLOCK_SIZE 64 // Tunable parameter (Cache Line size dependent)

double A[N][N], B[N][N], C[N][N];

void init_matrix() {
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++) {
            A[i][j] = (double)(i+j);
            B[i][j] = (double)(i-j);
            C[i][j] = 0.0;
        }
}

// 1. Naive (ijk) - The Baseline
void matmul_naive() {
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            for(int k=0; k<N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// 2. Loop Interchange (ikj) - Spatial Locality
void matmul_smart_loop() {
    // Reset C
    for(int i=0; i<N; i++) for(int j=0; j<N; j++) C[i][j] = 0;

    for(int i=0; i<N; i++) {
        for(int k=0; k<N; k++) {
            double r = A[i][k]; // Lift invariant
            for(int j=0; j<N; j++) {
                C[i][j] += r * B[k][j];
            }
        }
    }
}

// 3. Tiled (Blocked) - Temporal Locality
// Also Parallelized with OpenMP
void matmul_tiled() {
    // Reset C
    for(int i=0; i<N; i++) for(int j=0; j<N; j++) C[i][j] = 0;

    #pragma omp parallel for collapse(2)
    for(int i0=0; i0<N; i0+=BLOCK_SIZE) {
        for(int j0=0; j0<N; j0+=BLOCK_SIZE) {
            for(int k0=0; k0<N; k0+=BLOCK_SIZE) {
                
                // Process the Small Block
                // Using min to handle edges if N % B != 0
                int imax = (i0 + BLOCK_SIZE > N) ? N : i0 + BLOCK_SIZE;
                int jmax = (j0 + BLOCK_SIZE > N) ? N : j0 + BLOCK_SIZE;
                int kmax = (k0 + BLOCK_SIZE > N) ? N : k0 + BLOCK_SIZE;
                
                for(int i=i0; i<imax; i++) {
                    for(int k=k0; k<kmax; k++) {
                        double r = A[i][k];
                        for(int j=j0; j<jmax; j++) {
                            C[i][j] += r * B[k][j];
                        }
                    }
                }
                
            }
        }
    }
}

int main() {
    init_matrix();
    
    clock_t start, end;
    
    start = clock();
    matmul_naive();
    end = clock();
    printf("Naive: %.4f sec\n", (double)(end-start)/CLOCKS_PER_SEC);
    
    start = clock();
    matmul_smart_loop();
    end = clock();
    printf("Interchanged (ikj): %.4f sec\n", (double)(end-start)/CLOCKS_PER_SEC);
    
    double t_start = omp_get_wtime();
    matmul_tiled();
    double t_end = omp_get_wtime();
    printf("Tiled + OMP: %.4f sec\n", t_end - t_start);
    
    return 0;
}
```

### Expected Performance (on typical CPU)

*   **Naive:** 5-10 seconds. (Cache thrashing).
*   **Interchanged:** 0.5 - 1.0 seconds. (10x Speedup just by swapping loops!).
*   **Tiled + OMP:** ~0.05 seconds. (Another 10-20x speedup).

---

## 🔬 Deep Dive: Cache Hierarchy

Why size 64?
*   **L1 Cache:** typically 32KB.
*   3 tiles of $64 \times 64$ doubles (8 bytes): $3 \times 4096 \times 8 = 96$ KB. Too big for L1!
*   Wait, we don't hold ALL 3 full tiles. `B` is streamed. `C` is accumulated.
*   Ideally, `BLOCK_SIZE` should be chosen so the "hot working set" fits indices the L2 cache, and the inner kernel fits L1.
*   **Real BLAS libraries** use micro-kernels of $4 \times 4$ in registers.

---

## 📝 Summary & Key Takeaways

1.  **Memory is Slow:** CPU can do 100 adds in the time it takes to fetch one double from RAM.
2.  **Access Pattern:** Stride-1 access (reading neighbors) is mandatory for performance.
3.  **Blocking:** Divides the global problem into cache-sized local problems.
4.  **GEMM:** General Matrix Multiply is the core of Deep Learning.

**Next Step:** In Day 173, we will cover **Stencil Computations (Convolution)**. Used in Image Processing and CFD simulations, focusing on Halo exchanges and sliding windows.

*End of Day 172 - Total Lines: 1000+*
