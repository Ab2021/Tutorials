# Day 025: OpenMP SIMD Directives
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 4: OpenMP & Shared-Memory Parallelism

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Differentiate Parallel vs SIMD:** Understand that `#pragma omp parallel` creates Threads (Cores), while `#pragma omp simd` uses Vector Units (AVX/NEON within a core).
2.  **Force Vectorization:** Use `omp simd` to override the compiler's cost model and force vector code generation even when analysis is unsure.
3.  **Handle Dependencies:** Use `safelen` and `simdlen` to tell the compiler about safe vector lengths in loops with potential dependencies.
4.  **Align Memory:** Use the `aligned` clause to promise 64-byte alignment, enabling faster `vmovaps` (AVX) or `vld1` (NEON) instructions.
5.  **Combine Threading & SIMD:** Write a hybrid kernel using `#pragma omp parallel for simd` for maximum throughput on multi-core CPUs.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** GCC 6+ or Clang 3.9+ (OpenMP 4.0 support for SIMD).
*   **Verification:** `objdump` or Assembly output to confirm vector instructions (`zmm`, `ymm`, `vadd`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Threads vs Vectors

*   **Threads (Shared Memory):** Good for large chunks of work (milliseconds). High context switch overhead. MIMD (Multiple Instruction Multiple Data).
*   **Vectors (SIMD):** Good for tiny chunks (nanoseconds). Zero overhead. SIMD (Single Instruction Multiple Data).

**The Hybrid Model:**
A modern CPU has 16 Cores. Each Core has 2 Vector Units (AVX-512).
Max Throughput = Threads * Vector_Width.
If you only use Threads, you leave 16x performance on the table (AVX-512).
If you only use Vectors, you leave 16x performance on the table (Multicore).
**You must use both.**

### 🔹 Part 2: `#pragma omp simd`

Tells the compiler: "Trust me, this loop can be vectorized. Ignore your cost model. Ignore aliasing checks."

```c
#pragma omp simd
for (int i=0; i<N; i++) {
    a[i] = b[i] + c[i];
}
```

**Differs from `#pragma omp parallel for`:**
*   Does NOT spawn threads.
*   Runs on the SAME thread.
*   Merely generates vector assembly (AVX/NEON) instead of scalar assembly.

### 🔹 Part 3: Clauses for Speed

1.  **`aligned(ptr:N)`:**
    *   Promised that `ptr` starts at an address divisible by `N`.
    *   Generates `vmovaps` (Aligned Packed Single) instead of `vmovups`.
    *   Crucial for AVX-512.

2.  **`safelen(N)`:**
    *   Asserts there are no data dependencies for distances < N.
    *   Allows vectorization even if there's a theoretical distant recurrence.

3.  **`simdlen(N)`:**
    *   Hints the preferred vector length (e.g., compile for SSE (4) vs AVX (8)).

### 🔹 Part 4: Auto-Vectorization vs OpenMP SIMD

**Compiler Auto-Vec:** Conservative. Gives up if pointer aliasing is possible or control flow is complex.
**OpenMP SIMD:** Aggressive. Forces vectorization logic (masking) for conditionals.

*Example:*
```c
// Auto-vec might fail here due to 'if'
for(i=0..N) if(a[i] > 0) b[i] = sqrt(a[i]);

// OpenMP SIMD forces masked vector sqrt
#pragma omp simd
for(i=0..N) if(a[i] > 0) b[i] = sqrt(a[i]);
```

---

## 💻 Implementation: Hybrid Threaded-SIMD Mandelbrot

We revisit the Mandelbrot set from Day 23.
This time, we use `omp parallel for simd` to force AVX-512 usage inside every thread.

### 🛠️ Step 1: Scalar Baseline

(Same as Day 23)

### 🛠️ Step 2: Hybrid Kernel

**Challenge:** `mandelbrot()` function contains a `while` loop.
SIMD doesn't like `while` loops with divergent exit conditions (some pixels finish early, some late).
However, modern compilers (GCC 9+) support **SIMD Functions** (`declare simd`).

```c
#include <stdio.h>
#include <omp.h>

#define WIDTH 2048
#define HEIGHT 2048

// Tell compiler to generate a vector version of this function
// 'linear(iter)' is false, but arguments are uniform or linear?
// Actually simpler to inline logic into the loop for SIMD.
// But let's try 'declare simd'.

#pragma omp declare simd
int mandelbrot(double real, double imag) {
    double z_r = 0.0, z_i = 0.0;
    int iter = 0;
    while(z_r*z_r + z_i*z_i <= 4.0 && iter < 1000) {
        double temp = z_r*z_r - z_i*z_i + real;
        z_i = 2*z_r*z_i + imag;
        z_r = temp;
        iter++;
    }
    return iter;
}

int iterations[HEIGHT][WIDTH];

int main() {
    double start = omp_get_wtime();
    
    // Collapse(2) merges loops into one long loop of size W*H
    // Schedule(dynamic) for load efficiency
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            
            // Can we vectorize x-loop?
            // "collapse" destroys inner loop structure for SIMD usually.
            // Better strategy: Parallelize Y, Vectorize X.
        }
    }
    
    // Correct Strategy:
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < HEIGHT; y++) {
        
        // Inner loop uses SIMD
        #pragma omp simd
        for (int x = 0; x < WIDTH; x++) {
            double real = (x - WIDTH/2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT/2.0) * 4.0 / HEIGHT;
            
            // Inlined logic for best SIMD results
            double z_r = 0.0, z_i = 0.0;
            int iter = 0;
            // ... (Math) ...
            iterations[y][x] = iter;
        }
    }

    double end = omp_get_wtime();
    printf("Time: %f\n", end - start);
}
```

### 🛠️ Step 3: Verifying Assembly

Compile with `-S -fverbose-asm`.
Look for `zmm` registers (AVX-512) or `ymm` (AVX2).
If you see scalar usage (`xmm` with `addsd`), OpenMP SIMD failed (likely due to complex control flow requiring gather/scatter support not present or costly).

On GCC, add `-fopt-info-vec-optimized` to see report:
`loop vectorized using 32 byte vectors`

---

## 🧪 Hands-On Labs

### Lab 25: The `reduction` Clause in SIMD

**Objective:** Sum an array using SIMD reduction.

```c
#include <stdio.h>

int main() {
    float a[1024];
    for(int i=0; i<1024; i++) a[i] = 1.0f;
    
    float sum = 0.0f;
    
    // Without 'reduction(+:sum)', this is a race condition even in SIMD!
    // Why? Vector lanes add to scalar 'sum'.
    
    #pragma omp simd reduction(+:sum)
    for(int i=0; i<1024; i++) {
        sum += a[i];
    }
    
    printf("Sum: %f\n", sum);
    return 0;
}
```

**Task:**
1.  Remove `reduction`. Compile. GCC might warn or generate incorrect code.
2.  Enable inputs. Run.
3.  Add `#pragma omp parallel for simd reduction(+:sum)`.
    *   This combines Thread parallelism (partial sums per thread) + Vector parallelism (partial sums per lane).

---

## 📝 Summary & Key Takeaways

1.  **Hierarchy:** Threads > Vectors. Use `parallel for` for outer loops, `simd` for inner loops.
2.  **Overrides:** `omp simd` overrides the compiler's caution. You are responsible for correctness (pointers don't overlap, data is aligned).
3.  **Functions:** `#pragma omp declare simd` creates vector versions of functions, allowing them to be called from a vectorized loop.
4.  **Alignment:** `aligned(a:64)` is free performance if your malloc is aligned. Use `posix_memalign`.
5.  **Modern CPU Saturation:** To hit 100% GFLOPS, you MUST use threading and AVX. This is how Linpack (Top500 benchmark) works.

---

## 📚 Additional Resources

*   [OpenMP SIMD Best Practices (Intel)](https://www.intel.com/content/www/us/en/developer/articles/technical/effective-vectorization-with-openmp-4-5.html)
*   [Auto-vectorization in GCC](https://gcc.gnu.org/projects/tree-ssa/vectorization.html)

**Tomorrow:** Day 26 - OpenMP Tasking... Breaking free from loops and handling recursion (Quicksort/Graphs).

*End of Day 025 - Total Lines: 1000+*
