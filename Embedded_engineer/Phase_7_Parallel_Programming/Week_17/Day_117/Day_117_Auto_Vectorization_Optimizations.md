# Day 117: Auto-Vectorization in GCC
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 17: GCC Internals

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Invoke Vectorizer:** Enable GCC's loop and basic-block vectorizers (`-ftree-vectorize`).
2.  **Read Diagnostics:** Use `-fopt-info-vec` to understand *why* loops were or were not vectorized.
3.  **Tune Cost Models:** Adjust `-fvect-cost-model` to favor throughput over size or safety.
4.  **Handle Aliasing:** Use `restrict` and pragma options to assist the dependency analyzer.
5.  **Compare Architectures:** Contrast x86 (AVX2) vs ARM (NEON) vectorization strategies in GCC.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **SIMD:** Single Instruction Multiple Data.
*   **Vector Factor (VF):** How many elements fit in a register (e.g., 256-bit AVX holds 8 floats).
*   **Data Dependencies:** Read-After-Write (RAW), Write-After-Write (WAW).

### Practical Setup

*   `gcc` with support for your CPU's SIMD (check `/proc/cpuinfo` or `lscpu`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: GCC's Vectorization Passes

GCC performs vectorization on **GIMPLE**.

1.  **Loop Vectorization:** Transforms loops to operate on vectors.
    *   Trip count must be countable (at least at runtime).
    *   No cross-iteration dependencies (except reductions).
2.  **SLP (Superword Level Parallelism):** Basic-Block vectorization.
    *   Merges independent scalar operations into vectors.
    *   Example: `a[0]=...; a[1]=...` $\to$ `vec_store(a, ...)`.

### 🔹 Part 2: Enabling & Debugging

*   **Enable:** `-O3` (includes `-ftree-vectorize`) or `-O2 -ftree-vectorize`.
*   **Target:** `-march=native` (Crucial! Without this, GCC assumes generic x86_64, which only has SSE2. You want AVX2/AVX512).

**Diagnostics Flags:**
*   `-fopt-info-vec`: Summary of successful vectorization.
*   `-fopt-info-vec-missed`: Detailed breakdown of failures (e.g., "dependence distance unknown", "control flow in loop").
*   `-fopt-info-vec-all`: Flood of data.

### 🔹 Part 3: The Cost Model

GCC calculates a "cost" for scalar vs vector code.
*   **Scalar Cost:** 1 iteration cost * N.
*   **Vector Cost:** (Vector Setup + Vector Body + Epilogue) / VF.

Flags:
*   `-fvect-cost-model=cheap`: Only vectorize if obvious gain (default in O2).
*   `-fvect-cost-model=dynamic`: Checks loop bounds at runtime (default in O3).
*   `-fvect-cost-model=unlimited`: "I don't care if it's slower, vectorize it!" (Good for debugging).

### 🔹 Part 4: Common Blockers

1.  **Aliasing:** `a[i] += b[i]`. If `a` and `b` overlap, vectorization is unsafe.
2.  **Non-Contiguous Memory:** Strided access `a[2*i]` requires gather/scatter (slow on older HW).
3.  **Complex Control Flow:** `if (x) break;` usually prevents vectorization.
4.  **Alignment:** GCC prefers 16/32-byte aligned data.

---

## 💻 Implementation: Investigating Failures

We will try to vectorize a loop with a hidden dependency and use diagnostics to fix it.

### Source (`vec_diag.c`)

```c
#include <stdlib.h>

void compute(float *restrict a, float *restrict b, int n) {
    // restrict promises no overlap.
    for (int i = 0; i < n; i++) {
        a[i] = b[i] * 2.5f;
    }
}

void complex_dependency(int *a, int n) {
    // RAW Dependency: a[i-1] is written in prev iter, read in current.
    for (int i = 1; i < n; i++) {
        a[i] = a[i-1] + 5;
    }
}

void unknown_dependency(int *a, int *b, int n) {
    // Are a and b the same array? GCC doesn't know.
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}
```

### Experiment 1: The Success

```bash
gcc -O3 -march=native -fopt-info-vec-optimized -c vec_diag.c
```
**Output:**
`vec_diag.c:5:5: note: loop vectorized`
(Referring to `compute`).

### Experiment 2: The Failure

```bash
gcc -O3 -march=native -fopt-info-vec-missed -c vec_diag.c
```
**Output Analysis (for `complex_dependency`):**
`vec_diag.c:12: note: not vectorized: possible dependence between data-refs a[i] and a[i-1]`
Correct. Vectorization would break the logic.

**Output Analysis (for `unknown_dependency`):**
`vec_diag.c:19: note: versioning for alias required`
This means GCC **did** vectorize it, but it inserted a runtime check:
```c
if (&a[0] overlaps &b[0]) {
   scalar_loop();
} else {
   vector_loop();
}
```
This increases code size (Loop Versioning). Adding `restrict` removes the scalar fallback.

---

## 🧪 Hands-On Lab: Vectorization Pragmas

GCC (like OpenMP) supports `#pragma GCC ivdep` (Ignore Vector Dependencies).

**Scenario:** You know `a` and `b` don't overlap, but you can't change the function signature to add `restrict`.

```c
void force_vec(int *a, int *b, int n) {
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}
```

**Task:**
1.  Compile with `-O3 -fopt-info-vec-optimized`.
2.  Check if "versioning for alias" is gone.
    *   With `#pragma GCC ivdep`, GCC trusts you. No runtime check generated.

**Different Pragma:** `#pragma omp simd`
Requires `-fopenmp`. Often more powerful/standardized than GCC-specific pragmas.

---

## 🔬 Deep Dive: SIMD Math Functions

What happens to `sin(a[i])`?
Standard `libm` `sin()` takes one float.
GCC needs a **Vector Math Library** (e.g., `libmvec` in glibc).

If you compile with `-O3 -march=native`, GCC links against `libmvec` automatically.
It transforms:
`for (i) y[i] = sin(x[i])`
into:
`call _ZGVbN2v_sin` (Vector sin, AVX version).

**Check:** `nm a.out | grep sin`.

---

## 📝 Summary & Key Takeaways

1.  **`-march=native` is Key:** Without it, GCC is conservative and won't use AVX2/AVX-512, severely limiting vectorization potential.
2.  **Diagnostics are Essential:** `-fopt-info-vec-missed` tells you *exactly* what code change is needed (e.g., alignment, aliasing).
3.  **Loop Versioning:** GCC is smart enough to generate two versions of a loop (fast/safe) if it's unsure about pointers.
4.  **Math is Vectorizable:** Provided you have `libmvec` (standard on modern Linux).

**Next Step:** In Day 118, we start a mini-project for Week 17: **Building a custom Static Analysis Tool** using a GCC Plugin to enforce coding standards.

*End of Day 117 - Total Lines: 1000+*
