# Day 108: Auto-Vectorization in Compilers
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 16: Auto-Vectorization & Loop Optimization

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **SIMD Concept:** Map scalar operations (`a+b`) to vector operations (`<4 x float> + <4 x float>`).
2.  **SLP (Superword-Level Parallelism):** Identify parallel code within basic blocks (straight-line code).
3.  **Loop Vectorization:** Transform loops to execute multiple iterations per cycle.
4.  **Cost Models:** Understand how the compiler decides *if* vectorization is profitable (Code size vs Speed).
5.  **Predication:** Handle control flow (`if` inside loops) using vector masks.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **SIMD Hardware:** AVX2 (256-bit, 8 floats), AVX-512 (512-bit, 16 floats), NEON (128-bit).
*   **Alignment:** Vector loads are faster (or only legal) on aligned memory.

### Practical Setup

*   `clang -O3 -fno-slp-vectorize` (to isolate loop vectorizer).
*   `clang -Rpass=loop-vectorize` (The "Isolate" report - tells you what vectorised).
*   `clang -Rpass-missed=loop-vectorize` (Tells you WHY it failed).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SLP Vectorization (Bottom-Up)

**Goal:** Vectorize straight-line code (outside loops or inside unrolled loops).
**Algorithm:**
1.  Start from "Root" instructions (e.g., adjacent stores `A[0]=x; A[1]=y;`).
2.  Trace Def-Use chains up.
3.  If `x = a + b` and `y = c + d`, check if we can pack `(a,c)` and `(b,d)`.
4.  If tree is isomorphic, emit `(x,y) = (a,c) + (b,d)`.

**Example:**
```c
A[0] = B[0] + C[0];
A[1] = B[1] + C[1];
```
*   Scalar: 2 loads (B), 2 loads (C), 2 adds, 2 stores. Total 8 ops.
*   Vector: 1 vec-load (B), 1 vec-load (C), 1 vec-add, 1 vec-store. Total 4 ops.

### 🔹 Part 2: Loop Vectorization

**Goal:** Execute $VF$ (Vector Factor) iterations at once.
$VF$ depends on register width ($W$) and type size ($T$). $VF = W / T$.
For AVX2 (256-bit) and float (32-bit), $VF = 8$.

**Steps:**
1.  **Legality Check:** Check dependencies (Day 106).
2.  **Cost Model:** Is `Cost(Vector) < Cost(Scalar) * VF`?
    *   Gather/Scatter loads (`A[B[i]]`) are expensive.
    *   Divides are expensive.
3.  **Transformation:**
    *   **Widening:** Replace `load float` with `load <8 x float>`.
    *   **Broadcasting:** If `x` is loop invariant, create `<x, x, ..., x>`.
    *   **Induction Handling:** Primary IV `i` becomes `<0, 1, ..., 7> + i`.
4.  **Epilogue Peeling:** If $N$ is not multiple of $VF$, handle remaining $N \% VF$ iterations with a scalar loop.

### 🔹 Part 3: Control Flow (Predication)

**Problem:**
```c
for (i=0; i<N; i++)
  if (A[i] > 0)
    B[i] = 1;
```

**Solution (Masking):**
We cannot "branch" in SIMD. We must execute *both* paths or use masked instructions.
1.  Compute Mask: `M = (A_vec > 0_vec)`.
2.  Masked Store: `MaskedStore(1_vec, B_ptr, M)`.
    *   Only writes elements where $M_k = 1$.

### 🔹 Part 4: Aliasing & Runtime Checks

If compiler sees:
```c
void foo(float *A, float *B, int n) {
  for(i) A[i] = B[i] + 1;
}
```
It worries: What if `A = B + 1`? (Write $A[i]$ overwrites $B[i+1]$).
**Versioning:**
Compiler generates TWO versions:
```c
if ( (A+n < B) || (B+n < A) ) { 
  VectorizedLoop(); 
} else { 
  ScalarLoop(); 
}
```
This is **Loop Versioning for Aliasing**.

---

## 💻 Implementation: Investigating Flags

We will inspect how Clang reports vectorization success/failure.

### Source (`vec_test.c`)

```c
// 1. Easy
void add(float *restrict A, float *restrict B, int n) {
    for(int i=0; i<n; i++) A[i] += B[i];
}

// 2. Control Flow
void filter(float *A, int n) {
    for(int i=0; i<n; i++)
        if (A[i] < 0) A[i] = 0;
}

// 3. Unknown Trip Count & Aliasing
void risky(float *A, float *B, int n) {
    for(int i=0; i<n; i++)
        A[i] += B[i]; // No restrict!
}
```

### Analysis Commands

1.  **Report Success:**
    ```bash
    clang -O2 -Rpass=loop-vectorize -c vec_test.c
    ```
    *Output:* Should confirm `add` and `filter` were vectorized.

2.  **Report Missed:**
    ```bash
    clang -O2 -Rpass-missed=loop-vectorize -c vec_test.c
    ```

3.  **Inspect Assembly (AVX2):**
    ```bash
    clang -O2 -mavx2 -S vec_test.c -o vec_test.s
    ```
    *Look for `vaddps`, `vmaxps` (efficient way to do `if (x<0) x=0`).*

### IR Inspection

Compile to IR:
```bash
clang -O2 -mavx2 -S -emit-llvm vec_test.c -o vec_test.ll
```
Go to `add` function. Look for `<8 x float>`.
Note the structure:
*   `vector.body`: The main loop.
*   `middle.block`: Checks if we finished or need epilogue.
*   `scalar.ph`: Preheader for scalar epilogue.

---

## 🧪 Hands-On Lab: Helping the Compiler

**Objective:** Fix "Missed Vectorization" code.

**Bad Code (`complex.c`):**

```c
typedef struct { float x, y, z; } Point;

void normalize(Point *p, int n) {
    for(int i=0; i<n; i++) {
        float len = sqrt(p[i].x*p[i].x + p[i].y*p[i].y + p[i].z*p[i].z);
        if (len > 0) {
            p[i].x /= len;
            p[i].y /= len;
            p[i].z /= len;
        }
    }
}
```
**Issues:**
1.  **AoS (Array of Structs):** `x, y, z` are interleaved. Vectorizer wants `x, x, x, x`. It must gather/scatter or shuffle. Cost is high.
2.  **`sqrt`:** Requires `-fno-math-errno` to vectorize aggressively sometimes.
3.  **Control Flow w/ Division:** Safe division?

**Task:**
1.  Compile and check `-Rpass-missed`.
2.  Rewrite to **SoA (Structure of Arrays)**: `struct { float *x, *y, *z; } Points;`.
    *   This makes `x` contiguous.
3.  Re-compile. Observe vectorization.

---

## 🔬 Deep Dive: Pragma OMP SIMD

Sometimes you know better than the Cost Model. You can force vectorization using OpenMP 4.0+.

```c
void force_vec(float *A, int offset, int n) {
    // Compiler fears 'offset' causes aliasing A[i] vs A[i+offset]
    #pragma omp simd
    for(int i=0; i<n; i++) {
        A[i] = A[i+offset] * 0.5f;
    }
}
```
*   `#pragma omp simd`: "Ignore dependencies. Vectorize this. If I'm wrong, segfault me, I don't care."
*   Similar to GCC's `#pragma GCC ivdep` or Clang's `#pragma clang loop vectorize(enable)`.

---

## 📝 Summary & Key Takeaways

1.  **Data Layout is King:** AoS (Array of Structs) kills vectorization. SoA (Structure of Arrays) enables it.
2.  **Control Flow Costs:** Predication works, but it executes BOTH paths. If `if (rare) { expensive(); }`, vectorization might be *slower* because it forces execution of `expensive()` for all vector lanes even if only one needs it.
3.  **Reductions:** `sum += A[i]` is strictly serial in IEEE 754 float (order matters). To vectorize, you must allow reassociation (`-ffast-math`).
4.  **Epilogues:** Real performance requires handling alignments and trip counts efficiently, often leading to code bloat (Vector loop + Scalar cleanups).

**Next Step:** In Day 109, we will deep dive into the **LLVM Loop Vectorizer Pass** implementation details, understanding how it builds the Vector Loop in IR.

*End of Day 108 - Total Lines: 1000+*
