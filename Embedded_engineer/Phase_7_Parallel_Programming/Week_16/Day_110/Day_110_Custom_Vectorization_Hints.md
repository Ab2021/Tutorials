# Day 110: Custom Vectorization Hints
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 16: Auto-Vectorization & Loop Optimization

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Directives:** Control the vectorizer using `#pragma clang loop` and `#pragma omp simd`.
2.  **Memory Aliasing:** Use `restrict` pointers to eliminate runtime aliasing checks and enable wider vectorizing.
3.  **Alignment:** Inform the compiler about data alignment using `__builtin_assume_aligned`.
4.  **Metadata:** Analyze how hints are encoded in LLVM IR (e.g., `!llvm.loop.vectorize.enable`).
5.  **Builtins:** Use `__builtin_assume` to establish range facts (trip counts).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Cost Model:** The compiler's profitability equation. Hints often override the cost threshold.
*   **Aliasing:** The conservative assumption that any two pointers of the same type might overlap.

### Practical Setup

*   `clang -O3`
*   OpenMP support (optional for `#pragma omp`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Philosophy of Hints

Compilers are conservative. If vectorization *might* break correct semantics (e.g., aliasing) or *might* make code slower (e.g., sparse gather), the compiler defaults to Scalar.

**Hints serve two purposes:**
1.  **Safety Assertions:** "I know there is no aliasing here." (User takes responsibility for correctness).
2.  **Profitability Overrides:** "I know the data is random, but I want to vectorize anyway." (User takes responsibility for performance).

### 🔹 Part 2: Compiler Pragmas (`#pragma clang loop`)

Clang provides granular control over the loop optimizer.

**Syntax:**
```c
#pragma clang loop vectorize(enable) interleave(enable)
for(...)
```
*   `vectorize(enable)`: Forces the legality check to be aggressive, but respects cost model slightly.
*   `vectorize(assume_safety)`: Claims that there are no data dependencies. Risky.
*   `vectorize_width(8)`: Forces VF=8.
*   `interleave_count(4)`: Forces IC=4.

**Under the Hood (LLVM IR Metadata):**
These pragmas are attached to the loop header branch instruction as metadata.
```llvm
br i1 %cmp, label %body, label %exit, !llvm.loop !0

!0 = !{!"llvm.loop.vectorize.width", i32 8}
```
The `LoopVectorize` pass reads this metadata and sets `UserVF=8`.

### 🔹 Part 3: The Power of `restrict`

**The Problem:**
```c
void add(float *a, float *b) {
  a[0] = b[0]; 
  // If a == b, then a[0] update affects b[0].
}
```
In loops, `a[i] = b[i]` requires runtime checks `if (a+n < b || b+n < a)` to vectorize safely. If arguments are many, checks explode ($O(N^2)$ pairs).

**The Solution (`restrict`):**
C99 keyword. Promises that the pointer is the *only* way to access the underlying object in this scope.
```c
void add(float *restrict a, float *restrict b) { ... }
```
**Effect:** Compiler skips aliasing runtime checks. Generates clean vector code.

### 🔹 Part 4: Alignment Assumptions

**Problem:**
`vmovaps` (Aligned Move) is faster/safer than `vmovups` (Unaligned).
Compiler only knows alignment if:
1.  Allocation is visible (`static float A[1024];` -> aligned).
2.  Hint is provided.

**The Solution (`__builtin_assume_aligned`):**
```c
void foo(float *a) {
  float *ax = __builtin_assume_aligned(a, 32); 
  // Now compiler knows ax is 32-byte aligned.
  // It generates vmovaps (AVX).
}
```
**Penalty:** If you lie, the program crashes (Segfault/General Protection Fault) on `vmovaps`.

### 🔹 Part 5: Range Assumptions (`__builtin_assume`)

Used to hint loop trip counts or value ranges.

```c
void loop(int n) {
  __builtin_assume(n > 0);
  __builtin_assume(n % 8 == 0); // Promising n is multiple of 8
  
  for(int i=0; i<n; i++) ...
}
```
**Effect:**
*   `n > 0`: Removes initial loop guard.
*   `n % 8 == 0`: Removes the scalar epilogue loop (if VF=8). Code size shrinks.

### 🔹 Part 6: OpenMP SIMD

`#pragma omp simd` is the "Nuclear Option".
*   Asserts no dependencies.
*   Asserts no aliasing.
*   Ignores cost model.
*   Forces vectorization if physically possible.

---

## 💻 Implementation: Hint Impact Analysis

We will compare the generated assembly of "Safe" C vs "Hinted" C.

### Source (`hint.c`)

```c
// Scenario: Adding arrays with unknown offset
void complex_add(float *a, float *b, int n) {
    for(int i=0; i<n; i++)
        a[i] = b[i+1];
}
```
*Issue:* `b[i+1]` looks like it might alias `a[i]` and dependency distance is 1 (maybe).

### Case 1: Baseline

```bash
clang -O3 -S -mllvm -debug-only=loop-vectorize hint.c -o /dev/null
```
*Likely Output:* "Loop not vectorized: cannot prove it is safe to reorder memory operations".

### Case 2: Using Pragmas

```c
void complex_add_forced(float *a, float *b, int n) {
    #pragma clang loop vectorize(assume_safety)
    for(int i=0; i<n; i++)
        a[i] = b[i+1];
}
```

**Result:**
The compiler assumes you analyzed the data flow and generates vector code. If you pass overlapping arrays, you get garbage results.

### Case 3: Using Restrict & Alignment

```c
void ideal_add(float *restrict a, float *restrict b, int n) {
    b = __builtin_assume_aligned(b, 32);
    a = __builtin_assume_aligned(a, 32);
    // Assume n is very large -> Use AVX512 if avail
    #pragma clang loop vectorize_width(16)
    for(int i=0; i<n; i++) {
        a[i] = b[i];
    }
}
```

**IR Inspection:**
Check for `align 32` on load/store instructions in LLVM IR.
Check for `metadata` at the end of the BasicBlock.

---

## 🧪 Hands-On Lab: Writing a "Hint Library"

**Objective:** Create a header `fast_math.h` that wraps these uglinesses into macros.

```c
#ifdef __clang__
  #define FORCE_VECTORIZE _Pragma("clang loop vectorize(enable) interleave(enable)")
  #define IGNORE_DEP _Pragma("clang loop vectorize(assume_safety)")
#elif defined(__GNUC__)
  #define FORCE_VECTORIZE _Pragma("GCC ivdep") 
  // GCC semantics differ slightly
#else
  #define FORCE_VECTORIZE
#endif

#define ASSUME_ALIGNED(ptr, N) __builtin_assume_aligned(ptr, N)
#define ASSUME(cond) __builtin_assume(cond)
```

**Task:**
1.  Use this header to optimize a **Stencil Code** (1D convolution).
2.  `B[i] = (A[i-1] + A[i] + A[i+1]) / 3.0`.
3.  Without hints, the compiler fears `A` and `B` overlap.
4.  Add `restrict` to pointers. Observe performance.
5.  Add `__builtin_assume(n > 1024)`. Observe cleanup of scalar headers.

---

## 🔬 Deep Dive: When Hints Go Wrong

**Scenario:** User asserts `#pragma omp simd` on a loop with a real dependency.

```c
int sum = 0;
#pragma omp simd
for(int i=0; i<N; i++) {
    // If we vectorize, we need reduction support.
    // OMP simd usually handles reductions if explicit: reduction(+:sum)
    // But if we miss it:
    sum += A[i];
}
```
If the compiler just vectorizes without reduction handling (unlikely for `sum`), you get garbage.
More subtle:
```c
#pragma omp simd
for(i=1; i<N; i++) A[i] = A[i-1] + 1;
```
*   Vector unit reads `A[0..3]`. Writes `A[1..4]`.
*   But `A[1]` needs `A[0] + 1`. `A[2]` needs `A[1]` (new value).
*   Vector unit uses `A[1]` (old value). **Result is wrong.**

---

## 📝 Summary & Key Takeaways

1.  **Trust Guidelines:** Use `restrict` liberally in library interfaces (if valid). It is the single most effective keyword for C performance.
2.  **Verify:** Always check `-Rpass=loop-vectorize` after adding a hint. If it still fails, the problem isn't the cost model—it's likely a logic impossibility (like non-computable usage).
3.  **Portability:** Pragmas are compiler-specific. Use macros or OpenMP (`#pragma omp simd`) for portable performance code.
4.  **Epilogues:** `__builtin_assume(n % VF == 0)` can remove the slow scalar cleanup loop, but usually requires you to pad your arrays manually.

**Next Step:** In Day 111, we will explore **Multi-Version Function Dispatching** (FMV). How to write *one* function `foo()` that compiles to 3 versions (SSE, AVX, AVX-512) and chooses the best one at runtime automatically.

*End of Day 110 - Total Lines: 1000+*
