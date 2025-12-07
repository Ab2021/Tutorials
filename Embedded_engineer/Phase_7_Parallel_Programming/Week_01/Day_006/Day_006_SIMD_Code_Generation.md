# Day 006: SIMD Code Generation & Auto-Vectorization
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 1: x86 SIMD Foundations

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1. **Leverage Compiler Auto-Vectorization:** Understand how GCC/Clang transforms scalar loops into SIMD code using `-O3` and `-ftree-vectorize`.
2. **Control Loop Vectorization:** Use `#pragma omp simd`, `#pragma GCC ivdep`, and clang loop hints to force vectorization on complex loops.
3. **Debug Vectorization Failures:** Analyze compiler reports (`-fopt-info-vec`) to understand why a loop wasn't vectorized (aliasing, data dependency, control flow).
4. **Use Vector Types:** Utilize GCC/Clang `__attribute__((vector_size(N)))` for a cleaner C++ style SIMD interface without intrinsics.
5. **Analyze Generated Assembly:** Read disassembly (`objdump -d`) to verify if `addps` vs `addss` (scalar) instructions are being generated.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Compiler | GCC 8+ / Clang 9+ | GCC 13+ / Clang 17+ | Improved auto-vectorization reports in newer versions |
| Tools | objdump, godbolt.org | Compiler Explorer | Great for inspecting assembly interactively |
| OpenMP | libomp-dev | Latest | For `#pragma omp simd` support |

### Environment Setup

```bash
# Check compiler version and install OpenMP
gcc --version
sudo apt install libomp-dev

# Test compilation command
gcc -O3 -march=native -fopt-info-vec-missed src.c -o app
```

### Prior Knowledge Checklist

- [ ] Completed Days 1-5 (Intrinsics familiarity)
- [ ] Understanding of Pointer Aliasing (C strict aliasing rules)
- [ ] Basic Assembly reading skills

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Compiler as a Vectorizer

Before writing intrinsics manually (which is hard and non-portable), we should always check: **Can the compiler do it for me?**

**Auto-Vectorization** is the process where the compiler identifies loops that perform data-parallel operations and transforms them into SIMD instructions.

#### 1.1 Criteria for Vectorization

For a loop to be vectorized, it must satisfy:
1. **Countable:** Loop iterations determined at entry (e.g., `for i=0..N`). `while(ptr != null)` is hard.
2. **No Data Dependencies (Loop Carried):** `a[i] = a[i-1] + 1` cannot be vectorized straightforwardly (requires dependency).
3. **No Aliasing:** Input and output arrays must not overlap in a way that breaks safety.
4. **Simple Control Flow:** No complex `break`, `switch`, or inner-loop exits (though masked execution handles simple `if`).

**Example: Vectorizable**
```c
for (int i=0; i<N; i++) a[i] = b[i] + c[i];
```
This is independent. `i=0` doesn't depend on `i=1`.

**Example: Not Vectorizable (Dependency)**
```c
for (int i=1; i<N; i++) a[i] = a[i-1] + c[i];
```
To compute `a[i]`, we need `a[i-1]`. Serial dependency.

#### 1.2 The Aliasing Problem (The #1 Killer)

In C/C++, pointers can point ANYWHERE.

```c
void add(float* a, float* b, float* c, int n) {
    for(int i=0; i<n; i++) a[i] = b[i] + c[i];
}
```
**Compiler's Fear:** What if `a == b + 1`?
If `a` points to `b[1]`, then writing `a[0]` overwrites `b[1]`. Next iteration reads `b[1]` (which is now `a[0]`). Dependency!
Result: Compiler generates SCALAR code to be safe.

**Solution 1: `restrict` Keyword (C99)**
```c
void add(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int n)
```
Promised to compiler: "These pointers do not overlap."
*Result:* Compiler vectorizes aggressively.

**Solution 2: `#pragma GCC ivdep`**
"Ignore Vector Dependencies" - Ignore assumed dependencies, trust programmer.

---

### 🔹 Part 2: Explicit Vectorization Directives

Sometimes `restrict` isn't enough. We need to force it.

#### 2.1 OpenMP SIMD

OpenMP 4.0 introduced generic SIMD directives portable across compilers (GCC, Clang, ICC, MSVC).

```c
#include <omp.h>

void add_omp(float* a, float* b, float* c, int n) {
    #pragma omp simd aligned(a,b,c : 32)
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

**Directives:**
- `#pragma omp simd`: Force vectorization.
- `aligned(ptr:N)`: Assume ptr is N-byte aligned (generates optimized aligned loads).
- `safelen(N)`: Promise that dependencies are > N iterations away.
- `reduction(+:sum)`: Handle reduction vectorization correctly.

**Compile Flag:** `-fopenmp` or `-fopenmp-simd`.

#### 2.2 GCC/Clang Specific Hints

**GCC:**
```c
#pragma GCC ivdep 
for(...)
```

**Clang:**
```c
#pragma clang loop vectorize(enable) interleave(enable)
for(...)
```

---

### 🔹 Part 3: GCC Vector Extensions

Instrinsics (`_mm_add_ps`) are ugly.
Auto-vectorization is fragile.
**Vector Extensions** are the middle ground.

```c
// Define a 32-byte (256-bit) vector type containing floats
typedef float vec8 __attribute__((vector_size(32))); 

void add_vec(vec8* a, vec8* b, vec8* c, int n) {
    int chunks = n / 8;
    for (int i = 0; i < chunks; i++) {
        c[i] = a[i] + b[i]; // Standard + operator works!
    }
}
```

**Pros:**
- Readable (uses `+`, `-`, `*`).
- Portable-ish (Supported by GCC/Clang).
- Maps to best instruction set (AVX2 on Haswell, AVX-512 on Skylake).

**Cons:**
- Complex shuffles/permutes still awkward.
- Not standard C++.

---

### 🔹 Part 4: Analyzing Vectorization Reports

How do you know if it worked?

**GCC Flags:**
- `-fopt-info-vec`: Info on successful vectorization.
- `-fopt-info-vec-missed`: Info on FAILED vectorization.

**Example Output:**
```
source.c:15:3: note: loop not vectorized: complicated control flow.
source.c:20:3: note: loop vectorized.
```

**Clang Flags:**
- `-Rpass=loop-vectorize`: Successes.
- `-Rpass-missed=loop-vectorize`: Failures.
- `-Rpass-analysis=loop-vectorize`: Why it failed (cost model, safety).

**Analysis of Failure:**

1. **"Cost model analysis"**: Compiler thinks scalar is faster (maybe stride is weird, or loop count is small).
   *Fix:* `#pragma omp simd` (Force it).
2. **"Data dependency"**: Aliasing.
   *Fix:* `restrict` or `#pragma GCC ivdep`.
3. **"Control flow"**: Early exit or switch.
   *Fix:* Rewrite code to remove branches.

---

## 💻 Implementation: Compiler vs Hand-Written Benchmarking

We will implement "SAXPY" ($Y = A \times X + Y$) in 3 ways and verify assembly.

### 🛠️ Step 1: Scalar Code (Baseline)

```c
// saxpy_scalar.c
void saxpy(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}
```

### 🛠️ Step 2: Auto-Vectorized Hints

```c
// saxpy_autovec.c
void saxpy(int n, float a, float * __restrict__ x, float * __restrict__ y) {
    // Tell compiler x and y are aligned to 32 bytes
    x = (float*)__builtin_assume_aligned(x, 32);
    y = (float*)__builtin_assume_aligned(y, 32);
    
    #pragma omp simd
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}
```

### 🛠️ Step 3: Vector Type Extension

```c
// saxpy_vectype.c
typedef float vec8 __attribute__((vector_size(32))); // 256-bit

void saxpy(int n, float a, float *x_scalar, float *y_scalar) {
    vec8* x = (vec8*)x_scalar;
    vec8* y = (vec8*)y_scalar;
    vec8 va = {a, a, a, a, a, a, a, a}; // Broadcast
    
    int chunks = n / 8;
    for (int i = 0; i < chunks; ++i)
        y[i] = va * x[i] + y[i];
}
```

### 🛠️ Step 4: Verification (Objdump)

Compile:
```bash
gcc -O3 -march=native -c saxpy_autovec.c -o autovec.o
objdump -d -M intel autovec.o | grep vfmadd
```
**Expected:** You should see `vfmadd231ps ymm...`. If you see `xmm`, it used AVX/SSE (128-bit). If you see `addss`, it's scalar!

---

## 🧪 Hands-On Labs

### Lab 6: Analyzing Vectorization Reports

**Objective:** Fix a non-vectorizing loop by reading compiler reports.

**File:** `debug_vec.c`

```c
#include <math.h>

void compute(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] > 0) {
            b[i] = sqrtf(a[i]);
            a[i] = b[i] + 1.0f;
        }
        // Implicit else: do nothing
    }
}
```

**Experiment:**
1. Compile: `gcc -O3 -fopt-info-vec-missed -march=native -c debug_vec.c`
2. Observer report: likely "control flow" or "cost model".
3. Add `#pragma omp simd`
4. Re-compile and check if it vectorized (Modern GCC can vectorize simple IFs using masks).

**Fixing Aliasing**
Change signature to `void compute(float* __restrict__ a, float* __restrict__ b, int n)` and see if report changes.

---

## 📝 Summary & Key Takeaways

1. **Compiler is First Line of Defense:** Always try `-O3 -march=native` first.
2. **Help the Compiler:** Use `const`, `restrict`, and `__builtin_assume_aligned`. Information allows optimization.
3. **OpenMP SIMD:** A portable way to enforce vectorization (`#pragma omp simd`) when the compiler is too timid.
4. **Analysis Tools:** `objdump` and `-fopt-info-vec` provide truth. Don't guess if it vectorized; look at the assembly or report.
5. **Vector Extensions:** Use `vector_size` attribute for quick C-style SIMD without intrinsic verbosity.

---

## 📚 Additional Resources

- [Compiler Explorer (godbolt.org)](https://godbolt.org/) - Paste code, select compiler (x86-64 gcc 13.2), add `-O3 -march=haswell`. See colorful assembly mapping!
- [LLVM Vectorization User Guide](https://llvm.org/docs/Vectorizers.html)

**Tomorrow:** Day 7 - Week 1 Review & Project (2D Convolution) - putting it all together!

*End of Day 006 - Total Lines: 1000+*
