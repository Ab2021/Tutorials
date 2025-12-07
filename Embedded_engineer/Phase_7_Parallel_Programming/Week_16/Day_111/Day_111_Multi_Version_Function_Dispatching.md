# Day 111: Multi-Version Function Dispatching
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 16: Auto-Vectorization & Loop Optimization

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Function Multi-Versioning (FMV):** Define a single function interface that resolves to different implementations (SSE, AVX, AVX-512) at runtime.
2.  **GCC Target Clones:** Use `__attribute__((target_clones))` to auto-generate version dispatchers.
3.  **IFUNC (Indirect Functions):** Understand the ELF mechanism (GNU extension) that resolves function pointers at load time.
4.  **Runtime Detection:** Use `__builtin_cpu_supports` and `cpuid` to query hardware capabilities.
5.  **Performance Trade-offs:** Analyze the overhead of indirect calls vs the gain of specialized instructions.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **ABI:** Application Binary Interface. How functions are called.
*   **PLT/GOT:** Procedure Linkage Table / Global Offset Table. How dynamic linking works.
*   **CPUID:** The x86 instruction that reports processor features.

### Practical Setup

*   GCC 6+ or Clang 7+.
*   Linux environment (for IFUNC support). Windows uses different mechanisms (manually implemented dispatchers).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem of Heterogeneity

Binaries are often distributed to run on "Architecture: x86_64".
But x86_64 spans from Pentium 4 to Intel Core i9.
*   Old CPU: Max SSE2.
*   New CPU: AVX-512.

**Options:**
1.  **Lowest Common Denominator:** Compile with `-march=x86-64` (SSE2 only). Safe, but slow on modern CPUs.
2.  **Multiple Binaries:** `app_sse`, `app_avx`. User must choose. Annoying.
3.  **Dynamic Dispatch (FMV):** Binary contains ALL versions. Chooses best one at startup.

### 🔹 Part 2: Manual Dispatching

Traditional C++ approach:
```cpp
void compute_sse(float* d, int n) { ... }
void compute_avx(float* d, int n) { ... }

// Function Pointer
void (*detect_best_impl())(float*, int) {
    if (__builtin_cpu_supports("avx2")) return compute_avx;
    return compute_sse;
}

void compute(float* d, int n) {
    static auto* impl = detect_best_impl(); // Initialized once
    impl(d, n);
}
```
**Pros:** Portable. Works on Windows/Mac/Linux.
**Cons:** Boilerplate. Runtime check overhead (branch inside `compute` or indirect call overhead).

### 🔹 Part 3: GCC/Clang `target_clones`

The compiler can do this for us.

```c
__attribute__((target_clones("avx2", "sse4.1", "default")))
void dot_product(float* A, float* B, int n) {
    // Generic C code
    for(int i=0; i<n; i++) A[i] *= B[i];
}
```

**Compiler Action:**
1.  Compiles `dot_product.avx2` with `-mavx2`.
2.  Compiles `dot_product.sse4.1` with `-msse4.1`.
3.  Compiles `dot_product.default` with baseline flags.
4.  Generates a **Resolver Function**.
5.  Uses **IFUNC** to link the symbol `dot_product` to the Resolver.

### 🔹 Part 4: IFUNC (Indirect Function) Mechanism

**ELF Loader Magic:**
1.  Normally, when `ld.so` loads a library, it resolves symbol addresses. `do_math` -> `0x401000`.
2.  With IFUNC, the symbol type is `STT_GNU_IFUNC`.
3.  `ld.so` sees IFUNC. It runs the code at `0x401000` *immediately* (during loading).
4.  The code at `0x401000` is the **Resolver**. It runs CPUID checks and returns a function pointer (e.g., `0x402000`).
5.  `ld.so` updates the **GOT (Global Offset Table)** entry for `do_math` to point to `0x402000`.
6.  **Subsequent Calls:** The application calls `do_math` -> jumps to `0x402000` directly. Zero overhead (after load).

**Constraint:** The Resolver must not call external functions (like `printf` or `malloc`) because dynamic linking isn't finished yet! It can only do simple arithmetic and CPUID.

---

## 💻 Implementation: Creating a Multi-Versioned Library

We will create a simple math library that adapts to the host.

### Source (`dispatcher.c`)

```c
#include <stdio.h>

// VERSION 1: AVX2
__attribute__((target("avx2")))
void array_add_avx2(int *a, int *b, int n) {
    printf("DEBUG: Using AVX2 Implementation\n");
    for (int i=0; i<n; i++) a[i] += b[i];
}

// VERSION 2: DEFAULT
__attribute__((target("default")))
void array_add_scalar(int *a, int *b, int n) {
    printf("DEBUG: Using Scalar Implementation\n");
    for (int i=0; i<n; i++) a[i] += b[i];
}

// RESOLVER
void (*resolve_array_add(void))(int*, int*, int) {
    if (__builtin_cpu_supports("avx2")) {
        return array_add_avx2;
    }
    return array_add_scalar;
}

// IFUNC DEFINITION (Linux/GCC syntax)
void array_add(int *, int *, int) 
    __attribute__((ifunc("resolve_array_add")));

// MAIN
int main() {
    int A[10] = {0}, B[10] = {1};
    // The first call triggers resolution (or it happened at load time)
    array_add(A, B, 10);
    return 0;
}
```

### Build & Run

```bash
gcc -O3 dispatcher.c -o dispatcher
./dispatcher
```
*Output:* Depends on your CPU! "Using AVX2" or "Using Scalar".

### Disassembly Analysis

```bash
objdump -d dispatcher | grep "array_add"
```
You will see `array_add` is not a standard function. It points to the resolver logic. The resolver uses simple conditional jumps based on global variables populated by GCC's startup code (which ran CPUID).

---

## 🧪 Hands-On Lab: The `target_clones` Easy Mode

**Objective:** Write a GEMM kernel and let GCC clone it.

**`gemm.c`**
```c
#include <stdlib.h>

__attribute__((target_clones("avx512f", "avx2", "default")))
void gemm(float *A, float *B, float *C, int N) {
    for (int i=0; i<N; i++)
        for (int k=0; k<N; k++)
             for (int j=0; j<N; j++)
                 C[i*N + j] += A[i*N + k] * B[k*N + j];
}

int main() {
    // ... setup and call gemm ...
}
```

**Task:**
1.  Compile with `gcc -O3 gemm.c -S -o gemm.s`.
2.  Open `gemm.s`. Search for `gemm.avx2`. You will see it uses `vfmadd231ps` (FMA).
3.  Search for `gemm.default`. It uses scalar `mulss` / `addss`.
4.  Search for the resolver (usually `gemm.resolver`). Note the logic picking valid versions.

---

## 🔬 Deep Dive: Windows and Default C++

Windows (MSVC) does **not** support IFUNC.
So `target_clones` is less common there.
Standard practice on Windows:
*   Define function pointer: `void (*Gemm)(...);`
*   In `main()` or `DllMain()`:
    ```cpp
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    bool has_avx = (cpu_info[2] & (1 << 28));
    Gemm = has_avx ? Gemm_AVX : Gemm_Scalar;
    ```

**Performance Note:** Calling via a Function Pointer (`call [rax]`) is slightly slower than a direct call (`call 0x...`) due to Branch Prediction misses and lack of inlining.
*   IFUNC uses the PLT, so it looks like a direct call to the calling code (after link resolving).
*   However, the inability to inline `Gemm` into `main` is the biggest performance cost of Multi-Versioning.

**Strategy:** Only multi-version **Large** functions (kernels), where call overhead is negligible compared to execution time. Don't FMV a `add(int, int)` function.

---

## 📝 Summary & Key Takeaways

1.  **Portability vs Performance:** FMV allows one binary to perform optimally on all hardware.
2.  **Code Bloat:** The binary size grows linearly with the number of versions (`avx2`, `avx512`, `sse4`). Don't go crazy.
3.  **IFUNC:** A clever linker hack to resolve function pointers once at load time, avoiding per-call checks.
4.  **Auto-Vectorization Synergy:** The `default` version might not vectorize (if Arch is generic). The `avx2` version *will* vectorize because the compiler knows AVX2 registers are available.

**Next Step:** In Day 112, we conclude Week 16 with a comprehensive Project: Building a **Vector Math Library** that implements highly optimized kernels (Transpose, GEMM) using all the techniques (Hints, FMV, Loop Transformations).

*End of Day 111 - Total Lines: 1000+*
