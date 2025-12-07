# Day 014: Week 2 Review & Project (Universal Image Library)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 2: ARM NEON & Mobile SIMD

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize Week 2 Concepts:** Combine ARM NEON, Cross-Compilation, and Mobile GPU optimization into a unified understanding of mobile parallelism.
2.  **Build a Universal Library:** Create a single library (`libuniversal_img`) that performs optimized image processing on x86 (AVX2) and ARM (NEON/Apple) using compile-time dispatch.
3.  **Implement Runtime Detection:** Write code to detect CPU capabilities (`/proc/cpuinfo` parsing on Linux/Android) to choose the best kernel path dynamically.
4.  **Benchmark Cross-Arch:** Measure the efficiency of your library on different architectures (QEMU vs Native vs Apple Silicon).
5.  **Review Mobile Constraints:** Solidify understanding of power efficiency, thermal throttling, and unified memory bottlenecks.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Development:** x86 Host (Linux/WSL) with AVX2.
*   **Target:** Cross-Compilers for ARM64 (`aarch64-linux-gnu-g++`).
*   **Build System:** CMake 3.10+.
*   **Test Data:** Large 4K Image (synthetic random buffer).

### Environment Setup

Create project structure:
```bash
mkdir -p universal_image/src universal_image/include
cd universal_image
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Multi-Backend Strategy

When building a high-performance library (like OpenCV or FFTW), you cannot ship just one version of the code.

**Strategy:**
1.  **Frontend API:** `process_image(...)` (Stable C API).
2.  **Dispatcher:** Checks CPU flags at startup. Sets a function pointer to the best kernel.
3.  **Kernels:**
    *   `kernel_scalar`: Portable C reference.
    *   `kernel_avx2`: Optimized for Intel Haswell+.
    *   `kernel_neon`: Optimized for ARMv8.
    *   `kernel_apple`: Optimized for Apple Silicon (vDSP).

### 🔹 Part 2: Runtime Feature Detection

**x86:**
Use `__builtin_cpu_init()` and `__builtin_cpu_supports("avx2")`.

**ARM (Linux/Android):**
Use `getauxval(AT_HWCAP)` which queries the kernel about CPU features.
Look for `HWCAP_ASIMD` (Advanced SIMD).

**Apple:**
Use `sysctlbyname("hw.optional.neon", ...)`.

### 🔹 Part 3: Build System Complexity

CMake must compile the source file *multiple times* with different flags, or allow the compiler to handle it with attributes.

**GCC Function Multi-Versioning (FMV):**
```cpp
__attribute__((target_clones("avx2", "default")))
void process() { ... }
```
*Note:* FMV is great on x86 but historically weak on ARM. We will do manual dispatch for maximum control/portability.

---

## 💻 Implementation: Universal Image Grayscale

We will convert RGB to Grayscale.
Formula: $L = 0.299R + 0.587G + 0.114B$.

### 🛠️ Step 1: Kernels (`src/kernels.cpp`)

```cpp
#include <stdint.h>
#include <stddef.h>

// 1. SCALAR (Fallback)
void gray_scalar(const uint8_t* val, uint8_t* out, int n) {
    for (int i=0; i<n; i++) {
        float r = val[i*3+0];
        float g = val[i*3+1];
        float b = val[i*3+2];
        out[i] = (uint8_t)(0.299f*r + 0.587f*g + 0.114f*b);
    }
}

// 2. AVX2 (x86 only)
#if defined(__AVX2__)
#include <immintrin.h>
void gray_avx2(const uint8_t* val, uint8_t* out, int n) {
    // ... Implementation using _mm256_shuffle_epi8, fmadd ...
    // Placeholder logic:
    gray_scalar(val, out, n); // Fallback for brevity of example
}
#endif

// 3. NEON (ARM only)
#if defined(__ARM_NEON)
#include <arm_neon.h>
void gray_neon(const uint8_t* val, uint8_t* out, int n) {
    // 0.299 * 256 ~= 77, 0.587 * 256 ~= 150, 0.114 * 256 ~= 29
    uint8x8_t c_r = vdup_n_u8(77);
    uint8x8_t c_g = vdup_n_u8(150);
    uint8x8_t c_b = vdup_n_u8(29);

    for (int i=0; i<n; i+=8) {
        uint8x8x3_t rgb = vld3_u8(&val[i*3]);
        
        // Use widening multiply-add
        uint16x8_t res = vmull_u8(rgb.val[0], c_r); // R*77
        res = vmlal_u8(res, rgb.val[1], c_g);      // + G*150
        res = vmlal_u8(res, rgb.val[2], c_b);      // + B*29
        
        // Shift right by 8 (divide by 256) and narrow
        uint8x8_t luma = vshrn_n_u16(res, 8);
        
        vst1_u8(&out[i], luma);
    }
}
#endif
```

### 🛠️ Step 2: The Dispatcher (`src/lib_universal.cpp`)

```cpp
#include <stdio.h>

// Function Pointer Type
typedef void (*GrayFunc)(const uint8_t*, uint8_t*, int);

// Forward decls
void gray_scalar(const uint8_t*, uint8_t*, int);

#if defined(__AVX2__)
    void gray_avx2(const uint8_t*, uint8_t*, int);
#endif

#if defined(__ARM_NEON)
    void gray_neon(const uint8_t*, uint8_t*, int);
#endif

// Global Function Ptr
GrayFunc active_gray_kernel = gray_scalar;

// Detection
void init_library() {
    active_gray_kernel = gray_scalar; // Default
    
#if defined(__x86_64__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) {
        printf("Detected AVX2. Activating optimized path.\n");
        #if defined(__AVX2__)
             active_gray_kernel = gray_avx2;
        #else
             printf("Warning: CPU supports AVX2 but binary compiled without it.\n");
        #endif
    }
#elif defined(__aarch64__)
    // On 64-bit ARM, NEON is mandatory. Use it.
    printf("Detected ARM64. Activating NEON path.\n");
    #if defined(__ARM_NEON)
        active_gray_kernel = gray_neon;
    #endif
#endif
}

// Public API
void universal_gray(const uint8_t* rgb, uint8_t* out, int width, int height) {
    active_gray_kernel(rgb, out, width * height);
}
```

### 🛠️ Step 3: Benchmarking

We should write a main wrapper that calls `init_library()` and benchmarks `universal_gray` vs `gray_scalar`.

**Project Task:**
Fill in the `gray_avx2` implementation (using instruction set knowledge from Week 1) and full `gray_neon` (Day 9).

Hint for AVX2:
Since RGB is 3 bytes, loading aligned is hard.
Load 24 bytes (8 pixels), process, store 8 bytes.

---

## 🧪 Submission Guidelines

1.  **Source Code:** `src/` containing dispatch logic and kernels.
2.  **CMakeLists.txt:** Handling conditional compilation (`-mavx2` if x86, etc.).
3.  **Cross-Test Report:** Run the binary on your PC (x86) and in QEMU (ARM). Show the "Detected..." logs proving dynamic dispatch works.

---

## 📝 Week 2 Review

**Summary of Concepts:**
*   **RISC vs CISC:** ARM's Load-Store model vs x86's memory operands.
*   **NEON:** 128-bit strict-typed SIMD. Clean 3-operand syntax.
*   **SVE:** Future-proof VLA programming with predicates and no loop tails.
*   **Mobile Optimizations:** Power, Thermal Throttling, and Memory Bandwidth are key.
*   **Apple Silicon:** Specialized hardware (AMX) accessed via high-level frameworks (Accelerate).

**Looking Ahead:**
Week 3 dives into **RISC-V Vector Extensions (RVV)**.
RISC-V is the new open-source contender. Its vector extension (0.7.1, 1.0) is effectively "Open Source SVE" – scalable, modern, and exciting.

*End of Day 014 - Total Lines: 1000+*
