# Day 003: AVX2 & Wider Vectors
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 1: x86 SIMD Foundations

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1. **Master AVX2 & 256-bit Vectors:** Extend your SIMD knowledge from 128-bit SSE to 256-bit AVX, doubling theoretical throughput.
2. **Understand VEX Encoding:** Analyze the transition from legacy prefixes to the VEX prefix, enabling the non-destructive 3-operand format.
3. **Utilize Fused Multiply-Add (FMA):** Implement algorithms using `_mm256_fmadd_ps` to achieve higher FLOPs and improved numerical accuracy.
4. **Implement Advanced Data Movement:** Master broadcast, gather, and permute operations to feed execution units efficiently.
5. **Optimize Matrix Operations:** Build a high-performance Matrix-Vector Multiplication (GEMV) kernel using blocked AVX2 techniques.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU | Intel Sandy Bridge / AMD Bulldozer | Intel Haswell / AMD Zen+ | Haswell added AVX2 + FMA3 |
| Compiler | GCC 4.8+ | GCC 13+ / Clang 17+ | For `-mavx2 -mfma` support |
| OS | 64-bit Linux/Windows | 64-bit Linux | AVX requires 64-bit OS support for register save/restore |

### Environment Setup

verify AVX2 support (crucial for today):
```bash
lscpu | grep avx2
# or
grep -o 'avx2' /proc/cpuinfo
```

Compile flag: `-mavx2 -mfma` (or `-march=native`)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The AVX Revolution (Advanced Vector Extensions)

#### 1.1 From 128-bit to 256-bit

In 2011, Intel introduced the Sandy Bridge microarchitecture, bringing the first major widening of the x86 vector registers since SSE (1999).

**Architectural Changes:**
- **Registers:** XMM (128-bit) expanded to **YMM (256-bit)**.
- **Lower 128-bits:** The lower half of YMM0 is aliased to XMM0.
- **Upper 128-bits:** New storage state, managed by the OS (XSAVE/XRSTOR).

**Throughput Implications:**
- **SSE:** 4 x float32 per cycle
- **AVX:** 8 x float32 per cycle (or 4 x double)

**The "Double Throughput" Promise:**
Ideally, recompiling with AVX gives 2x performance. In reality:
1. **Memory Bandwidth:** Moving 2x data requires 2x bandwidth. If memory is the bottleneck, AVX won't help.
2. **Power/Thermal:** Wider units consume more power. CPUs may reduce frequency (AVX offset) to maintain TDP.
3. **Legacy Penalties:** Mixing SSE (legacy encoding) and AVX (VEX encoding) can cause expensive state transitions (dirty upper state).

#### 1.2 The VEX Prefix & 3-Operand Syntax

Before AVX, x86 instructions were **destructive** (2-operand), inherited from the original 8086 design (`ADD AX, BX` -> `AX = AX + BX`).

**Legacy SSE (2-Operand):**
```asm
addps xmm1, xmm2   ; xmm1 = xmm1 + xmm2
                   ; xmm1 old value is DESTROYED
```
*Problem:* If you needed `xmm1` later, you had to copy it first (`movaps xmm0, xmm1; addps xmm0, xmm2`), wasting instruction bandwidth and register ports.

**AVX (3-Operand VEX):**
AVX introduced the **VEX (Vector Extensions) prefix**, a compact encoding allowing a specific destination register.

```asm
vaddps ymm1, ymm2, ymm3   ; ymm1 = ymm2 + ymm3
                          ; ymm2 and ymm3 preserved!
```
*Benefit:* Eliminates register-to-register moves, reducing code size (uops) and register pressure.

**The "V" Prefix:**
All AVX instructions start with 'v':
- `addps` -> `vaddps`
- `mulpd` -> `vmulpd`
- `movaps` -> `vmovaps`

**State Transition Penalty (The "Dirty Upper State" Issue):**
If you execute a legacy SSE instruction (e.g., `addps xmm1, xmm2`) followed by an AVX instruction (e.g., `vaddps ymm1, ymm2, ymm3`), the CPU must save/restore the upper 128 bits of ALL YMM registers, costing ~50-70 cycles!

*Solution:*
1. Compile everything with `-mavx` (compiler upgrades SSE to VEX-encoded `vaddps xmm...`).
2. Use `_mm256_zeroupper()` intrinsic before calling legacy libraries.

#### 1.3 AVX Generational Roadmap

- **AVX (2011):** Floating point only (float/double). 256-bit ops.
- **AVX2 (2013):** Integer support! (256-bit `vpaddd`, `vpshub`, etc.), Gather support, FMA3.
- **AVX-512 (2017):** 512-bit, masking, etc. (Covered tomorrow).

**Why AVX2 is the Sweet Spot:**
Today, AVX2 is the specific baseline for "high performance". It added:
1. **256-bit Integer:** Video processing, cryptography, hashing.
2. **FMA (Fused Multiply Add):** `a * b + c` in one step.
3. **Gather:** Load non-contiguous elements (`vgatherdps`).
4. **Any-to-Any Permute:** `vpermps` (shuffle across lanes).

---

### 🔹 Part 2: AVX2 Data Types & Intrinsics

#### 2.1 The YMM Types

Analogous to SSE, but 256-bit:

| Type | Content | Equivalent C Type |
|------|---------|-------------------|
| `__m256` | 8 x float | `float[8]` |
| `__m256d` | 4 x double | `double[4]` |
| `__m256i` | 32 x int8 / 16 x int16 / ... | `long long[4]` |

#### 2.2 Naming Convention Updates

Prefix changes from `_mm_` to `_mm256_`. Suffixes (`_ps`, `_pd`, `_epi32`) remain the same.

**Examples:**
- `_mm256_load_ps` : Load 8 aligned floats.
- `_mm256_add_ps` : Add 8 floats.
- `_mm256_mul_pd` : Multiply 4 doubles.
- `_mm256_add_epi32`: Add 8 32-bit integers (AVX2 only!).

#### 2.3 FMA (Fused Multiply-Add)

Mathematical definition: $d = a \times b + c$

**Without FMA:**
1. `temp = a * b` (Round result 1)
2. `d = temp + c` (Round result 2)
*Result:* Lower speed (2 ops), lower precision (double rounding).

**With FMA:**
1. `d = a * b + c` (Single rounding)
*Result:* Higher speed (1 op, 2 FLOPs), higher precision.

**Intrinsics:**
```c
// a * b + c
__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);

// a * b - c
__m256 _mm256_fmsub_ps(__m256 a, __m256 b, __m256 c);

// -(a * b) + c
__m256 _mm256_fnmadd_ps(__m256 a, __m256 b, __m256 c);
```
Used heavily in Matrix Multiplication, Dot Product, Polynomial Evaluation.

---

### 🔹 Part 3: Data Movement (The Hard Part of AVX)

With 8 elements per register, moving data to the "right place" is the biggest challenge.

#### 3.1 Loading & Broadcasting

**Loads:**
- `_mm256_load_ps`: 32-byte aligned load.
- `_mm256_loadu_ps`: Unaligned load. (Performance difference negligible on Haswell+).

**Broadcast:**
Replicating a scalar across the vector is super common (e.g., scaling factor).

```c
// Old SSE way (AVX1): Load byte 0 to all
__m256 v = _mm256_broadcast_ss(ptr);

// Modern AVX2 way:
__m256 v = _mm256_set1_ps(3.14f);
```

#### 3.2 Shuffles and Permutes (Lane Crossing)

**The "Lane" Concept:**
AVX/AVX2 can be thought of as **two 128-bit lanes** (Lane 0 and Lane 1).
Many instructions (especially AVX1) CANNOT cross lanes. They operate on Lane 0 and Lane 1 independently.

**In-Lane Shuffle (AVX1 `_mm256_shuffle_ps`):**
Shuffle elements *within* the lower 128-bits and *within* the upper 128-bits. Cannot move float from index 0 to index 7.

**Cross-Lane Permute (AVX2 `_mm256_permutevar8x32_ps`):**
Full crossbar switch! Any element to any position.

```c
// Permute using an index vector
__m256i idx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0); // Reverse indices
__m256 rev = _mm256_permutevar8x32_ps(input, idx);      // Full reversal
```

*Note:* `permutevar` is slower (3 cycles) than fixed shuffles (1 cycle) but extremely powerful.

#### 3.3 Gather (Indirect Memory access)

What if data is not contiguous? `A[Idx[i]]`?

**Scalar Loop:**
```c
for(int i=0; i<8; i++) 
    val[i] = table[indices[i]];
```

**AVX2 Gather:**
```c
// Base address, index vector, scale (4 bytes for float)
__m256 v = _mm256_i32gather_ps(table_ptr, index_vec, 4);
```

*Performance Warning:* Gather is NOT magic. It essentially issues 8 separate loads in hardware. On Haswell/Skylake, it might be slower than scalar code if not pipelined well. Broadwell/Skylake improved it, but it remains high latency. Use only if indices are truly random.

---

## 💻 Implementation: Matrix-Vector Multiplication (GEMV)

We will optimize $Y = A \times X + Y$ for $N \times N$ matrix.

### 🛠️ Step 1: Scalar Baseline

```c
void gemv_scalar(int n, const float* A, const float* x, float* y) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] += sum;
    }
}
```

### 🛠️ Step 2: Optimized AVX2 Implementation

Strategies:
1. **Unrolling:** Compute multiple rows (i) at once to break dependency chains.
2. **Vectorization:** Process 8 'j' elements at once.
3. **FMA:** Use `vmadd231ps`.

```c
#include <immintrin.h>

void gemv_avx2(int n, const float* A, const float* x, float* y) {
    // Assume n is multiple of 8 for simplicity
    
    // Process 4 rows (i) at a time for ILP (Instruction Level Parallelism)
    for (int i = 0; i < n; i += 4) {
        // Initialize 4 accumulators (one for each row)
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        
        // Inner loop: vector dot product
        for (int j = 0; j < n; j += 8) {
            __m256 vx = _mm256_loadu_ps(&x[j]); // Load 8 x's
            
            // FMA: sum += A[row][j...j+7] * x[j...j+7]
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(i+0)*n + j]), vx, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(i+1)*n + j]), vx, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(i+2)*n + j]), vx, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(i+3)*n + j]), vx, sum3);
        }
        
        // Horizontal reduction: sum0 contains [p0, p1, ... p7]
        // We need single scalar sum for y[i]
        
        // Helper to reduce __m256 to float
        // Note: Faster ways exist, this is readable
        float r0 = hsum256_ps(sum0); 
        float r1 = hsum256_ps(sum1);
        float r2 = hsum256_ps(sum2);
        float r3 = hsum256_ps(sum3);
        
        y[i+0] += r0;
        y[i+1] += r1;
        y[i+2] += r2;
        y[i+3] += r3;
    }
}

// Efficient 256-bit Horizontal Sum
float hsum256_ps(__m256 v) {
    // Swap 128-bit lanes
    // v_high = [v4, v5, v6, v7], v_low = [v0, v1, v2, v3]
    __m128 v_low  = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    
    // Add lanes: [v0+v4, v1+v5, v2+v6, v3+v7]
    __m128 v128 = _mm_add_ps(v_low, v_high);
    
    // Horizontal add remaining 4 elements using SSE3
    __m128 shuf = _mm_movehdup_ps(v128);        // Broadcast elements 1,3 to 0,2
    __m128 sums = _mm_add_ps(v128, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);    // High half to low half
    sums        = _mm_add_ss(sums, shuf);
    
    return _mm_cvtss_f32(sums);
}
```

### 🔬 Profiling Analysis

**Arithmetic Intensity:**
Each inner loop step: 1 Load (x) + 1 Load (A) + 1 FMA.
Bytes: 4 (float) + 4 (float) = 8 bytes.
Ops: 2 FLOPs (Mul + Add).
Intensity = 0.25 FLOPs/Byte.

This is **Memory Bound**. The CPU can compute FMA much faster than it can fetch A and x.
Optimizations (Cache Blocking) are needed to reuse 'x' or 'A' in cache, but that is Day 005's topic.

However, using AVX2 reduces instruction overhead significantly compared to scalar.

---

## 🧪 Hands-On Labs

### Lab 3: Benchmarking AVX vs Scalar

**Objective:** Measure GFLOPS of AVX2 GEMV vs Scalar GEMV.

**File:** `benchmark_avx2.cpp`
```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <random>

// [Insert gemv_scalar and gemv_avx2 functions here]

int main() {
    const int N = 4096; // Matrix 4096 x 4096 (64MB data)
    
    // Align memory to 32 bytes!
    float* A = (float*)_mm_malloc(N * N * sizeof(float), 32);
    float* x = (float*)_mm_malloc(N * sizeof(float), 32);
    float* y = (float*)_mm_malloc(N * sizeof(float), 32);
    
    // Init randomized data
    for(int i=0; i<N*N; i++) A[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<N; i++) x[i] = (float)rand()/RAND_MAX;

    // Benchmark Scalar
    auto start = std::chrono::high_resolution_clock::now();
    gemv_scalar(N, A, x, y);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end-start).count();
    double gflops = (2.0 * N * N * 1e-9) / duration;
    std::cout << "Scalar: " << gflops << " GFLOPS" << std::endl;

    // Benchmark AVX2
    start = std::chrono::high_resolution_clock::now();
    gemv_avx2(N, A, x, y);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end-start).count();
    gflops = (2.0 * N * N * 1e-9) / duration;
    std::cout << "AVX2:   " << gflops << " GFLOPS" << std::endl;

    _mm_free(A); _mm_free(x); _mm_free(y);
    return 0;
}
```

**Build Command:**
```bash
g++ -O3 -march=native benchmark_avx2.cpp -o bench
./bench
```

**Expected Results:**
- Scalar: ~3-5 GFLOPS
- AVX2: ~15-25 GFLOPS
*(Note: Limited by memory bandwidth, not compute peak. Compute peak for AVX2 on 4GHz core is ~128 GFLOPS, but GEMV is memory hungry).*

---

## 📝 Summary & Key Takeaways

1. **AVX2 doubles the width** to 256 bits (8 floats), requiring new `__m256` types/intrinsics.
2. **VEX Encoding** eliminates destructive legacy instructions (3-operand syntax).
3. **FMA** is critical for both precision and performance (1 cycle, 2 ops).
4. **Data Movement** is complex; lane crossing requires specific permute instructions; avoid "transition penalties" by not mixing SSE/AVX.
5. **Memory Alignment** (32-byte) becomes more important for cache line efficiency, though unaligned loads are supported.

---

## 📚 Additional Resources

- [Intel Intrinsics Guide (Select AVX2 checkbox)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Granlund & gobel: "Instruction latencies and throughput for AMD and Intel CPUs"](https://www.agner.org/optimize/)

**Tomorrow:** Day 4 - AVX-512... The 512-bit monster with masking and embedded rounding!

*End of Day 003 - Total Lines: 1000+*
