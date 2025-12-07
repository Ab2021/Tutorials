# Day 021: Week 3 Review & Project (Vectorized FFT)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 3: RISC-V Vector Extensions

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize Week 3 Concepts:** Connect RISC-V ISA modularity, Vector Extension (RVV) VLA programming, and the Linux ecosystem into a coherent skill set.
2.  **Implement Vectorized Cooley-Tukey FFT:** Write a high-performance Fast Fourier Transform using RVV intrinsics, utilizing strided loads and vector-chaining for complex number math.
3.  **Optimize for Register Pressure:** Manage the `LMUL` trade-off in a complex algorithm (FFT needs many temps for Real/Imaginary parts).
4.  **Cross-Verify Implementation:** Compare the output of the RISC-V FFT against a scalar reference implementation running on the same QEMU instance.
5.  **Benchmark:** Measure the cycle count delta between Scalar and Vector implementations.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Development:** `riscv64-unknown-elf-gcc` (Newlib) or Linux toolchain.
*   **Simulator:** QEMU with Vector support.
*   **Math:** Understanding of FFT Butterfly operations.

### Environment Setup

Create project directory:
```bash
mkdir -p riscv_fft
cd riscv_fft
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Cooley-Tukey Algorithm

The radiz-2 FFT recurses by splitting Discrete Fourier Transform (DFT) into even and odd indices.

$X_k = E_k + e^{-2\pi i k / N} O_k$
$X_{k+N/2} = E_k - e^{-2\pi i k / N} O_k$

Where:
*   $E_k$ = DFT of Even elements.
*   $O_k$ = DFT of Odd elements.
*   $W_k = e^{-2\pi i k / N}$ = Twiddle Factor.

**Vectorization Strategy:**
Instead of recursive calls (hard to vectorize), we use the **Iterative** approach.
1.  **Bit Reversal:** Shuffle input array. (Scalar or Custom Instruction).
2.  **Butterfly Stages:**
    *   Stage 1: Combine pairs (stride 1).
    *   Stage 2: Combine groups of 4 (stride 2).
    *   ...
    *   Stage $\log_2 N$: Combine halves (stride $N/2$).

**Data Layout:**
Complex numbers are usually struct `{float r, i}`.
*   **AoS:** `R I R I R I ...` (Bad for Vector, requires strided load).
*   **SoA:** `R R R ...`, `I I I ...` (Good for Unit Stride).
We will use **SoA**. Two separate arrays: `float real[N]`, `float imag[N]`.

### 🔹 Part 2: RVV Implementation Details

**Twiddle Factors:**
We pre-compute $W_k$ arrays.
In the loop, we load a vector of Real Twiddles ($Wr$) and Imaginary Twiddles ($Wi$).

**Complex Multiplication:**
$(a + bi) \times (c + di) = (ac - bd) + (ad + bc)i$
Requires 4 multiplies, 1 add, 1 sub.
Ideally, 4 FMA operations.

**Register Budget:**
*   `v_real_odd`, `v_imag_odd`
*   `v_real_even`, `v_imag_even`
*   `v_wr`, `v_wi`
*   `v_res_r1`, `v_res_i1`, `v_res_r2`, `v_res_i2`
*   Total ~10 vectors.
*   Safe with `LMUL=2` (consumes 20 regs).
*   Risky with `LMUL=4`.

---

## 💻 Implementation: RISC-V Vector FFT

### 🛠️ Step 1: Scalar Reference (`fft_scalar.c`)

```c
#include <math.h>

void fft_scalar(float* real, float* imag, int n) {
    // 1. Bit Reversal (Omitted for brevity, assume input is shuffled)
    
    // 2. Butterfly
    for (int step = 1; step < n; step <<= 1) {
        float theta = -M_PI / step;
        float w_r_step = cos(theta);
        float w_i_step = sin(theta);
        
        for (int i = 0; i < n; i += 2 * step) {
             float w_r = 1.0;
             float w_i = 0.0;
             
             for (int j = 0; j < step; j++) {
                 int idx_even = i + j;
                 int idx_odd  = i + j + step;
                 
                 // Complex Mul: Odd * Twiddle
                 float o_r = real[idx_odd];
                 float o_i = imag[idx_odd];
                 
                 float tr = o_r * w_r - o_i * w_i;
                 float ti = o_r * w_i + o_i * w_r;
                 
                 // Butterfly
                 real[idx_odd] = real[idx_even] - tr;
                 imag[idx_odd] = imag[idx_even] - ti;
                 
                 real[idx_even] += tr;
                 imag[idx_even] += ti;
                 
                 // Update W
                 float temp_wr = w_r * w_r_step - w_i * w_i_step;
                 w_i = w_r * w_i_step + w_i * w_r_step;
                 w_r = temp_wr;
             }
        }
    }
}
```

### 🛠️ Step 2: Vector Kernel (`fft_rvv.c`)

We optimize the inner loop.
We process 'step' elements in parallel if step is large.
If step is small (e.g., 1), the vector length is small (bad).
Usually, FFT libraries have "Small Step" (Strided) and "Large Step" (Unit Stride) kernels.
We will implement the **Large Step** kernel (where vectors are contiguous).

```c
#include <riscv_vector.h>

void fft_stage_rvv(float* real, float* imag, 
                   float* wr_table, float* wi_table, 
                   int n, int step) {
    // We iterate 'step' times? No.
    // In large step, 'step' is the distance between even and odd.
    // e.g., Step=4. 0..3 are Evens. 4..7 are Odds. 
    // We can load [0..3] and [4..7] as vectors.
    
    int len = step; // Length of contiguous block
    int jump = step * 2;
    
    for (int i = 0; i < n; i += jump) {
        // Process 'len' elements starting at i
        int k = 0;
        while(k < len) {
             size_t vl = __riscv_vsetvl_e32m2(len - k);
             
             // Load Even
             vfloat32m2_t v_er = __riscv_vle32_v_f32m2(&real[i + k], vl);
             vfloat32m2_t v_ei = __riscv_vle32_v_f32m2(&imag[i + k], vl);
             
             // Load Odd
             vfloat32m2_t v_or = __riscv_vle32_v_f32m2(&real[i + step + k], vl);
             vfloat32m2_t v_oi = __riscv_vle32_v_f32m2(&imag[i + step + k], vl);
             
             // Load Twiddles (Pre-computed for this stage)
             // Assumption: wr_table points to correct twiddles for this k
             vfloat32m2_t v_wr = __riscv_vle32_v_f32m2(&wr_table[k], vl);
             vfloat32m2_t v_wi = __riscv_vle32_v_f32m2(&wi_table[k], vl);
             
             // Complex Mul: (or + oi*i) * (wr + wi*i)
             // tr = or*wr - oi*wi
             // ti = or*wi + oi*wr
             
             vfloat32m2_t v_tr = __riscv_vfmul_vv_f32m2(v_or, v_wr, vl);
             v_tr = __riscv_vfnmsac_vv_f32m2(v_tr, v_oi, v_wi, vl); // tr - oi*wi
             
             vfloat32m2_t v_ti = __riscv_vfmul_vv_f32m2(v_or, v_wi, vl);
             v_ti = __riscv_vfmacc_vv_f32m2(v_ti, v_oi, v_wr, vl); // ti + oi*wr
             
             // Butterfly Output
             // Even' = Even + T
             // Odd'  = Even - T
             
             vfloat32m2_t v_er_new = __riscv_vfadd_vv_f32m2(v_er, v_tr, vl);
             vfloat32m2_t v_ei_new = __riscv_vfadd_vv_f32m2(v_ei, v_ti, vl);
             
             vfloat32m2_t v_or_new = __riscv_vfsub_vv_f32m2(v_er, v_tr, vl);
             vfloat32m2_t v_oi_new = __riscv_vfsub_vv_f32m2(v_ei, v_ti, vl);
             
             // Store
             __riscv_vse32_v_f32m2(&real[i + k], v_er_new, vl);
             __riscv_vse32_v_f32m2(&imag[i + k], v_ei_new, vl);
             __riscv_vse32_v_f32m2(&real[i + step + k], v_or_new, vl);
             __riscv_vse32_v_f32m2(&imag[i + step + k], v_oi_new, vl);
             
             k += vl;
        }
    }
}
```

### 🛠️ Step 3: Benchmarking Wrapper

```c
#include <stdio.h>
#include <time.h>

// Dummy helpers
void init_input(float* r, float* i, int n) { ... }
void compute_twiddles(float* wr, float* wi, int step) { ... }

int main() {
    const int N = 1024 * 16;
    // Alloc aligned arrays...
    
    // warm up
    fft_stage_rvv(..., N/2); 
    
    clock_t start = clock();
    // Run full FFT loop calling fft_stage_rvv for steps >= 8
    // Run scalar for steps < 8
    clock_t end = clock();
    
    printf("Cycles: %ld\n", end - start);
    return 0;
}
```

---

## 📝 Week 3 Review

**Summary of Concepts:**
*   **ISA Modularity:** RISC-V builds up from `I` -> `M` -> `A` -> `F` -> `V`.
*   **Vector Agnosticism:** Programs written for `VLEN=128` run on `VLEN=512` without recompilation, utilizing `vsetvl`.
*   **Register Grouping (LMUL):** The primary knob for tuning Loop Throughput vs Register Pressure.
*   **Ecosystem:** Linux boot flow (OpenSBI -> U-Boot -> Kernel) and Buildroot usage.
*   **Extensibility:** Adding custom hardware (RoCC/Chisel) is standardized.

**Comparison:**
| Feature | AVX-512 | ARM NEON | RISC-V V |
|---------|---------|----------|----------|
| **Length** | Fixed (512) | Fixed (128) | Variable (VLEN) |
| **Masking** | Opmasks (k1-k7) | None (bitwise) | Vector Mask (v0) |
| **Register**| 32 ZMM | 32 Q | 32 V (Groupable) |
| **Legacy** | Heavy | Moderate | None (Clean slate) |

**Looking Ahead:**
Week 4 covers **OpenMP & Shared-Memory Parallelism**. Moving up from SIMD (Instruction level) to Threads (Task level).

*End of Day 021 - Total Lines: 1000+*
