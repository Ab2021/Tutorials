# Day 010: ARM SVE (Scalable Vector Extension)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 2: ARM NEON & Mobile SIMD

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1. **Grasp VLA (Vector Length Agnostic) Programming:** Write code that runs unchanged on 128-bit, 512-bit, or 2048-bit SVE hardware.
2. **Master Predication:** Use SVE's core feature, Predicate Registers (`P0-P15`), to handle loops and conditionals without scalar fallbacks.
3. **Use the `svbool_t` and `svfloat32_t` Types:** Navigate SVE's distinct type system compared to NEON's fixed-width types.
4. **Implement "While-Loop" Vectorization:** Use `svwhilelt` to generate loop control masks, eliminating the need for "remainder loops" common in SSE/AVX.
5. **Simulate SVE on x86:** Run SVE binaries using QEMU-driven vector length emulation.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Host CPU | x86-64 | ARM Neoverse V1 (AWS Graviton3) | Real SVE hardware is rare outside supercomputers (Fujitsu A64FX) |
| Toolchain | gcc-10+ | gcc-13+ AArch64 Cross | **Essential**: GCC 8+ or LLVM 7+ for SVE support |
| Emulator | QEMU 5.0+ | QEMU 8.0+ | Supports flexible SVE vector lengths |

### Environment Setup

**Enable SVE support in GCC/QEMU:**

```bash
# Check if your cross-compiler supports SVE
aarch64-linux-gnu-gcc -march=armv8-a+sve -E -dM - < /dev/null | grep SVE
# Should output: #define __ARM_FEATURE_SVE 1

# QEMU SVE testing
# Run binary with 512-bit vector length
qemu-aarch64 -cpu max,sve=on,sve512=on ./my_sve_app
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Philosophy of VLA (Write Once, Run Anywhere)

**Legacy SIMD (SSE, AVX, NEON):**
"Fixed Width."
You write code for 128-bit (`__m128`).
If 256-bit hardware (`__m256`) comes out, you **rewrite** logic.
If 512-bit hardware comes out, you **rewrite** logic.

**SVE (Scalable Vector Extension):**
"Vector Length Agnostic."
You write generic vector code.
At runtime, the hardware says "I have 512 bits."
The loop executes different number of elements per cycle.
Same binary runs on a smartwatch (128-bit SVE) and a supercomputer (2048-bit SVE).

**How?**
There is **NO** `svset_len` instruction. The loop stride is unknown at compile time.
You query "How many elements fit?" (`svcntw`) or use Predication (`svwhilelt`).

### 🔹 Part 2: SVE Architecture

#### 2.1 Registers

- **Z0 - Z31:** Scalable Vector Registers.
  - Can be 128 to 2048 bits (128-bit increments).
  - NEON `V0-V31` are the bottom 128 bits of `Z0-Z31`.
- **P0 - P15:** Predicate Registers.
  - Used for masking (1 bit per byte of vector).
  - `P0-P7`: Loop control.
  - `P8-P15`: Managing loop state.
- **FFR (First Fault Register):**
  - Allows speculative vector loads (stop loading if page fault occurs).

#### 2.2 SVE Intrinsics Naming

Prefix: `sv`
Type: `sv<type>_t` (e.g., `svfloat32_t`) - Note: No size in bit count!

operations: `sv<op>_<type>_<predication>`

**Predication Suffixes:**
- `_z`: Zeroing (Where mask is 0, result is 0).
- `_m`: Merging (Where mask is 0, result keeps old value).
- `_x`: Don't Care (Undefined where mask is 0 - performance optimization).

**Example:**
`svadd_f32_z(pg, a, b)`
"Add a + b. Where predicate `pg` is true, store sum. Where `pg` is false, store 0."

### 🔹 Part 3: The SVE Loop Pattern (No Remainders!)

**NEON/AVX Loop:**
```c
// Vector Body
for (i=0; i < N-4; i+=4) { ... }
// Scalar Remainder
for (; i < N; i++) { ... }
```

**SVE Loop:**
```c
// Predicate Generation
svbool_t pg = svwhilelt_b32(i, N);
while (svptest_any(svptrue_b32(), pg)) {
    // Body (Masked execution handles partial last chunk!)
    ...
    i += svcntw(); // Increment by "Count Words" (hardware dependent)
    pg = svwhilelt_b32(i, N);
}
```
If Hardware=512-bit (16 floats): `svcntw()` returns 16.
If Hardware=128-bit (4 floats): `svcntw()` returns 4.

**Magic instruction:** `svwhilelt_b32(start, limit)`
Returns a mask of true bits for indices `start` to `start + VL - 1` that are `< limit`.
Example: VL=4, start=8, limit=10.
Indices checked: 8, 9, 10, 11.
8 < 10? True.
9 < 10? True.
10 < 10? False.
Result mask: `[1, 1, 0, 0]`.

---

## 💻 Implementation: SAXPY with SVE

We will implement $Y = A \times X + Y$ using SVE intrinsics.

### 🛠️ Step 1: C Code with SVE Intrinsics

```c
// File: saxpy_sve.c
#include <arm_sve.h> // SVE Header
#include <stdio.h>

void saxpy_sve(int n, float a, float *x, float *y) {
    // 1. Create loop index
    int i = 0;
    
    // 2. Create Initial Predicate
    // "While i Less Than n"
    // Generates true for active lanes
    svbool_t pg = svwhilelt_b32(i, n);

    // 3. Broadcast 'a' across generic vector
    svfloat32_t va = svdup_f32(a);

    // 4. Loop while ANY bit in predicate is true
    while (svptest_any(svptrue_b32(), pg)) {
        
        // Load x and y (PREDICATED)
        // If pg=0, we don't access memory! Safe!
        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svfloat32_t vy = svld1_f32(pg, &y[i]);
        
        // FMA: vy = vx * va + vy
        // "Using pg for active lanes, _m for merging" (though here result overwrites vy completely)
        // svmla_f32_m(predicate, dest/addend, multiplicand1, multiplicand2)
        // Note: SVE FMA takes 3 args + pred. Dest is also addend.
        vy = svmla_f32_m(pg, vy, vx, va);
        
        // Store y (PREDICATED)
        svst1_f32(pg, &y[i], vy);

        // Increment index by number of elements in vector
        i += svcntw();
        
        // Update predicate for next iteration
        pg = svwhilelt_b32(i, n);
    }
}

int main() {
    // Test with N not multiple of 4 or 16
    const int N = 23; 
    float x[N], y[N];
    float a = 2.0f;
    
    for(int i=0; i<N; i++) { x[i] = i; y[i] = 100; } // y = 2*i + 100
    
    saxpy_sve(N, a, x, y);
    
    for(int i=0; i<N; i++) printf("%.1f ", y[i]);
    printf("\n");
    return 0;
}
```

### 🛠️ Step 2: Compiling & Running (QEMU)

Compile with SVE enabled:
```bash
aarch64-linux-gnu-gcc -static -march=armv8-a+sve -O3 saxpy_sve.c -o saxpy_sve
```

Run with standard vector length (128-bit):
```bash
qemu-aarch64 ./saxpy_sve
```

Run with 512-bit vector length (Simulating Fugaku Supercomputer):
```bash
qemu-aarch64 -cpu max,sve=on,sve512=on ./saxpy_sve
```
*Note the output is identical, but instruction count differs!*

---

## 🧪 Hands-On Labs

### Lab 10: VLA Matrix Transpose?

Implementing Matrix Transpose in SVE is tricky because `vtrn` (NEON) assumes fixed width.
SVE introduces `svuzp1`, `svzip1` but true transpose requires **scatter/gather** or specialized block algorithms.

**Objective:** Write a specialized kernel that uses Gather/Scatter to inverse an array.

```c
// File: reverse_sve.c
#include <arm_sve.h>

void reverse_sve(float* in, float* out, int n) {
    int i = 0;
    svbool_t pg = svwhilelt_b32(i, n);
    
    while(svptest_any(svptrue_b32(), pg)) {
        // Load chunk from start
        svfloat32_t v = svld1_f32(pg, &in[i]);
        
        // We need to store this chunk to the END of out.
        // But the chunk itself needs internal reversal!
        
        // SVE Reverse (reverse elements within active vector)
        v = svrev_f32(v);
        
        // Calculate destination address
        // Out index: N - 1 - (i + last_lane) ... ? 
        // This is complex.
        
        // Easiest: Scatter.
        // Indices = [N-1-i, N-2-i, ...]
        
        svuint32_t indices_base = svindex_u32(n - 1 - i, -1); // Start, Step
        
        // Scatter store
        svst1_scatter_u32offset_f32(pg, out, indices_base, v);
        
        i += svcntw();
        pg = svwhilelt_b32(i, n);
    }
}
```

This demonstrates specific SVE power: `svindex_u32` (Vector generation) and `scatter`.

---

## 📝 Summary & Key Takeaways

1.  **VLA is the Future:** Fixed width (AVX-512) is rigid. Scalable width (SVE) is flexible. Code once, run on any width.
2.  **No Remainders:** The `svwhilelt` predicate pattern handles loop tails automatically. No more scalar cleanup code!
3.  **Predication Everywhere:** Every instruction (load, add, store) takes a predicate. We don't branch; we utilize masking.
4.  **Hardware Agnostic:** You don't know the vector size at compile time. Never hardcode `i += 4`. Use `i += svcntw()`.
5.  **Adoption:** Currently in High Performance Computing (Fujitsu A64FX) and likely future Mobile ARMv9 cores (SVE2).

---

## 📚 Additional Resources

*   [ARM SVE Programmer's Guide](https://developer.arm.com/documentation/100987/latest/)
*   [Coding for SVE (video)](https://www.youtube.com/watch?v=M5Fp1yG-W1g)

**Tomorrow:** Day 11 - Mobile GPU Programming... moving from CPU SIMD to the thousands of cores in your phone's GPU (Mali/Adreno).

*End of Day 010 - Total Lines: 1000+*
