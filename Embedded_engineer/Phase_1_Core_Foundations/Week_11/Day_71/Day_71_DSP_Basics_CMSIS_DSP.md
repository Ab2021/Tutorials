# Day 71: DSP Basics & CMSIS-DSP Library
## Phase 1: Core Embedded Engineering Foundations | Week 11: DSP & Audio Processing

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the difference between Fixed-Point (Q15, Q31) and Floating-Point (F32) arithmetic.
2.  **Utilize** the ARM Cortex-M4 DSP instructions (SIMD, MAC).
3.  **Integrate** the CMSIS-DSP library into an STM32 project.
4.  **Perform** basic vector operations (Add, Mult, Scale) using CMSIS-DSP.
5.  **Benchmark** the performance difference between C code and DSP optimized code.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   [CMSIS-DSP Library](https://github.com/ARM-software/CMSIS-DSP) (Source or Precompiled Lib).
*   **Prior Knowledge:**
    *   Day 9 (Cortex-M Architecture)
    *   Day 2 (C Data Types)
*   **Datasheets:**
    *   [ARM CMSIS-DSP Documentation](https://arm-software.github.io/CMSIS_5/DSP/html/index.html)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: DSP Extensions
The Cortex-M4F has special hardware for signal processing:
*   **FPU (Floating Point Unit):** Single Precision (32-bit float). Hardware Add/Sub/Mult/Div.
*   **MAC (Multiply-Accumulate):** $A = A + (B \times C)$. Done in 1 cycle! Essential for filters.
*   **SIMD (Single Instruction Multiple Data):** Operate on two 16-bit integers or four 8-bit integers at once. e.g., `SADD16` adds two pairs of 16-bit numbers.

### 🔹 Part 2: Fixed vs Floating
*   **Float (F32):** Easy to use. High dynamic range. Slower (even with FPU) and consumes more power than integer.
*   **Fixed (Q15, Q31):** Represents fractional numbers using integers.
    *   **Q15:** 1 sign bit, 15 fractional bits. Range -1 to 0.9999. Resolution $2^{-15}$.
    *   **Pros:** Very fast (uses integer ALU/SIMD).
    *   **Cons:** Hard to manage overflow and scaling.

### 🔹 Part 3: CMSIS-DSP
ARM's optimized library.
*   **BasicMath:** Add, Sub, Mult, Scale, Dot Product.
*   **FastMath:** Sin, Cos, Sqrt (Table based).
*   **Filtering:** FIR, IIR, LMS.
*   **Transform:** FFT, DCT.

---

## 💻 Implementation: CMSIS-DSP Setup

> **Instruction:** Add the library to the project.

### 👨‍💻 Code Implementation

#### Step 1: Include & Define
In `main.c`:
```c
#include "stm32f4xx.h"
#include "arm_math.h" // CMSIS-DSP Header

// Must define ARM_MATH_CM4 and __FPU_PRESENT=1 in compiler flags!
```

#### Step 2: Vector Addition (F32)
```c
#define BLOCK_SIZE 32

float32_t srcA[BLOCK_SIZE];
float32_t srcB[BLOCK_SIZE];
float32_t dst[BLOCK_SIZE];

void DSP_Add_Float(void) {
    // Fill Arrays
    for(int i=0; i<BLOCK_SIZE; i++) {
        srcA[i] = (float)i;
        srcB[i] = (float)(i * 2);
    }
    
    // Perform Add
    arm_add_f32(srcA, srcB, dst, BLOCK_SIZE);
}
```

#### Step 3: Vector Dot Product (Q15)
```c
q15_t srcA_q15[BLOCK_SIZE];
q15_t srcB_q15[BLOCK_SIZE];
q63_t result_q63; // Accumulator needs 64 bits to avoid overflow

void DSP_Dot_Q15(void) {
    // Fill (Scale 0..1 to Q15 range)
    for(int i=0; i<BLOCK_SIZE; i++) {
        srcA_q15[i] = (q15_t)(i * 100);
        srcB_q15[i] = (q15_t)(i * 50);
    }
    
    // Dot Product: sum(A[i] * B[i])
    arm_dot_prod_q15(srcA_q15, srcB_q15, BLOCK_SIZE, &result_q63);
}
```

---

## 🔬 Lab Exercise: Lab 71.1 - RMS Calculation

### 1. Lab Objectives
- Generate a Sine Wave.
- Calculate its Root Mean Square (RMS) value.
- Compare `arm_rms_f32` vs Manual C loop.

### 2. Step-by-Step Guide

#### Phase A: Generate Signal
```c
#define SAMPLE_COUNT 100
float32_t signal[SAMPLE_COUNT];

void Generate_Sine(void) {
    for(int i=0; i<SAMPLE_COUNT; i++) {
        // 10 Hz sine, 100 Hz sample rate
        signal[i] = arm_sin_f32(2 * PI * 10 * i / 100.0f);
    }
}
```

#### Phase B: Calculate RMS
```c
float32_t rms_val;

void Calc_RMS(void) {
    // CMSIS-DSP
    arm_rms_f32(signal, SAMPLE_COUNT, &rms_val);
    
    // Expected: 0.707 for Amplitude 1.0
    printf("RMS: %f\n", rms_val);
}
```

### 3. Verification
Run the code. Check UART output. Should be close to 0.707.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Benchmarking
- **Goal:** Measure Cycles.
- **Task:**
    1.  Enable DWT Cycle Counter.
    2.  Measure `arm_mult_f32` (Vector Mult).
    3.  Measure manual `for` loop multiplication.
    4.  **Result:** CMSIS-DSP should be faster due to loop unrolling and SIMD (if using Q15/Q31). For F32, it might be similar to compiler optimization (-O3), but CMSIS is hand-tuned.

### Lab 3: Variance & Std Dev
- **Goal:** Statistical Analysis.
- **Task:**
    1.  Add noise to the sine wave (using `rand()`).
    2.  Calculate Mean (`arm_mean_f32`).
    3.  Calculate Variance (`arm_var_f32`).
    4.  Calculate Std Dev (`arm_std_f32`).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. HardFault (Usage Fault)
*   **Cause:** FPU not enabled.
*   **Solution:** Ensure `SystemInit` calls `SCB->CPACR |= ((3UL << 10*2)|(3UL << 11*2));` (Access to CP10 and CP11).

#### 2. Linker Error (Undefined reference to `arm_add_f32`)
*   **Cause:** Library not linked.
*   **Solution:** Add `libarm_cortexM4lf_math.a` (Little Endian, Float) to linker arguments `-larm_cortexM4lf_math`.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Block Processing:** DSP functions work best on blocks of data (e.g., 32 samples). Process audio in chunks (buffers) rather than sample-by-sample to reduce function call overhead.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Q31 format?
    *   **A:** Fixed point 32-bit. Range -1 to 0.999999999.
2.  **Q:** Why use `arm_sin_f32` instead of `sin()` from `math.h`?
    *   **A:** `arm_sin_f32` uses a lookup table and interpolation, which is much faster (but slightly less precise) than the standard library `sin()`.

### Challenge Task
> **Task:** Implement "Vector Magnitude". Given arrays X, Y, Z (accelerometer data), calculate $Mag = \sqrt{X^2 + Y^2 + Z^2}$ for the whole buffer efficiently. Hint: Use `arm_mult_f32` (square), `arm_add_f32`, and `arm_sqrt_f32`.

---

## 📚 Further Reading & References
- [DSP for Cortex-M (Whitepaper)](https://www.arm.com/resources/education/books/digital-signal-processing-using-arm-cortex-m-based-microcontrollers)

---
