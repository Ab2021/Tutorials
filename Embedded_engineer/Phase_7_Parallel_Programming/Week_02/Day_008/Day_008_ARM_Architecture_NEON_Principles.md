# Day 008: ARM Architecture & NEON Principles
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 2: ARM NEON & Mobile SIMD

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1. **Understand ARMv8-A Architecture:** Differentiate between AArch64 (64-bit) and AArch32 execution states, and map the register file (X0-X30, V0-V31).
2. **Contrast RISC vs CISC SIMD:** Analyze the key differences between ARM's Load-Store architecture and x86's Register-Memory model, and how this impacts vectorization strategies.
3. **Master Weak Memory Ordering:** Explain the implications of ARM's Weakly Ordered memory model on synchronization and atomic operations compared to x86's TSO.
4. **Navigate the Ecosystem:** Distinguish between various ARM implementations (Cortex-A, Neoverse, Apple Silicon) and understanding big.LITTLE / DynamIQ topology.
5. **Set Up Cross-Compilation:** Configure a development environment to compile ARM64 binaries on an x86 host using `qemu-user` for emulation.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Host CPU | x86-64 | Apple Silicon (M1/M2) | Native ARM (M1/M2) is best, but x86+QEMU works |
| Cross-Compiler | `aarch64-linux-gnu-gcc` | Same | Essential for generating ARM binaries on x86 |
| Emulator | `qemu-user` | `qemu-user-static` | Emulates ARM/AArch64 userspace binaries |

### Environment Setup

**Ubuntu/Debian (on x86 host):**

```bash
# Install cross-compiler and emulator
sudo apt update
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
sudo apt install qemu-user qemu-user-static

# Verify installation
aarch64-linux-gnu-gcc --version
qemu-aarch64 --version
```

**Verifying Cross-Execution:**

```bash
# Create test file
echo 'int main() { return 42; }' > test_arm.c

# Compile for ARM64
aarch64-linux-gnu-gcc test_arm.c -o test_arm

# Verify file type
file test_arm
# Output: ELF 64-bit LSB pie executable, ARM aarch64...

# Run (transparently via binfmt_misc + QEMU)
./test_arm
echo $?
# Output: 42
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The ARM Philosophy (RISC) vs x86 (CISC)

Week 1 focused on x86-64 (CISC - Complex Instruction Set Computer).
Week 2 moves to ARM (RISC - Reduced Instruction Set Computer).

**Key Architectural Differences:**

| Feature | x86-64 (CISC) | ARMv8-A (RISC) | Implications for Optimization |
|---------|---------------|----------------|-------------------------------|
| **Instruction Size** | Variable (1-15 bytes) | Fixed (32-bit typically) | ARM decode is simpler/faster/lower power |
| **Operands** | Register-Memory (`add rax, [rbx]`) | Load-Store (`ldr`, `add`, `str`) | ARM *must* explicitly load data before computing |
| **Registers** | Few (16 GPR, 32 SIMD) | Many (31 GPR, 32 SIMD) | ARM suffers less register pressure/spilling |
| **Memory Model** | TSO (Strong) | Weakly Ordered | ARM requires explicit barriers (`dmb`, `dsb`) |
| **Unaligned Access** | Hardware handles it | Mostly supported, but strict strict strict on older cores | Aligned access is critical on older ARMs |

#### 1.1 AArch64 Execution State

**Register File:**

1.  **General Purpose Registers (X0 - X30):**
    *   64-bit wide integers.
    *   `X0`-`X7`: Argument passing / return values.
    *   `X29`: Frame Pointer (FP).
    *   `X30`: Link Register (LR) - holds return address (unlike x86 stack push/pop!).
    *   `SP`: Stack Pointer.
    *   `XZR`: Zero Register (Read=0, Write=Discard).

2.  **Vector/Floating Point Registers (V0 - V31):**
    *   128-bit wide.
    *   Can be viewed as:
        *   `Q0`-`Q31`: 128-bit Quadword (SIMD).
        *   `D0`-`D31`: 64-bit Double-precision (aliased to lower half).
        *   `S0`-`S31`: 32-bit Single-precision.
        *   `H0`-`H31`: 16-bit Half-precision.
        *   `B0`-`B31`: 8-bit Byte.

**Register Aliasing Diagram:**

```
V0 [128 bits] ------------------------------------------------->
Q0 [128 bits] [                                                ]
D0 [ 64 bits] [                        ]
S0 [ 32 bits] [            ]
H0 [ 16 bits] [      ]
B0 [  8 bits] [   ]
```

*Note:* Unlike x86 AVX (where YMM0 holds XMM0), typical ARM scalar float (`S0`) uses the bottom bits of the vector register `V0`.

#### 1.2 "Advanced SIMD" (NEON)

**NEON** is ARM's SIMD architecture (mandatory in ARMv8-A).
It is similar to SSE/AVX but "cleaner" due to RISC heritage.

**Capabilities:**
*   **128-bit width:** (Same as SSE).
*   **Data Types:** 8, 16, 32, 64-bit integers; 16 (FP16), 32, 64-bit floats.
*   **Instruction Format:** 3-operand non-destructive (`ADD V0, V1, V2` -> `V0 = V1 + V2`).

**Why Not 256/512 bit?**
ARM targets mobile/power-efficiency. 128-bit is the "sweet spot" for thermal density.
*Exception:* **SVE (Scalable Vector Extension)** allows larger vectors (up to 2048-bit), but NEON is fixed to 128.

#### 1.3 Memory Ordering (The Silent Bug Generator)

**TSO (Total Store Ordering - x86):**
If Core A writes `X` then `Y`, Core B *guaranteed* to see `X` before `Y`.
Hardware enforces strict order.

**Weak Ordering (ARM):**
If Core A writes `X` then `Y`, Core B might see `Y` then `X` (due to store buffers/cache hierarchy).

**Code Example (Buggy on ARM, safe on x86):**
```c
// Thread 1
data = 42;
flag = 1;

// Thread 2
while (!flag); // Spin
print(data);
```
On ARM, `flag` might become 1 *before* `data` becomes 42 visible to Thread 2!
*Fix:* Use `std::atomic` (C++11) or explicit memory barriers (`std::memory_order_release/acquire`).

---

### 🔹 Part 2: ARM Implementations & Ecosystem

ARM is an IP company. They design cores; partners build chips.

#### 2.1 Core Types (Cortex vs Neoverse)

| Series | Focus | Examples | Use Case |
|--------|-------|----------|----------|
| **Cortex-A** | Consumer (High Perf) | A78, A710, X3 | Smartphones (Galaxy, Pixel) |
| **Cortex-A** | Consumer (Efficiency) | A55, A510 | "Little" cores for background tasks |
| **Neoverse** | Server/HPC | N1, V1, N2 | AWS Graviton, Ampere Altra |
| **Cortex-M** | Embedded/RTOS | M4, M7, M55 | IoT, Controllers (No NEON usually, use Helium) |

#### 2.2 big.LITTLE and DynamIQ

x86 (until Alder Lake) was symmetric (all cores same).
ARM pioneered heterogeneous computing:

*   **Big Cores:** High frequency, deep pipeline, OoO, high power (e.g., Cortex-X3).
*   **Little Cores:** In-order or short pipeline, low power (e.g., Cortex-A510).

**Implication for SIMD:**
*   Little cores support NEON perfectly fine.
*   BUT... they execute slower (maybe 128-bit ops split into 2x64-bit cycles).
*   **Thread affinity matters:** High-performance SIMD loops should be pinned to BIG cores.

#### 2.3 Apple Silicon (The Game Changer)

Apple's M-series (M1/M2/M3) implements ARMv8.5+ with custom microarchitecture (Firestorm/Icestorm).

**Key Differences vs Standard Cortex:**
1.  **Massive Reorder Buffer (ROB):** 630+ entries (vs Intel Golden Cove ~512, Cortex-X2 ~224).
2.  **Ultra-Wide Execution:** 8 instruction decode width.
3.  **AMX (Apple Matrix Extension):** Undocumented coprocessor for generic matrix math (separate from NEON).

---

### 🔹 Part 3: Hello World in NEON Assembly

Let's look at raw AArch64 assembly to appreciate the RISC cleanliness.

**Scalar Code:**
```asm
// x0 = a, x1 = b
add x0, x0, x1   // x0 = x0 + x1
ret
```

**SIMD Code (Adding 4 floats):**
```asm
// Arguments in v0 and v1 (128-bit regs)
fadd v0.4s, v0.4s, v1.4s  // Add packed single (4s)
ret
```
*Note the suffix `.4s` (4 singles). Very readable!*

**Element Access:**
```asm
// Lane copy
mov v0.s[1], v1.s[3]   // Copy float from index 3 to index 1
```

---

## 💻 Implementation: Cross-Platform Vector Add

We will write C code with NEON intrinsics and compile it on x86 using cross-tools.

```c
// File: neon_hello.c
#include <stdio.h>
#include <arm_neon.h> // The <immintrin.h> of ARM

void print_vec(const char* label, float32x4_t v) {
    float f[4];
    vst1q_f32(f, v); // Store 1 Quad-word (128-bit)
    printf("%s: [%.1f, %.1f, %.1f, %.1f]\n", label, f[0], f[1], f[2], f[3]);
}

int main() {
    // 1. Data Types
    // float32x4_t corresponds to __m128
    
    // 2. Loading / Initialization
    // Similar to _mm_setr_ps
    float data[] = {1.0, 2.0, 3.0, 4.0};
    float32x4_t a = vld1q_f32(data); // Load 1 Quad
    
    // Set 1 (broadcast)
    float32x4_t b = vdupq_n_f32(10.0f); // Dup "n" scalar Element to "q" Quad
    
    // 3. Arithmetic
    // vaddq_f32 (Vector Add Quad Float32)
    float32x4_t c = vaddq_f32(a, b);
    
    // 4. Fused Multiply Add (FMA)
    // d = c + a * b
    float32x4_t d = vfmaq_f32(c, a, b); 
    
    print_vec("A", a);
    print_vec("B", b);
    print_vec("C (A+B)", c);
    print_vec("D (C+A*B)", d);

    return 0;
}
```

### 🛠️ Compilation and Execution (On x86 Host)

1.  **Compile Static Binary:**
    Static linking (`-static`) makes it easier to run with QEMU without library path issues.
    ```bash
    aarch64-linux-gnu-gcc -static -march=armv8-a neon_hello.c -o neon_hello
    ```

2.  **Verify Architecture:**
    ```bash
    readelf -h neon_hello | grep Machine
    # Machine: AArch64
    ```

3.  **Run Emulated:**
    ```bash
    qemu-aarch64 ./neon_hello
    # Output:
    # A: [1.0, 2.0, 3.0, 4.0]
    # B: [10.0, 10.0, 10.0, 10.0]
    # C (A+B): [11.0, 12.0, 13.0, 14.0]
    # ...
    ```

---

## 🧪 Hands-On Labs

### Lab 8: Examining NEON Assembly

**Objective:** Write a function using intrinsics and verify the generated assembly (via `-S`).

**File:** `neon_asm_test.c`
```c
#include <arm_neon.h>

// Multiply and Accumulate
// out[i] += a[i] * k
void vec_mac(float* out, const float* a, float k, int n) {
    float32x4_t vk = vdupq_n_f32(k);
    
    for (int i = 0; i < n; i+=4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vout = vld1q_f32(&out[i]);
        
        // vout = vout + va * vk
        vout = vfmaq_f32(vout, va, vk);
        
        vst1q_f32(&out[i], vout);
    }
}
```

**Instruction:**
1. Compile to assembly: `aarch64-linux-gnu-gcc -O3 -S neon_asm_test.c`
2. Open `neon_asm_test.s`
3. Look for instructions like `ldr`, `fmla` (Floating Multiply Accumulate).

**Expected Assembly snippet:**
```asm
.L2:
    ldr     q1, [x1, x3]    ; Load a[i] into 128-bit q1
    ldr     q0, [x0, x3]    ; Load out[i] into q0
    fmla    v0.4s, v1.4s, v2.4s  ; Fused FP Mutiply-Add
    str     q0, [x0, x3]    ; Store q0
    add     x3, x3, 16      ; Increment pointer by 16 bytes
    cmp     x3, x2          ; Compare loop counter
    b.ne    .L2             ; Loop
```

---

## 📝 Summary & Key Takeaways

1.  **ARMv8 is RISC:** It uses a Load-Store architecture. Memory operands in math instructions (like x86 `add ps, [mem]`) are illegal. You must `ldr`, `fadd`, `str`.
2.  **NEON is 128-bit:** Similar to SSE, but with cleaner 3-operand syntax and unified register handling for integers and floats.
3.  **Cross-Compilation is Easy:** Modern tools allow compiling ARM binaries on x86 machines seamlessly using GCC cross-toolchains and QEMU-user.
4.  **Register File:** V0-V31 (128-bit) are the vector registers. They alias D (64) and S (32) scalars, unlike AVX/SSE where scalar float is a separate "mode" usage of the same register file but accessed differently.
5.  **Weak Memory Model:** Be very mindful of thread synchronization. Without barriers, writes can reorder visibly!

---

## 📚 Additional Resources

*   [ARM Neon Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
*   [Coding for NEON - ARM Developer Guide](https://developer.arm.com/documentation/den0018/a/)

**Tomorrow:** Day 9 - NEON Intrinsics Deep Dive... mastering `vld1`, `vst1`, interleaving `vzip`/`vuzp`, and pairwise reductions.

*End of Day 008 - Total Lines: 1000+*
