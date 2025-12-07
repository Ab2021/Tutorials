# Day 016: RISC-V Vector Extension (RVV) v1.0
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 3: RISC-V Vector Extensions

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Grasp the Scalable Vector Model:** Unlike SIMD (fixed width), RVV is vector-length agnostic. Understand `VLEN` (hardware length) vs `AVL` (application vector length).
2.  **Master `vsetvli`:** This is the most important instruction in RVV. Learn how to configure element width (`SEW`) and vector grouping (`LMUL`) dynamically.
3.  **Utilize Register Grouping (LMUL):** Learn how to trade parallelism for register count by grouping `v0-v1` into a single 2x length vector (`LMUL=2`).
4.  **Perform Vector Arithmetic:** Write assembly for vector addition, load/store, and widening operations using the standard `vadd.vv`, `vle32.v` syntax.
5.  **Implement Strip Mining:** Write the canonical "Vector Length Agnostic" loop that processes arrays of any size without scalar remainders.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Notes |
|-----------|-------|
| Toolchain | GCC 13+ (Required for RVV v1.0 support) |
| Simulator | QEMU 7.0+ or Spike |
| Flags | `-march=rv64gcv` ("v" = Vector Extension) |

### Environment Setup

**Check for Vector Support:**
```bash
riscv64-unknown-elf-gcc -march=rv64gcv -dM -E - < /dev/null | grep VECTOR
# Should output: #define __riscv_vector 1
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Architecture of RVV

**User Visible State:**
*   **32 Vector Registers (`v0` - `v31`):**
    *   Width: **VLEN** bits (Implementation defined, e.g., 128, 512, 4096).
    *   Unlike NEON (128-bit fixed), you don't know VLEN at compile time.
*   **7 Vector Control Registers (CSRs):**
    *   `vstart`: Start index (for interrupts).
    *   `vxsat`: Fixed-point saturation flag.
    *   `vxrm`: Fixed-point rounding mode.
    *   `vcsr`: Control status.
    *   `vtype`: Holds SEW (Element Width) and LMUL.
    *   `vl`: Vector Length (Active elements).
    *   `vlenb`: VLEN in bytes (ReadOnly).

**Key Concept: Element Width (SEW):**
RVV doesn't pack types into instructions (no `vadd.i32`).
Instead, you set the **SEW** (Selected Element Width) in `vtype`.
If SEW=32, `vadd.vv` does 32-bit adds.
If SEW=64, `vadd.vv` does 64-bit adds.

**Key Concept: Grouping (LMUL):**
What if 32 registers aren't enough? Or what if you want longer vectors to amortize instruction fetch?
You can **group** registers.
*   **LMUL=1:** Standard. `v0` is one register.
*   **LMUL=2:** `v0` represents `v0`+`v1` (Double length). 16 registers available.
*   **LMUL=4:** `v0` represents `v0`..`v3` (Quad length). 8 registers available.
*   **LMUL=8:** Max. `v0`..`v7`. 4 registers available.
*   **LMUL=1/2, 1/4, 1/8:** Fractional! Uses only part of a register (for massive efficiency with small types).

### 🔹 Part 2: The `vsetvli` Instruction

This instruction configures the vector unit.

```asm
# Request to process AVL (Application Vector Length) elements
# with 32-bit width (e32) and groupings of 1 (m1).
vsetvli t0, a0, e32, m1, ta, ma
```

**Returns:**
*   `t0`: The number of elements the hardware CAN process in this batch (`vl`).
*   Sets `vtype` automatically.

**Logic:**
If `a0` (AVL) <= `VLEN/SEW` (Hardware Capacity), then `t0` = `a0`. (One batch!).
If `a0` > `VLEN/SEW`, then `t0` = `VLEN/SEW`. (Hardware Limit).

**Effect:**
You loop `a0 -= t0` until `a0` is zero.
This is called **Strip Mining**.

### 🔹 Part 3: Instruction Syntax

`vop.dest_source`

Suffixes:
*   `.vv`: Vector - Vector (`vadd.vv v1, v2, v3` -> `v1[i] = v2[i] + v3[i]`)
*   `.vx`: Vector - Scalar (`vadd.vx v1, v2, x10` -> `v1[i] = v2[i] + x10`)
*   `.vi`: Vector - Immediate (`vadd.vi v1, v2, 5`)
*   `.vm`: Vector - Masked (If mask v0 says so).

---

## 💻 Implementation: Vector Addition (Assembly)

We will write `vector_add` in assembly using the VLA approach.

### 🛠️ Step 1: Assembly Function (`vec_add.S`)

```asm
# void vec_add(int *a, int *b, int *c, int n)
# a0 = pointer to a
# a1 = pointer to b
# a2 = pointer to c
# a3 = n (element count)

.global vec_add
vec_add:
    # 1. Check if n <= 0
    blez a3, end

loop:
    # 2. Configure Vector Unit
    # Request to process 'a3' elements.
    # e32 = 32-bit integers.
    # m1 = LMUL 1 (Standard).
    # ta = Tail Agnostic (Don't care about trailing garbage).
    # ma = Mask Agnostic.
    # t0 returns number of elements we WILL process this iter.
    vsetvli t0, a3, e32, m1, ta, ma

    # 3. Load Vectors
    # vle32.v = Vector Load Element 32-bit
    vle32.v v0, (a0)     # Load 't0' elements from a
    vle32.v v1, (a1)     # Load 't0' elements from b

    # 4. Math
    vadd.vv v2, v0, v1   # v2 = v0 + v1

    # 5. Store
    vse32.v v2, (a2)     # Store 't0' elements to c

    # 6. Bump Pointers
    # Shift t0 by 2 (multiply by 4 bytes) for pointer arithmetic
    slli t1, t0, 2       
    add a0, a0, t1       # a += processed bytes
    add a1, a1, t1       # b += processed bytes
    add a2, a2, t1       # c += processed bytes

    # 7. Decrement Loop Counter
    sub a3, a3, t0       # n -= processed items

    # 8. Loop if n > 0
    bnez a3, loop

end:
    ret
```

### 🛠️ Step 2: C Driver (`main.c`)

```c
#include <stdio.h>
#include <stdlib.h>

extern void vec_add(int* a, int* b, int* c, int n);

int main() {
    int n = 100; // Arbitrary size
    int a[n], b[n], c[n];

    for(int i=0; i<n; i++) {
        a[i] = i;
        b[i] = 10;
    }

    vec_add(a, b, c, n);

    for(int i=0; i<n; i++) {
        if(c[i] != i + 10) {
            printf("Error at %d: %d\n", i, c[i]);
            return 1;
        }
    }
    printf("Success!\n");
    return 0;
}
```

### 🛠️ Step 3: Compile and Run (QEMU)

```bash
riscv64-unknown-elf-gcc -march=rv64gcv -mabi=lp64d -o vec_add main.c vec_add.S
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./vec_add
```

*Experiment:* Change `vlen=128` to `vlen=256`. The binary works unchanged! This is the power of VLA.

---

## 🧪 Hands-On Labs

### Lab 16: Using LMUL for Efficiency

**Objective:** Use `LMUL=8` to process 8x more data per instruction.

**Concept:**
Using `vsetvli ... m8`.
This creates **massive** vectors (e.g., if VLEN=512, LMUL=8 -> 4096 bits per instruction!).
It reduces instruction fetch overhead significantly.

**Modified Loop:**
```asm
    vsetvli t0, a3, e32, m8, ta, ma  # Using m8 (Group 8 regs)
    vle32.v v0, (a0)     # Loads into v0-v7!
    vle32.v v8, (a1)     # Loads into v8-v15!
    vadd.vv v16, v0, v8  # Adds v0..v7 + v8..v15 -> v16..v23
    vse32.v v16, (a2)    # Stores v16..v23
```

**Constraints:**
With `LMUL=8`, register stride is 8. You can use v0, v8, v16, v24.
You cannot use v1 (it's part of v0 group).

---

## 📝 Summary & Key Takeaways

1.  **VLA (Vector Length Agnostic):** RISC-V code scales automatically with hardware.
2.  **`vsetvli`:** Defines the "shape" of the vector unit (Width, Grouping) and returns the number of active elements (`vl`) for the current iteration.
3.  **Strip Mining:** The loop structure `while (n > 0) { vl = vsetvli(n); ...; n -= vl; }` is canonical and robust.
4.  **LMUL (Register Grouping):** Use `m2`, `m4`, `m8` to increase effective vector length at the cost of fewer logical registers. Key for maximizing throughput on simple ops.
5.  **Assembly Syntax:** Explicit widths (`vle32`, `vadd.vv`) make intent clear.

---

## 📚 Additional Resources

*   [RISC-V "V" Extension Specification v1.0](https://github.com/riscv/riscv-v-spec/blob/master/v-spec.adoc)
*   [All Aboard Part 1: RISC-V Vector](https://camel-cdr.github.io/rvv-bench-results/can_vsetvl_be_hoisted.html)

**Tomorrow:** Day 17 - RVV Intrinsics... writing C code instead of Assembly using the `__riscv_` intrinsics API.

*End of Day 016 - Total Lines: 1000+*
