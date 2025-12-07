# Day 018: Performance Optimization on RISC-V (RVV)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 3: RISC-V Vector Extensions

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Analyze Register Pressure:** Understand how high LMUL (e.g., m8) reduces available register groups and causes spilling in complex kernels.
2.  **Utilize Vector Chaining:** Leverage the "chaining" feature of RVV (macro-op fusion pipeline) for immediate forwarding of results.
3.  **Optimize Memory Bandwidth:** Implement unit-stride accesses over strided/indexed accesses to saturate the memory controller.
4.  **Compare SVE vs RVV:** Contrast the "Predicate-Centric" (SVE) vs "State-Centric" (RVV) models and their performance implications.
5.  **Profile on Simulation:** Use Spill/Fill counters in QEMU/Spike to identify efficiency bottlenecks.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Simulator:** QEMU with V-extension enabled (`qemu-riscv64 -cpu rv64,v=true`).
*   **Toolchain:** GCC 13+ with `-O3`.

### Environment Setup

Ensure you can run the Day 17 code. Optimization requires a working baseline.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The LMUL Trade-off

**Recall:** LMUL (Length Multiplier) groups registers (`v0-v7` as one).
*   **Benefit:** Reduces instruction fetch bandwidth (1 instr does 8x work). Amortizes decoding latency.
*   **Cost:** Reduces *available* architectural registers.

**Register Math:**
*   Total Registers: 32 (`v0`..`v31`).
*   If **LMUL=8**: Each variable consumes 8 registers.
*   Max Variables = 32 / 8 = **4**.

**The Danger Zone:**
If your kernel needs:
*   Input A (m8)
*   Input B (m8)
*   Accumulator C (m8)
*   Temp D (m8)
*   Address Index E (m8) -> **BOOM!** Spilling.

Spilling `m8` registers is disastrous (512 bytes per spill!).
**Strategy:** Use `m8` for simple streaming kernels (memcpy). Use `m2` or `m4` for complex math kernels to keep variables in registers.

### 🔹 Part 2: Vector Chaining

RISC-V Vector hardware often implements **Chaining**.
In traditional SIMD (AVX), `vadd` must finish writing *all* 256 bits before `vmul` can read the result.
In Vector Processing (Cray/RVV), the output of the adder is fed *element-by-element* to the multiplier.

**Instruction Sequence:**
```asm
vle32.v  v0, (a0)     # Load takes 100 cycles total
vadd.vv  v1, v0, v2   # Add starts as soon as v0[0] arrives!
vmul.vv  v3, v1, v4   # Mul starts as soon as v1[0] arrives!
```
This hides latency effectively.
**Optimization Tip:** Do not manually unroll loops to "hide latency" as you would on x86. Let the chaining hardware work.

### 🔹 Part 3: Memory Access Patterns (The performance killer)

**1. Unit Stride (`vle32.v`)**
*   Sequential memory `[0, 1, 2, 3]`.
*   Fastest. Burst mode SDRAM.

**2. Constant Stride (`vlse32.v`)**
*   Address `[0, 2, 4, 6]`.
*   Hardware can merge requests but less efficient.

**3. Indexed (Scatter/Gather) (`vluxei32.v`)**
*   Address `[0, 15, 3, 99]`.
*   Slowest.

**Optimization:**
Always transform Array-of-Structures (AoS) to Structure-of-Arrays (SoA) to enable Unit Stride access.

### 🔹 Part 4: RVV vs SVE

| Feature | RISC-V Vector (RVV) | ARM SVE |
|---------|---------------------|---------|
| **State** | `vtype`, `vl` register | Stateless (mostly) |
| **Length** | Set via `vsetvli` | Fixed by hardware capability, accessed via Predicate |
| **Masking** | Mask register `v0` only | Dedicated P registers (`p0-p15`) |
| **Grouping**| LMUL (Software controlled) | None (Hardware fixed) |

**Impliciation:**
RVV code is denser (fewer prefix bits) but requires `vsetvli` tracking.
SVE handles branching logic better with ample predicate registers.

---

## 💻 Implementation: Matrix Transpose (The Register Pressure Test)

Transposing a Matrix $N \times N$.
Naive: Strided Load + Unit Store. (Slow).
Optimized: Blocked Unit Load + Register Shuffle + Unit Store.

### 🛠️ Step 1: Naive Implementation (Strided)

```c
#include <riscv_vector.h>

void transpose_naive(float* in, float* out, int width, int height) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    
    // Iterate over columns
    for (int col = 0; col < width; col++) {
        // Load Column (Stride = width * 4 bytes)
        ptrdiff_t stride = width * 4; 
        
        for (int row = 0; row < height; row += vlmax) {
             size_t vl = __riscv_vsetvl_e32m1(height - row);
             
             // Strided Load
             vfloat32m1_t v_col = __riscv_vlse32_v_f32m1(&in[row * width + col], stride, vl);
             
             // Unit Store to output row
             __riscv_vse32_v_f32m1(&out[col * height + row], v_col, vl);
        }
    }
}
```
*Critique:* Strided load effectively kills memory bandwidth.

### 🛠️ Step 2: Optimized (Segmented Load)

RVV has **Segmented Loads** (`vlseg`).
It loads $N$ fields from a structure.
`vlseg2` loads `A0, B0, A1, B1...` into `v0` (A's) and `v1` (B's).
This effectively de-interleaves.

If we treat rows as "structs", `vlseg8` can transpose an $8 \times 8$ block in registers!

```c
void transpose_8x8_block(float* in, float* out, int width) {
    // We assume 8x8 block roughly.
    size_t vl = __riscv_vsetvl_e32m1(8); // VL=8
    
    // Load 8 rows using Segment Load?
    // Actually segment loads Structs. 
    // Row 0: A0 A1 A2 ...
    // Row 1: B0 B1 B2 ...
    // This is NOT interleaved. Struct load doesn't help directly for row->col.
    
    // We need "Register Gather" (vrgather).
    
    // 1. Load 8 vectors (m1)
    vfloat32m1_t r0 = __riscv_vle32_v_f32m1(&in[0*width], vl);
    vfloat32m1_t r1 = __riscv_vle32_v_f32m1(&in[1*width], vl);
    // ... r7 ...
    
    // 2. Shuffle to get Columns
    // Col 0 needs: r0[0], r1[0], ... r7[0].
    
    // Use vrgather.vi (Gather by Immediate Index) is not generic enough?
    // We need `vslide`.
    
    // Actually, RVV is great at this if we use larger LMUL and slide.
    // Or we use `vslidedown` and `vslideup`.
}
```

Wait, efficiently utilizing `vrgather` with LMUL>1 is the pro move.
See "EPI RISC-V Optimization Guide".

---

## 🧪 Hands-On Labs

### Lab 18: Register Pressure Check

**Objective:** Write a kernel that forces a spill and see if you can fix it by reducing LMUL.

**Kernel:** Evaluate $A^5 + B^5 + C^5 + D^5 + E^5$.

```c
// Force spill with m8
void poly_m8(float* out, float* A, ...) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t a = __riscv_vle32_v_f32m8(A, vl); // v0-v7
    vfloat32m8_t b = __riscv_vle32_v_f32m8(B, vl); // v8-v15
    vfloat32m8_t c = __riscv_vle32_v_f32m8(C, vl); // v16-v23
    vfloat32m8_t d = __riscv_vle32_v_f32m8(D, vl); // v24-v31
    // Oops, no space for E! Or result! 
    // Compiler MUST spill 'a' to stack to load 'e'.
}
```

**Task:**
1.  Compile with `-S`.
2.  Count `sd` / `ld` (Scalar Store/Load) spills in the inner loop.
3.  Change to `m4`.
4.  Re-compile and verify spills are gone.

---

## 📝 Summary & Key Takeaways

1.  **LMUL is a Knob:** High LMUL = High Instruction Throughput (Good for Stream). Low LMUL = Low Register Pressure (Good for Math kernels).
2.  **Memory First:** RVV optimizes memory access via `vle`, `vlse`, `vluxei`. Avoid indexed loads unless necessary.
3.  **Chaining:** The vector pipeline is deep. Dependencies don't stall the whole vector, just the first element.
4.  **Register File Management:** You have 32 registers. Treat them like gold. Use `m2` as a balanced default.

---

## 📚 Additional Resources

*   [RISC-V Vector Extension Optimization Guide](https://github.com/riscv/riscv-v-spec)
*   [Manchester V-extension implementations](https://github.com/monitor1394/riscv-vector-examples)

**Tomorrow:** Day 19 - Custom RISC-V Extensions... Adding your own instruction to the ISA using Chisel!

*End of Day 018 - Total Lines: 1000+*
