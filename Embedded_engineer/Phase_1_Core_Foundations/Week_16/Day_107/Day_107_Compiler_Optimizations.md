# Day 107: Compiler Optimizations
## Phase 1: Core Embedded Engineering Foundations | Week 16: Advanced C & Optimization

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
1.  **Compare** GCC optimization levels (`-O0`, `-O1`, `-O2`, `-O3`, `-Os`) and their impact on size and speed.
2.  **Analyze** generated assembly code to verify optimizations (Loop Unrolling, Inlining).
3.  **Apply** `volatile` correctly to prevent dangerous optimizations on memory-mapped I/O.
4.  **Implement** `static inline` functions as a type-safe alternative to macros.
5.  **Enable** Link Time Optimization (LTO) for cross-module optimization.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   [Godbolt Compiler Explorer](https://godbolt.org/) (Optional, useful for visualization).
*   **Prior Knowledge:**
    *   Day 9 (Assembly Basics)
    *   Day 2 (Volatile)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Optimization Levels
*   **-O0 (None):** Default. Fast compile, slow code. 1:1 mapping between C and ASM. Best for debugging.
*   **-O1 (Optimize):** Basic logic simplification.
*   **-O2 (More):** Instruction scheduling, inlining. Standard for Release builds.
*   **-O3 (Aggressive):** Loop unrolling, vectorization. Can increase code size (cache pressure).
*   **-Os (Size):** Optimize for space. Critical for small Flash MCUs.

### 🔹 Part 2: What Compiler Does
1.  **Dead Code Elimination:** Removes code that never runs.
2.  **Constant Folding:** `x = 3 + 4` -> `x = 7`.
3.  **Loop Unrolling:** `for(i=0;i<4;i++) f();` -> `f(); f(); f(); f();`. Avoids loop overhead (compare/branch).
4.  **Function Inlining:** Replaces function call with function body. Saves stack push/pop.

### 🔹 Part 3: The `volatile` Trap
Compilers assume memory doesn't change unless *they* change it.
*   **Code:** `while(flag == 0);`
*   **Optimized:** `if(flag==0) while(1);` (Infinite loop if flag is 0 initially).
*   **Fix:** `volatile int flag;` tells compiler "This can change externally (ISR)".

---

## 💻 Implementation: Analyzing Assembly

> **Instruction:** Compile a loop with different levels and inspect ASM.

### 👨‍💻 Code Implementation

#### Step 1: C Code (`opt_test.c`)
```c
int Sum_Array(int *arr, int len) {
    int sum = 0;
    for(int i=0; i<len; i++) {
        sum += arr[i];
    }
    return sum;
}
```

#### Step 2: Compile & Disassemble
Terminal:
```bash
arm-none-eabi-gcc -c -O0 -mcpu=cortex-m4 opt_test.c -o opt_O0.o
arm-none-eabi-objdump -d opt_O0.o > opt_O0.asm

arm-none-eabi-gcc -c -O3 -mcpu=cortex-m4 opt_test.c -o opt_O3.o
arm-none-eabi-objdump -d opt_O3.o > opt_O3.asm
```

#### Step 3: Compare
*   **O0:** Loads `i` from stack, compares, adds, stores `i` back to stack. Lots of `LDR`/`STR`.
*   **O3:** Keeps `sum` and `i` in Registers (`R0`, `R1`). Uses `LDR.W` with post-increment. May unroll loop (process 4 items per iteration).

---

## 💻 Implementation: Inline Functions

> **Instruction:** Replace a macro with a static inline function.

### 👨‍💻 Code Implementation

#### Step 1: The Macro (Bad)
```c
#define MAX(a,b) ((a) > (b) ? (a) : (b))
// Issue: MAX(x++, y++) increments x twice!
```

#### Step 2: The Inline (Good)
```c
static inline int Max_Int(int a, int b) {
    return (a > b) ? a : b;
}
// Usage: Max_Int(x++, y++) works correctly.
```

#### Step 3: Force Inline
GCC keyword: `__attribute__((always_inline))`
```c
__attribute__((always_inline)) 
static inline void Critical_Section(void) {
    __disable_irq();
}
```

---

## 🔬 Lab Exercise: Lab 107.1 - The Benchmark

### 1. Lab Objectives
- Measure cycle count of a function.
- Compare -O0 vs -O3.

### 2. Step-by-Step Guide

#### Phase A: DWT Cycle Counter
Enable DWT (Data Watchpoint and Trace) to count CPU cycles.
```c
void DWT_Init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
```

#### Phase B: Test
```c
volatile int data[100]; // Volatile to prevent optimizing away the whole loop

void Benchmark(void) {
    DWT->CYCCNT = 0;
    uint32_t start = DWT->CYCCNT;
    
    Sum_Array((int*)data, 100);
    
    uint32_t end = DWT->CYCCNT;
    printf("Cycles: %lu\n", end - start);
}
```

#### Phase C: Results
1.  Compile -O0. Run. Record Cycles.
2.  Compile -O3. Run. Record Cycles.
3.  **Observation:** -O3 should be 5x-10x faster.

### 3. Verification
If cycles are 0, check if DWT is enabled. Note that `printf` takes thousands of cycles, don't include it in the measurement range.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Loop Unrolling Manually
- **Goal:** Beat the compiler?
- **Task:**
    1.  Write `Sum_Array_Unrolled` manually (process 4 ints per loop).
    2.  Compare with `-O3` auto-unroll.
    3.  Often compiler is better, but sometimes manual helps.

### Lab 3: Link Time Optimization (LTO)
- **Goal:** Cross-file inlining.
- **Task:**
    1.  Add `-flto` to CFLAGS and LDFLAGS in Makefile.
    2.  Define function `foo()` in `a.c`. Call it in `b.c`.
    3.  Without LTO: Call instruction.
    4.  With LTO: `foo()` code inlined into `b.c`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Debugging Optimized Code
*   **Problem:** Stepping in GDB jumps around erratically. Variables say "optimized out".
*   **Solution:** Always debug with `-O0` or `-Og` (Optimize for Debugging). Only switch to `-O2/3` for final release or performance testing.

#### 2. Race Conditions
*   **Problem:** Code works in `-O0` but fails in `-O2`.
*   **Cause:** Usually missing `volatile` on shared variables or timing loops being optimized away.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Premature Optimization:** "The root of all evil". Write clean, readable code first. Only optimize if measurements show a bottleneck.
- **Size vs Speed:** On 128KB Flash, `-Os` is usually preferred over `-O3`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `__packed`?
    *   **A:** Forces struct to have no padding. Saves RAM but causes unaligned access (slower or HardFault).
2.  **Q:** Why does `-O3` sometimes make code slower?
    *   **A:** Code becomes huge (unrolling). Doesn't fit in Instruction Cache (I-Cache). CPU stalls fetching from Flash.

### Challenge Task
> **Task:** "The Duff's Device". Implement a memory copy using Duff's Device (interlaced switch/case loop unrolling). Benchmark it against `memcpy`.

---

## 📚 Further Reading & References
- [GCC Optimization Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)

---
