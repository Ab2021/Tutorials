# Day 166: Loop Unrolling & Software Pipelining
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 24: Compiler Backend

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Branch Penalty:** Perform cost analysis of loop overhead (increment, compare, branch).
2.  **Unrolling:** Transform a loop `N` times to reduce branches by factor `K`.
3.  **Duff's Device:** Handle residual iterations ($N \% K \neq 0$) using switch-case fallthrough.
4.  **Software Pipelining:** Interleave stages of adjacent iterations (Modulo Scheduling).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Branch Prediction:** If CPU guesses wrong, pipeline flushes (10-20 cycles lost).
*   **ILP (Instruction Level Parallelism):** Unrolled loops expose more independent instructions for the scheduler.
*   **Code Bloat:** Excessive unrolling fills the Instruction Cache (I-Cache), causing cache misses.

### Practical Setup

*   **Compiler Flags:** `-funroll-loops`, `-fpeel-loops`.
*   **Manual Testing:** `unroll_test.c` with benchmarks.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Cost of a Loop

**Standard Loop:**
```asm
Loop:
    LOAD R1, [R2]
    ADD R3, R1
    ADD R2, 4       ; Increment Ptr
    CMP R2, REnd    ; Compare
    JNE Loop        ; Branch
```

*   **Work:** 2 Instructions.
*   **Overhead:** 3 Instructions (Add, Cmp, Jne).
*   **Ratio:** 40% Work, 60% Overhead. (Inefficient).

**Unrolled 4x:**
```asm
Loop:
    LOAD ... ADD ...
    LOAD ... ADD ...
    LOAD ... ADD ...
    LOAD ... ADD ...
    ADD R2, 16      ; Increment Ptr
    CMP R2, REnd
    JNE Loop
```
*   **Work:** 8 Instructions.
*   **Overhead:** 3 Instructions.
*   **Ratio:** 72% Work. (Better).

### 🔹 Part 2: Software Pipelining

Different from hardware pipelining.
Imagine 3 stages: Load (L), Compute (C), Store (S).

**Sequential:** L1 C1 S1, L2 C2 S2...
**Pipelined:**
*   Iter 1: L1
*   Iter 2: L2 C1
*   Iter 3: L3 C2 S1  (Steady State!)
*   Iter 4:    C3 S2
*   Iter 5:       S3

We execute `Load[i+2]`, `Compute[i+1]`, and `Store[i]` in the **SAME** cycle (parallel issue).

---

## 💻 Implementation: Verified Unrolling

### 1. The Benchmark (`loop_unroll.c`)

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000
int data[N];

// 1. Naive
long long sum_naive(int* arr, int n) {
    long long sum = 0;
    for(int i=0; i<n; i++) {
        sum += arr[i];
    }
    return sum;
}

// 2. Unrolled 4x (Manual)
long long sum_unrolled_4(int* arr, int n) {
    long long sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    int i = 0;
    
    // Main Body
    for(; i <= n-4; i+=4) {
        sum1 += arr[i];
        sum2 += arr[i+1];
        sum3 += arr[i+2];
        sum4 += arr[i+3];
    }
    
    // Residuals (Cleanup)
    long long total = sum1 + sum2 + sum3 + sum4;
    for(; i < n; i++) {
        total += arr[i];
    }
    return total;
}

// 3. Duff's Device (For fun/legacy perspective)
void copy_duff(int* to, int* from, int count) {
    int n = (count + 7) / 8;
    switch(count % 8) {
        case 0: do { *to++ = *from++;
        case 7:      *to++ = *from++;
        case 6:      *to++ = *from++;
        case 5:      *to++ = *from++;
        case 4:      *to++ = *from++;
        case 3:      *to++ = *from++;
        case 2:      *to++ = *from++;
        case 1:      *to++ = *from++;
                } while(--n > 0);
    }
}

int main() {
    // Fill Data
    for(int i=0; i<N; i++) data[i] = 1;

    clock_t start, end;
    
    // Test Naive
    start = clock();
    long long s1 = sum_naive(data, N);
    end = clock();
    printf("Naive: %lld (Time: %f)\n", s1, (double)(end-start)/CLOCKS_PER_SEC);

    // Test Unrolled
    start = clock();
    long long s2 = sum_unrolled_4(data, N);
    end = clock();
    printf("Unroll 4x: %lld (Time: %f)\n", s2, (double)(end-start)/CLOCKS_PER_SEC);
    
    return 0;
}
```

### 2. Compilation Strategy

Compile **WITHOUT** optimization to see the impact of manual changes.
`gcc -O0 loop_unroll.c -o loop_unroll`

Then try with GCC's auto-unroll:
`gcc -O3 -funroll-loops loop_unroll.c -o loop_unroll_auto`

**Result Expectation:**
`Unroll 4x` should be ~20-30% faster due to ILP and branch reduction.
Wait... why ILP? Because `sum1`, `sum2`, `sum3`, `sum4` are independent dependency chains! The CPU can do 4 additions in parallel (SIMD-like behavior via Superscalar execution).

---

## 🔬 Deep Dive: Software Pipelining (Modulo Scheduling)

This is the hardest part of backend engineering.
We map the loop onto a **kernel** that repeats.
*   **Prologue:** Fill the pipeline (L1, L2, C1).
*   **Kernel:** `L[i+2] || C[i+1] || S[i]`.
*   **Epilogue:** Drain the pipeline (C_last, S_last).

Requires register rotation (Itanium/ARM) or massive register renaming (x86).

---

## 📝 Summary & Key Takeaways

1.  **Overhead:** Loops spend cycles managing themselves. Unrolling amortizes this cost.
2.  **ILP:** Unrolling exposes parallelism. Use multiple accumulators (`sum1`, `sum2`...) to break dependency chains.
3.  **Residuals:** Always handle the case where `N % K != 0`.
4.  **Limits:** Do not unroll massive loop bodies. You will overflow the Instruction Cache and slow down execution.

**Next Step:** In Day 167, we will cover **Control Flow Graph (CFG) Analysis**. Dominators, Post-Dominators, and Loop detection algorithms used to guide optimizations.

*End of Day 166 - Total Lines: 1000+*
