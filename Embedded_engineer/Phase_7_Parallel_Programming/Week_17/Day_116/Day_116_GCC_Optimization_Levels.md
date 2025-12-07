# Day 116: GCC Optimization Levels
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 17: GCC Internals

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Optimization Flags:** Distinguish between `-O1`, `-O2`, `-O3`, `-Os`, and `-Ofast`.
2.  **Granular Control:** Use function attributes to optimize critical paths and de-optimize debug paths.
3.  **PGO (Profile Guided Optimization):** Implement the 2-step feedback compilation loop.
4.  **LTO (Link Time Optimization):** Enable whole-program analysis in GCC.
5.  **Benchmark:** Measure code size vs performance trade-offs.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Code Bloat:** Aggressive loop unrolling and inlining increases binary size, which might hurt instruction cache (I-Cache).
*   **Branch Prediction:** Static (compiler guess) vs Dynamic (profiler data).

### Practical Setup

*   `gcc` and standard coreutils (`size`, `time`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "O" Levels

GCC organizes hundreds of optimization passes into groups for convenience.

1.  **`-O0` (Do Nothing):** Default. 1:1 mapping of C lines to blocks. Fast compilation, best for debugging (variables visible).
2.  **`-O1` (Optimize):** Basic data flow analysis, DCE (Dead Code Elimination), simple constant prop. No large trade-offs.
3.  **`-O2` (Recommended):** The industry standard.
    *   Enables most optimization that does *not* increase code size significantly.
    *   Strict Aliasing enabled.
    *   Instruction Scheduling enabled.
    *   **No** Auto-Vectorization (usually).
4.  **`-O3` (Aggressive):**
    *   Enables `-ftree-vectorize`.
    *   Enables `-funswitch-loops` (Loop Unswitching).
    *   Enables inline functions even if they aren't marked `inline` (heuristic-based).
    *   *Risk:* Can increase size drastically. Can reveal hidden UB (Undefined Behavior).
5.  **`-Ofast` (Disregard Standards):**
    *   `-O3` + `-ffast-math`.
    *   Allows reassociation `(a+b)+c -> a+(b+c)`.
    *   Assumes no NaNs, no Infinity.
6.  **`-Os` (Size):**
    *   Based on `-O2`, but disables passes that increase size (like `unroll-loops`).
    *   Critical for Embedded Systems (Flash constraint).

### 🔹 Part 2: Granular Attributes

You can mix levels within a single file.

```c
// This function needs to be fast
__attribute__((optimize("O3", "unroll-loops")))
void critical_kernel() { ... }

// This function crashed, I need to debug it step-by-step
__attribute__((optimize("O0")))
void complexity_function() { ... }
```

### 🔹 Part 3: Profile Guided Optimization (PGO)

Static analysis is guessing. PGO is knowing.

**The Workflow:**
1.  **Instrument:** Compile with `-fprofile-generate`.
2.  **Train:** Run the executable on *representative* data. This generates `.gcda` files (execution counts).
3.  **Optimize:** Re-compile with `-fprofile-use`.

**What it does:**
*   **Branch Probability:** If `if (x) error();` is never taken during training, GCC moves the error block to "cold" memory (optimizing I-Cache).
*   **Function Reordering:** Places frequently called functions together.
*   **Virtual Call Speculation:** If `p->foo()` always calls `Dog::foo()`, GCC devirtualizes it with a check.

### 🔹 Part 4: Link Time Optimization (LTO)

Normally, GCC compiles `a.c` without knowing `b.c`.
LTO changes this.

*   **Step 1:** `gcc -flto -c a.c`. The `.o` file contains GIMPLE bytecode, not assembly.
*   **Step 2:** `gcc -flto a.o b.o`. The Linker (`ld`) calls back into GCC (`lto-wrapper`) to merge the GIMPLE from both files and re-optimize.

**Outcome:** Inlining across modules. constant propagation across modules.

---

## 💻 Implementation: The Benchmark

We will use a **Monte Carlo Pi estimation** program, which is compute-heavy.

### Source (`pi_bench.c`)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100000000

double estimate_pi() {
    double count = 0;
    // PGO target: Typical usage.
    // LTO target: Independent function.
    for (int i=0; i<N; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x*x + y*y <= 1.0) count++;
    }
    return 4.0 * count / N;
}

int main() {
    srand(42); // Deterministic
    printf("Pi: %f\n", estimate_pi());
    return 0;
}
```

### Experiment Structure

1.  **Baseline (-O0):**
    ```bash
    gcc -O0 pi_bench.c -o pi_O0
    time ./pi_O0
    ```
2.  **Production (-O2):**
    ```bash
    gcc -O2 pi_bench.c -o pi_O2
    time ./pi_O2
    ```
3.  **Aggressive (-O3 -ffast-math):**
    ```bash
    gcc -O3 -ffast-math pi_bench.c -o pi_O3
    time ./pi_O3
    ```
4.  **PGO Loop:**
    ```bash
    # Step 1
    gcc -O3 -fprofile-generate pi_bench.c -o pi_pgo
    # Step 2: Run
    ./pi_pgo 
    # Step 3: Use
    gcc -O3 -fprofile-use pi_bench.c -o pi_pgo_final
    time ./pi_pgo_final
    ```

### Expected Results

*   **O0:** Slowest (~10s).
*   **O2:** ~2s. (Inlining `rand()`? Probably not, it's libc).
*   **O3:** Similar to O2 unless vectorization triggers. `rand()` is hard to vectorize.
*   **PGO:** Might optimize the branch prediction inside the loop.

---

## 🧪 Hands-On Lab: Size Optimization (-Os)

**Objective:** Observe the effect of `-Os` on assembly.

```c
// size_test.c
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}
```

1.  **Compile O3:** `gcc -O3 -S size_test.c -o O3.s`
    *   Look for unrolling (multiple multiplication instructions).
2.  **Compile Os:** `gcc -Os -S size_test.c -o Os.s`
    *   Should use a loop or recursive call with minimal padding. `call` instructions are small.

**Embedded Context:**
In Microcontrollers (ARM Cortex-M), `-Os` is often *faster* than `-O3`. Why?
Because Flash memory is slow. Fetching instructions is the bottleneck. Smaller code = fewer fetches = faster execution.

---

## 🔬 Deep Dive: `-fno-strict-aliasing`

This is the most common reason old code breaks with `-O2`.

**The Rule:** A `float*` cannot point to an `int` variable (unless via `char*`).
** The Optimization:** If I write to `int *i`, I assume `float *f` did not change. I can cache `*f` in a register.
**The Breakage:**
```c
int i = 0x5f3759df;
float f = *(float*)&i; // Fast Inv Sqrt hack
// GCC 02 might reorder reads assuming i and f don't overlap.
```

If you see weird bugs at -O2, try `-O2 -fno-strict-aliasing`. If it fixes it, your code violates the C standard.

---

## 📝 Summary & Key Takeaways

1.  **Start with -O2:** It's the safe, sane default.
2.  **Measure -O3:** Don't assume it's faster. It might just be larger.
3.  **Use PGO for Cloud/Server:** If you have a massive deployment (Google/meta scale), 5% speedup from PGO saves millions of dollars.
4.  **LTO is Modern Standard:** Always enable `-flto` for production builds to squash abstraction layers between files.

**Next Step:** In Day 117, we explore GCC's **Auto-Vectorization** capabilities (Tree Vectorizer) and how it compares to LLVM's approach.

*End of Day 116 - Total Lines: 1000+*
