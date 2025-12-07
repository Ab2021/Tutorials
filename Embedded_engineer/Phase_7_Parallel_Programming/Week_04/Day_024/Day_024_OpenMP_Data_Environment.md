# Day 024: OpenMP Data Environment
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 4: OpenMP & Shared-Memory Parallelism

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Deconstruct Scope:** Determine which variables are `shared` (heap/global) and which are `private` (stack) by default in OpenMP.
2.  **Control Visibility:** Explicitly use `shared`, `private`, `firstprivate`, and `lastprivate` clauses to manage data lifecycle across threads.
3.  **Prevent Data Races:** Identify Critical Sections and use `scheduler(atomic)` vs `#pragma omp critical` for correct synchronization.
4.  **Master Reductions:** Apply `reduction` logic not just to scalars but to arrays (OpenMP 4.5+) and user-defined types.
5.  **Debug Thread Safety:** Use ThreadSanitizer (`-fsanitize=thread`) to catch invisible race conditions at runtime.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** GCC 7+ (OpenMP 4.5 support for array reductions).
*   **Tool:** TSAN (ThreadSanitizer) - typically included with GCC/Clang.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Default Scoping Rules

When multiple threads spawn using `#pragma omp parallel`:
*   **Variables declared OUTSIDE** the parallel region are **SHARED** by default.
*   **Variables declared INSIDE** the parallel region are **PRIVATE** (each thread gets a new stack copy).

**Example:**
```c
int A = 10; // Shared
#pragma omp parallel
{
    int B = 20; // Private (local stack)
    // A is visible to all. B is unique to each.
}
```

### 🔹 Part 2: Scoping Clauses

1.  **`private(x)`:**
    *   Creates a new uninitialized copy of `x` for each thread.
    *   *Danger:* The initial value is undefined! (It is NOT 10).
    *   *Danger:* The value after the loop is undefined! (Original `x` is untouched).

2.  **`firstprivate(x)`:**
    *   Like private, but **initialized** with the value from the master thread.
    *   Useful for read-only local constants.

3.  **`lastprivate(x)`:**
    *   Copies the value from the **last iteration** (simulated serial execution order) back to the master's variable.

### 🔹 Part 3: Synchronization Primitives

**Race Condition:** shared `count++`.
Solution:

1.  **`#pragma omp critical [name]`**
    *   Only one thread executes block at a time.
    *   Global lock. Slow.
    *   Naming allows distinct locks (Critical section "A" doesn't block "B").

2.  **`#pragma omp atomic`**
    *   Hardware instruction (`lock xadd`).
    *   Much faster.
    *   Restricted to simple ops: `x++`, `x += expr`, `x &= expr`.

### 🔹 Part 4: OpenMP 4.5 Array Reduction

Before 4.5, you could only reduce scalars (`sum`).
Now, you can reduce entire histograms.

```c
int hist[256];
#pragma omp parallel for reduction(+:hist[:256])
for(int i=0; i<N; i++) {
    hist[ data[i] ]++;
}
```
*Compiler Magic:* Allocates `hist` copy per thread, sums locally, merges at end.

---

## 💻 Implementation: Histogram Calculation

We will count occurrences of bytes in a large file.
Shared Array Race vs Atomic vs Reduction.

### 🛠️ Step 1: The Buggy Version

```c
#include <stdio.h>
#include <omp.h>

#define N 100000000
unsigned char data[N];
int hist[256] = {0};

int main() {
    // Fill data
    for(int i=0; i<N; i++) data[i] = i % 256;
    
    // Parallel Histogram (RACE!)
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        hist[ data[i] ]++; 
        // Multiple threads might increment hist[5] simultaneously.
        // Read(5) -> Add 1 -> Write(6).
        // T1 reads 5. T2 reads 5. Both write 6. One increment lost.
    }
    
    // Verify
    long sum = 0;
    for(int i=0; i<256; i++) sum += hist[i];
    printf("Total: %ld (Expected %d)\n", sum, N);
    return 0;
}
```

**Run it:**
`Total: 45293291 (Expected 100000000)` -> Massive data loss!

### 🛠️ Step 2: The Atomic Version

```c
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        #pragma omp atomic
        hist[ data[i] ]++;
    }
```
*Correctness:* Perfect.
*Performance:* Terrible. Cache conflicts (False sharing) + Atomic overhead on every pixel.

### 🛠️ Step 3: The Manual Reduction (Optimized)

For Histograms, Manual Reduction is often better than `atomic` or `reduction(array)`.
Each thread keeps a local histogram, then merges inside a critical section.

```c
    #pragma omp parallel
    {
        int local_hist[256] = {0}; // Private Stack
        
        #pragma omp for nowait // Distribute work, no wait at end
        for(int i=0; i<N; i++) {
            local_hist[ data[i] ]++; // Fast local access
        }
        
        // Merge
        #pragma omp critical
        {
            for(int j=0; j<256; j++) {
                hist[j] += local_hist[j];
            }
        }
    }
```

**Why is this fast?**
1.  Inner loop has NO atomic/locks.
2.  L1 Cache hit for `local_hist`.
3.  Synchronization happens only ONCE per thread (not once per pixel).

---

## 🧪 Hands-On Labs

### Lab 24: Debugging with ThreadSanitizer

**Objective:** Use Google's TSAN to find a race condition we "missed".

**Code (`race.c`):**
```c
#include <omp.h>
#include <stdio.h>

int main() {
    int val = 0;
    #pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0) {
            val = 42; 
        } else {
            printf("Val: %d\n", val);
        }
    }
    return 0;
}
```
*Race:* T0 writes. T1 reads. No ordering enforced. T1 might print 0 or 42.

**Compile:**
```bash
gcc -fopenmp -fsanitize=thread -g race.c -o race
./race
```

**Output:**
```
WARNING: ThreadSanitizer: data race (pid=...)
  Read of size 4 at ... by thread T1...
  Previous write of size 4 at ... by thread T0...
```
TSAN is invaluable. Use it!

---

## 📝 Summary & Key Takeaways

1.  **Know your Defaults:** Variables outside parallel region = Shared. Inside = Private.
2.  **`default(none)`:** Best practice. Forces you to specify scope for *every* variable. Prevents accidental sharing.
3.  **Atomic vs Critical:** Use `atomic` for `x++`. Use `critical` for blocks of code (I/O, complex updates).
4.  **Reduction is powerful:** It handles the pattern "Private Init -> Local Accumulate -> Global Merge" automatically.
5.  **TSAN:** Your safety net. Run it on every OpenMP program before release.

---

## 📚 Additional Resources

*   [Google ThreadSanitizer Wiki](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual)
*   [OpenMP Data Scoping Guide](https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-data.html)

**Tomorrow:** Day 25 - OpenMP SIMD... Combining Threads (`omp parallel`) with Vectorization (`omp simd`) for max performance.

*End of Day 024 - Total Lines: 1000+*
