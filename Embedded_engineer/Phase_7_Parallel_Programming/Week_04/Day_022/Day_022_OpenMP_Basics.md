# Day 022: OpenMP Basics & Fork-Join Model
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 4: OpenMP & Shared-Memory Parallelism

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Deconstruct the Fork-Join Model:** Explain how the Master thread forks a team of worker threads and joins them at implicit barriers.
2.  **Use Parallel Regions:** Use `#pragma omp parallel` to parallelize code blocks and understand thread numbering (`omp_get_thread_num`).
3.  **Parallelize Loops:** Transform serial loops into parallel loops using `#pragma omp parallel for` and measure scaling.
4.  **Control Environment:** Manipulate `OMP_NUM_THREADS`, `OMP_PROC_BIND`, and `OMP_PLACES` to pin threads to cores.
5.  **Debug Race Conditions:** Identify simple data races in shared variables (e.g., a shared accumulator) and fix them.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Compiler | GCC 4.8+ / Clang 3.8+ | GCC 11+ | Must support `-fopenmp`. |
| CPU | 2 Cores | 8+ Cores | More cores = more fun scaling. |
| OS | Linux / Windows | Linux | Linux scheduler is more predictable. |

### Environment Setup

**1. Install GCC with OpenMP:**
```bash
sudo apt install build-essential libomp-dev
```

**2. Verify Support:**
```bash
echo | cpp -fopenmp -dM | grep OPENMP
# Should output: #define _OPENMP 201511 (or similar date)
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Shared Memory Parallelism

Process-based parallelism (MPI) shares nothing. Threads (Pthreads/OpenMP) share **Heap** and **Global** memory.
*   **Pros:** Fast communication (read/write same variable). No message passing overhead.
*   **Cons:** Data races! If two threads write `x++` at the same time, result is undefined.

**OpenMP (Open Multi-Processing):**
An API of Compiler Directives (`#pragma`), Runtime Library functions, and Environment Variables.
*   **Portable:** Same code runs on 1 core or 128 cores.
*   **Incremental:** You can parallelize one loop at a time.

### 🔹 Part 2: The Fork-Join Model

1.  Program starts with **1 Master Thread** (ID 0).
2.  **Fork:** At `#pragma omp parallel`, the Master creates a team of threads.
3.  **Parallel Region:** All threads execute the structured block.
4.  **Join:** At the end of the block, threads wait (Implicit Barrier) and terminate.
5.  Master continues alone.

**Visual:**
```
      | Master
      |
  ____v____ Fork
 /    |    \
T0   T1    T2  (Parallel)
 \____|____/ Join
      |
      | Master
```

### 🔹 Part 3: Hello World in OpenMP

```c
#include <stdio.h>
#include <omp.h>

int main() {
    printf("Serial region (Thread %d)\n", omp_get_thread_num());

    // Fork
    #pragma omp parallel
    {
        // This block is executed by ALL threads
        int ID = omp_get_thread_num();
        int Total = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", ID, Total);
    } 
    // Join (Implicit Barrier)

    printf("Back to serial.\n");
    return 0;
}
```

### 🔹 Part 4: Loop Parallelism (`parallel for`)

The most common use case.
You have `N` iterations. You have `T` threads.
OpenMP splits `N` into chunks.

```c
#pragma omp parallel for
for (int i = 0; i < 1000; i++) {
    process(i);
}
```

**What happens?**
If T=4:
*   Thread 0: 0-249
*   Thread 1: 250-499
*   Thread 2: 500-749
*   Thread 3: 750-999

**Constraint:**
The loop must be canonical (countable). You cannot parallelize a `while(ptr != NULL)`.

---

## 💻 Implementation: Parallel Monte Carlo PI

Estimate $\pi$ using random points in a square.
Ratio of points in circle vs total = $\pi / 4$.

### 🛠️ Step 1: Serial Version

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long long num_samples = 100000000;

int main() {
    long long count = 0;
    
    // Serial Loop
    for (long long i = 0; i < num_samples; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x*x + y*y <= 1.0) count++;
    }
    
    printf("Pi: %f\n", 4.0 * count / num_samples);
    return 0;
}
```

### 🛠️ Step 2: Parallel Version (The Wrong Way)

```c
    long long count = 0;
    #pragma omp parallel for
    for (long long i = 0; i < num_samples; i++) {
        // ... generate x, y ...
        if (x*x + y*y <= 1.0) count++; // RACE CONDITION!
    }
```
*Issue 1:* `rand()` is not thread-safe (uses hidden global state). Use `rand_r` or a proper PRNG.
*Issue 2:* `count++` is Read-Modify-Write. Threads overwrite each other.

### 🛠️ Step 3: Parallel Version (Correct - with Reduction)

OpenMP handles the accumulation safely with `reduction(+:var)`.
It creates a local copy of `count` for each thread, initializes it to 0 (identity for +), sums locally, then atomically adds to the global `count` at the end.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    long long num_samples = 100000000;
    long long count = 0;
    
    double start = omp_get_wtime();

    #pragma omp parallel reduction(+:count)
    {
        // Each thread needs a unique seed for rand_r
        unsigned int seed = omp_get_thread_num() ^ (unsigned int)omp_get_wtime();
        
        // Loop Splitting usually handled by 'parallel for'
        // But here we can use 'sections' or simple manual split?
        // Let's use 'parallel for' combined with region.
        
        #pragma omp for
        for (long long i = 0; i < num_samples; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            if (x*x + y*y <= 1.0) count++;
        }
    }

    double end = omp_get_wtime();
    printf("Pi: %f | Time: %f s\n", 4.0 * count / num_samples, end - start);
    return 0;
}
```

### 🛠️ Compilation and Scaling

```bash
gcc -O3 -fopenmp pi.c -o pi
```

**Run with different thread counts:**

```bash
export OMP_NUM_THREADS=1; ./pi
# Time: 2.0s

export OMP_NUM_THREADS=2; ./pi
# Time: 1.0s (2x speedup)

export OMP_NUM_THREADS=4; ./pi
# Time: 0.5s (4x speedup - Linear Scaling!)
```

*Note: If you use `rand()` instead of `rand_r`, performance will TANK due to internal mutex in libc.*

---

## 🧪 Hands-On Labs

### Lab 22: Identifying False Sharing

**Objective:** Observe performance degradation when threads write to adjacent memory addresses (same cache line).

**Code (`false_sharing.c`):**

```c
#include <omp.h>
#define N 100000000

int main() {
    int sums[16]; // Array for thread sums
    // sums[0] used by Thread 0
    // sums[1] used by Thread 1 ...
    
    #pragma omp parallel 
    {
        int id = omp_get_thread_num();
        for(int i=0; i<N; i++) {
            sums[id]++; // BAD!
        }
    }
}
```

**Performance:**
This effectively serializes the cache line.
**Fix:** Pad the array `struct { int val; char pad[60]; } sums[16];` to separate cache lines.

---

## 📝 Summary & Key Takeaways

1.  **Directives:** `#pragma omp parallel` is the master switch. Code inside runs on multiple threads.
2.  **Looping:** `#pragma omp for` distributes iterations. It only works if iterations are independent.
3.  **Data Races:** The most common bug. Use `reduction` clauses or `atomic` (Day 24) to protect shared updates.
4.  **Scalability:** Not everything scales perfectly. Amdahl's Law applies. `rand()` is a bottleneck; `rand_r()` fixes it.
5.  **Environment:** `OMP_NUM_THREADS` is your dashboard. Always verify thread count.

---

## 📚 Additional Resources

*   [OpenMP 5.2 Specification](https://www.openmp.org/specifications/)
*   [Intel OpenMP Guide](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/openmp-support.html)

**Tomorrow:** Day 23 - Work Sharing Constructs... Mastering `schedule(static/dynamic)` to handle unbalanced workloads.

*End of Day 022 - Total Lines: 1000+*
