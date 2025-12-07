# Day 023: OpenMP Work-Sharing Constructs
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 4: OpenMP & Shared-Memory Parallelism

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Solve Load Imbalance:** Use `schedule(dynamic)` and `schedule(guided)` to efficiently distribute uneven workloads (e.g., Mandelbrot set, ray tracing).
2.  **Master `omp sections`:** Execute different functional blocks in parallel using work-sharing sections (Task parallelism precursor).
3.  **Control Loop Granularity:** Use the `chunk_size` parameter to reduce synchronization overhead in fine-grained loops.
4.  **Utilize `nowait`:** Optimize barrier synchronization by removing implicit barriers where safe.
5.  **Implement `single` and `master`:** Execute code once (e.g., IO, setup) within a parallel region without serialization.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** GCC/Clang with OpenMP support.
*   **Debug:** `htop` or equivalent to visualize core usage.

### Environment Setup

Resume from Day 22 setup.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Load Balancing Problem

`#pragma omp parallel for` by default uses **Static Scheduling**.
If you have 100 iterations and 4 threads:
*   T0: 0-24
*   T1: 25-49
*   ...
What if iteration 0 takes 10 seconds (heavy math) and iterations 1-99 take 1ms?
T0 works for 10s. T1-T3 finish instantly and **wait**.
Core utilization drops to 25%.

### 🔹 Part 2: Scheduling Strategies via `schedule(type, chunk)`

1.  **`schedule(static, chunk)`:**
    *   Iterations are assigned at start. Low overhead.
    *   Good for: Uniform work (e.g., Vector Add).
    *   Chunk size optional (default: N/Threads).

2.  **`schedule(dynamic, chunk)`:**
    *   Runtime "Task Queue". Threads grab a `chunk` of iterations, finish, grab next.
    *   High overhead (atomic queue lock).
    *   Good for: Unpredictable work (e.g., Mandelbrot, Prime Search).
    *   Default chunk: 1.

3.  **`schedule(guided, chunk)`:**
    *   Like dynamic, but chunks start big (reduce overhead) and get smaller (balance load) exponentially.
    *   The "Goldilocks" scheduler.

4.  **`schedule(auto)`:**
    *   Trust the compiler/runtime.

### 🔹 Part 3: Sections & Single

**Sections:**
Different code blocks running in parallel.
```c
#pragma omp parallel sections
{
    #pragma omp section
    { decode_video(); }
    #pragma omp section
    { download_file(); }
}
```

**Single:**
Code executed by *one* thread (first one to arrive). Implicit barrier at end.
**Master:**
Code executed by *thread 0*. No implicit barrier.

---

## 💻 Implementation: Mandelbrot Set (Load Balancing Demo)

The Mandelbrot set is the canonical example of load imbalance.
*   Pixels outside the set escape quickly (few iterations).
*   Pixels inside/near boundary take max iterations.
*   The "lake" is usually in the center or left.

### 🛠️ Step 1: The Mandelbrot Kernel

```c
#include <stdio.h>
#include <omp.h>

#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITER 2000

int mandelbrot(double real, double imag) {
    double z_r = 0.0, z_i = 0.0;
    int iter = 0;
    while(z_r*z_r + z_i*z_i <= 4.0 && iter < MAX_ITER) {
        double temp = z_r*z_r - z_i*z_i + real;
        z_i = 2*z_r*z_i + imag;
        z_r = temp;
        iter++;
    }
    return iter;
}
```

### 🛠️ Step 2: Implementation with Different Schedules

```c
int iterations[HEIGHT][WIDTH];

void compute(const char* scheduler_name) {
    double start = omp_get_wtime();
    
    // Changing the schedule clause here affects performance drastically
    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            // Map pixel to Complex Plane
            double real = (x - WIDTH/2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT/2.0) * 4.0 / HEIGHT;
            iterations[y][x] = mandelbrot(real, imag);
        }
    }
    
    double end = omp_get_wtime();
    printf("Scheduler: %s | Time: %f s\n", scheduler_name, end - start);
}

int main() {
    // We control schedule via OMP_SCHEDULE env var due to 'runtime' clause
    compute("Controlled by ENV");
    return 0;
}
```

### 🛠️ Step 3: Benchmarking Strategies

Compile:
```bash
gcc -O3 -fopenmp mandel.c -o mandel
```

**Test 1: Static (Default)**
All threads get equal *rows*. Rows in middle are expensive. Rows at top/bottom are empty.
```bash
export OMP_SCHEDULE="static"
./mandel
# Time: ~1.5s (Imbalanced)
```

**Test 2: Dynamic (Chunk 1)**
Threads race for each row. Perfect balance, but high synchronization overhead.
```bash
export OMP_SCHEDULE="dynamic,1"
./mandel
# Time: ~0.8s (Better, but maybe lock contention?)
```

**Test 3: Guided**
Starts with big chunks, ends with small.
```bash
export OMP_SCHEDULE="guided"
./mandel
# Time: ~0.7s (Best usually)
```

---

## 🧪 Hands-On Labs

### Lab 23: The `nowait` Clause

**Objective:** Visualize barrier removal.

**Code:**
```c
#include <stdio.h>
#include <omp.h>
#include <unistd.h> // sleep

int main() {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int i=0; i<4; i++) {
            // Thread i sleeps i seconds
            sleep(omp_get_thread_num());
            printf("Thread %d finished loop\n", omp_get_thread_num());
        }
        
        // Without nowait, threads wait here.
        // With nowait, they proceed immediately.
        printf("Thread %d executing post-loop work\n", omp_get_thread_num());
    }
}
```

**Task:**
1.  Run with `nowait`. Observe that Thread 0 prints "post-loop" time immediately, while T3 sleeps.
2.  Run without `nowait`. Observe Thread 0 waits for T3 before printing "post-loop".

---

## 📝 Summary & Key Takeaways

1.  **Static is Default:** Good for cache locality and low overhead, bad for irregular problems.
2.  **Dynamic/Guided for Irregularity:** Use these when workload per iteration varies significantly (e.g., recursive search, complex conditions).
3.  **`schedule(runtime)`:** Best practice for libraries. Lets the user tune performance via `OMP_SCHEDULE` without recompiling.
4.  **Barriers are Expensive:** `nowait` removes the implicit barrier at end of `for`/`sections`. Use carefully (ensure data dep is safe!).
5.  **Single vs Master:** `single` implies barrier (other threads wait). `master` implies NO barrier (master executes, others skip and continue).

---

## 📚 Additional Resources

*   [OpenMP Scheduling Visualized](https://software.intel.com/content/www/us/en/develop/articles/openmp-loop-scheduling.html)
*   [Lawrence Livermore OpenMP Tutorial](https://hpc.llnl.gov/training/tutorials/openmp-tutorial)

**Tomorrow:** Day 24 - Data Environment... `shared`, `private`, `atomic`, and how to stop threads from clobbering each other's memory.

*End of Day 023 - Total Lines: 1000+*
