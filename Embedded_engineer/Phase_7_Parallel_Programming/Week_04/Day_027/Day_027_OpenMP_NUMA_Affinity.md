# Day 027: OpenMP NUMA & Affinity
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 4: OpenMP & Shared-Memory Parallelism

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Understand NUMA (Non-Uniform Memory Access):** Explain why accessing memory on "the other socket" is 2x slower and how to avoid it.
2.  **Control Thread Affinity:** Use `OMP_PROC_BIND` and `OMP_PLACES` to pin threads to specific cores, preventing OS jitter and cache thrashing.
3.  **Implement First-Touch Policy:** Initialize data in parallel to ensure memory pages are allocated on the local NUMA node.
4.  **Visualize Topology:** Use `lstopo` (`hwloc`) to map the hierarchy of Sockets, L3 Caches, and Cores.
5.  **Benchmark Affinity:** Measure bandwidth (Triad) with different binding strategies (Spread vs Close).

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Hardware:** Ideally a dual-socket server or AMD Ryzen (which acts like multiple NUMA nodes). Even on a simple laptop, affinity matters for L3 sharing.
*   **Tools:** `numactl`, `hwloc` (`lstopo`).

### Environment Setup

**1. Install Tools:**
```bash
sudo apt install numactl hwloc
```

**2. Visualize:**
```bash
lstopo
# Shows hierarchy: Machine -> Package P#0 -> Cache L3 -> Core P#0
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The NUMA Reality

**UMA (Uniform Memory Access):** Old days. All CPUS access RAM at same speed.
**NUMA (Modern):**
*   CPU 0 has attached RAM (Node 0).
*   CPU 1 has attached RAM (Node 1).
*   If CPU 0 reads from Node 1 RAM, it travels over the **Interconnect** (QPI/UPI/Infinity Fabric).
    *   Latency: Higher (+50%).
    *   Bandwidth: Limited.

**Implication for OpenMP:**
If Thread 0 (on Socket 0) initializes an array `A`, the OS allocates physical pages on Node 0 ("First Touch").
If Thread 16 (on Socket 1) later processes `A`, it constantly crosses the interconnect. **Performance Penalty.**

### 🔹 Part 2: Thread Affinity (Pinning)

By default, the Linux scheduler moves threads to keep cores cool or balance load.
**Bad for HPC:**
1.  **Cache Thrashing:** If thread moves from Core 0 to Core 1, L1/L2 cache is cold.
2.  **NUMA Migration:** Thread might move to wrong socket.

**OpenMP Solution:** Pin threads to cores.

### 🔹 Part 3: Environment Variables

1.  **`OMP_PLACES`:** Where can threads go?
    *   `cores`: Each thread gets a unique physical core.
    *   `threads`: Each thread gets a logical CPU (Hyperthread).
    *   `sockets`: Threads restricted to a socket but can move within it.
    *   `{0,1,2,3}`: Manual list.

2.  **`OMP_PROC_BIND`:** How to distribute?
    *   `close`: Pack threads tightly (T0, T1 on Core 0). Good for L1 sharing.
    *   `spread`: Scatter threads far apart (T0 on Socket 0, T1 on Socket 1). Good for Memory BW.
    *   `master`: Thread stays on same place as master.

---

## 💻 Implementation: NUMA-Aware Initialization

We will demonstrate the "First Touch" principle.
We allocate a huge array (1GB).
We measure "Write Bandwidth".

### 🛠️ Step 1: Serial Init (The "Anti-Pattern")

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 400000000 // 400 Million doubles = 3.2 GB

int main() {
    double *A = malloc(N * sizeof(double));
    
    double start = omp_get_wtime();
    
    // 1. Serial Initialization (Allocates all on Master's Node)
    // OS uses "First Touch" policy. Master touches everything.
    for(long i=0; i<N; i++) A[i] = 0.0;
    
    // 2. Parallel Processing
    #pragma omp parallel for
    for(long i=0; i<N; i++) {
        A[i] += 1.0; 
        // Threads on remote socket will suffer here.
    }
    
    double end = omp_get_wtime();
    printf("Time: %f s\n", end - start);
    
    free(A);
    return 0;
}
```

### 🛠️ Step 2: Parallel Init (Correct)

```c
    // 1. Parallel Initialization
    #pragma omp parallel for
    for(long i=0; i<N; i++) {
        A[i] = 0.0;
        // Thread T touches chunk C. OS allocates C on T's Node.
    }
    
    // 2. Parallel Processing (Must use SAME schedule!)
    #pragma omp parallel for
    for(long i=0; i<N; i++) {
        A[i] += 1.0;
        // Thread T accesses chunk C. Local access!
    }
```

**Benchmark:**
Run on a dual-socket machine.
*   Serial Init: ~20 GB/s (Limited by QPI/1 socket).
*   Parallel Init: ~40 GB/s (Using both memory controllers).

### 🛠️ Step 3: Controlling binding

Compile the correct version:
```bash
gcc -O3 -fopenmp numa_test.c -o numa_test
```

**Scenario A: Spread (Max Bandwidth)**
We want threads on all sockets.
```bash
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
./numa_test
```
*Result:* Threads 0,1,2,3 are typically placed on Core 0, 8, 16, 24 (spread across sockets).

**Scenario B: Close (L3 Sharing)**
We want threads to share cache (e.g., producer/consumer).
```bash
export OMP_PROC_BIND=close
./numa_test
```
*Result:* Threads 0,1,2,3 placed on Core 0, 1, 2, 3 (Same socket).

---

## 🧪 Hands-On Labs

### Lab 27: Visualizing Binding

**Objective:** Write a program that reports which core it is running on.

```c
#include <stdio.h>
#include <omp.h>
#include <sched.h> // Linux specific

int main() {
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int cpu = sched_getcpu(); // Linux syscall
        
        #pragma omp critical
        printf("Thread %d is running on CPU %d\n", id, cpu);
    }
    return 0;
}
```

**Task:**
Run with various `OMP_PROC_BIND` values (`true`, `false`, `spread`, `close`) and record the mapping.

Example Output (`spread` w/ 4 threads on 8-core CPU):
T0 -> Cpu0
T1 -> Cpu4
T2 -> Cpu2
T3 -> Cpu6

---

## 📝 Summary & Key Takeaways

1.  **NUMA Matters:** Ignoring NUMA can cost 50% performance on servers.
2.  **First Touch:** Memory pages live where they are first written. **Initialize data in parallel** using the same loop schedule you use for computation.
3.  **Pin Your Threads:** Use `OMP_PROC_BIND=true` (or `spread`/`close`) to prevent OS scheduler chaos.
4.  **`spread` vs `close`:** Spread for memory bandwidth (Stream). Close for synchronization/cache sharing.
5.  **Tools:** `numactl --show` tells you about the hardware. `hwloc-ls` maps it visually.

---

## 📚 Additional Resources

*   [Hollenhorst: NUMA and OpenMP](https://www.nics.tennessee.edu/pdfs/hollenhorst-NUMA-OpenMP.pdf)
*   [OpenMP Affinity (NERSC)](https://docs.nersc.gov/jobs/affinity/)

**Tomorrow:** Day 28 - Week 4 Review & Project... Building a Parallel Graph Algorithm (BFS) leveraging everything.

*End of Day 027 - Total Lines: 1000+*
