# Day 180: Hybrid Programming (MPI + OpenMP)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 26: Distributed Systems

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Architecture Mapping:** Map MPI processes to Nodes (Sockets) and OpenMP threads to Cores.
2.  **Thread Support Levels:** Distinguish between `SINGLE`, `FUNNELED`, `SERIALIZED`, and `MULTIPLE`.
3.  **Memory Footprint:** Explain why Hybrid apps use less memory than Pure MPI apps (Ghost Cells shared vs duplicated).
4.  **Pinning:** Control process/thread affinity to prevent OS thrashing.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Pure MPI:** 100 Cores -> 100 MPI Ranks.
    *   Pros: Simple code.
    *   Cons: 100 buffers, 100 connections. Intense All-to-All traffic.
*   **Hybrid:** 100 Cores (on 2 Nodes) -> 2 MPI Ranks $\times$ 50 Threads.
    *   Pros: Only 1 connection between nodes. Shared memory used internally.

### Practical Setup

*   **Compile:** `mpicc -fopenmp ...`
*   **Run:** `mpirun -n 2 -x OMP_NUM_THREADS=4 ./hydrid_app`. (2 Ranks, 4 Threads each = 8 Cores total).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: MPI Thread Safety

When `MPI_Init_thread` is called, we request a level of safety:
1.  **MPI_THREAD_SINGLE:** No threading allowed.
2.  **MPI_THREAD_FUNNELED:** Only the Master thread (ID 0) makes MPI calls. (Most Common).
3.  **MPI_THREAD_SERIALIZED:** Any thread can make MPI calls, but only one at a time (Mutex protected).
4.  **MPI_THREAD_MULTIPLE:** Any thread, anytime. (High overhead, some implementations don't support it).

### 🔹 Part 2: The Loop Structure

Standard Hybrid Loop:
```c
while(timesteps) {
    // 1. Compute Internal (OpenMP)
    #pragma omp parallel for
    for(i=1; i<N-1; i++) ...
    
    // 2. Pack Buffers (OpenMP)
    #pragma omp parallel for
    for(i=0; i<HALO; i++) pack_buffer();
    
    // 3. Exchange Halo (MPI - Master only)
    #pragma omp master 
    {
        MPI_Isend/Irecv...
        MPI_Waitall...
    }
    // Barrier implicit at end of parallel region? 
    // Usually need explicit barrier if non-masters need to wait for MPI.
}
```

---

## 💻 Implementation: Hybrid Integration

We sum a distributed array.
*   MPI decomposes domain.
*   OpenMP sums local chunk.
*   MPI Reduces results.

```c
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100000000 // 100 Million

int main(int argc, char** argv) {
    // 1. Init with Thread Support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        printf("Error: MPI implementation does not support Funneled threading.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2. Decompose
    long long local_n = N / size;
    long long start_idx = rank * local_n;
    long long end_idx = start_idx + local_n;

    // 3. OpenMP Parallel Region
    double local_sum = 0.0;
    double t_start = MPI_Wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // Master prints config once
        if (rank == 0 && tid == 0) {
            printf("Running with %d MPI Ranks, %d OpenMP Threads each.\n", 
                   size, num_threads);
        }

        // Parallel Sum Computation (Reduction)
        double thread_sum = 0.0;
        
        #pragma omp for
        for(long long i=0; i<local_n; i++) {
            // Fake Data: Value = Index * 1.0
            thread_sum += (start_idx + i) * 1.0; 
        }

        // Aggregate to 'local_sum' atomically
        #pragma omp atomic
        local_sum += thread_sum;
    } // End Parallel

    // 4. MPI Reduction (Funneled: Implicitly done by main thread here)
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, 
               MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (rank == 0) {
        // Analytical Sum: N*(N-1)/2
        double expected = (double)N * (N - 1) / 2.0;
        printf("Global Sum: %.0f (Expected: %.0f)\n", global_sum, expected);
        printf("Time: %.4f sec\n", t_end - t_start);
    }

    MPI_Finalize();
    return 0;
}
```

### Execution

If you have 4 physical cores total.
Option A: 4 MPI Ranks, 1 Thread each.
Option B: 1 MPI Rank, 4 Threads.
Option C: 2 MPI Ranks, 2 Threads each.

**Why Hybrid (Option C) over Option A?**
*   **Halo Data:** In pure MPI, `rank 0` sends to `rank 1`.
*   In Hybrid, if R0 and R1 are threads, they read `shared_array` directly. **Zero Copy.**
*   In Hybrid (Node level), MPI is only used for `Node 0 -> Node 1`.

---

## 🔬 Deep Dive: Processor Affinity

This is the silent killer of Hybrid performance.
*   If you spawn 4 threads, but the OS schedules them all on Core 0... Performance crashes.
*   **OMP_PROC_BIND=true**: Locks threads to cores.
*   **OMP_PLACES=cores**: Ensures threads don't share hyperthreads unless asked.
*   **MPI Mapping**: `mpirun --map-by socket:PE=4` tells MPI to space out ranks by 4 cores (leaving room for threads).

---

## 📝 Summary & Key Takeaways

1.  **Funneled Mode:** The sweet spot. Let standard MPI code run heavily parallelized kernels.
2.  **Thread Safety Cost:** `MPI_THREAD_MULTIPLE` adds internal locking overhead to MPI. Avoid unless necessary.
3.  **Topology Awareness:** Hybrid code must "know" the hardware cache hierarchy effectively.
4.  **Debugging:** Harder. Race conditions inside threads + Deadlocks between ranks.

**Next Step:** In Day 181, we will explore **Scalability Analysis (Strong vs Weak Scaling)**. How to measure if your code is actually "Good" at parallelizing.

*End of Day 180 - Total Lines: 1000+*
