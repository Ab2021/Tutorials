# Day 178: MPI Collectives (Broadcast, Scatter, Reduce)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 26: Distributed Systems

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Collective Concept:** Understand that all processes in the communicator MUST call the function.
2.  **Data Movement Patterns:** Distinguish between One-to-All (Bcast), All-to-One (Gather/Reduce), and All-to-All.
3.  **Optimization:** Explain why `Reduce` uses a tree structure O(log P) instead of linear accumulation O(P).
4.  **Allreduce:** Implement global synchronization of results (vital for ML training).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Synchronous:** Collectives usually imply a barrier. (Rank 0 enters Reduce, waits for Rank N).
*   **Root:** The designated process that originates data (Bcast/Scatter) or receives results (Gather/Reduce).
*   **Algorithmic Complexity:**
    *   Naive Broadcast: Root sends to P-1 nodes sequentially. Time $O(P)$.
    *   Tree Broadcast: Root to 2, 2 to 4... Time $O(\log P)$.

### Practical Setup

*   **Scenario:** Monte Carlo approximation of Pi.
*   **Method:** Throw darts at a square. Count how many land in the circle.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Move Primitives

1.  **MPI_Bcast:**
    *   One buffer `buf` on Root is copied to `buf` on everyone.
    *   Arguments: `(void* buffer, int count, MPI_Datatype, int root, MPI_Comm)`.

2.  **MPI_Scatter:**
    *   Root has array `BigArr`.
    *   Scatter breaks `BigArr` into chunks.
    *   Rank i receives `BigArr[i * chunk_size]`.

3.  **MPI_Gather:**
    *   Inverse of Scatter. Everyone sends their `LocalArr`. Root reassembles `BigArr`.

4.  **MPI_Reduce:**
    *   Combine values using an Operator (`MPI_SUM`, `MPI_MAX`, `MPI_PROD`).
    *   Result ends up ONLY on Root.

5.  **MPI_Allreduce:**
    *   Same as Reduce, but result is Broadcast back to everyone.
    *   Cost $\approx 2 \times Reduce$.

---

## 💻 Implementation: Parallel Pi (Monte Carlo)

We utilize the embarrassingly parallel nature of Monte Carlo.
1.  Bcast the number of throws.
2.  Each rank throws darts locally.
3.  Reduce the total hits to Root.

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long total_throws = 0;
    
    // 1. Broadcast Input
    if (rank == 0) {
        total_throws = 100000000; // 100 Million throws
        printf("Master: Broadcasting task %lld throws to %d workers.\n", 
               total_throws, size);
    }
    
    // Everyone waits here until they receive 'total_throws' from Root (0).
    MPI_Bcast(&total_throws, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // 2. Perform Local Work
    long long throws_per_rank = total_throws / size;
    long long local_hits = 0;
    
    // Seed unique to rank (IMPORTANT!)
    unsigned int seed = (unsigned int)time(NULL) + rank * 999;
    
    for(long long i=0; i<throws_per_rank; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        
        if ((x*x + y*y) <= 1.0) {
            local_hits++;
        }
    }
    
    printf("Rank %d: %lld hits out of %lld throws.\n", 
           rank, local_hits, throws_per_rank);

    // 3. Reduce Results
    long long global_hits = 0;
    
    // All ranks call Reduce. 
    // Data moves from 'local_hits' -> combined via SUM -> 'global_hits' on Root.
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, 
               MPI_SUM, 0, MPI_COMM_WORLD);

    // 4. Output Result
    if (rank == 0) {
        double pi = 4.0 * (double)global_hits / (double)total_throws;
        printf("Result: Pi = %.10f\n", pi);
    }

    MPI_Finalize();
    return 0;
}
```

### Execution Flow (4 Nodes)

1.  **Bcast:** Rank 0 sends `100,000,000`. Nodes 1, 2, 3 receive it.
2.  **Compute:** Each simulates 25M throws.
3.  **Reduce:**
    *   Node 0 + Node 1 -> Intermediate A.
    *   Node 2 + Node 3 -> Intermediate B.
    *   A + B -> Final Result (on Node 0).
4.  **Print:** Node 0 prints Pi.

---

## 🔬 Deep Dive: Implementation of Reduction

How does MPI implement `Reduce` fast?
**Binomial Tree:**
*   Step 1: Rank 1 sends to 0. Rank 3 sends to 2. Rank 5 sends to 4...
    *   Active: 0, 2, 4, 6... (Size P/2)
*   Step 2: Rank 2 sends to 0. Rank 6 sends to 4...
    *   Active: 0, 4, 8... (Size P/4)
*   Step 3: Rank 4 sends to 0.
    *   Rank 0 has Sum(0..7).
*   **Latency:** $ \lceil \log_2 P \rceil $ steps.
*   Much better than linear $O(P)$ latency of sending everything to 0 one by one.

---

## 📝 Summary & Key Takeaways

1.  **Collective safety:** Everyone must call the collective. If Rank 1 skips `Bcast`, Rank 0 waits forever (or until timeout).
2.  **Type Safety:** `MPI_LONG_LONG` matches C's `long long`. Mismatch causes garbage data.
3.  **Efficiency:** Optimized Collectives exploit the underlying network topology (Hypercube, Mesh, Torus).
4.  **Allreduce:** The foundation of Distributed Deep Learning (Parameter averaging across GPUs).

**Next Step:** In Day 179, we will cover **Parallel I/O (MPI-IO)**. Writing to a single file from thousands of processes efficiently without destroying the file system.

*End of Day 178 - Total Lines: 1000+*
