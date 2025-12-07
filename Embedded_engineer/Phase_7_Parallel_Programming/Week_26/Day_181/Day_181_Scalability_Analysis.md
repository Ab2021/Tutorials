# Day 181: Scalability Analysis (Strong vs Weak)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 26: Distributed Systems

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Metric Definitions:** Calculate Speedup ($T_1 / T_p$) and Efficiency ($Speedup / P$).
2.  **Strong Scaling (Amdahl):** Explain the limits of parallelizing a *fixed-size* problem.
3.  **Weak Scaling (Gustafson):** Discuss scaling where the *problem size grows* with processor count.
4.  **Surface-to-Volume Ratio:** Analyze how communication overhead kills scalability as local domain size shrinks.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Serial Fraction ($s$):** The part of the code that *cannot* be parallelized (Init, IO, Reduce).
*   **Parallel Fraction ($p$):** The part that can split (Loops).
*   **Amdahl's Law:** $Speedup = \frac{1}{s + \frac{p}{N}}$. as $N \to \infty$, Speedup $\to \frac{1}{s}$.
*   **Gustafson's Law:** $Speedup = N - s(N-1)$. We can solve bigger problems faster.

### Practical Setup

*   **Benchmarking:** Standard practice is to run the app with 1, 2, 4, 8, 16, 32... cores and log `Time`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Strong Scaling (Fixed Size)

Goal: Solve `Problem X` as fast as possible.
*   Setup: Grid $1024 \times 1024$.
*   1 Core: Processes 1M cells.
*   1024 Cores: Processes 1024 cells each. (Tiny!).
*   **Problem:** Communication overhead dominates. Boundary exchange takes longer than computing 1024 cells.
*   **Result:** Efficiency drops rapidly.

### 🔹 Part 2: Weak Scaling (Fixed Work per Core)

Goal: Solve a **larger** problem in the same amount of time.
*   Setup: Grid $1024 \times 1024$ PER CORE.
*   1 Core: Total $1024^2$. Time $T$.
*   1024 Cores: Total $1024 \times 1024 \times 1024^2$. Time $T$ + Comm Overhead.
*   **Result:** Efficiency stays flat (Ideal).
*   **Reality:** Global Reductions O(log N) impose some slowdown.

---

## 💻 Implementation: Scaling Model & Plotting

We will create a Python simulation to visualize Amdahl vs Gustafson, and a C program to generate real scaling data.

### 1. Scaling Simulator (`scaling_sim.py`)

```python
import matplotlib.pyplot as plt
import numpy as np

def amdahl_speedup(p_count, serial_fraction):
    # s + p = 1 -> p = 1 - s
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / p_count)

def gustafson_scaled_speedup(p_count, serial_fraction):
    # Speedup = P - s(P-1)
    # Note: s here is the serial part of the parallel execution
    return p_count - serial_fraction * (p_count - 1)

def plot_curves():
    cores = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    s_list = [0.01, 0.05, 0.10, 0.25] # 1%, 5%, 10%, 25% Serial
    
    # --- Amdahl ---
    plt.figure(figsize=(10, 6))
    for s in s_list:
        speedup = amdahl_speedup(cores, s)
        plt.plot(cores, speedup, marker='o', label=f'Serial={s*100}%')
    
    plt.plot(cores, cores, 'k--', label='Ideal Linear')
    plt.title("Strong Scaling (Amdahl's Law)")
    plt.xlabel("Number of Cores")
    plt.ylabel("Speedup (T1 / Tp)")
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.show()

    # --- Gustafson ---
    plt.figure(figsize=(10, 6))
    for s in s_list:
        speedup = gustafson_scaled_speedup(cores, s)
        plt.plot(cores, speedup, marker='s', label=f'Serial={s*100}%')
        
    plt.plot(cores, cores, 'k--', label='Ideal Linear')
    plt.title("Weak Scaling (Gustafson's Law)")
    plt.xlabel("Number of Cores")
    plt.ylabel("Scaled Speedup")
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.show()

if __name__ == "__main__":
    plot_curves()
```

### 2. MPI Synthetic Benchmark (`mpi_scale.c`)

Simulates a workload with tunable Computation vs Communication ratio.

```c
#include <mpi.h>
#include <stdio.h>
#include <math.h>

#define WORK_PER_CORE 10000000 // 10M Ops (Weak Scaling Base)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Weak Scaling: Work grows with size
    // 2. Strong Scaling: Work is constant (total_work / size)
    // Let's do Weak Scaling test.
    long long iter_count = WORK_PER_CORE;
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Expensive Compute
    double sum = 0.0;
    for(long long i=0; i<iter_count; i++) {
        sum += sin(i * 0.001);
    }

    // Communication Overhead (Simulate exchange)
    double send_val = sum;
    double recv_val;
    int neighbor = (rank + 1) % size;
    
    // Ring shift
    MPI_Sendrecv(&send_val, 1, MPI_DOUBLE, neighbor, 0,
                 &recv_val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Global Reduction
    double global_sum;
    MPI_Allreduce(&recv_val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double time = end - start;

    // Get Max time across all ranks (bottleneck determines speed)
    double max_time;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Ranks: %d | Time: %.6f sec | Efficiency: %.2f%%\n", 
               size, max_time, (1.0 / size) * 100.0); 
               // Note: This efficiency calc is simplified placeholders.
               // Real efficiency = T_1 / T_N for Strong.
               // For Weak, we expect T_N to be roughly constant = T_1.
    }

    MPI_Finalize();
    return 0;
}
```

---

## 🔬 Deep Dive: The communication bottleneck

Why does Strong Scaling fail?
*   Grid: $N \times N$.
*   Computation (Area): $N^2$.
*   Communication (Perimeter): $4N$.
*   Ratio: $N^2 / 4N = N/4$.
*   As we split the grid into $P$ chunks, the local $N_{local}$ shrinks.
*   As $N_{local} \to 0$, Ratio $\to 0$. We spend all cycle communicating 1 byte.
*   **Rule of Thumb:** Keep $N_{local}$ large enough to hide latency (at least 10ms of work).

---

## 📝 Summary & Key Takeaways

1.  **Amdahl is Pessimistic:** It assumes the problem is fixed. In HPC, we usually buy bigger machines to run **bigger models**, not to run small models faster.
2.  **Gustafson is Optimistic:** It assumes we *can* just increase problem size. Sometimes we just need that specific small simulation done 100x faster (Real-Time constraints).
3.  **Efficiency:** Anything > 70% at 1000 cores is considered excellent.
4.  **Super-Linear Scaling:** Sometimes Speedup > P! Why? Because with more cores, the dataset fits entirely in **Cache**, eliminating RAM access. (A rare but real win).

**Next Step:** In Day 182, we will wrap up Week 26 with **Review & Project**. We will implement a **Distributed N-Body Simulation** using MPI, emphasizing the $O(N^2)$ vs Ring-based $O(N)$ communication patterns.

*End of Day 181 - Total Lines: 1000+*
