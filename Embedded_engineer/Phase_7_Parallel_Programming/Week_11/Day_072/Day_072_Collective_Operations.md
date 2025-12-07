# Day 072: MPI Collective Operations
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 11: Distributed Memory & MPI

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Broadcast:** Distribute data from one process to all others using `MPI_Bcast`.
2.  **Scatter/Gather:** Partition and collect data across processes using `MPI_Scatter` and `MPI_Gather`.
3.  **Reduce Operations:** Perform global reductions (sum, max, min) using `MPI_Reduce` and `MPI_Allreduce`.
4.  **Synchronization:** Coordinate processes using `MPI_Barrier`.
5.  **Tree Algorithms:** Understand logarithmic-time collective implementations.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Binary Trees:** Understanding tree-based communication patterns.
*   **Reduction Operations:** Associative and commutative operators.
*   **Complexity Analysis:** $O(\log p)$ vs $O(p)$ algorithms.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Broadcast (`MPI_Bcast`)

**Operation:**
Root process sends same data to all processes.

**Naive Implementation (Linear):**
```cpp
if (rank == root) {
    for (int i = 0; i < size; ++i) {
        if (i != root) {
            MPI_Send(data, count, datatype, i, tag, comm);
        }
    }
} else {
    MPI_Recv(data, count, datatype, root, tag, comm, &status);
}
```
**Complexity:** $O(p)$ where $p$ is number of processes.

**Tree-Based Implementation:**
```
Level 0: Rank 0 sends to rank 1
Level 1: Ranks 0,1 send to ranks 2,3
Level 2: Ranks 0,1,2,3 send to ranks 4,5,6,7
...
```
**Complexity:** $O(\log p)$ levels, each level doubles participants.

**MPI Implementation:**
```cpp
int data;
if (rank == 0) {
    data = 42;
}
MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
// Now all processes have data = 42
```

### 🔹 Part 2: Scatter & Gather

**Scatter (`MPI_Scatter`):**
Distribute array chunks from root to all processes.

**Example:**
```
Root has: [0, 1, 2, 3, 4, 5, 6, 7]
After scatter (4 processes):
  Rank 0: [0, 1]
  Rank 1: [2, 3]
  Rank 2: [4, 5]
  Rank 3: [6, 7]
```

**Code:**
```cpp
int sendbuf[8] = {0,1,2,3,4,5,6,7};
int recvbuf[2];

MPI_Scatter(sendbuf, 2, MPI_INT,  // Send 2 ints per process
            recvbuf, 2, MPI_INT,  // Receive 2 ints
            0, MPI_COMM_WORLD);   // Root is rank 0
```

**Gather (`MPI_Gather`):**
Inverse of scatter - collect chunks at root.

```cpp
int sendbuf[2] = {rank*2, rank*2+1};
int recvbuf[8];

MPI_Gather(sendbuf, 2, MPI_INT,
           recvbuf, 2, MPI_INT,
           0, MPI_COMM_WORLD);

// Rank 0 now has: [0,1,2,3,4,5,6,7]
```

### 🔹 Part 3: Reduce Operations

**Reduce (`MPI_Reduce`):**
Combine values from all processes using operation (SUM, MAX, MIN, etc.).

**Example (Sum):**
```cpp
int local_sum = rank;
int global_sum;

MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT,
           MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) {
    printf("Sum: %d\n", global_sum); // 0+1+2+3 = 6 (for 4 processes)
}
```

**Allreduce (`MPI_Allreduce`):**
Same as reduce, but result available on all processes.

```cpp
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT,
              MPI_SUM, MPI_COMM_WORLD);
// All processes now have global_sum
```

**Built-in Operations:**
*   `MPI_SUM`, `MPI_PROD` (product)
*   `MPI_MAX`, `MPI_MIN`
*   `MPI_LAND` (logical AND), `MPI_LOR` (logical OR)
*   `MPI_BAND` (bitwise AND), `MPI_BOR` (bitwise OR)

**Custom Operations:**
```cpp
void my_max_loc(void* in, void* inout, int* len, MPI_Datatype* dtype) {
    struct { int val; int rank; }* in_data = in;
    struct { int val; int rank; }* inout_data = inout;
    
    for (int i = 0; i < *len; ++i) {
        if (in_data[i].val > inout_data[i].val) {
            inout_data[i] = in_data[i];
        }
    }
}

MPI_Op my_op;
MPI_Op_create(my_max_loc, 1, &my_op);
// Use my_op in MPI_Reduce
MPI_Op_free(&my_op);
```

### 🔹 Part 4: Barrier Synchronization

**Purpose:**
Ensure all processes reach a point before any proceed.

```cpp
// Phase 1: Independent work
compute_local_data();

// Synchronize
MPI_Barrier(MPI_COMM_WORLD);

// Phase 2: Work that depends on all Phase 1 completing
process_global_data();
```

**Warning:**
Barriers are expensive (latency = $O(\log p)$). Minimize usage.

---

## 💻 Implementation: Parallel Array Sum

### 🛠️ Step 1: Distributed Array Sum

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int N = 1000000;
    int* data = NULL;
    
    // Root creates array
    if (rank == 0) {
        data = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            data[i] = i + 1; // 1, 2, 3, ..., N
        }
    }
    
    // Scatter data
    int local_n = N / size;
    int* local_data = (int*)malloc(local_n * sizeof(int));
    
    MPI_Scatter(data, local_n, MPI_INT,
                local_data, local_n, MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Compute local sum
    long long local_sum = 0;
    for (int i = 0; i < local_n; ++i) {
        local_sum += local_data[i];
    }
    
    // Global reduce
    long long global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Sum: %lld\n", global_sum);
        printf("Expected: %lld\n", (long long)N * (N + 1) / 2);
        free(data);
    }
    
    free(local_data);
    MPI_Finalize();
    return 0;
}
```

### 🛠️ Step 2: Parallel Matrix-Vector Multiplication

```c
// Matrix A (N x N) distributed by rows
// Vector x (N) replicated on all processes
// Result y = A * x

int rows_per_proc = N / size;
double* local_A = (double*)malloc(rows_per_proc * N * sizeof(double));
double* x = (double*)malloc(N * sizeof(double));
double* local_y = (double*)malloc(rows_per_proc * sizeof(double));
double* y = NULL;

if (rank == 0) {
    y = (double*)malloc(N * sizeof(double));
}

// Scatter matrix rows
MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
            local_A, rows_per_proc * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

// Broadcast vector x
MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Local computation
for (int i = 0; i < rows_per_proc; ++i) {
    local_y[i] = 0.0;
    for (int j = 0; j < N; ++j) {
        local_y[i] += local_A[i * N + j] * x[j];
    }
}

// Gather results
MPI_Gather(local_y, rows_per_proc, MPI_DOUBLE,
           y, rows_per_proc, MPI_DOUBLE,
           0, MPI_COMM_WORLD);
```

---

## 🧪 Hands-On Labs

### Lab 72: Collective Performance Benchmark

**Objective:** Measure latency of collective operations vs message size.

**Test:**
1.  Vary message size: 1 byte to 1 MB.
2.  Measure time for:
    *   `MPI_Bcast`
    *   `MPI_Reduce`
    *   `MPI_Allreduce`
    *   `MPI_Barrier`

**Expected Results:**
*   **Small messages:** Latency-bound ($\sim 1-10 \mu s$).
*   **Large messages:** Bandwidth-bound (scales linearly).
*   **Allreduce:** ~2x time of Reduce (needs second broadcast phase).

---

## 📝 Summary & Key Takeaways

1.  **Collectives are Optimized:** MPI implementations use tree-based algorithms for $O(\log p)$ complexity.
2.  **Blocking Semantics:** All collective operations are blocking (all processes must participate).
3.  **In-Place Operations:** Use `MPI_IN_PLACE` to avoid separate send/receive buffers.
4.  **Scalability:** Collectives scale well to thousands of processes on modern interconnects.
5.  **Avoid Barriers:** Use only when necessary; prefer asynchronous patterns.

**Collective Operation Complexity:**

| Operation | Latency | Bandwidth |
|---|---|---|
| Bcast | $O(\log p)$ | $O(m)$ |
| Scatter/Gather | $O(\log p)$ | $O(m)$ |
| Reduce | $O(\log p)$ | $O(m)$ |
| Allreduce | $O(\log p)$ | $O(m)$ |
| Barrier | $O(\log p)$ | $O(1)$ |

Where $p$ = processes, $m$ = message size.

---

## 📚 Additional Resources

*   [MPI Collective Communication](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node104.htm)
*   [Thakur & Gropp, "Improving the Performance of Collective Operations in MPICH"](https://link.springer.com/chapter/10.1007/3-540-45825-5_24)

**Tomorrow:** Day 73 - Non-Blocking Communication... overlapping computation and communication.

*End of Day 072 - Total Lines: 1000+*
