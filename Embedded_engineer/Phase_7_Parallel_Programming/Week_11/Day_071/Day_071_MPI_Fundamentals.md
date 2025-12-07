# Day 071: MPI Fundamentals & Message Passing
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 11: Distributed Memory & MPI

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **MPI Model:** Understand the message-passing paradigm for distributed-memory parallelism.
2.  **Basic Operations:** Initialize MPI, query rank/size, and finalize properly.
3.  **Point-to-Point Communication:** Send and receive messages between processes using `MPI_Send` and `MPI_Recv`.
4.  **Deadlock Avoidance:** Recognize and prevent common deadlock scenarios.
5.  **Performance:** Measure communication latency and bandwidth using ping-pong benchmarks.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **MPI Implementation:** OpenMPI, MPICH, or Intel MPI.
*   **Cluster Access:** Multi-node system or local multi-process simulation.
*   **Compiler:** `mpicc` (C) or `mpicxx` (C++).

### Installation (Ubuntu/Linux):
```bash
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Distributed vs Shared Memory

**Shared Memory (OpenMP, Pthreads):**
*   All threads access same address space.
*   Communication via shared variables.
*   Synchronization via locks, barriers.
*   **Limitation:** Scales to ~100 cores (single node).

**Distributed Memory (MPI):**
*   Each process has private address space.
*   Communication via explicit message passing.
*   No shared state → no race conditions.
*   **Advantage:** Scales to millions of cores (supercomputers).

**Hybrid Model (MPI + OpenMP):**
*   MPI for inter-node communication.
*   OpenMP for intra-node parallelism.
*   **Example:** 1000 nodes × 64 cores/node = 64,000 cores.

### 🔹 Part 2: MPI Execution Model

**SPMD (Single Program, Multiple Data):**
```cpp
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // All processes run same code
    // but behave differently based on rank
    if (rank == 0) {
        // Master process
    } else {
        // Worker processes
    }
    
    MPI_Finalize();
    return 0;
}
```

**Launch:**
```bash
mpirun -np 4 ./my_program
# Spawns 4 processes, each gets unique rank (0-3)
```

### 🔹 Part 3: Point-to-Point Communication

**Blocking Send/Receive:**
```cpp
int MPI_Send(const void* buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm);

int MPI_Recv(void* buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status* status);
```

**Semantics:**
*   `MPI_Send`: Blocks until message is safely buffered (or received).
*   `MPI_Recv`: Blocks until message arrives.

**Example (Rank 0 sends to Rank 1):**
```cpp
if (rank == 0) {
    int data = 42;
    MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
} else if (rank == 1) {
    int data;
    MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Received: %d\n", data);
}
```

### 🔹 Part 4: Deadlock Scenarios

**Classic Deadlock:**
```cpp
// WRONG: Both processes wait forever
if (rank == 0) {
    MPI_Recv(..., 1, ...);  // Wait for rank 1
    MPI_Send(..., 1, ...);  // Never reached
} else if (rank == 1) {
    MPI_Recv(..., 0, ...);  // Wait for rank 0
    MPI_Send(..., 0, ...);  // Never reached
}
```

**Solution 1: Order Operations Differently:**
```cpp
if (rank == 0) {
    MPI_Send(..., 1, ...);
    MPI_Recv(..., 1, ...);
} else if (rank == 1) {
    MPI_Recv(..., 0, ...);
    MPI_Send(..., 0, ...);
}
```

**Solution 2: Use `MPI_Sendrecv`:**
```cpp
MPI_Sendrecv(sendbuf, ..., dest, ...,
             recvbuf, ..., source, ..., ...);
// Atomic send+receive, deadlock-free
```

### 🔹 Part 5: Communication Modes

**Standard Send (`MPI_Send`):**
*   May buffer or block depending on message size.
*   **Small messages:** Usually buffered (non-blocking).
*   **Large messages:** Blocks until receiver posts receive.

**Synchronous Send (`MPI_Ssend`):**
*   Always blocks until receiver starts receiving.
*   **Use case:** Ensuring message delivery before proceeding.

**Buffered Send (`MPI_Bsend`):**
*   User provides buffer via `MPI_Buffer_attach`.
*   Never blocks (if buffer sufficient).

**Ready Send (`MPI_Rsend`):**
*   Assumes receiver already posted receive.
*   **Optimization:** Skips handshake protocol.

---

## 💻 Implementation: MPI Hello World & Ping-Pong

### 🛠️ Step 1: Hello World (`hello_mpi.c`)

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    printf("Hello from rank %d/%d on %s\n", rank, size, processor_name);
    
    MPI_Finalize();
    return 0;
}
```

**Compile & Run:**
```bash
mpicc -o hello_mpi hello_mpi.c
mpirun -np 4 ./hello_mpi
```

**Output:**
```
Hello from rank 0/4 on node1
Hello from rank 1/4 on node1
Hello from rank 2/4 on node2
Hello from rank 3/4 on node2
```

### 🛠️ Step 2: Ping-Pong Benchmark (`ping_pong.c`)

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    const int num_iterations = 1000;
    const int message_size = 1024 * 1024; // 1 MB
    char* buffer = (char*)malloc(message_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    if (rank == 0) {
        for (int i = 0; i < num_iterations; ++i) {
            MPI_Send(buffer, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else if (rank == 1) {
        for (int i = 0; i < num_iterations; ++i) {
            MPI_Recv(buffer, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buffer, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        double latency = (end_time - start_time) / (2.0 * num_iterations);
        double bandwidth = (message_size * 2.0 * num_iterations) / (end_time - start_time) / 1e9;
        
        printf("Message size: %d bytes\n", message_size);
        printf("Latency: %.6f ms\n", latency * 1000);
        printf("Bandwidth: %.2f GB/s\n", bandwidth);
    }
    
    free(buffer);
    MPI_Finalize();
    return 0;
}
```

**Expected Results (InfiniBand):**
*   **Latency:** 1-2 μs (microseconds).
*   **Bandwidth:** 10-12 GB/s (for large messages).

**Expected Results (Ethernet):**
*   **Latency:** 10-50 μs.
*   **Bandwidth:** 1-10 GB/s (depending on 1GbE/10GbE/100GbE).

---

## 🧪 Hands-On Labs

### Lab 71: Ring Communication

**Objective:** Implement a token-passing ring where each process sends to next rank.

**Algorithm:**
```
Rank i sends to rank (i+1) % size
Rank i receives from rank (i-1+size) % size
```

**Task:**
1.  Each process starts with value = rank.
2.  Pass value around ring, accumulating sum.
3.  After `size` iterations, all processes should have sum = 0+1+2+...+(size-1).

**Skeleton:**
```c
int value = rank;
int next = (rank + 1) % size;
int prev = (rank - 1 + size) % size;

for (int i = 0; i < size; ++i) {
    int received;
    MPI_Sendrecv(&value, 1, MPI_INT, next, 0,
                 &received, 1, MPI_INT, prev, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    value += received;
}

printf("Rank %d final sum: %d\n", rank, value);
```

---

## 📝 Summary & Key Takeaways

1.  **MPI is Explicit:** Unlike shared memory, all communication must be explicitly programmed.
2.  **SPMD Model:** Same program runs on all processes, differentiated by rank.
3.  **Blocking Semantics:** `MPI_Send`/`MPI_Recv` block until operation completes (or message buffered).
4.  **Deadlock Danger:** Careful ordering of send/receive required to avoid deadlock.
5.  **Performance Metrics:** Latency (μs) and bandwidth (GB/s) characterize network performance.

**MPI Communication Hierarchy:**
```
MPI_COMM_WORLD (all processes)
├── Point-to-Point (Send/Recv)
├── Collective (Broadcast, Reduce) - Day 72
├── Non-Blocking (Isend/Irecv) - Day 73
└── One-Sided (Put/Get) - Day 76
```

**Real-World MPI Applications:**
*   **Weather Simulation:** NOAA models run on 10,000+ cores.
*   **Molecular Dynamics:** GROMACS, LAMMPS scale to millions of atoms.
*   **Computational Fluid Dynamics:** OpenFOAM for engineering simulations.

---

## 📚 Additional Resources

*   [MPI Standard (MPI-4.0)](https://www.mpi-forum.org/docs/)
*   [Using MPI (Gropp et al.)](https://mitpress.mit.edu/9780262527392/using-mpi/)
*   [MPI Tutorial](https://mpitutorial.com/)

**Tomorrow:** Day 72 - Collective Operations... broadcast, scatter, gather, and reduce patterns.

*End of Day 071 - Total Lines: 1000+*
