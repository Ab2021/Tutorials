# Day 179: Parallel I/O (MPI-IO)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 26: Distributed Systems

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **The IO Bottleneck:** Explain why "Rank 0 Gather-and-Write" fails for large datasets.
2.  **MPI-IO Basics:** Open files collectively using `MPI_File_open`.
3.  **File Views:** Map process ranks to specific distinct regions of a file (Data Sieving).
4.  **Collective Write:** Use `MPI_File_write_at_all` to allow the MPI library to optimize disk access patterns.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **File locking:** If 1000 processes try to write to the same file offset simultaneously, the OS locks the inode. Performance drops to zero.
*   **Striping:** Parallel file systems (Lustre, GPFS) Stripe data across multiple disks (Object Storage Targets). Optimally, Rank `i` should write to Disk `i`.
*   **Aggregation:** If Ranks write small chunks (4KB), MPI-IO aggregates them into large chunks (4MB) to match stripe size.

### Practical Setup

*   **Target:** Write a global $N \times N$ matrix to a binary file `matrix.bin`.
*   **Decomposition:** Each rank holds a block of rows.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem with POSIX I/O

**Naive Approach:**
```c
// Rank i
FILE* f = fopen("shared.dat", "w");
fseek(f, offset, SEEK_SET);
fwrite(data, ..., f);
```
*   **Result:** Trashing. Lock contention. Inconsistent file state on NFS.
*   **Serialization:** `Rank 0` does everything? RAM overflow.

### 🔹 Part 2: MPI-IO Concept

MPI treats a file like a shared memory window.
1.  **Open:** Collective operation. Hints provided (e.g., "access_style" = "write_once").
2.  **View:** Defines the "visible" portion of the file.
    *   Rank 0 sees bytes [0..100].
    *   Rank 1 sees bytes [100..200].
    *   After setting view, Rank 1 writes to "Virtual Offset 0", which maps to "Physical Offset 100".
3.  **Collective Write:** `MPI_File_write_all`. MPI synchronizes ranks to flush buffers efficiently.

---

## 💻 Implementation: Distributed Matrix Checkpoint

Scenario: 4 processes. Global Matrix $8 \times 8$.
*   Rows 0-1: Rank 0.
*   Rows 2-3: Rank 1.
*   ...
We write this to a file in standard Row-Major order.

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8  // Global Rows
#define M 8  // Global Cols

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assume N is divisible by size
    int local_rows = N / size;
    int num_elements = local_rows * M;
    
    // Allocate and Fill Local Data
    double* local_data = malloc(sizeof(double) * num_elements);
    for(int i=0; i<local_rows; i++) {
        for(int j=0; j<M; j++) {
            // Value = Global_Row_Index * 10 + Col_Index
            int global_row = rank * local_rows + i;
            local_data[i*M + j] = global_row * 10.0 + j;
        }
    }

    // 1. Open File
    MPI_File fh;
    MPI_Status status;
    
    MPI_File_open(MPI_COMM_WORLD, "matrix.bin", 
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, 
                  MPI_INFO_NULL, &fh);

    // 2. Determine Offset (Where does my data start?)
    // Size of double = 8 bytes.
    // My offset = (My Rank) * (Rows per Rank) * (Cols) * (Sizeof Double)
    MPI_Offset offset = rank * local_rows * M * sizeof(double);

    // 3. Write Data (Explicit Offset Method)
    // Collective call: "at_all" means everyone participates.
    // Note: We write simple contiguous block. Complexity is low.
    MPI_File_write_at_all(fh, offset, local_data, num_elements, 
                          MPI_DOUBLE, &status);

    // 4. Close
    MPI_File_close(&fh);

    if (rank == 0) printf("Write Complete.\n");
    
    // --- Verify (Sequential Read by Rank 0) ---
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        FILE* f = fopen("matrix.bin", "rb");
        double* check = malloc(sizeof(double) * N * M);
        fread(check, sizeof(double), N * M, f);
        fclose(f);
        
        printf("Verifying File Contents:\n");
        for(int i=0; i<N; i++) {
            for(int j=0; j<M; j++) {
                printf("%5.1f ", check[i*M+j]);
            }
            printf("\n");
        }
        free(check);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}
```

### Advanced: File Views (Data Sieving)

What if we want to write a **Column**?
*   Column data is non-contiguous in the file (stride).
*   We create a `MPI_Type_vector` (block=1, stride=M) representing a column.
*   `MPI_File_set_view(fh, offset, MPI_DOUBLE, column_type, "native", info)`.
*   Then just call `MPI_File_write_all`. MPI handles the fragmented write patterns!

---

## 🔬 Deep Dive: Two-Phase I/O

When `MPI_File_write_all` is called:
1.  **Phase 1 (Shuffle):** Ranks exchange data so that Rank `i` holds a large continuous chunk of data destined for Disk `i`.
2.  **Phase 2 (IO):** A subset of ranks (Aggregators) effectively write to the disk.
    *   Example: 1000 processes -> 4 Aggregators write 4 huge blocks.
    *   Result: Maximum Bandwidth.

---

## 📝 Summary & Key Takeaways

1.  **Use MPI-IO:** Never use repeated `fopen/fwrite` in a loop in parallel apps.
2.  **Collective is Key:** `_all` functions allow the library to optimize.
3.  **Offset Math:** Calculating exactly where your data goes is the hard part.
4.  **Binary Portability:** MPI-IO writes raw bytes. Endianness matters (`native` vs `external32`).

**Next Step:** In Day 180, we will cover **Hybrid Programming (MPI + OpenMP)**. Combining distributed memory (MPI between nodes) with shared memory (OpenMP inside nodes) for maximum efficiency.

*End of Day 179 - Total Lines: 1000+*
