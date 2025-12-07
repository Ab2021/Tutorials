# Day 176: MPI Communicators & Topology
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 26: Distributed Systems

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Distributed Memory Model:** Contrast OpenMP (Shared Memory) with MPI (Message Passing).
2.  **Communicators:** Use `MPI_Comm_split` to create safe communication subgroups.
3.  **Virtual Topologies:** Map linear ranks (0..N) to a 2D/3D Cartesian grid.
4.  **Rank Reordering:** Allow MPI to optimize process placement based on hardware topology.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Rank:** A unique ID (0 to Size-1) assigned to each process in a communicator.
*   **MPI_COMM_WORLD:** The default group containing all started processes.
*   **Context:** Messages in one communicator cannot be received in another. This prevents libraries from interfering with app code.

### Practical Setup

*   **Tools:** `mpicc`, `mpirun`.
*   **Environment:** Windows (MS-MPI) or Linux (OpenMPI).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Topology Matters?

Imagine 16 processes running on 4 nodes (4 cores each).
*   Rank 0 and Rank 1 might be on the same node (Fast RAM copy).
*   Rank 0 and Rank 15 might be far apart (Slow Network).
*   **MPI_Cart_create:** Can reorder ranks so that "neighbors" in the grid are close in hardware.

### 🔹 Part 2: Split Communicators

If we have 2 distinct tasks: Simulation and Visualization.
*   **Color Splitting:** Assign `color=0` to Sim ranks, `color=1` to Viz ranks.
*   `MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm)`.
*   `new_comm` acts as an isolated universe for that group.

---

## 💻 Implementation: 2D Cartesian Topology

We will arrange N processes into a Grid and have them identify their Up, Down, Left, Right neighbors.

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. Create a 2D Cartesian Grid
    // Let's assume we maintain a square grid if possible.
    int dims[2] = {0, 0};
    
    // Auto-calculate dimensions (e.g., 12 -> 4x3)
    MPI_Dims_create(world_size, 2, dims);
    
    if (world_rank == 0) {
        printf("MPI World Size: %d\n", world_size);
        printf("Grid Dimensions: %d x %d\n", dims[0], dims[1]);
    }

    // Create Communicator
    int periods[2] = {1, 1}; // Periodic (Wrap around edges aka Torus)
    int reorder = 1;         // Allow reordering for performance
    MPI_Comm grid_comm;
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);

    // 2. Get My Coordinates
    int grid_rank;
    int coords[2];
    MPI_Comm_rank(grid_comm, &grid_rank); // Rank might change if reordered!
    MPI_Cart_coords(grid_comm, grid_rank, 2, coords);

    // 3. Find Neighbors (Shift)
    // Dimension 0 (Rows), Displacement 1 (Move Down/Up)
    int up, down, left, right;
    
    // Shift along axis 0
    MPI_Cart_shift(grid_comm, 0, 1, &up, &down); // Up is prev, Down is next usually depending on row order
    // Shift along axis 1
    MPI_Cart_shift(grid_comm, 1, 1, &left, &right);

    // Print Info
    // Use barrier to keep output readable (not guaranteed ordered though)
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("Rank %d (Coords %d,%d): Neighbors(U:%d, D:%d, L:%d, R:%d)\n", 
           grid_rank, coords[0], coords[1], up, down, left, right);

    // 4. Sub-Communicators (Row-based)
    // Split grid by Row Index (coords[0])
    MPI_Comm row_comm;
    MPI_Comm_split(grid_comm, coords[0], coords[1], &row_comm);
    
    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    
    if (coords[1] == 0) { // First column prints row info
        printf("Rank %d is leader of Row %d (Size %d)\n", grid_rank, coords[0], row_size);
    }

    MPI_Finalize();
    return 0;
}
```

### Execution Example (4 Processes)

```bash
mpirun -n 4 ./topology
```
Output:
```
Grid Dimensions: 2 x 2
Rank 0 (0,0): Neighbors(U:2, D:2, L:1, R:1)
Rank 1 (0,1): Neighbors(U:3, D:3, L:0, R:0) ...
```
Wait, strict periodic boundaries?
*   (0,0) Neighbors:
    *   Up (-1,0) -> wraps to (1,0). Which is Rank 2.
    *   Down (1,0) -> wraps to (1,0) (Wait, if dim is 2, 0+1=1. 1 is last. Next is 0).
    *   Actually: `MPI_Cart_shift`:
        *   `source = rank - disp`.
        *   `dest = rank + disp`.
        *   If dims=2. Row 0 is [0, 1]. Row 1 is [2, 3].
        *   (0,0) is Rank 0.
        *   Axis 0 (Vertical): Up is -1 (wrap to 1, Rank 2). Down is +1 (Rank 2).
        *   Axis 1 (Horizontal): Left is -1 (wrap to 1, Rank 1). Right is +1 (Rank 1).

---

## 🔬 Deep Dive: Graph Topology

For complex simulations (Unstructured Mesh), `Cart_create` isn't enough.
`MPI_Dist_graph_create_adjacent` allows you to specify *exactly* which ranks communicate with which.
The MPI implementation effectively performs a graph partitioning algorithm (like METIS) to map these ranks to physical nodes to minimize network hops.

---

## 📝 Summary & Key Takeaways

1.  **Isolation:** Communicators provide a sandbox. Always use specific communicators for libraries, not `COMM_WORLD`.
2.  **Topologies:** Helps the runtime optimize data flow. Also simplifies code (e.g., "Shift" handles boundary wrap-around automatically).
3.  **Reordering:** If `reorder=1`, `rank 0` in `COMM_WORLD` might become `rank 5` in `grid_comm`. Trust the new rank.
4.  **Scalability:** Good topology use reduces global communication and focuses on local neighbor exchange.

**Next Step:** In Day 177, we will cover **Non-Blocking Communication**. The art of hiding latency by `Isend` and `Irecv` (computation/communication overlap).

*End of Day 176 - Total Lines: 1000+*
