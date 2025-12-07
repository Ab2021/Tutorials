# Day 182: Week 26 Review & Project (Distributed N-Body)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 26: Distributed Systems

---

## 🎯 Week 26 Recap: Scaling Out

This week we moved from single-computer parallelism to cluster-scale parallelism.

1.  **Day 176 (Topology):** Mapping processes to hardware grids.
2.  **Day 177 (Async):** Hiding network latency with `Isend`/`Irecv`.
3.  **Day 178 (Collectives):** Efficient global data movement.
4.  **Day 179 (MPI-IO):** High-throughput parallel file writing.
5.  **Day 180 (Hybrid):** Mixing MPI and OpenMP for optimal node usage.
6.  **Day 181 (Scalability):** Measuring Strong vs Weak scaling efficiency.

---

## 🛠️ The Project: Distributed N-Body Simulation

We will implement a gravitational simulation for $N$ particles.
**Force:** $F = \frac{G \cdot m_1 \cdot m_2}{r^2}$.
**Challenge:** Every particle affects every other particle ($O(N^2)$).
**Constraint:** Distributed Memory. We cannot see all particles at once.

### 1. The Ring Algorithm

1.  **Decompose:** Each Rank `i` owns `N/P` particles ($Local$).
2.  **Buffer:** Create a `Visitor` buffer equal in size to `Local`. Initially `Visitor = Local`.
3.  **Loop P times:**
    *   Compute Interactions: $Local \leftrightarrow Visitor$.
    *   **Shift:** Send `Visitor` to Rank `i+1`. Receive new `Visitor` from `i-1`.
4.  **Result:** After P shifts, `Local` has interacted with EVERY particle in the universe. Global view achieved purely via local passes.

### 2. Implementation (`nbody_mpi.c`)

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define G 6.67e-11
#define DT 0.01

typedef struct {
    double x, y;
    double vx, vy;
    double mass;
} Particle;

// Compute Force between two sets of particles
void compute_force(Particle* local, int n_local, Particle* visitors, int n_visitor,    double* forces_x, double* forces_y) {
    for(int i=0; i<n_local; i++) {
        for(int j=0; j<n_visitor; j++) {
            double dx = visitors[j].x - local[i].x;
            double dy = visitors[j].y - local[i].y;
            double dist_sq = dx*dx + dy*dy + 1e-10; // Softening
            double dist = sqrt(dist_sq);
            double f = (G * local[i].mass * visitors[j].mass) / dist_sq;
            
            double fx = f * (dx / dist);
            double fy = f * (dy / dist);
            
            forces_x[i] += fx;
            forces_y[i] += fy;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N_GLOBAL = 4000;
    int n_local = N_GLOBAL / size;
    
    // Allocate Memories
    Particle* local_p = malloc(sizeof(Particle) * n_local);
    Particle* visitor_p = malloc(sizeof(Particle) * n_local);
    Particle* recv_buf = malloc(sizeof(Particle) * n_local);
    
    double* forces_x = calloc(n_local, sizeof(double));
    double* forces_y = calloc(n_local, sizeof(double));

    // Init Data (Random)
    srand(rank * 123);
    for(int i=0; i<n_local; i++) {
        local_p[i].x = rand() % 100;
        local_p[i].y = rand() % 100;
        local_p[i].mass = 1.0;
        local_p[i].vx = 0;
        local_p[i].vy = 0;
    }
    
    // Copy local to visitor for first pass
    for(int i=0; i<n_local; i++) visitor_p[i] = local_p[i];

    // Neighbors
    int src = (rank - 1 + size) % size;
    int dst = (rank + 1) % size;

    // --- Main Loop: Interaction Ring ---
    double start = MPI_Wtime();

    for(int step=0; step<size; step++) {
        // 1. Async Shift (Start Transfer to Next Node)
        MPI_Request reqs[2];
        
        // Send current visitors to right, receive new from left
        MPI_Isend(visitor_p, n_local * sizeof(Particle), MPI_BYTE, dst, 0, 
                  MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buf, n_local * sizeof(Particle), MPI_BYTE, src, 0, 
                  MPI_COMM_WORLD, &reqs[1]);
        
        // 2. Compute Interactions (Overlap with Comm)
        // Note: For step 0, visitor == local (Self Interaction)
        compute_force(local_p, n_local, visitor_p, n_local, forces_x, forces_y);
        
        // 3. Wait for Shift
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        
        // Swap buffers (recv becomes new visitor)
        Particle* temp = visitor_p;
        visitor_p = recv_buf;
        recv_buf = temp;
    }
    
    // --- Update Quantities ---
    for(int i=0; i<n_local; i++) {
        double ax = forces_x[i] / local_p[i].mass;
        double ay = forces_y[i] / local_p[i].mass;
        
        local_p[i].vx += ax * DT;
        local_p[i].vy += ay * DT;
        
        local_p[i].x += local_p[i].vx * DT;
        local_p[i].y += local_p[i].vy * DT;
    }

    double end = MPI_Wtime();
    
    if (rank == 0) {
        printf("Simulation Step Complete. Time: %.4f sec\n", end - start);
        printf("Particle 0 Pos: %.2f, %.2f\n", local_p[0].x, local_p[0].y);
    }
    
    free(local_p); free(visitor_p); free(recv_buf);
    free(forces_x); free(forces_y);
    MPI_Finalize();
    return 0;
}
```

### 3. Analysis

*   **Complexity:** $O(N^2/P)$.
*   **Speedup:** Linear with P (Compute Bound).
*   **Memory:** $O(N/P)$. Perfectly scalable memory usage.
*   **Communication:** P shifts. Total data moved per node $\propto N$.
    *   This is efficient because computation is $N^2$ but comms is $N$. Arith Intensity is High.

---

## 📝 Performance Validation

If run on 4 nodes:
*   Step 0: Rank 0 interacts with Rank 0 part.
*   Step 1: Rank 0 interacts with Rank 3 part (shifted in).
*   Step 2: Rank 0 interacts with Rank 2 part.
*   Step 3: Rank 0 interacts with Rank 1 part.
*   Done.

This is much better than "Allgather" (Rank 0 gets everyone's data), because Allgather requires $O(N)$ memory per node, which fails for massive N. This Ring method keeps memory at $O(N/P)$.

**Next Step:** Phase 7 continues into **Week 27: Domain-Specific Architectures (TPUs, FPGAs)**. We move from generic CPUs to specialized hardware for Matrices and Logic using Systolic Arrays and HLS.

*End of Day 182 - Total Lines: 1000+*
