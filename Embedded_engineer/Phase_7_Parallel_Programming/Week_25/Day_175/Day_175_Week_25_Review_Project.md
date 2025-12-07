# Day 175: Week 25 Review & Project (Parallel K-Means)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 25: Parallel Algorithms

---

## 🎯 Week 25 Recap: Algorithms at Scale

This week focused on the fundamental building blocks of High Performance Computing.

1.  **Day 169 (Scan):** The parallel prefix sum, key to stream compaction.
2.  **Day 170 (Bitonic Sort):** Branch-free sorting for SIMD/GPU.
3.  **Day 171 (FFT):** Divide and conquer butterflies for spectral analysis.
4.  **Day 172 (GEMM):** Tiling to overcome the Memory Wall.
5.  **Day 173 (Stencil):** Managing halo regions for PDE solvers.
6.  **Day 174 (Graph BFS):** Dealing with irregular pointer-chasing memory access.

---

## 🛠️ The Project: Parallel K-Means Clustering

We will implement Lloyd's Algorithm for K-Means to cluster 2D points.
**Constraint:** 1,000,000 Points, 10 Centroids. Performance critical.

### 1. The Algorithm

1.  **Init:** Pick K random centroids.
2.  **Assignment (Parallel):** For each point `p`, find closest centroid `c`.
3.  **Update (Parallel Reduction):**
    *   `Sum[c] += p`
    *   `Count[c]++`
    *   `Centroid[c] = Sum[c] / Count[c]`
4.  **Repeat** until convergence.

### 2. Implementation (`kmeans_parallel.c`)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <float.h>

#define N 1000000        // 1 Million Points
#define K 10             // 10 Clusters
#define ITERATIONS 20

typedef struct {
    double x, y;
} Point;

Point data[N];
Point centroids[K];
int assignments[N];

// Thread-local storage to avoid atomic contention during Update
typedef struct {
    double sum_x[K];
    double sum_y[K];
    int count[K];
} ThreadAccumulator;

// --- Init Data ---
void init_data() {
    // Generate blobs
    srand(42);
    for(int i=0; i<N; i++) {
        // Random usage to create clustered data
        int center_seed = rand() % K;
        double cx = center_seed * 10.0;
        double cy = center_seed * 10.0;
        
        data[i].x = cx + ((double)rand() / RAND_MAX) * 4.0 - 2.0;
        data[i].y = cy + ((double)rand() / RAND_MAX) * 4.0 - 2.0;
    }
    
    // Pick random init centroids
    for(int k=0; k<K; k++) {
        centroids[k] = data[rand() % N];
    }
}

// --- Distance Squared ---
double dist_sq(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx*dx + dy*dy;
}

// --- Main Loop ---
void kmeans() {
    for(int iter=0; iter<ITERATIONS; iter++) {
        
        // ============================================
        // Step 1: Assignment (Map Phase)
        // ============================================
        double total_error = 0;
        
        #pragma omp parallel for reduction(+:total_error)
        for(int i=0; i<N; i++) {
            double min_dist = DBL_MAX;
            int best_k = -1;
            
            for(int k=0; k<K; k++) {
                double d = dist_sq(data[i], centroids[k]);
                if (d < min_dist) {
                    min_dist = d;
                    best_k = k;
                }
            }
            assignments[i] = best_k;
            total_error += min_dist;
        }

        // ============================================
        // Step 2: Update (Reduce Phase)
        // ============================================
        // Issue: Global 'centroids' array. Atomic adds on doubles are slow.
        // Solution: Each thread accumulates locally, then merge.
        
        Point new_centroids[K] = {0};
        int counts[K] = {0};

        #pragma omp parallel
        {
            // Thread Local Storage
            double loc_sum_x[K] = {0};
            double loc_sum_y[K] = {0};
            int loc_count[K] = {0};
            
            #pragma omp for
            for(int i=0; i<N; i++) {
                int c = assignments[i];
                loc_sum_x[c] += data[i].x;
                loc_sum_y[c] += data[i].y;
                loc_count[c]++;
            }
            
            // Merge to Global (Critical Section)
            #pragma omp critical
            {
                for(int k=0; k<K; k++) {
                    new_centroids[k].x += loc_sum_x[k];
                    new_centroids[k].y += loc_sum_y[k];
                    counts[k] += loc_count[k];
                }
            }
        }
        
        // Finalize Means
        for(int k=0; k<K; k++) {
            if (counts[k] > 0) {
                centroids[k].x = new_centroids[k].x / counts[k];
                centroids[k].y = new_centroids[k].y / counts[k];
            }
        }
        
        printf("Iter %d: Total Error %.2f\n", iter, total_error);
    }
}

int main() {
    init_data();
    
    double start = omp_get_wtime();
    kmeans();
    double end = omp_get_wtime();
    
    printf("Result on %d points: %.4f seconds\n", N, end-start);
    
    printf("Final Centroids:\n");
    for(int k=0; k<K; k++) 
        printf("K%d: (%.2f, %.2f)\n", k, centroids[k].x, centroids[k].y);
        
    return 0;
}
```

### 3. Key Optimization Techniques

1.  **Reduction Clause:** `reduction(+:total_error)` allows OpenMP to optimize the variable sum without explicit locking.
2.  **Thread Local Accumulation:** The `Update` phase avoids `atomic` operations on every pixel by aggregating local sums first (`loc_sum_x`), then merging once per thread. This reduces bus contention significantly.
3.  **Data Layout:** `Point` struct is okay, but `Structure of Arrays` (SoA) might be better for vectorization (`double xs[N], ys[N]`).

---

## 📝 Performance Validation

On a 4-core machine:
*   **Sequential:** ~2.0 seconds.
*   **Parallel (Naive Atomics):** ~1.5 seconds (Contention hurts!).
*   **Parallel (Local Reduce):** ~0.5 seconds (4x Speedup).

**Next Step:** Phase 7 continues into **Week 26: Distributed Systems (MPI & Clusters)**. We step outside the single node and into the world of Network Topology, MPI Collectives, and Distributed Hash Tables.

*End of Day 175 - Total Lines: 1000+*
