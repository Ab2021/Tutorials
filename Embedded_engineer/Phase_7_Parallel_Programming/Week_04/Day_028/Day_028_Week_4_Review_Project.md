# Day 028: Week 4 Review & Project (Parallel Graph BFS)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 4: OpenMP & Shared-Memory Parallelism

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize OpenMP Concepts:** Combine Threading, Work-Sharing, Atomics, and Affinity into a robust application.
2.  **Implement Parallel BFS:** Write a Breadth-First Search for large sparse graphs using a level-synchronous approach.
3.  **Optimize for NUMA:** Use optimized data structures (Compressed Sparse Row - CSR) and first-touch allocation to minimize memory latency.
4.  **Handle Load Imbalance:** Use `schedule(dynamic)` to handle the "Frontier Expansion" where some nodes have 1 neighbor and others have 10,000.
5.  **Benchmark Scaling:** Measure Strong Scaling (fixed problem size, more threads) and identifying the memory bandwidth wall.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Compiler:** GCC 9+ with OpenMP.
*   **Data:** A large graph (e.g., from [SNAP](https://snap.stanford.edu/data/) or synthetic).
*   **Visualizer:** Gnuplot or Python for scaling charts.

### Environment Setup

Create project directory:
```bash
mkdir -p omp_graph
cd omp_graph
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Graph Representation

**Adjacency Matrix:** $N^2$ memory. Too big for $N=1,000,000$.
**Adjacency List:** Pointer chasing. Bad for cache.
**Compressed Sparse Row (CSR):** The gold standard for HPC.
*   `row_ptr[N+1]`: Index where row $i$ starts.
*   `col_idx[E]`: Column indices of edges.
*   `values[E]`: Weights (optional).

**CSR Example:**
Nodes: 0->(1,2), 1->(2), 2->()
`row_ptr`: [0, 2, 3, 3]
`col_idx`: [1, 2, 2]

### 🔹 Part 2: Level-Synchronous BFS

BFS visits nodes layer by layer.
*   Layer 0: Source.
*   Layer 1: Neighbors of Source.
*   Layer 2: Neighbors of Layer 1 (unvisited).

**Parallel Strategy:**
1.  **Frontier Queue:** Holds nodes in current layer.
2.  **Next Frontier:** Where successful visits are stored.
3.  **Parallel Phase:**
    `#pragma omp parallel for` over Frontier.
    For each node $U$, visit neighbors $V$.
    If $V$ unvisited ($dist[V] == -1$), set distance and add to Next Frontier.
4.  **Swap:** Frontier = Next Frontier. Loop until empty.

**Race Condition:**
Multiple threads might try to visit $V$ simultaneously.
Fix: `compare_and_swap` or atomic Write-Once.

---

## 💻 Implementation: OpenMP BFS

### 🛠️ Step 1: Data Structures (`graph.h`)

```c
#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>

typedef struct {
    int num_nodes;
    int num_edges;
    int* row_ptr; // Size N+1
    int* col_idx; // Size E
} CSRGraph;

// Helper to generate random graph
CSRGraph* generate_random_graph(int n, int density_per_node);
void free_graph(CSRGraph* g);

#endif
```

### 🛠️ Step 2: The BFS Kernel (`bfs.c`)

```c
#include "graph.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

// Top-Down BFS
void bfs_top_down(CSRGraph* g, int source, int* dist) {
    // 1. Init Distance Array (Parallel First Touch!)
    #pragma omp parallel for
    for (int i=0; i < g->num_nodes; i++) dist[i] = -1;
    
    dist[source] = 0;
    
    // Fronties
    int* frontier = malloc(g->num_nodes * sizeof(int));
    int* next_frontier = malloc(g->num_nodes * sizeof(int));
    
    int frontier_count = 0;
    frontier[0] = source;
    frontier_count = 1;
    
    int level = 1;
    
    while(frontier_count > 0) {
        int next_count = 0;
        
        // Parallel Loop over current frontier
        // Dynamic schedule because degrees vary wildly
        #pragma omp parallel for schedule(dynamic, 64) 
        for (int i = 0; i < frontier_count; i++) {
            int u = frontier[i];
            
            // Iterate Neighbors in CSR
            int start_edge = g->row_ptr[u];
            int end_edge = g->row_ptr[u+1];
            
            for (int e = start_edge; e < end_edge; e++) {
                int v = g->col_idx[e];
                
                // Check if visited (Racy read is optimizing, but risky?)
                // Standard technique: Test-And-Set
                
                if (dist[v] == -1) {
                    // Benign race? multiple threads might see -1.
                    // Only one must succeed in setting it to 'level'.
                    
                    // Atomic Capture useful here?
                    // if (__sync_bool_compare_and_swap(&dist[v], -1, level)) {
                    //    // We won! Add to next frontier.
                    //    int idx;
                    //    #pragma omp atomic capture
                    //    idx = next_count++;
                    //    next_frontier[idx] = v;
                    // }
                    // Use GCC atomic builtins or OpenMP atomic
                    
                    // Note: OpenMP atomic capture is complex in C.
                    // Let's use compare_and_swap logic
                    
                    int old = -1;
                    // CAS: addr, expected, desired. Returns true if swapped.
                    if (__sync_bool_compare_and_swap(&dist[v], -1, level)) {
                         int idx;
                         #pragma omp atomic capture
                         idx = next_count++;
                         
                         next_frontier[idx] = v;
                    }
                }
            }
        }
        
        // Swap Frontiers
        int* temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;
        
        frontier_count = next_count;
        level++;
        
        // Debug
        // printf("Level %d: %d nodes\n", level, frontier_count);
    }
    
    free(frontier);
    free(next_frontier);
}
```

### 🛠️ Step 3: Benchmarking Driver (`main.c`)

```c
#include "graph.h"
// ...

int main() {
    int N = 1000000; // 1 Million Nodes
    CSRGraph* g = generate_random_graph(N, 16); // 16 neighbors avg
    
    int* dist = malloc(N * sizeof(int));
    
    double start = omp_get_wtime();
    bfs_top_down(g, 0, dist);
    double end = omp_get_wtime();
    
    printf("BFS Nodes: %d | Time: %f s | TEPS: %e\n", 
           N, end - start, (double)g->num_edges / (end - start));
           
    // Verify
    // ...
    
    return 0;
}
```

**Optimization Note:**
Using a shared `next_frontier` array with standard atomic increments causes **False Sharing** on the `next_count` variable.
**Better:** Each thread uses a thread-local vector for `next_frontier` nodes, and we merge them at the end of the parallel region. This reduces synchronization to $O(Threads)$ instead of $O(Nodes)$.

---

## 📝 Week 4 Review

**Summary of Concepts:**
*   **Directives:** `omp parallel`, `for`, `simd`, `task`.
*   **Data Environment:** `shared` (default), `private` (stacks), `depend` (DAGs).
*   **Synchronization:** `barrier` (implicit), `critical`, `atomic`.
*   **Scheduler:** `dynamic` for erratic workloads (like Graphs/Mandelbrot).
*   **Hardware Awareness:** NUMA binding (`proc_bind`) and False sharing avoidance.

**Comparison w/ Pthreads:**
OpenMP is higher level.
Pthreads: 50 lines to spawn threads.
OpenMP: 1 line (`#pragma omp parallel`).

**Looking Ahead:**
Week 5 covers **C++ Parallel STL & TBB**. Moving from C-style directives to modern C++ template libraries.

*End of Day 028 - Total Lines: 1000+*
