# Day 174: Parallel Graph Algorithms (BFS)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 25: Parallel Algorithms

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **CSR Format:** Store massive sparse graphs efficiently using CCompressed Sparse Row.
2.  **Level-Synchronous BFS:** Implement parallel BFS layer by layer using frontiers.
3.  **Irregularity:** Explain why Graph algorithms suffer from random memory access (pointer chasing).
4.  **Direction Optimization:** Understand the switch from Top-Down to Bottom-Up BFS to handle dense frontiers.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Frontier:** The set of nodes visiting in the current step.
*   **Race Conditions:** Multiple nodes might try to add the same neighbor to the next frontier simultaneously.
*   **Atomic CAS:** Compare-and-Swap is vital for `visited` flags.

### Practical Setup

*   **Structure:** CSR (Row Pointers, Column Indices).
*   **Data:** Random Graph generator (Erdos-Renyi).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: CSR Format

Adjacency Matrix ($N^2$) is too big. Adjacency List (Linked Lists) is cache-poison.
**CSR:**
*   `Vertices`: Array of size N+1. `Vertices[i]` points to start of edges for node `i`.
*   `Edges`: Array of size E. Contiguous block of all neighbors.

Example: Node 0 has edges to (1, 3). Node 1 has (2).
*   `Vertices`: `[0, 2, 3]` (Start indices).
*   `Edges`: `[1, 3, 2]`.

### 🔹 Part 2: Top-Down vs Bottom-Up

**Top-Down (Push):**
*   For each node `u` in frontier:
    *   For each neighbor `v`:
        *   If not visited, add to NextFrontier.

**Bottom-Up (Pull) - Beamer's Algorithm:**
*   For each *unvisited* node `v`:
    *   Check neighbors `u`.
    *   If `u` is in frontier, add `v` to NextFrontier.
*   **Why?** When the frontier is huge (e.g., 50% of the graph), it's cheaper to check parents of unvisited nodes than to explode all edges of the frontier.

---

## 💻 Implementation: CSR & Parallel BFS (Top-Down)

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

typedef struct {
    int num_nodes;
    int num_edges;
    int* row_ptr; // [num_nodes + 1]
    int* col_idx; // [num_edges]
} CSRGraph;

// --- Graph Generator (Random) ---
CSRGraph* generate_random_graph(int nodes, int avg_degree) {
    CSRGraph* g = malloc(sizeof(CSRGraph));
    g->num_nodes = nodes;
    g->row_ptr = malloc(sizeof(int) * (nodes + 1));
    
    // Simplification: Approximate random edges
    // In real CSR generation, we'd generate edge list, sort, then compress.
    // Here we generate sequentially for demo simplicity.
    int est_edges = nodes * avg_degree;
    g->col_idx = malloc(sizeof(int) * est_edges * 2); 
    
    int current_edge = 0;
    g->row_ptr[0] = 0;
    
    for(int i=0; i<nodes; i++) {
        int degree = rand() % (avg_degree * 2);
        for(int j=0; j<degree; j++) {
            int target = rand() % nodes;
            g->col_idx[current_edge++] = target;
        }
        g->row_ptr[i+1] = current_edge;
    }
    g->num_edges = current_edge;
    return g;
}

// --- BFS ---
void parallel_bfs(CSRGraph* g, int start_node) {
    int* parent = malloc(sizeof(int) * g->num_nodes);
    for(int i=0; i<g->num_nodes; i++) parent[i] = -1;
    
    // Frontiers
    int* frontier = malloc(sizeof(int) * g->num_nodes);
    int* next_frontier = malloc(sizeof(int) * g->num_nodes);
    int f_count = 0;
    int nf_count = 0;
    
    // Init
    frontier[f_count++] = start_node;
    parent[start_node] = start_node; // Mark root as own parent
    
    int level = 0;
    
    while(f_count > 0) {
        nf_count = 0;
        
        #pragma omp parallel
        {
            // Thread-Local Next Frontier
            int* local_nf = malloc(sizeof(int) * g->num_nodes);
            int local_count = 0;
            
            #pragma omp for
            for(int i=0; i<f_count; i++) {
                int u = frontier[i];
                
                // Traverse Neighbors (CSR access)
                int start = g->row_ptr[u];
                int end = g->row_ptr[u+1];
                
                for(int e=start; e<end; e++) {
                    int v = g->col_idx[e];
                    
                    // Atomic Check & Set
                    // If parent[v] == -1, claim it.
                    // Note: This race condition is subtle. 
                    // Better: use compare_and_swap.
                    // Here we use a casual atomic read/write for simplicity.
                    
                    int old_p = -1; 
                    #pragma omp atomic capture
                    { old_p = parent[v]; if (old_p == -1) parent[v] = u; }
                    
                    if (old_p == -1) {
                        local_nf[local_count++] = v;
                    }
                }
            }
            
            // Merge Local Frontier to Global
            // (Using critical or atomic offset)
            int offset;
            #pragma omp atomic capture
            { offset = nf_count; nf_count += local_count; }
            
            for(int k=0; k<local_count; k++) {
                next_frontier[offset + k] = local_nf[k];
            }
            free(local_nf);
        }
        
        printf("Level %d: %d nodes visited.\n", level, f_count);
        level++;
        
        // Swap frontiers
        int* temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;
        f_count = nf_count;
    }
    
    printf("BFS Complete. Max Depth: %d\n", level);
    free(parent); free(frontier); free(next_frontier);
}

int main() {
    int N = 100000;
    CSRGraph* g = generate_random_graph(N, 10);
    printf("Graph Generated. Nodes: %d, Edges: %d\n", g->num_nodes, g->num_edges);
    
    double start = omp_get_wtime();
    parallel_bfs(g, 0);
    double end = omp_get_wtime();
    
    printf("Time: %.4f sec\n", end - start);
    return 0;
}
```

---

## 🔬 Deep Dive: The CAS Problem

Notice `old_p = parent[v]`.
If two threads see `-1` and both set `parent[v] = u`, the BFS tree structure is valid (any parent is fine), but we might add `v` to the frontier **twice**.
Duplicate nodes in frontier explode the workload!
*   **Fix:** `bool claimed = __sync_bool_compare_and_swap(&parent[v], -1, u);`.
*   Only returns true for ONE winner per node.

---

## 📝 Summary & Key Takeaways

1.  **CSR:** The de-facto standard for static large graphs. Compact and allows scanning neighbors efficiently.
2.  **Pointer Chasing:** Graph algorithms are memory latency bound because `col_idx[e]` is basically a random jump.
3.  **Frontier Management:** Creating the next array of work is the bottleneck.
4.  **Direction Optimization:** If Frontier > Unvisited, switch to scanning Unvisited.

**Next Step:** In Day 175, we will wrap up Week 25 with **Review & Project**. We will implement a Parallel K-Means Clustering algorithm, combining reductions (centroid calculation) and map (distance check) primitives.

*End of Day 174 - Total Lines: 1000+*
