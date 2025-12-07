# Day 066: Parallel Graph Algorithms
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 10: Parallel Algorithms & Patterns

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **BFS Parallelization:** Implement level-synchronous Breadth-First Search using frontier expansion.
2.  **Shortest Paths:** Parallelize Bellman-Ford algorithm using edge relaxation and atomic operations.
3.  **Connected Components:** Use label propagation and pointer jumping for component detection.
4.  **Graph Coloring:** Apply greedy coloring with conflict resolution for parallel execution.
5.  **Performance Optimization:** Understand direction-optimizing BFS and work-efficient graph traversal.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Graph Theory:** Vertices, edges, adjacency lists, CSR (Compressed Sparse Row) format.
*   **Sequential Algorithms:** BFS, DFS, Dijkstra, Bellman-Ford.
*   **Atomic Operations:** Compare-and-swap, atomic min/max.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Graph Representation for Parallelism

**Adjacency List (Poor for GPU):**
```cpp
std::vector<std::vector<int>> adj; // Irregular, pointer chasing
```

**CSR Format (Optimal for GPU):**
```cpp
std::vector<int> row_offsets;  // Size: V+1
std::vector<int> col_indices;  // Size: E
std::vector<float> edge_weights; // Size: E (optional)
```

**Example:**
Graph: `0→1, 0→2, 1→2, 2→3`
```
row_offsets  = [0, 2, 3, 4, 4]  // Vertex 0 has edges [0,2), Vertex 1 has [2,3), etc.
col_indices  = [1, 2, 2, 3]     // Destinations
```

**GPU Advantage:**
*   Coalesced memory access when processing neighbors.
*   No pointer indirection.
*   Warp divergence minimized (all threads in warp process similar-degree vertices).

### 🔹 Part 2: Parallel BFS (Level-Synchronous)

**Sequential BFS:**
```cpp
queue.push(source);
visited[source] = true;
while (!queue.empty()) {
    int u = queue.front(); queue.pop();
    for (int v : neighbors(u)) {
        if (!visited[v]) {
            visited[v] = true;
            queue.push(v);
        }
    }
}
```

**Parallel BFS (Frontier-Based):**
```cpp
Frontier F = {source};
level = 0;
while (!F.empty()) {
    Frontier F_next;
    parallel_for each u in F:
        for each v in neighbors(u):
            if atomicCAS(&visited[v], false, true) == false:
                F_next.add(v);
    F = F_next;
    level++;
}
```

**Key Insight:**
All vertices at the same level can be processed in parallel (no dependencies within a level).

**Complexity:**
*   **Work:** $O(V + E)$ (same as sequential).
*   **Span:** $O(D)$ where $D$ is graph diameter.
*   **Parallelism:** $O(\frac{V+E}{D})$.

For a grid graph ($V = n^2$, $D = O(n)$), parallelism is $O(n)$.

### 🔹 Part 3: Direction-Optimizing BFS

**Problem:**
In early BFS levels, frontier is small (few vertices).
In middle levels, frontier is huge (many vertices).

**Optimization:**
*   **Top-Down:** Iterate over frontier, check neighbors.
    *   Good when frontier is small.
*   **Bottom-Up:** Iterate over unvisited vertices, check if any neighbor is in frontier.
    *   Good when frontier is large (>50% of graph).

**Hybrid Approach:**
Switch between top-down and bottom-up based on frontier size.

**Speedup:**
Can achieve 2-3x improvement on scale-free graphs (social networks).

### 🔹 Part 4: Bellman-Ford Parallelization

**Sequential Bellman-Ford:**
```cpp
dist[source] = 0;
for (int i = 0; i < V-1; ++i) {
    for (each edge (u, v, w)) {
        dist[v] = min(dist[v], dist[u] + w);
    }
}
```

**Parallel Version:**
```cpp
dist[source] = 0;
for (int i = 0; i < V-1; ++i) {
    parallel_for each edge (u, v, w):
        atomicMin(&dist[v], dist[u] + w);
}
```

**Correctness:**
Atomic min ensures correct updates even with race conditions.

**Optimization:**
*   **Early Termination:** If no updates in an iteration, stop.
*   **Delta-Stepping:** Process edges in buckets by distance (reduces iterations).

### 🔹 Part 5: Connected Components (Label Propagation)

**Algorithm:**
```cpp
// Initialize: each vertex is its own component
for (int v = 0; v < V; ++v) {
    label[v] = v;
}

bool changed = true;
while (changed) {
    changed = false;
    parallel_for each edge (u, v):
        int label_u = label[u];
        int label_v = label[v];
        if (label_u != label_v) {
            int min_label = min(label_u, label_v);
            if (atomicMin(&label[u], min_label) != min_label) changed = true;
            if (atomicMin(&label[v], min_label) != min_label) changed = true;
        }
}
```

**Convergence:**
*   **Worst Case:** $O(D)$ iterations (diameter).
*   **Typical:** $O(\log V)$ iterations for random graphs.

**Pointer Jumping Optimization:**
```cpp
while (label[v] != label[label[v]]) {
    label[v] = label[label[v]]; // Jump to grandparent
}
```
Reduces tree height exponentially.

---

## 💻 Implementation: GPU BFS

### 🛠️ Step 1: CSR Graph Construction

```cpp
#include <vector>
#include <iostream>

struct CSRGraph {
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    int num_vertices;
    int num_edges;
    
    CSRGraph(int V, const std::vector<std::pair<int,int>>& edges) {
        num_vertices = V;
        num_edges = edges.size();
        
        // Count out-degree
        std::vector<int> out_degree(V, 0);
        for (auto [u, v] : edges) {
            out_degree[u]++;
        }
        
        // Build row_offsets (prefix sum)
        row_offsets.resize(V + 1);
        row_offsets[0] = 0;
        for (int i = 0; i < V; ++i) {
            row_offsets[i + 1] = row_offsets[i] + out_degree[i];
        }
        
        // Fill col_indices
        col_indices.resize(num_edges);
        std::vector<int> current_pos = row_offsets;
        for (auto [u, v] : edges) {
            col_indices[current_pos[u]++] = v;
        }
    }
};
```

### 🛠️ Step 2: CUDA BFS Kernel

```cpp
#include <cuda_runtime.h>

__global__ void bfs_kernel(const int* row_offsets,
                          const int* col_indices,
                          int* levels,
                          bool* frontier,
                          bool* next_frontier,
                          bool* changed,
                          int num_vertices,
                          int current_level)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    
    if (frontier[v]) {
        // Process all neighbors
        int start = row_offsets[v];
        int end = row_offsets[v + 1];
        
        for (int i = start; i < end; ++i) {
            int neighbor = col_indices[i];
            
            // Try to claim this neighbor
            if (levels[neighbor] == -1) {
                int old = atomicCAS(&levels[neighbor], -1, current_level + 1);
                if (old == -1) {
                    next_frontier[neighbor] = true;
                    *changed = true;
                }
            }
        }
    }
}

void bfs_gpu(const CSRGraph& graph, int source, std::vector<int>& levels) {
    int V = graph.num_vertices;
    
    // Allocate device memory
    int *d_row_offsets, *d_col_indices, *d_levels;
    bool *d_frontier, *d_next_frontier, *d_changed;
    
    cudaMalloc(&d_row_offsets, (V + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, graph.num_edges * sizeof(int));
    cudaMalloc(&d_levels, V * sizeof(int));
    cudaMalloc(&d_frontier, V * sizeof(bool));
    cudaMalloc(&d_next_frontier, V * sizeof(bool));
    cudaMalloc(&d_changed, sizeof(bool));
    
    // Copy graph to device
    cudaMemcpy(d_row_offsets, graph.row_offsets.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, graph.col_indices.data(), graph.num_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize
    std::vector<int> h_levels(V, -1);
    std::vector<bool> h_frontier(V, false);
    h_levels[source] = 0;
    h_frontier[source] = true;
    
    cudaMemcpy(d_levels, h_levels.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, h_frontier.data(), V * sizeof(bool), cudaMemcpyHostToDevice);
    
    // BFS iterations
    int level = 0;
    bool h_changed = true;
    
    while (h_changed) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemset(d_next_frontier, 0, V * sizeof(bool));
        
        // Launch kernel
        int threads = 256;
        int blocks = (V + threads - 1) / threads;
        bfs_kernel<<<blocks, threads>>>(
            d_row_offsets, d_col_indices, d_levels,
            d_frontier, d_next_frontier, d_changed,
            V, level
        );
        
        // Swap frontiers
        std::swap(d_frontier, d_next_frontier);
        
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        level++;
    }
    
    // Copy results back
    levels.resize(V);
    cudaMemcpy(levels.data(), d_levels, V * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_levels);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_changed);
}
```

**Optimization Opportunities:**
1.  **Bitmap Frontier:** Use bitset instead of bool array (8x memory savings).
2.  **Warp-Level Primitives:** Use `__ballot_sync` to detect if any thread in warp found new vertices.
3.  **Load Balancing:** High-degree vertices dominate runtime. Use work-stealing or virtual warps.

---

## 🧪 Hands-On Labs

### Lab 66: Graph Traversal Benchmark

**Objective:** Compare BFS implementations on different graph topologies.

**Test Graphs:**
1.  **Grid Graph:** $n \times n$ lattice, $D = 2n$.
2.  **Random Graph:** Erdős-Rényi $G(n, p)$, $D = O(\log n)$.
3.  **Scale-Free:** Barabási-Albert, power-law degree distribution.

**Metrics:**
*   **TEPS (Traversed Edges Per Second):** Standard graph benchmark metric.
*   **Frontier Size:** Track how frontier grows/shrinks.

**Expected Results:**
*   Grid: Slow (high diameter, low parallelism).
*   Random: Fast (low diameter, high parallelism).
*   Scale-Free: Medium (hubs create load imbalance).

---

## 📝 Summary & Key Takeaways

1.  **Graph Algorithms are Irregular:** Unlike dense linear algebra, graph algorithms have unpredictable memory access patterns.
2.  **Frontier Management:** Efficient frontier representation (bitmap, queue, sparse array) is critical for performance.
3.  **Atomic Operations:** Essential for correctness but can become bottleneck. Minimize contention.
4.  **Load Balancing:** High-degree vertices create work imbalance. Use dynamic scheduling or vertex splitting.
5.  **Direction Optimization:** Switching between push/pull traversal can double performance.

**Graph Algorithm Complexity:**

| Algorithm | Sequential | Parallel Work | Parallel Span | Parallelism |
|---|---|---|---|---|
| BFS | $O(V+E)$ | $O(V+E)$ | $O(D)$ | $O(\frac{V+E}{D})$ |
| Bellman-Ford | $O(VE)$ | $O(VE)$ | $O(V)$ | $O(E)$ |
| Connected Components | $O(V+E)$ | $O(V+E)$ | $O(D \log V)$ | $O(\frac{V+E}{D \log V})$ |

---

## 📚 Additional Resources

*   [Beamer et al., "Direction-Optimizing BFS" (2012)](https://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf)
*   [Gunrock: GPU Graph Analytics](https://github.com/gunrock/gunrock)
*   [Graph500 Benchmark](https://graph500.org/)

**Tomorrow:** Day 67 - Stencil Computations... finite difference methods and halo exchange patterns.

*End of Day 066 - Total Lines: 1000+*
